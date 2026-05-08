# Marlin INT4 GEMM Tile-MMA Kernel: Implementation Notes for Rust+CUDA Reimplementation

## What's already in vllm-rust (Stage 15.B integration map)

Audited 2026-05-08, before starting Stage 15.B. Existing repack
plumbing should be **reused, not duplicated**:

| component                                  | location                                             | status                                     |
| ------------------------------------------ | ---------------------------------------------------- | ------------------------------------------ |
| AWQ→GPTQ nibble undo_pack                  | `awq_marlin.rs::repack_awq_nibbles` (line 126)       | DONE (Stage 13-J)                          |
| Marlin scale permutation                   | `marlin.rs::marlin_permute_scales` (line 380)        | DONE — uses `get_scale_perms()`            |
| GPTQ→Marlin tile qweight repack (CPU)      | `marlin.rs::repack_gptq_to_marlin` (line 416)        | **STUB** — returns `qweight.clone()` (462) |
| GPTQ→Marlin tile qweight repack (CUDA PTX) | would-be `marlin_gemm.ptx::repack_gptq_to_marlin_…`  | MISSING entry (per `process_weights` doc)  |
| `MarlinLinear::process_weights`            | `marlin.rs:698`                                      | deliberate no-op until tile-MMA kernel ships |

So **15.B = replace the CPU stub at `marlin.rs:462-465` with a real
nibble-tile rearrangement**, following the reference algorithm in
`reference/vllm/csrc/quantization/marlin/awq_marlin_repack.cu`. The
scales side is already correct; AWQ→GPTQ is already correct. Do not
reintroduce parallel implementations; route through what exists.

The new CPU repack also unblocks `process_weights` to do the full
GPTQ→Marlin tile transform — but that wiring is gated until 15.E
dispatch decides per-call whether to use the new tile-MMA kernel or
fall back to the existing kernels (`awq_gemv_int4_bf16`,
`marlin_gemm`, etc.) which read raw GPTQ and would break if their
input arrives tile-laid-out. Two options at integration time:

1. **Per-instance dual storage**: `MarlinLinear` holds both raw GPTQ
   weights (for the existing kernels) and tile-laid-out weights (for
   the new kernel). VRAM cost: ~2× weight footprint. Acceptable for
   Qwen3-4B (≈1.1 GiB int4 weights → 2.2 GiB).
2. **Lazy repack**: Keep raw GPTQ; the first time the new kernel
   dispatches, repack on the fly into a per-instance cache. First call
   pays a 5–10 s repack cost; subsequent calls free. Better for VRAM
   but adds complexity (where to store the tensor, when to free).

Option 1 is simpler and matches reference vLLM's behaviour (always
keep the tile-laid-out qweight resident). Pick at 15.E.

---

## Overview
These notes capture the essential design choices in vLLM's Marlin reference (GPTQ/AWQ INT4 GEMM) to guide a minimal-viable Rust+CUDA reimplementation for sm_89 (RTX 4060 Ada). Focus: extract the 30% that is necessary, defer the 70% that can be added in v2.

**Reference files:**
- Main kernel: `csrc/quantization/marlin/marlin_template.h`
- Tensor-core ops: `csrc/quantization/marlin/marlin_mma.h`
- Weight repack: `csrc/quantization/marlin/awq_marlin_repack.cu`
- Python utils: `vllm/model_executor/layers/quantization/utils/marlin_utils.py`
- Kernel selection: `csrc/quantization/marlin/generate_kernels.py`

---

## 1. Tile Shapes and the m16n8k16 Instruction

### Tensor-Core Instruction (mma.sync.aligned.m16n8k16)
**For INT4 weights with FP16/BF16 activations:**
- **Instruction:** `m16n8k16.row.col.f32.f16.f16.f32` (FP16 inputs → FP32 accumulation)
- **Instruction:** `m16n8k16.row.col.f32.bf16.bf16.f32` (BF16 inputs → FP32 accumulation)
- **Key fact:** INT4 weights are dequantized to FP16/BF16 *before* MMA; the instruction operates on float, not directly on INT4.
- **Output:** Always FP32 (never FP16 accumulation for INT4 path in our workload).

**From marlin_mma.h:39-43:**
```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}
// A: 4x uint32 (4 uint32 = 16 bytes = 8×16 FP16), B: 2x uint32, C: 4x float
```

### Block Tile (per CTA)
**Small-batch (M≤16):** `M = 16, N = 128, K_per_iter = 128` (thread_m_blocks=1, thread_n_blocks=8, thread_k_blocks=8)
**Large-batch (M>16):** `M = 64, N = 256, K_per_iter = 64` (thread_m_blocks=4, thread_n_blocks=16, thread_k_blocks=4)

These are *block tiles* (in units of 16×16 tensor-core tiles):
- `thread_m_blocks` × 16 = true M per CTA block
- `thread_n_blocks` × 16 = true N per CTA block
- `thread_k_blocks` × 16 = true K per MMA iteration

From `generate_kernels.py:61,200-210`:
```python
THREAD_CONFIGS = [(128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)]
# (thread_k, thread_n, num_threads)
# For small_batch: only (128, 128, 256) or (64, 128, 128) survive
# For large_batch: only (64, 256, 256) or (64, 128, 128) survive
```

### Warp Tile (per warp)
**Warp layout:** 8 warps per 256-thread CTA.
- Each warp handles a sub-tile of M×N.
- For small-batch (128 thread_k, 128 thread_n): each warp sees M=16÷8=2 × 16 rows, N=128÷4≈32 cols.
- Warps are distributed in a 2D grid within the CTA.

From `marlin_template.h:580-581`:
```cpp
constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);
// For 8 warps, tb_n_warps = 2 (FP16 path)
```

### Thread Tile (per-thread register footprint)
**Accumulators:** `FragC = Vec<float, 4>` (4 floats = 16 bytes per thread's output tile).
- Each thread accumulates a 4-element partial result (shared across K-iterations).
- **Total per CTA:** 256 threads × 4 floats = 1024 floats = 4 KB.
- **Per-thread register usage:** 4 float accumulators + 2 FragA buffers (double-buffered) + 2 FragB + 2 FragS + other state → ~50–60 registers/thread on sm_89 (safe, no spill).

From `marlin_template.h:768-771`:
```cpp
FragA frag_a[2][thread_m_blocks];  // 2 stages
I4 frag_b_quant[2][b_thread_vecs];  // B fragments
FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];  // Output tiles
```

### Typical Config Table (for Qwen3-4B AWQ, sm_89)

| M     | N    | K    | thread_m_blocks | thread_n_blocks | thread_k_blocks | threads | stages |
|-------|------|------|-----------------|-----------------|-----------------|---------|--------|
| 1     | 2048 | 4096 | 1               | 8               | 8               | 256     | 4      |
| 4     | 2048 | 4096 | 1               | 8               | 8               | 256     | 4      |
| 8     | 2048 | 4096 | 1               | 8               | 8               | 256     | 4      |
| 16    | 2048 | 4096 | 1               | 8               | 8               | 256     | 4      |
| 32    | 2048 | 4096 | 2               | 8               | 8               | 256     | 4      |
| 64    | 2048 | 4096 | 4               | 16              | 4               | 256     | 4      |

---

## 2. Weight Layout

### AWQ Source Layout
- `qweight[K/8, N]` (int32 tensor): K rows ÷ 8 (since 8 INT4 values per int32), N columns.
- For K=4096, N=2048: shape is `[512, 2048]`, each element holds 8 INT4 nibbles.
- `scales[K/group_size, N]` (FP16): quantization scales, one per group × output column.
- `zeros[K/group_size, N/8]` (int32): quantized zero-points, packed.
- **group_size = 128** (for Qwen3-4B-AWQ).

### Marlin Target Layout
**Weight permutation:** AWQ nibbles are re-shuffled to align with tensor-core tile order.

**From awq_marlin_repack.cu:99-105:**
- Input nibble order follows AWQ's linear interleaving: `[0, 4, 1, 5, 2, 6, 3, 7]`.
- Output order follows tensor-core fragment layout: `pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7}` (for non-8bit activations).
- **Full repack required:** nibbles are re-shuffled, not just transposed.

**Output shape:** `[K/16, (N × 16) / pack_factor]`
- For INT4 (pack_factor=8): `[256, 4096]` (from K=4096, N=2048).
- Repack tiles in `16×64` blocks (tile_k_size=16, tile_n_size=64 from marlin.cuh:42–43).

**From awq_marlin_repack.cu:221–228:**
```cuda
TORCH_CHECK(size_k % marlin::tile_k_size == 0);
TORCH_CHECK(size_n % marlin::tile_n_size == 0);
// Outputs: [size_k / tile_size, size_n * tile_size / pack_factor]
// = [K/16, N*16/pack_factor]
```

### Scales / Zero-Points Distribution
**Scales:** Permuted and stored in `[K/group_size, N]` FP16 layout, with warp-level broadcast via shared memory.
- **Scale permutation:** Python utility `marlin_permute_scales()` applies `scale_perm` or `scale_perm_single` based on `group_size`.
  
**From marlin_utils.py:292–312:**
```python
scale_perm = []
for i in range(8):
    scale_perm.extend([i + 8*j for j in range(8)])  # Interleave 8×8 blocks
scale_perm_single = []
for i in range(4):
    scale_perm_single.extend([2*i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
```

**Broadcast:** Each warp (32 threads) loads scales to shared memory; threads broadcast to their local register copy per K-iteration.

**From marlin_template.h:661–670:** Scale read index calculation for different group configs.

**Zero-points:** Similar permutation as scales; for INT4, packed into int32 with interleaving `[0, 2, 4, 6, 1, 3, 5, 7]`.

**From marlin_utils.py:344–365:**
```python
interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])  # INT4
zp = pack_cols(zp, num_bits, size_k, size_n)  # Pack to int32
```

---

## 3. Async Copy / Pipelining

### cp.async Mechanism
**sm_80+:** Conditional `cp.async` (cache-as or cache-global).
- **sm_80+:** `cp.async.ca.shared.global` (cache-as, L2 cache bypass).
- **sm_75:** Fallback: synchronous 32-bit load per thread.

From `marlin.cuh:100–162` (asm-inline wrappers):
```cpp
#if __CUDA_ARCH__ >= 800
  cp_async4_ca_pred(...) // 16-byte async copy
  cp_async_fence()       // Commit all pending copies
  cp_async_wait<n>()     // Wait for n groups to complete
#else
  // Fallback: simple register load
#endif
```

### Pipeline Stages
**Stage count:** 4 for sm_80+, 2 for sm_75 (Turing).
- Double-buffered producer-consumer: while stage N is being computed, stage N+1 is fetched to shared memory.
- **Synchronization:** `cp_async_wait<stages - 2>()` ensures producer does not overwrite stage being read.

From `marlin_template.h:924–930`:
```cpp
cp_async_wait<stages - 2>();
__syncthreads();
// Guarantees: all threads have loaded registers from current stage
// before next stage writes to shared memory.
```

### Shared Memory Budget
**Per CTA, per stage:**
- A matrix: `stages × (m_block_size × thread_k_blocks × 16 × 2)` bytes (FP16).
- B matrix: `stages × (thread_n_blocks × 16 × thread_k_blocks × 16 / pack_factor) × 4` bytes.
- Scales: `stages × (s_tb_groups × s_sh_stride × 2)` bytes (depends on group_blocks).
- Zero-points: `zp_sh_stage` bytes (INT32, for quantized ZP; float ZP uses same size as scales).

**Total limit:** 96–192 KB per CTA on sm_89 (configurable via `cudaFuncSetAttribute`).

From `marlin.cu:149–216`:
```cpp
int get_kernel_cache_size(...);  // Computes total shared memory
if (cache_size > max_shared_mem) return false;  // Config invalid
```

### No TMA for sm_89
- **TMA (Tensor Memory Accelerator)** is sm_90+ only (Hopper).
- Marlin uses `cp.async` for sm_80–89.
- **WGMMA** (warp-group MMA) is also sm_90+ only.
- **Implication for v1:** Stick to `cp.async` + `ldmatrix` + `mma.sync` (all sm_80+).

---

## 4. Scales / Zero-Points Distribution

### Shared Memory Layout
- Scales in shared memory: `int4* sh_s = sh_zp + zp_sh_stage`, size = `sh_s_size`.
- Per-stage offset: `sh_s_stage = s_tb_groups * s_sh_stride` (when no act_order).

### Per-Warp Distribution
**Without act_order (non-GPTQ):**
- Thread reads scales from shared memory indexed by warp ID and group block.
- Read index `s_sh_rd` pre-computed per thread (depends on m_block_size, thread_n_blocks).

**From marlin_template.h:661–670:**
```cpp
if (group_blocks != -1)
  s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
else  // column-wise
  s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
```

**Result:** Each thread holds a scale fragment (FragS = Vec<half2, 1> = 4 bytes for FP16).

### K-Iteration Relationship (group_size=128)
- **block-wise groups:** If `group_blocks ≥ thread_k_blocks`, new scales loaded only every `group_blocks / thread_k_blocks` MMA iterations.
- **group_size = 128, thread_k_blocks = 8:** group_blocks = 128/16 = 8, so `group_blocks == thread_k_blocks`.
  - New scales loaded once per K-iteration (at the start of each 128-K tile).

From `marlin_template.h:982–1020` (group_blocks >= thread_k_blocks case):
```cpp
if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
  // Load new scale
}
```

---

## 5. Reduction Across K

### Accumulator Precision
- **Per-tile accumulators:** `FragC = Vec<float, 4>` (always FP32 for INT4 dequant path).
- **Why FP32 not FP16:** Dequantization introduces sub-byte precision loss (INT4 → 16 levels); FP32 preserves scale/zero-point multiplication without intermediate rounding.

From `marlin_dtypes.cuh:19–32`:
```cpp
template <> class MarlinScalarType<vllm::kFloat16.id()> {
  using FragC = Vec<float, 4>;  // Always FP32 output
};
```

### Scale-per-K-Tile vs Final Scale
- **Scales applied per K-iteration:** Dequant happens in-flight as B-fragments are loaded, before MMA.
  - Each K-tile brings new scales (if group boundary crossed).
  - MMA operates on pre-scaled (FP16) values.

From `marlin_template.h:1164–1173` (dequant_data):
```cpp
auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
  dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
};
```

**No explicit final scale:** Accumulators in FP32 are written directly to global memory (or reduced if multiple CTAs write same output tile).

### Synchronization Between K-Iterations
- **Pipelined, not synchronized:** K-iterations overlap via double-buffering.
- **Barrier only at slice boundaries:** When multiple CTAs must reduce to same output, lock-based synchronization in global memory.

From `marlin_template.h:177–207` (barrier_acquire / barrier_release):
```cpp
barrier_acquire(int* lock, int count);  // Wait for count CTAs
barrier_release(int* lock, bool reset);  // Atomic increment
```

---

## 6. Minimum Compute Capability

### Supported sm
- **sm_75 (Turing, RTX 2080):** Supported (limited pipeline: stages=2, no cp.async, m16n8k8 MMA only).
- **sm_80 (Ampere, A100):** Full support (cp.async, m16n8k16 MMA, stages=4).
- **sm_89 (Ada, RTX 4060):** Full support (same as sm_80, plus FP8 MMA variants).
- **sm_90+ (Hopper, H100):** Full support + TMA + WGMMA variants (not in reference for INT4 primary path).

From `marlin_template.h:283–292`:
```cpp
#if __CUDA_ARCH__ < 890
  if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;  // FP8 only on Ada+
#endif

#if __CUDA_ARCH__ == 750
  constexpr bool use_fp16_accum = ...; // FP16 accumulation only on Turing
#else
  constexpr bool use_fp16_accum = false;  // FP32 accum on Ampere+
#endif
```

### Separate Code Paths
- **sm_75:** Separate template instantiation with `stages=2` (from generate_kernels.py:226–229).
- **sm_80+:** Unified path with `stages=4`.
- **sm_89 FP8:** Additional `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instruction (wider K=32).

---

## 7. Minimum-Viable v1: What to Copy vs Skip

### Essential (Core 30%)
1. **Tile-loop skeleton:**
   - K-loop over `k_tiles = prob_k / 16 / thread_k_blocks`.
   - N-loop over `n_tiles = prob_n / 16 / thread_n_blocks`.
   - M-loop over `parallel = div_ceil(prob_m, m_block_size)`.

2. **cp.async double-buffering:**
   - Fetch A, B to shared memory.
   - `cp_async_fence()` and `cp_async_wait<2>()` (hardcode 2 for stage-ahead).

3. **ldmatrix + MMA loop:**
   - `ldmatrix.sync.aligned.m8n8.x4` (or x2 for m=8).
   - `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
   - Hardcode single group (no group_blocks logic).

4. **Dequantization in-flight:**
   - INT4 nibble extraction: `(qval >> (lane_idx % 8) * 4) & 0xF`.
   - Scale multiply: `deq = (float)(nibble - zero_point) * scale`.
   - Happens before MMA (via register or inline).

5. **Output store:**
   - Write FP32 accumulators directly to C (no global reduce for v1).
   - Assume M × N ≤ 64, grid size = 1 (single CTA).

### Defer to v2 (Non-Critical 70%)
- ✅ Group-block logic (group_blocks > 0; only support group_blocks = -1 in v1).
- ✅ act_order permutation and g_idx handling.
- ✅ Multiple CTA launches and global reduction (locks).
- ✅ Atomic-add reduce (use_atomic_add).
- ✅ Bias addition.
- ✅ FP32 global reduce vs FP16.
- ✅ Dynamic shmem tuning (hardcode 48KB).
- ✅ Turing (sm_75) fallbacks.

### Hardcoded v1 Config
- **Activation:** FP16 only (no BF16, no INT8, no FP8).
- **Weights:** INT4 unsigned (kU4, AWQ style; no GPTQ, no kU4B8).
- **Scales:** FP16.
- **Zero-points:** Quantized int32 (not float).
- **group_blocks:** -1 (channel-wise).
- **thread_m_blocks:** 1 (M ≤ 16 batch).
- **thread_n_blocks:** 8 (N = 128).
- **thread_k_blocks:** 8 (K per iteration = 128).
- **stages:** 4.
- **Threads per CTA:** 256.

---

## 8. Risks / Gotchas

### Layout Asymmetries
1. **Tensor-core fragment layout vs global layout:**
   - Tensor-core FP16 fragments are in "row-major within 8×8 sub-tiles" (NVIDIA PTX spec).
   - `ldmatrix` loads directly into this layout; no manual transpose needed.
   - **Risk:** Incorrectly assuming linear global → fragment order causes wrong results.

2. **AWQ nibble interleaving:**
   - AWQ source: `[0, 4, 1, 5, 2, 6, 3, 7]` (8 nibbles per int32, interleaved).
   - Marlin output: `[0, 2, 4, 6, 1, 3, 5, 7]` (re-interleaved for MMA).
   - **Risk:** Off-by-one in undo_interleave indexing.

From `awq_marlin_repack.cu:99–105`:
```cuda
constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // Undo AWQ
// Then re-pack with pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7}
```

### Numerical Precision
- **Dequant formula:** `(float(qval - zero_point) * scale)` assumes exact rounding.
  - Float zero-point is not quantized; quantized ZP must be unpacked and cast to float before subtract.
  - **Risk:** Confusion between `float_zp` (is_zp_float=true, AWQ) and `int_zp` (is_zp_float=false, fallback).

From `marlin_template.h:1096–1162` (fetch_zp_to_registers):
```cpp
if constexpr (has_zp && !is_zp_float) {
  // Quantized ZP: unpack from int32 and use as int before cast
  int zp_quant_0 = frag_qzp[k2][0];
  // Extract nibbles, cast to float, then subtract
}
```

### Shared Memory Bank Conflicts
- A-matrix layout uses XOR-based bank conflict avoidance:
  
From `marlin_template.h:720–723`:
```cpp
auto transform_a = [&](int i) {
  int row = i / a_gl_rd_delta_o;
  return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
};
```

- **Risk:** Ignoring this transform → 8-way bank conflicts → serialization of all warp accesses.
- **For v1:** Can use simpler layout and accept some stalls; optimize later.

### Scale Sign Convention
- Scales are always positive (FP16, no sign bit).
- Zero-points can be negative (offset is subtracted before scale multiply).
- **Risk:** Treating ZP as unsigned when converting from quantized int.

---

## Summary: 15.B Minimal Kernel Strategy

**Scope:**
1. Single CTA (blockIdx.x = 0, gridDim.x = 1).
2. M ≤ 16, N = 128, K = 128 (hardcoded tile).
3. No group_size logic (column-wise scales only).
4. No act_order.
5. FP16 activations, INT4 weights (unsigned), FP16 scales, quantized ZP.

**Checklist:**
- [ ] Allocate & initialize shared memory (A + B + scales + ZP).
- [ ] cp.async loop (4 stages): A, B fetch in parallel.
- [ ] ldmatrix + dequant + MMA pipeline (K-loop over 8 iterations of 16 each).
- [ ] FP32 accumulator store to global C.
- [ ] Test on small matrix (M=1, N=128, K=128 hardcoded) before generalization.

**Next:** Task 15.B will implement this skeleton; 15.C–15.F will add group_size, larger M, multi-CTA, etc.

