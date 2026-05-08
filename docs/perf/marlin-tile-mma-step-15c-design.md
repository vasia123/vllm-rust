# Stage 15.C — Minimum-Viable Marlin Tile-MMA Kernel: Design

> Detailed design captured 2026-05-08 to enable Stage 15.C kernel body
> implementation in the next iteration. All inline-PTX signatures and
> fragment layouts are extracted directly from the vLLM reference
> (`reference/vllm/csrc/quantization/marlin/`) at the line numbers cited.
>
> Scope: single-tile, M=16 N=16 K=16, BF16 activations, INT4 AWQ
> weights, **per-channel scales** (no group_size loop), single CTA,
> single warp variant first then optionally a 2-warp variant for the
> second n=8 half. Goal: byte-for-byte numeric correctness vs CPU
> dequant+matmul reference, then build out from there in 15.D.

## 1. Kernel signature

File: `crates/core/kernels/marlin_tile_mma.cu` (new). Build entry in
`crates/core/build.rs`. PTX consumed via `crates/core/src/quantization/
marlin_tile_cuda.rs` (new).

```c++
extern "C" __global__ void marlin_tile_mma_int4_bf16_m16n16k16(
    const __nv_bfloat16* __restrict__ a_ptr,    // [M=16, K=16] row-major
    const uint32_t*     __restrict__ b_tile,    // [1, 32] u32  (Marlin-tile-laid-out, see 15.B)
    const __nv_bfloat16* __restrict__ s_ptr,    // [N=16] per-channel scale
    __nv_bfloat16*       __restrict__ c_ptr,    // [M=16, N=16] row-major output (BF16)
    int /* m */, int /* n */, int /* k */);     // dispatch-time params, ignored in v1
```

Hard-coded shapes:
- M = 16, N = 16, K = 16, group_size = K (one scale per output channel).
- INT4 packed weights: 32 u32 = 256 nibbles = K × N = 256. ✓ (matches a single `[K/16, N×16/8]` tile from 15.B with K=16, N=16.)
- B tile is the output of `awq_to_marlin_tile_repack_cpu` (commit `acc1dd5`)
  with K=16, N=64, **truncated** to N=16 — i.e. the first quarter of
  the 128-u32 tile (32 u32 = 16 nibbles wide × 16 rows / 8). For a
  proper N=16-wide tile we re-emit the first quarter of the existing
  128-u32 tile output.

Block dims: `blockDim = (32, 1, 1)` (1 warp). Grid: `(1, 1, 1)`.
Shared memory budget: A-tile (16 × 16 × 2 = 512 B) + B-tile (32 × 4 =
128 B) + scales (16 × 2 = 32 B) ≈ 700 B. Trivially fits sm_89.

## 2. Fragment layouts (from `marlin_dtypes.cuh:30–68`)

For BF16 inputs, FP32 accumulator on sm_89:

```c++
using FragA = Vec<nv_bfloat162, 4>;  // 4 × bf16x2 = 8 bf16 = 16 B / thread
using FragB = Vec<nv_bfloat162, 2>;  // 2 × bf16x2 = 4 bf16 =  8 B / thread
using FragC = Vec<float,        4>;  // 4 × f32                = 16 B / thread
using FragS = Vec<nv_bfloat162, 1>;  // 1 × bf16x2 =  4 B (per-channel scale, 2 lanes)
```

Per-thread storage = 44 B = 11 registers. 1 warp × 32 threads × 16 B
output = 8 KB output region, but only 16 × 16 × 2 = 512 B written.

## 3. The MMA instruction (from `marlin_mma.h:68–75`)

For BF16 inputs, sm_80+ (we are sm_89):

```c++
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
```

This produces a 16×8 output. For 16×16 output we issue **two** MMAs
side-by-side (one for n=0..7, one for n=8..15), reusing the same A
fragment.

### Output thread → matrix mapping

For `m16n8` with FP32 accumulator, lane `t ∈ [0, 32)` writes:
- C[t/4 + 0, (t%4)*2 + 0] ← c[0]
- C[t/4 + 0, (t%4)*2 + 1] ← c[1]
- C[t/4 + 8, (t%4)*2 + 0] ← c[2]
- C[t/4 + 8, (t%4)*2 + 1] ← c[3]

So 4 floats per thread × 32 threads = 128 floats = 16 × 8 matrix. ✓

## 4. Loading A into FragA (BF16 ldmatrix)

A is row-major BF16 in shared memory, shape `[M=16, K=16]`. Use
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` (loads 4 8×8 BF16 sub-tiles
in one instruction; pattern from `marlin_template.h`):

```c++
uint32_t smem = __cvta_generic_to_shared(&sh_a[lane_id_row * 16 + lane_id_col * 8]);
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
    : "r"(smem));
```

The thread-to-shmem-element mapping inside `ldmatrix.x4` is the
canonical `m8n8.x4` pattern: each lane addresses one 8×8 sub-tile's
first row by lane index → ldmatrix permutes the loads so that after
the instruction, FragA's 8 BF16 elements per thread are arranged in
the layout the subsequent mma.m16n8k16 expects.

## 5. Loading B into FragB (INT4 dequant via LOP3 + bf16 SUB)

This is where 15.B's tile layout becomes load-bearing. After the
Stage 15.B repack, the B tile holds 32 u32 in `[k_tile_id, n_tile_id]
* 128 + th_id*4 + warp_id` order (we have 1 tile, 1 warp, 8 threads
participate per quarter — see below).

For each MMA call, FragB needs 4 BF16 values per thread (= 1 u32 of
INT4 nibbles). The reference packs 8 nibbles per u32 in
`pack_idx[0,2,4,6,1,3,5,7]` order so that the dequant LOP3 pass
yields two `bf16x2` halves, each containing the right nibbles for
the upper and lower halves of a fragment.

Dequant primitive (from `dequant.h:174–215`):

```c++
__device__ inline void dequant_int4_bf16(uint32_t q, nv_bfloat162* frag_b /* size 2 */) {
    // Step 1: LOP3 + bias trick produces bf16(nibble + 128.0)
    static constexpr uint32_t MASK = 0x000f000f;
    static constexpr uint32_t EX   = 0x43004300;  // bf16(128.0) duplicated
    uint32_t lo, hi;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(lo) : "r"(q),       "n"(MASK), "n"(EX));
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(hi) : "r"(q >> 4),  "n"(MASK), "n"(EX));

    // Step 2: subtract bf16(128.0) to recover raw nibble values
    static constexpr uint32_t SUB = 0x43004300;
    nv_bfloat162 lo_bf16 = *reinterpret_cast<nv_bfloat162*>(&lo);
    nv_bfloat162 hi_bf16 = *reinterpret_cast<nv_bfloat162*>(&hi);
    nv_bfloat162 sub_bf16 = *reinterpret_cast<const nv_bfloat162*>(&SUB);
    frag_b[0] = __hsub2(lo_bf16, sub_bf16);
    frag_b[1] = __hsub2(hi_bf16, sub_bf16);
}
```

The LOP3 immediate `0xea = (0xF0 & 0xCC) | 0xAA` evaluates as
`(q AND MASK) OR EX`, isolating the low 4 bits of each 16-bit half
and stamping the bf16 exponent `0x4300` (which is 2^7 = 128.0). After
SUB, each lane of `lo`/`hi` holds the integer nibble as bf16.

Output: 4 BF16 values per thread × 32 threads = 128 BF16 = K(16) ×
n=8 BF16 weights. Two MMAs need two FragB calls — second one uses a
different u32 slice of the tile.

## 6. Scale application

Per-channel scale for v1 (group_size = K = 16). Scales tensor `s_ptr
[N=16]`. Each thread holds 2 scale values (for n=col and n=col+1)
loaded once.

```c++
nv_bfloat162 frag_s = *reinterpret_cast<const nv_bfloat162*>(&s_ptr[(t%4)*2]);
frag_b[0] = __hmul2(frag_b[0], frag_s);
frag_b[1] = __hmul2(frag_b[1], frag_s);
```

(Repeat with `frag_s` advanced to next 2 channels for the second MMA.)

## 7. Output store: FragC FP32 → BF16

After the two MMAs, each thread holds 8 floats (4 from MMA-low, 4
from MMA-high, covering N=0..15). Convert to BF16 and write:

```c++
__nv_bfloat162 lo = __float22bfloat162_rn(make_float2(c_lo[0], c_lo[1]));
__nv_bfloat162 hi = __float22bfloat162_rn(make_float2(c_lo[2], c_lo[3]));
// store to c_ptr at canonical row/col positions per the mapping in §3
```

Output address: `c_ptr[(t/4 + 0) * N + (t%4)*2]` and 3 more strided
stores per thread.

## 8. CPU reference for correctness check

In Rust test (gated `gpu-test-small`):

```rust
let (m, n, k) = (16, 16, 16);
// 1. Random A: [M, K] bf16
// 2. Random AWQ qweight: [K, N/8] u32 (1 row × 2 u32)
// 3. Random scales: [N] bf16
// 4. AWQ→Marlin tile repack via awq_to_marlin_tile_repack_cpu(..., 16, 64) — take first 32 u32
// 5. Dispatch the new kernel → BF16 output
// 6. CPU reference: dequant nibbles via undo_pack → multiply by scales → bf16 matmul vs A
// 7. assert close (1e-2 abs tolerance for bf16 reduction over K=16)
```

## 9. Build wiring

Add to `crates/core/build.rs::KERNELS`:

```rust
KernelDef {
    source: "kernels/marlin_tile_mma.cu",
    output: "kernels/marlin_tile_mma.ptx",
    min_sm: 80,  // m16n8k16 BF16 needs sm_80; we run sm_89
},
```

Add to `crates/core/src/quantization/marlin_cuda.rs` (or new module
`marlin_tile_cuda.rs`):

```rust
const MARLIN_TILE_MMA_PTX: &str = include_str!("../../kernels/marlin_tile_mma.ptx");
```

## 10. Risks for the implementation step

1. **ldmatrix shmem address**: needs `__cvta_generic_to_shared` (we
   may have used `cvta.to.shared.u64` syntax in other kernels — match
   the project's convention by greping existing kernels).
2. **BF16 output store pattern**: m16n8 thread mapping in §3 must be
   followed exactly or output shape is silently wrong (32 threads
   fight over the same slot). Test with non-symmetric input matrices
   to surface any transposition.
3. **B tile layout vs MMA expectation**: 15.B repack lays out 8
   nibbles per u32 in pack_idx [0,2,4,6,1,3,5,7] order. The dequant
   LOP3 sequence yields `frag_b[0] = (nib0, nib1)` and `frag_b[1] =
   (nib2, nib3)` where `nib_i` are the i-th nibbles of `q` (post-
   shift). This means the 8 nibbles in `q` are consumed in raw order
   0..7, but the *spatial* mapping (which 8 weight values they are)
   depends on the warp/thread layout in the reference kernel. The
   reference's warp-id and tc_col/tc_row mapping at
   `awq_marlin_repack.cu:75–101` defines the producer side; the
   consumer (this kernel) must mirror it. Independent verification:
   for our v1 hard-coded single-tile case, write a per-thread offset
   table from the Marlin-tile bytes to the (K, N) coordinates and
   sanity-check it against the dequant output for a deterministic
   input.
4. **Per-channel vs per-group scales**: v1 uses one scale per
   output channel. AWQ default in production is group_size=128
   along K; our K=16 v1 case lets group_size = K, sidestepping the
   group-loop. Group_size handling lands in 15.D.
5. **Single-warp single-tile is M_TILES=1 ≠ 4 in reference**: the
   reference uses 4 warps × 16 cols of tile each. Single-warp v1
   produces only N=16 (not 64); 4-warp expansion lands in 15.D.

## 11. Files touched in 15.C

- `crates/core/kernels/marlin_tile_mma.cu` — **new**, ~150 lines
  (kernel + dequant helper).
- `crates/core/build.rs` — add KernelDef entry.
- `crates/core/src/quantization/marlin_tile_cuda.rs` — **new**, ~80
  lines (cudarc CustomOp wrapper + 1 GPU test).

## 12. Definition of done for 15.C

- `cargo build --features cuda-default` succeeds; `marlin_tile_mma.ptx`
  is generated.
- 1 GPU test under `gpu-test-small` passes: random `[16,16] BF16 @
  [16,16] INT4 (per-channel scaled)` matches CPU dequant+matmul to 1e-2.
- No production dispatch yet — kernel is dormant until 15.D extends
  to multi-tile and 15.E wires it into `MarlinLinear::forward`.
- Microbench under `awq_marlin_path_bench.rs` measures ms for the
  16×16×16 path (small but proves the pipeline is alive).

## 13. Out of scope for 15.C (explicitly deferred)

- group_size < K (defer to 15.D).
- M > 16, N > 16, K > 16 (defer to 15.D).
- Multiple warps per CTA (defer to 15.D).
- cp.async double buffering (defer to 15.D).
- Multi-CTA grid + global reduce (defer to 15.D).
- Scale permutation via `marlin_permute_scales` (per-channel
  doesn't permute; defer permute integration to 15.D).
- AWQ zero-point handling in the dequant path (the reference uses
  `kU4` no-bias dequant; AWQ symmetric-quantized weights with their
  separate `qzeros` tensor are applied AFTER the scale multiplication
  — handled in 15.D when group_size kicks in).

---

**Next iteration starts at**: writing `marlin_tile_mma.cu` against
this design. Reading reference is unnecessary — every load-bearing
fact is captured here with its source line number.
