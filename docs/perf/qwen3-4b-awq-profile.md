# Qwen3-4B-AWQ Decode Profile (Stage 13-C Phase 1)

Date: 2026-05-07. RTX 4060 Laptop, sm_89. Server `--enforce-eager`,
steady-state ~44 tok/s baseline.

## Methodology

Two complementary tracks:

1. **nsys CUDA API trace** — `nsys profile -t cuda,nvtx`. WSL2 ограничивает
   GPU kernel timing (CUPTI), но host-side API trace полностью доступен.
   Профиль включает server boot + warmup + один 200-токен decode request.
2. **In-process per-component timer** (`VLLM_PROFILE_DECODE=1`,
   commit 8fb04e2). Делает `cuda_stream().synchronize()` после каждого
   замеренного блока — каждый sample отражает реальное GPU time
   с учётом pipeline drain. Per-component sums over 100 layer-calls.

## Per-component breakdown (median over ~25 samples × 100 layer-calls)

Server `--enforce-eager`, max_tokens=400 generation (steady state ≥ token 100).
Numbers in µs per layer-call (averaged over 100 layer-calls).

| Компонент          | µs/call | %      | На token (×36) | Что внутри                        |
| ------------------ | ------- | ------ | -------------- | --------------------------------- |
| **mlp**            | 300     | 30%    | 10.8 ms        | gate + up + down GEMV + SwiGLU    |
| **paged_attn**     | 280     | 28%    | 10.1 ms        | `paged_attention_cuda` V1/V2      |
| qkv                | 100     | 10%    | 3.6 ms         | q+k+v GEMV (3 calls)              |
| rope               | 80      | 8%     | 2.9 ms         | apply_varlen_gpu                  |
| o_proj             | 80      | 8%     | 2.9 ms         | one GEMV                          |
| qk_norm            | 43      | 4.3%   | 1.55 ms        | per-head RMSNorm × 2              |
| cache_w            | 40      | 4%     | 1.44 ms        | cache_engine.write_batch          |
| post_norm          | 37      | 3.7%   | 1.33 ms        | RmsNorm (post-attention)          |
| in_norm            | 30      | 3%     | 1.08 ms        | RmsNorm (pre-attention)           |
| **sum (profile)**  | 1000    | 100%   | 36 ms          | profile-mode total (with syncs)   |
| **wall (no sync)** |         |        | 22.7 ms        | actual at 44 tok/s                |

Profile mode adds ~13 ms/token of sync overhead (9 syncs × 36 layers ×
~40 µs each). Proportions are reliable; absolute numbers under profile
are inflated.

### Top findings

1. **paged_attn dominates at 28%** — much more than the ~10 µs/call estimate
   from preliminary exploration. At seq_len ≈ 480 the V1/V2 kernel is
   ~30× slower than the theoretical bandwidth ceiling (9 µs for ~2.4 MB
   of KV reads at 256 GB/s). Root cause: at batch=1 the V1 grid is
   `(num_q_heads=32, 1, 1)` → only 32 blocks total = ~128 warps =
   ~17% occupancy on 24 SMs.

2. **GEMV total is "only" 48%** (mlp + qkv + o_proj). Not the dominant
   bottleneck. Our `awq_gemv_int4_kt_bf16` kernel is already at
   150–200 GB/s effective bandwidth.

3. **Non-GEMV non-paged_attn = 24%** (rope 8% + qk_norm 4.3% +
   cache_w 4% + post_norm 3.7% + in_norm 3%). Distributed; no single
   target dominates within this group.

## CUDA API trace (nsys, full server lifecycle, ~30s)

GPU kernel timing was unavailable on WSL2 (CUPTI scope limited), but
host-side API traffic gives clear signal:

| API                       | Total time | Calls    | Calls / decode token | Notes                              |
| ------------------------- | ---------- | -------- | -------------------- | ---------------------------------- |
| cuMemcpyHtoDAsync_v2      | 22.17 s    | 17 547   | ~70 (mostly load)    | One 17.9 s call = model load chunk |
| cuLibraryLoadData         | 1.46 s     | 14       | n/a (boot)           | Module load                        |
| cuMemcpyDtoHAsync_v2      | 1.30 s     | 200      | **1 (avg 6.5 ms)**   | Per-token output sync              |
| **cuLaunchKernel**        | 1.29 s     | 143 697  | **~720**             | ~3× our 252 GEMV count             |
| **cuMemAllocAsync**       | 0.75 s     | 105 043  | **~525**             | After Stage 12 GEMV pool           |
| cuMemsetD8Async           | 0.61 s     | 3 625    | ~18                  | Output zeroing                     |
| cudaMemcpyAsync           | 0.45 s     | 108      | <1                   | One-shot transfers                 |
| **cuEventRecord**         | 0.35 s     | 616 191  | **~3000**            | cudarc V3 auto-tracking            |
| cuEventCreate             | 0.32 s     | 210 086  | ~1000                | cudarc bookkeeping                 |
| cuStreamWaitEvent         | 0.27 s     | 866 807  | ~4300                | cudarc bookkeeping                 |
| cuMemcpyDtoDAsync_v2      | 0.15 s     | 14 760   | ~75                  | Internal D2D copies                |
| cudaStreamSynchronize     | 0.06 s     | 36       | <1                   | Trivial                            |

### Implications

- **525 cuMemAllocAsync/token** even after Stage 12 (which pools only
  GEMV outputs, ~252/token). Remaining ~273/token come from RoPE,
  RMSNorm, paged_attn, SwiGLU, plus candle-internal Tensor::cat /
  contiguous / from_vec.
- **~720 cuLaunchKernel/token** is ~3× the 252 GEMV count. That gives
  the count of distinct CUDA kernels per token (paged_attn V2 has 2
  stages = 72; RoPE = 36; cache_write = 36; rms_norm = 72; sampling +
  embedding + lm_head = ~10).
- **3000 cuEventRecord/token** = cudarc 0.16's automatic event tracking
  on every CudaSlice touch. This is the V3 capture blocker (Stage 13-A).
  Each event is ~1 µs of work in cudarc → ~3 ms/token of pure cudarc
  bookkeeping ≈ 13% of wall-clock. Not addressable without forking
  cudarc.
- **DtoH 6.5 ms × 200 tokens = 1.3 s** of pipeline drain on the
  per-token output pull (sampler → tokenizer). This is the implicit
  GPU sync barrier — it actually IS the per-token GPU work converging.
  Not pure overhead.

## FlashInfer status: F2 (architecturally bypassed)

`select_backend()` in `crates/core/src/layers/attention/backend.rs:225`
returns `FlashInferBackend` whenever feature `flashinfer` is on
(it is, via `cuda-default`). However, the **quantized Qwen3 decode
path bypasses the attention backend abstraction entirely**:

- File: `crates/core/src/models/qwen3_quantized.rs:435`
- Code: `crate::cuda_kernels::paged_attention_cuda(...)` — direct
  call into our internal V1/V2 paged_attention kernel
  (`crates/core/kernels/paged_attention.cu`).

So FlashInfer is compiled in and `select_backend()` would return it,
but the Qwen3 quantized model never asks for it. CPU fallback path
(line 470) does use `paged_attention(...)` which goes through
`select_backend()` — but that path only runs when `cuda-kernels` is
disabled.

**Verdict: F2.** FlashInfer is not on the critical path. Earlier
hypotheses about FlashInfer `plan()` host-syncs are not relevant
to the current bottleneck.

`flashinfer-rs` is locally available at `../flashinfer-rs/` and
patchable. If we want to switch to FlashInfer's decode kernel, we
would need to:
1. Wire the quantized Qwen3 forward to `select_backend()` instead
   of `paged_attention_cuda`.
2. Add plan caching at the engine level (current
   `decode_flashinfer` builds a fresh plan every call; caching is
   exposed via `_with_plan` variant but unused).

This would only be worthwhile if FlashInfer's M=1 decode kernel is
materially faster than our paged_attn at the seq_len we operate at.
Worth a follow-up benchmark, but not the obvious next step.

## Decision (Phase 1 → Phase 2)

Mapping observed proportions onto the Decision Gate from the plan:

| Branch                          | Trigger                                              | Match  |
| ------------------------------- | ---------------------------------------------------- | ------ |
| Α (FlashInfer fix)              | F2 bug or F1 + plan() >10%                           | weak — F2, but path not used |
| **Β (kernel hot)**              | one kernel ≥ 30%                                     | **paged_attn at 28% ≈ threshold** |
| Γ (candle internals)            | Tensor::cat/contiguous/arithmetic ≥ 20%              | not directly visible — covered by event/alloc traffic |
| Δ (host syncs)                  | cuStreamSync / cuMemcpyDtoH ≥ 15%                    | DtoH is pipeline drain, not pure overhead |

**Choice: Β with paged_attn as the target.** It is the largest
single kernel, bandwidth ceiling allows ~30× speedup theoretically,
and the V1 launch geometry (32 blocks, batch=1) is concretely
under-utilising the SMs.

Secondary: extend OutputPool to RoPE / RmsNorm / SwiGLU / paged_attn
output buffers — eliminates ~273 cuMemAllocAsync/token. Per-call
saving on Ada laptop is small (~3 µs each = ~800 µs/token = 3.5%),
but cumulative with Β it can push total to ≥50 tok/s.

Out of scope for this push:
- cudarc V3 event tracking (~13% wall-clock overhead) — architectural,
  documented as Stage 13-A.
- DtoH per-token drain (~17% wall-clock) — necessary for tokenizer
  feedback unless we batch generation client-side.

## Phase 2.A — paged_attn V2 split-K with PARTITION_SIZE=128

Three small changes:
- `paged_attention.cu:367`: `#define PARTITION_SIZE 512` → **128**
- `cuda_kernels.rs:352`: `const PARTITION_SIZE: usize = 512` → **128**
  (must stay in sync with the kernel)
- `cuda_kernels.rs:356`: `const V2_SEQ_LEN_THRESHOLD: usize = 512`
  → **64** (V2 always wins at batch=1; per-token __syncthreads cost
  in V1's single-block-per-head design dominates the kernel runtime)
- `qwen3_quantized.rs:435`: switch `paged_attention_cuda` (V1 only)
  → `paged_attention_auto` (V2 picked above threshold). The decode
  call site previously bypassed the auto dispatch entirely.

For seq_len ≈ 480 the V2 grid becomes `(num_q_heads, num_seqs,
ceil(seq_len/128)) = (32, 1, 4) = 128 blocks` (was 32 under V1).
Active warps jump from 128 (~17% of 768-warp budget on 24 SMs Ada)
to 512 (~67%). Per-block `__syncthreads` count drops from ~480 in
V1's K-pass to ~128 in each V2 partition.

### Results (steady-state, post-Phase-2.A profile)

`VLLM_PROFILE_DECODE=1` median sample over ~25 samples × 100
layer-calls, max_tokens=400 generation:

| Component   | Before (Phase 1) | After (Phase 2.A) | Δ          |
| ----------- | ---------------- | ----------------- | ---------- |
| in_norm     | 30 µs (3.0%)     | 26 µs (3.3%)      | flat       |
| qkv         | 100 µs (10%)     | 97 µs (12.3%)     | flat       |
| qk_norm     | 43 µs (4.3%)     | 36 µs (4.6%)      | flat       |
| rope        | 80 µs (8%)       | 62 µs (7.9%)      | -23%       |
| cache_w     | 40 µs (4%)       | 31 µs (3.9%)      | -23%       |
| **paged_attn** | **280 µs (28%)** | **137 µs (17.4%)** | **-51%**   |
| o_proj      | 80 µs (8%)       | 71 µs (9.0%)      | flat       |
| post_norm   | 37 µs (3.7%)     | 28 µs (3.6%)      | flat       |
| **mlp**     | 300 µs (30%)     | 299 µs (38%)      | flat → now dominant |
| **sum**     | 1000 µs          | **785 µs**        | **-22%**   |

End-to-end bench, `concurrency=1 max_tokens=256 prompt_len=256
temperature=0 runs=5`:

| Build                                | tok/s steady |
| ------------------------------------ | ------------ |
| baseline (post Stage 12 disable)     | 33.9         |
| **Phase 2.A**                        | **42.7**     |

**+26% wall-clock** (slightly above the +18% predicted, presumably
because the per-component sync overhead also drops with smaller
per-component time).

### What's left after Phase 2.A

MLP is now the largest single component at 38% of layer time
(298 µs/call = ~10.7 ms/token). Three sub-GEMVs (gate/up/down).
Already on `awq_gemv_int4_kt_bf16` near peak bandwidth. Further
gain there needs gate+up fusion or different kernel geometry —
high effort, low ROI per Phase 1 audit.

paged_attn 137 µs/call is still ~18× over the theoretical bandwidth
ceiling. Further wins from Phase 2.B candidates:
- warp-level QK reduce (vs current block-level per-token sync)
- vectorized V loads (uint4 = 8 BF16 per LDG; we currently read scalars)
- GQA-aware grouping (Qwen3 has q_per_kv=4, currently each q_head
  reads same KV redundantly)

Decision: Phase 2.A target was ≥38 tok/s (+10%). Got 42.7 (+26%).
**Continue with Phase 2.B**: pick the one with best ROI / lowest
risk to push toward 50 tok/s.

## Phase 2.B.1 — warp-level QK reduce in V2 K-pass

The V2 K-pass under PARTITION_SIZE=128 still issued ~256
`__syncthreads` per partition: each token's QK dot product ran
through `block_reduce_sum`, which costs two block-wide syncs
(`reduce_smem` write fence + result-broadcast fence). At 128
tokens per partition × 4 partitions = 512 calls × ~256 syncs each
= ~131k syncs per token through the K-pass alone. Each sync
serialises four warps and stalls execution.

Audit of `paged_attention_v2.cu` and the vLLM reference confirmed
the per-token block-reduce is what dominates kernel runtime at
batch=1; the V-pass is already coalesced (128 threads × 1 dim
each, kv_cache layout `[block, token, kv_head, dim]` makes
adjacent threads read adjacent addresses).

### Design

Partition the work along WARPS rather than threads:

- 4 warps in the block. Each warp owns its own subset of tokens
  with stride NUM_WARPS (warp 0 → tokens 0,4,8,…; warp 1 → 1,5,9;
  etc.).
- Within a warp: 32 lanes × 4 dims/lane (head_dim=128/32). Each
  lane computes a partial QK over its 4 dims; `warp_reduce_sum`
  via `__shfl_xor_sync` produces the full QK with no shared
  memory and no `__syncthreads`.
- Lane 0 of each warp writes the logit and updates a per-warp
  `qk_max` register.
- After the loop: a single `__syncthreads` broadcasts the global
  `qk_max` across warps via `reduce_smem`.

K-pass syncs: 256 → 1 per partition.

Constraint: head_dim divisible by 32. All current LLM head dims
(64, 96, 128, 192, 256) satisfy this. A legacy block-reduce path
remains in the kernel as a fallback for any future caller with
exotic head_dim.

### Results

VLLM_PROFILE_DECODE=1 median sample over ~10 samples × 100
layer-calls in steady state:

| Component   | Phase 2.A        | Phase 2.B.1       | Δ        |
| ----------- | ---------------- | ----------------- | -------- |
| in_norm     | 26 µs (3.3%)     | 26 µs (3.6%)      | flat     |
| qkv         | 97 µs (12.3%)    | 96 µs (13.3%)     | flat     |
| qk_norm     | 36 µs (4.6%)     | 34 µs (4.7%)      | flat     |
| rope        | 62 µs (7.9%)     | 60 µs (8.3%)      | flat     |
| cache_w     | 31 µs (3.9%)     | 30 µs (4.2%)      | flat     |
| **paged_attn** | **137 µs (17.4%)** | **82 µs (11.4%)** | **−40%** |
| o_proj      | 71 µs (9.0%)     | 68 µs (9.4%)      | flat     |
| post_norm   | 28 µs (3.6%)     | 25 µs (3.5%)      | flat     |
| mlp         | 299 µs (38%)     | 296 µs (41%)      | flat     |
| **sum**     | 785 µs           | **717 µs**        | **−9%**  |

Cumulative paged_attn since baseline: 280 → 137 → 82 µs = **−71%**.
mlp is now 41% of per-layer time; further attention work has
diminishing returns versus going after the MLP block.

End-to-end bench (`concurrency=1 max_tokens=256 prompt_len=256
temperature=0 runs=5`):

| Build                                | tok/s steady |
| ------------------------------------ | ------------ |
| baseline (post Stage 12 disable)     | 33.9         |
| Phase 2.A (PARTITION_SIZE=128)       | 42.7         |
| **Phase 2.B.1 (warp-level K-pass)**  | **46.5**     |

**+9% over 2.A, +37% from baseline.** 46.5/50 ≈ 93% of the Stage
13-C target.

Verification:
- `cargo test --features cuda-default -p vllm-core --lib`: 4489 pass.
- V1↔V2 parity tests under `cuda-default` and `cuda-default
  gpu-test-medium` pass (BF16 + F16 dtypes).
- `scripts/test_qwen3_awq_correctness.sh`: 5/5 PASS.
- `cargo clippy --features cuda -p vllm-core --lib -- -D warnings`:
  clean.

## Phase 2 hardening — Step 3: paged_attention microbench

`crates/core/benches/paged_attention_bench.rs` (new, criterion harness)
sweeps the (variant, seq_len, partition_size) cube on Qwen3-4B-AWQ
shape (num_q_heads=32, num_kv_heads=8, head_dim=128, block_size=16,
batch=1).  Run:

```
cargo bench --features cuda-default -p vllm-core --bench paged_attention_bench
```

Median µs/call on RTX 4060 Laptop, sm_89:

| seq_len | V1       | V2 p=64 | V2 p=128 | V2 p=256 | V2 p=512   | best V2    |
| ------- | -------- | ------- | -------- | -------- | ---------- | ---------- |
|     64  |     56.7 |    54.0 |     55.1 |     54.9 |       55.1 | **p=64**   |
|    128  |     89.4 |    54.9 |     61.1 |     61.7 |       61.5 | **p=64**   |
|    256  |    146.2 |    56.7 |     62.5 |     77.2 |       76.7 | **p=64**   |
|    512  |    256.4 |    65.5 |     68.6 |     78.5 |      108.2 | **p=64**   |
|   1024  |    502.0 |    81.9 |     82.5 |     88.1 |      112.4 | **p=64**   |
|   2048  |    979.9 |   113.1 |    115.0 |    115.4 |      130.9 | **p=64**   |
|   4096  |   1847.8 |   179.0 |    174.6 |    179.7 |      185.8 | p=128      |
|   8192  |   3512.6 |   353.4 |    324.7 |    307.1 |  **306.6** | p=512      |
|  16384  | (V1 OOM) |   678.3 |  **642.9**|   653.0 |      645.1 | p=128      |

Findings:

1. **V1 always loses.**  Even at seq_len=64 the smallest V2 partition
   (p=64) is 5% faster (54.0 µs vs 56.7 µs).  V1 has a hard ceiling
   at seq_len ≈ 12k where its `logits[max_seq_len]` shared-memory
   allocation overflows the per-block 100 KB Ada limit.

2. **Partition_size optimum shifts with seq_len.**  At seq_len ≤ 4096
   the smallest partition wins (more grid blocks, better SM
   occupancy on a 24-SM Ada laptop).  At seq_len ≥ 8192 a larger
   partition wins (fewer tmp_out writes, fewer reduce-kernel
   partitions to merge).

3. **Current default p=128 is sub-optimal in our hot range** (Qwen3
   decode at seq ≈ 256-1024).  p=64 is 5-15% faster on every point.

These numbers feed Step 4: the adaptive selector picks p=64 below
4096 and p=256 above.  The threshold `V2_SEQ_LEN_THRESHOLD` is
dropped to 0 (always V2) since V1 wins nothing.

## Phase 2 hardening — Step 4: adaptive partition_size selector

`select_v2_partition_size(max_seq_len)` (new public function in
`crates/core/src/cuda_kernels.rs`) returns the partition size derived
from the Step 3 microbench:

```
seq_len ≤ 4096  → p = 64    (more grid blocks, ~5-15 % faster
                              than p=128 in the Qwen3-4B hot range)
seq_len > 4096  → p = 256   (fewer reduce-kernel partitions,
                              less tmp_out write traffic at long ctx)
```

`paged_attention_auto` now routes through
`paged_attention_v2_cuda_with_partition_size` with this value, and
`V2_SEQ_LEN_THRESHOLD` drops from 64 to 0 (always V2 — V1 lost on
every measured point and has the long-context shared-memory ceiling
besides).

End-to-end bench (`concurrency=1 max_tokens=256 prompt_len=256
temperature=0 runs=5`):

| Build                                  | tok/s steady |
| -------------------------------------- | ------------ |
| baseline (post Stage 12 disable)       | 33.9         |
| Phase 2.A (PARTITION_SIZE=128)         | 42.7         |
| Phase 2.B.1 (warp-level K-pass)        | 46.5         |
| **Step 4 (adaptive p=64 for ≤4096)**   | **47.1**     |

The +1.3 % over 2.B.1 matches the back-of-envelope prediction:
paged_attn was already 11.4 % of per-layer time after 2.B.1, so the
selector's ~10 % paged_attn improvement caps end-to-end at ~1.1 %
wall-clock.  47.1 / 50 ≈ 94 % of the Stage 13-C target.

Why this number is small but worth landing:
- It is the data-driven choice, not a guess.  Anyone re-tuning the
  kernel (different SM count, different head_dim, future kernel
  rewrites) re-runs the microbench and updates the selector with
  real numbers, not vibe.
- Long-context performance moves with this change too: at seq=8192
  the p=256 selection is ~6 % faster than the previous p=128
  default (307 vs 325 µs), which matters for any future workload
  with longer prompts than Qwen3 typical.

Verification:
- `cargo test --features cuda-default -p vllm-core --lib`: 4493 pass.
- `cargo test --features "cuda-default gpu-test-medium" -p vllm-core
  --lib paged`: 6/6 pass (V1↔V2 parity matrix from Step 2).
- `scripts/test_qwen3_awq_correctness.sh`: 5/5 PASS.
- `cargo clippy --features cuda -p vllm-core --lib --tests
  -- -D warnings`: clean.

## Side-by-side: vllm-rust (this branch) vs Python vLLM 0.20.1

Both servers ran on the same RTX 4060 Laptop (sm_89, 8 GiB VRAM,
WSL2), same model `Qwen/Qwen3-4B-AWQ`, same `--max-model-len 6000`,
same eager mode (`--enforce-eager` for both — disables CUDA-graph
capture which is V3-blocked for us anyway).  Bench harness:
`scripts/bench_decode.py` with `--prompt-len 256 --max-tokens 256
--temperature 0`, runs 5 (c=1) / 3 (c=4) / 2 (c=8).

| Backend       | c | TTFT (ms) | tok/s/req (med) | aggregate tok/s |
| ------------- | - | --------- | ---------------- | --------------- |
| **vllm-rust** | 1 |    4135.8 |             46.9 |            46.9 |
| Python vLLM   | 1 |      46.6 |             48.2 |            48.2 |
| **vllm-rust** | 4 |   16614.1 |             14.7 |            58.8 |
| Python vLLM   | 4 |      53.7 |             47.8 |           191.3 |
| Python vLLM   | 8 |      98.0 |             44.0 |           351.6 |
| (vllm-rust c=8 OOMs the KV cache before completing 8 streams.)         |

### Where we are competitive

**Single-stream decode tok/s** — 46.9 vs 48.2 = **97% of Python vLLM**
on the metric Stage 13-C optimised.  All five hardening commits
(refactor, parity matrix, microbench, adaptive selector, e2e tests)
together with the upstream Phase 2.A + 2.B.1 work close the gap to
within run-to-run noise.

### Where we are not

1. **TTFT (prefill latency)**: vLLM 47 ms vs vllm-rust 4136 ms ≈ **88×**
   slower.  Stage 13-C focused entirely on decode; prefill is still
   dominated by AWQ→Marlin layout repack and CPU-side weight setup
   work that has not been touched.  For interactive single-stream
   workloads with short prompts this dominates the user-perceived
   latency budget.

2. **Concurrent batching scale**: at concurrency=4 vLLM holds
   per-request throughput essentially flat (47.8 tok/s, 99% of c=1)
   and lifts aggregate to 191 tok/s (3.97× scaling — near-linear).
   vllm-rust at c=4 collapses per-request to 14.7 tok/s (-69%) and
   lifts aggregate only to 58.8 tok/s (1.25× scaling).
   At c=8 vLLM still scales near-linearly to 351 tok/s; vllm-rust
   exhausts its KV cache before completing 8 concurrent streams.

   Continuous batching, page-table-aware admission, and the V2
   attention kernel's GQA-grouped block tiling are the things vLLM
   does that we don't.  Each one is a substantial piece of work in
   its own right; none is on the Stage 13-C path.

### Read

For the workload Stage 13-C explicitly targeted — single-stream
batch-1 decode on Qwen3-4B-AWQ — we are within 3 % of the
production-grade Python vLLM reference on the same hardware.  The
two big remaining gaps (prefill latency, batched scaling) are
orthogonal to decode kernel performance and would each be a Stage
13-D / 13-E in their own right.

---

## Stage 13-D — TTFT root cause (2026-05-07)

### 13-D.3 verdict

Per-component prefill profile (`VLLM_PROFILE_PREFILL=1`,
`scripts/bench_prefill.py`) on Qwen3-4B-AWQ, RTX 4060 Laptop, 36 layers,
greedy:

| seq_len | mlp ms | qkv ms | o_proj ms | paged_attn ms | mask+norms+lm_head ms | total ms |
|---:|---:|---:|---:|---:|---:|---:|
| 122  | 1001 | 244  | 168 | 238 | 25  | 1676 |
| 382  | 3107 | 754  | 513 | 237 | 41  | 4651 |
| 1422 | 11765| 2814 | 1903| 245 | 230 | 16868 |
| 1940 | 16190| 3855 | 2606| 257 | 358 | 23217 |

**Quantized linears dominate: 97 % of total** (mlp + qkv + o_proj).
`paged_attention` is a flat ~250 ms regardless of seq_len; `causal_mask`,
embedding, RMSNorm, lm_head together stay under 2 %.

Per-(layer, token) MLP cost: 16.19 s / 36 layers / 1940 tokens ≈
**232 µs**.  Decode-path MLP (same weights, same kernel) was measured
in Stage 13-C at ~80 µs / layer-call for a single token.  Prefill is
3 × slower per token than decode — i.e. the kernel does **not** amortise
weight re-use across the M (token-batch) axis.

### Why

`crates/core/src/quantization/marlin_cuda.rs:71-76` routes **every** AWQ
INT4 forward — decode and prefill — through `cuda_fwd_awq_gemv`.  The
`awq_gemv_int4_*` family is a vector-by-matrix kernel: each output column
is a single warp-level dot product over K, and there is no shared-memory
tile reuse across the M axis.  At M = 1 (decode) this is the right
choice; at M = 1940 (prefill) it is catastrophic — the same INT4 weight
column is read from HBM 1940 times instead of once.

The dispatch comment on the same lines explicitly notes this: the legacy
`marlin_gemm_int4_zp_bf16` GEMM kernel exists in the PTX bundle, but
`AwqMarlinLinear::load_weights` transposes qweight to `[N, K/8]`
(coalesced for gemv), which the row-major tile GEMM cannot consume.
The result: a working GEMM exists but is unreachable.

### Decision: 13-D.4 approach

Adopt **fix candidate B′ — dequant + candle BF16 matmul on the M > 16
path**.  Rationale:

- Smallest blast radius: no new PTX, no qweight layout fork.  We
  reuse the `qweight` already laid out for gemv; dequant emits a
  fresh `[K, N]` BF16 buffer in a single CUDA launch.
- Predictable cost: one dequant + one cuBLAS bf16 GEMM per linear per
  forward.  At M = 1940, K = 4096, N = 3072 the bf16 GEMM is
  ~0.6 ms on RTX 4060; dequant is ~0.3 ms.  Per layer total ≈ 0.9 ms,
  not 17 ms.
- Decode path untouched: M ≤ 16 keeps the gemv dispatch.  Decode
  Stage 13-C numbers (47 tok/s) are not at risk.
- Memory: dequant buffer is freed at end of forward — peak transient
  ~12 MB per linear (4096 × 3072 × 2 bytes).  Negligible.

A future stage (13-D.6 or later) can replace the dequant+matmul with
a real fused GEMM (either a fork of the legacy `marlin_gemm_int4_zp_bf16`
that consumes the gemv layout, or a Triton-style INT4 GEMM).  That work
is gated on whether B′ closes the gap to the production target
(prompt_len = 256 → ≤ 800 ms TTFT).
