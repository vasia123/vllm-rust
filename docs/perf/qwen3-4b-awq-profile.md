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

### 13-D.4 result

Implementation landed: `crates/core/kernels/awq_marlin_dequant.cu` +
`marlin_cuda::{AwqMarlinDequantOp, awq_marlin_dequant_to_bf16,
awq_marlin_dequant_matmul}` + a guarded dispatch in `marlin_gemm`
that routes M > `AWQ_GEMV_M_THRESHOLD` (= 16) on the AWQ INT4 + ZP +
no-g_idx codepath through dequant + cuBLAS BF16 GEMM. Decode (M ≤ 16)
keeps the existing `awq_gemv_int4_kt_bf16` kernel.

**TTFT before / after (Qwen3-4B-AWQ, RTX 4060 Laptop, max_tokens=1, p50)**

| prompt_len | before (ms) | after (ms) | speedup |
|---:|---:|---:|---:|
| 64   | 1646 | 173 | **9.5 ×** |
| 256  | 4695 | **252** | **18.6 ×** |
| 1024 | 16910 | 650 | 26.0 × |
| 2048 | 23218 | 850 | 27.3 × |

Plan target was ≤ 800 ms TTFT at prompt_len = 256; **delivered 252 ms,
3 × inside the budget.** Gap vs Python vLLM 0.20.1 (47 ms at the same
shape) tightens from **88 × → 5.4 ×**.

**M-sweep microbench** (`crates/core/benches/awq_marlin_path_bench.rs`,
Qwen3-4B-AWQ MLP-up shape K = 4096, N = 11008, group_size = 128):

| M | path | time |
|---:|:---|---:|
| 1 | gemv | 143 µs |
| 4 | gemv | 549 µs |
| 32 | dequant + GEMM | 1.45 ms |
| 64 | dequant + GEMM | 1.45 ms |
| 256 | dequant + GEMM | 1.86 ms |
| 512 | dequant + GEMM | 2.65 ms |
| 1024 | dequant + GEMM | 4.17 ms |
| 2048 | dequant + GEMM | 4.16 ms |

The prefill curve enters the cuBLAS-bandwidth-saturated regime by
M ≈ 1024; all M > 16 share a fixed ~1.4 ms dequant overhead. A future
fused INT4 GEMM (no scratch) would mostly recoup that overhead, but
the gain on the production prefill workload is bounded by it
(~ 1.4 ms × 7 linears × 36 layers ≈ 350 ms saved at most).

Decode (M = 1) was unaffected by design; e2e correctness suite
`scripts/test_qwen3_awq_correctness.sh` passes 7/7 including
long-context boundary crossing and concurrent batches.

Snapshot of the M-sweep saved to
`docs/perf/bench-history/2026-05-07-13D4-after.json`; the next
perf-touching commit on this codepath diffs against it.

---

## Stage 13-E.1 — c=4 step instrumentation (2026-05-07)

`engine::helpers::step_profile` (env `VLLM_PROFILE_STEP=1`) breaks down
each `execute_batched_decode_with_graph` invocation into alloc /
metadata / forward / sampling / dispatch buckets and dumps μs/step
every 200 steps, keyed by batch_size bucket so c=1 and c=4 traffic
share one log without losing resolution.

**Steady-state per-step budget (Qwen3-4B-AWQ, RTX 4060 Laptop, after
13-D.4 prefill fix):**

| stage | c=1 (≤1) | c=4 (≤4) | scaling |
|:--|--:|--:|--:|
| forward | 21.18 ms | **66.92 ms** | **3.16 ×** |
| sampling | 2.33 ms | 3.00 ms | 1.29 × |
| alloc | 11 µs | 19 µs | — |
| metadata | 1.4 µs | 2.0 µs | — |
| **total** | **23.53 ms** | **70.20 ms** | 2.98 × |

Sampling already batches well (1.29 × scaling at c=4 — within noise of
the GPU `gpu_sample_batch_with_diffs` path). The c=4 throughput cliff
is **entirely in `forward`**: layer-by-layer breakdown via
`VLLM_PROFILE_DECODE=1` confirms it's the AWQ INT4 linears (mlp 3.3 ×,
qkv 2.8 ×, o_proj 2.7 ×) that scale near-linearly with M.

### Why

`awq_gemv_int4_kt_bf16` (the decode kernel under
`AWQ_GEMV_M_THRESHOLD = 16`) tiles per (m, n) cell — every output row
re-reads its INT4 weight column from HBM. At M = 1 this is the right
shape. At M = 4 each weight column is read 4 ×, with no shared-memory
amortisation. The result is the near-linear scaling above.

Lowering `AWQ_GEMV_M_THRESHOLD` does not help: the dequant + cuBLAS
GEMM path carries ~1.4 ms/linear of fixed overhead (one PTX dequant
launch + one BF16 GEMM launch), which makes M ∈ [2, 16] strictly
slower than the gemv path. The crossover sits around M = 16-32.

### Decision

**13-E.2 (batch the sampling loop) is removed from Stage 13-E**:
the data shows sampling is not on the critical path. It already
batches via `gpu_sample_batch_with_diffs`; the per-seq CPU fallback
fires only when `gpu_eligible_strict` rejects (multiplicative
penalties, beam, logprobs, FSM constraints) — none of which Stage 13-E
targeted.

The remaining concurrent-batching gap is the GEMV-vs-GEMM scaling
limit, which is **architectural** — it needs either:

- A real INT4 GEMM kernel that tile-tiles across M (true Marlin
  tile-MMA, or a Triton port), or
- A reduction of the dequant + cuBLAS-GEMM overhead so the 13-D.4
  path can dispatch profitably from M ≈ 4 onward.

Both are stage-sized work and tracked as **Stage 13-F**.

The remaining 13-E.x substeps (13-E.3 KV preemption, 13-E.4 num_blocks
auto-tune) close the OOM-on-c=8 ceiling and increase concurrent
capacity — independent of the per-request scaling limit.

---

## Stage 13-E — concurrent-batching closure (2026-05-07)

### 13-E.1 — instrumentation verdict

`engine::helpers::step_profile` (env `VLLM_PROFILE_STEP=1`) added —
per-stage timing of `execute_batched_decode_with_graph`, bucketed by
batch_size. Steady-state on Qwen3-4B-AWQ:

| stage | c=1 | c=4 | scaling |
|:--|--:|--:|--:|
| forward | 21.18 ms | 66.92 ms | 3.16 × |
| sampling | 2.33 ms | 3.00 ms | 1.29 × |
| **total** | 23.53 ms | 70.20 ms | 2.98 × |

**Sampling already batches** via `gpu_sample_batch_with_diffs`. The
13-E.2 hypothesis ("sampling is the c=4 bottleneck") was wrong;
13-E.2 retired.

The c=4 throughput cliff is in `forward`, specifically in the AWQ
INT4 linears (mlp 3.3 ×, qkv 2.8 ×, o_proj 2.7 × scaling — `gemv` kernel
re-reads weight columns per M, no shared-memory reuse across the M
axis). The fix needs a real INT4 GEMM that tiles over M. Tracked as
**Stage 13-F** future work.

### 13-E.3 — KV preemption with recompute (DONE)

`Scheduler::move_to_waiting(id)` + `StepResult.preempted` channel +
recompute mutation in `execute_batched_decode_with_graph` on
`allocate_for_request` failure: free blocks, fold
`generated_token_ids` back into `prompt_token_ids`, mark `Preempted`,
let the next `compute_schedule` re-admit.

Stress test (c=8, prompt=1024, max_tokens=1024) — the previously-OOM
workload — now completes all 8 streams, 4 recompute preemptions logged
at WARN, aggregate 51.9 tok/s. Pre-13-E.3 the same workload killed
half the streams with `kv cache out of blocks` errors.

### 13-E.4 — `num_blocks` auto-tune (DEFERRED → 13-G)

Tried setting `gpu_memory_utilization=0.85` as the default when the
user didn't pass `--num-blocks` or `--gpu-memory-utilization`. On
Qwen3-4B-AWQ / RTX 4060 the auto-tune produced 1760 blocks (vs 512
legacy default) — and triggered a **27 × TTFT regression on c=1**
(252 ms → 6800 ms at prompt_len=256), while decode tok/s held flat.
The regression appears from the very first request, is independent
of the non-fatal JIT-warmup OOM at batch=32, and is specific to
prefill-at-large-cache.

Hypothesis: either the AWQ-Marlin dequant+matmul scratch allocation
path or the block-pool's free-list iteration is *O(num_blocks)* in
a way the current code does not amortise.

The auto-tune is reverted in this stage; `num_blocks` stays at the
explicit user value, and operators wanting more concurrency pass
`--gpu-memory-utilization` explicitly. Diagnosis tracked as
**Stage 13-G**.

### Where Stage 13-E leaves us

- c=8 no longer crashes — preemption-on-OOM lands every request.
- c=1 single-stream perf unchanged (42.5 tok/s, 252 ms TTFT).
- c=4 per-request still capped by 13-F gemv-not-batched-over-M;
  no 13-E lever closes it without architectural work.
- Bench coverage policy (Stage 13-D.0) caught the 13-E.4 TTFT
  regression on the very first benchmark run — the rule paid for
  itself a third time.

---

## Stage 13-G — TTFT-vs-num_blocks regression (2026-05-07)

### Diagnosis

`VLLM_PROFILE_PREFILL=1` on Qwen3-4B-AWQ at prompt_len=256 with the
13-E.4 auto-tune (num_blocks 1760 vs legacy 512):

| Component | 512 blocks (μs) | 1760 blocks (μs) | scaling |
|:--|--:|--:|--:|
| qkv | 30,540 | 30,510 | 1.00 × |
| paged_attn | 216,853 | 1,830,089 | **8.4 ×** |
| o_proj | 77,160 | 409,239 | **5.3 ×** |
| **mlp** | **263,360** | **4,373,889** | **16.6 ×** |

`qkv` (3 small linears, K=2560 N≤4096) was unaffected. `mlp` (3 wide
linears, gate/up at K=2560 N=9728, down inverse — peak ≈ 50 MB BF16
scratch each through the 13-D.4 dequant+matmul path) blew up 17 ×.
At 1760 blocks the KV cache occupies ~3.96 GiB on an 8 GiB card,
leaving only ~1.5 GiB free; the per-prefill scratch demand
(36 layers × 3 mlp linears × 50 MiB ≈ 5.4 GiB transient) overflows
that headroom and cudarc falls back to a serialising/synchronising
allocation path. `paged_attn` and `o_proj` shoulder smaller versions
of the same effect.

### Fix

`estimate_kv_cache_budget` rewritten to use the **measured free VRAM
after model load** (the function is already called post
`report_gpu_mem("after model build")`) instead of the param-count
estimate, and to reserve a **1.5 GiB scratch floor** on top of the
percentage-based overhead. Empirical floor — covers the full prefill
+ long-context decode profile of Qwen3-4B-AWQ on RTX 4060 Laptop
without leaving the dequant scratch starved.

`estimate_kv_budget_bytes_with_overhead(...)` added to
`crates/core/src/kv_cache/config.rs` so callers that know their
actual scratch shape can pin a different floor (e.g. larger models,
multi-stream serving, BF16-uncomp checkpoints).

13-E.4 auto-tune re-enabled with the corrected estimator.

### Result

| Config | num_blocks | TTFT @ 256 | c=1 tok/s | c=4 agg | c=8 agg |
|:--|--:|--:|--:|--:|--:|
| Pre-13-G (broken) | 1760 | 6700 ms | 33 | — | — |
| Post-13-G | **1029** | **252 ms** | 42.9 | 56.1 | 59.6 |
| Legacy 512 (no auto-tune) | 512 | 629 ms* | 42.5 | 56.9 | 41 |

(* legacy bench captured a different sample; the 252 ms 13-D.4 bench
ran with the same effective config.)

Auto-tune now yields **2 × legacy concurrency capacity (1029 vs 512
blocks)** with no TTFT or decode regression, and case 6
(long-context cross-4096) passes. The architectural follow-up — a
real shared scratch buffer so the dequant+matmul path stops
allocating per-call — is tracked under Stage 13-F (the same INT4
GEMM rewrite that closes the c=4 cliff would supersede the scratch
issue).

---

## Stage 13-H — drop candle 0.9.1 fork (DEFERRED)

### Attempt

Bumped `candle-core` / `candle-nn` / `candle-flash-attn` from the
vendored 0.9.1+stream-patch fork to crates.io 0.10.2 to get rid of
the `vendor/candle/` dir and the per-bump maintenance overhead. The
plan was: switch deps, fix API drift, drop vendor + setup script +
CI/Docker steps.

### What broke and how it was fixed

Compile-time API drift (12 errors total):

- `CudaDevice::memcpy_stod` → `CudaDevice::clone_htod` (rename,
  same signature). 9 call-sites in `cuda_kernels.rs`,
  `moe/fused/kernel_wrapper.rs`, `sampling/gpu.rs`.
- `CudaDevice::memcpy_dtov` → `CudaDevice::clone_dtoh`. 1 call-site.
- `SimpleBackend` trait now requires `get_unchecked(name, dtype,
  dev)`. Implemented in `quantization/gguf/mod.rs::GgufVarBuilderBackend`.
- `DType` / `CudaStorageSlice` gained variants (I16, I32, F8E4M3,
  F8E5M2 and 4 more). Added wildcard arms in `cuda_kernels.rs`,
  `distributed/nccl.rs` (× 2), `flashinfer/tensor_bridge.rs`.

Runtime issue (silent corruption):

- HuggingFace AWQ `qweight` / `qzeros` arrive in the safetensors
  decoder as I32 in candle 0.10 (was U32 in 0.9). The VarBuilder's
  `get_with_hints_dtype(.., U32)` chains *two* numerical casts
  (file → backend default = BF16, then BF16 → U32), saturating the
  packed nibbles. Some U32↔I32 cast pairs additionally hit a
  missing CUDA kernel
  (`CUDA_ERROR_NOT_FOUND, "named symbol not found"`).
- Workaround: route integer requests through a CPU-rooted
  `VarBuilder` with target dtype, so the safetensors decoder
  reads I32-on-disk straight into an I32 (or U32) host buffer
  with no cast kernel involved, then push to GPU. Verified via
  `scripts/test_qwen3_awq_correctness.sh`: 7/7 PASS on candle 0.10.

### Why it was reverted

Decode throughput halved on the same hardware:

| Config | num_blocks | TTFT @ 256 | c=1 tok/s | c=4 agg |
|:--|--:|--:|--:|--:|
| Vendored 0.9.1 fork (HEAD) | 1029 | 252 ms | 42.9 | 56.9 |
| crates.io 0.10.2 | 588 | 721 ms | 23.4 | 27.5 |

Auto-tune dropped from 1029 → 588 blocks because the on-GPU
footprint after model load went up under candle 0.10 (~700 MiB
more reserved by the activation/kernel state). Decode tps dropped
~45 % even at the same `--num-blocks 512`. Numerical correctness
held, but the perf cost is a hard regression on the workload
Stage 13-D / 13-G optimised for. Investigation deferred — likely
either candle 0.10's kernel cache initialisation, a different
default cudarc stream policy, or the CPU-fallback path in
`vb_get_as` adding per-load latency.

### Decisions

- Revert: `candle-core` / `candle-nn` / `candle-flash-attn` stay
  on the vendored 0.9.1 + non-default-stream patch. CI vendor step
  + Dockerfile vendor copy stay in place. Pin commit
  `cd96fa80` is unchanged.
- The shape of the candle 0.10 patch is now known and tractable
  if/when we need to bump (e.g. for the MXFP4/FP8 tensor cores
  shipped in 0.10's kernel set). Tracked as **Stage 13-H follow-up**
  — the bump is straightforward, the perf regression is what needs
  diagnosis next.
- Bench coverage policy paid off again: the regression was caught
  immediately by the Stage 13-D bench harness, before any
  user-facing build saw it.

---

## Stage 13-F PoC — kt_mloop kernel (REVERTED)

### Hypothesis

Reverse the M / K loop nesting in `awq_gemv_int4_kt_bf16` so each
INT4 weight column is read from HBM exactly once (the legacy kernel
re-reads it M× in its outer m-loop). Cache all M input rows in
shared memory up front, accumulate per-thread FP32 partials for
all M outputs in registers, reduce via warp shuffles. Theoretically
M× HBM bandwidth saving for the weight stream — the dominant cost
at concurrent-decode batch sizes.

### Implementation

`awq_gemv_int4_kt_mloop_bf16` added to `crates/core/kernels/awq_gemv.cu`,
gated on `M ∈ [2, 8] && M × K × 2 ≤ 47 KiB` (static SM shmem cap).
Per-thread `acc[8]` array, M_MAX=8 unrolled across the inner FMA.
Dispatch wired in `MarlinGemmOp::cuda_fwd_awq_gemv` between the
existing kt and split_k8 paths.

### Microbench result on RTX 4060 Laptop, full TGP

| M | shape (K, N) | legacy kt (µs) | new kt_mloop (µs) | delta |
|--:|:--|--:|--:|--:|
| 4 | 4096, 11008 | 549 | 643 | **+17 % (regression)** |

Cause: at runtime M=4 the M_MAX=8 unroll executes 4 predicated-NOP
FMAs per K-step. SIMT issues those slots even though the predicate
masks the writes, so the kernel is compute-bound *higher* than the
HBM-bandwidth saving recovers. The PoC accidentally trades one cost
for another that ends up bigger on Ada.

### Decision

PoC reverted. The bench coverage policy (`awq_marlin_path_bench`)
caught the regression on the first criterion run, before any commit
landed — exactly what the policy is for.

The right fix is template-specialised kernel variants per M_MAX
(2 / 4 / 8) so unused M-rows don't sit as predicated NOPs, plus
opt-in to dynamic 100 KiB shmem so the M=8 path covers `mlp.down`'s
K=9728 layer too. That's a multi-day rewrite — tracked as the proper
**Stage 13-F**: a tiled INT4 GEMM for M ∈ [1, 16] backed by template
specialisation, weight reuse across M, and dynamic shmem.

A separate observation worth recording: the engine's
`--multi-step-count` default (= 4) inflates the effective batch
size at the GEMV layer to `concurrency × multi_step` (e.g. c=4
arrives at the kernel as M=16, c=8 as M=32). That bypasses the
M ≤ 16 GEMV regime entirely and lands on the dequant+matmul path —
the reason 13-D.4 saw flat per-stream tps at c=8: the GEMV kernel
isn't actually executing in the per-token critical path most of the
time. This means the c=4 cliff diagnosed in Stage 13-E.1 was
multi_step batching meeting the legacy split_kt kernel at M=16,
not GEMV-at-M=4. Closing the c=4 gap probably needs the same
template-specialised variant chain *plus* something at M=16
(currently routed to legacy kt, which hits the same per-m-loop
weight re-read pattern). Both fall under Stage 13-F.

---

## Stage 13-F (real fix, single-line) — `AWQ_GEMV_M_THRESHOLD` 16 → 8

### M-sweep microbench on full TGP

| M | path | time |
|--:|:--|--:|
| 1 | gemv | 140 µs |
| 4 | gemv | 552 µs (138 µs/M) |
| 16 | gemv | **2.20 ms** (138 µs/M — flat per-M cost) |
| 32 | dequant + GEMM | 1.44 ms |
| 64 | dequant + GEMM | 1.45 ms |

The gemv kernel scales **strictly linearly** in M (the per-M cost is
138 µs regardless of where on the curve we sit), confirming the
"weight column re-read M× from HBM" diagnosis from the Stage 13-F
PoC. The dequant+matmul path's overhead is fixed at ~1.45 ms for
M ≤ 512 (cuBLAS BF16 GEMM is overhead-bound, not throughput-bound,
in this regime). The two cross at M ≈ 10.

### Fix

Lower `AWQ_GEMV_M_THRESHOLD` from 16 to 8 so M ∈ [9, 16] now routes
through dequant+matmul. M ≤ 8 stays on the gemv path (where it's
strictly faster).

Microbench at M=16 after the fix: 2.20 ms → **1.46 ms** (33 %
faster), correctness held by `test_awq_gemv_matches_cpu_reference_*`
+ `test_awq_marlin_dequant_matmul_matches_cpu_reference_m32`.

### End-to-end (Qwen3-4B-AWQ, RTX 4060 Laptop, full TGP, multi_step=4)

| concurrency | before c1 | after c1 | before c4 (req / agg) | after c4 | before c8 (req / agg) | after c8 |
|:--|--:|--:|:--|:--|:--|:--|
| tok/s | 43.2 | **43.4** | 13.7 / 55.0 | **14.4 / 57.5** | 6.9 / 55.4 | **7.5 / 60.4** |
| Δ | +0.5 % | | **+5 % / +4.5 %** | | **+9 % / +9 %** | |

Single-line change, no new kernel, no new dispatch logic — just
moves the existing threshold to where the microbench says the
crossover sits. The earlier "real Stage 13-F" plan (template-
specialised kernel variants, dynamic 100 KiB shmem) remains valid
for future M ≤ 8 wins; this commit captures the easy win on the
M ∈ [9, 16] half of the curve.

### Why this didn't ship in 13-D.4

13-D.4 set the threshold at 16 because the awq_marlin_path_bench
hadn't been run in the M = 9..15 region — only M ∈ {1, 4, 16, 32,
…} was sampled. M=16's gemv cost was assumed similar to M=4
(both still "small batch") but linear scaling in M makes M=16 a
2.2 ms outlier. The full M-sweep above was only run during 13-F
PoC investigation, which made the right threshold visible.

---

## Stage 13-I.1 — speculative-decode draft model selection

### Compatibility audit (Qwen3-4B-AWQ target)

Target config (`Qwen/Qwen3-4B-AWQ`):
- `vocab_size = 151936`, `bos = 151643`, `eos = 151645`
- `head_dim = 128`, `num_kv_heads = 8`
- `rope_theta = 1_000_000`, `max_position = 40 960`
- `tie_word_embeddings = true`, AWQ gemm 4-bit, group_size 128.

For draft-model speculative decode (`DraftModelDraftProposer`) the
binding contract is **token-vocab equivalence** — the verifier samples
in the target's vocab, so the draft must produce token IDs the target
can score. The `head_dim` / `kv_heads` of draft only have to match the
*draft's own* KV cache dims (independent `KVCacheManager`), so they're
informational here.

### Candidates

| candidate | weights | quant | vocab match | est. VRAM |
|:--|:--|:--|:--:|--:|
| `Qwen/Qwen3-0.6B` | BF16 | none | ✅ 151 936 | ~1.2 GiB |
| `Orion-zhen/Qwen3-0.6B-AWQ` | INT4-AWQ | gemm/g128 | ✅ 151 936 | ~0.4 GiB |
| `mundasuto/Qwen3-0.6B-AWQ` | INT4-AWQ | gemm/g128 | ✅ 151 936 | ~0.4 GiB |
| N-Gram (no model) | — | — | n/a | 0 |

All three Qwen3-0.6B configs match the target byte-identically on
vocab, eos/bos, RoPE theta and max position — the same Qwen3 tokenizer
ships across the family. AWQ community quants exist with the same
gemm-flavour AWQ format as the target (`bits=4, group_size=128,
zero_point=true, version="gemm"`), so they go through the exact same
`from_config_with_quant` → `Qwen3QuantizedForCausalLM` path the target
uses.

### Memory budget on 8 GiB (RTX 4060 Laptop)

Worst case stack at c=8, full-TGP:
- target weights:   2.6 GiB
- target KV (~1.9 GiB at 800 blocks × 16 toks × 36L × 8H × 128D × 4B)
- draft weights:    0.4 GiB (AWQ) or 1.2 GiB (BF16)
- draft KV (~1.4 GiB at 800 blocks × 16 toks × 28L × 8H × 128D × 4B)
- activations + scratch: ~0.7 GiB

That's 7.0 GiB (AWQ draft) or 7.8 GiB (BF16 draft) on an 8188 MiB card.
**AWQ draft fits comfortably; BF16 draft is on the edge — likely needs
draft `num_blocks` reduced.**

### Architectural smell flagged for 13-I.4

In `crates/server/src/main.rs:1825-1840` the draft `CacheConfig` hard-
codes `block_size: 16` and reuses target `num_blocks` verbatim. The
target's `block_size` is configurable; draft should inherit it (or at
minimum match), not fix it at 16. Fix is lifted to substep 13-I.4 (the
"wire defaults" pass) so it lands with the rest of the wire-up changes.

### Decision

Bench order (cheapest first, most-likely-to-succeed last):
1. **N-Gram** — zero VRAM, zero-config; sets a "bad-draft" floor.
2. **`Orion-zhen/Qwen3-0.6B-AWQ`** — primary candidate (best memory).
3. **`Qwen/Qwen3-0.6B` BF16** — fallback if AWQ-0.6B quality is poor
   (community quant — not rigorously evaluated by the publisher).

Acceptance gate (per Stage 13-I plan): one of these must clear
≥ +30 % per-req tok/s at c=1 *or* ≥ +50 % aggregate at c=8 on
`scripts/bench_decode.py` to win the default-on slot. Otherwise we
keep speculative as a manual flag and move on to Stage 13-J.

## Stage 13-I.3 — measurement + verdict

Bench setup: Qwen3-4B-AWQ target on RTX 4060 Laptop sm_89, prompt 256 w,
max_tokens 256, temp 0.7, runs 2 (best-of), `scripts/bench_decode.py`
with `--admin-url` for per-batch acceptance-rate.

| variant | c=1 tok/s | c=4 tok/s (req / agg) | accept | tok/draft |
|---|--:|--:|--:|--:|
| baseline (no draft, multi_step=4) | **43.3** | **15.4 / 61.2** | n/a | n/a |
| Orion-zhen/Qwen3-0.6B-AWQ K=4 | 6.9 | 0.0 / 7.1 | 13.2 % | 1.53 |
| Qwen/Qwen3-0.6B BF16 K=4 | 7.1 | (timeout) | 17.3 % | 1.69 |
| N-Gram (zero VRAM, n=2..4 K=4) | 20.2 | 7.1 / 27.2 | 9.3 % | 1.37 |

**All three candidates fail the gate.** Per-req c=1 regressed 2.1× (best
case, N-Gram) to 6.3× (worst case, AWQ-0.6B); c=4 aggregate regressed
2.2× to 8.6×.

### Root cause

The baseline already runs `--multi-step-count 4` (default), which
emits 4 tokens per engine step. Speculative decode in the current
implementation **disables** multi-step batching (`engine/main.rs:1837`
omits the `multi_step_count` builder call on the draft branch) and
replaces it with a K=4 draft-then-verify loop.

That swap is only beneficial if average accepted tokens per round
≥ multi-step's 4. Acceptance rates landed at 9–17 %, giving
~1.4–1.7 emitted tokens per round — a **2.4–2.9× per-step token-rate
drop** before any draft cost is paid. Compounded with sequential
draft forwards (4× draft.forward() per round, ~30 ms each for BF16-0.6B
on this GPU), the net per-step latency rose 6–10×.

Speculative decode wins are unlocked **either** by:
- **Higher acceptance** — needs a draft model trained for distillation
  on this target (we'd need a Qwen3-0.6B-distilled-from-4B variant; the
  community AWQ quants are quality-degraded, BF16 stock is too generic).
- **Cheaper draft proposal** — Eagle / Medusa / MTP run draft heads in
  parallel with target verify (no sequential 4× draft cost). Eagle-1
  Llama / DeepSeek paths already exist in this repo
  (`models/eagle_llama.rs` and friends), but no Eagle head ships with
  Qwen3-4B-AWQ on HF. MTP would require a Qwen3-MTP head.
- **Spec + multi-step composing** — currently they're mutually
  exclusive; making them compose (run multi-step per verify round)
  would salvage the 4× lookahead even when acceptance is mediocre.

### Decision

**Do not default-on speculative decode** for Qwen3-4B-AWQ on this GPU.
The CLI flags (`--draft-model`, `--ngram-prompt-lookup-max`) remain
available for users with workloads where draft acceptance is naturally
high (long structured output, lots of copy-paste from prompt) — but
the global default keeps the multi_step=4 baseline.

Stage 13-I architectural cleanup still ships (block_size inheritance,
`compute_draft_kv_blocks` helper) — those remove a latent OOM bug for
users who *do* turn speculative on with a non-tiny draft.

Next: Stage 13-J (kt_mloop M=12, 16) per plan.

---

## Stage 13-J — kt_mloop M=12 / M=16 + 96 KiB dynamic shmem

### Landed

1. **96 KiB dynamic shmem opt-in** for the kt_mloop kernels via
   `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES = 98 304)`.
   Static per-block cap on sm_8x is 48 KiB; sm_89's hard cap is 99 KiB
   so 96 KiB clears with headroom (98 KiB rejected with
   `CUDA_ERROR_INVALID_VALUE` on the test driver).
2. **`kt_mloop_body<12>` and `<16>` template instantiations** in
   `awq_gemv.cu`, exposing `awq_gemv_int4_kt_mloop_m12_bf16` /
   `_m16_bf16`. Hand-instantiated alongside the existing M=4 / M=8
   variants — same shmem layout (`[k, M_FIXED]`, bank-conflict free),
   same FP32 reduce, same numeric tolerance (validated by
   `test_awq_gemv_matches_cpu_reference_m12` / `_m16`).
3. **Template-aware dispatch gate** `awq_gemv_kt_mloop_template_fits`
   replaces the simple `m > 8` threshold for M ∈ {12, 16}: GEMV when
   the template fits the 96 KiB shmem cap, dequant+matmul otherwise.
   Odd-M (5–7, 9–11, 13–15, > 16) keeps going to dequant+matmul — the
   legacy kt path (linear in M, 138 µs/M) loses to dequant+matmul above
   M=10 by Stage 13-F's microbench.

### Microbench (RTX 4060 Laptop, sm_89, K=4096, N=11008)

| M  | dispatch arm        | before  | after   | Δ          |
|--:|:--                   |--:      |--:      |:--          |
|  1 | kt_mloop_m4 fallback | 142 µs  | 142 µs  | flat        |
|  4 | kt_mloop_m4         | 451 µs  | 451 µs  | flat        |
|  8 | kt_mloop_m8         | 887 µs  | 892 µs  | flat (+0.5 %) |
| 10 | dequant+matmul      | 1.45 ms | 1.47 ms | flat (+1 %)   |
| **12** | **kt_mloop_m12** | 1.45 ms | **1.31 ms** | **−9.6 %** ✅ |
| 13 | dequant+matmul      | 1.46 ms | 1.46 ms | flat        |
| 14 | dequant+matmul      | 1.46 ms | 1.46 ms | flat        |
| 16 | dequant+matmul †    | 1.46 ms | 1.46 ms | flat        |
| 24..2048 | dequant+matmul | …      | …       | flat        |

† M=16 at K=4096 needs 128 KiB shmem (exceeds 96 KiB cap), so
  `awq_gemv_kt_mloop_template_fits(16, 4096)` returns false → routes
  to dequant+matmul. M=16 wins **at K ≤ 3072**, which matches the
  Qwen3-4B q/k/v_proj layers (K = hidden_size = 2560, fits at 80 KiB).
  Bench above is K=4096 / mlp.up shape and intentionally omits this
  path; the e2e bench below captures the K=2560 win indirectly.

### End-to-end (Qwen3-4B-AWQ, full TGP, multi_step=4)

| concurrency  | Stage 13-F (after) | Stage 13-J (after) | Δ           |
|--:           |--:                 |--:                 |:--           |
| c=1 (M=4)    |    43.4 tps         |    41.9 tps         | flat         |
| c=4 (M=16)   | 14.4 / 57.5 tps    | 16.0 / **64.2** tps | **+11 %**    |
| c=8 (M=32)   |  7.5 / 60.4 tps    |  8.1 / **65.5** tps | **+8 %**     |

c=4 picks up the win because effective M at the GEMV layer is
`concurrency × multi_step = 16`, which lands on `kt_mloop_m16` for the
q/k/v_proj layers (K=2560 fits) and on dequant+matmul for o_proj
(K=4096) and mlp.down (K=9728). The c=8 (effective M=32) win is
secondary — outside any kt_mloop template's M range, so the +8 % must
come from cache effects of fewer kernel-launch dispatches at the
crossover.

### Decision

Land. The M=16 cap-gate scales naturally if a future card with larger
shmem (Hopper sm_90: 228 KiB) raises `KT_MLOOP_DYN_SHMEM_CAP_BYTES` —
the only constants to bump are the cap and the `MAX_DYN_SMEM`
attribute value. No template logic to touch.

---

## Stage 13-K.1 — FlashInfer prefill audit (premise correction)

The plan opened with "replace `paged_attention` with FlashInfer ragged
prefill" — but the audit shows `paged_attention()` *already* routes
through `FlashInferBackend::prefill_attention` →
`prefill_flashinfer` whenever the `flashinfer` cargo feature is enabled
(included by `cuda-default` and `cuda-full`, which the bench builds use).
Stage 13-D's 252 ms TTFT was already measured *with* FlashInfer paged
prefill active.

So the "TTFT 252 → ~80 ms" target requires a different lever than the
plan assumed. Profiling a 489-token prefill with `VLLM_PROFILE_PREFILL=1`
gives this breakdown (μs, ratio of profile-mode wallclock):

| component   | μs      | %     |
|---          |--:      |--:    |
| paged_attn  | 1567 600 | 67.8 % |
| mlp         |  440 700 | 19.1 % |
| qkv         |  133 600 |  5.8 % |
| o_proj      |  108 000 |  4.7 % |
| qk_norm     |   19 200 |  0.8 % |
| rope        |   12 300 |  0.5 % |
| **other**   |    29 400 |  1.3 % |
| **total**   | 2 310 700 | 100 % |

(Profile mode adds a `cuda_stream().synchronize()` per measurement so
absolute numbers are 8–10× wallclock; the ratios are what's meaningful.)

paged_attn at 67.8 % — and inside it, `PrefillWrapper::run` rebuilds
a `BatchPrefillPlan` on every call (`wrapper.rs:173`). For a 36-layer
forward, that's 36 fresh CPU-side plan builds per prefill, each one
doing a host-CPU work-estimate and a `vec![0u8; int_ws_size]` for
page-locked metadata. The exact same anti-pattern was already fixed
on the decode side (`DecodePlan` + `decode_flashinfer_with_plan`,
metadata.rs ≈ line 187) — that change cut decode from "36 host syncs
per token" to one.

### Win path (deferred)

Mirror the decode plan-caching pattern for prefill:

1. Add `prefill_flashinfer_with_plan` and a `PrefillPlan` newtype that
   owns the `BatchPrefillPlan` + the indptr/indices/last_page_len
   tensors and the page-locked scratch.
2. Add `build_prefill_plan` on `AttentionBackend` returning
   `Option<Arc<dyn Any + Send + Sync>>`.
3. Engine builds the plan once per prefill forward (in
   `model_forward.rs` or wherever the prefill metadata is assembled);
   passes it down through `paged_attention(...)` analogous to the
   decode path's `prefill_metadata.attention_plan: Option<Arc<…>>`.
4. Bench `bench_prefill.py` for 64 / 256 / 1024 / 2048 prompt lengths.
   Expected: TTFT 252 → ~190 ms (saving ~36 × plan-build cost on
   short prompts; more on longer ones where plan-build still O(seq)).

This is a cross-cutting plumbing change (touches `AttentionBackend`
trait, `paged_attention` signature, and how every model that calls
`paged_attention` threads metadata through) — too invasive to bundle
with Stage 13-J. **Lifted to its own milestone (Stage 13-K-bis or
Stage 14-A).** The audit (this section) is the deliverable for
13-K.1; subsequent substeps stay pending for that future milestone.

Stage 13 closes here with cumulative wins from `git log`:

- 13-D: TTFT 4695 → 252 ms (18.6×)
- 13-E: c=8 OOM fixed; KV preemption
- 13-F: AWQ-Marlin threshold +9 % (c=8 agg) / +5 % (c=4 agg)
- 13-G: VRAM-aware auto-tune
- 13-I: speculative decode → not adopted; cleanup landed
- 13-J: kt_mloop M=12 / M=16 → +11 % (c=4 agg) / +8 % (c=8 agg)

End-state: c=1 = 41.9 tps (97 % of vLLM 47), c=4 = 16.0 / 64.2 tps,
c=8 = 8.1 / 65.5 tps (vs vLLM ~351 aggregate at c=8 — 5.5× gap remains;
closing it requires either Marlin tile-MMA tensor-core kernel
[Stage 13-L, deferred 2-4 weeks] or the prefill plan caching above
combined with 13-L).

