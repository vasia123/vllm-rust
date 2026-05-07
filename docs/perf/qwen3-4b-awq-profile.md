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

## Next: Phase 2 plan (Β — paged_attn for M=1)

To be detailed in next plan iteration. Direction:

1. Audit current `paged_attn_v1.cu` / `paged_attn_v2.cu` kernel
   geometry. Confirm V1 vs V2 selection threshold for our seq_len
   range (≈ 100–800).
2. Design M=1 specialized kernel:
   - Split-query per group of KV heads (GQA). For Qwen3-4B
     (32 q-heads, 8 kv-heads, group=4), each block can serve one
     KV-head group → 8 blocks/seq instead of 32, but each block
     does 4× more compute per output. With 128 threads/block ×
     8 blocks = 32 warps active, still low occupancy on 24 SMs.
   - Better: split-K along sequence dimension. Each block processes
     a chunk of the seq, partial reduction across blocks via
     atomic add or two-stage reduce. This is essentially what
     V2 does, but tuned for M=1 (vs M up to 16 currently).
   - Or adopt FlashInfer decode kernel via the existing
     `_with_plan` infrastructure (commit a fresh plan caching
     pass at the engine level).
3. Measure on real shapes (seq_len 100, 500, 1000); pick the
   variant that wins all three.
4. Numerical tolerance: BF16 ±1e-2 vs current.

Expected gain: paged_attn 28% → ≤10% via 3× kernel speedup.
End-to-end: 22.7 ms/token → ~17 ms/token = **~58 tok/s**.
