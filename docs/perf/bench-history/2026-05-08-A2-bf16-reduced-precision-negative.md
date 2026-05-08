# Stage A'.2 — `set_gemm_reduced_precision_bf16(true)` — NEGATIVE result

Date: 2026-05-08
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Hypothesis tested:
  candle-core 0.10.2 defaults `MM_BF16_REDUCED_PRECISION = false` →
  cuBLAS BF16 matmul uses `CUBLAS_COMPUTE_32F` (FP32 accumulator).
  Setting it to `true` switches to `CUBLAS_COMPUTE_32F_FAST_16BF` which
  widens the algorithm picker's search to BF16 tensor-core fast
  variants. Theory: this would close the 5-7× gap between microbench
  (3 ms) and production (16-24 ms) lm_head matmul.

## Test method

Added `candle_core::cuda::set_gemm_reduced_precision_bf16(true)` at
the start of `main()` in `crates/server/src/main.rs`, rebuilt release,
ran the same VLLM_PROFILE_DECODE c=8 bench.

## Result

| metric | before | after |
|---|---|---|
| c=1 tps | 42.6 | 38.7 (within noise) |
| c=4 tps | 81.0 | 79.2 (within noise) |
| c=8 tps | 105 | 111.5 (within noise) |
| lm_head_evt µs | 15,694-23,544 | 18,116-27,299 |
| lm_head_sync µs | 15,789-23,723 | 18,036-27,092 |

**No measurable improvement.** lm_head time unchanged (slightly higher
than baseline within run-to-run variance). End-to-end tps unchanged.

## Implication

The 5-7× production-vs-microbench gap is NOT from cuBLAS picking a
slow algorithm under `CUBLAS_COMPUTE_32F`. The fast-16BF variant gives
the same time. Either:

1. cuBLAS already picks the same fast algo at this shape regardless of
   compute_type setting.
2. The bottleneck is **outside** the matmul kernel itself — somewhere
   between Rust host code launch and kernel start. Plausible candidates
   that REMAIN unverified without nvprof:
   - `dev.alloc::<bf16>(elem_count)` per-call output allocation under
     fragmented allocator
   - cuBLAS handle internal state churn from 252 prior matmuls per
     forward (each layer: qkv + 2× MLP via TC + o_proj = 4-5 matmuls)
   - Stream serialisation if FlashInfer plan/run uses a separate stream

## Decision

Reverted (single-line removal). Documented as negative.

Next steps to actually fix lm_head require:
- **nvprof / Nsight Compute** under production load to pin down where
  the 16-24 ms is spent (kernel time vs launch overhead vs stream
  wait). This is the highest-information action.
- If kernel itself is slow under context: replace lm_head with a
  direct cudarc `gemm` call (not `gemm_strided_batched`), with
  pre-allocated output buffer, on a dedicated stream. Multi-day scope.
- If launch/setup overhead is the issue: investigate persistent cuBLAS
  workspace, handle reuse, or full custom kernel.

Without nvprof data the next attempt is shooting in the dark, like
this one was. **Stop optimising lm_head until profiling data exists.**

Memory note `feedback_perf_assumption_test.md` updated implicitly: this
is the third "fix on theory without isolated profile" attempt that
reverted to baseline today (after 15.E.3 hybrid-software, lm_head
pre-transpose). Pattern is clear — without root-cause measurement,
hypothesis-driven fixes have ~0% hit rate.
