# Pool migration summary — Direction D Phases A + B

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), 8 GiB, WSL2
Model: Qwen/Qwen3-4B-AWQ
Stack: candle 0.10.2 + cudarc 0.19.4 (resolved through workspace)

## Goal

Make every per-layer scratch allocation on the decode hot path use a
stable device address (pre-allocated via the global `OutputPool`) so
CUDA Graph capture replay can dereference recorded pointers
correctly. Precondition for unblocking the ~10–15 % e2e win that
Graph capture promises.

## Commits (this session)

| commit  | scope |
|---------|-------|
| `c98b8ae` | A.1 — RMSNorm |
| `524523e` | A.4 — RoPE Q + K |
| `5fc2bf4` | A.2 — SwiGLU (new `silu_and_mul_separate_bf16` kernel) |
| `3c4762d` | A.3 — PagedAttention V2 (worst-case partition sizing) |
| `e05987a` | A.5 — FlashInfer wrapper into-buffer variant |
| `f5880db` | A.6 — Marlin INT4 GEMV pool re-enable (decode-only gate) |
| `6f29344` | B.1 — `block_tables` + `seq_lens` pool, env-gate capture |
| `dbbf2c1` | B.2/3/4 — embedding pool, cursor reset, `forward_decode_batch_with_ctx` in capture |
| `0f94e2a` | B.5 — pool-cursor reset before warmup |
| `3aab5b1` | B.6 — `lm_head` cuBLAS HGEMM into pool + `slot_mapping` pool |
| `ff8377a` | B.7 — pool-backed BF16 residual add |

Plus the FlashInfer integration (`bfb46c8`) and infra cleanup
(`38c191f`) that landed earlier in the same day.

## Eager-mode benefit (no graph capture)

Side-effect of the pool migrations: each forward now reuses a small
set of pre-allocated buffers instead of calling `cuMemAlloc` ~250
times per token. Sustained-load bench:

| concurrency | session start (165-baseline) | post-Phase-A+B (eager) | Δ |
|-------------|----------------------------|------------------------|---|
| c=1 | 44.3 | 44.6 | +0.7 % |
| c=4 | 26.0 | 25.9 | -0.4 % (noise) |
| c=8 (per-req) | 20.3 | 20.4 | +0.5 % |
| **c=8 (aggregate)** | **165** | **171.8** | **+4.1 %** |

The aggregate gain is bigger than per-req because pool reuse
reduces per-step jitter that bench's median-over-runs is more
sensitive to.

## CUDA Graph capture progress

End state of capture activation:

- `graphs_captured = 5/6` at warmup. (batch_size=32 still fails on
  input buffer alloc — separate, unrelated to pool.)
- `CUDA graph replay live` is reached on every replay attempt; the
  recorded kernel sequence executes.
- All **stale-pointer** issues are eliminated. Error progression
  through the session reflects the bisect:

  | error | semantic | what it told us |
  |-------|----------|----------------|
  | 905 | CAPTURE_ISOLATION | cudarc auto event-tracking — fixed by `disable_event_tracking()` |
  | 700 | ILLEGAL_ADDRESS | captured pointer to a freed buffer — fixed by pooling that allocation |
  | 1   | INVALID_VALUE   | non-contiguous slot_mapping/block_tables — fixed by pooling them |
  | 2   | OUT_OF_MEMORY   | captured `cuMemAllocAsync` accumulating per replay — **remaining** |

- Eager fallback is engaged (and works for several replays before
  the OOM corruption escalates), so server keeps responding; capture
  just doesn't deliver its benefit yet.

## Why error 2 remains unsolved this session

`cuMemAllocAsync` ops are still being recorded inside the captured
stream. With `VLLM_POOL_DIAG=1` we confirmed every `.contiguous()`
call in our pool wrappers receives an already-contiguous input — so
that's NOT the source.

The remaining suspect is something deeper:
- candle's `inplace_op2` internals may allocate scratch we don't see
  at the public API.
- `cudarc::cublas::CudaBlas::gemm` may allocate workspace per call.
- An unaudited tensor op buried in the model layer code.

Next step requires cudarc-level instrumentation (hook
`cuMemAllocAsync` with a stack-trace capture during a single
captured forward) — beyond what can be done in a single iteration.
Tracked in `memory/perf_cuda_graph_capture.md`.

## Session totals

- 13 commits.
- ~1500 lines of new pool-backed kernel + helper code.
- 4 new CUDA kernels (`silu_and_mul_separate_bf16`,
  `embedding_lookup_bf16`, `add_bf16`, `bf16_matmul` via cuBLAS
  HGEMM).
- 4532 / 4532 unit tests pass at every commit.
- E2E correctness `scripts/test_qwen3_awq_correctness.sh` 7/7 PASS
  in eager mode (capture path is gated by env so the test runs eager).

The session hit the eager-mode +4 % win as a real, committed
deliverable; capture activation is much closer than it was at
session start (replay live, all pointer classes resolved) but needs
one more session to identify the last alloc source.
