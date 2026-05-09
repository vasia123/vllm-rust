# CUDA Graph capture activation — Phase C A/B results

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), 8 GiB, WSL2
Model: Qwen/Qwen3-4B-AWQ
Stack: candle 0.10.2 + cudarc 0.19.4
Commit at measure time: 86806b2 (Phase B.8 — pool positions+slot_mapping)

## Setup

Both runs against the same release binary in the same shell session,
back-to-back. Server flags:

- capture: `--max-model-len 1024 --num-blocks 384` (no --enforce-eager)
  + `VLLM_DISABLE_EVENT_TRACKING=1`
- eager:   `--max-model-len 1024 --num-blocks 384 --enforce-eager`

Both with `VLLM_AWQ_HYBRID=1`. RUST_LOG=warn so logging cost is not
in the loop.

Bench: `scripts/bench_decode.py --prompt-len 128 --max-tokens 64
--runs 3` for c=1, c=4, c=8.

## Results

| concurrency | eager (med tps/req, agg) | capture (med tps/req, agg) | Δ agg |
|-------------|--------------------------|----------------------------|-------|
| c=1         | 48.1 / 48.1              | 53.1 / 53.1                | +10.4% |
| c=4         | 28.3 / 113.1             | 31.0 / 123.8               | +9.5%  |
| c=8         | 26.4 / 211.1             | 28.2 / 225.5               | +6.8%  |

All three concurrencies show capture > eager. The lift narrows at
higher concurrency because per-token GPU work dominates over kernel
launch overhead at c=8. At c=1 capture is purely a launch-overhead
optimization, so the +10% there is the cleanest signal.

## Capture-path health

Server log on capture run:
- `graphs_captured = 5` (sizes 1, 2, 4, 8, 16; batch=32 fails on
  input alloc, separate)
- `graph decode failed = 0` across the entire bench
- GPU memory stable across requests (no leak, no OOM)

This is a complete reversal from the pre-Phase-B.8 state, where
the SECOND decode forward of any multi-token generation crashed
with ILLEGAL_ADDRESS. See `memory/perf_cuda_graph_capture.md` for
the root cause.

## Decision

Capture-on is now production-ready for Qwen3-AWQ on this hardware:

- Correctness: probe (`/tmp/probe_batch.sh`) returns valid
  generations at 1/10/50 tokens.
- Perf: +6-10 % across c=1..8.
- Stability: no replay errors, no memory growth.

Default flip is gated by env (`VLLM_DISABLE_EVENT_TRACKING=1`)
because the underlying enabler — `disable_event_tracking()` in
cudarc 0.17+ — is not exercised by other code paths yet, so the
default-off keeps eager-mode users on a code path with broader
test coverage. The flip is a one-line server-startup change once
we want it global.

## Out-of-scope this snapshot

- batch=32 capture failure (input buffer alloc) — separate issue,
  not on Qwen3-4B-AWQ steady-state path.
- Async sampling (`VLLM_ASYNC_SAMPLING=1`) interaction — gated
  separately.
- Larger models / different shapes — tested only the standard
  Qwen3-4B-AWQ decode hot path.
