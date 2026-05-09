# CUDA Graph capture (Direction D) — investigation summary

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), 8 GiB, WSL2
Model: Qwen/Qwen3-4B-AWQ
Stack: candle 0.10.2 + cudarc 0.19.4 (resolved through workspace)

## Goal

Determine whether the historical CUDA Graph capture blocker
(memory note `perf_cuda_graph_capture`) is still real on the current
stack, and unblock if possible.

## Key findings

### 1. Stage 11 (non-default stream) is a public-API one-liner now

candle 0.10.2 exposes `Device::new_cuda_with_stream(ordinal)` →
`CudaContext::new_stream()` (non-default stream). The vendored fork
referenced in the old Stage 11 doc is no longer needed.

### 2. disable_event_tracking is fixed in cudarc 0.17+

The 0.16 version had a half-implemented fix: the flag was flipped
but `wait()`/`record()` paths still ran on already-allocated slices.
cudarc PR #436 (0.17.0) closed that. We're already on 0.19.4 via
candle 0.10.2.

Diagnostic confirms: alloc-only probe survives capture, and
`graphs_captured=5/6` at warmup with
`VLLM_DISABLE_EVENT_TRACKING=1`.

### 3. Production replay fails — different root cause

With capture working, the engine's first decode forward:
```
WARN CUDA graph decode failed; falling back to eager
     error=replay failed with CUDA error 2     # CUDA_ERROR_OUT_OF_MEMORY
... [×30 retries on each subsequent forward] ...
WARN CUDA graph decode failed; falling back to eager
     error=sync failed with CUDA error 700     # CUDA_ERROR_ILLEGAL_ADDRESS
ERROR eager decode fallback failed
     error=DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, ...)
```

Root cause: the captured graph's recorded operations reference
**specific device pointers** that were valid at capture time. On
replay, our forward path's per-layer scratch allocations
(`Tensor::cat`, `.contiguous()`, RoPE / RMSNorm / SwiGLU /
paged-attn intermediate buffers) land at *different* addresses, so
the graph reads from stale pointers → OOM-class corruption.

This is the same diagnosis the Stage 12 doc reached on the old
stack — `output_pool` covers only GEMV outputs; everything else
still uses ad-hoc `cuMemAlloc`. The CUDA Graph blocker is therefore
*architectural* (forward must use only pre-allocated buffers),
not a cudarc API issue.

## What landed today (infra cleanup)

- `crates/server/src/main.rs::create_cuda_device` helper — three
  call sites (`Device::new_cuda(...)` → `create_cuda_device(...)`),
  switching to a non-default stream so capture is not blocked at
  the very first call.
- `crates/core/src/engine/cuda_graph_runner.rs` — output buffer
  dtype now matches the warmup forward's actual output dtype (F32
  logits) instead of the model's compute dtype (BF16). The mismatch
  was masked before because event-tracking blocked capture upstream
  of the dtype check.

These are pure infra improvements: zero perf delta in eager mode,
and they make any future capture-activation work easier.

## What did NOT land

- `VLLM_DISABLE_EVENT_TRACKING=1` env gate — works to enable
  capture but breaks production at concurrent inference because
  *replay* hits the address-staleness issue described above. Not
  worth shipping until the underlying refactor lands.

## What's needed to actually close the 10-15% Graph win

Multi-week refactor (out of session scope):
1. Generalise `output_pool` to cover RMSNorm, RoPE, SwiGLU,
   PagedAttn, and any other per-layer intermediate that currently
   does `Tensor::zeros`.
2. Eliminate `Tensor::cat` and `.contiguous()` in the decode hot
   path (or route them through pool-allocated buffers).
3. Re-validate capture+replay with all per-layer buffers stable.

## Bench summary (3-mode A/B/C)

| Mode | c=1 | c=4 | c=8 |
|------|-----|-----|-----|
| baseline-eager (--enforce-eager) | 44.3 | 26.0 | 170.4 |
| graph-events-on (capture inactive — 905) | 44.4 | 26.3 | 170.7 |
| graph-no-events (capture active, replay fails) | 42.3 | (server crashed) | (none) |

`graph-events-on` matches eager because capture aborts at warmup
with 905 isolation, then engine falls back to eager — the warmup
overhead is small.

`graph-no-events` succeeds at warmup (5/6 captures) but every
replay fails, eventually corrupting GPU state.
