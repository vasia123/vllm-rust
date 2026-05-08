# nsys profile finding — DtoH stream-drain dominates 45% of CUDA API time

Date: 2026-05-08 (after Windows-side `RmProfilingAdminOnly=0` enabling
allowed nsys to capture properly in WSL2)
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Capture: `nsys profile -o /tmp/nsys-r4 -t cuda --duration=120 --delay=20
--sample=none`, 4× `bench_decode --concurrency 8 --max-tokens 64
--prompt-len 256` runs during the 120s capture window.

## Headline result

CUDA API time breakdown (90s capture, c=8 production load):

| Time % | Total (s) | Calls | Avg | API |
|---|---|---|---|---|
| **45%** | 2.23 | 54 | **41 ms** | `cuMemcpyDtoHAsync_v2` |
| 31% | 1.56 | 864 | 1.8 ms | `cudaMemcpyAsync` |
| 5% | 0.28 | 288 | 0.96 ms | `cudaStreamSynchronize` |
| 5% | 0.25 | 40,650 | 6.2 µs | `cuLaunchKernel` |
| 4% | 0.21 | 54,172 | 3.8 µs | `cuMemAllocAsync` |
| 2% | 0.11 | 18,918 | 5.7 µs | `cuMemsetD8Async` |
| 1% | 0.06 | 9,776 | 5.7 µs | `cuMemcpyHtoDAsync_v2` |

DtoH durations are tightly clustered at 13.6 ms minimum, 54 ms maximum
— **uniformly slow**. 32-byte sampler-result transfers are taking 13+
ms each. PCIe DMA bandwidth at this size is 1000× faster, so the time
isn't in the actual transfer.

## Root cause: sync wait, not DMA

`cuMemcpyDtoHAsync_v2` is "async" but the calling Rust code is
`Tensor::to_vec1()` which **blocks until the copy completes**. The
copy itself queues to the same CUDA stream as 36 layers of forward
work + lm_head + sampler kernel. The host must wait for the stream to
drain before the DtoH executes. **The 13-54 ms is GPU pipeline drain
time** — the duration of the c=8 forward pass leading up to the
sampler.

This explains the 5-7× lm_head slowdown observed in `decode_profile`:
the `lm_head_evt` slot times the lm_head matmul, but **measurement
points around the matmul are entangled with the sampler's blocking
to_vec1**. Even with the `time_sync` wrapper that calls
`synchronize()` before AND after the matmul, the synchronize() itself
takes time proportional to pending work — confounding the measurement.

The lm_head MATMUL is genuinely 3 ms (microbench confirmed). The 16-24
ms reading was **time spent in the synchronize call itself**, not
the kernel.

## Sites doing DtoH per forward (hot paths)

`crates/core/src/sampling/gpu.rs`:
- line 557: `gpu_argmax` result `to_vec1()` — 8 u32 = 32 B
- line 586: `gpu_top_k_top_p_sample` result `to_vec1()` — 8 u32 = 32 B
- line 590: argmax fallback `to_vec1()` — 8 u32 = 32 B (only on has_greedy)

`crates/core/src/layers/attention/flashinfer/wrapper.rs`:
- lines 506, 676, 888, 893: `kv_indptr.to_vec1::<u32>()` for plan()
  pre-conditions. Tiny arrays but per-layer (36×) per forward.

## Fix candidates

1. **Pinned memory + persistent host buffer for sampler result.** Avoid
   per-call allocator overhead. Marginal — bottleneck is sync wait,
   not transfer time itself.
2. **Async sampler result handling.** Don't block on `to_vec1()` — let
   GPU keep working on next forward while host receives the DtoH
   asynchronously. Multi-day refactor of the sampler→engine token
   delivery pipeline.
3. **Move FlashInfer plan() index uploads to a one-shot init**
   (currently per-layer DtoH→HtoD round-trip). Already done at one
   point per memory; verify it didn't regress.
4. **Single-pass sampler that delivers the token directly into the
   next forward's input_ids buffer on-device** — eliminate DtoH for
   sampler entirely. Hardest but biggest win.

## Estimated lift if DtoH eliminated

DtoH = 45% of CUDA API time = ~30% of step wallclock (since CUDA API
calls overlap with kernel work; not all 45% is on critical path). A
realistic e2e lift if fully eliminated: **+30-40% c=8 throughput**
(105 → 140-150 tps).

Combined with full-stack work (FlashInfer plan, lm_head alloc reuse),
could approach Python vLLM's 351 tps (currently 3.35× ahead). 

## Next concrete step

Profile with `decode_profile::time` BUT measure the SAMPLER step
separately to confirm the to_vec1() block is the dominant time.
Currently sampler is part of `step_profile::SAMPLING` slot which
showed 6.8 ms. But our nsys data shows DtoH takes much more — the 6.8
ms sampling slot probably also undercounts due to event-vs-host
timing mismatch.

Add a `synchronize() + Instant` timer around the sampler call to get
ground truth. If ground truth matches nsys (~30 ms wallclock per
sampler call at c=8), confirms the block — and step_profile's SAMPLING
slot was measuring only the kernel queue time, not the host stall.
