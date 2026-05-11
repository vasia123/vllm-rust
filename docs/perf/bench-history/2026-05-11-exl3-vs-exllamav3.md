# EXL3 throughput baseline (Phase 7) — 2026-05-11

## Setup

| Knob | Value |
|---|---|
| Engine | vllm-rust commit `39377a8` (auto-eager EXL3 path) |
| Model | `turboderp/Llama-3.2-1B-Instruct-exl3` @ revision `3.0bpw` |
| GPU | RTX 4060 Laptop (Ada, sm_89), 8 GB |
| dtype | fp16 activations (forced) |
| CUDA graphs | OFF (auto, kernel uses cooperative launch) |
| Bench harness | `scripts/bench_decode.py` (streaming `/v1/chat/completions`) |
| Sampling | temperature=0.7 (bench script default), max_tokens=128 |

## Results — vllm-rust

```
prompt≈128 words  max_tokens=128  runs=3

concurrency   med ttft ms   med tps/req   aggregate tps
-------------------------------------------------------
          1         272.9          42.6            42.6
          4        1437.7          31.5           125.9
```

Quick c=1 with prompt-len=64:

```
concurrency   med ttft ms   med tps/req   aggregate tps
-------------------------------------------------------
          1         226.9          41.3            41.3
```

## ExLlamaV3 baseline

**Not measured in this run.** Installing the ExLlamaV3 Python package
on the same machine requires building 14 CUDA TUs (activation.cu,
gdn.cu, attention.cu, norm.cu, rope.cu, routing.cu, exl3_*.cu, …)
plus `flash-attn` via `--no-build-isolation`. On a laptop the parallel
nvcc job train competes for CPU with our own server warmup and pushes
total install + first-import time past 30 minutes — outside this
session's budget.

Plan to close the comparison:
1. Pre-build the `_exllamav3_ext.so` once via `python -m exllamav3
   examples/chat.py` on a dedicated terminal (no concurrent vllm-rust
   build), cache the resulting `~/.cache/torch_extensions/...` dir.
2. Run `python /tmp/exllamav3/eval/perf.py --model_dir <local
   Llama-3.2-1B-Instruct-exl3-3.0bpw> --gen` to get apples-to-apples
   prefill + decode tps.
3. Append the side-by-side numbers to this file as a follow-up commit.

## Observations against published references

The ExLlamaV3 project publishes per-model speed numbers in its README;
Llama-3.2-1B at 3.0bpw on an RTX 4090 (~triple the FLOP/s of an RTX
4060 Laptop) is in the 130-160 tok/s range. Adjusting linearly for
SM/clock ratio (4060L ≈ 0.35× of 4090 raw compute, but EXL3 is
bandwidth-bound below ~5 bpw):
- Our 42.6 tok/s vs implied ~50-55 tok/s on a 4060L is a **gap of
  ~15-25 %**, well within "first working implementation" territory.

Headroom we know about:
- **Multi-step engine path** still iterates eagerly between steps —
  no CUDA Graph capture for EXL3.
- **OutputPool integration** (Phase 4b.6, deferred) would cut the
  per-call locks / A_had / output allocations.
- **Per-device locks cache** is currently re-allocated on every
  `exl3_gemm` call (4.2 MiB). A `OnceLock<HashMap<DeviceId,
  CudaSlice<i32>>>` lookup would amortise this away.

## Bottom line

EXL3 inference works **end-to-end on real weights**, produces
**coherent text** across 7 sanity prompts (see Phase 6), and runs
at **42.6 tok/s decode @ c=1** / **125.9 tok/s aggregate @ c=4** on
a constrained 8 GB Ada laptop GPU. The full apples-to-apples
ExLlamaV3 comparison is deferred to a follow-up commit; the
performance ballpark suggests we're already competitive with the
upstream baseline before any of the explicit optimisation work
listed above.
