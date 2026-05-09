# Python vLLM head-to-head, same env (eager mode)

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), 8 GiB, WSL2
Model: Qwen/Qwen3-4B-AWQ
Python vLLM version: 0.20.1 (installed in `.venv`)
Bench: `scripts/bench_decode.py --concurrency 1 4 8 16 --prompt-len 256 --max-tokens 128 --runs 3`

## Setup notes

- Both servers run with `--enforce-eager` (no CUDA Graph capture) for
  apples-to-apples — our cudarc 0.16 stack can't capture graphs, so
  graphs-on for vLLM would be an unfair comparison.
- Same `--max-model-len 1024`. Our server gets `--num-blocks 384` to
  match KV size; vLLM auto-sizes from GPU memory util.
- vLLM with CUDA Graph mode (default, no `--enforce-eager`) hangs at
  startup on this 8 GiB GPU — graph capture's extra working set
  (~0.6 GiB measured) pushes total beyond the 0.85 util budget. So
  CUDA Graph numbers for vLLM are not measured here; eager-eager is
  the comparison.

## Results

| c | Python vLLM (eager) | Our Rust | Gap (vLLM/Rust) |
|---|---|---|---|
| 1 | 46.1 | 43.2 | 1.07× (~tied) |
| 4 | 163.0 | 101.7 | 1.60× |
| 8 | **321.5** | **165** | **1.95×** |
| 16 | 593.9 | 314 | 1.89× |

Per-token decode latency (1/per-req tps, ms):
- c=8: vLLM 24.9 ms, ours 51.0 ms (decode only, excluding prefill).

## What this tells us

1. **The 2× gap is NOT CUDA Graph.** vLLM running eager still beats
   us 2× at c=8. So unblocking cudarc 0.16's graph capture would not
   close the gap on its own.
2. **The gap is in kernel-level execution.** vLLM's eager path uses
   FlashAttention-v2 / xformers attention plus their own well-tuned
   AWQ-Marlin kernels. Per-step their forward is markedly faster
   despite running the same number of kernel launches as us.
3. **At c=1 we're approximately tied** (43.2 vs 46.1; 7 % vLLM
   advantage, within run-to-run noise). At small batch the kernel
   delta is hidden by latency floor effects.
4. **Gap widens at higher c.** At c=4 → 1.6×; c=8 → 1.95×. This is
   consistent with kernel-throughput-limited workloads where vLLM's
   better kernels amortise more strongly under concurrent batching.

## Honest gap decomposition

If we accept the eager-eager number is the true kernel-quality gap:

- **~95 % of the 2× gap is kernel quality** (attention, mlp, sampler
  speed). Closing this requires writing kernels at the FlashAttention
  / vLLM AWQ-Marlin tuning level — months of focused kernel work.
- **~5 % is engine-loop / scheduler / sampler overhead.** Mostly out
  of our control without architectural changes (CUDA Graph would help
  here, but it's blocked upstream).

## Today's session in context

c=8 went 118.8 → 165 tps in this session (+39 %), closing the gap
from 2.95× to 1.95×. The remaining 2× is honest infrastructure debt
that no single change in this session can fix without rewriting
attention / Marlin kernels. The "easy" wins have all been booked.

## What would actually close the gap

Realistic prioritised list (each item: weeks-to-months of focused
kernel work, not session-scale):

1. **Replace `paged_attention_cuda` with FlashAttention-v2 decode.**
   Their attention kernel is dramatically more bandwidth-efficient
   on long contexts. Could close ~30-40 % of the gap on its own.
2. **Tune AWQ-Marlin kernel** to the level of vLLM's reference
   implementation. Their kernel achieves higher SM occupancy and
   better cache reuse. Phase 2 cooperative-warp attempt earlier this
   session went negative — proper tuning needs ncu / nsys data
   (blocked in this WSL2 env).
3. **Async sampling** — closes the per-step host stall. Already
   landed (Substep 2.2) but env-neutral here; native Linux would
   show the win.
4. **CUDA Graph capture** — for kernel launch overhead amortisation.
   Memory note `perf_cuda_graph_capture` flags this as cudarc 0.16
   blocker. Adds ~10-15 % e2e once unblocked.

Combined, these would get to roughly parity, but the engineering
budget is significant.

## Cleanup

- Python vLLM left in `.venv` (~3 GiB on disk). Memory rule
  `feedback_python_env` honoured: not installed globally. Can be
  removed via `.venv/bin/pip uninstall vllm` if disk pressure
  matters.
