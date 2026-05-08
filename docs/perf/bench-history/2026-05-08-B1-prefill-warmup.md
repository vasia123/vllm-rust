# Stage B.1 — prefill JIT warmup TTFT result

Date: 2026-05-08 (commit `67de005`)
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Model: Qwen/Qwen3-4B-AWQ
Server: `vllm-server serve --port 8765 --max-model-len 1024 --num-blocks 384`
Bench: 5 sequential `/v1/chat/completions max_tokens=1` with prompt ≈ 256 words

## Result

| req | no warmup (`time` curl) | with `VLLM_PREFILL_WARMUP_LENS=128,256` | Δ |
|---|---|---|---|
| 1 (cold) | 558 ms | 525 ms |  −6% |
| 2 (transition) | 599 ms | 318 ms | **−47%** |
| 3 | 306 ms | 314 ms | ~0% (noise) |
| 4 | 310 ms | 324 ms | ~0% |
| 5 | (n/a) | 337 ms | — |

## Analysis

Cold req 1 wins only 6% because not all of its latency is in
prefill-attention JIT. The real first request also pays for: model
embedding lookup (cold), tokenizer setup, candle-side compute graph
caching for the specific prompt shape, and lm_head (BF16 dense) on a
cold kernel cache.

**The big win is at req 2** (47% reduction). Without warmup, req 2
still suffers JIT cost on shapes req 1 didn't quite hit (slightly
different K-tile boundary, different N if any). With warmup, those
adjacent shapes are pre-compiled, so req 2 lands directly on the warm
path. After req 2 both modes converge to ~310-330 ms warm steady-state.

Net UX impact: on a "burst" session (chat opens → user sends first 2-3
prompts), warmup cuts user-visible TTFT for prompts 2-3 by ~50%.
First-prompt latency (cold-cold) is dominated by other cold paths
that warmup doesn't touch.

## What this DOESN'T fix (known)

- Cold req 1 model-load + tokenizer state still dominates that single
  point.
- Steady-state warm (req 3+) is the actual decode-bound number; no
  change there since attention/matmul kernels were already warm.
- The 7× TTFT gap to Python vLLM (47 ms warm) is mostly in things
  warmup doesn't address — likely a fundamentally different
  attention kernel choice (vLLM uses flash_attn for prefill, we use
  paged-attention even for the parallel-causal-attn case). That's
  Direction B.2/B.3 in the plan.

## Decisions

1. Land B.1 (commit `67de005`) — env-gated opt-in, no default change.
2. Document the partial win honestly; do NOT claim 50% TTFT reduction
   without qualifier (steady-state warm path unchanged).
3. **Deprioritise further B-direction work** — even the *full* TTFT
   gap is bounded above by ~300 ms warm steady-state. Closing all
   300 ms only helps interactive single-stream chat, marginal for
   throughput-bound workloads. Direction A' (matmul-side, lm_head)
   has higher leverage on aggregate tps.
4. Defer B.2 / B.3 (prefill flash_attn integration) until A' has been
   exhausted.
