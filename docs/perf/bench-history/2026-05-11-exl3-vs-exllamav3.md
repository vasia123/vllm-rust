# EXL3 throughput baseline (Phase 7) — 2026-05-11

## Setup

| Knob | Value |
|---|---|
| Engine A | vllm-rust commit `50a7a71` (auto-eager EXL3 path) |
| Engine B | ExLlamaV3 0.0.34 (github master clone + wheel-built .so) |
| Model | `turboderp/Llama-3.2-1B-Instruct-exl3` @ revision `3.0bpw` |
| GPU | RTX 4060 Laptop (Ada, sm_89), 8 GB |
| Activation dtype | fp16 (vllm-rust auto-overrides BF16→F16 for EXL3) |
| CUDA graphs | OFF on vllm-rust (cooperative-launch incompat); n/a for ExLlamaV3 |
| Harness — vllm-rust | `scripts/bench_decode.py` against `/v1/chat/completions` |
| Harness — ExLlamaV3 | `scripts/bench_exllamav3.py` (low-level `model.prefill` + `model.forward` loop, greedy sampling) |
| Sampling | temperature 0.7 (vllm-rust default), greedy (ExLlamaV3) |
| Prompt / max-tokens | 128 words ≈ 128 tokens / 128 new tokens, 3 runs |

## Side-by-side results

```
                       c=1 decode tps   c=1 ttft ms
---------------------- --------------- -------------
vllm-rust                       42.6           272.9
ExLlamaV3                      107.9            40.0   (median of 3 runs)
ratio                          2.53×           0.15×
```

```
                       c=4 aggregate tps   c=4 per-req tps
---------------------- ------------------ -----------------
vllm-rust                         125.9              31.5
ExLlamaV3                        n/a* (low-level harness is single-stream)
```

\* ExLlamaV3's high-level `Generator` exposes concurrency but tripped a
`sanity_check` assertion when bench-driven without `flash-attn`
installed. We have flash-attn stubbed (no pre-built wheel for our
torch 2.11+cu130 combo on this box), so we measure the low-level
single-stream decode path only. The single-stream comparison is the
most honest one — it isolates the per-token kernel-dispatch cost
without scheduling or batching effects.

## What the gap is

- vllm-rust 42.6 tok/s vs ExLlamaV3 107.9 tok/s = **2.53× gap**.

That gap is real and concentrated in three places we already
identified during implementation:

1. **No OutputPool / per-device locks cache.** Every `exl3_gemm` call
   allocates an output `[M, N]` fp16 buffer, an `A_had` scratch buffer
   `[M, K]`, and a fresh `int32[1024*1024 + 2*1024]` locks buffer
   (≈4 MiB). At decode time we issue these allocations per attention
   per layer — 7 lin × 16 layers = 112 cudaMallocAsync per token.
2. **No CUDA Graph capture** of the EXL3 path. The cooperative GEMM
   kernel can't be captured (runtime tile-scheduling barrier), so
   we lose the launch-overhead amortisation that the rest of our
   models get.
3. **No K=3 GEMV fast path** (upstream ships GEMV only for K=4 —
   verified in our `4b.4` commit). Llama-3.2-1B-EXL3-3.0bpw uses
   bpw=3, so our M=1 decode goes through the cooperative GEMM kernel
   at GEMM cost — even though ExLlamaV3's reference does the same.
   (So this one likely doesn't explain the 2.5× gap on its own;
   items 1+2 are the bigger drivers.)

## What's already correct

- 31/31 EXL3 lib tests pass (24 CPU + 5 GPU correctness +
  2 GPU smoke).
- 7/7 sanity prompts produce coherent + factually correct output
  through the full stack (Phase 6).
- The kernel math itself matches upstream:
  `exl3_gemm_matches_hadamard_then_matmul` validates against
  `had_r_128 ∘ matmul ∘ reconstruct` end-to-end, within fp16 noise.

So this is a pure engineering-throughput gap, not a correctness gap.

## Reproduction

```bash
# vllm-rust side (in another terminal, on cleanup):
./target/debug/vllm-server serve \
  --model "turboderp/Llama-3.2-1B-Instruct-exl3" \
  --revision "3.0bpw" --download-dir "./model_cache" \
  --max-model-len 1024 --gpu-memory-utilization 0.5 \
  --max-requests 4 --num-blocks 128 --port 18080

.venv-tmp/bin/python scripts/bench_decode.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model "turboderp/Llama-3.2-1B-Instruct-exl3" \
  --concurrency 1 4 --prompt-len 128 --max-tokens 128 --runs 3

# ExLlamaV3 side:
MODEL_DIR=$(ls -d ./model_cache/models--turboderp--Llama-3.2-1B-Instruct-exl3/snapshots/*/)
.venv-tmp/bin/python scripts/bench_exllamav3.py \
  --model-dir "$MODEL_DIR" \
  --prompt-len 128 --max-tokens 128 --runs 3
```

### Notes on setup

The 0.0.33 wheel of ExLlamaV3 is missing
`exllamav3/modules/attention/*.py` — the bench script prepends the
github source clone (`model_cache/exllamav3_src`) to `sys.path` and
pre-loads the wheel's compiled `.so` so `torch.utils.cpp_extension.load`
doesn't re-deadlock on its lock file. A no-op `flash_attn` shim sits
in `.venv-tmp/lib/python3.12/site-packages/flash_attn/` so the
attention dispatcher's import line succeeds and falls through to the
torch-sdpa path.

## Bottom line

EXL3 inference in vllm-rust **works end-to-end and is correct**, at
~40 % of ExLlamaV3's single-stream throughput on this hardware. The
gap is dominated by allocator pressure and missing CUDA Graph capture
on the cooperative-launch path — both have well-defined follow-up
work items (OutputPool integration; per-device locks cache; pool-
backed `A_had` scratch). At c=4 vllm-rust's batching pulls the
aggregate above per-stream ExLlamaV3 (125.9 vs 107.9), but that's a
different axis of comparison.
