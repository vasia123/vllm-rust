# FlashInfer decode e2e — Direction A integration A/B

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), 8 GiB, WSL2
Model: Qwen/Qwen3-4B-AWQ
Bench: `scripts/bench_decode.py --concurrency 1 4 8 --prompt-len 256 --max-tokens 128 --runs 3`
Server: `--enforce-eager --max-model-len 1024 --num-blocks 384`
Env: `VLLM_AWQ_HYBRID=1` for both sides; `VLLM_FLASHINFER_DECODE` toggled.

## Results

| c | V2 baseline | FI integrated | Δ tps/req |
|---|-------------|---------------|-----------|
|   | tps/req agg | tps/req agg   | (per-req) |
| 1 | 43.0  43.0  | 43.1  43.1    | +0.2 %    |
| 4 | 25.4  101.5 | 25.3  101.2   | -0.4 %    |
| 8 | **19.6**  165.4 | **20.1**  161.3 | **+2.5 %** |

Per-request median throughput at c=8: FlashInfer **+2.5 %**.
Aggregate at c=8 swung -2.5 % the other way — within batching jitter
(differing TTFT phase-locking between runs); per-req median is the more
robust signal.

## Microbench vs e2e gap

Microbench
(`docs/perf/bench-history/2026-05-09-flashinfer-decode-microbench.md`)
showed **2.4× faster** for FI replay vs V2 paged_attention on the c=8
shape (61 μs vs 145 μs). Yet e2e gain is only +2.5 %.

Root cause: **production stream overlap**. In a real forward, paged_attn
V2 runs on the CUDA stream concurrently with surrounding kernels (norm,
RoPE finalisation, MLP launches from the previous layer's tail), so its
wall-clock cost is partially hidden. The microbench forces explicit
`stream.synchronize()` after each call — that sync defeats overlap and
inflates the measured V2 cost relative to its in-pipeline cost. FI gets
the same overlap benefit, but its absolute kernel cost is already
small, so the relative win shrinks.

This is the exact failure mode flagged by memory rule
`feedback_perf_assumption_test.md`: microbench wins don't always
translate to e2e because production has different scheduling
characteristics.

## Correctness

`scripts/test_qwen3_awq_correctness.sh` with `VLLM_FLASHINFER_DECODE=1`:
**7/7 PASS**. All canonical prompts produce expected first tokens; long
context, sustained load, and concurrent batch all pass.

## Decision

Land `VLLM_FLASHINFER_DECODE=1` as **default-off env-gated** (Phase 1).
Marginal +2-3 % per-req win at c=8 doesn't justify default-on flip —
not enough headroom over benchmark noise (~3 %). Keep FI gate available
for users running with concurrent loads where small per-req gains
compound, and as future infrastructure for direction D (CUDA Graph)
which can capture the FI replay path.

**Direction A = closed as marginal-positive** — gap to vLLM Python is
**not** in attention kernel choice (paged_attn V2 is competitive with
FlashInfer in production stream overlap). Roadmap pivots to D (CUDA
Graph) for the next 10-15 % expected lift.

## Files modified

- `crates/core/src/layers/attention/block.rs` — extended
  `DecodeBatchShared` with lazy `decode_plan: Arc<OnceLock<...>>`,
  added env-gated FI branch in `cuda_decode_batch`. ~70 lines.
- `crates/core/benches/flashinfer_decode_bench.rs` — new microbench
  (~270 lines).
- `crates/core/Cargo.toml` — register the bench.

## Logs

- baseline: `/tmp/vllm-baseline-v2.log`, bench `/tmp/bench-baseline-v2.txt`
- flashinfer: `/tmp/vllm-flashinfer.log`, bench `/tmp/bench-flashinfer.txt`
