# Phase 1 — gate+up fusion (concat weights, single kernel call) — NEGATIVE

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), WSL2

## Hypothesis

The MLP forward calls 3 quantized linears (gate, up, down). At c=8
each kernel takes ~540 μs (production decode_profile) vs ~91 μs
(microbench, isolated) — a 6× discrepancy attributed to L2/HBM
cache contention with the surrounding RMSNorm + attention block.

Phase 1 conjecture: even without addressing cache contention, just
launching ONE fused gate+up kernel instead of two should save
launch overhead. Estimated 5–10 μs/launch × 36 layers × 4 multi-step
= ~1.4 ms/step ≈ 16 % step time at c=8.

## Implementation

- `QuantizedWeightLoader::try_load_fused_gate_up(gate_prefix, up_prefix, in, intermediate) -> Option<Box<dyn QuantizedLinear>>`
  default `Ok(None)`.
- AwqWeightLoader override: load gate+up `qweight`/`scales`/`qzeros`
  separately, concat each along output axis, build a single
  `AwqMarlinLinear` with `out_features = 2 * intermediate_size`. Wrap
  raw tensor pairs in inner blocks so originals drop before next
  pair loads.
- `QuantizedSwiGluMlp::new`: try fused load first; on `Some`, store
  in `fused_gate_up` and skip individual loads. Forward splits the
  combined output via `narrow` and runs SwiGLU.
- Env-gated `VLLM_AWQ_FUSE_GATE_UP=1`.

## Result: 0 % e2e lift (within bench noise)

Same-host bench `python3 scripts/bench_decode.py --concurrency 1 4 8
--prompt-len 256 --max-tokens 128 --runs 3`,
`VLLM_AWQ_HYBRID=1 ./target/release/vllm-server serve --enforce-eager
--max-model-len 1024 --num-blocks 384`:

| c | baseline | + VLLM_AWQ_FUSE_GATE_UP=1 |
|---|---|---|
| 1 | 45.9 | 42.9 (−7 %) |
| 4 | 85.8 | 87.2 (+2 %) |
| 8 | 118.8 | 116.3 (−2 %) |

All within run-to-run variance.

## Why it didn't help

1. **Launch overhead is not the bottleneck.** Per-matmul kernel time
   is dominated by HBM/L2 traffic for weights (~540 μs of which 90 %
   is bandwidth-bound). Saving 5-10 μs on each of 36 launches × 4
   multi-step is a real number on paper, but appears to be hidden
   inside other queue overlap on this host.
2. **Fused matmul kernel time = 2× separate** — same total work
   (2N output cols vs 2 × N), same grid size (just different shape).
   No L1/L2 reuse benefit because each block still loads its own
   16-col slice of weights.
3. The microbench analysis (
   `2026-05-09-awq-marlin-mlp-bandwidth-finding.md`) was correct:
   the gap is contention, not launch overhead.

## Decision: revert

Phase 1 plumbing did not deliver measurable gain. Reverted
`weight_loader.rs` and `qwen3_quantized.rs` to pre-fusion state.
Findings preserved here for the future Phase 2 attempt.

## What Phase 2 would need to attempt

The actual bandwidth-bound bottleneck requires kernel-level changes,
not load-time concatenation:

1. **Multi-warp blocks sharing weight tiles via shared memory.**
   Currently 1 warp per block of 32 threads. With 4 warps per block
   (128 threads) cooperatively loading a `[16, 64]` weight tile into
   smem, all 4 warps' 16-col output tiles share the global LDG cost
   = 4× reduction in HBM traffic per output cell. Real bandwidth
   relief.

2. **`cp.async` streaming for weight tiles**: overlap K-iter weight
   load with previous iter's mma compute. sm_80+ feature; fits sm_89.

These require ~1-2 days of careful kernel work each + a microbench
proving the per-matmul gain before integrating. Defer to a session
where the kernel changes can be the primary focus.

## Next direction

Three honest paths after this:
1. Accept current 118 tps c=8 (96 % of post-15.D snapshot 124 tps;
   sweet spot until kernel work is funded).
2. Phase 2 cooperative-warp kernel rewrite (1-2 days).
3. Pivot to lm_head — the 27 ms slot is the next-largest single
   piece of forward time and is a single matmul (different
   characteristics; potentially easier wins via cuBLAS routing or
   pre-transposed weight).
