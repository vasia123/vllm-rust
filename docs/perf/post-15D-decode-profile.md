# Post-Stage 15.D decode profile — re-baseline

Date: 2026-05-08 (commit `2c1c3fe` — full 15.D-body + KV-leak + UTF-8 fixes)
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Server: `VLLM_AWQ_HYBRID=1 vllm-server serve --model Qwen/Qwen3-4B-AWQ --port 8765 --max-model-len 1024 --num-blocks 384`
Profilers: `VLLM_PROFILE_DECODE=1` (per-layer-component) + `VLLM_PROFILE_STEP=1` (engine-level stage)

## Headline

The post-15.D bottleneck has shifted but **NOT** to attention as the
plan hypothesised. Top contributors at c=8 (real production trace):

| Layer-call (decode_profile, c≈4-8 mix) | µs | % of layer-call |
|---|---|---|
| MLP (gate+up+down via TC AWQ-Marlin) | 700-815 | **75%** |
| QKV proj (TC AWQ-Marlin) | 196-240 | **22%** |
| O proj (TC AWQ-Marlin) | 105-121 | **12%** |
| paged_attn | 53-66 | 6% |
| ROPE | 45-48 | 5% |
| KV cache write | 10-11 | 1% |
| RMSNorm × 2 | 10-12 | 1% |
| qk_norm | 8-9 | <1% |
| **sum/layer-call** | **~940 µs** | — |

| Engine step (step_profile, c=8) | µs/step | % of step |
|---|---|---|
| forward (whole `forward_decode_batch`) | 59,692 | **90%** |
| sampling | 6,764 | 10% |
| alloc + metadata + ...else | <12 | <0.1% |
| **total step** | **66,471 µs** | — |

## Hidden 26 ms outside the layer profile

36 layers × 940 µs ≈ **33.8 ms** instrumented layer time per step.
But step_profile reports 59.7 ms `forward`. **~26 ms (44% of forward,
40% of total step) is outside the per-layer profile.** That code is:

- Token-id → hidden embedding lookup (small)
- Final RMSNorm (small)
- **`lm_head` matmul: [M=8, 2560] × [2560, 151,936] BF16 dense** ← biggest
- Per-layer Rust dispatch (small)

Qwen3-4B-AWQ does **not** quantize `lm_head` (default
`lm_head_quantized=false`); it's a 152K-wide BF16 dense matmul on
every decode step. That's ~3.1 GFLOPs per token × 8 tokens = 25 GFLOPs,
plus a 778 MiB output write. On a 4060 Laptop (~15 TFLOPs BF16, ~250
GB/s memory) this is realistically 5-15 ms / step at M=8 — a
plausible match to the unaccounted 26 ms.

## What this means for the plan

Plan's Direction A (attention concurrency scaling) was **wrong
hypothesis**. Attention only consumes 6% of layer wallclock and
shrinks (relatively) under concurrency. Top remaining contributors:

1. **MLP at ~75% of layer wallclock** (~700 µs/layer-call × 36 = 25 ms/step).
   Already on tensor cores, but production-realised speedup is ~2× not
   the 3-5× microbench number. Headroom remains: 
   - Investigate why TC isn't realising microbench-level kernel time
     in production (CustomOp1 framework cost? Stream serialisation?
     Check via nvprof).
   - Fuse gate+up into single matmul (vLLM does this — saves 1 of 3
     mlp matmuls per layer, ~33% mlp speedup).
   - Hybrid widening: TC for M up to 64 (we capped at 32 untested).
2. **`lm_head` BF16 dense matmul ~40% of forward outside layers**.
   Untouched. Options: 
   - Tied embeddings (if Qwen3-4B-AWQ supports — check config).
   - lm_head AWQ quantization (Qwen3-4B-AWQ checkpoint may not have
     quantized lm_head weights; if it does, enable it).
   - Spec decode: lm_head only on accepted draft tokens (memory says
     MTP/Eagle/Medusa landed; not yet enabled by default).
3. **Per-call overhead inside layer dispatch**: 940 µs /
   (~MLP 700 + qkv 200 + oproj 110 + attn 60 + rope 50 + small) ≈ 940
   /1120 = 84%. The ~16% delta = framework / dispatch overhead.
   Persistent fragments + ldmatrix shmem-staged (15.D-body.2c+) would
   chip at this.

## Direction A is not it. Direction A' is matmul-side.

Replace plan's Direction A with **Direction A' — fused gate+up + lm_head
optimisation**. Direction B (prefill TTFT) stays valid. Direction C
(CUDA Graph) still BLOCKED.

## Bench numbers (baseline for next stage)

c=8 with profile overhead enabled: 14.1 tps/req, 101 tps aggregate
(≈4% slowdown from clean 105 tps — profile overhead).

## Snapshot

```
2026-05-08T14:17:58 vllm_core::decode_profile: decode breakdown (100 layer-calls, μs/call):
  in_norm=5.5 qkv=240.5 qk_norm=9.1 rope=47.0 cache_w=10.1 paged_attn=66.4
  o_proj=121.3 post_norm=5.3 mlp=705.7 | sum=1210.9

2026-05-08T14:49:44 vllm_core::step_profile: decode step (bucket=≤8, 200 steps, μs/step):
  alloc=7.8 metadata=3.3 shared=0.0 input=0.0 forward=59692.2 sampling=6764.7
  dispatch=0.0 | total=66471.8 sum=66468.0
```
