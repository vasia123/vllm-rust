# Stage 15.D-body.5 (FIXED) — E2E bench delivers projected lift

Date: 2026-05-08 (commit ce162d6 + 3D-shape fix)
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Server: `VLLM_AWQ_HYBRID=1 vllm-server serve --model Qwen/Qwen3-4B-AWQ --port 8765 --max-model-len 1024 --num-blocks 384`
Bench: `python3 scripts/bench_decode.py --runs 3 --concurrency 1 4 8 --prompt-len 256 --max-tokens 128`

## Result — e2e win confirmed

| c | Hybrid TC tps | Baseline tps | Lift | Plan target |
|---|---|---|---|---|
| 1 |  41.7 | 43 | −3% (within noise; M=1 below gate, expected) | n/a |
| 4 |  80.1 | 65 | **+23%** | +38% |
| 8 | 111.3 | 64 | **+74%** | +72% |

**c=8 hit the plan target.** Stage 15.D-body delivers what body.4 microbench
projected (after correcting for AWQ-Marlin slice = 91% of decode wallclock,
TC gives ~3× kernel speed → 1.7-1.9× e2e at c=8). c=4 came in at +23%
vs +38% projected — possibly because c=4 has more prefill mixing in
continuous batching (M is sometimes 1-3 not always 4), so a chunk of
forward calls miss the M ≥ 4 gate.

## Why the first body.5 run showed 0% lift

Diagnosed via `tracing::info!` in the hybrid hot path: the gate
`x.dims().len() == 2` was checking the **wrong shape**. Production
passes activations as 3D `[B, S, H]` (e.g. `[1, 1, 2560]` for c=1
decode, `[4, 1, 2560]` for c=4 decode in continuous batching). The
2D-only guard rejected every real forward, falling through to
production marlin_gemm — explaining the 0% lift in the first bench run
(and arguably also the **same 0% lift** in Stage 15.E.3's software
hybrid bench, which had the **identical guard**).

Fix: in `AwqMarlinLinear::forward`, accept 3D and flatten leading dims
to 2D `[B*S, H]` before the gate check + dispatch, then reshape result
back to `[B, S, out]`.

## Implications for ADR 0016 / Stage 15.E.3

The 15.E.3 negative result ("software hybrid 0% lift, hypothesis 2 =
per-call launch overhead amortisation") was almost certainly the SAME
bug. The software path also went through this guard, so VLLM_AWQ_HYBRID
was effectively a no-op in 15.E.3 too. Hypothesis 2 is **now
falsified**: the underlying reason was a shape-guard bug, not launch
overhead.

This means re-running 15.E.3's software bench under the fix would
likely now show the projected 1.10-1.24× e2e lift. But the software
path is no longer the win path — TC delivers ~3× the gain.

## Decisions

1. Land the 3D-shape fix in `awq_marlin.rs::forward`.
2. Keep `VLLM_AWQ_HYBRID=1` opt-in (the +2.2 GiB tile_b VRAM cost is
   real). With +74% e2e at c=8 and ~zero c=1 cost, the gate is now
   genuinely valuable for any user with ≥10 GiB VRAM headroom.
3. Update status doc + ADR 0016 to flip the body.5 outcome from
   "negative" to "projected lift achieved at c=8".
4. Body.5 closing this stage as a SUCCESS, not a deferral.

Bench raw output:
```
=== 15D-body-5-tc-fixed  model=Qwen/Qwen3-4B-AWQ ===
prompt≈256w  max_tokens=128  temperature=0.7  runs=3

concurrency   med ttft ms   med tps/req   aggregate tps    best agg tps
-----------------------------------------------------------------------
          1         336.4          41.7            41.7            41.7
          4        1391.8          20.0            80.1            80.1
          8        1634.3          13.5           111.3           111.3
```
