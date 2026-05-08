# 15.D-body Stage 3 (tensor cores) vs production AWQ-Marlin

Date: 2026-05-08 (commit 03e5896)
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Harness: criterion 10-sample median
group_size: 128

## Headline

Tensor-core (TC) tile-MMA path now beats production AWQ-Marlin
**2-5× across M ∈ {4, 8}** on Qwen3-4B AWQ shapes. M=1 ties (TC ≈ prod);
the small-M case isn't the TC sweet spot but no longer regresses
catastrophically as the software path did.

## Comparison (microbench data)

| Shape | M | prod µs | sw µs | **TC µs** | TC vs prod |
|---|---|---|---|---|---|
| down (K=11008 N=2560) | 4 | 317.6 | 284.4 | **149.2** | **2.13×** |
| down (K=11008 N=2560) | 8 | 558.4 | 557.3 | **174.1** | **3.21×** |
| mlp_up_old (K=4096 N=11008) | 1 | 145.5 | 114.1 | 146.1 | 1.00× |
| mlp_up_old (K=4096 N=11008) | 4 | 456.6 | 392.1 | **143.1** | **3.19×** |
| mlp_up_old (K=4096 N=11008) | 8 | 880.4 | 732.5 | **175.1** | **5.03×** |

Software-path baselines from
`docs/perf/bench-history/2026-05-08-15E-shape-sweep.json`.

## Implications for HYBRID gate

The Stage 15.E.3 gate `4 <= M <= 8 (excl. M=1)` was set assuming the
**software** path. With tensor cores the picture is now:

- `M=1`: TC and production tie. Keep on production (no benefit either
  way; tile_b VRAM cost not justified for M=1 alone).
- `M=4..8`: TC wins 2-5× over production. Strong opt-in case at default
  gate `[4, 8]`.
- `M ∈ {16, 32, 64+}`: untested in this microbench; m_tile fully
  saturated, expect TC to keep winning. Gate's upper bound stays at 8
  pending a follow-up sweep.

## Decisions

1. Keep `HYBRID_M_MIN = 4`, `HYBRID_M_MAX = 8` — proven range.
2. Update gate comment to reflect TC win (was 1.10-1.24× sw lift,
   now 2-5× vs production).
3. `VLLM_AWQ_HYBRID=1` opt-in stays — the +2.2 GiB tile_b VRAM is the
   trade-off; users with VRAM headroom opt in for the now-dramatic
   c=4 / c=8 decode lift.
4. Body.5 e2e bench under VLLM_AWQ_HYBRID=1 expected to confirm the
   ~2× lift on the AWQ-Marlin decode slice (= 0.91 of total) → e2e
   ≈ 1.7-1.9× at c=8.

Related: ADR 0016 (hybrid dispatch design); commit 03e5896 (body.3d
closing the production-shape contract on tensor cores).
