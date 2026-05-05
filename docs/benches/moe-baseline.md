# MoE Routing Baseline

This file records the **per-PR regression gate** for Phase 7 (MoE
consolidation). Every MoE migration PR must run
`cargo bench --bench moe_routing_bench` and verify the throughput
delta on `topk_router_softmax`, `topk_router_sigmoid`, and
`moe_layer_naive` is **≤ 2% slower** than the most recent baseline
recorded here.

## How to update

When a Phase 7 migration intentionally changes baseline numbers
(for example because it adds a new routing path consumed by the
bench), update this file in the **same PR** with the new numbers
and a one-line reason.

For routine migrations that don't touch shared MoE primitives, this
file does not need to change — the gate is "no regression vs. last
recorded baseline".

## Latest baseline

`cargo bench --bench moe_routing_bench` on the development host;
HTML report under `target/criterion/`. Numbers below are sampled at
the time of recording and are the **target throughput floor** for
subsequent PRs.

| Bench | tokens | sample size | min iter time | notes |
|-------|-------:|------------:|--------------:|-------|
| `topk_router_softmax/tokens/16`   | 16   | 100 | record on first run | softmax + top-k |
| `topk_router_softmax/tokens/64`   | 64   | 100 | record on first run | |
| `topk_router_softmax/tokens/256`  | 256  | 100 | record on first run | |
| `topk_router_softmax/tokens/1024` | 1024 | 100 | record on first run | |
| `topk_router_sigmoid/tokens/16`   | 16   | 100 | record on first run | sigmoid scoring (GLM4-MoE/MiniMax-M2) |
| `topk_router_sigmoid/tokens/64`   | 64   | 100 | record on first run | |
| `topk_router_sigmoid/tokens/256`  | 256  | 100 | record on first run | |
| `topk_router_sigmoid/tokens/1024` | 1024 | 100 | record on first run | |
| `moe_layer_naive/tokens/16`       | 16   | 100 | record on first run | per-token loop fallback |
| `moe_layer_naive/tokens/64`       | 64   | 100 | record on first run | |
| `moe_layer_naive/tokens/256`      | 256  | 100 | record on first run | |

The first numeric baseline is recorded by the developer running the
first migration PR (Phase 7.1 cohort A) on their machine, against the
bench skeleton that landed in this commit.

## What the bench measures

Mixtral-8x7B-shaped config scaled down for CPU runtime feasibility:

- `HIDDEN = 1024`
- `INTERMEDIATE = 3584`
- `NUM_EXPERTS = 8`
- `TOP_K = 2`

Production ratios match (intermediate ≈ 3.5 × hidden, ≈ 25%
activation density). CPU-only — the gate is **relative** delta
against the previous baseline on the same machine, not absolute
GPU throughput.

## Why this gate, not golden tests

Phase 7 migrations do not preserve bit-exact output: top-k
tie-breaking differs across implementations (Vec sort vs. tensor
arg_sort vs. CUDA `topk_softmax`), and the order of expert-output
summation changes when going from per-token to batched dispatch.
Differences are ≤ 1 ULP in production but break naive equality
asserts.

The throughput gate captures the practical contract: migrating
should make things faster (vectorized > scalar), or at worst
neutral. A 2% regression is the cliff at which we revisit.
