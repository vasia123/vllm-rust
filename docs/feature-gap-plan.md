# Feature Gap Plan: vLLM-Rust

Remaining work to reach full feature parity with Python vLLM `3025b3c`.
All completed phases removed — see git history for done items.

---

## Phase 1: Tensor Parallelism (~3–6 weeks)

TP layer code exists and is TP-aware (attention, linear, MLP weights are already shardable).
Missing: the multi-process runtime. PP is fully working and is the reference pattern to follow.

| Item | What's Missing | Fix | Effort |
|------|---------------|-----|--------|
| 1.1 | TP > 1 process spawning | Mirror `spawn_pipeline_workers()` for TP: spawn N−1 worker processes each owning one GPU; NCCL all-reduce after every layer | 30–60h |
| 1.2 | TP worker loop | Analogous to `run_pipeline_worker()`: worker receives activations, runs its weight shard, participates in all-reduce | 20–40h |
| 1.3 | TP-aware weight loading | `from_config_with_tp(rank, tp_size)` for major architectures; shard Q/K/V/O by column, gate/up by column, down by row | 20–40h |

**Key files:**
- `crates/server/src/main.rs:814` — bail to remove once wired
- `crates/server/src/distributed_launcher.rs` — PP launcher to mirror
- `crates/core/src/models/llama.rs` — `new_with_pp()` pattern to generalize to `new_with_tp()`
- `crates/core/src/layers/` — attention/linear/tp_layers already TP-aware

---

## Phase 2: MiniMax-VL-01 (~2–4 weeks)

Lightning Attention backend (§5.1, commit `1fa3229`) is complete. Remaining gap is the vision side.

| Item | What's Missing | Fix | Effort |
|------|---------------|-----|--------|
| 2.1 | MiniMax-VL-01 vision encoder | Image patch projection → `MiniMaxText01ForCausalLM`; multimodal input routing; `MultimodalProcessor` registration | 20–40h |

**Key files:**
- `crates/core/src/models/minimax_text01.rs` — text model to extend with vision input path
- `reference/vllm/vllm/model_executor/models/minimax_vl.py` — reference implementation
- `crates/core/src/multimodal/` — existing VLM pipeline to reuse

---

## Phase 3: Low Priority / Deferred (indefinite)

| Item | Reason |
|------|--------|
| `lora_extra_vocab_size` | DEFERRED: vLLM `3025b3c` explicitly removed extra vocab support ("always 0, will be removed"). Re-implement if upstream restores it. |
| FlashAttn-DiffKV | Research models only |
| MLA CUTLASS/Sparse variants | Optimization of already-working MLA backend |
| `POST /v1/images/generations` | Diffusion pipeline — architecturally out of scope |
| Audio sinc interpolation | `rubato` crate — optional quality improvement over linear resampling |

---

## Dependency Graph

```
Phase 1 (TP > 1)     — independent; needs multi-GPU hardware for integration tests
Phase 2 (MiniMax-VL) — independent; Lightning Attention already unblocks it
Phase 3 (deferred)   — no dependencies; implement opportunistically
```

## Effort Summary

| Phase | Area | Effort |
|-------|------|--------|
| 1 | Tensor Parallelism runtime | 3–6 weeks |
| 2 | MiniMax-VL-01 vision wiring | 2–4 weeks |
| 3 | Low priority / deferred | indefinite |
| **Total** | | **~5–10 weeks** |
