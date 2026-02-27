# Feature Gap Plan: vLLM-Rust

Remaining work to reach full feature parity with Python vLLM `3025b3c`.
All completed phases removed — see git history for done items.

---

## Phase 1: Tensor Parallelism — DONE

TP runtime implemented. All three items complete:

| Item | Status |
|------|--------|
| 1.1 | ✅ `spawn_tensor_workers()` + `is_tp_worker_process()` in `distributed_launcher.rs` |
| 1.2 | ✅ `tensor_worker_loop` + `TpStagedModel` in `crates/core/src/engine/tensor_parallel.rs` |
| 1.3 | ✅ `from_config_with_tp` was already fully implemented for 50+ architectures |

Coordinator wraps the model in `TpStagedModel`, which broadcasts inputs to workers before each
forward. Workers run `tensor_worker_loop`, participating in NCCL all-reduce inside each TP-aware
layer and discarding their output. The bail at `main.rs:814` has been removed; `--tensor-parallel-size N`
is now wired end-to-end.

**Integration testing requires multi-GPU hardware** (not available in CI).

---

## Phase 2: MiniMax-VL-01 — DONE

Vision encoder wiring implemented. All items complete:

| Item | Status |
|------|--------|
| 2.1 | ✅ `MiniMaxVL01ForConditionalGeneration` in `crates/core/src/models/minimax_vl_01.rs` |

CLIP/SigLIP vision tower → 2-layer MLP projector → `image_newline` append → MiniMaxText01 backbone.
`MiniMaxText01Config` and `MiniMaxText01DecoderLayer` promoted to `pub(crate)` and reused directly
so multimodal embeddings can be injected before the forward pass without wrapping the text model.

**AnyRes note**: the single-image path (project all patches + append `image_newline`) is complete.
Multi-resolution AnyRes packing (`pack_image_features` with sub-image grid reshaping) is deferred
until the preprocessor pipeline surfaces `image_sizes` through `MultimodalInputs`.

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
Phase 1 (TP > 1)     — DONE
Phase 2 (MiniMax-VL) — DONE
Phase 3 (deferred)   — no dependencies; implement opportunistically
```

## Effort Summary

| Phase | Area | Effort |
|-------|------|--------|
| 1 | Tensor Parallelism runtime | ✅ DONE |
| 2 | MiniMax-VL-01 vision wiring | ✅ DONE |
| 3 | Low priority / deferred | indefinite |
| **Total** | | **✅ All actionable phases complete** |
