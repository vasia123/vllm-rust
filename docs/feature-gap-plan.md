# Feature Gap Plan: vLLM-Rust

Remaining work to reach full feature parity with Python vLLM `3025b3c`. Only unfinished items listed — all completed work has been removed.

---

## Phase 1: Server CLI Wiring ✅ DONE

All infrastructure existed. Wired all remaining `let _ = ...; // TODO` lines in `main.rs`.

| Item | Status | Commit |
|------|--------|--------|
| 1.1 `enforce_eager` + `max_seq_len_to_capture` → `CudaGraphConfig` | ✅ DONE | `64e2161` |
| 1.2 `guided_decoding_backend` validation + debug log | ✅ DONE | `64e2161` |
| 1.3 `otlp_traces_endpoint` → OTLP/HTTP JSON exporter via tracing-opentelemetry | ✅ DONE | `aac166b` |
| 1.4 `load_format` → `LoadFormat` enum (auto/safetensors/pt/npcache/dummy); `tokenizer_mode` validation | ✅ DONE | `350e19b` |
| 1.5 `code_revision` → debug log; `max_parallel_loading_workers` → parallel shard downloads | ✅ DONE | `350e19b` |

---

## Phase 2: Server Infrastructure (~1 week)

Meaningful new logic needed.

| Item | What's Missing | Fix | Effort |
|------|---------------|-----|--------|
| 2.1 ⚠️ DEFERRED | `lora_extra_vocab_size` — LoRA vocab extension | vLLM reference (3025b3c) explicitly removed extra vocab support ("extra_vocab_size always to 0, will be removed"). CLI arg is preserved; warn when non-zero; re-implement when reference restores it. | — |
| 2.2 ✅ | `disable_mm_preprocessor_cache` — VLM preprocessor cache | SHA-256 keyed LRU cache in `PreprocessorCache`; `MultimodalProcessor` checks before encoding; `AppState.disable_mm_preprocessor_cache` propagates flag | `25652c3` |
| 2.3 ✅ | `max_cpu_loras` — LRU adapter eviction | `LoraAdapterRegistry` with `VecDeque` LRU; evicts on `/v1/load_lora_adapter` | `e45fd81` |

---

## Phase 3: GPU Kernel Optimization (~1–2 weeks)

CPU fallback paths work. These add GPU-accelerated paths.

| Item | Current State | Gap | Effort |
|------|--------------|-----|--------|
| 3.1 | AWQ-Marlin CPU path works (7 tests) | No `awq_marlin_repack_int4` PTX kernel for GPU repack | 10–20h |
| 3.2 | FBGEMM-FP8 Ada works, Ampere falls to CPU | Wire `MarlinScalarType::Float8E4m3fn` for Ampere | 5–10h |
| 3.3 | MXFP4 CPU emulation works (19 tests) | No GPU dispatch for MoE FP4 compute | 10–15h |

---

## Phase 4: Distributed (~2–3 weeks)

Requires multi-GPU testing.

| Item | What Exists | Gap | Effort |
|------|------------|-----|--------|
| 4.1 | PP signals + worker loop + 27 tests | Stage construction (layer slicing by rank), NCCL multi-process launch | 30–50h |
| 4.2 | EPLB state tracking + rebalance detect | `rearrange_expert_weights_inplace` via NCCL all-to-all | 15–25h |

---

## Phase 5: New CUDA Kernels — flashinfer-rs + vllm-rust (~4–8 weeks)

| Item | What's Needed | Where | Effort |
|------|--------------|-------|--------|
| 5.1 | Linear / Lightning Attention | flashinfer-rs: GDN recurrence kernel; vllm-rust: `attention/linear.rs` backend | 40–80h |
| 5.2 | Mamba SSD GPU | flashinfer-rs: parallel scan + chunk carry kernel; vllm-rust: wire in `ssm/ssd.rs` | 30–50h |
| 5.3 | DeepGEMM (grouped GEMM for MoE) | flashinfer-rs or new crate: variable-M grouped GEMM; vllm-rust: wire in `moe/` | 20–40h |

---

## Phase 6: Low Priority (indefinite)

| Item | Reason |
|------|--------|
| FlashAttn-DiffKV | Research models only |
| MLA CUTLASS/Sparse variants | Optimization of existing working MLA backend |
| `POST /v1/images/generations` | Diffusion pipeline — out of scope |
| MiniMax-VL-01 | Blocked on §5.1 Lightning Attention |
| Audio sinc interpolation | `rubato` crate — optional quality improvement over linear resampling |

---

## Dependency Graph

```
Phase 1 (CLI wiring) — no blockers, start immediately
Phase 2 (server infra) — no blockers, parallelizable with Phase 1
Phase 3 (GPU kernels) — independent, needs CUDA dev environment
Phase 4.1 (PP stage construction) — needs NCCL multi-process infra
Phase 4.2 (EPLB rearrange) — needs NCCL all-to-all
Phase 5.1 (Lightning Attention) — blocks MiniMax-VL-01
Phase 5.2 (Mamba SSD GPU) — blocks efficient Mamba/Jamba inference
Phase 5.3 (DeepGEMM) — blocks MoE GPU optimization
```

## Effort Summary

| Phase | Area | Effort |
|-------|------|--------|
| 1 | Server CLI wiring | 1–2 days |
| 2 | Server infrastructure | ~1 week |
| 3 | GPU kernel optimization | 1–2 weeks |
| 4 | Distributed (PP + EPLB) | 2–3 weeks |
| 5 | New CUDA kernels | 4–8 weeks |
| 6 | Low priority | indefinite |
| **Total** | | **~8–15 weeks** |
