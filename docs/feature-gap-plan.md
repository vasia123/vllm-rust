# Feature Gap Plan: vLLM-Rust

Remaining work to reach full feature parity with Python vLLM `3025b3c`. Only unfinished items listed — all completed work has been removed.

---

## Phase 1: Server CLI Wiring (~1–2 days)

All infrastructure exists. The gap is 10 `let _ = ...; // TODO` lines in `crates/server/src/main.rs:1126-1151`.

| Item | CLI Arg | Existing Infrastructure | Fix | Effort |
|------|---------|------------------------|-----|--------|
| 1.1 | `enforce_eager` + `max_seq_len_to_capture` | `cuda_graph.rs` (797L) + `cuda_graph_runner.rs` (906L) fully done | Pass to `CudaGraphConfig`: `!enforce_eager` → enabled, `max_seq_len_to_capture` → threshold | 1–2h |
| 1.2 | `guided_decoding_backend` | `sampling/grammar/` fully done: DFA compiler, regex, JSON schema, EBNF, bitmask | Wire request `guided_json/regex/grammar` → `GrammarCompiler` → `SamplingConstraint` | 2–3h |
| 1.3 | `otlp_traces_endpoint` | `tracing` crate in workspace | Add `opentelemetry` + `opentelemetry-otlp` + `tracing-opentelemetry` crates; init OTLP layer | 3–5h |
| 1.4 | `load_format` / `tokenizer_mode` | Loader auto-detects safetensors | Add `LoadFormat` enum dispatch (safetensors/pt/dummy); `tokenizer_mode` validation | 4–8h |
| 1.5 | `code_revision` / `max_parallel_loading_workers` | Parsed but unused | Pass `code_revision` to HF API; use semaphore for parallel shard loads | 2–4h |

---

## Phase 2: Server Infrastructure (~1 week)

Meaningful new logic needed.

| Item | What's Missing | Fix | Effort |
|------|---------------|-----|--------|
| 2.1 | `lora_extra_vocab_size` — LoRA vocab extension | Load extra embeddings from adapter, expand lm_head, per-request merge | 15–25h |
| 2.2 | `disable_mm_preprocessor_cache` — VLM preprocessor cache | SHA-256 keyed cache for vision encoder outputs, LRU eviction | 8–15h |
| 2.3 | `max_cpu_loras` — LRU adapter eviction | LRU map for dynamically loaded adapters, evict on `/v1/load_lora_adapter` | 4–8h |

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
