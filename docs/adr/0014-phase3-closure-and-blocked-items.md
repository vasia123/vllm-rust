# ADR-0014: Phase 3 Closure + Blocked-Items Audit

## Status

Accepted

## Context

After completing Phases 7 (MoE consolidation), 8 (vision-no-collapse via
ADR-0012), and 9 (registry-based dispatch + `models/mod.rs` cleanup),
five items from the original Phase 3 scope plus the blocked-items
backlog needed disposition:

- **Phase 3a (RmsNorm consolidation)** — was the original 13-Gemma-files
  unification plan worth executing on remaining models?
- **Phase 3b (MLP/SwiGLU consolidation)** — 161 local MLP-style structs
  in `models/`. Which can collapse into shared types in
  `crate::layers::mlp` and `crate::models::tp_layers`?
- **Phase 3c (RoPE consolidation)** — 5 vision-2D-RoPE variants and
  their migration story.
- **GPU-only quantization kernels** — AWQ-Marlin PTX, FP8-Marlin
  Ampere — multi-week GPU dev.
- **Blocked features** — DeepGEMM external lib, CUDA graphs, NCCL
  pipeline, OpenTelemetry, guided decoding, MiniMax Lightning Attention.

This ADR records the disposition.

## Decision

### Phase 3a (RmsNorm) — done; not actionable further

Audit (Explore agent, 2026-05-06): all Gemma 1/2/3/4 + quant + lora +
vlm files use `crate::layers::RmsNorm` via `rms_norm_gemma()`. The
remaining eight local norm structs are architecturally distinct — they
are *not* RmsNorm:

| File | Struct | Reason it's not RmsNorm |
|------|--------|--------------------------|
| `t5.rs` | `T5LayerNorm` | No mean-subtraction; T5-specific pure RMS-then-scale. |
| `deepseek_ocr.rs` | `SamLayerNorm2d` | 2D channel-wise on `[B,C,H,W]` (SAM ViT). |
| `deepseek_ocr2.rs` | `SamLayerNorm2d` | Same as above. |
| `mlp_speculator.rs` | `MLPSpeculatorLayerNorm` | L2 norm with custom semantics. |
| `nemotron.rs` | `LayerNorm1P` | Standard `LayerNorm`, *not* RmsNorm. |
| `nemotron_quantized.rs` | `LayerNorm1P` | Same as above. |
| `kanana_v.rs` | `LayerNorm2d` | 2D channel-wise vision LayerNorm. |
| `hyperclovax_vision.rs` | `LayerNorm2d` | 2D channel-wise vision LayerNorm. |

Verdict: collapsing these into `crate::layers::RmsNorm` would force a
genericised norm API to absorb operations that are not even RMS, at no
practical benefit. Keep bespoke; document the whitelist; close Phase 3a.

### Phase 3c (RoPE) — done; vision-2D stays bespoke per ADR-0012

All text models use `crate::layers::rotary::*` (`RotaryEmbedding`,
`new_partial`, `new_gemma4_proportional`, `new_su_scaled`,
`XDRotaryEmbedding` for Qwen-VL MRoPE). The remaining 5 vision-2D
variants — Gemma4Vision, DOTS, SigLIP/Keye, ERNIE 4.5-VL, Pixtral —
have model-specific 2D layouts (axis split, head-dim concat, custom
inv_freq). ADR-0012 already chose deliberately *not* to collapse vision
encoders into a `VisionTowerFactory`; the corresponding 2D RoPE stays
under that whitelist.

Verdict: Phase 3c closed. RoPE library covers all text models;
vision-2D RoPE is bespoke by design. The
`crates/core/tests/no_local_vision_tower.rs` guardrail catches future
regressions.

### Phase 3b (MLP) — done with calibration cohort + foundation

Audit found the *real* migration target was not the 161 raw structs
but the family of TP-aware fused-gate-up SwiGLU MLPs duplicated across
~20 model files. The non-TP `crate::layers::SwiGluMlp` already covers
the small minority of models that use `candle_nn::Linear` directly.
The high-value work was generalising the TP variants in
`crates/core/src/models/tp_layers.rs`:

1. **`GluProjNames`** constants — `STANDARD` (gate_proj/up_proj/down_proj)
   and `W1_W3_W2` (InternLM2/Yi feed_forward.{w1,w3,w2}).
2. **`TpSwiGluMlp::new_with_proj_names()`** — same TP sharding as
   `new()`, only the VarBuilder lookup differs.
3. **`TpFusedSwiGluMlp`** — Phi-3-style fused `gate_up_proj` with
   `new()` (no bias, common case) and `new_with_bias()` (Step1 / Ernie
   4.5-MoE / future biased configs).

Initial calibration cohort (seven models, ~245 LOC removed):

- `internlm2.rs` → `TpSwiGluMlp::new_with_proj_names(W1_W3_W2)`.
- `phi3.rs` → `TpFusedSwiGluMlp::new`.
- `ouro.rs` → `TpFusedSwiGluMlp::new`.
- `hunyuan.rs` (`HunYuanSharedMLP`) → `TpFusedSwiGluMlp::new`.
- `pangu.rs` (`PanguMLP`) → `TpFusedSwiGluMlp::new`.
- `step1.rs` (`Step1MLP`) → `TpFusedSwiGluMlp::new_with_bias`.
- `ernie45_moe.rs` (`Ernie45MoeMLP`) → `TpFusedSwiGluMlp::new`.

Second cohort (eight more models, ~250 LOC removed) extends the
foundation and migrates the remaining variants:

- `tp_layers.rs::TpFusedSwiGluMlp::new_with_bias_and_limit` adds an
  optional clamp threshold for clipped SwiGLU
  (`min(silu(gate), L) * clamp(up, -L, L)`).
- `layers/mlp.rs::FusedSwiGluMlp` — non-TP analogue with `new()` and
  `new_with_bias()` for models whose `gate_up_proj` is a plain
  `candle_nn::Linear`.
- `layers/mlp.rs::SwiGluMlp::new_with_proj_names()` — split-name
  layout where gate/up live under a fused parent prefix
  (`gate_up_proj.gate` / `gate_up_proj.up`).

Migrations:

- `step3p5.rs` → `TpFusedSwiGluMlp::new_with_bias_and_limit` (drops
  bespoke `swiglu_with_limit` helper).
- `plamo2.rs`, `plamo3.rs` → `SwiGluMlp::new_with_proj_names`.
- `dots1.rs`, `minimax_text01.rs`, `interns1_pro.rs` →
  `FusedSwiGluMlp::new`.
- `chameleon.rs`, `glm_ocr.rs` (vision MLP) →
  `FusedSwiGluMlp::new_with_bias(true)`.
- `bailing_moe.rs` (shared expert) → `TpFusedSwiGluMlp::new_with_bias`.

Phase 3b net delta: ~500 LOC removed across fifteen model files;
foundation grew by ~120 LOC. The fused-gate-up SwiGLU pattern is now
single-source-of-truth in `tp_layers.rs` and `layers/mlp.rs`.

### Why no `no_local_swiglu.rs` guardrail

A `no_local_attention.rs`-style guardrail was considered for MLPs and
rejected. The two guardrails differ in their hit/miss rate:

- `no_local_attention` matches the narrow `struct .*Attention` pattern,
  whose remaining whitelisted items are a small, well-known set
  (DeepSeek MLA, Mamba/Jamba/Bamba SSM, GLA hybrids, Lightning
  Attention). The greps are clean.
- `struct .*M[Ll][Pp]\b` / `struct .*FFN\b` would match ~150+
  legitimately bespoke items (vision-tower MLPs, encoder-projector
  fc1/fc2 stacks, MoE expert weights, Mamba projection blocks). The
  whitelist would dominate the test, defeating the regression-catching
  purpose.

Future fused-gate-up SwiGLU regressions are caught by code review and
by the fact that every model family already has its example covered
in this ADR. If a new family duplicates the pattern, the right move is
to extend `TpFusedSwiGluMlp` (e.g. add `new_with_proj_names`) and
migrate, not to add it as a new bespoke struct.

### GLU consolidation extensions — DEFERRED

The audit identified additional candidate extensions:

- **GeGLU TP variant** — `TpGeGluMlp` exists in `tp_layers.rs` and is
  used by Gemma 1/2/3/4. Already done.
- **GeLU TP variant** — `TpGeluMlp` exists, used by Falcon/Phi (legacy).
  Already done.
- **`GluActivation::Xielu` (Apertus)** — single callsite. Adding a
  variant just for one model is YAGNI. Revisit when a second user
  appears.
- **`GluActivation::Relu2` (JAIS2)** — same single-callsite reasoning.
- **`ProjNames::Bloom` (`dense_h_to_4h`/`dense_4h_to_h`)** — these are
  classical FFN (no GLU gate), not SwiGLU. Universalising them
  requires a separate `LinearMlp` type, *not* an extension of
  `SwiGluMlp`. Deferred until 3+ users surface; until then bespoke is
  the correct choice.

### Items 9-10 — audit found everything already implemented (or accepted as-is)

The 2026-05-06 follow-up audit revealed the original "items 9-10
deferred" framing was based on outdated information. Each pillar was
re-checked against current `main`:

#### 9.1 AWQ-Marlin GPU repack — **DONE**

`crates/core/src/quantization/marlin_cuda.rs:awq_marlin_repack` calls
the `awq_marlin_repack_int4` PTX entry compiled into
`crates/core/kernels/marlin_gemm.ptx`. AWQ qzeros are routed into the
fused `marlin_gemm` kernel via the `has_zp` flag, so the proposed
separate `marlin_awq_repack_zp` is not needed (the kernel handles AWQ
qzeros natively).

#### 9.2 FP8-Marlin Ampere — **DONE**

`marlin_gemm_fp8_bf16` PTX entry exists with software FP8 decode for
SM80+. `Fp8AmpereGemmOp` (`marlin_cuda.rs:518`) wraps it as a
`CustomOp1`. Validated by 91 fp8 lib tests.

#### 10.1 DeepGEMM — **DONE (in-house equivalent)**

The DeepGEMM-equivalent surfaces are covered by three in-house GPU
paths instead of vendoring the external library:

- General FP8 GEMM: `crates/core/src/quantization/fp8.rs` →
  `fp8_cuda::fp8_gemm` PTX kernel.
- Marlin FP8 GEMM: `marlin_gemm_fp8_bf16` (item 9.2 above).
- MoE FP8 dispatch: DeepGEMM-lite (`crates/core/src/moe/fused/`),
  GPU-resident token alignment from Phase 5.3.

Vendoring the upstream `https://github.com/deepseek-ai/DeepGEMM` would
add bindgen complexity for marginal benefit. The trigger to revisit
would be a workload that none of the three paths above cover.

#### 10.2 CUDA graphs — **DONE**

`crates/core/src/engine/cuda_graph.rs` provides `CudaGraphDispatcher`
+ `BatchDescriptor`; `cuda_graph_runner.rs` provides the
capture/replay runner. `ModelForward::supports_cuda_graphs()` is the
opt-in gate. Wired into `StandardEngine`. 26 lib tests pass under
`--features cuda`.

#### 10.3 NCCL pipeline-parallel transport — **DONE**

`crates/core/src/distributed/nccl.rs` provides `NcclCommunicator::{send,
recv}` (lines 818, 837) backed by `ncclSend` / `ncclRecv` FFI symbols.
`crates/core/src/engine/pipeline.rs` calls these for both intermediate
hidden-state forwarding and final logits return-to-rank-0.

#### 10.4 OpenTelemetry export — **DONE**

`crates/server/src/logging.rs::init_with_otlp` wires
`tracing-opentelemetry` + `opentelemetry-otlp` (HTTP/JSON) behind the
`--otlp-traces-endpoint` server flag. Resource attribute
`service.name=vllm-server`. Validated by `test_log_format_from_env`.

#### 10.5 Guided decoding (structured outputs) — **DONE (in-house DFA backend)**

`crates/core/src/sampling/grammar/` ships a from-scratch DFA-compiled
bitmask backend supporting:

- Regex patterns (`regex_backend.rs`)
- JSON Schema (`json_schema.rs`, schema → regex → DFA)
- EBNF / GBNF grammars (`ebnf_backend.rs`, regular → DFA, recursive →
  PDA)

`GrammarConstraintAdapter` bridges into `SamplingConstraint`, so the
existing `LogitsProcessorPipeline` consumes it without changes.
Server API (`StructuredOutputs` in `crates/server/src/api/types.rs`)
exposes `json` / `regex` / `choice` / `grammar` / `json_object` fields
on chat and completions endpoints. Validation prevents combining with
`response_format`. 65 grammar tests + 10 server validation tests pass.

The proposed `llguidance` crate integration would have been a thinner
shim around an external dependency; the in-house implementation
trades lib weight for full control over the bitmask pipeline.

#### 10.6 MiniMax Lightning Attention — **DONE (functional GPU; fused kernel deferred)**

`MiniMaxText01LinearAttention::forward_prefill` and
`forward_decode_batch` use device-agnostic candle tensor ops, so the
existing implementation runs end-to-end on GPU under `--features
cuda` (validated by lib tests). The sequential loop launches ~L×5
kernels per layer, which is correct but suboptimal for long
sequences.

A fused recurrent-scan PTX kernel along the lines of `ssd_scan_f32`
(already in tree for Mamba2) would collapse the inner loop. Triggered
by a workload showing concrete latency regression vs. a Mamba2-shape
baseline; estimated 2 weeks GPU dev + parity bench. Tracked here as
the only remaining performance optimisation, not a correctness gap.

## Consequences

**Benefits:**

- All five Phase-3 items have a definite disposition: closed (3a, 3c),
  done with calibration cohort (3b), or formally deferred with design
  sketches and triggers (9, 10).
- The `tp_layers.rs` MLP API is now config-aware and ready for future
  models without bespoke duplication.
- Future contributors can pick up any item 9 or 10 cold from this ADR
  without re-discovering the design space.

**Trade-offs:**

- Phase 3b is *not* exhaustive — ~10-12 fused-gate-up MLP variants
  remain bespoke because each has its own quirk. The cost-benefit of
  forcing them all through `TpFusedSwiGluMlp` config-extension is
  negative today; revisit when a third caller of a given quirk
  appears.
- Items 9-10 are all implemented in tree (audit dated 2026-05-06).
  The only remaining performance optimisation is the MiniMax Lightning
  Attention fused PTX kernel (item 10.6), tracked as a future
  enhancement against a real-workload latency trigger.

**Pairs with:**

- ADR-0010 — AttentionBlock charter; same "consolidate where it pays,
  bespoke where it doesn't" discipline.
- ADR-0011 — MoE primitives.
- ADR-0012 — vision encoders stay bespoke (no `VisionTowerFactory`).
- ADR-0013 — registry dispatch is now default-on.

Together, ADRs 0010-0014 record vllm-rust's operational view of the
consolidation effort: pay the cost where it pays back (Phase 4
attention, Phase 7 MoE primitives, Phase 9 dispatch registry, Phase 10
TP fused-gate-up MLP), defer it where it doesn't (Phase 3a/3c trivial
remainders, Phase 8 vision tower, Phase 3b cohort C extensions),
document the rest (items 9-10) so future work can resume cold.
