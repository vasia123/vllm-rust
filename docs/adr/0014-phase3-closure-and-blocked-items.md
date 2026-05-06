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

Calibration migrations (six models, ~245 LOC removed):

- `internlm2.rs` → `TpSwiGluMlp::new_with_proj_names(W1_W3_W2)`.
- `phi3.rs` → `TpFusedSwiGluMlp::new`.
- `ouro.rs` → `TpFusedSwiGluMlp::new`.
- `hunyuan.rs` (`HunYuanSharedMLP`) → `TpFusedSwiGluMlp::new`.
- `pangu.rs` (`PanguMLP`) → `TpFusedSwiGluMlp::new`.
- `step1.rs` (`Step1MLP`) → `TpFusedSwiGluMlp::new_with_bias`.
- `ernie45_moe.rs` (`Ernie45MoeMLP`) → `TpFusedSwiGluMlp::new`.

Verdict: foundation locked, calibration cohort proves shape. Remaining
fused-gate-up models (`step3p5` with `swiglu_with_limit`, `dots1`
non-TP, `chameleon`/`minimax_text01`/`interns1_pro`/`glm_ocr` with
non-TP `Linear`, plamo2/3 with `gate_up_proj.{gate,up}` split-name
layout) stay bespoke until they grow a TP path or until a new model
forces the issue. Each has a model-specific quirk that does not justify
a `TpFusedSwiGluMlp` config knob today (YAGNI).

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

### Items 9-10 (GPU kernels + blocked features) — deferred with design sketches

Each of these is a multi-week project requiring GPU dev or external
library integration. Recording design sketches and triggers so future
sessions can pick them up cold.

#### 9.1 AWQ-Marlin PTX repack (`awq_marlin_repack_int4` + `marlin_awq_repack_zp`)

Trigger: AWQ throughput on Hopper / SM89 deployments.

Design sketch:
- Reference impl: vLLM `csrc/quantization/awq_marlin/awq_marlin_repack.cu`.
- One PTX kernel per direction: AWQ→Marlin nibble repack for weights,
  AWQ→Marlin zero-point repack for qzeros.
- Replace the current CPU `repack_awq_nibbles` (already correct) with
  the GPU path via `cudarc::driver::CudaFunction`.
- Estimated: 1 week GPU dev + bench.

#### 9.2 FP8-Marlin Ampere

Trigger: Ampere (SM80) deployment without SM89.

Design sketch: SM80 lacks native FP8; needs FP8-emulation kernel
storing FP8 as `u8` with software conversion. Reference: vLLM
`csrc/quantization/fp8/`. Estimated: 1.5 weeks GPU dev.

#### 10.1 DeepGEMM (external lib)

Trigger: DeepSeek-V3 production deployment requirement.

Design sketch: vendor `https://github.com/deepseek-ai/DeepGEMM` under
`vendor/deepgemm/` with a `cuda-deepgemm` feature flag. Generate FFI
bindings via `bindgen`. Wire into `quantization::Fp8Config::matmul`.
Estimated: 3 days for binding + 2 days for benchmarking parity.

#### 10.2 CUDA graphs

Trigger: latency-critical decode workload.

Design sketch:
```rust
struct GraphedDecodeStep {
    graph: cudarc::driver::CudaGraph,
    captured_shape: BatchShape,
}
```
First decode at a given batch shape captures the graph; subsequent
decodes at the same shape replay it. Falls back to non-graphed path
when shape changes. Estimated: 1 week + tuning.

#### 10.3 NCCL pipeline-parallel transport

Trigger: multi-node deployment.

Design sketch: Phase 9 already established PP correctness via the
worker signal protocol. NCCL is a *throughput optimisation* — replace
point-to-point `Tensor.cat`/`split` with `ncclSend`/`ncclRecv`.
`crates/core/src/distributed/pp.rs` already wraps the protocol; only
the transport layer changes. Estimated: 1.5 weeks.

#### 10.4 OpenTelemetry export

Trigger: production telemetry requirement.

Design sketch: no architectural blockers — wrap existing `tracing`
spans in an `opentelemetry-otlp` exporter via the
`tracing-opentelemetry` crate. Add `--otlp-endpoint` server flag.
Estimated: 2 days.

#### 10.5 Guided decoding (outlines / llguidance)

Trigger: structured-output API (e.g. JSON schema, regex grammar).

Design sketch: CPU-side post-processor on logits.
```rust
trait GuidedLogitsProcessor {
    fn mask(&self, logits: &mut [f32], generated_so_far: &[TokenId]);
}
```
Bind `llguidance` Rust crate (already exists, no FFI needed).
Plug into `LogitsProcessor` chain after temperature/top-k. **Does
not require GPU work.** Estimated: 1 week.

#### 10.6 MiniMax-VL/Text01 Lightning Attention (GPU)

Trigger: production MiniMax-Text01 deployment.

Design sketch: `MiniMaxText01LinearAttention` is implemented in pure
Rust (CPU-correct). GPU port requires a fused recurrent-scan kernel —
either Triton via PyO3 or a CUDA PTX kernel along the lines of
`ssd_scan_f32` (already exists for Mamba2). Estimated: 2 weeks GPU
dev + parity bench.

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
- Items 9-10 are documented but unimplemented. Anyone needing AWQ-Marlin
  or DeepGEMM on a real workload will need a GPU-equipped session.

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
