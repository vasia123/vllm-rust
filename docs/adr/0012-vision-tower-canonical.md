# ADR-0012: Canonical Vision Tower (`multimodal::VisionEncoder`)

## Status

Accepted

## Context

35+ VLMs in `crates/core/src/models/` integrate a vision encoder with
their text backbone. Across them, vision-encoder structs add up to
~139 implementations of small variations on the ViT backbone (CLIP,
SigLIP, SigLIP2, MoonViT, RadioViT/InternViT, custom Gemma4 vision,
custom Qwen2-VL/Qwen3-VL towers, …).

Phase 4c of the architecture refactor planned a `VisionTowerFactory`
collapsing this to ~5 canonical variants + `Custom(Box<dyn
VisionTower>)`. The plan estimated −12000 to −14000 LOC.

Reality check on the existing codebase:

- `crates/core/src/multimodal/vision.rs` already exposes
  `VisionEncoder` + `VisionEncoderType { Clip, SigLip }` +
  `VisionEncoderConfig`. Several VLMs route through it as the
  canonical path.
- The remaining bespoke vision encoders (clip.rs, siglip.rs,
  pixtral.rs, gemma4_vision.rs, dots_ocr.rs, hunyuan_vision.rs,
  keye_vl.rs, qwen2_vl.rs, qwen2_5_vl.rs, qwen3_vl.rs,
  qwen3_vl_moe.rs, isaac.rs, radio.rs, ovis.rs, ovis2_5.rs,
  internvl.rs, interns1.rs, midashenglm.rs, llama4_vl.rs,
  step3_vl.rs, openpangu_vl.rs, glm4_1v.rs, glm4v.rs, glm_ocr.rs,
  ernie45_vl.rs, blip2.rs, aria.rs, minicpmv.rs, qwen_vl.rs,
  qwen3_omni_moe_thinker.rs, qwen2_audio.rs, granite_speech.rs,
  nemotron_parse.rs) each carry **model-specific** patch handling,
  positional encoding, normalisation, or activation choices that
  don't fit a "single encoder, 5 enum variants" shape:
  - Qwen2/2.5/3-VL: dynamic resolution + window attention + MRoPE.
  - Pixtral: 2D RoPE in vision tower.
  - Gemma4-Vision: GeLU-tanh + custom patch + image-token packing.
  - Pangu-VL / Hunyuan-Vision / Step3-VL: custom adaptive resolution.
  - RadioViT / InternViT (used by InternVL/Nemotron): conditional
    positional embedding (CPE) + ViT-H blocks.
  - Audio encoders (qwen2_audio, granite_speech, midashenglm): not
    vision at all — Mel/Conv1d-based encoders.

A "single canonical tower with N enum variants" abstraction would
either hide too much per-model behaviour (defeat the purpose) or
explode the variant count toward 1-per-VLM (defeat the abstraction).

## Decision

The canonical building block stays
[`crate::multimodal::VisionEncoder`] for VLMs whose vision tower is
plain CLIP or SigLIP. New VLMs in those two families MUST route
through it.

The remaining bespoke vision encoders stay bespoke and are recorded
in a `tests/no_local_vision_tower.rs` CI guardrail mirroring the
`no_local_attention.rs` mechanism (introduced in Phase 6 / ADR-0010).
Each bespoke entry carries a one-line reason explaining what
model-specific shape forced the bespoke implementation.

This is a **deliberate scope decision**: vision encoders are far less
uniform than decoder-only attention (which the plan pegged at 80%
canonicalisable). Forcing a `VisionTowerFactory` here would either
cost more than it saves (per the dimensionality of the variation
space) or sacrifice the per-model patch/RoPE/scaling subtleties that
matter for image quality.

### What this means concretely

- **Two canonical variants** today: `VisionEncoderType::Clip`,
  `VisionEncoderType::SigLip`. Adding a third (e.g. SigLIP2,
  Gemma4-Vision) requires evidence that ≥3 VLMs share its exact
  shape; otherwise it stays bespoke.
- **No `VisionTowerFactory` trait + dynamic dispatch.** The plan's
  `VisionVariant::Custom(Box<dyn VisionTower>)` escape hatch would
  immediately become the dominant path because most VLMs need it.
- **Guardrail enforces explicit decisions.** A new bespoke vision
  encoder is allowed if the contributor adds a whitelist entry with
  a reason; the test catches PRs that drift silently.

### What about the LOC target?

The plan's −14000 LOC estimate assumed every VLM could collapse to
~100 LOC behind a thin shim. In practice:

- VLMs that already use `VisionEncoder` (LLaVA, MiniCPM-V, Idefics3,
  Aria, BLIP-2, Tarsier) saved that LOC at write time, not later.
- Bespoke VLMs cost their own ~400 LOC each, but the LOC is
  load-bearing — it implements the model-specific image-feature
  pipeline. Removing it would require implementing the same logic
  inside `VisionEncoder` with per-model branches, which is the same
  total LOC plus indirection.

The honest target after this ADR: −0 LOC from collapse, but
**+discipline** through the guardrail and explicit bespoke list.

## Consequences

**Benefits:**

- Single source of truth for what counts as a "vision-tower bespoke"
  exception — codified as data in `tests/no_local_vision_tower.rs`.
- Future LLaVA-class VLMs slot into the existing `VisionEncoder` API
  without inventing their own.
- The plan's debt is closed: the boundary is explicit, the test
  catches drift.

**Trade-offs:**

- We don't get the −14000 LOC the plan envisioned. The estimate was
  optimistic about how much variation lives in vision towers; the
  real number is closer to per-model −5–10 LOC after some
  consolidation, dwarfed by audio and tokeniser code that VLMs
  require anyway.
- Adding a new VLM still requires per-model engineering decisions
  for the vision side. The `adding-a-model.md` guide already directs
  contributors to existing VLMs as templates.

**Future work:**

- If a future VLM cluster (e.g. all Qwen-VL variants converge on
  one shape) appears, the `VisionEncoderType` enum can grow a third
  variant with model-specific knobs on `VisionEncoderConfig`. The
  guardrail is the trigger: when ≥3 bespoke files start having the
  same `// reason:` line, that's the signal to canonicalise.
- The `no_local_vision_tower.rs` guardrail should be paired with
  `tests/no_local_attention.rs` and `tests/no_local_moe.rs` (when
  the latter is added) for a uniform "discipline-by-test" story.
