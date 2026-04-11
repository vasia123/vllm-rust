# Gemma 4 Quantized Support — Gap Analysis & Implementation Plan

**Date:** 2026-04-10
**Status (2026-04-10):** Phases 1–3 DONE, Phase 4 (TP) DEFERRED pending project-wide quantized-TP infrastructure. See "Implementation Status" section below.
**Scope:** What needs to be done to load quantized Gemma 4 (E2B/E4B/26B-A4B/31B) models via vllm-rust on consumer GPUs (8GB VRAM target).
**Target use case:** D&D Living World benchmark — text-only inference, Gemma 4 E2B/E4B with BNB/AWQ/GGUF quantization on RTX 4060 Laptop (8GB).

---

## Implementation Status (2026-04-10)

| Phase | Scope | Status | Notes |
|---|---|---|---|
| 1 | `gemma4_quantized.rs` text model + mod.rs dispatch + 6 unit tests | DONE | `QuantizedGemma4ForCausalLM` mirrors `gemma4.rs` with `Box<dyn QuantizedLinear>`; 6 unit tests pass, 1 ignored HF integration test |
| 3 | `gemma4_vlm_quantized.rs` VLM wrapper + `PrefixedWeightLoader` helper | DONE | Vision tower + projector stay FP; LM goes through prefixed loader so checkpoints nested under `language_model.*` resolve correctly; 4 tests pass |
| 2 | GGUF tensor-name fallback mapping (`to_llama_cpp_prefix`) | DONE (partial) | Standard Gemma-family dense path (attn, MLP, 4 layernorms, QKV norms, embeds) maps vLLM-style → `blk.N.*` llama.cpp-style. MoE stacked experts + PLE specifics NOT mapped (need real checkpoint to verify). 6 unit tests cover the mapping itself. |
| 4 | `QuantizedGemma4ForCausalLM::new_with_tp` | **DEFERRED** | No quantized model in the project has a TP variant. Blocking gap is project-wide: `QuantizedLinear` trait has no shard metadata, `from_config_with_quant_with_tp` does not exist. Implementation would require ~200–500 LOC of infrastructure touching all 20+ quant backends. Out of scope for the Gemma 4 feature; single-GPU path covers the 8GB benchmark target. |

**Net result for D&D benchmark (text, 8GB VRAM):** usable end-to-end via BnB 4-bit on single GPU. VLM path also usable for image prompts. GGUF path covers common dense configurations through the name mapping fallback; Gemma 4 MoE GGUFs remain TODO.

---

## TL;DR

vllm-rust has a **fully working full-precision Gemma 4** (`gemma4.rs` — 1865 lines, PLE + MoE + QKV norms + sliding window) and a working VLM wrapper (`gemma4_vlm.rs`). But there's **no `gemma4_quantized.rs`**, and the quantized dispatch in `from_config_with_quant()` has no Gemma 4 arm. Result: any attempt to load an AWQ/BNB/GGUF Gemma 4 checkpoint falls through.

**Critical path** (≈2000 LOC, one focused feature):
1. Create `crates/core/src/models/gemma4_quantized.rs` mirroring `gemma4.rs` but using `Box<dyn QuantizedLinear>`
2. Register module + re-export in `models/mod.rs`
3. Add dispatch arm in `from_config_with_quant()`
4. Extend GGUF tensor-name mapping for Gemma 4 specifics (PLE, router)
5. Integration test loading a real HF checkpoint

---

## Current State

### What already works for Gemma 4

| File | LOC | Role |
|------|-----|------|
| `crates/core/src/models/gemma4.rs` | 1865 | Full-precision text model (BF16/FP32). PLE embeddings, dense+sparse MoE in parallel, per-head Q/K/V norms, `k_eq_v` laptop variant, `layer_scalar`, soft cap, sliding/global attention pattern |
| `crates/core/src/models/gemma4_vlm.rs` | 466 | Vision-Language wrapper: SigLIP vision tower + `Gemma4MultimodalEmbedder` (Linear + UnweightedRMSNorm) + `Gemma4ForCausalLM` |
| `crates/core/src/models/mod.rs` | — | Registered in full-precision `from_config()` at lines 723, 851, 1708 (TP variant): `Gemma4ForCausalLM`, `Gemma4TextModel`, `Gemma4ForConditionalGeneration` |

### Gemma 4 specifics correctly handled in `gemma4.rs`

- `Gemma4RmsNorm` with `+1` offset (same as Gemma 3)
- `UnweightedRmsNorm` for `v_norm` and router norm
- PLE pipeline: `embed_tokens_per_layer` + `per_layer_model_projection` + `per_layer_projection_norm`
- `Gemma4MoEExpert` with fused `gate_up_proj` + GELU activation (not SiLU)
- Extra config fields extracted from `ModelConfig.extra` serde map:
  `enable_moe_block`, `moe_intermediate_size`, `hidden_size_per_layer_input`,
  `vocab_size_per_layer_input`, `use_double_wide_mlp`, `final_logit_softcap`,
  `sliding_window_pattern`, router-related fields

### Infrastructure that's ready to reuse

- Quantization backends (≈16k LOC across 25+ files):
  AWQ (667 LOC), AWQ-Marlin (252), BitsAndBytes (1341), GGUF (`gguf/` module),
  GPTQ (767 + cuda 468), Marlin (921 + cuda 857), FP8 (956 + cuda 766),
  MxFP4 (641 + cuda 278), CompressedTensors, MoeWNA16, FbgemmFp8, Quark, TorchAO
- `QuantizedWeightLoader` trait (`quantization/weight_loader.rs`, 1580 LOC) —
  dispatches to the right quantized linear based on detected config
- `detect_from_json()` / `detect_from_directory()` in `quantization/mod.rs` —
  auto-detects quantization method from `config.json` or `quantize_config.json`
- `Box<dyn QuantizedLinear>` trait from `quantization/config.rs` —
  unified interface for all quantized linear layers

---

## The Core Gap: Full-Precision vs. Quantized Duality

The project has a hard split between two model variants:

**Full-precision models** (`gemma4.rs`, `qwen3.rs`, `llama.rs`) use `TpLinear`
from `crates/core/src/models/tp_layers.rs`. This enum has **only** three variants:

```rust
pub enum TpLinear {
    Regular(candle_nn::Linear),          // single-GPU BF16/FP32
    ColumnParallel(ColumnParallelLinear), // TP split output dim
    RowParallel(RowParallelLinear),       // TP split input dim
}
```

**There is no quantized variant.** `TpLinear::column_parallel()` and
`TpLinear::row_parallel()` always call `candle_nn::linear(...)` or
`candle_nn::linear_no_bias(...)` under the hood.

**Quantized models** live in separate files named `<name>_quantized.rs`
(`gemma3_quantized.rs`, `mixtral_quantized.rs`, `qwen3_quantized.rs`, etc.).
They use `Box<dyn QuantizedLinear>` directly and take
`weight_loader: &dyn QuantizedWeightLoader` in their constructor.

Dispatch happens in `models/mod.rs::from_config_with_quant()`:

```rust
pub fn from_config_with_quant(
    cfg: &ModelConfig,
    vb: VarBuilder<'static>,
    quant_config: &DetectedQuantConfig,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;
    if quant_config.method == QuantizationMethod::None {
        return from_config(cfg, vb);  // full-precision path → gemma4.rs ✅
    }
    let weight_loader = create_weight_loader_with_params(vb.clone(), quant_config);
    match arch {
        "Qwen3ForCausalLM"                        => QuantizedQwen3ForCausalLM::new(...),
        "LlamaForCausalLM" | ...                  => QuantizedLlamaForCausalLM::new(...),
        "GemmaForCausalLM"                        => QuantizedGemmaForCausalLM::new(...),
        "Gemma2ForCausalLM" | "Gemma2Model"       => QuantizedGemma2ForCausalLM::new(...),
        "Gemma3ForCausalLM" | "Gemma3TextModel"   => QuantizedGemma3ForCausalLM::new(...),
        // ↑ gemma, gemma2, gemma3 exist — NO gemma4 arm ❌
        "MixtralForCausalLM"                      => QuantizedMixtralForCausalLM::new(...),
        // ... many more
    }
}
```

**Result:** feeding any quantized Gemma 4 checkpoint (AWQ/BNB/GGUF) into
vllm-rust hits the fallthrough and errors out. The full-precision path
(`gemma4.rs`) cannot load quantized tensor formats because weight names
and dtypes don't match (`qweight` U32 vs `weight` BF16, etc.).

---

## Implementation Plan

### MVP — Text-only quantized Gemma 4 (required for D&D benchmark)

#### 1. Create `crates/core/src/models/gemma4_quantized.rs`

Mirror the structure of `gemma4.rs` with these substitutions:

| `gemma4.rs` (full precision) | `gemma4_quantized.rs` |
|------------------------------|-----------------------|
| `q_proj: TpLinear` | `q_proj: Box<dyn QuantizedLinear>` |
| `k_proj: TpLinear` | `k_proj: Box<dyn QuantizedLinear>` |
| `v_proj: TpLinear` | `v_proj: Box<dyn QuantizedLinear>` |
| `o_proj: TpLinear` | `o_proj: Box<dyn QuantizedLinear>` |
| `gate_proj: TpLinear` | `gate_proj: Box<dyn QuantizedLinear>` |
| `up_proj: TpLinear` | `up_proj: Box<dyn QuantizedLinear>` |
| `down_proj: TpLinear` | `down_proj: Box<dyn QuantizedLinear>` |
| `lm_head: TpLinear` | `lm_head: Box<dyn QuantizedLinear>` |
| `per_layer_model_projection: Option<TpLinear>` | `Option<Box<dyn QuantizedLinear>>` |
| `TpLinear::column_parallel(in, out, bias, _, vb, pg)?` | `weight_loader.load_linear(in, out, bias, vb.pp("name"))?` |
| `candle_nn::linear_no_bias(...)` | same → `weight_loader.load_linear(...)` |

**Preserve unchanged:**
- `Gemma4RmsNorm` (+1 offset) and `UnweightedRmsNorm` (no learned weight)
- `soft_cap()` helper
- `sliding_window_mask()` helper
- `Gemma4ExtraConfig::from_model_config()` — all extra field extraction
- `is_kv_shared_layer()` logic for `use_double_wide_mlp` / `k_eq_v`
- `Gemma4MoEExpert` architecture (fused `gate_up_proj` + GELU)
- Layer scaling (`layer_scalar` per-layer multiplier)
- Final logit soft cap
- PLE pipeline: `embed_tokens_per_layer` stays as regular embedding
  (embeddings are never quantized in AWQ/GPTQ/BNB), only `per_layer_model_projection`
  becomes `QuantizedLinear`

**Constructor signature** (matches the pattern from `gemma3_quantized.rs`):

```rust
impl QuantizedGemma4ForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> { ... }
}
```

**Single-GPU scope for MVP** — skip `new_with_tp()` variant initially. Add it later
once we verify the basic quantized path works.

**Reference files to read while implementing:**
- `crates/core/src/models/gemma3_quantized.rs` — closest Gemma family quantized model;
  has soft cap, sliding window, Gemma-style RMSNorm, alternating attention pattern
- `crates/core/src/models/mixtral_quantized.rs` — reference for quantized MoE
  (how experts are loaded via QuantizedLinear)
- `crates/core/src/models/gemma4.rs` — the full-precision Gemma 4 we're mirroring
- `crates/core/src/quantization/weight_loader.rs` — to understand `load_linear()` signature and options

**Estimated LOC:** ≈1600-1800 lines (slightly less than `gemma4.rs` since TP variants are deferred).

#### 2. Register the module in `models/mod.rs`

Add at the top (near existing `pub mod gemma4;` and `pub mod gemma3_quantized;`):

```rust
pub mod gemma4_quantized;
```

And in the re-exports section:

```rust
pub use gemma4_quantized::QuantizedGemma4ForCausalLM;
```

#### 3. Add dispatch arm in `from_config_with_quant()`

In `crates/core/src/models/mod.rs`, inside the `match arch` block
(near existing Gemma 3 arm at line 1144):

```rust
"Gemma4ForCausalLM" | "Gemma4TextModel" => Ok(Box::new(
    QuantizedGemma4ForCausalLM::new(cfg, vb, weight_loader.as_ref())?,
)),
```

#### 4. GGUF tensor-name mapping for Gemma 4

**Check first** whether `crates/core/src/quantization/gguf/parser.rs` already has
generic enough name mapping to pick up `blk.N.attn_q.weight` etc. If yes, and
Gemma 4 GGUFs from unsloth/bartowski follow llama.cpp's standard naming, this
may work out of the box. If not, add Gemma 4 entries alongside the existing
Gemma 3 ones.

Gemma 4 specific tensors that need mapping if GGUF support is desired:
- `embed_tokens_per_layer` (PLE embeddings) — may be stored as regular embedding
- `per_layer_model_projection` — per-layer linear
- `per_layer_projection_norm` — RMSNorm weights
- MoE router weights: gate + UnweightedRmsNorm + `router_scale` + `router_gate`
- `layer_scalar` — per-layer learned scalar multiplier

Note: the `Gemma4ForCausalLM` config reads these via `cfg.extra.get(...)`, so
they're already config-aware. The gap is only in GGUF tensor discovery/loading.

#### 5. Integration test

Add to `crates/core/src/models/gemma4_quantized.rs` bottom `#[cfg(test)]` module:

```rust
#[test]
#[ignore = "requires model download"]
fn load_gemma4_e2b_bnb_4bit_from_hf() {
    // Downloads unsloth/gemma-4-E2B-it-unsloth-bnb-4bit,
    // constructs QuantizedGemma4ForCausalLM, runs one forward pass
    // with a dummy input, asserts logits shape = [1, seq_len, vocab_size].
}
```

And a unit test for config extraction similar to `gemma4.rs::test_model_config()`,
verifying the PLE + MoE config paths parse correctly.

---

### Nice to have — Multimodal quantized Gemma 4

#### 6. `crates/core/src/models/gemma4_vlm_quantized.rs`

Thin wrapper: same structure as `gemma4_vlm.rs` but holds
`QuantizedGemma4ForCausalLM` instead of `Gemma4ForCausalLM`.

**Keep these in FP16/BF16** (don't quantize — tiny savings, big complexity):
- SigLIP vision tower (≈150M parameters)
- `Gemma4MultimodalEmbedder` (single Linear)

Add dispatch arm in `from_config_with_quant()` for `"Gemma4ForConditionalGeneration"`.

**Estimated LOC:** ≈200-300 lines (mostly delegation).

#### 7. `tokens_per_image` default verification

`crates/core/src/models/gemma4_vlm.rs:108` hardcodes fallback to `280`.
Verify this matches the actual `vision_config.default_output_length` in
`google/gemma-4-E4B-it/config.json`. If it differs, multimodal token merging
will silently misalign and produce garbage. This is a 1-line fix but
high-impact if wrong.

---

### Out of scope for D&D benchmark (explicitly deferred)

#### 8. Audio encoder

Gemma 4 E2B/E4B supports native audio input via a USM-derived Conformer encoder.
Currently:
- `gemma4_vlm.rs` has **only** a `vision_tower` field — no audio tower
- `crates/core/src/multimodal/audio.rs` (966 LOC) has base infrastructure
  (mel spectrograms, audio loading, feature-gated behind `#[cfg(feature = "audio")]`)
  but no Gemma 4 specific encoder

**Implementation would require** ≈2000+ LOC: audio conformer blocks,
audio embedder (analogous to `Gemma4MultimodalEmbedder` but for audio tokens),
audio token merging in forward path, preprocessor.

**Not needed for text-only D&D NPC simulation.**

#### 9. PLE regression tests

PLE pipeline is subtle (per-layer embeddings routed through projections
then summed with residual stream). Unit test in the new quantized file
using a tiny dummy PLE config to catch shape/naming mismatches that won't
surface until a real checkpoint is loaded. ≈150 LOC.

---

## File-level Change Summary

| File | Action | Est. LOC |
|------|--------|----------|
| `crates/core/src/models/gemma4_quantized.rs` | **Create** | ~1700 |
| `crates/core/src/models/gemma4_vlm_quantized.rs` | **Create** (optional) | ~250 |
| `crates/core/src/models/mod.rs` | Add 2 `pub mod`, 2 re-exports, 2 match arms in `from_config_with_quant()` | ~10 |
| `crates/core/src/quantization/gguf/parser.rs` | Extend tensor name mapping for Gemma 4 PLE + MoE (if GGUF support wanted) | ~50-100 |
| `crates/core/src/models/gemma4_vlm.rs` | Verify `tokens_per_image` default against real config | 1-2 |
| `tests/...` or inline `#[cfg(test)]` | Integration test with real HF checkpoint | ~80 |

**Total for MVP (text-only quantized Gemma 4):** ≈1800-2000 LOC.

---

## Quality Checklist (per CLAUDE.md principles)

- [ ] TDD: Start with unit test for `Gemma4ExtraConfig` parsing in quantized path,
      then dummy-weight test for `QuantizedGemma4ForCausalLM::new()` construction,
      then forward-shape test with zero weights, then integration test with real HF model.
- [ ] `cargo fmt` — zero warnings
- [ ] `cargo clippy --workspace` — zero warnings
- [ ] `cargo check --workspace` after every edit
- [ ] No `unwrap()` in production paths; use `?` everywhere
- [ ] No stubs — if PLE MoE edge case is non-trivial to quantize, document it
      with `TODO` + root cause, don't silently omit
- [ ] Comments only explain **why** (e.g. "QKV norms use offset +1 like Gemma 3,
      not Gemma 2's straight weight"), never **what**
- [ ] Reference the relevant lines in `reference/vllm/vllm/model_executor/models/gemma4.py`
      where decisions about PLE/MoE/router behavior were made

---

## Recommended Starting Point

1. Open `crates/core/src/models/gemma3_quantized.rs` in one pane, `gemma4.rs` in another.
2. Create a new empty `gemma4_quantized.rs` with the module header,
   imports, and `Gemma4ExtraConfig` copy-pasted from `gemma4.rs`.
3. Start at the top: `Gemma4RmsNorm`, `UnweightedRmsNorm`, `soft_cap`, `sliding_window_mask` —
   these are identical to `gemma4.rs`, just copy them over.
4. `Gemma4Attention` struct — replace each `TpLinear::column_parallel(...)`
   with `weight_loader.load_linear(...)`.
5. `Gemma4MoEExpert` — same substitution for `gate_up_proj` and `down_proj`.
6. `Gemma4DecoderLayer` — straightforward composition.
7. `QuantizedGemma4ForCausalLM` — PLE block stays identical except
   `per_layer_model_projection` becomes `QuantizedLinear`.
8. `impl ModelForward` — copy forward logic from `gemma4.rs` verbatim.
9. Wire into `mod.rs` (step 2 and step 3 above).
10. `cargo check` → fix type errors → `cargo test` → fix logic.

---

## WSL2 / Consumer GPU Notes

The upstream Python vLLM bitsandbytes path hangs on WSL2 because
`pin_memory=False` is forced (verified 2026-04-10 in a failed
D&D benchmark run on `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` and
`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`). **This does not apply to vllm-rust** —
weight loading goes through `candle_core::safetensors` + custom
`QuantizedWeightLoader` implementations, bypassing PyTorch's pinned-memory code path entirely.

For a fair 8GB VRAM target, expected working configurations once quantized
Gemma 4 support lands:

| Model | Quantization | Expected VRAM |
|-------|--------------|---------------|
| Gemma 4 E2B | BNB 4-bit (pre-quantized, unsloth) | ≈2.5 GB weights + KV cache |
| Gemma 4 E2B | GGUF Q4_K_M | ≈2.5 GB weights + KV cache |
| Gemma 4 E4B | BNB 4-bit | ≈4 GB weights + KV cache |
| Gemma 4 E4B | GGUF Q4_K_M | ≈3.5 GB weights + KV cache |

All should fit comfortably under 8 GB with max context 2048-4096 and
`gpu_memory_utilization = 0.75`.

---

## References

- Full-precision impl: `crates/core/src/models/gemma4.rs` (1865 LOC)
- Full-precision VLM wrapper: `crates/core/src/models/gemma4_vlm.rs` (466 LOC)
- Python reference: `reference/vllm/vllm/model_executor/models/gemma4.py`
- Python multimodal reference: `reference/vllm/vllm/model_executor/models/gemma4_mm.py`
- Closest pattern for quantized text model: `crates/core/src/models/gemma3_quantized.rs`
- Closest pattern for quantized MoE: `crates/core/src/models/mixtral_quantized.rs`
- Quantization trait: `crates/core/src/quantization/config.rs::QuantizedLinear`
- Weight loader trait: `crates/core/src/quantization/weight_loader.rs::QuantizedWeightLoader`
- Dispatch site: `crates/core/src/models/mod.rs::from_config_with_quant()` (line ~1074)
- Adding-a-model guide: `docs/adding-a-model.md`
