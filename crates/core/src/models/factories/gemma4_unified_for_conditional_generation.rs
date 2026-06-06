//! Factory for `Gemma4UnifiedForConditionalGeneration` — the released
//! Gemma 4 12B "unified, encoder-free" multimodal architecture
//! (`model_type = "gemma4_unified"`, text `gemma4_unified_text`).
//!
//! This factory wires the **text-only** path. The published EXL3 quants
//! (`turboderp/gemma-4-12B-it-exl3`) carry only text tensors under
//! `model.language_model.*` plus a top-level `lm_head` — no vision weights
//! — so for text generation the model is a plain causal LM whose tensors
//! live one namespace deeper than a standalone `Gemma4ForCausalLM`.
//!
//! Both the quantized and unquantized paths therefore construct the Gemma 4
//! text backbone at the `model.language_model` root:
//!   - quantized: `QuantizedGemma4ForCausalLM::new_at_model_root` with a
//!     `RemappingWeightLoader` that rewrites the model's `"model.X"` load
//!     paths to the checkpoint's `"model.language_model.X"` (the top-level
//!     `lm_head` is left untouched by the remap).
//!   - unquantized: `Gemma4ForCausalLM::new_with_tp_at_root` with the
//!     VarBuilder positioned at `model.language_model` and the lm_head root
//!     at the checkpoint root.
//!
//! Full unified multimodal (vision) support is a separate, larger effort;
//! it is intentionally not wired here because the text-only quant cannot
//! exercise it (no vision tensors exist in the checkpoint).

#![allow(unused_imports, unused_variables)]

use std::any::Any;

use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::ModelForward;

use super::super::factory::{ArchFactory, ArchInfo, Capabilities};
use super::super::*;

pub const ARCH_NAMES: &[&str] = &[
    "Gemma4UnifiedForConditionalGeneration",
    "Gemma4UnifiedTextModel",
];

static INFO: ArchInfo = ArchInfo::new("Gemma4Unified", Capabilities::QUANTIZED);

pub struct Gemma4UnifiedForConditionalGenerationArchFactory;
pub static FACTORY: Gemma4UnifiedForConditionalGenerationArchFactory =
    Gemma4UnifiedForConditionalGenerationArchFactory;

impl ArchFactory for Gemma4UnifiedForConditionalGenerationArchFactory {
    fn arch_names(&self) -> &'static [&'static str] {
        ARCH_NAMES
    }
    fn info(&self) -> &'static ArchInfo {
        &INFO
    }

    fn build(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        // Text backbone at `model.language_model.*`; untied lm_head (when
        // present) sits at the checkpoint root.
        let vb_lm = vb.pp("model").pp("language_model");
        let model = gemma4::Gemma4ForCausalLM::new_with_tp_at_root(
            cfg,
            vb_lm,
            Some(vb),
            &crate::distributed::LocalProcessGroup::new(),
            tp_layers::TpContext::single_gpu(),
        )?;
        Ok(Box::new(model))
    }

    fn build_quant(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn crate::quantization::QuantizedWeightLoader,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        let vb_lm = vb.pp("model").pp("language_model");
        let remap = gemma4_vlm_quantized::RemappingWeightLoader::new(
            weight_loader,
            "model",
            "model.language_model",
        );
        let model =
            gemma4_quantized::QuantizedGemma4ForCausalLM::new_at_model_root(cfg, vb_lm, &remap)?;
        Ok(Box::new(model))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
