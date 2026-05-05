//! Llama family factory.
//!
//! Covers the canonical name plus 16 legacy / fork aliases that the
//! existing match-arm dispatch ([`models::from_config`] in mod.rs ~ line
//! 571) maps to `LlamaForCausalLM`. Same factory answers all of them
//! because the architectures are weight-identical or
//! candle-implementation-identical.
//!
//! Capabilities exercised: build / build_quant / build_with_tp /
//! build_with_pp / build_with_lora. Speculative draft companion
//! `EagleLlamaForCausalLM` lives in its own factory (separate
//! arch_name `EagleLlamaForCausalLM`).

use std::any::Any;

use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{PipelineStageConfig, ProcessGroup};
use crate::engine::ModelForward;
use crate::quantization::QuantizedWeightLoader;

use super::super::factory::{unsupported, ArchFactory, ArchInfo, Capabilities};
use super::super::llama::LlamaForCausalLM;
use super::super::llama_lora::LlamaWithLora;
use super::super::llama_quantized::QuantizedLlamaForCausalLM;
use super::super::tp_layers::TpContext;
use super::super::{LoraEnabledModel, ModelError};

/// All HuggingFace `architectures[]` names this factory answers to.
/// Order matches the legacy match-arm in `from_config` (mod.rs:571–587)
/// so `tests/registry_completeness.rs` can match line-by-line.
pub const LLAMA_ARCH_NAMES: &[&str] = &[
    "LlamaForCausalLM",
    "LlamaModel",
    "LLaMAForCausalLM",
    "AquilaModel",
    "AquilaForCausalLM",
    "CwmForCausalLM",
    "InternLMForCausalLM",
    "InternLM3ForCausalLM",
    "IQuestCoderForCausalLM",
    "XverseForCausalLM",
    "SolarForCausalLM",
    "Fairseq2LlamaForCausalLM",
    "OrionForCausalLM",
    "TeleChatForCausalLM",
    "TeleChat2ForCausalLM",
    "OlmoForCausalLM",
    "SmolLM3ForCausalLM",
];

static LLAMA_INFO: ArchInfo = ArchInfo::new(
    "Llama",
    Capabilities::QUANTIZED
        .union(Capabilities::TP)
        .union(Capabilities::PP)
        .union(Capabilities::LORA),
);

/// Llama (and its alias zoo) factory.
pub struct LlamaArchFactory;

/// `&'static` instance referenced from the registry.
pub static FACTORY: LlamaArchFactory = LlamaArchFactory;

impl ArchFactory for LlamaArchFactory {
    fn arch_names(&self) -> &'static [&'static str] {
        LLAMA_ARCH_NAMES
    }

    fn info(&self) -> &'static ArchInfo {
        &LLAMA_INFO
    }

    fn build(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?))
    }

    fn build_quant(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(QuantizedLlamaForCausalLM::new(
            cfg,
            vb,
            weight_loader,
        )?))
    }

    fn build_with_tp(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(LlamaForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?))
    }

    fn build_with_pp(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
        stage: &PipelineStageConfig,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(LlamaForCausalLM::new_with_pp(cfg, vb, stage)?))
    }

    fn build_with_lora(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<LoraEnabledModel, ModelError> {
        // Only the canonical `"LlamaForCausalLM"` name is wired into
        // LoRA today (see `from_config_with_lora` in mod.rs:1891).
        // Aliases that don't have LoRA support fall through the
        // default-`Unsupported` path because `from_config_with_lora`
        // gates by exact match. We keep this factory's `build_with_lora`
        // unconditional — the dispatcher decides which alias is
        // accepted via `cfg.architectures[0]`.
        let _ = unsupported(self.canonical_name(), "lora-non-canonical");
        Ok(LoraEnabledModel::Llama(LlamaWithLora::new(cfg, vb)?))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_advertises_all_capabilities() {
        let f = &FACTORY;
        let info = f.info();
        assert!(info.supports(Capabilities::TP));
        assert!(info.supports(Capabilities::QUANTIZED));
        assert!(info.supports(Capabilities::LORA));
        assert!(info.supports(Capabilities::PP));
    }

    #[test]
    fn arch_names_match_legacy_match_arm() {
        // Sanity: factory advertises 17 names (16 aliases + canonical).
        // tests/registry_completeness.rs (Phase 2.6) does the
        // line-by-line comparison against the old match-arm.
        assert_eq!(FACTORY.arch_names().len(), 17);
        assert_eq!(FACTORY.arch_names()[0], "LlamaForCausalLM");
    }

    #[test]
    fn factory_object_safe_in_dyn() {
        let f: &dyn ArchFactory = &FACTORY;
        assert_eq!(f.canonical_name(), "Llama");
    }
}
