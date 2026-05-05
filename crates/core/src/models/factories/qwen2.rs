//! Qwen2 factory (text model + quantized + TP + LoRA; no PP).

use std::any::Any;

use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::ProcessGroup;
use crate::engine::ModelForward;
use crate::quantization::QuantizedWeightLoader;

use super::super::factory::{ArchFactory, ArchInfo, Capabilities};
use super::super::qwen2::Qwen2ForCausalLM;
use super::super::qwen2_lora::Qwen2WithLora;
use super::super::qwen2_quantized::QuantizedQwen2ForCausalLM;
use super::super::tp_layers::TpContext;
use super::super::{LoraEnabledModel, ModelError};

pub const QWEN2_ARCH_NAMES: &[&str] = &["Qwen2ForCausalLM", "Qwen2Model"];

static QWEN2_INFO: ArchInfo = ArchInfo::new(
    "Qwen2",
    Capabilities::QUANTIZED
        .union(Capabilities::TP)
        .union(Capabilities::LORA),
);

pub struct Qwen2ArchFactory;
pub static FACTORY: Qwen2ArchFactory = Qwen2ArchFactory;

impl ArchFactory for Qwen2ArchFactory {
    fn arch_names(&self) -> &'static [&'static str] {
        QWEN2_ARCH_NAMES
    }
    fn info(&self) -> &'static ArchInfo {
        &QWEN2_INFO
    }
    fn build(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(Qwen2ForCausalLM::new(cfg, vb)?))
    }
    fn build_quant(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(QuantizedQwen2ForCausalLM::new(
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
        Ok(Box::new(Qwen2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?))
    }
    fn build_with_lora(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<LoraEnabledModel, ModelError> {
        Ok(LoraEnabledModel::Qwen2(Qwen2WithLora::new(cfg, vb)?))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
