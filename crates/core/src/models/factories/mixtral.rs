//! Mixtral factory.
//!
//! Edge case: TP variant uses a *different struct* (`MixtralTpForCausalLM`)
//! than the single-GPU one (`MixtralForCausalLM`) because the routed
//! experts shard differently. The factory hides that split — callers
//! see one ArchFactory regardless of `pg.world_size()`.

use std::any::Any;

use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::ProcessGroup;
use crate::engine::ModelForward;
use crate::quantization::QuantizedWeightLoader;

use super::super::factory::{ArchFactory, ArchInfo, Capabilities};
use super::super::mixtral::{MixtralForCausalLM, MixtralTpForCausalLM};
use super::super::mixtral_lora::MixtralWithLora;
use super::super::mixtral_quantized::QuantizedMixtralForCausalLM;
use super::super::tp_layers::TpContext;
use super::super::{LoraEnabledModel, ModelError};

pub const MIXTRAL_ARCH_NAMES: &[&str] = &["MixtralForCausalLM"];

static MIXTRAL_INFO: ArchInfo = ArchInfo::new(
    "Mixtral",
    Capabilities::QUANTIZED
        .union(Capabilities::TP)
        .union(Capabilities::LORA)
        .union(Capabilities::MOE),
);

pub struct MixtralArchFactory;
pub static FACTORY: MixtralArchFactory = MixtralArchFactory;

impl ArchFactory for MixtralArchFactory {
    fn arch_names(&self) -> &'static [&'static str] {
        MIXTRAL_ARCH_NAMES
    }
    fn info(&self) -> &'static ArchInfo {
        &MIXTRAL_INFO
    }
    fn build(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(MixtralForCausalLM::new(cfg, vb)?))
    }
    fn build_quant(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Ok(Box::new(QuantizedMixtralForCausalLM::new(
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
        // Different struct than `build`: TP-aware expert sharding.
        Ok(Box::new(MixtralTpForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?))
    }
    fn build_with_lora(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<LoraEnabledModel, ModelError> {
        Ok(LoraEnabledModel::Mixtral(MixtralWithLora::new(cfg, vb)?))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
