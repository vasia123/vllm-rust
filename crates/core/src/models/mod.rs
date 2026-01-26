pub mod llama;
pub mod llama_lora;
pub mod qwen3;
pub mod qwen3_lora;

pub use llama::LlamaForCausalLM;
pub use llama_lora::LlamaWithLora;
pub use qwen3::Qwen3ForCausalLM;
pub use qwen3_lora::Qwen3WithLora;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::lora::{LoraContext, LoraModel};

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("model load error: {0}")]
    Load(#[from] candle_core::Error),
}

/// Construct the appropriate model from config.architectures[0].
pub fn from_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = cfg
        .architectures
        .first()
        .ok_or_else(|| ModelError::UnsupportedArchitecture("empty architectures list".into()))?;
    match arch.as_str() {
        "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new(cfg, vb)?)),
        "LlamaForCausalLM" => Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Construct a LoRA-enabled model from config.architectures[0].
///
/// The returned model supports per-request adapter selection through LoraContext.
/// Call `register_lora()` on the model to add adapters before inference.
/// The model implements `ModelForward` and can be used with the engine.
pub fn from_config_with_lora(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<LoraEnabledModel, ModelError> {
    let arch = cfg
        .architectures
        .first()
        .ok_or_else(|| ModelError::UnsupportedArchitecture("empty architectures list".into()))?;
    match arch.as_str() {
        "Qwen3ForCausalLM" => Ok(LoraEnabledModel::Qwen3(Qwen3WithLora::new(cfg, vb)?)),
        "LlamaForCausalLM" => Ok(LoraEnabledModel::Llama(LlamaWithLora::new(cfg, vb)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// A LoRA-enabled model that can be used with the engine.
///
/// This enum wraps concrete LoRA-enabled model types and provides both
/// `ModelForward` for engine compatibility and LoRA-specific methods
/// for adapter registration.
pub enum LoraEnabledModel {
    Llama(LlamaWithLora),
    Qwen3(Qwen3WithLora),
}

impl LoraEnabledModel {
    /// Register a LoRA adapter with the model.
    pub fn register_lora(&mut self, lora_model: &LoraModel) {
        match self {
            LoraEnabledModel::Llama(m) => m.register_lora(lora_model),
            LoraEnabledModel::Qwen3(m) => m.register_lora(lora_model),
        }
    }

    /// Get the list of registered LoRA adapter names.
    pub fn lora_adapters(&self) -> Vec<String> {
        match self {
            LoraEnabledModel::Llama(m) => m.lora_adapters(),
            LoraEnabledModel::Qwen3(m) => m.lora_adapters(),
        }
    }
}

impl ModelForward for LoraEnabledModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        match self {
            LoraEnabledModel::Llama(m) => {
                ModelForward::forward(m, input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping)
            }
            LoraEnabledModel::Qwen3(m) => {
                ModelForward::forward(m, input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping)
            }
        }
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        match self {
            LoraEnabledModel::Llama(m) => {
                ModelForward::forward_decode_batch(m, input_ids, sequences, kv_cache_mgr)
            }
            LoraEnabledModel::Qwen3(m) => {
                ModelForward::forward_decode_batch(m, input_ids, sequences, kv_cache_mgr)
            }
        }
    }

    fn forward_decode_batch_with_ctx(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        ctx: &crate::engine::ForwardContext,
    ) -> candle_core::Result<Tensor> {
        match self {
            LoraEnabledModel::Llama(m) => {
                ModelForward::forward_decode_batch_with_ctx(m, input_ids, sequences, kv_cache_mgr, ctx)
            }
            LoraEnabledModel::Qwen3(m) => {
                ModelForward::forward_decode_batch_with_ctx(m, input_ids, sequences, kv_cache_mgr, ctx)
            }
        }
    }

    fn supports_cuda_graphs(&self) -> bool {
        match self {
            LoraEnabledModel::Llama(m) => ModelForward::supports_cuda_graphs(m),
            LoraEnabledModel::Qwen3(m) => ModelForward::supports_cuda_graphs(m),
        }
    }

    fn supports_lora(&self) -> bool {
        true // LoRA-enabled models always support LoRA
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> candle_core::Result<Tensor> {
        match self {
            LoraEnabledModel::Llama(m) => ModelForward::forward_with_lora(
                m,
                input_ids,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
                lora_ctx,
            ),
            LoraEnabledModel::Qwen3(m) => ModelForward::forward_with_lora(
                m,
                input_ids,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
                lora_ctx,
            ),
        }
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> candle_core::Result<Tensor> {
        match self {
            LoraEnabledModel::Llama(m) => {
                ModelForward::forward_decode_batch_with_lora(m, input_ids, sequences, kv_cache_mgr, lora_ctx)
            }
            LoraEnabledModel::Qwen3(m) => {
                ModelForward::forward_decode_batch_with_lora(m, input_ids, sequences, kv_cache_mgr, lora_ctx)
            }
        }
    }

    fn device(&self) -> &Device {
        match self {
            LoraEnabledModel::Llama(m) => ModelForward::device(m),
            LoraEnabledModel::Qwen3(m) => ModelForward::device(m),
        }
    }
}

/// Trait for models that support LoRA adapters.
///
/// This extends the base ModelForward trait with LoRA-specific methods.
pub trait ModelForwardWithLora: Send + Sync {
    /// Register a LoRA adapter with the model.
    fn register_lora(&mut self, lora_model: &crate::lora::LoraModel);

    /// Forward pass with optional LoRA adapter.
    fn forward_with_lora(
        &self,
        input_ids: &candle_core::Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut crate::kv_cache::KVCacheManager,
        block_table: &crate::kv_cache::BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &crate::lora::LoraContext,
    ) -> candle_core::Result<candle_core::Tensor>;

    /// Batched decode forward with optional LoRA adapter.
    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &candle_core::Tensor,
        sequences: &[crate::engine::DecodeSequenceMetadata],
        kv_cache_mgr: &mut crate::kv_cache::KVCacheManager,
        lora_ctx: &crate::lora::LoraContext,
    ) -> candle_core::Result<candle_core::Tensor>;

    /// Get the list of registered LoRA adapter names.
    fn lora_adapters(&self) -> Vec<String>;

    /// Get the device this model is on.
    fn device(&self) -> &candle_core::Device;
}
