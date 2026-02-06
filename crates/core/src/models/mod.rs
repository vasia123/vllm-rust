#[macro_use]
mod macros;
pub mod baichuan;
pub mod bert;
pub mod bloom;
pub mod cohere;
pub mod dbrx;
pub mod deepseek;
pub mod deepseek_quantized;
pub mod exaone;
pub mod falcon;
pub mod gemma;
pub mod gemma2;
pub mod gemma2_quantized;
pub mod gemma3;
pub mod gemma3_quantized;
pub mod gemma_lora;
pub mod gemma_quantized;
pub mod glm;
pub mod glm4;
pub mod glm4_moe;
pub mod gpt2;
pub mod gpt_neox;
pub mod internlm2;
pub mod jamba;
pub mod llama;
pub mod llama_lora;
pub mod llama_quantized;
pub mod llava;
pub mod mamba;
pub mod mamba2;
pub mod mistral;
pub mod mistral_lora;
pub mod mistral_quantized;
pub mod mixtral;
pub mod mixtral_quantized;
pub mod mpt;
pub mod olmo2;
pub mod persimmon;
pub mod phi;
pub mod phi3;
pub mod phi3_lora;
pub mod phi3_quantized;
pub mod qwen2;
pub mod qwen2_lora;
pub mod qwen2_moe;
pub mod qwen2_quantized;
pub mod qwen3;
pub mod qwen3_lora;
pub mod qwen3_moe;
pub mod qwen3_quantized;
pub mod registry;
pub mod starcoder2;
pub mod tp_layers;
pub mod yi;

// Re-export tensor parallelism abstractions
pub use tp_layers::{TpContext, TpEmbedding, TpGeGluMlp, TpGeluMlp, TpLinear, TpSwiGluMlp};

pub use baichuan::BaichuanForCausalLM;
pub use bert::BertForSequenceEmbedding;
pub use bloom::BloomForCausalLM;
pub use cohere::CohereForCausalLM;
pub use dbrx::DbrxForCausalLM;
pub use deepseek::DeepSeekForCausalLM;
pub use deepseek_quantized::QuantizedDeepSeekForCausalLM;
pub use exaone::ExaoneForCausalLM;
pub use falcon::FalconForCausalLM;
pub use gemma::GemmaForCausalLM;
pub use gemma2::Gemma2ForCausalLM;
pub use gemma2_quantized::QuantizedGemma2ForCausalLM;
pub use gemma3::Gemma3ForCausalLM;
pub use gemma3_quantized::QuantizedGemma3ForCausalLM;
pub use gemma_lora::GemmaWithLora;
pub use gemma_quantized::QuantizedGemmaForCausalLM;
pub use glm::GlmForCausalLM;
pub use glm4::Glm4ForCausalLM;
pub use glm4_moe::Glm4MoeForCausalLM;
pub use gpt2::GPT2LMHeadModel;
pub use gpt_neox::GPTNeoXForCausalLM;
pub use internlm2::InternLM2ForCausalLM;
pub use jamba::JambaForCausalLM;
pub use llama::LlamaForCausalLM;
pub use llama_lora::LlamaWithLora;
pub use llama_quantized::QuantizedLlamaForCausalLM;
pub use llava::LLaVAForConditionalGeneration;
pub use mamba::MambaForCausalLM;
pub use mamba2::Mamba2ForCausalLM;
pub use mistral::MistralForCausalLM;
pub use mistral_lora::MistralWithLora;
pub use mistral_quantized::QuantizedMistralForCausalLM;
pub use mixtral::{MixtralForCausalLM, MixtralTpForCausalLM};
pub use mixtral_quantized::QuantizedMixtralForCausalLM;
pub use mpt::MptForCausalLM;
pub use olmo2::Olmo2ForCausalLM;
pub use persimmon::PersimmonForCausalLM;
pub use phi::PhiForCausalLM;
pub use phi3::Phi3ForCausalLM;
pub use phi3_lora::Phi3WithLora;
pub use phi3_quantized::QuantizedPhi3ForCausalLM;
pub use qwen2::Qwen2ForCausalLM;
pub use qwen2_lora::Qwen2WithLora;
pub use qwen2_moe::Qwen2MoeForCausalLM;
pub use qwen2_quantized::QuantizedQwen2ForCausalLM;
pub use qwen3::Qwen3ForCausalLM;
pub use qwen3_lora::Qwen3WithLora;
pub use qwen3_moe::Qwen3MoeForCausalLM;
pub use qwen3_quantized::QuantizedQwen3ForCausalLM;
pub use registry::{
    find_architecture, supported_architectures, ArchitectureInfo, ModelCapabilities,
};
pub use starcoder2::StarCoder2ForCausalLM;
pub use yi::YiForCausalLM;

use std::path::Path;

use candle_core::Device;
use candle_nn::VarBuilder;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::ModelForward;
use crate::kv_cache::{CacheConfig, CacheError, KVCacheDtype, KVCacheManager, MLACacheConfig};
use crate::lora::LoraModel;
use crate::quantization::{
    create_weight_loader_with_params, detect_from_directory, DetectedQuantConfig,
    QuantizationMethod,
};

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("model load error: {0}")]
    Load(#[from] candle_core::Error),
}

/// Extract the architecture identifier from config, returning an error on empty list.
fn get_arch(cfg: &ModelConfig) -> Result<&str, ModelError> {
    cfg.architectures
        .first()
        .map(|s| s.as_str())
        .ok_or_else(|| ModelError::UnsupportedArchitecture("empty architectures list".into()))
}

/// Construct the appropriate model from config.architectures[0].
pub fn from_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
            Ok(Box::new(DeepSeekForCausalLM::new(cfg, vb)?))
        }
        "GemmaForCausalLM" => Ok(Box::new(GemmaForCausalLM::new(cfg, vb)?)),
        "Gemma2ForCausalLM" => Ok(Box::new(Gemma2ForCausalLM::new(cfg, vb)?)),
        "Gemma3ForCausalLM" => Ok(Box::new(Gemma3ForCausalLM::new(cfg, vb)?)),
        "JambaForCausalLM" => Ok(Box::new(JambaForCausalLM::new(cfg, vb)?)),
        "LlamaForCausalLM" => Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?)),
        "MistralForCausalLM" => Ok(Box::new(MistralForCausalLM::new(cfg, vb)?)),
        "MixtralForCausalLM" => Ok(Box::new(MixtralForCausalLM::new(cfg, vb)?)),
        "Qwen2ForCausalLM" => Ok(Box::new(Qwen2ForCausalLM::new(cfg, vb)?)),
        "Qwen2MoeForCausalLM" => Ok(Box::new(Qwen2MoeForCausalLM::new(cfg, vb)?)),
        "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new(cfg, vb)?)),
        "Qwen3MoeForCausalLM" => Ok(Box::new(Qwen3MoeForCausalLM::new(cfg, vb)?)),
        "Phi3ForCausalLM" => Ok(Box::new(Phi3ForCausalLM::new(cfg, vb)?)),
        "Olmo2ForCausalLM" => Ok(Box::new(Olmo2ForCausalLM::new(cfg, vb)?)),
        "GPT2LMHeadModel" => Ok(Box::new(GPT2LMHeadModel::new(cfg, vb)?)),
        "GlmForCausalLM" => Ok(Box::new(GlmForCausalLM::new(cfg, vb)?)),
        "Glm4ForCausalLM" => Ok(Box::new(Glm4ForCausalLM::new(cfg, vb)?)),
        "Glm4MoeForCausalLM" => Ok(Box::new(Glm4MoeForCausalLM::new(cfg, vb)?)),
        "BaichuanForCausalLM" => Ok(Box::new(BaichuanForCausalLM::new(cfg, vb)?)),
        "InternLM2ForCausalLM" => Ok(Box::new(InternLM2ForCausalLM::new(cfg, vb)?)),
        "CohereForCausalLM" => Ok(Box::new(CohereForCausalLM::new(cfg, vb)?)),
        "GPTNeoXForCausalLM" => Ok(Box::new(GPTNeoXForCausalLM::new(cfg, vb)?)),
        "Starcoder2ForCausalLM" => Ok(Box::new(StarCoder2ForCausalLM::new(cfg, vb)?)),
        "BloomForCausalLM" => Ok(Box::new(BloomForCausalLM::new(cfg, vb)?)),
        "FalconForCausalLM" => Ok(Box::new(FalconForCausalLM::new(cfg, vb)?)),
        "PhiForCausalLM" => Ok(Box::new(PhiForCausalLM::new(cfg, vb)?)),
        "YiForCausalLM" => Ok(Box::new(YiForCausalLM::new(cfg, vb)?)),
        "MptForCausalLM" => Ok(Box::new(MptForCausalLM::new(cfg, vb)?)),
        "PersimmonForCausalLM" => Ok(Box::new(PersimmonForCausalLM::new(cfg, vb)?)),
        "ExaoneForCausalLM" => Ok(Box::new(ExaoneForCausalLM::new(cfg, vb)?)),
        "DbrxForCausalLM" => Ok(Box::new(DbrxForCausalLM::new(cfg, vb)?)),
        "MambaForCausalLM" | "FalconMambaForCausalLM" => {
            Ok(Box::new(MambaForCausalLM::new(cfg, vb)?))
        }
        "Mamba2ForCausalLM" => Ok(Box::new(Mamba2ForCausalLM::new(cfg, vb)?)),
        "LlavaForConditionalGeneration" | "LlavaNextForConditionalGeneration" => Ok(Box::new(
            LLaVAForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "BertModel" | "BertForMaskedLM" | "BertForSequenceClassification" => {
            Ok(Box::new(BertForSequenceEmbedding::new(cfg, vb)?))
        }
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Construct a quantized model with automatic quantization detection.
///
/// This function detects the quantization method from the model directory
/// (by reading config.json and quantize_config.json) and returns the appropriate
/// quantized model variant.
///
/// If no quantization is detected, it falls back to the unquantized model.
///
/// # Arguments
/// * `cfg` - Model configuration
/// * `vb` - VarBuilder for loading weights
/// * `model_dir` - Path to the model directory for quantization detection
///
/// # Returns
/// A boxed model implementing ModelForward
pub fn from_config_quantized(
    cfg: &ModelConfig,
    vb: VarBuilder<'static>,
    model_dir: &Path,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let detected = detect_from_directory(model_dir);
    from_config_with_quant(cfg, vb, &detected)
}

/// Construct a model with explicit quantization configuration.
///
/// # Arguments
/// * `cfg` - Model configuration
/// * `vb` - VarBuilder for loading weights
/// * `quant_config` - Detected quantization configuration
pub fn from_config_with_quant(
    cfg: &ModelConfig,
    vb: VarBuilder<'static>,
    quant_config: &DetectedQuantConfig,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;

    if quant_config.method == QuantizationMethod::None {
        return from_config(cfg, vb);
    }

    let weight_loader = create_weight_loader_with_params(vb.clone(), quant_config);

    match arch {
        "Qwen3ForCausalLM" => Ok(Box::new(QuantizedQwen3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "LlamaForCausalLM" => Ok(Box::new(QuantizedLlamaForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GemmaForCausalLM" => Ok(Box::new(QuantizedGemmaForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Gemma2ForCausalLM" => Ok(Box::new(QuantizedGemma2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "MistralForCausalLM" => Ok(Box::new(QuantizedMistralForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Qwen2ForCausalLM" => Ok(Box::new(QuantizedQwen2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Phi3ForCausalLM" => Ok(Box::new(QuantizedPhi3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Gemma3ForCausalLM" => Ok(Box::new(QuantizedGemma3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "MixtralForCausalLM" => Ok(Box::new(QuantizedMixtralForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => Ok(Box::new(
            QuantizedDeepSeekForCausalLM::new(cfg, vb, weight_loader.as_ref())?,
        )),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Get the detected quantization method for a model directory.
///
/// This is useful for checking quantization before loading.
pub fn detect_quantization(model_dir: &Path) -> DetectedQuantConfig {
    detect_from_directory(model_dir)
}

/// Create a KVCacheManager appropriate for the given model configuration.
///
/// For MLA models (DeepSeek V2/V3), this creates a compressed MLA cache.
/// For standard models (Llama, Qwen, etc.), this creates a standard paged KV cache.
///
/// # Arguments
/// * `cfg` - Model configuration (determines cache type based on architecture)
/// * `block_size` - Tokens per block
/// * `num_blocks` - Total number of cache blocks
/// * `dtype` - Data type for cache storage
/// * `device` - Target device
///
/// # Returns
/// A KVCacheManager configured for the model type
pub fn create_cache_manager(
    cfg: &ModelConfig,
    block_size: usize,
    num_blocks: usize,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<KVCacheManager, CacheError> {
    create_cache_manager_with_tp(cfg, block_size, num_blocks, dtype, device, 1)
}

/// Create a KVCacheManager with tensor parallelism support.
///
/// For tensor-parallel execution, each GPU only needs to cache its local KV heads.
/// This function divides the head count by tp_size to allocate the correct amount
/// of cache per GPU.
///
/// # Arguments
/// * `cfg` - Model configuration (determines cache type based on architecture)
/// * `block_size` - Tokens per block
/// * `num_blocks` - Total number of cache blocks
/// * `dtype` - Data type for cache storage
/// * `device` - Target device
/// * `tp_size` - Tensor parallel world size (1 for single GPU)
///
/// # Returns
/// A KVCacheManager configured for the model type and TP configuration
///
/// # Panics
/// Panics if num_kv_heads is not divisible by tp_size
pub fn create_cache_manager_with_tp(
    cfg: &ModelConfig,
    block_size: usize,
    num_blocks: usize,
    dtype: candle_core::DType,
    device: &Device,
    tp_size: usize,
) -> Result<KVCacheManager, CacheError> {
    if cfg.is_mla_model() {
        let mla_dims = cfg
            .mla_dims()
            .expect("MLA model must have mla_dims in config");

        let local_num_heads = cfg.num_attention_heads / tp_size;
        assert_eq!(
            cfg.num_attention_heads % tp_size,
            0,
            "num_attention_heads ({}) must be divisible by tp_size ({})",
            cfg.num_attention_heads,
            tp_size
        );

        let mla_config = MLACacheConfig {
            kv_lora_rank: mla_dims.kv_lora_rank,
            qk_rope_head_dim: mla_dims.qk_rope_head_dim,
            qk_nope_head_dim: mla_dims.qk_nope_head_dim,
            v_head_dim: mla_dims.v_head_dim,
            num_heads: local_num_heads,
            block_size,
            num_blocks,
            num_layers: cfg.num_hidden_layers,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
        };
        KVCacheManager::new_mla(&mla_config)
    } else {
        let local_num_kv_heads = cfg.num_key_value_heads / tp_size;
        assert_eq!(
            cfg.num_key_value_heads % tp_size,
            0,
            "num_key_value_heads ({}) must be divisible by tp_size ({})",
            cfg.num_key_value_heads,
            tp_size
        );

        let cache_config = CacheConfig {
            block_size,
            num_blocks,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: local_num_kv_heads,
            head_dim: cfg.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        KVCacheManager::new(&cache_config)
    }
}

// ─── Tensor Parallelism Support ──────────────────────────────────────────────

/// Construct a model with tensor parallelism support.
///
/// This function creates a model that distributes computation across multiple GPUs
/// using tensor parallelism. Each GPU holds a shard of the model weights and
/// performs partial computation, with collective operations combining results.
///
/// # Arguments
/// * `cfg` - Model configuration
/// * `vb` - VarBuilder for loading weights
/// * `pg` - Process group defining the TP topology
/// * `tp_ctx` - Tensor parallelism context with communicator
///
/// # Returns
/// A boxed model implementing ModelForward
///
/// # Supported Architectures
/// Currently supports: LlamaForCausalLM, MistralForCausalLM, MixtralForCausalLM,
/// Qwen2ForCausalLM, Qwen2MoeForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM,
/// GlmForCausalLM, Glm4ForCausalLM, Glm4MoeForCausalLM, GemmaForCausalLM, Gemma2ForCausalLM,
/// Gemma3ForCausalLM, Phi3ForCausalLM, Olmo2ForCausalLM, BaichuanForCausalLM,
/// InternLM2ForCausalLM, CohereForCausalLM, GPTNeoXForCausalLM, Starcoder2ForCausalLM,
/// BloomForCausalLM, FalconForCausalLM, PhiForCausalLM, YiForCausalLM, GPT2LMHeadModel,
/// ExaoneForCausalLM, PersimmonForCausalLM, MptForCausalLM, DbrxForCausalLM
/// Other architectures fall back to single-GPU (warning logged)
pub fn from_config_with_tp(
    cfg: &ModelConfig,
    vb: VarBuilder,
    pg: &dyn ProcessGroup,
    tp_ctx: TpContext,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;

    if pg.world_size() == 1 {
        return from_config(cfg, vb);
    }

    match arch {
        "LlamaForCausalLM" => Ok(Box::new(LlamaForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MistralForCausalLM" => Ok(Box::new(MistralForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MixtralForCausalLM" => Ok(Box::new(MixtralTpForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen2ForCausalLM" => Ok(Box::new(Qwen2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen2MoeForCausalLM" => Ok(Box::new(Qwen2MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen3MoeForCausalLM" => Ok(Box::new(Qwen3MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GlmForCausalLM" => Ok(Box::new(GlmForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Glm4ForCausalLM" => Ok(Box::new(Glm4ForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Glm4MoeForCausalLM" => Ok(Box::new(Glm4MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GemmaForCausalLM" => Ok(Box::new(GemmaForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Gemma2ForCausalLM" => Ok(Box::new(Gemma2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Phi3ForCausalLM" => Ok(Box::new(Phi3ForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Olmo2ForCausalLM" => Ok(Box::new(Olmo2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "BaichuanForCausalLM" => Ok(Box::new(BaichuanForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "InternLM2ForCausalLM" => Ok(Box::new(InternLM2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "CohereForCausalLM" => Ok(Box::new(CohereForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GPTNeoXForCausalLM" => Ok(Box::new(GPTNeoXForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Starcoder2ForCausalLM" => Ok(Box::new(StarCoder2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "BloomForCausalLM" => Ok(Box::new(BloomForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "FalconForCausalLM" => Ok(Box::new(FalconForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PhiForCausalLM" => Ok(Box::new(PhiForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "YiForCausalLM" => Ok(Box::new(YiForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "GPT2LMHeadModel" => Ok(Box::new(GPT2LMHeadModel::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Gemma3ForCausalLM" => Ok(Box::new(Gemma3ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "ExaoneForCausalLM" => Ok(Box::new(ExaoneForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PersimmonForCausalLM" => Ok(Box::new(PersimmonForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MptForCausalLM" => Ok(Box::new(MptForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "DbrxForCausalLM" => Ok(Box::new(DbrxForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        other => {
            tracing::warn!(
                architecture = other,
                "TP not yet implemented for this architecture, using single-GPU fallback"
            );
            from_config(cfg, vb)
        }
    }
}

/// Construct a model using default single-GPU tensor parallelism context.
///
/// This is a convenience function that wraps `from_config_with_tp` with
/// single-GPU defaults. Use this when you want TP-aware models but are
/// running on a single GPU.
pub fn from_config_tp_aware(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let pg = LocalProcessGroup::new();
    let tp_ctx = TpContext::single_gpu();
    from_config_with_tp(cfg, vb, &pg, tp_ctx)
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
    let arch = get_arch(cfg)?;
    match arch {
        "Qwen3ForCausalLM" => Ok(LoraEnabledModel::Qwen3(Qwen3WithLora::new(cfg, vb)?)),
        "LlamaForCausalLM" => Ok(LoraEnabledModel::Llama(LlamaWithLora::new(cfg, vb)?)),
        "MistralForCausalLM" => Ok(LoraEnabledModel::Mistral(MistralWithLora::new(cfg, vb)?)),
        "Qwen2ForCausalLM" => Ok(LoraEnabledModel::Qwen2(Qwen2WithLora::new(cfg, vb)?)),
        "GemmaForCausalLM" => Ok(LoraEnabledModel::Gemma(GemmaWithLora::new(cfg, vb)?)),
        "Phi3ForCausalLM" => Ok(LoraEnabledModel::Phi3(Phi3WithLora::new(cfg, vb)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

delegate_model_forward! {
    /// A LoRA-enabled model that can be used with the engine.
    ///
    /// This enum wraps concrete LoRA-enabled model types and provides both
    /// `ModelForward` for engine compatibility and LoRA-specific methods
    /// for adapter registration.
    pub enum LoraEnabledModel {
        Llama(LlamaWithLora),
        Qwen3(Qwen3WithLora),
        Mistral(MistralWithLora),
        Qwen2(Qwen2WithLora),
        Gemma(GemmaWithLora),
        Phi3(Phi3WithLora),
    }
}

impl LoraEnabledModel {
    /// Register a LoRA adapter with the model.
    pub fn register_lora(&mut self, lora_model: &LoraModel) {
        match self {
            LoraEnabledModel::Llama(m) => m.register_lora(lora_model),
            LoraEnabledModel::Qwen3(m) => m.register_lora(lora_model),
            LoraEnabledModel::Mistral(m) => m.register_lora(lora_model),
            LoraEnabledModel::Qwen2(m) => m.register_lora(lora_model),
            LoraEnabledModel::Gemma(m) => m.register_lora(lora_model),
            LoraEnabledModel::Phi3(m) => m.register_lora(lora_model),
        }
    }

    /// Get the list of registered LoRA adapter names.
    pub fn lora_adapters(&self) -> Vec<String> {
        match self {
            LoraEnabledModel::Llama(m) => m.lora_adapters(),
            LoraEnabledModel::Qwen3(m) => m.lora_adapters(),
            LoraEnabledModel::Mistral(m) => m.lora_adapters(),
            LoraEnabledModel::Qwen2(m) => m.lora_adapters(),
            LoraEnabledModel::Gemma(m) => m.lora_adapters(),
            LoraEnabledModel::Phi3(m) => m.lora_adapters(),
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
