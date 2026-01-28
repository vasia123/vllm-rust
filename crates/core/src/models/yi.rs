//! Yi model implementation.
//!
//! Yi (01.AI) models are based on Llama architecture with minor modifications:
//! - Same architecture as Llama (RMSNorm, SwiGLU MLP, GQA)
//! - Different RoPE scaling configuration
//! - Typically larger context windows
//!
//! Since Yi uses the same architecture as Llama, this module is a thin wrapper
//! that re-exports the Llama implementation with Yi-specific configuration handling.
//!
//! Reference: https://huggingface.co/01-ai/Yi-6B

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};

// Yi uses the same architecture as Llama
use super::llama::LlamaForCausalLM;

/// Yi model for causal language modeling.
///
/// This is a wrapper around LlamaForCausalLM since Yi uses the same architecture.
/// The main differences are in the model configuration (RoPE scaling, context length).
pub struct YiForCausalLM {
    inner: LlamaForCausalLM,
}

impl YiForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        // Yi uses the same architecture as Llama
        // Configuration differences (RoPE scaling) are handled in ModelConfig
        let inner = LlamaForCausalLM::new(cfg, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.inner
            .forward(input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping)
    }

    pub fn device(&self) -> &Device {
        self.inner.device()
    }
}

impl crate::engine::ModelForward for YiForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        crate::engine::ModelForward::forward_decode_batch(
            &self.inner,
            input_ids,
            sequences,
            kv_cache_mgr,
        )
    }

    fn device(&self) -> &Device {
        self.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            architectures: vec!["YiForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2, // Yi uses GQA
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 4096, // Yi typically has larger context
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 5000000.0, // Yi uses different rope_theta
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
        }
    }

    #[test]
    fn test_yi_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = YiForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "YiForCausalLM should construct with zero weights");
    }

    #[test]
    fn test_yi_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = YiForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_yi_gqa_configuration() {
        // Yi uses GQA like Llama
        let cfg = test_config();
        let gqa_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(gqa_groups, 2, "test config uses GQA with 2 groups");
    }
}
