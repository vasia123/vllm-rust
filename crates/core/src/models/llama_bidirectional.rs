//! Bidirectional Llama model for text embedding.
//!
//! Same decoder stack as LlamaForCausalLM but with bidirectional attention
//! (no causal mask). Replaces the LM head with pooling for embedding tasks.
//!
//! Used by models like parasail-ai/GritLM-7B-vllm that set `is_causal=False`
//! in HuggingFace config.
//!
//! Key difference from standard Llama:
//! - No causal mask applied during attention (all tokens see all tokens)
//! - No LM head (returns pooled hidden states instead of logits)
//! - Implements ModelForEmbedding for embedding generation

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{config::CacheConfig, quantization::KVCacheDtype};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm};

use super::llama::LlamaDecoderLayer;
use super::tp_layers::{TpContext, TpEmbedding};

// ─── Bidirectional Llama Model ──────────────────────────────────────────────

/// Llama model with bidirectional attention for embedding tasks.
///
/// Structurally identical to LlamaForCausalLM but:
/// 1. Never applies causal masking (all tokens attend to all tokens)
/// 2. No LM head — returns hidden states for pooling
/// 3. Configurable pooling strategy (Mean, CLS, LastToken)
pub struct LlamaBidirectionalModel {
    embed_tokens: TpEmbedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    tp_ctx: TpContext,
    hidden_size: usize,
    max_position_embeddings: usize,
    pooling: PoolingStrategy,
    device: Device,
    // Stored for temporary cache creation in embed().
    num_kv_heads: usize,
    head_dim: usize,
    dtype: candle_core::DType,
}

impl LlamaBidirectionalModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Determine pooling strategy from config
        let pooling = cfg
            .extra
            .get("pooling")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "cls" => PoolingStrategy::Cls,
                "last" => PoolingStrategy::LastToken,
                _ => PoolingStrategy::Mean,
            })
            .unwrap_or(PoolingStrategy::Mean);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            tp_ctx,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            pooling,
            device: vb.device().clone(),
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: vb.dtype(),
        })
    }

    /// Run the Llama decoder stack without causal masking and return hidden states.
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Bidirectional: never apply causal mask
        let attention_mask: Option<&Tensor> = None;

        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask,
                0, // seqlen_offset=0 for full-sequence embedding
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }
        self.norm.forward(&xs)
    }
}

impl crate::engine::ModelForward for LlamaBidirectionalModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_hidden(input_ids, kv_cache_mgr, block_table, slot_mapping)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        // Bidirectional model is non-autoregressive; process all tokens at once
        let block_table = BlockTable::new(16);
        self.forward_hidden(input_ids, kv_cache_mgr, &block_table, &[])
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for LlamaBidirectionalModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Prefill attention only supports batch_size=1 (paged cache write is
        // per-sequence). Process each item independently and stack.
        let block_size = seq_len.max(1);
        let cache_config = CacheConfig {
            block_size,
            num_blocks: 2, // 1 active + 1 spare (pool requires ≥1 free)
            num_layers: self.layers.len(),
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
            device: self.device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };

        let mut outputs = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let ids_b = input_ids.narrow(0, b, 1)?;
            let mut kv_cache_mgr = KVCacheManager::new(&cache_config)
                .map_err(|e| candle_core::Error::Msg(format!("embed: cache alloc: {e}")))?;
            let mut block_table = BlockTable::new(block_size);
            kv_cache_mgr
                .allocate_for_request(&mut block_table, seq_len)
                .map_err(|e| candle_core::Error::Msg(format!("embed: block alloc: {e}")))?;
            let slot_mapping = block_table.slot_mapping(0, seq_len);
            outputs.push(self.forward_hidden(
                &ids_b,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )?);
        }
        Tensor::cat(&outputs, 0)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        self.pooling
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_normalize(&self) -> bool {
        true
    }

    fn normalize(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norm = embeddings
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sqrt()?;
        embeddings.broadcast_div(&norm)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use candle_core::DType;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["LlamaBidirectionalModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_construction() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = LlamaBidirectionalModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "LlamaBidirectionalModel should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.hidden_size, 64);
    }

    #[test]
    fn test_forward_hidden_shape() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate blocks");
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let output = model
            .forward_hidden(&input_ids, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("forward_hidden");

        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "hidden states should be [batch, seq_len, hidden_size]"
        );
    }

    #[test]
    fn test_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 4;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate blocks");
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let output = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("ModelForward::forward");

        assert_eq!(output.dims(), &[batch_size, seq_len, cfg.hidden_size]);
    }

    #[test]
    fn test_embedding_trait() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        let embeddings = model.embed(&input_ids, None).expect("embed");
        assert_eq!(
            embeddings.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "embed should return token embeddings"
        );
    }

    #[test]
    fn test_default_pooling_strategy_is_mean() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Mean
        );
    }

    #[test]
    fn test_cls_pooling_from_config() {
        let mut cfg = tiny_config();
        cfg.extra.insert(
            "pooling".to_string(),
            serde_json::Value::String("cls".to_string()),
        );

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Cls
        );
    }

    #[test]
    fn test_last_token_pooling_from_config() {
        let mut cfg = tiny_config();
        cfg.extra.insert(
            "pooling".to_string(),
            serde_json::Value::String("last".to_string()),
        );

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::LastToken
        );
    }

    #[test]
    fn test_embedding_dim() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::embedding_dim(&model), 64);
    }

    #[test]
    fn test_max_seq_len() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::max_seq_len(&model), 512);
    }

    #[test]
    fn test_supports_normalize() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        assert!(model.supports_normalize());
    }

    #[test]
    fn test_normalize() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlamaBidirectionalModel::new(&cfg, vb).expect("build model");

        let embeddings = Tensor::new(vec![vec![3.0f32, 4.0]], &device).expect("embedding tensor");
        let normalized = model.normalize(&embeddings).expect("normalize");

        let vals: Vec<Vec<f32>> = normalized.to_vec2().expect("to_vec2");
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
    }
}
