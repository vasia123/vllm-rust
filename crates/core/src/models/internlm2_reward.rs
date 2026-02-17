//! InternLM2 reward model for RLHF scoring.
//!
//! Wraps the InternLM2 transformer with a scalar reward head (`v_head`)
//! that maps hidden states to per-token scores.
//!
//! Architecture: same decoder stack as InternLM2ForCausalLM, but replaces
//! the vocabulary LM head with `RowParallelLinear(hidden_size, 1)`.
//! Used as a pooling model (single-pass, no autoregressive generation).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm};

use super::internlm2::InternLM2DecoderLayer;
use super::tp_layers::{TpContext, TpEmbedding};

// ─── InternLM2 Reward Model ─────────────────────────────────────────────────

/// InternLM2 reward model for per-token scoring.
///
/// Uses the InternLM2 decoder stack with a single-output reward head (`v_head`)
/// instead of the vocabulary LM head. Returns scalar scores per token.
pub struct InternLM2ForRewardModel {
    embed_tokens: TpEmbedding,
    layers: Vec<InternLM2DecoderLayer>,
    norm: RmsNorm,
    v_head: Linear,
    tp_ctx: TpContext,
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
    dtype: DType,
}

impl InternLM2ForRewardModel {
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

        let embed_tokens = TpEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("tok_embeddings"),
            pg,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(InternLM2DecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Reward head: hidden_size -> 1 (scalar score per token)
        let v_head = linear(cfg.hidden_size, 1, vb.pp("v_head"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            v_head,
            tp_ctx,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run the base transformer and return hidden states (before reward head).
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }
        self.norm.forward(&xs)
    }

    /// Run forward and apply the reward head.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )?;
        self.v_head.forward(&hidden)
    }
}

impl crate::engine::ModelForward for InternLM2ForRewardModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        InternLM2ForRewardModel::forward(
            self,
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
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.v_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for InternLM2ForRewardModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Return token embeddings from the embedding layer only.
        // Full model hidden states require KV cache; this returns the raw embeddings.
        self.embed_tokens.forward(input_ids, &self.tp_ctx)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        // Reward models typically use last-token pooling
        PoolingStrategy::LastToken
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
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["InternLM2ForRewardModel".to_string()],
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

        let model = InternLM2ForRewardModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "InternLM2ForRewardModel should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.hidden_size, 64);
    }

    #[test]
    fn test_forward_shape() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

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
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, 1],
            "reward model should output [batch, seq_len, 1]"
        );
    }

    #[test]
    fn test_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

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

        assert_eq!(output.dims(), &[batch_size, seq_len, 1]);
    }

    #[test]
    fn test_embedding_trait() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let embeddings = model.embed(&input_ids, None).expect("embed");
        assert_eq!(
            embeddings.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "embed should return token embeddings"
        );
    }

    #[test]
    fn test_pooling_strategy() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

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
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::embedding_dim(&model), 64);
    }

    #[test]
    fn test_max_seq_len() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::max_seq_len(&model), 512);
    }

    #[test]
    fn test_zero_weights_produce_zero_scores() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternLM2ForRewardModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..3).collect();

        let output = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        let vals: Vec<Vec<Vec<f32>>> = output.to_vec3().expect("to_vec3");
        for token_scores in &vals[0] {
            assert!(
                (token_scores[0]).abs() < 1e-6,
                "zero weights should produce zero scores"
            );
        }
    }
}
