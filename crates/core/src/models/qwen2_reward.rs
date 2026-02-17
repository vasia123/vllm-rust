//! Qwen2 reward models for RLHF scoring and process reward evaluation.
//!
//! Two variants:
//! - `Qwen2ForRewardModel`: scalar reward (num_labels=1), all-token pooling
//! - `Qwen2ForProcessRewardModel`: binary process reward (num_labels=2), step pooling
//!
//! Architecture: Qwen2 decoder stack + MLP scoring head
//! (Linear(hidden_size, hidden_size) -> ReLU -> Linear(hidden_size, num_labels))

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm};

use super::qwen2::Qwen2DecoderLayer;
use super::tp_layers::{TpContext, TpEmbedding};

// ─── Qwen2 Reward Base ──────────────────────────────────────────────────────

/// Qwen2 reward model with MLP scoring head.
///
/// Replaces the vocabulary LM head with a two-layer MLP:
/// `Linear(hidden_size, hidden_size) -> ReLU -> Linear(hidden_size, num_labels)`
pub struct Qwen2RewardModel {
    embed_tokens: TpEmbedding,
    layers: Vec<Qwen2DecoderLayer>,
    norm: RmsNorm,
    score_dense: Linear,
    score_out: Linear,
    tp_ctx: TpContext,
    #[allow(dead_code)]
    num_labels: usize,
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
    dtype: DType,
}

impl Qwen2RewardModel {
    fn new_inner(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
        num_labels: usize,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen2DecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Scoring MLP: hidden_size -> hidden_size -> num_labels
        let score_vb = vb.pp("score");
        let score_dense = linear(cfg.hidden_size, cfg.hidden_size, score_vb.pp("0"))?;
        let score_out = linear(cfg.hidden_size, num_labels, score_vb.pp("2"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            score_dense,
            score_out,
            tp_ctx,
            num_labels,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run the base transformer and return hidden states (before scoring head).
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

    /// Apply the scoring MLP to hidden states: dense -> ReLU -> out.
    fn apply_score(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.score_dense.forward(hidden_states)?;
        let x = x.relu()?;
        self.score_out.forward(&x)
    }

    /// Full forward: transformer + scoring head.
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
        self.apply_score(&hidden)
    }
}

// ─── Qwen2ForRewardModel (num_labels=1) ─────────────────────────────────────

/// Qwen2 reward model with scalar output (num_labels=1).
///
/// Used for preference scoring in RLHF pipelines.
pub struct Qwen2ForRewardModel(Qwen2RewardModel);

impl Qwen2ForRewardModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let num_labels = cfg
            .extra
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        Ok(Self(Qwen2RewardModel::new_inner(
            cfg, vb, pg, tp_ctx, num_labels,
        )?))
    }
}

impl crate::engine::ModelForward for Qwen2ForRewardModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.0.forward(
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
        let mut xs = self.0.embed_tokens.forward(input_ids, &self.0.tp_ctx)?;
        for (layer_idx, layer) in self.0.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.0.tp_ctx,
            )?;
        }
        let xs = self.0.norm.forward(&xs)?;
        self.0.apply_score(&xs)
    }

    fn device(&self) -> &Device {
        &self.0.device
    }
}

impl ModelForEmbedding for Qwen2ForRewardModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.0.embed_tokens.forward(input_ids, &self.0.tp_ctx)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::LastToken
    }

    fn embedding_dim(&self) -> usize {
        self.0.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.0.max_position_embeddings
    }

    fn device(&self) -> &Device {
        &self.0.device
    }
}

// ─── Qwen2ForProcessRewardModel (num_labels=2) ─────────────────────────────

/// Qwen2 process reward model with binary output (num_labels=2).
///
/// Used for step-level process rewards (validating intermediate reasoning steps).
/// Returns per-token binary scores (good/bad step).
pub struct Qwen2ForProcessRewardModel(Qwen2RewardModel);

impl Qwen2ForProcessRewardModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let num_labels = cfg
            .extra
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        Ok(Self(Qwen2RewardModel::new_inner(
            cfg, vb, pg, tp_ctx, num_labels,
        )?))
    }
}

impl crate::engine::ModelForward for Qwen2ForProcessRewardModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.0.forward(
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
        let mut xs = self.0.embed_tokens.forward(input_ids, &self.0.tp_ctx)?;
        for (layer_idx, layer) in self.0.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.0.tp_ctx,
            )?;
        }
        let xs = self.0.norm.forward(&xs)?;
        self.0.apply_score(&xs)
    }

    fn device(&self) -> &Device {
        &self.0.device
    }
}

impl ModelForEmbedding for Qwen2ForProcessRewardModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.0.embed_tokens.forward(input_ids, &self.0.tp_ctx)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::LastToken
    }

    fn embedding_dim(&self) -> usize {
        self.0.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.0.max_position_embeddings
    }

    fn device(&self) -> &Device {
        &self.0.device
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Qwen2ForRewardModel".to_string()],
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
            bos_token_id: 151643,
            eos_token_id: 151645,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
        }
    }

    fn tiny_process_config() -> ModelConfig {
        let mut cfg = tiny_config();
        cfg.architectures = vec!["Qwen2ForProcessRewardModel".to_string()];
        cfg.extra
            .insert("num_labels".to_string(), serde_json::Value::from(2));
        cfg
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

    // ── Qwen2ForRewardModel tests ──

    #[test]
    fn test_reward_construction() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Qwen2ForRewardModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen2ForRewardModel should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.0.layers.len(), 2);
        assert_eq!(model.0.num_labels, 1);
    }

    #[test]
    fn test_reward_forward_shape() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForRewardModel::new(&cfg, vb).expect("build model");

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
            .0
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
            "reward model should output [batch, seq_len, num_labels=1]"
        );
    }

    #[test]
    fn test_reward_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForRewardModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 4;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let output = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward");

        assert_eq!(output.dims(), &[1, seq_len, 1]);
    }

    #[test]
    fn test_reward_embedding_trait() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForRewardModel::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((2, 6), DType::U32, &device).expect("input");
        let embeddings = model.embed(&input_ids, None).expect("embed");
        assert_eq!(embeddings.dims(), &[2, 6, cfg.hidden_size]);
    }

    #[test]
    fn test_reward_pooling_strategy() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForRewardModel::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::LastToken
        );
    }

    #[test]
    fn test_reward_embedding_dim() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForRewardModel::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::embedding_dim(&model), 64);
    }

    #[test]
    fn test_reward_zero_weights() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForRewardModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..3).collect();

        let output = model
            .0
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

    // ── Qwen2ForProcessRewardModel tests ──

    #[test]
    fn test_process_reward_construction() {
        let cfg = tiny_process_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Qwen2ForProcessRewardModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen2ForProcessRewardModel should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.0.num_labels, 2);
    }

    #[test]
    fn test_process_reward_forward_shape() {
        let cfg = tiny_process_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForProcessRewardModel::new(&cfg, vb).expect("build model");

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
            .0
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
            &[batch_size, seq_len, 2],
            "process reward model should output [batch, seq_len, num_labels=2]"
        );
    }

    #[test]
    fn test_process_reward_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = tiny_process_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForProcessRewardModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 4;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let output = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward");

        assert_eq!(output.dims(), &[1, seq_len, 2]);
    }

    #[test]
    fn test_process_reward_default_num_labels() {
        let mut cfg = tiny_config();
        cfg.extra.remove("num_labels");

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2ForProcessRewardModel::new(&cfg, vb).expect("build model");

        assert_eq!(
            model.0.num_labels, 2,
            "ProcessRewardModel should default to 2 labels"
        );
    }
}
