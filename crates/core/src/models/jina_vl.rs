//! Jina VL ranking model.
//!
//! `JinaVLForRanking` wraps `Qwen2VLForConditionalGeneration` and replaces
//! the generation head with a two-layer scoring head used for cross-encoding
//! (image+text → relevance score).
//!
//! Architecture:
//! - Backbone: Qwen2-VL (vision encoder + language model layers + RMSNorm)
//! - Scoring head: `dense[H→H]` → ReLU → `out_proj[H→num_labels]`
//! - Pooling: last-token hidden state (EOS carries the sequence representation)
//!
//! Weight paths:
//! - All Qwen2-VL weights follow Qwen2VL naming under the model root
//! - `score.dense.{weight,bias}` — first projection (also at `score.0.` in old ckpts)
//! - `score.out_proj.{weight,bias}` — output projection (also `score.2.` in old ckpts)
//!
//! Reference: https://huggingface.co/jinaai/jina-reranker-m0

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::qwen2_vl::Qwen2VLForConditionalGeneration;

// ─── Scoring head ─────────────────────────────────────────────────────────────

/// Two-layer MLP scorer: dense[H→H] + ReLU + out_proj[H→labels].
struct JinaVLScorer {
    dense: candle_nn::Linear,
    out_proj: candle_nn::Linear,
}

impl JinaVLScorer {
    fn new(hidden_size: usize, num_labels: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let out_proj = linear(hidden_size, num_labels, vb.pp("out_proj"))?;
        Ok(Self { dense, out_proj })
    }

    /// Score [batch, hidden] → [batch, num_labels].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense.forward(x)?;
        let x = x.relu()?;
        self.out_proj.forward(&x)
    }
}

// ─── JinaVLForRanking ─────────────────────────────────────────────────────────

/// Qwen2-VL backbone + two-layer cross-encoding scoring head.
pub struct JinaVLForRanking {
    backbone: Qwen2VLForConditionalGeneration,
    scorer: JinaVLScorer,
    num_labels: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl JinaVLForRanking {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_labels = cfg
            .extra
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let backbone = Qwen2VLForConditionalGeneration::from_model_config(cfg, vb.clone())?;
        let scorer = JinaVLScorer::new(cfg.hidden_size, num_labels, vb.pp("score"))?;

        Ok(Self {
            backbone,
            scorer,
            num_labels,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Run backbone (text only) → last-token hidden state → score.
    /// Returns [batch, num_labels].
    fn score(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.backbone.forward_text_hidden_states(input_ids)?;
        // Take last token: [batch, seq, hidden] → [batch, hidden]
        let seq_len = hidden.dim(1)?;
        let last = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        self.scorer.forward(&last)
    }
}

impl crate::engine::ModelForward for JinaVLForRanking {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.score(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.score(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for JinaVLForRanking {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Return full hidden states for custom pooling downstream.
        self.backbone.forward_text_hidden_states(input_ids)
    }

    fn pool(&self, token_embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // Last-token pool + score.
        let seq_len = token_embeddings.dim(1)?;
        let last = token_embeddings.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        self.scorer.forward(&last)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::LastToken
    }

    fn embedding_dim(&self) -> usize {
        self.num_labels
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
    use candle_core::DType;

    fn tiny_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Minimal vision config so Qwen2VL can construct the vision tower
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "depth": 2,
                "embed_dim": 32,
                "num_heads": 2,
                "hidden_size": 32,
                "patch_size": 14,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
                "mlp_ratio": 2.0,
                "in_channels": 3
            }),
        );
        extra.insert("num_labels".to_string(), serde_json::json!(1));
        // mrope_section must sum to head_dim/2 = 8
        extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({"mrope_section": [2, 3, 3]}),
        );
        ModelConfig {
            architectures: vec!["JinaVLForRanking".to_string()],
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
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_construction() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JinaVLForRanking::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "JinaVLForRanking should construct: {:?}",
            model.err()
        );
        let m = model.unwrap();
        assert_eq!(m.num_labels, 1);
        assert_eq!(m.max_position_embeddings, 512);
    }

    #[test]
    fn test_score_shape() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JinaVLForRanking::new(&cfg, vb).expect("construct");

        let batch = 2;
        let seq = 5;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, &device).expect("input_ids");

        let scores = model.score(&input_ids).expect("score");
        assert_eq!(
            scores.dims(),
            &[batch, 1],
            "scores should be [batch, num_labels]"
        );
    }

    #[test]
    fn test_pooling_strategy_is_last_token() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JinaVLForRanking::new(&cfg, vb).expect("construct");
        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::LastToken
        );
    }

    #[test]
    fn test_embed_returns_hidden_states() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JinaVLForRanking::new(&cfg, vb).expect("construct");

        let batch = 1;
        let seq = 4;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, &device).expect("input_ids");

        let hidden = model.embed(&input_ids, None).expect("embed");
        assert_eq!(
            hidden.dims(),
            &[batch, seq, 64],
            "embed() should return [batch, seq, hidden_size]"
        );
    }

    #[test]
    fn test_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JinaVLForRanking::new(&cfg, vb).expect("construct");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv = KVCacheManager::new(&cache_config).expect("kv");
        let bt = BlockTable::new(16);

        let input_ids = Tensor::zeros((1usize, 3usize), DType::U32, &device).expect("ids");
        let out = ModelForward::forward(&model, &input_ids, 0, &mut kv, &bt, &[]).expect("forward");
        // forward returns scores [batch, num_labels]
        assert_eq!(out.dims(), &[1, 1]);
    }
}
