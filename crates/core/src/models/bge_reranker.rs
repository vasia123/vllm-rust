//! BGE-Reranker cross-encoder model for sequence classification / reranking.
//!
//! Wraps the BERT encoder with a RoBERTa-style classification head:
//! dense(hidden_size, hidden_size) -> tanh -> out_proj(hidden_size, num_labels).
//!
//! Architecture mirrors `RobertaForSequenceClassification` from Python vLLM.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::bert::BertForSequenceEmbedding;

// ─── BGE Reranker for Classification ─────────────────────────────────────────

/// BGE-Reranker cross-encoder for sequence classification / reranking.
///
/// Wraps `BertForSequenceEmbedding` with a RoBERTa-style classification head
/// (dense + tanh + out_proj). For rerankers, `num_labels` is typically 1,
/// producing a single relevance score per input pair.
pub struct BgeRerankerForClassification {
    model: BertForSequenceEmbedding,
    dense: Linear,
    out_proj: Linear,
    num_labels: usize,
    device: Device,
}

impl BgeRerankerForClassification {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_labels = cfg
            .extra
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let model = BertForSequenceEmbedding::new(cfg, vb.clone())?;

        let classifier_vb = vb.pp("classifier");
        let dense = linear(cfg.hidden_size, cfg.hidden_size, classifier_vb.pp("dense"))?;
        let out_proj = linear(cfg.hidden_size, num_labels, classifier_vb.pp("out_proj"))?;

        Ok(Self {
            model,
            dense,
            out_proj,
            num_labels,
            device: vb.device().clone(),
        })
    }

    /// Run classification: returns [batch_size, num_labels] scores.
    pub fn classify(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.model.encode_hidden(input_ids)?;

        // CLS token pooling (first token)
        let cls = hidden_states.narrow(1, 0, 1)?.squeeze(1)?;

        // dense -> tanh -> out_proj (RobertaClassificationHead)
        let output = self.dense.forward(&cls)?.tanh()?;
        self.out_proj.forward(&output)
    }
}

impl crate::engine::ModelForward for BgeRerankerForClassification {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.classify(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.classify(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for BgeRerankerForClassification {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.model.encode_hidden(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::Cls
    }

    fn pool(&self, token_embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // CLS token -> dense -> tanh -> out_proj (classification head as pooling)
        let cls = token_embeddings.narrow(1, 0, 1)?.squeeze(1)?;
        let output = self.dense.forward(&cls)?.tanh()?;
        self.out_proj.forward(&output)
    }

    fn embedding_dim(&self) -> usize {
        self.num_labels
    }

    fn max_seq_len(&self) -> usize {
        self.model.max_position_embeddings
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

    fn tiny_bge_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("layer_norm_eps".to_string(), serde_json::Value::from(1e-12));
        extra.insert("type_vocab_size".to_string(), serde_json::Value::from(2));
        extra.insert("num_labels".to_string(), serde_json::Value::from(1));

        ModelConfig {
            architectures: vec!["RobertaForSequenceClassification".to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 32,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-12,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    fn tiny_bge_multi_label_config() -> ModelConfig {
        let mut cfg = tiny_bge_config();
        cfg.extra
            .insert("num_labels".to_string(), serde_json::Value::from(3));
        cfg
    }

    #[test]
    fn test_bge_reranker_construction() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = BgeRerankerForClassification::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "BgeRerankerForClassification should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.expect("construction verified above");
        assert_eq!(model.num_labels, 1);
        assert_eq!(model.model.hidden_size, 64);
        assert_eq!(model.model.max_position_embeddings, 512);
    }

    #[test]
    fn test_bge_reranker_classify_shape() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 8;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let scores = model.classify(&input_ids).expect("classify");
        assert_eq!(
            scores.dims(),
            &[batch_size, 1],
            "classify output should be [batch, num_labels]"
        );
    }

    #[test]
    fn test_bge_reranker_classify_single_label() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(model.num_labels, 1);

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).expect("input tensor");
        let scores = model.classify(&input_ids).expect("classify");

        assert_eq!(scores.dims(), &[1, 1]);

        // With zero weights, tanh(0) = 0, so out_proj(0) = 0
        let vals: Vec<Vec<f32>> = scores.to_vec2().expect("to_vec2");
        assert!((vals[0][0]).abs() < 1e-6, "zero weights should produce zero scores");
    }

    #[test]
    fn test_bge_reranker_classify_multi_label() {
        let cfg = tiny_bge_multi_label_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(model.num_labels, 3);

        let batch_size = 4;
        let seq_len = 6;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let scores = model.classify(&input_ids).expect("classify");
        assert_eq!(
            scores.dims(),
            &[batch_size, 3],
            "multi-label output should be [batch, 3]"
        );
    }

    #[test]
    fn test_bge_reranker_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 16,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 4;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let output = model
            .forward(&input_ids, 0, &mut kv_cache_mgr, &block_table, &[])
            .expect("ModelForward::forward");

        assert_eq!(
            output.dims(),
            &[batch_size, 1],
            "ModelForward output should be classification scores"
        );
    }

    #[test]
    fn test_bge_reranker_embedding_trait() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let hidden_states = model.embed(&input_ids, None).expect("embed");
        assert_eq!(
            hidden_states.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "embed should return full hidden states"
        );
    }

    #[test]
    fn test_bge_reranker_pooling_strategy_cls() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Cls
        );
    }

    #[test]
    fn test_bge_reranker_embedding_dim() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::embedding_dim(&model),
            1,
            "embedding_dim should be num_labels"
        );
    }

    #[test]
    fn test_bge_reranker_embedding_dim_multi_label() {
        let cfg = tiny_bge_multi_label_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::embedding_dim(&model),
            3,
            "embedding_dim should be num_labels (3)"
        );
    }

    #[test]
    fn test_bge_reranker_max_seq_len() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::max_seq_len(&model),
            512,
            "max_seq_len should come from BERT config"
        );
    }

    #[test]
    fn test_bge_reranker_encode_shape() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");
        let attention_mask = Tensor::ones((batch_size, seq_len), DType::F32, &device)
            .expect("attention mask tensor");

        let encoded = model
            .encode(&input_ids, Some(&attention_mask))
            .expect("encode");

        assert_eq!(
            encoded.dims(),
            &[batch_size, 1],
            "encode should return [batch, num_labels] after pool()"
        );
    }

    #[test]
    fn test_bge_reranker_single_token() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input tensor");
        let scores = model.classify(&input_ids).expect("classify single token");
        assert_eq!(scores.dims(), &[1, 1]);
    }

    #[test]
    fn test_bge_reranker_batch_input() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let batch_size = 8;
        let seq_len = 10;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let scores = model.classify(&input_ids).expect("classify batch");
        assert_eq!(
            scores.dims(),
            &[batch_size, 1],
            "batch classification should produce [8, 1]"
        );
    }

    #[test]
    fn test_bge_reranker_pool_applies_classification_head() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 5;
        let token_embeddings =
            Tensor::ones((batch_size, seq_len, cfg.hidden_size), DType::F32, &device)
                .expect("token embeddings");
        let attention_mask =
            Tensor::ones((batch_size, seq_len), DType::F32, &device).expect("attention mask");

        let pooled = model
            .pool(&token_embeddings, &attention_mask)
            .expect("pool");

        assert_eq!(
            pooled.dims(),
            &[batch_size, 1],
            "pool should return [batch, num_labels] after classification head"
        );
    }

    #[test]
    fn test_bge_reranker_forward_decode_batch() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 16,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");

        let batch_size = 3;
        let seq_len = 4;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let output = model
            .forward_decode_batch(&input_ids, &[], &mut kv_cache_mgr)
            .expect("forward_decode_batch");

        assert_eq!(
            output.dims(),
            &[batch_size, 1],
            "forward_decode_batch should return classification scores"
        );
    }

    #[test]
    fn test_bge_reranker_default_num_labels() {
        // Config without num_labels should default to 1
        let mut cfg = tiny_bge_config();
        cfg.extra.remove("num_labels");

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert_eq!(model.num_labels, 1, "default num_labels should be 1");
    }

    #[test]
    fn test_bge_reranker_device() {
        let cfg = tiny_bge_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BgeRerankerForClassification::new(&cfg, vb).expect("build model");

        assert!(
            matches!(ModelForEmbedding::device(&model), Device::Cpu),
            "device should be CPU"
        );
    }
}
