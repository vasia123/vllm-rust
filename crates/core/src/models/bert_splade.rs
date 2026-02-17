//! BERT SPLADE sparse embedding model.
//!
//! Architecture: BERT encoder + MLM head + log1p(relu()) sparsification.
//! Produces sparse vocabulary-sized vectors for lexical retrieval.
//!
//! Pipeline: input → BERT encoder → MLM head → log1p(relu(logits)) → max over seq → [V]
//!
//! Reference: naver/splade-v3, vLLM BertSpladeSparseEmbeddingModel

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::bert::{BertEmbeddings, BertEncoder};

// ─── MLM Head ────────────────────────────────────────────────────────────────

/// BERT Masked Language Model head: dense → GELU → LayerNorm → decoder.
///
/// Projects hidden states to vocabulary logits. Weight path in HF:
/// `cls.predictions.transform.dense`, `cls.predictions.transform.LayerNorm`,
/// `cls.predictions.decoder`.
struct BertMLMHead {
    dense: Linear,
    layer_norm: LayerNorm,
    decoder: Linear,
}

impl BertMLMHead {
    fn new(
        hidden_size: usize,
        vocab_size: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm"))?;
        let decoder = linear(hidden_size, vocab_size, vb.pp("decoder"))?;
        Ok(Self {
            dense,
            layer_norm,
            decoder,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.dense.forward(hidden_states)?;
        let x = x.gelu_erf()?;
        let x = self.layer_norm.forward(&x)?;
        self.decoder.forward(&x)
    }
}

// ─── SPLADE Sparse Pooling ──────────────────────────────────────────────────

/// SPLADE pooling mode: max or sum over the sequence dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpladePooling {
    Max,
    Sum,
}

// ─── Full SPLADE Model ──────────────────────────────────────────────────────

/// BERT SPLADE sparse embedding model.
///
/// Produces sparse vocabulary-dimensional vectors using the SPLADE formula:
/// `max_{t in tokens}( log(1 + relu(mlm_logits_t)) )` per vocabulary dimension.
///
/// Output shape: [batch_size, vocab_size] — sparse, non-negative.
pub struct BertSpladeSparseEmbeddingModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    mlm_head: BertMLMHead,
    pooling: SpladePooling,
    vocab_size: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl BertSpladeSparseEmbeddingModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_pooling(cfg, vb, SpladePooling::Max)
    }

    pub fn new_with_pooling(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pooling: SpladePooling,
    ) -> Result<Self> {
        let layer_norm_eps = cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-12);

        let vb_bert = vb.pp("bert");

        let embeddings = BertEmbeddings::new(cfg, vb_bert.pp("embeddings"))?;
        let encoder = BertEncoder::new(cfg, layer_norm_eps, vb_bert.pp("encoder"))?;

        let mlm_head = BertMLMHead::new(
            cfg.hidden_size,
            cfg.vocab_size,
            layer_norm_eps,
            vb.pp("mlm_head"),
        )?;

        Ok(Self {
            embeddings,
            encoder,
            mlm_head,
            pooling,
            vocab_size: cfg.vocab_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Run BERT encoder and return last hidden states.
    fn encode_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(input_ids)?;
        self.encoder.forward(embeddings)
    }

    /// Compute SPLADE sparse vectors: MLM logits → log1p(relu) → pool over seq.
    pub fn forward_splade(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.encode_hidden(input_ids)?;
        let logits = self.mlm_head.forward(&hidden_states)?; // [batch, seq, vocab]

        // SPLADE activation: log(1 + relu(x))
        let scores = (logits.relu()? + 1.0)?.log()?; // [batch, seq, vocab]

        // Pool over sequence dimension
        match self.pooling {
            SpladePooling::Max => scores.max(1), // [batch, vocab]
            SpladePooling::Sum => scores.sum(1), // [batch, vocab]
        }
    }
}

impl crate::engine::ModelForward for BertSpladeSparseEmbeddingModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_splade(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.forward_splade(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for BertSpladeSparseEmbeddingModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_splade(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        // SPLADE does its own pooling internally; this is a no-op sentinel
        PoolingStrategy::Cls
    }

    fn pool(&self, token_embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // Already pooled to [batch, vocab] by forward_splade
        Ok(token_embeddings.clone())
    }

    fn encode(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // SPLADE handles everything in one pass
        self.forward_splade(input_ids)
    }

    fn embedding_dim(&self) -> usize {
        self.vocab_size
    }

    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_normalize(&self) -> bool {
        false
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn tiny_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("layer_norm_eps".to_string(), serde_json::Value::from(1e-12));
        extra.insert("type_vocab_size".to_string(), serde_json::Value::from(2));

        ModelConfig {
            architectures: vec!["BertSpladeSparseEmbeddingModel".to_string()],
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

    #[test]
    fn test_construction() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "SPLADE model should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_forward_splade_shape() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        let output = model.forward_splade(&input_ids).expect("forward_splade");
        assert_eq!(
            output.dims(),
            &[batch_size, cfg.vocab_size],
            "SPLADE output should be [batch, vocab_size]"
        );
    }

    #[test]
    fn test_output_non_negative() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        let output = model.forward_splade(&input_ids).expect("forward");

        let vals: Vec<Vec<f32>> = output.to_vec2().expect("to_vec2");
        for val in &vals[0] {
            assert!(*val >= 0.0, "SPLADE output must be non-negative, got {val}");
        }
    }

    #[test]
    fn test_embedding_trait() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        let embeddings = model.embed(&input_ids, None).expect("embed");
        assert_eq!(
            embeddings.dims(),
            &[batch_size, cfg.vocab_size],
            "embed should return [batch, vocab_size]"
        );
    }

    #[test]
    fn test_encode_returns_sparse_vector() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        let encoded = model.encode(&input_ids, None).expect("encode");
        assert_eq!(
            encoded.dims(),
            &[1, cfg.vocab_size],
            "encode should return [batch, vocab_size]"
        );
    }

    #[test]
    fn test_embedding_dim_is_vocab_size() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::embedding_dim(&model),
            256,
            "SPLADE embedding dim should equal vocab_size"
        );
    }

    #[test]
    fn test_max_seq_len() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::max_seq_len(&model), 512);
    }

    #[test]
    fn test_does_not_support_normalize() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        assert!(
            !model.supports_normalize(),
            "SPLADE should not support normalization"
        );
    }

    #[test]
    fn test_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

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

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        let output = model
            .forward(&input_ids, 0, &mut kv_cache_mgr, &block_table, &[])
            .expect("ModelForward::forward");

        assert_eq!(
            output.dims(),
            &[1, cfg.vocab_size],
            "ModelForward should return SPLADE sparse vector"
        );
    }

    #[test]
    fn test_sum_pooling_mode() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new_with_pooling(&cfg, vb, SpladePooling::Sum)
            .expect("build model");

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        let output = model.forward_splade(&input_ids).expect("forward");
        assert_eq!(output.dims(), &[1, cfg.vocab_size]);
    }

    #[test]
    fn test_zero_weights_produce_zero_outputs() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertSpladeSparseEmbeddingModel::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        let output = model.forward_splade(&input_ids).expect("forward");

        let vals: Vec<Vec<f32>> = output.to_vec2().expect("to_vec2");
        for val in &vals[0] {
            assert!(
                val.abs() < 1e-6,
                "zero weights should produce near-zero SPLADE scores"
            );
        }
    }
}
