//! ColBERT late-interaction retrieval model.
//!
//! Extends BERT with a linear projection to produce per-token embeddings
//! for late-interaction scoring (MaxSim). Unlike standard embedding models
//! that pool into a single vector, ColBERT returns one embedding per token.
//!
//! Architecture: BERT encoder → colbert_linear (hidden_size → colbert_dim) → L2 norm
//!
//! Reference: <https://arxiv.org/abs/2004.12832>

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::ModelForEmbedding;
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::bert::BertForSequenceEmbedding;

/// ColBERT late-interaction retrieval model.
///
/// Wraps a BERT encoder and adds a linear projection (`colbert_linear`)
/// that maps from `hidden_size` to `colbert_dim`. The model returns
/// per-token L2-normalized embeddings suitable for MaxSim scoring.
pub struct ColBERTForRetrieval {
    bert: BertForSequenceEmbedding,
    colbert_linear: Linear,
    colbert_dim: usize,
    #[allow(dead_code)] // used in tests for diagnostics
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
}

/// Read colbert_dim from config.extra, trying multiple field names.
fn read_colbert_dim(cfg: &ModelConfig) -> Option<usize> {
    for key in &["colbert_dim", "dim", "projection_dim"] {
        if let Some(v) = cfg.extra.get(*key).and_then(|v| v.as_u64()) {
            return Some(v as usize);
        }
    }
    None
}

impl ColBERTForRetrieval {
    /// Create a new ColBERT model.
    ///
    /// The `colbert_dim` is read from config (`colbert_dim`, `dim`, or
    /// `projection_dim`). If not found, defaults to 128.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let colbert_dim = read_colbert_dim(cfg).unwrap_or(128);

        let bert = BertForSequenceEmbedding::new(cfg, vb.clone())?;

        // colbert_linear: hidden_size → colbert_dim, no bias
        // Weight names: "colbert_linear.weight" or "linear.weight"
        let colbert_linear = linear_no_bias(cfg.hidden_size, colbert_dim, vb.pp("colbert_linear"))
            .or_else(|_| linear_no_bias(cfg.hidden_size, colbert_dim, vb.pp("linear")))?;

        Ok(Self {
            bert,
            colbert_linear,
            colbert_dim,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Run BERT encoder + colbert_linear projection + L2 normalization.
    ///
    /// Returns per-token embeddings: `[batch, seq_len, colbert_dim]`.
    fn encode_colbert(&self, input_ids: &Tensor) -> Result<Tensor> {
        // BERT encoder: [batch, seq_len, hidden_size]
        let hidden_states = self.bert.embed(input_ids, None)?;

        // Project: [batch, seq_len, hidden_size] → [batch, seq_len, colbert_dim]
        let projected = self.colbert_linear.forward(&hidden_states)?;

        // L2 normalize along last dimension
        l2_normalize(&projected)
    }
}

/// L2-normalize along the last dimension.
fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    let norm = tensor.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    // Clamp norm to avoid division by zero
    let norm = norm.clamp(1e-12, f64::MAX)?;
    tensor.broadcast_div(&norm)
}

impl crate::engine::ModelForward for ColBERTForRetrieval {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.encode_colbert(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.encode_colbert(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for ColBERTForRetrieval {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.encode_colbert(input_ids)
    }

    fn pooling_strategy(&self) -> crate::engine::PoolingStrategy {
        // ColBERT uses all tokens (no pooling reduction)
        crate::engine::PoolingStrategy::Cls
    }

    fn pool(&self, token_embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // ColBERT returns per-token embeddings; no pooling reduction.
        // For compatibility, return the 3D tensor as-is. Callers that
        // need per-token embeddings should use embed() directly.
        Ok(token_embeddings.clone())
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // ColBERT: encode returns per-token embeddings [batch, seq_len, colbert_dim]
        self.embed(input_ids, attention_mask)
    }

    fn embedding_dim(&self) -> usize {
        self.colbert_dim
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
        l2_normalize(embeddings)
    }
}

/// Compute MaxSim score between query and document embeddings.
///
/// For each query token, find the maximum similarity to any document token,
/// then sum across query tokens. This is the core ColBERT scoring function.
///
/// # Arguments
/// * `query_emb` - Query per-token embeddings `[q_len, colbert_dim]`
/// * `doc_emb` - Document per-token embeddings `[d_len, colbert_dim]`
///
/// # Returns
/// Scalar similarity score.
pub fn maxsim_score(query_emb: &Tensor, doc_emb: &Tensor) -> Result<f32> {
    // [q_len, d_len] = query @ doc.T
    let sim_matrix = query_emb.matmul(&doc_emb.t()?)?;

    // Max over document tokens for each query token: [q_len]
    let max_per_query = sim_matrix.max(candle_core::D::Minus1)?;

    // Sum over query tokens → scalar
    let score = max_per_query.sum_all()?;
    score.to_scalar::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::DType;

    fn tiny_colbert_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("layer_norm_eps".to_string(), serde_json::Value::from(1e-12));
        extra.insert("type_vocab_size".to_string(), serde_json::Value::from(2));
        extra.insert("colbert_dim".to_string(), serde_json::Value::from(32));

        ModelConfig {
            architectures: vec!["HF_ColBERT".to_string()],
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
    fn test_colbert_construction() {
        let cfg = tiny_colbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = ColBERTForRetrieval::new(&cfg, vb);
        assert!(model.is_ok(), "ColBERT should construct: {:?}", model.err());

        let model = model.unwrap();
        assert_eq!(model.colbert_dim, 32);
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.max_position_embeddings, 512);
    }

    #[test]
    fn test_colbert_encode_shape() {
        let cfg = tiny_colbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ColBERTForRetrieval::new(&cfg, vb).unwrap();

        let batch_size = 2;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let embeddings = model.encode_colbert(&input_ids).unwrap();
        assert_eq!(
            embeddings.dims(),
            &[batch_size, seq_len, 32],
            "output should be [batch, seq_len, colbert_dim]"
        );
    }

    #[test]
    fn test_colbert_per_token_output() {
        let cfg = tiny_colbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ColBERTForRetrieval::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let embeddings = model.embed(&input_ids, None).unwrap();

        // Should return per-token embeddings, not pooled
        assert_eq!(embeddings.dims().len(), 3, "should be 3D (per-token)");
        assert_eq!(embeddings.dims()[1], 8, "seq_len should be preserved");
        assert_eq!(embeddings.dims()[2], 32, "last dim should be colbert_dim");
    }

    #[test]
    fn test_colbert_l2_normalized() {
        let device = Device::Cpu;

        let tensor = Tensor::new(vec![vec![vec![3.0f32, 4.0], vec![1.0, 0.0]]], &device).unwrap();

        let normalized = l2_normalize(&tensor).unwrap();
        let vals: Vec<Vec<Vec<f32>>> = normalized.to_vec3().unwrap();

        // [3, 4] → norm=5 → [0.6, 0.8]
        assert!((vals[0][0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][0][1] - 0.8).abs() < 1e-5);

        // [1, 0] → norm=1 → [1.0, 0.0]
        assert!((vals[0][1][0] - 1.0).abs() < 1e-5);
        assert!((vals[0][1][1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_colbert_embedding_dim() {
        let cfg = tiny_colbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ColBERTForRetrieval::new(&cfg, vb).unwrap();

        assert_eq!(model.embedding_dim(), 32);
        assert_eq!(model.max_seq_len(), 512);
    }

    #[test]
    fn test_colbert_dim_from_config_alternatives() {
        let mut cfg = tiny_colbert_config();

        // "colbert_dim" takes priority
        assert_eq!(read_colbert_dim(&cfg), Some(32));

        // "dim" also works
        cfg.extra.remove("colbert_dim");
        cfg.extra
            .insert("dim".to_string(), serde_json::Value::from(96));
        assert_eq!(read_colbert_dim(&cfg), Some(96));

        // "projection_dim" also works
        cfg.extra.remove("dim");
        cfg.extra
            .insert("projection_dim".to_string(), serde_json::Value::from(64));
        assert_eq!(read_colbert_dim(&cfg), Some(64));

        // No field → None (caller defaults to 128)
        cfg.extra.remove("projection_dim");
        assert_eq!(read_colbert_dim(&cfg), None);
    }

    #[test]
    fn test_maxsim_score() {
        let device = Device::Cpu;

        // Query: 2 tokens × 3 dims
        let query =
            Tensor::new(vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]], &device).unwrap();

        // Document: 3 tokens × 3 dims
        let doc = Tensor::new(
            vec![
                vec![0.5f32, 0.5, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            &device,
        )
        .unwrap();

        let score = maxsim_score(&query, &doc).unwrap();

        // query[0]=[1,0,0] max sims: doc[0]=0.5, doc[1]=1.0, doc[2]=0.0 → max=1.0
        // query[1]=[0,1,0] max sims: doc[0]=0.5, doc[1]=0.0, doc[2]=0.0 → max=0.5
        // sum = 1.5
        assert!(
            (score - 1.5).abs() < 1e-5,
            "MaxSim score should be 1.5, got {score}"
        );
    }

    #[test]
    fn test_colbert_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_colbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ColBERTForRetrieval::new(&cfg, vb).unwrap();

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
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let output = model
            .forward(&input_ids, 0, &mut kv_cache_mgr, &block_table, &[])
            .unwrap();

        assert_eq!(
            output.dims(),
            &[1, 4, 32],
            "ModelForward output should be per-token embeddings"
        );
    }

    #[test]
    fn test_colbert_default_dim() {
        let mut cfg = tiny_colbert_config();
        cfg.extra.remove("colbert_dim");

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ColBERTForRetrieval::new(&cfg, vb).unwrap();

        assert_eq!(model.colbert_dim, 128, "default colbert_dim should be 128");
    }
}
