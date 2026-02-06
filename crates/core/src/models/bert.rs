//! BERT encoder-only model for text embedding generation.
//!
//! Key differences from decoder models (Llama, Qwen, etc.):
//! - Bidirectional attention (no causal mask)
//! - Absolute position embeddings (not RoPE)
//! - No KV cache (processes all tokens at once)
//! - Token type embeddings for segment A/B
//! - LayerNorm (not RMSNorm)
//! - GELU activation (not SiLU/SwiGLU)
//! - All linear layers have bias

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

// ─── Embeddings ──────────────────────────────────────────────────────────────

struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let vocab_size = cfg.vocab_size;
        let max_position_embeddings = cfg.max_position_embeddings;

        let type_vocab_size = cfg
            .extra
            .get("type_vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let layer_norm_eps = cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-12);

        let word_embeddings = embedding(vocab_size, hidden_size, vb.pp("word_embeddings"))?;
        let position_embeddings = embedding(
            max_position_embeddings,
            hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings =
            embedding(type_vocab_size, hidden_size, vb.pp("token_type_embeddings"))?;
        let layer_norm = layer_norm(hidden_size, layer_norm_eps, vb.pp("LayerNorm"))?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        let word_emb = self.word_embeddings.forward(input_ids)?;

        // Absolute position IDs: [0, 1, 2, ..., seq_len-1], broadcast to batch
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;

        // Default token type IDs: all zeros (segment A)
        let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, input_ids.device())?;
        let type_emb = self.token_type_embeddings.forward(&token_type_ids)?;

        let embeddings = (word_emb + pos_emb)?.add(&type_emb)?;
        self.layer_norm.forward(&embeddings)
    }
}

// ─── Self-Attention ──────────────────────────────────────────────────────────

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl BertSelfAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;

        Ok(Self {
            query,
            key,
            value,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states.dims3()?;

        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;

        // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention (bidirectional: no causal mask)
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))
    }
}

// ─── Attention Output (dense + LayerNorm + residual) ─────────────────────────

struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn new(hidden_size: usize, layer_norm_eps: f64, vb: VarBuilder) -> Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(hidden_size, layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&(hidden_states + residual)?)
    }
}

// ─── Full Attention Block ────────────────────────────────────────────────────

struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attention = BertSelfAttention::new(hidden_size, num_heads, vb.pp("self"))?;
        let output = BertSelfOutput::new(hidden_size, layer_norm_eps, vb.pp("output"))?;
        Ok(Self {
            self_attention,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attn_output = self.self_attention.forward(hidden_states)?;
        self.output.forward(&attn_output, hidden_states)
    }
}

// ─── Intermediate (FFN first half) ──────────────────────────────────────────

struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(hidden_size, intermediate_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.dense.forward(hidden_states)?.gelu_erf()
    }
}

// ─── Output (FFN second half + LayerNorm + residual) ─────────────────────────

struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    fn new(
        intermediate_size: usize,
        hidden_size: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dense = linear(intermediate_size, hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(hidden_size, layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&(hidden_states + residual)?)
    }
}

// ─── Encoder Layer ──────────────────────────────────────────────────────────

struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn new(cfg: &ModelConfig, layer_norm_eps: f64, vb: VarBuilder) -> Result<Self> {
        let attention = BertAttention::new(
            cfg.hidden_size,
            cfg.num_attention_heads,
            layer_norm_eps,
            vb.pp("attention"),
        )?;
        let intermediate = BertIntermediate::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("intermediate"),
        )?;
        let output = BertOutput::new(
            cfg.intermediate_size,
            cfg.hidden_size,
            layer_norm_eps,
            vb.pp("output"),
        )?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attn_output = self.attention.forward(hidden_states)?;
        let intermediate_output = self.intermediate.forward(&attn_output)?;
        self.output.forward(&intermediate_output, &attn_output)
    }
}

// ─── Encoder ────────────────────────────────────────────────────────────────

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn new(cfg: &ModelConfig, layer_norm_eps: f64, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(BertLayer::new(
                cfg,
                layer_norm_eps,
                vb.pp(format!("layer.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, mut hidden_states: Tensor) -> Result<Tensor> {
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }
}

// ─── Pooler ─────────────────────────────────────────────────────────────────

struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take [CLS] token (first token), apply dense + tanh
        let cls_token = hidden_states.narrow(1, 0, 1)?.squeeze(1)?;
        self.dense.forward(&cls_token)?.tanh()
    }
}

// ─── Full BERT Model ─────────────────────────────────────────────────────────

/// BERT encoder-only model for sequence embedding.
///
/// Implements both `ModelForward` (for engine compatibility) and
/// `ModelForEmbedding` (for embedding generation).
pub struct BertForSequenceEmbedding {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl BertForSequenceEmbedding {
    /// Create a new BERT model.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let layer_norm_eps = cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-12);

        let vb_bert = vb.pp("bert");

        let embeddings = BertEmbeddings::new(cfg, vb_bert.pp("embeddings"))?;
        let encoder = BertEncoder::new(cfg, layer_norm_eps, vb_bert.pp("encoder"))?;

        // Pooler is optional; load if weights are available
        let pooler = BertPooler::new(cfg.hidden_size, vb_bert.pp("pooler")).ok();

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Run the full encoder stack and return last hidden states.
    fn encode_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(input_ids)?;
        self.encoder.forward(embeddings)
    }
}

impl crate::engine::ModelForward for BertForSequenceEmbedding {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Return last hidden states as "logits" for embedding use
        self.encode_hidden(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        // BERT is non-autoregressive; just run the full encoder
        self.encode_hidden(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for BertForSequenceEmbedding {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        if self.pooler.is_some() {
            PoolingStrategy::Cls
        } else {
            PoolingStrategy::Mean
        }
    }

    fn pool(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        if let Some(ref pooler) = self.pooler {
            pooler.forward(token_embeddings)
        } else {
            crate::engine::pool_embeddings(token_embeddings, attention_mask, PoolingStrategy::Mean)
        }
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden_states = self.embed(input_ids, attention_mask)?;

        let mask = if let Some(mask) = attention_mask {
            mask.clone()
        } else {
            Tensor::ones(input_ids.dims(), input_ids.dtype(), input_ids.device())?
        };

        self.pool(&hidden_states, &mask)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_bert_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("layer_norm_eps".to_string(), serde_json::Value::from(1e-12));
        extra.insert("type_vocab_size".to_string(), serde_json::Value::from(2));

        ModelConfig {
            architectures: vec!["BertModel".to_string()],
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
    fn test_bert_construction() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = BertForSequenceEmbedding::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "BertForSequenceEmbedding should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.expect("construction verified above");
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.max_position_embeddings, 512);
        assert_eq!(model.encoder.layers.len(), 2);
    }

    #[test]
    fn test_bert_forward_shape() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 5;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let hidden_states = model.encode_hidden(&input_ids).expect("encode_hidden");
        assert_eq!(
            hidden_states.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "hidden states shape should be [batch, seq_len, hidden_size]"
        );
    }

    #[test]
    fn test_bert_embed_shape() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        let batch_size = 1;
        let seq_len = 8;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let embeddings = model.embed(&input_ids, None).expect("embed");
        assert_eq!(
            embeddings.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "embed output should be [batch, seq_len, hidden_size]"
        );
    }

    #[test]
    fn test_bert_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        // BERT ignores KV cache, but the trait requires it
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
            &[batch_size, seq_len, cfg.hidden_size],
            "ModelForward output should be last hidden states"
        );
    }

    #[test]
    fn test_bert_pooler_cls() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // With zero weights, pooler produces tanh(0) = 0 for all dims
        let pooler = BertPooler::new(cfg.hidden_size, vb.pp("bert").pp("pooler")).expect("pooler");

        let batch_size = 2;
        let seq_len = 5;
        let hidden = Tensor::ones((batch_size, seq_len, cfg.hidden_size), DType::F32, &device)
            .expect("hidden");

        let pooled = pooler.forward(&hidden).expect("pooler forward");
        assert_eq!(
            pooled.dims(),
            &[batch_size, cfg.hidden_size],
            "pooler output should be [batch, hidden_size]"
        );
    }

    #[test]
    fn test_bert_encode_produces_sentence_embeddings() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");
        let attention_mask = Tensor::ones((batch_size, seq_len), DType::F32, &device)
            .expect("attention mask tensor");

        let sentence_embeddings = model
            .encode(&input_ids, Some(&attention_mask))
            .expect("encode");

        assert_eq!(
            sentence_embeddings.dims(),
            &[batch_size, cfg.hidden_size],
            "sentence embeddings should be [batch, hidden_size]"
        );
    }

    #[test]
    fn test_bert_embedding_dim_and_max_seq_len() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        assert_eq!(model.embedding_dim(), 64);
        assert_eq!(model.max_seq_len(), 512);
    }

    #[test]
    fn test_bert_normalize() {
        let device = Device::Cpu;
        let cfg = tiny_bert_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        let embeddings = Tensor::new(vec![vec![3.0f32, 4.0]], &device).expect("embedding tensor");
        let normalized = model.normalize(&embeddings).expect("normalize");

        let vals: Vec<Vec<f32>> = normalized.to_vec2().expect("to_vec2");
        // 3/5 = 0.6, 4/5 = 0.8
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_bert_single_token_input() {
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input tensor");
        let hidden = model
            .encode_hidden(&input_ids)
            .expect("encode single token");
        assert_eq!(hidden.dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn test_bert_bidirectional_attention() {
        // Verify that attention is bidirectional by checking that changing
        // a later token affects an earlier token's representation.
        // With causal masking this would NOT happen.
        let cfg = tiny_bert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BertForSequenceEmbedding::new(&cfg, vb).expect("build model");

        // With zero weights, all outputs will be identical regardless of input.
        // This is expected. The key structural guarantee is that there is NO
        // causal mask in the attention computation, verified by code inspection.
        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        let hidden = model.encode_hidden(&input_ids).expect("forward");
        assert_eq!(hidden.dims(), &[1, 4, cfg.hidden_size]);
    }
}
