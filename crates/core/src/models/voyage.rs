//! Voyage embedding model (bidirectional Qwen3).
//!
//! Voyage adapts the Qwen3 architecture for embedding tasks by:
//! - Using **bidirectional attention** (no causal mask, no KV cache)
//! - Replacing the LM head with a linear projection (hidden_size → embedding_dim)
//! - Using mean pooling over token embeddings
//! - L2-normalizing the output
//!
//! Architecture: Qwen3 encoder (bidirectional) → RMSNorm → Linear → Mean pool → L2 norm
//!
//! Reference model: `voyageai/voyage-4-nano`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{apply_per_head_norm, RotaryEmbedding};

// ─── Bidirectional Attention ─────────────────────────────────────────────────

/// Qwen3-style attention without causal mask and without KV cache.
struct VoyageBidirectionalAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VoyageBidirectionalAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // [batch, seq, heads*head_dim] → [batch, heads, seq, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Qwen3-specific: per-head RMSNorm
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        // RoPE (offset=0, bidirectional model processes all tokens at once)
        let (q, k) = self.rotary_emb.apply(&q, &k, 0)?;

        // GQA: expand K/V heads to match Q heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = k
                .unsqueeze(2)?
                .expand((b_sz, self.num_kv_heads, repeat, seq_len, self.head_dim))?
                .reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let v = v
                .unsqueeze(2)?
                .expand((b_sz, self.num_kv_heads, repeat, seq_len, self.head_dim))?
                .reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            (k, v)
        } else {
            (k, v)
        };

        // Bidirectional scaled dot-product attention (no causal mask)
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [batch, heads, seq, head_dim] → [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}

// ─── SwiGLU MLP ──────────────────────────────────────────────────────────────

/// SwiGLU MLP matching Qwen3's FFN architecture.
struct VoyageMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VoyageMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Encoder Layer ───────────────────────────────────────────────────────────

struct VoyageEncoderLayer {
    self_attn: VoyageBidirectionalAttention,
    mlp: VoyageMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl VoyageEncoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VoyageBidirectionalAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = VoyageMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

// ─── Full Model ──────────────────────────────────────────────────────────────

/// Voyage embedding model: bidirectional Qwen3 + linear projection + mean pooling.
pub struct VoyageForEmbedding {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<VoyageEncoderLayer>,
    norm: RmsNorm,
    linear: Linear,
    embedding_size: usize,
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
}

/// Read embedding output dimension from config.
/// Tries `num_labels`, `embedding_size`, or falls back to hidden_size.
fn read_embedding_size(cfg: &ModelConfig) -> usize {
    for key in &["num_labels", "embedding_size"] {
        if let Some(v) = cfg.extra.get(*key).and_then(|v| v.as_u64()) {
            return v as usize;
        }
    }
    cfg.hidden_size
}

impl VoyageForEmbedding {
    /// Create a new Voyage embedding model.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let embedding_size = read_embedding_size(cfg);

        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(VoyageEncoderLayer::new(cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Embedding projection: hidden_size → embedding_size, no bias
        let linear = linear_no_bias(cfg.hidden_size, embedding_size, vb.pp("linear"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            linear,
            embedding_size,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Run the bidirectional encoder and return last hidden states.
    fn encode_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }

    /// Encode + project to embedding space.
    fn encode_and_project(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.encode_hidden(input_ids)?;
        self.linear.forward(&hidden)
    }
}

impl crate::engine::ModelForward for VoyageForEmbedding {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.encode_and_project(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.encode_and_project(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for VoyageForEmbedding {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.encode_and_project(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::Mean
    }

    fn pool(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        crate::engine::pool_embeddings(token_embeddings, attention_mask, PoolingStrategy::Mean)
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden_states = self.embed(input_ids, attention_mask)?;

        let mask = if let Some(mask) = attention_mask {
            mask.clone()
        } else {
            Tensor::ones(input_ids.dims(), input_ids.dtype(), input_ids.device())?
        };

        let pooled = self.pool(&hidden_states, &mask)?;
        l2_normalize(&pooled)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_size
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

/// L2-normalize along the last dimension.
fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    let norm = tensor
        .sqr()?
        .sum_keepdim(candle_core::D::Minus1)?
        .sqrt()?;
    let norm = norm.clamp(1e-12, f64::MAX)?;
    tensor.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn tiny_voyage_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_labels".to_string(), serde_json::Value::from(48));

        ModelConfig {
            architectures: vec!["VoyageQwen3BidirectionalEmbedModel".to_string()],
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
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_voyage_construction() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = VoyageForEmbedding::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Voyage should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.embedding_size, 48);
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_voyage_encode_shape() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoyageForEmbedding::new(&cfg, vb).unwrap();

        let batch_size = 2;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let projected = model.encode_and_project(&input_ids).unwrap();
        assert_eq!(
            projected.dims(),
            &[batch_size, seq_len, 48],
            "projected output should be [batch, seq_len, embedding_size]"
        );
    }

    #[test]
    fn test_voyage_sentence_embedding() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoyageForEmbedding::new(&cfg, vb).unwrap();

        let batch_size = 2;
        let seq_len = 6;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();
        let mask =
            Tensor::ones((batch_size, seq_len), DType::F32, &device).unwrap();

        let sentence_emb = model.encode(&input_ids, Some(&mask)).unwrap();
        assert_eq!(
            sentence_emb.dims(),
            &[batch_size, 48],
            "sentence embeddings should be [batch, embedding_size]"
        );
    }

    #[test]
    fn test_voyage_embedding_dim_and_max_seq() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoyageForEmbedding::new(&cfg, vb).unwrap();

        assert_eq!(model.embedding_dim(), 48);
        assert_eq!(model.max_seq_len(), 512);
    }

    #[test]
    fn test_voyage_pooling_strategy() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoyageForEmbedding::new(&cfg, vb).unwrap();

        assert_eq!(model.pooling_strategy(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_voyage_embedding_size_config_variants() {
        let mut cfg = tiny_voyage_config();

        // num_labels takes priority
        assert_eq!(read_embedding_size(&cfg), 48);

        // embedding_size also works
        cfg.extra.remove("num_labels");
        cfg.extra
            .insert("embedding_size".to_string(), serde_json::Value::from(96));
        assert_eq!(read_embedding_size(&cfg), 96);

        // Falls back to hidden_size
        cfg.extra.remove("embedding_size");
        assert_eq!(read_embedding_size(&cfg), cfg.hidden_size);
    }

    #[test]
    fn test_voyage_bidirectional_attention() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = VoyageBidirectionalAttention::new(&cfg, vb.pp("model").pp("layers").pp("0").pp("self_attn")).unwrap();

        let xs = Tensor::zeros((1, 4, 64), DType::F32, &device).unwrap();
        let output = attn.forward(&xs).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64], "attention output should preserve shape");
    }

    #[test]
    fn test_voyage_gqa_expansion() {
        // Verify that GQA works: 4 Q heads, 2 KV heads → expansion factor 2
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = VoyageBidirectionalAttention::new(&cfg, vb.pp("model").pp("layers").pp("0").pp("self_attn")).unwrap();
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);

        // Forward should work with GQA
        let xs = Tensor::zeros((2, 3, 64), DType::F32, &device).unwrap();
        let output = attn.forward(&xs).unwrap();
        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn test_voyage_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoyageForEmbedding::new(&cfg, vb).unwrap();

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
            &[1, 4, 48],
            "ModelForward output should be per-token projected embeddings"
        );
    }

    #[test]
    fn test_voyage_l2_normalize() {
        let device = Device::Cpu;
        let tensor = Tensor::new(vec![vec![3.0f32, 4.0]], &device).unwrap();
        let normalized = l2_normalize(&tensor).unwrap();
        let vals: Vec<Vec<f32>> = normalized.to_vec2().unwrap();
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_voyage_encode_normalized() {
        let cfg = tiny_voyage_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoyageForEmbedding::new(&cfg, vb).unwrap();

        assert!(model.supports_normalize());
    }
}
