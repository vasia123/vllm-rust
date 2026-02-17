//! E5-Mistral decoder-only embedding model.
//!
//! E5-Mistral is a Mistral model fine-tuned for embeddings. It uses:
//! - Causal attention (same as standard Mistral)
//! - Last-token pooling (the last token attends to all previous tokens)
//! - No lm_head (returns hidden states, not logits)
//! - No KV cache (single-pass encoding, no autoregressive generation)
//!
//! This covers: E5-Mistral-7B-Instruct, SFR-Embedding-Mistral, and similar
//! decoder-only embedding models based on Mistral/Llama architectures.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm, RotaryEmbedding};

// ─── Attention (no KV cache, causal mask for encoding) ──────────────────────

struct E5MistralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_groups: usize,
}

impl E5MistralAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let mk_linear = candle_nn::linear_no_bias;
        let q_proj = mk_linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = mk_linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = mk_linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = mk_linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            DType::F32,
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_groups: num_heads / num_kv_heads,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE (offset=0, single-pass encoding)
        let (q, k) = self.rotary_emb.apply(&q, &k, 0)?;

        // GQA: expand k, v if needed
        let k = if self.kv_groups > 1 {
            crate::layers::repeat_kv(k, self.kv_groups)?
        } else {
            k
        };
        let v = if self.kv_groups > 1 {
            crate::layers::repeat_kv(v, self.kv_groups)?
        } else {
            v
        };

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct E5MistralMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl E5MistralMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let mk_linear = candle_nn::linear_no_bias;
        let gate_proj = mk_linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = mk_linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = mk_linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct E5MistralLayer {
    self_attn: E5MistralAttention,
    mlp: E5MistralMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl E5MistralLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = E5MistralAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = E5MistralMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
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

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

// ─── E5-Mistral Embedding Model ─────────────────────────────────────────────

/// E5-Mistral decoder-only embedding model.
///
/// Uses causal attention with last-token pooling for sentence embeddings.
pub struct E5MistralForEmbedding {
    embed_tokens: Embedding,
    layers: Vec<E5MistralLayer>,
    norm: RmsNorm,
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
    dtype: DType,
}

impl E5MistralForEmbedding {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(E5MistralLayer::new(cfg, vb_m.pp(format!("layers.{i}")))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run the full decoder stack and return normalized hidden states.
    fn encode_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;

        // Causal mask for encoding (last token sees all previous tokens)
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                0,
                self.dtype,
                &self.device,
            )?)
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask.as_ref())?;
        }
        self.norm.forward(&xs)
    }
}

impl crate::engine::ModelForward for E5MistralForEmbedding {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for E5MistralForEmbedding {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_e5_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["MistralModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 32,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn test_e5_construction() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = E5MistralForEmbedding::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "E5MistralForEmbedding should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_e5_forward_shape() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        let batch_size = 2;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let hidden = model.encode_hidden(&input_ids).unwrap();
        assert_eq!(hidden.dims(), &[batch_size, seq_len, cfg.hidden_size]);
    }

    #[test]
    fn test_e5_embed_shape() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let embeddings = model.embed(&input_ids, None).unwrap();
        assert_eq!(embeddings.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_e5_encode_sentence_embeddings() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((2, 6), DType::U32, &device).unwrap();
        let mask = Tensor::ones((2, 6), DType::F32, &device).unwrap();

        let sentence_embeddings = model.encode(&input_ids, Some(&mask)).unwrap();
        assert_eq!(sentence_embeddings.dims(), &[2, 64]);
    }

    #[test]
    fn test_e5_pooling_strategy() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::LastToken
        );
    }

    #[test]
    fn test_e5_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

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
        assert_eq!(output.dims(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn test_e5_single_token() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden = model.encode_hidden(&input_ids).unwrap();
        assert_eq!(hidden.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_e5_normalize() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        let embeddings = Tensor::new(vec![vec![3.0f32, 4.0]], &device).unwrap();
        let normalized = model.normalize(&embeddings).unwrap();
        let vals: Vec<Vec<f32>> = normalized.to_vec2().unwrap();
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_e5_embedding_dim_and_max_seq_len() {
        let cfg = tiny_e5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = E5MistralForEmbedding::new(&cfg, vb).unwrap();

        assert_eq!(ModelForEmbedding::embedding_dim(&model), 64);
        assert_eq!(ModelForEmbedding::max_seq_len(&model), 512);
    }
}
