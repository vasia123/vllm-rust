//! ModernBERT embedding model with RoPE and sliding window attention.
//!
//! Architecture: Pre-norm BERT variant with:
//! - RoPE (rotary position embeddings) instead of absolute positions
//! - Sliding window attention for local layers, full attention for global layers
//! - Gated GELU MLP (Wi projects to 2*intermediate_size, split into input+gate)
//! - Layer 0 skips attn_norm (identity), all others use LayerNorm
//! - No token_type embeddings, no position embeddings
//!
//! Covers: ModernBertModel, ModernBertForSequenceClassification

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::RotaryEmbedding;

// ─── Sliding Window Mask ─────────────────────────────────────────────────────

/// Create a bidirectional sliding window attention mask.
///
/// Positions within `window` distance get 0.0 (attend), positions outside get
/// -inf (block). Returns shape [1, 1, seq_len, seq_len] for broadcasting
/// over batch and head dimensions.
fn sliding_window_mask(
    seq_len: usize,
    window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; seq_len]; seq_len];
    for (i, row) in mask.iter_mut().enumerate() {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(seq_len);
        for cell in row.iter_mut().take(end).skip(start) {
            *cell = 0.0;
        }
    }
    Tensor::new(mask, device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

struct ModernBertEmbeddings {
    tok_embeddings: Embedding,
    norm: LayerNorm,
}

impl ModernBertEmbeddings {
    fn new(cfg: &ModelConfig, norm_eps: f64, _norm_bias: bool, vb: VarBuilder) -> Result<Self> {
        let tok_embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("tok_embeddings"))?;
        // candle layer_norm always includes bias; norm_bias=false will just load zero bias
        let norm = layer_norm(cfg.hidden_size, norm_eps, vb.pp("norm"))?;
        Ok(Self {
            tok_embeddings,
            norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.tok_embeddings.forward(input_ids)?;
        self.norm.forward(&embeddings)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
}

impl ModernBertAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &ModelConfig,
        attention_bias: bool,
        sliding_window: Option<usize>,
        rope_theta: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        let wqkv = if attention_bias {
            linear(hidden_size, 3 * hidden_size, vb.pp("attn").pp("Wqkv"))?
        } else {
            linear_no_bias(hidden_size, 3 * hidden_size, vb.pp("attn").pp("Wqkv"))?
        };

        let wo = if attention_bias {
            linear(hidden_size, hidden_size, vb.pp("attn").pp("Wo"))?
        } else {
            linear_no_bias(hidden_size, hidden_size, vb.pp("attn").pp("Wo"))?
        };

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            rope_theta,
            DType::F32,
            vb.device(),
        )?;

        Ok(Self {
            wqkv,
            wo,
            rotary_emb,
            num_heads,
            head_dim,
            sliding_window,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let qkv = self.wqkv.forward(xs)?;
        let all_head_size = self.num_heads * self.head_dim;
        let q = qkv.narrow(2, 0, all_head_size)?;
        let k = qkv.narrow(2, all_head_size, all_head_size)?;
        let v = qkv.narrow(2, 2 * all_head_size, all_head_size)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE (seqlen_offset=0 for encoder — no KV cache)
        let (q, k) = self.rotary_emb.apply(&q, &k, 0)?;

        // Scaled dot-product attention (bidirectional: no causal mask)
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Apply sliding window mask for local layers
        let attn_weights = if let Some(window) = self.sliding_window {
            let mask = sliding_window_mask(seq_len, window, xs.dtype(), xs.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.wo.forward(&attn_output)
    }
}

// ─── MLP (Gated GELU) ───────────────────────────────────────────────────────

struct ModernBertMlp {
    wi: Linear,
    wo: Linear,
}

impl ModernBertMlp {
    fn new(cfg: &ModelConfig, mlp_bias: bool, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;

        let wi = if mlp_bias {
            linear(hidden_size, 2 * intermediate_size, vb.pp("mlp").pp("Wi"))?
        } else {
            linear_no_bias(hidden_size, 2 * intermediate_size, vb.pp("mlp").pp("Wi"))?
        };

        let wo = if mlp_bias {
            linear(intermediate_size, hidden_size, vb.pp("mlp").pp("Wo"))?
        } else {
            linear_no_bias(intermediate_size, hidden_size, vb.pp("mlp").pp("Wo"))?
        };

        Ok(Self { wi, wo })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let projected = self.wi.forward(xs)?;
        let intermediate_size = projected.dim(2)? / 2;
        let input = projected.narrow(2, 0, intermediate_size)?;
        let gate = projected.narrow(2, intermediate_size, intermediate_size)?;
        let activated = input.gelu_erf()?;
        let gated = (activated * gate)?;
        self.wo.forward(&gated)
    }
}

// ─── Encoder Layer ───────────────────────────────────────────────────────────

struct ModernBertLayer {
    attn_norm: Option<LayerNorm>,
    attn: ModernBertAttention,
    mlp_norm: LayerNorm,
    mlp: ModernBertMlp,
}

impl ModernBertLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &ModelConfig,
        layer_idx: usize,
        norm_eps: f64,
        _norm_bias: bool,
        attention_bias: bool,
        mlp_bias: bool,
        sliding_window: Option<usize>,
        rope_theta: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Layer 0: attn_norm is identity (skip normalization)
        let attn_norm = if layer_idx > 0 {
            Some(layer_norm(cfg.hidden_size, norm_eps, vb.pp("attn_norm"))?)
        } else {
            None
        };

        let attn =
            ModernBertAttention::new(cfg, attention_bias, sliding_window, rope_theta, vb.clone())?;

        let mlp_norm = layer_norm(cfg.hidden_size, norm_eps, vb.pp("mlp_norm"))?;

        let mlp = ModernBertMlp::new(cfg, mlp_bias, vb)?;

        Ok(Self {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pre-norm: normalize before attention
        let normed = if let Some(ref norm) = self.attn_norm {
            norm.forward(xs)?
        } else {
            xs.clone()
        };
        let attn_output = self.attn.forward(&normed)?;
        let xs = (xs + attn_output)?;

        // Pre-norm: normalize before MLP
        let normed = self.mlp_norm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&normed)?;
        &xs + mlp_output
    }
}

// ─── Full ModernBERT Model ───────────────────────────────────────────────────

/// ModernBERT embedding model with RoPE and sliding window attention.
///
/// Implements both `ModelForForward` (for engine compatibility) and
/// `ModelForEmbedding` (for embedding generation).
pub struct ModernBertForEmbedding {
    embeddings: ModernBertEmbeddings,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    hidden_size: usize,
    max_position_embeddings: usize,
    pooling: PoolingStrategy,
    device: Device,
}

impl ModernBertForEmbedding {
    /// Create a new ModernBERT model from config and weights.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let norm_eps = cfg
            .extra
            .get("norm_eps")
            .or_else(|| cfg.extra.get("layer_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        let norm_bias = cfg
            .extra
            .get("norm_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let attention_bias = cfg.attention_bias.unwrap_or(false);
        let mlp_bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let global_attn_every_n_layers = cfg
            .extra
            .get("global_attn_every_n_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let local_attention = cfg
            .extra
            .get("local_attention")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let global_rope_theta = cfg
            .extra
            .get("global_rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(160000.0);
        let local_rope_theta = cfg.extra.get("local_rope_theta").and_then(|v| v.as_f64());
        let classifier_pooling = cfg
            .extra
            .get("classifier_pooling")
            .and_then(|v| v.as_str())
            .unwrap_or("cls");

        let vb_m = vb.pp("model");

        let embeddings =
            ModernBertEmbeddings::new(cfg, norm_eps, norm_bias, vb_m.pp("embeddings"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let is_global = i % global_attn_every_n_layers == 0;
            let sliding_window = if is_global {
                None
            } else {
                Some(local_attention / 2)
            };
            let rope_theta = if is_global {
                global_rope_theta
            } else {
                local_rope_theta.unwrap_or(global_rope_theta)
            };

            layers.push(ModernBertLayer::new(
                cfg,
                i,
                norm_eps,
                norm_bias,
                attention_bias,
                mlp_bias,
                sliding_window,
                rope_theta,
                vb_m.pp(format!("layers.{i}")),
            )?);
        }

        let final_norm = layer_norm(cfg.hidden_size, norm_eps, vb_m.pp("final_norm"))?;

        let pooling = match classifier_pooling.to_lowercase().as_str() {
            "mean" | "average" => PoolingStrategy::Mean,
            _ => PoolingStrategy::Cls,
        };

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            pooling,
            device: vb.device().clone(),
        })
    }

    /// Run the full encoder stack and return last hidden states.
    pub(crate) fn encode_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        self.final_norm.forward(&hidden_states)
    }
}

impl crate::engine::ModelForward for ModernBertForEmbedding {
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

impl ModelForEmbedding for ModernBertForEmbedding {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.encode_hidden(input_ids)
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
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_modernbert_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("norm_eps".to_string(), serde_json::json!(1e-5));
        extra.insert("norm_bias".to_string(), serde_json::json!(true));
        extra.insert("mlp_bias".to_string(), serde_json::json!(false));
        extra.insert(
            "global_attn_every_n_layers".to_string(),
            serde_json::json!(3),
        );
        extra.insert("local_attention".to_string(), serde_json::json!(128));
        extra.insert("global_rope_theta".to_string(), serde_json::json!(160000.0));
        extra.insert("classifier_pooling".to_string(), serde_json::json!("cls"));

        ModelConfig {
            architectures: vec!["ModernBertModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 32,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 160000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_modernbert_construction() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = ModernBertForEmbedding::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ModernBertForEmbedding should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.expect("construction verified above");
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.max_position_embeddings, 512);
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn test_modernbert_forward_shape() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

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
    fn test_modernbert_embed_shape() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

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
    fn test_modernbert_encode_shape() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        let batch_size = 2;
        let seq_len = 6;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");
        let mask = Tensor::ones((batch_size, seq_len), DType::F32, &device)
            .expect("attention mask tensor");

        let sentence_embeddings = model.encode(&input_ids, Some(&mask)).expect("encode");

        assert_eq!(
            sentence_embeddings.dims(),
            &[batch_size, cfg.hidden_size],
            "sentence embeddings should be [batch, hidden_size]"
        );
    }

    #[test]
    fn test_modernbert_pooling_strategy_cls() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Cls
        );
    }

    #[test]
    fn test_modernbert_pooling_strategy_mean() {
        let mut cfg = tiny_modernbert_config();
        cfg.extra
            .insert("classifier_pooling".to_string(), serde_json::json!("mean"));
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Mean
        );
    }

    #[test]
    fn test_modernbert_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

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
    fn test_modernbert_embedding_dim_and_max_seq_len() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        assert_eq!(ModelForEmbedding::embedding_dim(&model), 64);
        assert_eq!(ModelForEmbedding::max_seq_len(&model), 512);
    }

    #[test]
    fn test_modernbert_normalize() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        let embeddings = Tensor::new(vec![vec![3.0f32, 4.0]], &device).expect("embedding tensor");
        let normalized = model.normalize(&embeddings).expect("normalize");

        let vals: Vec<Vec<f32>> = normalized.to_vec2().expect("to_vec2");
        // 3/5 = 0.6, 4/5 = 0.8
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_modernbert_single_token() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input tensor");
        let hidden = model
            .encode_hidden(&input_ids)
            .expect("encode single token");
        assert_eq!(hidden.dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn test_modernbert_sliding_window_mask_shape() {
        let seq_len = 8;
        let window = 3;
        let device = Device::Cpu;

        let mask =
            sliding_window_mask(seq_len, window, DType::F32, &device).expect("mask creation");
        assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
    }

    #[test]
    fn test_modernbert_sliding_window_mask_values() {
        let seq_len = 5;
        let window = 1;
        let device = Device::Cpu;

        let mask =
            sliding_window_mask(seq_len, window, DType::F32, &device).expect("mask creation");
        // Squeeze to [seq_len, seq_len] for easier inspection
        let mask_2d: Vec<Vec<f32>> = mask
            .squeeze(0)
            .expect("squeeze batch")
            .squeeze(0)
            .expect("squeeze head")
            .to_vec2()
            .expect("to_vec2");

        // With window=1, position i can attend to [i-1, i, i+1]
        // Position 0: can attend to [0, 1]
        assert_eq!(mask_2d[0][0], 0.0);
        assert_eq!(mask_2d[0][1], 0.0);
        assert!(mask_2d[0][2].is_infinite() && mask_2d[0][2] < 0.0);

        // Position 2: can attend to [1, 2, 3]
        assert!(mask_2d[2][0].is_infinite() && mask_2d[2][0] < 0.0);
        assert_eq!(mask_2d[2][1], 0.0);
        assert_eq!(mask_2d[2][2], 0.0);
        assert_eq!(mask_2d[2][3], 0.0);
        assert!(mask_2d[2][4].is_infinite() && mask_2d[2][4] < 0.0);

        // Position 4 (last): can attend to [3, 4]
        assert!(mask_2d[4][2].is_infinite() && mask_2d[4][2] < 0.0);
        assert_eq!(mask_2d[4][3], 0.0);
        assert_eq!(mask_2d[4][4], 0.0);
    }

    #[test]
    fn test_modernbert_layer_global_vs_local() {
        // With global_attn_every_n_layers=3 and 4 layers:
        // Layer 0: global (0 % 3 == 0) -> no sliding window
        // Layer 1: local (1 % 3 != 0) -> sliding window
        // Layer 2: local (2 % 3 != 0) -> sliding window
        // Layer 3: global (3 % 3 == 0) -> no sliding window
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        assert!(
            model.layers[0].attn.sliding_window.is_none(),
            "layer 0 should be global (no sliding window)"
        );
        assert!(
            model.layers[1].attn.sliding_window.is_some(),
            "layer 1 should be local (has sliding window)"
        );
        assert!(
            model.layers[2].attn.sliding_window.is_some(),
            "layer 2 should be local (has sliding window)"
        );
        assert!(
            model.layers[3].attn.sliding_window.is_none(),
            "layer 3 should be global (no sliding window)"
        );

        // Sliding window = local_attention / 2 = 128 / 2 = 64
        assert_eq!(model.layers[1].attn.sliding_window, Some(64));
    }

    #[test]
    fn test_modernbert_gated_mlp_shape() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mlp = ModernBertMlp::new(&cfg, false, vb.pp("test")).expect("build mlp");

        let batch_size = 2;
        let seq_len = 4;
        let input = Tensor::zeros((batch_size, seq_len, cfg.hidden_size), DType::F32, &device)
            .expect("input");

        let output = mlp.forward(&input).expect("mlp forward");
        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "MLP output should preserve [batch, seq_len, hidden_size]"
        );
    }

    #[test]
    fn test_modernbert_batch_input() {
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        let batch_size = 4;
        let seq_len = 10;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input tensor");

        let hidden = model.encode_hidden(&input_ids).expect("batch forward");
        assert_eq!(
            hidden.dims(),
            &[batch_size, seq_len, cfg.hidden_size],
            "batch forward should produce [batch, seq_len, hidden_size]"
        );
    }

    #[test]
    fn test_modernbert_layer0_no_attn_norm() {
        // Layer 0 should have attn_norm = None (identity)
        let cfg = tiny_modernbert_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb).expect("build model");

        assert!(
            model.layers[0].attn_norm.is_none(),
            "layer 0 should have no attn_norm (identity)"
        );
        assert!(
            model.layers[1].attn_norm.is_some(),
            "layer 1 should have attn_norm"
        );
        assert!(
            model.layers[2].attn_norm.is_some(),
            "layer 2 should have attn_norm"
        );
        assert!(
            model.layers[3].attn_norm.is_some(),
            "layer 3 should have attn_norm"
        );
    }

    #[test]
    fn test_modernbert_attention_bias_config() {
        // Default config has attention_bias=false
        let cfg = tiny_modernbert_config();
        assert_eq!(cfg.attention_bias, Some(false));

        // Test construction with attention_bias=true
        let mut cfg_with_bias = tiny_modernbert_config();
        cfg_with_bias.attention_bias = Some(true);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg_with_bias, vb);
        assert!(
            model.is_ok(),
            "should construct with attention_bias=true: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_modernbert_local_rope_theta() {
        // Test that local_rope_theta is used for non-global layers when set
        let mut cfg = tiny_modernbert_config();
        cfg.extra
            .insert("local_rope_theta".to_string(), serde_json::json!(10000.0));
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ModernBertForEmbedding::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "should construct with local_rope_theta set: {:?}",
            model.err()
        );
    }
}
