//! SigLIP standalone embedding model (`SiglipModel` / `SiglipEmbeddingModel`).
//!
//! Implements the SigLIP (Sigmoid Loss for Language Image Pre-Training) model
//! for joint text and image embedding.  Architecture:
//!
//! - **Text**: token_embedding + position_embedding → pre-norm encoder →
//!   final_layer_norm → head [H_t → projection_size]
//! - **Vision**: patch_embedding (Conv2d, **with** bias) + position_embedding →
//!   pre-norm encoder → post_layernorm  (no CLS token, no pre_layernorm)
//!
//! Key differences from CLIP:
//! - No `class_embedding` / CLS token in vision
//! - No `pre_layernorm` in vision transformer
//! - No external text_projection / visual_projection; text uses `.head`
//! - Vision returns raw hidden states (vision_hidden_size, not projected)
//! - Patch Conv2d **has** bias
//!
//! `ModelForEmbedding` exposes the **text path** with last-token (EOS) pooling
//! followed by the `head` projection.
//!
//! Weight paths:
//! - `text_model.embeddings.{token_embedding,position_embedding}.weight`
//! - `text_model.encoder.layers.{i}.{layer_norm1,layer_norm2,self_attn,mlp}.*`
//! - `text_model.final_layer_norm.{weight,bias}`
//! - `text_model.head.{weight,bias}`
//! - `vision_model.embeddings.patch_embedding.{weight,bias}`
//! - `vision_model.embeddings.position_embedding.weight`
//! - `vision_model.encoder.layers.{i}.*`
//! - `vision_model.post_layernorm.{weight,bias}`
//!
//! Reference: <https://huggingface.co/google/siglip-base-patch16-224>

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, embedding, layer_norm, linear, ops::softmax_last_dim, Conv2dConfig, Embedding,
    LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForEmbedding, ModelForward, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

// ─── Config ──────────────────────────────────────────────────────────────────

struct SiglipTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    projection_size: usize,
}

impl SiglipTextConfig {
    fn from_json(v: &serde_json::Value) -> Self {
        let g = |key, default: usize| {
            v.get(key)
                .and_then(|x| x.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let eps = v
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-6);
        let hidden_size = g("hidden_size", 768);
        Self {
            vocab_size: g("vocab_size", 32000),
            hidden_size,
            num_attention_heads: g("num_attention_heads", 12),
            num_hidden_layers: g("num_hidden_layers", 12),
            intermediate_size: g("intermediate_size", 3072),
            max_position_embeddings: g("max_position_embeddings", 64),
            layer_norm_eps: eps,
            projection_size: g("projection_size", hidden_size),
        }
    }
}

#[allow(dead_code)]
pub(crate) struct SiglipVisionConfig {
    pub(crate) hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    image_size: usize,
    patch_size: usize,
    num_channels: usize,
    layer_norm_eps: f64,
    num_patches: usize,
}

impl SiglipVisionConfig {
    pub(crate) fn from_json(v: &serde_json::Value) -> Self {
        let g = |key, default: usize| {
            v.get(key)
                .and_then(|x| x.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let eps = v
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-6);
        let image_size = g("image_size", 224);
        let patch_size = g("patch_size", 16);
        Self {
            hidden_size: g("hidden_size", 768),
            num_attention_heads: g("num_attention_heads", 16),
            num_hidden_layers: g("num_hidden_layers", 12),
            intermediate_size: g("intermediate_size", 3072),
            image_size,
            patch_size,
            num_channels: g("num_channels", 3),
            layer_norm_eps: eps,
            num_patches: (image_size / patch_size) * (image_size / patch_size),
        }
    }
}

// ─── Pre-norm encoder building blocks ────────────────────────────────────────

/// Encoder-only attention (no causal mask, bidirectional).
struct SiglipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SiglipAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            q_proj: linear(hidden_size, hidden_size, vb.pp("q_proj"))?,
            k_proj: linear(hidden_size, hidden_size, vb.pp("k_proj"))?,
            v_proj: linear(hidden_size, hidden_size, vb.pp("v_proj"))?,
            out_proj: linear(hidden_size, hidden_size, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = softmax_last_dim(&attn)?;
        let out = attn
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

struct SiglipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl SiglipMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(hidden_size, intermediate_size, vb.pp("fc1"))?,
            fc2: linear(intermediate_size, hidden_size, vb.pp("fc2"))?,
        })
    }

    // gelu_pytorch_tanh ≈ gelu_erf for inference
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

struct SiglipEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: SiglipAttention,
    layer_norm2: LayerNorm,
    mlp: SiglipMlp,
}

impl SiglipEncoderLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        ln_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(hidden_size, ln_eps, vb.pp("layer_norm1"))?,
            self_attn: SiglipAttention::new(hidden_size, num_heads, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(hidden_size, ln_eps, vb.pp("layer_norm2"))?,
            mlp: SiglipMlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.self_attn.forward(&self.layer_norm1.forward(x)?)?;
        let x = (residual + x)?;
        let residual = &x;
        let x = self.mlp.forward(&self.layer_norm2.forward(&x)?)?;
        residual + x
    }
}

struct SiglipEncoder {
    layers: Vec<SiglipEncoderLayer>,
}

impl SiglipEncoder {
    fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        ln_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|i| {
                SiglipEncoderLayer::new(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    ln_eps,
                    vb.pp("layers").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

// ─── Text transformer ─────────────────────────────────────────────────────────

struct SiglipTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    max_position_embeddings: usize,
    device: Device,
}

impl SiglipTextEmbeddings {
    fn new(cfg: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            token_embedding: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embedding"))?,
            position_embedding: embedding(
                cfg.max_position_embeddings,
                cfg.hidden_size,
                vb.pp("position_embedding"),
            )?,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let pos_ids = Tensor::arange(
            0u32,
            seq_len.min(self.max_position_embeddings) as u32,
            &self.device,
        )?;
        let tok = self.token_embedding.forward(input_ids)?;
        let pos = self.position_embedding.forward(&pos_ids)?;
        tok.broadcast_add(&pos)
    }
}

struct SiglipTextTransformer {
    embeddings: SiglipTextEmbeddings,
    encoder: SiglipEncoder,
    final_layer_norm: LayerNorm,
    head: Linear, // [H_t → projection_size], with bias
}

impl SiglipTextTransformer {
    fn new(cfg: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: SiglipTextEmbeddings::new(cfg, vb.pp("embeddings"))?,
            encoder: SiglipEncoder::new(
                cfg.num_hidden_layers,
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.intermediate_size,
                cfg.layer_norm_eps,
                vb.pp("encoder"),
            )?,
            final_layer_norm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            head: linear(cfg.hidden_size, cfg.projection_size, vb.pp("head"))?,
        })
    }

    /// Returns `[B, S, H_t]` hidden states (before `head` projection).
    fn forward_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        let x = self.encoder.forward(&x)?;
        self.final_layer_norm.forward(&x)
    }
}

// ─── Vision transformer ───────────────────────────────────────────────────────

struct SiglipVisionEmbeddings {
    patch_embedding: candle_nn::Conv2d, // with bias
    position_embedding: Embedding,
    num_patches: usize,
    device: Device,
}

impl SiglipVisionEmbeddings {
    fn new(cfg: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        // SigLIP patch_embedding has bias (unlike CLIP)
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = embedding(
            cfg.num_patches,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches: cfg.num_patches,
            device: vb.device().clone(),
        })
    }

    /// `pixel_values`: `[B, C, H, W]` → `[B, N, H_v]` (no CLS token).
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let patches = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten(2, 3)?
            .permute((0, 2, 1))?
            .contiguous()?;

        let seq = patches.dim(1)?;
        let pos_ids = Tensor::arange(0u32, seq.min(self.num_patches) as u32, &self.device)?;
        let pos_emb = self.position_embedding.forward(&pos_ids)?;
        patches.broadcast_add(&pos_emb)
    }
}

pub(crate) struct SiglipVisionTransformer {
    embeddings: SiglipVisionEmbeddings,
    encoder: SiglipEncoder,
    post_layernorm: LayerNorm,
}

impl SiglipVisionTransformer {
    pub(crate) fn new(cfg: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: SiglipVisionEmbeddings::new(cfg, vb.pp("embeddings"))?,
            encoder: SiglipEncoder::new(
                cfg.num_hidden_layers,
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.intermediate_size,
                cfg.layer_norm_eps,
                vb.pp("encoder"),
            )?,
            post_layernorm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("post_layernorm"),
            )?,
        })
    }

    /// Returns `[B, N, H_v]` — all patch hidden states (no CLS).
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(pixel_values)?;
        let x = self.encoder.forward(&x)?;
        self.post_layernorm.forward(&x)
    }
}

// ─── SiglipEmbeddingModel ─────────────────────────────────────────────────────

/// SigLIP joint text+image embedding model.
///
/// `ModelForEmbedding` exposes the text path (last-token EOS pool + head projection).
/// `encode_images()` provides mean-pooled vision features.
pub struct SiglipEmbeddingModel {
    text_model: SiglipTextTransformer,
    vision_model: SiglipVisionTransformer,
    projection_size: usize,
    #[allow(dead_code)]
    text_hidden: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl SiglipEmbeddingModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let text_json = cfg
            .extra
            .get("text_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let vision_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        let text_cfg = SiglipTextConfig::from_json(&text_json);
        let vision_cfg = SiglipVisionConfig::from_json(&vision_json);

        let projection_size = text_cfg.projection_size;
        let text_hidden = text_cfg.hidden_size;
        let max_position_embeddings = text_cfg.max_position_embeddings;

        Ok(Self {
            text_model: SiglipTextTransformer::new(&text_cfg, vb.pp("text_model"))?,
            vision_model: SiglipVisionTransformer::new(&vision_cfg, vb.pp("vision_model"))?,
            projection_size,
            text_hidden,
            max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Encode images: `[B, C, H, W]` → mean-pool → `[B, H_v]`.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // SigLIP uses mean pooling for vision features
        let features = self.vision_model.forward(pixel_values)?; // [B, N, H_v]
        features.mean(1) // [B, H_v]
    }

    fn text_hidden_states(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.forward_hidden(input_ids)
    }
}

// ─── ModelForForward ──────────────────────────────────────────────────────────

impl ModelForward for SiglipEmbeddingModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.text_hidden_states(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.text_hidden_states(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── ModelForEmbedding ────────────────────────────────────────────────────────

impl ModelForEmbedding for SiglipEmbeddingModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.text_hidden_states(input_ids)
    }

    fn pool(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Last-token (EOS) pool: [B, S, H_t] → [B, H_t]
        let pooled = crate::engine::pool_embeddings(
            token_embeddings,
            attention_mask,
            PoolingStrategy::LastToken,
        )?;
        // Apply head projection: [B, H_t] → [B, projection_size]
        self.text_model.head.forward(&pooled)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::LastToken
    }

    fn embedding_dim(&self) -> usize {
        self.projection_size
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
    use candle_core::{DType, Device};

    fn tiny_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "text_config".to_string(),
            serde_json::json!({
                "vocab_size": 64,
                "hidden_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "max_position_embeddings": 16,
                "layer_norm_eps": 1e-6,
                "projection_size": 16
            }),
        );
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "image_size": 16,
                "patch_size": 8,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
            }),
        );

        ModelConfig {
            architectures: vec!["SiglipModel".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 64,
            max_position_embeddings: 16,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
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
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = SiglipEmbeddingModel::new(&cfg, vb);
        assert!(model.is_ok(), "construct: {:?}", model.err());
        let m = model.unwrap();
        assert_eq!(m.projection_size, 16);
        assert_eq!(m.text_hidden, 32);
        assert_eq!(m.max_position_embeddings, 16);
    }

    #[test]
    fn test_embed_shape() {
        let cfg = tiny_cfg();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = SiglipEmbeddingModel::new(&cfg, vb).unwrap();

        let batch = 2;
        let seq = 5;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, &Device::Cpu).unwrap();
        let hidden = model.embed(&input_ids, None).unwrap();
        assert_eq!(hidden.dims(), &[batch, seq, 32]);
    }

    #[test]
    fn test_pool_shape() {
        let cfg = tiny_cfg();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = SiglipEmbeddingModel::new(&cfg, vb).unwrap();

        let batch = 2;
        let seq = 5;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, &Device::Cpu).unwrap();
        let mask = Tensor::ones((batch, seq), DType::F32, &Device::Cpu).unwrap();
        let hidden = model.embed(&input_ids, None).unwrap();
        let pooled = model.pool(&hidden, &mask).unwrap();
        // [batch, projection_size]
        assert_eq!(pooled.dims(), &[batch, 16]);
    }

    #[test]
    fn test_pooling_strategy() {
        let cfg = tiny_cfg();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = SiglipEmbeddingModel::new(&cfg, vb).unwrap();
        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::LastToken
        );
    }

    #[test]
    fn test_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = SiglipEmbeddingModel::new(&cfg, vb).unwrap();

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv = KVCacheManager::new(&cache_config).unwrap();
        let bt = BlockTable::new(16);

        let input_ids = Tensor::zeros((1usize, 3usize), DType::U32, &device).unwrap();
        let out = ModelForward::forward(&model, &input_ids, 0, &mut kv, &bt, &[]).unwrap();
        // forward returns hidden states [batch, seq, text_hidden]
        assert_eq!(out.dims(), &[1, 3, 32]);
    }
}
