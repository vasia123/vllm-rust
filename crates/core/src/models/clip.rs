//! CLIP standalone embedding model (`CLIPModel` / `CLIPEmbeddingModel`).
//!
//! Implements the CLIP (Contrastive Language-Image Pre-Training) model for
//! joint text and image embedding.  The architecture has two independent
//! transformers:
//!
//! - **Text**: token_embedding + position_embedding → pre-norm encoder →
//!   final_layer_norm → text_projection [H_t → P]
//! - **Vision**: class_embedding + patch_embedding (no bias) + pos_embedding →
//!   pre_layrnorm (typo preserved) → pre-norm encoder → post_layernorm →
//!   visual_projection [H_v → P]
//!
//! `ModelForEmbedding` exposes the **text path** (`embed` + `pool`).
//! `pool` extracts the last-token (EOS) hidden state and projects it.
//!
//! Weight paths:
//! - `text_model.embeddings.{token_embedding,position_embedding}.weight`
//! - `text_model.encoder.layers.{i}.{layer_norm1,layer_norm2,self_attn,mlp}.*`
//! - `text_model.final_layer_norm.{weight,bias}`
//! - `text_projection.weight`
//! - `vision_model.embeddings.{class_embedding,patch_embedding,position_embedding}.*`
//! - `vision_model.pre_layrnorm.{weight,bias}` (intentional typo)
//! - `vision_model.encoder.layers.{i}.*`
//! - `vision_model.post_layernorm.{weight,bias}`
//! - `visual_projection.weight`
//!
//! Reference: <https://huggingface.co/openai/clip-vit-base-patch32>

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    conv2d_no_bias, embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim,
    Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForEmbedding, ModelForward, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

// ─── Config ──────────────────────────────────────────────────────────────────

struct ClipTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
}

impl ClipTextConfig {
    fn from_json(v: &serde_json::Value) -> Self {
        let g = |key, default: usize| {
            v.get(key)
                .and_then(|x| x.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let eps = v
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-5);
        Self {
            vocab_size: g("vocab_size", 49408),
            hidden_size: g("hidden_size", 512),
            num_attention_heads: g("num_attention_heads", 8),
            num_hidden_layers: g("num_hidden_layers", 12),
            intermediate_size: g("intermediate_size", 2048),
            max_position_embeddings: g("max_position_embeddings", 77),
            layer_norm_eps: eps,
        }
    }
}

#[allow(dead_code)]
pub(crate) struct ClipVisionConfig {
    pub(crate) hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    image_size: usize,
    patch_size: usize,
    num_channels: usize,
    layer_norm_eps: f64,
    num_patches: usize,
    num_positions: usize, // patches + 1 (CLS)
}

impl ClipVisionConfig {
    pub(crate) fn from_json(v: &serde_json::Value) -> Self {
        let g = |key, default: usize| {
            v.get(key)
                .and_then(|x| x.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let eps = v
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-5);
        let image_size = g("image_size", 224);
        let patch_size = g("patch_size", 32);
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        Self {
            hidden_size: g("hidden_size", 768),
            num_attention_heads: g("num_attention_heads", 12),
            num_hidden_layers: g("num_hidden_layers", 12),
            intermediate_size: g("intermediate_size", 3072),
            image_size,
            patch_size,
            num_channels: g("num_channels", 3),
            layer_norm_eps: eps,
            num_patches,
            num_positions: num_patches + 1, // +1 for CLS token
        }
    }
}

// ─── Shared pre-norm encoder building blocks ──────────────────────────────────

/// Pre-norm encoder-only attention (no causal mask, bidirectional).
///
/// HF stores q_proj / k_proj / v_proj separately; loaded individually.
struct ClipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl ClipAttention {
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

    /// `x`: `[B, S, D]` → `[B, S, D]`.
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

struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(hidden_size, intermediate_size, vb.pp("fc1"))?,
            fc2: linear(intermediate_size, hidden_size, vb.pp("fc2"))?,
        })
    }

    // quick_gelu ≈ gelu_erf — close enough for inference
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

struct ClipEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: ClipAttention,
    layer_norm2: LayerNorm,
    mlp: ClipMlp,
}

impl ClipEncoderLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        ln_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(hidden_size, ln_eps, vb.pp("layer_norm1"))?,
            self_attn: ClipAttention::new(hidden_size, num_heads, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(hidden_size, ln_eps, vb.pp("layer_norm2"))?,
            mlp: ClipMlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?,
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

struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
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
                ClipEncoderLayer::new(
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

struct ClipTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    max_position_embeddings: usize,
    device: Device,
}

impl ClipTextEmbeddings {
    fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
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

struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: LayerNorm,
}

impl ClipTextTransformer {
    fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: ClipTextEmbeddings::new(cfg, vb.pp("embeddings"))?,
            encoder: ClipEncoder::new(
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
        })
    }

    /// Returns `[B, S, H_t]`.
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        let x = self.encoder.forward(&x)?;
        self.final_layer_norm.forward(&x)
    }
}

// ─── Vision transformer ───────────────────────────────────────────────────────

struct ClipVisionEmbeddings {
    class_embedding: Tensor,            // [H_v]
    patch_embedding: candle_nn::Conv2d, // no bias
    position_embedding: Embedding,
    num_positions: usize,
    device: Device,
}

impl ClipVisionEmbeddings {
    fn new(cfg: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let class_embedding = vb.get(cfg.hidden_size, "class_embedding")?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = embedding(
            cfg.num_positions,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
            num_positions: cfg.num_positions,
            device: vb.device().clone(),
        })
    }

    /// `pixel_values`: `[B, C, H, W]` → `[B, N+1, H_v]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        let h_v = self.class_embedding.dim(0)?;

        // Patch features: [B, H_v, h_grid, w_grid] → [B, N, H_v]
        let patches = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten(2, 3)?
            .permute((0, 2, 1))?
            .contiguous()?;

        // Prepend CLS token
        let cls = self
            .class_embedding
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, 1, h_v))?
            .contiguous()?;
        let embeddings = Tensor::cat(&[&cls, &patches], 1)?;

        // Positional embeddings
        let seq = embeddings.dim(1)?;
        let pos_ids = Tensor::arange(0u32, seq.min(self.num_positions) as u32, &self.device)?;
        let pos_emb = self.position_embedding.forward(&pos_ids)?;
        embeddings.broadcast_add(&pos_emb)
    }
}

pub(crate) struct ClipVisionTransformer {
    embeddings: ClipVisionEmbeddings,
    /// NOTE: `pre_layrnorm` is an intentional typo preserved from the HF
    /// checkpoint weight naming (matching `vision_model.pre_layrnorm.*`).
    pre_layrnorm: LayerNorm,
    encoder: ClipEncoder,
    post_layernorm: LayerNorm,
}

impl ClipVisionTransformer {
    pub(crate) fn new(cfg: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: ClipVisionEmbeddings::new(cfg, vb.pp("embeddings"))?,
            pre_layrnorm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("pre_layrnorm"))?,
            encoder: ClipEncoder::new(
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

    /// Returns `[B, N+1, H_v]`.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(pixel_values)?;
        let x = self.pre_layrnorm.forward(&x)?;
        let x = self.encoder.forward(&x)?;
        self.post_layernorm.forward(&x)
    }
}

// ─── CLIPEmbeddingModel ───────────────────────────────────────────────────────

/// CLIP joint text+image embedding model.
///
/// `ModelForEmbedding` exposes the text path (last-token pool + projection).
/// The vision encoder is available via `encode_images()` for image embeddings.
pub struct CLIPEmbeddingModel {
    text_model: ClipTextTransformer,
    vision_model: ClipVisionTransformer,
    text_projection: Linear,   // [H_t → projection_dim], no bias
    visual_projection: Linear, // [H_v → projection_dim], no bias
    projection_dim: usize,
    #[allow(dead_code)]
    text_hidden: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl CLIPEmbeddingModel {
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

        let text_cfg = ClipTextConfig::from_json(&text_json);
        let vision_cfg = ClipVisionConfig::from_json(&vision_json);

        let projection_dim = cfg
            .extra
            .get("projection_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;

        let text_hidden = text_cfg.hidden_size;
        let vision_hidden = vision_cfg.hidden_size;
        let max_position_embeddings = text_cfg.max_position_embeddings;

        let text_model = ClipTextTransformer::new(&text_cfg, vb.pp("text_model"))?;
        let vision_model = ClipVisionTransformer::new(&vision_cfg, vb.pp("vision_model"))?;
        let text_projection =
            linear_no_bias(text_hidden, projection_dim, vb.pp("text_projection"))?;
        let visual_projection =
            linear_no_bias(vision_hidden, projection_dim, vb.pp("visual_projection"))?;

        Ok(Self {
            text_model,
            vision_model,
            text_projection,
            visual_projection,
            projection_dim,
            text_hidden,
            max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Encode images to `[B, projection_dim]` embeddings (CLS token + projection).
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // [B, N+1, H_v] → CLS at position 0 → [B, H_v]
        let features = self.vision_model.forward(pixel_values)?;
        let cls = features.narrow(1, 0, 1)?.squeeze(1)?;
        self.visual_projection.forward(&cls)
    }

    /// Run text through the transformer and project: `[B, S]` → `[B, S, H_t]`.
    fn text_hidden_states(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.forward(input_ids)
    }
}

// ─── ModelForForward (required for from_config registry) ──────────────────────

impl ModelForward for CLIPEmbeddingModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // text path: hidden states of each token
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

impl ModelForEmbedding for CLIPEmbeddingModel {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.text_hidden_states(input_ids)
    }

    fn pool(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Last-token pool: [B, S, H_t] → [B, H_t]
        let pooled = crate::engine::pool_embeddings(
            token_embeddings,
            attention_mask,
            PoolingStrategy::LastToken,
        )?;
        // Project: [B, H_t] → [B, projection_dim]
        self.text_projection.forward(&pooled)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::LastToken
    }

    fn embedding_dim(&self) -> usize {
        self.projection_dim
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
                "layer_norm_eps": 1e-5
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
                "layer_norm_eps": 1e-5
            }),
        );
        extra.insert("projection_dim".to_string(), serde_json::json!(16));

        ModelConfig {
            architectures: vec!["CLIPModel".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 64,
            max_position_embeddings: 16,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
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
        let model = CLIPEmbeddingModel::new(&cfg, vb);
        assert!(model.is_ok(), "construct: {:?}", model.err());
        let m = model.unwrap();
        assert_eq!(m.projection_dim, 16);
        assert_eq!(m.text_hidden, 32);
        assert_eq!(m.max_position_embeddings, 16);
    }

    #[test]
    fn test_embed_shape() {
        let cfg = tiny_cfg();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = CLIPEmbeddingModel::new(&cfg, vb).unwrap();

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
        let model = CLIPEmbeddingModel::new(&cfg, vb).unwrap();

        let batch = 2;
        let seq = 5;
        let input_ids = Tensor::zeros((batch, seq), DType::U32, &Device::Cpu).unwrap();
        let mask = Tensor::ones((batch, seq), DType::F32, &Device::Cpu).unwrap();
        let hidden = model.embed(&input_ids, None).unwrap();
        let pooled = model.pool(&hidden, &mask).unwrap();
        // [batch, projection_dim]
        assert_eq!(pooled.dims(), &[batch, 16]);
    }

    #[test]
    fn test_pooling_strategy() {
        let cfg = tiny_cfg();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = CLIPEmbeddingModel::new(&cfg, vb).unwrap();
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
        let model = CLIPEmbeddingModel::new(&cfg, vb).unwrap();

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
