//! MiniCPM-V vision-language model implementation.
//!
//! MiniCPM-V is an efficient VLM family that combines a vision encoder with
//! a Perceiver Resampler and a language model. This implementation covers
//! v2.6 (Qwen2 backbone + Idefics2-style vision encoder + Resampler).
//!
//! # Architecture
//!
//! 1. Vision encoder: Conv2D patches → position embeddings → Transformer → LayerNorm
//! 2. Resampler: Cross-attention with learned queries + 2D sincos position embeddings
//! 3. LLM: Qwen2 decoder layers (reused from qwen2.rs)
//!
//! The resampler compresses variable-length vision tokens into a fixed number
//! of query tokens before insertion into the LLM's embedding space.
//!
//! Reference: https://github.com/OpenBMB/MiniCPM-V

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, embedding, layer_norm, linear, linear_no_bias, rms_norm, Conv2dConfig, Embedding,
    LayerNorm, Linear, RmsNorm, VarBuilder,
};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2DecoderLayer;
use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

/// MiniCPM-V vision encoder configuration (Idefics2-style).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MiniCPMVisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    layer_norm_eps: f64,
}

impl Default for MiniCPMVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 448,
            patch_size: 14,
            layer_norm_eps: 1e-6,
        }
    }
}

impl MiniCPMVisionConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    fn num_patches(&self) -> usize {
        let s = self.num_patches_per_side();
        s * s
    }

    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();
        Self {
            hidden_size: json
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.hidden_size as u64) as usize,
            intermediate_size: json
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.intermediate_size as u64)
                as usize,
            num_hidden_layers: json
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_hidden_layers as u64)
                as usize,
            num_attention_heads: json
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_attention_heads as u64)
                as usize,
            num_channels: json
                .get("num_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_channels as u64) as usize,
            image_size: json
                .get("image_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_size as u64) as usize,
            patch_size: json
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.patch_size as u64) as usize,
            layer_norm_eps: json
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps),
        }
    }
}

/// Top-level MiniCPM-V configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MiniCPMVConfig {
    model_config: ModelConfig,
    vision_config: MiniCPMVisionConfig,
    query_num: usize,
    image_token_id: u32,
}

impl MiniCPMVConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(MiniCPMVisionConfig::from_json)
            .unwrap_or_default();

        let query_num = cfg
            .extra
            .get("query_num")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(128244) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            query_num,
            image_token_id,
        }
    }
}

// ─── 2D Sinusoidal Position Embeddings ──────────────────────────────────────

/// Compute 2D sinusoidal position embeddings.
///
/// Produces a (grid_h * grid_w, embed_dim) tensor where the first half
/// of the embedding dimensions encode height and the second half encode width.
#[allow(dead_code)]
fn get_2d_sincos_pos_embed(
    embed_dim: usize,
    grid_h: usize,
    grid_w: usize,
    device: &Device,
) -> Result<Tensor> {
    let half_dim = embed_dim / 2;

    // 1D sincos embedding for a coordinate sequence
    let sincos_1d = |positions: &[f32], dim: usize| -> Vec<f32> {
        let half = dim / 2;
        let mut result = vec![0f32; positions.len() * dim];
        for (p_idx, &pos) in positions.iter().enumerate() {
            for i in 0..half {
                let omega = 1.0 / (10000.0f32).powf(2.0 * i as f32 / dim as f32);
                let angle = pos * omega;
                result[p_idx * dim + i] = angle.sin();
                result[p_idx * dim + half + i] = angle.cos();
            }
        }
        result
    };

    // Build flattened grid positions
    let num_patches = grid_h * grid_w;
    let mut data = vec![0f32; num_patches * embed_dim];

    // For each grid position, compute combined height+width sincos
    for h in 0..grid_h {
        let h_sincos = sincos_1d(&[h as f32], half_dim);
        for w in 0..grid_w {
            let w_sincos = sincos_1d(&[w as f32], half_dim);
            let idx = (h * grid_w + w) * embed_dim;
            // First half_dim: height encoding
            data[idx..idx + half_dim].copy_from_slice(&h_sincos);
            // Second half_dim: width encoding
            data[idx + half_dim..idx + embed_dim].copy_from_slice(&w_sincos);
        }
    }

    Tensor::from_vec(data, (num_patches, embed_dim), device)
}

// ─── Vision Encoder Components ──────────────────────────────────────────────

/// Idefics2-style vision embeddings: Conv2D patches + learned position embeddings.
#[allow(dead_code)]
struct VisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_embedding: Embedding,
    num_patches_per_side: usize,
}

#[allow(dead_code)]
impl VisionEmbeddings {
    fn new(cfg: &MiniCPMVisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;

        let num_patches = cfg.num_patches();
        let position_embedding =
            embedding(num_patches, cfg.hidden_size, vb.pp("position_embedding"))?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches_per_side: cfg.num_patches_per_side(),
        })
    }

    /// Forward: pixel_values (B, C, H, W) → (B, num_patches, hidden_size)
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Conv2D: [B, C, H, W] → [B, hidden, h_patches, w_patches]
        let patches = self.patch_embedding.forward(pixel_values)?;
        let (_b, _c, _hp, _wp) = patches.dims4()?;

        // Flatten spatial: [B, hidden, num_patches] → [B, num_patches, hidden]
        let embeddings = patches.flatten(2, 3)?.transpose(1, 2)?;

        // Position IDs: simple sequential for fixed-size images
        let num_patches = embeddings.dim(1)?;
        let pos_ids = Tensor::arange(0u32, num_patches as u32, pixel_values.device())?;
        let pos_embeds = self.position_embedding.forward(&pos_ids)?;

        // Add position embeddings: (B, num_patches, hidden) + (num_patches, hidden)
        embeddings.broadcast_add(&pos_embeds)
    }
}

/// Vision attention layer (standard multi-head attention).
#[allow(dead_code)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl VisionAttention {
    fn new(cfg: &MiniCPMVisionConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?;
        let out_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to multi-head: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q * scale)?.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v.contiguous()?)?;

        // Reshape back: (batch, heads, seq, head_dim) → (batch, seq, hidden)
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&out)
    }
}

/// Vision MLP: fc1 → GELU → fc2.
#[allow(dead_code)]
struct VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl VisionMLP {
    fn new(cfg: &MiniCPMVisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = candle_nn::Activation::Gelu.forward(&x)?;
        self.fc2.forward(&x)
    }
}

/// Vision encoder layer: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual.
#[allow(dead_code)]
struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMLP,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

#[allow(dead_code)]
impl VisionEncoderLayer {
    fn new(cfg: &MiniCPMVisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = VisionMLP::new(cfg, vb.pp("mlp"))?;
        let layer_norm1 = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?;
        let layer_norm2 = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.layer_norm1.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;
        let residual = &x;
        let out = self.mlp.forward(&self.layer_norm2.forward(&x)?)?;
        residual + out
    }
}

/// Full vision transformer: embeddings → encoder layers → post-LayerNorm.
#[allow(dead_code)]
struct VisionTransformer {
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    post_layernorm: LayerNorm,
}

#[allow(dead_code)]
impl VisionTransformer {
    fn new(cfg: &MiniCPMVisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_enc = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::new(cfg, vb_enc.pp(i))?);
        }

        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;

        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
        })
    }

    /// Forward: pixel_values (B, C, H, W) → (B, num_patches, hidden_size)
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.post_layernorm.forward(&x)
    }
}

// ─── Perceiver Resampler ────────────────────────────────────────────────────

/// Perceiver Resampler that compresses vision tokens via cross-attention.
///
/// Uses learned queries and 2D sinusoidal position embeddings to produce
/// a fixed number of output tokens regardless of input image resolution.
#[allow(dead_code)]
struct Resampler {
    query: Tensor,
    kv_proj: Option<Linear>,
    ln_q: LayerNorm,
    ln_kv: LayerNorm,
    ln_post: Option<LayerNorm>,
    proj: Option<Tensor>,
    pos_embed: Tensor,
    // Cross-attention weights
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
}

#[allow(dead_code)]
impl Resampler {
    fn new(
        embed_dim: usize,
        num_queries: usize,
        kv_dim: usize,
        num_heads: usize,
        do_post_projection: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let grid_size = (num_queries as f64).sqrt() as usize;

        // Learned queries
        let query = vb.get((num_queries, embed_dim), "query")?;

        // KV projection (if vision dim != embed dim)
        let kv_proj = if kv_dim != embed_dim {
            Some(linear_no_bias(kv_dim, embed_dim, vb.pp("kv_proj"))?)
        } else {
            None
        };

        // Layer norms
        let ln_q = layer_norm(embed_dim, 1e-6, vb.pp("ln_q"))?;
        let ln_kv = layer_norm(embed_dim, 1e-6, vb.pp("ln_kv"))?;

        // Post-projection (optional)
        let ln_post = if do_post_projection {
            Some(layer_norm(embed_dim, 1e-6, vb.pp("ln_post"))?)
        } else {
            None
        };

        let proj = if do_post_projection {
            Some(vb.get((embed_dim, embed_dim), "proj")?)
        } else {
            None
        };

        // 2D sincos position embeddings (precomputed, not learnable)
        let pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, grid_size, vb.device())?
            .to_dtype(vb.dtype())?;

        // Cross-attention weights (MultiheadAttention in_proj packs Q,K,V)
        let vb_attn = vb.pp("attn");
        let in_proj_weight = vb_attn.get((3 * embed_dim, embed_dim), "in_proj_weight")?;
        let in_proj_bias = vb_attn.get(3 * embed_dim, "in_proj_bias")?;
        let out_proj = linear(embed_dim, embed_dim, vb_attn.pp("out_proj"))?;

        Ok(Self {
            query,
            kv_proj,
            ln_q,
            ln_kv,
            ln_post,
            proj,
            pos_embed,
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
            head_dim,
            embed_dim,
        })
    }

    /// Cross-attention forward.
    ///
    /// q: (num_queries, batch, embed_dim)
    /// k: (num_kv, batch, embed_dim)
    /// v: (num_kv, batch, embed_dim)
    fn cross_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (q_len, batch, _) = q.dims3()?;
        let (kv_len, _, _) = k.dims3()?;

        // Project Q using the first third of in_proj_weight/bias
        let w_q = self.in_proj_weight.narrow(0, 0, self.embed_dim)?;
        let b_q = self.in_proj_bias.narrow(0, 0, self.embed_dim)?;
        // q: (q_len, batch, embed_dim) → project
        let q = q
            .reshape((q_len * batch, self.embed_dim))?
            .matmul(&w_q.t()?)?
            .broadcast_add(&b_q)?
            .reshape((q_len, batch, self.embed_dim))?;

        // Project K
        let w_k = self
            .in_proj_weight
            .narrow(0, self.embed_dim, self.embed_dim)?;
        let b_k = self
            .in_proj_bias
            .narrow(0, self.embed_dim, self.embed_dim)?;
        let k = k
            .reshape((kv_len * batch, self.embed_dim))?
            .matmul(&w_k.t()?)?
            .broadcast_add(&b_k)?
            .reshape((kv_len, batch, self.embed_dim))?;

        // Project V
        let w_v = self
            .in_proj_weight
            .narrow(0, 2 * self.embed_dim, self.embed_dim)?;
        let b_v = self
            .in_proj_bias
            .narrow(0, 2 * self.embed_dim, self.embed_dim)?;
        let v = v
            .reshape((kv_len * batch, self.embed_dim))?
            .matmul(&w_v.t()?)?
            .broadcast_add(&b_v)?
            .reshape((kv_len, batch, self.embed_dim))?;

        // Reshape for multi-head: (seq, batch, embed) → (batch, heads, seq, head_dim)
        let q = q
            .reshape((q_len, batch, self.num_heads, self.head_dim))?
            .permute((1, 2, 0, 3))?;
        let k = k
            .reshape((kv_len, batch, self.num_heads, self.head_dim))?
            .permute((1, 2, 0, 3))?;
        let v = v
            .reshape((kv_len, batch, self.num_heads, self.head_dim))?
            .permute((1, 2, 0, 3))?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q * scale)?.matmul(&k.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?;

        // Reshape: (batch, heads, q_len, head_dim) → (q_len, batch, embed)
        let out = out
            .permute((2, 0, 1, 3))?
            .reshape((q_len, batch, self.embed_dim))?;

        // Output projection
        let out = out.reshape((q_len * batch, self.embed_dim))?;
        let out = self.out_proj.forward(&out)?;
        out.reshape((q_len, batch, self.embed_dim))
    }

    /// Resample vision features into fixed-size query tokens.
    ///
    /// x: (batch, num_patches, kv_dim) → (batch, num_queries, embed_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _num_patches, _kv_dim) = x.dims3()?;

        // Project KV if needed
        let x = if let Some(ref kv_proj) = self.kv_proj {
            kv_proj.forward(x)?
        } else {
            x.clone()
        };
        let x = self.ln_kv.forward(&x)?;

        // Transpose to (num_patches, batch, embed_dim) for MHA convention
        let x = x.transpose(0, 1)?;

        // Prepare queries: (num_queries, embed_dim) → broadcast to (num_queries, batch, embed_dim)
        let q = self.ln_q.forward(&self.query)?;
        let q = q
            .unsqueeze(1)?
            .broadcast_as((q.dim(0)?, batch, self.embed_dim))?;

        // Add position embeddings to queries
        let pos_q = self.pos_embed.unsqueeze(1)?; // (num_queries, 1, embed_dim)
        let q = q.broadcast_add(&pos_q)?;

        // Cross-attention: queries attend to vision tokens
        let out = self.cross_attention(&q.contiguous()?, &x.contiguous()?, &x.contiguous()?)?;

        // Transpose back: (num_queries, batch, embed_dim) → (batch, num_queries, embed_dim)
        let out = out.transpose(0, 1)?;

        // Optional post-projection
        if let (Some(ref ln_post), Some(ref proj)) = (&self.ln_post, &self.proj) {
            let out = ln_post.forward(&out)?;
            // Candle 3D @ 2D doesn't broadcast — reshape to 2D, matmul, reshape back
            let (b, nq, _) = out.dims3()?;
            let out = out
                .reshape((b * nq, self.embed_dim))?
                .matmul(proj)?
                .reshape((b, nq, self.embed_dim))?;
            Ok(out)
        } else {
            Ok(out)
        }
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// MiniCPM-V vision-language model (v2.6 with Qwen2 backbone).
pub struct MiniCPMVForConditionalGeneration {
    // Vision
    vpm: VisionTransformer,
    resampler: Resampler,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<Qwen2DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    // Config
    #[allow(dead_code)]
    config: MiniCPMVConfig,
    device: Device,
    dtype: DType,
}

impl MiniCPMVForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = MiniCPMVConfig::from_model_config(cfg);
        let world_size = pg.world_size();

        // Build vision encoder
        let vpm = VisionTransformer::new(&config.vision_config, vb.pp("vpm"))?;

        // Build resampler
        let embed_dim = cfg.hidden_size;
        let num_heads = embed_dim / 128; // MiniCPM-V convention
        let resampler = Resampler::new(
            embed_dim,
            config.query_num,
            config.vision_config.hidden_size,
            num_heads.max(1),
            true, // do_post_projection
            vb.pp("resampler"),
        )?;

        // Build Qwen2 LLM backbone
        let vb_m = vb.pp("llm").pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen2DecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("llm").pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vpm,
            resampler,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Process images through vision encoder + resampler.
    ///
    /// pixel_values: (B, C, H, W) → (B, query_num, llm_hidden_size)
    #[allow(dead_code)]
    fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_output = self.vpm.forward(pixel_values)?;
        self.resampler.forward(&vision_output)
    }

    /// Merge text embeddings with vision embeddings at image token positions.
    fn merge_multimodal_embeddings(
        &self,
        _input_ids: &Tensor,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = text_embeddings.dims3()?;

        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Vision embeddings come pre-encoded. Run through resampler.
            let vision_emb = processed_image.embedding.unsqueeze(0)?;
            let resampled = self.resampler.forward(&vision_emb)?;
            let resampled = resampled.squeeze(0)?; // (query_num, llm_hidden)
            let img_emb: Vec<Vec<f32>> = resampled.to_dtype(DType::F32)?.to_vec2()?;

            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            if batch_idx >= batch_size {
                continue;
            }

            for (i, emb) in img_emb.iter().enumerate() {
                let target_pos = start_pos + i;
                if target_pos < seq_len {
                    merged[batch_idx][target_pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let text_embeddings = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(input_ids, &text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
        } else {
            text_embeddings
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    /// Build initial embeddings (text + merged vision), without running LLM layers.
    ///
    /// Called by `MiniCPMOForCausalLM` to obtain the embedding tensor before
    /// injecting audio embeddings, after which `forward_from_embeddings` runs
    /// the remaining LLM pass.
    pub(crate) fn embed_and_merge_vision(
        &self,
        input_ids: &Tensor,
        mm: Option<&MultimodalInputs>,
    ) -> Result<Tensor> {
        let text_embeddings = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        if let Some(mm_inputs) = mm {
            if mm_inputs.has_images() {
                return self.merge_multimodal_embeddings(input_ids, &text_embeddings, mm_inputs);
            }
        }
        Ok(text_embeddings)
    }

    /// Complete the LLM forward pass from an existing embedding tensor.
    ///
    /// Called by `MiniCPMOForCausalLM` after audio embedding injection.
    pub(crate) fn forward_from_embeddings(
        &self,
        xs: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let mut xs = xs.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }
}

impl crate::engine::ModelForward for MiniCPMVForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        MiniCPMVForConditionalGeneration::forward(
            self,
            input_ids,
            None,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        MiniCPMVForConditionalGeneration::forward(
            self,
            input_ids,
            multimodal_inputs,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_vision_config() -> MiniCPMVisionConfig {
        MiniCPMVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            image_size: 56,
            patch_size: 14,
            layer_norm_eps: 1e-6,
        }
    }

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        let vision = serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 56,
            "patch_size": 14,
            "layer_norm_eps": 1e-6,
        });
        extra.insert("vision_config".to_string(), vision);
        extra.insert("query_num".to_string(), serde_json::json!(4));

        ModelConfig {
            architectures: vec!["MiniCPMV".to_string()],
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
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 16,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    // ─── Config Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = MiniCPMVisionConfig::default();
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim(), 72);
        assert_eq!(cfg.num_patches_per_side(), 32);
        assert_eq!(cfg.num_patches(), 1024);
    }

    #[test]
    fn test_config_from_model_config() {
        let cfg = test_model_config();
        let mcv = MiniCPMVConfig::from_model_config(&cfg);
        assert_eq!(mcv.vision_config.hidden_size, 64);
        assert_eq!(mcv.query_num, 4);
        assert_eq!(mcv.image_token_id, 128244);
    }

    // ─── 2D Sincos Position Embedding Tests ─────────────────────────────────

    #[test]
    fn test_2d_sincos_pos_embed_shape() {
        let device = Device::Cpu;
        let embed = get_2d_sincos_pos_embed(32, 4, 4, &device).unwrap();
        assert_eq!(embed.dims(), &[16, 32]);
    }

    #[test]
    fn test_2d_sincos_pos_embed_rectangular() {
        let device = Device::Cpu;
        let embed = get_2d_sincos_pos_embed(16, 3, 2, &device).unwrap();
        assert_eq!(embed.dims(), &[6, 16]);
    }

    // ─── Vision Component Tests ─────────────────────────────────────────────

    #[test]
    fn test_vision_embeddings() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = VisionEmbeddings::new(&cfg, vb).unwrap();

        // 56x56 image with patch_size=14 → 4x4 = 16 patches
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 56, 56), &device).unwrap();
        let out = emb.forward(&pixel_values).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn test_vision_attention() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = VisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0f32, 1.0, (1, 16, 64), &device).unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn test_vision_mlp() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = VisionMLP::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0f32, 1.0, (1, 16, 64), &device).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn test_vision_encoder_layer() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let layer = VisionEncoderLayer::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0f32, 1.0, (1, 16, 64), &device).unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    #[test]
    fn test_vision_transformer() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vit = VisionTransformer::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 56, 56), &device).unwrap();
        let out = vit.forward(&pixel_values).unwrap();
        assert_eq!(out.dims(), &[1, 16, 64]);
    }

    // ─── Resampler Tests ────────────────────────────────────────────────────

    #[test]
    fn test_resampler_construction() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let resampler = Resampler::new(
            64,   // embed_dim
            4,    // num_queries (2x2 grid)
            64,   // kv_dim (same as embed_dim)
            4,    // num_heads
            true, // do_post_projection
            vb,
        );
        assert!(
            resampler.is_ok(),
            "Resampler should construct: {:?}",
            resampler.err()
        );
    }

    #[test]
    fn test_resampler_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let resampler = Resampler::new(64, 4, 64, 4, true, vb).unwrap();

        let x = Tensor::randn(0f32, 1.0, (1, 16, 64), &device).unwrap();
        let out = resampler.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 64]); // 16 patches → 4 query tokens
    }

    #[test]
    fn test_resampler_kv_projection() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        // kv_dim=32 != embed_dim=64, so kv_proj should be created
        let resampler = Resampler::new(64, 4, 32, 4, true, vb).unwrap();
        assert!(resampler.kv_proj.is_some());

        let x = Tensor::randn(0f32, 1.0, (1, 16, 32), &device).unwrap();
        let out = resampler.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 64]);
    }

    // ─── Full Model Tests ───────────────────────────────────────────────────

    #[test]
    fn test_minicpmv_construction() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPMVForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniCPMVForConditionalGeneration should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_minicpmv_text_only_forward() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPMVForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 5;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                None,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_minicpmv_decode_batch() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPMVForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);
        model
            .forward(
                &prompt,
                None,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let sequences = vec![DecodeSequenceMetadata {
            request_id: 0,
            seqlen_offset: 3,
            block_ids: block_table.block_ids().to_vec(),
            slot_mapping: slot_mapping.clone(),
        }];

        use crate::engine::ModelForward;
        let logits = model
            .forward_decode_batch(&next_token, &sequences, &mut kv_cache_mgr)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_minicpmv_supports_multimodal() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPMVForConditionalGeneration::new(&cfg, vb).unwrap();
        use crate::engine::ModelForward;
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_minicpmv_encode_images() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPMVForConditionalGeneration::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 56, 56), &device).unwrap();
        let out = model.encode_images(&pixel_values).unwrap();
        // 56/14=4 patches per side, 16 total → resampled to query_num=4 tokens, dim=64 (llm hidden)
        assert_eq!(out.dims(), &[1, 4, cfg.hidden_size]);
    }
}
