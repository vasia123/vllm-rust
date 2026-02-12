//! Pixtral vision-language model implementation.
//!
//! Pixtral combines a custom vision transformer with a Mistral language model.
//! The vision encoder uses 2D RoPE for spatial position encoding and processes
//! images at native resolution/aspect ratio without padding.
//!
//! # Architecture
//!
//! 1. Vision encoder: Conv2D patches → RMSNorm → Transformer blocks (2D RoPE + SwiGLU)
//! 2. Adapter: 2-layer GELU MLP (vision_dim → llm_dim)
//! 3. LLM: Mistral backbone (reused from mistral.rs)
//!
//! Images are processed into patch embeddings by the vision encoder, projected
//! to LLM space by the adapter, and inserted at [IMG] token positions.
//!
//! Reference: https://arxiv.org/abs/2410.07073

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d_no_bias, linear, linear_no_bias, rms_norm, Conv2dConfig, Linear, RmsNorm, VarBuilder,
};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;
use crate::multimodal::MultimodalInputs;

use super::mistral::{MistralDecoderLayer, TpContext};
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Pixtral vision encoder configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PixtralVisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    rope_theta: f64,
    adapter_bias: bool,
}

impl Default for PixtralVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 1024,
            patch_size: 16,
            rope_theta: 10000.0,
            adapter_bias: true,
        }
    }
}

impl PixtralVisionConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn max_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
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
            rope_theta: json
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.rope_theta),
            adapter_bias: json
                .get("adapter_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.adapter_bias),
        }
    }
}

/// Top-level Pixtral configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PixtralConfig {
    model_config: ModelConfig,
    vision_config: PixtralVisionConfig,
    image_token_id: u32,
}

impl PixtralConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(PixtralVisionConfig::from_json)
            .unwrap_or_default();

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            image_token_id,
        }
    }
}

// ─── 2D Rotary Position Embeddings ──────────────────────────────────────────

/// Precomputed 2D RoPE frequencies for vision transformer.
///
/// Stores cos/sin tables of shape (max_h, max_w, head_dim/2) indexed by
/// patch grid position. The frequency computation follows Pixtral's approach:
/// even frequency indices encode height, odd indices encode width.
#[allow(dead_code)]
struct RoPE2D {
    cos_table: Tensor, // (max_h, max_w, head_dim/2)
    sin_table: Tensor, // (max_h, max_w, head_dim/2)
}

#[allow(dead_code)]
impl RoPE2D {
    fn new(
        head_dim: usize,
        max_height: usize,
        max_width: usize,
        theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let quarter_dim = half_dim / 2;

        // Compute base frequencies: 1/(theta^(2i/dim)) for i in 0..half_dim
        let mut freqs = vec![0f32; half_dim];
        for (i, freq) in freqs.iter_mut().enumerate().take(half_dim) {
            *freq = 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32);
        }

        // Split into even (height) and odd (width) frequency indices
        let freqs_h: Vec<f32> = freqs.iter().step_by(2).copied().collect();
        let freqs_w: Vec<f32> = freqs.iter().skip(1).step_by(2).copied().collect();

        // Build 2D frequency table: (max_h, max_w, half_dim)
        let mut cos_data = vec![0f32; max_height * max_width * half_dim];
        let mut sin_data = vec![0f32; max_height * max_width * half_dim];

        for h in 0..max_height {
            for w in 0..max_width {
                let base = (h * max_width + w) * half_dim;
                // First quarter_dim elements: height frequencies
                for (i, &freq) in freqs_h.iter().enumerate().take(quarter_dim) {
                    let angle = h as f32 * freq;
                    cos_data[base + i] = angle.cos();
                    sin_data[base + i] = angle.sin();
                }
                // Second quarter_dim elements: width frequencies
                for (i, &freq) in freqs_w.iter().enumerate().take(quarter_dim) {
                    let angle = w as f32 * freq;
                    cos_data[base + quarter_dim + i] = angle.cos();
                    sin_data[base + quarter_dim + i] = angle.sin();
                }
            }
        }

        let cos_table = Tensor::from_vec(cos_data, (max_height, max_width, half_dim), device)?;
        let sin_table = Tensor::from_vec(sin_data, (max_height, max_width, half_dim), device)?;

        Ok(Self {
            cos_table,
            sin_table,
        })
    }

    /// Gather cos/sin for given (h, w) positions.
    ///
    /// positions: Vec<(usize, usize)> — (row, col) for each patch
    /// Returns: (cos, sin) each of shape (num_patches, head_dim/2)
    fn gather(&self, positions: &[(usize, usize)], device: &Device) -> Result<(Tensor, Tensor)> {
        let (max_h, max_w, half_dim) = self.cos_table.dims3()?;

        let mut cos_data = vec![0f32; positions.len() * half_dim];
        let mut sin_data = vec![0f32; positions.len() * half_dim];

        let cos_vec: Vec<f32> = self.cos_table.flatten_all()?.to_vec1()?;
        let sin_vec: Vec<f32> = self.sin_table.flatten_all()?.to_vec1()?;

        for (idx, &(h, w)) in positions.iter().enumerate() {
            let h = h.min(max_h - 1);
            let w = w.min(max_w - 1);
            let src_offset = (h * max_w + w) * half_dim;
            let dst_offset = idx * half_dim;
            cos_data[dst_offset..dst_offset + half_dim]
                .copy_from_slice(&cos_vec[src_offset..src_offset + half_dim]);
            sin_data[dst_offset..dst_offset + half_dim]
                .copy_from_slice(&sin_vec[src_offset..src_offset + half_dim]);
        }

        let cos = Tensor::from_vec(cos_data, (positions.len(), half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (positions.len(), half_dim), device)?;
        Ok((cos, sin))
    }
}

/// Apply 2D rotary embeddings to query and key tensors.
///
/// q, k: (batch, num_patches, num_heads, head_dim)
/// cos, sin: (num_patches, head_dim/2)
///
/// Uses complex multiplication: (a + bi)(cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
/// where pairs (a, b) come from adjacent elements of the head dimension.
#[allow(dead_code)]
fn apply_rotary_emb_2d(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, patches, heads, dim) = q.dims4()?;
    let half_dim = dim / 2;

    // Reshape to pair adjacent elements: (..., half_dim, 2)
    let q = q.reshape((batch, patches, heads, half_dim, 2))?;
    let k = k.reshape((batch, patches, heads, half_dim, 2))?;

    let q_even = q.narrow(4, 0, 1)?.squeeze(4)?; // (batch, patches, heads, half_dim)
    let q_odd = q.narrow(4, 1, 1)?.squeeze(4)?;
    let k_even = k.narrow(4, 0, 1)?.squeeze(4)?;
    let k_odd = k.narrow(4, 1, 1)?.squeeze(4)?;

    // Broadcast cos/sin: (1, patches, 1, half_dim)
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Complex multiply: out = (even * cos - odd * sin, even * sin + odd * cos)
    let q_out_even = (q_even.broadcast_mul(&cos)? - q_odd.broadcast_mul(&sin)?)?;
    let q_out_odd = (q_even.broadcast_mul(&sin)? + q_odd.broadcast_mul(&cos)?)?;
    let k_out_even = (k_even.broadcast_mul(&cos)? - k_odd.broadcast_mul(&sin)?)?;
    let k_out_odd = (k_even.broadcast_mul(&sin)? + k_odd.broadcast_mul(&cos)?)?;

    // Interleave back: stack on last dim then flatten
    let q_out =
        Tensor::stack(&[&q_out_even, &q_out_odd], 4)?.reshape((batch, patches, heads, dim))?;
    let k_out =
        Tensor::stack(&[&k_out_even, &k_out_odd], 4)?.reshape((batch, patches, heads, dim))?;

    Ok((q_out, k_out))
}

// ─── Vision Encoder Components ──────────────────────────────────────────────

/// SwiGLU feed-forward network for the vision encoder.
///
/// gate_proj (w1) and up_proj (w3) are parallel, followed by down_proj (w2):
/// output = w2(silu(w1(x)) * w3(x))
#[allow(dead_code)]
struct PixtralVisionMLP {
    w1: Linear, // gate
    w2: Linear, // down
    w3: Linear, // up
}

#[allow(dead_code)]
impl PixtralVisionMLP {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w1"))?;
        let w2 = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("w2"))?;
        let w3 = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// Vision attention layer with 2D RoPE.
#[allow(dead_code)]
struct PixtralVisionAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl PixtralVisionAttention {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let wq = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("wq"))?;
        let wk = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("wk"))?;
        let wv = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("wv"))?;
        let wo = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("wo"))?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Forward with 2D RoPE and optional attention mask.
    ///
    /// x: (batch, num_patches, hidden_size)
    /// mask: optional (1, 1, num_patches, num_patches)
    /// cos, sin: (num_patches, head_dim/2) — 2D position frequencies
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, patches, _) = x.dims3()?;

        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape to (batch, patches, num_heads, head_dim)
        let q = q.reshape((batch, patches, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, patches, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, patches, self.num_heads, self.head_dim))?;

        // Apply 2D RoPE
        let (q, k) = apply_rotary_emb_2d(&q, &k, cos, sin)?;

        // Transpose to (batch, heads, patches, head_dim) for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q * scale)?.matmul(&k.transpose(2, 3)?)?;

        // Apply attention mask (block-diagonal for multi-image)
        let attn_weights = if let Some(mask) = mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v)?;

        // Reshape back: (batch, heads, patches, head_dim) -> (batch, patches, hidden)
        let out = out
            .transpose(1, 2)?
            .reshape((batch, patches, self.num_heads * self.head_dim))?;

        self.wo.forward(&out)
    }
}

/// Vision transformer block: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual.
#[allow(dead_code)]
struct PixtralVisionBlock {
    attention: PixtralVisionAttention,
    feed_forward: PixtralVisionMLP,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

#[allow(dead_code)]
impl PixtralVisionBlock {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let attention = PixtralVisionAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = PixtralVisionMLP::new(cfg, vb.pp("feed_forward"))?;
        let attention_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let r = self
            .attention
            .forward(&self.attention_norm.forward(x)?, mask, cos, sin)?;
        let h = (x + r)?;
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        &h + r
    }
}

/// Build a block-diagonal attention mask for multiple images.
///
/// Each image's patches can attend to each other but not to patches from other images.
/// Returns a mask of shape (1, 1, total_patches, total_patches) with 0 for allowed
/// positions and -inf for blocked positions.
#[allow(dead_code)]
fn build_block_diagonal_mask(
    embed_sizes: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total: usize = embed_sizes.iter().sum();
    if embed_sizes.len() <= 1 {
        // Single image: no mask needed (all patches attend to each other)
        return Tensor::zeros((1, 1, total, total), dtype, device);
    }

    let mut mask = vec![f32::NEG_INFINITY; total * total];
    let mut offset = 0;
    for &size in embed_sizes {
        for i in 0..size {
            for j in 0..size {
                mask[(offset + i) * total + offset + j] = 0.0;
            }
        }
        offset += size;
    }

    Tensor::from_vec(mask, (1, 1, total, total), device)?.to_dtype(dtype)
}

/// Compute 2D position grid for patch embeddings.
///
/// For each image's patch grid (h_patches × w_patches), generates (row, col)
/// positions for each patch token.
#[allow(dead_code)]
fn position_meshgrid(patch_grid_sizes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    for &(h_patches, w_patches) in patch_grid_sizes {
        for h in 0..h_patches {
            for w in 0..w_patches {
                positions.push((h, w));
            }
        }
    }
    positions
}

/// Pixtral vision transformer.
///
/// Processes images through Conv2D patch embedding, RMSNorm, and transformer
/// blocks with 2D RoPE and block-diagonal attention masks.
#[allow(dead_code)]
struct PixtralVisionTransformer {
    patch_conv: candle_nn::Conv2d,
    ln_pre: RmsNorm,
    layers: Vec<PixtralVisionBlock>,
    rope_2d: RoPE2D,
    #[allow(dead_code)]
    config: PixtralVisionConfig,
}

#[allow(dead_code)]
impl PixtralVisionTransformer {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_conv = conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_conv"),
        )?;

        let ln_pre = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ln_pre"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("transformer").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(PixtralVisionBlock::new(cfg, vb_layers.pp(i))?);
        }

        let max_patches = cfg.max_patches_per_side();
        let rope_2d = RoPE2D::new(
            cfg.head_dim(),
            max_patches,
            max_patches,
            cfg.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            patch_conv,
            ln_pre,
            layers,
            rope_2d,
            config: cfg.clone(),
        })
    }

    /// Process a list of images through the vision encoder.
    ///
    /// Each image is (C, H, W) with potentially different sizes.
    /// Returns a list of (num_patches, hidden_size) tensors, one per image.
    fn forward(&self, images: &[Tensor]) -> Result<Vec<Tensor>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let device = images[0].device();
        let dtype = images[0].dtype();

        // Process each image through Conv2d independently
        let mut patch_embeds_list = Vec::with_capacity(images.len());
        let mut patch_grid_sizes = Vec::with_capacity(images.len());
        let mut embed_sizes = Vec::with_capacity(images.len());

        for img in images {
            // img: (C, H, W) → (1, C, H, W) for Conv2d
            let img = img.unsqueeze(0)?;
            let patches = self.patch_conv.forward(&img)?; // (1, hidden, h_patches, w_patches)
            let (_, _c, h_p, w_p) = patches.dims4()?;

            // Flatten spatial dims: (1, hidden, h*w) → (1, h*w, hidden)
            let patches = patches.flatten(2, 3)?.transpose(1, 2)?;
            let num_patches = h_p * w_p;

            embed_sizes.push(num_patches);
            patch_grid_sizes.push((h_p, w_p));
            patch_embeds_list.push(patches);
        }

        // Concatenate all patches: (1, total_patches, hidden)
        let patch_embeds = Tensor::cat(&patch_embeds_list, 1)?;
        let patch_embeds = self.ln_pre.forward(&patch_embeds)?;

        // Compute 2D positions and gather RoPE frequencies
        let positions = position_meshgrid(&patch_grid_sizes);
        let (cos, sin) = self.rope_2d.gather(&positions, device)?;
        let cos = cos.to_dtype(dtype)?;
        let sin = sin.to_dtype(dtype)?;

        // Build block-diagonal mask if multiple images
        let mask = if images.len() > 1 {
            Some(build_block_diagonal_mask(&embed_sizes, dtype, device)?)
        } else {
            None
        };

        // Forward through transformer blocks
        let mut x = patch_embeds;
        for layer in &self.layers {
            x = layer.forward(&x, mask.as_ref(), &cos, &sin)?;
        }

        // Split output back into per-image tensors
        let x = x.squeeze(0)?; // (total_patches, hidden)
        let mut results = Vec::with_capacity(images.len());
        let mut offset = 0;
        for &size in &embed_sizes {
            results.push(x.narrow(0, offset, size)?);
            offset += size;
        }

        Ok(results)
    }
}

// ─── Vision-Language Adapter ────────────────────────────────────────────────

/// Two-layer GELU MLP that projects vision embeddings to LLM hidden dimension.
#[allow(dead_code)]
struct PixtralAdapter {
    w_in: Linear,
    w_out: Linear,
}

#[allow(dead_code)]
impl PixtralAdapter {
    fn new(vision_dim: usize, llm_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let w_in = if bias {
            linear(vision_dim, llm_dim, vb.pp("w_in"))?
        } else {
            linear_no_bias(vision_dim, llm_dim, vb.pp("w_in"))?
        };
        let w_out = if bias {
            linear(llm_dim, llm_dim, vb.pp("w_out"))?
        } else {
            linear_no_bias(llm_dim, llm_dim, vb.pp("w_out"))?
        };
        Ok(Self { w_in, w_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w_in.forward(x)?;
        let x = candle_nn::Activation::Gelu.forward(&x)?;
        self.w_out.forward(&x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Pixtral vision-language model.
///
/// Combines a custom vision encoder with Mistral LLM backbone.
pub struct PixtralForConditionalGeneration {
    // Vision
    visual: PixtralVisionTransformer,
    adapter: PixtralAdapter,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<MistralDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    // Config
    #[allow(dead_code)]
    config: PixtralConfig,
    device: Device,
    dtype: DType,
}

impl PixtralForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = PixtralConfig::from_model_config(cfg);
        let world_size = pg.world_size();

        // Build vision encoder + adapter
        let vision_vb = vb.pp("vision_encoder");
        let visual = PixtralVisionTransformer::new(&config.vision_config, vision_vb)?;

        let adapter = PixtralAdapter::new(
            config.vision_config.hidden_size,
            cfg.hidden_size,
            config.vision_config.adapter_bias,
            vb.pp("vision_language_adapter"),
        )?;

        // Build LLM (reuse Mistral decoder layers)
        // Pixtral LLM has no sliding window — use a config with sliding_window = None
        let mut llm_cfg = cfg.clone();
        llm_cfg.sliding_window = None;

        let vb_m = vb.pp("language_model").pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MistralDecoderLayer::new_with_tp(&llm_cfg, vb_l.pp(i), pg)?);
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
                vb.pp("language_model").pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            visual,
            adapter,
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

    /// Process images through vision encoder and adapter.
    ///
    /// Returns a single tensor of all image tokens concatenated:
    /// (total_image_tokens, llm_hidden_size)
    #[allow(dead_code)]
    fn encode_images(&self, images: &[Tensor]) -> Result<Vec<Tensor>> {
        let image_features = self.visual.forward(images)?;
        let mut projected = Vec::with_capacity(image_features.len());
        for feat in &image_features {
            // feat: (num_patches, vision_hidden) → add batch dim → project → remove batch
            let feat = feat.unsqueeze(0)?;
            let proj = self.adapter.forward(&feat)?;
            projected.push(proj.squeeze(0)?);
        }
        Ok(projected)
    }

    /// Merge text embeddings with vision embeddings at image token positions.
    fn merge_multimodal_embeddings(
        &self,
        input_ids: &Tensor,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = text_embeddings.dims3()?;
        let _input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Processed image embeddings are already in vision space.
            // Project to LLM space via adapter.
            let vision_emb = processed_image.embedding.unsqueeze(0)?;
            let projected = self.adapter.forward(&vision_emb)?;
            let projected = projected.squeeze(0)?;
            let img_emb: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

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

        // Get text embeddings
        let text_embeddings = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        // Merge with image embeddings if present
        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(input_ids, &text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
        } else {
            text_embeddings
        };

        // Forward through LLM decoder layers
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

impl crate::engine::ModelForward for PixtralForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        PixtralForConditionalGeneration::forward(
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
        // During decode, no image tokens (already processed in prefill)
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
        PixtralForConditionalGeneration::forward(
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

    fn test_vision_config() -> PixtralVisionConfig {
        PixtralVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            image_size: 64,
            patch_size: 16,
            rope_theta: 10000.0,
            adapter_bias: true,
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
            "image_size": 64,
            "patch_size": 16,
            "rope_theta": 10000.0,
            "adapter_bias": true,
        });
        extra.insert("vision_config".to_string(), vision);
        extra.insert("image_token_id".to_string(), serde_json::json!(10));

        ModelConfig {
            architectures: vec!["PixtralForConditionalGeneration".to_string()],
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
    fn test_pixtral_vision_config_defaults() {
        let cfg = PixtralVisionConfig::default();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.max_patches_per_side(), 64);
    }

    #[test]
    fn test_pixtral_config_from_model_config() {
        let cfg = test_model_config();
        let pixtral_cfg = PixtralConfig::from_model_config(&cfg);
        assert_eq!(pixtral_cfg.vision_config.hidden_size, 64);
        assert_eq!(pixtral_cfg.vision_config.patch_size, 16);
        assert_eq!(pixtral_cfg.image_token_id, 10);
    }

    // ─── 2D RoPE Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_rope_2d_construction() {
        let device = Device::Cpu;
        let rope = RoPE2D::new(16, 4, 4, 10000.0, &device).expect("build RoPE2D");
        assert_eq!(rope.cos_table.dims(), &[4, 4, 8]);
        assert_eq!(rope.sin_table.dims(), &[4, 4, 8]);
    }

    #[test]
    fn test_rope_2d_gather() {
        let device = Device::Cpu;
        let rope = RoPE2D::new(16, 4, 4, 10000.0, &device).expect("build RoPE2D");
        let positions = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let (cos, sin) = rope.gather(&positions, &device).expect("gather");
        assert_eq!(cos.dims(), &[4, 8]);
        assert_eq!(sin.dims(), &[4, 8]);
    }

    #[test]
    fn test_rope_2d_origin_is_ones() {
        let device = Device::Cpu;
        let rope = RoPE2D::new(16, 4, 4, 10000.0, &device).expect("build RoPE2D");
        let positions = vec![(0, 0)];
        let (cos, _sin) = rope.gather(&positions, &device).expect("gather");
        // At position (0, 0), all angles are 0, so cos should be 1.0
        let cos_vals: Vec<f32> = cos.flatten_all().expect("flatten").to_vec1().expect("vec");
        for &v in &cos_vals {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "cos at origin should be 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_apply_rotary_emb_2d_shape() {
        let device = Device::Cpu;
        let batch = 1;
        let patches = 4;
        let heads = 2;
        let dim = 16;
        let half_dim = dim / 2;

        let q = Tensor::randn(0f32, 1.0, (batch, patches, heads, dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (batch, patches, heads, dim), &device).unwrap();
        let cos = Tensor::ones((patches, half_dim), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((patches, half_dim), DType::F32, &device).unwrap();

        let (q_out, k_out) = apply_rotary_emb_2d(&q, &k, &cos, &sin).unwrap();
        assert_eq!(q_out.dims(), &[batch, patches, heads, dim]);
        assert_eq!(k_out.dims(), &[batch, patches, heads, dim]);
    }

    #[test]
    fn test_apply_rotary_emb_2d_identity() {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 4, 2, 16), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 4, 2, 16), &device).unwrap();
        // cos=1, sin=0 should give identity rotation
        let cos = Tensor::ones((4, 8), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((4, 8), DType::F32, &device).unwrap();

        let (q_out, _k_out) = apply_rotary_emb_2d(&q, &k, &cos, &sin).unwrap();
        let diff = (q_out - &q).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            max_diff < 1e-5,
            "Identity rotation should preserve values, diff={max_diff}"
        );
    }

    // ─── Position Grid Tests ────────────────────────────────────────────────

    #[test]
    fn test_position_meshgrid() {
        let sizes = vec![(2, 3), (1, 2)];
        let positions = position_meshgrid(&sizes);
        assert_eq!(positions.len(), 2 * 3 + 1 * 2);
        assert_eq!(positions[0], (0, 0));
        assert_eq!(positions[1], (0, 1));
        assert_eq!(positions[2], (0, 2));
        assert_eq!(positions[3], (1, 0));
        assert_eq!(positions[5], (1, 2));
        // Second image starts at index 6
        assert_eq!(positions[6], (0, 0));
        assert_eq!(positions[7], (0, 1));
    }

    // ─── Block Diagonal Mask Tests ──────────────────────────────────────────

    #[test]
    fn test_block_diagonal_mask_single_image() {
        let device = Device::Cpu;
        let mask = build_block_diagonal_mask(&[4], DType::F32, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        // All zeros for single image
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_block_diagonal_mask_multi_image() {
        let device = Device::Cpu;
        let mask = build_block_diagonal_mask(&[2, 3], DType::F32, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 5, 5]);
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Image 1 block (2x2) at top-left should be 0
        assert_eq!(vals[0], 0.0); // (0,0)
        assert_eq!(vals[1], 0.0); // (0,1)
                                  // Cross-image should be -inf
        assert!(vals[2].is_infinite()); // (0,2) — image 1 → image 2
                                        // Image 2 block (3x3) at bottom-right should be 0
        assert_eq!(vals[5 * 2 + 2], 0.0); // (2,2)
    }

    // ─── Vision Component Tests ─────────────────────────────────────────────

    #[test]
    fn test_vision_mlp_forward() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = PixtralVisionMLP::new(&cfg, vb).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 4, cfg.hidden_size), &device).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn test_vision_attention_forward() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = PixtralVisionAttention::new(&cfg, vb).unwrap();

        let patches = 4;
        let half_dim = cfg.head_dim() / 2;
        let x = Tensor::randn(0f32, 1.0, (1, patches, cfg.hidden_size), &device).unwrap();
        let cos = Tensor::ones((patches, half_dim), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((patches, half_dim), DType::F32, &device).unwrap();

        let out = attn.forward(&x, None, &cos, &sin).unwrap();
        assert_eq!(out.dims(), &[1, patches, cfg.hidden_size]);
    }

    #[test]
    fn test_vision_block_forward() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = PixtralVisionBlock::new(&cfg, vb).unwrap();

        let patches = 4;
        let half_dim = cfg.head_dim() / 2;
        let x = Tensor::randn(0f32, 1.0, (1, patches, cfg.hidden_size), &device).unwrap();
        let cos = Tensor::ones((patches, half_dim), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((patches, half_dim), DType::F32, &device).unwrap();

        let out = block.forward(&x, None, &cos, &sin).unwrap();
        assert_eq!(out.dims(), &[1, patches, cfg.hidden_size]);
    }

    #[test]
    fn test_vision_transformer_forward() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vit = PixtralVisionTransformer::new(&cfg, vb).unwrap();

        // Single 48x32 image → 3x2 patches
        let img = Tensor::randn(0f32, 1.0, (3, 48, 32), &device).unwrap();
        let results = vit.forward(&[img]).unwrap();
        assert_eq!(results.len(), 1);
        // 48/16 = 3, 32/16 = 2 → 6 patches
        assert_eq!(results[0].dims(), &[6, cfg.hidden_size]);
    }

    #[test]
    fn test_vision_transformer_multi_image() {
        let cfg = test_vision_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vit = PixtralVisionTransformer::new(&cfg, vb).unwrap();

        // Two images of different sizes
        let img1 = Tensor::randn(0f32, 1.0, (3, 32, 32), &device).unwrap();
        let img2 = Tensor::randn(0f32, 1.0, (3, 48, 16), &device).unwrap();
        let results = vit.forward(&[img1, img2]).unwrap();
        assert_eq!(results.len(), 2);
        // 32/16=2, 32/16=2 → 4 patches
        assert_eq!(results[0].dims(), &[4, cfg.hidden_size]);
        // 48/16=3, 16/16=1 → 3 patches
        assert_eq!(results[1].dims(), &[3, cfg.hidden_size]);
    }

    // ─── Adapter Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_adapter_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let adapter = PixtralAdapter::new(64, 128, true, vb).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &device).unwrap();
        let out = adapter.forward(&x).unwrap();
        assert_eq!(out.dims(), &[4, 128]);
    }

    // ─── Full Model Tests ───────────────────────────────────────────────────

    #[test]
    fn test_pixtral_construction() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PixtralForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PixtralForConditionalGeneration should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_pixtral_text_only_forward() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PixtralForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_pixtral_decode_batch() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PixtralForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill first
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

        // Decode step
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
    fn test_pixtral_supports_multimodal() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PixtralForConditionalGeneration::new(&cfg, vb).unwrap();
        use crate::engine::ModelForward;
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_pixtral_encode_images() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PixtralForConditionalGeneration::new(&cfg, vb).unwrap();

        let img = Tensor::randn(0f32, 1.0, (3, 32, 32), &device).unwrap();
        let projected = model.encode_images(&[img]).unwrap();
        assert_eq!(projected.len(), 1);
        // 32/16=2, 32/16=2 → 4 patches, projected to llm_hidden=64
        assert_eq!(projected[0].dims(), &[4, cfg.hidden_size]);
    }
}
