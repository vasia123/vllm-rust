//! InternS1 vision-language model implementation.
//!
//! InternS1 uses the InternS1ViT vision encoder (separate Q/K/V projections,
//! layernorm_before/after naming, `encoder.layer` index) with an MLP projector
//! and a registered language model backbone (typically InternLM2).
//!
//! Key differences from InternVL:
//! - Vision attention: separate q/k/v projections (not fused QKV)
//! - Vision attention: QK norm on full embed_dim output (not per head_dim)
//! - Vision encoder: weight paths use `encoder.layer.{i}` (singular "layer")
//! - Vision embeddings: `patch_embeddings.projection`, `cls_token`, `position_embeddings`
//! - Vision layer norms: `layernorm_before` / `layernorm_after` (not `norm1`/`norm2`)
//! - Pixel shuffle: different reshaping pattern than InternVL
//! - Top-level paths: `vision_tower`, `multi_modal_projector`, `language_model`
//!
//! Reference: InternS1 (https://github.com/OpenGVLab/InternS1)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    layer_norm, linear, linear_b, rms_norm, Conv2d, Conv2dConfig, LayerNorm, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::internlm2::InternLM2ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// InternS1 vision encoder (InternS1ViT) configuration.
#[derive(Debug, Clone)]
pub struct InternS1VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    /// image_size[0]: height (square images, so [H, H] or just H)
    pub image_size: usize,
    /// patch_size[0]: height of each patch (square patches)
    pub patch_size: usize,
    pub attention_bias: bool,
    pub use_qk_norm: bool,
    pub layer_norm_eps: f64,
    /// Whether to use RMSNorm (true) or LayerNorm (false) in vision encoder.
    pub use_rms_norm: bool,
    pub use_absolute_position_embeddings: bool,
    pub use_mean_pooling: bool,
    pub layer_scale_init_value: f64,
    /// MLP activation in vision encoder (e.g. "gelu").
    pub hidden_act: String,
}

impl Default for InternS1VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3200,
            intermediate_size: 12800,
            num_attention_heads: 25,
            num_hidden_layers: 45,
            image_size: 448,
            patch_size: 14,
            attention_bias: true,
            use_qk_norm: true,
            layer_norm_eps: 1e-6,
            use_rms_norm: true,
            use_absolute_position_embeddings: true,
            use_mean_pooling: false,
            layer_scale_init_value: 0.1,
            hidden_act: "gelu".to_string(),
        }
    }
}

impl InternS1VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();

        // image_size can be int or [int, int]
        let image_size = if let Some(arr) = json.get("image_size").and_then(|v| v.as_array()) {
            arr[0].as_u64().unwrap_or(defaults.image_size as u64) as usize
        } else {
            json.get("image_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_size as u64) as usize
        };

        // patch_size can be int or [int, int]
        let patch_size = if let Some(arr) = json.get("patch_size").and_then(|v| v.as_array()) {
            arr[0].as_u64().unwrap_or(defaults.patch_size as u64) as usize
        } else {
            json.get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.patch_size as u64) as usize
        };

        let norm_type = json
            .get("norm_type")
            .and_then(|v| v.as_str())
            .unwrap_or("rms_norm");

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
            num_attention_heads: json
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_attention_heads as u64)
                as usize,
            num_hidden_layers: json
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_hidden_layers as u64)
                as usize,
            image_size,
            patch_size,
            attention_bias: json
                .get("attention_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.attention_bias),
            use_qk_norm: json
                .get("use_qk_norm")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.use_qk_norm),
            layer_norm_eps: json
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps),
            use_rms_norm: norm_type == "rms_norm",
            use_absolute_position_embeddings: json
                .get("use_absolute_position_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.use_absolute_position_embeddings),
            use_mean_pooling: json
                .get("use_mean_pooling")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.use_mean_pooling),
            layer_scale_init_value: json
                .get("layer_scale_init_value")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_scale_init_value),
            hidden_act: json
                .get("hidden_act")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.hidden_act)
                .to_string(),
        }
    }
}

/// Top-level InternS1 configuration.
#[derive(Debug, Clone)]
pub struct InternS1Config {
    pub model_config: ModelConfig,
    pub vision_config: InternS1VisionConfig,
    pub downsample_ratio: f64,
    pub image_token_id: u32,
    pub projector_hidden_act: String,
}

impl InternS1Config {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(InternS1VisionConfig::from_json)
            .unwrap_or_default();

        let downsample_ratio = cfg
            .extra
            .get("downsample_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151667) as u32;

        let projector_hidden_act = cfg
            .extra
            .get("projector_hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string();

        Self {
            model_config: cfg.clone(),
            vision_config,
            downsample_ratio,
            image_token_id,
            projector_hidden_act,
        }
    }

    /// Projector input dim: vision_hidden * (1/downsample_ratio)^2.
    pub fn projector_input_dim(&self) -> usize {
        let scale = (1.0 / self.downsample_ratio).round() as usize;
        self.vision_config.hidden_size * scale * scale
    }
}

// ─── Norm helper ─────────────────────────────────────────────────────────────

/// Either LayerNorm or RmsNorm, chosen by vision config.
#[allow(dead_code)]
enum InternS1Norm {
    LayerNorm(LayerNorm),
    RmsNorm(candle_nn::RmsNorm),
}

#[allow(dead_code)]
impl InternS1Norm {
    fn new(size: usize, eps: f64, use_rms: bool, vb: VarBuilder) -> Result<Self> {
        if use_rms {
            Ok(Self::RmsNorm(rms_norm(size, eps, vb)?))
        } else {
            Ok(Self::LayerNorm(layer_norm(size, eps, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(ln) => ln.forward(x),
            Self::RmsNorm(rn) => rn.forward(x),
        }
    }
}

// ─── Vision Encoder ─────────────────────────────────────────────────────────

/// Vision embeddings: Conv2d patch projection + CLS token + optional position embeddings.
///
/// Weight paths (under the `embeddings` prefix):
/// - `patch_embeddings.projection.weight` / `.bias`
/// - `cls_token`
/// - `position_embeddings` (optional, if use_absolute_position_embeddings)
#[allow(dead_code)]
struct InternS1VisionEmbeddings {
    cls_token: Tensor,
    patch_projection: Conv2d,
    position_embeddings: Option<Tensor>,
    num_patches: usize,
}

#[allow(dead_code)]
impl InternS1VisionEmbeddings {
    fn new(cfg: &InternS1VisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_patches = cfg.num_patches();

        let cls_token = vb.get((1, 1, cfg.hidden_size), "cls_token")?;

        // Patch projection: at `patch_embeddings.projection`
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_projection = candle_nn::conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embeddings").pp("projection"),
        )?;

        // Optional position embeddings
        let position_embeddings = if cfg.use_absolute_position_embeddings {
            Some(vb.get((1, num_patches + 1, cfg.hidden_size), "position_embeddings")?)
        } else {
            None
        };

        Ok(Self {
            cls_token,
            patch_projection,
            position_embeddings,
            num_patches,
        })
    }

    /// pixel_values: [B, 3, H, W] -> [B, num_patches+1, hidden_size]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;

        // Conv2d: [B, 3, H, W] -> [B, hidden, h, w]
        let patches = self.patch_projection.forward(pixel_values)?;
        let (_b, c, _h, _w) = patches.dims4()?;

        // Flatten: [B, hidden, num_patches] -> [B, num_patches, hidden]
        let patches = patches
            .reshape((batch_size, c, self.num_patches))?
            .transpose(1, 2)?;

        // Prepend CLS token: [B, 1, hidden]
        let cls = self
            .cls_token
            .broadcast_left(batch_size)?
            .reshape((batch_size, 1, c))?;
        let embeddings = Tensor::cat(&[cls, patches], 1)?;

        // Add position embeddings if present
        if let Some(ref pos_emb) = self.position_embeddings {
            embeddings + pos_emb
        } else {
            Ok(embeddings)
        }
    }
}

/// Vision attention with separate Q/K/V projections and optional QK normalization.
///
/// QK normalization is applied to the full `[B, seq, embed_dim]` Q/K before attention,
/// unlike InternVL which normalizes per-head (`[B, heads, seq, head_dim]`).
///
/// Weight paths (under `attention`):
/// - `q_proj.weight`, `k_proj.weight`, `v_proj.weight` (+ optional `.bias`)
/// - `q_norm.weight`, `k_norm.weight` (optional, size `embed_dim`)
/// - `projection_layer.weight` / `.bias`
#[allow(dead_code)]
struct InternS1VisionAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    q_norm: Option<candle_nn::RmsNorm>,
    k_norm: Option<candle_nn::RmsNorm>,
    projection_layer: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl InternS1VisionAttention {
    fn new(cfg: &InternS1VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let head_dim = cfg.head_dim();

        let q_proj = linear_b(embed_dim, embed_dim, cfg.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(embed_dim, embed_dim, cfg.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(embed_dim, embed_dim, cfg.attention_bias, vb.pp("v_proj"))?;

        let (q_norm, k_norm) = if cfg.use_qk_norm {
            // QK norm operates on full embed_dim (not per head_dim)
            (
                Some(rms_norm(embed_dim, cfg.layer_norm_eps, vb.pp("q_norm"))?),
                Some(rms_norm(embed_dim, cfg.layer_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        // projection_layer (biased by default in Python nn.Linear)
        let projection_layer = linear(embed_dim, embed_dim, vb.pp("projection_layer"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            q_norm,
            k_norm,
            projection_layer,
            num_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    /// x: [B, seq, hidden] -> [B, seq, hidden]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        // Separate Q/K/V projections -> [B, seq, embed_dim]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Optional QK normalization (on full embed_dim, before head reshape)
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        // Reshape to [B, heads, seq, head_dim]
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention (bidirectional, no causal mask)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, heads, seq, head_dim] -> [B, seq, embed_dim]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.projection_layer.forward(&attn_output)
    }
}

/// Vision MLP: fc1 + GELU + fc2 (biased).
#[allow(dead_code)]
struct InternS1VisionMLP {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

#[allow(dead_code)]
impl InternS1VisionMLP {
    fn new(cfg: &InternS1VisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python uses config.hidden_act; InternS1ViT defaults to GELU
        let x = self.fc1.forward(x)?.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

/// Vision encoder layer with residual scaling.
///
/// Forward:
///   x = x + attention(layernorm_before(x)) * lambda_1
///   x = x + mlp(layernorm_after(x)) * lambda_2
///
/// Weight paths (under `encoder.layer.{i}`):
/// - `attention.*`, `mlp.*`
/// - `layernorm_before.*`, `layernorm_after.*`
/// - `lambda_1`, `lambda_2`
#[allow(dead_code)]
struct InternS1VisionLayer {
    attention: InternS1VisionAttention,
    mlp: InternS1VisionMLP,
    layernorm_before: InternS1Norm,
    layernorm_after: InternS1Norm,
    lambda_1: Tensor,
    lambda_2: Tensor,
}

#[allow(dead_code)]
impl InternS1VisionLayer {
    fn new(cfg: &InternS1VisionConfig, vb: VarBuilder) -> Result<Self> {
        let attention = InternS1VisionAttention::new(cfg, vb.pp("attention"))?;
        let mlp = InternS1VisionMLP::new(cfg, vb.pp("mlp"))?;
        let layernorm_before = InternS1Norm::new(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            cfg.use_rms_norm,
            vb.pp("layernorm_before"),
        )?;
        let layernorm_after = InternS1Norm::new(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            cfg.use_rms_norm,
            vb.pp("layernorm_after"),
        )?;

        let lambda_1 = vb.get(cfg.hidden_size, "lambda_1")?;
        let lambda_2 = vb.get(cfg.hidden_size, "lambda_2")?;

        Ok(Self {
            attention,
            mlp,
            layernorm_before,
            layernorm_after,
            lambda_1,
            lambda_2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn_out = self.attention.forward(&self.layernorm_before.forward(x)?)?;
        let x = (x + attn_out.broadcast_mul(&self.lambda_1)?)?;
        let mlp_out = self.mlp.forward(&self.layernorm_after.forward(&x)?)?;
        x + mlp_out.broadcast_mul(&self.lambda_2)?
    }
}

/// Vision encoder: stack of InternS1VisionLayers at `encoder.layer.{i}`.
///
/// NOTE: Uses singular "layer" not "layers" unlike InternVL.
#[allow(dead_code)]
struct InternS1VisionEncoder {
    layers: Vec<InternS1VisionLayer>,
}

#[allow(dead_code)]
impl InternS1VisionEncoder {
    fn new(cfg: &InternS1VisionConfig, depth: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            // NOTE: singular "layer" in path (Python: encoder.layer.{i})
            layers.push(InternS1VisionLayer::new(cfg, vb.pp("layer").pp(i))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

/// InternS1 vision model (InternS1ViT).
///
/// Weight paths (under `vision_tower`):
/// - `embeddings.*`
/// - `encoder.layer.{i}.*`
/// - `layernorm.*` (optional, absent when use_mean_pooling=true)
#[allow(dead_code)]
pub(crate) struct InternS1VisionModel {
    embeddings: InternS1VisionEmbeddings,
    encoder: InternS1VisionEncoder,
    /// Final LayerNorm; absent when config.use_mean_pooling=true.
    layernorm: Option<LayerNorm>,
}

#[allow(dead_code)]
impl InternS1VisionModel {
    pub(crate) fn new(cfg: &InternS1VisionConfig, depth: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = InternS1VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = InternS1VisionEncoder::new(cfg, depth, vb.pp("encoder"))?;

        // Final norm: present unless use_mean_pooling=true
        let layernorm = if !cfg.use_mean_pooling {
            Some(layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("layernorm"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            layernorm,
        })
    }

    /// pixel_values: [B, 3, H, W] -> [B, num_patches+1, hidden]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(pixel_values)?;
        let x = self.encoder.forward(&x)?;
        if let Some(ref ln) = self.layernorm {
            ln.forward(&x)
        } else {
            Ok(x)
        }
    }
}

// ─── Pixel Shuffle ───────────────────────────────────────────────────────────

/// InternS1-specific pixel shuffle.
///
/// Spatially downsamples by `scale_factor` and increases channels by `1/scale^2`.
/// This is a different permutation pattern than InternVL's pixel_shuffle.
///
/// Input: `[B, h, w, c]` (square grid: h=w)
/// Output: `[B, h*scale, w*scale, c/scale^2]`
///
/// Python implementation (interns1.py):
///   x = x.view(n, w, int(h * scale), int(c / scale))
///   x = x.permute(0, 2, 1, 3)
///   x = x.view(n, int(h * scale), int(w * scale), int(c / (scale^2)))
///   x = x.permute(0, 2, 1, 3)
#[allow(dead_code)]
pub(crate) fn pixel_shuffle_interns1(x: &Tensor, scale_factor: f64) -> Result<Tensor> {
    let (n, h, w, c) = x.dims4()?;
    // NOTE: Python's variable naming swaps h and w (w_py=h, h_py=w) but since
    // grids are square (h=w), the result is the same either way.
    let new_w = (w as f64 * scale_factor) as usize;
    let new_h = (h as f64 * scale_factor) as usize;
    let c_half = (c as f64 / scale_factor) as usize;
    let new_c = (c as f64 / (scale_factor * scale_factor)) as usize;

    // Step 1: [n, h, w, c] -> [n, h, new_w, c/scale]
    let x = x.reshape((n, h, new_w, c_half))?;
    // Step 2: permute (0,2,1,3) -> [n, new_w, h, c/scale]
    let x = x.permute((0, 2, 1, 3))?.contiguous()?;
    // Step 3: [n, new_w, h, c/scale] -> [n, new_w, new_h, c/scale^2]
    let x = x.reshape((n, new_w, new_h, new_c))?;
    // Step 4: permute (0,2,1,3) -> [n, new_h, new_w, c/scale^2]
    x.permute((0, 2, 1, 3))?.contiguous()
}

// ─── Projector ───────────────────────────────────────────────────────────────

/// `multi_modal_projector`: LayerNorm + Linear + GELU + Linear.
///
/// Weight paths (under `multi_modal_projector`):
/// - `layer_norm.weight` / `.bias`
/// - `linear_1.weight` / `.bias`
/// - `linear_2.weight` / `.bias`
#[allow(dead_code)]
struct InternS1Projector {
    layer_norm: LayerNorm,
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
}

#[allow(dead_code)]
impl InternS1Projector {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let layer_norm = layer_norm(input_dim, 1e-5, vb.pp("layer_norm"))?;
        let linear_1 = linear(input_dim, output_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(output_dim, output_dim, vb.pp("linear_2"))?;
        Ok(Self {
            layer_norm,
            linear_1,
            linear_2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        let x = self.linear_1.forward(&x)?.gelu_erf()?;
        self.linear_2.forward(&x)
    }
}

// ─── Full Model ──────────────────────────────────────────────────────────────

/// InternS1 model for conditional generation.
///
/// Combines InternS1ViT vision encoder + pixel shuffle + MLP projector + InternLM2 LLM.
///
/// Weight paths (after HF->vLLM mapping):
/// - `vision_tower.*`          (from `model.vision_tower.*` in HF checkpoint)
/// - `multi_modal_projector.*` (from `model.multi_modal_projector.*`)
/// - `language_model.model.*`  (from `model.language_model.*`)
/// - `language_model.lm_head.*` (from `lm_head.*`)
pub struct InternS1ForConditionalGeneration {
    #[allow(dead_code)]
    vision_tower: InternS1VisionModel,
    #[allow(dead_code)]
    multi_modal_projector: InternS1Projector,
    language_model: InternLM2ForCausalLM,
    #[allow(dead_code)]
    config: InternS1Config,
    device: Device,
    dtype: DType,
}

impl InternS1ForConditionalGeneration {
    pub fn new(cfg: &InternS1Config, vb: VarBuilder) -> Result<Self> {
        let vision_config = &cfg.vision_config;
        let depth = vision_config.num_hidden_layers;

        let vision_tower = InternS1VisionModel::new(vision_config, depth, vb.pp("vision_tower"))?;

        let proj_input = cfg.projector_input_dim();
        let multi_modal_projector = InternS1Projector::new(
            proj_input,
            cfg.model_config.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;

        let language_model = InternLM2ForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(&InternS1Config::from_model_config(cfg), vb)
    }

    /// Encode pixel_values through vision tower + pixel shuffle + projector.
    ///
    /// pixel_values: [B, 3, H, W] -> [B, num_image_tokens, lm_hidden_size]
    #[allow(dead_code)]
    fn extract_feature(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Vision tower: [B, 3, H, W] -> [B, num_patches+1, vision_hidden]
        let vit_out = self.vision_tower.forward(pixel_values)?;

        // Remove CLS token: [B, num_patches+1, C] -> [B, num_patches, C]
        let (b, seq_plus_1, c) = vit_out.dims3()?;
        let vit_embeds = vit_out.narrow(1, 1, seq_plus_1 - 1)?;

        // Reshape to spatial grid: [B, h, w, C]
        let num_patches = seq_plus_1 - 1;
        let grid_size = (num_patches as f64).sqrt() as usize;
        let vit_embeds = vit_embeds.reshape((b, grid_size, grid_size, c))?;

        // Pixel shuffle (InternS1 variant): [B, h, w, C] -> [B, h', w', C']
        let shuffled = pixel_shuffle_interns1(&vit_embeds, self.config.downsample_ratio)?;

        // Flatten spatial: [B, h'*w', C']
        let (b2, nh, nw, nc) = shuffled.dims4()?;
        let flattened = shuffled.reshape((b2, nh * nw, nc))?;

        // MLP projector: [B, tokens, C'] -> [B, tokens, lm_hidden]
        self.multi_modal_projector.forward(&flattened)
    }

    /// Merge pre-projected image embeddings into text embeddings.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in emb_vec.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for InternS1ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.language_model.forward(
            input_ids,
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
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_decode_batch_with_embeddings(
            &embeddings,
            sequences,
            kv_cache_mgr,
        )
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
        let text_embeddings = self.language_model.embed_text(input_ids)?;

        let embeddings = if let Some(mm_inputs) = multimodal_inputs {
            self.merge_multimodal(&text_embeddings, mm_inputs)?
        } else {
            text_embeddings
        };

        self.language_model.forward_with_embeddings(
            &embeddings,
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
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_vision_config() -> InternS1VisionConfig {
        InternS1VisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            attention_bias: true,
            use_qk_norm: true,
            layer_norm_eps: 1e-6,
            use_rms_norm: true,
            use_absolute_position_embeddings: true,
            use_mean_pooling: false,
            layer_scale_init_value: 0.1,
            hidden_act: "gelu".to_string(),
        }
    }

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["InternS1ForConditionalGeneration".to_string()],
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
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn test_interns1_config() -> InternS1Config {
        InternS1Config {
            model_config: test_model_config(),
            vision_config: test_vision_config(),
            downsample_ratio: 0.5,
            image_token_id: 151667,
            projector_hidden_act: "gelu".to_string(),
        }
    }

    // ── Config Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = InternS1VisionConfig::default();
        assert_eq!(cfg.hidden_size, 3200);
        assert_eq!(cfg.num_hidden_layers, 45);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_patches(), 1024); // (448/14)^2
    }

    #[test]
    fn test_config_projector_dim() {
        let cfg = test_interns1_config();
        // downsample_ratio=0.5, scale=2, vision_hidden=64: 64*4=256
        assert_eq!(cfg.projector_input_dim(), 256);
    }

    #[test]
    fn test_vision_config_from_json_int_sizes() {
        let json = serde_json::json!({
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "image_size": 224,
            "patch_size": 16,
            "norm_type": "layer_norm",
            "use_qk_norm": false
        });
        let cfg = InternS1VisionConfig::from_json(&json);
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.image_size, 224);
        assert!(!cfg.use_rms_norm);
        assert!(!cfg.use_qk_norm);
    }

    #[test]
    fn test_vision_config_from_json_array_sizes() {
        let json = serde_json::json!({
            "hidden_size": 3200,
            "num_hidden_layers": 45,
            "num_attention_heads": 25,
            "image_size": [448, 448],
            "patch_size": [14, 14]
        });
        let cfg = InternS1VisionConfig::from_json(&json);
        assert_eq!(cfg.image_size, 448);
        assert_eq!(cfg.patch_size, 14);
    }

    #[test]
    fn test_config_from_model_config() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "image_size": [28, 28],
                "patch_size": [14, 14]
            }),
        );
        model_cfg
            .extra
            .insert("downsample_ratio".to_string(), serde_json::json!(0.5));
        model_cfg.extra.insert(
            "projector_hidden_act".to_string(),
            serde_json::json!("gelu"),
        );

        let cfg = InternS1Config::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.hidden_size, 128);
        assert_eq!(cfg.vision_config.image_size, 28);
        assert_eq!(cfg.downsample_ratio, 0.5);
        assert_eq!(cfg.projector_input_dim(), 512); // 128 * 4
    }

    // ── Vision Component Tests ───────────────────────────────────────────

    #[test]
    fn test_vision_embeddings() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = InternS1VisionEmbeddings::new(&cfg, vb).unwrap();

        // 28/14 = 2 patches/side, 2*2=4 patches + 1 CLS = 5
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = emb.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_embeddings_no_pos_emb() {
        let device = Device::Cpu;
        let mut cfg = test_vision_config();
        cfg.use_absolute_position_embeddings = false;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = InternS1VisionEmbeddings::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = emb.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_attention_with_qk_norm() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = InternS1VisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = attn.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_attention_no_qk_norm() {
        let device = Device::Cpu;
        let mut cfg = test_vision_config();
        cfg.use_qk_norm = false;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = InternS1VisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = attn.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_mlp() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = InternS1VisionMLP::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_layer() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let layer = InternS1VisionLayer::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = layer.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_model() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1VisionModel::new(&cfg, 2, vb).unwrap();

        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]); // 4 patches + 1 CLS
    }

    #[test]
    fn test_vision_model_no_layernorm() {
        let device = Device::Cpu;
        let mut cfg = test_vision_config();
        cfg.use_mean_pooling = true; // disables final layernorm
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1VisionModel::new(&cfg, 2, vb).unwrap();

        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    // ── Pixel Shuffle Tests ─────────────────────────────────────────────

    #[test]
    fn test_pixel_shuffle_4x4() {
        let device = Device::Cpu;
        // 4x4 grid, 64-dim -> downsample 0.5 -> 2x2 grid, 256-dim
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device).unwrap();
        let out = pixel_shuffle_interns1(&x, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 2, 2, 256]);
    }

    #[test]
    fn test_pixel_shuffle_32x32() {
        let device = Device::Cpu;
        // 32x32 grid (1024 patches from 448/14), 3200-dim -> 16x16, 12800-dim
        let x = Tensor::randn(0.0f32, 1.0, (1, 32, 32, 3200), &device).unwrap();
        let out = pixel_shuffle_interns1(&x, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 16, 16, 12800]);
    }

    #[test]
    fn test_pixel_shuffle_batched() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (3, 4, 4, 16), &device).unwrap();
        let out = pixel_shuffle_interns1(&x, 0.5).unwrap();
        assert_eq!(out.dims(), &[3, 2, 2, 64]);
    }

    // ── Projector Tests ─────────────────────────────────────────────────

    #[test]
    fn test_projector() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = InternS1Projector::new(256, 64, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 256), &device).unwrap();
        let output = proj.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    // ── Full Model Tests ────────────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_interns1_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_model_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_interns1_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.model_config.num_hidden_layers,
            num_kv_heads: cfg.model_config.num_key_value_heads,
            head_dim: cfg.model_config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = crate::kv_cache::BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_extract_feature() {
        let device = Device::Cpu;
        let cfg = test_interns1_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ForConditionalGeneration::new(&cfg, vb).unwrap();

        // 28/14=2 patches per side -> 4 patches -> after pixel_shuffle(0.5): 1 token, 256-dim -> projected to 64
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let features = model.extract_feature(&pixel_values).unwrap();
        // 2x2 grid, downsample 0.5 -> 1x1 grid -> 1 token
        assert_eq!(features.dims(), &[1, 1, cfg.model_config.hidden_size]);
    }

    #[test]
    fn test_model_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_interns1_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.model_config.num_hidden_layers,
            num_kv_heads: cfg.model_config.num_key_value_heads,
            head_dim: cfg.model_config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": [28, 28],
                "patch_size": [14, 14]
            }),
        );
        model_cfg
            .extra
            .insert("downsample_ratio".to_string(), serde_json::json!(0.5));

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(model.is_ok());
    }
}
