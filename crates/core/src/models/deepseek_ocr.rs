//! DeepSeek-OCR (v1) vision-language model for document understanding.
//!
//! # Architecture
//!
//! ```text
//! pixel_values [B, 3, H, W]
//!   → SamImageEncoderViT (SAM ViT-B)
//!       patch_embed Conv2d(3→embed_dim, kernel=patch_size, stride=patch_size)
//!       abs pos embed + 12 blocks (windowed attn for non-global, global for [2,5,8,11])
//!       neck: Conv2d(embed_dim→out_chans, 1) + LN2d + Conv2d(out_chans→out_chans, 3) + LN2d
//!       net_2: Conv2d(out_chans→512, 3, stride=2)
//!       net_3: Conv2d(512→last_conv=1024, 3, stride=2)
//!     → [B, 1024, H', W']  (H' = H/(patch*4))
//!
//!   → DeepCLIPVisionTransformer
//!       embeddings: CLS token ++ SAM output (as pre-computed patches) ++ pos_embed
//!       pre_layrnorm (typo preserved for weight compatibility)
//!       24-layer CLIP encoder (qkv_proj + out_proj + layer_norm1/2 + mlp.fc1/fc2)
//!     → [B, 1+H'*W', clip_hidden]
//!
//!   concat [vision[:, 1:], sam.flatten(2).T] → [B, H'*W', 2*clip_hidden]
//!   → MlpProjector (unfold dr² patches + 2-layer GELU MLP)
//!     → [B, (H'*W')/dr², lm_hidden]
//!
//!   Reshape with image_newline column separators + view_seperator
//!
//!   merge into language model token stream at image positions
//!
//!   → Qwen2ForCausalLM (text LM) → logits
//! ```
//!
//! # Weight paths (vllm-mapped, after hf_to_vllm_mapper removes "model." prefix)
//!
//! - `sam_model.*`           — SAM ImageEncoderViT
//! - `vision_model.*`        — DeepCLIPVisionTransformer
//! - `projector.*`           — MlpProjector
//! - `image_newline`         — learned row separator [lm_hidden]
//! - `view_seperator`        — learned image separator [lm_hidden]
//! - `model.*` / `lm_head.*` — Qwen2 language model (root level)
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/deepseek_ocr.py`
//! `reference/vllm/vllm/model_executor/models/deepencoder.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, conv2d_no_bias, embedding, layer_norm, linear, ops::softmax_last_dim, Conv2d,
    Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;

// ─── Production architecture constants ───────────────────────────────────────

const DEFAULT_SAM_DEPTH: usize = 12;
const DEFAULT_SAM_EMBED_DIM: usize = 768;
const DEFAULT_SAM_IMG_SIZE: usize = 1024;
const DEFAULT_SAM_PATCH_SIZE: usize = 16;
const DEFAULT_SAM_NUM_HEADS: usize = 12;
const DEFAULT_SAM_MLP_RATIO: f64 = 4.0;
const DEFAULT_SAM_OUT_CHANS: usize = 256;
/// SAM ViT-B final conv output — feeds into CLIP as patch embeds.
const DEFAULT_SAM_LAST_CONV: usize = 1024;
const DEFAULT_SAM_WINDOW_SIZE: usize = 14;
const SAM_GLOBAL_ATTN_INDEXES: [usize; 4] = [2, 5, 8, 11];

/// CLIP hidden_size matches SAM last_conv output.
const DEFAULT_CLIP_HIDDEN: usize = 1024;
/// CLIP reference image size (used for num_positions = 1 + (size/patch)^2).
const DEFAULT_CLIP_IMAGE_SIZE: usize = 224;
const DEFAULT_CLIP_PATCH_SIZE: usize = 14;
const DEFAULT_CLIP_LAYERS: usize = 24;
const DEFAULT_CLIP_HEADS: usize = 16;
const DEFAULT_CLIP_INTERMEDIATE: usize = 4096;
const DEFAULT_CLIP_LN_EPS: f64 = 1e-5;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Ocr1Config {
    // SAM fields
    sam_embed_dim: usize,
    sam_depth: usize,
    sam_num_heads: usize,
    sam_img_size: usize,
    sam_patch_size: usize,
    sam_mlp_ratio: f64,
    sam_out_chans: usize,
    sam_last_conv: usize,
    sam_window_size: usize,
    sam_use_rel_pos: bool,
    // CLIP fields
    clip_hidden: usize,
    clip_image_size: usize,
    clip_patch_size: usize,
    clip_layers: usize,
    clip_heads: usize,
    clip_intermediate: usize,
    clip_ln_eps: f64,
    // Projector fields
    proj_input_dim: usize,
    proj_n_embed: usize,
    proj_depth: usize,
    proj_downsample_ratio: usize,
}

impl Ocr1Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        let sc = extra.get("sam_config");
        let sam_embed_dim = sc
            .and_then(|v| v.get("embed_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_EMBED_DIM as u64) as usize;
        let sam_depth = sc
            .and_then(|v| v.get("depth"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_DEPTH as u64) as usize;
        let sam_num_heads = sc
            .and_then(|v| v.get("num_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_NUM_HEADS as u64) as usize;
        let sam_img_size = sc
            .and_then(|v| v.get("img_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_IMG_SIZE as u64) as usize;
        let sam_patch_size = sc
            .and_then(|v| v.get("patch_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_PATCH_SIZE as u64) as usize;
        let sam_out_chans = sc
            .and_then(|v| v.get("out_chans"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_OUT_CHANS as u64) as usize;
        let sam_last_conv = sc
            .and_then(|v| v.get("last_conv_output"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_LAST_CONV as u64) as usize;
        let sam_window_size = sc
            .and_then(|v| v.get("window_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_SAM_WINDOW_SIZE as u64) as usize;
        let sam_use_rel_pos = sc
            .and_then(|v| v.get("use_rel_pos"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let sam_mlp_ratio = sc
            .and_then(|v| v.get("mlp_ratio"))
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_SAM_MLP_RATIO);

        let vc = extra.get("vision_config");
        let clip_hidden = vc
            .and_then(|v| v.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CLIP_HIDDEN as u64) as usize;
        let clip_image_size = vc
            .and_then(|v| v.get("image_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CLIP_IMAGE_SIZE as u64) as usize;
        let clip_patch_size = vc
            .and_then(|v| v.get("patch_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CLIP_PATCH_SIZE as u64) as usize;
        let clip_layers = vc
            .and_then(|v| v.get("num_hidden_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CLIP_LAYERS as u64) as usize;
        let clip_heads = vc
            .and_then(|v| v.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CLIP_HEADS as u64) as usize;
        let clip_intermediate = vc
            .and_then(|v| v.get("intermediate_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_CLIP_INTERMEDIATE as u64) as usize;
        let clip_ln_eps = vc
            .and_then(|v| v.get("layer_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_CLIP_LN_EPS);

        let pc = extra.get("projector_config");
        // Default proj input = concatenated SAM + CLIP hidden
        let default_proj_input = (2 * clip_hidden) as u64;
        let proj_input_dim = pc
            .and_then(|v| v.get("input_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(default_proj_input) as usize;
        let proj_n_embed = pc
            .and_then(|v| v.get("n_embed"))
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.hidden_size as u64) as usize;
        let proj_depth = pc
            .and_then(|v| v.get("depth"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let proj_downsample_ratio = pc
            .and_then(|v| v.get("downsample_ratio"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        Self {
            sam_embed_dim,
            sam_depth,
            sam_num_heads,
            sam_img_size,
            sam_patch_size,
            sam_mlp_ratio,
            sam_out_chans,
            sam_last_conv,
            sam_window_size,
            sam_use_rel_pos,
            clip_hidden,
            clip_image_size,
            clip_patch_size,
            clip_layers,
            clip_heads,
            clip_intermediate,
            clip_ln_eps,
            proj_input_dim,
            proj_n_embed,
            proj_depth,
            proj_downsample_ratio,
        }
    }

    fn sam_spatial_size(&self) -> usize {
        self.sam_img_size / self.sam_patch_size
    }

    /// Number of patches in reference CLIP config (used for positional embedding size).
    fn clip_num_patches(&self) -> usize {
        let side = self.clip_image_size / self.clip_patch_size;
        side * side
    }

    /// CLIP positional embedding size = 1 (CLS) + num_patches.
    fn clip_num_positions(&self) -> usize {
        1 + self.clip_num_patches()
    }
}

// ─── SAM helpers ─────────────────────────────────────────────────────────────

/// Partition `[B, H, W, C]` into non-overlapping windows of `window_size`.
fn window_partition(x: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let x = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(1, 0, pad_h)?.pad_with_zeros(2, 0, pad_w)?
    } else {
        x.clone()
    };

    let h_p = h + pad_h;
    let w_p = w + pad_w;
    let num_h = h_p / window_size;
    let num_w = w_p / window_size;

    // [B, num_h, ws, num_w, ws, C] → [B*num_h*num_w, ws, ws, C]
    let x = x
        .reshape((b, num_h, window_size, num_w, window_size, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * num_h * num_w, window_size, window_size, c))?;

    Ok((x, (pad_h, pad_w)))
}

/// Reverse window partition back to `[B, H, W, C]`.
fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    pad_hw: (usize, usize),
    orig_hw: (usize, usize),
) -> Result<Tensor> {
    let (pad_h, pad_w) = pad_hw;
    let (h, w) = orig_hw;
    let h_p = h + pad_h;
    let w_p = w + pad_w;
    let num_h = h_p / window_size;
    let num_w = w_p / window_size;
    let (bw, _ws, _ws2, c) = windows.dims4()?;
    let b = bw / (num_h * num_w);

    let x = windows
        .reshape((b, num_h, num_w, window_size, window_size, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b, h_p, w_p, c))?;

    if pad_h > 0 || pad_w > 0 {
        x.narrow(1, 0, h)?.narrow(2, 0, w)
    } else {
        Ok(x)
    }
}

/// Build relative position bias for a query/key pair of size `q_size` × `k_size`.
fn get_rel_pos(q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;
    let rel_pos_resized = if rel_pos.dim(0)? != max_rel_dist {
        let rp = rel_pos.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?;
        rp.upsample_nearest2d(1, max_rel_dist)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(rel_pos.dtype())?
    } else {
        rel_pos.clone()
    };

    let q_coords = Tensor::arange(0u32, q_size as u32, rel_pos.device())?.reshape((q_size, 1))?;
    let k_coords = Tensor::arange(0u32, k_size as u32, rel_pos.device())?.reshape((1, k_size))?;

    let scale = ((k_size as f64 - 1.0) / (q_size as f64 - 1.0).max(1.0)).max(1.0);
    let q_coords_scaled = (q_coords.to_dtype(DType::F32)? * scale)?;
    let k_coords_f = k_coords.to_dtype(DType::F32)?;

    let rel_coords = (q_coords_scaled.broadcast_sub(&k_coords_f)? + (k_size as f64 - 1.0))?
        .to_dtype(DType::U32)?;

    rel_pos_resized
        .index_select(&rel_coords.flatten_all()?, 0)?
        .reshape((q_size, k_size, rel_pos_resized.dim(1)?))
}

/// Add decomposed (factorized) relative position biases to attention logits.
fn add_decomposed_rel_pos(
    attn: &Tensor,
    rel_pos_h: &Tensor,
    rel_pos_w: &Tensor,
    q_size: (usize, usize),
    k_size: (usize, usize),
) -> Result<Tensor> {
    let (q_h, q_w) = q_size;
    let (k_h, k_w) = k_size;

    let rh = get_rel_pos(q_h, k_h, rel_pos_h)?; // [q_h, k_h, head_dim]
    let rw = get_rel_pos(q_w, k_w, rel_pos_w)?; // [q_w, k_w, head_dim]

    let (bh, _n, _head_dim) = attn.dims3()?;
    // NOTE: attn here is q reshaped to [B*heads, q_h*q_w, head_dim]
    // We compute [q_h*q_w, k_h*k_w] bias by outer products of rh and rw
    let r_q = attn.reshape((bh, q_h, q_w, _head_dim))?;

    // rel_h: [B*heads, q_h, q_w, k_h] = einsum("bhwc,hkc->bhwk", r_q, rh)
    let rel_h = r_q
        .reshape((bh * q_h, q_w, _head_dim))?
        .matmul(&rh.reshape((q_h * _head_dim / q_h, k_h))?.t()?)?;
    let _ = rel_h; // not straightforward — use the simpler broadcasting form below

    // Use correct einsum-equivalent: batch matmul along the h/w axes
    let rel_h = {
        // r_q: [bh, q_h, q_w, head_dim] × rh: [q_h, k_h, head_dim]
        // Result: [bh, q_h, q_w, k_h]
        let rq_bqhd = r_q
            .permute((1, 0, 2, 3))?
            .reshape((q_h, bh * q_w, _head_dim))?;
        let rh_t = rh.permute((0, 2, 1))?; // [q_h, head_dim, k_h]
        let out = rq_bqhd.matmul(&rh_t)?; // [q_h, bh*q_w, k_h]
        out.reshape((q_h, bh, q_w, k_h))?.permute((1, 0, 2, 3))? // [bh, q_h, q_w, k_h]
    };

    let rel_w = {
        // r_q: [bh, q_h, q_w, head_dim] × rw: [q_w, k_w, head_dim]
        let rq_bqwd = r_q
            .permute((2, 0, 1, 3))?
            .reshape((q_w, bh * q_h, _head_dim))?;
        let rw_t = rw.permute((0, 2, 1))?; // [q_w, head_dim, k_w]
        let out = rq_bqwd.matmul(&rw_t)?; // [q_w, bh*q_h, k_w]
        out.reshape((q_w, bh, q_h, k_w))?.permute((1, 2, 0, 3))? // [bh, q_h, q_w, k_w]
    };

    // attn_bias: [bh, q_h*q_w, k_h*k_w]
    let rel_h_e = rel_h
        .unsqueeze(3)?
        .expand((bh, q_h, q_w, k_h, k_w))?
        .contiguous()?
        .reshape((bh, q_h * q_w, k_h * k_w))?;
    let rel_w_e = rel_w
        .unsqueeze(2)?
        .expand((bh, q_h, q_w, k_h, k_w))?
        .contiguous()?
        .reshape((bh, q_h * q_w, k_h * k_w))?;

    attn.reshape((bh, q_h * q_w, k_h * k_w))? + rel_h_e + rel_w_e
}

// ─── SAM component structs ────────────────────────────────────────────────────

struct SamLayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl SamLayerNorm2d {
    fn new(num_channels: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    /// Forward on `[B, C, H, W]` — normalise over channel dim.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let u = x.mean_keepdim(1)?;
        let s = x.broadcast_sub(&u)?.sqr()?.mean_keepdim(1)?;
        let x = x
            .broadcast_sub(&u)?
            .broadcast_div(&(s + self.eps)?.sqrt()?)?;
        let w = self.weight.reshape((1, self.weight.dim(0)?, 1, 1))?;
        let b = self.bias.reshape((1, self.bias.dim(0)?, 1, 1))?;
        x.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

struct SamPatchEmbed {
    proj: Conv2d,
}

impl SamPatchEmbed {
    fn new(in_chans: usize, embed_dim: usize, patch_size: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = conv2d(in_chans, embed_dim, patch_size, cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    /// Input `[B, C, H, W]` → output `[B, H/p, W/p, embed_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?; // [B, D, H/p, W/p]
        x.permute((0, 2, 3, 1)) // [B, H/p, W/p, D]
    }
}

struct SamMlpBlock {
    lin1: Linear,
    lin2: Linear,
}

impl SamMlpBlock {
    fn new(embed_dim: usize, mlp_dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = linear(embed_dim, mlp_dim, vb.pp("lin1"))?;
        let lin2 = linear(mlp_dim, embed_dim, vb.pp("lin2"))?;
        Ok(Self { lin1, lin2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.lin2.forward(&self.lin1.forward(x)?.gelu_erf()?)
    }
}

struct SamRelPosAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl SamRelPosAttention {
    fn new(
        dim: usize,
        num_heads: usize,
        use_rel_pos: bool,
        input_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;

        let (rel_pos_h, rel_pos_w) = if use_rel_pos {
            let rph = vb.get((2 * input_size.0 - 1, head_dim), "rel_pos_h")?;
            let rpw = vb.get((2 * input_size.1 - 1, head_dim), "rel_pos_w")?;
            (Some(rph), Some(rpw))
        } else {
            (None, None)
        };

        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        })
    }

    /// Input `[B, H, W, C]` → output `[B, H, W, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, _c) = x.dims4()?;
        let n = h * w;

        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        // → 3 × [B, heads, n, head_dim]
        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?; // [B, heads, n, head_dim]
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?;

        // Flatten batch+heads for attn computation
        let bh = b * self.num_heads;
        let q_f = q.reshape((bh, n, self.head_dim))?;
        let k_f = k.reshape((bh, n, self.head_dim))?;
        let v_f = v.reshape((bh, n, self.head_dim))?;

        let mut attn = (q_f.matmul(&k_f.t()?)? * self.scale)?;

        if self.use_rel_pos {
            if let (Some(rph), Some(rpw)) = (&self.rel_pos_h, &self.rel_pos_w) {
                // add_decomposed_rel_pos returns scaled logits [bh, n, n]
                attn = (add_decomposed_rel_pos(&q_f, rph, rpw, (h, w), (h, w))? * self.scale)?;
            }
        }

        let attn = softmax_last_dim(&attn)?;
        let x = attn
            .matmul(&v_f)?
            .reshape((b, self.num_heads, h, w, self.head_dim))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()?
            .reshape((b, h, w, self.num_heads * self.head_dim))?;

        self.proj.forward(&x)
    }
}

struct SamBlock {
    norm1: LayerNorm,
    attn: SamRelPosAttention,
    norm2: LayerNorm,
    mlp: SamMlpBlock,
    window_size: usize,
}

impl SamBlock {
    fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        window_size: usize,
        use_rel_pos: bool,
        input_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let mlp_dim = (dim as f64 * mlp_ratio) as usize;
        let mlp = SamMlpBlock::new(dim, mlp_dim, vb.pp("mlp"))?;

        let attn_size = if window_size > 0 {
            (window_size, window_size)
        } else {
            input_size
        };
        let attn = SamRelPosAttention::new(dim, num_heads, use_rel_pos, attn_size, vb.pp("attn"))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;

        let (x, pad_hw, orig_hw) = if self.window_size > 0 {
            let (_b, h, w) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
            let (win, pad_hw) = window_partition(&x, self.window_size)?;
            (win, pad_hw, (h, w))
        } else {
            let h = x.dim(1)?;
            let w = x.dim(2)?;
            (x, (0, 0), (h, w))
        };

        let x = self.attn.forward(&x)?;

        let x = if self.window_size > 0 {
            window_unpartition(&x, self.window_size, pad_hw, orig_hw)?
        } else {
            x
        };

        let x = (shortcut + x)?;
        let x = (x.clone() + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

// ─── SAM ImageEncoderViT ──────────────────────────────────────────────────────

struct SamImageEncoderViT {
    patch_embed: SamPatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<SamBlock>,
    neck0: Conv2d,
    neck1: SamLayerNorm2d,
    neck2: Conv2d,
    neck3: SamLayerNorm2d,
    net_2: Conv2d,
    net_3: Conv2d,
}

impl SamImageEncoderViT {
    fn new(cfg: &Ocr1Config, vb: VarBuilder) -> Result<Self> {
        let patch_embed = SamPatchEmbed::new(
            3,
            cfg.sam_embed_dim,
            cfg.sam_patch_size,
            vb.pp("patch_embed"),
        )?;

        let spatial = cfg.sam_spatial_size();
        let pos_embed = vb
            .get((1, spatial, spatial, cfg.sam_embed_dim), "pos_embed")
            .ok();

        let global_attn_set: Vec<usize> = SAM_GLOBAL_ATTN_INDEXES.to_vec();
        let vb_blocks = vb.pp("blocks");
        let blocks = (0..cfg.sam_depth)
            .map(|i| {
                let is_global = global_attn_set.contains(&i);
                let win_size = if is_global { 0 } else { cfg.sam_window_size };
                SamBlock::new(
                    cfg.sam_embed_dim,
                    cfg.sam_num_heads,
                    cfg.sam_mlp_ratio,
                    win_size,
                    cfg.sam_use_rel_pos,
                    (spatial, spatial),
                    vb_blocks.pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let neck_vb = vb.pp("neck");
        let neck0 = conv2d_no_bias(
            cfg.sam_embed_dim,
            cfg.sam_out_chans,
            1,
            Default::default(),
            neck_vb.pp("0"),
        )?;
        let neck1 = SamLayerNorm2d::new(cfg.sam_out_chans, neck_vb.pp("1"))?;
        let neck2_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let neck2 = conv2d_no_bias(
            cfg.sam_out_chans,
            cfg.sam_out_chans,
            3,
            neck2_cfg,
            neck_vb.pp("2"),
        )?;
        let neck3 = SamLayerNorm2d::new(cfg.sam_out_chans, neck_vb.pp("3"))?;

        let stride2_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let net_2 = conv2d_no_bias(cfg.sam_out_chans, 512, 3, stride2_cfg, vb.pp("net_2"))?;
        let net_3 = conv2d_no_bias(512, cfg.sam_last_conv, 3, stride2_cfg, vb.pp("net_3"))?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            neck0,
            neck1,
            neck2,
            neck3,
            net_2,
            net_3,
        })
    }

    /// Input `[B, 3, H, W]` → output `[B, last_conv, H/(patch*4), W/(patch*4)]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(x)?; // [B, h, w, D]

        if let Some(pos) = &self.pos_embed {
            let tgt_h = x.dim(1)?;
            let src_h = pos.dim(1)?;
            let pos_to_add = if src_h != tgt_h {
                let pos_bchw = pos.permute((0, 3, 1, 2))?.to_dtype(DType::F32)?;
                let tgt_w = x.dim(2)?;
                pos_bchw
                    .upsample_nearest2d(tgt_h, tgt_w)?
                    .permute((0, 2, 3, 1))?
                    .to_dtype(x.dtype())?
            } else {
                pos.clone()
            };
            x = x.broadcast_add(&pos_to_add)?;
        }

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let x = x.permute((0, 3, 1, 2))?.contiguous()?; // [B, D, h, w]
        let x = self.neck0.forward(&x)?;
        let x = self.neck1.forward(&x)?;
        let x = self.neck2.forward(&x)?;
        let x = self.neck3.forward(&x)?;
        let x = self.net_2.forward(&x)?;
        self.net_3.forward(&x) // [B, last_conv, h', w']
    }
}

// ─── DeepCLIP Vision Transformer ─────────────────────────────────────────────

/// CLIP-style self-attention using a single fused qkv_proj projection.
struct DeepCLIPAttention {
    qkv_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl DeepCLIPAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let qkv_proj = linear(hidden_size, hidden_size * 3, vb.pp("qkv_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;
        Ok(Self {
            qkv_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Input `[B, S, D]` → output `[B, S, D]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;

        let qkv = self
            .qkv_proj
            .forward(x)?
            .reshape((b, s, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;

        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?; // [B, heads, S, head_dim]
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?;

        let attn = (q.matmul(&k.t()?)? * self.scale)?;
        let attn = softmax_last_dim(&attn)?;

        let x = attn
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&x)
    }
}

struct DeepCLIPMlp {
    fc1: Linear,
    fc2: Linear,
}

impl DeepCLIPMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(intermediate_size, hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

struct DeepCLIPEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: DeepCLIPAttention,
    layer_norm2: LayerNorm,
    mlp: DeepCLIPMlp,
}

impl DeepCLIPEncoderLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        ln_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layer_norm1 = layer_norm(hidden_size, ln_eps, vb.pp("layer_norm1"))?;
        let layer_norm2 = layer_norm(hidden_size, ln_eps, vb.pp("layer_norm2"))?;
        let self_attn = DeepCLIPAttention::new(hidden_size, num_heads, vb.pp("self_attn"))?;
        let mlp = DeepCLIPMlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.self_attn.forward(&self.layer_norm1.forward(x)?)?;
        let x = (residual + x)?;
        let residual = x.clone();
        let x = self.mlp.forward(&self.layer_norm2.forward(&x)?)?;
        residual + x
    }
}

struct DeepCLIPEncoder {
    layers: Vec<DeepCLIPEncoderLayer>,
}

impl DeepCLIPEncoder {
    fn new(cfg: &Ocr1Config, vb: VarBuilder) -> Result<Self> {
        let layers = (0..cfg.clip_layers)
            .map(|i| {
                DeepCLIPEncoderLayer::new(
                    cfg.clip_hidden,
                    cfg.clip_heads,
                    cfg.clip_intermediate,
                    cfg.clip_ln_eps,
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

/// CLIP vision embeddings with CLS token and positional embeddings.
///
/// When `patch_embeds` (SAM output) is provided, skips the Conv2d patch embedding
/// and uses the pre-computed features directly.
struct DeepCLIPVisionEmbeddings {
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    num_positions: usize,
}

impl DeepCLIPVisionEmbeddings {
    fn new(cfg: &Ocr1Config, vb: VarBuilder) -> Result<Self> {
        let class_embedding = vb.get(cfg.clip_hidden, "class_embedding")?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.clip_patch_size,
            ..Default::default()
        };
        // No bias on CLIP patch embedding
        let patch_embedding = conv2d_no_bias(
            3,
            cfg.clip_hidden,
            cfg.clip_patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let num_positions = cfg.clip_num_positions();
        let position_embedding =
            embedding(num_positions, cfg.clip_hidden, vb.pp("position_embedding"))?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
            num_positions,
        })
    }

    /// Build embeddings from pixel_values or pre-computed patch_embeds.
    ///
    /// `patch_embeds`: SAM output `[B, clip_hidden, H', W']`
    /// Returns `[B, 1+H'*W', clip_hidden]`.
    fn forward(&self, pixel_values: &Tensor, patch_embeds: Option<&Tensor>) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        let device = pixel_values.device();

        let patch_embeds = match patch_embeds {
            Some(pe) => {
                // SAM output: [B, C, H, W] → flatten(2) → [B, C, H*W] → t → [B, H*W, C]
                pe.flatten(2, 3)?.permute((0, 2, 1))?
            }
            None => {
                // Standard CLIP path: Conv2d → [B, C, H, W] → flatten+T
                self.patch_embedding
                    .forward(pixel_values)?
                    .flatten(2, 3)?
                    .permute((0, 2, 1))?
            }
        };

        let n_patches = patch_embeds.dim(1)?;

        // CLS token: [1, hidden] → [B, 1, hidden]
        let cls = self
            .class_embedding
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, 1, self.class_embedding.dim(0)?))?
            .contiguous()?;

        let embeddings = Tensor::cat(&[&cls, &patch_embeds], 1)?; // [B, 1+n_patches, hidden]

        // Positional embeddings
        let n_tokens = 1 + n_patches;
        let pos_ids = Tensor::arange(0u32, n_tokens as u32, device)?;
        let pos_emb = if n_tokens == self.num_positions {
            self.position_embedding.forward(&pos_ids)?
        } else {
            // Interpolate: [1+n_positions, hidden] → [1+n_tokens, hidden]
            // NOTE: Uses nearest-neighbor; production uses bicubic.
            let full_pos = self.position_embedding.forward(&Tensor::arange(
                0u32,
                self.num_positions as u32,
                device,
            )?)?; // [num_positions, hidden]
            let cls_pos = full_pos.narrow(0, 0, 1)?; // [1, hidden]
            let patch_pos = full_pos.narrow(0, 1, self.num_positions - 1)?; // [src_patches, hidden]
            let src_side = ((self.num_positions - 1) as f64).sqrt() as usize;
            let tgt_side = (n_patches as f64).sqrt() as usize;
            let hidden = patch_pos.dim(1)?;
            let patch_pos_resized = patch_pos
                .reshape((1, src_side, src_side, hidden))?
                .permute((0, 3, 1, 2))?
                .contiguous()? // [1, hidden, src_side, src_side]
                .upsample_nearest2d(tgt_side, tgt_side)?
                .permute((0, 2, 3, 1))?
                .reshape((tgt_side * tgt_side, hidden))?;
            Tensor::cat(&[&cls_pos, &patch_pos_resized], 0)?
        };

        let pos_emb = pos_emb
            .unsqueeze(0)?
            .broadcast_as((b, n_tokens, pos_emb.dim(1)?))?;
        embeddings.broadcast_add(&pos_emb)
    }
}

/// DeepCLIP Vision Transformer: embeddings + pre_layrnorm + CLIP encoder.
///
/// NOTE: `pre_layrnorm` has the intentional typo from the Python source,
/// preserved here for weight-path compatibility.
struct DeepCLIPVisionTransformer {
    embeddings: DeepCLIPVisionEmbeddings,
    pre_layrnorm: LayerNorm, // NOTE: typo preserved for weight compatibility
    transformer: DeepCLIPEncoder,
}

impl DeepCLIPVisionTransformer {
    fn new(cfg: &Ocr1Config, vb: VarBuilder) -> Result<Self> {
        let embeddings = DeepCLIPVisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let pre_layrnorm = layer_norm(cfg.clip_hidden, cfg.clip_ln_eps, vb.pp("pre_layrnorm"))?;
        let transformer = DeepCLIPEncoder::new(cfg, vb.pp("transformer"))?;
        Ok(Self {
            embeddings,
            pre_layrnorm,
            transformer,
        })
    }

    /// Forward: `pixel_values [B,3,H,W]`, optional SAM `patch_embeds [B,C,S,S]`
    /// → `[B, 1+S², clip_hidden]`.
    fn forward(&self, pixel_values: &Tensor, patch_embeds: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.embeddings.forward(pixel_values, patch_embeds)?;
        let hidden = self.pre_layrnorm.forward(&hidden)?;
        self.transformer.forward(&hidden)
    }
}

// ─── MLP Projector ────────────────────────────────────────────────────────────

/// Two-layer GELU MLP with optional spatial downsampling via pixel-shuffle.
struct OcrMlpProjector {
    linear1: Linear,
    linear2: Linear,
    downsample_ratio: usize,
}

impl OcrMlpProjector {
    fn new(
        input_dim: usize,
        output_dim: usize,
        depth: usize,
        downsample_ratio: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let unfolded_dim = input_dim * downsample_ratio * downsample_ratio;
        let intermediate_dim = output_dim;

        if depth == 1 {
            let l = linear(unfolded_dim, output_dim, vb_l.pp(0))?;
            return Ok(Self {
                linear1: l.clone(),
                linear2: l,
                downsample_ratio,
            });
        }

        // Python Sequential: 0=Linear, 1=GELU, 2=Linear
        let linear1 = linear(unfolded_dim, intermediate_dim, vb_l.pp(0))?;
        let linear2 = linear(intermediate_dim, output_dim, vb_l.pp(2))?;
        Ok(Self {
            linear1,
            linear2,
            downsample_ratio,
        })
    }

    /// Input `[B, S, D]` → unfold ratio² patches → MLP → `[B, S/ratio², out_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let ratio = self.downsample_ratio;

        if ratio == 1 {
            let h = self.linear1.forward(x)?.gelu_erf()?;
            return self.linear2.forward(&h);
        }

        let h = (s as f64).sqrt() as usize;
        let w = h;
        let hr = h / ratio;
        let wr = w / ratio;

        let x = x.reshape((b, h, w, d))?;
        let x = x.reshape((b, hr, ratio, wr, ratio, d))?;
        let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
        let x = x.reshape((b, hr * wr, d * ratio * ratio))?;

        let h = self.linear1.forward(&x)?.gelu_erf()?;
        self.linear2.forward(&h)
    }
}

// ─── Main model ───────────────────────────────────────────────────────────────

/// DeepSeek-OCR (v1) vision-language model.
pub struct DeepseekOCRForCausalLM {
    sam_model: SamImageEncoderViT,
    vision_model: DeepCLIPVisionTransformer,
    projector: OcrMlpProjector,
    /// Learned row-separator token appended as a column after each row of tiles.
    image_newline: Tensor,
    /// Learned separator token appended after all image features.
    view_seperator: Tensor,
    language_model: Qwen2ForCausalLM,
    device: Device,
    dtype: DType,
}

impl DeepseekOCRForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ocr_cfg = Ocr1Config::from_model_config(cfg);
        let dtype = vb.dtype();
        let device = vb.device().clone();

        let vb_m = vb.pp("model");

        let sam_model = SamImageEncoderViT::new(&ocr_cfg, vb_m.pp("sam_model"))?;
        let vision_model = DeepCLIPVisionTransformer::new(&ocr_cfg, vb_m.pp("vision_model"))?;

        let projector = OcrMlpProjector::new(
            ocr_cfg.proj_input_dim,
            ocr_cfg.proj_n_embed,
            ocr_cfg.proj_depth,
            ocr_cfg.proj_downsample_ratio,
            vb_m.pp("projector"),
        )?;

        let image_newline = vb_m.get(ocr_cfg.proj_n_embed, "image_newline")?;
        let view_seperator = vb_m.get(ocr_cfg.proj_n_embed, "view_seperator")?;

        // Language model: Qwen2ForCausalLM (weights at model.*, lm_head.*)
        let language_model = Qwen2ForCausalLM::new(cfg, vb)?;

        Ok(Self {
            sam_model,
            vision_model,
            projector,
            image_newline,
            view_seperator,
            language_model,
            device,
            dtype,
        })
    }

    /// Encode a single image (global view) to projected features with newline tokens.
    ///
    /// `pixel_values`: `[B, 3, H, W]`
    /// Returns: `[B * (n_proj*(side+1)/side + 1), lm_hidden]`
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let sam_out = self.sam_model.forward(pixel_values)?; // [B, clip_hidden, s, s]
        let vision_out = self.vision_model.forward(pixel_values, Some(&sam_out))?; // [B, 1+s², hidden]

        // Concatenate: vision[:, 1:] (exclude CLS) ++ sam.flatten.T
        let vision_patches = vision_out.narrow(1, 1, vision_out.dim(1)? - 1)?; // [B, s², hidden]
        let sam_flat = sam_out.flatten(2, 3)?.permute((0, 2, 1))?; // [B, s², hidden]
        let features = Tensor::cat(&[&vision_patches, &sam_flat], 2)?; // [B, s², 2*hidden]

        let projected = self.projector.forward(&features)?; // [B, p, lm_hidden]

        let (b, p, d) = projected.dims3()?;
        let side = (p as f64).sqrt() as usize;

        // Add image_newline as a column separator: reshape to [side, side, d],
        // append newline column → [side, side+1, d], flatten.
        let mut per_image = Vec::with_capacity(b);
        for bi in 0..b {
            let feat = projected.narrow(0, bi, 1)?.squeeze(0)?; // [p, d]
            let feat = feat.reshape((side, side, d))?;

            let newline_col = self
                .image_newline
                .reshape((1, 1, d))?
                .broadcast_as((side, 1, d))?
                .contiguous()?;
            let with_nl = Tensor::cat(&[&feat, &newline_col], 1)?; // [side, side+1, d]
            let flat = with_nl.reshape((side * (side + 1), d))?; // [(side*(side+1)), d]

            let sep = self.view_seperator.reshape((1, d))?;
            per_image.push(Tensor::cat(&[&flat, &sep], 0)?);
        }

        // Stack across batch and flatten
        let all_imgs = Tensor::cat(&per_image, 0)?; // [B*(tokens_per_img), d]
        Ok(all_imgs)
    }

    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_, seq_len, _) = text_embeddings.dims3()?;
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

impl crate::engine::ModelForward for DeepseekOCRForCausalLM {
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
        let embeddings = if let Some(mm) = multimodal_inputs {
            self.merge_multimodal(&text_embeddings, mm)?
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use serde_json::json;

    /// Tiny test config.
    ///
    /// SAM: img_size=32, patch_size=4 → spatial=8 → after 2×stride-2 convs → 2×2
    /// CLIP: image_size=16, patch_size=8 → (16/8)^2=4 patches → num_positions=5
    /// SAM last_conv = clip_hidden = 16 (they must match for concat to work)
    /// Projector: input_dim=32 (16+16), n_embed=8 (LM hidden), dr=1
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "sam_config".to_string(),
            json!({
                "embed_dim": 8,
                "depth": 2,
                "num_heads": 2,
                "img_size": 32,
                "patch_size": 4,
                "mlp_ratio": 2.0,
                "out_chans": 4,
                "last_conv_output": 16,
                "window_size": 0,
                "use_rel_pos": false
            }),
        );
        // CLIP: hidden_size=16 must equal SAM last_conv; image_size/patch_size → 4 patches
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 16,
                "image_size": 16,
                "patch_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "layer_norm_eps": 1e-5
            }),
        );
        // Projector: input_dim=32 (16+16 concat), n_embed=8, dr=1
        extra.insert(
            "projector_config".to_string(),
            json!({
                "input_dim": 32,
                "n_embed": 8,
                "depth": 2,
                "downsample_ratio": 1
            }),
        );
        ModelConfig {
            architectures: vec!["DeepseekOCRForCausalLM".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 16,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(true), // Qwen2 uses attention bias
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_deepseek_ocr_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCRForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "DeepseekOCRForCausalLM construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_deepseek_ocr_sam_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCRForCausalLM::new(&cfg, vb).unwrap();
        let ocr_cfg = Ocr1Config::from_model_config(&cfg);

        // SAM: img_size=32, patch_size=4 → 2 stride-2 convs → spatial 8/2/2=2
        let pixel_values =
            Tensor::zeros((1usize, 3usize, 32usize, 32usize), DType::F32, &device).unwrap();
        let sam_out = model.sam_model.forward(&pixel_values);
        assert!(sam_out.is_ok(), "SAM forward failed: {:?}", sam_out.err());
        let out = sam_out.unwrap();
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(1).unwrap(), ocr_cfg.sam_last_conv);
        // spatial: 32/4=8 → /2 → /2 = 2
        assert_eq!(out.dim(2).unwrap(), 2);
        assert_eq!(out.dim(3).unwrap(), 2);
    }

    #[test]
    fn test_deepseek_ocr_clip_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCRForCausalLM::new(&cfg, vb).unwrap();
        let ocr_cfg = Ocr1Config::from_model_config(&cfg);

        // CLIP: image_size=16, patch_size=8 → 4 patches → 5 tokens
        let pixel_values =
            Tensor::zeros((1usize, 3usize, 16usize, 16usize), DType::F32, &device).unwrap();
        // Use SAM-shaped pre-computed patch embeds: [B, clip_hidden, 2, 2]
        let sam_embeds = Tensor::zeros(
            (1usize, ocr_cfg.clip_hidden, 2usize, 2usize),
            DType::F32,
            &device,
        )
        .unwrap();
        let clip_out = model.vision_model.forward(&pixel_values, Some(&sam_embeds));
        assert!(
            clip_out.is_ok(),
            "CLIP forward failed: {:?}",
            clip_out.err()
        );
        let out = clip_out.unwrap();
        // [1, 1+4, 16] = [1, 5, 16]
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(1).unwrap(), 1 + 4); // CLS + 4 patches from 2×2 SAM spatial
        assert_eq!(out.dim(2).unwrap(), ocr_cfg.clip_hidden);
    }

    #[test]
    fn test_deepseek_ocr_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCRForCausalLM::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        let seq_len = 4usize;
        let mut bt = crate::kv_cache::BlockTable::new(16);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let result =
            model.forward_multimodal(&input_ids, None, 0, &mut kv_cache, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_deepseek_ocr_projector_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let ocr_cfg = Ocr1Config::from_model_config(&cfg);

        let proj = OcrMlpProjector::new(
            ocr_cfg.proj_input_dim,
            ocr_cfg.proj_n_embed,
            ocr_cfg.proj_depth,
            ocr_cfg.proj_downsample_ratio,
            vb.pp("model").pp("projector"),
        )
        .unwrap();

        // Input: [1, 4, 32] (4 spatial tokens, 32-dim = 2×16 concat)
        let x = Tensor::zeros(
            (1usize, 4usize, ocr_cfg.proj_input_dim),
            DType::F32,
            &device,
        )
        .unwrap();
        let out = proj.forward(&x);
        assert!(out.is_ok(), "projector forward failed: {:?}", out.err());
        let o = out.unwrap();
        assert_eq!(o.dim(0).unwrap(), 1);
        assert_eq!(o.dim(1).unwrap(), 4); // dr=1, no spatial reduction
        assert_eq!(o.dim(2).unwrap(), ocr_cfg.proj_n_embed);
    }
}
