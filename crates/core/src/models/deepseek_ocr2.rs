//! DeepSeek-OCR2 vision-language model for document understanding.
//!
//! # Architecture
//!
//! ```text
//! pixel_values [B, 3, H, W]
//!   → SamImageEncoderViT
//!       patch_embed Conv2d(3→embed_dim, kernel=patch_size, stride=patch_size)
//!       abs pos embed + 12 blocks (windowed attn for non-global, global for [2,5,8,11])
//!       neck: Conv2d(embed_dim→256, 1) + LN2d + Conv2d(256→256, 3) + LN2d
//!       net_2: Conv2d(256→512, 3, stride=2)
//!       net_3: Conv2d(512→896, 3, stride=2)
//!     → [B, 896, H/64, W/64]
//!
//!   → Ocr2Qwen2Encoder (24 Qwen2 decoder layers as encoder)
//!       flatten(2).T → [B, n_patches, 896]
//!       cat with learnable queries [n_patches, 896]
//!       custom dual-mode attention mask (image=non-causal, query=causal)
//!       return only query outputs [B, n_patches, 896]
//!     → [B, n_patches, 896]
//!
//!   → Ocr2MlpProjector  (2-layer downsample MLP)
//!     → [B, n_patches/dr², lm_hidden]
//!
//!   + view_seperator token
//!
//!   merge into language model token stream at image positions
//!
//!   → Qwen2ForCausalLM (text LM) → logits
//! ```
//!
//! # Weight paths (HF checkpoint, pre-mapper)
//!
//! - `model.sam_model.*`          — SAM ImageEncoderViT
//! - `model.qwen2_model.*`        — Qwen2Decoder2Encoder
//! - `model.projector.*`          — MlpProjector
//! - `model.view_seperator`       — learned separator token [n_embed]
//! - `model.*` / `lm_head.*`      — Qwen2 language model (root level)
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/deepseek_ocr2.py`
//! `reference/vllm/vllm/model_executor/models/deepencoder.py`
//! `reference/vllm/vllm/model_executor/models/deepencoder2.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, conv2d_no_bias, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, Conv2d,
    Conv2dConfig, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;

// ─── Production architecture constants ───────────────────────────────────────

/// Default SAM ViT depth (number of transformer blocks).
const DEFAULT_SAM_DEPTH: usize = 12;
/// Default SAM ViT embedding dimension.
const DEFAULT_SAM_EMBED_DIM: usize = 768;
/// Default SAM ViT image size (pixels).
const DEFAULT_SAM_IMG_SIZE: usize = 1024;
/// Default SAM ViT patch size (pixels).
const DEFAULT_SAM_PATCH_SIZE: usize = 16;
/// Default SAM ViT number of attention heads.
const DEFAULT_SAM_NUM_HEADS: usize = 12;
/// Default SAM ViT MLP hidden ratio.
const DEFAULT_SAM_MLP_RATIO: f64 = 4.0;
/// Default SAM ViT neck output channels.
const DEFAULT_SAM_OUT_CHANS: usize = 256;
/// Default final conv2d output channels (fed into Qwen2 encoder).
const DEFAULT_SAM_LAST_CONV: usize = 896;
/// Default window size for windowed attention blocks.
const DEFAULT_SAM_WINDOW_SIZE: usize = 14;
/// Blocks using global (non-windowed) attention.
const SAM_GLOBAL_ATTN_INDEXES: [usize; 4] = [2, 5, 8, 11];

/// Default Qwen2 encoder depth.
const DEFAULT_QWEN2_LAYERS: usize = 24;
/// Default Qwen2 encoder hidden dimension.
const DEFAULT_QWEN2_HIDDEN: usize = 896;
/// Default Qwen2 encoder attention heads.
const DEFAULT_QWEN2_HEADS: usize = 14;
/// Default Qwen2 encoder KV heads.
const DEFAULT_QWEN2_KV_HEADS: usize = 2;
/// Default Qwen2 encoder MLP intermediate dimension.
const DEFAULT_QWEN2_INTERMEDIATE: usize = 4864;
/// Default Qwen2 encoder RoPE theta.
const DEFAULT_QWEN2_ROPE_THETA: f64 = 1_000_000.0;
/// Default Qwen2 encoder RMS norm epsilon.
const DEFAULT_QWEN2_RMS_EPS: f64 = 1e-6;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Full OCR2 model configuration (read from model cfg.extra).
#[derive(Debug, Clone)]
struct Ocr2Config {
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
    // Qwen2 encoder fields
    qwen2_layers: usize,
    qwen2_hidden: usize,
    qwen2_heads: usize,
    qwen2_kv_heads: usize,
    qwen2_intermediate: usize,
    // Projector fields
    proj_input_dim: usize,
    proj_n_embed: usize,
    proj_depth: usize,
    proj_downsample_ratio: usize,
}

impl Ocr2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        // SAM config from extra["sam_config"] or defaults
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

        // Qwen2 encoder config
        let qc = extra.get("qwen2_encoder_config");
        let qwen2_layers = qc
            .and_then(|v| v.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_QWEN2_LAYERS as u64) as usize;
        let qwen2_hidden = qc
            .and_then(|v| v.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_QWEN2_HIDDEN as u64) as usize;
        let qwen2_heads = qc
            .and_then(|v| v.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_QWEN2_HEADS as u64) as usize;
        let qwen2_kv_heads = qc
            .and_then(|v| v.get("num_key_value_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_QWEN2_KV_HEADS as u64) as usize;
        let qwen2_intermediate = qc
            .and_then(|v| v.get("intermediate_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_QWEN2_INTERMEDIATE as u64) as usize;

        // Projector config
        let pc = extra.get("projector_config");
        let proj_input_dim = pc
            .and_then(|v| v.get("input_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(sam_last_conv as u64) as usize;
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
            .unwrap_or(4) as usize;

        let sam_mlp_ratio = sc
            .and_then(|v| v.get("mlp_ratio"))
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_SAM_MLP_RATIO);

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
            qwen2_layers,
            qwen2_hidden,
            qwen2_heads,
            qwen2_kv_heads,
            qwen2_intermediate,
            proj_input_dim,
            proj_n_embed,
            proj_depth,
            proj_downsample_ratio,
        }
    }

    /// Number of patches per spatial dimension after patch embedding.
    fn sam_spatial_size(&self) -> usize {
        self.sam_img_size / self.sam_patch_size
    }

    fn qwen2_head_dim(&self) -> usize {
        self.qwen2_hidden / self.qwen2_heads
    }
}

// ─── SAM helpers: window partition / unpartition ─────────────────────────────

/// Partition `[B, H, W, C]` into non-overlapping windows, padding if needed.
/// Returns `([B*num_windows, win, win, C], (padded_H, padded_W))`.
fn window_partition(x: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let pad_h = if h % window_size != 0 {
        window_size - h % window_size
    } else {
        0
    };
    let pad_w = if w % window_size != 0 {
        window_size - w % window_size
    } else {
        0
    };
    let x = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(1, 0, pad_h)?.pad_with_zeros(2, 0, pad_w)?
    } else {
        x.clone()
    };
    let hp = h + pad_h;
    let wp = w + pad_w;
    let hn = hp / window_size;
    let wn = wp / window_size;
    // [B, Hp, Wp, C] → [B, hn, win, wn, win, C] → [B, hn, wn, win, win, C] → [B*hn*wn, win, win, C]
    let x = x.reshape((b, hn, window_size, wn, window_size, c))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
    let windows = x.reshape((b * hn * wn, window_size, window_size, c))?;
    Ok((windows, (hp, wp)))
}

/// Reverse window partition; remove padding to restore `[B, H, W, C]`.
fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let c = windows.dim(3)?;
    let n_windows = windows.dim(0)?;
    let hn = hp / window_size;
    let wn = wp / window_size;
    let b = n_windows / (hn * wn);
    // [B*hn*wn, win, win, C] → [B, Hp, Wp, C]
    let x = windows.reshape((b, hn, wn, window_size, window_size, c))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
    let x = x.reshape((b, hp, wp, c))?;
    if hp > h || wp > w {
        x.narrow(1, 0, h)?.narrow(2, 0, w)
    } else {
        Ok(x)
    }
}

// ─── SAM relative position embedding helpers ─────────────────────────────────

/// Compute relative positional embedding table for a query–key pair.
///
/// `rel_pos`: `[2*max(q_size,k_size)-1, head_dim]`
/// Returns: `[q_size, k_size, head_dim]`
fn get_rel_pos(q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;
    let head_dim = rel_pos.dim(1)?;
    let device = rel_pos.device();
    let rel_pos_len = rel_pos.dim(0)?;

    let q_scale = (k_size as f64 / q_size as f64).max(1.0);
    let k_scale = (q_size as f64 / k_size as f64).max(1.0);
    let offset = (k_size as f64 - 1.0) * k_scale;

    let mut indices = vec![0u32; q_size * k_size];
    for i in 0..q_size {
        for j in 0..k_size {
            let rel = (i as f64 * q_scale - j as f64 * k_scale + offset).round() as i64;
            let idx = rel.clamp(0, (max_rel_dist.min(rel_pos_len) - 1) as i64) as u32;
            indices[i * k_size + j] = idx;
        }
    }

    let idx_t = Tensor::from_vec(indices, q_size * k_size, device)?;
    let selected = rel_pos.index_select(&idx_t, 0)?; // [q_size*k_size, head_dim]
    selected.reshape((q_size, k_size, head_dim))
}

/// Compute decomposed relative position biases (height and width separately).
///
/// `q`: `[B*num_heads, H*W, head_dim]`  (B here is real batch × num_heads)
///
/// Returns: `(rel_h [B*H, H*W, k_h, 1], rel_w [B*H, H*W, 1, k_w])`
fn add_decomposed_rel_pos(
    q: &Tensor,
    rel_pos_h: &Tensor,
    rel_pos_w: &Tensor,
    q_size: (usize, usize),
    k_size: (usize, usize),
) -> Result<(Tensor, Tensor)> {
    let (q_h, q_w) = q_size;
    let (k_h, k_w) = k_size;
    let b_heads = q.dim(0)?;
    let head_dim = q.dim(2)?;

    let rh = get_rel_pos(q_h, k_h, rel_pos_h)?; // [q_h, k_h, head_dim]
    let rw = get_rel_pos(q_w, k_w, rel_pos_w)?; // [q_w, k_w, head_dim]

    // r_q: [B_heads, q_h, q_w, head_dim]
    let r_q = q.reshape((b_heads, q_h, q_w, head_dim))?;

    // rel_h: einsum("bhwc,hkc->bhwk", r_q, rh) = [B_heads, q_h, q_w, k_h]
    let r_q_flat = r_q.reshape((b_heads * q_h, q_w, head_dim))?;
    let rh_t = rh.transpose(1, 2)?.contiguous()?; // [q_h, head_dim, k_h]
    let rh_t_exp = rh_t
        .unsqueeze(0)?
        .broadcast_as((b_heads, q_h, head_dim, k_h))?;
    let rh_t_flat = rh_t_exp.reshape((b_heads * q_h, head_dim, k_h))?;
    let rel_h = r_q_flat
        .matmul(&rh_t_flat)? // [B*q_h, q_w, k_h]
        .reshape((b_heads, q_h, q_w, k_h))?
        .reshape((b_heads, q_h * q_w, k_h, 1))?;

    // rel_w: einsum("bhwc,wkc->bhwk", r_q, rw) = [B_heads, q_h, q_w, k_w]
    let r_q_perm = r_q.permute((0, 2, 1, 3))?.contiguous()?; // [B_heads, q_w, q_h, head_dim]
    let r_q_flat2 = r_q_perm.reshape((b_heads * q_w, q_h, head_dim))?;
    let rw_t = rw.transpose(1, 2)?.contiguous()?; // [q_w, head_dim, k_w]
    let rw_t_exp = rw_t
        .unsqueeze(0)?
        .broadcast_as((b_heads, q_w, head_dim, k_w))?;
    let rw_t_flat = rw_t_exp.reshape((b_heads * q_w, head_dim, k_w))?;
    let rel_w = r_q_flat2
        .matmul(&rw_t_flat)? // [B*q_w, q_h, k_w]
        .reshape((b_heads, q_w, q_h, k_w))?
        .permute((0, 2, 1, 3))? // [B_heads, q_h, q_w, k_w]
        .reshape((b_heads, q_h * q_w, 1, k_w))?;

    Ok((rel_h, rel_w))
}

// ─── SAM LayerNorm2d ──────────────────────────────────────────────────────────

/// Channel-wise LayerNorm for `[B, C, H, W]` tensors (normalises over C dim).
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, H, W] — normalise over C dim (dim=1)
        let (_, c, _, _) = x.dims4()?;
        let mean = x.mean_keepdim(1)?;
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(1)?;
        let x_norm = diff.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w = self.weight.reshape((1, c, 1, 1))?;
        let b = self.bias.reshape((1, c, 1, 1))?;
        x_norm.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

// ─── SAM PatchEmbed ───────────────────────────────────────────────────────────

/// Conv2d patch embedding: `[B, C, H, W]` → `[B, H/p, W/p, embed_dim]`.
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, C, H, W] → Conv → [B, D, H/p, W/p] → permute → [B, H/p, W/p, D]
        let x = self.proj.forward(x)?;
        x.permute((0, 2, 3, 1))
    }
}

// ─── SAM MLP block ────────────────────────────────────────────────────────────

struct SamMlpBlock {
    lin1: Linear,
    lin2: Linear,
}

impl SamMlpBlock {
    fn new(embed_dim: usize, mlp_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: linear(embed_dim, mlp_dim, vb.pp("lin1"))?,
            lin2: linear(mlp_dim, embed_dim, vb.pp("lin2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // GELU activation (standard, not QuickGELU)
        let h = self.lin1.forward(x)?.gelu_erf()?;
        self.lin2.forward(&h)
    }
}

// ─── SAM RelPos Attention ─────────────────────────────────────────────────────

struct SamRelPosAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    // Optional decomposed relative position embeddings
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl SamRelPosAttention {
    fn new(
        embed_dim: usize,
        num_heads: usize,
        use_rel_pos: bool,
        input_size: Option<(usize, usize)>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let qkv = linear(embed_dim, embed_dim * 3, vb.pp("qkv"))?;
        let proj = linear(embed_dim, embed_dim, vb.pp("proj"))?;

        let (rel_pos_h, rel_pos_w) = if use_rel_pos {
            let sz = input_size.unwrap_or((14, 14));
            let rph = vb.get((2 * sz.0 - 1, head_dim), "rel_pos_h")?;
            let rpw = vb.get((2 * sz.1 - 1, head_dim), "rel_pos_w")?;
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
            rel_pos_h,
            rel_pos_w,
        })
    }

    /// Forward on `[B, H, W, C]` — returns `[B, H, W, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let seq = h * w;
        let d = self.num_heads * self.head_dim;

        let x_flat = x.reshape((b, seq, c))?;
        let qkv = self.qkv.forward(&x_flat)?; // [B, S, 3*D]
        let q_raw = qkv.narrow(2, 0, d)?;
        let k_raw = qkv.narrow(2, d, d)?;
        let v_raw = qkv.narrow(2, 2 * d, d)?;

        // Reshape to [B, num_heads, S, head_dim]
        let q = q_raw
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = k_raw
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = v_raw
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        // Scaled dot-product: [B, H_num, S, S]
        let attn = q
            .affine(self.scale, 0.0)?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?;

        // Add decomposed relative position bias if enabled
        let attn = if let (Some(rph), Some(rpw)) = (&self.rel_pos_h, &self.rel_pos_w) {
            // Use actual spatial size (may differ from stored if window resized)
            let q_flat_for_rp = q.reshape((b * self.num_heads, seq, self.head_dim))?;
            let (rel_h, rel_w) = add_decomposed_rel_pos(&q_flat_for_rp, rph, rpw, (h, w), (h, w))?;
            // rel_h: [B*H_num, S, h, 1], rel_w: [B*H_num, S, 1, w]
            let bias = rel_h
                .broadcast_add(&rel_w)?
                .reshape((b * self.num_heads, seq, h * w))?
                .reshape((b, self.num_heads, seq, h * w))?;
            (attn + bias)?
        } else {
            attn
        };

        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [B, H_num, S, head_dim]

        // Reshape back to [B, H, W, D]
        let out = out
            .reshape((b, self.num_heads, h, w, self.head_dim))?
            .permute((0, 2, 3, 1, 4))?
            .reshape((b, h, w, d))?;
        // Apply output projection (operates on last dim)
        let out_flat = out.reshape((b, seq, d))?;
        self.proj.forward(&out_flat)?.reshape((b, h, w, d))
    }
}

// ─── SAM Block ────────────────────────────────────────────────────────────────

struct SamBlock {
    norm1: LayerNorm,
    attn: SamRelPosAttention,
    norm2: LayerNorm,
    mlp: SamMlpBlock,
    /// 0 = global attention; >0 = window attention of this size.
    window_size: usize,
}

impl SamBlock {
    fn new(
        embed_dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        window_size: usize,
        use_rel_pos: bool,
        input_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(embed_dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(embed_dim, 1e-6, vb.pp("norm2"))?;
        let mlp_dim = (embed_dim as f64 * mlp_ratio) as usize;
        let mlp = SamMlpBlock::new(embed_dim, mlp_dim, vb.pp("mlp"))?;

        // Relative position embedding size depends on window_size vs global
        let rel_input_size = if window_size == 0 {
            input_size
        } else {
            (window_size, window_size)
        };
        let attn = SamRelPosAttention::new(
            embed_dim,
            num_heads,
            use_rel_pos,
            Some(rel_input_size),
            vb.pp("attn"),
        )?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, H, W, C]
        let (_, h, w, _) = x.dims4()?;
        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;

        if self.window_size > 0 {
            // Windowed attention
            let (windows, pad_hw) = window_partition(&x, self.window_size)?;
            let windows = self.attn.forward(&windows)?;
            let x = window_unpartition(&windows, self.window_size, pad_hw, (h, w))?;
            let x = (shortcut + x)?;
            let x = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
            Ok(x)
        } else {
            // Global attention
            let x = self.attn.forward(&x)?;
            let x = (shortcut + x)?;
            let x = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
            Ok(x)
        }
    }
}

// ─── SAM ImageEncoderViT ──────────────────────────────────────────────────────

/// SAM-style ViT image encoder with windowed+global attention.
///
/// Output: `[B, last_conv_output, H/64, W/64]` for a 1024px image.
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
    fn new(cfg: &Ocr2Config, vb: VarBuilder) -> Result<Self> {
        let patch_embed = SamPatchEmbed::new(
            3,
            cfg.sam_embed_dim,
            cfg.sam_patch_size,
            vb.pp("patch_embed"),
        )?;

        // Absolute pos embed: [1, spatial, spatial, embed_dim]
        let spatial = cfg.sam_spatial_size();
        let pos_embed = vb
            .get((1, spatial, spatial, cfg.sam_embed_dim), "pos_embed")
            .ok();

        // Transformer blocks
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

        // Neck: Conv2d(embed→out_chans, 1, no bias) + LN2d + Conv2d(out_chans→out_chans, 3, no bias) + LN2d
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

        // net_2: Conv2d(out_chans→512, 3, stride=2, pad=1, no bias)
        let net2_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let net_2 = conv2d_no_bias(cfg.sam_out_chans, 512, 3, net2_cfg, vb.pp("net_2"))?;

        // net_3: Conv2d(512→last_conv, 3, stride=2, pad=1, no bias)
        let net_3 = conv2d_no_bias(512, cfg.sam_last_conv, 3, net2_cfg, vb.pp("net_3"))?;

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

    /// Forward: `[B, 3, H, W]` → `[B, last_conv_output, H/64, W/64]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(x)?; // [B, H/p, W/p, D]

        // Add absolute position embedding (with optional bilinear resize if needed)
        if let Some(pos) = &self.pos_embed {
            // pos: [1, src_h, src_w, D]
            let tgt_h = x.dim(1)?;
            let src_h = pos.dim(1)?;
            let pos_for_add = if src_h != tgt_h {
                // Resize pos embed: permute to [1, D, src_h, src_w], interpolate, permute back
                // Simple nearest-neighbor interpolation for non-production use
                // NOTE: Production would use bicubic interpolation
                let pos_bchw = pos.permute((0, 3, 1, 2))?.to_dtype(DType::F32)?; // [1, D, src_h, src_w]
                let tgt_w = x.dim(2)?;
                // Use candle's upsample_nearest2d for simplicity
                let pos_resized = pos_bchw.upsample_nearest2d(tgt_h, tgt_w)?;
                pos_resized.permute((0, 2, 3, 1))?.to_dtype(x.dtype())?
            } else {
                pos.clone()
            };
            let _ = src_h; // suppress unused
            x = x.broadcast_add(&pos_for_add)?;
        }

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Neck: permute to [B, D, H, W] for convolutions
        let x = x.permute((0, 3, 1, 2))?.contiguous()?; // [B, D, H, W]
        let x = self.neck0.forward(&x)?;
        let x = self.neck1.forward(&x)?;
        let x = self.neck2.forward(&x)?;
        let x = self.neck3.forward(&x)?;

        // Final conv downsampling
        let x = self.net_2.forward(&x)?;
        self.net_3.forward(&x)
    }
}

// ─── Qwen2 Encoder (Qwen2Decoder2Encoder) ────────────────────────────────────

/// Qwen2 self-attention for encoder (no paged KV cache, full-sequence).
struct Ocr2Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Ocr2Qwen2Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Qwen2 uses bias=True on QKV projections
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let rotary_emb = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, dtype, device)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Full self-attention (no KV cache).
    ///
    /// `x`: `[B, S, D]`
    /// `mask`: optional additive attention mask `[B, 1, S, S]` (0=attend, -inf=mask)
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B, H, S, D_h]
        let k = k
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B, KV_H, S, D_h]
        let v = v
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE (positions 0..s)
        let (q, k) = self.rotary_emb.apply(&q, &k, 0)?;

        // Expand KV heads to match Q heads (GQA)
        let ratio = self.num_heads / self.num_kv_heads;
        let k = if ratio > 1 {
            k.unsqueeze(2)?
                .broadcast_as((b, self.num_kv_heads, ratio, s, self.head_dim))?
                .reshape((b, self.num_heads, s, self.head_dim))?
        } else {
            k
        };
        let v = if ratio > 1 {
            v.unsqueeze(2)?
                .broadcast_as((b, self.num_kv_heads, ratio, s, self.head_dim))?
                .reshape((b, self.num_heads, s, self.head_dim))?
        } else {
            v
        };

        // Scaled dot-product attention
        let attn = q
            .affine(self.scale, 0.0)?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?; // [B, H, S, S]

        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };

        let attn = softmax_last_dim(&attn)?;
        let out = attn
            .matmul(&v.contiguous()?)? // [B, H, S, D_h]
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&out)
    }
}

/// SwiGLU MLP for the Qwen2 encoder (no KV cache context).
struct Ocr2Qwen2Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Ocr2Qwen2Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let h = (gate * up)?;
        self.down_proj.forward(&h)
    }
}

/// Single Qwen2 decoder layer operating as an encoder (no KV cache).
struct Ocr2Qwen2Layer {
    input_layernorm: RmsNorm,
    self_attn: Ocr2Qwen2Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Ocr2Qwen2Mlp,
}

impl Ocr2Qwen2Layer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rope_theta: f64,
        rms_eps: f64,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_layernorm = rms_norm(hidden_size, rms_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(hidden_size, rms_eps, vb.pp("post_attention_layernorm"))?;
        let self_attn = Ocr2Qwen2Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            max_seq_len,
            dtype,
            device,
            vb.pp("self_attn"),
        )?;
        let mlp = Ocr2Qwen2Mlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self
            .self_attn
            .forward(&self.input_layernorm.forward(x)?, mask)?;
        let x = (x + h)?;
        let h = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?;
        x + h
    }
}

/// Qwen2-based decoder used as a visual feature encoder.
///
/// Input: `[B, encoder_hidden, H, W]` (SAM output)
/// Process:
///   1. Flatten spatial: `[B, H*W, encoder_hidden]`
///   2. Concatenate with learnable query tokens: `[B, 2*n_img, encoder_hidden]`
///   3. Run through Qwen2 layers with dual-mode attention mask
///   4. Return only query outputs: `[B, n_img, encoder_hidden]`
struct Ocr2Qwen2Encoder {
    layers: Vec<Ocr2Qwen2Layer>,
    norm: RmsNorm,
    /// Learnable queries for 12×12=144 image patches (768px → net3 = 12×12)
    query_144: Tensor,
    /// Learnable queries for 16×16=256 image patches (1024px → net3 = 16×16)
    query_256: Tensor,
    hidden_size: usize,
    device: Device,
}

impl Ocr2Qwen2Encoder {
    fn new(cfg: &Ocr2Config, dtype: DType, device: &Device, vb: VarBuilder) -> Result<Self> {
        let qc = cfg;
        let head_dim = qc.qwen2_head_dim();

        // The internal Qwen2 model in Python is at `qwen2_model.model.layers.*`
        let model_vb = vb.pp("model");
        let vb_layers = model_vb.pp("layers");
        let layers = (0..qc.qwen2_layers)
            .map(|i| {
                Ocr2Qwen2Layer::new(
                    qc.qwen2_hidden,
                    qc.qwen2_heads,
                    qc.qwen2_kv_heads,
                    head_dim,
                    qc.qwen2_intermediate,
                    DEFAULT_QWEN2_ROPE_THETA,
                    DEFAULT_QWEN2_RMS_EPS,
                    // Max seq len = 2 * largest query size (256) for safety
                    1024,
                    dtype,
                    device,
                    vb_layers.pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = rms_norm(qc.qwen2_hidden, DEFAULT_QWEN2_RMS_EPS, model_vb.pp("norm"))?;

        // Learnable query embeddings (nn.Embedding in Python → weight tensor)
        let query_144 = vb.pp("query_768").get((144, qc.qwen2_hidden), "weight")?;
        let query_256 = vb.pp("query_1024").get((256, qc.qwen2_hidden), "weight")?;

        Ok(Self {
            layers,
            norm,
            query_144,
            query_256,
            hidden_size: qc.qwen2_hidden,
            device: device.clone(),
        })
    }

    /// Build the dual-mode attention mask.
    ///
    /// `n_img`: number of image patch tokens (non-causal, type=0)
    /// `n_query`: number of learnable query tokens (causal, type=1)
    ///
    /// Returns `[1, 1, n_img+n_query, n_img+n_query]`.
    fn build_attention_mask(&self, n_img: usize, n_query: usize, dtype: DType) -> Result<Tensor> {
        let s = n_img + n_query;
        let neg_inf = f32::NEG_INFINITY;
        let mut mask = vec![neg_inf; s * s];

        // Image tokens: can attend to all other image tokens (non-causal)
        for i in 0..n_img {
            for j in 0..n_img {
                mask[i * s + j] = 0.0;
            }
        }
        // Query tokens: attend to all image tokens + previous query tokens (causal)
        for i_q in 0..n_query {
            let i = n_img + i_q;
            // Attend to all image tokens
            for j in 0..n_img {
                mask[i * s + j] = 0.0;
            }
            // Attend causally to query tokens up to and including self
            for j_q in 0..=i_q {
                let j = n_img + j_q;
                mask[i * s + j] = 0.0;
            }
        }

        let mask_t = Tensor::from_vec(mask, (s, s), &self.device)?.to_dtype(dtype)?;
        mask_t.reshape((1, 1, s, s))
    }

    /// Encode SAM output features.
    ///
    /// `x`: `[B, encoder_hidden, H, W]`  (SAM net_3 output)
    /// Returns: `[B, n_img, encoder_hidden]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let (b, _c, _h, _w) = x.dims4()?;

        // Flatten spatial: [B, encoder_hidden, H, W] → [B, H*W, encoder_hidden]
        let x_flat = x.flatten(2, 3)?.transpose(1, 2)?; // [B, n_img, D]
        let n_img = x_flat.dim(1)?;

        // Select query based on n_img
        let query_weight = if n_img == 144 {
            &self.query_144
        } else {
            // Default: use 256-query embedding (production: 1024px → 256 patches)
            &self.query_256
        };
        let n_query = query_weight.dim(0)?;

        // Expand queries to batch: [n_query, D] → [B, n_query, D]
        let queries = query_weight
            .unsqueeze(0)?
            .broadcast_as((b, n_query, self.hidden_size))?;

        // Concatenate: [B, n_img + n_query, D]
        let combined = Tensor::cat(&[&x_flat, &queries], 1)?;

        // Build attention mask
        let mask = self.build_attention_mask(n_img, n_query, dtype)?;

        // Run through Qwen2 layers
        let mut hs = combined;
        for layer in &self.layers {
            hs = layer.forward(&hs, Some(&mask))?;
        }
        let hs = self.norm.forward(&hs)?;

        // Return only query outputs (last n_query tokens)
        hs.narrow(1, n_img, n_query)
    }
}

// ─── OCR2 MLP Projector ───────────────────────────────────────────────────────

/// Downsample MLP projector (same design as in deepseek_vl2.rs).
///
/// Unfolds `downsample_ratio × downsample_ratio` spatial patches into channel dim,
/// then applies a 2-layer MLP to project to LM hidden size.
///
/// Weight paths (`vb = projector VarBuilder`):
/// - `layers.0.{weight,bias}` — first linear
/// - `layers.1` — GELU (no weights)
/// - `layers.2.{weight,bias}` — second linear
struct Ocr2MlpProjector {
    linear1: Linear,
    linear2: Linear,
    downsample_ratio: usize,
}

impl Ocr2MlpProjector {
    fn new(
        input_dim: usize,
        output_dim: usize,
        depth: usize,
        downsample_ratio: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_l = vb.pp("layers");
        // Input after spatial unfold: input_dim * ratio²
        let unfolded_dim = input_dim * downsample_ratio * downsample_ratio;
        // Intermediate dim = n_embed * mlp_ratio (default mlp_ratio=1 for OCR2)
        let intermediate_dim = output_dim;

        if depth == 1 {
            // For depth=1, both linear1 and linear2 point to the same layer
            let l = linear(unfolded_dim, output_dim, vb_l.pp(0))?;
            return Ok(Self {
                linear1: l.clone(),
                linear2: l,
                downsample_ratio,
            });
        }

        // depth >= 2: layers.0 → GELU → layers.2 (Python Sequential with GELU at index 1)
        let linear1 = linear(unfolded_dim, intermediate_dim, vb_l.pp(0))?;
        // Middle layers (if depth > 2) are at indices 2, 3, ...
        // For depth=2: the second linear is at index 2 (Python: 0=Linear, 1=GELU, 2=Linear)
        let linear2 = linear(intermediate_dim, output_dim, vb_l.pp(2))?;

        Ok(Self {
            linear1,
            linear2,
            downsample_ratio,
        })
    }

    /// Forward: `[B, S, D]` → unfold ratio² patches → MLP → `[B, S/ratio², out_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let ratio = self.downsample_ratio;
        let h = (s as f64).sqrt() as usize;

        // Spatial unfold: [B, h, w, D] → [B, h/r, r, w/r, r, D] → [B, h/r, w/r, D*r²]
        let w = h;
        let hr = h / ratio;
        let wr = w / ratio;

        let x = x.reshape((b, h, w, d))?;
        let x = x.reshape((b, hr, ratio, wr, ratio, d))?;
        let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?; // [B, hr, wr, r, r, D]
        let x = x.reshape((b, hr * wr, d * ratio * ratio))?; // [B, S', D*r²]

        // 2-layer MLP with GELU
        let h = self.linear1.forward(&x)?.gelu_erf()?;
        self.linear2.forward(&h) // [B, S', out_dim]
    }
}

// ─── Main model ───────────────────────────────────────────────────────────────

/// DeepSeek-OCR2 vision-language model.
pub struct DeepseekOCR2ForCausalLM {
    sam_model: SamImageEncoderViT,
    qwen2_model: Ocr2Qwen2Encoder,
    projector: Ocr2MlpProjector,
    /// Learned separator token appended after each image's features.
    view_seperator: Tensor,
    language_model: Qwen2ForCausalLM,
    device: Device,
    dtype: DType,
}

impl DeepseekOCR2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ocr_cfg = Ocr2Config::from_model_config(cfg);
        let dtype = vb.dtype();
        let device = vb.device().clone();

        // SAM ViT: weights at model.sam_model.*
        let sam_model = SamImageEncoderViT::new(&ocr_cfg, vb.pp("model").pp("sam_model"))?;

        // Qwen2 encoder: weights at model.qwen2_model.*
        let qwen2_model =
            Ocr2Qwen2Encoder::new(&ocr_cfg, dtype, &device, vb.pp("model").pp("qwen2_model"))?;

        // Projector: weights at model.projector.*
        let projector = Ocr2MlpProjector::new(
            ocr_cfg.proj_input_dim,
            ocr_cfg.proj_n_embed,
            ocr_cfg.proj_depth,
            ocr_cfg.proj_downsample_ratio,
            vb.pp("model").pp("projector"),
        )?;

        // view_seperator: weights at model.view_seperator
        let view_seperator = vb.pp("model").get(ocr_cfg.proj_n_embed, "view_seperator")?;

        // Language model: Qwen2ForCausalLM (weights at model.*, lm_head.*)
        let language_model = Qwen2ForCausalLM::new(cfg, vb)?;

        Ok(Self {
            sam_model,
            qwen2_model,
            projector,
            view_seperator,
            language_model,
            device,
            dtype,
        })
    }

    /// Encode a batch of pixel values into projected image features.
    ///
    /// `pixel_values`: `[B, 3, H, W]`
    /// Returns: `[B * (n_projected + 1), lm_hidden]` — n_projected tokens + 1 separator per image.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let sam_out = self.sam_model.forward(pixel_values)?; // [B, 896, H', W']
        let enc_out = self.qwen2_model.forward(&sam_out)?; // [B, n_query, 896]
        let projected = self.projector.forward(&enc_out)?; // [B, n_proj, lm_hidden]

        // Append view_seperator to each image's features
        let (b, n_proj, d) = projected.dims3()?;
        let sep = self
            .view_seperator
            .reshape((1, 1, d))?
            .broadcast_as((b, 1, d))?;
        let combined = Tensor::cat(&[&projected, &sep], 1)?; // [B, n_proj+1, d]
                                                             // Flatten to [B*(n_proj+1), d]
        combined.reshape((b * (n_proj + 1), d))
    }

    /// Replace image-token positions in text_embeddings with image features.
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

impl crate::engine::ModelForward for DeepseekOCR2ForCausalLM {
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

    /// Build a tiny model config for fast CPU tests.
    ///
    /// Uses 2 SAM blocks, 2 Qwen2 encoder layers, 2 LM layers.
    /// img_size=32, patch_size=16 → 2×2=4 patches
    /// After neck+net_2+net_3: 2→1→1 spatial → 1×1=1 patch (n_query=1 ≤ query_256 so uses 256)
    /// NOTE: since n_query=1 ≠ 144 and 256, defaults to using query_256 (256 rows).
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // SAM config: tiny
        extra.insert(
            "sam_config".to_string(),
            json!({
                "embed_dim": 8,
                "depth": 2,
                "num_heads": 2,
                "img_size": 32,
                "patch_size": 16,
                "mlp_ratio": 2.0,
                "out_chans": 4,
                "last_conv_output": 4,
                "window_size": 0,      // use global attention for all blocks (no windowing)
                "use_rel_pos": false   // skip rel pos embeds for speed
            }),
        );
        // Qwen2 encoder config: tiny (hidden must match sam last_conv_output → proj_input_dim)
        extra.insert(
            "qwen2_encoder_config".to_string(),
            json!({
                "num_layers": 2,
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 8
            }),
        );
        // Projector config: input_dim=4 (qwen2 hidden), n_embed=8 (LM hidden), dr=1
        extra.insert(
            "projector_config".to_string(),
            json!({
                "input_dim": 4,
                "n_embed": 8,
                "depth": 2,
                "downsample_ratio": 1
            }),
        );
        ModelConfig {
            architectures: vec!["DeepseekOCR2ForCausalLM".to_string()],
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
    fn test_deepseek_ocr2_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCR2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "DeepseekOCR2ForCausalLM construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_deepseek_ocr2_sam_vit_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCR2ForCausalLM::new(&cfg, vb).unwrap();
        let ocr_cfg = Ocr2Config::from_model_config(&cfg);

        // pixel_values: [1, 3, 32, 32]
        let pixel_values =
            Tensor::zeros((1usize, 3usize, 32usize, 32usize), DType::F32, &device).unwrap();
        let sam_out = model.sam_model.forward(&pixel_values);
        assert!(
            sam_out.is_ok(),
            "SAM ViT forward failed: {:?}",
            sam_out.err()
        );
        let out = sam_out.unwrap();
        // Output: [1, last_conv_output, H', W'] where H' = spatial/4 (two stride=2 convs)
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(1).unwrap(), ocr_cfg.sam_last_conv);
    }

    #[test]
    fn test_deepseek_ocr2_qwen2_encoder_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCR2ForCausalLM::new(&cfg, vb).unwrap();
        let ocr_cfg = Ocr2Config::from_model_config(&cfg);

        // Simulate SAM output: [1, sam_last_conv, H', W']
        // For 32px image with 2 stride-2 convolutions: 2→1→1
        let sam_out = Tensor::zeros(
            (1usize, ocr_cfg.sam_last_conv, 1usize, 1usize),
            DType::F32,
            &device,
        )
        .unwrap();

        let enc_out = model.qwen2_model.forward(&sam_out);
        assert!(
            enc_out.is_ok(),
            "Qwen2 encoder forward failed: {:?}",
            enc_out.err()
        );
        let out = enc_out.unwrap();
        // Output: [B, n_query, qwen2_hidden] where n_query=256 (default)
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(2).unwrap(), ocr_cfg.qwen2_hidden);
    }

    #[test]
    fn test_deepseek_ocr2_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepseekOCR2ForCausalLM::new(&cfg, vb).unwrap();
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
    fn test_deepseek_ocr2_projector_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let ocr_cfg = Ocr2Config::from_model_config(&cfg);

        // Create projector directly: input_dim=4, n_embed=8, depth=2, dr=1
        let proj = Ocr2MlpProjector::new(
            ocr_cfg.proj_input_dim,
            ocr_cfg.proj_n_embed,
            ocr_cfg.proj_depth,
            ocr_cfg.proj_downsample_ratio,
            vb.pp("projector"),
        )
        .unwrap();

        // Input: [1, 1, 4] (1 spatial position, 4-dim features)
        let x = Tensor::zeros(
            (1usize, 1usize, ocr_cfg.proj_input_dim),
            DType::F32,
            &device,
        )
        .unwrap();
        let out = proj.forward(&x);
        assert!(out.is_ok(), "projector forward failed: {:?}", out.err());
        let o = out.unwrap();
        assert_eq!(o.dim(0).unwrap(), 1);
        assert_eq!(o.dim(2).unwrap(), ocr_cfg.proj_n_embed);
    }
}
