//! ERNIE 4.5 Vision-Language Model.
//!
//! Architecture:
//! - `Ernie4_5_VisionTransformer`: linear patch embed + 2-D spatial RoPE + N blocks + LayerNorm
//! - `VariableResolutionResamplerModel`: spatial group-pooling (spatial_conv_size²) + 2-layer MLP
//! - `Ernie45MoeForCausalLM`: existing ERNIE 4.5 MoE text model
//!
//! # Key differences from Step3-VL
//!
//! - Patch embed is a **Linear** layer on pre-flattened patches (`[np, C*p²]`), NOT Conv2d.
//! - Vision encoder uses 2-D spatial RoPE on (h, w) positions (no CLS token, no cls-padding).
//! - Projector is a 2-layer MLP with standard GELU (NOT QuickGELU), followed by RMSNorm.
//! - Resampler weight paths use `spatial_linear.{0,2,3}` (nn.Sequential indices in HF checkpoint).
//!
//! # Weight paths (HF checkpoint → VarBuilder)
//!
//! ```text
//! vision_model.*                             → vb.pp("vision_model").*
//! model.resampler_model.spatial_linear.0.*   → vb.pp("model").pp("resampler_model")
//!                                                .pp("spatial_linear").pp("0").*
//! model.resampler_model.mlp.*               → vb.pp("model").pp("resampler_model").pp("mlp").*
//! model.resampler_model.after_norm.*        → …pp("after_norm").*
//! model.layers.{i}.*                        → Ernie45MoeForCausalLM (vb root)
//! lm_head.*                                 → Ernie45MoeForCausalLM (vb root)
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/ernie45_vl.py`

// Intentional: match Baidu's Python naming convention (Ernie4_5_*)
#![allow(non_camel_case_types)]

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm};
use crate::multimodal::MultimodalInputs;

use super::ernie45_moe::Ernie45MoeForCausalLM;

// ─── Vision Config ────────────────────────────────────────────────────────────

/// Vision encoder configuration (from `cfg.extra["vision_config"]`).
#[derive(Debug, Clone, Deserialize)]
struct Ernie45VisionConfig {
    /// Patch size (square).
    #[serde(default = "default_patch_size")]
    patch_size: usize,
    /// Input channels.
    #[serde(default = "default_in_channels")]
    in_channels: usize,
    /// ViT embedding / hidden dimension (`embed_dim`).
    #[serde(rename = "hidden_size")]
    embed_dim: usize,
    /// Number of transformer blocks.
    #[serde(rename = "depth")]
    depth: usize,
    /// Number of attention heads.
    #[serde(rename = "num_heads")]
    num_heads: usize,
    /// MLP hidden = embed_dim * mlp_ratio.
    #[serde(default = "default_mlp_ratio")]
    mlp_ratio: f64,
}

fn default_patch_size() -> usize {
    14
}
fn default_in_channels() -> usize {
    3
}
fn default_mlp_ratio() -> f64 {
    4.0
}

impl Ernie45VisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> candle_core::Result<Self> {
        let raw = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        serde_json::from_value(raw)
            .map_err(|e| candle_core::Error::Msg(format!("ernie45_vl: bad vision_config: {e}")))
    }
}

/// Extract top-level VL construction parameters from `ModelConfig`.
fn vl_params(cfg: &ModelConfig, vis: &Ernie45VisionConfig) -> (usize, usize, usize, f64) {
    let spatial_conv_size = cfg
        .extra
        .get("spatial_conv_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    (
        spatial_conv_size,
        vis.embed_dim,   // pixel_hidden_size
        cfg.hidden_size, // LLM hidden size
        cfg.rms_norm_eps,
    )
}

// ─── 2-D Spatial RoPE ────────────────────────────────────────────────────────

/// Pre-computed inverse frequencies for 2-D spatial RoPE used by the ViT.
///
/// Unlike text-model RoPE, this embeds (h, w) patch positions independently
/// using two separate lookup tables for the h and w axes, then concatenates.
struct Ernie4_5_VisionRotaryEmbedding {
    /// `[dim/2]` inverse frequencies where `dim = head_dim // 2`.
    inv_freq: Vec<f32>,
}

impl Ernie4_5_VisionRotaryEmbedding {
    fn new(dim: usize) -> Self {
        // dim = head_dim // 2; inv_freq length = dim / 2 = head_dim / 4
        let n = dim / 2;
        let inv_freq: Vec<f32> = (0..n)
            .map(|i| 1.0_f32 / 10000_f32.powf((2 * i) as f32 / dim as f32))
            .collect();
        Self { inv_freq }
    }

    /// Return frequency table: `[seqlen, head_dim/4]`.
    fn freq_table(&self, seqlen: usize) -> Vec<Vec<f32>> {
        (0..seqlen)
            .map(|pos| {
                self.inv_freq
                    .iter()
                    .map(|&f| pos as f32 * f)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Build position embeddings for all patches given a list of (t, h, w) grids.
    ///
    /// Returns `[np, head_dim/2]` freqs (before .cos()/.sin()).
    ///
    /// The Python reference interleaves h/w position IDs using a tiled reshape
    /// with `spatial_merge_size`, then looks up freqs independently for h and w,
    /// and concatenates them to form `[np, head_dim/2]`.
    fn build_pos_emb(
        &self,
        grid_thw: &[[usize; 3]],
        spatial_merge_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let n_freq = self.inv_freq.len(); // head_dim/4
        let max_grid = grid_thw
            .iter()
            .flat_map(|&[_, h, w]| [h, w])
            .max()
            .unwrap_or(1);
        let freq_table = self.freq_table(max_grid);

        let mut all_freqs: Vec<f32> = Vec::new();

        for &[t, h, w] in grid_thw {
            // Build h/w position grids with the same interleaved tiling the Python uses.
            // Reshape h×w → (h/m, m, w/m, m), permute (0,2,1,3), flatten.
            let m = spatial_merge_size;
            let hm = h / m;
            let wm = w / m;

            // h_ids[row][col] = row index for patch at (row, col)
            // After the tiled reshape + flatten they are in the order:
            // (tile_row, tile_col, local_row, local_col) → flatten
            let mut h_ids: Vec<usize> = Vec::with_capacity(h * w);
            let mut w_ids: Vec<usize> = Vec::with_capacity(h * w);
            for tr in 0..hm {
                for tc in 0..wm {
                    for lr in 0..m {
                        for lc in 0..m {
                            h_ids.push(tr * m + lr);
                            w_ids.push(tc * m + lc);
                        }
                    }
                }
            }

            // Repeat for t temporal frames
            for _ in 0..t {
                for (&hi, &wi) in h_ids.iter().zip(w_ids.iter()) {
                    // h freqs and w freqs, each [head_dim/4], concatenated → [head_dim/2]
                    all_freqs.extend_from_slice(&freq_table[hi]);
                    all_freqs.extend_from_slice(&freq_table[wi]);
                }
            }
        }

        let np = all_freqs.len() / (2 * n_freq); // total patches
        Tensor::from_vec(all_freqs, (np, 2 * n_freq), device)?.to_dtype(dtype)
    }
}

// ─── Vision Attention ─────────────────────────────────────────────────────────

/// Vision self-attention with 2-D spatial RoPE (no KV cache — encoder only).
///
/// Input shape: `[S, 1, D]` (sequence-first, batch=1).
struct Ernie4_5_VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Ernie4_5_VisionAttention {
    fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let qkv = linear(embed_dim, 3 * embed_dim, vb.pp("qkv"))?;
        let proj = linear(embed_dim, embed_dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
        })
    }

    /// Apply standard rotary embeddings to `x: [B, heads, S, head_dim]`.
    ///
    /// `pos_emb: [S, rotary_dim]` where `rotary_dim = head_dim / 2`.
    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = x.dim(3)?;
        let rotary_dim = cos.dim(1)?;
        let x_rot = x.narrow(3, 0, rotary_dim)?;
        let x_pass = x.narrow(3, rotary_dim, d - rotary_dim)?;

        // rotate_half: split rotary_dim in half, form [-x2, x1]
        let half = rotary_dim / 2;
        let x1 = x_rot.narrow(3, 0, half)?;
        let x2 = x_rot.narrow(3, half, half)?;
        let x_rot_h = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

        // Broadcast cos/sin [S, rotary_dim] → [1, 1, S, rotary_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x_rot_new = (x_rot.broadcast_mul(&cos)? + x_rot_h.broadcast_mul(&sin)?)?;
        Tensor::cat(&[&x_rot_new, &x_pass], 3)
    }

    /// Forward attention.
    ///
    /// `x: [S, 1, D]`, `rotary_pos_emb: [S, head_dim/2]` (cos/sin computed inside).
    fn forward(&self, x: &Tensor, rotary_pos_emb: &Tensor) -> Result<Tensor> {
        let (s, _b, d) = x.dims3()?;

        // [S, 1, D] → [S, 1, 3D] → split → 3 × [S, 1, D]
        let qkv = self.qkv.forward(x)?;
        let q = qkv.narrow(2, 0, d)?.contiguous()?;
        let k = qkv.narrow(2, d, d)?.contiguous()?;
        let v = qkv.narrow(2, 2 * d, d)?.contiguous()?;

        // [S, 1, D] → [S, 1, heads, head_dim] → [1, heads, S, head_dim]
        let reshape_qkv = |t: &Tensor| -> Result<Tensor> {
            t.reshape((s, 1, self.num_heads, self.head_dim))?
                .permute((1, 2, 0, 3))
        };
        let q = reshape_qkv(&q)?;
        let k = reshape_qkv(&k)?;
        let v = reshape_qkv(&v)?;

        // Apply 2-D spatial RoPE to Q and K
        let cos = rotary_pos_emb.cos()?;
        let sin = rotary_pos_emb.sin()?;
        let q = Self::apply_rotary(&q, &cos, &sin)?;
        let k = Self::apply_rotary(&k, &cos, &sin)?;

        // Scaled dot-product attention [1, heads, S, S]
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // [1, heads, S, head_dim] → [S, 1, D]
        let out = out.permute((2, 0, 1, 3))?.reshape((s, 1, d))?;

        self.proj.forward(&out)
    }
}

// ─── Vision MLP ──────────────────────────────────────────────────────────────

/// Two-layer MLP with QuickGELU activation used in each vision encoder block.
struct Ernie4_5_VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl Ernie4_5_VisionMLP {
    fn new(embed_dim: usize, mlp_hidden: usize, vb: VarBuilder) -> Result<Self> {
        // Python: ColumnParallelLinear / RowParallelLinear, default bias=True
        let fc1 = linear(embed_dim, mlp_hidden, vb.pp("fc1"))?;
        let fc2 = linear(mlp_hidden, embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // QuickGELU: x * sigmoid(1.702 * x)
        let h = self.fc1.forward(x)?;
        let h = (candle_nn::ops::sigmoid(&h.affine(1.702, 0.0)?)?.mul(&h))?;
        self.fc2.forward(&h)
    }
}

// ─── Vision Encoder Block ─────────────────────────────────────────────────────

struct Ernie4_5_VisionBlock {
    norm1: LayerNorm,
    attn: Ernie4_5_VisionAttention,
    norm2: LayerNorm,
    mlp: Ernie4_5_VisionMLP,
}

impl Ernie4_5_VisionBlock {
    fn new(cfg: &Ernie45VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mlp_hidden = (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize;
        let norm1 = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm2"))?;
        let attn = Ernie4_5_VisionAttention::new(cfg.embed_dim, cfg.num_heads, vb.pp("attn"))?;
        let mlp = Ernie4_5_VisionMLP::new(cfg.embed_dim, mlp_hidden, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, rotary_pos_emb: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?, rotary_pos_emb)?)?;
        let x = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

// ─── Vision Transformer ───────────────────────────────────────────────────────

/// ERNIE 4.5 vision encoder.
///
/// Takes pre-flattened patches `[np, in_channels * patch_size²]` + grid_thw, produces
/// `[np, embed_dim]`.
struct Ernie4_5_VisionTransformer {
    /// Linear patch embedding (NO bias, per Python `nn.Linear(..., bias=False)`).
    patch_embed: Linear,
    rotary_pos_emb: Ernie4_5_VisionRotaryEmbedding,
    blocks: Vec<Ernie4_5_VisionBlock>,
    /// Final layer norm.
    ln: LayerNorm,
    spatial_merge_size: usize,
}

impl Ernie4_5_VisionTransformer {
    fn new(cfg: &Ernie45VisionConfig, spatial_merge_size: usize, vb: VarBuilder) -> Result<Self> {
        let cps = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let patch_embed = linear_no_bias(cps, cfg.embed_dim, vb.pp("patch_embed").pp("proj"))?;

        let head_dim = cfg.embed_dim / cfg.num_heads;
        let rotary_pos_emb = Ernie4_5_VisionRotaryEmbedding::new(head_dim / 2);

        let mut blocks = Vec::with_capacity(cfg.depth);
        let vb_blocks = vb.pp("blocks");
        for i in 0..cfg.depth {
            blocks.push(Ernie4_5_VisionBlock::new(cfg, vb_blocks.pp(i))?);
        }

        let ln = layer_norm(cfg.embed_dim, 1e-6, vb.pp("ln"))?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            ln,
            spatial_merge_size,
        })
    }

    fn forward(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Tensor> {
        // pixel_values: [np, cps]
        let device = pixel_values.device();
        let dtype = pixel_values.dtype();

        // Embed patches: [np, cps] → [np, D]
        let x = self.patch_embed.forward(pixel_values)?;

        // Build 2-D RoPE position embeddings: [np, head_dim/2]
        let rotary_pos_emb =
            self.rotary_pos_emb
                .build_pos_emb(grid_thw, self.spatial_merge_size, device, dtype)?;

        // Add batch dim for attention: [np, D] → [np, 1, D]
        let mut x = x.unsqueeze(1)?;

        for block in &self.blocks {
            x = block.forward(&x, &rotary_pos_emb)?;
        }

        // Remove batch dim and apply final norm: [np, 1, D] → [np, D]
        let x = x.squeeze(1)?;
        self.ln.forward(&x)
    }
}

// ─── Variable Resolution Resampler ───────────────────────────────────────────

/// Spatial pooling resampler: groups spatial_conv_size² patches and projects to LM hidden dim.
///
/// Weight paths relative to `model.resampler_model` in the HF checkpoint:
/// - `spatial_linear.0.*` → `spatial_linear1` (Linear with bias)
/// - `spatial_linear.2.*` → `spatial_linear2` (Linear with bias)
/// - `spatial_linear.3.*` → `spatial_norm`    (LayerNorm)
/// - `mlp.*`              → `mlp`             (Linear with bias)
/// - `after_norm.*`       → `after_norm`      (RMSNorm)
struct VariableResolutionResamplerModel {
    spatial_linear1: Linear,
    spatial_linear2: Linear,
    spatial_norm: LayerNorm,
    mlp: Linear,
    after_norm: RmsNorm,
    spatial_conv_size: usize,
}

impl VariableResolutionResamplerModel {
    fn new(
        spatial_dim: usize, // pixel_hidden_size * spatial_conv_size²
        out_dim: usize,     // LLM hidden_size
        spatial_conv_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder, // rooted at `model.resampler_model`
    ) -> Result<Self> {
        let vb_sl = vb.pp("spatial_linear");
        let spatial_linear1 = linear(spatial_dim, spatial_dim, vb_sl.pp("0"))?;
        let spatial_linear2 = linear(spatial_dim, spatial_dim, vb_sl.pp("2"))?;
        let spatial_norm = layer_norm(spatial_dim, 1e-6, vb_sl.pp("3"))?;
        let mlp = linear(spatial_dim, out_dim, vb.pp("mlp"))?;
        let after_norm = rms_norm(out_dim, rms_norm_eps, vb.pp("after_norm"))?;
        Ok(Self {
            spatial_linear1,
            spatial_linear2,
            spatial_norm,
            mlp,
            after_norm,
            spatial_conv_size,
        })
    }

    /// Compress `[np, embed_dim]` → `[np/m², out_dim]`.
    ///
    /// `np` must be divisible by `spatial_conv_size²`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (s, c) = x.dims2()?;
        let m = self.spatial_conv_size;

        // spatial_conv_reshape: group m² adjacent patches → [s/m², c*m²]
        let x = x.reshape((s / (m * m), c * m * m))?;

        // Two-layer MLP with standard GELU
        let x = self.spatial_linear1.forward(&x)?;
        let x = x.gelu_erf()?;
        let x = self.spatial_linear2.forward(&x)?;
        let x = self.spatial_norm.forward(&x)?;

        // Final projection + RMSNorm
        let x = self.mlp.forward(&x)?;
        self.after_norm.forward(&x)
    }
}

// ─── Multimodal Merge ─────────────────────────────────────────────────────────

/// Replace image-patch token embeddings with projected visual features.
///
/// For each `(position, ProcessedImage)` in `mm_inputs.image_embeddings`, the
/// `embedding` rows are written into `text_embeds` starting at `position`.
///
/// `text_embeds: [B, S, D]`, each image embedding: `[num_tokens, D]`.
fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }

    let (_batch_size, seq_len, _hidden) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_vec3::<f32>()?;

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

    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main Model ───────────────────────────────────────────────────────────────

/// ERNIE 4.5 vision-language model for conditional generation.
pub struct Ernie4_5_VLForConditionalGeneration {
    vision_model: Ernie4_5_VisionTransformer,
    resampler_model: VariableResolutionResamplerModel,
    language_model: Ernie45MoeForCausalLM,
    device: Device,
}

impl Ernie4_5_VLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_cfg = Ernie45VisionConfig::from_model_config(cfg)?;
        let (spatial_conv_size, pixel_hidden_size, hidden_size, rms_norm_eps) =
            vl_params(cfg, &vis_cfg);

        let vision_model =
            Ernie4_5_VisionTransformer::new(&vis_cfg, spatial_conv_size, vb.pp("vision_model"))?;

        // Resampler input dim = ViT embed_dim × (spatial_conv_size)²
        let spatial_dim = pixel_hidden_size * spatial_conv_size * spatial_conv_size;
        let resampler_model = VariableResolutionResamplerModel::new(
            spatial_dim,
            hidden_size,
            spatial_conv_size,
            rms_norm_eps,
            vb.pp("model").pp("resampler_model"),
        )?;

        // The language model loads from the same vb root (model.* and lm_head.*)
        let language_model = Ernie45MoeForCausalLM::new(cfg, vb.clone())?;

        Ok(Self {
            vision_model,
            resampler_model,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Run vision encoder + resampler on all images.
    ///
    /// `pixel_values: [np, cps]`, `grid_thw`: one `[t, h, w]` entry per image.
    /// Returns: `[np/m², hidden_size]` (all images concatenated in sequence order).
    pub fn encode_images(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Tensor> {
        // Run all patches through the vision transformer
        let all_features = self.vision_model.forward(pixel_values, grid_thw)?; // [np, D]

        // Run resampler per-image and concatenate
        let mut results = Vec::with_capacity(grid_thw.len());
        let mut offset = 0usize;
        for &[t, h, w] in grid_thw {
            let np_i = t * h * w;
            let features_i = all_features.narrow(0, offset, np_i)?;
            let resampled = self.resampler_model.forward(&features_i)?;
            results.push(resampled);
            offset += np_i;
        }

        if results.len() == 1 {
            Ok(results.remove(0))
        } else {
            Tensor::cat(&results, 0)
        }
    }
}

impl crate::engine::ModelForward for Ernie4_5_VLForConditionalGeneration {
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
            merge_multimodal(&text_embeddings, mm_inputs, &self.device)?
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
    use candle_core::DType;
    use serde_json::json;

    /// Tiny model config for fast CPU tests.
    ///
    /// patch_size=2, in_channels=1 → cps=4
    /// embed_dim=16, depth=1, num_heads=2, mlp_ratio=2.0
    /// spatial_conv_size=2: 4 patches per image group → 1 resampled token
    /// hidden_size=32 (LLM)
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Vision config
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 16,
                "depth": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "patch_size": 2,
                "in_channels": 1
            }),
        );
        // VL config
        extra.insert("spatial_conv_size".to_string(), json!(2));
        extra.insert("im_patch_id".to_string(), json!(5));
        // Ernie45Moe text config (dense only, no MoE)
        extra.insert("moe_num_experts".to_string(), json!(0));
        extra.insert("use_bias".to_string(), json!(false));

        ModelConfig {
            architectures: vec!["Ernie4_5_VLMoeForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 64,
            max_position_embeddings: 128,
            head_dim: 16,
            hidden_act: "silu".to_string(),
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

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 4,
            num_blocks: 32,
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
    fn test_ernie45_vl_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ernie4_5_VLForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "model construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_ernie45_vl_vision_only() {
        // 4 patches (2×2 grid, t=1), cps = 1*2*2 = 4
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ernie4_5_VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let grid_thw = [[1usize, 2, 2]];

        let features = model.vision_model.forward(&pixel_values, &grid_thw);
        assert!(
            features.is_ok(),
            "vision forward failed: {:?}",
            features.err()
        );
        let features = features.unwrap();
        assert_eq!(features.dims(), &[4, 16], "expected [np=4, embed_dim=16]");
    }

    #[test]
    fn test_ernie45_vl_resampler() {
        // 4 patches → 1 resampled token (4 / 2² = 1)
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ernie4_5_VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let grid_thw = [[1usize, 2, 2]];
        let features = model
            .vision_model
            .forward(&pixel_values, &grid_thw)
            .unwrap();
        let resampled = model.resampler_model.forward(&features);
        assert!(resampled.is_ok(), "resampler failed: {:?}", resampled.err());
        assert_eq!(
            resampled.unwrap().dims(),
            &[1, 32],
            "expected [1 token, hidden=32]"
        );
    }

    #[test]
    fn test_ernie45_vl_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ernie4_5_VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        use crate::kv_cache::BlockTable;
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let result = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_ernie45_vl_with_image() {
        use crate::multimodal::{MultimodalInputs, ProcessedImage};

        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ernie4_5_VLForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode 1 image: 2×2 grid (4 patches, spatial_conv_size=2 → 1 projected token)
        let pixel_values = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let grid_thw = [[1usize, 2, 2]];
        let image_feats = model
            .encode_images(&pixel_values, &grid_thw)
            .expect("encode_images failed"); // [1, 32]

        // Build multimodal inputs: image embedding starts at position 1 (seq pos 1, batch 0)
        let processed = ProcessedImage::new(image_feats, 1);
        let mm = MultimodalInputs::with_images(vec![0, 5, 0, 0], vec![(1, processed)]);

        // Input sequence: [<text>, <im_patch>, <text>, <text>] — length 4
        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        use crate::kv_cache::BlockTable;
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::from_slice(&[0u32, 5, 0, 0], (1usize, seq_len), &device).unwrap();
        let result =
            model.forward_multimodal(&input_ids, Some(&mm), 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "multimodal forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }
}
