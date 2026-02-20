//! GLM-4.1V vision-language model.
//!
//! `Glm4vForConditionalGeneration` combines a GLM-4.1V vision transformer with
//! `Glm4ForCausalLM` as the language backbone.
//!
//! # Architecture
//!
//! ```text
//! pixel_values [np, C*T*P*P]
//!   → Linear patch embed [np, hidden]          (bias=True, Conv3D equiv)
//!   → post_conv_layernorm (RmsNorm)
//!   → Glm4vVisionEmbeddings: bilinear-interpolated 2D pos emb added
//!   → 2D spatial RoPE (partial_rotary_factor=0.5, neox-style, sm-block-tiled h/w)
//!   → depth × Glm4vVisionBlock
//!       norm1(RmsNorm) + attn(qkv no-bias, no q/k norm, partial RoPE, SDPA) + res
//!       norm2(RmsNorm) + mlp(SwiGLU no-bias, mlp_dim=out_hidden) + res
//!   → post_layernorm (RmsNorm)
//!   → reshape [np/sm², sm, sm, hidden] → permute → [np/sm², hidden, sm, sm]
//!   → Conv2d downsample (hidden→out_hidden, kernel=sm, stride=sm)
//!   → Glm4vPatchMerger (proj → LayerNorm → GELU → SwiGLU, context_dim=intermediate_size)
//!   → merge into LLM embeddings at image-token positions
//!   → Glm4ForCausalLM → logits
//! ```
//!
//! # Key Differences from GLM-OCR
//!
//! - No per-head `q_norm`/`k_norm` in attention; `qkv`/`proj` bias=False.
//! - Vision block MLP `hidden_dim = out_hidden_size` (not `intermediate_size`);
//!   bias=False for gate_up and down.
//! - Vision transformer adds `post_conv_layernorm` + learnable 2D position embeddings.
//! - Merger `context_dim = intermediate_size`.
//!
//! # Reference
//! `reference/vllm/vllm/model_executor/models/glm4_1v.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, layer_norm, linear_b, linear_no_bias, ops::softmax_last_dim, Conv2d, Conv2dConfig,
    Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::glm4::Glm4ForCausalLM;

// ─── Vision Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Glm4vVisionConfig {
    hidden_size: usize,
    out_hidden_size: usize,
    num_heads: usize,
    depth: usize,
    /// Used for merger context_dim (distinct from vision block MLP dim).
    intermediate_size: usize,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    /// Base image size; determines the learnable position embedding grid.
    image_size: usize,
    rms_norm_eps: f64,
}

impl Glm4vVisionConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn patch_flat_size(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Number of positions in the base embedding grid (= (image_size/patch_size)²).
    fn num_positions(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Grid side length of the base embedding (= sqrt(num_positions)).
    fn orig_emb_size(&self) -> usize {
        self.image_size / self.patch_size
    }
}

impl Default for Glm4vVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536,
            out_hidden_size: 4096,
            num_heads: 12,
            depth: 24,
            intermediate_size: 13824,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            image_size: 1120,
            rms_norm_eps: 1e-6,
        }
    }
}

impl Glm4vVisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vc = cfg.extra.get("vision_config");
        let d = Self::default();
        Self {
            hidden_size: vc
                .and_then(|v| v.get("hidden_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.hidden_size as u64) as usize,
            out_hidden_size: vc
                .and_then(|v| v.get("out_hidden_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.out_hidden_size as u64) as usize,
            num_heads: vc
                .and_then(|v| v.get("num_heads"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.num_heads as u64) as usize,
            depth: vc
                .and_then(|v| v.get("depth"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.depth as u64) as usize,
            intermediate_size: vc
                .and_then(|v| v.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.intermediate_size as u64) as usize,
            in_channels: vc
                .and_then(|v| v.get("in_channels"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.in_channels as u64) as usize,
            patch_size: vc
                .and_then(|v| v.get("patch_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.patch_size as u64) as usize,
            temporal_patch_size: vc
                .and_then(|v| v.get("temporal_patch_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.temporal_patch_size as u64) as usize,
            spatial_merge_size: vc
                .and_then(|v| v.get("spatial_merge_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.spatial_merge_size as u64) as usize,
            image_size: vc
                .and_then(|v| v.get("image_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.image_size as u64) as usize,
            rms_norm_eps: vc
                .and_then(|v| v.get("rms_norm_eps"))
                .and_then(|v| v.as_f64())
                .unwrap_or(d.rms_norm_eps),
        }
    }
}

// ─── RoPE Helper ─────────────────────────────────────────────────────────────

/// Apply neox-style partial RoPE to `x`.
///
/// - `x`: `[S, num_heads, head_dim]`
/// - `cos`, `sin`: `[S, rotary_dim]` where `rotary_dim = head_dim / 2`
///
/// Rotates the first `rotary_dim` dims; the remainder passes through unchanged.
fn apply_partial_rope_neox(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<Tensor> {
    let (_, _, head_dim) = x.dims3()?;
    let half_rot = rotary_dim / 2;

    let x_rot = x.narrow(2, 0, rotary_dim)?;
    let x_pass = x.narrow(2, rotary_dim, head_dim - rotary_dim)?;

    // Neox rotate_half: [-x2, x1] where x_rot = [x1, x2].
    let x1 = x_rot.narrow(2, 0, half_rot)?;
    let x2 = x_rot.narrow(2, half_rot, half_rot)?;
    let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], 2)?;

    let cos = cos.unsqueeze(1)?; // [S, 1, rotary_dim]
    let sin = sin.unsqueeze(1)?;

    let x_rot_out = (x_rot.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&x_rot_out, &x_pass], 2)
}

// ─── Vision Embeddings ───────────────────────────────────────────────────────

/// Learnable 2D position embeddings with bilinear interpolation.
///
/// The learned table has `orig_size × orig_size` entries.  At inference time,
/// variable-resolution images may have different grid sizes, so the embeddings
/// are sampled via bilinear interpolation at the actual patch coordinates.
///
/// Python reference: `Glm4vVisionEmbeddings.forward` using `F.grid_sample`
/// with `mode='bicubic', align_corners=False, padding_mode='border'`.
struct Glm4vVisionEmbeddings {
    position_embedding: Embedding, // [num_positions, embed_dim]
    orig_size: usize,              // = image_size / patch_size
    embed_dim: usize,
}

impl Glm4vVisionEmbeddings {
    fn new(cfg: &Glm4vVisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_positions = cfg.num_positions();
        let embed_dim = cfg.hidden_size;
        let position_embedding =
            candle_nn::embedding(num_positions, embed_dim, vb.pp("position_embedding"))?;
        Ok(Self {
            position_embedding,
            orig_size: cfg.orig_emb_size(),
            embed_dim,
        })
    }

    /// Add bilinear-interpolated position embeddings to patch embeddings.
    ///
    /// - `embeddings`: `[total_np, embed_dim]`
    /// - `grid_thw`: `[[t, h, w]]` per image
    /// - `h_ids`, `w_ids`: integer patch grid coordinates, shape `[total_np]`
    ///
    /// Returns `[total_np, embed_dim]`.
    fn forward(
        &self,
        embeddings: &Tensor,
        grid_thw: &[[usize; 3]],
        h_ids: &[u32],
        w_ids: &[u32],
    ) -> Result<Tensor> {
        let total_np = h_ids.len();
        if total_np == 0 {
            return Ok(embeddings.clone());
        }

        let orig = self.orig_size;
        let d = self.embed_dim;

        // Extract embedding weights to CPU for bilinear sampling.
        let weight = self.position_embedding.embeddings().to_dtype(DType::F32)?;
        let weight_vec: Vec<Vec<f32>> = weight.to_vec2()?; // [orig*orig, d]

        // Build per-patch target dimensions (target_h, target_w).
        let mut target_dims: Vec<(f32, f32)> = Vec::with_capacity(total_np);
        for &[t, h, w] in grid_thw {
            let count = t * h * w;
            for _ in 0..count {
                target_dims.push((h as f32, w as f32));
            }
        }

        // Bilinear sample: for each patch compute float pixel coords then blend.
        //
        // Python align_corners=False formula:
        //   norm = ((coord + 0.5) / target) * 2 - 1
        //   src_pixel = ((norm + 1) / 2) * orig - 0.5
        //             = (coord + 0.5) * orig / target - 0.5
        let mut adapted: Vec<f32> = Vec::with_capacity(total_np * d);
        for i in 0..total_np {
            let (th, tw) = target_dims[i];
            let src_h = (h_ids[i] as f32 + 0.5) * orig as f32 / th - 0.5;
            let src_w = (w_ids[i] as f32 + 0.5) * orig as f32 / tw - 0.5;

            // Bilinear neighbourhood with border clamping.
            let clamp = |v: i64| -> usize { v.max(0).min(orig as i64 - 1) as usize };
            let h0 = src_h.floor() as i64;
            let w0 = src_w.floor() as i64;
            let fh = (src_h - src_h.floor()).clamp(0.0, 1.0);
            let fw = (src_w - src_w.floor()).clamp(0.0, 1.0);

            let p00 = &weight_vec[clamp(h0) * orig + clamp(w0)];
            let p01 = &weight_vec[clamp(h0) * orig + clamp(w0 + 1)];
            let p10 = &weight_vec[clamp(h0 + 1) * orig + clamp(w0)];
            let p11 = &weight_vec[clamp(h0 + 1) * orig + clamp(w0 + 1)];

            for j in 0..d {
                let v = (1.0 - fh) * ((1.0 - fw) * p00[j] + fw * p01[j])
                    + fh * ((1.0 - fw) * p10[j] + fw * p11[j]);
                adapted.push(v);
            }
        }

        let pos_emb = Tensor::from_vec(adapted, (total_np, d), embeddings.device())?
            .to_dtype(embeddings.dtype())?;

        embeddings + pos_emb
    }
}

// ─── Vision Attention ────────────────────────────────────────────────────────

/// GLM-4.1V vision attention: standard QKV without per-head q/k norms.
///
/// qkv and proj have bias=False (unlike GLM-OCR which uses bias=True and adds
/// per-head RMSNorm on q and k).
struct Glm4vVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Glm4vVisionAttention {
    fn new(cfg: &Glm4vVisionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let qkv = linear_no_bias(d, 3 * d, vb.pp("qkv"))?;
        let proj = linear_no_bias(d, d, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    /// Dense self-attention with partial neox RoPE.
    ///
    /// - `x`: `[np, hidden]`
    /// - `cos`, `sin`: `[np, rotary_dim]`
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (np, _) = x.dims2()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let qkv = self.qkv.forward(x)?; // [np, 3*hidden]
        let q = qkv.narrow(1, 0, h * d)?.reshape((np, h, d))?;
        let k = qkv.narrow(1, h * d, h * d)?.reshape((np, h, d))?;
        let v = qkv.narrow(1, 2 * h * d, h * d)?.reshape((np, h, d))?;

        // Partial RoPE: first head_dim/2 dims, neox style.
        let rotary_dim = d / 2;
        let q = apply_partial_rope_neox(&q, cos, sin, rotary_dim)?;
        let k = apply_partial_rope_neox(&k, cos, sin, rotary_dim)?;

        // Dense SDPA: [h, np, np].
        let scale = (d as f64).powf(-0.5);
        let q_t = q.permute((1, 0, 2))?; // [h, np, d]
        let k_t = k.permute((1, 2, 0))?; // [h, d, np]
        let v_t = v.permute((1, 0, 2))?; // [h, np, d]

        let scores = (q_t.matmul(&k_t)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_t)?; // [h, np, d]

        let ctx = ctx.permute((1, 0, 2))?.reshape((np, h * d))?;
        self.proj.forward(&ctx)
    }
}

// ─── Vision MLP ──────────────────────────────────────────────────────────────

/// SwiGLU MLP with bias=False and `mlp_dim = out_hidden_size`.
struct Glm4vVisionMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    mlp_dim: usize,
}

impl Glm4vVisionMLP {
    /// `dim` = block hidden size; `mlp_dim` = out_hidden_size (bottleneck).
    fn new(dim: usize, mlp_dim: usize, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = linear_no_bias(dim, 2 * mlp_dim, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(mlp_dim, dim, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            mlp_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate_up.narrow(1, 0, self.mlp_dim)?)?;
        let up = gate_up.narrow(1, self.mlp_dim, self.mlp_dim)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Vision Block ────────────────────────────────────────────────────────────

struct Glm4vVisionBlock {
    norm1: RmsNorm,
    attn: Glm4vVisionAttention,
    norm2: RmsNorm,
    mlp: Glm4vVisionMLP,
}

impl Glm4vVisionBlock {
    fn new(cfg: &Glm4vVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm1"))?;
        let attn = Glm4vVisionAttention::new(cfg, vb.pp("attn"))?;
        let norm2 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm2"))?;
        // mlp_dim = out_hidden_size for GLM4-1V (not intermediate_size).
        let mlp = Glm4vVisionMLP::new(cfg.hidden_size, cfg.out_hidden_size, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x1 = (&self.attn.forward(&self.norm1.forward(x)?, cos, sin)? + x)?;
        let x2 = (&self.mlp.forward(&self.norm2.forward(&x1)?)? + &x1)?;
        Ok(x2)
    }
}

// ─── Patch Merger ────────────────────────────────────────────────────────────

/// `proj` (no bias) → `LayerNorm` → `GELU` → `gate_up` (no bias) → `SiluAndMul` → `down` (no bias).
///
/// `context_dim = intermediate_size` for GLM4-1V.
struct Glm4vPatchMerger {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_up_proj: Linear,
    down_proj: Linear,
    context_dim: usize,
}

impl Glm4vPatchMerger {
    fn new(out_hidden_size: usize, context_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(out_hidden_size, out_hidden_size, vb.pp("proj"))?;
        let post_projection_norm =
            layer_norm(out_hidden_size, 1e-6, vb.pp("post_projection_norm"))?;
        let gate_up_proj = linear_no_bias(out_hidden_size, 2 * context_dim, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(context_dim, out_hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            proj,
            post_projection_norm,
            gate_up_proj,
            down_proj,
            context_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?;
        let x = self.post_projection_norm.forward(&x)?.gelu_erf()?;
        let gate_up = self.gate_up_proj.forward(&x)?;
        let gate = candle_nn::ops::silu(&gate_up.narrow(1, 0, self.context_dim)?)?;
        let up = gate_up.narrow(1, self.context_dim, self.context_dim)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Vision Transformer ──────────────────────────────────────────────────────

pub(crate) struct Glm4vVisionTransformer {
    patch_embed: Linear,
    post_conv_layernorm: RmsNorm,
    embeddings: Glm4vVisionEmbeddings,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<Glm4vVisionBlock>,
    post_layernorm: RmsNorm,
    downsample: Conv2d,
    merger: Glm4vPatchMerger,
    spatial_merge_size: usize,
    out_hidden_size: usize,
    device: Device,
}

impl Glm4vVisionTransformer {
    fn new(cfg: &Glm4vVisionConfig, vb: VarBuilder) -> Result<Self> {
        // Patch embed: equivalent to Conv3D(C, T, P, P) → Linear(C*T*P², hidden) with bias.
        let patch_embed = linear_b(
            cfg.patch_flat_size(),
            cfg.hidden_size,
            true,
            vb.pp("patch_embed").pp("proj"),
        )?;

        let post_conv_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_conv_layernorm"),
        )?;

        let embeddings = Glm4vVisionEmbeddings::new(cfg, vb.pp("embeddings"))?;

        // partial_rotary_factor=0.5, is_neox_style=true — matches Python get_rope() call.
        let head_dim = cfg.head_dim();
        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            8192,
            10000.0,
            0.5,
            true,
            vb.dtype(),
            vb.device(),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Glm4vVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let post_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_layernorm"))?;

        let sm = cfg.spatial_merge_size;
        let downsample = conv2d(
            cfg.hidden_size,
            cfg.out_hidden_size,
            sm,
            Conv2dConfig {
                padding: 0,
                stride: sm,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("downsample"),
        )?;

        let merger =
            Glm4vPatchMerger::new(cfg.out_hidden_size, cfg.intermediate_size, vb.pp("merger"))?;

        Ok(Self {
            patch_embed,
            post_conv_layernorm,
            embeddings,
            rotary_emb,
            blocks,
            post_layernorm,
            downsample,
            merger,
            spatial_merge_size: sm,
            out_hidden_size: cfg.out_hidden_size,
            device: vb.device().clone(),
        })
    }

    /// Compute 2D block-tiled RoPE and return position id arrays for embeddings.
    ///
    /// Returns `(cos, sin, h_ids, w_ids)` where cos/sin are `[total_np, rotary_dim]`
    /// and h_ids/w_ids are the integer grid coordinates per patch.
    fn rot_pos_emb(&self, grid_thw: &[[usize; 3]]) -> Result<(Tensor, Tensor, Vec<u32>, Vec<u32>)> {
        let sm = self.spatial_merge_size;
        let cos_table = self.rotary_emb.cos(); // [max_pos, rotary_dim/2]
        let sin_table = self.rotary_emb.sin();

        let mut cos_list: Vec<Tensor> = Vec::new();
        let mut sin_list: Vec<Tensor> = Vec::new();
        let mut all_h_ids: Vec<u32> = Vec::new();
        let mut all_w_ids: Vec<u32> = Vec::new();

        for &[t, h, w] in grid_thw {
            let np = h * w;

            // Block-tiled order: (h/sm, w/sm, sm, sm) — matches Python
            //   hpos_ids.reshape(h//sm, sm, w//sm, sm).permute(0,2,1,3).flatten()
            let mut h_ids: Vec<u32> = Vec::with_capacity(np);
            let mut w_ids: Vec<u32> = Vec::with_capacity(np);
            for ha in 0..(h / sm) {
                for wa in 0..(w / sm) {
                    for hb in 0..sm {
                        for wb in 0..sm {
                            h_ids.push((ha * sm + hb) as u32);
                            w_ids.push((wa * sm + wb) as u32);
                        }
                    }
                }
            }

            let h_pos = Tensor::from_vec(h_ids.clone(), (np,), &self.device)?;
            let w_pos = Tensor::from_vec(w_ids.clone(), (np,), &self.device)?;

            let cos_h = cos_table.index_select(&h_pos, 0)?;
            let cos_w = cos_table.index_select(&w_pos, 0)?;
            let sin_h = sin_table.index_select(&h_pos, 0)?;
            let sin_w = sin_table.index_select(&w_pos, 0)?;

            let cos_img = Tensor::cat(&[&cos_h, &cos_w], 1)?;
            let sin_img = Tensor::cat(&[&sin_h, &sin_w], 1)?;

            for _ in 0..t {
                cos_list.push(cos_img.clone());
                sin_list.push(sin_img.clone());
                all_h_ids.extend_from_slice(&h_ids);
                all_w_ids.extend_from_slice(&w_ids);
            }
        }

        let cos = Tensor::cat(&cos_list, 0)?;
        let sin = Tensor::cat(&sin_list, 0)?;
        Ok((cos, sin, all_h_ids, all_w_ids))
    }

    /// Encode patches into language-model–space features.
    ///
    /// - `pixel_values`: `[total_np, C*T*P*P]`
    /// - `grid_thw`: `[[t, h, w]]` per image
    ///
    /// Returns `[merged_tokens, out_hidden_size]`.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(pixel_values)?; // [np, hidden]
        x = self.post_conv_layernorm.forward(&x)?;

        let (cos, sin, h_ids, w_ids) = self.rot_pos_emb(grid_thw)?;

        x = self.embeddings.forward(&x, grid_thw, &h_ids, &w_ids)?;

        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }

        let x = self.post_layernorm.forward(&x)?; // [np, hidden]

        let np = x.dim(0)?;
        let hidden = x.dim(1)?;
        let sm = self.spatial_merge_size;
        let merged = np / (sm * sm);

        let x = x.reshape((merged, sm, sm, hidden))?;
        let x = x.permute((0, 3, 1, 2))?; // [merged, hidden, sm, sm]
        let x = self.downsample.forward(&x)?;
        let x = x.reshape((merged, self.out_hidden_size))?;

        self.merger.forward(&x)
    }
}

// ─── merge_multimodal ────────────────────────────────────────────────────────

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

// ─── Main Model ──────────────────────────────────────────────────────────────

/// GLM-4.1V: vision-language model combining GLM-4.1V ViT with Glm4ForCausalLM.
pub struct Glm4vForConditionalGeneration {
    vision_model: Glm4vVisionTransformer,
    language_model: Glm4ForCausalLM,
    device: Device,
}

impl Glm4vForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_cfg = Glm4vVisionConfig::from_model_config(cfg);
        let device = vb.device().clone();
        let vision_model = Glm4vVisionTransformer::new(&vis_cfg, vb.pp("visual"))?;
        let language_model = Glm4ForCausalLM::new(cfg, vb)?;
        Ok(Self {
            vision_model,
            language_model,
            device,
        })
    }

    /// Encode pixel values through the vision transformer.
    ///
    /// Returns `[merged_tokens, out_hidden_size]`.
    pub fn encode_images(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Tensor> {
        self.vision_model.forward(pixel_values, grid_thw)
    }
}

impl crate::engine::ModelForward for Glm4vForConditionalGeneration {
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
            merge_multimodal(&text_embeddings, mm, &self.device)?
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use serde_json::json;

    /// Tiny config:
    ///   ViT: hidden=16, out_hidden=32, 1 block, 2 heads, sm=2, image_size=4, patch_size=2
    ///   → orig_emb_size=2, num_positions=4
    ///   → patch_flat = 1*1*4 = 4 (in_channels=1, temporal_patch_size=1)
    ///   → 2×2 grid (t=1) → 4 patches → 1 merged token
    ///   LLM: hidden=32, 2 layers; out_hidden must equal LLM hidden_size.
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 16,
                "out_hidden_size": 32,
                "num_heads": 2,
                "depth": 1,
                "intermediate_size": 16,
                "in_channels": 1,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "image_size": 4,
                "rms_norm_eps": 1e-5
            }),
        );
        ModelConfig {
            architectures: vec!["Glm4vForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 64,
            max_position_embeddings: 256,
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
    fn test_glm4v_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4vForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
    }

    #[test]
    fn test_glm4v_vision_only() {
        // 4 patches (2×2 grid, t=1), cps = 1*1*2*2 = 4
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4vForConditionalGeneration::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let features = model.vision_model.forward(&pixel_values, &[[1, 2, 2]]);
        assert!(
            features.is_ok(),
            "vision forward failed: {:?}",
            features.err()
        );
        // 4 patches / sm² = 4/4 = 1 merged token
        assert_eq!(
            features.unwrap().dims(),
            &[1, 32],
            "expected [1, out_hidden=32]"
        );
    }

    #[test]
    fn test_glm4v_pos_embeddings() {
        // Test that position embedding forward works and adds to patches.
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vis_cfg = Glm4vVisionConfig::from_model_config(&cfg);
        let emb = Glm4vVisionEmbeddings::new(&vis_cfg, vb.pp("embeddings")).unwrap();

        let patches = Tensor::zeros((4usize, 16usize), DType::F32, &device).unwrap();
        let h_ids = vec![0u32, 0, 1, 1];
        let w_ids = vec![0u32, 1, 0, 1];
        let result = emb.forward(&patches, &[[1, 2, 2]], &h_ids, &w_ids);
        assert!(result.is_ok(), "pos_emb forward failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[4, 16]);
    }

    #[test]
    fn test_glm4v_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4vForConditionalGeneration::new(&cfg, vb).unwrap();

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
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
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_glm4v_with_image() {
        use crate::multimodal::{MultimodalInputs, ProcessedImage};

        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4vForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode 1 image: 2×2 grid → 4 patches → 1 merged token.
        let pixel_values = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let image_feats = model.encode_images(&pixel_values, &[[1, 2, 2]]).unwrap(); // [1, 32]

        let processed = ProcessedImage::new(image_feats, 1);
        let mm = MultimodalInputs::with_images(vec![0, 5, 0, 0], vec![(1, processed)]);

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
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
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.vocab_size]);
    }
}
