//! Legacy Qwen-VL vision-language model.
//!
//! `QwenVLForConditionalGeneration` wraps a CLIP-style visual encoder and the
//! original Qwen-v1 language model.
//!
//! # Architecture
//!
//! 1. **VisionTransformer** (CLIP-style):
//!    - `conv1`: Conv2d patch embed (no bias), stride = patch_size
//!    - `positional_embedding`: learnable `[n_queries, width]`; bicubic-
//!      interpolated at runtime to match actual patch count
//!    - `ln_pre`: LayerNorm
//!    - `transformer`: N × VisualAttentionBlock (`ln_1 + attn + ln_2 + mlp`)
//!    - `attn_pool`: Resampler2 cross-attention (no post-projection) that
//!      compresses variable-length patch tokens to `n_queries`
//!    - `ln_post`: LayerNorm on the resampled output
//!    - `proj`: learnable `[output_dim, output_dim]` right-multiply
//!
//! 2. **Language model**: `QWenLMHeadModel` (Qwen-v1) from `qwen.rs`
//!
//! Weight paths:
//! - Vision: `transformer.visual.*`
//! - LLM:    `transformer.{wte,h,ln_f}.*`, `lm_head.*`
//!
//! Reference: https://huggingface.co/Qwen/Qwen-VL

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, layer_norm, linear, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen::QWenLMHeadModel;
use super::tp_layers::TpContext;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Vision config extracted from `ModelConfig::extra["visual"]`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QwenVLVisionConfig {
    image_size: usize,
    patch_size: usize,
    width: usize,
    layers: usize,
    heads: usize,
    mlp_ratio: f64,
    n_queries: usize,
    output_dim: usize,
    /// Number of attention heads in the attn_pool.
    /// Defaults to `output_dim / 128` (Qwen-VL convention: 4096/128=32).
    pool_heads: usize,
}

impl QwenVLVisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let v = cfg
            .extra
            .get("visual")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let get_usize = |key: &str, default: usize| -> usize {
            v.get(key)
                .and_then(|x| x.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
        };

        let output_dim = get_usize("output_dim", cfg.hidden_size);
        let pool_heads = get_usize("pool_heads", (output_dim / 128).max(1));
        Self {
            image_size: get_usize("image_size", 448),
            patch_size: get_usize("patch_size", 14),
            width: get_usize("width", 1024),
            layers: get_usize("layers", 48),
            heads: get_usize("heads", 16),
            mlp_ratio: get_f64("mlp_ratio", 4.75),
            n_queries: get_usize("n_queries", 256),
            output_dim,
            pool_heads,
        }
    }

    fn mlp_width(&self) -> usize {
        (self.width as f64 * self.mlp_ratio).round() as usize
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// Self-attention for visual transformer blocks.
///
/// Uses a single fused `in_proj` (Q+K+V) and a separate `out_proj`.
/// Operates in (seq, batch, dim) layout (LND).
struct QwenVLAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl QwenVLAttention {
    fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let in_proj = linear(embed_dim, 3 * embed_dim, vb.pp("in_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// x: `(seq, batch, dim)` → `(seq, batch, dim)`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, batch, dim) = x.dims3()?;

        // Fused QKV: (seq*batch, 3*dim)
        let mixed = self
            .in_proj
            .forward(&x.reshape((seq * batch, dim))?)?
            .reshape((seq, batch, 3 * dim))?;

        let q = mixed.narrow(2, 0, dim)?.contiguous()?;
        let k = mixed.narrow(2, dim, dim)?.contiguous()?;
        let v = mixed.narrow(2, 2 * dim, dim)?.contiguous()?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = k
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = v
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let attn = (q * self.scale)?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // (batch, heads, seq, head_dim)

        // Back to (seq, batch, dim)
        let out = out
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .reshape((seq, batch, dim))?;

        self.out_proj
            .forward(&out.reshape((seq * batch, dim))?)?
            .reshape((seq, batch, dim))
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct QwenVLMlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl QwenVLMlp {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_fc: linear(hidden, intermediate, vb.pp("c_fc"))?,
            c_proj: linear(intermediate, hidden, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(x)?.gelu_erf()?)
    }
}

// ─── Attention Block ─────────────────────────────────────────────────────────

/// Pre-norm residual block: `x + attn(ln_1(x)) + mlp(ln_2(x))`.
struct QwenVLAttentionBlock {
    ln_1: LayerNorm,
    attn: QwenVLAttention,
    ln_2: LayerNorm,
    mlp: QwenVLMlp,
}

impl QwenVLAttentionBlock {
    fn new(width: usize, heads: usize, mlp_width: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_1: layer_norm(width, 1e-6, vb.pp("ln_1"))?,
            attn: QwenVLAttention::new(width, heads, vb.pp("attn"))?,
            ln_2: layer_norm(width, 1e-6, vb.pp("ln_2"))?,
            mlp: QwenVLMlp::new(width, mlp_width, vb.pp("mlp"))?,
        })
    }

    /// x: `(seq, batch, width)` → `(seq, batch, width)`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln_1.forward(x)?)?)?;
        let x = (&x + self.mlp.forward(&self.ln_2.forward(&x)?)?)?;
        Ok(x)
    }
}

// ─── Transformer Encoder ─────────────────────────────────────────────────────

struct QwenVLTransformerEncoder {
    resblocks: Vec<QwenVLAttentionBlock>,
}

impl QwenVLTransformerEncoder {
    fn new(
        width: usize,
        layers: usize,
        heads: usize,
        mlp_width: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_rb = vb.pp("resblocks");
        let mut resblocks = Vec::with_capacity(layers);
        for i in 0..layers {
            resblocks.push(QwenVLAttentionBlock::new(
                width,
                heads,
                mlp_width,
                vb_rb.pp(i),
            )?);
        }
        Ok(Self { resblocks })
    }

    /// x: `(seq, batch, width)` → `(seq, batch, width)`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.resblocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

// ─── Attention Pool (Resampler2, no post-projection) ─────────────────────────

/// Cross-attention resampler: compresses patch tokens to `n_queries`.
///
/// Implements Resampler2 with `do_post_projection=False`:
/// - Q = `ln_q(query) + pos_embed` (learnable queries, 16×16 sincos pos)
/// - K = `ln_kv(kv_proj(x)) + get_abs_pos(pos_embed, tgt_size)` (interpolated)
/// - V = `ln_kv(kv_proj(x))`  (no pos embed on V)
struct QwenVLAttnPool {
    query: Tensor,
    kv_proj: Linear,
    ln_q: LayerNorm,
    ln_kv: LayerNorm,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: Linear,
    pos_embed: Tensor,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    n_queries: usize,
}

impl QwenVLAttnPool {
    fn new(
        output_dim: usize,
        kv_dim: usize,
        n_queries: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = output_dim / num_heads;

        let query = vb.get((n_queries, output_dim), "query")?;
        let kv_proj = linear(kv_dim, output_dim, vb.pp("kv_proj"))?;
        let ln_q = layer_norm(output_dim, 1e-6, vb.pp("ln_q"))?;
        let ln_kv = layer_norm(output_dim, 1e-6, vb.pp("ln_kv"))?;

        let vb_attn = vb.pp("attn");
        let in_proj_weight = vb_attn.get((3 * output_dim, output_dim), "in_proj_weight")?;
        let in_proj_bias = vb_attn.get(3 * output_dim, "in_proj_bias")?;
        let out_proj = linear(output_dim, output_dim, vb_attn.pp("out_proj"))?;

        let pos_embed = vb.get((n_queries, output_dim), "pos_embed")?;

        Ok(Self {
            query,
            kv_proj,
            ln_q,
            ln_kv,
            in_proj_weight,
            in_proj_bias,
            out_proj,
            pos_embed,
            num_heads,
            head_dim,
            embed_dim: output_dim,
            n_queries,
        })
    }

    /// Cross-attention: Q `(nq, B, D)`, K `(L, B, D)`, V `(L, B, D)` → `(nq, B, D)`.
    fn cross_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (q_len, batch, _) = q.dims3()?;
        let (kv_len, _, _) = k.dims3()?;

        let project = |x: &Tensor, offset: usize, len: usize, seq: usize| -> Result<Tensor> {
            let w = self.in_proj_weight.narrow(0, offset, len)?;
            let b = self.in_proj_bias.narrow(0, offset, len)?;
            x.reshape((seq * batch, len))?
                .matmul(&w.t()?)?
                .broadcast_add(&b)?
                .reshape((seq, batch, len))
        };

        let q = project(q, 0, self.embed_dim, q_len)?;
        let k = project(k, self.embed_dim, self.embed_dim, kv_len)?;
        let v = project(v, 2 * self.embed_dim, self.embed_dim, kv_len)?;

        // (seq, batch, dim) → (batch, heads, seq, head_dim)
        let reshape_for_mha = |t: &Tensor, seq: usize| -> Result<Tensor> {
            t.reshape((seq, batch, self.num_heads, self.head_dim))?
                .permute((1, 2, 0, 3))
        };
        let q = reshape_for_mha(&q, q_len)?;
        let k = reshape_for_mha(&k, kv_len)?;
        let v = reshape_for_mha(&v, kv_len)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q * scale)?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?; // (batch, heads, q_len, head_dim)

        // → (q_len, batch, embed_dim)
        let out = out
            .permute((2, 0, 1, 3))?
            .reshape((q_len, batch, self.embed_dim))?;

        self.out_proj
            .forward(&out.reshape((q_len * batch, self.embed_dim))?)?
            .reshape((q_len, batch, self.embed_dim))
    }

    /// Resample `x: (B, L, kv_dim)` → `(B, n_queries, output_dim)`.
    ///
    /// `patch_grid` is the sqrt of the number of input patches (e.g. 32 for
    /// a 448×448 image with 14×14 patches).  The stored `pos_embed` covers
    /// a 16×16 grid and is bicubic-interpolated to `patch_grid×patch_grid`
    /// before being added to the keys.
    fn forward(&self, x: &Tensor, patch_grid: usize) -> Result<Tensor> {
        let (batch, _l, _) = x.dims3()?;

        // Project and normalize KV.
        let x = self.kv_proj.forward(x)?; // (B, L, D)
        let x = self.ln_kv.forward(&x)?;
        let x = x.transpose(0, 1)?; // (L, B, D)

        // Build adaptive pos embed for keys via nearest-neighbor interpolation.
        // pos_embed is (n_queries, D) representing a sqrt(n_queries)×sqrt(n_queries) grid.
        let q_grid = (self.n_queries as f64).sqrt() as usize;
        let kv_pos = if patch_grid == q_grid {
            self.pos_embed.clone()
        } else {
            // NOTE: Uses nearest-neighbor; production uses bicubic.
            let hidden = self.pos_embed.dim(1)?;
            self.pos_embed
                .reshape((1, q_grid, q_grid, hidden))?
                .permute((0, 3, 1, 2))?
                .contiguous()?
                .upsample_nearest2d(patch_grid, patch_grid)?
                .permute((0, 2, 3, 1))?
                .reshape((patch_grid * patch_grid, hidden))?
        };

        // Add pos embed to K.
        let kv_pos_expanded = kv_pos.unsqueeze(1)?; // (L, 1, D)
        let x_with_pos = x.broadcast_add(&kv_pos_expanded)?; // (L, B, D)

        // Prepare Q: (n_queries, D) → (n_queries, B, D).
        let q = self.ln_q.forward(&self.query)?;
        let q_pos = self.pos_embed.unsqueeze(1)?; // (n_queries, 1, D)
        let q = q
            .unsqueeze(1)?
            .broadcast_as((self.n_queries, batch, self.embed_dim))?;
        let q = q.broadcast_add(&q_pos)?; // (n_queries, B, D)

        // Cross-attention: Q attends to K=x+pos, V=x (no pos on V).
        let out = self.cross_attention(
            &q.contiguous()?,
            &x_with_pos.contiguous()?,
            &x.contiguous()?,
        )?;

        // (n_queries, B, D) → (B, n_queries, D)
        out.transpose(0, 1)
    }
}

// ─── Vision Transformer ──────────────────────────────────────────────────────

/// CLIP-style visual encoder used by Qwen-VL.
///
/// Produces `[batch, n_queries, output_dim]` from `[batch, 3, H, W]` pixels.
struct QwenVLVisionTransformer {
    conv1: candle_nn::Conv2d,
    positional_embedding: Tensor,
    ln_pre: LayerNorm,
    transformer: QwenVLTransformerEncoder,
    attn_pool: QwenVLAttnPool,
    ln_post: LayerNorm,
    proj: Tensor,
}

impl QwenVLVisionTransformer {
    fn new(vision_cfg: &QwenVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let cfg = vision_cfg.clone();
        let n_queries = cfg.n_queries;
        let width = cfg.width;
        let mlp_width = cfg.mlp_width();

        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let conv1 = conv2d(3, width, cfg.patch_size, conv_cfg, vb.pp("conv1"))?;

        // Learnable positional embedding: [n_queries, width].
        let positional_embedding = vb.get((n_queries, width), "positional_embedding")?;

        let ln_pre = layer_norm(width, 1e-6, vb.pp("ln_pre"))?;

        let transformer = QwenVLTransformerEncoder::new(
            width,
            cfg.layers,
            cfg.heads,
            mlp_width,
            vb.pp("transformer"),
        )?;

        let attn_pool = QwenVLAttnPool::new(
            cfg.output_dim,
            width,
            n_queries,
            cfg.pool_heads,
            vb.pp("attn_pool"),
        )?;

        let ln_post = layer_norm(cfg.output_dim, 1e-6, vb.pp("ln_post"))?;

        let proj = vb.get((cfg.output_dim, cfg.output_dim), "proj")?;

        Ok(Self {
            conv1,
            positional_embedding,
            ln_pre,
            transformer,
            attn_pool,
            ln_post,
            proj,
        })
    }

    /// pixel_values: `(B, 3, H, W)` → visual features `(B, n_queries, output_dim)`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Patch embed: (B, width, grid, grid)
        let x = self.conv1.forward(pixel_values)?;
        let (_batch, _width, grid_h, grid_w) = x.dims4()?;
        let n_patches = grid_h * grid_w;

        // (B, width, grid, grid) → (B, n_patches, width)
        let x = x.flatten(2, 3)?.permute((0, 2, 1))?;

        // Add interpolated positional embeddings.
        let pos = get_abs_pos(&self.positional_embedding, grid_h)?;
        let x = x.broadcast_add(&pos.unsqueeze(0)?)?;

        let x = self.ln_pre.forward(&x)?;

        // LND layout for the transformer
        let x = x.permute((1, 0, 2))?; // (n_patches, B, width)
        let x = self.transformer.forward(&x)?;
        let x = x.permute((1, 0, 2))?; // (B, n_patches, width)

        // Attention-pool to (B, n_queries, output_dim)
        let patch_grid = (n_patches as f64).sqrt() as usize;
        let x = self.attn_pool.forward(&x, patch_grid)?;

        let x = self.ln_post.forward(&x)?;

        // Right-multiply by proj: (B, n_queries, output_dim) @ (output_dim, output_dim)
        let (b, nq, d) = x.dims3()?;
        x.reshape((b * nq, d))?
            .matmul(&self.proj)?
            .reshape((b, nq, d))
    }
}

/// Interpolate positional embeddings to match the actual patch grid.
///
/// `abs_pos`: `[n_base, dim]` covering a `sqrt(n_base) × sqrt(n_base)` grid.
/// Returns `[n_target, dim]` for the target `grid_size × grid_size` layout.
///
/// NOTE: Uses nearest-neighbor interpolation; production uses bicubic.
fn get_abs_pos(abs_pos: &Tensor, grid_size: usize) -> Result<Tensor> {
    let (n, dim) = abs_pos.dims2()?;
    let src_size = (n as f64).sqrt() as usize;
    if src_size == grid_size {
        return Ok(abs_pos.clone());
    }
    abs_pos
        .reshape((1, src_size, src_size, dim))?
        .permute((0, 3, 1, 2))?
        .contiguous()?
        .upsample_nearest2d(grid_size, grid_size)?
        .permute((0, 2, 3, 1))?
        .reshape((grid_size * grid_size, dim))
}

// ─── Full Model ───────────────────────────────────────────────────────────────

/// Qwen-VL (legacy Qwen-v1 backbone + CLIP-style ViT with Resampler2).
pub struct QwenVLForConditionalGeneration {
    visual: QwenVLVisionTransformer,
    language_model: QWenLMHeadModel,
    device: Device,
    dtype: DType,
}

impl QwenVLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vision_cfg = QwenVLVisionConfig::from_model_config(cfg);
        let visual = QwenVLVisionTransformer::new(&vision_cfg, vb.pp("transformer").pp("visual"))?;
        // QWenLMHeadModel loads weights under transformer.{wte,h,ln_f} and lm_head.
        let language_model = QWenLMHeadModel::new_with_tp(
            cfg,
            vb.clone(),
            &LocalProcessGroup::new(),
            TpContext::single_gpu(),
        )?;
        Ok(Self {
            visual,
            language_model,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Encode pixel values → `(batch, n_queries, output_dim)`.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.visual.forward(pixel_values)
    }

    /// Merge visual embeddings into text embeddings at image pad positions.
    ///
    /// `image_token_id` identifies positions where visual features should be
    /// injected.  Tokens outside any image region retain their text embeddings.
    fn merge_vision_features(
        &self,
        input_ids: &Tensor,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            let img_emb = processed_image
                .embedding
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let batch_idx = position / seq_len;
            let start_pos = position % seq_len;
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

        let _ = input_ids; // kept for API symmetry with other VLMs
        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let text_embeddings = self.language_model.embed_tokens(input_ids)?;
        let embeddings = if let Some(mm) = multimodal_inputs {
            if mm.has_images() {
                self.merge_vision_features(input_ids, &text_embeddings, mm)?
            } else {
                text_embeddings
            }
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
}

// ─── Trait Implementations ────────────────────────────────────────────────────

impl crate::engine::ModelForward for QwenVLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_inner(
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
        // Decode step: single token per sequence; no vision injection needed.
        crate::engine::ModelForward::forward_decode_batch(
            &self.language_model,
            input_ids,
            sequences,
            kv_cache_mgr,
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
    use candle_core::DType;

    /// Tiny vision config: 16×16 image, 8×8 patches → 4 patches = 2×2 grid.
    /// n_queries=4 so positional_embedding is [4, width] and no interpolation
    /// is needed (grid_size=2 = sqrt(n_queries)=2).
    fn tiny_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "visual".to_string(),
            serde_json::json!({
                "image_size": 16,
                "patch_size": 8,
                "width": 16,
                "layers": 2,
                "heads": 2,
                "mlp_ratio": 2.0,
                "n_queries": 4,
                "output_dim": 32,
                // 32 / 128 = 0; override to 2 heads (head_dim=16)
                "pool_heads": 2
            }),
        );
        // QWen v1 config
        extra.insert("layer_norm_epsilon".to_string(), serde_json::json!(1e-5));
        ModelConfig {
            architectures: vec!["QwenVLForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 128,
            max_position_embeddings: 256,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    #[test]
    fn test_construction() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = QwenVLForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "QwenVLForConditionalGeneration should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_vision_encoder_shape() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = QwenVLForConditionalGeneration::new(&cfg, vb).expect("construct");

        // 16×16 image, 8×8 patch → 2×2 = 4 patches
        let pixel_values =
            Tensor::zeros((1usize, 3usize, 16usize, 16usize), DType::F32, &device).expect("pixels");
        let features = model.encode_images(&pixel_values).expect("encode_images");
        // n_queries=4, output_dim=32
        assert_eq!(
            features.dims(),
            &[1, 4, 32],
            "visual features should be [B, n_queries, output_dim]"
        );
    }

    #[test]
    fn test_vision_encoder_batch() {
        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = QwenVLForConditionalGeneration::new(&cfg, vb).expect("construct");

        let pixel_values =
            Tensor::zeros((3usize, 3usize, 16usize, 16usize), DType::F32, &device).expect("pixels");
        let features = model
            .encode_images(&pixel_values)
            .expect("encode_images batch");
        assert_eq!(features.dims(), &[3, 4, 32]);
    }

    #[test]
    fn test_text_only_prefill() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = QwenVLForConditionalGeneration::new(&cfg, vb).expect("construct");

        let seq_len = 5usize;
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv = KVCacheManager::new(&cache_config).expect("kv");
        let mut bt = BlockTable::new(16);
        kv.allocate_for_request(&mut bt, seq_len).expect("allocate");
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).expect("input_ids");
        let out = ModelForward::forward(&model, &input_ids, 0, &mut kv, &bt, &slot_mapping)
            .expect("text forward");
        assert_eq!(out.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_get_abs_pos_passthrough() {
        // When src_size == grid_size, get_abs_pos returns the tensor unchanged.
        let device = Device::Cpu;
        let pos = Tensor::zeros((4usize, 8usize), DType::F32, &device).expect("pos");
        let result = get_abs_pos(&pos, 2).expect("get_abs_pos 2x2");
        assert_eq!(result.dims(), &[4, 8]);
    }
}
