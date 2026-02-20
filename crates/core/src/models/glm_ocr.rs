//! GLM-OCR vision-language model.
//!
//! GlmOcrForConditionalGeneration combines a GLM-OCR vision transformer with
//! Glm4ForCausalLM as the language backbone.
//!
//! # Architecture
//!
//! ```text
//! pixel_values [np, C*T*P*P]
//!   → Linear patch embed [np, hidden]
//!   → 2D spatial RoPE (partial_rotary_factor=0.5, neox-style, sm-block-tiled h/w)
//!   → depth × GlmOcrVisionBlock
//!       norm1(RMSNorm) + attn(qkv+bias, q_norm, k_norm, partial RoPE, SDPA) + residual
//!       norm2(RMSNorm) + mlp(SwiGLU with bias) + residual
//!   → post_layernorm (RMSNorm)
//!   → reshape [np/sm², sm, sm, hidden] → permute → [np/sm², hidden, sm, sm]
//!   → Conv2d downsample (hidden→out_hidden, kernel=sm, stride=sm)
//!   → GlmOcrPatchMerger (proj → LayerNorm → GELU → SwiGLU)
//!   → merge into LLM embeddings at image-token positions
//!   → Glm4ForCausalLM → logits
//! ```
//!
//! # Reference
//! `reference/vllm/vllm/model_executor/models/glm_ocr.py`
//! `reference/vllm/vllm/model_executor/models/glm4_1v.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, layer_norm, linear_b, linear_no_bias, ops::softmax_last_dim, Conv2d, Conv2dConfig,
    LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::glm4::Glm4ForCausalLM;

// ─── Vision Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct GlmOcrVisionConfig {
    hidden_size: usize,
    out_hidden_size: usize,
    num_heads: usize,
    depth: usize,
    intermediate_size: usize,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    rms_norm_eps: f64,
}

impl GlmOcrVisionConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn patch_flat_size(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// context_dim for the PatchMerger = out_hidden * in_channels (from Python: GlmOcrPatchMerger).
    fn merger_context_dim(&self) -> usize {
        self.out_hidden_size * self.in_channels
    }
}

impl Default for GlmOcrVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1792,
            out_hidden_size: 2048,
            num_heads: 16,
            depth: 32,
            intermediate_size: 8960,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            rms_norm_eps: 1e-5,
        }
    }
}

impl GlmOcrVisionConfig {
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
/// Returns `x` with the first `rotary_dim` dims rotated in-place (via reshape).
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

    // neox rotate_half: [-x2, x1] where x_rot = [x1, x2].
    let x1 = x_rot.narrow(2, 0, half_rot)?;
    let x2 = x_rot.narrow(2, half_rot, half_rot)?;
    let neg_x2 = x2.neg()?;
    let x_rotated = Tensor::cat(&[&neg_x2, &x1], 2)?;

    // Broadcast cos/sin from [S, rotary_dim] to [S, 1, rotary_dim].
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let x_rot_out = (x_rot.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&x_rot_out, &x_pass], 2)
}

// ─── Vision Attention ────────────────────────────────────────────────────────

struct GlmOcrVisionAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl GlmOcrVisionAttention {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let qkv = linear_b(d, 3 * d, true, vb.pp("qkv"))?;
        let proj = linear_b(d, d, true, vb.pp("proj"))?;
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    /// Forward dense self-attention.
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

        // Per-head RMSNorm on q and k.
        let q = self
            .q_norm
            .forward(&q.reshape((np * h, d))?)?
            .reshape((np, h, d))?;
        let k = self
            .k_norm
            .forward(&k.reshape((np * h, d))?)?
            .reshape((np, h, d))?;

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

struct GlmOcrVisionMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl GlmOcrVisionMLP {
    fn new(dim: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // bias=True for GlmOcr (overridden from Glm4v default of False).
        let gate_up_proj = linear_b(dim, 2 * intermediate_size, true, vb.pp("gate_up_proj"))?;
        let down_proj = linear_b(intermediate_size, dim, true, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate_up.narrow(1, 0, self.intermediate_size)?)?;
        let up = gate_up.narrow(1, self.intermediate_size, self.intermediate_size)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Vision Block ────────────────────────────────────────────────────────────

struct GlmOcrVisionBlock {
    norm1: RmsNorm,
    attn: GlmOcrVisionAttention,
    norm2: RmsNorm,
    mlp: GlmOcrVisionMLP,
}

impl GlmOcrVisionBlock {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm1"))?;
        let attn = GlmOcrVisionAttention::new(cfg, vb.pp("attn"))?;
        let norm2 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm2"))?;
        let mlp = GlmOcrVisionMLP::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    /// Pre-norm transformer block with residual connections.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x1 = (&self.attn.forward(&self.norm1.forward(x)?, cos, sin)? + x)?;
        let x2 = (&self.mlp.forward(&self.norm2.forward(&x1)?)? + &x1)?;
        Ok(x2)
    }
}

// ─── Patch Merger ────────────────────────────────────────────────────────────

struct GlmOcrPatchMerger {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_up_proj: Linear,
    down_proj: Linear,
    context_dim: usize,
}

impl GlmOcrPatchMerger {
    /// `out_hidden_size` = d_model; `context_dim` = out_hidden * in_channels.
    fn new(out_hidden_size: usize, context_dim: usize, vb: VarBuilder) -> Result<Self> {
        // bias=False for GlmOcr merger.
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
        // GELU after LayerNorm.
        let x = self.post_projection_norm.forward(&x)?.gelu_erf()?;
        // SwiGLU (SiluAndMul).
        let gate_up = self.gate_up_proj.forward(&x)?;
        let gate = candle_nn::ops::silu(&gate_up.narrow(1, 0, self.context_dim)?)?;
        let up = gate_up.narrow(1, self.context_dim, self.context_dim)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Vision Transformer ──────────────────────────────────────────────────────

pub(crate) struct GlmOcrVisionTransformer {
    patch_embed: Linear,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<GlmOcrVisionBlock>,
    post_layernorm: RmsNorm,
    downsample: Conv2d,
    merger: GlmOcrPatchMerger,
    spatial_merge_size: usize,
    out_hidden_size: usize,
    device: Device,
}

impl GlmOcrVisionTransformer {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilder) -> Result<Self> {
        // Patch embed: equivalent to Conv3D(C, T, P, P) → Linear(C*T*P², hidden) with bias.
        let patch_embed = linear_b(
            cfg.patch_flat_size(),
            cfg.hidden_size,
            true,
            vb.pp("patch_embed").pp("proj"),
        )?;

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
            blocks.push(GlmOcrVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
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

        let merger = GlmOcrPatchMerger::new(
            cfg.out_hidden_size,
            cfg.merger_context_dim(),
            vb.pp("merger"),
        )?;

        Ok(Self {
            patch_embed,
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

    /// Compute 2D block-tiled position embeddings.
    ///
    /// Patches are ordered in spatial_merge_size×sm blocks so that the Conv2d
    /// downsample sees each sm×sm block as a contiguous group.
    ///
    /// Returns `(cos, sin)` of shape `[total_np, rotary_dim]`.
    fn rot_pos_emb(&self, grid_thw: &[[usize; 3]]) -> Result<(Tensor, Tensor)> {
        let sm = self.spatial_merge_size;
        let cos_table = self.rotary_emb.cos(); // [max_pos, rotary_dim/2]
        let sin_table = self.rotary_emb.sin();

        let mut cos_list: Vec<Tensor> = Vec::new();
        let mut sin_list: Vec<Tensor> = Vec::new();

        for &[t, h, w] in grid_thw {
            let np = h * w;

            // Build (h_ids, w_ids) in block-tiled order.
            // For each sm×sm block (ha, wa) iterate over inner offsets (hb, wb).
            // This matches the Python reshape-permute pattern:
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

            let h_pos = Tensor::from_vec(h_ids, (np,), &self.device)?;
            let w_pos = Tensor::from_vec(w_ids, (np,), &self.device)?;

            // Index and concatenate h and w position embeddings.
            let cos_h = cos_table.index_select(&h_pos, 0)?;
            let cos_w = cos_table.index_select(&w_pos, 0)?;
            let sin_h = sin_table.index_select(&h_pos, 0)?;
            let sin_w = sin_table.index_select(&w_pos, 0)?;

            let cos_img = Tensor::cat(&[&cos_h, &cos_w], 1)?; // [np, rotary_dim]
            let sin_img = Tensor::cat(&[&sin_h, &sin_w], 1)?;

            for _ in 0..t {
                cos_list.push(cos_img.clone());
                sin_list.push(sin_img.clone());
            }
        }

        let cos = Tensor::cat(&cos_list, 0)?;
        let sin = Tensor::cat(&sin_list, 0)?;
        Ok((cos, sin))
    }

    /// Encode patches into language-model–space features.
    ///
    /// - `pixel_values`: `[total_np, C*T*P*P]`
    /// - `grid_thw`: `[[t, h, w]]` per image
    ///
    /// Returns `[merged_tokens, out_hidden_size]`.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(pixel_values)?; // [np, hidden]
        let (cos, sin) = self.rot_pos_emb(grid_thw)?;

        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }

        let x = self.post_layernorm.forward(&x)?; // [np, hidden]

        let np = x.dim(0)?;
        let hidden = x.dim(1)?;
        let sm = self.spatial_merge_size;
        let merged = np / (sm * sm);

        // Reshape for Conv2d: [np, hidden] → [merged, sm, sm, hidden]
        //   → permute → [merged, hidden, sm, sm].
        let x = x.reshape((merged, sm, sm, hidden))?;
        let x = x.permute((0, 3, 1, 2))?;

        // Downsample: [merged, hidden, sm, sm] → [merged, out_hidden, 1, 1].
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

/// GLM-OCR: vision-language model combining GLM-OCR ViT with Glm4ForCausalLM.
pub struct GlmOcrForConditionalGeneration {
    vision_model: GlmOcrVisionTransformer,
    language_model: Glm4ForCausalLM,
    device: Device,
}

impl GlmOcrForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_cfg = GlmOcrVisionConfig::from_model_config(cfg);
        let device = vb.device().clone();
        let vision_model = GlmOcrVisionTransformer::new(&vis_cfg, vb.pp("visual"))?;
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

impl crate::engine::ModelForward for GlmOcrForConditionalGeneration {
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

    /// Tiny config: ViT embed=16, 1 block, 2 heads, sm=2; LLM hidden=32, 2 layers.
    ///
    /// out_hidden_size must equal LLM hidden_size (32) so vision features merge correctly.
    /// patch_size=2, in_channels=1, temporal_patch_size=1 → cps = 1*1*4 = 4.
    /// 2×2 grid → 4 patches; sm=2 → 4/4 = 1 merged token.
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
                "rms_norm_eps": 1e-5
            }),
        );
        ModelConfig {
            architectures: vec!["GlmOcrForConditionalGeneration".to_string()],
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
    fn test_glm_ocr_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GlmOcrForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
    }

    #[test]
    fn test_glm_ocr_vision_only() {
        // 4 patches (2×2 grid, t=1), cps = 1*1*2*2 = 4
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GlmOcrForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_glm_ocr_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GlmOcrForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_glm_ocr_encode_images() {
        // 4×4 grid, t=1, cps=4 → 16 patches → 4 merged tokens.
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GlmOcrForConditionalGeneration::new(&cfg, vb).unwrap();

        let pixel_values = Tensor::zeros((16usize, 4usize), DType::F32, &device).unwrap();
        let features = model.encode_images(&pixel_values, &[[1, 4, 4]]);
        assert!(
            features.is_ok(),
            "encode_images failed: {:?}",
            features.err()
        );
        assert_eq!(
            features.unwrap().dims(),
            &[4, 32],
            "expected [4, out_hidden=32]"
        );
    }

    #[test]
    fn test_glm_ocr_with_image() {
        use crate::multimodal::{MultimodalInputs, ProcessedImage};

        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GlmOcrForConditionalGeneration::new(&cfg, vb).unwrap();

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
