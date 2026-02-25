//! LFM2-VL vision-language model.
//!
//! Architecture:
//! - Vision encoder: Siglip2VisionTransformer (SigLIP2 ViT, no RoPE)
//!   - `Siglip2VisionEmbeddings`: Linear patch embed + bilinear-interpolated position embeds
//!   - `Siglip2Encoder`: N standard attention + MLP layers
//!   - Optional `post_layernorm`
//! - Multimodal projector: `Lfm2VLMultiModalProjector`
//!   - Pixel-shuffle downsampling (factor×factor tokens → 1 token with factor²×D features)
//!   - Optional LayerNorm, `linear_1` + GELU + `linear_2`
//! - Language model: `Lfm2ForCausalLM` (dense hybrid attention+short_conv)
//!
//! # Weight paths (HF checkpoint)
//!
//! ```text
//! model.vision_tower.vision_model.embeddings.patch_embedding.{weight,bias}
//! model.vision_tower.vision_model.embeddings.position_embedding.weight
//! model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.{weight,bias}
//! model.vision_tower.vision_model.encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.*
//! model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.{weight,bias}
//! model.vision_tower.vision_model.encoder.layers.{i}.mlp.{fc1,fc2}.*
//! model.vision_tower.vision_model.post_layernorm.{weight,bias}
//! model.multi_modal_projector.{layer_norm,linear_1,linear_2}.*
//! model.language_model.{embed_tokens,layers,embedding_norm}.*
//! lm_head.*
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/lfm2_vl.py`
//! `reference/vllm/vllm/model_executor/models/lfm2_siglip2.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::lfm2::Lfm2ForCausalLM;

// ─── Vision Config ────────────────────────────────────────────────────────────

struct Siglip2VisionConfig {
    patch_size: usize,
    num_channels: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    /// Total position embedding slots (= (image_size / patch_size)²).
    num_patches: usize,
    /// Square root of `num_patches`; base grid side for position embedding resize.
    pe_size: usize,
}

impl Siglip2VisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> candle_core::Result<Self> {
        let v = cfg
            .extra
            .get("vision_config")
            .ok_or_else(|| candle_core::Error::Msg("missing vision_config".into()))?;

        let get_usize = |key: &str, default: usize| -> usize {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
        };

        let patch_size = get_usize("patch_size", 14);
        let num_channels = get_usize("num_channels", 3);
        let hidden_size = get_usize("hidden_size", 1152);
        let num_hidden_layers = get_usize("num_hidden_layers", 27);
        let num_attention_heads = get_usize("num_attention_heads", 16);
        let intermediate_size = get_usize("intermediate_size", 4304);
        let layer_norm_eps = get_f64("layer_norm_eps", 1e-6);
        // num_patches = (image_size / patch_size)² — or fallback default 729 (27²)
        let num_patches = get_usize("num_patches", 729);
        let pe_size = (num_patches as f64).sqrt().round() as usize;

        Ok(Self {
            patch_size,
            num_channels,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            layer_norm_eps,
            num_patches,
            pe_size,
        })
    }

    fn patch_dim(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ─── VL Config ────────────────────────────────────────────────────────────────

struct Lfm2VLConfig {
    downsample_factor: usize,
    projector_hidden_size: usize,
    projector_bias: bool,
    projector_use_layernorm: bool,
    image_token_id: i64,
}

impl Lfm2VLConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let get_usize = |key: &str, default: usize| -> usize {
            cfg.extra
                .get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };
        let downsample_factor = get_usize("downsample_factor", 2);
        let projector_hidden_size = get_usize("projector_hidden_size", cfg.hidden_size);
        let projector_bias = cfg
            .extra
            .get("projector_bias")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let projector_use_layernorm = cfg
            .extra
            .get("projector_use_layernorm")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|x| x.as_i64())
            .unwrap_or(-1);
        Self {
            downsample_factor,
            projector_hidden_size,
            projector_bias,
            projector_use_layernorm,
            image_token_id,
        }
    }
}

// ─── Bilinear Position Embedding Resize ──────────────────────────────────────

/// Bilinearly resize a packed `[pe_size*pe_size, D]` position embedding grid to `[h*w, D]`.
///
/// Implements `align_corners=False` (pixel centres at 0.5, 1.5, …).  Runs on CPU via
/// `to_vec1` — acceptable because this is called once per tile per forward pass.
///
/// TODO: For GPU performance, replace with a CUDA bilinear-interpolation kernel.
fn bilinear_resize_pos_emb(
    pe: &Tensor,    // [pe_size*pe_size, D]
    pe_size: usize, // base grid side length
    h_out: usize,
    w_out: usize,
) -> Result<Tensor> {
    let d = pe.dim(1)?;

    // Fast path: no resize needed when dimensions match.
    if pe_size == h_out && pe_size == w_out {
        return pe.reshape((h_out * w_out, d));
    }

    let pe_f32 = pe.to_dtype(DType::F32)?;
    let pe_vec: Vec<f32> = pe_f32.flatten_all()?.to_vec1()?;

    let mut result = vec![0.0f32; h_out * w_out * d];

    for r_out in 0..h_out {
        for c_out in 0..w_out {
            // align_corners=False coordinate mapping
            let y = (r_out as f32 + 0.5) * (pe_size as f32 / h_out as f32) - 0.5;
            let x = (c_out as f32 + 0.5) * (pe_size as f32 / w_out as f32) - 0.5;

            let y0 = (y.floor() as isize).clamp(0, pe_size as isize - 1) as usize;
            let y1 = (y0 + 1).min(pe_size - 1);
            let x0 = (x.floor() as isize).clamp(0, pe_size as isize - 1) as usize;
            let x1 = (x0 + 1).min(pe_size - 1);

            let dy = y - y.floor();
            let dx = x - x.floor();

            let w00 = (1.0 - dy) * (1.0 - dx);
            let w01 = (1.0 - dy) * dx;
            let w10 = dy * (1.0 - dx);
            let w11 = dy * dx;

            let out_idx = (r_out * w_out + c_out) * d;
            for ch in 0..d {
                let v00 = pe_vec[(y0 * pe_size + x0) * d + ch];
                let v01 = pe_vec[(y0 * pe_size + x1) * d + ch];
                let v10 = pe_vec[(y1 * pe_size + x0) * d + ch];
                let v11 = pe_vec[(y1 * pe_size + x1) * d + ch];
                result[out_idx + ch] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
            }
        }
    }

    Tensor::new(result, pe.device())?.to_dtype(pe.dtype())
}

// ─── Vision Embeddings ────────────────────────────────────────────────────────

struct Siglip2VisionEmbeddings {
    patch_embedding: Linear,
    /// Raw position embedding table `[num_patches, hidden]`; resized per tile during forward.
    position_embedding_weight: Tensor,
    pe_size: usize,
}

impl Siglip2VisionEmbeddings {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = linear(cfg.patch_dim(), cfg.hidden_size, vb.pp("patch_embedding"))?;
        // Embedding table; we use only the `.weight` tensor directly for resizing.
        let pe_emb = embedding(
            cfg.num_patches,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        let position_embedding_weight = pe_emb.embeddings().clone();
        Ok(Self {
            patch_embedding,
            position_embedding_weight,
            pe_size: cfg.pe_size,
        })
    }

    /// `pixel_values`: `[S, patch_dim]` packed patches.
    /// `spatial_shapes`: `(h, w)` per tile in patch-grid units.
    /// Returns `[S, hidden]`.
    fn forward(&self, pixel_values: &Tensor, spatial_shapes: &[(usize, usize)]) -> Result<Tensor> {
        // Linear patch embedding.
        let patch_embeds = self.patch_embedding.forward(pixel_values)?; // [S, D]

        // Build packed positional embeddings with per-tile bilinear resize.
        let pe_weight = &self.position_embedding_weight; // [num_patches, D]
        let mut pos_parts: Vec<Tensor> = Vec::with_capacity(spatial_shapes.len());
        for &(h, w) in spatial_shapes {
            let pos = bilinear_resize_pos_emb(pe_weight, self.pe_size, h, w)?; // [h*w, D]
            pos_parts.push(pos);
        }

        let pos_embeds = if pos_parts.len() == 1 {
            pos_parts.remove(0)
        } else {
            Tensor::cat(&pos_parts, 0)?
        };

        patch_embeds + pos_embeds
    }
}

// ─── Vision MLP ───────────────────────────────────────────────────────────────

struct Siglip2MLP {
    fc1: Linear,
    fc2: Linear,
}

impl Siglip2MLP {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            // SigLIP2 uses bias=True in both layers.
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SigLIP2 `hidden_act = "gelu_pytorch_tanh"` — tanh-approximated GELU.
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

// ─── Vision Attention ─────────────────────────────────────────────────────────

struct Siglip2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Siglip2Attention {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        // HF checkpoint has separate q_proj / k_proj / v_proj (vLLM stacks into qkv_proj,
        // but we load from HF directly and keep them separate).
        let q_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?;
        let out_proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?;
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[S, D]` packed tokens (no batch dim).
    /// Returns `[S, D]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (s, _) = x.dims2()?;

        let q = self.q_proj.forward(x)?; // [S, D]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: [S, H, hd]
        let q = q.reshape((s, self.num_heads, self.head_dim))?;
        let k = k.reshape((s, self.num_heads, self.head_dim))?;
        let v = v.reshape((s, self.num_heads, self.head_dim))?;

        // [H, S, hd]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let attn = (q.matmul(&k.transpose(1, 2)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [H, S, hd]

        let out = out
            .transpose(0, 1)?
            .contiguous()?
            .reshape((s, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

// ─── Vision Encoder Layer ─────────────────────────────────────────────────────

struct Siglip2EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Siglip2Attention,
    layer_norm2: LayerNorm,
    mlp: Siglip2MLP,
}

impl Siglip2EncoderLayer {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, eps, vb.pp("layer_norm1"))?,
            self_attn: Siglip2Attention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.hidden_size, eps, vb.pp("layer_norm2"))?,
            mlp: Siglip2MLP::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.self_attn.forward(&self.layer_norm1.forward(x)?)?;
        let x = (residual + &x)?;
        let residual = &x;
        let x = self.mlp.forward(&self.layer_norm2.forward(&x)?)?;
        residual + x
    }
}

// ─── Vision Encoder ───────────────────────────────────────────────────────────

struct Siglip2Encoder {
    layers: Vec<Siglip2EncoderLayer>,
}

impl Siglip2Encoder {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Siglip2EncoderLayer::new(cfg, vb.pp("layers").pp(i))?);
        }
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

// ─── Vision Transformer ───────────────────────────────────────────────────────

struct Siglip2VisionTransformer {
    embeddings: Siglip2VisionEmbeddings,
    encoder: Siglip2Encoder,
    /// Present when all encoder layers are used (standard configuration).
    post_layernorm: Option<LayerNorm>,
}

impl Siglip2VisionTransformer {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: Siglip2VisionEmbeddings::new(cfg, vb.pp("embeddings"))?,
            encoder: Siglip2Encoder::new(cfg, vb.pp("encoder"))?,
            post_layernorm: Some(layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("post_layernorm"),
            )?),
        })
    }

    /// `pixel_values`: `[S, patch_dim]`. `spatial_shapes`: `(h, w)` per tile.
    /// Returns `[S, hidden]`.
    fn forward(&self, pixel_values: &Tensor, spatial_shapes: &[(usize, usize)]) -> Result<Tensor> {
        let x = self.embeddings.forward(pixel_values, spatial_shapes)?;
        let x = self.encoder.forward(&x)?;
        if let Some(ref ln) = self.post_layernorm {
            ln.forward(&x)
        } else {
            Ok(x)
        }
    }
}

// ─── Multimodal Projector ─────────────────────────────────────────────────────

/// Pixel-shuffle multimodal projector.
///
/// Gathers `factor×factor` spatially-neighbouring tokens into one output token
/// (inverse of pixel-shuffle / sub-pixel upsampling), then projects the expanded
/// feature through an optional LayerNorm → linear_1 → GELU → linear_2.
///
/// Weight paths: `multi_modal_projector.{layer_norm,linear_1,linear_2}.*`.
struct Lfm2VLMultiModalProjector {
    layer_norm: Option<LayerNorm>,
    linear_1: Linear,
    linear_2: Linear,
    factor: usize,
}

impl Lfm2VLMultiModalProjector {
    fn new(
        vision_hidden: usize,
        proj_hidden: usize,
        text_hidden: usize,
        factor: usize,
        bias: bool,
        use_layernorm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_dim = vision_hidden * factor * factor;
        let layer_norm = if use_layernorm {
            Some(layer_norm(in_dim, 1e-5, vb.pp("layer_norm"))?)
        } else {
            None
        };
        let linear_1 = if bias {
            linear(in_dim, proj_hidden, vb.pp("linear_1"))?
        } else {
            candle_nn::linear_no_bias(in_dim, proj_hidden, vb.pp("linear_1"))?
        };
        let linear_2 = if bias {
            linear(proj_hidden, text_hidden, vb.pp("linear_2"))?
        } else {
            candle_nn::linear_no_bias(proj_hidden, text_hidden, vb.pp("linear_2"))?
        };
        Ok(Self {
            layer_norm,
            linear_1,
            linear_2,
            factor,
        })
    }

    /// `vision_features`: `[S, vision_hidden]` packed tokens.
    /// `spatial_shapes`: `(h, w)` per tile.
    /// Returns `[S/factor², text_hidden]`.
    fn forward(
        &self,
        vision_features: &Tensor,
        spatial_shapes: &[(usize, usize)],
    ) -> Result<Tensor> {
        let factor = self.factor;
        let (_, hidden) = vision_features.dims2()?;
        let device = vision_features.device();

        // Build gather indices for pixel-shuffle downsampling.
        //
        // For each output position (r_out, c_out), collect the factor×factor
        // source tokens in row-major order: (r_out*f+dh, c_out*f+dw).
        let mut gather_idx: Vec<u32> = Vec::new();
        let mut offset = 0usize;

        for &(height, width) in spatial_shapes {
            let length = height * width;
            if length == 0 {
                continue;
            }
            let height_out = height / factor;
            let width_out = width / factor;

            for r_out in 0..height_out {
                for c_out in 0..width_out {
                    for dh in 0..factor {
                        for dw in 0..factor {
                            let token_idx = (r_out * factor + dh) * width + (c_out * factor + dw);
                            gather_idx.push((offset + token_idx) as u32);
                        }
                    }
                }
            }
            offset += length;
        }

        let n_gathered = gather_idx.len();
        let idx_t = Tensor::from_vec(gather_idx, n_gathered, device)?;
        let gathered = vision_features.index_select(&idx_t, 0)?; // [n_gathered, hidden]

        // Reshape so each output position has factor²×hidden features.
        let n_out = n_gathered / (factor * factor);
        let unshuffled = gathered.reshape((n_out, factor * factor * hidden))?;

        let x = if let Some(ref ln) = self.layer_norm {
            ln.forward(&unshuffled)?
        } else {
            unshuffled
        };

        // Projector MLP with exact GELU (config default "gelu").
        let x = self.linear_1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear_2.forward(&x)
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

/// LFM2-VL vision-language model for conditional generation.
///
/// Combines a SigLIP2 vision encoder with the LFM2 hybrid text model.
pub struct Lfm2VLForConditionalGeneration {
    vision_tower: Siglip2VisionTransformer,
    multi_modal_projector: Lfm2VLMultiModalProjector,
    language_model: Lfm2ForCausalLM,
    device: Device,
    #[allow(dead_code)]
    downsample_factor: usize,
    #[allow(dead_code)]
    image_token_id: i64,
}

impl Lfm2VLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_cfg = Siglip2VisionConfig::from_model_config(cfg)?;
        let vl_cfg = Lfm2VLConfig::from_model_config(cfg);

        // Vision encoder: model.vision_tower.vision_model.*
        let vb_vision = vb.pp("model").pp("vision_tower").pp("vision_model");
        let vision_tower = Siglip2VisionTransformer::new(&vis_cfg, vb_vision)?;

        // Projector: model.multi_modal_projector.*
        let vb_proj = vb.pp("model").pp("multi_modal_projector");
        let multi_modal_projector = Lfm2VLMultiModalProjector::new(
            vis_cfg.hidden_size,
            vl_cfg.projector_hidden_size,
            cfg.hidden_size,
            vl_cfg.downsample_factor,
            vl_cfg.projector_bias,
            vl_cfg.projector_use_layernorm,
            vb_proj,
        )?;

        // LLM: body at model.language_model.*, head at lm_head.* (root).
        let language_model = Lfm2ForCausalLM::new_vlm(cfg, vb.clone())?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            device: vb.device().clone(),
            downsample_factor: vl_cfg.downsample_factor,
            image_token_id: vl_cfg.image_token_id,
        })
    }

    /// Encode pixel values and project into the text embedding space.
    ///
    /// `pixel_values`: `[S, patch_dim]` packed patches (all tiles concatenated).
    /// `spatial_shapes`: `(h, w)` per tile in patch-grid units.
    /// Returns `[S/factor², hidden_size]`.
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        spatial_shapes: &[(usize, usize)],
    ) -> Result<Tensor> {
        let features = self.vision_tower.forward(pixel_values, spatial_shapes)?;
        self.multi_modal_projector
            .forward(&features, spatial_shapes)
    }
}

impl crate::engine::ModelForward for Lfm2VLForConditionalGeneration {
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
        // Decode steps have no image tokens; delegate directly to the language model.
        self.language_model
            .forward_decode_batch(input_ids, sequences, kv_cache_mgr)
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
        let text_embeddings = self.language_model.embed_text(input_ids)?; // [B, S, D]

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

    /// Small vision config: patch_size=2, in_channels=1, hidden=8, num_heads=2,
    /// intermediate=16, depth=1, pe_size=2 (num_patches=4).
    fn vis_cfg_json() -> serde_json::Value {
        json!({
            "patch_size": 2,
            "num_channels": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "layer_norm_eps": 1e-6,
            "num_patches": 4  // pe_size = 2
        })
    }

    /// Full LFM2-VL model config for CPU testing.
    ///
    /// Vision: hidden=8, patch=2, heads=2, layers=1, pe_size=2
    /// Projector: factor=2, projector_hidden=16, no bias, no LN
    /// LLM: hidden=16, 1 attention layer (matches tiny Lfm2 dense config)
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("vision_config".to_string(), vis_cfg_json());
        extra.insert("downsample_factor".to_string(), json!(2));
        extra.insert("projector_hidden_size".to_string(), json!(16));
        extra.insert("projector_bias".to_string(), json!(false));
        extra.insert("projector_use_layernorm".to_string(), json!(false));
        extra.insert("image_token_id".to_string(), json!(32000));
        // LFM2 dense: single full-attention layer (no short_conv)
        extra.insert("layer_types".to_string(), json!(["full_attention"]));
        extra.insert("conv_L_cache".to_string(), json!(4));

        ModelConfig {
            architectures: vec!["Lfm2VLForConditionalGeneration".to_string()],
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 32,
            vocab_size: 64,
            max_position_embeddings: 128,
            head_dim: 8,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            extra,
            ..Default::default()
        }
    }

    fn make_kv_mgr(cfg: &ModelConfig) -> crate::kv_cache::KVCacheManager {
        crate::kv_cache::KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 32,
            num_layers: 1, // one attention layer in the test LFM2
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: candle_core::Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .expect("KVCacheManager::new")
    }

    #[test]
    fn test_vision_embeddings() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let vis_cfg = Siglip2VisionConfig {
            patch_size: 2,
            num_channels: 1,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            layer_norm_eps: 1e-6,
            num_patches: 4,
            pe_size: 2,
        };

        let emb = Siglip2VisionEmbeddings::new(&vis_cfg, vb).unwrap();
        // 4 patches (2×2 tile), patch_dim = 1*2*2 = 4
        let px = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let spatial_shapes = vec![(2usize, 2usize)];
        let out = emb.forward(&px, &spatial_shapes).unwrap();
        assert_eq!(out.dims(), &[4, 8], "embeddings shape");
    }

    #[test]
    fn test_projector() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // factor=2: 4 tokens → 1 output token with 4*8=32 input features → 4 output
        let proj = Lfm2VLMultiModalProjector::new(8, 16, 4, 2, false, false, vb).unwrap();
        let features = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let spatial_shapes = vec![(2usize, 2usize)];
        let out = proj.forward(&features, &spatial_shapes).unwrap();
        // 4 tokens / factor²=4 → 1 output token; linear_2 output dim = 4
        assert_eq!(out.dims(), &[1, 4], "projector output shape");
    }

    #[test]
    fn test_vision_encoder() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let vis_cfg = Siglip2VisionConfig {
            patch_size: 2,
            num_channels: 1,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            layer_norm_eps: 1e-6,
            num_patches: 4,
            pe_size: 2,
        };

        let vt = Siglip2VisionTransformer::new(&vis_cfg, vb).unwrap();
        let px = Tensor::zeros((4, 4), DType::F32, &device).unwrap(); // 4 patches, patch_dim=4
        let spatial_shapes = vec![(2usize, 2usize)];
        let out = vt.forward(&px, &spatial_shapes).unwrap();
        assert_eq!(out.dims(), &[4, 8], "vision encoder output shape");
    }

    #[test]
    fn test_forward_text_only() {
        let cfg = make_cfg();
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2VLForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv_mgr = make_kv_mgr(&cfg);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let bt = crate::kv_cache::BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();
        let logits = model
            .forward(&input_ids, 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        // logits: [B, S, vocab] — check vocab dimension
        assert_eq!(logits.dims().last().copied().unwrap(), cfg.vocab_size);
    }

    #[test]
    fn test_forward_multimodal() {
        use crate::multimodal::{MultimodalInputs, ProcessedImage};

        let cfg = make_cfg();
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2VLForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode a 2×2 tile of patches (patch_dim = 1*2*2 = 4) through vision encoder.
        let px = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let spatial_shapes = vec![(2usize, 2usize)];
        // encode_images: 4 vis tokens → 1 output token (factor²=4), hidden=16
        let img_features = model.encode_images(&px, &spatial_shapes).unwrap();
        assert_eq!(
            img_features.dims(),
            &[1, 16],
            "encoded image features shape"
        );

        // Build multimodal inputs: place image at token position 0 in a seq of 5 tokens.
        let processed = ProcessedImage::new(img_features, 1);
        let mm_inputs = MultimodalInputs::with_images(
            vec![32000, 0, 0, 0, 0], // image_token at pos 0, then text tokens
            vec![(0, processed)],
        );

        let mut kv_mgr = make_kv_mgr(&cfg);
        // 5 tokens: position 0 is the image token (1 visual), positions 1-4 are text.
        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).unwrap();
        let bt = crate::kv_cache::BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..5).collect();

        let logits = model
            .forward_multimodal(
                &input_ids,
                Some(&mm_inputs),
                0,
                &mut kv_mgr,
                &bt,
                &slot_mapping,
            )
            .unwrap();
        // logits: [B, S, vocab] — check vocab dimension
        assert_eq!(logits.dims().last().copied().unwrap(), cfg.vocab_size);
    }
}
