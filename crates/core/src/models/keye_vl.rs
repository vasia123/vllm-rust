//! Keye-VL 1.5 vision-language model.
//!
//! Architecture:
//! - Vision: KeyeSiglipVisionTransformer — SigLIP ViT with 2D RoPE instead of fixed
//!   positional embeddings, bilinear-interpolated learned position embeddings as additive bias.
//! - Projector: KeyeVL1_5Projector — 2×2 spatial merge, LayerNorm, two linear layers with GELU.
//! - Language: Qwen3ForCausalLM.
//!
//! # Weight paths (after vLLM WeightsMapper: `model.` → `language_model.model.`,
//! `lm_head.` → `language_model.lm_head.`)
//!
//! ```text
//! visual.vision_model.embeddings.patch_embedding.{weight,bias}
//! visual.vision_model.embeddings.position_embedding.weight
//! visual.vision_model.embeddings.packing_position_embedding.weight
//! visual.vision_model.encoder.layers.{i}.{layer_norm1,layer_norm2}.*
//! visual.vision_model.encoder.layers.{i}.self_attn.{qkv_proj,out_proj}.*
//! visual.vision_model.encoder.layers.{i}.mlp.{fc1,fc2}.*
//! visual.vision_model.post_layernorm.*
//! mlp_AR.{pre_norm,linear_1,linear_2}.*
//! language_model.{model.*,lm_head.*}
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/keye.py`
//! `reference/vllm/vllm/model_executor/models/keye_vl1_5.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen3::Qwen3ForCausalLM;

// ─── Vision Config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct KeyeVisionConfig {
    pub(crate) patch_size: usize,
    pub(crate) in_channels: usize,
    pub(crate) embed_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) depth: usize,
    pub(crate) layer_norm_eps: f64,
}

impl KeyeVisionConfig {
    pub(crate) fn from_json(json: &serde_json::Value) -> Self {
        let patch_size = json
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(14) as usize;
        let in_channels = json
            .get("num_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let embed_dim = json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1152) as usize;
        let num_heads = json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let intermediate_size = json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4304) as usize;
        let depth = json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(27) as usize;
        let layer_norm_eps = json
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6);
        Self {
            patch_size,
            in_channels,
            embed_dim,
            num_heads,
            intermediate_size,
            depth,
            layer_norm_eps,
        }
    }

    fn patch_input_dim(&self) -> usize {
        self.in_channels * self.patch_size * self.patch_size
    }

    fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

// ─── SigLIP Rotary Embedding ──────────────────────────────────────────────────

/// 1D rotary embedding with dimension `dim = head_dim / 2`.
///
/// `forward(seqlen)` returns `[seqlen, dim/2]` (raw freqs before cos/sin).
/// `build_2d_rope(h, w)` returns `(cos, sin)` each `[h*w, head_dim/2]`
/// following the vLLM Keye convention: index h/w freqs independently, concatenate,
/// take cos/sin of the result.
struct SigLipRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl SigLipRotaryEmbedding {
    fn new(dim: usize) -> Self {
        // inv_freq[i] = 1 / 10000^(2i/dim)  for i in 0..dim/2
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0_f32 / 10000.0_f32.powf(2.0 * i as f32 / dim as f32))
            .collect();
        Self { inv_freq }
    }

    /// Returns `[seqlen, dim/2]` — outer product of positions × inv_freq.
    fn freqs(&self, seqlen: usize, device: &Device) -> Result<Tensor> {
        let seq: Vec<f32> = (0..seqlen).map(|i| i as f32).collect();
        let seq = Tensor::from_vec(seq, (seqlen,), device)?;
        let inv = Tensor::from_vec(self.inv_freq.clone(), (self.inv_freq.len(),), device)?;
        // outer product: [seqlen, dim/2]
        seq.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)
    }

    /// Builds 2D RoPE for `h × w` positions.
    ///
    /// Returns `(cos, sin)` each of shape `[h*w, head_dim/2]`.
    /// Follows vLLM Keye: `pids = [[h0,w0], ...]`, index into freq table,
    /// concatenate h/w freqs, then cos/sin.
    fn build_2d_rope(&self, h: usize, w: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let max_grid = h.max(w);
        let freqs_table = self.freqs(max_grid, device)?; // [max_grid, head_dim/4]

        let mut h_ids: Vec<u32> = Vec::with_capacity(h * w);
        let mut w_ids: Vec<u32> = Vec::with_capacity(h * w);
        for i in 0..h {
            for j in 0..w {
                h_ids.push(i as u32);
                w_ids.push(j as u32);
            }
        }
        let h_t = Tensor::from_vec(h_ids, (h * w,), device)?;
        let w_t = Tensor::from_vec(w_ids, (h * w,), device)?;

        let freqs_h = freqs_table.index_select(&h_t, 0)?; // [h*w, head_dim/4]
        let freqs_w = freqs_table.index_select(&w_t, 0)?; // [h*w, head_dim/4]

        // Concatenate → [h*w, head_dim/2]; take cos/sin directly.
        // This is equivalent to the vLLM logic:
        //   rope_emb = freqs_hw.repeat(1, 2) → [h*w, head_dim]
        //   cos, sin = rope_emb.cos(), rope_emb.sin()
        //   apply_rotary: take first half → same as cos/sin of freqs_hw
        let freqs = Tensor::cat(&[&freqs_h, &freqs_w], 1)?; // [h*w, head_dim/2]
        Ok((freqs.cos()?, freqs.sin()?))
    }
}

/// Apply split-half RoPE to `q` or `k`.
///
/// - `x`: `[N, heads, head_dim]`
/// - `cos`, `sin`: `[N, head_dim/2]`
///
/// Returns `[N, heads, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (n, _heads, head_dim) = x.dims3()?;
    let half = head_dim / 2;
    // cos/sin [N, half] → [N, 1, half] for broadcasting
    let cos = cos.reshape((n, 1, half))?;
    let sin = sin.reshape((n, 1, half))?;
    let x1 = x.narrow(2, 0, half)?; // [N, heads, half]
    let x2 = x.narrow(2, half, half)?; // [N, heads, half]
                                       // rotate_half: [x1*cos - x2*sin, x2*cos + x1*sin]
    let out1 = (x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?))?;
    let out2 = (x2
        .broadcast_mul(&cos)?
        .broadcast_add(&x1.broadcast_mul(&sin)?))?;
    Tensor::cat(&[&out1, &out2], 2)
}

// ─── Vision Embeddings ────────────────────────────────────────────────────────

struct KeyeVisionEmbeddings {
    patch_embedding: Linear, // linear_no_bias(patch_input_dim → embed_dim)
    packing_position_embedding: Embedding, // (32768, embed_dim) for packed multi-image
}

impl KeyeVisionEmbeddings {
    fn new(cfg: &KeyeVisionConfig, vb: VarBuilder) -> Result<Self> {
        // Conv2d patch embed treated as linear (equivalent computation)
        let patch_embedding = linear_no_bias(
            cfg.patch_input_dim(),
            cfg.embed_dim,
            vb.pp("patch_embedding"),
        )?;
        // packing_position_embedding is used for packed multi-image position IDs
        let packing_position_embedding =
            embedding(32768, cfg.embed_dim, vb.pp("packing_position_embedding"))?;
        Ok(Self {
            patch_embedding,
            packing_position_embedding,
        })
    }

    /// `patches`: `[N, patch_input_dim]` (pre-extracted patches).
    /// Returns `[N, embed_dim]`.
    fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let n = patches.dim(0)?;
        let x = self.patch_embedding.forward(patches)?; // [N, embed_dim]
                                                        // Add packing position embeddings using sequential IDs
        let pos_ids: Vec<u32> = (0..n as u32).collect();
        let pos_t = Tensor::from_vec(pos_ids, (n,), patches.device())?;
        let pos_emb = self.packing_position_embedding.forward(&pos_t)?; // [N, embed_dim]
        x + pos_emb
    }
}

// ─── Vision MLP ───────────────────────────────────────────────────────────────

struct KeyeSiglipMLP {
    fc1: Linear, // linear_no_bias(embed_dim → intermediate_size)
    fc2: Linear, // linear_no_bias(intermediate_size → embed_dim)
}

impl KeyeSiglipMLP {
    fn new(cfg: &KeyeVisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_no_bias(cfg.embed_dim, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(cfg.intermediate_size, cfg.embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── Vision Attention ─────────────────────────────────────────────────────────

struct KeyeSiglipAttention {
    qkv_proj: Linear, // linear(embed_dim → 3*embed_dim) WITH bias; loaded as fused QKV
    out_proj: Linear, // linear_no_bias(embed_dim → embed_dim)
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl KeyeSiglipAttention {
    fn new(cfg: &KeyeVisionConfig, vb: VarBuilder) -> Result<Self> {
        let qkv_proj = linear(cfg.embed_dim, 3 * cfg.embed_dim, vb.pp("qkv_proj"))?;
        let out_proj = linear_no_bias(cfg.embed_dim, cfg.embed_dim, vb.pp("out_proj"))?;
        let head_dim = cfg.head_dim();
        Ok(Self {
            qkv_proj,
            out_proj,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[N, embed_dim]`; `cos`, `sin`: `[N, head_dim/2]` or `None`.
    fn forward(&self, x: &Tensor, cos_sin: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (n, _) = x.dims2()?;
        let qkv = self.qkv_proj.forward(x)?; // [N, 3*D]
        let q = qkv.narrow(1, 0, self.num_heads * self.head_dim)?;
        let k = qkv.narrow(
            1,
            self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;
        let v = qkv.narrow(
            1,
            2 * self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;

        // [N, heads, head_dim]
        let q = q.reshape((n, self.num_heads, self.head_dim))?;
        let k = k.reshape((n, self.num_heads, self.head_dim))?;
        let v = v.reshape((n, self.num_heads, self.head_dim))?;

        let (q, k) = if let Some((cos, sin)) = cos_sin {
            (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
        } else {
            (q, k)
        };

        // SDPA: q/k/v → [heads, N, head_dim]
        let q = q.transpose(0, 1)?.contiguous()?; // [heads, N, head_dim]
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let attn = (q.matmul(&k.transpose(1, 2)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [heads, N, head_dim]

        // → [N, embed_dim]
        let out = out
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

// ─── Vision Encoder Layer ─────────────────────────────────────────────────────

struct KeyeSiglipEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: KeyeSiglipAttention,
    layer_norm2: LayerNorm,
    mlp: KeyeSiglipMLP,
}

impl KeyeSiglipEncoderLayer {
    fn new(cfg: &KeyeVisionConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: layer_norm(cfg.embed_dim, eps, vb.pp("layer_norm1"))?,
            self_attn: KeyeSiglipAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.embed_dim, eps, vb.pp("layer_norm2"))?,
            mlp: KeyeSiglipMLP::new(cfg, vb.pp("mlp"))?,
        })
    }

    /// `x`: `[N, embed_dim]`; `cos_sin`: `[N, head_dim/2]` or None.
    fn forward(&self, x: &Tensor, cos_sin: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let residual = x;
        let x = self
            .self_attn
            .forward(&self.layer_norm1.forward(x)?, cos_sin)?;
        let x = (residual + &x)?;
        let residual = &x;
        let x = self.mlp.forward(&self.layer_norm2.forward(&x)?)?;
        residual + x
    }
}

// ─── Vision Encoder ───────────────────────────────────────────────────────────

struct KeyeSiglipEncoder {
    layers: Vec<KeyeSiglipEncoderLayer>,
    rotary_emb: SigLipRotaryEmbedding, // dim = head_dim / 2
}

impl KeyeSiglipEncoder {
    fn new(cfg: &KeyeVisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            layers.push(KeyeSiglipEncoderLayer::new(cfg, vb.pp("layers").pp(i))?);
        }
        // SigLIPRotaryEmbedding dim = head_dim // 2
        let rotary_emb = SigLipRotaryEmbedding::new(cfg.head_dim() / 2);
        Ok(Self { layers, rotary_emb })
    }

    /// `x`: `[N, embed_dim]`; `grid`: `(h, w)` in patch space.
    fn forward(&self, x: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let (h, w) = grid;
        let (cos, sin) = self.rotary_emb.build_2d_rope(h, w, x.device())?;
        let cos_sin = (&cos, &sin);
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, Some(cos_sin))?;
        }
        Ok(x)
    }
}

// ─── Vision Transformer ───────────────────────────────────────────────────────

pub(crate) struct KeyeSiglipVisionTransformer {
    embeddings: KeyeVisionEmbeddings,
    encoder: KeyeSiglipEncoder,
    post_layernorm: LayerNorm,
}

impl KeyeSiglipVisionTransformer {
    pub(crate) fn new(cfg: &KeyeVisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb_vm = vb.pp("vision_model");
        Ok(Self {
            embeddings: KeyeVisionEmbeddings::new(cfg, vb_vm.pp("embeddings"))?,
            encoder: KeyeSiglipEncoder::new(cfg, vb_vm.pp("encoder"))?,
            post_layernorm: layer_norm(
                cfg.embed_dim,
                cfg.layer_norm_eps,
                vb_vm.pp("post_layernorm"),
            )?,
        })
    }

    /// `patches`: `[N, patch_input_dim]`; `grid`: `(h, w)`.
    /// Returns `[N, embed_dim]`.
    pub(crate) fn forward(&self, patches: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let x = self.embeddings.forward(patches)?; // [N, embed_dim]
        let x = self.encoder.forward(&x, grid)?; // [N, embed_dim]
        self.post_layernorm.forward(&x)
    }
}

// ─── Projector ────────────────────────────────────────────────────────────────

/// KeyeVL1_5 spatial projector: 2×2 spatial merge → LayerNorm → linear_1 → GELU → linear_2.
///
/// Weight paths: `mlp_AR.{pre_norm,linear_1,linear_2}.*`.
struct KeyeVL1_5Projector {
    pre_norm: LayerNorm,
    linear_1: Linear, // (merge_dim → merge_dim) WITH bias
    linear_2: Linear, // (merge_dim → text_hidden) WITH bias
    merge: usize,     // merge_kernel_size = 2
}

impl KeyeVL1_5Projector {
    fn new(embed_dim: usize, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let merge = 2usize;
        let merge_dim = embed_dim * merge * merge;
        Ok(Self {
            pre_norm: layer_norm(merge_dim, 1e-5, vb.pp("pre_norm"))?,
            linear_1: linear(merge_dim, merge_dim, vb.pp("linear_1"))?,
            linear_2: linear(merge_dim, text_hidden, vb.pp("linear_2"))?,
            merge,
        })
    }

    /// `x`: `[t*h*w, embed_dim]`; `grid`: `(t, h, w)` where h, w in raw patch counts.
    /// Returns `[t * (h/m) * (w/m), text_hidden]`.
    fn forward(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, h, w) = grid;
        let m = self.merge;
        // Rearrange: (t h p1 w p2) d -> (t h w) (p1 p2 d)
        // where h = h_raw // m, p1 = m, w = w_raw // m, p2 = m
        let embed_dim = x.dim(1)?;
        let h_out = h / m;
        let w_out = w / m;
        // x shape: [t * h * w, embed_dim]
        // → [t, h_out, m, w_out, m, embed_dim]
        let x = x.reshape((t, h_out, m, w_out, m, embed_dim))?;
        // → [t, h_out, w_out, m, m, embed_dim]
        let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
        // → [t * h_out * w_out, m * m * embed_dim]
        let merge_dim = m * m * embed_dim;
        let x = x.reshape((t * h_out * w_out, merge_dim))?;

        let x = self.pre_norm.forward(&x)?;
        let x = self.linear_1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear_2.forward(&x)
    }
}

// ─── Keye base projector ──────────────────────────────────────────────────────

/// Keye base spatial projector: LayerNorm per-patch → 2×2 spatial merge → linear_1 → GELU → linear_2.
///
/// Unlike `KeyeVL1_5Projector`, `pre_norm` operates on the raw patch dimension (`embed_dim`)
/// BEFORE the spatial merge, matching `keye.py` `Projector`.
///
/// Weight paths: `mlp_AR.{pre_norm,linear_1,linear_2}.*`.
pub(crate) struct KeyeProjector {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    merge: usize,
}

impl KeyeProjector {
    pub(crate) fn new(embed_dim: usize, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let merge = 2usize;
        let merge_dim = embed_dim * merge * merge;
        Ok(Self {
            pre_norm: layer_norm(embed_dim, 1e-5, vb.pp("pre_norm"))?,
            linear_1: linear(merge_dim, merge_dim, vb.pp("linear_1"))?,
            linear_2: linear(merge_dim, text_hidden, vb.pp("linear_2"))?,
            merge,
        })
    }

    /// `x`: `[t*h*w, embed_dim]`; `grid`: `(t, h, w)` where h, w in raw patch counts.
    /// Returns `[t * (h/m) * (w/m), text_hidden]`.
    pub(crate) fn forward(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        // Norm each patch independently before merging — key difference vs VL1.5.
        let x = self.pre_norm.forward(x)?;

        let (t, h, w) = grid;
        let m = self.merge;
        let embed_dim = x.dim(1)?;
        let h_out = h / m;
        let w_out = w / m;
        let x = x.reshape((t, h_out, m, w_out, m, embed_dim))?;
        let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
        let merge_dim = m * m * embed_dim;
        let x = x.reshape((t * h_out * w_out, merge_dim))?;

        let x = self.linear_1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear_2.forward(&x)
    }
}

// ─── merge_multimodal ─────────────────────────────────────────────────────────

/// Replace image-patch positions in `text_embeds` with encoded image embeddings.
///
/// Uses position-based replacement matching the `mm_inputs.image_embeddings` layout.
fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }
    let (_b, seq_len, _d) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_vec3::<f32>()?;
    for (position, processed) in &mm_inputs.image_embeddings {
        let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
        let batch_idx = position / seq_len;
        let start_pos = position % seq_len;
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

/// Keye-VL 1.5 vision-language model.
pub struct KeyeVL1_5ForConditionalGeneration {
    visual: KeyeSiglipVisionTransformer,
    mlp_ar: KeyeVL1_5Projector,
    language_model: Qwen3ForCausalLM,
    device: Device,
}

impl KeyeVL1_5ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let vis_cfg = KeyeVisionConfig::from_json(&vis_json);

        let visual = KeyeSiglipVisionTransformer::new(&vis_cfg, vb.pp("visual"))?;
        let mlp_ar = KeyeVL1_5Projector::new(vis_cfg.embed_dim, cfg.hidden_size, vb.pp("mlp_AR"))?;
        let language_model = Qwen3ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            visual,
            mlp_ar,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Encode image patches for one image: `patches [t*h*w, in_ch*ps²]`, grid `(t, h, w)`.
    /// Returns `[t*(h/2)*(w/2), text_hidden]`.
    pub fn encode_images(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, h, w) = grid;
        let vis_feats = self.visual.forward(patches, (h, w))?; // [t*h*w, embed_dim]
        self.mlp_ar.forward(&vis_feats, (t, h, w))
    }
}

impl crate::engine::ModelForward for KeyeVL1_5ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_with_embeddings(
            &embeddings,
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

    fn device(&self) -> &Device {
        &self.device
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
}

// ─── Keye base model ──────────────────────────────────────────────────────────

/// Keye vision-language model (base variant — `keye.py`).
///
/// Identical to `KeyeVL1_5ForConditionalGeneration` except the projector applies
/// LayerNorm on individual patches BEFORE the 2×2 spatial merge.
pub struct KeyeForConditionalGeneration {
    visual: KeyeSiglipVisionTransformer,
    mlp_ar: KeyeProjector,
    language_model: Qwen3ForCausalLM,
    device: Device,
}

impl KeyeForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let vis_cfg = KeyeVisionConfig::from_json(&vis_json);

        let visual = KeyeSiglipVisionTransformer::new(&vis_cfg, vb.pp("visual"))?;
        let mlp_ar = KeyeProjector::new(vis_cfg.embed_dim, cfg.hidden_size, vb.pp("mlp_AR"))?;
        let language_model = Qwen3ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            visual,
            mlp_ar,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Encode image patches: `patches [t*h*w, in_ch*ps²]`, grid `(t, h, w)`.
    /// Returns `[t*(h/2)*(w/2), text_hidden]`.
    pub fn encode_images(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, h, w) = grid;
        let vis_feats = self.visual.forward(patches, (h, w))?;
        self.mlp_ar.forward(&vis_feats, (t, h, w))
    }
}

impl crate::engine::ModelForward for KeyeForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_with_embeddings(
            &embeddings,
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

    fn device(&self) -> &Device {
        &self.device
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
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use crate::multimodal::ProcessedImage;
    use candle_core::DType;
    use serde_json::json;

    fn test_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Vision config
        extra.insert(
            "vision_config".to_string(),
            json!({
                "patch_size": 2,
                "num_channels": 1,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "image_size": 8,    // gives num_positions = (8/2)^2 = 16
                "layer_norm_eps": 1e-6,
            }),
        );
        extra.insert("image_token_id".to_string(), json!(9u32));

        ModelConfig {
            architectures: vec!["KeyeVL1_5ForConditionalGeneration".to_string()],
            hidden_size: 32,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 64,
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
    fn test_keye_vl_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeVL1_5ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_keye_vl_vision_encode() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeVL1_5ForConditionalGeneration::new(&cfg, vb).unwrap();

        // 16 patches (4×4 grid), patch_input_dim = 1*2*2 = 4
        let patches = Tensor::zeros((16usize, 4), DType::F32, &device).unwrap();
        let result = model.encode_images(&patches, (1, 4, 4));
        assert!(result.is_ok(), "encode_images failed: {:?}", result.err());
        // 4×4 → 2×2 after 2×2 merge → 4 merged tokens, text_hidden=32
        assert_eq!(result.unwrap().dims(), &[4, 32]);
    }

    #[test]
    fn test_keye_vl_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeVL1_5ForConditionalGeneration::new(&cfg, vb).unwrap();

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
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_keye_vl_with_image() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeVL1_5ForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode image: 16 patches → 4 projected tokens of dim 32
        let patches = Tensor::zeros((16usize, 4), DType::F32, &device).unwrap();
        let img_feats = model.encode_images(&patches, (1, 4, 4)).unwrap();
        let processed = ProcessedImage::new(img_feats, 4);

        // Sequence: 6 tokens with 4 image-pad tokens at positions 1..5
        let input_ids = Tensor::from_slice(&[0u32, 9, 9, 9, 9, 0], (1usize, 6), &device).unwrap();
        let mm = MultimodalInputs::with_images(vec![0u32, 9, 9, 9, 9, 0], vec![(1, processed)]);

        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 6).unwrap();
        let slot_mapping = bt.slot_mapping(0, 6);

        let result =
            model.forward_multimodal(&input_ids, Some(&mm), 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "multimodal forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, 6, 64]);
    }

    #[test]
    fn test_keye_vl_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeVL1_5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 4).unwrap();
        let slot = bt.slot_mapping(3, 1);
        let seq = crate::engine::DecodeSequenceMetadata {
            request_id: 1,
            seqlen_offset: 3,
            slot_mapping: slot.clone(),
            block_ids: bt.block_ids().to_vec(),
        };

        let tok = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let result = model.forward_decode_batch(&tok, &[seq], &mut kv);
        assert!(result.is_ok(), "decode_batch failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, 1, 64]);
    }

    // ── KeyeForConditionalGeneration (base variant) tests ──────────────────

    #[test]
    fn test_keye_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_keye_vision_encode() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeForConditionalGeneration::new(&cfg, vb).unwrap();

        // 16 patches (4×4 grid), patch_input_dim = 1*2*2 = 4
        let patches = Tensor::zeros((16usize, 4), DType::F32, &device).unwrap();
        let result = model.encode_images(&patches, (1, 4, 4));
        assert!(result.is_ok(), "encode_images failed: {:?}", result.err());
        // 4×4 → 2×2 after 2×2 merge → 4 merged tokens, text_hidden=32
        assert_eq!(result.unwrap().dims(), &[4, 32]);
    }

    #[test]
    fn test_keye_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KeyeForConditionalGeneration::new(&cfg, vb).unwrap();

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
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }
}
