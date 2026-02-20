//! Kimi-K2.5 Vision-Language Model.
//!
//! Architecture: MoonViT3dPretrainedModel + KimiK25MultiModalProjector + DeepSeekForCausalLM.
//!
//! The vision tower is a 3D variant of MoonViT that handles video chunks (T frames per clip).
//! Key differences from Kimi-VL:
//! - 3D patch embed with temporal sincos positional embedding
//! - `Rope2DPosEmbRepeated`: spatial freqs repeated T times (not expanded temporally)
//! - `tpool_patch_merger`: temporal mean-pooling + spatial grouping
//! - Weight path `mm_projector.*` (not `multi_modal_projector.*`)
//!
//! # Weight paths
//!
//! ```text
//! vision_tower.patch_embed.proj.*         → Conv2d projection
//! vision_tower.patch_embed.pos_emb.weight → Learnable2DInterpPosEmbWithTemporal
//! vision_tower.encoder.blocks.{i}.*      → MoonVitEncoderLayer (shared impl)
//! vision_tower.encoder.final_layernorm.* → Final LayerNorm
//! mm_projector.pre_norm.*                → LayerNorm before flatten
//! mm_projector.linear_1.*                → First linear + GELU
//! mm_projector.linear_2.*                → Second linear
//! language_model.*                       → DeepSeekForCausalLM (DeepSeek-V2)
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/kimi_k25.py`
//! `reference/vllm/vllm/model_executor/models/kimi_k25_vit.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::deepseek::DeepSeekForCausalLM;
use super::moonvit::{bilinear_interp_2d, MoonVitConfig, MoonVitEncoderLayer, Rope2DPosEmb};

// ─── Vision Config ────────────────────────────────────────────────────────────

/// Kimi-K2.5 vision encoder configuration.
#[derive(Debug, Clone)]
struct KimiK25VisionConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    in_channels: usize,
    patch_size: usize,
    init_pos_emb_height: usize,
    init_pos_emb_width: usize,
    /// Number of temporal frames for sincos temporal pos emb (default 4).
    init_pos_emb_time: usize,
    /// Spatial merge kernel (kh, kw); default (2, 2).
    merge_kernel_size: [usize; 2],
    /// LLM hidden size (used for projector output dim).
    mm_hidden_size: usize,
}

impl KimiK25VisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vc = cfg
            .extra
            .get("vision_config")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let get = |key: &str, default: usize| -> usize {
            vc.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let merge_kernel_size = vc
            .get("merge_kernel_size")
            .and_then(|v| v.as_array())
            .map(|arr| {
                let h = arr.first().and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                let w = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                [h, w]
            })
            .unwrap_or([2, 2]);

        Self {
            hidden_size: get("hidden_size", 1024),
            num_hidden_layers: get("num_hidden_layers", 24),
            num_attention_heads: get("num_attention_heads", 16),
            intermediate_size: get("intermediate_size", 4096),
            in_channels: get("in_channels", 3),
            patch_size: get("patch_size", 14),
            init_pos_emb_height: get("init_pos_emb_height", 32),
            init_pos_emb_width: get("init_pos_emb_width", 32),
            init_pos_emb_time: get("init_pos_emb_time", 4),
            merge_kernel_size,
            mm_hidden_size: cfg.hidden_size,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Convert to `MoonVitConfig` for reuse of encoder layer constructor.
    fn as_moonvit_cfg(&self) -> MoonVitConfig {
        MoonVitConfig {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            in_channels: self.in_channels,
            patch_size: self.patch_size,
            init_pos_emb_height: self.init_pos_emb_height,
            init_pos_emb_width: self.init_pos_emb_width,
            merge_kernel_size: self.merge_kernel_size,
            // RoPE table size is hardcoded 512×512 in Python (not from config)
            rope_max_height: 512,
            rope_max_width: 512,
            rope_theta: 10000.0,
        }
    }
}

// ─── Sincos 1D Positional Embedding ──────────────────────────────────────────

/// Compute 1D sincos positional embedding for temporal positions.
///
/// Returns a flat buffer of shape `[t_size, dim]`:
/// - First `dim/2` entries per row: sin(pos * omega)
/// - Second `dim/2` entries per row: cos(pos * omega)
///   where `omega[d] = 1 / 10000^(d / (dim/2))`.
fn sincos_1d_pos_embed(dim: usize, t_size: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut data = vec![0.0f32; t_size * dim];
    for t in 0..t_size {
        for d in 0..half {
            let omega = 1.0f32 / 10000.0f32.powf(d as f32 / half as f32);
            let angle = t as f32 * omega;
            data[t * dim + d] = angle.sin();
            data[t * dim + half + d] = angle.cos();
        }
    }
    data
}

// ─── 3D Patch Embedding ───────────────────────────────────────────────────────

/// Learnable 2D positional embedding extended with fixed 1D sincos temporal embedding.
///
/// For single-frame inputs (t=1) only the 2D spatial embedding is applied.
/// For multi-frame inputs (t>1) the temporal sincos embedding is broadcast-added.
///
/// Python reference: `Learnable2DInterpPosEmbDivided_fixed`.
struct Learnable2DInterpPosEmbWithTemporal {
    weight: Tensor,      // [H_init, W_init, D] — learnable spatial embedding
    time_weight: Tensor, // [num_frames, 1, D]  — fixed sincos (NOT in checkpoint)
    init_height: usize,
    init_width: usize,
}

impl Learnable2DInterpPosEmbWithTemporal {
    fn new(
        height: usize,
        width: usize,
        num_frames: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((height, width, dim), "weight")?;

        // Precompute fixed sincos temporal embedding — not saved in checkpoint
        let sincos = sincos_1d_pos_embed(dim, num_frames); // [num_frames * dim]
        let time_weight =
            Tensor::from_vec(sincos, (num_frames, 1, dim), vb.device())?.to_dtype(DType::F32)?;

        Ok(Self {
            weight,
            time_weight,
            init_height: height,
            init_width: width,
        })
    }

    /// Add 3D positional embedding to packed sequence `x: [L, D]`.
    ///
    /// `grid_thws`: `[[t, h, w], ...]` — frames, height, width per media item.
    /// `L = sum(t_i * h_i * w_i)`.
    fn forward(&self, x: &Tensor, grid_thws: &[[usize; 3]]) -> Result<Tensor> {
        let dim = self.weight.dim(2)?;
        let mut parts = Vec::new();

        for &[t, h, w] in grid_thws {
            // Spatial embedding for this h×w grid
            let pos_2d = if h == self.init_height && w == self.init_width {
                self.weight
                    .contiguous()?
                    .reshape((h * w, dim))?
                    .to_dtype(x.dtype())?
            } else {
                bilinear_interp_2d(&self.weight, h, w)?.to_dtype(x.dtype())?
            };

            let pos_3d = if t == 1 {
                // Image: no temporal addition needed
                pos_2d // [h*w, D]
            } else {
                // Video chunk: broadcast-add temporal embedding
                // pos_2d:  [h*w, D] → [1, h*w, D]
                // time_weight[0:t]: [t, 1, D] broadcast → [t, h*w, D]
                let spatial = pos_2d.unsqueeze(0)?; // [1, h*w, D]
                let tw = self.time_weight.narrow(0, 0, t)?.to_dtype(x.dtype())?; // [t, 1, D]
                spatial.broadcast_add(&tw)? // [t, h*w, D]
            };

            parts.push(pos_3d.reshape((t * h * w, dim))?);
        }

        let pos_cat = Tensor::cat(&parts, 0)?;
        x + pos_cat
    }
}

/// Project pre-extracted patches and add 3D positional embedding.
///
/// Input: `pixel_values [L, C, ps, ps]` — all patches concatenated.
/// Output: `[L, hidden_size]`.
struct MoonVision3dPatchEmbed {
    proj: Conv2d,
    pos_emb: Learnable2DInterpPosEmbWithTemporal,
}

impl MoonVision3dPatchEmbed {
    fn new(cfg: &KimiK25VisionConfig, vb: VarBuilder) -> Result<Self> {
        let ps = cfg.patch_size;
        let conv_cfg = Conv2dConfig {
            stride: ps,
            padding: 0,
            ..Default::default()
        };
        let proj = conv2d(
            cfg.in_channels,
            cfg.hidden_size,
            ps,
            conv_cfg,
            vb.pp("proj"),
        )?;
        let pos_emb = Learnable2DInterpPosEmbWithTemporal::new(
            cfg.init_pos_emb_height,
            cfg.init_pos_emb_width,
            cfg.init_pos_emb_time,
            cfg.hidden_size,
            vb.pp("pos_emb"),
        )?;
        Ok(Self { proj, pos_emb })
    }

    fn forward(&self, x: &Tensor, grid_thws: &[[usize; 3]]) -> Result<Tensor> {
        // [L, C, ps, ps] → [L, hidden, 1, 1] → [L, hidden]
        let x = self.proj.forward(x)?;
        let (l, c, _, _) = x.dims4()?;
        let x = x.reshape((l, c))?;
        self.pos_emb.forward(&x, grid_thws)
    }
}

// ─── 3D Encoder ──────────────────────────────────────────────────────────────

/// Full transformer encoder stack with 3D RoPE (spatial repeated T times per frame).
///
/// Reuses `MoonVitEncoderLayer` weight layout; only RoPE frequency construction differs.
struct MoonViT3dEncoder {
    /// Shared precomputed RoPE tables; `get_freqs_by_seqlens_3d` handles T repetition.
    rope_2d: Rope2DPosEmb,
    blocks: Vec<MoonVitEncoderLayer>,
    final_layernorm: LayerNorm,
}

impl MoonViT3dEncoder {
    fn new(cfg: &KimiK25VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mvit_cfg = cfg.as_moonvit_cfg();
        // Python hardcodes 512×512 for the RoPE table size
        let rope_2d = Rope2DPosEmb::new(cfg.head_dim(), 512, 512, 10000.0, vb.device())?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| MoonVitEncoderLayer::new(&mvit_cfg, vb.pp("blocks").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let final_layernorm = layer_norm(cfg.hidden_size, 1e-5, vb.pp("final_layernorm"))?;
        Ok(Self {
            rope_2d,
            blocks,
            final_layernorm,
        })
    }

    /// `x`: `[L, hidden]` — packed patches; `L = sum(t_i * h_i * w_i)`.
    fn forward(&self, x: &Tensor, grid_thws: &[[usize; 3]]) -> Result<Tensor> {
        let (freqs_cos, freqs_sin) = self.rope_2d.get_freqs_by_seqlens_3d(grid_thws)?;
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, &freqs_cos, &freqs_sin)?;
        }
        self.final_layernorm.forward(&x)
    }
}

// ─── Temporal Pooling Patch Merger ───────────────────────────────────────────

/// Temporal mean-pooling + spatial grouping.
///
/// For each media item with grid `(t, h, w)` and merge kernel `(kh, kw)`:
/// 1. Input slice: `[t*h*w, D]`
/// 2. Reshape to `[t, nh, kh, nw, kw, D]` where `nh = h/kh`, `nw = w/kw`
/// 3. Mean over `dim=0` (temporal pooling) → `[nh, kh, nw, kw, D]`
/// 4. Permute `(0,2,1,3,4)` → `[nh, nw, kh, kw, D]`
/// 5. Reshape to `[nh*nw, kh*kw, D]`
///
/// Returns one tensor per media item.
pub fn tpool_patch_merger(
    x: &Tensor,
    grid_thws: &[[usize; 3]],
    merge_kernel_size: [usize; 2],
) -> Result<Vec<Tensor>> {
    let d = x.dim(1)?;
    let [kh, kw] = merge_kernel_size;
    let mut outputs = Vec::with_capacity(grid_thws.len());
    let mut offset = 0usize;

    for &[t, h, w] in grid_thws {
        let seq_len = t * h * w;
        let seq = x.narrow(0, offset, seq_len)?; // [t*h*w, D]
        let nh = h / kh;
        let nw = w / kw;

        // Reshape → temporal pool → spatial rearrange
        let v = seq.reshape((t, nh, kh, nw, kw, d))?;
        // Mean over temporal dim=0 → [nh, kh, nw, kw, D]
        let v = v.mean(0)?;
        // Permute (0,2,1,3,4): nh,kh,nw,kw,D → nh,nw,kh,kw,D
        let out = v
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((nh * nw, kh * kw, d))?;
        outputs.push(out);
        offset += seq_len;
    }

    Ok(outputs)
}

// ─── Vision Tower ─────────────────────────────────────────────────────────────

/// MoonViT 3D pretrained vision encoder for Kimi-K2.5.
struct MoonViT3dPretrainedModel {
    patch_embed: MoonVision3dPatchEmbed,
    encoder: MoonViT3dEncoder,
    merge_kernel_size: [usize; 2],
    #[allow(dead_code)]
    cfg: KimiK25VisionConfig,
}

impl MoonViT3dPretrainedModel {
    fn new(cfg: &KimiK25VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = MoonVision3dPatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let encoder = MoonViT3dEncoder::new(cfg, vb.pp("encoder"))?;
        Ok(Self {
            patch_embed,
            encoder,
            merge_kernel_size: cfg.merge_kernel_size,
            cfg: cfg.clone(),
        })
    }

    /// Encode video/image patches through embed → transformer → tpool merger.
    ///
    /// `pixel_values`: `[L, C, ps, ps]` — pre-extracted patches, all items concatenated.
    /// `grid_thws`: `[[t, h, w], ...]` — frames, height, width per item.
    ///
    /// Returns one tensor per item: `[nh*nw, kh*kw, hidden]`.
    fn forward(&self, pixel_values: &Tensor, grid_thws: &[[usize; 3]]) -> Result<Vec<Tensor>> {
        let x = self.patch_embed.forward(pixel_values, grid_thws)?; // [L, D]
        let x = self.encoder.forward(&x, grid_thws)?; // [L, D]
        tpool_patch_merger(&x, grid_thws, self.merge_kernel_size)
    }
}

// ─── Multimodal Projector ────────────────────────────────────────────────────

/// Projects merged patch groups to LLM hidden size.
///
/// Input: `[N, kh*kw, vision_hidden]` — N merged tokens.
/// Output: `[N, text_hidden]`.
///
/// Weight path: `mm_projector.*` (HF checkpoint may ship `mm_projector.proj.{0,2}.*`,
/// which vLLM maps via `WeightsMapper` to `mm_projector.linear_{1,2}.*`).
struct KimiK25MultiModalProjector {
    pre_norm: LayerNorm, // LayerNorm(vision_hidden, eps=1e-5)
    linear_1: Linear,    // [D_flat → D_flat] bias=true, D_flat = vision_hidden * kh * kw
    linear_2: Linear,    // [D_flat → text_hidden] bias=true
    flat_dim: usize,
}

impl KimiK25MultiModalProjector {
    fn new(vision_cfg: &KimiK25VisionConfig, vb: VarBuilder) -> Result<Self> {
        let [kh, kw] = vision_cfg.merge_kernel_size;
        let flat_dim = vision_cfg.hidden_size * kh * kw;

        let pre_norm = layer_norm(vision_cfg.hidden_size, 1e-5, vb.pp("pre_norm"))?;
        let linear_1 = linear(flat_dim, flat_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(flat_dim, vision_cfg.mm_hidden_size, vb.pp("linear_2"))?;

        Ok(Self {
            pre_norm,
            linear_1,
            linear_2,
            flat_dim,
        })
    }

    /// `image_features`: `[N, kh*kw, vision_hidden]` → `[N, text_hidden]`.
    fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(image_features)?;
        let n = x.dim(0)?;
        let x = x.reshape((n, self.flat_dim))?;
        let x = self.linear_1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear_2.forward(&x)
    }
}

// ─── Main Config ─────────────────────────────────────────────────────────────

struct KimiK25Config {
    vision: KimiK25VisionConfig,
    image_token_id: u32,
}

impl KimiK25Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision = KimiK25VisionConfig::from_model_config(cfg);
        let image_token_id = cfg
            .extra
            .get("media_placeholder_token_id")
            .or_else(|| cfg.extra.get("image_token_id"))
            .and_then(|v| v.as_u64())
            .unwrap_or(151646) as u32;
        Self {
            vision,
            image_token_id,
        }
    }
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// Kimi-K2.5: MoonViT3d vision encoder + projector + DeepSeek-V2 language model.
pub struct KimiK25ForConditionalGeneration {
    vision_tower: MoonViT3dPretrainedModel,
    mm_projector: KimiK25MultiModalProjector,
    language_model: DeepSeekForCausalLM,
    #[allow(dead_code)]
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl KimiK25ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let k25_cfg = KimiK25Config::from_model_config(cfg);

        let vision_tower = MoonViT3dPretrainedModel::new(&k25_cfg.vision, vb.pp("vision_tower"))?;
        let mm_projector = KimiK25MultiModalProjector::new(&k25_cfg.vision, vb.pp("mm_projector"))?;
        let language_model = DeepSeekForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            mm_projector,
            language_model,
            image_token_id: k25_cfg.image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(cfg, vb)
    }

    /// Encode images/video chunks: vision tower → projector → `[total_tokens, text_hidden]`.
    pub fn encode_images(&self, pixel_values: &Tensor, grid_thws: &[[usize; 3]]) -> Result<Tensor> {
        let patches = self.vision_tower.forward(pixel_values, grid_thws)?;
        let all_patches = Tensor::cat(&patches, 0)?; // [total_merged, kh*kw, D]
        self.mm_projector.forward(&all_patches) // [total_merged, text_hidden]
    }

    /// Merge projected image features into text embeddings at image-token positions.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;
            for (i, row) in emb.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = row.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for KimiK25ForConditionalGeneration {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::mla_cache_config::MLACacheConfig;
    use candle_core::{DType, Device};
    use serde_json::json;

    /// Tiny config: small MoonViT3d + small DeepSeek-V2 text model.
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        // KimiK25-specific
        extra.insert("media_placeholder_token_id".to_string(), json!(5));

        // DeepSeek-V2 (MLA) text config fields
        extra.insert("kv_lora_rank".to_string(), json!(4));
        extra.insert("qk_nope_head_dim".to_string(), json!(8));
        extra.insert("qk_rope_head_dim".to_string(), json!(8));
        extra.insert("v_head_dim".to_string(), json!(8));
        extra.insert("q_lora_rank".to_string(), serde_json::Value::Null);
        extra.insert("num_experts".to_string(), json!(4));
        extra.insert("num_experts_per_tok".to_string(), json!(2));
        extra.insert("moe_intermediate_size".to_string(), json!(32));
        extra.insert("first_k_dense_replace".to_string(), json!(1));
        extra.insert("moe_layer_freq".to_string(), json!(1));
        extra.insert("n_shared_experts".to_string(), json!(1));
        extra.insert("routed_scaling_factor".to_string(), json!(1.0));
        extra.insert("norm_topk_prob".to_string(), json!(false));
        extra.insert("topk_method".to_string(), json!("greedy"));
        extra.insert("n_group".to_string(), json!(1));
        extra.insert("topk_group".to_string(), json!(1));
        extra.insert("scoring_func".to_string(), json!("softmax"));

        // Vision config (small)
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "in_channels": 1,
                "patch_size": 2,
                "init_pos_emb_height": 4,
                "init_pos_emb_width": 4,
                "init_pos_emb_time": 4,
                "merge_kernel_size": [2, 2]
            }),
        );

        ModelConfig {
            architectures: vec!["KimiK25ForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
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
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let kv_lora_rank = cfg
            .extra
            .get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let qk_rope_head_dim = cfg
            .extra
            .get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let qk_nope_head_dim = cfg
            .extra
            .get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let mla_cfg = MLACacheConfig::new(
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            cfg.num_attention_heads,
            16, // block_size
            8,  // num_blocks
            cfg.num_hidden_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_cfg).unwrap()
    }

    #[test]
    fn test_kimi_k25_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiK25ForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "KimiK25ForConditionalGeneration construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_tpool_merger_shape() {
        let device = Device::Cpu;
        // grid_thws = [[2, 4, 4]] → t=2 frames, h=4, w=4 → 32 patches
        // merge_kernel=(2,2): nh=2, nw=2
        let x = Tensor::zeros((32usize, 16usize), DType::F32, &device).unwrap();
        let grid_thws = [[2usize, 4usize, 4usize]];
        let out = tpool_patch_merger(&x, &grid_thws, [2, 2]).unwrap();
        assert_eq!(out.len(), 1);
        // nh*nw = 4, kh*kw = 4, D = 16
        assert_eq!(out[0].dims(), &[4, 4, 16]);
    }

    #[test]
    fn test_vision_tower_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiK25ForConditionalGeneration::new(&cfg, vb).unwrap();
        let vc = &model.vision_tower.cfg;

        // Single image (t=1), 4×4 grid = 16 patches; in_channels=1, patch_size=2
        let ps = vc.patch_size;
        let pixel_values =
            Tensor::zeros((16usize, vc.in_channels, ps, ps), DType::F32, &device).unwrap();
        let grid_thws = [[1usize, 4usize, 4usize]];

        let out = model
            .vision_tower
            .forward(&pixel_values, &grid_thws)
            .unwrap();
        assert_eq!(out.len(), 1);
        // After tpool_merge(2,2): nh=2, nw=2 → 4 merged tokens, each [kh*kw=4, D=16]
        assert_eq!(out[0].dims(), &[4, 4, 16]);
    }

    #[test]
    fn test_projector_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiK25ForConditionalGeneration::new(&cfg, vb).unwrap();

        // 4 merged tokens, each with 4 patches of dim 16
        let features = Tensor::zeros((4usize, 4usize, 16usize), DType::F32, &device).unwrap();
        let out = model.mm_projector.forward(&features).unwrap();
        // [4, text_hidden=64]
        assert_eq!(out.dims(), &[4, 64]);
    }

    #[test]
    fn test_text_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiK25ForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv = make_cache(&cfg, &device);

        let input_ids = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let block_table = crate::kv_cache::BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let out = model.forward(&input_ids, 0, &mut kv, &block_table, &slot_mapping);
        assert!(out.is_ok(), "text forward failed: {:?}", out.err());
        let out = out.unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.vocab_size]);
    }
}
