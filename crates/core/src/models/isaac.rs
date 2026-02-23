//! Isaac Vision-Language Model.
//!
//! ```text
//! pixel_values [N, in_ch*ps²]    image_grid_thw [ni, 3=(T,H,W)]
//!
//! → Siglip2VisionTransformer
//!     embeddings: Linear patch_embed (no bias) + bilinear-interp position_embedding
//!     encoder × depth: pre-norm [ln1 → attn → res] + [ln2 → MLP → res]
//!     post_layernorm
//!     pixel_shuffle: [N, D] → [N/scale², D*scale²]
//!   → [N/scale², D*scale²]
//!
//! → IsaacVisionEmbedding (projector)
//!     linear_fc1: no bias, (D*scale², 4*D*scale²)
//!     SiLU
//!     linear_fc2: no bias, (4*D*scale², llm_hidden)
//!   → [N/scale², llm_hidden]
//!
//! → merge_multimodal → Qwen3ForCausalLM → logits [B, S, vocab]
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/isaac.py`

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
struct IsaacVisionConfig {
    patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    num_heads: usize,
    intermediate_size: usize,
    depth: usize,
    /// Square root of this is the baseline pos_emb grid size.
    num_patches: usize,
    pixel_shuffle_scale: usize,
    layer_norm_eps: f64,
}

impl IsaacVisionConfig {
    fn from_json(json: &serde_json::Value) -> Self {
        let patch_size = json
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
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
        let num_patches = json
            .get("num_patches")
            .and_then(|v| v.as_u64())
            .unwrap_or(729) as usize;
        let pixel_shuffle_scale = json
            .get("pixel_shuffle_scale_factor")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
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
            num_patches,
            pixel_shuffle_scale,
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

// ─── Pixel Shuffle ────────────────────────────────────────────────────────────

/// Merge `scale²` neighboring patches into one token with expanded channels.
///
/// - `x`: `[h*w, D]` — flat patch sequence for one image
/// - `(h, w)`: grid dimensions (both must be divisible by `scale`)
///
/// Returns `[h*w / scale², D * scale²]`.
fn pixel_shuffle(x: &Tensor, h: usize, w: usize, scale: usize) -> Result<Tensor> {
    let d = x.dim(1)?;
    // [h, w, D] → [h/s, s, w/s, s, D] → permute(0,2,1,3,4) → [h/s, w/s, s, s, D]
    // → [h*w/s², s²*D]
    let x = x.reshape((h, w, d))?;
    let s = scale;
    let x = x.reshape((h / s, s, w / s, s, d))?;
    let x = x.permute((0, 2, 1, 3, 4))?.contiguous()?;
    x.reshape((h * w / (s * s), d * s * s))
}

// ─── Positional Embedding Interpolation ──────────────────────────────────────

/// Bilinear interpolation of a square positional embedding grid.
///
/// Replicates `F.interpolate(..., mode='bilinear', align_corners=False)` (no antialias).
/// When the target size equals the source size the input is returned unchanged.
fn bilinear_interp_pos_emb(
    pos_emb: &Tensor, // [num_patches, D]
    tgt_h: usize,
    tgt_w: usize,
) -> Result<Tensor> {
    let num_patches = pos_emb.dim(0)?;
    let d = pos_emb.dim(1)?;
    let src_size = (num_patches as f64).sqrt().round() as usize;

    if tgt_h == src_size && tgt_w == src_size {
        return Ok(pos_emb.clone());
    }

    // Pull to CPU F32 for the inner loop.
    let pe = pos_emb.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let pe_data = pe.to_vec2::<f32>()?;

    let mut out = vec![0f32; tgt_h * tgt_w * d];
    for ty in 0..tgt_h {
        for tx in 0..tgt_w {
            // align_corners=False source coordinate
            let sy = ((ty as f32 + 0.5) * src_size as f32 / tgt_h as f32 - 0.5)
                .clamp(0.0, (src_size - 1) as f32);
            let sx = ((tx as f32 + 0.5) * src_size as f32 / tgt_w as f32 - 0.5)
                .clamp(0.0, (src_size - 1) as f32);
            let sy0 = sy.floor() as usize;
            let sy1 = (sy0 + 1).min(src_size - 1);
            let sx0 = sx.floor() as usize;
            let sx1 = (sx0 + 1).min(src_size - 1);
            let wy1 = sy - sy0 as f32;
            let wy0 = 1.0 - wy1;
            let wx1 = sx - sx0 as f32;
            let wx0 = 1.0 - wx1;
            let dst = (ty * tgt_w + tx) * d;
            for k in 0..d {
                let v00 = pe_data[sy0 * src_size + sx0][k];
                let v01 = pe_data[sy0 * src_size + sx1][k];
                let v10 = pe_data[sy1 * src_size + sx0][k];
                let v11 = pe_data[sy1 * src_size + sx1][k];
                out[dst + k] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
            }
        }
    }
    let result = Tensor::from_vec(out, (tgt_h * tgt_w, d), pos_emb.device())?;
    result.to_dtype(pos_emb.dtype())
}

// ─── Vision Embeddings ────────────────────────────────────────────────────────

struct Siglip2Embeddings {
    patch_embedding: Linear,       // no bias: (patch_dim → embed_dim)
    position_embedding: Embedding, // (num_patches, embed_dim)
    pos_emb_size: usize,           // sqrt(num_patches)
}

impl Siglip2Embeddings {
    fn new(cfg: &IsaacVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = linear_no_bias(
            cfg.patch_input_dim(),
            cfg.embed_dim,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding =
            embedding(cfg.num_patches, cfg.embed_dim, vb.pp("position_embedding"))?;
        let pos_emb_size = (cfg.num_patches as f64).sqrt().round() as usize;
        Ok(Self {
            patch_embedding,
            position_embedding,
            pos_emb_size,
        })
    }

    /// `patches`: `[N, patch_dim]`; `grid`: `(h, w)`.
    ///
    /// Returns `[N, embed_dim]`.
    fn forward(&self, patches: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let (h, w) = grid;
        let patch_embeds = self.patch_embedding.forward(patches)?;
        let pos_emb = bilinear_interp_pos_emb(
            self.position_embedding.embeddings(),
            if h == 0 { self.pos_emb_size } else { h },
            if w == 0 { self.pos_emb_size } else { w },
        )?;
        patch_embeds.broadcast_add(&pos_emb)
    }
}

// ─── Vision MLP ───────────────────────────────────────────────────────────────

struct Siglip2MLP {
    fc1: Linear, // with bias
    fc2: Linear, // with bias
}

impl Siglip2MLP {
    fn new(cfg: &IsaacVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.embed_dim, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.embed_dim, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?; // gelu_pytorch_tanh ≈ erf-GELU for SigLIP-2
        self.fc2.forward(&x)
    }
}

// ─── Vision Attention ─────────────────────────────────────────────────────────

/// Pre-norm SigLIP-2 attention.
///
/// HF checkpoint stores separate `q_proj`, `k_proj`, `v_proj` which are loaded
/// individually and concatenated into a fused projection at construction time.
struct Siglip2Attention {
    q_proj: Linear,   // with bias
    k_proj: Linear,   // with bias
    v_proj: Linear,   // with bias
    out_proj: Linear, // with bias
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Siglip2Attention {
    fn new(cfg: &IsaacVisionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_proj: linear(d, d, vb.pp("q_proj"))?,
            k_proj: linear(d, d, vb.pp("k_proj"))?,
            v_proj: linear(d, d, vb.pp("v_proj"))?,
            out_proj: linear(d, d, vb.pp("out_proj"))?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[N, D]` — flat patch sequence.
    ///
    /// Returns `[N, D]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        // [N, heads, head_dim]
        let q = q
            .reshape((n, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((n, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((n, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        // SDPA: [heads, N, N] → [heads, N, head_dim] → [N, heads*head_dim]
        let attn = (q.matmul(&k.transpose(1, 2)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [heads, N, head_dim]
        let out = out
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

// ─── Encoder Layer ────────────────────────────────────────────────────────────

struct Siglip2EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Siglip2Attention,
    layer_norm2: LayerNorm,
    mlp: Siglip2MLP,
}

impl Siglip2EncoderLayer {
    fn new(cfg: &IsaacVisionConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: layer_norm(cfg.embed_dim, eps, vb.pp("layer_norm1"))?,
            self_attn: Siglip2Attention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.embed_dim, eps, vb.pp("layer_norm2"))?,
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

// ─── Encoder ─────────────────────────────────────────────────────────────────

struct Siglip2Encoder {
    layers: Vec<Siglip2EncoderLayer>,
}

impl Siglip2Encoder {
    fn new(cfg: &IsaacVisionConfig, vb: VarBuilder) -> Result<Self> {
        let layers = (0..cfg.depth)
            .map(|i| Siglip2EncoderLayer::new(cfg, vb.pp("layers").pp(i)))
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

// ─── Vision Transformer ───────────────────────────────────────────────────────

struct Siglip2VisionTransformer {
    embeddings: Siglip2Embeddings,
    encoder: Siglip2Encoder,
    post_layernorm: LayerNorm,
    pixel_shuffle_scale: usize,
}

impl Siglip2VisionTransformer {
    fn new(cfg: &IsaacVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: Siglip2Embeddings::new(cfg, vb.pp("embeddings"))?,
            encoder: Siglip2Encoder::new(cfg, vb.pp("encoder"))?,
            post_layernorm: layer_norm(cfg.embed_dim, cfg.layer_norm_eps, vb.pp("post_layernorm"))?,
            pixel_shuffle_scale: cfg.pixel_shuffle_scale,
        })
    }

    /// `patches`: `[N, patch_dim]`; `grid`: `(h, w)`.
    ///
    /// Returns `[N / scale², embed_dim * scale²]` after pixel-shuffle compression.
    fn forward(&self, patches: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let (h, w) = grid;
        let x = self.embeddings.forward(patches, (h, w))?; // [N, D]
        let x = self.encoder.forward(&x)?; // [N, D]
        let x = self.post_layernorm.forward(&x)?; // [N, D]
        let s = self.pixel_shuffle_scale;
        if s > 1 {
            pixel_shuffle(&x, h, w, s)
        } else {
            Ok(x)
        }
    }
}

// ─── Isaac Vision Embedding (Projector) ──────────────────────────────────────

struct IsaacVisionEmbedding {
    transformer: Siglip2VisionTransformer,
    linear_fc1: Linear, // no bias: (D*scale², 4*D*scale²)
    linear_fc2: Linear, // no bias: (4*D*scale², llm_hidden)
}

impl IsaacVisionEmbedding {
    fn new(vis_cfg: &IsaacVisionConfig, llm_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim =
            vis_cfg.embed_dim * vis_cfg.pixel_shuffle_scale * vis_cfg.pixel_shuffle_scale;
        Ok(Self {
            transformer: Siglip2VisionTransformer::new(vis_cfg, vb.pp("transformer"))?,
            linear_fc1: linear_no_bias(hidden_dim, 4 * hidden_dim, vb.pp("linear_fc1"))?,
            linear_fc2: linear_no_bias(4 * hidden_dim, llm_hidden, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, patches: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let x = self.transformer.forward(patches, grid)?; // [N/scale², D*scale²]
        let x = self.linear_fc1.forward(&x)?;
        let x = candle_nn::ops::silu(&x)?;
        self.linear_fc2.forward(&x)
    }
}

// ─── Multimodal Merging ───────────────────────────────────────────────────────

/// Replace image-pad positions in `text_embeds` with rows from `mm_inputs`.
fn merge_multimodal(
    text_embeds: &Tensor, // [B, S, D]
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

/// Isaac vision-language model.
///
/// Architecture: Siglip2VisionTransformer + pixel-shuffle projector + Qwen3.
pub struct IsaacForConditionalGeneration {
    vision_embedding: IsaacVisionEmbedding,
    language_model: Qwen3ForCausalLM,
    device: Device,
}

impl IsaacForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let vis_cfg = IsaacVisionConfig::from_json(&vis_json);

        let vision_embedding =
            IsaacVisionEmbedding::new(&vis_cfg, cfg.hidden_size, vb.pp("vision_embedding"))?;
        let language_model = Qwen3ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_embedding,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Encode image patches: `patches [N, in_ch*ps²]`, `grid (h, w)`.
    ///
    /// Returns `[N / scale², llm_hidden_size]`.
    pub fn encode_images(&self, patches: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        self.vision_embedding.forward(patches, grid)
    }
}

impl crate::engine::ModelForward for IsaacForConditionalGeneration {
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

    /// Minimal config: patch_size=2, in_ch=1, embed_dim=16, heads=2,
    /// intermediate=8, depth=1, num_patches=4, scale=2.
    /// LLM: hidden=32, 2 layers.
    fn test_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            json!({
                "patch_size": 2,
                "num_channels": 1,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_patches": 4,
                "pixel_shuffle_scale_factor": 2,
                "layer_norm_eps": 1e-6
            }),
        );
        extra.insert("image_token_id".to_string(), json!(9u32));
        ModelConfig {
            architectures: vec!["IsaacForConditionalGeneration".to_string()],
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 64,
            max_position_embeddings: 512,
            head_dim: 8,
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
    fn test_isaac_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = IsaacForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_isaac_vision_encode() {
        // patch_size=2, in_ch=1 → patch_dim=4; grid (2,2) → 4 patches
        // pixel_shuffle scale=2: 4 patches → 1 token of dim 16*4=64
        // linear_fc1: 64→256, linear_fc2: 256→32 → output [1, 32]
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = IsaacForConditionalGeneration::new(&cfg, vb).unwrap();
        let patches = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let feats = model.encode_images(&patches, (2, 2)).unwrap();
        assert_eq!(feats.dims(), &[1, 32]);
    }

    #[test]
    fn test_isaac_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = IsaacForConditionalGeneration::new(&cfg, vb).unwrap();

        let input_ids = Tensor::from_slice(&[0u32, 1, 2, 3], (1usize, 4usize), &device).unwrap();
        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 4).unwrap();
        let slot_mapping = bt.slot_mapping(0, 4);

        let result = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_isaac_with_image() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = IsaacForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode an image: 4 patches → 1 token of dim 32
        let patches = Tensor::zeros((4usize, 4usize), DType::F32, &device).unwrap();
        let img_feats = model.encode_images(&patches, (2, 2)).unwrap();
        let processed = ProcessedImage::new(img_feats, 1);

        // Sequence: 4 tokens; image pad at position 1
        let input_ids = Tensor::from_slice(&[0u32, 9, 2, 3], (1usize, 4usize), &device).unwrap();
        let mm = MultimodalInputs::with_images(vec![0u32, 9, 2, 3], vec![(1, processed)]);

        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 4).unwrap();
        let slot_mapping = bt.slot_mapping(0, 4);

        let result =
            model.forward_multimodal(&input_ids, Some(&mm), 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "with_image forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_isaac_decode_batch() {
        use crate::engine::DecodeSequenceMetadata;

        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = IsaacForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 4).unwrap();
        let slot = bt.slot_mapping(3, 1);

        let seq = DecodeSequenceMetadata {
            request_id: 0,
            seqlen_offset: 3,
            slot_mapping: slot,
            block_ids: bt.block_ids().to_vec(),
        };
        let tok = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let result = model.forward_decode_batch(&tok, &[seq], &mut kv);
        assert!(result.is_ok(), "decode_batch failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, 1, 64]);
    }
}
