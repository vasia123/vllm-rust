//! Step3-VL vision-language model.
//!
//! Combines the Step3 Vision Transformer encoder with two Conv2d downsamplers
//! and a Linear projector, feeding into the Step3Text MoE language model.
//!
//! # Architecture
//!
//! ```text
//! Pixel [B, 3, 728, 728]
//!   → Step3VisionTransformer (63 layers, hidden=1792, heads=16, patch=14)
//!   → [B, 2708, 1792]  (3 TP-padding CLS copies + CLS + 2704 patches)
//!   → [:, 4:]          → [B, 2704, 1792]
//!   → permute + reshape [B, 1792, 52, 52]
//!   → vit_downsampler  Conv2d(1792→4096, kernel=2, stride=1)  → [B, 4096, 51, 51]
//!   → vit_downsampler2 Conv2d(4096→8192, kernel=3, stride=2, pad=1) → [B, 8192, 26, 26]
//!   → permute + reshape [B, 676, 8192]
//!   → vit_large_projector Linear(8192→lm_hidden)  → [B, 676, lm_hidden]
//!   → merge into LLM embedding stream at image-token positions
//!   → Step3TextForCausalLM → logits
//! ```
//!
//! # Weight paths (post HF→vLLM remap)
//!
//! - `vision_model.embeddings.*`         — ViT embeddings
//! - `vision_model.transformer.layers.*` — 63 ViT layers
//! - `vit_downsampler.*`                 — first Conv2d
//! - `vit_downsampler2.*`                — second Conv2d
//! - `vit_large_projector.*`             — Linear projection
//! - `language_model.model.*`            — Step3Text layers  (HF "model.*")
//! - `language_model.lm_head.*`          — LM head           (HF "lm_head.*")
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/step3_vl.py`
//! `reference/vllm/vllm/transformers_utils/configs/step3_vl.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d, embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, Conv2d,
    Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::step3_text::Step3TextForCausalLM;

// ─── Vision Config ───────────────────────────────────────────────────────────

/// Step3 vision encoder configuration.
#[derive(Debug, Clone)]
struct Step3VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    /// Output channels for first Conv2d downsampler.
    output_hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    image_size: usize,
    patch_size: usize,
    layer_norm_eps: f64,
}

impl Default for Step3VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1792,
            intermediate_size: 3072,
            output_hidden_size: 4096,
            num_hidden_layers: 63,
            num_attention_heads: 16,
            image_size: 728,
            patch_size: 14,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Step3VisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vc = cfg.extra.get("vision_config");
        let d = Self::default();
        Self {
            hidden_size: vc
                .and_then(|v| v.get("hidden_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.hidden_size as u64) as usize,
            intermediate_size: vc
                .and_then(|v| v.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.intermediate_size as u64) as usize,
            output_hidden_size: vc
                .and_then(|v| v.get("output_hidden_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.output_hidden_size as u64) as usize,
            num_hidden_layers: vc
                .and_then(|v| v.get("num_hidden_layers"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.num_hidden_layers as u64) as usize,
            num_attention_heads: vc
                .and_then(|v| v.get("num_attention_heads"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.num_attention_heads as u64) as usize,
            image_size: vc
                .and_then(|v| v.get("image_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.image_size as u64) as usize,
            patch_size: vc
                .and_then(|v| v.get("patch_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(d.patch_size as u64) as usize,
            layer_norm_eps: vc
                .and_then(|v| v.get("layer_norm_eps"))
                .and_then(|v| v.as_f64())
                .unwrap_or(d.layer_norm_eps),
        }
    }

    fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ─── Top-level VL Config ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Step3VLConfig {
    vision_config: Step3VisionConfig,
    /// Stride for the first Conv2d projector (default 1).
    understand_projector_stride: usize,
    /// Whether the linear projector includes a bias (default true).
    projector_bias: bool,
}

impl Step3VLConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        Self {
            vision_config: Step3VisionConfig::from_model_config(cfg),
            understand_projector_stride: cfg
                .extra
                .get("understand_projector_stride")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize,
            projector_bias: cfg
                .extra
                .get("projector_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        }
    }
}

// ─── Vision Attention ────────────────────────────────────────────────────────

struct Step3VisionAttention {
    qkv_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Step3VisionAttention {
    fn new(vc: &Step3VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = vc.hidden_size;
        let heads = vc.num_attention_heads;
        let head_dim = vc.head_dim();
        let qkv_proj = linear(hidden, 3 * hidden, vb.pp("qkv_proj"))?;
        let out_proj = linear(hidden, hidden, vb.pp("out_proj"))?;
        Ok(Self {
            qkv_proj,
            out_proj,
            num_heads: heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let d = self.num_heads * self.head_dim;

        let qkv = self.qkv_proj.forward(xs)?; // [B, T, 3D]
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        // Reshape to [B, H, T, D_h]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        // Scaled dot-product attention (no causal mask — encoder)
        // Note: transpose produces non-contiguous tensor; must call .contiguous()
        let attn = q
            .affine(self.scale, 0.0)?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?; // [B, H, T, T]
        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?; // [B, H, T, D_h]

        // Reshape back to [B, T, D]
        let out = out
            .permute((0, 2, 1, 3))?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

// ─── Vision MLP ──────────────────────────────────────────────────────────────

struct Step3VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl Step3VisionMLP {
    fn new(vc: &Step3VisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(vc.hidden_size, vc.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(vc.intermediate_size, vc.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // QuickGELU: x * sigmoid(1.702 * x)
        let h = self.fc1.forward(xs)?;
        let h = candle_nn::ops::sigmoid(&h.affine(1.702, 0.0)?)?.mul(&h)?;
        self.fc2.forward(&h)
    }
}

// ─── Vision Encoder Layer ────────────────────────────────────────────────────

struct Step3VisionEncoderLayer {
    self_attn: Step3VisionAttention,
    layer_norm1: LayerNorm,
    mlp: Step3VisionMLP,
    layer_norm2: LayerNorm,
}

impl Step3VisionEncoderLayer {
    fn new(vc: &Step3VisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Step3VisionAttention::new(vc, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(vc.hidden_size, vc.layer_norm_eps, vb.pp("layer_norm1"))?;
        let mlp = Step3VisionMLP::new(vc, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(vc.hidden_size, vc.layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pre-norm: attn branch, then MLP branch
        let xs = xs.add(&self.self_attn.forward(&self.layer_norm1.forward(xs)?)?)?;
        let xs = xs.add(&self.mlp.forward(&self.layer_norm2.forward(&xs)?)?)?;
        Ok(xs)
    }
}

// ─── Vision Embeddings ───────────────────────────────────────────────────────

struct Step3VisionEmbeddings {
    patch_embedding: Conv2d,
    class_embedding: Tensor,
    position_embedding: Embedding,
    num_positions: usize,
    hidden_size: usize,
}

impl Step3VisionEmbeddings {
    fn new(vc: &Step3VisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_patches = vc.num_patches();
        let num_positions = num_patches + 1; // CLS + patches

        let patch_cfg = Conv2dConfig {
            stride: vc.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            3,
            vc.hidden_size,
            vc.patch_size,
            patch_cfg,
            vb.pp("patch_embedding"),
        )?;

        // class_embedding: [1, hidden_size] learnable parameter
        let class_embedding = vb.get((1, vc.hidden_size), "class_embedding")?;

        let position_embedding =
            embedding(num_positions, vc.hidden_size, vb.pp("position_embedding"))?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            num_positions,
            hidden_size: vc.hidden_size,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        let device = pixel_values.device();

        // Patch embedding: [B, D, H', W'] → permute → [B, H'*W', D]
        let p = self.patch_embedding.forward(pixel_values)?;
        let (_, c, h, w) = p.dims4()?;
        let patch_embeds = p.permute((0, 2, 3, 1))?.reshape((b, h * w, c))?;

        // Prepend CLS token: class_embedding is [1, D], expand to [B, 1, D]
        let cls = self
            .class_embedding
            .unsqueeze(0)? // [1, 1, D]
            .broadcast_as((b, 1, self.hidden_size))?;
        let embeddings = Tensor::cat(&[&cls, &patch_embeds], 1)?; // [B, 2705, D]

        // Add position embedding
        let pos_ids = Tensor::arange(0u32, self.num_positions as u32, device)?.unsqueeze(0)?; // [1, 2705]
        let pos_embed = self.position_embedding.forward(&pos_ids)?; // [1, 2705, D]
        let embeddings = embeddings.broadcast_add(&pos_embed)?; // [B, 2705, D]

        // TP-padding: prepend 3 copies of CLS token (pad_tp_size - 1 = 3)
        // Vision model produces 2708 tokens; downstream drops first 4 → 2704 patches
        let pad = embeddings.narrow(1, 0, 1)?.repeat(&[1, 3, 1])?; // [B, 3, D]
        let embeddings = Tensor::cat(&[&pad, &embeddings], 1)?; // [B, 2708, D]
        Ok(embeddings)
    }
}

// ─── Vision Transformer ──────────────────────────────────────────────────────

struct Step3VisionTransformer {
    embeddings: Step3VisionEmbeddings,
    layers: Vec<Step3VisionEncoderLayer>,
}

impl Step3VisionTransformer {
    fn new(vc: &Step3VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = Step3VisionEmbeddings::new(vc, vb.pp("embeddings"))?;
        let vb_layers = vb.pp("transformer").pp("layers");
        let layers = (0..vc.num_hidden_layers)
            .map(|i| Step3VisionEncoderLayer::new(vc, vb_layers.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { embeddings, layers })
    }

    /// Returns raw encoder output including TP-padding tokens: [B, 2708, D]
    fn forward_raw(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }

    /// Encode pixel_values to patch features: [B, num_patches, D]
    ///
    /// Drops the 4 leading TP-padding + CLS tokens, leaving only the spatial patches.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let xs = self.forward_raw(pixel_values)?;
        // Drop first 4 tokens (3 CLS pads + CLS); keep num_patches = 2704 spatial tokens
        let total = xs.dim(1)?;
        xs.narrow(1, 4, total - 4)
    }
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// Step3-VL: vision-language model combining Step3 ViT + Step3Text LLM.
pub struct Step3VLForConditionalGeneration {
    vision_model: Step3VisionTransformer,
    vit_downsampler: Conv2d,
    vit_downsampler2: Conv2d,
    vit_large_projector: Linear,
    language_model: Step3TextForCausalLM,
    vl_cfg: Step3VLConfig,
    device: Device,
    dtype: DType,
}

impl Step3VLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vl_cfg = Step3VLConfig::from_model_config(cfg);
        let vc = &vl_cfg.vision_config;

        let vision_model = Step3VisionTransformer::new(vc, vb.pp("vision_model"))?;

        let ds1_cfg = Conv2dConfig {
            stride: vl_cfg.understand_projector_stride,
            ..Default::default()
        };
        let vit_downsampler = conv2d(
            vc.hidden_size,
            vc.output_hidden_size,
            2,
            ds1_cfg,
            vb.pp("vit_downsampler"),
        )?;

        let ds2_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let vit_downsampler2 = conv2d(
            vc.output_hidden_size,
            vc.output_hidden_size * 2,
            3,
            ds2_cfg,
            vb.pp("vit_downsampler2"),
        )?;

        let vit_large_projector = if vl_cfg.projector_bias {
            linear(
                vc.output_hidden_size * 2,
                cfg.hidden_size,
                vb.pp("vit_large_projector"),
            )?
        } else {
            linear_no_bias(
                vc.output_hidden_size * 2,
                cfg.hidden_size,
                vb.pp("vit_large_projector"),
            )?
        };

        let language_model = Step3TextForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
            vit_downsampler,
            vit_downsampler2,
            vit_large_projector,
            language_model,
            vl_cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Process vision encoder output through the two Conv2d downsamplers and linear projector.
    ///
    /// Input:  `[B, num_patches, vision_hidden]` — e.g. [B, 2704, 1792]
    /// Output: `[B, projected_tokens, lm_hidden]` — e.g. [B, 676, 7168]
    fn process_image_features(&self, image_features: &Tensor) -> Result<Tensor> {
        let vc = &self.vl_cfg.vision_config;
        let (b, p, _) = image_features.dims3()?;
        let hw = (p as f64).sqrt() as usize;

        // [B, P, D] → [B, D, HW, HW]
        let features = image_features
            .permute((0, 2, 1))?
            .reshape((b, vc.hidden_size, hw, hw))?;

        // Spatial downsampling
        let features = self.vit_downsampler.forward(&features)?;
        let features = self.vit_downsampler2.forward(&features)?;

        // Flatten spatial → sequence: [B, C, H', W'] → [B, H'*W', C]
        let (b2, c2, h2, w2) = features.dims4()?;
        let features = features.permute((0, 2, 3, 1))?.reshape((b2, h2 * w2, c2))?;

        // Project to LM hidden size
        self.vit_large_projector.forward(&features)
    }

    /// Extract image features from raw pixel values.
    ///
    /// pixel_values: [B, 3, H, W]
    /// Returns: [B, num_projected_tokens, lm_hidden]
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let patch_features = self.vision_model.forward(pixel_values)?;
        self.process_image_features(&patch_features)
    }

    /// Merge pre-projected image embeddings into text embeddings at image-token positions.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
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

impl crate::engine::ModelForward for Step3VLForConditionalGeneration {
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
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use serde_json::json;

    /// Minimal config: tiny vision encoder + tiny Step3Text LLM.
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Step3Text fields (small for tests)
        extra.insert("moe_num_experts".to_string(), json!(4));
        extra.insert("moe_top_k".to_string(), json!(2));
        extra.insert("moe_intermediate_size".to_string(), json!(32));
        extra.insert("share_expert_dim".to_string(), json!(32));
        extra.insert("share_q_dim".to_string(), json!(16));
        extra.insert("norm_expert_weight".to_string(), json!(false));
        extra.insert("max_position_embedding".to_string(), json!(512));
        // Layer 0 = dense, layer 1 = MoE
        extra.insert("moe_layers_enum".to_string(), json!("1"));

        // Step3VL fields
        extra.insert("image_token_id".to_string(), json!(128001));
        extra.insert("understand_projector_stride".to_string(), json!(1));
        extra.insert("projector_bias".to_string(), json!(true));

        // Small vision config: 28×28 image, patch=14 → 2×2=4 patches
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "output_hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 28,
                "patch_size": 14,
                "layer_norm_eps": 1e-5
            }),
        );

        ModelConfig {
            architectures: vec!["Step3VLForConditionalGeneration".to_string()],
            hidden_size: 64, // LLM hidden = projector output
            num_attention_heads: 4,
            num_key_value_heads: 1, // Step3Text uses 1 KV head
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
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: 1, // Step3Text single KV head
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_step3_vl_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3VLForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Step3VLForConditionalGeneration construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_step3_vl_vision_encoder() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3VLForConditionalGeneration::new(&cfg, vb).unwrap();
        let vc = &model.vl_cfg.vision_config;

        // pixel_values: [1, 3, 28, 28]
        let pixel_values = Tensor::zeros(
            (1usize, 3usize, vc.image_size, vc.image_size),
            DType::F32,
            &device,
        )
        .unwrap();

        let features = model.vision_model.forward(&pixel_values);
        assert!(
            features.is_ok(),
            "vision encoder failed: {:?}",
            features.err()
        );
        // [1, 4, 32]: 4 patches (2×2), 32 = vision hidden
        let f = features.unwrap();
        assert_eq!(f.dims()[0], 1);
        assert_eq!(f.dims()[1], vc.num_patches());
        assert_eq!(f.dims()[2], vc.hidden_size);
    }

    #[test]
    fn test_step3_vl_process_image() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3VLForConditionalGeneration::new(&cfg, vb).unwrap();
        let vc = &model.vl_cfg.vision_config;

        // Simulate vision encoder output: [1, 4, 32]
        let vision_out = Tensor::zeros(
            (1usize, vc.num_patches(), vc.hidden_size),
            DType::F32,
            &device,
        )
        .unwrap();

        let projected = model.process_image_features(&vision_out);
        assert!(
            projected.is_ok(),
            "process_image_features failed: {:?}",
            projected.err()
        );
        let p = projected.unwrap();
        // After two Conv2d + flatten + Linear → [1, N, 64]
        assert_eq!(p.dims()[0], 1);
        assert_eq!(p.dims()[2], cfg.hidden_size); // = 64
    }

    #[test]
    fn test_step3_vl_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3VLForConditionalGeneration::new(&cfg, vb).unwrap();
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
    fn test_step3_vl_encode_images() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3VLForConditionalGeneration::new(&cfg, vb).unwrap();
        let vc = &model.vl_cfg.vision_config;

        let pixel_values = Tensor::zeros(
            (2usize, 3usize, vc.image_size, vc.image_size),
            DType::F32,
            &device,
        )
        .unwrap();

        let result = model.encode_images(&pixel_values);
        assert!(result.is_ok(), "encode_images failed: {:?}", result.err());
        let r = result.unwrap();
        // Batch=2, some number of tokens, hidden=64
        assert_eq!(r.dims()[0], 2);
        assert_eq!(r.dims()[2], cfg.hidden_size);
    }
}
