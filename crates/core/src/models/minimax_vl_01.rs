//! MiniMax-VL-01 vision-language model.
//!
//! Architecture: CLIP/SigLIP vision encoder → 2-layer MLP projector →
//! MiniMaxText01 backbone (hybrid linear attention + standard attention).
//!
//! Weight paths (matching HuggingFace checkpoint layout):
//!
//! ```text
//! vision_tower.*                    — CLIP/SigLIP encoder
//! multi_modal_projector.linear_1.*  — vision_hidden → text_hidden (with bias)
//! multi_modal_projector.linear_2.*  — text_hidden → text_hidden (with bias)
//! image_newline                     — learnable [hidden_size] appended to each image
//! language_model.model.embed_tokens — token embeddings
//! language_model.model.layers.*    — MiniMaxText01 decoder layers
//! language_model.model.norm        — final RMSNorm
//! language_model.lm_head          — output projection (tied or separate)
//! ```
//!
//! Reference: reference/vllm/vllm/model_executor/models/minimax_vl_01.py
//!
//! ## AnyRes note
//!
//! The Python implementation supports multi-resolution (AnyRes) images via
//! `pack_image_features`. In the Rust preprocessing pipeline the per-image
//! sub-patch information is not yet surfaced through `MultimodalInputs`, so
//! only the single-image path (base image + `image_newline` append) is
//! implemented here. AnyRes can be wired once the preprocessor exposes the
//! required metadata.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::minimax_text01::{MiniMaxText01Config, MiniMaxText01DecoderLayer};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MiniMaxVL01Config {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    image_token_index: u32,
}

impl MiniMaxVL01Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig::clip_vit_l_14_336();

        let vision_config = if let Some(vc) = cfg.extra.get("vision_config") {
            let encoder_type = match vc
                .get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("clip")
            {
                "siglip_vision_model" | "siglip" => VisionEncoderType::SigLip,
                _ => VisionEncoderType::Clip,
            };

            VisionEncoderConfig {
                encoder_type,
                hidden_size: vc
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.hidden_size as u64) as usize,
                intermediate_size: vc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.intermediate_size as u64)
                    as usize,
                num_attention_heads: vc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_attention_heads as u64)
                    as usize,
                num_hidden_layers: vc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_hidden_layers as u64)
                    as usize,
                image_size: vc
                    .get("image_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.image_size as u64) as usize,
                patch_size: vc
                    .get("patch_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.patch_size as u64) as usize,
                num_channels: vc
                    .get("num_channels")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_channels as u64) as usize,
                layer_norm_eps: vc
                    .get("layer_norm_eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(defaults.layer_norm_eps),
            }
        } else {
            defaults
        };

        let image_token_index = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            image_token_index,
        }
    }
}

// ─── Multi-Modal Projector ──────────────────────────────────────────────────

/// 2-layer MLP projector bridging vision and language spaces.
///
/// Matches Python's `MiniMaxVL01MultiModalProjector`:
///   linear_1 (vision_hidden → text_hidden, with bias)
///   GELU activation
///   linear_2 (text_hidden → text_hidden, with bias)
struct MiniMaxVL01MultiModalProjector {
    linear_1: Linear,
    linear_2: Linear,
}

impl MiniMaxVL01MultiModalProjector {
    fn new(vision_hidden_size: usize, text_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = linear(vision_hidden_size, text_hidden_size, vb.pp("linear_1"))?;
        let linear_2 = linear(text_hidden_size, text_hidden_size, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let hidden = self.linear_1.forward(image_features)?;
        let hidden = hidden.gelu_erf()?;
        self.linear_2.forward(&hidden)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// MiniMax-VL-01 vision-language model.
///
/// Combines a CLIP/SigLIP vision encoder with the MiniMaxText01 hybrid
/// attention backbone. The vision encoder runs without the final post-layernorm
/// (matching `require_post_norm=False` in the Python reference).
pub struct MiniMaxVL01ForConditionalGeneration {
    // Vision
    #[allow(dead_code)]
    vision_tower: VisionEncoder,
    multi_modal_projector: MiniMaxVL01MultiModalProjector,
    /// Learnable [hidden_size] token appended after each image's projected features.
    image_newline: Tensor,
    // LLM backbone (owned directly to support multimodal embedding injection)
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMaxText01DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    // Metadata
    #[allow(dead_code)]
    config: MiniMaxVL01Config,
    device: Device,
    dtype: DType,
    num_attn_layers: usize,
}

impl MiniMaxVL01ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = MiniMaxVL01Config::from_model_config(cfg);

        // Vision tower: no post-layernorm (require_post_norm=False in Python reference).
        let vision_tower = VisionEncoder::new(&config.vision_config, vb.pp("vision_tower"))?;

        let multi_modal_projector = MiniMaxVL01MultiModalProjector::new(
            config.vision_config.hidden_size,
            cfg.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;

        let image_newline = vb.get(cfg.hidden_size, "image_newline")?;

        // Language model backbone. The HF checkpoint stores it under "language_model".
        let vb_lm = vb.pp("language_model");
        let vb_model = vb_lm.pp("model");

        let minimax_cfg = MiniMaxText01Config::from_model_config(cfg);
        let num_linear_layers = minimax_cfg.num_linear_layers();

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        let mut attn_layer_count = 0usize;
        let mut linear_layer_count = 0usize;

        for i in 0..cfg.num_hidden_layers {
            let (cache_layer_idx, linear_layer_idx) = if minimax_cfg.is_linear_attention(i) {
                let idx = linear_layer_count;
                linear_layer_count += 1;
                (None, idx)
            } else {
                let idx = attn_layer_count;
                attn_layer_count += 1;
                (Some(idx), 0)
            };

            layers.push(MiniMaxText01DecoderLayer::new(
                cfg,
                &minimax_cfg,
                i,
                cache_layer_idx,
                num_linear_layers,
                linear_layer_idx,
                vb_layers.pp(i),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_lm.pp("lm_head"))?
        };

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            image_newline,
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            num_attn_layers: attn_layer_count,
        })
    }

    /// Get the number of full-attention layers (for KV cache sizing).
    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }

    /// Project raw vision features into text space and append the image_newline token.
    ///
    /// Input: `[num_vision_tokens, vision_hidden]` (raw encoder output before post-norm)
    /// Output: `[num_vision_tokens + 1, text_hidden]`
    fn project_image(&self, vision_features: &Tensor) -> Result<Tensor> {
        let projected = self.multi_modal_projector.forward(vision_features)?;
        // Append image_newline: unsqueeze to [1, text_hidden] and cat.
        let newline = self
            .image_newline
            .unsqueeze(0)?
            .to_dtype(projected.dtype())?;
        Tensor::cat(&[projected, newline], 0)
    }

    /// Merge pre-processed vision embeddings into text embeddings at image placeholder positions.
    ///
    /// `mm_inputs.image_embeddings` stores raw vision encoder outputs (i.e. the output
    /// of `VisionEncoder::forward_no_post_norm`). This method projects them through the
    /// MLP projector, appends `image_newline`, and replaces the corresponding positions
    /// in the text embedding sequence.
    fn merge_multimodal_embeddings(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            let projected = self.project_image(&processed_image.embedding)?;
            let img_emb: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in img_emb.iter().enumerate() {
                let target_pos = start_pos + i;
                if target_pos < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target_pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    /// Core forward pass: embed tokens, optionally merge vision features, run layers.
    pub fn forward_inner(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let text_embeddings = self.embed_tokens.forward(input_ids)?;

        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(&text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
        } else {
            text_embeddings
        };

        for layer in self.layers.iter() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl crate::engine::ModelForward for MiniMaxVL01ForConditionalGeneration {
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
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for layer in self.layers.iter() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr)?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
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
        self.forward_inner(
            input_ids,
            multimodal_inputs,
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    /// Minimal config: tiny MiniMax backbone + tiny CLIP vision encoder.
    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "model_type": "clip_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "image_size": 28,
                "patch_size": 14,
                "layer_norm_eps": 1e-5
            }),
        );
        extra.insert("image_token_index".to_string(), serde_json::json!(32000u32));
        // MiniMaxText01 specific fields
        extra.insert(
            "attn_type_list".to_string(),
            serde_json::json!([0u8, 1u8]), // layer 0: linear, layer 1: full
        );

        ModelConfig {
            architectures: vec!["MiniMaxVL01ForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 128,
            max_position_embeddings: 256,
            head_dim: 8,
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

    fn test_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxVL01ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxVL01ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxVL01ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![DecodeSequenceMetadata {
            request_id: 1,
            seqlen_offset: 3,
            block_ids: vec![0],
            slot_mapping: vec![3],
        }];

        let input_ids = Tensor::from_vec(vec![5u32], (1, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 1);
    }

    #[test]
    fn test_multimodal_forward_text_only() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxVL01ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward_multimodal(
                &input_ids,
                None,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_image_newline_shape() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxVL01ForConditionalGeneration::new(&cfg, vb).unwrap();

        assert_eq!(
            model.image_newline.dims(),
            &[cfg.hidden_size],
            "image_newline must be [hidden_size]"
        );
    }
}
