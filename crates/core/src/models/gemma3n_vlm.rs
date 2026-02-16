//! Gemma3n vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: SigLIP
//! - Vision embedder: Embedding + RMSNorm + Linear projection + RMSNorm
//! - Language model: Gemma3nForCausalLM (with AltUp, per-layer projections)
//!
//! NOTE: Audio support is deferred; only vision is implemented.
//!
//! Reference: reference/vllm/vllm/model_executor/models/gemma3n_mm.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::gemma3n::Gemma3nForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Gemma3nVLMConfig {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    vision_hidden_size: usize,
    tokens_per_image: usize,
}

impl Gemma3nVLMConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        };

        let vision_config = if let Some(vc) = cfg.extra.get("vision_config") {
            VisionEncoderConfig {
                encoder_type: VisionEncoderType::SigLip,
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
                num_channels: defaults.num_channels,
                layer_norm_eps: defaults.layer_norm_eps,
            }
        } else {
            defaults
        };

        let vision_hidden_size = vision_config.hidden_size;

        let tokens_per_image = cfg
            .extra
            .get("vision_soft_tokens_per_image")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        Self {
            model_config: cfg.clone(),
            vision_config,
            vision_hidden_size,
            tokens_per_image,
        }
    }
}

// ─── Multimodal Embedder ────────────────────────────────────────────────────

/// Gemma3n multimodal embedder: projects vision features to language model space.
///
/// Pipeline: soft_embedding → soft_embedding_norm → embedding_projection → post_projection_norm
struct Gemma3nMultimodalEmbedder {
    soft_embedding_norm: RmsNorm,
    embedding_projection: Linear,
    post_projection_norm: RmsNorm,
}

impl Gemma3nMultimodalEmbedder {
    fn new(
        multimodal_hidden_size: usize,
        text_hidden_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let soft_embedding_norm = rms_norm(
            multimodal_hidden_size,
            rms_norm_eps,
            vb.pp("soft_embedding_norm"),
        )?;
        let embedding_projection = candle_nn::linear(
            multimodal_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;
        let post_projection_norm = rms_norm(
            text_hidden_size,
            rms_norm_eps,
            vb.pp("embedding_post_projection_norm"),
        )?;
        Ok(Self {
            soft_embedding_norm,
            embedding_projection,
            post_projection_norm,
        })
    }

    /// Project vision features to language model space.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.soft_embedding_norm.forward(x)?;
        let x = self.embedding_projection.forward(&x)?;
        self.post_projection_norm.forward(&x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Gemma3n vision-language model for conditional generation.
///
/// SigLIP vision encoder + multimodal embedder + Gemma3n language model.
pub struct Gemma3nForConditionalGeneration {
    #[allow(dead_code)]
    vision_tower: VisionEncoder,
    embed_vision: Gemma3nMultimodalEmbedder,
    language_model: Gemma3nForCausalLM,
    #[allow(dead_code)]
    config: Gemma3nVLMConfig,
    device: Device,
    dtype: DType,
}

impl Gemma3nForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = Gemma3nVLMConfig::from_model_config(cfg);

        // Vision encoder (SigLIP)
        let vision_tower = VisionEncoder::new(&config.vision_config, vb.pp("vision_tower"))?;

        // Vision embedder
        let embed_vision = Gemma3nMultimodalEmbedder::new(
            config.vision_hidden_size,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("embed_vision"),
        )?;

        // Language model
        let language_model = Gemma3nForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            embed_vision,
            language_model,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn merge_multimodal_embeddings(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Vision tower → embedder projection
            let vision_emb = processed_image.embedding.unsqueeze(0)?;
            let projected = self.embed_vision.forward(&vision_emb)?;
            let projected = projected.squeeze(0)?;

            // Scale by sqrt(hidden_size) to match Gemma3n's embedding scaling
            let normalizer = (self.config.model_config.hidden_size as f64).sqrt();
            let projected = (projected * normalizer)?;

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
}

impl crate::engine::ModelForward for Gemma3nForConditionalGeneration {
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
        crate::engine::ModelForward::forward_decode_batch(
            &self.language_model,
            input_ids,
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
        if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                // Get text embeddings from the language model
                let text_embeddings = self.language_model.embed_tokens(input_ids)?;

                // Merge vision embeddings
                let hidden_states_0 =
                    self.merge_multimodal_embeddings(&text_embeddings, mm_inputs)?;

                // Forward through language model from hidden states
                return self.language_model.forward_from_hidden(
                    &hidden_states_0,
                    seqlen_offset,
                    kv_cache_mgr,
                    block_table,
                    slot_mapping,
                );
            }
        }

        self.language_model.forward(
            input_ids,
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

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 56,
                "patch_size": 14,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
                "model_type": "siglip"
            }),
        );
        extra.insert(
            "vision_soft_tokens_per_image".to_string(),
            serde_json::json!(256),
        );
        // Gemma3n-specific config
        extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::json!(256.0),
        );
        extra.insert("altup_num_inputs".to_string(), serde_json::json!(2));
        extra.insert("altup_active_idx".to_string(), serde_json::json!(0));
        extra.insert("altup_coef_clip".to_string(), serde_json::json!(1.0));
        extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(32),
        );
        extra.insert("laurel_rank".to_string(), serde_json::json!(16));

        ModelConfig {
            architectures: vec!["Gemma3nForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
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

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
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
    fn test_multimodal_embedder() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let embedder = Gemma3nMultimodalEmbedder::new(64, 128, 1e-6, vb).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 64), &device).unwrap();
        let output = embedder.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 8, 128]);
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3nForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3nForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
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
        let model = Gemma3nForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }
}
