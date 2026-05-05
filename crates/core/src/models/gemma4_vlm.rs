//! Gemma 4 Vision-Language Model.
//!
//! Combines a SigLIP vision encoder with a Gemma4 language model
//! via a simplified multimodal projector (Linear + UnweightedRMSNorm).
//!
//! # Architecture
//!
//! 1. SigLIP vision encoder processes images into patch embeddings
//! 2. Linear projection maps vision_hidden → text_hidden (no bias)
//! 3. UnweightedRMSNorm normalizes projected features
//! 4. Gemma4 language model generates text conditioned on merged embeddings
//!
//! Unlike Gemma3 VLM (which uses AvgPool2d + learned RMSNorm + weight matrix),
//! Gemma4's embedder is simpler: just Linear + unweighted RMSNorm.
//!
//! # Weight mapping
//!
//! - `vision_tower.*` → SigLIP vision encoder
//! - `embed_vision.embedding_projection.*` → Linear projection
//! - `embed_vision.embedding_post_projection_norm.*` → UnweightedRMSNorm
//! - `language_model.*` → Gemma4 backbone
//!
//! Reference: `reference/vllm/vllm/model_executor/models/gemma4_mm.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower};

use super::gemma4::Gemma4ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Gemma4VLMConfig {
    pub model_config: ModelConfig,
    pub vision_config: Gemma4VisionConfig,
    pub vision_hidden_size: usize,
    pub tokens_per_image: usize,
    pub image_token_id: u32,
}

impl Gemma4VLMConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = Gemma4VisionConfig::from_model_config(cfg);
        let vision_hidden_size = vision_config.hidden_size;
        let tokens_per_image = vision_config.default_output_length;

        // `image_token_id` (new key) / legacy `image_token_index` alias.
        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .or_else(|| cfg.extra.get("image_token_index"))
            .and_then(|v| v.as_u64())
            .unwrap_or(262144) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            vision_hidden_size,
            tokens_per_image,
            image_token_id,
        }
    }
}

// ─── Multimodal Embedder ────────────────────────────────────────────────────

/// Gemma4 multimodal embedder: Linear → UnweightedRMSNorm.
///
/// Simpler than Gemma3/Gemma3n projectors — no AvgPool, no learned norm weights.
struct Gemma4MultimodalEmbedder {
    embedding_projection: candle_nn::Linear,
    eps: f64,
}

impl Gemma4MultimodalEmbedder {
    fn new(
        multimodal_hidden_size: usize,
        text_hidden_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding_projection = candle_nn::linear_no_bias(
            multimodal_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;

        Ok(Self {
            embedding_projection,
            eps: rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.embedding_projection.forward(x)?;
        // Unweighted RMSNorm (no learned scale)
        let dtype = projected.dtype();
        let projected_f32 = projected.to_dtype(DType::F32)?;
        let variance = projected_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let normed = projected_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Gemma4ForConditionalGeneration {
    vision_tower: Option<Gemma4VisionTower>,
    embed_vision: Option<Gemma4MultimodalEmbedder>,
    language_model: Gemma4ForCausalLM,
    #[allow(dead_code)]
    config: Gemma4VLMConfig,
    device: Device,
    dtype: DType,
}

/// `cfg.extra` flag honored by both Gemma 4 VLM constructors: when set
/// the vision tower + multimodal embedder are skipped at load time so
/// the language model fits a smaller VRAM budget. Image inputs become
/// errors in that mode (see `encode_images` / `merge_multimodal`).
pub(crate) const SKIP_MULTIMODAL_KEY: &str = "vllm_rust.disable_multimodal";

pub(crate) fn should_skip_multimodal(cfg: &ModelConfig) -> bool {
    cfg.extra
        .get(SKIP_MULTIMODAL_KEY)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

impl Gemma4ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = Gemma4VLMConfig::from_model_config(cfg);

        // HF checkpoints wrap everything under a `model.*` root; descend
        // once here so sub-module VarBuilders are positioned correctly.
        let vb_root = vb.pp("model");

        let skip_mm = should_skip_multimodal(cfg);

        let (vision_tower, embed_vision) = if skip_mm {
            (None, None)
        } else {
            let vt = Gemma4VisionTower::new(&config.vision_config, vb_root.pp("vision_tower"))?;
            let ev = Gemma4MultimodalEmbedder::new(
                config.vision_hidden_size,
                cfg.hidden_size,
                config.vision_config.rms_norm_eps,
                vb_root.pp("embed_vision"),
            )?;
            (Some(vt), Some(ev))
        };

        let language_model = Gemma4ForCausalLM::new(cfg, vb_root.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            embed_vision,
            language_model,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Encode images: vision encoder → embedder projection.
    ///
    /// `pixel_values`       : `[B, L, 3·ps·ps]` flattened patches.
    /// `pixel_position_ids` : `i64 [B, L, 2]` (`-1, -1` marks padding).
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
    ) -> Result<Tensor> {
        let vt = self.vision_tower.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "Gemma 4 VLM was loaded with multimodal disabled (text-only mode); \
                 image inputs are not supported in this configuration"
                    .to_string(),
            )
        })?;
        let ev = self.embed_vision.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("embed_vision projector was skipped at load time".to_string())
        })?;
        let vision_features = vt.forward(pixel_values, pixel_position_ids)?;
        ev.forward(&vision_features)
    }

    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        // If the user disabled multimodal at load time, refuse to mix in
        // image embeddings rather than silently dropping them.
        let Some(embed_vision) = self.embed_vision.as_ref() else {
            return Err(candle_core::Error::Msg(
                "Gemma 4 VLM was loaded with multimodal disabled but the request \
                 carries image embeddings — re-load without --text-only or drop the images"
                    .to_string(),
            ));
        };

        let (_batch_size, seq_len, _hidden_size) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = embed_vision.forward(&vision_emb)?;
            let projected = projected.squeeze(0)?;
            let emb_vec: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

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

impl crate::engine::ModelForward for Gemma4ForConditionalGeneration {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::multimodal::ProcessedImage;

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));
        extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(16),
        );
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "num_channels": 3,
                "rms_norm_eps": 1e-6,
                "default_output_length": 4,
                "model_type": "siglip"
            }),
        );
        extra.insert("image_token_index".to_string(), serde_json::json!(262144));

        ModelConfig {
            architectures: vec!["Gemma4ForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
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
    fn test_vlm_config_parsing() {
        let cfg = test_model_config();
        let vlm_cfg = Gemma4VLMConfig::from_model_config(&cfg);
        assert_eq!(vlm_cfg.vision_config.hidden_size, 32);
        assert_eq!(vlm_cfg.vision_config.patch_size, 14);
        assert_eq!(vlm_cfg.tokens_per_image, 4);
        assert_eq!(vlm_cfg.image_token_id, 262144);
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Gemma4ForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Gemma4 VLM should construct: {:?}",
            model.err()
        );
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_skips_vision_tower() {
        // With `vllm_rust.disable_multimodal=true` set, VLM construction
        // must skip the vision tower + embed_vision projector and image
        // inputs must be rejected at runtime.
        let device = Device::Cpu;
        let mut cfg = test_model_config();
        cfg.extra.insert(
            super::super::gemma4_vlm::SKIP_MULTIMODAL_KEY.to_string(),
            serde_json::json!(true),
        );
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForConditionalGeneration::new(&cfg, vb).expect("text-only build");

        // Internals must be `None`.
        assert!(model.vision_tower.is_none(), "vision_tower must be skipped");
        assert!(model.embed_vision.is_none(), "embed_vision must be skipped");

        // `encode_images` must return a clear error.
        let dummy_pixels = Tensor::zeros((1, 1, 768), DType::F32, &device).unwrap();
        let dummy_pos = Tensor::zeros((1, 1, 2), DType::I64, &device).unwrap();
        let err = model.encode_images(&dummy_pixels, &dummy_pos).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("text-only") || msg.contains("multimodal"),
            "expected clear text-only error, got: {msg}"
        );
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_multimodal_forward_with_images() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1, 2, 3], 0);

        let seq_len = 8;
        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();

        // 4 image tokens
        let img_embedding = Tensor::randn(0f32, 1.0, (4, 32), &device).unwrap();
        let processed = ProcessedImage::new(img_embedding, 4);
        let mm_inputs = MultimodalInputs::with_images(vec![0u32; seq_len], vec![(0, processed)]);

        let logits = model
            .forward_multimodal(
                &input_ids,
                Some(&mm_inputs),
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_embedder_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let embedder = Gemma4MultimodalEmbedder::new(32, 64, 1e-6, vb).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 32), &device).unwrap();
        let output = embedder.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 8, 64]);
    }
}
