//! Fuyu vision-language model implementation (Adept AI).
//!
//! Architecture:
//! - Vision encoder: Simple linear patch embedding (NOT CLIP/SigLIP)
//!   Each patch (patch_size² × num_channels) is flattened and linearly projected
//!   to hidden_size. No attention layers, no positional embeddings in vision.
//! - Projector: None (vision embedding IS the projector — single linear layer)
//! - Language model: Persimmon (reused from persimmon.rs)
//!
//! Supports only a single image per request.
//!
//! Reference: reference/vllm/vllm/model_executor/models/fuyu.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::persimmon::PersimmonForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FuyuConfig {
    patch_size: usize,
    num_channels: usize,
    image_feature_size: usize,
}

impl FuyuConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let patch_size = cfg
            .extra
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(30) as usize;

        let num_channels = cfg
            .extra
            .get("num_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let image_feature_size = patch_size * patch_size * num_channels;

        Self {
            patch_size,
            num_channels,
            image_feature_size,
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct FuyuForCausalLM {
    vision_embed_tokens: Linear,
    language_model: PersimmonForCausalLM,
    device: Device,
    dtype: DType,
}

impl FuyuForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let fuyu_cfg = FuyuConfig::from_model_config(cfg);

        // Vision patch embedding: [patch_size² × channels] → [hidden_size]
        // Weight key: vision_embed_tokens (under model prefix)
        let vision_embed_tokens = linear(
            fuyu_cfg.image_feature_size,
            cfg.hidden_size,
            vb.pp("vision_embed_tokens"),
        )?;

        // The Persimmon backbone lives under "language_model"
        // HF weights: model.language_model.model.* → we pass vb.pp("language_model")
        let language_model = PersimmonForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_embed_tokens,
            language_model,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Project image patch features to the language model's hidden space.
    fn embed_image_patches(&self, image_features: &Tensor) -> Result<Tensor> {
        // image_features: [num_patches, patch_size² × num_channels]
        // Output: [num_patches, hidden_size]
        self.vision_embed_tokens.forward(image_features)
    }

    /// Merge text and image embeddings based on multimodal inputs.
    ///
    /// Image tokens in the input_ids are replaced with vision embeddings
    /// from the patch projector at the positions specified in MultimodalInputs.
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

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Fuyu: the image embedding is already patch features.
            // Project through the linear layer.
            let vision_embeds = self.embed_image_patches(&processed_image.embedding)?;
            let vision_embeds = vision_embeds.to_dtype(DType::F32)?;
            let img_emb: Vec<Vec<f32>> = vision_embeds.to_vec2()?;

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

impl crate::engine::ModelForward for FuyuForCausalLM {
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
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use crate::multimodal::ProcessedImage;

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("patch_size".to_string(), serde_json::json!(4));
        extra.insert("num_channels".to_string(), serde_json::json!(3));
        extra.insert("layer_norm_eps".to_string(), serde_json::json!(1e-5));

        ModelConfig {
            architectures: vec!["FuyuForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(true),
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
    fn test_fuyu_config_extraction() {
        let cfg = test_model_config();
        let fuyu_cfg = FuyuConfig::from_model_config(&cfg);

        assert_eq!(fuyu_cfg.patch_size, 4);
        assert_eq!(fuyu_cfg.num_channels, 3);
        assert_eq!(fuyu_cfg.image_feature_size, 48); // 4*4*3
    }

    #[test]
    fn test_fuyu_config_defaults() {
        let cfg = ModelConfig {
            architectures: vec!["FuyuForCausalLM".to_string()],
            hidden_size: 4096,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            num_hidden_layers: 36,
            intermediate_size: 16384,
            vocab_size: 262144,
            max_position_embeddings: 16384,
            head_dim: 64,
            hidden_act: "relu2".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: None,
            extra: serde_json::Map::new(),
        };

        let fuyu_cfg = FuyuConfig::from_model_config(&cfg);
        assert_eq!(fuyu_cfg.patch_size, 30);
        assert_eq!(fuyu_cfg.num_channels, 3);
        assert_eq!(fuyu_cfg.image_feature_size, 2700); // 30*30*3
    }

    #[test]
    fn test_fuyu_model_construction() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FuyuForCausalLM::new(&cfg, vb).expect("should build Fuyu model");

        assert!(model.supports_multimodal());
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_fuyu_vision_embedding() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FuyuForCausalLM::new(&cfg, vb).expect("should build Fuyu model");

        // Create fake patch features: 4 patches, each 48-dim (4*4*3)
        let patches = Tensor::zeros((4, 48), DType::F32, &device).unwrap();
        let embedded = model.embed_image_patches(&patches).unwrap();
        assert_eq!(embedded.dims(), &[4, 64]); // [num_patches, hidden_size]
    }

    #[test]
    fn test_fuyu_text_forward() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FuyuForCausalLM::new(&cfg, vb).expect("should build Fuyu model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("should run text-only forward pass");
        assert_eq!(logits.dims(), &[1, 4, 256]); // [batch, seq_len, vocab_size]
    }

    #[test]
    fn test_fuyu_multimodal_forward() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FuyuForCausalLM::new(&cfg, vb).expect("should build Fuyu model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..6).collect();

        let input_ids = Tensor::zeros((1, 6), DType::U32, &device).unwrap();

        // 2 patches of 48-dim each (patch_size=4, channels=3)
        let patches = Tensor::zeros((2, 48), DType::F32, &device).unwrap();
        let image = ProcessedImage::new(patches, 2);
        let mm_inputs = MultimodalInputs::with_images(vec![0u32; 6], vec![(0, image)]);

        let logits = model
            .forward_multimodal(
                &input_ids,
                Some(&mm_inputs),
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("should run multimodal forward pass");
        assert_eq!(logits.dims(), &[1, 6, 256]); // [batch, seq_len, vocab_size]
    }

    #[test]
    fn test_fuyu_no_image_fallback() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FuyuForCausalLM::new(&cfg, vb).expect("should build Fuyu model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        // No images — should fall back to text-only path
        let logits = model
            .forward_multimodal(
                &input_ids,
                None,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("should run text-only fallback");
        assert_eq!(logits.dims(), &[1, 4, 256]);
    }
}
