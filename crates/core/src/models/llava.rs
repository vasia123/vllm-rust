//! LLaVA (Large Language and Vision Assistant) model implementation.
//!
//! LLaVA is a multimodal model that combines a vision encoder (CLIP/SigLIP)
//! with a language model (typically Llama) using a projector to map
//! vision embeddings to the language model's hidden space.
//!
//! # Architecture
//!
//! 1. Vision encoder processes images into patch embeddings
//! 2. Projector maps vision embeddings to LLM hidden dimension
//! 3. Image embeddings are inserted at <image> token positions
//! 4. Language model generates text conditioned on merged embeddings
//!
//! # Supported versions
//!
//! - LLaVA 1.5: CLIP ViT-L/14 @ 336px + 2-layer MLP projector + Llama
//! - LLaVA 1.6: SigLIP + MLP projector + various LLMs
//!
//! Reference: https://llava-vl.github.io/

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{
    MultimodalInputs, MultimodalProjector, ProjectorConfig, VisionEncoder, VisionEncoderConfig,
};

/// Configuration for LLaVA model.
#[derive(Debug, Clone)]
pub struct LLaVAConfig {
    /// Model configuration (for language model).
    pub model_config: ModelConfig,
    /// Vision encoder configuration.
    pub vision_config: VisionEncoderConfig,
    /// Projector configuration.
    pub projector_config: ProjectorConfig,
    /// Image token ID used as placeholder.
    pub image_token_id: u32,
    /// Select which vision features to use.
    pub mm_vision_select_feature: VisionSelectFeature,
}

/// Which vision features to use from the encoder output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionSelectFeature {
    /// Use all features including CLS token (default for CLIP).
    #[default]
    Default,
    /// Use only patch features (exclude CLS token).
    PatchOnly,
    /// Use only CLS token.
    ClsOnly,
}

impl VisionSelectFeature {
    /// Parse from HuggingFace config strategy string.
    pub fn from_strategy_str(s: &str) -> Self {
        match s {
            "patch" => Self::PatchOnly,
            "cls" => Self::ClsOnly,
            "default" | "full" => Self::Default,
            _ => Self::Default,
        }
    }
}

impl LLaVAConfig {
    /// Parse LLaVA config from a ModelConfig, extracting vision_config from extra fields.
    ///
    /// Falls back to LLaVA 1.5 defaults for any missing fields.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig::clip_vit_l_14_336();

        let (vision_config, select_feature, image_token_id) = if let Some(vc) =
            cfg.extra.get("vision_config")
        {
            let hidden_size = vc
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.hidden_size as u64) as usize;
            let intermediate_size =
                vc.get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.intermediate_size as u64) as usize;
            let num_attention_heads =
                vc.get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_attention_heads as u64) as usize;
            let num_hidden_layers =
                vc.get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_hidden_layers as u64) as usize;
            let image_size = vc
                .get("image_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_size as u64) as usize;
            let patch_size = vc
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.patch_size as u64) as usize;
            let num_channels = vc
                .get("num_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_channels as u64) as usize;
            let layer_norm_eps = vc
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps);

            let encoder_type = if vc.get("model_type").and_then(|v| v.as_str()) == Some("siglip") {
                crate::multimodal::VisionEncoderType::SigLip
            } else {
                crate::multimodal::VisionEncoderType::Clip
            };

            let vision_config = VisionEncoderConfig {
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_hidden_layers,
                image_size,
                patch_size,
                num_channels,
                layer_norm_eps,
                encoder_type,
            };

            let select_feature = cfg
                .extra
                .get("vision_feature_select_strategy")
                .and_then(|v| v.as_str())
                .map(VisionSelectFeature::from_strategy_str)
                .unwrap_or_default();

            let image_token_id = cfg
                .extra
                .get("image_token_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(32000) as u32;

            (vision_config, select_feature, image_token_id)
        } else {
            (defaults, VisionSelectFeature::Default, 32000)
        };

        let projector_config = ProjectorConfig::mlp(vision_config.hidden_size, cfg.hidden_size);

        Self {
            model_config: cfg.clone(),
            vision_config,
            projector_config,
            image_token_id,
            mm_vision_select_feature: select_feature,
        }
    }

    /// Create LLaVA 1.5 configuration (CLIP ViT-L/14 @ 336px + Llama).
    pub fn llava_1_5(model_config: ModelConfig) -> Self {
        let vision_config = VisionEncoderConfig::clip_vit_l_14_336();
        let projector_config =
            ProjectorConfig::mlp(vision_config.hidden_size, model_config.hidden_size);

        Self {
            model_config,
            vision_config,
            projector_config,
            image_token_id: 32000, // Default LLaVA image token
            mm_vision_select_feature: VisionSelectFeature::Default,
        }
    }

    /// Create LLaVA 1.6 configuration (SigLIP + various LLMs).
    pub fn llava_1_6(model_config: ModelConfig) -> Self {
        let vision_config = VisionEncoderConfig::siglip_so400m_384();
        let projector_config =
            ProjectorConfig::mlp(vision_config.hidden_size, model_config.hidden_size);

        Self {
            model_config,
            vision_config,
            projector_config,
            image_token_id: 32000,
            mm_vision_select_feature: VisionSelectFeature::Default,
        }
    }

    /// Number of image tokens per image.
    pub fn num_image_tokens(&self) -> usize {
        match self.mm_vision_select_feature {
            VisionSelectFeature::ClsOnly => 1,
            VisionSelectFeature::PatchOnly => self.vision_config.num_patches(),
            VisionSelectFeature::Default => self.vision_config.seq_len(),
        }
    }
}

/// LLaVA model for conditional generation.
pub struct LLaVAForConditionalGeneration {
    /// Vision encoder (CLIP/SigLIP).
    vision_encoder: VisionEncoder,
    /// Projector to map vision embeddings to LLM space.
    projector: MultimodalProjector,
    /// Language model embedding layer (shared with LLM).
    embed_tokens: Embedding,
    /// Language model layers.
    language_model: Box<dyn LanguageModelCore>,
    /// Image token ID for placeholder detection.
    #[allow(dead_code)] // Used when processing raw pixel values
    image_token_id: u32,
    /// Vision feature selection.
    mm_vision_select_feature: VisionSelectFeature,
    /// Device.
    device: Device,
    /// Dtype.
    dtype: DType,
}

/// Core language model interface (without embedding layer).
///
/// This allows LLaVA to manage embeddings and inject image features.
pub trait LanguageModelCore: Send + 'static {
    /// Forward through transformer layers with pre-computed embeddings.
    fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor>;

    /// Forward through transformer layers for batched decode.
    fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor>;

    /// Get the device.
    fn device(&self) -> &Device;
}

impl LLaVAForConditionalGeneration {
    /// Create a new LLaVA model.
    pub fn new(cfg: &LLaVAConfig, vb: VarBuilder) -> Result<Self> {
        // Vision encoder
        let vision_encoder = VisionEncoder::new(&cfg.vision_config, vb.pp("vision_tower"))?;

        // Projector
        let projector =
            MultimodalProjector::new(&cfg.projector_config, vb.pp("multi_modal_projector"))?;

        // Embedding layer
        let embed_tokens = candle_nn::embedding(
            cfg.model_config.vocab_size,
            cfg.model_config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        // For now, create a dummy language model core
        // In practice, you would pass a real Llama/Mistral core
        let language_model = Box::new(DummyLanguageModelCore {
            device: vb.device().clone(),
        });

        Ok(Self {
            vision_encoder,
            projector,
            embed_tokens,
            language_model,
            image_token_id: cfg.image_token_id,
            mm_vision_select_feature: cfg.mm_vision_select_feature,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create a LLaVA model from a generic ModelConfig.
    ///
    /// Extracts vision and projector configuration from the ModelConfig's extra
    /// fields, falling back to LLaVA 1.5 defaults when not specified.
    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let llava_cfg = LLaVAConfig::from_model_config(cfg);
        Self::new(&llava_cfg, vb)
    }

    /// Encode images using the vision encoder and projector.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Get vision features
        let vision_features = self.vision_encoder.forward(pixel_values)?;

        // Select features based on configuration
        let vision_features = self.select_vision_features(&vision_features)?;

        // Project to LLM space
        self.projector.project(&vision_features)
    }

    /// Select vision features based on configuration.
    fn select_vision_features(&self, features: &Tensor) -> Result<Tensor> {
        match self.mm_vision_select_feature {
            VisionSelectFeature::Default => Ok(features.clone()),
            VisionSelectFeature::PatchOnly => {
                // Skip CLS token (first token)
                let seq_len = features.dim(1)?;
                features.narrow(1, 1, seq_len - 1)
            }
            VisionSelectFeature::ClsOnly => {
                // Only CLS token (first token)
                features.narrow(1, 0, 1)
            }
        }
    }

    /// Merge text and image embeddings.
    ///
    /// Replaces image token positions in text embeddings with image embeddings.
    /// Used when processing raw pixel values directly (before multimodal processor).
    #[allow(dead_code)]
    fn merge_embeddings(
        &self,
        input_ids: &Tensor,
        text_embeddings: &Tensor,
        image_embeddings: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = text_embeddings.dims3()?;
        let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

        // Find image token positions
        let image_positions: Vec<usize> = input_ids_vec
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.image_token_id)
            .map(|(i, _)| i)
            .collect();

        if image_positions.is_empty() {
            // No images to merge
            return Ok(text_embeddings.clone());
        }

        // Get number of image tokens (used for validation in full implementation)
        let _num_image_tokens = image_embeddings.dim(1)?;

        // For simplicity, we assume single image case for now
        // Full implementation would handle multiple images per batch item
        if image_positions.len() != batch_size {
            // Different handling needed for multiple images per sample
            // For now, just handle one image token per batch item
        }

        // Clone text embeddings and replace at image positions
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (batch_idx, &pos) in image_positions.iter().enumerate() {
            let img_emb: Vec<Vec<f32>> = image_embeddings
                .narrow(0, batch_idx.min(image_embeddings.dim(0)? - 1), 1)?
                .squeeze(0)?
                .to_vec2()?;

            // Replace tokens starting at image position
            let start = pos % seq_len;
            for (i, emb) in img_emb.iter().enumerate() {
                let target_pos = start + i;
                if target_pos < seq_len {
                    merged[batch_idx][target_pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    /// Merge text embeddings with pre-processed multimodal embeddings.
    ///
    /// The image embeddings in MultimodalInputs have already been processed by
    /// the vision encoder. We need to project them to LLM space and insert them
    /// at the correct positions.
    fn merge_multimodal_embeddings(
        &self,
        input_ids: &Tensor,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = text_embeddings.dims3()?;
        // input_ids_vec could be used for additional validation in future
        let _input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

        // Convert text embeddings to Vec for in-place modification
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        // Process each image embedding
        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Project vision embeddings to LLM space
            // Input: [num_tokens, vision_hidden_size]
            // Output: [num_tokens, llm_hidden_size]
            let vision_emb = processed_image.embedding.unsqueeze(0)?; // [1, num_tokens, vision_hidden]
            let projected = self.projector.project(&vision_emb)?;
            let projected = projected.squeeze(0)?; // [num_tokens, llm_hidden]
            let img_emb: Vec<Vec<f32>> = projected.to_vec2()?;

            // Calculate batch index and position within sequence
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            if batch_idx >= batch_size {
                continue;
            }

            // Insert image tokens at the position, replacing placeholder tokens
            for (i, emb) in img_emb.iter().enumerate() {
                let target_pos = start_pos + i;
                if target_pos < seq_len {
                    merged[batch_idx][target_pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    /// Forward pass with multimodal inputs.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Get text embeddings
        let text_embeddings = self.embed_tokens.forward(input_ids)?;

        // Merge with image embeddings if present
        let embeddings = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                // Image embeddings are already processed by the multimodal processor.
                // They come as (position, ProcessedImage) pairs.
                // ProcessedImage.embedding is [num_tokens, hidden_size] from the vision encoder.
                // We need to project them to LLM space and merge into text embeddings.
                self.merge_multimodal_embeddings(input_ids, &text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
        } else {
            text_embeddings
        };

        // Forward through language model
        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the vision encoder.
    pub fn vision_encoder(&self) -> &VisionEncoder {
        &self.vision_encoder
    }

    /// Get the projector.
    pub fn projector(&self) -> &MultimodalProjector {
        &self.projector
    }
}

impl crate::engine::ModelForward for LLaVAForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
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
        // During decode, no image tokens (already processed in prefill)
        let embeddings = self.embed_tokens.forward(input_ids)?;
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
        self.forward(
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

/// Dummy language model core for testing.
struct DummyLanguageModelCore {
    device: Device,
}

impl LanguageModelCore for DummyLanguageModelCore {
    fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Return dummy logits matching embedding batch/seq
        let (batch_size, seq_len, _) = embeddings.dims3()?;
        Tensor::zeros((batch_size, seq_len, 32000), DType::F32, &self.device)
    }

    fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = embeddings.dims3()?;
        Tensor::zeros((batch_size, seq_len, 32000), DType::F32, &self.device)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["LLaVAForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 32001,
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
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn test_llava_config_1_5() {
        let model_cfg = test_model_config();
        let cfg = LLaVAConfig::llava_1_5(model_cfg);

        assert_eq!(cfg.vision_config.image_size, 336);
        assert_eq!(cfg.vision_config.hidden_size, 1024);
        assert_eq!(cfg.num_image_tokens(), 577); // 576 patches + 1 CLS
    }

    #[test]
    fn test_llava_config_1_6() {
        let model_cfg = test_model_config();
        let cfg = LLaVAConfig::llava_1_6(model_cfg);

        assert_eq!(cfg.vision_config.image_size, 384);
        assert_eq!(cfg.vision_config.hidden_size, 1152);
        // SigLIP has no CLS token
        assert_eq!(cfg.num_image_tokens(), 729); // 27*27 patches
    }

    #[test]
    fn test_vision_select_feature() {
        let model_cfg = test_model_config();
        let mut cfg = LLaVAConfig::llava_1_5(model_cfg);

        cfg.mm_vision_select_feature = VisionSelectFeature::PatchOnly;
        assert_eq!(cfg.num_image_tokens(), 576); // patches only

        cfg.mm_vision_select_feature = VisionSelectFeature::ClsOnly;
        assert_eq!(cfg.num_image_tokens(), 1); // CLS only
    }

    #[test]
    fn test_llava_creation() {
        let device = Device::Cpu;
        let model_cfg = test_model_config();

        // Create minimal config
        let vision_config = VisionEncoderConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            ..Default::default()
        };

        let projector_config = ProjectorConfig::mlp(64, 64);

        let cfg = LLaVAConfig {
            model_config: model_cfg,
            vision_config,
            projector_config,
            image_token_id: 32000,
            mm_vision_select_feature: VisionSelectFeature::Default,
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LLaVAForConditionalGeneration::new(&cfg, vb);

        assert!(model.is_ok());

        let model = model.unwrap();
        assert!(model.supports_multimodal());
    }
}
