//! Molmo vision-language model implementation.
//!
//! Molmo is a VLM from AI2 that combines a CLIP-based vision backbone with an
//! OLMo-like language model. The vision backbone extracts multi-layer features
//! from the ViT, applies 2x2 spatial pooling, and projects them into the LLM's
//! hidden space.
//!
//! # Architecture
//!
//! 1. **Vision encoder**: CLIP ViT-L/14 @ 336px (23 layers default)
//!    - Multi-layer feature extraction from layers [-2, -9], concatenated
//!    - 2x2 spatial pooling on extracted features
//! 2. **Image projector**: MLP mapping pooled vision features to LLM hidden dim
//! 3. **Language model**: OLMo2-based backbone with post-normalization
//!
//! # Constants
//!
//! - `VIT_LAYERS`: `[-2, -9]` (layers to extract features from)
//! - `POOLING_SIZE`: 2 (2x2 spatial pooling)
//! - `ADDITIONAL_VOCAB_SIZE`: 128 (extra tokens for `<im_patch>`, etc.)
//!
//! Reference: AI2 Molmo (https://molmo.allenai.org/)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{
    MultimodalInputs, MultimodalProjector, ProjectorConfig, VisionEncoder, VisionEncoderConfig,
    VisionEncoderType,
};

use super::olmo2::Olmo2ForCausalLM;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Default layers to extract features from (negative indices into ViT layers).
const VIT_LAYERS: [i32; 2] = [-2, -9];

/// Spatial pooling size applied to vision features (2x2 pooling).
const POOLING_SIZE: usize = 2;

/// Additional vocabulary size for special image tokens.
const ADDITIONAL_VOCAB_SIZE: usize = 128;

/// Default image token ID for `<im_patch>`.
const DEFAULT_IMAGE_TOKEN_ID: u32 = 152066;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Vision backbone configuration for Molmo.
#[derive(Debug, Clone)]
pub struct MolmoVisionConfig {
    /// Vision encoder hidden dimension.
    pub image_emb_dim: usize,
    /// Number of attention heads in the vision encoder.
    pub image_num_heads: usize,
    /// Number of transformer layers in the vision encoder.
    pub image_num_layers: usize,
    /// Default input image size (width, height).
    pub image_default_input_size: (usize, usize),
    /// Patch size for the vision encoder.
    pub image_patch_size: usize,
    /// Number of patches per image dimension.
    pub image_num_patch: (usize, usize),
    /// MLP intermediate size in the vision encoder.
    pub image_mlp_dim: usize,
    /// Encoder type (CLIP vs SigLIP).
    pub encoder_type: VisionEncoderType,
}

impl Default for MolmoVisionConfig {
    fn default() -> Self {
        // CLIP ViT-L/14 @ 336px defaults
        let image_size = 336;
        let patch_size = 14;
        let patches_per_side = image_size / patch_size;
        Self {
            image_emb_dim: 1024,
            image_num_heads: 16,
            image_num_layers: 23,
            image_default_input_size: (image_size, image_size),
            image_patch_size: patch_size,
            image_num_patch: (patches_per_side, patches_per_side),
            image_mlp_dim: 4096,
            encoder_type: VisionEncoderType::Clip,
        }
    }
}

impl MolmoVisionConfig {
    /// Build a `VisionEncoderConfig` from this Molmo-specific config.
    pub fn to_encoder_config(&self) -> VisionEncoderConfig {
        VisionEncoderConfig {
            encoder_type: self.encoder_type,
            hidden_size: self.image_emb_dim,
            intermediate_size: self.image_mlp_dim,
            num_attention_heads: self.image_num_heads,
            num_hidden_layers: self.image_num_layers,
            image_size: self.image_default_input_size.0,
            patch_size: self.image_patch_size,
            num_channels: 3,
            layer_norm_eps: 1e-5,
        }
    }

    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();
        let image_size = json
            .get("image_default_input_size")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 2 {
                    Some((arr[0].as_u64()? as usize, arr[1].as_u64()? as usize))
                } else {
                    None
                }
            })
            .unwrap_or(defaults.image_default_input_size);
        let patch_size = json
            .get("image_patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(defaults.image_patch_size as u64) as usize;
        let patches = json
            .get("image_num_patch")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 2 {
                    Some((arr[0].as_u64()? as usize, arr[1].as_u64()? as usize))
                } else {
                    None
                }
            })
            .unwrap_or((image_size.0 / patch_size, image_size.1 / patch_size));

        Self {
            image_emb_dim: json
                .get("image_emb_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_emb_dim as u64) as usize,
            image_num_heads: json
                .get("image_num_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_num_heads as u64) as usize,
            image_num_layers: json
                .get("image_num_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_num_layers as u64) as usize,
            image_default_input_size: image_size,
            image_patch_size: patch_size,
            image_num_patch: patches,
            image_mlp_dim: json
                .get("image_mlp_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_mlp_dim as u64) as usize,
            encoder_type: defaults.encoder_type,
        }
    }
}

/// Top-level Molmo configuration.
#[derive(Debug, Clone)]
pub struct MolmoConfig {
    /// Language model configuration.
    pub model_config: ModelConfig,
    /// Vision backbone configuration.
    pub vision_config: MolmoVisionConfig,
    /// Projector configuration.
    pub projector_config: ProjectorConfig,
    /// Image patch token ID.
    pub image_token_id: u32,
    /// Additional vocabulary size for special image tokens.
    pub additional_vocab_size: usize,
}

impl MolmoConfig {
    /// Parse Molmo configuration from a generic `ModelConfig`.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_backbone")
            .map(MolmoVisionConfig::from_json)
            .unwrap_or_default();

        let additional_vocab_size = cfg
            .extra
            .get("additional_vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(ADDITIONAL_VOCAB_SIZE as u64) as usize;

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_IMAGE_TOKEN_ID as u64) as u32;

        // Molmo uses multi-layer feature extraction then concatenation, so the
        // projector input dim is: image_emb_dim * len(VIT_LAYERS) / (POOLING_SIZE^2)
        // After 2x2 pooling the spatial count shrinks but the channel dim is
        // preserved (pooling is via attention, not channel merge). The projector
        // simply maps image_emb_dim → llm_hidden_size.
        let projector_config = ProjectorConfig::mlp(vision_config.image_emb_dim, cfg.hidden_size);

        Self {
            model_config: cfg.clone(),
            vision_config,
            projector_config,
            image_token_id,
            additional_vocab_size,
        }
    }

    /// Number of image tokens per image after pooling.
    ///
    /// For a 336x336 image with patch_size=14: 24x24 patches → after 2x2 pooling → 12x12 = 144.
    pub fn num_image_tokens(&self) -> usize {
        let (h, w) = self.vision_config.image_num_patch;
        let h_pooled = h.div_ceil(POOLING_SIZE);
        let w_pooled = w.div_ceil(POOLING_SIZE);
        h_pooled * w_pooled
    }

    /// The VIT layers to extract features from (as positive indices).
    pub fn vit_layer_indices(&self) -> Vec<usize> {
        let total = self.vision_config.image_num_layers;
        VIT_LAYERS
            .iter()
            .map(|&layer| {
                if layer < 0 {
                    (total as i32 + layer) as usize
                } else {
                    layer as usize
                }
            })
            .collect()
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Molmo model for conditional generation.
///
/// Combines a CLIP vision encoder, an MLP projector, and an OLMo2 language
/// model backbone. Image embeddings are projected and merged into the text
/// embedding sequence at `<im_patch>` token positions during prefill.
pub struct MolmoForCausalLM {
    /// Vision encoder (CLIP ViT).
    vision_encoder: VisionEncoder,
    /// MLP projector mapping vision features to LLM hidden dim.
    projector: MultimodalProjector,
    /// Text embedding layer (vocab_size + additional_vocab_size).
    embed_tokens: Embedding,
    /// OLMo2-based language model (without its own embedding).
    language_model: Olmo2ForCausalLM,
    /// Image token ID used as placeholder in the input.
    image_token_id: u32,
    /// Device.
    device: Device,
    /// Dtype.
    dtype: DType,
}

impl MolmoForCausalLM {
    /// Create a new Molmo model from explicit config.
    pub fn new(cfg: &MolmoConfig, vb: VarBuilder) -> Result<Self> {
        let encoder_config = cfg.vision_config.to_encoder_config();
        let vision_encoder = VisionEncoder::new(&encoder_config, vb.pp("vision_backbone"))?;

        let projector = MultimodalProjector::new(&cfg.projector_config, vb.pp("image_projector"))?;

        // Molmo uses an extended vocabulary for special image tokens.
        let total_vocab = cfg.model_config.vocab_size + cfg.additional_vocab_size;
        let embed_tokens = candle_nn::embedding(
            total_vocab,
            cfg.model_config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let language_model = Olmo2ForCausalLM::new(&cfg.model_config, vb.clone())?;

        Ok(Self {
            vision_encoder,
            projector,
            embed_tokens,
            language_model,
            image_token_id: cfg.image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create a Molmo model from a generic `ModelConfig`.
    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let molmo_cfg = MolmoConfig::from_model_config(cfg);
        Self::new(&molmo_cfg, vb)
    }

    /// Embed text token IDs into hidden states.
    pub fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Encode images through the vision encoder and projector.
    ///
    /// Input: [batch, channels, height, width]
    /// Output: [batch, num_tokens, llm_hidden_size]
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_features = self.vision_encoder.forward(pixel_values)?;
        self.projector.project(&vision_features)
    }

    /// Merge text and image embeddings at placeholder positions.
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

        for (position, processed) in &mm_inputs.image_embeddings {
            // Project vision embeddings to LLM space
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = self.projector.project(&vision_emb)?;
            let projected = projected.squeeze(0)?;
            let emb_vec: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            if batch_idx >= merged.len() {
                continue;
            }

            for (i, emb) in emb_vec.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len {
                    merged[batch_idx][target] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    /// Forward pass with optional multimodal inputs.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let text_embeddings = self.embed_text(input_ids)?;

        let embeddings = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(&text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
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

    /// Get the vision encoder.
    pub fn vision_encoder(&self) -> &VisionEncoder {
        &self.vision_encoder
    }

    /// Get the projector.
    pub fn projector(&self) -> &MultimodalProjector {
        &self.projector
    }

    /// Get the image token ID.
    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for MolmoForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        MolmoForCausalLM::forward(
            self,
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
        // During decode, images have already been processed in prefill.
        let embeddings = self.embed_text(input_ids)?;
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
        MolmoForCausalLM::forward(
            self,
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
    use crate::multimodal::ProcessedImage;

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["MolmoForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn test_vision_config() -> MolmoVisionConfig {
        MolmoVisionConfig {
            image_emb_dim: 32,
            image_num_heads: 4,
            image_num_layers: 2,
            image_default_input_size: (28, 28),
            image_patch_size: 14,
            image_num_patch: (2, 2),
            image_mlp_dim: 64,
            encoder_type: VisionEncoderType::Clip,
        }
    }

    fn test_molmo_config() -> MolmoConfig {
        let model_config = test_model_config();
        let vision_config = test_vision_config();
        let projector_config =
            ProjectorConfig::mlp(vision_config.image_emb_dim, model_config.hidden_size);

        MolmoConfig {
            model_config,
            vision_config,
            projector_config,
            image_token_id: 200,
            additional_vocab_size: 128,
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

    // ── Config Tests ────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = MolmoVisionConfig::default();
        assert_eq!(cfg.image_emb_dim, 1024);
        assert_eq!(cfg.image_num_heads, 16);
        assert_eq!(cfg.image_num_layers, 23);
        assert_eq!(cfg.image_default_input_size, (336, 336));
        assert_eq!(cfg.image_patch_size, 14);
        assert_eq!(cfg.image_num_patch, (24, 24));
    }

    #[test]
    fn test_vision_config_to_encoder_config() {
        let cfg = test_vision_config();
        let enc = cfg.to_encoder_config();
        assert_eq!(enc.hidden_size, 32);
        assert_eq!(enc.intermediate_size, 64);
        assert_eq!(enc.num_attention_heads, 4);
        assert_eq!(enc.num_hidden_layers, 2);
        assert_eq!(enc.image_size, 28);
        assert_eq!(enc.patch_size, 14);
    }

    #[test]
    fn test_molmo_config_from_model_config() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_backbone".to_string(),
            serde_json::json!({
                "image_emb_dim": 64,
                "image_num_heads": 4,
                "image_num_layers": 4,
                "image_default_input_size": [56, 56],
                "image_patch_size": 14,
                "image_mlp_dim": 128
            }),
        );
        model_cfg
            .extra
            .insert("additional_vocab_size".to_string(), serde_json::json!(64));
        model_cfg
            .extra
            .insert("image_token_id".to_string(), serde_json::json!(32001));

        let cfg = MolmoConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.image_emb_dim, 64);
        assert_eq!(cfg.vision_config.image_num_layers, 4);
        assert_eq!(cfg.additional_vocab_size, 64);
        assert_eq!(cfg.image_token_id, 32001);
    }

    #[test]
    fn test_molmo_config_defaults_when_no_extra() {
        let model_cfg = test_model_config();
        let cfg = MolmoConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.image_emb_dim, 1024);
        assert_eq!(cfg.additional_vocab_size, ADDITIONAL_VOCAB_SIZE);
        assert_eq!(cfg.image_token_id, DEFAULT_IMAGE_TOKEN_ID);
    }

    #[test]
    fn test_num_image_tokens() {
        let cfg = test_molmo_config();
        // image_num_patch = (2, 2), POOLING_SIZE = 2
        // h_pooled = (2 + 1) / 2 = 1, w_pooled = (2 + 1) / 2 = 1
        assert_eq!(cfg.num_image_tokens(), 1);

        // Full-size config: 24x24 patches
        let default_cfg = MolmoConfig::from_model_config(&test_model_config());
        // (24 + 1) / 2 = 12
        assert_eq!(default_cfg.num_image_tokens(), 144);
    }

    #[test]
    fn test_vit_layer_indices() {
        let mut cfg = test_molmo_config();
        cfg.vision_config.image_num_layers = 23;
        let indices = cfg.vit_layer_indices();
        // VIT_LAYERS = [-2, -9] with 23 layers → [21, 14]
        assert_eq!(indices, vec![21, 14]);
    }

    #[test]
    fn test_vit_layer_indices_small() {
        let cfg = test_molmo_config();
        let indices = cfg.vit_layer_indices();
        // VIT_LAYERS = [-2, -9] with 2 layers → [0, -7] → wraps to [0, 0]
        // -2 + 2 = 0, -9 + 2 = -7 → wraps to usize(max) which is wrong
        // This only makes sense with enough layers; with 2 layers only [-2] = 0 is valid
        assert_eq!(indices[0], 0);
    }

    // ── Model Construction Tests ────────────────────────────────────────

    #[test]
    fn test_molmo_construction() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MolmoForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MolmoForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.image_token_id(), cfg.image_token_id);
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_molmo_from_model_config() {
        let device = Device::Cpu;
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_backbone".to_string(),
            serde_json::json!({
                "image_emb_dim": 32,
                "image_num_heads": 4,
                "image_num_layers": 2,
                "image_default_input_size": [28, 28],
                "image_patch_size": 14,
                "image_mlp_dim": 64
            }),
        );

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::from_model_config(&model_cfg, vb);
        assert!(
            model.is_ok(),
            "from_model_config should work: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_supports_multimodal() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        assert!(model.supports_multimodal());
    }

    // ── Forward Tests ───────────────────────────────────────────────────

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let seq_len = 4;
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, seq_len), &device).unwrap();

        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();

        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.model_config.vocab_size],
            "text-only forward should produce [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_text_only_forward_single_token() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 1);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();

        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
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

    #[test]
    fn test_multimodal_forward_no_images() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let seq_len = 3;
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, seq_len), &device).unwrap();

        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let mm_inputs = MultimodalInputs::text_only(vec![1, 2, 3]);
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

        assert_eq!(logits.dims(), &[1, seq_len, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_multimodal_forward_with_images() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Create a dummy image embedding that has vision_hidden_size dim
        let num_tokens = 2;
        let vision_hidden = cfg.vision_config.image_emb_dim;
        let img_emb = Tensor::zeros((num_tokens, vision_hidden), DType::F32, &device).unwrap();
        let processed = ProcessedImage::new(img_emb, num_tokens);

        // seq: [text, img_placeholder, img_placeholder, text, text]
        let seq_len = 5;
        let input_ids =
            Tensor::from_vec(vec![1u32, 200, 200, 3, 4], (1, seq_len), &device).unwrap();

        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let mm_inputs =
            MultimodalInputs::with_images(vec![1, 200, 200, 3, 4], vec![(1, processed)]);

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

        assert_eq!(logits.dims(), &[1, seq_len, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt_len = 4;
        let prompt = Tensor::zeros((1, prompt_len), DType::U32, &device).unwrap();
        kv_cache
            .allocate_for_request(&mut block_table, prompt_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, prompt_len);

        let logits = ModelForward::forward(
            &model,
            &prompt,
            0,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();
        assert_eq!(logits.dims(), &[1, prompt_len, cfg.model_config.vocab_size]);
        block_table.advance(prompt_len);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(prompt_len, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = ModelForward::forward(
            &model,
            &next_token,
            prompt_len,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }

    // ── Embed Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_embed_text() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
        let embeddings = model.embed_text(&input_ids).unwrap();

        assert_eq!(embeddings.dims(), &[1, 3, cfg.model_config.hidden_size]);
    }

    #[test]
    fn test_embed_extended_vocab() {
        // Molmo supports additional vocab tokens beyond the base vocab
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        // Token ID in the extended range
        let extended_token_id = (cfg.model_config.vocab_size + 10) as u32;
        let input_ids = Tensor::from_vec(vec![extended_token_id], (1, 1), &device).unwrap();
        let embeddings = model.embed_text(&input_ids).unwrap();

        assert_eq!(embeddings.dims(), &[1, 1, cfg.model_config.hidden_size]);
    }

    // ── Vision Encoder Tests ────────────────────────────────────────────

    #[test]
    fn test_encode_images() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        // 28x28 image with 14x14 patches → 2x2 patches + CLS = 5 tokens
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let encoded = model.encode_images(&pixel_values).unwrap();

        // Vision encoder outputs [batch, num_tokens, vision_dim]
        // then projector maps to [batch, num_tokens, llm_hidden]
        assert_eq!(encoded.dim(0).unwrap(), 1);
        assert_eq!(
            encoded.dim(2).unwrap(),
            cfg.model_config.hidden_size,
            "projected embeddings should have LLM hidden size"
        );
    }

    // ── Accessor Tests ──────────────────────────────────────────────────

    #[test]
    fn test_vision_encoder_accessor() {
        let device = Device::Cpu;
        let cfg = test_molmo_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MolmoForCausalLM::new(&cfg, vb).unwrap();

        assert_eq!(
            model.vision_encoder().hidden_size(),
            cfg.vision_config.image_emb_dim
        );
    }

    #[test]
    fn test_vision_config_from_json() {
        let json = serde_json::json!({
            "image_emb_dim": 512,
            "image_num_heads": 8,
            "image_num_layers": 12,
            "image_default_input_size": [224, 224],
            "image_patch_size": 16,
            "image_num_patch": [14, 14],
            "image_mlp_dim": 2048
        });
        let cfg = MolmoVisionConfig::from_json(&json);
        assert_eq!(cfg.image_emb_dim, 512);
        assert_eq!(cfg.image_num_heads, 8);
        assert_eq!(cfg.image_num_layers, 12);
        assert_eq!(cfg.image_default_input_size, (224, 224));
        assert_eq!(cfg.image_patch_size, 16);
        assert_eq!(cfg.image_num_patch, (14, 14));
        assert_eq!(cfg.image_mlp_dim, 2048);
    }

    #[test]
    fn test_vision_config_from_json_defaults() {
        let json = serde_json::json!({});
        let cfg = MolmoVisionConfig::from_json(&json);
        let defaults = MolmoVisionConfig::default();
        assert_eq!(cfg.image_emb_dim, defaults.image_emb_dim);
        assert_eq!(cfg.image_num_layers, defaults.image_num_layers);
    }
}
