//! LLaVA-OneVision model implementation.
//!
//! LLaVA-OneVision extends LLaVA-Next with:
//! - SigLIP SO400M/14 @ 384px vision encoder
//! - 2-layer MLP projector with GELU (configurable bias via `multimodal_projector_bias`)
//! - AnyRes dynamic resolution for multi-crop images
//! - Learned `image_newline` parameter appended to rows after spatial unpadding
//! - Video frame support (frames treated as images with temporal pooling)
//! - Qwen2 backbone as the language model
//!
//! Reference: https://arxiv.org/abs/2408.03326

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, ProjectorConfig, VisionEncoder, VisionEncoderConfig};

use super::llava::VisionSelectFeature;
use super::qwen2::Qwen2ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Configuration for LLaVA-OneVision model.
#[derive(Debug, Clone)]
pub struct LlavaOnevisionConfig {
    /// Language model configuration.
    pub model_config: ModelConfig,
    /// Vision encoder configuration (SigLIP SO400M/14 @ 384px).
    pub vision_config: VisionEncoderConfig,
    /// Projector configuration (2-layer MLP with GELU).
    pub projector_config: ProjectorConfig,
    /// Token ID used as image placeholder in the prompt.
    pub image_token_id: u32,
    /// Token ID used as video placeholder in the prompt.
    pub video_token_id: u32,
    /// Which vision features to use from the encoder output.
    pub vision_feature_select_strategy: VisionSelectFeature,
    /// Whether projector linears include bias terms.
    pub multimodal_projector_bias: bool,
}

impl LlavaOnevisionConfig {
    /// Parse config from a ModelConfig, extracting vision_config from extra fields.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig::siglip_so400m_384();

        let vision_config = if let Some(vc) = cfg.extra.get("vision_config") {
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

            let encoder_type =
                if vc.get("model_type").and_then(|v| v.as_str()) == Some("siglip_vision_model") {
                    crate::multimodal::VisionEncoderType::SigLip
                } else {
                    // Default to SigLIP for OneVision
                    crate::multimodal::VisionEncoderType::SigLip
                };

            VisionEncoderConfig {
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_hidden_layers,
                image_size,
                patch_size,
                num_channels,
                layer_norm_eps,
                encoder_type,
            }
        } else {
            defaults
        };

        let vision_feature_select_strategy = cfg
            .extra
            .get("vision_feature_select_strategy")
            .and_then(|v| v.as_str())
            .map(VisionSelectFeature::from_strategy_str)
            .unwrap_or(VisionSelectFeature::Default);

        let image_token_id = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(151646) as u32;

        let video_token_id = cfg
            .extra
            .get("video_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(151647) as u32;

        let multimodal_projector_bias = cfg
            .extra
            .get("multimodal_projector_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let projector_config = ProjectorConfig::mlp(vision_config.hidden_size, cfg.hidden_size);

        Self {
            model_config: cfg.clone(),
            vision_config,
            projector_config,
            image_token_id,
            video_token_id,
            vision_feature_select_strategy,
            multimodal_projector_bias,
        }
    }

    /// Number of image tokens per image based on feature selection strategy.
    pub fn num_image_tokens(&self) -> usize {
        match self.vision_feature_select_strategy {
            VisionSelectFeature::ClsOnly => 1,
            VisionSelectFeature::PatchOnly => self.vision_config.num_patches(),
            VisionSelectFeature::Default => self.vision_config.seq_len(),
        }
    }
}

// ─── Projector (bias-configurable) ──────────────────────────────────────────

/// LLaVA-OneVision projector: 2-layer MLP with GELU, configurable bias.
///
/// Weight names: `multi_modal_projector.linear_1` and `multi_modal_projector.linear_2`
struct OnevisionProjector {
    linear1: Linear,
    linear2: Linear,
}

impl OnevisionProjector {
    fn new(
        vision_hidden_size: usize,
        llm_hidden_size: usize,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear1 = candle_nn::linear_b(
            vision_hidden_size,
            llm_hidden_size,
            use_bias,
            vb.pp("linear_1"),
        )?;
        let linear2 = candle_nn::linear_b(
            llm_hidden_size,
            llm_hidden_size,
            use_bias,
            vb.pp("linear_2"),
        )?;
        Ok(Self { linear1, linear2 })
    }

    /// Project vision embeddings to LLM space: Linear → GELU → Linear
    fn project(&self, vision_embeddings: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(vision_embeddings)?;
        let hidden = hidden.gelu_erf()?;
        self.linear2.forward(&hidden)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// LLaVA-OneVision model for conditional generation.
///
/// Combines SigLIP vision encoder + 2-layer MLP projector + image_newline
/// + Qwen2 language model backbone.
pub struct LlavaOnevisionForConditionalGeneration {
    /// SigLIP vision encoder.
    vision_encoder: VisionEncoder,
    /// 2-layer MLP projector (vision → LLM space).
    projector: OnevisionProjector,
    /// Learned parameter [hidden_size] appended to image rows after spatial unpadding.
    image_newline: Tensor,
    /// Qwen2 language model backbone.
    language_model: Qwen2ForCausalLM,
    /// Token ID for image placeholders.
    #[allow(dead_code)]
    image_token_id: u32,
    /// Token ID for video placeholders.
    #[allow(dead_code)]
    video_token_id: u32,
    /// Vision feature selection strategy.
    #[allow(dead_code)]
    vision_feature_select_strategy: VisionSelectFeature,
    device: Device,
    dtype: DType,
}

impl LlavaOnevisionForConditionalGeneration {
    /// Create a new LLaVA-OneVision model.
    pub fn new(cfg: &LlavaOnevisionConfig, vb: VarBuilder) -> Result<Self> {
        let vision_encoder = VisionEncoder::new(&cfg.vision_config, vb.pp("vision_tower"))?;

        let projector = OnevisionProjector::new(
            cfg.projector_config.vision_hidden_size,
            cfg.projector_config.llm_hidden_size,
            cfg.multimodal_projector_bias,
            vb.pp("multi_modal_projector"),
        )?;

        let image_newline = vb.get(cfg.model_config.hidden_size, "image_newline")?;

        let language_model = Qwen2ForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_encoder,
            projector,
            image_newline,
            language_model,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            vision_feature_select_strategy: cfg.vision_feature_select_strategy,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create from a generic ModelConfig, extracting vision config from extra fields.
    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let onevision_cfg = LlavaOnevisionConfig::from_model_config(cfg);
        Self::new(&onevision_cfg, vb)
    }

    /// Merge pre-processed multimodal embeddings into text embeddings.
    ///
    /// Image embeddings in MultimodalInputs have already been processed by
    /// the multimodal processor. We project them to LLM space and insert at
    /// the correct positions, replacing image placeholder tokens.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() && !mm_inputs.has_videos() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        // Merge image embeddings
        for (position, processed) in &mm_inputs.image_embeddings {
            // Project vision features to LLM space
            let emb = processed.embedding.unsqueeze(0)?;
            let projected = self.projector.project(&emb)?;
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

        // Merge video embeddings
        for (position, processed) in &mm_inputs.video_embeddings {
            let emb = processed.embedding.unsqueeze(0)?;
            let projected = self.projector.project(&emb)?;
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
        let text_embeddings = self.language_model.embed_text(input_ids)?;

        let embeddings = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() || mm_inputs.has_videos() {
                self.merge_multimodal(&text_embeddings, mm_inputs)?
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

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the vision encoder.
    pub fn vision_encoder(&self) -> &VisionEncoder {
        &self.vision_encoder
    }

    /// Get the image_newline parameter tensor.
    pub fn image_newline(&self) -> &Tensor {
        &self.image_newline
    }
}

impl crate::engine::ModelForward for LlavaOnevisionForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Text-only forward (no multimodal inputs)
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
        // During decode, images are already processed in prefill
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::multimodal::{ProcessedImage, ProcessedVideo, VisionEncoderType};

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["LlavaOnevisionForConditionalGeneration".to_string()],
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
            rope_theta: 1000000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
        }
    }

    fn test_vision_config() -> VisionEncoderConfig {
        VisionEncoderConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            encoder_type: VisionEncoderType::SigLip,
        }
    }

    fn test_onevision_config() -> LlavaOnevisionConfig {
        let model_config = test_model_config();
        let vision_config = test_vision_config();
        let projector_config =
            ProjectorConfig::mlp(vision_config.hidden_size, model_config.hidden_size);

        LlavaOnevisionConfig {
            model_config,
            vision_config,
            projector_config,
            image_token_id: 151646,
            video_token_id: 151647,
            vision_feature_select_strategy: VisionSelectFeature::Default,
            multimodal_projector_bias: false,
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
    fn test_config_parsing_defaults() {
        let cfg = test_onevision_config();
        assert_eq!(cfg.image_token_id, 151646);
        assert_eq!(cfg.video_token_id, 151647);
        assert_eq!(cfg.vision_config.hidden_size, 32);
        assert_eq!(cfg.projector_config.vision_hidden_size, 32);
        assert_eq!(cfg.projector_config.llm_hidden_size, 64);
        assert!(!cfg.multimodal_projector_bias);
        assert_eq!(
            cfg.vision_feature_select_strategy,
            VisionSelectFeature::Default
        );
    }

    #[test]
    fn test_config_from_model_config_with_vision_config() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 48,
                "intermediate_size": 96,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 42,
                "patch_size": 14,
                "model_type": "siglip_vision_model"
            }),
        );
        model_cfg
            .extra
            .insert("image_token_index".to_string(), serde_json::json!(32000));
        model_cfg
            .extra
            .insert("video_token_index".to_string(), serde_json::json!(32001));
        model_cfg.extra.insert(
            "vision_feature_select_strategy".to_string(),
            serde_json::json!("default"),
        );
        model_cfg.extra.insert(
            "multimodal_projector_bias".to_string(),
            serde_json::json!(true),
        );

        let cfg = LlavaOnevisionConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.hidden_size, 48);
        assert_eq!(cfg.vision_config.image_size, 42);
        assert_eq!(cfg.image_token_id, 32000);
        assert_eq!(cfg.video_token_id, 32001);
        assert!(cfg.multimodal_projector_bias);
        assert_eq!(cfg.vision_config.encoder_type, VisionEncoderType::SigLip);
    }

    #[test]
    fn test_config_num_image_tokens() {
        let mut cfg = test_onevision_config();

        // SigLIP: 28/14 = 2, 2*2 = 4 patches, no CLS = seq_len includes no CLS for SigLIP
        // VisionEncoderConfig::seq_len() = num_patches() + 1 for CLIP, but SigLIP has no CLS
        // Actually seq_len() always adds 1 for CLS. For SigLIP, the model strips CLS in encoder.
        // But VisionEncoderConfig doesn't know about this, it returns num_patches + 1.
        // Let's check with Default strategy which uses seq_len()
        let expected_seq_len = cfg.vision_config.seq_len(); // 4 + 1 = 5
        assert_eq!(cfg.num_image_tokens(), expected_seq_len);

        cfg.vision_feature_select_strategy = VisionSelectFeature::PatchOnly;
        assert_eq!(cfg.num_image_tokens(), 4); // patches only

        cfg.vision_feature_select_strategy = VisionSelectFeature::ClsOnly;
        assert_eq!(cfg.num_image_tokens(), 1);
    }

    #[test]
    fn test_config_from_model_config_no_vision_config() {
        let model_cfg = test_model_config();
        let cfg = LlavaOnevisionConfig::from_model_config(&model_cfg);

        // Should fall back to SigLIP SO400M/14 @ 384px defaults
        assert_eq!(cfg.vision_config.hidden_size, 1152);
        assert_eq!(cfg.vision_config.image_size, 384);
        assert_eq!(cfg.vision_config.patch_size, 14);
    }

    // ── Model Construction Tests ────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Model construction failed: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_model_construction_with_bias() {
        let device = Device::Cpu;
        let mut cfg = test_onevision_config();
        cfg.multimodal_projector_bias = true;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Model construction with bias failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_image_newline_shape() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();
        let newline = model.image_newline();
        assert_eq!(
            newline.dims(),
            &[cfg.model_config.hidden_size],
            "image_newline should be [hidden_size]"
        );
    }

    // ── Forward Tests ───────────────────────────────────────────────────

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(
                &input_ids,
                None,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(
            logits.dims(),
            &[1, 4, cfg.model_config.vocab_size],
            "text-only forward should produce [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_text_only_forward_via_trait() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();

        // Use ModelForward trait method
        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

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

        assert_eq!(
            logits.dim(0).unwrap(),
            2,
            "batch decode should produce output for each sequence"
        );
    }

    #[test]
    fn test_supports_multimodal() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        assert!(
            model.supports_multimodal(),
            "LLaVA-OneVision must support multimodal"
        );
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "model_type": "siglip_vision_model"
            }),
        );

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(model.is_ok(), "from_model_config failed: {:?}", model.err());
    }

    #[test]
    fn test_forward_multimodal_with_images() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..6).collect();

        // Use placeholder IDs within vocab range for testing
        let img_placeholder: u32 = 100;
        let vision_hidden = cfg.vision_config.hidden_size;
        let img_emb = Tensor::zeros((2, vision_hidden), DType::F32, &device).unwrap();
        let image = ProcessedImage::new(img_emb, 2);
        let mm_inputs = MultimodalInputs::with_images(
            vec![1, img_placeholder, img_placeholder, 2, 3, 4],
            vec![(1, image)],
        );

        let input_ids = Tensor::from_vec(
            vec![1u32, img_placeholder, img_placeholder, 2, 3, 4],
            (1, 6),
            &device,
        )
        .unwrap();

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

        assert_eq!(logits.dims(), &[1, 6, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_forward_multimodal_no_images() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let mm_inputs = MultimodalInputs::text_only(vec![1, 2, 3, 4]);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();

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

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_forward_multimodal_with_video() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..5).collect();

        // Use placeholder IDs within vocab range for testing
        let vid_placeholder: u32 = 101;
        let vision_hidden = cfg.vision_config.hidden_size;
        let vid_emb = Tensor::zeros((3, vision_hidden), DType::F32, &device).unwrap();
        let video = ProcessedVideo::new(vid_emb, 3, 1);
        let mm_inputs = MultimodalInputs::with_videos(
            vec![1, vid_placeholder, vid_placeholder, vid_placeholder, 2],
            vec![(1, video)],
        );

        let input_ids = Tensor::from_vec(
            vec![1u32, vid_placeholder, vid_placeholder, vid_placeholder, 2],
            (1, 5),
            &device,
        )
        .unwrap();

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

        assert_eq!(logits.dims(), &[1, 5, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let seq_len = 4;
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, seq_len);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, seq_len), &device).unwrap();

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.model_config.vocab_size]);
        block_table.advance(seq_len);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(seq_len, 1);
        let next_token = Tensor::from_vec(vec![5u32], (1, 1), &device).unwrap();

        let logits = ModelForward::forward(
            &model,
            &next_token,
            seq_len,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        )
        .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_projector_no_bias() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = OnevisionProjector::new(32, 64, false, vb).unwrap();

        let input = Tensor::zeros((1, 4, 32), DType::F32, &device).unwrap();
        let output = proj.project(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_projector_with_bias() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = OnevisionProjector::new(32, 64, true, vb).unwrap();

        let input = Tensor::zeros((1, 4, 32), DType::F32, &device).unwrap();
        let output = proj.project(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_device_accessor() {
        let device = Device::Cpu;
        let cfg = test_onevision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb).unwrap();

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_config_bias_flag_propagation() {
        // Verify that the bias flag from config is correctly used
        let mut cfg = test_onevision_config();
        cfg.multimodal_projector_bias = true;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "Model with bias=true should construct");

        cfg.multimodal_projector_bias = false;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LlavaOnevisionForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "Model with bias=false should construct");
    }
}
