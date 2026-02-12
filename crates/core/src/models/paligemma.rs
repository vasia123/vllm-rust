//! PaliGemma vision-language model implementation.
//!
//! PaliGemma combines a SigLIP vision encoder with a Gemma language model
//! using a simple linear projection layer.
//!
//! # Architecture
//!
//! 1. SigLIP vision encoder processes images into patch embeddings (no CLS token)
//! 2. Single linear projection maps vision embeddings to Gemma hidden dimension
//! 3. Vision embeddings are scaled by `hidden_size^{-0.5}` before merging
//! 4. Gemma language model generates text conditioned on merged embeddings
//!
//! # Weight mapping
//!
//! - `vision_tower.*` -> SigLIP vision encoder
//! - `multi_modal_projector.linear` -> linear projection
//! - `language_model.*` -> Gemma backbone
//!
//! Reference: https://arxiv.org/abs/2407.07726

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{
    MultimodalInputs, MultimodalProjector, ProjectorConfig, VisionEncoder, VisionEncoderConfig,
    VisionEncoderType,
};

use super::gemma::GemmaForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// PaliGemma model configuration.
#[derive(Debug, Clone)]
pub struct PaliGemmaConfig {
    /// Language model configuration.
    pub model_config: ModelConfig,
    /// Vision encoder configuration (SigLIP).
    pub vision_config: VisionEncoderConfig,
    /// Projector configuration (single linear layer).
    pub projector_config: ProjectorConfig,
    /// Image token ID used as placeholder (typically 257152).
    pub image_token_id: u32,
}

impl PaliGemmaConfig {
    /// Parse PaliGemma config from a ModelConfig.
    ///
    /// Extracts `vision_config` and `image_token_index` from `cfg.extra`.
    /// Falls back to SigLIP SO400M/14 @ 224px defaults for missing fields.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
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

            // PaliGemma projection_dim may override vision hidden_size for projection
            let _projection_dim = vc
                .get("projection_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);

            VisionEncoderConfig {
                encoder_type: VisionEncoderType::SigLip,
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_hidden_layers,
                image_size,
                patch_size,
                num_channels,
                layer_norm_eps,
            }
        } else {
            defaults
        };

        let image_token_id = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(257152) as u32;

        // PaliGemma uses a single linear projection (no GELU, no 2-layer MLP)
        let projector_config = ProjectorConfig::linear(vision_config.hidden_size, cfg.hidden_size);

        Self {
            model_config: cfg.clone(),
            vision_config,
            projector_config,
            image_token_id,
        }
    }

    /// Number of image tokens per image (SigLIP has no CLS token).
    pub fn num_image_tokens(&self) -> usize {
        self.vision_config.num_patches()
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// PaliGemma model for conditional generation.
///
/// Combines SigLIP vision encoder + linear projector + Gemma language model.
pub struct PaliGemmaForConditionalGeneration {
    vision_encoder: VisionEncoder,
    projector: MultimodalProjector,
    language_model: GemmaForCausalLM,
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl PaliGemmaForConditionalGeneration {
    /// Create a new PaliGemma model.
    pub fn new(cfg: &PaliGemmaConfig, vb: VarBuilder) -> Result<Self> {
        let vision_encoder = VisionEncoder::new(&cfg.vision_config, vb.pp("vision_tower"))?;

        let projector =
            MultimodalProjector::new(&cfg.projector_config, vb.pp("multi_modal_projector"))?;

        let language_model = GemmaForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_encoder,
            projector,
            language_model,
            image_token_id: cfg.image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create a PaliGemma model from a generic ModelConfig.
    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let paligemma_cfg = PaliGemmaConfig::from_model_config(cfg);
        Self::new(&paligemma_cfg, vb)
    }

    /// Encode images: vision encoder -> projector -> scale by hidden_size^{-0.5}.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_features = self.vision_encoder.forward(pixel_values)?;
        let projected = self.projector.project(&vision_features)?;

        // PaliGemma scales projected vision embeddings by hidden_size^{-0.5}
        // (the inverse of Gemma's sqrt(hidden_size) embedding normalization)
        let hidden_size = projected.dim(2)?;
        let scale = (hidden_size as f64).powf(-0.5);
        projected * scale
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

    /// Merge text embeddings with pre-processed multimodal image embeddings.
    ///
    /// For each image in the multimodal inputs:
    /// 1. Project the vision embedding via the linear projector
    /// 2. Scale by hidden_size^{-0.5}
    /// 3. Replace text embeddings at the image position
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, hidden_size) = text_embeddings.dims3()?;
        let scale = (hidden_size as f64).powf(-0.5);
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            // Project vision embeddings: [num_tokens, vision_hidden] -> [num_tokens, llm_hidden]
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = self.projector.project(&vision_emb)?;
            let projected = (projected * scale)?;
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

impl crate::engine::ModelForward for PaliGemmaForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Text-only forward: delegate to Gemma
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
        // Decode: images already processed in prefill, embed text and run LLM
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
    use crate::multimodal::{ProcessedImage, ProjectorType};

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["PaliGemmaForConditionalGeneration".to_string()],
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
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn test_vision_config() -> VisionEncoderConfig {
        VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        }
    }

    fn test_paligemma_config() -> PaliGemmaConfig {
        PaliGemmaConfig {
            model_config: test_model_config(),
            vision_config: test_vision_config(),
            projector_config: ProjectorConfig::linear(32, 64),
            image_token_id: 257152,
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

    // ── Config Parsing Tests ─────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = test_paligemma_config();
        assert_eq!(cfg.image_token_id, 257152);
        assert_eq!(cfg.vision_config.encoder_type, VisionEncoderType::SigLip);
        assert_eq!(cfg.vision_config.hidden_size, 32);
        assert_eq!(cfg.projector_config.projector_type, ProjectorType::Linear);
    }

    #[test]
    fn test_config_from_model_config_defaults() {
        let model_cfg = test_model_config();
        let cfg = PaliGemmaConfig::from_model_config(&model_cfg);

        // Without extra fields, uses SigLIP SO400M/14 @ 224px defaults
        assert_eq!(cfg.vision_config.hidden_size, 1152);
        assert_eq!(cfg.vision_config.intermediate_size, 4304);
        assert_eq!(cfg.vision_config.num_attention_heads, 16);
        assert_eq!(cfg.vision_config.num_hidden_layers, 27);
        assert_eq!(cfg.vision_config.image_size, 224);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.image_token_id, 257152);
        assert_eq!(cfg.projector_config.projector_type, ProjectorType::Linear);
        assert_eq!(cfg.projector_config.vision_hidden_size, 1152);
        assert_eq!(cfg.projector_config.llm_hidden_size, 64);
    }

    #[test]
    fn test_config_from_model_config_with_vision_config() {
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
                "num_channels": 3,
                "layer_norm_eps": 1e-6
            }),
        );
        model_cfg
            .extra
            .insert("image_token_index".to_string(), serde_json::json!(257152));

        let cfg = PaliGemmaConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.hidden_size, 32);
        assert_eq!(cfg.vision_config.intermediate_size, 64);
        assert_eq!(cfg.vision_config.image_size, 28);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.image_token_id, 257152);
        assert_eq!(cfg.projector_config.vision_hidden_size, 32);
        assert_eq!(cfg.projector_config.llm_hidden_size, 64);
    }

    #[test]
    fn test_config_custom_image_token_id() {
        let mut model_cfg = test_model_config();
        model_cfg
            .extra
            .insert("image_token_index".to_string(), serde_json::json!(99999));

        let cfg = PaliGemmaConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.image_token_id, 99999);
    }

    #[test]
    fn test_config_num_image_tokens() {
        let cfg = test_paligemma_config();
        // 28/14 = 2, 2*2 = 4 patches (no CLS for SigLIP)
        assert_eq!(cfg.num_image_tokens(), 4);
    }

    #[test]
    fn test_config_num_image_tokens_224() {
        let model_cfg = test_model_config();
        let cfg = PaliGemmaConfig::from_model_config(&model_cfg);
        // 224/14 = 16, 16*16 = 256 patches
        assert_eq!(cfg.num_image_tokens(), 256);
    }

    #[test]
    fn test_config_siglip_no_cls() {
        let cfg = test_paligemma_config();
        // SigLIP seq_len == num_patches (no CLS token)
        assert_eq!(cfg.vision_config.seq_len(), cfg.vision_config.num_patches());
    }

    // ── Projector Tests ──────────────────────────────────────────────────

    #[test]
    fn test_projector_is_linear() {
        let cfg = test_paligemma_config();
        assert_eq!(cfg.projector_config.projector_type, ProjectorType::Linear);
        assert!(cfg.projector_config.intermediate_size.is_none());
    }

    #[test]
    fn test_projector_dimensions() {
        let cfg = test_paligemma_config();
        assert_eq!(cfg.projector_config.vision_hidden_size, 32);
        assert_eq!(cfg.projector_config.llm_hidden_size, 64);
    }

    // ── Model Construction Tests ─────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PaliGemma should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.supports_multimodal());
        assert_eq!(model.image_token_id(), 257152);
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
                "patch_size": 14
            }),
        );

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(
            model.is_ok(),
            "from_model_config should work: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_supports_multimodal() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();
        assert!(model.supports_multimodal());
    }

    // ── Forward Tests ────────────────────────────────────────────────────

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(
            logits.dims(),
            &[1, 4, cfg.model_config.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

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

        assert_eq!(logits.dim(0).unwrap(), 2, "batch dim should be 2");
    }

    #[test]
    fn test_multimodal_forward_no_images() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();

        // No multimodal inputs -> text-only path
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

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_multimodal_forward_with_images() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1, 2, 3], 0);

        // 4 image tokens + 4 text tokens = 8 total
        let seq_len = 8;
        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();

        // Create a fake image embedding with vision_hidden_size=32 (matching test config)
        let num_image_tokens = cfg.num_image_tokens(); // 4
        let img_embedding = Tensor::randn(0f32, 1.0, (num_image_tokens, 32), &device).unwrap();
        let processed = ProcessedImage::new(img_embedding, num_image_tokens);

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

        assert_eq!(logits.dims(), &[1, seq_len, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_vision_encoder_forward() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

        // SigLIP: 28x28 image, 14x14 patches -> 2x2 = 4 patches (no CLS)
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 28, 28), &device).unwrap();
        let encoded = model.encode_images(&pixel_values).unwrap();

        // Output: [1, 4, 64] (4 patches, projected to llm_hidden=64)
        assert_eq!(encoded.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_vision_encoder_scaling() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

        // With zero weights, all projections produce zero, so scaling is also zero.
        // But we verify the shape is correct and no errors occur.
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let encoded = model.encode_images(&pixel_values).unwrap();
        assert_eq!(encoded.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_projector_forward_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let cfg = ProjectorConfig::linear(32, 64);
        let projector = MultimodalProjector::new(&cfg, vb).unwrap();

        assert_eq!(projector.projector_type(), ProjectorType::Linear);

        let input = Tensor::randn(0f32, 1.0, (1, 4, 32), &device).unwrap();
        let output = projector.project(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_paligemma_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaliGemmaForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.model_config.vocab_size]);
        block_table.advance(3);

        // Decode step at seqlen_offset=3
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }
}
