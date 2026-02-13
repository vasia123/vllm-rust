//! Gemma 3 Vision-Language Model.
//!
//! Combines a SigLIP vision encoder with a Gemma3 language model
//! via a multi-modal projector that uses average pooling, Gemma RMSNorm,
//! and a linear projection.
//!
//! # Architecture
//!
//! 1. SigLIP vision encoder processes images into patch embeddings
//! 2. AvgPool2d reduces spatial resolution (e.g., 64×64 → 16×16 = 256 tokens)
//! 3. Gemma RMSNorm normalizes pooled features
//! 4. Linear projection maps vision_hidden → text_hidden
//! 5. Gemma3 language model generates text conditioned on merged embeddings
//!
//! # Weight mapping
//!
//! - `vision_tower.*` → SigLIP vision encoder
//! - `multi_modal_projector.mm_soft_emb_norm.*` → Gemma RMSNorm
//! - `multi_modal_projector.mm_input_projection_weight` → projection weight
//! - `language_model.*` → Gemma3 backbone
//!
//! Reference: Google Gemma 3 technical report

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::gemma3::{Gemma3ForCausalLM, Gemma3RmsNorm};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Gemma3 VLM configuration.
#[derive(Debug, Clone)]
pub struct Gemma3VLMConfig {
    pub model_config: ModelConfig,
    pub vision_config: VisionEncoderConfig,
    /// Number of output tokens per image after pooling (default 256).
    pub mm_tokens_per_image: usize,
    /// Image token ID used as placeholder.
    pub image_token_id: u32,
}

impl Gemma3VLMConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 896,
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
                    .unwrap_or(defaults.intermediate_size as u64) as usize,
                num_attention_heads: vc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_attention_heads as u64) as usize,
                num_hidden_layers: vc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_hidden_layers as u64) as usize,
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

        let mm_tokens_per_image = cfg
            .extra
            .get("mm_tokens_per_image")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        let image_token_id = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(262144) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            mm_tokens_per_image,
            image_token_id,
        }
    }

    /// Number of patches per image side (before pooling).
    pub fn patches_per_image(&self) -> usize {
        self.vision_config.image_size / self.vision_config.patch_size
    }

    /// Number of output tokens per image side (after pooling).
    pub fn tokens_per_side(&self) -> usize {
        (self.mm_tokens_per_image as f64).sqrt() as usize
    }

    /// AvgPool kernel size.
    pub fn pool_kernel_size(&self) -> usize {
        self.patches_per_image() / self.tokens_per_side()
    }
}

// ─── Multi-Modal Projector ──────────────────────────────────────────────────

/// Gemma3 multi-modal projector: AvgPool2d → GemmaRMSNorm → Linear.
struct Gemma3MultiModalProjector {
    mm_input_projection_weight: Tensor,
    mm_soft_emb_norm: Gemma3RmsNorm,
    patches_per_image: usize,
    kernel_size: usize,
}

impl Gemma3MultiModalProjector {
    fn new(
        vision_hidden: usize,
        text_hidden: usize,
        patches_per_image: usize,
        kernel_size: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mm_input_projection_weight =
            vb.get((vision_hidden, text_hidden), "mm_input_projection_weight")?;

        let mm_soft_emb_norm =
            Gemma3RmsNorm::new(vision_hidden, layer_norm_eps, vb.pp("mm_soft_emb_norm"))?;

        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            kernel_size,
        })
    }

    /// Project vision features to text embedding space.
    ///
    /// Input: `[batch, num_patches, vision_hidden]`
    /// Output: `[batch, mm_tokens_per_image, text_hidden]`
    fn forward(&self, vision_outputs: &Tensor) -> Result<Tensor> {
        let (batch_size, _num_patches, vision_hidden) = vision_outputs.dims3()?;

        // Transpose to [batch, vision_hidden, num_patches] for 2D pooling
        let reshaped = vision_outputs.transpose(1, 2)?;

        // Reshape to [batch, vision_hidden, patches_h, patches_w]
        let reshaped = reshaped.reshape((
            batch_size,
            vision_hidden,
            self.patches_per_image,
            self.patches_per_image,
        ))?;

        // Average pooling: [batch, vision_hidden, tokens_per_side, tokens_per_side]
        let pooled = self.avg_pool_2d(&reshaped)?;

        // Flatten spatial dims: [batch, vision_hidden, mm_tokens_per_image]
        let pooled = pooled.flatten(2, 3)?;

        // Transpose back: [batch, mm_tokens_per_image, vision_hidden]
        let pooled = pooled.transpose(1, 2)?;

        // RMSNorm
        let normed = self.mm_soft_emb_norm.forward(&pooled)?;

        // Linear projection: matmul with [vision_hidden, text_hidden]
        // Candle 3D @ 2D doesn't broadcast — unsqueeze weight for batch matmul
        let weight = self.mm_input_projection_weight.unsqueeze(0)?;
        let projected = normed.matmul(&weight)?;

        Ok(projected)
    }

    /// 2D average pooling with kernel_size == stride (non-overlapping).
    fn avg_pool_2d(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, h, w) = xs.dims4()?;
        let kh = self.kernel_size;
        let kw = self.kernel_size;
        let out_h = h / kh;
        let out_w = w / kw;

        // Reshape to [batch, channels, out_h, kh, out_w, kw]
        let xs = xs.reshape((batch, channels, out_h, kh, out_w, kw))?;
        // Permute to [batch, channels, out_h, out_w, kh, kw]
        let xs = xs.permute([0, 1, 2, 4, 3, 5])?;
        // Flatten kernel dims and mean
        let xs = xs.reshape((batch, channels, out_h, out_w, kh * kw))?;
        xs.mean(4)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Gemma3 Vision-Language model for conditional generation.
pub struct Gemma3ForConditionalGeneration {
    vision_tower: VisionEncoder,
    multi_modal_projector: Gemma3MultiModalProjector,
    language_model: Gemma3ForCausalLM,
    #[allow(dead_code)]
    mm_tokens_per_image: usize,
    #[allow(dead_code)]
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl Gemma3ForConditionalGeneration {
    pub fn new(cfg: &Gemma3VLMConfig, vb: VarBuilder) -> Result<Self> {
        let vision_tower = VisionEncoder::new(&cfg.vision_config, vb.pp("vision_tower"))?;

        let multi_modal_projector = Gemma3MultiModalProjector::new(
            cfg.vision_config.hidden_size,
            cfg.model_config.hidden_size,
            cfg.patches_per_image(),
            cfg.pool_kernel_size(),
            cfg.vision_config.layer_norm_eps,
            vb.pp("multi_modal_projector"),
        )?;

        let language_model =
            Gemma3ForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            mm_tokens_per_image: cfg.mm_tokens_per_image,
            image_token_id: cfg.image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vlm_cfg = Gemma3VLMConfig::from_model_config(cfg);
        Self::new(&vlm_cfg, vb)
    }

    /// Encode images: vision encoder → projector.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_features = self.vision_tower.forward(pixel_values)?;
        self.multi_modal_projector.forward(&vision_features)
    }

    /// Merge text embeddings with pre-processed multimodal image embeddings.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden_size) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = self.multi_modal_projector.forward(&vision_emb)?;
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

impl crate::engine::ModelForward for Gemma3ForConditionalGeneration {
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
        self.language_model
            .forward_decode_batch_with_embeddings(&embeddings, sequences, kv_cache_mgr)
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
        extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::json!(256.0),
        );
        extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));
        extra.insert("mm_tokens_per_image".to_string(), serde_json::json!(4));
        extra.insert("image_token_index".to_string(), serde_json::json!(262144));
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
                "layer_norm_eps": 1e-6
            }),
        );

        ModelConfig {
            architectures: vec!["Gemma3ForConditionalGeneration".to_string()],
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
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_vlm_config() -> Gemma3VLMConfig {
        Gemma3VLMConfig::from_model_config(&test_model_config())
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

    // ── Config Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_vlm_config_parsing() {
        let cfg = test_vlm_config();
        assert_eq!(cfg.vision_config.hidden_size, 32);
        assert_eq!(cfg.vision_config.image_size, 28);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.mm_tokens_per_image, 4);
        assert_eq!(cfg.image_token_id, 262144);
    }

    #[test]
    fn test_vlm_config_computed() {
        let cfg = test_vlm_config();
        // patches_per_image = 28/14 = 2
        assert_eq!(cfg.patches_per_image(), 2);
        // tokens_per_side = sqrt(4) = 2
        assert_eq!(cfg.tokens_per_side(), 2);
        // pool_kernel_size = 2/2 = 1
        assert_eq!(cfg.pool_kernel_size(), 1);
    }

    #[test]
    fn test_vlm_config_defaults() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.remove("mm_tokens_per_image");
        model_cfg.extra.remove("image_token_index");
        model_cfg.extra.remove("vision_config");

        let cfg = Gemma3VLMConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.mm_tokens_per_image, 256);
        assert_eq!(cfg.image_token_id, 262144);
        assert_eq!(cfg.vision_config.image_size, 896);
        assert_eq!(cfg.vision_config.patch_size, 14);
    }

    #[test]
    fn test_vlm_config_default_geometry() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.remove("mm_tokens_per_image");
        model_cfg.extra.remove("vision_config");

        let cfg = Gemma3VLMConfig::from_model_config(&model_cfg);
        // patches_per_image = 896/14 = 64
        assert_eq!(cfg.patches_per_image(), 64);
        // tokens_per_side = sqrt(256) = 16
        assert_eq!(cfg.tokens_per_side(), 16);
        // pool_kernel_size = 64/16 = 4
        assert_eq!(cfg.pool_kernel_size(), 4);
    }

    // ── Model Construction Tests ─────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Gemma3ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "Gemma3 VLM should construct: {:?}", model.err());

        let model = model.unwrap();
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let model_cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Gemma3ForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(model.is_ok(), "from_model_config should work: {:?}", model.err());
    }

    // ── Projector Tests ──────────────────────────────────────────────────

    #[test]
    fn test_projector_forward_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // patches_per_image=2, kernel=1
        let projector = Gemma3MultiModalProjector::new(
            32, 64, 2, 1, 1e-6, vb,
        ).unwrap();

        // Input: [1, 4, 32] (4 patches = 2×2, 32 vision_hidden)
        let input = Tensor::randn(0f32, 1.0, (1, 4, 32), &device).unwrap();
        let output = projector.forward(&input).unwrap();

        // Output: [1, 4, 64] (4 tokens = 2×2 with kernel=1, 64 text_hidden)
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_projector_pooling_reduction() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // patches_per_image=4, kernel=2 → reduces 16 patches to 4
        let projector = Gemma3MultiModalProjector::new(
            32, 64, 4, 2, 1e-6, vb,
        ).unwrap();

        // Input: [1, 16, 32] (16 patches = 4×4)
        let input = Tensor::randn(0f32, 1.0, (1, 16, 32), &device).unwrap();
        let output = projector.forward(&input).unwrap();

        // Output: [1, 4, 64] (4 tokens = 2×2 after pooling)
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_avg_pool_2d() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let projector = Gemma3MultiModalProjector::new(
            1, 1, 4, 2, 1e-6, vb,
        ).unwrap();

        // [1, 1, 4, 4] with known values
        let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = Tensor::from_vec(values, (1, 1, 4, 4), &device).unwrap();
        let pooled = projector.avg_pool_2d(&input).unwrap();

        assert_eq!(pooled.dims(), &[1, 1, 2, 2]);
        let result: Vec<f32> = pooled.flatten_all().unwrap().to_vec1().unwrap();
        // Top-left 2×2: (0+1+4+5)/4 = 2.5
        assert!((result[0] - 2.5).abs() < 1e-6);
        // Top-right 2×2: (2+3+6+7)/4 = 4.5
        assert!((result[1] - 4.5).abs() < 1e-6);
        // Bottom-left 2×2: (8+9+12+13)/4 = 10.5
        assert!((result[2] - 10.5).abs() < 1e-6);
        // Bottom-right 2×2: (10+11+14+15)/4 = 12.5
        assert!((result[3] - 12.5).abs() < 1e-6);
    }

    // ── Forward Tests ────────────────────────────────────────────────────

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_multimodal_forward_no_images() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward_multimodal(
                &input_ids, None, 0, &mut kv_cache, &block_table, &slot_mapping,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_multimodal_forward_with_images() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1, 2, 3], 0);

        // 4 image tokens + 4 text tokens = 8 total
        let seq_len = 8;
        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();

        // SigLIP: 28x28 image, 14x14 patches → 2×2 = 4 patches
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

        assert_eq!(logits.dims(), &[1, seq_len, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3ForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_vision_encoder_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3ForConditionalGeneration::new(&cfg, vb).unwrap();

        // SigLIP: 28×28 image → 2×2 = 4 patches → kernel=1 → 4 output tokens
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 28, 28), &device).unwrap();
        let encoded = model.encode_images(&pixel_values).unwrap();

        // Output: [1, 4, 64] (4 tokens, projected to text_hidden=64)
        assert_eq!(encoded.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.model_config.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }
}
