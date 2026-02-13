//! DeepSeek VL V2 vision-language model implementation.
//!
//! DeepSeek VL V2 combines a SigLIP vision encoder with a DeepSeek V2/V3 MoE
//! language model via a downsample MLP projector that groups 2x2 neighboring
//! patches into one token, achieving 4x spatial reduction.
//!
//! # Architecture
//!
//! 1. SigLIP ViT encodes images into patch embeddings
//! 2. Downsample MLP projector: unfold 2x2 patches -> Linear -> GELU -> Linear
//! 3. Learned `image_newline` and `view_seperator` parameters for view boundaries
//! 4. DeepSeek V2/V3 MoE language model generates text conditioned on merged embeddings
//!
//! # Weight mapping
//!
//! - `vision.*` -> SigLIP vision encoder (via timm in reference; our SigLIP encoder)
//! - `projector.layers.0` -> first linear (input_dim * downsample_ratio^2, n_embed * mlp_ratio)
//! - `projector.layers.1` -> GELU (not a weight)
//! - `projector.layers.2` -> second linear (n_embed * mlp_ratio, n_embed)
//! - `image_newline` -> learned newline parameter
//! - `view_seperator` -> learned view separator parameter
//! - `language_model.*` -> DeepSeek backbone
//!
//! Reference: https://arxiv.org/abs/2412.10302

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::deepseek::DeepSeekForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// DeepSeek VL V2 model configuration.
#[derive(Debug, Clone)]
pub struct DeepSeekVLV2Config {
    /// Language model configuration.
    pub model_config: ModelConfig,
    /// Vision encoder configuration (SigLIP).
    pub vision_config: VisionEncoderConfig,
    /// Downsample ratio for 2D unfold (default 2 -> groups 2x2 patches).
    pub downsample_ratio: usize,
    /// MLP ratio for projector intermediate size (default 1).
    pub mlp_ratio: usize,
    /// MLP depth for projector (default 2).
    pub mlp_depth: usize,
    /// Image token ID used as placeholder.
    pub image_token_id: u32,
}

impl DeepSeekVLV2Config {
    /// Parse DeepSeek VL V2 config from a ModelConfig.
    ///
    /// Extracts `vision_config` and projector settings from `cfg.extra`.
    /// Falls back to SigLIP SO400M/14 @ 384px defaults.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 384,
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

        // Projector config from extra.projector_config or extra top-level
        let projector_cfg = cfg.extra.get("projector_config");

        let downsample_ratio = projector_cfg
            .and_then(|pc| pc.get("downsample_ratio"))
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.extra.get("downsample_ratio").and_then(|v| v.as_u64()))
            .unwrap_or(2) as usize;

        let mlp_ratio = projector_cfg
            .and_then(|pc| pc.get("mlp_ratio"))
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.extra.get("mlp_ratio").and_then(|v| v.as_u64()))
            .unwrap_or(1) as usize;

        let mlp_depth = projector_cfg
            .and_then(|pc| pc.get("depth"))
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.extra.get("projector_depth").and_then(|v| v.as_u64()))
            .unwrap_or(2) as usize;

        let image_token_id = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.extra.get("image_token_id").and_then(|v| v.as_u64()))
            .unwrap_or(100015) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            downsample_ratio,
            mlp_ratio,
            mlp_depth,
            image_token_id,
        }
    }

    /// Number of patches per side before downsampling.
    pub fn patches_per_side(&self) -> usize {
        self.vision_config.image_size / self.vision_config.patch_size
    }

    /// Number of output tokens per side after downsampling.
    pub fn tokens_per_side(&self) -> usize {
        // Ceiling division in case of non-divisible sizes
        self.patches_per_side().div_ceil(self.downsample_ratio)
    }

    /// Total number of output tokens per image (single tile) after downsampling.
    pub fn num_image_tokens(&self) -> usize {
        let t = self.tokens_per_side();
        t * t
    }

    /// Projector input dimension (vision_hidden * downsample_ratio^2).
    pub fn projector_input_dim(&self) -> usize {
        self.vision_config.hidden_size * self.downsample_ratio * self.downsample_ratio
    }

    /// Projector intermediate dimension (n_embed * mlp_ratio).
    pub fn projector_intermediate_dim(&self) -> usize {
        self.model_config.hidden_size * self.mlp_ratio
    }
}

// ─── Downsample MLP Projector ───────────────────────────────────────────────

/// MLP projector with 2D downsampling for DeepSeek VL V2.
///
/// Groups `downsample_ratio x downsample_ratio` neighboring patches into a
/// single token by concatenating features, then projects through a multi-layer
/// MLP with GELU activations.
struct DownsampleMlpProjector {
    /// MLP layers: alternating Linear and GELU.
    /// For depth=2: [Linear(input_dim, intermediate), GELU, Linear(intermediate, n_embed)]
    layers: Vec<Linear>,
    downsample_ratio: usize,
}

impl DownsampleMlpProjector {
    fn new(
        input_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
        depth: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::new();

        if depth == 1 {
            // Single linear layer
            layers.push(linear(input_dim, output_dim, vb_layers.pp(0))?);
        } else {
            // First layer: input_dim -> intermediate
            layers.push(linear(input_dim, intermediate_dim, vb_layers.pp(0))?);
            // Middle layers (if depth > 2)
            let mut weight_idx = 1;
            for _ in 1..depth - 1 {
                // Skip GELU (index weight_idx is GELU, weight_idx+1 is Linear)
                weight_idx += 1;
                layers.push(linear(
                    intermediate_dim,
                    intermediate_dim,
                    vb_layers.pp(weight_idx),
                )?);
                weight_idx += 1;
            }
            // Last layer: intermediate -> output
            // GELU at index (weight_idx), then Linear at (weight_idx + 1)
            weight_idx += 1;
            layers.push(linear(
                intermediate_dim,
                output_dim,
                vb_layers.pp(weight_idx),
            )?);
        }

        // Extract downsample_ratio from input_dim / output_dim relationship
        // Actually, we take it as a constructor parameter
        Ok(Self {
            layers,
            downsample_ratio: 0, // Will be set by the model
        })
    }

    fn with_downsample_ratio(mut self, ratio: usize) -> Self {
        self.downsample_ratio = ratio;
        self
    }

    /// Forward pass with 2D downsampling.
    ///
    /// Input: `[batch, num_patches, vision_hidden]`
    /// Output: `[batch, num_patches / downsample_ratio^2, llm_hidden]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, hw, input_dim) = x.dims3()?;
        let h = (hw as f64).sqrt() as usize;
        let w = h;

        // Compute padding if needed
        let pad = if !h.is_multiple_of(self.downsample_ratio) {
            self.downsample_ratio - h % self.downsample_ratio
        } else {
            0
        };

        // Reshape to spatial: [batch, h, w, dim]
        let x = x.reshape((batch_size, h, w, input_dim))?;

        // Apply padding if needed
        let x = if pad > 0 {
            let padded_h = h + pad;

            // Pad height by concatenating zero rows
            let zero_rows = Tensor::zeros((batch_size, pad, w, input_dim), x.dtype(), x.device())?;
            let x = Tensor::cat(&[&x, &zero_rows], 1)?;

            // Pad width by concatenating zero columns
            let zero_cols = Tensor::zeros(
                (batch_size, padded_h, pad, input_dim),
                x.dtype(),
                x.device(),
            )?;
            Tensor::cat(&[&x, &zero_cols], 2)?
        } else {
            x
        };

        let actual_h = h + pad;
        let actual_w = w + pad;
        let r = self.downsample_ratio;
        let out_h = actual_h / r;
        let out_w = actual_w / r;

        // Permute to [batch, dim, h, w] for unfold-like operation
        let x = x.permute((0, 3, 1, 2))?;

        // Simulate unfold with kernel_size=r, stride=r:
        // Reshape to [batch, dim, out_h, r, out_w, r]
        let x = x.reshape((batch_size, input_dim, out_h, r, out_w, r))?;

        // Permute to [batch, out_h, out_w, dim, r, r]
        let x = x.permute((0, 2, 4, 1, 3, 5))?;

        // Flatten to [batch, out_h * out_w, dim * r * r]
        let x = x.reshape((batch_size, out_h * out_w, input_dim * r * r))?;

        // Apply MLP layers with GELU between each pair
        let mut hidden = self.layers[0].forward(&x)?;
        for layer in self.layers.iter().skip(1) {
            hidden = hidden.gelu_erf()?;
            hidden = layer.forward(&hidden)?;
        }

        Ok(hidden)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// DeepSeek VL V2 model for conditional generation.
///
/// Combines SigLIP vision encoder + downsample MLP projector + DeepSeek V2/V3
/// MoE language model. Supports global and local view processing for images,
/// with learned separator and newline tokens.
pub struct DeepSeekVLV2ForConditionalGeneration {
    vision_encoder: VisionEncoder,
    projector: DownsampleMlpProjector,
    language_model: DeepSeekForCausalLM,
    /// Learned newline embedding for image token sequences.
    image_newline: Tensor,
    /// Learned view separator embedding.
    view_seperator: Tensor,
    #[allow(dead_code)]
    image_token_id: u32,
    #[allow(dead_code)]
    downsample_ratio: usize,
    device: Device,
    dtype: DType,
}

impl DeepSeekVLV2ForConditionalGeneration {
    pub fn new(cfg: &DeepSeekVLV2Config, vb: VarBuilder) -> Result<Self> {
        let vision_encoder = VisionEncoder::new(&cfg.vision_config, vb.pp("vision"))?;

        let input_dim = cfg.projector_input_dim();
        let intermediate_dim = cfg.projector_intermediate_dim();
        let output_dim = cfg.model_config.hidden_size;

        let projector = DownsampleMlpProjector::new(
            input_dim,
            intermediate_dim,
            output_dim,
            cfg.mlp_depth,
            vb.pp("projector"),
        )?
        .with_downsample_ratio(cfg.downsample_ratio);

        let language_model = DeepSeekForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        // Learned special token embeddings
        let image_newline = vb.get(output_dim, "image_newline")?;
        let view_seperator = vb.get(output_dim, "view_seperator")?;

        Ok(Self {
            vision_encoder,
            projector,
            language_model,
            image_newline,
            view_seperator,
            image_token_id: cfg.image_token_id,
            downsample_ratio: cfg.downsample_ratio,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create from a generic ModelConfig.
    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vlm_cfg = DeepSeekVLV2Config::from_model_config(cfg);
        Self::new(&vlm_cfg, vb)
    }

    /// Encode images through vision encoder + projector.
    ///
    /// Input: `[batch, channels, height, width]`
    /// Output: `[batch, num_tokens_after_downsample, llm_hidden]`
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_features = self.vision_encoder.forward(pixel_values)?;
        self.projector.forward(&vision_features)
    }

    /// Get the image newline parameter (for testing).
    #[allow(dead_code)]
    pub fn image_newline(&self) -> &Tensor {
        &self.image_newline
    }

    /// Get the view separator parameter (for testing).
    #[allow(dead_code)]
    pub fn view_seperator(&self) -> &Tensor {
        &self.view_seperator
    }

    /// Merge text embeddings with pre-processed multimodal image embeddings.
    ///
    /// For each image in the multimodal inputs:
    /// 1. Project the vision embedding via the downsample MLP projector
    /// 2. Replace text embeddings at the image position
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
            // Project vision embeddings through downsample projector
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = self.projector.forward(&vision_emb)?;
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

impl crate::engine::ModelForward for DeepSeekVLV2ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Text-only forward: delegate to DeepSeek LM
        self.language_model.forward_with_embeddings(
            &self.language_model.embed_text(input_ids)?,
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
    use crate::kv_cache::mla_cache_config::MLACacheConfig;
    use crate::multimodal::ProcessedImage;

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // MLA config for DeepSeek backbone
        extra.insert(
            "qk_nope_head_dim".into(),
            serde_json::Value::Number(16.into()),
        );
        extra.insert(
            "qk_rope_head_dim".into(),
            serde_json::Value::Number(8.into()),
        );
        extra.insert("v_head_dim".into(), serde_json::Value::Number(16.into()));
        extra.insert("kv_lora_rank".into(), serde_json::Value::Number(32.into()));

        // Vision config
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

        // Projector config
        extra.insert(
            "projector_config".to_string(),
            serde_json::json!({
                "downsample_ratio": 2,
                "mlp_ratio": 1,
                "depth": 2
            }),
        );

        extra.insert("image_token_index".to_string(), serde_json::json!(100015));

        ModelConfig {
            architectures: vec!["DeepseekVLV2ForConditionalGeneration".to_string()],
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 24,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn test_vlm_config() -> DeepSeekVLV2Config {
        DeepSeekVLV2Config::from_model_config(&test_model_config())
    }

    fn create_mla_cache_manager(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32, // kv_lora_rank
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            cfg.num_attention_heads,
            4,  // block_size
            16, // num_blocks
            cfg.num_hidden_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).unwrap()
    }

    // ── Config Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_config_parsing() {
        let cfg = test_vlm_config();
        assert_eq!(cfg.vision_config.hidden_size, 32);
        assert_eq!(cfg.vision_config.image_size, 28);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.downsample_ratio, 2);
        assert_eq!(cfg.mlp_ratio, 1);
        assert_eq!(cfg.mlp_depth, 2);
        assert_eq!(cfg.image_token_id, 100015);
    }

    #[test]
    fn test_config_defaults() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.remove("vision_config");
        model_cfg.extra.remove("projector_config");
        model_cfg.extra.remove("image_token_index");

        let cfg = DeepSeekVLV2Config::from_model_config(&model_cfg);

        // SigLIP SO400M/14 @ 384px defaults
        assert_eq!(cfg.vision_config.hidden_size, 1152);
        assert_eq!(cfg.vision_config.image_size, 384);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.downsample_ratio, 2);
        assert_eq!(cfg.mlp_ratio, 1);
        assert_eq!(cfg.mlp_depth, 2);
        assert_eq!(cfg.image_token_id, 100015);
    }

    #[test]
    fn test_config_computed_dims() {
        let cfg = test_vlm_config();

        // 28 / 14 = 2 patches per side
        assert_eq!(cfg.patches_per_side(), 2);

        // With downsample_ratio=2: 2 / 2 = 1 token per side
        assert_eq!(cfg.tokens_per_side(), 1);

        // Total output tokens: 1 * 1 = 1
        assert_eq!(cfg.num_image_tokens(), 1);

        // Projector input dim: 32 * 2 * 2 = 128
        assert_eq!(cfg.projector_input_dim(), 128);

        // Projector intermediate dim: 128 * 1 = 128
        assert_eq!(cfg.projector_intermediate_dim(), 128);
    }

    #[test]
    fn test_config_computed_dims_larger() {
        let mut cfg = test_vlm_config();
        cfg.vision_config.image_size = 384;
        cfg.vision_config.patch_size = 14;
        cfg.vision_config.hidden_size = 1152;
        cfg.downsample_ratio = 2;

        // 384 / 14 = 27 patches per side (integer division)
        assert_eq!(cfg.patches_per_side(), 27);

        // With downsample_ratio=2: ceil(27/2) = 14 tokens per side
        assert_eq!(cfg.tokens_per_side(), 14);

        // Projector input dim: 1152 * 4 = 4608
        assert_eq!(cfg.projector_input_dim(), 4608);
    }

    // ── Model Construction Tests ─────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "DeepSeek VL V2 should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let model_cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekVLV2ForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(
            model.is_ok(),
            "from_model_config should work: {:?}",
            model.err()
        );
    }

    // ── Projector Tests ──────────────────────────────────────────────────

    #[test]
    fn test_downsample_projector_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // vision_hidden=32, downsample_ratio=2 -> input_dim = 32*4 = 128
        // intermediate = 128, output = 128
        let projector = DownsampleMlpProjector::new(128, 128, 128, 2, vb)
            .unwrap()
            .with_downsample_ratio(2);

        // 4 patches = 2x2 grid, downsample 2x2 -> 1 token
        let input = Tensor::randn(0f32, 1.0, (1, 4, 32), &device).unwrap();
        let output = projector.forward(&input).unwrap();

        // After 2x2 downsample: 1 token, dim = 32 * 4 = 128 -> projected to 128
        assert_eq!(output.dims(), &[1, 1, 128]);
    }

    #[test]
    fn test_downsample_projector_4x_reduction() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // 16 patches = 4x4 grid, downsample 2x2 -> 4 tokens
        let projector = DownsampleMlpProjector::new(128, 64, 64, 2, vb)
            .unwrap()
            .with_downsample_ratio(2);

        let input = Tensor::randn(0f32, 1.0, (1, 16, 32), &device).unwrap();
        let output = projector.forward(&input).unwrap();

        // 16 patches / 4 (2x2 downsample) = 4 output tokens
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    // ── Forward Tests ────────────────────────────────────────────────────

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_cache = create_mla_cache_manager(&cfg.model_config, &device);
        let mut block_table = BlockTable::new(4);
        kv_cache.allocate_for_request(&mut block_table, 4).unwrap();
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
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
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_cache = create_mla_cache_manager(&cfg.model_config, &device);
        let mut block_table = BlockTable::new(4);
        kv_cache.allocate_for_request(&mut block_table, 4).unwrap();
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
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
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_cache = create_mla_cache_manager(&cfg.model_config, &device);

        // 1 image token (after 2x2 downsample from 2x2 patches) + 7 text tokens = 8 total
        let seq_len = 8;
        let mut block_table = BlockTable::new(4);
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();

        // Image embedding: 4 patches (2x2), vision_hidden=32
        // After projection this becomes 1 token of dim=128, but we supply raw vision patches
        let num_vision_patches = 4;
        let img_embedding = Tensor::randn(0f32, 1.0, (num_vision_patches, 32), &device).unwrap();
        let processed = ProcessedImage::new(img_embedding, cfg.num_image_tokens());

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
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_cache = create_mla_cache_manager(&cfg.model_config, &device);

        // block_size=4: offset=3 fits in 1 block, offset=2 fits in 1 block
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 3,
                block_ids: vec![0],
                slot_mapping: vec![3],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 2,
                block_ids: vec![1],
                slot_mapping: vec![2],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_cache = create_mla_cache_manager(&cfg.model_config, &device);

        // Prefill with 3 tokens
        let mut block_table = BlockTable::new(4);
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.model_config.vocab_size]);
        block_table.advance(3);

        // Decode step
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let decode_input = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let sequences = vec![DecodeSequenceMetadata {
            request_id: 0,
            seqlen_offset: 3,
            block_ids: block_table.block_ids().to_vec(),
            slot_mapping: vec![3],
        }];

        let logits = model
            .forward_decode_batch(&decode_input, &sequences, &mut kv_cache)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_vision_encoder_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        // SigLIP: 28x28 image, 14x14 patches -> 2x2 = 4 patches
        // After 2x2 downsample -> 1 token, dim = 128
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 28, 28), &device).unwrap();
        let encoded = model.encode_images(&pixel_values).unwrap();

        // Output: [1, 1, 128] (1 token after 4x reduction, projected to llm_hidden=128)
        assert_eq!(encoded.dims(), &[1, 1, 128]);
    }

    #[test]
    fn test_special_tokens_exist() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekVLV2ForConditionalGeneration::new(&cfg, vb).unwrap();

        // Learned parameters should have correct shape
        assert_eq!(
            model.image_newline().dims(),
            &[cfg.model_config.hidden_size]
        );
        assert_eq!(
            model.view_seperator().dims(),
            &[cfg.model_config.hidden_size]
        );
    }
}
