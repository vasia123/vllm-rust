//! AyaVision / Cohere2Vision vision-language model implementation.
//!
//! Combines a SigLIP vision encoder with a Cohere2 LLM backbone using a
//! pixel-shuffle + SwiGLU projector. Two variants share this implementation:
//!
//! - **AyaVision**: pixel shuffle → LayerNorm → Linear → SwiGLU → Linear
//! - **Cohere2Vision**: pixel shuffle → Linear → SwiGLU → Linear (no LayerNorm)
//!
//! Reference: https://huggingface.co/CohereForAI/aya-vision-8b

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::cohere::CohereForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// AyaVision / Cohere2Vision configuration.
#[derive(Debug, Clone)]
pub struct AyaVisionConfig {
    pub vision_config: VisionEncoderConfig,
    pub downsample_factor: usize,
    /// Intermediate size for SwiGLU in projector (defaults to text hidden_size).
    pub alignment_intermediate_size: usize,
    pub text_hidden_size: usize,
    pub adapter_layer_norm_eps: f64,
    /// Whether to use LayerNorm in the projector (true for AyaVision, false for Cohere2Vision).
    pub use_projector_layernorm: bool,
}

impl AyaVisionConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = if let Some(vc) = cfg.extra.get("vision_config") {
            VisionEncoderConfig {
                encoder_type: VisionEncoderType::SigLip,
                hidden_size: vc
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1152) as usize,
                intermediate_size: vc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4304) as usize,
                num_attention_heads: vc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize,
                num_hidden_layers: vc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(27) as usize,
                image_size: vc.get("image_size").and_then(|v| v.as_u64()).unwrap_or(384) as usize,
                patch_size: vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as usize,
                num_channels: vc.get("num_channels").and_then(|v| v.as_u64()).unwrap_or(3) as usize,
                layer_norm_eps: vc
                    .get("layer_norm_eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1e-6),
            }
        } else {
            VisionEncoderConfig::siglip_so400m_384()
        };

        let downsample_factor = cfg
            .extra
            .get("downsample_factor")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let alignment_intermediate_size = cfg
            .extra
            .get("alignment_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.hidden_size as u64) as usize;

        let adapter_layer_norm_eps = cfg
            .extra
            .get("adapter_layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6);

        // AyaVision has LayerNorm in projector; detect by architecture name
        let use_projector_layernorm = cfg
            .architectures
            .first()
            .map(|a| a != "Cohere2VisionForConditionalGeneration")
            .unwrap_or(true);

        Self {
            vision_config,
            downsample_factor,
            alignment_intermediate_size,
            text_hidden_size: cfg.hidden_size,
            adapter_layer_norm_eps,
            use_projector_layernorm,
        }
    }

    /// Projector input dimension after pixel shuffle.
    pub fn projector_input_dim(&self) -> usize {
        self.vision_config.hidden_size * self.downsample_factor * self.downsample_factor
    }
}

// ─── Projector ──────────────────────────────────────────────────────────────

/// Vision projector: pixel shuffle → [LayerNorm] → Linear → SwiGLU → Linear.
///
/// The SwiGLU gate splits the intermediate tensor in half: `silu(gate) * x`.
/// AyaVision uses LayerNorm; Cohere2Vision omits it.
#[allow(dead_code)]
struct AyaVisionProjector {
    layernorm: Option<LayerNorm>,
    linear_1: Linear,
    linear_2: Linear,
    downsample_factor: usize,
}

#[allow(dead_code)]
impl AyaVisionProjector {
    fn new(cfg: &AyaVisionConfig, vb: VarBuilder) -> Result<Self> {
        let input_dim = cfg.projector_input_dim();
        let layernorm = if cfg.use_projector_layernorm {
            Some(layer_norm(
                input_dim,
                cfg.adapter_layer_norm_eps,
                vb.pp("layernorm"),
            )?)
        } else {
            None
        };
        let linear_1 = candle_nn::linear(
            input_dim,
            cfg.alignment_intermediate_size,
            vb.pp("linear_1"),
        )?;
        // SwiGLU splits intermediate in half, so linear_2 takes half the intermediate size
        let linear_2 = candle_nn::linear(
            cfg.alignment_intermediate_size / 2,
            cfg.text_hidden_size,
            vb.pp("linear_2"),
        )?;
        Ok(Self {
            layernorm,
            linear_1,
            linear_2,
            downsample_factor: cfg.downsample_factor,
        })
    }

    /// Pixel shuffle: merge spatial patches by downsample_factor.
    ///
    /// Input: [B, S, D] where S = H*W patches
    /// Output: [B, S/(f*f), D*f*f] where f = downsample_factor
    fn pixel_shuffle(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_length, _channels) = x.dims3()?;
        let height = (seq_length as f64).sqrt() as usize;
        let width = height;
        let f = self.downsample_factor;

        // [B, S, D] → [B, W, H, D]
        let x = x.reshape((batch_size, width, height, ()))?;
        let channels = x.dim(3)?;

        // [B, W, H, D] → [B, W, H/f, D*f]
        let x = x.reshape((batch_size, width, height / f, channels * f))?;
        // [B, W, H/f, D*f] → [B, H/f, W, D*f]
        let x = x.permute((0, 2, 1, 3))?;
        // [B, H/f, W, D*f] → [B, H/f, W/f, D*f*f]
        let x = x.reshape((batch_size, height / f, width / f, ()))?;
        // [B, H/f, W/f, D*f*f] → [B, W/f, H/f, D*f*f]
        let x = x.permute((0, 2, 1, 3))?;
        // Flatten spatial dims: [B, (W/f)*(H/f), D*f*f]
        let new_seq = (width / f) * (height / f);
        x.reshape((batch_size, new_seq, ()))?.contiguous()
    }

    fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let x = self.pixel_shuffle(image_features)?;
        let x = if let Some(ln) = &self.layernorm {
            ln.forward(&x)?
        } else {
            x
        };
        let hidden = self.linear_1.forward(&x)?;

        // SwiGLU: split in half, apply silu to gate, multiply
        let half_dim = hidden.dim(2)? / 2;
        let x_part = hidden.narrow(2, 0, half_dim)?;
        let gate = hidden.narrow(2, half_dim, half_dim)?;
        let hidden = (candle_nn::ops::silu(&gate)? * x_part)?;

        self.linear_2.forward(&hidden)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// AyaVision / Cohere2Vision model for conditional generation.
///
/// SigLIP vision encoder + pixel shuffle + SwiGLU projector + Cohere2 LLM.
/// AyaVision uses LayerNorm in projector; Cohere2Vision omits it.
pub struct AyaVisionForConditionalGeneration {
    #[allow(dead_code)]
    vision_tower: VisionEncoder,
    #[allow(dead_code)]
    projector: AyaVisionProjector,
    language_model: CohereForCausalLM,
    #[allow(dead_code)]
    config: AyaVisionConfig,
    device: Device,
    dtype: DType,
}

impl AyaVisionForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = AyaVisionConfig::from_model_config(cfg);

        let vision_tower = VisionEncoder::new(&config.vision_config, vb.pp("vision_tower"))?;

        let projector = AyaVisionProjector::new(&config, vb.pp("multi_modal_projector"))?;

        let language_model = CohereForCausalLM::new(cfg, vb.pp("model"))?;

        Ok(Self {
            vision_tower,
            projector,
            language_model,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Merge pre-encoded image embeddings with text embeddings.
    fn merge_multimodal(
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
            let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
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

impl crate::engine::ModelForward for AyaVisionForConditionalGeneration {
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

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "model_type": "siglip_vision_model"
            }),
        );
        extra.insert("downsample_factor".to_string(), serde_json::json!(2));
        extra.insert(
            "alignment_intermediate_size".to_string(),
            serde_json::json!(128),
        );

        ModelConfig {
            architectures: vec!["AyaVisionForConditionalGeneration".to_string()],
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
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_aya_config() {
        let cfg = test_model_config();
        let aya_cfg = AyaVisionConfig::from_model_config(&cfg);
        assert_eq!(aya_cfg.downsample_factor, 2);
        assert_eq!(aya_cfg.alignment_intermediate_size, 128);
        assert_eq!(aya_cfg.text_hidden_size, 64);
        assert!(aya_cfg.use_projector_layernorm);
        assert_eq!(aya_cfg.projector_input_dim(), 256);
    }

    #[test]
    fn test_cohere2_vision_config_no_layernorm() {
        let mut cfg = test_model_config();
        cfg.architectures = vec!["Cohere2VisionForConditionalGeneration".to_string()];
        let aya_cfg = AyaVisionConfig::from_model_config(&cfg);
        assert!(!aya_cfg.use_projector_layernorm);
    }

    #[test]
    fn test_cohere2_vision_model_construction() {
        let device = Device::Cpu;
        let mut cfg = test_model_config();
        cfg.architectures = vec!["Cohere2VisionForConditionalGeneration".to_string()];
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AyaVisionForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Cohere2Vision should construct: {:?}",
            model.err()
        );
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_aya_projector() {
        let device = Device::Cpu;
        let cfg = AyaVisionConfig {
            vision_config: VisionEncoderConfig::siglip_so400m_384(),
            downsample_factor: 2,
            alignment_intermediate_size: 128,
            text_hidden_size: 64,
            adapter_layer_norm_eps: 1e-6,
            use_projector_layernorm: true,
        };
        // Override vision hidden_size to 64 for testing
        let cfg = AyaVisionConfig {
            vision_config: VisionEncoderConfig {
                hidden_size: 64,
                ..cfg.vision_config
            },
            ..cfg
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = AyaVisionProjector::new(&cfg, vb).unwrap();

        // 4x4=16 patches, 64-dim → pixel shuffle(f=2) → 2x2=4 patches, 256-dim
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 64), &device).unwrap();
        let output = proj.forward(&x).unwrap();
        // After projector: 4 patches, 64-dim (text_hidden_size)
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_aya_pixel_shuffle() {
        let cfg = AyaVisionConfig {
            vision_config: VisionEncoderConfig {
                hidden_size: 64,
                ..VisionEncoderConfig::siglip_so400m_384()
            },
            downsample_factor: 2,
            alignment_intermediate_size: 128,
            text_hidden_size: 64,
            adapter_layer_norm_eps: 1e-6,
            use_projector_layernorm: true,
        };
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let proj = AyaVisionProjector::new(&cfg, vb).unwrap();

        // 4x4=16 patches, 32-dim → f=2 → 2x2=4 patches, 128-dim
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 32), &Device::Cpu).unwrap();
        let shuffled = proj.pixel_shuffle(&x).unwrap();
        assert_eq!(shuffled.dims(), &[1, 4, 128]);
    }

    #[test]
    fn test_aya_swiglu() {
        let device = Device::Cpu;
        let cfg = AyaVisionConfig {
            vision_config: VisionEncoderConfig {
                hidden_size: 64,
                ..VisionEncoderConfig::siglip_so400m_384()
            },
            downsample_factor: 2,
            alignment_intermediate_size: 128,
            text_hidden_size: 64,
            adapter_layer_norm_eps: 1e-6,
            use_projector_layernorm: true,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = AyaVisionProjector::new(&cfg, vb).unwrap();

        // Forward should work end to end with SwiGLU
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 64), &device).unwrap();
        let output = proj.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AyaVisionForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "AyaVision should construct: {:?}",
            model.err()
        );
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AyaVisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
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
        let model = AyaVisionForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
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
