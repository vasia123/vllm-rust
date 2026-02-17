//! NVLM-D (NVIDIA Vision Language Model - Dense) implementation.
//!
//! NVLM-D is an InternVL2 variant from NVIDIA with two key differences:
//! 1. The mlp1 projector routes through `llm_intermediate_size` (not `llm_hidden_size`)
//!    with no-bias linear layers for better accuracy
//! 2. The InternViT encoder uses 7 dummy heads for tensor parallelism divisibility
//!
//! Everything else (vision encoder, pixel shuffle, LLM backbone) is identical to InternVL2.
//!
//! Reference: NVLM-D (https://huggingface.co/nvidia/NVLM-D-72B)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::internlm2::InternLM2ForCausalLM;
use super::internvl::{InternVLConfig, InternVisionModel};

// ─── NVLM-D Projector ──────────────────────────────────────────────────────

/// NVLM-D projector: LayerNorm + Linear(in → intermediate, no bias) + GELU + Linear(intermediate → out, no bias).
///
/// Unlike InternVL2's projector which uses `llm_hidden_size` as intermediate dim with bias,
/// NVLM-D routes through `llm_intermediate_size` (typically much larger) without bias.
#[allow(dead_code)]
struct NVLMProjector {
    ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl NVLMProjector {
    fn new(
        input_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln = layer_norm(input_dim, 1e-6, vb.pp("0"))?;
        let fc1 = candle_nn::linear_no_bias(input_dim, intermediate_dim, vb.pp("1"))?;
        let fc2 = candle_nn::linear_no_bias(intermediate_dim, output_dim, vb.pp("3"))?;
        Ok(Self { ln, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── NVLM-D Config ─────────────────────────────────────────────────────────

/// NVLM-D configuration extending InternVL config with intermediate size.
#[derive(Debug, Clone)]
pub struct NVLMDConfig {
    pub internvl_config: InternVLConfig,
    /// LLM intermediate size for the projector bottleneck.
    pub llm_intermediate_size: usize,
}

impl NVLMDConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let internvl_config = InternVLConfig::from_model_config(cfg);
        let llm_intermediate_size = cfg.intermediate_size;
        Self {
            internvl_config,
            llm_intermediate_size,
        }
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// NVLM-D model for conditional generation.
///
/// Identical to InternVL2 except:
/// - Projector uses `llm_intermediate_size` bottleneck with no-bias linears
/// - Vision encoder uses 7 dummy heads (handled at weight loading, not architecture)
pub struct NVLMDModel {
    #[allow(dead_code)]
    vision_model: InternVisionModel,
    #[allow(dead_code)]
    projector: NVLMProjector,
    language_model: InternLM2ForCausalLM,
    #[allow(dead_code)]
    config: NVLMDConfig,
    device: Device,
    dtype: DType,
}

impl NVLMDModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = NVLMDConfig::from_model_config(cfg);
        let depth = config.internvl_config.effective_depth();
        let vision_model = InternVisionModel::new(
            &config.internvl_config.vision_config,
            depth,
            vb.pp("vision_model"),
        )?;

        let proj_input = config.internvl_config.projector_input_dim();
        let projector = NVLMProjector::new(
            proj_input,
            config.llm_intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp1"),
        )?;

        let language_model = InternLM2ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
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

impl crate::engine::ModelForward for NVLMDModel {
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
                "norm_type": "rms_norm"
            }),
        );
        extra.insert("downsample_ratio".to_string(), serde_json::json!(0.5));
        extra.insert("select_layer".to_string(), serde_json::json!(-1));

        ModelConfig {
            architectures: vec!["NVLM_D_Model".to_string()],
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
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_nvlm_config() {
        let cfg = test_model_config();
        let nvlm_cfg = NVLMDConfig::from_model_config(&cfg);
        assert_eq!(nvlm_cfg.llm_intermediate_size, 128);
        assert_eq!(nvlm_cfg.internvl_config.downsample_ratio, 0.5);
    }

    #[test]
    fn test_nvlm_projector() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        // input=256 (vit_hidden*4), intermediate=128, output=64
        let proj = NVLMProjector::new(256, 128, 64, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 256), &device).unwrap();
        let output = proj.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_nvlm_projector_no_bias() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = NVLMProjector::new(256, 512, 64, vb).unwrap();

        // With zero weights and no bias, output should be zero
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 256), &device).unwrap();
        let output = proj.forward(&x).unwrap();
        let sum: f32 = output
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            sum < 1e-6,
            "no-bias projector with zero weights should output ~0"
        );
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NVLMDModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "NVLM-D model should construct: {:?}",
            model.err()
        );
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NVLMDModel::new(&cfg, vb).unwrap();

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
        let model = NVLMDModel::new(&cfg, vb).unwrap();

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

    #[test]
    fn test_pixel_shuffle_for_nvlm() {
        use crate::models::internvl::pixel_shuffle;
        let device = Device::Cpu;
        // NVLM-D uses same pixel shuffle as InternVL2
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 64), &device).unwrap();
        let out = pixel_shuffle(&x, 4, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 4, 256]);
    }
}
