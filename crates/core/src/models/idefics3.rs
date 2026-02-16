//! Idefics3 vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: SigLIP (reused from multimodal/vision.rs)
//! - Connector: Pixel shuffle (spatial downsampling) + linear projection
//! - Language model: LLaMA decoder layers (reused from llama.rs)
//!
//! SmolVLM is a thin alias of Idefics3 with different default token IDs.
//!
//! Reference: reference/vllm/vllm/model_executor/models/idefics3.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::llama::{LlamaDecoderLayer, TpContext};
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Idefics3Config {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    image_token_id: u32,
    scale_factor: usize,
}

impl Idefics3Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
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

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(32001) as u32;

        let scale_factor = cfg
            .extra
            .get("scale_factor")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        Self {
            model_config: cfg.clone(),
            vision_config,
            image_token_id,
            scale_factor,
        }
    }
}

// ─── Pixel Shuffle Connector ────────────────────────────────────────────────

/// Idefics3 connector: pixel shuffle + linear projection.
///
/// Pixel shuffle reduces spatial dimensions by `scale_factor` in each direction,
/// combining `scale_factor^2` adjacent patches into one with concatenated features.
/// Then a single linear layer projects the combined features to LLM hidden size.
struct Idefics3Connector {
    proj: Linear,
    scale_factor: usize,
}

impl Idefics3Connector {
    fn new(
        vision_hidden_size: usize,
        llm_hidden_size: usize,
        scale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_dim = vision_hidden_size * scale_factor * scale_factor;
        let proj = linear_no_bias(input_dim, llm_hidden_size, vb.pp("modality_projection"))?;
        Ok(Self { proj, scale_factor })
    }

    /// Apply pixel shuffle and project.
    ///
    /// Input: (batch, seq_len, vision_hidden_size)
    /// Output: (batch, seq_len / scale_factor^2, llm_hidden_size)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let sf = self.scale_factor;

        // Determine spatial dimensions from sequence length
        let h = (seq_len as f64).sqrt() as usize;
        let w = seq_len / h;

        // Reshape: (batch, h, w, hidden)
        let x = x.reshape((batch, h, w, hidden))?;

        // Pixel shuffle: rearrange (h, w) → (h/sf, sf, w/sf, sf) → (h/sf, w/sf, sf*sf*hidden)
        let h_out = h / sf;
        let w_out = w / sf;

        // (batch, h_out, sf, w_out, sf, hidden)
        let x = x.reshape((batch, h_out, sf, w_out, sf, hidden))?;
        // (batch, h_out, w_out, sf, sf, hidden)
        let x = x.permute((0, 1, 3, 2, 4, 5))?;
        // (batch, h_out * w_out, sf * sf * hidden)
        let x = x.reshape((batch, h_out * w_out, sf * sf * hidden))?;

        self.proj.forward(&x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Idefics3 vision-language model.
///
/// Uses SigLIP vision encoder + pixel shuffle connector + LLaMA backbone.
/// SmolVLM uses the same architecture.
pub struct Idefics3ForConditionalGeneration {
    // Vision
    #[allow(dead_code)]
    vision_encoder: VisionEncoder,
    connector: Idefics3Connector,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    // Config
    #[allow(dead_code)]
    config: Idefics3Config,
    device: Device,
    dtype: DType,
}

impl Idefics3ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = Idefics3Config::from_model_config(cfg);
        let world_size = pg.world_size();

        // Vision encoder (SigLIP)
        let vision_encoder =
            VisionEncoder::new(&config.vision_config, vb.pp("model").pp("vision_model"))?;

        // Connector (pixel shuffle + linear)
        let connector = Idefics3Connector::new(
            config.vision_config.hidden_size,
            cfg.hidden_size,
            config.scale_factor,
            vb.pp("model").pp("connector"),
        )?;

        // LLM (LLaMA backbone)
        let vb_m = vb.pp("model").pp("text_model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vision_encoder,
            connector,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Merge text embeddings with vision embeddings at image token positions.
    fn merge_multimodal_embeddings(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Project vision features through connector
            let vision_emb = processed_image.embedding.unsqueeze(0)?;
            let projected = self.connector.forward(&vision_emb)?;
            let projected = projected.squeeze(0)?;
            let img_emb: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

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

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let text_embeddings = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(&text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
        } else {
            text_embeddings
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }
}

impl crate::engine::ModelForward for Idefics3ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_inner(
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
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
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
        self.forward_inner(
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

// ─── SmolVLM (thin wrapper) ────────────────────────────────────────────────

/// SmolVLM model — thin alias of Idefics3 with SmolLM backbone.
///
/// SmolVLM uses the same architecture as Idefics3 but with different
/// default token IDs and a smaller language model.
pub type SmolVLMForConditionalGeneration = Idefics3ForConditionalGeneration;

// ─── Tests ──────────────────────────────────────────────────────────────────

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
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 56,
                "patch_size": 14,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
                "model_type": "siglip"
            }),
        );
        extra.insert("image_token_id".to_string(), serde_json::json!(32001));
        extra.insert("scale_factor".to_string(), serde_json::json!(2));

        ModelConfig {
            architectures: vec!["Idefics3ForConditionalGeneration".to_string()],
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
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
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
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Idefics3ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_pixel_shuffle_connector() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let connector = Idefics3Connector::new(64, 128, 2, vb).unwrap();

        // 4x4 patch grid (16 patches) → 2x2 (4 patches) after scale_factor=2
        let input = Tensor::zeros((1, 16, 64), DType::F32, &device).unwrap();
        let output = connector.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 4, 128]);
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Idefics3ForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Idefics3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
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
    fn test_multimodal_forward_text_only() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Idefics3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
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

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Idefics3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_smolvlm_is_alias() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = SmolVLMForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok());
        assert!(model.unwrap().supports_multimodal());
    }
}
