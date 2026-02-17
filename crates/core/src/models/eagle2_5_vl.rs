//! Eagle2.5-VL vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: SigLIP
//! - Projector: Pixel shuffle (Eagle variant) + LayerNorm + Linear + GELU + Linear
//! - Language model: Qwen2 (LLaMA-compatible decoder layers)
//!
//! Reference: reference/vllm/vllm/model_executor/models/eagle2_5_vl.py

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};

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
struct Eagle25VLConfig {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    downsample_ratio: f64,
    select_layer: i32,
    image_token_index: u32,
}

impl Eagle25VLConfig {
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

        let downsample_ratio = cfg
            .extra
            .get("downsample_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let select_layer = cfg
            .extra
            .get("select_layer")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1) as i32;

        let image_token_index = cfg
            .extra
            .get("img_context_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151667) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            downsample_ratio,
            select_layer,
            image_token_index,
        }
    }
}

// ─── Pixel Shuffle (Eagle2.5 variant) ───────────────────────────────────────

/// Eagle2.5 pixel shuffle: two-step spatial downsampling via permutations.
///
/// Input: (n, w, h, c) → Output: (n, w*s, h*s, c/(s^2)) where s = scale_factor.
/// For scale_factor=0.5, this halves spatial dimensions and 4x the channels.
fn pixel_shuffle_eagle(x: &Tensor, scale_factor: f64) -> Result<Tensor> {
    let (n, w, h, c) = x.dims4()?;
    let new_h = (h as f64 * scale_factor) as usize;
    let new_c1 = (c as f64 / scale_factor) as usize;
    let new_w = (w as f64 * scale_factor) as usize;
    let final_c = (c as f64 / (scale_factor * scale_factor)) as usize;

    // Step 1: (n, w, h, c) → (n, w, h*s, c/s)
    let x = x.reshape((n, w, new_h, new_c1))?;
    // Step 2: permute → (n, h*s, w, c/s)
    let x = x.permute((0, 2, 1, 3))?.contiguous()?;
    // Step 3: (n, h*s, w, c/s) → (n, h*s, w*s, c/(s^2))
    let x = x.reshape((n, new_h, new_w, final_c))?;
    // Step 4: permute → (n, w*s, h*s, c/(s^2))
    let x = x.permute((0, 2, 1, 3))?.contiguous()?;

    Ok(x)
}

// ─── MLP Projector ──────────────────────────────────────────────────────────

/// Eagle2.5 MLP projector: LayerNorm → Linear → GELU → Linear.
///
/// Same architecture as InternVL's mlp1 projector.
struct Eagle25VLProjector {
    ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Eagle25VLProjector {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln = layer_norm(input_dim, 1e-6, vb.pp("0"))?;
        let fc1 = candle_nn::linear(input_dim, output_dim, vb.pp("1"))?;
        let fc2 = candle_nn::linear(output_dim, output_dim, vb.pp("3"))?;
        Ok(Self { ln, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Eagle2.5-VL model for conditional generation.
///
/// SigLIP vision encoder + pixel shuffle + MLP projector + Qwen2 LLM backbone.
pub struct Eagle25VLForConditionalGeneration {
    #[allow(dead_code)]
    vision_encoder: VisionEncoder,
    projector: Eagle25VLProjector,
    downsample_ratio: f64,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    // Config
    #[allow(dead_code)]
    config: Eagle25VLConfig,
    device: Device,
    dtype: DType,
}

impl Eagle25VLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = Eagle25VLConfig::from_model_config(cfg);
        let world_size = pg.world_size();

        // Vision encoder (SigLIP)
        let vision_encoder = VisionEncoder::new(&config.vision_config, vb.pp("vision_model"))?;

        // Projector: pixel shuffle expands channels by (1/downsample_ratio)^2
        let inv_ratio = (1.0 / config.downsample_ratio) as usize;
        let proj_input = config.vision_config.hidden_size * inv_ratio * inv_ratio;
        let projector = Eagle25VLProjector::new(proj_input, cfg.hidden_size, vb.pp("mlp1"))?;

        // LLM backbone (Qwen2, structurally identical to LLaMA)
        let vb_lm = vb.pp("language_model").pp("model");

        let embed_tokens = TpEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_lm.pp("embed_tokens"),
            pg,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_lm.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_lm.pp("norm"))?;

        let vb_head = vb.pp("language_model");
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
                vb_lm.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_head.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vision_encoder,
            projector,
            downsample_ratio: config.downsample_ratio,
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

    /// Extract vision features: vision encoder → pixel shuffle → MLP projector.
    #[allow(dead_code)]
    fn extract_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Vision encoder output: (batch, num_patches, vision_hidden)
        let vit_embeds = self.vision_encoder.forward(pixel_values)?;

        // Compute spatial dims from sequence length
        let (batch, seq_len, hidden) = vit_embeds.dims3()?;
        let h = (seq_len as f64).sqrt() as usize;
        let w = seq_len / h;

        // Reshape to spatial: (batch, h, w, hidden)
        let vit_embeds = vit_embeds.reshape((batch, h, w, hidden))?;

        // Pixel shuffle: downsample spatial, expand channels
        let vit_embeds = pixel_shuffle_eagle(&vit_embeds, self.downsample_ratio)?;

        // Flatten back to sequence: (batch, new_seq, expanded_hidden)
        let (b, w2, h2, c2) = vit_embeds.dims4()?;
        let vit_embeds = vit_embeds.reshape((b, w2 * h2, c2))?;

        // Project to LLM hidden size
        self.projector.forward(&vit_embeds)
    }

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
            let vision_emb = processed_image.embedding.unsqueeze(0)?;
            let projected = self.extract_features(&vision_emb)?;
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

impl crate::engine::ModelForward for Eagle25VLForConditionalGeneration {
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
        extra.insert("downsample_ratio".to_string(), serde_json::json!(0.5));
        extra.insert("select_layer".to_string(), serde_json::json!(-1));

        ModelConfig {
            architectures: vec!["Eagle2_5_VLForConditionalGeneration".to_string()],
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
    fn test_pixel_shuffle_eagle() {
        let device = Device::Cpu;
        // (1, 4, 4, 8) with scale_factor=0.5 → (1, 2, 2, 32)
        let input = Tensor::zeros((1, 4, 4, 8), DType::F32, &device).unwrap();
        let output = pixel_shuffle_eagle(&input, 0.5).unwrap();
        assert_eq!(output.dims(), &[1, 2, 2, 32]);
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Eagle25VLForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Eagle25VLForConditionalGeneration::new(&cfg, vb).unwrap();

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
        let model = Eagle25VLForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Eagle25VLForConditionalGeneration::new(&cfg, vb).unwrap();

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
}
