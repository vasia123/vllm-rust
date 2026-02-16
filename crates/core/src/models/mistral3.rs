//! Mistral3 vision-language model implementation.
//!
//! Combines the Pixtral vision encoder with a Mistral language model backbone,
//! adding a multi-modal projector with patch merging for spatial reduction.
//!
//! Architecture:
//! - Vision encoder: Pixtral (reused from pixtral.rs)
//! - Multi-modal projector: RMSNorm → PatchMerger → 2-layer GELU MLP
//! - Language model: Mistral decoder layers (reused from mistral.rs)
//!
//! Reference: reference/vllm/vllm/model_executor/models/mistral3.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;
use crate::multimodal::MultimodalInputs;

use super::mistral::{MistralDecoderLayer, TpContext};
use super::pixtral::{PixtralVisionConfig, PixtralVisionTransformer};
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Mistral3Config {
    model_config: ModelConfig,
    vision_config: PixtralVisionConfig,
    image_token_index: u32,
    spatial_merge_size: usize,
    projector_hidden_act: String,
    multimodal_projector_bias: bool,
}

impl Mistral3Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(PixtralVisionConfig::from_json)
            .unwrap_or_default();

        let image_token_index = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;

        let spatial_merge_size = cfg
            .extra
            .get("spatial_merge_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let projector_hidden_act = cfg
            .extra
            .get("projector_hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string();

        let multimodal_projector_bias = cfg
            .extra
            .get("multimodal_projector_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Self {
            model_config: cfg.clone(),
            vision_config,
            image_token_index,
            spatial_merge_size,
            projector_hidden_act,
            multimodal_projector_bias,
        }
    }
}

// ─── Patch Merger ───────────────────────────────────────────────────────────

/// Learned merging of spatial_merge_size^2 adjacent patches.
///
/// For spatial_merge_size=2, groups 2x2 patches into one, reducing total
/// patch count by 4x. Uses a linear layer to project concatenated features
/// back to vision_hidden_size.
struct Mistral3PatchMerger {
    merging_layer: Linear,
    spatial_merge_size: usize,
}

impl Mistral3PatchMerger {
    fn new(vision_hidden_size: usize, spatial_merge_size: usize, vb: VarBuilder) -> Result<Self> {
        let input_dim = vision_hidden_size * spatial_merge_size * spatial_merge_size;
        let merging_layer = linear_no_bias(input_dim, vision_hidden_size, vb.pp("merging_layer"))?;
        Ok(Self {
            merging_layer,
            spatial_merge_size,
        })
    }

    /// Merge patches for a single image.
    ///
    /// Takes (h_patches * w_patches, hidden_dim) and returns
    /// (h_merged * w_merged, hidden_dim) where h_merged = h_patches / merge_size.
    fn forward(
        &self,
        image_features: &Tensor,
        h_patches: usize,
        w_patches: usize,
    ) -> Result<Tensor> {
        let ms = self.spatial_merge_size;
        let d = image_features.dim(1)?;

        // Reshape to spatial grid: (h, w, d)
        let grid = image_features.reshape((h_patches, w_patches, d))?;
        // Permute to (d, h, w) for unfold-like operation
        let grid = grid.permute((2, 0, 1))?;

        let h_merged = h_patches / ms;
        let w_merged = w_patches / ms;

        // Manual unfold: extract ms×ms windows and concatenate features
        let mut windows = Vec::with_capacity(h_merged * w_merged);
        for i in 0..h_merged {
            for j in 0..w_merged {
                // Extract the ms×ms window from (d, h, w)
                let window = grid.narrow(1, i * ms, ms)?.narrow(2, j * ms, ms)?;
                // Flatten to (d * ms * ms,)
                let flat = window.flatten_all()?;
                windows.push(flat);
            }
        }

        // Stack: (h_merged * w_merged, d * ms * ms)
        let merged = Tensor::stack(&windows, 0)?;
        // Project back: (h_merged * w_merged, d)
        self.merging_layer.forward(&merged)
    }
}

// ─── Multi-Modal Projector ──────────────────────────────────────────────────

/// Mistral3 multi-modal projector: RMSNorm → PatchMerger → 2-layer MLP.
struct Mistral3MultiModalProjector {
    norm: RmsNorm,
    patch_merger: Mistral3PatchMerger,
    linear_1: Linear,
    linear_2: Linear,
}

impl Mistral3MultiModalProjector {
    fn new(
        vision_hidden_size: usize,
        text_hidden_size: usize,
        spatial_merge_size: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = rms_norm(vision_hidden_size, 1e-5, vb.pp("norm"))?;
        let patch_merger = Mistral3PatchMerger::new(
            vision_hidden_size,
            spatial_merge_size,
            vb.pp("patch_merger"),
        )?;

        let linear_1 = if bias {
            linear(vision_hidden_size, text_hidden_size, vb.pp("linear_1"))?
        } else {
            linear_no_bias(vision_hidden_size, text_hidden_size, vb.pp("linear_1"))?
        };
        let linear_2 = if bias {
            linear(text_hidden_size, text_hidden_size, vb.pp("linear_2"))?
        } else {
            linear_no_bias(text_hidden_size, text_hidden_size, vb.pp("linear_2"))?
        };

        Ok(Self {
            norm,
            patch_merger,
            linear_1,
            linear_2,
        })
    }

    /// Project vision features for a single image.
    fn forward(
        &self,
        image_features: &Tensor,
        h_patches: usize,
        w_patches: usize,
    ) -> Result<Tensor> {
        let features = self.norm.forward(image_features)?;
        let features = self.patch_merger.forward(&features, h_patches, w_patches)?;
        let features = self.linear_1.forward(&features)?;
        let features = candle_nn::Activation::Gelu.forward(&features)?;
        self.linear_2.forward(&features)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Mistral3 vision-language model.
///
/// Combines Pixtral vision encoder with Mistral LLM backbone via a multi-modal
/// projector that includes patch merging for spatial reduction.
pub struct Mistral3ForConditionalGeneration {
    // Vision
    vision_tower: PixtralVisionTransformer,
    multi_modal_projector: Mistral3MultiModalProjector,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<MistralDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    // Config
    #[allow(dead_code)]
    config: Mistral3Config,
    device: Device,
    dtype: DType,
}

impl Mistral3ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = Mistral3Config::from_model_config(cfg);
        let world_size = pg.world_size();

        // Vision tower (Pixtral encoder)
        let vision_tower =
            PixtralVisionTransformer::new(&config.vision_config, vb.pp("vision_tower"))?;

        // Multi-modal projector
        let multi_modal_projector = Mistral3MultiModalProjector::new(
            config.vision_config.hidden_size,
            cfg.hidden_size,
            config.spatial_merge_size,
            config.multimodal_projector_bias,
            vb.pp("multi_modal_projector"),
        )?;

        // Language model (Mistral backbone)
        let vb_m = vb.pp("language_model").pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MistralDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
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
                vb.pp("language_model").pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vision_tower,
            multi_modal_projector,
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

    /// Encode images through vision tower and projector.
    #[allow(dead_code)]
    fn encode_images(&self, images: &[Tensor]) -> Result<Vec<Tensor>> {
        let image_features = self.vision_tower.forward(images)?;

        // Record patch grid sizes before flattening
        let patch_size = self.config.vision_config.patch_size;
        let image_sizes: Vec<(usize, usize)> = images
            .iter()
            .map(|img| {
                let dims = img.dims();
                let h = dims[dims.len() - 2];
                let w = dims[dims.len() - 1];
                (h / patch_size, w / patch_size)
            })
            .collect();

        let mut projected = Vec::with_capacity(image_features.len());
        for (feat, &(h_patches, w_patches)) in image_features.iter().zip(image_sizes.iter()) {
            let proj = self
                .multi_modal_projector
                .forward(feat, h_patches, w_patches)?;
            projected.push(proj);
        }
        Ok(projected)
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
            // Image embeddings are already projected to LLM space
            let img_emb: Vec<Vec<f32>> =
                processed_image.embedding.to_dtype(DType::F32)?.to_vec2()?;

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

    pub fn forward_inner(
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

impl crate::engine::ModelForward for Mistral3ForConditionalGeneration {
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
                "num_channels": 3,
                "image_size": 64,
                "patch_size": 16,
                "rope_theta": 10000.0,
                "adapter_bias": true
            }),
        );
        extra.insert("image_token_index".to_string(), serde_json::json!(10));
        extra.insert("spatial_merge_size".to_string(), serde_json::json!(2));
        extra.insert(
            "projector_hidden_act".to_string(),
            serde_json::json!("gelu"),
        );
        extra.insert(
            "multimodal_projector_bias".to_string(),
            serde_json::json!(true),
        );

        ModelConfig {
            architectures: vec!["Mistral3ForConditionalGeneration".to_string()],
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
            sliding_window: Some(256),
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
        let model = Mistral3ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mistral3ForConditionalGeneration::new(&cfg, vb).unwrap();

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
        let model = Mistral3ForConditionalGeneration::new(&cfg, vb).unwrap();

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
        let model = Mistral3ForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_patch_merger() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let merger = Mistral3PatchMerger::new(64, 2, vb).unwrap();

        // 4x4 patch grid → 2x2 after merging
        let features = Tensor::zeros((16, 64), DType::F32, &device).unwrap();
        let merged = merger.forward(&features, 4, 4).unwrap();

        assert_eq!(merged.dims(), &[4, 64]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mistral3ForConditionalGeneration::new(&cfg, vb).unwrap();

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
