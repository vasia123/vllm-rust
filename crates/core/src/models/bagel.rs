//! Bagel vision-language model implementation (Bytedance).
//!
//! Architecture:
//! - Vision encoder: SigLIP ViT (reused from multimodal/vision.rs)
//! - Connector: BagelVisionMLP (fc1 → gelu_tanh → fc2)
//! - Position embedding: 2D sin-cos (computed, not learned; added after connector)
//! - Language model: Qwen2 (reused from qwen2.rs)
//!
//! Only image understanding (vision-to-text) is supported.
//! Image generation capabilities are not implemented.
//!
//! Reference: reference/vllm/vllm/model_executor/models/bagel.py

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig};

use super::qwen2::Qwen2ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BagelConfig {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    vit_max_num_patch_per_side: usize,
    connector_act: String,
}

impl BagelConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vit_config")
            .map(|v| {
                let hidden_size = v
                    .get("hidden_size")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(1152) as usize;
                let intermediate_size = v
                    .get("intermediate_size")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(4304) as usize;
                let num_attention_heads = v
                    .get("num_attention_heads")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(16) as usize;
                let num_hidden_layers = v
                    .get("num_hidden_layers")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(26) as usize;
                let image_size =
                    v.get("image_size").and_then(|x| x.as_u64()).unwrap_or(384) as usize;
                let patch_size =
                    v.get("patch_size").and_then(|x| x.as_u64()).unwrap_or(14) as usize;
                let num_channels =
                    v.get("num_channels").and_then(|x| x.as_u64()).unwrap_or(3) as usize;
                let layer_norm_eps = v
                    .get("layer_norm_eps")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(1e-6);

                VisionEncoderConfig {
                    encoder_type: crate::multimodal::VisionEncoderType::SigLip,
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    num_hidden_layers,
                    image_size,
                    patch_size,
                    num_channels,
                    layer_norm_eps,
                }
            })
            .unwrap_or_else(VisionEncoderConfig::siglip_so400m_384);

        let vit_max_num_patch_per_side = cfg
            .extra
            .get("vit_max_num_patch_per_side")
            .and_then(|v| v.as_u64())
            .unwrap_or(27) as usize;

        let connector_act = cfg
            .extra
            .get("connector_act")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu_pytorch_tanh")
            .to_string();

        Self {
            model_config: cfg.clone(),
            vision_config,
            vit_max_num_patch_per_side,
            connector_act,
        }
    }
}

// ─── BagelVisionMLP ─────────────────────────────────────────────────────────

/// MLP connector for vision features.
/// fc1 → activation → fc2
struct BagelVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl BagelVisionMLP {
    fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2 = linear(hidden_features, out_features, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = candle_core::Module::forward(&self.fc1, x)?;
        // gelu_pytorch_tanh approximation
        let x = x.gelu_erf()?;
        let x = candle_core::Module::forward(&self.fc2, &x)?;
        Ok(x)
    }
}

// ─── 2D Sin-Cos Position Embedding ──────────────────────────────────────────

/// Generate 1D sin-cos position embeddings.
///
/// Returns a tensor of shape [num_positions, embed_dim].
fn sincos_1d(embed_dim: usize, num_positions: usize, device: &Device) -> Result<Tensor> {
    assert!(embed_dim.is_multiple_of(2));
    let half_dim = embed_dim / 2;

    // omega = 1 / 10000^(2i/embed_dim)
    let mut omega = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        omega.push(1.0f32 / 10000.0f32.powf(i as f32 / half_dim as f32));
    }
    let omega = Tensor::from_vec(omega, (1, half_dim), device)?;

    // positions [num_positions, 1]
    let positions: Vec<f32> = (0..num_positions).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (num_positions, 1), device)?;

    // outer product: [num_positions, half_dim]
    let angles = positions.matmul(&omega)?;

    let sin_emb = angles.sin()?;
    let cos_emb = angles.cos()?;

    // [num_positions, embed_dim]
    Tensor::cat(&[sin_emb, cos_emb], 1)
}

/// Generate 2D sin-cos position embeddings for a grid.
///
/// Returns a tensor of shape [grid_size * grid_size, embed_dim].
fn sincos_2d(embed_dim: usize, grid_size: usize, device: &Device) -> Result<Tensor> {
    assert!(embed_dim.is_multiple_of(2));
    let half_dim = embed_dim / 2;

    // emb_h for height dimension
    let emb_h = sincos_1d(half_dim, grid_size, device)?; // [grid_size, half_dim]
                                                         // emb_w for width dimension
    let emb_w = sincos_1d(half_dim, grid_size, device)?; // [grid_size, half_dim]

    // Expand to 2D grid: emb_h[h] concat emb_w[w] for each (h, w)
    // emb_h: [grid_size, 1, half_dim] → broadcast to [grid_size, grid_size, half_dim]
    let emb_h = emb_h
        .unsqueeze(1)?
        .broadcast_as((grid_size, grid_size, half_dim))?;
    // emb_w: [1, grid_size, half_dim] → broadcast to [grid_size, grid_size, half_dim]
    let emb_w = emb_w
        .unsqueeze(0)?
        .broadcast_as((grid_size, grid_size, half_dim))?;

    // Concatenate: [grid_size, grid_size, embed_dim]
    let emb = Tensor::cat(&[emb_h.contiguous()?, emb_w.contiguous()?], 2)?;

    // Flatten to [grid_size * grid_size, embed_dim]
    emb.reshape((grid_size * grid_size, embed_dim))
}

// ─── Model ──────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct BagelForConditionalGeneration {
    vit_model: VisionEncoder,
    connector: BagelVisionMLP,
    pos_embed: Tensor,
    language_model: Qwen2ForCausalLM,
    config: BagelConfig,
    device: Device,
    dtype: DType,
}

impl BagelForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let bagel_cfg = BagelConfig::from_model_config(cfg);

        // Vision encoder (SigLIP)
        let vit_model = VisionEncoder::new(&bagel_cfg.vision_config, vb.pp("vit_model"))?;

        // Connector MLP: vit_hidden_size → llm_hidden_size → llm_hidden_size
        let vit_hidden_size = bagel_cfg.vision_config.hidden_size;
        let llm_hidden_size = cfg.hidden_size;
        let connector = BagelVisionMLP::new(
            vit_hidden_size,
            llm_hidden_size,
            llm_hidden_size,
            vb.pp("connector"),
        )?;

        // Pre-compute 2D sin-cos position embeddings
        let pos_embed = sincos_2d(
            llm_hidden_size,
            bagel_cfg.vit_max_num_patch_per_side,
            vb.device(),
        )?
        .to_dtype(vb.dtype())?;

        // Language model (Qwen2)
        let language_model = Qwen2ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vit_model,
            connector,
            pos_embed,
            language_model,
            config: bagel_cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Process image through vision encoder + connector + position embedding.
    ///
    /// Input: pixel_values [batch, channels, height, width]
    /// Output: vision_embeds [batch, num_patches, llm_hidden_size]
    #[allow(dead_code)]
    fn process_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // SigLIP forward: [batch, C, H, W] → [batch, num_patches, vit_hidden_size]
        let vision_features = self.vit_model.forward(pixel_values)?;

        // MLP connector: [batch, num_patches, vit_hidden_size] → [batch, num_patches, llm_hidden_size]
        let vision_embeds = self.connector.forward(&vision_features)?;

        // Add 2D position embeddings
        let (batch_size, num_patches, hidden_size) = vision_embeds.dims3()?;
        let patch_size = self.config.vision_config.patch_size;
        let image_size = self.config.vision_config.image_size;
        let num_patches_per_side = image_size / patch_size;
        let max_side = self.config.vit_max_num_patch_per_side;

        // Build position IDs mapping actual grid positions to the full position embedding table
        // pos_id = h * max_side + w
        let mut position_ids = Vec::with_capacity(num_patches);
        for h in 0..num_patches_per_side {
            for w in 0..num_patches_per_side {
                position_ids.push((h * max_side + w) as u32);
            }
        }

        // Index into the position embedding table
        let pos_ids = Tensor::from_vec(position_ids, (num_patches,), &self.device)?;
        let pos_embeds = self.pos_embed.index_select(&pos_ids, 0)?; // [num_patches, hidden_size]
        let pos_embeds = pos_embeds
            .unsqueeze(0)?
            .broadcast_as((batch_size, num_patches, hidden_size))?
            .contiguous()?;

        vision_embeds.broadcast_add(&pos_embeds)
    }

    /// Merge text and image embeddings.
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

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // For Bagel, the embedding is raw patch features [num_patches, vit_hidden_size].
            // Project through connector + add position embeddings.
            let img_emb = processed_image.embedding.unsqueeze(0)?;
            let projected = self.connector.forward(&img_emb)?;

            // Add position embeddings
            let (_, num_patches, hidden) = projected.dims3()?;
            let patch_size = self.config.vision_config.patch_size;
            let image_size = self.config.vision_config.image_size;
            let num_patches_per_side = image_size / patch_size;
            let max_side = self.config.vit_max_num_patch_per_side;

            let mut pos_ids_vec = Vec::with_capacity(num_patches);
            for h in 0..num_patches_per_side {
                for w in 0..num_patches_per_side {
                    pos_ids_vec.push((h * max_side + w) as u32);
                }
            }
            let pos_ids = Tensor::from_vec(pos_ids_vec, (num_patches,), &self.device)?;
            let pos_embeds = self.pos_embed.index_select(&pos_ids, 0)?;
            let pos_embeds = pos_embeds
                .unsqueeze(0)?
                .broadcast_as((1, num_patches, hidden))?
                .contiguous()?;

            let projected = projected.broadcast_add(&pos_embeds)?;
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

impl crate::engine::ModelForward for BagelForConditionalGeneration {
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
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use crate::multimodal::ProcessedImage;

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        // Vision config (small for testing)
        let vit_config = serde_json::json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "image_size": 28,
            "patch_size": 14,
            "num_channels": 3,
            "layer_norm_eps": 1e-6,
        });
        extra.insert("vit_config".to_string(), vit_config);
        extra.insert(
            "vit_max_num_patch_per_side".to_string(),
            serde_json::json!(4),
        );
        extra.insert(
            "connector_act".to_string(),
            serde_json::json!("gelu_pytorch_tanh"),
        );

        ModelConfig {
            architectures: vec!["BagelForConditionalGeneration".to_string()],
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
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: None,
            extra,
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

    #[test]
    fn test_bagel_config_extraction() {
        let cfg = test_model_config();
        let bagel_cfg = BagelConfig::from_model_config(&cfg);

        assert_eq!(bagel_cfg.vision_config.hidden_size, 32);
        assert_eq!(bagel_cfg.vision_config.patch_size, 14);
        assert_eq!(bagel_cfg.vision_config.image_size, 28);
        assert_eq!(bagel_cfg.vit_max_num_patch_per_side, 4);
        assert_eq!(bagel_cfg.connector_act, "gelu_pytorch_tanh");
    }

    #[test]
    fn test_sincos_1d() {
        let device = Device::Cpu;
        let emb = sincos_1d(8, 4, &device).unwrap();
        assert_eq!(emb.dims(), &[4, 8]);
    }

    #[test]
    fn test_sincos_2d() {
        let device = Device::Cpu;
        let emb = sincos_2d(16, 3, &device).unwrap();
        assert_eq!(emb.dims(), &[9, 16]); // 3*3=9 positions, 16 dims
    }

    #[test]
    fn test_sincos_2d_values() {
        let device = Device::Cpu;
        let emb = sincos_2d(4, 2, &device).unwrap();
        assert_eq!(emb.dims(), &[4, 4]); // 2*2=4 positions

        // Position (0,0) should be all zeros (sin(0)=0, cos(0)=1)
        let row0 = emb.get(0).unwrap().to_vec1::<f32>().unwrap();
        assert!((row0[0] - 0.0).abs() < 1e-5); // sin(0)
        assert!((row0[1] - 1.0).abs() < 1e-5); // cos(0)
    }

    #[test]
    fn test_bagel_model_construction() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BagelForConditionalGeneration::new(&cfg, vb).expect("should build Bagel model");

        assert!(model.supports_multimodal());
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_bagel_vision_mlp() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = BagelVisionMLP::new(32, 64, 64, vb).unwrap();

        let input = Tensor::zeros((2, 4, 32), DType::F32, &device).unwrap();
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 4, 64]);
    }

    #[test]
    fn test_bagel_text_forward() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BagelForConditionalGeneration::new(&cfg, vb).expect("should build Bagel model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("should run text-only forward pass");
        assert_eq!(logits.dims(), &[1, 4, 256]);
    }

    #[test]
    fn test_bagel_multimodal_forward() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BagelForConditionalGeneration::new(&cfg, vb).expect("should build Bagel model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..8).collect();

        let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();

        // image_size=28, patch_size=14 → 2x2=4 patches
        // Each patch embedding: [4, 32] (vit_hidden_size=32)
        let patches = Tensor::zeros((4, 32), DType::F32, &device).unwrap();
        let image = ProcessedImage::new(patches, 4);
        let mm_inputs = MultimodalInputs::with_images(vec![0u32; 8], vec![(0, image)]);

        let logits = model
            .forward_multimodal(
                &input_ids,
                Some(&mm_inputs),
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("should run multimodal forward pass");
        assert_eq!(logits.dims(), &[1, 8, 256]);
    }

    #[test]
    fn test_bagel_no_image_fallback() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BagelForConditionalGeneration::new(&cfg, vb).expect("should build Bagel model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let block_table = BlockTable::from_block_ids(vec![0], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        let logits = model
            .forward_multimodal(
                &input_ids,
                None,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("should run text-only fallback");
        assert_eq!(logits.dims(), &[1, 4, 256]);
    }

    #[test]
    fn test_bagel_pos_embed_shape() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = BagelForConditionalGeneration::new(&cfg, vb).expect("should build Bagel model");

        // vit_max_num_patch_per_side=4, hidden_size=64
        // pos_embed should be [4*4, 64] = [16, 64]
        assert_eq!(model.pos_embed.dims(), &[16, 64]);
    }
}
