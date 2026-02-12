//! Phi-3 Vision (Phi3VForCausalLM) vision-language model implementation.
//!
//! Phi-3 Vision combines a CLIP ViT-L/14 @ 336px vision encoder with the Phi3
//! language model backbone using an HD image embedding module that merges 2x2
//! patches and projects them via a 2-layer MLP with GELU.
//!
//! Key architectural features:
//! - CLIP ViT-L/14 @ 336px (first 23 of 24 layers, layer_idx=-2)
//! - HD image embedding: 2x2 patch merge (576 patches -> 144 merged, each 4096-dim)
//! - Learnable glb_GN / sub_GN newline separators
//! - 2-layer MLP projection: [image_dim_out*4] -> [hidden_size] -> GELU -> [hidden_size]
//! - HD transform order "sub_glb": [sub_features, glb_GN, global_features]
//! - type_feature="patch" means skip CLS token (features[:, 1:])
//!
//! Reference: microsoft/Phi-3-vision-128k-instruct

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::phi3::Phi3ForCausalLM;

// ─── Constants ───────────────────────────────────────────────────────────────

/// The special token ID used as image placeholder in Phi-3 Vision prompts.
const IMAGE_TOKEN_ID: u32 = 32044;

/// CLIP ViT-L/14 @ 336px output dimension per patch.
const DEFAULT_IMAGE_DIM_OUT: usize = 1024;

/// Default number of CLIP layers to use (layer_idx=-2 means 24-2+1=23).
const DEFAULT_NUM_CLIP_LAYERS: usize = 23;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Phi-3 Vision configuration parsed from `ModelConfig.extra`.
#[derive(Debug, Clone)]
pub struct Phi3VConfig {
    pub model_config: ModelConfig,
    pub image_dim_out: usize,
    pub num_clip_layers: usize,
    pub image_token_id: u32,
    pub hidden_size: usize,
}

impl Phi3VConfig {
    /// Parse Phi-3 Vision config from `ModelConfig.extra["img_processor"]`.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let hidden_size = cfg
            .extra
            .get("n_embd")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let (image_dim_out, num_clip_layers) =
            if let Some(img_proc) = cfg.extra.get("img_processor") {
                let dim_out = img_proc
                    .get("image_dim_out")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_IMAGE_DIM_OUT as u64) as usize;

                let layer_idx = img_proc
                    .get("layer_idx")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-2);

                // CLIP has 24 layers. layer_idx=-2 means use first 23 layers.
                let num_layers = if layer_idx < 0 {
                    (24 + layer_idx + 1) as usize
                } else {
                    (layer_idx + 1) as usize
                };

                (dim_out, num_layers)
            } else {
                (DEFAULT_IMAGE_DIM_OUT, DEFAULT_NUM_CLIP_LAYERS)
            };

        Self {
            model_config: cfg.clone(),
            image_dim_out,
            num_clip_layers,
            image_token_id: IMAGE_TOKEN_ID,
            hidden_size,
        }
    }

    /// Dimension of merged 2x2 patches (image_dim_out * 4).
    pub fn merged_dim(&self) -> usize {
        self.image_dim_out * 4
    }
}

// ─── HD Image Embedding ──────────────────────────────────────────────────────

/// Reshape patches into 2x2 groups and merge their channels.
///
/// Input: `[N, H*W, C]` where H=W=sqrt(L) (e.g. 24x24=576 patches, C=1024)
/// For `h_crop x w_crop` arrangement:
///   1. Reshape to `[N, H, W, C]`
///   2. Group 2x2: `[N, H/2, 2, W/2, 2, C]`
///   3. Permute to `[N, H/2, W/2, 2, 2, C]`
///   4. Merge: `[N, H/2 * W/2, 4*C]` = `[N, 144, 4096]` for CLIP
///   5. Arrange crops: `[num_images, h_crop*H/2, w_crop*W/2, 4*C]`
///
/// Returns `[num_images][h_crop*(H/2)][w_crop*(W/2)][4*C]` as a 4D Vec.
fn reshape_hd_patches_2x2merge(
    image_features: &[Vec<Vec<f32>>],
    h_crop: usize,
    w_crop: usize,
    image_dim_out: usize,
) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
    let num_crops = h_crop * w_crop;
    let total = image_features.len();
    if total == 0 || !total.is_multiple_of(num_crops) {
        return Err(candle_core::Error::Msg(format!(
            "reshape_hd_patches_2x2merge: total patches ({total}) not divisible by num_crops ({num_crops})"
        )));
    }
    let num_images = total / num_crops;
    let l = image_features[0].len(); // patches per crop (e.g. 576)
    let c = image_dim_out;
    let h = (l as f64).sqrt() as usize; // e.g. 24
    if h * h != l {
        return Err(candle_core::Error::Msg(format!(
            "reshape_hd_patches_2x2merge: L={l} is not a perfect square"
        )));
    }
    let h_half = h / 2;
    let merged_c = 4 * c;
    let out_h = h_crop * h_half;
    let out_w = w_crop * h_half;

    let mut result = Vec::with_capacity(num_images);
    for img_idx in 0..num_images {
        // Output: [out_h, out_w, merged_c]
        let mut img_out = vec![vec![vec![0.0f32; merged_c]; out_w]; out_h];

        for crop_h in 0..h_crop {
            for crop_w in 0..w_crop {
                let crop_idx = img_idx * num_crops + crop_h * w_crop + crop_w;
                let crop_features = &image_features[crop_idx];

                for r2 in 0..h_half {
                    for c2 in 0..h_half {
                        // 2x2 group at (r2*2, c2*2), (r2*2, c2*2+1), (r2*2+1, c2*2), (r2*2+1, c2*2+1)
                        let row_out = crop_h * h_half + r2;
                        let col_out = crop_w * h_half + c2;

                        let idx00 = (r2 * 2) * h + (c2 * 2);
                        let idx01 = (r2 * 2) * h + (c2 * 2 + 1);
                        let idx10 = (r2 * 2 + 1) * h + (c2 * 2);
                        let idx11 = (r2 * 2 + 1) * h + (c2 * 2 + 1);

                        let out = &mut img_out[row_out][col_out];
                        out[..c].copy_from_slice(&crop_features[idx00]);
                        out[c..2 * c].copy_from_slice(&crop_features[idx01]);
                        out[2 * c..3 * c].copy_from_slice(&crop_features[idx10]);
                        out[3 * c..4 * c].copy_from_slice(&crop_features[idx11]);
                    }
                }
            }
        }

        result.push(img_out);
    }

    Ok(result)
}

/// Add sub_GN newline embedding to each row of HD features for a single image.
///
/// Input: `[h][w][dim]` (a single image's 3D spatial features)
/// Output: `[h*(w+1)][dim]` (flattened with newline appended to each row)
///
/// For each row, appends the sub_GN embedding, then flattens rows.
fn add_image_newline(image_features_hd: &[Vec<Vec<f32>>], sub_gn: &[f32]) -> Vec<Vec<f32>> {
    if image_features_hd.is_empty() {
        return Vec::new();
    }
    let h = image_features_hd.len();
    let w = image_features_hd[0].len();
    let mut flat = Vec::with_capacity(h * (w + 1));

    for row in image_features_hd {
        for col_vec in row {
            flat.push(col_vec.clone());
        }
        flat.push(sub_gn.to_vec());
    }

    flat
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Phi-3 Vision model for conditional generation.
///
/// Wraps CLIP ViT-L/14 vision encoder + HD image embedding + Phi3 LLM.
pub struct Phi3VForCausalLM {
    #[allow(dead_code)]
    vision_encoder: VisionEncoder,
    #[allow(dead_code)]
    glb_gn: Tensor,
    #[allow(dead_code)]
    sub_gn: Tensor,
    #[allow(dead_code)]
    img_projection: Vec<Linear>,
    language_model: Phi3ForCausalLM,
    image_token_id: u32,
    image_dim_out: usize,
    device: Device,
    dtype: DType,
}

impl Phi3VForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let phi3v_cfg = Phi3VConfig::from_model_config(cfg);
        Self::new_with_config(&phi3v_cfg, vb)
    }

    pub fn new_with_config(cfg: &Phi3VConfig, vb: VarBuilder) -> Result<Self> {
        let merged_dim = cfg.merged_dim();

        // CLIP ViT-L/14 @ 336px with first `num_clip_layers` layers
        let clip_config = VisionEncoderConfig {
            encoder_type: VisionEncoderType::Clip,
            hidden_size: cfg.image_dim_out,
            intermediate_size: 4096,
            num_attention_heads: 16,
            num_hidden_layers: cfg.num_clip_layers,
            image_size: 336,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-5,
        };
        let vision_encoder = VisionEncoder::new(
            &clip_config,
            vb.pp("model.vision_embed_tokens.img_processor"),
        )?;

        // Learnable newline separators
        let glb_gn = vb.get((1, 1, merged_dim), "model.vision_embed_tokens.glb_GN")?;
        let sub_gn = vb.get((1, 1, 1, merged_dim), "model.vision_embed_tokens.sub_GN")?;

        // 2-layer MLP: Linear(merged_dim, hidden_size) -> GELU -> Linear(hidden_size, hidden_size)
        let proj_vb = vb.pp("model.vision_embed_tokens.img_projection");
        let fc1 = candle_nn::linear(merged_dim, cfg.hidden_size, proj_vb.pp("0"))?;
        let fc2 = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, proj_vb.pp("2"))?;
        let img_projection = vec![fc1, fc2];

        let language_model = Phi3ForCausalLM::new(&cfg.model_config, vb.clone())?;

        Ok(Self {
            vision_encoder,
            glb_gn,
            sub_gn,
            img_projection,
            language_model,
            image_token_id: cfg.image_token_id,
            image_dim_out: cfg.image_dim_out,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(cfg, vb)
    }

    /// Run the 2-layer MLP projection: fc1 -> GELU -> fc2.
    #[allow(dead_code)]
    fn project(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.img_projection[0].forward(x)?;
        h = h.gelu_erf()?;
        self.img_projection[1].forward(&h)
    }

    /// Merge pre-encoded image embeddings with text embeddings at image_token_id positions.
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

    /// Get the image token ID used as placeholder.
    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    /// Get the image dimension output from CLIP.
    pub fn image_dim_out(&self) -> usize {
        self.image_dim_out
    }
}

impl crate::engine::ModelForward for Phi3VForCausalLM {
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

// ─── Standalone HD transform functions (for use outside the model) ──────────

/// Perform the full HD feature transform on CLIP patch features.
///
/// This is the Rust equivalent of `Phi3HDImageEmbedding.hd_feature_transform`
/// from the reference Python implementation.
///
/// # Arguments
/// * `global_features` - CLIP patch features for the global view `[576, C]`
/// * `sub_features` - CLIP patch features for sub-image crops `[N*576, C]`
/// * `h_crop` - Number of vertical crops
/// * `w_crop` - Number of horizontal crops
/// * `image_dim_out` - CLIP hidden dimension (e.g. 1024)
/// * `glb_gn` - Global newline embedding `[dim*4]`
/// * `sub_gn` - Sub-image newline embedding `[dim*4]`
///
/// # Returns
/// Flattened HD features `[num_tokens, dim*4]` ready for MLP projection.
pub fn hd_feature_transform(
    global_features: &[Vec<f32>],
    sub_features: &[Vec<f32>],
    h_crop: usize,
    w_crop: usize,
    image_dim_out: usize,
    glb_gn: &[f32],
    sub_gn: &[f32],
) -> Result<Vec<Vec<f32>>> {
    // Global: 1x1 crop arrangement
    // reshape returns [num_images][out_h][out_w][merged_c], we take image 0
    let global_hd = reshape_hd_patches_2x2merge(&[global_features.to_vec()], 1, 1, image_dim_out)?;
    // global_hd[0] is [out_h][out_w][merged_c] -> add_image_newline flattens to [tokens][dim]
    let global_hd_newline = add_image_newline(&global_hd[0], sub_gn);

    // Sub-images: split sub_features into per-crop chunks
    let patches_per_crop = global_features.len();
    let sub_crops: Vec<Vec<Vec<f32>>> = sub_features
        .chunks(patches_per_crop)
        .map(|chunk| chunk.to_vec())
        .collect();
    let sub_hd = reshape_hd_patches_2x2merge(&sub_crops, h_crop, w_crop, image_dim_out)?;
    // sub_hd[0] is [out_h][out_w][merged_c]
    let sub_hd_newline = add_image_newline(&sub_hd[0], sub_gn);

    // Combine: [sub_features_hd_newline, glb_GN, global_features_hd_newline]
    let mut combined = Vec::with_capacity(sub_hd_newline.len() + 1 + global_hd_newline.len());
    combined.extend(sub_hd_newline);
    combined.push(glb_gn.to_vec());
    combined.extend(global_hd_newline);

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Phi3VForCausalLM".to_string()],
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
            extra: serde_json::Map::new(),
        }
    }

    fn test_phi3v_config() -> Phi3VConfig {
        // Small test config with tiny dimensions
        let mut cfg = test_model_config();

        // Set img_processor extra fields
        let mut img_proc = serde_json::Map::new();
        img_proc.insert("image_dim_out".into(), serde_json::json!(16));
        img_proc.insert("layer_idx".into(), serde_json::json!(-2));
        img_proc.insert("num_img_tokens".into(), serde_json::json!(144));
        img_proc.insert("type_feature".into(), serde_json::json!("patch"));
        cfg.extra
            .insert("img_processor".into(), serde_json::Value::Object(img_proc));

        Phi3VConfig {
            model_config: cfg,
            image_dim_out: 16,
            num_clip_layers: 2,
            image_token_id: IMAGE_TOKEN_ID,
            hidden_size: 64,
        }
    }

    fn test_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
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
    fn test_config_defaults() {
        let cfg = test_model_config();
        let phi3v_cfg = Phi3VConfig::from_model_config(&cfg);

        assert_eq!(phi3v_cfg.image_dim_out, DEFAULT_IMAGE_DIM_OUT);
        assert_eq!(phi3v_cfg.num_clip_layers, DEFAULT_NUM_CLIP_LAYERS);
        assert_eq!(phi3v_cfg.image_token_id, IMAGE_TOKEN_ID);
        assert_eq!(phi3v_cfg.hidden_size, 64);
    }

    #[test]
    fn test_config_from_extra() {
        let mut cfg = test_model_config();
        let mut img_proc = serde_json::Map::new();
        img_proc.insert("image_dim_out".into(), serde_json::json!(512));
        img_proc.insert("layer_idx".into(), serde_json::json!(-3));
        cfg.extra
            .insert("img_processor".into(), serde_json::Value::Object(img_proc));

        let phi3v_cfg = Phi3VConfig::from_model_config(&cfg);
        assert_eq!(phi3v_cfg.image_dim_out, 512);
        // 24 + (-3) + 1 = 22
        assert_eq!(phi3v_cfg.num_clip_layers, 22);
    }

    #[test]
    fn test_config_positive_layer_idx() {
        let mut cfg = test_model_config();
        let mut img_proc = serde_json::Map::new();
        img_proc.insert("layer_idx".into(), serde_json::json!(10));
        cfg.extra
            .insert("img_processor".into(), serde_json::Value::Object(img_proc));

        let phi3v_cfg = Phi3VConfig::from_model_config(&cfg);
        // layer_idx=10 -> 10 + 1 = 11 layers
        assert_eq!(phi3v_cfg.num_clip_layers, 11);
    }

    #[test]
    fn test_config_merged_dim() {
        let cfg = test_phi3v_config();
        assert_eq!(cfg.merged_dim(), 16 * 4);
    }

    #[test]
    fn test_config_n_embd_override() {
        let mut cfg = test_model_config();
        cfg.extra.insert("n_embd".into(), serde_json::json!(128));

        let phi3v_cfg = Phi3VConfig::from_model_config(&cfg);
        assert_eq!(phi3v_cfg.hidden_size, 128);
    }

    // ── HD Transform Tests ──────────────────────────────────────────────

    #[test]
    fn test_reshape_hd_patches_2x2merge_basic() {
        // 1 image, 1x1 crop, 4 patches (H=2, W=2), C=2
        // After 2x2 merge: 1 merged patch with 4*2=8 channels
        let features = vec![vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]];

        let result = reshape_hd_patches_2x2merge(&features, 1, 1, 2).unwrap();
        assert_eq!(result.len(), 1); // 1 image
        assert_eq!(result[0].len(), 1); // 1 row (H/2=1)
        assert_eq!(result[0][0].len(), 1); // 1 col (W/2=1)
        assert_eq!(result[0][0][0].len(), 8); // 4*C=8

        // Verify merge order: (0,0), (0,1), (1,0), (1,1)
        assert_eq!(result[0][0][0][0..2], [1.0, 2.0]); // patch (0,0)
        assert_eq!(result[0][0][0][2..4], [3.0, 4.0]); // patch (0,1)
        assert_eq!(result[0][0][0][4..6], [5.0, 6.0]); // patch (1,0)
        assert_eq!(result[0][0][0][6..8], [7.0, 8.0]); // patch (1,1)
    }

    #[test]
    fn test_reshape_hd_patches_2x2merge_4x4() {
        // 1 image, 1x1 crop, 16 patches (H=4, W=4), C=1
        // After 2x2 merge: 4 merged patches in 2x2 grid, each with 4*1=4 channels
        let features: Vec<Vec<Vec<f32>>> = vec![(0..16).map(|i| vec![i as f32]).collect()];

        let result = reshape_hd_patches_2x2merge(&features, 1, 1, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2); // H/2=2 rows
        assert_eq!(result[0][0].len(), 2); // W/2=2 cols
        assert_eq!(result[0][0][0].len(), 4); // 4*C=4

        // First merged patch: (0,0), (0,1), (1,0), (1,1) in 4x4 grid
        // = indices 0, 1, 4, 5
        assert_eq!(result[0][0][0], [0.0, 1.0, 4.0, 5.0]);
        // Second merged patch in first row: (0,2), (0,3), (1,2), (1,3)
        // = indices 2, 3, 6, 7
        assert_eq!(result[0][0][1], [2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_reshape_hd_patches_non_square_crop() {
        // 2 crops in 1x2 arrangement (1 row, 2 cols), 4 patches each (H=2, W=2), C=1
        let features: Vec<Vec<Vec<f32>>> = vec![
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec![vec![5.0], vec![6.0], vec![7.0], vec![8.0]],
        ];

        let result = reshape_hd_patches_2x2merge(&features, 1, 2, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1); // h_crop*H/2 = 1*1 = 1
        assert_eq!(result[0][0].len(), 2); // w_crop*W/2 = 2*1 = 2
    }

    #[test]
    fn test_reshape_hd_patches_invalid_square() {
        // 3 patches is not a perfect square
        let features = vec![vec![vec![1.0]; 3]];
        let result = reshape_hd_patches_2x2merge(&features, 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_image_newline() {
        // Single image: 2 rows, 3 cols, dim=2
        let features = vec![
            // row 0
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            // row 1
            vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]],
        ];
        let sub_gn = vec![99.0, 99.0];

        let result = add_image_newline(&features, &sub_gn);

        // Each row gets 3 + 1 (newline) = 4 tokens, total = 2*4 = 8
        assert_eq!(result.len(), 8);

        // Check newline is at position 3 and 7
        assert_eq!(result[3], vec![99.0, 99.0]);
        assert_eq!(result[7], vec![99.0, 99.0]);
    }

    #[test]
    fn test_add_image_newline_single_row() {
        let features = vec![vec![vec![1.0], vec![2.0]]];
        let sub_gn = vec![0.0];

        let result = add_image_newline(&features, &sub_gn);
        // 1 row * (2 cols + 1 newline) = 3
        assert_eq!(result.len(), 3);
    }

    // ── HD Feature Transform Integration ────────────────────────────────

    #[test]
    fn test_hd_feature_transform_1x1() {
        // 4 patches (2x2 grid), dim=1
        let global_features: Vec<Vec<f32>> = vec![vec![1.0]; 4];
        let sub_features: Vec<Vec<f32>> = vec![vec![2.0]; 4];
        let glb_gn = vec![0.0; 4]; // merged_dim = 4*1 = 4
        let sub_gn = vec![0.0; 4];

        let result =
            hd_feature_transform(&global_features, &sub_features, 1, 1, 1, &glb_gn, &sub_gn)
                .unwrap();

        // sub: 1 merged patch in 1x1 grid + 1 newline = 2 tokens
        // glb_GN: 1 token
        // global: 1 merged patch in 1x1 grid + 1 newline = 2 tokens
        // total = 2 + 1 + 2 = 5
        assert_eq!(result.len(), 5);
        // Each token has dim 4 (merged)
        assert_eq!(result[0].len(), 4);
    }

    // ── Model Construction Tests ────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb);
        assert!(
            model.is_ok(),
            "Phi3VForCausalLM should construct: {:?}",
            model.err()
        );
        let model = model.unwrap();
        assert_eq!(model.image_token_id(), IMAGE_TOKEN_ID);
        assert_eq!(model.image_dim_out(), 16);
    }

    #[test]
    fn test_supports_multimodal() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg.model_config, &device);
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
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg.model_config, &device);
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
    fn test_from_model_config() {
        let device = Device::Cpu;
        let mut cfg = test_model_config();

        let mut img_proc = serde_json::Map::new();
        img_proc.insert("image_dim_out".into(), serde_json::json!(16));
        img_proc.insert("layer_idx".into(), serde_json::json!(-2));
        cfg.extra
            .insert("img_processor".into(), serde_json::Value::Object(img_proc));

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::from_model_config(&cfg, vb);
        assert!(
            model.is_ok(),
            "from_model_config should work: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_projection_shape() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        // Input: [batch=1, tokens=3, merged_dim=64]
        let merged_dim = cfg.merged_dim(); // 16*4=64
        let input = Tensor::randn(0.0f32, 1.0, (1, 3, merged_dim), &device).unwrap();
        let output = model.project(&input).unwrap();
        assert_eq!(output.dims(), &[1, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_forward_multimodal_no_images() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg.model_config, &device);
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

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_forward_multimodal_with_images() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);

        let seq_len = 6;
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        // Use a placeholder token ID within test vocab range (256)
        // In production, IMAGE_TOKEN_ID=32044 but test vocab is only 256
        let test_image_token: u32 = 100;

        let input_ids = Tensor::from_vec(
            vec![
                1u32,
                test_image_token,
                test_image_token,
                test_image_token,
                5,
                6,
            ],
            (1, seq_len),
            &device,
        )
        .unwrap();

        // Create fake image embedding for 3 tokens
        let embedding =
            Tensor::zeros((3, cfg.model_config.hidden_size), DType::F32, &device).unwrap();
        let processed = crate::multimodal::ProcessedImage::new(embedding, 3);
        let mm_inputs = MultimodalInputs::with_images(
            vec![
                1,
                test_image_token,
                test_image_token,
                test_image_token,
                5,
                6,
            ],
            vec![(1, processed)],
        );

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
    fn test_device_method() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_model_img_projection_layers() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();
        assert_eq!(model.img_projection.len(), 2);
    }

    #[test]
    fn test_single_token_forward() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 1);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_phi3v_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi3VForCausalLM::new_with_config(&cfg, vb).unwrap();

        let cache_cfg = test_cache_config(&cfg.model_config, &device);
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
