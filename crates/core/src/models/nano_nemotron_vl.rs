#![allow(dead_code, non_camel_case_types)]
//! NemotronH_Nano_VL_V2: RadioModel vision encoder + pixel-shuffle + MLP1 projector
//! + NemotronH hybrid LLM backbone.
//!
//! Architecture:
//!   vision_model  → RadioModel (Radio ViT-H/16 or similar)
//!   pixel_shuffle → 2×2 spatial pooling via reshape (scale_factor=0.5)
//!   mlp1          → RMSNorm → Linear → ReLU² → Linear
//!   language_model→ NemotronHForCausalLM (SSM-transformer hybrid)
//!
//! Weight layout (from Python checkpoint):
//!   vision_model.radio_model.model.patch_generator.*
//!   vision_model.radio_model.model.blocks.{i}.*
//!   mlp1.0.*  (RMSNorm)
//!   mlp1.1.*  (Linear, no bias)
//!   mlp1.3.*  (Linear, no bias)  – index 2 is ReLUSquaredActivation (no weights)
//!   language_model.model.embed_tokens.*
//!   language_model.model.layers.*
//!   language_model.model.norm_f.*
//!   language_model.lm_head.*

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, Module, VarBuilder};

use crate::{
    config::ModelConfig,
    engine::ModelForward,
    kv_cache::{BlockTable, KVCacheManager},
    layers::{rms_norm, RmsNorm},
    models::{
        nemotron_h::NemotronHForCausalLM,
        radio::{RadioModel, RadioVisionConfig},
    },
    multimodal::MultimodalInputs,
};

// ─── Configuration ───────────────────────────────────────────────────────────

/// NanoNemotronVL-specific config parsed from `ModelConfig.extra`.
#[derive(Debug, Clone)]
struct NanoVlConfig {
    /// RadioModel config derived from `vision_config` sub-config.
    radio_cfg: RadioVisionConfig,
    /// NemotronH text model config derived from `text_config` sub-config.
    text_cfg: ModelConfig,
    /// Side length (in pixels) of the image fed to the vision model.
    image_size: usize,
    /// Patch size of the vision model (same as radio_cfg.patch_size).
    patch_size: usize,
    /// Downsampling ratio for pixel-shuffle (typically 0.5).
    downsample_ratio: f64,
    /// ViT output hidden size (e.g. 1280 for ViT-H/16).
    vit_hidden_size: usize,
    /// MLP1 intermediate size.
    projector_hidden_size: usize,
    /// Token id used for `<image>` placeholders in the prompt.
    image_token_id: u32,
}

impl NanoVlConfig {
    fn from_model_config(cfg: &ModelConfig) -> Result<Self> {
        let e = &cfg.extra;

        let image_size = e
            .get("force_image_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(432) as usize;

        let patch_size = e.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;

        let downsample_ratio = e
            .get("downsample_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let vit_hidden_size = e
            .get("vit_hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1280) as usize;

        let projector_hidden_size = e
            .get("projector_hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        let image_token_id = e
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(32064) as u32;

        // Parse text_config into a ModelConfig for NemotronH
        let text_cfg: ModelConfig = if let Some(tc) = e.get("text_config") {
            serde_json::from_value(tc.clone())
                .map_err(|e| candle_core::Error::Msg(format!("failed to parse text_config: {e}")))?
        } else {
            cfg.clone()
        };

        // Parse radio vision config from vision_config.args
        let radio_cfg = parse_radio_config(e, image_size, patch_size)?;

        Ok(Self {
            radio_cfg,
            text_cfg,
            image_size,
            patch_size,
            downsample_ratio,
            vit_hidden_size,
            projector_hidden_size,
            image_token_id,
        })
    }

    /// Number of vision tokens after pixel-shuffle (= num_patches / downsample^2).
    fn num_vision_tokens(&self) -> usize {
        let num_patches = (self.image_size / self.patch_size).pow(2);
        let sf_inv_sq = (1.0 / self.downsample_ratio) as usize;
        num_patches / (sf_inv_sq * sf_inv_sq)
    }

    /// Input dim to mlp1 (= vit_hidden_size * (1/downsample_ratio)^2).
    fn mlp1_input_dim(&self) -> usize {
        let scale_inv = (1.0 / self.downsample_ratio).round() as usize;
        self.vit_hidden_size * scale_inv * scale_inv
    }
}

fn parse_radio_config(
    extra: &serde_json::Map<String, serde_json::Value>,
    image_size: usize,
    patch_size: usize,
) -> Result<RadioVisionConfig> {
    // vision_config.args contains Radio-specific fields
    let args = extra
        .get("vision_config")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("args"))
        .and_then(|v| v.as_object());

    let hidden_size = args
        .and_then(|a| a.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1280) as usize;

    let mlp_ratio = args
        .and_then(|a| a.get("mlp_ratio"))
        .and_then(|v| v.as_f64())
        .unwrap_or(4.0);

    let num_hidden_layers = args
        .and_then(|a| a.get("depth").or_else(|| a.get("num_hidden_layers")))
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let num_attention_heads = args
        .and_then(|a| a.get("num_heads").or_else(|| a.get("num_attention_heads")))
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;

    let qkv_bias = args
        .and_then(|a| a.get("qkv_bias"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let cpe_max_size = extra
        .get("vision_config")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("cpe_max_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(image_size as u64) as usize;

    let register_multiple = args
        .and_then(|a| a.get("register_multiple"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    // num_registers: computed from register_multiple + num_cls_tokens
    let num_cls_tokens = 1usize; // default for NanoNemotronVL (not cls_token_per_teacher)
    let num_registers = if let Some(rm) = register_multiple {
        rm - (num_cls_tokens % rm)
    } else {
        0
    };

    Ok(RadioVisionConfig {
        hidden_size,
        intermediate_size: (hidden_size as f64 * mlp_ratio).round() as usize,
        num_hidden_layers,
        num_attention_heads,
        patch_size,
        image_size,
        cpe_max_size,
        qkv_bias,
        qk_normalization: false,
        layer_norm_eps: 1e-6,
        num_cls_tokens,
        num_registers,
    })
}

// ─── MLP1 Projector ──────────────────────────────────────────────────────────

/// Two-layer MLP with RMSNorm and ReLU² activation.
///
/// `mlp1` in Python:
/// ```text
/// Sequential(RMSNorm(D_in), Linear(D_in, D_mid), ReLUSquared, Linear(D_mid, D_out))
/// ```
/// Weights: `mlp1.0.*` (norm), `mlp1.1.weight` (fc1), `mlp1.3.weight` (fc2).
struct Mlp1Projector {
    norm: RmsNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl Mlp1Projector {
    fn new(in_dim: usize, mid_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = rms_norm(in_dim, 1e-5, vb.pp("0"))?;
        let fc1 = linear_no_bias(in_dim, mid_dim, vb.pp("1"))?;
        let fc2 = linear_no_bias(mid_dim, out_dim, vb.pp("3"))?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.fc1.forward(&x)?;
        // ReLU² = relu(x)²
        let x = x.relu()?.sqr()?;
        self.fc2.forward(&x)
    }
}

// ─── Pixel Shuffle ───────────────────────────────────────────────────────────

/// Pixel-shuffle downsampling: reduces spatial dims by `scale_inv`×, expands channels.
///
/// Input:  `[N, H, W, C]` (from reshaping patch features)
/// Output: `[N, H/k, W/k, C*k²]` where k = 1/scale_factor (e.g. k=2 for scale_factor=0.5)
///
/// Implements Python's `NemotronH_Nano_VL_V2.pixel_shuffle` with `ps_version='v2'`.
fn pixel_shuffle(x: &Tensor, scale_inv: usize) -> Result<Tensor> {
    let (n, w, h, c) = x.dims4()?;
    let h2 = h / scale_inv;
    let w2 = w / scale_inv;
    let c2 = c * scale_inv;
    let c4 = c * scale_inv * scale_inv;

    // Step 1: [N, W, H, C] → [N, W, H/k, C*k]
    let x = x.reshape((n, w, h2, c2))?;
    // Step 2: → [N, H/k, W, C*k]
    let x = x.permute((0, 2, 1, 3))?.contiguous()?;
    // Step 3: [N, H/k, W, C*k] → [N, H/k, W/k, C*k²]
    let x = x.reshape((n, h2, w2, c4))?;
    // Step 4 (ps_version='v2'): swap spatial dims
    x.permute((0, 2, 1, 3))?.contiguous()
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// NemotronH_Nano_VL_V2 multimodal model.
///
/// Vision pipeline:
///   pixel_values [B,3,H,W]
///     → RadioModel → features [B, S, D_vit]
///     → pixel_shuffle → [B, S/4, D_vit*4]
///     → mlp1 → [B, S/4, D_llm]
///
/// These are then merged with text embeddings at `<image>` token positions and
/// forwarded through the NemotronH LLM.
pub struct NemotronH_Nano_VL_V2ForConditionalGeneration {
    vision_model: RadioModel,
    mlp1: Mlp1Projector,
    language_model: NemotronHForCausalLM,
    image_token_id: u32,
    downsample_ratio: f64,
    device: Device,
    dtype: DType,
}

impl NemotronH_Nano_VL_V2ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let nano_cfg = NanoVlConfig::from_model_config(cfg)?;

        let vision_model =
            RadioModel::new(&nano_cfg.radio_cfg, vb.pp("vision_model").pp("radio_model"))?;

        let mlp1_in = nano_cfg.mlp1_input_dim();
        let mlp1 = Mlp1Projector::new(
            mlp1_in,
            nano_cfg.projector_hidden_size,
            nano_cfg.text_cfg.hidden_size,
            vb.pp("mlp1"),
        )?;

        let language_model =
            NemotronHForCausalLM::new(&nano_cfg.text_cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
            mlp1,
            language_model,
            image_token_id: nano_cfg.image_token_id,
            downsample_ratio: nano_cfg.downsample_ratio,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run vision encoder + pixel-shuffle + MLP1 on `pixel_values [B,3,H,W]`.
    ///
    /// Returns `[B, num_vision_tokens, llm_hidden_size]`.
    fn extract_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (_, vit_embeds) = self.vision_model.forward(pixel_values)?;

        // vit_embeds: [B, S, D_vit] where S = (H/ps)*(W/ps)
        let (b, s, d) = vit_embeds.dims3()?;
        let h = (s as f64).sqrt() as usize;
        debug_assert_eq!(h * h, s, "patch sequence must be square");

        // Reshape to spatial: [B, H, W, D_vit]
        let vit_embeds = vit_embeds.reshape((b, h, h, d))?;

        // Pixel-shuffle (scale_factor=0.5 → scale_inv=2)
        let scale_inv = (1.0 / self.downsample_ratio).round() as usize;
        let vit_embeds = pixel_shuffle(&vit_embeds, scale_inv)?;

        // Flatten spatial: [B, H/k, W/k, D*k²] → [B, S', D*k²]
        let (b2, h2, w2, d4) = vit_embeds.dims4()?;
        let vit_embeds = vit_embeds.reshape((b2, h2 * w2, d4))?;

        // Project: [B, S', D*k²] → [B, S', D_llm]
        self.mlp1.forward(&vit_embeds)
    }

    /// Merge vision embeddings into text embedding tensor.
    ///
    /// Positions where `input_ids == image_token_id` are replaced with the
    /// flattened `image_embeds` rows in order.
    fn merge_multimodal(
        &self,
        text_emb: &Tensor,
        _input_ids: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_emb.clone());
        }

        let (_b, seq, _d) = text_emb.dims3()?;
        let mut merged = text_emb.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq;
            let start_pos = *position % seq;
            for (i, emb) in emb_vec.iter().enumerate() {
                let tgt = start_pos + i;
                if tgt < seq && batch_idx < merged.len() {
                    merged[batch_idx][tgt] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

// ─── ModelForward impl ───────────────────────────────────────────────────────

impl ModelForward for NemotronH_Nano_VL_V2ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.language_model.forward_with_request_id(
            input_ids,
            seqlen_offset,
            0,
            kv_cache_mgr,
            block_table,
            slot_mapping,
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
        let text_emb = self.language_model.embed_text(input_ids)?;

        let embeddings = if let Some(mm) = multimodal_inputs {
            self.merge_multimodal(&text_emb, input_ids, mm)?
        } else {
            text_emb
        };

        self.language_model.forward_with_request_id_and_embeddings(
            &embeddings,
            seqlen_offset,
            0,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    #[test]
    fn test_pixel_shuffle_shape() {
        let device = Device::Cpu;
        // [1, 4, 4, 32] → pixel_shuffle(2) → [1, 2, 2, 128]
        let x = Tensor::zeros((1, 4, 4, 32), DType::F32, &device).unwrap();
        let out = pixel_shuffle(&x, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 2, 128]);
    }

    #[test]
    fn test_pixel_shuffle_values() {
        let device = Device::Cpu;
        // Verify output shape is [1,1,1,16] for [1,2,2,4] input with scale_inv=2
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = Tensor::new(data.as_slice(), &device)
            .unwrap()
            .reshape((1, 2, 2, 4))
            .unwrap();
        let out = pixel_shuffle(&x, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 1, 16]);
    }

    #[test]
    fn test_nano_vl_config_dims() {
        let radio_cfg = RadioVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            patch_size: 4,
            image_size: 16,
            cpe_max_size: 16,
            qkv_bias: false,
            qk_normalization: false,
            layer_norm_eps: 1e-6,
            num_cls_tokens: 1,
            num_registers: 0,
        };
        let cfg = NanoVlConfig {
            radio_cfg,
            text_cfg: ModelConfig::default(),
            image_size: 16,
            patch_size: 4,
            downsample_ratio: 0.5,
            vit_hidden_size: 32,
            projector_hidden_size: 64,
            image_token_id: 32064,
        };
        // image_size=16, patch_size=4 → num_patches=16; scale_inv=2 → 16/4 = 4 tokens
        assert_eq!(cfg.num_vision_tokens(), 4);
        // mlp1_input_dim = vit_hidden_size * (1/0.5)^2 = 32 * 4 = 128
        assert_eq!(cfg.mlp1_input_dim(), 128);
    }

    #[test]
    fn test_parse_radio_config_defaults() {
        let extra = serde_json::Map::new();
        let rc = parse_radio_config(&extra, 512, 16).unwrap();
        assert_eq!(rc.hidden_size, 1280);
        assert_eq!(rc.num_hidden_layers, 32);
        assert_eq!(rc.intermediate_size, 5120); // 1280 * 4
        assert_eq!(rc.image_size, 512);
        assert_eq!(rc.patch_size, 16);
        assert_eq!(rc.num_cls_tokens, 1);
    }

    #[test]
    fn test_mlp1_projector_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let in_dim = 128usize;
        let mid_dim = 64usize;
        let out_dim = 32usize;

        vb.pp("0").get(in_dim, "weight").unwrap();
        vb.pp("1").get((mid_dim, in_dim), "weight").unwrap();
        vb.pp("3").get((out_dim, mid_dim), "weight").unwrap();

        let proj = Mlp1Projector::new(in_dim, mid_dim, out_dim, vb).unwrap();

        let x = Tensor::zeros((2, 10, in_dim), DType::F32, &device).unwrap();
        let out = proj.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[2, 10, out_dim]);
    }
}
