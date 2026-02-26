//! LlamaNemotronVLChatModel: RadioModel vision encoder + pixel-shuffle + MLP1 projector
//! + LlamaForCausalLM backbone.
//!
//! Architecture for nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1:
//!   vision_model → RadioModel (Radio ViT-H/14)
//!   pixel_shuffle → 2×2 spatial downsampling (scale_factor=0.5, ps_version='v2')
//!   mlp1 → LayerNorm(bias=True) → Linear(bias=True) → GELU → Linear(bias=True)
//!   language_model → LlamaForCausalLM (Llama-3.1-8B)
//!
//! Weight layout:
//!   vision_model.model.patch_generator.*   (RadioModel)
//!   vision_model.model.blocks.{i}.*        (RadioModel)
//!   vision_model.model.cls_token           (RadioModel)
//!   mlp1.0.{weight,bias}  (LayerNorm)
//!   mlp1.1.{weight,bias}  (Linear)
//!   mlp1.3.{weight,bias}  (Linear, index 2 = GELU has no weights)
//!   language_model.model.embed_tokens.*
//!   language_model.model.layers.*
//!   language_model.lm_head.*
//!
//! NOTE: Differs from NemotronH_Nano_VL_V2 (nano_nemotron_vl.rs):
//!   - Vision: `vision_model.*` (no `radio_model` nesting)
//!   - Config: `vision_config.*` flat (no `vision_config.args` nesting)
//!   - Projector: LayerNorm+GELU (vs RMSNorm+ReLU²)
//!   - LLM: LlamaForCausalLM (vs NemotronHForCausalLM)
//!
//! Reference: `reference/vllm/vllm/model_executor/models/nemotron_vl.py`

#![allow(dead_code)]

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, VarBuilder};

use crate::{
    config::ModelConfig,
    engine::ModelForward,
    kv_cache::{BlockTable, KVCacheManager},
    models::{llama::LlamaForCausalLM, radio::RadioVisionConfig},
    multimodal::MultimodalInputs,
};

use super::radio::RadioModel;

// ─── Configuration ───────────────────────────────────────────────────────────

/// NemotronVL config parsed from `ModelConfig.extra`.
#[derive(Debug, Clone)]
struct NemotronVLConfig {
    /// RadioModel vision config.
    radio_cfg: RadioVisionConfig,
    /// LLM (Llama) config from `text_config`.
    text_cfg: ModelConfig,
    /// Side length of the image fed to the vision model.
    image_size: usize,
    /// Patch size of the vision model.
    patch_size: usize,
    /// Pixel-shuffle downsampling ratio (typically 0.5).
    downsample_ratio: f64,
    /// ViT output hidden size.
    vit_hidden_size: usize,
    /// MLP1 intermediate hidden size.
    projector_hidden_size: usize,
    /// Image placeholder token ID.
    image_token_id: u32,
}

impl NemotronVLConfig {
    fn from_model_config(cfg: &ModelConfig) -> Result<Self> {
        let e = &cfg.extra;

        let image_size = e
            .get("force_image_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(560) as usize;

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
            .unwrap_or(128006) as u32;

        // Parse text_config into ModelConfig for LlamaForCausalLM
        let text_cfg: ModelConfig = if let Some(tc) = e.get("text_config") {
            serde_json::from_value(tc.clone()).map_err(|err| {
                candle_core::Error::Msg(format!("failed to parse text_config: {err}"))
            })?
        } else {
            cfg.clone()
        };

        // Parse radio config from vision_config (flat layout, no 'args' nesting)
        let radio_cfg = parse_radio_config_flat(e, image_size)?;
        let patch_size = radio_cfg.patch_size;

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

    /// Number of vision tokens after pixel-shuffle.
    fn num_vision_tokens(&self) -> usize {
        let num_patches = (self.image_size / self.patch_size).pow(2);
        let scale_inv = (1.0 / self.downsample_ratio).round() as usize;
        num_patches / (scale_inv * scale_inv)
    }

    /// Input dimension to mlp1 = vit_hidden_size × (1/downsample_ratio)².
    fn mlp1_input_dim(&self) -> usize {
        let scale_inv = (1.0 / self.downsample_ratio).round() as usize;
        self.vit_hidden_size * scale_inv * scale_inv
    }
}

/// Parse RadioVisionConfig from the flat `vision_config` object in the HF config.
///
/// For `LlamaNemotronVLChatModel`, the vision_config is a standard HF RadioConfig
/// with fields at top level (not nested under `args`).
fn parse_radio_config_flat(
    extra: &serde_json::Map<String, serde_json::Value>,
    image_size: usize,
) -> Result<RadioVisionConfig> {
    let vc = extra
        .get("vision_config")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let hidden_size = vc
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1280) as usize;

    let mlp_ratio = vc.get("mlp_ratio").and_then(|v| v.as_f64()).unwrap_or(4.0);

    let num_hidden_layers = vc
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let num_attention_heads = vc
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize;

    let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as usize;

    let cpe_max_size = vc
        .get("cpe_max_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;

    let qkv_bias = vc.get("qkv_bias").and_then(|v| v.as_bool()).unwrap_or(true);

    let num_cls_tokens = vc
        .get("num_cls_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    // register_multiple: total tokens must be divisible by this.
    let register_multiple = vc
        .get("register_multiple")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let num_tokens = (image_size / patch_size).pow(2);
    let num_registers = if register_multiple > 0 {
        let rem = num_tokens % register_multiple;
        if rem != 0 {
            register_multiple - rem
        } else {
            0
        }
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

/// MLP1 projector: LayerNorm(bias) → Linear(bias) → GELU → Linear(bias).
///
/// Weight keys: `{prefix}.0.{weight,bias}`, `{prefix}.1.{weight,bias}`,
///              `{prefix}.3.{weight,bias}` (index 2 = GELU, no weights).
struct NemotronVLProjector {
    norm: candle_nn::LayerNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl NemotronVLProjector {
    fn new(in_dim: usize, mid_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = layer_norm(in_dim, 1e-5, vb.pp("0"))?;
        let fc1 = linear(in_dim, mid_dim, vb.pp("1"))?;
        let fc2 = linear(mid_dim, out_dim, vb.pp("3"))?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── Pixel Shuffle ───────────────────────────────────────────────────────────

/// Pixel-shuffle downsampling (ps_version='v2').
///
/// Input:  `[N, H, W, C]`
/// Output: `[N, H/k, W/k, C*k²]`  where k = 1/scale_factor (e.g. k=2 for scale_factor=0.5)
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

/// LlamaNemotronVL: RadioModel vision encoder + pixel-shuffle + MLP1 + LlamaForCausalLM.
///
/// Registered under architecture name `"Llama_Nemotron_Nano_VL"`.
pub struct LlamaNemotronVLForConditionalGeneration {
    vision_model: RadioModel,
    mlp1: NemotronVLProjector,
    language_model: LlamaForCausalLM,
    image_token_id: u32,
    downsample_ratio: f64,
    device: Device,
    dtype: DType,
}

impl LlamaNemotronVLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vl_cfg = NemotronVLConfig::from_model_config(cfg)?;

        // Vision encoder: RadioModel at `vision_model.*`
        // (RadioModel internally adds `model.*`, so full path is `vision_model.model.*`)
        let vision_model = RadioModel::new(&vl_cfg.radio_cfg, vb.pp("vision_model"))?;

        // MLP1 projector
        let mlp1_in = vl_cfg.mlp1_input_dim();
        let mlp1 = NemotronVLProjector::new(
            mlp1_in,
            vl_cfg.projector_hidden_size,
            vl_cfg.text_cfg.hidden_size,
            vb.pp("mlp1"),
        )?;

        // Llama LLM at `language_model.*`
        let language_model = LlamaForCausalLM::new(&vl_cfg.text_cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
            mlp1,
            language_model,
            image_token_id: vl_cfg.image_token_id,
            downsample_ratio: vl_cfg.downsample_ratio,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run vision pipeline on `pixel_values [B,3,H,W]`.
    ///
    /// Returns `[B, num_vision_tokens, llm_hidden_size]`.
    fn extract_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // RadioModel.forward returns (summary, features)
        let (_, features) = self.vision_model.forward(pixel_values)?;

        // features: [B, S, D_vit]
        let (b, s, d) = features.dims3()?;
        let h = (s as f64).sqrt() as usize;
        debug_assert_eq!(h * h, s, "patch sequence must be square");

        // Reshape to spatial: [B, H, W, D]
        let x = features.reshape((b, h, h, d))?;

        // Pixel-shuffle (scale_factor=0.5 → scale_inv=2)
        let scale_inv = (1.0 / self.downsample_ratio).round() as usize;
        let x = pixel_shuffle(&x, scale_inv)?;

        // Flatten to sequence: [B, H/k, W/k, D*k²] → [B, S', D*k²]
        let (b2, h2, w2, d4) = x.dims4()?;
        let x = x.reshape((b2, h2 * w2, d4))?;

        // Project: [B, S', D*k²] → [B, S', D_llm]
        self.mlp1.forward(&x)
    }

    /// Merge vision embeddings into text embedding tensor at `<image>` token positions.
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
            for (i, row) in emb_vec.iter().enumerate() {
                let tgt = start_pos + i;
                if tgt < seq && batch_idx < merged.len() {
                    merged[batch_idx][tgt] = row.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

// ─── ModelForward ────────────────────────────────────────────────────────────

impl ModelForward for LlamaNemotronVLForConditionalGeneration {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;
    use serde_json::json;

    fn make_extra(json: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        json.as_object().expect("json must be object").clone()
    }

    #[test]
    fn test_config_defaults() {
        let cfg = ModelConfig::default();
        let vl_cfg = NemotronVLConfig::from_model_config(&cfg).unwrap();
        // Default image_size=560, patch_size=14 → num_patches=1600; scale_inv=2 → 400 tokens
        assert_eq!(vl_cfg.image_size, 560);
        assert_eq!(vl_cfg.patch_size, 14);
        assert_eq!(vl_cfg.downsample_ratio, 0.5);
        assert_eq!(vl_cfg.num_vision_tokens(), 400);
        // mlp1_input_dim = 1280 * 4 = 5120
        assert_eq!(vl_cfg.mlp1_input_dim(), 5120);
    }

    #[test]
    fn test_config_custom() {
        let mut cfg = ModelConfig::default();
        cfg.extra = make_extra(json!({
            "force_image_size": 224,
            "vit_hidden_size": 64,
            "projector_hidden_size": 128,
            "downsample_ratio": 0.5,
            "vision_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 16,
            },
            "text_config": {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "vocab_size": 100,
                "max_position_embeddings": 128,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        }));
        let vl_cfg = NemotronVLConfig::from_model_config(&cfg).unwrap();
        // image_size=224, patch_size=16 → 196 patches; /4 = 49 tokens
        assert_eq!(vl_cfg.num_vision_tokens(), 49);
        assert_eq!(vl_cfg.mlp1_input_dim(), 64 * 4); // 256
        assert_eq!(vl_cfg.projector_hidden_size, 128);
    }

    #[test]
    fn test_pixel_shuffle_shape() {
        let device = Device::Cpu;
        // [1, 4, 4, 32] → scale_inv=2 → [1, 2, 2, 128]
        let x = Tensor::zeros((1, 4, 4, 32), DType::F32, &device).unwrap();
        let out = pixel_shuffle(&x, 2).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 2, 128]);
    }

    #[test]
    fn test_projector_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let in_dim = 64usize;
        let mid_dim = 32usize;
        let out_dim = 16usize;

        // Pre-populate weights for LayerNorm (weight + bias), fc1, fc2
        vb.pp("0").get(in_dim, "weight").unwrap();
        vb.pp("0").get(in_dim, "bias").unwrap();
        vb.pp("1").get((mid_dim, in_dim), "weight").unwrap();
        vb.pp("1").get(mid_dim, "bias").unwrap();
        vb.pp("3").get((out_dim, mid_dim), "weight").unwrap();
        vb.pp("3").get(out_dim, "bias").unwrap();

        let proj = NemotronVLProjector::new(in_dim, mid_dim, out_dim, vb).unwrap();

        let x = Tensor::zeros((2, 10, in_dim), DType::F32, &device).unwrap();
        let out = proj.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[2, 10, out_dim]);
    }

    #[test]
    fn test_radio_config_flat_parse() {
        let extra = make_extra(json!({
            "force_image_size": 448,
            "vision_config": {
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "patch_size": 16,
                "mlp_ratio": 4.0,
                "num_cls_tokens": 1,
            }
        }));
        let rc = parse_radio_config_flat(&extra, 448).unwrap();
        assert_eq!(rc.hidden_size, 128);
        assert_eq!(rc.num_hidden_layers, 4);
        assert_eq!(rc.patch_size, 16);
        assert_eq!(rc.image_size, 448);
        assert_eq!(rc.intermediate_size, 512); // 128 * 4
        assert_eq!(rc.num_cls_tokens, 1);
        assert_eq!(rc.num_registers, 0); // no register_multiple
    }
}
