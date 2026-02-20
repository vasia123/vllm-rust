//! Kimi-VL Vision-Language Model.
//!
//! Architecture: MoonVitPretrainedModel + KimiVLMultiModalProjector + DeepSeekForCausalLM.
//!
//! # Weight paths
//!
//! ```text
//! vision_tower.*              → MoonVitPretrainedModel
//! multi_modal_projector.*     → KimiVLMultiModalProjector
//! language_model.*            → DeepSeekForCausalLM
//! ```
//!
//! # Projector
//!
//! For each merged group of `kh*kw` patches with per-patch dimension `D`:
//! 1. `pre_norm`: LayerNorm(D, eps=1e-5) applied per patch position
//! 2. Flatten `kh*kw` patches → dim `D*kh*kw`
//! 3. `linear_1`: `[D*kh*kw → D*kh*kw]` with bias + GELU
//! 4. `linear_2`: `[D*kh*kw → text_hidden]` with bias
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/kimi_vl.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::deepseek::DeepSeekForCausalLM;
use super::moonvit::{MoonVitConfig, MoonVitPretrainedModel};

// ─── Config ──────────────────────────────────────────────────────────────────

struct KimiVLConfig {
    vision: MoonVitConfig,
    image_token_id: u32,
    text_hidden_size: usize,
}

impl KimiVLConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision = MoonVitConfig::from_extra(&cfg.extra);
        let image_token_id = cfg
            .extra
            .get("media_placeholder_token_id")
            .or_else(|| cfg.extra.get("image_token_id"))
            .and_then(|v| v.as_u64())
            .unwrap_or(151646) as u32;
        Self {
            vision,
            image_token_id,
            text_hidden_size: cfg.hidden_size,
        }
    }
}

// ─── Projector ───────────────────────────────────────────────────────────────

/// Projects MoonViT patch groups to LLM hidden size.
///
/// Input: `[N, kh*kw, vision_hidden]` — N merged tokens, each a group of `kh*kw` patches.
/// Output: `[N, text_hidden]`.
struct KimiVLMultiModalProjector {
    pre_norm: LayerNorm, // LayerNorm(vision_hidden, eps=1e-5) — per patch position
    linear_1: Linear,    // [D_flat → D_flat] with bias, D_flat = vision_hidden * kh * kw
    linear_2: Linear,    // [D_flat → text_hidden] with bias
    flat_dim: usize,     // D_flat = vision_hidden * kh * kw
}

impl KimiVLMultiModalProjector {
    fn new(vision_cfg: &MoonVitConfig, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let [kh, kw] = vision_cfg.merge_kernel_size;
        let flat_dim = vision_cfg.hidden_size * kh * kw;

        // pre_norm operates on per-patch dim (vision_hidden) before flattening
        let pre_norm = layer_norm(vision_cfg.hidden_size, 1e-5, vb.pp("pre_norm"))?;
        let linear_1 = linear(flat_dim, flat_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(flat_dim, text_hidden, vb.pp("linear_2"))?;

        Ok(Self {
            pre_norm,
            linear_1,
            linear_2,
            flat_dim,
        })
    }

    /// Process concatenated patch groups.
    ///
    /// `image_features`: `[N, kh*kw, vision_hidden]`.
    /// Returns `[N, text_hidden]`.
    fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        // LayerNorm on last dim (vision_hidden): [N, kh*kw, D] → [N, kh*kw, D]
        let x = self.pre_norm.forward(image_features)?;

        // Flatten patch groups: [N, kh*kw, D] → [N, D_flat]
        let n = x.dim(0)?;
        let x = x.reshape((n, self.flat_dim))?;

        // Two-layer MLP
        let x = self.linear_1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear_2.forward(&x)
    }
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// Kimi-VL: MoonVit vision encoder + projector + DeepSeek-V2 language model.
pub struct KimiVLForConditionalGeneration {
    vision_tower: MoonVitPretrainedModel,
    projector: KimiVLMultiModalProjector,
    language_model: DeepSeekForCausalLM,
    #[allow(dead_code)]
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl KimiVLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let kimi_cfg = KimiVLConfig::from_model_config(cfg);

        let vision_tower = MoonVitPretrainedModel::new(&kimi_cfg.vision, vb.pp("vision_tower"))?;
        let projector = KimiVLMultiModalProjector::new(
            &kimi_cfg.vision,
            kimi_cfg.text_hidden_size,
            vb.pp("multi_modal_projector"),
        )?;
        // DeepSeek-V2 LLM reads text config from the top-level ModelConfig
        let language_model = DeepSeekForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            projector,
            language_model,
            image_token_id: kimi_cfg.image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(cfg, vb)
    }

    /// Encode images: vision tower → projector → `[total_tokens, text_hidden]`.
    ///
    /// `pixel_values`: `[L, C, ps, ps]`.
    /// `grid_hws`: `(h, w)` per image.
    pub fn encode_images(&self, pixel_values: &Tensor, grid_hws: &[[usize; 2]]) -> Result<Tensor> {
        let patches = self.vision_tower.forward(pixel_values, grid_hws)?;
        let all_patches = Tensor::cat(&patches, 0)?; // [total_merged, kh*kw, D]
        self.projector.forward(&all_patches) // [total_merged, text_hidden]
    }

    /// Merge projected image features into text embeddings at image-token positions.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;
            for (i, row) in emb.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = row.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for KimiVLForConditionalGeneration {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::mla_cache_config::MLACacheConfig;
    use candle_core::{DType, Device};
    use serde_json::json;

    /// Minimal config: tiny MoonViT + tiny DeepSeek-V2 text model.
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        // KimiVL-specific
        extra.insert("media_placeholder_token_id".to_string(), json!(5));
        // DeepSeek-V2 text config fields
        extra.insert("kv_lora_rank".to_string(), json!(4));
        extra.insert("qk_nope_head_dim".to_string(), json!(8));
        extra.insert("qk_rope_head_dim".to_string(), json!(8));
        extra.insert("v_head_dim".to_string(), json!(8));
        extra.insert("q_lora_rank".to_string(), serde_json::Value::Null);
        extra.insert("num_experts".to_string(), json!(4));
        extra.insert("num_experts_per_tok".to_string(), json!(2));
        extra.insert("moe_intermediate_size".to_string(), json!(32));
        extra.insert("first_k_dense_replace".to_string(), json!(1));
        extra.insert("moe_layer_freq".to_string(), json!(1));
        extra.insert("n_shared_experts".to_string(), json!(1));
        extra.insert("routed_scaling_factor".to_string(), json!(1.0));
        extra.insert("norm_topk_prob".to_string(), json!(false));
        extra.insert("topk_method".to_string(), json!("greedy"));
        extra.insert("n_group".to_string(), json!(1));
        extra.insert("topk_group".to_string(), json!(1));
        extra.insert("scoring_func".to_string(), json!("softmax"));

        // Vision config (small)
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "in_channels": 1,
                "patch_size": 2,
                "init_pos_emb_height": 4,
                "init_pos_emb_width": 4,
                "merge_kernel_size": [2, 2],
                "rope_max_height": 16,
                "rope_max_width": 16,
                "rope_theta": 10000.0
            }),
        );

        ModelConfig {
            architectures: vec!["KimiVLForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
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

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        // DeepSeek-V2 uses MLA cache
        let kv_lora_rank = cfg
            .extra
            .get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let qk_rope_head_dim = cfg
            .extra
            .get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let qk_nope_head_dim = cfg
            .extra
            .get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let mla_cfg = MLACacheConfig::new(
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            cfg.num_attention_heads,
            16, // block_size
            8,  // num_blocks
            cfg.num_hidden_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_cfg).unwrap()
    }

    #[test]
    fn test_kimi_vl_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiVLForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "KimiVLForConditionalGeneration construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_vision_tower_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiVLForConditionalGeneration::new(&cfg, vb).unwrap();
        let vc = &model.vision_tower.cfg;

        // 4×4 grid = 16 patches; in_channels=1, patch_size=2
        let ps = vc.patch_size;
        let pixel_values =
            Tensor::zeros((16usize, vc.in_channels, ps, ps), DType::F32, &device).unwrap();
        let grid_hws = [[4usize, 4usize]];

        let out = model
            .vision_tower
            .forward(&pixel_values, &grid_hws)
            .unwrap();
        // After merge(2,2): 2×2 = 4 tokens, each [4, D=16]
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dims(), &[4, 4, 16]);
    }

    #[test]
    fn test_projector_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiVLForConditionalGeneration::new(&cfg, vb).unwrap();

        // 4 merged tokens, each with 4 patches of dim 16
        let features = Tensor::zeros((4usize, 4usize, 16usize), DType::F32, &device).unwrap();
        let out = model.projector.forward(&features).unwrap();
        // [4, text_hidden=64]
        assert_eq!(out.dims(), &[4, 64]);
    }

    #[test]
    fn test_text_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiVLForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv = make_cache(&cfg, &device);

        let input_ids = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let block_table = crate::kv_cache::BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let out = model.forward(&input_ids, 0, &mut kv, &block_table, &slot_mapping);
        assert!(out.is_ok(), "text forward failed: {:?}", out.err());
        let out = out.unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_encode_images_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KimiVLForConditionalGeneration::new(&cfg, vb).unwrap();
        let vc = &model.vision_tower.cfg;

        // 4×4 grid (16 patches)
        let ps = vc.patch_size;
        let pixel_values =
            Tensor::zeros((16usize, vc.in_channels, ps, ps), DType::F32, &device).unwrap();
        let grid_hws = [[4usize, 4usize]];

        let out = model.encode_images(&pixel_values, &grid_hws).unwrap();
        // 4 merged tokens → [4, text_hidden=64]
        assert_eq!(out.dims(), &[4, 64]);
    }
}
