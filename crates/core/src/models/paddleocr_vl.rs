//! PaddleOCR-VL vision-language model.
//!
//! Architecture:
//! - Vision: SiglipVisionModel — SigLIP ViT with 2D RoPE + packing position embeddings.
//!   Structurally identical to `KeyeSiglipVisionTransformer` (keye_vl.rs).
//! - Projector: `Projector` — LayerNorm (per-patch) → 2×2 spatial merge → linear_1(GELU) → linear_2.
//!   Same structure as `KeyeProjector`.
//! - Language: `Ernie4_5ForCausalLM` — dense ERNIE 4.5, architecturally equivalent to
//!   `LlamaForCausalLM` with modified RoPE style. We use `Ernie45MoeForCausalLM` via config alias.
//!
//! # Weight paths
//!
//! ```text
//! visual.vision_model.embeddings.patch_embedding.{weight,bias}
//! visual.vision_model.embeddings.position_embedding.weight
//! visual.vision_model.embeddings.packing_position_embedding.weight
//! visual.vision_model.encoder.layers.{i}.{layer_norm1,layer_norm2}.*
//! visual.vision_model.encoder.layers.{i}.self_attn.{qkv_proj,out_proj}.*
//! visual.vision_model.encoder.layers.{i}.mlp.{fc1,fc2}.*
//! visual.vision_model.post_layernorm.*
//! mlp_AR.{pre_norm,linear_1,linear_2}.*
//! language_model.{model.*,lm_head.*}
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/paddleocr_vl.py`

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::ernie45_moe::Ernie45MoeForCausalLM;
use super::keye_vl::{KeyeProjector, KeyeSiglipVisionTransformer, KeyeVisionConfig};

// ─── merge_multimodal ─────────────────────────────────────────────────────────

/// Replace image-patch positions in `text_embeds` with projected image embeddings.
fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }
    let (_b, seq_len, _d) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_vec3::<f32>()?;
    for (position, processed) in &mm_inputs.image_embeddings {
        let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
        let batch_idx = position / seq_len;
        let start_pos = position % seq_len;
        for (i, emb) in emb_vec.iter().enumerate() {
            let target = start_pos + i;
            if target < seq_len && batch_idx < merged.len() {
                merged[batch_idx][target] = emb.clone();
            }
        }
    }
    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main Model ───────────────────────────────────────────────────────────────

/// PaddleOCR-VL vision-language model.
pub struct PaddleOCRVLForConditionalGeneration {
    visual: KeyeSiglipVisionTransformer,
    mlp_ar: KeyeProjector,
    language_model: Ernie45MoeForCausalLM,
    device: Device,
}

impl PaddleOCRVLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let vis_cfg = KeyeVisionConfig::from_json(&vis_json);

        let visual = KeyeSiglipVisionTransformer::new(&vis_cfg, vb.pp("visual"))?;
        let mlp_ar = KeyeProjector::new(vis_cfg.embed_dim, cfg.hidden_size, vb.pp("mlp_AR"))?;
        let language_model = Ernie45MoeForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            visual,
            mlp_ar,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Encode image patches: `patches [t*h*w, in_ch*ps²]`, grid `(t, h, w)`.
    /// Returns `[t*(h/2)*(w/2), text_hidden]`.
    pub fn encode_images(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, h, w) = grid;
        let vis_feats = self.visual.forward(patches, (h, w))?;
        self.mlp_ar.forward(&vis_feats, (t, h, w))
    }
}

impl crate::engine::ModelForward for PaddleOCRVLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_with_embeddings(
            &embeddings,
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

    fn device(&self) -> &Device {
        &self.device
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
        let embeddings = if let Some(mm) = multimodal_inputs {
            merge_multimodal(&text_embeddings, mm, &self.device)?
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
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use candle_core::DType;
    use serde_json::json;

    fn test_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            json!({
                "patch_size": 2,
                "num_channels": 1,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "image_size": 8,
                "layer_norm_eps": 1e-6,
            }),
        );
        extra.insert("image_token_id".to_string(), json!(9u32));
        // PaddleOCR VL uses Ernie45 MoE config fields
        extra.insert("moe_num_experts".to_string(), json!(0u32));
        extra.insert("num_experts_per_tok".to_string(), json!(0u32));

        ModelConfig {
            architectures: vec!["PaddleOCRVLForConditionalGeneration".to_string()],
            hidden_size: 32,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 64,
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
        KVCacheManager::new(&CacheConfig {
            block_size: 4,
            num_blocks: 32,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_paddleocr_vl_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaddleOCRVLForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_paddleocr_vl_vision_encode() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaddleOCRVLForConditionalGeneration::new(&cfg, vb).unwrap();

        // 16 patches (4×4 grid), patch_input_dim = 1*2*2 = 4
        let patches = Tensor::zeros((16usize, 4), DType::F32, &device).unwrap();
        let result = model.encode_images(&patches, (1, 4, 4));
        assert!(result.is_ok(), "encode_images failed: {:?}", result.err());
        // 4×4 → 2×2 after 2×2 merge → 4 merged tokens, text_hidden=32
        assert_eq!(result.unwrap().dims(), &[4, 32]);
    }

    #[test]
    fn test_paddleocr_vl_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = PaddleOCRVLForConditionalGeneration::new(&cfg, vb).unwrap();

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let result = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }
}
