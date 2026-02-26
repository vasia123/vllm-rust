//! Tarsier VLM (`TarsierForConditionalGeneration`).
//!
//! Tarsier is a vision-language model combining a CLIP or SigLIP vision encoder
//! with any registered LLM backbone (LLaMA, Mistral, Qwen2, etc.).
//!
//! **Architecture** (vision → projector → split-token augmentation → LLM):
//! - `vision_tower.vision_model.*`  — CLIP or SigLIP ViT
//! - `multi_modal_projector.{linear_1,linear_2}.*`  — 2-layer MLP projector
//! - `language_model.*`  — LLM backbone (determined by `text_config.architectures`)
//!
//! **Tarsier split-token logic** (`_add_tarsier_split_tokens` in Python):
//! Adds `image_newline` after each row of patches and `image_new` at the end
//! to give the model 2-D structural cues.  The two special-token embeddings are
//! looked up from the LLM's embedding table at construction time.
//!
//! **vision_feature_select_strategy = "default"**:
//! - CLIP: drop CLS token at position 0 → use patch positions 1..N
//! - SigLIP: use all positions (no CLS)
//!
//! Reference: `reference/vllm/vllm/model_executor/models/tarsier.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::models::clip::{ClipVisionConfig, ClipVisionTransformer};
use crate::models::llama::LlamaForCausalLM;
use crate::models::mistral::MistralForCausalLM;
use crate::models::qwen2::Qwen2ForCausalLM;
use crate::models::siglip::{SiglipVisionConfig, SiglipVisionTransformer};
use crate::multimodal::MultimodalInputs;

// ─── Vision tower ─────────────────────────────────────────────────────────────

/// Wraps CLIP or SigLIP vision transformer.  Selection driven by
/// `vision_config.model_type`: anything containing "siglip" → SigLIP,
/// otherwise → CLIP.
#[allow(dead_code)]
enum TarsierVisionTower {
    Clip(Box<ClipVisionTransformer>),
    Siglip(Box<SiglipVisionTransformer>),
}

impl TarsierVisionTower {
    fn new(vision_json: &serde_json::Value, vb: VarBuilder) -> Result<(Self, usize)> {
        let model_type = vision_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("clip_vision_model");

        if model_type.contains("siglip") {
            let cfg = SiglipVisionConfig::from_json(vision_json);
            let hidden = cfg.hidden_size;
            let tower = SiglipVisionTransformer::new(&cfg, vb.pp("vision_model"))?;
            Ok((Self::Siglip(Box::new(tower)), hidden))
        } else {
            let cfg = ClipVisionConfig::from_json(vision_json);
            let hidden = cfg.hidden_size;
            let tower = ClipVisionTransformer::new(&cfg, vb.pp("vision_model"))?;
            Ok((Self::Clip(Box::new(tower)), hidden))
        }
    }

    /// Extract patch features using "default" strategy.
    ///
    /// - CLIP: `[B, N+1, D]` → drop CLS at pos 0 → `[B, N, D]`
    /// - SigLIP: `[B, N, D]` — all positions (no CLS token)
    #[allow(dead_code)]
    fn forward_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        match self {
            Self::Clip(m) => {
                let x = m.forward(pixel_values)?;
                let n_plus_one = x.dim(1)?;
                x.narrow(1, 1, n_plus_one - 1)
            }
            Self::Siglip(m) => m.forward(pixel_values),
        }
    }
}

// ─── 2-layer MLP projector ────────────────────────────────────────────────────

/// `vision_hidden → gelu/relu → text_hidden`; optional bias on both layers.
struct TarsierMultiModalProjector {
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
    use_gelu: bool,
}

impl TarsierMultiModalProjector {
    fn new(
        vision_hidden: usize,
        text_hidden: usize,
        act: &str,
        with_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (l1, l2) = if with_bias {
            (
                linear(vision_hidden, text_hidden, vb.pp("linear_1"))?,
                linear(text_hidden, text_hidden, vb.pp("linear_2"))?,
            )
        } else {
            (
                linear_no_bias(vision_hidden, text_hidden, vb.pp("linear_1"))?,
                linear_no_bias(text_hidden, text_hidden, vb.pp("linear_2"))?,
            )
        };
        Ok(Self {
            linear_1: l1,
            linear_2: l2,
            use_gelu: !act.contains("relu"),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = if self.use_gelu {
            x.gelu_erf()?
        } else {
            x.relu()?
        };
        self.linear_2.forward(&x)
    }
}

// ─── LLM backbone enum ────────────────────────────────────────────────────────

enum TarsierLlm {
    Llama(Box<LlamaForCausalLM>),
    Mistral(Box<MistralForCausalLM>),
    Qwen2(Box<Qwen2ForCausalLM>),
}

impl TarsierLlm {
    fn new(arch: &str, cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        match arch {
            a if a.contains("Mistral") => {
                Ok(Self::Mistral(Box::new(MistralForCausalLM::new(cfg, vb)?)))
            }
            a if a.contains("Qwen2") => Ok(Self::Qwen2(Box::new(Qwen2ForCausalLM::new(cfg, vb)?))),
            _ => Ok(Self::Llama(Box::new(LlamaForCausalLM::new(cfg, vb)?))),
        }
    }

    fn embed_text(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.embed_text(ids),
            Self::Mistral(m) => m.embed_text(ids),
            Self::Qwen2(m) => m.embed_text(ids),
        }
    }

    fn forward_with_embeddings(
        &self,
        emb: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward_with_embeddings(
                emb,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
            Self::Mistral(m) => m.forward_with_embeddings(
                emb,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
            Self::Qwen2(m) => m.forward_with_embeddings(
                emb,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
        }
    }

    fn forward_decode_batch_with_embeddings(
        &self,
        emb: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward_decode_batch_with_embeddings(emb, sequences, kv_cache_mgr),
            Self::Mistral(m) => {
                m.forward_decode_batch_with_embeddings(emb, sequences, kv_cache_mgr)
            }
            Self::Qwen2(m) => m.forward_decode_batch_with_embeddings(emb, sequences, kv_cache_mgr),
        }
    }
}

// ─── Split-token logic ────────────────────────────────────────────────────────

/// Replicate Python `_add_tarsier_split_tokens`.
///
/// ```text
/// input:  [N, P, D]          (N images, P patches each)
/// output: [N, P + H + 1, D]  (H = sqrt(P); row newlines + end token appended)
/// ```
///
/// Grid layout: reshape `[N, P, D]` → `[N, H, W, D]`, cat newline_emb after
/// each row → `[N, H, W+1, D]`, flatten → `[N, H*(W+1), D]`, cat new_emb →
/// `[N, H*(W+1)+1, D]`.
fn add_tarsier_split_tokens(
    projected: &Tensor,   // [N, P, D]
    newline_emb: &Tensor, // [D]
    new_emb: &Tensor,     // [D]
) -> Result<Tensor> {
    let (n, p, d) = projected.dims3()?;
    let h = (p as f64).sqrt() as usize;
    let w = if h > 0 { p / h } else { 1 };

    let grid = projected.reshape((n, h, w, d))?;

    let nl = newline_emb
        .unsqueeze(0)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_as((n, h, 1, d))?
        .contiguous()?;
    let with_nl = Tensor::cat(&[&grid, &nl], 2)?; // [N, H, W+1, D]
    let flat = with_nl.reshape((n, h * (w + 1), d))?;

    let end = new_emb
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_as((n, 1, d))?
        .contiguous()?;
    Tensor::cat(&[&flat, &end], 1) // [N, H*(W+1)+1, D]
}

// ─── Merge helper ─────────────────────────────────────────────────────────────

/// Replace positions where `input_ids == image_token_id` with rows from
/// `image_embs` in sequence order.
fn merge_image_tokens(
    text_embs: &Tensor,  // [B, S, D]
    image_embs: &Tensor, // [total_img_tokens, D]
    input_ids: &Tensor,  // [B, S]
    image_token_id: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (b, s, _d) = text_embs.dims3()?;
    let ids: Vec<u32> = input_ids.to_dtype(DType::U32)?.reshape(b * s)?.to_vec1()?;
    let img_rows = image_embs.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut merged = text_embs.to_dtype(DType::F32)?.to_vec3::<f32>()?;

    let mut cursor = 0usize;
    for (flat, &id) in ids.iter().enumerate() {
        if id == image_token_id && cursor < img_rows.len() {
            let bi = flat / s;
            let si = flat % s;
            if bi < b {
                merged[bi][si].clone_from(&img_rows[cursor]);
                cursor += 1;
            }
        }
    }
    Tensor::new(merged, device)?.to_dtype(dtype)
}

// ─── TarsierForConditionalGeneration ─────────────────────────────────────────

/// Tarsier VLM — CLIP/SigLIP vision + 2-layer MLP projector + LLM.
///
/// Config extra fields consumed:
/// - `vision_config`: nested JSON with `model_type` + ViT parameters
/// - `text_config`: nested JSON with `architectures`, LLM parameters
/// - `image_token_index`: image placeholder token ID (default 32000)
/// - `projector_hidden_act`: activation in projector (default "gelu")
/// - `image_newline_idx`: token ID embedded as row separator (default 13 = `\n`)
/// - `image_new_idx`: token ID embedded as end-of-image marker (default 32001)
/// - `multimodal_projector_bias`: projector linear bias (default true)
pub struct TarsierForConditionalGeneration {
    #[allow(dead_code)]
    vision_tower: TarsierVisionTower,
    projector: TarsierMultiModalProjector,
    language_model: TarsierLlm,
    image_token_id: u32,
    /// Row-separator embedding looked up from LLM embedding table at init.
    image_newline_emb: Tensor, // [D_t]
    /// End-of-image embedding looked up from LLM embedding table at init.
    image_new_emb: Tensor, // [D_t]
    dtype: DType,
    device: Device,
}

impl TarsierForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let empty = serde_json::Value::Object(Default::default());
        let vision_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(empty.clone());
        let text_json = cfg
            .extra
            .get("text_config")
            .cloned()
            .unwrap_or(empty.clone());

        // Vision tower
        let (vision_tower, vision_hidden) =
            TarsierVisionTower::new(&vision_json, vb.pp("vision_tower"))?;

        // Parse text-backbone arch and build a ModelConfig for it
        let text_arch = text_json
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("LlamaForCausalLM")
            .to_string();

        let mut text_cfg = cfg.clone();
        let field_u = |key: &str| text_json.get(key).and_then(|v| v.as_u64());
        let field_f = |key: &str| text_json.get(key).and_then(|v| v.as_f64());
        if let Some(v) = field_u("hidden_size") {
            text_cfg.hidden_size = v as usize;
        }
        if let Some(v) = field_u("num_attention_heads") {
            text_cfg.num_attention_heads = v as usize;
        }
        if let Some(v) = field_u("num_key_value_heads") {
            text_cfg.num_key_value_heads = v as usize;
        }
        if let Some(v) = field_u("num_hidden_layers") {
            text_cfg.num_hidden_layers = v as usize;
        }
        if let Some(v) = field_u("intermediate_size") {
            text_cfg.intermediate_size = v as usize;
        }
        if let Some(v) = field_u("vocab_size") {
            text_cfg.vocab_size = v as usize;
        }
        if let Some(v) = field_u("max_position_embeddings") {
            text_cfg.max_position_embeddings = v as usize;
        }
        if let Some(v) = field_f("rms_norm_eps") {
            text_cfg.rms_norm_eps = v;
        }
        if let Some(v) = field_f("rope_theta") {
            text_cfg.rope_theta = v;
        }
        text_cfg.architectures = vec![text_arch.clone()];

        let text_hidden = text_cfg.hidden_size;

        let act = cfg
            .extra
            .get("projector_hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string();
        let with_bias = cfg
            .extra
            .get("multimodal_projector_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let image_token_id = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as u32;
        let image_newline_idx = cfg
            .extra
            .get("image_newline_idx")
            .and_then(|v| v.as_u64())
            .unwrap_or(13) as u32;
        let image_new_idx = cfg
            .extra
            .get("image_new_idx")
            .and_then(|v| v.as_u64())
            .unwrap_or(32001) as u32;

        let projector = TarsierMultiModalProjector::new(
            vision_hidden,
            text_hidden,
            &act,
            with_bias,
            vb.pp("multi_modal_projector"),
        )?;

        let language_model = TarsierLlm::new(&text_arch, &text_cfg, vb.pp("language_model"))?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Pre-look up the special-token embeddings from the LLM embedding table.
        let newline_ids = Tensor::new(&[image_newline_idx], &device)?;
        let image_newline_emb = language_model.embed_text(&newline_ids)?.squeeze(0)?;

        let new_ids = Tensor::new(&[image_new_idx], &device)?;
        let image_new_emb = language_model.embed_text(&new_ids)?.squeeze(0)?;

        Ok(Self {
            vision_tower,
            projector,
            language_model,
            image_token_id,
            image_newline_emb,
            image_new_emb,
            dtype,
            device,
        })
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
        let text_embs = self.language_model.embed_text(input_ids)?;

        let embeddings = match multimodal_inputs {
            Some(mm) if mm.has_images() => {
                // Collect raw vision encoder outputs from ProcessedImage.embedding
                // [P, D_v] per image; stack → [N, P, D_v].
                let mut img_list: Vec<Tensor> = Vec::new();
                for (_pos, pi) in &mm.image_embeddings {
                    img_list.push(pi.embedding.unsqueeze(0)?);
                }
                if img_list.is_empty() {
                    text_embs
                } else {
                    let stacked = Tensor::cat(&img_list, 0)?; // [N, P, D_v]
                    let projected = self.projector.forward(&stacked)?; // [N, P, D_t]
                    let with_splits = add_tarsier_split_tokens(
                        &projected,
                        &self.image_newline_emb,
                        &self.image_new_emb,
                    )?; // [N, P', D_t]
                    let n = with_splits.dim(0)?;
                    let pp = with_splits.dim(1)?;
                    let d = with_splits.dim(2)?;
                    let flat_img = with_splits.reshape((n * pp, d))?; // [total, D_t]

                    merge_image_tokens(
                        &text_embs,
                        &flat_img,
                        input_ids,
                        self.image_token_id,
                        &self.device,
                        self.dtype,
                    )?
                }
            }
            _ => text_embs,
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

impl ModelForward for TarsierForConditionalGeneration {
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
        let embs = self.language_model.embed_text(input_ids)?;
        self.language_model
            .forward_decode_batch_with_embeddings(&embs, sequences, kv_cache_mgr)
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;
    use serde_json::json;

    fn make_config(vision_model_type: &str, text_arch: &str) -> ModelConfig {
        ModelConfig {
            architectures: vec!["TarsierForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            extra: {
                let mut m = serde_json::Map::new();
                m.insert(
                    "vision_config".to_string(),
                    json!({
                        "model_type": vision_model_type,
                        "hidden_size": 32,
                        "num_attention_heads": 2,
                        "num_hidden_layers": 1,
                        "intermediate_size": 64,
                        "image_size": 8,
                        "patch_size": 4,
                        "num_channels": 3,
                        "layer_norm_eps": 1e-5,
                        "projection_size": 16,
                    }),
                );
                m.insert(
                    "text_config".to_string(),
                    json!({
                        "architectures": [text_arch],
                        "hidden_size": 64,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "num_hidden_layers": 2,
                        "intermediate_size": 128,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "rms_norm_eps": 1e-5,
                        "rope_theta": 10000.0,
                    }),
                );
                m.insert("image_token_index".to_string(), json!(32));
                m.insert("projector_hidden_act".to_string(), json!("gelu"));
                m.insert("image_newline_idx".to_string(), json!(10));
                m.insert("image_new_idx".to_string(), json!(33));
                m.insert("multimodal_projector_bias".to_string(), json!(true));
                m
            },
            ..Default::default()
        }
    }

    fn cpu_vb() -> VarBuilder<'static> {
        let vm = VarMap::new();
        VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)
    }

    #[test]
    fn tarsier_clip_llama_construction() {
        let cfg = make_config("clip_vision_model", "LlamaForCausalLM");
        let m = TarsierForConditionalGeneration::new(&cfg, cpu_vb());
        assert!(m.is_ok(), "CLIP+LLaMA construction failed: {:?}", m.err());
        assert_eq!(m.unwrap().image_token_id, 32);
    }

    #[test]
    fn tarsier_siglip_llama_construction() {
        let cfg = make_config("siglip_vision_model", "LlamaForCausalLM");
        let m = TarsierForConditionalGeneration::new(&cfg, cpu_vb());
        assert!(m.is_ok(), "SigLIP+LLaMA construction failed: {:?}", m.err());
    }

    #[test]
    fn tarsier_clip_qwen2_construction() {
        let cfg = make_config("clip_vision_model", "Qwen2ForCausalLM");
        let m = TarsierForConditionalGeneration::new(&cfg, cpu_vb());
        assert!(m.is_ok(), "CLIP+Qwen2 construction failed: {:?}", m.err());
    }

    #[test]
    fn add_split_tokens_shape() {
        // P=4 patches (2×2 grid): expected output = 2*(2+1) + 1 = 7
        let dev = &Device::Cpu;
        let projected = Tensor::zeros((2, 4, 8), DType::F32, dev).unwrap();
        let newline = Tensor::zeros((8,), DType::F32, dev).unwrap();
        let end = Tensor::zeros((8,), DType::F32, dev).unwrap();
        let out = add_tarsier_split_tokens(&projected, &newline, &end).unwrap();
        assert_eq!(out.dims(), &[2, 7, 8]);
    }

    #[test]
    fn supports_multimodal_true() {
        let cfg = make_config("clip_vision_model", "LlamaForCausalLM");
        let m = TarsierForConditionalGeneration::new(&cfg, cpu_vb()).unwrap();
        assert!(m.supports_multimodal());
    }
}
