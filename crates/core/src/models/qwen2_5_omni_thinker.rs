//! Qwen2.5-Omni Thinker model: audio + vision + text unified model.
//!
//! Architecture (three-tower):
//! - `audio_tower`: `Qwen2AudioEncoder` (same as qwen2_audio.rs)
//!   mel `[B, 128, T]` → `[B, T/4, d_model=1280]`
//! - `visual`: `Qwen25VisionTransformer` (same as qwen2_5_vl.rs)
//!   patches `[np, cps]` + grid `(h, w)` → `[np/(m*m), out_hidden]`
//! - `language_model`: `Qwen2ForCausalLM`
//!
//! Config reads from `model_config.extra`:
//! - `audio_config.*` — Qwen2AudioEncoderConfig fields (d_model, num_mel_bins, …)
//! - `vision_config.*` — Qwen25VLVisionConfig fields (depth, hidden_size, …)
//! - `audio_token_index` (u32, default 151647)
//! - `image_token_id` (u32, default 151655)
//!
//! Weight mapping (HF checkpoint has `thinker.*` prefix; caller must strip it):
//! - `audio_tower.*` → Qwen2AudioEncoder
//! - `visual.*` → Qwen25VisionTransformer
//! - `language_model.model.*` / `language_model.lm_head.*` → Qwen2ForCausalLM
//!
//! Note: The talker branch and token2wav vocoder weights are skipped at load time.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;
use super::qwen2_5_vl::{Qwen25VLVisionConfig, Qwen25VisionTransformer};
use super::qwen2_audio::{Qwen2AudioEncoder, Qwen2AudioEncoderConfig};

// ─── Config ─────────────────────────────────────────────────────────────────

struct Qwen2_5OmniThinkerConfig {
    vision_config: Qwen25VLVisionConfig,
    audio_cfg: Qwen2AudioEncoderConfig,
    audio_token_index: u32,
    image_token_id: u32,
}

impl Qwen2_5OmniThinkerConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        let vision_json = extra.get("vision_config").cloned().unwrap_or_default();
        let vision_config = Qwen25VLVisionConfig::from_json(&vision_json);

        let audio_json = extra.get("audio_config").cloned().unwrap_or_default();
        let audio_cfg = Qwen2AudioEncoderConfig::from_json(&audio_json);

        let audio_token_index = extra
            .get("audio_token_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(151647);

        let image_token_id = extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(151655);

        Self {
            vision_config,
            audio_cfg,
            audio_token_index,
            image_token_id,
        }
    }
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// Qwen2.5-Omni Thinker for multimodal conditional generation.
///
/// Handles text, audio, and image inputs in a unified three-tower architecture.
/// Audio and image features are merged into the text embedding sequence before
/// being passed to the Qwen2 language model.
pub struct Qwen2_5OmniThinkerForConditionalGeneration {
    audio_tower: Qwen2AudioEncoder,
    #[allow(dead_code)]
    visual: Qwen25VisionTransformer,
    language_model: Qwen2ForCausalLM,
    audio_token_index: u32,
    image_token_id: u32,
    #[allow(dead_code)]
    spatial_merge_size: usize,
    #[allow(dead_code)]
    dtype: DType,
    device: Device,
}

impl Qwen2_5OmniThinkerForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let omni_cfg = Qwen2_5OmniThinkerConfig::from_model_config(cfg);

        let audio_tower = Qwen2AudioEncoder::new(&omni_cfg.audio_cfg, vb.pp("audio_tower"))?;

        let spatial_merge_size = omni_cfg.vision_config.spatial_merge_size;
        let visual = Qwen25VisionTransformer::new(&omni_cfg.vision_config, vb.pp("visual"))?;

        let language_model = Qwen2ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            visual,
            language_model,
            audio_token_index: omni_cfg.audio_token_index,
            image_token_id: omni_cfg.image_token_id,
            spatial_merge_size,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Encode mel spectrograms through the audio tower.
    ///
    /// * `mel` — `[B, n_mels, T]`
    ///
    /// Returns `[B, T/4, d_model]`.
    pub fn encode_audio(&self, mel: &Tensor) -> Result<Tensor> {
        self.audio_tower.forward(mel)
    }

    /// Merge audio and image embeddings into text embeddings.
    ///
    /// Replaces audio placeholder tokens with audio features and image
    /// placeholder tokens with visual features (consumed left to right).
    fn merge_multimodal(&self, text_embeds: &Tensor, mm: &MultimodalInputs) -> Result<Tensor> {
        let (b, s, d) = text_embeds.dims3()?;

        let has_audio = mm.has_audio();
        let has_images = mm.has_images();
        if !has_audio && !has_images {
            return Ok(text_embeds.clone());
        }

        // Collect audio clips ordered by position.
        let mut audio_clips: Vec<(usize, Tensor)> = mm
            .audio_embeddings
            .iter()
            .map(|(pos, pa)| (*pos, pa.embedding.clone()))
            .collect();
        audio_clips.sort_by_key(|(pos, _)| *pos);

        // Collect image embeddings ordered by position.
        let mut image_clips: Vec<(usize, Tensor)> = mm
            .image_embeddings
            .iter()
            .map(|(pos, pi)| (*pos, pi.embedding.clone()))
            .collect();
        image_clips.sort_by_key(|(pos, _)| *pos);

        let flat_embeds = text_embeds.reshape((b * s, d))?;
        let token_ids = &mm.token_ids;

        let mut rows: Vec<Tensor> = Vec::with_capacity(b * s);
        let mut audio_clip_idx = 0usize;
        let mut audio_clip_offset = 0usize;
        let mut image_clip_idx = 0usize;
        let mut image_clip_offset = 0usize;

        for (seq_idx, &tok) in token_ids.iter().enumerate() {
            if tok == self.audio_token_index && audio_clip_idx < audio_clips.len() {
                let clip = &audio_clips[audio_clip_idx].1;
                let clip_len = clip.dim(0)?;
                let row = clip.narrow(0, audio_clip_offset, 1)?.squeeze(0)?;
                rows.push(row);
                audio_clip_offset += 1;
                if audio_clip_offset >= clip_len {
                    audio_clip_idx += 1;
                    audio_clip_offset = 0;
                }
            } else if tok == self.image_token_id && image_clip_idx < image_clips.len() {
                let clip = &image_clips[image_clip_idx].1;
                let clip_len = clip.dim(0)?;
                let row = clip.narrow(0, image_clip_offset, 1)?.squeeze(0)?;
                rows.push(row);
                image_clip_offset += 1;
                if image_clip_offset >= clip_len {
                    image_clip_idx += 1;
                    image_clip_offset = 0;
                }
            } else {
                rows.push(flat_embeds.narrow(0, seq_idx, 1)?.squeeze(0)?);
            }
        }

        Tensor::stack(&rows, 0)?.reshape((b, s, d))
    }
}

impl ModelForward for Qwen2_5OmniThinkerForConditionalGeneration {
    fn device(&self) -> &Device {
        &self.device
    }

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
        let text_embeds = self.language_model.embed_text(input_ids)?;

        let embeddings = if let Some(mm) = multimodal_inputs {
            if mm.has_audio() || mm.has_images() {
                self.merge_multimodal(&text_embeds, mm)?
            } else {
                text_embeds
            }
        } else {
            text_embeds
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use serde_json::json;

    use crate::kv_cache::{CacheConfig, KVCacheDtype, KVCacheManager};

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_index".into(), json!(200));
        extra.insert("image_token_id".into(), json!(201));
        extra.insert(
            "audio_config".into(),
            json!({
                "d_model": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "num_mel_bins": 8,
                "max_source_positions": 32,
            }),
        );
        extra.insert(
            "vision_config".into(),
            json!({
                "depth": 2,
                "hidden_size": 32,
                "num_heads": 4,
                "intermediate_size": 64,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "window_size": 28,
                "out_hidden_size": 32,
                "fullatt_block_indexes": [0, 1],
            }),
        );
        ModelConfig {
            architectures: vec!["Qwen2_5OmniThinkerForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 64,
            vocab_size: 256,
            max_position_embeddings: 64,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
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
    fn test_omni_thinker_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2_5OmniThinkerForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen2_5OmniThinkerForConditionalGeneration construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_omni_thinker_audio_encode() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2_5OmniThinkerForConditionalGeneration::new(&cfg, vb).unwrap();

        // [B=1, n_mels=8, T=32] → after conv2 (stride=2) → [1, 16, d_model=16]
        // → after avgpool (k=2,s=2) → [1, 8, 16]
        let mel = Tensor::zeros((1usize, 8usize, 32usize), DType::F32, &device).unwrap();
        let out = model.encode_audio(&mel).unwrap();
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(2).unwrap(), 16); // d_model
    }

    #[test]
    fn test_omni_thinker_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2_5OmniThinkerForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        let seq_len = 4usize;
        let mut bt = crate::kv_cache::BlockTable::new(16);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_omni_thinker_with_audio() {
        use crate::multimodal::{MultimodalInputs, ProcessedAudio};
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2_5OmniThinkerForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        // 2 text tokens + 2 audio tokens (audio_token_id=200 twice)
        let seq_len = 4usize;
        let mut bt = crate::kv_cache::BlockTable::new(16);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        // Sequence: [0, 200, 200, 0]
        let token_ids: Vec<u32> = vec![0, 200, 200, 0];
        // Audio embedding: [2 tokens, hidden=32]
        let audio_emb = Tensor::zeros((2usize, 32usize), DType::F32, &device).unwrap();
        let mm = MultimodalInputs::with_audio(
            token_ids.clone(),
            vec![(1, ProcessedAudio::new(audio_emb, 2))],
        );

        let input_ids = Tensor::new(token_ids, &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let result =
            model.forward_multimodal(&input_ids, Some(&mm), 0, &mut kv_cache, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "forward_multimodal failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_omni_thinker_spatial_merge_size() {
        let cfg = make_cfg();
        let omni_cfg = Qwen2_5OmniThinkerConfig::from_model_config(&cfg);
        assert_eq!(omni_cfg.vision_config.spatial_merge_size, 2);
    }
}
