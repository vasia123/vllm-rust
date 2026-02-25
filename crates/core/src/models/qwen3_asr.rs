//! Qwen3-ASR: audio speech recognition model.
//!
//! Architecture: audio-only tower paired with a Qwen3 language model.
//!
//! ```text
//! mel [B, n_mels, T]
//!   └── Qwen3OmniMoeAudioEncoder (Conv2d×3 + transformer + proj)
//!         → [B, T_out, output_dim]
//! text_ids → Qwen3ForCausalLM (embed)
//!   scatter audio embeddings at <|audio_pad|> positions
//!   → Qwen3 transformer → logits
//! ```
//!
//! The `Qwen3OmniMoeAudioEncoder` is shared with Qwen3-Omni-MoE Thinker.
//! Audio embeddings are pre-encoded by the processor (not at model forward time).
//!
//! ## Weight paths (after stripping `thinker.` prefix)
//! - `audio_tower.*` → `Qwen3OmniMoeAudioEncoder`
//! - `language_model.*` → `Qwen3ForCausalLM`
//!   (`thinker.model.*` → `language_model.model.*`,
//!   `thinker.lm_head.*` → `language_model.lm_head.*`)
//!
//! ## HF config layout
//! ```json
//! {
//!   "thinker_config": {
//!     "audio_config": { ... },
//!     "text_config": { ... }
//!   },
//!   "audio_token_index": 151647
//! }
//! ```

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen3::Qwen3ForCausalLM;
use super::qwen3_omni_moe_thinker::{Qwen3OmniAudioCfg, Qwen3OmniMoeAudioEncoder};

// ─── Config ──────────────────────────────────────────────────────────────────

struct Qwen3AsrCfg {
    audio_cfg: Qwen3OmniAudioCfg,
    audio_token_index: u32,
}

impl Qwen3AsrCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        // Audio config may live under thinker_config.audio_config or directly.
        let audio_cfg_json = extra
            .get("thinker_config")
            .and_then(|v| v.get("audio_config"))
            .cloned()
            .or_else(|| extra.get("audio_config").cloned())
            .unwrap_or_default();
        let audio_cfg = Qwen3OmniAudioCfg::from_json(&audio_cfg_json);

        let audio_token_index = extra
            .get("audio_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(151647) as u32;

        Self {
            audio_cfg,
            audio_token_index,
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Qwen3-ASR for conditional generation.
///
/// Audio-only model: Qwen3-Omni audio encoder + Qwen3 LLM.
pub struct Qwen3ASRForConditionalGeneration {
    #[allow(dead_code)]
    audio_tower: Qwen3OmniMoeAudioEncoder,
    language_model: Qwen3ForCausalLM,
    audio_token_index: u32,
    device: Device,
}

impl Qwen3ASRForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let asr_cfg = Qwen3AsrCfg::from_model_config(cfg);

        let audio_tower = Qwen3OmniMoeAudioEncoder::new(&asr_cfg.audio_cfg, vb.pp("audio_tower"))?;
        let language_model = Qwen3ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            language_model,
            audio_token_index: asr_cfg.audio_token_index,
            device: vb.device().clone(),
        })
    }
}

impl ModelForward for Qwen3ASRForConditionalGeneration {
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
            if mm.has_audio() {
                scatter_audio_into_text(&text_embeds, mm, self.audio_token_index)?
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

// ─── Multimodal scatter ───────────────────────────────────────────────────────

/// Replace `audio_token_index` positions in `text_embeds` with pre-encoded audio embeddings.
///
/// Audio clips are already encoded by the processor (`ProcessedAudio.embedding`).
/// This mirrors the `merge_multimodal` pattern used in Qwen3-Omni-MoE Thinker.
fn scatter_audio_into_text(
    text_embeds: &Tensor,
    mm: &MultimodalInputs,
    audio_token_index: u32,
) -> Result<Tensor> {
    if mm.audio_embeddings.is_empty() {
        return Ok(text_embeds.clone());
    }

    let (b, s, d) = text_embeds.dims3()?;

    let mut audio_clips: Vec<(usize, Tensor)> = mm
        .audio_embeddings
        .iter()
        .map(|(pos, pa)| (*pos, pa.embedding.clone()))
        .collect();
    audio_clips.sort_by_key(|(pos, _)| *pos);

    let token_ids = &mm.token_ids;
    let flat_embeds = text_embeds.reshape((b * s, d))?;

    let mut rows: Vec<Tensor> = Vec::with_capacity(b * s);
    let mut clip_idx = 0usize;
    let mut clip_offset = 0usize;

    for (seq_idx, &tok) in token_ids.iter().enumerate() {
        if tok == audio_token_index && clip_idx < audio_clips.len() {
            let clip = &audio_clips[clip_idx].1;
            let clip_len = clip.dim(0)?;
            rows.push(clip.narrow(0, clip_offset, 1)?.squeeze(0)?);
            clip_offset += 1;
            if clip_offset >= clip_len {
                clip_idx += 1;
                clip_offset = 0;
            }
        } else {
            rows.push(flat_embeds.narrow(0, seq_idx, 1)?.squeeze(0)?);
        }
    }

    Tensor::stack(&rows, 0)?.reshape((b, s, d))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use serde_json::json;

    use crate::config::ModelConfig;
    use crate::kv_cache::{BlockTable, CacheConfig, KVCacheDtype, KVCacheManager};
    use crate::multimodal::ProcessedAudio;

    /// Small config: audio d_model=8, layers=1, heads=2, ffn=16, mels=8, max_pos=8
    /// LM: hidden=8, layers=1, heads=2, kv=2, inter=16, vocab=32
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_index".into(), json!(10u32));
        extra.insert(
            "audio_config".into(),
            json!({
                "d_model": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "num_mel_bins": 8,
                "max_source_positions": 8,
                "downsample_hidden_size": 4,
                "output_dim": 8
            }),
        );
        ModelConfig {
            architectures: vec!["Qwen3ASRForConditionalGeneration".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    fn make_cache(cfg: &ModelConfig, dev: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: dev.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn qwen3_asr_new() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _model = Qwen3ASRForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn qwen3_asr_forward_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Qwen3ASRForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let seq_len = 4usize;
        let mut bt = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &dev).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        // forward returns [1, seq_len, vocab] — last token logits
        assert_eq!(logits.dim(0).unwrap(), 1);
    }

    #[test]
    fn qwen3_asr_thinker_config_nesting() {
        // Ensure config reads audio_config from under thinker_config when present
        let mut extra = serde_json::Map::new();
        extra.insert(
            "thinker_config".into(),
            json!({
                "audio_config": {
                    "d_model": 8,
                    "encoder_layers": 1,
                    "encoder_attention_heads": 2,
                    "encoder_ffn_dim": 16,
                    "num_mel_bins": 8,
                    "max_source_positions": 8,
                    "downsample_hidden_size": 4,
                    "output_dim": 8
                },
                "text_config": {}
            }),
        );
        extra.insert("audio_token_index".into(), json!(10u32));
        let cfg = ModelConfig {
            architectures: vec!["Qwen3ASRForConditionalGeneration".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        };
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _model = Qwen3ASRForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn qwen3_asr_decode_batch_shape() {
        use crate::engine::DecodeSequenceMetadata;

        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Qwen3ASRForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);

        // Pre-allocate two sequences with 4 tokens each
        let mut bt0 = BlockTable::new(16);
        let mut bt1 = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt0, 4).unwrap();
        kv_mgr.allocate_for_request(&mut bt1, 4).unwrap();
        let slot0 = bt0.slot_mapping(4, 1);
        let slot1 = bt1.slot_mapping(4, 1);

        let input_ids = Tensor::zeros((2usize, 1usize), DType::U32, &dev).unwrap();
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 0,
                seqlen_offset: 4,
                block_ids: bt0.block_ids().to_vec(),
                slot_mapping: slot0,
            },
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 4,
                block_ids: bt1.block_ids().to_vec(),
                slot_mapping: slot1,
            },
        ];
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_mgr)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn qwen3_asr_multimodal_scatter() {
        // Verify that audio embeddings from ProcessedAudio are scattered into text at <|audio_pad|> positions
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Qwen3ASRForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);

        // Sequence: [text, audio_pad, audio_pad, text] — 4 tokens total
        let audio_token_index: u32 = 10;
        let token_ids = vec![1u32, audio_token_index, audio_token_index, 2u32];

        // Audio embedding: 2 tokens × hidden=8
        let audio_emb = Tensor::ones((2usize, 8usize), DType::F32, &dev).unwrap();
        let processed = ProcessedAudio::new(audio_emb, 2);
        let mm = MultimodalInputs::with_audio(token_ids.clone(), vec![(1, processed)]);

        let input_ids = Tensor::from_vec(token_ids, (1usize, 4usize), &dev)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();

        let seq_len = 4usize;
        let mut bt = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let logits = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
    }
}
