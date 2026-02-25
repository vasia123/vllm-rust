//! AudioFlamingo3: Whisper-style audio encoder + 2-layer MLP projector + Qwen2 LLM.
//!
//! ```text
//! mel [B, n_mels, T]
//!   └── AudioFlamingo3Encoder (Conv1d×2 + GELU + sinusoidal pos + N transformer layers
//!                              + AvgPool1d(k=2,s=2) + LayerNorm)
//!         → [B, T/4, audio_hidden]
//!   └── AudioFlamingo3MultiModalProjector (Linear → GELU → Linear)
//!         → [B, T/4, text_hidden]
//! text_ids → Qwen2ForCausalLM
//!   scatter audio embeddings at <sound> / audio token positions
//! ```
//!
//! The encoder is structurally identical to `Qwen2AudioEncoder` (both are
//! Whisper-style Conv1d×2 + transformer layers + AvgPool1d(k=2,s=2) + LayerNorm).
//! AudioFlamingo3 adds a dummy `pos_emb.freqs` buffer present in checkpoints.
//!
//! ## Weight paths (HuggingFace format)
//! - `audio_tower.conv1/conv2.*`
//! - `audio_tower.embed_positions.weight`
//! - `audio_tower.layers.{i}.*` (same layout as Qwen2AudioEncoder)
//! - `audio_tower.layer_norm.*`
//! - `audio_tower.pos_emb.freqs` (dummy, not used in forward)
//! - `multi_modal_projector.linear_1.{weight,bias}`
//! - `multi_modal_projector.linear_2.{weight,bias}`
//! - `language_model.*` → Qwen2ForCausalLM
//!
//! ## HF config layout
//! ```json
//! {
//!   "audio_config": { "hidden_size": 1280, "encoder_layers": 32, ... },
//!   "text_config":  { "hidden_size": 3584, ... },
//!   "projector_bias": true,
//!   "projector_hidden_act": "gelu"
//! }
//! ```

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;
use super::qwen2_audio::{Qwen2AudioEncoder, Qwen2AudioEncoderConfig};

// ─── Config ──────────────────────────────────────────────────────────────────

struct Af3Cfg {
    encoder: Qwen2AudioEncoderConfig,
    audio_hidden: usize,
    text_hidden: usize,
    projector_bias: bool,
    audio_token_index: u32,
}

impl Af3Cfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;
        let audio_json = extra.get("audio_config").cloned().unwrap_or_default();
        let text_json = extra.get("text_config").cloned().unwrap_or_default();

        // AudioFlamingo3 uses "hidden_size" for d_model; Qwen2Audio uses "d_model".
        // Try hidden_size first, then fall back to d_model.
        let encoder_json = {
            let mut v = audio_json.clone();
            if let Some(obj) = v.as_object_mut() {
                if !obj.contains_key("d_model") {
                    if let Some(hs) = obj.get("hidden_size").cloned() {
                        obj.insert("d_model".to_string(), hs);
                    }
                }
            }
            v
        };
        let encoder = Qwen2AudioEncoderConfig::from_json(&encoder_json);

        let audio_hidden = audio_json
            .get("hidden_size")
            .or_else(|| audio_json.get("d_model"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(encoder.d_model);

        let text_hidden = text_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let projector_bias = extra
            .get("projector_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let audio_token_index = extra
            .get("audio_token_id")
            .or_else(|| extra.get("audio_token_index"))
            .and_then(|v| v.as_u64())
            .unwrap_or(151646) as u32;

        Self {
            encoder,
            audio_hidden,
            text_hidden,
            projector_bias,
            audio_token_index,
        }
    }
}

// ─── Encoder wrapper ──────────────────────────────────────────────────────────

/// AudioFlamingo3 audio encoder.
///
/// Structurally identical to `Qwen2AudioEncoder`; adds a dummy `pos_emb.freqs`
/// buffer present in HF checkpoints (unused during forward).
#[allow(dead_code)]
struct AudioFlamingo3Encoder {
    inner: Qwen2AudioEncoder,
}

#[allow(dead_code)]
impl AudioFlamingo3Encoder {
    fn new(cfg: &Af3Cfg, vb: VarBuilder) -> Result<Self> {
        // Load the dummy freqs parameter so weight loading completes cleanly.
        let _ = vb.get((cfg.encoder.num_mel_bins,), "pos_emb.freqs");

        Ok(Self {
            inner: Qwen2AudioEncoder::new(&cfg.encoder, vb)?,
        })
    }

    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        self.inner.forward(mel)
    }
}

// ─── Projector ────────────────────────────────────────────────────────────────

/// 2-layer MLP: Linear → GELU → Linear.
#[allow(dead_code)]
struct AudioFlamingo3Projector {
    linear_1: Linear,
    linear_2: Linear,
}

#[allow(dead_code)]
impl AudioFlamingo3Projector {
    fn new(audio_hidden: usize, text_hidden: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_b(audio_hidden, text_hidden, bias, vb.pp("linear_1"))?,
            linear_2: linear_b(text_hidden, text_hidden, bias, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_2
            .forward(&self.linear_1.forward(xs)?.gelu_erf()?)
    }
}

// ─── Top-level model ─────────────────────────────────────────────────────────

/// AudioFlamingo3 for conditional generation.
///
/// Audio is pre-encoded by the processor; `forward_multimodal` scatters
/// `ProcessedAudio.embedding` tensors at audio placeholder positions.
pub struct AudioFlamingo3ForConditionalGeneration {
    #[allow(dead_code)]
    audio_tower: AudioFlamingo3Encoder,
    #[allow(dead_code)]
    multi_modal_projector: AudioFlamingo3Projector,
    language_model: Qwen2ForCausalLM,
    audio_token_index: u32,
    device: Device,
}

impl AudioFlamingo3ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let af3_cfg = Af3Cfg::from_model_config(cfg);
        let audio_token_index = af3_cfg.audio_token_index;

        let audio_tower = AudioFlamingo3Encoder::new(&af3_cfg, vb.pp("audio_tower"))?;
        let multi_modal_projector = AudioFlamingo3Projector::new(
            af3_cfg.audio_hidden,
            af3_cfg.text_hidden,
            af3_cfg.projector_bias,
            vb.pp("multi_modal_projector"),
        )?;
        let language_model = Qwen2ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            multi_modal_projector,
            language_model,
            audio_token_index,
            device: vb.device().clone(),
        })
    }
}

impl ModelForward for AudioFlamingo3ForConditionalGeneration {
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

    let flat_embeds = text_embeds.reshape((b * s, d))?;
    let token_ids = &mm.token_ids;

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

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_id".into(), json!(10u32));
        extra.insert(
            "audio_config".into(),
            json!({
                "hidden_size": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "num_mel_bins": 8,
                "max_source_positions": 8
            }),
        );
        extra.insert("text_config".into(), json!({ "hidden_size": 8 }));
        extra.insert("projector_bias".into(), json!(true));
        extra.insert("projector_hidden_act".into(), json!("gelu"));
        ModelConfig {
            architectures: vec!["AudioFlamingo3ForConditionalGeneration".to_string()],
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
    fn audioflamingo3_new() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _model = AudioFlamingo3ForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn audioflamingo3_forward_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = AudioFlamingo3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let seq_len = 4usize;
        let mut bt = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &dev).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
    }

    #[test]
    fn audioflamingo3_decode_batch_shape() {
        use crate::engine::DecodeSequenceMetadata;

        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = AudioFlamingo3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
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
    fn audioflamingo3_multimodal_scatter() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = AudioFlamingo3ForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let audio_token_index: u32 = 10;
        let token_ids = vec![1u32, audio_token_index, audio_token_index, 2u32];

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

    #[test]
    fn audioflamingo3_projector_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let af3_cfg = Af3Cfg::from_model_config(&cfg);

        let projector = AudioFlamingo3Projector::new(
            af3_cfg.audio_hidden,
            af3_cfg.text_hidden,
            af3_cfg.projector_bias,
            vb.pp("multi_modal_projector"),
        )
        .unwrap();
        let input = Tensor::zeros((1usize, 4usize, 8usize), DType::F32, &dev).unwrap();
        let out = projector.forward(&input).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8]);
    }
}
