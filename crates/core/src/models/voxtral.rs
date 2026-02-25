//! Voxtral: Whisper audio encoder + AudioLanguageAdapter + Mistral LLM.
//!
//! ```text
//! mel [B, n_mels, T]
//!   └── VoxtralEncoderModel (WhisperEncoder: Conv1d×2 + N-layer transformer + LayerNorm)
//!         → [B, T/2, d_model]
//!   └── AudioLanguageAdapter (reshape[T/factor, d_model×factor] → w_in → GELU → w_out)
//!         → [T/factor, text_hidden]
//! text_ids → MistralForCausalLM
//!   scatter audio embeddings at audio placeholder positions
//! ```
//!
//! Audio is pre-encoded by the processor; `forward_multimodal` scatters
//! `ProcessedAudio.embedding` at audio token positions.
//!
//! ## Weight paths (HuggingFace format)
//! - `whisper_encoder.whisper_encoder.conv1/conv2.*`
//! - `whisper_encoder.whisper_encoder.layers.{i}.*`
//! - `whisper_encoder.whisper_encoder.layer_norm.*`
//! - `audio_language_adapter.w_in.weight` (no bias)
//! - `audio_language_adapter.w_out.weight` (no bias)
//! - `language_model.*` → MistralForCausalLM
//!
//! ## HF config layout
//! ```json
//! {
//!   "audio_config": {
//!     "d_model": 1280, "encoder_layers": 32, "downsample_factor": 2, ...
//!   },
//!   "text_config": { "hidden_size": 4096, ... }
//! }
//! ```

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::mistral::MistralForCausalLM;
use super::whisper::{WhisperConfig, WhisperEncoder};

// ─── Config ──────────────────────────────────────────────────────────────────

struct VoxtralCfg {
    whisper: WhisperConfig,
    downsample_factor: usize,
    text_hidden: usize,
    audio_token_index: u32,
}

impl VoxtralCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;
        let audio_json = extra.get("audio_config").cloned().unwrap_or_default();
        let text_json = extra.get("text_config").cloned().unwrap_or_default();

        let whisper = WhisperConfig::from_json(&audio_json);

        let downsample_factor = audio_json
            .get("downsample_factor")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let text_hidden = text_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let audio_token_index = extra
            .get("audio_token_id")
            .or_else(|| extra.get("audio_token_index"))
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;

        Self {
            whisper,
            downsample_factor,
            text_hidden,
            audio_token_index,
        }
    }
}

// ─── WhisperConfig JSON ───────────────────────────────────────────────────────

// NOTE: We extend WhisperConfig with a from_json constructor so VoxtralCfg can
// build it from an arbitrary JSON blob without requiring a full ModelConfig.
trait WhisperConfigFromJson {
    fn from_json(json: &serde_json::Value) -> Self;
}

impl WhisperConfigFromJson for WhisperConfig {
    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = WhisperConfig::default();
        let get_u = |k: &str, d: usize| {
            json.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        WhisperConfig {
            d_model: get_u("d_model", defaults.d_model),
            encoder_layers: get_u("encoder_layers", defaults.encoder_layers),
            encoder_attention_heads: get_u(
                "encoder_attention_heads",
                defaults.encoder_attention_heads,
            ),
            encoder_ffn_dim: get_u("encoder_ffn_dim", defaults.encoder_ffn_dim),
            decoder_layers: get_u("decoder_layers", defaults.decoder_layers),
            decoder_attention_heads: get_u(
                "decoder_attention_heads",
                defaults.decoder_attention_heads,
            ),
            decoder_ffn_dim: get_u("decoder_ffn_dim", defaults.decoder_ffn_dim),
            num_mel_bins: get_u("num_mel_bins", defaults.num_mel_bins),
            max_source_positions: get_u("max_source_positions", defaults.max_source_positions),
            max_target_positions: get_u("max_target_positions", defaults.max_target_positions),
            vocab_size: get_u("vocab_size", defaults.vocab_size),
            activation_function: json
                .get("activation_function")
                .and_then(|v| v.as_str())
                .unwrap_or(&defaults.activation_function)
                .to_string(),
            decoder_start_token_id: json
                .get("decoder_start_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(defaults.decoder_start_token_id),
            scale_embedding: json
                .get("scale_embedding")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.scale_embedding),
            layer_norm_eps: json
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps),
        }
    }
}

// ─── Audio language adapter ───────────────────────────────────────────────────

/// Projects downsampled Whisper encoder output to the LLM embedding space.
///
/// Input: `[T/factor, d_model * downsample_factor]`
/// Output: `[T/factor, text_hidden]`
///
/// Weight paths: `w_in.weight`, `w_out.weight` (no bias for either).
#[allow(dead_code)]
struct AudioLanguageAdapter {
    w_in: Linear,
    w_out: Linear,
}

#[allow(dead_code)]
impl AudioLanguageAdapter {
    fn new(audio_dim: usize, text_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_in: linear_no_bias(audio_dim, text_dim, vb.pp("w_in"))?,
            w_out: linear_no_bias(text_dim, text_dim, vb.pp("w_out"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.w_out.forward(&self.w_in.forward(xs)?.gelu_erf()?)
    }
}

// ─── Encoder model ────────────────────────────────────────────────────────────

/// Voxtral encoder: WhisperEncoder with downsampling reshape.
///
/// The encoder output `[T, d_model]` is padded to a multiple of `downsample_factor`,
/// then reshaped to `[T/factor, d_model * factor]` before the adapter projection.
///
/// Weight prefix: `whisper_encoder.*` (nested under top-level `whisper_encoder.*`)
#[allow(dead_code)]
struct VoxtralEncoderModel {
    whisper_encoder: WhisperEncoder,
    downsample_factor: usize,
    d_model: usize,
}

#[allow(dead_code)]
impl VoxtralEncoderModel {
    fn new(cfg: &VoxtralCfg, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            whisper_encoder: WhisperEncoder::new(&cfg.whisper, vb.pp("whisper_encoder"))?,
            downsample_factor: cfg.downsample_factor,
            d_model: cfg.whisper.d_model,
        })
    }

    /// Encode mel features and downsample.
    ///
    /// * `mel` — `[B, n_mels, T]`
    ///
    /// Returns `[T_out/factor, d_model * factor]` (batch=1 per clip, concatenated).
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // WhisperEncoder: [B, n_mels, T] → [B, T/2, d_model]
        let encoded = self.whisper_encoder.forward(mel)?;
        let (_b, t, _d) = encoded.dims3()?;

        // Squeeze batch dim — assume single clip per call
        let encoded = encoded.squeeze(0)?; // [T/2, d_model]

        // Pad T to multiple of downsample_factor
        let factor = self.downsample_factor;
        let t_pad = t.div_ceil(factor) * factor;
        let encoded = if t_pad > t {
            let pad_len = t_pad - t;
            let zeros = Tensor::zeros((pad_len, self.d_model), encoded.dtype(), encoded.device())?;
            Tensor::cat(&[&encoded, &zeros], 0)?
        } else {
            encoded
        };

        // Reshape: [T_pad, d_model] → [T_pad/factor, d_model * factor]
        encoded.reshape((t_pad / factor, self.d_model * factor))
    }
}

// ─── Top-level model ─────────────────────────────────────────────────────────

/// Voxtral for conditional generation.
///
/// Pre-encoded audio embeddings from `ProcessedAudio.embedding` are scattered
/// at audio placeholder positions in `forward_multimodal`.
pub struct VoxtralForConditionalGeneration {
    #[allow(dead_code)]
    whisper_encoder: VoxtralEncoderModel,
    #[allow(dead_code)]
    audio_language_adapter: AudioLanguageAdapter,
    language_model: MistralForCausalLM,
    audio_token_index: u32,
    device: Device,
}

impl VoxtralForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let voxtral_cfg = VoxtralCfg::from_model_config(cfg);
        let audio_token_index = voxtral_cfg.audio_token_index;

        let adapter_in = voxtral_cfg.whisper.d_model * voxtral_cfg.downsample_factor;
        let adapter_out = voxtral_cfg.text_hidden;

        let whisper_encoder = VoxtralEncoderModel::new(&voxtral_cfg, vb.pp("whisper_encoder"))?;
        let audio_language_adapter =
            AudioLanguageAdapter::new(adapter_in, adapter_out, vb.pp("audio_language_adapter"))?;
        let language_model = MistralForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            whisper_encoder,
            audio_language_adapter,
            language_model,
            audio_token_index,
            device: vb.device().clone(),
        })
    }
}

impl ModelForward for VoxtralForConditionalGeneration {
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
                "d_model": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "num_mel_bins": 8,
                "max_source_positions": 8,
                "downsample_factor": 2
            }),
        );
        extra.insert("text_config".into(), json!({ "hidden_size": 8 }));
        ModelConfig {
            architectures: vec!["VoxtralForConditionalGeneration".to_string()],
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
    fn voxtral_new() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _model = VoxtralForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn voxtral_forward_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = VoxtralForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn voxtral_decode_batch_shape() {
        use crate::engine::DecodeSequenceMetadata;

        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = VoxtralForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn voxtral_multimodal_scatter() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = VoxtralForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn voxtral_adapter_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        // downsample_factor=2, d_model=8: adapter_in = 8*2=16, adapter_out=8
        let adapter = AudioLanguageAdapter::new(16, 8, vb.pp("audio_language_adapter")).unwrap();
        let xs = Tensor::zeros((4usize, 16usize), DType::F32, &dev).unwrap();
        let out = adapter.forward(&xs).unwrap();
        assert_eq!(out.dims(), &[4, 8]);
    }
}
