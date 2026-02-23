//! Ultravox audio-language model.
//!
//! Architecture:
//! - `WhisperEncoder` (reused from whisper.rs): mel `[B, n_mels, T]` → `[B, T/2, d_model]`
//! - `StackAudioFrames`: `[B, T', D]` → `[B, T'/stack_factor, D*stack_factor]` (with padding)
//! - `UltravoxFeedForwardProjector` (num_projector_layers == 0):
//!   RmsNorm → Linear → SwiGLU → (optional RmsNorm) → Linear → (optional RmsNorm)
//! - `LlamaForCausalLM` as language backbone (most common Ultravox text model)
//!
//! Config reads from `model_config.extra`:
//! - `audio_config.{d_model, num_hidden_layers, num_attention_heads, max_source_positions,
//!   num_mel_bins, scale_embedding, encoder_ffn_dim, layer_norm_epsilon}` — Whisper params
//! - `stack_factor` (default 8)
//! - `hidden_size` (projector intermediate size == text hidden size)
//! - `projector_act` ("swiglu" | other, default "swiglu")
//! - `projector_ln_mid` (bool, default false; v0.5+ uses true)
//! - `num_projector_layers` (0 = FFProjector, >0 = TransformerProjector; only 0 supported)
//! - `audio_token_index` (u32, default 32000)
//!
//! Weight paths (HuggingFace format):
//! - `audio_tower.*` → WhisperEncoder layers
//! - `multi_modal_projector.{ln_pre,linear_1,ln_mid,ln_post,linear_2}.*`
//! - `language_model.model.*` / `language_model.lm_head.*`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::RmsNorm;
use crate::multimodal::MultimodalInputs;

use super::llama::LlamaForCausalLM;
use super::whisper::{WhisperConfig, WhisperEncoder};

// ─── Config ─────────────────────────────────────────────────────────────────

struct UltravoxConfig {
    audio_d_model: usize,
    stack_factor: usize,
    hidden_size: usize,
    projector_ln_mid: bool,
    num_projector_layers: usize,
    audio_token_index: u32,
    /// Whisper-like config for the audio tower.
    whisper_cfg: WhisperConfig,
}

impl UltravoxConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        let stack_factor = extra
            .get("stack_factor")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(8);

        let hidden_size = extra
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let projector_ln_mid = extra
            .get("projector_ln_mid")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let num_projector_layers = extra
            .get("num_projector_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let audio_token_index = extra
            .get("audio_token_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(32000);

        // Build Whisper config from nested audio_config sub-object.
        let audio_cfg_json = extra.get("audio_config").cloned().unwrap_or_default();
        let whisper_cfg = WhisperConfig::from_model_config_and_json(cfg, &audio_cfg_json);

        UltravoxConfig {
            audio_d_model: whisper_cfg.d_model,
            stack_factor,
            hidden_size,
            projector_ln_mid,
            num_projector_layers,
            audio_token_index,
            whisper_cfg,
        }
    }
}

// ─── WhisperConfig helper ────────────────────────────────────────────────────

impl WhisperConfig {
    /// Build a WhisperConfig from a nested `audio_config` JSON object.
    ///
    /// Falls back to a small default config for tests when the audio_config JSON
    /// does not specify a field.
    pub(crate) fn from_model_config_and_json(
        _outer_cfg: &ModelConfig,
        json: &serde_json::Value,
    ) -> Self {
        let get_usize = |k: &str, d: usize| {
            json.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let get_f64 = |k: &str, d: f64| json.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let get_bool = |k: &str, d: bool| json.get(k).and_then(|v| v.as_bool()).unwrap_or(d);
        Self {
            d_model: get_usize("d_model", get_usize("hidden_size", 384)),
            encoder_layers: get_usize("encoder_layers", get_usize("num_hidden_layers", 12)),
            encoder_attention_heads: get_usize(
                "encoder_attention_heads",
                get_usize("num_attention_heads", 6),
            ),
            encoder_ffn_dim: get_usize("encoder_ffn_dim", 1536),
            num_mel_bins: get_usize("num_mel_bins", 80),
            max_source_positions: get_usize("max_source_positions", 1500),
            scale_embedding: get_bool("scale_embedding", false),
            layer_norm_eps: get_f64("layer_norm_epsilon", 1e-5),
            // Decoder fields are unused by the audio-only encoder in Ultravox.
            decoder_layers: 0,
            decoder_attention_heads: 0,
            decoder_ffn_dim: 0,
            max_target_positions: 0,
            vocab_size: 0,
            activation_function: "gelu".to_string(),
            decoder_start_token_id: 0,
        }
    }
}

// ─── StackAudioFrames ────────────────────────────────────────────────────────

/// Downsample audio sequence by grouping frames: `[B, T, D]` → `[B, T', D*k]`.
///
/// Pads T to the nearest multiple of `stack_factor` before reshaping.
fn stack_audio_frames(x: &Tensor, stack_factor: usize) -> Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    let t_pad = t.div_ceil(stack_factor) * stack_factor;
    let x = if t_pad > t {
        let pad_len = t_pad - t;
        let pad = Tensor::zeros((b, pad_len, d), x.dtype(), x.device())?;
        Tensor::cat(&[x, &pad], 1)?
    } else {
        x.clone()
    };
    // [B, T_pad, D] → [B, T_pad/k, D*k]
    x.reshape((b, t_pad / stack_factor, d * stack_factor))
}

// ─── FeedForward Projector ───────────────────────────────────────────────────

/// Projector used by all known public Ultravox checkpoints (num_projector_layers == 0).
///
/// Pipeline:
///   stack_audio_frames → ln_pre → linear_1 → SwiGLU
///   → (ln_mid or Identity) → linear_2 → (ln_post or Identity)
struct UltravoxFeedForwardProjector {
    ln_pre: RmsNorm,
    linear_1: Linear,
    ln_mid: Option<RmsNorm>,
    linear_2: Linear,
    ln_post: Option<RmsNorm>,
    stack_factor: usize,
    /// dim_mid = hidden_size / 2 (SwiGLU halves the dimension)
    #[allow(dead_code)]
    dim_mid: usize,
}

impl UltravoxFeedForwardProjector {
    fn new(
        dim_in: usize,
        hidden_size: usize,
        text_hidden_size: usize,
        projector_ln_mid: bool,
        stack_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim_mid = hidden_size / 2; // SwiGLU halves: linear_1 outputs hidden_size, gate splits → hidden_size/2

        let ln_pre = crate::layers::rms_norm(dim_in, 1e-5, vb.pp("ln_pre"))?;
        let linear_1 = linear_no_bias(dim_in, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = linear_no_bias(dim_mid, text_hidden_size, vb.pp("linear_2"))?;

        // v0.5+: ln after first activation (ln_mid); earlier: ln after second linear (ln_post)
        let (ln_mid, ln_post) = if projector_ln_mid {
            (
                Some(crate::layers::rms_norm(dim_mid, 1e-5, vb.pp("ln_mid"))?),
                None,
            )
        } else {
            (
                None,
                Some(crate::layers::rms_norm(
                    text_hidden_size,
                    1e-5,
                    vb.pp("ln_post"),
                )?),
            )
        };

        Ok(Self {
            ln_pre,
            linear_1,
            ln_mid,
            linear_2,
            ln_post,
            stack_factor,
            dim_mid,
        })
    }

    /// * `audio_features` — `[B, T, d_model]` from WhisperEncoder
    ///
    /// Returns `[B, T', text_hidden_size]`.
    fn forward(&self, audio_features: &Tensor) -> Result<Tensor> {
        let xs = stack_audio_frames(audio_features, self.stack_factor)?;
        let xs = self.ln_pre.forward(&xs)?;
        let xs = self.linear_1.forward(&xs)?;
        // SwiGLU: x[.., :d] * silu(x[.., d:]) where d = dim // 2
        let xs = swiglu(&xs)?;
        let xs = match &self.ln_mid {
            Some(norm) => norm.forward(&xs)?,
            None => xs,
        };
        let xs = self.linear_2.forward(&xs)?;
        match &self.ln_post {
            Some(norm) => norm.forward(&xs),
            None => Ok(xs),
        }
    }
}

/// SwiGLU: splits last dim in half, `x[.., :d] * silu(x[.., d:])`.
fn swiglu(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)? / 2;
    let gate = x.narrow(candle_core::D::Minus1, 0, d)?;
    let up = x.narrow(candle_core::D::Minus1, d, d)?;
    gate.mul(&up.silu()?)
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// Ultravox audio-language model (`UltravoxModel` in HF transformers).
///
/// Implements `ModelForForward` with `supports_multimodal = true`.
/// `forward_multimodal` replaces audio placeholder tokens with projected audio embeddings.
pub struct UltravoxModel {
    audio_tower: WhisperEncoder,
    projector: UltravoxFeedForwardProjector,
    language_model: LlamaForCausalLM,
    audio_token_index: u32,
    #[allow(dead_code)]
    dtype: DType,
    device: Device,
}

impl UltravoxModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ultravox_cfg = UltravoxConfig::from_model_config(cfg);

        if ultravox_cfg.num_projector_layers > 0 {
            candle_core::bail!(
                "UltravoxTransformerProjector (num_projector_layers={}) is not yet \
                 implemented; only FeedForwardProjector (num_projector_layers=0) is supported",
                ultravox_cfg.num_projector_layers
            );
        }

        let audio_tower = WhisperEncoder::new(&ultravox_cfg.whisper_cfg, vb.pp("audio_tower"))?;

        let dim_in = ultravox_cfg.audio_d_model * ultravox_cfg.stack_factor;
        let text_hidden_size = cfg.hidden_size;
        let projector = UltravoxFeedForwardProjector::new(
            dim_in,
            ultravox_cfg.hidden_size,
            text_hidden_size,
            ultravox_cfg.projector_ln_mid,
            ultravox_cfg.stack_factor,
            vb.pp("multi_modal_projector"),
        )?;

        let language_model = LlamaForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            projector,
            language_model,
            audio_token_index: ultravox_cfg.audio_token_index,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Encode audio features through tower + projector.
    ///
    /// * `mel` — `[B, n_mels, T]`
    ///
    /// Returns `[B, T', text_hidden_size]`.
    pub fn encode_audio(&self, mel: &Tensor) -> Result<Tensor> {
        let features = self.audio_tower.forward(mel)?;
        self.projector.forward(&features)
    }

    /// Merge projected audio embeddings into pre-computed text embeddings.
    ///
    /// Replaces token positions equal to `audio_token_index` with rows from
    /// the flattened audio embedding tensor (tokens are consumed left to right).
    fn merge_multimodal(&self, text_embeds: &Tensor, mm: &MultimodalInputs) -> Result<Tensor> {
        // Collect audio features: one Tensor per (placeholder_pos, clip).
        // We need all audio tokens as a flat buffer [total_audio_tokens, D].
        if !mm.has_audio() {
            return Ok(text_embeds.clone());
        }

        let (b, s, d) = text_embeds.dims3()?;
        let ids_flat = mm.token_ids.clone();

        // Collect all audio embeddings ordered by position.
        let mut clips: Vec<(usize, Tensor)> = mm
            .audio_embeddings
            .iter()
            .map(|(pos, pa)| (*pos, pa.embedding.clone()))
            .collect();
        clips.sort_by_key(|(pos, _)| *pos);

        // Build merged embedding row-by-row in CPU Vec.
        // audio_iter walks through clips providing rows sequentially.
        let mut audio_row_idx = 0usize;
        let mut audio_clip_offset = 0usize; // offset within current clip
        let mut clip_idx = 0usize;
        let flat_embeds = text_embeds.reshape((b * s, d))?;

        let mut rows: Vec<Tensor> = Vec::with_capacity(b * s);
        for (seq_idx, &tok) in ids_flat.iter().enumerate() {
            if tok == self.audio_token_index && clip_idx < clips.len() {
                let clip_tensor = &clips[clip_idx].1; // [num_tokens, D]
                let clip_len = clip_tensor.dim(0)?;
                let row = clip_tensor.narrow(0, audio_clip_offset, 1)?.squeeze(0)?;
                rows.push(row);
                audio_clip_offset += 1;
                if audio_clip_offset >= clip_len {
                    clip_idx += 1;
                    audio_clip_offset = 0;
                }
            } else {
                rows.push(flat_embeds.narrow(0, seq_idx, 1)?.squeeze(0)?);
            }
            audio_row_idx += 1;
        }
        let _ = audio_row_idx; // only used for bounds tracking

        let merged = Tensor::stack(&rows, 0)?; // [B*S, D]
        merged.reshape((b, s, d))
    }
}

impl ModelForward for UltravoxModel {
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

        // Encode any audio clips provided in the multimodal payload.
        let embeddings = if let Some(mm) = multimodal_inputs {
            if mm.has_audio() {
                // Re-encode: audio_embeddings in mm hold *pre-projected* features or
                // already-projected embeddings stored by the serving layer.
                // Here we trust that mm.audio_embeddings contains projected [tokens, D] tensors.
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

    use crate::config::ModelConfig;
    use crate::kv_cache::{CacheConfig, KVCacheDtype, KVCacheManager};

    fn small_whisper_cfg() -> WhisperConfig {
        WhisperConfig {
            d_model: 16,
            encoder_layers: 1,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 32,
            num_mel_bins: 8,
            max_source_positions: 32,
            scale_embedding: false,
            layer_norm_eps: 1e-5,
            activation_function: "gelu".to_string(),
            // Decoder fields unused in encoder-only usage.
            decoder_layers: 0,
            decoder_attention_heads: 0,
            decoder_ffn_dim: 0,
            max_target_positions: 0,
            vocab_size: 0,
            decoder_start_token_id: 0,
        }
    }

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("stack_factor".into(), json!(4));
        extra.insert("projector_ln_mid".into(), json!(false));
        extra.insert("num_projector_layers".into(), json!(0));
        extra.insert("audio_token_index".into(), json!(100));
        extra.insert(
            "audio_config".into(),
            json!({
                "d_model": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "num_mel_bins": 8,
                "max_source_positions": 32,
            }),
        );
        ModelConfig {
            architectures: vec!["UltravoxModel".to_string()],
            hidden_size: 32, // text model hidden + projector output
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
    fn test_stack_audio_frames_shape() {
        let device = Device::Cpu;
        // T=9 → padded to 16, stack_factor=8: [2, 9, 4] → [2, 2, 32]
        let x = Tensor::zeros((2usize, 9usize, 4usize), DType::F32, &device).unwrap();
        let stacked = stack_audio_frames(&x, 8).unwrap();
        assert_eq!(stacked.dims(), &[2, 2, 32]);
    }

    #[test]
    fn test_stack_audio_frames_exact_multiple() {
        let device = Device::Cpu;
        // T=8 exactly: [1, 8, 4] → [1, 1, 32]
        let x = Tensor::zeros((1usize, 8usize, 4usize), DType::F32, &device).unwrap();
        let stacked = stack_audio_frames(&x, 8).unwrap();
        assert_eq!(stacked.dims(), &[1, 1, 32]);
    }

    #[test]
    fn test_projector_ff_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);

        // dim_in=64 (16*4), hidden_size=32 → dim_mid=16, text_hidden=24
        let proj = UltravoxFeedForwardProjector::new(64, 32, 24, false, 4, vb.pp("proj")).unwrap();

        // Input: [2, 12, 16]; stack_factor=4 → [2, 3, 64] after stack
        let feats = Tensor::zeros((2usize, 12usize, 16usize), dtype, &device).unwrap();
        let out = proj.forward(&feats).unwrap();
        assert_eq!(out.dims(), &[2, 3, 24]);
    }

    #[test]
    fn test_audio_encoder_shape() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::zeros(dtype, &device);
        let cfg = small_whisper_cfg();

        let encoder = WhisperEncoder::new(&cfg, vb.pp("audio_tower")).unwrap();
        // [B=1, n_mels=8, T=16] → conv2 stride=2 → [1, 8, 16]
        let mel = Tensor::zeros((1usize, 8usize, 16usize), dtype, &device).unwrap();
        let out = encoder.forward(&mel).unwrap();
        assert_eq!(out.dims(), &[1, 8, 16]); // [B, T/2, d_model]
    }

    #[test]
    fn test_ultravox_text_only() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(dtype, &device);
        let model = UltravoxModel::new(&cfg, vb).unwrap();

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
}
