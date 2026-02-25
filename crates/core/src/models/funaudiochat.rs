//! FunAudioChat: Qwen3 LLM augmented with continuous + discrete audio encoders.
//!
//! ```text
//! continuous path:  mel [B, n_mels, T]
//!   └── FunAudioChatAudioEncoder (Conv1d×2 + N-layer transformer + LN + AvgPool + proj)
//!         → [B, T_out, output_dim]
//!
//! discrete path:    speech_ids [B, L]
//!   └── FunAudioChatDiscreteEncoder (embed → group-avg → output_matching)
//!         → [B, L//group_size, output_dim]
//!
//!   fusion → Qwen3ForCausalLM, audio tokens scattered at <|AUDIO|> positions
//! ```
//!
//! Both encoders are pre-run by the processor; `forward_multimodal` only scatters
//! pre-encoded `ProcessedAudio` embeddings at `audio_token_index` positions.
//!
//! ## Weight paths (HuggingFace format)
//!
//! **Continuous encoder** under `audio_config.*`:
//! - `audio_config.conv1/conv2.*`
//! - `audio_config.embed_positions.positional_embedding` (non-persistent buffer, recomputed)
//! - `audio_config.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}` (k_proj: no bias)
//! - `audio_config.layers.{i}.self_attn_layer_norm.*`
//! - `audio_config.layers.{i}.fc1/fc2.*`
//! - `audio_config.layers.{i}.final_layer_norm.*`
//! - `audio_config.ln_post.*`, `audio_config.proj.*`
//! - `audio_config.audio_bos_eos_token.weight`
//!
//! **Discrete encoder** under `audio_config.*`:
//! - `audio_config.embed_tokens.weight`
//! - `audio_config.output_matching.weight`
//! - `audio_config.continual_output_matching.weight`
//!
//! **Language model**: `language_model.*` → Qwen3ForCausalLM
//!
//! ## HF config layout
//! ```json
//! {
//!   "audio_token_index": 151646,
//!   "audio_config": {
//!     "num_mel_bins": 128, "d_model": 1280, "encoder_layers": 32,
//!     "encoder_attention_heads": 20, "encoder_ffn_dim": 5120,
//!     "max_source_positions": 1500, "output_dim": 3584,
//!     "codebook_size": ..., "group_size": 5, "pad_token_id": ...,
//!     "continuous_features_mode": "replace"
//!   }
//! }
//! ```

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    conv1d, embedding, layer_norm, linear, linear_b, linear_no_bias, Conv1d, Conv1dConfig,
    Embedding, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen3::Qwen3ForCausalLM;
use super::whisper::build_sinusoidal_embeddings_pub;

// ─── Config ──────────────────────────────────────────────────────────────────

struct FunAudioCfg {
    /// Number of mel bins (default 128).
    num_mel_bins: usize,
    d_model: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    max_source_positions: usize,
    /// Audio output dimension (projected), defaults to LLM hidden_size.
    output_dim: usize,
    scale_embedding: bool,
    /// Speech codebook size for the discrete encoder.
    codebook_size: usize,
    /// Group size for downsampling discrete tokens.
    group_size: usize,
    /// Padding token index for the discrete encoder.
    pad_token_id: u32,
    /// Whether continuous features are added ("add") or replaced ("replace") by
    /// discrete features when both are present.
    continuous_features_mode: String,
    /// `<|AUDIO|>` token index used to locate placeholder positions.
    audio_token_index: u32,
}

impl FunAudioCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        let audio_json = extra.get("audio_config").cloned().unwrap_or_default();

        let get_u = |k: &str, default: usize| {
            audio_json
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let output_dim = get_u("output_dim", cfg.hidden_size);

        let codebook_size = get_u("codebook_size", 4096);
        let group_size = get_u("group_size", 5);
        let pad_token_id = audio_json
            .get("pad_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let continuous_features_mode = audio_json
            .get("continuous_features_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("replace")
            .to_string();

        let audio_token_index = extra
            .get("audio_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(151646) as u32;

        Self {
            num_mel_bins: get_u("num_mel_bins", 128),
            d_model: get_u("d_model", 1280),
            encoder_layers: get_u("encoder_layers", 32),
            encoder_attention_heads: get_u("encoder_attention_heads", 20),
            encoder_ffn_dim: get_u("encoder_ffn_dim", 5120),
            max_source_positions: get_u("max_source_positions", 1500),
            output_dim,
            scale_embedding: audio_json
                .get("scale_embedding")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            codebook_size,
            group_size,
            pad_token_id,
            continuous_features_mode,
            audio_token_index,
        }
    }
}

// ─── Continuous encoder attention ────────────────────────────────────────────

/// Multi-head self-attention for the continuous audio tower.
///
/// k_proj has no bias; q/v/out_proj have bias.
#[allow(dead_code)]
struct FunAudioChatSelfAttn {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

#[allow(dead_code)]
impl FunAudioChatSelfAttn {
    fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj: linear_b(d_model, d_model, true, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d_model, d_model, vb.pp("k_proj"))?,
            v_proj: linear_b(d_model, d_model, true, vb.pp("v_proj"))?,
            out_proj: linear_b(d_model, d_model, true, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `[B, T, D]` → `[B, T, D]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)
    }
}

// ─── Continuous encoder layer ─────────────────────────────────────────────────

/// Single FunAudioChat encoder transformer layer (pre-norm, residual connections).
///
/// Layout: pre-norm LN → self-attn → residual → pre-norm LN → fc1 → GELU → fc2 → residual.
#[allow(dead_code)]
struct FunAudioChatEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: FunAudioChatSelfAttn,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl FunAudioChatEncoderLayer {
    fn new(cfg: &FunAudioCfg, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?,
            self_attn: FunAudioChatSelfAttn::new(
                cfg.d_model,
                cfg.encoder_attention_heads,
                vb.pp("self_attn"),
            )?,
            final_layer_norm: layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?,
            fc1: linear_b(cfg.d_model, cfg.encoder_ffn_dim, true, vb.pp("fc1"))?,
            fc2: linear_b(cfg.encoder_ffn_dim, cfg.d_model, true, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.fc1.forward(&xs)?.gelu_erf()?;
        residual + self.fc2.forward(&xs)?
    }
}

// ─── Continuous audio encoder ─────────────────────────────────────────────────

/// Continuous audio tower: Whisper-mel → downsampled hidden states.
///
/// Pipeline: Conv1d(k=3,s=1) → GELU → Conv1d(k=3,s=2) → GELU → +sinusoidal_pe
///           → N encoder layers → LayerNorm → AvgPool1d(k=2,s=2) → linear → `[B, T_out, output_dim]`
#[allow(dead_code)]
struct FunAudioChatAudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// Sinusoidal position embeddings `[max_source_positions, d_model]` (always recomputed).
    embed_positions: Tensor,
    layers: Vec<FunAudioChatEncoderLayer>,
    ln_post: LayerNorm,
    proj: Linear,
    /// BOS/EOS token embedding present in weights; unused during S2T inference.
    audio_bos_eos_token: Embedding,
    embed_scale: f64,
}

#[allow(dead_code)]
impl FunAudioChatAudioEncoder {
    fn new(cfg: &FunAudioCfg, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv1d(
            cfg.num_mel_bins,
            cfg.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = conv1d(
            cfg.d_model,
            cfg.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        // Sinusoidal PE — non-persistent buffer, always recomputed (not in checkpoint).
        let embed_positions =
            build_sinusoidal_embeddings_pub(cfg.max_source_positions, cfg.d_model, vb.device())?
                .to_dtype(vb.dtype())?;

        let layers = (0..cfg.encoder_layers)
            .map(|i| FunAudioChatEncoderLayer::new(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let ln_post = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj = linear(cfg.d_model, cfg.output_dim, vb.pp("proj"))?;
        let audio_bos_eos_token = embedding(2, cfg.output_dim, vb.pp("audio_bos_eos_token"))?;

        let embed_scale = if cfg.scale_embedding {
            (cfg.d_model as f64).sqrt()
        } else {
            1.0
        };

        Ok(Self {
            conv1,
            conv2,
            embed_positions,
            layers,
            ln_post,
            proj,
            audio_bos_eos_token,
            embed_scale,
        })
    }

    /// Encode mel features `[B, n_mels, T]` → `[B, T_out, output_dim]`.
    ///
    /// T_out ≈ T / 4 (conv stride-2 + avg_pool stride-2).
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(mel)?.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = if (self.embed_scale - 1.0).abs() > 1e-6 {
            (xs * self.embed_scale)?
        } else {
            xs
        };
        let t = xs.dim(1)?;
        let pos = self.embed_positions.narrow(0, 0, t)?.unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&pos)?;

        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }

        // AvgPool1d(k=2, s=2) then project.
        let xs = avg_pool1d_k2_s2(&xs)?;
        let xs = self.ln_post.forward(&xs)?;
        self.proj.forward(&xs)
    }
}

/// Average pool `[B, T, C]` → `[B, T//2, C]` with kernel=2, stride=2.
fn avg_pool1d_k2_s2(x: &Tensor) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    let t_even = (t / 2) * 2;
    let x = if t_even < t {
        x.narrow(1, 0, t_even)?
    } else {
        x.clone()
    };
    x.reshape((b, t_even / 2, 2, c))?.mean(2)
}

// ─── Discrete audio encoder ───────────────────────────────────────────────────

/// Discrete audio encoder: speech token IDs → grouped + projected embeddings.
///
/// Pipeline: embed → reshape groups → mean(group_size) → output_matching.
/// When continuous features are provided, fuse via add or replace.
#[allow(dead_code)]
struct FunAudioChatDiscreteEncoder {
    embed_tokens: Embedding,
    output_matching: Linear,
    continual_output_matching: Linear,
    group_size: usize,
    padding_idx: u32,
    continuous_features_mode: String,
    hidden_size: usize,
}

#[allow(dead_code)]
impl FunAudioChatDiscreteEncoder {
    fn new(cfg: &FunAudioCfg, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.codebook_size, cfg.output_dim, vb.pp("embed_tokens"))?;
        let output_matching =
            linear_no_bias(cfg.output_dim, cfg.output_dim, vb.pp("output_matching"))?;
        let continual_output_matching = linear_no_bias(
            cfg.output_dim,
            cfg.output_dim,
            vb.pp("continual_output_matching"),
        )?;

        Ok(Self {
            embed_tokens,
            output_matching,
            continual_output_matching,
            group_size: cfg.group_size,
            padding_idx: cfg.pad_token_id,
            continuous_features_mode: cfg.continuous_features_mode.clone(),
            hidden_size: cfg.output_dim,
        })
    }

    /// Encode discrete speech token IDs.
    ///
    /// * `audio_ids` — `[B, L]` where L is padded to a multiple of `group_size`
    ///
    /// Returns `[B, L // group_size, output_dim]`.
    fn forward(&self, audio_ids: &Tensor) -> Result<Tensor> {
        let (b, l) = audio_ids.dims2()?;
        let g = self.group_size;
        let l_g = l / g;
        // embed → [B, L, D] → reshape to [B, L//g, g, D] → mean → [B, L//g, D]
        let embeds = self.embed_tokens.forward(audio_ids)?; // [B, L, D]
        let grouped = embeds.reshape((b, l_g, g, self.hidden_size))?.mean(2)?; // [B, L//g, D]
        self.output_matching.forward(&grouped)
    }
}

// ─── Top-level model ─────────────────────────────────────────────────────────

/// FunAudioChat for conditional generation.
///
/// Contains both the continuous (mel-based) and discrete (speech-tokenizer-based)
/// audio encoders plus a Qwen3 LLM. In the pre-encoded inference path, both
/// encoders are unused — the processor produces `ProcessedAudio.embedding` which
/// is scattered into text at `<|AUDIO|>` placeholder positions.
pub struct FunAudioChatForConditionalGeneration {
    #[allow(dead_code)]
    continuous_audio_tower: FunAudioChatAudioEncoder,
    #[allow(dead_code)]
    audio_tower: FunAudioChatDiscreteEncoder,
    language_model: Qwen3ForCausalLM,
    audio_token_index: u32,
    device: Device,
}

impl FunAudioChatForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let audio_cfg = FunAudioCfg::from_model_config(cfg);
        let audio_token_index = audio_cfg.audio_token_index;

        let vb_audio = vb.pp("audio_config");
        let continuous_audio_tower = FunAudioChatAudioEncoder::new(&audio_cfg, vb_audio.clone())?;
        let audio_tower = FunAudioChatDiscreteEncoder::new(&audio_cfg, vb_audio)?;
        let language_model = Qwen3ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            continuous_audio_tower,
            audio_tower,
            language_model,
            audio_token_index,
            device: vb.device().clone(),
        })
    }
}

impl ModelForward for FunAudioChatForConditionalGeneration {
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
/// Audio clips are pre-encoded by the processor (`ProcessedAudio.embedding`).
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

    fn make_audio_cfg() -> serde_json::Value {
        json!({
            "num_mel_bins": 8,
            "d_model": 8,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "max_source_positions": 8,
            "output_dim": 8,
            "codebook_size": 16,
            "group_size": 2,
            "pad_token_id": 0,
            "continuous_features_mode": "replace"
        })
    }

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_index".into(), json!(10u32));
        extra.insert("audio_config".into(), make_audio_cfg());
        ModelConfig {
            architectures: vec!["FunAudioChatForConditionalGeneration".to_string()],
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
    fn funaudiochat_new() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _model = FunAudioChatForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn funaudiochat_forward_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = FunAudioChatForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn funaudiochat_decode_batch_shape() {
        use crate::engine::DecodeSequenceMetadata;

        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = FunAudioChatForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn funaudiochat_multimodal_scatter() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = FunAudioChatForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let audio_token_index: u32 = 10;
        // Sequence: [text, audio, audio, text]
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
    fn funaudiochat_discrete_encoder_grouping() {
        // Verify discrete encoder groups tokens by group_size and projects.
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let audio_cfg = FunAudioCfg::from_model_config(&cfg);

        let encoder = FunAudioChatDiscreteEncoder::new(&audio_cfg, vb.pp("audio_config")).unwrap();
        // group_size=2, so 4 tokens → 2 output positions
        let audio_ids = Tensor::zeros((1usize, 4usize), DType::U32, &dev).unwrap();
        let out = encoder.forward(&audio_ids).unwrap();
        // [1, 4//2, output_dim] = [1, 2, 8]
        assert_eq!(out.dims(), &[1, 2, 8]);
    }
}
