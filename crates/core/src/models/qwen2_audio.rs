//! Qwen2-Audio encoder-decoder model for audio understanding.
//!
//! Architecture:
//! - `Qwen2AudioEncoder`: Whisper-style Conv1d×2 + N encoder layers + AvgPool1d(k=2,s=2)
//!   + LayerNorm. Outputs `[B, 750, d_model]` from `[B, 128, 3000]` mel features.
//! - `Qwen2AudioProjector`: Single linear layer `[d_model → text_hidden_size]`.
//! - `Qwen2AudioForConditionalGeneration`: encoder → projector → Qwen2 LLM.
//!   Implements `ModelForward` with `supports_multimodal = true`.
//!
//! Key difference from standard Whisper encoder:
//! - conv1 has stride=1 (NOT stride=2 as in Whisper); only conv2 has stride=2
//! - After the N transformer layers, an AvgPool1d(kernel=2, stride=2) halves T again
//! - Total subsampling: 3000 → 1500 (conv2) → 750 (avg_pool)
//!
//! Weight paths (HuggingFace format):
//! - `audio_tower.conv1/conv2.*`, `audio_tower.embed_positions.weight`
//! - `audio_tower.layers.{i}.*` (same layout as Whisper encoder layers)
//! - `audio_tower.layer_norm.*`
//! - `multi_modal_projector.linear.{weight,bias}`
//! - `language_model.model.*` / `language_model.lm_head.*`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv1d, layer_norm, linear, linear_b, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm, Linear,
    VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Audio encoder configuration for Qwen2-Audio.
#[derive(Debug, Clone)]
pub struct Qwen2AudioEncoderConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub scale_embedding: bool,
    pub layer_norm_eps: f64,
}

impl Qwen2AudioEncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
}

impl Default for Qwen2AudioEncoderConfig {
    fn default() -> Self {
        Self {
            d_model: 1280,
            encoder_layers: 32,
            encoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            num_mel_bins: 128,
            max_source_positions: 1500,
            scale_embedding: false,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Qwen2AudioEncoderConfig {
    pub(crate) fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();
        let get_usize = |k: &str, d: usize| {
            json.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        Self {
            d_model: get_usize("d_model", defaults.d_model),
            encoder_layers: get_usize("encoder_layers", defaults.encoder_layers),
            encoder_attention_heads: get_usize(
                "encoder_attention_heads",
                defaults.encoder_attention_heads,
            ),
            encoder_ffn_dim: get_usize("encoder_ffn_dim", defaults.encoder_ffn_dim),
            num_mel_bins: get_usize("num_mel_bins", defaults.num_mel_bins),
            max_source_positions: get_usize("max_source_positions", defaults.max_source_positions),
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

// ─── Encoder attention ──────────────────────────────────────────────────────

/// Multi-head self-attention for the audio encoder.
///
/// k_proj has no bias; q/v/out_proj have bias (same as Whisper/HF Qwen2Audio).
struct Qwen2AudioSelfAttention {
    q_proj: Linear,
    k_proj: Linear, // no bias
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Qwen2AudioSelfAttention {
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

    /// Forward: `[B, T, D]` → `[B, T, D]`.
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj.forward(&out)
    }
}

// ─── Encoder MLP ────────────────────────────────────────────────────────────

struct Qwen2AudioMlp {
    fc1: Linear,
    fc2: Linear,
}

impl Qwen2AudioMlp {
    fn new(d_model: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear_b(d_model, ffn_dim, true, vb.pp("fc1"))?,
            fc2: linear_b(ffn_dim, d_model, true, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

// ─── Encoder layer ──────────────────────────────────────────────────────────

struct Qwen2AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: Qwen2AudioSelfAttention,
    final_layer_norm: LayerNorm,
    mlp: Qwen2AudioMlp,
}

impl Qwen2AudioEncoderLayer {
    fn new(cfg: &Qwen2AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            self_attn: Qwen2AudioSelfAttention::new(
                cfg.d_model,
                cfg.encoder_attention_heads,
                vb.pp("self_attn"),
            )?,
            final_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            mlp: Qwen2AudioMlp::new(cfg.d_model, cfg.encoder_ffn_dim, vb)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ─── Encoder ────────────────────────────────────────────────────────────────

/// Qwen2-Audio encoder: mel spectrogram → downsampled hidden states.
///
/// Pipeline: Conv1d(s=1) → GELU → Conv1d(s=2) → GELU → +pos_embed → N layers
///           → AvgPool1d(k=2,s=2) → LayerNorm → `[B, T/4, d_model]`
pub(crate) struct Qwen2AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// Sinusoidal position embeddings `[max_source_positions, d_model]`.
    embed_positions: Tensor,
    layers: Vec<Qwen2AudioEncoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
}

impl Qwen2AudioEncoder {
    pub(crate) fn new(cfg: &Qwen2AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // conv1: stride=1, conv2: stride=2 (unlike Whisper where both are stride 1 and 2)
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

        // Sinusoidal pos embed — try loading from checkpoint (HF stores it as a frozen param).
        // Fall back to recomputing if not in checkpoint.
        let embed_positions = vb
            .get(
                (cfg.max_source_positions, cfg.d_model),
                "embed_positions.weight",
            )
            .or_else(|_| {
                super::whisper::build_sinusoidal_embeddings_pub(
                    cfg.max_source_positions,
                    cfg.d_model,
                    vb.device(),
                )
            })?
            .to_dtype(vb.dtype())?;

        let layers = (0..cfg.encoder_layers)
            .map(|i| Qwen2AudioEncoderLayer::new(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let layer_norm = layer_norm(cfg.d_model, cfg.layer_norm_eps, vb.pp("layer_norm"))?;

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
            layer_norm,
            embed_scale,
        })
    }

    /// Encode mel features.
    ///
    /// * `input_features` — `[B, num_mel_bins, T]` where T is typically 3000
    ///
    /// Returns `[B, T/4, d_model]` (750 tokens for T=3000).
    pub(crate) fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // Conv subsampling: [B, mel, T] → [B, d_model, T/2]
        let xs = self.conv1.forward(input_features)?.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;

        // Transpose to [B, T/2, d_model] and add sinusoidal pos embed
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = if (self.embed_scale - 1.0).abs() > 1e-6 {
            (xs * self.embed_scale)?
        } else {
            xs
        };
        let t_out = xs.dim(1)?;
        let pos_emb = self.embed_positions.narrow(0, 0, t_out)?.unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&pos_emb)?;

        // N encoder layers
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }

        // AvgPool1d(kernel=2, stride=2): [B, T/2, D] → [B, T/4, D]
        xs = avg_pool1d_k2_s2(&xs)?;

        self.layer_norm.forward(&xs)
    }
}

/// 1D average pooling with kernel=2, stride=2.
///
/// Input: `[B, T, C]` → Output: `[B, T//2, C]`.
/// Each output position averages adjacent pairs of input positions.
/// If T is odd, the last input position is dropped (same as PyTorch AvgPool1d).
fn avg_pool1d_k2_s2(x: &Tensor) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    let t_even = (t / 2) * 2;
    // Trim to even length if necessary
    let x = if t_even < t {
        x.narrow(1, 0, t_even)?
    } else {
        x.clone()
    };
    // Reshape to [B, T//2, 2, C] then mean over dim 2
    x.reshape((b, t_even / 2, 2, c))?.mean(2)
}

// ─── Projector ──────────────────────────────────────────────────────────────

/// Single linear projection from audio encoder dimension to LLM hidden size.
struct Qwen2AudioProjector {
    linear: Linear,
}

impl Qwen2AudioProjector {
    fn new(audio_dim: usize, text_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear(audio_dim, text_dim, vb.pp("linear"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

// ─── Top-level model ─────────────────────────────────────────────────────────

/// Qwen2-Audio model for audio understanding.
///
/// Implements [`ModelForward`] with `supports_multimodal = true`. The `encode_audio`
/// method encodes mel features and projects them to the LLM embedding space. The
/// caller stores the result in [`ProcessedAudio`] and passes it via
/// [`MultimodalInputs`] to `forward_multimodal`.
///
/// Audio token placeholder: model-specific (typically `<|AUDIO|>`, token id from config).
pub struct Qwen2AudioForConditionalGeneration {
    audio_tower: Qwen2AudioEncoder,
    projector: Qwen2AudioProjector,
    language_model: Qwen2ForCausalLM,
    dtype: DType,
    device: Device,
}

impl Qwen2AudioForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let audio_cfg = cfg
            .extra
            .get("audio_config")
            .map(Qwen2AudioEncoderConfig::from_json)
            .unwrap_or_default();

        let text_hidden_size = cfg
            .extra
            .get("text_config")
            .and_then(|v| v.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let audio_tower = Qwen2AudioEncoder::new(&audio_cfg, vb.pp("audio_tower"))?;
        let projector = Qwen2AudioProjector::new(
            audio_cfg.d_model,
            text_hidden_size,
            vb.pp("multi_modal_projector"),
        )?;
        let language_model = Qwen2ForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            projector,
            language_model,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Encode mel spectrogram features and project to LLM embedding space.
    ///
    /// * `mel_features` — `[na, num_mel_bins, T]`
    ///
    /// Returns `[na, num_tokens, text_hidden_size]` where `num_tokens = T/4`.
    pub fn encode_audio(&self, mel_features: &Tensor) -> Result<Tensor> {
        let encoder_out = self.audio_tower.forward(mel_features)?; // [na, T/4, d_model]
        self.projector.forward(&encoder_out) // [na, T/4, text_hidden]
    }

    /// Inject pre-projected audio embeddings into text embeddings at placeholder positions.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_audio() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, audio) in &mm_inputs.audio_embeddings {
            let emb_vec: Vec<Vec<f32>> = audio.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in emb_vec.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl ModelForward for Qwen2AudioForConditionalGeneration {
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
            self.merge_multimodal(&text_embeddings, mm)?
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    fn small_audio_cfg() -> Qwen2AudioEncoderConfig {
        Qwen2AudioEncoderConfig {
            d_model: 32,
            encoder_layers: 2,
            encoder_attention_heads: 4,
            encoder_ffn_dim: 64,
            num_mel_bins: 8,
            max_source_positions: 16,
            scale_embedding: false,
            layer_norm_eps: 1e-5,
        }
    }

    fn build_encoder(cfg: &Qwen2AudioEncoderConfig) -> Qwen2AudioEncoder {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        Qwen2AudioEncoder::new(cfg, vb).unwrap()
    }

    #[test]
    fn test_avg_pool1d_k2_s2_shape() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2usize, 8usize, 16usize), DType::F32, &device).unwrap();
        let out = avg_pool1d_k2_s2(&x).unwrap();
        // [B, T, C] = [2, 8, 16] → [2, 4, 16]
        assert_eq!(out.dims(), &[2, 4, 16]);
    }

    #[test]
    fn test_avg_pool1d_odd_length() {
        let device = Device::Cpu;
        // T=9: last element dropped, 9 → 4
        let x = Tensor::zeros((1usize, 9usize, 4usize), DType::F32, &device).unwrap();
        let out = avg_pool1d_k2_s2(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 4]);
    }

    #[test]
    fn test_encoder_output_shape() {
        let cfg = small_audio_cfg();
        // T=8: after conv2 stride=2 → 4, after avg_pool → 2
        let device = Device::Cpu;
        let encoder = build_encoder(&cfg);
        let mel = Tensor::zeros((1usize, cfg.num_mel_bins, 8usize), DType::F32, &device).unwrap();
        let out = encoder.forward(&mel).unwrap();
        // After conv1(s=1): 8→8, conv2(s=2): 8→4, avg_pool: 4→2
        assert_eq!(out.dims(), &[1, 2, cfg.d_model]);
    }

    #[test]
    fn test_projector_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = Qwen2AudioProjector::new(32, 48, vb).unwrap();
        let x = Tensor::zeros((1usize, 4usize, 32usize), DType::F32, &device).unwrap();
        let out = proj.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 48]);
    }

    #[test]
    fn test_full_encode_audio_shape() {
        // Test the full encode_audio path: encoder + projector
        let audio_cfg = small_audio_cfg();
        let text_hidden = 48usize;
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = Qwen2AudioEncoder::new(&audio_cfg, vb.pp("audio_tower")).unwrap();
        let projector = Qwen2AudioProjector::new(
            audio_cfg.d_model,
            text_hidden,
            vb.pp("multi_modal_projector"),
        )
        .unwrap();

        // T=8 mel: after conv2→4, avg_pool→2 tokens
        let mel = Tensor::zeros(
            (2usize, audio_cfg.num_mel_bins, 8usize),
            DType::F32,
            &device,
        )
        .unwrap();
        let enc_out = encoder.forward(&mel).unwrap(); // [2, 2, 32]
        let proj_out = projector.forward(&enc_out).unwrap(); // [2, 2, 48]
        assert_eq!(proj_out.dims(), &[2, 2, text_hidden]);
    }
}
