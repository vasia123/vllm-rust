//! Whisper encoder-decoder model for automatic speech recognition.
//!
//! Architecture:
//! - Encoder: Conv1d×2 (stride=2) + sinusoidal pos embed + N self-attention layers + LayerNorm
//! - Decoder: token embed + learned pos embed + N decoder layers + LayerNorm
//! - Decoder layers: pre-norm self-attn + pre-norm cross-attn + pre-norm FFN
//! - Cross-attention: Q from decoder hidden states, K/V from encoder output
//!
//! Weight paths follow HuggingFace Whisper format:
//! - `model.encoder.*`, `model.decoder.*`, `proj_out.weight`
//! - fc1/fc2 are directly under `layers.{i}.*` (no `mlp.*` prefix)
//! - k_proj has no bias; q_proj/v_proj/out_proj have bias
//!
//! Implements [`ModelForEncoderDecoder`]. The `encode()` method accepts mel
//! spectrogram features `[batch, n_mels, time_frames]` as the input tensor.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    conv1d, embedding, layer_norm, linear_b, linear_no_bias, Conv1d, Conv1dConfig, Embedding,
    LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{EncoderOutput, ModelForEncoderDecoder};
use crate::kv_cache::{BlockTable, KVCacheManager};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Whisper model configuration, read from HuggingFace `config.json`.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Transformer model dimension.
    pub d_model: usize,
    /// Number of encoder transformer layers.
    pub encoder_layers: usize,
    /// Number of decoder transformer layers.
    pub decoder_layers: usize,
    /// Number of attention heads in the encoder.
    pub encoder_attention_heads: usize,
    /// Number of attention heads in the decoder.
    pub decoder_attention_heads: usize,
    /// FFN hidden dimension in the encoder.
    pub encoder_ffn_dim: usize,
    /// FFN hidden dimension in the decoder.
    pub decoder_ffn_dim: usize,
    /// Number of mel frequency bins (typically 80 or 128).
    pub num_mel_bins: usize,
    /// Maximum encoder sequence length (number of audio frames after subsampling).
    pub max_source_positions: usize,
    /// Maximum decoder sequence length (number of output tokens).
    pub max_target_positions: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Activation function name (typically "gelu").
    pub activation_function: String,
    /// Token ID that starts decoder generation (e.g., `<|startoftranscript|>`).
    pub decoder_start_token_id: u32,
    /// Whether to scale embeddings by sqrt(d_model).
    pub scale_embedding: bool,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
}

impl WhisperConfig {
    /// Head dimension for encoder attention.
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Head dimension for decoder attention.
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// Build from a generic `ModelConfig`.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;
        let get_usize = |key: &str, default: usize| -> usize {
            extra
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let d_model = get_usize("d_model", 512);
        Self {
            d_model,
            encoder_layers: get_usize("encoder_layers", 6),
            decoder_layers: get_usize("decoder_layers", 6),
            encoder_attention_heads: get_usize("encoder_attention_heads", 8),
            decoder_attention_heads: get_usize("decoder_attention_heads", 8),
            encoder_ffn_dim: get_usize("encoder_ffn_dim", 2048),
            decoder_ffn_dim: get_usize("decoder_ffn_dim", 2048),
            num_mel_bins: get_usize("num_mel_bins", 80),
            max_source_positions: get_usize("max_source_positions", 1500),
            max_target_positions: get_usize("max_target_positions", 448),
            vocab_size: cfg.vocab_size,
            activation_function: extra
                .get("activation_function")
                .and_then(|v| v.as_str())
                .unwrap_or("gelu")
                .to_string(),
            decoder_start_token_id: extra
                .get("decoder_start_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(50258) as u32,
            scale_embedding: extra
                .get("scale_embedding")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            layer_norm_eps: extra
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5),
        }
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            encoder_layers: 6,
            decoder_layers: 6,
            encoder_attention_heads: 8,
            decoder_attention_heads: 8,
            encoder_ffn_dim: 2048,
            decoder_ffn_dim: 2048,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            vocab_size: 51865,
            activation_function: "gelu".to_string(),
            decoder_start_token_id: 50258,
            scale_embedding: false,
            layer_norm_eps: 1e-5,
        }
    }
}

// ─── Sinusoidal Position Embeddings ─────────────────────────────────────────

/// Build Whisper-style sinusoidal position embeddings.
///
/// Output layout: `[sin(positions/freqs) | cos(positions/freqs)]` — first half
/// is all sines, second half is all cosines (not interleaved).
///
/// This matches `transformers.models.whisper.modeling_whisper.sinusoids`.
/// Crate-visible alias used by audio models sharing this sinusoidal PE (e.g. Qwen2-Audio).
pub(crate) fn build_sinusoidal_embeddings_pub(
    max_positions: usize,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    build_sinusoidal_embeddings(max_positions, channels, device)
}

fn build_sinusoidal_embeddings(
    max_positions: usize,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    assert_eq!(channels % 2, 0, "channels must be even for sinusoidal PE");
    let half = channels / 2;

    // inv_timescales[i] = 1 / 10000^(i / (half - 1)) for i in 0..half
    // Use (half - 1) as divisor to spread from timescale 1 to 1/10000.
    let denom = if half > 1 { (half - 1) as f64 } else { 1.0_f64 };
    let log_max = 10000.0_f64.ln();

    let inv_timescales: Vec<f32> = (0..half)
        .map(|i| (-(log_max * i as f64 / denom)).exp() as f32)
        .collect();

    // scaled_time[p][i] = p * inv_timescales[i]
    let mut data = vec![0.0f32; max_positions * channels];
    for p in 0..max_positions {
        for (i, &freq) in inv_timescales.iter().enumerate() {
            let angle = p as f32 * freq;
            data[p * channels + i] = angle.sin();
            data[p * channels + half + i] = angle.cos();
        }
    }

    Tensor::from_vec(data, (max_positions, channels), device)
}

// ─── Shared Attention ────────────────────────────────────────────────────────

/// Whisper multi-head attention (encoder self-attention or decoder self-attention).
///
/// All projections have bias except k_proj, following the original implementation.
/// Uses scaled dot-product attention without causal masking (for encoder) or
/// with an explicit mask tensor (for decoder).
struct WhisperSelfAttention {
    q_proj: Linear,
    k_proj: Linear, // no bias
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl WhisperSelfAttention {
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

    /// Forward pass.
    ///
    /// * `x`    — `[B, T, D]`
    /// * `mask` — optional causal mask `[1, 1, T, T]` (−∞ for masked positions)
    ///
    /// Returns `[B, T, D]`.
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self.q_proj.forward(x)?; // [B, T, D]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, heads, T, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        // k transposed: [B, heads, head_dim, T]
        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // [B, heads, T, head_dim] -> [B, T, D]
        let out = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&out)
    }
}

// ─── Cross-Attention ─────────────────────────────────────────────────────────

/// Whisper cross-attention: Q from decoder, K/V from encoder.
///
/// Cross-attention K/V projections use the same weight format as self-attention.
/// The encoder hidden states are projected to K/V each step (cached by caller).
struct WhisperCrossAttention {
    q_proj: Linear,
    k_proj: Linear, // no bias
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl WhisperCrossAttention {
    fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / num_heads;
        // encoder_attn weights under vb
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

    /// Forward pass.
    ///
    /// * `decoder_hidden` — `[B, T_dec, D]`
    /// * `encoder_hidden` — `[B, T_enc, D]`
    ///
    /// Returns `[B, T_dec, D]`.
    fn forward(&self, decoder_hidden: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let (b, t_dec, _d) = decoder_hidden.dims3()?;
        let (_, t_enc, _) = encoder_hidden.dims3()?;

        let q = self.q_proj.forward(decoder_hidden)?;
        let k = self.k_proj.forward(encoder_hidden)?;
        let v = self.v_proj.forward(encoder_hidden)?;

        let q = q
            .reshape((b, t_dec, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t_enc, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t_enc, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t_dec,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&out)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

/// Whisper FFN: fc1 → GELU → fc2.
///
/// Both layers have bias. Weight paths are `fc1.weight/bias` and `fc2.weight/bias`
/// directly under the layer's VarBuilder scope (no `mlp.*` prefix in HF weights).
struct WhisperMlp {
    fc1: Linear,
    fc2: Linear,
}

impl WhisperMlp {
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

// ─── Encoder Layer ───────────────────────────────────────────────────────────

/// Single Whisper encoder layer: pre-norm self-attn + pre-norm FFN.
struct WhisperEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: WhisperSelfAttention,
    final_layer_norm: LayerNorm,
    mlp: WhisperMlp,
}

impl WhisperEncoderLayer {
    fn new(cfg: &WhisperConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            self_attn: WhisperSelfAttention::new(
                cfg.d_model,
                cfg.encoder_attention_heads,
                vb.pp("self_attn"),
            )?,
            final_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            mlp: WhisperMlp::new(cfg.d_model, cfg.encoder_ffn_dim, vb)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Self-attention with pre-norm
        let residual = xs;
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None)?;
        let xs = (residual + xs)?;

        // FFN with pre-norm
        let residual = &xs;
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

/// Single Whisper decoder layer: self-attn + cross-attn + FFN, all pre-norm.
struct WhisperDecoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: WhisperSelfAttention,
    encoder_attn_layer_norm: LayerNorm,
    encoder_attn: WhisperCrossAttention,
    final_layer_norm: LayerNorm,
    mlp: WhisperMlp,
}

impl WhisperDecoderLayer {
    fn new(cfg: &WhisperConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            self_attn: WhisperSelfAttention::new(
                cfg.d_model,
                cfg.decoder_attention_heads,
                vb.pp("self_attn"),
            )?,
            encoder_attn_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("encoder_attn_layer_norm"),
            )?,
            encoder_attn: WhisperCrossAttention::new(
                cfg.d_model,
                cfg.decoder_attention_heads,
                vb.pp("encoder_attn"),
            )?,
            final_layer_norm: layer_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            mlp: WhisperMlp::new(cfg.d_model, cfg.decoder_ffn_dim, vb)?,
        })
    }

    /// Forward pass.
    ///
    /// * `xs`             — `[B, T_dec, D]`
    /// * `encoder_hidden` — `[B, T_enc, D]`
    /// * `mask`           — optional causal mask for self-attention
    fn forward(
        &self,
        xs: &Tensor,
        encoder_hidden: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention
        let residual = xs;
        let normed = self.self_attn_layer_norm.forward(xs)?;
        let sa_out = self.self_attn.forward(&normed, mask)?;
        let xs = (residual + sa_out)?;

        // Cross-attention
        let residual = &xs;
        let normed = self.encoder_attn_layer_norm.forward(&xs)?;
        let ca_out = self.encoder_attn.forward(&normed, encoder_hidden)?;
        let xs = (residual + ca_out)?;

        // FFN
        let residual = &xs;
        let normed = self.final_layer_norm.forward(&xs)?;
        let ff_out = self.mlp.forward(&normed)?;
        residual + ff_out
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// Whisper encoder: mel spectrogram → hidden states.
///
/// Two 1D strided convolutions subsample the time axis by 2, followed by
/// sinusoidal position embeddings and N transformer layers.
struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// Fixed sinusoidal position embeddings `[max_source_positions, d_model]`.
    embed_positions: Tensor,
    layers: Vec<WhisperEncoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
}

impl WhisperEncoder {
    fn new(cfg: &WhisperConfig, vb: VarBuilder) -> Result<Self> {
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

        // Sinusoidal position embeddings — precomputed and fixed.
        let embed_positions =
            build_sinusoidal_embeddings(cfg.max_source_positions, cfg.d_model, vb.device())?
                .to_dtype(vb.dtype())?;

        let layers = (0..cfg.encoder_layers)
            .map(|i| WhisperEncoderLayer::new(cfg, vb.pp(format!("layers.{i}"))))
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

    /// Encode mel spectrogram features.
    ///
    /// * `input_features` — `[B, n_mels, T]`
    ///
    /// Returns `[B, T/2, d_model]`.
    fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // Conv subsampling: [B, n_mels, T] → [B, d_model, T/2]
        let xs = self.conv1.forward(input_features)?.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;

        // Transpose to [B, T/2, d_model] and apply scaling
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = if (self.embed_scale - 1.0).abs() > 1e-6 {
            (xs * self.embed_scale)?
        } else {
            xs
        };

        // Add sinusoidal position embeddings (broadcast over batch)
        let t_out = xs.dim(1)?;
        let pos_emb = self.embed_positions.narrow(0, 0, t_out)?.unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&pos_emb)?;

        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }

        self.layer_norm.forward(&xs)
    }
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

/// Whisper decoder: token IDs → logit-ready hidden states.
///
/// Uses learned position embeddings. Generates causally by accepting
/// all previously generated tokens at each step.
struct WhisperDecoder {
    embed_tokens: Embedding,
    embed_positions: Embedding,
    layers: Vec<WhisperDecoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
}

impl WhisperDecoder {
    fn new(cfg: &WhisperConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;
        let embed_positions = embedding(
            cfg.max_target_positions,
            cfg.d_model,
            vb.pp("embed_positions"),
        )?;

        let layers = (0..cfg.decoder_layers)
            .map(|i| WhisperDecoderLayer::new(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let layer_norm = layer_norm(cfg.d_model, cfg.layer_norm_eps, vb.pp("layer_norm"))?;

        let embed_scale = if cfg.scale_embedding {
            (cfg.d_model as f64).sqrt()
        } else {
            1.0
        };

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            embed_scale,
        })
    }

    /// Forward pass of the decoder.
    ///
    /// * `input_ids`      — `[B, T_dec]` (may be prefix or single new token)
    /// * `encoder_hidden` — `[B, T_enc, d_model]` from the encoder
    /// * `seqlen_offset`  — number of previously decoded tokens (for pos embed index)
    ///
    /// Returns `[B, T_dec, d_model]`.
    fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (_, t_dec) = input_ids.dims2()?;
        let device = input_ids.device();

        // Token embeddings
        let mut xs = self.embed_tokens.forward(input_ids)?;
        if (self.embed_scale - 1.0).abs() > 1e-6 {
            xs = (xs * self.embed_scale)?;
        }

        // Positional embeddings: indices [seqlen_offset, ..., seqlen_offset + t_dec - 1]
        let pos_ids: Vec<u32> = (seqlen_offset as u32..(seqlen_offset + t_dec) as u32).collect();
        let pos_ids = Tensor::from_vec(pos_ids, t_dec, device)?;
        let pos_emb = self.embed_positions.forward(&pos_ids)?.unsqueeze(0)?;
        let mut xs = xs.broadcast_add(&pos_emb)?;

        // Causal mask for self-attention (only needed when processing >1 token)
        let mask = if t_dec > 1 {
            Some(crate::layers::causal_mask(
                t_dec,
                seqlen_offset,
                xs.dtype(),
                device,
            )?)
        } else {
            None
        };

        for layer in &self.layers {
            xs = layer.forward(&xs, encoder_hidden, mask.as_ref())?;
        }

        self.layer_norm.forward(&xs)
    }
}

// ─── Top-level model ─────────────────────────────────────────────────────────

/// Whisper encoder-decoder model for automatic speech recognition.
///
/// Implements [`ModelForEncoderDecoder`]:
/// - `encode()` accepts mel spectrogram `[B, n_mels, T]` as the input tensor.
/// - `decode()` accepts decoder token IDs `[B, T_dec]`.
///
/// The encoder is bidirectional; the decoder is causal with cross-attention to
/// the encoder output, which is computed once and passed in via [`EncoderOutput`].
pub struct WhisperForConditionalGeneration {
    encoder: WhisperEncoder,
    decoder: WhisperDecoder,
    /// Output projection (weight-tied with `embed_tokens`).
    proj_out: Linear,
    decoder_start_token_id: u32,
    device: Device,
}

impl WhisperForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let wcfg = WhisperConfig::from_model_config(cfg);

        let model_vb = vb.pp("model");
        let encoder = WhisperEncoder::new(&wcfg, model_vb.pp("encoder"))?;
        let decoder = WhisperDecoder::new(&wcfg, model_vb.pp("decoder"))?;

        // proj_out is weight-tied with embed_tokens in HF Whisper.
        // Load the embed_tokens weight and reuse it as proj_out (no extra bias).
        // NOTE: In HF the tied weight is stored as `proj_out.weight`. When that
        //   is absent the weight is shared with embed_tokens. We fall back to
        //   the embed_tokens weight to handle both cases.
        let proj_out_weight = vb
            .get((wcfg.vocab_size, wcfg.d_model), "proj_out.weight")
            .or_else(|_| {
                vb.get(
                    (wcfg.vocab_size, wcfg.d_model),
                    "model.decoder.embed_tokens.weight",
                )
            })?;
        let proj_out = Linear::new(proj_out_weight, None);

        Ok(Self {
            encoder,
            decoder,
            proj_out,
            decoder_start_token_id: wcfg.decoder_start_token_id,
            device: vb.device().clone(),
        })
    }
}

impl ModelForEncoderDecoder for WhisperForConditionalGeneration {
    /// Encode mel spectrogram features.
    ///
    /// The `input_ids` parameter carries the mel features `[B, n_mels, T]`.
    /// The attention mask is ignored (Whisper always uses full-length audio frames).
    fn encode(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<EncoderOutput> {
        let hidden = self.encoder.forward(input_ids)?;
        EncoderOutput::new(hidden)
    }

    /// Decode one or more tokens given cached encoder output.
    ///
    /// `seqlen_offset` is the number of tokens already generated, used as the
    /// starting index into the learned decoder position embeddings.
    fn decode(
        &self,
        decoder_input_ids: &Tensor,
        encoder_output: &EncoderOutput,
        seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let hidden = self.decoder.forward(
            decoder_input_ids,
            &encoder_output.hidden_states,
            seqlen_offset,
        )?;
        // Project to vocabulary logits: [B, T_dec, vocab_size]
        hidden.apply(&self.proj_out)
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.decoder_start_token_id
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_source_len(&self) -> usize {
        self.encoder.embed_positions.dim(0).unwrap_or(1500)
    }

    fn max_target_len(&self) -> usize {
        self.decoder
            .embed_positions
            .embeddings()
            .dim(0)
            .unwrap_or(448)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn small_cfg() -> WhisperConfig {
        WhisperConfig {
            d_model: 64,
            encoder_layers: 2,
            decoder_layers: 2,
            encoder_attention_heads: 4,
            decoder_attention_heads: 4,
            encoder_ffn_dim: 128,
            decoder_ffn_dim: 128,
            num_mel_bins: 16,
            max_source_positions: 32,
            max_target_positions: 16,
            vocab_size: 128,
            activation_function: "gelu".to_string(),
            decoder_start_token_id: 1,
            scale_embedding: false,
            layer_norm_eps: 1e-5,
        }
    }

    fn build_model(cfg: &WhisperConfig) -> WhisperForConditionalGeneration {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // Build encoder
        let model_vb = vb.pp("model");
        let encoder = WhisperEncoder::new(cfg, model_vb.pp("encoder")).unwrap();
        let decoder = WhisperDecoder::new(cfg, model_vb.pp("decoder")).unwrap();

        // Tied proj_out weights (use zeros for test)
        let proj_out_weight = Tensor::zeros((cfg.vocab_size, cfg.d_model), DType::F32, &device)
            .expect("zeros tensor");
        let proj_out = Linear::new(proj_out_weight, None);

        WhisperForConditionalGeneration {
            encoder,
            decoder,
            proj_out,
            decoder_start_token_id: cfg.decoder_start_token_id,
            device,
        }
    }

    #[test]
    fn test_sinusoidal_embeddings_shape() {
        let device = Device::Cpu;
        let emb = build_sinusoidal_embeddings(32, 64, &device).unwrap();
        assert_eq!(emb.dims(), &[32, 64]);
    }

    #[test]
    fn test_sinusoidal_first_row_zero_angle() {
        // At position 0, sin(0) = 0 and cos(0) = 1.
        let device = Device::Cpu;
        let emb = build_sinusoidal_embeddings(4, 8, &device).unwrap();
        let row0 = emb.get(0).unwrap().to_vec1::<f32>().unwrap();
        // First half should all be 0 (sin(0)), second half should all be 1 (cos(0))
        for &v in &row0[..4] {
            assert!(v.abs() < 1e-6, "sin(0) should be 0, got {v}");
        }
        for &v in &row0[4..] {
            assert!((v - 1.0).abs() < 1e-6, "cos(0) should be 1, got {v}");
        }
    }

    #[test]
    fn test_encoder_forward_shape() {
        let cfg = small_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let encoder = WhisperEncoder::new(&cfg, vb.pp("model").pp("encoder")).unwrap();

        // mel features: [1, n_mels=16, T=8]
        let features = Tensor::zeros((1usize, 16usize, 8usize), DType::F32, &device).unwrap();
        let out = encoder.forward(&features).unwrap();
        // After conv2 with stride=2: T/2 = 4 frames
        assert_eq!(out.dims(), &[1, 4, cfg.d_model]);
    }

    #[test]
    fn test_decoder_forward_shape() {
        let cfg = small_cfg();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let decoder = WhisperDecoder::new(&cfg, vb.pp("model").pp("decoder")).unwrap();

        let input_ids = Tensor::zeros((1usize, 3usize), DType::U32, &device).unwrap();
        let encoder_hidden =
            Tensor::zeros((1usize, 4usize, cfg.d_model), DType::F32, &device).unwrap();
        let out = decoder.forward(&input_ids, &encoder_hidden, 0).unwrap();
        assert_eq!(out.dims(), &[1, 3, cfg.d_model]);
    }

    #[test]
    fn test_full_forward_shape() {
        let cfg = small_cfg();
        let device = Device::Cpu;
        let model = build_model(&cfg);

        let mut kv = KVCacheManager::new(&crate::kv_cache::config::CacheConfig {
            block_size: 16,
            num_blocks: 16,
            num_layers: cfg.decoder_layers,
            num_kv_heads: cfg.decoder_attention_heads,
            head_dim: cfg.decoder_head_dim(),
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap();

        let mut bt = BlockTable::new(16);
        kv.allocate_for_request(&mut bt, 4).unwrap();
        let slot_mapping = bt.slot_mapping(0, 4);

        // Encode: mel features [1, 16, 8]
        let features = Tensor::zeros((1usize, 16usize, 8usize), DType::F32, &device).unwrap();
        let encoder_output = model.encode(&features, None).unwrap();
        assert_eq!(encoder_output.src_len, 4); // T/2

        // Decode: 4-token prefix
        let decoder_ids = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let logits = model
            .decode(
                &decoder_ids,
                &encoder_output,
                0,
                &mut kv,
                &bt,
                &slot_mapping,
            )
            .unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }
}
