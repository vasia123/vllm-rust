//! Qwen3-Omni-MoE Thinker model: audio + vision + text unified model.
//!
//! Three-tower architecture:
//! - Audio tower: `Qwen3OmniMoeAudioEncoder`
//!   Conv2d×3 (GELU) → flatten → linear → sinusoidal pos-emb → N transformer layers
//!   → ln_post → proj1 → GELU → proj2
//!   mel [B, n_mels, T] → [B, T_out, output_dim]
//! - Vision tower: `Qwen3OmniVisionTransformer`
//!   ViT with LayerNorm (not RmsNorm) and SiLU MLP (linear_fc1/fc2 names)
//!   patches [np, cps] + grid (h, w) → [np/(m*m), out_hidden_size]
//! - Language model: `Qwen3MoeForCausalLM`
//!
//! Config from `model_config.extra`:
//! - `audio_config.*` — audio encoder fields
//! - `vision_config.*` — vision transformer fields
//! - `audio_token_index` (u32, default 151647)
//! - `image_token_id` (u32, default 151655)
//!
//! Weight prefix: HF checkpoint has `thinker.*`; caller must strip it before loading.
//! - `audio_tower.*` → Qwen3OmniMoeAudioEncoder
//! - `visual.*` → Qwen3OmniVisionTransformer
//! - `language_model.*` → Qwen3MoeForCausalLM

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::RotaryEmbedding;
use crate::multimodal::MultimodalInputs;

use super::qwen3_moe::Qwen3MoeForCausalLM;

// ─── Audio Encoder Config ────────────────────────────────────────────────────

pub(crate) struct Qwen3OmniAudioCfg {
    pub(crate) d_model: usize,
    pub(crate) encoder_layers: usize,
    pub(crate) encoder_attention_heads: usize,
    pub(crate) encoder_ffn_dim: usize,
    pub(crate) num_mel_bins: usize,
    pub(crate) max_source_positions: usize,
    pub(crate) downsample_hidden_size: usize,
    pub(crate) output_dim: usize,
}

impl Default for Qwen3OmniAudioCfg {
    // needed by Qwen3-ASR which reuses this config via pub(crate)
    fn default() -> Self {
        Self {
            d_model: 1536,
            encoder_layers: 32,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            num_mel_bins: 128,
            max_source_positions: 1500,
            downsample_hidden_size: 128,
            output_dim: 3584,
        }
    }
}

impl Qwen3OmniAudioCfg {
    pub(crate) fn from_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
        Self {
            d_model: v
                .get("d_model")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.d_model as u64) as usize,
            encoder_layers: v
                .get("encoder_layers")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.encoder_layers as u64) as usize,
            encoder_attention_heads: v
                .get("encoder_attention_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.encoder_attention_heads as u64)
                as usize,
            encoder_ffn_dim: v
                .get("encoder_ffn_dim")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.encoder_ffn_dim as u64) as usize,
            num_mel_bins: v
                .get("num_mel_bins")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.num_mel_bins as u64) as usize,
            max_source_positions: v
                .get("max_source_positions")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.max_source_positions as u64)
                as usize,
            downsample_hidden_size: v
                .get("downsample_hidden_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.downsample_hidden_size as u64)
                as usize,
            output_dim: v
                .get("output_dim")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.output_dim as u64) as usize,
        }
    }

    /// Frequency dimension after 3 stride-2 Conv2d layers (kernel=3, padding=1).
    ///
    /// Each layer: out = ceil(in / 2).
    fn conv_freq_out(&self) -> usize {
        let f1 = self.num_mel_bins.div_ceil(2);
        let f2 = f1.div_ceil(2);
        f2.div_ceil(2)
    }

    fn conv_out_dim(&self) -> usize {
        self.downsample_hidden_size * self.conv_freq_out()
    }
}

// ─── Vision Config ───────────────────────────────────────────────────────────

struct Qwen3OmniVisionCfg {
    hidden_size: usize,
    num_heads: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    spatial_merge_size: usize,
    depth: usize,
    intermediate_size: usize,
    out_hidden_size: usize,
}

impl Default for Qwen3OmniVisionCfg {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            spatial_merge_size: 2,
            depth: 32,
            intermediate_size: 5120,
            out_hidden_size: 3584,
        }
    }
}

impl Qwen3OmniVisionCfg {
    fn from_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
        Self {
            hidden_size: v
                .get("hidden_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.hidden_size as u64) as usize,
            num_heads: v
                .get("num_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.num_heads as u64) as usize,
            patch_size: v
                .get("patch_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.patch_size as u64) as usize,
            temporal_patch_size: v
                .get("temporal_patch_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.temporal_patch_size as u64) as usize,
            in_channels: v
                .get("in_channels")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.in_channels as u64) as usize,
            spatial_merge_size: v
                .get("spatial_merge_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.spatial_merge_size as u64) as usize,
            depth: v
                .get("depth")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.depth as u64) as usize,
            intermediate_size: v
                .get("intermediate_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.intermediate_size as u64) as usize,
            out_hidden_size: v
                .get("out_hidden_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(d.out_hidden_size as u64) as usize,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn patch_input_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

// ─── Thinker Config ──────────────────────────────────────────────────────────

struct Qwen3OmniThinkerCfg {
    audio_cfg: Qwen3OmniAudioCfg,
    vision_cfg: Qwen3OmniVisionCfg,
    audio_token_index: u32,
    image_token_id: u32,
}

impl Qwen3OmniThinkerCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        let audio_cfg =
            Qwen3OmniAudioCfg::from_json(&extra.get("audio_config").cloned().unwrap_or_default());
        let vision_cfg =
            Qwen3OmniVisionCfg::from_json(&extra.get("vision_config").cloned().unwrap_or_default());

        let audio_token_index = extra
            .get("audio_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(151647) as u32;

        let image_token_id = extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151655) as u32;

        Self {
            audio_cfg,
            vision_cfg,
            audio_token_index,
            image_token_id,
        }
    }
}

// ─── Sinusoidal Position Embedding ───────────────────────────────────────────

/// Precompute sinusoidal position embedding [length, channels].
///
/// First half: sin, second half: cos, log-spaced timescales.
fn sinusoidal_pos_emb(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let half = channels / 2;
    // Avoid division by zero for half=1
    let log_inc = if half > 1 {
        (10000.0_f32).ln() / ((half - 1) as f32)
    } else {
        0.0
    };
    let inv_ts: Vec<f32> = (0..half).map(|i| (-log_inc * i as f32).exp()).collect();

    let mut data = vec![0.0_f32; length * channels];
    for t in 0..length {
        for i in 0..half {
            let x = t as f32 * inv_ts[i];
            data[t * channels + i] = x.sin();
            data[t * channels + half + i] = x.cos();
        }
    }
    Tensor::from_vec(data, (length, channels), device)
}

// ─── Audio Encoder Attention ─────────────────────────────────────────────────

struct Qwen3OmniAudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Qwen3OmniAudioAttention {
    fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, S, d_model]
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [B, S, H, D] → [B, H, S, D]
        let q = q
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?;

        // [B, H, S, D] → [B, S, H*D]
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

// ─── Audio Encoder Layer ─────────────────────────────────────────────────────

struct Qwen3OmniAudioEncoderLayer {
    self_attn: Qwen3OmniAudioAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl Qwen3OmniAudioEncoderLayer {
    fn new(d_model: usize, num_heads: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3OmniAudioAttention::new(d_model, num_heads, vb.pp("self_attn"))?;
        let self_attn_layer_norm =
            candle_nn::layer_norm(d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = candle_nn::linear(d_model, ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(ffn_dim, d_model, vb.pp("fc2"))?;
        let final_layer_norm = candle_nn::layer_norm(d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm + attention residual
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (residual + &x)?;
        // Pre-norm + FFN residual
        let residual = &x;
        let x_normed = self.final_layer_norm.forward(&x)?;
        let x_ffn = self.fc1.forward(&x_normed)?.gelu_erf()?;
        let x_ffn = self.fc2.forward(&x_ffn)?;
        residual + x_ffn
    }
}

// ─── Audio Encoder ───────────────────────────────────────────────────────────

pub(crate) struct Qwen3OmniMoeAudioEncoder {
    conv2d1: candle_nn::Conv2d,
    conv2d2: candle_nn::Conv2d,
    conv2d3: candle_nn::Conv2d,
    conv_out: Linear,
    positional_embedding: Tensor,
    layers: Vec<Qwen3OmniAudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
}

impl Qwen3OmniMoeAudioEncoder {
    pub(crate) fn new(cfg: &Qwen3OmniAudioCfg, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv2d1 =
            candle_nn::conv2d(1, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = candle_nn::conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv2d3 = candle_nn::conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;

        let conv_out =
            candle_nn::linear_no_bias(cfg.conv_out_dim(), cfg.d_model, vb.pp("conv_out"))?;

        let positional_embedding =
            sinusoidal_pos_emb(cfg.max_source_positions, cfg.d_model, vb.device())?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(Qwen3OmniAudioEncoderLayer::new(
                cfg.d_model,
                cfg.encoder_attention_heads,
                cfg.encoder_ffn_dim,
                vb.pp("layers").pp(i),
            )?);
        }

        let ln_post = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1 = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = candle_nn::linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            positional_embedding,
            layers,
            ln_post,
            proj2,
            proj1,
        })
    }

    /// Forward pass for mel spectrograms.
    ///
    /// * `mel` — `[B, num_mel_bins, T]`
    ///
    /// Returns `[B, T_out, output_dim]`.
    pub(crate) fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (b, _n_mels, _t) = mel.dims3()?;

        // [B, 1, n_mels, T] → conv2d×3 → [B, hidden, freq, time]
        let x = mel.unsqueeze(1)?;
        let x = self.conv2d1.forward(&x)?.gelu_erf()?;
        let x = self.conv2d2.forward(&x)?.gelu_erf()?;
        let x = self.conv2d3.forward(&x)?.gelu_erf()?;

        // [B, hidden, freq, time] → [B, time, hidden*freq]
        let (_, c, f, t) = x.dims4()?;
        let x = x.permute((0, 3, 1, 2))?.contiguous()?; // [B, time, hidden, freq]
        let x = x.reshape((b, t, c * f))?;

        // Linear projection to d_model
        let x = self.conv_out.forward(&x)?; // [B, time, d_model]

        // Add positional embedding (slice to actual sequence length)
        let pos_emb =
            self.positional_embedding
                .narrow(0, 0, t.min(self.positional_embedding.dim(0)?))?;
        let pos_emb = pos_emb.unsqueeze(0)?; // [1, time, d_model]
        let x = x.broadcast_add(&pos_emb)?;

        // Transformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Output norm and projection
        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?.gelu_erf()?;
        self.proj2.forward(&x)
    }
}

// ─── Vision Transformer ──────────────────────────────────────────────────────

/// Vision MLP with SiLU activation.
///
/// Weight paths: `mlp.linear_fc1.{weight,bias}`, `mlp.linear_fc2.{weight,bias}`.
struct Qwen3OmniVisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl Qwen3OmniVisionMlp {
    fn new(embed_dim: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(embed_dim, intermediate_size, vb.pp("linear_fc1"))?;
        let fc2 = candle_nn::linear(intermediate_size, embed_dim, vb.pp("linear_fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = candle_nn::ops::silu(&self.fc1.forward(x)?)?;
        self.fc2.forward(&x)
    }
}

/// Vision attention with fused QKV and 2D RoPE.
///
/// Weight paths: `attn.qkv.{weight,bias}`, `attn.proj.{weight,bias}`.
struct Qwen3OmniVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Qwen3OmniVisionAttention {
    fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let qkv = candle_nn::linear(embed_dim, 3 * embed_dim, vb.pp("qkv"))?;
        let proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Forward with optional 2D rotary embeddings.
    ///
    /// * `x` — `[num_tokens, embed_dim]`
    /// * `rotary_emb` — optional `(cos, sin)` each `[num_tokens, rotary_dim/2]`
    fn forward(&self, x: &Tensor, rotary_emb: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (num_tokens, _) = x.dims2()?;

        let qkv = self.qkv.forward(x)?;
        let q_size = self.num_heads * self.head_dim;

        let q = qkv.narrow(1, 0, q_size)?;
        let k = qkv.narrow(1, q_size, q_size)?;
        let v = qkv.narrow(1, 2 * q_size, q_size)?;

        // [S, H*D] → [H, S, D]
        let q = q
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;

        // Apply 2D RoPE (partial rotation on first rotary_dim dimensions)
        let (q, k) = if let Some((cos, sin)) = rotary_emb {
            let rotary_dim = cos.dim(1)? * 2;
            let q_rot = q.narrow(2, 0, rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;
            let k_rot = k.narrow(2, 0, rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;

            let q_rot = q_rot.unsqueeze(0)?;
            let k_rot = k_rot.unsqueeze(0)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, cos, sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, cos, sin)?;
            let q_rot = q_rot.squeeze(0)?;
            let k_rot = k_rot.squeeze(0)?;

            (
                Tensor::cat(&[q_rot, q_pass.contiguous()?], 2)?,
                Tensor::cat(&[k_rot, k_pass.contiguous()?], 2)?,
            )
        } else {
            (q, k)
        };

        // SDPA: [1, H, S, D]
        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let v = v.unsqueeze(0)?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v.contiguous()?)?;

        // [1, H, S, D] → [S, H*D]
        let out = out
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((num_tokens, self.num_heads * self.head_dim))?;

        self.proj.forward(&out)
    }
}

/// Vision block: LayerNorm + attention + LayerNorm + SiLU MLP.
struct Qwen3OmniVisionBlock {
    norm1: LayerNorm,
    attn: Qwen3OmniVisionAttention,
    norm2: LayerNorm,
    mlp: Qwen3OmniVisionMlp,
}

impl Qwen3OmniVisionBlock {
    fn new(cfg: &Qwen3OmniVisionCfg, vb: VarBuilder) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(cfg.hidden_size, 1e-6, vb.pp("norm1"))?;
        let attn = Qwen3OmniVisionAttention::new(cfg.hidden_size, cfg.num_heads, vb.pp("attn"))?;
        let norm2 = candle_nn::layer_norm(cfg.hidden_size, 1e-6, vb.pp("norm2"))?;
        let mlp = Qwen3OmniVisionMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, rotary_emb: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let residual = x;
        let x_attn = self.attn.forward(&self.norm1.forward(x)?, rotary_emb)?;
        let x = (residual + &x_attn)?;
        let residual = &x;
        let x_mlp = self.mlp.forward(&self.norm2.forward(&x)?)?;
        residual + x_mlp
    }
}

/// Patch merger with LayerNorm.
///
/// Weight paths: `merger.ln_q.{weight,bias}`, `merger.mlp.0.{weight,bias}`,
/// `merger.mlp.2.{weight,bias}`.
struct Qwen3OmniPatchMerger {
    ln_q: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    spatial_merge_size: usize,
}

impl Qwen3OmniPatchMerger {
    fn new(
        hidden_size: usize,
        out_hidden_size: usize,
        spatial_merge_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let merger_hidden = hidden_size * spatial_merge_size * spatial_merge_size;
        let ln_q = candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("ln_q"))?;
        let mlp_fc1 = candle_nn::linear(merger_hidden, merger_hidden, vb.pp("mlp").pp("0"))?;
        let mlp_fc2 = candle_nn::linear(merger_hidden, out_hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            ln_q,
            mlp_fc1,
            mlp_fc2,
            spatial_merge_size,
        })
    }

    fn forward(&self, x: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let x = self.ln_q.forward(x)?;
        let embed_dim = x.dim(1)?;
        let m = self.spatial_merge_size;

        // Reshape to interleave spatial merge units
        let x = x.reshape((grid_h / m, m, grid_w / m, m, embed_dim))?;
        let x = x.permute((0, 2, 1, 3, 4))?;
        let num_merged = (grid_h / m) * (grid_w / m);
        let x = x.reshape((num_merged, m * m * embed_dim))?;

        let x = self.mlp_fc1.forward(&x)?.gelu_erf()?;
        self.mlp_fc2.forward(&x)
    }
}

/// Qwen3 Omni vision transformer.
///
/// Same attention as Qwen2.5-VL but LayerNorm norms and `linear_fc1/fc2` MLP.
/// No window attention — uses full attention across all tokens.
struct Qwen3OmniVisionTransformer {
    patch_embed: Linear,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<Qwen3OmniVisionBlock>,
    merger: Qwen3OmniPatchMerger,
}

impl Qwen3OmniVisionTransformer {
    fn new(cfg: &Qwen3OmniVisionCfg, vb: VarBuilder) -> Result<Self> {
        // Patch embed: Conv3D-as-Linear with bias
        let patch_embed = candle_nn::linear(
            cfg.patch_input_dim(),
            cfg.hidden_size,
            vb.pp("patch_embed").pp("proj"),
        )?;

        // RoPE: 50% partial rotation, same as Qwen2.5-VL
        let rotary_emb = RotaryEmbedding::new_partial(
            cfg.head_dim(),
            8192,
            10000.0,
            0.5,
            true,
            DType::F32,
            vb.device(),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Qwen3OmniVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let merger = Qwen3OmniPatchMerger::new(
            cfg.hidden_size,
            cfg.out_hidden_size,
            cfg.spatial_merge_size,
            vb.pp("merger"),
        )?;

        Ok(Self {
            patch_embed,
            rotary_emb,
            blocks,
            merger,
        })
    }

    /// Compute 2D rotary embeddings for a grid of size (grid_h, grid_w).
    fn compute_2d_rotary(
        &self,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let rotary_dim = self.rotary_emb.rotary_dim();
        let half_rotary = rotary_dim / 2;

        let mut h_positions = Vec::with_capacity(grid_h * grid_w);
        let mut w_positions = Vec::with_capacity(grid_h * grid_w);
        for h in 0..grid_h {
            for w in 0..grid_w {
                h_positions.push(h as u32);
                w_positions.push(w as u32);
            }
        }
        let h_pos = Tensor::from_vec(h_positions, (grid_h * grid_w,), device)?;
        let w_pos = Tensor::from_vec(w_positions, (grid_h * grid_w,), device)?;

        let cos_h = self.rotary_emb.cos().index_select(&h_pos, 0)?;
        let sin_h = self.rotary_emb.sin().index_select(&h_pos, 0)?;
        let cos_w = self.rotary_emb.cos().index_select(&w_pos, 0)?;
        let sin_w = self.rotary_emb.sin().index_select(&w_pos, 0)?;

        let cos_h = cos_h.narrow(1, 0, half_rotary)?;
        let sin_h = sin_h.narrow(1, 0, half_rotary)?;
        let cos_w = cos_w.narrow(1, 0, half_rotary)?;
        let sin_w = sin_w.narrow(1, 0, half_rotary)?;

        let cos = Tensor::cat(&[cos_h, cos_w], 1)?;
        let sin = Tensor::cat(&[sin_h, sin_w], 1)?;
        Ok(Some((cos, sin)))
    }

    /// Encode image patches.
    ///
    /// * `patches` — `[num_patches, in_channels * temporal * H * W]`
    /// * `grid_h` — height in patches
    /// * `grid_w` — width in patches
    ///
    /// Returns `[num_patches / (merge*merge), out_hidden_size]`.
    #[allow(dead_code)]
    fn forward(&self, patches: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(patches)?; // [np, hidden]
        let rotary_emb = self.compute_2d_rotary(grid_h, grid_w, patches.device())?;
        let rotary_ref = rotary_emb
            .as_ref()
            .map(|(cos, sin)| (cos as &Tensor, sin as &Tensor));
        for block in &self.blocks {
            x = block.forward(&x, rotary_ref)?;
        }
        self.merger.forward(&x, grid_h, grid_w)
    }
}

// ─── Multimodal Merge ────────────────────────────────────────────────────────

/// Merge pre-encoded audio/image features into text embeddings.
///
/// Replaces audio and image placeholder tokens with the corresponding feature
/// vectors (consumed left-to-right).
fn merge_multimodal(
    text_embeds: &Tensor,
    mm: &MultimodalInputs,
    audio_token_id: u32,
    image_token_id: u32,
) -> Result<Tensor> {
    let (b, s, d) = text_embeds.dims3()?;

    if !mm.has_audio() && !mm.has_images() {
        return Ok(text_embeds.clone());
    }

    let mut audio_clips: Vec<(usize, Tensor)> = mm
        .audio_embeddings
        .iter()
        .map(|(pos, pa)| (*pos, pa.embedding.clone()))
        .collect();
    audio_clips.sort_by_key(|(pos, _)| *pos);

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
        if tok == audio_token_id && audio_clip_idx < audio_clips.len() {
            let clip = &audio_clips[audio_clip_idx].1;
            let clip_len = clip.dim(0)?;
            rows.push(clip.narrow(0, audio_clip_offset, 1)?.squeeze(0)?);
            audio_clip_offset += 1;
            if audio_clip_offset >= clip_len {
                audio_clip_idx += 1;
                audio_clip_offset = 0;
            }
        } else if tok == image_token_id && image_clip_idx < image_clips.len() {
            let clip = &image_clips[image_clip_idx].1;
            let clip_len = clip.dim(0)?;
            rows.push(clip.narrow(0, image_clip_offset, 1)?.squeeze(0)?);
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

// ─── Main Model ──────────────────────────────────────────────────────────────

/// Qwen3-Omni-MoE Thinker for conditional generation.
///
/// Three-tower model with Conv2d-based audio encoder, ViT vision encoder
/// (LayerNorm + SiLU MLP), and Qwen3-MoE language model.
pub struct Qwen3OmniMoeThinkerForConditionalGeneration {
    audio_tower: Qwen3OmniMoeAudioEncoder,
    #[allow(dead_code)]
    visual: Qwen3OmniVisionTransformer,
    language_model: Qwen3MoeForCausalLM,
    audio_token_index: u32,
    image_token_id: u32,
    #[allow(dead_code)]
    dtype: DType,
    device: Device,
}

impl Qwen3OmniMoeThinkerForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let thinker_cfg = Qwen3OmniThinkerCfg::from_model_config(cfg);

        let audio_tower =
            Qwen3OmniMoeAudioEncoder::new(&thinker_cfg.audio_cfg, vb.pp("audio_tower"))?;
        let visual = Qwen3OmniVisionTransformer::new(&thinker_cfg.vision_cfg, vb.pp("visual"))?;
        let language_model = Qwen3MoeForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            visual,
            language_model,
            audio_token_index: thinker_cfg.audio_token_index,
            image_token_id: thinker_cfg.image_token_id,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Encode mel spectrograms through the audio tower.
    ///
    /// * `mel` — `[B, n_mels, T]`
    ///
    /// Returns `[B, T_out, output_dim]`.
    pub fn encode_audio(&self, mel: &Tensor) -> Result<Tensor> {
        self.audio_tower.forward(mel)
    }
}

impl ModelForward for Qwen3OmniMoeThinkerForConditionalGeneration {
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
                merge_multimodal(
                    &text_embeds,
                    mm,
                    self.audio_token_index,
                    self.image_token_id,
                )?
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

    /// Small config for unit tests:
    /// audio: d_model=8, layers=1, heads=2, ffn=16, mels=8, max_pos=16, hidden=4, out=8
    /// vision: hidden=16, heads=2, patch=2, temporal=2, in_ch=1, merge=2, depth=1, inter=32, out=16
    /// lm: hidden=16, layers=1, heads=2, kv=2, inter=32, vocab=64
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_experts".into(), json!(4));
        extra.insert("num_experts_per_tok".into(), json!(2));
        extra.insert("decoder_sparse_step".into(), json!(1));
        extra.insert("moe_intermediate_size".into(), json!(32));
        extra.insert("norm_topk_prob".into(), json!(true));
        extra.insert("audio_token_index".into(), json!(50u32));
        extra.insert("image_token_id".into(), json!(51u32));
        extra.insert(
            "audio_config".into(),
            json!({
                "d_model": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "num_mel_bins": 8,
                "max_source_positions": 16,
                "downsample_hidden_size": 4,
                "output_dim": 16,
            }),
        );
        extra.insert(
            "vision_config".into(),
            json!({
                "hidden_size": 16,
                "num_heads": 2,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 1,
                "spatial_merge_size": 2,
                "depth": 1,
                "intermediate_size": 32,
                "out_hidden_size": 16,
            }),
        );
        ModelConfig {
            architectures: vec!["Qwen3OmniMoeThinkerForConditionalGeneration".to_string()],
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 32,
            vocab_size: 64,
            max_position_embeddings: 64,
            head_dim: 8,
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
    fn test_qwen3_omni_moe_thinker_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3OmniMoeThinkerForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen3OmniMoeThinkerForConditionalGeneration construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen3_omni_moe_audio_encode() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3OmniMoeThinkerForConditionalGeneration::new(&cfg, vb).unwrap();

        // [B=1, n_mels=8, T=16] → conv2d×3 → [1, T_out, output_dim=16]
        let mel = Tensor::zeros((1usize, 8usize, 16usize), DType::F32, &device).unwrap();
        let out = model.encode_audio(&mel).unwrap();
        assert_eq!(out.dim(0).unwrap(), 1, "batch mismatch");
        assert_eq!(out.dim(2).unwrap(), 16, "output_dim mismatch"); // output_dim=16 from cfg
    }

    #[test]
    fn test_qwen3_omni_moe_vision_encode() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3OmniMoeThinkerForConditionalGeneration::new(&cfg, vb).unwrap();

        // patches: [4, 1*2*2*2=8], grid 2×2 → merger with merge_size=2 → [1, 16]
        let patches = Tensor::zeros((4usize, 8usize), DType::F32, &device).unwrap();
        let out = model.visual.forward(&patches, 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, 16]);
    }

    #[test]
    fn test_qwen3_omni_moe_text_only() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3OmniMoeThinkerForConditionalGeneration::new(&cfg, vb).unwrap();
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
    fn test_qwen3_omni_moe_with_audio() {
        use crate::multimodal::{MultimodalInputs, ProcessedAudio};
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3OmniMoeThinkerForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        // Sequence: [0, 50, 50, 0] (2 audio placeholder tokens; audio_token_index=50)
        let seq_len = 4usize;
        let mut bt = crate::kv_cache::BlockTable::new(16);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let token_ids: Vec<u32> = vec![0, 50, 50, 0];
        // Audio embedding: [2 tokens, hidden=16]  (output_dim=16 from cfg)
        let audio_emb = Tensor::zeros((2usize, 16usize), DType::F32, &device).unwrap();
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
}
