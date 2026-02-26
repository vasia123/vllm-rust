#![allow(dead_code)]
//! NemotronParse encoder-decoder model for OCR/document understanding.
//!
//! Architecture:
//! - Encoder: `RadioWithNeck` — RadioModel (RADIO ViT) + Conv1d(1280→1024) + LayerNorm +
//!   Conv2d(1024→1024, kernel=(1,4), stride=(1,4)) + LayerNorm + Linear(3840→1024) + LayerNorm
//! - Decoder: `MBartDecoderNoPos` — pre-norm Bart decoder without positional embeddings,
//!   using WhisperSelfAttention + WhisperCrossAttention per layer
//!
//! Checkpoint layout:
//!   `encoder.model_encoder.radio_model.model.{patch_generator,blocks}.*`
//!   `encoder.{conv1,layer_norm1,conv2,layer_norm2,sum_proj,layer_norm3}.*`
//!   `decoder.{embed_tokens,layernorm_embedding,layers,layer_norm}.*`
//!   `lm_head.weight`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear_b, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{EncoderOutput, ModelForEncoderDecoder};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::models::radio::{RadioModel, RadioVisionConfig};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Config for the RadioWithNeck vision encoder.
#[derive(Debug, Clone)]
struct NemotronParseEncoderCfg {
    radio: RadioVisionConfig,
    /// Output dimension of neck layers. Always 1024 in the published checkpoint.
    neck_dim: usize,
    /// Input size for sum_proj = num_cls_tokens × radio.hidden_size.
    summary_dim: usize,
}

/// Config for the MBartDecoderNoPos text decoder.
#[derive(Debug, Clone)]
struct NemotronParseDecoderCfg {
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    ffn_dim: usize,
    vocab_size: usize,
    /// Whether to multiply embeddings by sqrt(d_model).
    scale_embedding: bool,
    layer_norm_eps: f64,
    decoder_start_token_id: u32,
}

impl NemotronParseDecoderCfg {
    fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    fn embed_scale(&self) -> f64 {
        if self.scale_embedding {
            (self.d_model as f64).sqrt()
        } else {
            1.0
        }
    }
}

/// Full NemotronParse config parsed from `ModelConfig.extra`.
#[derive(Debug, Clone)]
pub struct NemotronParseConfig {
    encoder: NemotronParseEncoderCfg,
    decoder: NemotronParseDecoderCfg,
    image_h: usize,
    image_w: usize,
}

impl NemotronParseConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        // ── image_size ───────────────────────────────────────────────────────
        let (image_h, image_w) = extra
            .get("image_size")
            .and_then(|v| v.as_array())
            .and_then(|a| {
                let h = a.first()?.as_u64()? as usize;
                let w = a.get(1)?.as_u64()? as usize;
                Some((h, w))
            })
            .unwrap_or((2048, 1648));

        // ── encoder / RadioVisionConfig ───────────────────────────────────────
        let enc_v = extra
            .get("encoder")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let enc = &enc_v;

        let get_u = |obj: &serde_json::Value, key: &str, default: usize| -> usize {
            obj.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_b = |obj: &serde_json::Value, key: &str, default: bool| -> bool {
            obj.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
        };
        let get_f = |obj: &serde_json::Value, key: &str, default: f64| -> f64 {
            obj.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };

        let patch_size = get_u(enc, "patch_size", 16);
        let hidden_size = get_u(enc, "hidden_size", 1280);
        let intermediate_size = get_u(enc, "intermediate_size", hidden_size * 4);
        let num_hidden_layers = get_u(enc, "num_hidden_layers", 32);
        let num_attention_heads = get_u(enc, "num_attention_heads", 16);
        let cpe_max_size = get_u(enc, "cpe_max_size", 512);

        // num_cls_tokens: NemotronParse uses 3 teacher CLS tokens by default.
        let num_cls_tokens = get_u(enc, "num_cls_tokens", 3);
        // num_registers: computed from register_multiple if present.
        let num_registers = {
            let reg_mult = get_u(enc, "register_multiple", 0);
            if reg_mult > 0 {
                // Python: num_registers = register_multiple - (num_tokens % register_multiple)
                let rem = num_cls_tokens % reg_mult;
                if rem == 0 {
                    0
                } else {
                    reg_mult - rem
                }
            } else {
                get_u(enc, "num_registers", 0)
            }
        };

        let radio = RadioVisionConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            patch_size,
            image_size: image_h.min(image_w), // square fallback; not actually used for interpolation
            cpe_max_size,
            qkv_bias: get_b(enc, "qkv_bias", false),
            qk_normalization: get_b(enc, "qk_normalization", false),
            layer_norm_eps: get_f(enc, "layer_norm_eps", 1e-6),
            num_cls_tokens,
            num_registers,
        };

        let summary_dim = num_cls_tokens * hidden_size;
        let neck_dim = get_u(enc, "neck_hidden_size", 1024);

        // ── decoder / MBartDecoderNoPos ────────────────────────────────────
        let dec_v = extra
            .get("decoder")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let dec = &dec_v;

        let decoder = NemotronParseDecoderCfg {
            d_model: get_u(dec, "d_model", 1024),
            num_heads: get_u(dec, "decoder_attention_heads", 16),
            num_layers: get_u(dec, "decoder_layers", 12),
            ffn_dim: get_u(dec, "encoder_ffn_dim", 4096),
            vocab_size: get_u(dec, "vocab_size", 50_266),
            scale_embedding: get_b(dec, "scale_embedding", true),
            layer_norm_eps: get_f(dec, "layer_norm_eps", 1e-5),
            decoder_start_token_id: extra
                .get("decoder_start_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0),
        };

        Self {
            encoder: NemotronParseEncoderCfg {
                radio,
                neck_dim,
                summary_dim,
            },
            decoder,
            image_h,
            image_w,
        }
    }

    fn num_vision_tokens(&self) -> usize {
        let h_patches = self.image_h / self.encoder.radio.patch_size;
        let w_patches = self.image_w / self.encoder.radio.patch_size;
        // conv2 stride=(1,4) halves width by 4; +1 for summary token
        h_patches * (w_patches / 4) + 1
    }
}

// ─── RadioWithNeck ───────────────────────────────────────────────────────────

/// RADIO vision encoder with neck projections.
///
/// Neck:
///  1. `Conv1d(hidden→neck_dim, 1)` — per-patch channel projection
///  2. `LayerNorm(neck_dim)`
///  3. `Conv2d(neck_dim→neck_dim, (1,4), stride=(1,4))` — 4× width compression
///  4. `LayerNorm(neck_dim)`
///  5. `Linear(summary_dim→neck_dim)` — project CLS summary
///  6. `LayerNorm(neck_dim)`
///
/// Output: `[B, h_patches * (w_patches/4) + 1, neck_dim]`
struct RadioWithNeck {
    model_encoder: RadioModel,
    /// `[neck_dim, hidden_size, 1]` — Conv1d kernel
    conv1_weight: Tensor,
    /// `[neck_dim]` — Conv1d bias
    conv1_bias: Tensor,
    layer_norm1: LayerNorm,
    /// `[neck_dim, neck_dim, 1, 4]` — Conv2d kernel (bias=False)
    conv2_weight: Tensor,
    layer_norm2: LayerNorm,
    sum_proj: Linear,
    layer_norm3: LayerNorm,
    patch_size: usize,
}

impl RadioWithNeck {
    fn new(enc_cfg: &NemotronParseEncoderCfg, vb: VarBuilder) -> Result<Self> {
        let radio_cfg = &enc_cfg.radio;
        let hidden = radio_cfg.hidden_size;
        let neck = enc_cfg.neck_dim;
        let summary = enc_cfg.summary_dim;

        // RadioModel lives under model_encoder.radio_model.*
        let model_encoder = RadioModel::new(radio_cfg, vb.pp("model_encoder").pp("radio_model"))?;

        // Conv1d(hidden→neck, 1): weight [neck, hidden, 1], bias [neck]
        let conv1_weight = vb.pp("conv1").get((neck, hidden, 1), "weight")?;
        let conv1_bias = vb.pp("conv1").get(neck, "bias")?;

        let layer_norm1 = layer_norm(neck, 1e-6, vb.pp("layer_norm1"))?;

        // Conv2d(neck→neck, (1,4), stride=(1,4), bias=False): weight [neck, neck, 1, 4]
        let conv2_weight = vb.pp("conv2").get((neck, neck, 1, 4), "weight")?;

        let layer_norm2 = layer_norm(neck, 1e-6, vb.pp("layer_norm2"))?;

        // sum_proj: Linear(summary→neck) with bias
        let sum_proj = linear_b(summary, neck, true, vb.pp("sum_proj"))?;

        let layer_norm3 = layer_norm(neck, 1e-6, vb.pp("layer_norm3"))?;

        Ok(Self {
            model_encoder,
            conv1_weight,
            conv1_bias,
            layer_norm1,
            conv2_weight,
            layer_norm2,
            sum_proj,
            layer_norm3,
            patch_size: radio_cfg.patch_size,
        })
    }

    /// Forward: `pixel_values [B, 3, H, W]` → `[B, S'+1, neck_dim]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, _c, img_h, img_w) = pixel_values.dims4()?;
        let h_patches = img_h / self.patch_size;
        let w_patches = img_w / self.patch_size;

        let (summary, features) = self.model_encoder.forward(pixel_values)?;
        // summary: [B, summary_dim], features: [B, h_patches*w_patches, hidden]

        // Conv1d: [B, S, hidden] → [B, S, neck]
        // Permute to [B, hidden, S], apply 1D conv, permute back.

        // [B, hidden, S]
        let x = features.permute((0, 2, 1))?;
        // Weight: [neck, hidden, 1] → apply as conv1d
        let x = x.conv1d(&self.conv1_weight, 0, 1, 1, 1)?;
        // Bias: [neck] → [1, neck, 1]
        let bias = self.conv1_bias.reshape((1, self.conv1_bias.dim(0)?, 1))?;
        let x = x.broadcast_add(&bias)?;
        // [B, neck, S] → [B, S, neck]
        let x = x.permute((0, 2, 1))?;
        let x = self.layer_norm1.forward(&x)?;
        let neck_dim = x.dim(2)?;

        // Rearrange to [B, neck, h_patches, w_patches]
        let x = x
            .reshape((b, h_patches, w_patches, neck_dim))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;

        // Conv2d(neck→neck, (1,4), stride=(1,4)):
        // Since kernel_h=1 and stride_h=1, reshape to [B*h, neck, w] and apply conv1d.
        let conv2_kernel = self.conv2_weight.squeeze(2)?; // [neck, neck, 4]
        let x = x.permute((0, 2, 1, 3))?.contiguous()?; // [B, h, neck, w]
        let x = x.reshape((b * h_patches, neck_dim, w_patches))?;
        let x = x.conv1d(&conv2_kernel, 0, 4, 1, 1)?; // [B*h, neck, w/4]
        let w_out = w_patches / 4;
        let x = x.reshape((b, h_patches, neck_dim, w_out))?;
        let x = x.permute((0, 1, 3, 2))?.contiguous()?; // [B, h, w/4, neck]
        let x = x.reshape((b, h_patches * w_out, neck_dim))?;
        let x = self.layer_norm2.forward(&x)?;

        // Summary: [B, summary_dim] → [B, neck] → unsqueeze → [B, 1, neck]
        let summary = self.sum_proj.forward(&summary)?;
        let summary = self.layer_norm3.forward(&summary)?;
        let summary = summary.unsqueeze(1)?;

        // Concatenate: [B, S'+1, neck]
        Tensor::cat(&[x, summary], 1)
    }
}

// ─── MBartDecoderLayer ───────────────────────────────────────────────────────

/// Pre-norm MBART decoder layer.
///
/// Forward order (pre-norm variant):
///   `self_attn_layer_norm` → `self_attn` → residual
///   `encoder_attn_layer_norm` → `encoder_attn` → residual
///   `final_layer_norm` → fc1 → GELU → fc2 → residual
struct MBartDecoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: MBartSelfAttention,
    encoder_attn_layer_norm: LayerNorm,
    encoder_attn: MBartCrossAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl MBartDecoderLayer {
    fn new(cfg: &NemotronParseDecoderCfg, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            self_attn_layer_norm: layer_norm(d, eps, vb.pp("self_attn_layer_norm"))?,
            self_attn: MBartSelfAttention::new(cfg, vb.pp("self_attn"))?,
            encoder_attn_layer_norm: layer_norm(d, eps, vb.pp("encoder_attn_layer_norm"))?,
            encoder_attn: MBartCrossAttention::new(cfg, vb.pp("encoder_attn"))?,
            final_layer_norm: layer_norm(d, eps, vb.pp("final_layer_norm"))?,
            fc1: linear_b(d, cfg.ffn_dim, true, vb.pp("fc1"))?,
            fc2: linear_b(cfg.ffn_dim, d, true, vb.pp("fc2"))?,
        })
    }

    /// * `xs`     — `[B, T_dec, D]`
    /// * `enc_hs` — `[B, S_enc, D]`
    /// * `mask`   — optional causal mask `[1, 1, T, T]`
    fn forward(&self, xs: &Tensor, enc_hs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention (pre-norm)
        let residual = xs;
        let normed = self.self_attn_layer_norm.forward(xs)?;
        let sa = self.self_attn.forward(&normed, mask)?;
        let xs = (residual + sa)?;

        // Cross-attention (pre-norm)
        let residual = &xs;
        let normed = self.encoder_attn_layer_norm.forward(&xs)?;
        let ca = self.encoder_attn.forward(&normed, enc_hs)?;
        let xs = (residual + ca)?;

        // FFN (pre-norm)
        let residual = &xs;
        let normed = self.final_layer_norm.forward(&xs)?;
        let ff = self.fc2.forward(&self.fc1.forward(&normed)?.gelu_erf()?)?;
        residual + ff
    }
}

// ─── MBartSelfAttention ──────────────────────────────────────────────────────

/// Scaled dot-product self-attention with optional causal mask.
///
/// Weight paths: `q_proj`, `k_proj` (no bias), `v_proj`, `out_proj`.
struct MBartSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MBartSelfAttention {
    fn new(cfg: &NemotronParseDecoderCfg, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: linear_b(d, d, true, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d, d, vb.pp("k_proj"))?,
            v_proj: linear_b(d, d, true, vb.pp("v_proj"))?,
            out_proj: linear_b(d, d, true, vb.pp("out_proj"))?,
            num_heads: cfg.num_heads,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    /// `x [B, T, D]` → `[B, T, D]`
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, nh * hd))?;
        self.out_proj.forward(&out)
    }
}

// ─── MBartCrossAttention ─────────────────────────────────────────────────────

/// Cross-attention: Q from decoder, K/V from encoder.
///
/// Weight paths: `q_proj`, `k_proj` (no bias), `v_proj`, `out_proj`.
struct MBartCrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MBartCrossAttention {
    fn new(cfg: &NemotronParseDecoderCfg, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: linear_b(d, d, true, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d, d, vb.pp("k_proj"))?,
            v_proj: linear_b(d, d, true, vb.pp("v_proj"))?,
            out_proj: linear_b(d, d, true, vb.pp("out_proj"))?,
            num_heads: cfg.num_heads,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    /// * `dec [B, T_dec, D]`, `enc [B, S_enc, D]` → `[B, T_dec, D]`
    fn forward(&self, dec: &Tensor, enc: &Tensor) -> Result<Tensor> {
        let (b, t_dec, _) = dec.dims3()?;
        let t_enc = enc.dim(1)?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self
            .q_proj
            .forward(dec)?
            .reshape((b, t_dec, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(enc)?
            .reshape((b, t_enc, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(enc)?
            .reshape((b, t_enc, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t_dec, nh * hd))?;
        self.out_proj.forward(&out)
    }
}

// ─── MBartDecoderNoPos ───────────────────────────────────────────────────────

/// MBART-style decoder without positional embeddings.
///
/// Forward: embed_tokens × embed_scale → layernorm_embedding → decoder layers → layer_norm
struct MBartDecoderNoPos {
    embed_tokens: Embedding,
    layernorm_embedding: LayerNorm,
    layers: Vec<MBartDecoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
}

impl MBartDecoderNoPos {
    fn new(cfg: &NemotronParseDecoderCfg, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;
        let layernorm_embedding = layer_norm(
            cfg.d_model,
            cfg.layer_norm_eps,
            vb.pp("layernorm_embedding"),
        )?;
        let layers = (0..cfg.num_layers)
            .map(|i| MBartDecoderLayer::new(cfg, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = layer_norm(cfg.d_model, cfg.layer_norm_eps, vb.pp("layer_norm"))?;
        Ok(Self {
            embed_tokens,
            layernorm_embedding,
            layers,
            layer_norm,
            embed_scale: cfg.embed_scale(),
        })
    }

    /// * `input_ids [B, T]`, `encoder_hidden [B, S_enc, D]` → `[B, T, D]`
    fn forward(&self, input_ids: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let t_dec = input_ids.dim(1)?;
        let device = input_ids.device();

        let mut xs = self.embed_tokens.forward(input_ids)?;
        if (self.embed_scale - 1.0).abs() > 1e-6 {
            xs = (xs * self.embed_scale)?;
        }
        let mut xs = self.layernorm_embedding.forward(&xs)?;

        // Causal mask for self-attention (only needed for prefill / T > 1)
        let mask = if t_dec > 1 {
            Some(crate::layers::causal_mask(t_dec, 0, xs.dtype(), device)?)
        } else {
            None
        };

        for layer in &self.layers {
            xs = layer.forward(&xs, encoder_hidden, mask.as_ref())?;
        }
        self.layer_norm.forward(&xs)
    }
}

// ─── NemotronParseForConditionalGeneration ───────────────────────────────────

/// NemotronParse document-understanding model.
///
/// Implements [`ModelForEncoderDecoder`]:
/// - `encode()` takes pixel values `[B, 3, H, W]` and returns `[B, S', neck_dim]`.
/// - `decode()` takes decoder token IDs `[B, T]` and returns logits `[B, T, vocab_size]`.
pub struct NemotronParseForConditionalGeneration {
    encoder: RadioWithNeck,
    decoder: MBartDecoderNoPos,
    lm_head: Linear,
    cfg: NemotronParseConfig,
    device: Device,
}

impl NemotronParseForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let parse_cfg = NemotronParseConfig::from_model_config(cfg);
        let encoder = RadioWithNeck::new(&parse_cfg.encoder, vb.pp("encoder"))?;
        let decoder = MBartDecoderNoPos::new(&parse_cfg.decoder, vb.pp("decoder"))?;

        let vocab = parse_cfg.decoder.vocab_size;
        let d = parse_cfg.decoder.d_model;
        let lm_head_weight = vb.get((vocab, d), "lm_head.weight")?;
        let lm_head = Linear::new(lm_head_weight, None);

        Ok(Self {
            encoder,
            decoder,
            lm_head,
            device: vb.device().clone(),
            cfg: parse_cfg,
        })
    }
}

impl ModelForEncoderDecoder for NemotronParseForConditionalGeneration {
    /// Encode pixel values `[B, 3, H, W]` → visual encoder output.
    fn encode(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<EncoderOutput> {
        let hidden = self.encoder.forward(input_ids)?;
        EncoderOutput::new(hidden)
    }

    /// Decode one step: `input_ids [B, T]` + encoder output → logits `[B, T, vocab]`.
    fn decode(
        &self,
        decoder_input_ids: &Tensor,
        encoder_output: &EncoderOutput,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let hidden = self
            .decoder
            .forward(decoder_input_ids, &encoder_output.hidden_states)?;
        hidden.apply(&self.lm_head)
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.cfg.decoder.decoder_start_token_id
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_source_len(&self) -> usize {
        self.cfg.num_vision_tokens()
    }

    fn max_target_len(&self) -> usize {
        self.cfg.decoder.vocab_size // conservative upper bound
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    fn small_enc_cfg() -> NemotronParseEncoderCfg {
        let radio = RadioVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            patch_size: 4,
            image_size: 32,
            cpe_max_size: 32,
            qkv_bias: false,
            qk_normalization: false,
            layer_norm_eps: 1e-6,
            num_cls_tokens: 2,
            num_registers: 0,
        };
        NemotronParseEncoderCfg {
            radio,
            neck_dim: 16,
            summary_dim: 2 * 32, // num_cls_tokens * hidden_size
        }
    }

    fn small_dec_cfg() -> NemotronParseDecoderCfg {
        NemotronParseDecoderCfg {
            d_model: 16,
            num_heads: 4,
            num_layers: 1,
            ffn_dim: 32,
            vocab_size: 64,
            scale_embedding: true,
            layer_norm_eps: 1e-5,
            decoder_start_token_id: 0,
        }
    }

    /// Build a VarMap/VarBuilder with all weights needed by RadioWithNeck.
    fn make_neck_vb(enc: &NemotronParseEncoderCfg, device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let radio = &enc.radio;

        let radio_vb = vb.pp("model_encoder").pp("radio_model").pp("model");
        let pg = radio_vb.pp("patch_generator");
        let num_patches = (radio.cpe_max_size / radio.patch_size).pow(2);
        let patch_flat = 3 * radio.patch_size * radio.patch_size;
        pg.get((1, num_patches, radio.hidden_size), "pos_embed")
            .unwrap();
        pg.pp("embedder")
            .get((radio.hidden_size, patch_flat), "weight")
            .unwrap();
        let num_cls_total = radio.num_cls_tokens + radio.num_registers;
        if num_cls_total > 0 {
            pg.pp("cls_token")
                .get((num_cls_total, radio.hidden_size), "token")
                .unwrap();
        }
        for i in 0..radio.num_hidden_layers {
            let b = radio_vb.pp("blocks").pp(i);
            for nm in ["norm1", "norm2"] {
                b.pp(nm).get(radio.hidden_size, "weight").unwrap();
                b.pp(nm).get(radio.hidden_size, "bias").unwrap();
            }
            let attn = b.pp("attn");
            attn.get((3 * radio.hidden_size, radio.hidden_size), "weight")
                .unwrap();
            attn.pp("proj")
                .get((radio.hidden_size, radio.hidden_size), "weight")
                .unwrap();
            let mlp = b.pp("mlp");
            mlp.pp("fc1")
                .get((radio.intermediate_size, radio.hidden_size), "weight")
                .unwrap();
            mlp.pp("fc2")
                .get((radio.hidden_size, radio.intermediate_size), "weight")
                .unwrap();
        }

        let h = radio.hidden_size;
        let neck = enc.neck_dim;
        let summary = enc.summary_dim;
        vb.pp("conv1").get((neck, h, 1), "weight").unwrap();
        vb.pp("conv1").get(neck, "bias").unwrap();
        for w in ["weight", "bias"] {
            vb.pp("layer_norm1").get(neck, w).unwrap();
            vb.pp("layer_norm2").get(neck, w).unwrap();
            vb.pp("layer_norm3").get(neck, w).unwrap();
        }
        vb.pp("conv2").get((neck, neck, 1, 4), "weight").unwrap();
        vb.pp("sum_proj").get((neck, summary), "weight").unwrap();
        vb.pp("sum_proj").get(neck, "bias").unwrap();

        vb
    }

    /// Build a VarBuilder with all weights needed by MBartDecoderNoPos.
    fn make_decoder_vb(dec: &NemotronParseDecoderCfg, device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let d = dec.d_model;
        let ffn = dec.ffn_dim;
        let v = dec.vocab_size;

        vb.pp("embed_tokens").get((v, d), "weight").unwrap();
        for w in ["weight", "bias"] {
            vb.pp("layernorm_embedding").get(d, w).unwrap();
            vb.pp("layer_norm").get(d, w).unwrap();
        }
        let lv = vb.pp("layers").pp(0);
        for nm in [
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ] {
            lv.pp(nm).get(d, "weight").unwrap();
            lv.pp(nm).get(d, "bias").unwrap();
        }
        for attn in ["self_attn", "encoder_attn"] {
            for proj in ["q_proj", "v_proj", "out_proj"] {
                lv.pp(attn).pp(proj).get((d, d), "weight").unwrap();
                lv.pp(attn).pp(proj).get(d, "bias").unwrap();
            }
            lv.pp(attn).pp("k_proj").get((d, d), "weight").unwrap();
        }
        lv.pp("fc1").get((ffn, d), "weight").unwrap();
        lv.pp("fc1").get(ffn, "bias").unwrap();
        lv.pp("fc2").get((d, ffn), "weight").unwrap();
        lv.pp("fc2").get(d, "bias").unwrap();

        vb
    }

    fn make_extra(json: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        json.as_object().expect("json must be object").clone()
    }

    #[test]
    fn test_neck_config_dims() {
        let enc = small_enc_cfg();
        assert_eq!(enc.summary_dim, 64); // 2 * 32
        assert_eq!(enc.neck_dim, 16);
    }

    #[test]
    fn test_parse_config_defaults() {
        let extra = make_extra(serde_json::json!({
            "image_size": [2048, 1648],
            "encoder": {
                "patch_size": 16,
                "hidden_size": 1280,
                "num_cls_tokens": 3,
                "register_multiple": 4
            },
            "decoder": {
                "d_model": 1024,
                "decoder_attention_heads": 16,
                "decoder_layers": 12,
                "encoder_ffn_dim": 4096,
                "vocab_size": 50266
            }
        }));
        let cfg = ModelConfig {
            extra,
            ..Default::default()
        };
        let pc = NemotronParseConfig::from_model_config(&cfg);
        assert_eq!(pc.image_h, 2048);
        assert_eq!(pc.image_w, 1648);
        assert_eq!(pc.encoder.radio.num_cls_tokens, 3);
        // register_multiple=4, num_cls_tokens=3 → num_registers = 4 - (3%4) = 1
        assert_eq!(pc.encoder.radio.num_registers, 1);
        assert_eq!(pc.encoder.summary_dim, 3 * 1280);
        assert_eq!(pc.decoder.d_model, 1024);
        assert_eq!(pc.decoder.num_layers, 12);
    }

    #[test]
    fn test_radio_with_neck_shape() {
        let device = Device::Cpu;
        let enc = small_enc_cfg();
        let vb = make_neck_vb(&enc, &device);
        let neck = RadioWithNeck::new(&enc, vb).unwrap();

        let img = Tensor::zeros((1, 3, 32, 32), DType::F32, &device).unwrap();
        let out = neck.forward(&img).unwrap();
        // h_patches=8, w_patches=8, w_out=8/4=2, spatial=8*2=16, +1 summary = 17
        let expected_s = (32 / 4) * (32 / 4 / 4) + 1;
        assert_eq!(out.dims(), &[1, expected_s, enc.neck_dim]);
    }

    #[test]
    fn test_mbartdecodernopos_shape() {
        let device = Device::Cpu;
        let dec_cfg = small_dec_cfg();
        let vb = make_decoder_vb(&dec_cfg, &device);
        let decoder = MBartDecoderNoPos::new(&dec_cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).unwrap();
        let enc_hs = Tensor::zeros((1, 17, dec_cfg.d_model), DType::F32, &device).unwrap();
        let out = decoder.forward(&input_ids, &enc_hs).unwrap();
        assert_eq!(out.dims(), &[1, 5, dec_cfg.d_model]);
    }

    #[test]
    fn test_num_vision_tokens() {
        let extra = make_extra(serde_json::json!({
            "image_size": [2048, 1648],
            "encoder": { "patch_size": 16, "num_cls_tokens": 3 },
            "decoder": { "d_model": 1024, "vocab_size": 50266 }
        }));
        let cfg = ModelConfig {
            extra,
            ..Default::default()
        };
        let pc = NemotronParseConfig::from_model_config(&cfg);
        // h_patches=128, w_patches=103, w_out=103/4=25, total=128*25+1=3201
        let expected = (2048 / 16) * (1648 / 16 / 4) + 1;
        assert_eq!(pc.num_vision_tokens(), expected);
    }
}
