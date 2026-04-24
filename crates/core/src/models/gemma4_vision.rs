//! Gemma 4 Vision Tower.
//!
//! Mirrors `Gemma4VisionModel` from HuggingFace `transformers` (Gemma 4
//! family). Unlike CLIP / SigLIP, Gemma 4 uses a patch *linear* projection
//! (not Conv2d), a 2-dimensional position-embedding lookup table, and a
//! Gemma-style encoder layer (4 RmsNorms, Q/K/V norms, 2D multidimensional
//! RoPE, SwiGLU MLP). All linears are wrapped in `Gemma4ClippableLinear`
//! which can apply per-tensor activation clipping.
//!
//! In HuggingFace `unsloth` BnB 4-bit checkpoints the whole vision tower
//! sits in `llm_int8_skip_modules`, so the weights live in BF16/FP16 and
//! this module loads them via plain `VarBuilder::get` (no quantization).

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use serde_json::Value;

use crate::config::ModelConfig;

use super::gemma4::{Gemma4RmsNorm, UnweightedRmsNorm};

// ─── Config ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub position_embedding_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub use_clipped_linears: bool,
    pub standardize: bool,
}

impl Gemma4VisionConfig {
    /// Parse from the nested `vision_config` JSON blob we keep in
    /// `ModelConfig::extras`. Falls back to sensible defaults so that
    /// test constructors can pass a partial config.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vc = cfg
            .extra
            .get("vision_config")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        let get_usize = |k: &str, default: usize| -> usize {
            vc.get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f64 =
            |k: &str, default: f64| -> f64 { vc.get(k).and_then(Value::as_f64).unwrap_or(default) };
        let get_bool = |k: &str, default: bool| -> bool {
            vc.get(k).and_then(Value::as_bool).unwrap_or(default)
        };
        let hidden_size = get_usize("hidden_size", 768);
        let num_attention_heads = get_usize("num_attention_heads", 12);
        let num_key_value_heads = get_usize("num_key_value_heads", num_attention_heads);
        let head_dim = get_usize("head_dim", hidden_size / num_attention_heads.max(1));

        // rope_parameters: {rope_theta, rope_type}
        let rope_theta = vc
            .get("rope_parameters")
            .and_then(Value::as_object)
            .and_then(|rp: &serde_json::Map<String, Value>| rp.get("rope_theta"))
            .and_then(Value::as_f64)
            .unwrap_or(100.0);

        Self {
            hidden_size,
            intermediate_size: get_usize("intermediate_size", 3072),
            num_hidden_layers: get_usize("num_hidden_layers", 16),
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            patch_size: get_usize("patch_size", 16),
            position_embedding_size: get_usize("position_embedding_size", 10240),
            pooling_kernel_size: get_usize("pooling_kernel_size", 3),
            default_output_length: get_usize("default_output_length", 280),
            rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
            rope_theta,
            use_clipped_linears: get_bool("use_clipped_linears", true),
            standardize: get_bool("standardize", false),
        }
    }
}

// ─── ClippableLinear ───────────────────────────────────────────────────────
//
// Wraps a plain `Linear` and (optionally) clamps activations in the
// `[input_min, input_max]` / `[output_min, output_max]` ranges stored as
// scalar buffers in the checkpoint. Reference:
// `transformers.models.gemma4.modeling_gemma4.Gemma4ClippableLinear`.
//
// For most HF checkpoints (including `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`)
// the clips default to ±inf, making clamping a no-op — but we still load
// the buffers so fine-tuned checkpoints with non-trivial ranges work.

struct ClipRanges {
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

struct Gemma4ClippableLinear {
    linear: Linear,
    clips: Option<ClipRanges>,
}

impl Gemma4ClippableLinear {
    fn new(
        in_features: usize,
        out_features: usize,
        use_clipped_linears: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Real weight lives under `{prefix}.linear.weight`.
        let vb_l = vb.pp("linear");
        let weight = vb_l.get((out_features, in_features), "weight")?;
        let linear = Linear::new(weight, None);

        let clips = if use_clipped_linears {
            // Scalar buffers stored as shape `()` (0-dim) or `(1,)`. HF uses
            // 0-dim `torch.tensor(-float("inf"))`, so we try both shapes.
            let load_scalar = |name: &str, default: f32| -> f32 {
                vb.get((), name)
                    .or_else(|_| vb.get(1, name))
                    .and_then(|t| t.to_dtype(DType::F32)?.flatten_all()?.to_scalar::<f32>())
                    .unwrap_or(default)
            };
            Some(ClipRanges {
                input_min: load_scalar("input_min", f32::NEG_INFINITY),
                input_max: load_scalar("input_max", f32::INFINITY),
                output_min: load_scalar("output_min", f32::NEG_INFINITY),
                output_max: load_scalar("output_max", f32::INFINITY),
            })
        } else {
            None
        };

        Ok(Self { linear, clips })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = match &self.clips {
            Some(c) if c.input_min.is_finite() || c.input_max.is_finite() => clamp_scalar(
                x,
                if c.input_min.is_finite() {
                    Some(c.input_min)
                } else {
                    None
                },
                if c.input_max.is_finite() {
                    Some(c.input_max)
                } else {
                    None
                },
            )?,
            _ => x.clone(),
        };

        let y = self.linear.forward(&x)?;

        match &self.clips {
            Some(c) if c.output_min.is_finite() || c.output_max.is_finite() => clamp_scalar(
                &y,
                if c.output_min.is_finite() {
                    Some(c.output_min)
                } else {
                    None
                },
                if c.output_max.is_finite() {
                    Some(c.output_max)
                } else {
                    None
                },
            ),
            _ => Ok(y),
        }
    }
}

fn clamp_scalar(x: &Tensor, lo: Option<f32>, hi: Option<f32>) -> Result<Tensor> {
    let mut y = x.clone();
    if let Some(lo) = lo {
        y = y.maximum(&Tensor::new(lo, x.device())?.to_dtype(x.dtype())?)?;
    }
    if let Some(hi) = hi {
        y = y.minimum(&Tensor::new(hi, x.device())?.to_dtype(x.dtype())?)?;
    }
    Ok(y)
}

// ─── PatchEmbedder ─────────────────────────────────────────────────────────

struct Gemma4VisionPatchEmbedder {
    input_proj: Linear,
    position_embedding_table: Tensor, // [2, position_embedding_size, hidden_size]
    position_embedding_size: usize,
    hidden_size: usize,
}

impl Gemma4VisionPatchEmbedder {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let in_features = 3 * cfg.patch_size * cfg.patch_size;
        let input_proj_w = vb
            .pp("input_proj")
            .get((cfg.hidden_size, in_features), "weight")?;
        let input_proj = Linear::new(input_proj_w, None);

        let position_embedding_table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;

        Ok(Self {
            input_proj,
            position_embedding_table,
            position_embedding_size: cfg.position_embedding_size,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Compute positional embeddings by summing lookups along x and y axes.
    ///
    /// `pixel_position_ids`: i64 `[B, L, 2]` with -1 for padding patches.
    /// `padding_positions`:   bool `[B, L]` (true where padded).
    fn position_embeddings(
        &self,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        let b = pixel_position_ids.dim(0)?;
        let l = pixel_position_ids.dim(1)?;
        let dtype = self.position_embedding_table.dtype();

        // Clamp -1 padding → 0 for the lookup; padding is zeroed out below.
        let clamped = pixel_position_ids.clamp(0i64, (self.position_embedding_size - 1) as i64)?;

        // Split (x, y) axes. Each has shape [B, L].
        let x_ids = clamped.i((.., .., 0))?.contiguous()?;
        let y_ids = clamped.i((.., .., 1))?.contiguous()?;

        // Flat indices → embedding lookups per axis.
        let table_x = self.position_embedding_table.i(0)?; // [P, H]
        let table_y = self.position_embedding_table.i(1)?; // [P, H]

        let flat_x = x_ids.flatten_all()?;
        let flat_y = y_ids.flatten_all()?;
        let emb_x = table_x
            .index_select(&flat_x, 0)?
            .reshape((b, l, self.hidden_size))?;
        let emb_y = table_y
            .index_select(&flat_y, 0)?
            .reshape((b, l, self.hidden_size))?;
        let mut pos = (emb_x + emb_y)?;

        // Zero-out padding rows.
        let mask = padding_positions
            .to_dtype(dtype)?
            .unsqueeze(D::Minus1)?
            .broadcast_as(pos.shape())?;
        let one = Tensor::ones_like(&mask)?;
        pos = pos.mul(&(one - mask)?)?;

        Ok(pos)
    }

    fn forward(
        &self,
        pixel_values: &Tensor,       // [B, L, 3*ps*ps], float
        pixel_position_ids: &Tensor, // i64 [B, L, 2]
        padding_positions: &Tensor,  // bool [B, L]
    ) -> Result<Tensor> {
        // HF: no normalization, scales pixels in-place as `2 * (x - 0.5)`.
        let x = pixel_values
            .affine(2.0, -1.0)?
            .to_dtype(self.input_proj.weight().dtype())?;
        let h = self.input_proj.forward(&x)?;
        let pos = self.position_embeddings(pixel_position_ids, padding_positions)?;
        h.broadcast_add(&pos)
    }
}

// ─── MLP (SwiGLU with clippable linears) ────────────────────────────────────

struct Gemma4VisionMLP {
    gate_proj: Gemma4ClippableLinear,
    up_proj: Gemma4ClippableLinear,
    down_proj: Gemma4ClippableLinear,
    // Only `gelu_pytorch_tanh` is used by Gemma 4 vision; we implement it
    // via `candle_nn::ops::gelu` which matches the PyTorch "tanh" approx.
}

impl Gemma4VisionMLP {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = Gemma4ClippableLinear::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_clipped_linears,
            vb.pp("gate_proj"),
        )?;
        let up_proj = Gemma4ClippableLinear::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_clipped_linears,
            vb.pp("up_proj"),
        )?;
        let down_proj = Gemma4ClippableLinear::new(
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_clipped_linears,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // gelu_pytorch_tanh ≈ candle's gelu (tanh approximation).
        let activated = gate.gelu()?;
        self.down_proj.forward(&(activated * up)?)
    }
}

// ─── Rotary Embeddings (2D multidimensional) ───────────────────────────────

struct Gemma4VisionRotaryEmb {
    inv_freq: Tensor, // [spatial_dim/2]
}

impl Gemma4VisionRotaryEmb {
    fn new(cfg: &Gemma4VisionConfig, device: &Device) -> Result<Self> {
        // Each of the 2 axes gets a `spatial_dim = head_dim / 2` slice.
        // Within that slice, RoPE pairs are produced from `arange(0, spatial_dim, 2)`.
        let spatial_dim = cfg.head_dim / 2;
        let n = spatial_dim.div_ceil(2);
        let mut freqs = Vec::with_capacity(n);
        for i in (0..spatial_dim).step_by(2) {
            let exp = (i as f64) / (spatial_dim as f64);
            freqs.push((1.0f64 / cfg.rope_theta.powf(exp)) as f32);
        }
        let inv_freq = Tensor::from_vec(freqs, n, device)?;
        Ok(Self { inv_freq })
    }

    /// Build (cos, sin) for a batch of 2D positions.
    ///
    /// `position_ids`: i64 [B, L, 2].
    /// Returns `(cos, sin)` of shape [B, L, head_dim] each — already
    /// concatenated across the 2 spatial axes and ready for
    /// `apply_multidimensional_rope`.
    fn forward(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let b = position_ids.dim(0)?;
        let l = position_ids.dim(1)?;

        let inv_freq_f32 = self.inv_freq.to_dtype(DType::F32)?;
        let n = inv_freq_f32.dim(0)?;

        let mut cos_parts = Vec::with_capacity(2);
        let mut sin_parts = Vec::with_capacity(2);
        for axis in 0..2 {
            // pos_axis: [B, L] i64 → f32 [B, L, 1]
            let pos = position_ids
                .i((.., .., axis))?
                .to_dtype(DType::F32)?
                .unsqueeze(D::Minus1)?; // [B, L, 1]

            // broadcast_mul with inv_freq [n] gives [B, L, n]
            let freqs = pos.broadcast_mul(&inv_freq_f32.reshape((1, 1, n))?)?;

            // Duplicate freqs to form head_dim/2 = 2*n channels per axis
            // (HF: `emb = torch.cat((freqs, freqs), dim=-1)` along the
            // axis-local dim). Later `apply_rotary_pos_emb` will split it
            // into 2×n pairs internally via `rotate_half`.
            let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
            cos_parts.push(emb.cos()?);
            sin_parts.push(emb.sin()?);
        }
        let cos = Tensor::cat(&cos_parts, D::Minus1)?.to_dtype(dtype)?;
        let sin = Tensor::cat(&sin_parts, D::Minus1)?.to_dtype(dtype)?;

        // Expected shape: [B, L, head_dim]
        assert_eq!(cos.dims(), &[b, l, 4 * n]);

        Ok((cos, sin))
    }
}

/// HF `rotate_half`: split last dim in two, swap and negate.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, d - half)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
}

/// Apply RoPE to `x` with per-axis pair splitting, matching HF's
/// `apply_multidimensional_rope`.
///
/// `x`    : [B, L, H, head_dim] (head-major)
/// `cos`  : [B, L, head_dim]
/// `sin`  : [B, L, head_dim]
fn apply_multidim_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Number of axes = 2; equal split across last dim.
    let ndim = 2usize;
    let d = x.dim(D::Minus1)?;
    let per_axis = 2 * (d / (2 * ndim)); // matches HF num_rotated_channels_per_dim

    let cos_h = cos.unsqueeze(2)?; // [B, L, 1, head_dim]
    let sin_h = sin.unsqueeze(2)?;

    let mut parts = Vec::with_capacity(ndim);
    for k in 0..ndim {
        let start = k * per_axis;
        let x_k = x.narrow(D::Minus1, start, per_axis)?;
        let cos_k = cos_h.narrow(D::Minus1, start, per_axis)?;
        let sin_k = sin_h.narrow(D::Minus1, start, per_axis)?;
        let rotated = rotate_half(&x_k)?;
        let y_k = x_k.broadcast_mul(&cos_k)? + rotated.broadcast_mul(&sin_k)?;
        parts.push(y_k?);
    }
    Tensor::cat(&parts, D::Minus1)
}

// ─── Attention ─────────────────────────────────────────────────────────────

struct Gemma4VisionAttention {
    q_proj: Gemma4ClippableLinear,
    k_proj: Gemma4ClippableLinear,
    v_proj: Gemma4ClippableLinear,
    o_proj: Gemma4ClippableLinear,
    q_norm: Gemma4RmsNorm,
    k_norm: Gemma4RmsNorm,
    v_norm: UnweightedRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Gemma4VisionAttention {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;

        let q_proj = Gemma4ClippableLinear::new(
            cfg.hidden_size,
            q_out,
            cfg.use_clipped_linears,
            vb.pp("q_proj"),
        )?;
        let k_proj = Gemma4ClippableLinear::new(
            cfg.hidden_size,
            kv_out,
            cfg.use_clipped_linears,
            vb.pp("k_proj"),
        )?;
        let v_proj = Gemma4ClippableLinear::new(
            cfg.hidden_size,
            kv_out,
            cfg.use_clipped_linears,
            vb.pp("v_proj"),
        )?;
        let o_proj = Gemma4ClippableLinear::new(
            q_out,
            cfg.hidden_size,
            cfg.use_clipped_linears,
            vb.pp("o_proj"),
        )?;

        let q_norm = Gemma4RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Gemma4RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let v_norm = UnweightedRmsNorm::new(cfg.rms_norm_eps);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,          // [B, L, H]
        cos: &Tensor,                    // [B, L, head_dim]
        sin: &Tensor,                    // [B, L, head_dim]
        attention_mask: Option<&Tensor>, // additive mask [B, 1, L, L] or None
    ) -> Result<Tensor> {
        let b = hidden_states.dim(0)?;
        let l = hidden_states.dim(1)?;

        // Project + reshape to (B, L, H, Dh).
        let q =
            self.q_proj
                .forward(hidden_states)?
                .reshape((b, l, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = apply_multidim_rope(&q, cos, sin)?;
        let q = q.transpose(1, 2)?.contiguous()?; // [B, H, L, Dh]

        let k = self.k_proj.forward(hidden_states)?.reshape((
            b,
            l,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k)?;
        let k = apply_multidim_rope(&k, cos, sin)?;
        let k = k.transpose(1, 2)?.contiguous()?; // [B, Hk, L, Dh]

        let v = self.v_proj.forward(hidden_states)?.reshape((
            b,
            l,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = self.v_norm.forward(&v)?;
        let v = v.transpose(1, 2)?.contiguous()?; // [B, Hk, L, Dh]

        // Expand KV heads for GQA, if any (Gemma 4 vision uses MHA but
        // we stay general).
        let k = repeat_kv(&k, self.num_heads / self.num_kv_heads.max(1))?;
        let v = repeat_kv(&v, self.num_heads / self.num_kv_heads.max(1))?;

        // Attention. `scaling = 1.0` per HF (the QK/V norms replace 1/√d).
        let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        let attn_out = attn_weights.matmul(&v)?.transpose(1, 2)?.contiguous()?;
        let attn_out = attn_out.reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_out)
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep <= 1 {
        return Ok(x.clone());
    }
    let (b, h_kv, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h_kv, n_rep, l, d))?
        .reshape((b, h_kv * n_rep, l, d))
}

// ─── Encoder Layer ─────────────────────────────────────────────────────────

struct Gemma4VisionEncoderLayer {
    self_attn: Gemma4VisionAttention,
    mlp: Gemma4VisionMLP,
    input_layernorm: Gemma4RmsNorm,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
}

impl Gemma4VisionEncoderLayer {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Gemma4VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Gemma4VisionMLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            Gemma4RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Gemma4RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = Gemma4RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = Gemma4RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self.self_attn.forward(&h, cos, sin, attention_mask)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = (residual + h)?;

        let residual = h.clone();
        let mlp_in = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        residual + mlp_out
    }
}

// ─── Encoder ───────────────────────────────────────────────────────────────

struct Gemma4VisionEncoder {
    layers: Vec<Gemma4VisionEncoderLayer>,
    rotary_emb: Gemma4VisionRotaryEmb,
}

impl Gemma4VisionEncoder {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let rotary_emb = Gemma4VisionRotaryEmb::new(cfg, vb.device())?;
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma4VisionEncoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        Ok(Self { layers, rotary_emb })
    }

    fn forward(
        &self,
        inputs_embeds: &Tensor,      // [B, L, H]
        pixel_position_ids: &Tensor, // i64 [B, L, 2]
        padding_positions: &Tensor,  // bool [B, L]
    ) -> Result<Tensor> {
        let dtype = inputs_embeds.dtype();
        let (cos, sin) = self.rotary_emb.forward(pixel_position_ids, dtype)?;

        // Bidirectional mask: valid tokens attend to valid tokens; padded
        // tokens are masked out along the KEY axis.
        let attention_mask = bidirectional_additive_mask(padding_positions, dtype)?;

        let mut h = inputs_embeds.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, Some(&attention_mask))?;
        }
        Ok(h)
    }
}

/// Build additive mask `[B, 1, L, L]` with -inf at padded key positions.
fn bidirectional_additive_mask(padding_positions: &Tensor, dtype: DType) -> Result<Tensor> {
    // padding_positions: bool [B, L]; true = padded.
    let b = padding_positions.dim(0)?;
    let l = padding_positions.dim(1)?;
    let device = padding_positions.device();

    // [B, 1, 1, L] in f32, then cast to target dtype.
    let pad_f = padding_positions
        .to_dtype(DType::F32)?
        .reshape((b, 1, 1, l))?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, pad_f.shape(), device)?;
    let zero = Tensor::zeros(pad_f.shape(), DType::F32, device)?;
    // where(padding, -inf, 0)
    let ones = Tensor::ones_like(&pad_f)?;
    let mask = pad_f.broadcast_mul(&neg_inf)? + (ones - pad_f)?.broadcast_mul(&zero)?;
    let mask = mask?;
    mask.broadcast_as((b, 1, l, l))?.to_dtype(dtype)
}

// ─── Pooler ─────────────────────────────────────────────────────────────────

struct Gemma4VisionPooler {
    root_hidden_size: f64,
}

impl Gemma4VisionPooler {
    fn new(cfg: &Gemma4VisionConfig) -> Self {
        Self {
            root_hidden_size: (cfg.hidden_size as f64).sqrt(),
        }
    }

    /// Avg-pool over k×k grids aligned to (x, y) patch coordinates,
    /// producing exactly `output_length` soft tokens.
    ///
    /// Returns `(pooled, pooled_mask)` with shapes `[B, output_length, H]`
    /// and `[B, output_length]` (true = valid).
    fn avg_pool_by_positions(
        &self,
        hidden_states: &Tensor,      // [B, L, H]
        pixel_position_ids: &Tensor, // i64 [B, L, 2]
        output_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let l = hidden_states.dim(1)?;
        let device = hidden_states.device();

        // Derive `k` from input:output ratio (HF: `k = (L/output_length)**.5`).
        let k = ((l / output_length) as f64).sqrt() as usize;
        let k_squared = k * k;
        assert!(
            k > 0 && k_squared * output_length == l,
            "avg_pool_by_positions: cannot pool {l} tokens to {output_length} with k={k}"
        );

        // clamped = max(0, position); -1 → 0 (padding will contribute zeros).
        let clamped = pixel_position_ids.clamp(0i64, i64::MAX)?; // [B, L, 2]
        let xs = clamped.i((.., .., 0))?; // [B, L]
        let ys = clamped.i((.., .., 1))?; // [B, L]

        // max_x = max(xs, dim=-1) + 1 (per-batch scalar).
        let max_x = xs.max_keepdim(D::Minus1)?.affine(1.0, 1.0)?; // [B, 1]

        // kernel_idx = xs//k + (max_x//k) * (ys//k).
        // Candle lacks integer `/scalar`; convert to f32, divide, truncate.
        let kf = k as f64;
        let floor_div = |t: &Tensor| -> Result<Tensor> {
            t.to_dtype(DType::F32)?
                .affine(1.0 / kf, 0.0)?
                .to_dtype(DType::I64)
        };
        let xs_div = floor_div(&xs)?;
        let ys_div = floor_div(&ys)?;
        let max_x_div = floor_div(&max_x)?.broadcast_as(xs_div.shape())?;
        let kernel_idx = (xs_div + max_x_div.mul(&ys_div)?)?; // [B, L]

        // One-hot [B, L, output_length] / k_squared
        let one_hot = one_hot_f32(&kernel_idx, output_length, device)?;
        let weights = one_hot.affine(1.0 / k_squared as f64, 0.0)?; // [B, L, O]

        // Pool: weights.T @ hidden_states  →  [B, O, H]
        let output = weights
            .transpose(1, 2)?
            .contiguous()?
            .matmul(&hidden_states.to_dtype(DType::F32)?)?
            .to_dtype(hidden_states.dtype())?;

        // mask: any weight > 0 along L axis → valid
        let pooled_mask = weights.sum(1)?.gt(0.0f32)?; // [B, O] bool

        Ok((output, pooled_mask))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,      // [B, L, H]
        pixel_position_ids: &Tensor, // i64 [B, L, 2]
        padding_positions: &Tensor,  // bool [B, L]
        output_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let l = hidden_states.dim(1)?;
        assert!(
            output_length <= l,
            "pooler output_length {output_length} exceeds input patches {l}"
        );

        // Zero out padded rows so they contribute nothing to the mean.
        let dtype = hidden_states.dtype();
        let non_pad = padding_positions
            .to_dtype(dtype)?
            .affine(-1.0, 1.0)? // 1 - pad
            .unsqueeze(D::Minus1)?
            .broadcast_as(hidden_states.shape())?;
        let zeroed = hidden_states.mul(&non_pad)?;

        let (pooled, mask) = if l != output_length {
            self.avg_pool_by_positions(&zeroed, pixel_position_ids, output_length)?
        } else {
            let mask = padding_positions
                .to_dtype(DType::U8)?
                .affine(-1.0, 1.0)?
                .gt(0u8)?;
            (zeroed, mask)
        };

        // Scale by √H (HF).
        let scaled = pooled.affine(self.root_hidden_size, 0.0)?;
        Ok((scaled, mask))
    }
}

fn one_hot_f32(indices: &Tensor, num_classes: usize, device: &Device) -> Result<Tensor> {
    // indices: i64 [B, L]. Result: f32 [B, L, C].
    let b = indices.dim(0)?;
    let l = indices.dim(1)?;
    let idx_flat = indices.flatten_all()?; // [B*L]
    let eye = Tensor::eye(num_classes, DType::F32, device)?; // [C, C]
    let oh = eye.index_select(&idx_flat, 0)?; // [B*L, C]
    oh.reshape((b, l, num_classes))
}

// ─── Top-level Vision Model ─────────────────────────────────────────────────

pub(crate) struct Gemma4VisionTower {
    patch_embedder: Gemma4VisionPatchEmbedder,
    encoder: Gemma4VisionEncoder,
    pooler: Gemma4VisionPooler,
    std_bias: Option<Tensor>,  // [H], only if standardize=True
    std_scale: Option<Tensor>, // [H]
    cfg: Gemma4VisionConfig,
}

impl Gemma4VisionTower {
    pub(crate) fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedder = Gemma4VisionPatchEmbedder::new(cfg, vb.pp("patch_embedder"))?;
        let encoder = Gemma4VisionEncoder::new(cfg, vb.pp("encoder"))?;
        let pooler = Gemma4VisionPooler::new(cfg);

        let (std_bias, std_scale) = if cfg.standardize {
            (
                Some(vb.get(cfg.hidden_size, "std_bias")?),
                Some(vb.get(cfg.hidden_size, "std_scale")?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            patch_embedder,
            encoder,
            pooler,
            std_bias,
            std_scale,
            cfg: cfg.clone(),
        })
    }

    /// Full vision pipeline.
    ///
    /// Input:
    /// * `pixel_values`       : float [B, L, 3·ps·ps] — flattened patches.
    /// * `pixel_position_ids` : i64    [B, L, 2] — (x, y) coords, -1 for padding.
    ///
    /// Output:
    /// * hidden states `[total_valid_soft_tokens, H]` (padding stripped).
    pub(crate) fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
    ) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        let l_in = pixel_values.dim(1)?;

        // output_length = L / (pooling_k²). HF computes this from the actual
        // input length so callers don't need to pass it explicitly.
        let pk = self.cfg.pooling_kernel_size;
        let output_length = l_in / (pk * pk);
        assert!(
            output_length > 0,
            "vision input too small for pooling kernel"
        );

        // Derive padding_positions from position_ids == -1 (all components).
        let neg_ones = Tensor::full(
            -1i64,
            pixel_position_ids.shape(),
            pixel_position_ids.device(),
        )?;
        let eq_neg1 = pixel_position_ids.eq(&neg_ones)?; // bool [B, L, 2]
        let padding_positions = eq_neg1.min(D::Minus1)?.to_dtype(DType::U8)?.gt(0u8)?; // [B, L]

        let embeds =
            self.patch_embedder
                .forward(pixel_values, pixel_position_ids, &padding_positions)?;
        let encoded = self
            .encoder
            .forward(&embeds, pixel_position_ids, &padding_positions)?;

        let (pooled, pooled_mask) = self.pooler.forward(
            &encoded,
            pixel_position_ids,
            &padding_positions,
            output_length,
        )?;

        // Strip padded output positions (flatten B × valid rows).
        let mask_flat = pooled_mask.reshape((b * output_length,))?;
        let h = pooled.reshape((b * output_length, self.cfg.hidden_size))?;

        // Gather rows where mask is true.
        let mask_u32 = mask_flat.to_dtype(DType::U32)?;
        let valid_count: u32 = mask_u32.sum_all()?.to_scalar()?;
        if valid_count == 0 {
            return Tensor::zeros((0, self.cfg.hidden_size), h.dtype(), h.device());
        }

        let idxs: Vec<u32> = (0..(b * output_length) as u32).collect();
        let idx_tensor = Tensor::from_vec(idxs, b * output_length, h.device())?;
        let mask_vec: Vec<u8> = pooled_mask.flatten_all()?.to_vec1()?;
        let selected: Vec<u32> = idx_tensor
            .to_vec1::<u32>()?
            .into_iter()
            .zip(mask_vec)
            .filter_map(|(i, m)| if m != 0 { Some(i) } else { None })
            .collect();
        let selected_tensor = Tensor::from_vec(selected, valid_count as usize, h.device())?;
        let mut h = h.index_select(&selected_tensor, 0)?;

        if let (Some(bias), Some(scale)) = (&self.std_bias, &self.std_scale) {
            let bias = bias.to_dtype(h.dtype())?;
            let scale = scale.to_dtype(h.dtype())?;
            h = h.broadcast_sub(&bias)?.broadcast_mul(&scale)?;
        }

        Ok(h)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use serde_json::json;

    fn tiny_cfg() -> Gemma4VisionConfig {
        Gemma4VisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 8,
            patch_size: 4,
            position_embedding_size: 16,
            pooling_kernel_size: 2,
            default_output_length: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            use_clipped_linears: false,
            standardize: false,
        }
    }

    #[test]
    fn test_config_from_model_config() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".into(),
            json!({
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 16,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
                "patch_size": 16,
                "position_embedding_size": 10240,
                "pooling_kernel_size": 3,
                "default_output_length": 280,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
                "use_clipped_linears": true,
                "standardize": false,
                "hidden_activation": "gelu_pytorch_tanh",
            }),
        );
        let mc = ModelConfig {
            extra,
            ..ModelConfig::default()
        };
        let cfg = Gemma4VisionConfig::from_model_config(&mc);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert!(cfg.use_clipped_linears);
        assert_eq!(cfg.rope_theta, 100.0);
    }

    #[test]
    fn test_rotary_shape() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let rope = Gemma4VisionRotaryEmb::new(&cfg, &dev).unwrap();
        let pos = Tensor::from_vec(vec![0i64, 0, 1, 0, 0, 1, 1, 1], (1, 4, 2), &dev).unwrap();
        let (cos, sin) = rope.forward(&pos, DType::F32).unwrap();
        assert_eq!(cos.dims(), &[1, 4, cfg.head_dim]);
        assert_eq!(sin.dims(), &[1, 4, cfg.head_dim]);
    }

    #[test]
    fn test_tower_construction_and_forward_shape() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let tower = Gemma4VisionTower::new(&cfg, vb).unwrap();

        // L = 16 patches so pooling_kernel_size=2 → output_length=4.
        let b = 1;
        let l = 16;
        let patch_px = 3 * cfg.patch_size * cfg.patch_size;
        let pixels = Tensor::rand(0.0f32, 1.0f32, (b, l, patch_px), &dev).unwrap();

        // Positions: 4×4 grid.
        let mut ids = Vec::with_capacity(b * l * 2);
        for y in 0..4i64 {
            for x in 0..4i64 {
                ids.push(x);
                ids.push(y);
            }
        }
        let pos = Tensor::from_vec(ids, (b, l, 2), &dev).unwrap();
        let out = tower.forward(&pixels, &pos).unwrap();
        assert_eq!(out.dim(1).unwrap(), cfg.hidden_size);
        // Output rows = b * output_length (no padding in this test).
        assert_eq!(out.dim(0).unwrap(), 4);
    }

    #[test]
    fn test_pooler_avg_correctness() {
        let cfg = Gemma4VisionConfig {
            hidden_size: 4,
            pooling_kernel_size: 2,
            ..tiny_cfg()
        };
        let dev = Device::Cpu;
        let pooler = Gemma4VisionPooler::new(&cfg);

        // 2×2 grid of 4 patches, each hidden = (x+y)·[1,1,1,1]
        // After pool k=2 → single output = mean of all four = (0+1+1+2)/4·ones = ones.
        let b = 1;
        let l = 4;
        let h = 4;
        let values = vec![
            0.0f32, 0.0, 0.0, 0.0, // (0,0)
            1.0, 1.0, 1.0, 1.0, // (1,0)
            1.0, 1.0, 1.0, 1.0, // (0,1)
            2.0, 2.0, 2.0, 2.0, // (1,1)
        ];
        let hidden = Tensor::from_vec(values, (b, l, h), &dev).unwrap();
        let positions = Tensor::from_vec(vec![0i64, 0, 1, 0, 0, 1, 1, 1], (b, l, 2), &dev).unwrap();
        let pad = Tensor::zeros((b, l), DType::U8, &dev)
            .unwrap()
            .gt(0u8)
            .unwrap();

        let (pooled, mask) = pooler.forward(&hidden, &positions, &pad, 1).unwrap();
        assert_eq!(pooled.dims(), &[1, 1, 4]);
        let values = pooled.to_vec3::<f32>().unwrap();
        // mean = 1.0, scaled by √H = 2.0 → 2.0.
        for v in &values[0][0] {
            assert!((v - 2.0).abs() < 1e-5, "got {v}");
        }
        // All valid.
        let mask_vec: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask_vec, vec![1]);
    }
}
