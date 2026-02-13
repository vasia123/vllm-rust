//! Gemma3n model architecture with AltUp mechanism.
//!
//! Gemma3n extends Gemma3 with Alternating Updates (AltUp), which wraps
//! transformer layers with learned routing between parallel sub-experts.
//!
//! Key differences from Gemma3:
//! - AltUp: predict + correct steps around each transformer layer
//! - Per-layer input embeddings (PLE): separate vocabulary embedding for per-layer gating
//! - Laurel blocks: low-rank learned augmented residual layers
//! - AltUp embedding/unembedding projections for parallel sub-expert initialization
//! - Per-layer input gating with activation and projection
//! - V-norm on values (without learned weight)
//!
//! Architecture:
//! ```text
//! Embedding (* sqrt(hidden_size)) -> AltUp Embed -> [Gemma3nLayer x N] -> AltUp Unembed -> RMSNorm -> LM Head
//!
//! Gemma3nLayer:
//!   AltUp Predict -> InputLayerNorm -> Laurel -> Attention -> PostAttnNorm -> Residual
//!   PreFFNorm -> GeGLU MLP -> PostFFNorm -> Residual
//!   AltUp Correct -> PerLayerGating -> PerLayerProjection -> PostPerLayerNorm
//! ```
//!
//! Reference: `reference/vllm/vllm/model_executor/models/gemma3n.py`

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::repeat_kv;
use crate::layers::RotaryEmbedding;

use super::tp_layers::{TpContext, TpEmbedding, TpGeGluMlp, TpLinear};

// ─── Gemma3n RMSNorm (offset by +1, same as Gemma3) ────────────────────────

struct Gemma3nRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Gemma3nRmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for Gemma3nRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let scale = (&self.weight.to_dtype(DType::F32)? + 1.0)?;
        xs_normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

// ─── RMSNorm without learned weight (for v_norm) ───────────────────────────

struct UnweightedRmsNorm {
    eps: f64,
}

impl UnweightedRmsNorm {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for UnweightedRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs_normed.to_dtype(dtype)
    }
}

// ─── Soft Capping ───────────────────────────────────────────────────────────

fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

// ─── Sliding Window Mask ────────────────────────────────────────────────────

fn sliding_window_mask(
    q_len: usize,
    kv_len: usize,
    seqlen_offset: usize,
    window_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![f32::NEG_INFINITY; q_len * kv_len];
    for i in 0..q_len {
        let query_pos = seqlen_offset + i;
        for j in 0..kv_len {
            let is_causal = j <= query_pos;
            let is_in_window = query_pos < window_size || j > query_pos - window_size;
            if is_causal && is_in_window {
                mask[i * kv_len + j] = 0.0;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

// ─── Gemma3n Config Extraction ──────────────────────────────────────────────

#[allow(dead_code)]
struct Gemma3nExtraConfig {
    query_pre_attn_scalar: f64,
    attn_logit_softcap: Option<f64>,
    final_logit_softcap: Option<f64>,
    sliding_window_pattern: usize,
    rope_theta_local: f64,
    altup_num_inputs: usize,
    altup_active_idx: usize,
    altup_coef_clip: f64,
    hidden_size_per_layer_input: usize,
    vocab_size_per_layer_input: usize,
    laurel_rank: usize,
    layer_types: Vec<String>,
    intermediate_sizes: Vec<usize>,
}

impl Gemma3nExtraConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let query_pre_attn_scalar = cfg
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or((cfg.head_dim as f64).sqrt());

        let attn_logit_softcap = cfg
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64());

        let final_logit_softcap = cfg
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());

        let sliding_window_pattern = cfg
            .extra
            .get("sliding_window_pattern")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let rope_theta_local = cfg
            .extra
            .get("rope_local_base_freq")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);

        let altup_num_inputs = cfg
            .extra
            .get("altup_num_inputs")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let altup_active_idx = cfg
            .extra
            .get("altup_active_idx")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let altup_coef_clip = cfg
            .extra
            .get("altup_coef_clip")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let hidden_size_per_layer_input = cfg
            .extra
            .get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);

        let vocab_size_per_layer_input = cfg
            .extra
            .get("vocab_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);

        let laurel_rank = cfg
            .extra
            .get("laurel_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);

        let layer_types = cfg
            .extra
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("full_attention").to_string())
                    .collect()
            })
            .unwrap_or_else(|| {
                (0..cfg.num_hidden_layers)
                    .map(|i| {
                        if sliding_window_pattern > 0 && i.is_multiple_of(sliding_window_pattern) {
                            "sliding_attention".to_string()
                        } else {
                            "full_attention".to_string()
                        }
                    })
                    .collect()
            });

        let intermediate_sizes = cfg
            .extra
            .get("intermediate_size")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_u64().unwrap_or(cfg.intermediate_size as u64) as usize)
                    .collect()
            })
            .unwrap_or_else(|| vec![cfg.intermediate_size; cfg.num_hidden_layers]);

        Self {
            query_pre_attn_scalar,
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window_pattern,
            rope_theta_local,
            altup_num_inputs,
            altup_active_idx,
            altup_coef_clip,
            hidden_size_per_layer_input,
            vocab_size_per_layer_input,
            laurel_rank,
            layer_types,
            intermediate_sizes,
        }
    }

    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.layer_types.len() {
            self.layer_types[layer_idx] == "sliding_attention"
        } else if self.sliding_window_pattern == 0 {
            false
        } else {
            layer_idx.is_multiple_of(self.sliding_window_pattern)
        }
    }
}

// ─── AltUp ──────────────────────────────────────────────────────────────────
//
// Alternating Updates: learned routing between parallel sub-experts.
// The predict step modifies each sub-expert's input using learned coefficients,
// and the correct step propagates the activated expert's output to all.

struct Gemma3nAltUp {
    correction_coefs: candle_nn::Linear,
    prediction_coefs: candle_nn::Linear,
    modality_router: candle_nn::Linear,
    router_norm: Gemma3nRmsNorm,
    router_input_scale: f64,
    correct_output_scale: Tensor,
    altup_num_inputs: usize,
    altup_active_idx: usize,
    #[allow(dead_code)]
    altup_coef_clip: f64,
}

impl Gemma3nAltUp {
    fn new(
        hidden_size: usize,
        rms_norm_eps: f64,
        altup_num_inputs: usize,
        altup_coef_clip: f64,
        altup_active_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let correction_coefs = candle_nn::linear_no_bias(
            altup_num_inputs,
            altup_num_inputs,
            vb.pp("correction_coefs"),
        )?;
        let prediction_coefs = candle_nn::linear_no_bias(
            altup_num_inputs,
            altup_num_inputs * altup_num_inputs,
            vb.pp("prediction_coefs"),
        )?;
        let modality_router =
            candle_nn::linear_no_bias(hidden_size, altup_num_inputs, vb.pp("modality_router"))?;
        let router_norm = Gemma3nRmsNorm::new(hidden_size, rms_norm_eps, vb.pp("router_norm"))?;
        let router_input_scale = (hidden_size as f64).powf(-1.0);
        let correct_output_scale = vb.get(hidden_size, "correct_output_scale")?;

        Ok(Self {
            correction_coefs,
            prediction_coefs,
            modality_router,
            router_norm,
            router_input_scale,
            correct_output_scale,
            altup_num_inputs,
            altup_active_idx,
            altup_coef_clip,
        })
    }

    /// Compute router modalities: norm + scale + route + tanh
    fn compute_router_modalities(&self, x: &Tensor) -> Result<Tensor> {
        let router_inputs = (self.router_norm.forward(x)? * self.router_input_scale)?;
        let routed = self.modality_router.forward(&router_inputs)?;
        routed.to_dtype(DType::F32)?.tanh()?.to_dtype(x.dtype())
    }

    /// Scale corrected output by learned per-dimension scale
    fn scale_corrected_output(&self, corrected: &Tensor) -> Result<Tensor> {
        corrected.broadcast_mul(&self.correct_output_scale)
    }

    /// Predict step: routes hidden states through learned prediction coefficients.
    /// Input: Vec of [..., hidden_size] with altup_num_inputs entries
    /// Output: Vec of [..., hidden_size] with altup_num_inputs entries
    fn predict(&self, hidden_states: &[Tensor]) -> Result<Vec<Tensor>> {
        let n = self.altup_num_inputs;
        let active = &hidden_states[self.altup_active_idx];
        let orig_dims = active.dims().to_vec();
        let hidden_size = *orig_dims.last().unwrap();

        // Flatten to 2D: [total_tokens, hidden_size]
        let total_tokens: usize = orig_dims[..orig_dims.len() - 1].iter().product();
        let flat_states: Vec<Tensor> = hidden_states
            .iter()
            .map(|t| t.reshape((total_tokens, hidden_size)))
            .collect::<Result<_>>()?;
        let flat_active = &flat_states[self.altup_active_idx];

        // modalities: [total_tokens, n]
        let modalities = self.compute_router_modalities(flat_active)?;
        // all_coefs: [total_tokens, n*n]
        let all_coefs = self.prediction_coefs.forward(&modalities)?;

        // Reshape to [total_tokens, n, n] then transpose last two dims
        let all_coefs_t = all_coefs.reshape((total_tokens, n, n))?.transpose(1, 2)?;

        // Stack hidden_states: [total_tokens, hidden_size, n]
        let stacked = Tensor::stack(&flat_states, 2)?;

        // Matmul: [total_tokens, hidden_size, n] @ [total_tokens, n, n] = [total_tokens, hidden_size, n]
        let predictions = stacked.matmul(&all_coefs_t)?;

        // Add residual, unstack, restore original shape
        let mut result = Vec::with_capacity(n);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let pred_i = predictions.narrow(2, i, 1)?.squeeze(2)?;
            let res = (&pred_i + &flat_states[i])?;
            result.push(res.reshape(&orig_dims[..])?.contiguous()?);
        }

        Ok(result)
    }

    /// Correct step: propagates activated output back to all sub-experts.
    /// Input: predictions (Vec<Tensor>), activated tensor [..., hidden_size]
    /// Output: corrected Vec of [..., hidden_size]
    fn correct(&self, predictions: &[Tensor], activated: &Tensor) -> Result<Vec<Tensor>> {
        let n = self.altup_num_inputs;
        let orig_dims = activated.dims().to_vec();
        let hidden_size = *orig_dims.last().unwrap();
        let total_tokens: usize = orig_dims[..orig_dims.len() - 1].iter().product();

        // Flatten to 2D
        let flat_activated = activated.reshape((total_tokens, hidden_size))?;
        let flat_preds: Vec<Tensor> = predictions
            .iter()
            .map(|t| t.reshape((total_tokens, hidden_size)))
            .collect::<Result<_>>()?;

        // modalities: [total_tokens, n]
        let modalities = self.compute_router_modalities(&flat_activated)?;
        // innovation: activated - predictions[active_idx]
        let innovation = (&flat_activated - &flat_preds[self.altup_active_idx])?;

        // all_coefs: [total_tokens, n], then add 1.0
        let all_coefs = (self.correction_coefs.forward(&modalities)? + 1.0)?;

        let mut corrected = Vec::with_capacity(n);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            // coef_i: [total_tokens, 1]
            let coef_i = all_coefs.narrow(1, i, 1)?;
            // elementwise: innovation * coef_i + predictions[i]
            let c = (innovation.broadcast_mul(&coef_i)? + &flat_preds[i])?;
            corrected.push(c.reshape(&orig_dims[..])?.contiguous()?);
        }

        Ok(corrected)
    }
}

// ─── Laurel Block ───────────────────────────────────────────────────────────
//
// Learned Augmented Residual Layer: low-rank projection as an additional
// residual pathway parallel to attention.

struct Gemma3nLaurelBlock {
    linear_left: candle_nn::Linear,
    linear_right: candle_nn::Linear,
    post_laurel_norm: Gemma3nRmsNorm,
}

impl Gemma3nLaurelBlock {
    fn new(
        hidden_size: usize,
        laurel_rank: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_left =
            candle_nn::linear_no_bias(hidden_size, laurel_rank, vb.pp("linear_left"))?;
        let linear_right =
            candle_nn::linear_no_bias(laurel_rank, hidden_size, vb.pp("linear_right"))?;
        let post_laurel_norm =
            Gemma3nRmsNorm::new(hidden_size, rms_norm_eps, vb.pp("post_laurel_norm"))?;

        Ok(Self {
            linear_left,
            linear_right,
            post_laurel_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let laurel_x = self.linear_left.forward(x)?;
        let laurel_x = self.linear_right.forward(&laurel_x)?;
        let normed = self.post_laurel_norm.forward(&laurel_x)?;
        x + normed
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct Gemma3nAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    q_norm: Gemma3nRmsNorm,
    k_norm: Gemma3nRmsNorm,
    v_norm: UnweightedRmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_logit_softcap: Option<f64>,
    sliding_window: Option<usize>,
}

impl Gemma3nAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3nExtraConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        if world_size > 1 {
            if !num_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
                )));
            }
            if !num_kv_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})"
                )));
            }
        }

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

        let q_norm = Gemma3nRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Gemma3nRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let v_norm = UnweightedRmsNorm::new(cfg.rms_norm_eps);

        let is_local = extra_cfg.is_sliding_layer(layer_idx);
        let rope_theta = if is_local {
            extra_cfg.rope_theta_local
        } else {
            cfg.rope_theta
        };

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        let sliding_window = if is_local { cfg.sliding_window } else { None };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            rotary_emb,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            attn_logit_softcap: extra_cfg.attn_logit_softcap,
            sliding_window,
        })
    }

    /// Apply per-head RMSNorm: reshape [b, h, s, d] -> [b*h*s, d], norm, reshape back
    fn apply_per_head_norm(x: &Tensor, norm: &Gemma3nRmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let flat = x.reshape((b * h * s, d))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, h, s, d))
    }

    fn apply_per_head_norm_unweighted(x: &Tensor, norm: &UnweightedRmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let flat = x.reshape((b * h * s, d))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, h, s, d))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        _attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let device = xs.device();
        let dtype = xs.dtype();

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;
        let k = Self::apply_per_head_norm(&k, &self.k_norm)?;
        let v = Self::apply_per_head_norm_unweighted(&v, &self.v_norm)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Write new K/V to cache
        let k_for_cache = k.squeeze(0)?.contiguous()?;
        let v_for_cache = v.squeeze(0)?.contiguous()?;
        cache_engine
            .write(&k_for_cache, &v_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        let num_tokens = seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(block_table.block_ids(), num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        let kv_len = k_full.dim(2)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        // Gemma3n uses scale=1.0 (scaling handled by query_pre_attn_scalar in config)
        let mut attn_weights = q.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;

        if let Some(cap) = self.attn_logit_softcap {
            attn_weights = soft_cap(&attn_weights, cap)?;
        }

        let mask = if let Some(window_size) = self.sliding_window {
            sliding_window_mask(q_len, kv_len, seqlen_offset, window_size, dtype, device)?
        } else {
            crate::layers::causal_mask(q_len, seqlen_offset, dtype, device)?
        };

        attn_weights = attn_weights.broadcast_add(&mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_full)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output, tp_ctx)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let device = xs.device();
        let dtype = xs.dtype();

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;
        let k = Self::apply_per_head_norm(&k, &self.k_norm)?;
        let v = Self::apply_per_head_norm_unweighted(&v, &self.v_norm)?;

        let mut outputs = Vec::with_capacity(batch_size);
        let num_kv_groups = self.num_heads / self.num_kv_heads;

        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

            let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

            let k_for_cache = k_i.squeeze(0)?.contiguous()?;
            let v_for_cache = v_i.squeeze(0)?.contiguous()?;
            cache_engine
                .write(&k_for_cache, &v_for_cache, &seq.slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let kv_len = seq.seqlen_offset + 1;
            let (k_full, v_full) = cache_engine
                .read(&seq.block_ids, kv_len)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

            let k_full = repeat_kv(k_full, num_kv_groups)?;
            let v_full = repeat_kv(v_full, num_kv_groups)?;

            let mut attn_weights = q_i.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;

            if let Some(cap) = self.attn_logit_softcap {
                attn_weights = soft_cap(&attn_weights, cap)?;
            }

            if let Some(window_size) = self.sliding_window {
                let mask =
                    sliding_window_mask(1, kv_len, seq.seqlen_offset, window_size, dtype, device)?;
                attn_weights = attn_weights.broadcast_add(&mask)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v_full)?;
            let attn_output =
                attn_output
                    .transpose(1, 2)?
                    .reshape((1, 1, self.num_heads * self.head_dim))?;
            outputs.push(attn_output);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output, tp_ctx)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct Gemma3nDecoderLayer {
    altup: Gemma3nAltUp,
    self_attn: Gemma3nAttention,
    mlp: TpGeGluMlp,
    laurel: Gemma3nLaurelBlock,
    per_layer_input_gate: candle_nn::Linear,
    per_layer_projection: candle_nn::Linear,
    input_layernorm: Gemma3nRmsNorm,
    post_attention_layernorm: Gemma3nRmsNorm,
    pre_feedforward_layernorm: Gemma3nRmsNorm,
    post_feedforward_layernorm: Gemma3nRmsNorm,
    post_per_layer_input_norm: Gemma3nRmsNorm,
    altup_active_idx: usize,
}

impl Gemma3nDecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3nExtraConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let altup = Gemma3nAltUp::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            extra_cfg.altup_num_inputs,
            extra_cfg.altup_coef_clip,
            extra_cfg.altup_active_idx,
            vb.pp("altup"),
        )?;

        let self_attn =
            Gemma3nAttention::new_with_tp(cfg, extra_cfg, layer_idx, vb.pp("self_attn"), pg)?;

        let intermediate_size = if layer_idx < extra_cfg.intermediate_sizes.len() {
            extra_cfg.intermediate_sizes[layer_idx]
        } else {
            cfg.intermediate_size
        };

        let mlp = TpGeGluMlp::new(cfg.hidden_size, intermediate_size, vb.pp("mlp"), pg)?;

        let laurel = Gemma3nLaurelBlock::new(
            cfg.hidden_size,
            extra_cfg.laurel_rank,
            cfg.rms_norm_eps,
            vb.pp("laurel"),
        )?;

        let per_layer_input_gate = candle_nn::linear_no_bias(
            cfg.hidden_size,
            extra_cfg.hidden_size_per_layer_input,
            vb.pp("per_layer_input_gate"),
        )?;
        let per_layer_projection = candle_nn::linear_no_bias(
            extra_cfg.hidden_size_per_layer_input,
            cfg.hidden_size,
            vb.pp("per_layer_projection"),
        )?;

        let input_layernorm =
            Gemma3nRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Gemma3nRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = Gemma3nRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = Gemma3nRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_per_layer_input_norm = Gemma3nRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_per_layer_input_norm"),
        )?;

        Ok(Self {
            altup,
            self_attn,
            mlp,
            laurel,
            per_layer_input_gate,
            per_layer_projection,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_per_layer_input_norm,
            altup_active_idx: extra_cfg.altup_active_idx,
        })
    }

    /// Forward pass for a single sequence.
    /// hidden_states: Vec of [1, seq_len, hidden_size] with altup_num_inputs entries
    /// per_layer_input: [1, seq_len, hidden_size_per_layer_input]
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden_states: &[Tensor],
        per_layer_input: &Tensor,
        _attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Vec<Tensor>> {
        // AltUp predict
        let predictions = self.altup.predict(hidden_states)?;
        let active_prediction = &predictions[self.altup_active_idx];
        let active_prediction_normed = self.input_layernorm.forward(active_prediction)?;
        let laurel_output = self.laurel.forward(&active_prediction_normed)?;

        // Attention
        let attn = self.self_attn.forward(
            &active_prediction_normed,
            None,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let attn = self.post_attention_layernorm.forward(&attn)?;
        let attn_gated = (&attn + active_prediction)?;
        let sqrt2 = (2.0_f64).sqrt();
        let attn_laurel = ((&attn_gated + &laurel_output)? / sqrt2)?;

        // MLP
        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel)?;
        let attn_ffw = self.mlp.forward(&attn_norm, tp_ctx)?;
        let attn_ffw_norm = self.post_feedforward_layernorm.forward(&attn_ffw)?;
        let attn_ffw_laurel_gated = (&attn_laurel + &attn_ffw_norm)?;

        // AltUp correct
        let mut corrected = self.altup.correct(&predictions, &attn_ffw_laurel_gated)?;
        let first_prediction = &corrected[self.altup_active_idx];
        let scaled = self.altup.scale_corrected_output(first_prediction)?;

        // Per-layer input gating: gate -> gelu -> elementwise mul with per_layer_input -> project
        let gated = self.per_layer_input_gate.forward(&scaled)?;
        let gated = candle_nn::Activation::Gelu.forward(&gated)?;
        let gated = gated.broadcast_mul(per_layer_input)?;
        let projected = self.per_layer_projection.forward(&gated)?;
        let projected = self.post_per_layer_input_norm.forward(&projected)?;

        // Add projected to all non-active sub-experts
        for (i, c) in corrected.iter_mut().enumerate() {
            if i != self.altup_active_idx {
                *c = (c.clone() + &projected)?;
            }
        }

        Ok(corrected)
    }

    fn forward_decode_batch(
        &self,
        hidden_states: &[Tensor],
        per_layer_input: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Vec<Tensor>> {
        // AltUp predict
        let predictions = self.altup.predict(hidden_states)?;
        let active_prediction = &predictions[self.altup_active_idx];
        let active_prediction_normed = self.input_layernorm.forward(active_prediction)?;
        let laurel_output = self.laurel.forward(&active_prediction_normed)?;

        // Attention
        let attn = self.self_attn.forward_decode_batch(
            &active_prediction_normed,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let attn = self.post_attention_layernorm.forward(&attn)?;
        let attn_gated = (&attn + active_prediction)?;
        let sqrt2 = (2.0_f64).sqrt();
        let attn_laurel = ((&attn_gated + &laurel_output)? / sqrt2)?;

        // MLP
        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel)?;
        let attn_ffw = self.mlp.forward(&attn_norm, tp_ctx)?;
        let attn_ffw_norm = self.post_feedforward_layernorm.forward(&attn_ffw)?;
        let attn_ffw_laurel_gated = (&attn_laurel + &attn_ffw_norm)?;

        // AltUp correct
        let mut corrected = self.altup.correct(&predictions, &attn_ffw_laurel_gated)?;
        let first_prediction = &corrected[self.altup_active_idx];
        let scaled = self.altup.scale_corrected_output(first_prediction)?;

        // Per-layer input gating
        let gated = self.per_layer_input_gate.forward(&scaled)?;
        let gated = candle_nn::Activation::Gelu.forward(&gated)?;
        let gated = gated.broadcast_mul(per_layer_input)?;
        let projected = self.per_layer_projection.forward(&gated)?;
        let projected = self.post_per_layer_input_norm.forward(&projected)?;

        for (i, c) in corrected.iter_mut().enumerate() {
            if i != self.altup_active_idx {
                *c = (c.clone() + &projected)?;
            }
        }

        Ok(corrected)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Gemma3nForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Gemma3nDecoderLayer>,
    norm: Gemma3nRmsNorm,
    lm_head: TpLinear,
    // AltUp embedding projections (for non-active sub-experts)
    altup_projections: Vec<candle_nn::Linear>,
    // AltUp unembed projections
    altup_unembed_projections: Vec<candle_nn::Linear>,
    // Per-layer input projection from hidden_size to num_layers * hidden_size_per_layer_input
    per_layer_model_projection: candle_nn::Linear,
    per_layer_projection_norm: Gemma3nRmsNorm,
    hidden_size: usize,
    num_hidden_layers: usize,
    altup_num_inputs: usize,
    _altup_active_idx: usize,
    hidden_size_per_layer_input: usize,
    final_logit_softcap: Option<f64>,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Gemma3nForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let extra_cfg = Gemma3nExtraConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        // AltUp embedding projections: project hidden_states[i] for i > 0
        let mut altup_projections = Vec::new();
        let vb_altup_proj = vb_m.pp("self_decoder").pp("altup_projections");
        for i in 0..(extra_cfg.altup_num_inputs - 1) {
            let proj =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb_altup_proj.pp(i))?;
            altup_projections.push(proj);
        }

        // AltUp unembed projections
        let mut altup_unembed_projections = Vec::new();
        let vb_unembed = vb_m.pp("altup_unembed_projections");
        for i in 0..(extra_cfg.altup_num_inputs - 1) {
            let proj =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb_unembed.pp(i))?;
            altup_unembed_projections.push(proj);
        }

        // Per-layer model projection
        let per_layer_model_projection = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_hidden_layers * extra_cfg.hidden_size_per_layer_input,
            vb_m.pp("self_decoder").pp("per_layer_model_projection"),
        )?;

        let per_layer_projection_norm = Gemma3nRmsNorm::new(
            extra_cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
            vb_m.pp("self_decoder").pp("per_layer_projection_norm"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma3nDecoderLayer::new_with_tp(
                cfg,
                &extra_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = Gemma3nRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            altup_projections,
            altup_unembed_projections,
            per_layer_model_projection,
            per_layer_projection_norm,
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            altup_num_inputs: extra_cfg.altup_num_inputs,
            _altup_active_idx: extra_cfg.altup_active_idx,
            hidden_size_per_layer_input: extra_cfg.hidden_size_per_layer_input,
            final_logit_softcap: extra_cfg.final_logit_softcap,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// AltUp embedding: project hidden_states_0 into altup_num_inputs copies,
    /// with magnitude normalization for non-active sub-experts.
    fn altup_embed(&self, hidden_states_0: &Tensor) -> Result<Vec<Tensor>> {
        let n = self.altup_num_inputs;
        let mut hidden_states = vec![hidden_states_0.clone(); n];

        // Target magnitude from first (active) sub-expert
        let target_magnitude = hidden_states_0.sqr()?.mean_keepdim(D::Minus1)?.sqrt()?;

        let eps = Tensor::new(&[f32::MIN_POSITIVE], hidden_states_0.device())?
            .to_dtype(hidden_states_0.dtype())?;

        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            hidden_states[i] = self.altup_projections[i - 1].forward(&hidden_states[i])?;
            let new_magnitude = hidden_states[i].sqr()?.mean_keepdim(D::Minus1)?.sqrt()?;
            let scale = target_magnitude.broadcast_div(&new_magnitude.broadcast_maximum(&eps)?)?;
            hidden_states[i] = hidden_states[i].broadcast_mul(&scale)?;
        }

        Ok(hidden_states)
    }

    /// AltUp unembedding: project back and average all sub-experts.
    fn altup_unembed(&self, hidden_states: &mut [Tensor]) -> Result<Tensor> {
        let n = self.altup_num_inputs;

        let target_magnitude = hidden_states[0].sqr()?.mean_keepdim(D::Minus1)?.sqrt()?;

        let eps = Tensor::new(&[f32::MIN_POSITIVE], hidden_states[0].device())?
            .to_dtype(hidden_states[0].dtype())?;

        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            hidden_states[i] = self.altup_unembed_projections[i - 1].forward(&hidden_states[i])?;
            let new_magnitude = hidden_states[i].sqr()?.mean_keepdim(D::Minus1)?.sqrt()?;
            let scale = target_magnitude.broadcast_div(&new_magnitude.broadcast_maximum(&eps)?)?;
            hidden_states[i] = hidden_states[i].broadcast_mul(&scale)?;
        }

        // Average all sub-experts
        let stacked = Tensor::stack(hidden_states, 0)?;
        stacked.mean(0)
    }

    /// Compute per-layer inputs from hidden state projection
    fn get_per_layer_inputs(&self, hidden_states_0: &Tensor) -> Result<Tensor> {
        let projected = self.per_layer_model_projection.forward(hidden_states_0)?;
        // Reshape to [..., num_layers, hidden_size_per_layer_input]
        let shape = hidden_states_0.dims();
        let batch_seq: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let mut new_shape = batch_seq;
        new_shape.push(self.num_hidden_layers);
        new_shape.push(self.hidden_size_per_layer_input);
        let projected = projected.reshape(&new_shape[..])?;

        // Normalize each per-layer slice
        let num_layers = self.num_hidden_layers;
        let mut normed_slices = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let slice = projected.narrow(projected.dims().len() - 2, l, 1)?;
            let slice = slice.squeeze(projected.dims().len() - 2)?;
            let normed = self.per_layer_projection_norm.forward(&slice)?;
            normed_slices.push(normed.unsqueeze(projected.dims().len() - 2)?);
        }
        Tensor::cat(&normed_slices, projected.dims().len() - 2)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let normalizer = (self.hidden_size as f64).sqrt();
        let hidden_states_0 = (self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer)?;

        // Get per-layer inputs
        let per_layer_inputs = self.get_per_layer_inputs(&hidden_states_0)?;

        // AltUp embed: create altup_num_inputs copies with magnitude normalization
        let mut hidden_states = self.altup_embed(&hidden_states_0)?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Extract per-layer input for this layer: [..., layer_idx, hidden_size_per_layer_input]
            let pli = per_layer_inputs.narrow(per_layer_inputs.dims().len() - 2, layer_idx, 1)?;
            let pli = pli.squeeze(per_layer_inputs.dims().len() - 2)?;

            hidden_states = layer.forward(
                &hidden_states,
                &pli,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        // AltUp unembed
        let hidden_states = self.altup_unembed(&mut hidden_states)?;
        let hidden_states = self.norm.forward(&hidden_states)?;

        let mut logits = self.lm_head.forward(&hidden_states, &self.tp_ctx)?;
        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Gemma3nForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Gemma3nForCausalLM::forward(
            self,
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
        let normalizer = (self.hidden_size as f64).sqrt();
        let hidden_states_0 = (self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer)?;

        let per_layer_inputs = self.get_per_layer_inputs(&hidden_states_0)?;
        let mut hidden_states = self.altup_embed(&hidden_states_0)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = per_layer_inputs.narrow(per_layer_inputs.dims().len() - 2, layer_idx, 1)?;
            let pli = pli.squeeze(per_layer_inputs.dims().len() - 2)?;

            hidden_states = layer.forward_decode_batch(
                &hidden_states,
                &pli,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let hidden_states = self.altup_unembed(&mut hidden_states)?;
        let hidden_states = self.norm.forward(&hidden_states)?;

        let mut logits = self.lm_head.forward(&hidden_states, &self.tp_ctx)?;
        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::json!(256.0),
        );
        extra.insert("altup_num_inputs".to_string(), serde_json::json!(2));
        extra.insert("altup_active_idx".to_string(), serde_json::json!(0));
        extra.insert("altup_coef_clip".to_string(), serde_json::json!(1.0));
        extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(32),
        );
        extra.insert("laurel_rank".to_string(), serde_json::json!(16));
        extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));
        extra.insert(
            "final_logit_softcapping".to_string(),
            serde_json::json!(30.0),
        );

        ModelConfig {
            architectures: vec!["Gemma3nForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_gemma3n_extra_config_parsing() {
        let cfg = test_config();
        let extra_cfg = Gemma3nExtraConfig::from_model_config(&cfg);

        assert_eq!(extra_cfg.altup_num_inputs, 2);
        assert_eq!(extra_cfg.altup_active_idx, 0);
        assert_eq!(extra_cfg.hidden_size_per_layer_input, 32);
        assert_eq!(extra_cfg.laurel_rank, 16);
        assert_eq!(extra_cfg.final_logit_softcap, Some(30.0));

        assert!(extra_cfg.is_sliding_layer(0));
        assert!(!extra_cfg.is_sliding_layer(1));
    }

    #[test]
    fn test_gemma3n_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Gemma3nForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Gemma3nForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.altup_num_inputs, 2);
        assert_eq!(model.altup_projections.len(), 1);
        assert_eq!(model.altup_unembed_projections.len(), 1);
    }

    #[test]
    fn test_gemma3n_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3nForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 3;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_gemma3n_single_token() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3nForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_gemma3n_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3nForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = model
            .forward(
                &next_token,
                3,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_gemma3n_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3nForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_altup_predict_correct_shapes() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let altup = Gemma3nAltUp::new(64, 1e-6, 2, 1.0, 0, vb.pp("altup")).expect("altup");

        let h0 = Tensor::zeros((4, 64), DType::F32, &device).expect("h0");
        let h1 = Tensor::zeros((4, 64), DType::F32, &device).expect("h1");

        let predictions = altup.predict(&[h0, h1]).expect("predict");
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].dims(), &[4, 64]);
        assert_eq!(predictions[1].dims(), &[4, 64]);

        let activated = Tensor::zeros((4, 64), DType::F32, &device).expect("activated");
        let corrected = altup.correct(&predictions, &activated).expect("correct");
        assert_eq!(corrected.len(), 2);
        assert_eq!(corrected[0].dims(), &[4, 64]);
    }

    #[test]
    fn test_laurel_block() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let laurel = Gemma3nLaurelBlock::new(64, 16, 1e-6, vb.pp("laurel")).expect("laurel");

        let x = Tensor::randn(0.0f32, 1.0, (2, 64), &device).expect("input");
        let out = laurel.forward(&x).expect("forward");
        assert_eq!(out.dims(), &[2, 64]);
    }

    #[test]
    fn test_unweighted_rms_norm() {
        let norm = UnweightedRmsNorm::new(1e-6);
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 8), &device).expect("input");
        let out = norm.forward(&x).expect("forward");
        assert_eq!(out.dims(), &[2, 8]);
    }

    #[test]
    fn test_gemma3n_rms_norm_offset() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let norm = Gemma3nRmsNorm::new(8, 1e-6, vb.pp("norm")).expect("norm");

        // With zero weights, scale = (0 + 1) = 1, so output should be normalized
        let x = Tensor::ones((1, 8), DType::F32, &device).expect("input");
        let out = norm.forward(&x).expect("forward");
        assert_eq!(out.dims(), &[1, 8]);
    }
}
