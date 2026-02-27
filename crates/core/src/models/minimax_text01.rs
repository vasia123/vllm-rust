//! MiniMaxText01 (hybrid linear attention + standard attention) model.
//!
//! MiniMaxText01 is a hybrid architecture where some layers use linear attention
//! (SSM-like recurrence) and others use standard multi-head attention. Some layers
//! also use Mixture-of-Experts (MoE) with optional shared experts.
//!
//! Architecture aliases:
//! - MiniMaxForCausalLM -> MiniMaxText01ForCausalLM
//! - MiniMaxM1ForCausalLM -> MiniMaxText01ForCausalLM
//! - MiniMaxText01ForCausalLM
//!
//! Architecture:
//! ```text
//! Embedding -> [DecoderLayer x N] -> RMSNorm -> LM Head
//!
//! DecoderLayer (linear attention variant, attention_type=0):
//!   RMSNorm -> LinearAttention -> alpha/beta scale -> RMSNorm -> MLP/MoE -> scale
//!
//! DecoderLayer (full attention variant, attention_type=1):
//!   RMSNorm -> SelfAttention(RoPE) -> alpha/beta scale -> RMSNorm -> MLP/MoE -> scale
//! ```
//!
//! Config keys from extra:
//! - `attn_type_list` or `decoder_attention_types` or `layer_types`: per-layer attention type
//! - `num_local_experts`: number of MoE experts (can be int or list)
//! - `num_experts_per_tok`: top-k experts
//! - `shared_intermediate_size`: shared expert intermediate size (0 = no shared expert)
//! - `shared_moe_mode`: "softmax" or "sigmoid" for shared expert mixing
//! - `layernorm_linear_attention_alpha`, `layernorm_linear_attention_beta`: scaling for linear attn layers
//! - `layernorm_full_attention_alpha`, `layernorm_full_attention_beta`: scaling for full attn layers
//! - `layernorm_mlp_alpha`, `layernorm_mlp_beta`: scaling for MLP output
//! - `postnorm`: whether to use post-norm (default false)

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Parsed per-layer config from the HuggingFace MiniMax config.
pub(crate) struct MiniMaxText01Config {
    /// Per-layer attention type: 0 = linear, 1 = full attention
    decoder_attention_types: Vec<u8>,
    /// Per-layer number of experts (1 = dense MLP)
    num_local_experts: Vec<usize>,
    /// Top-k experts per token
    num_experts_per_tok: usize,
    /// Per-layer shared expert intermediate size (0 = no shared expert)
    shared_intermediate_sizes: Vec<usize>,
    /// Shared MoE mode: "softmax" or "sigmoid"
    shared_moe_mode: String,
    /// Scaling factors for attention residual connections
    linear_attn_alpha: f64,
    linear_attn_beta: f64,
    full_attn_alpha: f64,
    full_attn_beta: f64,
    /// Scaling factors for MLP residual connections
    mlp_alpha: f64,
    mlp_beta: f64,
    /// Whether to use post-norm (default false)
    postnorm: bool,
}

impl MiniMaxText01Config {
    pub(crate) fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_layers = cfg.num_hidden_layers;

        // Parse attention type list
        let decoder_attention_types = Self::parse_attention_types(cfg, num_layers);

        // Parse expert counts per layer
        let num_local_experts = Self::parse_expert_counts(cfg, num_layers);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        // Parse shared intermediate sizes
        let shared_intermediate_sizes = Self::parse_shared_intermediate(cfg, num_layers);

        let shared_moe_mode = cfg
            .extra
            .get("shared_moe_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
            .to_string();

        let linear_attn_alpha = cfg
            .extra
            .get("layernorm_linear_attention_alpha")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("linear_attn_alpha_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let linear_attn_beta = cfg
            .extra
            .get("layernorm_linear_attention_beta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("linear_attn_beta_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let full_attn_alpha = cfg
            .extra
            .get("layernorm_full_attention_alpha")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("full_attn_alpha_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let full_attn_beta = cfg
            .extra
            .get("layernorm_full_attention_beta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("full_attn_beta_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let mlp_alpha = cfg
            .extra
            .get("layernorm_mlp_alpha")
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.extra.get("mlp_alpha_factor").and_then(|v| v.as_f64()))
            .unwrap_or(1.0);

        let mlp_beta = cfg
            .extra
            .get("layernorm_mlp_beta")
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.extra.get("mlp_beta_factor").and_then(|v| v.as_f64()))
            .unwrap_or(1.0);

        let postnorm = cfg
            .extra
            .get("postnorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            decoder_attention_types,
            num_local_experts,
            num_experts_per_tok,
            shared_intermediate_sizes,
            shared_moe_mode,
            linear_attn_alpha,
            linear_attn_beta,
            full_attn_alpha,
            full_attn_beta,
            mlp_alpha,
            mlp_beta,
            postnorm,
        }
    }

    fn parse_attention_types(cfg: &ModelConfig, num_layers: usize) -> Vec<u8> {
        // Try attn_type_list first
        if let Some(list) = cfg.extra.get("attn_type_list").and_then(|v| v.as_array()) {
            return list.iter().map(|v| v.as_u64().unwrap_or(1) as u8).collect();
        }

        // Try decoder_attention_types
        if let Some(list) = cfg
            .extra
            .get("decoder_attention_types")
            .and_then(|v| v.as_array())
        {
            return list.iter().map(|v| v.as_u64().unwrap_or(1) as u8).collect();
        }

        // Try layer_types (HF format)
        if let Some(list) = cfg.extra.get("layer_types").and_then(|v| v.as_array()) {
            return list
                .iter()
                .map(|v| match v.as_str() {
                    Some("linear_attention") => 0,
                    Some("full_attention") => 1,
                    _ => 1,
                })
                .collect();
        }

        // Default: all full attention
        vec![1; num_layers]
    }

    fn parse_expert_counts(cfg: &ModelConfig, num_layers: usize) -> Vec<usize> {
        if let Some(list) = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_array())
        {
            return list
                .iter()
                .map(|v| v.as_u64().unwrap_or(1) as usize)
                .collect();
        }

        if let Some(n) = cfg.extra.get("num_local_experts").and_then(|v| v.as_u64()) {
            return vec![n as usize; num_layers];
        }

        vec![1; num_layers]
    }

    fn parse_shared_intermediate(cfg: &ModelConfig, num_layers: usize) -> Vec<usize> {
        if let Some(list) = cfg
            .extra
            .get("shared_intermediate_size")
            .and_then(|v| v.as_array())
        {
            let mut sizes: Vec<usize> = list
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            sizes.resize(num_layers, 0);
            return sizes;
        }

        if let Some(n) = cfg
            .extra
            .get("shared_intermediate_size")
            .and_then(|v| v.as_u64())
        {
            return vec![n as usize; num_layers];
        }

        vec![0; num_layers]
    }

    pub(crate) fn attn_alpha(&self, layer_idx: usize) -> f64 {
        if self.decoder_attention_types[layer_idx] == 0 {
            self.linear_attn_alpha
        } else {
            self.full_attn_alpha
        }
    }

    pub(crate) fn attn_beta(&self, layer_idx: usize) -> f64 {
        if self.decoder_attention_types[layer_idx] == 0 {
            self.linear_attn_beta
        } else {
            self.full_attn_beta
        }
    }

    pub(crate) fn is_linear_attention(&self, layer_idx: usize) -> bool {
        self.decoder_attention_types
            .get(layer_idx)
            .copied()
            .unwrap_or(1)
            == 0
    }

    pub(crate) fn expert_count(&self, layer_idx: usize) -> usize {
        self.num_local_experts.get(layer_idx).copied().unwrap_or(1)
    }

    pub(crate) fn shared_intermediate(&self, layer_idx: usize) -> usize {
        self.shared_intermediate_sizes
            .get(layer_idx)
            .copied()
            .unwrap_or(0)
    }

    /// Count the number of linear-attention layers (used for slope-rate scaling).
    pub(crate) fn num_linear_layers(&self) -> usize {
        self.decoder_attention_types
            .iter()
            .filter(|&&t| t == 0)
            .count()
            .max(1)
    }

    #[cfg(test)]
    fn num_attention_layers(&self) -> usize {
        self.decoder_attention_types
            .iter()
            .filter(|&&t| t == 1)
            .count()
    }
}

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct MiniMaxText01Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
}

impl MiniMaxText01Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // Merged gate+up: hidden_size -> 2*intermediate_size
        let gate_up_proj =
            linear_no_bias(hidden_size, 2 * intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let hidden = gate.mul(&chunks[1])?;
        self.down_proj.forward(&hidden)
    }
}

// ─── MoE Layer ──────────────────────────────────────────────────────────────

struct MiniMaxText01MoE {
    gate: Linear,
    experts: Vec<MiniMaxText01MoEExpert>,
    num_experts: usize,
    top_k: usize,
}

struct MiniMaxText01MoEExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MiniMaxText01MoEExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("w1"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("w2"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("w3"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

impl MiniMaxText01MoE {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(MiniMaxText01MoEExpert::new(
                hidden_size,
                intermediate_size,
                vb_experts.pp(i),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Router logits in FP32
        let router_logits = self
            .gate
            .forward(&xs_2d.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let routing_data: Vec<f32> = routing_weights
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let flat_data: Vec<f32> = xs_2d.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut output_data = vec![0.0f32; num_tokens * hidden_dim];

        for token_idx in 0..num_tokens {
            let weights =
                &routing_data[token_idx * self.num_experts..(token_idx + 1) * self.num_experts];

            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Renormalize top-k weights
            let top_sum: f32 = indexed[..self.top_k].iter().map(|(_, w)| w).sum();

            let token_input = Tensor::from_vec(
                flat_data[token_idx * hidden_dim..(token_idx + 1) * hidden_dim].to_vec(),
                (1, hidden_dim),
                xs.device(),
            )?;

            for &(expert_idx, weight) in indexed[..self.top_k].iter() {
                let norm_weight = if top_sum > 0.0 {
                    weight / top_sum
                } else {
                    1.0 / self.top_k as f32
                };
                let expert_out = self.experts[expert_idx].forward(&token_input)?;
                let expert_data: Vec<f32> =
                    expert_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for j in 0..hidden_dim {
                    output_data[token_idx * hidden_dim + j] += norm_weight * expert_data[j];
                }
            }
        }

        Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&orig_shape),
            xs.device(),
        )?
        .to_dtype(xs.dtype())
    }
}

// ─── Linear Attention (Lightning Attention recurrence) ───────────────────────
//
// Implements the MiniMaxText01 linear attention layer using the lightning
// attention recurrence formula from the reference Python implementation.
//
// Algorithm:
//   qkv = SiLU(qkv_proj(x))              [B, L, H, D] each
//   kv_outer = k ⊗ v                      [B, H, D, D]
//   state_t = exp(-slope_h) * state_{t-1} + kv_outer_t
//   out_t = q_t @ state_t                 [B, H, D]
//   hidden = norm(out) * sigmoid(gate)
//   return out_proj(hidden)
//
// State shape per sequence: [H, D, D]
// Decode state is stored per request_id in a Mutex<HashMap>.

/// ALiBi-style slope rates for the linear attention decay.
/// Returns a Vec of length `n` following the reference ALiBi geometric series.
fn build_slope_rates(n: usize) -> Vec<f32> {
    fn slopes_pow2(n: usize) -> Vec<f32> {
        // start = 2^(-(2^(-(log2(n)-3))))
        let log2n = (n as f32).log2();
        let exp_inner = -(log2n - 3.0);
        let start = 2.0f32.powf(-(2.0f32.powf(exp_inner)));
        let ratio = start;
        (0..n).map(|i| start * ratio.powi(i as i32)).collect()
    }

    fn slopes_recursive(n: usize) -> Vec<f32> {
        if n.is_power_of_two() {
            return slopes_pow2(n);
        }
        // Interpolate: take power-of-2 floor, then even entries from 2x
        let closest = n.next_power_of_two() >> 1; // largest power_of_2 < n
        let mut base = slopes_pow2(closest);
        let extended: Vec<f32> = slopes_recursive(closest * 2)
            .into_iter()
            .step_by(2)
            .take(n - closest)
            .collect();
        base.extend(extended);
        base
    }

    slopes_recursive(n)
}

struct MiniMaxText01LinearAttention {
    qkv_proj: Linear,
    output_gate: Linear,
    out_proj: Linear,
    norm: RmsNorm,
    /// Per-head decay: exp(-slope_rate[h]), shape [H].
    decay: Vec<f32>,
    num_heads: usize,
    head_dim: usize,
    hidden_inner_size: usize,
    /// Per-sequence linear attention state: request_id → Tensor [H, D, D].
    states: Mutex<HashMap<u64, Tensor>>,
}

impl MiniMaxText01LinearAttention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_linear_layers: usize,
        linear_layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_inner_size = num_heads * head_dim;

        let qkv_proj = linear_no_bias(hidden_size, hidden_inner_size * 3, vb.pp("qkv_proj"))?;
        let output_gate = linear_no_bias(hidden_size, hidden_inner_size, vb.pp("output_gate"))?;
        let out_proj = linear_no_bias(hidden_inner_size, hidden_size, vb.pp("out_proj"))?;
        let norm = rms_norm(hidden_inner_size, 1e-5, vb.pp("norm"))?;

        // Layer-scaled slope rates: reference scales by (1 - idx/(n-1) + 1e-5)
        let base_slopes = build_slope_rates(num_heads);
        let scale = if num_linear_layers <= 1 {
            1.0f32 + 1e-5
        } else {
            1.0 - linear_layer_idx as f32 / (num_linear_layers as f32 - 1.0) + 1e-5
        };
        let decay: Vec<f32> = base_slopes.iter().map(|&s| (-s * scale).exp()).collect();

        Ok(Self {
            qkv_proj,
            output_gate,
            out_proj,
            norm,
            decay,
            num_heads,
            head_dim,
            hidden_inner_size,
            states: Mutex::new(HashMap::new()),
        })
    }

    /// Project + SiLU + split into q, k, v.
    ///
    /// Input: `[B, L, hidden_size]`
    /// Output: three tensors each of shape `[B, L, H, D]` in F32.
    fn qkv_split(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = xs.dims();
        let b = dims[0];
        let l = dims[1];

        let qkv = self.qkv_proj.forward(xs)?;
        // Cast to F32 before SiLU for numerical stability (matches reference).
        let qkv = qkv.to_dtype(DType::F32)?;
        let qkv = candle_nn::ops::silu(&qkv)?;

        let q = qkv.narrow(2, 0, self.hidden_inner_size)?.reshape((
            b,
            l,
            self.num_heads,
            self.head_dim,
        ))?;
        let k = qkv
            .narrow(2, self.hidden_inner_size, self.hidden_inner_size)?
            .reshape((b, l, self.num_heads, self.head_dim))?;
        let v = qkv
            .narrow(2, 2 * self.hidden_inner_size, self.hidden_inner_size)?
            .reshape((b, l, self.num_heads, self.head_dim))?;

        Ok((q, k, v))
    }

    /// Build the decay tensor `[1, H, 1, 1]` on the target device.
    fn decay_tensor(&self, device: &Device) -> Result<Tensor> {
        Tensor::from_slice(&self.decay, (1, self.num_heads, 1, 1), device)
    }

    /// Apply gate + norm + out_proj and return `[B, L, hidden_size]`.
    fn apply_gate_and_project(&self, xs: &Tensor, hidden: Tensor) -> Result<Tensor> {
        let hidden = self.norm.forward(&hidden)?;
        let gate = self.output_gate.forward(xs)?.to_dtype(DType::F32)?;
        let gate = candle_nn::ops::sigmoid(&gate)?;
        let mixed = hidden.broadcast_mul(&gate)?;
        let mixed = mixed.to_dtype(xs.dtype())?;
        self.out_proj.forward(&mixed)
    }

    /// Prefill forward: runs the linear recurrence from zero state over all
    /// timesteps.  State is NOT persisted — each prefill starts from zero.
    ///
    /// `xs`: `[B, L, hidden_size]`, returns `[B, L, hidden_size]`.
    fn forward_prefill(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let (b, l) = (dims[0], dims[1]);
        let (q, k, v) = self.qkv_split(xs)?;
        // q/k/v: [B, L, H, D]

        let dev = xs.device();
        let decay = self.decay_tensor(dev)?; // [1, H, 1, 1]

        // Zero state: [B, H, D, D]
        let mut state = Tensor::zeros(
            (b, self.num_heads, self.head_dim, self.head_dim),
            DType::F32,
            dev,
        )?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(l);
        for t in 0..l {
            // Slice time step t: [B, H, D]
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;

            // Outer product kv_t: [B, H, D, 1] × [B, H, 1, D] → [B, H, D, D]
            let kv_t = k_t.unsqueeze(3)?.broadcast_mul(&v_t.unsqueeze(2)?)?;

            // State update: state = decay * state + kv_t
            state = (state.broadcast_mul(&decay)? + kv_t)?;

            // out_t = q_t @ state: [B, H, 1, D] → [B, H, D]
            let out_t = q_t.unsqueeze(2)?.matmul(&state)?.squeeze(2)?;
            outputs.push(out_t);
        }

        // Stack: list of [B, H, D] → [B, L, H, D] → [B, L, H*D]
        let hidden = Tensor::stack(&outputs, 1)?.reshape((b, l, self.hidden_inner_size))?;

        // xs: [B, L, hidden_size] for gate projection
        self.apply_gate_and_project(xs, hidden)
    }

    /// Decode forward for a batch of sequences, one token per sequence.
    ///
    /// Reads each sequence's state from the internal map (zero if absent),
    /// runs one recurrence step, writes the updated state back.
    ///
    /// `xs`: `[B, 1, hidden_size]` — one token per sequence (L=1).
    /// Returns `[B, 1, hidden_size]`.
    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
    ) -> Result<Tensor> {
        // xs: [B, 1, hidden_size]
        let batch = sequences.len();
        let dev = xs.device();

        let (q, k, v) = self.qkv_split(xs)?;
        // q/k/v: [B, 1, H, D] — squeeze the L=1 dim
        let q = q.squeeze(1)?; // [B, H, D]
        let k = k.squeeze(1)?;
        let v = v.squeeze(1)?;

        let decay = self.decay_tensor(dev)?; // [1, H, 1, 1]

        // Gather per-sequence states: each [H, D, D]
        let per_seq_states: Vec<Tensor> = {
            let guard = self.states.lock().unwrap();
            sequences
                .iter()
                .map(|seq| -> Result<Tensor> {
                    match guard.get(&seq.request_id) {
                        Some(t) => t.to_device(dev),
                        None => Tensor::zeros(
                            (self.num_heads, self.head_dim, self.head_dim),
                            DType::F32,
                            dev,
                        ),
                    }
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Stack states: [B, H, D, D]
        let state_batch = Tensor::stack(&per_seq_states, 0)?;

        // kv outer product: k[B,H,D,1] × v[B,H,1,D] → [B, H, D, D]
        let kv = k.unsqueeze(3)?.broadcast_mul(&v.unsqueeze(2)?)?;

        // State update: [B, H, D, D]
        let new_state = (state_batch.broadcast_mul(&decay)? + kv)?;

        // Output: q[B, H, 1, D] @ state[B, H, D, D] → [B, H, 1, D] → [B, H, D]
        let out = q.unsqueeze(2)?.matmul(&new_state)?.squeeze(2)?;

        // Persist updated states.
        {
            let mut guard = self.states.lock().unwrap();
            for (i, seq) in sequences.iter().enumerate() {
                let s = new_state.narrow(0, i, 1)?.squeeze(0)?;
                guard.insert(seq.request_id, s);
            }
        }

        // Reshape: [B, H, D] → [B, H*D] → [B, 1, H*D] for gate projection
        let hidden = out.reshape((batch, self.hidden_inner_size))?.unsqueeze(1)?; // [B, 1, H*D]

        // apply_gate_and_project expects xs: [B, 1, hidden_size]
        self.apply_gate_and_project(xs, hidden)
        // Returns [B, 1, hidden_size]
    }

    /// Free the stored state for a completed sequence.
    #[allow(dead_code)]
    fn free_state(&self, request_id: u64) {
        self.states.lock().unwrap().remove(&request_id);
    }

    /// Number of sequences with live state (test helper).
    #[cfg(test)]
    fn state_count(&self) -> usize {
        self.states.lock().unwrap().len()
    }
}

// ─── Standard Attention ─────────────────────────────────────────────────────

struct MiniMaxText01Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl MiniMaxText01Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let total_qkv = q_size + 2 * kv_size;

        let qkv_proj = linear_no_bias(cfg.hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        let attn_output = paged_attention(
            &q,
            &k,
            &v,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )?;

        self.o_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

            let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

            let attn_out = paged_attention(
                &q_i,
                &k_i,
                &v_i,
                None,
                seq.seqlen_offset,
                cache_engine,
                &seq.block_ids,
                &seq.slot_mapping,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )?;
            outputs.push(attn_out);
        }

        Tensor::cat(&outputs, 0)
    }
}

// ─── Attention Variant ──────────────────────────────────────────────────────

enum AttentionVariant {
    Linear(MiniMaxText01LinearAttention),
    Full(MiniMaxText01Attention),
}

// ─── Feed-forward Variant ───────────────────────────────────────────────────

enum FfnVariant {
    Dense(MiniMaxText01Mlp),
    MoE(MiniMaxText01MoE),
}

struct SharedExpert {
    mlp: MiniMaxText01Mlp,
    coefficient: Linear,
    mode: String,
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

pub(crate) struct MiniMaxText01DecoderLayer {
    self_attn: AttentionVariant,
    ffn: FfnVariant,
    shared_expert: Option<SharedExpert>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn_alpha: f64,
    attn_beta: f64,
    mlp_alpha: f64,
    mlp_beta: f64,
    postnorm: bool,
    /// Which KV cache layer index (only for full attention layers)
    cache_layer_idx: Option<usize>,
}

impl MiniMaxText01DecoderLayer {
    pub(crate) fn new(
        cfg: &ModelConfig,
        minimax_cfg: &MiniMaxText01Config,
        layer_idx: usize,
        cache_layer_idx: Option<usize>,
        num_linear_layers: usize,
        linear_layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_linear = minimax_cfg.is_linear_attention(layer_idx);

        let self_attn = if is_linear {
            AttentionVariant::Linear(MiniMaxText01LinearAttention::new(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.head_dim,
                num_linear_layers,
                linear_layer_idx,
                vb.pp("self_attn"),
            )?)
        } else {
            AttentionVariant::Full(MiniMaxText01Attention::new(cfg, vb.pp("self_attn"))?)
        };

        let expert_count = minimax_cfg.expert_count(layer_idx);
        let ffn = if expert_count > 1 {
            FfnVariant::MoE(MiniMaxText01MoE::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                expert_count,
                minimax_cfg.num_experts_per_tok,
                vb.pp("block_sparse_moe"),
            )?)
        } else {
            FfnVariant::Dense(MiniMaxText01Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
            )?)
        };

        let shared_size = minimax_cfg.shared_intermediate(layer_idx);
        let shared_expert = if shared_size > 0 {
            let shared_mlp =
                MiniMaxText01Mlp::new(cfg.hidden_size, shared_size, vb.pp("shared_mlp"))?;
            let coefficient = linear_no_bias(cfg.hidden_size, 1, vb.pp("coefficient"))?;
            Some(SharedExpert {
                mlp: shared_mlp,
                coefficient,
                mode: minimax_cfg.shared_moe_mode.clone(),
            })
        } else {
            None
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            ffn,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            attn_alpha: minimax_cfg.attn_alpha(layer_idx),
            attn_beta: minimax_cfg.attn_beta(layer_idx),
            mlp_alpha: minimax_cfg.mlp_alpha,
            mlp_beta: minimax_cfg.mlp_beta,
            postnorm: minimax_cfg.postnorm,
            cache_layer_idx,
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(hidden_states)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            hidden_states.clone()
        };

        let attn_output = match &self.self_attn {
            AttentionVariant::Linear(lin_attn) => lin_attn.forward_prefill(&layernorm_output)?,
            AttentionVariant::Full(attn) => {
                let cache_idx = self
                    .cache_layer_idx
                    .expect("full attention layer should have cache index");
                attn.forward(
                    &layernorm_output,
                    attention_mask,
                    seqlen_offset,
                    kv_cache_mgr.engine_mut(cache_idx),
                    block_table,
                    slot_mapping,
                )?
            }
        };

        // Scale residual and attention output
        let residual = residual.affine(self.attn_alpha, 0.0)?;
        let attn_output = attn_output.affine(self.attn_beta, 0.0)?;

        let layernorm_input = (residual + attn_output)?;
        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            layernorm_input
        };

        let ffn_output = match &self.ffn {
            FfnVariant::Dense(mlp) => mlp.forward(&layernorm_output)?,
            FfnVariant::MoE(moe) => {
                let moe_output = moe.forward(&layernorm_output)?;
                self.apply_shared_expert(&layernorm_output, &moe_output)?
            }
        };

        let residual = residual.affine(self.mlp_alpha, 0.0)?;
        let ffn_output = ffn_output.affine(self.mlp_beta, 0.0)?;

        residual + ffn_output
    }

    pub(crate) fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(xs)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            xs.clone()
        };

        let attn_output = match &self.self_attn {
            AttentionVariant::Linear(lin_attn) => {
                lin_attn.forward_decode_batch(&layernorm_output, sequences)?
            }
            AttentionVariant::Full(attn) => {
                let cache_idx = self
                    .cache_layer_idx
                    .expect("full attention layer should have cache index");
                attn.forward_decode_batch(
                    &layernorm_output,
                    sequences,
                    kv_cache_mgr.engine_mut(cache_idx),
                )?
            }
        };

        let residual = residual.affine(self.attn_alpha, 0.0)?;
        let attn_output = attn_output.affine(self.attn_beta, 0.0)?;

        let layernorm_input = (residual + attn_output)?;
        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            layernorm_input
        };

        let ffn_output = match &self.ffn {
            FfnVariant::Dense(mlp) => mlp.forward(&layernorm_output)?,
            FfnVariant::MoE(moe) => {
                let moe_output = moe.forward(&layernorm_output)?;
                self.apply_shared_expert(&layernorm_output, &moe_output)?
            }
        };

        let residual = residual.affine(self.mlp_alpha, 0.0)?;
        let ffn_output = ffn_output.affine(self.mlp_beta, 0.0)?;

        residual + ffn_output
    }

    /// Apply shared expert mixing if configured.
    fn apply_shared_expert(&self, input: &Tensor, moe_output: &Tensor) -> Result<Tensor> {
        match &self.shared_expert {
            Some(shared) => {
                let moe_fp32 = moe_output.to_dtype(DType::F32)?;
                let shared_output = shared.mlp.forward(input)?.to_dtype(DType::F32)?;
                let coef = shared.coefficient.forward(&input.to_dtype(DType::F32)?)?;

                let mixed = match shared.mode.as_str() {
                    "sigmoid" => {
                        let coef = candle_nn::ops::sigmoid(&coef)?;
                        let one_minus = coef.affine(-1.0, 1.0)?;
                        (moe_fp32.broadcast_mul(&one_minus)?
                            + shared_output.broadcast_mul(&coef)?)?
                    }
                    _ => {
                        // softmax (default)
                        let coef = candle_nn::ops::softmax_last_dim(&coef)?;
                        let one_minus = coef.affine(-1.0, 1.0)?;
                        (moe_fp32.broadcast_mul(&one_minus)?
                            + shared_output.broadcast_mul(&coef)?)?
                    }
                };

                mixed.to_dtype(input.dtype())
            }
            None => Ok(moe_output.clone()),
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MiniMaxText01ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMaxText01DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    /// Number of attention layers (for KV cache sizing)
    num_attn_layers: usize,
}

impl MiniMaxText01ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let minimax_cfg = MiniMaxText01Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        // Pre-count linear attention layers for slope rate scaling.
        let num_linear_layers = minimax_cfg
            .decoder_attention_types
            .iter()
            .filter(|&&t| t == 0)
            .count()
            .max(1);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        let mut attn_layer_count = 0;
        let mut linear_layer_count = 0;

        for i in 0..cfg.num_hidden_layers {
            let (cache_layer_idx, linear_layer_idx) = if minimax_cfg.is_linear_attention(i) {
                let idx = linear_layer_count;
                linear_layer_count += 1;
                (None, idx)
            } else {
                let idx = attn_layer_count;
                attn_layer_count += 1;
                (Some(idx), 0)
            };

            layers.push(MiniMaxText01DecoderLayer::new(
                cfg,
                &minimax_cfg,
                i,
                cache_layer_idx,
                num_linear_layers,
                linear_layer_idx,
                vb_layers.pp(i),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            num_attn_layers: attn_layer_count,
        })
    }

    /// Get the number of full attention layers (for KV cache sizing).
    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }
}

impl crate::engine::ModelForward for MiniMaxText01ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

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

        let mut xs = self.embed_tokens.forward(input_ids)?;

        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for layer in &self.layers {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr)?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_minimax_text01_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // 4 layers: layers 0,2 are linear attention, layers 1,3 are full attention
        extra.insert(
            "attn_type_list".to_string(),
            serde_json::json!([0, 1, 0, 1]),
        );
        // All layers have 1 expert (dense MLP)
        extra.insert("num_local_experts".to_string(), serde_json::json!(1));

        ModelConfig {
            architectures: vec!["MiniMaxText01ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn test_minimax_text01_moe_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // 2 layers: layer 0 linear, layer 1 full
        extra.insert("attn_type_list".to_string(), serde_json::json!([0, 1]));
        // 4 experts per layer
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        // Shared expert on both layers
        extra.insert(
            "shared_intermediate_size".to_string(),
            serde_json::json!(64),
        );
        extra.insert("shared_moe_mode".to_string(), serde_json::json!("softmax"));

        ModelConfig {
            architectures: vec!["MiniMaxText01ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn create_cache(cfg: &ModelConfig, num_attn_layers: usize) -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: num_attn_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_parsing_attn_type_list() {
        let cfg = test_minimax_text01_config();
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.decoder_attention_types, vec![0, 1, 0, 1]);
        assert!(minimax_cfg.is_linear_attention(0));
        assert!(!minimax_cfg.is_linear_attention(1));
        assert!(minimax_cfg.is_linear_attention(2));
        assert!(!minimax_cfg.is_linear_attention(3));
        assert_eq!(minimax_cfg.num_attention_layers(), 2);
    }

    #[test]
    fn test_config_parsing_layer_types_format() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.remove("attn_type_list");
        cfg.extra.insert(
            "layer_types".to_string(),
            serde_json::json!([
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention"
            ]),
        );
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.decoder_attention_types, vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_config_default_all_full_attention() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.remove("attn_type_list");
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.decoder_attention_types, vec![1, 1, 1, 1]);
        assert_eq!(minimax_cfg.num_attention_layers(), 4);
    }

    #[test]
    fn test_config_expert_counts_list() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.insert(
            "num_local_experts".to_string(),
            serde_json::json!([1, 4, 1, 8]),
        );
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.expert_count(0), 1);
        assert_eq!(minimax_cfg.expert_count(1), 4);
        assert_eq!(minimax_cfg.expert_count(2), 1);
        assert_eq!(minimax_cfg.expert_count(3), 8);
    }

    #[test]
    fn test_config_shared_intermediate() {
        let cfg = test_minimax_text01_moe_config();
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.shared_intermediate(0), 64);
        assert_eq!(minimax_cfg.shared_intermediate(1), 64);
    }

    #[test]
    fn test_config_scaling_defaults() {
        let cfg = test_minimax_text01_config();
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.linear_attn_alpha, 1.0);
        assert_eq!(minimax_cfg.linear_attn_beta, 1.0);
        assert_eq!(minimax_cfg.full_attn_alpha, 1.0);
        assert_eq!(minimax_cfg.full_attn_beta, 1.0);
        assert_eq!(minimax_cfg.mlp_alpha, 1.0);
        assert_eq!(minimax_cfg.mlp_beta, 1.0);
        assert!(!minimax_cfg.postnorm);
    }

    #[test]
    fn test_config_scaling_custom() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.insert(
            "layernorm_linear_attention_alpha".to_string(),
            serde_json::json!(0.5),
        );
        cfg.extra.insert(
            "layernorm_full_attention_beta".to_string(),
            serde_json::json!(2.0),
        );
        cfg.extra
            .insert("postnorm".to_string(), serde_json::json!(true));

        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.linear_attn_alpha, 0.5);
        assert_eq!(minimax_cfg.full_attn_beta, 2.0);
        assert!(minimax_cfg.postnorm);
    }

    // ─── MLP Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mlp_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mlp = MiniMaxText01Mlp::new(64, 128, vb.pp("mlp")).expect("mlp");
        let input = Tensor::zeros((2, 3, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, 64]);
    }

    // ─── Linear Attention Tests ─────────────────────────────────────────────────

    #[test]
    fn test_linear_attention_prefill() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // hidden_size=64, num_heads=4, head_dim=16, 2 linear layers, this is layer 0
        let attn = MiniMaxText01LinearAttention::new(64, 4, 16, 2, 0, vb.pp("attn")).expect("attn");
        let input = Tensor::zeros((1, 3, 64), DType::F32, &device).expect("input");
        let output = attn.forward_prefill(&input);
        assert!(output.is_ok(), "linear attn prefill: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[1, 3, 64]);
    }

    #[test]
    fn test_linear_attention_decode_batch() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = MiniMaxText01LinearAttention::new(64, 4, 16, 2, 0, vb.pp("attn")).expect("attn");
        // Simulate 2 decode sequences, xs: [2, 1, hidden_size]
        let input = Tensor::zeros((2, 1, 64), DType::F32, &device).expect("input");
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 10,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 5,
                block_ids: vec![],
                slot_mapping: vec![],
            },
        ];
        let output = attn.forward_decode_batch(&input, &sequences);
        assert!(output.is_ok(), "linear attn decode: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 1, 64]);
    }

    #[test]
    fn test_linear_attention_state_update() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = MiniMaxText01LinearAttention::new(64, 4, 16, 1, 0, vb.pp("attn")).expect("attn");
        let input = Tensor::zeros((1, 1, 64), DType::F32, &device).expect("input");
        let seq = vec![DecodeSequenceMetadata {
            request_id: 42,
            seqlen_offset: 0,
            block_ids: vec![],
            slot_mapping: vec![],
        }];

        // Before any decode, no state stored.
        assert_eq!(attn.state_count(), 0);

        // First decode step allocates state for request 42.
        let out1 = attn.forward_decode_batch(&input, &seq).expect("step1");
        assert_eq!(out1.dims(), &[1, 1, 64]);
        assert_eq!(
            attn.state_count(),
            1,
            "state should be stored after first decode"
        );

        // Second decode step reuses the stored state.
        let out2 = attn.forward_decode_batch(&input, &seq).expect("step2");
        assert_eq!(out2.dims(), &[1, 1, 64]);
        assert_eq!(
            attn.state_count(),
            1,
            "still one state entry after second decode"
        );

        // free_state removes the entry.
        attn.free_state(42);
        assert_eq!(attn.state_count(), 0, "state should be gone after free");

        // Third step starts from zero state again — same shape, no error.
        let out3 = attn.forward_decode_batch(&input, &seq).expect("step3");
        assert_eq!(out3.dims(), &[1, 1, 64]);
        assert_eq!(attn.state_count(), 1, "new state entry after step3");
        let _ = (out1, out2, out3);
    }

    // ─── MoE Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_moe_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let moe = MiniMaxText01MoE::new(64, 128, 4, 2, vb.pp("moe")).expect("moe");
        let input = Tensor::zeros((2, 3, 64), DType::F32, &device).expect("input");
        let output = moe.forward(&input);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, 64]);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_construction_hybrid() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxText01ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxText01ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), 4);
        // 2 full attention layers (indices 1 and 3)
        assert_eq!(model.num_attn_layers, 2);
    }

    #[test]
    fn test_construction_moe() {
        let cfg = test_minimax_text01_moe_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxText01ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxText01ForCausalLM MoE should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_layer_type_classification() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("model");

        // Layers 0,2 are linear; layers 1,3 are full attention
        assert!(matches!(
            model.layers[0].self_attn,
            AttentionVariant::Linear(_)
        ));
        assert!(matches!(
            model.layers[1].self_attn,
            AttentionVariant::Full(_)
        ));
        assert!(matches!(
            model.layers[2].self_attn,
            AttentionVariant::Linear(_)
        ));
        assert!(matches!(
            model.layers[3].self_attn,
            AttentionVariant::Full(_)
        ));

        assert!(model.layers[0].cache_layer_idx.is_none());
        assert_eq!(model.layers[1].cache_layer_idx, Some(0));
        assert!(model.layers[2].cache_layer_idx.is_none());
        assert_eq!(model.layers[3].cache_layer_idx, Some(1));
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_forward_shape() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

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

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
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
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");

        let logits = model
            .forward(&next, 3, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_model_forward_trait() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_device() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("model");

        assert!(matches!(ModelForward::device(&model), Device::Cpu));
    }

    // ─── MoE Forward Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_moe_model_forward() {
        let cfg = test_minimax_text01_moe_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("build moe model");

        // Only 1 full attention layer (layer 1)
        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("moe forward");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
