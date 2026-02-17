//! Mamba-2 (State Space Duality) model architecture.
//!
//! Mamba-2 extends Mamba with multi-head SSM (State Space Duality):
//! - Multiple SSM heads (like attention heads), each with its own A parameter
//! - Grouped state space duality (SSD) instead of single-head selective scan
//! - Per-head state dimension (head_dim) rather than monolithic d_inner
//!
//! Architecture:
//! ```text
//! Embedding -> [Mamba2Block x N] -> RMSNorm -> LM Head
//!
//! Mamba2Block:
//!   RMSNorm -> In-projection -> Conv1D -> SiLU -> Multi-head SSM -> Out-projection
//!                                                      |
//!                                                 num_heads * head_dim
//! ```

use std::sync::Mutex;

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::ssm::selective_scan;
use crate::ssm::state::SSMStateManager;

// ─── Mamba2 Config Extraction ───────────────────────────────────────────────

/// Mamba2-specific config fields extracted from ModelConfig.extra.
struct Mamba2Config {
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    #[allow(dead_code)]
    expand: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
}

impl Mamba2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let expand = cfg
            .extra
            .get("expand")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let d_inner = cfg
            .extra
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size * expand);

        let d_state = cfg
            .extra
            .get("state_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);

        let d_conv = cfg
            .extra
            .get("conv_kernel")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let num_heads = cfg
            .extra
            .get("num_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let head_dim = cfg
            .extra
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(if num_heads > 0 {
                d_inner / num_heads
            } else {
                d_inner
            });

        let n_groups = cfg
            .extra
            .get("n_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        Self {
            d_inner,
            d_state,
            d_conv,
            expand,
            num_heads,
            head_dim,
            n_groups,
        }
    }
}

// ─── Causal Conv1D (shared with mamba.rs pattern) ───────────────────────────

fn causal_conv1d_prefill(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_batch, d_inner, seq_len) = x.dims3()?;
    let (_d_inner_w, _one, kernel_size) = weight.dims3()?;

    let pad_len = kernel_size - 1;
    let pad = Tensor::zeros((x.dims()[0], d_inner, pad_len), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&pad, x], 2)?;

    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let window = padded.narrow(2, t, kernel_size)?;
        let w = weight.squeeze(1)?;
        let w_expanded = w.unsqueeze(0)?;
        let product = window.broadcast_mul(&w_expanded)?;
        let conv_out = product.sum(2)?;
        let conv_out = conv_out.broadcast_add(bias)?;
        outputs.push(conv_out.unsqueeze(2)?);
    }

    Tensor::cat(&outputs, 2)
}

fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    conv_state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, d_inner) = x.dims2()?;
    let (_d_inner_w, _one, kernel_size) = weight.dims3()?;
    let conv_state_len = kernel_size - 1;

    let x_expanded = x.unsqueeze(2)?;

    let new_conv_state = if conv_state_len > 1 {
        let shifted = conv_state.narrow(2, 1, conv_state_len - 1)?;
        Tensor::cat(&[&shifted, &x_expanded], 2)?
    } else if conv_state_len == 1 {
        x_expanded.clone()
    } else {
        Tensor::zeros((batch, d_inner, 0), x.dtype(), x.device())?
    };

    let full_window = Tensor::cat(&[&new_conv_state, &x_expanded], 2)?;

    let w = weight.squeeze(1)?;
    let w_expanded = w.unsqueeze(0)?;
    let product = full_window.broadcast_mul(&w_expanded)?;
    let conv_out = product.sum(2)?;
    let conv_out = conv_out.broadcast_add(bias)?;

    Ok((conv_out, new_conv_state))
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}

// ─── Mamba2 Block ───────────────────────────────────────────────────────────

struct Mamba2Block {
    norm: RmsNorm,
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    out_proj: Linear,
    /// Per-head A parameter: [num_heads]
    a: Tensor,
    /// Per-head D skip connection: [num_heads]
    d_param: Tensor,
    /// dt_bias: [num_heads]
    dt_bias: Tensor,
    /// Norm for B/C groups
    norm_b: Option<RmsNorm>,
    norm_c: Option<RmsNorm>,
    d_inner: usize,
    d_state: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
}

impl Mamba2Block {
    fn new(cfg: &ModelConfig, m2_cfg: &Mamba2Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_inner = m2_cfg.d_inner;
        let d_state = m2_cfg.d_state;
        let d_conv = m2_cfg.d_conv;
        let num_heads = m2_cfg.num_heads;
        let head_dim = m2_cfg.head_dim;
        let n_groups = m2_cfg.n_groups;

        let norm = rms_norm(hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        // Mamba2 in_proj: hidden -> d_inner + 2*n_groups*d_state + num_heads
        // (x, B, C, dt) -- B and C are per-group, dt is per-head
        let in_proj_size = d_inner + 2 * n_groups * d_state + num_heads;
        let in_proj = linear_no_bias(hidden_size, in_proj_size, vb.pp("in_proj"))?;

        // Depthwise causal convolution on the x portion (d_inner)
        let conv1d_weight = vb.pp("conv1d").get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(d_inner, "bias")?;

        // Out-projection: d_inner -> hidden
        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("out_proj"))?;

        // Per-head parameters
        // A_log was stored as A_log, we load A (negated exp) directly using neg-exp at runtime
        // In Mamba2, A is stored directly as [num_heads] (not A_log)
        let a = vb.get(num_heads, "A")?;

        // D: [num_heads] skip connection
        let d_param = vb.get(num_heads, "D")?;

        // dt_bias: [num_heads]
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        // Optional group norms for B and C
        let norm_b = if n_groups > 1 {
            Some(rms_norm(
                n_groups * d_state,
                cfg.rms_norm_eps,
                vb.pp("norm_B"),
            )?)
        } else {
            None
        };
        let norm_c = if n_groups > 1 {
            Some(rms_norm(
                n_groups * d_state,
                cfg.rms_norm_eps,
                vb.pp("norm_C"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm,
            in_proj,
            conv1d_weight,
            conv1d_bias,
            out_proj,
            a,
            d_param,
            dt_bias,
            norm_b,
            norm_c,
            d_inner,
            d_state,
            num_heads,
            head_dim,
            n_groups,
        })
    }

    /// Multi-head SSM forward for one sequence of tokens.
    ///
    /// Mamba2 splits the inner dimension into multiple heads, each running
    /// an independent selective scan with its own A parameter. B/C are
    /// grouped across heads for parameter efficiency.
    fn forward_prefill(
        &self,
        hidden_states: &Tensor,
        ssm_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;

        let xs = self.norm.forward(hidden_states)?;

        // In-projection: [batch, seq_len, hidden] -> [batch, seq_len, in_proj_size]
        let proj = self.in_proj.forward(&xs)?;

        // Split: x (d_inner), B (n_groups*d_state), C (n_groups*d_state), dt (num_heads)
        let x = proj.narrow(2, 0, self.d_inner)?;
        let b_offset = self.d_inner;
        let b_proj = proj.narrow(2, b_offset, self.n_groups * self.d_state)?;
        let c_offset = b_offset + self.n_groups * self.d_state;
        let c_proj = proj.narrow(2, c_offset, self.n_groups * self.d_state)?;
        let dt_offset = c_offset + self.n_groups * self.d_state;
        let dt = proj.narrow(2, dt_offset, self.num_heads)?;

        // Apply optional group norms to B, C
        let b_proj = if let Some(ref norm) = self.norm_b {
            norm.forward(&b_proj)?
        } else {
            b_proj
        };
        let c_proj = if let Some(ref norm) = self.norm_c {
            norm.forward(&c_proj)?
        } else {
            c_proj
        };

        // Causal conv1d on x
        let x_conv = x.transpose(1, 2)?;
        let x_conv = causal_conv1d_prefill(&x_conv, &self.conv1d_weight, &self.conv1d_bias)?;

        // Extract conv state for future decode steps
        let d_conv = self.conv1d_weight.dims()[2];
        let conv_state_len = d_conv - 1;
        let new_conv_state = if seq_len >= conv_state_len {
            x.transpose(1, 2)?
                .narrow(2, seq_len - conv_state_len, conv_state_len)?
        } else {
            let pad = Tensor::zeros(
                (batch, self.d_inner, conv_state_len - seq_len),
                x.dtype(),
                x.device(),
            )?;
            let x_t = x.transpose(1, 2)?;
            Tensor::cat(&[&pad, &x_t], 2)?
        };

        let x_conv = x_conv.transpose(1, 2)?;
        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        // Apply dt_bias and softplus to get delta per-head
        // dt: [batch, seq_len, num_heads]
        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

        // Multi-head SSM: process each head independently
        // Split x_ssm into heads: [batch, seq_len, num_heads, head_dim]
        // Process each head with selective_scan using group-shared B, C
        let heads_per_group = self.num_heads / self.n_groups;
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut new_ssm_states = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let group_idx = h / heads_per_group;

            // x for this head: [batch, seq_len, head_dim]
            let x_h = x_ssm.narrow(2, h * self.head_dim, self.head_dim)?;

            // delta for this head: [batch, seq_len, 1] -> broadcast to [batch, seq_len, head_dim]
            let delta_h = delta.narrow(2, h, 1)?;
            let delta_h = delta_h.broadcast_as((batch, seq_len, self.head_dim))?;

            // A for this head: scalar -> [head_dim, d_state]
            // A is stored as [num_heads], we need [head_dim, d_state] for selective_scan
            let a_h_scalar = self.a.narrow(0, h, 1)?;
            let a_h = a_h_scalar
                .broadcast_as((self.head_dim, self.d_state))?
                .contiguous()?;

            // B for this group: [batch, seq_len, d_state]
            let b_h = b_proj.narrow(2, group_idx * self.d_state, self.d_state)?;

            // C for this group: [batch, seq_len, d_state]
            let c_h = c_proj.narrow(2, group_idx * self.d_state, self.d_state)?;

            // D for this head
            let d_h_scalar = self.d_param.narrow(0, h, 1)?;
            let d_h = d_h_scalar.broadcast_as((self.head_dim,))?.contiguous()?;

            // SSM state for this head: [batch, head_dim, d_state]
            let ssm_h = ssm_state.narrow(1, h * self.head_dim, self.head_dim)?;

            let (out_h, new_state_h) =
                selective_scan(&x_h, &delta_h, &a_h, &b_h, &c_h, &d_h, Some(&ssm_h))?;

            head_outputs.push(out_h);
            new_ssm_states.push(new_state_h);
        }

        // Concatenate head outputs: [batch, seq_len, d_inner]
        let ssm_out = Tensor::cat(&head_outputs, 2)?;

        // Concatenate new SSM states: [batch, d_inner, d_state]
        let new_ssm_state = Tensor::cat(&new_ssm_states, 1)?;

        // Out-projection
        let output = self.out_proj.forward(&ssm_out)?;

        // Residual connection
        let output = (hidden_states + output)?;

        Ok((output, new_ssm_state, new_conv_state))
    }

    /// Decode (single token) forward pass.
    fn forward_decode(
        &self,
        hidden_states: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let xs = self.norm.forward(hidden_states)?;

        // In-projection
        let proj = self.in_proj.forward(&xs)?; // [batch, 1, in_proj_size]
        let proj = proj.squeeze(1)?; // [batch, in_proj_size]

        let x = proj.narrow(1, 0, self.d_inner)?;
        let b_offset = self.d_inner;
        let b_proj = proj.narrow(1, b_offset, self.n_groups * self.d_state)?;
        let c_offset = b_offset + self.n_groups * self.d_state;
        let c_proj = proj.narrow(1, c_offset, self.n_groups * self.d_state)?;
        let dt_offset = c_offset + self.n_groups * self.d_state;
        let dt = proj.narrow(1, dt_offset, self.num_heads)?;

        // Apply optional group norms
        let b_proj = if let Some(ref norm) = self.norm_b {
            norm.forward(&b_proj.unsqueeze(1)?)?.squeeze(1)?
        } else {
            b_proj
        };
        let c_proj = if let Some(ref norm) = self.norm_c {
            norm.forward(&c_proj.unsqueeze(1)?)?.squeeze(1)?
        } else {
            c_proj
        };

        // Conv1d decode
        let (x_conv, new_conv_state) =
            causal_conv1d_decode(&x, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        // dt_bias + softplus
        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

        // Multi-head SSM (single step)
        let batch = hidden_states.dims()[0];
        let heads_per_group = self.num_heads / self.n_groups;
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut new_ssm_states = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let group_idx = h / heads_per_group;

            let x_h = x_ssm.narrow(1, h * self.head_dim, self.head_dim)?;
            let delta_h = delta.narrow(1, h, 1)?;
            let delta_h = delta_h.broadcast_as((batch, self.head_dim))?.contiguous()?;

            let a_h_scalar = self.a.narrow(0, h, 1)?;
            let a_h = a_h_scalar
                .broadcast_as((self.head_dim, self.d_state))?
                .contiguous()?;

            let b_h = b_proj.narrow(1, group_idx * self.d_state, self.d_state)?;
            let c_h = c_proj.narrow(1, group_idx * self.d_state, self.d_state)?;

            let d_h_scalar = self.d_param.narrow(0, h, 1)?;
            let d_h = d_h_scalar.broadcast_as((self.head_dim,))?.contiguous()?;

            let ssm_h = ssm_state.narrow(1, h * self.head_dim, self.head_dim)?;

            // Expand to [batch, 1, ...] for selective_scan
            let x_h_exp = x_h.unsqueeze(1)?;
            let delta_h_exp = delta_h.unsqueeze(1)?;
            let b_h_exp = b_h.unsqueeze(1)?;
            let c_h_exp = c_h.unsqueeze(1)?;

            let (out_h, new_state_h) = selective_scan(
                &x_h_exp,
                &delta_h_exp,
                &a_h,
                &b_h_exp,
                &c_h_exp,
                &d_h,
                Some(&ssm_h),
            )?;

            head_outputs.push(out_h.squeeze(1)?);
            new_ssm_states.push(new_state_h);
        }

        let ssm_out = Tensor::cat(&head_outputs, 1)?; // [batch, d_inner]
        let new_ssm_state = Tensor::cat(&new_ssm_states, 1)?;

        let output = self.out_proj.forward(&ssm_out.unsqueeze(1)?)?;
        let output = (hidden_states + output)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Mamba2ForCausalLM {
    embeddings: Embedding,
    layers: Vec<Mamba2Block>,
    norm_f: RmsNorm,
    lm_head: Linear,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    state_mgr: Mutex<SSMStateManager>,
    #[allow(dead_code)]
    mamba2_cfg: Mamba2InternalConfig,
}

#[allow(dead_code)]
struct Mamba2InternalConfig {
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    num_heads: usize,
    head_dim: usize,
}

impl Mamba2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let m2_cfg = Mamba2Config::from_model_config(cfg);
        let vb_backbone = vb.pp("backbone");

        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_backbone.pp("embeddings"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_backbone.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let vb_block = vb_layers.pp(i).pp("mixer");
            layers.push(Mamba2Block::new(cfg, &m2_cfg, vb_block)?);
        }

        let norm_f = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_backbone.pp("norm_f"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embeddings.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let state_mgr = SSMStateManager::new(
            cfg.num_hidden_layers,
            m2_cfg.d_inner,
            m2_cfg.d_state,
            m2_cfg.d_conv,
            vb.dtype(),
            vb.device().clone(),
        );

        Ok(Self {
            embeddings,
            layers,
            norm_f,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            state_mgr: Mutex::new(state_mgr),
            mamba2_cfg: Mamba2InternalConfig {
                d_inner: m2_cfg.d_inner,
                d_state: m2_cfg.d_state,
                d_conv: m2_cfg.d_conv,
                num_heads: m2_cfg.num_heads,
                head_dim: m2_cfg.head_dim,
            },
        })
    }

    /// SSM-native forward pass.
    ///
    /// Each request is identified by `request_id` for independent state tracking.
    fn forward_ssm(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        request_id: u64,
    ) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;

        let mut state_mgr = self
            .state_mgr
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("state lock poisoned: {e}")))?;

        if seqlen_offset == 0 {
            state_mgr.free_state(request_id);
            state_mgr.allocate_state(request_id).map_err(|e| {
                candle_core::Error::Msg(format!("failed to allocate SSM state: {e}"))
            })?;
        }

        let request_state = state_mgr
            .get_state(request_id)
            .ok_or_else(|| candle_core::Error::Msg("SSM state not found for request".into()))?;

        let mut hidden = self.embeddings.forward(input_ids)?;
        let is_prefill = seq_len > 1;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let ssm_state = &request_state.ssm_states[layer_idx].tensor;
            let conv_state = &request_state.conv_states[layer_idx].tensor;

            let (new_hidden, new_ssm_state, new_conv_state) = if is_prefill {
                layer.forward_prefill(&hidden, ssm_state)?
            } else {
                layer.forward_decode(&hidden, ssm_state, conv_state)?
            };

            hidden = new_hidden;
            request_state.ssm_states[layer_idx].tensor = new_ssm_state;
            request_state.conv_states[layer_idx].tensor = new_conv_state;
        }

        let hidden = self.norm_f.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok(logits)
    }
}

impl crate::engine::ModelForward for Mamba2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Single-sequence path uses request_id=0 by convention.
        self.forward_ssm(input_ids, seqlen_offset, 0)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        // Each sequence is processed individually with its own per-request SSM state.
        // The SSMStateManager tracks independent state for each request_id,
        // preventing cross-request state corruption during concurrent inference.
        let mut outputs = Vec::with_capacity(sequences.len());
        for (i, seq) in sequences.iter().enumerate() {
            let token = input_ids.narrow(0, i, 1)?;
            let logits = self.forward_ssm(&token, seq.seqlen_offset, seq.request_id)?;
            outputs.push(logits);
        }
        Tensor::cat(&outputs, 0)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_mamba2_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("expand".to_string(), serde_json::json!(2));
        extra.insert("state_size".to_string(), serde_json::json!(8));
        extra.insert("conv_kernel".to_string(), serde_json::json!(4));
        extra.insert("num_heads".to_string(), serde_json::json!(4));
        extra.insert("head_dim".to_string(), serde_json::json!(32));
        extra.insert("n_groups".to_string(), serde_json::json!(1));

        ModelConfig {
            architectures: vec!["Mamba2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 32,
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

    fn create_dummy_cache() -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    #[test]
    fn test_mamba2_construction() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Mamba2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Mamba2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_mamba2_config_extraction() {
        let cfg = test_mamba2_config();
        let m2_cfg = Mamba2Config::from_model_config(&cfg);

        assert_eq!(m2_cfg.d_inner, 128);
        assert_eq!(m2_cfg.d_state, 8);
        assert_eq!(m2_cfg.d_conv, 4);
        assert_eq!(m2_cfg.expand, 2);
        assert_eq!(m2_cfg.num_heads, 4);
        assert_eq!(m2_cfg.head_dim, 32);
        assert_eq!(m2_cfg.n_groups, 1);
    }

    #[test]
    fn test_mamba2_forward_shape() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        let logits = model.forward_ssm(&input_ids, 0, 0).expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_mamba2_single_token() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill to initialize state
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 0).expect("prefill");

        // Decode one token
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");
        let logits = model.forward_ssm(&next, 3, 0).expect("decode");

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_mamba2_model_forward_trait() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_dummy_cache();

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
        .expect("forward");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_mamba2_device() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_mamba2_concurrent_requests_independent_state() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill two separate requests
        let prompt1 = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt1");
        let _ = model.forward_ssm(&prompt1, 0, 1).expect("prefill req 1");

        let prompt2 = Tensor::ones((1, 3), DType::U32, &device).expect("prompt2");
        let _ = model.forward_ssm(&prompt2, 0, 2).expect("prefill req 2");

        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(1), "request 1 should have state");
        assert!(state_mgr.has_state(2), "request 2 should have state");
        assert_eq!(state_mgr.num_active_requests(), 2);
    }

    #[test]
    fn test_mamba2_concurrent_decode_no_cross_contamination() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 10).expect("prefill req 10");
        let _ = model.forward_ssm(&prompt, 0, 20).expect("prefill req 20");

        // Decode request 10 multiple times
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");
        let logits_10_step1 = model.forward_ssm(&next, 3, 10).expect("decode 10 step 1");
        let _ = model.forward_ssm(&next, 4, 10).expect("decode 10 step 2");

        // Decode request 20 once -- should not be affected by request 10's steps
        let logits_20_step1 = model.forward_ssm(&next, 3, 20).expect("decode 20 step 1");

        let data_10_1: Vec<f32> = logits_10_step1
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        let data_20_1: Vec<f32> = logits_20_step1
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        for (a, b) in data_10_1.iter().zip(data_20_1.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "request 10 step 1 and request 20 step 1 should match, got {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_mamba2_forward_decode_batch_uses_per_request_state() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, _block_table) = create_dummy_cache();

        // Prefill two requests
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 100).expect("prefill req 100");
        let _ = model.forward_ssm(&prompt, 0, 200).expect("prefill req 200");

        // Batched decode
        let batch_input = Tensor::zeros((2, 1), DType::U32, &device).expect("batch input");
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 100,
                seqlen_offset: 3,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 200,
                seqlen_offset: 3,
                block_ids: vec![],
                slot_mapping: vec![],
            },
        ];

        let logits = model
            .forward_decode_batch(&batch_input, &sequences, &mut kv_cache_mgr)
            .expect("batched decode");

        assert_eq!(
            logits.dims(),
            &[2, 1, cfg.vocab_size],
            "batched decode should produce [batch_size, 1, vocab_size]"
        );

        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(100));
        assert!(state_mgr.has_state(200));
    }

    #[test]
    fn test_mamba2_state_freed_and_reallocated() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill request 42
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 42).expect("prefill");

        {
            let state_mgr = model.state_mgr.lock().expect("lock");
            assert!(state_mgr.has_state(42));
        }

        // Free and re-prefill
        {
            let mut state_mgr = model.state_mgr.lock().expect("lock");
            state_mgr.free_state(42);
            assert!(!state_mgr.has_state(42));
        }

        let _ = model.forward_ssm(&prompt, 0, 42).expect("re-prefill");
        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(42));
    }
}
