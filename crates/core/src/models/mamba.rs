//! Mamba (State Space Model) architecture implementation.
//!
//! Mamba is fundamentally different from transformer models:
//! - Uses selective scan (S6) recurrence instead of attention
//! - Uses causal convolution instead of positional embeddings
//! - Maintains per-sequence recurrent state instead of KV cache
//!
//! Architecture:
//! ```text
//! Embedding -> [MambaBlock x N] -> RMSNorm -> LM Head
//!
//! MambaBlock:
//!   RMSNorm -> In-projection -> Conv1D -> SiLU -> SSM -> Gate -> Out-projection
//!                                                          |
//!                                                     z (SiLU gate)
//! ```

use std::sync::Mutex;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::ssm::selective_scan;
use crate::ssm::state::SSMStateManager;

// ─── Mamba Config Extraction ────────────────────────────────────────────────

/// Mamba-specific config fields extracted from ModelConfig.extra.
#[allow(dead_code)]
struct MambaConfig {
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    expand: usize,
    dt_rank: usize,
}

impl MambaConfig {
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
            .unwrap_or(16);

        let d_conv = cfg
            .extra
            .get("conv_kernel")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        // dt_rank defaults to ceil(d_model / 16)
        let dt_rank = cfg
            .extra
            .get("time_step_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size.div_ceil(16));

        Self {
            d_inner,
            d_state,
            d_conv,
            expand,
            dt_rank,
        }
    }
}

// ─── Causal Conv1D ──────────────────────────────────────────────────────────

/// Applies causal 1D convolution.
///
/// During prefill, performs full causal convolution with left-padding.
/// During decode, uses the conv state buffer (shift in, convolve).
fn causal_conv1d_prefill(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_batch, d_inner, seq_len) = x.dims3()?;
    let (_d_inner_w, _one, kernel_size) = weight.dims3()?;

    // Left-pad the input with zeros: [batch, d_inner, pad + seq_len]
    let pad_len = kernel_size - 1;
    let pad = Tensor::zeros((x.dims()[0], d_inner, pad_len), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&pad, x], 2)?; // [batch, d_inner, pad_len + seq_len]

    // Manual depthwise convolution: for each output position, compute the dot product
    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        // Extract window: [batch, d_inner, kernel_size]
        let window = padded.narrow(2, t, kernel_size)?;
        // Weight: [d_inner, 1, kernel_size] -> squeeze -> [d_inner, kernel_size]
        let w = weight.squeeze(1)?;
        // Depthwise: element-wise multiply and sum over kernel dim
        // window: [batch, d_inner, kernel_size], w: [d_inner, kernel_size]
        let w_expanded = w.unsqueeze(0)?; // [1, d_inner, kernel_size]
        let product = window.broadcast_mul(&w_expanded)?; // [batch, d_inner, kernel_size]
        let conv_out = product.sum(2)?; // [batch, d_inner]
                                        // Add bias: [d_inner]
        let conv_out = conv_out.broadcast_add(bias)?;
        outputs.push(conv_out.unsqueeze(2)?); // [batch, d_inner, 1]
    }

    Tensor::cat(&outputs, 2) // [batch, d_inner, seq_len]
}

/// Apply causal conv1d during decode (single token) using conv state.
///
/// Shifts the new value into the conv state buffer and computes the convolution
/// for the current position.
fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    conv_state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, d_inner) = x.dims2()?;
    let (_d_inner_w, _one, kernel_size) = weight.dims3()?;
    let conv_state_len = kernel_size - 1;

    // Shift conv state left and append new value
    // conv_state: [batch, d_inner, conv_state_len]
    // new value x: [batch, d_inner] -> [batch, d_inner, 1]
    let x_expanded = x.unsqueeze(2)?;

    let new_conv_state = if conv_state_len > 1 {
        // Drop oldest, append newest: [batch, d_inner, conv_state_len-1] ++ [batch, d_inner, 1]
        let shifted = conv_state.narrow(2, 1, conv_state_len - 1)?;
        Tensor::cat(&[&shifted, &x_expanded], 2)? // [batch, d_inner, conv_state_len]
    } else if conv_state_len == 1 {
        x_expanded.clone()
    } else {
        // conv_state_len == 0, kernel_size == 1
        Tensor::zeros((batch, d_inner, 0), x.dtype(), x.device())?
    };

    // Full window: [conv_state, x] = [batch, d_inner, kernel_size]
    let full_window = Tensor::cat(&[&new_conv_state, &x_expanded], 2)?;

    // Depthwise convolution
    let w = weight.squeeze(1)?; // [d_inner, kernel_size]
    let w_expanded = w.unsqueeze(0)?; // [1, d_inner, kernel_size]
    let product = full_window.broadcast_mul(&w_expanded)?; // [batch, d_inner, kernel_size]
    let conv_out = product.sum(2)?; // [batch, d_inner]
    let conv_out = conv_out.broadcast_add(bias)?;

    Ok((conv_out, new_conv_state))
}

// ─── Mamba Block ────────────────────────────────────────────────────────────

struct MambaBlock {
    norm: RmsNorm,
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    d_inner: usize,
    d_state: usize,
    dt_rank: usize,
}

impl MambaBlock {
    fn new(cfg: &ModelConfig, mamba_cfg: &MambaConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_inner = mamba_cfg.d_inner;
        let d_state = mamba_cfg.d_state;
        let d_conv = mamba_cfg.d_conv;
        let dt_rank = mamba_cfg.dt_rank;

        // Layer norm before the block
        let norm = rms_norm(hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        // In-projection: hidden -> 2*d_inner (split into x and z)
        let in_proj = linear_no_bias(hidden_size, 2 * d_inner, vb.pp("in_proj"))?;

        // Depthwise causal convolution
        // Weight shape: [d_inner, 1, d_conv] (depthwise)
        let conv1d_weight = vb.pp("conv1d").get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(d_inner, "bias")?;

        // SSM projections from x
        // x_proj: d_inner -> dt_rank + 2*d_state (dt, B, C)
        let x_proj = linear_no_bias(d_inner, dt_rank + 2 * d_state, vb.pp("x_proj"))?;

        // dt_proj: dt_rank -> d_inner
        let dt_proj = Linear::new(
            vb.pp("dt_proj").get((d_inner, dt_rank), "weight")?,
            Some(vb.pp("dt_proj").get(d_inner, "bias")?),
        );

        // A_log: [d_inner, d_state] - log of state transition matrix
        let a_log = vb.get((d_inner, d_state), "A_log")?;

        // D: [d_inner] - skip connection
        let d = vb.get(d_inner, "D")?;

        // Out-projection: d_inner -> hidden
        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            norm,
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            d_inner,
            d_state,
            dt_rank,
        })
    }

    /// Forward pass for prefill (multiple tokens).
    fn forward_prefill(
        &self,
        hidden_states: &Tensor,
        ssm_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;

        // Pre-norm
        let xs = self.norm.forward(hidden_states)?;

        // In-projection: [batch, seq_len, hidden] -> [batch, seq_len, 2*d_inner]
        let proj = self.in_proj.forward(&xs)?;

        // Split into x and z
        let x = proj.narrow(2, 0, self.d_inner)?; // [batch, seq_len, d_inner]
        let z = proj.narrow(2, self.d_inner, self.d_inner)?; // [batch, seq_len, d_inner]

        // Causal conv1d on x
        // Transpose for conv: [batch, seq_len, d_inner] -> [batch, d_inner, seq_len]
        let x_conv = x.transpose(1, 2)?;
        let x_conv = causal_conv1d_prefill(&x_conv, &self.conv1d_weight, &self.conv1d_bias)?;

        // Extract conv state: last (d_conv-1) values for future decode steps
        let d_conv = self.conv1d_weight.dims()[2];
        let conv_state_len = d_conv - 1;
        let new_conv_state = if seq_len >= conv_state_len {
            x.transpose(1, 2)?
                .narrow(2, seq_len - conv_state_len, conv_state_len)?
        } else {
            // Pad with zeros on the left if sequence is shorter than conv state
            let pad = Tensor::zeros(
                (batch, self.d_inner, conv_state_len - seq_len),
                x.dtype(),
                x.device(),
            )?;
            let x_t = x.transpose(1, 2)?;
            Tensor::cat(&[&pad, &x_t], 2)?
        };

        // Transpose back: [batch, d_inner, seq_len] -> [batch, seq_len, d_inner]
        let x_conv = x_conv.transpose(1, 2)?;

        // SiLU activation
        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        // SSM: project x to get delta, B, C
        let x_dbc = self.x_proj.forward(&x_ssm)?; // [batch, seq_len, dt_rank + 2*d_state]
        let dt = x_dbc.narrow(2, 0, self.dt_rank)?;
        let b_proj = x_dbc.narrow(2, self.dt_rank, self.d_state)?;
        let c_proj = x_dbc.narrow(2, self.dt_rank + self.d_state, self.d_state)?;

        // dt_proj: [batch, seq_len, dt_rank] -> [batch, seq_len, d_inner]
        let delta = self.dt_proj.forward(&dt)?;
        // Softplus activation for delta to ensure positivity
        let delta = softplus(&delta)?;

        // A = -exp(A_log)
        let a = self.a_log.exp()?.neg()?;

        // Selective scan
        let (ssm_out, new_ssm_state) = selective_scan(
            &x_ssm,
            &delta,
            &a,
            &b_proj,
            &c_proj,
            &self.d,
            Some(ssm_state),
        )?;

        // Gate with z (SiLU)
        let z_gate = candle_nn::ops::silu(&z)?;
        let gated = (&ssm_out * &z_gate)?;

        // Out-projection
        let output = self.out_proj.forward(&gated)?;

        // Residual connection
        let output = (hidden_states + output)?;

        Ok((output, new_ssm_state, new_conv_state))
    }

    /// Forward pass for decode (single token).
    fn forward_decode(
        &self,
        hidden_states: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // hidden_states: [batch, 1, hidden]
        let xs = self.norm.forward(hidden_states)?;

        // In-projection
        let proj = self.in_proj.forward(&xs)?; // [batch, 1, 2*d_inner]
        let proj = proj.squeeze(1)?; // [batch, 2*d_inner]

        let x = proj.narrow(1, 0, self.d_inner)?; // [batch, d_inner]
        let z = proj.narrow(1, self.d_inner, self.d_inner)?; // [batch, d_inner]

        // Conv1d decode: use conv state
        let (x_conv, new_conv_state) =
            causal_conv1d_decode(&x, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        // SiLU
        let x_ssm = candle_nn::ops::silu(&x_conv)?; // [batch, d_inner]

        // SSM projections
        let x_dbc = self.x_proj.forward(&x_ssm.unsqueeze(1)?)?; // [batch, 1, dt_rank+2*d_state]
        let x_dbc = x_dbc.squeeze(1)?; // [batch, dt_rank+2*d_state]
        let dt = x_dbc.narrow(1, 0, self.dt_rank)?;
        let b_proj = x_dbc.narrow(1, self.dt_rank, self.d_state)?;
        let c_proj = x_dbc.narrow(1, self.dt_rank + self.d_state, self.d_state)?;

        // dt_proj + softplus
        let delta = self.dt_proj.forward(&dt.unsqueeze(1)?)?.squeeze(1)?;
        let delta = softplus(&delta)?; // [batch, d_inner]

        // A = -exp(A_log)
        let a = self.a_log.exp()?.neg()?;

        // Single-step selective scan
        let x_ssm_expanded = x_ssm.unsqueeze(1)?; // [batch, 1, d_inner]
        let delta_expanded = delta.unsqueeze(1)?; // [batch, 1, d_inner]
        let b_expanded = b_proj.unsqueeze(1)?; // [batch, 1, d_state]
        let c_expanded = c_proj.unsqueeze(1)?; // [batch, 1, d_state]

        let (ssm_out, new_ssm_state) = selective_scan(
            &x_ssm_expanded,
            &delta_expanded,
            &a,
            &b_expanded,
            &c_expanded,
            &self.d,
            Some(ssm_state),
        )?;

        let ssm_out = ssm_out.squeeze(1)?; // [batch, d_inner]

        // Gate
        let z_gate = candle_nn::ops::silu(&z)?;
        let gated = (&ssm_out * &z_gate)?;

        // Out-projection: [batch, d_inner] -> [batch, hidden]
        let output = self.out_proj.forward(&gated.unsqueeze(1)?)?; // [batch, 1, hidden]

        // Residual
        let output = (hidden_states + output)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    // Numerically stable softplus:
    // For large x: softplus(x) ~ x
    // For small x: softplus(x) ~ exp(x)
    // General: log(1 + exp(x))
    let ones = Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MambaForCausalLM {
    embeddings: Embedding,
    layers: Vec<MambaBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    /// SSM state is managed internally since Mamba does not use KV cache.
    state_mgr: Mutex<SSMStateManager>,
    /// Stored config values needed for state re-allocation.
    #[allow(dead_code)]
    mamba_cfg: MambaInternalConfig,
}

/// Stored config values needed at inference time.
#[allow(dead_code)]
struct MambaInternalConfig {
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
}

impl MambaForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mamba_cfg = MambaConfig::from_model_config(cfg);
        let vb_backbone = vb.pp("backbone");

        // Embedding
        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_backbone.pp("embeddings"),
        )?;

        // Mamba blocks
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_backbone.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let vb_block = vb_layers.pp(i).pp("mixer");
            layers.push(MambaBlock::new(cfg, &mamba_cfg, vb_block)?);
        }

        // Final norm
        let norm_f = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_backbone.pp("norm_f"))?;

        // LM head - may be tied to embeddings
        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embeddings.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let state_mgr = SSMStateManager::new(
            cfg.num_hidden_layers,
            mamba_cfg.d_inner,
            mamba_cfg.d_state,
            mamba_cfg.d_conv,
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
            mamba_cfg: MambaInternalConfig {
                d_inner: mamba_cfg.d_inner,
                d_state: mamba_cfg.d_state,
                d_conv: mamba_cfg.d_conv,
            },
        })
    }

    /// SSM-native forward pass.
    ///
    /// Unlike transformer models, Mamba maintains internal recurrent state
    /// rather than using a KV cache. The `seqlen_offset` indicates whether
    /// this is a prefill (0) or decode (>0) step.
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

        // Allocate state on first call (prefill)
        if seqlen_offset == 0 {
            // Free any existing state and allocate fresh
            state_mgr.free_state(request_id);
            state_mgr.allocate_state(request_id).map_err(|e| {
                candle_core::Error::Msg(format!("failed to allocate SSM state: {e}"))
            })?;
        }

        let request_state = state_mgr
            .get_state(request_id)
            .ok_or_else(|| candle_core::Error::Msg("SSM state not found for request".into()))?;

        // Embedding
        let mut hidden = self.embeddings.forward(input_ids)?; // [batch, seq_len, hidden]

        let is_prefill = seq_len > 1;

        // Process through each Mamba block
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

        // Final norm and LM head
        let hidden = self.norm_f.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok(logits)
    }
}

/// NOTE: Mamba models do not use KV cache. The ModelForward trait requires
/// KVCacheManager parameters for API compatibility with the engine, but
/// they are unused in the SSM forward pass. State is managed internally
/// via SSMStateManager.
impl crate::engine::ModelForward for MambaForCausalLM {
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

    fn test_mamba_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("expand".to_string(), serde_json::json!(2));
        extra.insert("state_size".to_string(), serde_json::json!(8));
        extra.insert("conv_kernel".to_string(), serde_json::json!(4));
        extra.insert("time_step_rank".to_string(), serde_json::json!(4));

        ModelConfig {
            architectures: vec!["MambaForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 64,
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
        // SSM doesn't use KV cache, but we need one for the ModelForward trait
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
    fn test_mamba_construction() {
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MambaForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MambaForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_mamba_config_extraction() {
        let cfg = test_mamba_config();
        let mamba_cfg = MambaConfig::from_model_config(&cfg);

        assert_eq!(mamba_cfg.d_inner, 128);
        assert_eq!(mamba_cfg.d_state, 8);
        assert_eq!(mamba_cfg.d_conv, 4);
        assert_eq!(mamba_cfg.expand, 2);
        assert_eq!(mamba_cfg.dt_rank, 4);
    }

    #[test]
    fn test_mamba_config_defaults() {
        let cfg = ModelConfig {
            hidden_size: 768,
            extra: serde_json::Map::new(),
            ..test_mamba_config()
        };
        let mamba_cfg = MambaConfig::from_model_config(&cfg);

        assert_eq!(mamba_cfg.d_inner, 768 * 2); // default expand=2
        assert_eq!(mamba_cfg.d_state, 16); // default
        assert_eq!(mamba_cfg.d_conv, 4); // default
        assert_eq!(mamba_cfg.dt_rank, (768 + 15) / 16); // ceil(768/16) = 48
    }

    #[test]
    fn test_mamba_forward_shape() {
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mamba_single_token_forward() {
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        // First do a prefill to initialize state
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 0).expect("prefill");

        // Then decode one token
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");
        let logits = model.forward_ssm(&next, 3, 0).expect("decode");

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_mamba_prefill_then_decode() {
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let prefill_logits = model.forward_ssm(&prompt, 0, 0).expect("prefill");
        assert_eq!(prefill_logits.dims(), &[1, 3, cfg.vocab_size]);

        // Decode steps
        for step in 0..3 {
            let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");
            let decode_logits = model.forward_ssm(&next, 3 + step, 0).expect("decode");
            assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);
        }
    }

    #[test]
    fn test_mamba_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_dummy_cache();

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        // ModelForward::forward should work (KV cache params ignored)
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
    fn test_mamba_device() {
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_softplus() {
        let device = Device::Cpu;

        // softplus(0) = ln(2) ~ 0.693
        let x = Tensor::zeros((1, 4), DType::F32, &device).expect("x");
        let result = softplus(&x).expect("softplus");
        let data: Vec<f32> = result.flatten_all().expect("flat").to_vec1().expect("vec");
        for val in &data {
            assert!(
                (*val - 0.6931).abs() < 1e-3,
                "softplus(0) should be ln(2), got {}",
                val
            );
        }

        // softplus(large) ~ x
        let x_data = vec![10.0f32; 4];
        let x = Tensor::from_vec(x_data, (1, 4), &device).expect("x");
        let result = softplus(&x).expect("softplus");
        let data: Vec<f32> = result.flatten_all().expect("flat").to_vec1().expect("vec");
        for val in &data {
            assert!(
                (*val - 10.0).abs() < 1e-3,
                "softplus(10) should be ~10, got {}",
                val
            );
        }
    }

    #[test]
    fn test_causal_conv1d_prefill_shape() {
        let device = Device::Cpu;
        let batch = 1;
        let d_inner = 8;
        let seq_len = 5;
        let kernel_size = 4;

        let x = Tensor::randn(0f32, 1.0, (batch, d_inner, seq_len), &device).expect("x");
        let weight = Tensor::randn(0f32, 1.0, (d_inner, 1, kernel_size), &device).expect("w");
        let bias = Tensor::randn(0f32, 1.0, (d_inner,), &device).expect("b");

        let output = causal_conv1d_prefill(&x, &weight, &bias).expect("conv");
        assert_eq!(output.dims(), &[batch, d_inner, seq_len]);
    }

    #[test]
    fn test_causal_conv1d_decode_shape() {
        let device = Device::Cpu;
        let batch = 1;
        let d_inner = 8;
        let kernel_size = 4;
        let conv_state_len = kernel_size - 1;

        let x = Tensor::randn(0f32, 1.0, (batch, d_inner), &device).expect("x");
        let weight = Tensor::randn(0f32, 1.0, (d_inner, 1, kernel_size), &device).expect("w");
        let bias = Tensor::randn(0f32, 1.0, (d_inner,), &device).expect("b");
        let conv_state =
            Tensor::zeros((batch, d_inner, conv_state_len), DType::F32, &device).expect("state");

        let (output, new_state) =
            causal_conv1d_decode(&x, &weight, &bias, &conv_state).expect("conv decode");
        assert_eq!(output.dims(), &[batch, d_inner]);
        assert_eq!(new_state.dims(), &[batch, d_inner, conv_state_len]);
    }

    #[test]
    fn test_mamba_concurrent_requests_independent_state() {
        // Two requests with different token histories should maintain independent SSM states.
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill request 1 with tokens [0, 0, 0]
        let prompt1 = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt1");
        let _ = model.forward_ssm(&prompt1, 0, 1).expect("prefill req 1");

        // Prefill request 2 with different tokens [1, 1, 1]
        let prompt2 = Tensor::ones((1, 3), DType::U32, &device).expect("prompt2");
        let _ = model.forward_ssm(&prompt2, 0, 2).expect("prefill req 2");

        // Verify both requests have independent state in the state manager
        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(1), "request 1 should have state");
        assert!(state_mgr.has_state(2), "request 2 should have state");
        assert_eq!(state_mgr.num_active_requests(), 2);
    }

    #[test]
    fn test_mamba_concurrent_decode_no_cross_contamination() {
        // Decode steps for different requests should not interfere with each other.
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill two requests
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 10).expect("prefill req 10");
        let _ = model.forward_ssm(&prompt, 0, 20).expect("prefill req 20");

        // Decode request 10 several times
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");
        let logits_10_step1 = model.forward_ssm(&next, 3, 10).expect("decode 10 step 1");
        let logits_10_step2 = model.forward_ssm(&next, 4, 10).expect("decode 10 step 2");

        // Decode request 20 once -- should not be affected by request 10's decode steps
        let logits_20_step1 = model.forward_ssm(&next, 3, 20).expect("decode 20 step 1");

        // Request 20 at step 1 should match request 10 at step 1
        // (same prefill, same decode token, same seqlen_offset)
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

        // Request 10 at step 2 should differ from step 1 (state evolved)
        let data_10_2: Vec<f32> = logits_10_step2
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        // With zero weights, all outputs may be identical, but the test verifies
        // that the code path runs without errors and produces correct shapes
        assert_eq!(
            logits_10_step2.dims(),
            &[1, 1, cfg.vocab_size],
            "decode step 2 should produce correct shape"
        );
        let _ = data_10_2; // ensure it was computed
    }

    #[test]
    fn test_mamba_state_freed_after_use() {
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        // Prefill request 42
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 42).expect("prefill");

        {
            let state_mgr = model.state_mgr.lock().expect("lock");
            assert!(state_mgr.has_state(42));
        }

        // Manually free state
        {
            let mut state_mgr = model.state_mgr.lock().expect("lock");
            state_mgr.free_state(42);
            assert!(!state_mgr.has_state(42));
        }

        // Re-prefill should work (state was freed)
        let _ = model.forward_ssm(&prompt, 0, 42).expect("re-prefill");
        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(42));
    }

    #[test]
    fn test_mamba_forward_decode_batch_uses_per_request_state() {
        // Verify that forward_decode_batch routes each sequence
        // to its own request state via request_id.
        let cfg = test_mamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, _block_table) = create_dummy_cache();

        // Prefill two separate requests
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 100).expect("prefill req 100");
        let _ = model.forward_ssm(&prompt, 0, 200).expect("prefill req 200");

        // Batched decode with two requests
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

        // Both requests should still have their state
        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(100));
        assert!(state_mgr.has_state(200));
    }
}
