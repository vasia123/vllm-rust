//! Mamba-2 (State Space Duality) model architecture.
//!
//! Mamba-2 extends Mamba-1 with multi-head SSM and a gated output norm:
//! - Multiple SSM heads (like attention heads), each with its own scalar A
//! - Conv operates on the full xBC tensor (not just x)
//! - Output gate: `rms_norm_gated(ssm_out, gate)` replaces the pre-norm
//!
//! Architecture per block (MambaMixer2):
//! ```text
//! in_proj → [gate | xBC | dt]
//!              conv1d(xBC) → [x | B | C]  (x=d_inner, B=n_groups*d_state, C=same)
//!              silu(x) → multi-head SSM(x, B, C, dt)
//!              rms_norm_gated(ssm_out, gate)
//!           → out_proj
//! ```
//!
//! Weight paths (HuggingFace Mamba-2 / state-spaces/mamba2):
//! - `backbone.embeddings.weight`
//! - `backbone.layers.{i}.norm.weight`   ← pre-norm (at decoder layer level)
//! - `backbone.layers.{i}.mixer.*`       ← MambaMixer2 components
//! - `backbone.norm_f.weight`
//! - `lm_head.weight`
//!
//! NOTE: The checkpoint stores `A_log`; vLLM's weight loader renames it to `A`
//! and applies `-exp()` during loading. We load `A` directly and assume the
//! checkpoint has already been converted, OR apply `-exp(A_log)` during init
//! if `A_log` is found.

use std::sync::Mutex;

use crate::layers::{rms_norm, RmsNorm};
use crate::ssm::{causal_conv1d_decode, causal_conv1d_prefill};
use crate::ssm::{rms_norm_gated, selective_scan, SSMStateManager};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};

// ─── Config ─────────────────────────────────────────────────────────────────

struct Mamba2Config {
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    #[allow(dead_code)]
    expand: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
    rms_norm_eps: f64,
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

        let rms_norm_eps = cfg
            .extra
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        Self {
            d_inner,
            d_state,
            d_conv,
            expand,
            num_heads,
            head_dim,
            n_groups,
            rms_norm_eps,
        }
    }

    /// `conv_dim = d_inner + 2 * n_groups * d_state`
    ///
    /// This is the size of the tensor passed through causal_conv1d in Mamba-2.
    fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_groups * self.d_state
    }
}

/// Softplus activation: `log(1 + exp(x))`
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.dims(), x.dtype(), x.device())?;
    (&x.exp()? + &ones)?.log()
}

// ─── MambaMixer2 (the SSM mixer block, weight prefix: `mixer.*`) ─────────────

struct Mamba2Block {
    /// in_proj output: [gate(d_inner) | xBC(conv_dim) | dt(num_heads)]
    in_proj: Linear,
    /// Depthwise causal conv on xBC: [conv_dim, 1, d_conv]
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    /// Gated output norm weight: [d_inner]
    out_norm_weight: Tensor,
    out_proj: Linear,
    /// Per-head A: [num_heads] (stored as `-exp(A_log)` in HF checkpoints)
    a: Tensor,
    /// Per-head D skip connection: [num_heads]
    d_param: Tensor,
    /// dt_bias: [num_heads]
    dt_bias: Tensor,
    d_inner: usize,
    d_state: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
    conv_dim: usize,
    rms_norm_eps: f64,
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
        let conv_dim = m2_cfg.conv_dim();

        // in_proj: hidden → gate(d_inner) + xBC(conv_dim) + dt(num_heads)
        let in_proj_size = d_inner + conv_dim + num_heads;
        let in_proj = linear_no_bias(hidden_size, in_proj_size, vb.pp("in_proj"))?;

        // Depthwise causal conv on the full xBC tensor
        let conv1d_weight = vb.pp("conv1d").get((conv_dim, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(conv_dim, "bias")?;

        // Gated output norm (Mixer2RMSNormGated), weight path: mixer.norm.weight
        let out_norm_weight = vb.pp("norm").get(d_inner, "weight")?;

        // Out-projection: d_inner → hidden
        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("out_proj"))?;

        // Per-head parameters.
        // NOTE: HF checkpoints store A_log; vLLM renames to A and applies
        // `-exp()` at load time. Here we load A directly; for real checkpoints
        // the weight loader must perform this conversion.
        let a = vb.get(num_heads, "A")?;
        let d_param = vb.get(num_heads, "D")?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            out_norm_weight,
            out_proj,
            a,
            d_param,
            dt_bias,
            d_inner,
            d_state,
            num_heads,
            head_dim,
            n_groups,
            conv_dim,
            rms_norm_eps: m2_cfg.rms_norm_eps,
        })
    }

    /// Prefill (full-sequence) forward.
    ///
    /// Returns `(output [batch, seq_len, hidden], new_ssm_state, new_conv_state)`.
    /// `normed_hidden` is the pre-normed input (pre-norm applied by the decoder layer).
    fn forward_prefill(
        &self,
        normed_hidden: &Tensor,
        original_hidden: &Tensor,
        ssm_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = normed_hidden.dims3()?;

        // in_proj → gate + xBC + dt
        let proj = self.in_proj.forward(normed_hidden)?;
        let gate = proj.narrow(2, 0, self.d_inner)?;
        let xbc = proj.narrow(2, self.d_inner, self.conv_dim)?;
        let dt = proj.narrow(2, self.d_inner + self.conv_dim, self.num_heads)?;

        // Causal conv1d over the full xBC tensor
        let xbc_t = xbc.transpose(1, 2)?; // [batch, conv_dim, seq_len]
        let xbc_conv_t = causal_conv1d_prefill(&xbc_t, &self.conv1d_weight, &self.conv1d_bias)?;

        // Extract conv state: last (d_conv-1) columns of the pre-conv xBC
        let d_conv = self.conv1d_weight.dims()[2];
        let conv_state_len = d_conv - 1;
        let new_conv_state = if seq_len >= conv_state_len {
            xbc_t.narrow(2, seq_len - conv_state_len, conv_state_len)?
        } else {
            let pad = Tensor::zeros(
                (batch, self.conv_dim, conv_state_len - seq_len),
                xbc_t.dtype(),
                xbc_t.device(),
            )?;
            Tensor::cat(&[&pad, &xbc_t], 2)?
        };

        let xbc_conv = xbc_conv_t.transpose(1, 2)?; // [batch, seq_len, conv_dim]

        // Split xBC_conv → x + B + C, then SiLU on x
        let x = xbc_conv.narrow(2, 0, self.d_inner)?;
        let b_proj = xbc_conv.narrow(2, self.d_inner, self.n_groups * self.d_state)?;
        let c_proj = xbc_conv.narrow(
            2,
            self.d_inner + self.n_groups * self.d_state,
            self.n_groups * self.d_state,
        )?;
        let x_ssm = candle_nn::ops::silu(&x)?;

        // dt_bias + softplus → delta [batch, seq_len, num_heads]
        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

        // Multi-head SSM (sequential recurrence per head; mathematically equivalent to SSD)
        let heads_per_group = self.num_heads / self.n_groups;
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut new_ssm_states = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let group_idx = h / heads_per_group;

            let x_h = x_ssm.narrow(2, h * self.head_dim, self.head_dim)?;
            let delta_h = delta
                .narrow(2, h, 1)?
                .broadcast_as((batch, seq_len, self.head_dim))?
                .contiguous()?;

            // A is scalar per head → broadcast to [head_dim, d_state]
            let a_h = self
                .a
                .narrow(0, h, 1)?
                .broadcast_as((self.head_dim, self.d_state))?
                .contiguous()?;

            let b_h = b_proj.narrow(2, group_idx * self.d_state, self.d_state)?;
            let c_h = c_proj.narrow(2, group_idx * self.d_state, self.d_state)?;

            let d_h = self
                .d_param
                .narrow(0, h, 1)?
                .broadcast_as((self.head_dim,))?
                .contiguous()?;

            let ssm_h = ssm_state.narrow(1, h * self.head_dim, self.head_dim)?;

            let (out_h, new_state_h) =
                selective_scan(&x_h, &delta_h, &a_h, &b_h, &c_h, &d_h, Some(&ssm_h))?;

            head_outputs.push(out_h);
            new_ssm_states.push(new_state_h);
        }

        // Concat head outputs: [batch, seq_len, d_inner]
        let ssm_out = Tensor::cat(&head_outputs, 2)?;
        let new_ssm_state = Tensor::cat(&new_ssm_states, 1)?;

        // Gated output norm: rms_norm(ssm_out * silu(gate))
        let y = rms_norm_gated(
            &ssm_out,
            &gate,
            &self.out_norm_weight,
            self.rms_norm_eps,
            false,
        )?;

        // Out-projection + residual
        let proj_out = self.out_proj.forward(&y)?;
        let output = (original_hidden + proj_out)?;

        Ok((output, new_ssm_state, new_conv_state))
    }

    /// Decode (single-token) forward.
    fn forward_decode(
        &self,
        normed_hidden: &Tensor,
        original_hidden: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // in_proj: [batch, 1, hidden] → squeeze → [batch, in_proj_size]
        let proj = self.in_proj.forward(normed_hidden)?.squeeze(1)?;

        let gate = proj.narrow(1, 0, self.d_inner)?;
        let xbc = proj.narrow(1, self.d_inner, self.conv_dim)?;
        let dt = proj.narrow(1, self.d_inner + self.conv_dim, self.num_heads)?;

        // Conv1d decode on full xBC
        let (xbc_conv, new_conv_state) =
            causal_conv1d_decode(&xbc, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        // Split xBC_conv → x + B + C, SiLU on x
        let x = xbc_conv.narrow(1, 0, self.d_inner)?;
        let b_proj = xbc_conv.narrow(1, self.d_inner, self.n_groups * self.d_state)?;
        let c_proj = xbc_conv.narrow(
            1,
            self.d_inner + self.n_groups * self.d_state,
            self.n_groups * self.d_state,
        )?;
        let x_ssm = candle_nn::ops::silu(&x)?;

        // dt: [batch, num_heads] → broadcast_add dt_bias
        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

        let batch = normed_hidden.dims()[0];
        let heads_per_group = self.num_heads / self.n_groups;
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut new_ssm_states = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let group_idx = h / heads_per_group;

            let x_h = x_ssm.narrow(1, h * self.head_dim, self.head_dim)?;
            let delta_h = delta
                .narrow(1, h, 1)?
                .broadcast_as((batch, self.head_dim))?
                .contiguous()?;

            let a_h = self
                .a
                .narrow(0, h, 1)?
                .broadcast_as((self.head_dim, self.d_state))?
                .contiguous()?;

            let b_h = b_proj.narrow(1, group_idx * self.d_state, self.d_state)?;
            let c_h = c_proj.narrow(1, group_idx * self.d_state, self.d_state)?;

            let d_h = self
                .d_param
                .narrow(0, h, 1)?
                .broadcast_as((self.head_dim,))?
                .contiguous()?;

            let ssm_h = ssm_state.narrow(1, h * self.head_dim, self.head_dim)?;

            // Expand to [batch, 1, ...] for selective_scan (seq_len=1)
            let (out_h, new_state_h) = selective_scan(
                &x_h.unsqueeze(1)?,
                &delta_h.unsqueeze(1)?,
                &a_h,
                &b_h.unsqueeze(1)?,
                &c_h.unsqueeze(1)?,
                &d_h,
                Some(&ssm_h),
            )?;

            head_outputs.push(out_h.squeeze(1)?);
            new_ssm_states.push(new_state_h);
        }

        let ssm_out = Tensor::cat(&head_outputs, 1)?; // [batch, d_inner]
        let new_ssm_state = Tensor::cat(&new_ssm_states, 1)?;

        // Gated output norm (operate on [batch, 1, d_inner])
        let ssm_out_3d = ssm_out.unsqueeze(1)?;
        let gate_3d = gate.unsqueeze(1)?;
        let y_3d = rms_norm_gated(
            &ssm_out_3d,
            &gate_3d,
            &self.out_norm_weight,
            self.rms_norm_eps,
            false,
        )?;

        let proj_out = self.out_proj.forward(&y_3d)?;
        let output = (original_hidden + proj_out)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

// ─── Decoder layer (pre-norm + mixer) ────────────────────────────────────────

struct Mamba2DecoderLayer {
    /// Pre-norm: weight at `layers.{i}.norm.weight`
    pre_norm: RmsNorm,
    /// MambaMixer2: weights at `layers.{i}.mixer.*`
    mixer: Mamba2Block,
}

impl Mamba2DecoderLayer {
    fn new(cfg: &ModelConfig, m2_cfg: &Mamba2Config, vb_layer: VarBuilder) -> Result<Self> {
        let pre_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_layer.pp("norm"))?;
        let mixer = Mamba2Block::new(cfg, m2_cfg, vb_layer.pp("mixer"))?;
        Ok(Self { pre_norm, mixer })
    }

    fn forward_prefill(
        &self,
        hidden: &Tensor,
        ssm_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = self.pre_norm.forward(hidden)?;
        self.mixer.forward_prefill(&normed, hidden, ssm_state)
    }

    fn forward_decode(
        &self,
        hidden: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = self.pre_norm.forward(hidden)?;
        self.mixer
            .forward_decode(&normed, hidden, ssm_state, conv_state)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Mamba2ForCausalLM {
    embeddings: Embedding,
    layers: Vec<Mamba2DecoderLayer>,
    norm_f: RmsNorm,
    lm_head: Linear,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    state_mgr: Mutex<SSMStateManager>,
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
            layers.push(Mamba2DecoderLayer::new(cfg, &m2_cfg, vb_layers.pp(i))?);
        }

        let norm_f = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_backbone.pp("norm_f"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embeddings.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let conv_dim = m2_cfg.conv_dim();
        let state_mgr = SSMStateManager::new_with_conv_channels(
            cfg.num_hidden_layers,
            m2_cfg.d_inner,
            m2_cfg.d_state,
            m2_cfg.d_conv,
            conv_dim,
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
        })
    }

    /// SSM-native forward pass.
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
        self.forward_ssm(input_ids, seqlen_offset, 0)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
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
        extra.insert("rms_norm_eps".to_string(), serde_json::json!(1e-5));

        ModelConfig {
            architectures: vec!["Mamba2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_hidden_layers: 2,
            intermediate_size: 128, // d_inner = 128 = 4 heads * 32 head_dim
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
        // conv_dim = 128 + 2*1*8 = 144
        assert_eq!(m2_cfg.conv_dim(), 144);
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
    fn test_mamba2_in_proj_size() {
        // in_proj output = d_inner + conv_dim + num_heads = 128 + 144 + 4 = 276
        let cfg = test_mamba2_config();
        let m2_cfg = Mamba2Config::from_model_config(&cfg);
        let expected = m2_cfg.d_inner + m2_cfg.conv_dim() + m2_cfg.num_heads;
        assert_eq!(expected, 276);
    }

    #[test]
    fn test_mamba2_forward_shape() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).expect("input");
        let logits = model.forward_ssm(&input_ids, 0, 0).expect("forward");

        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
    }

    #[test]
    fn test_mamba2_single_token_decode() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 0).expect("prefill");

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

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 1).expect("prefill req 1");
        let _ = model
            .forward_ssm(
                &Tensor::ones((1, 3), DType::U32, &device).expect("p2"),
                0,
                2,
            )
            .expect("prefill req 2");

        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(1));
        assert!(state_mgr.has_state(2));
        assert_eq!(state_mgr.num_active_requests(), 2);
    }

    #[test]
    fn test_mamba2_conv_state_shape() {
        // conv state should use conv_dim channels, not d_inner
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 5).expect("prefill");

        let mut state_mgr = model.state_mgr.lock().expect("lock");
        let state = state_mgr.get_state(5).expect("state exists");
        // conv_dim = 128 + 2*1*8 = 144; d_conv-1 = 3
        assert_eq!(
            state.conv_states[0].tensor.dims(),
            &[1, 144, 3],
            "conv state should have conv_dim channels"
        );
        // SSM state: [1, d_inner, d_state] = [1, 128, 8]
        assert_eq!(
            state.ssm_states[0].tensor.dims(),
            &[1, 128, 8],
            "SSM state should have d_inner channels"
        );
    }

    #[test]
    fn test_mamba2_forward_decode_batch() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");
        let (mut kv_cache_mgr, _block_table) = create_dummy_cache();

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 100).expect("prefill 100");
        let _ = model.forward_ssm(&prompt, 0, 200).expect("prefill 200");

        let batch_input = Tensor::zeros((2, 1), DType::U32, &device).expect("batch");
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

        assert_eq!(logits.dims(), &[2, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_mamba2_state_freed_and_reallocated() {
        let cfg = test_mamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Mamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let _ = model.forward_ssm(&prompt, 0, 42).expect("prefill");

        {
            let state_mgr = model.state_mgr.lock().expect("lock");
            assert!(state_mgr.has_state(42));
        }
        {
            let mut state_mgr = model.state_mgr.lock().expect("lock");
            state_mgr.free_state(42);
        }

        let _ = model.forward_ssm(&prompt, 0, 42).expect("re-prefill");
        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(42));
    }
}
