//! PLaMo-2 model architecture (hybrid Attention + Mamba).
//!
//! PLaMo-2 (Preferred Networks) alternates between Mamba-2 SSM layers and
//! standard attention layers based on `mamba_step`:
//! - Every `mamba_step`-th layer at position `mamba_step // 2` is attention
//! - All other layers are Mamba-2 (multi-head SSM with conv1d)
//!
//! Architecture:
//! ```text
//! Embedding -> [Plamo2DecoderLayer x N] -> RMSNorm -> LM Head
//!
//! Plamo2DecoderLayer (Attention variant):
//!   pre_mixer_norm -> QK-norm + RoPE Attention -> post_mixer_norm
//!   -> pre_mlp_norm -> SwiGLU MLP -> post_mlp_norm
//!
//! Plamo2DecoderLayer (Mamba variant):
//!   pre_mixer_norm -> Mamba2 Mixer -> post_mixer_norm
//!   -> pre_mlp_norm -> SwiGLU MLP -> post_mlp_norm
//! ```
//!
//! Config keys from extra:
//! - `mamba_d_state`: SSM state size (default 64)
//! - `mamba_d_conv`: SSM conv kernel size (default 4)
//! - `mamba_num_heads`: number of SSM heads (default 64)
//! - `mamba_step`: layer alternation period (default 2)
//! - `hidden_size_per_head`: head dimension for both attention and Mamba

use std::sync::Mutex;

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};
use crate::ssm::selective_scan;
use crate::ssm::state::SSMStateManager;

// ─── Plamo2 Config ──────────────────────────────────────────────────────────

struct Plamo2Config {
    mamba_d_state: usize,
    mamba_d_conv: usize,
    mamba_num_heads: usize,
    mamba_step: usize,
    hidden_size_per_head: usize,
    /// Mamba intermediate_size = mamba_num_heads * hidden_size_per_head
    mamba_intermediate_size: usize,
    /// dt_rank = max(64, hidden_size / 16)
    time_step_rank: usize,
}

impl Plamo2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let mamba_d_state = cfg
            .extra
            .get("mamba_d_state")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(64);

        let mamba_d_conv = cfg
            .extra
            .get("mamba_d_conv")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let mamba_num_heads = cfg
            .extra
            .get("mamba_num_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(64);

        let mamba_step = cfg
            .extra
            .get("mamba_step")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let hidden_size_per_head = cfg
            .extra
            .get("hidden_size_per_head")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let mamba_intermediate_size = mamba_num_heads * hidden_size_per_head;
        let time_step_rank = std::cmp::max(64, cfg.hidden_size / 16);

        Self {
            mamba_d_state,
            mamba_d_conv,
            mamba_num_heads,
            mamba_step,
            hidden_size_per_head,
            mamba_intermediate_size,
            time_step_rank,
        }
    }

    /// Determine whether layer `i` is a Mamba layer.
    ///
    /// Logic from reference:
    /// - If num_hidden_layers <= mamba_step/2, use attention only for the last layer
    /// - Otherwise: layer i is Mamba if (i % mamba_step) != (mamba_step / 2)
    fn is_mamba(&self, layer_idx: usize, num_hidden_layers: usize) -> bool {
        if self.mamba_step <= 1 {
            return false;
        }
        if num_hidden_layers <= (self.mamba_step / 2) {
            return layer_idx != num_hidden_layers - 1;
        }
        (layer_idx % self.mamba_step) != (self.mamba_step / 2)
    }
}

// ─── Causal Conv1D helpers ──────────────────────────────────────────────────

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

fn softplus(x: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct Plamo2Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Plamo2Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_up_proj.gate"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_up_proj.up"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Mamba2 Mixer ───────────────────────────────────────────────────────────
//
// PLaMo2's Mamba mixer uses Mamba-2 with:
// - Merged in_proj for gate + hidden
// - Causal Conv1D
// - bcdt_proj: project to B, C, dt parameters
// - RMSNorm on B, C, dt
// - Selective scan with A, D, dt_bias parameters

struct Plamo2MambaMixer {
    in_proj_gate: Linear,
    in_proj_hidden: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    bcdt_proj: Linear,
    dt_proj: Linear,
    dt_norm: RmsNorm,
    b_norm: RmsNorm,
    c_norm: RmsNorm,
    /// A parameter: [num_heads] (pre-processed as -exp(A_log))
    a: Tensor,
    /// D skip connection: [num_heads]
    d_param: Tensor,
    /// dt_bias: [num_heads]
    dt_bias: Tensor,
    out_proj: Linear,
    d_inner: usize,
    d_state: usize,
    num_heads: usize,
    head_dim: usize,
    time_step_rank: usize,
}

impl Plamo2MambaMixer {
    fn new(cfg: &ModelConfig, p2_cfg: &Plamo2Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_inner = p2_cfg.mamba_intermediate_size;
        let d_state = p2_cfg.mamba_d_state;
        let d_conv = p2_cfg.mamba_d_conv;
        let num_heads = p2_cfg.mamba_num_heads;
        let head_dim = p2_cfg.hidden_size_per_head;
        let time_step_rank = p2_cfg.time_step_rank;

        // Gate and hidden in_proj (split from merged in_proj)
        let in_proj_gate = linear_no_bias(hidden_size, d_inner, vb.pp("in_proj.gate"))?;
        let in_proj_hidden = linear_no_bias(hidden_size, d_inner, vb.pp("in_proj.hidden"))?;

        // Conv1D weights
        let conv1d_weight = vb.pp("conv1d").get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(d_inner, "bias")?;

        // Project to B, C, dt
        let bcdt_proj = linear_no_bias(d_inner, time_step_rank + d_state * 2, vb.pp("bcdt_proj"))?;

        let dt_proj = linear_no_bias(time_step_rank, num_heads, vb.pp("dt_proj"))?;

        // Norms for B, C, dt
        let dt_norm = rms_norm(time_step_rank, cfg.rms_norm_eps, vb.pp("dt_norm"))?;
        let b_norm = rms_norm(d_state, cfg.rms_norm_eps, vb.pp("B_norm"))?;
        let c_norm = rms_norm(d_state, cfg.rms_norm_eps, vb.pp("C_norm"))?;

        // A parameter (stored as A_log, transformed to -exp(A_log))
        let a = vb.get(num_heads, "A")?;
        let d_param = vb.get(num_heads, "D")?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_gate,
            in_proj_hidden,
            conv1d_weight,
            conv1d_bias,
            bcdt_proj,
            dt_proj,
            dt_norm,
            b_norm,
            c_norm,
            a,
            d_param,
            dt_bias,
            out_proj,
            d_inner,
            d_state,
            num_heads,
            head_dim,
            time_step_rank,
        })
    }

    /// Project hidden states to B, C, dt parameters with normalization.
    fn project_ssm_parameters(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let ssm_params = self.bcdt_proj.forward(hidden_states)?;

        let b = ssm_params.narrow(2, 0, self.d_state)?;
        let c = ssm_params.narrow(2, self.d_state, self.d_state)?;
        let time_step = ssm_params.narrow(2, self.d_state * 2, self.time_step_rank)?;

        // Apply RMSNorm to each
        let time_step = self.dt_norm.forward(&time_step.contiguous()?)?;
        let b = self.b_norm.forward(&b.contiguous()?)?;
        let c = self.c_norm.forward(&c.contiguous()?)?;

        let dt = self.dt_proj.forward(&time_step)?;
        Ok((b, c, dt))
    }

    fn forward_prefill(&self, xs: &Tensor, ssm_state: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = xs.dims3()?;

        // Gate + hidden projection
        let gate = self.in_proj_gate.forward(xs)?;
        let hidden = self.in_proj_hidden.forward(xs)?;

        // Conv1D on hidden states
        let x_conv = hidden.transpose(1, 2)?;
        let x_conv = causal_conv1d_prefill(&x_conv, &self.conv1d_weight, &self.conv1d_bias)?;

        // Save conv state (last d_conv-1 columns)
        let d_conv = self.conv1d_weight.dims()[2];
        let conv_state_len = d_conv - 1;
        let new_conv_state = if seq_len >= conv_state_len {
            hidden
                .transpose(1, 2)?
                .narrow(2, seq_len - conv_state_len, conv_state_len)?
        } else {
            let pad = Tensor::zeros(
                (batch, self.d_inner, conv_state_len - seq_len),
                xs.dtype(),
                xs.device(),
            )?;
            Tensor::cat(&[&pad, &hidden.transpose(1, 2)?], 2)?
        };

        let hidden = x_conv.transpose(1, 2)?;
        let hidden = candle_nn::ops::silu(&hidden)?;

        // Project to B, C, dt
        let (b, c, dt) = self.project_ssm_parameters(&hidden)?;

        // Apply softplus to dt and add dt_bias
        // dt: [batch, seq_len, num_heads] — need to expand to [batch, seq_len, d_inner]
        let dt = softplus(&dt)?;
        let dt_bias = self.dt_bias.unsqueeze(0)?.unsqueeze(0)?;
        let dt = dt.broadcast_add(&dt_bias)?;
        // Expand dt from [batch, seq_len, num_heads] to [batch, seq_len, d_inner]
        // by repeating each head's dt value head_dim times
        let dt = dt
            .unsqueeze(3)?
            .expand((batch, seq_len, self.num_heads, self.head_dim))?
            .reshape((batch, seq_len, self.d_inner))?
            .contiguous()?;

        // Build A matrix: expand [num_heads] to [d_inner, d_state] shape for selective_scan
        let a_expanded = self
            .a
            .unsqueeze(1)?
            .expand((self.num_heads, self.head_dim))?
            .reshape(self.d_inner)?
            .unsqueeze(1)?
            .expand((self.d_inner, self.d_state))?;

        // D skip connection: expand [num_heads] -> [d_inner]
        let d_expanded = self
            .d_param
            .unsqueeze(1)?
            .expand((self.num_heads, self.head_dim))?
            .reshape(self.d_inner)?;

        let (ssm_out, new_ssm_state) = selective_scan(
            &hidden,
            &dt,
            &a_expanded,
            &b,
            &c,
            &d_expanded,
            Some(ssm_state),
        )?;

        // Gate with SiLU
        let gated = (&ssm_out * &candle_nn::ops::silu(&gate)?)?;
        let output = self.out_proj.forward(&gated)?;

        Ok((output, new_ssm_state, new_conv_state))
    }

    fn forward_decode(
        &self,
        xs: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // xs: [batch, 1, hidden]
        let gate = self.in_proj_gate.forward(xs)?;
        let hidden = self.in_proj_hidden.forward(xs)?;

        let gate = gate.squeeze(1)?;
        let hidden = hidden.squeeze(1)?;

        // Conv1D decode step
        let (hidden, new_conv_state) =
            causal_conv1d_decode(&hidden, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        let hidden = candle_nn::ops::silu(&hidden)?;

        // Project to B, C, dt (unsqueeze for seq_len=1)
        let (b, c, dt) = self.project_ssm_parameters(&hidden.unsqueeze(1)?)?;
        let b = b.squeeze(1)?;
        let c = c.squeeze(1)?;
        let dt = dt.squeeze(1)?;

        // Apply softplus to dt and add dt_bias
        // dt: [batch, num_heads] — need to expand to [batch, d_inner]
        let dt = softplus(&dt)?;
        let dt_bias = self.dt_bias.unsqueeze(0)?;
        let dt = dt.broadcast_add(&dt_bias)?;
        let batch = xs.dims3()?.0;
        // Expand dt from [batch, num_heads] to [batch, d_inner]
        let dt = dt
            .unsqueeze(2)?
            .expand((batch, self.num_heads, self.head_dim))?
            .reshape((batch, self.d_inner))?
            .contiguous()?;

        // A matrix expansion for selective_scan
        let a_expanded = self
            .a
            .unsqueeze(1)?
            .expand((self.num_heads, self.head_dim))?
            .reshape(self.d_inner)?
            .unsqueeze(1)?
            .expand((self.d_inner, self.d_state))?;

        let d_expanded = self
            .d_param
            .unsqueeze(1)?
            .expand((self.num_heads, self.head_dim))?
            .reshape(self.d_inner)?;

        let hidden_exp = hidden.unsqueeze(1)?;
        let dt_exp = dt.unsqueeze(1)?;
        let b_exp = b.unsqueeze(1)?;
        let c_exp = c.unsqueeze(1)?;

        let (ssm_out, new_ssm_state) = selective_scan(
            &hidden_exp,
            &dt_exp,
            &a_expanded,
            &b_exp,
            &c_exp,
            &d_expanded,
            Some(ssm_state),
        )?;

        let ssm_out = ssm_out.squeeze(1)?;
        let gated = (&ssm_out * &candle_nn::ops::silu(&gate)?)?;
        let output = self.out_proj.forward(&gated.unsqueeze(1)?)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

// ─── Attention Mixer ────────────────────────────────────────────────────────

struct Plamo2AttentionMixer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Plamo2AttentionMixer {
    fn new(cfg: &ModelConfig, p2_cfg: &Plamo2Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = p2_cfg.hidden_size_per_head;

        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        // Per-head QK normalization
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
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

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK normalization
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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
}

// ─── Layer Types ────────────────────────────────────────────────────────────

enum Plamo2MixerType {
    Mamba(Plamo2MambaMixer),
    Attention(Plamo2AttentionMixer),
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct Plamo2DecoderLayer {
    mixer: Plamo2MixerType,
    mlp: Plamo2Mlp,
    pre_mixer_norm: RmsNorm,
    post_mixer_norm: RmsNorm,
    pre_mlp_norm: RmsNorm,
    post_mlp_norm: RmsNorm,
}

impl Plamo2DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        p2_cfg: &Plamo2Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_mamba = p2_cfg.is_mamba(layer_idx, cfg.num_hidden_layers);

        let mixer = if is_mamba {
            Plamo2MixerType::Mamba(Plamo2MambaMixer::new(cfg, p2_cfg, vb.pp("mixer"))?)
        } else {
            Plamo2MixerType::Attention(Plamo2AttentionMixer::new(cfg, p2_cfg, vb.pp("mixer"))?)
        };

        let mlp = Plamo2Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let pre_mixer_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_mixer_norm"))?;
        let post_mixer_norm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_mixer_norm"))?;
        let pre_mlp_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_mlp_norm"))?;
        let post_mlp_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_mlp_norm"))?;

        Ok(Self {
            mixer,
            mlp,
            pre_mixer_norm,
            post_mixer_norm,
            pre_mlp_norm,
            post_mlp_norm,
        })
    }

    fn is_attention(&self) -> bool {
        matches!(self.mixer, Plamo2MixerType::Attention(_))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Plamo2ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Plamo2DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    /// SSM state for Mamba layers
    state_mgr: Mutex<SSMStateManager>,
    /// Number of attention layers (for KV cache layer count)
    num_attn_layers: usize,
    /// Mapping from model layer index to KV cache layer index (attention layers only)
    attn_layer_cache_idx: Vec<Option<usize>>,
}

impl Plamo2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let p2_cfg = Plamo2Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers").pp("layers");
        let mut num_attn_layers = 0;
        let mut attn_layer_cache_idx = Vec::with_capacity(cfg.num_hidden_layers);

        for i in 0..cfg.num_hidden_layers {
            let layer = Plamo2DecoderLayer::new(cfg, &p2_cfg, i, vb_layers.pp(i))?;
            if layer.is_attention() {
                attn_layer_cache_idx.push(Some(num_attn_layers));
                num_attn_layers += 1;
            } else {
                attn_layer_cache_idx.push(None);
            }
            layers.push(layer);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        // SSM state manager for Mamba layers
        let state_mgr = SSMStateManager::new(
            cfg.num_hidden_layers,
            p2_cfg.mamba_intermediate_size,
            p2_cfg.mamba_d_state,
            p2_cfg.mamba_d_conv,
            vb.dtype(),
            vb.device().clone(),
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            state_mgr: Mutex::new(state_mgr),
            num_attn_layers,
            attn_layer_cache_idx,
        })
    }

    /// Get the number of attention layers (for KV cache configuration).
    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }

    /// Forward pass with per-request SSM state tracking.
    pub fn forward_with_request_id(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        request_id: u64,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;

        // Manage SSM state
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
            .ok_or_else(|| candle_core::Error::Msg("SSM state not found".into()))?;

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

        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-mixer norm
            let residual = &hidden;
            let normed = layer.pre_mixer_norm.forward(&hidden)?;

            // Process through Mamba or Attention
            let mixer_output = match &layer.mixer {
                Plamo2MixerType::Mamba(mamba) => {
                    let ssm_state = &request_state.ssm_states[layer_idx].tensor;
                    let conv_state = &request_state.conv_states[layer_idx].tensor;

                    let is_prefill = seq_len > 1;
                    let (output, new_ssm, new_conv) = if is_prefill {
                        mamba.forward_prefill(&normed, ssm_state)?
                    } else {
                        mamba.forward_decode(&normed, ssm_state, conv_state)?
                    };

                    request_state.ssm_states[layer_idx].tensor = new_ssm;
                    request_state.conv_states[layer_idx].tensor = new_conv;
                    output
                }
                Plamo2MixerType::Attention(attn) => {
                    let cache_idx = self.attn_layer_cache_idx[layer_idx]
                        .expect("attention layer should have cache index");
                    attn.forward(
                        &normed,
                        attention_mask.as_ref(),
                        seqlen_offset,
                        kv_cache_mgr.engine_mut(cache_idx),
                        block_table,
                        slot_mapping,
                    )?
                }
            };

            // Post-mixer norm + residual
            let mixer_normed = layer.post_mixer_norm.forward(&mixer_output)?;
            let hidden_after_mixer = (residual + mixer_normed)?;

            // Pre-MLP norm + MLP + post-MLP norm + residual
            let residual_mlp = &hidden_after_mixer;
            let mlp_normed = layer.pre_mlp_norm.forward(&hidden_after_mixer)?;
            let mlp_out = layer.mlp.forward(&mlp_normed)?;
            let mlp_post_normed = layer.post_mlp_norm.forward(&mlp_out)?;
            hidden = (residual_mlp + mlp_post_normed)?;
        }

        let hidden = self.norm.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }
}

impl crate::engine::ModelForward for Plamo2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_with_request_id(
            input_ids,
            seqlen_offset,
            0,
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
        let mut outputs = Vec::with_capacity(sequences.len());
        for (i, seq) in sequences.iter().enumerate() {
            let token = input_ids.narrow(0, i, 1)?;
            let block_table = BlockTable::from_block_ids(seq.block_ids.clone(), seq.seqlen_offset);
            let logits = self.forward_with_request_id(
                &token,
                seq.seqlen_offset,
                seq.request_id,
                kv_cache_mgr,
                &block_table,
                &seq.slot_mapping,
            )?;
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

    fn test_plamo2_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("mamba_d_state".to_string(), serde_json::json!(8));
        extra.insert("mamba_d_conv".to_string(), serde_json::json!(4));
        extra.insert("mamba_num_heads".to_string(), serde_json::json!(4));
        extra.insert("mamba_step".to_string(), serde_json::json!(2));
        extra.insert("hidden_size_per_head".to_string(), serde_json::json!(16));

        ModelConfig {
            architectures: vec!["Plamo2ForCausalLM".to_string()],
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

    fn create_plamo2_cache(
        cfg: &ModelConfig,
        num_attn_layers: usize,
    ) -> (KVCacheManager, BlockTable) {
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

    #[test]
    fn test_plamo2_config_extraction() {
        let cfg = test_plamo2_config();
        let p2_cfg = Plamo2Config::from_model_config(&cfg);

        assert_eq!(p2_cfg.mamba_d_state, 8);
        assert_eq!(p2_cfg.mamba_d_conv, 4);
        assert_eq!(p2_cfg.mamba_num_heads, 4);
        assert_eq!(p2_cfg.mamba_step, 2);
        assert_eq!(p2_cfg.hidden_size_per_head, 16);
        assert_eq!(p2_cfg.mamba_intermediate_size, 64); // 4 * 16
        assert_eq!(p2_cfg.time_step_rank, 64); // max(64, 64/16)
    }

    #[test]
    fn test_plamo2_layer_classification() {
        let cfg = test_plamo2_config();
        let p2_cfg = Plamo2Config::from_model_config(&cfg);

        // mamba_step=2: attention at (i % 2 == 1), mamba at (i % 2 == 0)
        // So layers 0,2 are mamba, layers 1,3 are attention
        assert!(
            p2_cfg.is_mamba(0, cfg.num_hidden_layers),
            "layer 0 should be mamba"
        );
        assert!(
            !p2_cfg.is_mamba(1, cfg.num_hidden_layers),
            "layer 1 should be attention"
        );
        assert!(
            p2_cfg.is_mamba(2, cfg.num_hidden_layers),
            "layer 2 should be mamba"
        );
        assert!(
            !p2_cfg.is_mamba(3, cfg.num_hidden_layers),
            "layer 3 should be attention"
        );
    }

    #[test]
    fn test_plamo2_construction() {
        let cfg = test_plamo2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Plamo2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Plamo2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        // With mamba_step=2, half the layers are attention
        assert_eq!(model.num_attn_layers, 2);
    }

    #[test]
    fn test_plamo2_forward_shape() {
        let cfg = test_plamo2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_plamo2_cache(&cfg, model.num_attn_layers);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward_with_request_id(
                &input_ids,
                0,
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
    fn test_plamo2_prefill_then_decode() {
        let cfg = test_plamo2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_plamo2_cache(&cfg, model.num_attn_layers);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);
        let logits = model
            .forward_with_request_id(
                &prompt,
                0,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
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
            .forward_with_request_id(&next, 3, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_plamo2_model_forward_trait() {
        let cfg = test_plamo2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_plamo2_cache(&cfg, model.num_attn_layers);
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
    fn test_plamo2_device() {
        let cfg = test_plamo2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo2ForCausalLM::new(&cfg, vb).expect("build model");
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_plamo2_concurrent_requests() {
        let cfg = test_plamo2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_plamo2_cache(&cfg, model.num_attn_layers);

        // Prefill two requests with different IDs
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let _ = model
            .forward_with_request_id(
                &prompt,
                0,
                1,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 1");
        let _ = model
            .forward_with_request_id(
                &prompt,
                0,
                2,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 2");

        let state_mgr = model.state_mgr.lock().expect("lock");
        assert!(state_mgr.has_state(1), "request 1 should have state");
        assert!(state_mgr.has_state(2), "request 2 should have state");
        assert_eq!(state_mgr.num_active_requests(), 2);
    }

    #[test]
    fn test_plamo2_mamba_step_edge_case() {
        // Test with mamba_step=1 (all layers should be attention)
        let mut cfg = test_plamo2_config();
        cfg.extra
            .insert("mamba_step".to_string(), serde_json::json!(1));

        let p2_cfg = Plamo2Config::from_model_config(&cfg);
        for i in 0..cfg.num_hidden_layers {
            assert!(
                !p2_cfg.is_mamba(i, cfg.num_hidden_layers),
                "mamba_step=1 should make all layers attention"
            );
        }
    }

    #[test]
    fn test_plamo2_few_layers_edge_case() {
        // Test where num_hidden_layers <= mamba_step/2
        // Should use attention in last layer only
        let mut cfg = test_plamo2_config();
        cfg.num_hidden_layers = 1;
        cfg.extra
            .insert("mamba_step".to_string(), serde_json::json!(4));

        let p2_cfg = Plamo2Config::from_model_config(&cfg);
        // Only 1 layer, and 1 <= 4/2=2, so last layer (0) is attention
        assert!(!p2_cfg.is_mamba(0, 1), "single layer should be attention");
    }
}
