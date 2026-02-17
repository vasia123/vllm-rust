//! FalconH1 (hybrid parallel SSM + attention) model architecture.
//!
//! FalconH1 uses a unique parallel hybrid architecture where each layer
//! processes input through BOTH an attention branch and a Mamba2 SSM branch
//! simultaneously. The outputs are combined with configurable multipliers.
//!
//! Architecture:
//! ```text
//! Embedding (*embedding_multiplier) -> [FalconH1Layer x N] -> RMSNorm -> LM Head
//!
//! FalconH1Layer (parallel hybrid):
//!   RMSNorm -> ┬─ Attention (*attn_in_mul) -> (*attn_out_mul)  ─┬─ + residual
//!              └─ Mamba2    (*ssm_in_mul)  -> (*ssm_out_mul)   ─┘
//!           -> RMSNorm -> MLP -> + residual
//! ```
//!
//! Config keys from extra:
//! - `mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `mamba_n_heads`, etc.
//! - `embedding_multiplier`: scale factor for embeddings (default 1.0)
//! - `ssm_in_multiplier`, `ssm_out_multiplier`: SSM branch scaling
//! - `attention_in_multiplier`, `attention_out_multiplier`: attention branch scaling
//! - `mlp_multipliers`: [gate_mul, down_mul] for MLP scaling
//! - `lm_head_multiplier`: logits scaling (default 1.0)

use std::sync::Mutex;

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::ssm::selective_scan;
use crate::ssm::state::SSMStateManager;

// ─── Config Extraction ──────────────────────────────────────────────────────

struct FalconH1Config {
    mamba_d_state: usize,
    mamba_d_conv: usize,
    #[allow(dead_code)]
    mamba_expand: usize,
    mamba_n_heads: usize,
    mamba_d_head: usize,
    mamba_n_groups: usize,
    d_inner: usize,
    embedding_multiplier: f64,
    ssm_in_multiplier: f64,
    ssm_out_multiplier: f64,
    attention_in_multiplier: f64,
    attention_out_multiplier: f64,
    lm_head_multiplier: f64,
}

impl FalconH1Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let mamba_expand = cfg
            .extra
            .get("mamba_expand")
            .and_then(|v| v.as_f64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let d_inner = cfg
            .extra
            .get("mamba_d_ssm")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size * mamba_expand);

        let mamba_d_state = cfg
            .extra
            .get("mamba_d_state")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);

        let mamba_d_conv = cfg
            .extra
            .get("mamba_d_conv")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let mamba_n_heads = cfg
            .extra
            .get("mamba_n_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let mamba_d_head = cfg
            .extra
            .get("mamba_d_head")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(if mamba_n_heads > 0 {
                d_inner / mamba_n_heads
            } else {
                d_inner
            });

        let mamba_n_groups = cfg
            .extra
            .get("mamba_n_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let embedding_multiplier = cfg
            .extra
            .get("embedding_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let ssm_in_multiplier = cfg
            .extra
            .get("ssm_in_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let ssm_out_multiplier = cfg
            .extra
            .get("ssm_out_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let attention_in_multiplier = cfg
            .extra
            .get("attention_in_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let attention_out_multiplier = cfg
            .extra
            .get("attention_out_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let lm_head_multiplier = cfg
            .extra
            .get("lm_head_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Self {
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
            mamba_n_heads,
            mamba_d_head,
            mamba_n_groups,
            d_inner,
            embedding_multiplier,
            ssm_in_multiplier,
            ssm_out_multiplier,
            attention_in_multiplier,
            attention_out_multiplier,
            lm_head_multiplier,
        }
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

struct FalconH1Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FalconH1Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
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

// ─── Mamba2 SSM Block ───────────────────────────────────────────────────────

struct FalconH1Mamba2Block {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    out_proj: Linear,
    a: Tensor,
    d_param: Tensor,
    dt_bias: Tensor,
    norm_b: Option<RmsNorm>,
    norm_c: Option<RmsNorm>,
    d_inner: usize,
    d_state: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
}

impl FalconH1Mamba2Block {
    fn new(cfg: &ModelConfig, falcon_cfg: &FalconH1Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_inner = falcon_cfg.d_inner;
        let d_state = falcon_cfg.mamba_d_state;
        let d_conv = falcon_cfg.mamba_d_conv;
        let num_heads = falcon_cfg.mamba_n_heads;
        let head_dim = falcon_cfg.mamba_d_head;
        let n_groups = falcon_cfg.mamba_n_groups;

        let in_proj_size = d_inner + 2 * n_groups * d_state + num_heads;
        let in_proj = linear_no_bias(hidden_size, in_proj_size, vb.pp("in_proj"))?;

        let conv1d_weight = vb.pp("conv1d").get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(d_inner, "bias")?;

        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("out_proj"))?;

        let a = vb.get(num_heads, "A")?;
        let d_param = vb.get(num_heads, "D")?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;

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

    fn forward_prefill(&self, xs: &Tensor, ssm_state: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = xs.dims3()?;

        let proj = self.in_proj.forward(xs)?;

        let x = proj.narrow(2, 0, self.d_inner)?;
        let b_offset = self.d_inner;
        let b_proj = proj.narrow(2, b_offset, self.n_groups * self.d_state)?;
        let c_offset = b_offset + self.n_groups * self.d_state;
        let c_proj = proj.narrow(2, c_offset, self.n_groups * self.d_state)?;
        let dt_offset = c_offset + self.n_groups * self.d_state;
        let dt = proj.narrow(2, dt_offset, self.num_heads)?;

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

        let x_conv = x.transpose(1, 2)?;
        let x_conv = causal_conv1d_prefill(&x_conv, &self.conv1d_weight, &self.conv1d_bias)?;

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

        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

        let heads_per_group = self.num_heads / self.n_groups;
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut new_ssm_states = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let group_idx = h / heads_per_group;

            let x_h = x_ssm.narrow(2, h * self.head_dim, self.head_dim)?;
            let delta_h = delta.narrow(2, h, 1)?;
            let delta_h = delta_h.broadcast_as((batch, seq_len, self.head_dim))?;

            let a_h_scalar = self.a.narrow(0, h, 1)?;
            let a_h = a_h_scalar
                .broadcast_as((self.head_dim, self.d_state))?
                .contiguous()?;

            let b_h = b_proj.narrow(2, group_idx * self.d_state, self.d_state)?;
            let c_h = c_proj.narrow(2, group_idx * self.d_state, self.d_state)?;

            let d_h_scalar = self.d_param.narrow(0, h, 1)?;
            let d_h = d_h_scalar.broadcast_as((self.head_dim,))?.contiguous()?;

            let ssm_h = ssm_state.narrow(1, h * self.head_dim, self.head_dim)?;

            let (out_h, new_state_h) =
                selective_scan(&x_h, &delta_h, &a_h, &b_h, &c_h, &d_h, Some(&ssm_h))?;

            head_outputs.push(out_h);
            new_ssm_states.push(new_state_h);
        }

        let ssm_out = Tensor::cat(&head_outputs, 2)?;
        let new_ssm_state = Tensor::cat(&new_ssm_states, 1)?;

        let output = self.out_proj.forward(&ssm_out)?;

        Ok((output, new_ssm_state, new_conv_state))
    }

    fn forward_decode(
        &self,
        xs: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let proj = self.in_proj.forward(xs)?;
        let proj = proj.squeeze(1)?;

        let x = proj.narrow(1, 0, self.d_inner)?;
        let b_offset = self.d_inner;
        let b_proj = proj.narrow(1, b_offset, self.n_groups * self.d_state)?;
        let c_offset = b_offset + self.n_groups * self.d_state;
        let c_proj = proj.narrow(1, c_offset, self.n_groups * self.d_state)?;
        let dt_offset = c_offset + self.n_groups * self.d_state;
        let dt = proj.narrow(1, dt_offset, self.num_heads)?;

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

        let (x_conv, new_conv_state) =
            causal_conv1d_decode(&x, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

        let batch = xs.dims()[0];
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

        let ssm_out = Tensor::cat(&head_outputs, 1)?;
        let new_ssm_state = Tensor::cat(&new_ssm_states, 1)?;

        let output = self.out_proj.forward(&ssm_out.unsqueeze(1)?)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

// ─── Attention Block ────────────────────────────────────────────────────────

struct FalconH1Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl FalconH1Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

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

// ─── Parallel Hybrid Layer ──────────────────────────────────────────────────

struct FalconH1ParallelHybridLayer {
    self_attn: FalconH1Attention,
    mamba: FalconH1Mamba2Block,
    feed_forward: FalconH1Mlp,
    input_layernorm: RmsNorm,
    pre_ff_layernorm: RmsNorm,
    ssm_in_multiplier: f64,
    ssm_out_multiplier: f64,
    attention_in_multiplier: f64,
    attention_out_multiplier: f64,
}

impl FalconH1ParallelHybridLayer {
    fn new(cfg: &ModelConfig, falcon_cfg: &FalconH1Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = FalconH1Attention::new(cfg, vb.pp("self_attn"))?;
        let mamba = FalconH1Mamba2Block::new(cfg, falcon_cfg, vb.pp("mamba"))?;
        let feed_forward = FalconH1Mlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("feed_forward"),
        )?;

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_ff_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_ff_layernorm"))?;

        Ok(Self {
            self_attn,
            mamba,
            feed_forward,
            input_layernorm,
            pre_ff_layernorm,
            ssm_in_multiplier: falcon_cfg.ssm_in_multiplier,
            ssm_out_multiplier: falcon_cfg.ssm_out_multiplier,
            attention_in_multiplier: falcon_cfg.attention_in_multiplier,
            attention_out_multiplier: falcon_cfg.attention_out_multiplier,
        })
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct FalconH1ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<FalconH1ParallelHybridLayer>,
    final_layernorm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    state_mgr: Mutex<SSMStateManager>,
    num_layers: usize,
    embedding_multiplier: f64,
    lm_head_multiplier: f64,
}

impl FalconH1ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let falcon_cfg = FalconH1Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");

        for i in 0..cfg.num_hidden_layers {
            layers.push(FalconH1ParallelHybridLayer::new(
                cfg,
                &falcon_cfg,
                vb_layers.pp(i),
            )?);
        }

        let final_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_model.pp("final_layernorm"),
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let state_mgr = SSMStateManager::new(
            cfg.num_hidden_layers,
            falcon_cfg.d_inner,
            falcon_cfg.mamba_d_state,
            falcon_cfg.mamba_d_conv,
            vb.dtype(),
            vb.device().clone(),
        );

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            state_mgr: Mutex::new(state_mgr),
            num_layers: cfg.num_hidden_layers,
            embedding_multiplier: falcon_cfg.embedding_multiplier,
            lm_head_multiplier: falcon_cfg.lm_head_multiplier,
        })
    }

    /// Each layer runs attention and SSM in parallel, sums their outputs.
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

        // Apply embedding multiplier
        if (self.embedding_multiplier - 1.0).abs() > 1e-9 {
            hidden = (hidden * self.embedding_multiplier)?;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = layer.input_layernorm.forward(&hidden)?;

            // Attention branch (with input multiplier)
            let attn_input = if (layer.attention_in_multiplier - 1.0).abs() > 1e-9 {
                (normed.clone() * layer.attention_in_multiplier)?
            } else {
                normed.clone()
            };
            let attn_output = layer.self_attn.forward(
                &attn_input,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;

            // SSM branch (with input multiplier)
            let ssm_input = if (layer.ssm_in_multiplier - 1.0).abs() > 1e-9 {
                (&normed * layer.ssm_in_multiplier)?
            } else {
                normed
            };

            let ssm_state = &request_state.ssm_states[layer_idx].tensor;
            let conv_state = &request_state.conv_states[layer_idx].tensor;

            let is_prefill = seq_len > 1;
            let (ssm_output, new_ssm, new_conv) = if is_prefill {
                layer.mamba.forward_prefill(&ssm_input, ssm_state)?
            } else {
                layer
                    .mamba
                    .forward_decode(&ssm_input, ssm_state, conv_state)?
            };

            request_state.ssm_states[layer_idx].tensor = new_ssm;
            request_state.conv_states[layer_idx].tensor = new_conv;

            // Combine: attn*out_mul + ssm*out_mul + residual
            let attn_scaled = if (layer.attention_out_multiplier - 1.0).abs() > 1e-9 {
                (attn_output * layer.attention_out_multiplier)?
            } else {
                attn_output
            };
            let ssm_scaled = if (layer.ssm_out_multiplier - 1.0).abs() > 1e-9 {
                (ssm_output * layer.ssm_out_multiplier)?
            } else {
                ssm_output
            };

            hidden = ((&attn_scaled + &ssm_scaled)? + &residual)?;

            // Feed-forward
            let ff_residual = hidden.clone();
            let ff_normed = layer.pre_ff_layernorm.forward(&hidden)?;
            let ff_output = layer.feed_forward.forward(&ff_normed)?;
            hidden = (&ff_residual + ff_output)?;
        }

        let hidden = self.final_layernorm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        // Apply lm_head_multiplier
        let logits = if (self.lm_head_multiplier - 1.0).abs() > 1e-9 {
            (logits * self.lm_head_multiplier)?
        } else {
            logits
        };

        Ok(logits)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl crate::engine::ModelForward for FalconH1ForCausalLM {
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

    fn test_falcon_h1_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("mamba_d_state".to_string(), serde_json::json!(8));
        extra.insert("mamba_d_conv".to_string(), serde_json::json!(4));
        extra.insert("mamba_expand".to_string(), serde_json::json!(2));
        extra.insert("mamba_n_heads".to_string(), serde_json::json!(4));
        extra.insert("mamba_d_head".to_string(), serde_json::json!(32));
        extra.insert("mamba_n_groups".to_string(), serde_json::json!(1));
        extra.insert("embedding_multiplier".to_string(), serde_json::json!(1.0));
        extra.insert("ssm_in_multiplier".to_string(), serde_json::json!(1.0));
        extra.insert("ssm_out_multiplier".to_string(), serde_json::json!(0.5));
        extra.insert(
            "attention_in_multiplier".to_string(),
            serde_json::json!(1.0),
        );
        extra.insert(
            "attention_out_multiplier".to_string(),
            serde_json::json!(0.5),
        );
        extra.insert("lm_head_multiplier".to_string(), serde_json::json!(1.0));

        ModelConfig {
            architectures: vec!["FalconH1ForCausalLM".to_string()],
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

    fn create_falcon_h1_cache(cfg: &ModelConfig) -> (KVCacheManager, BlockTable) {
        // Every layer has attention in FalconH1 (parallel hybrid)
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
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
    fn test_falcon_h1_construction() {
        let cfg = test_falcon_h1_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = FalconH1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "FalconH1ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_falcon_h1_config_extraction() {
        let cfg = test_falcon_h1_config();
        let falcon_cfg = FalconH1Config::from_model_config(&cfg);

        assert_eq!(falcon_cfg.mamba_d_state, 8);
        assert_eq!(falcon_cfg.mamba_d_conv, 4);
        assert_eq!(falcon_cfg.d_inner, 128);
        assert!((falcon_cfg.ssm_out_multiplier - 0.5).abs() < 1e-9);
        assert!((falcon_cfg.attention_out_multiplier - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_falcon_h1_forward_shape() {
        let cfg = test_falcon_h1_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FalconH1ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_falcon_h1_cache(&cfg);

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

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_falcon_h1_prefill_then_decode() {
        let cfg = test_falcon_h1_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FalconH1ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_falcon_h1_cache(&cfg);

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
    fn test_falcon_h1_model_forward_trait() {
        let cfg = test_falcon_h1_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FalconH1ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_falcon_h1_cache(&cfg);

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
    fn test_falcon_h1_device() {
        let cfg = test_falcon_h1_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FalconH1ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_falcon_h1_concurrent_requests() {
        let cfg = test_falcon_h1_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = FalconH1ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_falcon_h1_cache(&cfg);

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
        assert!(state_mgr.has_state(1));
        assert!(state_mgr.has_state(2));
        assert_eq!(state_mgr.num_active_requests(), 2);
    }
}
