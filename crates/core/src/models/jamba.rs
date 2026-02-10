//! Jamba (hybrid SSM + attention) model architecture.
//!
//! Jamba (AI21 Labs) alternates between Mamba-style SSM blocks and standard
//! transformer attention blocks. Some layers may also use MoE for the MLP.
//!
//! Architecture:
//! ```text
//! Embedding -> [JambaLayer x N] -> RMSNorm -> LM Head
//!
//! JambaLayer (Mamba variant):
//!   RMSNorm -> MambaMixer -> RMSNorm -> MLP/MoE
//!
//! JambaLayer (Attention variant, every attn_layer_period-th layer):
//!   RMSNorm -> SelfAttention -> RMSNorm -> MLP/MoE
//! ```
//!
//! Config keys from extra:
//! - `attn_layer_period`: how often an attention layer appears (default 8)
//! - `attn_layer_offset`: offset for attention layer indexing (default 4)
//! - `num_experts`: number of MoE experts per layer (default 1 = no MoE)
//! - `num_experts_per_tok`: top-k experts (default 2)
//! - `mamba_d_state`: SSM state size (default 16)
//! - `mamba_d_conv`: SSM conv kernel size (default 4)
//! - `mamba_expand`: SSM expansion factor (default 2)
//! - `mamba_dt_rank`: SSM dt rank (default ceil(hidden/16))

use std::sync::Mutex;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::ssm::selective_scan;
use crate::ssm::state::SSMStateManager;

// ─── Jamba Config Extraction ────────────────────────────────────────────────

struct JambaConfig {
    attn_layer_period: usize,
    attn_layer_offset: usize,
    mamba_d_state: usize,
    mamba_d_conv: usize,
    #[allow(dead_code)]
    mamba_expand: usize,
    mamba_dt_rank: usize,
    d_inner: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl JambaConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let attn_layer_period = cfg
            .extra
            .get("attn_layer_period")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(8);

        let attn_layer_offset = cfg
            .extra
            .get("attn_layer_offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

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

        let mamba_expand = cfg
            .extra
            .get("mamba_expand")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let mamba_dt_rank = cfg
            .extra
            .get("mamba_dt_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size.div_ceil(16));

        let d_inner = cfg.hidden_size * mamba_expand;

        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        Self {
            attn_layer_period,
            attn_layer_offset,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
            mamba_dt_rank,
            d_inner,
            num_experts,
            num_experts_per_tok,
        }
    }

    /// Determine whether a given layer index uses attention (true) or Mamba (false).
    fn is_attention_layer(&self, layer_idx: usize) -> bool {
        if self.attn_layer_period == 0 {
            return false;
        }
        layer_idx >= self.attn_layer_offset
            && (layer_idx - self.attn_layer_offset).is_multiple_of(self.attn_layer_period)
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

struct JambaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl JambaMlp {
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

// ─── Simple MoE (top-k routing) ────────────────────────────────────────────

struct JambaMoE {
    router: Linear,
    experts: Vec<JambaMlp>,
    num_experts: usize,
    top_k: usize,
}

impl JambaMoE {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = linear_no_bias(hidden_size, num_experts, vb.pp("router"))?;
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(JambaMlp::new(
                hidden_size,
                intermediate_size,
                vb_experts.pp(i),
            )?);
        }
        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;
        let flat = xs.reshape((batch * seq_len, hidden))?;

        // Route: [tokens, num_experts]
        let router_logits = self.router.forward(&flat)?;

        // Softmax for routing weights
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Simple top-k routing via argmax for each token
        // For CPU: iterate tokens and pick top_k experts
        let routing_data: Vec<f32> = routing_weights
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let num_tokens = batch * seq_len;

        let mut output_data = vec![0.0f32; num_tokens * hidden];
        let flat_data: Vec<f32> = flat.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        for token_idx in 0..num_tokens {
            let weights =
                &routing_data[token_idx * self.num_experts..(token_idx + 1) * self.num_experts];

            // Find top_k experts
            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Renormalize top-k weights
            let top_sum: f32 = indexed[..self.top_k].iter().map(|(_, w)| w).sum();

            let token_input = Tensor::from_vec(
                flat_data[token_idx * hidden..(token_idx + 1) * hidden].to_vec(),
                (1, hidden),
                xs.device(),
            )?;

            for &(expert_idx, weight) in indexed[..self.top_k].iter() {
                let norm_weight = if top_sum > 0.0 {
                    weight / top_sum
                } else {
                    1.0 / self.top_k as f32
                };
                let expert_out = self.experts[expert_idx].forward(&token_input.unsqueeze(0)?)?;
                let expert_data: Vec<f32> =
                    expert_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for j in 0..hidden {
                    output_data[token_idx * hidden + j] += norm_weight * expert_data[j];
                }
            }
        }

        Tensor::from_vec(output_data, (batch, seq_len, hidden), xs.device())?.to_dtype(xs.dtype())
    }
}

// ─── Feed-forward (MLP or MoE) ─────────────────────────────────────────────

enum JambaFeedForward {
    Mlp(JambaMlp),
    MoE(JambaMoE),
}

impl JambaFeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            JambaFeedForward::Mlp(mlp) => mlp.forward(xs),
            JambaFeedForward::MoE(moe) => moe.forward(xs),
        }
    }
}

// ─── Mamba Mixer Block ──────────────────────────────────────────────────────

struct JambaMambaBlock {
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

impl JambaMambaBlock {
    fn new(cfg: &ModelConfig, jamba_cfg: &JambaConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_inner = jamba_cfg.d_inner;
        let d_state = jamba_cfg.mamba_d_state;
        let d_conv = jamba_cfg.mamba_d_conv;
        let dt_rank = jamba_cfg.mamba_dt_rank;

        let in_proj = linear_no_bias(hidden_size, 2 * d_inner, vb.pp("in_proj"))?;
        let conv1d_weight = vb.pp("conv1d").get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(d_inner, "bias")?;
        let x_proj = linear_no_bias(d_inner, dt_rank + 2 * d_state, vb.pp("x_proj"))?;
        let dt_proj = Linear::new(
            vb.pp("dt_proj").get((d_inner, dt_rank), "weight")?,
            Some(vb.pp("dt_proj").get(d_inner, "bias")?),
        );
        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
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

    fn forward_prefill(&self, xs: &Tensor, ssm_state: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = xs.dims3()?;

        let proj = self.in_proj.forward(xs)?;
        let x = proj.narrow(2, 0, self.d_inner)?;
        let z = proj.narrow(2, self.d_inner, self.d_inner)?;

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
            Tensor::cat(&[&pad, &x.transpose(1, 2)?], 2)?
        };

        let x_conv = x_conv.transpose(1, 2)?;
        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        let x_dbc = self.x_proj.forward(&x_ssm)?;
        let dt = x_dbc.narrow(2, 0, self.dt_rank)?;
        let b_proj = x_dbc.narrow(2, self.dt_rank, self.d_state)?;
        let c_proj = x_dbc.narrow(2, self.dt_rank + self.d_state, self.d_state)?;

        let delta = self.dt_proj.forward(&dt)?;
        let delta = softplus(&delta)?;
        let a = self.a_log.exp()?.neg()?;

        let (ssm_out, new_ssm_state) = selective_scan(
            &x_ssm,
            &delta,
            &a,
            &b_proj,
            &c_proj,
            &self.d,
            Some(ssm_state),
        )?;

        let z_gate = candle_nn::ops::silu(&z)?;
        let gated = (&ssm_out * &z_gate)?;
        let output = self.out_proj.forward(&gated)?;

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
        let z = proj.narrow(1, self.d_inner, self.d_inner)?;

        let (x_conv, new_conv_state) =
            causal_conv1d_decode(&x, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        let x_dbc = self.x_proj.forward(&x_ssm.unsqueeze(1)?)?.squeeze(1)?;
        let dt = x_dbc.narrow(1, 0, self.dt_rank)?;
        let b_proj = x_dbc.narrow(1, self.dt_rank, self.d_state)?;
        let c_proj = x_dbc.narrow(1, self.dt_rank + self.d_state, self.d_state)?;

        let delta = self.dt_proj.forward(&dt.unsqueeze(1)?)?.squeeze(1)?;
        let delta = softplus(&delta)?;
        let a = self.a_log.exp()?.neg()?;

        let x_ssm_exp = x_ssm.unsqueeze(1)?;
        let delta_exp = delta.unsqueeze(1)?;
        let b_exp = b_proj.unsqueeze(1)?;
        let c_exp = c_proj.unsqueeze(1)?;

        let (ssm_out, new_ssm_state) = selective_scan(
            &x_ssm_exp,
            &delta_exp,
            &a,
            &b_exp,
            &c_exp,
            &self.d,
            Some(ssm_state),
        )?;

        let ssm_out = ssm_out.squeeze(1)?;
        let z_gate = candle_nn::ops::silu(&z)?;
        let gated = (&ssm_out * &z_gate)?;
        let output = self.out_proj.forward(&gated.unsqueeze(1)?)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

// ─── Attention Block ────────────────────────────────────────────────────────

struct JambaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl JambaAttention {
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

    #[allow(dead_code)]
    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

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

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

enum JambaLayerType {
    Mamba(JambaMambaBlock),
    Attention(JambaAttention),
}

struct JambaDecoderLayer {
    layer_type: JambaLayerType,
    feed_forward: JambaFeedForward,
    input_layernorm: RmsNorm,
    pre_ff_layernorm: RmsNorm,
}

impl JambaDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        jamba_cfg: &JambaConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_attn = jamba_cfg.is_attention_layer(layer_idx);

        let layer_type = if is_attn {
            JambaLayerType::Attention(JambaAttention::new(cfg, vb.pp("self_attn"))?)
        } else {
            JambaLayerType::Mamba(JambaMambaBlock::new(cfg, jamba_cfg, vb.pp("mamba"))?)
        };

        let feed_forward = if jamba_cfg.num_experts > 1 {
            JambaFeedForward::MoE(JambaMoE::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                jamba_cfg.num_experts,
                jamba_cfg.num_experts_per_tok,
                vb.pp("feed_forward"),
            )?)
        } else {
            JambaFeedForward::Mlp(JambaMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("feed_forward"),
            )?)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_ff_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_ff_layernorm"))?;

        Ok(Self {
            layer_type,
            feed_forward,
            input_layernorm,
            pre_ff_layernorm,
        })
    }

    fn is_attention(&self) -> bool {
        matches!(self.layer_type, JambaLayerType::Attention(_))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct JambaForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<JambaDecoderLayer>,
    final_layernorm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    /// SSM state for Mamba layers (attention layers use KV cache)
    state_mgr: Mutex<SSMStateManager>,
    /// Number of attention layers (for determining KV cache layer count)
    num_attn_layers: usize,
    /// Mapping from model layer index to KV cache layer index (for attention layers only)
    attn_layer_cache_idx: Vec<Option<usize>>,
}

impl JambaForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let jamba_cfg = JambaConfig::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        let mut num_attn_layers = 0;
        let mut attn_layer_cache_idx = Vec::with_capacity(cfg.num_hidden_layers);

        for i in 0..cfg.num_hidden_layers {
            let layer = JambaDecoderLayer::new(cfg, &jamba_cfg, i, vb_layers.pp(i))?;
            if layer.is_attention() {
                attn_layer_cache_idx.push(Some(num_attn_layers));
                num_attn_layers += 1;
            } else {
                attn_layer_cache_idx.push(None);
            }
            layers.push(layer);
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

        // SSM state manager for Mamba layers
        let state_mgr = SSMStateManager::new(
            cfg.num_hidden_layers, // allocate for all layers; attention layers will be unused
            jamba_cfg.d_inner,
            jamba_cfg.mamba_d_state,
            jamba_cfg.mamba_d_conv,
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
            num_attn_layers,
            attn_layer_cache_idx,
        })
    }

    /// Forward pass for the hybrid model.
    ///
    /// Attention layers use the KV cache through KVCacheManager.
    /// Mamba layers use internal SSM state through SSMStateManager.
    ///
    /// Each request is identified by `request_id` for independent SSM state tracking.
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
            // Pre-norm
            let residual = &hidden;
            let normed = layer.input_layernorm.forward(&hidden)?;

            // Process through Mamba or Attention
            let mixer_output = match &layer.layer_type {
                JambaLayerType::Mamba(mamba) => {
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
                JambaLayerType::Attention(attn) => {
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

            // Residual + pre-FF norm + feed-forward + residual
            let hidden_after_mixer = (residual + mixer_output)?;
            let ff_normed = layer.pre_ff_layernorm.forward(&hidden_after_mixer)?;
            let ff_output = layer.feed_forward.forward(&ff_normed)?;
            hidden = (&hidden_after_mixer + ff_output)?;
        }

        let hidden = self.final_layernorm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok(logits)
    }

    /// Get the number of attention layers (needed for cache configuration).
    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }
}

impl crate::engine::ModelForward for JambaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Single-sequence path uses request_id=0 by convention.
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
        // Each sequence is processed individually with its own per-request SSM state.
        // The SSMStateManager tracks independent state for each request_id,
        // preventing cross-request state corruption during concurrent inference.
        // Attention layers still use the shared KV cache via block_table per-request.
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

    fn test_jamba_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Every 4th layer (offset 2) is attention, so layers 2 and 6 are attention
        // With 4 layers: layers 0,1,3 are Mamba, layer 2 is attention
        extra.insert("attn_layer_period".to_string(), serde_json::json!(4));
        extra.insert("attn_layer_offset".to_string(), serde_json::json!(2));
        extra.insert("mamba_d_state".to_string(), serde_json::json!(8));
        extra.insert("mamba_d_conv".to_string(), serde_json::json!(4));
        extra.insert("mamba_expand".to_string(), serde_json::json!(2));
        extra.insert("mamba_dt_rank".to_string(), serde_json::json!(4));
        extra.insert("num_experts".to_string(), serde_json::json!(1));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(1));

        ModelConfig {
            architectures: vec!["JambaForCausalLM".to_string()],
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

    fn create_jamba_cache(
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
    fn test_jamba_construction() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = JambaForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "JambaForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        // With period=4, offset=2, layers 0..4: layer 2 is attention
        assert_eq!(model.num_attn_layers, 1);
    }

    #[test]
    fn test_jamba_config_extraction() {
        let cfg = test_jamba_config();
        let jamba_cfg = JambaConfig::from_model_config(&cfg);

        assert_eq!(jamba_cfg.attn_layer_period, 4);
        assert_eq!(jamba_cfg.attn_layer_offset, 2);
        assert_eq!(jamba_cfg.mamba_d_state, 8);
        assert_eq!(jamba_cfg.mamba_d_conv, 4);
        assert_eq!(jamba_cfg.mamba_expand, 2);
        assert_eq!(jamba_cfg.d_inner, 128);

        // Layer classification
        assert!(!jamba_cfg.is_attention_layer(0));
        assert!(!jamba_cfg.is_attention_layer(1));
        assert!(jamba_cfg.is_attention_layer(2));
        assert!(!jamba_cfg.is_attention_layer(3));
    }

    #[test]
    fn test_jamba_forward_shape() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_jamba_cache(&cfg, model.num_attn_layers);

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
    fn test_jamba_single_token() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_jamba_cache(&cfg, model.num_attn_layers);

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
    fn test_jamba_model_forward_trait() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_jamba_cache(&cfg, model.num_attn_layers);

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
    fn test_jamba_device() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_jamba_layer_type_classification() {
        let cfg = test_jamba_config();
        let jamba_cfg = JambaConfig::from_model_config(&cfg);

        // attn_layer_period=4, offset=2 -> only layer 2 is attention in 4-layer model
        for i in 0..cfg.num_hidden_layers {
            let expected = i == 2;
            assert_eq!(
                jamba_cfg.is_attention_layer(i),
                expected,
                "layer {} should be {}, got {}",
                i,
                if expected { "attention" } else { "mamba" },
                if jamba_cfg.is_attention_layer(i) {
                    "attention"
                } else {
                    "mamba"
                }
            );
        }
    }

    #[test]
    fn test_jamba_concurrent_requests_independent_state() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_jamba_cache(&cfg, model.num_attn_layers);

        // Prefill request 1
        let prompt1 = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt1");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);
        let _ = model
            .forward_with_request_id(
                &prompt1,
                0,
                1,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 1");

        // Prefill request 2 (reuse same block table for simplicity, different request_id)
        let prompt2 = Tensor::ones((1, 3), DType::U32, &device).expect("prompt2");
        let _ = model
            .forward_with_request_id(
                &prompt2,
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
    fn test_jamba_concurrent_decode_no_cross_contamination() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_jamba_cache(&cfg, model.num_attn_layers);

        // Prefill two requests
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);
        let _ = model
            .forward_with_request_id(
                &prompt,
                0,
                10,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 10");
        let _ = model
            .forward_with_request_id(
                &prompt,
                0,
                20,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 20");

        block_table.advance(3);

        // Decode request 10 with extra step
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping_decode = block_table.slot_mapping(3, 1);
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");

        let logits_10_step1 = model
            .forward_with_request_id(
                &next,
                3,
                10,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping_decode,
            )
            .expect("decode 10 step 1");

        // Decode request 20 once -- should not be affected by request 10's decode
        let logits_20_step1 = model
            .forward_with_request_id(
                &next,
                3,
                20,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping_decode,
            )
            .expect("decode 20 step 1");

        // Both should produce valid outputs with correct shape
        assert_eq!(logits_10_step1.dims(), &[1, 1, cfg.vocab_size]);
        assert_eq!(logits_20_step1.dims(), &[1, 1, cfg.vocab_size]);

        // With same prefill, same decode token, and same seqlen_offset,
        // the SSM state component should be identical across requests.
        let data_10: Vec<f32> = logits_10_step1
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        let data_20: Vec<f32> = logits_20_step1
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        for (a, b) in data_10.iter().zip(data_20.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "same input should produce same output, got {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_jamba_forward_decode_batch_uses_per_request_state() {
        let cfg = test_jamba_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = JambaForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_jamba_cache(&cfg, model.num_attn_layers);

        // Prefill two requests
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);
        let _ = model
            .forward_with_request_id(
                &prompt,
                0,
                100,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 100");
        let _ = model
            .forward_with_request_id(
                &prompt,
                0,
                200,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("prefill req 200");
        block_table.advance(3);

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let block_ids = block_table.block_ids().to_vec();
        let decode_slot_mapping = block_table.slot_mapping(3, 1);

        // Batched decode
        let batch_input = Tensor::zeros((2, 1), DType::U32, &device).expect("batch input");
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 100,
                seqlen_offset: 3,
                block_ids: block_ids.clone(),
                slot_mapping: decode_slot_mapping.clone(),
            },
            DecodeSequenceMetadata {
                request_id: 200,
                seqlen_offset: 3,
                block_ids,
                slot_mapping: decode_slot_mapping,
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
}
