//! Zamba2 (hybrid SSM + shared attention) model architecture.
//!
//! Zamba2 combines Mamba2 SSM layers with shared transformer attention layers.
//! The key innovation is weight-shared attention blocks that are reused across
//! multiple "hybrid" positions, with per-layer LoRA adapters for specialization.
//!
//! Architecture:
//! ```text
//! Embedding -> [Layer x N] -> RMSNorm -> LM Head
//!
//! Layer (mamba-only):
//!   RMSNorm -> Mamba2 Mixer -> + residual
//!
//! Layer (hybrid):
//!   SharedAttention(concat(hidden, original)) -> Linear -> + Mamba2(hidden)
//! ```
//!
//! Config keys from extra:
//! - `layers_block_type`: list of "mamba" or "hybrid" per layer
//! - `mamba_d_state`: SSM state size (default 64)
//! - `mamba_d_conv`: SSM conv kernel size (default 4)
//! - `mamba_expand`: SSM expansion factor (default 2)
//! - `n_mamba_heads`: number of Mamba heads
//! - `mamba_ngroups`: number of groups for Mamba
//! - `attention_hidden_size`: hidden size for attention blocks
//! - `attention_head_dim`: head dimension for attention
//! - `num_mem_blocks`: number of shared transformer blocks

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
use crate::ssm::{causal_conv1d_decode, causal_conv1d_prefill};

// ─── Config Extraction ──────────────────────────────────────────────────────

struct Zamba2Config {
    mamba_d_state: usize,
    mamba_d_conv: usize,
    #[allow(dead_code)]
    mamba_expand: usize,
    n_mamba_heads: usize,
    mamba_head_dim: usize,
    mamba_ngroups: usize,
    d_inner: usize,
    attention_hidden_size: usize,
    attention_head_dim: usize,
    num_attention_heads: usize,
    layer_types: Vec<String>,
}

impl Zamba2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let mamba_expand = cfg
            .extra
            .get("mamba_expand")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let d_inner = cfg.hidden_size * mamba_expand;

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

        let n_mamba_heads = cfg
            .extra
            .get("n_mamba_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let mamba_head_dim = if n_mamba_heads > 0 {
            d_inner / n_mamba_heads
        } else {
            d_inner
        };

        let mamba_ngroups = cfg
            .extra
            .get("mamba_ngroups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let attention_hidden_size = cfg
            .extra
            .get("attention_hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let attention_head_dim = cfg
            .extra
            .get("attention_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let num_attention_heads = cfg
            .extra
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_attention_heads);

        let layer_types = cfg
            .extra
            .get("layers_block_type")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Self {
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
            n_mamba_heads,
            mamba_head_dim,
            mamba_ngroups,
            d_inner,
            attention_hidden_size,
            attention_head_dim,
            num_attention_heads,
            layer_types,
        }
    }

    fn is_hybrid_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "hybrid")
            .unwrap_or(false)
    }
}
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}

// ─── Mamba2 Block ───────────────────────────────────────────────────────────

struct Zamba2Mamba2Block {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    out_proj: Linear,
    a: Tensor,
    d_param: Tensor,
    dt_bias: Tensor,
    input_layernorm: RmsNorm,
    d_inner: usize,
    d_state: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
}

impl Zamba2Mamba2Block {
    fn new(cfg: &ModelConfig, z_cfg: &Zamba2Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_inner = z_cfg.d_inner;
        let d_state = z_cfg.mamba_d_state;
        let d_conv = z_cfg.mamba_d_conv;
        let num_heads = z_cfg.n_mamba_heads;
        let head_dim = z_cfg.mamba_head_dim;
        let n_groups = z_cfg.mamba_ngroups;

        let in_proj_size = d_inner + 2 * n_groups * d_state + num_heads;
        let in_proj = linear_no_bias(hidden_size, in_proj_size, vb.pp("mixer").pp("in_proj"))?;

        let conv1d_weight = vb
            .pp("mixer")
            .pp("conv1d")
            .get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("mixer").pp("conv1d").get(d_inner, "bias")?;

        let out_proj = linear_no_bias(d_inner, hidden_size, vb.pp("mixer").pp("out_proj"))?;

        let a = vb.pp("mixer").get(num_heads, "A")?;
        let d_param = vb.pp("mixer").get(num_heads, "D")?;
        let dt_bias = vb.pp("mixer").get(num_heads, "dt_bias")?;

        let input_layernorm = rms_norm(hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            out_proj,
            a,
            d_param,
            dt_bias,
            input_layernorm,
            d_inner,
            d_state,
            num_heads,
            head_dim,
            n_groups,
        })
    }

    /// Forward pass with optional transformer injection (for hybrid layers).
    fn forward_prefill(
        &self,
        hidden_states: &Tensor,
        ssm_state: &Tensor,
        transformer_hidden: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;

        let residual = hidden_states;

        // Inject transformer output if present
        let xs = if let Some(th) = transformer_hidden {
            (hidden_states + th)?
        } else {
            hidden_states.clone()
        };

        let normed = self.input_layernorm.forward(&xs)?;
        let proj = self.in_proj.forward(&normed)?;

        let x = proj.narrow(2, 0, self.d_inner)?;
        let b_offset = self.d_inner;
        let b_proj = proj.narrow(2, b_offset, self.n_groups * self.d_state)?;
        let c_offset = b_offset + self.n_groups * self.d_state;
        let c_proj = proj.narrow(2, c_offset, self.n_groups * self.d_state)?;
        let dt_offset = c_offset + self.n_groups * self.d_state;
        let dt = proj.narrow(2, dt_offset, self.num_heads)?;

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
        let output = (residual + output)?;

        Ok((output, new_ssm_state, new_conv_state))
    }

    fn forward_decode(
        &self,
        hidden_states: &Tensor,
        ssm_state: &Tensor,
        conv_state: &Tensor,
        transformer_hidden: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let residual = hidden_states;

        let xs = if let Some(th) = transformer_hidden {
            (hidden_states + th)?
        } else {
            hidden_states.clone()
        };

        let normed = self.input_layernorm.forward(&xs)?;
        let proj = self.in_proj.forward(&normed)?;
        let proj = proj.squeeze(1)?;

        let x = proj.narrow(1, 0, self.d_inner)?;
        let b_offset = self.d_inner;
        let b_proj = proj.narrow(1, b_offset, self.n_groups * self.d_state)?;
        let c_offset = b_offset + self.n_groups * self.d_state;
        let c_proj = proj.narrow(1, c_offset, self.n_groups * self.d_state)?;
        let dt_offset = c_offset + self.n_groups * self.d_state;
        let dt = proj.narrow(1, dt_offset, self.num_heads)?;

        let (x_conv, new_conv_state) =
            causal_conv1d_decode(&x, &self.conv1d_weight, &self.conv1d_bias, conv_state)?;

        let x_ssm = candle_nn::ops::silu(&x_conv)?;

        let dt = dt.broadcast_add(&self.dt_bias.unsqueeze(0)?)?;
        let delta = softplus(&dt)?;

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
        let output = (residual + output)?;

        Ok((output, new_ssm_state, new_conv_state))
    }
}

// ─── Attention Block ────────────────────────────────────────────────────────

/// Simplified shared attention for Zamba2 hybrid layers.
/// In the reference, attention weights are shared across hybrid layers
/// and adapted via per-layer LoRA. We implement the core attention without
/// weight-sharing or LoRA for simplicity.
struct Zamba2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
    /// Input norm operates on concatenated [hidden, original] = 2*hidden_size
    input_layernorm: RmsNorm,
    pre_ff_layernorm: RmsNorm,
    /// Linear projection for transformer output before injection into Mamba
    linear: Linear,
    /// Simple MLP for post-attention
    ff_gate_proj: Linear,
    ff_up_proj: Linear,
    ff_down_proj: Linear,
}

impl Zamba2Attention {
    fn new(cfg: &ModelConfig, z_cfg: &Zamba2Config, vb: VarBuilder) -> Result<Self> {
        let attn_hidden = z_cfg.attention_hidden_size;
        let head_dim = z_cfg.attention_head_dim;
        let num_heads = z_cfg.num_attention_heads;

        let vb_attn = vb.pp("shared_transformer");
        let q_proj = linear_no_bias(attn_hidden, num_heads * head_dim, vb_attn.pp("q_proj"))?;
        let k_proj = linear_no_bias(attn_hidden, num_heads * head_dim, vb_attn.pp("k_proj"))?;
        let v_proj = linear_no_bias(attn_hidden, num_heads * head_dim, vb_attn.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb_attn.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        // Input norm on concatenated (2 * hidden_size)
        let input_layernorm = rms_norm(
            2 * cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_attn.pp("input_layernorm"),
        )?;

        let pre_ff_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_attn.pp("pre_ff_layernorm"),
        )?;

        // Linear projection after attention, before Mamba injection
        let linear = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("linear"))?;

        // Feed-forward MLP (GELU-based in reference; we use SiLU for simplicity)
        let ff_gate_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb_attn.pp("feed_forward").pp("gate_proj"),
        )?;
        let ff_up_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb_attn.pp("feed_forward").pp("up_proj"),
        )?;
        let ff_down_proj = linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb_attn.pp("feed_forward").pp("down_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            head_dim,
            input_layernorm,
            pre_ff_layernorm,
            linear,
            ff_gate_proj,
            ff_up_proj,
            ff_down_proj,
        })
    }

    /// Forward pass: takes hidden_states and original_hidden_states,
    /// concatenates them, runs attention + MLP, returns projected output
    /// to be injected into the Mamba pathway.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden_states: &Tensor,
        original_hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = hidden_states.dims3()?;

        // Concatenate hidden + original along last dim
        let concat = Tensor::cat(&[hidden_states, original_hidden_states], 2)?;
        let normed = self.input_layernorm.forward(&concat)?;

        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
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
            self.num_heads, // Zamba2 uses same number of Q and KV heads
            self.head_dim,
        )?;

        let attn_output = self.o_proj.forward(&attn_output)?;

        // Pre-FF norm + MLP
        let ff_input = self.pre_ff_layernorm.forward(&attn_output)?;
        let gate = candle_nn::ops::silu(&self.ff_gate_proj.forward(&ff_input)?)?;
        let up = self.ff_up_proj.forward(&ff_input)?;
        let ff_output = self.ff_down_proj.forward(&(gate * up)?)?;

        // Project for Mamba injection
        let projected = self.linear.forward(&ff_output)?;

        Ok(projected)
    }
}

// ─── Layer Types ────────────────────────────────────────────────────────────

enum Zamba2LayerType {
    /// Mamba-only layer (no attention)
    MambaOnly,
    /// Hybrid layer (shared attention + Mamba)
    Hybrid(Box<Zamba2Attention>),
}

struct Zamba2Layer {
    layer_type: Zamba2LayerType,
    mamba: Zamba2Mamba2Block,
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Zamba2ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Zamba2Layer>,
    final_layernorm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    state_mgr: Mutex<SSMStateManager>,
    /// Number of attention-bearing layers (for KV cache)
    num_attn_layers: usize,
    /// Mapping from model layer index to KV cache layer index
    attn_layer_cache_idx: Vec<Option<usize>>,
}

impl Zamba2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let z_cfg = Zamba2Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        let mut num_attn_layers = 0;
        let mut attn_layer_cache_idx = Vec::with_capacity(cfg.num_hidden_layers);

        for i in 0..cfg.num_hidden_layers {
            let is_hybrid = z_cfg.is_hybrid_layer(i);
            let layer_type = if is_hybrid {
                let attn = Zamba2Attention::new(cfg, &z_cfg, vb_layers.pp(i))?;
                attn_layer_cache_idx.push(Some(num_attn_layers));
                num_attn_layers += 1;
                Zamba2LayerType::Hybrid(Box::new(attn))
            } else {
                attn_layer_cache_idx.push(None);
                Zamba2LayerType::MambaOnly
            };

            let mamba = Zamba2Mamba2Block::new(cfg, &z_cfg, vb_layers.pp(i))?;

            layers.push(Zamba2Layer { layer_type, mamba });
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
            z_cfg.d_inner,
            z_cfg.mamba_d_state,
            z_cfg.mamba_d_conv,
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
        // Store original embeddings for hybrid layers
        let original_hidden = hidden.clone();

        let is_prefill = seq_len > 1;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let ssm_state = &request_state.ssm_states[layer_idx].tensor;
            let conv_state = &request_state.conv_states[layer_idx].tensor;

            // For hybrid layers, run attention first to get transformer injection
            let transformer_hidden = match &layer.layer_type {
                Zamba2LayerType::MambaOnly => None,
                Zamba2LayerType::Hybrid(attn) => {
                    let cache_idx = self.attn_layer_cache_idx[layer_idx]
                        .expect("hybrid layer should have cache index");
                    let th = attn.forward(
                        &hidden,
                        &original_hidden,
                        attention_mask.as_ref(),
                        seqlen_offset,
                        kv_cache_mgr.engine_mut(cache_idx),
                        block_table,
                        slot_mapping,
                    )?;
                    Some(th)
                }
            };

            let (new_hidden, new_ssm, new_conv) = if is_prefill {
                layer
                    .mamba
                    .forward_prefill(&hidden, ssm_state, transformer_hidden.as_ref())?
            } else {
                layer.mamba.forward_decode(
                    &hidden,
                    ssm_state,
                    conv_state,
                    transformer_hidden.as_ref(),
                )?
            };

            hidden = new_hidden;
            request_state.ssm_states[layer_idx].tensor = new_ssm;
            request_state.conv_states[layer_idx].tensor = new_conv;
        }

        let hidden = self.final_layernorm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok(logits)
    }

    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }
}

impl crate::engine::ModelForward for Zamba2ForCausalLM {
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

    fn test_zamba2_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // 4 layers: mamba, hybrid, mamba, hybrid
        extra.insert(
            "layers_block_type".to_string(),
            serde_json::json!(["mamba", "hybrid", "mamba", "hybrid"]),
        );
        extra.insert("mamba_d_state".to_string(), serde_json::json!(8));
        extra.insert("mamba_d_conv".to_string(), serde_json::json!(4));
        extra.insert("mamba_expand".to_string(), serde_json::json!(2));
        extra.insert("n_mamba_heads".to_string(), serde_json::json!(4));
        extra.insert("mamba_ngroups".to_string(), serde_json::json!(1));
        extra.insert("attention_hidden_size".to_string(), serde_json::json!(128));
        extra.insert("attention_head_dim".to_string(), serde_json::json!(16));
        extra.insert("num_attention_heads".to_string(), serde_json::json!(4));

        ModelConfig {
            architectures: vec!["Zamba2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
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

    fn create_zamba2_cache(
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
    fn test_zamba2_construction() {
        let cfg = test_zamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Zamba2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Zamba2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        // Layers 1 and 3 are hybrid
        assert_eq!(model.num_attn_layers, 2);
    }

    #[test]
    fn test_zamba2_config_extraction() {
        let cfg = test_zamba2_config();
        let z_cfg = Zamba2Config::from_model_config(&cfg);

        assert_eq!(z_cfg.mamba_d_state, 8);
        assert_eq!(z_cfg.mamba_d_conv, 4);
        assert_eq!(z_cfg.d_inner, 128);
        assert_eq!(z_cfg.n_mamba_heads, 4);
        assert_eq!(z_cfg.attention_hidden_size, 128);

        assert!(!z_cfg.is_hybrid_layer(0));
        assert!(z_cfg.is_hybrid_layer(1));
        assert!(!z_cfg.is_hybrid_layer(2));
        assert!(z_cfg.is_hybrid_layer(3));
    }

    #[test]
    fn test_zamba2_forward_shape() {
        let cfg = test_zamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Zamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_zamba2_cache(&cfg, model.num_attn_layers);

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
    fn test_zamba2_prefill_then_decode() {
        let cfg = test_zamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Zamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_zamba2_cache(&cfg, model.num_attn_layers);

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
    fn test_zamba2_model_forward_trait() {
        let cfg = test_zamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Zamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_zamba2_cache(&cfg, model.num_attn_layers);

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
    fn test_zamba2_device() {
        let cfg = test_zamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Zamba2ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_zamba2_concurrent_requests() {
        let cfg = test_zamba2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Zamba2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_zamba2_cache(&cfg, model.num_attn_layers);

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
