//! NemotronH hybrid model architecture (NemotronHForCausalLM / NemotronHPuzzleForCausalLM).
//!
//! A complex hybrid architecture with heterogeneous layer types per position,
//! determined by `hybrid_override_pattern` in the config. Each character in the
//! pattern maps to a layer type:
//!
//! - `M` = Mamba (SSM) layer
//! - `-` = MLP layer (dense, no attention)
//! - `*` = Attention layer (with KV cache)
//! - `E` = MoE layer (mixture of experts)
//!
//! The "Puzzle" variant has varying expert counts per layer, read from
//! `block_configs` in the config.
//!
//! Architecture:
//! ```text
//! Embedding -> [HeterogeneousLayer x N] -> RMSNorm -> LM Head
//!
//! Each layer: RMSNorm -> Mixer(type-dependent) -> residual
//! ```
//!
//! Reference: reference/vllm/vllm/model_executor/models/nemotron_h.py

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

// ─── NemotronH Config ───────────────────────────────────────────────────────

/// Parsed NemotronH-specific configuration from ModelConfig.extra.
struct NemotronHConfig {
    hybrid_override_pattern: Vec<char>,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    layer_norm_epsilon: f64,
    mlp_bias: bool,
    /// Per-MLP-layer intermediate sizes (may vary across MLP layers).
    intermediate_sizes: Vec<usize>,
    // Mamba SSM parameters
    ssm_state_size: usize,
    conv_kernel: usize,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    #[allow(dead_code)]
    n_groups: usize,
    // MoE parameters
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    n_shared_experts: usize,
    moe_shared_expert_intermediate_size: usize,
    routed_scaling_factor: f64,
    /// Per-layer expert counts from block_configs (for Puzzle variant).
    per_layer_n_routed_experts: Vec<Option<usize>>,
}

impl NemotronHConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let pattern_str = cfg
            .extra
            .get("hybrid_override_pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let hybrid_override_pattern: Vec<char> = pattern_str.chars().collect();

        let layer_norm_epsilon = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        let mlp_bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // intermediate_size can be a scalar or a list
        let intermediate_sizes = cfg
            .extra
            .get("intermediate_size")
            .and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
            })
            .unwrap_or_else(|| vec![cfg.intermediate_size]);

        let ssm_state_size = cfg
            .extra
            .get("ssm_state_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);

        let conv_kernel = cfg
            .extra
            .get("conv_kernel")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let mamba_num_heads = cfg
            .extra
            .get("mamba_num_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_attention_heads);

        let mamba_head_dim = cfg
            .extra
            .get("mamba_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let n_groups = cfg
            .extra
            .get("n_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let n_routed_experts = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(8);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let moe_intermediate_size = cfg
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let n_shared_experts = cfg
            .extra
            .get("n_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let moe_shared_expert_intermediate_size = cfg
            .extra
            .get("moe_shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(moe_intermediate_size);

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        // Parse per-layer expert counts from block_configs (Puzzle variant)
        let per_layer_n_routed_experts = Self::parse_block_configs(cfg, &hybrid_override_pattern);

        NemotronHConfig {
            hybrid_override_pattern,
            hidden_size: cfg.hidden_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            layer_norm_epsilon,
            mlp_bias,
            intermediate_sizes,
            ssm_state_size,
            conv_kernel,
            mamba_num_heads,
            mamba_head_dim,
            n_groups,
            n_routed_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            n_shared_experts,
            moe_shared_expert_intermediate_size,
            routed_scaling_factor,
            per_layer_n_routed_experts,
        }
    }

    /// Parse block_configs from the HF config for Puzzle variant.
    /// Returns a per-layer Option<usize> for the number of routed experts.
    fn parse_block_configs(cfg: &ModelConfig, pattern: &[char]) -> Vec<Option<usize>> {
        let block_configs = cfg.extra.get("block_configs").and_then(|v| v.as_array());
        if block_configs.is_none() {
            return vec![None; pattern.len()];
        }
        let block_configs = block_configs.expect("just checked");

        // Count MoE layers to map block_configs to layer indices
        let mut moe_idx = 0;
        let mut result = Vec::with_capacity(pattern.len());
        for &ch in pattern {
            if ch == 'E' {
                let n = block_configs.get(moe_idx).and_then(|b| {
                    b.get("n_routed_experts")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize)
                });
                result.push(n);
                moe_idx += 1;
            } else {
                result.push(None);
            }
        }
        result
    }

    /// Get the intermediate size for a specific MLP layer.
    /// `mlp_index` is the index among MLP layers only (not overall layer index).
    fn intermediate_size_for_mlp(&self, mlp_index: usize) -> usize {
        if self.intermediate_sizes.len() == 1 {
            self.intermediate_sizes[0]
        } else {
            self.intermediate_sizes
                .get(mlp_index)
                .copied()
                .unwrap_or(self.intermediate_sizes[0])
        }
    }

    /// Get the number of routed experts for a specific MoE layer.
    fn n_routed_experts_for_layer(&self, layer_idx: usize) -> usize {
        self.per_layer_n_routed_experts
            .get(layer_idx)
            .and_then(|v| *v)
            .unwrap_or(self.n_routed_experts)
    }
}

// ─── ReLU^2 Activation ──────────────────────────────────────────────────────

fn relu_squared(xs: &Tensor) -> Result<Tensor> {
    xs.relu()?.sqr()
}
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ones = Tensor::ones(x.dims(), x.dtype(), x.device())?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}

// ─── NemotronH MLP (dense, no gate, ReLU^2) ────────────────────────────────

struct NemotronHMlp {
    up_proj: Linear,
    down_proj: Linear,
}

impl NemotronHMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let up_proj = if bias {
            candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?
        } else {
            linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?
        };
        let down_proj = if bias {
            candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?
        } else {
            linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?
        };
        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.up_proj.forward(xs)?;
        let xs = relu_squared(&xs)?;
        self.down_proj.forward(&xs)
    }
}

// ─── NemotronH MoE ─────────────────────────────────────────────────────────

struct NemotronHMoE {
    gate: Linear,
    experts: Vec<NemotronHMlpExpert>,
    shared_experts: Option<NemotronHMlp>,
    num_experts: usize,
    top_k: usize,
    routed_scaling_factor: f64,
}

/// A single non-gated MoE expert (up_proj + relu^2 + down_proj).
struct NemotronHMlpExpert {
    up_proj: Linear,
    down_proj: Linear,
}

impl NemotronHMlpExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.up_proj.forward(xs)?;
        let xs = relu_squared(&xs)?;
        self.down_proj.forward(&xs)
    }
}

impl NemotronHMoE {
    fn new(nhc: &NemotronHConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let num_experts = nhc.n_routed_experts_for_layer(layer_idx);
        let top_k = nhc.num_experts_per_tok;

        let gate = linear_no_bias(nhc.hidden_size, num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(NemotronHMlpExpert::new(
                nhc.hidden_size,
                nhc.moe_intermediate_size,
                vb_experts.pp(i),
            )?);
        }

        let shared_experts = if nhc.n_shared_experts > 0 {
            let shared_intermediate =
                nhc.moe_shared_expert_intermediate_size * nhc.n_shared_experts;
            Some(NemotronHMlp::new(
                nhc.hidden_size,
                shared_intermediate,
                nhc.mlp_bias,
                vb.pp("shared_experts"),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            num_experts,
            top_k,
            routed_scaling_factor: nhc.routed_scaling_factor,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;
        let flat = xs.reshape((batch * seq_len, hidden))?;
        let num_tokens = batch * seq_len;

        // Router: [tokens, num_experts]
        let router_logits = self.gate.forward(&flat.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let routing_data: Vec<f32> = routing_weights.flatten_all()?.to_vec1()?;
        let flat_data: Vec<f32> = flat.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut output_data = vec![0.0f32; num_tokens * hidden];

        for token_idx in 0..num_tokens {
            let weights =
                &routing_data[token_idx * self.num_experts..(token_idx + 1) * self.num_experts];

            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
                let expert_out = self.experts[expert_idx].forward(&token_input)?;
                let expert_data: Vec<f32> =
                    expert_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for j in 0..hidden {
                    output_data[token_idx * hidden + j] += norm_weight * expert_data[j];
                }
            }
        }

        let mut routed_output = Tensor::from_vec(output_data, (num_tokens, hidden), xs.device())?
            .to_dtype(xs.dtype())?;

        // Apply scaling factor
        routed_output = (routed_output * self.routed_scaling_factor)?;

        // Add shared expert output if present
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(&flat)?;
            routed_output = (routed_output + shared_out)?;
        }

        routed_output.reshape((batch, seq_len, hidden))
    }
}

// ─── NemotronH Mamba Mixer ──────────────────────────────────────────────────

struct NemotronHMambaMixer {
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

impl NemotronHMambaMixer {
    fn new(nhc: &NemotronHConfig, vb: VarBuilder) -> Result<Self> {
        let d_inner = nhc.mamba_num_heads * nhc.mamba_head_dim;
        let d_state = nhc.ssm_state_size;
        let d_conv = nhc.conv_kernel;
        let dt_rank = nhc.hidden_size.div_ceil(16);

        let in_proj = linear_no_bias(nhc.hidden_size, 2 * d_inner, vb.pp("in_proj"))?;
        let conv1d_weight = vb.pp("conv1d").get((d_inner, 1, d_conv), "weight")?;
        let conv1d_bias = vb.pp("conv1d").get(d_inner, "bias")?;
        let x_proj = linear_no_bias(d_inner, dt_rank + 2 * d_state, vb.pp("x_proj"))?;
        let dt_proj = Linear::new(
            vb.pp("dt_proj").get((d_inner, dt_rank), "weight")?,
            Some(vb.pp("dt_proj").get(d_inner, "bias")?),
        );
        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, nhc.hidden_size, vb.pp("out_proj"))?;

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

// ─── NemotronH Attention ────────────────────────────────────────────────────

struct NemotronHAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl NemotronHAttention {
    fn new(nhc: &NemotronHConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = nhc.num_attention_heads;
        let num_kv_heads = nhc.num_key_value_heads;
        let head_dim = nhc.head_dim;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let total_qkv = q_size + 2 * kv_size;

        let qkv_proj = linear_no_bias(nhc.hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, nhc.hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            // NemotronH typically uses a large context window
            4096,
            10000.0,
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
}

// ─── Decoder Layer Types ────────────────────────────────────────────────────

/// The mixer variant for each layer position.
enum NemotronHLayerMixer {
    Mamba(NemotronHMambaMixer),
    Mlp(NemotronHMlp),
    Attention(NemotronHAttention),
    MoE(NemotronHMoE),
}

/// A single heterogeneous decoder layer: RMSNorm -> Mixer -> residual.
struct NemotronHDecoderLayer {
    mixer: NemotronHLayerMixer,
    norm: RmsNorm,
    layer_type_char: char,
}

impl NemotronHDecoderLayer {
    fn new(
        nhc: &NemotronHConfig,
        layer_idx: usize,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layer_type_char = nhc.hybrid_override_pattern[layer_idx];

        let mixer = match layer_type_char {
            'M' => NemotronHLayerMixer::Mamba(NemotronHMambaMixer::new(nhc, vb.pp("mixer"))?),
            '-' => {
                // Count how many MLP layers precede this one
                let mlp_index = nhc.hybrid_override_pattern[..layer_idx + 1]
                    .iter()
                    .filter(|&&c| c == '-')
                    .count()
                    - 1;
                let intermediate_size = nhc.intermediate_size_for_mlp(mlp_index);
                NemotronHLayerMixer::Mlp(NemotronHMlp::new(
                    nhc.hidden_size,
                    intermediate_size,
                    nhc.mlp_bias,
                    vb.pp("mixer"),
                )?)
            }
            '*' => NemotronHLayerMixer::Attention(NemotronHAttention::new(nhc, vb.pp("mixer"))?),
            'E' => NemotronHLayerMixer::MoE(NemotronHMoE::new(nhc, layer_idx, vb.pp("mixer"))?),
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown layer type character '{other}' in hybrid_override_pattern"
                )));
            }
        };

        let norm = rms_norm(cfg.hidden_size, nhc.layer_norm_epsilon, vb.pp("norm"))?;

        Ok(Self {
            mixer,
            norm,
            layer_type_char,
        })
    }

    fn is_attention(&self) -> bool {
        matches!(self.mixer, NemotronHLayerMixer::Attention(_))
    }

    #[cfg(test)]
    fn is_mamba(&self) -> bool {
        matches!(self.mixer, NemotronHLayerMixer::Mamba(_))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct NemotronHForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<NemotronHDecoderLayer>,
    norm_f: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    /// SSM state for Mamba layers
    state_mgr: Mutex<SSMStateManager>,
    /// Number of attention layers (for KV cache sizing)
    num_attn_layers: usize,
    /// Mapping from model layer index to KV cache layer index (attention layers only)
    attn_layer_cache_idx: Vec<Option<usize>>,
    /// Mamba SSM dimensions for state management
    #[allow(dead_code)]
    d_inner: usize,
    #[allow(dead_code)]
    d_state: usize,
    #[allow(dead_code)]
    d_conv: usize,
}

impl NemotronHForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let nhc = NemotronHConfig::from_model_config(cfg);

        let vb_model = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let num_layers = nhc.hybrid_override_pattern.len();
        let mut layers = Vec::with_capacity(num_layers);
        let mut num_attn_layers = 0;
        let mut attn_layer_cache_idx = Vec::with_capacity(num_layers);
        let vb_layers = vb_model.pp("layers");

        for i in 0..num_layers {
            let layer = NemotronHDecoderLayer::new(&nhc, i, cfg, vb_layers.pp(i))?;
            if layer.is_attention() {
                attn_layer_cache_idx.push(Some(num_attn_layers));
                num_attn_layers += 1;
            } else {
                attn_layer_cache_idx.push(None);
            }
            layers.push(layer);
        }

        let norm_f = rms_norm(
            cfg.hidden_size,
            nhc.layer_norm_epsilon,
            vb_model.pp("norm_f"),
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let d_inner = nhc.mamba_num_heads * nhc.mamba_head_dim;
        let d_state = nhc.ssm_state_size;
        let d_conv = nhc.conv_kernel;

        // SSM state manager for Mamba layers
        let state_mgr = SSMStateManager::new(
            num_layers,
            d_inner,
            d_state,
            d_conv,
            vb.dtype(),
            vb.device().clone(),
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm_f,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            state_mgr: Mutex::new(state_mgr),
            num_attn_layers,
            attn_layer_cache_idx,
            d_inner,
            d_state,
            d_conv,
        })
    }

    /// Get the number of attention layers (for cache configuration).
    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get each layer's type character.
    pub fn layer_types(&self) -> Vec<char> {
        self.layers.iter().map(|l| l.layer_type_char).collect()
    }

    /// Forward pass for the hybrid model.
    ///
    /// Attention layers use the KV cache through KVCacheManager.
    /// Mamba layers use internal SSM state through SSMStateManager.
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
        let mut residual: Option<Tensor> = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-norm with residual fusion (as per reference: norm(hidden, residual))
            let (normed, new_residual) = if let Some(ref res) = residual {
                let new_res = (&hidden + res)?;
                let normed = layer.norm.forward(&new_res)?;
                (normed, new_res)
            } else {
                let new_res = hidden.clone();
                let normed = layer.norm.forward(&hidden)?;
                (normed, new_res)
            };
            residual = Some(new_residual);

            hidden = match &layer.mixer {
                NemotronHLayerMixer::Mamba(mamba) => {
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
                NemotronHLayerMixer::Mlp(mlp) => mlp.forward(&normed)?,
                NemotronHLayerMixer::Attention(attn) => {
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
                NemotronHLayerMixer::MoE(moe) => moe.forward(&normed)?,
            };
        }

        // Final norm with residual
        let hidden = if let Some(ref res) = residual {
            self.norm_f.forward(&(hidden + res)?)?
        } else {
            self.norm_f.forward(&hidden)?
        };

        let logits = self.lm_head.forward(&hidden)?;
        Ok(logits)
    }
}

impl crate::engine::ModelForward for NemotronHForCausalLM {
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

/// Type alias for the Puzzle variant (same model, varying expert counts).
pub type NemotronHPuzzleForCausalLM = NemotronHForCausalLM;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_nemotron_h_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Pattern: Mamba, MLP, Attention, MoE
        extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("M-*E"),
        );
        extra.insert("layer_norm_epsilon".to_string(), serde_json::json!(1e-5));
        extra.insert("mlp_bias".to_string(), serde_json::json!(false));
        extra.insert("ssm_state_size".to_string(), serde_json::json!(8));
        extra.insert("conv_kernel".to_string(), serde_json::json!(4));
        extra.insert("mamba_num_heads".to_string(), serde_json::json!(4));
        extra.insert("mamba_head_dim".to_string(), serde_json::json!(16));
        extra.insert("n_groups".to_string(), serde_json::json!(1));
        extra.insert("n_routed_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("n_shared_experts".to_string(), serde_json::json!(0));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));

        ModelConfig {
            architectures: vec!["NemotronHForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "relu2".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_nemotron_h_cache(
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
    fn test_nemotron_h_config_parsing() {
        let cfg = test_nemotron_h_config();
        let nhc = NemotronHConfig::from_model_config(&cfg);

        assert_eq!(nhc.hybrid_override_pattern, vec!['M', '-', '*', 'E']);
        assert_eq!(nhc.hidden_size, 64);
        assert_eq!(nhc.ssm_state_size, 8);
        assert_eq!(nhc.conv_kernel, 4);
        assert_eq!(nhc.mamba_num_heads, 4);
        assert_eq!(nhc.mamba_head_dim, 16);
        assert_eq!(nhc.n_routed_experts, 4);
        assert_eq!(nhc.num_experts_per_tok, 2);
    }

    #[test]
    fn test_nemotron_h_construction() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = NemotronHForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "NemotronHForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.num_layers(), 4);
        assert_eq!(model.num_attention_layers(), 1);
        assert_eq!(model.layer_types(), vec!['M', '-', '*', 'E']);
    }

    #[test]
    fn test_nemotron_h_layer_type_classification() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        assert!(model.layers[0].is_mamba());
        assert!(!model.layers[0].is_attention());

        assert!(!model.layers[1].is_mamba());
        assert!(!model.layers[1].is_attention());

        assert!(!model.layers[2].is_mamba());
        assert!(model.layers[2].is_attention());

        assert!(!model.layers[3].is_mamba());
        assert!(!model.layers[3].is_attention());
    }

    #[test]
    fn test_nemotron_h_forward_shape() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) =
            create_nemotron_h_cache(&cfg, model.num_attention_layers());

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
    fn test_nemotron_h_prefill_then_decode() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) =
            create_nemotron_h_cache(&cfg, model.num_attention_layers());

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
    fn test_nemotron_h_model_forward_trait() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) =
            create_nemotron_h_cache(&cfg, model.num_attention_layers());

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
    fn test_nemotron_h_relu_squared() {
        let device = Device::Cpu;
        let xs = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], &device).expect("tensor");
        let result = relu_squared(&xs).expect("relu_squared");
        let vals: Vec<f32> = result.to_vec1().expect("to_vec");
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_nemotron_h_mlp_only_pattern() {
        // Test a model with only MLP layers (no KV cache needed for Mamba/Attention)
        let mut cfg = test_nemotron_h_config();
        cfg.extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("---"),
        );
        cfg.num_hidden_layers = 3;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        assert_eq!(model.num_layers(), 3);
        assert_eq!(model.num_attention_layers(), 0);
        assert_eq!(model.layer_types(), vec!['-', '-', '-']);

        // No attention layers means 0 KV cache layers
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 0,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache");
        let block_table = BlockTable::new(cache_config.block_size);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        let logits = ModelForward::forward(&model, &input, 0, &mut kv_cache_mgr, &block_table, &[])
            .expect("forward");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_nemotron_h_moe_only_pattern() {
        let mut cfg = test_nemotron_h_config();
        cfg.extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("EE"),
        );
        cfg.num_hidden_layers = 2;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.num_attention_layers(), 0);
        assert_eq!(model.layer_types(), vec!['E', 'E']);
    }

    #[test]
    fn test_nemotron_h_attention_only_pattern() {
        let mut cfg = test_nemotron_h_config();
        cfg.extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("**"),
        );
        cfg.num_hidden_layers = 2;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.num_attention_layers(), 2);
    }

    #[test]
    fn test_nemotron_h_concurrent_requests() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) =
            create_nemotron_h_cache(&cfg, model.num_attention_layers());

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        // Prefill two different requests
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

    #[test]
    fn test_nemotron_h_puzzle_per_layer_experts() {
        let mut cfg = test_nemotron_h_config();
        cfg.extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("EE"),
        );
        cfg.num_hidden_layers = 2;

        // Puzzle variant: different expert counts per layer
        cfg.extra.insert(
            "block_configs".to_string(),
            serde_json::json!([
                {"block_type": "moe", "n_routed_experts": 4},
                {"block_type": "moe", "n_routed_experts": 8}
            ]),
        );

        let nhc = NemotronHConfig::from_model_config(&cfg);
        assert_eq!(nhc.n_routed_experts_for_layer(0), 4);
        assert_eq!(nhc.n_routed_experts_for_layer(1), 8);
    }

    #[test]
    fn test_nemotron_h_intermediate_size_per_mlp() {
        let mut cfg = test_nemotron_h_config();
        cfg.extra.insert(
            "intermediate_size".to_string(),
            serde_json::json!([128, 256, 64]),
        );
        cfg.extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("---"),
        );

        let nhc = NemotronHConfig::from_model_config(&cfg);
        assert_eq!(nhc.intermediate_size_for_mlp(0), 128);
        assert_eq!(nhc.intermediate_size_for_mlp(1), 256);
        assert_eq!(nhc.intermediate_size_for_mlp(2), 64);
    }

    #[test]
    fn test_nemotron_h_shared_experts() {
        let mut cfg = test_nemotron_h_config();
        cfg.extra
            .insert("n_shared_experts".to_string(), serde_json::json!(2));
        cfg.extra.insert(
            "moe_shared_expert_intermediate_size".to_string(),
            serde_json::json!(32),
        );
        cfg.extra.insert(
            "hybrid_override_pattern".to_string(),
            serde_json::json!("E"),
        );
        cfg.num_hidden_layers = 1;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "NemotronH with shared experts should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_nemotron_h_device() {
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = NemotronHForCausalLM::new(&cfg, vb).expect("model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_nemotron_h_puzzle_type_alias() {
        // NemotronHPuzzleForCausalLM is a type alias for NemotronHForCausalLM
        let cfg = test_nemotron_h_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model: NemotronHPuzzleForCausalLM = NemotronHForCausalLM::new(&cfg, vb).expect("model");
        assert_eq!(model.num_layers(), 4);
    }
}
