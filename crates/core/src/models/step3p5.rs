//! Step-3.5-Flash (AI21 Jurassic) model implementation.
//!
//! Step-3.5 is a hybrid MoE model with:
//! - Per-head QK normalization (like Qwen3)
//! - Partial RoPE with per-layer configurable factors
//! - SwiGLU activation with optional per-layer clamping limits
//! - Mixed MoE/MLP layers: layer 0 = dense MLP, layers 1+ = MoE
//! - Shared expert in every MoE block (always active, output added to routed)
//! - FP32 router with sigmoid scoring and learnable bias
//! - Optional head-wise attention gating

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Step3p5 Config (parsed from ModelConfig.extra) ─────────────────────────

/// Parsed Step3p5-specific configuration from the HuggingFace config.
struct Step3p5Config {
    moe_num_experts: usize,
    moe_top_k: usize,
    moe_intermediate_size: usize,
    share_expert_dim: usize,
    norm_expert_weight: bool,
    moe_layers: Vec<usize>,
    partial_rotary_factors: Vec<f64>,
    swiglu_limits_shared: Vec<Option<f64>>,
    swiglu_limits: Vec<Option<f64>>,
    use_head_wise_attn_gate: bool,
    use_rope_layers: Vec<bool>,
    moe_router_activation: ScoringFunc,
    #[allow(dead_code)]
    moe_router_scaling_factor: f64,
}

impl Step3p5Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let moe_num_experts = cfg
            .extra
            .get("moe_num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);

        let moe_top_k = cfg
            .extra
            .get("moe_top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let share_expert_dim = cfg
            .extra
            .get("share_expert_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(moe_intermediate_size * moe_top_k);

        let norm_expert_weight = cfg
            .extra
            .get("norm_expert_weight")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let use_head_wise_attn_gate = cfg
            .extra
            .get("use_head_wise_attn_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Parse moe_layers_enum: comma-separated layer indices
        let moe_layers = cfg
            .extra
            .get("moe_layers_enum")
            .and_then(|v| v.as_str())
            .map(|s| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_else(|| (1..cfg.num_hidden_layers).collect());

        // Per-layer partial rotary factors (default 1.0 = full rotation)
        let partial_rotary_factors = cfg
            .extra
            .get("partial_rotary_factors")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(1.0)).collect())
            .unwrap_or_default();

        // Per-layer SwiGLU limits for shared experts
        let swiglu_limits_shared = cfg
            .extra
            .get("swiglu_limits_shared")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_f64().filter(|&f| f != 0.0))
                    .collect()
            })
            .unwrap_or_default();

        // Per-layer SwiGLU limits for routed experts
        let swiglu_limits = cfg
            .extra
            .get("swiglu_limits")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_f64().filter(|&f| f != 0.0))
                    .collect()
            })
            .unwrap_or_default();

        // Per-layer RoPE toggle (default: all layers use RoPE)
        let use_rope_layers = cfg
            .extra
            .get("use_rope_layers")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(true)).collect())
            .unwrap_or_default();

        let moe_router_activation = cfg
            .extra
            .get("moe_router_activation")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "sigmoid" => ScoringFunc::Sigmoid,
                _ => ScoringFunc::Softmax,
            })
            .unwrap_or(ScoringFunc::Sigmoid);

        let moe_router_scaling_factor = cfg
            .extra
            .get("moe_router_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Self {
            moe_num_experts,
            moe_top_k,
            moe_intermediate_size,
            share_expert_dim,
            norm_expert_weight,
            moe_layers,
            partial_rotary_factors,
            swiglu_limits_shared,
            swiglu_limits,
            use_head_wise_attn_gate,
            use_rope_layers,
            moe_router_activation,
            moe_router_scaling_factor,
        }
    }

    fn partial_rotary_factor(&self, layer_idx: usize) -> f64 {
        self.partial_rotary_factors
            .get(layer_idx)
            .copied()
            .unwrap_or(1.0)
    }

    fn swiglu_limit_shared(&self, layer_idx: usize) -> Option<f64> {
        self.swiglu_limits_shared
            .get(layer_idx)
            .copied()
            .unwrap_or(None)
    }

    #[allow(dead_code)]
    fn swiglu_limit(&self, layer_idx: usize) -> Option<f64> {
        self.swiglu_limits.get(layer_idx).copied().unwrap_or(None)
    }

    fn use_rope(&self, layer_idx: usize) -> bool {
        self.use_rope_layers.get(layer_idx).copied().unwrap_or(true)
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_layers.contains(&layer_idx)
    }
}

// ─── Clamped SwiGLU Activation ──────────────────────────────────────────────

/// Clamped SwiGLU: min(silu(gate), limit) * clamp(up, -limit, limit).
///
/// When no limit is set, falls back to standard SwiGLU: silu(gate) * up.
fn swiglu_with_limit(gate_up: &Tensor, limit: Option<f64>) -> Result<Tensor> {
    let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
    let gate = &chunks[0];
    let up = &chunks[1];

    let gate_act = candle_nn::ops::silu(gate)?;

    match limit {
        Some(l) => {
            let gate_clamped = gate_act.clamp(-l, l)?;
            let up_clamped = up.clamp(-l, l)?;
            gate_clamped.mul(&up_clamped)
        }
        None => gate_act.mul(up),
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct Step3p5Attention {
    qkv_proj: TpLinear,
    o_proj: TpLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    g_proj: Option<TpLinear>,
    rotary_emb: RotaryEmbedding,
    use_rope: bool,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl Step3p5Attention {
    fn new_with_tp(
        cfg: &ModelConfig,
        step_cfg: &Step3p5Config,
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

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;
        let q_size = num_heads_per_gpu * head_dim;
        let kv_size = num_kv_heads_per_gpu * head_dim;

        // Merged QKV projection: hidden_size -> (q_size + 2*kv_size)
        let total_qkv = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            total_qkv,
            false, // no bias
            false, // no gather
            vb.pp("qkv_proj"),
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

        // Per-head RMSNorm on Q and K (shared norm, per head_dim)
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Optional head-wise attention gate
        let g_proj = if step_cfg.use_head_wise_attn_gate {
            Some(TpLinear::column_parallel(
                cfg.hidden_size,
                num_heads,
                false,
                false,
                vb.pp("g_proj"),
                pg,
            )?)
        } else {
            None
        };

        // Partial RoPE: per-layer configurable factor (0.5 or 1.0)
        let partial_rotary_factor = step_cfg.partial_rotary_factor(layer_idx);
        let use_rope = step_cfg.use_rope(layer_idx);

        let rotary_emb = if partial_rotary_factor < 1.0 {
            RotaryEmbedding::new_partial(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                partial_rotary_factor,
                true, // neox style
                vb.dtype(),
                vb.device(),
            )?
        } else {
            RotaryEmbedding::new(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                vb.dtype(),
                vb.device(),
            )?
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            g_proj,
            rotary_emb,
            use_rope,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
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
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Merged QKV projection, then split
        let qkv = self.qkv_proj.forward(xs, tp_ctx)?;
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

        // Per-head RMSNorm on Q and K
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        // RoPE (may be disabled for certain layers)
        let (q, k) = if self.use_rope {
            self.rotary_emb.apply(&q, &k, seqlen_offset)?
        } else {
            (q, k)
        };

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

        // Optional head-wise attention gating
        let attn_output = if let Some(ref g_proj) = self.g_proj {
            let gate = g_proj.forward(xs, tp_ctx)?;
            let gate = candle_nn::ops::sigmoid(&gate)?;
            // attn_output: [b, 1, num_heads * head_dim]
            // gate: [b, q_len, num_heads]
            let attn_3d = attn_output.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
            let gate_4d = gate.unsqueeze(3)?; // [b, q_len, num_heads, 1]
            let gated = (attn_3d * gate_4d)?;
            gated.reshape((b_sz, q_len, self.num_heads * self.head_dim))?
        } else {
            attn_output
        };

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

        let qkv = self.qkv_proj.forward(xs, tp_ctx)?;
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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = if self.use_rope {
                self.rotary_emb.apply_varlen(&q, &k, &positions)?
            } else {
                (q, k)
            };

            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let max_blocks_per_seq = sequences
                .iter()
                .map(|s| s.block_ids.len())
                .max()
                .unwrap_or(1);
            let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
            for (i, seq) in sequences.iter().enumerate() {
                for (j, &block_id) in seq.block_ids.iter().enumerate() {
                    bt_data[i * max_blocks_per_seq + j] = block_id as u32;
                }
            }
            let block_tables =
                Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), q.device())?;

            let seq_lens_data: Vec<u32> = sequences
                .iter()
                .map(|s| (s.seqlen_offset + 1) as u32)
                .collect();
            let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
            let seq_lens = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

            let scale = 1.0 / (self.head_dim as f32).sqrt();

            let attn_output = crate::cuda_kernels::paged_attention_cuda(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
            )?;

            self.o_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = if self.use_rope {
                    self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?
                } else {
                    (q_i, k_i)
                };

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
            self.o_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Step3p5 MLP (gate_up merged, optional clamping) ────────────────────────

struct Step3p5MLP {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
    limit: Option<f64>,
}

impl Step3p5MLP {
    fn new_with_tp(
        hidden_size: usize,
        intermediate_size: usize,
        limit: Option<f64>,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // Merged gate+up projection: hidden_size -> 2*intermediate_size
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            false,
            false,
            vb.pp("gate_up_proj"),
            pg,
        )?;

        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true,
            vb.pp("down_proj"),
            pg,
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            limit,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs, tp_ctx)?;
        let intermediate = swiglu_with_limit(&gate_up, self.limit)?;
        self.down_proj.forward(&intermediate, tp_ctx)
    }
}

// ─── MoE Expert (SwiGLU) ───────────────────────────────────────────────────

struct Step3p5MoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Step3p5MoEExpert {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let gate_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("gate_proj"),
            pg,
        )?;
        let up_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true,
            vb.pp("down_proj"),
            pg,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs, tp_ctx)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ─── FusedMoEBlock (shared expert + routed experts) ─────────────────────────

struct Step3p5MoEBlock {
    router: TopKRouter,
    experts: Vec<Step3p5MoEExpert>,
    share_expert: Step3p5MLP,
    num_experts: usize,
    #[allow(dead_code)]
    top_k: usize,
}

impl Step3p5MoEBlock {
    fn new(
        cfg: &ModelConfig,
        step_cfg: &Step3p5Config,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = step_cfg.moe_num_experts;
        let top_k = step_cfg.moe_top_k;

        // FP32 replicated router with sigmoid scoring and bias
        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize: step_cfg.norm_expert_weight,
            scoring_func: step_cfg.moe_router_activation,
            ..Default::default()
        };
        // Router bias is a learnable parameter loaded from checkpoint;
        // initialized to zeros here (set_e_score_correction_bias during weight loading)
        let bias = Tensor::zeros(num_experts, DType::F32, vb.device())?;
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), Some(bias))?;

        // Routed experts
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(Step3p5MoEExpert::new(
                hidden_size,
                step_cfg.moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        // Shared expert (always active, processes all tokens)
        let shared_limit = step_cfg.swiglu_limit_shared(layer_idx);
        let share_expert = Step3p5MLP::new_with_tp(
            hidden_size,
            step_cfg.share_expert_dim,
            shared_limit,
            vb.pp("share_expert"),
            pg,
        )?;

        Ok(Self {
            router,
            experts,
            share_expert,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Shared expert (always active for all tokens)
        let shared_output = self.share_expert.forward(&xs_2d, tp_ctx)?;

        // Router: compute routing weights and expert assignments
        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;

        // Routed expert computation
        let mut routed_output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for token_idx in 0..num_tokens {
            let token_input = xs_2d.narrow(0, token_idx, 1)?;
            let token_weights = routing_weights.narrow(0, token_idx, 1)?;
            let token_experts = selected_experts.narrow(0, token_idx, 1)?;

            let expert_indices: Vec<u32> = token_experts.flatten_all()?.to_vec1()?;
            let weights: Vec<f32> = token_weights
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1()?;

            let mut token_output = Tensor::zeros((1, hidden_dim), xs.dtype(), xs.device())?;
            for (k, &expert_idx) in expert_indices.iter().enumerate() {
                let expert_idx = expert_idx as usize;
                if expert_idx < self.num_experts {
                    let expert_out = self.experts[expert_idx].forward(&token_input, tp_ctx)?;
                    let weighted = expert_out.affine(weights[k] as f64, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            routed_output = routed_output.index_add(&indices, &token_output, 0)?;
        }

        // Combine: shared_output + routed_output
        let combined = (shared_output + routed_output)?;
        combined.reshape(orig_shape)
    }
}

// ─── FFN Variant ────────────────────────────────────────────────────────────

enum FfnVariant {
    Dense(TpSwiGluMlp),
    MoE(Step3p5MoEBlock),
}

impl FfnVariant {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            FfnVariant::Dense(mlp) => mlp.forward(xs, tp_ctx),
            FfnVariant::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct Step3p5DecoderLayer {
    self_attn: Step3p5Attention,
    ffn: FfnVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Step3p5DecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        step_cfg: &Step3p5Config,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn =
            Step3p5Attention::new_with_tp(cfg, step_cfg, layer_idx, vb.pp("self_attn"), pg)?;

        let ffn = if step_cfg.is_moe_layer(layer_idx) {
            FfnVariant::MoE(Step3p5MoEBlock::new(
                cfg,
                step_cfg,
                layer_idx,
                vb.pp("moe"),
                pg,
            )?)
        } else {
            // Dense MLP with optional SwiGLU clamping for shared expert
            FfnVariant::Dense(TpSwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
                pg,
            )?)
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
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.ffn.forward(&xs, tp_ctx)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.ffn.forward(&xs, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Step-3.5-Flash (AI21 Jurassic) model for causal language modeling.
pub struct Step3p5ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Step3p5DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Step3p5ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let step_cfg = Step3p5Config::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Step3p5DecoderLayer::new_with_tp(
                cfg,
                &step_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
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

        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Step3p5ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Step3p5ForCausalLM::forward(
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
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("moe_num_experts".to_string(), serde_json::json!(4));
        extra.insert("moe_top_k".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("share_expert_dim".to_string(), serde_json::json!(128));
        extra.insert("norm_expert_weight".to_string(), serde_json::json!(true));
        extra.insert(
            "moe_router_activation".to_string(),
            serde_json::json!("sigmoid"),
        );
        // Layers 1+ are MoE, layer 0 is dense
        extra.insert("moe_layers_enum".to_string(), serde_json::json!("1"));

        ModelConfig {
            architectures: vec!["Step3p5ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
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

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_step3p5_config_parsing_defaults() {
        let cfg = ModelConfig::default();
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        assert_eq!(step_cfg.moe_num_experts, 16);
        assert_eq!(step_cfg.moe_top_k, 4);
        assert!(step_cfg.use_rope(0));
        assert_eq!(step_cfg.partial_rotary_factor(0), 1.0);
        assert!(step_cfg.swiglu_limit_shared(0).is_none());
    }

    #[test]
    fn test_step3p5_config_parsing_custom() {
        let cfg = test_config();
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        assert_eq!(step_cfg.moe_num_experts, 4);
        assert_eq!(step_cfg.moe_top_k, 2);
        assert_eq!(step_cfg.share_expert_dim, 128);
        assert!(step_cfg.is_moe_layer(1));
        assert!(!step_cfg.is_moe_layer(0));
    }

    #[test]
    fn test_step3p5_config_partial_rotary() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "partial_rotary_factors".to_string(),
            serde_json::json!([1.0, 0.5]),
        );
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        assert_eq!(step_cfg.partial_rotary_factor(0), 1.0);
        assert_eq!(step_cfg.partial_rotary_factor(1), 0.5);
        assert_eq!(step_cfg.partial_rotary_factor(99), 1.0); // out of bounds = default
    }

    #[test]
    fn test_step3p5_config_swiglu_limits() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "swiglu_limits_shared".to_string(),
            serde_json::json!([0, 7.0]),
        );
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        assert!(step_cfg.swiglu_limit_shared(0).is_none()); // 0 → None
        assert_eq!(step_cfg.swiglu_limit_shared(1), Some(7.0));
    }

    #[test]
    fn test_step3p5_config_use_rope_layers() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "use_rope_layers".to_string(),
            serde_json::json!([true, false]),
        );
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        assert!(step_cfg.use_rope(0));
        assert!(!step_cfg.use_rope(1));
    }

    // ─── Activation Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_without_limit() {
        let device = Device::Cpu;
        // gate=1.0, up=2.0 → silu(1.0) * 2.0
        let gate_up = Tensor::new(&[[1.0_f32, 2.0]], &device).unwrap();
        let result = swiglu_with_limit(&gate_up, None).unwrap();
        let values: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected = 1.0 / (1.0 + (-1.0_f32).exp()) * 2.0; // silu(1) * 2
        assert!((values[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_swiglu_with_limit() {
        let device = Device::Cpu;
        // gate=100.0 (silu→~100), up=100.0, limit=7.0
        // → min(silu(100), 7) * clamp(100, -7, 7) = 7.0 * 7.0 = 49.0
        let gate_up = Tensor::new(&[[100.0_f32, 100.0]], &device).unwrap();
        let result = swiglu_with_limit(&gate_up, Some(7.0)).unwrap();
        let values: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((values[0] - 49.0).abs() < 1e-3, "got {}", values[0]);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_step3p5_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Step3p5ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Step3p5ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_step3p5_mixed_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Step3p5ForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0 = dense MLP, Layer 1 = MoE
        assert!(matches!(model.layers[0].ffn, FfnVariant::Dense(_)));
        assert!(matches!(model.layers[1].ffn, FfnVariant::MoE(_)));
    }

    #[test]
    fn test_step3p5_attention_qk_norm() {
        let cfg = test_config();
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let attn =
            Step3p5Attention::new_with_tp(&cfg, &step_cfg, 0, vb.pp("self_attn"), &pg).unwrap();
        assert_eq!(attn.num_heads, cfg.num_attention_heads);
        assert_eq!(attn.num_kv_heads, cfg.num_key_value_heads);
        assert_eq!(attn.head_dim, cfg.head_dim);
    }

    #[test]
    fn test_step3p5_partial_rotary_attention() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "partial_rotary_factors".to_string(),
            serde_json::json!([1.0, 0.5]),
        );
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        // Layer 0: full rotation, layer 1: half rotation
        let attn0 = Step3p5Attention::new_with_tp(&cfg, &step_cfg, 0, vb.pp("attn0"), &pg).unwrap();
        let attn1 = Step3p5Attention::new_with_tp(&cfg, &step_cfg, 1, vb.pp("attn1"), &pg).unwrap();

        // Both should construct successfully
        assert!(attn0.use_rope);
        assert!(attn1.use_rope);
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_step3p5_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5ForCausalLM::new(&cfg, vb).expect("build model");

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

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_step3p5_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_step3p5_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_step3p5_moe_block_forward() {
        let cfg = test_config();
        let step_cfg = Step3p5Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = Step3p5MoEBlock::new(&cfg, &step_cfg, 1, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_step3p5_mlp_with_limit() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let mlp = Step3p5MLP::new_with_tp(64, 128, Some(7.0), vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }
}
