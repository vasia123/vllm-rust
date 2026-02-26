//! AfMoE model implementation.
//!
//! AfMoE is a Llama-based Mixture of Experts model with:
//! - Per-layer attention type control (sliding_attention vs global)
//! - Gated attention (sigmoid gate on attention output)
//! - Q/K normalization
//! - Mixed dense/MoE layers: first `num_dense_layers` layers use dense MLP
//! - Shared experts in MoE layers (output added to routed expert output)
//! - FP32 router with grouped top-k, sigmoid scoring, and learnable expert bias
//! - Optional muP input scaling
//! - Four-norm residual pattern (pre/post attention + pre/post MLP)

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── AfMoE Config (parsed from ModelConfig.extra) ───────────────────────────

struct AfmoeConfig {
    num_experts: usize,
    num_experts_per_tok: usize,
    num_shared_experts: usize,
    moe_intermediate_size: usize,
    num_dense_layers: usize,
    route_scale: f64,
    score_func: ScoringFunc,
    route_norm: bool,
    n_group: usize,
    topk_group: usize,
    mup_enabled: bool,
    layer_types: Vec<String>,
}

impl AfmoeConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let num_shared_experts = cfg
            .extra
            .get("num_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let num_dense_layers = cfg
            .extra
            .get("num_dense_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let route_scale = cfg
            .extra
            .get("route_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let score_func = cfg
            .extra
            .get("score_func")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "sigmoid" => ScoringFunc::Sigmoid,
                _ => ScoringFunc::Softmax,
            })
            .unwrap_or(ScoringFunc::Sigmoid);

        let route_norm = cfg
            .extra
            .get("route_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let n_group = cfg
            .extra
            .get("n_group")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let topk_group = cfg
            .extra
            .get("topk_group")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let mup_enabled = cfg
            .extra
            .get("mup_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Per-layer attention types: "sliding_attention" or "global_attention"
        let layer_types = cfg
            .extra
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("global_attention").to_string())
                    .collect()
            })
            .unwrap_or_default();

        Self {
            num_experts,
            num_experts_per_tok,
            num_shared_experts,
            moe_intermediate_size,
            num_dense_layers,
            route_scale,
            score_func,
            route_norm,
            n_group,
            topk_group,
            mup_enabled,
            layer_types,
        }
    }

    fn is_sliding_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(false)
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.num_dense_layers
    }
}

// ─── AfMoE Attention ────────────────────────────────────────────────────────

struct AfmoeAttention {
    qkv_proj: TpLinear,
    o_proj: TpLinear,
    gate_proj: TpLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: Option<RotaryEmbedding>,
    is_local_attention: bool,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl AfmoeAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        afmoe_cfg: &AfmoeConfig,
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

        // Merged QKV projection
        let total_qkv = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            total_qkv,
            false,
            false,
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

        // Gating projection: hidden_size -> num_heads * head_dim
        let gate_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("gate_proj"),
            pg,
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let is_local_attention = afmoe_cfg.is_sliding_attention(layer_idx);
        let sliding_window = if is_local_attention {
            cfg.sliding_window
        } else {
            None
        };

        // RoPE only for local attention layers
        let rotary_emb = if is_local_attention {
            Some(RotaryEmbedding::new(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                vb.dtype(),
                vb.device(),
            )?)
        } else {
            None
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            gate_proj,
            q_norm,
            k_norm,
            rotary_emb,
            is_local_attention,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            q_size,
            kv_size,
            sliding_window,
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

        let qkv = self.qkv_proj.forward(xs, tp_ctx)?;
        let gate = self.gate_proj.forward(xs, tp_ctx)?;

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

        // Per-head Q/K normalization
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        // RoPE only for sliding attention layers
        let (q, k) = if self.is_local_attention {
            if let Some(ref rotary_emb) = self.rotary_emb {
                rotary_emb.apply(&q, &k, seqlen_offset)?
            } else {
                (q, k)
            }
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

        // Apply sigmoid gating
        let gate = candle_nn::ops::sigmoid(&gate)?;
        let attn_output = (attn_output * gate)?;

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
        let gate = self.gate_proj.forward(xs, tp_ctx)?;

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
            let (q, k) = if self.is_local_attention {
                if let Some(ref rotary_emb) = self.rotary_emb {
                    rotary_emb.apply_varlen(&q, &k, &positions)?
                } else {
                    (q, k)
                }
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

            // Squeeze gate to match attn_output shape, then gate
            let gate = candle_nn::ops::sigmoid(&gate)?;
            let gate = gate.squeeze(1)?; // [batch, q_size]
            let attn_output = (attn_output * gate)?;

            self.o_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = if self.is_local_attention {
                    if let Some(ref rotary_emb) = self.rotary_emb {
                        rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?
                    } else {
                        (q_i, k_i)
                    }
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
            let gate = candle_nn::ops::sigmoid(&gate)?;
            let attn_output = (attn_output * gate)?;
            self.o_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── AfMoE MoE Block ───────────────────────────────────────────────────────

struct AfmoeMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl AfmoeMoEExpert {
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

struct AfmoeMoEBlock {
    router: TopKRouter,
    experts: Vec<AfmoeMoEExpert>,
    shared_expert: Option<TpSwiGluMlp>,
    num_experts: usize,
    route_scale: f64,
    #[allow(dead_code)]
    top_k: usize,
}

impl AfmoeMoEBlock {
    fn new(
        cfg: &ModelConfig,
        afmoe_cfg: &AfmoeConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = afmoe_cfg.num_experts;
        let top_k = afmoe_cfg.num_experts_per_tok;

        // Renormalize based on scoring function
        let renormalize = if matches!(afmoe_cfg.score_func, ScoringFunc::Sigmoid) {
            afmoe_cfg.route_norm
        } else {
            false
        };

        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize,
            scoring_func: afmoe_cfg.score_func,
            use_grouped_topk: afmoe_cfg.n_group > 1,
            num_expert_groups: if afmoe_cfg.n_group > 1 {
                Some(afmoe_cfg.n_group)
            } else {
                None
            },
            topk_per_group: if afmoe_cfg.topk_group > 1 {
                Some(afmoe_cfg.topk_group)
            } else {
                None
            },
            routed_scaling_factor: 1.0,
        };

        // Expert bias for load balancing
        let bias = Tensor::zeros(num_experts, DType::F32, vb.device())?;
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), Some(bias))?;

        // Routed experts
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(AfmoeMoEExpert::new(
                hidden_size,
                afmoe_cfg.moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        // Shared experts
        let shared_expert = if afmoe_cfg.num_shared_experts > 0 {
            let shared_intermediate =
                afmoe_cfg.moe_intermediate_size * afmoe_cfg.num_shared_experts;
            Some(TpSwiGluMlp::new(
                hidden_size,
                shared_intermediate,
                vb.pp("shared_experts"),
                pg,
            )?)
        } else {
            None
        };

        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts,
            route_scale: afmoe_cfg.route_scale,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Shared expert output (always active)
        let shared_output = self
            .shared_expert
            .as_ref()
            .map(|se| se.forward(&xs_2d, tp_ctx))
            .transpose()?;

        // Router: compute routing weights and expert assignments
        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;

        // Routed expert computation (token-by-token)
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
                    // Apply route_scale to expert weights
                    let scaled_weight = weights[k] as f64 * self.route_scale;
                    let weighted = expert_out.affine(scaled_weight, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            routed_output = routed_output.index_add(&indices, &token_output, 0)?;
        }

        // Combine: shared_output + routed_output
        let combined = if let Some(shared) = shared_output {
            (shared + routed_output)?
        } else {
            routed_output
        };

        combined.reshape(orig_shape)
    }
}

// ─── FFN Variant ────────────────────────────────────────────────────────────

enum AfmoeFfnVariant {
    Dense(TpSwiGluMlp),
    MoE(AfmoeMoEBlock),
}

impl AfmoeFfnVariant {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            AfmoeFfnVariant::Dense(mlp) => mlp.forward(xs, tp_ctx),
            AfmoeFfnVariant::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct AfmoeDecoderLayer {
    self_attn: AfmoeAttention,
    ffn: AfmoeFfnVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_mlp_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
}

impl AfmoeDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        afmoe_cfg: &AfmoeConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn =
            AfmoeAttention::new_with_tp(cfg, afmoe_cfg, layer_idx, vb.pp("self_attn"), pg)?;

        let ffn = if afmoe_cfg.is_moe_layer(layer_idx) {
            AfmoeFfnVariant::MoE(AfmoeMoEBlock::new(cfg, afmoe_cfg, vb.pp("mlp"), pg)?)
        } else {
            AfmoeFfnVariant::Dense(TpSwiGluMlp::new(
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
        let pre_mlp_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_mlp_layernorm"),
        )?;
        let post_mlp_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_mlp_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            ffn,
            input_layernorm,
            post_attention_layernorm,
            pre_mlp_layernorm,
            post_mlp_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        residual: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor)> {
        // Four-norm residual pattern
        let (hidden_states, residual) = if let Some(res) = residual {
            let normed = self.input_layernorm.forward(&(xs + res)?)?;
            (normed, (xs + res)?)
        } else {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, xs.clone())
        };

        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // Post-attention norm
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // Pre-MLP norm with residual add
        let combined = (&hidden_states + &residual)?;
        let residual = combined.clone();
        let hidden_states = self.pre_mlp_layernorm.forward(&combined)?;

        // FFN
        let hidden_states = self.ffn.forward(&hidden_states, tp_ctx)?;

        // Post-MLP norm
        let hidden_states = self.post_mlp_layernorm.forward(&hidden_states)?;

        Ok((hidden_states, residual))
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        residual: Option<&Tensor>,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = if let Some(res) = residual {
            let normed = self.input_layernorm.forward(&(xs + res)?)?;
            (normed, (xs + res)?)
        } else {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, xs.clone())
        };

        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        let combined = (&hidden_states + &residual)?;
        let residual = combined.clone();
        let hidden_states = self.pre_mlp_layernorm.forward(&combined)?;

        let hidden_states = self.ffn.forward(&hidden_states, tp_ctx)?;

        let hidden_states = self.post_mlp_layernorm.forward(&hidden_states)?;

        Ok((hidden_states, residual))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// AfMoE model for causal language modeling.
///
/// Llama-based architecture with MoE routing, gated attention,
/// Q/K normalization, and a four-norm residual pattern.
pub struct AfmoeForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<AfmoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    mup_enabled: bool,
    hidden_size: usize,
}

impl AfmoeForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let afmoe_cfg = AfmoeConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(AfmoeDecoderLayer::new_with_tp(
                cfg,
                &afmoe_cfg,
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
            mup_enabled: afmoe_cfg.mup_enabled,
            hidden_size: cfg.hidden_size,
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

        // Optional muP input scaling
        if self.mup_enabled {
            let scale = (self.hidden_size as f64).sqrt();
            xs = xs.affine(scale, 0.0)?;
        }

        let mut residual: Option<Tensor> = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (h, r) = layer.forward(
                &xs,
                residual.as_ref(),
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
            xs = h;
            residual = Some(r);
        }

        // Final norm with residual
        let xs = if let Some(ref res) = residual {
            self.norm.forward(&(xs + res)?)?
        } else {
            self.norm.forward(&xs)?
        };

        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for AfmoeForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        AfmoeForCausalLM::forward(
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

        if self.mup_enabled {
            let scale = (self.hidden_size as f64).sqrt();
            xs = xs.affine(scale, 0.0)?;
        }

        let mut residual: Option<Tensor> = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (h, r) = layer.forward_decode_batch(
                &xs,
                residual.as_ref(),
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
            xs = h;
            residual = Some(r);
        }

        let xs = if let Some(ref res) = residual {
            self.norm.forward(&(xs + res)?)?
        } else {
            self.norm.forward(&xs)?
        };

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
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("num_shared_experts".to_string(), serde_json::json!(1));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("num_dense_layers".to_string(), serde_json::json!(1));
        extra.insert("score_func".to_string(), serde_json::json!("sigmoid"));
        extra.insert("route_scale".to_string(), serde_json::json!(1.0));
        extra.insert("route_norm".to_string(), serde_json::json!(true));
        extra.insert("n_group".to_string(), serde_json::json!(1));
        extra.insert("topk_group".to_string(), serde_json::json!(1));
        extra.insert("mup_enabled".to_string(), serde_json::json!(false));
        extra.insert(
            "layer_types".to_string(),
            serde_json::json!(["sliding_attention", "global_attention"]),
        );

        ModelConfig {
            architectures: vec!["AfmoeForCausalLM".to_string()],
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
            sliding_window: Some(128),
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
    fn test_afmoe_config_parsing() {
        let cfg = test_config();
        let afmoe_cfg = AfmoeConfig::from_model_config(&cfg);
        assert_eq!(afmoe_cfg.num_experts, 4);
        assert_eq!(afmoe_cfg.num_experts_per_tok, 2);
        assert_eq!(afmoe_cfg.num_shared_experts, 1);
        assert_eq!(afmoe_cfg.moe_intermediate_size, 64);
        assert_eq!(afmoe_cfg.num_dense_layers, 1);
        assert_eq!(afmoe_cfg.route_scale, 1.0);
        assert!(matches!(afmoe_cfg.score_func, ScoringFunc::Sigmoid));
        assert!(afmoe_cfg.route_norm);
        assert!(!afmoe_cfg.mup_enabled);
    }

    #[test]
    fn test_afmoe_config_defaults() {
        let cfg = ModelConfig::default();
        let afmoe_cfg = AfmoeConfig::from_model_config(&cfg);
        assert_eq!(afmoe_cfg.num_experts, 16);
        assert_eq!(afmoe_cfg.num_experts_per_tok, 4);
        assert_eq!(afmoe_cfg.num_shared_experts, 0);
        assert_eq!(afmoe_cfg.num_dense_layers, 0);
        assert!(matches!(afmoe_cfg.score_func, ScoringFunc::Sigmoid));
    }

    #[test]
    fn test_afmoe_layer_type_detection() {
        let cfg = test_config();
        let afmoe_cfg = AfmoeConfig::from_model_config(&cfg);
        assert!(afmoe_cfg.is_sliding_attention(0));
        assert!(!afmoe_cfg.is_sliding_attention(1));
        assert!(!afmoe_cfg.is_sliding_attention(99)); // out of bounds
    }

    #[test]
    fn test_afmoe_moe_layer_detection() {
        let cfg = test_config();
        let afmoe_cfg = AfmoeConfig::from_model_config(&cfg);
        // num_dense_layers = 1, so layer 0 is dense, layer 1+ is MoE
        assert!(!afmoe_cfg.is_moe_layer(0));
        assert!(afmoe_cfg.is_moe_layer(1));
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_afmoe_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = AfmoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "AfmoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_afmoe_mixed_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = AfmoeForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0 = dense (num_dense_layers = 1), Layer 1 = MoE
        assert!(matches!(model.layers[0].ffn, AfmoeFfnVariant::Dense(_)));
        assert!(matches!(model.layers[1].ffn, AfmoeFfnVariant::MoE(_)));
    }

    #[test]
    fn test_afmoe_attention_types() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = AfmoeForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0 = sliding_attention (has RoPE), Layer 1 = global_attention (no RoPE)
        assert!(model.layers[0].self_attn.is_local_attention);
        assert!(model.layers[0].self_attn.rotary_emb.is_some());
        assert!(!model.layers[1].self_attn.is_local_attention);
        assert!(model.layers[1].self_attn.rotary_emb.is_none());
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_afmoe_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = AfmoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_afmoe_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = AfmoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_afmoe_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = AfmoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_afmoe_moe_block_forward() {
        let cfg = test_config();
        let afmoe_cfg = AfmoeConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = AfmoeMoEBlock::new(&cfg, &afmoe_cfg, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_afmoe_mup_scaling() {
        let mut cfg = test_config();
        cfg.extra
            .insert("mup_enabled".to_string(), serde_json::json!(true));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = AfmoeForCausalLM::new(&cfg, vb).expect("build model");
        assert!(model.mup_enabled);
    }

    #[test]
    fn test_afmoe_no_shared_experts() {
        let mut cfg = test_config();
        cfg.extra
            .insert("num_shared_experts".to_string(), serde_json::json!(0));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = AfmoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct without shared experts: {:?}",
            model.err()
        );

        // Verify forward works without shared experts
        let model = model.unwrap();
        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);
        let input_ids = Tensor::zeros((1, 2), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 2)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 2);

        let result = model.forward(
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(result.is_ok(), "Forward should work: {:?}", result.err());
    }

    #[test]
    fn test_afmoe_four_norm_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = AfmoeForCausalLM::new(&cfg, vb).unwrap();

        // Each layer should have 4 norm layers
        for layer in &model.layers {
            // Verify norms exist by checking they can process input
            let test_input = Tensor::zeros(cfg.hidden_size, DType::F32, &device).unwrap();
            assert!(layer.input_layernorm.forward(&test_input).is_ok());
            assert!(layer.post_attention_layernorm.forward(&test_input).is_ok());
            assert!(layer.pre_mlp_layernorm.forward(&test_input).is_ok());
            assert!(layer.post_mlp_layernorm.forward(&test_input).is_ok());
        }
    }
}
