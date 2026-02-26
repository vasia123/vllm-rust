//! MiMo V2 Flash model implementation.
//!
//! MiMo V2 Flash is a hybrid dense/MoE model with:
//! - Hybrid attention: alternating full attention and sliding window attention (SWA)
//!   controlled by `hybrid_layer_pattern` (0 = full, 1 = sliding window)
//! - Optional asymmetric V head dimension (`v_head_dim` != `head_dim`)
//! - V scaling: attention value scaling factor
//! - Mixed MoE/dense layers: `moe_layer_freq` list determines which layers use MoE
//! - SwiGLU activation
//! - RMSNorm
//!
//! Architecturally similar to Qwen2 with DeepSeek V2-style attention features.

use crate::layers::{rms_norm, RmsNorm};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── MiMo V2 Attention ──────────────────────────────────────────────────────
//
// Supports asymmetric head dims (v_head_dim may differ from head_dim)
// and optional value scaling.

struct MiMoV2Attention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    v_scale: Option<f64>,
}

impl MiMoV2Attention {
    #[allow(clippy::too_many_arguments)]
    fn new_with_tp(
        cfg: &ModelConfig,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        v_head_dim: Option<usize>,
        v_scale: Option<f64>,
        use_bias: bool,
        rope_theta: f64,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let world_size = pg.world_size();
        let v_head_dim = v_head_dim.unwrap_or(head_dim);

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

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            use_bias,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            use_bias,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * v_head_dim,
            use_bias,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        // Output projection maps from num_heads * v_head_dim
        let o_proj = TpLinear::row_parallel(
            num_heads * v_head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            v_head_dim,
            v_scale,
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

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Handle v_head_dim != head_dim by padding V to head_dim for cache compatibility
        let v = if self.v_head_dim != self.head_dim {
            let v = v.reshape((b_sz, q_len, self.num_kv_heads, self.v_head_dim))?;
            // Apply v_scale before attention
            let v = if let Some(scale) = self.v_scale {
                (v * scale)?
            } else {
                v
            };
            // Pad V to head_dim for cache storage
            let pad_size = self.head_dim - self.v_head_dim;
            let padding = Tensor::zeros(
                (b_sz, q_len, self.num_kv_heads, pad_size),
                v.dtype(),
                v.device(),
            )?;
            let v_padded = Tensor::cat(&[&v, &padding], 3)?;
            v_padded.transpose(1, 2)?
        } else {
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            if let Some(scale) = self.v_scale {
                (v * scale)?
            } else {
                v
            }
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

        // If v_head_dim != head_dim, truncate the attention output back
        let attn_output = if self.v_head_dim != self.head_dim {
            // attn_output: [b, seq, num_heads * head_dim] -> reshape, narrow, reshape
            let attn = attn_output.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
            let attn = attn.narrow(3, 0, self.v_head_dim)?;
            attn.reshape((b_sz, q_len, self.num_heads * self.v_head_dim))?
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

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Pad V if needed
        let v = if self.v_head_dim != self.head_dim {
            let v = v.reshape((batch_size, 1, self.num_kv_heads, self.v_head_dim))?;
            let v = if let Some(scale) = self.v_scale {
                (v * scale)?
            } else {
                v
            };
            let pad_size = self.head_dim - self.v_head_dim;
            let padding = Tensor::zeros(
                (batch_size, 1, self.num_kv_heads, pad_size),
                v.dtype(),
                v.device(),
            )?;
            Tensor::cat(&[&v, &padding], 3)?.transpose(1, 2)?
        } else {
            let v = v
                .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            if let Some(scale) = self.v_scale {
                (v * scale)?
            } else {
                v
            }
        };

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

            // Truncate back to v_head_dim if needed
            let attn_out = if self.v_head_dim != self.head_dim {
                let attn = attn_out.reshape((1, 1, self.num_heads, self.head_dim))?;
                let attn = attn.narrow(3, 0, self.v_head_dim)?;
                attn.reshape((1, 1, self.num_heads * self.v_head_dim))?
            } else {
                attn_out
            };

            outputs.push(attn_out);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output, tp_ctx)
    }
}

// ─── MiMo V2 Config ─────────────────────────────────────────────────────────

struct MiMoV2Config {
    hybrid_layer_pattern: Vec<u8>,
    moe_layer_freq: Vec<bool>,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: usize,
    swa_head_dim: usize,
    swa_v_head_dim: Option<usize>,
    swa_rope_theta: f64,
    #[allow(dead_code)]
    sliding_window_size: Option<usize>,
    v_head_dim: Option<usize>,
    v_scale: Option<f64>,
    layernorm_epsilon: f64,
    // MoE
    n_routed_experts: Option<usize>,
    num_experts_per_tok: usize,
    moe_intermediate_size: Option<usize>,
    norm_topk_prob: bool,
    n_group: Option<usize>,
    topk_group: Option<usize>,
}

impl MiMoV2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let hybrid_layer_pattern = cfg
            .extra
            .get("hybrid_layer_pattern")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_u64().unwrap_or(0) as u8).collect())
            .unwrap_or_else(|| vec![0; cfg.num_hidden_layers]);

        let moe_layer_freq = cfg
            .extra
            .get("moe_layer_freq")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
            .unwrap_or_else(|| vec![false; cfg.num_hidden_layers]);

        let swa_num_attention_heads = cfg
            .extra
            .get("swa_num_attention_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_attention_heads);

        let swa_num_key_value_heads = cfg
            .extra
            .get("swa_num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_key_value_heads);

        let swa_head_dim = cfg
            .extra
            .get("swa_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let swa_v_head_dim = cfg
            .extra
            .get("swa_v_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let swa_rope_theta = cfg
            .extra
            .get("swa_rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);

        let sliding_window_size = cfg
            .extra
            .get("sliding_window_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or(cfg.sliding_window);

        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let v_scale = cfg
            .extra
            .get("attention_value_scale")
            .and_then(|v| v.as_f64());

        let layernorm_epsilon = cfg
            .extra
            .get("layernorm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        let n_routed_experts = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let moe_intermediate_size = cfg
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let norm_topk_prob = cfg
            .extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let n_group = cfg
            .extra
            .get("n_group")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let topk_group = cfg
            .extra
            .get("topk_group")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Self {
            hybrid_layer_pattern,
            moe_layer_freq,
            swa_num_attention_heads,
            swa_num_key_value_heads,
            swa_head_dim,
            swa_v_head_dim,
            swa_rope_theta,
            sliding_window_size,
            v_head_dim,
            v_scale,
            layernorm_epsilon,
            n_routed_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob,
            n_group,
            topk_group,
        }
    }

    fn is_swa_layer(&self, layer_idx: usize) -> bool {
        self.hybrid_layer_pattern
            .get(layer_idx)
            .copied()
            .unwrap_or(0)
            == 1
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_layer_freq.get(layer_idx).copied().unwrap_or(false)
    }
}

// ─── MoE Block ───────────────────────────────────────────────────────────────

struct MiMoV2Expert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl MiMoV2Expert {
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
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs, tp_ctx)?)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        self.down_proj.forward(&gate.mul(&up)?, tp_ctx)
    }
}

struct MiMoV2MoEBlock {
    router: TopKRouter,
    experts: Vec<MiMoV2Expert>,
    num_experts: usize,
}

impl MiMoV2MoEBlock {
    fn new(
        cfg: &ModelConfig,
        mimo_cfg: &MiMoV2Config,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_experts = mimo_cfg.n_routed_experts.unwrap_or(0);
        let top_k = mimo_cfg.num_experts_per_tok;
        let moe_intermediate_size = mimo_cfg
            .moe_intermediate_size
            .unwrap_or(cfg.intermediate_size);

        let router_config = RouterConfig {
            hidden_size: cfg.hidden_size,
            num_experts,
            top_k,
            renormalize: mimo_cfg.norm_topk_prob,
            scoring_func: ScoringFunc::Sigmoid,
            use_grouped_topk: mimo_cfg.n_group.is_some(),
            num_expert_groups: mimo_cfg.n_group,
            topk_per_group: mimo_cfg.topk_group,
            routed_scaling_factor: 1.0,
        };

        // Load optional score correction bias stored alongside the gate weights.
        let bias = vb
            .pp("gate")
            .get((num_experts,), "e_score_correction_bias")
            .ok();
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), bias)?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(MiMoV2Expert::new(
                cfg.hidden_size,
                moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;

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
                    token_output = (token_output + expert_out.affine(weights[k] as f64, 0.0)?)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            routed_output = routed_output.index_add(&indices, &token_output, 0)?;
        }

        routed_output.reshape(orig_shape)
    }
}

enum MiMoV2Mlp {
    Dense(TpSwiGluMlp),
    Sparse(MiMoV2MoEBlock),
}

impl MiMoV2Mlp {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            MiMoV2Mlp::Dense(mlp) => mlp.forward(xs, tp_ctx),
            MiMoV2Mlp::Sparse(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct MiMoV2FlashDecoderLayer {
    self_attn: MiMoV2Attention,
    mlp: MiMoV2Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MiMoV2FlashDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        mimo_cfg: &MiMoV2Config,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let use_bias = cfg.attention_bias.unwrap_or(false);

        let self_attn = if mimo_cfg.is_swa_layer(layer_idx) {
            // Sliding window attention layer uses SWA-specific head config
            MiMoV2Attention::new_with_tp(
                cfg,
                mimo_cfg.swa_num_attention_heads,
                mimo_cfg.swa_num_key_value_heads,
                mimo_cfg.swa_head_dim,
                mimo_cfg.swa_v_head_dim,
                mimo_cfg.v_scale,
                use_bias,
                mimo_cfg.swa_rope_theta,
                vb.pp("self_attn"),
                pg,
            )?
        } else {
            // Full attention layer uses standard config
            MiMoV2Attention::new_with_tp(
                cfg,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                mimo_cfg.v_head_dim,
                mimo_cfg.v_scale,
                use_bias,
                cfg.rope_theta,
                vb.pp("self_attn"),
                pg,
            )?
        };

        let mlp = if mimo_cfg.is_moe_layer(layer_idx) {
            MiMoV2Mlp::Sparse(MiMoV2MoEBlock::new(cfg, mimo_cfg, vb.pp("mlp"), pg)?)
        } else {
            MiMoV2Mlp::Dense(TpSwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
                pg,
            )?)
        };

        let input_layernorm = rms_norm(
            cfg.hidden_size,
            mimo_cfg.layernorm_epsilon,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            mimo_cfg.layernorm_epsilon,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
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
        cache_engine: &mut CacheEngine,
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
            cache_engine,
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_decode_batch(&xs, sequences, cache_engine, tp_ctx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// MiMo V2 Flash model for causal language modeling.
///
/// Hybrid dense/MoE model with alternating full and sliding window attention layers.
/// Supports asymmetric V head dimensions and value scaling.
pub struct MiMoV2FlashForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<MiMoV2FlashDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl MiMoV2FlashForCausalLM {
    /// Create a new MiMo V2 Flash model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new MiMo V2 Flash model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let mimo_cfg = MiMoV2Config::from_model_config(cfg);
        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MiMoV2FlashDecoderLayer::new_with_tp(
                cfg,
                &mimo_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, mimo_cfg.layernorm_epsilon, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
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
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for MiMoV2FlashForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        MiMoV2FlashForCausalLM::forward(
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
                kv_cache_mgr.engine_mut(layer_idx),
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> crate::config::ModelConfig {
        let mut extra = serde_json::Map::new();
        // hybrid_layer_pattern: layer 0 = full attention, layer 1 = SWA
        extra.insert(
            "hybrid_layer_pattern".to_string(),
            serde_json::json!([0, 1]),
        );
        extra.insert(
            "moe_layer_freq".to_string(),
            serde_json::json!([false, false]),
        );
        extra.insert(
            "layernorm_epsilon".to_string(),
            serde_json::Value::from(1e-6),
        );

        crate::config::ModelConfig {
            architectures: vec!["MiMoV2FlashForCausalLM".to_string()],
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
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
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

    #[test]
    fn test_mimo_v2_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MiMoV2FlashForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiMoV2FlashForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_mimo_v2_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiMoV2FlashForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
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

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_mimo_v2_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiMoV2FlashForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_mimo_v2_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiMoV2FlashForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mimo_v2_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiMoV2FlashForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
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
    fn test_mimo_v2_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiMoV2FlashForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_mimo_v2_hybrid_layer_pattern() {
        let cfg = test_config();
        let mimo_cfg = MiMoV2Config::from_model_config(&cfg);

        assert!(
            !mimo_cfg.is_swa_layer(0),
            "layer 0 should be full attention"
        );
        assert!(mimo_cfg.is_swa_layer(1), "layer 1 should be SWA");
    }

    #[test]
    fn test_mimo_v2_moe_layer_detection() {
        let cfg = test_config();
        let mimo_cfg = MiMoV2Config::from_model_config(&cfg);

        assert!(!mimo_cfg.is_moe_layer(0), "layer 0 should be dense");
        assert!(!mimo_cfg.is_moe_layer(1), "layer 1 should be dense");
    }

    #[test]
    fn test_mimo_v2_moe_construction() {
        let mut cfg = test_config();
        // Layer 0 = MoE, layer 1 = dense
        cfg.extra.insert(
            "moe_layer_freq".to_string(),
            serde_json::json!([true, false]),
        );
        cfg.extra
            .insert("n_routed_experts".to_string(), serde_json::json!(4u64));
        cfg.extra
            .insert("num_experts_per_tok".to_string(), serde_json::json!(2u64));
        cfg.extra.insert(
            "moe_intermediate_size".to_string(),
            serde_json::json!(64u64),
        );
        cfg.extra
            .insert("n_group".to_string(), serde_json::json!(2u64));
        cfg.extra
            .insert("topk_group".to_string(), serde_json::json!(1u64));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiMoV2FlashForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MoE model construction failed: {:?}",
            model.err()
        );
    }
}
