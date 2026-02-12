//! OpenPangu model family: PanguEmbeddedForCausalLM, PanguProMoEV2ForCausalLM,
//! PanguUltraMoEForCausalLM.
//!
//! All three share a common decoder infrastructure with:
//! - Standard (GQA) attention or MLA attention (config-driven)
//! - Dense MLP or MoE (with shared experts, sigmoid routing)
//! - Optional sandwich norm (pre-MLP + post-MLP normalization)
//! - SiLU activation, RMSNorm, RoPE
//!
//! PanguEmbedded is a dense transformer.
//! PanguProMoEV2 and PanguUltraMoE are MoE variants (layers >= first_k_dense_replace
//! use MoE instead of dense MLP).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Pangu Config (parsed from ModelConfig.extra) ───────────────────────────

/// Parsed Pangu-specific configuration from the HuggingFace config.
#[allow(dead_code)]
struct PanguConfig {
    n_routed_experts: Option<usize>,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    n_shared_experts: Option<usize>,
    first_k_dense_replace: usize,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
    sandwich_norm: bool,
    mlp_bias: bool,
    router_enable_expert_bias: bool,
}

impl PanguConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let n_routed_experts = cfg.num_routed_experts();

        let num_experts_per_tok = cfg.num_experts_per_tok().unwrap_or(2);

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let n_shared_experts = cfg
            .extra
            .get("n_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let first_k_dense_replace = cfg.first_k_dense_replace().unwrap_or(cfg.num_hidden_layers);

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let norm_topk_prob = cfg
            .extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sandwich_norm = cfg
            .extra
            .get("sandwich_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mlp_bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let router_enable_expert_bias = cfg
            .extra
            .get("router_enable_expert_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            n_routed_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            n_shared_experts,
            first_k_dense_replace,
            routed_scaling_factor,
            norm_topk_prob,
            sandwich_norm,
            mlp_bias,
            router_enable_expert_bias,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some() && layer_idx >= self.first_k_dense_replace
    }
}

// ─── Pangu MLP (SwiGLU: merged gate+up, then down) ─────────────────────────

struct PanguMLP {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
}

impl PanguMLP {
    fn new_with_tp(
        hidden_size: usize,
        intermediate_size: usize,
        _bias: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // Merged gate+up projection: hidden_size -> 2*intermediate_size
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            false, // bias not implemented for simplicity (most Pangu configs use bias=false)
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
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs, tp_ctx)?;
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = &chunks[0];
        let up = &chunks[1];
        let gate_act = candle_nn::ops::silu(gate)?;
        let intermediate = gate_act.mul(up)?;
        self.down_proj.forward(&intermediate, tp_ctx)
    }
}

// ─── MoE Expert ─────────────────────────────────────────────────────────────

struct PanguMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl PanguMoEExpert {
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

// ─── Pangu MoE Block ───────────────────────────────────────────────────────

struct PanguMoE {
    router: TopKRouter,
    experts: Vec<PanguMoEExpert>,
    shared_expert: Option<PanguMLP>,
    num_experts: usize,
    routed_scaling_factor: f64,
}

impl PanguMoE {
    fn new(
        cfg: &ModelConfig,
        pangu_cfg: &PanguConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = pangu_cfg.n_routed_experts.unwrap_or(8);

        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k: pangu_cfg.num_experts_per_tok,
            renormalize: pangu_cfg.norm_topk_prob,
            scoring_func: ScoringFunc::Sigmoid,
            ..Default::default()
        };

        let bias = if pangu_cfg.router_enable_expert_bias {
            Some(Tensor::zeros(num_experts, DType::F32, vb.device())?)
        } else {
            None
        };
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), bias)?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(PanguMoEExpert::new(
                hidden_size,
                pangu_cfg.moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        let shared_expert = if let Some(n_shared) = pangu_cfg.n_shared_experts {
            let shared_intermediate = pangu_cfg.moe_intermediate_size * n_shared;
            Some(PanguMLP::new_with_tp(
                hidden_size,
                shared_intermediate,
                false,
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
            routed_scaling_factor: pangu_cfg.routed_scaling_factor,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Router: compute routing weights and expert assignments
        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;

        // Routed expert computation (token-by-token for correctness)
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

        // Apply routed scaling factor
        let final_output = routed_output.affine(self.routed_scaling_factor, 0.0)?;

        // Add shared expert output if present
        let final_output = if let Some(ref shared_expert) = self.shared_expert {
            let shared_output = shared_expert.forward(&xs_2d, tp_ctx)?;
            (final_output + shared_output)?
        } else {
            final_output
        };

        final_output.reshape(orig_shape)
    }
}

// ─── FFN Variant ────────────────────────────────────────────────────────────

enum FfnVariant {
    Dense(TpSwiGluMlp),
    MoE(PanguMoE),
}

impl FfnVariant {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            FfnVariant::Dense(mlp) => mlp.forward(xs, tp_ctx),
            FfnVariant::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct PanguAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl PanguAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
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

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("v_proj"),
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

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

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
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
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
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

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
            self.o_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct PanguDecoderLayer {
    self_attn: PanguAttention,
    ffn: FfnVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_mlp_layernorm: Option<RmsNorm>,
    post_mlp_layernorm: Option<RmsNorm>,
}

impl PanguDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        pangu_cfg: &PanguConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = PanguAttention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;

        let ffn = if pangu_cfg.is_moe_layer(layer_idx) {
            FfnVariant::MoE(PanguMoE::new(cfg, pangu_cfg, vb.pp("mlp"), pg)?)
        } else {
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

        let (pre_mlp_layernorm, post_mlp_layernorm) = if pangu_cfg.sandwich_norm {
            (
                Some(rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("pre_mlp_layernorm"),
                )?),
                Some(rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_mlp_layernorm"),
                )?),
            )
        } else {
            (None, None)
        };

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

        // Sandwich norm: post-attn norm, then optional pre-MLP norm
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = if let Some(ref pre_mlp_norm) = self.pre_mlp_layernorm {
            pre_mlp_norm.forward(&xs)?
        } else {
            xs
        };

        let xs = self.ffn.forward(&xs, tp_ctx)?;

        let xs = if let Some(ref post_mlp_norm) = self.post_mlp_layernorm {
            post_mlp_norm.forward(&xs)?
        } else {
            xs
        };

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

        // Sandwich norm: post-attn norm, then optional pre-MLP norm
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = if let Some(ref pre_mlp_norm) = self.pre_mlp_layernorm {
            pre_mlp_norm.forward(&xs)?
        } else {
            xs
        };

        let xs = self.ffn.forward(&xs, tp_ctx)?;

        let xs = if let Some(ref post_mlp_norm) = self.post_mlp_layernorm {
            post_mlp_norm.forward(&xs)?
        } else {
            xs
        };

        residual + xs
    }
}

// ─── Model Base ─────────────────────────────────────────────────────────────

/// Shared model infrastructure for all Pangu variants.
struct PanguModel {
    embed_tokens: TpEmbedding,
    layers: Vec<PanguDecoderLayer>,
    norm: RmsNorm,
}

impl PanguModel {
    fn new_with_tp(
        cfg: &ModelConfig,
        pangu_cfg: &PanguConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let vb_m = vb;

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(PanguDecoderLayer::new_with_tp(
                cfg,
                pangu_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

// ─── PanguEmbeddedForCausalLM (dense variant) ──────────────────────────────

pub struct PanguEmbeddedForCausalLM {
    model: PanguModel,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl PanguEmbeddedForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let pangu_cfg = PanguConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let model = PanguModel::new_with_tp(cfg, &pangu_cfg, vb_m.clone(), pg)?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = model
                .embed_tokens
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
            model,
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

        let mut xs = self.model.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
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
        let xs = self.model.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for PanguEmbeddedForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        PanguEmbeddedForCausalLM::forward(
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
        let mut xs = self.model.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.model.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── PanguProMoEV2ForCausalLM (MoE variant) ────────────────────────────────

pub struct PanguProMoEV2ForCausalLM {
    model: PanguModel,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl PanguProMoEV2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let pangu_cfg = PanguConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let model = PanguModel::new_with_tp(cfg, &pangu_cfg, vb_m.clone(), pg)?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = model
                .embed_tokens
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
            model,
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

        let mut xs = self.model.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
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
        let xs = self.model.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for PanguProMoEV2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        PanguProMoEV2ForCausalLM::forward(
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
        let mut xs = self.model.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.model.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── PanguUltraMoEForCausalLM (MoE variant) ────────────────────────────────

pub struct PanguUltraMoEForCausalLM {
    model: PanguModel,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl PanguUltraMoEForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let pangu_cfg = PanguConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let model = PanguModel::new_with_tp(cfg, &pangu_cfg, vb_m.clone(), pg)?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = model
                .embed_tokens
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
            model,
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

        let mut xs = self.model.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
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
        let xs = self.model.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for PanguUltraMoEForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        PanguUltraMoEForCausalLM::forward(
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
        let mut xs = self.model.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.model.norm.forward(&xs)?;
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

    fn dense_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["PanguEmbeddedForCausalLM".to_string()],
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
            extra: serde_json::Map::new(),
        }
    }

    fn moe_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("n_routed_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("n_shared_experts".to_string(), serde_json::json!(1));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(0));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["PanguProMoEV2ForCausalLM".to_string()],
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
    fn test_pangu_config_defaults() {
        let cfg = dense_config();
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert!(pangu_cfg.n_routed_experts.is_none());
        assert_eq!(pangu_cfg.num_experts_per_tok, 2);
        assert_eq!(pangu_cfg.first_k_dense_replace, cfg.num_hidden_layers);
        assert!(!pangu_cfg.sandwich_norm);
        assert!(!pangu_cfg.mlp_bias);
        assert_eq!(pangu_cfg.routed_scaling_factor, 1.0);
    }

    #[test]
    fn test_pangu_config_moe() {
        let cfg = moe_config();
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert_eq!(pangu_cfg.n_routed_experts, Some(4));
        assert_eq!(pangu_cfg.num_experts_per_tok, 2);
        assert_eq!(pangu_cfg.moe_intermediate_size, 64);
        assert_eq!(pangu_cfg.n_shared_experts, Some(1));
        assert_eq!(pangu_cfg.first_k_dense_replace, 0);
        assert!(pangu_cfg.is_moe_layer(0));
        assert!(pangu_cfg.is_moe_layer(1));
    }

    #[test]
    fn test_pangu_config_dense_no_moe_layers() {
        let cfg = dense_config();
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert!(!pangu_cfg.is_moe_layer(0));
        assert!(!pangu_cfg.is_moe_layer(1));
    }

    #[test]
    fn test_pangu_config_partial_dense_replace() {
        let mut extra = serde_json::Map::new();
        extra.insert("n_routed_experts".to_string(), serde_json::json!(4));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(1));
        let mut cfg = dense_config();
        cfg.extra = extra;
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert!(!pangu_cfg.is_moe_layer(0));
        assert!(pangu_cfg.is_moe_layer(1));
    }

    #[test]
    fn test_pangu_config_sandwich_norm() {
        let mut cfg = dense_config();
        cfg.extra
            .insert("sandwich_norm".to_string(), serde_json::json!(true));
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert!(pangu_cfg.sandwich_norm);
    }

    // ─── Dense Model Construction Tests ─────────────────────────────────────────

    #[test]
    fn test_pangu_embedded_construction() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = PanguEmbeddedForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PanguEmbeddedForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_pangu_embedded_forward_shape() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguEmbeddedForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_pangu_embedded_prefill_then_decode() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguEmbeddedForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_pangu_embedded_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguEmbeddedForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_pangu_embedded_untied_embeddings() {
        let mut cfg = dense_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguEmbeddedForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.model.layers.len(), cfg.num_hidden_layers);
    }

    // ─── MoE Model Construction Tests ───────────────────────────────────────────

    #[test]
    fn test_pangu_pro_moe_v2_construction() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = PanguProMoEV2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PanguProMoEV2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_pangu_pro_moe_v2_forward_shape() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguProMoEV2ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_pangu_ultra_moe_construction() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = PanguUltraMoEForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PanguUltraMoEForCausalLM should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_pangu_ultra_moe_forward_shape() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguUltraMoEForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_pangu_moe_mixed_layers() {
        let mut extra = serde_json::Map::new();
        extra.insert("n_routed_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(1));

        let mut cfg = dense_config();
        cfg.extra = extra;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguProMoEV2ForCausalLM::new(&cfg, vb).expect("build model");

        // Layer 0 = dense (before first_k_dense_replace), Layer 1 = MoE
        assert!(matches!(model.model.layers[0].ffn, FfnVariant::Dense(_)));
        assert!(matches!(model.model.layers[1].ffn, FfnVariant::MoE(_)));
    }

    #[test]
    fn test_pangu_pro_moe_v2_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguProMoEV2ForCausalLM::new(&cfg, vb).expect("build model");

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

    // ─── Sandwich Norm Tests ────────────────────────────────────────────────────

    #[test]
    fn test_pangu_sandwich_norm_construction() {
        let mut cfg = dense_config();
        cfg.extra
            .insert("sandwich_norm".to_string(), serde_json::json!(true));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = PanguEmbeddedForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PanguEmbedded with sandwich_norm should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.model.layers[0].pre_mlp_layernorm.is_some());
        assert!(model.model.layers[0].post_mlp_layernorm.is_some());
    }

    #[test]
    fn test_pangu_sandwich_norm_forward() {
        let mut cfg = dense_config();
        cfg.extra
            .insert("sandwich_norm".to_string(), serde_json::json!(true));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PanguEmbeddedForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    // ─── MoE Block Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_pangu_moe_block_forward() {
        let cfg = moe_config();
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = PanguMoE::new(&cfg, &pangu_cfg, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_pangu_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let mlp = PanguMLP::new_with_tp(64, 128, false, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }

    #[test]
    fn test_pangu_moe_with_shared_experts() {
        let cfg = moe_config();
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert!(pangu_cfg.n_shared_experts.is_some());

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let moe = PanguMoE::new(&cfg, &pangu_cfg, vb.pp("moe"), &pg).unwrap();
        assert!(moe.shared_expert.is_some());
    }

    #[test]
    fn test_pangu_moe_without_shared_experts() {
        let mut extra = serde_json::Map::new();
        extra.insert("n_routed_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(0));

        let mut cfg = dense_config();
        cfg.extra = extra;
        let pangu_cfg = PanguConfig::from_model_config(&cfg);
        assert!(pangu_cfg.n_shared_experts.is_none());

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let moe = PanguMoE::new(&cfg, &pangu_cfg, vb.pp("moe"), &pg).unwrap();
        assert!(moe.shared_expert.is_none());
    }
}
