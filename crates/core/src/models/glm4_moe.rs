//! GLM4-MoE model implementation with Tensor Parallelism support.
//!
//! Glm4MoeForCausalLM features:
//! - SharedFusedMoE with shared experts
//! - first_k_dense_replace: first K layers are dense, rest are MoE
//! - Grouped top-k routing with sigmoid scoring
//! - e_score_correction_bias for routing
//! - routed_scaling_factor for output scaling
//! - Optional use_qk_norm for per-head QK normalization
//! - partial_rotary_factor = 0.5
//!
//! Tensor Parallelism:
//! - Attention: Q/K/V column-parallel, O row-parallel
//! - MoE: Router replicated, experts TP-sharded
//! - Embeddings: Vocab-parallel

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

// Re-export for public API
pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Constants ───────────────────────────────────────────────────────────────

const GLM4_MOE_PARTIAL_ROTARY_FACTOR: f64 = 0.5;
const GLM4_MOE_IS_NEOX_STYLE: bool = false;

// ─── Attention ───────────────────────────────────────────────────────────────

struct Glm4MoeAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Glm4MoeAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        // For TP: num_heads and num_kv_heads must be divisible by world_size
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

        // GLM4-MoE uses bias based on config, defaulting to false
        let use_bias = cfg.attention_bias.unwrap_or(false);

        // Q/K/V are column-parallel (split output heads)
        // O is row-parallel (reduce partial outputs)
        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            use_bias,
            false, // no gather (goes to local attention)
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
            num_kv_heads * head_dim,
            use_bias,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true, // input is parallel (from local attention)
            vb.pp("o_proj"),
            pg,
        )?;

        // For TP: each GPU handles num_heads/world_size heads
        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

        // Optional QK norm
        let use_qk_norm = cfg.use_qk_norm();
        let (q_norm, k_norm) = if use_qk_norm {
            let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
            let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };

        // GLM4-specific: partial rotary, split style
        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            GLM4_MOE_PARTIAL_ROTARY_FACTOR,
            GLM4_MOE_IS_NEOX_STYLE,
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

        // Optional per-head QK norm
        let q = if let Some(ref q_norm) = self.q_norm {
            apply_per_head_norm(&q, q_norm)?
        } else {
            q
        };
        let k = if let Some(ref k_norm) = self.k_norm {
            apply_per_head_norm(&k, k_norm)?
        } else {
            k
        };

        // Apply partial RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Paged attention
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

        let q = if let Some(ref q_norm) = self.q_norm {
            apply_per_head_norm(&q, q_norm)?
        } else {
            q
        };
        let k = if let Some(ref k_norm) = self.k_norm {
            apply_per_head_norm(&k, k_norm)?
        } else {
            k
        };

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

// ─── TP-aware MoE Expert ─────────────────────────────────────────────────────

/// A single MoE expert with tensor parallelism support.
///
/// Uses SwiGLU activation with TP-sharded linear layers:
/// - gate_proj and up_proj are column-parallel (split intermediate dim)
/// - down_proj is row-parallel (reduce partial outputs)
struct TpMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl TpMoEExpert {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // GLM4-MoE uses w1/w3/w2 naming convention
        let gate_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false, // no bias
            false, // no gather (goes to element-wise mul)
            vb.pp("w1"),
            pg,
        )?;
        let up_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("w3"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true, // input is parallel
            vb.pp("w2"),
            pg,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        // SwiGLU: silu(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(xs, tp_ctx)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ─── MoE Block ───────────────────────────────────────────────────────────────

/// GLM4-MoE sparse block with shared experts and tensor parallelism.
///
/// Features:
/// - Grouped top-k routing with sigmoid scoring (replicated)
/// - e_score_correction_bias
/// - Shared experts (processed for all tokens)
/// - routed_scaling_factor for output scaling
/// - TP within each expert's linear layers
struct Glm4MoeBlock {
    router: TopKRouter,
    experts: Vec<TpMoEExpert>,
    shared_expert: Option<TpMoEExpert>,
    num_experts: usize,
    routed_scaling_factor: f64,
}

impl Glm4MoeBlock {
    fn new(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_experts = cfg.num_experts().unwrap_or(8);
        let top_k = cfg.num_experts_per_tok().unwrap_or(2);
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);
        let renormalize = cfg.norm_topk_prob();
        let routed_scaling_factor = cfg.routed_scaling_factor();

        // Grouped top-k configuration
        let n_group = cfg.n_group();
        let topk_group = cfg.topk_group();

        // Router with sigmoid scoring and grouped top-k
        // Router stays replicated across all TP ranks for consistent routing decisions
        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize,
            scoring_func: ScoringFunc::Sigmoid,
            use_grouped_topk: n_group.is_some(),
            num_expert_groups: n_group,
            topk_per_group: topk_group,
            routed_scaling_factor: cfg.routed_scaling_factor(),
        };
        let router = TopKRouter::new(router_config, vb.pp("gate"))?;

        // Routed experts with TP
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(TpMoEExpert::new(
                hidden_size,
                moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        // Shared experts (optional) with TP
        let n_shared_experts = cfg.n_shared_experts();
        let shared_expert = if let Some(n_shared) = n_shared_experts {
            if n_shared > 0 {
                let shared_intermediate_size = moe_intermediate_size * n_shared;
                Some(TpMoEExpert::new(
                    hidden_size,
                    shared_intermediate_size,
                    vb.pp("shared_experts"),
                    pg,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts,
            routed_scaling_factor,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Get routing weights and indices (same on all TP ranks)
        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;

        // Compute routed expert outputs
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
                    let weight = weights[k];
                    let weighted = expert_out.affine(weight as f64, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            routed_output = routed_output.index_add(&indices, &token_output, 0)?;
        }

        // Apply routed scaling factor
        let routed_output = routed_output.affine(self.routed_scaling_factor, 0.0)?;

        // Add shared expert output if present
        let final_output = if let Some(ref shared_expert) = self.shared_expert {
            let shared_out = shared_expert.forward(&xs_2d, tp_ctx)?;
            (routed_output + shared_out)?
        } else {
            routed_output
        };

        final_output.reshape(orig_shape)
    }
}

// ─── MLP Variant ─────────────────────────────────────────────────────────────

enum MlpVariant {
    Dense(TpSwiGluMlp),
    MoE(Glm4MoeBlock),
}

impl MlpVariant {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            MlpVariant::Dense(mlp) => mlp.forward(xs, tp_ctx),
            MlpVariant::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

pub(crate) struct Glm4MoeDecoderLayer {
    self_attn: Glm4MoeAttention,
    mlp: MlpVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Glm4MoeDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Glm4MoeAttention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;

        // first_k_dense_replace: first K layers are dense, rest are MoE
        let first_k_dense = cfg.first_k_dense_replace().unwrap_or(0);
        let num_experts = cfg.num_experts().unwrap_or(0);

        let is_moe_layer = num_experts > 0 && layer_idx >= first_k_dense;

        let mlp = if is_moe_layer {
            MlpVariant::MoE(Glm4MoeBlock::new(cfg, vb.pp("mlp"), pg)?)
        } else {
            MlpVariant::Dense(TpSwiGluMlp::new(
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
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub(crate) fn new_for_mtp(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let pg = LocalProcessGroup::new();
        Self::new_with_tp(cfg, layer_idx, vb, &pg)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
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
        let xs = self.mlp.forward(&xs, tp_ctx)?;
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
        let xs = self.mlp.forward(&xs, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// GLM4-MoE model for causal language modeling with tensor parallelism support.
///
/// Key features:
/// - SharedFusedMoE with shared experts
/// - first_k_dense_replace for dense/MoE layer split
/// - Grouped top-k routing with sigmoid scoring
/// - e_score_correction_bias and routed_scaling_factor
/// - Optional use_qk_norm
/// - partial_rotary_factor = 0.5
///
/// Tensor Parallelism:
/// - Attention: Q/K/V column-parallel, O row-parallel
/// - MoE: Router replicated, experts TP-sharded within each expert
/// - Embeddings: Vocab-parallel, lm_head column-parallel with gather
pub struct Glm4MoeForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Glm4MoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Glm4MoeForCausalLM {
    /// Create a new GLM4-MoE model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new GLM4-MoE model with tensor parallelism.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for tensor parallelism
    /// * `tp_ctx` - Tensor parallelism context (holds communicator)
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Glm4MoeDecoderLayer::new_with_tp(cfg, i, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // LM head: output projection to vocabulary
        //
        // For single GPU with tied embeddings: reuse embedding weights directly
        // For TP: use column-parallel linear that loads from embed_tokens (tied) or lm_head (separate)
        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            // Single GPU with tied embeddings: reuse embedding weights
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            // TP with tied embeddings: load from embed_tokens path
            // The weights are the same as embedding, just used as a linear projection
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,                    // gather output to get full vocab logits
                vb_m.pp("embed_tokens"), // Use embed_tokens weights for tied case
                pg,
            )?
        } else {
            // Separate lm_head (no tied embeddings)
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true, // gather output to get full vocab logits
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
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a reference to the TP context.
    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Glm4MoeForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
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

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_experts".to_string(), serde_json::json!(8));
        extra.insert("n_routed_experts".to_string(), serde_json::json!(8));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(1)); // First layer dense
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("n_shared_experts".to_string(), serde_json::json!(2));
        extra.insert("n_group".to_string(), serde_json::json!(2));
        extra.insert("topk_group".to_string(), serde_json::json!(1));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("use_qk_norm".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["Glm4MoeForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
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
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_config_with_qk_norm() -> ModelConfig {
        let mut cfg = test_config();
        cfg.extra
            .insert("use_qk_norm".to_string(), serde_json::json!(true));
        cfg
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

    #[test]
    fn test_glm4_moe_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Glm4MoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Glm4MoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_glm4_moe_with_qk_norm() {
        let cfg = test_config_with_qk_norm();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Glm4MoeForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "Should construct with QK norm");
    }

    #[test]
    fn test_glm4_moe_block() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let moe_block = Glm4MoeBlock::new(&cfg, vb.pp("mlp"), &pg);
        assert!(moe_block.is_ok(), "MoE block should construct");

        let moe_block = moe_block.unwrap();
        assert_eq!(moe_block.num_experts, 8);
        assert!(moe_block.shared_expert.is_some());
    }

    #[test]
    fn test_glm4_moe_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Glm4MoeForCausalLM::new(&cfg, vb).expect("build model");

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

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_glm4_moe_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Glm4MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_glm4_moe_mixed_layers() {
        // With first_k_dense_replace=1, layer 0 is dense, layer 1 is MoE
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Glm4MoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct with mixed dense/MoE layers"
        );
    }

    #[test]
    fn test_glm4_moe_uses_partial_rotary() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let attn =
            Glm4MoeAttention::new_with_tp(&cfg, vb.pp("self_attn"), &pg).expect("build attention");

        assert!(
            attn.rotary_emb.is_partial(),
            "GLM4-MoE should use partial rotary embedding"
        );
    }

    #[test]
    fn test_glm4_moe_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Glm4MoeForCausalLM::new(&cfg, vb).expect("build model");

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

    // ─── Tensor Parallelism Tests ────────────────────────────────────────────────

    fn test_config_tp_compatible() -> ModelConfig {
        // Config with heads divisible by 2 for TP=2 testing
        let mut extra = serde_json::Map::new();
        extra.insert("num_experts".to_string(), serde_json::json!(8));
        extra.insert("n_routed_experts".to_string(), serde_json::json!(8));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(1));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("n_shared_experts".to_string(), serde_json::json!(2));
        extra.insert("n_group".to_string(), serde_json::json!(2));
        extra.insert("topk_group".to_string(), serde_json::json!(1));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("use_qk_norm".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["Glm4MoeForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4, // divisible by 2
            num_key_value_heads: 2, // divisible by 2
            num_hidden_layers: 2,
            intermediate_size: 128, // divisible by 2
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
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_glm4_moe_tp_construction_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Simulate TP with world_size=2 (ProcessGroup and TpContext must match)
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = Glm4MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Glm4MoeForCausalLM should construct with TP=2: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_glm4_moe_tp_forward_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Create model with TP=2 simulation (ProcessGroup and TpContext must match)
        let tp_size = 2;
        let pg = LocalProcessGroup::with_rank(0, tp_size);
        let tp_ctx = TpContext::mock_multi_gpu(0, tp_size);
        let model = Glm4MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

        // Create cache with LOCAL kv_heads (divided by tp_size)
        let local_kv_heads = cfg.num_key_value_heads / tp_size;
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: local_kv_heads, // Important: local heads, not global
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
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

        // With TP=2 and MockCommunicator's all_gather simulation,
        // the output should be gathered to full vocab_size
        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_glm4_moe_tp_heads_divisibility_check() {
        // Test that TP fails with error when heads aren't divisible
        let mut cfg = test_config();
        cfg.num_key_value_heads = 3; // Not divisible by 2

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Try TP=2 with 3 kv_heads - should return error
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let result = Glm4MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);

        match result {
            Ok(_) => panic!("Should fail when num_kv_heads is not divisible by world_size"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("divisible"),
                    "Error should mention divisibility: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn test_glm4_moe_tp_attention_construction() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Single GPU
        let pg_single = LocalProcessGroup::new();
        let attn_single =
            Glm4MoeAttention::new_with_tp(&cfg, vb.pp("attn"), &pg_single).expect("single GPU");
        assert_eq!(attn_single.num_heads, cfg.num_attention_heads);
        assert_eq!(attn_single.num_kv_heads, cfg.num_key_value_heads);

        // TP=2
        let pg_tp = LocalProcessGroup::with_rank(0, 2);
        let vb2 = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let attn_tp = Glm4MoeAttention::new_with_tp(&cfg, vb2.pp("attn"), &pg_tp).expect("TP=2");
        assert_eq!(
            attn_tp.num_heads,
            cfg.num_attention_heads / 2,
            "TP=2 should have half the heads"
        );
        assert_eq!(
            attn_tp.num_kv_heads,
            cfg.num_key_value_heads / 2,
            "TP=2 should have half the kv_heads"
        );
    }

    #[test]
    fn test_glm4_moe_tp_expert_construction() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Single GPU expert
        let pg_single = LocalProcessGroup::new();
        let expert_single = TpMoEExpert::new(64, 128, vb.pp("expert"), &pg_single);
        assert!(expert_single.is_ok(), "Single GPU expert should construct");

        // TP=2 expert
        let pg_tp = LocalProcessGroup::with_rank(0, 2);
        let vb2 = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let expert_tp = TpMoEExpert::new(64, 128, vb2.pp("expert"), &pg_tp);
        assert!(expert_tp.is_ok(), "TP=2 expert should construct");
    }

    #[test]
    fn test_glm4_moe_tp_context() {
        // Verify TpContext is accessible
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Glm4MoeForCausalLM::new(&cfg, vb).expect("build model");

        let ctx = model.tp_context();
        assert!(
            ctx.is_single(),
            "Single GPU model should have single TpContext"
        );
        assert_eq!(ctx.world_size, 1);
        assert_eq!(ctx.rank, 0);
    }

    #[test]
    fn test_glm4_moe_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Glm4MoeForCausalLM::new(&cfg, vb).expect("build model");

        // Model should construct successfully with separate lm_head
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_glm4_moe_tp_untied_embeddings() {
        let mut cfg = test_config_tp_compatible();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = Glm4MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "TP model with untied embeddings should construct: {:?}",
            model.err()
        );
    }
}
