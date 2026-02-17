//! Qwen2-MoE model implementation with Tensor Parallelism and Expert Parallelism support.
//!
//! Qwen2-MoE is a Mixture of Experts variant of Qwen2 with:
//! - Sparse MoE blocks on certain layers (controlled by `decoder_sparse_step`)
//! - Shared expert that processes all tokens
//! - Sigmoid-gated shared expert output
//! - Dense MLP layers interspersed with MoE layers
//!
//! ## Parallelism Modes
//!
//! This implementation supports multiple parallelism strategies:
//!
//! - **Tensor Parallelism (TP)**: Standard TP with column-parallel Q/K/V and row-parallel O.
//!   MoE experts are replicated across GPUs.
//!
//! - **Expert Parallelism (EP)**: Routed experts distributed across GPUs. Each GPU stores
//!   `num_experts / ep_size` experts. Uses all-to-all communication for token dispatch.
//!   Shared expert remains replicated on all GPUs.

use std::sync::Arc;

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{DeviceCommunicator, EPContext, LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding, SwiGluMlp};
use crate::moe::{
    EPMoEConfig, EPMoELayer, ExpertPlacement, MoEExpert, MoEExpertConfig, MoERouter, RouterConfig,
    ScoringFunc, TopKRouter,
};

// Re-export for public API
pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Attention ───────────────────────────────────────────────────────────────

struct Qwen2MoeAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2MoeAttention {
    #[allow(dead_code)]
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new())
    }

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

        // Qwen2-MoE uses bias=True for QKV projections (same as Qwen2)
        let use_bias = cfg.attention_bias.unwrap_or(true);

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
            false, // o_proj has no bias in Qwen2-MoE
            true,  // input is parallel (from local attention)
            vb.pp("o_proj"),
            pg,
        )?;

        // For TP: each GPU handles num_heads/world_size heads
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

// ─── Sparse MoE Block ────────────────────────────────────────────────────────

/// Routed experts mode (replicated or EP).
enum RoutedExpertsMode {
    /// All experts replicated on all GPUs.
    Replicated {
        router: TopKRouter,
        experts: Vec<MoEExpert>,
        num_experts: usize,
    },
    /// Experts distributed using Expert Parallelism.
    ExpertParallel(Box<EPMoELayer>),
}

/// Sparse MoE block with shared expert.
///
/// This implements the MoE layer used in Qwen2-MoE which combines:
/// - Routed experts selected via top-k routing (replicated or EP)
/// - A shared expert that processes all tokens (always replicated)
///
/// ## Parallelism Modes
///
/// - **Replicated**: All experts on all GPUs. Simple but memory-inefficient.
/// - **Expert Parallel (EP)**: Routed experts distributed across GPUs. Shared expert
///   remains replicated since it processes all tokens.
struct Qwen2MoeSparseMoeBlock {
    routed: RoutedExpertsMode,
    shared_expert: MoEExpert,
    shared_expert_gate: Linear,
    #[allow(dead_code)]
    top_k: usize,
}

impl Qwen2MoeSparseMoeBlock {
    /// Create MoE block with replicated experts (no EP).
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts().unwrap_or(60);
        let top_k = cfg.num_experts_per_tok().unwrap_or(4);
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);
        let shared_expert_intermediate_size = cfg
            .shared_expert_intermediate_size()
            .unwrap_or(cfg.intermediate_size);
        let renormalize = cfg.norm_topk_prob();

        // Router: hidden_size -> num_experts
        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize,
            scoring_func: ScoringFunc::Softmax,
            ..Default::default()
        };
        let router = TopKRouter::new(router_config, vb.pp("gate"))?;

        // Routed experts (all replicated)
        let expert_config = MoEExpertConfig {
            hidden_size,
            intermediate_size: moe_intermediate_size,
        };
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(MoEExpert::new(&expert_config, vb_experts.pp(i))?);
        }

        // Shared expert (always replicated)
        let shared_expert_config = MoEExpertConfig {
            hidden_size,
            intermediate_size: shared_expert_intermediate_size,
        };
        let shared_expert = MoEExpert::new(&shared_expert_config, vb.pp("shared_expert"))?;

        // Shared expert gate: hidden_size -> 1 (sigmoid gating)
        let shared_expert_gate = linear_no_bias(hidden_size, 1, vb.pp("shared_expert_gate"))?;

        Ok(Self {
            routed: RoutedExpertsMode::Replicated {
                router,
                experts,
                num_experts,
            },
            shared_expert,
            shared_expert_gate,
            top_k,
        })
    }

    /// Create MoE block with Expert Parallelism for routed experts.
    ///
    /// Shared expert remains replicated on all GPUs.
    fn new_with_ep(
        cfg: &ModelConfig,
        vb: VarBuilder,
        ep_ctx: &EPContext,
        comm: Arc<dyn DeviceCommunicator>,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts().unwrap_or(60);
        let top_k = cfg.num_experts_per_tok().unwrap_or(4);
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);
        let shared_expert_intermediate_size = cfg
            .shared_expert_intermediate_size()
            .unwrap_or(cfg.intermediate_size);
        let renormalize = cfg.norm_topk_prob();

        // Shared expert (always replicated)
        let shared_expert_config = MoEExpertConfig {
            hidden_size,
            intermediate_size: shared_expert_intermediate_size,
        };
        let shared_expert = MoEExpert::new(&shared_expert_config, vb.pp("shared_expert"))?;

        // Shared expert gate
        let shared_expert_gate = linear_no_bias(hidden_size, 1, vb.pp("shared_expert_gate"))?;

        // Routed experts with EP
        let routed = if ep_ctx.ep_size == 1 {
            // Fallback to replicated mode
            let router_config = RouterConfig {
                hidden_size,
                num_experts,
                top_k,
                renormalize,
                scoring_func: ScoringFunc::Softmax,
                ..Default::default()
            };
            let router = TopKRouter::new(router_config, vb.pp("gate"))?;

            let expert_config = MoEExpertConfig {
                hidden_size,
                intermediate_size: moe_intermediate_size,
            };
            let mut experts = Vec::with_capacity(num_experts);
            let vb_experts = vb.pp("experts");
            for i in 0..num_experts {
                experts.push(MoEExpert::new(&expert_config, vb_experts.pp(i))?);
            }

            RoutedExpertsMode::Replicated {
                router,
                experts,
                num_experts,
            }
        } else {
            // Expert Parallelism mode
            let ep_config = EPMoEConfig {
                hidden_size,
                intermediate_size: moe_intermediate_size,
                num_experts,
                top_k,
                ep_size: ep_ctx.ep_size,
                ep_rank: ep_ctx.ep_rank,
                placement: ExpertPlacement::Linear,
                renormalize,
            };
            let ep_layer = EPMoELayer::new(ep_config, vb.clone(), comm)?;
            RoutedExpertsMode::ExpertParallel(Box::new(ep_layer))
        };

        Ok(Self {
            routed,
            shared_expert,
            shared_expert_gate,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;

        // Compute routed expert output
        let routed_output = match &self.routed {
            RoutedExpertsMode::Replicated {
                router,
                experts,
                num_experts,
            } => {
                let num_tokens = xs_2d.dim(0)?;
                let (routing_weights, selected_experts) = router.route(&xs_2d)?;
                let mut final_output =
                    Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

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
                        if expert_idx < *num_experts {
                            let expert_out = experts[expert_idx].forward(&token_input)?;
                            let weight = weights[k];
                            let weighted = expert_out.affine(weight as f64, 0.0)?;
                            token_output = (token_output + weighted)?;
                        }
                    }

                    let indices = Tensor::new(&[token_idx as u32], xs.device())?;
                    final_output = final_output.index_add(&indices, &token_output, 0)?;
                }
                final_output
            }
            RoutedExpertsMode::ExpertParallel(ep_layer) => ep_layer.forward(&xs_2d)?,
        };

        // Compute shared expert output with sigmoid gating
        let shared_out = self.shared_expert.forward(&xs_2d)?;
        let gate_logits = self.shared_expert_gate.forward(&xs_2d)?;
        let gate = candle_nn::ops::sigmoid(&gate_logits)?;
        let gated_shared = shared_out.broadcast_mul(&gate)?;

        // Combine routed and shared outputs
        let output = (routed_output + gated_shared)?;
        output.reshape(orig_shape)
    }
}

// ─── MLP Variant ─────────────────────────────────────────────────────────────

/// MLP variant that can be either dense or sparse MoE.
enum MlpVariant {
    Dense(SwiGluMlp),
    Sparse(Box<Qwen2MoeSparseMoeBlock>),
}

impl MlpVariant {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            MlpVariant::Dense(mlp) => xs.apply(mlp),
            MlpVariant::Sparse(moe) => moe.forward(xs),
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct Qwen2MoeDecoderLayer {
    self_attn: Qwen2MoeAttention,
    mlp: MlpVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2MoeDecoderLayer {
    #[allow(dead_code)]
    fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, layer_idx, vb, &LocalProcessGroup::new())
    }

    fn new_with_tp(
        cfg: &ModelConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Qwen2MoeAttention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;

        // Determine if this layer should be MoE or dense
        let decoder_sparse_step = cfg.decoder_sparse_step().unwrap_or(1);
        let mlp_only_layers = cfg.mlp_only_layers();
        let num_experts = cfg.num_experts().unwrap_or(0);

        let is_moe_layer = !mlp_only_layers.contains(&layer_idx)
            && num_experts > 0
            && (layer_idx + 1).is_multiple_of(decoder_sparse_step);

        // NOTE: MoE block remains replicated across all GPUs (no expert parallelism)
        // Dense MLP also remains replicated for simplicity
        let mlp = if is_moe_layer {
            MlpVariant::Sparse(Box::new(Qwen2MoeSparseMoeBlock::new(cfg, vb.pp("mlp"))?))
        } else {
            MlpVariant::Dense(SwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
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

    /// Create decoder layer with Expert Parallelism for MoE blocks.
    fn new_with_ep(
        cfg: &ModelConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        ep_ctx: &EPContext,
        comm: Arc<dyn DeviceCommunicator>,
    ) -> Result<Self> {
        let self_attn = Qwen2MoeAttention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;

        let decoder_sparse_step = cfg.decoder_sparse_step().unwrap_or(1);
        let mlp_only_layers = cfg.mlp_only_layers();
        let num_experts = cfg.num_experts().unwrap_or(0);

        let is_moe_layer = !mlp_only_layers.contains(&layer_idx)
            && num_experts > 0
            && (layer_idx + 1).is_multiple_of(decoder_sparse_step);

        let mlp = if is_moe_layer {
            MlpVariant::Sparse(Box::new(Qwen2MoeSparseMoeBlock::new_with_ep(
                cfg,
                vb.pp("mlp"),
                ep_ctx,
                comm,
            )?))
        } else {
            MlpVariant::Dense(SwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
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
        let xs = self.mlp.forward(&xs)?;
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
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Qwen2-MoE model for causal language modeling.
///
/// This is a Mixture of Experts variant of Qwen2 with:
/// - Sparse MoE layers (controlled by `decoder_sparse_step`)
/// - Shared expert with sigmoid gating
/// - Dense MLP layers interspersed with MoE layers
///
/// ## Tensor Parallelism
///
/// The model supports tensor parallelism via `new_with_tp()`:
/// - Attention Q/K/V/O projections are sharded across GPUs
/// - Embeddings use vocab-parallel distribution
/// - MoE layers remain replicated (expert parallelism not implemented)
pub struct Qwen2MoeForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Qwen2MoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Qwen2MoeForCausalLM {
    /// Create a new Qwen2-MoE model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Qwen2-MoE model with tensor parallelism.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for tensor parallelism
    /// * `tp_ctx` - Tensor parallelism context (holds communicator)
    ///
    /// # Errors
    /// Returns an error if:
    /// - `num_attention_heads` is not divisible by `world_size`
    /// - `num_key_value_heads` is not divisible by `world_size`
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
            layers.push(Qwen2MoeDecoderLayer::new_with_tp(cfg, i, vb_l.pp(i), pg)?);
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

    /// Create a new Qwen2-MoE model with Expert Parallelism.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for tensor parallelism (attention layers)
    /// * `tp_ctx` - Tensor parallelism context
    /// * `ep_ctx` - Expert Parallelism context
    /// * `ep_comm` - Communicator for EP all-to-all operations
    pub fn new_with_ep(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
        ep_ctx: EPContext,
        ep_comm: Arc<dyn DeviceCommunicator>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen2MoeDecoderLayer::new_with_ep(
                cfg,
                i,
                vb_l.pp(i),
                pg,
                &ep_ctx,
                ep_comm.clone(),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // LM head: output projection to vocabulary
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

impl crate::engine::ModelForward for Qwen2MoeForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Qwen2MoeForCausalLM::forward(
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
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("decoder_sparse_step".to_string(), serde_json::json!(1));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert(
            "shared_expert_intermediate_size".to_string(),
            serde_json::json!(128),
        );
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["Qwen2MoeForCausalLM".to_string()],
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
            rope_theta: 1000000.0,
            tie_word_embeddings: true,
            bos_token_id: 151643,
            eos_token_id: 151645,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    fn test_config_dense_layers() -> ModelConfig {
        let mut cfg = test_config();
        // decoder_sparse_step=2 means only odd-indexed layers (1, 3, ...) are MoE
        cfg.extra
            .insert("decoder_sparse_step".to_string(), serde_json::json!(2));
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
    fn test_qwen2_moe_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen2MoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen2MoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen2_moe_moe_block() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let moe_block = Qwen2MoeSparseMoeBlock::new(&cfg, vb.pp("mlp"));
        assert!(moe_block.is_ok(), "MoE block should construct");

        let moe_block = moe_block.unwrap();
        // top_k is available directly, num_experts is inside routed mode
        assert_eq!(moe_block.top_k, 2);
    }

    #[test]
    fn test_qwen2_moe_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_qwen2_moe_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_qwen2_moe_mixed_layers() {
        // Test with decoder_sparse_step=2: layer 0 dense, layer 1 MoE
        let cfg = test_config_dense_layers();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen2MoeForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "Should construct with mixed layers");
    }

    #[test]
    fn test_qwen2_moe_sparse_block_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let moe_block = Qwen2MoeSparseMoeBlock::new(&cfg, vb.pp("mlp")).expect("moe block");

        let batch_size = 2;
        let seq_len = 3;
        let hidden_size = cfg.hidden_size;
        let xs =
            Tensor::zeros((batch_size, seq_len, hidden_size), DType::F32, &device).expect("input");

        let output = moe_block.forward(&xs);
        assert!(
            output.is_ok(),
            "MoE forward should work: {:?}",
            output.err()
        );

        let output = output.unwrap();
        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, hidden_size],
            "MoE output should preserve shape"
        );
    }

    #[test]
    fn test_qwen2_moe_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2MoeForCausalLM::new(&cfg, vb).expect("build model");

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
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("decoder_sparse_step".to_string(), serde_json::json!(1));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert(
            "shared_expert_intermediate_size".to_string(),
            serde_json::json!(128),
        );
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["Qwen2MoeForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4, // divisible by 2
            num_key_value_heads: 2, // divisible by 2
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            tie_word_embeddings: true,
            bos_token_id: 151643,
            eos_token_id: 151645,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    #[test]
    fn test_qwen2_moe_tp_construction_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Simulate TP with world_size=2
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = Qwen2MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Qwen2MoeForCausalLM should construct with TP=2: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen2_moe_tp_forward_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Create model with TP=2 simulation
        let tp_size = 2;
        let pg = LocalProcessGroup::with_rank(0, tp_size);
        let tp_ctx = TpContext::mock_multi_gpu(0, tp_size);
        let model = Qwen2MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

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
    fn test_qwen2_moe_tp_heads_divisibility_check() {
        // Test that TP fails with error when heads aren't divisible
        let mut cfg = test_config();
        cfg.num_key_value_heads = 3; // Not divisible by 2

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Try TP=2 with 3 kv_heads - should return error
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let result = Qwen2MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);

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
    fn test_qwen2_moe_tp_untied_embeddings() {
        let mut cfg = test_config_tp_compatible();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = Qwen2MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Should construct with untied embeddings in TP mode: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen2_moe_tp_context_access() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);
        let model = Qwen2MoeForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

        let ctx = model.tp_context();
        assert_eq!(ctx.world_size, 2);
        assert_eq!(ctx.rank, 0);
    }

    #[test]
    fn test_qwen2_moe_single_gpu_backward_compat() {
        // Verify single GPU path still works with new TP infrastructure
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen2MoeForCausalLM::new(&cfg, vb).expect("build model");

        // Should have world_size=1
        assert!(model.tp_context().is_single());

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
}
