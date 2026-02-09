//! Mixtral model implementation with tensor parallelism support.
//!
//! Mixtral is a Mixture of Experts (MoE) model based on Mistral architecture.
//! It uses 8 experts with top-2 routing, providing more capacity while
//! keeping inference cost similar to Mistral 7B.
//!
//! ## Parallelism Modes
//!
//! This implementation supports multiple parallelism strategies:
//!
//! - **Tensor Parallelism (TP)**: Standard TP with column-parallel Q/K/V and row-parallel O.
//!   MoE experts are replicated across GPUs.
//!
//! - **Expert Parallelism (EP)**: Experts distributed across GPUs. Each GPU stores
//!   `num_experts / ep_size` experts. Uses all-to-all communication for token dispatch.
//!   More memory-efficient than replication for large expert counts.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{DeviceCommunicator, EPContext, LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{EPMoEConfig, EPMoELayer, ExpertPlacement, MoELayer, MoELayerConfig};

// Re-export for public API
pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── MoE Configuration ───────────────────────────────────────────────────────

/// Extended model config for Mixtral MoE parameters.
pub struct MixtralConfig {
    /// Base model configuration.
    pub base: ModelConfig,
    /// Number of experts in the MoE layer.
    pub num_local_experts: usize,
    /// Number of experts activated per token.
    pub num_experts_per_tok: usize,
}

impl MixtralConfig {
    /// Create a MixtralConfig from a base ModelConfig.
    ///
    /// Extracts MoE-specific parameters from the extra field, with defaults
    /// for standard Mixtral 8x7B configuration.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        Self {
            base: cfg.clone(),
            num_local_experts,
            num_experts_per_tok,
        }
    }
}

// ─── Attention (Non-TP) ──────────────────────────────────────────────────────

struct MixtralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl MixtralAttention {
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
            sliding_window: cfg.sliding_window,
        })
    }

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

        attn_output.apply(&self.o_proj)
    }

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

            attn_output.apply(&self.o_proj)?.unsqueeze(1)
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
            attn_output.apply(&self.o_proj)
        }
    }
}

// ─── TP Attention ────────────────────────────────────────────────────────────

struct MixtralTpAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl MixtralTpAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        // For TP: num_heads and num_kv_heads must be divisible by world_size
        if world_size > 1 {
            if num_heads % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
                )));
            }
            if num_kv_heads % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})"
                )));
            }
        }

        // Q/K/V are column-parallel (split output heads)
        // O is row-parallel (reduce partial outputs)
        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false, // no bias
            false, // no gather (goes to local attention)
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
            true, // input is parallel (from local attention)
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
            sliding_window: cfg.sliding_window,
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

        // Batched Q/K/V projections: [batch, 1, hidden] -> [batch, 1, proj_dim]
        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        // Reshape to [batch, heads, 1, head_dim]
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
            // Squeeze seq_len=1 dim: [batch, heads, 1, head_dim] -> [batch, heads, head_dim]
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            // Batched RoPE with per-sequence positions
            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

            // Write new K/V to cache
            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            // Build block_tables: [num_seqs, max_blocks_per_seq] u32
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

            // Build seq_lens: total KV length per sequence (including new token)
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

            // [batch, hidden] -> [batch, 1, hidden] to match residual shape
            self.o_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            // Per-sequence: RoPE (position-dependent) + cache write/read + attention
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
                    None, // no causal mask for single-token decode
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

// ─── TP MoE Layer ────────────────────────────────────────────────────────────

/// MoE execution mode.
enum MoEMode {
    /// Experts replicated on all GPUs (default).
    Replicated(MoELayer),
    /// Experts distributed across GPUs using Expert Parallelism.
    ExpertParallel(Box<EPMoELayer>),
}

/// Tensor-parallel MoE layer with optional Expert Parallelism.
///
/// Supports two modes:
/// 1. **Replicated** (EP=1): All experts replicated on all GPUs. Simple but
///    memory-inefficient for large expert counts.
/// 2. **Expert Parallel** (EP>1): Experts distributed across GPUs. Each GPU
///    stores `num_experts / ep_size` experts. Uses all-to-all communication.
struct MixtralTpMoE {
    mode: MoEMode,
}

impl MixtralTpMoE {
    /// Create MoE layer with replicated experts (no EP).
    fn new(cfg: &MixtralConfig, vb: VarBuilder, _pg: &dyn ProcessGroup) -> Result<Self> {
        let moe_config = MoELayerConfig {
            hidden_size: cfg.base.hidden_size,
            intermediate_size: cfg.base.intermediate_size,
            num_experts: cfg.num_local_experts,
            top_k: cfg.num_experts_per_tok,
            renormalize: true,
            inplace: false,
            is_act_and_mul: true,
        };
        let moe = MoELayer::new(moe_config, vb)?;

        Ok(Self {
            mode: MoEMode::Replicated(moe),
        })
    }

    /// Create MoE layer with Expert Parallelism.
    ///
    /// # Arguments
    /// * `cfg` - Mixtral configuration
    /// * `vb` - Variable builder for weight loading
    /// * `ep_ctx` - Expert Parallelism context (rank, size)
    /// * `comm` - Device communicator for all-to-all operations
    fn new_with_ep(
        cfg: &MixtralConfig,
        vb: VarBuilder,
        ep_ctx: &EPContext,
        comm: Arc<dyn DeviceCommunicator>,
    ) -> Result<Self> {
        if ep_ctx.ep_size == 1 {
            // Fallback to replicated mode
            let moe_config = MoELayerConfig {
                hidden_size: cfg.base.hidden_size,
                intermediate_size: cfg.base.intermediate_size,
                num_experts: cfg.num_local_experts,
                top_k: cfg.num_experts_per_tok,
                renormalize: true,
                inplace: false,
                is_act_and_mul: true,
            };
            let moe = MoELayer::new(moe_config, vb)?;
            return Ok(Self {
                mode: MoEMode::Replicated(moe),
            });
        }

        // Expert Parallelism mode
        let ep_config = EPMoEConfig {
            hidden_size: cfg.base.hidden_size,
            intermediate_size: cfg.base.intermediate_size,
            num_experts: cfg.num_local_experts,
            top_k: cfg.num_experts_per_tok,
            ep_size: ep_ctx.ep_size,
            ep_rank: ep_ctx.ep_rank,
            placement: ExpertPlacement::Linear,
            renormalize: true,
        };
        let ep_moe = EPMoELayer::new(ep_config, vb, comm)?;

        Ok(Self {
            mode: MoEMode::ExpertParallel(Box::new(ep_moe)),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match &self.mode {
            MoEMode::Replicated(moe) => moe.forward(hidden_states),
            MoEMode::ExpertParallel(ep_moe) => ep_moe.forward(hidden_states),
        }
    }

    /// Check if using Expert Parallelism.
    #[allow(dead_code)]
    fn uses_expert_parallelism(&self) -> bool {
        matches!(self.mode, MoEMode::ExpertParallel(_))
    }
}

// ─── Decoder Layer (Non-TP) ──────────────────────────────────────────────────

struct MixtralDecoderLayer {
    self_attn: MixtralAttention,
    block_sparse_moe: MoELayer,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MixtralDecoderLayer {
    fn new(cfg: &MixtralConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = MixtralAttention::new(&cfg.base, vb.pp("self_attn"))?;

        let moe_config = MoELayerConfig {
            hidden_size: cfg.base.hidden_size,
            intermediate_size: cfg.base.intermediate_size,
            num_experts: cfg.num_local_experts,
            top_k: cfg.num_experts_per_tok,
            renormalize: true,
            inplace: false,
            is_act_and_mul: true,
        };
        let block_sparse_moe = MoELayer::new(moe_config, vb.pp("block_sparse_moe"))?;

        let input_layernorm = rms_norm(
            cfg.base.hidden_size,
            cfg.base.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.base.hidden_size,
            cfg.base.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe,
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
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.block_sparse_moe.forward(&xs)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.block_sparse_moe.forward(&xs)?;
        residual + xs
    }
}

// ─── TP Decoder Layer ────────────────────────────────────────────────────────

struct MixtralTpDecoderLayer {
    self_attn: MixtralTpAttention,
    block_sparse_moe: MixtralTpMoE,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MixtralTpDecoderLayer {
    fn new_with_tp(cfg: &MixtralConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let self_attn = MixtralTpAttention::new_with_tp(&cfg.base, vb.pp("self_attn"), pg)?;
        let block_sparse_moe = MixtralTpMoE::new(cfg, vb.pp("block_sparse_moe"), pg)?;

        let input_layernorm = rms_norm(
            cfg.base.hidden_size,
            cfg.base.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.base.hidden_size,
            cfg.base.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Create decoder layer with Expert Parallelism for MoE.
    fn new_with_ep(
        cfg: &MixtralConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        ep_ctx: &EPContext,
        comm: Arc<dyn DeviceCommunicator>,
    ) -> Result<Self> {
        let self_attn = MixtralTpAttention::new_with_tp(&cfg.base, vb.pp("self_attn"), pg)?;
        let block_sparse_moe =
            MixtralTpMoE::new_with_ep(cfg, vb.pp("block_sparse_moe"), ep_ctx, comm)?;

        let input_layernorm = rms_norm(
            cfg.base.hidden_size,
            cfg.base.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.base.hidden_size,
            cfg.base.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe,
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
        let xs = self.block_sparse_moe.forward(&xs)?;
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
        let xs = self.block_sparse_moe.forward(&xs)?;
        residual + xs
    }
}

// ─── Model (Non-TP) ──────────────────────────────────────────────────────────

/// Mixtral MoE model for causal language modeling.
///
/// Uses 8 experts with top-2 routing per token.
pub struct MixtralForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<MixtralDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl MixtralForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mixtral_cfg = MixtralConfig::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MixtralDecoderLayer::new(&mixtral_cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            num_experts: mixtral_cfg.num_local_experts,
            num_experts_per_tok: mixtral_cfg.num_experts_per_tok,
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

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the number of experts in this model.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the number of experts activated per token.
    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }
}

impl crate::engine::ModelForward for MixtralForCausalLM {
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
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;
        Ok(logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── TP Model ────────────────────────────────────────────────────────────────

/// Mixtral MoE model with tensor parallelism support.
///
/// This variant uses tensor parallelism for attention layers and keeps
/// MoE experts replicated across GPUs.
pub struct MixtralTpForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<MixtralTpDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl MixtralTpForCausalLM {
    /// Create a new Mixtral model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Mixtral model with tensor parallelism.
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
        let mixtral_cfg = MixtralConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MixtralTpDecoderLayer::new_with_tp(
                &mixtral_cfg,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // LM head: output projection to vocabulary
        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            // Single GPU with tied embeddings: reuse embedding weights
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            // TP with tied embeddings: load from embed_tokens path
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true, // gather output to get full vocab logits
                vb_m.pp("embed_tokens"),
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
            num_experts: mixtral_cfg.num_local_experts,
            num_experts_per_tok: mixtral_cfg.num_experts_per_tok,
        })
    }

    /// Create a new Mixtral model with Expert Parallelism.
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
        let mixtral_cfg = MixtralConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MixtralTpDecoderLayer::new_with_ep(
                &mixtral_cfg,
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
            num_experts: mixtral_cfg.num_local_experts,
            num_experts_per_tok: mixtral_cfg.num_experts_per_tok,
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

    /// Get the number of experts in this model.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the number of experts activated per token.
    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }
}

impl crate::engine::ModelForward for MixtralTpForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        MixtralTpForCausalLM::forward(
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

    fn test_config() -> crate::config::ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".to_string(), serde_json::json!(4)); // Use 4 for tests
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));

        crate::config::ModelConfig {
            architectures: vec!["MixtralForCausalLM".to_string()],
            hidden_size: 32, // Small for tests
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 8,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: Some(256),
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
    fn test_mixtral_config_extraction() {
        let cfg = test_config();
        let mixtral_cfg = MixtralConfig::from_model_config(&cfg);

        assert_eq!(mixtral_cfg.num_local_experts, 4);
        assert_eq!(mixtral_cfg.num_experts_per_tok, 2);
    }

    #[test]
    fn test_mixtral_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MixtralForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MixtralForCausalLM should construct with zero weights"
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.num_experts(), 4);
        assert_eq!(model.num_experts_per_tok(), 2);
    }

    #[test]
    fn test_mixtral_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MixtralForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mixtral_moe_routing() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MixtralForCausalLM::new(&cfg, vb).expect("build model");

        // Model should route to 2 out of 4 experts per token
        assert_eq!(model.num_experts(), 4);
        assert_eq!(model.num_experts_per_tok(), 2);

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
        assert!(result.is_ok(), "MoE forward should work");
    }

    #[test]
    fn test_mixtral_default_config() {
        // Test that missing MoE config defaults to 8x7B settings
        let cfg = crate::config::ModelConfig {
            architectures: vec!["MixtralForCausalLM".to_string()],
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 64,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 8,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra: serde_json::Map::new(), // No MoE config specified
        };

        let mixtral_cfg = MixtralConfig::from_model_config(&cfg);

        // Should default to 8 experts, top-2
        assert_eq!(mixtral_cfg.num_local_experts, 8);
        assert_eq!(mixtral_cfg.num_experts_per_tok, 2);
    }

    #[test]
    fn test_mixtral_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MixtralForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 3 tokens
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

        // Decode step
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

    // ─── Tensor Parallelism Tests ────────────────────────────────────────────────

    fn test_config_tp_compatible() -> crate::config::ModelConfig {
        // Config with heads divisible by 2 for TP=2 testing
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));

        crate::config::ModelConfig {
            architectures: vec!["MixtralForCausalLM".to_string()],
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
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_mixtral_tp_construction_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Simulate TP with world_size=2
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = MixtralTpForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "MixtralTpForCausalLM should construct with TP=2: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.num_experts(), 4);
        assert_eq!(model.num_experts_per_tok(), 2);
    }

    #[test]
    fn test_mixtral_tp_forward_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Create model with TP=2 simulation
        let tp_size = 2;
        let pg = LocalProcessGroup::with_rank(0, tp_size);
        let tp_ctx = TpContext::mock_multi_gpu(0, tp_size);
        let model = MixtralTpForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

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
    fn test_mixtral_tp_heads_divisibility_check() {
        // Test that TP fails with error when heads aren't divisible
        let mut cfg = test_config_tp_compatible();
        cfg.num_key_value_heads = 3; // Not divisible by 2

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Try TP=2 with 3 kv_heads - should return error
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let result = MixtralTpForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);

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
    fn test_mixtral_tp_moe_forward() {
        // Test that MoE layers work correctly in TP mode
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let tp_size = 2;
        let pg = LocalProcessGroup::with_rank(0, tp_size);
        let tp_ctx = TpContext::mock_multi_gpu(0, tp_size);
        let model = MixtralTpForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

        // Verify MoE configuration
        assert_eq!(model.num_experts(), 4);
        assert_eq!(model.num_experts_per_tok(), 2);

        let local_kv_heads = cfg.num_key_value_heads / tp_size;
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: local_kv_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
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
        assert!(
            result.is_ok(),
            "TP MoE forward should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_mixtral_tp_single_gpu_fallback() {
        // Test that single GPU mode works (TP=1)
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MixtralTpForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MixtralTpForCausalLM should work with single GPU: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.tp_context().is_single());
    }

    #[test]
    fn test_mixtral_tp_prefill_then_decode() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let tp_size = 2;
        let pg = LocalProcessGroup::with_rank(0, tp_size);
        let tp_ctx = TpContext::mock_multi_gpu(0, tp_size);
        let model = MixtralTpForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

        let local_kv_heads = cfg.num_key_value_heads / tp_size;
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: local_kv_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 3 tokens
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

        // Decode step
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
}
