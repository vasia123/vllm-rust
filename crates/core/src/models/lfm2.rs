//! Lfm2 model implementation (Dense + MoE variants).
//!
//! Lfm2 Dense: hybrid architecture with attention and short-convolution layers.
//! Lfm2 MoE: hybrid with top-k routed experts, shared dense layers, and sigmoid scoring.
//!
//! Both variants share core attention structure:
//! - QKV parallel projection with per-head Q/K RMSNorm (always enabled)
//! - RoPE after QK norm
//! - Output projection via `out_proj` (not `o_proj`)
//! - SwiGLU MLP using `w1` (gate+up merged) and `w2` (down)
//! - Pre-norm with `operator_norm` (input) and `ffn_norm` (post-attention)
//!
//! Key config fields (from extra):
//! - `layer_types`: list of "full_attention" or "short_conv" per layer
//! - `block_dim`: MLP input dimension (default = hidden_size)
//! - `block_ff_dim`: MLP intermediate dimension (default = intermediate_size)
//! - `block_multiple_of`: alignment factor for auto-adjusted ff_dim
//! - `block_auto_adjust_ff_dim`: whether to apply 2/3 adjustment (default false)
//! - `block_ffn_dim_multiplier`: optional multiplier for ff_dim
//! - `norm_eps`: RMSNorm epsilon (default = rms_norm_eps)
//!
//! MoE-specific config fields:
//! - `num_experts`: number of routed experts
//! - `num_experts_per_tok`: top-k experts per token
//! - `moe_intermediate_size`: intermediate size for MoE expert MLPs
//! - `num_dense_layers`: first N layers use dense MLP, rest use MoE
//! - `routed_scaling_factor`: scale factor for routed expert output
//! - `norm_topk_prob`: whether to renormalize top-k probabilities
//! - `use_expert_bias`: use learnable expert bias (e_score_correction)
//!
//! NOTE: Short-convolution layers (Mamba-like state-space layers) require
//! SSM state infrastructure. This implementation supports only the
//! full_attention layers. Models using short-conv layers need additional
//! infrastructure (see Jamba for SSM patterns).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ---- Lfm2 Config (parsed from ModelConfig.extra) ----

struct Lfm2Config {
    block_dim: usize,
    block_ff_dim: usize,
    block_multiple_of: usize,
    block_auto_adjust_ff_dim: bool,
    block_ffn_dim_multiplier: Option<f64>,
    norm_eps: f64,
}

impl Lfm2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let block_dim = cfg
            .extra
            .get("block_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let block_ff_dim = cfg
            .extra
            .get("block_ff_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let block_multiple_of = cfg
            .extra
            .get("block_multiple_of")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(256);

        let block_auto_adjust_ff_dim = cfg
            .extra
            .get("block_auto_adjust_ff_dim")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let block_ffn_dim_multiplier = cfg
            .extra
            .get("block_ffn_dim_multiplier")
            .and_then(|v| v.as_f64());

        let norm_eps = cfg
            .extra
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        Self {
            block_dim,
            block_ff_dim,
            block_multiple_of,
            block_auto_adjust_ff_dim,
            block_ffn_dim_multiplier,
            norm_eps,
        }
    }

    /// Compute effective ff_dim after auto-adjustment.
    fn effective_ff_dim(&self) -> usize {
        if self.block_auto_adjust_ff_dim {
            let mut ff = (2 * self.block_ff_dim) / 3;
            if let Some(mult) = self.block_ffn_dim_multiplier {
                ff = (mult * ff as f64) as usize;
            }
            let m = self.block_multiple_of;
            m * ff.div_ceil(m)
        } else {
            self.block_ff_dim
        }
    }
}

// ---- Lfm2 MoE Config ----

struct Lfm2MoeConfig {
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    num_dense_layers: usize,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
    use_expert_bias: bool,
    norm_eps: f64,
}

impl Lfm2MoeConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_experts = cfg.num_experts().unwrap_or(8);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let num_dense_layers = cfg
            .extra
            .get("num_dense_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let norm_topk_prob = cfg
            .extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let use_expert_bias = cfg
            .extra
            .get("use_expert_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let norm_eps = cfg
            .extra
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        Self {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            num_dense_layers,
            routed_scaling_factor,
            norm_topk_prob,
            use_expert_bias,
            norm_eps,
        }
    }
}

// ---- Lfm2 Attention ----

struct Lfm2Attention {
    qkv_proj: TpLinear,
    out_proj: TpLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl Lfm2Attention {
    fn new_with_tp(
        cfg: &ModelConfig,
        norm_eps: f64,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.hidden_size / num_heads;
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

        let total_qkv = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            total_qkv,
            false,
            false,
            vb.pp("qkv_proj"),
            pg,
        )?;

        let out_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("out_proj"),
            pg,
        )?;

        // Lfm2 always uses QK normalization
        let q_norm = rms_norm(head_dim, norm_eps, vb.pp("q_layernorm"))?;
        let k_norm = rms_norm(head_dim, norm_eps, vb.pp("k_layernorm"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            out_proj,
            q_norm,
            k_norm,
            rotary_emb,
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

        // Lfm2: QK norm BEFORE RoPE (different from HunYuan)
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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

        self.out_proj.forward(&attn_output, tp_ctx)
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

        // QK norm
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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

            self.out_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
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
            self.out_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ---- Lfm2 MLP (gate+up merged, SwiGLU) ----

struct Lfm2MLP {
    w1: TpLinear, // gate+up merged
    w2: TpLinear, // down
}

impl Lfm2MLP {
    fn new(dim: usize, ff_dim: usize, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let w1 = TpLinear::column_parallel(dim, 2 * ff_dim, false, false, vb.pp("w1"), pg)?;
        let w2 = TpLinear::row_parallel(ff_dim, dim, false, true, vb.pp("w2"), pg)?;

        Ok(Self { w1, w2 })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate_up = self.w1.forward(xs, tp_ctx)?;
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let hidden = gate.mul(&chunks[1])?;
        self.w2.forward(&hidden, tp_ctx)
    }
}

// ---- Lfm2 MoE Expert ----

struct Lfm2MoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Lfm2MoEExpert {
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
        let down_proj =
            TpLinear::row_parallel(intermediate_size, hidden_size, false, true, vb.pp("w2"), pg)?;

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

// ---- Lfm2 SparseMoE Block ----

struct Lfm2SparseMoeBlock {
    router: TopKRouter,
    experts: Vec<Lfm2MoEExpert>,
    num_experts: usize,
    #[allow(dead_code)]
    top_k: usize,
    routed_scaling_factor: f64,
}

impl Lfm2SparseMoeBlock {
    fn new(
        cfg: &ModelConfig,
        moe_cfg: &Lfm2MoeConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = moe_cfg.num_experts;
        let top_k = moe_cfg.num_experts_per_tok;

        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize: moe_cfg.norm_topk_prob,
            scoring_func: ScoringFunc::Sigmoid,
            ..Default::default()
        };

        let bias = if moe_cfg.use_expert_bias {
            Some(Tensor::zeros(num_experts, DType::F32, vb.device())?)
        } else {
            None
        };
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), bias)?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(Lfm2MoEExpert::new(
                hidden_size,
                moe_cfg.moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;

        let mut output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

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
            output = output.index_add(&indices, &token_output, 0)?;
        }

        // Apply routed scaling factor
        let output = if (self.routed_scaling_factor - 1.0).abs() > 1e-9 {
            output.affine(self.routed_scaling_factor, 0.0)?
        } else {
            output
        };

        output.reshape(orig_shape)
    }
}

// ---- FFN Variant (Dense or MoE) ----

enum MoeFfnVariant {
    Dense(Lfm2MLP),
    MoE(Lfm2SparseMoeBlock),
}

impl MoeFfnVariant {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            MoeFfnVariant::Dense(mlp) => mlp.forward(xs, tp_ctx),
            MoeFfnVariant::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ---- Lfm2 Attention Decoder Layer ----

struct Lfm2AttentionDecoderLayer {
    self_attn: Lfm2Attention,
    feed_forward: Lfm2MLP,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl Lfm2AttentionDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        lfm_cfg: &Lfm2Config,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Lfm2Attention::new_with_tp(cfg, lfm_cfg.norm_eps, vb.pp("self_attn"), pg)?;

        let ff_dim = lfm_cfg.effective_ff_dim();
        let feed_forward = Lfm2MLP::new(lfm_cfg.block_dim, ff_dim, vb.pp("feed_forward"), pg)?;

        let operator_norm = rms_norm(cfg.hidden_size, lfm_cfg.norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, lfm_cfg.norm_eps, vb.pp("ffn_norm"))?;

        Ok(Self {
            self_attn,
            feed_forward,
            operator_norm,
            ffn_norm,
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
        let xs = self.operator_norm.forward(xs)?;
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
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs, tp_ctx)?;
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
        let xs = self.operator_norm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs, tp_ctx)?;
        residual + xs
    }
}

// ---- Lfm2 MoE Attention Decoder Layer ----

struct Lfm2MoeAttentionDecoderLayer {
    self_attn: Lfm2Attention,
    feed_forward: MoeFfnVariant,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl Lfm2MoeAttentionDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        moe_cfg: &Lfm2MoeConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Lfm2Attention::new_with_tp(cfg, moe_cfg.norm_eps, vb.pp("self_attn"), pg)?;

        let feed_forward = if layer_idx < moe_cfg.num_dense_layers {
            MoeFfnVariant::Dense(Lfm2MLP::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("feed_forward"),
                pg,
            )?)
        } else {
            MoeFfnVariant::MoE(Lfm2SparseMoeBlock::new(
                cfg,
                moe_cfg,
                vb.pp("feed_forward"),
                pg,
            )?)
        };

        let operator_norm = rms_norm(cfg.hidden_size, moe_cfg.norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, moe_cfg.norm_eps, vb.pp("ffn_norm"))?;

        Ok(Self {
            self_attn,
            feed_forward,
            operator_norm,
            ffn_norm,
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
        let xs = self.operator_norm.forward(xs)?;
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
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs, tp_ctx)?;
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
        let xs = self.operator_norm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs, tp_ctx)?;
        residual + xs
    }
}

// ---- Lfm2ForCausalLM (Dense) ----

/// Lfm2 Dense model for causal language modeling.
///
/// NOTE: This implementation supports the attention-layer subset of the hybrid
/// architecture. Short-convolution (ShortConv) layers require SSM state
/// infrastructure not yet available. For models with `layer_types` containing
/// "short_conv", only the attention layers will process tokens; short-conv
/// layers are treated as identity (TODO: implement ShortConv layer support).
pub struct Lfm2ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Lfm2AttentionDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Lfm2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let lfm_cfg = Lfm2Config::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        // Build only attention layers (all layers treated as attention for dense)
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Lfm2AttentionDecoderLayer::new_with_tp(
                cfg,
                &lfm_cfg,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, lfm_cfg.norm_eps, vb_m.pp("embedding_norm"))?;

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

impl crate::engine::ModelForward for Lfm2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Lfm2ForCausalLM::forward(
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

// ---- Lfm2MoeForCausalLM ----

/// Lfm2 MoE model for causal language modeling.
///
/// Uses sigmoid-scored top-k routing with optional expert bias. First
/// `num_dense_layers` layers use dense MLP, remaining layers use MoE.
///
/// NOTE: Same ShortConv limitation as Lfm2ForCausalLM applies. See above.
pub struct Lfm2MoeForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Lfm2MoeAttentionDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Lfm2MoeForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let moe_cfg = Lfm2MoeConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Lfm2MoeAttentionDecoderLayer::new_with_tp(
                cfg,
                &moe_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, moe_cfg.norm_eps, vb_m.pp("embedding_norm"))?;

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

impl crate::engine::ModelForward for Lfm2MoeForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Lfm2MoeForCausalLM::forward(
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

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn dense_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Lfm2ForCausalLM".to_string()],
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
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("num_dense_layers".to_string(), serde_json::json!(1));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["Lfm2MoeForCausalLM".to_string()],
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

    // ---- Config Parsing Tests ----

    #[test]
    fn test_lfm2_config_defaults() {
        let cfg = dense_config();
        let lfm_cfg = Lfm2Config::from_model_config(&cfg);
        assert_eq!(lfm_cfg.block_dim, cfg.hidden_size);
        assert_eq!(lfm_cfg.block_ff_dim, cfg.intermediate_size);
        assert_eq!(lfm_cfg.block_multiple_of, 256);
        assert!(!lfm_cfg.block_auto_adjust_ff_dim);
        assert!(lfm_cfg.block_ffn_dim_multiplier.is_none());
    }

    #[test]
    fn test_lfm2_config_auto_adjust_ff_dim() {
        let mut cfg = dense_config();
        cfg.extra.insert(
            "block_auto_adjust_ff_dim".to_string(),
            serde_json::json!(true),
        );
        cfg.extra
            .insert("block_ff_dim".to_string(), serde_json::json!(384));
        cfg.extra
            .insert("block_multiple_of".to_string(), serde_json::json!(64));
        let lfm_cfg = Lfm2Config::from_model_config(&cfg);
        assert!(lfm_cfg.block_auto_adjust_ff_dim);
        let ff = lfm_cfg.effective_ff_dim();
        // 2 * 384 / 3 = 256, aligned to 64 = 256
        assert_eq!(ff, 256);
    }

    #[test]
    fn test_lfm2_config_auto_adjust_with_multiplier() {
        let mut cfg = dense_config();
        cfg.extra.insert(
            "block_auto_adjust_ff_dim".to_string(),
            serde_json::json!(true),
        );
        cfg.extra
            .insert("block_ff_dim".to_string(), serde_json::json!(384));
        cfg.extra
            .insert("block_multiple_of".to_string(), serde_json::json!(64));
        cfg.extra.insert(
            "block_ffn_dim_multiplier".to_string(),
            serde_json::json!(1.5),
        );
        let lfm_cfg = Lfm2Config::from_model_config(&cfg);
        let ff = lfm_cfg.effective_ff_dim();
        // 2*384/3 = 256, * 1.5 = 384, aligned to 64 = 384
        assert_eq!(ff, 384);
    }

    #[test]
    fn test_lfm2_moe_config_defaults() {
        let cfg = moe_config();
        let moe_cfg = Lfm2MoeConfig::from_model_config(&cfg);
        assert_eq!(moe_cfg.num_experts, 4);
        assert_eq!(moe_cfg.num_experts_per_tok, 2);
        assert_eq!(moe_cfg.moe_intermediate_size, 64);
        assert_eq!(moe_cfg.num_dense_layers, 1);
        assert!((moe_cfg.routed_scaling_factor - 1.0).abs() < 1e-9);
        assert!(moe_cfg.norm_topk_prob);
        assert!(!moe_cfg.use_expert_bias);
    }

    // ---- Dense Construction Tests ----

    #[test]
    fn test_lfm2_dense_construction() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Lfm2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Lfm2ForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_lfm2_dense_forward_shape() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_lfm2_dense_prefill_then_decode() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_lfm2_dense_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_lfm2_dense_device() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_lfm2_dense_untied_embeddings() {
        let mut cfg = dense_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2ForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    // ---- MLP Forward Test ----

    #[test]
    fn test_lfm2_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let mlp = Lfm2MLP::new(64, 128, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }

    // ---- MoE Construction Tests ----

    #[test]
    fn test_lfm2_moe_construction() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Lfm2MoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Lfm2MoeForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_lfm2_moe_mixed_layers() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Lfm2MoeForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0: dense (num_dense_layers=1), Layer 1: MoE
        assert!(matches!(
            model.layers[0].feed_forward,
            MoeFfnVariant::Dense(_)
        ));
        assert!(matches!(
            model.layers[1].feed_forward,
            MoeFfnVariant::MoE(_)
        ));
    }

    #[test]
    fn test_lfm2_moe_forward_shape() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_lfm2_moe_prefill_then_decode() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_lfm2_moe_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Lfm2MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_lfm2_moe_sparse_block_forward() {
        let cfg = moe_config();
        let moe_cfg = Lfm2MoeConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = Lfm2SparseMoeBlock::new(&cfg, &moe_cfg, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_lfm2_moe_with_scaling_factor() {
        let mut cfg = moe_config();
        cfg.extra
            .insert("routed_scaling_factor".to_string(), serde_json::json!(2.5));
        let moe_cfg = Lfm2MoeConfig::from_model_config(&cfg);
        assert!((moe_cfg.routed_scaling_factor - 2.5).abs() < 1e-9);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = Lfm2SparseMoeBlock::new(&cfg, &moe_cfg, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((1, 2, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok());
    }

    #[test]
    fn test_lfm2_moe_with_expert_bias() {
        let mut cfg = moe_config();
        cfg.extra
            .insert("use_expert_bias".to_string(), serde_json::json!(true));
        let moe_cfg = Lfm2MoeConfig::from_model_config(&cfg);
        assert!(moe_cfg.use_expert_bias);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let moe = Lfm2SparseMoeBlock::new(&cfg, &moe_cfg, vb.pp("moe"), &pg);
        assert!(moe.is_ok(), "MoE with expert bias: {:?}", moe.err());
    }
}
