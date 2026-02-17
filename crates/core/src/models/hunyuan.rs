//! HunYuan V1 model implementation (Dense + MoE variants).
//!
//! HunYuan Dense V1: standard transformer with SiLU activation and optional QK norm.
//! HunYuan MoE V1: top-k routing with shared experts (mixed MLP/MoE layers).
//!
//! Both variants share the same attention and decoder layer structure:
//! - Merged QKV projection with RoPE
//! - Optional per-head Q/K RMSNorm (use_qk_norm config flag)
//! - Pre-norm residual connections (input_layernorm + post_attention_layernorm)
//!
//! Key config fields (from extra):
//! - `use_qk_norm`: bool (enables per-head Q/K normalization)
//! - `num_experts`: int (triggers MoE when > 1)
//! - `moe_topk`: int or list (per-layer top-k, default 2)
//! - `moe_intermediate_size`: int or null (overrides intermediate_size for MoE layers)
//! - `use_mixed_mlp_moe`: int (0 = no shared expert, >0 = shared expert)
//! - `num_shared_expert`: int or list (shared expert count multiplier)

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

// ---- HunYuan Config (parsed from ModelConfig.extra) ----

#[allow(dead_code)]
struct HunYuanConfig {
    use_qk_norm: bool,
    is_moe: bool,
    moe_topk: Vec<usize>,
    moe_intermediate_sizes: Vec<usize>,
    use_mixed_mlp_moe: bool,
    num_shared_expert: Vec<usize>,
    attention_bias: bool,
    mlp_bias: bool,
}

impl HunYuanConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let use_qk_norm = cfg.use_qk_norm();

        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| {
                v.as_u64().map(|n| vec![n as usize]).or_else(|| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.as_u64().map(|n| n as usize))
                            .collect()
                    })
                })
            })
            .unwrap_or_default();

        let is_moe = if let Some(&single) = num_experts.first() {
            if num_experts.len() == 1 {
                single > 1
            } else {
                num_experts.iter().any(|&e| e > 1)
            }
        } else {
            false
        };

        // moe_topk can be int or list
        let moe_topk = cfg
            .extra
            .get("moe_topk")
            .and_then(|v| {
                v.as_u64().map(|n| vec![n as usize]).or_else(|| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.as_u64().map(|n| n as usize))
                            .collect()
                    })
                })
            })
            .unwrap_or_else(|| vec![2]);

        // moe_intermediate_size can be int or list
        let moe_intermediate_sizes = cfg
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| {
                v.as_u64().map(|n| vec![n as usize]).or_else(|| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.as_u64().map(|n| n as usize))
                            .collect()
                    })
                })
            })
            .unwrap_or_else(|| vec![cfg.intermediate_size]);

        let use_mixed_mlp_moe = cfg
            .extra
            .get("use_mixed_mlp_moe")
            .and_then(|v| v.as_u64())
            .map(|v| v > 0)
            .unwrap_or(false);

        // num_shared_expert can be int or list
        let num_shared_expert = cfg
            .extra
            .get("num_shared_expert")
            .and_then(|v| {
                v.as_u64().map(|n| vec![n as usize]).or_else(|| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.as_u64().map(|n| n as usize))
                            .collect()
                    })
                })
            })
            .unwrap_or_else(|| vec![1]);

        let attention_bias = cfg.attention_bias.unwrap_or(false)
            || cfg
                .extra
                .get("bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

        let mlp_bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            use_qk_norm,
            is_moe,
            moe_topk,
            moe_intermediate_sizes,
            use_mixed_mlp_moe,
            num_shared_expert,
            attention_bias,
            mlp_bias,
        }
    }

    fn moe_topk_for_layer(&self, layer_idx: usize) -> usize {
        if self.moe_topk.len() == 1 {
            self.moe_topk[0]
        } else {
            self.moe_topk.get(layer_idx).copied().unwrap_or(2)
        }
    }

    fn moe_intermediate_size_for_layer(&self, layer_idx: usize) -> usize {
        if self.moe_intermediate_sizes.len() == 1 {
            self.moe_intermediate_sizes[0]
        } else {
            self.moe_intermediate_sizes
                .get(layer_idx)
                .copied()
                .unwrap_or(self.moe_intermediate_sizes[0])
        }
    }

    fn num_shared_expert_for_layer(&self, layer_idx: usize) -> usize {
        if self.num_shared_expert.len() == 1 {
            self.num_shared_expert[0]
        } else {
            self.num_shared_expert.get(layer_idx).copied().unwrap_or(1)
        }
    }
}

// ---- HunYuan Attention ----

struct HunYuanAttention {
    qkv_proj: TpLinear,
    o_proj: TpLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl HunYuanAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        hy_cfg: &HunYuanConfig,
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

        let total_qkv = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            total_qkv,
            hy_cfg.attention_bias,
            false,
            vb.pp("qkv_proj"),
            pg,
        )?;

        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            hy_cfg.attention_bias,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let (q_norm, k_norm) = if hy_cfg.use_qk_norm {
            (
                Some(rms_norm(
                    head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("query_layernorm"),
                )?),
                Some(rms_norm(
                    head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("key_layernorm"),
                )?),
            )
        } else {
            (None, None)
        };

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
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

        // RoPE before QK norm (HunYuan applies RoPE first)
        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Optional per-head QK normalization
        let q = if let Some(ref norm) = self.q_norm {
            apply_per_head_norm(&q, norm)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            apply_per_head_norm(&k, norm)?
        } else {
            k
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

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

            // Optional QK norm (3D: [batch, heads, head_dim])
            let q = if let Some(ref norm) = self.q_norm {
                let (b, h, d) = q.dims3()?;
                let q_flat = q.reshape((b * h, d))?;
                let q_norm = candle_nn::Module::forward(norm, &q_flat)?;
                q_norm.reshape((b, h, d))?
            } else {
                q
            };
            let k = if let Some(ref norm) = self.k_norm {
                let (b, h, d) = k.dims3()?;
                let k_flat = k.reshape((b * h, d))?;
                let k_norm = candle_nn::Module::forward(norm, &k_flat)?;
                k_norm.reshape((b, h, d))?
            } else {
                k
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

                let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

                let q_i = if let Some(ref norm) = self.q_norm {
                    apply_per_head_norm(&q_i, norm)?
                } else {
                    q_i
                };
                let k_i = if let Some(ref norm) = self.k_norm {
                    apply_per_head_norm(&k_i, norm)?
                } else {
                    k_i
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

// ---- MoE Expert ----

struct HunYuanMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl HunYuanMoEExpert {
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

// ---- Shared MLP (gate_up merged) ----

struct HunYuanSharedMLP {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
}

impl HunYuanSharedMLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            bias,
            false,
            vb.pp("gate_up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            bias,
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
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let hidden = gate.mul(&chunks[1])?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ---- SparseMoE Block (router + experts + optional shared expert) ----

struct HunYuanSparseMoeBlock {
    router: TopKRouter,
    experts: Vec<HunYuanMoEExpert>,
    shared_mlp: Option<HunYuanSharedMLP>,
    num_experts: usize,
    #[allow(dead_code)]
    top_k: usize,
}

impl HunYuanSparseMoeBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &ModelConfig,
        hy_cfg: &HunYuanConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = cfg.num_experts().unwrap_or(8);
        let top_k = hy_cfg.moe_topk_for_layer(layer_idx);
        let intermediate_size = hy_cfg.moe_intermediate_size_for_layer(layer_idx);

        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize: top_k > 1,
            scoring_func: ScoringFunc::Softmax,
            ..Default::default()
        };
        let router = TopKRouter::new(router_config, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(HunYuanMoEExpert::new(
                hidden_size,
                intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        let shared_mlp = if hy_cfg.use_mixed_mlp_moe {
            let num_shared = hy_cfg.num_shared_expert_for_layer(layer_idx);
            let shared_intermediate = cfg.intermediate_size * num_shared;
            Some(HunYuanSharedMLP::new(
                hidden_size,
                shared_intermediate,
                false,
                vb.pp("shared_mlp"),
                pg,
            )?)
        } else {
            None
        };

        Ok(Self {
            router,
            experts,
            shared_mlp,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Shared expert (always active for all tokens) if present
        let shared_output = if let Some(ref shared_mlp) = self.shared_mlp {
            Some(shared_mlp.forward(&xs_2d, tp_ctx)?)
        } else {
            None
        };

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

        // Combine: shared_output + routed_output (if shared present)
        let combined = if let Some(shared) = shared_output {
            (shared + routed_output)?
        } else {
            routed_output
        };

        combined.reshape(orig_shape)
    }
}

// ---- FFN Variant ----

enum FfnVariant {
    Dense(TpSwiGluMlp),
    MoE(HunYuanSparseMoeBlock),
}

impl FfnVariant {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            FfnVariant::Dense(mlp) => mlp.forward(xs, tp_ctx),
            FfnVariant::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ---- Decoder Layer ----

struct HunYuanDecoderLayer {
    self_attn: HunYuanAttention,
    ffn: FfnVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl HunYuanDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        hy_cfg: &HunYuanConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = HunYuanAttention::new_with_tp(cfg, hy_cfg, vb.pp("self_attn"), pg)?;

        let ffn = if hy_cfg.is_moe {
            FfnVariant::MoE(HunYuanSparseMoeBlock::new(
                cfg,
                hy_cfg,
                layer_idx,
                vb.pp("mlp"),
                pg,
            )?)
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

// ---- HunYuanDenseV1ForCausalLM ----

pub struct HunYuanDenseV1ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<HunYuanDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl HunYuanDenseV1ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let hy_cfg = HunYuanConfig::from_model_config(cfg);
        // Dense variant should not have MoE config; force is_moe = false
        let hy_cfg = HunYuanConfig {
            is_moe: false,
            ..hy_cfg
        };

        Self::new_inner(cfg, &hy_cfg, vb, pg, tp_ctx)
    }

    fn new_inner(
        cfg: &ModelConfig,
        hy_cfg: &HunYuanConfig,
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
            layers.push(HunYuanDecoderLayer::new_with_tp(
                cfg,
                hy_cfg,
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

impl crate::engine::ModelForward for HunYuanDenseV1ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        HunYuanDenseV1ForCausalLM::forward(
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

// ---- HunYuanMoEV1ForCausalLM ----

pub struct HunYuanMoEV1ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<HunYuanDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl HunYuanMoEV1ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let hy_cfg = HunYuanConfig::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(HunYuanDecoderLayer::new_with_tp(
                cfg,
                &hy_cfg,
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

impl crate::engine::ModelForward for HunYuanMoEV1ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        HunYuanMoEV1ForCausalLM::forward(
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
            architectures: vec!["HunYuanDenseV1ForCausalLM".to_string()],
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

    fn dense_config_with_qk_norm() -> ModelConfig {
        let mut cfg = dense_config();
        cfg.extra
            .insert("use_qk_norm".to_string(), serde_json::json!(true));
        cfg
    }

    fn moe_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("moe_topk".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("use_mixed_mlp_moe".to_string(), serde_json::json!(1));
        extra.insert("num_shared_expert".to_string(), serde_json::json!(1));

        ModelConfig {
            architectures: vec!["HunYuanMoEV1ForCausalLM".to_string()],
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
    fn test_hunyuan_config_defaults() {
        let cfg = dense_config();
        let hy_cfg = HunYuanConfig::from_model_config(&cfg);
        assert!(!hy_cfg.use_qk_norm);
        assert!(!hy_cfg.is_moe);
        assert_eq!(hy_cfg.moe_topk_for_layer(0), 2);
        assert!(!hy_cfg.use_mixed_mlp_moe);
        assert!(!hy_cfg.attention_bias);
        assert!(!hy_cfg.mlp_bias);
    }

    #[test]
    fn test_hunyuan_config_moe() {
        let cfg = moe_config();
        let hy_cfg = HunYuanConfig::from_model_config(&cfg);
        assert!(hy_cfg.is_moe);
        assert_eq!(hy_cfg.moe_topk_for_layer(0), 2);
        assert_eq!(hy_cfg.moe_intermediate_size_for_layer(0), 64);
        assert!(hy_cfg.use_mixed_mlp_moe);
        assert_eq!(hy_cfg.num_shared_expert_for_layer(0), 1);
    }

    #[test]
    fn test_hunyuan_config_qk_norm() {
        let cfg = dense_config_with_qk_norm();
        let hy_cfg = HunYuanConfig::from_model_config(&cfg);
        assert!(hy_cfg.use_qk_norm);
    }

    #[test]
    fn test_hunyuan_config_per_layer_topk() {
        let mut cfg = moe_config();
        cfg.extra
            .insert("moe_topk".to_string(), serde_json::json!([1, 3]));
        let hy_cfg = HunYuanConfig::from_model_config(&cfg);
        assert_eq!(hy_cfg.moe_topk_for_layer(0), 1);
        assert_eq!(hy_cfg.moe_topk_for_layer(1), 3);
        assert_eq!(hy_cfg.moe_topk_for_layer(99), 2); // out of bounds => default
    }

    // ---- Dense Construction Tests ----

    #[test]
    fn test_hunyuan_dense_construction() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "HunYuanDenseV1ForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_hunyuan_dense_with_qk_norm_construction() {
        let cfg = dense_config_with_qk_norm();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Dense with QK norm should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_hunyuan_dense_forward_shape() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_hunyuan_dense_prefill_then_decode() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_hunyuan_dense_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_hunyuan_dense_device() {
        let cfg = dense_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    // ---- MoE Construction Tests ----

    #[test]
    fn test_hunyuan_moe_construction() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = HunYuanMoEV1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "HunYuanMoEV1ForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_hunyuan_moe_has_moe_layers() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = HunYuanMoEV1ForCausalLM::new(&cfg, vb).unwrap();

        // All layers should be MoE (num_experts > 1)
        for layer in &model.layers {
            assert!(matches!(layer.ffn, FfnVariant::MoE(_)));
        }
    }

    #[test]
    fn test_hunyuan_moe_forward_shape() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanMoEV1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_hunyuan_moe_prefill_then_decode() {
        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanMoEV1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_hunyuan_moe_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = moe_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanMoEV1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_hunyuan_moe_block_forward() {
        let cfg = moe_config();
        let hy_cfg = HunYuanConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = HunYuanSparseMoeBlock::new(&cfg, &hy_cfg, 0, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_hunyuan_moe_without_shared_expert() {
        let mut cfg = moe_config();
        cfg.extra
            .insert("use_mixed_mlp_moe".to_string(), serde_json::json!(0));
        let hy_cfg = HunYuanConfig::from_model_config(&cfg);
        assert!(!hy_cfg.use_mixed_mlp_moe);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = HunYuanSparseMoeBlock::new(&cfg, &hy_cfg, 0, vb.pp("moe"), &pg).unwrap();
        assert!(moe.shared_mlp.is_none());

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE without shared: {:?}", output.err());
    }

    #[test]
    fn test_hunyuan_shared_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let shared = HunYuanSharedMLP::new(64, 128, false, vb.pp("shared"), &pg).unwrap();

        let input = Tensor::zeros((2, 64), DType::F32, &device).expect("input");
        let output = shared.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "Shared MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }

    #[test]
    fn test_hunyuan_dense_untied_embeddings() {
        let mut cfg = dense_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = HunYuanDenseV1ForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }
}
