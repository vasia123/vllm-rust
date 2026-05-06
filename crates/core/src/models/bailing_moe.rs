//! BailingMoE model implementation (v1 + v2).
//!
//! Baichuan MoE architecture from inclusionAI/Ling. Both variants share the same
//! structure: dense layers for early indices (< first_k_dense_replace), MoE layers
//! for the rest. Features:
//!
//! - Merged QKV projection with optional QK normalization
//! - RoPE with configurable partial rotation
//! - SwiGLU activation (SiLU gate + multiply)
//! - Top-k routing with optional sigmoid scoring and correction bias
//! - Optional shared experts alongside routed experts
//! - Routed scaling factor
//!
//! BailingMoeV2ForCausalLM is architecturally identical to BailingMoeForCausalLM.

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{
    AttentionBias, AttentionBlock, AttentionConfig, ProjNames, QkNormVariant,
};
use crate::layers::RotaryEmbedding;
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

use super::tp_layers::{TpContext, TpEmbedding, TpFusedSwiGluMlp, TpLinear, TpSwiGluMlp};

// ─── Config ──────────────────────────────────────────────────────────────────

struct BailingMoeConfig {
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    first_k_dense_replace: usize,
    num_shared_experts: usize,
    moe_intermediate_size: usize,
    moe_shared_expert_intermediate_size: Option<usize>,
    use_qk_norm: bool,
    use_bias: bool,
    use_qkv_bias: bool,
    score_function: ScoringFunc,
    routed_scaling_factor: f64,
    n_group: Option<usize>,
    topk_group: Option<usize>,
}

impl BailingMoeConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let norm_topk_prob = cfg
            .extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let first_k_dense_replace = cfg
            .extra
            .get("first_k_dense_replace")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let num_shared_experts = cfg
            .extra
            .get("num_shared_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let moe_shared_expert_intermediate_size = cfg
            .extra
            .get("moe_shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let use_qk_norm = cfg
            .extra
            .get("use_qk_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let use_bias = cfg
            .extra
            .get("use_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let use_qkv_bias = cfg
            .extra
            .get("use_qkv_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let score_function = cfg
            .extra
            .get("score_function")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "sigmoid" => ScoringFunc::Sigmoid,
                _ => ScoringFunc::Softmax,
            })
            .unwrap_or(ScoringFunc::Softmax);

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

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
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
            first_k_dense_replace,
            num_shared_experts,
            moe_intermediate_size,
            moe_shared_expert_intermediate_size,
            use_qk_norm,
            use_bias,
            use_qkv_bias,
            score_function,
            routed_scaling_factor,
            n_group,
            topk_group,
        }
    }

    fn has_bias(&self) -> bool {
        self.use_bias || self.use_qkv_bias
    }

    fn use_grouped_topk(&self) -> bool {
        self.n_group.is_some() && self.topk_group.is_some()
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

// Bailing-MoE = vanilla GQA + fused 'query_key_value' + 'dense' (output) +
// optional per-head QK RMSNorm with names 'query_layernorm'/'key_layernorm'.
// QKV bias controlled by has_bias() = use_bias || use_qkv_bias; O bias only by
// use_bias.
struct BailingAttention {
    inner: AttentionBlock,
}

impl BailingAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        bm_cfg: &BailingMoeConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let bias = AttentionBias {
            q: bm_cfg.has_bias(),
            k: bm_cfg.has_bias(),
            v: bm_cfg.has_bias(),
            o: bm_cfg.use_bias,
        };
        let mut attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        )
        .with_qkv_fused()
        .with_bias(bias)
        .with_proj_names(ProjNames {
            qkv: "query_key_value",
            o: "dense",
            q_norm: "query_layernorm",
            k_norm: "key_layernorm",
            ..Default::default()
        });
        if bm_cfg.use_qk_norm {
            attn_cfg = attn_cfg.with_qk_norm(QkNormVariant::PerHead, cfg.rms_norm_eps);
        }
        let rotary_emb = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        let inner = AttentionBlock::new(&attn_cfg, vb, pg, rotary_emb)?;
        Ok(Self { inner })
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
        self.inner.forward(
            xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
            tp_ctx,
        )
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.inner
            .forward_decode_batch(xs, sequences, cache_engine, tp_ctx)
    }
}

// ─── MLP (SwiGLU with merged gate_up) ───────────────────────────────────────

// Bailing-MoE shared expert MLP is the fused-gate-up SwiGLU pattern
// with optional bias (`use_bias` config flag).
type BailingMLP = TpFusedSwiGluMlp;

// ─── MoE Expert ──────────────────────────────────────────────────────────────

struct BailingMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl BailingMoEExpert {
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
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ─── MoE Block ───────────────────────────────────────────────────────────────

struct BailingMoEBlock {
    router: TopKRouter,
    experts: Vec<BailingMoEExpert>,
    shared_experts: Option<BailingMLP>,
    num_experts: usize,
    routed_scaling_factor: f64,
}

impl BailingMoEBlock {
    fn new(
        cfg: &ModelConfig,
        bm_cfg: &BailingMoeConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = bm_cfg.num_experts;
        let top_k = bm_cfg.num_experts_per_tok;

        // Determine if we need correction bias (sigmoid scoring requires it)
        let has_correction_bias = bm_cfg.score_function == ScoringFunc::Sigmoid;

        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize: bm_cfg.norm_topk_prob,
            scoring_func: bm_cfg.score_function,
            use_grouped_topk: bm_cfg.use_grouped_topk(),
            num_expert_groups: bm_cfg.n_group,
            topk_per_group: bm_cfg.topk_group,
            routed_scaling_factor: bm_cfg.routed_scaling_factor,
        };

        let bias = if has_correction_bias {
            Some(Tensor::zeros(num_experts, DType::F32, vb.device())?)
        } else {
            None
        };
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), bias)?;

        // Routed experts
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(BailingMoEExpert::new(
                hidden_size,
                bm_cfg.moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        // Shared experts
        let shared_experts = if bm_cfg.num_shared_experts > 0 {
            let shared_intermediate = bm_cfg
                .moe_shared_expert_intermediate_size
                .unwrap_or(bm_cfg.moe_intermediate_size)
                * bm_cfg.num_shared_experts;
            Some(BailingMLP::new_with_bias(
                hidden_size,
                shared_intermediate,
                bm_cfg.use_bias,
                vb.pp("shared_experts"),
                pg,
            )?)
        } else {
            None
        };

        Ok(Self {
            router,
            experts,
            shared_experts,
            num_experts,
            routed_scaling_factor: bm_cfg.routed_scaling_factor,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Shared expert output (always active)
        let shared_output = match &self.shared_experts {
            Some(shared) => Some(shared.forward(&xs_2d, tp_ctx)?),
            None => None,
        };

        // Router
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

        // Apply routed scaling factor
        let routed_output = if (self.routed_scaling_factor - 1.0).abs() > f64::EPSILON {
            routed_output.affine(self.routed_scaling_factor, 0.0)?
        } else {
            routed_output
        };

        // Combine: routed + shared
        let combined = match shared_output {
            Some(shared) => (routed_output + shared)?,
            None => routed_output,
        };

        combined.reshape(orig_shape)
    }
}

// ─── FFN variant ─────────────────────────────────────────────────────────────

enum BailingFfn {
    Dense(TpSwiGluMlp),
    MoE(BailingMoEBlock),
}

impl BailingFfn {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            BailingFfn::Dense(mlp) => mlp.forward(xs, tp_ctx),
            BailingFfn::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct BailingMoeDecoderLayer {
    self_attn: BailingAttention,
    ffn: BailingFfn,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl BailingMoeDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        bm_cfg: &BailingMoeConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = BailingAttention::new_with_tp(cfg, bm_cfg, vb.pp("attention"), pg)?;

        let ffn = if layer_idx < bm_cfg.first_k_dense_replace {
            // Dense MLP for early layers
            BailingFfn::Dense(TpSwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
                pg,
            )?)
        } else {
            // MoE for later layers
            BailingFfn::MoE(BailingMoEBlock::new(cfg, bm_cfg, vb.pp("mlp"), pg)?)
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

// ─── Model ───────────────────────────────────────────────────────────────────

/// BailingMoE model for causal language modeling.
///
/// Supports both BailingMoeForCausalLM and BailingMoeV2ForCausalLM architectures
/// (they are structurally identical).
pub struct BailingMoeForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<BailingMoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    #[allow(dead_code)]
    norm_head: bool,
}

impl BailingMoeForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let bm_cfg = BailingMoeConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens = TpEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("word_embeddings"),
            pg,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(BailingMoeDecoderLayer::new_with_tp(
                cfg,
                &bm_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let norm_head = cfg
            .extra
            .get("norm_head")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

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
                vb_m.pp("word_embeddings"),
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
            norm_head,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for BailingMoeForCausalLM {
    fn forward(
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

/// BailingMoeV2 is structurally identical to BailingMoe.
pub type BailingMoeV2ForCausalLM = BailingMoeForCausalLM;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(1));
        extra.insert("num_shared_experts".to_string(), serde_json::json!(0));
        extra.insert("use_qk_norm".to_string(), serde_json::json!(false));
        extra.insert("use_bias".to_string(), serde_json::json!(false));
        extra.insert("use_qkv_bias".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["BailingMoeForCausalLM".to_string()],
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
            bos_token_id: Some(1),
            eos_token_id: Some(2),
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

    // ─── Config Parsing ──────────────────────────────────────────────────────────

    #[test]
    fn test_bailing_config_defaults() {
        let cfg = ModelConfig::default();
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        assert_eq!(bm_cfg.num_experts, 8);
        assert_eq!(bm_cfg.num_experts_per_tok, 2);
        assert_eq!(bm_cfg.first_k_dense_replace, 0);
        assert_eq!(bm_cfg.num_shared_experts, 0);
        assert!(!bm_cfg.use_qk_norm);
        assert_eq!(bm_cfg.score_function, ScoringFunc::Softmax);
        assert_eq!(bm_cfg.routed_scaling_factor, 1.0);
    }

    #[test]
    fn test_bailing_config_custom() {
        let cfg = test_config();
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        assert_eq!(bm_cfg.num_experts, 4);
        assert_eq!(bm_cfg.num_experts_per_tok, 2);
        assert_eq!(bm_cfg.first_k_dense_replace, 1);
        assert_eq!(bm_cfg.moe_intermediate_size, 64);
        assert!(bm_cfg.norm_topk_prob);
    }

    #[test]
    fn test_bailing_config_sigmoid_scoring() {
        let mut cfg = test_config();
        cfg.extra
            .insert("score_function".to_string(), serde_json::json!("sigmoid"));
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        assert_eq!(bm_cfg.score_function, ScoringFunc::Sigmoid);
    }

    #[test]
    fn test_bailing_config_grouped_topk() {
        let mut cfg = test_config();
        cfg.extra
            .insert("n_group".to_string(), serde_json::json!(2));
        cfg.extra
            .insert("topk_group".to_string(), serde_json::json!(1));
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        assert!(bm_cfg.use_grouped_topk());
        assert_eq!(bm_cfg.n_group, Some(2));
        assert_eq!(bm_cfg.topk_group, Some(1));
    }

    #[test]
    fn test_bailing_config_shared_experts() {
        let mut cfg = test_config();
        cfg.extra
            .insert("num_shared_experts".to_string(), serde_json::json!(2));
        cfg.extra.insert(
            "moe_shared_expert_intermediate_size".to_string(),
            serde_json::json!(32),
        );
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        assert_eq!(bm_cfg.num_shared_experts, 2);
        assert_eq!(bm_cfg.moe_shared_expert_intermediate_size, Some(32));
    }

    // ─── Construction ────────────────────────────────────────────────────────────

    #[test]
    fn test_bailing_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BailingMoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "BailingMoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_bailing_mixed_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BailingMoeForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0 is dense (first_k_dense_replace=1), layer 1 is MoE
        assert!(matches!(model.layers[0].ffn, BailingFfn::Dense(_)));
        assert!(matches!(model.layers[1].ffn, BailingFfn::MoE(_)));
    }

    #[test]
    fn test_bailing_all_moe_layers() {
        let mut cfg = test_config();
        cfg.extra
            .insert("first_k_dense_replace".to_string(), serde_json::json!(0));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BailingMoeForCausalLM::new(&cfg, vb).unwrap();

        // All layers should be MoE when first_k_dense_replace=0
        for layer in &model.layers {
            assert!(matches!(layer.ffn, BailingFfn::MoE(_)));
        }
    }

    #[test]
    fn test_bailing_with_shared_experts() {
        let mut cfg = test_config();
        cfg.extra
            .insert("num_shared_experts".to_string(), serde_json::json!(2));
        cfg.extra
            .insert("first_k_dense_replace".to_string(), serde_json::json!(0));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BailingMoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct with shared experts: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_bailing_with_qk_norm() {
        let mut cfg = test_config();
        cfg.extra
            .insert("use_qk_norm".to_string(), serde_json::json!(true));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BailingMoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct with QK norm: {:?}",
            model.err()
        );
    }

    // ─── Forward ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_bailing_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BailingMoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_bailing_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BailingMoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_bailing_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BailingMoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_bailing_v2_is_same_type() {
        // V2 is a type alias, should construct identically
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BailingMoeV2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "BailingMoeV2ForCausalLM should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_bailing_routed_scaling_factor() {
        let mut cfg = test_config();
        cfg.extra
            .insert("routed_scaling_factor".to_string(), serde_json::json!(2.0));
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        assert!((bm_cfg.routed_scaling_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bailing_moe_block_forward() {
        let cfg = test_config();
        let bm_cfg = BailingMoeConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = BailingMoEBlock::new(&cfg, &bm_cfg, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }
}
