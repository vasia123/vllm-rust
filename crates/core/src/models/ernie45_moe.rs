//! ERNIE 4.5 MoE model implementation.
//!
//! Baidu ERNIE 4.5 MoE variant. Features:
//!
//! - Standard Llama-like attention with QKV parallel projection
//! - SiLU activation, RoPE, RMSNorm
//! - Mixed dense/MoE layers based on configurable layer range and interval
//! - Top-k routing with sigmoid scoring and learnable correction bias
//! - Optional shared experts alongside routed experts
//! - FP32 router gate

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBias, AttentionBlock, AttentionConfig};
use crate::layers::RotaryEmbedding;
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Config ──────────────────────────────────────────────────────────────────

struct Ernie45MoeConfig {
    moe_num_experts: usize,
    moe_k: usize,
    moe_intermediate_size: usize,
    moe_num_shared_experts: usize,
    moe_layer_start_index: usize,
    moe_layer_end_index: usize,
    moe_layer_interval: usize,
    use_bias: bool,
}

impl Ernie45MoeConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let moe_num_experts = cfg
            .extra
            .get("moe_num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let moe_k = cfg.extra.get("moe_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let moe_num_shared_experts = cfg
            .extra
            .get("moe_num_shared_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let moe_layer_start_index = cfg
            .extra
            .get("moe_layer_start_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let moe_layer_end_index =
            cfg.extra
                .get("moe_layer_end_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.num_hidden_layers.saturating_sub(1) as u64) as usize;

        let moe_layer_interval = cfg
            .extra
            .get("moe_layer_interval")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let use_bias = cfg
            .extra
            .get("use_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            moe_num_experts,
            moe_k,
            moe_intermediate_size,
            moe_num_shared_experts,
            moe_layer_start_index,
            moe_layer_end_index,
            moe_layer_interval,
            use_bias,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        let use_moe = self.moe_num_experts > 0;
        use_moe
            && (layer_idx + 1).is_multiple_of(self.moe_layer_interval)
            && layer_idx >= self.moe_layer_start_index
            && layer_idx <= self.moe_layer_end_index
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

// Ernie4.5-MoE = vanilla GQA + fused qkv_proj + bias on QKV (configurable).
// NOTE: Original code allowed `num_kv_heads < world_size` (replicated KV heads
// via `max(1, num_kv_heads/world_size)`). AttentionBlock requires divisibility,
// so this migration tightens TP requirements: callers with a tiny KV-head count
// must use world_size <= num_kv_heads. Tests are single-GPU so this path is
// unaffected.
struct Ernie45MoeAttention {
    inner: AttentionBlock,
}

impl Ernie45MoeAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        e_cfg: &Ernie45MoeConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let bias = if e_cfg.use_bias {
            AttentionBias::QKV_ONLY
        } else {
            AttentionBias::NONE
        };
        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        )
        .with_qkv_fused()
        .with_bias(bias);
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

struct Ernie45MoeMLP {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Ernie45MoeMLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        use_bias: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            use_bias,
            false,
            vb.pp("gate_up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            use_bias,
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

// ─── MoE Expert ──────────────────────────────────────────────────────────────

struct Ernie45MoeExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Ernie45MoeExpert {
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

struct Ernie45MoEBlock {
    router: TopKRouter,
    experts: Vec<Ernie45MoeExpert>,
    shared_experts: Option<Ernie45MoeMLP>,
    num_experts: usize,
    has_shared_experts: bool,
}

impl Ernie45MoEBlock {
    fn new(
        cfg: &ModelConfig,
        e_cfg: &Ernie45MoeConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = e_cfg.moe_num_experts;
        let top_k = e_cfg.moe_k;

        // FP32 router with sigmoid scoring and correction bias
        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize: true,
            scoring_func: ScoringFunc::Sigmoid,
            ..Default::default()
        };

        let bias = Tensor::zeros(num_experts, DType::F32, vb.device())?;
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), Some(bias))?;

        // Routed experts
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(Ernie45MoeExpert::new(
                hidden_size,
                e_cfg.moe_intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        // Shared experts
        let has_shared_experts = e_cfg.moe_num_shared_experts > 0;
        let shared_experts = if has_shared_experts {
            let shared_intermediate = e_cfg.moe_intermediate_size * e_cfg.moe_num_shared_experts;
            Some(Ernie45MoeMLP::new(
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
            shared_experts,
            num_experts,
            has_shared_experts,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Shared expert output
        let shared_output = match &self.shared_experts {
            Some(shared) => Some(shared.forward(&xs_2d, tp_ctx)?),
            None => None,
        };

        // Router (cast to FP32 for routing)
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

        // Combine: shared + routed
        let combined = if self.has_shared_experts {
            match shared_output {
                Some(shared) => (shared + routed_output)?,
                None => routed_output,
            }
        } else {
            routed_output
        };

        combined.reshape(orig_shape)
    }
}

// ─── FFN Variant ─────────────────────────────────────────────────────────────

enum Ernie45Ffn {
    Dense(TpSwiGluMlp),
    MoE(Ernie45MoEBlock),
}

impl Ernie45Ffn {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            Ernie45Ffn::Dense(mlp) => mlp.forward(xs, tp_ctx),
            Ernie45Ffn::MoE(moe) => moe.forward(xs, tp_ctx),
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct Ernie45MoeDecoderLayer {
    self_attn: Ernie45MoeAttention,
    ffn: Ernie45Ffn,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Ernie45MoeDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        e_cfg: &Ernie45MoeConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Ernie45MoeAttention::new_with_tp(cfg, e_cfg, vb.pp("self_attn"), pg)?;

        let ffn = if e_cfg.is_moe_layer(layer_idx) {
            Ernie45Ffn::MoE(Ernie45MoEBlock::new(cfg, e_cfg, vb.pp("mlp"), pg)?)
        } else {
            Ernie45Ffn::Dense(TpSwiGluMlp::new(
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

// ─── Model ───────────────────────────────────────────────────────────────────

/// ERNIE 4.5 MoE model for causal language modeling.
pub struct Ernie45MoeForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Ernie45MoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Ernie45MoeForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let e_cfg = Ernie45MoeConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Ernie45MoeDecoderLayer::new_with_tp(
                cfg,
                &e_cfg,
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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }

    /// Embed token IDs into hidden states `[B, S, hidden_size]`.
    ///
    /// Used by VLM wrappers that need to splice image features before running the LLM.
    pub fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids, &self.tp_ctx)
    }

    /// Run the LLM transformer + norm + lm_head on pre-built embeddings.
    ///
    /// Used by VLM wrappers after merging image features into the embedding stream.
    pub fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let seq_len = embeddings.dim(1)?;
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
        let mut xs = embeddings.clone();
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

    /// Batch-decode path on pre-built embeddings.
    ///
    /// Used by VLM wrappers for decode steps after merging image features.
    pub fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = embeddings.clone();
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
}

impl crate::engine::ModelForward for Ernie45MoeForCausalLM {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("moe_num_experts".to_string(), serde_json::json!(4));
        extra.insert("moe_k".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("moe_num_shared_experts".to_string(), serde_json::json!(1));
        extra.insert("moe_layer_start_index".to_string(), serde_json::json!(0));
        extra.insert("moe_layer_end_index".to_string(), serde_json::json!(1));
        extra.insert("moe_layer_interval".to_string(), serde_json::json!(1));
        extra.insert("use_bias".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["Ernie4_5_MoeForCausalLM".to_string()],
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
            rope_theta: 500000.0,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_config_dense_only() -> ModelConfig {
        // No MoE experts — all layers are dense
        let mut cfg = test_config();
        cfg.extra
            .insert("moe_num_experts".to_string(), serde_json::json!(0));
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

    // ─── Config Parsing ──────────────────────────────────────────────────────────

    #[test]
    fn test_ernie45_config_defaults() {
        let cfg = ModelConfig::default();
        let e_cfg = Ernie45MoeConfig::from_model_config(&cfg);
        assert_eq!(e_cfg.moe_num_experts, 0);
        assert_eq!(e_cfg.moe_k, 2);
        assert_eq!(e_cfg.moe_layer_interval, 1);
        assert!(!e_cfg.use_bias);
    }

    #[test]
    fn test_ernie45_config_custom() {
        let cfg = test_config();
        let e_cfg = Ernie45MoeConfig::from_model_config(&cfg);
        assert_eq!(e_cfg.moe_num_experts, 4);
        assert_eq!(e_cfg.moe_k, 2);
        assert_eq!(e_cfg.moe_intermediate_size, 64);
        assert_eq!(e_cfg.moe_num_shared_experts, 1);
        assert_eq!(e_cfg.moe_layer_start_index, 0);
        assert_eq!(e_cfg.moe_layer_end_index, 1);
    }

    #[test]
    fn test_ernie45_moe_layer_detection() {
        let cfg = test_config();
        let e_cfg = Ernie45MoeConfig::from_model_config(&cfg);
        // interval=1, start=0, end=1: layers (0+1)%1==0 -> true, (1+1)%1==0 -> true
        assert!(e_cfg.is_moe_layer(0));
        assert!(e_cfg.is_moe_layer(1));
    }

    #[test]
    fn test_ernie45_moe_layer_interval() {
        let mut cfg = test_config();
        cfg.num_hidden_layers = 4;
        cfg.extra
            .insert("moe_layer_interval".to_string(), serde_json::json!(2));
        cfg.extra
            .insert("moe_layer_end_index".to_string(), serde_json::json!(3));
        let e_cfg = Ernie45MoeConfig::from_model_config(&cfg);
        // (0+1)%2=1 -> false, (1+1)%2=0 -> true, (2+1)%2=1 -> false, (3+1)%2=0 -> true
        assert!(!e_cfg.is_moe_layer(0));
        assert!(e_cfg.is_moe_layer(1));
        assert!(!e_cfg.is_moe_layer(2));
        assert!(e_cfg.is_moe_layer(3));
    }

    #[test]
    fn test_ernie45_no_moe_when_zero_experts() {
        let cfg = test_config_dense_only();
        let e_cfg = Ernie45MoeConfig::from_model_config(&cfg);
        assert!(!e_cfg.is_moe_layer(0));
        assert!(!e_cfg.is_moe_layer(1));
    }

    // ─── Construction ────────────────────────────────────────────────────────────

    #[test]
    fn test_ernie45_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Ernie45MoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Ernie45MoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_ernie45_all_moe_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Ernie45MoeForCausalLM::new(&cfg, vb).unwrap();

        // Both layers should be MoE (interval=1, start=0, end=1)
        for layer in &model.layers {
            assert!(matches!(layer.ffn, Ernie45Ffn::MoE(_)));
        }
    }

    #[test]
    fn test_ernie45_dense_only_construction() {
        let cfg = test_config_dense_only();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Ernie45MoeForCausalLM::new(&cfg, vb).unwrap();

        // All layers should be dense when moe_num_experts=0
        for layer in &model.layers {
            assert!(matches!(layer.ffn, Ernie45Ffn::Dense(_)));
        }
    }

    #[test]
    fn test_ernie45_mixed_layers() {
        let mut cfg = test_config();
        cfg.num_hidden_layers = 4;
        cfg.extra
            .insert("moe_layer_interval".to_string(), serde_json::json!(2));
        cfg.extra
            .insert("moe_layer_end_index".to_string(), serde_json::json!(3));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Ernie45MoeForCausalLM::new(&cfg, vb).unwrap();

        assert!(matches!(model.layers[0].ffn, Ernie45Ffn::Dense(_)));
        assert!(matches!(model.layers[1].ffn, Ernie45Ffn::MoE(_)));
        assert!(matches!(model.layers[2].ffn, Ernie45Ffn::Dense(_)));
        assert!(matches!(model.layers[3].ffn, Ernie45Ffn::MoE(_)));
    }

    #[test]
    fn test_ernie45_with_shared_experts() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Ernie45MoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct with shared experts: {:?}",
            model.err()
        );
    }

    // ─── Forward ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_ernie45_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Ernie45MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_ernie45_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Ernie45MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_ernie45_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Ernie45MoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_ernie45_dense_forward() {
        let cfg = test_config_dense_only();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Ernie45MoeForCausalLM::new(&cfg, vb).expect("build dense model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 2), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 2)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 2);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");
        assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
    }

    #[test]
    fn test_ernie45_moe_block_forward() {
        let cfg = test_config();
        let e_cfg = Ernie45MoeConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let moe = Ernie45MoEBlock::new(&cfg, &e_cfg, vb.pp("moe"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }
}
