//! KimiLinear model architecture with linear attention (KDA).
//!
//! KimiLinear is a hybrid model that combines:
//! - Multi-head Latent Attention (MLA) layers for standard attention
//! - Kimi Delta Attention (KDA) layers for linear attention
//! - Mixture of Experts (MoE) with shared experts
//!
//! The key innovation is using linear attention (KDA) in certain layers
//! instead of standard softmax attention, enabling O(n) rather than O(n^2)
//! computation for those layers.
//!
//! Architecture:
//! ```text
//! Embedding -> [KimiDecoderLayer x N] -> RMSNorm -> LM Head
//!
//! KimiDecoderLayer:
//!   InputLayerNorm -> (KDA or MLA) Attention -> PostAttnNorm -> Residual
//!   MLP or MoE -> Residual
//! ```
//!
//! Since KDA (Kimi Delta Attention) requires specialized state management
//! (Mamba-style recurrent states + conv states) that depends on CUDA kernels,
//! this implementation uses standard MLA attention for all layers. KDA layers
//! would be a future extension requiring the linear attention kernel infrastructure.
//!
//! Reference: `reference/vllm/vllm/model_executor/models/kimi_linear.py`

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── KimiLinear Config Extraction ───────────────────────────────────────────

#[allow(dead_code)]
struct KimiLinearExtraConfig {
    moe_num_experts: usize,
    moe_top_k: usize,
    moe_intermediate_size: usize,
    num_shared_experts: usize,
    first_k_dense_replace: usize,
    moe_layer_freq: usize,
    routed_scaling_factor: f64,
    is_moe: bool,
    // MLA-specific
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    // KDA layer selection
    kda_layers: Vec<usize>,
}

impl KimiLinearExtraConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let is_moe = cfg
            .extra
            .get("is_moe")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let moe_num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);

        let moe_top_k = cfg
            .extra
            .get("num_experts_per_token")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let moe_intermediate_size = cfg
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let num_shared_experts = cfg
            .extra
            .get("num_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let first_k_dense_replace = cfg
            .extra
            .get("first_k_dense_replace")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let moe_layer_freq = cfg
            .extra
            .get("moe_layer_freq")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let qk_nope_head_dim = cfg
            .extra
            .get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let qk_rope_head_dim = cfg
            .extra
            .get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let kv_lora_rank = cfg
            .extra
            .get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);

        // KDA layers: indices of layers that use linear attention
        let kda_layers = cfg
            .extra
            .get("kda_layers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            moe_num_experts,
            moe_top_k,
            moe_intermediate_size,
            num_shared_experts,
            first_k_dense_replace,
            moe_layer_freq,
            routed_scaling_factor,
            is_moe,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            kda_layers,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_moe
            && layer_idx >= self.first_k_dense_replace
            && layer_idx.is_multiple_of(self.moe_layer_freq)
    }

    fn is_kda_layer(&self, layer_idx: usize) -> bool {
        self.kda_layers.contains(&layer_idx)
    }
}

// ─── KimiMLA Attention ──────────────────────────────────────────────────────
//
// Standard MLA-based attention with Q/KV projections.
// This implementation uses standard attention mechanics since MLA absorption
// requires the full DeepSeek-style MLA infrastructure.
// For simplicity and correctness, we use standard Q/K/V with RoPE.

struct KimiAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KimiAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        _extra_cfg: &KimiLinearExtraConfig,
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
            if num_kv_heads > 1 && !num_kv_heads.is_multiple_of(world_size) {
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
        let num_kv_heads_per_gpu = std::cmp::max(1, num_kv_heads / world_size);

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

// ─── Decoder Layer ──────────────────────────────────────────────────────────

enum KimiMlp {
    Dense(TpSwiGluMlp),
    Moe {
        moe: MoELayer,
        shared_experts: Option<TpSwiGluMlp>,
        routed_scaling_factor: f64,
    },
}

struct KimiDecoderLayer {
    self_attn: KimiAttention,
    mlp: KimiMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl KimiDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &KimiLinearExtraConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // NOTE: KDA layers require linear attention kernels (Mamba-style state management).
        // For now, all layers use standard attention. KDA support would be added when
        // the linear attention kernel infrastructure is available.
        if extra_cfg.is_kda_layer(layer_idx) {
            // TODO: Implement KDA (Kimi Delta Attention) when linear attention kernels are available.
            // For now, fall through to standard MLA attention.
        }

        let self_attn = KimiAttention::new_with_tp(cfg, extra_cfg, vb.pp("self_attn"), pg)?;

        let mlp = if extra_cfg.is_moe_layer(layer_idx) {
            let moe_cfg = MoELayerConfig {
                num_experts: extra_cfg.moe_num_experts,
                top_k: extra_cfg.moe_top_k,
                hidden_size: cfg.hidden_size,
                intermediate_size: extra_cfg.moe_intermediate_size,
                renormalize: false,
                inplace: false,
                is_act_and_mul: true,
            };
            let moe = MoELayer::new(moe_cfg, vb.pp("block_sparse_moe"))?;

            let shared_experts = if extra_cfg.num_shared_experts > 0 {
                let shared_intermediate =
                    extra_cfg.moe_intermediate_size * extra_cfg.num_shared_experts;
                Some(TpSwiGluMlp::new(
                    cfg.hidden_size,
                    shared_intermediate,
                    vb.pp("block_sparse_moe").pp("shared_experts"),
                    pg,
                )?)
            } else {
                None
            };

            KimiMlp::Moe {
                moe,
                shared_experts,
                routed_scaling_factor: extra_cfg.routed_scaling_factor,
            }
        } else {
            KimiMlp::Dense(TpSwiGluMlp::new(
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
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        let hidden_states = match &self.mlp {
            KimiMlp::Dense(mlp) => mlp.forward(&hidden_states, tp_ctx)?,
            KimiMlp::Moe {
                moe,
                shared_experts,
                routed_scaling_factor,
            } => {
                let routed = moe.forward(&hidden_states)?;
                let routed = (routed * *routed_scaling_factor)?;
                if let Some(shared) = shared_experts {
                    let shared_out = shared.forward(&hidden_states, tp_ctx)?;
                    (routed + shared_out)?
                } else {
                    routed
                }
            }
        };

        residual + hidden_states
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
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        let hidden_states = match &self.mlp {
            KimiMlp::Dense(mlp) => mlp.forward(&hidden_states, tp_ctx)?,
            KimiMlp::Moe {
                moe,
                shared_experts,
                routed_scaling_factor,
            } => {
                let routed = moe.forward(&hidden_states)?;
                let routed = (routed * *routed_scaling_factor)?;
                if let Some(shared) = shared_experts {
                    let shared_out = shared.forward(&hidden_states, tp_ctx)?;
                    (routed + shared_out)?
                } else {
                    routed
                }
            }
        };

        residual + hidden_states
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct KimiLinearForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<KimiDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl KimiLinearForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let extra_cfg = KimiLinearExtraConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(KimiDecoderLayer::new_with_tp(
                cfg,
                &extra_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // LM head: separate (not tied) for KimiLinear
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

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for KimiLinearForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        KimiLinearForCausalLM::forward(
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
        ModelConfig {
            architectures: vec!["KimiLinearForCausalLM".to_string()],
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
            extra: serde_json::Map::new(),
        }
    }

    fn test_config_moe() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("is_moe".to_string(), serde_json::json!(true));
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_token".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("num_shared_experts".to_string(), serde_json::json!(1));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(0));
        extra.insert("moe_layer_freq".to_string(), serde_json::json!(1));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));

        ModelConfig {
            architectures: vec!["KimiLinearForCausalLM".to_string()],
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
    fn test_kimi_linear_extra_config() {
        let cfg = test_config_moe();
        let extra = KimiLinearExtraConfig::from_model_config(&cfg);

        assert!(extra.is_moe);
        assert_eq!(extra.moe_num_experts, 4);
        assert_eq!(extra.moe_top_k, 2);
        assert!(extra.is_moe_layer(0));
        assert!(extra.is_moe_layer(1));
        assert!(!extra.is_kda_layer(0));
    }

    #[test]
    fn test_kimi_linear_construction_dense() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = KimiLinearForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "KimiLinearForCausalLM (dense) should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_kimi_linear_construction_moe() {
        let cfg = test_config_moe();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = KimiLinearForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "KimiLinearForCausalLM (MoE) should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_kimi_linear_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = KimiLinearForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_kimi_linear_single_token() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = KimiLinearForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_kimi_linear_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = KimiLinearForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_kimi_linear_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = KimiLinearForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_kimi_linear_moe_forward_shape() {
        let cfg = test_config_moe();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = KimiLinearForCausalLM::new(&cfg, vb).expect("build moe model");

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
}
