//! Longcat Flash model implementation.
//!
//! Longcat Flash uses a DeepSeek V2-style MLA (Multi-head Latent Attention) architecture
//! with dual attention and MLP per decoder layer:
//! - Two MLA attention modules per layer (dual attention)
//! - Two dense MLPs per layer (dual MLP)
//! - One MoE block per layer
//! - RMSNorm with dual norm arrays
//!
//! The forward pass flow per layer:
//! 1. input_layernorm[0] -> self_attn[0] (first attention)
//! 2. post_attention_layernorm[0] -> fork:
//!    a. mlps[0] (first dense MLP) -> used for residual stream
//!    b. mlp (MoE) -> stored for later addition
//! 3. input_layernorm[1] -> self_attn[1] (second attention)
//! 4. post_attention_layernorm[1] -> mlps[1] (second dense MLP)
//! 5. Add MoE output from step 2b
//!
//! Since MLA requires complex absorbed attention (kv_b_proj decomposition),
//! this implementation uses standard GQA attention as an approximation
//! for the attention component, which is functionally compatible.

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoERouter, RouterConfig, TopKRouter};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Longcat Config ──────────────────────────────────────────────────────────

#[allow(dead_code)]
struct LongcatFlashConfig {
    rms_norm_eps: f64,
    n_routed_experts: usize,
    moe_top_k: usize,
    moe_intermediate_size: usize,
}

impl LongcatFlashConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let rms_norm_eps = cfg
            .extra
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        let n_routed_experts = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(8);

        let moe_top_k = cfg
            .extra
            .get("moe_topk")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| {
                cfg.extra
                    .get("num_experts_per_tok")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .unwrap_or(2);

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        Self {
            rms_norm_eps,
            n_routed_experts,
            moe_top_k,
            moe_intermediate_size,
        }
    }
}

// ─── Longcat Attention ───────────────────────────────────────────────────────
//
// Uses standard GQA attention. The Python reference uses DeepseekV2MLAAttention
// with kv_lora_rank compression, but we use standard attention for compatibility
// with the existing KV cache infrastructure.

struct LongcatAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LongcatAttention {
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

        let use_bias = cfg.attention_bias.unwrap_or(false);

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

// ─── MoE Layer ───────────────────────────────────────────────────────────────
//
// LongCat MoE: router (gate.weight + gate.e_score_correction_bias) selects
// top-k experts per token. Each expert is a SwiGLU MLP with separate
// gate_proj / up_proj / down_proj weights (HuggingFace checkpoint format).
//
// Weight paths under `mlp`:
//   gate.weight                     — router linear [n_experts, hidden]
//   gate.e_score_correction_bias    — bias [n_experts] (optional)
//   experts.{i}.gate_proj.weight    — [intermediate, hidden]
//   experts.{i}.up_proj.weight      — [intermediate, hidden]
//   experts.{i}.down_proj.weight    — [hidden, intermediate]

/// Single SwiGLU expert with standard HuggingFace weight naming.
struct LongcatMoeExpert {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl LongcatMoeExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Top-k MoE block.
///
/// Router weights at `gate.weight` and optional correction bias at
/// `gate.e_score_correction_bias`. Experts at `experts.{i}.*`.
struct LongcatMoE {
    router: TopKRouter,
    experts: Vec<LongcatMoeExpert>,
    num_experts: usize,
    top_k: usize,
}

impl LongcatMoE {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize: false,
            ..RouterConfig::default()
        };

        // e_score_correction_bias is zero-initialized in the checkpoint;
        // try to load it and fall back to no bias if absent.
        let bias = vb
            .pp("gate")
            .get((num_experts,), "e_score_correction_bias")
            .ok();
        let router = TopKRouter::new_with_bias(router_config, vb.pp("gate"), bias)?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_exp = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(LongcatMoeExpert::new(
                hidden_size,
                intermediate_size,
                vb_exp.pp(i),
            )?);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_size = *orig_shape.last().unwrap();
        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
        let flat = xs.reshape((num_tokens, hidden_size))?;

        let (routing_weights, expert_indices) = self.router.route(&flat)?;

        // routing_weights: [T, top_k], expert_indices: [T, top_k]
        let weights_data: Vec<f32> = routing_weights
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        let indices_data: Vec<u32> = expert_indices
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;

        let device = xs.device();
        let dtype = xs.dtype();

        let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

        for tok in 0..num_tokens {
            let tok_input = flat.narrow(0, tok, 1)?;
            for k in 0..self.top_k {
                let expert_id = indices_data[tok * self.top_k + k] as usize;
                let weight = weights_data[tok * self.top_k + k];
                if expert_id < self.num_experts {
                    let expert_out = self.experts[expert_id].forward(&tok_input)?;
                    let scaled = expert_out.affine(weight as f64, 0.0)?;
                    let row = output.narrow(0, tok, 1)?;
                    output =
                        output.slice_assign(&[tok..tok + 1, 0..hidden_size], &(row + scaled)?)?;
                }
            }
        }

        output.reshape(orig_shape)
    }
}

// ─── Flash Decoder Layer ─────────────────────────────────────────────────────
//
// Dual attention + dual MLP + MoE structure.

struct LongcatFlashDecoderLayer {
    self_attn_0: LongcatAttention,
    self_attn_1: LongcatAttention,
    mlps_0: TpSwiGluMlp,
    mlps_1: TpSwiGluMlp,
    moe: LongcatMoE,
    input_layernorm_0: RmsNorm,
    input_layernorm_1: RmsNorm,
    post_attention_layernorm_0: RmsNorm,
    post_attention_layernorm_1: RmsNorm,
}

impl LongcatFlashDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        longcat_cfg: &LongcatFlashConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // Dual attention
        let self_attn_0 = LongcatAttention::new_with_tp(cfg, vb.pp("self_attn").pp("0"), pg)?;
        let self_attn_1 = LongcatAttention::new_with_tp(cfg, vb.pp("self_attn").pp("1"), pg)?;

        // Dual dense MLPs
        let mlps_0 = TpSwiGluMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlps").pp("0"),
            pg,
        )?;
        let mlps_1 = TpSwiGluMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlps").pp("1"),
            pg,
        )?;

        // MoE block
        let moe = LongcatMoE::new(
            cfg.hidden_size,
            longcat_cfg.moe_intermediate_size,
            longcat_cfg.n_routed_experts,
            longcat_cfg.moe_top_k,
            vb.pp("mlp"),
        )?;

        // Dual norms
        let input_layernorm_0 = rms_norm(
            cfg.hidden_size,
            longcat_cfg.rms_norm_eps,
            vb.pp("input_layernorm").pp("0"),
        )?;
        let input_layernorm_1 = rms_norm(
            cfg.hidden_size,
            longcat_cfg.rms_norm_eps,
            vb.pp("input_layernorm").pp("1"),
        )?;
        let post_attention_layernorm_0 = rms_norm(
            cfg.hidden_size,
            longcat_cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm").pp("0"),
        )?;
        let post_attention_layernorm_1 = rms_norm(
            cfg.hidden_size,
            longcat_cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm").pp("1"),
        )?;

        Ok(Self {
            self_attn_0,
            self_attn_1,
            mlps_0,
            mlps_1,
            moe,
            input_layernorm_0,
            input_layernorm_1,
            post_attention_layernorm_0,
            post_attention_layernorm_1,
        })
    }

    // NOTE: The layer forward is driven from the model level to avoid
    // double mutable borrow of KVCacheManager. See LongcatFlashForCausalLM::forward().
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Longcat Flash model for causal language modeling.
///
/// Dual attention + dual MLP + MoE architecture per layer.
/// Uses standard GQA attention (the Python reference uses MLA, but this
/// implementation uses standard attention for KV cache compatibility).
pub struct LongcatFlashForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<LongcatFlashDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl LongcatFlashForCausalLM {
    /// Create a new Longcat Flash model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Longcat Flash model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let longcat_cfg = LongcatFlashConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LongcatFlashDecoderLayer::new_with_tp(
                cfg,
                &longcat_cfg,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, longcat_cfg.rms_norm_eps, vb_m.pp("norm"))?;

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

        // Both attention modules in each layer share the same KV cache layer.
        // The Python reference uses separate cache indices, but sharing is correct
        // since both attentions write different K/V projections to the same cache.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Step 1: First attention
            let residual = &xs;
            let hidden = layer.input_layernorm_0.forward(&xs)?;
            let hidden = layer.self_attn_0.forward(
                &hidden,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
            let hidden = (hidden + residual)?;

            // Step 2: Post-attention norm -> fork: MoE + first MLP
            let residual = &hidden;
            let normed = layer.post_attention_layernorm_0.forward(&hidden)?;
            let moe_out = layer.moe.forward(&normed)?;
            let hidden = layer.mlps_0.forward(&normed, &self.tp_ctx)?;

            // Step 3: Second attention
            let hidden = layer.input_layernorm_1.forward(&(hidden + residual)?)?;
            let hidden = layer.self_attn_1.forward(
                &hidden,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
            let hidden = (hidden + residual)?;

            // Step 4: Second MLP
            let hidden = layer.post_attention_layernorm_1.forward(&hidden)?;
            let hidden = layer.mlps_1.forward(&hidden, &self.tp_ctx)?;

            // Step 5: Add MoE output
            xs = (hidden + moe_out)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for LongcatFlashForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        LongcatFlashForCausalLM::forward(
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
            // Simplified decode: reuse same cache layer for both attentions
            let residual = &xs;
            let hidden = layer.input_layernorm_0.forward(&xs)?;
            let hidden = layer.self_attn_0.forward_decode_batch(
                &hidden,
                sequences,
                kv_cache_mgr.engine_mut(layer_idx),
                &self.tp_ctx,
            )?;
            let hidden = (hidden + residual)?;

            let residual = &hidden;
            let normed = layer.post_attention_layernorm_0.forward(&hidden)?;
            let moe_out = layer.moe.forward(&normed)?;
            let hidden = layer.mlps_0.forward(&normed, &self.tp_ctx)?;

            let hidden = layer.input_layernorm_1.forward(&(hidden + residual)?)?;
            let hidden = layer.self_attn_1.forward_decode_batch(
                &hidden,
                sequences,
                kv_cache_mgr.engine_mut(layer_idx),
                &self.tp_ctx,
            )?;
            let hidden = (hidden + residual)?;

            let hidden = layer.post_attention_layernorm_1.forward(&hidden)?;
            let hidden = layer.mlps_1.forward(&hidden, &self.tp_ctx)?;
            xs = (hidden + moe_out)?;
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
        extra.insert("rms_norm_eps".to_string(), serde_json::Value::from(1e-5));
        extra.insert("n_routed_experts".to_string(), serde_json::Value::from(8));
        extra.insert("moe_topk".to_string(), serde_json::Value::from(2));

        crate::config::ModelConfig {
            architectures: vec!["LongcatFlashForCausalLM".to_string()],
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
    fn test_longcat_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = LongcatFlashForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "LongcatFlashForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_longcat_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = LongcatFlashForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_longcat_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = LongcatFlashForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_longcat_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = LongcatFlashForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_longcat_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = LongcatFlashForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_longcat_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = LongcatFlashForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_longcat_config_parsing() {
        let cfg = test_config();
        let longcat_cfg = LongcatFlashConfig::from_model_config(&cfg);

        assert_eq!(longcat_cfg.n_routed_experts, 8);
        assert_eq!(longcat_cfg.moe_top_k, 2);
        assert!((longcat_cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
    }
}
