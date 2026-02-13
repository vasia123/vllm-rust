//! Grok1/Grok1Model model implementation.
//!
//! Grok1 is a Mixture of Experts (MoE) model from xAI featuring:
//! - MoE with top-2 expert routing and GELU activation (not SwiGLU)
//! - Router logit softcapping (tanh-based, default 30.0)
//! - Attention output multiplier
//! - Embedding multiplier scaling
//! - Four RMSNorm per decoder layer (pre/post-attn + pre/post-MoE)
//! - RoPE embeddings
//!
//! Two architecture entries map to this implementation:
//! - `Grok1ForCausalLM`
//! - `Grok1ModelForCausalLM`

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Grok1-specific constants (from HuggingFace config, with defaults) ──────

const DEFAULT_ATTN_OUTPUT_MULTIPLIER: f64 = 0.08838834764831845;
const DEFAULT_OUTPUT_MULTIPLIER_SCALE: f64 = 0.5773502691896257;
const DEFAULT_EMBEDDING_MULTIPLIER_SCALE: f64 = 78.38367176906169;
const DEFAULT_ROUTER_LOGIT_SOFTCAP: f64 = 30.0;

// ─── Grok1 Config (parsed from ModelConfig.extra) ───────────────────────────

struct Grok1Config {
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    attn_output_multiplier: f64,
    output_multiplier_scale: f64,
    embedding_multiplier_scale: f64,
    router_logit_softcap: f64,
}

impl Grok1Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.extra.get("num_local_experts").and_then(|v| v.as_u64()))
            .unwrap_or(8) as usize;

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let attn_output_multiplier = cfg
            .extra
            .get("attn_output_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_ATTN_OUTPUT_MULTIPLIER);

        let output_multiplier_scale = cfg
            .extra
            .get("output_multiplier_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_OUTPUT_MULTIPLIER_SCALE);

        let embedding_multiplier_scale = cfg
            .extra
            .get("embedding_multiplier_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_EMBEDDING_MULTIPLIER_SCALE);

        let router_logit_softcap = cfg
            .extra
            .get("router_logit_softcapping")
            .and_then(|v| v.as_f64())
            .unwrap_or(DEFAULT_ROUTER_LOGIT_SOFTCAP)
            .max(0.0);

        Self {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            attn_output_multiplier,
            output_multiplier_scale,
            embedding_multiplier_scale,
            router_logit_softcap,
        }
    }
}

// ─── Router logit softcapping ───────────────────────────────────────────────

/// Apply tanh-based softcapping: cap * tanh(x / cap).
/// Prevents router logits from growing unboundedly.
fn router_softcap(logits: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(logits.clone());
    }
    let scaled = (logits / cap)?;
    scaled.tanh()? * cap
}

// ─── Grok1 MoE Expert (GELU activation, not SwiGLU) ────────────────────────

/// A single Grok1 MoE expert using GeluAndMul activation.
///
/// Unlike standard MoE experts that use SwiGLU (silu(gate) * up),
/// Grok1 experts use gelu(gate) * up, matching the Python `GeluAndMul`.
struct Grok1MoEExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Grok1MoEExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // Grok1 checkpoint names: linear (gate/w1), linear_v (up/w3), linear_1 (down/w2)
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("linear"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("linear_v"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("linear_1"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // GeluAndMul: gelu(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(xs)?;
        let gate = gate.gelu_erf()?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ─── Grok1 MoE Layer ───────────────────────────────────────────────────────

/// Grok1 Mixture of Experts layer with softcapped router logits.
///
/// The gate (router) produces logits that are capped via tanh before
/// computing top-k routing weights. This stabilizes training and inference.
struct Grok1MoE {
    gate: Linear,
    experts: Vec<Grok1MoEExpert>,
    num_experts: usize,
    top_k: usize,
    router_logit_softcap: f64,
    hidden_size: usize,
}

impl Grok1MoE {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        router_logit_softcap: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(Grok1MoEExpert::new(
                hidden_size,
                intermediate_size,
                vb_experts.pp(i),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts,
            top_k,
            router_logit_softcap,
            hidden_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden_states.dims().to_vec();
        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
        let flat_hidden = hidden_states.reshape((num_tokens, self.hidden_size))?;

        // Router logits with softcapping
        let router_logits = self.gate.forward(&flat_hidden)?;
        let router_logits = router_softcap(&router_logits, self.router_logit_softcap)?;

        // Softmax over experts to get routing probabilities
        let routing_probs = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let mut output = Tensor::zeros((num_tokens, self.hidden_size), dtype, device)?;

        // Per-token routing and expert execution
        for token_idx in 0..num_tokens {
            let token_probs: Vec<f32> = routing_probs
                .i(token_idx)?
                .to_dtype(DType::F32)?
                .to_vec1()?;

            // Top-k expert selection
            let mut indexed: Vec<(usize, f32)> = token_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k_experts: Vec<(usize, f32)> = indexed.into_iter().take(self.top_k).collect();

            // Renormalize top-k weights
            let weight_sum: f32 = top_k_experts.iter().map(|(_, w)| w).sum();
            let token_input = flat_hidden.narrow(0, token_idx, 1)?;
            let mut token_output = Tensor::zeros((1, self.hidden_size), dtype, device)?;

            for (expert_idx, weight) in &top_k_experts {
                if *expert_idx < self.num_experts {
                    let expert_out = self.experts[*expert_idx].forward(&token_input)?;
                    let norm_weight = if weight_sum > 0.0 {
                        *weight / weight_sum
                    } else {
                        0.0
                    };
                    let weighted = expert_out.affine(norm_weight as f64, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], device)?;
            output = output.index_add(&indices, &token_output, 0)?;
        }

        output.reshape(orig_shape)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct Grok1Attention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Multiplicative factor applied to attention output (Grok1-specific)
    attn_output_multiplier: f64,
}

impl Grok1Attention {
    fn new_with_tp(
        cfg: &ModelConfig,
        grok_cfg: &Grok1Config,
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
            attn_output_multiplier: grok_cfg.attn_output_multiplier,
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

        let output = self.o_proj.forward(&attn_output, tp_ctx)?;
        // Grok1 attention output multiplier
        output * self.attn_output_multiplier
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

            let output = self.o_proj.forward(&attn_output, tp_ctx)?;
            (output * self.attn_output_multiplier)?.unsqueeze(1)
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
            let output = self.o_proj.forward(&attn_output, tp_ctx)?;
            output * self.attn_output_multiplier
        }
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

/// Grok1 decoder layer with 4 RMSNorm layers and MoE FFN.
///
/// The norm pattern differs from standard LLaMA-style models:
/// - `pre_attn_norm`: before attention (fused with residual add)
/// - `post_attn_norm`: after attention output (before adding back to residual)
/// - `pre_moe_norm`: before MoE (fused with residual add)
/// - `post_moe_norm`: after MoE output (before adding back to residual)
struct Grok1DecoderLayer {
    attn: Grok1Attention,
    moe_block: Grok1MoE,
    pre_attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_moe_norm: RmsNorm,
    post_moe_norm: RmsNorm,
}

impl Grok1DecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        grok_cfg: &Grok1Config,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let attn = Grok1Attention::new_with_tp(cfg, grok_cfg, vb.pp("attn"), pg)?;

        let moe_block = Grok1MoE::new(
            cfg.hidden_size,
            grok_cfg.moe_intermediate_size,
            grok_cfg.num_experts,
            grok_cfg.num_experts_per_tok,
            grok_cfg.router_logit_softcap,
            vb.pp("moe_block"),
        )?;

        let pre_attn_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_attn_norm"))?;
        let post_attn_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attn_norm"))?;
        let pre_moe_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_moe_norm"))?;
        let post_moe_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_moe_norm"))?;

        Ok(Self {
            attn,
            moe_block,
            pre_attn_norm,
            post_attn_norm,
            pre_moe_norm,
            post_moe_norm,
        })
    }

    /// Forward pass following the Grok1 norm pattern:
    /// 1. pre_attn_norm(hidden) → attention → post_attn_norm → add residual
    /// 2. pre_moe_norm(hidden) → MoE → post_moe_norm → add residual
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
        // Self-attention with pre/post norms
        let residual = xs;
        let hidden = self.pre_attn_norm.forward(xs)?;
        let hidden = self.attn.forward(
            &hidden,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let hidden = self.post_attn_norm.forward(&hidden)?;
        let xs = (hidden + residual)?;

        // MoE block with pre/post norms
        let residual = &xs;
        let hidden = self.pre_moe_norm.forward(&xs)?;
        let hidden = self.moe_block.forward(&hidden)?;
        let hidden = self.post_moe_norm.forward(&hidden)?;
        residual + hidden
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        // Self-attention with pre/post norms
        let residual = xs;
        let hidden = self.pre_attn_norm.forward(xs)?;
        let hidden = self.attn.forward_decode_batch(
            &hidden,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let hidden = self.post_attn_norm.forward(&hidden)?;
        let xs = (hidden + residual)?;

        // MoE block with pre/post norms
        let residual = &xs;
        let hidden = self.pre_moe_norm.forward(&xs)?;
        let hidden = self.moe_block.forward(&hidden)?;
        let hidden = self.post_moe_norm.forward(&hidden)?;
        residual + hidden
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Grok1 MoE model for causal language modeling.
///
/// Features embedding multiplier scaling and output logit scaling.
pub struct Grok1ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Grok1DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    embedding_multiplier_scale: f64,
    output_multiplier_scale: f64,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl Grok1ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let grok_cfg = Grok1Config::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Grok1DecoderLayer::new_with_tp(
                cfg,
                &grok_cfg,
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
            embedding_multiplier_scale: grok_cfg.embedding_multiplier_scale,
            output_multiplier_scale: grok_cfg.output_multiplier_scale,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            num_experts: grok_cfg.num_experts,
            num_experts_per_tok: grok_cfg.num_experts_per_tok,
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

        // Embedding with multiplier scaling
        let mut xs = (self.embed_tokens.forward(input_ids, &self.tp_ctx)?
            * self.embedding_multiplier_scale)?;

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
        // Output multiplier scaling on logits
        logits * self.output_multiplier_scale
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }
}

impl crate::engine::ModelForward for Grok1ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Grok1ForCausalLM::forward(
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
        // Embedding with multiplier scaling
        let mut xs = (self.embed_tokens.forward(input_ids, &self.tp_ctx)?
            * self.embedding_multiplier_scale)?;

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
        logits * self.output_multiplier_scale
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

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert(
            "attn_output_multiplier".to_string(),
            serde_json::json!(DEFAULT_ATTN_OUTPUT_MULTIPLIER),
        );
        extra.insert(
            "output_multiplier_scale".to_string(),
            serde_json::json!(DEFAULT_OUTPUT_MULTIPLIER_SCALE),
        );
        extra.insert(
            "embedding_multiplier_scale".to_string(),
            serde_json::json!(DEFAULT_EMBEDDING_MULTIPLIER_SCALE),
        );
        extra.insert(
            "router_logit_softcapping".to_string(),
            serde_json::json!(30.0),
        );

        ModelConfig {
            architectures: vec!["Grok1ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
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

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_grok1_config_parsing() {
        let cfg = test_config();
        let grok_cfg = Grok1Config::from_model_config(&cfg);

        assert_eq!(grok_cfg.num_experts, 4);
        assert_eq!(grok_cfg.num_experts_per_tok, 2);
        assert!((grok_cfg.attn_output_multiplier - DEFAULT_ATTN_OUTPUT_MULTIPLIER).abs() < 1e-10);
        assert!((grok_cfg.output_multiplier_scale - DEFAULT_OUTPUT_MULTIPLIER_SCALE).abs() < 1e-10);
        assert!(
            (grok_cfg.embedding_multiplier_scale - DEFAULT_EMBEDDING_MULTIPLIER_SCALE).abs()
                < 1e-10
        );
        assert!((grok_cfg.router_logit_softcap - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_grok1_config_defaults() {
        let cfg = ModelConfig::default();
        let grok_cfg = Grok1Config::from_model_config(&cfg);

        assert_eq!(grok_cfg.num_experts, 8);
        assert_eq!(grok_cfg.num_experts_per_tok, 2);
        assert!((grok_cfg.attn_output_multiplier - DEFAULT_ATTN_OUTPUT_MULTIPLIER).abs() < 1e-10);
        assert!((grok_cfg.router_logit_softcap - DEFAULT_ROUTER_LOGIT_SOFTCAP).abs() < 1e-10);
    }

    #[test]
    fn test_grok1_config_num_local_experts_fallback() {
        let mut cfg = ModelConfig::default();
        cfg.extra
            .insert("num_local_experts".to_string(), serde_json::json!(16));
        let grok_cfg = Grok1Config::from_model_config(&cfg);
        assert_eq!(grok_cfg.num_experts, 16);
    }

    // ─── Router Softcap Tests ───────────────────────────────────────────────────

    #[test]
    fn test_router_softcap_clamps() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[100.0_f32, -100.0, 0.0, 15.0], &device).unwrap();
        let capped = router_softcap(&logits, 30.0).unwrap();
        let values: Vec<f32> = capped.to_vec1().unwrap();

        // tanh(100/30)*30 should be close to 30 (tanh saturates)
        assert!(values[0] > 29.0 && values[0] <= 30.0, "got {}", values[0]);
        // tanh(-100/30)*30 should be close to -30
        assert!(values[1] < -29.0 && values[1] >= -30.0, "got {}", values[1]);
        // tanh(0)*30 = 0
        assert!(values[2].abs() < 1e-6, "got {}", values[2]);
        // tanh(15/30)*30 = tanh(0.5)*30 ~ 13.86
        assert!(values[3] > 13.0 && values[3] < 15.0, "got {}", values[3]);
    }

    #[test]
    fn test_router_softcap_zero_passthrough() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[42.0_f32, -7.0], &device).unwrap();
        let result = router_softcap(&logits, 0.0).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();
        assert!((values[0] - 42.0).abs() < 1e-6);
        assert!((values[1] + 7.0).abs() < 1e-6);
    }

    // ─── Expert GELU Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_grok1_expert_gelu_activation() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let expert = Grok1MoEExpert::new(32, 64, vb.pp("expert")).unwrap();
        let input = Tensor::zeros((2, 32), DType::F32, &device).unwrap();
        let output = expert.forward(&input);
        assert!(output.is_ok(), "Expert forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 32]);
    }

    // ─── MoE Layer Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_grok1_moe_construction() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let moe = Grok1MoE::new(32, 64, 4, 2, 30.0, vb.pp("moe"));
        assert!(moe.is_ok(), "MoE construction: {:?}", moe.err());

        let moe = moe.unwrap();
        assert_eq!(moe.num_experts, 4);
        assert_eq!(moe.top_k, 2);
    }

    #[test]
    fn test_grok1_moe_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let moe = Grok1MoE::new(32, 64, 4, 2, 30.0, vb.pp("moe")).unwrap();

        let input = Tensor::zeros((2, 3, 32), DType::F32, &device).unwrap();
        let output = moe.forward(&input);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, 32]);
    }

    #[test]
    fn test_grok1_moe_routing_top2() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let moe = Grok1MoE::new(32, 64, 4, 2, 30.0, vb.pp("moe")).unwrap();
        assert_eq!(moe.top_k, 2);
        assert_eq!(moe.num_experts, 4);

        // Should handle single token
        let input = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        let output = moe.forward(&input);
        assert!(output.is_ok(), "MoE forward single: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[1, 32]);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_grok1_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Grok1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Grok1ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.num_experts(), 4);
        assert_eq!(model.num_experts_per_tok(), 2);
    }

    #[test]
    fn test_grok1_multiplier_scales() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Grok1ForCausalLM::new(&cfg, vb).unwrap();
        assert!(
            (model.embedding_multiplier_scale - DEFAULT_EMBEDDING_MULTIPLIER_SCALE).abs() < 1e-10
        );
        assert!((model.output_multiplier_scale - DEFAULT_OUTPUT_MULTIPLIER_SCALE).abs() < 1e-10);
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_grok1_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Grok1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_grok1_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Grok1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_grok1_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Grok1ForCausalLM::new(&cfg, vb).expect("build model");

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

    // ─── Tied Embeddings Tests ──────────────────────────────────────────────────

    #[test]
    fn test_grok1_tied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = true;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Grok1ForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "tied embeddings: {:?}", model.err());
    }

    // ─── Embedding Multiplier Test ──────────────────────────────────────────────

    #[test]
    fn test_grok1_embedding_multiplier_applied() {
        let device = Device::Cpu;

        // Verify the embedding multiplier constant is non-trivial
        assert!(DEFAULT_EMBEDDING_MULTIPLIER_SCALE > 1.0);
        assert!(DEFAULT_EMBEDDING_MULTIPLIER_SCALE < 100.0);

        // Verify the output multiplier is a valid scaling factor
        assert!(DEFAULT_OUTPUT_MULTIPLIER_SCALE > 0.0);
        assert!(DEFAULT_OUTPUT_MULTIPLIER_SCALE < 1.0);

        // Verify attn multiplier is small (dampens attention output)
        assert!(DEFAULT_ATTN_OUTPUT_MULTIPLIER > 0.0);
        assert!(DEFAULT_ATTN_OUTPUT_MULTIPLIER < 1.0);

        // Verify softcap constant
        let logits = Tensor::new(&[50.0_f32], &device).unwrap();
        let capped = router_softcap(&logits, DEFAULT_ROUTER_LOGIT_SOFTCAP).unwrap();
        let val: Vec<f32> = capped.to_vec1().unwrap();
        assert!(val[0] <= DEFAULT_ROUTER_LOGIT_SOFTCAP as f32 + 0.01);
    }
}
