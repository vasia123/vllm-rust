//! Quantized DeepSeek V2/V3 model implementation.
//!
//! This module provides a quantized version of the DeepSeek V2/V3 model that
//! supports various quantization methods (FP8, GPTQ, AWQ) through the
//! QuantizedWeightLoader abstraction.
//!
//! DeepSeek-specific features preserved in the quantized path:
//! - Multi-head Latent Attention (MLA) with compressed KV cache
//! - Mixture of Experts (MoE) with shared experts
//! - Router (gate) stays unquantized (small linear layer)
//! - MLA projections (Q, KV, O) use QuantizedLinear
//! - Expert gate/up/down projections use QuantizedLinear
//! - RMSNorm stays as-is (not quantized)
//! - Embeddings stay as-is (not quantized)
//! - YaRN RoPE scaling for long contexts

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::mla_cache_engine::MLACacheEngine;
use crate::kv_cache::{BlockId, BlockTable, KVCacheManager};
use crate::layers::RotaryEmbedding;
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Quantized DeepSeek MLP ─────────────────────────────────────────────────

struct QuantizedDeepSeekMLP {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedDeepSeekMLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let gate_proj = loader.load_linear(
            &format!("{prefix}.gate_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let up_proj = loader.load_linear(
            &format!("{prefix}.up_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let down_proj = loader.load_linear(
            &format!("{prefix}.down_proj"),
            intermediate_size,
            hidden_size,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Quantized MoE Expert ───────────────────────────────────────────────────

struct QuantizedDeepSeekMoEExpert {
    w1: Box<dyn QuantizedLinear>, // gate_proj
    w2: Box<dyn QuantizedLinear>, // down_proj
    w3: Box<dyn QuantizedLinear>, // up_proj
}

impl QuantizedDeepSeekMoEExpert {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let w1 = loader.load_linear(
            &format!("{prefix}.w1"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let w2 = loader.load_linear(
            &format!("{prefix}.w2"),
            intermediate_size,
            hidden_size,
            false,
        )?;
        let w3 = loader.load_linear(
            &format!("{prefix}.w3"),
            hidden_size,
            intermediate_size,
            false,
        )?;

        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(w1(x)) * w3(x)
        let gate = candle_nn::ops::silu(&self.w1.forward(xs)?)?;
        let up = self.w3.forward(xs)?;
        self.w2.forward(&(gate * up)?)
    }
}

// ─── Quantized MoE Layer ────────────────────────────────────────────────────

struct QuantizedDeepSeekMoELayer {
    gate: Linear, // Router stays unquantized
    experts: Vec<QuantizedDeepSeekMoEExpert>,
    shared_experts: Option<QuantizedDeepSeekMLP>,
    num_experts_per_tok: usize,
    routed_scaling_factor: f64,
}

impl QuantizedDeepSeekMoELayer {
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let n_routed = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;

        let top_k = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let moe_intermediate = cfg
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.intermediate_size as u64) as usize;

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        // Router (gate) stays unquantized
        let gate = linear_no_bias(cfg.hidden_size, n_routed, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(n_routed);
        for i in 0..n_routed {
            experts.push(QuantizedDeepSeekMoEExpert::new(
                cfg.hidden_size,
                moe_intermediate,
                loader,
                &format!("{prefix}.experts.{i}"),
            )?);
        }

        // Shared experts
        let n_shared = cfg
            .extra
            .get("n_shared_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let shared_experts = if n_shared > 0 {
            Some(QuantizedDeepSeekMLP::new(
                cfg.hidden_size,
                moe_intermediate * n_shared,
                loader,
                &format!("{prefix}.shared_experts"),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            num_experts_per_tok: top_k,
            routed_scaling_factor,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_shape = xs.dims().to_vec();
        let hidden_size = *original_shape
            .last()
            .expect("input must have at least 1 dim");
        let num_tokens = xs.elem_count() / hidden_size;

        // Flatten to [num_tokens, hidden_size]
        let xs_2d = xs.reshape((num_tokens, hidden_size))?;

        // Router: [num_tokens, num_experts]
        let router_logits = self.gate.forward(&xs_2d)?;
        let num_experts = self.experts.len();

        // Softmax over experts
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Top-K selection
        let mut output = Tensor::zeros_like(&xs_2d)?;

        for token_idx in 0..num_tokens {
            let token = xs_2d.i(token_idx)?.unsqueeze(0)?;
            let token_weights = routing_weights.i(token_idx)?;
            let token_weights_vec: Vec<f32> = token_weights.to_dtype(DType::F32)?.to_vec1()?;

            // Find top-K experts
            let mut expert_indices: Vec<(usize, f32)> = token_weights_vec
                .iter()
                .enumerate()
                .map(|(i, &w)| (i, w))
                .collect();
            expert_indices
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            expert_indices.truncate(self.num_experts_per_tok);

            // Renormalize weights
            let weight_sum: f32 = expert_indices.iter().map(|(_, w)| w).sum();
            let renorm_factor = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                1.0 / self.num_experts_per_tok as f32
            };

            let mut token_output = Tensor::zeros((1, hidden_size), xs_2d.dtype(), xs_2d.device())?;
            for &(expert_idx, weight) in &expert_indices {
                if expert_idx < num_experts {
                    let expert_out = self.experts[expert_idx].forward(&token)?;
                    let scaled = (expert_out * (weight as f64 * renorm_factor as f64))?;
                    token_output = (token_output + scaled)?;
                }
            }

            output =
                output.slice_assign(&[token_idx..token_idx + 1, 0..hidden_size], &token_output)?;
        }

        // Apply routed scaling factor
        let output = (output * self.routed_scaling_factor)?;

        // Add shared expert output
        let output = if let Some(shared) = &self.shared_experts {
            (output + shared.forward(&xs_2d)?)?
        } else {
            output
        };

        output.reshape(original_shape)
    }
}

// ─── Quantized MLA Attention ────────────────────────────────────────────────

/// Quantized MLA attention that uses QuantizedLinear for projections.
///
/// The KV latent compression and RoPE logic follow the same pattern as
/// the unquantized MLAAttention, but all linear projections are quantized.
/// RMSNorm layers remain unquantized.
struct QuantizedMLAAttention {
    // Query projections (low-rank path)
    q_a_proj: Option<Box<dyn QuantizedLinear>>,
    q_a_layernorm: Option<RmsNorm>,
    q_b_proj: Option<Box<dyn QuantizedLinear>>,
    // Query projection (direct path, for smaller models)
    q_proj: Option<Box<dyn QuantizedLinear>>,
    // KV projections
    kv_a_proj_with_mqa: Box<dyn QuantizedLinear>,
    kv_a_layernorm: RmsNorm,
    // NOTE: kv_b_proj stays as candle_nn::Linear because MLACacheEngine::read_expand_prefill
    // requires a &Linear reference for expanding the latent representation.
    kv_b_proj: Linear,
    // Output projection
    o_proj: Box<dyn QuantizedLinear>,
    // RoPE
    rotary_emb: RotaryEmbedding,
    // Config
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    head_dim: usize,
    q_scale: f64,
}

impl QuantizedMLAAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;

        let qk_nope_head_dim = cfg
            .extra
            .get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let qk_rope_head_dim = cfg
            .extra
            .get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let q_lora_rank = cfg
            .extra
            .get("q_lora_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let kv_lora_rank = cfg
            .extra
            .get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;

        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let vb_attn = vb.pp("self_attn");

        // Query projection (low-rank or direct)
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_rank) = q_lora_rank {
            let q_a = loader.load_linear(
                &format!("{prefix}.q_a_proj"),
                cfg.hidden_size,
                q_rank,
                false,
            )?;
            let q_a_ln = rms_norm(q_rank, cfg.rms_norm_eps, vb_attn.pp("q_a_layernorm"))?;
            let q_b = loader.load_linear(
                &format!("{prefix}.q_b_proj"),
                q_rank,
                num_heads * qk_head_dim,
                false,
            )?;
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q = loader.load_linear(
                &format!("{prefix}.q_proj"),
                cfg.hidden_size,
                num_heads * qk_head_dim,
                false,
            )?;
            (None, None, None, Some(q))
        };

        // KV projection -- quantized
        let kv_a_proj_with_mqa = loader.load_linear(
            &format!("{prefix}.kv_a_proj_with_mqa"),
            cfg.hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            false,
        )?;
        let kv_a_layernorm =
            rms_norm(kv_lora_rank, cfg.rms_norm_eps, vb_attn.pp("kv_a_layernorm"))?;

        // kv_b_proj stays as Linear for MLACacheEngine compatibility
        let kv_b_proj = linear_no_bias(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            vb_attn.pp("kv_b_proj"),
        )?;

        // Output projection -- quantized
        let o_proj = loader.load_linear(
            &format!("{prefix}.o_proj"),
            num_heads * v_head_dim,
            cfg.hidden_size,
            false,
        )?;

        // RoPE
        let rotary_emb = RotaryEmbedding::new(
            qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            loader.dtype(),
            loader.device(),
        )?;

        // YaRN mscale
        let mscale = cfg
            .extra
            .get("rope_scaling")
            .and_then(|v| v.get("mscale"))
            .and_then(|v| v.as_f64())
            .map(|mscale| {
                let factor = cfg
                    .extra
                    .get("rope_scaling")
                    .and_then(|v| v.get("factor"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                if factor <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * factor.ln() + 1.0
                }
            })
            .unwrap_or(1.0);

        let q_scale = mscale * mscale;

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            head_dim: qk_nope_head_dim + qk_rope_head_dim,
            q_scale,
        })
    }

    fn project_query(&self, x: &Tensor) -> Result<Tensor> {
        if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_latent = q_a.forward(x)?;
            let q_latent = q_a_ln.forward(&q_latent)?;
            q_b.forward(&q_latent)
        } else if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)
        } else {
            candle_core::bail!("No query projection configured")
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache: &mut MLACacheEngine,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Query projection
        let q = self.project_query(x)?;
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Split Q into nope and rope parts
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection through latent space
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize kv_a
        let kv_a = self.kv_a_layernorm.forward(&kv_a)?;

        // Write compressed latent to MLA cache
        let kv_c_for_cache = kv_a.reshape((batch_size * seq_len, self.kv_lora_rank))?;

        // Apply RoPE to k_pe
        let q_pe_for_rope = q_pe.transpose(1, 2)?;
        let k_pe_for_rope = k_pe_raw.reshape((batch_size, 1, seq_len, self.qk_rope_head_dim))?;
        let (q_pe_rotated, k_pe_rotated) =
            self.rotary_emb
                .apply(&q_pe_for_rope, &k_pe_for_rope, seqlen_offset)?;
        let q_pe = q_pe_rotated.transpose(1, 2)?;

        // Flatten k_pe for cache
        let k_pe_for_cache = k_pe_rotated
            .squeeze(1)?
            .transpose(0, 1)?
            .reshape((batch_size * seq_len, self.qk_rope_head_dim))?;

        // Write to MLA cache
        cache
            .write(&kv_c_for_cache, &k_pe_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache write: {e}")))?;

        // Read and expand from cache for attention
        let num_tokens = seqlen_offset + seq_len;
        let (k_nope_cached, k_pe_cached, v_cached) = cache
            .read_expand_prefill(block_ids, num_tokens, &self.kv_b_proj)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache read: {e}")))?;

        // Broadcast k_pe to all heads
        let k_pe_expanded = k_pe_cached
            .unsqueeze(1)?
            .broadcast_as((num_tokens, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Concatenate nope and rope parts for K
        let k_full = Tensor::cat(&[&k_nope_cached, &k_pe_expanded], D::Minus1)?;

        // Concatenate Q parts
        let q_full = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;

        // Apply q_scale
        let q_full = (q_full * self.q_scale)?;

        // Compute attention
        let attn_output = self.compute_attention(
            &q_full,
            &k_full,
            &v_cached,
            attention_mask,
            batch_size,
            seq_len,
            num_tokens,
        )?;

        // Output projection
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.v_head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        cache: &mut MLACacheEngine,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        debug_assert_eq!(seq_len, 1, "Decode expects single token");

        // Query projection
        let q = self.project_query(x)?;
        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;

        // Split Q
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection for new token
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize
        let kv_a = self.kv_a_layernorm.forward(&kv_a)?;

        // Apply RoPE
        let q_pe_for_rope = q_pe.transpose(1, 2)?;
        let k_pe_for_rope = k_pe_raw.reshape((batch_size, 1, 1, self.qk_rope_head_dim))?;
        let (q_pe_rotated, k_pe_rotated) =
            self.rotary_emb
                .apply(&q_pe_for_rope, &k_pe_for_rope, seqlen_offset)?;
        let q_pe = q_pe_rotated.transpose(1, 2)?;

        // Write new token to cache
        let kv_c_for_cache = kv_a.reshape((batch_size, self.kv_lora_rank))?;
        let k_pe_for_cache = k_pe_rotated.reshape((batch_size, self.qk_rope_head_dim))?;

        cache
            .write(&kv_c_for_cache, &k_pe_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache write: {e}")))?;

        // Read and expand from cache for attention
        let num_tokens = seqlen_offset + 1;
        let (k_nope_cached, k_pe_cached, v_cached) = cache
            .read_expand_prefill(block_ids, num_tokens, &self.kv_b_proj)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache read: {e}")))?;

        // Broadcast k_pe to all heads
        let k_pe_expanded = k_pe_cached
            .unsqueeze(1)?
            .broadcast_as((num_tokens, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Full K
        let k_full = Tensor::cat(&[&k_nope_cached, &k_pe_expanded], D::Minus1)?;

        // Full Q
        let q_full = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;
        let q_full = (q_full * self.q_scale)?;

        // Compute attention
        let attn_output =
            self.compute_attention(&q_full, &k_full, &v_cached, None, batch_size, 1, num_tokens)?;

        // Output projection
        let attn_output = attn_output.reshape((batch_size, 1, self.num_heads * self.v_head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        _batch_size: usize,
        _q_len: usize,
        _kv_len: usize,
    ) -> Result<Tensor> {
        // q: [batch, q_len, num_heads, head_dim]
        // k: [kv_len, num_heads, head_dim]
        // v: [kv_len, num_heads, v_head_dim]

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // Reshape for batch matmul
        let q = q.transpose(1, 2)?;
        let k = k.unsqueeze(0)?.transpose(1, 2)?;
        let v = v.unsqueeze(0)?.transpose(1, 2)?;

        // Attention scores: [batch, num_heads, q_len, kv_len]
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // Apply causal mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Output: [batch, num_heads, q_len, v_head_dim]
        let output = attn_weights.matmul(&v)?;

        // Reshape: [batch, q_len, num_heads, v_head_dim]
        output.transpose(1, 2)
    }
}

// ─── Quantized Decoder Layer ────────────────────────────────────────────────

struct QuantizedDeepSeekDecoderLayer {
    self_attn: QuantizedMLAAttention,
    mlp: Option<QuantizedDeepSeekMLP>,
    moe: Option<QuantizedDeepSeekMoELayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedDeepSeekDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        layer_idx: usize,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        let vb_layer = vb.pp("model").pp("layers").pp(layer_idx);

        let self_attn = QuantizedMLAAttention::new(
            cfg,
            loader,
            vb_layer.clone(),
            &format!("{prefix}.self_attn"),
        )?;

        // Layer norms
        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
        )?;

        // MoE setup: layer 0 uses dense MLP, other layers may use MoE
        let n_routed = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let is_moe = n_routed.is_some() && layer_idx > 0;

        let (mlp, moe) = if is_moe {
            let moe_layer = QuantizedDeepSeekMoELayer::new(
                cfg,
                loader,
                vb_layer.pp("mlp"),
                &format!("{prefix}.mlp"),
            )?;
            (None, Some(moe_layer))
        } else {
            let mlp = QuantizedDeepSeekMLP::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                loader,
                &format!("{prefix}.mlp"),
            )?;
            (Some(mlp), None)
        };

        Ok(Self {
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_prefill(
            &x,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.mla_engine_mut(layer_idx),
            block_ids,
            slot_mapping,
        )?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;

        let x = if let Some(moe) = &self.moe {
            moe.forward(&x)?
        } else if let Some(mlp) = &self.mlp {
            mlp.forward(&x)?
        } else {
            x
        };

        residual + x
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_decode(
            &x,
            seqlen_offset,
            kv_cache_mgr.mla_engine_mut(layer_idx),
            block_ids,
            slot_mapping,
        )?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;

        let x = if let Some(moe) = &self.moe {
            moe.forward(&x)?
        } else if let Some(mlp) = &self.mlp {
            mlp.forward(&x)?
        } else {
            x
        };

        residual + x
    }
}

// ─── Quantized Model ────────────────────────────────────────────────────────

/// Quantized DeepSeek V2/V3 model supporting FP8, GPTQ, AWQ, and unquantized weights.
///
/// Uses Multi-head Latent Attention (MLA) with compressed KV cache.
/// Router (gate) stays unquantized. MLA projections, MLP, and expert linear
/// layers use QuantizedLinear. RMSNorm and embeddings stay unquantized.
///
/// # Requirements
///
/// This model requires a KVCacheManager created with `new_mla()`:
///
/// ```ignore
/// let kv_cache_mgr = KVCacheManager::new_mla(&mla_config)?;
/// ```
pub struct QuantizedDeepSeekForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<QuantizedDeepSeekDecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl QuantizedDeepSeekForCausalLM {
    /// Create a new quantized DeepSeek model.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration (must have kv_lora_rank in extra)
    /// * `vb` - VarBuilder for loading non-quantized weights (embeddings, norms)
    /// * `weight_loader` - Quantized weight loader for linear layers
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedDeepSeekDecoderLayer::new(
                cfg,
                i,
                weight_loader,
                vb.clone(),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead {
                weight: embed_tokens.embeddings().clone(),
            }) as Box<dyn QuantizedLinear>
        } else {
            weight_loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper for tied embedding lm_head.
struct TiedEmbeddingHead {
    weight: Tensor,
}

impl QuantizedLinear for TiedEmbeddingHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match x.dims().len() {
            3 => self.weight.broadcast_left(x.dim(0)?)?,
            _ => self.weight.clone(),
        };
        x.matmul(&w.t()?)
    }

    fn load_weights(&mut self, _weights: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        self.weight.dtype()
    }

    fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    fn has_bias(&self) -> bool {
        false
    }
}

impl crate::engine::ModelForward for QuantizedDeepSeekForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mask = if seq_len > 1 {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                xs.dtype(),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table.block_ids(),
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);

        for (seq_idx, seq) in sequences.iter().enumerate() {
            let x = input_ids.i(seq_idx)?.unsqueeze(0)?;
            let mut xs = self.embed_tokens.forward(&x)?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                xs = layer.forward_decode(
                    &xs,
                    seq.seqlen_offset,
                    kv_cache_mgr,
                    layer_idx,
                    &seq.block_ids,
                    &seq.slot_mapping,
                )?;
            }

            let xs = self.norm.forward(&xs)?;
            let logits = self.lm_head.forward(&xs)?;
            outputs.push(logits.squeeze(0)?);
        }

        Tensor::stack(&outputs, 0)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::mla_cache_config::MLACacheConfig;
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "qk_nope_head_dim".into(),
            serde_json::Value::Number(16.into()),
        );
        extra.insert(
            "qk_rope_head_dim".into(),
            serde_json::Value::Number(8.into()),
        );
        extra.insert("v_head_dim".into(), serde_json::Value::Number(16.into()));
        extra.insert("kv_lora_rank".into(), serde_json::Value::Number(32.into()));

        ModelConfig {
            architectures: vec!["DeepseekV2ForCausalLM".to_string()],
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 24, // qk_nope + qk_rope
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn create_mla_cache_manager(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32, // kv_lora_rank
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            cfg.num_attention_heads,
            4,  // block_size
            16, // num_blocks
            cfg.num_hidden_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).expect("create MLA cache manager")
    }

    #[test]
    fn test_quantized_deepseek_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedDeepSeekForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedDeepSeekForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("construction succeeded");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_deepseek_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedDeepSeekForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);

        // Verify manager is MLA type
        assert!(kv_cache_mgr.is_mla());

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        let mut block_table = BlockTable::new(4);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..4).collect();

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
            &[1, 4, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }
}
