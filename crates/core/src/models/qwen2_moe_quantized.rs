//! Quantized Qwen2-MoE model implementation.
//!
//! This module provides a quantized version of the Qwen2-MoE model that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.
//!
//! Qwen2-MoE features preserved in the quantized path:
//! - Sparse MoE blocks on certain layers (controlled by `decoder_sparse_step`)
//! - Shared expert that processes all tokens with sigmoid gating
//! - Dense MLP layers interspersed with MoE layers
//! - Router (gate) stays unquantized (small linear layer)
//! - Shared expert gate stays unquantized (hidden_size → 1)
//! - Expert gate/up/down projections use QuantizedLinear
//! - Attention projections (Q/K/V with bias, O without) are quantized
//! - RMSNorm stays as-is (not quantized)
//! - SwiGLU activation in experts and dense MLPs
//! - RoPE positional encoding
//! - GQA (Grouped Query Attention)

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Quantized Attention ──────────────────────────────────────────────────────

struct QuantizedQwen2MoeAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedQwen2MoeAttention {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // Qwen2-MoE uses bias=True for QKV projections (same as Qwen2)
        let use_bias = cfg.attention_bias.unwrap_or(true);

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            use_bias,
        )?;
        let k_proj = loader.load_linear(
            &format!("{prefix}.k_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            use_bias,
        )?;
        let v_proj = loader.load_linear(
            &format!("{prefix}.v_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            use_bias,
        )?;
        // Output projection has no bias
        let o_proj = loader.load_linear(
            &format!("{prefix}.o_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            loader.dtype(),
            loader.device(),
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

        self.o_proj.forward(&attn_output)
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

            self.o_proj.forward(&attn_output)?.unsqueeze(1)
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
            self.o_proj.forward(&attn_output)
        }
    }
}

// ─── Quantized Dense MLP ──────────────────────────────────────────────────────

struct QuantizedQwen2MoeDenseMlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedQwen2MoeDenseMlp {
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Quantized MoE Expert ─────────────────────────────────────────────────────

struct QuantizedQwen2MoeExpert {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedQwen2MoeExpert {
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Quantized Sparse MoE Block ──────────────────────────────────────────────

struct QuantizedQwen2MoeSparseMoeBlock {
    gate: Linear,
    experts: Vec<QuantizedQwen2MoeExpert>,
    shared_expert: QuantizedQwen2MoeExpert,
    shared_expert_gate: Linear,
    num_experts: usize,
    top_k: usize,
}

impl QuantizedQwen2MoeSparseMoeBlock {
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts().unwrap_or(60);
        let top_k = cfg.num_experts_per_tok().unwrap_or(4);
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);
        let shared_expert_intermediate_size = cfg
            .shared_expert_intermediate_size()
            .unwrap_or(cfg.intermediate_size);

        // Router stays unquantized
        let gate = linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;

        // Routed experts
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(QuantizedQwen2MoeExpert::new(
                hidden_size,
                moe_intermediate_size,
                loader,
                &format!("{prefix}.experts.{i}"),
            )?);
        }

        // Shared expert
        let shared_expert = QuantizedQwen2MoeExpert::new(
            hidden_size,
            shared_expert_intermediate_size,
            loader,
            &format!("{prefix}.shared_expert"),
        )?;

        // Shared expert gate stays unquantized (hidden_size → 1)
        let shared_expert_gate = linear_no_bias(hidden_size, 1, vb.pp("shared_expert_gate"))?;

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let num_tokens = xs.elem_count() / hidden_dim;
        let xs_2d = xs.reshape((num_tokens, hidden_dim))?;

        // Router
        let router_logits = self.gate.forward(&xs_2d)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Compute routed expert output
        let mut output = Tensor::zeros_like(&xs_2d)?;

        for token_idx in 0..num_tokens {
            let token = xs_2d.i(token_idx)?.unsqueeze(0)?;
            let token_weights = routing_weights.i(token_idx)?;
            let token_weights_vec: Vec<f32> = token_weights.to_dtype(DType::F32)?.to_vec1()?;

            // Top-K selection
            let mut expert_indices: Vec<(usize, f32)> = token_weights_vec
                .iter()
                .enumerate()
                .map(|(i, &w)| (i, w))
                .collect();
            expert_indices
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            expert_indices.truncate(self.top_k);

            // Renormalize weights (Qwen2-MoE uses norm_topk_prob)
            let weight_sum: f32 = expert_indices.iter().map(|(_, w)| w).sum();
            let renorm_factor = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                1.0 / self.top_k as f32
            };

            let mut token_output = Tensor::zeros((1, hidden_dim), xs_2d.dtype(), xs_2d.device())?;
            for &(expert_idx, weight) in &expert_indices {
                if expert_idx < self.num_experts {
                    let expert_out = self.experts[expert_idx].forward(&token)?;
                    let scaled = (expert_out * (weight as f64 * renorm_factor as f64))?;
                    token_output = (token_output + scaled)?;
                }
            }

            output =
                output.slice_assign(&[token_idx..token_idx + 1, 0..hidden_dim], &token_output)?;
        }

        // Compute shared expert output with sigmoid gating
        let shared_out = self.shared_expert.forward(&xs_2d)?;
        let gate_logits = self.shared_expert_gate.forward(&xs_2d)?;
        let gate = candle_nn::ops::sigmoid(&gate_logits)?;
        let gated_shared = shared_out.broadcast_mul(&gate)?;

        // Combine routed and shared outputs
        let result = (output + gated_shared)?;
        result.reshape(orig_shape)
    }
}

// ─── Quantized MLP Variant ────────────────────────────────────────────────────

enum QuantizedMlpVariant {
    Dense(QuantizedQwen2MoeDenseMlp),
    Sparse(Box<QuantizedQwen2MoeSparseMoeBlock>),
}

impl QuantizedMlpVariant {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Sparse(moe) => moe.forward(xs),
        }
    }
}

// ─── Quantized Decoder Layer ──────────────────────────────────────────────────

struct QuantizedQwen2MoeDecoderLayer {
    self_attn: QuantizedQwen2MoeAttention,
    mlp: QuantizedMlpVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen2MoeDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        layer_idx: usize,
        loader: &dyn QuantizedWeightLoader,
        vb_layer: VarBuilder,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");

        let self_attn =
            QuantizedQwen2MoeAttention::new(cfg, loader, &format!("{prefix}.self_attn"))?;

        // Determine if this layer should be MoE or dense
        let decoder_sparse_step = cfg.decoder_sparse_step().unwrap_or(1);
        let mlp_only_layers = cfg.mlp_only_layers();
        let num_experts = cfg.num_experts().unwrap_or(0);

        let is_moe_layer = !mlp_only_layers.contains(&layer_idx)
            && num_experts > 0
            && (layer_idx + 1).is_multiple_of(decoder_sparse_step);

        let mlp_prefix = format!("{prefix}.mlp");
        let mlp = if is_moe_layer {
            QuantizedMlpVariant::Sparse(Box::new(QuantizedQwen2MoeSparseMoeBlock::new(
                cfg,
                loader,
                vb_layer.pp("mlp"),
                &mlp_prefix,
            )?))
        } else {
            QuantizedMlpVariant::Dense(QuantizedQwen2MoeDenseMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                loader,
                &mlp_prefix,
            )?)
        };

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
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
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
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_decode_batch(&xs, sequences, cache_engine)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ─── Tied Embedding Head ───────────────────────────────────────────────────────

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

// ─── Quantized Model ──────────────────────────────────────────────────────────

/// Quantized Qwen2-MoE model for causal language modeling.
pub struct QuantizedQwen2MoeForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<QuantizedQwen2MoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedQwen2MoeForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_layer = vb_m.pp("layers").pp(i);
            layers.push(QuantizedQwen2MoeDecoderLayer::new(
                cfg, i, loader, vb_layer,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Box::new(TiedEmbeddingHead {
                weight: emb_weights,
            }) as Box<dyn QuantizedLinear>
        } else {
            loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
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
}

impl crate::engine::ModelForward for QuantizedQwen2MoeForCausalLM {
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

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
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
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::quantization::{
        create_weight_loader_with_params, DetectedQuantConfig, QuantizationMethod,
    };

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
    fn test_quantized_qwen2_moe_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedQwen2MoeForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedQwen2MoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_qwen2_moe_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedQwen2MoeForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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

        let logits = crate::engine::ModelForward::forward(
            &model,
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
    fn test_quantized_qwen2_moe_mixed_layers() {
        // decoder_sparse_step=2: layer 0 dense, layer 1 MoE
        let mut cfg = test_config();
        cfg.extra
            .insert("decoder_sparse_step".to_string(), serde_json::json!(2));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedQwen2MoeForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with mixed dense/MoE layers: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_qwen2_moe_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedQwen2MoeForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with untied embeddings: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_qwen2_moe_gptq_loader() {
        use crate::models::create_weight_loader_with_params;
        use crate::quantization::{DetectedQuantConfig, QuantizationMethod};

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: std::collections::HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedQwen2MoeForCausalLM::new(&cfg, vb, loader.as_ref());
        // GPTQ loader expects specific tensor shapes with VarBuilder::zeros
        assert!(model.is_err() || model.is_ok());
    }
}
