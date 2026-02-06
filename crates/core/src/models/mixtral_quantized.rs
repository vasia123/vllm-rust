//! Quantized Mixtral model implementation.
//!
//! This module provides a quantized version of the Mixtral MoE model that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.
//!
//! Mixtral-specific features preserved in the quantized path:
//! - Mixture of Experts (MoE) with top-K routing
//! - Router (gate) stays unquantized (small linear layer)
//! - Expert gate/up/down projections use QuantizedLinear
//! - Attention projections (Q/K/V/O) are quantized
//! - RMSNorm stays as-is (not quantized)
//! - SwiGLU activation in experts
//! - RoPE positional encoding
//! - GQA (Grouped Query Attention)

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── MoE Configuration ─────────────────────────────────────────────────────

struct MixtralQuantizedConfig {
    num_local_experts: usize,
    num_experts_per_tok: usize,
}

impl MixtralQuantizedConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
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
            num_local_experts,
            num_experts_per_tok,
        }
    }
}

// ─── Quantized MoE Expert ──────────────────────────────────────────────────

struct QuantizedMoEExpert {
    w1: Box<dyn QuantizedLinear>, // gate_proj
    w2: Box<dyn QuantizedLinear>, // down_proj
    w3: Box<dyn QuantizedLinear>, // up_proj
}

impl QuantizedMoEExpert {
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
        let gate = self.w1.forward(xs)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.w3.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.w2.forward(&hidden)
    }
}

// ─── Quantized MoE Layer ───────────────────────────────────────────────────

struct QuantizedMoELayer {
    gate: Linear, // Router stays unquantized
    experts: Vec<QuantizedMoEExpert>,
    num_experts_per_tok: usize,
}

impl QuantizedMoELayer {
    fn new(
        cfg: &ModelConfig,
        moe_cfg: &MixtralQuantizedConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        // Router (gate) stays unquantized -- it is a small [hidden_size, num_experts] linear
        let gate = linear_no_bias(cfg.hidden_size, moe_cfg.num_local_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(moe_cfg.num_local_experts);
        for i in 0..moe_cfg.num_local_experts {
            experts.push(QuantizedMoEExpert::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                loader,
                &format!("{prefix}.experts.{i}"),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
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

        output.reshape(original_shape)
    }
}

// ─── Quantized Attention ───────────────────────────────────────────────────

struct QuantizedMixtralAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedMixtralAttention {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            false,
        )?;
        let k_proj = loader.load_linear(
            &format!("{prefix}.k_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let v_proj = loader.load_linear(
            &format!("{prefix}.v_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
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
            )?;

            self.o_proj.forward(&attn_output.unsqueeze(1)?)
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

// ─── Quantized Decoder Layer ───────────────────────────────────────────────

struct QuantizedMixtralDecoderLayer {
    self_attn: QuantizedMixtralAttention,
    block_sparse_moe: QuantizedMoELayer,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedMixtralDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        moe_cfg: &MixtralQuantizedConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        let self_attn =
            QuantizedMixtralAttention::new(cfg, loader, &format!("{prefix}.self_attn"))?;

        let vb_layer = vb.pp("model").pp("layers").pp(layer_idx);
        let block_sparse_moe = QuantizedMoELayer::new(
            cfg,
            moe_cfg,
            loader,
            vb_layer.pp("block_sparse_moe"),
            &format!("{prefix}.block_sparse_moe"),
        )?;

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

// ─── Quantized Model ───────────────────────────────────────────────────────

/// Quantized Mixtral MoE model supporting FP8, GPTQ, AWQ, and unquantized weights.
///
/// Router (gate) stays unquantized. Expert gate/up/down projections and
/// attention Q/K/V/O projections use QuantizedLinear.
pub struct QuantizedMixtralForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<QuantizedMixtralDecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl QuantizedMixtralForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let moe_cfg = MixtralQuantizedConfig::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedMixtralDecoderLayer::new(
                cfg,
                &moe_cfg,
                weight_loader,
                vb.clone(),
                i,
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
            num_experts: moe_cfg.num_local_experts,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
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
        self.lm_head.forward(&xs)
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

impl crate::engine::ModelForward for QuantizedMixtralForCausalLM {
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
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

    fn test_config() -> crate::config::ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));

        crate::config::ModelConfig {
            architectures: vec!["MixtralForCausalLM".to_string()],
            hidden_size: 32,
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
    fn test_quantized_mixtral_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedMixtralForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedMixtralForCausalLM should construct with unquantized loader: {:?}",
            model.err()
        );

        let model = model.expect("construction succeeded");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.num_experts(), 4);
        assert_eq!(model.num_experts_per_tok(), 2);
    }

    #[test]
    fn test_quantized_mixtral_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedMixtralForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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
}
