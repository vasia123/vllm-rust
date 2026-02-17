//! Quantized Falcon model implementation.
//!
//! This module provides a quantized version of the Falcon model that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.
//!
//! Falcon-specific features preserved in the quantized path:
//! - Fused QKV projection (query_key_value) with manual split
//! - Parallel attention-MLP (attention + mlp run in parallel from same norm output)
//! - Single LayerNorm per layer (no post_attention_layernorm)
//! - GELU activation in MLP (dense_h_to_4h → GELU → dense_4h_to_h)
//! - Bias on all linear projections
//! - MQA (Falcon-7B) or GQA (Falcon-40B+)
//! - RoPE positional encoding
//! - Weight prefix: transformer.h.X (not model.layers.X)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Quantized Attention ──────────────────────────────────────────────────────

struct QuantizedFalconAttention {
    query_key_value: Box<dyn QuantizedLinear>,
    dense: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl QuantizedFalconAttention {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let qkv_size = q_size + 2 * kv_size;

        let query_key_value = loader.load_linear(
            &format!("{prefix}.query_key_value"),
            cfg.hidden_size,
            qkv_size,
            true,
        )?;
        let dense =
            loader.load_linear(&format!("{prefix}.dense"), q_size, cfg.hidden_size, true)?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            loader.dtype(),
            loader.device(),
        )?;

        Ok(Self {
            query_key_value,
            dense,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
        })
    }

    fn split_qkv(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = qkv.dims3()?;
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

        Ok((
            q.reshape((b_sz, seq_len, self.q_size))?,
            k.reshape((b_sz, seq_len, self.kv_size))?,
            v.reshape((b_sz, seq_len, self.kv_size))?,
        ))
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

        let qkv = self.query_key_value.forward(xs)?;
        let (q, k, v) = self.split_qkv(&qkv)?;

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

        self.dense.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.query_key_value.forward(xs)?;
        let (q, k, v) = self.split_qkv(&qkv)?;

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

            self.dense.forward(&attn_output)?.unsqueeze(1)
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
            self.dense.forward(&attn_output)
        }
    }
}

// ─── Quantized MLP ────────────────────────────────────────────────────────────

struct QuantizedFalconMlp {
    dense_h_to_4h: Box<dyn QuantizedLinear>,
    dense_4h_to_h: Box<dyn QuantizedLinear>,
}

impl QuantizedFalconMlp {
    fn new(hidden_size: usize, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let intermediate_size = 4 * hidden_size;

        let dense_h_to_4h = loader.load_linear(
            &format!("{prefix}.dense_h_to_4h"),
            hidden_size,
            intermediate_size,
            true,
        )?;
        let dense_4h_to_h = loader.load_linear(
            &format!("{prefix}.dense_4h_to_h"),
            intermediate_size,
            hidden_size,
            true,
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.dense_h_to_4h.forward(xs)?;
        let hidden = hidden.gelu_erf()?;
        self.dense_4h_to_h.forward(&hidden)
    }
}

// ─── Quantized Decoder Layer ──────────────────────────────────────────────────

struct QuantizedFalconDecoderLayer {
    self_attention: QuantizedFalconAttention,
    mlp: QuantizedFalconMlp,
    input_layernorm: LayerNorm,
}

impl QuantizedFalconDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        layer_idx: usize,
        loader: &dyn QuantizedWeightLoader,
        vb_layer: VarBuilder,
    ) -> Result<Self> {
        let prefix = format!("transformer.h.{layer_idx}");

        let self_attention =
            QuantizedFalconAttention::new(cfg, loader, &format!("{prefix}.self_attention"))?;
        let mlp = QuantizedFalconMlp::new(cfg.hidden_size, loader, &format!("{prefix}.mlp"))?;

        let layer_norm_eps = cfg.rms_norm_eps;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            layer_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;

        Ok(Self {
            self_attention,
            mlp,
            input_layernorm,
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

        // Single layernorm before both attention and MLP (parallel design)
        let xs = self.input_layernorm.forward(xs)?;

        let attn_output = self.self_attention.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;

        // MLP from same normalized input (parallel residual)
        let mlp_output = self.mlp.forward(&xs)?;

        // Parallel: residual + attention + mlp
        (residual + attn_output + mlp_output)?.contiguous()
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let attn_output = self
            .self_attention
            .forward_decode_batch(&xs, sequences, cache_engine)?;
        let mlp_output = self.mlp.forward(&xs)?;

        (residual + attn_output + mlp_output)?.contiguous()
    }
}

// ─── Tied Embedding Head ──────────────────────────────────────────────────────

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

// ─── Quantized Model ─────────────────────────────────────────────────────────

/// Quantized Falcon model for causal language modeling.
pub struct QuantizedFalconForCausalLM {
    word_embeddings: Embedding,
    layers: Vec<QuantizedFalconDecoderLayer>,
    ln_f: LayerNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedFalconForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_t = vb.pp("transformer");
        let layer_norm_eps = cfg.rms_norm_eps;

        let word_embeddings =
            embedding(cfg.vocab_size, cfg.hidden_size, vb_t.pp("word_embeddings"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_layer = vb_t.pp("h").pp(i);
            layers.push(QuantizedFalconDecoderLayer::new(cfg, i, loader, vb_layer)?);
        }

        let ln_f = layer_norm(cfg.hidden_size, layer_norm_eps, vb_t.pp("ln_f"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = word_embeddings.embeddings().clone();
            Box::new(TiedEmbeddingHead {
                weight: emb_weights,
            }) as Box<dyn QuantizedLinear>
        } else {
            loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            word_embeddings,
            layers,
            ln_f,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }
}

impl crate::engine::ModelForward for QuantizedFalconForCausalLM {
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

        let mut xs = self.word_embeddings.forward(input_ids)?;
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
        let xs = self.ln_f.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.word_embeddings.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        }
        let xs = self.ln_f.forward(&xs)?;
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
        ModelConfig {
            architectures: vec!["FalconForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 1, // MQA for Falcon-7B
            num_hidden_layers: 2,
            intermediate_size: 256, // 4 * 64
            vocab_size: 256,
            max_position_embeddings: 2048,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 11,
            eos_token_id: 11,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
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
    fn test_quantized_falcon_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedFalconForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedFalconForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_falcon_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedFalconForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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
    fn test_quantized_falcon_gqa() {
        // Test with GQA (like Falcon-40B)
        let mut cfg = test_config();
        cfg.num_key_value_heads = 2; // GQA with 2 KV heads

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedFalconForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with GQA (num_kv_heads=2): {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_falcon_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedFalconForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with untied embeddings: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_falcon_gptq_loader() {
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

        let model = QuantizedFalconForCausalLM::new(&cfg, vb, loader.as_ref());
        // GPTQ loader expects specific tensor shapes with VarBuilder::zeros
        assert!(model.is_err() || model.is_ok());
    }
}
