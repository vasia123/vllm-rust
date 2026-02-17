//! Quantized Bloom model implementation.
//!
//! Bloom-specific features preserved in the quantized path:
//! - ALiBi positional encoding (no RoPE or learned position embeddings)
//! - Fused QKV projection (query_key_value), split into Q/K/V after forward
//! - LayerNorm (not RMSNorm) with configurable layer_norm_epsilon
//! - GELU MLP (dense_h_to_4h → GELU → dense_4h_to_h)
//! - word_embeddings_layernorm after embedding layer
//! - All linears have bias=true
//! - MHA (num_kv_heads == num_q_heads)
//! - Weight prefix: transformer.h.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, AlibiAttentionBias};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Tied Embedding Head ─────────────────────────────────────────────────────

struct TiedEmbeddingHead {
    weight: Tensor,
}

impl TiedEmbeddingHead {
    fn new(weight: Tensor) -> Self {
        Self { weight }
    }
}

impl QuantizedLinear for TiedEmbeddingHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match x.dims().len() {
            3 => self.weight.broadcast_left(x.dim(0)?)?,
            _ => self.weight.clone(),
        };
        x.matmul(&w.t()?)
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
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

// ─── Config Extraction ───────────────────────────────────────────────────────

fn get_layer_norm_epsilon(cfg: &ModelConfig) -> f64 {
    cfg.extra
        .get("layer_norm_epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5)
}

// ─── Quantized Attention ─────────────────────────────────────────────────────

struct QuantizedBloomAttention {
    query_key_value: Box<dyn QuantizedLinear>,
    dense: Box<dyn QuantizedLinear>,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    head_dim: usize,
}

impl QuantizedBloomAttention {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;

        // Fused QKV: [hidden_size, 3 * hidden_size]
        let query_key_value = loader.load_linear(
            &format!("{prefix}.query_key_value"),
            cfg.hidden_size,
            cfg.hidden_size * 3,
            true,
        )?;

        let dense = loader.load_linear(
            &format!("{prefix}.dense"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;

        let alibi = AlibiAttentionBias::new(num_heads, loader.dtype(), loader.device())?;

        Ok(Self {
            query_key_value,
            dense,
            alibi,
            num_heads,
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

        let qkv = self.query_key_value.forward(xs)?;

        // Split fused QKV
        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // ALiBi bias
        let kv_len = q_len + seqlen_offset;
        let alibi_bias = self.alibi.build_bias_matrix(q_len, kv_len)?;

        let combined_mask = if let Some(causal_mask) = attention_mask {
            Some(causal_mask.broadcast_add(&alibi_bias)?)
        } else {
            Some(alibi_bias)
        };

        let attn_output = paged_attention(
            &q,
            &k,
            &v,
            combined_mask.as_ref(),
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_heads, // MHA
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

        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

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

            let attn_output = crate::cuda_kernels::paged_attention_cuda_alibi(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
                self.alibi.slopes(),
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

                let kv_len = seq.seqlen_offset + 1;
                let alibi_bias = self.alibi.build_bias_matrix(1, kv_len)?;

                let attn_out = paged_attention(
                    &q_i,
                    &k_i,
                    &v_i,
                    Some(&alibi_bias),
                    seq.seqlen_offset,
                    cache_engine,
                    &seq.block_ids,
                    &seq.slot_mapping,
                    self.num_heads,
                    self.num_heads,
                    self.head_dim,
                )?;
                outputs.push(attn_out);
            }

            let attn_output = Tensor::cat(&outputs, 0)?;
            self.dense.forward(&attn_output)
        }
    }
}

// ─── Quantized MLP ───────────────────────────────────────────────────────────

struct QuantizedBloomMlp {
    dense_h_to_4h: Box<dyn QuantizedLinear>,
    dense_4h_to_h: Box<dyn QuantizedLinear>,
}

impl QuantizedBloomMlp {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let dense_h_to_4h = loader.load_linear(
            &format!("{prefix}.dense_h_to_4h"),
            cfg.hidden_size,
            cfg.intermediate_size,
            true,
        )?;
        let dense_4h_to_h = loader.load_linear(
            &format!("{prefix}.dense_4h_to_h"),
            cfg.intermediate_size,
            cfg.hidden_size,
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

// ─── Quantized Decoder Layer ─────────────────────────────────────────────────

struct QuantizedBloomDecoderLayer {
    input_layernorm: LayerNorm,
    self_attention: QuantizedBloomAttention,
    post_attention_layernorm: LayerNorm,
    mlp: QuantizedBloomMlp,
}

impl QuantizedBloomDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        vb: &VarBuilder,
        loader: &dyn QuantizedWeightLoader,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("transformer.h.{layer_idx}");
        let ln_eps = get_layer_norm_epsilon(cfg);

        let input_layernorm = layer_norm(
            cfg.hidden_size,
            ln_eps,
            vb.pp(format!("{prefix}.input_layernorm")),
        )?;

        let self_attention =
            QuantizedBloomAttention::new(cfg, loader, &format!("{prefix}.self_attention"))?;

        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            ln_eps,
            vb.pp(format!("{prefix}.post_attention_layernorm")),
        )?;

        let mlp = QuantizedBloomMlp::new(cfg, loader, &format!("{prefix}.mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
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
        let attn_output = self.self_attention.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
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
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

// ─── Quantized Model ─────────────────────────────────────────────────────────

pub struct QuantizedBloomForCausalLM {
    word_embeddings: Embedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<QuantizedBloomDecoderLayer>,
    ln_f: LayerNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedBloomForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let ln_eps = get_layer_norm_epsilon(cfg);

        let word_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("transformer.word_embeddings"),
        )?;

        let word_embeddings_layernorm = layer_norm(
            cfg.hidden_size,
            ln_eps,
            vb.pp("transformer.word_embeddings_layernorm"),
        )?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            h.push(QuantizedBloomDecoderLayer::new(cfg, &vb, loader, i)?);
        }

        let ln_f = layer_norm(cfg.hidden_size, ln_eps, vb.pp("transformer.ln_f"))?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(word_embeddings.embeddings().clone()))
        } else {
            loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            word_embeddings,
            word_embeddings_layernorm,
            h,
            ln_f,
            lm_head,
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

        let mut xs = self.word_embeddings.forward(input_ids)?;
        xs = self.word_embeddings_layernorm.forward(&xs)?;

        for (layer_idx, layer) in self.h.iter().enumerate() {
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

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedBloomForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedBloomForCausalLM::forward(
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
        let mut xs = self.word_embeddings.forward(input_ids)?;
        xs = self.word_embeddings_layernorm.forward(&xs)?;

        for (layer_idx, layer) in self.h.iter().enumerate() {
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
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()),
        );

        ModelConfig {
            architectures: vec!["BloomForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
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
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_quantized_bloom_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedBloomForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Failed: {:?}", model.err());
        assert_eq!(model.unwrap().h.len(), 2);
    }

    #[test]
    fn test_quantized_bloom_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedBloomForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 5)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 5);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[1, 5, 256]);
    }

    #[test]
    fn test_quantized_bloom_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedBloomForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Failed: {:?}", model.err());
    }

    #[test]
    fn test_quantized_bloom_gptq_loader() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig {
            method: crate::quantization::QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            ..Default::default()
        };
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedBloomForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "GPTQ loader failed: {:?}", model.err());
    }
}
