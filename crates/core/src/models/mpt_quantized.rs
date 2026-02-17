//! Quantized MPT (MosaicML Pretrained Transformer) model implementation.
//!
//! MPT-specific features preserved in the quantized path:
//! - ALiBi positional encoding (no RoPE, no learned position embeddings)
//! - Fused QKV projection (attn.Wqkv), split into Q/K/V after forward
//! - LayerNorm with configurable layer_norm_epsilon
//! - GELU MLP (up_proj → GELU → down_proj)
//! - NO bias on any projections (attention or MLP)
//! - MHA (num_kv_heads == num_q_heads)
//! - Tied word embeddings by default
//! - Weight prefix: transformer.blocks.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

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

// ─── Config Helpers ──────────────────────────────────────────────────────────

fn get_layer_norm_epsilon(cfg: &ModelConfig) -> f64 {
    cfg.extra
        .get("layer_norm_epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5)
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedMptAttention {
    wqkv: Box<dyn QuantizedLinear>,
    out_proj: Box<dyn QuantizedLinear>,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    head_dim: usize,
}

impl QuantizedMptAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;

        // Fused QKV (no bias)
        let wqkv = loader.load_linear(
            &format!("{}.Wqkv", vb.prefix()),
            cfg.hidden_size,
            cfg.hidden_size * 3,
            false,
        )?;

        let out_proj = loader.load_linear(
            &format!("{}.out_proj", vb.prefix()),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;

        let alibi = AlibiAttentionBias::new(num_heads, vb.dtype(), vb.device())?;

        Ok(Self {
            wqkv,
            out_proj,
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

        let qkv = self.wqkv.forward(xs)?;

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
            self.num_heads,
            self.head_dim,
        )?;

        self.out_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.wqkv.forward(xs)?;

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
        self.out_proj.forward(&attn_output)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct QuantizedMptMlp {
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedMptMlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        // No bias on MLP projections
        let up_proj = loader.load_linear(
            &format!("{}.up_proj", vb.prefix()),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let down_proj = loader.load_linear(
            &format!("{}.down_proj", vb.prefix()),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
        )?;

        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.up_proj.forward(xs)?;
        let hidden = hidden.gelu_erf()?;
        self.down_proj.forward(&hidden)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct QuantizedMptDecoderLayer {
    norm_1: LayerNorm,
    attn: QuantizedMptAttention,
    norm_2: LayerNorm,
    ffn: QuantizedMptMlp,
}

impl QuantizedMptDecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let eps = get_layer_norm_epsilon(cfg);

        let norm_1 = layer_norm(cfg.hidden_size, eps, vb.pp("norm_1"))?;
        let attn = QuantizedMptAttention::new(cfg, vb.pp("attn"), loader)?;
        let norm_2 = layer_norm(cfg.hidden_size, eps, vb.pp("norm_2"))?;
        let ffn = QuantizedMptMlp::new(cfg, vb.pp("ffn"), loader)?;

        Ok(Self {
            norm_1,
            attn,
            norm_2,
            ffn,
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
        let xs = self.norm_1.forward(xs)?;
        let attn_output = self.attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.norm_2.forward(&xs)?;
        let mlp_output = self.ffn.forward(&xs)?;
        residual + mlp_output
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm_1.forward(xs)?;
        let attn_output =
            self.attn
                .forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.norm_2.forward(&xs)?;
        let mlp_output = self.ffn.forward(&xs)?;
        residual + mlp_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedMptForCausalLM {
    wte: candle_nn::Embedding,
    blocks: Vec<QuantizedMptDecoderLayer>,
    norm_f: LayerNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedMptForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let eps = get_layer_norm_epsilon(cfg);
        let vb_t = vb.pp("transformer");

        let wte = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"))?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_t.pp("blocks");
        for i in 0..cfg.num_hidden_layers {
            blocks.push(QuantizedMptDecoderLayer::new(cfg, vb_b.pp(i), loader)?);
        }

        let norm_f = layer_norm(cfg.hidden_size, eps, vb_t.pp("norm_f"))?;

        // MPT uses tied embeddings by default
        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(wte.embeddings().clone()))
        } else {
            loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            wte,
            blocks,
            norm_f,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedMptForCausalLM {
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

        let mut xs = self.wte.forward(input_ids)?;

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
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

        let xs = self.norm_f.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.wte.forward(input_ids)?;

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.norm_f.forward(&xs)?;
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
            architectures: vec!["MptForCausalLM".to_string()],
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
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_quantized_mpt_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedMptForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedMptForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().blocks.len(), 2);
    }

    #[test]
    fn test_quantized_mpt_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedMptForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");

        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let logits = crate::engine::ModelForward::forward(
            &model,
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
    fn test_quantized_mpt_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedMptForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Untied MPT should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_mpt_gptq_loader() {
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

        let model = QuantizedMptForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "GPTQ MPT should construct: {:?}",
            model.err()
        );
    }
}
