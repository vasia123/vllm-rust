//! Quantized Apertus model implementation (Swiss AI Initiative).
//!
//! Apertus uses:
//! - XIELU activation (x * (sigmoid(x) + eps * x + beta)) with learned eps/beta params
//! - QK normalization via per-head RMSNorm
//! - No gate_proj (single up_proj + XIELU + down_proj)
//! - RoPE, GQA, no bias on projections
//! - Weight prefix: model.layers.X
//! - Norm names: attention_layernorm, feedforward_layernorm (not input_layernorm)

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
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

// ─── XIELU Activation ───────────────────────────────────────────────────────
// xIELU(x) = x * (sigmoid(x) + eps * x + beta)
// eps and beta are learned parameters loaded from weights.

struct Xielu {
    eps: Tensor,
    beta: Tensor,
}

impl Xielu {
    fn new(vb: VarBuilder) -> Result<Self> {
        let eps = vb.get(1, "eps")?;
        let beta = vb.get(1, "beta")?;
        Ok(Self { eps, beta })
    }
}

impl Module for Xielu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let sigmoid = candle_nn::ops::sigmoid(xs)?;
        let eps_x = xs.broadcast_mul(&self.eps)?;
        let gate = (sigmoid + eps_x)?.broadcast_add(&self.beta)?;
        xs.mul(&gate)
    }
}

// ─── XIELU MLP ──────────────────────────────────────────────────────────────

struct QuantizedXieluMlp {
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
    act_fn: Xielu,
}

impl QuantizedXieluMlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
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
        let act_fn = Xielu::new(vb.pp("act_fn"))?;

        Ok(Self {
            up_proj,
            down_proj,
            act_fn,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.up_proj.forward(xs)?;
        let h = self.act_fn.forward(&h)?;
        self.down_proj.forward(&h)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct QuantizedApertusAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedApertusAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = loader.load_linear(
            &format!("{}.q_proj", vb.prefix()),
            cfg.hidden_size,
            num_heads * head_dim,
            false,
        )?;
        let k_proj = loader.load_linear(
            &format!("{}.k_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let v_proj = loader.load_linear(
            &format!("{}.v_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let o_proj = loader.load_linear(
            &format!("{}.o_proj", vb.prefix()),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

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
            q_norm,
            k_norm,
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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct QuantizedApertusDecoderLayer {
    self_attn: QuantizedApertusAttention,
    mlp: QuantizedXieluMlp,
    attention_layernorm: RmsNorm,
    feedforward_layernorm: RmsNorm,
}

impl QuantizedApertusDecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let self_attn = QuantizedApertusAttention::new(cfg, vb.pp("self_attn"), loader)?;
        let mlp = QuantizedXieluMlp::new(cfg, vb.pp("mlp"), loader)?;
        let attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("attention_layernorm"),
        )?;
        let feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("feedforward_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            attention_layernorm,
            feedforward_layernorm,
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
        let xs = self.attention_layernorm.forward(xs)?;
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
        let xs = self
            .mlp
            .forward(&self.feedforward_layernorm.forward(&xs)?)?;
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
        let xs = self.attention_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.feedforward_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct QuantizedApertusForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedApertusDecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedApertusForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedApertusDecoderLayer::new(cfg, vb_l.pp(i), loader)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(embed_tokens.embeddings().clone()))
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

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedApertusForCausalLM {
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
                kv_cache_mgr,
                layer_idx,
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
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::*;
    use crate::config::ModelConfig;
    use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheManager};

    fn apertus_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["ApertusForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            hidden_act: "xielu".to_string(),
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_quantized_apertus_construction() {
        let cfg = apertus_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig::default();
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedApertusForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("should build quantized apertus model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_apertus_forward_shape() {
        let cfg = apertus_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig::default();
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedApertusForCausalLM::new(&cfg, vb, loader.as_ref()).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_cfg).unwrap();
        let mut bt = BlockTable::new(cache_cfg.block_size);
        let seq_len = 4;
        let input = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        kv_cache_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let logits = crate::engine::ModelForward::forward(
            &model,
            &input,
            0,
            &mut kv_cache_mgr,
            &bt,
            &slot_mapping,
        )
        .unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_apertus_untied_embeddings() {
        let mut cfg = apertus_config();
        cfg.tie_word_embeddings = false;
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig::default();
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedApertusForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("should build with untied embeddings");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_apertus_gptq_loader() {
        let cfg = apertus_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig {
            method: crate::quantization::QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            ..Default::default()
        };
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedApertusForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok());
    }
}
