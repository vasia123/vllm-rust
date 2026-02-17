//! Quantized Nemotron model implementation (NVIDIA).
//!
//! Nemotron-specific features preserved in the quantized path:
//! - LayerNorm1P (weight + 1 trick) instead of standard LayerNorm
//! - SqReLU activation (relu(x)^2) instead of SiLU/GELU
//! - No gate projection (just up_proj → sq_relu → down_proj)
//! - Partial RoPE (configurable factor, default 0.5)
//! - Configurable bias on attention and MLP projections
//! - Weight prefix: model.layers.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
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

// ─── LayerNorm1P ─────────────────────────────────────────────────────────────

struct LayerNorm1P {
    inner: LayerNorm,
    eps: f64,
}

impl LayerNorm1P {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = layer_norm(size, eps, vb)?;
        Ok(Self { inner, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight_plus_one = (self.inner.weight() + 1.0)?;
        let mean = xs.mean_keepdim(candle_core::D::Minus1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps = Tensor::new(&[self.eps as f32], xs.device())?
            .to_dtype(xs.dtype())?
            .broadcast_as(variance.shape())?;
        let inv_std = (variance + eps)?.sqrt()?.recip()?;
        let normalized = centered.broadcast_mul(&inv_std)?;
        let result = normalized.broadcast_mul(&weight_plus_one)?;
        if let Some(bias) = self.inner.bias() {
            result.broadcast_add(bias)
        } else {
            Ok(result)
        }
    }
}

// ─── SqReLU Activation ──────────────────────────────────────────────────────

fn sq_relu(xs: &Tensor) -> Result<Tensor> {
    xs.relu()?.sqr()
}

// ─── Config Helpers ──────────────────────────────────────────────────────────

fn get_norm_eps(cfg: &ModelConfig) -> f64 {
    cfg.extra
        .get("norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(cfg.rms_norm_eps)
}

fn get_partial_rotary_factor(cfg: &ModelConfig) -> f64 {
    cfg.extra
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            cfg.extra
                .get("rope_parameters")
                .and_then(|v| v.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(0.5)
}

fn get_mlp_bias(cfg: &ModelConfig) -> bool {
    cfg.extra
        .get("mlp_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

// ─── MLP (SqReLU, no gate) ─────────────────────────────────────────────────

struct QuantizedNemotronMlp {
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedNemotronMlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let bias = get_mlp_bias(cfg);

        let up_proj = loader.load_linear(
            &format!("{}.up_proj", vb.prefix()),
            cfg.hidden_size,
            cfg.intermediate_size,
            bias,
        )?;
        let down_proj = loader.load_linear(
            &format!("{}.down_proj", vb.prefix()),
            cfg.intermediate_size,
            cfg.hidden_size,
            bias,
        )?;

        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.up_proj.forward(xs)?;
        let xs = sq_relu(&xs)?;
        self.down_proj.forward(&xs)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedNemotronAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedNemotronAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias.unwrap_or(false);

        let q_proj = loader.load_linear(
            &format!("{}.q_proj", vb.prefix()),
            cfg.hidden_size,
            num_heads * head_dim,
            bias,
        )?;
        let k_proj = loader.load_linear(
            &format!("{}.k_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
        )?;
        let v_proj = loader.load_linear(
            &format!("{}.v_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
        )?;
        let o_proj = loader.load_linear(
            &format!("{}.o_proj", vb.prefix()),
            num_heads * head_dim,
            cfg.hidden_size,
            bias,
        )?;

        let partial_rotary_factor = get_partial_rotary_factor(cfg);

        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            partial_rotary_factor,
            true, // neox style
            vb.dtype(),
            vb.device(),
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

        Tensor::cat(&outputs, 0)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct QuantizedNemotronDecoderLayer {
    self_attn: QuantizedNemotronAttention,
    mlp: QuantizedNemotronMlp,
    input_layernorm: LayerNorm1P,
    post_attention_layernorm: LayerNorm1P,
}

impl QuantizedNemotronDecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let norm_eps = get_norm_eps(cfg);

        let self_attn = QuantizedNemotronAttention::new(cfg, vb.pp("self_attn"), loader)?;
        let mlp = QuantizedNemotronMlp::new(cfg, vb.pp("mlp"), loader)?;
        let input_layernorm =
            LayerNorm1P::new(cfg.hidden_size, norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            LayerNorm1P::new(cfg.hidden_size, norm_eps, vb.pp("post_attention_layernorm"))?;

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
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
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
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedNemotronForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedNemotronDecoderLayer>,
    norm: LayerNorm1P,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedNemotronForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let norm_eps = get_norm_eps(cfg);

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedNemotronDecoderLayer::new(cfg, vb_l.pp(i), loader)?);
        }

        let norm = LayerNorm1P::new(cfg.hidden_size, norm_eps, vb_m.pp("norm"))?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(embed_tokens.embeddings().clone()))
        } else {
            loader.load_linear(
                &format!("{}.lm_head", vb.prefix()),
                cfg.hidden_size,
                cfg.vocab_size,
                false,
            )?
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

impl crate::engine::ModelForward for QuantizedNemotronForCausalLM {
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
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("partial_rotary_factor".into(), serde_json::Value::from(0.5));
        extra.insert("norm_eps".into(), serde_json::Value::from(1e-5));

        ModelConfig {
            architectures: vec!["NemotronForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "relu2".to_string(),
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
    fn test_quantized_nemotron_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedNemotronForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedNemotronForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_nemotron_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedNemotronForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build");

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
    fn test_quantized_nemotron_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedNemotronForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Should construct with untied embeddings");
    }

    #[test]
    fn test_quantized_nemotron_gptq_loader() {
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

        let model = QuantizedNemotronForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Should construct with GPTQ loader");
    }
}
