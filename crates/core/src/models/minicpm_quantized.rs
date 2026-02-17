//! Quantized MiniCPM model implementation.
//!
//! MiniCPM is a Llama-variant with three unique scaling factors:
//! - `scale_emb`: multiplier applied to token embeddings
//! - `scale_depth / sqrt(num_layers)`: multiplier applied to each layer's residual
//! - `scale_width = hidden_size / dim_model_base`: divisor applied to final hidden states
//!
//! MLP uses SwiGLU (gate_proj + up_proj + down_proj) with silu or FatReLU activation.
//! Weight prefix: model.layers.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
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

struct MiniCPMScaling {
    scale_emb: f64,
    residual_scale: f64,
    output_scale: f64,
}

impl MiniCPMScaling {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let scale_emb = cfg
            .extra
            .get("scale_emb")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let scale_depth = cfg
            .extra
            .get("scale_depth")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let dim_model_base = cfg
            .extra
            .get("dim_model_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.hidden_size as f64);

        let residual_scale = scale_depth / (cfg.num_hidden_layers as f64).sqrt();
        let output_scale = cfg.hidden_size as f64 / dim_model_base;

        Self {
            scale_emb,
            residual_scale,
            output_scale,
        }
    }
}

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct QuantizedMiniCPMMlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedMiniCPMMlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let gate_proj = loader.load_linear(
            &format!("{}.gate_proj", vb.prefix()),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
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

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = (gate.silu()? * up)?;
        self.down_proj.forward(&hidden)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct QuantizedMiniCPMAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedMiniCPMAttention {
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

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct QuantizedMiniCPMDecoderLayer {
    self_attn: QuantizedMiniCPMAttention,
    mlp: QuantizedMiniCPMMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_scale: f64,
}

impl QuantizedMiniCPMDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        residual_scale: f64,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let self_attn = QuantizedMiniCPMAttention::new(cfg, vb.pp("self_attn"), loader)?;
        let mlp = QuantizedMiniCPMMlp::new(cfg, vb.pp("mlp"), loader)?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_scale,
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
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward(
            &hidden,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (residual + (hidden * self.residual_scale)?)?;

        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual + (hidden * self.residual_scale)?
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward_decode_batch(
            &hidden,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (residual + (hidden * self.residual_scale)?)?;

        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual + (hidden * self.residual_scale)?
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct QuantizedMiniCPMForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedMiniCPMDecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    scale_emb: f64,
    output_scale: f64,
    device: Device,
    dtype: DType,
}

impl QuantizedMiniCPMForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let scaling = MiniCPMScaling::from_model_config(cfg);

        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedMiniCPMDecoderLayer::new(
                cfg,
                scaling.residual_scale,
                vb_l.pp(i),
                loader,
            )?);
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
            scale_emb: scaling.scale_emb,
            output_scale: scaling.output_scale,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedMiniCPMForCausalLM {
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
        xs = (xs * self.scale_emb)?;

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
        let xs = (xs / self.output_scale)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = (xs * self.scale_emb)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        let xs = self.norm.forward(&xs)?;
        let xs = (xs / self.output_scale)?;
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

    fn minicpm_config() -> ModelConfig {
        let mut cfg = ModelConfig {
            architectures: vec!["MiniCPMForCausalLM".to_string()],
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
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            ..Default::default()
        };
        cfg.extra
            .insert("scale_emb".to_string(), serde_json::Value::from(12.0));
        cfg.extra
            .insert("scale_depth".to_string(), serde_json::Value::from(1.4));
        cfg.extra
            .insert("dim_model_base".to_string(), serde_json::Value::from(256.0));
        cfg
    }

    #[test]
    fn test_quantized_minicpm_construction() {
        let cfg = minicpm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig::default();
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedMiniCPMForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("should build quantized minicpm model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_minicpm_forward_shape() {
        let cfg = minicpm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig::default();
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedMiniCPMForCausalLM::new(&cfg, vb, loader.as_ref()).unwrap();

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
    fn test_quantized_minicpm_untied_embeddings() {
        let mut cfg = minicpm_config();
        cfg.tie_word_embeddings = false;
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = crate::quantization::DetectedQuantConfig::default();
        let loader = crate::quantization::create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedMiniCPMForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("should build with untied embeddings");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_minicpm_gptq_loader() {
        let cfg = minicpm_config();
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
        let model = QuantizedMiniCPMForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok());
    }
}
