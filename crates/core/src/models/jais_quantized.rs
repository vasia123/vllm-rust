//! Quantized JAIS model implementation.
//!
//! JAIS-specific features preserved in the quantized path:
//! - ALiBi positional encoding (no RoPE)
//! - MuP (Maximal Update Parametrization) scaling:
//!   - embeddings_scale: scales embeddings after lookup
//!   - output_logits_scale: scales final logits
//!   - mup_scale_qk_dot_by_d: attention scale 1/d (MuP) vs 1/sqrt(d) (standard)
//! - Optional SwiGLU activation (vs GELU)
//! - GPT2-style architecture with LayerNorm
//! - Fused QKV projection (c_attn)
//! - Weight prefix: transformer.h.{i}.*

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

// ─── JAIS Config Extraction ─────────────────────────────────────────────────

struct JAISConfig {
    n_inner: usize,
    layer_norm_epsilon: f64,
    use_swiglu: bool,
    embeddings_scale: f64,
    output_logits_scale: f64,
    mup_scale_qk_dot_by_d: bool,
}

impl JAISConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let n_inner = cfg
            .extra
            .get("n_inner")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4 * cfg.hidden_size);

        let layer_norm_epsilon = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let act_str = cfg
            .extra
            .get("activation_function")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu_new");
        let use_swiglu = act_str == "swiglu";

        let embeddings_scale = cfg
            .extra
            .get("embeddings_scale")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("mup_embeddings_scale")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let output_logits_scale = cfg
            .extra
            .get("width_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                let mup_output_alpha = cfg
                    .extra
                    .get("mup_output_alpha")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                let mup_width_scale = cfg
                    .extra
                    .get("mup_width_scale")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                mup_output_alpha * mup_width_scale
            });

        let mup_scale_qk_dot_by_d = cfg
            .extra
            .get("scale_qk_dot_by_d")
            .and_then(|v| v.as_bool())
            .or_else(|| {
                cfg.extra
                    .get("mup_scale_qk_dot_by_d")
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false);

        Self {
            n_inner,
            layer_norm_epsilon,
            use_swiglu,
            embeddings_scale,
            output_logits_scale,
            mup_scale_qk_dot_by_d,
        }
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct QuantizedJAISMlp {
    c_fc: Box<dyn QuantizedLinear>,
    c_fc2: Option<Box<dyn QuantizedLinear>>,
    c_proj: Box<dyn QuantizedLinear>,
    use_swiglu: bool,
}

impl QuantizedJAISMlp {
    fn new(
        cfg: &ModelConfig,
        jais_cfg: &JAISConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let c_fc = loader.load_linear(
            &format!("{}.c_fc", vb.prefix()),
            cfg.hidden_size,
            jais_cfg.n_inner,
            true,
        )?;

        let c_fc2 = if jais_cfg.use_swiglu {
            Some(loader.load_linear(
                &format!("{}.c_fc2", vb.prefix()),
                cfg.hidden_size,
                jais_cfg.n_inner,
                true,
            )?)
        } else {
            None
        };

        let c_proj = loader.load_linear(
            &format!("{}.c_proj", vb.prefix()),
            jais_cfg.n_inner,
            cfg.hidden_size,
            true,
        )?;

        Ok(Self {
            c_fc,
            c_fc2,
            c_proj,
            use_swiglu: jais_cfg.use_swiglu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = if self.use_swiglu {
            let x1 = self.c_fc.forward(xs)?;
            let x2 = self
                .c_fc2
                .as_ref()
                .expect("c_fc2 must exist for SwiGLU")
                .forward(xs)?;
            (x1 * candle_nn::Activation::Silu.forward(&x2)?)?
        } else {
            let h = self.c_fc.forward(xs)?;
            candle_nn::Activation::NewGelu.forward(&h)?
        };
        self.c_proj.forward(&hidden)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedJAISAttention {
    c_attn: Box<dyn QuantizedLinear>,
    c_proj: Box<dyn QuantizedLinear>,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    head_dim: usize,
    attn_scale: f64,
}

impl QuantizedJAISAttention {
    fn new(
        cfg: &ModelConfig,
        jais_cfg: &JAISConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;

        let c_attn = loader.load_linear(
            &format!("{}.c_attn", vb.prefix()),
            cfg.hidden_size,
            cfg.hidden_size * 3,
            true,
        )?;

        let c_proj = loader.load_linear(
            &format!("{}.c_proj", vb.prefix()),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;

        let alibi = AlibiAttentionBias::new(num_heads, vb.dtype(), vb.device())?;

        let attn_scale_power = if jais_cfg.mup_scale_qk_dot_by_d {
            1.0
        } else {
            0.5
        };
        let attn_scale = (head_dim as f64).powf(-attn_scale_power);

        Ok(Self {
            c_attn,
            c_proj,
            alibi,
            num_heads,
            head_dim,
            attn_scale,
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

        let qkv = self.c_attn.forward(xs)?;

        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        let q = (q * self.attn_scale)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

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

        self.c_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.c_attn.forward(xs)?;

        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        let q = (q * self.attn_scale)?;

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

        Tensor::cat(&outputs, 0)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct QuantizedJAISBlock {
    ln_1: LayerNorm,
    attn: QuantizedJAISAttention,
    ln_2: LayerNorm,
    mlp: QuantizedJAISMlp,
}

impl QuantizedJAISBlock {
    fn new(
        cfg: &ModelConfig,
        jais_cfg: &JAISConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let ln_1 = layer_norm(cfg.hidden_size, jais_cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = QuantizedJAISAttention::new(cfg, jais_cfg, vb.pp("attn"), loader)?;
        let ln_2 = layer_norm(cfg.hidden_size, jais_cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = QuantizedJAISMlp::new(cfg, jais_cfg, vb.pp("mlp"), loader)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
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
        let xs = self.ln_1.forward(xs)?;
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
        let xs = self.ln_2.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
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
        let xs = self.ln_1.forward(xs)?;
        let attn_output =
            self.attn
                .forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedJAISLMHeadModel {
    wte: candle_nn::Embedding,
    h: Vec<QuantizedJAISBlock>,
    ln_f: LayerNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
    embeddings_scale: f64,
    output_logits_scale: f64,
}

impl QuantizedJAISLMHeadModel {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let jais_cfg = JAISConfig::from_model_config(cfg);
        let vb_t = vb.pp("transformer");

        let wte = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"))?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            h.push(QuantizedJAISBlock::new(cfg, &jais_cfg, vb_h.pp(i), loader)?);
        }

        let ln_f = layer_norm(
            cfg.hidden_size,
            jais_cfg.layer_norm_epsilon,
            vb_t.pp("ln_f"),
        )?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(wte.embeddings().clone()))
        } else {
            loader.load_linear(
                &format!("{}.lm_head", vb.prefix()),
                cfg.hidden_size,
                cfg.vocab_size,
                false,
            )?
        };

        Ok(Self {
            wte,
            h,
            ln_f,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            embeddings_scale: jais_cfg.embeddings_scale,
            output_logits_scale: jais_cfg.output_logits_scale,
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

        let mut xs = self.wte.forward(input_ids)?;
        xs = (xs * self.embeddings_scale)?;

        for (layer_idx, layer) in self.h.iter().enumerate() {
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

        let xs = self.ln_f.forward(&xs)?;

        let logits = self.lm_head.forward(&xs)?;
        logits * self.output_logits_scale
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedJAISLMHeadModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedJAISLMHeadModel::forward(
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
        let mut xs = self.wte.forward(input_ids)?;
        xs = (xs * self.embeddings_scale)?;

        for (layer_idx, layer) in self.h.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.ln_f.forward(&xs)?;

        let logits = self.lm_head.forward(&xs)?;
        logits * self.output_logits_scale
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
        extra.insert(
            "activation_function".to_string(),
            serde_json::Value::String("swiglu".to_string()),
        );
        extra.insert(
            "embeddings_scale".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(10.0).unwrap()),
        );
        extra.insert(
            "width_scale".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap()),
        );
        extra.insert(
            "mup_scale_qk_dot_by_d".to_string(),
            serde_json::Value::Bool(true),
        );

        ModelConfig {
            architectures: vec!["JAISLMHeadModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "swiglu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 0,
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
    fn test_quantized_jais_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedJAISLMHeadModel::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedJAISLMHeadModel should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_jais_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedJAISLMHeadModel::new(&cfg, vb, loader.as_ref()).expect("build");

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

    #[test]
    fn test_quantized_jais_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedJAISLMHeadModel::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with untied embeddings: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_jais_gptq_loader() {
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

        let model = QuantizedJAISLMHeadModel::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with GPTQ loader: {:?}",
            model.err()
        );
    }
}
