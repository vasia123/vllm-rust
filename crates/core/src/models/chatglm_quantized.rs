//! Quantized ChatGLM model implementation (THUDM ChatGLM2/3/4 series).
//!
//! ChatGLM-specific features preserved in the quantized path:
//! - Packed QKV projection (single query_key_value linear)
//! - Packed gate+up projection (single dense_h_to_4h linear for SwiGLU)
//! - Partial RoPE (factor 0.5) with configurable neox style
//! - Configurable MQA/GQA via `multi_query_attention` config flag
//! - Configurable norm type (RMSNorm or LayerNorm)
//! - Optional `apply_residual_connection_post_layernorm`
//! - Optional `post_layer_norm` (final layer norm before output)
//! - Weight prefix: `transformer.encoder.layers.{i}.*`

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Constants ───────────────────────────────────────────────────────────────

const CHATGLM_PARTIAL_ROTARY_FACTOR: f64 = 0.5;

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

// ─── Normalization Abstraction ──────────────────────────────────────────────

enum Norm {
    Rms(RmsNorm),
    Layer(LayerNorm),
}

impl Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Norm::Rms(n) => n.forward(xs),
            Norm::Layer(n) => n.forward(xs),
        }
    }
}

fn create_norm(hidden_size: usize, eps: f64, use_rmsnorm: bool, vb: VarBuilder) -> Result<Norm> {
    if use_rmsnorm {
        Ok(Norm::Rms(rms_norm(hidden_size, eps, vb)?))
    } else {
        Ok(Norm::Layer(layer_norm(hidden_size, eps, vb)?))
    }
}

// ─── Config Helpers ──────────────────────────────────────────────────────────

fn get_use_rmsnorm(cfg: &ModelConfig) -> bool {
    cfg.extra
        .get("rmsnorm")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

fn get_layernorm_epsilon(cfg: &ModelConfig) -> f64 {
    cfg.extra
        .get("layernorm_epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(cfg.rms_norm_eps)
}

fn get_ffn_hidden_size(cfg: &ModelConfig) -> usize {
    cfg.extra
        .get("ffn_hidden_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.intermediate_size)
}

fn get_add_bias_linear(cfg: &ModelConfig) -> bool {
    cfg.extra
        .get("add_bias_linear")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn get_has_qkv_bias(cfg: &ModelConfig) -> bool {
    get_add_bias_linear(cfg)
        || cfg
            .extra
            .get("add_qkv_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
}

fn get_apply_residual_connection_post_layernorm(cfg: &ModelConfig) -> bool {
    cfg.extra
        .get("apply_residual_connection_post_layernorm")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn get_padded_vocab_size(cfg: &ModelConfig) -> usize {
    cfg.extra
        .get("padded_vocab_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.vocab_size)
}

fn get_kv_channels(cfg: &ModelConfig) -> usize {
    cfg.extra
        .get("kv_channels")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.head_dim)
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedChatGLMAttention {
    query_key_value: Box<dyn QuantizedLinear>,
    dense: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl QuantizedChatGLMAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = get_kv_channels(cfg);

        let multi_query_attention = cfg
            .extra
            .get("multi_query_attention")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let num_kv_heads = if multi_query_attention {
            cfg.num_key_value_heads
        } else {
            num_heads
        };

        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let has_bias = get_has_qkv_bias(cfg);

        let query_key_value = loader.load_linear(
            &format!("{}.query_key_value", vb.prefix()),
            cfg.hidden_size,
            q_size + 2 * kv_size,
            has_bias,
        )?;

        let dense = loader.load_linear(
            &format!("{}.dense", vb.prefix()),
            q_size,
            cfg.hidden_size,
            get_add_bias_linear(cfg),
        )?;

        let rope_ratio = cfg
            .extra
            .get("rope_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_theta = 10000.0 * rope_ratio;

        let max_positions = cfg
            .extra
            .get("seq_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.max_position_embeddings);

        let original_rope = cfg
            .extra
            .get("original_rope")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let is_neox_style = !original_rope;

        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            max_positions,
            rope_theta,
            CHATGLM_PARTIAL_ROTARY_FACTOR,
            is_neox_style,
            vb.dtype(),
            vb.device(),
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

        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

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
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

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

// ─── MLP (SwiGLU with packed gate+up) ───────────────────────────────────────

struct QuantizedChatGLMMlp {
    dense_h_to_4h: Box<dyn QuantizedLinear>,
    dense_4h_to_h: Box<dyn QuantizedLinear>,
    ffn_hidden_size: usize,
}

impl QuantizedChatGLMMlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let ffn_hidden_size = get_ffn_hidden_size(cfg);
        let add_bias_linear = get_add_bias_linear(cfg);

        let dense_h_to_4h = loader.load_linear(
            &format!("{}.dense_h_to_4h", vb.prefix()),
            cfg.hidden_size,
            2 * ffn_hidden_size,
            add_bias_linear,
        )?;

        let dense_4h_to_h = loader.load_linear(
            &format!("{}.dense_4h_to_h", vb.prefix()),
            ffn_hidden_size,
            cfg.hidden_size,
            add_bias_linear,
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            ffn_hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let intermediate = self.dense_h_to_4h.forward(xs)?;

        let gate = intermediate.narrow(candle_core::D::Minus1, 0, self.ffn_hidden_size)?;
        let up = intermediate.narrow(
            candle_core::D::Minus1,
            self.ffn_hidden_size,
            self.ffn_hidden_size,
        )?;
        let hidden = (candle_nn::ops::silu(&gate)? * up)?;

        self.dense_4h_to_h.forward(&hidden)
    }
}

// ─── Decoder Layer (GLMBlock) ───────────────────────────────────────────────

struct QuantizedChatGLMBlock {
    self_attention: QuantizedChatGLMAttention,
    mlp: QuantizedChatGLMMlp,
    input_layernorm: Norm,
    post_attention_layernorm: Norm,
    apply_residual_connection_post_layernorm: bool,
}

impl QuantizedChatGLMBlock {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let use_rmsnorm = get_use_rmsnorm(cfg);
        let eps = get_layernorm_epsilon(cfg);

        let self_attention = QuantizedChatGLMAttention::new(cfg, vb.pp("self_attention"), loader)?;
        let mlp = QuantizedChatGLMMlp::new(cfg, vb.pp("mlp"), loader)?;

        let input_layernorm =
            create_norm(cfg.hidden_size, eps, use_rmsnorm, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = create_norm(
            cfg.hidden_size,
            eps,
            use_rmsnorm,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            apply_residual_connection_post_layernorm: get_apply_residual_connection_post_layernorm(
                cfg,
            ),
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
        let layernorm_output = self.input_layernorm.forward(xs)?;

        let attention_output = self.self_attention.forward(
            &layernorm_output,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            xs
        };
        let layernorm_input = (residual + attention_output)?;

        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &layernorm_input
        };

        let mlp_output = self.mlp.forward(&layernorm_output)?;
        mlp_output + residual
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(xs)?;

        let attention_output = self.self_attention.forward_decode_batch(
            &layernorm_output,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            xs
        };
        let layernorm_input = (residual + attention_output)?;

        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &layernorm_input
        };

        let mlp_output = self.mlp.forward(&layernorm_output)?;
        mlp_output + residual
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedChatGLMForCausalLM {
    embedding: candle_nn::Embedding,
    layers: Vec<QuantizedChatGLMBlock>,
    final_layernorm: Option<Norm>,
    output_layer: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedChatGLMForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_t = vb.pp("transformer");
        let vocab_size = get_padded_vocab_size(cfg);

        let embedding = candle_nn::embedding(
            vocab_size,
            cfg.hidden_size,
            vb_t.pp("embedding").pp("word_embeddings"),
        )?;

        let post_layer_norm = cfg
            .extra
            .get("post_layer_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let use_rmsnorm = get_use_rmsnorm(cfg);
        let eps = get_layernorm_epsilon(cfg);

        let vb_enc = vb_t.pp("encoder");

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_enc.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedChatGLMBlock::new(cfg, vb_l.pp(i), loader)?);
        }

        let final_layernorm = if post_layer_norm {
            Some(create_norm(
                cfg.hidden_size,
                eps,
                use_rmsnorm,
                vb_enc.pp("final_layernorm"),
            )?)
        } else {
            None
        };

        let output_layer: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(embedding.embeddings().clone()))
        } else {
            loader.load_linear(
                &format!("{}.output_layer", vb_t.prefix()),
                cfg.hidden_size,
                vocab_size,
                false,
            )?
        };

        Ok(Self {
            embedding,
            layers,
            final_layernorm,
            output_layer,
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

        let mut xs = self.embedding.forward(input_ids)?;
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

        if let Some(ref norm) = self.final_layernorm {
            xs = norm.forward(&xs)?;
        }

        self.output_layer.forward(&xs)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedChatGLMForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedChatGLMForCausalLM::forward(
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
        let mut xs = self.embedding.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        if let Some(ref norm) = self.final_layernorm {
            xs = norm.forward(&xs)?;
        }

        self.output_layer.forward(&xs)
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
            "multi_query_attention".into(),
            serde_json::Value::Bool(true),
        );
        extra.insert("padded_vocab_size".into(), serde_json::Value::from(256u64));
        extra.insert(
            "apply_residual_connection_post_layernorm".into(),
            serde_json::Value::Bool(false),
        );
        extra.insert("post_layer_norm".into(), serde_json::Value::Bool(true));
        extra.insert("rmsnorm".into(), serde_json::Value::Bool(true));
        extra.insert("layernorm_epsilon".into(), serde_json::Value::from(1e-5));
        extra.insert("ffn_hidden_size".into(), serde_json::Value::from(128u64));

        ModelConfig {
            architectures: vec!["ChatGLMForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
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
    fn test_quantized_chatglm_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedChatGLMForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedChatGLMForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_layernorm.is_some());
    }

    #[test]
    fn test_quantized_chatglm_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedChatGLMForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build");

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

        let vocab_size = get_padded_vocab_size(&cfg);

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
            &[batch_size, seq_len, vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_quantized_chatglm_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedChatGLMForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with untied embeddings: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_chatglm_gptq_loader() {
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

        let model = QuantizedChatGLMForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Should construct with GPTQ loader: {:?}",
            model.err()
        );
    }
}
