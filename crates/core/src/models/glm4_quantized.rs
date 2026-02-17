//! Quantized GLM-4 model implementation.
//!
//! GLM-4-specific features preserved in the quantized path:
//! - 4 layer norms per decoder layer (input, post_self_attn, post_attention, post_mlp)
//! - Partial RoPE (factor 0.5) with split rotation style (not neox)
//! - Configurable attention bias on Q/K/V, no bias on o_proj
//! - RmsNorm, SwiGLU MLP, GQA
//! - Weight prefix: model.layers.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

const GLM4_PARTIAL_ROTARY_FACTOR: f64 = 0.5;
const GLM4_IS_NEOX_STYLE: bool = false;

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

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct QuantizedSwiGluMlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedSwiGluMlp {
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

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedGlm4Attention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedGlm4Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let use_bias = cfg.attention_bias.unwrap_or(false);

        let q_proj = loader.load_linear(
            &format!("{}.q_proj", vb.prefix()),
            cfg.hidden_size,
            num_heads * head_dim,
            use_bias,
        )?;
        let k_proj = loader.load_linear(
            &format!("{}.k_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            use_bias,
        )?;
        let v_proj = loader.load_linear(
            &format!("{}.v_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            use_bias,
        )?;
        // o_proj always has no bias in GLM4
        let o_proj = loader.load_linear(
            &format!("{}.o_proj", vb.prefix()),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            GLM4_PARTIAL_ROTARY_FACTOR,
            GLM4_IS_NEOX_STYLE,
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

// ─── Decoder Layer (4 layer norms) ──────────────────────────────────────────

struct QuantizedGlm4DecoderLayer {
    self_attn: QuantizedGlm4Attention,
    mlp: QuantizedSwiGluMlp,
    input_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
}

impl QuantizedGlm4DecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let self_attn = QuantizedGlm4Attention::new(cfg, vb.pp("self_attn"), loader)?;
        let mlp = QuantizedSwiGluMlp::new(cfg, vb.pp("mlp"), loader)?;

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_self_attn_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_self_attn_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let post_mlp_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_mlp_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_self_attn_layernorm,
            post_attention_layernorm,
            post_mlp_layernorm,
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
        let xs = self.post_self_attn_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;

        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_layernorm.forward(&xs)?;

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
        let xs = self.post_self_attn_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;

        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_layernorm.forward(&xs)?;

        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedGlm4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedGlm4DecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedGlm4ForCausalLM {
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
            layers.push(QuantizedGlm4DecoderLayer::new(cfg, vb_l.pp(i), loader)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

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
}

impl crate::engine::ModelForward for QuantizedGlm4ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedGlm4ForCausalLM::forward(
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
        ModelConfig {
            architectures: vec!["Glm4ForCausalLM".to_string()],
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
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
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
    fn test_quantized_glm4_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGlm4ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedGlm4ForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_glm4_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedGlm4ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build");

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

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_glm4_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGlm4ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Should construct with untied embeddings");
    }

    #[test]
    fn test_quantized_glm4_gptq_loader() {
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

        let model = QuantizedGlm4ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Should construct with GPTQ loader");
    }
}
