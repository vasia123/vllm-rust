//! Quantized GPTBigCode (StarCoder1) model implementation.
//!
//! GPTBigCode-specific features preserved in the quantized path:
//! - Multi-Query Attention (MQA) when multi_query=true
//! - Fused QKV (c_attn), split asymmetrically for MQA
//! - Absolute position embeddings (wpe), no RoPE
//! - LayerNorm (ln_1, ln_2) per layer, ln_f at end
//! - Configurable activation (NewGeLU default) MLP (c_fc → activation → c_proj)
//! - All linears have bias=true
//! - Weight prefix: transformer.h.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::paged_attention;
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

fn get_n_inner(cfg: &ModelConfig) -> usize {
    cfg.extra
        .get("n_inner")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(4 * cfg.hidden_size)
}

fn get_layer_norm_epsilon(cfg: &ModelConfig) -> f64 {
    cfg.extra
        .get("layer_norm_epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5)
}

fn get_activation(cfg: &ModelConfig) -> candle_nn::Activation {
    let act_str = cfg
        .extra
        .get("activation_function")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_new");
    match act_str {
        "gelu_new" => candle_nn::Activation::NewGelu,
        "gelu" => candle_nn::Activation::Gelu,
        "relu" => candle_nn::Activation::Relu,
        "silu" | "swish" => candle_nn::Activation::Silu,
        _ => candle_nn::Activation::NewGelu,
    }
}

fn get_multi_query(cfg: &ModelConfig) -> bool {
    cfg.extra
        .get("multi_query")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

// ─── Quantized Attention ─────────────────────────────────────────────────────

struct QuantizedGPTBigCodeAttention {
    c_attn: Box<dyn QuantizedLinear>,
    c_proj: Box<dyn QuantizedLinear>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl QuantizedGPTBigCodeAttention {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let multi_query = get_multi_query(cfg);

        let (num_kv_heads, total_num_kv_heads) = if multi_query {
            (1, 1)
        } else {
            (num_heads, num_heads)
        };

        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let total_qkv_size = num_heads * head_dim + 2 * total_num_kv_heads * head_dim;

        let c_attn = loader.load_linear(
            &format!("{prefix}.c_attn"),
            cfg.hidden_size,
            total_qkv_size,
            true,
        )?;
        let c_proj = loader.load_linear(
            &format!("{prefix}.c_proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;

        Ok(Self {
            c_attn,
            c_proj,
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

        let qkv = self.c_attn.forward(xs)?;

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
        self.c_proj.forward(&attn_output)
    }
}

// ─── Quantized MLP ───────────────────────────────────────────────────────────

struct QuantizedGPTBigCodeMLP {
    c_fc: Box<dyn QuantizedLinear>,
    c_proj: Box<dyn QuantizedLinear>,
    activation: candle_nn::Activation,
}

impl QuantizedGPTBigCodeMLP {
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
        activation: candle_nn::Activation,
    ) -> Result<Self> {
        let n_inner = get_n_inner(cfg);
        let c_fc = loader.load_linear(&format!("{prefix}.c_fc"), cfg.hidden_size, n_inner, true)?;
        let c_proj =
            loader.load_linear(&format!("{prefix}.c_proj"), n_inner, cfg.hidden_size, true)?;
        Ok(Self {
            c_fc,
            c_proj,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.c_fc.forward(xs)?;
        let hidden = self.activation.forward(&hidden)?;
        self.c_proj.forward(&hidden)
    }
}

// ─── Quantized Block ─────────────────────────────────────────────────────────

struct QuantizedGPTBigCodeBlock {
    ln_1: LayerNorm,
    attn: QuantizedGPTBigCodeAttention,
    ln_2: LayerNorm,
    mlp: QuantizedGPTBigCodeMLP,
}

impl QuantizedGPTBigCodeBlock {
    fn new(
        cfg: &ModelConfig,
        vb: &VarBuilder,
        loader: &dyn QuantizedWeightLoader,
        layer_idx: usize,
        activation: candle_nn::Activation,
    ) -> Result<Self> {
        let prefix = format!("transformer.h.{layer_idx}");
        let ln_eps = get_layer_norm_epsilon(cfg);

        let ln_1 = layer_norm(cfg.hidden_size, ln_eps, vb.pp(format!("{prefix}.ln_1")))?;
        let attn = QuantizedGPTBigCodeAttention::new(cfg, loader, &format!("{prefix}.attn"))?;
        let ln_2 = layer_norm(cfg.hidden_size, ln_eps, vb.pp(format!("{prefix}.ln_2")))?;
        let mlp = QuantizedGPTBigCodeMLP::new(cfg, loader, &format!("{prefix}.mlp"), activation)?;

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
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln_1.forward(xs)?;
        let attn_output = self.attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
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
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln_1.forward(xs)?;
        let attn_output = self
            .attn
            .forward_decode_batch(&xs, sequences, cache_engine)?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

// ─── Quantized Model ─────────────────────────────────────────────────────────

pub struct QuantizedGPTBigCodeForCausalLM {
    wte: Embedding,
    wpe: Embedding,
    h: Vec<QuantizedGPTBigCodeBlock>,
    ln_f: LayerNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedGPTBigCodeForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let ln_eps = get_layer_norm_epsilon(cfg);
        let activation = get_activation(cfg);

        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("transformer.wte"))?;
        let wpe = embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("transformer.wpe"),
        )?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            h.push(QuantizedGPTBigCodeBlock::new(
                cfg, &vb, loader, i, activation,
            )?);
        }

        let ln_f = layer_norm(cfg.hidden_size, ln_eps, vb.pp("transformer.ln_f"))?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(wte.embeddings().clone()))
        } else {
            loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn position_ids(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let positions: Vec<u32> = (0..seq_len).map(|i| (seqlen_offset + i) as u32).collect();
        Tensor::from_vec(positions, (1, seq_len), &self.device)
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

        let token_embeds = self.wte.forward(input_ids)?;
        let pos_ids = self.position_ids(seq_len, seqlen_offset)?;
        let pos_embeds = self.wpe.forward(&pos_ids)?;
        let mut xs = token_embeds.broadcast_add(&pos_embeds)?;

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

impl crate::engine::ModelForward for QuantizedGPTBigCodeForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedGPTBigCodeForCausalLM::forward(
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
        let batch_size = sequences.len();
        let pos_ids_data: Vec<u32> = sequences.iter().map(|s| s.seqlen_offset as u32).collect();
        let pos_ids = Tensor::from_vec(pos_ids_data, (batch_size, 1), &self.device)?;

        let token_embeds = self.wte.forward(input_ids)?;
        let pos_embeds = self.wpe.forward(&pos_ids)?;
        let mut xs = (token_embeds + pos_embeds)?;

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
        extra.insert("multi_query".to_string(), serde_json::Value::Bool(true));

        ModelConfig {
            architectures: vec!["GPTBigCodeForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_new".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 0,
            eos_token_id: 0,
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
            num_kv_heads: 1, // MQA
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_quantized_gpt_bigcode_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGPTBigCodeForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Failed: {:?}", model.err());
        assert_eq!(model.unwrap().h.len(), 2);
    }

    #[test]
    fn test_quantized_gpt_bigcode_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model =
            QuantizedGPTBigCodeForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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
    fn test_quantized_gpt_bigcode_gptq_loader() {
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
        let model = QuantizedGPTBigCodeForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "GPTQ loader failed: {:?}", model.err());
    }
}
