//! Quantized Step1 model implementation.
//!
//! Step1-specific features preserved in the quantized path:
//! - **Sqrt ALiBi** attention bias (no RoPE): bias = slope * (-sqrt(distance))
//! - **Merged QKV** projection: qkv_proj → narrow split to Q, K, V
//! - **Merged gate_up** MLP: gate_up_proj → chunk → silu(gate) * up → down_proj
//! - **Configurable bias** on attention and MLP projections
//! - **GQA** via `num_attention_groups` config key
//! - Weight prefix: model.layers.X

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, AlibiAttentionBias, RmsNorm};
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

// ─── Config ──────────────────────────────────────────────────────────────────

struct Step1Config {
    attention_bias: bool,
    mlp_bias: bool,
    num_kv_heads: usize,
}

impl Step1Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let attention_bias = cfg.attention_bias.unwrap_or(false);

        let mlp_bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let num_kv_heads = cfg
            .extra
            .get("num_attention_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_key_value_heads)
            .max(1);

        Self {
            attention_bias,
            mlp_bias,
            num_kv_heads,
        }
    }
}

// ─── Sqrt ALiBi Bias ─────────────────────────────────────────────────────────

fn build_sqrt_alibi_bias(
    alibi: &AlibiAttentionBias,
    seq_len: usize,
    kv_len: usize,
) -> Result<Tensor> {
    let device = alibi.slopes().device();
    let dtype = alibi.slopes().dtype();
    let num_heads = alibi.num_heads();

    let kv_offset = kv_len - seq_len;

    let distances: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                let query_pos = kv_offset + i;
                let relative_pos = j as i64 - query_pos as i64;
                if relative_pos <= 0 {
                    -((-relative_pos) as f32).sqrt()
                } else {
                    0.0
                }
            })
        })
        .collect();

    let distance_matrix =
        Tensor::from_vec(distances, (1, 1, seq_len, kv_len), device)?.to_dtype(dtype)?;
    let slopes_expanded = alibi.slopes().reshape((1, num_heads, 1, 1))?;
    distance_matrix.broadcast_mul(&slopes_expanded)
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedStep1Attention {
    qkv_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl QuantizedStep1Attention {
    fn new(
        cfg: &ModelConfig,
        step_cfg: &Step1Config,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = step_cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let total_qkv = q_size + 2 * kv_size;
        let qkv_proj = loader.load_linear(
            &format!("{}.qkv_proj", vb.prefix()),
            cfg.hidden_size,
            total_qkv,
            step_cfg.attention_bias,
        )?;

        let o_proj = loader.load_linear(
            &format!("{}.o_proj", vb.prefix()),
            num_heads * head_dim,
            cfg.hidden_size,
            step_cfg.attention_bias,
        )?;

        let alibi = AlibiAttentionBias::new(num_heads, vb.dtype(), vb.device())?;

        Ok(Self {
            qkv_proj,
            o_proj,
            alibi,
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

        let qkv = self.qkv_proj.forward(xs)?;
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

        // Sqrt ALiBi bias (no RoPE)
        let kv_len = q_len + seqlen_offset;
        let alibi_bias = build_sqrt_alibi_bias(&self.alibi, q_len, kv_len)?;

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

        let qkv = self.qkv_proj.forward(xs)?;
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

            let kv_len = seq.seqlen_offset + 1;
            let alibi_bias = build_sqrt_alibi_bias(&self.alibi, 1, kv_len)?;

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
                self.num_kv_heads,
                self.head_dim,
            )?;
            outputs.push(attn_out);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output)
    }
}

// ─── MLP (merged gate_up) ───────────────────────────────────────────────────

struct QuantizedStep1Mlp {
    gate_up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedStep1Mlp {
    fn new(
        cfg: &ModelConfig,
        step_cfg: &Step1Config,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let gate_up_proj = loader.load_linear(
            &format!("{}.gate_up_proj", vb.prefix()),
            cfg.hidden_size,
            2 * cfg.intermediate_size,
            step_cfg.mlp_bias,
        )?;

        let down_proj = loader.load_linear(
            &format!("{}.down_proj", vb.prefix()),
            cfg.intermediate_size,
            cfg.hidden_size,
            step_cfg.mlp_bias,
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = chunks[0].silu()?;
        let intermediate = gate.mul(&chunks[1])?;
        self.down_proj.forward(&intermediate)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct QuantizedStep1DecoderLayer {
    self_attn: QuantizedStep1Attention,
    mlp: QuantizedStep1Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedStep1DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        step_cfg: &Step1Config,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let self_attn = QuantizedStep1Attention::new(cfg, step_cfg, vb.pp("self_attn"), loader)?;
        let mlp = QuantizedStep1Mlp::new(cfg, step_cfg, vb.pp("mlp"), loader)?;
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedStep1ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedStep1DecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedStep1ForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let step_cfg = Step1Config::from_model_config(cfg);
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedStep1DecoderLayer::new(
                cfg,
                &step_cfg,
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
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }
}

impl crate::engine::ModelForward for QuantizedStep1ForCausalLM {
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
        xs = self.norm.forward(&xs)?;
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
        xs = self.norm.forward(&xs)?;
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
        extra.insert("num_attention_groups".to_string(), serde_json::json!(2));

        ModelConfig {
            architectures: vec!["Step1ForCausalLM".to_string()],
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
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        let num_kv_heads = cfg
            .extra
            .get("num_attention_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_key_value_heads);

        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_quantized_step1_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedStep1ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedStep1ForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), 2);
    }

    #[test]
    fn test_quantized_step1_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedStep1ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 5;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");

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

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_step1_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedStep1ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Untied should construct: {:?}", model.err());
    }

    #[test]
    fn test_quantized_step1_gptq_loader() {
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

        let model = QuantizedStep1ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "GPTQ should construct: {:?}", model.err());
    }
}
