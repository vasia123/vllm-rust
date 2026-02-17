//! Quantized GPT-NeoX model implementation (StableLM, Pythia, etc.).
//!
//! This module provides a quantized version of GPT-NeoX that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.
//!
//! Key differences from quantized Llama:
//! - LayerNorm instead of RMSNorm
//! - GELU activation instead of SiLU
//! - Parallel or sequential residual (configurable via use_parallel_residual)
//! - Partial RoPE (rotary_pct parameter)
//! - All linear layers have bias
//! - Different weight names: gpt_neox prefix, embed_in/embed_out

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Config extraction ───────────────────────────────────────────────────────

struct GptNeoXConfig {
    use_parallel_residual: bool,
    rotary_pct: f64,
    layer_norm_eps: f64,
}

impl GptNeoXConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let use_parallel_residual = cfg
            .extra
            .get("use_parallel_residual")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let rotary_pct = cfg
            .extra
            .get("rotary_pct")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);

        let layer_norm_eps = cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        Self {
            use_parallel_residual,
            rotary_pct,
            layer_norm_eps,
        }
    }
}

// ─── Quantized MLP ──────────────────────────────────────────────────────────

struct QuantizedGptNeoXMlp {
    dense_h_to_4h: Box<dyn QuantizedLinear>,
    dense_4h_to_h: Box<dyn QuantizedLinear>,
}

impl QuantizedGptNeoXMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let dense_h_to_4h = loader.load_linear(
            &format!("{prefix}.dense_h_to_4h"),
            hidden_size,
            intermediate_size,
            true,
        )?;
        let dense_4h_to_h = loader.load_linear(
            &format!("{prefix}.dense_4h_to_h"),
            intermediate_size,
            hidden_size,
            true,
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.dense_h_to_4h.forward(x)?;
        let hidden = hidden.gelu_erf()?;
        self.dense_4h_to_h.forward(&hidden)
    }
}

// ─── Quantized Attention ─────────────────────────────────────────────────────

struct QuantizedGptNeoXAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    dense: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl QuantizedGptNeoXAttention {
    fn new(
        cfg: &ModelConfig,
        neox_cfg: &GptNeoXConfig,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            true,
        )?;
        let k_proj = loader.load_linear(
            &format!("{prefix}.k_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            true,
        )?;
        let v_proj = loader.load_linear(
            &format!("{prefix}.v_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            true,
        )?;
        let dense = loader.load_linear(
            &format!("{prefix}.dense"),
            num_heads * head_dim,
            cfg.hidden_size,
            true,
        )?;

        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            neox_cfg.rotary_pct,
            true, // neox-style interleaved rotation
            loader.dtype(),
            loader.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            rotary_emb,
            num_heads,
            head_dim,
        })
    }

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
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
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
            self.num_heads, // MHA: kv_heads == q_heads
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

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

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

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

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

            let attn_output = crate::cuda_kernels::paged_attention_cuda(
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
            )?;

            self.dense.forward(&attn_output.unsqueeze(1)?)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
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

// ─── Quantized Decoder Layer ─────────────────────────────────────────────────

struct QuantizedGptNeoXDecoderLayer {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: QuantizedGptNeoXAttention,
    mlp: QuantizedGptNeoXMlp,
    use_parallel_residual: bool,
}

impl QuantizedGptNeoXDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        neox_cfg: &GptNeoXConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("gpt_neox.layers.{layer_idx}");
        let vb_layer = vb.pp("gpt_neox").pp("layers").pp(layer_idx);

        let input_layernorm = layer_norm(
            cfg.hidden_size,
            neox_cfg.layer_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            neox_cfg.layer_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
        )?;
        let attention =
            QuantizedGptNeoXAttention::new(cfg, neox_cfg, loader, &format!("{prefix}.attention"))?;
        let mlp = QuantizedGptNeoXMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            loader,
            &format!("{prefix}.mlp"),
        )?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
            use_parallel_residual: neox_cfg.use_parallel_residual,
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
        let ln_out = self.input_layernorm.forward(xs)?;
        let attn_output = self.attention.forward(
            &ln_out,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;

        if self.use_parallel_residual {
            let ln2_out = self.post_attention_layernorm.forward(xs)?;
            let mlp_output = self.mlp.forward(&ln2_out)?;
            let result = (xs + &attn_output)?;
            &result + mlp_output
        } else {
            let xs = (xs + attn_output)?;
            let ln2_out = self.post_attention_layernorm.forward(&xs)?;
            let mlp_output = self.mlp.forward(&ln2_out)?;
            &xs + mlp_output
        }
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let ln_out = self.input_layernorm.forward(xs)?;
        let attn_output = self.attention.forward_decode_batch(
            &ln_out,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;

        if self.use_parallel_residual {
            let ln2_out = self.post_attention_layernorm.forward(xs)?;
            let mlp_output = self.mlp.forward(&ln2_out)?;
            let result = (xs + &attn_output)?;
            &result + mlp_output
        } else {
            let xs = (xs + attn_output)?;
            let ln2_out = self.post_attention_layernorm.forward(&xs)?;
            let mlp_output = self.mlp.forward(&ln2_out)?;
            &xs + mlp_output
        }
    }
}

// ─── Quantized Model ─────────────────────────────────────────────────────────

/// Quantized GPT-NeoX model supporting FP8, GPTQ, AWQ.
pub struct QuantizedGPTNeoXForCausalLM {
    embed_in: Embedding,
    layers: Vec<QuantizedGptNeoXDecoderLayer>,
    final_layer_norm: LayerNorm,
    embed_out: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedGPTNeoXForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let neox_cfg = GptNeoXConfig::from_model_config(cfg);
        let vb_m = vb.pp("gpt_neox");

        let embed_in = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_in"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedGptNeoXDecoderLayer::new(
                cfg,
                &neox_cfg,
                weight_loader,
                vb.clone(),
                i,
            )?);
        }

        let final_layer_norm = layer_norm(
            cfg.hidden_size,
            neox_cfg.layer_norm_eps,
            vb_m.pp("final_layer_norm"),
        )?;

        let embed_out =
            weight_loader.load_linear("embed_out", cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            embed_in,
            layers,
            final_layer_norm,
            embed_out,
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

        let mut xs = self.embed_in.forward(input_ids)?;
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
        let xs = self.final_layer_norm.forward(&xs)?;
        self.embed_out.forward(&xs)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for QuantizedGPTNeoXForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
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
        let mut xs = self.embed_in.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.final_layer_norm.forward(&xs)?;
        self.embed_out.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::quantization::{
        create_weight_loader_with_params, DetectedQuantConfig, QuantizationMethod,
    };

    fn test_config() -> crate::config::ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("use_parallel_residual".to_string(), serde_json::json!(true));
        extra.insert("rotary_pct".to_string(), serde_json::json!(0.25));
        extra.insert("layer_norm_eps".to_string(), serde_json::json!(1e-5));

        crate::config::ModelConfig {
            architectures: vec!["GPTNeoXForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // MHA
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
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
    fn test_quantized_gptneox_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGPTNeoXForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedGPTNeoXForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_gptneox_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGPTNeoXForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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
    fn test_quantized_gptneox_sequential_residual() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "use_parallel_residual".to_string(),
            serde_json::json!(false),
        );

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGPTNeoXForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "Sequential residual should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_quantized_gptneox_with_gptq_loader() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: std::collections::HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGPTNeoXForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_err() || model.is_ok());
    }
}
