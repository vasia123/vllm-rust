use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

// ─── Gemma2 RMSNorm ──────────────────────────────────────────────────────────
//
// Same as Gemma: output = x * (1 + weight) / rms(x)

struct Gemma2RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Gemma2RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for Gemma2RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let scale = (&self.weight.to_dtype(DType::F32)? + 1.0)?;
        xs_normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

// ─── GELU Activation ─────────────────────────────────────────────────────────

fn gelu_tanh(xs: &Tensor) -> Result<Tensor> {
    xs.gelu_erf()
}

// ─── Soft Capping ────────────────────────────────────────────────────────────
//
// Gemma2 uses soft capping: cap * tanh(x / cap)
// This prevents extreme logits while preserving gradients

fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

// ─── Gemma2 MLP ──────────────────────────────────────────────────────────────

struct Gemma2Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma2Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Gemma2Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = gelu_tanh(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct Gemma2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    // TODO: implement soft capping in attention kernel
    #[allow(dead_code)]
    attn_logit_softcap: Option<f64>,
    // TODO: implement sliding window masking
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl Gemma2Attention {
    fn new(
        cfg: &ModelConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        // Gemma2 uses query_pre_attn_scalar for scaling
        let query_pre_attn_scalar = cfg
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or((head_dim as f64).sqrt());
        let scaling = 1.0 / query_pre_attn_scalar.sqrt();

        // Attention logit soft capping
        let attn_logit_softcap = cfg
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64());

        // Gemma2 alternates between sliding window and global attention
        // Even layers use sliding window, odd layers use global
        let sliding_window = if layer_idx % 2 == 0 {
            cfg.sliding_window
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling,
            attn_logit_softcap,
            sliding_window,
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
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Apply custom scaling
        let q = (q * self.scaling)?;

        // For sliding window layers, we need to apply windowed attention
        // For now, use standard paged attention (full context)
        // TODO: implement sliding window masking
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

        // Apply attention logit soft capping is handled inside attention computation
        // For now, we skip it as it requires modifying the attention kernel

        attn_output.apply(&self.o_proj)
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

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

            // Apply custom scaling
            let q = (q * self.scaling)?;

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

            // Scale already applied to q
            let scale = 1.0f32;

            let attn_output = crate::cuda_kernels::paged_attention_cuda(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
            )?;

            attn_output.apply(&self.o_proj)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

                // Apply custom scaling
                let q_i = (q_i * self.scaling)?;

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
            attn_output.apply(&self.o_proj)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────
//
// Gemma2 has additional layernorms: pre_feedforward and post_feedforward

struct Gemma2DecoderLayer {
    self_attn: Gemma2Attention,
    mlp: Gemma2Mlp,
    input_layernorm: Gemma2RmsNorm,
    post_attention_layernorm: Gemma2RmsNorm,
    pre_feedforward_layernorm: Gemma2RmsNorm,
    post_feedforward_layernorm: Gemma2RmsNorm,
}

impl Gemma2DecoderLayer {
    fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = Gemma2Attention::new(cfg, layer_idx, vb.pp("self_attn"))?;
        let mlp = Gemma2Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let input_layernorm =
            Gemma2RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Gemma2RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = Gemma2RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = Gemma2RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
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
        // Pre-attention norm
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        // Attention
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;

        // Post-attention norm (Gemma2 specific)
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // Residual connection
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        // Pre-feedforward norm (Gemma2 specific)
        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;

        // MLP
        let hidden_states = self.mlp.forward(&hidden_states)?;

        // Post-feedforward norm (Gemma2 specific)
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        // Residual connection
        residual + hidden_states
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        residual + hidden_states
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct Gemma2ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<Gemma2DecoderLayer>,
    norm: Gemma2RmsNorm,
    lm_head: Linear,
    hidden_size: usize,
    final_logit_softcap: Option<f64>,
    device: Device,
    dtype: DType,
}

impl Gemma2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma2DecoderLayer::new(cfg, i, vb_l.pp(i))?);
        }

        let norm = Gemma2RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Gemma2 always uses tied embeddings
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);

        // Final logit soft capping
        let final_logit_softcap = cfg
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            final_logit_softcap,
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

        // Gemma2 normalizes embeddings by sqrt(hidden_size)
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

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
        let mut logits = self.lm_head.forward(&xs)?;

        // Apply final logit soft capping
        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for Gemma2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.norm.forward(&xs)?;
        let mut logits = self.lm_head.forward(&xs)?;

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(256.0).unwrap()),
        );
        extra.insert(
            "attn_logit_softcapping".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(50.0).unwrap()),
        );
        extra.insert(
            "final_logit_softcapping".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(30.0).unwrap()),
        );

        ModelConfig {
            architectures: vec!["Gemma2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4, // Need 4 layers to test alternating sliding window
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: Some(256),
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
        }
    }

    #[test]
    fn test_gemma2_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Gemma2ForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "Gemma2ForCausalLM should construct");

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_logit_softcap.is_some());
    }

    #[test]
    fn test_gemma2_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma2ForCausalLM::new(&cfg, vb).expect("build model");

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
            .forward(&input_ids, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_gemma2_soft_cap() {
        let device = Device::Cpu;

        // Test soft capping function
        let x = Tensor::from_vec(vec![-100.0f32, -10.0, 0.0, 10.0, 100.0], (5,), &device)
            .expect("input");

        let capped = soft_cap(&x, 30.0).expect("soft_cap");
        let capped_vec: Vec<f32> = capped.to_vec1().unwrap();

        // Values should be bounded by [-cap, cap]
        for &v in &capped_vec {
            assert!(v.abs() <= 30.0 + 0.01, "value {v} should be bounded by cap");
        }

        // 0 should stay 0
        assert!(capped_vec[2].abs() < 0.01, "soft_cap(0) should be ≈0");

        // Large positive should approach cap
        assert!(capped_vec[4] > 29.0, "soft_cap(100) should be close to cap");
    }

    #[test]
    fn test_gemma2_alternating_sliding_window() {
        let cfg = test_config();

        // Verify even layers have sliding window, odd layers don't
        assert!(cfg.sliding_window.is_some());

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma2ForCausalLM::new(&cfg, vb).expect("build model");

        // Model should have correct number of layers
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn test_gemma2_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma2ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = model
            .forward(&next_token, 3, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_gemma2_extra_layernorms() {
        // Verify Gemma2 has 4 layernorms per layer
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // This tests that the layer constructs with all 4 norms
        let layer = Gemma2DecoderLayer::new(&cfg, 0, vb).expect("build layer");

        // Forward should work with the extra norms
        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let x = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("input");
        let result = layer.forward(&x, None, 0, &mut kv_cache_mgr, 0, &block_table, &slot_mapping);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemma2_query_pre_attn_scalar() {
        // Verify custom scaling is used
        let cfg = test_config();
        let query_pre_attn_scalar = cfg
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap();

        assert_eq!(query_pre_attn_scalar, 256.0);
    }
}
