//! MiniCPM model implementation.
//!
//! MiniCPM is a Llama-variant with three unique scaling factors:
//! - `scale_emb`: multiplier applied to token embeddings
//! - `scale_depth / sqrt(num_layers)`: multiplier applied to each layer's residual
//! - `scale_width = hidden_size / dim_model_base`: divisor applied to final hidden states
//!
//! Also supports FatReLU activation (threshold-based ReLU gating) and optional MoE.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

// ─── MiniCPM Config ─────────────────────────────────────────────────────────

struct MiniCPMConfig {
    scale_emb: f64,
    scale_depth: f64,
    dim_model_base: f64,
    num_experts: usize,
    num_experts_per_tok: usize,
    hidden_act: String,
    #[allow(dead_code)]
    hidden_act_param: f64,
}

impl MiniCPMConfig {
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
        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let hidden_act = cfg.hidden_act.clone();
        let hidden_act_param = cfg
            .extra
            .get("hidden_act_param")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        Self {
            scale_emb,
            scale_depth,
            dim_model_base,
            num_experts,
            num_experts_per_tok,
            hidden_act,
            hidden_act_param,
        }
    }

    fn residual_scale(&self, num_layers: usize) -> f64 {
        self.scale_depth / (num_layers as f64).sqrt()
    }

    fn output_scale(&self, hidden_size: usize) -> f64 {
        hidden_size as f64 / self.dim_model_base
    }
}

// ─── MLP ────────────────────────────────────────────────────────────────────

enum MiniCPMActivation {
    Silu,
    FatRelu(f64),
}

struct MiniCPMMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: MiniCPMActivation,
}

impl MiniCPMMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: MiniCPMActivation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;

        let activated = match &self.activation {
            MiniCPMActivation::Silu => (candle_nn::ops::silu(&gate)? * up)?,
            MiniCPMActivation::FatRelu(threshold) => {
                // FatReLU: zero out values below threshold, then gate with up
                let threshold_t = Tensor::full(*threshold as f32, gate.shape(), gate.device())?
                    .to_dtype(gate.dtype())?;
                let mask = gate.ge(&threshold_t)?;
                let dtype = gate.dtype();
                let gated = (gate * mask.to_dtype(dtype)?)?;
                (gated * up)?
            }
        };

        self.down_proj.forward(&activated)
    }
}

// ─── Feed Forward (Dense or MoE) ───────────────────────────────────────────

enum MiniCPMFeedForward {
    Dense(MiniCPMMlp),
    Moe(MoELayer),
}

impl MiniCPMFeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct MiniCPMAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MiniCPMAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
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

struct MiniCPMDecoderLayer {
    self_attn: MiniCPMAttention,
    feed_forward: MiniCPMFeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_scale: f64,
}

impl MiniCPMDecoderLayer {
    fn new(cfg: &ModelConfig, mini_cfg: &MiniCPMConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = MiniCPMAttention::new(cfg, vb.pp("self_attn"))?;

        let feed_forward = if mini_cfg.num_experts > 0 {
            let moe_cfg = MoELayerConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: cfg.intermediate_size,
                num_experts: mini_cfg.num_experts,
                top_k: mini_cfg.num_experts_per_tok,
                renormalize: true,
                inplace: true,
                is_act_and_mul: true,
            };
            MiniCPMFeedForward::Moe(MoELayer::new(moe_cfg, vb.pp("mlp"))?)
        } else {
            let activation = match mini_cfg.hidden_act.as_str() {
                "fatrelu" => MiniCPMActivation::FatRelu(mini_cfg.hidden_act_param),
                _ => MiniCPMActivation::Silu,
            };
            MiniCPMFeedForward::Dense(MiniCPMMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                activation,
                vb.pp("mlp"),
            )?)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let residual_scale = mini_cfg.residual_scale(cfg.num_hidden_layers);

        Ok(Self {
            self_attn,
            feed_forward,
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
        // Self attention with scaled residual
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

        // FFN with scaled residual
        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let hidden = self.feed_forward.forward(&hidden)?;
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
        let hidden = self.feed_forward.forward(&hidden)?;
        residual + (hidden * self.residual_scale)?
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MiniCPMForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<MiniCPMDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    scale_emb: f64,
    output_scale: f64,
    device: Device,
    dtype: DType,
}

impl MiniCPMForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mini_cfg = MiniCPMConfig::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MiniCPMDecoderLayer::new(cfg, &mini_cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let output_scale = mini_cfg.output_scale(cfg.hidden_size);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            scale_emb: mini_cfg.scale_emb,
            output_scale,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.embed_tokens.forward(input_ids)?;
        hidden * self.scale_emb
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for MiniCPMForCausalLM {
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

        let mut xs = self.embed(input_ids)?;
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
        // Apply output scaling (inverse of scale_width)
        xs = (xs / self.output_scale)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        xs = self.norm.forward(&xs)?;
        xs = (xs / self.output_scale)?;
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

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("scale_emb".into(), serde_json::Value::from(12.0));
        extra.insert("scale_depth".into(), serde_json::Value::from(1.4));
        extra.insert("dim_model_base".into(), serde_json::Value::from(256.0));

        ModelConfig {
            architectures: vec!["MiniCPMForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_minicpm_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MiniCPMForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniCPMForCausalLM should construct: {:?}",
            model.err()
        );
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
        assert!((model.scale_emb - 12.0).abs() < 1e-6);
        assert!((model.output_scale - 0.25).abs() < 1e-6); // 64 / 256 = 0.25
    }

    #[test]
    fn test_minicpm_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPMForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

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
        .expect("forward pass");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_minicpm_scaling_factors() {
        let mini_cfg = MiniCPMConfig {
            scale_emb: 12.0,
            scale_depth: 1.4,
            dim_model_base: 256.0,
            num_experts: 0,
            num_experts_per_tok: 2,
            hidden_act: "silu".to_string(),
            hidden_act_param: 0.0,
        };

        // residual_scale = scale_depth / sqrt(num_layers)
        let rs = mini_cfg.residual_scale(4);
        assert!((rs - 1.4 / 2.0).abs() < 1e-6); // sqrt(4) = 2

        // output_scale = hidden_size / dim_model_base
        let os = mini_cfg.output_scale(64);
        assert!((os - 0.25).abs() < 1e-6); // 64 / 256
    }

    #[test]
    fn test_minicpm_with_moe_config() {
        let mut cfg = test_config();
        cfg.extra
            .insert("num_experts".into(), serde_json::Value::from(4));
        cfg.extra
            .insert("num_experts_per_tok".into(), serde_json::Value::from(2));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MiniCPMForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniCPMForCausalLM with MoE should construct: {:?}",
            model.err()
        );
    }
}
