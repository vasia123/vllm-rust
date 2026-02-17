//! MiniMaxM2 model implementation.
//!
//! MiniMaxM2 is a Mixture-of-Experts model with:
//! - Q/K normalization (RMSNorm per head)
//! - Configurable rotary dimension (partial RoPE)
//! - MoE with optional e_score_correction_bias
//! - FP32 router with configurable scoring function
//!
//! Architecture:
//! ```text
//! Embedding -> [DecoderLayer x N] -> RMSNorm -> LM Head
//!
//! DecoderLayer:
//!   RMSNorm -> Attention(QK-norm, partial RoPE) -> RMSNorm -> MoE
//! ```
//!
//! Config keys from extra:
//! - `num_local_experts`: number of experts
//! - `num_experts_per_tok`: top-k experts
//! - `rotary_dim`: rotary embedding dimension (may differ from head_dim)
//! - `use_routing_bias`: whether MoE uses e_score_correction_bias
//! - `scoring_func`: "softmax" or "sigmoid"
//! - `attention_bias`: whether QKV has bias

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};

// ─── Config ─────────────────────────────────────────────────────────────────

struct MiniMaxM2Config {
    num_local_experts: usize,
    num_experts_per_tok: usize,
    rotary_dim: usize,
    use_routing_bias: bool,
    scoring_func: String,
    attention_bias: bool,
}

impl MiniMaxM2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(8);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let rotary_dim = cfg
            .extra
            .get("rotary_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let use_routing_bias = cfg
            .extra
            .get("use_routing_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let scoring_func = cfg
            .extra
            .get("scoring_func")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
            .to_string();

        let attention_bias = cfg.attention_bias.unwrap_or(false);

        Self {
            num_local_experts,
            num_experts_per_tok,
            rotary_dim,
            use_routing_bias,
            scoring_func,
            attention_bias,
        }
    }
}

// ─── MoE ────────────────────────────────────────────────────────────────────

struct MiniMaxM2MoEExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MiniMaxM2MoEExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("w1"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("w2"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("w3"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

struct MiniMaxM2MoE {
    gate: Linear,
    experts: Vec<MiniMaxM2MoEExpert>,
    e_score_correction_bias: Option<Tensor>,
    num_experts: usize,
    top_k: usize,
    scoring_func: String,
}

impl MiniMaxM2MoE {
    fn new(cfg: &ModelConfig, m2_cfg: &MiniMaxM2Config, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, m2_cfg.num_local_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(m2_cfg.num_local_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..m2_cfg.num_local_experts {
            experts.push(MiniMaxM2MoEExpert::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb_experts.pp(i),
            )?);
        }

        let e_score_correction_bias = if m2_cfg.use_routing_bias {
            Some(
                vb.get(m2_cfg.num_local_experts, "e_score_correction_bias")?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            e_score_correction_bias,
            num_experts: m2_cfg.num_local_experts,
            top_k: m2_cfg.num_experts_per_tok,
            scoring_func: m2_cfg.scoring_func.clone(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Router in FP32
        let mut router_logits = self.gate.forward(&xs_2d.to_dtype(DType::F32)?)?;

        // Apply e_score_correction_bias if present
        if let Some(ref bias) = self.e_score_correction_bias {
            router_logits = router_logits.broadcast_add(bias)?;
        }

        // Scoring function
        let routing_weights = match self.scoring_func.as_str() {
            "sigmoid" => candle_nn::ops::sigmoid(&router_logits)?,
            _ => candle_nn::ops::softmax_last_dim(&router_logits)?,
        };

        let routing_data: Vec<f32> = routing_weights.flatten_all()?.to_vec1()?;
        let flat_data: Vec<f32> = xs_2d.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut output_data = vec![0.0f32; num_tokens * hidden_dim];

        for token_idx in 0..num_tokens {
            let weights =
                &routing_data[token_idx * self.num_experts..(token_idx + 1) * self.num_experts];

            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Renormalize top-k weights
            let top_sum: f32 = indexed[..self.top_k].iter().map(|(_, w)| w).sum();

            let token_input = Tensor::from_vec(
                flat_data[token_idx * hidden_dim..(token_idx + 1) * hidden_dim].to_vec(),
                (1, hidden_dim),
                xs.device(),
            )?;

            for &(expert_idx, weight) in indexed[..self.top_k].iter() {
                let norm_weight = if top_sum > 0.0 {
                    weight / top_sum
                } else {
                    1.0 / self.top_k as f32
                };
                let expert_out = self.experts[expert_idx].forward(&token_input)?;
                let expert_data: Vec<f32> =
                    expert_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for j in 0..hidden_dim {
                    output_data[token_idx * hidden_dim + j] += norm_weight * expert_data[j];
                }
            }
        }

        Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&orig_shape),
            xs.device(),
        )?
        .to_dtype(xs.dtype())
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct MiniMaxM2Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl MiniMaxM2Attention {
    fn new(cfg: &ModelConfig, m2_cfg: &MiniMaxM2Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let total_qkv = q_size + 2 * kv_size;

        let qkv_proj = if m2_cfg.attention_bias {
            let w = vb
                .pp("qkv_proj")
                .get((total_qkv, cfg.hidden_size), "weight")?;
            let b = vb.pp("qkv_proj").get(total_qkv, "bias")?;
            Linear::new(w, Some(b))
        } else {
            linear_no_bias(cfg.hidden_size, total_qkv, vb.pp("qkv_proj"))?
        };

        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        // Per-head Q/K normalization
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Partial RoPE based on rotary_dim
        let rotary_emb = if m2_cfg.rotary_dim < head_dim {
            let partial_factor = m2_cfg.rotary_dim as f64 / head_dim as f64;
            RotaryEmbedding::new_partial(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                partial_factor,
                true, // neox style
                vb.dtype(),
                vb.device(),
            )?
        } else {
            RotaryEmbedding::new(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                vb.dtype(),
                vb.device(),
            )?
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
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

        // Per-head RMSNorm on Q and K
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        // RoPE
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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct MiniMaxM2DecoderLayer {
    self_attn: MiniMaxM2Attention,
    moe: MiniMaxM2MoE,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MiniMaxM2DecoderLayer {
    fn new(cfg: &ModelConfig, m2_cfg: &MiniMaxM2Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = MiniMaxM2Attention::new(cfg, m2_cfg, vb.pp("self_attn"))?;
        let moe = MiniMaxM2MoE::new(cfg, m2_cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            moe,
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
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Pre-norm + attention
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;

        // Pre-norm + MoE
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.moe.forward(&xs)?;
        let hidden_states = (xs + residual)?;

        let device = hidden_states.device().clone();
        Ok((hidden_states, Tensor::zeros(1, DType::F32, &device)?))
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_decode_batch(&xs, sequences, cache_engine)?;
        let xs = (xs + residual)?;

        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.moe.forward(&xs)?;
        residual + xs
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MiniMaxM2ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMaxM2DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl MiniMaxM2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let m2_cfg = MiniMaxM2Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MiniMaxM2DecoderLayer::new(cfg, &m2_cfg, vb_layers.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
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

impl crate::engine::ModelForward for MiniMaxM2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

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
            let (hidden, _residual) = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
            xs = hidden;
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
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_m2_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("rotary_dim".to_string(), serde_json::json!(16));
        extra.insert("use_routing_bias".to_string(), serde_json::json!(false));
        extra.insert("scoring_func".to_string(), serde_json::json!("softmax"));

        ModelConfig {
            architectures: vec!["MiniMaxM2ForCausalLM".to_string()],
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

    fn test_m2_config_with_bias() -> ModelConfig {
        let mut cfg = test_m2_config();
        cfg.extra
            .insert("use_routing_bias".to_string(), serde_json::json!(true));
        cfg.extra
            .insert("scoring_func".to_string(), serde_json::json!("sigmoid"));
        cfg
    }

    fn test_m2_config_partial_rotary() -> ModelConfig {
        let mut cfg = test_m2_config();
        // head_dim=16, rotary_dim=8 -> partial RoPE
        cfg.extra
            .insert("rotary_dim".to_string(), serde_json::json!(8));
        cfg
    }

    fn create_cache(cfg: &ModelConfig) -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_parsing_defaults() {
        let cfg = test_m2_config();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);

        assert_eq!(m2_cfg.num_local_experts, 4);
        assert_eq!(m2_cfg.num_experts_per_tok, 2);
        assert_eq!(m2_cfg.rotary_dim, 16);
        assert!(!m2_cfg.use_routing_bias);
        assert_eq!(m2_cfg.scoring_func, "softmax");
        assert!(!m2_cfg.attention_bias);
    }

    #[test]
    fn test_config_parsing_with_bias() {
        let cfg = test_m2_config_with_bias();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);

        assert!(m2_cfg.use_routing_bias);
        assert_eq!(m2_cfg.scoring_func, "sigmoid");
    }

    #[test]
    fn test_config_partial_rotary() {
        let cfg = test_m2_config_partial_rotary();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);

        assert_eq!(m2_cfg.rotary_dim, 8);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_construction() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxM2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxM2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_construction_partial_rotary() {
        let cfg = test_m2_config_partial_rotary();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxM2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxM2ForCausalLM with partial rotary should construct: {:?}",
            model.err()
        );
    }

    // ─── Component Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_moe_forward() {
        let cfg = test_m2_config();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let moe = MiniMaxM2MoE::new(&cfg, &m2_cfg, vb.pp("moe")).expect("moe");
        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_attention_qk_norm() {
        let cfg = test_m2_config();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = MiniMaxM2Attention::new(&cfg, &m2_cfg, vb.pp("attn")).expect("attn");
        assert_eq!(attn.num_heads, cfg.num_attention_heads);
        assert_eq!(attn.num_kv_heads, cfg.num_key_value_heads);
        assert_eq!(attn.head_dim, cfg.head_dim);
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_forward_shape() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let batch_size = 1;
        let seq_len = 3;
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
    fn test_prefill_then_decode() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
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
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");

        let logits = model
            .forward(&next, 3, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_model_forward_trait() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_device() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("model");

        assert!(matches!(ModelForward::device(&model), Device::Cpu));
    }

    #[test]
    fn test_partial_rotary_forward() {
        let cfg = test_m2_config_partial_rotary();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("forward with partial rotary");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
