//! Eagle 1 speculative decoding draft model (Llama architecture).
//!
//! Eagle 1 generates draft tokens by combining target model hidden states
//! with token embeddings via an FC layer:
//!   `fc(cat(embed(token), hidden_state))` → decoder layers → output
//!
//! Key differences from Eagle 3:
//! - FC projection: `cat(embed, hidden) → fc → hidden_size` (Eagle 3 uses layer-0 concat)
//! - First decoder layer has `input_layernorm` replaced with Identity
//! - No auxiliary hidden states or prenorm chaining
//!
//! Reference: `reference/vllm/vllm/model_executor/models/llama_eagle.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

// ─── Eagle1 Config ─────────────────────────────────────────────────────────

/// Configuration for Eagle 1 draft models.
#[derive(Debug, Clone)]
pub struct Eagle1Config {
    /// Whether the attention layers use bias.
    pub attention_bias: bool,
    /// Draft vocabulary size (may differ from target vocab).
    pub draft_vocab_size: Option<usize>,
    /// Logit scaling factor.
    pub logit_scale: f64,
}

impl Eagle1Config {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let attention_bias = cfg.attention_bias.unwrap_or(false);

        let draft_vocab_size = cfg
            .extra
            .get("draft_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let logit_scale = cfg
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Self {
            attention_bias,
            draft_vocab_size,
            logit_scale,
        }
    }
}

// ─── SwiGLU MLP ────────────────────────────────────────────────────────────

struct Eagle1Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Eagle1Mlp {
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Eagle1 Attention ──────────────────────────────────────────────────────

struct Eagle1Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Eagle1Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, attention_bias: bool) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = candle_nn::linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("v_proj"),
        )?;
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
}

// ─── Eagle1 Decoder Layer ──────────────────────────────────────────────────

struct Eagle1DecoderLayer {
    self_attn: Eagle1Attention,
    mlp: Eagle1Mlp,
    input_layernorm: Option<RmsNorm>,
    post_attention_layernorm: RmsNorm,
}

impl Eagle1DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        eagle_cfg: &Eagle1Config,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Eagle1Attention::new(cfg, vb.pp("self_attn"), eagle_cfg.attention_bias)?;
        let mlp = Eagle1Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;

        // Eagle 1: first layer has no input_layernorm (Identity)
        let input_layernorm = if layer_idx == 0 {
            None
        } else {
            Some(rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?)
        };

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
        residual: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Pre-norm residual: x = x + residual; residual = x; x = norm(x)
        let (hs, residual) = if let Some(res) = residual {
            let xs = (xs + res)?;
            let res = xs.clone();
            let xs = if let Some(ref norm) = self.input_layernorm {
                norm.forward(&xs)?
            } else {
                xs
            };
            (xs, res)
        } else {
            // First layer: no residual
            let xs = if let Some(ref norm) = self.input_layernorm {
                norm.forward(xs)?
            } else {
                xs.clone()
            };
            (xs.clone(), xs)
        };

        let hs = self.self_attn.forward(
            &hs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;

        // Post-attention: residual + norm + MLP
        let xs = (hs + &residual)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;

        Ok((xs, residual))
    }
}

// ─── Eagle1 Llama Model ────────────────────────────────────────────────────

/// Eagle 1 draft model with Llama architecture.
///
/// Forward: `fc(cat(embed(token), hidden_state))` → decoder layers → norm → logits
pub struct EagleLlamaForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<Eagle1DecoderLayer>,
    fc: Linear,
    norm: RmsNorm,
    lm_head: Linear,
    #[allow(dead_code)]
    logit_scale: f64,
    device: Device,
}

/// Trait for Eagle 1 draft models.
///
/// Similar to `Eagle3DraftModel` but with the Eagle 1 forward signature:
/// - Takes `(input_ids, hidden_states)` — no aux hidden state
/// - Returns `(post_norm, pre_norm)` hidden states
pub trait Eagle1DraftModel: Send {
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)>;

    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor>;
    fn device(&self) -> &Device;
}

impl EagleLlamaForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let eagle_cfg = Eagle1Config::from_model_config(cfg);

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Eagle1DecoderLayer::new(cfg, &eagle_cfg, vb_l.pp(i), i)?);
        }

        // FC: cat(embed, hidden) = 2*hidden_size → hidden_size
        let fc = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb.pp("model.fc"))?;

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // LM head: try loading explicit lm_head, fall back to tied embeddings
        let lm_head = if let Ok(lm) = candle_nn::linear_no_bias(
            cfg.hidden_size,
            eagle_cfg.draft_vocab_size.unwrap_or(cfg.vocab_size),
            vb.pp("lm_head"),
        ) {
            lm
        } else {
            let emb = embed_tokens.embeddings().clone();
            candle_nn::Linear::new(emb, None)
        };

        Ok(Self {
            embed_tokens,
            layers,
            fc,
            norm,
            lm_head,
            logit_scale: eagle_cfg.logit_scale,
            device: vb.device().clone(),
        })
    }
}

impl Eagle1DraftModel for EagleLlamaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len) = input_ids.dims2()?;
        let input_embeds = self.embed_tokens.forward(input_ids)?;

        // Combine: fc(cat(embed, hidden)) → hidden_size
        let combined = Tensor::cat(&[&input_embeds, hidden_states], 2)?;
        let mut hs = self.fc.forward(&combined)?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                hs.dtype(),
                hs.device(),
            )?)
        };

        let mut residual: Option<Tensor> = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (new_hs, new_residual) = layer.forward(
                &hs,
                residual.as_ref(),
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
            hs = new_hs;
            residual = Some(new_residual);
        }

        // Final: hs + residual
        let residual = residual.expect("at least one layer");
        let prenorm = (hs + &residual)?;
        let postnorm = self.norm.forward(&prenorm)?;

        Ok((postnorm, prenorm))
    }

    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use candle_core::DType;

    fn test_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["EagleLlamaForCausalLM".to_string()],
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
    fn test_eagle1_config_defaults() {
        let cfg = test_config();
        let eagle_cfg = Eagle1Config::from_model_config(&cfg);
        assert!(!eagle_cfg.attention_bias);
        assert_eq!(eagle_cfg.draft_vocab_size, None);
        assert_eq!(eagle_cfg.logit_scale, 1.0);
    }

    #[test]
    fn test_eagle1_config_custom() {
        let mut cfg = test_config();
        cfg.extra
            .insert("draft_vocab_size".to_string(), serde_json::json!(128));
        cfg.extra
            .insert("logit_scale".to_string(), serde_json::json!(2.0));
        cfg.attention_bias = Some(true);

        let eagle_cfg = Eagle1Config::from_model_config(&cfg);
        assert!(eagle_cfg.attention_bias);
        assert_eq!(eagle_cfg.draft_vocab_size, Some(128));
        assert_eq!(eagle_cfg.logit_scale, 2.0);
    }

    #[test]
    fn test_eagle1_model_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = EagleLlamaForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Eagle1 Llama should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_eagle1_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlamaForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let seq_len = 3;
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (1, seq_len, 64), &device).unwrap();

        let (postnorm, prenorm) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(postnorm.dims(), &[1, seq_len, 64]);
        assert_eq!(prenorm.dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_eagle1_compute_logits() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlamaForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::randn(0f32, 1.0, (1, 1, 64), &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();

        assert_eq!(logits.dims(), &[1, 1, 256]);
    }

    #[test]
    fn test_eagle1_single_token_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlamaForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Single token decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 1);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (1, 1, 64), &device).unwrap();

        let (postnorm, _) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(postnorm.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_eagle1_first_layer_no_layernorm() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlamaForCausalLM::new(&cfg, vb).unwrap();

        // First layer should have no input_layernorm
        assert!(model.layers[0].input_layernorm.is_none());
        // Second layer should have input_layernorm
        assert!(model.layers[1].input_layernorm.is_some());
    }

    #[test]
    fn test_eagle1_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlamaForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill with 3 tokens
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (1, 3, 64), &device).unwrap();

        let (postnorm, _) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();
        assert_eq!(postnorm.dims(), &[1, 3, 64]);
        block_table.advance(3);

        // Decode single token at offset=3
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (1, 1, 64), &device).unwrap();

        let (postnorm, _) = model
            .forward(
                &input_ids,
                &hidden_states,
                3,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();
        assert_eq!(postnorm.dims(), &[1, 1, 64]);
    }
}
