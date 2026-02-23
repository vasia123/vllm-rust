//! Eagle-1 speculative decoding draft model for Llama 4.
//!
//! Eagle Llama4 generates draft tokens using the same architecture as the
//! regular Llama 4 decoder, with the Eagle-1 fc-fusion mechanism:
//!   `fc(cat(embed(token), hidden_state))` → Llama4 decoder layers → norm → logits
//!
//! The draft model shares the same Llama4 decoder layer structure (including
//! MoE and rope-free layers), but always runs on a single GPU (no TP).
//!
//! Reference: `reference/vllm/vllm/model_executor/models/llama4_eagle.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};
use crate::models::tp_layers::TpContext;

use super::eagle_llama::Eagle1DraftModel;
use super::llama4::{Llama4Config, Llama4DecoderLayer};

// ─── Eagle Llama4 Model ────────────────────────────────────────────────────

/// Eagle-1 speculative decoding draft model for Llama 4.
///
/// Uses standard Llama4 decoder layers with an fc fusion of token embeddings
/// and target model hidden states before the decoder stack.
pub struct EagleLlama4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    /// fc: `cat(embed, hidden)` [B, S, 2*H] → [B, S, H]
    fc: Linear,
    layers: Vec<Llama4DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    tp_ctx: TpContext,
    device: Device,
}

impl EagleLlama4ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let l4_cfg = Llama4Config::from_model_config(cfg);
        // Eagle always runs on a single GPU — no tensor parallelism.
        let pg = LocalProcessGroup::default();
        let tp_ctx = TpContext::single_gpu();

        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let fc = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb_m.pp("fc"))?;

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Llama4DecoderLayer::new_with_tp(
                cfg,
                &l4_cfg,
                i,
                vb_l.pp(i),
                &pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            // Eagle Llama4 may have a reduced draft vocabulary.
            let vocab_size = cfg
                .extra
                .get("draft_vocab_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(cfg.vocab_size);
            linear_no_bias(cfg.hidden_size, vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            fc,
            layers,
            norm,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
        })
    }
}

impl Eagle1DraftModel for EagleLlama4ForCausalLM {
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
        let embeds = self.embed_tokens.forward(input_ids)?;

        // Combine: fc(cat(embed, hidden)) → [B, S, H]
        let combined = Tensor::cat(&[&embeds, hidden_states], 2)?;
        let mut hs = self.fc.forward(&combined)?;

        let mask = if seq_len > 1 {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                hs.dtype(),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hs = layer.forward(
                &hs,
                mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        // Eagle-1 returns (postnorm, prenorm) — postnorm is used for logits,
        // prenorm is passed as hidden state to the next Eagle step.
        let prenorm = hs;
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
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager};
    use candle_core::DType;

    fn test_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["EagleLlama4ForCausalLM".to_string()],
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
            attention_bias: None,
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_eagle_llama4_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlama4ForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_eagle_llama4_forward_prefill() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlama4ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_cache(&cfg, &device);
        let mut bt = BlockTable::new(16);

        let seq_len = 4;
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slots = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, seq_len, 64), DType::F32, &device).unwrap();

        let (postnorm, prenorm) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();

        assert_eq!(postnorm.dims(), &[1, seq_len, 64]);
        assert_eq!(prenorm.dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_eagle_llama4_forward_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlama4ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_cache(&cfg, &device);
        let mut bt = BlockTable::new(16);

        kv.allocate_for_request(&mut bt, 1).unwrap();
        let slots = bt.slot_mapping(0, 1);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();

        let (postnorm, _) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();
        assert_eq!(postnorm.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_eagle_llama4_compute_logits() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlama4ForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 3, 64), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();
        // tie_word_embeddings=true → vocab_size=256
        assert_eq!(logits.dims(), &[1, 3, 256]);
    }

    #[test]
    fn test_eagle_llama4_draft_vocab_size() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;
        cfg.extra
            .insert("draft_vocab_size".to_string(), serde_json::json!(128));

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleLlama4ForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 128]);
    }
}
