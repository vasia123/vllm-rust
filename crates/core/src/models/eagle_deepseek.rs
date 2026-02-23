//! Eagle-1 speculative decoding draft model for DeepSeek V2/V3.
//!
//! Eagle DeepSeek generates draft tokens using DeepSeek V2 MLA decoder layers
//! with the Eagle-1 fc-fusion mechanism:
//!   `fc(cat(enorm(embed(token)), hnorm(hidden_state)))` → MLA layers → norm
//!
//! Key differences from Eagle Llama:
//! - Uses MLA attention (DeepSeek V2 layers), requires `KVCacheManager::new_mla()`
//! - Normalises each input separately with `enorm` / `hnorm` before concatenation
//! - Returns `(hidden_states, hidden_states)` — postnorm and prenorm are the same
//!
//! Architecture name in checkpoints: `EagleDeepSeekMTPModel`
//!
//! Weight paths: `model.embed_tokens`, `model.enorm`, `model.hnorm`, `model.fc`,
//! `model.layers.{i}.*`, `model.norm`, `lm_head`.
//!
//! Reference: `reference/vllm/vllm/model_executor/models/deepseek_eagle.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};

use super::deepseek::DeepSeekDecoderLayer;
use super::eagle_llama::Eagle1DraftModel;

// ─── Eagle DeepSeek Model ─────────────────────────────────────────────────

/// Eagle-1 speculative decoding draft model for DeepSeek V2/V3.
///
/// Uses MLA attention via DeepSeek V2 decoder layers. Requires an MLA-backed
/// `KVCacheManager` (created with `KVCacheManager::new_mla()`).
///
/// The fc-fusion normalises both inputs before concatenation:
/// `fc(cat(enorm(embed(x)), hnorm(target_hs)))` → layers → norm.
pub struct EagleDeepSeekForCausalLM {
    embed_tokens: candle_nn::Embedding,
    /// Normalise token embeddings before fc-fusion.
    enorm: RmsNorm,
    /// Normalise target hidden states before fc-fusion.
    hnorm: RmsNorm,
    /// fc: `cat(enorm(embed), hnorm(hidden))` [B, S, 2*H] → [B, S, H]
    fc: Linear,
    layers: Vec<DeepSeekDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
}

impl EagleDeepSeekForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let enorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("enorm"))?;
        let hnorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("hnorm"))?;

        let fc = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb_m.pp("fc"))?;

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DeepSeekDecoderLayer::new(cfg, i, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            enorm,
            hnorm,
            fc,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
        })
    }
}

impl Eagle1DraftModel for EagleDeepSeekForCausalLM {
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
        // Normalise each input separately before fc-fusion.
        let normed_embeds = self.enorm.forward(&embeds)?;
        let normed_hidden = self.hnorm.forward(hidden_states)?;

        let combined = Tensor::cat(&[&normed_embeds, &normed_hidden], 2)?;
        let mut xs = self.fc.forward(&combined)?;

        let mask = if seq_len > 1 {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                xs.dtype(),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = if seq_len > 1 {
                layer.forward(
                    &xs,
                    mask.as_ref(),
                    seqlen_offset,
                    kv_cache_mgr,
                    layer_idx,
                    block_table.block_ids(),
                    slot_mapping,
                )?
            } else {
                layer.forward_decode(
                    &xs,
                    seqlen_offset,
                    kv_cache_mgr,
                    layer_idx,
                    block_table.block_ids(),
                    slot_mapping,
                )?
            };
        }

        // Apply final norm — postnorm and prenorm are the same in this model.
        let out = self.norm.forward(&xs)?;
        Ok((out.clone(), out))
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
    use crate::kv_cache::{mla_cache_config::MLACacheConfig, KVCacheManager};
    use candle_core::DType;

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "qk_nope_head_dim".into(),
            serde_json::Value::Number(16.into()),
        );
        extra.insert(
            "qk_rope_head_dim".into(),
            serde_json::Value::Number(8.into()),
        );
        extra.insert("v_head_dim".into(), serde_json::Value::Number(16.into()));
        extra.insert("kv_lora_rank".into(), serde_json::Value::Number(32.into()));

        ModelConfig {
            architectures: vec!["EagleDeepSeekMTPModel".to_string()],
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 24, // qk_nope + qk_rope
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn create_mla_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32, // kv_lora_rank
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            cfg.num_attention_heads,
            4,  // block_size
            16, // num_blocks
            cfg.num_hidden_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).unwrap()
    }

    #[test]
    fn test_eagle_deepseek_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleDeepSeekForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_eagle_deepseek_forward_prefill() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleDeepSeekForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_mla_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);

        let seq_len = 4;
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slots: Vec<usize> = (0..seq_len).collect();

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, seq_len, 128), DType::F32, &device).unwrap();

        let (postnorm, prenorm) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();

        assert_eq!(postnorm.dims(), &[1, seq_len, 128]);
        assert_eq!(prenorm.dims(), &[1, seq_len, 128]);
    }

    #[test]
    fn test_eagle_deepseek_forward_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleDeepSeekForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_mla_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);

        kv.allocate_for_request(&mut bt, 1).unwrap();
        let slots = vec![0_usize];

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, 1, 128), DType::F32, &device).unwrap();

        let (postnorm, _) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();
        assert_eq!(postnorm.dims(), &[1, 1, 128]);
    }

    #[test]
    fn test_eagle_deepseek_returns_same_tensor() {
        // EagleDeepSeek returns (postnorm, postnorm) — both identical.
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleDeepSeekForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_mla_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);

        kv.allocate_for_request(&mut bt, 1).unwrap();
        let slots = vec![0_usize];

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, 1, 128), DType::F32, &device).unwrap();

        let (out1, out2) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();

        let v1: Vec<f32> = out1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_eagle_deepseek_compute_logits() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleDeepSeekForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 2, 128), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();
        assert_eq!(logits.dims(), &[1, 2, 1000]);
    }
}
