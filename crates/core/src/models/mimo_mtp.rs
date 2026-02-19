//! MiMo Multi-Token Prediction (MTP) draft model.
//!
//! MiMo MTP uses a reversed-cat fusion (hidden_states first, then embeddings)
//! with Qwen2 decoder blocks. It is **single-token-only**: only `spec_step_idx == 0`
//! is supported, producing one draft token per target step.
//!
//! # Architecture
//!
//! ```text
//! target_hs       → hidden_layernorm → │
//!                                      ├─ cat ─→ input_proj ─→ mtp_block ─→ final_layernorm
//! embed(input_ids) → token_layernorm  → │
//!
//! final_layernorm output → lm_head → logits
//! ```
//!
//! Note: cat order is **[hidden_states, embeddings]**, reversed from DeepSeek.
//!
//! # Weight paths
//!
//! MTP layer is at absolute index `num_hidden_layers` in `model.mtp_layers`:
//!   - `model.mtp_layers.{N}.token_layernorm`
//!   - `model.mtp_layers.{N}.hidden_layernorm`
//!   - `model.mtp_layers.{N}.input_proj`
//!   - `model.mtp_layers.{N}.final_layernorm`
//!   - `model.mtp_layers.{N}.mtp_block.*`   (Qwen2DecoderLayer structure)
//!
//! The `lm_head` is loaded from `model.lm_head` (same as the target model).
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/mimo_mtp.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};
use crate::models::tp_layers::TpContext;

use super::mtp_base::MtpDraftModel;
use super::qwen2::Qwen2DecoderLayer;

// ─── Layer ──────────────────────────────────────────────────────────────────

/// One MiMo MTP prediction layer.
struct MiMoMtpLayer {
    token_layernorm: RmsNorm,
    hidden_layernorm: RmsNorm,
    input_proj: Linear,
    final_layernorm: RmsNorm,
    mtp_block: Qwen2DecoderLayer,
}

impl MiMoMtpLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let pg = LocalProcessGroup::new();
        let token_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("token_layernorm"))?;
        let hidden_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hidden_layernorm"))?;
        let input_proj = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb.pp("input_proj"))?;
        let final_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("final_layernorm"))?;
        let mtp_block = Qwen2DecoderLayer::new_with_tp(cfg, vb.pp("mtp_block"), &pg)?;
        Ok(Self {
            token_layernorm,
            hidden_layernorm,
            input_proj,
            final_layernorm,
            mtp_block,
        })
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// MiMo Multi-Token Prediction draft model.
///
/// Single-token-only: only `spec_step_idx == 0` is supported.
pub struct MiMoMtpModel {
    embed_tokens: Embedding,
    layer: MiMoMtpLayer,
    lm_head: Linear,
    tp_ctx: TpContext,
    device: Device,
}

impl MiMoMtpModel {
    /// Load the MiMo MTP model from a VarBuilder rooted at the model file.
    ///
    /// The single MTP layer is at `model.mtp_layers.{num_hidden_layers}`.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // MTP layer at absolute index num_hidden_layers
        let abs_idx = cfg.num_hidden_layers;
        let layer = MiMoMtpLayer::new(cfg, vb_m.pp("mtp_layers").pp(abs_idx))?;

        // LM head shared with main model
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_m.pp("lm_head"))?;

        Ok(Self {
            embed_tokens,
            layer,
            lm_head,
            tp_ctx: TpContext::single_gpu(),
            device: vb.device().clone(),
        })
    }
}

impl MtpDraftModel for MiMoMtpModel {
    /// Run one MiMo MTP forward pass.
    ///
    /// MiMo MTP is single-token-only: returns an error if `spec_step_idx > 0`.
    /// The fusion cat order is reversed: `[hidden_states, embeddings]`.
    fn forward(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        seqlen_offset: usize,
        spec_step_idx: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        if spec_step_idx > 0 {
            candle_core::bail!(
                "MiMoMtpModel is single-token-only: spec_step_idx must be 0, got {}",
                spec_step_idx
            );
        }

        let (_, seq_len) = input_ids.dims2()?;

        // Embed input tokens: [1, seq_len, hidden_size]
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // Normalize: token embeddings and target hidden states
        let normed_embeds = self.layer.token_layernorm.forward(&inputs_embeds)?;
        let normed_hs = self
            .layer
            .hidden_layernorm
            .forward(previous_hidden_states)?;

        // Cat order is REVERSED vs DeepSeek: [hidden_states, embeddings]
        let combined = Tensor::cat(&[&normed_hs, &normed_embeds], 2)?;
        let hidden = self.layer.input_proj.forward(&combined)?;

        // Run through Qwen2 transformer block
        let mask = if seq_len > 1 {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                hidden.dtype(),
                hidden.device(),
            )?)
        } else {
            None
        };
        let hidden = self.layer.mtp_block.forward(
            &hidden,
            mask.as_ref(),
            seqlen_offset,
            kv_cache_mgr,
            0, // MTP has its own single-layer KV cache at index 0
            block_table,
            slot_mapping,
            &self.tp_ctx,
        )?;

        // Apply final layer norm
        let hidden = self.layer.final_layernorm.forward(&hidden)?;

        Ok(hidden)
    }

    fn compute_logits(&self, hidden_states: &Tensor, _spec_step_idx: usize) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }

    fn num_mtp_layers(&self) -> usize {
        1
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use candle_core::DType;

    fn make_cfg() -> ModelConfig {
        ModelConfig {
            architectures: vec!["MiMoMTPModel".to_string()],
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let cache_config = CacheConfig {
            block_size: 4,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        KVCacheManager::new(&cache_config).unwrap()
    }

    #[test]
    fn test_mimo_mtp_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiMoMtpModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiMoMtpModel creation failed: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().num_mtp_layers(), 1);
    }

    #[test]
    fn test_mimo_mtp_single_token_constraint() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiMoMtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        let mut bt = BlockTable::new(4);
        kv_cache.allocate_for_request(&mut bt, 1).unwrap();
        let slot_mapping = bt.slot_mapping(0, 1);

        let tok = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let hs = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();

        // spec_step_idx=1 must fail
        let result = model.forward(&tok, &hs, 0, 1, &mut kv_cache, &bt, &slot_mapping);
        assert!(result.is_err(), "Should fail for spec_step_idx=1");
    }

    #[test]
    fn test_mimo_mtp_decode_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiMoMtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        let mut bt = BlockTable::new(4);
        kv_cache.allocate_for_request(&mut bt, 1).unwrap();
        let slot_mapping = bt.slot_mapping(0, 1);

        let tok = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let hs = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();

        let result = model.forward(&tok, &hs, 0, 0, &mut kv_cache, &bt, &slot_mapping);
        assert!(result.is_ok(), "decode failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn test_mimo_mtp_logits_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiMoMtpModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden, 0);
        assert!(logits.is_ok(), "compute_logits failed: {:?}", logits.err());
        assert_eq!(logits.unwrap().dims(), &[1, 1, cfg.vocab_size]);
    }
}
