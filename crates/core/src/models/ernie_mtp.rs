//! ERNIE Multi-Token Prediction (MTP) draft model.
//!
//! ERNIE MTP uses the same DeepSeek-style fusion architecture (enorm + hnorm +
//! eh_proj + mtp_block + shared_head) but is **single-token-only**: it only
//! supports speculative step 0 (one draft token per target step).
//!
//! # Architecture
//!
//! ```text
//! embed(input_ids) → enorm → │
//!                            ├─ cat ─→ eh_proj ─→ mtp_block ─→ hidden_states
//! target_hs       → hnorm → │
//!
//! hidden_states → shared_head.norm → lm_head → logits
//! ```
//!
//! # Weight paths
//!
//! Weights live under `model.mtp.*` (not under `model.layers.*`):
//!   - `model.mtp.enorm`
//!   - `model.mtp.hnorm`
//!   - `model.mtp.eh_proj`
//!   - `model.mtp.shared_head.norm`
//!   - `model.mtp.shared_head.head`
//!   - `model.mtp.mtp_block.*`   (LlamaDecoderLayer structure)
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/ernie_mtp.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};
use crate::models::tp_layers::TpContext;

use super::llama::LlamaDecoderLayer;
use super::mtp_base::MtpDraftModel;

// ─── Layer ──────────────────────────────────────────────────────────────────

/// One ERNIE MTP prediction layer.
struct ErnieMtpLayer {
    enorm: RmsNorm,
    hnorm: RmsNorm,
    eh_proj: Linear,
    shared_head_norm: RmsNorm,
    shared_head: Linear,
    mtp_block: LlamaDecoderLayer,
}

impl ErnieMtpLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let pg = LocalProcessGroup::new();
        let enorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("enorm"))?;
        let hnorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hnorm"))?;
        let eh_proj = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb.pp("eh_proj"))?;
        let vb_head = vb.pp("shared_head");
        let shared_head_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_head.pp("norm"))?;
        let shared_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_head.pp("head"))?;
        let mtp_block = LlamaDecoderLayer::new_with_tp(cfg, vb.pp("mtp_block"), &pg)?;
        Ok(Self {
            enorm,
            hnorm,
            eh_proj,
            shared_head_norm,
            shared_head,
            mtp_block,
        })
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// ERNIE Multi-Token Prediction draft model.
///
/// Single-token-only: only `spec_step_idx == 0` is supported. Use this model
/// with `num_speculative_tokens = 1`.
pub struct ErnieMtpModel {
    embed_tokens: Embedding,
    layer: ErnieMtpLayer,
    tp_ctx: TpContext,
    device: Device,
}

impl ErnieMtpModel {
    /// Load the ERNIE MTP model from a VarBuilder rooted at the model file.
    ///
    /// Weights are at `model.mtp.*`.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let layer = ErnieMtpLayer::new(cfg, vb_m.pp("mtp"))?;
        Ok(Self {
            embed_tokens,
            layer,
            tp_ctx: TpContext::single_gpu(),
            device: vb.device().clone(),
        })
    }
}

impl MtpDraftModel for ErnieMtpModel {
    /// Run one ERNIE MTP forward pass.
    ///
    /// ERNIE MTP is single-token-only: returns an error if `spec_step_idx > 0`.
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
                "ErnieMtpModel is single-token-only: spec_step_idx must be 0, got {}",
                spec_step_idx
            );
        }

        let (_, seq_len) = input_ids.dims2()?;

        // Embed input tokens: [1, seq_len, hidden_size]
        let mut inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // BOS masking: zero out position 0's embedding during prefill.
        if seqlen_offset == 0 && seq_len > 1 {
            let zeros = Tensor::zeros(
                (1usize, 1usize, inputs_embeds.dim(2)?),
                inputs_embeds.dtype(),
                inputs_embeds.device(),
            )?;
            let rest = inputs_embeds.narrow(1, 1, seq_len - 1)?;
            inputs_embeds = Tensor::cat(&[&zeros, &rest], 1)?;
        }

        // Normalize and fuse: [1, seq_len, hidden_size]
        let normed_embeds = self.layer.enorm.forward(&inputs_embeds)?;
        let normed_hs = self.layer.hnorm.forward(previous_hidden_states)?;
        let combined = Tensor::cat(&[&normed_embeds, &normed_hs], 2)?;
        let hidden = self.layer.eh_proj.forward(&combined)?;

        // Run through Llama transformer block
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

        Ok(hidden)
    }

    fn compute_logits(&self, hidden_states: &Tensor, _spec_step_idx: usize) -> Result<Tensor> {
        let normed = self.layer.shared_head_norm.forward(hidden_states)?;
        self.layer.shared_head.forward(&normed)
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
            architectures: vec!["ErnieMTPModel".to_string()],
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
            attention_bias: None,
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
    fn test_ernie_mtp_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ErnieMtpModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ErnieMtpModel creation failed: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().num_mtp_layers(), 1);
    }

    #[test]
    fn test_ernie_mtp_single_token_constraint() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ErnieMtpModel::new(&cfg, vb).unwrap();
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
    fn test_ernie_mtp_decode_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ErnieMtpModel::new(&cfg, vb).unwrap();
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
    fn test_ernie_mtp_logits_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ErnieMtpModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden, 0);
        assert!(logits.is_ok(), "compute_logits failed: {:?}", logits.err());
        assert_eq!(logits.unwrap().dims(), &[1, 1, cfg.vocab_size]);
    }
}
