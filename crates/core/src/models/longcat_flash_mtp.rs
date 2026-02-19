//! LongCat-Flash Multi-Token Prediction (MTP) draft model.
//!
//! LongCat-Flash MTP uses the standard DeepSeek-style fusion (enorm + hnorm +
//! eh_proj + mtp_block) but with two special properties:
//! 1. The MTP start layer index is `num_hidden_layers * 2` (double offset).
//! 2. There is always exactly **1 MTP layer** regardless of config.
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
//! The single MTP layer is at absolute index `num_hidden_layers * 2` under
//! `model.mtp.layers`:
//!   - `model.mtp.layers.0.enorm`
//!   - `model.mtp.layers.0.hnorm`
//!   - `model.mtp.layers.0.eh_proj`
//!   - `model.mtp.layers.0.shared_head.norm`
//!   - `model.mtp.layers.0.shared_head.head`
//!   - `model.mtp.layers.0.mtp_block.*`  (DeepSeekDecoderLayer structure)
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/longcat_flash_mtp.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};

use super::deepseek::DeepSeekDecoderLayer;
use super::mtp_base::MtpDraftModel;

// ─── Layer ──────────────────────────────────────────────────────────────────

/// The single LongCat-Flash MTP prediction layer.
struct LongCatFlashMtpLayer {
    enorm: RmsNorm,
    hnorm: RmsNorm,
    eh_proj: Linear,
    shared_head_norm: RmsNorm,
    shared_head: Linear,
    mtp_block: DeepSeekDecoderLayer,
}

impl LongCatFlashMtpLayer {
    fn new(cfg: &ModelConfig, abs_layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let enorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("enorm"))?;
        let hnorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hnorm"))?;
        let eh_proj = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb.pp("eh_proj"))?;
        let vb_head = vb.pp("shared_head");
        let shared_head_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_head.pp("norm"))?;
        let shared_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_head.pp("head"))?;
        let mtp_block = DeepSeekDecoderLayer::new(cfg, abs_layer_idx, vb.pp("mtp_block"))?;
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

/// LongCat-Flash Multi-Token Prediction draft model.
///
/// Uses `DeepSeekDecoderLayer` but places the single MTP layer at
/// absolute index `num_hidden_layers * 2` (double the normal offset).
pub struct LongCatFlashMtpModel {
    embed_tokens: Embedding,
    layer: LongCatFlashMtpLayer,
    device: Device,
}

impl LongCatFlashMtpModel {
    /// Load the LongCat-Flash MTP model from a VarBuilder rooted at the model file.
    ///
    /// The single MTP layer is at `model.mtp.layers.0.*`.
    /// The absolute layer index passed to `DeepSeekDecoderLayer` is `num_hidden_layers * 2`
    /// to correctly trigger MoE routing.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // Double-offset: mtp_start = num_hidden_layers * 2
        let abs_layer_idx = cfg.num_hidden_layers * 2;

        // Weight path: model.mtp.layers.0.*
        let layer =
            LongCatFlashMtpLayer::new(cfg, abs_layer_idx, vb_m.pp("mtp").pp("layers").pp(0))?;

        Ok(Self {
            embed_tokens,
            layer,
            device: vb.device().clone(),
        })
    }
}

impl MtpDraftModel for LongCatFlashMtpModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        seqlen_offset: usize,
        _spec_step_idx: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Embed input tokens: [1, seq_len, hidden_size]
        let mut inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // BOS masking during prefill
        if seqlen_offset == 0 && seq_len > 1 {
            let zeros = Tensor::zeros(
                (1usize, 1usize, inputs_embeds.dim(2)?),
                inputs_embeds.dtype(),
                inputs_embeds.device(),
            )?;
            let rest = inputs_embeds.narrow(1, 1, seq_len - 1)?;
            inputs_embeds = Tensor::cat(&[&zeros, &rest], 1)?;
        }

        // Normalize and fuse
        let normed_embeds = self.layer.enorm.forward(&inputs_embeds)?;
        let normed_hs = self.layer.hnorm.forward(previous_hidden_states)?;
        let combined = Tensor::cat(&[&normed_embeds, &normed_hs], 2)?;
        let hidden = self.layer.eh_proj.forward(&combined)?;

        // Run through DeepSeek transformer block
        let hidden = if seq_len > 1 {
            let mask = causal_mask(seq_len, seqlen_offset, hidden.dtype(), hidden.device())?;
            self.layer.mtp_block.forward(
                &hidden,
                Some(&mask),
                seqlen_offset,
                kv_cache_mgr,
                0,
                block_table.block_ids(),
                slot_mapping,
            )?
        } else {
            self.layer.mtp_block.forward_decode(
                &hidden,
                seqlen_offset,
                kv_cache_mgr,
                0,
                block_table.block_ids(),
                slot_mapping,
            )?
        };

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
    use crate::kv_cache::MLACacheConfig;
    use candle_core::DType;
    use serde_json::json;

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("qk_nope_head_dim".to_string(), json!(16));
        extra.insert("qk_rope_head_dim".to_string(), json!(8));
        extra.insert("v_head_dim".to_string(), json!(16));
        extra.insert("kv_lora_rank".to_string(), json!(32));
        extra.insert("n_routed_experts".to_string(), json!(4));
        extra.insert("n_shared_experts".to_string(), json!(1));
        extra.insert("num_experts_per_tok".to_string(), json!(2));
        extra.insert("moe_intermediate_size".to_string(), json!(64));
        extra.insert("routed_scaling_factor".to_string(), json!(1.0));

        ModelConfig {
            architectures: vec!["LongCatFlashMTPModel".to_string()],
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 24, // qk_nope + qk_rope
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn make_mla_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32,
            8,
            16,
            16,
            cfg.num_attention_heads,
            4,
            64,
            1,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).unwrap()
    }

    #[test]
    fn test_longcat_flash_mtp_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LongCatFlashMtpModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "LongCatFlashMtpModel creation failed: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().num_mtp_layers(), 1);
    }

    #[test]
    fn test_longcat_flash_mtp_prefill_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LongCatFlashMtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_mla_cache(&cfg, &device);

        let seq_len = 4usize;
        let mut bt = BlockTable::new(4);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let target_hs =
            Tensor::zeros((1usize, seq_len, cfg.hidden_size), DType::F32, &device).unwrap();

        let result = model.forward(
            &input_ids,
            &target_hs,
            0,
            0,
            &mut kv_cache,
            &bt,
            &slot_mapping,
        );
        assert!(result.is_ok(), "prefill failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.hidden_size]);
    }

    #[test]
    fn test_longcat_flash_mtp_decode_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LongCatFlashMtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_mla_cache(&cfg, &device);

        let mut bt = BlockTable::new(4);

        // Prefill
        let seq_len = 3usize;
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);
        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let target_hs =
            Tensor::zeros((1usize, seq_len, cfg.hidden_size), DType::F32, &device).unwrap();
        model
            .forward(
                &input_ids,
                &target_hs,
                0,
                0,
                &mut kv_cache,
                &bt,
                &slot_mapping,
            )
            .unwrap();
        bt.advance(seq_len);

        // Decode
        kv_cache.allocate_for_request(&mut bt, 1).unwrap();
        let slot_mapping = bt.slot_mapping(seq_len, 1);
        let tok = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let hs = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();

        let result = model.forward(&tok, &hs, seq_len, 0, &mut kv_cache, &bt, &slot_mapping);
        assert!(result.is_ok(), "decode failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn test_longcat_flash_mtp_logits_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LongCatFlashMtpModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden, 0);
        assert!(logits.is_ok(), "compute_logits failed: {:?}", logits.err());
        assert_eq!(logits.unwrap().dims(), &[1, 1, cfg.vocab_size]);
    }
}
