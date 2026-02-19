//! Step3.5 Multi-Token Prediction (MTP) draft model.
//!
//! Step3.5 MTP uses the standard DeepSeek-style fusion (enorm + hnorm +
//! eh_proj + mtp_block + shared_head) with Step3p5DecoderLayer blocks.
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
//! MTP layers start at absolute index `num_hidden_layers` under `model.layers`:
//!   - `model.layers.{mtp_start+i}.enorm`
//!   - `model.layers.{mtp_start+i}.hnorm`
//!   - `model.layers.{mtp_start+i}.eh_proj`
//!   - `model.layers.{mtp_start+i}.shared_head.norm`
//!   - `model.layers.{mtp_start+i}.shared_head.head`
//!   - `model.layers.{mtp_start+i}.mtp_block.*`  (Step3p5DecoderLayer structure)
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/step3p5_mtp.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};
use crate::models::tp_layers::TpContext;

use super::mtp_base::MtpDraftModel;
use super::step3p5::Step3p5DecoderLayer;

// ─── Layer ──────────────────────────────────────────────────────────────────

/// One Step3.5 MTP prediction layer.
struct Step3p5MtpLayer {
    enorm: RmsNorm,
    hnorm: RmsNorm,
    eh_proj: Linear,
    shared_head_norm: RmsNorm,
    shared_head: Linear,
    mtp_block: Step3p5DecoderLayer,
}

impl Step3p5MtpLayer {
    fn new(cfg: &ModelConfig, abs_layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let enorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("enorm"))?;
        let hnorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hnorm"))?;
        let eh_proj = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb.pp("eh_proj"))?;
        let vb_head = vb.pp("shared_head");
        let shared_head_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_head.pp("norm"))?;
        let shared_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_head.pp("head"))?;
        let mtp_block = Step3p5DecoderLayer::new_for_mtp(cfg, abs_layer_idx, vb.pp("mtp_block"))?;
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

/// Step3.5 Multi-Token Prediction draft model.
///
/// Supports multiple MTP layers; `spec_step_idx` cycles through them.
pub struct Step3p5MtpModel {
    embed_tokens: Embedding,
    layers: Vec<Step3p5MtpLayer>,
    tp_ctx: TpContext,
    device: Device,
}

impl Step3p5MtpModel {
    /// Load the Step3.5 MTP model from a VarBuilder rooted at the model file.
    ///
    /// Layer count from `num_nextn_predict_layers` in `cfg.extra` (defaults to 1).
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let num_mtp = cfg
            .extra
            .get("num_nextn_predict_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let mtp_start = cfg.num_hidden_layers;
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(num_mtp);
        for i in 0..num_mtp {
            let abs_idx = mtp_start + i;
            layers.push(Step3p5MtpLayer::new(cfg, abs_idx, vb_l.pp(abs_idx))?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            tp_ctx: TpContext::single_gpu(),
            device: vb.device().clone(),
        })
    }
}

impl MtpDraftModel for Step3p5MtpModel {
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
        let (_, seq_len) = input_ids.dims2()?;
        let layer_idx = spec_step_idx % self.layers.len();
        let layer = &self.layers[layer_idx];

        // Step3.5 MTP does not apply BOS masking (unlike DeepSeek/GLM4 variants)
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;

        let normed_embeds = layer.enorm.forward(&inputs_embeds)?;
        let normed_hs = layer.hnorm.forward(previous_hidden_states)?;
        let combined = Tensor::cat(&[&normed_embeds, &normed_hs], 2)?;
        let hidden = layer.eh_proj.forward(&combined)?;

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

        let hidden = layer.mtp_block.forward(
            &hidden,
            mask.as_ref(),
            seqlen_offset,
            kv_cache_mgr,
            layer_idx,
            block_table,
            slot_mapping,
            &self.tp_ctx,
        )?;

        Ok(hidden)
    }

    fn compute_logits(&self, hidden_states: &Tensor, spec_step_idx: usize) -> Result<Tensor> {
        let layer = &self.layers[spec_step_idx % self.layers.len()];
        let normed = layer.shared_head_norm.forward(hidden_states)?;
        layer.shared_head.forward(&normed)
    }

    fn num_mtp_layers(&self) -> usize {
        self.layers.len()
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
    use serde_json::json;

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_nextn_predict_layers".to_string(), json!(1));
        // All target layers dense (no MoE needed for tests)
        extra.insert("moe_layers".to_string(), json!([]));

        ModelConfig {
            architectures: vec!["Step3p5MTP".to_string()],
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
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, num_layers: usize, device: &Device) -> KVCacheManager {
        let cache_config = CacheConfig {
            block_size: 4,
            num_blocks: 64,
            num_layers,
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
    fn test_step3p5_mtp_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5MtpModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Step3p5MtpModel creation failed: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().num_mtp_layers(), 1);
    }

    #[test]
    fn test_step3p5_mtp_prefill_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5MtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, 1, &device);

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
    fn test_step3p5_mtp_decode_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5MtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, 1, &device);

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
    fn test_step3p5_mtp_logits_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Step3p5MtpModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden, 0);
        assert!(logits.is_ok(), "compute_logits failed: {:?}", logits.err());
        assert_eq!(logits.unwrap().dims(), &[1, 1, cfg.vocab_size]);
    }
}
