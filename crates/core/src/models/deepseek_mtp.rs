//! DeepSeek Multi-Token Prediction (MTP) draft model.
//!
//! Implements the `DeepSeekMultiTokenPredictor` architecture for speculative
//! decoding. Each MTP layer fuses the draft token embedding with the target
//! model's hidden states, then runs a lightweight DeepSeek transformer block
//! to produce hidden states for logit computation.
//!
//! # Architecture (per MTP layer)
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
//! MTP layers are stored at absolute layer indices `[N, N+num_mtp_layers)` in
//! the checkpoint, where `N = config.num_hidden_layers`. Within each layer:
//!   - `model.layers.{N+i}.enorm`
//!   - `model.layers.{N+i}.hnorm`
//!   - `model.layers.{N+i}.eh_proj`
//!   - `model.layers.{N+i}.shared_head.norm`
//!   - `model.layers.{N+i}.shared_head.head`
//!   - `model.layers.{N+i}.mtp_block.*`
//!
//! The `mtp_block` uses the same `DeepSeekDecoderLayer` as the main model,
//! with MoE activated for `abs_layer_idx > 0` (always true for MTP layers).
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/deepseek_mtp.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};

use super::deepseek::DeepSeekDecoderLayer;
use super::mtp_base::MtpDraftModel;

// ─── Layer ──────────────────────────────────────────────────────────────────

/// One MTP prediction layer.
///
/// Corresponds to `DeepSeekMultiTokenPredictorLayer` in Python.
struct DeepSeekMtpLayer {
    /// RMSNorm for token embeddings (enorm in Python).
    enorm: RmsNorm,
    /// RMSNorm for target hidden states (hnorm in Python).
    hnorm: RmsNorm,
    /// Fusion projection: cat(enorm_out, hnorm_out) [2H] → [H].
    eh_proj: Linear,
    /// shared_head: RMSNorm applied before logit projection.
    shared_head_norm: RmsNorm,
    /// shared_head LM head (weights shared with target model's lm_head in Python).
    shared_head: Linear,
    /// The transformer block (DeepSeek decoder layer with MoE).
    mtp_block: DeepSeekDecoderLayer,
}

impl DeepSeekMtpLayer {
    /// Load one MTP layer.
    ///
    /// `abs_layer_idx` is the absolute index within the full model (e.g. 61
    /// for DeepSeek-V3), used to select MoE vs dense in `DeepSeekDecoderLayer`.
    /// `vb` must already point to `model.layers.{abs_layer_idx}`.
    fn new(cfg: &ModelConfig, abs_layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let enorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("enorm"))?;
        let hnorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hnorm"))?;
        let eh_proj = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb.pp("eh_proj"))?;

        let vb_head = vb.pp("shared_head");
        let shared_head_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_head.pp("norm"))?;
        let shared_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_head.pp("head"))?;

        // The mtp_block is a DeepSeekDecoderLayer. abs_layer_idx is > 0 for all
        // MTP layers (they start at num_hidden_layers), so MoE is enabled.
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

/// DeepSeek Multi-Token Prediction draft model.
///
/// Wraps one or more [`DeepSeekMtpLayer`]s. Multiple layers cycle via
/// `spec_step_idx % num_mtp_layers`. For DeepSeek-V3, `num_nextn_predict_layers = 1`
/// so there is always exactly one layer and no cycling occurs.
///
/// The `embed_tokens` and LM head are loaded from the MTP model's own weight
/// paths. In Python vLLM these are shared with the target model, but in Rust
/// each model carries its own weights (loaded from the same checkpoint data).
pub struct DeepSeekMtpModel {
    embed_tokens: Embedding,
    layers: Vec<DeepSeekMtpLayer>,
    device: Device,
}

impl DeepSeekMtpModel {
    /// Load the MTP model from a VarBuilder rooted at the model file.
    ///
    /// Reads `num_nextn_predict_layers` from `cfg.extra` to determine how many
    /// MTP layers to load. Layers are at absolute indices
    /// `[num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)`.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_mtp_layers = cfg
            .extra
            .get("num_nextn_predict_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let mtp_start = cfg.num_hidden_layers;

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let vb_layers = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(num_mtp_layers);
        for i in 0..num_mtp_layers {
            let abs_idx = mtp_start + i;
            layers.push(DeepSeekMtpLayer::new(cfg, abs_idx, vb_layers.pp(abs_idx))?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            device: vb.device().clone(),
        })
    }
}

impl MtpDraftModel for DeepSeekMtpModel {
    /// Run one MTP forward pass.
    ///
    /// During prefill (`seqlen_offset == 0` and `seq_len > 1`):
    ///   - Processes the full `prompt + last_token` sequence to warm up the KV cache.
    ///   - Position 0 embeddings are zeroed (BOS masking, per Python reference).
    ///   - `spec_step_idx = 0` → uses `layers[0]`.
    ///
    /// During decode (`seq_len == 1`):
    ///   - Processes a single draft token.
    ///   - `spec_step_idx = k` → uses `layers[k % num_mtp_layers]`.
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

        // Embed input tokens: [1, seq_len, hidden_size]
        let mut inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // BOS masking: zero out position 0's embedding (Python: positions.unsqueeze(-1) == 0).
        // Only applies during prefill where seqlen_offset == 0 and seq_len > 1.
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
        let normed_embeds = layer.enorm.forward(&inputs_embeds)?;
        let normed_hs = layer.hnorm.forward(previous_hidden_states)?;

        // Concat along hidden dim: [1, seq_len, 2*hidden_size] → [1, seq_len, hidden_size]
        let combined = Tensor::cat(&[&normed_embeds, &normed_hs], 2)?;
        let hidden = layer.eh_proj.forward(&combined)?;

        // Run through DeepSeek transformer block
        let hidden = if seq_len > 1 {
            // Prefill: causal attention mask
            let mask = causal_mask(seq_len, seqlen_offset, hidden.dtype(), hidden.device())?;
            layer.mtp_block.forward(
                &hidden,
                Some(&mask),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table.block_ids(),
                slot_mapping,
            )?
        } else {
            // Decode: single-token step, no causal mask
            layer.mtp_block.forward_decode(
                &hidden,
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table.block_ids(),
                slot_mapping,
            )?
        };

        Ok(hidden)
    }

    /// Compute logits via the layer's shared_head (norm + LM head projection).
    fn compute_logits(&self, hidden_states: &Tensor, spec_step_idx: usize) -> Result<Tensor> {
        let layer_idx = spec_step_idx % self.layers.len();
        let layer = &self.layers[layer_idx];
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
    use crate::kv_cache::MLACacheConfig;
    use candle_core::DType;
    use serde_json::json;

    fn make_cfg(num_mtp_layers: usize) -> ModelConfig {
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
        extra.insert(
            "num_nextn_predict_layers".to_string(),
            json!(num_mtp_layers),
        );

        ModelConfig {
            architectures: vec!["DeepSeekMTPForCausalLM".to_string()],
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

    fn make_mla_cache(cfg: &ModelConfig, num_layers: usize, device: &Device) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32, // kv_lora_rank
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            cfg.num_attention_heads,
            4,  // block_size
            16, // num_blocks
            num_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).unwrap()
    }

    #[test]
    fn test_deepseek_mtp_creation_single_layer() {
        let device = Device::Cpu;
        let cfg = make_cfg(1);
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekMtpModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "DeepSeekMtpModel creation failed: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().num_mtp_layers(), 1);
    }

    #[test]
    fn test_deepseek_mtp_creation_two_layers() {
        let device = Device::Cpu;
        let cfg = make_cfg(2);
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekMtpModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "DeepSeekMtpModel creation (2 layers) failed: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().num_mtp_layers(), 2);
    }

    #[test]
    fn test_deepseek_mtp_prefill() {
        let device = Device::Cpu;
        let cfg = make_cfg(1);
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekMtpModel::new(&cfg, vb).unwrap();

        // num_mtp_layers MLA cache layers
        let mut kv_cache = make_mla_cache(&cfg, 1, &device);

        let batch = 1usize;
        let seq_len = 5usize;
        let input_ids = Tensor::zeros((batch, seq_len), DType::U32, &device).unwrap();
        let target_hs =
            Tensor::zeros((batch, seq_len, cfg.hidden_size), DType::F32, &device).unwrap();

        let block_table = crate::kv_cache::BlockTable::new(4);
        kv_cache
            .allocate_for_request(&mut { block_table.clone() }, seq_len)
            .unwrap();
        let mut bt = crate::kv_cache::BlockTable::new(4);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let result = model.forward(
            &input_ids,
            &target_hs,
            0,
            0,
            &mut kv_cache,
            &bt,
            &slot_mapping,
        );
        assert!(result.is_ok(), "MTP prefill failed: {:?}", result.err());
        let out = result.unwrap();
        assert_eq!(out.dims(), &[batch, seq_len, cfg.hidden_size]);
    }

    #[test]
    fn test_deepseek_mtp_decode() {
        let device = Device::Cpu;
        let cfg = make_cfg(1);
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekMtpModel::new(&cfg, vb).unwrap();
        let mut kv_cache = make_mla_cache(&cfg, 1, &device);

        let mut bt = crate::kv_cache::BlockTable::new(4);

        // Prefill first to fill KV cache (seqlen=3)
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

        // Now decode single token at position 3
        kv_cache.allocate_for_request(&mut bt, 1).unwrap();
        let slot_mapping = bt.slot_mapping(seq_len, 1);
        let tok = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let hs_last =
            Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();

        let result = model.forward(
            &tok,
            &hs_last,
            seq_len,
            0,
            &mut kv_cache,
            &bt,
            &slot_mapping,
        );
        assert!(result.is_ok(), "MTP decode failed: {:?}", result.err());
        let out = result.unwrap();
        assert_eq!(out.dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn test_deepseek_mtp_compute_logits() {
        let device = Device::Cpu;
        let cfg = make_cfg(1);
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekMtpModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden, 0);
        assert!(logits.is_ok(), "compute_logits failed: {:?}", logits.err());
        assert_eq!(logits.unwrap().dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_deepseek_mtp_layer_cycling() {
        // Verify that spec_step_idx cycles through layers correctly.
        let device = Device::Cpu;
        let cfg = make_cfg(2); // 2 MTP layers
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekMtpModel::new(&cfg, vb).unwrap();
        assert_eq!(model.num_mtp_layers(), 2);

        let hidden = Tensor::zeros((1usize, 1usize, cfg.hidden_size), DType::F32, &device).unwrap();
        // spec_step_idx 0 → layer 0, spec_step_idx 2 → layer 0 (cycling)
        let l0 = model.compute_logits(&hidden, 0);
        let l2 = model.compute_logits(&hidden, 2);
        assert!(l0.is_ok() && l2.is_ok());
        // Both produce same shape (both use layer 0)
        assert_eq!(l0.unwrap().shape(), l2.unwrap().shape());
    }
}
