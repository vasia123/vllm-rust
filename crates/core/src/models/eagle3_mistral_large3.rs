//! Eagle-3 speculative decoding draft model for Mistral Large 3 (DeepSeek V2 architecture).
//!
//! Unlike Eagle3Llama which concatenates embeddings and hidden states at layer 0's
//! QKV projection, this model applies a single `fc` projection
//! (`2×hidden_size → hidden_size`) before all DeepSeek V2 decoder layers.
//!
//! Forward: `fc(cat(embed(input_ids), hidden_states))` → DeepSeek layers → norm
//! Returns `(hidden_states, hidden_states)` — same value for both outputs.
//!
//! Key differences from Eagle3LlamaForCausalLM:
//! - Uses MLA attention (DeepSeek V2 layers), not standard multi-head attention
//! - fc applied once before all layers, not per-layer-0 concatenation
//! - No auxiliary hidden state combination (no 3×hs → hs fc)
//! - No draft vocabulary remapping or logit scaling
//! - Requires MLA KV cache (`KVCacheManager::new_mla()`)
//!
//! Reference: `reference/vllm/vllm/model_executor/models/mistral_large_3_eagle.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;

use super::deepseek::DeepSeekDecoderLayer;
use super::eagle3::Eagle3DraftModel;

// ─── Eagle3 Mistral Large 3 Model ──────────────────────────────────────────

/// Eagle-3 speculative decoding draft model for Mistral Large 3.
///
/// Based on DeepSeek V2 architecture with MLA attention. Generates draft tokens
/// by combining target model hidden states with token embeddings via an fc
/// projection, then running through standard DeepSeek decoder layers.
///
/// This model does NOT implement `ModelForward` because its forward pass
/// requires `hidden_states` from the target model — a different signature.
/// Integration with speculative decoding happens through a proposer wrapper.
pub struct Eagle3MistralLarge3ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    fc: Linear,
    layers: Vec<DeepSeekDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl Eagle3MistralLarge3ForCausalLM {
    /// Create a new Eagle3 Mistral Large 3 draft model.
    ///
    /// The config must include MLA parameters in `extra`:
    /// `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // fc: cat(embeds, hidden_states) → combined hidden states
        let fc = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb_m.pp("fc"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
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
            fc,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Forward pass through the Eagle3 Mistral Large 3 draft model.
    ///
    /// Combines embeddings with target hidden states via fc projection,
    /// then runs through DeepSeek V2 decoder layers with MLA attention.
    ///
    /// Returns `(hidden_states, hidden_states)` — same value for both outputs,
    /// matching the Eagle3 proposer interface that chains the second output.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
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

        // Combine embeddings and target hidden states: [batch, seq, 2*hs]
        let combined = Tensor::cat(&[&embeds, hidden_states], 2)?;
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

        let xs = self.norm.forward(&xs)?;
        Ok((xs.clone(), xs))
    }

    /// Compute logits from hidden states.
    pub fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Eagle3DraftModel for Eagle3MistralLarge3ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        Eagle3MistralLarge3ForCausalLM::forward(
            self,
            input_ids,
            hidden_states,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Eagle3MistralLarge3ForCausalLM::compute_logits(self, hidden_states)
    }

    fn combine_hidden_states(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Ok(hidden_states.clone())
    }

    fn use_aux_hidden_state(&self) -> bool {
        false
    }

    fn device(&self) -> &Device {
        Eagle3MistralLarge3ForCausalLM::device(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::mla_cache_config::MLACacheConfig;

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
            architectures: vec!["EagleMistralLarge3ForCausalLM".to_string()],
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

    fn create_mla_cache_manager(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
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

    // ─── Construction ──────────────────────────────────────────────────────

    #[test]
    fn eagle3_mistral_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());

        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn eagle3_mistral_construction_no_tie() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb);
        assert!(model.is_ok());
    }

    // ─── Forward ────────────────────────────────────────────────────────────

    #[test]
    fn eagle3_mistral_forward_prefill() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = BlockTable::new(4);

        let batch_size = 1;
        let seq_len = 4;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();
        let hidden_states =
            Tensor::zeros((batch_size, seq_len, cfg.hidden_size), DType::F32, &device).unwrap();

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let (hs, hs2) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(hs.dims(), &[batch_size, seq_len, cfg.hidden_size]);
        assert_eq!(hs2.dims(), &[batch_size, seq_len, cfg.hidden_size]);
    }

    #[test]
    fn eagle3_mistral_forward_single_token() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = BlockTable::new(4);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden_states = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).unwrap();

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .unwrap();
        let slot_mapping = vec![0];

        let (hs, _) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(hs.dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn eagle3_mistral_returns_same_tensor() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = BlockTable::new(4);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden_states = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).unwrap();

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .unwrap();
        let slot_mapping = vec![0];

        let (hs1, hs2) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        // Both outputs should have identical values
        let v1: Vec<f32> = hs1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = hs2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }

    // ─── Logits ─────────────────────────────────────────────────────────────

    #[test]
    fn eagle3_mistral_compute_logits() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn eagle3_mistral_compute_logits_multi_token() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();

        assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
    }

    // ─── Prefill + Decode Workflow ──────────────────────────────────────────

    #[test]
    fn eagle3_mistral_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = BlockTable::new(4);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        let hs_prefill = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).unwrap();

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..3).collect();

        let (out, _) = model
            .forward(
                &prompt,
                &hs_prefill,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 3, cfg.hidden_size]);
        block_table.advance(3);

        // Decode step at offset=3
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let next_hs = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).unwrap();

        let (out, _) = model
            .forward(
                &next_token,
                &next_hs,
                3,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 1, cfg.hidden_size]);
    }

    // ─── With MoE Layers ────────────────────────────────────────────────────

    #[test]
    fn eagle3_mistral_with_moe() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "n_routed_experts".into(),
            serde_json::Value::Number(4.into()),
        );
        cfg.extra.insert(
            "num_experts_per_tok".into(),
            serde_json::Value::Number(2.into()),
        );
        cfg.extra.insert(
            "moe_intermediate_size".into(),
            serde_json::Value::Number(128.into()),
        );

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = BlockTable::new(4);

        let input_ids = Tensor::zeros((1, 2), DType::U32, &device).unwrap();
        let hidden_states = Tensor::zeros((1, 2, cfg.hidden_size), DType::F32, &device).unwrap();

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 2)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..2).collect();

        let result = model.forward(
            &input_ids,
            &hidden_states,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(result.is_ok(), "forward with MoE: {:?}", result.err());
    }

    // ─── End-to-End: Forward + Logits ───────────────────────────────────────

    #[test]
    fn eagle3_mistral_forward_then_logits() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3MistralLarge3ForCausalLM::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = BlockTable::new(4);

        let input_ids = Tensor::zeros((1, 2), DType::U32, &device).unwrap();
        let hidden_states = Tensor::zeros((1, 2, cfg.hidden_size), DType::F32, &device).unwrap();

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 2)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..2).collect();

        let (hs, _) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        let logits = model.compute_logits(&hs).unwrap();
        assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
    }
}
