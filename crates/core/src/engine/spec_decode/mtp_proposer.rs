//! Multi-Token Prediction (MTP) DraftProposer implementation.
//!
//! Wraps any [`MtpDraftModel`] with full lifecycle management. Unlike Eagle,
//! MTP uses FIXED target hidden states for ALL draft steps — the proposer
//! does NOT chain its own output as input to the next step.
//!
//! # Draft token generation flow
//!
//! 1. Target model produces hidden states → engine calls `set_target_hidden_states`
//! 2. On first call (`needs_prefill = true`):
//!    a. Run full prefill through MTP to warm up its KV cache.
//!    b. Extract the last target hidden state for decode steps.
//!    c. Generate K draft tokens, each using the FIXED last target hidden state.
//! 3. On subsequent calls:
//!    a. `target_hidden_states` already has shape [1, 1, H].
//!    b. Generate K draft tokens using the FIXED target hidden state.
//!
//! # MTP vs Eagle
//!
//! Eagle chains hidden states:
//!   `h_{k+1} = eagle(draft_k, h_k)` where h_0 = postnorm(target)
//!
//! MTP uses fixed hidden states:
//!   `h_k = mtp(draft_k, target_hs)` for ALL k, with the SAME target_hs
//!
//! # KV cache
//!
//! The MTP transformer block (`mtp_block`) maintains its own KV cache,
//! separate from the target model. This cache must be created with
//! [`KVCacheManager::new_mla`] for DeepSeek-based MTP models.
//!
//! Reference: `reference/vllm/vllm/v1/spec_decode/eagle.py`

use std::collections::HashMap;

use candle_core::Tensor;
use tracing::warn;

use crate::engine::types::EngineError;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::models::MtpDraftModel;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

use super::sample_speculative;
use super::DraftProposer;

// ─── Per-request state ───────────────────────────────────────────────────────

/// Per-request MTP draft state.
struct MtpRequestState {
    block_table: BlockTable,
    /// Position of the next token to write in the MTP KV cache.
    seqlen_offset: usize,
    /// Prompt tokens stored for the initial prefill.
    prompt_tokens: Vec<u32>,
    /// Target model hidden states for the current proposal round.
    ///
    /// Shape on prefill call: `[1, prompt_len + 1, hidden_size]` (full sequence).
    /// Shape on decode calls: `[1, 1, hidden_size]` (last token only).
    target_hidden_states: Option<Tensor>,
    /// Whether the MTP KV cache needs an initial prefill.
    needs_prefill: bool,
}

// ─── MtpProposer ─────────────────────────────────────────────────────────────

/// [`DraftProposer`] backed by a Multi-Token Prediction model.
///
/// Works with any model implementing [`MtpDraftModel`], including:
/// - [`DeepSeekMtpModel`](crate::models::DeepSeekMtpModel): DeepSeek-V3 MTP
///
/// The engine must call [`set_target_hidden_states`] before each
/// `propose_for_request` call to provide the target model's context.
pub struct MtpProposer {
    model: Box<dyn MtpDraftModel>,
    kv_cache: KVCacheManager,
    num_speculative_tokens: usize,
    requests: HashMap<RequestId, MtpRequestState>,
}

impl MtpProposer {
    pub fn new(
        model: Box<dyn MtpDraftModel>,
        kv_cache: KVCacheManager,
        num_speculative_tokens: usize,
    ) -> Self {
        Self {
            model,
            kv_cache,
            num_speculative_tokens,
            requests: HashMap::new(),
        }
    }
}

impl DraftProposer for MtpProposer {
    fn init_request(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<(), EngineError> {
        let block_table = BlockTable::new(self.kv_cache.block_size());
        self.requests.insert(
            request_id,
            MtpRequestState {
                block_table,
                seqlen_offset: 0,
                prompt_tokens: prompt_tokens.to_vec(),
                target_hidden_states: None,
                needs_prefill: true,
            },
        );
        Ok(())
    }

    fn set_target_hidden_states(
        &mut self,
        request_id: RequestId,
        hidden_states: Tensor,
    ) -> Result<(), EngineError> {
        let req = self.requests.get_mut(&request_id).ok_or_else(|| {
            EngineError::Model(format!(
                "MTP: set_target_hidden_states for unknown request {request_id}"
            ))
        })?;
        req.target_hidden_states = Some(hidden_states);
        Ok(())
    }

    fn propose_for_request(
        &mut self,
        request_id: RequestId,
        last_token: u32,
        state: &mut SequenceState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<Vec<u32>, EngineError> {
        let req = self.requests.get_mut(&request_id).ok_or_else(|| {
            EngineError::Model(format!(
                "MTP: propose_for_request for unknown request {request_id}"
            ))
        })?;

        let target_hs = match req.target_hidden_states.take() {
            Some(hs) => hs,
            None => {
                warn!("MTP: no target hidden states for request {request_id}");
                return Ok(Vec::new());
            }
        };

        let device = self.model.device().clone();
        let k = self.num_speculative_tokens;
        let mut draft_tokens = Vec::with_capacity(k);

        if req.needs_prefill {
            // ── Prefill path ─────────────────────────────────────────────────
            // Run the full prompt + last_token through the MTP model to warm up
            // its KV cache. target_hs has shape [1, prompt_len+1, hidden_size].
            let all_tokens: Vec<u32> = req
                .prompt_tokens
                .iter()
                .copied()
                .chain(std::iter::once(last_token))
                .collect();

            let seq_len = all_tokens.len();
            self.kv_cache
                .allocate_for_request(&mut req.block_table, seq_len)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            let slot_mapping = req.block_table.slot_mapping(0, seq_len);

            let input_tensor = Tensor::from_vec(all_tokens, (1, seq_len), &device)
                .map_err(|e| EngineError::Model(e.to_string()))?;

            // Prefill the MTP KV cache. Output is discarded — we don't chain
            // MTP's own hidden states; all K decode steps use the fixed target_hs.
            self.model
                .forward(
                    &input_tensor,
                    &target_hs,
                    0,
                    0,
                    &mut self.kv_cache,
                    &req.block_table,
                    &slot_mapping,
                )
                .map_err(|e| EngineError::Model(e.to_string()))?;

            req.block_table.advance(seq_len);
            req.seqlen_offset = seq_len;
            req.needs_prefill = false;

            // Extract the last target hidden state for decode steps:
            // [1, seq_len, H] → [1, 1, H]
            let hs_decode = target_hs
                .narrow(1, seq_len - 1, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;

            // Allocate all K decode positions upfront and advance immediately so
            // subsequent slot_mapping() calls see the correct block assignments.
            self.kv_cache
                .allocate_for_request(&mut req.block_table, k)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            req.block_table.advance(k);

            // Generate K draft tokens using the FIXED last target hidden state
            for step in 0..k {
                let token_to_use = if step == 0 {
                    last_token
                } else {
                    draft_tokens[step - 1]
                };

                let slot_mapping = req.block_table.slot_mapping(req.seqlen_offset + step, 1);

                let token_tensor = Tensor::new(&[[token_to_use]], &device)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                // MTP forward: uses FIXED hs_decode every step
                let hidden = self
                    .model
                    .forward(
                        &token_tensor,
                        &hs_decode,
                        req.seqlen_offset + step,
                        step,
                        &mut self.kv_cache,
                        &req.block_table,
                        &slot_mapping,
                    )
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let logits = self
                    .model
                    .compute_logits(&hidden, step)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let token = sample_speculative(&logits, state, tokenizer)
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                draft_tokens.push(token);
            }

            req.seqlen_offset += k;
        } else {
            // ── Decode path ───────────────────────────────────────────────────
            // target_hs has shape [1, 1, H] — the last token's hidden state from
            // the target model. This is the FIXED context for all K draft steps.

            // Allocate all K positions upfront and advance so slot_mapping is correct.
            self.kv_cache
                .allocate_for_request(&mut req.block_table, k)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            req.block_table.advance(k);

            for step in 0..k {
                let token_to_use = if step == 0 {
                    last_token
                } else {
                    draft_tokens[step - 1]
                };

                let slot_mapping = req.block_table.slot_mapping(req.seqlen_offset + step, 1);

                let token_tensor = Tensor::new(&[[token_to_use]], &device)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                // MTP forward: uses FIXED target_hs every step
                let hidden = self
                    .model
                    .forward(
                        &token_tensor,
                        &target_hs,
                        req.seqlen_offset + step,
                        step,
                        &mut self.kv_cache,
                        &req.block_table,
                        &slot_mapping,
                    )
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let logits = self
                    .model
                    .compute_logits(&hidden, step)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let token = sample_speculative(&logits, state, tokenizer)
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                draft_tokens.push(token);
            }

            req.seqlen_offset += k;
        }

        Ok(draft_tokens)
    }

    fn on_tokens_verified(
        &mut self,
        request_id: RequestId,
        num_accepted: usize,
        original_offset: usize,
    ) -> Result<(), EngineError> {
        let req = self.requests.get_mut(&request_id).ok_or_else(|| {
            EngineError::Model(format!(
                "MTP: on_tokens_verified for unknown request {request_id}"
            ))
        })?;

        let new_total = original_offset + num_accepted;
        let freed = req.block_table.trim_to(new_total);
        if !freed.is_empty() {
            self.kv_cache
                .free_blocks(&freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        req.seqlen_offset = new_total;
        Ok(())
    }

    fn finish_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(mut rs) = self.requests.remove(&request_id) {
            let freed = rs.block_table.release();
            if !freed.is_empty() {
                if let Err(e) = self.kv_cache.free_blocks(&freed) {
                    warn!(
                        error = %e,
                        request_id = request_id,
                        "Failed to free MTP cache blocks on request completion"
                    );
                }
            }
        }
        Ok(())
    }

    fn preempt_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(mut rs) = self.requests.remove(&request_id) {
            let freed = rs.block_table.release();
            if !freed.is_empty() {
                if let Err(e) = self.kv_cache.free_blocks(&freed) {
                    warn!(
                        error = %e,
                        request_id = request_id,
                        "Failed to free MTP cache blocks on preemption"
                    );
                }
            }
        }
        Ok(())
    }

    fn num_speculative_tokens(&self) -> usize {
        self.num_speculative_tokens
    }

    fn name(&self) -> &str {
        "mtp"
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::MLACacheConfig;
    use crate::models::{DeepSeekMtpModel, MtpDraftModel};
    use crate::tokenizer::TokenizerWrapper;
    use candle_core::{DType, Device};
    use serde_json::json;

    fn make_mtp_cfg() -> crate::config::ModelConfig {
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
        extra.insert("num_nextn_predict_layers".to_string(), json!(1u64));

        crate::config::ModelConfig {
            architectures: vec!["DeepSeekMTPForCausalLM".to_string()],
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 24,
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

    fn make_kv_cache(
        cfg: &crate::config::ModelConfig,
        num_layers: usize,
        device: &Device,
    ) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32,
            8,
            16,
            16,
            cfg.num_attention_heads,
            4,
            64,
            num_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).unwrap()
    }

    fn make_proposer(device: &Device) -> MtpProposer {
        let cfg = make_mtp_cfg();
        let vb = VarBuilder::zeros(DType::F32, device);
        let model = DeepSeekMtpModel::new(&cfg, vb).unwrap();
        let kv_cache = make_kv_cache(&cfg, model.num_mtp_layers(), device);
        MtpProposer::new(Box::new(model), kv_cache, 3)
    }

    fn make_state() -> SequenceState {
        // request_id=0, prompt_tokens=[], max_new_tokens=100, eos=2, block_size=4, arrival_order=0
        SequenceState::new(0, vec![], 100, 2, 4, 0)
    }

    fn make_tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::for_testing(1000)
    }

    use candle_nn::VarBuilder;

    #[test]
    fn test_mtp_init_propose_finish_lifecycle() {
        let device = Device::Cpu;
        let mut proposer = make_proposer(&device);
        let tokenizer = make_tokenizer();

        let req_id: RequestId = 1;
        let prompt = vec![10u32, 20, 30];

        proposer.init_request(req_id, &prompt).unwrap();

        // Provide target hidden states: [1, prompt_len+1, H] for prefill
        let hs = Tensor::zeros((1usize, prompt.len() + 1, 64usize), DType::F32, &device).unwrap();
        proposer.set_target_hidden_states(req_id, hs).unwrap();

        let mut state = make_state();
        let result = proposer.propose_for_request(req_id, 40, &mut state, &tokenizer);
        assert!(result.is_ok(), "First propose failed: {:?}", result.err());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 3, "Expected 3 draft tokens");

        // Verify accepted 2 out of 3
        proposer
            .on_tokens_verified(req_id, 2, prompt.len() + 1)
            .unwrap();

        // Second decode round: provide [1, 1, H] hidden state
        let hs2 = Tensor::zeros((1usize, 1usize, 64usize), DType::F32, &device).unwrap();
        proposer.set_target_hidden_states(req_id, hs2).unwrap();
        let result2 = proposer.propose_for_request(req_id, 50, &mut state, &tokenizer);
        assert!(
            result2.is_ok(),
            "Second propose failed: {:?}",
            result2.err()
        );
        assert_eq!(result2.unwrap().len(), 3);

        proposer.finish_request(req_id).unwrap();
        // Second finish is safe (idempotent)
        proposer.finish_request(req_id).unwrap();
    }

    #[test]
    fn test_mtp_preempt() {
        let device = Device::Cpu;
        let mut proposer = make_proposer(&device);
        let tokenizer = make_tokenizer();

        let req_id: RequestId = 2;
        proposer.init_request(req_id, &[1u32, 2]).unwrap();

        let hs = Tensor::zeros((1usize, 3usize, 64usize), DType::F32, &device).unwrap();
        proposer.set_target_hidden_states(req_id, hs).unwrap();

        let mut state = make_state();
        proposer
            .propose_for_request(req_id, 3, &mut state, &tokenizer)
            .unwrap();

        // Preempt should free all blocks without panicking
        proposer.preempt_request(req_id).unwrap();

        // After preemption the request is gone — proposing again should return empty
        let hs2 = Tensor::zeros((1usize, 1usize, 64usize), DType::F32, &device).unwrap();
        let _ = proposer.set_target_hidden_states(req_id, hs2);
        let result = proposer.propose_for_request(req_id, 10, &mut state, &tokenizer);
        // Should either fail gracefully or return empty
        match result {
            Ok(tokens) => assert!(tokens.is_empty()),
            Err(_) => {} // acceptable — request was preempted
        }
    }

    #[test]
    fn test_mtp_missing_hidden_states_returns_empty() {
        let device = Device::Cpu;
        let mut proposer = make_proposer(&device);
        let tokenizer = make_tokenizer();

        let req_id: RequestId = 3;
        proposer.init_request(req_id, &[1u32, 2]).unwrap();
        // Do NOT call set_target_hidden_states — should return empty gracefully

        let mut state = make_state();
        let result = proposer.propose_for_request(req_id, 3, &mut state, &tokenizer);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
