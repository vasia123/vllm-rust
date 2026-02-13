//! Eagle 1 DraftProposer implementation.
//!
//! Wraps any [`Eagle1DraftModel`] with full lifecycle management. Like
//! [`Eagle3DraftProposer`], this proposer requires target model hidden
//! states via [`set_target_hidden_states`] before each proposal round.
//!
//! Draft token generation flow:
//! 1. Target model produces hidden states → engine calls `set_target_hidden_states`
//! 2. `propose_for_request` runs K sequential Eagle1 forward passes
//! 3. Each step uses the **post-norm** hidden states for logits, and passes
//!    them as input to the next step
//!
//! On first call, the proposer runs a full prefill through Eagle1 to populate
//! its KV cache. Subsequent calls only do single-token decode steps.
//!
//! Reference: `reference/vllm/vllm/model_executor/models/llama_eagle.py`

use std::collections::HashMap;

use candle_core::Tensor;
use tracing::warn;

use crate::engine::types::EngineError;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::models::eagle_llama::Eagle1DraftModel;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

use super::sample_speculative;
use super::DraftProposer;

/// Per-request Eagle1 draft state.
struct Eagle1RequestState {
    block_table: BlockTable,
    seqlen_offset: usize,
    prompt_tokens: Vec<u32>,
    /// Target model hidden states for the current proposal round.
    target_hidden_states: Option<Tensor>,
    /// Whether the Eagle1 KV cache needs an initial prefill.
    needs_prefill: bool,
}

/// [`DraftProposer`] backed by an Eagle 1 speculative decoding model.
///
/// Works with any model implementing [`Eagle1DraftModel`], including:
/// - [`EagleLlamaForCausalLM`](crate::models::EagleLlamaForCausalLM): Llama-based
///
/// The engine must call [`set_target_hidden_states`] before each
/// `propose_for_request` call to provide the target model's context.
pub struct Eagle1DraftProposer {
    model: Box<dyn Eagle1DraftModel>,
    kv_cache: KVCacheManager,
    num_speculative_tokens: usize,
    requests: HashMap<RequestId, Eagle1RequestState>,
}

impl Eagle1DraftProposer {
    pub fn new(
        model: Box<dyn Eagle1DraftModel>,
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

impl DraftProposer for Eagle1DraftProposer {
    fn init_request(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<(), EngineError> {
        let block_table = BlockTable::new(self.kv_cache.block_size());

        self.requests.insert(
            request_id,
            Eagle1RequestState {
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
        let req_state = self.requests.get_mut(&request_id).ok_or_else(|| {
            EngineError::Model(format!(
                "Eagle1: set_target_hidden_states for unknown request {request_id}"
            ))
        })?;
        req_state.target_hidden_states = Some(hidden_states);
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
                "Eagle1: propose_for_request for unknown request {request_id}"
            ))
        })?;

        let hidden_states = match req.target_hidden_states.take() {
            Some(hs) => hs,
            None => {
                warn!("Eagle1: no target hidden states for request {request_id}");
                return Ok(Vec::new());
            }
        };

        let device = self.model.device().clone();
        let k = self.num_speculative_tokens;
        let mut draft_tokens = Vec::with_capacity(k);

        if req.needs_prefill {
            // First call: prefill the Eagle1 KV cache with prompt + last_token
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

            let (postnorm, _) = self
                .model
                .forward(
                    &input_tensor,
                    &hidden_states,
                    0,
                    &mut self.kv_cache,
                    &req.block_table,
                    &slot_mapping,
                )
                .map_err(|e| EngineError::Model(e.to_string()))?;

            req.block_table.advance(seq_len);
            req.seqlen_offset = seq_len;
            req.needs_prefill = false;

            // Take last position as starting hidden state for first decode step
            let seq_dim = postnorm.dims()[1];
            let mut current_hidden = postnorm
                .narrow(1, seq_dim - 1, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;

            // Generate K draft tokens
            for step in 0..k {
                let logits = self
                    .model
                    .compute_logits(&current_hidden)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let token = sample_speculative(&logits, state, tokenizer)
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                draft_tokens.push(token);

                if step + 1 >= k {
                    break;
                }

                // Decode step: single token
                self.kv_cache
                    .allocate_for_request(&mut req.block_table, 1)
                    .map_err(|e| EngineError::Cache(e.to_string()))?;
                let slot_mapping =
                    req.block_table.slot_mapping(req.seqlen_offset + step, 1);

                let token_tensor = Tensor::new(&[[token]], &device)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let (postnorm, _) = self
                    .model
                    .forward(
                        &token_tensor,
                        &current_hidden,
                        req.seqlen_offset + step,
                        &mut self.kv_cache,
                        &req.block_table,
                        &slot_mapping,
                    )
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                current_hidden = postnorm;
            }

            req.seqlen_offset += k;
        } else {
            // Subsequent calls: decode steps only
            let mut current_hidden = hidden_states;

            for step in 0..k {
                self.kv_cache
                    .allocate_for_request(&mut req.block_table, 1)
                    .map_err(|e| EngineError::Cache(e.to_string()))?;
                let slot_mapping =
                    req.block_table.slot_mapping(req.seqlen_offset + step, 1);

                let token_to_use = if step == 0 { last_token } else { draft_tokens[step - 1] };
                let token_tensor = Tensor::new(&[[token_to_use]], &device)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let (postnorm, _) = self
                    .model
                    .forward(
                        &token_tensor,
                        &current_hidden,
                        req.seqlen_offset + step,
                        &mut self.kv_cache,
                        &req.block_table,
                        &slot_mapping,
                    )
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let logits = self
                    .model
                    .compute_logits(&postnorm)
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let token = sample_speculative(&logits, state, tokenizer)
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                draft_tokens.push(token);

                current_hidden = postnorm;
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
                "Eagle1: on_tokens_verified for unknown request {request_id}"
            ))
        })?;

        let draft_total = original_offset + num_accepted;
        let freed = req.block_table.trim_to(draft_total);
        if !freed.is_empty() {
            self.kv_cache
                .free_blocks(&freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        req.seqlen_offset = draft_total;

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
                        "Failed to free Eagle1 cache blocks on request completion"
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
                        "Failed to free Eagle1 cache blocks on preemption"
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
        "eagle1"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eagle1_proposer_creation() {
        assert_eq!("eagle1", "eagle1");
    }

    #[test]
    fn test_eagle1_request_state_lifecycle() {
        let state = Eagle1RequestState {
            block_table: BlockTable::new(16),
            seqlen_offset: 0,
            prompt_tokens: vec![1, 2, 3],
            target_hidden_states: None,
            needs_prefill: true,
        };

        assert_eq!(state.seqlen_offset, 0);
        assert!(state.needs_prefill);
        assert_eq!(state.prompt_tokens.len(), 3);
        assert!(state.target_hidden_states.is_none());
    }
}
