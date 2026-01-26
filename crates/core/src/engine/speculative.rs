//! Speculative decoding execution strategy.

use candle_core::Tensor;
use tracing::warn;

use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::request::{RequestId, RequestStatus};
use crate::scheduler::SchedulerOutput;
use crate::tokenizer::TokenizerWrapper;

use super::context::{ActiveRequest, DraftState, OwnedExecutionState};
use super::helpers::{execute_decode, execute_prefill, finish_request_with_error, greedy_sample};
use super::model_forward::ModelForward;
use super::strategy::ExecutionStrategy;
use super::types::EngineError;

/// Speculative decoding execution strategy.
///
/// Uses a smaller draft model to speculatively generate tokens,
/// then verifies them with the target model in a single forward pass.
pub struct SpeculativeExecution<M: ModelForward, D: ModelForward> {
    target_model: M,
    draft_model: D,
    draft_kv_cache: KVCacheManager,
    num_speculative_tokens: usize,
}

impl<M: ModelForward, D: ModelForward> SpeculativeExecution<M, D> {
    pub fn new(
        target_model: M,
        draft_model: D,
        draft_kv_cache: KVCacheManager,
        num_speculative_tokens: usize,
    ) -> Self {
        Self {
            target_model,
            draft_model,
            draft_kv_cache,
            num_speculative_tokens,
        }
    }

    fn execute_draft_prefill(
        &mut self,
        req_id: RequestId,
        requests: &mut std::collections::HashMap<RequestId, ActiveRequest>,
    ) -> Result<(), EngineError> {
        let req = requests
            .get_mut(&req_id)
            .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
        let prompt_len = req.state.prompt_token_ids.len();

        let mut draft_block_table = BlockTable::new(self.draft_kv_cache.block_size());
        self.draft_kv_cache
            .allocate_for_request(&mut draft_block_table, prompt_len)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = draft_block_table.slot_mapping(0, prompt_len);

        let input = Tensor::from_vec(
            req.state.prompt_token_ids.clone(),
            (1, prompt_len),
            self.draft_model.device(),
        )
        .map_err(|e| EngineError::Model(e.to_string()))?;

        let _logits = self
            .draft_model
            .forward(
                &input,
                0,
                &mut self.draft_kv_cache,
                &draft_block_table,
                &slot_mapping,
            )
            .map_err(|e| EngineError::Model(e.to_string()))?;

        draft_block_table.advance(prompt_len);

        req.draft_state = Some(DraftState {
            block_table: draft_block_table,
            seqlen_offset: prompt_len,
        });

        Ok(())
    }

    fn execute_speculative_decode(
        &mut self,
        req_id: RequestId,
        target_kv_cache: &mut KVCacheManager,
        requests: &mut std::collections::HashMap<RequestId, ActiveRequest>,
        tokenizer: &TokenizerWrapper,
    ) -> Result<(), EngineError> {
        let k = {
            let req = requests
                .get(&req_id)
                .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
            let tokens_remaining = req
                .state
                .max_new_tokens
                .saturating_sub(req.state.num_generated());
            self.num_speculative_tokens
                .min(tokens_remaining.saturating_sub(1))
        };

        if k == 0 {
            return execute_decode(req_id, &self.target_model, target_kv_cache, requests, tokenizer);
        }

        let req = requests
            .get_mut(&req_id)
            .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
        let draft_state = req
            .draft_state
            .as_mut()
            .ok_or_else(|| EngineError::Model("draft state not initialized".to_string()))?;

        // --- Draft phase: generate K tokens ---
        let mut draft_tokens = Vec::with_capacity(k);
        let last_target_token = *req.state.generated_token_ids.last().ok_or_else(|| {
            EngineError::Model("no generated tokens for speculative decode".to_string())
        })?;
        let mut draft_input_token = last_target_token;

        for _ in 0..k {
            self.draft_kv_cache
                .allocate_for_request(&mut draft_state.block_table, 1)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            let slot_mapping = draft_state
                .block_table
                .slot_mapping(draft_state.seqlen_offset, 1);

            let input = Tensor::new(&[[draft_input_token]], self.draft_model.device())
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let logits = self
                .draft_model
                .forward(
                    &input,
                    draft_state.seqlen_offset,
                    &mut self.draft_kv_cache,
                    &draft_state.block_table,
                    &slot_mapping,
                )
                .map_err(|e| EngineError::Model(e.to_string()))?;

            draft_state.block_table.advance(1);
            draft_state.seqlen_offset += 1;

            let seq_dim = logits.dims()[1];
            let logits = logits
                .narrow(1, seq_dim - 1, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let token = greedy_sample(&logits).map_err(|e| EngineError::Model(e.to_string()))?;
            draft_tokens.push(token);
            draft_input_token = token;
        }

        // --- Verify phase: target model forward on K+1 tokens ---
        let k_plus_1 = k + 1;
        target_kv_cache
            .allocate_for_request(&mut req.state.block_table, k_plus_1)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = req
            .state
            .block_table
            .slot_mapping(req.state.seqlen_offset, k_plus_1);

        let mut verify_input = vec![last_target_token];
        verify_input.extend_from_slice(&draft_tokens);
        let input = Tensor::from_vec(verify_input, (1, k_plus_1), self.target_model.device())
            .map_err(|e| EngineError::Model(e.to_string()))?;

        let logits = self
            .target_model
            .forward(
                &input,
                req.state.seqlen_offset,
                target_kv_cache,
                &req.state.block_table,
                &slot_mapping,
            )
            .map_err(|e| EngineError::Model(e.to_string()))?;

        req.state.block_table.advance(k_plus_1);

        // --- Greedy verification ---
        let mut accepted = 0;
        for i in 0..k {
            let pos_logits = logits
                .narrow(1, i, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let target_token =
                greedy_sample(&pos_logits).map_err(|e| EngineError::Model(e.to_string()))?;
            if target_token == draft_tokens[i] {
                accepted += 1;
            } else {
                req.state.generated_token_ids.extend(&draft_tokens[..i]);
                req.state.generated_token_ids.push(target_token);
                break;
            }
        }

        if accepted == k {
            let bonus_logits = logits
                .narrow(1, k, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let bonus_token =
                greedy_sample(&bonus_logits).map_err(|e| EngineError::Model(e.to_string()))?;
            req.state.generated_token_ids.extend(&draft_tokens);
            req.state.generated_token_ids.push(bonus_token);
        }

        // --- Rollback: trim caches to actual accepted length ---
        let original_offset = req.state.seqlen_offset;
        let new_tokens = accepted + 1;
        let target_total = original_offset + new_tokens;
        let draft_total = original_offset + accepted;

        let target_freed = req.state.block_table.trim_to(target_total);
        if !target_freed.is_empty() {
            target_kv_cache
                .free_blocks(&target_freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        req.state.seqlen_offset = target_total;

        let draft_state = req.draft_state.as_mut().ok_or_else(|| {
            EngineError::Model("draft state not initialized for rollback".to_string())
        })?;
        let draft_freed = draft_state.block_table.trim_to(draft_total);
        if !draft_freed.is_empty() {
            self.draft_kv_cache
                .free_blocks(&draft_freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        draft_state.seqlen_offset = draft_total;

        Ok(())
    }
}

impl<M: ModelForward, D: ModelForward> ExecutionStrategy for SpeculativeExecution<M, D> {
    fn execute_prefills(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        tokenizer: &TokenizerWrapper,
    ) {
        for schedule in &output.prefill_requests {
            let req_id = schedule.request_id;

            // Target model prefill
            if let Err(e) = execute_prefill(
                req_id,
                schedule.chunk_size,
                &self.target_model,
                kv_cache_mgr,
                &mut state.requests,
                tokenizer,
            ) {
                finish_request_with_error(req_id, e, &mut state.scheduler, &mut state.requests);
                continue;
            }

            // Draft model prefill
            if let Err(e) = self.execute_draft_prefill(req_id, &mut state.requests) {
                finish_request_with_error(req_id, e, &mut state.scheduler, &mut state.requests);
                continue;
            }
        }
    }

    fn execute_decodes(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        _multi_step_count: usize,
        tokenizer: &TokenizerWrapper,
    ) {
        // Speculative decoding doesn't use multi-step in the same way
        for &req_id in &output.decode_requests {
            let result = self.execute_speculative_decode(req_id, kv_cache_mgr, &mut state.requests, tokenizer);
            if let Err(e) = result {
                finish_request_with_error(req_id, e, &mut state.scheduler, &mut state.requests);
            }
        }
    }

    fn handle_preemptions(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
    ) {
        for &req_id in &output.preempted_requests {
            let Some(req) = state.requests.get_mut(&req_id) else {
                continue;
            };

            if let Err(e) = kv_cache_mgr.free_request(&mut req.state.block_table) {
                warn!(
                    error = %e,
                    request_id = req_id,
                    "Failed to free target cache blocks during preemption"
                );
            }

            if let Some(ref mut ds) = req.draft_state {
                let freed = ds.block_table.release();
                if !freed.is_empty() {
                    if let Err(e) = self.draft_kv_cache.free_blocks(&freed) {
                        warn!(
                            error = %e,
                            request_id = req_id,
                            blocks = ?freed,
                            "Failed to free draft cache blocks during preemption"
                        );
                    }
                }
            }
            req.draft_state = None;

            req.state.status = RequestStatus::Preempted;
            req.state.generated_token_ids.clear();
            req.state.seqlen_offset = 0;
        }
    }

    fn on_request_completed(&mut self, req_id: RequestId, state: &mut OwnedExecutionState) {
        if let Some(req) = state.requests.get_mut(&req_id) {
            if let Some(mut ds) = req.draft_state.take() {
                let freed = ds.block_table.release();
                if !freed.is_empty() {
                    if let Err(e) = self.draft_kv_cache.free_blocks(&freed) {
                        warn!(
                            error = %e,
                            request_id = req_id,
                            blocks = ?freed,
                            "Failed to free draft cache blocks on request completion"
                        );
                    }
                }
            }
        }
    }
}
