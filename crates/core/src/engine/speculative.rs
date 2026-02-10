//! Speculative decoding execution strategy.
//!
//! Delegates draft token generation to a [`DraftProposer`] and handles
//! verification with the target model. The proposer manages its own KV cache
//! and per-request state, keeping the strategy focused on orchestration.

use candle_core::Tensor;
use tracing::warn;

use crate::kv_cache::KVCacheManager;
use crate::request::{RequestId, RequestStatus, SequenceState};
use crate::sampling;
use crate::scheduler::SchedulerOutput;
use crate::tokenizer::TokenizerWrapper;

use super::context::{ActiveRequest, OwnedExecutionState};
use super::helpers::{execute_decode, execute_prefill, finish_request_with_error, greedy_sample};
use super::model_forward::ModelForward;
use super::spec_decode::DraftProposer;
use super::strategy::ExecutionStrategy;
use super::types::{EngineError, SpecDecodingStats};

/// Sample a token from logits applying the request's sampling params, penalties,
/// and optional constraint masking. Falls back to greedy argmax when the request
/// uses greedy sampling with no penalties and no constraints.
fn sample_speculative(
    logits: &Tensor,
    state: &mut SequenceState,
    tokenizer: &TokenizerWrapper,
) -> Result<u32, EngineError> {
    let has_penalties = state.sampling_params.repetition_penalty != 1.0
        || state.sampling_params.frequency_penalty != 0.0
        || state.sampling_params.presence_penalty != 0.0
        || state.sampling_params.logit_bias.is_some()
        || state
            .sampling_params
            .banned_token_ids
            .as_ref()
            .is_some_and(|b| !b.is_empty())
        || state.sampling_params.allowed_token_ids.is_some()
        || (state.sampling_params.min_tokens > 0 && state.sampling_params.eos_token_id.is_some());
    let has_constraint = state.constraint.is_some();

    if state.sampling_params.is_greedy() && !has_penalties && !has_constraint {
        return greedy_sample(logits).map_err(|e| EngineError::Model(e.to_string()));
    }

    let mut logits_vec: Vec<f32> = logits
        .squeeze(0)
        .and_then(|t| t.squeeze(0))
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    if let Some(ref mut constraint) = state.constraint {
        let generated_text = tokenizer
            .decode(&state.generated_token_ids)
            .unwrap_or_default();
        constraint
            .mask_logits(&mut logits_vec, &generated_text)
            .map_err(|e| EngineError::Model(format!("constraint error: {}", e)))?;
    }

    let result = sampling::sample(
        &logits_vec,
        &state.sampling_params,
        &state.generated_token_ids,
        &mut state.sampler_state,
        None,
    );

    Ok(result.token_id)
}

/// Speculative decoding execution strategy.
///
/// Uses a [`DraftProposer`] to speculatively generate tokens, then verifies
/// them with the target model in a single forward pass. The proposer owns
/// the draft model and its KV cache; this strategy only owns the target model.
pub struct SpeculativeExecution<M: ModelForward> {
    target_model: M,
    proposer: Box<dyn DraftProposer>,
    stats: SpecDecodingStats,
}

impl<M: ModelForward> SpeculativeExecution<M> {
    pub fn new(target_model: M, proposer: Box<dyn DraftProposer>) -> Self {
        let k = proposer.num_speculative_tokens();
        Self {
            target_model,
            proposer,
            stats: SpecDecodingStats::new(k),
        }
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
            self.proposer
                .num_speculative_tokens()
                .min(tokens_remaining.saturating_sub(1))
        };

        if k == 0 {
            return execute_decode(
                req_id,
                &self.target_model,
                target_kv_cache,
                requests,
                tokenizer,
            );
        }

        // --- Draft phase: delegate to proposer ---
        let last_target_token = {
            let req = requests
                .get(&req_id)
                .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
            *req.state.generated_token_ids.last().ok_or_else(|| {
                EngineError::Model("no generated tokens for speculative decode".to_string())
            })?
        };

        let draft_tokens = {
            let req = requests
                .get_mut(&req_id)
                .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
            self.proposer.propose_for_request(
                req_id,
                last_target_token,
                &mut req.state,
                tokenizer,
            )?
        };
        let actual_k = draft_tokens.len().min(k);

        if actual_k == 0 {
            return execute_decode(
                req_id,
                &self.target_model,
                target_kv_cache,
                requests,
                tokenizer,
            );
        }

        // --- Verify phase: target model forward on K+1 tokens ---
        let req = requests
            .get_mut(&req_id)
            .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;

        let k_plus_1 = actual_k + 1;
        target_kv_cache
            .allocate_for_request(&mut req.state.block_table, k_plus_1)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = req
            .state
            .block_table
            .slot_mapping(req.state.seqlen_offset, k_plus_1);

        let mut verify_input = vec![last_target_token];
        verify_input.extend_from_slice(&draft_tokens[..actual_k]);
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

        // --- Verification with penalties and constraints ---
        let mut accepted = 0;
        let gen_len_before = req.state.generated_token_ids.len();
        for (i, &draft_token) in draft_tokens.iter().enumerate().take(actual_k) {
            let pos_logits = logits
                .narrow(1, i, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let target_token = sample_speculative(&pos_logits, &mut req.state, tokenizer)?;
            if target_token == draft_token {
                accepted += 1;
                req.state.generated_token_ids.push(draft_token);
            } else {
                req.state.generated_token_ids.push(target_token);
                break;
            }
        }

        if accepted == actual_k {
            let bonus_logits = logits
                .narrow(1, actual_k, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let bonus_token = sample_speculative(&bonus_logits, &mut req.state, tokenizer)?;
            req.state.generated_token_ids.push(bonus_token);
        }

        debug_assert_eq!(
            req.state.generated_token_ids.len(),
            gen_len_before + accepted + 1
        );

        // --- Rollback: trim target cache to actual accepted length ---
        let original_offset = req.state.seqlen_offset;
        let new_tokens = if req.state.generated_token_ids.is_empty() {
            warn!(request_id = req_id, "speculative decode produced no tokens");
            0
        } else {
            accepted + 1
        };
        let target_total = original_offset + new_tokens;

        let target_freed = req.state.block_table.trim_to(target_total);
        if !target_freed.is_empty() {
            target_kv_cache
                .free_blocks(&target_freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        req.state.seqlen_offset = target_total;

        // Notify proposer of verification results so it can trim its KV cache
        self.proposer
            .on_tokens_verified(req_id, accepted, original_offset)?;

        // Record spec decode statistics
        self.stats.observe_draft(actual_k, accepted);

        Ok(())
    }
}

impl<M: ModelForward> ExecutionStrategy for SpeculativeExecution<M> {
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

            // Draft model prefill (delegated to proposer)
            let prompt_tokens = match state.requests.get(&req_id) {
                Some(req) => req.state.prompt_token_ids.clone(),
                None => continue,
            };
            if let Err(e) = self.proposer.init_request(req_id, &prompt_tokens) {
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
        // When chunked prefill is active in the same step, skip speculative
        // decoding and fall back to single-token decode.
        let has_active_prefill = !output.prefill_requests.is_empty();

        for &req_id in &output.decode_requests {
            let result = if has_active_prefill {
                execute_decode(
                    req_id,
                    &self.target_model,
                    kv_cache_mgr,
                    &mut state.requests,
                    tokenizer,
                )
            } else {
                self.execute_speculative_decode(
                    req_id,
                    kv_cache_mgr,
                    &mut state.requests,
                    tokenizer,
                )
            };
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

            // Delegate draft cache cleanup to proposer
            if let Err(e) = self.proposer.preempt_request(req_id) {
                warn!(
                    error = %e,
                    request_id = req_id,
                    "Failed to preempt draft proposer state"
                );
            }

            req.state.status = RequestStatus::Preempted;
            req.state.generated_token_ids.clear();
            req.state.num_computed_tokens = 0;
            req.state.seqlen_offset = 0;
        }
    }

    fn spec_decode_stats(&self) -> Option<SpecDecodingStats> {
        Some(self.stats.clone())
    }

    fn on_request_completed(&mut self, req_id: RequestId, state: &mut OwnedExecutionState) {
        if state.requests.contains_key(&req_id) {
            if let Err(e) = self.proposer.finish_request(req_id) {
                warn!(
                    error = %e,
                    request_id = req_id,
                    "Failed to clean up draft proposer state"
                );
            }
        }
    }
}
