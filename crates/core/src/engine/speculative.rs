//! Speculative decoding execution strategy.
//!
//! Delegates draft token generation to a [`DraftProposer`] and handles
//! verification with the target model. The proposer manages its own KV cache
//! and per-request state, keeping the strategy focused on orchestration.

use candle_core::Tensor;
use rand::Rng;
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

/// Compute target model probabilities for a single position.
///
/// Applies all penalties, constraints, temperature scaling, and softmax to produce
/// a valid probability distribution. Used in the random-sampling rejection path.
fn compute_target_probs(
    logits: &Tensor,
    state: &mut SequenceState,
    tokenizer: &TokenizerWrapper,
) -> Result<Vec<f32>, EngineError> {
    let mut logits_vec: Vec<f32> = logits
        .squeeze(0)
        .and_then(|t| t.squeeze(0))
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    // Apply penalties (same as sample_speculative)
    if state.sampling_params.repetition_penalty != 1.0 {
        sampling::apply_repetition_penalty(
            &mut logits_vec,
            &state.generated_token_ids,
            state.sampling_params.repetition_penalty,
        );
    }
    if state.sampling_params.frequency_penalty != 0.0
        || state.sampling_params.presence_penalty != 0.0
    {
        sampling::apply_frequency_presence_penalty(
            &mut logits_vec,
            &state.generated_token_ids,
            state.sampling_params.frequency_penalty,
            state.sampling_params.presence_penalty,
        );
    }
    if let Some(ref bias) = state.sampling_params.logit_bias {
        sampling::apply_logit_bias(&mut logits_vec, bias);
    }
    if let Some(ref banned) = state.sampling_params.banned_token_ids {
        if !banned.is_empty() {
            sampling::apply_banned_tokens(&mut logits_vec, banned);
        }
    }
    if let Some(ref allowed) = state.sampling_params.allowed_token_ids {
        if !allowed.is_empty() {
            sampling::apply_allowed_tokens(&mut logits_vec, allowed);
        }
    }
    if let Some(ref bad_words) = state.sampling_params.bad_words_token_ids {
        if !bad_words.is_empty() {
            sampling::apply_bad_words(&mut logits_vec, bad_words, &state.generated_token_ids);
        }
    }
    if let Some(ref mut constraint) = state.constraint {
        let generated_text = tokenizer
            .decode(&state.generated_token_ids)
            .unwrap_or_default();
        constraint
            .mask_logits(&mut logits_vec, &generated_text)
            .map_err(|e| EngineError::Model(format!("constraint error: {}", e)))?;
    }
    if state.sampling_params.min_tokens > 0 {
        if let Some(eos_id) = state.sampling_params.eos_token_id {
            sampling::apply_min_tokens_suppression(
                &mut logits_vec,
                eos_id,
                state.sampling_params.min_tokens,
                state.generated_token_ids.len(),
            );
        }
    }

    // Apply temperature
    let temperature = state.sampling_params.temperature;
    if temperature != 1.0 && temperature > 1e-6 {
        let inv_temp = 1.0 / temperature;
        for logit in logits_vec.iter_mut() {
            *logit *= inv_temp;
        }
    }

    // Softmax to get probabilities
    Ok(sampling::softmax(&logits_vec))
}

/// Sample a recovered token when a draft token is rejected.
///
/// Uses the adjusted distribution: max(target_prob - draft_prob, 0) for each vocab token.
/// When draft probabilities aren't available (ngram/suffix proposers), draft_prob is
/// effectively 1 for the draft token and 0 for others, so the adjusted distribution
/// is target_prob with the draft token zeroed out.
fn sample_recovered_token(
    target_probs: &[f32],
    draft_token: u32,
    state: &mut SequenceState,
) -> u32 {
    // Adjusted distribution: zero out the draft token, keep everything else
    // This is the no-draft-probs case: max(target_prob - 1{token==draft}, 0) = target_prob for non-draft
    let mut adjusted = target_probs.to_vec();
    if let Some(p) = adjusted.get_mut(draft_token as usize) {
        *p = 0.0;
    }

    // Renormalize
    let sum: f32 = adjusted.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in adjusted.iter_mut() {
            *p *= inv_sum;
        }
    }

    // Sample from adjusted distribution
    let r: f32 = state.sampler_state.rng_mut().gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in adjusted.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    // Fallback: return last non-zero
    adjusted.len() as u32 - 1
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

        // --- Verification via rejection sampling (arxiv:2211.17192) ---
        //
        // Greedy path: compare draft token with target argmax.
        // Random path: accept draft with probability min(1, target_prob[draft] / draft_prob[draft]),
        // where draft_prob = 1 when the proposer doesn't provide probabilities.
        // On rejection, sample a "recovered" token from max(target_prob - draft_prob, 0).
        let is_greedy = req.state.sampling_params.is_greedy();
        let mut accepted = 0;
        let gen_len_before = req.state.generated_token_ids.len();

        for (i, &draft_token) in draft_tokens.iter().enumerate().take(actual_k) {
            let pos_logits = logits
                .narrow(1, i, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;

            if is_greedy {
                // Greedy: accept if target argmax matches draft
                let target_token =
                    sample_speculative(&pos_logits, &mut req.state, tokenizer)?;
                if target_token == draft_token {
                    accepted += 1;
                    req.state.generated_token_ids.push(draft_token);
                } else {
                    req.state.generated_token_ids.push(target_token);
                    break;
                }
            } else {
                // Random: probability-based acceptance/rejection
                let target_probs = compute_target_probs(
                    &pos_logits,
                    &mut req.state,
                    tokenizer,
                )?;
                let target_prob_of_draft = target_probs
                    .get(draft_token as usize)
                    .copied()
                    .unwrap_or(0.0);

                // Accept with probability target_prob (draft_prob = 1 for no-probs proposers)
                let u: f64 = req.state.sampler_state.rng_mut().gen();
                if (target_prob_of_draft as f64) >= u {
                    accepted += 1;
                    req.state.generated_token_ids.push(draft_token);
                } else {
                    // Reject: sample recovered token from adjusted distribution
                    let recovered = sample_recovered_token(
                        &target_probs,
                        draft_token,
                        &mut req.state,
                    );
                    req.state.generated_token_ids.push(recovered);
                    break;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::SequenceState;
    use crate::sampling::SamplerState;
    use crate::tokenizer::TokenizerWrapper;

    fn make_state_with_temp(temperature: f32, seed: u64) -> SequenceState {
        let mut state = SequenceState::new(0, vec![1], 100, 2, 16, 0);
        state.sampling_params.temperature = temperature;
        state.sampler_state = SamplerState::new(Some(seed));
        state.generated_token_ids.push(10);
        state.seqlen_offset = 1;
        state
    }

    #[test]
    fn test_sample_recovered_token_zeros_draft() {
        // With uniform probs [0.25, 0.25, 0.25, 0.25] and draft_token=1,
        // adjusted should be [0.25, 0.0, 0.25, 0.25] → normalized [1/3, 0, 1/3, 1/3]
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let mut state = make_state_with_temp(1.0, 42);
        let token = sample_recovered_token(&probs, 1, &mut state);
        // Should never return the draft token (1)
        assert_ne!(token, 1);
        assert!(token < 4);
    }

    #[test]
    fn test_sample_recovered_token_deterministic_with_seed() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        let mut state1 = make_state_with_temp(1.0, 123);
        let mut state2 = make_state_with_temp(1.0, 123);

        let t1 = sample_recovered_token(&probs, 1, &mut state1);
        let t2 = sample_recovered_token(&probs, 1, &mut state2);
        assert_eq!(t1, t2, "same seed should produce same recovered token");
    }

    #[test]
    fn test_sample_recovered_token_with_dominant_prob() {
        // Token 2 has 0.99 probability, draft token is 0
        // After zeroing token 0: [0.0, 0.005, 0.99, 0.005] → almost always picks token 2
        let probs = vec![0.005, 0.005, 0.99, 0.005];
        let mut count_2 = 0;
        for i in 0..100 {
            let mut s = make_state_with_temp(1.0, i);
            let t = sample_recovered_token(&probs, 0, &mut s);
            if t == 2 {
                count_2 += 1;
            }
        }
        assert!(count_2 > 90, "token 2 should dominate: got {count_2}/100");
    }

    #[test]
    fn test_compute_target_probs_sums_to_one() {
        let logits_data = vec![1.0f32, 2.0, 3.0, 0.5];
        let logits = Tensor::from_vec(logits_data, (1, 1, 4), &candle_core::Device::Cpu).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(4);
        let mut state = make_state_with_temp(1.0, 42);

        let probs = compute_target_probs(&logits, &mut state, &tokenizer).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probs should sum to ~1.0, got {sum}"
        );
        assert!(probs.iter().all(|&p| p >= 0.0), "all probs must be >= 0");
    }

    #[test]
    fn test_compute_target_probs_with_temperature() {
        let logits_data = vec![1.0f32, 3.0, 1.0, 1.0];
        let logits = Tensor::from_vec(logits_data, (1, 1, 4), &candle_core::Device::Cpu).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(4);

        // Low temperature → sharper distribution
        let mut state_low = make_state_with_temp(0.1, 42);
        let probs_low = compute_target_probs(&logits, &mut state_low, &tokenizer).unwrap();

        // High temperature → flatter distribution
        let mut state_high = make_state_with_temp(2.0, 42);
        let probs_high = compute_target_probs(&logits, &mut state_high, &tokenizer).unwrap();

        // Token 1 (highest logit) should have higher prob at low temp
        assert!(
            probs_low[1] > probs_high[1],
            "low temp should sharpen: {:.4} vs {:.4}",
            probs_low[1],
            probs_high[1]
        );
    }

    #[test]
    fn test_compute_target_probs_with_repetition_penalty() {
        let logits_data = vec![2.0f32, 2.0, 2.0, 2.0];
        let logits = Tensor::from_vec(logits_data, (1, 1, 4), &candle_core::Device::Cpu).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(4);

        // Token 10 was already generated, but it's out of vocab range (4 tokens),
        // so use token 0 as the repeated token
        let mut state = make_state_with_temp(1.0, 42);
        state.generated_token_ids = vec![0]; // token 0 was generated
        state.sampling_params.repetition_penalty = 2.0;

        let probs = compute_target_probs(&logits, &mut state, &tokenizer).unwrap();
        // Token 0 should have lower probability due to repetition penalty
        assert!(
            probs[0] < probs[1],
            "repeated token should have lower prob: {:.4} vs {:.4}",
            probs[0],
            probs[1]
        );
    }

    #[test]
    fn test_rejection_acceptance_rate_greedy_perfect() {
        // When draft tokens match target argmax, acceptance should be 100%
        // This is a property test: with deterministic logits, greedy sampling is predictable
        let logits_data = vec![
            // Position 0: argmax at token 5
            0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            // Position 1: argmax at token 3
            0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let logits =
            Tensor::from_vec(logits_data, (1, 2, 8), &candle_core::Device::Cpu).unwrap();

        // Draft tokens match exactly
        let draft_tokens = vec![5u32, 3];
        let mut accepted = 0;

        for (i, &draft_token) in draft_tokens.iter().enumerate().take(2) {
            let pos_logits = logits.narrow(1, i, 1).unwrap();
            let target = greedy_sample(&pos_logits).unwrap();
            if target == draft_token {
                accepted += 1;
            } else {
                break;
            }
        }
        assert_eq!(accepted, 2, "all draft tokens should be accepted");
    }

    #[test]
    fn test_rejection_greedy_mismatch_stops() {
        let logits_data = vec![
            // Position 0: argmax at token 5
            0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            // Position 1: argmax at token 3
            0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let logits =
            Tensor::from_vec(logits_data, (1, 2, 8), &candle_core::Device::Cpu).unwrap();

        // Draft tokens: first matches, second doesn't
        let draft_tokens = vec![5u32, 7];
        let mut accepted = 0;

        for (i, &draft_token) in draft_tokens.iter().enumerate().take(2) {
            let pos_logits = logits.narrow(1, i, 1).unwrap();
            let target = greedy_sample(&pos_logits).unwrap();
            if target == draft_token {
                accepted += 1;
            } else {
                break;
            }
        }
        assert_eq!(accepted, 1, "should accept first, reject second");
    }

    #[test]
    fn test_rejection_random_high_prob_accept() {
        // If target prob for draft token is very high (0.99), almost all seeds accept
        let logits_data = vec![10.0f32, -10.0, -10.0, -10.0]; // softmax ≈ [0.999, ~0, ~0, ~0]
        let logits =
            Tensor::from_vec(logits_data, (1, 1, 4), &candle_core::Device::Cpu).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(4);

        let mut accept_count = 0;
        for seed in 0..100u64 {
            let mut state = make_state_with_temp(1.0, seed);
            let probs = compute_target_probs(&logits, &mut state, &tokenizer).unwrap();
            let prob_draft = probs[0]; // draft token = 0

            let u: f64 = state.sampler_state.rng_mut().gen();
            if (prob_draft as f64) >= u {
                accept_count += 1;
            }
        }
        assert!(
            accept_count > 90,
            "high-prob draft should be accepted often: {accept_count}/100"
        );
    }

    #[test]
    fn test_rejection_random_low_prob_reject() {
        // If target prob for draft token is very low (~0.0), almost all seeds reject
        let logits_data = vec![-10.0f32, 10.0, -10.0, -10.0]; // softmax ≈ [~0, 0.999, ~0, ~0]
        let logits =
            Tensor::from_vec(logits_data, (1, 1, 4), &candle_core::Device::Cpu).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(4);

        let mut reject_count = 0;
        for seed in 0..100u64 {
            let mut state = make_state_with_temp(1.0, seed);
            let probs = compute_target_probs(&logits, &mut state, &tokenizer).unwrap();
            let prob_draft = probs[0]; // draft token = 0 (very low prob)

            let u: f64 = state.sampler_state.rng_mut().gen();
            if (prob_draft as f64) < u {
                reject_count += 1;
            }
        }
        assert!(
            reject_count > 90,
            "low-prob draft should be rejected often: {reject_count}/100"
        );
    }
}
