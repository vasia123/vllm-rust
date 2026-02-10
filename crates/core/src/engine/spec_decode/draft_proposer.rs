//! DraftProposer implementation backed by a draft language model.
//!
//! Extracts the draft-model forward loop, KV cache management, and per-request
//! state from the monolithic `SpeculativeExecution` into a self-contained
//! proposer. The engine strategy only needs to call `propose_for_request()` and
//! handle verification — all draft-model details are encapsulated here.

use std::collections::HashMap;

use candle_core::Tensor;
use tracing::warn;

use crate::engine::model_forward::ModelForward;
use crate::engine::types::EngineError;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

use super::DraftProposer;

/// Per-request draft model state.
struct DraftRequestState {
    block_table: BlockTable,
    seqlen_offset: usize,
}

/// [`DraftProposer`] backed by a small draft language model.
///
/// Manages its own KV cache and per-request block tables. Each call to
/// `propose_for_request` runs K sequential forward passes through the draft
/// model, applying the request's sampling parameters (penalties, constraints)
/// at each step via [`sample_speculative`].
pub struct DraftModelDraftProposer<D: ModelForward> {
    model: D,
    kv_cache: KVCacheManager,
    num_speculative_tokens: usize,
    /// Per-request draft state (block table + seqlen offset).
    requests: HashMap<RequestId, DraftRequestState>,
}

impl<D: ModelForward> DraftModelDraftProposer<D> {
    pub fn new(model: D, kv_cache: KVCacheManager, num_speculative_tokens: usize) -> Self {
        Self {
            model,
            kv_cache,
            num_speculative_tokens,
            requests: HashMap::new(),
        }
    }
}

/// Sample a token from logits applying the request's sampling params, penalties,
/// and optional constraint masking. Falls back to greedy argmax when the request
/// uses greedy sampling with no penalties and no constraints.
pub(crate) fn sample_speculative(
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
        return greedy_sample(logits);
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

    let result = crate::sampling::sample(
        &logits_vec,
        &state.sampling_params,
        &state.generated_token_ids,
        &mut state.sampler_state,
        None,
        &state.stop_token_ids,
    );

    Ok(result.token_id)
}

fn greedy_sample(logits: &Tensor) -> Result<u32, EngineError> {
    crate::engine::helpers::greedy_sample(logits).map_err(|e| EngineError::Model(e.to_string()))
}

impl<D: ModelForward> DraftProposer for DraftModelDraftProposer<D> {
    fn init_request(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<(), EngineError> {
        let prompt_len = prompt_tokens.len();
        let mut block_table = BlockTable::new(self.kv_cache.block_size());

        self.kv_cache
            .allocate_for_request(&mut block_table, prompt_len)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = block_table.slot_mapping(0, prompt_len);

        let input = Tensor::from_vec(prompt_tokens.to_vec(), (1, prompt_len), self.model.device())
            .map_err(|e| EngineError::Model(e.to_string()))?;

        let _logits = self
            .model
            .forward(&input, 0, &mut self.kv_cache, &block_table, &slot_mapping)
            .map_err(|e| EngineError::Model(e.to_string()))?;

        block_table.advance(prompt_len);

        self.requests.insert(
            request_id,
            DraftRequestState {
                block_table,
                seqlen_offset: prompt_len,
            },
        );

        Ok(())
    }

    fn propose_for_request(
        &mut self,
        request_id: RequestId,
        last_token: u32,
        state: &mut SequenceState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<Vec<u32>, EngineError> {
        let k = self.num_speculative_tokens;
        let draft_state = self
            .requests
            .get_mut(&request_id)
            .ok_or_else(|| EngineError::Model(format!("draft state for {request_id} not found")))?;

        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_input_token = last_token;

        for _ in 0..k {
            self.kv_cache
                .allocate_for_request(&mut draft_state.block_table, 1)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            let slot_mapping = draft_state
                .block_table
                .slot_mapping(draft_state.seqlen_offset, 1);

            let input = Tensor::new(&[[draft_input_token]], self.model.device())
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let logits = self
                .model
                .forward(
                    &input,
                    draft_state.seqlen_offset,
                    &mut self.kv_cache,
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
            let token = sample_speculative(&logits, state, tokenizer)?;
            draft_tokens.push(token);
            draft_input_token = token;
        }

        Ok(draft_tokens)
    }

    fn on_tokens_verified(
        &mut self,
        request_id: RequestId,
        num_accepted: usize,
        original_offset: usize,
    ) -> Result<(), EngineError> {
        let draft_state = self
            .requests
            .get_mut(&request_id)
            .ok_or_else(|| EngineError::Model(format!("draft state for {request_id} not found")))?;

        let draft_total = original_offset + num_accepted;
        let freed = draft_state.block_table.trim_to(draft_total);
        if !freed.is_empty() {
            self.kv_cache
                .free_blocks(&freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        draft_state.seqlen_offset = draft_total;

        Ok(())
    }

    fn finish_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(mut ds) = self.requests.remove(&request_id) {
            let freed = ds.block_table.release();
            if !freed.is_empty() {
                if let Err(e) = self.kv_cache.free_blocks(&freed) {
                    warn!(
                        error = %e,
                        request_id = request_id,
                        "Failed to free draft cache blocks on request completion"
                    );
                }
            }
        }
        Ok(())
    }

    fn preempt_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(mut ds) = self.requests.remove(&request_id) {
            let freed = ds.block_table.release();
            if !freed.is_empty() {
                if let Err(e) = self.kv_cache.free_blocks(&freed) {
                    warn!(
                        error = %e,
                        request_id = request_id,
                        "Failed to free draft cache blocks during preemption"
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
        "draft_model"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model_forward::ModelForward;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use crate::request::SequenceState;
    use crate::tokenizer::TokenizerWrapper;
    use candle_core::{DType, Device};

    struct ConstantMockModel {
        output_token: u32,
        vocab_size: usize,
        device: Device,
    }

    impl ConstantMockModel {
        fn new(output_token: u32, vocab_size: usize) -> Self {
            Self {
                output_token,
                vocab_size,
                device: Device::Cpu,
            }
        }
    }

    impl ModelForward for ConstantMockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits_vec[pos * self.vocab_size + self.output_token as usize] = 100.0;
            }
            Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn test_cache_config() -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    fn make_proposer(output_token: u32) -> DraftModelDraftProposer<ConstantMockModel> {
        let model = ConstantMockModel::new(output_token, 1000);
        let kv_cache = KVCacheManager::new(&test_cache_config()).expect("cache creation");
        DraftModelDraftProposer::new(model, kv_cache, 3)
    }

    fn make_state(prompt: &[u32]) -> SequenceState {
        SequenceState::new(0, prompt.to_vec(), 100, 999, 16, 0)
    }

    fn tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::for_testing(1000)
    }

    #[test]
    fn init_and_propose_basic() {
        let mut proposer = make_proposer(42);
        let prompt = vec![1u32, 2, 3, 4, 5];
        proposer.init_request(0, &prompt).unwrap();

        let mut state = make_state(&prompt);
        state.generated_token_ids.push(10); // simulate one decode step
        let tok = tokenizer();

        let drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        assert_eq!(drafts, vec![42, 42, 42]);
    }

    #[test]
    fn propose_respects_sampling_params() {
        let mut proposer = make_proposer(42);
        let prompt = vec![1u32, 2, 3];
        proposer.init_request(0, &prompt).unwrap();

        let mut state = make_state(&prompt);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        assert_eq!(drafts.len(), 3);
    }

    #[test]
    fn on_tokens_verified_trims_cache() {
        let mut proposer = make_proposer(42);
        let prompt = vec![1u32, 2, 3, 4, 5];
        proposer.init_request(0, &prompt).unwrap();

        let mut state = make_state(&prompt);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let original_offset = 5; // prompt len
        let _drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();

        // Only 1 of 3 drafts accepted
        proposer.on_tokens_verified(0, 1, original_offset).unwrap();

        // Draft state should be at offset 6 (5 + 1 accepted)
        let ds = proposer.requests.get(&0).unwrap();
        assert_eq!(ds.seqlen_offset, 6);
    }

    #[test]
    fn finish_request_frees_blocks() {
        let mut proposer = make_proposer(42);
        let initial_free = proposer.kv_cache.num_free_blocks();

        proposer.init_request(0, &[1, 2, 3, 4, 5]).unwrap();
        assert!(proposer.kv_cache.num_free_blocks() < initial_free);

        proposer.finish_request(0).unwrap();
        assert_eq!(proposer.kv_cache.num_free_blocks(), initial_free);
        assert!(!proposer.requests.contains_key(&0));
    }

    #[test]
    fn preempt_request_frees_blocks() {
        let mut proposer = make_proposer(42);
        let initial_free = proposer.kv_cache.num_free_blocks();

        proposer.init_request(0, &[1, 2, 3, 4, 5]).unwrap();
        proposer.preempt_request(0).unwrap();
        assert_eq!(proposer.kv_cache.num_free_blocks(), initial_free);
    }

    #[test]
    fn finish_nonexistent_request_is_noop() {
        let mut proposer = make_proposer(42);
        proposer.finish_request(999).unwrap();
    }

    #[test]
    fn name_is_draft_model() {
        let proposer = make_proposer(42);
        assert_eq!(proposer.name(), "draft_model");
    }

    #[test]
    fn num_speculative_tokens_correct() {
        let proposer = make_proposer(42);
        assert_eq!(proposer.num_speculative_tokens(), 3);
    }

    #[test]
    fn multiple_requests_independent() {
        let mut proposer = make_proposer(42);
        proposer.init_request(0, &[1, 2, 3]).unwrap();
        proposer.init_request(1, &[4, 5, 6, 7]).unwrap();

        let mut state0 = make_state(&[1, 2, 3]);
        state0.generated_token_ids.push(10);
        let mut state1 = make_state(&[4, 5, 6, 7]);
        state1.generated_token_ids.push(20);
        let tok = tokenizer();

        let d0 = proposer
            .propose_for_request(0, 10, &mut state0, &tok)
            .unwrap();
        let d1 = proposer
            .propose_for_request(1, 20, &mut state1, &tok)
            .unwrap();

        assert_eq!(d0.len(), 3);
        assert_eq!(d1.len(), 3);

        // Finish one, other still works
        proposer.finish_request(0).unwrap();
        assert!(!proposer.requests.contains_key(&0));
        assert!(proposer.requests.contains_key(&1));
        proposer.finish_request(1).unwrap();
    }

    #[test]
    fn propose_without_init_fails() {
        let mut proposer = make_proposer(42);
        let mut state = make_state(&[1, 2, 3]);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let result = proposer.propose_for_request(999, 10, &mut state, &tok);
        assert!(result.is_err());
    }

    #[test]
    fn on_verified_without_init_fails() {
        let mut proposer = make_proposer(42);
        let result = proposer.on_tokens_verified(999, 1, 5);
        assert!(result.is_err());
    }
}
