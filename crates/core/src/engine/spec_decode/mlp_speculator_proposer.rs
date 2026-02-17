//! MLP Speculator draft proposer for speculative decoding.
//!
//! Wraps [`MLPSpeculatorModel`] as a [`DraftProposer`]. The proposer receives
//! target model hidden states via `set_target_hidden_states()` and generates
//! K draft tokens by sequential MLP prediction (no KV cache, no attention).
//!
//! The MLP Speculator is stateless except for cached hidden states — it has
//! no KV cache to manage, making lifecycle hooks trivial.

use std::collections::HashMap;

use candle_core::Tensor;

use super::{sample_speculative, DraftProposer};
use crate::engine::types::EngineError;
use crate::models::mlp_speculator::MLPSpeculatorModel;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

/// Per-request state for MLP Speculator proposer.
struct RequestState {
    /// Cached target model hidden states for next proposal.
    target_hidden_states: Option<Tensor>,
}

/// DraftProposer implementation backed by an MLP Speculator model.
///
/// The proposer:
/// 1. Receives target hidden states from the engine (via `set_target_hidden_states`)
/// 2. Runs the MLP Speculator to produce K sequential token predictions
/// 3. Returns the predicted tokens as draft proposals
///
/// No KV cache management is needed since MLP Speculator uses only MLPs.
pub struct MLPSpeculatorDraftProposer {
    model: MLPSpeculatorModel,
    num_speculative_tokens: usize,
    requests: HashMap<RequestId, RequestState>,
}

impl MLPSpeculatorDraftProposer {
    pub fn new(model: MLPSpeculatorModel) -> Self {
        let num_speculative_tokens = model.n_predict();
        Self {
            model,
            num_speculative_tokens,
            requests: HashMap::new(),
        }
    }
}

impl DraftProposer for MLPSpeculatorDraftProposer {
    fn init_request(
        &mut self,
        request_id: RequestId,
        _prompt_tokens: &[u32],
    ) -> Result<(), EngineError> {
        self.requests.insert(
            request_id,
            RequestState {
                target_hidden_states: None,
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
        let req_state = self
            .requests
            .get(&request_id)
            .ok_or_else(|| EngineError::Model(format!("unknown request {request_id}")))?;

        let hidden_states = req_state.target_hidden_states.as_ref().ok_or_else(|| {
            EngineError::Model(format!(
                "MLP Speculator: no target hidden states for request {request_id}"
            ))
        })?;

        // Run MLP Speculator to get logits for each head
        let logits_list = self
            .model
            .forward(hidden_states, last_token)
            .map_err(|e| EngineError::Model(format!("MLP Speculator forward: {e}")))?;

        // Sample from each head's logits using the sampling pipeline
        let mut draft_tokens = Vec::with_capacity(logits_list.len());
        for logits in &logits_list {
            let token = sample_speculative(logits, state, tokenizer)?;
            draft_tokens.push(token);
        }

        Ok(draft_tokens)
    }

    fn on_tokens_verified(
        &mut self,
        _request_id: RequestId,
        _num_accepted: usize,
        _original_offset: usize,
    ) -> Result<(), EngineError> {
        // No KV cache to trim
        Ok(())
    }

    fn finish_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        self.requests.remove(&request_id);
        Ok(())
    }

    fn preempt_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        self.requests.remove(&request_id);
        Ok(())
    }

    fn set_target_hidden_states(
        &mut self,
        request_id: RequestId,
        hidden_states: Tensor,
    ) -> Result<(), EngineError> {
        if let Some(req_state) = self.requests.get_mut(&request_id) {
            req_state.target_hidden_states = Some(hidden_states);
        }
        Ok(())
    }

    fn num_speculative_tokens(&self) -> usize {
        self.num_speculative_tokens
    }

    fn name(&self) -> &str {
        "mlp_speculator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    const VOCAB_SIZE: usize = 256;
    const EMB_DIM: usize = 64;
    const INNER_DIM: usize = 32;
    const N_PREDICT: usize = 3;

    fn create_proposer() -> MLPSpeculatorDraftProposer {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, false, vb)
                .expect("build model");
        MLPSpeculatorDraftProposer::new(model)
    }

    fn create_state() -> SequenceState {
        let mut state = SequenceState::new(0, vec![1, 2, 3], 100, 255, 16, 0);
        state.generated_token_ids.push(4);
        state
    }

    #[test]
    fn test_construction() {
        let proposer = create_proposer();
        assert_eq!(proposer.num_speculative_tokens(), N_PREDICT);
        assert_eq!(proposer.name(), "mlp_speculator");
    }

    #[test]
    fn test_lifecycle() {
        let mut proposer = create_proposer();
        let req_id = 1u64;

        // Init
        proposer.init_request(req_id, &[1, 2, 3]).unwrap();
        assert!(proposer.requests.contains_key(&req_id));

        // Finish
        proposer.finish_request(req_id).unwrap();
        assert!(!proposer.requests.contains_key(&req_id));
    }

    #[test]
    fn test_preempt() {
        let mut proposer = create_proposer();
        let req_id = 1u64;

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();
        proposer.preempt_request(req_id).unwrap();
        assert!(!proposer.requests.contains_key(&req_id));
    }

    #[test]
    fn test_propose_with_hidden_states() {
        let mut proposer = create_proposer();
        let req_id = 1u64;
        let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();

        let hidden = Tensor::zeros((1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden states");
        proposer.set_target_hidden_states(req_id, hidden).unwrap();

        let mut state = create_state();
        let tokens = proposer
            .propose_for_request(req_id, 4, &mut state, &tokenizer)
            .unwrap();

        assert_eq!(tokens.len(), N_PREDICT);
        for &t in &tokens {
            assert!(
                (t as usize) < VOCAB_SIZE,
                "draft token {t} should be < {VOCAB_SIZE}"
            );
        }
    }

    #[test]
    fn test_propose_without_hidden_states_fails() {
        let mut proposer = create_proposer();
        let req_id = 1u64;
        let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();

        // Don't set hidden states
        let mut state = create_state();
        let result = proposer.propose_for_request(req_id, 4, &mut state, &tokenizer);
        assert!(result.is_err(), "should fail without hidden states");
    }

    #[test]
    fn test_on_tokens_verified_is_noop() {
        let mut proposer = create_proposer();
        let req_id = 1u64;

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();
        // Should succeed without error (no KV cache)
        proposer.on_tokens_verified(req_id, 2, 10).unwrap();
    }

    #[test]
    fn test_multiple_requests() {
        let mut proposer = create_proposer();
        let req1 = 1u64;
        let req2 = 2u64;

        proposer.init_request(req1, &[1, 2]).unwrap();
        proposer.init_request(req2, &[3, 4]).unwrap();

        assert!(proposer.requests.contains_key(&req1));
        assert!(proposer.requests.contains_key(&req2));

        proposer.finish_request(req1).unwrap();
        assert!(!proposer.requests.contains_key(&req1));
        assert!(proposer.requests.contains_key(&req2));
    }
}
