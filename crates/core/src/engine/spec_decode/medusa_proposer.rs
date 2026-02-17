//! Medusa draft proposer for speculative decoding.
//!
//! Wraps [`MedusaHead`]s as a [`DraftProposer`]. The proposer receives target
//! model hidden states via `set_target_hidden_states()` and generates K draft
//! tokens by running each head independently on the same hidden states.
//!
//! Medusa is stateless except for cached hidden states — it has no KV cache
//! to manage, making lifecycle hooks trivial.

use std::collections::HashMap;

use candle_core::Tensor;

use super::medusa::MedusaHead;
use super::{sample_speculative, DraftProposer};
use crate::engine::types::EngineError;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

/// Per-request state for Medusa proposer.
struct RequestState {
    /// Cached target model hidden states for next proposal.
    target_hidden_states: Option<Tensor>,
}

/// DraftProposer implementation backed by Medusa prediction heads.
///
/// The proposer:
/// 1. Receives target hidden states from the engine (via `set_target_hidden_states`)
/// 2. Runs each Medusa head independently on the hidden states
/// 3. Samples from each head's logits to produce K draft tokens
///
/// No KV cache management is needed since Medusa heads are simple MLPs.
pub struct MedusaDraftProposer {
    heads: Vec<MedusaHead>,
    num_speculative_tokens: usize,
    requests: HashMap<RequestId, RequestState>,
}

impl MedusaDraftProposer {
    pub fn new(heads: Vec<MedusaHead>) -> Self {
        let num_speculative_tokens = heads.len();
        Self {
            heads,
            num_speculative_tokens,
            requests: HashMap::new(),
        }
    }
}

impl DraftProposer for MedusaDraftProposer {
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
        _last_token: u32,
        state: &mut SequenceState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<Vec<u32>, EngineError> {
        let req_state = self
            .requests
            .get(&request_id)
            .ok_or_else(|| EngineError::Model(format!("unknown request {request_id}")))?;

        let hidden_states = req_state.target_hidden_states.as_ref().ok_or_else(|| {
            EngineError::Model(format!(
                "Medusa: no target hidden states for request {request_id}"
            ))
        })?;

        // Run each Medusa head independently and sample
        let mut draft_tokens = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            let logits = head
                .forward(hidden_states)
                .map_err(|e| EngineError::Model(format!("Medusa head forward: {e}")))?;
            let token = sample_speculative(&logits, state, tokenizer)?;
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
        "medusa"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    const HIDDEN_SIZE: usize = 64;
    const VOCAB_SIZE: usize = 256;
    const NUM_HEADS: usize = 3;

    fn create_heads() -> Vec<MedusaHead> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        (0..NUM_HEADS)
            .map(|i| MedusaHead::new(HIDDEN_SIZE, VOCAB_SIZE, 1, vb.pp(format!("head.{i}"))))
            .collect::<Result<Vec<_>, _>>()
            .expect("create heads")
    }

    fn create_state() -> SequenceState {
        let mut state = SequenceState::new(0, vec![1, 2, 3], 100, 255, 16, 0);
        state.generated_token_ids.push(4);
        state
    }

    #[test]
    fn test_construction() {
        let proposer = MedusaDraftProposer::new(create_heads());
        assert_eq!(proposer.num_speculative_tokens(), NUM_HEADS);
        assert_eq!(proposer.name(), "medusa");
    }

    #[test]
    fn test_lifecycle() {
        let mut proposer = MedusaDraftProposer::new(create_heads());
        let req_id = 1u64;

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();
        assert!(proposer.requests.contains_key(&req_id));

        proposer.finish_request(req_id).unwrap();
        assert!(!proposer.requests.contains_key(&req_id));
    }

    #[test]
    fn test_preempt() {
        let mut proposer = MedusaDraftProposer::new(create_heads());
        let req_id = 1u64;

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();
        proposer.preempt_request(req_id).unwrap();
        assert!(!proposer.requests.contains_key(&req_id));
    }

    #[test]
    fn test_propose_with_hidden_states() {
        let mut proposer = MedusaDraftProposer::new(create_heads());
        let req_id = 1u64;
        let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();

        let hidden =
            Tensor::zeros((1, HIDDEN_SIZE), DType::F32, &Device::Cpu).expect("hidden states");
        proposer.set_target_hidden_states(req_id, hidden).unwrap();

        let mut state = create_state();
        let tokens = proposer
            .propose_for_request(req_id, 4, &mut state, &tokenizer)
            .unwrap();

        assert_eq!(tokens.len(), NUM_HEADS);
        for &t in &tokens {
            assert!(
                (t as usize) < VOCAB_SIZE,
                "draft token {t} should be < {VOCAB_SIZE}"
            );
        }
    }

    #[test]
    fn test_propose_without_hidden_states_fails() {
        let mut proposer = MedusaDraftProposer::new(create_heads());
        let req_id = 1u64;
        let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();

        let mut state = create_state();
        let result = proposer.propose_for_request(req_id, 4, &mut state, &tokenizer);
        assert!(result.is_err(), "should fail without hidden states");
    }

    #[test]
    fn test_on_tokens_verified_is_noop() {
        let mut proposer = MedusaDraftProposer::new(create_heads());
        let req_id = 1u64;

        proposer.init_request(req_id, &[1, 2, 3]).unwrap();
        proposer.on_tokens_verified(req_id, 2, 10).unwrap();
    }

    #[test]
    fn test_multiple_requests() {
        let mut proposer = MedusaDraftProposer::new(create_heads());
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
