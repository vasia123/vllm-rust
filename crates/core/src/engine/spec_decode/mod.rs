//! Speculative decoding proposer abstractions.
//!
//! This module defines two proposer traits:
//!
//! - [`DraftProposer`]: The primary trait with full lifecycle management,
//!   per-request state, and integration with the engine's execution strategy.
//!   Used by [`SpeculativeExecution`] for speculative decoding.
//!
//! - [`SpeculativeProposer`]: Legacy simple trait for standalone usage.
//!   Does not manage per-request state or KV cache lifecycle.
//!
//! `DraftProposer` implementations:
//! - [`DraftModelDraftProposer`]: GPU-based draft model with KV cache lifecycle
//! - [`NGramProposer`]: CPU-based n-gram suffix matching (zero VRAM, stateless)
//! - [`SuffixArrayProposer`]: CPU-based suffix array matching (zero VRAM, stateless)
//!
//! `SpeculativeProposer` (legacy) implementations:
//! - [`DraftModelProposer`]: GPU-based draft model
//! - [`NGramProposer`]: CPU-based n-gram (also implements `DraftProposer`)
//! - [`SuffixArrayProposer`]: CPU-based suffix array (also implements `DraftProposer`)
//! - [`EagleProposer`]: Feature-level autoregressive draft network
//! - [`MedusaProposer`]: Multiple independent prediction heads

mod draft_model;
mod draft_proposer;
pub mod eagle;
pub mod medusa;
mod ngram;
mod suffix;
pub mod tree_attention;

pub use draft_model::DraftModelProposer;
pub use draft_proposer::DraftModelDraftProposer;
pub use eagle::{EagleConfig, EagleProposer};
pub use medusa::{MedusaHead, MedusaProposer};
pub use ngram::{NGramConfig, NGramProposer};
pub use suffix::{SuffixArrayConfig, SuffixArrayProposer};
pub use tree_attention::SpeculationTree;

use crate::engine::types::EngineError;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

/// Trait for speculative token proposers with full lifecycle management.
///
/// Unlike [`SpeculativeProposer`], this trait:
/// - Manages per-request state internally (e.g., draft KV cache, block tables)
/// - Has lifecycle hooks for request setup, teardown, and preemption
/// - Receives full [`SequenceState`] for applying penalties and constraints
/// - Returns `Result` for proper error propagation
/// - Uses `&mut self` (no interior mutability needed since the engine loop
///   is single-threaded)
///
/// The engine's `SpeculativeExecution` strategy delegates draft token
/// generation to a `DraftProposer`, keeping only verification logic.
pub trait DraftProposer: Send {
    /// Initialize per-request state (e.g., prefill the draft model's KV cache).
    ///
    /// Called once when a request first enters the decode phase after prefill.
    fn init_request(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<(), EngineError>;

    /// Generate draft tokens for a single request.
    ///
    /// Returns up to `num_speculative_tokens()` draft token IDs. The proposer
    /// should advance its internal state (e.g., draft KV cache position) as if
    /// the draft tokens were accepted. If fewer tokens are accepted during
    /// verification, `on_tokens_verified` will be called to roll back.
    ///
    /// `state` provides the sequence's sampling params, generated tokens, and
    /// constraint for applying penalties during draft generation.
    fn propose_for_request(
        &mut self,
        request_id: RequestId,
        last_token: u32,
        state: &mut SequenceState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<Vec<u32>, EngineError>;

    /// Notify the proposer of verification results.
    ///
    /// The proposer should trim its KV cache to `original_offset + num_accepted`
    /// and free any excess blocks. This is called after the target model verifies
    /// the draft tokens.
    fn on_tokens_verified(
        &mut self,
        request_id: RequestId,
        num_accepted: usize,
        original_offset: usize,
    ) -> Result<(), EngineError>;

    /// Clean up all state for a completed request (free KV cache blocks, etc.).
    fn finish_request(&mut self, request_id: RequestId) -> Result<(), EngineError>;

    /// Handle preemption: free all resources for the request and reset state.
    fn preempt_request(&mut self, request_id: RequestId) -> Result<(), EngineError>;

    /// Number of speculative tokens this proposer targets per step.
    fn num_speculative_tokens(&self) -> usize;

    /// Name for logging and diagnostics.
    fn name(&self) -> &str;
}

/// Legacy trait for speculative token proposers.
///
/// A proposer examines the tokens generated so far and proposes candidate
/// continuations. These candidates are then verified by the target model
/// in a single batched forward pass.
///
/// Proposers must be `Send + Sync` to support concurrent engine access.
///
/// For new implementations, prefer [`DraftProposer`] which provides full
/// lifecycle management and better integration with the engine.
pub trait SpeculativeProposer: Send + Sync {
    /// Propose speculative tokens based on the sequence so far.
    ///
    /// Returns up to `max_tokens` proposed token IDs. An empty vec
    /// indicates no proposals could be made.
    fn propose(&self, token_ids: &[u32], max_tokens: usize) -> Vec<u32>;

    /// Notify the proposer that tokens were accepted for a request.
    ///
    /// Proposers that maintain per-request state (e.g., caches or statistics)
    /// can use this callback to update their internal state.
    fn on_tokens_accepted(&mut self, _request_id: u64, _accepted_tokens: &[u32]) {}

    /// Notify the proposer that a request has finished.
    ///
    /// Proposers that maintain per-request state should clean up here.
    fn on_request_finished(&mut self, _request_id: u64) {}

    /// Name of the proposer for logging and diagnostics.
    fn name(&self) -> &str;
}
