//! Speculative decoding proposer abstractions.
//!
//! This module defines the [`SpeculativeProposer`] trait and implementations
//! for different token proposal strategies:
//!
//! - [`NGramProposer`]: CPU-based n-gram suffix matching (zero VRAM)
//! - [`DraftModelProposer`]: Wraps a smaller draft model for greedy proposals
//!
//! The trait decouples the *proposal* logic from the *verification* logic in
//! the engine, allowing different proposers to be plugged in without changing
//! the speculative decoding execution loop.

mod draft_model;
pub mod eagle;
pub mod medusa;
mod ngram;
mod suffix;
pub mod tree_attention;

pub use draft_model::DraftModelProposer;
pub use eagle::{EagleConfig, EagleProposer};
pub use medusa::{MedusaHead, MedusaProposer};
pub use ngram::{NGramConfig, NGramProposer};
pub use suffix::{SuffixArrayConfig, SuffixArrayProposer};
pub use tree_attention::SpeculationTree;

/// Trait for speculative token proposers.
///
/// A proposer examines the tokens generated so far and proposes candidate
/// continuations. These candidates are then verified by the target model
/// in a single batched forward pass.
///
/// Proposers must be `Send + Sync` to support concurrent engine access.
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
