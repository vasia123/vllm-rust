//! Grammar-based structured output with DFA-compiled bitmask constraints.
//!
//! This module provides high-performance constrained generation using
//! pre-compiled automaton states and packed bitmasks. Supported backends:
//! - Regex patterns compiled to DFA with per-state token bitmasks
//! - JSON Schema converted to regex then compiled to DFA
//! - EBNF/GBNF grammars (regular → DFA, recursive → pushdown automaton)

pub mod bitmask;
pub mod compiler;
pub mod ebnf_backend;
pub mod ebnf_parser;
pub mod json_schema;
pub mod regex_backend;
pub mod vocabulary;

pub use bitmask::PackedBitmask;
pub use compiler::GrammarCompiler;
pub use vocabulary::VocabularyIndex;

use crate::sampling::SamplingConstraint;

/// Core trait for grammar-based structured output constraints.
///
/// Unlike `SamplingConstraint` which works with decoded text,
/// `StructuredOutputGrammar` operates on token IDs and produces
/// packed bitmasks for O(vocab_size/32) logit masking.
pub trait StructuredOutputGrammar: Send + Sync {
    /// Feed accepted tokens into the grammar state machine.
    ///
    /// Returns `true` if all tokens were accepted (valid transitions),
    /// `false` if any token caused an invalid transition.
    fn accept_tokens(&mut self, tokens: &[u32]) -> bool;

    /// Fill the bitmask row for `batch_index` with allowed tokens
    /// given the current grammar state.
    fn fill_bitmask(&self, bitmask: &mut PackedBitmask, batch_index: usize);

    /// Roll back `num_tokens` from the state history.
    ///
    /// Used by speculative decoding to undo rejected draft tokens.
    fn rollback(&mut self, num_tokens: usize);

    /// Whether the grammar has reached a terminal/accepting state.
    fn is_terminated(&self) -> bool;

    /// Reset the grammar to its initial state.
    fn reset(&mut self);
}

/// Adapter that wraps a `StructuredOutputGrammar` as a `SamplingConstraint`.
///
/// Bridges the new grammar system into the existing sampling pipeline,
/// requiring zero changes to the 4 engine call sites.
pub struct GrammarConstraintAdapter {
    grammar: Box<dyn StructuredOutputGrammar>,
    vocab_size: usize,
    /// Reusable bitmask buffer (single-row, allocated once).
    bitmask_buf: PackedBitmask,
}

impl std::fmt::Debug for GrammarConstraintAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarConstraintAdapter")
            .field("vocab_size", &self.vocab_size)
            .field("is_terminated", &self.grammar.is_terminated())
            .finish()
    }
}

impl GrammarConstraintAdapter {
    pub fn new(grammar: Box<dyn StructuredOutputGrammar>, vocab_size: usize) -> Self {
        let bitmask_buf = PackedBitmask::new(1, vocab_size);
        Self {
            grammar,
            vocab_size,
            bitmask_buf,
        }
    }
}

impl SamplingConstraint for GrammarConstraintAdapter {
    fn mask_logits(&mut self, logits: &mut [f32], _generated_text: &str) -> anyhow::Result<()> {
        if self.grammar.is_terminated() {
            for logit in logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            return Ok(());
        }

        self.bitmask_buf.set_all_zeros();
        self.grammar.fill_bitmask(&mut self.bitmask_buf, 0);
        self.bitmask_buf.apply_to_logits(logits, 0);
        Ok(())
    }

    fn is_complete(&self, _generated: &[u32], _text: &str) -> bool {
        self.grammar.is_terminated()
    }

    fn reset(&mut self) {
        self.grammar.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Minimal mock grammar for testing the adapter.
    struct MockGrammar {
        terminated: Arc<AtomicBool>,
        allowed_tokens: Vec<u32>,
        accepted_count: usize,
    }

    impl MockGrammar {
        fn new(allowed_tokens: Vec<u32>) -> Self {
            Self {
                terminated: Arc::new(AtomicBool::new(false)),
                allowed_tokens,
                accepted_count: 0,
            }
        }
    }

    impl StructuredOutputGrammar for MockGrammar {
        fn accept_tokens(&mut self, tokens: &[u32]) -> bool {
            self.accepted_count += tokens.len();
            true
        }

        fn fill_bitmask(&self, bitmask: &mut PackedBitmask, batch_index: usize) {
            for &token_id in &self.allowed_tokens {
                bitmask.set_bit(batch_index, token_id as usize);
            }
        }

        fn rollback(&mut self, num_tokens: usize) {
            self.accepted_count = self.accepted_count.saturating_sub(num_tokens);
        }

        fn is_terminated(&self) -> bool {
            self.terminated.load(Ordering::Relaxed)
        }

        fn reset(&mut self) {
            self.accepted_count = 0;
            self.terminated.store(false, Ordering::Relaxed);
        }
    }

    #[test]
    fn adapter_masks_disallowed_tokens() {
        let grammar = MockGrammar::new(vec![2, 5, 7]);
        let mut adapter = GrammarConstraintAdapter::new(Box::new(grammar), 10);

        let mut logits = vec![1.0f32; 10];
        adapter.mask_logits(&mut logits, "").unwrap();

        // Only tokens 2, 5, 7 should remain finite
        for (i, &l) in logits.iter().enumerate() {
            if i == 2 || i == 5 || i == 7 {
                assert!(l.is_finite(), "token {i} should be allowed");
            } else {
                assert!(
                    l == f32::NEG_INFINITY,
                    "token {i} should be masked, got {l}"
                );
            }
        }
    }

    #[test]
    fn adapter_masks_all_when_terminated() {
        let grammar = MockGrammar::new(vec![1, 2, 3]);
        let terminated = grammar.terminated.clone();
        let mut adapter = GrammarConstraintAdapter::new(Box::new(grammar), 10);

        terminated.store(true, Ordering::Relaxed);

        let mut logits = vec![1.0f32; 10];
        adapter.mask_logits(&mut logits, "").unwrap();

        for &l in &logits {
            assert!(l == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn adapter_is_complete_reflects_grammar() {
        let grammar = MockGrammar::new(vec![]);
        let terminated = grammar.terminated.clone();
        let adapter = GrammarConstraintAdapter::new(Box::new(grammar), 10);

        assert!(!adapter.is_complete(&[], ""));
        terminated.store(true, Ordering::Relaxed);
        assert!(adapter.is_complete(&[], ""));
    }

    #[test]
    fn adapter_reset_clears_grammar() {
        let grammar = MockGrammar::new(vec![1]);
        let terminated = grammar.terminated.clone();
        let mut adapter = GrammarConstraintAdapter::new(Box::new(grammar), 10);

        terminated.store(true, Ordering::Relaxed);
        assert!(adapter.is_complete(&[], ""));

        adapter.reset();
        assert!(!adapter.is_complete(&[], ""));
    }
}
