//! Grammar-based structured output with DFA-compiled bitmask constraints.
//!
//! This module provides high-performance constrained generation using
//! pre-compiled automaton states and packed bitmasks. Supported backends:
//! - Regex patterns compiled to DFA with per-state token bitmasks
//! - JSON Schema converted to regex then compiled to DFA
//! - EBNF/GBNF grammars (regular → DFA, recursive → pushdown automaton)

pub mod bitmask;
pub mod compiler;
pub mod vocabulary;

// Structured-output compilation. Removed the partial native Rust
// port (ebnf_backend / ebnf_parser / json_schema / regex_backend)
// in v1.8 — it never enforced JSON-Schema's `pattern + length` or
// strict-object key boundaries correctly (Bug 10 root cause).
// Production now goes through the xgrammar FFI bridge; the
// `xgrammar` feature gate stays so CPU dev / CI builds without
// CUDA can opt out of the C++ compile (at the cost of no
// structured output support in that configuration).
#[cfg(feature = "xgrammar")]
pub mod schema_to_ebnf;
#[cfg(feature = "xgrammar")]
pub mod xgrammar_backend;

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

    /// Expose the underlying xgrammar matcher for batched parallel
    /// bitmask fill (`xgrammar_rs::BatchMatcher`). Returns `None` for
    /// non-xgrammar backends. The engine locks several of these at once
    /// to dispatch one thread-pooled fill across a constrained batch.
    #[cfg(feature = "xgrammar")]
    fn xgrammar_matcher(&self) -> Option<&std::sync::Mutex<xgrammar_rs::GrammarMatcher>> {
        None
    }
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
        // Padded-vocab tail mask. The grammar bitmask is sized to
        // the tokenizer's `vocab_size` (e.g. Qwen3 = 151669), but
        // the actual logits tensor often pads up to the next
        // multiple supported by the matmul (Qwen3 lm_head emits
        // 151936). Tokens in `[vocab_size, logits.len())` are
        // typically reserved/unused padding slots — the bitmask
        // doesn't touch them, so without an explicit -inf the
        // sampler can still draw one and produce undefined bytes.
        if logits.len() > self.vocab_size {
            for logit in &mut logits[self.vocab_size..] {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn is_complete(&self, _generated: &[u32], _text: &str) -> bool {
        self.grammar.is_terminated()
    }

    fn reset(&mut self) {
        self.grammar.reset();
    }

    fn accept_token(&mut self, token_id: u32) -> bool {
        // Advance the underlying xgrammar matcher so the next
        // `fill_bitmask` reflects the post-`token_id` grammar state.
        // Without this hook the matcher stays at position 0 for the
        // entire stream and only the first token gets enforced.
        self.grammar.accept_tokens(&[token_id])
    }

    fn supports_gpu(&self) -> bool {
        // Adapter wraps an xgrammar matcher which produces packed-i32
        // bitmask rows by construction.
        true
    }

    fn fill_cpu_bitmask_for_gpu(&mut self, bitmask_row: &mut [i32]) -> Option<anyhow::Result<()>> {
        // Grammar-side bitmask covers `[0, self.vocab_size)`. Engine
        // pre-allocates a row sized to the LOGITS vocab (may be
        // larger due to lm_head padding) and is responsible for
        // zero-initialising the whole row before this call — that
        // way any tail bits past `grammar_words` stay zero and the
        // GPU apply kernel masks padded tokens to -inf automatically.
        if self.grammar.is_terminated() {
            // Terminated: forbid all tokens. Engine row is already
            // zero, so we just leave it untouched.
            for w in bitmask_row.iter_mut() {
                *w = 0;
            }
            return Some(Ok(()));
        }
        let grammar_words = self.vocab_size.div_ceil(32);
        if bitmask_row.len() < grammar_words {
            return Some(Err(anyhow::anyhow!(
                "fill_cpu_bitmask_for_gpu: row len {} < required {}",
                bitmask_row.len(),
                grammar_words
            )));
        }
        // Re-zero the adapter's scratch row; xgrammar will then
        // OR-in the allowed-token bits.
        self.bitmask_buf.set_all_zeros();
        self.grammar.fill_bitmask(&mut self.bitmask_buf, 0);
        let src = self.bitmask_buf.row(0);
        bitmask_row[..grammar_words].copy_from_slice(&src[..grammar_words]);
        Some(Ok(()))
    }

    #[cfg(feature = "xgrammar")]
    fn xgrammar_matcher(&self) -> Option<&std::sync::Mutex<xgrammar_rs::GrammarMatcher>> {
        self.grammar.xgrammar_matcher()
    }

    #[cfg(feature = "xgrammar")]
    fn grammar_bitmask_words(&self) -> Option<usize> {
        // Only meaningful when an xgrammar matcher backs this adapter.
        self.grammar
            .xgrammar_matcher()
            .map(|_| self.vocab_size.div_ceil(32))
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
