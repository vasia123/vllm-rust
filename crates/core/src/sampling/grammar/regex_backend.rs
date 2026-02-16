//! DFA-compiled regex grammar backend.
//!
//! Compiles regex patterns to `regex_automata::dfa::dense::DFA`,
//! precomputes per-state allowed-token bitmasks, and provides
//! O(1) bitmask lookup at inference time.

use std::collections::HashMap;
use std::sync::Arc;

use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use regex_automata::Anchored;

use super::bitmask::PackedBitmask;
use super::vocabulary::VocabularyIndex;
use super::StructuredOutputGrammar;

/// DFA-based regex grammar with precomputed per-state token bitmasks.
///
/// Construction:
/// 1. Compile pattern to dense DFA
/// 2. BFS reachable states from start
/// 3. For each state, walk every token's byte sequence through the DFA
/// 4. If non-dead result → token allowed; store in packed bitmask row
///
/// At inference time, `fill_bitmask` copies the precomputed row for
/// the current state — O(vocab_size/32) words.
pub struct RegexDfaGrammar {
    dfa: DFA<Vec<u32>>,
    state_to_bitmask: HashMap<StateID, Vec<i32>>,
    current_state: StateID,
    state_history: Vec<StateID>,
    vocab_index: Arc<VocabularyIndex>,
}

impl std::fmt::Debug for RegexDfaGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexDfaGrammar")
            .field("num_states_cached", &self.state_to_bitmask.len())
            .field("current_state", &self.current_state)
            .field("history_len", &self.state_history.len())
            .finish()
    }
}

impl RegexDfaGrammar {
    /// Compile a regex pattern into a DFA grammar with precomputed bitmasks.
    ///
    /// The pattern is automatically anchored to match the full output.
    pub fn new(pattern: &str, vocab_index: Arc<VocabularyIndex>) -> anyhow::Result<Self> {
        let anchored_pattern = format!("^(?:{})$", pattern);

        let dfa = DFA::builder()
            .configure(
                DFA::config()
                    .minimize(false)
                    .start_kind(regex_automata::dfa::StartKind::Anchored),
            )
            .build(&anchored_pattern)
            .map_err(|e| anyhow::anyhow!("DFA compilation failed: {e}"))?;

        let vocab_size = vocab_index.vocab_size();
        let words_per_row = vocab_size.div_ceil(32);

        let start_config = regex_automata::util::start::Config::new().anchored(Anchored::Yes);

        let start_state = dfa
            .start_state(&start_config)
            .map_err(|e| anyhow::anyhow!("DFA start state error: {e}"))?;

        // BFS to find all reachable states and precompute bitmasks
        let state_to_bitmask =
            Self::precompute_bitmasks(&dfa, start_state, &vocab_index, words_per_row);

        Ok(Self {
            dfa,
            state_to_bitmask,
            current_state: start_state,
            state_history: Vec::new(),
            vocab_index,
        })
    }

    /// BFS from start state, computing allowed-token bitmask for each reachable state.
    fn precompute_bitmasks(
        dfa: &DFA<Vec<u32>>,
        start_state: StateID,
        vocab_index: &VocabularyIndex,
        words_per_row: usize,
    ) -> HashMap<StateID, Vec<i32>> {
        use std::collections::VecDeque;

        let mut visited: HashMap<StateID, Vec<i32>> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(start_state);
        visited.insert(start_state, vec![0i32; words_per_row]);

        while let Some(state) = queue.pop_front() {
            let mut bitmask_row = vec![0i32; words_per_row];

            for (token_id, token_bytes) in vocab_index.iter() {
                if token_bytes.is_empty() {
                    continue;
                }

                // Walk the DFA through this token's byte sequence
                let mut s = state;
                let mut valid = true;
                for &byte in token_bytes {
                    s = dfa.next_state(s, byte);
                    if dfa.is_dead_state(s) {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    // Token is allowed at this state
                    let word_idx = token_id as usize / 32;
                    let bit_idx = token_id as usize % 32;
                    bitmask_row[word_idx] |= 1i32 << bit_idx;

                    // Enqueue the resulting state if not visited
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(s) {
                        e.insert(vec![0i32; words_per_row]);
                        queue.push_back(s);
                    }
                }
            }

            visited.insert(state, bitmask_row);
        }

        visited
    }

    /// Walk the DFA through a byte sequence from current state, returning the result state.
    fn walk_bytes(&self, from: StateID, bytes: &[u8]) -> Option<StateID> {
        let mut state = from;
        for &byte in bytes {
            state = self.dfa.next_state(state, byte);
            if self.dfa.is_dead_state(state) {
                return None;
            }
        }
        Some(state)
    }
}

impl StructuredOutputGrammar for RegexDfaGrammar {
    fn accept_tokens(&mut self, tokens: &[u32]) -> bool {
        for &token_id in tokens {
            let token_bytes = self.vocab_index.token_bytes(token_id);
            if token_bytes.is_empty() {
                return false;
            }

            match self.walk_bytes(self.current_state, token_bytes) {
                Some(new_state) => {
                    self.state_history.push(self.current_state);
                    self.current_state = new_state;
                }
                None => return false,
            }
        }
        true
    }

    fn fill_bitmask(&self, bitmask: &mut PackedBitmask, batch_index: usize) {
        if let Some(row) = self.state_to_bitmask.get(&self.current_state) {
            bitmask.copy_row_from(batch_index, row);
        }
        // If state not found (shouldn't happen after proper precomputation),
        // bitmask stays zero (all tokens disallowed)
    }

    fn rollback(&mut self, num_tokens: usize) {
        for _ in 0..num_tokens {
            if let Some(prev_state) = self.state_history.pop() {
                self.current_state = prev_state;
            }
        }
    }

    fn is_terminated(&self) -> bool {
        // Check if current state is a match state after EOI
        let eoi_state = self.dfa.next_eoi_state(self.current_state);
        self.dfa.is_match_state(eoi_state)
    }

    fn reset(&mut self) {
        let start_config = regex_automata::util::start::Config::new().anchored(Anchored::Yes);
        if let Ok(start) = self.dfa.start_state(&start_config) {
            self.current_state = start;
        }
        self.state_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenizerWrapper;

    fn make_vocab_index(vocab_size: usize) -> Arc<VocabularyIndex> {
        let tokenizer = TokenizerWrapper::for_testing(vocab_size);
        Arc::new(VocabularyIndex::from_tokenizer(&tokenizer))
    }

    #[test]
    fn simple_pattern_compiles() {
        let vi = make_vocab_index(100);
        let grammar = RegexDfaGrammar::new("[a-z]+", vi);
        assert!(grammar.is_ok());
    }

    #[test]
    fn invalid_pattern_returns_error() {
        let vi = make_vocab_index(100);
        let result = RegexDfaGrammar::new("[invalid", vi);
        assert!(result.is_err());
    }

    #[test]
    fn is_terminated_at_start_for_star_pattern() {
        let vi = make_vocab_index(10);
        // Pattern that accepts empty string
        let grammar = RegexDfaGrammar::new("[a-z]*", vi).unwrap();
        assert!(grammar.is_terminated(), "empty string matches [a-z]*");
    }

    #[test]
    fn is_not_terminated_at_start_for_plus_pattern() {
        let vi = make_vocab_index(10);
        let grammar = RegexDfaGrammar::new("[a-z]+", vi).unwrap();
        assert!(
            !grammar.is_terminated(),
            "empty string does not match [a-z]+"
        );
    }

    #[test]
    fn rollback_restores_state() {
        let vi = make_vocab_index(100);
        let mut grammar = RegexDfaGrammar::new(".*", vi).unwrap();

        let initial_state = grammar.current_state;
        // Accept some token
        grammar.accept_tokens(&[0]);
        let after_one = grammar.current_state;

        grammar.accept_tokens(&[1]);
        // Rollback one step
        grammar.rollback(1);
        assert_eq!(grammar.current_state, after_one);

        // Rollback again
        grammar.rollback(1);
        assert_eq!(grammar.current_state, initial_state);
    }

    #[test]
    fn reset_goes_to_initial() {
        let vi = make_vocab_index(100);
        let mut grammar = RegexDfaGrammar::new(".*", vi).unwrap();

        let initial_state = grammar.current_state;
        grammar.accept_tokens(&[0, 1, 2]);
        assert_ne!(grammar.state_history.len(), 0);

        grammar.reset();
        assert_eq!(grammar.current_state, initial_state);
        assert!(grammar.state_history.is_empty());
    }

    #[test]
    fn fill_bitmask_produces_nonzero_for_wildcard() {
        let vi = make_vocab_index(10);
        let grammar = RegexDfaGrammar::new(".*", vi).unwrap();

        let mut bitmask = PackedBitmask::new(1, 10);
        grammar.fill_bitmask(&mut bitmask, 0);

        // Wildcard should allow at least some tokens
        let mut any_allowed = false;
        for token in 0..10 {
            if bitmask.get_bit(0, token) {
                any_allowed = true;
                break;
            }
        }
        assert!(any_allowed, "wildcard pattern should allow some tokens");
    }
}
