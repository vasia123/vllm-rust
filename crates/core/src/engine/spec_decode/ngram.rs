//! N-gram based speculative token proposer.
//!
//! Finds the longest suffix of the current token sequence that matches a
//! substring earlier in the same sequence, then proposes the tokens that
//! followed that earlier occurrence. This is a pure CPU operation requiring
//! zero VRAM.
//!
//! The algorithm uses a KMP-style failure function on the reversed token
//! sequence. Reversing transforms the suffix-matching problem into a
//! prefix-matching problem, which the KMP failure function solves directly.
//!
//! Reference: vLLM `vllm/v1/spec_decode/ngram_proposer.py`

use super::SpeculativeProposer;

/// Configuration for the n-gram proposer.
#[derive(Debug, Clone)]
pub struct NGramConfig {
    /// Minimum n-gram length to match (inclusive).
    pub min_n: usize,
    /// Maximum n-gram length to match (inclusive).
    pub max_n: usize,
    /// Number of speculative tokens to propose (K).
    pub num_speculative_tokens: usize,
}

impl Default for NGramConfig {
    fn default() -> Self {
        Self {
            min_n: 1,
            max_n: 5,
            num_speculative_tokens: 5,
        }
    }
}

/// N-gram based speculative token proposer.
///
/// Examines the token sequence for repeated n-gram patterns and proposes
/// continuations based on what followed previous occurrences of the same
/// pattern.
///
/// # Algorithm
///
/// 1. Reverse the token sequence so the suffix becomes a prefix.
/// 2. Compute a bounded KMP failure function (LPS array) capped at `max_n`.
/// 3. Track the longest prefix match and its position.
/// 4. If the longest match is >= `min_n`, extract up to K tokens following
///    the match position in the original (non-reversed) sequence.
///
/// This finds the *earliest* occurrence of the longest matching n-gram,
/// which maximizes the number of tokens available after the match.
#[derive(Debug)]
pub struct NGramProposer {
    config: NGramConfig,
}

impl NGramProposer {
    /// Create a new n-gram proposer with the given configuration.
    pub fn new(config: NGramConfig) -> Self {
        Self { config }
    }

    /// Create a proposer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(NGramConfig::default())
    }
}

impl SpeculativeProposer for NGramProposer {
    fn propose(&self, token_ids: &[u32], max_tokens: usize) -> Vec<u32> {
        find_ngram_proposals(
            token_ids,
            self.config.min_n,
            self.config.max_n,
            max_tokens.min(self.config.num_speculative_tokens),
        )
    }

    fn name(&self) -> &str {
        "ngram"
    }
}

/// Core n-gram matching algorithm.
///
/// Finds the longest suffix of `token_ids` (with length in `[min_n, max_n]`)
/// that also appears earlier in the sequence. Returns the tokens that follow
/// that earlier occurrence, up to `k` tokens.
///
/// Returns an empty vec if no match is found or the sequence is too short.
fn find_ngram_proposals(token_ids: &[u32], min_n: usize, max_n: usize, k: usize) -> Vec<u32> {
    let total = token_ids.len();

    if total < min_n || k == 0 {
        return Vec::new();
    }

    // Reverse the token sequence: suffix matching becomes prefix matching.
    // We work with indices into token_ids rather than copying.
    // reversed[i] = token_ids[total - 1 - i]
    let reversed = |i: usize| -> u32 { token_ids[total - 1 - i] };

    // KMP failure function (LPS) bounded to max_n entries.
    // lps[i] = length of the longest proper prefix of reversed[0..=i]
    //          that is also a suffix of reversed[0..=i].
    let mut lps = vec![0usize; max_n];

    let mut longest_ngram: usize = 0;
    let mut best_position: usize = 0;

    // Build LPS and track longest match. Start at i=1 (lps[0] is always 0).
    let mut prev_lps: usize = 0;
    let mut i: usize = 1;

    while i < total {
        if reversed(prev_lps) == reversed(i) {
            prev_lps += 1;

            // Track longest match. Use >= so we get the latest position in
            // the reversed sequence, which corresponds to the earliest
            // occurrence in the original sequence.
            if prev_lps >= longest_ngram {
                longest_ngram = prev_lps;
                best_position = i;
            }

            if i < max_n {
                lps[i] = prev_lps;
            }

            if prev_lps == max_n {
                // Cap at max_n: fall back to avoid matching longer than max_n.
                prev_lps = lps[max_n - 1];
            }

            i += 1;
        } else if prev_lps != 0 {
            prev_lps = lps[prev_lps - 1];
        } else {
            i += 1;
        }
    }

    if longest_ngram < min_n {
        return Vec::new();
    }

    // Convert position back to the original (non-reversed) sequence.
    // In the reversed sequence, the match ends at `best_position`.
    // In the original sequence, the matched n-gram starts at:
    //   total - 1 - best_position
    // and the tokens to propose start at:
    //   total - 1 - best_position + longest_ngram
    let start = total - 1 - best_position + longest_ngram;
    let available = total - start;
    let take = k.min(available);

    token_ids[start..start + take].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── find_ngram_proposals unit tests ──────────────────────────────────

    #[test]
    fn empty_sequence_returns_empty() {
        assert!(find_ngram_proposals(&[], 1, 3, 5).is_empty());
    }

    #[test]
    fn sequence_shorter_than_min_n_returns_empty() {
        assert!(find_ngram_proposals(&[1, 2], 3, 5, 5).is_empty());
    }

    #[test]
    fn zero_k_returns_empty() {
        assert!(find_ngram_proposals(&[1, 2, 3, 1, 2, 3], 1, 3, 0).is_empty());
    }

    #[test]
    fn no_match_returns_empty() {
        // All unique tokens, no repeating patterns.
        assert!(find_ngram_proposals(&[1, 2, 3, 4, 5], 2, 2, 2).is_empty());
    }

    #[test]
    fn suffix_matches_but_trailing_tokens_differ() {
        // Suffix "1, 2, 3" matches the prefix, but the tokens after differ.
        // This is the test from vLLM: [1, 2, 3, 4, 1, 2, 3, 5, 6]
        // With min_n=2, max_n=2: the suffix [2,3] matches at position 1..3,
        // but the tokens after [2,3] at pos 1 are [4,1,2,3,5,6] so we'd
        // propose starting from after position 3 which is token 4.
        // Wait, the suffix is [5, 6] not [2, 3]. Let me re-check.
        // token_ids = [1, 2, 3, 4, 1, 2, 3, 5, 6]
        // The suffix of length 2 is [5, 6]. This doesn't appear earlier.
        // The suffix of length 1 is [6]. This doesn't appear earlier either.
        // So no match -> empty. This matches the vLLM test.
        let tokens = [1u32, 2, 3, 4, 1, 2, 3, 5, 6];
        assert!(find_ngram_proposals(&tokens, 2, 2, 2).is_empty());
    }

    #[test]
    fn basic_2gram_match() {
        // [1, 2, 3, 4, 1, 2, 3] -> suffix [2, 3] matches at positions 1..3.
        // Tokens after the match: [4, 1, 2] (up to 3 tokens).
        let tokens = [1u32, 2, 3, 4, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 2, 2, 3), vec![4, 1, 2]);
        assert_eq!(find_ngram_proposals(&tokens, 2, 2, 2), vec![4, 1]);
    }

    #[test]
    fn unigram_match() {
        // [1, 2, 3, 4, 1, 2, 3] with min_n=1, max_n=1, k=3.
        // The suffix of length 1 is [3]. First occurrence of 3 is at index 2.
        // Tokens after index 2: [4, 1, 2, 3]. Take up to 3 -> [4, 1, 2].
        let tokens = [1u32, 2, 3, 4, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 1, 1, 3), vec![4, 1, 2]);
        assert_eq!(find_ngram_proposals(&tokens, 1, 1, 2), vec![4, 1]);
    }

    #[test]
    fn longer_context_2gram_match() {
        // [1, 3, 6, 2, 3, 4, 1, 2, 3] with 2-gram.
        // Suffix [2, 3] appears at positions 3..5. After: [4, 1, 2, 3].
        let tokens = [1u32, 3, 6, 2, 3, 4, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 2, 2, 3), vec![4, 1, 2]);
    }

    #[test]
    fn unigram_picks_earliest_occurrence() {
        // [1, 3, 6, 2, 3, 4, 1, 2, 3] with 1-gram.
        // Suffix [3] first appears at index 1 (value 3).
        // Tokens after index 1: [6, 2, 3, 4, 1, 2, 3]. Take 2 -> [6, 2].
        let tokens = [1u32, 3, 6, 2, 3, 4, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 1, 1, 2), vec![6, 2]);
    }

    #[test]
    fn no_match_for_4gram_but_match_for_3gram() {
        // [1, 2, 3, 4, 1, 2, 3] with min_n=3, max_n=4.
        // 4-gram suffix [4, 1, 2, 3] doesn't appear earlier.
        // 3-gram suffix [1, 2, 3] matches at position 0..3. After: [4, 1].
        let tokens = [1u32, 2, 3, 4, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 3, 4, 2), vec![4, 1]);
    }

    #[test]
    fn match_for_both_4gram_and_3gram_picks_longest() {
        // [2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4] with min_n=3, max_n=4.
        // 4-gram suffix [1, 2, 3, 4] matches at position 4..8. After: [1, 2].
        // 3-gram suffix [2, 3, 4] also matches. But 4-gram is longer, so it wins.
        let tokens = [2u32, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(find_ngram_proposals(&tokens, 3, 4, 2), vec![1, 2]);
    }

    #[test]
    fn match_3gram_not_4gram() {
        // [3, 4, 5, 2, 3, 4, 1, 2, 3, 4] with min_n=2, max_n=4.
        // 4-gram suffix [2, 3, 4] doesn't extend to 4.
        // Actually the suffix of length 4 is [1, 2, 3, 4]. Does it appear earlier?
        // No, 1 first appears at index 6.
        // 3-gram suffix [2, 3, 4] matches at positions 3..6. After: [1, 2].
        let tokens = [3u32, 4, 5, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(find_ngram_proposals(&tokens, 2, 4, 2), vec![1, 2]);
    }

    #[test]
    fn multiple_matches_picks_earliest() {
        // [1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3] with 3-gram.
        // Suffix [1, 2, 3] appears at indices 0, 4, 8. Earliest is 0.
        // After index 0+3=3: [100, 1, 2, 3, 200, ...]. Take 2 -> [100, 1].
        let tokens = [1u32, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 3, 3, 2), vec![100, 1]);
    }

    #[test]
    fn k_exceeds_available_tokens() {
        // [1, 2, 1, 2] with k=10. Suffix [1, 2] matches at 0..2.
        // After position 2: [1, 2]. Only 2 tokens available.
        let tokens = [1u32, 2, 1, 2];
        assert_eq!(find_ngram_proposals(&tokens, 2, 2, 10), vec![1, 2]);
    }

    #[test]
    fn single_repeated_token() {
        // [5, 5, 5, 5, 5] with 1-gram.
        // Suffix [5] first appears at index 0. After: [5, 5, 5, 5]. Take 3.
        let tokens = [5u32, 5, 5, 5, 5];
        assert_eq!(find_ngram_proposals(&tokens, 1, 1, 3), vec![5, 5, 5]);
    }

    #[test]
    fn two_tokens_with_1gram() {
        // [7, 7] with 1-gram.
        // Suffix [7] appears at index 0. After index 1: [7]. Take 1.
        let tokens = [7u32, 7];
        assert_eq!(find_ngram_proposals(&tokens, 1, 1, 5), vec![7]);
    }

    #[test]
    fn min_n_equals_max_n_no_match() {
        // [1, 2, 3, 4, 5] with n=3. No 3-gram suffix repeats.
        assert!(find_ngram_proposals(&[1, 2, 3, 4, 5], 3, 3, 5).is_empty());
    }

    // ─── NGramProposer trait tests ────────────────────────────────────────

    #[test]
    fn proposer_trait_name() {
        let proposer = NGramProposer::with_defaults();
        assert_eq!(proposer.name(), "ngram");
    }

    #[test]
    fn proposer_respects_max_tokens() {
        let proposer = NGramProposer::new(NGramConfig {
            min_n: 2,
            max_n: 2,
            num_speculative_tokens: 10,
        });
        let tokens = [1u32, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2];
        // Suffix [1, 2] matches at index 0. Plenty of tokens after.
        // max_tokens = 2 should cap the output.
        let result = proposer.propose(&tokens, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn proposer_num_speculative_tokens_caps_output() {
        let proposer = NGramProposer::new(NGramConfig {
            min_n: 2,
            max_n: 2,
            num_speculative_tokens: 1,
        });
        let tokens = [1u32, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2];
        // Even though max_tokens is large, num_speculative_tokens=1 caps it.
        let result = proposer.propose(&tokens, 100);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn proposer_with_defaults_works() {
        let proposer = NGramProposer::with_defaults();
        // Default: min_n=1, max_n=5, num_speculative_tokens=5
        let tokens = [10u32, 20, 30, 10, 20, 30];
        let result = proposer.propose(&tokens, 5);
        // 3-gram [10, 20, 30] matches. After index 3: [10, 20, 30]. Take min(5, 3) = 3.
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn proposer_on_tokens_accepted_is_noop() {
        let mut proposer = NGramProposer::with_defaults();
        // Should not panic.
        proposer.on_tokens_accepted(42, &[1, 2, 3]);
    }

    #[test]
    fn proposer_on_request_finished_is_noop() {
        let mut proposer = NGramProposer::with_defaults();
        // Should not panic.
        proposer.on_request_finished(42);
    }

    // ─── Regression tests matching vLLM test suite ────────────────────────

    #[test]
    fn vllm_test_no_match() {
        let tokens = [1u32, 2, 3, 4, 5];
        assert!(find_ngram_proposals(&tokens, 2, 2, 2).is_empty());
    }

    #[test]
    fn vllm_test_no_match_4gram() {
        let tokens = [1u32, 2, 3, 4, 1, 2, 3];
        assert!(find_ngram_proposals(&tokens, 4, 4, 2).is_empty());
    }

    #[test]
    fn vllm_test_no_4gram_but_3gram_match() {
        let tokens = [1u32, 2, 3, 4, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 3, 4, 2), vec![4, 1]);
    }

    #[test]
    fn vllm_test_both_4gram_and_3gram_picks_4gram() {
        let tokens = [2u32, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(find_ngram_proposals(&tokens, 3, 4, 2), vec![1, 2]);
    }

    #[test]
    fn vllm_test_2gram_and_3gram_not_4gram() {
        let tokens = [3u32, 4, 5, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(find_ngram_proposals(&tokens, 2, 4, 2), vec![1, 2]);
    }

    #[test]
    fn vllm_test_multiple_3gram_picks_earliest() {
        let tokens = [1u32, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3];
        assert_eq!(find_ngram_proposals(&tokens, 3, 3, 2), vec![100, 1]);
    }
}
