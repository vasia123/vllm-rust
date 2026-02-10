//! Suffix-array based speculative token proposer.
//!
//! Builds a suffix array over the prompt token IDs and uses binary search to
//! find the longest suffix of the recent context that appears earlier in the
//! sequence. The continuation after that match is proposed as speculative tokens.
//!
//! This approach is particularly effective for repetitive/structured text such
//! as code, JSON, or templated output where long repeated substrings are common.
//! The suffix array enables O(log n) lookup per query versus the O(n) scan in
//! the n-gram proposer.
//!
//! The data structure lives entirely on CPU and requires zero VRAM.
//!
//! Reference: SA-IS (Nong et al., 2009) for linear-time construction. We use a
//! simpler O(n log^2 n) approach here since prompt lengths are bounded and
//! construction cost is amortized across many proposal steps.

use crate::engine::types::EngineError;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

use super::{DraftProposer, SpeculativeProposer};

/// Configuration for the suffix array proposer.
#[derive(Debug, Clone)]
pub struct SuffixArrayConfig {
    /// Maximum tokens to propose per step.
    pub max_speculation_length: usize,
    /// Minimum suffix match length to consider a match valid.
    pub min_match_length: usize,
    /// Number of recent tokens to try matching against the suffix array.
    /// Larger values search for longer matches but take more time.
    pub context_window: usize,
}

impl Default for SuffixArrayConfig {
    fn default() -> Self {
        Self {
            max_speculation_length: 5,
            min_match_length: 3,
            context_window: 32,
        }
    }
}

/// Suffix array over a token sequence.
///
/// Stores the sorted suffix indices and an LCP (Longest Common Prefix) array
/// for efficient longest-match queries via binary search.
#[derive(Debug)]
struct SuffixArray {
    /// The original token sequence.
    tokens: Vec<u32>,
    /// Suffix array: `sa[i]` is the starting index of the i-th lexicographically
    /// smallest suffix in `tokens`.
    sa: Vec<usize>,
    /// Inverse suffix array: `rank[i]` is the rank of the suffix starting at
    /// position `i` in the sorted order. Retained as part of the canonical SA
    /// data structure; used by Kasai's algorithm during construction and verified
    /// in tests.
    #[allow(dead_code)]
    rank: Vec<usize>,
    /// LCP array: `lcp[i]` is the length of the longest common prefix between
    /// suffixes `sa[i-1]` and `sa[i]`. `lcp[0]` is 0 by convention. Retained
    /// for future range-minimum LCP queries and verified in tests.
    #[allow(dead_code)]
    lcp: Vec<usize>,
}

impl SuffixArray {
    /// Build a suffix array from a token sequence using O(n log^2 n) construction.
    ///
    /// Uses the prefix-doubling algorithm: start with single-character ranks,
    /// then iteratively double the comparison length, sorting by pairs of ranks
    /// at each step.
    fn build(tokens: &[u32]) -> Self {
        let n = tokens.len();
        if n == 0 {
            return Self {
                tokens: Vec::new(),
                sa: Vec::new(),
                rank: Vec::new(),
                lcp: Vec::new(),
            };
        }

        // Initialize suffix array and rank from single tokens.
        let mut sa: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];
        let mut tmp_rank: Vec<usize> = vec![0; n];

        // Initial ranks from token values. Map to 1-based dense ranks so that
        // 0 can serve as a sentinel for "past end of suffix" in the pair
        // comparisons below. A shorter suffix must sort before a longer one
        // that shares the same prefix.
        {
            let mut sorted_tokens: Vec<u32> = tokens.to_vec();
            sorted_tokens.sort_unstable();
            sorted_tokens.dedup();
            for i in 0..n {
                rank[i] = sorted_tokens
                    .binary_search(&tokens[i])
                    .expect("token must exist in sorted set")
                    + 1; // 1-based
            }
        }

        let mut gap = 1usize;
        while gap < n {
            // Sort suffixes by (rank[i], rank[i + gap]) pairs.
            // Sentinel value 0 for past-end ensures shorter suffixes sort first.
            let current_rank = rank.clone();
            sa.sort_unstable_by(|&a, &b| {
                let ra = current_rank[a];
                let rb = current_rank[b];
                if ra != rb {
                    return ra.cmp(&rb);
                }
                let ra2 = if a + gap < n {
                    current_rank[a + gap]
                } else {
                    0
                };
                let rb2 = if b + gap < n {
                    current_rank[b + gap]
                } else {
                    0
                };
                ra2.cmp(&rb2)
            });

            // Recompute ranks from the sorted order (1-based to preserve sentinel).
            tmp_rank[sa[0]] = 1;
            for i in 1..n {
                let prev = sa[i - 1];
                let curr = sa[i];
                let same_first = current_rank[prev] == current_rank[curr];
                let same_second = {
                    let rp = if prev + gap < n {
                        current_rank[prev + gap]
                    } else {
                        0
                    };
                    let rc = if curr + gap < n {
                        current_rank[curr + gap]
                    } else {
                        0
                    };
                    rp == rc
                };
                tmp_rank[curr] = if same_first && same_second {
                    tmp_rank[sa[i - 1]]
                } else {
                    tmp_rank[sa[i - 1]] + 1
                };
            }
            rank.copy_from_slice(&tmp_rank);

            // Early termination: all ranks are unique.
            if rank[sa[n - 1]] == n {
                break;
            }

            gap *= 2;
        }

        // Build LCP array using Kasai's algorithm: O(n).
        let mut inv = vec![0usize; n];
        for i in 0..n {
            inv[sa[i]] = i;
        }

        let mut lcp = vec![0usize; n];
        let mut k = 0usize;
        for i in 0..n {
            if inv[i] == 0 {
                k = 0;
                continue;
            }
            let j = sa[inv[i] - 1];
            while i + k < n && j + k < n && tokens[i + k] == tokens[j + k] {
                k += 1;
            }
            lcp[inv[i]] = k;
            k = k.saturating_sub(1);
        }

        Self {
            tokens: tokens.to_vec(),
            sa,
            rank: inv, // Store the inverse (position -> rank) for lookups
            lcp,
        }
    }

    /// For a suffix starting at position `suffix_pos`, find the longest common
    /// prefix with any other suffix that starts at a position < `exclude_start`.
    ///
    /// Returns `(match_position, match_length)` where `match_position` is the
    /// starting index in `self.tokens` of the best earlier match, and
    /// `match_length` is the LCP length. Returns `(0, 0)` if no valid match
    /// is found with length >= `min_match_length`.
    ///
    /// Works by looking at the suffix's neighbors in the SA (via the rank/inverse
    /// array) and computing LCP by direct comparison.
    fn find_best_earlier_match(
        &self,
        suffix_pos: usize,
        min_match_length: usize,
        exclude_start: usize,
    ) -> (usize, usize) {
        let n = self.tokens.len();
        if n == 0 || suffix_pos >= n {
            return (0, 0);
        }

        let suffix_rank = self.rank[suffix_pos];

        let mut best_pos = 0usize;
        let mut best_len = 0usize;

        // Check neighbors in the SA going left (rank - 1, rank - 2, ...).
        // Stop when LCP drops below current best (the LCP with the i-th
        // neighbor is at most the LCP with the (i-1)-th neighbor).
        // We use a bounded scan for efficiency.
        let max_scan = 64; // Bound scan distance for performance.

        // Scan left neighbors
        {
            let mut min_lcp = usize::MAX;
            let mut i = suffix_rank;
            let mut scanned = 0;
            while i > 0 && scanned < max_scan {
                i -= 1;
                scanned += 1;
                let neighbor_pos = self.sa[i];
                // The LCP between sa[i] and sa[i+1] is lcp[i+1].
                // For non-adjacent entries, the LCP is the minimum of
                // intervening LCP values.
                min_lcp = min_lcp.min(self.lcp[i + 1]);
                if min_lcp < min_match_length {
                    break;
                }
                if neighbor_pos < exclude_start
                    && (min_lcp > best_len || (min_lcp == best_len && neighbor_pos < best_pos))
                {
                    // Prefer longer matches; for equal length, prefer the
                    // earliest position to maximize continuation room.
                    best_len = min_lcp;
                    best_pos = neighbor_pos;
                }
            }
        }

        // Scan right neighbors
        {
            let mut min_lcp = usize::MAX;
            let mut i = suffix_rank;
            let mut scanned = 0;
            while i + 1 < n && scanned < max_scan {
                i += 1;
                scanned += 1;
                let neighbor_pos = self.sa[i];
                min_lcp = min_lcp.min(self.lcp[i]);
                if min_lcp < min_match_length {
                    break;
                }
                if neighbor_pos < exclude_start
                    && (min_lcp > best_len || (min_lcp == best_len && neighbor_pos < best_pos))
                {
                    best_len = min_lcp;
                    best_pos = neighbor_pos;
                }
            }
        }

        if best_len >= min_match_length {
            (best_pos, best_len)
        } else {
            (0, 0)
        }
    }
}

/// Suffix-array based speculative token proposer.
///
/// Builds a suffix array over the full token sequence on each `propose()` call,
/// then searches for the longest suffix of the recent context that also appears
/// earlier in the sequence. The tokens following that earlier occurrence are
/// proposed as speculative continuations.
///
/// # When to use
///
/// Best for inputs with substantial repetition: code generation, JSON/XML
/// output, structured data, boilerplate text. For inputs without repetition
/// the proposer gracefully returns empty proposals.
///
/// # Complexity
///
/// - Construction: O(n log^2 n) where n = len(token_ids)
/// - Lookup: O(m log n) where m = context_window
/// - Space: O(n) on CPU, zero VRAM
#[derive(Debug)]
pub struct SuffixArrayProposer {
    config: SuffixArrayConfig,
}

impl SuffixArrayProposer {
    /// Create a new suffix array proposer with the given configuration.
    pub fn new(config: SuffixArrayConfig) -> Self {
        Self { config }
    }

    /// Create a proposer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SuffixArrayConfig::default())
    }
}

impl SpeculativeProposer for SuffixArrayProposer {
    fn propose(&self, token_ids: &[u32], max_tokens: usize) -> Vec<u32> {
        let k = max_tokens.min(self.config.max_speculation_length);
        if k == 0 || token_ids.len() < self.config.min_match_length {
            return Vec::new();
        }

        find_suffix_proposals(
            token_ids,
            self.config.min_match_length,
            self.config.context_window,
            k,
        )
    }

    fn name(&self) -> &str {
        "suffix_array"
    }
}

impl DraftProposer for SuffixArrayProposer {
    fn init_request(
        &mut self,
        _request_id: RequestId,
        _prompt_tokens: &[u32],
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn propose_for_request(
        &mut self,
        _request_id: RequestId,
        _last_token: u32,
        state: &mut SequenceState,
        _tokenizer: &TokenizerWrapper,
    ) -> Result<Vec<u32>, EngineError> {
        let mut all_tokens =
            Vec::with_capacity(state.prompt_token_ids.len() + state.generated_token_ids.len());
        all_tokens.extend_from_slice(&state.prompt_token_ids);
        all_tokens.extend_from_slice(&state.generated_token_ids);
        Ok(self.propose(&all_tokens, self.config.max_speculation_length))
    }

    fn on_tokens_verified(
        &mut self,
        _request_id: RequestId,
        _num_accepted: usize,
        _original_offset: usize,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn finish_request(&mut self, _request_id: RequestId) -> Result<(), EngineError> {
        Ok(())
    }

    fn preempt_request(&mut self, _request_id: RequestId) -> Result<(), EngineError> {
        Ok(())
    }

    fn num_speculative_tokens(&self) -> usize {
        self.config.max_speculation_length
    }

    fn name(&self) -> &str {
        "suffix_array"
    }
}

/// Core suffix-array matching algorithm.
///
/// Builds a suffix array over `token_ids`, then for each suffix starting
/// position near the end of the sequence, finds the longest matching earlier
/// occurrence using the LCP array. Returns the continuation tokens following
/// the best match.
///
/// The algorithm tries each suffix starting position from `n - min_match_length`
/// down to `max(1, n - context_window)`, looking up the SA neighbors to find
/// the longest match with an earlier position. The best match (longest LCP
/// with an earlier occurrence) across all tried positions is used.
fn find_suffix_proposals(
    token_ids: &[u32],
    min_match_length: usize,
    context_window: usize,
    k: usize,
) -> Vec<u32> {
    let n = token_ids.len();
    if n < min_match_length * 2 || k == 0 {
        // Need at least min_match_length tokens before and after the split point
        // for a valid match to exist.
        return Vec::new();
    }

    let sa = SuffixArray::build(token_ids);

    // Try suffix starting positions from the end moving backward.
    // For each position, the suffix at that position represents the "recent
    // context" we want to match against earlier occurrences.
    let earliest_start = n.saturating_sub(context_window).max(1);
    let latest_start = n.saturating_sub(min_match_length);

    let mut best_match_pos = 0usize;
    let mut best_match_len = 0usize;
    let mut best_suffix_pos = 0usize;

    for suffix_pos in earliest_start..=latest_start {
        let (match_pos, match_len) =
            sa.find_best_earlier_match(suffix_pos, min_match_length, suffix_pos);

        if match_len > best_match_len
            || (match_len == best_match_len && match_len > 0 && suffix_pos > best_suffix_pos)
        {
            // Prefer longer matches; for equal length, prefer the suffix
            // closest to the end of the sequence (latest suffix_pos) since it
            // matches the most recent context and leaves more room for
            // continuation tokens between the earlier occurrence and the suffix.
            best_match_len = match_len;
            best_match_pos = match_pos;
            best_suffix_pos = suffix_pos;
        }
    }

    if best_match_len < min_match_length {
        return Vec::new();
    }

    // The continuation starts right after the matched portion in the earlier
    // occurrence. Cap match_len so the match doesn't extend into the suffix
    // region (where our query suffix starts).
    let effective_match_len = best_match_len.min(n - best_suffix_pos);
    let continuation_start = best_match_pos + effective_match_len;
    // Don't read past the suffix region: the continuation must come from
    // tokens *before* the suffix we matched from.
    let continuation_end = best_suffix_pos.min(continuation_start + k);
    if continuation_start >= continuation_end {
        return Vec::new();
    }

    token_ids[continuation_start..continuation_end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── SuffixArray construction tests ──────────────────────────────────

    #[test]
    fn suffix_array_empty() {
        let sa = SuffixArray::build(&[]);
        assert!(sa.tokens.is_empty());
        assert!(sa.sa.is_empty());
        assert!(sa.lcp.is_empty());
    }

    #[test]
    fn suffix_array_single_token() {
        let sa = SuffixArray::build(&[42]);
        assert_eq!(sa.tokens, vec![42]);
        assert_eq!(sa.sa, vec![0]);
        assert_eq!(sa.lcp, vec![0]);
    }

    #[test]
    fn suffix_array_sorted_order() {
        // tokens: [3, 1, 2]
        // Suffixes:
        //   0: [3, 1, 2]
        //   1: [1, 2]
        //   2: [2]
        // Sorted: [1,2] < [2] < [3,1,2]
        // SA should be [1, 2, 0]
        let sa = SuffixArray::build(&[3, 1, 2]);
        assert_eq!(sa.sa, vec![1, 2, 0]);
    }

    #[test]
    fn suffix_array_all_same_tokens() {
        // tokens: [5, 5, 5, 5]
        // Suffixes (sorted by length, all start with 5):
        //   3: [5]
        //   2: [5, 5]
        //   1: [5, 5, 5]
        //   0: [5, 5, 5, 5]
        // SA should be [3, 2, 1, 0]
        let sa = SuffixArray::build(&[5, 5, 5, 5]);
        assert_eq!(sa.sa, vec![3, 2, 1, 0]);
    }

    #[test]
    fn suffix_array_lcp_basic() {
        // tokens: [1, 2, 1, 2, 3]
        // Suffixes sorted:
        //   0: [1, 2, 1, 2, 3]
        //   2: [1, 2, 3]
        //   1: [2, 1, 2, 3]
        //   3: [2, 3]
        //   4: [3]
        // SA: [0, 2, 1, 3, 4]
        // LCP: [0, 2, 0, 1, 0]
        //   lcp[0] = 0 (by convention)
        //   lcp[1] = lcp(sa[0], sa[1]) = lcp([1,2,1,2,3], [1,2,3]) = 2
        //   lcp[2] = lcp(sa[1], sa[2]) = lcp([1,2,3], [2,1,2,3]) = 0
        //   lcp[3] = lcp(sa[2], sa[3]) = lcp([2,1,2,3], [2,3]) = 1
        //   lcp[4] = lcp(sa[3], sa[4]) = lcp([2,3], [3]) = 0
        let sa = SuffixArray::build(&[1, 2, 1, 2, 3]);
        assert_eq!(sa.sa, vec![0, 2, 1, 3, 4]);
        assert_eq!(sa.lcp, vec![0, 2, 0, 1, 0]);
    }

    // ─── find_best_earlier_match tests ──────────────────────────────────

    #[test]
    fn find_match_basic_repetition() {
        // tokens: [1, 2, 3, 4, 1, 2, 3]
        // Suffix at position 4 is [1, 2, 3], which matches suffix at pos 0.
        // LCP = 3. Match at position 0, exclude_start = 4.
        let sa = SuffixArray::build(&[1, 2, 3, 4, 1, 2, 3]);
        let (pos, len) = sa.find_best_earlier_match(4, 2, 4);
        assert_eq!(len, 3);
        assert_eq!(pos, 0);
    }

    #[test]
    fn find_match_no_match() {
        // tokens: [1, 2, 3, 4, 5] -- all unique, no suffix shares min 2 tokens
        // with an earlier one.
        let sa = SuffixArray::build(&[1, 2, 3, 4, 5]);
        let (_, len) = sa.find_best_earlier_match(3, 2, 3);
        assert_eq!(len, 0);
    }

    #[test]
    fn find_match_respects_min_length() {
        // tokens: [1, 2, 3, 4, 1]
        // Suffix at pos 4 is [1], matches pos 0 with LCP = 1.
        // But min_match = 2, so it should not match.
        let sa = SuffixArray::build(&[1, 2, 3, 4, 1]);
        let (_, len) = sa.find_best_earlier_match(4, 2, 4);
        assert_eq!(len, 0);
    }

    #[test]
    fn find_match_excludes_suffix_region() {
        // tokens: [1, 2, 3, 1, 2, 3]
        // Suffix at pos 3 is [1, 2, 3], matches pos 0 with LCP = 3.
        // exclude_start = 3 means positions >= 3 are excluded.
        // Pos 0 is valid.
        let sa = SuffixArray::build(&[1, 2, 3, 1, 2, 3]);
        let (pos, len) = sa.find_best_earlier_match(3, 2, 3);
        assert_eq!(len, 3);
        assert_eq!(pos, 0);
    }

    // ─── find_suffix_proposals tests ─────────────────────────────────────

    #[test]
    fn proposals_from_repeated_pattern() {
        // tokens: [1, 2, 3, 4, 5, 1, 2, 3]
        // Suffix [1, 2, 3] matches at position 0, continuation is [4, 5]
        let result = find_suffix_proposals(&[1, 2, 3, 4, 5, 1, 2, 3], 3, 32, 5);
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn proposals_empty_when_no_match() {
        // All unique tokens, no repeating pattern.
        let result = find_suffix_proposals(&[1, 2, 3, 4, 5], 3, 32, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn proposals_empty_for_short_input() {
        let result = find_suffix_proposals(&[1, 2], 3, 32, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn proposals_empty_for_empty_input() {
        let result = find_suffix_proposals(&[], 3, 32, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn proposals_respect_k_limit() {
        // tokens: [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
        // Match at position 0, continuation [4, 5, 6, 7] but k=2
        let result = find_suffix_proposals(&[1, 2, 3, 4, 5, 6, 7, 1, 2, 3], 3, 32, 2);
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn proposals_zero_k_returns_empty() {
        let result = find_suffix_proposals(&[1, 2, 3, 1, 2, 3], 2, 32, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn proposals_repetitive_input_yields_good_results() {
        // Simulating JSON-like repetition:
        // "key": 1, "key": 2, "key":
        let tokens: Vec<u32> = vec![10, 20, 30, 1, 10, 20, 30, 2, 10, 20, 30];
        // Suffix [10, 20, 30] matches at position 0 (continuation [1, 10, 20, 30])
        // or position 4 (continuation [2, 10, 20, 30]).
        // Should pick the match with most continuation tokens -- earliest.
        let result = find_suffix_proposals(&tokens, 3, 32, 5);
        assert!(!result.is_empty());
        // Earliest match at pos 0 gives continuation [1, 10, 20, 30, 2, ...]
        // but capped by exclude_start=8 so continuation_end = min(n, pos+match+k)
        // Actually: match at 0, match_len=3, continuation_start=3,
        // continuation_end = min(8, 3+5) = 8. So [1, 10, 20, 30, 2]
        assert_eq!(result, vec![1, 10, 20, 30, 2]);
    }

    #[test]
    fn proposals_all_same_tokens() {
        // [5, 5, 5, 5, 5] with min_match=3, context_window=32, k=3.
        // n=5, need min_match*2=6 tokens for a valid setup, but n=5 < 6.
        // Returns empty because there aren't enough tokens for both a match
        // and an earlier occurrence.
        let result = find_suffix_proposals(&[5, 5, 5, 5, 5], 3, 32, 3);
        assert!(result.is_empty());

        // With min_match=2, n=5 >= 4 (min_match*2). The suffix at pos 2
        // matches pos 1 with LCP=3, but the match extends right up to the
        // suffix region leaving no room for continuation tokens. Returns empty.
        let result = find_suffix_proposals(&[5, 5, 5, 5, 5], 2, 3, 2);
        assert!(result.is_empty());

        // With more tokens before the suffix, more continuation is available.
        let result = find_suffix_proposals(&[5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 2, 3, 2);
        // n=10. Suffix at pos 8 matches pos 0 with LCP=2.
        // Continuation: tokens[2..min(8, 4)] = tokens[2..4] = [5, 5].
        assert_eq!(result, vec![5, 5]);
    }

    #[test]
    fn proposals_with_long_context() {
        // Simulate a long context with a repeated block at the end.
        let mut tokens: Vec<u32> = (0..100).collect();
        // Append a repetition of tokens [10..20]
        tokens.extend_from_slice(&(10..20).collect::<Vec<u32>>());

        let result = find_suffix_proposals(&tokens, 3, 32, 5);
        // The suffix [10, 11, 12, ..., 19] should match at position 10.
        // Continuation after position 10+10=20 is [20, 21, 22, 23, 24].
        assert_eq!(result, vec![20, 21, 22, 23, 24]);
    }

    #[test]
    fn proposals_min_match_length_respected() {
        // tokens: [1, 2, 3, 4, 2, 3]
        // With min_match=3: suffix [4, 2, 3] -- 4 doesn't repeat, so no 3-length match.
        // The longest matching prefix of pattern starting at pos 3 is [2, 3] (length 2).
        // With min_match=3, this is too short.
        let result = find_suffix_proposals(&[1, 2, 3, 4, 2, 3], 3, 32, 5);
        assert!(result.is_empty());

        // With min_match=2: [2, 3] matches at position 1, continuation is [4].
        let result = find_suffix_proposals(&[1, 2, 3, 4, 2, 3], 2, 32, 5);
        assert_eq!(result, vec![4]);
    }

    #[test]
    fn proposals_context_window_limits_search() {
        // tokens: [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
        // With context_window=2: pattern = [1, 2] (last 2 tokens)
        // Matches at position 0, continuation = [3, 4, 5]
        let result = find_suffix_proposals(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 2], 2, 2, 3);
        assert_eq!(result, vec![3, 4, 5]);

        // With context_window=5: the algorithm tries suffix positions 5..8.
        // Suffix at position 8 is [1, 2], which matches position 0 with LCP=2.
        // Continuation: tokens[2..5] = [3, 4, 5].
        // The suffix array approach iterates over multiple suffix starting
        // positions, so it finds matches even when the full context window
        // starts with non-repeating tokens.
        let result = find_suffix_proposals(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 2], 2, 5, 3);
        assert_eq!(result, vec![3, 4, 5]);
    }

    // ─── SuffixArrayProposer trait tests ─────────────────────────────────

    #[test]
    fn proposer_trait_name() {
        let proposer = SuffixArrayProposer::with_defaults();
        assert_eq!(SpeculativeProposer::name(&proposer), "suffix_array");
    }

    #[test]
    fn proposer_respects_max_tokens() {
        let proposer = SuffixArrayProposer::new(SuffixArrayConfig {
            max_speculation_length: 10,
            min_match_length: 3,
            context_window: 32,
        });
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3];
        // Match at 0, continuation [4,5,6,7,8,9,10]. max_tokens=2 caps it.
        let result = proposer.propose(&tokens, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn proposer_max_speculation_length_caps_output() {
        let proposer = SuffixArrayProposer::new(SuffixArrayConfig {
            max_speculation_length: 1,
            min_match_length: 3,
            context_window: 32,
        });
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3];
        // Even though max_tokens is large, max_speculation_length=1 caps it.
        let result = proposer.propose(&tokens, 100);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn proposer_with_defaults_works() {
        let proposer = SuffixArrayProposer::with_defaults();
        // Default: max_speculation_length=5, min_match_length=3, context_window=32
        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 10, 20, 30];
        let result = proposer.propose(&tokens, 5);
        // [10, 20, 30] matches at pos 0, continuation [40, 50]
        assert_eq!(result, vec![40, 50]);
    }

    #[test]
    fn proposer_empty_input() {
        let proposer = SuffixArrayProposer::with_defaults();
        let result = proposer.propose(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn proposer_input_shorter_than_min_match() {
        let proposer = SuffixArrayProposer::with_defaults();
        // Default min_match_length=3, only 2 tokens.
        let result = proposer.propose(&[1, 2], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn proposer_on_tokens_accepted_is_noop() {
        let mut proposer = SuffixArrayProposer::with_defaults();
        // Should not panic.
        proposer.on_tokens_accepted(42, &[1, 2, 3]);
    }

    #[test]
    fn proposer_on_request_finished_is_noop() {
        let mut proposer = SuffixArrayProposer::with_defaults();
        // Should not panic.
        proposer.on_request_finished(42);
    }

    #[test]
    fn proposer_zero_max_tokens() {
        let proposer = SuffixArrayProposer::with_defaults();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 1, 2, 3];
        let result = proposer.propose(&tokens, 0);
        assert!(result.is_empty());
    }

    // ─── Edge case and integration tests ─────────────────────────────────

    #[test]
    fn code_like_repetition() {
        // Simulating code with repeated function structure:
        // fn_token open_paren param close_paren open_brace body close_brace
        let fn_tok = 100u32;
        let open_p = 101;
        let param = 102;
        let close_p = 103;
        let open_b = 104;
        let body = 105;
        let close_b = 106;
        let newline = 107;

        let tokens: Vec<u32> = vec![
            fn_tok, open_p, param, close_p, open_b, body, close_b, newline, fn_tok, open_p, param,
            close_p,
        ];

        let proposer = SuffixArrayProposer::new(SuffixArrayConfig {
            max_speculation_length: 5,
            min_match_length: 3,
            context_window: 8,
        });

        let result = proposer.propose(&tokens, 5);
        // Suffix [fn_tok, open_p, param, close_p] matches at pos 0.
        // Continuation: [open_b, body, close_b, newline]
        assert_eq!(result, vec![open_b, body, close_b, newline]);
    }

    #[test]
    fn json_like_repetition() {
        // {"key": 1, "key": 2, "key":
        let brace = 200u32;
        let quote = 201;
        let key = 202;
        let colon = 203;
        let comma = 204;
        let one = 205;
        let two = 206;

        let tokens: Vec<u32> = vec![
            brace, quote, key, quote, colon, one, comma, quote, key, quote, colon, two, comma,
            quote, key, quote, colon,
        ];

        let proposer = SuffixArrayProposer::new(SuffixArrayConfig {
            max_speculation_length: 5,
            min_match_length: 3,
            context_window: 10,
        });

        let result = proposer.propose(&tokens, 5);
        // Should find a match and propose continuation tokens.
        assert!(!result.is_empty());
    }

    #[test]
    fn suffix_array_deterministic() {
        // Building the same suffix array twice should yield identical results.
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let sa1 = SuffixArray::build(&tokens);
        let sa2 = SuffixArray::build(&tokens);
        assert_eq!(sa1.sa, sa2.sa);
        assert_eq!(sa1.lcp, sa2.lcp);
    }

    #[test]
    fn proposer_implements_send_sync() {
        // Compile-time check that the trait bounds are satisfied.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SuffixArrayProposer>();
    }

    #[test]
    fn proposer_as_trait_object() {
        // Verify the proposer can be used as a trait object.
        let proposer: Box<dyn SpeculativeProposer> = Box::new(SuffixArrayProposer::with_defaults());
        assert_eq!(proposer.name(), "suffix_array");

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 1, 2, 3];
        let result = proposer.propose(&tokens, 5);
        assert_eq!(result, vec![4, 5]);
    }

    // ─── DraftProposer trait tests ────────────────────────────────────────

    use crate::request::SequenceState;
    use crate::tokenizer::TokenizerWrapper;

    fn test_tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::for_testing(1000)
    }

    #[test]
    fn draft_proposer_lifecycle_noop() {
        let mut proposer = SuffixArrayProposer::with_defaults();
        assert!(proposer.init_request(0, &[1, 2, 3, 4, 5]).is_ok());
        assert!(proposer.on_tokens_verified(0, 2, 5).is_ok());
        assert!(proposer.finish_request(0).is_ok());
        assert!(proposer.preempt_request(0).is_ok());
    }

    #[test]
    fn draft_proposer_name_and_tokens() {
        let proposer = SuffixArrayProposer::with_defaults();
        assert_eq!(DraftProposer::name(&proposer), "suffix_array");
        assert_eq!(proposer.num_speculative_tokens(), 5);
    }

    #[test]
    fn draft_proposer_propose_uses_full_sequence() {
        let mut proposer = SuffixArrayProposer::new(SuffixArrayConfig {
            max_speculation_length: 5,
            min_match_length: 3,
            context_window: 32,
        });
        let tokenizer = test_tokenizer();
        let mut state = SequenceState::new(0, vec![1, 2, 3, 4, 5], 100, 99, 16, 0);
        // Repeat a prompt pattern in generated tokens
        state.generated_token_ids = vec![1, 2, 3];

        // Full sequence: [1, 2, 3, 4, 5, 1, 2, 3]
        // Suffix [1, 2, 3] matches at position 0, continuation: [4, 5]
        let result = proposer
            .propose_for_request(0, 3, &mut state, &tokenizer)
            .unwrap();
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn draft_proposer_as_trait_object() {
        let proposer: Box<dyn DraftProposer> = Box::new(SuffixArrayProposer::with_defaults());
        assert_eq!(proposer.name(), "suffix_array");
        assert_eq!(proposer.num_speculative_tokens(), 5);
    }
}
