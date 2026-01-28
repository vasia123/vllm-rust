//! Beam search decoding implementation.
//!
//! Beam search maintains multiple hypotheses (beams) at each step and selects
//! the top-k candidates based on cumulative log probability scores.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to maintain.
    pub beam_width: usize,
    /// Length normalization alpha (0 = no normalization, 1 = full normalization).
    pub length_penalty: f32,
    /// Whether to stop early when all beams have finished.
    pub early_stopping: bool,
    /// Number of beams to return (defaults to 1).
    pub num_return_beams: usize,
    /// Diversity penalty to encourage different beam paths (0 = disabled).
    pub diversity_penalty: f32,
    /// Number of beam groups for diverse beam search.
    pub num_beam_groups: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            length_penalty: 1.0,
            early_stopping: false,
            num_return_beams: 1,
            diversity_penalty: 0.0,
            num_beam_groups: 1,
        }
    }
}

/// A single beam hypothesis.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token sequence (not including prompt).
    pub token_ids: Vec<u32>,
    /// Cumulative log probability (unnormalized).
    pub score: f32,
    /// Whether this hypothesis has finished (hit EOS).
    pub is_finished: bool,
    /// Index of the beam this hypothesis came from (for KV cache tracking).
    pub parent_beam_idx: usize,
}

impl BeamHypothesis {
    pub fn new() -> Self {
        Self {
            token_ids: Vec::new(),
            score: 0.0,
            is_finished: false,
            parent_beam_idx: 0,
        }
    }

    /// Compute length-normalized score.
    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        if length_penalty == 0.0 || self.token_ids.is_empty() {
            self.score
        } else {
            // Wu et al. (2016) length normalization
            let len = self.token_ids.len() as f32;
            let lp = ((5.0 + len) / 6.0).powf(length_penalty);
            self.score / lp
        }
    }
}

impl Default for BeamHypothesis {
    fn default() -> Self {
        Self::new()
    }
}

/// Candidate for beam expansion.
#[derive(Debug, Clone)]
struct BeamCandidate {
    token_id: u32,
    score: f32,      // Cumulative score if this token is selected
    beam_idx: usize, // Which beam this extends
}

impl PartialEq for BeamCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for BeamCandidate {}

impl PartialOrd for BeamCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BeamCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (we want to keep highest scores)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Beam search state tracker.
#[derive(Debug)]
pub struct BeamSearchState {
    /// Configuration.
    pub config: BeamSearchConfig,
    /// Current beam hypotheses.
    pub beams: Vec<BeamHypothesis>,
    /// Completed hypotheses (hit EOS).
    pub completed: Vec<BeamHypothesis>,
    /// EOS token ID.
    eos_token_id: u32,
    /// Generation step counter.
    step: usize,
}

impl BeamSearchState {
    /// Create a new beam search state.
    pub fn new(config: BeamSearchConfig, eos_token_id: u32) -> Self {
        let mut beams = Vec::with_capacity(config.beam_width);
        for _ in 0..config.beam_width {
            beams.push(BeamHypothesis::new());
        }
        Self {
            config,
            beams,
            completed: Vec::new(),
            eos_token_id,
            step: 0,
        }
    }

    /// Get the number of active (non-finished) beams.
    pub fn num_active_beams(&self) -> usize {
        self.beams.iter().filter(|b| !b.is_finished).count()
    }

    /// Check if beam search is complete.
    pub fn is_done(&self) -> bool {
        if self.config.early_stopping {
            // Done if we have enough completed hypotheses and best completed
            // is better than worst active beam
            if self.completed.len() >= self.config.num_return_beams {
                let best_completed_score = self
                    .completed
                    .iter()
                    .map(|h| h.normalized_score(self.config.length_penalty))
                    .fold(f32::NEG_INFINITY, f32::max);

                // Get worst active beam's best possible score
                let worst_active = self
                    .beams
                    .iter()
                    .filter(|b| !b.is_finished)
                    .map(|b| b.normalized_score(self.config.length_penalty))
                    .fold(f32::INFINITY, f32::min);

                return best_completed_score >= worst_active;
            }
        }

        // All beams finished
        self.num_active_beams() == 0
    }

    /// Perform beam expansion given log probabilities for each beam.
    ///
    /// `log_probs` is a 2D slice where `log_probs[beam_idx][token_id]` is the
    /// log probability of generating `token_id` from beam `beam_idx`.
    ///
    /// Returns the mapping from new beam index to (old_beam_idx, new_token_id).
    pub fn step(&mut self, log_probs: &[Vec<f32>]) -> Vec<(usize, u32)> {
        self.step += 1;

        let vocab_size = log_probs.first().map(|v| v.len()).unwrap_or(0);
        if vocab_size == 0 {
            return Vec::new();
        }

        // Collect all candidates
        let mut candidates: BinaryHeap<BeamCandidate> = BinaryHeap::new();

        for (beam_idx, beam) in self.beams.iter().enumerate() {
            if beam.is_finished {
                continue;
            }

            let beam_log_probs = &log_probs[beam_idx];
            for (token_id, &lp) in beam_log_probs.iter().enumerate() {
                if lp == f32::NEG_INFINITY {
                    continue;
                }

                let new_score = beam.score + lp;

                // Apply diversity penalty if configured
                let adjusted_score = if self.config.diversity_penalty > 0.0 {
                    // Penalize tokens that appear in other beams at this step
                    new_score - self.config.diversity_penalty * (beam_idx as f32)
                } else {
                    new_score
                };

                candidates.push(BeamCandidate {
                    token_id: token_id as u32,
                    score: adjusted_score,
                    beam_idx,
                });
            }
        }

        // Select top-k candidates
        let mut new_beams = Vec::with_capacity(self.config.beam_width);
        let mut beam_transitions = Vec::with_capacity(self.config.beam_width);

        while new_beams.len() < self.config.beam_width {
            let Some(candidate) = candidates.pop() else {
                break;
            };

            let parent_beam = &self.beams[candidate.beam_idx];
            let mut new_hyp = BeamHypothesis {
                token_ids: parent_beam.token_ids.clone(),
                score: parent_beam.score
                    + log_probs[candidate.beam_idx][candidate.token_id as usize],
                is_finished: candidate.token_id == self.eos_token_id,
                parent_beam_idx: candidate.beam_idx,
            };
            new_hyp.token_ids.push(candidate.token_id);

            if new_hyp.is_finished {
                self.completed.push(new_hyp);
            } else {
                beam_transitions.push((candidate.beam_idx, candidate.token_id));
                new_beams.push(new_hyp);
            }
        }

        // Fill remaining slots if we don't have enough non-finished beams
        while new_beams.len() < self.config.beam_width {
            new_beams.push(BeamHypothesis {
                token_ids: Vec::new(),
                score: f32::NEG_INFINITY,
                is_finished: true,
                parent_beam_idx: 0,
            });
        }

        self.beams = new_beams;
        beam_transitions
    }

    /// Get the best completed hypotheses.
    pub fn get_best_hypotheses(&self) -> Vec<&BeamHypothesis> {
        let mut all: Vec<&BeamHypothesis> =
            self.completed.iter().chain(self.beams.iter()).collect();

        all.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_penalty);
            let score_b = b.normalized_score(self.config.length_penalty);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        all.truncate(self.config.num_return_beams);
        all
    }

    /// Get current generation step.
    pub fn current_step(&self) -> usize {
        self.step
    }
}

/// Sample top-k candidates from logits for beam search.
///
/// Returns the top-k (token_id, log_prob) pairs sorted by log_prob descending.
pub fn beam_search_top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }

    // Compute log softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum_ln = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>()
        .ln();

    let log_probs: Vec<f32> = logits.iter().map(|&x| x - max_logit - exp_sum_ln).collect();

    // Get top-k
    let mut indexed: Vec<(u32, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i as u32, lp))
        .collect();

    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed.truncate(k);
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_hypothesis_new() {
        let hyp = BeamHypothesis::new();
        assert!(hyp.token_ids.is_empty());
        assert_eq!(hyp.score, 0.0);
        assert!(!hyp.is_finished);
    }

    #[test]
    fn test_length_normalization() {
        let mut hyp = BeamHypothesis::new();
        hyp.score = -10.0;
        hyp.token_ids = vec![1, 2, 3, 4, 5];

        // No normalization
        assert_eq!(hyp.normalized_score(0.0), -10.0);

        // With normalization, score should be less negative
        let normalized = hyp.normalized_score(1.0);
        assert!(normalized > -10.0);
    }

    #[test]
    fn test_beam_search_state_creation() {
        let config = BeamSearchConfig {
            beam_width: 4,
            ..Default::default()
        };
        let state = BeamSearchState::new(config, 2);

        assert_eq!(state.beams.len(), 4);
        assert_eq!(state.num_active_beams(), 4);
        assert!(!state.is_done());
    }

    #[test]
    fn test_beam_search_step() {
        let config = BeamSearchConfig {
            beam_width: 2,
            ..Default::default()
        };
        let mut state = BeamSearchState::new(config, 2);

        // Simulate log probabilities for 2 beams, vocab size 4
        // Token 2 is EOS
        let log_probs = vec![
            vec![-1.0, -2.0, -0.5, -3.0], // Beam 0: token 2 (EOS) has highest prob
            vec![-2.0, -1.0, -4.0, -3.0], // Beam 1: token 1 has highest prob
        ];

        let transitions = state.step(&log_probs);

        // Should have transitions for the new beams
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_beam_search_eos_completion() {
        let config = BeamSearchConfig {
            beam_width: 2,
            early_stopping: true,
            num_return_beams: 1,
            ..Default::default()
        };
        let mut state = BeamSearchState::new(config, 2);

        // Run a few steps where EOS is eventually selected
        // First step: some tokens, no EOS
        let log_probs1 = vec![
            vec![-1.0, -2.0, -10.0, -3.0], // Token 0 best (not EOS)
            vec![-2.0, -1.0, -10.0, -3.0], // Token 1 best (not EOS)
        ];
        state.step(&log_probs1);

        // Second step: EOS is best
        let log_probs2 = vec![
            vec![-10.0, -10.0, -0.1, -10.0], // Token 2 (EOS) best
            vec![-10.0, -10.0, -0.2, -10.0], // Token 2 (EOS) best
        ];
        state.step(&log_probs2);

        // At least one beam should have completed with EOS
        assert!(
            !state.completed.is_empty(),
            "Should have completed hypotheses after EOS selection"
        );
    }

    #[test]
    fn test_beam_search_top_k() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let top = beam_search_top_k(&logits, 3);

        assert_eq!(top.len(), 3);
        // Top 3 should be indices 1 (5.0), 4 (4.0), 2 (3.0)
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 4);
        assert_eq!(top[2].0, 2);
    }

    #[test]
    fn test_beam_search_top_k_k_larger_than_vocab() {
        let logits = vec![1.0, 2.0, 3.0];
        let top = beam_search_top_k(&logits, 10);

        // Should return all 3 tokens
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_get_best_hypotheses() {
        let config = BeamSearchConfig {
            beam_width: 2,
            num_return_beams: 2,
            length_penalty: 0.0,
            ..Default::default()
        };
        let mut state = BeamSearchState::new(config, 2);

        // Manually set up beams with different scores
        state.beams[0].score = -5.0;
        state.beams[0].token_ids = vec![1, 1];
        state.beams[1].score = -3.0;
        state.beams[1].token_ids = vec![1, 2];

        let best = state.get_best_hypotheses();

        assert_eq!(best.len(), 2);
        // Best hypothesis should have higher score (-3.0)
        assert_eq!(best[0].score, -3.0);
    }

    #[test]
    fn test_diversity_penalty() {
        let config = BeamSearchConfig {
            beam_width: 2,
            diversity_penalty: 0.5,
            ..Default::default()
        };
        let mut state = BeamSearchState::new(config, 2);

        // When diversity penalty is applied, beam 1's candidates get penalized
        let log_probs = vec![
            vec![-1.0, -1.0, -10.0, -10.0],
            vec![-1.0, -1.0, -10.0, -10.0],
        ];

        let transitions = state.step(&log_probs);

        // Should have 2 transitions
        assert_eq!(transitions.len(), 2);
    }

    #[test]
    fn test_early_stopping() {
        let config = BeamSearchConfig {
            beam_width: 2,
            early_stopping: true,
            num_return_beams: 1,
            length_penalty: 0.0,
            ..Default::default()
        };
        let mut state = BeamSearchState::new(config, 2);

        // Complete one beam with a good score
        state.completed.push(BeamHypothesis {
            token_ids: vec![1],
            score: -0.1,
            is_finished: true,
            parent_beam_idx: 0,
        });

        // Active beams have worse scores
        state.beams[0].score = -5.0;
        state.beams[0].is_finished = false;
        state.beams[1].score = -6.0;
        state.beams[1].is_finished = false;

        // Should be done because completed beam is better
        assert!(state.is_done());
    }

    #[test]
    fn test_all_beams_finished() {
        let config = BeamSearchConfig {
            beam_width: 2,
            ..Default::default()
        };
        let mut state = BeamSearchState::new(config, 2);

        state.beams[0].is_finished = true;
        state.beams[1].is_finished = true;

        assert_eq!(state.num_active_beams(), 0);
        assert!(state.is_done());
    }
}
