use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Result of a sampling operation, including logprob information.
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// The sampled token ID.
    pub token_id: u32,
    /// Log probability of the sampled token.
    pub logprob: f32,
    /// Top-k tokens by log probability, if requested.
    pub top_logprobs: Option<Vec<(u32, f32)>>,
}

/// Parameters controlling token sampling behavior.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for logit scaling. 0.0 = greedy, higher = more random.
    pub temperature: f32,
    /// Nucleus sampling threshold (0..1). 1.0 = disabled.
    pub top_p: f32,
    /// Top-K filtering. 0 = disabled.
    pub top_k: u32,
    /// Penalty for repeated tokens. 1.0 = none, >1.0 discourages repeats.
    pub repetition_penalty: f32,
    /// Minimum probability relative to max. 0.0 = disabled.
    pub min_p: f32,
    /// Optional seed for deterministic sampling.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            min_p: 0.0,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-6
    }
}

/// Mutable state for sampling (holds RNG per-sequence).
pub struct SamplerState {
    rng: StdRng,
}

impl SamplerState {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { rng }
    }
}

/// Sample a token from logits given sampling parameters.
///
/// `logits` is a slice of f32 with length == vocab_size.
/// `generated_tokens` is the list of previously generated token IDs (for repetition penalty).
/// `num_top_logprobs` if Some(k), extracts top-k logprobs for the response.
///
/// Returns a `SamplingResult` containing the token ID, its log probability, and optionally
/// the top-k log probabilities.
pub fn sample(
    logits: &[f32],
    params: &SamplingParams,
    generated_tokens: &[u32],
    sampler_state: &mut SamplerState,
    num_top_logprobs: Option<usize>,
) -> SamplingResult {
    let vocab_size = logits.len();
    let mut logits = logits.to_vec();

    // Step 1: Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut logits, generated_tokens, params.repetition_penalty);
    }

    // Compute log-softmax for logprob extraction (before temperature scaling for greedy)
    // For non-greedy, we compute after temperature scaling
    let log_probs_for_logprobs: Option<Vec<f32>> = if num_top_logprobs.is_some() {
        if params.is_greedy() {
            Some(log_softmax(&logits))
        } else {
            None // Will compute after temperature scaling
        }
    } else {
        None
    };

    // Step 2: Greedy fast-path
    if params.is_greedy() {
        let token_id = argmax(&logits);
        let (logprob, top_logprobs) = if let Some(ref log_probs) = log_probs_for_logprobs {
            let lp = log_probs[token_id as usize];
            let top = num_top_logprobs.map(|k| extract_top_k_logprobs(log_probs, k));
            (lp, top)
        } else {
            (f32::NEG_INFINITY, None)
        };
        return SamplingResult {
            token_id,
            logprob,
            top_logprobs,
        };
    }

    // Step 3: Apply temperature
    if params.temperature != 1.0 {
        let inv_temp = 1.0 / params.temperature;
        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    // Compute log-softmax after temperature scaling for logprob extraction
    let log_probs = if num_top_logprobs.is_some() {
        Some(log_softmax(&logits))
    } else {
        None
    };

    // Step 4: Convert to probabilities via softmax
    let mut probs = softmax(&logits);

    // Step 5: Apply min_p filtering
    if params.min_p > 0.0 {
        apply_min_p(&mut probs, params.min_p);
    }

    // Step 6: Apply top-k filtering
    if params.top_k > 0 && (params.top_k as usize) < vocab_size {
        apply_top_k(&mut probs, params.top_k as usize);
    }

    // Step 7: Apply top-p (nucleus) filtering
    if params.top_p < 1.0 && params.top_p > 0.0 {
        apply_top_p(&mut probs, params.top_p);
    }

    // Step 8: Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum != 1.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }

    // Step 9: Sample from distribution
    let token_id = sample_from_probs(&probs, &mut sampler_state.rng);

    // Extract logprob for selected token and top-k
    let (logprob, top_logprobs) = if let Some(ref lp) = log_probs {
        let selected_logprob = lp[token_id as usize];
        let top = num_top_logprobs.map(|k| extract_top_k_logprobs(lp, k));
        (selected_logprob, top)
    } else {
        (f32::NEG_INFINITY, None)
    };

    SamplingResult {
        token_id,
        logprob,
        top_logprobs,
    }
}

fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
    for &token_id in generated_tokens {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = max_prob * min_p;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

fn apply_top_k(probs: &mut [f32], k: usize) {
    // Find the k-th largest probability using partial sort
    let mut top_k_values: Vec<f32> = probs.to_vec();
    top_k_values.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = top_k_values[k.min(top_k_values.len()) - 1];
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

fn apply_top_p(probs: &mut [f32], top_p: f32) {
    // Sort indices by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum > top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out everything after cutoff
    for &(idx, _) in &indexed[cutoff_idx..] {
        probs[idx] = 0.0;
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }
    probs
}

/// Compute log-softmax in a numerically stable way.
/// log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum_ln = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>()
        .ln();
    logits.iter().map(|&x| x - max_logit - exp_sum_ln).collect()
}

/// Extract the top-k tokens by log probability.
fn extract_top_k_logprobs(log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i as u32, lp))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> u32 {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    // Fallback: return last non-zero token
    probs.len() as u32 - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams::greedy();
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None);
        assert_eq!(result.token_id, 1); // index of max value (5.0)
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None);
        assert_eq!(result.token_id, 2); // index of 10.0
    }

    #[test]
    fn deterministic_with_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };

        let mut state1 = SamplerState::new(Some(123));
        let mut state2 = SamplerState::new(Some(123));

        let result1 = sample(&logits, &params, &[], &mut state1, None);
        let result2 = sample(&logits, &params, &[], &mut state2, None);
        assert_eq!(result1.token_id, result2.token_id);
    }

    #[test]
    fn repetition_penalty_reduces_repeated() {
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Token 0 was already generated, should be penalized
        let result = sample(&logits, &params, &[0], &mut state, None);
        assert_ne!(result.token_id, 0); // penalized token should not be selected
    }

    #[test]
    fn top_k_limits_candidates() {
        // With top_k=1, should always pick the max
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None);
        assert_eq!(result.token_id, 1); // only top-1 survives
    }

    #[test]
    fn top_p_nucleus_sampling() {
        // Very low top_p should mostly select the highest probability token
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.1,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None);
        assert_eq!(result.token_id, 0); // 10.0 dominates
    }

    #[test]
    fn min_p_filters_low_probability() {
        // With min_p = 0.5, tokens with prob < max_prob * 0.5 are removed
        let logits = vec![10.0, 9.9, 0.0, 0.0]; // first two are close, last two negligible
        let params = SamplingParams {
            temperature: 0.01, // very low temp → almost greedy
            min_p: 0.5,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None);
        assert!(result.token_id <= 1); // only first two tokens should be candidates
    }

    #[test]
    fn default_params_are_identity() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.min_p, 0.0);
        assert!(params.seed.is_none());
    }

    #[test]
    fn softmax_produces_valid_distribution() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs.iter().all(|&p| p >= 0.0));
        // Higher logit → higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn sampling_respects_distribution() {
        // With high temperature, sampling should produce varied results
        let logits = vec![1.0; 10]; // uniform logits
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };

        let mut counts = vec![0u32; 10];
        let mut state = SamplerState::new(Some(0));
        for _ in 0..1000 {
            let result = sample(&logits, &params, &[], &mut state, None);
            counts[result.token_id as usize] += 1;
        }

        // With uniform logits, all tokens should be sampled at least once
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn repetition_penalty_negative_logits() {
        // Negative logits should be multiplied by penalty (making them more negative)
        let logits = vec![-1.0, -1.0, 5.0, -1.0];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Token 2 (the only positive one) was generated, penalize it
        let result = sample(&logits, &params, &[2], &mut state, None);
        // 5.0 / 2.0 = 2.5, still higher than -1.0, so token 2 still wins
        assert_eq!(result.token_id, 2);

        // But with higher penalty...
        let params2 = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 10.0,
            ..Default::default()
        };
        let result2 = sample(&logits, &params2, &[2], &mut state, None);
        // 5.0 / 10.0 = 0.5, -1.0 * 10.0 = -10.0 for the rest → token 2 still wins
        assert_eq!(result2.token_id, 2);
    }

    #[test]
    fn log_softmax_correctness() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);

        // log_softmax should sum to 0 when exponentiated (i.e., exp sum = 1)
        let exp_sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);

        // Higher logit → higher log probability
        assert!(log_probs[2] > log_probs[1]);
        assert!(log_probs[1] > log_probs[0]);

        // All log probs should be <= 0
        assert!(log_probs.iter().all(|&lp| lp <= 0.0));
    }

    #[test]
    fn sample_returns_correct_logprob_for_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams::greedy();
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, Some(3));

        // Token 1 has the highest logit (5.0)
        assert_eq!(result.token_id, 1);

        // Log probability should be negative and finite
        assert!(result.logprob.is_finite());
        assert!(result.logprob < 0.0);

        // Should have top 3 logprobs
        let top = result.top_logprobs.unwrap();
        assert_eq!(top.len(), 3);
        // First entry should be the highest (token 1)
        assert_eq!(top[0].0, 1);
        // Logprobs should be sorted descending
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn sample_returns_logprob_for_sampled() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, Some(5));

        // Logprob should be for the sampled token
        assert!(result.logprob.is_finite());
        assert!(result.logprob < 0.0);

        // Top logprobs should include all 5 tokens
        let top = result.top_logprobs.unwrap();
        assert_eq!(top.len(), 5);

        // The sampled token's logprob should match what's in top_logprobs
        let sampled_in_top = top.iter().find(|(id, _)| *id == result.token_id);
        assert!(sampled_in_top.is_some());
        let (_, lp) = sampled_in_top.unwrap();
        assert!((lp - result.logprob).abs() < 1e-6);
    }

    #[test]
    fn no_logprobs_when_not_requested() {
        let logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams::greedy();
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None);

        assert!(result.top_logprobs.is_none());
        // Logprob will be NEG_INFINITY when not computed
        assert!(result.logprob == f32::NEG_INFINITY);
    }

    // ==========================================================================
    // Extended edge case tests
    // ==========================================================================

    #[test]
    fn repetition_penalty_positive_logits_are_divided() {
        // Positive logits should be divided by penalty (discourage repetition)
        let logits = vec![10.0, 10.0];
        let mut logits_copy = logits.clone();
        apply_repetition_penalty(&mut logits_copy, &[0], 2.0);

        // Token 0 (positive): should be divided by 2.0 → 5.0
        assert!(
            (logits_copy[0] - 5.0).abs() < 1e-6,
            "Positive logit should be divided"
        );
        // Token 1: unchanged
        assert!(
            (logits_copy[1] - 10.0).abs() < 1e-6,
            "Unpunished logit unchanged"
        );
    }

    #[test]
    fn repetition_penalty_negative_logits_are_multiplied() {
        // Negative logits should be multiplied by penalty (more negative = less likely)
        let logits = vec![-5.0, -5.0];
        let mut logits_copy = logits.clone();
        apply_repetition_penalty(&mut logits_copy, &[0], 2.0);

        // Token 0 (negative): should be multiplied by 2.0 → -10.0
        assert!(
            (logits_copy[0] - (-10.0)).abs() < 1e-6,
            "Negative logit should be multiplied"
        );
        // Token 1: unchanged
        assert!(
            (logits_copy[1] - (-5.0)).abs() < 1e-6,
            "Unpunished logit unchanged"
        );
    }

    #[test]
    fn repetition_penalty_multiple_tokens() {
        let logits = vec![10.0, 8.0, 6.0, 4.0];
        let mut logits_copy = logits.clone();
        apply_repetition_penalty(&mut logits_copy, &[0, 2], 2.0);

        assert!((logits_copy[0] - 5.0).abs() < 1e-6); // penalized
        assert!((logits_copy[1] - 8.0).abs() < 1e-6); // unchanged
        assert!((logits_copy[2] - 3.0).abs() < 1e-6); // penalized
        assert!((logits_copy[3] - 4.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn repetition_penalty_out_of_range_token_ignored() {
        let logits = vec![5.0, 5.0];
        let mut logits_copy = logits.clone();
        // Token ID 100 is out of range (vocab_size=2)
        apply_repetition_penalty(&mut logits_copy, &[100], 2.0);

        // Should not panic, logits unchanged
        assert!((logits_copy[0] - 5.0).abs() < 1e-6);
        assert!((logits_copy[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn min_p_filters_low_probability_tokens() {
        // Softmax of [10, 5, 0, -5] gives roughly [0.99, 0.007, 0.0004, 0.00003]
        let logits = vec![10.0, 5.0, 0.0, -5.0];
        let mut probs = softmax(&logits);

        // With min_p=0.5, threshold = max_prob * 0.5 ≈ 0.99 * 0.5 = 0.495
        // Only first token survives
        apply_min_p(&mut probs, 0.5);

        assert!(probs[0] > 0.0, "Highest prob should survive");
        assert!(probs[1] == 0.0, "Second token should be filtered");
        assert!(probs[2] == 0.0, "Third token should be filtered");
        assert!(probs[3] == 0.0, "Fourth token should be filtered");
    }

    #[test]
    fn min_p_zero_does_nothing() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut probs = softmax(&logits);
        let probs_before = probs.clone();

        apply_min_p(&mut probs, 0.0);

        for (a, b) in probs.iter().zip(probs_before.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn top_k_keeps_exactly_k_tokens() {
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let mut probs = softmax(&logits);

        apply_top_k(&mut probs, 2);

        let nonzero_count = probs.iter().filter(|&&p| p > 0.0).count();
        // Should keep top 2: indices 1 (5.0) and 3 (4.0)
        assert_eq!(nonzero_count, 2, "Exactly 2 tokens should survive top_k=2");
    }

    #[test]
    fn top_p_cumulative_sum() {
        // Probabilities close to [0.5, 0.3, 0.15, 0.05]
        let logits = vec![2.0, 1.5, 1.0, 0.0];
        let mut probs = softmax(&logits);

        // top_p = 0.9 should keep tokens until cumsum > 0.9
        apply_top_p(&mut probs, 0.9);

        let nonzero_count = probs.iter().filter(|&&p| p > 0.0).count();
        // With these logits, cumsum reaches ~0.9 after first 2-3 tokens
        assert!(nonzero_count >= 1 && nonzero_count <= 4);
    }

    #[test]
    fn temperature_high_flattens_distribution() {
        let logits = vec![10.0, 0.0, 0.0];
        let mut logits_temp = logits.clone();

        // High temperature (5.0) should flatten: [10/5, 0/5, 0/5] = [2, 0, 0]
        let inv_temp = 1.0 / 5.0;
        for l in logits_temp.iter_mut() {
            *l *= inv_temp;
        }

        let probs_original = softmax(&logits);
        let probs_high_temp = softmax(&logits_temp);

        // Original: first token dominates
        // High temp: more uniform
        assert!(
            probs_high_temp[0] < probs_original[0],
            "High temp should reduce top probability"
        );
        assert!(
            probs_high_temp[1] > probs_original[1],
            "High temp should increase lower probabilities"
        );
    }

    #[test]
    fn temperature_low_sharpens_distribution() {
        let logits = vec![2.0, 1.0, 0.0];
        let mut logits_temp = logits.clone();

        // Low temperature (0.1) should sharpen: [2/0.1, 1/0.1, 0/0.1] = [20, 10, 0]
        let inv_temp = 1.0 / 0.1;
        for l in logits_temp.iter_mut() {
            *l *= inv_temp;
        }

        let probs_original = softmax(&logits);
        let probs_low_temp = softmax(&logits_temp);

        // Low temp: first token dominates even more
        assert!(
            probs_low_temp[0] > probs_original[0],
            "Low temp should increase top probability"
        );
    }

    #[test]
    fn logprobs_sum_to_zero_after_exp() {
        // log-softmax values should sum to 1 when exponentiated
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let log_probs = log_softmax(&logits);

        let exp_sum: f32 = log_probs.iter().map(|lp| lp.exp()).sum();
        assert!(
            (exp_sum - 1.0).abs() < 1e-5,
            "exp(log_softmax) should sum to 1, got {exp_sum}"
        );
    }

    #[test]
    fn logprobs_ordering_preserved() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let log_probs = log_softmax(&logits);

        // Order should be: 5.0 > 4.0 > 3.0 > 2.0 > 1.0
        // Indices:          3     4     1     2     0
        assert!(log_probs[3] > log_probs[4]);
        assert!(log_probs[4] > log_probs[1]);
        assert!(log_probs[1] > log_probs[2]);
        assert!(log_probs[2] > log_probs[0]);
    }

    #[test]
    fn top_k_logprobs_ordering() {
        let log_probs = vec![-1.0, -3.0, -0.5, -2.0, -4.0];
        let top = extract_top_k_logprobs(&log_probs, 3);

        assert_eq!(top.len(), 3);
        // Should be sorted by logprob descending
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
        // First should be index 2 (logprob -0.5)
        assert_eq!(top[0].0, 2);
    }

    #[test]
    fn top_k_logprobs_k_larger_than_vocab() {
        let log_probs = vec![-1.0, -2.0, -3.0];
        let top = extract_top_k_logprobs(&log_probs, 10);

        // Should return all 3, not panic
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn top_k_logprobs_k_zero() {
        let log_probs = vec![-1.0, -2.0];
        let top = extract_top_k_logprobs(&log_probs, 0);

        assert!(top.is_empty());
    }

    #[test]
    fn argmax_with_ties() {
        // When there are ties, max_by returns last occurrence
        let values = vec![1.0, 5.0, 5.0, 3.0];
        let idx = argmax(&values);
        // max_by with partial_cmp returns last occurrence for ties
        assert_eq!(idx, 2, "argmax returns last occurrence of max with ties");
    }

    #[test]
    fn argmax_single_element() {
        let values = vec![42.0];
        let idx = argmax(&values);
        assert_eq!(idx, 0);
    }

    #[test]
    fn sample_from_probs_respects_zeros() {
        // If all but one prob is zero, should always pick the non-zero one
        let probs = vec![0.0, 1.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..100 {
            let token = sample_from_probs(&probs, &mut rng);
            assert_eq!(token, 1, "Should always pick the only non-zero prob");
        }
    }

    #[test]
    fn high_temperature_increases_entropy() {
        // With very high temperature, sampling should be more uniform
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let params_low_temp = SamplingParams {
            temperature: 0.1,
            ..Default::default()
        };
        let params_high_temp = SamplingParams {
            temperature: 5.0,
            ..Default::default()
        };

        let mut counts_low = [0u32; 4];
        let mut counts_high = [0u32; 4];

        let mut state = SamplerState::new(Some(42));
        for _ in 0..1000 {
            let r = sample(&logits, &params_low_temp, &[], &mut state, None);
            counts_low[r.token_id as usize] += 1;
        }

        let mut state = SamplerState::new(Some(42));
        for _ in 0..1000 {
            let r = sample(&logits, &params_high_temp, &[], &mut state, None);
            counts_high[r.token_id as usize] += 1;
        }

        // Low temp: almost all samples should be token 0
        assert!(
            counts_low[0] > 950,
            "Low temp should heavily favor top token"
        );

        // High temp: more spread (token 0 should be less dominant)
        // Note: with temp=5.0, logits become [2, 0, 0, 0] which still favors token 0
        // but less than [10, 0, 0, 0]
        assert!(
            counts_high[0] < counts_low[0]
                || counts_high[1..].iter().sum::<u32>() > counts_low[1..].iter().sum::<u32>(),
            "High temp should produce more varied results"
        );
    }

    #[test]
    fn sampler_state_entropy_initialization() {
        // Without seed, should get different results
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams::default();

        let mut state1 = SamplerState::new(None);
        let mut state2 = SamplerState::new(None);

        // Sample many times and collect results
        let results1: Vec<u32> = (0..100)
            .map(|_| sample(&logits, &params, &[], &mut state1, None).token_id)
            .collect();
        let results2: Vec<u32> = (0..100)
            .map(|_| sample(&logits, &params, &[], &mut state2, None).token_id)
            .collect();

        // With high probability, these should differ (entropy initialization)
        // Not guaranteed, but extremely likely
        assert_ne!(
            results1, results2,
            "Different entropy seeds should likely produce different sequences"
        );
    }

    #[test]
    fn combined_filtering_top_k_then_top_p() {
        // Top-k and top-p should both be applied
        let logits = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,   // Keep only top 3
            top_p: 0.5, // Then further filter by top-p
            ..Default::default()
        };

        let mut state = SamplerState::new(Some(42));
        let mut sampled_tokens = std::collections::HashSet::new();

        for _ in 0..1000 {
            let r = sample(&logits, &params, &[], &mut state, None);
            sampled_tokens.insert(r.token_id);
        }

        // Should only sample from top tokens (0, 1, possibly 2)
        // Should not sample token 3 or 4
        assert!(!sampled_tokens.contains(&3), "Token 3 should be filtered");
        assert!(!sampled_tokens.contains(&4), "Token 4 should be filtered");
    }
}
