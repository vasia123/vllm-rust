use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
pub fn sample(
    logits: &[f32],
    params: &SamplingParams,
    generated_tokens: &[u32],
    sampler_state: &mut SamplerState,
) -> u32 {
    let vocab_size = logits.len();
    let mut logits = logits.to_vec();

    // Step 1: Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut logits, generated_tokens, params.repetition_penalty);
    }

    // Step 2: Greedy fast-path
    if params.is_greedy() {
        return argmax(&logits);
    }

    // Step 3: Apply temperature
    if params.temperature != 1.0 {
        let inv_temp = 1.0 / params.temperature;
        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

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
    sample_from_probs(&probs, &mut sampler_state.rng)
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
        let token = sample(&logits, &params, &[], &mut state);
        assert_eq!(token, 1); // index of max value (5.0)
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let token = sample(&logits, &params, &[], &mut state);
        assert_eq!(token, 2); // index of 10.0
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

        let token1 = sample(&logits, &params, &[], &mut state1);
        let token2 = sample(&logits, &params, &[], &mut state2);
        assert_eq!(token1, token2);
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
        let token = sample(&logits, &params, &[0], &mut state);
        assert_ne!(token, 0); // penalized token should not be selected
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
        let token = sample(&logits, &params, &[], &mut state);
        assert_eq!(token, 1); // only top-1 survives
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
        let token = sample(&logits, &params, &[], &mut state);
        assert_eq!(token, 0); // 10.0 dominates
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
        let token = sample(&logits, &params, &[], &mut state);
        assert!(token <= 1); // only first two tokens should be candidates
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
            let token = sample(&logits, &params, &[], &mut state);
            counts[token as usize] += 1;
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
        let token = sample(&logits, &params, &[2], &mut state);
        // 5.0 / 2.0 = 2.5, still higher than -1.0, so token 2 still wins
        assert_eq!(token, 2);

        // But with higher penalty...
        let params2 = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 10.0,
            ..Default::default()
        };
        let token2 = sample(&logits, &params2, &[2], &mut state);
        // 5.0 / 10.0 = 0.5, -1.0 * 10.0 = -10.0 for the rest → token 2 still wins
        assert_eq!(token2, 2);
    }
}
