//! Sampling and constrained generation infrastructure.
//!
//! This module provides:
//! - Token sampling with temperature, top-k, top-p, and repetition penalty
//! - Constrained generation with choice, regex, and JSON schema constraints
//! - Beam search decoding for higher quality generation

mod beam;
mod constraint;
pub mod gpu;
pub mod grammar;
pub mod logits_processor;

pub use beam::{beam_search_top_k, BeamHypothesis, BeamSearchConfig, BeamSearchState};
pub use constraint::{ChoiceConstraint, JsonSchemaConstraint, RegexConstraint, SamplingConstraint};
pub use logits_processor::{
    AllowedTokenIdsProcessor, BadWordsProcessor, FrequencyPresencePenaltyProcessor,
    LogitBiasProcessor, LogitsProcessor, LogitsProcessorPipeline, MinTokensProcessor,
    NoBadWordsProcessor, RepetitionPenaltyProcessor,
};

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
    /// Frequency penalty (OpenAI convention). Range: -2.0 to 2.0.
    /// Penalizes tokens proportional to their occurrence count in generated_tokens.
    /// 0.0 = disabled.
    pub frequency_penalty: f32,
    /// Presence penalty (OpenAI convention). Range: -2.0 to 2.0.
    /// Applies a one-time penalty to any token that has appeared in generated_tokens.
    /// 0.0 = disabled.
    pub presence_penalty: f32,
    /// Minimum probability relative to max. 0.0 = disabled.
    pub min_p: f32,
    /// Optional seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Beam search configuration. None = disabled (use standard sampling).
    pub beam_search: Option<BeamSearchConfig>,
    /// Token logit bias: (token_id, bias_value) pairs. Bias is added to the
    /// raw logit before temperature scaling. Values follow OpenAI convention
    /// (-100.0 to 100.0). None = no bias.
    pub logit_bias: Option<Vec<(u32, f32)>>,
    /// Minimum number of tokens to generate before allowing EOS.
    /// Requires `eos_token_id` to be set. 0 = disabled.
    pub min_tokens: usize,
    /// EOS token ID for min_tokens enforcement. Required when min_tokens > 0.
    pub eos_token_id: Option<u32>,
    /// Token IDs that are banned from generation. These tokens will have
    /// their logits set to -inf at every step.
    pub banned_token_ids: Option<Vec<u32>>,
    /// If set, only these token IDs are allowed in generation. All other
    /// tokens will have their logits set to -inf.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Bad words as token ID sequences. Each inner Vec is a token sequence;
    /// single-token sequences are banned unconditionally, multi-token sequences
    /// ban the last token only when generated tokens end with the prefix.
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    /// Typical sampling threshold (0..1). 1.0 = disabled.
    /// Keeps tokens whose information content `|log(p_i) - entropy|` is small,
    /// accumulating probability mass until `typical_p` is reached.
    pub typical_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_p: 0.0,
            seed: None,
            beam_search: None,
            logit_bias: None,
            min_tokens: 0,
            eos_token_id: None,
            banned_token_ids: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            typical_p: 1.0,
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

    pub fn is_beam_search(&self) -> bool {
        self.beam_search.is_some()
    }

    /// Whether this sequence can use the GPU sampling fast path.
    ///
    /// GPU sampling handles greedy (argmax) and standard top-k/top-p multinomial.
    /// Sequences requiring penalties, logit bias, min_p, typical_p, constraints,
    /// banned/allowed tokens, or bad words must fall back to CPU sampling.
    pub fn gpu_eligible(&self) -> bool {
        self.repetition_penalty == 1.0
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
            && self.min_p == 0.0
            && self.typical_p == 1.0
            && self.logit_bias.is_none()
            && self.banned_token_ids.is_none()
            && self.allowed_token_ids.is_none()
            && self.beam_search.is_none()
    }

    /// Create sampling params configured for beam search.
    pub fn beam_search(beam_width: usize) -> Self {
        Self {
            beam_search: Some(BeamSearchConfig {
                beam_width,
                ..Default::default()
            }),
            ..Default::default()
        }
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

    /// Access the underlying RNG for use in rejection sampling.
    pub fn rng_mut(&mut self) -> &mut StdRng {
        &mut self.rng
    }

    /// Generate a uniform random f32 in [0, 1) for GPU sampling.
    pub fn next_rand_f32(&mut self) -> f32 {
        use rand::Rng;
        self.rng.gen::<f32>()
    }
}

/// Draw N independent samples from the same logit distribution.
///
/// For greedy decoding (temperature ~0), all N results will be identical (the argmax).
/// For stochastic sampling, each draw is independent using the same `SamplerState` RNG,
/// so successive samples are different but reproducible given the same seed.
///
/// This avoids redundant logit processing (penalties, temperature, top-k/top-p filtering)
/// by computing the probability distribution once and sampling from it N times.
pub fn sample_n(
    logits: &[f32],
    params: &SamplingParams,
    generated_tokens: &[u32],
    sampler_state: &mut SamplerState,
    num_top_logprobs: Option<usize>,
    n: usize,
    stop_token_ids: &[u32],
) -> Vec<SamplingResult> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![sample(
            logits,
            params,
            generated_tokens,
            sampler_state,
            num_top_logprobs,
            stop_token_ids,
        )];
    }

    // For greedy decoding, all N results are identical
    if params.is_greedy() {
        let result = sample(
            logits,
            params,
            generated_tokens,
            sampler_state,
            num_top_logprobs,
            stop_token_ids,
        );
        return vec![result; n];
    }

    // Stochastic path: compute the distribution once, sample N times
    let vocab_size = logits.len();
    let mut logits = logits.to_vec();

    // Apply penalties
    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut logits, generated_tokens, params.repetition_penalty);
    }
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        apply_frequency_presence_penalty(
            &mut logits,
            generated_tokens,
            params.frequency_penalty,
            params.presence_penalty,
        );
    }
    if let Some(ref bias) = params.logit_bias {
        apply_logit_bias(&mut logits, bias);
    }
    if let Some(ref banned) = params.banned_token_ids {
        if !banned.is_empty() {
            apply_banned_tokens(&mut logits, banned);
        }
    }
    if let Some(ref allowed) = params.allowed_token_ids {
        if !allowed.is_empty() {
            apply_allowed_tokens(&mut logits, allowed);
        }
    }
    if let Some(ref bad_words) = params.bad_words_token_ids {
        if !bad_words.is_empty() {
            apply_bad_words(&mut logits, bad_words, generated_tokens);
        }
    }
    if params.min_tokens > 0 {
        if let Some(eos_id) = params.eos_token_id {
            apply_min_tokens_suppression(
                &mut logits,
                eos_id,
                stop_token_ids,
                params.min_tokens,
                generated_tokens.len(),
            );
        }
    }

    // Apply temperature
    if params.temperature != 1.0 {
        let inv_temp = 1.0 / params.temperature;
        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    // Compute log-softmax for logprob extraction
    let log_probs = if num_top_logprobs.is_some() {
        Some(log_softmax(&logits))
    } else {
        None
    };

    // Convert to probabilities and apply filtering
    let mut probs = softmax(&logits);

    if params.min_p > 0.0 {
        apply_min_p(&mut probs, params.min_p);
    }
    if params.top_k > 0 && (params.top_k as usize) < vocab_size {
        apply_top_k(&mut probs, params.top_k as usize);
    }
    if params.top_p < 1.0 && params.top_p > 0.0 {
        apply_top_p(&mut probs, params.top_p);
    }
    if params.typical_p < 1.0 && params.typical_p > 0.0 {
        apply_typical_p(&mut probs, params.typical_p);
    }

    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum != 1.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }

    // Pre-compute top-k logprobs (shared across all samples)
    let shared_top_logprobs = log_probs
        .as_ref()
        .and_then(|lp| num_top_logprobs.map(|k| extract_top_k_logprobs(lp, k)));

    // Sample N times from the same distribution
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let token_id = sample_from_probs(&probs, &mut sampler_state.rng);

        let (logprob, top_logprobs) = if let Some(ref lp) = log_probs {
            (lp[token_id as usize], shared_top_logprobs.clone())
        } else {
            (f32::NEG_INFINITY, None)
        };

        results.push(SamplingResult {
            token_id,
            logprob,
            top_logprobs,
        });
    }

    results
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
    stop_token_ids: &[u32],
) -> SamplingResult {
    let vocab_size = logits.len();
    let mut logits = logits.to_vec();

    // Step 1: Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut logits, generated_tokens, params.repetition_penalty);
    }

    // Step 1b: Apply frequency and presence penalties (OpenAI convention)
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        apply_frequency_presence_penalty(
            &mut logits,
            generated_tokens,
            params.frequency_penalty,
            params.presence_penalty,
        );
    }

    // Step 1c: Apply logit bias
    if let Some(ref bias) = params.logit_bias {
        apply_logit_bias(&mut logits, bias);
    }

    // Step 1d: Apply banned token IDs
    if let Some(ref banned) = params.banned_token_ids {
        if !banned.is_empty() {
            apply_banned_tokens(&mut logits, banned);
        }
    }

    // Step 1e: Apply allowed token IDs (whitelist)
    if let Some(ref allowed) = params.allowed_token_ids {
        if !allowed.is_empty() {
            apply_allowed_tokens(&mut logits, allowed);
        }
    }

    // Step 1e2: Apply bad words (multi-token sequence banning)
    if let Some(ref bad_words) = params.bad_words_token_ids {
        if !bad_words.is_empty() {
            apply_bad_words(&mut logits, bad_words, generated_tokens);
        }
    }

    // Step 1f: Suppress EOS and stop tokens until min_tokens generated
    if params.min_tokens > 0 {
        if let Some(eos_id) = params.eos_token_id {
            apply_min_tokens_suppression(
                &mut logits,
                eos_id,
                stop_token_ids,
                params.min_tokens,
                generated_tokens.len(),
            );
        }
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

    // Step 7b: Apply typical_p filtering
    if params.typical_p < 1.0 && params.typical_p > 0.0 {
        apply_typical_p(&mut probs, params.typical_p);
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

pub(crate) fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
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

/// Apply frequency and presence penalties following the OpenAI convention.
///
/// For each token that appears in `generated_tokens`:
///   logits[token_id] -= frequency_penalty * count + presence_penalty * 1.0
///
/// Where `count` is the number of times the token appears in `generated_tokens`.
pub(crate) fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    generated_tokens: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    // Count occurrences of each token
    let mut counts = std::collections::HashMap::<u32, u32>::new();
    for &token_id in generated_tokens {
        *counts.entry(token_id).or_insert(0) += 1;
    }

    for (&token_id, &count) in &counts {
        let idx = token_id as usize;
        if idx < logits.len() {
            logits[idx] -= frequency_penalty * count as f32
                + presence_penalty * if count > 0 { 1.0 } else { 0.0 };
        }
    }
}

pub(crate) fn apply_logit_bias(logits: &mut [f32], bias: &[(u32, f32)]) {
    for &(token_id, bias_value) in bias {
        let idx = token_id as usize;
        if idx < logits.len() {
            logits[idx] += bias_value;
        }
    }
}

pub(crate) fn apply_banned_tokens(logits: &mut [f32], banned: &[u32]) {
    for &token_id in banned {
        let idx = token_id as usize;
        if idx < logits.len() {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

pub(crate) fn apply_allowed_tokens(logits: &mut [f32], allowed: &[u32]) {
    for (idx, logit) in logits.iter_mut().enumerate() {
        if !allowed.contains(&(idx as u32)) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

pub(crate) fn apply_bad_words(
    logits: &mut [f32],
    bad_words: &[Vec<u32>],
    generated_tokens: &[u32],
) {
    for word_ids in bad_words {
        if word_ids.is_empty() {
            continue;
        }
        if word_ids.len() == 1 {
            let idx = word_ids[0] as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
            continue;
        }
        let prefix_len = word_ids.len() - 1;
        if prefix_len > generated_tokens.len() {
            continue;
        }
        let actual_prefix = &generated_tokens[generated_tokens.len() - prefix_len..];
        let expected_prefix = &word_ids[..prefix_len];
        if actual_prefix == expected_prefix {
            let last_token = *word_ids.last().unwrap() as usize;
            if last_token < logits.len() {
                logits[last_token] = f32::NEG_INFINITY;
            }
        }
    }
}

pub(crate) fn apply_min_tokens_suppression(
    logits: &mut [f32],
    eos_id: u32,
    stop_token_ids: &[u32],
    min_tokens: usize,
    generated_len: usize,
) {
    if generated_len < min_tokens {
        let eos_idx = eos_id as usize;
        if eos_idx < logits.len() {
            logits[eos_idx] = f32::NEG_INFINITY;
        }
        for &stop_id in stop_token_ids {
            let idx = stop_id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
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

/// Typical sampling: keep tokens whose information content is close to the
/// distribution entropy, accumulating until `typical_p` probability mass.
///
/// Reference: Meister et al. "Typical Decoding for Natural Language Generation"
/// https://arxiv.org/abs/2202.00666
fn apply_typical_p(probs: &mut [f32], typical_p: f32) {
    // Compute entropy H = -sum(p * log(p))
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Compute |log(p_i) - (-H)| = |log(p_i) + H| for each token
    // Smaller deviation = more "typical"
    let mut indexed: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(i, &p)| {
            let deviation = (p.ln() + entropy).abs();
            (i, p, deviation)
        })
        .collect();

    // Sort by deviation ascending (most typical first)
    indexed.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate probability until typical_p threshold
    let mut cumsum = 0.0f32;
    let mut keep = std::collections::HashSet::new();
    for &(idx, p, _) in &indexed {
        cumsum += p;
        keep.insert(idx);
        if cumsum >= typical_p {
            break;
        }
    }

    // Zero out non-typical tokens
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
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
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
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
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
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

        let result1 = sample(&logits, &params, &[], &mut state1, None, &[]);
        let result2 = sample(&logits, &params, &[], &mut state2, None, &[]);
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
        let result = sample(&logits, &params, &[0], &mut state, None, &[]);
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
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
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
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
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
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
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
        assert!(params.logit_bias.is_none());
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

        let mut counts = [0u32; 10];
        let mut state = SamplerState::new(Some(0));
        for _ in 0..1000 {
            let result = sample(&logits, &params, &[], &mut state, None, &[]);
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
        let result = sample(&logits, &params, &[2], &mut state, None, &[]);
        // 5.0 / 2.0 = 2.5, still higher than -1.0, so token 2 still wins
        assert_eq!(result.token_id, 2);

        // But with higher penalty...
        let params2 = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 10.0,
            ..Default::default()
        };
        let result2 = sample(&logits, &params2, &[2], &mut state, None, &[]);
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
        let result = sample(&logits, &params, &[], &mut state, Some(3), &[]);

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
        let result = sample(&logits, &params, &[], &mut state, Some(5), &[]);

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
        let result = sample(&logits, &params, &[], &mut state, None, &[]);

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
        assert!((1..=4).contains(&nonzero_count));
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
            let r = sample(&logits, &params_low_temp, &[], &mut state, None, &[]);
            counts_low[r.token_id as usize] += 1;
        }

        let mut state = SamplerState::new(Some(42));
        for _ in 0..1000 {
            let r = sample(&logits, &params_high_temp, &[], &mut state, None, &[]);
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
            .map(|_| sample(&logits, &params, &[], &mut state1, None, &[]).token_id)
            .collect();
        let results2: Vec<u32> = (0..100)
            .map(|_| sample(&logits, &params, &[], &mut state2, None, &[]).token_id)
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
            let r = sample(&logits, &params, &[], &mut state, None, &[]);
            sampled_tokens.insert(r.token_id);
        }

        // Should only sample from top tokens (0, 1, possibly 2)
        // Should not sample token 3 or 4
        assert!(!sampled_tokens.contains(&3), "Token 3 should be filtered");
        assert!(!sampled_tokens.contains(&4), "Token 4 should be filtered");
    }

    // ==========================================================================
    // Logit bias tests
    // ==========================================================================

    #[test]
    fn logit_bias_increases_token_probability() {
        // Token 2 has the lowest logit; a large positive bias should make it the argmax
        let logits = vec![5.0, 4.0, 1.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            logit_bias: Some(vec![(2, 50.0)]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        assert_eq!(
            result.token_id, 2,
            "Positive bias should boost token 2 to top"
        );
    }

    #[test]
    fn logit_bias_decreases_token_probability() {
        // Token 0 has the highest logit; a large negative bias should prevent selection
        let logits = vec![10.0, 5.0, 5.0, 5.0];
        let params = SamplingParams {
            temperature: 0.0,
            logit_bias: Some(vec![(0, -20.0)]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        assert_ne!(
            result.token_id, 0,
            "Negative bias should prevent token 0 from being selected"
        );
    }

    #[test]
    fn logit_bias_large_negative_effectively_bans_token() {
        // -100 bias on the would-be argmax should make it essentially impossible
        let logits = vec![10.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            logit_bias: Some(vec![(0, -100.0)]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Sample many times; token 0 should never appear
        for _ in 0..1000 {
            let result = sample(&logits, &params, &[], &mut state, None, &[]);
            assert_ne!(
                result.token_id, 0,
                "Token 0 should be effectively banned with -100 bias"
            );
        }
    }

    #[test]
    fn logit_bias_none_is_identity() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params_no_bias = SamplingParams {
            temperature: 0.0,
            logit_bias: None,
            ..Default::default()
        };
        let params_default = SamplingParams::greedy();

        let mut state1 = SamplerState::new(Some(42));
        let mut state2 = SamplerState::new(Some(42));

        let r1 = sample(&logits, &params_no_bias, &[], &mut state1, None, &[]);
        let r2 = sample(&logits, &params_default, &[], &mut state2, None, &[]);
        assert_eq!(
            r1.token_id, r2.token_id,
            "None logit_bias should behave identically to default"
        );
    }

    #[test]
    fn logit_bias_out_of_range_token_ignored() {
        let logits = vec![1.0, 5.0, 3.0];
        let mut logits_copy = logits.clone();
        // Token ID 9999 is far beyond vocab_size=3
        apply_logit_bias(&mut logits_copy, &[(9999, 50.0)]);

        // Logits should be unchanged
        assert!((logits_copy[0] - 1.0).abs() < 1e-6);
        assert!((logits_copy[1] - 5.0).abs() < 1e-6);
        assert!((logits_copy[2] - 3.0).abs() < 1e-6);
    }

    // ==========================================================================
    // Frequency and presence penalty tests
    // ==========================================================================

    #[test]
    fn frequency_penalty_reduces_repeated_tokens() {
        // Token 0 appears 3 times, token 1 appears 1 time.
        // With frequency_penalty, token 0 should be penalized more than token 1.
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let mut logits_copy = logits.clone();
        let generated = vec![0, 0, 0, 1];

        apply_frequency_presence_penalty(&mut logits_copy, &generated, 1.0, 0.0);

        // Token 0: 5.0 - 1.0 * 3 = 2.0
        assert!(
            (logits_copy[0] - 2.0).abs() < 1e-6,
            "Token 0 should be penalized by freq_pen * 3 = 3.0, got {}",
            logits_copy[0]
        );
        // Token 1: 5.0 - 1.0 * 1 = 4.0
        assert!(
            (logits_copy[1] - 4.0).abs() < 1e-6,
            "Token 1 should be penalized by freq_pen * 1 = 1.0, got {}",
            logits_copy[1]
        );
        // Token 2: unchanged (never generated)
        assert!(
            (logits_copy[2] - 5.0).abs() < 1e-6,
            "Token 2 should be unchanged"
        );
        // Token 3: unchanged (never generated)
        assert!(
            (logits_copy[3] - 5.0).abs() < 1e-6,
            "Token 3 should be unchanged"
        );

        // More frequent token penalized more -> lower logit
        assert!(
            logits_copy[0] < logits_copy[1],
            "Token with count 3 should have lower logit than token with count 1"
        );
    }

    #[test]
    fn presence_penalty_applies_once_per_token() {
        // Token 0 appears 5 times, token 1 appears 1 time.
        // With only presence_penalty, both should be penalized equally (once).
        let logits = vec![5.0, 5.0, 5.0];
        let mut logits_copy = logits.clone();
        let generated = vec![0, 0, 0, 0, 0, 1];

        apply_frequency_presence_penalty(&mut logits_copy, &generated, 0.0, 1.5);

        // Token 0: 5.0 - 0.0 * 5 - 1.5 * 1.0 = 3.5
        assert!(
            (logits_copy[0] - 3.5).abs() < 1e-6,
            "Token 0 should be penalized by pres_pen once = 1.5, got {}",
            logits_copy[0]
        );
        // Token 1: 5.0 - 0.0 * 1 - 1.5 * 1.0 = 3.5
        assert!(
            (logits_copy[1] - 3.5).abs() < 1e-6,
            "Token 1 should be penalized by pres_pen once = 1.5, got {}",
            logits_copy[1]
        );
        // Token 2: unchanged (never generated)
        assert!(
            (logits_copy[2] - 5.0).abs() < 1e-6,
            "Token 2 should be unchanged"
        );
    }

    #[test]
    fn combined_frequency_presence_penalty() {
        // Token 0 appears 3 times, token 1 appears 1 time.
        // frequency_penalty=0.5, presence_penalty=1.0
        let logits = vec![10.0, 10.0, 10.0];
        let mut logits_copy = logits.clone();
        let generated = vec![0, 0, 0, 1];

        apply_frequency_presence_penalty(&mut logits_copy, &generated, 0.5, 1.0);

        // Token 0: 10.0 - 0.5 * 3 - 1.0 * 1.0 = 10.0 - 1.5 - 1.0 = 7.5
        assert!(
            (logits_copy[0] - 7.5).abs() < 1e-6,
            "Token 0 expected 7.5, got {}",
            logits_copy[0]
        );
        // Token 1: 10.0 - 0.5 * 1 - 1.0 * 1.0 = 10.0 - 0.5 - 1.0 = 8.5
        assert!(
            (logits_copy[1] - 8.5).abs() < 1e-6,
            "Token 1 expected 8.5, got {}",
            logits_copy[1]
        );
        // Token 2: unchanged (never generated)
        assert!(
            (logits_copy[2] - 10.0).abs() < 1e-6,
            "Token 2 should be unchanged"
        );
    }

    #[test]
    fn zero_penalties_are_identity() {
        let logits = vec![3.0, 7.0, 1.0, 5.0];
        let mut logits_copy = logits.clone();
        let generated = vec![0, 1, 1, 2, 0, 0];

        apply_frequency_presence_penalty(&mut logits_copy, &generated, 0.0, 0.0);

        for (orig, modified) in logits.iter().zip(logits_copy.iter()) {
            assert!(
                (orig - modified).abs() < 1e-6,
                "Zero penalties should not change logits"
            );
        }
    }

    #[test]
    fn negative_penalties_encourage_repetition() {
        // Negative frequency_penalty should increase logits for repeated tokens
        let logits = vec![5.0, 5.0, 5.0];
        let mut logits_copy = logits.clone();
        let generated = vec![0, 0, 1];

        apply_frequency_presence_penalty(&mut logits_copy, &generated, -1.0, 0.0);

        // Token 0: 5.0 - (-1.0) * 2 = 5.0 + 2.0 = 7.0
        assert!(
            (logits_copy[0] - 7.0).abs() < 1e-6,
            "Negative freq penalty should increase logit, got {}",
            logits_copy[0]
        );
        // Token 1: 5.0 - (-1.0) * 1 = 5.0 + 1.0 = 6.0
        assert!(
            (logits_copy[1] - 6.0).abs() < 1e-6,
            "Negative freq penalty should increase logit, got {}",
            logits_copy[1]
        );
        // Token 2: unchanged
        assert!(
            (logits_copy[2] - 5.0).abs() < 1e-6,
            "Token 2 should be unchanged"
        );

        // Repeated tokens should now have higher logits
        assert!(
            logits_copy[0] > logits[0],
            "Negative penalty should encourage repeated token"
        );
    }

    #[test]
    fn frequency_presence_penalty_integration_with_sample() {
        // End-to-end: with greedy sampling, frequency penalty should shift
        // the argmax away from the most-repeated token.
        let logits = vec![5.0, 4.9, 4.8, 0.0]; // Token 0 barely leads
        let params = SamplingParams {
            temperature: 0.0,
            frequency_penalty: 2.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Token 0 appeared 3 times -> penalized by 2.0 * 3 = 6.0 -> logit becomes -1.0
        // Token 1 appeared 0 times -> stays at 4.9
        let result = sample(&logits, &params, &[0, 0, 0], &mut state, None, &[]);
        assert_eq!(
            result.token_id, 1,
            "Heavily penalized token 0 should lose to token 1"
        );
    }

    #[test]
    fn default_params_include_zero_penalties() {
        let params = SamplingParams::default();
        assert_eq!(params.frequency_penalty, 0.0);
        assert_eq!(params.presence_penalty, 0.0);
    }

    #[test]
    fn greedy_params_include_zero_penalties() {
        let params = SamplingParams::greedy();
        assert_eq!(params.frequency_penalty, 0.0);
        assert_eq!(params.presence_penalty, 0.0);
    }

    // ==========================================================================
    // sample_n tests
    // ==========================================================================

    #[test]
    fn sample_n_zero_returns_empty() {
        let logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams::default();
        let mut state = SamplerState::new(Some(42));
        let results = sample_n(&logits, &params, &[], &mut state, None, 0, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn sample_n_one_matches_single_sample() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };

        let mut state1 = SamplerState::new(Some(42));
        let mut state2 = SamplerState::new(Some(42));

        let single = sample(&logits, &params, &[], &mut state1, None, &[]);
        let multi = sample_n(&logits, &params, &[], &mut state2, None, 1, &[]);

        assert_eq!(multi.len(), 1);
        assert_eq!(multi[0].token_id, single.token_id);
    }

    #[test]
    fn sample_n_greedy_returns_identical_results() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams::greedy();
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, None, 5, &[]);

        assert_eq!(results.len(), 5);
        // All should be the argmax (token 1)
        for r in &results {
            assert_eq!(
                r.token_id, 1,
                "Greedy sample_n should return argmax for all"
            );
        }
    }

    #[test]
    fn sample_n_stochastic_returns_n_results() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, None, 10, &[]);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn sample_n_stochastic_produces_varied_results() {
        // With uniform logits and enough samples, we should see variety
        let logits = vec![1.0; 5];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, None, 100, &[]);
        let unique: std::collections::HashSet<u32> = results.iter().map(|r| r.token_id).collect();

        assert!(
            unique.len() > 1,
            "Stochastic sample_n should produce varied results with uniform logits"
        );
    }

    #[test]
    fn sample_n_deterministic_with_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };

        let mut state1 = SamplerState::new(Some(123));
        let mut state2 = SamplerState::new(Some(123));

        let results1 = sample_n(&logits, &params, &[], &mut state1, None, 10, &[]);
        let results2 = sample_n(&logits, &params, &[], &mut state2, None, 10, &[]);

        let ids1: Vec<u32> = results1.iter().map(|r| r.token_id).collect();
        let ids2: Vec<u32> = results2.iter().map(|r| r.token_id).collect();
        assert_eq!(ids1, ids2, "Same seed should produce identical sequences");
    }

    #[test]
    fn sample_n_with_logprobs() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams::greedy();
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, Some(3), 3, &[]);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.logprob.is_finite());
            assert!(r.logprob < 0.0);
            let top = r.top_logprobs.as_ref().expect("should have top logprobs");
            assert_eq!(top.len(), 3);
        }
    }

    #[test]
    fn sample_n_stochastic_with_logprobs_shares_top_logprobs() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, Some(2), 3, &[]);

        // Top logprobs should be the same for all samples (computed once from same distribution)
        let top0 = results[0].top_logprobs.as_ref().unwrap();
        for r in &results[1..] {
            let top = r.top_logprobs.as_ref().unwrap();
            assert_eq!(
                top.len(),
                top0.len(),
                "All samples should share the same top logprobs"
            );
            for (a, b) in top0.iter().zip(top.iter()) {
                assert_eq!(a.0, b.0, "Top logprob token IDs should match");
                assert!((a.1 - b.1).abs() < 1e-6, "Top logprob values should match");
            }
        }
    }

    #[test]
    fn sample_n_applies_repetition_penalty() {
        // Token 0 is the argmax, but with repetition penalty and greedy,
        // we want to confirm penalty is applied
        let logits = vec![5.0, 4.9, 4.8, 0.0];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Token 0 was already generated, should be penalized: 5.0 / 2.0 = 2.5
        // Token 1 should now be argmax at 4.9
        let results = sample_n(&logits, &params, &[0], &mut state, None, 3, &[]);
        for r in &results {
            assert_eq!(
                r.token_id, 1,
                "With penalty on token 0, token 1 should be argmax"
            );
        }
    }

    #[test]
    fn sample_n_applies_top_k() {
        // With top_k=1, only the max should survive
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, None, 10, &[]);
        for r in &results {
            assert_eq!(r.token_id, 1, "top_k=1 should only allow the argmax");
        }
    }

    // ==========================================================================
    // Banned / allowed token IDs and min_tokens inline tests
    // ==========================================================================

    #[test]
    fn banned_tokens_suppresses_in_sample() {
        // Token 1 is the greedy argmax (5.0), but ban it
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            banned_token_ids: Some(vec![1]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        // Token 1 banned → next best is token 2 (3.0)
        assert_eq!(result.token_id, 2, "Banned token should not be selected");
    }

    #[test]
    fn banned_tokens_multiple() {
        let logits = vec![1.0, 5.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            banned_token_ids: Some(vec![1, 2]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        // Tokens 1, 2 banned → next best is token 3 (3.0)
        assert_eq!(result.token_id, 3);
    }

    #[test]
    fn banned_tokens_empty_is_noop() {
        let logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            banned_token_ids: Some(vec![]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        assert_eq!(result.token_id, 1);
    }

    #[test]
    fn allowed_tokens_restricts_in_sample() {
        // Only allow tokens 0 and 3
        let logits = vec![1.0, 5.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            allowed_token_ids: Some(vec![0, 3]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        // Only tokens 0 (1.0) and 3 (3.0) allowed → greedy picks 3
        assert_eq!(result.token_id, 3);
    }

    #[test]
    fn allowed_tokens_single() {
        let logits = vec![1.0, 5.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 1.0,
            allowed_token_ids: Some(vec![2]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        for _ in 0..100 {
            let result = sample(&logits, &params, &[], &mut state, None, &[]);
            assert_eq!(result.token_id, 2, "Only token 2 should be allowed");
        }
    }

    #[test]
    fn min_tokens_suppresses_eos_early() {
        // EOS token (id=1) is the greedy argmax
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            min_tokens: 5,
            eos_token_id: Some(1),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Only 2 tokens generated so far (< min_tokens=5) → EOS suppressed
        let result = sample(&logits, &params, &[10, 20], &mut state, None, &[]);
        assert_ne!(
            result.token_id, 1,
            "EOS should be suppressed before min_tokens"
        );
        assert_eq!(result.token_id, 2, "Next best after suppressed EOS");
    }

    #[test]
    fn min_tokens_allows_eos_after_threshold() {
        // After enough tokens, EOS should be allowed
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            min_tokens: 3,
            eos_token_id: Some(1),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // 3 tokens already generated (= min_tokens) → EOS allowed
        let result = sample(&logits, &params, &[10, 20, 30], &mut state, None, &[]);
        assert_eq!(
            result.token_id, 1,
            "EOS should be allowed once min_tokens reached"
        );
    }

    #[test]
    fn min_tokens_without_eos_id_is_noop() {
        let logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            min_tokens: 10,
            eos_token_id: None,
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        assert_eq!(
            result.token_id, 1,
            "Without eos_token_id, min_tokens has no effect"
        );
    }

    #[test]
    fn sample_n_applies_banned_tokens() {
        let logits = vec![1.0, 5.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 1.0,
            banned_token_ids: Some(vec![1]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, None, 50, &[]);
        for r in &results {
            assert_ne!(
                r.token_id, 1,
                "Banned token should never appear in sample_n"
            );
        }
    }

    #[test]
    fn sample_n_applies_allowed_tokens() {
        let logits = vec![1.0, 5.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 1.0,
            allowed_token_ids: Some(vec![0, 2]),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        let results = sample_n(&logits, &params, &[], &mut state, None, 50, &[]);
        for r in &results {
            assert!(
                r.token_id == 0 || r.token_id == 2,
                "Only allowed tokens should appear, got {}",
                r.token_id
            );
        }
    }

    #[test]
    fn sample_n_applies_min_tokens() {
        // EOS is token 1 (highest logit)
        let logits = vec![1.0, 5.0, 4.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            min_tokens: 5,
            eos_token_id: Some(1),
            ..Default::default()
        };
        let mut state = SamplerState::new(Some(42));

        // Only 1 token generated → EOS suppressed
        let results = sample_n(&logits, &params, &[10], &mut state, None, 3, &[]);
        for r in &results {
            assert_ne!(r.token_id, 1, "EOS should be suppressed in sample_n too");
        }
    }

    #[test]
    fn apply_banned_tokens_helper() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_banned_tokens(&mut logits, &[1, 3]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn apply_banned_tokens_out_of_range() {
        let mut logits = vec![1.0, 2.0];
        apply_banned_tokens(&mut logits, &[999]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
    }

    #[test]
    fn apply_allowed_tokens_helper() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_allowed_tokens(&mut logits, &[1, 3]);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
    }

    #[test]
    fn apply_min_tokens_suppression_helper() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_min_tokens_suppression(&mut logits, 1, &[], 5, 3); // 3 < 5 → suppress
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn apply_min_tokens_suppression_allows_after_threshold() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_min_tokens_suppression(&mut logits, 1, &[], 5, 5); // 5 >= 5 → no suppression
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn apply_min_tokens_suppression_with_stop_token_ids() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // EOS=1, stop_token_ids=[3, 4], min_tokens=5, generated=2 → suppress all
        apply_min_tokens_suppression(&mut logits, 1, &[3, 4], 5, 2);
        assert_eq!(logits[0], 1.0); // not a stop token
        assert_eq!(logits[1], f32::NEG_INFINITY); // EOS
        assert_eq!(logits[2], 3.0); // not a stop token
        assert_eq!(logits[3], f32::NEG_INFINITY); // stop token
        assert_eq!(logits[4], f32::NEG_INFINITY); // stop token
    }

    #[test]
    fn apply_min_tokens_suppression_stop_tokens_allowed_after_threshold() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // EOS=1, stop_token_ids=[3, 4], min_tokens=3, generated=3 → no suppression
        apply_min_tokens_suppression(&mut logits, 1, &[3, 4], 3, 3);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0); // EOS not suppressed
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 4.0); // stop token not suppressed
        assert_eq!(logits[4], 5.0); // stop token not suppressed
    }

    #[test]
    fn apply_bad_words_single_token() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_bad_words(&mut logits, &[vec![1], vec![3]], &[]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn apply_bad_words_multi_token_match() {
        let mut logits = vec![0.0; 10];
        logits[7] = 5.0;
        // Bad word [5, 6, 7]: prefix [5, 6] matches tail of generated
        apply_bad_words(&mut logits, &[vec![5, 6, 7]], &[1, 2, 5, 6]);
        assert_eq!(logits[7], f32::NEG_INFINITY);
    }

    #[test]
    fn apply_bad_words_multi_token_no_match() {
        let mut logits = vec![0.0; 10];
        logits[7] = 5.0;
        // Bad word [5, 6, 7]: prefix [5, 6] does NOT match tail of generated
        apply_bad_words(&mut logits, &[vec![5, 6, 7]], &[1, 2, 5, 9]);
        assert_eq!(logits[7], 5.0);
    }

    #[test]
    fn sample_with_bad_words_token_ids() {
        let params = SamplingParams {
            bad_words_token_ids: Some(vec![vec![1], vec![2, 3]]),
            ..Default::default()
        };
        let mut state = SamplerState::new(None);

        // Token 1 should be unconditionally banned
        let logits = vec![0.0, 100.0, 0.0, 0.0];
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        assert_ne!(result.token_id, 1);

        // Token 3 should be banned when last generated token is 2
        let logits = vec![0.0, 0.0, 0.0, 100.0];
        let result = sample(&logits, &params, &[2], &mut state, None, &[]);
        assert_ne!(result.token_id, 3);
    }

    // ---- typical_p tests ----

    #[test]
    fn test_apply_typical_p_disabled() {
        // typical_p = 1.0 should not be called (disabled), but if called keeps all tokens
        let mut probs = vec![0.25, 0.25, 0.25, 0.25];
        apply_typical_p(&mut probs, 1.0);
        // All tokens are equally typical; cumsum reaches 1.0 only after all
        let nonzero: usize = probs.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero, 4);
    }

    #[test]
    fn test_apply_typical_p_filters_outliers() {
        // Distribution: one dominant token, rest low
        // Token 0: 0.9, Token 1: 0.05, Token 2: 0.03, Token 3: 0.02
        let mut probs = vec![0.9, 0.05, 0.03, 0.02];
        apply_typical_p(&mut probs, 0.5);
        // With typical_p=0.5, only the most typical token(s) should remain
        let nonzero: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .collect();
        assert!(!nonzero.is_empty());
        // Sum of remaining should be >= 0.5 of original cumulative
        let remaining_sum: f32 = probs.iter().sum();
        assert!(remaining_sum >= 0.5 * 0.99); // allowing for float tolerance
    }

    #[test]
    fn test_apply_typical_p_uniform_keeps_all() {
        // Uniform distribution: all tokens equally typical
        let mut probs = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        apply_typical_p(&mut probs, 0.9);
        // All tokens have same deviation, cumsum grows linearly
        let nonzero: usize = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(nonzero >= 4); // need at least 4 to reach 0.8 ≥ 0.9 might need 5
    }

    #[test]
    fn test_apply_typical_p_preserves_zeros() {
        // Tokens with zero probability stay zero
        let mut probs = vec![0.5, 0.0, 0.3, 0.0, 0.2];
        apply_typical_p(&mut probs, 0.9);
        assert_eq!(probs[1], 0.0);
        assert_eq!(probs[3], 0.0);
    }

    #[test]
    fn test_typical_p_sampling_integration() {
        // End-to-end: sample with typical_p should still pick a valid token
        let logits = vec![10.0, 0.0, -5.0, -10.0];
        let params = SamplingParams {
            temperature: 1.0,
            typical_p: 0.5,
            ..Default::default()
        };
        let mut state = SamplerState::new(None);
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        assert!(result.token_id < 4);
    }

    #[test]
    fn test_typical_p_greedy_still_works() {
        // typical_p with temperature 0 (greedy): typical_p is applied after softmax
        // but before greedy argmax
        let logits = vec![10.0, 0.0, -5.0, -10.0];
        let params = SamplingParams {
            temperature: 0.0,
            typical_p: 0.9,
            ..Default::default()
        };
        let mut state = SamplerState::new(None);
        let result = sample(&logits, &params, &[], &mut state, None, &[]);
        // Token 0 has highest probability, should still win
        assert_eq!(result.token_id, 0);
    }

    #[test]
    fn gpu_eligible_greedy() {
        let params = SamplingParams::greedy();
        assert!(params.gpu_eligible());
    }

    #[test]
    fn gpu_eligible_standard_stochastic() {
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            ..Default::default()
        };
        assert!(params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_repetition_penalty() {
        let params = SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_frequency_penalty() {
        let params = SamplingParams {
            frequency_penalty: 0.5,
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_presence_penalty() {
        let params = SamplingParams {
            presence_penalty: 0.5,
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_min_p() {
        let params = SamplingParams {
            min_p: 0.1,
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_typical_p() {
        let params = SamplingParams {
            typical_p: 0.9,
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_logit_bias() {
        let params = SamplingParams {
            logit_bias: Some(vec![(42, 5.0)]),
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_banned_tokens() {
        let params = SamplingParams {
            banned_token_ids: Some(vec![42]),
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_allowed_tokens() {
        let params = SamplingParams {
            allowed_token_ids: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn gpu_ineligible_with_beam_search() {
        let params = SamplingParams::beam_search(4);
        assert!(!params.gpu_eligible());
    }

    #[test]
    fn next_rand_f32_in_range() {
        let mut state = SamplerState::new(Some(42));
        for _ in 0..100 {
            let val = state.next_rand_f32();
            assert!((0.0..1.0).contains(&val), "rand_f32 out of range: {val}");
        }
    }

    #[test]
    fn next_rand_f32_deterministic_with_seed() {
        let mut state1 = SamplerState::new(Some(123));
        let mut state2 = SamplerState::new(Some(123));
        for _ in 0..10 {
            assert_eq!(state1.next_rand_f32(), state2.next_rand_f32());
        }
    }
}
