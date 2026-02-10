//! Extensible logits processor pipeline.
//!
//! Each processor applies a transformation to the raw logit vector before sampling.
//! Processors execute in order: penalty processors first, then filtering processors.
//!
//! # Usage
//! ```ignore
//! let mut pipeline = LogitsProcessorPipeline::from_params(&params, generated_tokens);
//! pipeline.process(&mut logits);
//! ```

/// Trait for logit transformations applied before sampling.
///
/// Implementors modify the logits slice in place. The slice has length `vocab_size`
/// and contains raw (unnormalized) log-probabilities.
pub trait LogitsProcessor: Send + Sync {
    /// Apply this processor's transformation to the logit vector.
    fn process(&self, logits: &mut [f32], generated_tokens: &[u32]);

    /// Human-readable name for debugging and metrics.
    fn name(&self) -> &'static str;
}

/// Repetition penalty processor (Ctrl-style).
///
/// Discourages repeated tokens by dividing positive logits and multiplying
/// negative logits by the penalty factor.
pub struct RepetitionPenaltyProcessor {
    penalty: f32,
}

impl RepetitionPenaltyProcessor {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        for &token_id in generated_tokens {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Frequency and presence penalty processor (OpenAI convention).
///
/// - Frequency penalty: proportional to token count in generated text
/// - Presence penalty: one-time penalty for any previously-seen token
pub struct FrequencyPresencePenaltyProcessor {
    frequency_penalty: f32,
    presence_penalty: f32,
}

impl FrequencyPresencePenaltyProcessor {
    pub fn new(frequency_penalty: f32, presence_penalty: f32) -> Self {
        Self {
            frequency_penalty,
            presence_penalty,
        }
    }
}

impl LogitsProcessor for FrequencyPresencePenaltyProcessor {
    fn process(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        // Count token frequencies
        let mut counts: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::with_capacity(generated_tokens.len());
        for &token_id in generated_tokens {
            *counts.entry(token_id).or_insert(0) += 1;
        }

        for (&token_id, &count) in &counts {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] -= self.frequency_penalty * count as f32 + self.presence_penalty;
            }
        }
    }

    fn name(&self) -> &'static str {
        "frequency_presence_penalty"
    }
}

/// Logit bias processor — adds per-token bias values.
pub struct LogitBiasProcessor {
    biases: Vec<(u32, f32)>,
}

impl LogitBiasProcessor {
    pub fn new(biases: Vec<(u32, f32)>) -> Self {
        Self { biases }
    }
}

impl LogitsProcessor for LogitBiasProcessor {
    fn process(&self, logits: &mut [f32], _generated_tokens: &[u32]) {
        for &(token_id, bias) in &self.biases {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] += bias;
            }
        }
    }

    fn name(&self) -> &'static str {
        "logit_bias"
    }
}

/// Bad words (banned tokens) processor — sets specified tokens to -inf.
pub struct BadWordsProcessor {
    banned_token_ids: Vec<u32>,
}

impl BadWordsProcessor {
    pub fn new(banned_token_ids: Vec<u32>) -> Self {
        Self { banned_token_ids }
    }
}

impl LogitsProcessor for BadWordsProcessor {
    fn process(&self, logits: &mut [f32], _generated_tokens: &[u32]) {
        for &token_id in &self.banned_token_ids {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "bad_words"
    }
}

/// Bad-words (multi-token sequence) processor.
///
/// Handles both single-token and multi-token bad word sequences:
/// - Single-token words: unconditionally set to -inf (static bias)
/// - Multi-token words: only ban the last token when preceding generated tokens
///   match the prefix of the sequence (dynamic matching)
pub struct NoBadWordsProcessor {
    bad_words_ids: Vec<Vec<u32>>,
}

impl NoBadWordsProcessor {
    pub fn new(bad_words_ids: Vec<Vec<u32>>) -> Self {
        Self { bad_words_ids }
    }
}

impl LogitsProcessor for NoBadWordsProcessor {
    fn process(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        for word_ids in &self.bad_words_ids {
            if word_ids.is_empty() {
                continue;
            }

            if word_ids.len() == 1 {
                // Single-token: unconditional ban
                let idx = word_ids[0] as usize;
                if idx < logits.len() {
                    logits[idx] = f32::NEG_INFINITY;
                }
                continue;
            }

            // Multi-token: check if generated tokens end with the prefix
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

    fn name(&self) -> &'static str {
        "no_bad_words"
    }
}

/// Allowed token IDs processor — masks all tokens except an allow-list.
///
/// Sets logits of all tokens not in `allowed_ids` to `-inf`,
/// restricting generation to the specified vocabulary subset.
pub struct AllowedTokenIdsProcessor {
    allowed_ids: Vec<u32>,
}

impl AllowedTokenIdsProcessor {
    pub fn new(allowed_ids: Vec<u32>) -> Self {
        Self { allowed_ids }
    }
}

impl LogitsProcessor for AllowedTokenIdsProcessor {
    fn process(&self, logits: &mut [f32], _generated_tokens: &[u32]) {
        // Build a mask: only allowed tokens survive
        for (idx, logit) in logits.iter_mut().enumerate() {
            if !self.allowed_ids.contains(&(idx as u32)) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "allowed_token_ids"
    }
}

/// Min-tokens processor — prevents EOS token until min_tokens generated.
pub struct MinTokensProcessor {
    min_tokens: usize,
    eos_token_id: u32,
}

impl MinTokensProcessor {
    pub fn new(min_tokens: usize, eos_token_id: u32) -> Self {
        Self {
            min_tokens,
            eos_token_id,
        }
    }
}

impl LogitsProcessor for MinTokensProcessor {
    fn process(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        if generated_tokens.len() < self.min_tokens {
            let idx = self.eos_token_id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "min_tokens"
    }
}

/// Pipeline of logits processors applied in sequence.
pub struct LogitsProcessorPipeline {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the end of the pipeline.
    pub fn push(&mut self, processor: Box<dyn LogitsProcessor>) {
        self.processors.push(processor);
    }

    /// Apply all processors in order.
    pub fn process(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        for processor in &self.processors {
            processor.process(logits, generated_tokens);
        }
    }

    /// Returns true if the pipeline has no processors.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Number of processors in the pipeline.
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Build a pipeline from SamplingParams.
    pub fn from_params(params: &super::SamplingParams) -> Self {
        let mut pipeline = Self::new();

        if params.repetition_penalty != 1.0 {
            pipeline.push(Box::new(RepetitionPenaltyProcessor::new(
                params.repetition_penalty,
            )));
        }

        if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
            pipeline.push(Box::new(FrequencyPresencePenaltyProcessor::new(
                params.frequency_penalty,
                params.presence_penalty,
            )));
        }

        if let Some(ref biases) = params.logit_bias {
            if !biases.is_empty() {
                pipeline.push(Box::new(LogitBiasProcessor::new(biases.clone())));
            }
        }

        if let Some(ref banned) = params.banned_token_ids {
            if !banned.is_empty() {
                pipeline.push(Box::new(BadWordsProcessor::new(banned.clone())));
            }
        }

        if let Some(ref allowed) = params.allowed_token_ids {
            if !allowed.is_empty() {
                pipeline.push(Box::new(AllowedTokenIdsProcessor::new(allowed.clone())));
            }
        }

        if let Some(ref bad_words) = params.bad_words_token_ids {
            if !bad_words.is_empty() {
                pipeline.push(Box::new(NoBadWordsProcessor::new(bad_words.clone())));
            }
        }

        if params.min_tokens > 0 {
            if let Some(eos_id) = params.eos_token_id {
                pipeline.push(Box::new(MinTokensProcessor::new(
                    params.min_tokens,
                    eos_id,
                )));
            }
        }

        pipeline
    }
}

impl Default for LogitsProcessorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_penalty_processor() {
        let proc = RepetitionPenaltyProcessor::new(2.0);
        let mut logits = vec![1.0, 2.0, -1.0, 0.5];
        let generated = vec![0, 2]; // tokens 0 and 2 were generated

        proc.process(&mut logits, &generated);

        // Token 0: positive, divided by 2.0 → 0.5
        assert!((logits[0] - 0.5).abs() < 1e-6);
        // Token 1: unchanged
        assert!((logits[1] - 2.0).abs() < 1e-6);
        // Token 2: negative, multiplied by 2.0 → -2.0
        assert!((logits[2] - (-2.0)).abs() < 1e-6);
        // Token 3: unchanged
        assert!((logits[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_presence_penalty() {
        let proc = FrequencyPresencePenaltyProcessor::new(0.5, 0.1);
        let mut logits = vec![1.0, 2.0, 3.0];
        // Token 0 appeared twice, token 2 appeared once
        let generated = vec![0, 0, 2];

        proc.process(&mut logits, &generated);

        // Token 0: -0.5 * 2 - 0.1 = -1.1
        assert!((logits[0] - (1.0 - 1.1)).abs() < 1e-6);
        // Token 1: unchanged
        assert!((logits[1] - 2.0).abs() < 1e-6);
        // Token 2: -0.5 * 1 - 0.1 = -0.6
        assert!((logits[2] - (3.0 - 0.6)).abs() < 1e-6);
    }

    #[test]
    fn test_bad_words_processor() {
        let proc = BadWordsProcessor::new(vec![1, 3]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        proc.process(&mut logits, &[]);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!(logits[1] == f32::NEG_INFINITY);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        assert!(logits[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_tokens_processor() {
        let proc = MinTokensProcessor::new(3, 2);

        // Not enough tokens generated — EOS (token 2) should be -inf
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        proc.process(&mut logits, &[5, 6]);
        assert!(logits[2] == f32::NEG_INFINITY);

        // Enough tokens generated — no change
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        proc.process(&mut logits, &[5, 6, 7]);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_logit_bias_processor() {
        let proc = LogitBiasProcessor::new(vec![(1, -100.0), (3, 5.0)]);
        let mut logits = vec![0.0, 0.0, 0.0, 0.0];
        proc.process(&mut logits, &[]);

        assert!((logits[0] - 0.0).abs() < 1e-6);
        assert!((logits[1] - (-100.0)).abs() < 1e-6);
        assert!((logits[2] - 0.0).abs() < 1e-6);
        assert!((logits[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_from_params() {
        let params = super::super::SamplingParams {
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            logit_bias: Some(vec![(0, 1.0)]),
            ..Default::default()
        };

        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert_eq!(pipeline.len(), 3);
    }

    #[test]
    fn test_pipeline_default_empty() {
        let params = super::super::SamplingParams::default();
        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_order_matters() {
        // Repetition penalty + bias: order should give different results
        let mut pipeline = LogitsProcessorPipeline::new();
        pipeline.push(Box::new(RepetitionPenaltyProcessor::new(2.0)));
        pipeline.push(Box::new(LogitBiasProcessor::new(vec![(0, 1.0)])));

        let mut logits = vec![2.0, 1.0];
        pipeline.process(&mut logits, &[0]);

        // Token 0: 2.0 / 2.0 = 1.0, then +1.0 = 2.0
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_allowed_token_ids_processor() {
        let proc = AllowedTokenIdsProcessor::new(vec![1, 3]);
        let mut logits = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        proc.process(&mut logits, &[]);

        // Only tokens 1 and 3 should survive
        assert!(logits[0] == f32::NEG_INFINITY);
        assert!((logits[1] - 5.0).abs() < 1e-6);
        assert!(logits[2] == f32::NEG_INFINITY);
        assert!((logits[3] - 5.0).abs() < 1e-6);
        assert!(logits[4] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_allowed_token_ids_empty_allows_nothing() {
        let proc = AllowedTokenIdsProcessor::new(vec![]);
        let mut logits = vec![5.0, 5.0, 5.0];
        proc.process(&mut logits, &[]);

        // Empty allow-list → all tokens blocked
        assert!(logits.iter().all(|&l| l == f32::NEG_INFINITY));
    }

    #[test]
    fn test_pipeline_from_params_with_banned_tokens() {
        let params = super::super::SamplingParams {
            banned_token_ids: Some(vec![1, 2]),
            ..Default::default()
        };

        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert_eq!(pipeline.len(), 1);

        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        pipeline.process(&mut logits, &[]);
        assert!(logits[1] == f32::NEG_INFINITY);
        assert!(logits[2] == f32::NEG_INFINITY);
        assert!((logits[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_from_params_with_allowed_tokens() {
        let params = super::super::SamplingParams {
            allowed_token_ids: Some(vec![0, 2]),
            ..Default::default()
        };

        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert_eq!(pipeline.len(), 1);

        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        pipeline.process(&mut logits, &[]);
        assert!((logits[0] - 5.0).abs() < 1e-6);
        assert!(logits[1] == f32::NEG_INFINITY);
        assert!((logits[2] - 5.0).abs() < 1e-6);
        assert!(logits[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_pipeline_from_params_with_min_tokens() {
        let params = super::super::SamplingParams {
            min_tokens: 5,
            eos_token_id: Some(2),
            ..Default::default()
        };

        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert_eq!(pipeline.len(), 1);

        // Not enough tokens generated — EOS blocked
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        pipeline.process(&mut logits, &[10, 11]);
        assert!(logits[2] == f32::NEG_INFINITY);

        // Enough tokens generated — EOS allowed
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        pipeline.process(&mut logits, &[10, 11, 12, 13, 14]);
        assert!((logits[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_from_params_min_tokens_needs_eos_id() {
        // min_tokens without eos_token_id → no processor added
        let params = super::super::SamplingParams {
            min_tokens: 5,
            eos_token_id: None,
            ..Default::default()
        };

        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_from_params_all_features() {
        let params = super::super::SamplingParams {
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            logit_bias: Some(vec![(0, 1.0)]),
            banned_token_ids: Some(vec![5]),
            allowed_token_ids: Some(vec![0, 1, 2, 3]),
            min_tokens: 3,
            eos_token_id: Some(1),
            ..Default::default()
        };

        let pipeline = LogitsProcessorPipeline::from_params(&params);
        // rep_pen + freq_pres + logit_bias + banned + allowed + min_tokens = 6
        assert_eq!(pipeline.len(), 6);
    }

    #[test]
    fn test_no_bad_words_single_token() {
        let proc = NoBadWordsProcessor::new(vec![vec![1], vec![3]]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        proc.process(&mut logits, &[]);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!(logits[1] == f32::NEG_INFINITY);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        assert!(logits[3] == f32::NEG_INFINITY);
        assert!((logits[4] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_bad_words_multi_token_no_match() {
        // Bad word is [10, 20] — prefix [10] must match last generated token
        let proc = NoBadWordsProcessor::new(vec![vec![10, 20]]);
        let mut logits = vec![1.0, 2.0, 3.0];
        // Last generated token is 5, not 10 → no ban
        proc.process(&mut logits, &[5]);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_bad_words_multi_token_match() {
        // Bad word is [10, 20] — prefix [10] matches last generated token
        let proc = NoBadWordsProcessor::new(vec![vec![10, 20]]);
        let mut logits = vec![0.0; 30];
        logits[20] = 5.0;
        // Last generated token is 10 → token 20 should be banned
        proc.process(&mut logits, &[3, 10]);
        assert!(logits[20] == f32::NEG_INFINITY);
        // Other tokens unchanged
        assert!((logits[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_bad_words_three_token_sequence() {
        // Bad word is [5, 6, 7] — prefix [5, 6] must be last 2 generated tokens
        let proc = NoBadWordsProcessor::new(vec![vec![5, 6, 7]]);
        let mut logits = vec![0.0; 10];
        logits[7] = 3.0;

        // Only partial prefix match → no ban
        proc.process(&mut logits, &[5]);
        assert!((logits[7] - 3.0).abs() < 1e-6);

        // Full prefix match → ban
        let mut logits = vec![0.0; 10];
        logits[7] = 3.0;
        proc.process(&mut logits, &[1, 2, 5, 6]);
        assert!(logits[7] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_no_bad_words_empty_sequences_ignored() {
        let proc = NoBadWordsProcessor::new(vec![vec![], vec![2]]);
        let mut logits = vec![1.0, 2.0, 3.0];
        proc.process(&mut logits, &[]);
        // Only token 2 should be banned (empty sequence ignored)
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!(logits[2] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_no_bad_words_not_enough_history() {
        // Bad word [10, 20, 30] needs 2 tokens in history; only 1 exists
        let proc = NoBadWordsProcessor::new(vec![vec![10, 20, 30]]);
        let mut logits = vec![0.0; 40];
        logits[30] = 5.0;
        proc.process(&mut logits, &[20]);
        // Not enough history → no ban
        assert!((logits[30] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_from_params_with_bad_words() {
        let params = super::super::SamplingParams {
            bad_words_token_ids: Some(vec![vec![1], vec![2, 3]]),
            ..Default::default()
        };
        let pipeline = LogitsProcessorPipeline::from_params(&params);
        assert_eq!(pipeline.len(), 1);

        let mut logits = vec![5.0; 5];
        pipeline.process(&mut logits, &[2]);
        // Single-token [1] → banned unconditionally
        assert!(logits[1] == f32::NEG_INFINITY);
        // Multi-token [2, 3] → prefix [2] matches generated → token 3 banned
        assert!(logits[3] == f32::NEG_INFINITY);
        // Others unchanged
        assert!((logits[0] - 5.0).abs() < 1e-6);
        assert!((logits[4] - 5.0).abs() < 1e-6);
    }
}
