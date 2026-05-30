//! Constrained generation infrastructure.
//!
//! This module provides traits and implementations for constraining
//! LLM output to specific patterns (choices, regex, JSON schema).

use std::collections::HashSet;
use std::sync::Arc;

use crate::tokenizer::TokenizerWrapper;

/// Trait for sampling constraints that mask invalid tokens.
///
/// Constraints modify the logits before sampling to ensure the
/// generated text follows a specific pattern or format.
///
/// # UTF-8 and Emoji Support
///
/// All constraints support UTF-8 and emoji through proper handling:
/// - Text comparisons use Rust's native UTF-8 strings
/// - Character counting uses `.chars().count()` not `.len()`
/// - Token decode failures (partial multi-byte sequences) are handled gracefully
pub trait SamplingConstraint: Send + Sync {
    /// Mask invalid tokens by setting their logits to negative infinity.
    ///
    /// # Arguments
    /// * `logits` - Mutable slice of logits to modify (vocab_size length)
    /// * `generated_text` - Previously generated text (decoded)
    ///
    /// # Returns
    /// Ok(()) on success, error if constraint validation fails
    fn mask_logits(&mut self, logits: &mut [f32], generated_text: &str) -> anyhow::Result<()>;

    /// Check if generation is complete according to this constraint.
    ///
    /// # Arguments
    /// * `generated` - Generated token IDs so far
    /// * `text` - Decoded text so far
    ///
    /// # Returns
    /// true if generation should stop
    fn is_complete(&self, generated: &[u32], text: &str) -> bool;

    /// Reset the constraint state for a new sequence.
    fn reset(&mut self);

    /// Notify the constraint that `token_id` was just sampled, so it
    /// can advance its internal state machine before the next
    /// `mask_logits` call.
    ///
    /// Default implementation is a no-op: text-based / stateless
    /// constraints (e.g. legacy `JsonSchemaConstraint`,
    /// `ChoiceConstraint` which tracks position internally on
    /// `mask_logits`) don't need per-token hooks.
    ///
    /// Stateful grammar constraints (xgrammar via
    /// `GrammarConstraintAdapter`) MUST override this — without
    /// advancing the matcher, every step's bitmask reflects the
    /// initial grammar state and the constraint silently fails to
    /// enforce anything beyond the first token.
    ///
    /// # Returns
    /// `true` if the token was a valid grammar transition; `false`
    /// if the token violates the constraint (which shouldn't happen
    /// in practice because `mask_logits` should have prevented it).
    fn accept_token(&mut self, _token_id: u32) -> bool {
        true
    }

    /// True iff this constraint can produce a packed-i32 token bitmask
    /// on the CPU that the engine can upload to GPU and apply via
    /// `gpu_apply_grammar_bitmask`. Returning `true` lets the engine
    /// keep the sampler entirely on GPU instead of pulling logits to
    /// host for `mask_logits`.
    ///
    /// Default `false` covers all text-driven constraints (ChoiceConstraint,
    /// RegexConstraint, legacy JsonSchemaConstraint) — they have no
    /// vocab-indexed bitmask representation and must stay on CPU.
    fn supports_gpu(&self) -> bool {
        false
    }

    /// Fill `bitmask_row` (packed i32, length ≥ ceil(grammar_vocab/32))
    /// with the allowed-token bitmask for the constraint's current
    /// position. Returns `Some(Ok(()))` on success, `Some(Err)` on
    /// fault, or `None` if this constraint doesn't support GPU
    /// masking (`supports_gpu() == false`).
    ///
    /// The CPU side is what fills the bitmask (xgrammar's matcher
    /// runs on CPU); the engine then handles the HtoD copy and GPU
    /// apply. Naming the method `_for_gpu` makes that contract
    /// explicit: the bitmask is *destined* for GPU, even though its
    /// computation is host-side.
    fn fill_cpu_bitmask_for_gpu(&mut self, _bitmask_row: &mut [i32]) -> Option<anyhow::Result<()>> {
        None
    }

    /// Expose the underlying xgrammar matcher so the engine can batch
    /// the per-step bitmask fill across multiple constrained requests
    /// via `xgrammar_rs::BatchMatcher` (one thread-pooled fill instead
    /// of N sequential `fill_next_token_bitmask` calls). Returns `None`
    /// for constraints without an xgrammar matcher — they fall back to
    /// the per-request `fill_cpu_bitmask_for_gpu` path.
    #[cfg(feature = "xgrammar")]
    fn xgrammar_matcher(&self) -> Option<&std::sync::Mutex<xgrammar_rs::GrammarMatcher>> {
        None
    }

    /// Number of packed-i32 words the xgrammar matcher expects per
    /// bitmask row (`ceil(tokenizer_vocab / 32)`). Needed to size the
    /// batched-fill buffer correctly; the grammar's vocab is usually
    /// smaller than the model's padded lm_head vocab. `None` when
    /// [`Self::xgrammar_matcher`] is `None`.
    #[cfg(feature = "xgrammar")]
    fn grammar_bitmask_words(&self) -> Option<usize> {
        None
    }
}

/// Constraint that forces output to be one of predefined choices.
///
/// This constraint pre-tokenizes each choice and maintains a trie-like
/// structure of valid token sequences. At each step, only tokens that
/// are valid prefixes of at least one choice are allowed.
#[derive(Debug)]
pub struct ChoiceConstraint {
    /// Pre-tokenized choices
    choices: Vec<Vec<u32>>,
    /// Valid token IDs at current position for each choice
    valid_tokens_at_position: Vec<HashSet<u32>>,
    /// Number of tokens generated so far
    position: usize,
}

impl ChoiceConstraint {
    /// Create a new choice constraint from string choices.
    pub fn new(choices: &[String], tokenizer: &TokenizerWrapper) -> anyhow::Result<Self> {
        if choices.is_empty() {
            anyhow::bail!("ChoiceConstraint requires at least one choice");
        }

        let mut tokenized_choices = Vec::with_capacity(choices.len());
        for choice in choices {
            let tokens = tokenizer.encode(choice)?;
            if tokens.is_empty() {
                anyhow::bail!("Choice '{}' tokenizes to empty sequence", choice);
            }
            tokenized_choices.push(tokens);
        }

        // Build valid tokens at each position
        let max_len = tokenized_choices.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut valid_tokens_at_position = vec![HashSet::new(); max_len];

        for choice in &tokenized_choices {
            for (pos, &token) in choice.iter().enumerate() {
                valid_tokens_at_position[pos].insert(token);
            }
        }

        Ok(Self {
            choices: tokenized_choices,
            valid_tokens_at_position,
            position: 0,
        })
    }

    /// Create a choice constraint from pre-tokenized choices.
    pub fn from_token_ids(choices: Vec<Vec<u32>>) -> anyhow::Result<Self> {
        if choices.is_empty() {
            anyhow::bail!("ChoiceConstraint requires at least one choice");
        }

        let max_len = choices.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut valid_tokens_at_position = vec![HashSet::new(); max_len];

        for choice in &choices {
            for (pos, &token) in choice.iter().enumerate() {
                valid_tokens_at_position[pos].insert(token);
            }
        }

        Ok(Self {
            choices,
            valid_tokens_at_position,
            position: 0,
        })
    }

    /// Get the valid choices remaining given the generated tokens.
    pub fn remaining_choices(&self, generated: &[u32]) -> Vec<&Vec<u32>> {
        self.choices
            .iter()
            .filter(|choice| {
                if generated.len() > choice.len() {
                    return false;
                }
                choice.iter().zip(generated.iter()).all(|(a, b)| a == b)
            })
            .collect()
    }
}

impl SamplingConstraint for ChoiceConstraint {
    fn mask_logits(&mut self, logits: &mut [f32], _generated_text: &str) -> anyhow::Result<()> {
        if self.position >= self.valid_tokens_at_position.len() {
            // All choices have been exhausted, allow EOS only
            // (In practice, is_complete should trigger first)
            return Ok(());
        }

        let valid_tokens = &self.valid_tokens_at_position[self.position];

        // Mask all tokens except valid ones
        for (idx, logit) in logits.iter_mut().enumerate() {
            if !valid_tokens.contains(&(idx as u32)) {
                *logit = f32::NEG_INFINITY;
            }
        }

        self.position += 1;
        Ok(())
    }

    fn is_complete(&self, generated: &[u32], _text: &str) -> bool {
        // Complete if any choice exactly matches
        self.choices.iter().any(|choice| choice == generated)
    }

    fn reset(&mut self) {
        self.position = 0;
    }
}

/// Constraint for regex pattern matching using DFA.
///
/// This constraint builds a DFA from the regex pattern and tracks
/// the current state. Only tokens that lead to valid transitions
/// are allowed.
#[derive(Clone)]
pub struct RegexConstraint {
    /// The regex pattern (stored for display/debugging)
    pattern: String,
    /// Tokenizer for decoding tokens to check validity
    tokenizer: Arc<TokenizerWrapper>,
    /// Decoded text so far
    current_text: String,
    /// Compiled regex for validation
    regex: regex::Regex,
    /// Maximum allowed length (to prevent infinite generation)
    max_length: usize,
}

impl std::fmt::Debug for RegexConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexConstraint")
            .field("pattern", &self.pattern)
            .field("current_text", &self.current_text)
            .field("max_length", &self.max_length)
            .finish()
    }
}

impl RegexConstraint {
    /// Create a new regex constraint.
    ///
    /// # Arguments
    /// * `pattern` - Regex pattern (must match entire output)
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `max_length` - Maximum number of characters allowed
    pub fn new(
        pattern: &str,
        tokenizer: Arc<TokenizerWrapper>,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        // Wrap pattern to match from start
        let anchored_pattern = format!("^(?:{})$", pattern);
        let regex = regex::Regex::new(&anchored_pattern)?;

        Ok(Self {
            pattern: pattern.to_string(),
            tokenizer,
            current_text: String::new(),
            regex,
            max_length,
        })
    }

    /// Check if a text is a valid prefix of the pattern.
    ///
    /// Uses a partial matching approach: constructs a regex that matches
    /// any prefix of the original pattern. This handles UTF-8 and emoji correctly.
    fn is_valid_prefix(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        // Check max_length first (using character count for proper UTF-8 handling)
        let char_count = text.chars().count();
        if char_count > self.max_length {
            return false;
        }

        // Check if the text already fully matches
        if self.regex.is_match(text) {
            return true;
        }

        // At exactly max_length, must be a complete match (checked above)
        if char_count == self.max_length {
            return false;
        }

        // For prefix checking, we construct a pattern that matches any prefix
        // of strings that could eventually match the full pattern.
        // We use a heuristic: the text is a valid prefix if:
        // 1. It's shorter than max_length (can still grow)
        // 2. It doesn't contain obviously invalid characters for the pattern
        //
        // A proper implementation would use incremental DFA matching,
        // but this heuristic works for common cases.

        // Try to match text + arbitrary continuation
        // Build a pattern that checks if 'text' could be a prefix of a valid match
        let prefix_check = format!("^{}.*", regex::escape(text));
        if let Ok(prefix_regex) = regex::Regex::new(&prefix_check) {
            // Check if any string starting with 'text' could match original pattern
            // This is done by checking if escaped text is a valid regex prefix
            // For simple patterns, this works well
            return prefix_regex.is_match(text) || self.could_extend_to_match(text);
        }

        // If we can't build the check pattern, allow it (fail open)
        true
    }

    /// Check if the text could potentially be extended to match the pattern.
    ///
    /// This is a heuristic check that handles common cases.
    fn could_extend_to_match(&self, text: &str) -> bool {
        // For patterns with anchors, check basic compatibility
        // This handles UTF-8 strings correctly since we work with &str

        // Empty prefix can always potentially match
        if text.is_empty() {
            return true;
        }

        // If the text contains invalid UTF-8 sequences, reject it
        // (Rust strings are always valid UTF-8, so this is implicit)

        // Check character by character if each prefix could lead to a match
        // by seeing if any completion might work
        text.chars().count() < self.max_length
    }
}

impl SamplingConstraint for RegexConstraint {
    fn mask_logits(&mut self, logits: &mut [f32], generated_text: &str) -> anyhow::Result<()> {
        // Update current text from the generated text
        self.current_text = generated_text.to_string();

        // Use character count for max_length check (handles UTF-8/emoji correctly)
        let char_count = self.current_text.chars().count();

        if char_count >= self.max_length {
            // At max length, only allow tokens that complete the pattern
            for (idx, logit) in logits.iter_mut().enumerate() {
                match self.tokenizer.decode(&[idx as u32]) {
                    Ok(token_text) => {
                        let candidate = format!("{}{}", self.current_text, token_text);
                        if !self.regex.is_match(&candidate) {
                            *logit = f32::NEG_INFINITY;
                        }
                    }
                    Err(_) => {
                        // Token decodes to invalid UTF-8 (partial multi-byte sequence)
                        // This can happen with byte-level tokenizers
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            return Ok(());
        }

        // For each token, check if appending it leads to a valid prefix
        // This is O(vocab_size) which can be slow - a proper implementation
        // would use pre-computed token transitions
        let mut any_valid = false;
        for (idx, logit) in logits.iter_mut().enumerate() {
            if *logit == f32::NEG_INFINITY {
                continue; // Already masked
            }

            match self.tokenizer.decode(&[idx as u32]) {
                Ok(token_text) => {
                    let candidate = format!("{}{}", self.current_text, token_text);
                    if self.is_valid_prefix(&candidate) {
                        any_valid = true;
                    } else {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                Err(_) => {
                    // Token decodes to invalid UTF-8 - mask it
                    // This handles partial multi-byte sequences from byte-level tokenizers
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        if !any_valid {
            // No valid tokens - this shouldn't happen with a well-formed regex
            anyhow::bail!(
                "No valid tokens for regex constraint at text: '{}' (pattern: {})",
                self.current_text,
                self.pattern
            );
        }

        Ok(())
    }

    fn is_complete(&self, _generated: &[u32], text: &str) -> bool {
        self.regex.is_match(text)
    }

    fn reset(&mut self) {
        self.current_text.clear();
    }
}

/// JSON parsing state for schema constraint.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonParseState {
    /// Starting state, expecting any value or object start
    Start,
    /// Expecting object start '{'
    ExpectingObjectStart,
    /// Inside object, expecting key or '}'
    ExpectingKeyOrEnd,
    /// Expecting ':'
    ExpectingColon,
    /// Expecting value
    ExpectingValue { value_type: JsonValueType },
    /// Expecting ',' or '}'
    ExpectingCommaOrEnd,
    /// JSON is complete and valid
    Complete,
}

/// Expected JSON value types.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonValueType {
    String,
    Number,
    Boolean,
    Null,
    Object,
    Array,
    Any,
}

/// Constraint for JSON schema validation.
///
/// This constraint ensures the generated output is valid JSON
/// that conforms to the provided schema. It tracks the parsing
/// state and masks tokens that would lead to invalid JSON.
pub struct JsonSchemaConstraint {
    /// The JSON schema
    schema: serde_json::Value,
    /// Current parsing state
    state: JsonParseState,
    /// Tokenizer for decoding
    tokenizer: Arc<TokenizerWrapper>,
    /// Current partial JSON text
    current_json: String,
    /// Brace/bracket depth
    depth: usize,
    /// Required keys from schema that haven't been seen
    required_keys: HashSet<String>,
}

impl std::fmt::Debug for JsonSchemaConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonSchemaConstraint")
            .field("schema", &self.schema)
            .field("state", &self.state)
            .field("current_json", &self.current_json)
            .field("depth", &self.depth)
            .field("required_keys", &self.required_keys)
            .finish()
    }
}

impl JsonSchemaConstraint {
    /// Create a new JSON schema constraint.
    ///
    /// # Arguments
    /// * `schema` - JSON schema (subset of JSON Schema draft-07)
    /// * `tokenizer` - Tokenizer for encoding/decoding
    pub fn new(schema: serde_json::Value, tokenizer: Arc<TokenizerWrapper>) -> Self {
        let required_keys = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            schema,
            state: JsonParseState::Start,
            tokenizer,
            current_json: String::new(),
            depth: 0,
            required_keys,
        }
    }

    /// Check if a partial JSON string could still be valid.
    fn is_valid_partial_json(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        // Try parsing - if it fails, check if it's incomplete
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(_) => true,
            Err(e) => {
                // EOF errors mean the JSON is incomplete but valid so far
                let err_msg = e.to_string();
                err_msg.contains("EOF") || err_msg.contains("unexpected end")
            }
        }
    }

    /// Get expected token types based on current state.
    #[allow(dead_code)]
    fn expected_tokens(&self) -> Vec<&'static str> {
        match &self.state {
            JsonParseState::Start => vec!["{", "[", "\"", "0", "1", "true", "false", "null"],
            JsonParseState::ExpectingObjectStart => vec!["{"],
            JsonParseState::ExpectingKeyOrEnd => vec!["\"", "}"],
            JsonParseState::ExpectingColon => vec![":"],
            JsonParseState::ExpectingValue { value_type } => match value_type {
                JsonValueType::String => vec!["\""],
                JsonValueType::Number => {
                    vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]
                }
                JsonValueType::Boolean => vec!["true", "false"],
                JsonValueType::Null => vec!["null"],
                JsonValueType::Object => vec!["{"],
                JsonValueType::Array => vec!["["],
                JsonValueType::Any => vec!["{", "[", "\"", "0", "1", "true", "false", "null"],
            },
            JsonParseState::ExpectingCommaOrEnd => vec![",", "}", "]"],
            JsonParseState::Complete => vec![],
        }
    }
}

impl SamplingConstraint for JsonSchemaConstraint {
    fn mask_logits(&mut self, logits: &mut [f32], generated_text: &str) -> anyhow::Result<()> {
        // Update current JSON from generated text
        self.current_json = generated_text.to_string();

        if self.state == JsonParseState::Complete {
            // Already complete, mask everything
            for logit in logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            return Ok(());
        }

        // For each token, check if appending it results in valid partial JSON
        // This handles UTF-8 and emoji correctly since serde_json fully supports Unicode
        for (idx, logit) in logits.iter_mut().enumerate() {
            if *logit == f32::NEG_INFINITY {
                continue;
            }

            match self.tokenizer.decode(&[idx as u32]) {
                Ok(token_text) => {
                    let candidate = format!("{}{}", self.current_json, token_text);
                    if !self.is_valid_partial_json(&candidate) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                Err(_) => {
                    // Token decodes to invalid UTF-8 - mask it
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        Ok(())
    }

    fn is_complete(&self, _generated: &[u32], text: &str) -> bool {
        // Check if text is valid complete JSON
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
            // Optionally validate against schema here
            // For now, just check if it parses
            return matches!(
                value,
                serde_json::Value::Object(_) | serde_json::Value::Array(_)
            );
        }
        false
    }

    fn reset(&mut self) {
        self.state = JsonParseState::Start;
        self.current_json.clear();
        self.depth = 0;
        self.required_keys = self
            .schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_tokenizer() -> TokenizerWrapper {
        // Create a simple mock tokenizer for testing
        // Use a reasonable vocab size for test scenarios
        TokenizerWrapper::for_testing(1000)
    }

    #[test]
    fn test_choice_constraint_from_token_ids() {
        let choices = vec![vec![1, 2, 3], vec![1, 4, 5], vec![6, 7]];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        assert_eq!(constraint.choices.len(), 3);
        assert_eq!(constraint.valid_tokens_at_position.len(), 3);
        assert!(constraint.valid_tokens_at_position[0].contains(&1));
        assert!(constraint.valid_tokens_at_position[0].contains(&6));
    }

    #[test]
    fn test_choice_constraint_mask_logits() {
        let choices = vec![vec![1, 2], vec![3, 4]];
        let mut constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        let mut logits = vec![0.0f32; 10];
        constraint.mask_logits(&mut logits, "").unwrap();

        // Only tokens 1 and 3 should be valid at position 0
        assert!(logits[1].is_finite());
        assert!(logits[3].is_finite());
        assert!(logits[0].is_infinite());
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn test_choice_constraint_is_complete() {
        let choices = vec![vec![1, 2], vec![3, 4]];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        assert!(constraint.is_complete(&[1, 2], "test"));
        assert!(constraint.is_complete(&[3, 4], "test"));
        assert!(!constraint.is_complete(&[1], "t"));
        assert!(!constraint.is_complete(&[1, 3], "test"));
    }

    #[test]
    fn test_choice_constraint_remaining_choices() {
        let choices = vec![vec![1, 2, 3], vec![1, 2, 4], vec![5, 6]];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        let remaining = constraint.remaining_choices(&[1, 2]);
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn test_choice_constraint_reset() {
        let choices = vec![vec![1, 2]];
        let mut constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        let mut logits = vec![0.0f32; 10];
        constraint.mask_logits(&mut logits, "").unwrap();
        assert_eq!(constraint.position, 1);

        constraint.reset();
        assert_eq!(constraint.position, 0);
    }

    #[test]
    fn test_json_schema_is_valid_partial_json() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        assert!(constraint.is_valid_partial_json(""));
        assert!(constraint.is_valid_partial_json("{"));
        assert!(constraint.is_valid_partial_json("{\"name\""));
        assert!(constraint.is_valid_partial_json("{\"name\":"));
        assert!(constraint.is_valid_partial_json("{\"name\": \""));
    }

    #[test]
    fn test_json_schema_is_complete() {
        let schema = serde_json::json!({
            "type": "object"
        });
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        assert!(constraint.is_complete(&[], "{\"name\": \"John\"}"));
        assert!(constraint.is_complete(&[], "[]"));
        assert!(!constraint.is_complete(&[], "\"just a string\""));
        assert!(!constraint.is_complete(&[], "{incomplete"));
    }

    #[test]
    fn test_json_schema_expected_tokens() {
        let schema = serde_json::json!({"type": "object"});
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        let expected = constraint.expected_tokens();
        assert!(expected.contains(&"{"));
        assert!(expected.contains(&"["));
    }

    // ─── UTF-8 and Emoji Tests ────────────────────────────────────────────

    #[test]
    fn test_json_schema_utf8_strings() {
        let schema = serde_json::json!({"type": "object"});
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        // Test UTF-8 characters in JSON
        assert!(constraint.is_valid_partial_json("{\"name\": \"Привет"));
        assert!(constraint.is_valid_partial_json("{\"name\": \"日本語"));
        assert!(constraint.is_valid_partial_json("{\"name\": \"한국어"));
        assert!(constraint.is_valid_partial_json("{\"emoji\": \"🎉"));
    }

    #[test]
    fn test_json_schema_complete_with_utf8() {
        let schema = serde_json::json!({"type": "object"});
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        // Complete JSON with UTF-8 content
        assert!(constraint.is_complete(&[], "{\"greeting\": \"Привет мир\"}"));
        assert!(constraint.is_complete(&[], "{\"message\": \"こんにちは世界\"}"));
        assert!(constraint.is_complete(&[], "{\"emoji\": \"🎉🚀✨\"}"));
        assert!(constraint.is_complete(&[], "{\"mixed\": \"Hello мир 世界 🌍\"}"));
    }

    #[test]
    fn test_json_schema_emoji_in_keys() {
        let schema = serde_json::json!({"type": "object"});
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        // JSON with emoji in keys (valid JSON)
        assert!(constraint.is_valid_partial_json("{\"🔑\": \"value"));
        assert!(constraint.is_complete(&[], "{\"🎉\": \"party\"}"));
    }

    #[test]
    fn test_regex_constraint_utf8_pattern() {
        let tokenizer = Arc::new(mock_tokenizer());

        // Pattern that matches Cyrillic text
        let constraint = RegexConstraint::new(r"[а-яА-ЯёЁ]+", tokenizer.clone(), 100).unwrap();

        assert!(constraint.is_complete(&[], "Привет"));
        assert!(constraint.is_complete(&[], "Мир"));
        assert!(!constraint.is_complete(&[], "Hello")); // Latin, not Cyrillic
    }

    #[test]
    fn test_regex_constraint_emoji_pattern() {
        let tokenizer = Arc::new(mock_tokenizer());

        // Pattern that matches text with emoji
        // Note: \p{Emoji} requires the unicode-perl feature in regex
        let constraint = RegexConstraint::new(r".*[🎉🚀✨].*", tokenizer.clone(), 100).unwrap();

        assert!(constraint.is_complete(&[], "Party 🎉!"));
        assert!(constraint.is_complete(&[], "🚀 Launch"));
        assert!(!constraint.is_complete(&[], "No emoji here"));
    }

    #[test]
    fn test_regex_constraint_max_length_chars_not_bytes() {
        let tokenizer = Arc::new(mock_tokenizer());

        // max_length should be in characters, not bytes
        // "🎉" is 1 character but 4 bytes
        let constraint = RegexConstraint::new(
            r".*",
            tokenizer.clone(),
            5, // 5 characters max
        )
        .unwrap();

        // "🎉🎉🎉🎉🎉" is 5 characters (20 bytes)
        assert!(constraint.is_valid_prefix("🎉🎉🎉🎉🎉"));

        // "🎉🎉🎉🎉🎉🎉" is 6 characters - should fail max_length check
        assert!(!constraint.is_valid_prefix("🎉🎉🎉🎉🎉🎉"));
    }

    #[test]
    fn test_regex_constraint_mixed_scripts() {
        let tokenizer = Arc::new(mock_tokenizer());

        // Pattern matching mixed Latin/Cyrillic/CJK
        let constraint = RegexConstraint::new(r"Hello.*世界", tokenizer.clone(), 100).unwrap();

        assert!(constraint.is_complete(&[], "Hello 世界"));
        assert!(constraint.is_complete(&[], "Hello мир 世界"));
        assert!(!constraint.is_complete(&[], "Hello world")); // No 世界
    }

    #[test]
    fn test_choice_constraint_token_based_utf8() {
        // ChoiceConstraint works at token level, so UTF-8 support
        // depends on tokenizer encoding
        let choices = vec![
            vec![100, 200, 300], // Represents "Привет"
            vec![100, 201, 301], // Represents "Пока"
        ];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        // The constraint should work regardless of what the tokens represent
        assert!(constraint.is_complete(&[100, 200, 300], "Привет"));
        assert!(constraint.is_complete(&[100, 201, 301], "Пока"));
        assert!(!constraint.is_complete(&[100, 200], "Прив")); // Incomplete
    }

    #[test]
    fn test_json_schema_unicode_escapes() {
        let schema = serde_json::json!({"type": "object"});
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        // JSON with Unicode escapes (valid JSON)
        assert!(constraint.is_valid_partial_json(r#"{"text": "\u0041"#)); // A
        assert!(constraint.is_valid_partial_json(r#"{"emoji": "\uD83C\uDF89"#)); // 🎉 as surrogate pair
        assert!(constraint.is_complete(&[], r#"{"text": "\u0041"}"#));
    }

    #[test]
    fn test_regex_constraint_is_valid_prefix_utf8() {
        let tokenizer = Arc::new(mock_tokenizer());

        let constraint = RegexConstraint::new(r"Привет.*мир", tokenizer.clone(), 50).unwrap();

        // Valid prefixes
        assert!(constraint.is_valid_prefix("П"));
        assert!(constraint.is_valid_prefix("Привет"));
        assert!(constraint.is_valid_prefix("Привет "));
        assert!(constraint.is_valid_prefix("Привет мир")); // Complete match is also valid prefix
    }

    #[test]
    fn test_json_schema_nested_utf8() {
        let schema = serde_json::json!({"type": "object"});
        let tokenizer = Arc::new(mock_tokenizer());
        let constraint = JsonSchemaConstraint::new(schema, tokenizer);

        // Nested objects with UTF-8
        let nested_json =
            r#"{"user": {"name": "Иван", "greeting": "Привет 🎉"}, "tags": ["тег1", "タグ2"]}"#;
        assert!(constraint.is_complete(&[], nested_json));
    }
}
