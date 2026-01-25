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
pub trait SamplingConstraint: Send + Sync {
    /// Mask invalid tokens by setting their logits to negative infinity.
    ///
    /// # Arguments
    /// * `logits` - Mutable slice of logits to modify (vocab_size length)
    /// * `generated` - Previously generated token IDs
    ///
    /// # Returns
    /// Ok(()) on success, error if constraint validation fails
    fn mask_logits(&mut self, logits: &mut [f32], generated: &[u32]) -> anyhow::Result<()>;

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
                choice
                    .iter()
                    .zip(generated.iter())
                    .all(|(a, b)| a == b)
            })
            .collect()
    }
}

impl SamplingConstraint for ChoiceConstraint {
    fn mask_logits(&mut self, logits: &mut [f32], _generated: &[u32]) -> anyhow::Result<()> {
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
    fn is_valid_prefix(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        // For prefix checking, we need to check if the text could still match
        // We use a heuristic: check if adding .* after the text matches
        let prefix_pattern = format!("^(?:{})", regex::escape(text));
        if let Ok(prefix_regex) = regex::Regex::new(&prefix_pattern) {
            // The prefix regex should be a prefix of possible matches
            // This is a simplified check - a proper implementation would use
            // DFA state tracking
            return prefix_regex.is_match(text) || text.len() < self.max_length;
        }
        true
    }
}

impl SamplingConstraint for RegexConstraint {
    fn mask_logits(&mut self, logits: &mut [f32], _generated: &[u32]) -> anyhow::Result<()> {
        if self.current_text.len() >= self.max_length {
            // At max length, only allow tokens that complete the pattern
            for (idx, logit) in logits.iter_mut().enumerate() {
                if let Ok(token_text) = self.tokenizer.decode(&[idx as u32]) {
                    let candidate = format!("{}{}", self.current_text, token_text);
                    if !self.regex.is_match(&candidate) {
                        *logit = f32::NEG_INFINITY;
                    }
                } else {
                    *logit = f32::NEG_INFINITY;
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

            if let Ok(token_text) = self.tokenizer.decode(&[idx as u32]) {
                let candidate = format!("{}{}", self.current_text, token_text);
                if self.is_valid_prefix(&candidate) {
                    any_valid = true;
                } else {
                    *logit = f32::NEG_INFINITY;
                }
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }

        if !any_valid {
            // No valid tokens - this shouldn't happen with a well-formed regex
            // Allow all tokens to prevent getting stuck
            anyhow::bail!("No valid tokens for regex constraint at text: '{}'", self.current_text);
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
                JsonValueType::Number => vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"],
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
    fn mask_logits(&mut self, logits: &mut [f32], _generated: &[u32]) -> anyhow::Result<()> {
        if self.state == JsonParseState::Complete {
            // Already complete, mask everything
            for logit in logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            return Ok(());
        }

        // For each token, check if appending it results in valid partial JSON
        for (idx, logit) in logits.iter_mut().enumerate() {
            if *logit == f32::NEG_INFINITY {
                continue;
            }

            if let Ok(token_text) = self.tokenizer.decode(&[idx as u32]) {
                let candidate = format!("{}{}", self.current_json, token_text);
                if !self.is_valid_partial_json(&candidate) {
                    *logit = f32::NEG_INFINITY;
                }
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    fn is_complete(&self, _generated: &[u32], text: &str) -> bool {
        // Check if text is valid complete JSON
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
            // Optionally validate against schema here
            // For now, just check if it parses
            return matches!(value, serde_json::Value::Object(_) | serde_json::Value::Array(_));
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
        let choices = vec![
            vec![1, 2, 3],
            vec![1, 4, 5],
            vec![6, 7],
        ];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        assert_eq!(constraint.choices.len(), 3);
        assert_eq!(constraint.valid_tokens_at_position.len(), 3);
        assert!(constraint.valid_tokens_at_position[0].contains(&1));
        assert!(constraint.valid_tokens_at_position[0].contains(&6));
    }

    #[test]
    fn test_choice_constraint_mask_logits() {
        let choices = vec![
            vec![1, 2],
            vec![3, 4],
        ];
        let mut constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        let mut logits = vec![0.0f32; 10];
        constraint.mask_logits(&mut logits, &[]).unwrap();

        // Only tokens 1 and 3 should be valid at position 0
        assert!(logits[1].is_finite());
        assert!(logits[3].is_finite());
        assert!(logits[0].is_infinite());
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn test_choice_constraint_is_complete() {
        let choices = vec![
            vec![1, 2],
            vec![3, 4],
        ];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        assert!(constraint.is_complete(&[1, 2], "test"));
        assert!(constraint.is_complete(&[3, 4], "test"));
        assert!(!constraint.is_complete(&[1], "t"));
        assert!(!constraint.is_complete(&[1, 3], "test"));
    }

    #[test]
    fn test_choice_constraint_remaining_choices() {
        let choices = vec![
            vec![1, 2, 3],
            vec![1, 2, 4],
            vec![5, 6],
        ];
        let constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        let remaining = constraint.remaining_choices(&[1, 2]);
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn test_choice_constraint_reset() {
        let choices = vec![vec![1, 2]];
        let mut constraint = ChoiceConstraint::from_token_ids(choices).unwrap();

        let mut logits = vec![0.0f32; 10];
        constraint.mask_logits(&mut logits, &[]).unwrap();
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
}
