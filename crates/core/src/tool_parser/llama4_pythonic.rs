//! Llama 4 Pythonic tool call parser.
//!
//! Parses tool calls in pythonic format with optional Python start/end tags:
//! ```text
//! <|python_start|>[get_weather(city='NYC')]<|python_end|>
//! ```
//!
//! Also handles plain pythonic format without tags:
//! ```text
//! [get_weather(city='NYC'), get_time(tz='EST')]
//! ```
//!
//! Delegates to [`PythonicToolParser`] after stripping the Python tags.

use super::pythonic::parse_pythonic_tool_calls;
use super::{ToolCall, ToolCallParser};

const PYTHON_START_TAG: &str = "<|python_start|>";
const PYTHON_END_TAG: &str = "<|python_end|>";

/// Parser for Llama 4-style pythonic tool calls.
///
/// Strips `<|python_start|>` / `<|python_end|>` markers if present,
/// then parses pythonic `[func(args)]` expressions.
#[derive(Debug, Clone, Default)]
pub struct Llama4PythonicToolParser;

impl Llama4PythonicToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Strip Python tags and extract the inner expression.
    fn strip_tags(output: &str) -> &str {
        let s = output.trim();
        let s = s.strip_prefix(PYTHON_START_TAG).unwrap_or(s);
        let s = s.strip_suffix(PYTHON_END_TAG).unwrap_or(s);
        s.trim()
    }
}

impl ToolCallParser for Llama4PythonicToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let stripped = Self::strip_tags(output);

        // Must be wrapped in brackets to be pythonic
        if !stripped.starts_with('[') || !stripped.ends_with(']') {
            return Ok(Vec::new());
        }

        parse_pythonic_tool_calls(stripped)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        let trimmed = output.trim();

        // Find the earliest tool call marker
        let tag_pos = trimmed.find(PYTHON_START_TAG);
        let bracket_pos = trimmed.find('[');

        let marker_pos = match (tag_pos, bracket_pos) {
            (Some(t), Some(b)) => Some(t.min(b)),
            (Some(t), None) => Some(t),
            (None, Some(b)) => {
                // Only treat '[' as tool call if it parses successfully
                if self.parse(output).map(|c| !c.is_empty()).unwrap_or(false) {
                    Some(b)
                } else {
                    None
                }
            }
            (None, None) => None,
        };

        match marker_pos {
            Some(0) => None, // Starts with tool call
            Some(pos) => {
                let content = trimmed[..pos].trim();
                if content.is_empty() {
                    None
                } else {
                    Some(content.to_string())
                }
            }
            None => {
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_with_python_tags() {
        let parser = Llama4PythonicToolParser::new();
        let output = "<|python_start|>[get_weather(city='NYC')]<|python_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_without_python_tags() {
        let parser = Llama4PythonicToolParser::new();
        let output = "[get_weather(city='NYC')]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_multiple_calls() {
        let parser = Llama4PythonicToolParser::new();
        let output = "<|python_start|>[get_weather(city='NYC'), get_time(tz='EST')]<|python_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = Llama4PythonicToolParser::new();
        let calls = parser.parse("Just a response.").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_empty_list() {
        let parser = Llama4PythonicToolParser::new();
        let calls = parser.parse("<|python_start|>[]<|python_end|>").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_with_nested_args() {
        let parser = Llama4PythonicToolParser::new();
        let output = "[search(query='rust', filters={'lang': 'en'})]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["filters"]["lang"], "en");
    }

    #[test]
    fn extract_content_with_tags() {
        let parser = Llama4PythonicToolParser::new();
        let output = "<|python_start|>[test()]<|python_end|>";
        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = Llama4PythonicToolParser::new();
        let content = parser.extract_content("Normal text.").unwrap();
        assert_eq!(content, "Normal text.");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = Llama4PythonicToolParser::new();
        let output = "[test()]";
        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_python_values() {
        let parser = Llama4PythonicToolParser::new();
        let output = "[config(name='test', count=42, active=True, extra=None)]";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["name"], "test");
        assert_eq!(args["count"], 42);
        assert_eq!(args["active"], true);
        assert!(args["extra"].is_null());
    }
}
