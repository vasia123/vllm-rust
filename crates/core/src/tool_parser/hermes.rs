//! Hermes-style tool call parser.
//!
//! Parses tool calls in the Hermes format:
//! ```text
//! <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
//! ```
//!
//! This format is used by several open-source models including:
//! - NousResearch/Hermes models
//! - Some Mistral fine-tunes
//! - Various function-calling fine-tuned models

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

/// Parser for Hermes-style tool calls.
///
/// Hermes format uses XML-like tags:
/// ```text
/// <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
/// ```
#[derive(Debug, Clone, Default)]
pub struct HermesToolParser;

// Regex to match tool_call tags and capture JSON content
// Uses (?s) to make . match newlines, captures all content between tags
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>")
        .expect("TOOL_CALL_REGEX pattern is invalid - this is a build-time error")
});

// Regex to match any tool_call content (including malformed)
static TOOL_CALL_TAG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>[\s\S]*?</tool_call>")
        .expect("TOOL_CALL_TAG_REGEX pattern is invalid - this is a build-time error")
});

/// Internal representation of parsed Hermes tool call.
#[derive(Debug, serde::Deserialize)]
struct HermesToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for HermesToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut calls = Vec::new();

        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let json_str = cap.get(1).map(|m| m.as_str()).unwrap_or("");

            // Try to parse the JSON
            match serde_json::from_str::<HermesToolCallJson>(json_str) {
                Ok(parsed) => {
                    // Convert arguments to string
                    let arguments = match &parsed.arguments {
                        serde_json::Value::Null => "{}".to_string(),
                        serde_json::Value::Object(_) => serde_json::to_string(&parsed.arguments)?,
                        other => other.to_string(),
                    };

                    calls.push(ToolCall {
                        id: generate_tool_call_id(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: parsed.name,
                            arguments,
                        },
                    });
                }
                Err(e) => {
                    // Log but continue trying to parse other calls
                    tracing::warn!("Failed to parse tool call JSON: {}: {}", json_str, e);
                }
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        // Remove all tool_call tags and return remaining content
        let content = TOOL_CALL_TAG_REGEX.replace_all(output, "");
        let trimmed = content.trim();

        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn has_tool_calls(&self, output: &str) -> bool {
        // Check if there are any valid, parseable tool calls
        self.parse(output).map(|c| !c.is_empty()).unwrap_or(false)
    }
}

impl HermesToolParser {
    /// Create a new Hermes tool parser.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_tool_call() {
        let parser = HermesToolParser::new();
        let output =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let parser = HermesToolParser::new();
        let output = r#"
            <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
            <tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>
        "#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn test_parse_tool_call_with_content() {
        let parser = HermesToolParser::new();
        let output = r#"Let me check the weather for you.
<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
I'll get that information now."#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert!(content.contains("Let me check the weather"));
        assert!(content.contains("I'll get that information"));
        assert!(!content.contains("tool_call"));
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let parser = HermesToolParser::new();
        let output = "Just a normal response without any tool calls.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn test_extract_content_only_tool_call() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn test_parse_empty_arguments() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name": "get_time"}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_parse_complex_arguments() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name": "search", "arguments": {"query": "test", "limit": 10, "filters": {"type": "article"}}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "test");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn test_has_tool_calls() {
        let parser = HermesToolParser::new();

        assert!(
            parser.has_tool_calls(r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#)
        );
        assert!(!parser.has_tool_calls("no tool calls here"));
        assert!(!parser.has_tool_calls("<tool_call>invalid json</tool_call>")); // Invalid JSON doesn't match
    }

    #[test]
    fn test_parse_whitespace_handling() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>
            {
                "name": "get_weather",
                "arguments": {
                    "city": "NYC"
                }
            }
        </tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }
}
