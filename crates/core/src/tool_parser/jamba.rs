//! Jamba-style tool call parser.
//!
//! Parses tool calls in the Jamba (AI21) format:
//! ```text
//! <tool_calls>[{"name": "func", "arguments": {...}}, ...]</tool_calls>
//! ```
//!
//! This format is used by AI21 Jamba models (Jamba 1.5 Large, Jamba 1.5 Mini).
//! Content between tags is a JSON array of tool call objects.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALLS_START: &str = "<tool_calls>";

/// Regex to capture content between `<tool_calls>` tags.
static TOOL_CALLS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_calls>\s*(.*?)\s*</tool_calls>")
        .expect("TOOL_CALLS_REGEX pattern is invalid")
});

/// Parser for Jamba-style tool calls.
///
/// Jamba uses XML-like tags wrapping a JSON array:
/// ```text
/// <tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}]</tool_calls>
/// ```
///
/// Supports parallel tool calls as array elements.
#[derive(Debug, Clone, Default)]
pub struct JambaToolParser;

impl JambaToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation for a single Jamba tool call.
#[derive(Debug, serde::Deserialize)]
struct JambaToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for JambaToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALLS_START) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();

        for cap in TOOL_CALLS_REGEX.captures_iter(output) {
            let json_str = cap.get(1).map(|m| m.as_str()).unwrap_or("");

            match serde_json::from_str::<Vec<JambaToolCallJson>>(json_str) {
                Ok(tool_call_arr) => {
                    for tc in tool_call_arr {
                        let arguments = match &tc.arguments {
                            serde_json::Value::Null => "{}".to_string(),
                            serde_json::Value::Object(_) => {
                                serde_json::to_string(&tc.arguments)?
                            }
                            other => other.to_string(),
                        };

                        calls.push(ToolCall {
                            id: generate_tool_call_id(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: tc.name,
                                arguments,
                            },
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to parse Jamba tool calls JSON: {json_str}: {e}");
                }
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if !output.contains(TOOL_CALLS_START) {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }

        // Content is everything before the first <tool_calls> tag
        let content = output
            .split(TOOL_CALLS_START)
            .next()
            .unwrap_or("")
            .trim();

        if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let parser = JambaToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "get_time", "arguments": {"tz": "EST"}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_with_preceding_content() {
        let parser = JambaToolParser::new();
        let output = r#"Let me help you with that.
<tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help you with that.");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = JambaToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = JambaToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "test", "arguments": {}}]</tool_calls>"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = JambaToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "get_time"}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_complex_arguments() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name": "search", "arguments": {"query": "test", "limit": 10, "filters": {"type": "article"}}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "test");
        assert_eq!(args["limit"], 10);
        assert_eq!(args["filters"]["type"], "article");
    }

    #[test]
    fn parse_multiline_json() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>
[
    {
        "name": "get_weather",
        "arguments": {
            "city": "NYC"
        }
    }
]
</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = JambaToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "test", "arguments": {}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn has_tool_calls_detection() {
        let parser = JambaToolParser::new();
        assert!(parser.has_tool_calls(
            r#"<tool_calls>[{"name": "test", "arguments": {}}]</tool_calls>"#
        ));
        assert!(!parser.has_tool_calls("no tool calls here"));
    }

    #[test]
    fn parse_ignores_malformed_json() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>not valid json</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }
}
