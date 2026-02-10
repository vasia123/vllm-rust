//! ERNIE-4.5 tool call parser.
//!
//! Parses tool calls in the Baidu ERNIE-4.5 format:
//! ```text
//! <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
//! ```
//!
//! Supports `<think>...</think>` thinking tags and `<response>...</response>` tags.
//! Multiple tool calls use repeated `<tool_call>...</tool_call>` blocks.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALL_START: &str = "<tool_call>";

/// Regex to match `<tool_call>JSON</tool_call>` blocks.
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>")
        .expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Parser for ERNIE-4.5-style tool calls.
///
/// Extracts JSON tool calls from `<tool_call>` tags. Preserves
/// thinking and response content as extracted text.
#[derive(Debug, Clone, Default)]
pub struct Ernie45ToolParser;

impl Ernie45ToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation for an ERNIE-4.5 tool call.
#[derive(Debug, serde::Deserialize)]
struct Ernie45ToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for Ernie45ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALL_START) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let json_str = cap.name("json").map(|m| m.as_str().trim()).unwrap_or("");
            if json_str.is_empty() {
                continue;
            }

            match serde_json::from_str::<Ernie45ToolCallJson>(json_str) {
                Ok(tc) => {
                    let arguments = match &tc.arguments {
                        serde_json::Value::Null => "{}".to_string(),
                        serde_json::Value::Object(_) => serde_json::to_string(&tc.arguments)?,
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
                Err(e) => {
                    tracing::warn!("Failed to parse ERNIE-4.5 tool call JSON: {json_str}: {e}");
                }
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(idx) = output.find(TOOL_CALL_START) {
            let content = output[..idx].trim();
            if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            }
        } else {
            let trimmed = output.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let parser = Ernie45ToolParser::new();
        let output =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = Ernie45ToolParser::new();
        let output = r#"<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"tz": "EST"}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_with_thinking() {
        let parser = Ernie45ToolParser::new();
        let output = r#"<think>I need to check the weather.</think><tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert!(content.contains("think"));
    }

    #[test]
    fn parse_with_response_tag() {
        let parser = Ernie45ToolParser::new();
        let output = r#"<think>thinking...</think><response>Let me check.</response><tool_call>{"name": "test", "arguments": {}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert!(content.contains("response"));
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = Ernie45ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = Ernie45ToolParser::new();
        let output = r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = Ernie45ToolParser::new();
        let output = r#"<tool_call>{"name": "get_time", "arguments": {}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = Ernie45ToolParser::new();
        let output = r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
