//! Hunyuan-A13B tool call parser.
//!
//! Parses tool calls in the Hunyuan format:
//! ```text
//! <tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}]</tool_calls>
//! ```
//!
//! Handles `<think>...</think>` thinking sections — tool calls inside thinking
//! blocks are ignored.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALLS_START: &str = "<tool_calls>";
#[allow(dead_code)]
const TOOL_CALLS_END: &str = "</tool_calls>";

/// Regex to extract content between `<tool_calls>` tags.
static TOOL_CALLS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_calls>(.*?)</tool_calls>").expect("TOOL_CALLS_REGEX pattern is invalid")
});

/// Regex to detect `<think>...</think>` sections.
static THINK_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<think>.*?</think>").expect("THINK_REGEX pattern is invalid")
});

/// Parser for Hunyuan-A13B-style tool calls.
///
/// Extracts JSON arrays from `<tool_calls>` tags, ignoring any that
/// appear inside `<think>` blocks.
#[derive(Debug, Clone, Default)]
pub struct HunyuanToolParser;

impl HunyuanToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation for a Hunyuan tool call.
#[derive(Debug, serde::Deserialize)]
struct HunyuanToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for HunyuanToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALLS_START) {
            return Ok(Vec::new());
        }

        // Remove thinking blocks to avoid parsing tool calls inside them
        let cleaned = THINK_REGEX.replace_all(output, "");

        let mut calls = Vec::new();
        for cap in TOOL_CALLS_REGEX.captures_iter(&cleaned) {
            let json_str = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            if json_str.is_empty() {
                continue;
            }

            // Try parsing as JSON array
            if let Ok(arr) = serde_json::from_str::<Vec<HunyuanToolCallJson>>(json_str) {
                for tc in arr {
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
            } else if let Ok(tc) = serde_json::from_str::<HunyuanToolCallJson>(json_str) {
                // Single object without array wrapper
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
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(idx) = output.find(TOOL_CALLS_START) {
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
        let parser = HunyuanToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = HunyuanToolParser::new();
        let output = r#"<tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "get_time", "arguments": {"tz": "EST"}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = HunyuanToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn ignore_tool_calls_in_thinking() {
        let parser = HunyuanToolParser::new();
        let output = r#"<think>I should use <tool_calls>[{"name": "test", "arguments": {}}]</tool_calls></think>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_with_thinking_and_tool_calls() {
        let parser = HunyuanToolParser::new();
        let output = r#"<think>Let me think...</think><tool_calls>[{"name": "get_weather", "arguments": {"city": "NYC"}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = HunyuanToolParser::new();
        let output = r#"Here's what I found: <tool_calls>[{"name": "test", "arguments": {}}]</tool_calls>"#;

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Here's what I found:");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = HunyuanToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "test", "arguments": {}}]</tool_calls>"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = HunyuanToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "get_time", "arguments": {}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = HunyuanToolParser::new();
        let output =
            r#"<tool_calls>[{"name": "test", "arguments": {}}]</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_single_object_no_array() {
        let parser = HunyuanToolParser::new();
        let output =
            r#"<tool_calls>{"name": "test", "arguments": {"x": 1}}</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "test");
    }
}
