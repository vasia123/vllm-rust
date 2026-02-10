//! Qwen3 Coder tool call parser.
//!
//! Parses tool calls in the Qwen3 Coder XML format:
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>NYC</parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Similar to Seed-OSS but uses `<tool_call>` wrapper instead of `<seed:tool_call>`,
//! and `<function=name>` / `<parameter=key>value</parameter>` tags.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALL_START: &str = "<tool_call>";
const FUNCTION_PREFIX: &str = "<function=";

/// Regex to match complete `<tool_call>...</tool_call>` blocks.
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Regex to extract function name from `<function=NAME>`.
static FUNCTION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<function=([^>]+)>").expect("FUNCTION_REGEX pattern is invalid"));

/// Regex to extract parameters: `<parameter=KEY>VALUE</parameter>`.
static PARAMETER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
        .expect("PARAMETER_REGEX pattern is invalid")
});

/// Parser for Qwen3 Coder-style tool calls.
///
/// Extracts function calls from `<tool_call>` blocks with
/// `<function=name>` and `<parameter=key>value</parameter>` tags.
#[derive(Debug, Clone, Default)]
pub struct Qwen3CoderToolParser;

impl Qwen3CoderToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a single `<tool_call>` block content.
    fn parse_block(block: &str) -> Option<ToolCall> {
        let func_name = FUNCTION_REGEX
            .captures(block)?
            .get(1)
            .map(|m| m.as_str().trim())?;

        if func_name.is_empty() {
            return None;
        }

        // Extract parameters
        let mut params = serde_json::Map::new();
        for cap in PARAMETER_REGEX.captures_iter(block) {
            let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let value = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if !key.is_empty() {
                // Try to parse as typed JSON value
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(value) {
                    params.insert(key.to_string(), v);
                } else {
                    params.insert(
                        key.to_string(),
                        serde_json::Value::String(value.to_string()),
                    );
                }
            }
        }

        let arguments = if params.is_empty() {
            "{}".to_string()
        } else {
            serde_json::to_string(&serde_json::Value::Object(params)).ok()?
        };

        Some(ToolCall {
            id: generate_tool_call_id(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: func_name.to_string(),
                arguments,
            },
        })
    }
}

impl ToolCallParser for Qwen3CoderToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(FUNCTION_PREFIX) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();

        // Try to find <tool_call>...</tool_call> blocks first
        let mut found_blocks = false;
        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            found_blocks = true;
            let block = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            if let Some(call) = Self::parse_block(block) {
                calls.push(call);
            }
        }

        // Fallback: parse directly if no <tool_call> wrapper
        if !found_blocks {
            if let Some(call) = Self::parse_block(output) {
                calls.push(call);
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        let marker = if output.contains(TOOL_CALL_START) {
            TOOL_CALL_START
        } else if output.contains(FUNCTION_PREFIX) {
            FUNCTION_PREFIX
        } else {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        };

        let content = output.split(marker).next().unwrap_or("").trim();
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
        let parser = Qwen3CoderToolParser::new();
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>NYC</parameter>\n</function>\n</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = Qwen3CoderToolParser::new();
        let output = "<tool_call>\n<function=get_weather>\n<parameter=city>NYC</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=tz>EST</parameter>\n</function>\n</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_multiple_params() {
        let parser = Qwen3CoderToolParser::new();
        let output = "<tool_call>\n<function=search>\n<parameter=query>rust</parameter>\n<parameter=limit>10</parameter>\n</function>\n</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = Qwen3CoderToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_without_tool_call_wrapper() {
        let parser = Qwen3CoderToolParser::new();
        let output = "<function=get_weather>\n<parameter=city>NYC</parameter>\n</function>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = Qwen3CoderToolParser::new();
        let output = "Let me check.\n<tool_call>\n<function=test>\n</function>\n</tool_call>";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = Qwen3CoderToolParser::new();
        let output = "<tool_call>\n<function=test>\n</function>\n</tool_call>";

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_no_parameters() {
        let parser = Qwen3CoderToolParser::new();
        let output = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = Qwen3CoderToolParser::new();
        let output = "<tool_call>\n<function=test>\n</function>\n</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
