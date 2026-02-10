//! Seed-OSS tool call parser.
//!
//! Parses tool calls in the ByteDance Seed-OSS format:
//! ```text
//! <seed:tool_call>
//! <function=get_weather>
//! <parameter=city>NYC</parameter>
//! </function>
//! </seed:tool_call>
//! ```
//!
//! Also handles `<seed:think>...</seed:think>` thinking sections.
//! Parameters are extracted as key-value pairs from `<parameter=name>value</parameter>` tags.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const SEED_TOOL_CALL_START: &str = "<seed:tool_call>";

/// Regex to extract complete `<seed:tool_call>...</seed:tool_call>` blocks.
static SEED_TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<seed:tool_call>(.*?)</seed:tool_call>")
        .expect("SEED_TOOL_CALL_REGEX pattern is invalid")
});

/// Regex to extract function name from `<function=NAME>`.
static FUNCTION_NAME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<function=([^>]+)>").expect("FUNCTION_NAME_REGEX pattern is invalid")
});

/// Regex to extract parameters: `<parameter=KEY>VALUE</parameter>`.
static PARAMETER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
        .expect("PARAMETER_REGEX pattern is invalid")
});

/// Parser for Seed-OSS-style tool calls.
///
/// Extracts function calls from `<seed:tool_call>` blocks with
/// `<function=name>` and `<parameter=key>value</parameter>` tags.
#[derive(Debug, Clone, Default)]
pub struct SeedOssToolParser;

impl SeedOssToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a single `<seed:tool_call>` block content into a ToolCall.
    fn parse_tool_call_block(block: &str) -> Option<ToolCall> {
        let func_name = FUNCTION_NAME_REGEX
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

impl ToolCallParser for SeedOssToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(SEED_TOOL_CALL_START) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for cap in SEED_TOOL_CALL_REGEX.captures_iter(output) {
            let block = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            if let Some(call) = Self::parse_tool_call_block(block) {
                calls.push(call);
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(idx) = output.find(SEED_TOOL_CALL_START) {
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
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=get_weather>\n<parameter=city>NYC</parameter>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_params() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=search>\n<parameter=query>rust lang</parameter>\n<parameter=limit>10</parameter>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust lang");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=get_weather>\n<parameter=city>NYC</parameter>\n</function>\n</seed:tool_call>\n<seed:tool_call>\n<function=get_time>\n<parameter=tz>EST</parameter>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = SeedOssToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_with_thinking() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:think>I need to check the weather.</seed:think>\n<seed:tool_call>\n<function=get_weather>\n<parameter=city>NYC</parameter>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert!(content.contains("seed:think"));
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=test>\n</function>\n</seed:tool_call>";

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_no_params() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=get_time>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = SeedOssToolParser::new();
        let output = "Let me check. <seed:tool_call>\n<function=test>\n</function>\n</seed:tool_call>";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check.");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=test>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_string_value_with_spaces() {
        let parser = SeedOssToolParser::new();
        let output = "<seed:tool_call>\n<function=search>\n<parameter=query>hello world test</parameter>\n</function>\n</seed:tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "hello world test");
    }
}
