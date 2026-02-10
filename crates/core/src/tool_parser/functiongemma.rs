//! FunctionGemma tool call parser.
//!
//! Parses tool calls in the Google FunctionGemma format:
//! ```text
//! <start_function_call>call:get_weather{city:<escape>NYC<escape>}<end_function_call>
//! ```
//!
//! Parameters use `<escape>` delimiters around values.
//! Multiple parameters are separated within `{param1:<escape>val1<escape>, param2:<escape>val2<escape>}`.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const START_TAG: &str = "<start_function_call>";
#[allow(dead_code)]
const END_TAG: &str = "<end_function_call>";

/// Regex to match complete function calls:
/// `<start_function_call>call:NAME{PARAMS}<end_function_call>`
static COMPLETE_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<start_function_call>call:(?P<name>\w+)\{(?P<params>.*?)\}<end_function_call>")
        .expect("COMPLETE_CALL_REGEX pattern is invalid")
});

/// Regex to extract individual parameters: `key:<escape>value<escape>`
static PARAM_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\w+):<escape>(.*?)<escape>").expect("PARAM_REGEX pattern is invalid")
});

/// Parser for FunctionGemma-style tool calls.
///
/// Extracts function name and parameters from `call:name{...}` blocks
/// with `<escape>` delimited values.
#[derive(Debug, Clone, Default)]
pub struct FunctionGemmaToolParser;

impl FunctionGemmaToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse parameter string into a JSON object.
    /// Parameters are in format: `key:<escape>value<escape>, key2:<escape>value2<escape>`
    fn parse_params(params_str: &str) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for cap in PARAM_REGEX.captures_iter(params_str) {
            let key = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let value = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            if !key.is_empty() {
                // Try to parse as a typed JSON value (number, bool, etc.)
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(value) {
                    map.insert(key.to_string(), v);
                } else {
                    map.insert(
                        key.to_string(),
                        serde_json::Value::String(value.to_string()),
                    );
                }
            }
        }
        serde_json::Value::Object(map)
    }
}

impl ToolCallParser for FunctionGemmaToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(START_TAG) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for cap in COMPLETE_CALL_REGEX.captures_iter(output) {
            let name = cap.name("name").map(|m| m.as_str()).unwrap_or("");
            let params_str = cap.name("params").map(|m| m.as_str()).unwrap_or("");

            if name.is_empty() {
                continue;
            }

            let arguments = if params_str.is_empty() {
                "{}".to_string()
            } else {
                let params_json = Self::parse_params(params_str);
                serde_json::to_string(&params_json)?
            };

            calls.push(ToolCall {
                id: generate_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            });
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(idx) = output.find(START_TAG) {
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
        let parser = FunctionGemmaToolParser::new();
        let output =
            "<start_function_call>call:get_weather{city:<escape>NYC<escape>}<end_function_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_params() {
        let parser = FunctionGemmaToolParser::new();
        let output = "<start_function_call>call:search{query:<escape>rust lang<escape>, limit:<escape>10<escape>}<end_function_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust lang");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = FunctionGemmaToolParser::new();
        let output = "<start_function_call>call:get_weather{city:<escape>NYC<escape>}<end_function_call><start_function_call>call:get_time{tz:<escape>EST<escape>}<end_function_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = FunctionGemmaToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_no_params() {
        let parser = FunctionGemmaToolParser::new();
        let output = "<start_function_call>call:get_time{}<end_function_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = FunctionGemmaToolParser::new();
        let output = "Let me help you. <start_function_call>call:test{}<end_function_call>";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help you.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = FunctionGemmaToolParser::new();
        let output = "<start_function_call>call:test{}<end_function_call>";

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn tool_call_id_format() {
        let parser = FunctionGemmaToolParser::new();
        let output = "<start_function_call>call:test{}<end_function_call>";

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_boolean_param() {
        let parser = FunctionGemmaToolParser::new();
        let output =
            "<start_function_call>call:set_alarm{enabled:<escape>true<escape>}<end_function_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["enabled"], true);
    }
}
