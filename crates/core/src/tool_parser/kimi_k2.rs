//! Kimi K2 tool call parser.
//!
//! Parses tool calls in the Moonshot Kimi K2 format:
//! ```text
//! <|tool_calls_section_begin|>
//! <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "NYC"}<|tool_call_end|>
//! <|tool_calls_section_end|>
//! ```
//!
//! The tool call ID has format `functions.name:index` or `name:index`.
//! Function name is extracted as the part after the last `.` and before `:`.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALLS_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";

/// Regex to match individual tool calls:
/// `<|tool_call_begin|> ID <|tool_call_argument_begin|> ARGS <|tool_call_end|>`
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"<\|tool_call_begin\|>\s*(?P<id>[^<]+?)\s*<\|tool_call_argument_begin\|>\s*(?P<args>.*?)\s*<\|tool_call_end\|>",
    )
    .expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Parser for Kimi K2 tool calls.
///
/// Supports both `<|tool_calls_section_begin|>` and `<|tool_call_section_begin|>` variants.
/// Tool call IDs follow the format `functions.name:index` or `name:index`.
#[derive(Debug, Clone, Default)]
pub struct KimiK2ToolParser;

impl KimiK2ToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Extract function name from a tool call ID like "functions.get_weather:0".
    fn extract_function_name(tool_id: &str) -> &str {
        let before_colon = tool_id.split(':').next().unwrap_or(tool_id);
        before_colon.rsplit('.').next().unwrap_or(before_colon)
    }
}

impl ToolCallParser for KimiK2ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALL_BEGIN) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let tool_id = cap.name("id").map(|m| m.as_str().trim()).unwrap_or("");
            let args = cap.name("args").map(|m| m.as_str().trim()).unwrap_or("{}");

            let function_name = Self::extract_function_name(tool_id);
            if function_name.is_empty() {
                continue;
            }

            calls.push(ToolCall {
                id: generate_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: function_name.to_string(),
                    arguments: if args.is_empty() {
                        "{}".to_string()
                    } else {
                        args.to_string()
                    },
                },
            });
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        let marker = if output.contains(TOOL_CALLS_SECTION_BEGIN) {
            TOOL_CALLS_SECTION_BEGIN
        } else if output.contains(TOOL_CALL_BEGIN) {
            TOOL_CALL_BEGIN
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
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"NYC\"}<|tool_call_end|><|tool_calls_section_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|>\
            <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"NYC\"}<|tool_call_end|>\
            <|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"tz\": \"EST\"}<|tool_call_end|>\
            <|tool_calls_section_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_simple_name_format() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|><|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{\"city\": \"NYC\"}<|tool_call_end|><|tool_calls_section_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_with_preceding_content() {
        let parser = KimiK2ToolParser::new();
        let output = "Let me check that for you.\n<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"NYC\"}<|tool_call_end|><|tool_calls_section_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check that for you.");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = KimiK2ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn extract_function_name_from_id() {
        assert_eq!(
            KimiK2ToolParser::extract_function_name("functions.get_weather:0"),
            "get_weather"
        );
        assert_eq!(
            KimiK2ToolParser::extract_function_name("get_weather:0"),
            "get_weather"
        );
        assert_eq!(
            KimiK2ToolParser::extract_function_name("a.b.get_weather:1"),
            "get_weather"
        );
        assert_eq!(
            KimiK2ToolParser::extract_function_name("get_weather"),
            "get_weather"
        );
    }

    #[test]
    fn tool_call_id_format() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_complex_arguments() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|><|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{\"query\": \"test\", \"limit\": 10, \"filters\": {\"type\": \"article\"}}<|tool_call_end|><|tool_calls_section_end|>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "test");
        assert_eq!(args["filters"]["type"], "article");
    }
}
