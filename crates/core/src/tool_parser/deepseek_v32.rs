//! DeepSeek V3.2 tool call parser.
//!
//! Parses tool calls in the DeepSeek V3.2 DSML format:
//! ```text
//! <｜DSML｜function_calls>
//! <｜DSML｜invoke name="get_weather">
//! <｜DSML｜parameter name="city" string="true">NYC</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜function_calls>
//! ```
//!
//! Uses fullwidth Unicode tokens (`｜` = U+FF5C) like other DeepSeek variants.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const FUNCTION_CALLS_START: &str = "<｜DSML｜function_calls>";

/// Regex to match complete `<｜DSML｜function_calls>...</｜DSML｜function_calls>` blocks.
static FUNCTION_CALLS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>")
        .expect("FUNCTION_CALLS_REGEX pattern is invalid")
});

/// Regex to match individual invocations:
/// `<｜DSML｜invoke name="NAME">PARAMS</｜DSML｜invoke>`
static INVOKE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</｜DSML｜invoke>"#)
        .expect("INVOKE_REGEX pattern is invalid")
});

/// Regex to match parameters:
/// `<｜DSML｜parameter name="KEY" string="true|false">VALUE</｜DSML｜parameter>`
static PARAMETER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?s)<｜DSML｜parameter\s+name="(?P<key>[^"]+)"\s+string="(?:true|false)"\s*>(?P<val>.*?)</｜DSML｜parameter>"#,
    )
    .expect("PARAMETER_REGEX pattern is invalid")
});

/// Parser for DeepSeek V3.2 DSML-style tool calls.
///
/// Extracts function invocations from `<｜DSML｜function_calls>` blocks,
/// parsing `<｜DSML｜invoke>` and `<｜DSML｜parameter>` tags.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV32ToolParser;

impl DeepSeekV32ToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for DeepSeekV32ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(FUNCTION_CALLS_START) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();

        for fc_cap in FUNCTION_CALLS_REGEX.captures_iter(output) {
            let block = fc_cap.get(1).map(|m| m.as_str()).unwrap_or("");

            for invoke_cap in INVOKE_REGEX.captures_iter(block) {
                let name = invoke_cap.name("name").map(|m| m.as_str()).unwrap_or("");
                let body = invoke_cap.name("body").map(|m| m.as_str()).unwrap_or("");

                if name.is_empty() {
                    continue;
                }

                // Extract parameters
                let mut params = serde_json::Map::new();
                for param_cap in PARAMETER_REGEX.captures_iter(body) {
                    let key = param_cap.name("key").map(|m| m.as_str()).unwrap_or("");
                    let val = param_cap.name("val").map(|m| m.as_str()).unwrap_or("");
                    if !key.is_empty() {
                        // Try to parse as typed JSON value
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(val) {
                            params.insert(key.to_string(), v);
                        } else {
                            params.insert(
                                key.to_string(),
                                serde_json::Value::String(val.to_string()),
                            );
                        }
                    }
                }

                let arguments = serde_json::to_string(&serde_json::Value::Object(params))?;

                calls.push(ToolCall {
                    id: generate_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(idx) = output.find(FUNCTION_CALLS_START) {
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
    fn parse_single_invoke() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_invokes() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"get_time\">\n",
            "<｜DSML｜parameter name=\"tz\" string=\"true\">EST</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_multiple_parameters() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"search\">\n",
            "<｜DSML｜parameter name=\"query\" string=\"true\">rust lang</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"limit\" string=\"false\">10</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust lang");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = DeepSeekV32ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "Let me check.\n",
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"test\">\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"test\">\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn tool_call_id_format() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"test\">\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_no_parameters() {
        let parser = DeepSeekV32ToolParser::new();
        let output = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"get_time\">\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }
}
