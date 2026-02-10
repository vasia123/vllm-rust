//! Step-3 / Step-3.5 tool call parser.
//!
//! Parses tool calls using Unicode-delimited blocks and steptml XML:
//! ```text
//! <｜tool_calls_begin｜>
//! <｜tool_call_begin｜>function<｜tool_sep｜>
//! <steptml:invoke name="get_weather">
//! <steptml:parameter name="city">NYC</steptml:parameter>
//! </steptml:invoke>
//! <｜tool_call_end｜>
//! <｜tool_calls_end｜>
//! ```
//!
//! The outer delimiters use fullwidth Unicode (`｜` = U+FF5C).
//! The inner content uses steptml XML format for function name and parameters.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALLS_BEGIN: &str = "<｜tool_calls_begin｜>";
const TOOL_CALLS_END: &str = "<｜tool_calls_end｜>";
const TOOL_CALL_BEGIN: &str = "<｜tool_call_begin｜>";
const TOOL_CALL_END: &str = "<｜tool_call_end｜>";
const TOOL_SEP: &str = "<｜tool_sep｜>";

/// Regex to extract function name from `<steptml:invoke name="NAME">`.
static STEPTML_INVOKE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<steptml:invoke\s+name="(?P<name>[^"]+)">"#)
        .expect("STEPTML_INVOKE_REGEX pattern is invalid")
});

/// Regex to extract parameters: `<steptml:parameter name="KEY">VALUE</steptml:parameter>`.
static STEPTML_PARAM_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<steptml:parameter\s+name="(?P<key>[^"]+)">(?P<val>[^<]*)</steptml:parameter>"#)
        .expect("STEPTML_PARAM_REGEX pattern is invalid")
});

/// Parser for Step-3 and Step-3.5 tool calls.
///
/// Extracts steptml XML invocations from Unicode-delimited tool call blocks.
#[derive(Debug, Clone, Default)]
pub struct Step3ToolParser;

impl Step3ToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a single tool call block (content between tool_call_begin and tool_call_end).
    fn parse_tool_call_block(block: &str) -> Option<ToolCall> {
        // The block format is: "function<｜tool_sep｜><steptml:invoke ...>...</steptml:invoke>"
        // Split on tool_sep to get the invoke part
        let invoke_part = if let Some((_type_part, rest)) = block.split_once(TOOL_SEP) {
            rest
        } else {
            block
        };

        // Extract function name
        let name = STEPTML_INVOKE_REGEX
            .captures(invoke_part)?
            .name("name")
            .map(|m| m.as_str())?;

        if name.is_empty() {
            return None;
        }

        // Extract parameters
        let mut params = serde_json::Map::new();
        for cap in STEPTML_PARAM_REGEX.captures_iter(invoke_part) {
            let key = cap.name("key").map(|m| m.as_str()).unwrap_or("");
            let val = cap.name("val").map(|m| m.as_str().trim()).unwrap_or("");
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

        let arguments = serde_json::to_string(&serde_json::Value::Object(params)).ok()?;

        Some(ToolCall {
            id: generate_tool_call_id(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments,
            },
        })
    }
}

impl ToolCallParser for Step3ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALLS_BEGIN) {
            return Ok(Vec::new());
        }

        // Extract the tool block between tool_calls_begin and tool_calls_end
        let rest = match output.split_once(TOOL_CALLS_BEGIN) {
            Some((_, rest)) => rest,
            None => return Ok(Vec::new()),
        };

        let tool_block = if let Some((block, _)) = rest.split_once(TOOL_CALLS_END) {
            block
        } else {
            rest
        };

        // Split by tool_call_begin markers and parse each
        let mut calls = Vec::new();
        for part in tool_block.split(TOOL_CALL_BEGIN) {
            if part.is_empty() {
                continue;
            }

            // Get content before tool_call_end
            let call_content = if let Some((content, _)) = part.split_once(TOOL_CALL_END) {
                content
            } else {
                continue;
            };

            if let Some(call) = Self::parse_tool_call_block(call_content) {
                calls.push(call);
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(idx) = output.find(TOOL_CALLS_BEGIN) {
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
        let parser = Step3ToolParser::new();
        let output = concat!(
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"get_weather\">",
            "<steptml:parameter name=\"city\">NYC</steptml:parameter>",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = Step3ToolParser::new();
        let output = concat!(
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"get_weather\">",
            "<steptml:parameter name=\"city\">NYC</steptml:parameter>",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"get_time\">",
            "<steptml:parameter name=\"tz\">EST</steptml:parameter>",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_multiple_parameters() {
        let parser = Step3ToolParser::new();
        let output = concat!(
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"search\">",
            "<steptml:parameter name=\"query\">rust lang</steptml:parameter>",
            "<steptml:parameter name=\"limit\">10</steptml:parameter>",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust lang");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = Step3ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = Step3ToolParser::new();
        let output = concat!(
            "Let me check.\n",
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"test\">",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = Step3ToolParser::new();
        let output = concat!(
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"test\">",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_no_parameters() {
        let parser = Step3ToolParser::new();
        let output = concat!(
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"get_time\">",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = Step3ToolParser::new();
        let output = concat!(
            "<｜tool_calls_begin｜>",
            "<｜tool_call_begin｜>function<｜tool_sep｜>",
            "<steptml:invoke name=\"test\">",
            "</steptml:invoke>",
            "<｜tool_call_end｜>",
            "<｜tool_calls_end｜>"
        );

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
