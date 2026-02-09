//! DeepSeek V3-style tool call parser.
//!
//! Uses Unicode special tokens to delimit tool calls:
//! ```text
//! <｜tool▁calls▁begin｜>
//! <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
//! ```json
//! {"arg": "value"}
//! ```<｜tool▁call▁end｜>
//! <｜tool▁calls▁end｜>
//! ```
//!
//! The Unicode tokens use fullwidth vertical bars (U+FF5C `｜`) and
//! lower one-eighth block (U+2581 `▁`).
//!
//! This format is used by DeepSeek-V3 and related models.

use regex::Regex;
use std::sync::LazyLock;

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};

/// Opening tag for the tool calls section.
const TOOL_CALLS_BEGIN: &str = "<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>";

/// Regex to extract individual tool calls from the Unicode-delimited format.
///
/// Captures:
/// - `type`: the call type (e.g. "function")
/// - `name`: the function name
/// - `args`: the JSON arguments inside the ` ```json ``` ` code block
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"<\x{FF5C}tool\x{2581}call\x{2581}begin\x{FF5C}>",
        r"(?P<type>[^<]*)",
        r"<\x{FF5C}tool\x{2581}sep\x{FF5C}>",
        r"(?P<name>[^\n]*)\n",
        r"```json\n",
        r"(?P<args>[\s\S]*?)",
        r"\n```",
        r"<\x{FF5C}tool\x{2581}call\x{2581}end\x{FF5C}>",
    ))
    .expect("DEEPSEEK_TOOL_CALL_REGEX is invalid")
});

/// Parser for DeepSeek V3-style tool calls.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV3ToolParser;

impl ToolCallParser for DeepSeekV3ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut calls = Vec::new();

        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let name = cap.name("name").map(|m| m.as_str().trim()).unwrap_or("");
            let args_str = cap.name("args").map(|m| m.as_str().trim()).unwrap_or("{}");

            if name.is_empty() {
                continue;
            }

            // Validate and normalize JSON arguments
            let arguments = match serde_json::from_str::<serde_json::Value>(args_str) {
                Ok(serde_json::Value::Null) => "{}".to_string(),
                Ok(val) => serde_json::to_string(&val)?,
                Err(_) => args_str.to_string(),
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
        let text = output.trim();

        if let Some(idx) = text.find(TOOL_CALLS_BEGIN) {
            let content = text[..idx].trim();
            return if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            };
        }

        // No tool calls markers — whole output is content
        Some(text.to_string())
    }
}

impl DeepSeekV3ToolParser {
    /// Create a new DeepSeek V3 tool parser.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a single tool call in DeepSeek V3 format.
    fn ds_tool_call(name: &str, args: &str) -> String {
        format!(
            "<\u{FF5C}tool\u{2581}call\u{2581}begin\u{FF5C}>function\
             <\u{FF5C}tool\u{2581}sep\u{FF5C}>{name}\n\
             ```json\n\
             {args}\n\
             ```\
             <\u{FF5C}tool\u{2581}call\u{2581}end\u{FF5C}>"
        )
    }

    /// Helper: wrap tool calls in the begin/end section markers.
    fn ds_tool_section(inner: &str) -> String {
        format!(
            "<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>\
             {inner}\
             <\u{FF5C}tool\u{2581}calls\u{2581}end\u{FF5C}>"
        )
    }

    #[test]
    fn parse_single_tool_call() {
        let parser = DeepSeekV3ToolParser::new();
        let call = ds_tool_call("get_weather", r#"{"city": "NYC"}"#);
        let output = ds_tool_section(&call);

        let calls = parser.parse(&output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = DeepSeekV3ToolParser::new();
        let call1 = ds_tool_call("get_weather", r#"{"city": "NYC"}"#);
        let call2 = ds_tool_call("get_time", r#"{"tz": "EST"}"#);
        let output = ds_tool_section(&format!("{call1}\n{call2}"));

        let calls = parser.parse(&output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = DeepSeekV3ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = DeepSeekV3ToolParser::new();
        let call = ds_tool_call("ping", "{}");
        let output = ds_tool_section(&call);

        let calls = parser.parse(&output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "ping");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_complex_nested_arguments() {
        let parser = DeepSeekV3ToolParser::new();
        let args = r#"{"data": [1, 2, 3], "config": {"verbose": true}}"#;
        let call = ds_tool_call("analyze", args);
        let output = ds_tool_section(&call);

        let calls = parser.parse(&output).unwrap();
        assert_eq!(calls.len(), 1);
        let parsed_args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(parsed_args["data"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn extract_content_before_tools() {
        let parser = DeepSeekV3ToolParser::new();
        let call = ds_tool_call("fn", "{}");
        let output = format!("Let me help you.{}", ds_tool_section(&call));

        let content = parser.extract_content(&output).unwrap();
        assert_eq!(content, "Let me help you.");
    }

    #[test]
    fn extract_content_only_tools() {
        let parser = DeepSeekV3ToolParser::new();
        let call = ds_tool_call("fn", "{}");
        let output = ds_tool_section(&call);

        assert!(parser.extract_content(&output).is_none());
    }

    #[test]
    fn extract_content_no_tools() {
        let parser = DeepSeekV3ToolParser::new();
        let output = "Normal response without tools.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Normal response without tools.");
    }

    #[test]
    fn has_tool_calls_positive() {
        let parser = DeepSeekV3ToolParser::new();
        let call = ds_tool_call("fn", "{}");
        let output = ds_tool_section(&call);
        assert!(parser.has_tool_calls(&output));
    }

    #[test]
    fn has_tool_calls_negative() {
        let parser = DeepSeekV3ToolParser::new();
        assert!(!parser.has_tool_calls("no tool calls here"));
    }

    #[test]
    fn parse_empty_input() {
        let parser = DeepSeekV3ToolParser::new();
        assert!(parser.parse("").unwrap().is_empty());
    }

    #[test]
    fn parse_tool_name_whitespace_trimmed() {
        let parser = DeepSeekV3ToolParser::new();
        let call = ds_tool_call("  get_weather  ", r#"{"city": "NYC"}"#);
        let output = ds_tool_section(&call);

        let calls = parser.parse(&output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }
}
