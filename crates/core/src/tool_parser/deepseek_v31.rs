//! DeepSeek V3.1 tool call parser.
//!
//! Parses tool calls in the DeepSeek V3.1 format using Unicode special tokens:
//! ```text
//! <｜tool▁calls▁begin｜>
//! <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>{"arg": "val"}<｜tool▁call▁end｜>
//! <｜tool▁calls▁end｜>
//! ```

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALLS_BEGIN: &str = "<｜tool\u{2581}calls\u{2581}begin｜>";
const TOOL_CALL_BEGIN: &str = "<｜tool\u{2581}call\u{2581}begin｜>";
#[allow(dead_code)]
const TOOL_CALL_END: &str = "<｜tool\u{2581}call\u{2581}end｜>";

/// Regex to match individual tool calls:
/// `<｜tool▁call▁begin｜>name<｜tool▁sep｜>arguments<｜tool▁call▁end｜>`
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "<｜tool\u{2581}call\u{2581}begin｜>(?P<name>.*?)<｜tool\u{2581}sep｜>(?P<args>.*?)<｜tool\u{2581}call\u{2581}end｜>",
    )
    .expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Parser for DeepSeek V3.1 tool calls.
///
/// Uses Unicode special tokens (`▁` = U+2581, `｜` = U+FF5C) as delimiters.
/// Each tool call contains the function name and arguments separated by `<｜tool▁sep｜>`.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV31ToolParser;

impl DeepSeekV31ToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for DeepSeekV31ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALL_BEGIN) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let name = cap.name("name").map(|m| m.as_str()).unwrap_or("").trim();
            let args = cap.name("args").map(|m| m.as_str()).unwrap_or("{}").trim();

            if name.is_empty() {
                continue;
            }

            calls.push(ToolCall {
                id: generate_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
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
        if !output.contains(TOOL_CALLS_BEGIN) && !output.contains(TOOL_CALL_BEGIN) {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }

        // Content is everything before the first tool calls begin marker
        let content = if let Some(pos) = output.find(TOOL_CALLS_BEGIN) {
            &output[..pos]
        } else if let Some(pos) = output.find(TOOL_CALL_BEGIN) {
            &output[..pos]
        } else {
            output
        };

        let trimmed = content.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "<｜tool\u{2581}calls\u{2581}begin｜><｜tool\u{2581}call\u{2581}begin｜>get_weather<｜tool\u{2581}sep｜>{\"city\": \"NYC\"}<｜tool\u{2581}call\u{2581}end｜><｜tool\u{2581}calls\u{2581}end｜>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "<｜tool\u{2581}calls\u{2581}begin｜>\
            <｜tool\u{2581}call\u{2581}begin｜>get_weather<｜tool\u{2581}sep｜>{\"city\": \"NYC\"}<｜tool\u{2581}call\u{2581}end｜>\
            <｜tool\u{2581}call\u{2581}begin｜>get_time<｜tool\u{2581}sep｜>{\"tz\": \"EST\"}<｜tool\u{2581}call\u{2581}end｜>\
            <｜tool\u{2581}calls\u{2581}end｜>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_with_preceding_content() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "Let me help you.\n<｜tool\u{2581}calls\u{2581}begin｜><｜tool\u{2581}call\u{2581}begin｜>get_weather<｜tool\u{2581}sep｜>{\"city\": \"NYC\"}<｜tool\u{2581}call\u{2581}end｜><｜tool\u{2581}calls\u{2581}end｜>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help you.");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "<｜tool\u{2581}calls\u{2581}begin｜><｜tool\u{2581}call\u{2581}begin｜>test<｜tool\u{2581}sep｜>{}<｜tool\u{2581}call\u{2581}end｜><｜tool\u{2581}calls\u{2581}end｜>";

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn tool_call_id_format() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "<｜tool\u{2581}calls\u{2581}begin｜><｜tool\u{2581}call\u{2581}begin｜>test<｜tool\u{2581}sep｜>{}<｜tool\u{2581}call\u{2581}end｜><｜tool\u{2581}calls\u{2581}end｜>";

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = DeepSeekV31ToolParser::new();
        let output = "<｜tool\u{2581}calls\u{2581}begin｜><｜tool\u{2581}call\u{2581}begin｜>get_time<｜tool\u{2581}sep｜><｜tool\u{2581}call\u{2581}end｜><｜tool\u{2581}calls\u{2581}end｜>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }
}
