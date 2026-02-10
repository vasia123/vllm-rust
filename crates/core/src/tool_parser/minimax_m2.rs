//! MiniMax M2 tool call parser.
//!
//! Parses tool calls in XML invoke/parameter format:
//! ```text
//! <minimax:tool_call>
//! <invoke name="get_weather">
//! <parameter name="city">NYC</parameter>
//! <parameter name="metric">celsius</parameter>
//! </invoke>
//! </minimax:tool_call>
//! ```
//!
//! Values are type-inferred: strings stay as-is, numbers/booleans/null are
//! deserialized as their JSON equivalents.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const START_TAG: &str = "<minimax:tool_call>";

/// Regex to extract `<minimax:tool_call>...</minimax:tool_call>` blocks.
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<minimax:tool_call>(.*?)</minimax:tool_call>")
        .expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Regex to extract function name from `<invoke name="NAME">`.
static INVOKE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<invoke\s+name="([^"]+)">(.*?)</invoke>"#)
        .expect("INVOKE_REGEX pattern is invalid")
});

/// Regex to extract parameters: `<parameter name="KEY">VALUE</parameter>`.
static PARAMETER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<parameter\s+name="([^"]+)">(.*?)</parameter>"#)
        .expect("PARAMETER_REGEX pattern is invalid")
});

/// Parser for MiniMax M2-style tool calls.
///
/// XML-based format with `<invoke>` and `<parameter>` tags inside
/// `<minimax:tool_call>` blocks.
#[derive(Debug, Clone, Default)]
pub struct MinimaxM2ToolParser;

impl MinimaxM2ToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a single `<minimax:tool_call>` block.
    fn parse_block(block: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        for invoke_cap in INVOKE_REGEX.captures_iter(block) {
            let name = invoke_cap
                .get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or("");
            let invoke_body = invoke_cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if name.is_empty() {
                continue;
            }

            // Extract parameters
            let mut params = serde_json::Map::new();
            for param_cap in PARAMETER_REGEX.captures_iter(invoke_body) {
                let key = param_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                let value = param_cap
                    .get(2)
                    .map(|m| m.as_str().trim())
                    .unwrap_or("");

                if key.is_empty() {
                    continue;
                }

                // Type inference: try JSON parse, fall back to string
                let json_value = if let Ok(v) = serde_json::from_str::<serde_json::Value>(value) {
                    v
                } else {
                    serde_json::Value::String(value.to_string())
                };
                params.insert(key.to_string(), json_value);
            }

            let arguments = if params.is_empty() {
                "{}".to_string()
            } else {
                serde_json::to_string(&serde_json::Value::Object(params))
                    .unwrap_or_else(|_| "{}".to_string())
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

        calls
    }
}

impl ToolCallParser for MinimaxM2ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut all_calls = Vec::new();

        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let block = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            all_calls.extend(Self::parse_block(block));
        }

        Ok(all_calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(pos) = output.find(START_TAG) {
            let content = output[..pos].trim();
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
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">NYC</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_params() {
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="search">
<parameter name="query">rust programming</parameter>
<parameter name="limit">10</parameter>
<parameter name="active">true</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
        assert_eq!(args["limit"], 10);
        assert_eq!(args["active"], true);
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">NYC</parameter>
</invoke>
</minimax:tool_call>
<minimax:tool_call>
<invoke name="get_time">
<parameter name="tz">EST</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_multiple_invocations_in_one_block() {
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="func1">
<parameter name="a">1</parameter>
</invoke>
<invoke name="func2">
<parameter name="b">2</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "func1");
        assert_eq!(calls[1].function.name, "func2");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = MinimaxM2ToolParser::new();
        let calls = parser.parse("Normal response.").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_no_parameters() {
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="get_time">
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_null_value() {
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="test">
<parameter name="extra">null</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args["extra"].is_null());
    }

    #[test]
    fn parse_json_object_value() {
        let parser = MinimaxM2ToolParser::new();
        let output = r#"<minimax:tool_call>
<invoke name="test">
<parameter name="config">{"nested": true}</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["config"]["nested"], true);
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = MinimaxM2ToolParser::new();
        let output = "Let me help.\n<minimax:tool_call>\n<invoke name=\"test\">\n</invoke>\n</minimax:tool_call>";
        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help.");
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = MinimaxM2ToolParser::new();
        let content = parser.extract_content("Normal text.").unwrap();
        assert_eq!(content, "Normal text.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = MinimaxM2ToolParser::new();
        let content = parser.extract_content(
            "<minimax:tool_call>\n<invoke name=\"test\">\n</invoke>\n</minimax:tool_call>",
        );
        assert!(content.is_none());
    }

    #[test]
    fn tool_call_id_format() {
        let parser = MinimaxM2ToolParser::new();
        let output =
            "<minimax:tool_call>\n<invoke name=\"test\">\n</invoke>\n</minimax:tool_call>";
        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
