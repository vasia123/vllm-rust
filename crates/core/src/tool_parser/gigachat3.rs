//! GigaChat3 tool call parser.
//!
//! Parses tool calls in the GigaChat3 format:
//! ```text
//! function call{"name": "get_weather", "arguments": {"city": "NYC"}}
//! ```
//!
//! The trigger phrase is `function call` followed by a JSON object.

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const FUNCTION_CALL_TRIGGER: &str = "function call";

/// Regex to capture JSON after `function call` trigger.
/// Also handles optional `<|role_sep|>\n` between trigger and JSON.
static FUNCTION_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"function call(?:<\|role_sep\|>\n?)?\s*(\{.*)")
        .expect("FUNCTION_CALL_REGEX pattern is invalid")
});

/// Parser for GigaChat3-style tool calls.
///
/// Detects `function call{...}` patterns and extracts tool calls from the JSON payload.
#[derive(Debug, Clone, Default)]
pub struct GigaChat3ToolParser;

impl GigaChat3ToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation for a GigaChat3 tool call.
#[derive(Debug, serde::Deserialize)]
struct GigaChat3ToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for GigaChat3ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let cap = match FUNCTION_CALL_REGEX.captures(output) {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let json_str = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if json_str.is_empty() {
            return Ok(Vec::new());
        }

        // Try direct parse first
        if let Ok(tc) = serde_json::from_str::<GigaChat3ToolCallJson>(json_str) {
            let arguments = match &tc.arguments {
                serde_json::Value::Null => "{}".to_string(),
                serde_json::Value::Object(_) => serde_json::to_string(&tc.arguments)?,
                other => other.to_string(),
            };
            return Ok(vec![ToolCall {
                id: generate_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tc.name,
                    arguments,
                },
            }]);
        }

        // Fallback: find individual JSON objects
        let objects = find_json_objects(json_str);
        let mut calls = Vec::new();
        for obj_str in objects {
            if let Ok(tc) = serde_json::from_str::<GigaChat3ToolCallJson>(obj_str) {
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
        if let Some(idx) = output.find(FUNCTION_CALL_TRIGGER) {
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
    fn parse_basic_tool_call() {
        let parser = GigaChat3ToolParser::new();
        let output = r#"function call{"name": "get_weather", "arguments": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_with_role_sep() {
        let parser = GigaChat3ToolParser::new();
        let output =
            "function call<|role_sep|>\n{\"name\": \"test\", \"arguments\": {\"x\": 1}}";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "test");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = GigaChat3ToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = GigaChat3ToolParser::new();
        let output = r#"function call{"name": "get_time", "arguments": {}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = GigaChat3ToolParser::new();
        let output =
            "Let me check. function call{\"name\": \"test\", \"arguments\": {}}";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check.");

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn extract_content_only_tool_call() {
        let parser = GigaChat3ToolParser::new();
        let output = r#"function call{"name": "test", "arguments": {}}"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn extract_content_plain_text() {
        let parser = GigaChat3ToolParser::new();
        let output = "Hello, how can I help?";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Hello, how can I help?");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = GigaChat3ToolParser::new();
        let output = r#"function call{"name": "test", "arguments": {}}"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
