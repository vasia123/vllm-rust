//! Phi-4 Mini tool call parser.
//!
//! Parses tool calls in the Microsoft Phi-4 Mini format:
//! ```text
//! functools[{"name": "func", "arguments": {...}}, {"name": "func2", "parameters": {...}}]
//! ```
//!
//! The `functools[...]` wrapper contains a JSON array of tool call objects.
//! Supports both `arguments` and `parameters` fields.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const FUNCTOOLS_PREFIX: &str = "functools[";

/// Regex to extract content inside `functools[...]`.
static FUNCTOOLS_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"functools\[(.*)\]").expect("FUNCTOOLS_REGEX pattern is invalid"));

/// Parser for Phi-4 Mini tool calls.
///
/// Wraps tool calls in `functools[JSON_ARRAY]` format.
#[derive(Debug, Clone, Default)]
pub struct Phi4MiniToolParser;

impl Phi4MiniToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation for a Phi-4 Mini tool call.
#[derive(Debug, serde::Deserialize)]
struct Phi4ToolCallJson {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

impl Phi4ToolCallJson {
    fn get_arguments(&self) -> serde_json::Value {
        self.arguments
            .clone()
            .or_else(|| self.parameters.clone())
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
    }
}

impl ToolCallParser for Phi4MiniToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(FUNCTOOLS_PREFIX) {
            return Ok(Vec::new());
        }

        let cap = match FUNCTOOLS_REGEX.captures(output) {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let json_content = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        // Wrap in array brackets since the regex strips them
        let json_str = format!("[{json_content}]");

        match serde_json::from_str::<Vec<Phi4ToolCallJson>>(&json_str) {
            Ok(tool_calls) => {
                let mut calls = Vec::new();
                for tc in tool_calls {
                    let arguments = tc.get_arguments();
                    let arguments_str = match &arguments {
                        serde_json::Value::Null => "{}".to_string(),
                        serde_json::Value::Object(_) => serde_json::to_string(&arguments)?,
                        other => other.to_string(),
                    };

                    calls.push(ToolCall {
                        id: generate_tool_call_id(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.name,
                            arguments: arguments_str,
                        },
                    });
                }
                Ok(calls)
            }
            Err(e) => {
                tracing::warn!("Failed to parse Phi-4 Mini tool calls JSON: {json_str}: {e}");
                Ok(Vec::new())
            }
        }
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        // Phi-4 Mini doesn't typically emit content alongside tool calls
        if !output.contains(FUNCTOOLS_PREFIX) {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let parser = Phi4MiniToolParser::new();
        let output = r#"functools[{"name": "get_weather", "arguments": {"city": "NYC"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = Phi4MiniToolParser::new();
        let output = r#"functools[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "get_time", "arguments": {"tz": "EST"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_with_parameters_field() {
        let parser = Phi4MiniToolParser::new();
        let output = r#"functools[{"name": "search", "parameters": {"query": "rust"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = Phi4MiniToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_no_tools() {
        let parser = Phi4MiniToolParser::new();
        let output = "Normal text response.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Normal text response.");
    }

    #[test]
    fn extract_content_with_tools() {
        let parser = Phi4MiniToolParser::new();
        let output = r#"functools[{"name": "test", "arguments": {}}]"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = Phi4MiniToolParser::new();
        let output = r#"functools[{"name": "get_time"}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = Phi4MiniToolParser::new();
        let output = r#"functools[{"name": "test", "arguments": {}}]"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
