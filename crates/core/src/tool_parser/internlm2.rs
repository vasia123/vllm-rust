//! InternLM2-style tool call parser.
//!
//! Parses tool calls in the InternLM2 format:
//! ```text
//! <|action_start|><|plugin|>{"name": "func", "parameters": {...}}<|action_end|>
//! ```
//!
//! This format is used by InternLM2-Chat models from Shanghai AI Laboratory.
//! Supports both `parameters` and `arguments` fields for the function arguments.

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};

const ACTION_START: &str = "<|action_start|><|plugin|>";
const ACTION_END: &str = "<|action_end|>";

/// Parser for InternLM2-style tool calls.
///
/// InternLM2 uses special tokens to delimit a single JSON object:
/// ```text
/// Some text<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "NYC"}}<|action_end|>
/// ```
///
/// Only a single tool call per response is supported (no parallel calls).
#[derive(Debug, Clone, Default)]
pub struct InternLm2ToolParser;

impl InternLm2ToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation — accepts both `parameters` and `arguments`.
#[derive(Debug, serde::Deserialize)]
struct InternLm2ToolCallJson {
    name: String,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
}

impl InternLm2ToolCallJson {
    fn get_arguments(&self) -> serde_json::Value {
        self.parameters
            .clone()
            .or_else(|| self.arguments.clone())
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
    }
}

impl ToolCallParser for InternLm2ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(ACTION_START) {
            return Ok(Vec::new());
        }

        let parts: Vec<&str> = output.splitn(2, ACTION_START).collect();
        let action_part = parts.get(1).unwrap_or(&"");

        // Strip the end token if present
        let action_json = action_part.split(ACTION_END).next().unwrap_or("");
        let action_json = action_json.trim();

        // Find the first complete JSON object
        let json_objects = find_json_objects(action_json);
        let json_str = json_objects.first().copied().unwrap_or(action_json);

        match serde_json::from_str::<InternLm2ToolCallJson>(json_str) {
            Ok(parsed) => {
                let arguments = parsed.get_arguments();
                let arguments_str = match &arguments {
                    serde_json::Value::Null => "{}".to_string(),
                    serde_json::Value::Object(_) => serde_json::to_string(&arguments)?,
                    other => other.to_string(),
                };

                Ok(vec![ToolCall {
                    id: generate_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: parsed.name,
                        arguments: arguments_str,
                    },
                }])
            }
            Err(e) => {
                tracing::warn!("Failed to parse InternLM2 tool call JSON: {json_str}: {e}");
                Ok(Vec::new())
            }
        }
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if !output.contains(ACTION_START) {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }

        let content = output.split(ACTION_START).next().unwrap_or("");
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
    fn parse_single_tool_call_with_parameters() {
        let parser = InternLm2ToolParser::new();
        let output = r#"<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "NYC"}}<|action_end|>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_tool_call_with_arguments_field() {
        let parser = InternLm2ToolParser::new();
        let output = r#"<|action_start|><|plugin|>{"name": "search", "arguments": {"query": "rust"}}<|action_end|>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust");
    }

    #[test]
    fn parse_tool_call_with_preceding_text() {
        let parser = InternLm2ToolParser::new();
        let output = r#"Let me check the weather for you.<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "NYC"}}<|action_end|>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check the weather for you.");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = InternLm2ToolParser::new();
        let output = "Just a normal response without any tool calls.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn extract_content_only_tool_call() {
        let parser = InternLm2ToolParser::new();
        let output =
            r#"<|action_start|><|plugin|>{"name": "test", "parameters": {}}<|action_end|>"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_empty_parameters() {
        let parser = InternLm2ToolParser::new();
        let output = r#"<|action_start|><|plugin|>{"name": "get_time"}<|action_end|>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_complex_nested_parameters() {
        let parser = InternLm2ToolParser::new();
        let output = r#"<|action_start|><|plugin|>{"name": "search", "parameters": {"query": "test", "filters": {"type": "article", "limit": 10}}}<|action_end|>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "test");
        assert_eq!(args["filters"]["type"], "article");
        assert_eq!(args["filters"]["limit"], 10);
    }

    #[test]
    fn parse_without_action_end_token() {
        let parser = InternLm2ToolParser::new();
        // Model might not emit the end token
        let output =
            r#"<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn has_tool_calls_detection() {
        let parser = InternLm2ToolParser::new();
        assert!(parser.has_tool_calls(
            r#"<|action_start|><|plugin|>{"name": "test", "parameters": {}}<|action_end|>"#
        ));
        assert!(!parser.has_tool_calls("no tool calls here"));
    }

    #[test]
    fn tool_call_id_format() {
        let parser = InternLm2ToolParser::new();
        let output =
            r#"<|action_start|><|plugin|>{"name": "test", "parameters": {}}<|action_end|>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
