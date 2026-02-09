//! Llama 3.x/4-style tool call parser.
//!
//! Parses tool calls in the Llama format, which uses an optional
//! `<|python_tag|>` prefix followed by one or more JSON objects
//! separated by semicolons.
//!
//! Single tool call:
//! ```text
//! <|python_tag|>{"name": "get_weather", "arguments": {"city": "NYC"}}
//! ```
//!
//! Multiple tool calls:
//! ```text
//! <|python_tag|>{"name": "fn1", "arguments": {...}}; {"name": "fn2", "arguments": {...}}
//! ```
//!
//! Supports both `"arguments"` and `"parameters"` keys, normalizing the
//! latter to `"arguments"` for OpenAI API compatibility.
//!
//! This format is used by Meta Llama 3.x and Llama 4 Instruct models.

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};

const PYTHON_TAG: &str = "<|python_tag|>";

/// Parser for Llama 3.x/4-style tool calls.
#[derive(Debug, Clone, Default)]
pub struct LlamaToolParser;

/// Internal JSON format for a Llama tool call.
#[derive(Debug, serde::Deserialize)]
struct LlamaToolCallJson {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

impl ToolCallParser for LlamaToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let text = output.trim();

        // Strip optional <|python_tag|> prefix
        let text = text.strip_prefix(PYTHON_TAG).unwrap_or(text).trim();

        if text.is_empty() {
            return Ok(Vec::new());
        }

        let json_objects = find_json_objects(text);
        if json_objects.is_empty() {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for json_str in json_objects {
            if let Ok(parsed) = serde_json::from_str::<LlamaToolCallJson>(json_str) {
                // Normalize: "parameters" → "arguments"
                let args = parsed.arguments.or(parsed.parameters);
                let arguments = match args {
                    None | Some(serde_json::Value::Null) => "{}".to_string(),
                    Some(ref val @ serde_json::Value::Object(_)) => {
                        serde_json::to_string(val)?
                    }
                    Some(other) => other.to_string(),
                };

                calls.push(ToolCall {
                    id: generate_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: parsed.name,
                        arguments,
                    },
                });
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        let text = output.trim();

        // If python_tag present, content is everything before it
        if let Some(idx) = text.find(PYTHON_TAG) {
            let content = text[..idx].trim();
            return if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            };
        }

        // No python_tag — if we have tool calls, content is before the first '{'
        if self.has_tool_calls(output) {
            if let Some(idx) = text.find('{') {
                let content = text[..idx].trim();
                return if content.is_empty() {
                    None
                } else {
                    Some(content.to_string())
                };
            }
            return None;
        }

        // No tool calls — the whole output is content
        Some(text.to_string())
    }
}

impl LlamaToolParser {
    /// Create a new Llama tool parser.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_with_python_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "get_weather", "arguments": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_single_without_python_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"{"name": "get_weather", "arguments": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_multiple_semicolon_separated() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "get_weather", "arguments": {"city": "NYC"}}; {"name": "get_time", "arguments": {"tz": "EST"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_multiple_newline_separated() {
        let parser = LlamaToolParser::new();
        let output = "<|python_tag|>{\"name\": \"fn1\", \"arguments\": {}}\n{\"name\": \"fn2\", \"arguments\": {}}";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "fn1");
        assert_eq!(calls[1].function.name, "fn2");
    }

    #[test]
    fn parse_parameters_normalized_to_arguments() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "search", "parameters": {"query": "rust"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        assert!(calls[0].function.arguments.contains("rust"));
    }

    #[test]
    fn parse_arguments_takes_priority_over_parameters() {
        let parser = LlamaToolParser::new();
        // If both are present, arguments wins (or() short-circuits)
        let output =
            r#"{"name": "fn", "arguments": {"from": "args"}, "parameters": {"from": "params"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].function.arguments.contains("args"));
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "get_time"}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = LlamaToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn parse_empty_input() {
        let parser = LlamaToolParser::new();
        assert!(parser.parse("").unwrap().is_empty());
        assert!(parser.parse("   ").unwrap().is_empty());
    }

    #[test]
    fn parse_non_tool_json_ignored() {
        let parser = LlamaToolParser::new();
        // JSON without "name" field is not a tool call
        let output = r#"{"message": "hello"}"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_with_python_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"Let me help you.<|python_tag|>{"name": "search", "arguments": {}}"#;

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help you.");
    }

    #[test]
    fn extract_content_only_tool_call() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "search", "arguments": {}}"#;

        assert!(parser.extract_content(output).is_none());
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = LlamaToolParser::new();
        let output = "Regular response text.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Regular response text.");
    }

    #[test]
    fn extract_content_before_json_without_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"Here is the result: {"name": "fn", "arguments": {"x": 1}}"#;

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Here is the result:");
    }

    #[test]
    fn has_tool_calls_positive() {
        let parser = LlamaToolParser::new();
        assert!(parser.has_tool_calls(
            r#"<|python_tag|>{"name": "fn", "arguments": {}}"#
        ));
    }

    #[test]
    fn has_tool_calls_negative() {
        let parser = LlamaToolParser::new();
        assert!(!parser.has_tool_calls("no tool calls here"));
        assert!(!parser.has_tool_calls(r#"{"message": "not a tool call"}"#));
    }

    #[test]
    fn parse_complex_nested_arguments() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "analyze", "arguments": {"data": {"items": [1, 2, 3], "config": {"verbose": true}}}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["data"]["items"], serde_json::json!([1, 2, 3]));
        assert_eq!(args["data"]["config"]["verbose"], true);
    }

    #[test]
    fn parse_whitespace_around_python_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"  <|python_tag|>  {"name": "fn", "arguments": {}}  "#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
    }
}
