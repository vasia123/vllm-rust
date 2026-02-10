//! Longcat (SambaNova) tool call parser.
//!
//! Parses tool calls in the Longcat Flash format — JSON objects wrapped in custom tags:
//! ```text
//! <longcat_tool_call>{"name": "func", "arguments": {...}}</longcat_tool_call>
//! ```
//!
//! This is structurally identical to the Hermes format but with different tag names.

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const TOOL_CALL_START: &str = "<longcat_tool_call>";

/// Regex to capture content between `<longcat_tool_call>` tags.
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<longcat_tool_call>\s*(.*?)\s*</longcat_tool_call>|<longcat_tool_call>\s*(.*)")
        .expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Parser for Longcat Flash tool calls.
///
/// Uses `<longcat_tool_call>` tags instead of `<tool_call>` (Hermes).
/// Content inside is a JSON object with `name` and `arguments`/`parameters`.
#[derive(Debug, Clone, Default)]
pub struct LongcatToolParser;

impl LongcatToolParser {
    pub fn new() -> Self {
        Self
    }
}

/// Internal JSON representation.
#[derive(Debug, serde::Deserialize)]
struct LongcatToolCallJson {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

impl LongcatToolCallJson {
    fn get_arguments(&self) -> serde_json::Value {
        self.arguments
            .clone()
            .or_else(|| self.parameters.clone())
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
    }
}

impl ToolCallParser for LongcatToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(TOOL_CALL_START) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();
        for cap in TOOL_CALL_REGEX.captures_iter(output) {
            let content = cap
                .get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str())
                .unwrap_or("");

            let json_objects = find_json_objects(content);
            for json_str in json_objects {
                match serde_json::from_str::<LongcatToolCallJson>(json_str) {
                    Ok(parsed) => {
                        let arguments = parsed.get_arguments();
                        let arguments_str = match &arguments {
                            serde_json::Value::Null => "{}".to_string(),
                            serde_json::Value::Object(_) => serde_json::to_string(&arguments)?,
                            other => other.to_string(),
                        };

                        calls.push(ToolCall {
                            id: generate_tool_call_id(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: parsed.name,
                                arguments: arguments_str,
                            },
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse Longcat tool call JSON: {json_str}: {e}");
                    }
                }
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if !output.contains(TOOL_CALL_START) {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }

        let content = output.split(TOOL_CALL_START).next().unwrap_or("").trim();
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
        let parser = LongcatToolParser::new();
        let output = r#"<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</longcat_tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = LongcatToolParser::new();
        let output = r#"<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</longcat_tool_call>
<longcat_tool_call>{"name": "get_time", "arguments": {"tz": "EST"}}</longcat_tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_with_preceding_content() {
        let parser = LongcatToolParser::new();
        let output = r#"Let me help you.
<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</longcat_tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help you.");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = LongcatToolParser::new();
        let output = "Normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = LongcatToolParser::new();
        let output = r#"<longcat_tool_call>{"name": "test", "arguments": {}}</longcat_tool_call>"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_with_parameters_field() {
        let parser = LongcatToolParser::new();
        let output = r#"<longcat_tool_call>{"name": "search", "parameters": {"query": "test"}}</longcat_tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "test");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = LongcatToolParser::new();
        let output = r#"<longcat_tool_call>{"name": "test", "arguments": {}}</longcat_tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
