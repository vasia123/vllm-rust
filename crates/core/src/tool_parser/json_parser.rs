//! Raw JSON tool call parser.
//!
//! Parses tool calls in raw JSON format. This parser handles output that is
//! pure JSON, typically a single tool call object or an array of tool calls.
//!
//! Expected formats:
//! ```json
//! {"name": "function_name", "arguments": {...}}
//! ```
//! or
//! ```json
//! [{"name": "fn1", "arguments": {...}}, {"name": "fn2", "arguments": {...}}]
//! ```

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};

/// Parser for raw JSON tool calls.
///
/// This parser expects the entire output to be valid JSON representing
/// either a single tool call or an array of tool calls.
#[derive(Debug, Clone, Default)]
pub struct JsonToolParser;

/// Internal representation of parsed JSON tool call.
#[derive(Debug, serde::Deserialize)]
struct JsonToolCallFormat {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for JsonToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let trimmed = output.trim();

        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        // Try to parse as JSON
        let value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => {
                // Not valid JSON, return empty
                return Ok(Vec::new());
            }
        };

        let mut calls = Vec::new();

        match value {
            serde_json::Value::Array(arr) => {
                // Array of tool calls
                for item in arr {
                    if let Ok(tool_call) = serde_json::from_value::<JsonToolCallFormat>(item) {
                        calls.push(convert_to_tool_call(tool_call)?);
                    }
                }
            }
            serde_json::Value::Object(_) => {
                // Single tool call
                if let Ok(tool_call) = serde_json::from_value::<JsonToolCallFormat>(value) {
                    calls.push(convert_to_tool_call(tool_call)?);
                }
            }
            _ => {
                // Other JSON types are not tool calls
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        // For raw JSON output, there's no separate content
        // If it's valid JSON tool calls, content is None
        // If it's not valid JSON, the whole output is content
        let trimmed = output.trim();

        if trimmed.is_empty() {
            return None;
        }

        // Check if it's valid JSON that could be tool calls
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            match &value {
                serde_json::Value::Array(arr) => {
                    // Check if any element is a tool call
                    let has_tool_calls = arr.iter().any(|item| item.get("name").is_some());
                    if has_tool_calls {
                        return None;
                    }
                }
                serde_json::Value::Object(obj) => {
                    if obj.contains_key("name") {
                        return None;
                    }
                }
                _ => {}
            }
        }

        // Not tool calls, return as content
        Some(trimmed.to_string())
    }

    fn has_tool_calls(&self, output: &str) -> bool {
        self.parse(output).map(|c| !c.is_empty()).unwrap_or(false)
    }
}

fn convert_to_tool_call(parsed: JsonToolCallFormat) -> anyhow::Result<ToolCall> {
    let arguments = match &parsed.arguments {
        serde_json::Value::Null => "{}".to_string(),
        serde_json::Value::Object(_) => serde_json::to_string(&parsed.arguments)?,
        other => other.to_string(),
    };

    Ok(ToolCall {
        id: generate_tool_call_id(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: parsed.name,
            arguments,
        },
    })
}

impl JsonToolParser {
    /// Create a new JSON tool parser.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_object() {
        let parser = JsonToolParser::new();
        let output = r#"{"name": "get_weather", "arguments": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
    }

    #[test]
    fn test_parse_array() {
        let parser = JsonToolParser::new();
        let output = r#"[
            {"name": "get_weather", "arguments": {"city": "NYC"}},
            {"name": "get_time", "arguments": {"timezone": "EST"}}
        ]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn test_parse_invalid_json() {
        let parser = JsonToolParser::new();
        let output = "this is not json";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_non_tool_json() {
        let parser = JsonToolParser::new();
        let output = r#"{"message": "hello", "count": 42}"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_extract_content_tool_call() {
        let parser = JsonToolParser::new();
        let output = r#"{"name": "test", "arguments": {}}"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn test_extract_content_plain_text() {
        let parser = JsonToolParser::new();
        let output = "Just a regular response";

        let content = parser.extract_content(output);
        assert_eq!(content, Some("Just a regular response".to_string()));
    }

    #[test]
    fn test_extract_content_non_tool_json() {
        let parser = JsonToolParser::new();
        let output = r#"{"message": "hello"}"#;

        let content = parser.extract_content(output);
        assert_eq!(content, Some(r#"{"message": "hello"}"#.to_string()));
    }

    #[test]
    fn test_parse_empty_arguments() {
        let parser = JsonToolParser::new();
        let output = r#"{"name": "get_time"}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_has_tool_calls() {
        let parser = JsonToolParser::new();

        assert!(parser.has_tool_calls(r#"{"name": "test", "arguments": {}}"#));
        assert!(parser.has_tool_calls(r#"[{"name": "test", "arguments": {}}]"#));
        assert!(!parser.has_tool_calls("not json"));
        assert!(!parser.has_tool_calls(r#"{"message": "hello"}"#));
    }

    #[test]
    fn test_parse_empty_input() {
        let parser = JsonToolParser::new();

        assert!(parser.parse("").unwrap().is_empty());
        assert!(parser.parse("   ").unwrap().is_empty());
    }
}
