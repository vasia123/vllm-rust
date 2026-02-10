//! Granite-style tool call parsers.
//!
//! Two parsers for IBM Granite models:
//!
//! ## GraniteToolParser
//! Parses tool calls in Granite 3.0/3.1 format — JSON array prefixed by a special token:
//! ```text
//! <|tool_call|>[{"name": "func", "arguments": {...}}]
//! ```
//! or:
//! ```text
//! <tool_call>[{"name": "func", "arguments": {...}}]
//! ```
//!
//! ## Granite20bFCToolParser
//! Parses tool calls in Granite 20B Function Calling format — repeated tagged JSON objects:
//! ```text
//! <function_call>{"name": "func", "arguments": {...}}
//! <function_call>{"name": "func2", "arguments": {...}}
//! ```

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

// ─── Granite 3.0/3.1 ────────────────────────────────────────────────────────

/// Token prefixes for Granite 3.0 (special token) and 3.1 (text token).
const GRANITE_PREFIX_SPECIAL: &str = "<|tool_call|>";
const GRANITE_PREFIX_TEXT: &str = "<tool_call>";

/// Parser for Granite 3.0/3.1 tool calls.
///
/// Accepts either `<|tool_call|>` (3.0 special token) or `<tool_call>` (3.1 text).
/// The content after the prefix is a JSON array of `{name, arguments}` objects.
#[derive(Debug, Clone, Default)]
pub struct GraniteToolParser;

impl GraniteToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Find the prefix token and return content after it.
    fn strip_prefix<'a>(&self, output: &'a str) -> Option<(usize, &'a str)> {
        if let Some(pos) = output.find(GRANITE_PREFIX_SPECIAL) {
            Some((pos, &output[pos + GRANITE_PREFIX_SPECIAL.len()..]))
        } else if let Some(pos) = output.find(GRANITE_PREFIX_TEXT) {
            Some((pos, &output[pos + GRANITE_PREFIX_TEXT.len()..]))
        } else {
            None
        }
    }
}

/// Internal JSON representation for a Granite tool call.
#[derive(Debug, serde::Deserialize)]
struct GraniteToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for GraniteToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let (_, after_prefix) = match self.strip_prefix(output) {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        let json_str = after_prefix.trim();
        if !json_str.starts_with('[') {
            return Ok(Vec::new());
        }

        match serde_json::from_str::<Vec<GraniteToolCallJson>>(json_str) {
            Ok(tool_call_arr) => {
                let mut calls = Vec::new();
                for tc in tool_call_arr {
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
                Ok(calls)
            }
            Err(e) => {
                tracing::warn!("Failed to parse Granite tool calls JSON: {json_str}: {e}");
                Ok(Vec::new())
            }
        }
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        match self.strip_prefix(output) {
            Some((pos, _)) => {
                let content = output[..pos].trim();
                if content.is_empty() {
                    None
                } else {
                    Some(content.to_string())
                }
            }
            None => {
                let trimmed = output.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
        }
    }
}

// ─── Granite 20B Function Calling ────────────────────────────────────────────

const FUNCTION_CALL_TAG: &str = "<function_call>";

/// Regex to split on `<function_call>` tags (with optional whitespace).
static FUNCTION_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<function_call>\s*").expect("FUNCTION_CALL_REGEX pattern is invalid")
});

/// Parser for Granite 20B Function Calling format.
///
/// Each tool call is a separate JSON object preceded by `<function_call>`:
/// ```text
/// <function_call>{"name": "get_weather", "arguments": {"city": "NYC"}}
/// <function_call>{"name": "get_time", "arguments": {"tz": "EST"}}
/// ```
#[derive(Debug, Clone, Default)]
pub struct Granite20bFCToolParser;

impl Granite20bFCToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for Granite20bFCToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        if !output.contains(FUNCTION_CALL_TAG) {
            return Ok(Vec::new());
        }

        let mut calls = Vec::new();

        // Split on <function_call> tags and parse each JSON object
        let parts: Vec<&str> = FUNCTION_CALL_REGEX.split(output).collect();
        // Skip the first part (content before first tag)
        for part in parts.iter().skip(1) {
            let json_objects = find_json_objects(part);
            for json_str in json_objects {
                match serde_json::from_str::<GraniteToolCallJson>(json_str) {
                    Ok(parsed) => {
                        let arguments = match &parsed.arguments {
                            serde_json::Value::Null => "{}".to_string(),
                            serde_json::Value::Object(_) => {
                                serde_json::to_string(&parsed.arguments)?
                            }
                            other => other.to_string(),
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
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse Granite 20B tool call JSON: {json_str}: {e}"
                        );
                    }
                }
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if !output.contains(FUNCTION_CALL_TAG) {
            let trimmed = output.trim();
            return if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }

        // Content is everything before the first <function_call> tag
        let content = output.split(FUNCTION_CALL_TAG).next().unwrap_or("").trim();

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

    // ─── GraniteToolParser tests ─────────────────────────────────────────

    #[test]
    fn granite_parse_single_tool_call_special_token() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name": "get_weather", "arguments": {"city": "NYC"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn granite_parse_single_tool_call_text_token() {
        let parser = GraniteToolParser::new();
        let output = r#"<tool_call>[{"name": "get_weather", "arguments": {"city": "NYC"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn granite_parse_multiple_tool_calls() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "get_time", "arguments": {"tz": "EST"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn granite_parse_with_preceding_content() {
        let parser = GraniteToolParser::new();
        let output = r#"Let me help you.
<|tool_call|>[{"name": "get_weather", "arguments": {"city": "NYC"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help you.");
    }

    #[test]
    fn granite_parse_no_tool_calls() {
        let parser = GraniteToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn granite_extract_content_only_tool_call() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name": "test", "arguments": {}}]"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn granite_parse_empty_arguments() {
        let parser = GraniteToolParser::new();
        let output = r#"<tool_call>[{"name": "get_time"}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    // ─── Granite20bFCToolParser tests ────────────────────────────────────

    #[test]
    fn granite20b_parse_single_tool_call() {
        let parser = Granite20bFCToolParser::new();
        let output = r#"<function_call>{"name": "get_weather", "arguments": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn granite20b_parse_multiple_tool_calls() {
        let parser = Granite20bFCToolParser::new();
        let output = r#"<function_call>{"name": "get_weather", "arguments": {"city": "NYC"}}
<function_call>{"name": "get_time", "arguments": {"tz": "EST"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn granite20b_parse_with_preceding_content() {
        let parser = Granite20bFCToolParser::new();
        let output = r#"I can help with both.
<function_call>{"name": "get_weather", "arguments": {"city": "NYC"}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "I can help with both.");
    }

    #[test]
    fn granite20b_parse_no_tool_calls() {
        let parser = Granite20bFCToolParser::new();
        let output = "Normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn granite20b_extract_content_only_tool_calls() {
        let parser = Granite20bFCToolParser::new();
        let output = r#"<function_call>{"name": "test", "arguments": {}}"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn granite20b_parse_complex_arguments() {
        let parser = Granite20bFCToolParser::new();
        let output = r#"<function_call>{"name": "search", "arguments": {"query": "test", "limit": 10, "filters": {"type": "article"}}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "test");
        assert_eq!(args["filters"]["type"], "article");
    }

    #[test]
    fn granite20b_tool_call_id_format() {
        let parser = Granite20bFCToolParser::new();
        let output = r#"<function_call>{"name": "test", "arguments": {}}"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
