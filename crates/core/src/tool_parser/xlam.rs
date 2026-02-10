//! xLAM (Salesforce) tool call parser.
//!
//! Flexible parser that handles multiple tool call formats:
//! - JSON array: `[{"name": "func", "arguments": {...}}]`
//! - JSON code blocks: `` ```json [{"name": "func", "arguments": {...}}] ``` ``
//! - `[TOOL_CALLS]` prefix
//! - `<tool_call>...</tool_call>` tags
//! - After `</think>` thinking tags
//!
//! Designed for Salesforce xLAM models which can output tool calls in various formats.

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

/// Patterns to find JSON tool call arrays.
static JSON_CODE_BLOCK_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```(?:json)?\s*(.*?)```").expect("JSON_CODE_BLOCK_REGEX pattern is invalid")
});

static TOOL_CALLS_PREFIX_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\[TOOL_CALLS\]\s*(.*)").expect("TOOL_CALLS_PREFIX_REGEX pattern is invalid")
});

static TOOL_CALL_TAG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>")
        .expect("TOOL_CALL_TAG_REGEX pattern is invalid")
});

static THINKING_TAG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)</think>\s*(.*)").expect("THINKING_TAG_REGEX pattern is invalid")
});

/// Parser for xLAM-style tool calls.
///
/// Attempts multiple extraction strategies in order:
/// 1. Content after `</think>` tag (if present)
/// 2. JSON code blocks (`` ```json ... ``` ``)
/// 3. `[TOOL_CALLS]` prefix
/// 4. `<tool_call>` tags
/// 5. Bare JSON array starting with `[`
#[derive(Debug, Clone, Default)]
pub struct XLamToolParser;

impl XLamToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Try to extract JSON tool calls content from the model output.
    /// Returns (content_before, json_str) or None.
    fn extract_json(&self, output: &str) -> Option<(Option<String>, String)> {
        // Check for thinking tag — extract content after </think>
        if let Some(cap) = THINKING_TAG_REGEX.captures(output) {
            let after_think = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            // Try to parse the after-think content as JSON
            if serde_json::from_str::<serde_json::Value>(after_think).is_ok() {
                let content = output[..output.find("</think>").unwrap_or(0) + "</think>".len()]
                    .trim()
                    .to_string();
                return Some((
                    if content.is_empty() {
                        None
                    } else {
                        Some(content)
                    },
                    after_think.to_string(),
                ));
            }
            // Look for JSON code blocks in the after-think content
            if let Some(json_str) = self.find_json_in_code_blocks(after_think) {
                let content = output[..output.find("</think>").unwrap_or(0) + "</think>".len()]
                    .trim()
                    .to_string();
                return Some((
                    if content.is_empty() {
                        None
                    } else {
                        Some(content)
                    },
                    json_str,
                ));
            }
        }

        // Check for JSON code blocks
        if let Some(json_str) = self.find_json_in_code_blocks(output) {
            return Some((None, json_str));
        }

        // Check for [TOOL_CALLS] prefix
        if let Some(cap) = TOOL_CALLS_PREFIX_REGEX.captures(output) {
            let after_prefix = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            if !after_prefix.is_empty() {
                return Some((None, after_prefix.to_string()));
            }
        }

        // Check for <tool_call> tags
        let mut tag_jsons = Vec::new();
        for cap in TOOL_CALL_TAG_REGEX.captures_iter(output) {
            if let Some(content) = cap.get(1) {
                tag_jsons.push(content.as_str().trim().to_string());
            }
        }
        if !tag_jsons.is_empty() {
            let combined = format!("[{}]", tag_jsons.join(","));
            return Some((None, combined));
        }

        // Bare JSON array
        let trimmed = output.trim();
        if trimmed.starts_with('[') && serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return Some((None, trimmed.to_string()));
        }

        None
    }

    fn find_json_in_code_blocks(&self, text: &str) -> Option<String> {
        for cap in JSON_CODE_BLOCK_REGEX.captures_iter(text) {
            let json_str = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                return Some(json_str.to_string());
            }
        }
        None
    }
}

/// Internal JSON representation for an xLAM tool call.
#[derive(Debug, serde::Deserialize)]
struct XLamToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for XLamToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let (_, json_str) = match self.extract_json(output) {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        // Try parsing as a JSON array first
        if let Ok(arr) = serde_json::from_str::<Vec<XLamToolCallJson>>(&json_str) {
            let mut calls = Vec::new();
            for tc in arr {
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
            return Ok(calls);
        }

        // Fallback: find individual JSON objects
        let json_objects = find_json_objects(&json_str);
        let mut calls = Vec::new();
        for obj_str in json_objects {
            match serde_json::from_str::<XLamToolCallJson>(obj_str) {
                Ok(tc) => {
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
                Err(e) => {
                    tracing::warn!("Failed to parse xLAM tool call JSON: {obj_str}: {e}");
                }
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        match self.extract_json(output) {
            Some((content, _)) => content,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_array() {
        let parser = XLamToolParser::new();
        let output = r#"[{"name": "get_weather", "arguments": {"city": "NYC"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_json_code_block() {
        let parser = XLamToolParser::new();
        let output =
            "```json\n[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}]\n```";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_after_thinking() {
        let parser = XLamToolParser::new();
        let output = "<think>I need to check the weather.</think>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let content = parser.extract_content(output);
        assert!(content.is_some());
        assert!(content.unwrap().contains("think"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = XLamToolParser::new();
        let output = r#"[{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "get_time", "arguments": {"tz": "EST"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = XLamToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_call_tags() {
        let parser = XLamToolParser::new();
        let output =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = XLamToolParser::new();
        let output = r#"[{"name": "get_time", "arguments": {}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = XLamToolParser::new();
        let output = r#"[{"name": "test", "arguments": {}}]"#;

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn extract_content_plain_text() {
        let parser = XLamToolParser::new();
        let output = "Normal text response.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Normal text response.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = XLamToolParser::new();
        let output = r#"[{"name": "test", "arguments": {}}]"#;

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }
}
