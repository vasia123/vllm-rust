//! Mistral-style tool call parser.
//!
//! Supports two formats depending on the Mistral tokenizer version:
//!
//! **v11+ format** (newer):
//! ```text
//! [TOOL_CALLS]tool_name{"arg1": "val1"}
//! ```
//!
//! **Pre-v11 format** (older):
//! ```text
//! [TOOL_CALLS][{"name": "tool_name", "arguments": {...}}]
//! ```
//!
//! Multiple tool calls use `[TOOL_CALLS]` as separator (v11+) or
//! a JSON array (pre-v11).
//!
//! This format is used by Mistral 7B Instruct v0.3 and later.

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};

const TOOL_CALLS_TOKEN: &str = "[TOOL_CALLS]";

/// Parser for Mistral-style tool calls.
///
/// Auto-detects between v11+ and pre-v11 formats.
#[derive(Debug, Clone, Default)]
pub struct MistralToolParser;

/// Internal format for pre-v11 Mistral tool calls.
#[derive(Debug, serde::Deserialize)]
struct MistralToolCallJson {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl ToolCallParser for MistralToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let text = output.trim();

        // Must contain [TOOL_CALLS] token
        if !text.contains(TOOL_CALLS_TOKEN) {
            return Ok(Vec::new());
        }

        // Get text after the first [TOOL_CALLS] token
        let after_first = match text.find(TOOL_CALLS_TOKEN) {
            Some(idx) => &text[idx + TOOL_CALLS_TOKEN.len()..],
            None => return Ok(Vec::new()),
        };
        let after_trimmed = after_first.trim_start();

        // Try pre-v11 format first: JSON array after [TOOL_CALLS]
        if after_trimmed.starts_with('[') {
            if let Ok(calls) = self.parse_pre_v11(after_trimmed) {
                if !calls.is_empty() {
                    return Ok(calls);
                }
            }
        }

        // Fall back to v11+ format: tool_name{args} segments
        self.parse_v11(text)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        let text = output.trim();

        if let Some(idx) = text.find(TOOL_CALLS_TOKEN) {
            let content = text[..idx].trim();
            return if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            };
        }

        // No [TOOL_CALLS] token — whole output is content
        Some(text.to_string())
    }
}

impl MistralToolParser {
    /// Create a new Mistral tool parser.
    pub fn new() -> Self {
        Self
    }

    /// Parse pre-v11 format: JSON array of tool calls.
    fn parse_pre_v11(&self, text: &str) -> anyhow::Result<Vec<ToolCall>> {
        let arr: Vec<MistralToolCallJson> = serde_json::from_str(text)?;
        let mut calls = Vec::new();

        for parsed in arr {
            let arguments = match &parsed.arguments {
                serde_json::Value::Null => "{}".to_string(),
                serde_json::Value::Object(_) => serde_json::to_string(&parsed.arguments)?,
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

        Ok(calls)
    }

    /// Parse v11+ format: `tool_name{args}` after `[TOOL_CALLS]` tokens.
    fn parse_v11(&self, text: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut calls = Vec::new();
        let mut parts = text.split(TOOL_CALLS_TOKEN);

        // Skip the first part — content before any [TOOL_CALLS]
        parts.next();

        for segment in parts {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }

            // Tool name is before the first '{'
            let brace_idx = match segment.find('{') {
                Some(idx) => idx,
                None => continue,
            };

            let name = segment[..brace_idx].trim();
            if name.is_empty() {
                continue;
            }

            let json_objects = find_json_objects(&segment[brace_idx..]);
            if let Some(json_str) = json_objects.first() {
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(arguments) => {
                        let args_str = match &arguments {
                            serde_json::Value::Null => "{}".to_string(),
                            _ => serde_json::to_string(&arguments)?,
                        };

                        calls.push(ToolCall {
                            id: generate_tool_call_id(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: name.to_string(),
                                arguments: args_str,
                            },
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse Mistral v11+ tool arguments: {e}");
                    }
                }
            }
        }

        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Pre-v11 format ─────────────────────────────────────────────────

    #[test]
    fn parse_pre_v11_single() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "NYC"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_pre_v11_multiple() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "NYC"}}, {"name": "get_time", "arguments": {"tz": "EST"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_pre_v11_empty_arguments() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS][{"name": "ping"}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    // ─── v11+ format ────────────────────────────────────────────────────

    #[test]
    fn parse_v11_single() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city": "NYC"}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
    }

    #[test]
    fn parse_v11_multiple() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city": "NYC"}[TOOL_CALLS]get_time{"tz": "EST"}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_v11_with_underscore_name() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]search_documents{"query": "rust"}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_documents");
    }

    // ─── Content extraction ─────────────────────────────────────────────

    #[test]
    fn extract_content_with_tool_calls() {
        let parser = MistralToolParser::new();
        let output = r#"I'll check the weather.[TOOL_CALLS]get_weather{"city": "NYC"}"#;

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "I'll check the weather.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city": "NYC"}"#;

        assert!(parser.extract_content(output).is_none());
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = MistralToolParser::new();
        let output = "Just a regular response.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Just a regular response.");
    }

    // ─── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn parse_no_tool_calls() {
        let parser = MistralToolParser::new();
        let output = "Hello, how can I help you?";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn parse_empty_input() {
        let parser = MistralToolParser::new();
        assert!(parser.parse("").unwrap().is_empty());
        assert!(parser.parse("   ").unwrap().is_empty());
    }

    #[test]
    fn has_tool_calls_positive() {
        let parser = MistralToolParser::new();
        assert!(parser.has_tool_calls(r#"[TOOL_CALLS][{"name": "fn", "arguments": {}}]"#));
        assert!(parser.has_tool_calls(r#"[TOOL_CALLS]fn{"x": 1}"#));
    }

    #[test]
    fn has_tool_calls_negative() {
        let parser = MistralToolParser::new();
        assert!(!parser.has_tool_calls("no tool calls"));
    }

    #[test]
    fn parse_v11_complex_arguments() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]analyze{"data": [1, 2, 3], "config": {"verbose": true}}"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["data"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn parse_pre_v11_with_content_before() {
        let parser = MistralToolParser::new();
        let output =
            r#"Let me search for you.[TOOL_CALLS][{"name": "search", "arguments": {"q": "test"}}]"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me search for you.");
    }
}
