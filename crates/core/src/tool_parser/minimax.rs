//! MiniMax tool call parser.
//!
//! Parses tool calls in JSON format within `<tool_calls>` XML tags:
//! ```text
//! <tool_calls>
//! {"name": "get_weather", "arguments": {"city": "NYC"}}
//! {"name": "get_time", "arguments": {"tz": "EST"}}
//! </tool_calls>
//! ```
//!
//! Also handles:
//! - `<think>...</think>` blocks: tool calls inside thinking tags are stripped
//! - Newline-separated JSON objects
//! - Standard `{"name": ..., "arguments": ...}` format

use super::{find_json_objects, generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const START_TAG: &str = "<tool_calls>";

/// Regex to extract content between `<tool_calls>` tags.
static TOOL_CALLS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_calls>(.*?)</tool_calls>").expect("TOOL_CALLS_REGEX pattern is invalid")
});

/// Regex to match `<think>...</think>` blocks for removal.
static THINK_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<think>.*?</think>").expect("THINK_REGEX pattern is invalid")
});

/// Parser for MiniMax-style tool calls.
///
/// JSON objects within `<tool_calls>` tags, with `<think>` tag stripping.
#[derive(Debug, Clone, Default)]
pub struct MinimaxToolParser;

impl MinimaxToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a JSON object as a tool call.
    fn parse_json_tool_call(json_str: &str) -> Option<ToolCall> {
        let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let name = v.get("name")?.as_str()?.to_string();
        if name.is_empty() {
            return None;
        }

        let arguments = match v.get("arguments") {
            Some(args) => {
                if args.is_string() {
                    args.as_str().unwrap_or("{}").to_string()
                } else {
                    serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                }
            }
            None => match v.get("parameters") {
                Some(params) => {
                    serde_json::to_string(params).unwrap_or_else(|_| "{}".to_string())
                }
                None => "{}".to_string(),
            },
        };

        Some(ToolCall {
            id: generate_tool_call_id(),
            call_type: "function".to_string(),
            function: FunctionCall { name, arguments },
        })
    }
}

impl ToolCallParser for MinimaxToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        // Strip <think> blocks first
        let cleaned = THINK_REGEX.replace_all(output, "");

        let mut all_calls = Vec::new();

        for cap in TOOL_CALLS_REGEX.captures_iter(&cleaned) {
            let inner = cap.get(1).map(|m| m.as_str()).unwrap_or("");

            // Find JSON objects within the tool_calls block
            let json_objects = find_json_objects(inner);
            for json_str in json_objects {
                if let Some(call) = Self::parse_json_tool_call(json_str) {
                    all_calls.push(call);
                }
            }
        }

        Ok(all_calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        // Strip <think> blocks for content extraction too
        let cleaned = THINK_REGEX.replace_all(output, "");

        if let Some(pos) = cleaned.find(START_TAG) {
            let content = cleaned[..pos].trim();
            if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            }
        } else {
            let trimmed = cleaned.trim();
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
        let parser = MinimaxToolParser::new();
        let output =
            r#"<tool_calls>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = MinimaxToolParser::new();
        let output = "<tool_calls>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"EST\"}}\n</tool_calls>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_with_think_tags_stripped() {
        let parser = MinimaxToolParser::new();
        let output = "<think>I should check the weather.\n<tool_calls>{\"name\": \"fake\"}</tool_calls>\n</think>\n<tool_calls>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}</tool_calls>";

        let calls = parser.parse(output).unwrap();
        // Only the call outside <think> should be parsed
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = MinimaxToolParser::new();
        let calls = parser.parse("Normal response.").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_empty_tool_calls_block() {
        let parser = MinimaxToolParser::new();
        let calls = parser.parse("<tool_calls></tool_calls>").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_with_parameters_key() {
        let parser = MinimaxToolParser::new();
        let output =
            r#"<tool_calls>{"name": "search", "parameters": {"query": "rust"}}</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust");
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = MinimaxToolParser::new();
        let output = "Let me help.\n<tool_calls>{\"name\": \"test\"}</tool_calls>";
        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help.");
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = MinimaxToolParser::new();
        let content = parser.extract_content("Normal text.").unwrap();
        assert_eq!(content, "Normal text.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = MinimaxToolParser::new();
        let content =
            parser.extract_content(r#"<tool_calls>{"name": "test"}</tool_calls>"#);
        assert!(content.is_none());
    }

    #[test]
    fn extract_content_strips_think() {
        let parser = MinimaxToolParser::new();
        let output = "<think>Thinking...</think>\n<tool_calls>{\"name\": \"test\"}</tool_calls>";
        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_string_arguments() {
        let parser = MinimaxToolParser::new();
        let output =
            r#"<tool_calls>{"name": "test", "arguments": "{\"key\": \"value\"}"}</tool_calls>"#;

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        // String arguments are passed through as-is
        assert!(calls[0].function.arguments.contains("key"));
    }

    #[test]
    fn tool_call_id_format() {
        let parser = MinimaxToolParser::new();
        let output = r#"<tool_calls>{"name": "test"}</tool_calls>"#;
        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
