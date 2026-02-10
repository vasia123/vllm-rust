//! GLM-4 tool call parser.
//!
//! Parses tool calls in the GLM-4 XML-like format:
//! ```text
//! <tool_call>function_name
//! <arg_key>param1</arg_key>
//! <arg_value>value1</arg_value>
//! <arg_key>param2</arg_key>
//! <arg_value>value2</arg_value>
//! </tool_call>
//! ```
//!
//! This format is used by GLM-4 and GLM-4-MoE models from THUDM/Zhipu AI.
//! Arguments are emitted as `<arg_key>` / `<arg_value>` pairs rather than
//! JSON objects, and values are deserialized when they look like JSON literals.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

/// Parser for GLM-4-style tool calls.
#[derive(Debug, Clone, Default)]
pub struct Glm4ToolParser;

/// Matches complete `<tool_call>...</tool_call>` blocks (dotall for multiline).
static FUNC_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>.*?</tool_call>").expect("FUNC_CALL_REGEX pattern is invalid")
});

/// Extracts function name (group 1) and arguments body (group 2) from a block.
/// The name is everything on the first line after `<tool_call>`, the rest is args.
static FUNC_DETAIL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>")
        .expect("FUNC_DETAIL_REGEX pattern is invalid")
});

/// Extracts individual `<arg_key>...</arg_key> <arg_value>...</arg_value>` pairs.
static FUNC_ARG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>")
        .expect("FUNC_ARG_REGEX pattern is invalid")
});

/// Matches the opening `<tool_call>` tag (for content extraction).
static TOOL_CALL_START_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<tool_call>").expect("TOOL_CALL_START_REGEX pattern is invalid"));

/// Try to deserialize a raw string value into a JSON value.
///
/// Order: JSON literal → raw string. This mirrors vLLM's `_deserialize()`:
/// non-string argument values like `42`, `true`, `[1,2]` become their JSON
/// equivalents, while everything else stays as a JSON string.
fn deserialize_value(raw: &str) -> serde_json::Value {
    // Try JSON first (handles numbers, bools, arrays, objects, null)
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return v;
    }
    // Fall back to string
    serde_json::Value::String(raw.to_string())
}

/// Parse `<arg_key>/<arg_value>` pairs from the arguments body and return
/// a JSON object string like `{"key1":"val1","key2":42}`.
fn parse_arguments(body: &str) -> String {
    let mut map = serde_json::Map::new();

    for cap in FUNC_ARG_REGEX.captures_iter(body) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let raw_value = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");

        if key.is_empty() {
            continue;
        }

        map.insert(key.to_string(), deserialize_value(raw_value));
    }

    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

impl ToolCallParser for Glm4ToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut calls = Vec::new();

        for block in FUNC_CALL_REGEX.find_iter(output) {
            let block_str = block.as_str();

            if let Some(detail) = FUNC_DETAIL_REGEX.captures(block_str) {
                let name = detail
                    .get(1)
                    .map(|m| m.as_str().trim())
                    .unwrap_or("")
                    .to_string();
                let args_body = detail.get(2).map(|m| m.as_str()).unwrap_or("");

                if name.is_empty() {
                    tracing::warn!("GLM-4 tool call with empty function name, skipping");
                    continue;
                }

                let arguments = parse_arguments(args_body);

                calls.push(ToolCall {
                    id: generate_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall { name, arguments },
                });
            } else {
                tracing::warn!(
                    "GLM-4 tool call block did not match detail regex: {}",
                    block_str
                );
            }
        }

        Ok(calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        // Content is everything before the first <tool_call> tag.
        if let Some(m) = TOOL_CALL_START_REGEX.find(output) {
            let before = output[..m.start()].trim();
            if before.is_empty() {
                None
            } else {
                Some(before.to_string())
            }
        } else {
            // No tool calls — entire output is content
            let trimmed = output.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
    }

    fn has_tool_calls(&self, output: &str) -> bool {
        FUNC_CALL_REGEX.is_match(output)
            && self.parse(output).map(|c| !c.is_empty()).unwrap_or(false)
    }
}

impl Glm4ToolParser {
    /// Create a new GLM-4 tool parser.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tool_call() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2025-08-01</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].id.starts_with("call_"));

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Beijing");
        assert_eq!(args["date"], "2025-08-01");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>
<tool_call>get_time
<arg_key>timezone</arg_key>
<arg_value>Asia/Shanghai</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_numeric_value_deserialized() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>search
<arg_key>query</arg_key>
<arg_value>rust programming</arg_value>
<arg_key>limit</arg_key>
<arg_value>42</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
        assert_eq!(args["limit"], 42); // deserialized to number
    }

    #[test]
    fn parse_boolean_and_null_values() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>configure
<arg_key>verbose</arg_key>
<arg_value>true</arg_value>
<arg_key>debug</arg_key>
<arg_value>false</arg_value>
<arg_key>extra</arg_key>
<arg_value>null</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["verbose"], true);
        assert_eq!(args["debug"], false);
        assert!(args["extra"].is_null());
    }

    #[test]
    fn parse_array_and_object_values() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>complex_fn
<arg_key>ids</arg_key>
<arg_value>[1, 2, 3]</arg_value>
<arg_key>config</arg_key>
<arg_value>{\"nested\": true}</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["ids"], serde_json::json!([1, 2, 3]));
        assert_eq!(args["config"], serde_json::json!({"nested": true}));
    }

    #[test]
    fn parse_no_arguments() {
        let parser = Glm4ToolParser::new();
        // Tool call with name but no arg_key/arg_value pairs.
        let output = "\
<tool_call>get_time
</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn extract_content_before_tool_call() {
        let parser = Glm4ToolParser::new();
        let output = "Let me check the weather for you.\n\
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>NYC</arg_value>
</tool_call>";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me check the weather for you.");

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = Glm4ToolParser::new();
        let output = "Just a normal response with no tool calls.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, output);
    }

    #[test]
    fn extract_content_only_tool_call() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>test_fn
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>";

        assert!(parser.extract_content(output).is_none());
    }

    #[test]
    fn has_tool_calls_positive() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>fn1
<arg_key>a</arg_key>
<arg_value>b</arg_value>
</tool_call>";
        assert!(parser.has_tool_calls(output));
    }

    #[test]
    fn has_tool_calls_negative() {
        let parser = Glm4ToolParser::new();
        assert!(!parser.has_tool_calls("no tool calls here"));
    }

    #[test]
    fn empty_tool_call_block_skipped() {
        let parser = Glm4ToolParser::new();
        // Empty block — no function name, no newline before close
        let output = "<tool_call></tool_call>";
        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn whitespace_around_values_stripped() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>  fn_name
<arg_key>  key1  </arg_key>
<arg_value>  value1  </arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls[0].function.name, "fn_name");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["key1"], "value1");
    }

    #[test]
    fn no_tool_calls_returns_empty() {
        let parser = Glm4ToolParser::new();
        let calls = parser.parse("Hello, world!").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn deserialize_value_json_number() {
        assert_eq!(deserialize_value("42"), serde_json::json!(42));
        assert_eq!(deserialize_value("3.14"), serde_json::json!(3.14));
    }

    #[test]
    fn deserialize_value_json_bool() {
        assert_eq!(deserialize_value("true"), serde_json::json!(true));
        assert_eq!(deserialize_value("false"), serde_json::json!(false));
    }

    #[test]
    fn deserialize_value_json_null() {
        assert_eq!(deserialize_value("null"), serde_json::Value::Null);
    }

    #[test]
    fn deserialize_value_plain_string() {
        assert_eq!(
            deserialize_value("hello world"),
            serde_json::json!("hello world")
        );
    }

    #[test]
    fn deserialize_value_json_array() {
        assert_eq!(deserialize_value("[1, 2, 3]"), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn content_with_thinking_tags() {
        let parser = Glm4ToolParser::new();
        let output = "<think>Let me think about this...</think>\n\
I'll help you.\n\
<tool_call>helper
<arg_key>task</arg_key>
<arg_value>assist</arg_value>
</tool_call>";

        let content = parser.extract_content(output).unwrap();
        assert!(content.contains("<think>"));
        assert!(content.contains("I'll help you."));
        assert!(!content.contains("<tool_call>"));
    }

    #[test]
    fn float_value_deserialized() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>set_temp
<arg_key>temperature</arg_key>
<arg_value>0.7</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["temperature"], 0.7);
    }

    #[test]
    fn duplicate_arg_keys_last_wins() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>fn1
<arg_key>x</arg_key>
<arg_value>first</arg_value>
<arg_key>x</arg_key>
<arg_value>second</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        // serde_json::Map insert overwrites on duplicate key
        assert_eq!(args["x"], "second");
    }

    #[test]
    fn empty_arg_key_skipped() {
        let parser = Glm4ToolParser::new();
        let output = "\
<tool_call>fn1
<arg_key></arg_key>
<arg_value>orphan</arg_value>
<arg_key>real</arg_key>
<arg_value>value</arg_value>
</tool_call>";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.get("").is_none());
        assert_eq!(args["real"], "value");
    }
}
