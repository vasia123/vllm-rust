//! Olmo3 Pythonic tool call parser.
//!
//! Parses tool calls in newline-separated pythonic function calls within XML tags:
//! ```text
//! <function_calls>
//! get_weather(city='NYC')
//! get_time(tz='EST')
//! </function_calls>
//! ```
//!
//! Differences from standard [`PythonicToolParser`]:
//! - Wrapped in `<function_calls>...</function_calls>` tags
//! - Newline-separated (not comma-separated in brackets)
//! - Handles `null`, `true`, `false` (JSON literals) in addition to Python `None`/`True`/`False`

use super::pythonic::{parse_single_call, split_top_level_calls};
use super::{ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const START_TAG: &str = "<function_calls>";

/// Regex to extract content between `<function_calls>` tags.
static FUNC_CALLS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<function_calls>(.*?)</function_calls>")
        .expect("FUNC_CALLS_REGEX pattern is invalid")
});

/// Parser for Olmo3-style pythonic tool calls.
#[derive(Debug, Clone, Default)]
pub struct Olmo3PythonicToolParser;

impl Olmo3PythonicToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for Olmo3PythonicToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut all_calls = Vec::new();

        for cap in FUNC_CALLS_REGEX.captures_iter(output) {
            let inner = cap.get(1).map(|m| m.as_str()).unwrap_or("");

            // Preprocess: replace JSON literals with Python literals for our parser
            let preprocessed = inner
                .replace("null", "None")
                .replace("true", "True")
                .replace("false", "False");

            // Split by newlines and parse each as a pythonic function call
            for line in preprocessed.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Try to parse as a single call
                match parse_single_call(line) {
                    Ok(call) => all_calls.push(call),
                    Err(_) => {
                        // Try as comma-separated list in brackets
                        if line.starts_with('[') && line.ends_with(']') {
                            let inner_list = &line[1..line.len() - 1];
                            for call_str in split_top_level_calls(inner_list) {
                                let call_str = call_str.trim();
                                if !call_str.is_empty() {
                                    if let Ok(call) = parse_single_call(call_str) {
                                        all_calls.push(call);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(all_calls)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        if let Some(pos) = output.find(START_TAG) {
            let content = output[..pos].trim();
            if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            }
        } else {
            let trimmed = output.trim();
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
    fn parse_single_call() {
        let parser = Olmo3PythonicToolParser::new();
        let output = "<function_calls>\nget_weather(city='NYC')\n</function_calls>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn parse_multiple_calls_newline_separated() {
        let parser = Olmo3PythonicToolParser::new();
        let output =
            "<function_calls>\nget_weather(city='NYC')\nget_time(tz='EST')\n</function_calls>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_json_literals() {
        let parser = Olmo3PythonicToolParser::new();
        let output =
            "<function_calls>\nconfig(verbose=true, debug=false, extra=null)\n</function_calls>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["verbose"], true);
        assert_eq!(args["debug"], false);
        assert!(args["extra"].is_null());
    }

    #[test]
    fn parse_no_tool_calls() {
        let parser = Olmo3PythonicToolParser::new();
        let calls = parser.parse("Just a response.").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_empty_function_calls_block() {
        let parser = Olmo3PythonicToolParser::new();
        let calls = parser.parse("<function_calls>\n</function_calls>").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_content_with_prefix() {
        let parser = Olmo3PythonicToolParser::new();
        let output = "Let me help.\n<function_calls>\ntest()\n</function_calls>";
        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Let me help.");
    }

    #[test]
    fn extract_content_no_tool_calls() {
        let parser = Olmo3PythonicToolParser::new();
        let content = parser.extract_content("Normal text.").unwrap();
        assert_eq!(content, "Normal text.");
    }

    #[test]
    fn extract_content_only_tool_calls() {
        let parser = Olmo3PythonicToolParser::new();
        let content = parser.extract_content("<function_calls>\ntest()\n</function_calls>");
        assert!(content.is_none());
    }

    #[test]
    fn parse_with_nested_args() {
        let parser = Olmo3PythonicToolParser::new();
        let output =
            "<function_calls>\nsearch(query='rust', tags=['systems', 'lang'])\n</function_calls>";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["tags"][0], "systems");
    }

    #[test]
    fn tool_call_id_format() {
        let parser = Olmo3PythonicToolParser::new();
        let output = "<function_calls>\ntest()\n</function_calls>";
        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }
}
