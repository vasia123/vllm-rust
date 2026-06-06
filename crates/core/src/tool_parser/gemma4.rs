//! Gemma 4 tool call parser.
//!
//! Parses tool calls in the released Gemma 4 format (vLLM reference:
//! `tool_parsers/gemma4_utils.py` / `gemma4_tool_parser.py`):
//!
//! ```text
//! <|tool_call>call:get_weather{city:<|"|>NYC<|"|>}<tool_call|>
//! ```
//!
//! - Values are wrapped in the Gemma 4 escape token `<|"|>` (a literal
//!   stand-in for `"`), so the argument body becomes valid JSON once the
//!   escape tokens are replaced with quotes.
//! - Some checkpoints terminate the call with `<turn|>` instead of
//!   `<tool_call|>` — both are accepted.
//! - Known output variations (fragmented special tokens, multimodal
//!   prompts) drop the delimiters entirely; a fallback tier matches bare
//!   `call:name{...}` / `<call>name{...}` patterns, mirroring vLLM's
//!   non-strict mode.

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

const START_TAG: &str = "<|tool_call>";
/// Escape token wrapping string values: `key:<|"|>value<|"|>`.
const ESCAPE_TOKEN: &str = "<|\"|>";

/// Tier 1 — standard format. `<turn|>` is accepted as an alternative end
/// tag (some Gemma 4 checkpoints emit it instead of `<tool_call|>`).
static STANDARD_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<\|tool_call>call:(?P<name>\w+)\{(?P<args>.*?)\}(?:<tool_call\|>|<turn\|>)")
        .expect("STANDARD_CALL_REGEX pattern is invalid")
});

/// Tier 2 — fallback for known output variations: `<call>name{...}` or a
/// bare `call:name{...}` at the start of the text / after whitespace.
static FALLBACK_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)(?:<call>|(?:^|\s)call:)(?P<name>\w+)\{(?P<args>.*?)\}")
        .expect("FALLBACK_CALL_REGEX pattern is invalid")
});

/// `key:"value"` pairs after escape-token replacement (optional space).
static QUOTED_ARG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\w+):\s*"([^"]*)""#).expect("QUOTED_ARG_REGEX pattern is invalid")
});

/// Last-resort unquoted `key:value` pairs.
static BARE_ARG_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\w+):\s*([^,}]+)").expect("BARE_ARG_REGEX pattern is invalid"));

/// Parser for Gemma 4 tool calls.
#[derive(Debug, Clone, Default)]
pub struct Gemma4ToolParser;

impl Gemma4ToolParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse the argument body of `call:name{...}` into a JSON object.
    ///
    /// Tiered like the vLLM reference: JSON-after-unescape first (preserves
    /// nested values/arrays/numbers), then quoted `key:"value"` extraction,
    /// then bare `key:value` as a last resort.
    fn parse_args(args_str: &str) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if args_str.trim().is_empty() {
            return serde_json::Value::Object(map);
        }

        let cleaned = args_str.replace(ESCAPE_TOKEN, "\"");

        if let Ok(serde_json::Value::Object(obj)) =
            serde_json::from_str::<serde_json::Value>(&format!("{{{cleaned}}}"))
        {
            return serde_json::Value::Object(obj);
        }

        // The compact Gemma 4 format leaves keys unquoted
        // (`query:"rust", limit:5`); quoting them yields valid JSON and
        // preserves typed values (numbers, bools, nested objects). vLLM's
        // reference drops unquoted values here — we do strictly better.
        static KEY_QUOTE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(^|[{,]\s*)(\w+)\s*:").expect("KEY_QUOTE_REGEX pattern is invalid")
        });
        let keyed = KEY_QUOTE_REGEX.replace_all(&cleaned, "$1\"$2\":");
        if let Ok(serde_json::Value::Object(obj)) =
            serde_json::from_str::<serde_json::Value>(&format!("{{{keyed}}}"))
        {
            return serde_json::Value::Object(obj);
        }

        for cap in QUOTED_ARG_REGEX.captures_iter(&cleaned) {
            map.insert(
                cap[1].to_string(),
                serde_json::Value::String(cap[2].to_string()),
            );
        }
        if !map.is_empty() {
            return serde_json::Value::Object(map);
        }

        for cap in BARE_ARG_REGEX.captures_iter(args_str) {
            let value = cap[2].trim().trim_matches('"').replace(ESCAPE_TOKEN, "");
            map.insert(cap[1].to_string(), serde_json::Value::String(value));
        }
        serde_json::Value::Object(map)
    }

    fn calls_from_regex(regex: &Regex, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let mut calls = Vec::new();
        for cap in regex.captures_iter(output) {
            let name = cap.name("name").map(|m| m.as_str()).unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let args_str = cap.name("args").map(|m| m.as_str()).unwrap_or("");
            calls.push(ToolCall {
                id: generate_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: serde_json::to_string(&Self::parse_args(args_str))?,
                },
            });
        }
        Ok(calls)
    }
}

impl ToolCallParser for Gemma4ToolParser {
    /// Gemma 4 tool-call delimiters are special tokens; parsing needs the
    /// raw (specials-preserved) decode.
    fn prefers_raw_decode(&self) -> bool {
        true
    }

    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let standard = Self::calls_from_regex(&STANDARD_CALL_REGEX, output)?;
        if !standard.is_empty() {
            return Ok(standard);
        }
        Self::calls_from_regex(&FALLBACK_CALL_REGEX, output)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        // Content is whatever precedes the first tool-call marker (standard
        // tag first, then the fallback patterns).
        let cut = output.find(START_TAG).or_else(|| {
            FALLBACK_CALL_REGEX
                .captures(output)
                .and_then(|c| c.get(0))
                .map(|m| m.start())
        });
        let content = match cut {
            Some(idx) => output[..idx].trim(),
            None => output.trim(),
        };
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

    fn args_of(call: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&call.function.arguments).expect("arguments are valid JSON")
    }

    #[test]
    fn parse_standard_call_with_escape_tokens() {
        let parser = Gemma4ToolParser::new();
        let out = r#"<|tool_call>call:get_weather{city:<|"|>New York<|"|>}<tool_call|>"#;
        let calls = parser.parse(out).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(args_of(&calls[0])["city"], "New York");
    }

    #[test]
    fn parse_accepts_turn_end_tag_variant() {
        // Some checkpoints close the call with <turn|> instead of <tool_call|>.
        let parser = Gemma4ToolParser::new();
        let out = r#"<|tool_call>call:lookup{id:<|"|>42<|"|>}<turn|>"#;
        let calls = parser.parse(out).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(args_of(&calls[0])["id"], "42");
    }

    #[test]
    fn parse_multiple_args_and_typed_values() {
        let parser = Gemma4ToolParser::new();
        let out =
            r#"<|tool_call>call:search{query:<|"|>rust<|"|>, limit:5, exact:true}<tool_call|>"#;
        let calls = parser.parse(out).unwrap();
        let args = args_of(&calls[0]);
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 5);
        assert_eq!(args["exact"], true);
    }

    #[test]
    fn parse_multiple_calls() {
        let parser = Gemma4ToolParser::new();
        let out = concat!(
            r#"<|tool_call>call:a{x:<|"|>1<|"|>}<tool_call|>"#,
            r#"<|tool_call>call:b{y:<|"|>2<|"|>}<tool_call|>"#,
        );
        let calls = parser.parse(out).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
    }

    #[test]
    fn parse_empty_args() {
        let parser = Gemma4ToolParser::new();
        let calls = parser.parse("<|tool_call>call:ping{}<tool_call|>").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn fallback_bare_call_without_delimiters() {
        // Known Gemma 4 variation: delimiters dropped, bare `call:` pattern.
        let parser = Gemma4ToolParser::new();
        let calls = parser
            .parse(r#"call:get_weather{city:<|"|>Paris<|"|>}"#)
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(args_of(&calls[0])["city"], "Paris");
    }

    #[test]
    fn fallback_fragmented_call_tag() {
        let parser = Gemma4ToolParser::new();
        let calls = parser.parse(r#"<call>lookup{id:<|"|>7<|"|>}"#).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(args_of(&calls[0])["id"], "7");
    }

    #[test]
    fn fallback_not_used_when_standard_matches() {
        // A standard call plus stray `call:`-looking text → only tier 1 fires.
        let parser = Gemma4ToolParser::new();
        let out = r#"<|tool_call>call:real{a:<|"|>1<|"|>}<tool_call|> and call:fake{b:2}"#;
        let calls = parser.parse(out).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "real");
    }

    #[test]
    fn plain_text_has_no_calls() {
        let parser = Gemma4ToolParser::new();
        assert!(parser
            .parse("The capital of France is Paris.")
            .unwrap()
            .is_empty());
        assert!(!parser.has_tool_calls("Just a callous remark {not a call}"));
    }

    #[test]
    fn extract_content_before_call() {
        let parser = Gemma4ToolParser::new();
        let out = r#"Let me check the weather.<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#;
        assert_eq!(
            parser.extract_content(out).as_deref(),
            Some("Let me check the weather.")
        );
        // Pure tool call → no content.
        let pure = r#"<|tool_call>call:get_weather{}<tool_call|>"#;
        assert_eq!(parser.extract_content(pure), None);
    }

    #[test]
    fn quoted_fallback_when_json_invalid() {
        // Unbalanced body defeats the JSON tier; quoted-pair extraction
        // still recovers the arguments.
        let parser = Gemma4ToolParser::new();
        let out = r#"<|tool_call>call:f{a:<|"|>x<|"|>, broken:}<tool_call|>"#;
        let calls = parser.parse(out).unwrap();
        assert_eq!(args_of(&calls[0])["a"], "x");
    }
}
