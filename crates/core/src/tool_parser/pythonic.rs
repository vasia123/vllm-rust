//! Pythonic-style tool call parser.
//!
//! Parses tool calls in Python function call syntax:
//! ```text
//! [get_weather(city='NYC', metric='celsius')]
//! ```
//!
//! This format is used by models trained with ToolACE, Llama 3.2, and
//! other models that emit tool calls as Python expressions.
//!
//! Supports:
//! - Single and multiple function calls within a list
//! - Python keyword arguments (`key=value`)
//! - String literals (single-quoted), numbers, booleans (`True`/`False`), `None`
//! - Nested dicts `{...}` and lists `[...]`

use super::{generate_tool_call_id, FunctionCall, ToolCall, ToolCallParser};
use regex::Regex;
use std::sync::LazyLock;

/// Regex that matches the overall pythonic tool call pattern.
/// Validates `[func(args), func2(args)]` structure before detailed parsing.
static TOOL_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?s)^\[([a-zA-Z]\w*\(([a-zA-Z]\w*=.*?,\s*)*([a-zA-Z]\w*=.*?\s*)?\),\s*)*([a-zA-Z]\w*\(([a-zA-Z]\w*=.*?,\s*)*([a-zA-Z]\w*=.*?\s*)?\)\s*)+\]$",
    )
    .expect("TOOL_CALL_REGEX pattern is invalid")
});

/// Parser for pythonic-style tool calls.
///
/// ```text
/// [get_weather(city='San Francisco', metric='celsius')]
/// [get_weather(city='NYC'), get_time(tz='EST')]
/// ```
#[derive(Debug, Clone, Default)]
pub struct PythonicToolParser;

impl PythonicToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for PythonicToolParser {
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>> {
        let trimmed = output.trim();

        if !TOOL_CALL_REGEX.is_match(trimmed) {
            return Ok(Vec::new());
        }

        parse_pythonic_tool_calls(trimmed)
    }

    fn extract_content(&self, output: &str) -> Option<String> {
        let trimmed = output.trim();
        if TOOL_CALL_REGEX.is_match(trimmed) {
            // Entire output is tool calls
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}

/// Parse the pythonic tool call list expression.
fn parse_pythonic_tool_calls(input: &str) -> anyhow::Result<Vec<ToolCall>> {
    let inner = input
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| anyhow::anyhow!("tool call must be wrapped in []"))?
        .trim();

    if inner.is_empty() {
        return Ok(Vec::new());
    }

    let call_strs = split_top_level_calls(inner);
    let mut calls = Vec::new();

    for call_str in call_strs {
        let call_str = call_str.trim();
        if call_str.is_empty() {
            continue;
        }
        match parse_single_call(call_str) {
            Ok(tc) => calls.push(tc),
            Err(e) => {
                tracing::warn!("Failed to parse pythonic tool call '{call_str}': {e}");
                return Err(e);
            }
        }
    }

    Ok(calls)
}

/// Split top-level function calls separated by commas.
/// Respects parentheses, brackets, braces, and strings.
fn split_top_level_calls(s: &str) -> Vec<&str> {
    let mut results = Vec::new();
    let mut depth = 0i32;
    let mut in_string: Option<char> = None;
    let mut escape = false;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        if let Some(quote) = in_string {
            if ch == '\\' {
                escape = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => in_string = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                results.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        results.push(&s[start..]);
    }

    results
}

/// Parse a single function call like `get_weather(city='NYC', metric='celsius')`.
fn parse_single_call(s: &str) -> anyhow::Result<ToolCall> {
    let paren_pos = s
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("missing '(' in function call"))?;
    let name = s[..paren_pos].trim();

    if name.is_empty() || !name.chars().next().unwrap().is_alphabetic() {
        anyhow::bail!("invalid function name: {name}");
    }

    // Extract content between outer parens
    let rest = &s[paren_pos + 1..];
    let close_paren = find_matching_close(rest, '(', ')')
        .ok_or_else(|| anyhow::anyhow!("unmatched '(' in function call"))?;
    let args_str = &rest[..close_paren].trim();

    let arguments = if args_str.is_empty() {
        serde_json::Map::new()
    } else {
        parse_kwargs(args_str)?
    };

    let arguments_json = serde_json::to_string(&serde_json::Value::Object(arguments))?;

    Ok(ToolCall {
        id: generate_tool_call_id(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments_json,
        },
    })
}

/// Find the index of the matching closing bracket, respecting nesting and strings.
fn find_matching_close(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1i32;
    let mut in_string: Option<char> = None;
    let mut escape = false;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        if let Some(quote) = in_string {
            if ch == '\\' {
                escape = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => in_string = Some(ch),
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
}

/// Parse keyword arguments: `key1=val1, key2=val2, ...`
fn parse_kwargs(s: &str) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut map = serde_json::Map::new();
    let pairs = split_top_level_kwargs(s);

    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        let eq_pos = pair
            .find('=')
            .ok_or_else(|| anyhow::anyhow!("missing '=' in keyword argument: {pair}"))?;

        let key = pair[..eq_pos].trim();
        let value_str = pair[eq_pos + 1..].trim();

        if key.is_empty() {
            anyhow::bail!("empty keyword argument name");
        }

        let value = parse_python_value(value_str)?;
        map.insert(key.to_string(), value);
    }

    Ok(map)
}

/// Split kwargs at top-level commas (outside strings, parens, brackets, braces).
fn split_top_level_kwargs(s: &str) -> Vec<&str> {
    let mut results = Vec::new();
    let mut depth = 0i32;
    let mut in_string: Option<char> = None;
    let mut escape = false;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        if let Some(quote) = in_string {
            if ch == '\\' {
                escape = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => in_string = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                results.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        results.push(&s[start..]);
    }

    results
}

/// Parse a Python literal value into a serde_json::Value.
fn parse_python_value(s: &str) -> anyhow::Result<serde_json::Value> {
    let s = s.trim();

    // None → null
    if s == "None" {
        return Ok(serde_json::Value::Null);
    }

    // True / False → bool
    if s == "True" {
        return Ok(serde_json::Value::Bool(true));
    }
    if s == "False" {
        return Ok(serde_json::Value::Bool(false));
    }

    // String literal (single or double quoted)
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        let inner = &s[1..s.len() - 1];
        // Unescape: \' → ', \" → ", \\ → \
        let unescaped = inner
            .replace("\\'", "'")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
        return Ok(serde_json::Value::String(unescaped));
    }

    // Integer
    if let Ok(n) = s.parse::<i64>() {
        return Ok(serde_json::Value::Number(n.into()));
    }

    // Float
    if let Ok(f) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(serde_json::Value::Number(n));
        }
    }

    // Dict literal: {key: value, ...}
    if s.starts_with('{') && s.ends_with('}') {
        return parse_python_dict(s);
    }

    // List literal: [val, val, ...]
    if s.starts_with('[') && s.ends_with(']') {
        return parse_python_list(s);
    }

    anyhow::bail!("unsupported Python value: {s}")
}

/// Parse a Python dict literal: `{'key': 'value', 'key2': 42}`
fn parse_python_dict(s: &str) -> anyhow::Result<serde_json::Value> {
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }

    let mut map = serde_json::Map::new();
    let entries = split_top_level_kwargs(inner);

    for entry in entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        // Find the colon separating key and value
        let colon_pos = find_top_level_colon(entry)
            .ok_or_else(|| anyhow::anyhow!("missing ':' in dict entry: {entry}"))?;

        let key_str = entry[..colon_pos].trim();
        let val_str = entry[colon_pos + 1..].trim();

        // Key must be a string literal
        let key = if (key_str.starts_with('\'') && key_str.ends_with('\''))
            || (key_str.starts_with('"') && key_str.ends_with('"'))
        {
            key_str[1..key_str.len() - 1].to_string()
        } else {
            anyhow::bail!("dict key must be a string literal: {key_str}");
        };

        let value = parse_python_value(val_str)?;
        map.insert(key, value);
    }

    Ok(serde_json::Value::Object(map))
}

/// Parse a Python list literal: `[1, 'two', True]`
fn parse_python_list(s: &str) -> anyhow::Result<serde_json::Value> {
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Ok(serde_json::Value::Array(Vec::new()));
    }

    let items = split_top_level_kwargs(inner);
    let mut values = Vec::new();

    for item in items {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        values.push(parse_python_value(item)?);
    }

    Ok(serde_json::Value::Array(values))
}

/// Find the first top-level colon (outside strings and nested structures).
fn find_top_level_colon(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string: Option<char> = None;
    let mut escape = false;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        if let Some(quote) = in_string {
            if ch == '\\' {
                escape = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => in_string = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ':' if depth == 0 => return Some(i),
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_function_call() {
        let parser = PythonicToolParser::new();
        let output = "[get_weather(city='San Francisco', metric='celsius')]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "San Francisco");
        assert_eq!(args["metric"], "celsius");
    }

    #[test]
    fn parse_multiple_function_calls() {
        let parser = PythonicToolParser::new();
        let output = "[get_weather(city='NYC'), get_time(tz='EST')]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn parse_empty_arguments() {
        let parser = PythonicToolParser::new();
        let output = "[get_time()]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn parse_python_literals() {
        let parser = PythonicToolParser::new();
        let output = "[register(name='John', age=37, active=True, role=None)]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["name"], "John");
        assert_eq!(args["age"], 37);
        assert_eq!(args["active"], true);
        assert!(args["role"].is_null());
    }

    #[test]
    fn parse_nested_dict() {
        let parser = PythonicToolParser::new();
        let output = "[register(name='John', address={'city': 'SF', 'state': 'CA'})]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["name"], "John");
        assert_eq!(args["address"]["city"], "SF");
        assert_eq!(args["address"]["state"], "CA");
    }

    #[test]
    fn parse_nested_list() {
        let parser = PythonicToolParser::new();
        let output = "[register(name='John', aliases=['John', 'Johnny'])]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["aliases"][0], "John");
        assert_eq!(args["aliases"][1], "Johnny");
    }

    #[test]
    fn parse_complex_nested() {
        let parser = PythonicToolParser::new();
        let output = "[register_user(name='John Doe', age=37, address={'city': 'San Francisco', 'state': 'CA'}, role=None, passed_test=True, aliases=['John', 'Johnny'])]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "register_user");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["name"], "John Doe");
        assert_eq!(args["age"], 37);
        assert!(args["role"].is_null());
        assert_eq!(args["passed_test"], true);
        assert_eq!(args["address"]["city"], "San Francisco");
        assert_eq!(args["aliases"][0], "John");
    }

    #[test]
    fn parse_not_tool_call() {
        let parser = PythonicToolParser::new();
        let output = "Just a normal response.";

        let calls = parser.parse(output).unwrap();
        assert!(calls.is_empty());
        assert!(!parser.has_tool_calls(output));
    }

    #[test]
    fn extract_content_non_tool() {
        let parser = PythonicToolParser::new();
        let output = "Here is some information.";

        let content = parser.extract_content(output).unwrap();
        assert_eq!(content, "Here is some information.");
    }

    #[test]
    fn extract_content_tool_call() {
        let parser = PythonicToolParser::new();
        let output = "[get_weather(city='NYC')]";

        let content = parser.extract_content(output);
        assert!(content.is_none());
    }

    #[test]
    fn parse_escaped_quotes() {
        let parser = PythonicToolParser::new();
        let output = r"[get_info(place='Martha\'s Vineyard')]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["place"], "Martha's Vineyard");
    }

    #[test]
    fn parse_float_value() {
        let parser = PythonicToolParser::new();
        let output = "[set_temp(value=36.6)]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!((args["value"].as_f64().unwrap() - 36.6).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_false_value() {
        let parser = PythonicToolParser::new();
        let output = "[toggle(enabled=False)]";

        let calls = parser.parse(output).unwrap();
        assert_eq!(calls.len(), 1);

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["enabled"], false);
    }

    #[test]
    fn tool_call_id_and_type() {
        let parser = PythonicToolParser::new();
        let output = "[test()]";

        let calls = parser.parse(output).unwrap();
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_empty_dict() {
        let parser = PythonicToolParser::new();
        let output = "[create(config={})]";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["config"], serde_json::json!({}));
    }

    #[test]
    fn parse_empty_list() {
        let parser = PythonicToolParser::new();
        let output = "[create(items=[])]";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["items"], serde_json::json!([]));
    }

    #[test]
    fn parse_integer_negative() {
        let parser = PythonicToolParser::new();
        let output = "[adjust(offset=-5)]";

        let calls = parser.parse(output).unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["offset"], -5);
    }
}
