//! JSON Schema to regex conversion.
//!
//! Converts a subset of JSON Schema to regular expressions,
//! then delegates to `RegexDfaGrammar` for DFA-based token masking.

use std::sync::Arc;

use super::regex_backend::RegexDfaGrammar;
use super::vocabulary::VocabularyIndex;
use super::StructuredOutputGrammar;

/// Default whitespace pattern inserted between structural JSON tokens.
const DEFAULT_WHITESPACE: &str = r"[ \t\n\r]*";

/// Convert a JSON Schema to a `StructuredOutputGrammar` via regex compilation.
///
/// Supports: string, integer, number, boolean, null, enum, object (with
/// properties/required), array (with items), anyOf/oneOf, and $ref resolution.
pub fn json_schema_to_grammar(
    schema: &serde_json::Value,
    vocab_index: Arc<VocabularyIndex>,
    whitespace_pattern: Option<&str>,
) -> anyhow::Result<Box<dyn StructuredOutputGrammar>> {
    let ws = whitespace_pattern.unwrap_or(DEFAULT_WHITESPACE);
    let regex = schema_to_regex(schema, ws, schema)?;
    let grammar = RegexDfaGrammar::new(&regex, vocab_index)?;
    Ok(Box::new(grammar))
}

/// Recursively convert a JSON Schema node to a regex pattern.
fn schema_to_regex(
    node: &serde_json::Value,
    ws: &str,
    root: &serde_json::Value,
) -> anyhow::Result<String> {
    // Handle $ref
    if let Some(ref_path) = node.get("$ref").and_then(|v| v.as_str()) {
        let resolved = resolve_ref(ref_path, root)?;
        return schema_to_regex(resolved, ws, root);
    }

    // Handle anyOf / oneOf
    if let Some(any_of) = node.get("anyOf").and_then(|v| v.as_array()) {
        return alternation(any_of, ws, root);
    }
    if let Some(one_of) = node.get("oneOf").and_then(|v| v.as_array()) {
        return alternation(one_of, ws, root);
    }

    // Handle enum
    if let Some(enum_vals) = node.get("enum").and_then(|v| v.as_array()) {
        return enum_to_regex(enum_vals);
    }

    // Handle const
    if let Some(const_val) = node.get("const") {
        return const_to_regex(const_val);
    }

    // Handle by type
    let type_val = node.get("type").and_then(|v| v.as_str()).unwrap_or("any");

    match type_val {
        "string" => Ok(string_regex()),
        "integer" => Ok(integer_regex()),
        "number" => Ok(number_regex()),
        "boolean" => Ok(boolean_regex()),
        "null" => Ok(null_regex()),
        "object" => object_regex(node, ws, root),
        "array" => array_regex(node, ws, root),
        _ => {
            // "any" or unknown type — accept any JSON value
            Ok(any_json_value_regex(ws))
        }
    }
}

fn string_regex() -> String {
    r#""([^"\\]|\\.)*""#.to_string()
}

fn integer_regex() -> String {
    r"(0|-?[1-9][0-9]*)".to_string()
}

fn number_regex() -> String {
    r"(0|-?[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?".to_string()
}

fn boolean_regex() -> String {
    "(true|false)".to_string()
}

fn null_regex() -> String {
    "null".to_string()
}

fn any_json_value_regex(ws: &str) -> String {
    // Matches any JSON primitive, object, or array (non-recursive approximation)
    let str_re = string_regex();
    let num_re = number_regex();
    let bool_re = boolean_regex();
    let null_re = null_regex();
    let val = format!("({str_re}|{num_re}|{bool_re}|{null_re})");
    let obj = format!(
        r#"(\{{{ws}({ws}{str_re}{ws}:{ws}{val}{ws}(,{ws}{str_re}{ws}:{ws}{val}{ws})*)?{ws}\}})"#
    );
    let arr = format!(r#"(\[{ws}({val}{ws}(,{ws}{val}{ws})*)?{ws}\])"#);
    format!("({str_re}|{num_re}|{bool_re}|{null_re}|{obj}|{arr})")
}

fn object_regex(
    node: &serde_json::Value,
    ws: &str,
    root: &serde_json::Value,
) -> anyhow::Result<String> {
    let properties = node.get("properties").and_then(|v| v.as_object());
    let required: Vec<&str> = node
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    if let Some(props) = properties {
        if props.is_empty() {
            // Empty object
            return Ok(format!(r"\{{{ws}\}}", ws = ws));
        }

        // Build regex for each property in order
        let mut parts = Vec::new();
        let required_set: std::collections::HashSet<&str> = required.into_iter().collect();

        for (key, value_schema) in props {
            let key_escaped = regex_escape_json_string(key);
            let value_regex = schema_to_regex(value_schema, ws, root)?;
            let kv = format!(
                r#"{ws}"{key}"{ws}:{ws}{value}{ws}"#,
                ws = ws,
                key = key_escaped,
                value = value_regex,
            );

            if required_set.contains(key.as_str()) {
                parts.push(kv);
            } else {
                // Optional property
                parts.push(format!("({kv})?"));
            }
        }

        // Join properties with commas
        let inner = if parts.len() == 1 {
            parts.into_iter().next().unwrap()
        } else {
            // For simplicity, require all required properties in order with commas
            let mut result = String::new();
            for (i, part) in parts.iter().enumerate() {
                if i > 0 {
                    result.push_str(&format!(",{ws}"));
                }
                result.push_str(part);
            }
            result
        };

        Ok(format!(r"\{{{ws}{inner}{ws}\}}", ws = ws, inner = inner))
    } else {
        // No properties specified — any object
        // Use a simple pattern that matches `{}`  or `{ "key": value, ... }`
        Ok(format!(
            r#"\{{{ws}({ws}{key}{ws}:{ws}{val}{ws}(,{ws}{key}{ws}:{ws}{val}{ws})*)?{ws}\}}"#,
            ws = ws,
            key = string_regex(),
            val = format!(
                "({}|{}|{}|{})",
                string_regex(),
                number_regex(),
                boolean_regex(),
                null_regex()
            ),
        ))
    }
}

fn array_regex(
    node: &serde_json::Value,
    ws: &str,
    root: &serde_json::Value,
) -> anyhow::Result<String> {
    let items_regex = if let Some(items) = node.get("items") {
        schema_to_regex(items, ws, root)?
    } else {
        // Any JSON value
        format!(
            "({}|{}|{}|{})",
            string_regex(),
            number_regex(),
            boolean_regex(),
            null_regex()
        )
    };

    // Handle minItems / maxItems
    let min_items = node.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let max_items = node.get("maxItems").and_then(|v| v.as_u64());

    if min_items == 0 && max_items.is_none() {
        // `[]` or `[item, item, ...]`
        Ok(format!(
            r"\[{ws}({item}{ws}(,{ws}{item}{ws})*)?{ws}\]",
            ws = ws,
            item = items_regex,
        ))
    } else if let Some(max) = max_items {
        // Bounded array
        if min_items == 0 && max == 0 {
            return Ok(format!(r"\[{ws}\]", ws = ws));
        }
        // Approximate with repetition
        let first = &items_regex;
        let rest = format!(",{ws}{item}", ws = ws, item = items_regex);
        let min_rest = if min_items > 0 { min_items - 1 } else { 0 };
        let max_rest = (max as usize).saturating_sub(1);
        Ok(format!(
            r"\[{ws}{first}({rest}){{{min},{max}}}{ws}\]",
            ws = ws,
            first = first,
            rest = rest,
            min = min_rest,
            max = max_rest,
        ))
    } else {
        // Min items only
        if min_items <= 1 {
            Ok(format!(
                r"\[{ws}({item}{ws}(,{ws}{item}{ws})*)?{ws}\]",
                ws = ws,
                item = items_regex,
            ))
        } else {
            let first = &items_regex;
            let rest = format!(",{ws}{item}", ws = ws, item = items_regex);
            let required_rest = min_items - 1;
            Ok(format!(
                r"\[{ws}{first}({rest}){{{req},}}{ws}\]",
                ws = ws,
                first = first,
                rest = rest,
                req = required_rest,
            ))
        }
    }
}

fn enum_to_regex(values: &[serde_json::Value]) -> anyhow::Result<String> {
    let alternatives: Vec<String> = values.iter().map(json_value_to_regex).collect();
    Ok(format!("({})", alternatives.join("|")))
}

fn const_to_regex(value: &serde_json::Value) -> anyhow::Result<String> {
    Ok(json_value_to_regex(value))
}

fn json_value_to_regex(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => {
            format!(r#""{}""#, regex_escape_json_string(s))
        }
        serde_json::Value::Number(n) => regex::escape(&n.to_string()),
        serde_json::Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        serde_json::Value::Null => "null".to_string(),
        _ => regex::escape(&value.to_string()),
    }
}

fn alternation(
    schemas: &[serde_json::Value],
    ws: &str,
    root: &serde_json::Value,
) -> anyhow::Result<String> {
    let alternatives: Result<Vec<String>, _> = schemas
        .iter()
        .map(|s| schema_to_regex(s, ws, root))
        .collect();
    let alts = alternatives?;
    Ok(format!("({})", alts.join("|")))
}

/// Escape a string for use in a JSON string regex pattern.
fn regex_escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => result.push_str(r"\\"),
            '"' => result.push_str(r#"\""#),
            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' => {
                result.push('\\');
                result.push(ch);
            }
            _ => result.push(ch),
        }
    }
    result
}

/// Resolve a JSON $ref path (only supports `#/definitions/Foo` and `#/$defs/Foo`).
fn resolve_ref<'a>(
    ref_path: &str,
    root: &'a serde_json::Value,
) -> anyhow::Result<&'a serde_json::Value> {
    let path = ref_path
        .strip_prefix("#/")
        .ok_or_else(|| anyhow::anyhow!("unsupported $ref path: {ref_path}"))?;

    let mut current = root;
    for segment in path.split('/') {
        current = current.get(segment).ok_or_else(|| {
            anyhow::anyhow!("$ref path segment not found: {segment} in {ref_path}")
        })?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_type() {
        let schema = serde_json::json!({"type": "string"});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match(r#""hello""#));
        assert!(re.is_match(r#""with \"escape\"""#));
        assert!(!re.is_match("hello"));
    }

    #[test]
    fn integer_type() {
        let schema = serde_json::json!({"type": "integer"});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("0"));
        assert!(re.is_match("42"));
        assert!(re.is_match("-100"));
        assert!(!re.is_match("01"));
        assert!(!re.is_match("3.14"));
    }

    #[test]
    fn number_type() {
        let schema = serde_json::json!({"type": "number"});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("0"));
        assert!(re.is_match("3.14"));
        assert!(re.is_match("-1.5e10"));
        assert!(re.is_match("42"));
    }

    #[test]
    fn boolean_type() {
        let schema = serde_json::json!({"type": "boolean"});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("true"));
        assert!(re.is_match("false"));
        assert!(!re.is_match("True"));
    }

    #[test]
    fn null_type() {
        let schema = serde_json::json!({"type": "null"});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("null"));
        assert!(!re.is_match("nil"));
    }

    #[test]
    fn enum_values() {
        let schema = serde_json::json!({"enum": ["red", "green", "blue"]});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match(r#""red""#));
        assert!(re.is_match(r#""green""#));
        assert!(re.is_match(r#""blue""#));
        assert!(!re.is_match(r#""yellow""#));
    }

    #[test]
    fn simple_object() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });
        let regex = schema_to_regex(&schema, "", &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match(r#"{"name":"John"}"#));
    }

    #[test]
    fn any_of() {
        let schema = serde_json::json!({
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        });
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match(r#""hello""#));
        assert!(re.is_match("42"));
        assert!(!re.is_match("true"));
    }

    #[test]
    fn const_value() {
        let schema = serde_json::json!({"const": "fixed"});
        let regex = schema_to_regex(&schema, DEFAULT_WHITESPACE, &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match(r#""fixed""#));
        assert!(!re.is_match(r#""other""#));
    }

    #[test]
    fn ref_resolution() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "value": {"$ref": "#/definitions/MyType"}
            },
            "required": ["value"],
            "definitions": {
                "MyType": {"type": "integer"}
            }
        });
        let regex = schema_to_regex(&schema, "", &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match(r#"{"value":42}"#));
    }

    #[test]
    fn empty_array() {
        let schema = serde_json::json!({
            "type": "array",
            "items": {"type": "integer"},
            "maxItems": 0
        });
        let regex = schema_to_regex(&schema, "", &schema).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("[]"));
        assert!(!re.is_match("[1]"));
    }
}
