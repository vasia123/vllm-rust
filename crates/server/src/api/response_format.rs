//! Response format enforcement for structured output.
//!
//! This module provides two levels of enforcement for `response_format`:
//!
//! 1. **Prompt injection**: For chat completions, injects a system message instructing
//!    the model to produce valid JSON output. This guides the model toward correct
//!    formatting during generation.
//!
//! 2. **Post-generation validation**: After generation completes, validates that the
//!    output conforms to the requested format (valid JSON, schema compliance). Returns
//!    an error to the client if validation fails.
//!
//! The engine-level `SamplingConstraint` (logit masking) provides a third layer of
//! enforcement during token selection, but that is handled in `crates/core`.

use super::types::ResponseFormat;
use vllm_core::tokenizer::ChatMessage;

/// System prompt text for JSON object mode.
const JSON_OBJECT_SYSTEM_PROMPT: &str =
    "You must respond with valid JSON only. Do not include any text outside the JSON object. \
     Your entire response must be a single valid JSON object.";

/// Inject a system message for JSON response format enforcement.
///
/// For `json_object` mode: adds a system prompt instructing JSON-only output.
/// For `json_schema` mode: adds a system prompt with the schema description.
/// For `text` mode or `None`: no-op (returns messages unchanged).
///
/// The injected message is prepended so the model sees it as the first instruction.
/// If a system message already exists, the JSON instruction is appended to it
/// to avoid conflicting system messages.
pub fn inject_json_system_prompt(
    messages: &mut Vec<ChatMessage>,
    response_format: Option<&ResponseFormat>,
) {
    let instruction = match response_format {
        None | Some(ResponseFormat::Text) => return,
        Some(ResponseFormat::JsonObject) => JSON_OBJECT_SYSTEM_PROMPT.to_string(),
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            let mut instruction = String::from(
                "You must respond with valid JSON that conforms to the following schema. \
                 Do not include any text outside the JSON. Your entire response must be \
                 a single valid JSON object or array matching this schema.\n\nSchema",
            );
            if let Some(ref name) = json_schema.name {
                instruction.push_str(&format!(" ({})", name));
            }
            instruction.push_str(":\n");
            // Compact schema representation for the system prompt
            instruction.push_str(
                &serde_json::to_string_pretty(&json_schema.schema).unwrap_or_else(|_| {
                    serde_json::to_string(&json_schema.schema).unwrap_or_default()
                }),
            );
            instruction
        }
    };

    // Check if the first message is already a system message
    if let Some(first) = messages.first_mut() {
        if first.role == "system" {
            // Append JSON instruction to existing system message
            let existing_text = first.text();
            let combined = format!("{}\n\n{}", existing_text, instruction);
            *first = ChatMessage::new("system", combined);
            return;
        }
    }

    // No existing system message -- prepend a new one
    messages.insert(0, ChatMessage::new("system", instruction));
}

/// Validation error for response format enforcement.
#[derive(Debug)]
pub struct ResponseFormatError {
    pub message: String,
}

impl std::fmt::Display for ResponseFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Validate that generated text conforms to the requested response format.
///
/// For `text` mode or `None`: always passes.
/// For `json_object` mode: validates the output is a valid JSON object.
/// For `json_schema` mode: validates the output is valid JSON matching the schema.
///
/// Returns `Ok(())` if validation passes, or `Err(ResponseFormatError)` with a
/// descriptive message if it fails.
pub fn validate_response_format(
    generated_text: &str,
    response_format: Option<&ResponseFormat>,
) -> Result<(), ResponseFormatError> {
    match response_format {
        None | Some(ResponseFormat::Text) => Ok(()),
        Some(ResponseFormat::JsonObject) => validate_json_object(generated_text),
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            // First validate it's valid JSON
            let value = parse_json(generated_text)?;
            // Then validate against schema
            validate_against_schema(&value, &json_schema.schema, json_schema.strict)
        }
    }
}

/// Validate that text is a valid JSON object.
fn validate_json_object(text: &str) -> Result<(), ResponseFormatError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ResponseFormatError {
            message: "response_format is 'json_object' but model output is empty".to_string(),
        });
    }

    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(serde_json::Value::Object(_)) => Ok(()),
        Ok(other) => Err(ResponseFormatError {
            message: format!(
                "response_format is 'json_object' but model output is not a JSON object, \
                 got JSON {} instead",
                json_type_name(&other)
            ),
        }),
        Err(e) => Err(ResponseFormatError {
            message: format!(
                "response_format is 'json_object' but model output is not valid JSON: {}",
                e
            ),
        }),
    }
}

/// Parse text as JSON, returning a descriptive error on failure.
fn parse_json(text: &str) -> Result<serde_json::Value, ResponseFormatError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ResponseFormatError {
            message: "response_format is 'json_schema' but model output is empty".to_string(),
        });
    }

    serde_json::from_str(trimmed).map_err(|e| ResponseFormatError {
        message: format!(
            "response_format is 'json_schema' but model output is not valid JSON: {}",
            e
        ),
    })
}

/// Validate a JSON value against a JSON schema (subset of JSON Schema draft-07).
///
/// This implements basic schema validation for the most common constraints:
/// - `type`: validates the JSON value type
/// - `required`: validates required object properties are present
/// - `properties`: recursively validates object properties
/// - `items`: recursively validates array items
/// - `enum`: validates the value is one of the allowed values
///
/// This is intentionally not a full JSON Schema validator. For production use with
/// complex schemas, consider integrating the `jsonschema` crate.
fn validate_against_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    strict: bool,
) -> Result<(), ResponseFormatError> {
    // If schema is a boolean true, everything validates
    if schema.as_bool() == Some(true) {
        return Ok(());
    }
    // If schema is a boolean false, nothing validates
    if schema.as_bool() == Some(false) {
        return Err(ResponseFormatError {
            message: "schema is 'false', no value can match".to_string(),
        });
    }

    let schema_obj = match schema.as_object() {
        Some(obj) => obj,
        None => return Ok(()), // Non-object schema without constraints passes
    };

    // Validate "type" constraint
    if let Some(expected_type) = schema_obj.get("type").and_then(|t| t.as_str()) {
        let actual_type = json_type_name(value);
        if actual_type != expected_type {
            return Err(ResponseFormatError {
                message: format!(
                    "schema validation failed: expected type '{}', got '{}'",
                    expected_type, actual_type
                ),
            });
        }
    }

    // Validate "enum" constraint
    if let Some(enum_values) = schema_obj.get("enum").and_then(|e| e.as_array()) {
        if !enum_values.contains(value) {
            return Err(ResponseFormatError {
                message: format!(
                    "schema validation failed: value does not match any enum variant, \
                     expected one of: {}",
                    serde_json::to_string(enum_values).unwrap_or_default()
                ),
            });
        }
    }

    // Validate object-specific constraints
    if let Some(obj) = value.as_object() {
        // Check required properties
        if let Some(required) = schema_obj.get("required").and_then(|r| r.as_array()) {
            for req_prop in required {
                if let Some(prop_name) = req_prop.as_str() {
                    if !obj.contains_key(prop_name) {
                        return Err(ResponseFormatError {
                            message: format!(
                                "schema validation failed: missing required property '{}'",
                                prop_name
                            ),
                        });
                    }
                }
            }
        }

        // Validate individual properties against their sub-schemas
        if let Some(properties) = schema_obj.get("properties").and_then(|p| p.as_object()) {
            for (prop_name, prop_schema) in properties {
                if let Some(prop_value) = obj.get(prop_name) {
                    validate_against_schema(prop_value, prop_schema, strict)?;
                }
            }
        }

        // In strict mode, reject additional properties not in schema
        if strict {
            if let Some(properties) = schema_obj.get("properties").and_then(|p| p.as_object()) {
                let allows_additional = schema_obj
                    .get("additionalProperties")
                    .and_then(|a| a.as_bool())
                    .unwrap_or(true);

                if !allows_additional {
                    for key in obj.keys() {
                        if !properties.contains_key(key) {
                            return Err(ResponseFormatError {
                                message: format!(
                                    "schema validation failed: unexpected additional property '{}'",
                                    key
                                ),
                            });
                        }
                    }
                }
            }
        }
    }

    // Validate array items
    if let Some(arr) = value.as_array() {
        if let Some(items_schema) = schema_obj.get("items") {
            for (i, item) in arr.iter().enumerate() {
                validate_against_schema(item, items_schema, strict).map_err(|e| {
                    ResponseFormatError {
                        message: format!("array item [{}]: {}", i, e.message),
                    }
                })?;
            }
        }
    }

    Ok(())
}

/// Get the JSON Schema type name for a serde_json Value.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::JsonSchemaSpec;

    // ─── System prompt injection tests ────────────────────────────────────

    #[test]
    fn inject_no_op_for_text_format() {
        let mut messages = vec![ChatMessage::new("user", "Hello")];
        inject_json_system_prompt(&mut messages, Some(&ResponseFormat::Text));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn inject_no_op_for_none() {
        let mut messages = vec![ChatMessage::new("user", "Hello")];
        inject_json_system_prompt(&mut messages, None);
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn inject_system_prompt_for_json_object() {
        let mut messages = vec![ChatMessage::new("user", "Give me some data")];
        inject_json_system_prompt(&mut messages, Some(&ResponseFormat::JsonObject));

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        let text = messages[0].text();
        assert!(text.contains("valid JSON"));
        assert!(text.contains("JSON object"));
    }

    #[test]
    fn inject_appends_to_existing_system_message() {
        let mut messages = vec![
            ChatMessage::new("system", "You are a helpful assistant."),
            ChatMessage::new("user", "Give me some data"),
        ];
        inject_json_system_prompt(&mut messages, Some(&ResponseFormat::JsonObject));

        // Should not add a new message, should modify existing system message
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        let text = messages[0].text();
        assert!(text.contains("helpful assistant"));
        assert!(text.contains("valid JSON"));
    }

    #[test]
    fn inject_system_prompt_for_json_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: Some("Person".to_string()),
                description: None,
                schema,
                strict: false,
            },
        };

        let mut messages = vec![ChatMessage::new("user", "Tell me about Alice")];
        inject_json_system_prompt(&mut messages, Some(&format));

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        let text = messages[0].text();
        assert!(text.contains("valid JSON"));
        assert!(text.contains("schema"));
        assert!(text.contains("Person"));
        assert!(text.contains("name"));
    }

    // ─── Post-generation validation tests ──────────────────────────────────

    #[test]
    fn validate_text_format_always_passes() {
        assert!(validate_response_format("anything", Some(&ResponseFormat::Text)).is_ok());
        assert!(validate_response_format("", Some(&ResponseFormat::Text)).is_ok());
        assert!(validate_response_format("not json", None).is_ok());
    }

    #[test]
    fn validate_json_object_valid() {
        let result = validate_response_format(
            r#"{"name": "Alice", "age": 30}"#,
            Some(&ResponseFormat::JsonObject),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn validate_json_object_with_whitespace() {
        let result = validate_response_format(
            r#"  {"name": "Alice"}  "#,
            Some(&ResponseFormat::JsonObject),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn validate_json_object_rejects_array() {
        let result = validate_response_format(r#"[1, 2, 3]"#, Some(&ResponseFormat::JsonObject));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("not a JSON object"));
        assert!(err.message.contains("array"));
    }

    #[test]
    fn validate_json_object_rejects_string() {
        let result =
            validate_response_format(r#""just a string""#, Some(&ResponseFormat::JsonObject));
        assert!(result.is_err());
    }

    #[test]
    fn validate_json_object_rejects_invalid_json() {
        let result =
            validate_response_format(r#"{invalid json"#, Some(&ResponseFormat::JsonObject));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("not valid JSON"));
    }

    #[test]
    fn validate_json_object_rejects_empty() {
        let result = validate_response_format("", Some(&ResponseFormat::JsonObject));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn validate_json_schema_valid() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        let result = validate_response_format(r#"{"name": "Alice", "age": 30}"#, Some(&format));
        assert!(result.is_ok());
    }

    #[test]
    fn validate_json_schema_missing_required() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        let result = validate_response_format(r#"{"name": "Alice"}"#, Some(&format));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("missing required property"));
        assert!(err.message.contains("age"));
    }

    #[test]
    fn validate_json_schema_wrong_type() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        // age should be integer but we provide a string
        let result =
            validate_response_format(r#"{"name": "Alice", "age": "thirty"}"#, Some(&format));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("expected type 'integer'"));
    }

    #[test]
    fn validate_json_schema_top_level_type_mismatch() {
        let schema = serde_json::json!({"type": "object"});
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        let result = validate_response_format(r#"[1, 2, 3]"#, Some(&format));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("expected type 'object'"));
    }

    #[test]
    fn validate_json_schema_nested_property_type() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        // Valid nested structure
        let result = validate_response_format(
            r#"{"address": {"city": "NYC", "zip": "10001"}}"#,
            Some(&format),
        );
        assert!(result.is_ok());

        // Missing required nested property
        let result = validate_response_format(r#"{"address": {"zip": "10001"}}"#, Some(&format));
        assert!(result.is_err());
    }

    #[test]
    fn validate_json_schema_array_items() {
        let schema = serde_json::json!({
            "type": "array",
            "items": {"type": "string"}
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        let result = validate_response_format(r#"["hello", "world"]"#, Some(&format));
        assert!(result.is_ok());

        let result = validate_response_format(r#"["hello", 42]"#, Some(&format));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("array item [1]"));
    }

    #[test]
    fn validate_json_schema_enum_constraint() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"]
                }
            }
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        let result = validate_response_format(r#"{"status": "active"}"#, Some(&format));
        assert!(result.is_ok());

        let result = validate_response_format(r#"{"status": "deleted"}"#, Some(&format));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("enum"));
    }

    #[test]
    fn validate_json_schema_strict_rejects_additional_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": false
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: true,
            },
        };

        // Only "name" allowed
        let result = validate_response_format(r#"{"name": "Alice"}"#, Some(&format));
        assert!(result.is_ok());

        // Extra property "age" should fail in strict mode
        let result = validate_response_format(r#"{"name": "Alice", "age": 30}"#, Some(&format));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("additional property"));
    }

    #[test]
    fn validate_json_schema_strict_allows_additional_by_default() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: true,
            },
        };

        // additionalProperties defaults to true
        let result = validate_response_format(r#"{"name": "Alice", "age": 30}"#, Some(&format));
        assert!(result.is_ok());
    }

    #[test]
    fn validate_json_schema_boolean_true_schema() {
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema: serde_json::Value::Bool(true),
                strict: false,
            },
        };

        let result = validate_response_format(r#"{"anything": "goes"}"#, Some(&format));
        assert!(result.is_ok());
    }

    #[test]
    fn validate_json_schema_empty_output() {
        let schema = serde_json::json!({"type": "object"});
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: None,
                description: None,
                schema,
                strict: false,
            },
        };

        let result = validate_response_format("", Some(&format));
        assert!(result.is_err());
    }

    // ─── json_type_name tests ──────────────────────────────────────────────

    #[test]
    fn json_type_names() {
        assert_eq!(json_type_name(&serde_json::Value::Null), "null");
        assert_eq!(json_type_name(&serde_json::Value::Bool(true)), "boolean");
        assert_eq!(
            json_type_name(&serde_json::Value::Number(42.into())),
            "integer"
        );
        assert_eq!(json_type_name(&serde_json::json!(3.14)), "number");
        assert_eq!(
            json_type_name(&serde_json::Value::String("hi".into())),
            "string"
        );
        assert_eq!(json_type_name(&serde_json::json!([])), "array");
        assert_eq!(json_type_name(&serde_json::json!({})), "object");
    }

    // ─── ResponseFormatError Display ───────────────────────────────────────

    #[test]
    fn response_format_error_display() {
        let err = ResponseFormatError {
            message: "test error".to_string(),
        };
        assert_eq!(format!("{}", err), "test error");
    }
}
