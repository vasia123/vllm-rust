//! Tool/function calling parser infrastructure.
//!
//! This module provides parsing of tool calls from LLM output in various formats:
//! - **Hermes**: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
//! - **JSON**: Raw JSON tool call arrays
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::tool_parser::{HermesToolParser, ToolCallParser};
//!
//! let parser = HermesToolParser;
//! let output = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}</tool_call>";
//! let calls = parser.parse(output)?;
//! ```

mod hermes;
mod json_parser;

pub use hermes::HermesToolParser;
pub use json_parser::JsonToolParser;

use serde::{Deserialize, Serialize};

/// Parsed tool call from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// Type of the call (typically "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function being called
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// Arguments as a JSON string
    pub arguments: String,
}

/// Tool definition for the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Type of tool (typically "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    #[serde(default)]
    pub description: Option<String>,
    /// JSON schema describing the function parameters
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// Tool choice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let the model decide whether to use tools
    Auto(ToolChoiceAuto),
    /// Force a specific tool
    Specific(ToolChoiceSpecific),
}

/// Auto tool choice mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceAuto {
    /// Model decides whether to call functions
    Auto,
    /// Model must not call functions
    None,
    /// Model must call at least one function
    Required,
}

/// Force a specific tool to be called.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceSpecific {
    /// Type of tool (typically "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function to call
    pub function: ToolChoiceFunction,
}

/// Function specification in tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// Name of the function to call
    pub name: String,
}

/// Trait for parsing tool calls from LLM output.
///
/// Different models use different formats for emitting tool calls.
/// This trait provides a common interface for parsing these formats.
pub trait ToolCallParser: Send + Sync {
    /// Parse tool calls from the given output text.
    ///
    /// # Arguments
    /// * `output` - The raw output text from the LLM
    ///
    /// # Returns
    /// A vector of parsed tool calls, or an error if parsing fails
    fn parse(&self, output: &str) -> anyhow::Result<Vec<ToolCall>>;

    /// Extract the content portion (non-tool-call text) from output.
    ///
    /// # Arguments
    /// * `output` - The raw output text from the LLM
    ///
    /// # Returns
    /// The content text if present, None if the output is purely tool calls
    fn extract_content(&self, output: &str) -> Option<String>;

    /// Check if the output contains any tool calls.
    fn has_tool_calls(&self, output: &str) -> bool {
        self.parse(output).map(|c| !c.is_empty()).unwrap_or(false)
    }
}

/// Generate a unique tool call ID.
pub fn generate_tool_call_id() -> String {
    let uuid_str = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("call_{}", &uuid_str[..24])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city": "NYC"}"#.to_string(),
            },
        };

        let json = serde_json::to_string(&call).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("get_weather"));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the weather for a city".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                })),
            },
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("get_weather"));
        assert!(json.contains("Get the weather"));
    }

    #[test]
    fn test_tool_call_id_generation() {
        let id1 = generate_tool_call_id();
        let id2 = generate_tool_call_id();

        assert!(id1.starts_with("call_"));
        assert_eq!(id1.len(), 29); // "call_" + 24 chars
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_tool_choice_auto_serialization() {
        let choice = ToolChoice::Auto(ToolChoiceAuto::Auto);
        let json = serde_json::to_string(&choice).unwrap();
        assert_eq!(json, r#""auto""#);

        let choice = ToolChoice::Auto(ToolChoiceAuto::Required);
        let json = serde_json::to_string(&choice).unwrap();
        assert_eq!(json, r#""required""#);
    }

    #[test]
    fn test_tool_choice_specific_serialization() {
        let choice = ToolChoice::Specific(ToolChoiceSpecific {
            tool_type: "function".to_string(),
            function: ToolChoiceFunction {
                name: "get_weather".to_string(),
            },
        });
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("get_weather"));
        assert!(json.contains("function"));
    }
}
