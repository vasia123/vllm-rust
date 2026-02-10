//! Tool/function calling parser infrastructure.
//!
//! This module provides parsing of tool calls from LLM output in various formats:
//! - **Hermes**: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
//! - **GLM-4**: `<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
//! - **InternLM2**: `<|action_start|><|plugin|>{"name": ..., "parameters": ...}<|action_end|>`
//! - **Jamba**: `<tool_calls>[{"name": ..., "arguments": ...}]</tool_calls>`
//! - **JSON**: Raw JSON tool call arrays
//! - **Llama**: `<|python_tag|>{"name": ..., "arguments": ...}` (semicolon-separated)
//! - **Mistral**: `[TOOL_CALLS]` with v11+ or pre-v11 formats
//! - **DeepSeek V3**: Unicode token-delimited with ` ```json ` blocks
//! - **Pythonic**: `[func(arg='val')]` Python function call syntax
//! - **DeepSeek V3.1**: Unicode token-delimited `<｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜>`
//! - **Kimi K2**: `<|tool_call_begin|>functions.name:0<|tool_call_argument_begin|>{...}<|tool_call_end|>`
//! - **Phi-4 Mini**: `functools[{"name": ..., "arguments": ...}]`
//! - **Longcat**: `<longcat_tool_call>{"name": ..., "arguments": ...}</longcat_tool_call>`
//! - **xLAM**: Flexible — JSON arrays, code blocks, `[TOOL_CALLS]`, `<tool_call>` tags
//! - **GigaChat3**: `function call{"name": ..., "arguments": ...}`
//! - **FunctionGemma**: `<start_function_call>call:name{param:<escape>val<escape>}<end_function_call>`
//! - **Hunyuan**: `<tool_calls>[{"name": ..., "arguments": ...}]</tool_calls>`
//! - **ERNIE-4.5**: `<tool_call>{"name": ..., "arguments": ...}</tool_call>` with `<think>` tags
//! - **Seed-OSS**: `<seed:tool_call><function=name><parameter=key>val</parameter></function></seed:tool_call>`
//! - **DeepSeek V3.2**: `<｜DSML｜function_calls><｜DSML｜invoke name="...">` DSML XML format
//! - **Step-3**: `<｜tool_calls_begin｜>` + `<steptml:invoke name="...">` XML format
//! - **Qwen3 Coder**: `<tool_call><function=name><parameter=key>val</parameter></function></tool_call>`
//! - **Olmo3**: `<function_calls>func(args)</function_calls>` newline-separated pythonic
//! - **Llama4 Pythonic**: `<|python_start|>[func(args)]<|python_end|>` pythonic with tags
//! - **MiniMax**: `<tool_calls>{"name": ..., "arguments": ...}</tool_calls>` JSON with `<think>` stripping
//! - **MiniMax M2**: `<minimax:tool_call><invoke name="..."><parameter name="...">val</parameter></invoke></minimax:tool_call>`
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

mod deepseek_v3;
mod deepseek_v31;
mod deepseek_v32;
mod ernie45;
mod functiongemma;
mod gigachat3;
mod glm4;
mod granite;
mod hermes;
mod hunyuan;
mod internlm2;
mod jamba;
mod json_parser;
mod kimi_k2;
mod llama;
mod llama4_pythonic;
mod longcat;
mod minimax;
mod minimax_m2;
mod mistral;
mod olmo3;
mod phi4mini;
pub(super) mod pythonic;
mod qwen3coder;
mod seed_oss;
mod step3;
mod xlam;

pub use deepseek_v3::DeepSeekV3ToolParser;
pub use deepseek_v31::DeepSeekV31ToolParser;
pub use deepseek_v32::DeepSeekV32ToolParser;
pub use ernie45::Ernie45ToolParser;
pub use functiongemma::FunctionGemmaToolParser;
pub use gigachat3::GigaChat3ToolParser;
pub use glm4::Glm4ToolParser;
pub use granite::{Granite20bFCToolParser, GraniteToolParser};
pub use hermes::HermesToolParser;
pub use hunyuan::HunyuanToolParser;
pub use internlm2::InternLm2ToolParser;
pub use jamba::JambaToolParser;
pub use json_parser::JsonToolParser;
pub use kimi_k2::KimiK2ToolParser;
pub use llama::LlamaToolParser;
pub use llama4_pythonic::Llama4PythonicToolParser;
pub use longcat::LongcatToolParser;
pub use minimax::MinimaxToolParser;
pub use minimax_m2::MinimaxM2ToolParser;
pub use mistral::MistralToolParser;
pub use olmo3::Olmo3PythonicToolParser;
pub use phi4mini::Phi4MiniToolParser;
pub use pythonic::PythonicToolParser;
pub use qwen3coder::Qwen3CoderToolParser;
pub use seed_oss::SeedOssToolParser;
pub use step3::Step3ToolParser;
pub use xlam::XLamToolParser;

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

/// Find complete JSON objects at the top level of a string.
///
/// Tracks brace depth to correctly handle nested objects and
/// string escaping. Returns slices of each complete `{...}` object.
pub(super) fn find_json_objects(s: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let mut depth: i32 = 0;
    let mut start = None;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s_idx) = start {
                        objects.push(&s[s_idx..=i]);
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }

    objects
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

    #[test]
    fn find_json_objects_single() {
        let objs = find_json_objects(r#"{"a": 1}"#);
        assert_eq!(objs, vec![r#"{"a": 1}"#]);
    }

    #[test]
    fn find_json_objects_multiple_semicolon_separated() {
        let objs = find_json_objects(r#"{"a": 1}; {"b": 2}"#);
        assert_eq!(objs, vec![r#"{"a": 1}"#, r#"{"b": 2}"#]);
    }

    #[test]
    fn find_json_objects_nested() {
        let input = r#"{"a": {"b": {"c": 1}}}"#;
        let objs = find_json_objects(input);
        assert_eq!(objs, vec![input]);
    }

    #[test]
    fn find_json_objects_braces_in_strings() {
        let input = r#"{"msg": "hello {world}"}"#;
        let objs = find_json_objects(input);
        assert_eq!(objs, vec![input]);
    }

    #[test]
    fn find_json_objects_escaped_quotes() {
        let input = r#"{"msg": "say \"hi\""}"#;
        let objs = find_json_objects(input);
        assert_eq!(objs, vec![input]);
    }

    #[test]
    fn find_json_objects_no_objects() {
        assert!(find_json_objects("no json here").is_empty());
        assert!(find_json_objects("").is_empty());
    }

    #[test]
    fn find_json_objects_adjacent() {
        let objs = find_json_objects(r#"{"a":1}{"b":2}"#);
        assert_eq!(objs, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }
}
