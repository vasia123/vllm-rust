use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use vllm_core::tokenizer::ChatMessage;
pub use vllm_core::tool_parser::{
    FunctionCall, FunctionDefinition, ToolCall, ToolChoice, ToolChoiceAuto,
    ToolChoiceFunction, ToolChoiceSpecific, ToolDefinition,
};

// ─── Prompt (string, array of strings, token IDs, or array of token IDs) ─

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Single(String),
    MultipleStrings(Vec<String>),
    TokenIds(Vec<u32>),
    MultipleTokenIds(Vec<Vec<u32>>),
}

#[derive(Debug, Clone)]
pub enum PromptInput {
    Text(String),
    TokenIds(Vec<u32>),
}

impl Prompt {
    pub fn into_inputs(self) -> Vec<PromptInput> {
        match self {
            Prompt::Single(s) => vec![PromptInput::Text(s)],
            Prompt::MultipleStrings(v) => v.into_iter().map(PromptInput::Text).collect(),
            Prompt::TokenIds(ids) => vec![PromptInput::TokenIds(ids)],
            Prompt::MultipleTokenIds(v) => v.into_iter().map(PromptInput::TokenIds).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Prompt::Single(_) => 1,
            Prompt::MultipleStrings(v) => v.len(),
            Prompt::TokenIds(_) => 1,
            Prompt::MultipleTokenIds(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── Completions ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Number of top logprobs to return per token (None = no logprobs).
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// If true, include prompt tokens in output with their logprobs.
    #[serde(default)]
    pub echo: bool,
    /// Response format for structured output (JSON schema, etc.)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogProbs>,
}

/// Log probabilities for completion tokens, following OpenAI format.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogProbs {
    /// Byte offsets in the text for each token.
    pub text_offset: Vec<usize>,
    /// Log probability of each token (None for prompt tokens without logprobs).
    pub token_logprobs: Vec<Option<f32>>,
    /// String representation of each token.
    pub tokens: Vec<String>,
    /// Top-k logprobs for each position (token string -> logprob).
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// ─── Chat Completions ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Response format for structured output (JSON schema, etc.)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Available tools for the model to use
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls how the model uses tools
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub message: ChatMessageResponse,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub delta: ChatDelta,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ─── Models ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

// ─── Common ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

fn default_max_tokens() -> usize {
    64
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

// ─── Structured Output / Response Format ─────────────────────────────────

/// Response format specification for structured output.
///
/// Supports text (default), JSON object, or JSON schema constraints.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// Plain text output (default, no constraints)
    #[default]
    #[serde(rename = "text")]
    Text,
    /// JSON object output (valid JSON required)
    #[serde(rename = "json_object")]
    JsonObject,
    /// JSON output conforming to a specific schema
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The JSON schema specification
        json_schema: JsonSchemaSpec,
    },
}

/// JSON schema specification for structured output.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonSchemaSpec {
    /// Optional name for the schema
    #[serde(default)]
    pub name: Option<String>,
    /// Optional description
    #[serde(default)]
    pub description: Option<String>,
    /// The actual JSON schema
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema validation
    #[serde(default)]
    pub strict: bool,
}

pub fn finish_reason_str(reason: &vllm_core::request::FinishReason) -> String {
    match reason {
        vllm_core::request::FinishReason::Eos => "stop".to_string(),
        vllm_core::request::FinishReason::Length => "length".to_string(),
        vllm_core::request::FinishReason::Stop => "stop".to_string(),
    }
}

pub fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
