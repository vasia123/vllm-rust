use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use vllm_core::tokenizer::ChatMessage;
pub use vllm_core::tool_parser::{
    FunctionCall, FunctionDefinition, ToolCall, ToolChoice, ToolChoiceAuto, ToolChoiceFunction,
    ToolChoiceSpecific, ToolDefinition,
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
    /// Options for streaming responses. Only used when `stream` is true.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
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
    /// Frequency penalty (OpenAI convention). Range: -2.0 to 2.0.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty (OpenAI convention). Range: -2.0 to 2.0.
    #[serde(default)]
    pub presence_penalty: f32,
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
    /// Token logit bias: map of token ID (as string) to bias value (-100.0 to 100.0).
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Response format for structured output (JSON schema, etc.)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Name of the LoRA adapter to use for this request.
    /// Must match an adapter loaded at server startup.
    #[serde(default)]
    pub lora_name: Option<String>,
    /// A unique identifier representing the end-user (for tracking/abuse detection).
    #[serde(default)]
    pub user: Option<String>,
    /// Number of completions to return for each prompt. Default is 1.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Generate `best_of` completions and return the best `n`.
    /// Must be >= n. Default is 1 (or n when n > 1).
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Beam search width. When set (> 1), uses beam search decoding instead of sampling.
    #[serde(default)]
    pub beam_width: Option<usize>,
    /// Length penalty for beam search (0 = no normalization, 1 = full normalization).
    #[serde(default)]
    pub length_penalty: Option<f32>,
    /// Whether to stop early when all beams have finished in beam search.
    #[serde(default)]
    pub early_stopping: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: &'static str,
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
    pub id: Arc<str>,
    pub object: &'static str,
    pub created: u64,
    pub model: Arc<str>,
    pub system_fingerprint: &'static str,
    pub choices: Vec<CompletionChunkChoice>,
    /// Usage statistics, included in the final chunk when stream_options.include_usage is true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogProbs>,
}

// ─── Stream Options ───────────────────────────────────────────────────────

/// Options for streaming responses.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StreamOptions {
    /// If true, the final streaming chunk will include a `usage` field with
    /// token counts (prompt_tokens, completion_tokens, total_tokens).
    #[serde(default)]
    pub include_usage: bool,
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
    /// Options for streaming responses. Only used when `stream` is true.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
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
    /// Frequency penalty (OpenAI convention). Range: -2.0 to 2.0.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty (OpenAI convention). Range: -2.0 to 2.0.
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Token logit bias: map of token ID (as string) to bias value (-100.0 to 100.0).
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Response format for structured output (JSON schema, etc.)
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Available tools for the model to use
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls how the model uses tools
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Name of the LoRA adapter to use for this request.
    /// Must match an adapter loaded at server startup.
    #[serde(default)]
    pub lora_name: Option<String>,
    /// Whether to return log probabilities of the output tokens.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of most likely tokens to return at each position (1-20).
    /// Requires logprobs to be true.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// A unique identifier representing the end-user (for tracking/abuse detection).
    #[serde(default)]
    pub user: Option<String>,
    /// Number of chat completion choices to generate. Default 1.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Generate `best_of` completions and return the best `n`.
    /// Must be >= n. Default is n (no extra candidates).
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Beam search width. When set (> 1), uses beam search decoding instead of sampling.
    #[serde(default)]
    pub beam_width: Option<usize>,
    /// Length penalty for beam search (0 = no normalization, 1 = full normalization).
    #[serde(default)]
    pub length_penalty: Option<f32>,
    /// Whether to stop early when all beams have finished in beam search.
    #[serde(default)]
    pub early_stopping: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: &'static str,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub message: ChatMessageResponse,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

/// Log probabilities for chat completion tokens, following OpenAI format.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    /// A list of message content tokens with log probability information.
    pub content: Vec<ChatLogProbToken>,
}

/// Log probability information for a single token in chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbToken {
    /// The token string.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// UTF-8 byte representation of the token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Top alternative tokens with their log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<ChatTopLogProb>>,
}

/// A top-k log probability entry for chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatTopLogProb {
    /// The token string.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// UTF-8 byte representation of the token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Deprecated alias for backward compatibility with clients expecting `reasoning_content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: Arc<str>,
    pub object: &'static str,
    pub created: u64,
    pub model: Arc<str>,
    pub system_fingerprint: &'static str,
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// Usage statistics, included in the final chunk when stream_options.include_usage is true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub delta: ChatDelta,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Deprecated alias for backward compatibility with clients expecting `reasoning_content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    pub root: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub permission: Vec<ModelPermission>,
}

#[derive(Debug, Serialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
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

fn default_n() -> usize {
    1
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

/// Return a deterministic system fingerprint derived from the package version.
///
/// The value is computed once and cached for the lifetime of the process.
/// Format: `fp_` followed by the first 10 hex characters of a hash of the
/// cargo package version, matching the OpenAI `system_fingerprint` convention.
pub fn system_fingerprint() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        let version = env!("CARGO_PKG_VERSION");
        // FNV-1a 64-bit hash — simple, no-dependency, deterministic.
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in version.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        // Mask to 40 bits so the hex representation is exactly 10 characters.
        format!("fp_{:010x}", hash & 0xff_ffff_ffff)
    })
}

/// Convert OpenAI-style logit_bias (string token IDs -> bias) to engine format (token_id, bias) pairs.
///
/// OpenAI API accepts token IDs as JSON object keys (strings). The engine expects
/// `Vec<(u32, f32)>`. Invalid (non-numeric) keys are silently skipped.
pub fn convert_logit_bias(bias: &Option<HashMap<String, f32>>) -> Option<Vec<(u32, f32)>> {
    bias.as_ref().map(|map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    })
}

// ─── Tokenize / Detokenize ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
    pub max_model_len: usize,
}

#[derive(Debug, Deserialize)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub prompt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── StreamOptions tests ─────────────────────────────────────────────

    #[test]
    fn stream_options_deserialize_include_usage_true() {
        let json = r#"{"include_usage": true}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();
        assert!(opts.include_usage);
    }

    #[test]
    fn stream_options_deserialize_include_usage_false() {
        let json = r#"{"include_usage": false}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();
        assert!(!opts.include_usage);
    }

    #[test]
    fn stream_options_deserialize_empty_object_defaults() {
        let json = r#"{}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();
        assert!(!opts.include_usage);
    }

    #[test]
    fn stream_options_serialize_roundtrip() {
        let opts = StreamOptions {
            include_usage: true,
        };
        let json = serde_json::to_string(&opts).unwrap();
        let parsed: StreamOptions = serde_json::from_str(&json).unwrap();
        assert!(parsed.include_usage);
    }

    // ─── ChatCompletionRequest stream_options parsing ────────────────────

    #[test]
    fn chat_request_with_stream_options() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.unwrap().include_usage);
    }

    #[test]
    fn chat_request_without_stream_options() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
        assert!(req.stream_options.is_none());
    }

    #[test]
    fn chat_request_with_logprobs() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": true,
            "top_logprobs": 5
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
    }

    #[test]
    fn chat_request_logprobs_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.logprobs.is_none());
        assert!(req.top_logprobs.is_none());
    }

    // ─── CompletionRequest stream_options parsing ────────────────────────

    #[test]
    fn completion_request_with_stream_options() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "stream": true,
            "stream_options": {"include_usage": true}
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.unwrap().include_usage);
    }

    #[test]
    fn completion_request_without_stream_options() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream_options.is_none());
    }

    // ─── Models response format ──────────────────────────────────────────

    #[test]
    fn models_response_serialization() {
        let response = ModelsResponse {
            object: "list",
            data: vec![ModelObject {
                id: "test-model".to_string(),
                object: "model",
                created: 1234567890,
                owned_by: "vllm-rust".to_string(),
                root: "test-model".to_string(),
                parent: None,
                permission: vec![ModelPermission {
                    id: "modelperm-test-123".to_string(),
                    object: "model_permission",
                    created: 1234567890,
                    allow_create_engine: true,
                    allow_sampling: true,
                    allow_logprobs: true,
                    allow_search_indices: true,
                    allow_view: true,
                    allow_fine_tuning: true,
                    organization: "*".to_string(),
                    group: None,
                    is_blocking: false,
                }],
            }],
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["created"], 1234567890);
        assert_eq!(json["data"][0]["owned_by"], "vllm-rust");
    }

    // ─── Chat completion chunk with usage ────────────────────────────────

    #[test]
    fn chat_chunk_without_usage_omits_field() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-123".into(),
            object: "chat.completion.chunk",
            created: 1234567890,
            model: "test-model".into(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn chat_chunk_with_usage_includes_field() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-123".into(),
            object: "chat.completion.chunk",
            created: 1234567890,
            model: "test-model".into(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["completion_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 15);
        assert!(json["choices"].as_array().unwrap().is_empty());
    }

    // ─── Completion chunk with usage ─────────────────────────────────────

    #[test]
    fn completion_chunk_without_usage_omits_field() {
        let chunk = CompletionChunk {
            id: "cmpl-123".into(),
            object: "text_completion",
            created: 1234567890,
            model: "test-model".into(),
            system_fingerprint: system_fingerprint(),
            choices: vec![CompletionChunkChoice {
                text: "hello".to_string(),
                index: 0,
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn completion_chunk_with_usage_includes_field() {
        let chunk = CompletionChunk {
            id: "cmpl-123".into(),
            object: "text_completion",
            created: 1234567890,
            model: "test-model".into(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens: 8,
                completion_tokens: 3,
                total_tokens: 11,
            }),
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 8);
        assert_eq!(json["usage"]["completion_tokens"], 3);
        assert_eq!(json["usage"]["total_tokens"], 11);
    }

    // ─── Chunk choices with logprobs ──────────────────────────────────────

    #[test]
    fn completion_chunk_choice_without_logprobs_omits_field() {
        let choice = CompletionChunkChoice {
            text: "hello".to_string(),
            index: 0,
            finish_reason: None,
            logprobs: None,
        };
        let json = serde_json::to_value(&choice).unwrap();
        assert!(json.get("logprobs").is_none());
    }

    #[test]
    fn completion_chunk_choice_with_logprobs_includes_field() {
        let mut top = HashMap::new();
        top.insert("hello".to_string(), -0.5f32);
        top.insert("hi".to_string(), -1.2f32);

        let choice = CompletionChunkChoice {
            text: "hello".to_string(),
            index: 0,
            finish_reason: None,
            logprobs: Some(CompletionLogProbs {
                text_offset: vec![0],
                token_logprobs: vec![Some(-0.5)],
                tokens: vec!["hello".to_string()],
                top_logprobs: vec![Some(top)],
            }),
        };
        let json = serde_json::to_value(&choice).unwrap();
        let lp = &json["logprobs"];
        assert!(lp.is_object());
        assert_eq!(lp["tokens"][0], "hello");
        assert_eq!(lp["text_offset"][0], 0);
        assert!(lp["token_logprobs"][0].as_f64().unwrap() < 0.0);
        assert!(lp["top_logprobs"][0].is_object());
    }

    #[test]
    fn chat_chunk_choice_without_logprobs_omits_field() {
        let choice = ChatCompletionChunkChoice {
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: Some("hello".to_string()),
                reasoning: None,
                reasoning_content: None,
            },
            index: 0,
            finish_reason: None,
            logprobs: None,
        };
        let json = serde_json::to_value(&choice).unwrap();
        assert!(json.get("logprobs").is_none());
    }

    #[test]
    fn chat_chunk_choice_with_logprobs_includes_field() {
        let choice = ChatCompletionChunkChoice {
            delta: ChatDelta {
                role: None,
                content: Some("hello".to_string()),
                reasoning: None,
                reasoning_content: None,
            },
            index: 0,
            finish_reason: None,
            logprobs: Some(ChatLogProbs {
                content: vec![ChatLogProbToken {
                    token: "hello".to_string(),
                    logprob: -0.5,
                    bytes: Some(vec![104, 101, 108, 108, 111]),
                    top_logprobs: Some(vec![ChatTopLogProb {
                        token: "hello".to_string(),
                        logprob: -0.5,
                        bytes: Some(vec![104, 101, 108, 108, 111]),
                    }]),
                }],
            }),
        };
        let json = serde_json::to_value(&choice).unwrap();
        let lp = &json["logprobs"];
        assert!(lp.is_object());
        assert_eq!(lp["content"][0]["token"], "hello");
        assert!(lp["content"][0]["logprob"].as_f64().unwrap() < 0.0);
        assert_eq!(
            lp["content"][0]["top_logprobs"].as_array().unwrap().len(),
            1
        );
    }

    // ─── Logprobs types serialization ────────────────────────────────────

    #[test]
    fn chat_logprobs_serialization() {
        let logprobs = ChatLogProbs {
            content: vec![ChatLogProbToken {
                token: "hello".to_string(),
                logprob: -0.5,
                bytes: Some(vec![104, 101, 108, 108, 111]),
                top_logprobs: Some(vec![
                    ChatTopLogProb {
                        token: "hello".to_string(),
                        logprob: -0.5,
                        bytes: Some(vec![104, 101, 108, 108, 111]),
                    },
                    ChatTopLogProb {
                        token: "hi".to_string(),
                        logprob: -1.2,
                        bytes: Some(vec![104, 105]),
                    },
                ]),
            }],
        };
        let json = serde_json::to_value(&logprobs).unwrap();
        assert_eq!(json["content"][0]["token"], "hello");
        assert!(json["content"][0]["logprob"].as_f64().unwrap() < 0.0);
        assert_eq!(
            json["content"][0]["top_logprobs"].as_array().unwrap().len(),
            2
        );
        assert_eq!(json["content"][0]["top_logprobs"][1]["token"], "hi");
    }

    #[test]
    fn chat_logprobs_omits_none_fields() {
        let logprobs = ChatLogProbs {
            content: vec![ChatLogProbToken {
                token: "test".to_string(),
                logprob: -1.0,
                bytes: None,
                top_logprobs: None,
            }],
        };
        let json = serde_json::to_value(&logprobs).unwrap();
        assert!(json["content"][0].get("bytes").is_none());
        assert!(json["content"][0].get("top_logprobs").is_none());
    }

    // ─── Usage serialization ─────────────────────────────────────────────

    #[test]
    fn usage_serialization() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["prompt_tokens"], 10);
        assert_eq!(json["completion_tokens"], 20);
        assert_eq!(json["total_tokens"], 30);
    }

    // ─── ChatCompletionResponse format ───────────────────────────────────

    #[test]
    fn chat_completion_response_includes_usage() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion",
            created: 1234567890,
            model: "test-model".to_string(),
            system_fingerprint: system_fingerprint(),
            choices: vec![ChatCompletionChoice {
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    reasoning: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                index: 0,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["completion_tokens"], 1);
        assert_eq!(json["usage"]["total_tokens"], 6);
    }

    // ─── CompletionResponse format ───────────────────────────────────────

    #[test]
    fn completion_response_includes_usage() {
        let response = CompletionResponse {
            id: "cmpl-123".to_string(),
            object: "text_completion",
            created: 1234567890,
            model: "test-model".to_string(),
            system_fingerprint: system_fingerprint(),
            choices: vec![CompletionChoice {
                text: "world".to_string(),
                index: 0,
                finish_reason: Some("length".to_string()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 3);
        assert_eq!(json["usage"]["completion_tokens"], 2);
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    // ─── timestamp_now sanity ────────────────────────────────────────────

    #[test]
    fn timestamp_now_returns_reasonable_value() {
        let ts = timestamp_now();
        // Should be after 2024-01-01 (1704067200)
        assert!(ts > 1_704_067_200);
    }

    // ─── TokenizeRequest / TokenizeResponse ──────────────────────────────

    #[test]
    fn tokenize_request_deserialization() {
        let json = r#"{
            "model": "test-model",
            "prompt": "Hello world"
        }"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test-model");
        assert_eq!(req.prompt, "Hello world");
    }

    #[test]
    fn tokenize_response_serialization() {
        let response = TokenizeResponse {
            tokens: vec![15496, 995],
            count: 2,
            max_model_len: 4096,
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["tokens"], serde_json::json!([15496, 995]));
        assert_eq!(json["count"], 2);
        assert_eq!(json["max_model_len"], 4096);
    }

    // ─── DetokenizeRequest / DetokenizeResponse ─────────────────────────

    #[test]
    fn detokenize_request_deserialization() {
        let json = r#"{
            "model": "test-model",
            "tokens": [15496, 995]
        }"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test-model");
        assert_eq!(req.tokens, vec![15496, 995]);
    }

    #[test]
    fn detokenize_request_empty_tokens() {
        let json = r#"{
            "model": "test-model",
            "tokens": []
        }"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(req.tokens.is_empty());
    }

    #[test]
    fn detokenize_response_serialization() {
        let response = DetokenizeResponse {
            prompt: "Hello world".to_string(),
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["prompt"], "Hello world");
    }

    // ─── Multimodal chat message deserialization ─────────────────────────

    #[test]
    fn chat_request_with_string_content_backward_compatible() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].text(), "Hello, how are you?");
        assert!(!req.messages[0].has_images());
    }

    #[test]
    fn chat_request_with_multimodal_content() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
                    ]
                }
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].text(), "What's in this image?");
        assert!(req.messages[0].has_images());
    }

    #[test]
    fn chat_request_with_base64_image() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
                    ]
                }
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert!(req.messages[0].has_images());
        let urls = req.messages[0].content.image_urls();
        assert_eq!(urls.len(), 1);
        assert!(urls[0].starts_with("data:image/png;base64,"));
    }

    #[test]
    fn chat_request_with_image_detail_field() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg", "detail": "high"}}
                    ]
                }
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages[0].has_images());
    }

    #[test]
    fn chat_request_mixed_text_and_multimodal_messages() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                    ]
                },
                {"role": "assistant", "content": "It is a cat."},
                {"role": "user", "content": "Are you sure?"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 4);
        // System message is text-only
        assert!(!req.messages[0].has_images());
        // User message is multimodal
        assert!(req.messages[1].has_images());
        // Assistant reply is text-only
        assert!(!req.messages[2].has_images());
        // Follow-up is text-only
        assert!(!req.messages[3].has_images());
    }

    #[test]
    fn chat_request_multiple_images_in_one_message() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}}
                    ]
                }
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let urls = req.messages[0].content.image_urls();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://example.com/a.jpg");
        assert_eq!(urls[1], "https://example.com/b.jpg");
    }

    // ─── Echo parameter deserialization ───────────────────────────────

    #[test]
    fn completion_request_echo_true_deserializes() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "echo": true
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.echo);
    }

    #[test]
    fn completion_request_echo_defaults_to_false() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(!req.echo);
    }

    // ─── User field deserialization ───────────────────────────────────

    #[test]
    fn completion_request_user_field_deserializes() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "user": "user-123"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.user.as_deref(), Some("user-123"));
    }

    #[test]
    fn completion_request_user_field_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.user.is_none());
    }

    #[test]
    fn chat_request_user_field_deserializes() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "user": "user-456"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.user.as_deref(), Some("user-456"));
    }

    #[test]
    fn chat_request_user_field_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.user.is_none());
    }

    // ─── best_of field deserialization ────────────────────────────────

    #[test]
    fn completion_request_best_of_explicit_value() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "best_of": 5
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.best_of, Some(5));
    }

    #[test]
    fn completion_request_best_of_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.best_of, None);
    }

    #[test]
    fn completion_request_n_explicit_value() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "n": 3
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, 3);
    }

    #[test]
    fn completion_request_n_defaults_to_one() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, 1);
    }

    #[test]
    fn chat_request_best_of_explicit_value() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "best_of": 5
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.best_of, Some(5));
    }

    #[test]
    fn chat_request_best_of_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.best_of, None);
    }

    // ─── n field deserialization ──────────────────────────────────────

    #[test]
    fn chat_request_n_explicit_value() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 3
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, 3);
    }

    #[test]
    fn chat_request_n_defaults_to_one() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, 1);
    }

    // ─── system_fingerprint ──────────────────────────────────────────

    #[test]
    fn system_fingerprint_is_deterministic() {
        let a = system_fingerprint();
        let b = system_fingerprint();
        assert_eq!(a, b);
    }

    #[test]
    fn system_fingerprint_starts_with_prefix() {
        let fp = system_fingerprint();
        assert!(
            fp.starts_with("fp_"),
            "expected fingerprint to start with 'fp_', got: {fp}"
        );
    }

    #[test]
    fn system_fingerprint_has_reasonable_length() {
        let fp = system_fingerprint();
        // "fp_" (3) + 10 hex chars = 13
        assert_eq!(fp.len(), 13, "unexpected fingerprint length: {fp}");
    }

    #[test]
    fn completion_response_serializes_system_fingerprint() {
        let response = CompletionResponse {
            id: "cmpl-test".to_string(),
            object: "text_completion",
            created: 0,
            model: "m".to_string(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        let fp = json["system_fingerprint"].as_str().unwrap();
        assert!(fp.starts_with("fp_"));
    }

    #[test]
    fn chat_completion_response_serializes_system_fingerprint() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 0,
            model: "m".to_string(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        let fp = json["system_fingerprint"].as_str().unwrap();
        assert!(fp.starts_with("fp_"));
    }

    #[test]
    fn completion_chunk_serializes_system_fingerprint() {
        let chunk = CompletionChunk {
            id: "cmpl-test".into(),
            object: "text_completion",
            created: 0,
            model: "m".into(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        let fp = json["system_fingerprint"].as_str().unwrap();
        assert!(fp.starts_with("fp_"));
    }

    #[test]
    fn chat_completion_chunk_serializes_system_fingerprint() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            system_fingerprint: system_fingerprint(),
            choices: vec![],
            usage: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        let fp = json["system_fingerprint"].as_str().unwrap();
        assert!(fp.starts_with("fp_"));
    }

    // ─── Frequency and presence penalty deserialization ──────────────

    #[test]
    fn completion_request_frequency_penalty_explicit() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "frequency_penalty": 1.5
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.frequency_penalty - 1.5).abs() < 1e-6);
    }

    #[test]
    fn completion_request_presence_penalty_explicit() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "presence_penalty": -0.5
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.presence_penalty - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn completion_request_penalties_default_to_zero() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.frequency_penalty).abs() < 1e-6);
        assert!((req.presence_penalty).abs() < 1e-6);
    }

    #[test]
    fn chat_request_frequency_penalty_explicit() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "frequency_penalty": 2.0
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.frequency_penalty - 2.0).abs() < 1e-6);
    }

    #[test]
    fn chat_request_presence_penalty_explicit() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "presence_penalty": -1.0
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.presence_penalty - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn chat_request_penalties_default_to_zero() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.frequency_penalty).abs() < 1e-6);
        assert!((req.presence_penalty).abs() < 1e-6);
    }

    #[test]
    fn completion_request_both_penalties_together() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "frequency_penalty": 0.8,
            "presence_penalty": 0.6
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!((req.frequency_penalty - 0.8).abs() < 1e-6);
        assert!((req.presence_penalty - 0.6).abs() < 1e-6);
    }

    // ─── beam_width deserialization ─────────────────────────────────

    #[test]
    fn completion_request_beam_width_explicit() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello",
            "beam_width": 4,
            "length_penalty": 0.8,
            "early_stopping": true
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.beam_width, Some(4));
        assert!((req.length_penalty.unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(req.early_stopping, Some(true));
    }

    #[test]
    fn completion_request_beam_width_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "prompt": "hello"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.beam_width.is_none());
        assert!(req.length_penalty.is_none());
        assert!(req.early_stopping.is_none());
    }

    #[test]
    fn chat_request_beam_width_explicit() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "beam_width": 2,
            "length_penalty": 1.5
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.beam_width, Some(2));
        assert!((req.length_penalty.unwrap() - 1.5).abs() < 1e-6);
        assert!(req.early_stopping.is_none());
    }

    #[test]
    fn chat_request_beam_width_defaults_to_none() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.beam_width.is_none());
    }

    // ─── reasoning / interleaved thinking ───────────────────────────

    #[test]
    fn chat_message_response_omits_reasoning_when_none() {
        let msg = ChatMessageResponse {
            role: "assistant".to_string(),
            content: Some("Hello".to_string()),
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("reasoning").is_none());
        assert!(json.get("reasoning_content").is_none());
    }

    #[test]
    fn chat_message_response_includes_reasoning_when_present() {
        let msg = ChatMessageResponse {
            role: "assistant".to_string(),
            content: Some("42".to_string()),
            reasoning: Some("Let me think...".to_string()),
            reasoning_content: Some("Let me think...".to_string()),
            tool_calls: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["reasoning"], "Let me think...");
        assert_eq!(json["reasoning_content"], "Let me think...");
    }

    #[test]
    fn chat_delta_omits_reasoning_when_none() {
        let delta = ChatDelta {
            role: None,
            content: Some("hello".to_string()),
            reasoning: None,
            reasoning_content: None,
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert!(json.get("reasoning").is_none());
        assert!(json.get("reasoning_content").is_none());
    }

    #[test]
    fn chat_delta_includes_reasoning_when_present() {
        let delta = ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
            reasoning: Some("thinking...".to_string()),
            reasoning_content: Some("thinking...".to_string()),
        };
        let json = serde_json::to_value(&delta).unwrap();
        assert_eq!(json["reasoning"], "thinking...");
        assert_eq!(json["reasoning_content"], "thinking...");
        assert!(json.get("content").is_none());
    }

    #[test]
    fn chat_request_with_reasoning_message() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning": "Simple arithmetic"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(
            req.messages[1].reasoning.as_deref(),
            Some("Simple arithmetic")
        );
    }

    #[test]
    fn chat_request_with_reasoning_content_legacy() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {"role": "assistant", "content": "Yes.", "reasoning_content": "I think so"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.messages[0].reasoning.as_deref(),
            Some("I think so")
        );
    }
}
