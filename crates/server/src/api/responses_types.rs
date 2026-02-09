//! Types for the OpenAI Responses API (`POST /v1/responses`).
//!
//! The Responses API is OpenAI's newer conversation-centric API that provides
//! richer output structure (message items, reasoning, tool calls) and finer
//! streaming granularity than the legacy chat completions endpoint.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use vllm_core::tokenizer::ChatMessage;
pub use vllm_core::tool_parser::{ToolCall, ToolDefinition};

use super::types::{ResponseFormat, StreamOptions, ToolChoice, Usage};

// ─── Request ─────────────────────────────────────────────────────────────

/// Input to the Responses API: either a plain string or a list of
/// structured conversation items.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// Simple text prompt (wrapped as a user message internally).
    Text(String),
    /// Structured conversation items (messages, tool results, etc.).
    Messages(Vec<ChatMessage>),
}

/// `POST /v1/responses` request body.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    /// Model identifier (must match the loaded model).
    pub model: String,
    /// The input prompt or conversation history.
    pub input: ResponseInput,
    /// System-level instructions (alternative to a system message in `input`).
    #[serde(default)]
    pub instructions: Option<String>,
    /// Maximum number of output tokens to generate.
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Nucleus sampling probability mass.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k sampling (0 = disabled).
    #[serde(default)]
    pub top_k: u32,
    /// Repetition penalty (1.0 = none).
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Frequency penalty (OpenAI convention). Range: -2.0 to 2.0.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty (OpenAI convention). Range: -2.0 to 2.0.
    #[serde(default)]
    pub presence_penalty: f32,
    /// Random seed for reproducibility.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stop sequences.
    #[serde(default)]
    pub stop: Vec<String>,
    /// Whether to stream the response via SSE.
    #[serde(default)]
    pub stream: bool,
    /// Options for streaming (e.g., include_usage).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Available tools for the model.
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls how the model uses tools.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Response format for structured output (JSON schema, etc.).
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Token logit bias: map of token ID (as string) to bias value.
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Name of the LoRA adapter to use for this request.
    #[serde(default)]
    pub lora_name: Option<String>,
    /// A unique identifier representing the end-user.
    #[serde(default)]
    pub user: Option<String>,
    /// Custom metadata for tracking.
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

// ─── Response ────────────────────────────────────────────────────────────

/// Status of the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Generation is complete.
    Completed,
    /// Generation was stopped early (e.g., max tokens).
    Incomplete,
    /// Generation is in progress (used in streaming).
    InProgress,
    /// Generation failed.
    Failed,
}

/// `POST /v1/responses` response body.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesResponse {
    /// Unique response identifier (format: `resp_<uuid>`).
    pub id: String,
    /// Object type, always `"response"`.
    pub object: &'static str,
    /// Unix timestamp of creation.
    pub created_at: u64,
    /// Model identifier.
    pub model: String,
    /// Response status.
    pub status: ResponseStatus,
    /// Output items (messages, reasoning, tool calls).
    pub output: Vec<ResponseOutputItem>,
    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    /// Why the response is incomplete (if status = incomplete).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    /// User-provided metadata echoed back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Details about why a response is incomplete.
#[derive(Debug, Clone, Serialize)]
pub struct IncompleteDetails {
    /// Reason for incompleteness (e.g., "max_output_tokens").
    pub reason: String,
}

// ─── Output items ────────────────────────────────────────────────────────

/// A single output item in the response.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    /// Text message output.
    Message(ResponseOutputMessage),
    /// Function/tool call output.
    FunctionCall(ResponseFunctionCall),
}

/// A text message output item.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputMessage {
    /// Unique item identifier.
    pub id: String,
    /// Item status.
    pub status: ResponseStatus,
    /// Role (always "assistant" for output).
    pub role: String,
    /// Content parts.
    pub content: Vec<ResponseContentPart>,
}

/// A content part within a message.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentPart {
    /// Text content.
    OutputText(ResponseOutputText),
}

/// Text content within a message.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputText {
    /// The generated text.
    pub text: String,
}

/// A function/tool call output item.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseFunctionCall {
    /// Unique item identifier.
    pub id: String,
    /// Item status.
    pub status: ResponseStatus,
    /// Unique call identifier (for matching with tool results).
    pub call_id: String,
    /// Function name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
}

// ─── Usage ───────────────────────────────────────────────────────────────

/// Token usage statistics for the Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseUsage {
    /// Number of input (prompt) tokens.
    pub input_tokens: usize,
    /// Number of output (completion) tokens.
    pub output_tokens: usize,
    /// Total tokens (input + output).
    pub total_tokens: usize,
}

impl From<Usage> for ResponseUsage {
    fn from(u: Usage) -> Self {
        Self {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

// ─── Streaming events ────────────────────────────────────────────────────

/// SSE event types for streaming responses.
///
/// Each variant serializes with a `type` field used as the SSE `event:` name.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseStreamEvent {
    /// Initial response object (status = in_progress).
    #[serde(rename = "response.created")]
    ResponseCreated {
        response: ResponsesResponse,
    },
    /// Response is now in progress.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        response: ResponsesResponse,
    },
    /// A new output item was added.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: ResponseOutputItem,
    },
    /// A new content part was added to an item.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        output_index: usize,
        content_index: usize,
        part: ResponseContentPart,
    },
    /// Incremental text delta.
    #[serde(rename = "response.output_text.delta")]
    TextDelta {
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    /// Text content part is done.
    #[serde(rename = "response.output_text.done")]
    TextDone {
        output_index: usize,
        content_index: usize,
        text: String,
    },
    /// Content part is done.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        output_index: usize,
        content_index: usize,
        part: ResponseContentPart,
    },
    /// Output item is done.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: ResponseOutputItem,
    },
    /// Final completed response.
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        response: ResponsesResponse,
    },
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn default_max_output_tokens() -> usize {
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

/// Generate a unique response ID.
pub fn generate_response_id() -> String {
    let uuid_str = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("resp_{}", &uuid_str[..24])
}

/// Generate a unique item ID.
pub fn generate_item_id() -> String {
    let uuid_str = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("item_{}", &uuid_str[..24])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ResponseStatus::Completed).unwrap(),
            "\"completed\""
        );
        assert_eq!(
            serde_json::to_string(&ResponseStatus::Incomplete).unwrap(),
            "\"incomplete\""
        );
        assert_eq!(
            serde_json::to_string(&ResponseStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&ResponseStatus::Failed).unwrap(),
            "\"failed\""
        );
    }

    #[test]
    fn response_id_format() {
        let id = generate_response_id();
        assert!(id.starts_with("resp_"), "id = {id}");
        assert_eq!(id.len(), 29); // "resp_" + 24 chars
    }

    #[test]
    fn item_id_format() {
        let id = generate_item_id();
        assert!(id.starts_with("item_"), "id = {id}");
        assert_eq!(id.len(), 29);
    }

    #[test]
    fn responses_request_text_input_deserializes() {
        let json = r#"{
            "model": "test-model",
            "input": "Hello, world!"
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test-model");
        assert!(matches!(req.input, ResponseInput::Text(ref s) if s == "Hello, world!"));
    }

    #[test]
    fn responses_request_messages_input_deserializes() {
        let json = r#"{
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, ResponseInput::Messages(ref msgs) if msgs.len() == 1));
    }

    #[test]
    fn responses_request_defaults() {
        let json = r#"{
            "model": "test-model",
            "input": "test"
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_output_tokens, 64);
        assert!((req.temperature - 1.0).abs() < 1e-6);
        assert!((req.top_p - 1.0).abs() < 1e-6);
        assert_eq!(req.top_k, 0);
        assert!(!req.stream);
        assert!(req.tools.is_none());
        assert!(req.instructions.is_none());
        assert!(req.metadata.is_none());
    }

    #[test]
    fn responses_request_with_all_fields() {
        let json = r#"{
            "model": "test-model",
            "input": "test",
            "instructions": "Be concise.",
            "max_output_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "stream": true,
            "seed": 42,
            "stop": ["END"],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "metadata": {"session": "abc"}
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_output_tokens, 256);
        assert!((req.temperature - 0.7).abs() < 1e-6);
        assert!(req.stream);
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.stop, vec!["END"]);
        assert!(req.tools.is_some());
        assert_eq!(req.instructions.as_deref(), Some("Be concise."));
        assert!(req.metadata.is_some());
    }

    #[test]
    fn responses_response_serialization() {
        let resp = ResponsesResponse {
            id: "resp_test123".to_string(),
            object: "response",
            created_at: 1234567890,
            model: "test-model".to_string(),
            status: ResponseStatus::Completed,
            output: vec![ResponseOutputItem::Message(ResponseOutputMessage {
                id: "item_test123".to_string(),
                status: ResponseStatus::Completed,
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText(ResponseOutputText {
                    text: "Hello!".to_string(),
                })],
            })],
            usage: Some(ResponseUsage {
                input_tokens: 5,
                output_tokens: 1,
                total_tokens: 6,
            }),
            incomplete_details: None,
            metadata: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "resp_test123");
        assert_eq!(json["object"], "response");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["output"][0]["type"], "message");
        assert_eq!(json["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(json["output"][0]["content"][0]["text"], "Hello!");
        assert_eq!(json["usage"]["input_tokens"], 5);
    }

    #[test]
    fn incomplete_response_includes_details() {
        let resp = ResponsesResponse {
            id: "resp_test".to_string(),
            object: "response",
            created_at: 0,
            model: "m".to_string(),
            status: ResponseStatus::Incomplete,
            output: vec![],
            usage: None,
            incomplete_details: Some(IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            }),
            metadata: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "incomplete");
        assert_eq!(json["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn function_call_output_item_serialization() {
        let item = ResponseOutputItem::FunctionCall(ResponseFunctionCall {
            id: "item_fc1".to_string(),
            status: ResponseStatus::Completed,
            call_id: "call_abc123".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"city":"NYC"}"#.to_string(),
        });
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "function_call");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["call_id"], "call_abc123");
    }

    #[test]
    fn response_usage_from_usage() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let resp_usage = ResponseUsage::from(usage);
        assert_eq!(resp_usage.input_tokens, 10);
        assert_eq!(resp_usage.output_tokens, 20);
        assert_eq!(resp_usage.total_tokens, 30);
    }

    #[test]
    fn stream_event_response_created_serialization() {
        let event = ResponseStreamEvent::ResponseCreated {
            response: ResponsesResponse {
                id: "resp_1".to_string(),
                object: "response",
                created_at: 0,
                model: "m".to_string(),
                status: ResponseStatus::InProgress,
                output: vec![],
                usage: None,
                incomplete_details: None,
                metadata: None,
            },
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "response.created");
        assert_eq!(json["response"]["status"], "in_progress");
    }

    #[test]
    fn stream_event_text_delta_serialization() {
        let event = ResponseStreamEvent::TextDelta {
            output_index: 0,
            content_index: 0,
            delta: "Hello".to_string(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "response.output_text.delta");
        assert_eq!(json["delta"], "Hello");
        assert_eq!(json["output_index"], 0);
        assert_eq!(json["content_index"], 0);
    }

    #[test]
    fn stream_event_completed_serialization() {
        let event = ResponseStreamEvent::ResponseCompleted {
            response: ResponsesResponse {
                id: "resp_1".to_string(),
                object: "response",
                created_at: 0,
                model: "m".to_string(),
                status: ResponseStatus::Completed,
                output: vec![],
                usage: Some(ResponseUsage {
                    input_tokens: 5,
                    output_tokens: 10,
                    total_tokens: 15,
                }),
                incomplete_details: None,
                metadata: None,
            },
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "response.completed");
        assert_eq!(json["response"]["status"], "completed");
        assert_eq!(json["response"]["usage"]["total_tokens"], 15);
    }

    #[test]
    fn response_metadata_echoed_when_present() {
        let mut metadata = HashMap::new();
        metadata.insert("session".to_string(), serde_json::json!("abc"));

        let resp = ResponsesResponse {
            id: "resp_1".to_string(),
            object: "response",
            created_at: 0,
            model: "m".to_string(),
            status: ResponseStatus::Completed,
            output: vec![],
            usage: None,
            incomplete_details: None,
            metadata: Some(metadata),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["metadata"]["session"], "abc");
    }

    #[test]
    fn response_metadata_omitted_when_none() {
        let resp = ResponsesResponse {
            id: "resp_1".to_string(),
            object: "response",
            created_at: 0,
            model: "m".to_string(),
            status: ResponseStatus::Completed,
            output: vec![],
            usage: None,
            incomplete_details: None,
            metadata: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json.get("metadata").is_none());
    }
}
