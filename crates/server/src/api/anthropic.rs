/// Anthropic Messages API — `/v1/messages`
///
/// Provides compatibility with the Anthropic Messages API format.
/// Incoming requests are converted to the internal representation,
/// forwarded to the engine, and responses are translated back to
/// Anthropic format.
///
/// Reference: <https://docs.anthropic.com/en/api/messages>
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use vllm_core::engine::{GenerationRequest, StreamEvent};
use vllm_core::sampling::SamplingParams;
use vllm_core::tokenizer::{ChatMessage, MessageContent};
use vllm_core::tool_parser::{
    FunctionDefinition, ToolChoice, ToolChoiceAuto, ToolChoiceFunction, ToolChoiceSpecific,
    ToolDefinition,
};

use super::admin::prometheus;
use super::error::ApiError;
use super::streaming::AbortHandle;
use super::types::finish_reason_str;
use super::AppState;

// ─── Protocol types ────────────────────────────────────────────────────────

/// A single content block in an Anthropic message.
///
/// Uses a flat struct with optional fields rather than a tagged union, matching
/// the Anthropic API wire format where all block types share the same JSON object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    /// Text content (type = "text").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Image source (type = "image").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
    /// Tool call / tool result ID (type = "tool_use" or "tool_result").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool name (type = "tool_use").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool input arguments (type = "tool_use").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    /// Tool result content (type = "tool_result"). Can be text or nested blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    /// Whether the tool result is an error (type = "tool_result").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl AnthropicContentBlock {
    fn text(text: impl Into<String>) -> Self {
        Self {
            block_type: "text".to_string(),
            text: Some(text.into()),
            source: None,
            id: None,
            name: None,
            input: None,
            content: None,
            is_error: None,
        }
    }

    fn tool_use(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            block_type: "tool_use".to_string(),
            text: None,
            source: None,
            id: Some(id.into()),
            name: Some(name.into()),
            input: Some(input),
            content: None,
            is_error: None,
        }
    }
}

/// Message content — either a plain string or a list of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// A single message in the Anthropic conversation.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

/// Tool definition following the Anthropic API schema.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Tool choice configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    /// Tool name — only used when `type = "tool"`.
    #[serde(default)]
    pub name: Option<String>,
}

/// System prompt — either a plain string or a list of text content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic Messages API request body.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

/// Token usage in an Anthropic response.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Non-streaming Anthropic Messages API response.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub role: &'static str,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
}

// ─── Request conversion ─────────────────────────────────────────────────────

type ConvertedRequest = (
    Vec<ChatMessage>,
    Option<Vec<ToolDefinition>>,
    Option<ToolChoice>,
);

/// Convert an `AnthropicMessagesRequest` into internal types ready for engine
/// submission.
///
/// Returns `(messages, tools, tool_choice)`.
fn convert_request(req: &AnthropicMessagesRequest) -> Result<ConvertedRequest, ApiError> {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Prepend system message when provided.
    if let Some(ref sys) = req.system {
        let text = match sys {
            AnthropicSystem::Text(t) => t.clone(),
            AnthropicSystem::Blocks(blocks) => blocks
                .iter()
                .filter(|b| b.block_type == "text")
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        };
        if !text.is_empty() {
            messages.push(ChatMessage::new("system", text));
        }
    }

    // Convert each Anthropic message to one or more internal ChatMessages.
    for msg in &req.messages {
        match &msg.content {
            AnthropicMessageContent::Text(text) => {
                messages.push(ChatMessage::new(&msg.role, text));
            }
            AnthropicMessageContent::Blocks(blocks) => {
                convert_blocks_to_messages(&msg.role, blocks, &mut messages);
            }
        }
    }

    // Convert tool definitions.
    let tools = req.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|t| {
                let mut schema = t.input_schema.clone();
                // Ensure input_schema has "type" field as required by JSON Schema.
                if let serde_json::Value::Object(ref mut map) = schema {
                    map.entry("type")
                        .or_insert_with(|| serde_json::Value::String("object".to_string()));
                }
                ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: Some(schema),
                    },
                }
            })
            .collect()
    });

    // Convert tool_choice.
    let tool_choice = match &req.tool_choice {
        None => None,
        Some(tc) => match tc.choice_type.as_str() {
            "auto" => Some(ToolChoice::Auto(ToolChoiceAuto::Auto)),
            // "any" = required — force the model to use at least one tool.
            "any" => Some(ToolChoice::Auto(ToolChoiceAuto::Required)),
            "tool" => {
                let name = tc.name.as_deref().unwrap_or("").to_string();
                Some(ToolChoice::Specific(ToolChoiceSpecific {
                    tool_type: "function".to_string(),
                    function: ToolChoiceFunction { name },
                }))
            }
            _ => None,
        },
    };

    Ok((messages, tools, tool_choice))
}

/// Convert content blocks from a single Anthropic message into internal
/// `ChatMessage`s.
///
/// `tool_result` blocks in a user message are emitted as separate `tool` role
/// messages (OpenAI convention) so the chat template can correlate them with
/// prior tool calls.
fn convert_blocks_to_messages(
    role: &str,
    blocks: &[AnthropicContentBlock],
    out: &mut Vec<ChatMessage>,
) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls_json: Vec<serde_json::Value> = Vec::new();

    for block in blocks {
        match block.block_type.as_str() {
            "text" => {
                if let Some(ref t) = block.text {
                    text_parts.push(t.clone());
                }
            }
            "image" => {
                // Images in Anthropic format have a `source` field. Represent as
                // a placeholder for now; full multimodal support requires
                // extracting data URIs and building ImageData.
                if let Some(ref src) = block.source {
                    let url = src.get("url").and_then(|v| v.as_str()).unwrap_or("<image>");
                    text_parts.push(format!("[image: {url}]"));
                }
            }
            "tool_use" => {
                // Collect tool calls to attach to the assistant message.
                let id = block.id.clone().unwrap_or_default();
                let name = block.name.clone().unwrap_or_default();
                let args = block
                    .input
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".to_string());
                tool_calls_json.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": args }
                }));
            }
            "tool_result" => {
                // tool_result in a user message → emit as a "tool" role message.
                // Flush any pending text/calls before inserting the tool message.
                let tool_call_id = block.id.clone().unwrap_or_default();
                let content = match &block.content {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                flush_pending(role, &mut text_parts, &mut tool_calls_json, out);
                out.push(ChatMessage {
                    role: "tool".to_string(),
                    content: MessageContent::Text(format!(
                        "[tool_call_id: {tool_call_id}]\n{content}"
                    )),
                    reasoning: None,
                });
            }
            _ => {}
        }
    }

    flush_pending(role, &mut text_parts, &mut tool_calls_json, out);
}

/// Flush accumulated text and tool-call content as a single `ChatMessage`.
fn flush_pending(
    role: &str,
    text_parts: &mut Vec<String>,
    tool_calls_json: &mut Vec<serde_json::Value>,
    out: &mut Vec<ChatMessage>,
) {
    if text_parts.is_empty() && tool_calls_json.is_empty() {
        return;
    }

    let text = if tool_calls_json.is_empty() {
        text_parts.join("")
    } else {
        // Embed tool calls as JSON so the chat template can render them. Most
        // templates handle this via role="assistant" + tool_calls content.
        let calls_str = serde_json::to_string(&tool_calls_json).unwrap_or_default();
        let prefix = text_parts.join("");
        if prefix.is_empty() {
            calls_str
        } else {
            format!("{prefix}\n{calls_str}")
        }
    };

    out.push(ChatMessage {
        role: role.to_string(),
        content: MessageContent::Text(text),
        reasoning: None,
    });

    text_parts.clear();
    tool_calls_json.clear();
}

// ─── Response conversion ────────────────────────────────────────────────────

/// Map a vLLM finish reason string to an Anthropic stop reason.
fn map_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

/// Generate a unique Anthropic-style message ID.
fn msg_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("msg_{ts:x}")
}

// ─── Handler ────────────────────────────────────────────────────────────────

/// POST /v1/messages — Anthropic Messages API.
pub async fn create_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Result<impl IntoResponse, ApiError> {
    prometheus::inc_requests_total();

    if req.model != state.model_id {
        prometheus::inc_requests_error();
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    let chat_template = state.chat_template.as_ref().ok_or_else(|| {
        prometheus::inc_requests_error();
        ApiError::TemplateError("no chat template available".to_string())
    })?;

    let (messages, tools, tool_choice) = convert_request(&req)?;

    let prompt = chat_template
        .apply_with_tools(&messages, tools.as_deref(), true)
        .map_err(|e| {
            prometheus::inc_requests_error();
            ApiError::TemplateError(e.to_string())
        })?;

    super::validation::validate_prompt_char_length(
        prompt.len(),
        req.max_tokens,
        state.max_model_len,
        state.tokenizer.max_chars_per_token(),
    )?;

    let should_parse_tools = tool_choice.is_some() || tools.is_some();

    let gen_req = GenerationRequest {
        prompt: prompt.clone(),
        max_new_tokens: req.max_tokens,
        eos_token_id: state.eos_token_id,
        sampling_params: SamplingParams {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(0),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_p: 0.0,
            seed: None,
            beam_search: None,
            logit_bias: None,
            min_tokens: 0,
            eos_token_id: Some(state.eos_token_id),
            banned_token_ids: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            typical_p: 1.0,
        },
        stop_strings: req.stop_sequences.clone().unwrap_or_default(),
        stop_token_ids: Vec::new(),
        include_stop_str_in_output: false,
        ignore_eos: false,
        logprobs: None,
        echo: false,
        lora_request: None,
        prompt_adapter_request: None,
        constraint: None,
        image_inputs: Vec::new(),
        audio_inputs: Vec::new(),
        skip_prefix_cache: false,
    };

    if req.stream {
        let prompt_tokens = state
            .tokenizer
            .encode(&prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);

        let engine = state.engine.get();
        let (engine_request_id, rx) = engine.generate_stream(gen_req).await.map_err(|e| {
            prometheus::inc_requests_error();
            ApiError::EngineError(e.to_string())
        })?;

        let request_id = msg_id();
        let model = state.model_id.clone();
        let abort_handle = AbortHandle::new(engine, engine_request_id);

        Ok(
            anthropic_sse_stream(request_id, model, prompt_tokens, rx, abort_handle)
                .into_response(),
        )
    } else {
        let prompt_tokens = state
            .tokenizer
            .encode(&prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);

        let result = state.engine.get().generate(gen_req).await.map_err(|e| {
            prometheus::inc_requests_error();
            ApiError::EngineError(e.to_string())
        })?;

        let completion_tokens = result.generated_token_ids.len();

        // Parse tool calls from the response if tools were provided.
        let (content_text, tool_calls) = if should_parse_tools {
            let parser = &state.tool_call_parser;
            match parser.parse(&result.generated_text) {
                Ok(calls) if !calls.is_empty() => {
                    let text = parser.extract_content(&result.generated_text);
                    (text, Some(calls))
                }
                _ => (Some(result.generated_text.clone()), None),
            }
        } else {
            (Some(result.generated_text.clone()), None)
        };

        let stop_reason = map_stop_reason(&finish_reason_str(&result.finish_reason));

        let mut content: Vec<AnthropicContentBlock> = Vec::new();
        if let Some(text) = content_text {
            if !text.is_empty() {
                content.push(AnthropicContentBlock::text(text));
            }
        }
        if let Some(calls) = tool_calls {
            for call in calls {
                let input = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                content.push(AnthropicContentBlock::tool_use(
                    call.id,
                    call.function.name,
                    input,
                ));
            }
        }

        let response = AnthropicMessagesResponse {
            id: msg_id(),
            response_type: "message",
            role: "assistant",
            content,
            model: state.model_id.clone(),
            stop_reason: Some(stop_reason),
            stop_sequence: None,
            usage: Some(AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens: completion_tokens,
            }),
        };

        prometheus::inc_requests_success();
        Ok(Json(response).into_response())
    }
}

// ─── Streaming ──────────────────────────────────────────────────────────────

/// Build an Anthropic-format SSE stream from engine `StreamEvent`s.
///
/// The Anthropic streaming protocol requires both `event:` and `data:` SSE
/// fields. Axum's `Event::default().event(name).data(json)` produces this
/// format automatically.
fn anthropic_sse_stream(
    request_id: String,
    model: String,
    prompt_tokens: usize,
    rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    abort_handle: AbortHandle,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let output_stream = async_stream::stream! {
        let mut _abort_guard = Some(super::streaming::AbortGuard::new(abort_handle));
        let mut rx_stream = ReceiverStream::new(rx);
        let mut completion_tokens: usize = 0;
        let mut content_block_started = false;
        let content_block_index: usize = 0;

        // message_start — announce the message with zero output tokens.
        let message_start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": &request_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": &model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": 0
                }
            }
        });
        yield Ok::<_, Infallible>(
            Event::default()
                .event("message_start")
                .data(message_start.to_string()),
        );

        while let Some(event) = rx_stream.next().await {
            match event {
                StreamEvent::Token { token_text, .. } => {
                    completion_tokens += 1;

                    if !content_block_started {
                        // Open the first text content block.
                        let block_start = serde_json::json!({
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": { "type": "text", "text": "" }
                        });
                        yield Ok(
                            Event::default()
                                .event("content_block_start")
                                .data(block_start.to_string()),
                        );
                        content_block_started = true;
                    }

                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": { "type": "text_delta", "text": token_text }
                    });
                    yield Ok(
                        Event::default()
                            .event("content_block_delta")
                            .data(delta.to_string()),
                    );
                }
                StreamEvent::Done { finish_reason, .. } => {
                    if let Some(ref mut guard) = _abort_guard {
                        guard.defuse();
                    }

                    if content_block_started {
                        let block_stop = serde_json::json!({
                            "type": "content_block_stop",
                            "index": content_block_index
                        });
                        yield Ok(
                            Event::default()
                                .event("content_block_stop")
                                .data(block_stop.to_string()),
                        );
                    }

                    let finish_str = finish_reason_str(&finish_reason);
                    let anthropic_stop = map_stop_reason(&finish_str);

                    let message_delta = serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": anthropic_stop,
                            "stop_sequence": null
                        },
                        "usage": { "output_tokens": completion_tokens }
                    });
                    yield Ok(
                        Event::default()
                            .event("message_delta")
                            .data(message_delta.to_string()),
                    );

                    let message_stop = serde_json::json!({ "type": "message_stop" });
                    yield Ok(
                        Event::default()
                            .event("message_stop")
                            .data(message_stop.to_string()),
                    );
                }
                StreamEvent::Error { error } => {
                    let error_event = serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "internal_error",
                            "message": error
                        }
                    });
                    yield Ok(
                        Event::default()
                            .event("error")
                            .data(error_event.to_string()),
                    );
                }
            }
        }
    };

    Sse::new(output_stream).keep_alive(KeepAlive::default())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn text_block(text: &str) -> AnthropicContentBlock {
        AnthropicContentBlock {
            block_type: "text".to_string(),
            text: Some(text.to_string()),
            source: None,
            id: None,
            name: None,
            input: None,
            content: None,
            is_error: None,
        }
    }

    fn tool_result_block(id: &str, content: &str) -> AnthropicContentBlock {
        AnthropicContentBlock {
            block_type: "tool_result".to_string(),
            text: None,
            source: None,
            id: Some(id.to_string()),
            name: None,
            input: None,
            content: Some(serde_json::Value::String(content.to_string())),
            is_error: None,
        }
    }

    fn make_request(
        system: Option<&str>,
        messages: Vec<AnthropicMessage>,
    ) -> AnthropicMessagesRequest {
        AnthropicMessagesRequest {
            model: "test-model".to_string(),
            messages,
            max_tokens: 256,
            metadata: None,
            stop_sequences: None,
            stream: false,
            system: system.map(|s| AnthropicSystem::Text(s.to_string())),
            temperature: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
        }
    }

    #[test]
    fn test_simple_text_message_conversion() {
        let req = make_request(
            Some("You are helpful."),
            vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text("Hello!".to_string()),
            }],
        );

        let (messages, tools, tool_choice) = convert_request(&req).unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content.as_text(), "You are helpful.");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_text(), "Hello!");
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
    }

    #[test]
    fn test_system_as_blocks() {
        let req = AnthropicMessagesRequest {
            model: "m".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text("Hi".to_string()),
            }],
            max_tokens: 10,
            metadata: None,
            stop_sequences: None,
            stream: false,
            system: Some(AnthropicSystem::Blocks(vec![
                text_block("Block one. "),
                text_block("Block two."),
            ])),
            temperature: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
        };

        let (messages, _, _) = convert_request(&req).unwrap();

        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content.as_text(), "Block one. Block two.");
    }

    #[test]
    fn test_tool_definition_conversion() {
        let mut req = make_request(
            None,
            vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text("Call get_weather".to_string()),
            }],
        );
        req.tools = Some(vec![AnthropicTool {
            name: "get_weather".to_string(),
            description: Some("Get current weather".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
        }]);
        req.tool_choice = Some(AnthropicToolChoice {
            choice_type: "auto".to_string(),
            name: None,
        });

        let (_, tools, tool_choice) = convert_request(&req).unwrap();

        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(
            tools[0].function.description.as_deref(),
            Some("Get current weather")
        );
        assert!(matches!(
            tool_choice,
            Some(ToolChoice::Auto(ToolChoiceAuto::Auto))
        ));
    }

    #[test]
    fn test_tool_result_emits_tool_role_message() {
        let req = make_request(
            None,
            vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Text("Use get_weather".to_string()),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_result_block(
                        "call_abc", "28°C",
                    )]),
                },
            ],
        );

        let (messages, _, _) = convert_request(&req).unwrap();

        // user + (tool_result block → tool role message)
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, "tool");
        assert!(messages[1].content.as_text().contains("28°C"));
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("tool_calls"), "tool_use");
        assert_eq!(map_stop_reason("unknown"), "end_turn");
    }
}
