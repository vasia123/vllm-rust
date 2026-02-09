//! Handler for the OpenAI Responses API (`POST /v1/responses`).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use vllm_core::engine::{GenerationRequest, StreamEvent};
use vllm_core::lora::LoraRequest;
use vllm_core::request::FinishReason;
use vllm_core::sampling::SamplingParams;
use vllm_core::tokenizer::ChatMessage;
use vllm_core::tool_parser::ToolCallParser;

use super::admin::prometheus;
use super::error::ApiError;
use super::response_format::inject_json_system_prompt;
use super::responses_types::*;
use super::types::{convert_logit_bias, timestamp_now};
use super::AppState;

pub async fn create_response(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let start_time = Instant::now();
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

    // Convert input to chat messages
    let mut messages = input_to_messages(req.input, req.instructions.as_deref());

    // Inject system prompt for JSON response format enforcement
    inject_json_system_prompt(&mut messages, req.response_format.as_ref());

    // Apply template with tools
    let prompt = chat_template
        .apply_with_tools(&messages, req.tools.as_deref(), true)
        .map_err(|e| ApiError::TemplateError(e.to_string()))?;

    // Early-fail: reject prompts that are clearly too long
    super::validation::validate_prompt_char_length(
        prompt.len(),
        req.max_output_tokens,
        state.max_model_len,
        state.tokenizer.max_chars_per_token(),
    )?;

    let has_tools = req.tools.is_some();
    let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);
    let logit_bias = convert_logit_bias(&req.logit_bias);

    // Create constraint from response_format
    let constraint = super::chat::create_constraint_from_response_format(
        req.response_format.as_ref(),
        &state.tokenizer,
    );

    let response_id = generate_response_id();
    let model = state.model_id.clone();
    let created_at = timestamp_now();

    if req.stream {
        let gen_req = GenerationRequest {
            prompt: prompt.clone(),
            max_new_tokens: req.max_output_tokens,
            eos_token_id: state.eos_token_id,
            sampling_params: SamplingParams {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                repetition_penalty: req.repetition_penalty,
                frequency_penalty: req.frequency_penalty,
                presence_penalty: req.presence_penalty,
                min_p: 0.0,
                seed: req.seed,
                beam_search: None,
                logit_bias,
                min_tokens: 0,
                eos_token_id: Some(state.eos_token_id),
                banned_token_ids: None,
                allowed_token_ids: None,
            },
            stop_strings: req.stop,
            stop_token_ids: Vec::new(),
            include_stop_str_in_output: false,
            logprobs: None,
            echo: false,
            lora_request,
            constraint,
            image_inputs: Vec::new(),
        };

        let include_usage = req
            .stream_options
            .as_ref()
            .is_some_and(|opts| opts.include_usage);
        let prompt_tokens = if include_usage {
            state
                .tokenizer
                .encode(&prompt)
                .map(|ids| ids.len())
                .unwrap_or(0)
        } else {
            0
        };

        let rx = state
            .engine
            .get()
            .generate_stream(gen_req)
            .await
            .map_err(|e| {
                prometheus::inc_requests_error();
                ApiError::EngineError(e.to_string())
            })?;

        let metadata = req.metadata;
        Ok(responses_sse_stream(
            response_id,
            model,
            created_at,
            rx,
            include_usage,
            prompt_tokens,
            metadata,
        )
        .into_response())
    } else {
        let prompt_tokens = state
            .tokenizer
            .encode(&prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);

        let gen_req = GenerationRequest {
            prompt,
            max_new_tokens: req.max_output_tokens,
            eos_token_id: state.eos_token_id,
            sampling_params: SamplingParams {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                repetition_penalty: req.repetition_penalty,
                frequency_penalty: req.frequency_penalty,
                presence_penalty: req.presence_penalty,
                min_p: 0.0,
                seed: req.seed,
                beam_search: None,
                logit_bias,
                min_tokens: 0,
                eos_token_id: Some(state.eos_token_id),
                banned_token_ids: None,
                allowed_token_ids: None,
            },
            stop_strings: req.stop,
            stop_token_ids: Vec::new(),
            include_stop_str_in_output: false,
            logprobs: None,
            echo: false,
            lora_request,
            constraint,
            image_inputs: Vec::new(),
        };

        let result = state
            .engine
            .get()
            .generate(gen_req)
            .await
            .map_err(|e| {
                prometheus::inc_requests_error();
                ApiError::EngineError(e.to_string())
            })?;

        let completion_tokens = result.generated_token_ids.len();

        // Determine status and incomplete details
        let (status, incomplete_details) = match result.finish_reason {
            FinishReason::Length => (
                ResponseStatus::Incomplete,
                Some(IncompleteDetails {
                    reason: "max_output_tokens".to_string(),
                }),
            ),
            _ => (ResponseStatus::Completed, None),
        };

        // Build output items: parse tool calls if tools were provided
        let output = build_output_items(
            &result.generated_text,
            &result.finish_reason,
            has_tools,
            state.tool_call_parser.as_ref(),
        );

        let response = ResponsesResponse {
            id: response_id,
            object: "response",
            created_at,
            model,
            status,
            output,
            usage: Some(ResponseUsage {
                input_tokens: prompt_tokens,
                output_tokens: completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            incomplete_details,
            metadata: req.metadata,
        };

        // Record metrics
        let elapsed = start_time.elapsed().as_secs_f64();
        prometheus::observe_e2e_latency(elapsed);
        if completion_tokens > 0 && elapsed > 0.0 {
            prometheus::observe_tps(completion_tokens as f64 / elapsed);
        }
        prometheus::inc_requests_success();

        Ok(Json(response).into_response())
    }
}

/// Convert `ResponseInput` into a list of `ChatMessage` for template application.
fn input_to_messages(input: ResponseInput, instructions: Option<&str>) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // Inject instructions as a system message if provided
    if let Some(instructions) = instructions {
        messages.push(ChatMessage::new("system", instructions));
    }

    match input {
        ResponseInput::Text(text) => {
            messages.push(ChatMessage::new("user", &text));
        }
        ResponseInput::Messages(msgs) => {
            messages.extend(msgs);
        }
    }

    messages
}

/// Build output items from generated text, optionally parsing tool calls.
fn build_output_items(
    generated_text: &str,
    finish_reason: &FinishReason,
    has_tools: bool,
    parser: &dyn ToolCallParser,
) -> Vec<ResponseOutputItem> {
    let mut items = Vec::new();

    if has_tools {
        if let Ok(calls) = parser.parse(generated_text) {
            if !calls.is_empty() {
                // Emit text content before tool calls, if any
                if let Some(content_text) = parser.extract_content(generated_text) {
                    items.push(ResponseOutputItem::Message(ResponseOutputMessage {
                        id: generate_item_id(),
                        status: ResponseStatus::Completed,
                        role: "assistant".to_string(),
                        content: vec![ResponseContentPart::OutputText(ResponseOutputText {
                            text: content_text,
                        })],
                    }));
                }

                // Emit each tool call as a separate output item
                for call in calls {
                    items.push(ResponseOutputItem::FunctionCall(ResponseFunctionCall {
                        id: generate_item_id(),
                        status: ResponseStatus::Completed,
                        call_id: call.id,
                        name: call.function.name,
                        arguments: call.function.arguments,
                    }));
                }

                return items;
            }
        }
    }

    // Default: single message output
    let status = match finish_reason {
        FinishReason::Length => ResponseStatus::Incomplete,
        _ => ResponseStatus::Completed,
    };

    items.push(ResponseOutputItem::Message(ResponseOutputMessage {
        id: generate_item_id(),
        status,
        role: "assistant".to_string(),
        content: vec![ResponseContentPart::OutputText(ResponseOutputText {
            text: generated_text.to_string(),
        })],
    }));

    items
}

/// Build an SSE stream for the Responses API.
fn responses_sse_stream(
    response_id: String,
    model: String,
    created_at: u64,
    rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    include_usage: bool,
    prompt_tokens: usize,
    metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let builder = StreamingResponseBuilder {
        id: response_id.into(),
        model: model.into(),
        created_at,
        metadata,
    };

    let output_stream = async_stream::stream! {
        let item_id = generate_item_id();
        let mut completion_tokens: usize = 0;
        let mut full_text = String::new();
        let mut finish_reason_val = FinishReason::Eos;

        // Emit response.created
        let initial_response = builder.build(ResponseStatus::InProgress, None, None);
        yield Ok::<_, Infallible>(sse_event("response.created", &initial_response));

        // Emit output_item.added (message item)
        let msg_item = ResponseOutputItem::Message(ResponseOutputMessage {
            id: item_id.clone(),
            status: ResponseStatus::InProgress,
            role: "assistant".to_string(),
            content: vec![],
        });
        let item_added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": msg_item,
        });
        yield Ok(Event::default().data(serde_json::to_string(&item_added).unwrap_or_default()));

        // Emit content_part.added
        let part = ResponseContentPart::OutputText(ResponseOutputText {
            text: String::new(),
        });
        let part_added = serde_json::json!({
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": part,
        });
        yield Ok(Event::default().data(serde_json::to_string(&part_added).unwrap_or_default()));

        let mut rx_stream = ReceiverStream::new(rx);

        while let Some(event) = rx_stream.next().await {
            match event {
                StreamEvent::Token { token_text, .. } => {
                    completion_tokens += 1;
                    full_text.push_str(&token_text);

                    let delta = serde_json::json!({
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": token_text,
                    });
                    yield Ok(Event::default().data(serde_json::to_string(&delta).unwrap_or_default()));
                }
                StreamEvent::Done { finish_reason, .. } => {
                    finish_reason_val = finish_reason;
                }
                StreamEvent::Error { error } => {
                    yield Ok(Event::default().data(format!("{{\"error\":\"{error}\"}}")));
                }
            }
        }

        // Emit text.done
        let text_done = serde_json::json!({
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        });
        yield Ok(Event::default().data(serde_json::to_string(&text_done).unwrap_or_default()));

        // Emit content_part.done
        let part_done_val = ResponseContentPart::OutputText(ResponseOutputText {
            text: full_text.clone(),
        });
        let part_done = serde_json::json!({
            "type": "response.content_part.done",
            "output_index": 0,
            "content_index": 0,
            "part": part_done_val,
        });
        yield Ok(Event::default().data(serde_json::to_string(&part_done).unwrap_or_default()));

        // Emit output_item.done
        let final_item = ResponseOutputItem::Message(ResponseOutputMessage {
            id: item_id,
            status: ResponseStatus::Completed,
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText(ResponseOutputText {
                text: full_text,
            })],
        });
        let item_done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": final_item,
        });
        yield Ok(Event::default().data(serde_json::to_string(&item_done).unwrap_or_default()));

        // Emit response.completed
        let (status, incomplete_details) = match finish_reason_val {
            FinishReason::Length => (
                ResponseStatus::Incomplete,
                Some(IncompleteDetails {
                    reason: "max_output_tokens".to_string(),
                }),
            ),
            _ => (ResponseStatus::Completed, None),
        };

        let usage = if include_usage {
            Some(ResponseUsage {
                input_tokens: prompt_tokens,
                output_tokens: completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            })
        } else {
            None
        };

        // NOTE: We don't include full output in the streaming completed event to
        // avoid duplicating large payloads. Clients should accumulate from deltas.
        let final_response = builder.build(status, usage, incomplete_details);
        yield Ok(sse_event("response.completed", &final_response));

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(output_stream).keep_alive(KeepAlive::default())
}

/// Shared fields for building streaming response lifecycle events.
struct StreamingResponseBuilder {
    id: Arc<str>,
    model: Arc<str>,
    created_at: u64,
    metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl StreamingResponseBuilder {
    fn build(
        &self,
        status: ResponseStatus,
        usage: Option<ResponseUsage>,
        incomplete_details: Option<IncompleteDetails>,
    ) -> serde_json::Value {
        let resp = ResponsesResponse {
            id: self.id.to_string(),
            object: "response",
            created_at: self.created_at,
            model: self.model.to_string(),
            status,
            output: vec![],
            usage,
            incomplete_details,
            metadata: self.metadata.clone(),
        };
        serde_json::to_value(&resp).unwrap_or_default()
    }
}

/// Create an SSE event with a type field.
fn sse_event(event_type: &str, payload: &serde_json::Value) -> Event {
    // Insert the type field into the payload for clients that parse data only
    let mut data = payload.clone();
    if let Some(obj) = data.as_object_mut() {
        obj.insert(
            "type".to_string(),
            serde_json::Value::String(event_type.to_string()),
        );
    }
    Event::default()
        .event(event_type)
        .data(serde_json::to_string(&data).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_to_messages_text_input() {
        let messages = input_to_messages(ResponseInput::Text("Hello!".to_string()), None);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].text(), "Hello!");
    }

    #[test]
    fn input_to_messages_text_with_instructions() {
        let messages = input_to_messages(
            ResponseInput::Text("Hello!".to_string()),
            Some("Be helpful."),
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].text(), "Be helpful.");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn input_to_messages_structured_input() {
        let input_msgs = vec![
            ChatMessage::new("user", "What is 2+2?"),
            ChatMessage::new("assistant", "4"),
            ChatMessage::new("user", "And 3+3?"),
        ];
        let messages = input_to_messages(ResponseInput::Messages(input_msgs), None);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn build_output_simple_text() {
        let parser = vllm_core::tool_parser::HermesToolParser::new();
        let items = build_output_items("Hello, world!", &FinishReason::Eos, false, &parser);
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::Message(msg) => {
                assert_eq!(msg.status, ResponseStatus::Completed);
                assert_eq!(msg.content.len(), 1);
                match &msg.content[0] {
                    ResponseContentPart::OutputText(t) => {
                        assert_eq!(t.text, "Hello, world!");
                    }
                }
            }
            _ => panic!("expected Message output"),
        }
    }

    #[test]
    fn build_output_length_truncated() {
        let parser = vllm_core::tool_parser::HermesToolParser::new();
        let items = build_output_items("partial output", &FinishReason::Length, false, &parser);
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::Message(msg) => {
                assert_eq!(msg.status, ResponseStatus::Incomplete);
            }
            _ => panic!("expected Message output"),
        }
    }

    #[test]
    fn build_output_with_tool_calls() {
        let parser = vllm_core::tool_parser::HermesToolParser::new();
        let text = r#"Let me check. <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#;
        let items = build_output_items(text, &FinishReason::Eos, true, &parser);
        // Should have: message (content before tool call) + function call
        assert_eq!(items.len(), 2);
        match &items[0] {
            ResponseOutputItem::Message(msg) => {
                assert!(msg.content[0].text().contains("Let me check."));
            }
            _ => panic!("expected Message first"),
        }
        match &items[1] {
            ResponseOutputItem::FunctionCall(fc) => {
                assert_eq!(fc.name, "get_weather");
            }
            _ => panic!("expected FunctionCall second"),
        }
    }

    #[test]
    fn build_output_no_tool_calls_when_tools_false() {
        let parser = vllm_core::tool_parser::HermesToolParser::new();
        let text = r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#;
        let items = build_output_items(text, &FinishReason::Eos, false, &parser);
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::Message(msg) => {
                assert!(msg.content[0].text().contains("<tool_call>"));
            }
            _ => panic!("expected Message"),
        }
    }
}

// Helper trait to extract text from content parts in tests
#[cfg(test)]
impl ResponseContentPart {
    fn text(&self) -> &str {
        match self {
            ResponseContentPart::OutputText(t) => &t.text,
        }
    }
}
