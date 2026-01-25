use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use vllm_core::engine::GenerationRequest;
use vllm_core::sampling::SamplingParams;
use vllm_core::tool_parser::{HermesToolParser, ToolCallParser};

use super::error::ApiError;
use super::streaming::chat_completion_sse_stream;
use super::types::{
    finish_reason_str, timestamp_now, ChatCompletionChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessageResponse, Usage,
};
use super::AppState;

pub async fn create_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.model != state.model_id {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    let chat_template = state
        .chat_template
        .as_ref()
        .ok_or_else(|| ApiError::TemplateError("no chat template available".to_string()))?;

    // Apply template with tools if provided
    let prompt = chat_template
        .apply_with_tools(&req.messages, req.tools.as_deref(), true)
        .map_err(|e| ApiError::TemplateError(e.to_string()))?;

    // Determine if we should parse tool calls from the output
    let has_tools = req.tools.is_some();

    let gen_req = GenerationRequest {
        prompt: prompt.clone(),
        max_new_tokens: req.max_tokens,
        eos_token_id: state.eos_token_id,
        sampling_params: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repetition_penalty: req.repetition_penalty,
            min_p: req.min_p,
            seed: req.seed,
        },
        stop_strings: req.stop,
        stop_token_ids: req.stop_token_ids,
        include_stop_str_in_output: req.include_stop_str_in_output,
        logprobs: None, // Chat completions don't support logprobs in this implementation
        echo: false,
    };

    if req.stream {
        let rx = state
            .engine
            .get()
            .generate_stream(gen_req)
            .await
            .map_err(|e| ApiError::EngineError(e.to_string()))?;

        let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        Ok(chat_completion_sse_stream(request_id, state.model_id.clone(), rx).into_response())
    } else {
        let result = state
            .engine
            .get()
            .generate(gen_req)
            .await
            .map_err(|e| ApiError::EngineError(e.to_string()))?;

        let prompt_tokens = state
            .tokenizer
            .encode(&prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);
        let completion_tokens = result.generated_token_ids.len();

        // Parse tool calls if tools were provided
        let (content, tool_calls, finish_reason) = if has_tools {
            let parser = HermesToolParser::new();
            if let Ok(calls) = parser.parse(&result.generated_text) {
                if !calls.is_empty() {
                    // Extract non-tool-call content
                    let content = parser.extract_content(&result.generated_text);
                    (content, Some(calls), "tool_calls".to_string())
                } else {
                    (Some(result.generated_text), None, finish_reason_str(&result.finish_reason))
                }
            } else {
                (Some(result.generated_text), None, finish_reason_str(&result.finish_reason))
            }
        } else {
            (Some(result.generated_text), None, finish_reason_str(&result.finish_reason))
        };

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion",
            created: timestamp_now(),
            model: state.model_id.clone(),
            choices: vec![ChatCompletionChoice {
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                },
                index: 0,
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}
