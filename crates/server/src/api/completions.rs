use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use vllm_core::engine::GenerationRequest;

use super::error::ApiError;
use super::streaming::completion_sse_stream;
use super::types::{
    finish_reason_str, timestamp_now, CompletionChoice, CompletionRequest, CompletionResponse,
    Usage,
};
use super::AppState;

pub async fn create_completion(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.model != state.model_id {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    let gen_req = GenerationRequest {
        prompt: req.prompt.clone(),
        max_new_tokens: req.max_tokens,
        eos_token_id: state.eos_token_id,
    };

    if req.stream {
        let rx = state
            .engine
            .generate_stream(gen_req)
            .await
            .map_err(|e| ApiError::EngineError(e.to_string()))?;

        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        Ok(completion_sse_stream(request_id, state.model_id.clone(), rx).into_response())
    } else {
        let result = state
            .engine
            .generate(gen_req)
            .await
            .map_err(|e| ApiError::EngineError(e.to_string()))?;

        let prompt_tokens = state
            .tokenizer
            .encode(&req.prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);
        let completion_tokens = result.generated_token_ids.len();

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created: timestamp_now(),
            model: state.model_id.clone(),
            choices: vec![CompletionChoice {
                text: result.generated_text,
                index: 0,
                finish_reason: Some(finish_reason_str(&result.finish_reason)),
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
