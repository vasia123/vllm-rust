use axum::extract::State;
use axum::Json;

use super::error::ApiError;
use super::types::{DetokenizeRequest, DetokenizeResponse, TokenizeRequest, TokenizeResponse};
use super::AppState;

/// Tokenize a text prompt into token IDs.
///
/// POST /v1/tokenize
pub async fn tokenize(
    State(state): State<AppState>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    if req.model != state.model_id {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    let tokens = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|e| ApiError::InternalError(format!("tokenization failed: {}", e)))?;

    let count = tokens.len();
    Ok(Json(TokenizeResponse {
        tokens,
        count,
        max_model_len: state.max_model_len,
    }))
}

/// Detokenize token IDs back into text.
///
/// POST /v1/detokenize
pub async fn detokenize(
    State(state): State<AppState>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ApiError> {
    if req.model != state.model_id {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    let prompt = state
        .tokenizer
        .decode(&req.tokens)
        .map_err(|e| ApiError::InternalError(format!("detokenization failed: {}", e)))?;

    Ok(Json(DetokenizeResponse { prompt }))
}
