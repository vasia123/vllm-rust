use std::collections::HashMap;

use axum::extract::State;
use axum::Json;
use serde::Serialize;

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

/// Tokenizer metadata response, equivalent to tokenizer_config.json.
#[derive(Debug, Serialize)]
pub struct TokenizerInfoResponse {
    pub tokenizer_class: &'static str,
    pub max_chars_per_token: usize,
    pub model_max_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Get tokenizer configuration metadata.
///
/// GET /tokenizer_info
pub async fn get_tokenizer_info(State(state): State<AppState>) -> Json<TokenizerInfoResponse> {
    let chat_template = state
        .chat_template
        .as_ref()
        .map(|ct| ct.raw_template().to_string());

    Json(TokenizerInfoResponse {
        tokenizer_class: "PreTrainedTokenizerFast",
        max_chars_per_token: state.tokenizer.max_chars_per_token(),
        model_max_length: state.max_model_len,
        chat_template,
        extra: HashMap::new(),
    })
}
