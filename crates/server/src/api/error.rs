use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug)]
pub enum ApiError {
    EngineError(String),
    InvalidRequest(String),
    ModelNotFound(String),
    TemplateError(String),
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
    code: Option<&'static str>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, code, message) = match self {
            ApiError::EngineError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None, msg)
            }
            ApiError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", None, msg)
            }
            ApiError::ModelNotFound(msg) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                Some("model_not_found"),
                msg,
            ),
            ApiError::TemplateError(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", None, msg)
            }
        };

        let body = ErrorResponse {
            error: ErrorBody {
                message,
                r#type: error_type,
                code,
            },
        };

        (status, axum::Json(body)).into_response()
    }
}
