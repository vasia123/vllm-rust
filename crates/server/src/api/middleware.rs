//! Middleware for request handling during server lifecycle events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Middleware that rejects requests with 503 when the server is not accepting new requests.
/// Used during graceful restart to drain existing connections.
pub async fn reject_during_restart(
    State(accepting): State<Arc<AtomicBool>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !accepting.load(Ordering::SeqCst) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "message": "Server is restarting, please retry",
                    "type": "service_unavailable"
                }
            })),
        )
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    async fn test_handler() -> &'static str {
        "OK"
    }

    #[tokio::test]
    async fn accepts_when_accepting_is_true() {
        let accepting = Arc::new(AtomicBool::new(true));
        let app = Router::new().route("/test", get(test_handler)).layer(
            axum::middleware::from_fn_with_state(accepting.clone(), reject_during_restart),
        );

        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rejects_when_accepting_is_false() {
        let accepting = Arc::new(AtomicBool::new(false));
        let app = Router::new().route("/test", get(test_handler)).layer(
            axum::middleware::from_fn_with_state(accepting.clone(), reject_during_restart),
        );

        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("restarting"));
    }
}
