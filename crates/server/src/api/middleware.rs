//! Middleware for request handling during server lifecycle events.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use governor::{Quota, RateLimiter};
use serde_json::json;
use tracing::{error, info, warn};

/// Rate limiter type using in-memory state.
pub type GlobalRateLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

/// Create a global rate limiter with the specified requests per second.
/// Returns None if rps is 0 (no limit).
pub fn create_rate_limiter(rps: u32) -> Option<Arc<GlobalRateLimiter>> {
    if rps == 0 {
        return None;
    }
    let quota = Quota::per_second(NonZeroU32::new(rps)?);
    Some(Arc::new(RateLimiter::direct(quota)))
}

/// State for rate limiting middleware.
#[derive(Clone)]
pub struct RateLimitState {
    /// Rate limiter (None = no limit).
    pub limiter: Option<Arc<GlobalRateLimiter>>,
    /// Current queue depth counter.
    pub queue_depth: Arc<AtomicUsize>,
    /// Maximum queue depth (0 = no limit).
    pub max_queue_depth: usize,
}

impl RateLimitState {
    pub fn new(rps: u32, max_queue_depth: usize) -> Self {
        Self {
            limiter: create_rate_limiter(rps),
            queue_depth: Arc::new(AtomicUsize::new(0)),
            max_queue_depth,
        }
    }

    /// Create a no-op rate limit state (no limits).
    pub fn unlimited() -> Self {
        Self {
            limiter: None,
            queue_depth: Arc::new(AtomicUsize::new(0)),
            max_queue_depth: 0,
        }
    }
}

/// Middleware that applies rate limiting and queue depth limits.
pub async fn rate_limit(
    State(state): State<RateLimitState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Check rate limit
    if let Some(ref limiter) = state.limiter {
        if limiter.check().is_err() {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": {
                        "message": "Rate limit exceeded, please slow down",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit"
                    }
                })),
            )
                .into_response();
        }
    }

    // Check queue depth
    if state.max_queue_depth > 0 {
        let current = state.queue_depth.fetch_add(1, Ordering::SeqCst);
        if current >= state.max_queue_depth {
            state.queue_depth.fetch_sub(1, Ordering::SeqCst);
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": {
                        "message": "Server is at capacity, please retry later",
                        "type": "server_overloaded",
                        "code": "capacity_exceeded"
                    }
                })),
            )
                .into_response();
        }
    }

    let response = next.run(request).await;

    // Decrement queue depth after request completes
    if state.max_queue_depth > 0 {
        state.queue_depth.fetch_sub(1, Ordering::SeqCst);
    }

    response
}

/// A request ID extracted from or generated for an incoming request.
///
/// Stored in request extensions so downstream handlers can access it
/// via `request.extensions().get::<RequestId>()`.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Middleware that assigns a unique request ID to every request.
///
/// If the client sends an `X-Request-Id` header, that value is reused.
/// Otherwise a new UUID v4 is generated. The ID is:
/// - inserted into request extensions (accessible by handlers)
/// - echoed back in the `X-Request-Id` response header
pub async fn request_id(mut request: Request<Body>, next: Next) -> Response {
    let id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    request.extensions_mut().insert(RequestId(id.clone()));

    let mut response = next.run(request).await;

    if let Ok(header_value) = id.parse() {
        response.headers_mut().insert("x-request-id", header_value);
    }

    response
}

/// Middleware that logs HTTP request/response details using `tracing`.
///
/// Logs include: method, path, status code, latency, and request ID.
/// Log level is selected by response status:
/// - 2xx/3xx: INFO
/// - 4xx: WARN
/// - 5xx: ERROR
pub async fn http_logging(request: Request<Body>, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let response = next.run(request).await;

    let latency = start.elapsed();
    let status = response.status().as_u16();

    // Extract request ID set by the request_id middleware (which runs
    // as an outer layer, so it executes before this one).
    let req_id = response
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");

    if status >= 500 {
        error!(
            method = %method,
            path = %path,
            status = status,
            latency_ms = latency.as_millis() as u64,
            request_id = %req_id,
            "HTTP request completed"
        );
    } else if status >= 400 {
        warn!(
            method = %method,
            path = %path,
            status = status,
            latency_ms = latency.as_millis() as u64,
            request_id = %req_id,
            "HTTP request completed"
        );
    } else {
        info!(
            method = %method,
            path = %path,
            status = status,
            latency_ms = latency.as_millis() as u64,
            request_id = %req_id,
            "HTTP request completed"
        );
    }

    response
}

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

/// State for API key authentication middleware.
#[derive(Clone)]
pub struct ApiKeyState {
    /// The expected API key. None means no authentication required.
    pub api_key: Option<Arc<String>>,
}

impl ApiKeyState {
    /// Create an API key state with a required key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key: Some(Arc::new(api_key)),
        }
    }

    /// Create an API key state with no authentication.
    pub fn disabled() -> Self {
        Self { api_key: None }
    }
}

/// Paths that are excluded from API key authentication.
const AUTH_EXEMPT_PATHS: &[&str] = &["/health", "/version", "/metrics"];

/// Middleware that checks for a valid API key in the `Authorization: Bearer <key>` header.
///
/// Excluded paths (health, version, metrics) are always allowed without auth.
/// Returns 401 Unauthorized if the key is missing or invalid.
pub async fn api_key_auth(
    State(state): State<ApiKeyState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let expected_key = match &state.api_key {
        Some(key) => key,
        None => return next.run(request).await, // No auth configured
    };

    let path = request.uri().path();
    if AUTH_EXEMPT_PATHS.contains(&path) {
        return next.run(request).await;
    }

    let auth_header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let provided_key = &header[7..];
            if provided_key == expected_key.as_str() {
                next.run(request).await
            } else {
                unauthorized_response("Invalid API key")
            }
        }
        Some(_) => unauthorized_response("Authorization header must use Bearer scheme"),
        None => unauthorized_response("Missing Authorization header"),
    }
}

fn unauthorized_response(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "message": message,
                "type": "authentication_error",
                "code": "invalid_api_key"
            }
        })),
    )
        .into_response()
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

    // Helper: build a router with request_id + http_logging middleware.
    fn app_with_request_id() -> Router {
        Router::new()
            .route("/test", get(test_handler))
            .layer(axum::middleware::from_fn(http_logging))
            .layer(axum::middleware::from_fn(request_id))
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

    #[tokio::test]
    async fn rate_limit_allows_requests_under_limit() {
        let state = RateLimitState::new(10, 0); // 10 RPS, no queue limit
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(axum::middleware::from_fn_with_state(state, rate_limit));

        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rate_limit_rejects_when_exceeded() {
        let state = RateLimitState::new(1, 0); // 1 RPS
        let app = Router::new().route("/test", get(test_handler)).layer(
            axum::middleware::from_fn_with_state(state.clone(), rate_limit),
        );

        // First request should succeed
        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request immediately after should be rate limited
        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn queue_depth_rejects_when_exceeded() {
        let state = RateLimitState::new(0, 1); // No rate limit, queue depth 1

        // Manually set queue depth to max
        state.queue_depth.store(1, Ordering::SeqCst);

        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(axum::middleware::from_fn_with_state(state, rate_limit));

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
            .contains("capacity"));
    }

    #[tokio::test]
    async fn unlimited_allows_all_requests() {
        let state = RateLimitState::unlimited();
        let app = Router::new().route("/test", get(test_handler)).layer(
            axum::middleware::from_fn_with_state(state.clone(), rate_limit),
        );

        // Multiple requests should all succeed
        for _ in 0..10 {
            let req = Request::get("/test").body(Body::empty()).unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    // ─── Request ID tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn request_id_generated_when_absent() {
        let app = app_with_request_id();

        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let id = resp
            .headers()
            .get("x-request-id")
            .expect("response must contain x-request-id header")
            .to_str()
            .expect("header value must be valid UTF-8");

        // The generated ID should be a valid UUID v4
        assert!(
            uuid::Uuid::parse_str(id).is_ok(),
            "generated request ID '{}' is not a valid UUID",
            id
        );
    }

    #[tokio::test]
    async fn request_id_echoed_when_provided() {
        let app = app_with_request_id();

        let client_id = "my-custom-request-id-12345";
        let req = Request::get("/test")
            .header("x-request-id", client_id)
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let echoed = resp
            .headers()
            .get("x-request-id")
            .expect("response must contain x-request-id header")
            .to_str()
            .expect("header value must be valid UTF-8");

        assert_eq!(echoed, client_id);
    }

    // ─── API key auth tests ───────────────────────────────────────────

    fn app_with_api_key(key: &str) -> Router {
        let state = ApiKeyState::new(key.to_string());
        Router::new()
            .route("/test", get(test_handler))
            .route("/health", get(test_handler))
            .route("/version", get(test_handler))
            .route("/metrics", get(test_handler))
            .layer(axum::middleware::from_fn_with_state(state, api_key_auth))
    }

    #[tokio::test]
    async fn api_key_auth_passes_with_valid_key() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/test")
            .header("authorization", "Bearer secret123")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_key_auth_rejects_invalid_key() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/test")
            .header("authorization", "Bearer wrong-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn api_key_auth_rejects_missing_header() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Missing"));
    }

    #[tokio::test]
    async fn api_key_auth_rejects_non_bearer_scheme() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/test")
            .header("authorization", "Basic c2VjcmV0MTIz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn api_key_auth_exempts_health() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/health").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_key_auth_exempts_version() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/version").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_key_auth_exempts_metrics() {
        let app = app_with_api_key("secret123");
        let req = Request::get("/metrics").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_key_auth_disabled_allows_all() {
        let state = ApiKeyState::disabled();
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(axum::middleware::from_fn_with_state(state, api_key_auth));
        let req = Request::get("/test").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn request_id_is_unique_across_requests() {
        let app = app_with_request_id();

        let req1 = Request::get("/test").body(Body::empty()).unwrap();
        let resp1 = app.clone().oneshot(req1).await.unwrap();
        let id1 = resp1
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let req2 = Request::get("/test").body(Body::empty()).unwrap();
        let resp2 = app.oneshot(req2).await.unwrap();
        let id2 = resp2
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        assert_ne!(id1, id2, "each request must get a distinct ID");
    }
}
