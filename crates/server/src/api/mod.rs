pub mod admin;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod middleware;
pub mod models;
pub mod response_format;
pub mod responses;
pub mod responses_types;
pub mod streaming;
pub mod tokenize;
pub mod types;
pub mod validation;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::http::{HeaderName, HeaderValue, Method};
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
use vllm_core::tokenizer::{ChatTemplateEngine, TokenizerWrapper};

pub use admin::restart::{
    AtomicEngineHandle, EngineBuilder, EngineController, ProductionEngineBuilder,
};
pub use admin::{create_admin_router, AdminState};

#[derive(Clone)]
pub struct AppState {
    pub engine: AtomicEngineHandle,
    pub model_id: String,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub chat_template: Option<Arc<ChatTemplateEngine>>,
    pub eos_token_id: u32,
    pub max_model_len: usize,
    /// Whether the server is accepting new requests.
    accepting: Arc<AtomicBool>,
}

impl AppState {
    pub fn new(
        engine: AtomicEngineHandle,
        model_id: String,
        tokenizer: Arc<TokenizerWrapper>,
        chat_template: Option<Arc<ChatTemplateEngine>>,
        eos_token_id: u32,
        max_model_len: usize,
        accepting: Arc<AtomicBool>,
    ) -> Self {
        Self {
            engine,
            model_id,
            tokenizer,
            chat_template,
            eos_token_id,
            max_model_len,
            accepting,
        }
    }

    pub fn accepting_requests(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }
}

/// Configuration for CORS middleware.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Comma-separated allowed origins, or "*" for all.
    pub allowed_origins: String,
    /// Comma-separated allowed methods.
    pub allowed_methods: String,
    /// Comma-separated allowed headers, or "*" for all.
    pub allowed_headers: String,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: "*".to_string(),
            allowed_methods: "GET,POST,OPTIONS".to_string(),
            allowed_headers: "*".to_string(),
        }
    }
}

/// Build a `CorsLayer` from a `CorsConfig`.
///
/// When all three fields use their wildcard defaults ("*" for origins/headers,
/// "GET,POST,OPTIONS" for methods), this returns `CorsLayer::very_permissive()`.
/// Otherwise it parses each field into the corresponding typed values.
pub fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
    if config.allowed_origins == "*"
        && config.allowed_headers == "*"
        && config.allowed_methods == "GET,POST,OPTIONS"
    {
        return CorsLayer::very_permissive();
    }

    let mut layer = CorsLayer::new();

    // Origins
    if config.allowed_origins == "*" {
        layer = layer.allow_origin(AllowOrigin::any());
    } else {
        let origins: Vec<HeaderValue> = config
            .allowed_origins
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return None;
                }
                HeaderValue::from_str(trimmed).ok()
            })
            .collect();
        layer = layer.allow_origin(origins);
    }

    // Methods
    let methods: Vec<Method> = config
        .allowed_methods
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            trimmed.parse::<Method>().ok()
        })
        .collect();
    layer = layer.allow_methods(AllowMethods::list(methods));

    // Headers
    if config.allowed_headers == "*" {
        layer = layer.allow_headers(AllowHeaders::any());
    } else {
        let headers: Vec<HeaderName> = config
            .allowed_headers
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return None;
                }
                trimmed.parse::<HeaderName>().ok()
            })
            .collect();
        layer = layer.allow_headers(AllowHeaders::list(headers));
    }

    layer
}

pub fn create_router(state: AppState) -> Router {
    create_router_with_cors(state, CorsLayer::very_permissive())
}

pub fn create_router_with_cors(state: AppState, cors: CorsLayer) -> Router {
    let accepting = state.accepting.clone();
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::create_completion))
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route("/v1/responses", post(responses::create_response))
        .route("/v1/embeddings", post(embeddings::create_embedding))
        .route("/v1/tokenize", post(tokenize::tokenize))
        .route("/v1/detokenize", post(tokenize::detokenize))
        .layer(axum::middleware::from_fn_with_state(
            accepting,
            middleware::reject_during_restart,
        ))
        .layer(cors)
        .with_state(state)
}

/// Create the full router including admin endpoints.
pub fn create_full_router(app_state: AppState, admin_state: AdminState) -> Router {
    create_full_router_with_cors_and_rate_limit(
        app_state,
        admin_state,
        CorsLayer::very_permissive(),
        middleware::RateLimitState::unlimited(),
    )
}

/// Create the full router with CORS and rate limiting.
pub fn create_full_router_with_cors_and_rate_limit(
    app_state: AppState,
    admin_state: AdminState,
    cors: CorsLayer,
    rate_limit_state: middleware::RateLimitState,
) -> Router {
    let accepting = app_state.accepting.clone();
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::create_completion))
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route("/v1/embeddings", post(embeddings::create_embedding))
        .route("/v1/tokenize", post(tokenize::tokenize))
        .route("/v1/detokenize", post(tokenize::detokenize))
        .layer(axum::middleware::from_fn_with_state(
            rate_limit_state,
            middleware::rate_limit,
        ))
        .layer(axum::middleware::from_fn_with_state(
            accepting,
            middleware::reject_during_restart,
        ))
        .layer(cors)
        .with_state(app_state)
        .nest("/admin", create_admin_router(admin_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use candle_core::{Device, Tensor};
    use tower::ServiceExt;
    use vllm_core::{
        engine::{start_engine, EngineConfig, ModelForward},
        kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager},
        scheduler::SchedulerConfig,
        tokenizer::{ChatTemplateEngine, TokenizerWrapper},
    };

    struct MockModel {
        output_token: u32,
        vocab_size: usize,
        device: Device,
    }

    impl ModelForward for MockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            let mut logits = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits[pos * self.vocab_size + self.output_token as usize] = 100.0;
            }
            Tensor::from_vec(logits, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn test_app_state() -> AppState {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: candle_core::DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel {
            output_token: 42,
            vocab_size: 1000,
            device: Device::Cpu,
        };
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let engine_config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
            cuda_graph_config: vllm_core::engine::CudaGraphConfig::default(),
        };
        let handle = start_engine(model, tokenizer, kv_cache_mgr, engine_config);
        let (atomic_handle, _controller) = AtomicEngineHandle::new(handle);

        let api_tokenizer = TokenizerWrapper::for_testing(1000);
        let chat_template = ChatTemplateEngine::new(
            r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#
                .to_string(),
            "".to_string(),
            "".to_string(),
        );

        let accepting = Arc::new(AtomicBool::new(true));
        // 64 blocks * 16 tokens/block = 1024 max model length
        let max_model_len = 64 * 16;
        AppState::new(
            atomic_handle,
            "test-model".to_string(),
            Arc::new(api_tokenizer),
            Some(Arc::new(chat_template)),
            999,
            max_model_len,
            accepting,
        )
    }

    #[tokio::test]
    async fn models_endpoint() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/v1/models").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "vllm-rust");
        assert!(json["data"][0]["created"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn models_endpoint_full_fields() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/v1/models").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let model = &json["data"][0];

        // root matches id
        assert_eq!(model["root"], "test-model");
        assert_eq!(model["root"], model["id"]);

        // parent is omitted (None with skip_serializing_if)
        assert!(model.get("parent").is_none());

        // permission array
        let perms = model["permission"].as_array().unwrap();
        assert_eq!(perms.len(), 1);
        let perm = &perms[0];
        assert_eq!(perm["object"], "model_permission");
        assert!(perm["id"].as_str().unwrap().starts_with("modelperm-"));
        assert!(perm["allow_sampling"].as_bool().unwrap());
        assert!(perm["allow_logprobs"].as_bool().unwrap());
        assert!(perm["allow_view"].as_bool().unwrap());
        assert!(perm["allow_fine_tuning"].as_bool().unwrap());
        assert!(perm["allow_create_engine"].as_bool().unwrap());
        assert!(perm["allow_search_indices"].as_bool().unwrap());
        assert_eq!(perm["organization"], "*");
        assert!(perm["group"].is_null());
        assert!(!perm["is_blocking"].as_bool().unwrap());
        assert!(perm["created"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn completions_non_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "text_completion");
        assert!(json["choices"][0]["text"].as_str().is_some());
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(json["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completions_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "prompt": "t1 t2",
            "max_tokens": 3
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn chat_completions_non_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn completions_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "stream": true
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream"));

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("data: "));
        assert!(body_str.contains("[DONE]"));
    }

    #[tokio::test]
    async fn chat_completions_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "stream": true
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream"));

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("data: "));
        assert!(body_str.contains("[DONE]"));
    }

    #[tokio::test]
    async fn chat_completions_multi_turn() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test]
    async fn chat_completions_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn completions_batch_prompts() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": ["t1 t2", "t3 t4 t5"],
            "max_tokens": 2
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["choices"].as_array().unwrap().len(), 2);
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][1]["index"], 1);
    }

    #[tokio::test]
    async fn completions_with_sampling_params() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2",
            "max_tokens": 3,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completions_token_ids_input() {
        let state = test_app_state();
        let app = create_router(state);

        // Token IDs as prompt input
        let body = serde_json::json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 2
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn completions_usage_tokens_counted() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3 t4",
            "max_tokens": 5
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
        let total_tokens = json["usage"]["total_tokens"].as_u64().unwrap();

        assert!(prompt_tokens > 0);
        assert_eq!(completion_tokens, 5);
        assert_eq!(total_tokens, prompt_tokens + completion_tokens);
    }

    #[tokio::test]
    async fn completions_response_format() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1",
            "max_tokens": 1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify OpenAI-compatible response format
        assert!(json["id"].as_str().unwrap().starts_with("cmpl-"));
        assert_eq!(json["object"], "text_completion");
        assert!(json["created"].as_u64().is_some());
        assert_eq!(json["model"], "test-model");
        assert!(json["choices"].is_array());
        assert!(json["usage"].is_object());
    }

    #[tokio::test]
    async fn chat_completions_response_format() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify OpenAI-compatible response format
        assert!(json["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(json["object"], "chat.completion");
        assert!(json["created"].as_u64().is_some());
        assert_eq!(json["model"], "test-model");
        assert!(json["choices"].is_array());
        assert!(json["usage"].is_object());
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test]
    async fn chat_completions_with_system_message() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
    }

    #[tokio::test]
    async fn error_response_format() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "nonexistent-model",
            "prompt": "test",
            "max_tokens": 1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // OpenAI-compatible error format
        assert!(json["error"].is_object());
        assert!(json["error"]["message"].is_string());
        assert!(json["error"]["type"].is_string());
    }

    #[tokio::test]
    async fn chat_completions_streaming_with_stream_options_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        // Should contain data chunks and [DONE]
        assert!(body_str.contains("data: "));
        assert!(body_str.contains("[DONE]"));

        // Should contain a usage chunk with prompt_tokens, completion_tokens, total_tokens
        assert!(body_str.contains("\"usage\""));
        assert!(body_str.contains("\"prompt_tokens\""));
        assert!(body_str.contains("\"completion_tokens\""));
        assert!(body_str.contains("\"total_tokens\""));
    }

    #[tokio::test]
    async fn chat_completions_streaming_without_stream_options_no_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "stream": true
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("[DONE]"));
        // Without stream_options, no usage chunk should appear
        // Parse each SSE data line and check none have "usage" key
        let has_usage_chunk = body_str.lines().any(|line| {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    return json.get("usage").is_some();
                }
            }
            false
        });
        assert!(!has_usage_chunk);
    }

    #[tokio::test]
    async fn completions_streaming_with_stream_options_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("[DONE]"));
        assert!(body_str.contains("\"usage\""));
        assert!(body_str.contains("\"prompt_tokens\""));
    }

    #[tokio::test]
    async fn chat_completions_non_streaming_has_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Non-streaming responses must always include usage
        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
        let total_tokens = json["usage"]["total_tokens"].as_u64().unwrap();
        assert!(prompt_tokens > 0);
        assert_eq!(completion_tokens, 3);
        assert_eq!(total_tokens, prompt_tokens + completion_tokens);
    }

    #[tokio::test]
    async fn chat_completions_with_tools() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
            "max_tokens": 3,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "The city name"}
                            },
                            "required": ["city"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify response format is correct
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        // The mock model doesn't produce tool calls, so content should be present
        // In a real scenario with a tool-calling model, tool_calls would be populated
        assert!(json["choices"][0]["message"].is_object());
    }

    // ─── Tokenize / Detokenize integration tests ─────────────────────────

    #[tokio::test]
    async fn tokenize_success() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3"
        });
        let req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let tokens = json["tokens"].as_array().unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(json["count"], 3);
        assert_eq!(json["max_model_len"], 1024);
    }

    #[tokio::test]
    async fn tokenize_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "prompt": "t1 t2"
        });
        let req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn tokenize_empty_prompt() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": ""
        });
        let req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["count"], 0);
        assert!(json["tokens"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn detokenize_success() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // First tokenize to get valid token IDs
        let tok_body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3"
        });
        let tok_req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&tok_body).unwrap()))
            .unwrap();
        let tok_resp = app.oneshot(tok_req).await.unwrap();
        let tok_body = axum::body::to_bytes(tok_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let tok_json: serde_json::Value = serde_json::from_slice(&tok_body).unwrap();
        let tokens: Vec<u32> = tok_json["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        // Now detokenize those IDs
        let app2 = create_router(state);
        let body = serde_json::json!({
            "model": "test-model",
            "tokens": tokens
        });
        let req = Request::post("/v1/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app2.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["prompt"], "t1 t2 t3");
    }

    #[tokio::test]
    async fn detokenize_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "tokens": [1, 2, 3]
        });
        let req = Request::post("/v1/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn detokenize_empty_tokens() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "tokens": []
        });
        let req = Request::post("/v1/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["prompt"].as_str().is_some());
    }

    #[tokio::test]
    async fn completions_invalid_temperature() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 3.0
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("temperature"));
    }

    #[tokio::test]
    async fn chat_completions_invalid_frequency_penalty() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "frequency_penalty": 5.0
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("frequency_penalty"));
    }

    // ─── n > 1 for chat completions ─────────────────────────────────

    #[tokio::test]
    async fn chat_completions_n_greater_than_one() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "n": 2
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[1]["index"], 1);
        assert_eq!(choices[0]["message"]["role"], "assistant");
        assert_eq!(choices[1]["message"]["role"], "assistant");

        // Usage should sum completion tokens across all choices
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
        assert_eq!(completion_tokens, 6); // 3 tokens * 2 choices
    }

    #[tokio::test]
    async fn chat_completions_n_streaming_error() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "n": 2,
            "stream": true
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"].as_str().unwrap().contains("n > 1"));
    }

    // ─── best_of for completions ────────────────────────────────────

    #[tokio::test]
    async fn completions_best_of_returns_single_choice() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "best_of": 3
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // best_of generates multiple candidates but returns only the best one
        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0]["index"], 0);

        // Logprobs should not be present since user didn't request them
        assert!(choices[0]["logprobs"].is_null());
    }

    #[tokio::test]
    async fn completions_best_of_with_logprobs() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "best_of": 2,
            "logprobs": 1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);

        // Logprobs should be present since user requested them
        assert!(choices[0]["logprobs"].is_object());
    }

    // ─── CORS tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn cors_preflight_returns_headers() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::builder()
            .method("OPTIONS")
            .uri("/v1/models")
            .header("origin", "http://example.com")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let headers = resp.headers();
        assert!(headers.get("access-control-allow-origin").is_some());
        assert!(headers.get("access-control-allow-methods").is_some());
    }

    #[tokio::test]
    async fn cors_regular_request_includes_headers() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/v1/models")
            .header("origin", "http://example.com")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing access-control-allow-origin header")
            .to_str()
            .expect("header should be valid UTF-8");
        // very_permissive() mirrors the request origin or returns "*"
        assert!(
            allow_origin == "*" || allow_origin == "http://example.com",
            "expected wildcard or mirrored origin, got: {allow_origin}"
        );
    }

    #[tokio::test]
    async fn cors_custom_origin_restricts_access() {
        let state = test_app_state();
        let cors = build_cors_layer(&CorsConfig {
            allowed_origins: "http://allowed.example.com".to_string(),
            allowed_methods: "GET,POST".to_string(),
            allowed_headers: "content-type".to_string(),
        });
        let app = create_router_with_cors(state, cors);

        // Preflight from an allowed origin
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/v1/models")
            .header("origin", "http://allowed.example.com")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing access-control-allow-origin header");
        assert_eq!(allow_origin, "http://allowed.example.com");
    }

    #[tokio::test]
    async fn cors_default_config_is_very_permissive() {
        let config = CorsConfig::default();
        assert_eq!(config.allowed_origins, "*");
        assert_eq!(config.allowed_methods, "GET,POST,OPTIONS");
        assert_eq!(config.allowed_headers, "*");

        // Should produce a layer that allows any origin
        let state = test_app_state();
        let cors = build_cors_layer(&config);
        let app = create_router_with_cors(state, cors);

        let req = Request::get("/v1/models")
            .header("origin", "http://any-origin.example.com")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("access-control-allow-origin").is_some());
    }

    #[tokio::test]
    async fn cors_build_layer_with_multiple_origins() {
        let config = CorsConfig {
            allowed_origins: "http://a.com, http://b.com".to_string(),
            allowed_methods: "GET,POST".to_string(),
            allowed_headers: "*".to_string(),
        };

        let state = test_app_state();
        let cors = build_cors_layer(&config);
        let app = create_router_with_cors(state, cors);

        // Request from first allowed origin
        let req = Request::get("/v1/models")
            .header("origin", "http://a.com")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing access-control-allow-origin header");
        assert_eq!(allow_origin, "http://a.com");
    }
}
