pub mod admin;
pub mod chat;
pub mod completions;
pub mod error;
pub mod middleware;
pub mod models;
pub mod streaming;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
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
        accepting: Arc<AtomicBool>,
    ) -> Self {
        Self {
            engine,
            model_id,
            tokenizer,
            chat_template,
            eos_token_id,
            accepting,
        }
    }

    pub fn accepting_requests(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }
}

pub fn create_router(state: AppState) -> Router {
    let accepting = state.accepting.clone();
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::create_completion))
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .layer(axum::middleware::from_fn_with_state(
            accepting,
            middleware::reject_during_restart,
        ))
        .with_state(state)
}

/// Create the full router including admin endpoints.
pub fn create_full_router(app_state: AppState, admin_state: AdminState) -> Router {
    let accepting = app_state.accepting.clone();
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route("/v1/completions", post(completions::create_completion))
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .layer(axum::middleware::from_fn_with_state(
            accepting,
            middleware::reject_during_restart,
        ))
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
        kv_cache::{config::CacheConfig, BlockTable, KVCacheManager},
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
            _kv_cache_mgr: &KVCacheManager,
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
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
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
        AppState::new(
            atomic_handle,
            "test-model".to_string(),
            Arc::new(api_tokenizer),
            Some(Arc::new(chat_template)),
            999,
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
}
