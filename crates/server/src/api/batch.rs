//! Batch processing API — OpenAI-compatible batch endpoint.
//!
//! Supports asynchronous batch inference: submit a list of requests,
//! poll for completion, and retrieve results as JSONL.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::error::ApiError;
use super::AppState;
use vllm_core::engine::GenerationRequest;

// ─── Types ──────────────────────────────────────────────────────────

/// Status of a batch job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Counts of requests within a batch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestCounts {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

/// A single request within a batch JSONL input.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchRequestItem {
    /// Caller-provided correlation ID (echoed in output).
    pub custom_id: String,
    /// HTTP method — always POST for inference.
    #[serde(default = "default_method")]
    pub method: String,
    /// Endpoint path (e.g. "/v1/completions").
    #[serde(default = "default_url")]
    pub url: String,
    /// The completion request body.
    pub body: BatchRequestBody,
}

fn default_method() -> String {
    "POST".to_string()
}
fn default_url() -> String {
    "/v1/completions".to_string()
}

/// Simplified request body for batch items.
#[derive(Debug, Clone, Deserialize)]
pub struct BatchRequestBody {
    pub model: Option<String>,
    pub prompt: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

fn default_max_tokens() -> usize {
    64
}

/// A single result in the batch output JSONL.
#[derive(Debug, Clone, Serialize)]
pub struct BatchResultItem {
    pub custom_id: String,
    pub response: Option<BatchResponseBody>,
    pub error: Option<BatchError>,
}

/// Successful response body for a batch item.
#[derive(Debug, Clone, Serialize)]
pub struct BatchResponseBody {
    pub status_code: u16,
    pub body: serde_json::Value,
}

/// Error body for a failed batch item.
#[derive(Debug, Clone, Serialize)]
pub struct BatchError {
    pub code: String,
    pub message: String,
}

/// In-memory batch job state.
#[derive(Debug, Clone)]
pub struct BatchJob {
    pub id: String,
    pub status: BatchStatus,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub request_counts: RequestCounts,
    pub results: Vec<BatchResultItem>,
    /// Set to true to signal cancellation to the processing task.
    pub cancel_requested: bool,
}

/// Create batch request.
#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    /// JSONL string containing batch request items.
    pub input: String,
    /// Endpoint for all requests (e.g. "/v1/completions").
    #[serde(default = "default_url")]
    pub endpoint: String,
    /// Completion window (ignored, always "24h").
    #[serde(default = "default_completion_window")]
    pub completion_window: String,
}

fn default_completion_window() -> String {
    "24h".to_string()
}

/// Batch job response (matches OpenAI format).
#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub id: String,
    pub object: &'static str,
    pub endpoint: String,
    pub status: BatchStatus,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub request_counts: RequestCounts,
}

/// Shared batch store type.
pub type BatchStore = Arc<RwLock<HashMap<String, BatchJob>>>;

/// Create a new empty batch store.
pub fn new_batch_store() -> BatchStore {
    Arc::new(RwLock::new(HashMap::new()))
}

// ─── Handlers ───────────────────────────────────────────────────────

/// POST /v1/batches — Submit a batch job.
pub async fn create_batch(
    State(state): State<AppState>,
    Json(req): Json<CreateBatchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Parse JSONL input
    let items: Vec<BatchRequestItem> = req
        .input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|e| {
                ApiError::InvalidRequest(format!("Invalid JSON at line {}: {}", i + 1, e))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if items.is_empty() {
        return Err(ApiError::InvalidRequest(
            "Batch input must contain at least one request".to_string(),
        ));
    }

    let total_items = items.len();
    let batch_id = format!("batch_{}", uuid::Uuid::new_v4().as_simple());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = BatchJob {
        id: batch_id.clone(),
        status: BatchStatus::InProgress,
        created_at: now,
        completed_at: None,
        request_counts: RequestCounts {
            total: total_items,
            completed: 0,
            failed: 0,
        },
        results: Vec::with_capacity(total_items),
        cancel_requested: false,
    };

    // Store job
    {
        let mut store = state.batch_store.write().await;
        store.insert(batch_id.clone(), job);
    }

    // Spawn processing task
    let batch_store = state.batch_store.clone();
    let engine = state.engine.clone();
    let eos_token_id = state.eos_token_id;
    let spawned_batch_id = batch_id.clone();

    tokio::spawn(async move {
        process_batch(spawned_batch_id, items, engine, eos_token_id, batch_store).await;
    });

    let response = BatchResponse {
        id: batch_id,
        object: "batch",
        endpoint: req.endpoint,
        status: BatchStatus::InProgress,
        created_at: now,
        completed_at: None,
        request_counts: RequestCounts {
            total: total_items,
            completed: 0,
            failed: 0,
        },
    };

    Ok(Json(response))
}

/// GET /v1/batches/{batch_id} — Poll batch status.
pub async fn get_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let store = state.batch_store.read().await;
    let job = store
        .get(&batch_id)
        .ok_or_else(|| ApiError::InvalidRequest(format!("Batch '{}' not found", batch_id)))?;

    let response = BatchResponse {
        id: job.id.clone(),
        object: "batch",
        endpoint: "/v1/completions".to_string(),
        status: job.status,
        created_at: job.created_at,
        completed_at: job.completed_at,
        request_counts: job.request_counts.clone(),
    };

    Ok(Json(response))
}

/// GET /v1/batches/{batch_id}/output — Retrieve batch results as JSONL.
pub async fn get_batch_output(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let store = state.batch_store.read().await;
    let job = store
        .get(&batch_id)
        .ok_or_else(|| ApiError::InvalidRequest(format!("Batch '{}' not found", batch_id)))?;

    if job.status != BatchStatus::Completed
        && job.status != BatchStatus::Failed
        && job.status != BatchStatus::Cancelled
    {
        return Err(ApiError::InvalidRequest(format!(
            "Batch '{}' is still {:?}. Wait for completion.",
            batch_id, job.status
        )));
    }

    let mut output = String::new();
    for result in &job.results {
        let line = serde_json::to_string(result)
            .map_err(|e| ApiError::InternalError(format!("Failed to serialize result: {}", e)))?;
        output.push_str(&line);
        output.push('\n');
    }

    Ok(output.into_response())
}

/// POST /v1/batches/{batch_id}/cancel — Cancel a batch job.
pub async fn cancel_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let mut store = state.batch_store.write().await;
    let job = store
        .get_mut(&batch_id)
        .ok_or_else(|| ApiError::InvalidRequest(format!("Batch '{}' not found", batch_id)))?;

    if job.status != BatchStatus::InProgress {
        return Err(ApiError::InvalidRequest(format!(
            "Batch '{}' is {:?}, cannot cancel",
            batch_id, job.status
        )));
    }

    job.cancel_requested = true;

    Ok(Json(BatchResponse {
        id: job.id.clone(),
        object: "batch",
        endpoint: "/v1/completions".to_string(),
        status: job.status,
        created_at: job.created_at,
        completed_at: job.completed_at,
        request_counts: job.request_counts.clone(),
    }))
}

// ─── Processing ─────────────────────────────────────────────────────

/// Process batch items sequentially through the engine.
async fn process_batch(
    batch_id: String,
    items: Vec<BatchRequestItem>,
    engine: super::AtomicEngineHandle,
    eos_token_id: u32,
    store: BatchStore,
) {
    for item in &items {
        // Check for cancellation
        {
            let guard = store.read().await;
            if let Some(job) = guard.get(&batch_id) {
                if job.cancel_requested {
                    break;
                }
            }
        }

        let gen_req = GenerationRequest {
            prompt: item.body.prompt.clone().unwrap_or_default(),
            max_new_tokens: item.body.max_tokens,
            eos_token_id,
            sampling_params: {
                let mut params = vllm_core::sampling::SamplingParams::default();
                if let Some(temp) = item.body.temperature {
                    params.temperature = temp;
                }
                if let Some(tp) = item.body.top_p {
                    params.top_p = tp;
                }
                params
            },
            ..Default::default()
        };

        let result = engine.get().generate(gen_req).await;

        let batch_result = match result {
            Ok(gen) => {
                let body = serde_json::json!({
                    "id": format!("cmpl-{}", uuid::Uuid::new_v4().as_simple()),
                    "object": "text_completion",
                    "model": item.body.model.as_deref().unwrap_or("default"),
                    "choices": [{
                        "text": gen.generated_text,
                        "index": 0,
                        "finish_reason": match gen.finish_reason {
                            vllm_core::request::FinishReason::Eos => "stop",
                            vllm_core::request::FinishReason::Length => "length",
                            vllm_core::request::FinishReason::Stop => "stop",
                        },
                    }],
                    "usage": {
                        "prompt_tokens": gen.prompt_token_ids.as_ref().map(|t| t.len()).unwrap_or(0),
                        "completion_tokens": gen.generated_token_ids.len(),
                        "total_tokens": gen.prompt_token_ids.as_ref().map(|t| t.len()).unwrap_or(0) + gen.generated_token_ids.len(),
                    }
                });

                BatchResultItem {
                    custom_id: item.custom_id.clone(),
                    response: Some(BatchResponseBody {
                        status_code: 200,
                        body,
                    }),
                    error: None,
                }
            }
            Err(e) => BatchResultItem {
                custom_id: item.custom_id.clone(),
                response: None,
                error: Some(BatchError {
                    code: "engine_error".to_string(),
                    message: e.to_string(),
                }),
            },
        };

        // Update job state
        let is_error = batch_result.error.is_some();
        {
            let mut guard = store.write().await;
            if let Some(job) = guard.get_mut(&batch_id) {
                job.results.push(batch_result);
                if is_error {
                    job.request_counts.failed += 1;
                } else {
                    job.request_counts.completed += 1;
                }
            }
        }
    }

    // Finalize job status
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut guard = store.write().await;
    if let Some(job) = guard.get_mut(&batch_id) {
        job.completed_at = Some(now);
        if job.cancel_requested {
            job.status = BatchStatus::Cancelled;
        } else if job.request_counts.failed == job.request_counts.total {
            job.status = BatchStatus::Failed;
        } else {
            job.status = BatchStatus::Completed;
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_batch_request_item() {
        let json = r#"{"custom_id":"req-1","body":{"prompt":"Hello","max_tokens":10}}"#;
        let item: BatchRequestItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.custom_id, "req-1");
        assert_eq!(item.body.prompt.as_deref(), Some("Hello"));
        assert_eq!(item.body.max_tokens, 10);
        assert_eq!(item.method, "POST");
        assert_eq!(item.url, "/v1/completions");
    }

    #[test]
    fn parse_batch_request_with_method() {
        let json = r#"{"custom_id":"req-2","method":"POST","url":"/v1/chat/completions","body":{"prompt":"Hi","max_tokens":5}}"#;
        let item: BatchRequestItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.custom_id, "req-2");
        assert_eq!(item.method, "POST");
        assert_eq!(item.url, "/v1/chat/completions");
    }

    #[test]
    fn serialize_batch_result_success() {
        let result = BatchResultItem {
            custom_id: "req-1".to_string(),
            response: Some(BatchResponseBody {
                status_code: 200,
                body: serde_json::json!({"text": "world"}),
            }),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("req-1"));
        assert!(json.contains("world"));
        assert!(json.contains("200"));
    }

    #[test]
    fn serialize_batch_result_error() {
        let result = BatchResultItem {
            custom_id: "req-2".to_string(),
            response: None,
            error: Some(BatchError {
                code: "engine_error".to_string(),
                message: "out of memory".to_string(),
            }),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("engine_error"));
        assert!(json.contains("out of memory"));
    }

    #[test]
    fn batch_status_serialization() {
        assert_eq!(
            serde_json::to_string(&BatchStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&BatchStatus::Completed).unwrap(),
            "\"completed\""
        );
        assert_eq!(
            serde_json::to_string(&BatchStatus::Cancelled).unwrap(),
            "\"cancelled\""
        );
    }

    #[test]
    fn parse_jsonl_input() {
        let input = r#"{"custom_id":"1","body":{"prompt":"a","max_tokens":5}}
{"custom_id":"2","body":{"prompt":"b","max_tokens":10}}
{"custom_id":"3","body":{"prompt":"c","max_tokens":15}}"#;

        let items: Vec<BatchRequestItem> = input
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].custom_id, "1");
        assert_eq!(items[1].body.max_tokens, 10);
        assert_eq!(items[2].body.prompt.as_deref(), Some("c"));
    }

    #[test]
    fn batch_job_default_state() {
        let job = BatchJob {
            id: "batch_test".to_string(),
            status: BatchStatus::InProgress,
            created_at: 1000,
            completed_at: None,
            request_counts: RequestCounts {
                total: 5,
                completed: 0,
                failed: 0,
            },
            results: vec![],
            cancel_requested: false,
        };
        assert_eq!(job.request_counts.total, 5);
        assert!(!job.cancel_requested);
    }
}
