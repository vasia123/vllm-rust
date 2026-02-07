//! Prometheus metrics endpoint for production monitoring.

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use once_cell::sync::Lazy;
use prometheus::{
    register_gauge, register_histogram, register_int_counter, register_int_gauge, Encoder, Gauge,
    Histogram, IntCounter, IntGauge, TextEncoder,
};

use super::AdminState;
use crate::api::error::ApiError;

// Request counters
static REQUESTS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("vllm_requests_total", "Total number of requests received")
        .expect("failed to register vllm_requests_total")
});

static REQUESTS_SUCCESS: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "vllm_requests_success_total",
        "Total number of successful requests"
    )
    .expect("failed to register vllm_requests_success_total")
});

static REQUESTS_ERROR: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "vllm_requests_error_total",
        "Total number of failed requests"
    )
    .expect("failed to register vllm_requests_error_total")
});

// Active request gauges
static RUNNING_REQUESTS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(
        "vllm_running_requests",
        "Number of requests currently being processed"
    )
    .expect("failed to register vllm_running_requests")
});

static WAITING_REQUESTS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(
        "vllm_waiting_requests",
        "Number of requests waiting in queue"
    )
    .expect("failed to register vllm_waiting_requests")
});

// KV cache gauges
static KV_CACHE_FREE_BLOCKS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(
        "vllm_kv_cache_free_blocks",
        "Number of free KV cache blocks"
    )
    .expect("failed to register vllm_kv_cache_free_blocks")
});

static KV_CACHE_TOTAL_BLOCKS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(
        "vllm_kv_cache_total_blocks",
        "Total number of KV cache blocks"
    )
    .expect("failed to register vllm_kv_cache_total_blocks")
});

static KV_CACHE_USAGE_RATIO: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        "vllm_kv_cache_usage_ratio",
        "Ratio of used KV cache blocks (0.0 - 1.0)"
    )
    .expect("failed to register vllm_kv_cache_usage_ratio")
});

// Prefix cache gauges
static PREFIX_CACHE_BLOCKS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!("vllm_prefix_cache_blocks", "Number of cached prefix blocks")
        .expect("failed to register vllm_prefix_cache_blocks")
});

static PREFIX_CACHE_EVICTABLE: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(
        "vllm_prefix_cache_evictable_blocks",
        "Number of evictable prefix cache blocks"
    )
    .expect("failed to register vllm_prefix_cache_evictable_blocks")
});

// Latency histograms
static TIME_TO_FIRST_TOKEN: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "vllm_time_to_first_token_seconds",
        "Time to first token in seconds",
        vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .expect("failed to register vllm_time_to_first_token_seconds")
});

static E2E_LATENCY: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "vllm_e2e_latency_seconds",
        "End-to-end request latency in seconds",
        vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
    )
    .expect("failed to register vllm_e2e_latency_seconds")
});

static TOKENS_PER_SECOND: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "vllm_tokens_per_second",
        "Token generation throughput per request",
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    )
    .expect("failed to register vllm_tokens_per_second")
});

// Per-token latency
static TIME_PER_OUTPUT_TOKEN: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "vllm_time_per_output_token_seconds",
        "Time per output token in seconds",
        vec![0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5]
    )
    .expect("failed to register vllm_time_per_output_token_seconds")
});

// Model forward pass time
static MODEL_FORWARD_TIME: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "vllm_model_forward_time_seconds",
        "Time for model forward pass in seconds",
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )
    .expect("failed to register vllm_model_forward_time_seconds")
});

// Token counters
static PROMPT_TOKENS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "vllm_prompt_tokens_total",
        "Total number of prompt tokens processed"
    )
    .expect("failed to register vllm_prompt_tokens_total")
});

static GENERATION_TOKENS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "vllm_generation_tokens_total",
        "Total number of tokens generated"
    )
    .expect("failed to register vllm_generation_tokens_total")
});

// Scheduler metrics
static NUM_PREEMPTIONS: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "vllm_num_preemptions_total",
        "Total number of request preemptions"
    )
    .expect("failed to register vllm_num_preemptions_total")
});

// Prefix cache hit ratio
static PREFIX_CACHE_HIT_RATIO: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        "vllm_prefix_cache_hit_ratio",
        "Prefix cache hit ratio (0.0 - 1.0)"
    )
    .expect("failed to register vllm_prefix_cache_hit_ratio")
});

// Batch size tracking
static BATCH_SIZE: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "vllm_batch_size",
        "Number of sequences per forward pass",
        vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    )
    .expect("failed to register vllm_batch_size")
});

// Server health
static SERVER_ACCEPTING: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(
        "vllm_server_accepting_requests",
        "Whether the server is accepting new requests (1 = yes, 0 = no)"
    )
    .expect("failed to register vllm_server_accepting_requests")
});

static SERVER_UPTIME: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!("vllm_server_uptime_seconds", "Server uptime in seconds")
        .expect("failed to register vllm_server_uptime_seconds")
});

/// Initialize all metrics (call once at startup to ensure registration).
pub fn init_metrics() {
    // Force lazy initialization by referencing each metric
    let _ = &*REQUESTS_TOTAL;
    let _ = &*REQUESTS_SUCCESS;
    let _ = &*REQUESTS_ERROR;
    let _ = &*RUNNING_REQUESTS;
    let _ = &*WAITING_REQUESTS;
    let _ = &*KV_CACHE_FREE_BLOCKS;
    let _ = &*KV_CACHE_TOTAL_BLOCKS;
    let _ = &*KV_CACHE_USAGE_RATIO;
    let _ = &*PREFIX_CACHE_BLOCKS;
    let _ = &*PREFIX_CACHE_EVICTABLE;
    let _ = &*TIME_TO_FIRST_TOKEN;
    let _ = &*E2E_LATENCY;
    let _ = &*TOKENS_PER_SECOND;
    let _ = &*TIME_PER_OUTPUT_TOKEN;
    let _ = &*MODEL_FORWARD_TIME;
    let _ = &*PROMPT_TOKENS_TOTAL;
    let _ = &*GENERATION_TOKENS_TOTAL;
    let _ = &*NUM_PREEMPTIONS;
    let _ = &*PREFIX_CACHE_HIT_RATIO;
    let _ = &*BATCH_SIZE;
    let _ = &*SERVER_ACCEPTING;
    let _ = &*SERVER_UPTIME;
}

/// Record a new request received.
pub fn inc_requests_total() {
    REQUESTS_TOTAL.inc();
}

/// Record a successful request completion.
pub fn inc_requests_success() {
    REQUESTS_SUCCESS.inc();
}

/// Record a failed request.
pub fn inc_requests_error() {
    REQUESTS_ERROR.inc();
}

/// Record time to first token.
pub fn observe_ttft(seconds: f64) {
    TIME_TO_FIRST_TOKEN.observe(seconds);
}

/// Record end-to-end latency.
pub fn observe_e2e_latency(seconds: f64) {
    E2E_LATENCY.observe(seconds);
}

/// Record tokens per second for a request.
pub fn observe_tps(tokens_per_second: f64) {
    TOKENS_PER_SECOND.observe(tokens_per_second);
}

/// Record time per output token.
pub fn observe_time_per_token(seconds: f64) {
    TIME_PER_OUTPUT_TOKEN.observe(seconds);
}

/// Record model forward pass duration.
pub fn observe_model_forward_time(seconds: f64) {
    MODEL_FORWARD_TIME.observe(seconds);
}

/// Record prompt tokens processed.
pub fn inc_prompt_tokens(count: u64) {
    PROMPT_TOKENS_TOTAL.inc_by(count);
}

/// Record generation tokens produced.
pub fn inc_generation_tokens(count: u64) {
    GENERATION_TOKENS_TOTAL.inc_by(count);
}

/// Record a preemption event.
pub fn inc_preemptions() {
    NUM_PREEMPTIONS.inc();
}

/// Record batch size for a forward pass.
pub fn observe_batch_size(size: f64) {
    BATCH_SIZE.observe(size);
}

/// GET /admin/metrics/prometheus - Prometheus format metrics.
pub async fn prometheus_metrics(State(state): State<AdminState>) -> Result<Response, ApiError> {
    // Update gauges from current engine state
    let engine_stats = state
        .engine
        .get()
        .get_stats()
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    // Update running/waiting requests
    RUNNING_REQUESTS.set(engine_stats.num_running_requests as i64);
    WAITING_REQUESTS.set(engine_stats.num_waiting_requests as i64);

    // Update KV cache metrics
    KV_CACHE_FREE_BLOCKS.set(engine_stats.num_free_blocks as i64);
    KV_CACHE_TOTAL_BLOCKS.set(engine_stats.num_total_blocks as i64);

    if engine_stats.num_total_blocks > 0 {
        let used = engine_stats.num_total_blocks - engine_stats.num_free_blocks;
        let ratio = used as f64 / engine_stats.num_total_blocks as f64;
        KV_CACHE_USAGE_RATIO.set(ratio);
    }

    // Update prefix cache metrics
    if let Some((cached, evictable)) = engine_stats.prefix_cache_stats {
        PREFIX_CACHE_BLOCKS.set(cached as i64);
        PREFIX_CACHE_EVICTABLE.set(evictable as i64);
    }

    // Update prefix cache hit ratio from detailed stats
    if let Some(ref detailed) = engine_stats.prefix_cache_detailed_stats {
        let total = detailed.num_hits + detailed.num_misses;
        if total > 0 {
            let ratio = detailed.num_hits as f64 / total as f64;
            PREFIX_CACHE_HIT_RATIO.set(ratio);
        }
    } else if let Some(ref recent) = engine_stats.prefix_cache_recent_stats {
        PREFIX_CACHE_HIT_RATIO.set(recent.hit_rate);
    }

    // Update server health
    SERVER_ACCEPTING.set(if state.accepting_requests() { 1 } else { 0 });

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let uptime = now
        .checked_sub(state.start_time)
        .unwrap_or_default()
        .as_secs();
    SERVER_UPTIME.set(uptime as i64);

    // Encode all metrics in Prometheus text format
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|e| ApiError::InternalError(format!("Failed to encode metrics: {}", e)))?;

    Ok((
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        buffer,
    )
        .into_response())
}
