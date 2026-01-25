//! Metrics endpoints for admin API.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::{self, Stream};

use super::types::{AdminMetrics, PrefixCacheStats};
use crate::api::admin::AdminState;
use crate::api::error::ApiError;

/// GET /admin/metrics - Get current metrics snapshot.
pub async fn get_metrics(State(state): State<AdminState>) -> Result<impl IntoResponse, ApiError> {
    let engine_stats = state
        .engine
        .get_stats()
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let uptime = now
        .checked_sub(state.start_time)
        .unwrap_or_default()
        .as_secs();

    let prefix_cache_stats =
        engine_stats
            .prefix_cache_stats
            .map(|(cached, evictable)| PrefixCacheStats {
                cached_blocks: cached,
                evictable_blocks: evictable,
            });

    let metrics = AdminMetrics {
        kv_cache: engine_stats.kv_cache_metrics,
        running_requests: engine_stats.num_running_requests,
        waiting_requests: engine_stats.num_waiting_requests,
        model_id: state.model_id.clone(),
        uptime_seconds: uptime,
        accepting_requests: state.accepting_requests(),
        timestamp_ms: now.as_millis() as u64,
        num_free_blocks: engine_stats.num_free_blocks,
        num_total_blocks: engine_stats.num_total_blocks,
        block_size: engine_stats.block_size,
        prefix_cache_stats,
    };

    Ok(Json(metrics))
}

/// GET /admin/metrics/stream - SSE stream of metrics updates.
pub async fn metrics_stream(
    State(state): State<AdminState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::unfold(state, |state| async move {
        tokio::time::sleep(Duration::from_secs(1)).await;

        let metrics = match state.engine.get_stats().await {
            Ok(stats) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                let uptime = now
                    .checked_sub(state.start_time)
                    .unwrap_or_default()
                    .as_secs();

                let prefix_cache_stats =
                    stats
                        .prefix_cache_stats
                        .map(|(cached, evictable)| PrefixCacheStats {
                            cached_blocks: cached,
                            evictable_blocks: evictable,
                        });

                AdminMetrics {
                    kv_cache: stats.kv_cache_metrics,
                    running_requests: stats.num_running_requests,
                    waiting_requests: stats.num_waiting_requests,
                    model_id: state.model_id.clone(),
                    uptime_seconds: uptime,
                    accepting_requests: state.accepting_requests(),
                    timestamp_ms: now.as_millis() as u64,
                    num_free_blocks: stats.num_free_blocks,
                    num_total_blocks: stats.num_total_blocks,
                    block_size: stats.block_size,
                    prefix_cache_stats,
                }
            }
            Err(_) => return None,
        };

        let event = Event::default().event("metrics").json_data(&metrics).ok()?;

        Some((Ok(event), state))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
