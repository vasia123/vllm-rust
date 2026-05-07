//! Metrics endpoints for admin API.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::{self, Stream};

use super::types::{AdminMetrics, PrefixCacheStats, SpecDecodeStatsView};
use crate::api::admin::AdminState;
use crate::api::error::ApiError;
use vllm_core::engine::SpecDecodingStats;

fn spec_view(stats: &SpecDecodingStats) -> SpecDecodeStatsView {
    SpecDecodeStatsView {
        num_drafts: stats.num_drafts,
        num_draft_tokens: stats.num_draft_tokens,
        num_accepted_tokens: stats.num_accepted_tokens,
        num_accepted_tokens_per_pos: stats.num_accepted_tokens_per_pos.clone(),
        acceptance_rate: stats.acceptance_rate(),
        mean_accepted_per_draft: stats.mean_acceptance_length(),
    }
}

/// GET /admin/metrics - Get current metrics snapshot.
pub async fn get_metrics(State(state): State<AdminState>) -> Result<impl IntoResponse, ApiError> {
    let engine_stats = state
        .engine
        .get()
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

    let spec_decode = engine_stats.spec_decode_stats.as_ref().map(spec_view);

    let metrics = AdminMetrics {
        kv_cache: engine_stats.kv_cache_metrics,
        running_requests: engine_stats.num_running_requests,
        waiting_requests: engine_stats.num_waiting_requests,
        model_id: state.model_id.read().await.clone(),
        uptime_seconds: uptime,
        accepting_requests: state.accepting_requests(),
        timestamp_ms: now.as_millis() as u64,
        num_free_blocks: engine_stats.num_free_blocks,
        num_total_blocks: engine_stats.num_total_blocks,
        block_size: engine_stats.block_size,
        prefix_cache_stats,
        spec_decode,
    };

    Ok(Json(metrics))
}

/// GET /admin/metrics/stream - SSE stream of metrics updates.
pub async fn metrics_stream(
    State(state): State<AdminState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::unfold(state, |state| async move {
        tokio::time::sleep(Duration::from_secs(1)).await;

        let engine = state.engine.get();
        let metrics = match engine.get_stats().await {
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

                let model_id = state.model_id.read().await.clone();
                let spec_decode = stats.spec_decode_stats.as_ref().map(spec_view);

                AdminMetrics {
                    kv_cache: stats.kv_cache_metrics,
                    running_requests: stats.num_running_requests,
                    waiting_requests: stats.num_waiting_requests,
                    model_id,
                    uptime_seconds: uptime,
                    accepting_requests: state.accepting_requests(),
                    timestamp_ms: now.as_millis() as u64,
                    num_free_blocks: stats.num_free_blocks,
                    num_total_blocks: stats.num_total_blocks,
                    block_size: stats.block_size,
                    prefix_cache_stats,
                    spec_decode,
                }
            }
            Err(_) => return None,
        };

        let event = Event::default().event("metrics").json_data(&metrics).ok()?;

        Some((Ok(event), state))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_view_propagates_counters_and_derived_rates() {
        let mut stats = SpecDecodingStats::new(4);
        // Two drafts: one fully accepted (4/4), one partial (2/4).
        stats.observe_draft(4, 4);
        stats.observe_draft(4, 2);
        let view = spec_view(&stats);

        assert_eq!(view.num_drafts, 2);
        assert_eq!(view.num_draft_tokens, 8);
        assert_eq!(view.num_accepted_tokens, 6);
        assert_eq!(view.num_accepted_tokens_per_pos, vec![2, 2, 1, 1]);
        // 6 / 8 = 0.75 acceptance.
        assert!((view.acceptance_rate - 0.75).abs() < 1e-9);
        // 1 + 6 / 2 = 4.0 tokens emitted per draft round.
        assert!((view.mean_accepted_per_draft - 4.0).abs() < 1e-9);
    }

    #[test]
    fn spec_view_empty_stats_has_zero_rates() {
        let stats = SpecDecodingStats::new(3);
        let view = spec_view(&stats);
        assert_eq!(view.num_drafts, 0);
        assert_eq!(view.acceptance_rate, 0.0);
        assert_eq!(view.mean_accepted_per_draft, 0.0);
        assert_eq!(view.num_accepted_tokens_per_pos, vec![0, 0, 0]);
    }

    #[test]
    fn spec_view_serializes_to_json_with_expected_field_names() {
        let mut stats = SpecDecodingStats::new(2);
        stats.observe_draft(2, 1);
        let view = spec_view(&stats);
        let json = serde_json::to_value(&view).expect("serialize");

        // Field names form the public bench-harness contract.
        assert_eq!(json["num_drafts"], 1);
        assert_eq!(json["num_draft_tokens"], 2);
        assert_eq!(json["num_accepted_tokens"], 1);
        assert!(json.get("acceptance_rate").is_some());
        assert!(json.get("mean_accepted_per_draft").is_some());
        assert!(json.get("num_accepted_tokens_per_pos").is_some());
    }
}
