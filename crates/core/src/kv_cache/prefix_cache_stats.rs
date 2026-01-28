//! Prefix cache statistics for monitoring and debugging.
//!
//! Provides both instantaneous and sliding window metrics for prefix cache
//! hit/miss tracking, following the vLLM reference implementation.

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counters for prefix cache statistics.
///
/// Thread-safe counters that can be shared across components. For read-only
/// snapshots, use [`PrefixCacheStatsSnapshot`].
pub struct PrefixCacheStats {
    /// Total number of block hits (blocks reused from cache).
    num_hits: AtomicU64,
    /// Total number of block misses (blocks needed computation).
    num_misses: AtomicU64,
    /// Total number of blocks evicted from cache.
    num_evictions: AtomicU64,
    /// Total cache queries (match_prefix calls).
    num_queries: AtomicU64,
    /// Total number of tokens queried.
    tokens_queried: AtomicU64,
    /// Total number of tokens that hit cache.
    tokens_hit: AtomicU64,
}

impl Default for PrefixCacheStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixCacheStats {
    /// Create new stats with all counters at zero.
    pub fn new() -> Self {
        Self {
            num_hits: AtomicU64::new(0),
            num_misses: AtomicU64::new(0),
            num_evictions: AtomicU64::new(0),
            num_queries: AtomicU64::new(0),
            tokens_queried: AtomicU64::new(0),
            tokens_hit: AtomicU64::new(0),
        }
    }

    /// Record a cache query result.
    ///
    /// # Arguments
    /// * `block_hits` - Number of blocks that hit the cache.
    /// * `block_misses` - Number of blocks that missed the cache.
    /// * `tokens_queried` - Total number of tokens queried.
    /// * `tokens_hit` - Number of tokens covered by cached blocks.
    pub fn record_query(
        &self,
        block_hits: usize,
        block_misses: usize,
        tokens_queried: usize,
        tokens_hit: usize,
    ) {
        self.num_queries.fetch_add(1, Ordering::Relaxed);
        self.num_hits
            .fetch_add(block_hits as u64, Ordering::Relaxed);
        self.num_misses
            .fetch_add(block_misses as u64, Ordering::Relaxed);
        self.tokens_queried
            .fetch_add(tokens_queried as u64, Ordering::Relaxed);
        self.tokens_hit
            .fetch_add(tokens_hit as u64, Ordering::Relaxed);
    }

    /// Record an eviction event.
    pub fn record_eviction(&self, num_blocks: usize) {
        self.num_evictions
            .fetch_add(num_blocks as u64, Ordering::Relaxed);
    }

    /// Get total number of block hits.
    pub fn num_hits(&self) -> u64 {
        self.num_hits.load(Ordering::Relaxed)
    }

    /// Get total number of block misses.
    pub fn num_misses(&self) -> u64 {
        self.num_misses.load(Ordering::Relaxed)
    }

    /// Get total number of blocks evicted.
    pub fn num_evictions(&self) -> u64 {
        self.num_evictions.load(Ordering::Relaxed)
    }

    /// Get total number of cache queries.
    pub fn num_queries(&self) -> u64 {
        self.num_queries.load(Ordering::Relaxed)
    }

    /// Get total number of tokens queried.
    pub fn tokens_queried(&self) -> u64 {
        self.tokens_queried.load(Ordering::Relaxed)
    }

    /// Get total number of tokens that hit cache.
    pub fn tokens_hit(&self) -> u64 {
        self.tokens_hit.load(Ordering::Relaxed)
    }

    /// Calculate block hit rate (0.0 to 1.0).
    ///
    /// Returns `None` if no blocks have been queried.
    pub fn block_hit_rate(&self) -> Option<f64> {
        let hits = self.num_hits.load(Ordering::Relaxed);
        let misses = self.num_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            None
        } else {
            Some(hits as f64 / total as f64)
        }
    }

    /// Calculate token hit rate (0.0 to 1.0).
    ///
    /// Returns `None` if no tokens have been queried.
    pub fn token_hit_rate(&self) -> Option<f64> {
        let queried = self.tokens_queried.load(Ordering::Relaxed);
        let hit = self.tokens_hit.load(Ordering::Relaxed);
        if queried == 0 {
            None
        } else {
            Some(hit as f64 / queried as f64)
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.num_hits.store(0, Ordering::Relaxed);
        self.num_misses.store(0, Ordering::Relaxed);
        self.num_evictions.store(0, Ordering::Relaxed);
        self.num_queries.store(0, Ordering::Relaxed);
        self.tokens_queried.store(0, Ordering::Relaxed);
        self.tokens_hit.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of all statistics.
    pub fn snapshot(&self) -> PrefixCacheStatsSnapshot {
        let num_hits = self.num_hits.load(Ordering::Relaxed);
        let num_misses = self.num_misses.load(Ordering::Relaxed);
        let num_evictions = self.num_evictions.load(Ordering::Relaxed);
        let num_queries = self.num_queries.load(Ordering::Relaxed);
        let tokens_queried = self.tokens_queried.load(Ordering::Relaxed);
        let tokens_hit = self.tokens_hit.load(Ordering::Relaxed);

        let block_hit_rate = if num_hits + num_misses > 0 {
            Some(num_hits as f64 / (num_hits + num_misses) as f64)
        } else {
            None
        };

        let token_hit_rate = if tokens_queried > 0 {
            Some(tokens_hit as f64 / tokens_queried as f64)
        } else {
            None
        };

        PrefixCacheStatsSnapshot {
            num_hits,
            num_misses,
            num_evictions,
            num_queries,
            tokens_queried,
            tokens_hit,
            block_hit_rate,
            token_hit_rate,
        }
    }
}

/// Point-in-time snapshot of prefix cache statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrefixCacheStatsSnapshot {
    /// Total number of block hits.
    pub num_hits: u64,
    /// Total number of block misses.
    pub num_misses: u64,
    /// Total number of blocks evicted.
    pub num_evictions: u64,
    /// Total number of cache queries.
    pub num_queries: u64,
    /// Total number of tokens queried.
    pub tokens_queried: u64,
    /// Total number of tokens that hit cache.
    pub tokens_hit: u64,
    /// Block hit rate (hits / (hits + misses)), None if no data.
    pub block_hit_rate: Option<f64>,
    /// Token hit rate (tokens_hit / tokens_queried), None if no data.
    pub token_hit_rate: Option<f64>,
}

/// Single entry in the sliding window metrics.
#[derive(Debug, Clone)]
struct WindowEntry {
    /// Number of requests in this batch.
    requests: u64,
    /// Number of tokens queried in this batch.
    tokens_queried: u64,
    /// Number of tokens that hit cache in this batch.
    tokens_hit: u64,
}

/// Sliding window metrics for recent prefix cache performance.
///
/// Tracks hit rate over the most recent N requests, similar to vLLM's
/// CachingMetrics class. This provides a more responsive view of cache
/// performance compared to lifetime averages.
pub struct SlidingWindowMetrics {
    /// Maximum number of requests to track.
    max_requests: u64,
    /// Current aggregated request count.
    aggregated_requests: u64,
    /// Current aggregated tokens queried.
    aggregated_tokens_queried: u64,
    /// Current aggregated tokens hit.
    aggregated_tokens_hit: u64,
    /// Queue of recent entries.
    queue: VecDeque<WindowEntry>,
}

impl SlidingWindowMetrics {
    /// Create a new sliding window metrics tracker.
    ///
    /// # Arguments
    /// * `max_requests` - Maximum number of requests to include in the window.
    ///   Defaults to 1000 if 0 is passed.
    pub fn new(max_requests: u64) -> Self {
        Self {
            max_requests: if max_requests == 0 {
                1000
            } else {
                max_requests
            },
            aggregated_requests: 0,
            aggregated_tokens_queried: 0,
            aggregated_tokens_hit: 0,
            queue: VecDeque::new(),
        }
    }

    /// Observe a batch of requests.
    ///
    /// # Arguments
    /// * `num_requests` - Number of requests in this batch.
    /// * `tokens_queried` - Total tokens queried across all requests.
    /// * `tokens_hit` - Total tokens that hit cache across all requests.
    /// * `reset` - If true, reset all metrics before recording (used after cache reset).
    pub fn observe(
        &mut self,
        num_requests: u64,
        tokens_queried: u64,
        tokens_hit: u64,
        reset: bool,
    ) {
        if reset {
            self.reset();
        }

        // Skip empty observations to preserve useful data
        if num_requests == 0 {
            return;
        }

        // Add the new entry
        let entry = WindowEntry {
            requests: num_requests,
            tokens_queried,
            tokens_hit,
        };
        self.queue.push_back(entry);
        self.aggregated_requests += num_requests;
        self.aggregated_tokens_queried += tokens_queried;
        self.aggregated_tokens_hit += tokens_hit;

        // Remove old entries until we're within the window, keeping at least one entry
        while self.queue.len() > 1 && self.aggregated_requests > self.max_requests {
            if let Some(old) = self.queue.pop_front() {
                self.aggregated_requests -= old.requests;
                self.aggregated_tokens_queried -= old.tokens_queried;
                self.aggregated_tokens_hit -= old.tokens_hit;
            }
        }
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.aggregated_requests = 0;
        self.aggregated_tokens_queried = 0;
        self.aggregated_tokens_hit = 0;
        self.queue.clear();
    }

    /// Check if any data has been observed.
    pub fn is_empty(&self) -> bool {
        self.aggregated_requests == 0
    }

    /// Get the number of requests currently in the window.
    pub fn num_requests(&self) -> u64 {
        self.aggregated_requests
    }

    /// Get the total tokens queried in the window.
    pub fn tokens_queried(&self) -> u64 {
        self.aggregated_tokens_queried
    }

    /// Get the total tokens hit in the window.
    pub fn tokens_hit(&self) -> u64 {
        self.aggregated_tokens_hit
    }

    /// Calculate the token hit rate for the window.
    ///
    /// Returns 0.0 if no tokens have been queried.
    pub fn hit_rate(&self) -> f64 {
        if self.aggregated_tokens_queried == 0 {
            0.0
        } else {
            self.aggregated_tokens_hit as f64 / self.aggregated_tokens_queried as f64
        }
    }

    /// Get a snapshot of sliding window metrics.
    pub fn snapshot(&self) -> SlidingWindowSnapshot {
        SlidingWindowSnapshot {
            num_requests: self.aggregated_requests,
            tokens_queried: self.aggregated_tokens_queried,
            tokens_hit: self.aggregated_tokens_hit,
            hit_rate: self.hit_rate(),
            window_size: self.max_requests,
        }
    }
}

impl Default for SlidingWindowMetrics {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Snapshot of sliding window metrics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SlidingWindowSnapshot {
    /// Number of requests in the current window.
    pub num_requests: u64,
    /// Total tokens queried in the window.
    pub tokens_queried: u64,
    /// Total tokens hit in the window.
    pub tokens_hit: u64,
    /// Token hit rate for the window.
    pub hit_rate: f64,
    /// Maximum window size (for reference).
    pub window_size: u64,
}

/// Prometheus-style metric representation.
#[derive(Debug, Clone)]
pub struct PrometheusMetric {
    pub name: String,
    pub help: String,
    pub metric_type: PrometheusMetricType,
    pub value: f64,
    pub labels: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy)]
pub enum PrometheusMetricType {
    Counter,
    Gauge,
}

impl PrometheusMetric {
    /// Format as Prometheus exposition format.
    pub fn to_prometheus_format(&self) -> String {
        let type_str = match self.metric_type {
            PrometheusMetricType::Counter => "counter",
            PrometheusMetricType::Gauge => "gauge",
        };

        let labels_str = if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        };

        format!(
            "# HELP {} {}\n# TYPE {} {}\n{}{} {}\n",
            self.name, self.help, self.name, type_str, self.name, labels_str, self.value
        )
    }
}

/// Export prefix cache stats as Prometheus metrics.
pub fn export_prometheus_metrics(
    stats: &PrefixCacheStatsSnapshot,
    sliding: Option<&SlidingWindowSnapshot>,
) -> Vec<PrometheusMetric> {
    let mut metrics = vec![
        PrometheusMetric {
            name: "vllm_prefix_cache_hits_total".to_string(),
            help: "Total number of prefix cache block hits".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: stats.num_hits as f64,
            labels: vec![],
        },
        PrometheusMetric {
            name: "vllm_prefix_cache_misses_total".to_string(),
            help: "Total number of prefix cache block misses".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: stats.num_misses as f64,
            labels: vec![],
        },
        PrometheusMetric {
            name: "vllm_prefix_cache_evictions_total".to_string(),
            help: "Total number of prefix cache evictions".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: stats.num_evictions as f64,
            labels: vec![],
        },
        PrometheusMetric {
            name: "vllm_prefix_cache_queries_total".to_string(),
            help: "Total number of prefix cache queries".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: stats.num_queries as f64,
            labels: vec![],
        },
        PrometheusMetric {
            name: "vllm_prefix_cache_tokens_queried_total".to_string(),
            help: "Total number of tokens queried from prefix cache".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: stats.tokens_queried as f64,
            labels: vec![],
        },
        PrometheusMetric {
            name: "vllm_prefix_cache_tokens_hit_total".to_string(),
            help: "Total number of tokens that hit prefix cache".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: stats.tokens_hit as f64,
            labels: vec![],
        },
    ];

    // Add hit rate gauges
    if let Some(rate) = stats.block_hit_rate {
        metrics.push(PrometheusMetric {
            name: "vllm_prefix_cache_block_hit_rate".to_string(),
            help: "Prefix cache block hit rate (lifetime)".to_string(),
            metric_type: PrometheusMetricType::Gauge,
            value: rate,
            labels: vec![],
        });
    }

    if let Some(rate) = stats.token_hit_rate {
        metrics.push(PrometheusMetric {
            name: "vllm_prefix_cache_token_hit_rate".to_string(),
            help: "Prefix cache token hit rate (lifetime)".to_string(),
            metric_type: PrometheusMetricType::Gauge,
            value: rate,
            labels: vec![],
        });
    }

    // Add sliding window metrics if available
    if let Some(sw) = sliding {
        metrics.push(PrometheusMetric {
            name: "vllm_prefix_cache_recent_hit_rate".to_string(),
            help: "Prefix cache token hit rate (recent window)".to_string(),
            metric_type: PrometheusMetricType::Gauge,
            value: sw.hit_rate,
            labels: vec![],
        });
        metrics.push(PrometheusMetric {
            name: "vllm_prefix_cache_recent_requests".to_string(),
            help: "Number of requests in recent metrics window".to_string(),
            metric_type: PrometheusMetricType::Gauge,
            value: sw.num_requests as f64,
            labels: vec![],
        });
    }

    metrics
}

/// Format metrics in Prometheus exposition format.
pub fn format_prometheus_output(
    stats: &PrefixCacheStatsSnapshot,
    sliding: Option<&SlidingWindowSnapshot>,
) -> String {
    let metrics = export_prometheus_metrics(stats, sliding);
    metrics
        .iter()
        .map(|m| m.to_prometheus_format())
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PrefixCacheStats tests ────────────────────────────────────────────────

    #[test]
    fn stats_new_is_zero() {
        let stats = PrefixCacheStats::new();
        assert_eq!(stats.num_hits(), 0);
        assert_eq!(stats.num_misses(), 0);
        assert_eq!(stats.num_evictions(), 0);
        assert_eq!(stats.num_queries(), 0);
        assert_eq!(stats.tokens_queried(), 0);
        assert_eq!(stats.tokens_hit(), 0);
        assert_eq!(stats.block_hit_rate(), None);
        assert_eq!(stats.token_hit_rate(), None);
    }

    #[test]
    fn stats_record_query() {
        let stats = PrefixCacheStats::new();

        // 2 hits, 1 miss, 12 tokens queried (3 blocks * 4 tokens), 8 tokens hit (2 blocks * 4)
        stats.record_query(2, 1, 12, 8);

        assert_eq!(stats.num_hits(), 2);
        assert_eq!(stats.num_misses(), 1);
        assert_eq!(stats.num_queries(), 1);
        assert_eq!(stats.tokens_queried(), 12);
        assert_eq!(stats.tokens_hit(), 8);
    }

    #[test]
    fn stats_record_eviction() {
        let stats = PrefixCacheStats::new();
        stats.record_eviction(3);
        assert_eq!(stats.num_evictions(), 3);
        stats.record_eviction(2);
        assert_eq!(stats.num_evictions(), 5);
    }

    #[test]
    fn stats_block_hit_rate() {
        let stats = PrefixCacheStats::new();

        // No data
        assert_eq!(stats.block_hit_rate(), None);

        // 3 hits, 1 miss = 75%
        stats.record_query(3, 1, 16, 12);
        let rate = stats.block_hit_rate().unwrap();
        assert!((rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn stats_token_hit_rate() {
        let stats = PrefixCacheStats::new();

        // No data
        assert_eq!(stats.token_hit_rate(), None);

        // 8 tokens hit out of 16 = 50%
        stats.record_query(2, 2, 16, 8);
        let rate = stats.token_hit_rate().unwrap();
        assert!((rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn stats_reset() {
        let stats = PrefixCacheStats::new();
        stats.record_query(5, 3, 32, 20);
        stats.record_eviction(2);

        stats.reset();

        assert_eq!(stats.num_hits(), 0);
        assert_eq!(stats.num_misses(), 0);
        assert_eq!(stats.num_evictions(), 0);
        assert_eq!(stats.num_queries(), 0);
        assert_eq!(stats.tokens_queried(), 0);
        assert_eq!(stats.tokens_hit(), 0);
    }

    #[test]
    fn stats_snapshot() {
        let stats = PrefixCacheStats::new();
        stats.record_query(4, 2, 24, 16);
        stats.record_eviction(1);

        let snap = stats.snapshot();

        assert_eq!(snap.num_hits, 4);
        assert_eq!(snap.num_misses, 2);
        assert_eq!(snap.num_evictions, 1);
        assert_eq!(snap.num_queries, 1);
        assert_eq!(snap.tokens_queried, 24);
        assert_eq!(snap.tokens_hit, 16);

        // 4 / (4 + 2) = 0.666...
        let block_rate = snap.block_hit_rate.unwrap();
        assert!((block_rate - 0.6666).abs() < 0.01);

        // 16 / 24 = 0.666...
        let token_rate = snap.token_hit_rate.unwrap();
        assert!((token_rate - 0.6666).abs() < 0.01);
    }

    #[test]
    fn stats_accumulates() {
        let stats = PrefixCacheStats::new();

        stats.record_query(2, 2, 16, 8); // 50% hit
        stats.record_query(4, 0, 16, 16); // 100% hit

        assert_eq!(stats.num_hits(), 6);
        assert_eq!(stats.num_misses(), 2);
        assert_eq!(stats.num_queries(), 2);
        assert_eq!(stats.tokens_queried(), 32);
        assert_eq!(stats.tokens_hit(), 24);

        // Overall: 6 / 8 = 75%
        let rate = stats.block_hit_rate().unwrap();
        assert!((rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn stats_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        let stats = Arc::new(PrefixCacheStats::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let s = Arc::clone(&stats);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    s.record_query(1, 1, 8, 4);
                    s.record_eviction(1);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(stats.num_queries(), 1000);
        assert_eq!(stats.num_hits(), 1000);
        assert_eq!(stats.num_misses(), 1000);
        assert_eq!(stats.num_evictions(), 1000);
    }

    // ─── SlidingWindowMetrics tests ────────────────────────────────────────────

    #[test]
    fn sliding_new_is_empty() {
        let window = SlidingWindowMetrics::new(100);
        assert!(window.is_empty());
        assert_eq!(window.num_requests(), 0);
        assert_eq!(window.tokens_queried(), 0);
        assert_eq!(window.tokens_hit(), 0);
        assert!((window.hit_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn sliding_zero_defaults_to_1000() {
        let window = SlidingWindowMetrics::new(0);
        let snap = window.snapshot();
        assert_eq!(snap.window_size, 1000);
    }

    #[test]
    fn sliding_observe_basic() {
        let mut window = SlidingWindowMetrics::new(100);

        window.observe(5, 40, 20, false); // 5 requests, 40 tokens queried, 20 hit

        assert!(!window.is_empty());
        assert_eq!(window.num_requests(), 5);
        assert_eq!(window.tokens_queried(), 40);
        assert_eq!(window.tokens_hit(), 20);
        assert!((window.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn sliding_observe_accumulates() {
        let mut window = SlidingWindowMetrics::new(100);

        window.observe(3, 24, 12, false); // 50% hit
        window.observe(2, 16, 16, false); // 100% hit

        assert_eq!(window.num_requests(), 5);
        assert_eq!(window.tokens_queried(), 40);
        assert_eq!(window.tokens_hit(), 28);
        // 28 / 40 = 70%
        assert!((window.hit_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn sliding_window_evicts_old_entries() {
        let mut window = SlidingWindowMetrics::new(10); // Small window

        // Add entries that exceed window size
        window.observe(4, 32, 16, false);
        window.observe(4, 32, 24, false);
        window.observe(4, 32, 32, false); // This should trigger eviction

        // Should have evicted first entry (4 requests)
        // Remaining: 4 + 4 = 8, or if window exactly evicts: depends on implementation
        // With our logic: 4 + 4 + 4 = 12 > 10, so evict until <= 10
        // After evicting first: 4 + 4 = 8 <= 10
        assert!(window.num_requests() <= 10);

        // The hit rate should reflect only recent data
        // Second entry: 24/32 = 75%, Third: 32/32 = 100%
        // Combined: 56/64 = 87.5% (if both kept)
        assert!(window.hit_rate() > 0.5);
    }

    #[test]
    fn sliding_skips_empty_observations() {
        let mut window = SlidingWindowMetrics::new(100);

        window.observe(5, 40, 20, false);
        window.observe(0, 0, 0, false); // Should be skipped

        assert_eq!(window.num_requests(), 5);
        assert_eq!(window.tokens_queried(), 40);
    }

    #[test]
    fn sliding_reset_clears_all() {
        let mut window = SlidingWindowMetrics::new(100);

        window.observe(10, 80, 40, false);
        window.reset();

        assert!(window.is_empty());
        assert_eq!(window.num_requests(), 0);
        assert_eq!(window.tokens_queried(), 0);
        assert_eq!(window.tokens_hit(), 0);
    }

    #[test]
    fn sliding_observe_with_reset_flag() {
        let mut window = SlidingWindowMetrics::new(100);

        window.observe(5, 40, 20, false);
        window.observe(3, 24, 18, true); // Reset before adding

        // Should only have the second observation
        assert_eq!(window.num_requests(), 3);
        assert_eq!(window.tokens_queried(), 24);
        assert_eq!(window.tokens_hit(), 18);
    }

    #[test]
    fn sliding_snapshot() {
        let mut window = SlidingWindowMetrics::new(1000);

        window.observe(5, 40, 30, false);

        let snap = window.snapshot();
        assert_eq!(snap.num_requests, 5);
        assert_eq!(snap.tokens_queried, 40);
        assert_eq!(snap.tokens_hit, 30);
        assert!((snap.hit_rate - 0.75).abs() < 0.001);
        assert_eq!(snap.window_size, 1000);
    }

    #[test]
    fn sliding_preserves_last_entry_even_if_over_limit() {
        let mut window = SlidingWindowMetrics::new(5);

        // Single entry with 10 requests (over limit)
        window.observe(10, 80, 40, false);

        // Should keep this entry (we always keep at least one)
        assert_eq!(window.num_requests(), 10);
        assert!((window.hit_rate() - 0.5).abs() < 0.001);
    }

    // ─── Prometheus export tests ────────────────────────────────────────────────

    #[test]
    fn prometheus_metric_format() {
        let metric = PrometheusMetric {
            name: "test_metric".to_string(),
            help: "A test metric".to_string(),
            metric_type: PrometheusMetricType::Counter,
            value: 42.0,
            labels: vec![],
        };

        let output = metric.to_prometheus_format();
        assert!(output.contains("# HELP test_metric A test metric"));
        assert!(output.contains("# TYPE test_metric counter"));
        assert!(output.contains("test_metric 42"));
    }

    #[test]
    fn prometheus_metric_with_labels() {
        let metric = PrometheusMetric {
            name: "test_metric".to_string(),
            help: "A test metric".to_string(),
            metric_type: PrometheusMetricType::Gauge,
            value: 3.14,
            labels: vec![
                ("model".to_string(), "llama".to_string()),
                ("device".to_string(), "cuda:0".to_string()),
            ],
        };

        let output = metric.to_prometheus_format();
        assert!(output.contains("# TYPE test_metric gauge"));
        assert!(output.contains("test_metric{model=\"llama\",device=\"cuda:0\"} 3.14"));
    }

    #[test]
    fn export_prometheus_metrics_basic() {
        let stats = PrefixCacheStatsSnapshot {
            num_hits: 100,
            num_misses: 50,
            num_evictions: 10,
            num_queries: 30,
            tokens_queried: 1200,
            tokens_hit: 800,
            block_hit_rate: Some(0.666),
            token_hit_rate: Some(0.666),
        };

        let metrics = export_prometheus_metrics(&stats, None);

        // Should have at least the basic counter metrics
        assert!(metrics.len() >= 6);

        let names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"vllm_prefix_cache_hits_total"));
        assert!(names.contains(&"vllm_prefix_cache_misses_total"));
        assert!(names.contains(&"vllm_prefix_cache_evictions_total"));
    }

    #[test]
    fn export_prometheus_metrics_with_sliding() {
        let stats = PrefixCacheStatsSnapshot::default();
        let sliding = SlidingWindowSnapshot {
            num_requests: 100,
            tokens_queried: 8000,
            tokens_hit: 6000,
            hit_rate: 0.75,
            window_size: 1000,
        };

        let metrics = export_prometheus_metrics(&stats, Some(&sliding));

        let names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"vllm_prefix_cache_recent_hit_rate"));
        assert!(names.contains(&"vllm_prefix_cache_recent_requests"));
    }

    #[test]
    fn format_prometheus_output_complete() {
        let stats = PrefixCacheStatsSnapshot {
            num_hits: 50,
            num_misses: 50,
            num_evictions: 5,
            num_queries: 20,
            tokens_queried: 800,
            tokens_hit: 400,
            block_hit_rate: Some(0.5),
            token_hit_rate: Some(0.5),
        };

        let output = format_prometheus_output(&stats, None);

        assert!(output.contains("vllm_prefix_cache_hits_total 50"));
        assert!(output.contains("vllm_prefix_cache_misses_total 50"));
        assert!(output.contains("vllm_prefix_cache_block_hit_rate 0.5"));
    }

    #[test]
    fn default_implementations() {
        let stats = PrefixCacheStats::default();
        assert_eq!(stats.num_queries(), 0);

        let window = SlidingWindowMetrics::default();
        assert!(window.is_empty());

        let snap = PrefixCacheStatsSnapshot::default();
        assert_eq!(snap.num_hits, 0);
    }
}
