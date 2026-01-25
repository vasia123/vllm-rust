//! KV cache metrics for monitoring and debugging.
//!
//! Tracks allocation, eviction, and prefix cache hit rates.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Statistics for KV cache operations.
///
/// Thread-safe counters for monitoring cache performance.
pub struct KVCacheMetrics {
    /// Total allocation requests
    allocations: AtomicU64,
    /// Total blocks allocated
    blocks_allocated: AtomicU64,
    /// Total blocks freed
    blocks_freed: AtomicU64,
    /// Total blocks evicted from prefix cache
    blocks_evicted: AtomicU64,
    /// Prefix cache hits (blocks reused)
    cache_hits: AtomicU64,
    /// Prefix cache misses (blocks computed)
    cache_misses: AtomicU64,
    /// Total queries to prefix cache
    cache_queries: AtomicU64,
}

impl Default for KVCacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCacheMetrics {
    pub fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            blocks_allocated: AtomicU64::new(0),
            blocks_freed: AtomicU64::new(0),
            blocks_evicted: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_queries: AtomicU64::new(0),
        }
    }

    /// Record a block allocation event.
    pub fn record_allocation(&self, num_blocks: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.blocks_allocated
            .fetch_add(num_blocks as u64, Ordering::Relaxed);
    }

    /// Record blocks being freed.
    pub fn record_free(&self, num_blocks: usize) {
        self.blocks_freed
            .fetch_add(num_blocks as u64, Ordering::Relaxed);
    }

    /// Record blocks evicted from prefix cache.
    pub fn record_eviction(&self, num_blocks: usize) {
        self.blocks_evicted
            .fetch_add(num_blocks as u64, Ordering::Relaxed);
    }

    /// Record a prefix cache query result.
    pub fn record_cache_query(&self, hits: usize, misses: usize) {
        self.cache_queries.fetch_add(1, Ordering::Relaxed);
        self.cache_hits.fetch_add(hits as u64, Ordering::Relaxed);
        self.cache_misses
            .fetch_add(misses as u64, Ordering::Relaxed);
    }

    /// Get total number of allocation requests.
    pub fn allocations(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Get total blocks allocated.
    pub fn blocks_allocated(&self) -> u64 {
        self.blocks_allocated.load(Ordering::Relaxed)
    }

    /// Get total blocks freed.
    pub fn blocks_freed(&self) -> u64 {
        self.blocks_freed.load(Ordering::Relaxed)
    }

    /// Get total blocks evicted.
    pub fn blocks_evicted(&self) -> u64 {
        self.blocks_evicted.load(Ordering::Relaxed)
    }

    /// Get total cache hits (blocks reused from prefix cache).
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Get total cache misses (blocks that needed computation).
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Get total prefix cache queries.
    pub fn cache_queries(&self) -> u64 {
        self.cache_queries.load(Ordering::Relaxed)
    }

    /// Calculate prefix cache hit rate (0.0 to 1.0).
    ///
    /// Returns `None` if no cache queries have been made.
    pub fn hit_rate(&self) -> Option<f64> {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            None
        } else {
            Some(hits as f64 / total as f64)
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.allocations.store(0, Ordering::Relaxed);
        self.blocks_allocated.store(0, Ordering::Relaxed);
        self.blocks_freed.store(0, Ordering::Relaxed);
        self.blocks_evicted.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.cache_queries.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            allocations: self.allocations(),
            blocks_allocated: self.blocks_allocated(),
            blocks_freed: self.blocks_freed(),
            blocks_evicted: self.blocks_evicted(),
            cache_hits: self.cache_hits(),
            cache_misses: self.cache_misses(),
            cache_queries: self.cache_queries(),
            hit_rate: self.hit_rate(),
        }
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MetricsSnapshot {
    pub allocations: u64,
    pub blocks_allocated: u64,
    pub blocks_freed: u64,
    pub blocks_evicted: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_queries: u64,
    pub hit_rate: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_are_zero() {
        let metrics = KVCacheMetrics::new();
        assert_eq!(metrics.allocations(), 0);
        assert_eq!(metrics.blocks_allocated(), 0);
        assert_eq!(metrics.blocks_freed(), 0);
        assert_eq!(metrics.blocks_evicted(), 0);
        assert_eq!(metrics.cache_hits(), 0);
        assert_eq!(metrics.cache_misses(), 0);
        assert_eq!(metrics.cache_queries(), 0);
        assert_eq!(metrics.hit_rate(), None);
    }

    #[test]
    fn record_allocation() {
        let metrics = KVCacheMetrics::new();
        metrics.record_allocation(3);
        assert_eq!(metrics.allocations(), 1);
        assert_eq!(metrics.blocks_allocated(), 3);

        metrics.record_allocation(5);
        assert_eq!(metrics.allocations(), 2);
        assert_eq!(metrics.blocks_allocated(), 8);
    }

    #[test]
    fn record_free() {
        let metrics = KVCacheMetrics::new();
        metrics.record_free(2);
        metrics.record_free(3);
        assert_eq!(metrics.blocks_freed(), 5);
    }

    #[test]
    fn record_eviction() {
        let metrics = KVCacheMetrics::new();
        metrics.record_eviction(4);
        assert_eq!(metrics.blocks_evicted(), 4);
    }

    #[test]
    fn record_cache_query_all_hits() {
        let metrics = KVCacheMetrics::new();
        metrics.record_cache_query(5, 0); // 5 hits, 0 misses
        assert_eq!(metrics.cache_queries(), 1);
        assert_eq!(metrics.cache_hits(), 5);
        assert_eq!(metrics.cache_misses(), 0);
        assert_eq!(metrics.hit_rate(), Some(1.0));
    }

    #[test]
    fn record_cache_query_all_misses() {
        let metrics = KVCacheMetrics::new();
        metrics.record_cache_query(0, 5); // 0 hits, 5 misses
        assert_eq!(metrics.cache_queries(), 1);
        assert_eq!(metrics.cache_hits(), 0);
        assert_eq!(metrics.cache_misses(), 5);
        assert_eq!(metrics.hit_rate(), Some(0.0));
    }

    #[test]
    fn record_cache_query_mixed() {
        let metrics = KVCacheMetrics::new();
        metrics.record_cache_query(3, 2); // 3 hits, 2 misses
        assert_eq!(metrics.cache_queries(), 1);
        assert_eq!(metrics.cache_hits(), 3);
        assert_eq!(metrics.cache_misses(), 2);
        let rate = metrics.hit_rate().unwrap();
        assert!((rate - 0.6).abs() < 0.001); // 3/5 = 0.6
    }

    #[test]
    fn hit_rate_accumulates_across_queries() {
        let metrics = KVCacheMetrics::new();
        metrics.record_cache_query(2, 2); // 50%
        metrics.record_cache_query(4, 0); // 100%
                                          // Total: 6 hits, 2 misses = 75%
        assert_eq!(metrics.cache_hits(), 6);
        assert_eq!(metrics.cache_misses(), 2);
        let rate = metrics.hit_rate().unwrap();
        assert!((rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn reset_clears_all() {
        let metrics = KVCacheMetrics::new();
        metrics.record_allocation(5);
        metrics.record_free(2);
        metrics.record_eviction(1);
        metrics.record_cache_query(3, 1);

        metrics.reset();

        assert_eq!(metrics.allocations(), 0);
        assert_eq!(metrics.blocks_allocated(), 0);
        assert_eq!(metrics.blocks_freed(), 0);
        assert_eq!(metrics.blocks_evicted(), 0);
        assert_eq!(metrics.cache_hits(), 0);
        assert_eq!(metrics.cache_misses(), 0);
        assert_eq!(metrics.cache_queries(), 0);
    }

    #[test]
    fn snapshot_captures_state() {
        let metrics = KVCacheMetrics::new();
        metrics.record_allocation(10);
        metrics.record_free(3);
        metrics.record_eviction(2);
        metrics.record_cache_query(5, 5);

        let snap = metrics.snapshot();
        assert_eq!(snap.allocations, 1);
        assert_eq!(snap.blocks_allocated, 10);
        assert_eq!(snap.blocks_freed, 3);
        assert_eq!(snap.blocks_evicted, 2);
        assert_eq!(snap.cache_hits, 5);
        assert_eq!(snap.cache_misses, 5);
        assert_eq!(snap.cache_queries, 1);
        assert_eq!(snap.hit_rate, Some(0.5));
    }

    #[test]
    fn thread_safe_concurrent_updates() {
        use std::sync::Arc;
        use std::thread;

        let metrics = Arc::new(KVCacheMetrics::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let m = Arc::clone(&metrics);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    m.record_allocation(1);
                    m.record_cache_query(1, 1);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(metrics.allocations(), 1000);
        assert_eq!(metrics.blocks_allocated(), 1000);
        assert_eq!(metrics.cache_queries(), 1000);
        assert_eq!(metrics.cache_hits(), 1000);
        assert_eq!(metrics.cache_misses(), 1000);
    }

    #[test]
    fn default_impl() {
        let metrics = KVCacheMetrics::default();
        assert_eq!(metrics.allocations(), 0);
    }
}
