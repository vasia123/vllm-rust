//! Metrics for CPU offload operations.
//!
//! Tracks stores, loads, hits, misses, evictions, and prefetches
//! for monitoring CPU offload cache performance.

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe counters for CPU offload operations.
pub struct CpuOffloadMetrics {
    /// Total blocks stored to CPU from GPU.
    pub(crate) stores: AtomicU64,
    /// Total blocks loaded from CPU to GPU.
    pub(crate) loads: AtomicU64,
    /// Successful CPU cache lookups (block found in CPU cache).
    pub(crate) cpu_hits: AtomicU64,
    /// Failed CPU cache lookups (block not in CPU cache).
    pub(crate) cpu_misses: AtomicU64,
    /// Total CPU-side block evictions (LRU).
    pub(crate) evictions: AtomicU64,
    /// Total prefetch operations executed.
    pub(crate) prefetches: AtomicU64,
}

impl Default for CpuOffloadMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuOffloadMetrics {
    pub fn new() -> Self {
        Self {
            stores: AtomicU64::new(0),
            loads: AtomicU64::new(0),
            cpu_hits: AtomicU64::new(0),
            cpu_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            prefetches: AtomicU64::new(0),
        }
    }

    pub fn record_store(&self) {
        self.stores.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_load(&self) {
        self.loads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_hit(&self) {
        self.cpu_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.cpu_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefetch(&self) {
        self.prefetches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stores(&self) -> u64 {
        self.stores.load(Ordering::Relaxed)
    }

    pub fn loads(&self) -> u64 {
        self.loads.load(Ordering::Relaxed)
    }

    pub fn cpu_hits(&self) -> u64 {
        self.cpu_hits.load(Ordering::Relaxed)
    }

    pub fn cpu_misses(&self) -> u64 {
        self.cpu_misses.load(Ordering::Relaxed)
    }

    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    pub fn prefetches(&self) -> u64 {
        self.prefetches.load(Ordering::Relaxed)
    }

    /// CPU cache hit rate: hits / (hits + misses).
    ///
    /// Returns `None` if no lookups have been performed.
    pub fn hit_rate(&self) -> Option<f64> {
        let hits = self.cpu_hits.load(Ordering::Relaxed);
        let misses = self.cpu_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            None
        } else {
            Some(hits as f64 / total as f64)
        }
    }

    /// Get a snapshot of all metrics.
    pub fn snapshot(&self) -> CpuOffloadMetricsSnapshot {
        CpuOffloadMetricsSnapshot {
            stores: self.stores(),
            loads: self.loads(),
            cpu_hits: self.cpu_hits(),
            cpu_misses: self.cpu_misses(),
            evictions: self.evictions(),
            prefetches: self.prefetches(),
            hit_rate: self.hit_rate(),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.stores.store(0, Ordering::Relaxed);
        self.loads.store(0, Ordering::Relaxed);
        self.cpu_hits.store(0, Ordering::Relaxed);
        self.cpu_misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.prefetches.store(0, Ordering::Relaxed);
    }
}

/// Point-in-time snapshot of CPU offload metrics.
#[derive(Debug, Clone)]
pub struct CpuOffloadMetricsSnapshot {
    pub stores: u64,
    pub loads: u64,
    pub cpu_hits: u64,
    pub cpu_misses: u64,
    pub evictions: u64,
    pub prefetches: u64,
    pub hit_rate: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_are_zero() {
        let m = CpuOffloadMetrics::new();
        assert_eq!(m.stores(), 0);
        assert_eq!(m.loads(), 0);
        assert_eq!(m.cpu_hits(), 0);
        assert_eq!(m.cpu_misses(), 0);
        assert_eq!(m.evictions(), 0);
        assert_eq!(m.prefetches(), 0);
        assert_eq!(m.hit_rate(), None);
    }

    #[test]
    fn record_and_read() {
        let m = CpuOffloadMetrics::new();
        m.record_store();
        m.record_store();
        m.record_load();
        m.record_hit();
        m.record_hit();
        m.record_hit();
        m.record_miss();
        m.record_eviction();
        m.record_prefetch();
        m.record_prefetch();

        assert_eq!(m.stores(), 2);
        assert_eq!(m.loads(), 1);
        assert_eq!(m.cpu_hits(), 3);
        assert_eq!(m.cpu_misses(), 1);
        assert_eq!(m.evictions(), 1);
        assert_eq!(m.prefetches(), 2);
    }

    #[test]
    fn hit_rate_calculation() {
        let m = CpuOffloadMetrics::new();
        m.record_hit();
        m.record_hit();
        m.record_hit();
        m.record_miss();
        let rate = m.hit_rate().unwrap();
        assert!((rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn hit_rate_all_hits() {
        let m = CpuOffloadMetrics::new();
        m.record_hit();
        assert_eq!(m.hit_rate(), Some(1.0));
    }

    #[test]
    fn hit_rate_all_misses() {
        let m = CpuOffloadMetrics::new();
        m.record_miss();
        assert_eq!(m.hit_rate(), Some(0.0));
    }

    #[test]
    fn snapshot_captures_state() {
        let m = CpuOffloadMetrics::new();
        m.record_store();
        m.record_load();
        m.record_hit();
        m.record_miss();
        m.record_eviction();
        m.record_prefetch();

        let snap = m.snapshot();
        assert_eq!(snap.stores, 1);
        assert_eq!(snap.loads, 1);
        assert_eq!(snap.cpu_hits, 1);
        assert_eq!(snap.cpu_misses, 1);
        assert_eq!(snap.evictions, 1);
        assert_eq!(snap.prefetches, 1);
        assert_eq!(snap.hit_rate, Some(0.5));
    }

    #[test]
    fn reset_clears_all() {
        let m = CpuOffloadMetrics::new();
        m.record_store();
        m.record_load();
        m.record_hit();
        m.record_miss();
        m.record_eviction();
        m.record_prefetch();

        m.reset();

        assert_eq!(m.stores(), 0);
        assert_eq!(m.loads(), 0);
        assert_eq!(m.cpu_hits(), 0);
        assert_eq!(m.cpu_misses(), 0);
        assert_eq!(m.evictions(), 0);
        assert_eq!(m.prefetches(), 0);
        assert_eq!(m.hit_rate(), None);
    }

    #[test]
    fn default_impl() {
        let m = CpuOffloadMetrics::default();
        assert_eq!(m.stores(), 0);
    }

    #[test]
    fn thread_safe_concurrent_updates() {
        use std::sync::Arc;
        use std::thread;

        let m = Arc::new(CpuOffloadMetrics::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let mc = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    mc.record_store();
                    mc.record_load();
                    mc.record_hit();
                    mc.record_miss();
                    mc.record_eviction();
                    mc.record_prefetch();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(m.stores(), 1000);
        assert_eq!(m.loads(), 1000);
        assert_eq!(m.cpu_hits(), 1000);
        assert_eq!(m.cpu_misses(), 1000);
        assert_eq!(m.evictions(), 1000);
        assert_eq!(m.prefetches(), 1000);
    }
}
