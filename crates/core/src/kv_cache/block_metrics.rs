//! Block-level lifecycle metrics for debugging and profiling.
//!
//! Tracks individual block allocation, access patterns, and eviction events.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::block_pool::BlockId;

/// Per-block state for metrics tracking.
#[derive(Debug)]
pub struct BlockMetricsState {
    /// When the block was allocated
    birth_time: Instant,
    /// Last access timestamp
    last_access: Instant,
    /// Ring buffer of recent access times (max 4)
    access_history: VecDeque<Instant>,
}

const MAX_ACCESS_HISTORY: usize = 4;

impl BlockMetricsState {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            birth_time: now,
            last_access: now,
            access_history: VecDeque::with_capacity(MAX_ACCESS_HISTORY),
        }
    }

    /// Record an access to this block.
    pub fn record_access(&mut self) {
        let now = Instant::now();
        self.last_access = now;

        if self.access_history.len() >= MAX_ACCESS_HISTORY {
            self.access_history.pop_front();
        }
        self.access_history.push_back(now);
    }

    /// Get block lifetime in seconds (from allocation to now).
    pub fn lifetime_seconds(&self) -> f64 {
        self.birth_time.elapsed().as_secs_f64()
    }

    /// Get idle time in seconds (from last access to now).
    pub fn idle_seconds(&self) -> f64 {
        self.last_access.elapsed().as_secs_f64()
    }

    /// Get gaps between consecutive accesses in seconds.
    pub fn reuse_gaps_seconds(&self) -> Vec<f64> {
        if self.access_history.len() < 2 {
            return Vec::new();
        }

        let mut gaps = Vec::with_capacity(self.access_history.len() - 1);
        let history: Vec<_> = self.access_history.iter().collect();

        for i in 1..history.len() {
            let gap = history[i].duration_since(*history[i - 1]).as_secs_f64();
            gaps.push(gap);
        }
        gaps
    }

    /// Number of recorded accesses.
    pub fn access_count(&self) -> usize {
        self.access_history.len()
    }
}

impl Default for BlockMetricsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Event emitted when a block is evicted.
#[derive(Debug, Clone)]
pub struct BlockEvictionEvent {
    pub block_id: BlockId,
    pub lifetime_seconds: f64,
    pub idle_seconds: f64,
    pub reuse_gaps_seconds: Vec<f64>,
    pub access_count: usize,
}

/// Collector for block-level metrics with sampling support.
pub struct BlockMetricsCollector {
    /// Sample rate (0.0 to 1.0)
    sample_rate: f64,
    /// Per-block metrics state
    block_metrics: HashMap<BlockId, BlockMetricsState>,
    /// Pending eviction events
    eviction_events: Vec<BlockEvictionEvent>,
    /// RNG state for sampling
    sample_counter: u64,
}

impl BlockMetricsCollector {
    /// Create a new collector with given sample rate.
    ///
    /// # Panics
    /// Panics if sample_rate is not in (0.0, 1.0].
    pub fn new(sample_rate: f64) -> Self {
        assert!(
            sample_rate > 0.0 && sample_rate <= 1.0,
            "sample_rate must be in (0.0, 1.0], got {}",
            sample_rate
        );
        Self {
            sample_rate,
            block_metrics: HashMap::new(),
            eviction_events: Vec::new(),
            sample_counter: 0,
        }
    }

    /// Determine if we should sample the next block.
    pub fn should_sample(&mut self) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        // Simple deterministic sampling based on counter
        self.sample_counter = self.sample_counter.wrapping_add(1);
        let threshold = (self.sample_rate * u64::MAX as f64) as u64;
        // Use a simple hash-like mixing
        let mixed = self.sample_counter.wrapping_mul(0x9e3779b97f4a7c15);
        mixed < threshold
    }

    /// Record block allocation (if sampled).
    pub fn on_block_allocated(&mut self, block_id: BlockId) {
        if self.should_sample() {
            self.block_metrics
                .insert(block_id, BlockMetricsState::new());
        }
    }

    /// Record block access.
    pub fn on_block_accessed(&mut self, block_id: BlockId) {
        if let Some(state) = self.block_metrics.get_mut(&block_id) {
            state.record_access();
        }
    }

    /// Record block eviction, generating an event.
    pub fn on_block_evicted(&mut self, block_id: BlockId) {
        if let Some(state) = self.block_metrics.remove(&block_id) {
            self.eviction_events.push(BlockEvictionEvent {
                block_id,
                lifetime_seconds: state.lifetime_seconds(),
                idle_seconds: state.idle_seconds(),
                reuse_gaps_seconds: state.reuse_gaps_seconds(),
                access_count: state.access_count(),
            });
        }
    }

    /// Drain all pending eviction events.
    pub fn drain_events(&mut self) -> Vec<BlockEvictionEvent> {
        std::mem::take(&mut self.eviction_events)
    }

    /// Number of blocks currently being tracked.
    pub fn tracked_blocks(&self) -> usize {
        self.block_metrics.len()
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.block_metrics.clear();
        self.eviction_events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn block_state_init() {
        let state = BlockMetricsState::new();
        assert!(state.lifetime_seconds() >= 0.0);
        assert!(state.idle_seconds() >= 0.0);
        assert_eq!(state.access_count(), 0);
        assert!(state.reuse_gaps_seconds().is_empty());
    }

    #[test]
    fn block_state_access_tracking() {
        let mut state = BlockMetricsState::new();
        thread::sleep(Duration::from_millis(10));

        state.record_access();
        assert_eq!(state.access_count(), 1);
        assert!(state.idle_seconds() < state.lifetime_seconds());
    }

    #[test]
    fn block_state_ring_buffer_wraps_at_4() {
        let mut state = BlockMetricsState::new();

        for _ in 0..6 {
            state.record_access();
        }

        assert_eq!(state.access_count(), MAX_ACCESS_HISTORY);
    }

    #[test]
    fn block_state_reuse_gaps() {
        let mut state = BlockMetricsState::new();

        // Record accesses with small delays
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(5));
            state.record_access();
        }

        let gaps = state.reuse_gaps_seconds();
        assert_eq!(gaps.len(), 2);
        for gap in gaps {
            assert!(gap >= 0.004); // at least 4ms due to sleep
        }
    }

    #[test]
    fn collector_sample_rate_validation() {
        // Valid rates
        let _ = BlockMetricsCollector::new(0.5);
        let _ = BlockMetricsCollector::new(1.0);
        let _ = BlockMetricsCollector::new(0.001);
    }

    #[test]
    #[should_panic(expected = "sample_rate must be in (0.0, 1.0]")]
    fn collector_invalid_sample_rate_zero() {
        let _ = BlockMetricsCollector::new(0.0);
    }

    #[test]
    #[should_panic(expected = "sample_rate must be in (0.0, 1.0]")]
    fn collector_invalid_sample_rate_negative() {
        let _ = BlockMetricsCollector::new(-0.5);
    }

    #[test]
    #[should_panic(expected = "sample_rate must be in (0.0, 1.0]")]
    fn collector_invalid_sample_rate_above_one() {
        let _ = BlockMetricsCollector::new(1.5);
    }

    #[test]
    fn collector_full_sampling() {
        let mut c = BlockMetricsCollector::new(1.0);
        let samples: usize = (0..100).filter(|_| c.should_sample()).count();
        assert_eq!(samples, 100);
    }

    #[test]
    fn collector_partial_sampling() {
        let mut c = BlockMetricsCollector::new(0.5);
        let samples: usize = (0..1000).filter(|_| c.should_sample()).count();
        // Should be roughly 50% ± 10%
        assert!(samples > 400 && samples < 600, "samples = {}", samples);
    }

    #[test]
    fn collector_allocation() {
        let mut c = BlockMetricsCollector::new(1.0);

        for i in 0..5 {
            c.on_block_allocated(i);
        }

        assert_eq!(c.tracked_blocks(), 5);
    }

    #[test]
    fn collector_access() {
        let mut c = BlockMetricsCollector::new(1.0);
        c.on_block_allocated(0);

        c.on_block_accessed(0);
        c.on_block_accessed(0);
        c.on_block_accessed(0);

        // Verify access was recorded (check via eviction event)
        c.on_block_evicted(0);
        let events = c.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].access_count, 3);
    }

    #[test]
    fn collector_eviction_no_accesses() {
        let mut c = BlockMetricsCollector::new(1.0);
        c.on_block_allocated(0);
        thread::sleep(Duration::from_millis(10));
        c.on_block_evicted(0);

        let events = c.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].block_id, 0);
        assert_eq!(events[0].access_count, 0);
        // lifetime ≈ idle when no accesses
        assert!((events[0].lifetime_seconds - events[0].idle_seconds).abs() < 0.001);
    }

    #[test]
    fn collector_eviction_with_accesses() {
        let mut c = BlockMetricsCollector::new(1.0);
        c.on_block_allocated(0);

        thread::sleep(Duration::from_millis(5));
        c.on_block_accessed(0);
        thread::sleep(Duration::from_millis(5));
        c.on_block_accessed(0);
        thread::sleep(Duration::from_millis(5));
        c.on_block_evicted(0);

        let events = c.drain_events();
        assert_eq!(events.len(), 1);

        let e = &events[0];
        assert!(e.lifetime_seconds > 0.01); // at least 15ms
        assert!(e.idle_seconds < e.lifetime_seconds);
        assert_eq!(e.access_count, 2);
        assert_eq!(e.reuse_gaps_seconds.len(), 1);
    }

    #[test]
    fn collector_reset() {
        let mut c = BlockMetricsCollector::new(1.0);

        for i in 0..5 {
            c.on_block_allocated(i);
        }
        assert_eq!(c.tracked_blocks(), 5);

        c.reset();
        assert_eq!(c.tracked_blocks(), 0);

        c.on_block_allocated(10);
        assert_eq!(c.tracked_blocks(), 1);
    }

    #[test]
    fn collector_drain_events_clears() {
        let mut c = BlockMetricsCollector::new(1.0);
        c.on_block_allocated(0);
        c.on_block_evicted(0);

        assert_eq!(c.drain_events().len(), 1);
        assert_eq!(c.drain_events().len(), 0); // already drained
    }

    #[test]
    fn collector_untracked_block_access_ignored() {
        let mut c = BlockMetricsCollector::new(1.0);
        // Access block that was never allocated (or not sampled)
        c.on_block_accessed(999);
        c.on_block_evicted(999);
        // Should not crash, no events generated
        assert!(c.drain_events().is_empty());
    }
}
