//! Expert Parallel Load Balancing (EPLB) state machine.
//!
//! In expert parallelism, experts naturally become load-imbalanced: some experts receive
//! far more tokens than others during inference. EPLB addresses this by:
//! 1. Tracking per-expert token counts over a sliding window
//! 2. Signalling when rebalancing is beneficial (high imbalance ratio)
//! 3. (Future) Migrating expert weights to redistribute load
//!
//! ## Current Scope
//!
//! This module implements load tracking and rebalance detection. The actual weight
//! migration (`rearrange_expert_weights_inplace`) is **not yet implemented** — it
//! requires NCCL all-to-all transfers of model weight tensors, which depends on a
//! weight-streaming infrastructure not yet available.
//!
//! ## Reference
//!
//! `reference/vllm/vllm/distributed/eplb/eplb_state.py::EplbState::step()`

use std::collections::VecDeque;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the EPLB state machine.
#[derive(Debug, Clone)]
pub struct EplbConfig {
    /// Total number of experts (global count, not per-rank).
    pub num_experts: usize,
    /// Sliding window depth in steps. The load average is computed over the last
    /// `window_size` steps. Minimum 1.
    pub window_size: usize,
    /// Steps between rebalance checks. 0 means rebalancing is disabled entirely.
    /// When non-zero, `should_rebalance()` only returns `true` at multiples of
    /// this interval and when the imbalance ratio exceeds the threshold.
    pub rebalance_interval: usize,
    /// Load imbalance ratio above which rebalancing is triggered.
    /// Ratio = max_load / mean_load. Default threshold: 2.0 (max expert gets 2×
    /// the average tokens). Must be ≥ 1.0.
    pub imbalance_threshold: f64,
}

impl EplbConfig {
    /// Create a default configuration with sensible parameters.
    ///
    /// - `window_size = 10` steps
    /// - `rebalance_interval = 100` steps
    /// - `imbalance_threshold = 2.0` (trigger when hottest expert gets 2× the mean)
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            window_size: 10,
            rebalance_interval: 100,
            imbalance_threshold: 2.0,
        }
    }

    /// Create a configuration with rebalancing disabled.
    pub fn disabled(num_experts: usize) -> Self {
        Self {
            num_experts,
            window_size: 10,
            rebalance_interval: 0,
            imbalance_threshold: 2.0,
        }
    }
}

// ─── ExpertLoadStats ──────────────────────────────────────────────────────────

/// Per-step expert utilisation: token counts routed to each expert.
#[derive(Debug, Clone)]
pub struct ExpertLoadStats {
    /// Token count per expert (indexed by global expert ID).
    pub token_counts: Vec<u64>,
}

impl ExpertLoadStats {
    fn zeros(num_experts: usize) -> Self {
        Self {
            token_counts: vec![0; num_experts],
        }
    }
}

// ─── EplbState ────────────────────────────────────────────────────────────────

/// Expert Parallel Load Balancing state machine.
///
/// Maintains a sliding window of per-step expert utilisation and exposes
/// helpers to query load imbalance and decide when to trigger rebalancing.
///
/// ## Usage pattern
///
/// After each forward pass, call [`record_expert_usage`][Self::record_expert_usage]
/// for every token's top-k expert assignments, then call [`step`][Self::step] to
/// advance the window. Query [`should_rebalance`][Self::should_rebalance] to decide
/// whether to trigger weight rearrangement.
///
/// ```ignore
/// let mut eplb = EplbState::new(EplbConfig::new(8));
///
/// // After forward pass:
/// eplb.record_expert_usage(&[0, 3, 0, 7], None); // 4 tokens, expert IDs
/// eplb.step();
///
/// if eplb.should_rebalance() {
///     // TODO: migrate weights
/// }
/// ```
pub struct EplbState {
    config: EplbConfig,
    /// Sliding window of past step loads. Front = oldest, back = newest.
    window: VecDeque<ExpertLoadStats>,
    /// Accumulator for the current (not yet committed) step.
    current: ExpertLoadStats,
    /// Total completed steps.
    step_count: usize,
}

impl EplbState {
    /// Create a new EPLB state machine.
    ///
    /// # Panics
    /// Panics if `config.num_experts == 0` or `config.window_size == 0`.
    pub fn new(config: EplbConfig) -> Self {
        assert!(config.num_experts > 0, "EPLB num_experts must be > 0");
        assert!(config.window_size > 0, "EPLB window_size must be > 0");
        assert!(
            config.imbalance_threshold >= 1.0,
            "EPLB imbalance_threshold must be >= 1.0, got {}",
            config.imbalance_threshold
        );
        let current = ExpertLoadStats::zeros(config.num_experts);
        Self {
            config,
            window: VecDeque::new(),
            current,
            step_count: 0,
        }
    }

    /// Record expert token assignments for the current step.
    ///
    /// `expert_ids` contains the global expert ID for each dispatched token.
    /// `weights` (optional) contains a routing weight per token; when `None`,
    /// each token contributes 1.0 to the load count. Fractional weights allow
    /// tracking effective load when top-k routing uses non-uniform weights.
    pub fn record_expert_usage(&mut self, expert_ids: &[usize], weights: Option<&[f64]>) {
        debug_assert!(
            weights.is_none_or(|w| w.len() == expert_ids.len()),
            "weights length must match expert_ids length"
        );

        for (i, &expert_id) in expert_ids.iter().enumerate() {
            if expert_id < self.config.num_experts {
                let inc = weights.map_or(1, |w| w[i].round() as u64);
                self.current.token_counts[expert_id] += inc;
            }
        }
    }

    /// Commit the current step's stats into the sliding window and reset the accumulator.
    ///
    /// Call once after each forward pass, after all [`record_expert_usage`][Self::record_expert_usage]
    /// calls for that pass.
    pub fn step(&mut self) {
        // Push current step into the window, evicting the oldest if full.
        let committed = std::mem::replace(
            &mut self.current,
            ExpertLoadStats::zeros(self.config.num_experts),
        );
        self.window.push_back(committed);
        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }
        self.step_count += 1;
    }

    /// Compute the mean token count per expert over the sliding window.
    ///
    /// Returns a vector of length `num_experts` with the average token count.
    /// Returns zeros if no steps have been committed yet.
    pub fn current_load(&self) -> Vec<f64> {
        let n = self.config.num_experts;
        if self.window.is_empty() {
            return vec![0.0; n];
        }
        let mut sums = vec![0u64; n];
        for step in &self.window {
            for (s, &c) in sums.iter_mut().zip(step.token_counts.iter()) {
                *s += c;
            }
        }
        let steps = self.window.len() as f64;
        sums.into_iter().map(|s| s as f64 / steps).collect()
    }

    /// Compute the load imbalance ratio: `max_load / mean_load`.
    ///
    /// Returns `1.0` (perfect balance) when there is no data or when all
    /// experts receive exactly the same number of tokens.
    /// Returns `1.0` when mean_load is zero (no tokens yet).
    pub fn max_load_imbalance(&self) -> f64 {
        let load = self.current_load();
        let mean = load.iter().sum::<f64>() / load.len() as f64;
        if mean < 1e-9 {
            return 1.0;
        }
        let max = load.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max / mean
    }

    /// Returns `true` when load rebalancing is recommended.
    ///
    /// Conditions (all must hold):
    /// - `rebalance_interval > 0` (rebalancing is enabled)
    /// - `step_count` is a positive multiple of `rebalance_interval`
    /// - `max_load_imbalance() > imbalance_threshold`
    ///
    /// # Note
    ///
    /// Calling this does NOT trigger any weight migration. The actual rebalancing
    /// must be performed externally once this method returns `true`.
    ///
    /// TODO: Implement `rearrange_weights()` once NCCL weight-streaming is available.
    pub fn should_rebalance(&self) -> bool {
        let interval = self.config.rebalance_interval;
        if interval == 0 {
            return false;
        }
        if self.step_count == 0 || !self.step_count.is_multiple_of(interval) {
            return false;
        }
        self.max_load_imbalance() > self.config.imbalance_threshold
    }

    /// Total number of completed steps since creation.
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_eplb(num_experts: usize) -> EplbState {
        EplbState::new(EplbConfig {
            num_experts,
            window_size: 5,
            rebalance_interval: 10,
            imbalance_threshold: 2.0,
        })
    }

    #[test]
    fn test_eplb_new() {
        let eplb = make_eplb(8);
        assert_eq!(eplb.step_count(), 0);
        let load = eplb.current_load();
        assert_eq!(load.len(), 8);
        assert!(load.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_eplb_record_usage_single_step() {
        let mut eplb = make_eplb(4);
        // Route 3 tokens to expert 0, 1 to expert 2
        eplb.record_expert_usage(&[0, 0, 2, 0], None);
        eplb.step();

        let load = eplb.current_load();
        assert_eq!(load[0], 3.0);
        assert_eq!(load[1], 0.0);
        assert_eq!(load[2], 1.0);
        assert_eq!(load[3], 0.0);
    }

    #[test]
    fn test_eplb_step_advances_window() {
        let mut eplb = make_eplb(2);

        // Step 1: all tokens to expert 0
        eplb.record_expert_usage(&[0, 0], None);
        eplb.step();

        // Step 2: all tokens to expert 1
        eplb.record_expert_usage(&[1, 1], None);
        eplb.step();

        assert_eq!(eplb.step_count(), 2);
        // Average over 2 steps: expert 0 → 1.0, expert 1 → 1.0
        let load = eplb.current_load();
        assert!((load[0] - 1.0).abs() < 1e-9);
        assert!((load[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_eplb_window_eviction() {
        let mut eplb = EplbState::new(EplbConfig {
            num_experts: 2,
            window_size: 2, // only 2 steps kept
            rebalance_interval: 0,
            imbalance_threshold: 2.0,
        });

        // Step 1: heavy expert 0
        eplb.record_expert_usage(&[0; 10], None);
        eplb.step();
        // Step 2: heavy expert 0
        eplb.record_expert_usage(&[0; 10], None);
        eplb.step();
        // Step 3: heavy expert 1 (should push step 1 out of window)
        eplb.record_expert_usage(&[1; 10], None);
        eplb.step();

        // Window (size=2) now contains steps 2 and 3:
        //   step 2: expert0=10, expert1=0
        //   step 3: expert0=0,  expert1=10
        // Average: expert0 = (10+0)/2 = 5.0, expert1 = (0+10)/2 = 5.0
        let load = eplb.current_load();
        assert!((load[0] - 5.0).abs() < 1e-9, "load[0]={}", load[0]);
        assert!((load[1] - 5.0).abs() < 1e-9, "load[1]={}", load[1]);
    }

    #[test]
    fn test_eplb_current_load_empty() {
        let eplb = make_eplb(4);
        let load = eplb.current_load();
        assert!(load.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_eplb_max_load_imbalance_balanced() {
        let mut eplb = make_eplb(4);
        // All 4 experts get exactly 1 token each → perfectly balanced
        eplb.record_expert_usage(&[0, 1, 2, 3], None);
        eplb.step();

        let ratio = eplb.max_load_imbalance();
        assert!((ratio - 1.0).abs() < 1e-9, "ratio={ratio}");
    }

    #[test]
    fn test_eplb_max_load_imbalance_unbalanced() {
        let mut eplb = make_eplb(4);
        // Expert 0 gets 8 tokens, experts 1-3 get 1 each → mean=11/4=2.75, max=8
        eplb.record_expert_usage(&[0; 8], None);
        eplb.record_expert_usage(&[1, 2, 3], None);
        eplb.step();

        let ratio = eplb.max_load_imbalance();
        // mean = (8+1+1+1)/4 = 2.75, max/mean = 8/2.75 ≈ 2.909
        assert!(ratio > 2.0, "ratio={ratio}");
    }

    #[test]
    fn test_eplb_should_rebalance_threshold() {
        let mut eplb = EplbState::new(EplbConfig {
            num_experts: 4,
            window_size: 5,
            rebalance_interval: 1, // trigger every step
            imbalance_threshold: 2.0,
        });
        // Heavy imbalance: expert 0 gets 100 tokens, others get 1
        eplb.record_expert_usage(&[0; 100], None);
        eplb.record_expert_usage(&[1, 2, 3], None);
        eplb.step(); // step_count = 1, which is a multiple of interval=1

        assert!(eplb.should_rebalance());
    }

    #[test]
    fn test_eplb_should_rebalance_below_threshold() {
        let mut eplb = EplbState::new(EplbConfig {
            num_experts: 4,
            window_size: 5,
            rebalance_interval: 1,
            imbalance_threshold: 10.0, // very high threshold
        });
        // Mild imbalance: ratio ≈ 2.9, below threshold of 10.0
        eplb.record_expert_usage(&[0; 8], None);
        eplb.record_expert_usage(&[1, 2, 3], None);
        eplb.step();

        assert!(!eplb.should_rebalance());
    }

    #[test]
    fn test_eplb_rebalance_interval_disabled() {
        let mut eplb = EplbState::new(EplbConfig {
            num_experts: 4,
            window_size: 5,
            rebalance_interval: 0, // disabled
            imbalance_threshold: 1.0,
        });
        // Even with extreme imbalance, should_rebalance() must return false
        eplb.record_expert_usage(&[0; 1000], None);
        eplb.step();

        assert!(!eplb.should_rebalance());
    }

    #[test]
    fn test_eplb_rebalance_interval_not_at_boundary() {
        let mut eplb = EplbState::new(EplbConfig {
            num_experts: 2,
            window_size: 5,
            rebalance_interval: 5, // only at steps 5, 10, 15, ...
            imbalance_threshold: 1.5,
        });
        // Run 3 unbalanced steps — step_count=3, not a multiple of 5
        for _ in 0..3 {
            eplb.record_expert_usage(&[0; 10], None);
            eplb.step();
        }
        assert!(!eplb.should_rebalance(), "should not rebalance at step 3");
    }
}
