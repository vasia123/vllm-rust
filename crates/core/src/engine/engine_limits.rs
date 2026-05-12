//! Process-global runtime limits exposed to hot-path pool sizing.
//!
//! Background: pool buffers used on the CUDA Graph capture replay path
//! (paged_attention V2 partitions, block_tables stride) need a constant
//! shape across forwards. The right upper bound is `--max-model-len`,
//! which is a CLI flag at the server, but the pool sites are deep inside
//! `vllm-core` and the value would need to thread through 200+
//! `CacheConfig` literals to reach them.
//!
//! ## API surface
//!
//! There are two layers:
//!
//! * Low-level `set_max_model_len` / `set_max_seq_len_to_capture` /
//!   `pool_worst_case_seq_len` — direct `OnceLock` access used by the
//!   pool sites that read these values from anywhere in the code base.
//!   Idempotent: only the first `set_*` call wins.
//!
//! * High-level [`EngineLimits`] + [`EngineLimitsBuilder`] —
//!   phase-typed witness consumed by every `start_engine*` entry.
//!   Constructing one publishes both `OnceLock`s as a side effect, so
//!   the compiler enforces "limits are set before the engine starts".
//!   This catches the Bug B.2 regression class (limits set AFTER
//!   `start_engine`, with capture warmup falling back to the 1024
//!   default and recording a stale block_tables stride into the
//!   captured CUDA graph) at compile time.

use std::sync::OnceLock;

static MAX_MODEL_LEN: OnceLock<usize> = OnceLock::new();
static MAX_SEQ_LEN_TO_CAPTURE: OnceLock<usize> = OnceLock::new();

/// Set the engine-wide `--max-model-len`. Idempotent — only the first
/// call wins. Safe to call multiple times with the same value.
///
/// Prefer constructing an [`EngineLimits`] via [`EngineLimitsBuilder`]
/// from the server entry point; this raw setter exists for crates that
/// embed `vllm-core` without going through the standard `start_engine`
/// path (older tests, ad-hoc benches).
pub fn set_max_model_len(value: usize) {
    let _ = MAX_MODEL_LEN.set(value);
}

/// Get the engine-wide `--max-model-len`. `None` until `set_max_model_len`
/// is called; pool sites should fall back to a conservative default
/// (1024) when `None`.
pub fn max_model_len() -> Option<usize> {
    MAX_MODEL_LEN.get().copied()
}

/// Set the engine-wide `--max-seq-len-to-capture`. Smaller than
/// `--max-model-len` and used specifically to size CUDA-graph capture
/// pool buffers (paged_attention V2 partitions, block_tables stride):
/// supporting the full model-len worst case would balloon scratch
/// memset costs (e.g. 131072 / 256 = 512 partitions for Llama 3.2)
/// even when actual workloads stay well under that. Idempotent —
/// only the first call wins.
pub fn set_max_seq_len_to_capture(value: usize) {
    let _ = MAX_SEQ_LEN_TO_CAPTURE.set(value);
}

/// Get the engine-wide `--max-seq-len-to-capture`. `None` until set;
/// pool sites should fall back to [`max_model_len`] (capture must size
/// at least for the longest sequence the engine might serve), then to
/// the conservative 1024 default.
pub fn max_seq_len_to_capture() -> Option<usize> {
    MAX_SEQ_LEN_TO_CAPTURE.get().copied()
}

/// Worst-case sequence length used for stable-shape pool reservations
/// inside captured graphs. Prefers the smaller `max_seq_len_to_capture`
/// (cheap memsets) and only falls through to `max_model_len` when the
/// server didn't publish a separate capture cap, or to `1024` for
/// crates that use vllm-core without either hook (tests / benches).
pub fn pool_worst_case_seq_len() -> usize {
    max_seq_len_to_capture()
        .or_else(max_model_len)
        .unwrap_or(1024)
}

// ─── Phase-typed witness ─────────────────────────────────────────────

/// Engine-wide pool sizing limits. Constructing one publishes the
/// values to the process-global `OnceLock`s read by all hot-path pool
/// sites (`crate::layers::attention::block`, the quantized model
/// forwards), so passing an `EngineLimits` to `start_engine*` is the
/// compile-time witness that the OnceLock state has been initialised
/// before the engine's capture-warmup forwards run.
///
/// Construct via [`EngineLimitsBuilder`] in production. Tests and
/// benches that don't care about the capture path may use
/// [`EngineLimits::for_testing`] which publishes the historical 1024
/// fallback. There is no `Default` impl on purpose — silently defaulting
/// the limits would defeat the point of the witness type.
#[derive(Debug, Clone, Copy)]
pub struct EngineLimits {
    max_model_len: usize,
    max_seq_len_to_capture: usize,
}

impl EngineLimits {
    /// Witness an already-published global limit state. Returns
    /// `Some(limits)` reflecting the current `OnceLock` values when both
    /// have been set; `None` when the engine hasn't published limits yet
    /// (use the builder instead). Intended for restart / hot-swap paths
    /// that pick up limits installed by the original `start_engine` call.
    pub fn from_globals() -> Option<Self> {
        let max_model_len = MAX_MODEL_LEN.get().copied()?;
        let max_seq_len_to_capture = MAX_SEQ_LEN_TO_CAPTURE
            .get()
            .copied()
            .unwrap_or(max_model_len);
        Some(Self {
            max_model_len,
            max_seq_len_to_capture,
        })
    }

    /// Test/bench-only constructor. Publishes `max_model_len = 1024`
    /// and `max_seq_len_to_capture = 1024`, matching the historical
    /// pool fallback so existing test corpora keep observing the
    /// shapes they did before this type was introduced. Production
    /// callers must use [`EngineLimitsBuilder`].
    pub fn for_testing() -> Self {
        EngineLimitsBuilder::new(1024).build()
    }

    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    pub fn max_seq_len_to_capture(&self) -> usize {
        self.max_seq_len_to_capture
    }

    /// Worst-case sequence length for pool sizing — same logic as the
    /// free function [`pool_worst_case_seq_len`] but reads from `self`
    /// rather than the globals. Useful when threading `EngineLimits`
    /// explicitly through a fresh code path that would rather not
    /// touch the global state.
    pub fn pool_worst_case_seq_len(&self) -> usize {
        self.max_seq_len_to_capture.min(self.max_model_len)
    }
}

/// Builder for [`EngineLimits`]. The required field `max_model_len`
/// is supplied to [`Self::new`]; `max_seq_len_to_capture` defaults to
/// the same value (no separate capture cap).
pub struct EngineLimitsBuilder {
    max_model_len: usize,
    max_seq_len_to_capture: Option<usize>,
}

impl EngineLimitsBuilder {
    /// Start a new builder with the engine-wide `--max-model-len`.
    /// Must be the value the server publishes to clients — anything
    /// smaller will cause the engine to truncate longer prompts; anything
    /// larger will balloon pool scratch buffers without benefit.
    pub fn new(max_model_len: usize) -> Self {
        Self {
            max_model_len,
            max_seq_len_to_capture: None,
        }
    }

    /// Override the capture-only sequence cap. Pool buffers sized for
    /// captured graphs (paged_attention V2 partitions, block_tables
    /// stride) use this when set; otherwise `max_model_len` is used
    /// for capture too. Clamped to `<= max_model_len` because capture
    /// cannot promise stability for sequences longer than the engine
    /// supports.
    pub fn max_seq_len_to_capture(mut self, value: usize) -> Self {
        self.max_seq_len_to_capture = Some(value);
        self
    }

    /// Finalise and publish to process-global `OnceLock` state. Calling
    /// this twice in the same process is safe — only the first set
    /// wins, but the returned struct still reflects the values from
    /// THIS call (so callers don't accidentally believe the second
    /// build took effect on the globals). For consistency it's
    /// recommended to construct at most one `EngineLimits` per
    /// process, immediately before the first `start_engine*` call.
    pub fn build(self) -> EngineLimits {
        let max_model_len = self.max_model_len;
        let max_seq_len_to_capture = self.max_seq_len_to_capture.unwrap_or(max_model_len);
        let capture_cap = max_seq_len_to_capture.min(max_model_len);
        let _ = MAX_MODEL_LEN.set(max_model_len);
        let _ = MAX_SEQ_LEN_TO_CAPTURE.set(capture_cap);
        EngineLimits {
            max_model_len,
            max_seq_len_to_capture: capture_cap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_clamps_capture_cap_to_max_model_len() {
        // Capture cap larger than max_model_len would advertise a
        // window the engine can't actually serve. Builder must clamp.
        let limits = EngineLimitsBuilder::new(1024)
            .max_seq_len_to_capture(8192)
            .build();
        assert_eq!(limits.max_model_len(), 1024);
        assert_eq!(limits.max_seq_len_to_capture(), 1024);
        assert_eq!(limits.pool_worst_case_seq_len(), 1024);
    }

    #[test]
    fn builder_capture_cap_defaults_to_max_model_len() {
        let limits = EngineLimitsBuilder::new(4096).build();
        assert_eq!(limits.max_seq_len_to_capture(), 4096);
    }

    #[test]
    fn for_testing_returns_legacy_1024_default() {
        let limits = EngineLimits::for_testing();
        assert_eq!(limits.max_model_len(), 1024);
        assert_eq!(limits.pool_worst_case_seq_len(), 1024);
    }
}
