//! Process-global runtime limits exposed to hot-path pool sizing.
//!
//! Background: pool buffers used on the CUDA Graph capture replay path
//! (paged_attention V2 partitions, block_tables stride) need a constant
//! shape across forwards. The right upper bound is `--max-model-len`,
//! which is a CLI flag at the server, but the pool sites are deep inside
//! `vllm-core` and the value would need to thread through 200+
//! `CacheConfig` literals to reach them.
//!
//! Instead, the server initialises this once at startup; pool sites read
//! from here. `None` keeps the historical hard-coded 1024 default so
//! crates that use vllm-core without this hook (tests, benches,
//! standalone use) continue working unchanged.

use std::sync::OnceLock;

static MAX_MODEL_LEN: OnceLock<usize> = OnceLock::new();
static MAX_SEQ_LEN_TO_CAPTURE: OnceLock<usize> = OnceLock::new();

/// Set the engine-wide `--max-model-len`. Idempotent — only the first
/// call wins. Safe to call multiple times with the same value.
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
