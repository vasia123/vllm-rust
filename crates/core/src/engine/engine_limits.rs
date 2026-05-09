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
