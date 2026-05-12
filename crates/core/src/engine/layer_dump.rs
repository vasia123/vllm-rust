//! Phase D10: per-layer activation snapshots that are SAFE inside a
//! captured CUDA graph.
//!
//! Activated by `VLLM_LAYER_DUMP_DIR=<path>`. When set, calls to
//! [`dump_at`] copy the source tensor into a dedicated per-name buffer
//! via a captured `cuMemcpyDtoDAsync`. Buffers are allocated lazily on
//! first call (so all allocations happen on warmup forwards BEFORE the
//! capture session begins), then reused forever — the device pointer
//! is stable for capture-replay.
//!
//! After a forward completes (and `cuCtxSynchronize` has been issued),
//! call [`flush`] to dtoh every retained buffer and write its raw bytes
//! to `<path>/<name>.bin`. The host copies must happen AFTER the
//! captured graph finishes — calling [`flush`] mid-replay would
//! interleave host-syncs with subsequent replay launches and trigger
//! `CUDA_ERROR_ILLEGAL_ADDRESS` (see `helpers.rs` one-shot dump
//! comment).
//!
//! Typical workflow:
//! 1. Run with `VLLM_LAYER_DUMP_DIR=/tmp/eager VLLM_EXL3_CAPTURE_MAX_M=0`
//!    → eager pool-V2 forward fills buffers, flushed to `/tmp/eager/*.bin`.
//! 2. Run with `VLLM_LAYER_DUMP_DIR=/tmp/capture VLLM_EXL3_CAPTURE_MAX_M=16`
//!    → captured replay fills buffers, flushed once after forward 1.
//! 3. `diff -r /tmp/eager /tmp/capture` — first differing file is the
//!    first divergent layer.

use candle_core::Tensor;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static BUFFERS: OnceLock<Mutex<BTreeMap<String, Tensor>>> = OnceLock::new();

thread_local! {
    /// Layer index of the layer currently being forwarded. Set by
    /// `with_current_layer` so inner ops (attention, MLP) can include
    /// the layer index in their dump names. Default: `usize::MAX` means
    /// "no active layer scope" — dumps under this should be skipped or
    /// use a layer-agnostic name.
    static CURRENT_LAYER: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Set the active layer index for the duration of `f`. Inner dump
/// sites can read it via [`current_layer`] to build layer-tagged names
/// without threading `layer_idx` through every signature.
pub fn with_current_layer<F, R>(layer_idx: usize, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = CURRENT_LAYER.with(|c| c.replace(layer_idx));
    let result = f();
    CURRENT_LAYER.with(|c| c.set(prev));
    result
}

/// Read the active layer index. `usize::MAX` if no layer scope is set.
pub fn current_layer() -> usize {
    CURRENT_LAYER.with(|c| c.get())
}

fn dir() -> Option<&'static PathBuf> {
    DIR.get_or_init(|| {
        std::env::var("VLLM_LAYER_DUMP_DIR").ok().and_then(|s| {
            let p = PathBuf::from(s);
            match std::fs::create_dir_all(&p) {
                Ok(()) => Some(p),
                Err(e) => {
                    eprintln!("layer_dump: failed to create {p:?}: {e}");
                    None
                }
            }
        })
    })
    .as_ref()
}

/// Returns `true` when layer dumping is active for this process.
#[inline]
pub fn is_enabled() -> bool {
    dir().is_some()
}

fn buffers() -> &'static Mutex<BTreeMap<String, Tensor>> {
    BUFFERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Snapshot `t` into a buffer associated with `name`. The first call
/// allocates a fresh persistent buffer of the same shape/dtype on the
/// same device — that buffer is then reused for every subsequent call
/// with the same name (its device pointer stays stable so the
/// `cuMemcpyDtoDAsync` it records is replay-safe).
///
/// **Important**: the first call for any name MUST occur before
/// `cuStreamBeginCapture` (i.e., during warmup forwards). Otherwise the
/// fresh allocation happens inside the capture session, which is
/// undefined for cuMemAllocAsync.
pub fn dump_at(name: &str, t: &Tensor) {
    if !is_enabled() {
        return;
    }
    // Normalise to leading batch dim = 1. Captured graphs are recorded
    // against the padded batch size (e.g., 16), while eager runs use the
    // request's actual batch (often 1). Comparing dumps requires a
    // uniform shape, so we always narrow the first dim to a single
    // sequence (index 0). For 1-D or empty tensors this is a no-op.
    let narrowed = match t.dims().first().copied() {
        Some(d) if d > 1 => match t.narrow(0, 0, 1).and_then(|v| v.contiguous()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("layer_dump[{name}]: narrow/contig failed: {e}");
                return;
            }
        },
        _ => match t.contiguous() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("layer_dump[{name}]: contiguous failed: {e}");
                return;
            }
        },
    };

    let mut guard = match buffers().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    use std::collections::btree_map::Entry;
    let buf = match guard.entry(name.to_string()) {
        Entry::Occupied(o) => o.into_mut(),
        Entry::Vacant(v) => {
            let fresh = match Tensor::zeros(narrowed.dims(), narrowed.dtype(), narrowed.device()) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("layer_dump[{name}]: zeros failed: {e}");
                    return;
                }
            };
            v.insert(fresh)
        }
    };
    let buf = buf.clone();
    drop(guard);

    if let Err(e) = super::cuda_graph_runner::cuda_memcpy_inplace(&buf, &narrowed) {
        eprintln!("layer_dump[{name}]: memcpy failed: {e}");
    }
}

/// dtoh every retained buffer and write its raw bytes to `<dir>/<name>.bin`
/// plus a `<name>.meta` sidecar. Call once after a forward completes and
/// the device has synchronised.
pub fn flush() {
    let Some(dir) = dir() else {
        return;
    };
    let guard = match buffers().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    for (name, buf) in guard.iter() {
        let bytes = match super::debug_dump::tensor_to_bytes_public(buf) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("layer_dump[{name}]: bytes failed: {e}");
                continue;
            }
        };
        let bin = dir.join(format!("{name}.bin"));
        let meta = dir.join(format!("{name}.meta"));
        if let Err(e) = std::fs::write(&bin, &bytes) {
            eprintln!("layer_dump[{name}]: write {bin:?} failed: {e}");
            continue;
        }
        let m = format!(
            "name={name}\nshape={:?}\ndtype={:?}\nbytes={}\n",
            buf.dims(),
            buf.dtype(),
            bytes.len()
        );
        let _ = std::fs::write(&meta, m);
    }
    eprintln!("layer_dump: flushed {} buffers to {:?}", guard.len(), dir);
}

/// Convenience for tests / setup paths: drops every retained buffer.
/// Production code should not call this — the buffers' device pointers
/// may already be encoded into captured CUDA graphs.
#[allow(dead_code)]
pub fn clear() {
    if let Some(m) = BUFFERS.get() {
        if let Ok(mut g) = m.lock() {
            g.clear();
        }
    }
}

/// Format a layer-output name: `layer.{idx:02}.out`. Centralised so
/// dump producers and offline diff tools agree on the naming scheme.
pub fn layer_out_name(layer_idx: usize) -> String {
    format!("layer.{layer_idx:02}.out")
}

/// Format a named sub-op slot within a layer: `layer.{idx:02}.{tag}`.
pub fn layer_slot_name(layer_idx: usize, tag: &str) -> String {
    format!("layer.{layer_idx:02}.{tag}")
}
