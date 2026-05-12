//! Named-tensor dump utility for replay-correctness diagnostics.
//!
//! Activated by setting `VLLM_EXL3_DUMP_DIR=<path>`. When set, every
//! `dump_tensor("name", &t)` call copies `t` to host and writes raw bytes
//! to `<path>/<name>.<seq>.bin` (+ a `.meta` sidecar with shape/dtype).
//! Each name has its own monotonic sequence counter, so per-layer iteration
//! produces distinct files. When the env var is unset the function is a
//! near-no-op (single env-cache lookup + early return).
//!
//! Workflow:
//! - Run forward eagerly with `VLLM_EXL3_DUMP_DIR=/tmp/eager`.
//! - Run forward via captured replay with `VLLM_EXL3_DUMP_DIR=/tmp/capture`.
//! - `diff -r /tmp/eager /tmp/capture` — first differing `<name>.<seq>.bin`
//!   pinpoints the diverging op.

use candle_core::{DType, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static DUMP_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static COUNTERS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

fn dump_dir() -> Option<&'static PathBuf> {
    DUMP_DIR
        .get_or_init(|| {
            std::env::var("VLLM_EXL3_DUMP_DIR").ok().and_then(|s| {
                let p = PathBuf::from(s);
                match std::fs::create_dir_all(&p) {
                    Ok(()) => Some(p),
                    Err(e) => {
                        eprintln!("debug_dump: failed to create {p:?}: {e}");
                        None
                    }
                }
            })
        })
        .as_ref()
}

/// Returns `true` when dumping is active for this process.
pub fn is_enabled() -> bool {
    dump_dir().is_some()
}

/// Reset all per-name sequence counters. Call once per forward so that
/// `embed.out.0000.bin` in run A corresponds to `embed.out.0000.bin` in
/// run B (otherwise multi-forward runs would have monotonically growing
/// counters that don't align across runs).
pub fn reset_counters() {
    if let Some(m) = COUNTERS.get() {
        if let Ok(mut g) = m.lock() {
            g.clear();
        }
    }
}

fn next_seq(name: &str) -> u64 {
    let m = COUNTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().expect("debug_dump counter lock poisoned");
    let c = g.entry(name.to_string()).or_insert(0);
    let v = *c;
    *c += 1;
    v
}

/// Dump a tensor under the given name. Cheap no-op when env is unset.
///
/// Errors are intentionally swallowed (diagnostic-only path). The cost
/// is a host copy + file write — only ever active during debug runs.
pub fn dump_tensor(name: &str, t: &Tensor) {
    let Some(dir) = dump_dir() else {
        return;
    };

    let seq = next_seq(name);
    let stem = format!("{name}.{seq:04}");
    let bin_path = dir.join(format!("{stem}.bin"));
    let meta_path = dir.join(format!("{stem}.meta"));

    let bytes = match tensor_to_bytes(t) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("debug_dump[{name}.{seq:04}]: bytes failed: {e}");
            return;
        }
    };
    if let Err(e) = std::fs::write(&bin_path, &bytes) {
        eprintln!("debug_dump[{name}.{seq:04}]: write {bin_path:?} failed: {e}");
        return;
    }
    let meta = format!(
        "name={name}\nseq={seq}\nshape={:?}\ndtype={:?}\nbytes={}\n",
        t.dims(),
        t.dtype(),
        bytes.len()
    );
    let _ = std::fs::write(&meta_path, meta);
}

pub(crate) fn tensor_to_bytes_public(t: &Tensor) -> candle_core::Result<Vec<u8>> {
    tensor_to_bytes(t)
}

fn tensor_to_bytes(t: &Tensor) -> candle_core::Result<Vec<u8>> {
    let t = t.contiguous()?;
    let t = t.flatten_all()?;
    Ok(match t.dtype() {
        DType::F32 => {
            let v = t.to_vec1::<f32>()?;
            let mut out = Vec::with_capacity(v.len() * 4);
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        DType::F16 => {
            let v = t.to_vec1::<half::f16>()?;
            let mut out = Vec::with_capacity(v.len() * 2);
            for x in v {
                out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            out
        }
        DType::BF16 => {
            let v = t.to_vec1::<half::bf16>()?;
            let mut out = Vec::with_capacity(v.len() * 2);
            for x in v {
                out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            out
        }
        DType::U32 => {
            let v = t.to_vec1::<u32>()?;
            let mut out = Vec::with_capacity(v.len() * 4);
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        DType::U8 => t.to_vec1::<u8>()?,
        DType::I64 => {
            let v = t.to_vec1::<i64>()?;
            let mut out = Vec::with_capacity(v.len() * 8);
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        DType::F64 => {
            let v = t.to_vec1::<f64>()?;
            let mut out = Vec::with_capacity(v.len() * 8);
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }
        other => {
            return Err(candle_core::Error::Msg(format!(
                "debug_dump: dtype {other:?} not supported"
            )));
        }
    })
}
