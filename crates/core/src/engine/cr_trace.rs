// Phase CR.15 / D1 diagnostic: per-call invocation log of pool wrappers
// and OutputPool::reserve / reserve_pooled. Logs the exact sequence of
// (op, dtype, shape, slot_index) so we can diff captures across rounds
// in `cuda_kernels::tests::zz_mini_decoder_second_with_input_shift_after_prior`.
//
// All operations are zero-cost when `CR_TRACE_OPS` is not set in the
// environment (a single relaxed-atomic check + early return).

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

static INIT: OnceLock<bool> = OnceLock::new();
static FILE: Mutex<Option<BufWriter<File>>> = Mutex::new(None);
static IN_ROUND: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub fn enabled() -> bool {
    *INIT.get_or_init(|| env::var("CR_TRACE_OPS").is_ok())
}

/// Open a new log file `/tmp/cr_ops_round_{label}.log` and start
/// logging. Any prior round's file is flushed and closed.
pub fn begin_round(label: &str) {
    if !enabled() {
        return;
    }
    end_round();
    let path = format!("/tmp/cr_ops_round_{label}.log");
    match File::create(&path) {
        Ok(f) => {
            let mut guard = FILE.lock().expect("cr_trace mutex poisoned");
            *guard = Some(BufWriter::new(f));
            IN_ROUND.store(true, Ordering::Release);
            eprintln!("cr_trace: opened {path}");
        }
        Err(e) => eprintln!("cr_trace: failed to open {path}: {e}"),
    }
}

/// Flush and close the current round's log file. Subsequent log calls
/// are no-ops until the next `begin_round`.
pub fn end_round() {
    if !enabled() {
        return;
    }
    IN_ROUND.store(false, Ordering::Release);
    let mut guard = FILE.lock().expect("cr_trace mutex poisoned");
    if let Some(mut bw) = guard.take() {
        let _ = bw.flush();
    }
}

/// Mark the entry into a pool wrapper. Should be the first line of
/// each `*_pooled` function.
#[inline]
pub fn mark_op(op_name: &str) {
    if !enabled() || !IN_ROUND.load(Ordering::Acquire) {
        return;
    }
    let mut guard = FILE.lock().expect("cr_trace mutex poisoned");
    if let Some(bw) = guard.as_mut() {
        let _ = writeln!(bw, "[OP] {op_name}");
    }
}

/// Log a single `OutputPool::reserve` allocation. `slot_idx` is the
/// `entry.cursor` BEFORE increment (= 0-based slot number for this
/// (dtype, shape) bucket).
#[inline]
pub fn log_reserve(dtype: candle_core::DType, shape: &[usize], slot_idx: usize, ptr: u64) {
    if !enabled() || !IN_ROUND.load(Ordering::Acquire) {
        return;
    }
    let mut guard = FILE.lock().expect("cr_trace mutex poisoned");
    if let Some(bw) = guard.as_mut() {
        let _ = writeln!(
            bw,
            "[RESERVE] dtype={dtype:?} shape={shape:?} slot={slot_idx} ptr=0x{ptr:x}"
        );
    }
}

/// Log a free-form annotation (used for `reset_cursors`, `clear_all`,
/// or significant lifecycle events from the test harness).
#[inline]
pub fn note(msg: &str) {
    if !enabled() || !IN_ROUND.load(Ordering::Acquire) {
        return;
    }
    let mut guard = FILE.lock().expect("cr_trace mutex poisoned");
    if let Some(bw) = guard.as_mut() {
        let _ = writeln!(bw, "[NOTE] {msg}");
    }
}
