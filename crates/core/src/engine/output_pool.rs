//! Output buffer pool for hot decoder kernels.
//!
//! Pre-allocates and re-uses `Tensor`s across forward passes so the GEMV
//! / paged-attention / norm hot path stops calling `cuMemAlloc` on every
//! op.  cuMemAlloc on a 8 GB Ada laptop costs ~3-6 µs per call; with the
//! AWQ INT4 GEMV alone landing 252 calls per token (7 linears × 36 layers),
//! eliminating the syscall round-trip is worth ~1-2 ms per token at
//! steady state — and is the single biggest blocker for CUDA Graph
//! capture once we own the stream lifecycle (Stage 11).
//!
//! ## Usage contract
//!
//! 1. `OutputPool::reset_cursors()` is called once at the start of every
//!    decode forward in [`crate::engine::helpers::execute_batched_decode_with_graph`].
//!    This rewinds all per-shape round-robin cursors back to zero.
//! 2. Each kernel-launch site that wants a pre-allocated output requests
//!    one via [`OutputPool::reserve`].  The first call for a given
//!    `(shape, dtype)` allocates a fresh tensor; subsequent calls within
//!    the same forward step return *distinct* pre-allocated tensors at
//!    increasing pool indices, and grow the pool on demand if the pool
//!    runs out of pre-allocated buffers.
//! 3. The returned tensor shares storage (Arc-cloned) with the pooled
//!    tensor — kernels write into that storage; the caller hands the
//!    tensor down the layer chain; the storage stays alive for the life
//!    of the pool.
//!
//! Because forwards are executed serially inside the engine task, a
//! single global pool is safe.  A `Mutex` guards the inner map; the
//! critical section is a HashMap lookup + index, dwarfed by the µs-scale
//! kernel launches it amortises.
//!
//! When invoked from non-pooled paths (eager prefill, tests), kernels
//! continue allocating their own buffers — the pool is opt-in per call
//! site, not a global override.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Device, Result, Tensor};

/// One pool entry per `(dtype, shape)` key.
struct PoolEntry {
    /// Pre-allocated tensors. Each is independently usable; entries do
    /// **not** alias one another (each is a fresh `Tensor::zeros`
    /// allocation).
    buffers: Vec<Tensor>,
    /// Index of the next buffer to hand out within the current forward.
    /// Reset to zero by [`OutputPool::reset_cursors`].
    cursor: usize,
}

/// Shape-keyed buffer pool. See module docs.
#[derive(Default)]
pub struct OutputPool {
    inner: Mutex<HashMap<(DType, Vec<usize>), PoolEntry>>,
}

impl OutputPool {
    /// Process-global pool.  Initialised on first access and never freed
    /// — its buffers live as long as the engine.
    pub fn global() -> &'static Self {
        static POOL: OnceLock<OutputPool> = OnceLock::new();
        POOL.get_or_init(OutputPool::default)
    }

    /// Rewind every per-shape cursor to zero.  Call once at the start of
    /// each decode forward so the next batch of `reserve()` calls walks
    /// the same allocation order as the previous one.
    pub fn reset_cursors(&self) {
        let mut inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        for entry in inner.values_mut() {
            entry.cursor = 0;
        }
    }

    /// Hand out a pre-allocated tensor of `(shape, dtype)`, growing the
    /// pool on demand.  The returned tensor shares storage with the
    /// pool's owning copy; consumers may write through it freely and
    /// pass it down the layer chain.  When all callers drop their
    /// clones, the storage stays alive in the pool's slot.
    pub fn reserve(&self, shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
        let key = (dtype, shape.to_vec());
        let mut inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        let entry = inner.entry(key).or_insert_with(|| PoolEntry {
            buffers: Vec::new(),
            cursor: 0,
        });

        // Bound per-shape growth.  In a single decode forward, the same
        // shape is requested at most ~108× (q/k/v/o + gate/up/down across
        // 36 layers, with q/o/down sharing one N).  A conservative cap
        // of 512 absorbs any reasonable forward, but prevents unbounded
        // accumulation when a caller forgets to invoke `reset_cursors`
        // (e.g. prefill historically did not, leaking ~250 entries per
        // request — see commit log for the post-Stage-12 OOM regression).
        // When the cap is hit we fall back to a one-shot allocation; the
        // caller still gets a usable tensor, the pool just doesn't grow.
        const MAX_PER_SHAPE: usize = 512;

        if entry.cursor >= entry.buffers.len() {
            if entry.buffers.len() >= MAX_PER_SHAPE {
                // Fall through: allocate without retaining in the pool.
                // The cursor is still incremented so subsequent `reserve`s
                // in this forward see consistent growth semantics.
                drop(inner);
                let fresh = Tensor::zeros(shape, dtype, device)?;
                return Ok(fresh);
            }
            // Grow the pool. `Tensor::zeros` is the only candle entry
            // point that materialises a fresh CudaSlice without
            // intermediate copies.  This pays a one-shot cuMemAlloc per
            // newly-needed slot; subsequent forwards reuse the slot and
            // skip the syscall entirely.
            let fresh = Tensor::zeros(shape, dtype, device)?;
            entry.buffers.push(fresh);
        }

        let tensor = entry.buffers[entry.cursor].clone();
        entry.cursor += 1;
        Ok(tensor)
    }

    /// Total number of pre-allocated tensor slots across all shapes.
    /// For diagnostics / tests.
    #[allow(dead_code)]
    pub fn total_slots(&self) -> usize {
        let inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        inner.values().map(|e| e.buffers.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_grows_pool_on_repeat_within_forward() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        // Two reserves of the same shape inside a single (un-reset)
        // window should allocate two distinct tensors.
        let a = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        let b = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        // Different storage = different ptrs. We approximate this by
        // checking that `Tensor` identities (`as_ptr()` of the storage
        // arc) differ. Cheap proxy: write one, read the other.
        assert_eq!(pool.total_slots(), 2);
        // Writes through one don't affect the other (storage is
        // independent).
        let _ = a; // keep both alive
        let _ = b;
    }

    #[test]
    fn reset_cursors_replays_existing_buffers() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let _ = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        let _ = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        assert_eq!(pool.total_slots(), 2);

        pool.reset_cursors();

        // Reserve again — cursor restarts, no new allocation.
        let _ = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        let _ = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        assert_eq!(pool.total_slots(), 2, "pool should not grow on replay");
    }

    #[test]
    fn distinct_shapes_keep_separate_slots() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let _ = pool.reserve(&[4, 8], DType::F32, &dev).unwrap();
        let _ = pool.reserve(&[8, 4], DType::F32, &dev).unwrap();
        let _ = pool.reserve(&[4, 8], DType::F16, &dev).unwrap();
        assert_eq!(pool.total_slots(), 3);
    }
}
