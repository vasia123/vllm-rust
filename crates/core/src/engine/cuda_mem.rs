//! CUDA stream-ordered memory-pool hygiene.
//!
//! candle 0.10 frees device buffers promptly on drop (`cuMemFreeAsync`),
//! but the CUDA driver's stream-ordered memory pool retains that memory
//! for reuse instead of returning it to the OS. Under a workload that
//! churns many differently-shaped transient allocations — concurrent
//! prefills of varied length, decode batches of varied size,
//! preempt-recompute re-prefills — the pool grows monotonically toward
//! the device limit (observed live on Gemma 4 12B EXL3 @ 8 GB: a single
//! 6-request burst grew used VRAM 5945 → 7865 MiB), after which a
//! transient activation allocation OOMs even though almost nothing is
//! actually live. This is the project's known "async-mempool leak".
//!
//! This module exposes two best-effort controls over the device's
//! default pool:
//!   - [`init`] sets `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` so the pool
//!     keeps only a bounded cache and releases the rest at the next
//!     synchronization (the engine synchronizes every step via the
//!     sampler DtoH copy).
//!   - [`trim_under_pressure`] / [`trim`] synchronize and call
//!     `cuMemPoolTrimTo` to hand retained memory back to the OS when
//!     free VRAM drops below a watermark, and after an OOM.
//!
//! All entry points are no-ops on non-CUDA builds and never panic — a
//! failed driver call is logged and ignored, since this is memory
//! hygiene, not correctness.

#[cfg(feature = "cuda")]
mod imp {
    use candle_core::cuda::cudarc::driver::{result, sys};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// Device ordinal whose default pool we manage. Set once by [`init`];
    /// the per-step trim path reads it without threading the ordinal
    /// through `EngineConfig`/`run_engine_loop`. Single-GPU in practice;
    /// multi-GPU would call `init` per device (last write wins here,
    /// which is acceptable for the current single-device server).
    static ORDINAL: OnceLock<usize> = OnceLock::new();

    fn default_pool(ordinal: usize) -> Option<sys::CUmemoryPool> {
        // SAFETY: `device::get` returns a valid handle for an existing
        // ordinal; `get_default_mem_pool` requires exactly that.
        unsafe {
            let dev = result::device::get(ordinal as i32).ok()?;
            result::device::get_default_mem_pool(dev).ok()
        }
    }

    pub fn init(ordinal: usize, release_threshold_bytes: u64) {
        let _ = ORDINAL.set(ordinal);
        let Some(pool) = default_pool(ordinal) else {
            tracing::warn!(
                target: "vllm_core::cuda_mem",
                ordinal,
                "could not resolve default CUDA mem pool; mempool reclaim disabled"
            );
            return;
        };
        let mut threshold = release_threshold_bytes;
        // SAFETY: `pool` is the valid default pool; the attribute takes a
        // `cuuint64_t*` and RELEASE_THRESHOLD is a u64.
        let rc = unsafe {
            result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut threshold as *mut u64 as *mut c_void,
            )
        };
        match rc {
            Ok(()) => eprintln!(
                "CUDA mem-pool reclaim active (ordinal {ordinal}, release threshold {} MiB)",
                release_threshold_bytes / (1024 * 1024)
            ),
            Err(e) => eprintln!("WARNING: failed to set CUDA mem pool release threshold: {e}"),
        }
    }

    /// Free VRAM in bytes (`cuMemGetInfo`), or `None` if unavailable.
    pub fn free_vram() -> Option<usize> {
        result::mem_get_info().ok().map(|(free, _total)| free)
    }

    /// Synchronize and release pool memory beyond `keep_bytes` back to
    /// the OS. Best-effort; logs and ignores driver errors.
    pub fn trim(keep_bytes: usize) {
        let Some(&ordinal) = ORDINAL.get() else {
            return;
        };
        let Some(pool) = default_pool(ordinal) else {
            return;
        };
        // Drain pending async frees so the pool's "reserved but unused"
        // accounting reflects reality before we trim.
        // SAFETY: no outstanding borrows of device memory cross this call
        // in the single-stream engine step; trim only releases unused
        // pool memory.
        unsafe {
            let _ = result::ctx::synchronize();
            if let Err(e) = result::mem_pool::trim_to(pool, keep_bytes) {
                tracing::debug!(target: "vllm_core::cuda_mem", error = %e, "mem pool trim_to failed");
            }
        }
    }

    /// Trim only when free VRAM has dropped below `watermark_bytes`.
    /// Cheap to call once per engine iteration: a single `mem_get_info`
    /// when above the watermark, a synchronize + trim only under
    /// pressure. Returns `true` if a trim was performed.
    pub fn trim_under_pressure(watermark_bytes: usize, keep_bytes: usize) -> bool {
        match free_vram() {
            Some(free) if free < watermark_bytes => {
                trim(keep_bytes);
                if std::env::var("VLLM_CUDA_MEMPOOL_DEBUG").is_ok() {
                    let after = free_vram().unwrap_or(0);
                    eprintln!(
                        "[mempool] trim: free {} MiB -> {} MiB (reclaimed {} MiB)",
                        free / (1 << 20),
                        after / (1 << 20),
                        after.saturating_sub(free) / (1 << 20),
                    );
                }
                true
            }
            _ => false,
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod imp {
    pub fn init(_ordinal: usize, _release_threshold_bytes: u64) {}
    pub fn free_vram() -> Option<usize> {
        None
    }
    pub fn trim(_keep_bytes: usize) {}
    pub fn trim_under_pressure(_watermark_bytes: usize, _keep_bytes: usize) -> bool {
        false
    }
}

pub use imp::{free_vram, init, trim, trim_under_pressure};

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use candle_core::{DType, Device, Tensor};

    /// Smoke test (GPU): after a large tensor is dropped, [`trim`] must
    /// hand the pooled memory back to the OS (free VRAM recovers).
    /// Confirms the cudarc mem-pool FFI path actually reclaims — the
    /// whole premise of the burst-OOM fix. Ignored by default (needs a
    /// CUDA device with ~1.5 GB free); run explicitly with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn trim_reclaims_dropped_allocations() {
        let dev = Device::new_cuda(0).expect("cuda device");
        super::init(0, 0);
        let free_start = super::free_vram().expect("free_vram");
        {
            // ~1 GiB transient, then dropped → candle free_async returns
            // it to the stream-ordered pool (retained, not yet to OS).
            let _t = Tensor::zeros((256, 1024, 1024), DType::F32, &dev).unwrap();
            let _ = _t.sum_all().unwrap(); // force the alloc to materialize
        }
        let free_pooled = super::free_vram().unwrap();
        super::trim(0);
        let free_after = super::free_vram().unwrap();
        eprintln!(
            "free_start={} MiB, free_after_drop={} MiB, free_after_trim={} MiB",
            free_start / (1 << 20),
            free_pooled / (1 << 20),
            free_after / (1 << 20),
        );
        assert!(
            free_after >= free_pooled,
            "trim must not reduce free memory"
        );
        // The reclaim should bring us back within ~64 MiB of the start.
        assert!(
            free_after + (64 << 20) >= free_start,
            "trim should reclaim the dropped ~1 GiB (start={free_start}, after_trim={free_after})"
        );
    }
}

/// Bytes the pool may retain after [`init`]'s release-threshold is set,
/// and the floor passed to [`trim`]. Keeping a modest cache avoids
/// churning `cuMemAlloc`/free on the steady-state decode path while
/// still bounding growth. Overridable via `VLLM_CUDA_MEMPOOL_KEEP_MB`.
pub fn keep_bytes() -> usize {
    const DEFAULT_KEEP_MB: usize = 256;
    let mb = std::env::var("VLLM_CUDA_MEMPOOL_KEEP_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_KEEP_MB);
    mb * 1024 * 1024
}

/// Free-VRAM floor below which the engine trims the pool at the end of a
/// step. Overridable via `VLLM_CUDA_MEMPOOL_WATERMARK_MB`.
pub fn watermark_bytes() -> usize {
    const DEFAULT_WATERMARK_MB: usize = 1024;
    let mb = std::env::var("VLLM_CUDA_MEMPOOL_WATERMARK_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WATERMARK_MB);
    mb * 1024 * 1024
}
