//! Per-device scratch buffers shared by the EXL3 CUDA kernels.
//!
//! `Exl3GemmOp::cuda_fwd` previously allocated a fresh `int32[1 MiB + 2 KiB]`
//! locks workspace per launch via `dev.alloc_zeros::<i32>(...)`. At Llama-3.2-1B
//! decode rates that costs ~112 4 MiB `cuMemAllocAsync` calls per token —
//! the single largest source of host-side overhead in the EXL3 hot path,
//! eclipsing kernel time itself on this hardware.
//!
//! Upstream ExLlamaV3 sidesteps the issue by holding **one** locks buffer
//! per CUDA device for the life of the process (see `quant/exl3_devctx.cu`:
//! `cudaMalloc + cudaMemset` once, never freed, never reset). The buffer is
//! self-restoring across launches: `barrier_release(reset=true)` zeroes
//! locks at the tail of each column, and `group_barrier` uses sense-
//! reversal so the sense bit's *value* doesn't matter — only that it
//! flips. We mirror that lifetime exactly: a `OnceLock`-protected
//! `HashMap<DeviceId, Arc<CudaSlice<i32>>>` populated on first use per
//! device.
//!
//! Safety: `CudaSlice<T>` is `Send + Sync` (see cudarc 0.16
//! `safe/core.rs:495`). We hand callers an `Arc` clone — the underlying
//! device memory stays alive for the process lifetime.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::cuda_backend::{CudaDevice, DeviceId};
use candle_core::Result;

/// Length of the EXL3 GEMM locks workspace, in `int32` elements.
/// Mirrors `(MAX_TILES_C + MAX_BARRIERS * 2)` from
/// `kernels/exl3/exl3_devctx.cuh`. Kept in sync via the assert in tests.
pub const EXL3_LOCKS_ELEMS: usize = (1024 * 1024) + (1024 * 2);

static LOCKS_CACHE: OnceLock<Mutex<HashMap<DeviceId, Arc<CudaSlice<i32>>>>> = OnceLock::new();
static NULL_SCALE_CACHE: OnceLock<Mutex<HashMap<DeviceId, Arc<CudaSlice<half::f16>>>>> =
    OnceLock::new();

/// Get (or lazily allocate) the per-device EXL3 GEMM locks buffer.
///
/// First call per device pays one `cudaMalloc + cudaMemset(0)` of ~4 MiB.
/// All subsequent calls return an `Arc` clone — zero CUDA traffic.
///
/// The kernel's barrier protocol leaves the buffer in a consistent
/// resetable state after every successful launch (see module docs); no
/// per-call memset is required.
pub fn exl3_locks(dev: &CudaDevice) -> Result<Arc<CudaSlice<i32>>> {
    let cache = LOCKS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let id = dev.id();
    {
        let guard = cache.lock().expect("exl3_locks: cache mutex poisoned");
        if let Some(buf) = guard.get(&id) {
            return Ok(buf.clone());
        }
    }
    // Allocate outside the lock — `alloc_zeros` issues a stream operation
    // and we don't want to serialise unrelated device traffic on this
    // mutex.
    let fresh = dev
        .alloc_zeros::<i32>(EXL3_LOCKS_ELEMS)
        .map_err(|e| candle_core::Error::Msg(format!("exl3_locks alloc: {e}")))?;
    let arc = Arc::new(fresh);
    let mut guard = cache.lock().expect("exl3_locks: cache mutex poisoned");
    // Re-check: another thread may have populated between our drop+reacquire.
    Ok(guard.entry(id).or_insert(arc).clone())
}

/// Get (or lazily allocate) a per-device 1-element fp16 sentinel buffer.
///
/// `HadR128Fp16InplaceOp` passes a non-NULL `scale` device pointer to
/// the Hadamard kernel even in `HadScale::None` mode (the kernel
/// dereferences the pointer unconditionally; the dereference is then
/// gated by a compile-time `Lb<has_scale>` template flag). Without
/// caching we'd issue a 2-byte `cudaMallocAsync` every call. The
/// content is irrelevant — the kernel never reads through this pointer
/// in the `<false,false>` template instance — so a single zero-init
/// buffer per device is enough.
pub fn exl3_had_null_scale(dev: &CudaDevice) -> Result<Arc<CudaSlice<half::f16>>> {
    let cache = NULL_SCALE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let id = dev.id();
    {
        let guard = cache
            .lock()
            .expect("exl3_had_null_scale: cache mutex poisoned");
        if let Some(buf) = guard.get(&id) {
            return Ok(buf.clone());
        }
    }
    let fresh = dev
        .alloc_zeros::<half::f16>(1)
        .map_err(|e| candle_core::Error::Msg(format!("exl3_had_null_scale alloc: {e}")))?;
    let arc = Arc::new(fresh);
    let mut guard = cache
        .lock()
        .expect("exl3_had_null_scale: cache mutex poisoned");
    Ok(guard.entry(id).or_insert(arc).clone())
}

static KERNEL_NAME_CACHE: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();

/// Intern a kernel name string for `dev.get_or_load_custom_func`, which
/// takes a `&'static str` key. Without interning every `cuda_fwd` call
/// `Box::leak`s a fresh copy of the same mangled symbol — small (~1 KB)
/// but grows unbounded with launch count. Interning leaks each unique
/// string at most once.
///
/// Usage: `let name = intern_kernel_name(format!("...", ...));`
pub fn intern_kernel_name(name: String) -> &'static str {
    let cache = KERNEL_NAME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .expect("intern_kernel_name: cache mutex poisoned");
    if let Some(&s) = guard.get(&name) {
        return s;
    }
    let leaked: &'static str = Box::leak(name.clone().into_boxed_str());
    guard.insert(name, leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_elems_matches_kernel_define() {
        // exl3_devctx.cuh: MAX_TILES_C = (1024*1024), MAX_BARRIERS = 1024.
        // Buffer length = MAX_TILES_C + MAX_BARRIERS * 2.
        assert_eq!(EXL3_LOCKS_ELEMS, (1024 * 1024) + (1024 * 2));
    }

    #[cfg(feature = "gpu-test-small")]
    #[test]
    fn locks_cache_returns_same_arc_per_device() {
        let dev = candle_core::Device::new_cuda(0).unwrap();
        let cuda = match &dev {
            candle_core::Device::Cuda(c) => c,
            _ => unreachable!(),
        };
        let a = exl3_locks(cuda).unwrap();
        let b = exl3_locks(cuda).unwrap();
        // Same underlying allocation — Arc pointer-equal.
        assert!(Arc::ptr_eq(&a, &b));
    }
}
