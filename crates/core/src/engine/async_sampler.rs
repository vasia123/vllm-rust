//! Async sampling infrastructure (ADR 0017 / Substep 2.2).
//!
//! Provides a dedicated CUDA stream and pinned host buffers for issuing
//! sampler-result `DtoH` copies that overlap with the next forward
//! pass. Without this path, the engine's `to_vec1()` after each step
//! blocks the host until the main stream drains 36 layers + lm_head +
//! sampler — measured at ~50 ms per call at c=8 (see
//! `docs/perf/post-15D-nsys-finding.md`). With this path the host
//! enqueues the next forward immediately while the previous step's
//! sampler IDs DMA on a side stream; the host waits for the events
//! only at the end of the multi-step block.
//!
//! Lifecycle: `AsyncSamplerInfra` is owned by `StandardExecution` via
//! `Mutex<Option<...>>` and lazy-initialised on the first decode
//! invocation when the model device is known.

use std::sync::Arc;

use candle_core::cuda::cudarc::driver::sys;
use candle_core::cuda::cudarc::driver::{
    result as cuda_result, CudaContext, CudaEvent, CudaStream,
};
use candle_core::Device;

/// Per-step pinned host buffer for sampler result DtoH.
///
/// We don't use cudarc 0.16's `alloc_pinned` because it hard-codes
/// `CU_MEMHOSTALLOC_WRITECOMBINED` — that flag optimises the buffer
/// for `HtoD` writes by the host but is incompatible with the
/// `DtoH` direction we need. The driver returns
/// `CUDA_ERROR_INVALID_VALUE` when memcpy_dtoh targets a WC-pinned
/// buffer. We allocate via `cuMemHostAlloc(flags=0)` directly for a
/// plain page-locked buffer that supports both directions.
///
/// `len` is fixed at construction to `max_batch`. We allocate
/// `max_multi_step` of these so an entire multi-step block can fly
/// without the next step waiting for the previous step's DtoH.
pub(crate) struct SamplerPinnedSlot {
    ptr: *mut u32,
    len: usize,
    /// Internal event tracking the most recent operation on this
    /// buffer. cudarc's `HostSlice` contract uses this so the dtoh
    /// stream can chain copies safely (next memcpy waits for prior
    /// memcpy completion).
    event: CudaEvent,
    /// Hold the context Arc so the device stays alive at least as
    /// long as our allocation. Drop order: ptr first via cuMemFreeHost,
    /// then ctx.
    _ctx: std::sync::Arc<CudaContext>,
}

// SAFETY: the underlying memory is allocated by the CUDA driver and
// is page-locked; access is single-threaded inside the engine's
// blocking task.
unsafe impl Send for SamplerPinnedSlot {}
unsafe impl Sync for SamplerPinnedSlot {}

impl SamplerPinnedSlot {
    /// Read the buffer's contents. Must be called AFTER the caller
    /// synchronises this slot's event (e.g. via `event_synchronize`).
    pub fn as_slice(&self) -> &[u32] {
        // SAFETY: ptr was allocated for `len` u32s and remains valid
        // until Drop. The caller has synchronised the DtoH event
        // before calling, so the memory contains valid data.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Sub-slice the underlying pinned buffer for the FIRST `n` elements
    /// to use as a `Dst` in `cudarc::CudaStream::memcpy_dtoh`. The
    /// `[T]` `HostSlice` impl uses `SyncOnDrop::Sync(None)` (no
    /// post-op sync), and because the pointer is registered as pinned
    /// by the driver, the DtoH stays asynchronous on the side stream.
    ///
    /// SAFETY: caller must ensure no concurrent access to the slot
    /// and must record an event on `dtoh_stream` after queueing the
    /// memcpy, then synchronise that event before reading via
    /// `as_slice` or `as_slice_first(n)`.
    pub unsafe fn as_mut_slice_first(&mut self, n: usize) -> &mut [u32] {
        let n = n.min(self.len);
        std::slice::from_raw_parts_mut(self.ptr, n)
    }

    /// Manually record `event` onto the given stream. Used after
    /// queuing a memcpy on `dtoh_stream` so the host can wait via
    /// `event_synchronize` later.
    pub fn record_event_on(&self, stream: &CudaStream) -> Result<(), candle_core::Error> {
        self.event
            .record(stream)
            .map_err(|e| candle_core::Error::Msg(format!("event.record: {e}")))
    }

    /// Block the host until the most recent memcpy_dtoh on this slot
    /// completes. Called once per slot at the end of a multi-step
    /// block before reading.
    pub fn event_synchronize(&self) -> Result<(), candle_core::Error> {
        self.event
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("SamplerPinnedSlot synchronize: {e}")))
    }
}

impl Drop for SamplerPinnedSlot {
    fn drop(&mut self) {
        // SAFETY: paired with the cuMemHostAlloc above; ptr is the
        // exact pointer returned. Any in-flight DtoH must complete
        // before drop — caller's responsibility (we sync events
        // before letting the slot drop, by using `as_slice` which
        // implies prior synchronisation).
        unsafe {
            let _ = cuda_result::free_host(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// Async-sampler resources owned per `StandardExecution`.
pub(crate) struct AsyncSamplerInfra {
    /// Dedicated CUDA stream for `DtoH` copies of sampler results.
    /// Created from the same `CudaContext` as the model's main stream
    /// so events can synchronise across the two streams.
    pub dtoh_stream: Arc<CudaStream>,
    /// Pre-allocated pinned host buffers, one per multi-step slot.
    /// Each slot holds `max_batch` u32 token IDs.
    pub slots: Vec<SamplerPinnedSlot>,
    /// Maximum batch size each slot can hold (also the slot length).
    pub max_batch: usize,
    /// How many slots — equals `max_multi_step` at construction.
    pub max_steps: usize,
    /// Diagnostic counter: how many pipelined-decode calls have run.
    /// Used to confirm the env-gated path is actually exercised in
    /// production benches (memory rule `feedback_shape_gate_diag.md`).
    pub call_counter: std::sync::atomic::AtomicU64,
}

impl AsyncSamplerInfra {
    /// Construct from a candle CUDA device. Allocates `max_steps`
    /// pinned slots of size `max_batch`. Returns `Err` if the device
    /// is not CUDA or the CUDA allocation fails.
    pub fn new(device: &Device, max_batch: usize, max_steps: usize) -> candle_core::Result<Self> {
        let cuda_dev = match device {
            Device::Cuda(d) => d,
            _ => {
                return Err(candle_core::Error::Msg(
                    "AsyncSamplerInfra requires a CUDA device".into(),
                ));
            }
        };
        let main_stream = cuda_dev.cuda_stream();
        let ctx: &Arc<CudaContext> = main_stream.context();

        // Dedicated stream for DtoH copies. Uses the same context so
        // events can cross-synchronise.
        let dtoh_stream = ctx
            .new_stream()
            .map_err(|e| candle_core::Error::Msg(format!("AsyncSamplerInfra new_stream: {e}")))?;

        let mut slots = Vec::with_capacity(max_steps);
        for _ in 0..max_steps {
            // SAFETY: cuMemHostAlloc is unsafe because the memory is
            // uninitialised. We never read before a DtoH writes it.
            // flags=0 gives a plain page-locked buffer (no WC, no
            // DEVICEMAP) — works for both DtoH and HtoD.
            let _ = ctx.bind_to_thread();
            let bytes = max_batch * std::mem::size_of::<u32>();
            let raw = unsafe { cuda_result::malloc_host(bytes, 0) }.map_err(|e| {
                candle_core::Error::Msg(format!("AsyncSamplerInfra malloc_host: {e}"))
            })?;
            // sys reference kept for module path validation — silences
            // dead-import lint without needing a feature gate.
            let _ = sys::CU_MEMHOSTALLOC_PORTABLE;
            let event = ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| candle_core::Error::Msg(format!("new_event: {e}")))?;
            slots.push(SamplerPinnedSlot {
                ptr: raw as *mut u32,
                len: max_batch,
                event,
                _ctx: Arc::clone(ctx),
            });
        }

        Ok(Self {
            dtoh_stream,
            slots,
            max_batch,
            max_steps,
            call_counter: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_rejected() {
        let dev = Device::Cpu;
        let res = AsyncSamplerInfra::new(&dev, 16, 4);
        assert!(res.is_err());
    }
}
