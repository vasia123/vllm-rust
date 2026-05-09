//! CUDA Graph Runner for optimized model inference.
//!
//! This module provides the `CudaGraphRunner` which captures and replays CUDA graphs
//! for batched decode operations, eliminating CPU kernel launch overhead.
//!
//! # How it works
//!
//! 1. During warmup, the runner allocates fixed-size input/output buffers for each
//!    configured batch size and captures the forward pass into a CUDA graph.
//!
//! 2. During inference, when a batch matches a captured size:
//!    - Copy input tokens to the pre-allocated input buffer
//!    - Replay the captured graph (no kernel launches, just graph execution)
//!    - Read results from the pre-allocated output buffer
//!
//! 3. For batch sizes without captured graphs, fall back to eager execution.
//!
//! # Requirements
//!
//! - CUDA device (compute capability 7.0+)
//! - Model forward must be deterministic (same inputs → same CUDA calls)
//! - All memory allocations must happen before capture

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use super::cuda_graph::CudaGraphConfig;

// Re-export for tests
#[cfg(test)]
use super::cuda_graph::RuntimeMode;

// ─── In-place tensor copy ─────────────────────────────────────────────────

/// Copies tensor data from `src` into `dst` in-place using device-to-device memcpy.
///
/// Both tensors must:
/// - Have the same dtype
/// - Have the same number of elements
/// - Reside on the same device
///
/// For CUDA tensors, this performs an async device-to-device memcpy on the device's
/// stream. For CPU tensors, this copies via the standard byte slice path.
///
/// This is essential for CUDA graph capture and replay, where the graph's input
/// buffers must be updated in-place before replaying the recorded operations.
pub fn cuda_memcpy_inplace(dst: &Tensor, src: &Tensor) -> Result<(), CudaGraphRunnerError> {
    if dst.dtype() != src.dtype() {
        return Err(CudaGraphRunnerError::InplaceCopyFailed(format!(
            "dtype mismatch: dst={:?}, src={:?}",
            dst.dtype(),
            src.dtype()
        )));
    }
    if dst.elem_count() != src.elem_count() {
        return Err(CudaGraphRunnerError::InplaceCopyFailed(format!(
            "element count mismatch: dst={}, src={}",
            dst.elem_count(),
            src.elem_count()
        )));
    }
    if dst.device().location() != src.device().location() {
        return Err(CudaGraphRunnerError::InplaceCopyFailed(format!(
            "device mismatch: dst={:?}, src={:?}",
            dst.device().location(),
            src.device().location()
        )));
    }

    dst.inplace_op2(src, &InplaceCopyOp)
        .map_err(|e| CudaGraphRunnerError::InplaceCopyFailed(e.to_string()))
}

/// Candle InplaceOp2 that copies data from one tensor's storage into another.
struct InplaceCopyOp;

impl candle_core::InplaceOp2 for InplaceCopyOp {
    fn name(&self) -> &'static str {
        "inplace_copy_dtod"
    }

    fn cpu_fwd(
        &self,
        dst_storage: &mut candle_core::CpuStorage,
        dst_layout: &candle_core::Layout,
        src_storage: &candle_core::CpuStorage,
        src_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use candle_core::CpuStorage;

        let dst_offset = dst_layout.start_offset();
        let src_offset = src_layout.start_offset();
        let elem_count = dst_layout.shape().elem_count();

        match (dst_storage, src_storage) {
            (CpuStorage::U8(d), CpuStorage::U8(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            (CpuStorage::U32(d), CpuStorage::U32(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            (CpuStorage::I64(d), CpuStorage::I64(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            (CpuStorage::BF16(d), CpuStorage::BF16(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            (CpuStorage::F16(d), CpuStorage::F16(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            (CpuStorage::F32(d), CpuStorage::F32(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            (CpuStorage::F64(d), CpuStorage::F64(s)) => {
                d[dst_offset..dst_offset + elem_count]
                    .copy_from_slice(&s[src_offset..src_offset + elem_count]);
            }
            _ => candle_core::bail!("inplace_copy: unsupported or mismatched dtypes"),
        }
        Ok(())
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        dst_storage: &mut candle_core::CudaStorage,
        dst_layout: &candle_core::Layout,
        src_storage: &candle_core::CudaStorage,
        src_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda::CudaStorageSlice;

        let elem_count = dst_layout.shape().elem_count();
        let dst_offset = dst_layout.start_offset();
        let src_offset = src_layout.start_offset();

        match (&mut dst_storage.slice, &src_storage.slice) {
            (CudaStorageSlice::U8(d), CudaStorageSlice::U8(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            (CudaStorageSlice::U32(d), CudaStorageSlice::U32(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            (CudaStorageSlice::I64(d), CudaStorageSlice::I64(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            (CudaStorageSlice::BF16(d), CudaStorageSlice::BF16(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            (CudaStorageSlice::F16(d), CudaStorageSlice::F16(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            (CudaStorageSlice::F32(d), CudaStorageSlice::F32(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            (CudaStorageSlice::F64(d), CudaStorageSlice::F64(s)) => {
                let sv = s.slice(src_offset..src_offset + elem_count);
                let mut dv = d.slice_mut(dst_offset..dst_offset + elem_count);
                dst_storage.device.memcpy_dtod(&sv, &mut dv)?;
            }
            _ => candle_core::bail!("inplace_copy: unsupported or mismatched CUDA dtypes"),
        }
        Ok(())
    }
}

/// Pre-allocated buffers for a specific batch size.
#[cfg(feature = "cuda-kernels")]
struct CaptureBuffers {
    /// Input token IDs [batch_size, 1]
    input_ids: Tensor,
    /// Output hidden states / logits
    output: Tensor,
}

/// A captured CUDA graph with its associated buffers.
#[cfg(feature = "cuda-kernels")]
struct CapturedGraph {
    /// The captured graph handle
    graph: candle_core::cuda::cudarc::driver::sys::CUgraph,
    /// The executable graph
    graph_exec: candle_core::cuda::cudarc::driver::sys::CUgraphExec,
    /// Pre-allocated buffers
    buffers: CaptureBuffers,
}

// SAFETY: CUDA graph handles are thread-safe when properly synchronized.
// The CudaGraphRunner uses interior mutability with proper locking.
#[cfg(feature = "cuda-kernels")]
unsafe impl Send for CapturedGraph {}
#[cfg(feature = "cuda-kernels")]
unsafe impl Sync for CapturedGraph {}

#[cfg(feature = "cuda-kernels")]
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        use candle_core::cuda::cudarc::driver::sys::{cuGraphDestroy, cuGraphExecDestroy};
        unsafe {
            if !self.graph_exec.is_null() {
                cuGraphExecDestroy(self.graph_exec);
            }
            if !self.graph.is_null() {
                cuGraphDestroy(self.graph);
            }
        }
    }
}

/// CUDA Graph Runner manages capture and replay of model forward passes.
///
/// This struct owns the captured graphs and provides methods for:
/// - Warmup: Capturing graphs for configured batch sizes
/// - Execution: Replaying graphs or falling back to eager execution
pub struct CudaGraphRunner {
    config: CudaGraphConfig,
    /// Device for tensor operations
    device: Device,
    /// Vocabulary size for output allocation
    #[allow(dead_code)] // Used during warmup in capture_fn closure
    vocab_size: usize,
    /// DType for tensors
    #[allow(dead_code)] // Reserved for future output dtype configuration
    dtype: DType,
    /// Captured graphs by batch size
    #[cfg(feature = "cuda-kernels")]
    graphs: HashMap<usize, CapturedGraph>,
    /// Mapping from actual batch size to padded size
    size_mapping: HashMap<usize, usize>,
    /// Whether warmup has been completed
    warmed_up: bool,
}

impl CudaGraphRunner {
    /// Create a new CUDA Graph Runner.
    ///
    /// # Arguments
    /// * `config` - CUDA graph configuration
    /// * `device` - CUDA device for tensor operations
    /// * `vocab_size` - Vocabulary size for output allocation
    /// * `dtype` - Data type for tensors
    pub fn new(config: CudaGraphConfig, device: Device, vocab_size: usize, dtype: DType) -> Self {
        // Build size mapping: actual size -> padded size
        let mut size_mapping = HashMap::new();
        if config.enabled {
            let mut sorted_sizes: Vec<_> = config
                .capture_sizes
                .iter()
                .filter(|&&s| s <= config.max_capture_size)
                .copied()
                .collect();
            sorted_sizes.sort();

            for actual_size in 1..=config.max_capture_size {
                // Find smallest capture size >= actual_size
                if let Some(&padded) = sorted_sizes.iter().find(|&&s| s >= actual_size) {
                    size_mapping.insert(actual_size, padded);
                }
            }
        }

        Self {
            config,
            device,
            vocab_size,
            dtype,
            #[cfg(feature = "cuda-kernels")]
            graphs: HashMap::new(),
            size_mapping,
            warmed_up: false,
        }
    }

    /// Check if CUDA graphs are enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && self.device.is_cuda()
    }

    /// Check if warmup is needed.
    pub fn needs_warmup(&self) -> bool {
        self.is_enabled() && !self.warmed_up
    }

    /// Get the padded batch size for a given actual batch size.
    ///
    /// Returns `None` if no graph is available for this size.
    pub fn get_padded_size(&self, actual_size: usize) -> Option<usize> {
        if !self.is_enabled() || !self.warmed_up {
            return None;
        }
        self.size_mapping.get(&actual_size).copied()
    }

    /// Check if a graph is available for the given batch size.
    #[cfg(feature = "cuda-kernels")]
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    #[cfg(not(feature = "cuda-kernels"))]
    pub fn has_graph(&self, _batch_size: usize) -> bool {
        false
    }

    /// Run warmup to capture CUDA graphs for all configured batch sizes.
    ///
    /// # Arguments
    /// * `capture_fn` - Function that executes model forward pass.
    ///   Takes (input_ids, batch_size) and returns output tensor.
    ///
    /// # Returns
    /// Number of graphs successfully captured.
    #[cfg(feature = "cuda-kernels")]
    pub fn warmup<F>(&mut self, mut capture_fn: F) -> Result<usize, CudaGraphRunnerError>
    where
        F: FnMut(&Tensor) -> candle_core::Result<Tensor>,
    {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphInstantiateWithFlags, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph,
            CUgraphExec, CUresult, CUstreamCaptureMode,
        };

        if !self.is_enabled() {
            self.warmed_up = true;
            return Ok(0);
        }

        // Get CUDA stream from device
        let stream = self.get_cuda_stream()?;

        let mut captured = 0;
        let capture_sizes: Vec<_> = self
            .config
            .capture_sizes
            .iter()
            .filter(|&&s| s <= self.config.max_capture_size)
            .copied()
            .collect();

        // Capture largest sizes first (for memory efficiency)
        let mut sorted_sizes = capture_sizes.clone();
        sorted_sizes.sort_by(|a, b| b.cmp(a));

        for batch_size in sorted_sizes {
            // Allocate buffers for this batch size
            let input_ids =
                Tensor::zeros((batch_size, 1), DType::U32, &self.device).map_err(|e| {
                    CudaGraphRunnerError::BufferAllocation(format!(
                        "Failed to allocate input buffer: {}",
                        e
                    ))
                })?;

            // Warmup run (without capture) to allocate all internal buffers
            let warmup_output = capture_fn(&input_ids).map_err(|e| {
                CudaGraphRunnerError::WarmupFailed(format!("Warmup forward failed: {}", e))
            })?;

            // Allocate output buffer matching warmup output shape
            let output_shape = warmup_output.dims().to_vec();
            let output =
                Tensor::zeros(&output_shape[..], self.dtype, &self.device).map_err(|e| {
                    CudaGraphRunnerError::BufferAllocation(format!(
                        "Failed to allocate output buffer: {}",
                        e
                    ))
                })?;

            // Synchronize before capture
            self.sync_device()?;

            // Begin capture
            let capture_result = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            };
            if capture_result != CUresult::CUDA_SUCCESS {
                // Skip this batch size if capture fails
                continue;
            }

            // Run forward pass (operations are recorded, not executed)
            let capture_output = match capture_fn(&input_ids) {
                Ok(out) => out,
                Err(_) => {
                    // Abort capture on error
                    let mut graph: CUgraph = std::ptr::null_mut();
                    unsafe {
                        cuStreamEndCapture(stream, &mut graph);
                    }
                    continue;
                }
            };

            // Copy the captured forward output into the pre-allocated output buffer.
            // This records the memcpy as part of the CUDA graph so that on replay,
            // the graph writes its results into the same output buffer.
            cuda_memcpy_inplace(&output, &capture_output).map_err(|e| {
                CudaGraphRunnerError::WarmupFailed(format!(
                    "Failed to record output copy in graph: {}",
                    e
                ))
            })?;

            // End capture
            let mut graph: CUgraph = std::ptr::null_mut();
            let end_result = unsafe { cuStreamEndCapture(stream, &mut graph) };
            if end_result != CUresult::CUDA_SUCCESS || graph.is_null() {
                continue;
            }

            // Instantiate graph
            let mut graph_exec: CUgraphExec = std::ptr::null_mut();
            let inst_result = unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
            if inst_result != CUresult::CUDA_SUCCESS {
                unsafe {
                    candle_core::cuda::cudarc::driver::sys::cuGraphDestroy(graph);
                }
                continue;
            }

            // Store captured graph
            self.graphs.insert(
                batch_size,
                CapturedGraph {
                    graph,
                    graph_exec,
                    buffers: CaptureBuffers { input_ids, output },
                },
            );
            captured += 1;
        }

        self.warmed_up = true;
        Ok(captured)
    }

    #[cfg(not(feature = "cuda-kernels"))]
    pub fn warmup<F>(&mut self, _capture_fn: F) -> Result<usize, CudaGraphRunnerError>
    where
        F: FnMut(&Tensor) -> candle_core::Result<Tensor>,
    {
        self.warmed_up = true;
        Ok(0)
    }

    /// Capture a CUDA graph for a single decode batch size.
    ///
    /// Calls `capture_fn` twice: once outside the capture stream as a JIT
    /// warmup (so any per-shape kernel compilation happens before recording),
    /// and once inside `cuStreamBeginCapture_v2` / `cuStreamEndCapture` to
    /// record the forward pass.  The output of the recorded run is copied
    /// into a pre-allocated buffer so subsequent replays can read results
    /// from a stable address.
    ///
    /// `capture_fn` receives the runner-owned input buffer (shape `[batch_size, 1]`,
    /// `DType::U32`) — callers that need additional metadata (sequences,
    /// kv-cache manager, etc.) should close over them in the closure.
    ///
    /// Returns `Ok(true)` on a successful capture, `Ok(false)` if any of
    /// the CUDA capture/instantiate steps fail (caller should fall back to
    /// eager); errors propagate only from buffer allocation, the warmup
    /// forward, or the in-place output copy.
    #[cfg(feature = "cuda-kernels")]
    pub fn try_capture_decode_step<F>(
        &mut self,
        batch_size: usize,
        mut capture_fn: F,
    ) -> Result<bool, CudaGraphRunnerError>
    where
        F: FnMut(&Tensor) -> candle_core::Result<Tensor>,
    {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphInstantiateWithFlags, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph,
            CUgraphExec, CUresult, CUstreamCaptureMode,
        };

        if !self.is_enabled() {
            return Ok(false);
        }

        let stream = self.get_cuda_stream()?;

        // Stage 13-A diagnostic: at first call, dump CUDA capability state
        // so we can classify the capture failure (V1 — no memory pools,
        // V2 — wrong stream/alloc path, V3 — cross-stream events).
        log_capture_diagnostics_once(&self.device, stream);
        run_alloc_only_capture_probe_once(&self.device, stream, self.dtype);

        // Pre-allocated input buffer; identity-stable address that the
        // captured graph reads from on every replay.
        let input_ids = Tensor::zeros((batch_size, 1), DType::U32, &self.device).map_err(|e| {
            CudaGraphRunnerError::BufferAllocation(format!("Failed to allocate input buffer: {e}"))
        })?;

        // First call is JIT warmup (allocates internal buffers, compiles
        // any per-shape kernels) — running it inside capture would record
        // the allocation, which is illegal.
        // 2026-05-09 (Phase B.5): rewind pool cursors before BOTH the
        // warmup and captured passes. The warmup pass primes JIT
        // compilation + grows the pool to whatever maximum the forward
        // needs; the captured pass reuses those slots from cursor 0.
        // Real runtime forwards (engine/helpers.rs) also reset cursors
        // at the start, so they reuse the same slot 0..N buffers as
        // the captured pass — addresses match.
        super::output_pool::OutputPool::global().reset_cursors();
        let warmup_output = capture_fn(&input_ids)
            .map_err(|e| CudaGraphRunnerError::WarmupFailed(format!("Warmup forward: {e}")))?;

        let output_shape = warmup_output.dims().to_vec();
        // 2026-05-09: use the warmup forward's actual dtype, not `self.dtype`
        // (which is the model's compute dtype, typically BF16). The forward
        // returns logits in F32, and `cuda_memcpy_inplace` rejects dtype
        // mismatches. Dormant before today because event-tracking blocked
        // capture before the dtype check ever fired.
        let output = Tensor::zeros(&output_shape[..], warmup_output.dtype(), &self.device)
            .map_err(|e| {
                CudaGraphRunnerError::BufferAllocation(format!(
                    "Failed to allocate output buffer: {e}"
                ))
            })?;

        self.sync_device()?;

        // 2026-05-09: rewind the global OutputPool's per-shape cursors before
        // the captured forward pass. The captured graph encodes the device
        // addresses of pool slots used inside it; replays at runtime call
        // `reset_cursors()` at the start of every decode forward
        // (engine/helpers.rs) — to make replay addresses match capture
        // addresses, the captured pass must also start from cursor=0.
        // The earlier `warmup_output` invocation may have advanced the
        // cursors; without this reset, captured slot indices were N+1, N+2
        // etc. while replay used 0, 1, 2, … and the graph dereferenced
        // stale pool slots.
        super::output_pool::OutputPool::global().reset_cursors();

        // RELAXED capture mode: GLOBAL aborts the capture the moment any
        // op inside it touches a buffer that still has outstanding work on
        // a stream other than the captured one. With cudarc's auto event
        // tracking that happens on the very first kernel: every CudaSlice
        // touched by warmup carries an event-edge to the warmup stream
        // state. RELAXED allows dependencies on uncaptured work to be
        // resolved at submission time, which is exactly what we want for
        // the cold-start replay.
        let begin_result = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        if begin_result != CUresult::CUDA_SUCCESS {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                cuda_result = begin_result as i32,
                batch_size,
                "cuStreamBeginCapture_v2 failed"
            );
            return Ok(false);
        }

        let capture_output = match capture_fn(&input_ids) {
            Ok(out) => out,
            Err(e) => {
                // Abort the capture cleanly — even if the forward errored
                // we still need to consume the stream's capture state.
                let mut graph: CUgraph = std::ptr::null_mut();
                unsafe {
                    cuStreamEndCapture(stream, &mut graph);
                    if !graph.is_null() {
                        candle_core::cuda::cudarc::driver::sys::cuGraphDestroy(graph);
                    }
                }
                tracing::warn!(
                    target: "vllm_core::cuda_graph",
                    error = %e,
                    batch_size,
                    "graph capture aborted: forward pass errored inside cuStreamBeginCapture; \
                     a kernel is doing host-blocking work or a non-captureable allocation. \
                     Eager path remains live for this batch."
                );
                return Ok(false);
            }
        };

        // Stage 13-A: localize where capture state died. If status is
        // INVALIDATED here, the forward itself broke capture (kernel-internal
        // alloc, host pull, or cross-stream event). If still ACTIVE, the
        // failure happens later during memcpy_inplace or cuStreamEndCapture.
        log_capture_status_once_after_forward(stream);

        // Record the device→device copy into the stable output buffer as
        // part of the graph; on replay this gives the caller a fixed read
        // address and lets us narrow() it back to the requested shape.
        cuda_memcpy_inplace(&output, &capture_output)?;

        let mut graph: CUgraph = std::ptr::null_mut();
        let end_result = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if end_result != CUresult::CUDA_SUCCESS || graph.is_null() {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                cuda_result = end_result as i32,
                graph_null = graph.is_null(),
                batch_size,
                "cuStreamEndCapture failed — most often means the captured stream \
                 hit a non-captureable operation (cuMemAlloc, sync device→host copy, \
                 etc.) and the driver invalidated the capture state. \
                 CUDA error 900 = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED."
            );
            return Ok(false);
        }

        let mut graph_exec: CUgraphExec = std::ptr::null_mut();
        let inst_result = unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        if inst_result != CUresult::CUDA_SUCCESS {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                cuda_result = inst_result as i32,
                batch_size,
                "cuGraphInstantiateWithFlags failed"
            );
            unsafe {
                candle_core::cuda::cudarc::driver::sys::cuGraphDestroy(graph);
            }
            return Ok(false);
        }

        self.graphs.insert(
            batch_size,
            CapturedGraph {
                graph,
                graph_exec,
                buffers: CaptureBuffers { input_ids, output },
            },
        );

        Ok(true)
    }

    #[cfg(not(feature = "cuda-kernels"))]
    pub fn try_capture_decode_step<F>(
        &mut self,
        _batch_size: usize,
        _capture_fn: F,
    ) -> Result<bool, CudaGraphRunnerError>
    where
        F: FnMut(&Tensor) -> candle_core::Result<Tensor>,
    {
        Ok(false)
    }

    /// Mark warmup as complete.  Call once after all
    /// `try_capture_decode_step` calls have run so subsequent
    /// `get_padded_size` / `execute` calls dispatch to captured graphs.
    pub fn mark_warmed_up(&mut self) {
        self.warmed_up = true;
        #[cfg(feature = "cuda-kernels")]
        tracing::info!(
            target: "vllm_core::cuda_graph",
            "mark_warmed_up: graphs_count={}, size_mapping_count={}",
            self.graphs.len(),
            self.size_mapping.len()
        );
    }

    /// Execute with CUDA graph if available, otherwise fall back to eager.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch_size, 1]
    /// * `eager_fn` - Function for eager execution if no graph available
    ///
    /// # Returns
    /// Output tensor (logits)
    #[cfg(feature = "cuda-kernels")]
    pub fn execute<F>(
        &self,
        input_ids: &Tensor,
        eager_fn: F,
    ) -> Result<Tensor, CudaGraphRunnerError>
    where
        F: FnOnce(&Tensor) -> candle_core::Result<Tensor>,
    {
        use candle_core::cuda::cudarc::driver::sys::{cuGraphLaunch, CUresult};

        let batch_size = input_ids.dims()[0];

        // Check if we have a graph for this batch size
        let padded_size = self.get_padded_size(batch_size);
        static FIRST_DECISION: std::sync::Once = std::sync::Once::new();
        FIRST_DECISION.call_once(|| {
            tracing::info!(
                target: "vllm_core::cuda_graph",
                "execute() entered (first call): batch_size={}, padded_size={:?}, \
                 graphs_count={}, is_enabled={}, warmed_up={}",
                batch_size,
                padded_size,
                self.graphs.len(),
                self.is_enabled(),
                self.warmed_up,
            );
        });
        if let Some(padded_size) = padded_size {
            if let Some(captured) = self.graphs.get(&padded_size) {
                static FIRST: std::sync::Once = std::sync::Once::new();
                FIRST.call_once(|| {
                    tracing::info!(
                        target: "vllm_core::cuda_graph",
                        "CUDA graph replay live (first replay: batch_size={}, padded={})",
                        batch_size,
                        padded_size
                    );
                });
                // Prepare input: pad if necessary
                let padded_input = if batch_size < padded_size {
                    // Create padded input by concatenating with zeros
                    let padding_size = padded_size - batch_size;
                    let padding = Tensor::zeros((padding_size, 1), DType::U32, &self.device)
                        .map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))?;
                    let input_contiguous = input_ids
                        .contiguous()
                        .map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))?;
                    Tensor::cat(&[&input_contiguous, &padding], 0)
                        .map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))?
                } else {
                    input_ids
                        .contiguous()
                        .map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))?
                };

                // Copy new input data into the captured graph's input buffer.
                // The graph was recorded reading from this buffer, so updating
                // it in-place before replay feeds the new tokens to the model.
                cuda_memcpy_inplace(&captured.buffers.input_ids, &padded_input).map_err(|e| {
                    CudaGraphRunnerError::ExecutionFailed(format!(
                        "Failed to copy input for graph replay: {}",
                        e
                    ))
                })?;

                // Get stream and replay graph
                let stream = self.get_cuda_stream()?;
                let result = unsafe { cuGraphLaunch(captured.graph_exec, stream) };
                if result != CUresult::CUDA_SUCCESS {
                    return Err(CudaGraphRunnerError::ReplayFailed(result as i32));
                }

                // Sync to ensure graph execution completes
                self.sync_device()?;

                // Return slice of output buffer (unpad)
                let output = if batch_size < padded_size {
                    captured
                        .buffers
                        .output
                        .narrow(0, 0, batch_size)
                        .map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))?
                } else {
                    captured.buffers.output.clone()
                };

                return Ok(output);
            }
        }

        // Fall back to eager execution
        eager_fn(input_ids).map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))
    }

    #[cfg(not(feature = "cuda-kernels"))]
    pub fn execute<F>(
        &self,
        input_ids: &Tensor,
        eager_fn: F,
    ) -> Result<Tensor, CudaGraphRunnerError>
    where
        F: FnOnce(&Tensor) -> candle_core::Result<Tensor>,
    {
        eager_fn(input_ids).map_err(|e| CudaGraphRunnerError::ExecutionFailed(e.to_string()))
    }

    /// Get statistics about captured graphs.
    pub fn stats(&self) -> CudaGraphRunnerStats {
        #[cfg(feature = "cuda-kernels")]
        let num_graphs = self.graphs.len();
        #[cfg(not(feature = "cuda-kernels"))]
        let num_graphs = 0;

        CudaGraphRunnerStats {
            enabled: self.is_enabled(),
            warmed_up: self.warmed_up,
            num_captured_graphs: num_graphs,
            capture_sizes: self.config.capture_sizes.clone(),
        }
    }

    /// Get CUDA stream from device.
    #[cfg(feature = "cuda-kernels")]
    fn get_cuda_stream(
        &self,
    ) -> Result<candle_core::cuda::cudarc::driver::sys::CUstream, CudaGraphRunnerError> {
        // candle 0.9 / cudarc 0.16 dispatch every kernel through the
        // CudaContext's *default* stream, and `default_stream()` returns
        // a CudaStream whose `cu_stream` is `std::ptr::null_mut()` — the
        // legacy null stream.  CUDA stream capture is unconditionally
        // unsupported on that stream:
        //
        //   `cuStreamBeginCapture_v2` returns
        //   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (900).
        //
        // Picking a freshly created `cuStreamCreate` stream here would
        // capture *that* stream's queue (which is empty), not the queue
        // candle is feeding. So unless candle is taught to use a
        // non-default stream, every `try_capture_decode_step` call will
        // hit error 900, fall through to `Ok(false)`, and the engine
        // will run eager forever.  This is a known upstream limitation;
        // `mark_warmed_up` keeps `warmed_up = true` so callers see a
        // consistent "tried, fell back" path rather than an unbounded
        // retry loop.
        match &self.device {
            candle_core::Device::Cuda(cuda_device) => Ok(cuda_device.cuda_stream().cu_stream()),
            _ => Err(CudaGraphRunnerError::DeviceError(
                "Device is not CUDA".to_string(),
            )),
        }
    }

    /// Synchronize CUDA device.
    #[cfg(feature = "cuda-kernels")]
    fn sync_device(&self) -> Result<(), CudaGraphRunnerError> {
        use candle_core::cuda::cudarc::driver::sys::{cuCtxSynchronize, CUresult};
        let result = unsafe { cuCtxSynchronize() };
        if result != CUresult::CUDA_SUCCESS {
            return Err(CudaGraphRunnerError::SyncFailed(result as i32));
        }
        Ok(())
    }
}

/// Stage 13-A diagnostic. Dumps CUDA capture pre-conditions exactly once
/// per process so we can classify why `cuStreamBeginCapture_v2`/in-window ops
/// fail on this specific GPU + driver:
///
/// - `memory_pools_supported` (CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, id 115):
///   if 0, cudarc 0.16.6 falls back to synchronous `cuMemAlloc` which is
///   capture-incompatible → expect error 905 (CAPTURE_ISOLATION) on the
///   first allocation inside the capture window.
/// - `stream_ptr_null`: should be false after the Stage 11 candle patch
///   (vendored `new_stream()`). If true, the patch did not take effect.
#[cfg(feature = "cuda-kernels")]
fn log_capture_diagnostics_once(
    device: &Device,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
) {
    use candle_core::cuda::cudarc::driver::sys::{
        cuDeviceGetAttribute, CUdevice_attribute, CUresult,
    };
    use std::sync::Once;

    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let cuda_device = match device {
            Device::Cuda(d) => d,
            _ => return,
        };
        let cu_device = cuda_device.cuda_stream().context().cu_device();
        let mut memory_pools_supported: i32 = -1;
        let attr_result = unsafe {
            cuDeviceGetAttribute(
                &mut memory_pools_supported,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                cu_device,
            )
        };
        let attr_ok = attr_result == CUresult::CUDA_SUCCESS;
        let stream_ptr_null = stream.is_null();

        tracing::warn!(
            target: "vllm_core::cuda_graph",
            stage = "13-A",
            stream_ptr_null,
            memory_pools_supported = if attr_ok { memory_pools_supported } else { -1 },
            attr_query_ok = attr_ok,
            "CUDA graph capture diagnostic: stream/alloc preconditions"
        );

        if !attr_ok {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                cuda_result = attr_result as i32,
                "cuDeviceGetAttribute(MEMORY_POOLS_SUPPORTED) failed — async-alloc unknown"
            );
        } else if memory_pools_supported == 0 {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                "Verdict candidate V1: device does NOT support memory pools — \
                 cudarc 0.16 will issue synchronous cuMemAlloc, capture will hit 905. \
                 Remediation: force cuMemAllocAsync via vendored cudarc patch."
            );
        } else {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                "Memory pools supported — async-alloc should be active. \
                 If capture still fails with 905, blocker is V2 (alloc path bypass) \
                 or V3 (cross-stream events). Check next warmup attempt logs."
            );
        }
    });
}

/// Stage 13-A.2 probe: does a TRIVIAL allocation inside a capture window
/// already invalidate? Allocates one BF16 tensor between BeginCapture and
/// EndCapture. If this fails, root cause is the alloc path itself (V1 or V2).
/// Run-once per process; logs result + any failure code.
#[cfg(feature = "cuda-kernels")]
fn run_alloc_only_capture_probe_once(
    device: &Device,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    dtype: DType,
) {
    use candle_core::cuda::cudarc::driver::sys::{
        cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture, cuStreamGetCaptureInfo_v2,
        CUgraph, CUresult, CUstreamCaptureMode, CUstreamCaptureStatus,
    };
    use std::sync::Once;

    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Best-effort sync; ignore errors here — purely diagnostic.
        unsafe {
            let _ = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let begin_result = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        if begin_result != CUresult::CUDA_SUCCESS {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                stage = "13-A.2",
                cuda_result = begin_result as i32,
                "alloc-only probe: cuStreamBeginCapture_v2 failed at probe entry"
            );
            return;
        }

        // Single trivial allocation inside the capture window.
        let alloc_result = Tensor::zeros((1024_usize,), dtype, device);
        let alloc_ok = alloc_result.is_ok();
        let alloc_err = alloc_result.as_ref().err().map(|e| e.to_string());

        // Status snapshot before EndCapture: did the alloc invalidate the stream?
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id: u64 = 0;
        let mut graph_probe: CUgraph = std::ptr::null_mut();
        let mut deps_ptr: *const candle_core::cuda::cudarc::driver::sys::CUgraphNode =
            std::ptr::null();
        let mut deps_count: usize = 0;
        let info_result = unsafe {
            cuStreamGetCaptureInfo_v2(
                stream,
                &mut status,
                &mut id as *mut u64,
                &mut graph_probe,
                &mut deps_ptr,
                &mut deps_count,
            )
        };

        let mut graph: CUgraph = std::ptr::null_mut();
        let end_result = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if !graph.is_null() {
            unsafe {
                cuGraphDestroy(graph);
            }
        }

        let status_str = match status {
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => "NONE",
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => "ACTIVE",
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => "INVALIDATED",
        };

        tracing::warn!(
            target: "vllm_core::cuda_graph",
            stage = "13-A.2",
            alloc_ok,
            alloc_err = alloc_err.as_deref().unwrap_or(""),
            info_result = info_result as i32,
            status_after_alloc = status_str,
            end_result = end_result as i32,
            "alloc-only capture probe: did a single Tensor::zeros invalidate capture?"
        );

        if status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                "Verdict candidate V1/V2: a single Tensor::zeros INSIDE capture \
                 invalidated the stream. The alloc path is not capture-safe \
                 (sync cuMemAlloc, htod path, or cross-stream event)."
            );
        } else if status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            tracing::warn!(
                target: "vllm_core::cuda_graph",
                "Alloc-only probe SURVIVED capture. cuMemAllocAsync is active \
                 (or alloc was elided). The forward-time 905 must come from \
                 something OTHER than a single Tensor::zeros — likely \
                 a host-blocking op (Tensor::from_vec / .to_vec) or cross-stream \
                 event. Verdict candidate: V2 or V3."
            );
        }
    });
}

/// Stage 13-A.2 status check: log whether stream is still being captured
/// after `capture_fn` returns. Run-once.
#[cfg(feature = "cuda-kernels")]
fn log_capture_status_once_after_forward(stream: candle_core::cuda::cudarc::driver::sys::CUstream) {
    use candle_core::cuda::cudarc::driver::sys::{
        cuStreamGetCaptureInfo_v2, CUgraph, CUgraphNode, CUstreamCaptureStatus,
    };
    use std::sync::Once;

    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id: u64 = 0;
        let mut graph_probe: CUgraph = std::ptr::null_mut();
        let mut deps_ptr: *const CUgraphNode = std::ptr::null();
        let mut deps_count: usize = 0;
        let info_result = unsafe {
            cuStreamGetCaptureInfo_v2(
                stream,
                &mut status,
                &mut id as *mut u64,
                &mut graph_probe,
                &mut deps_ptr,
                &mut deps_count,
            )
        };
        let status_str = match status {
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => "NONE",
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => "ACTIVE",
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => "INVALIDATED",
        };
        tracing::warn!(
            target: "vllm_core::cuda_graph",
            stage = "13-A.2",
            info_result = info_result as i32,
            status_after_forward = status_str,
            "capture status snapshot after forward — INVALIDATED means forward broke it; \
             ACTIVE means failure is in memcpy_inplace or EndCapture."
        );
    });
}

/// Statistics about CUDA graph runner state.
#[derive(Debug, Clone)]
pub struct CudaGraphRunnerStats {
    /// Whether CUDA graphs are enabled
    pub enabled: bool,
    /// Whether warmup has been completed
    pub warmed_up: bool,
    /// Number of captured graphs
    pub num_captured_graphs: usize,
    /// Configured capture sizes
    pub capture_sizes: Vec<usize>,
}

/// Errors from CUDA graph runner operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaGraphRunnerError {
    #[error("buffer allocation failed: {0}")]
    BufferAllocation(String),
    #[error("warmup failed: {0}")]
    WarmupFailed(String),
    #[error("device error: {0}")]
    DeviceError(String),
    #[error("sync failed with CUDA error {0}")]
    SyncFailed(i32),
    #[error("capture failed with CUDA error {0}")]
    CaptureFailed(i32),
    #[error("replay failed with CUDA error {0}")]
    ReplayFailed(i32),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("in-place copy failed: {0}")]
    InplaceCopyFailed(String),
}

/// Builder for creating CUDA graph runner with proper configuration.
pub struct CudaGraphRunnerBuilder {
    config: CudaGraphConfig,
    device: Option<Device>,
    vocab_size: Option<usize>,
    dtype: DType,
}

impl CudaGraphRunnerBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: CudaGraphConfig::default(),
            device: None,
            vocab_size: None,
            dtype: DType::F32,
        }
    }

    /// Set the CUDA graph configuration.
    pub fn config(mut self, config: CudaGraphConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the vocabulary size.
    pub fn vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    /// Set the data type.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Build the CUDA graph runner.
    pub fn build(self) -> Result<CudaGraphRunner, &'static str> {
        let device = self.device.ok_or("device is required")?;
        let vocab_size = self.vocab_size.ok_or("vocab_size is required")?;

        Ok(CudaGraphRunner::new(
            self.config,
            device,
            vocab_size,
            self.dtype,
        ))
    }
}

impl Default for CudaGraphRunnerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_disabled() {
        let config = CudaGraphConfig::default();
        let runner = CudaGraphRunner::new(config, Device::Cpu, 32000, DType::F32);

        assert!(!runner.is_enabled());
        assert!(!runner.needs_warmup());
        assert!(!runner.has_graph(4));
    }

    #[test]
    fn test_size_mapping() {
        let config = CudaGraphConfig {
            enabled: true,
            mode: RuntimeMode::Full,
            capture_sizes: vec![1, 4, 8, 16],
            max_capture_size: 32,
            graph_pool_size: 256 * 1024 * 1024,
        };
        // Note: CPU device means graphs won't actually be enabled
        let runner = CudaGraphRunner::new(config, Device::Cpu, 32000, DType::F32);

        // CPU device means is_enabled() returns false
        assert!(!runner.is_enabled());

        // Size mapping is still computed
        assert_eq!(runner.size_mapping.get(&1), Some(&1));
        assert_eq!(runner.size_mapping.get(&2), Some(&4));
        assert_eq!(runner.size_mapping.get(&3), Some(&4));
        assert_eq!(runner.size_mapping.get(&4), Some(&4));
        assert_eq!(runner.size_mapping.get(&5), Some(&8));
        assert_eq!(runner.size_mapping.get(&16), Some(&16));
        assert_eq!(runner.size_mapping.get(&17), None); // > max configured size
    }

    #[test]
    fn test_builder() {
        let runner = CudaGraphRunnerBuilder::new()
            .config(CudaGraphConfig::enabled())
            .device(Device::Cpu)
            .vocab_size(32000)
            .dtype(DType::BF16)
            .build()
            .unwrap();

        assert!(!runner.is_enabled()); // CPU device
        assert_eq!(runner.vocab_size, 32000);
    }

    #[test]
    fn test_builder_missing_device() {
        let result = CudaGraphRunnerBuilder::new().vocab_size(32000).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_stats() {
        let config = CudaGraphConfig::default();
        let runner = CudaGraphRunner::new(config.clone(), Device::Cpu, 32000, DType::F32);

        let stats = runner.stats();
        assert!(!stats.enabled);
        assert!(!stats.warmed_up);
        assert_eq!(stats.num_captured_graphs, 0);
        assert_eq!(stats.capture_sizes, config.capture_sizes);
    }

    #[test]
    fn test_warmup_cpu() {
        let config = CudaGraphConfig::default();
        let mut runner = CudaGraphRunner::new(config, Device::Cpu, 32000, DType::F32);

        // Warmup should succeed but capture 0 graphs on CPU
        let result = runner.warmup(|_input| Tensor::zeros((1, 32000), DType::F32, &Device::Cpu));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert!(runner.warmed_up);
    }

    #[test]
    fn test_execute_eager_fallback() {
        let config = CudaGraphConfig::default();
        let runner = CudaGraphRunner::new(config, Device::Cpu, 32000, DType::F32);

        let input = Tensor::zeros((4, 1), DType::U32, &Device::Cpu).unwrap();
        let result = runner.execute(&input, |_| {
            Tensor::zeros((4, 32000), DType::F32, &Device::Cpu)
        });

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims(), &[4, 32000]);
    }

    // ─── In-place copy tests ──────────────────────────────────────────────

    #[test]
    fn test_inplace_copy_f32() {
        let device = Device::Cpu;
        let src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let dst = Tensor::zeros((2, 2), DType::F32, &device).unwrap();

        cuda_memcpy_inplace(&dst, &src).unwrap();

        let dst_data: Vec<f32> = dst.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(dst_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_inplace_copy_u32() {
        let device = Device::Cpu;
        let src = Tensor::from_vec(vec![10u32, 20, 30, 40], (4, 1), &device).unwrap();
        let dst = Tensor::zeros((4, 1), DType::U32, &device).unwrap();

        cuda_memcpy_inplace(&dst, &src).unwrap();

        let dst_data: Vec<u32> = dst.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(dst_data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_inplace_copy_bf16() {
        let device = Device::Cpu;
        let src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let dst = Tensor::zeros((2, 2), DType::BF16, &device).unwrap();

        cuda_memcpy_inplace(&dst, &src).unwrap();

        let dst_f32 = dst.to_dtype(DType::F32).unwrap();
        let dst_data: Vec<f32> = dst_f32.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(dst_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_inplace_copy_overwrites_existing_data() {
        let device = Device::Cpu;
        let dst = Tensor::from_vec(vec![100.0f32, 200.0, 300.0], (3,), &device).unwrap();
        let src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();

        cuda_memcpy_inplace(&dst, &src).unwrap();

        let dst_data: Vec<f32> = dst.to_vec1().unwrap();
        assert_eq!(dst_data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_inplace_copy_dtype_mismatch_rejected() {
        let device = Device::Cpu;
        let src = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let dst = Tensor::zeros((2, 2), DType::U32, &device).unwrap();

        let result = cuda_memcpy_inplace(&dst, &src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("dtype mismatch"),
            "Expected dtype mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn test_inplace_copy_shape_mismatch_rejected() {
        let device = Device::Cpu;
        let src = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        let dst = Tensor::zeros((3, 3), DType::F32, &device).unwrap();

        let result = cuda_memcpy_inplace(&dst, &src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("element count mismatch"),
            "Expected element count mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn test_inplace_copy_same_element_count_different_shape() {
        // Shapes differ but element counts match -- should succeed
        let device = Device::Cpu;
        let src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();
        let dst = Tensor::zeros((3, 2), DType::F32, &device).unwrap();

        cuda_memcpy_inplace(&dst, &src).unwrap();

        let dst_data: Vec<f32> = dst.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(dst_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
