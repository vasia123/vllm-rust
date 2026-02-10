//! CUDA Graph infrastructure for optimized kernel execution.
//!
//! This module provides CUDA Graph capture and replay capabilities to eliminate
//! CPU overhead for repetitive kernel launch patterns during inference.
//!
//! # Architecture
//!
//! The system follows vLLM's design with:
//! - `BatchDescriptor`: Key for dispatching to cached graphs
//! - `CudaGraphDispatcher`: Manages batch shape -> graph mapping
//! - `CudaGraphWrapper`: Captures and replays CUDA graphs
//!
//! # Execution Modes
//!
//! - `None`: Standard eager execution (no graph capture)
//! - `Full`: Single graph captures entire model forward pass
//! - `Piecewise`: Per-layer graphs (allows interleaved operations)

#[cfg(feature = "cuda-kernels")]
use std::collections::HashMap;
#[cfg(feature = "cuda-kernels")]
use std::ptr;
#[cfg(feature = "cuda-kernels")]
use std::sync::Arc;

#[cfg(feature = "cuda-kernels")]
use candle_core::cuda::cudarc::driver::sys::{
    cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
    cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult, CUstream,
    CUstreamCaptureMode,
};

/// Runtime execution mode for CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RuntimeMode {
    /// No CUDA graph - standard eager execution
    #[default]
    None,
    /// Full model forward captured in single graph
    Full,
    /// Per-layer graphs for more flexibility
    Piecewise,
}

/// Describes a batch shape for CUDA graph dispatch.
///
/// Used as a key to look up cached CUDA graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchDescriptor {
    /// Total number of tokens in the batch (padded)
    pub num_tokens: usize,
    /// Number of requests in the batch
    pub num_reqs: usize,
    /// Whether all requests have the same query length (enables more optimizations)
    pub uniform: bool,
}

impl BatchDescriptor {
    /// Create a new batch descriptor.
    pub fn new(num_tokens: usize, num_reqs: usize, uniform: bool) -> Self {
        Self {
            num_tokens,
            num_reqs,
            uniform,
        }
    }

    /// Create descriptor for a decode batch where all sequences generate 1 token.
    pub fn for_decode(num_reqs: usize) -> Self {
        Self {
            num_tokens: num_reqs,
            num_reqs,
            uniform: true,
        }
    }

    /// Create descriptor for a prefill batch.
    pub fn for_prefill(num_tokens: usize, num_reqs: usize) -> Self {
        Self {
            num_tokens,
            num_reqs,
            uniform: num_reqs == 1,
        }
    }
}

/// Configuration for CUDA graph execution.
#[derive(Debug, Clone)]
pub struct CudaGraphConfig {
    /// Enable CUDA graph capture/replay
    pub enabled: bool,
    /// Preferred runtime mode
    pub mode: RuntimeMode,
    /// Batch sizes to pre-capture graphs for during warmup
    pub capture_sizes: Vec<usize>,
    /// Maximum batch size for graph capture (larger batches use eager)
    pub max_capture_size: usize,
    /// Memory pool size for captured graphs (bytes)
    pub graph_pool_size: usize,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: RuntimeMode::None,
            capture_sizes: vec![1, 2, 4, 8, 16, 32],
            max_capture_size: 256,
            graph_pool_size: 256 * 1024 * 1024, // 256 MB
        }
    }
}

impl CudaGraphConfig {
    /// Create config with CUDA graphs enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            mode: RuntimeMode::Full,
            ..Default::default()
        }
    }

    /// Create config with piecewise mode.
    pub fn piecewise() -> Self {
        Self {
            enabled: true,
            mode: RuntimeMode::Piecewise,
            ..Default::default()
        }
    }
}

/// Entry for a cached CUDA graph.
#[cfg(feature = "cuda-kernels")]
struct CudaGraphEntry {
    /// The captured graph (template)
    graph: CUgraph,
    /// The instantiated executable graph
    graph_exec: CUgraphExec,
    /// Input buffer device pointer (for updating inputs)
    #[allow(dead_code)] // Used for buffer updates during replay
    input_ptr: u64,
    /// Output buffer device pointer
    #[allow(dead_code)] // Used for buffer updates during replay
    output_ptr: u64,
}

// SAFETY: CUDA graph handles are opaque pointers that can be accessed from any thread.
// The RwLock in CudaGraphDispatcher ensures synchronized access.
#[cfg(feature = "cuda-kernels")]
unsafe impl Send for CudaGraphEntry {}
#[cfg(feature = "cuda-kernels")]
unsafe impl Sync for CudaGraphEntry {}

#[cfg(feature = "cuda-kernels")]
impl Drop for CudaGraphEntry {
    fn drop(&mut self) {
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

/// Result of CUDA graph dispatch.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Runtime mode to use
    pub mode: RuntimeMode,
    /// Actual batch descriptor (may be padded)
    pub descriptor: BatchDescriptor,
    /// Whether a cached graph exists
    pub graph_available: bool,
}

/// Dispatcher that manages CUDA graph cache and selection.
pub struct CudaGraphDispatcher {
    config: CudaGraphConfig,
    /// Valid batch descriptors for FULL mode
    full_mode_keys: Vec<BatchDescriptor>,
    /// Valid batch descriptors for PIECEWISE mode
    piecewise_mode_keys: Vec<BatchDescriptor>,
    #[cfg(feature = "cuda-kernels")]
    /// Cached graphs by batch descriptor
    graph_cache: HashMap<BatchDescriptor, CudaGraphEntry>,
    #[cfg(not(feature = "cuda-kernels"))]
    /// Placeholder for non-CUDA builds
    _phantom: std::marker::PhantomData<()>,
}

impl CudaGraphDispatcher {
    /// Create a new dispatcher with the given configuration.
    pub fn new(config: CudaGraphConfig) -> Self {
        let full_mode_keys = if config.enabled && config.mode == RuntimeMode::Full {
            config
                .capture_sizes
                .iter()
                .filter(|&&size| size <= config.max_capture_size)
                .map(|&size| BatchDescriptor::for_decode(size))
                .collect()
        } else {
            Vec::new()
        };

        let piecewise_mode_keys = if config.enabled && config.mode == RuntimeMode::Piecewise {
            config
                .capture_sizes
                .iter()
                .filter(|&&size| size <= config.max_capture_size)
                .map(|&size| BatchDescriptor::for_decode(size))
                .collect()
        } else {
            Vec::new()
        };

        Self {
            config,
            full_mode_keys,
            piecewise_mode_keys,
            #[cfg(feature = "cuda-kernels")]
            graph_cache: HashMap::new(),
            #[cfg(not(feature = "cuda-kernels"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check if CUDA graphs are enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current runtime mode.
    pub fn mode(&self) -> RuntimeMode {
        self.config.mode
    }

    /// Dispatch a batch to determine which execution mode to use.
    ///
    /// Returns the runtime mode and whether a cached graph is available.
    pub fn dispatch(&self, descriptor: BatchDescriptor) -> DispatchResult {
        if !self.config.enabled {
            return DispatchResult {
                mode: RuntimeMode::None,
                descriptor,
                graph_available: false,
            };
        }

        // Check if batch size exceeds maximum
        if descriptor.num_reqs > self.config.max_capture_size {
            return DispatchResult {
                mode: RuntimeMode::None,
                descriptor,
                graph_available: false,
            };
        }

        // Find the smallest valid key that fits this batch
        let find_key = |keys: &[BatchDescriptor]| -> Option<BatchDescriptor> {
            keys.iter()
                .filter(|k| k.num_reqs >= descriptor.num_reqs && k.uniform == descriptor.uniform)
                .min_by_key(|k| k.num_reqs)
                .copied()
        };

        // Try FULL mode first (higher priority)
        if let Some(key) = find_key(&self.full_mode_keys) {
            #[cfg(feature = "cuda-kernels")]
            let graph_available = self.graph_cache.contains_key(&key);
            #[cfg(not(feature = "cuda-kernels"))]
            let graph_available = false;

            return DispatchResult {
                mode: RuntimeMode::Full,
                descriptor: key,
                graph_available,
            };
        }

        // Try PIECEWISE mode
        if let Some(key) = find_key(&self.piecewise_mode_keys) {
            #[cfg(feature = "cuda-kernels")]
            let graph_available = self.graph_cache.contains_key(&key);
            #[cfg(not(feature = "cuda-kernels"))]
            let graph_available = false;

            return DispatchResult {
                mode: RuntimeMode::Piecewise,
                descriptor: key,
                graph_available,
            };
        }

        // Fall back to eager execution
        DispatchResult {
            mode: RuntimeMode::None,
            descriptor,
            graph_available: false,
        }
    }

    /// Get the batch sizes that should be warmed up.
    pub fn warmup_sizes(&self) -> &[usize] {
        &self.config.capture_sizes
    }

    /// Register a batch descriptor as valid for the current mode.
    pub fn register_valid_key(&mut self, descriptor: BatchDescriptor) {
        match self.config.mode {
            RuntimeMode::Full => {
                if !self.full_mode_keys.contains(&descriptor) {
                    self.full_mode_keys.push(descriptor);
                }
            }
            RuntimeMode::Piecewise => {
                if !self.piecewise_mode_keys.contains(&descriptor) {
                    self.piecewise_mode_keys.push(descriptor);
                }
            }
            RuntimeMode::None => {}
        }
    }

    /// Get statistics about cached graphs.
    pub fn stats(&self) -> CudaGraphStats {
        #[cfg(feature = "cuda-kernels")]
        let num_cached = self.graph_cache.len();
        #[cfg(not(feature = "cuda-kernels"))]
        let num_cached = 0;

        CudaGraphStats {
            enabled: self.config.enabled,
            mode: self.config.mode,
            num_cached_graphs: num_cached,
            configured_sizes: self.config.capture_sizes.clone(),
            max_capture_size: self.config.max_capture_size,
        }
    }
}

/// Statistics about CUDA graph cache.
#[derive(Debug, Clone)]
pub struct CudaGraphStats {
    /// Whether CUDA graphs are enabled
    pub enabled: bool,
    /// Current runtime mode
    pub mode: RuntimeMode,
    /// Number of cached graphs
    pub num_cached_graphs: usize,
    /// Configured capture sizes
    pub configured_sizes: Vec<usize>,
    /// Maximum batch size for capture
    pub max_capture_size: usize,
}

/// Result of warmup operation.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Number of graphs successfully captured
    pub graphs_captured: usize,
    /// Number of graphs that failed to capture
    pub graphs_failed: usize,
    /// Batch sizes that were warmed up
    pub warmed_sizes: Vec<usize>,
}

impl WarmupResult {
    /// Create a new warmup result for when warmup is skipped.
    pub fn skipped() -> Self {
        Self {
            graphs_captured: 0,
            graphs_failed: 0,
            warmed_sizes: Vec::new(),
        }
    }
}

/// Warmup manager for capturing CUDA graphs during initialization.
///
/// This struct manages the warmup process, which involves running
/// dummy forward passes to capture CUDA graphs for configured batch sizes.
pub struct WarmupManager {
    /// Batch sizes to warm up
    sizes: Vec<usize>,
    /// Whether warmup is enabled
    enabled: bool,
}

impl WarmupManager {
    /// Create a new warmup manager from configuration.
    pub fn new(config: &CudaGraphConfig) -> Self {
        Self {
            sizes: config.capture_sizes.clone(),
            enabled: config.enabled,
        }
    }

    /// Check if warmup is needed.
    pub fn needs_warmup(&self) -> bool {
        self.enabled && !self.sizes.is_empty()
    }

    /// Get the batch sizes to warm up.
    pub fn warmup_sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Run warmup with a provided forward function.
    ///
    /// The `forward_fn` should execute a forward pass for the given batch size
    /// and return true on success, false on failure.
    ///
    /// Note: Full CUDA graph capture requires additional setup (buffer allocation,
    /// stream capture) that's not yet implemented. This method currently just
    /// runs the forward passes for JIT warmup without graph capture.
    pub fn run_warmup<F>(&self, mut forward_fn: F) -> WarmupResult
    where
        F: FnMut(usize) -> bool,
    {
        if !self.enabled {
            return WarmupResult::skipped();
        }

        let mut captured = 0;
        let mut failed = 0;
        let mut warmed = Vec::new();

        for &size in &self.sizes {
            if forward_fn(size) {
                captured += 1;
                warmed.push(size);
            } else {
                failed += 1;
            }
        }

        WarmupResult {
            graphs_captured: captured,
            graphs_failed: failed,
            warmed_sizes: warmed,
        }
    }
}

/// CUDA Graph wrapper for capture and replay.
///
/// Wraps a forward function to automatically capture and replay CUDA graphs
/// based on batch descriptors.
#[cfg(feature = "cuda-kernels")]
pub struct CudaGraphWrapper {
    /// Dispatcher for graph selection
    dispatcher: Arc<std::sync::RwLock<CudaGraphDispatcher>>,
    /// CUDA stream for capture
    stream: CUstream,
    /// Currently capturing
    capturing: bool,
}

#[cfg(feature = "cuda-kernels")]
impl CudaGraphWrapper {
    /// Create a new CUDA graph wrapper.
    pub fn new(dispatcher: Arc<std::sync::RwLock<CudaGraphDispatcher>>, stream: CUstream) -> Self {
        Self {
            dispatcher,
            stream,
            capturing: false,
        }
    }

    /// Begin capturing CUDA operations.
    ///
    /// All subsequent CUDA operations on the stream will be recorded into a graph.
    ///
    /// # Safety
    /// Must be followed by `end_capture()` before any other graph operations.
    pub unsafe fn begin_capture(&mut self) -> Result<(), CudaGraphError> {
        if self.capturing {
            return Err(CudaGraphError::AlreadyCapturing);
        }

        let result = cuStreamBeginCapture_v2(
            self.stream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        );

        if result != CUresult::CUDA_SUCCESS {
            return Err(CudaGraphError::CaptureBeginFailed(result as i32));
        }

        self.capturing = true;
        Ok(())
    }

    /// End capture and instantiate the graph.
    ///
    /// # Safety
    /// Must be called after `begin_capture()`.
    pub unsafe fn end_capture(
        &mut self,
        descriptor: BatchDescriptor,
        input_ptr: u64,
        output_ptr: u64,
    ) -> Result<(), CudaGraphError> {
        if !self.capturing {
            return Err(CudaGraphError::NotCapturing);
        }

        let mut graph: CUgraph = ptr::null_mut();
        let result = cuStreamEndCapture(self.stream, &mut graph);

        self.capturing = false;

        if result != CUresult::CUDA_SUCCESS {
            return Err(CudaGraphError::CaptureEndFailed(result as i32));
        }

        if graph.is_null() {
            return Err(CudaGraphError::EmptyGraph);
        }

        // Instantiate the graph
        let mut graph_exec: CUgraphExec = ptr::null_mut();
        let result = cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0);

        if result != CUresult::CUDA_SUCCESS {
            cuGraphDestroy(graph);
            return Err(CudaGraphError::InstantiateFailed(result as i32));
        }

        let entry = CudaGraphEntry {
            graph,
            graph_exec,
            input_ptr,
            output_ptr,
        };

        let mut dispatcher = self
            .dispatcher
            .write()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        dispatcher.graph_cache.insert(descriptor, entry);
        dispatcher.register_valid_key(descriptor);

        Ok(())
    }

    /// Execute a cached graph.
    ///
    /// # Safety
    /// The input buffers must be updated with the correct data before calling this.
    pub unsafe fn replay(&self, descriptor: &BatchDescriptor) -> Result<(), CudaGraphError> {
        let dispatcher = self
            .dispatcher
            .read()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let entry = dispatcher
            .graph_cache
            .get(descriptor)
            .ok_or(CudaGraphError::GraphNotFound)?;

        let result = cuGraphLaunch(entry.graph_exec, self.stream);

        if result != CUresult::CUDA_SUCCESS {
            return Err(CudaGraphError::LaunchFailed(result as i32));
        }

        Ok(())
    }

    /// Check if a graph is cached for the given descriptor.
    pub fn has_graph(&self, descriptor: &BatchDescriptor) -> bool {
        self.dispatcher
            .read()
            .map(|guard| guard.graph_cache.contains_key(descriptor))
            .unwrap_or(false)
    }

    /// Check if currently capturing.
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }
}

/// Errors that can occur during CUDA graph operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaGraphError {
    #[error("already capturing")]
    AlreadyCapturing,
    #[error("not currently capturing")]
    NotCapturing,
    #[error("capture begin failed with CUDA error {0}")]
    CaptureBeginFailed(i32),
    #[error("capture end failed with CUDA error {0}")]
    CaptureEndFailed(i32),
    #[error("graph instantiation failed with CUDA error {0}")]
    InstantiateFailed(i32),
    #[error("graph launch failed with CUDA error {0}")]
    LaunchFailed(i32),
    #[error("captured graph is empty")]
    EmptyGraph,
    #[error("graph not found in cache")]
    GraphNotFound,
    #[error("dispatcher lock poisoned")]
    LockPoisoned,
}

/// Forward context that tracks the current execution state.
///
/// Used to communicate between dispatcher and wrapper about which
/// mode/graph to use for the current forward pass.
#[derive(Debug, Clone, Default)]
pub struct ForwardContext {
    /// Current runtime mode
    pub mode: RuntimeMode,
    /// Current batch descriptor
    pub descriptor: Option<BatchDescriptor>,
    /// Whether we should capture on this forward
    pub should_capture: bool,
    /// Whether we can replay on this forward
    pub can_replay: bool,
}

impl ForwardContext {
    /// Create a new forward context from dispatch result.
    pub fn from_dispatch(result: DispatchResult) -> Self {
        Self {
            mode: result.mode,
            descriptor: Some(result.descriptor),
            should_capture: result.mode != RuntimeMode::None && !result.graph_available,
            can_replay: result.graph_available,
        }
    }

    /// Create context for eager execution.
    pub fn eager() -> Self {
        Self {
            mode: RuntimeMode::None,
            descriptor: None,
            should_capture: false,
            can_replay: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_descriptor_decode() {
        let desc = BatchDescriptor::for_decode(8);
        assert_eq!(desc.num_tokens, 8);
        assert_eq!(desc.num_reqs, 8);
        assert!(desc.uniform);
    }

    #[test]
    fn test_batch_descriptor_prefill() {
        let desc = BatchDescriptor::for_prefill(128, 1);
        assert_eq!(desc.num_tokens, 128);
        assert_eq!(desc.num_reqs, 1);
        assert!(desc.uniform);

        let desc = BatchDescriptor::for_prefill(256, 2);
        assert!(!desc.uniform);
    }

    #[test]
    fn test_dispatcher_disabled() {
        let config = CudaGraphConfig::default();
        let dispatcher = CudaGraphDispatcher::new(config);

        let result = dispatcher.dispatch(BatchDescriptor::for_decode(8));
        assert_eq!(result.mode, RuntimeMode::None);
        assert!(!result.graph_available);
    }

    #[test]
    fn test_dispatcher_enabled_full_mode() {
        let config = CudaGraphConfig::enabled();
        let dispatcher = CudaGraphDispatcher::new(config);

        // Should find matching key
        let result = dispatcher.dispatch(BatchDescriptor::for_decode(4));
        assert_eq!(result.mode, RuntimeMode::Full);
        assert_eq!(result.descriptor.num_reqs, 4);
        assert!(!result.graph_available);

        // Should find next larger key
        let result = dispatcher.dispatch(BatchDescriptor::for_decode(3));
        assert_eq!(result.mode, RuntimeMode::Full);
        assert_eq!(result.descriptor.num_reqs, 4);

        // Should fall back to eager for too large
        let result = dispatcher.dispatch(BatchDescriptor::for_decode(512));
        assert_eq!(result.mode, RuntimeMode::None);
    }

    #[test]
    fn test_forward_context_from_dispatch() {
        let result = DispatchResult {
            mode: RuntimeMode::Full,
            descriptor: BatchDescriptor::for_decode(8),
            graph_available: false,
        };

        let ctx = ForwardContext::from_dispatch(result);
        assert_eq!(ctx.mode, RuntimeMode::Full);
        assert!(ctx.should_capture);
        assert!(!ctx.can_replay);
    }

    #[test]
    fn test_forward_context_with_cached_graph() {
        let result = DispatchResult {
            mode: RuntimeMode::Full,
            descriptor: BatchDescriptor::for_decode(8),
            graph_available: true,
        };

        let ctx = ForwardContext::from_dispatch(result);
        assert!(!ctx.should_capture);
        assert!(ctx.can_replay);
    }

    #[test]
    fn test_dispatcher_stats() {
        let config = CudaGraphConfig::enabled();
        let dispatcher = CudaGraphDispatcher::new(config);

        let stats = dispatcher.stats();
        assert!(stats.enabled);
        assert_eq!(stats.mode, RuntimeMode::Full);
        assert_eq!(stats.num_cached_graphs, 0);
        assert_eq!(stats.configured_sizes, vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(stats.max_capture_size, 256);
    }

    #[test]
    fn test_warmup_manager_disabled() {
        let config = CudaGraphConfig::default();
        let manager = WarmupManager::new(&config);

        assert!(!manager.needs_warmup());

        let result = manager.run_warmup(|_| true);
        assert_eq!(result.graphs_captured, 0);
        assert!(result.warmed_sizes.is_empty());
    }

    #[test]
    fn test_warmup_manager_enabled() {
        let config = CudaGraphConfig::enabled();
        let manager = WarmupManager::new(&config);

        assert!(manager.needs_warmup());
        assert_eq!(manager.warmup_sizes(), &[1, 2, 4, 8, 16, 32]);

        // Simulate successful warmup
        let result = manager.run_warmup(|_size| true);
        assert_eq!(result.graphs_captured, 6);
        assert_eq!(result.graphs_failed, 0);
        assert_eq!(result.warmed_sizes, vec![1, 2, 4, 8, 16, 32]);
    }

    #[test]
    fn test_warmup_manager_partial_failure() {
        let config = CudaGraphConfig {
            enabled: true,
            mode: RuntimeMode::Full,
            capture_sizes: vec![1, 4, 8, 16],
            max_capture_size: 256,
            graph_pool_size: 256 * 1024 * 1024,
        };
        let manager = WarmupManager::new(&config);

        // Simulate warmup with failures for sizes > 8
        let result = manager.run_warmup(|size| size <= 8);
        assert_eq!(result.graphs_captured, 3); // 1, 4, 8
        assert_eq!(result.graphs_failed, 1); // 16
        assert_eq!(result.warmed_sizes, vec![1, 4, 8]);
    }

    #[test]
    fn test_warmup_result_skipped() {
        let result = WarmupResult::skipped();
        assert_eq!(result.graphs_captured, 0);
        assert_eq!(result.graphs_failed, 0);
        assert!(result.warmed_sizes.is_empty());
    }
}
