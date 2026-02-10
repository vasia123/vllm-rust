//! NCCL (NVIDIA Collective Communications Library) bindings.
//!
//! This module provides dynamic loading of NCCL for multi-GPU communication.
//! The library is loaded at runtime, allowing graceful fallback when NCCL
//! is not available.
//!
//! # Architecture
//!
//! Following vLLM's pattern:
//! - Dynamic loading via libloading (no static linking)
//! - Unique ID broadcast from rank 0 before communicator creation
//! - Stream-based async operations
//! - Graceful fallback for single-GPU or missing NCCL
//!
//! # Usage
//!
//! ```ignore
//! use vllm_core::distributed::nccl::{NcclLibrary, NcclCommunicator};
//!
//! // Load NCCL library
//! let nccl = NcclLibrary::new()?;
//!
//! // Create communicator (requires multi-GPU setup)
//! let comm = NcclCommunicator::new(&nccl, world_size, rank, device)?;
//! ```

use std::ffi::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::Arc;

use libloading::{Library, Symbol};

use super::error::{DistributedError, Result};

/// NCCL result code.
pub type NcclResult = c_int;

/// NCCL communicator handle.
pub type NcclComm = *mut c_void;

/// CUDA stream handle.
pub type CudaStream = *mut c_void;

/// NCCL unique ID for communicator initialization.
/// Must be broadcast from rank 0 to all other ranks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NcclUniqueId {
    internal: [c_char; 128],
}

impl Default for NcclUniqueId {
    fn default() -> Self {
        Self { internal: [0; 128] }
    }
}

/// NCCL data types.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

impl NcclDataType {
    /// Convert from candle DType.
    pub fn from_dtype(dtype: candle_core::DType) -> Option<Self> {
        match dtype {
            candle_core::DType::U8 => Some(Self::Uint8),
            candle_core::DType::U32 => Some(Self::Uint32),
            candle_core::DType::I64 => Some(Self::Int64),
            candle_core::DType::BF16 => Some(Self::Bfloat16),
            candle_core::DType::F16 => Some(Self::Float16),
            candle_core::DType::F32 => Some(Self::Float32),
            candle_core::DType::F64 => Some(Self::Float64),
        }
    }
}

/// NCCL reduction operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

impl From<super::communicator::ReduceOp> for NcclRedOp {
    fn from(op: super::communicator::ReduceOp) -> Self {
        match op {
            super::communicator::ReduceOp::Sum => Self::Sum,
            super::communicator::ReduceOp::Product => Self::Prod,
            super::communicator::ReduceOp::Max => Self::Max,
            super::communicator::ReduceOp::Min => Self::Min,
            super::communicator::ReduceOp::Average => Self::Avg,
        }
    }
}

/// NCCL result codes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum NcclResultCode {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
    NumResults = 8,
}

impl From<NcclResult> for NcclResultCode {
    fn from(code: NcclResult) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::UnhandledCudaError,
            2 => Self::SystemError,
            3 => Self::InternalError,
            4 => Self::InvalidArgument,
            5 => Self::InvalidUsage,
            6 => Self::RemoteError,
            7 => Self::InProgress,
            _ => Self::NumResults,
        }
    }
}

// ============================================================================
// CUDA Runtime Library Wrapper
// ============================================================================

/// CUDA runtime result code (cudaError_t).
pub type CudaError = c_int;

/// CUDA success code.
const CUDA_SUCCESS: CudaError = 0;

/// Environment variable for custom CUDA runtime library path.
/// Follows vLLM convention.
const VLLM_CUDART_SO_PATH_ENV: &str = "VLLM_CUDART_SO_PATH";

/// Type aliases for CUDA runtime function signatures.
type CudaSetDeviceFn = unsafe extern "C" fn(c_int) -> CudaError;
type CudaGetDeviceFn = unsafe extern "C" fn(*mut c_int) -> CudaError;
type CudaDeviceSynchronizeFn = unsafe extern "C" fn() -> CudaError;
type CudaMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaError;
type CudaFreeFn = unsafe extern "C" fn(*mut c_void) -> CudaError;
type CudaGetErrorStringFn = unsafe extern "C" fn(CudaError) -> *const c_char;

/// Dynamically loaded CUDA runtime library.
///
/// Provides access to essential CUDA runtime functions for device management.
/// Following vLLM's pattern, this wrapper enables proper device context setup
/// before NCCL initialization in multi-GPU environments.
///
/// # Loading Order
///
/// The library is loaded using the following fallback chain:
/// 1. `VLLM_CUDART_SO_PATH` environment variable (if set)
/// 2. `libcudart.so.12` (CUDA 12.x)
/// 3. `libcudart.so.11` (CUDA 11.x)
/// 4. `libcudart.so` (default symlink)
pub struct CudaRuntimeLibrary {
    #[allow(dead_code)]
    library: Library,
    set_device: CudaSetDeviceFn,
    get_device: CudaGetDeviceFn,
    device_synchronize: CudaDeviceSynchronizeFn,
    malloc: CudaMallocFn,
    free: CudaFreeFn,
    get_error_string: CudaGetErrorStringFn,
}

impl CudaRuntimeLibrary {
    /// Load CUDA runtime library.
    ///
    /// Tries multiple library names with fallback, following vLLM's pattern.
    pub fn new() -> std::result::Result<Self, String> {
        // Check environment variable first (vLLM compatibility)
        if let Ok(path) = std::env::var(VLLM_CUDART_SO_PATH_ENV) {
            if let Ok(lib) = Self::from_path(&path) {
                tracing::debug!(path = %path, "Loaded CUDA runtime from VLLM_CUDART_SO_PATH");
                return Ok(lib);
            }
            tracing::warn!(
                path = %path,
                "VLLM_CUDART_SO_PATH set but failed to load, trying default paths"
            );
        }

        // Try common library names
        let lib_names = ["libcudart.so.12", "libcudart.so.11", "libcudart.so"];

        for lib_name in &lib_names {
            match Self::from_path(lib_name) {
                Ok(lib) => {
                    tracing::debug!(library = %lib_name, "Loaded CUDA runtime");
                    return Ok(lib);
                }
                Err(_) => continue,
            }
        }

        Err(
            "Failed to load CUDA runtime library. Tried: VLLM_CUDART_SO_PATH env var, \
             libcudart.so.12, libcudart.so.11, libcudart.so"
                .to_string(),
        )
    }

    /// Load CUDA runtime from a specific path.
    fn from_path(path: &str) -> std::result::Result<Self, String> {
        let library =
            unsafe { Library::new(path) }.map_err(|e| format!("Failed to load {}: {}", path, e))?;

        // Load all required function pointers
        let (set_device, get_device, device_synchronize, malloc, free, get_error_string) = unsafe {
            let set_device: libloading::Symbol<CudaSetDeviceFn> = library
                .get(b"cudaSetDevice")
                .map_err(|e| format!("cudaSetDevice: {}", e))?;

            let get_device: libloading::Symbol<CudaGetDeviceFn> = library
                .get(b"cudaGetDevice")
                .map_err(|e| format!("cudaGetDevice: {}", e))?;

            let device_synchronize: libloading::Symbol<CudaDeviceSynchronizeFn> = library
                .get(b"cudaDeviceSynchronize")
                .map_err(|e| format!("cudaDeviceSynchronize: {}", e))?;

            let malloc: libloading::Symbol<CudaMallocFn> = library
                .get(b"cudaMalloc")
                .map_err(|e| format!("cudaMalloc: {}", e))?;

            let free: libloading::Symbol<CudaFreeFn> = library
                .get(b"cudaFree")
                .map_err(|e| format!("cudaFree: {}", e))?;

            let get_error_string: libloading::Symbol<CudaGetErrorStringFn> = library
                .get(b"cudaGetErrorString")
                .map_err(|e| format!("cudaGetErrorString: {}", e))?;

            (
                *set_device,
                *get_device,
                *device_synchronize,
                *malloc,
                *free,
                *get_error_string,
            )
        };

        Ok(Self {
            library,
            set_device,
            get_device,
            device_synchronize,
            malloc,
            free,
            get_error_string,
        })
    }

    /// Get error string for a CUDA error code.
    pub fn error_string(&self, error: CudaError) -> String {
        let ptr = unsafe { (self.get_error_string)(error) };
        if ptr.is_null() {
            return format!("Unknown CUDA error: {}", error);
        }
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Check CUDA result and convert to error if needed.
    fn check(&self, result: CudaError, operation: &str) -> std::result::Result<(), String> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!(
                "{} failed: {} ({})",
                operation,
                self.error_string(result),
                result
            ))
        }
    }

    /// Set the current CUDA device.
    ///
    /// This binds subsequent CUDA operations to the specified device.
    pub fn set_device(&self, device: usize) -> std::result::Result<(), String> {
        let result = unsafe { (self.set_device)(device as c_int) };
        self.check(result, &format!("cudaSetDevice({})", device))
    }

    /// Get the current CUDA device.
    pub fn get_device(&self) -> std::result::Result<usize, String> {
        let mut device: c_int = -1;
        let result = unsafe { (self.get_device)(&mut device) };
        self.check(result, "cudaGetDevice")?;
        Ok(device as usize)
    }

    /// Synchronize the current device.
    ///
    /// Blocks until all previously issued commands complete.
    pub fn device_synchronize(&self) -> std::result::Result<(), String> {
        let result = unsafe { (self.device_synchronize)() };
        self.check(result, "cudaDeviceSynchronize")
    }

    /// Allocate device memory.
    ///
    /// Returns a pointer to the allocated memory.
    pub fn malloc(&self, size: usize) -> std::result::Result<*mut c_void, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { (self.malloc)(&mut ptr, size) };
        self.check(result, &format!("cudaMalloc({})", size))?;
        Ok(ptr)
    }

    /// Free device memory.
    ///
    /// # Safety
    /// The pointer must have been allocated by `malloc` and not already freed.
    pub unsafe fn free(&self, ptr: *mut c_void) -> std::result::Result<(), String> {
        let result = (self.free)(ptr);
        self.check(result, "cudaFree")
    }

    /// Set device and force eager context initialization.
    ///
    /// Following vLLM's pattern, this allocates and frees a small buffer
    /// to force CUDA to eagerly initialize the device context. This prevents
    /// lazy initialization issues that can cause problems with NCCL.
    ///
    /// See: https://github.com/pytorch/pytorch/issues/155668
    pub fn set_device_eager(&self, device: usize) -> std::result::Result<(), String> {
        // Set the device
        self.set_device(device)?;

        // Force eager context initialization by allocating a small buffer
        // This is the same trick vLLM uses: torch.zeros(1, device=device)
        let ptr = self.malloc(64)?; // Allocate 64 bytes
        unsafe { self.free(ptr)? };

        // Synchronize to ensure context is fully initialized
        self.device_synchronize()?;

        tracing::trace!(device = device, "CUDA device context eagerly initialized");
        Ok(())
    }
}

// Safety: CudaRuntimeLibrary only contains function pointers and a Library handle,
// which are safe to share across threads.
unsafe impl Send for CudaRuntimeLibrary {}
unsafe impl Sync for CudaRuntimeLibrary {}

// ============================================================================
// NCCL Library Wrapper
// ============================================================================

/// Type aliases for NCCL function signatures.
type NcclGetVersionFn = unsafe extern "C" fn(*mut c_int) -> NcclResult;
type NcclGetUniqueIdFn = unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult;
type NcclCommInitRankFn =
    unsafe extern "C" fn(*mut NcclComm, c_int, NcclUniqueId, c_int) -> NcclResult;
type NcclCommDestroyFn = unsafe extern "C" fn(NcclComm) -> NcclResult;
type NcclGetErrorStringFn = unsafe extern "C" fn(NcclResult) -> *const c_char;

type NcclAllReduceFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclRedOp,
    NcclComm,
    CudaStream,
) -> NcclResult;

type NcclAllGatherFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclComm,
    CudaStream,
) -> NcclResult;

type NcclReduceScatterFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    NcclRedOp,
    NcclComm,
    CudaStream,
) -> NcclResult;

type NcclBroadcastFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    NcclDataType,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type NcclSendFn = unsafe extern "C" fn(
    *const c_void,
    usize,
    NcclDataType,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type NcclRecvFn = unsafe extern "C" fn(
    *mut c_void,
    usize,
    NcclDataType,
    c_int,
    NcclComm,
    CudaStream,
) -> NcclResult;

type NcclGroupStartFn = unsafe extern "C" fn() -> NcclResult;
type NcclGroupEndFn = unsafe extern "C" fn() -> NcclResult;

/// Dynamically loaded NCCL library.
///
/// Loads libnccl.so at runtime and provides access to NCCL functions.
/// Optionally loads CUDA runtime for proper device context management.
pub struct NcclLibrary {
    #[allow(dead_code)]
    library: Library,
    /// CUDA runtime for device management (optional but recommended for multi-GPU)
    cuda_runtime: Option<CudaRuntimeLibrary>,
    version: i32,

    // Function pointers
    get_unique_id: NcclGetUniqueIdFn,
    comm_init_rank: NcclCommInitRankFn,
    comm_destroy: NcclCommDestroyFn,
    get_error_string: NcclGetErrorStringFn,
    all_reduce: NcclAllReduceFn,
    all_gather: NcclAllGatherFn,
    reduce_scatter: NcclReduceScatterFn,
    broadcast: NcclBroadcastFn,
    send: NcclSendFn,
    recv: NcclRecvFn,
    group_start: NcclGroupStartFn,
    group_end: NcclGroupEndFn,
}

impl NcclLibrary {
    /// Load NCCL library from default location.
    ///
    /// Tries to load libnccl.so.2 from system paths.
    pub fn new() -> Result<Self> {
        Self::from_path("libnccl.so.2")
    }

    /// Load NCCL library from specified path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let library = unsafe { Library::new(path.as_ref()) }
            .map_err(|e| DistributedError::NcclError(format!("Failed to load NCCL: {}", e)))?;

        // Load all function pointers - extract raw pointers immediately
        // so we can move library into struct
        let (
            get_version_fn,
            get_unique_id_fn,
            comm_init_rank_fn,
            comm_destroy_fn,
            get_error_string_fn,
            all_reduce_fn,
            all_gather_fn,
            reduce_scatter_fn,
            broadcast_fn,
            send_fn,
            recv_fn,
            group_start_fn,
            group_end_fn,
        ) = unsafe {
            let get_version: Symbol<NcclGetVersionFn> = library
                .get(b"ncclGetVersion")
                .map_err(|e| DistributedError::NcclError(format!("ncclGetVersion: {}", e)))?;

            let get_unique_id: Symbol<NcclGetUniqueIdFn> = library
                .get(b"ncclGetUniqueId")
                .map_err(|e| DistributedError::NcclError(format!("ncclGetUniqueId: {}", e)))?;

            let comm_init_rank: Symbol<NcclCommInitRankFn> = library
                .get(b"ncclCommInitRank")
                .map_err(|e| DistributedError::NcclError(format!("ncclCommInitRank: {}", e)))?;

            let comm_destroy: Symbol<NcclCommDestroyFn> = library
                .get(b"ncclCommDestroy")
                .map_err(|e| DistributedError::NcclError(format!("ncclCommDestroy: {}", e)))?;

            let get_error_string: Symbol<NcclGetErrorStringFn> = library
                .get(b"ncclGetErrorString")
                .map_err(|e| DistributedError::NcclError(format!("ncclGetErrorString: {}", e)))?;

            let all_reduce: Symbol<NcclAllReduceFn> = library
                .get(b"ncclAllReduce")
                .map_err(|e| DistributedError::NcclError(format!("ncclAllReduce: {}", e)))?;

            let all_gather: Symbol<NcclAllGatherFn> = library
                .get(b"ncclAllGather")
                .map_err(|e| DistributedError::NcclError(format!("ncclAllGather: {}", e)))?;

            let reduce_scatter: Symbol<NcclReduceScatterFn> = library
                .get(b"ncclReduceScatter")
                .map_err(|e| DistributedError::NcclError(format!("ncclReduceScatter: {}", e)))?;

            let broadcast: Symbol<NcclBroadcastFn> = library
                .get(b"ncclBroadcast")
                .map_err(|e| DistributedError::NcclError(format!("ncclBroadcast: {}", e)))?;

            let send: Symbol<NcclSendFn> = library
                .get(b"ncclSend")
                .map_err(|e| DistributedError::NcclError(format!("ncclSend: {}", e)))?;

            let recv: Symbol<NcclRecvFn> = library
                .get(b"ncclRecv")
                .map_err(|e| DistributedError::NcclError(format!("ncclRecv: {}", e)))?;

            let group_start: Symbol<NcclGroupStartFn> = library
                .get(b"ncclGroupStart")
                .map_err(|e| DistributedError::NcclError(format!("ncclGroupStart: {}", e)))?;

            let group_end: Symbol<NcclGroupEndFn> = library
                .get(b"ncclGroupEnd")
                .map_err(|e| DistributedError::NcclError(format!("ncclGroupEnd: {}", e)))?;

            // Extract raw function pointers before symbols go out of scope
            (
                *get_version,
                *get_unique_id,
                *comm_init_rank,
                *comm_destroy,
                *get_error_string,
                *all_reduce,
                *all_gather,
                *reduce_scatter,
                *broadcast,
                *send,
                *recv,
                *group_start,
                *group_end,
            )
        };

        // Get version
        let mut version: c_int = 0;
        let result = unsafe { get_version_fn(&mut version) };
        if result != 0 {
            return Err(DistributedError::NcclError(
                "Failed to get NCCL version".to_string(),
            ));
        }

        // Load CUDA runtime for device management (optional but recommended)
        let cuda_runtime = match CudaRuntimeLibrary::new() {
            Ok(rt) => {
                tracing::info!("CUDA runtime loaded for device management");
                Some(rt)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "CUDA runtime not loaded - multi-GPU setups may have issues"
                );
                None
            }
        };

        Ok(Self {
            library,
            cuda_runtime,
            version,
            get_unique_id: get_unique_id_fn,
            comm_init_rank: comm_init_rank_fn,
            comm_destroy: comm_destroy_fn,
            get_error_string: get_error_string_fn,
            all_reduce: all_reduce_fn,
            all_gather: all_gather_fn,
            reduce_scatter: reduce_scatter_fn,
            broadcast: broadcast_fn,
            send: send_fn,
            recv: recv_fn,
            group_start: group_start_fn,
            group_end: group_end_fn,
        })
    }

    /// Get reference to CUDA runtime library (if available).
    pub fn cuda_runtime(&self) -> Option<&CudaRuntimeLibrary> {
        self.cuda_runtime.as_ref()
    }

    /// Set the CUDA device for the current thread with eager context initialization.
    ///
    /// This should be called before creating NCCL communicators in multi-GPU setups.
    /// Following vLLM's pattern, this forces eager device context initialization
    /// to prevent lazy initialization issues.
    ///
    /// # Arguments
    /// * `device` - CUDA device ordinal (0-indexed)
    ///
    /// # Returns
    /// * `Ok(())` - Device was set successfully (or CUDA runtime not available)
    /// * `Err(...)` - Failed to set device
    pub fn set_cuda_device(&self, device: usize) -> Result<()> {
        if let Some(ref cuda_rt) = self.cuda_runtime {
            cuda_rt.set_device_eager(device).map_err(|e| {
                DistributedError::NcclError(format!("Failed to set CUDA device {}: {}", device, e))
            })?;
            tracing::debug!(device = device, "CUDA device set with eager initialization");
            Ok(())
        } else {
            // CUDA runtime not available - this is acceptable for single-GPU setups
            // but may cause issues with multi-GPU configurations
            tracing::trace!(
                device = device,
                "CUDA runtime not available, skipping explicit device selection"
            );
            Ok(())
        }
    }

    /// Get NCCL version as (major, minor, patch).
    pub fn version(&self) -> (i32, i32, i32) {
        let major = self.version / 10000;
        let minor = (self.version % 10000) / 100;
        let patch = self.version % 100;
        (major, minor, patch)
    }

    /// Get raw version number.
    pub fn raw_version(&self) -> i32 {
        self.version
    }

    /// Generate a unique ID for communicator initialization.
    ///
    /// This should be called by rank 0 and broadcast to all other ranks.
    pub fn get_unique_id(&self) -> Result<NcclUniqueId> {
        let mut id = NcclUniqueId::default();
        let result = unsafe { (self.get_unique_id)(&mut id) };
        self.check_result(result)?;
        Ok(id)
    }

    /// Initialize a communicator.
    ///
    /// All ranks must call this with the same unique_id (broadcast from rank 0).
    pub fn comm_init_rank(
        &self,
        world_size: usize,
        unique_id: NcclUniqueId,
        rank: usize,
    ) -> Result<NcclComm> {
        let mut comm: NcclComm = std::ptr::null_mut();
        let result = unsafe {
            (self.comm_init_rank)(&mut comm, world_size as c_int, unique_id, rank as c_int)
        };
        self.check_result(result)?;
        Ok(comm)
    }

    /// Destroy a communicator.
    ///
    /// # Safety
    /// The `comm` handle must be valid and not already destroyed.
    pub unsafe fn comm_destroy(&self, comm: NcclComm) -> Result<()> {
        let result = (self.comm_destroy)(comm);
        self.check_result(result)
    }

    /// Get error string for a result code.
    pub fn get_error_string(&self, result: NcclResult) -> String {
        let ptr = unsafe { (self.get_error_string)(result) };
        if ptr.is_null() {
            return format!("Unknown NCCL error: {}", result);
        }
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Check NCCL result and convert to error if needed.
    fn check_result(&self, result: NcclResult) -> Result<()> {
        if result == 0 {
            Ok(())
        } else {
            Err(DistributedError::NcclError(self.get_error_string(result)))
        }
    }

    // Collective operations

    /// All-reduce operation.
    ///
    /// # Safety
    /// - `send_buf` and `recv_buf` must point to valid GPU memory
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn all_reduce(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: usize,
        dtype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let result = (self.all_reduce)(send_buf, recv_buf, count, dtype, op, comm, stream);
        self.check_result(result)
    }

    /// All-gather operation.
    ///
    /// # Safety
    /// - `send_buf` and `recv_buf` must point to valid GPU memory
    /// - `recv_buf` must have space for `count * world_size` elements
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn all_gather(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: usize,
        dtype: NcclDataType,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let result = (self.all_gather)(send_buf, recv_buf, count, dtype, comm, stream);
        self.check_result(result)
    }

    /// Reduce-scatter operation.
    ///
    /// # Safety
    /// - `send_buf` and `recv_buf` must point to valid GPU memory
    /// - `send_buf` must have `count * world_size` elements
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn reduce_scatter(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: usize,
        dtype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let result = (self.reduce_scatter)(send_buf, recv_buf, count, dtype, op, comm, stream);
        self.check_result(result)
    }

    /// Broadcast operation.
    ///
    /// # Safety
    /// - `send_buf` (on root) and `recv_buf` (all ranks) must point to valid GPU memory
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn broadcast(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: usize,
        dtype: NcclDataType,
        root: usize,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let result = (self.broadcast)(
            send_buf,
            recv_buf,
            count,
            dtype,
            root as c_int,
            comm,
            stream,
        );
        self.check_result(result)
    }

    /// Point-to-point send.
    ///
    /// # Safety
    /// - `buf` must point to valid GPU memory with `count` elements
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    pub unsafe fn send(
        &self,
        buf: *const c_void,
        count: usize,
        dtype: NcclDataType,
        dest: usize,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let result = (self.send)(buf, count, dtype, dest as c_int, comm, stream);
        self.check_result(result)
    }

    /// Point-to-point receive.
    ///
    /// # Safety
    /// - `buf` must point to valid GPU memory with space for `count` elements
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    pub unsafe fn recv(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: NcclDataType,
        src: usize,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let result = (self.recv)(buf, count, dtype, src as c_int, comm, stream);
        self.check_result(result)
    }

    /// Start a group of operations.
    pub fn group_start(&self) -> Result<()> {
        let result = unsafe { (self.group_start)() };
        self.check_result(result)
    }

    /// End a group of operations.
    pub fn group_end(&self) -> Result<()> {
        let result = unsafe { (self.group_end)() };
        self.check_result(result)
    }

    /// All-to-all operation using grouped send/recv.
    ///
    /// Each rank sends `count_per_rank` elements to every other rank.
    /// Uses NCCL's send/recv within a group for efficient implementation.
    ///
    /// # Safety
    /// - `send_buf` must point to valid GPU memory with `count_per_rank * world_size` elements
    /// - `recv_buf` must point to valid GPU memory with space for `count_per_rank * world_size` elements
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn all_to_all(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count_per_rank: usize,
        dtype: NcclDataType,
        world_size: usize,
        rank: usize,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        let dtype_size = Self::dtype_size(dtype);
        let chunk_bytes = count_per_rank * dtype_size;

        self.group_start()?;

        for peer in 0..world_size {
            let send_offset = peer * chunk_bytes;
            let recv_offset = peer * chunk_bytes;

            if peer != rank {
                // Send to peer
                (self.send)(
                    (send_buf as *const u8).add(send_offset) as *const c_void,
                    count_per_rank,
                    dtype,
                    peer as c_int,
                    comm,
                    stream,
                );
                // Receive from peer
                (self.recv)(
                    (recv_buf as *mut u8).add(recv_offset) as *mut c_void,
                    count_per_rank,
                    dtype,
                    peer as c_int,
                    comm,
                    stream,
                );
            } else {
                // Local copy for self-send
                std::ptr::copy_nonoverlapping(
                    (send_buf as *const u8).add(send_offset),
                    (recv_buf as *mut u8).add(recv_offset),
                    chunk_bytes,
                );
            }
        }

        self.group_end()
    }

    /// Variable-size all-to-all operation using grouped send/recv.
    ///
    /// Each rank sends different amounts to each other rank.
    ///
    /// # Safety
    /// - `send_buf` must point to valid GPU memory with sum(send_counts) elements
    /// - `recv_buf` must point to valid GPU memory with space for sum(recv_counts) elements
    /// - `send_counts` and `recv_counts` must have length == world_size
    /// - `comm` must be a valid NCCL communicator
    /// - `stream` must be a valid CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn all_to_all_v(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        send_counts: &[usize],
        recv_counts: &[usize],
        dtype: NcclDataType,
        world_size: usize,
        rank: usize,
        comm: NcclComm,
        stream: CudaStream,
    ) -> Result<()> {
        debug_assert_eq!(send_counts.len(), world_size);
        debug_assert_eq!(recv_counts.len(), world_size);

        let dtype_size = Self::dtype_size(dtype);

        // Compute offsets for send and recv buffers
        let mut send_offsets = Vec::with_capacity(world_size);
        let mut recv_offsets = Vec::with_capacity(world_size);
        let mut send_offset = 0usize;
        let mut recv_offset = 0usize;

        for i in 0..world_size {
            send_offsets.push(send_offset);
            recv_offsets.push(recv_offset);
            send_offset += send_counts[i] * dtype_size;
            recv_offset += recv_counts[i] * dtype_size;
        }

        self.group_start()?;

        for peer in 0..world_size {
            if peer != rank {
                if send_counts[peer] > 0 {
                    (self.send)(
                        (send_buf as *const u8).add(send_offsets[peer]) as *const c_void,
                        send_counts[peer],
                        dtype,
                        peer as c_int,
                        comm,
                        stream,
                    );
                }
                if recv_counts[peer] > 0 {
                    (self.recv)(
                        (recv_buf as *mut u8).add(recv_offsets[peer]) as *mut c_void,
                        recv_counts[peer],
                        dtype,
                        peer as c_int,
                        comm,
                        stream,
                    );
                }
            } else {
                // Local copy for self-send
                if send_counts[peer] > 0 {
                    let copy_bytes = send_counts[peer] * dtype_size;
                    std::ptr::copy_nonoverlapping(
                        (send_buf as *const u8).add(send_offsets[peer]),
                        (recv_buf as *mut u8).add(recv_offsets[peer]),
                        copy_bytes,
                    );
                }
            }
        }

        self.group_end()
    }

    /// Get the size in bytes for a given NCCL data type.
    fn dtype_size(dtype: NcclDataType) -> usize {
        match dtype {
            NcclDataType::Int8 | NcclDataType::Uint8 => 1,
            NcclDataType::Float16 | NcclDataType::Bfloat16 => 2,
            NcclDataType::Int32 | NcclDataType::Uint32 | NcclDataType::Float32 => 4,
            NcclDataType::Int64 | NcclDataType::Uint64 | NcclDataType::Float64 => 8,
        }
    }
}

// Safety: NcclLibrary only contains function pointers and a Library handle,
// which are safe to share across threads.
unsafe impl Send for NcclLibrary {}
unsafe impl Sync for NcclLibrary {}

/// NCCL-based communicator for multi-GPU operations.
///
/// Implements the DeviceCommunicator trait using NCCL.
pub struct NcclCommunicator {
    /// Shared reference to NCCL library.
    nccl: Arc<NcclLibrary>,
    /// NCCL communicator handle.
    comm: NcclComm,
    /// This process's rank.
    rank: usize,
    /// Total number of processes.
    world_size: usize,
    /// CUDA device ordinal.
    #[allow(dead_code)]
    device: usize,
}

impl NcclCommunicator {
    /// Create a new NCCL communicator.
    ///
    /// This follows vLLM's initialization pattern:
    /// 1. Set CUDA device with eager context initialization
    /// 2. Create NCCL communicator
    /// 3. Synchronize to ensure initialization is complete
    ///
    /// # Arguments
    /// * `nccl` - Shared NCCL library instance
    /// * `unique_id` - Unique ID from rank 0 (must be broadcast first)
    /// * `world_size` - Total number of processes
    /// * `rank` - This process's rank
    /// * `device` - CUDA device ordinal for this rank
    ///
    /// # Warmup
    ///
    /// For production use, it's recommended to perform a warmup all_reduce
    /// after creating the communicator to ensure all ranks are synchronized
    /// and NCCL internal buffers are allocated. Example:
    ///
    /// ```ignore
    /// let comm = NcclCommunicator::new(nccl, unique_id, world_size, rank, device)?;
    /// // Perform warmup all_reduce with a small buffer
    /// unsafe { comm.all_reduce_inplace(warmup_buffer, 1, dtype, ReduceOp::Sum, stream)?; }
    /// stream.synchronize()?;
    /// ```
    pub fn new(
        nccl: Arc<NcclLibrary>,
        unique_id: NcclUniqueId,
        world_size: usize,
        rank: usize,
        device: usize,
    ) -> Result<Self> {
        if rank >= world_size {
            return Err(DistributedError::InvalidRank { rank, world_size });
        }

        // Step 1: Set CUDA device with eager context initialization
        // This ensures ncclCommInitRank uses the correct GPU context.
        // Following vLLM's pattern, we force eager initialization to prevent
        // lazy initialization issues.
        nccl.set_cuda_device(device)?;

        // Step 2: Create NCCL communicator
        tracing::debug!(
            rank = rank,
            world_size = world_size,
            device = device,
            "Initializing NCCL communicator"
        );

        let comm = nccl.comm_init_rank(world_size, unique_id, rank)?;

        // Step 3: Synchronize device to ensure NCCL initialization is complete
        // This is important for multi-GPU setups where NCCL may allocate
        // internal buffers asynchronously
        if let Some(ref cuda_rt) = nccl.cuda_runtime {
            if let Err(e) = cuda_rt.device_synchronize() {
                tracing::warn!(
                    error = %e,
                    "Failed to synchronize after NCCL init (non-fatal)"
                );
            }
        }

        tracing::info!(
            rank = rank,
            world_size = world_size,
            device = device,
            "NCCL communicator initialized successfully"
        );

        Ok(Self {
            nccl,
            comm,
            rank,
            world_size,
            device,
        })
    }

    /// Get the raw NCCL communicator handle.
    ///
    /// # Safety
    /// The returned handle is only valid while this `NcclCommunicator` exists.
    /// Do not store or use after the communicator is dropped.
    pub fn raw_comm(&self) -> NcclComm {
        self.comm
    }

    /// Get the device ordinal this communicator is bound to.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Get the NCCL library reference.
    pub fn nccl(&self) -> &NcclLibrary {
        &self.nccl
    }

    /// Get rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        // Best effort cleanup - ignore errors
        // Safety: self.comm is valid because we created it in new()
        let _ = unsafe { self.nccl.comm_destroy(self.comm) };
    }
}

// Safety: NcclCommunicator manages its own comm handle lifecycle
// and NCCL handles are thread-safe when used correctly.
unsafe impl Send for NcclCommunicator {}
unsafe impl Sync for NcclCommunicator {}

// ─── DeviceCommunicator Implementation ─────────────────────────────────────────

use super::communicator::{DeviceCommunicator, ReduceOp};
use super::process_group::ProcessGroup;
use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::{cuda::CudaStorageSlice, DType, Device, Storage, Tensor};

/// NCCL-based device communicator with process group.
///
/// This wraps NcclCommunicator to implement the DeviceCommunicator trait,
/// providing high-level tensor operations for multi-GPU communication.
pub struct NcclDeviceCommunicator<P: ProcessGroup> {
    /// The underlying NCCL communicator.
    nccl_comm: NcclCommunicator,
    /// Process group for rank/size information.
    process_group: P,
}

impl<P: ProcessGroup> NcclDeviceCommunicator<P> {
    /// Create a new NCCL device communicator.
    pub fn new(nccl_comm: NcclCommunicator, process_group: P) -> Self {
        Self {
            nccl_comm,
            process_group,
        }
    }

    /// Get the underlying NCCL communicator.
    pub fn nccl_comm(&self) -> &NcclCommunicator {
        &self.nccl_comm
    }

    /// Get raw device pointer from a CUDA tensor.
    ///
    /// Returns (pointer, element_count, dtype).
    /// Synchronizes the tensor's stream before returning to ensure data is ready.
    fn get_cuda_ptr(tensor: &Tensor) -> Result<(*const c_void, usize, NcclDataType)> {
        let (storage, layout) = tensor.storage_and_layout();

        // Verify tensor is contiguous
        if !layout.is_contiguous() {
            return Err(DistributedError::InvalidTensor(
                "NCCL operations require contiguous tensors".to_string(),
            ));
        }

        let cuda_storage = match &*storage {
            Storage::Cuda(cs) => cs,
            _ => {
                return Err(DistributedError::InvalidTensor(
                    "NCCL operations require CUDA tensors".to_string(),
                ))
            }
        };

        let elem_count = layout.shape().elem_count();
        let nccl_dtype = NcclDataType::from_dtype(tensor.dtype()).ok_or_else(|| {
            DistributedError::InvalidTensor(format!("Unsupported dtype: {:?}", tensor.dtype()))
        })?;

        // Get raw pointer based on dtype (with sync)
        let ptr = Self::get_slice_ptr(&cuda_storage.slice, layout.start_offset())?;

        Ok((ptr, elem_count, nccl_dtype))
    }

    /// Get raw pointer from CudaStorageSlice.
    ///
    /// Synchronizes the stream and returns the device pointer.
    fn get_slice_ptr(slice: &CudaStorageSlice, offset: usize) -> Result<*const c_void> {
        // Helper macro to get ptr from slice
        // Uses device_ptr which borrows the stream, so we extract ptr and drop guard
        macro_rules! get_ptr {
            ($s:expr, $offset:expr) => {{
                let view = $s.slice($offset..);
                let stream = $s.stream();
                let (ptr, _guard) = view.device_ptr(stream);
                // The _guard will be dropped here, causing sync
                ptr as *const c_void
            }};
        }

        let ptr = match slice {
            CudaStorageSlice::U8(s) => get_ptr!(s, offset),
            CudaStorageSlice::U32(s) => get_ptr!(s, offset),
            CudaStorageSlice::I64(s) => get_ptr!(s, offset),
            CudaStorageSlice::BF16(s) => get_ptr!(s, offset),
            CudaStorageSlice::F16(s) => get_ptr!(s, offset),
            CudaStorageSlice::F32(s) => get_ptr!(s, offset),
            CudaStorageSlice::F64(s) => get_ptr!(s, offset),
        };
        Ok(ptr)
    }

    /// Allocate output tensor with same dtype and device as input.
    fn alloc_output(input: &Tensor, shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor::zeros(shape, input.dtype(), input.device())?)
    }

    /// Get mutable device pointer from a tensor.
    fn get_cuda_ptr_mut(tensor: &Tensor) -> Result<(*mut c_void, usize, NcclDataType)> {
        let (ptr, count, dtype) = Self::get_cuda_ptr(tensor)?;
        Ok((ptr as *mut c_void, count, dtype))
    }

    /// Get the default CUDA stream (stream 0).
    ///
    /// For production use, consider using per-device streams.
    fn default_stream() -> CudaStream {
        std::ptr::null_mut()
    }

    /// Assert that a tensor is on the correct CUDA device.
    ///
    /// vLLM pattern: verify tensor device matches communicator device
    /// on every operation to prevent silent data corruption.
    fn assert_device(&self, tensor: &Tensor) -> Result<()> {
        use candle_core::DeviceLocation;

        match tensor.device().location() {
            DeviceLocation::Cuda { gpu_id } if gpu_id == self.nccl_comm.device => Ok(()),
            DeviceLocation::Cuda { gpu_id } => Err(DistributedError::DeviceMismatch {
                expected: format!("cuda:{}", self.nccl_comm.device),
                actual: format!("cuda:{}", gpu_id),
            }),
            location => Err(DistributedError::DeviceMismatch {
                expected: format!("cuda:{}", self.nccl_comm.device),
                actual: format!("{:?}", location),
            }),
        }
    }

    /// Perform warmup all_reduce to initialize NCCL internal buffers.
    ///
    /// vLLM pattern: warmup ensures all ranks are synchronized
    /// and NCCL allocates its internal buffers before real work.
    /// Should be called after communicator creation.
    pub fn warmup(&self) -> Result<()> {
        if self.process_group.is_single() {
            return Ok(());
        }

        let device = Device::cuda_if_available(self.nccl_comm.device)?;
        let dummy = Tensor::zeros(&[1], DType::F32, &device)?;
        let _ = self.all_reduce(&dummy, ReduceOp::Sum)?;
        self.barrier()?;

        tracing::debug!(
            rank = self.process_group.rank(),
            world_size = self.process_group.world_size(),
            "NCCL warmup complete"
        );

        Ok(())
    }
}

impl<P: ProcessGroup + Send + Sync> DeviceCommunicator for NcclDeviceCommunicator<P> {
    fn process_group(&self) -> &dyn ProcessGroup {
        &self.process_group
    }

    fn all_reduce(&self, tensor: &Tensor, op: ReduceOp) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }

        self.assert_device(tensor)?;
        let (send_ptr, count, dtype) = Self::get_cuda_ptr(tensor)?;
        let output = Self::alloc_output(tensor, tensor.dims())?;
        let (recv_ptr, _, _) = Self::get_cuda_ptr_mut(&output)?;

        let nccl_op = NcclRedOp::from(op);
        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.all_reduce(
                send_ptr,
                recv_ptr,
                count,
                dtype,
                nccl_op,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }

    fn all_gather(&self, tensor: &Tensor, gather_dim: usize) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }

        self.assert_device(tensor)?;
        let world_size = self.process_group.world_size();
        let (send_ptr, count, dtype) = Self::get_cuda_ptr(tensor)?;

        // Output shape: dim[gather_dim] *= world_size
        let mut output_shape = tensor.dims().to_vec();
        output_shape[gather_dim] *= world_size;
        let output = Self::alloc_output(tensor, &output_shape)?;
        let (recv_ptr, _, _) = Self::get_cuda_ptr_mut(&output)?;

        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.all_gather(
                send_ptr,
                recv_ptr,
                count,
                dtype,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }

    fn reduce_scatter(&self, tensor: &Tensor, scatter_dim: usize, op: ReduceOp) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }

        self.assert_device(tensor)?;
        let world_size = self.process_group.world_size();
        let dims = tensor.dims();

        if !dims[scatter_dim].is_multiple_of(world_size) {
            return Err(DistributedError::ShapeMismatch {
                expected: vec![dims[scatter_dim] / world_size],
                actual: vec![dims[scatter_dim]],
            });
        }

        let (send_ptr, _, dtype) = Self::get_cuda_ptr(tensor)?;

        // Output shape: dim[scatter_dim] /= world_size
        let mut output_shape = dims.to_vec();
        output_shape[scatter_dim] /= world_size;
        let output = Self::alloc_output(tensor, &output_shape)?;
        let (recv_ptr, recv_count, _) = Self::get_cuda_ptr_mut(&output)?;

        let nccl_op = NcclRedOp::from(op);
        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.reduce_scatter(
                send_ptr,
                recv_ptr,
                recv_count,
                dtype,
                nccl_op,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }

    fn broadcast(&self, tensor: &Tensor, src_rank: usize) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }

        self.assert_device(tensor)?;
        let (send_ptr, count, dtype) = Self::get_cuda_ptr(tensor)?;
        let output = Self::alloc_output(tensor, tensor.dims())?;
        let (recv_ptr, _, _) = Self::get_cuda_ptr_mut(&output)?;

        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.broadcast(
                send_ptr,
                recv_ptr,
                count,
                dtype,
                src_rank,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }

    fn send(&self, tensor: &Tensor, dst_rank: usize) -> Result<()> {
        if self.process_group.is_single() {
            return Ok(());
        }

        self.assert_device(tensor)?;
        let (send_ptr, count, dtype) = Self::get_cuda_ptr(tensor)?;
        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.send(
                send_ptr,
                count,
                dtype,
                dst_rank,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(())
    }

    fn recv(&self, shape: &[usize], dtype: candle_core::DType, src_rank: usize) -> Result<Tensor> {
        if self.process_group.is_single() {
            let device = Device::cuda_if_available(self.nccl_comm.device)?;
            return Ok(Tensor::zeros(shape, dtype, &device)?);
        }

        let device = Device::cuda_if_available(self.nccl_comm.device)?;
        let output = Tensor::zeros(shape, dtype, &device)?;
        let (recv_ptr, count, nccl_dtype) = Self::get_cuda_ptr_mut(&output)?;
        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.recv(
                recv_ptr,
                count,
                nccl_dtype,
                src_rank,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }

    fn barrier(&self) -> Result<()> {
        if self.process_group.is_single() {
            return Ok(());
        }

        // NCCL doesn't have explicit barrier, use small all_reduce as sync point
        let device = Device::cuda_if_available(self.nccl_comm.device)?;
        let sync_tensor = Tensor::zeros(&[1], DType::F32, &device)?;
        let _ = self.all_reduce(&sync_tensor, ReduceOp::Sum)?;

        // Synchronize to ensure completion
        if let Some(ref cuda_rt) = self.nccl_comm.nccl.cuda_runtime {
            cuda_rt
                .device_synchronize()
                .map_err(DistributedError::NcclError)?;
        }

        Ok(())
    }

    fn all_to_all(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }

        self.assert_device(tensor)?;
        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();
        let dims = tensor.dims();

        if dims.is_empty() || !dims[0].is_multiple_of(world_size) {
            return Err(DistributedError::ShapeMismatch {
                expected: vec![world_size],
                actual: dims.to_vec(),
            });
        }

        let count_per_rank = dims.iter().product::<usize>() / world_size;
        let (send_ptr, _, dtype) = Self::get_cuda_ptr(tensor)?;

        // Output has same shape as input
        let output = Self::alloc_output(tensor, dims)?;
        let (recv_ptr, _, _) = Self::get_cuda_ptr_mut(&output)?;

        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.all_to_all(
                send_ptr,
                recv_ptr,
                count_per_rank,
                dtype,
                world_size,
                rank,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }

    fn all_to_all_v(
        &self,
        tensor: &Tensor,
        send_splits: &[usize],
        recv_splits: &[usize],
    ) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }

        self.assert_device(tensor)?;
        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();

        if send_splits.len() != world_size || recv_splits.len() != world_size {
            return Err(DistributedError::ShapeMismatch {
                expected: vec![world_size],
                actual: vec![send_splits.len()],
            });
        }

        let (send_ptr, _, dtype) = Self::get_cuda_ptr(tensor)?;

        // Output shape: first dim is sum of recv_splits
        let total_recv: usize = recv_splits.iter().sum();
        let mut output_shape = tensor.dims().to_vec();
        output_shape[0] = total_recv;
        let output = Self::alloc_output(tensor, &output_shape)?;
        let (recv_ptr, _, _) = Self::get_cuda_ptr_mut(&output)?;

        let stream = Self::default_stream();

        unsafe {
            self.nccl_comm.nccl.all_to_all_v(
                send_ptr,
                recv_ptr,
                send_splits,
                recv_splits,
                dtype,
                world_size,
                rank,
                self.nccl_comm.comm,
                stream,
            )?;
        }

        Ok(output)
    }
}

/// Check if NCCL is available on this system.
pub fn is_nccl_available() -> bool {
    NcclLibrary::new().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_data_type_from_dtype() {
        assert_eq!(
            NcclDataType::from_dtype(candle_core::DType::F32),
            Some(NcclDataType::Float32)
        );
        assert_eq!(
            NcclDataType::from_dtype(candle_core::DType::F16),
            Some(NcclDataType::Float16)
        );
        assert_eq!(
            NcclDataType::from_dtype(candle_core::DType::BF16),
            Some(NcclDataType::Bfloat16)
        );
    }

    #[test]
    fn nccl_red_op_from_reduce_op() {
        use super::super::communicator::ReduceOp;

        assert_eq!(NcclRedOp::from(ReduceOp::Sum), NcclRedOp::Sum);
        assert_eq!(NcclRedOp::from(ReduceOp::Product), NcclRedOp::Prod);
        assert_eq!(NcclRedOp::from(ReduceOp::Max), NcclRedOp::Max);
        assert_eq!(NcclRedOp::from(ReduceOp::Min), NcclRedOp::Min);
        assert_eq!(NcclRedOp::from(ReduceOp::Average), NcclRedOp::Avg);
    }

    #[test]
    fn nccl_result_code_conversion() {
        assert_eq!(NcclResultCode::from(0), NcclResultCode::Success);
        assert_eq!(NcclResultCode::from(1), NcclResultCode::UnhandledCudaError);
        assert_eq!(NcclResultCode::from(4), NcclResultCode::InvalidArgument);
    }

    #[test]
    fn nccl_unique_id_default() {
        let id = NcclUniqueId::default();
        assert!(id.internal.iter().all(|&b| b == 0));
    }

    #[test]
    #[ignore = "Requires NCCL library to be installed"]
    fn nccl_library_load() {
        let nccl = NcclLibrary::new().expect("Failed to load NCCL");
        let (major, minor, patch) = nccl.version();
        println!("NCCL version: {}.{}.{}", major, minor, patch);
        assert!(major >= 2, "NCCL version should be 2.x or higher");
    }

    #[test]
    #[ignore = "Requires NCCL library to be installed"]
    fn nccl_get_unique_id() {
        let nccl = NcclLibrary::new().expect("Failed to load NCCL");
        let id = nccl.get_unique_id().expect("Failed to get unique ID");
        // Unique ID should have some non-zero bytes
        assert!(id.internal.iter().any(|&b| b != 0));
    }

    #[test]
    #[ignore = "Requires multi-GPU setup"]
    fn nccl_comm_init_single_rank() {
        let nccl = Arc::new(NcclLibrary::new().expect("Failed to load NCCL"));
        let id = nccl.get_unique_id().expect("Failed to get unique ID");

        // This will likely fail without proper CUDA context
        // but tests the API surface
        let _comm = NcclCommunicator::new(nccl, id, 1, 0, 0);
    }

    #[test]
    fn is_nccl_available_check() {
        // This should not panic regardless of NCCL availability
        let available = is_nccl_available();
        println!("NCCL available: {}", available);
    }

    // CUDA Runtime Library tests

    #[test]
    fn cuda_runtime_env_var_name() {
        // Verify the env var name matches vLLM convention
        assert_eq!(VLLM_CUDART_SO_PATH_ENV, "VLLM_CUDART_SO_PATH");
    }

    #[test]
    fn cuda_runtime_load_check() {
        // This should not panic regardless of CUDA availability
        let result = CudaRuntimeLibrary::new();
        match result {
            Ok(_) => println!("CUDA runtime loaded successfully"),
            Err(e) => println!("CUDA runtime not available: {}", e),
        }
    }

    #[test]
    #[ignore = "Requires CUDA runtime to be installed"]
    fn cuda_runtime_set_device() {
        let cuda_rt = CudaRuntimeLibrary::new().expect("Failed to load CUDA runtime");

        // Get current device
        let device = cuda_rt.get_device().expect("Failed to get device");
        println!("Current CUDA device: {}", device);

        // Set device 0 (should always exist if CUDA is available)
        cuda_rt.set_device(0).expect("Failed to set device 0");
    }

    #[test]
    #[ignore = "Requires CUDA runtime to be installed"]
    fn cuda_runtime_set_device_eager() {
        let cuda_rt = CudaRuntimeLibrary::new().expect("Failed to load CUDA runtime");

        // This tests the full eager initialization path
        cuda_rt
            .set_device_eager(0)
            .expect("Failed to set device with eager init");

        // Verify device is set
        let device = cuda_rt.get_device().expect("Failed to get device");
        assert_eq!(device, 0);
    }

    #[test]
    #[ignore = "Requires CUDA runtime to be installed"]
    fn cuda_runtime_malloc_free() {
        let cuda_rt = CudaRuntimeLibrary::new().expect("Failed to load CUDA runtime");
        cuda_rt.set_device(0).expect("Failed to set device");

        // Allocate some memory
        let ptr = cuda_rt.malloc(1024).expect("Failed to allocate memory");
        assert!(!ptr.is_null());

        // Free it
        unsafe {
            cuda_rt.free(ptr).expect("Failed to free memory");
        }
    }

    #[test]
    #[ignore = "Requires CUDA runtime to be installed"]
    fn cuda_runtime_error_string() {
        let cuda_rt = CudaRuntimeLibrary::new().expect("Failed to load CUDA runtime");

        // Error code 0 should be "success" or similar
        let success_str = cuda_rt.error_string(0);
        println!("CUDA success string: {}", success_str);

        // Error code 1 is typically "invalid value"
        let error_str = cuda_rt.error_string(1);
        println!("CUDA error 1 string: {}", error_str);
    }

    #[test]
    #[ignore = "Requires NCCL and CUDA to be installed"]
    fn nccl_with_cuda_runtime() {
        let nccl = NcclLibrary::new().expect("Failed to load NCCL");

        // Check that CUDA runtime was loaded
        assert!(
            nccl.cuda_runtime().is_some(),
            "CUDA runtime should be loaded with NCCL"
        );

        // Test set_cuda_device through NcclLibrary
        nccl.set_cuda_device(0)
            .expect("Failed to set CUDA device via NCCL");
    }

    // NcclDeviceCommunicator tests

    #[test]
    fn device_mismatch_error_display() {
        let err = DistributedError::DeviceMismatch {
            expected: "cuda:0".to_string(),
            actual: "cuda:1".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("cuda:0"));
        assert!(msg.contains("cuda:1"));
        assert!(msg.contains("mismatch"));
    }

    #[test]
    fn device_mismatch_error_cpu_tensor() {
        let err = DistributedError::DeviceMismatch {
            expected: "cuda:0".to_string(),
            actual: "Cpu".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("cuda:0"));
        assert!(msg.contains("Cpu"));
    }

    #[test]
    #[ignore = "Requires CUDA"]
    fn nccl_device_communicator_single_rank_passthrough() {
        use super::super::process_group::LocalProcessGroup;

        // For single rank, operations should just clone the tensor
        let nccl = Arc::new(NcclLibrary::new().expect("Failed to load NCCL"));
        let id = nccl.get_unique_id().expect("Failed to get unique ID");
        let comm = NcclCommunicator::new(nccl, id, 1, 0, 0).expect("Failed to create communicator");
        let pg = LocalProcessGroup::with_rank(0, 1);
        let device_comm = NcclDeviceCommunicator::new(comm, pg);

        let device = Device::cuda_if_available(0).expect("Need CUDA");
        let input = Tensor::zeros(&[4, 4], DType::F32, &device).expect("Failed to create tensor");

        // Single-rank all_reduce should return clone
        let output = device_comm
            .all_reduce(&input, ReduceOp::Sum)
            .expect("all_reduce failed");
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    #[ignore = "Requires multi-GPU setup"]
    fn nccl_device_communicator_warmup() {
        use super::super::process_group::LocalProcessGroup;

        let nccl = Arc::new(NcclLibrary::new().expect("Failed to load NCCL"));
        let id = nccl.get_unique_id().expect("Failed to get unique ID");
        let comm = NcclCommunicator::new(nccl, id, 1, 0, 0).expect("Failed to create communicator");
        let pg = LocalProcessGroup::with_rank(0, 1);
        let device_comm = NcclDeviceCommunicator::new(comm, pg);

        // Warmup should complete without error
        device_comm.warmup().expect("warmup failed");
    }

    #[test]
    #[ignore = "Requires multi-GPU setup"]
    fn nccl_device_communicator_all_reduce_shape_preserved() {
        use super::super::process_group::LocalProcessGroup;

        let nccl = Arc::new(NcclLibrary::new().expect("Failed to load NCCL"));
        let id = nccl.get_unique_id().expect("Failed to get unique ID");
        let comm = NcclCommunicator::new(nccl, id, 2, 0, 0).expect("Failed to create communicator");
        let pg = LocalProcessGroup::with_rank(0, 2);
        let device_comm = NcclDeviceCommunicator::new(comm, pg);

        let device = Device::cuda_if_available(0).expect("Need CUDA");
        let input = Tensor::zeros(&[8, 16], DType::F32, &device).expect("Failed to create tensor");

        let output = device_comm
            .all_reduce(&input, ReduceOp::Sum)
            .expect("all_reduce failed");
        assert_eq!(output.dims(), &[8, 16]);
        assert_eq!(output.dtype(), DType::F32);
    }

    #[test]
    #[ignore = "Requires multi-GPU setup"]
    fn nccl_device_communicator_all_gather_output_shape() {
        use super::super::process_group::LocalProcessGroup;

        let nccl = Arc::new(NcclLibrary::new().expect("Failed to load NCCL"));
        let id = nccl.get_unique_id().expect("Failed to get unique ID");
        let comm = NcclCommunicator::new(nccl, id, 2, 0, 0).expect("Failed to create communicator");
        let pg = LocalProcessGroup::with_rank(0, 2);
        let device_comm = NcclDeviceCommunicator::new(comm, pg);

        let device = Device::cuda_if_available(0).expect("Need CUDA");
        let input = Tensor::zeros(&[4, 8], DType::F32, &device).expect("Failed to create tensor");

        // all_gather on dim 0 should double first dimension
        let output = device_comm
            .all_gather(&input, 0)
            .expect("all_gather failed");
        assert_eq!(output.dims(), &[8, 8]); // 4 * 2 = 8
    }

    #[test]
    #[ignore = "Requires multi-GPU setup"]
    fn nccl_device_communicator_all_to_all_output_shape() {
        use super::super::process_group::LocalProcessGroup;

        let nccl = Arc::new(NcclLibrary::new().expect("Failed to load NCCL"));
        let id = nccl.get_unique_id().expect("Failed to get unique ID");
        let comm = NcclCommunicator::new(nccl, id, 2, 0, 0).expect("Failed to create communicator");
        let pg = LocalProcessGroup::with_rank(0, 2);
        let device_comm = NcclDeviceCommunicator::new(comm, pg);

        let device = Device::cuda_if_available(0).expect("Need CUDA");
        // Input shape must be divisible by world_size in first dim
        let input = Tensor::zeros(&[4, 8], DType::F32, &device).expect("Failed to create tensor");

        // all_to_all should preserve shape
        let output = device_comm.all_to_all(&input).expect("all_to_all failed");
        assert_eq!(output.dims(), &[4, 8]);
    }
}
