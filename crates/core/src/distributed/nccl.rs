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
pub struct NcclLibrary {
    #[allow(dead_code)]
    library: Library,
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

        Ok(Self {
            library,
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
    /// # Arguments
    /// * `nccl` - Shared NCCL library instance
    /// * `unique_id` - Unique ID from rank 0 (must be broadcast first)
    /// * `world_size` - Total number of processes
    /// * `rank` - This process's rank
    /// * `device` - CUDA device ordinal for this rank
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

        // TODO: Set CUDA device before initializing communicator
        // This requires CUDA runtime bindings

        let comm = nccl.comm_init_rank(world_size, unique_id, rank)?;

        Ok(Self {
            nccl,
            comm,
            rank,
            world_size,
            device,
        })
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
}
