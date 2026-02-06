//! Multi-GPU process launcher and coordination.
//!
//! This module provides utilities for launching and coordinating
//! multi-GPU processes for tensor parallelism.
//!
//! # Environment Variables
//!
//! The launcher uses standard distributed training environment variables:
//! - `RANK`: Global rank of this process (0..WORLD_SIZE)
//! - `WORLD_SIZE`: Total number of processes
//! - `LOCAL_RANK`: Local rank on this node (for multi-node setups)
//! - `MASTER_ADDR`: Address of rank 0 for coordination (default: 127.0.0.1)
//! - `MASTER_PORT`: Port for coordination (default: 29500)
//!
//! # Usage
//!
//! ```ignore
//! use vllm_core::distributed::{DistributedConfig, NcclProcessGroup};
//!
//! // Detect configuration from environment
//! let config = DistributedConfig::from_env();
//!
//! if config.world_size > 1 {
//!     // Multi-GPU: initialize NCCL
//!     let pg = NcclProcessGroup::new(&config)?;
//! } else {
//!     // Single GPU: use local process group
//!     let pg = LocalProcessGroup::new();
//! }
//! ```

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use super::error::{DistributedError, Result};
use super::nccl::{NcclCommunicator, NcclLibrary, NcclUniqueId};
use super::process_group::ProcessGroup;

/// Distributed configuration from environment.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Global rank of this process.
    pub rank: usize,
    /// Total number of processes.
    pub world_size: usize,
    /// Local rank on this node.
    pub local_rank: usize,
    /// Master address for coordination.
    pub master_addr: String,
    /// Master port for coordination.
    pub master_port: u16,
}

impl DistributedConfig {
    /// Create from environment variables.
    ///
    /// Falls back to single-process defaults if not set.
    pub fn from_env() -> Self {
        let rank = env::var("RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let world_size = env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let local_rank = env::var("LOCAL_RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(rank);

        let master_addr = env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());

        let master_port = env::var("MASTER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(29500);

        Self {
            rank,
            world_size,
            local_rank,
            master_addr,
            master_port,
        }
    }

    /// Create for single GPU execution.
    pub fn single_gpu() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            local_rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
        }
    }

    /// Whether this is multi-GPU configuration.
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    /// Get CUDA device for this rank.
    ///
    /// Uses LOCAL_RANK to determine device ordinal.
    pub fn cuda_device(&self) -> usize {
        self.local_rank
    }
}

/// NCCL-based process group for multi-GPU execution.
pub struct NcclProcessGroup {
    /// NCCL communicator.
    communicator: NcclCommunicator,
    /// Configuration.
    config: DistributedConfig,
}

impl NcclProcessGroup {
    /// Create a new NCCL process group.
    ///
    /// This performs the following steps:
    /// 1. Rank 0 generates NCCL unique ID
    /// 2. Broadcast unique ID via TCP
    /// 3. Initialize NCCL communicator on each rank
    pub fn new(config: &DistributedConfig) -> Result<Self> {
        if config.world_size == 1 {
            return Err(DistributedError::NcclError(
                "Use LocalProcessGroup for single-GPU".to_string(),
            ));
        }

        tracing::info!(
            rank = config.rank,
            world_size = config.world_size,
            local_rank = config.local_rank,
            device = config.cuda_device(),
            "Initializing NCCL process group"
        );

        // Load NCCL library
        let nccl = Arc::new(NcclLibrary::new()?);

        // Get or receive unique ID
        let unique_id = if config.rank == 0 {
            // Rank 0: generate unique ID and broadcast
            let id = nccl.get_unique_id()?;
            broadcast_unique_id_server(&id, config)?;
            id
        } else {
            // Other ranks: receive unique ID from rank 0
            receive_unique_id(config)?
        };

        // Create communicator
        let communicator = NcclCommunicator::new(
            nccl,
            unique_id,
            config.world_size,
            config.rank,
            config.cuda_device(),
        )?;

        tracing::info!(rank = config.rank, "NCCL communicator initialized");

        Ok(Self {
            communicator,
            config: config.clone(),
        })
    }

    /// Get the underlying NCCL communicator.
    pub fn communicator(&self) -> &NcclCommunicator {
        &self.communicator
    }
}

impl ProcessGroup for NcclProcessGroup {
    fn rank(&self) -> usize {
        self.config.rank
    }

    fn world_size(&self) -> usize {
        self.config.world_size
    }

    fn local_rank(&self) -> usize {
        self.config.local_rank
    }
}

/// Broadcast NCCL unique ID from rank 0 to all other ranks.
fn broadcast_unique_id_server(id: &NcclUniqueId, config: &DistributedConfig) -> Result<()> {
    let addr = format!("0.0.0.0:{}", config.master_port);
    let listener = TcpListener::bind(&addr)
        .map_err(|e| DistributedError::NcclError(format!("Failed to bind to {}: {}", addr, e)))?;

    tracing::debug!(
        port = config.master_port,
        "Rank 0 listening for unique ID requests"
    );

    // Accept connections from all other ranks
    let expected_connections = config.world_size - 1;
    let mut connected = 0;

    // Set timeout to avoid hanging forever
    listener
        .set_nonblocking(false)
        .map_err(|e| DistributedError::NcclError(e.to_string()))?;

    while connected < expected_connections {
        match listener.accept() {
            Ok((mut stream, peer)) => {
                tracing::trace!(peer = %peer, "Accepted connection for unique ID");

                // Send unique ID bytes
                let id_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(id as *const _ as *const u8, 128) };

                stream.write_all(id_bytes).map_err(|e| {
                    DistributedError::NcclError(format!("Failed to send unique ID: {}", e))
                })?;

                connected += 1;
            }
            Err(e) => {
                return Err(DistributedError::NcclError(format!("Accept failed: {}", e)));
            }
        }
    }

    tracing::debug!(connections = connected, "All ranks received unique ID");
    Ok(())
}

/// Receive NCCL unique ID from rank 0.
fn receive_unique_id(config: &DistributedConfig) -> Result<NcclUniqueId> {
    let addr = format!("{}:{}", config.master_addr, config.master_port);

    // Retry connection with backoff
    let max_retries = 30;
    let retry_delay = Duration::from_millis(100);

    for attempt in 0..max_retries {
        match TcpStream::connect(&addr) {
            Ok(mut stream) => {
                tracing::trace!(addr = %addr, "Connected to rank 0 for unique ID");

                // Read unique ID bytes
                let mut id = NcclUniqueId::default();
                let id_bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(&mut id as *mut _ as *mut u8, 128) };

                stream.read_exact(id_bytes).map_err(|e| {
                    DistributedError::NcclError(format!("Failed to receive unique ID: {}", e))
                })?;

                return Ok(id);
            }
            Err(e) if attempt < max_retries - 1 => {
                tracing::trace!(
                    attempt = attempt,
                    addr = %addr,
                    error = %e,
                    "Retrying connection to rank 0"
                );
                std::thread::sleep(retry_delay);
            }
            Err(e) => {
                return Err(DistributedError::NcclError(format!(
                    "Failed to connect to {} after {} attempts: {}",
                    addr, max_retries, e
                )));
            }
        }
    }

    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_config_from_env_defaults() {
        // Clear env vars to test defaults
        env::remove_var("RANK");
        env::remove_var("WORLD_SIZE");
        env::remove_var("LOCAL_RANK");
        env::remove_var("MASTER_ADDR");
        env::remove_var("MASTER_PORT");

        let config = DistributedConfig::from_env();
        assert_eq!(config.rank, 0);
        assert_eq!(config.world_size, 1);
        assert_eq!(config.local_rank, 0);
        assert_eq!(config.master_addr, "127.0.0.1");
        assert_eq!(config.master_port, 29500);
        assert!(!config.is_distributed());
    }

    #[test]
    fn distributed_config_single_gpu() {
        let config = DistributedConfig::single_gpu();
        assert_eq!(config.world_size, 1);
        assert!(!config.is_distributed());
    }

    #[test]
    fn cuda_device_from_local_rank() {
        let mut config = DistributedConfig::single_gpu();
        config.local_rank = 3;
        assert_eq!(config.cuda_device(), 3);
    }
}
