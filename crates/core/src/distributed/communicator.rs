//! Device communicator for collective operations.
//!
//! Provides abstractions for GPU-to-GPU communication primitives
//! like all_reduce, all_gather, reduce_scatter, and broadcast.

use candle_core::Tensor;

use super::error::Result;
use super::process_group::ProcessGroup;

/// Reduction operations for collective primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise product.
    Product,
    /// Element-wise minimum.
    Min,
    /// Element-wise maximum.
    Max,
    /// Average (sum / world_size).
    Average,
}

/// Trait for device-to-device communication.
///
/// Implementations can use NCCL for real multi-GPU, or be no-ops for single GPU.
pub trait DeviceCommunicator: Send + Sync {
    /// Get the underlying process group.
    fn process_group(&self) -> &dyn ProcessGroup;

    /// All-reduce: apply reduction across all ranks, result on all ranks.
    ///
    /// For single GPU, this is identity (returns input unchanged).
    fn all_reduce(&self, tensor: &Tensor, op: ReduceOp) -> Result<Tensor>;

    /// All-gather: gather tensors from all ranks along dimension.
    ///
    /// Input shape: [dim0, dim1, ...]
    /// Output shape: [dim0 * world_size, dim1, ...] (if gather_dim=0)
    ///
    /// For single GPU, this is identity.
    fn all_gather(&self, tensor: &Tensor, gather_dim: usize) -> Result<Tensor>;

    /// Reduce-scatter: reduce and scatter result across ranks.
    ///
    /// Opposite of all_gather: each rank gets a portion of the reduced result.
    ///
    /// For single GPU, this is identity.
    fn reduce_scatter(&self, tensor: &Tensor, scatter_dim: usize, op: ReduceOp) -> Result<Tensor>;

    /// Broadcast: send tensor from source rank to all other ranks.
    ///
    /// For single GPU, this is identity.
    fn broadcast(&self, tensor: &Tensor, src_rank: usize) -> Result<Tensor>;

    /// Point-to-point send.
    fn send(&self, tensor: &Tensor, dst_rank: usize) -> Result<()>;

    /// Point-to-point receive.
    fn recv(&self, shape: &[usize], dtype: candle_core::DType, src_rank: usize) -> Result<Tensor>;

    /// Barrier: synchronize all ranks.
    fn barrier(&self) -> Result<()>;
}

/// Mock communicator for single-GPU execution.
///
/// All collective operations are identity/no-ops since there's only one rank.
/// This follows vLLM's pattern: `if world_size == 1: return input_`
pub struct MockCommunicator<P: ProcessGroup> {
    process_group: P,
}

impl<P: ProcessGroup> MockCommunicator<P> {
    /// Create a new mock communicator with the given process group.
    pub fn new(process_group: P) -> Self {
        Self { process_group }
    }
}

impl<P: ProcessGroup + Send + Sync> DeviceCommunicator for MockCommunicator<P> {
    fn process_group(&self) -> &dyn ProcessGroup {
        &self.process_group
    }

    fn all_reduce(&self, tensor: &Tensor, _op: ReduceOp) -> Result<Tensor> {
        // Single GPU: identity
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }
        // For testing multi-GPU logic: still return input
        // Real NCCL would do actual reduction
        Ok(tensor.clone())
    }

    fn all_gather(&self, tensor: &Tensor, gather_dim: usize) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }
        // For testing: simulate gathering by repeating tensor
        let world_size = self.process_group.world_size();
        let tensors: Vec<Tensor> = (0..world_size).map(|_| tensor.clone()).collect();
        Ok(Tensor::cat(&tensors, gather_dim)?)
    }

    fn reduce_scatter(&self, tensor: &Tensor, scatter_dim: usize, _op: ReduceOp) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }
        // For testing: simulate scatter by taking a slice
        let world_size = self.process_group.world_size();
        let rank = self.process_group.rank();
        let dim_size = tensor.dim(scatter_dim)?;
        let chunk_size = dim_size / world_size;
        let start = rank * chunk_size;
        Ok(tensor.narrow(scatter_dim, start, chunk_size)?)
    }

    fn broadcast(&self, tensor: &Tensor, _src_rank: usize) -> Result<Tensor> {
        // Broadcast is identity for mock (every rank has same data)
        Ok(tensor.clone())
    }

    fn send(&self, _tensor: &Tensor, _dst_rank: usize) -> Result<()> {
        // No-op for mock
        Ok(())
    }

    fn recv(&self, shape: &[usize], dtype: candle_core::DType, _src_rank: usize) -> Result<Tensor> {
        // Return zeros for mock
        Ok(Tensor::zeros(shape, dtype, &candle_core::Device::Cpu)?)
    }

    fn barrier(&self) -> Result<()> {
        // No-op for mock
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::LocalProcessGroup;
    use candle_core::{DType, Device};

    fn make_test_tensor(shape: &[usize]) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn mock_all_reduce_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[2, 3]);
        let output = comm.all_reduce(&input, ReduceOp::Sum).unwrap();

        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn mock_all_gather_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[2, 3]);
        let output = comm.all_gather(&input, 0).unwrap();

        // Single GPU: identity
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn mock_all_gather_multi_gpu_simulation() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[2, 3]);
        let output = comm.all_gather(&input, 0).unwrap();

        // Simulated: 4 GPUs gathering along dim 0
        assert_eq!(output.dims(), &[8, 3]);
    }

    #[test]
    fn mock_reduce_scatter_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[4, 3]);
        let output = comm.reduce_scatter(&input, 0, ReduceOp::Sum).unwrap();

        // Single GPU: identity
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn mock_reduce_scatter_multi_gpu_simulation() {
        let pg = LocalProcessGroup::with_rank(1, 4);
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[8, 3]);
        let output = comm.reduce_scatter(&input, 0, ReduceOp::Sum).unwrap();

        // Simulated: rank 1 gets slice [2:4, :]
        assert_eq!(output.dims(), &[2, 3]);
    }

    #[test]
    fn mock_broadcast_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[2, 3]);
        let output = comm.broadcast(&input, 0).unwrap();

        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn mock_barrier_no_error() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        comm.barrier().unwrap();
    }

    #[test]
    fn mock_send_recv_no_error() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let tensor = make_test_tensor(&[2, 3]);
        comm.send(&tensor, 0).unwrap();

        let received = comm.recv(&[2, 3], DType::F32, 0).unwrap();
        assert_eq!(received.dims(), &[2, 3]);
    }

    #[test]
    fn reduce_op_debug() {
        // Ensure all ReduceOp variants are accessible
        let ops = [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Average,
        ];
        for op in ops {
            let _ = format!("{:?}", op);
        }
    }

    #[test]
    fn process_group_accessible_via_trait() {
        let pg = LocalProcessGroup::with_rank(2, 8);
        let comm = MockCommunicator::new(pg);

        let pg_ref = comm.process_group();
        assert_eq!(pg_ref.rank(), 2);
        assert_eq!(pg_ref.world_size(), 8);
    }
}
