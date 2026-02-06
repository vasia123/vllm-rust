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

    /// All-to-all: each rank sends distinct data to each other rank.
    ///
    /// Input tensor is split into `world_size` equal chunks along dimension 0.
    /// Chunk i is sent to rank i, and this rank receives chunk j from rank j.
    ///
    /// Input shape: [world_size * chunk_size, ...]
    /// Output shape: [world_size * chunk_size, ...] (same shape, different data)
    ///
    /// For single GPU, this is identity.
    fn all_to_all(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Variable-size all-to-all: each rank sends/receives different amounts to/from each rank.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor, total size along dim 0 equals sum of send_splits
    /// * `send_splits` - Number of elements to send to each rank (length = world_size)
    /// * `recv_splits` - Number of elements to receive from each rank (length = world_size)
    ///
    /// Output tensor has size sum(recv_splits) along dimension 0.
    ///
    /// For single GPU, this is identity (send_splits and recv_splits must be equal).
    fn all_to_all_v(
        &self,
        tensor: &Tensor,
        send_splits: &[usize],
        recv_splits: &[usize],
    ) -> Result<Tensor>;
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

    fn all_to_all(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.process_group.is_single() {
            return Ok(tensor.clone());
        }
        // For testing multi-GPU: simulate by permuting chunks
        // In a real all-to-all, chunk i from rank j goes to rank i from rank j
        // For mock with simulated multi-rank, we just return identity
        // since we can't actually communicate between processes
        Ok(tensor.clone())
    }

    fn all_to_all_v(
        &self,
        tensor: &Tensor,
        send_splits: &[usize],
        recv_splits: &[usize],
    ) -> Result<Tensor> {
        if self.process_group.is_single() {
            // For single GPU, send_splits should equal recv_splits
            debug_assert_eq!(send_splits, recv_splits);
            return Ok(tensor.clone());
        }

        // For mock multi-GPU testing:
        // We can't actually exchange data between processes, but we can
        // simulate the shape transformation for flow testing.
        let total_send: usize = send_splits.iter().sum();
        let total_recv: usize = recv_splits.iter().sum();
        let dims = tensor.dims();

        if dims.is_empty() {
            return Err(super::error::DistributedError::ShapeMismatch {
                expected: vec![total_recv],
                actual: dims.to_vec(),
            });
        }

        // If sizes match, return tensor as-is (best mock for flow testing)
        if total_send == total_recv {
            return Ok(tensor.clone());
        }

        // If sizes differ, we need to reshape output
        // For smaller recv: narrow the tensor
        // For larger recv: pad with zeros
        let mut new_dims = dims.to_vec();
        new_dims[0] = total_recv;

        if total_recv <= total_send {
            // Take first total_recv elements
            Ok(tensor.narrow(0, 0, total_recv)?)
        } else {
            // Pad with zeros
            let output = Tensor::zeros(new_dims.as_slice(), tensor.dtype(), tensor.device())?;
            // Copy existing data to start of output
            let indices = Tensor::from_vec(
                (0..total_send as u32).collect::<Vec<u32>>(),
                total_send,
                tensor.device(),
            )?;
            Ok(output.index_add(&indices, tensor, 0)?)
        }
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

    #[test]
    fn mock_all_to_all_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[4, 3]);
        let output = comm.all_to_all(&input).unwrap();

        // Single GPU: identity
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn mock_all_to_all_multi_gpu_simulation() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let comm = MockCommunicator::new(pg);

        // Input: [world_size * chunk_size, hidden]
        let input = make_test_tensor(&[8, 3]); // 4 ranks, 2 elements each
        let output = comm.all_to_all(&input).unwrap();

        // Mock returns identity for simulated multi-GPU
        assert_eq!(output.dims(), &[8, 3]);
    }

    #[test]
    fn mock_all_to_all_v_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[5, 3]);
        let send_splits = vec![5];
        let recv_splits = vec![5];

        let output = comm
            .all_to_all_v(&input, &send_splits, &recv_splits)
            .unwrap();

        // Single GPU: identity
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn mock_all_to_all_v_multi_gpu_simulation() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let comm = MockCommunicator::new(pg);

        // Sending different amounts to each rank
        let input = make_test_tensor(&[10, 3]); // Total 10 tokens to send
        let send_splits = vec![2, 3, 2, 3]; // Send 2,3,2,3 to ranks 0,1,2,3
        let recv_splits = vec![1, 4, 2, 3]; // Receive 1,4,2,3 from ranks 0,1,2,3

        let output = comm
            .all_to_all_v(&input, &send_splits, &recv_splits)
            .unwrap();

        // Output should have sum(recv_splits) = 10 along dim 0
        assert_eq!(output.dims(), &[10, 3]);
    }

    #[test]
    fn mock_all_to_all_v_variable_output_size() {
        let pg = LocalProcessGroup::with_rank(1, 2);
        let comm = MockCommunicator::new(pg);

        let input = make_test_tensor(&[6, 4]);
        let send_splits = vec![2, 4]; // Send 2 to rank 0, 4 to rank 1
        let recv_splits = vec![3, 5]; // Receive 3 from rank 0, 5 from rank 1

        let output = comm
            .all_to_all_v(&input, &send_splits, &recv_splits)
            .unwrap();

        // Output shape based on recv_splits sum
        assert_eq!(output.dims(), &[8, 4]);
    }
}
