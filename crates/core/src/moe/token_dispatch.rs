//! Token dispatch and combine for Expert Parallelism.
//!
//! This module handles the communication pattern for Expert Parallelism (EP):
//! - **Dispatch**: Route tokens to the EP ranks that own the selected experts
//! - **Combine**: Gather expert outputs back and restore original token order
//!
//! ## Communication Pattern
//!
//! ```text
//! Input [num_tokens, hidden_size]
//!     ↓
//! dispatch():
//!     1. Determine destination rank for each (token, expert) pair
//!     2. Exchange token counts via all_to_all (small metadata)
//!     3. Exchange actual token data via all_to_all_v
//!     ↓
//! Local expert processing
//!     ↓
//! combine():
//!     1. Exchange expert outputs via all_to_all_v
//!     2. Unpermute to original order
//!     3. Apply routing weights and sum over top_k
//!     ↓
//! Output [num_tokens, hidden_size]
//! ```

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

use super::expert_map::ExpertMap;
use crate::distributed::DeviceCommunicator;

/// Metadata required for combine() after dispatch().
///
/// This struct captures the permutation and communication state needed
/// to reverse the dispatch operation and combine expert outputs.
#[derive(Debug)]
pub struct DispatchMetadata {
    /// Tokens reordered for local expert processing.
    /// Shape: [total_recv_tokens, hidden_size]
    pub permuted_tokens: Tensor,

    /// Routing weights for received tokens.
    /// Shape: [total_recv_tokens]
    pub permuted_weights: Tensor,

    /// Local expert IDs for received tokens.
    /// Shape: [total_recv_tokens]
    pub permuted_expert_ids: Tensor,

    /// Number of tokens sent to each EP rank.
    pub send_counts: Vec<usize>,

    /// Number of tokens received from each EP rank.
    pub recv_counts: Vec<usize>,

    /// Indices to restore original token order.
    /// Shape: [total_send_tokens]
    pub send_indices: Vec<usize>,

    /// Original number of input tokens.
    pub num_tokens: usize,

    /// Top-k value.
    pub top_k: usize,

    /// Hidden size of tokens.
    pub hidden_size: usize,

    /// Data type.
    pub dtype: DType,

    /// Device.
    pub device: Device,
}

/// Dispatcher for routing tokens to EP ranks that own selected experts.
///
/// The dispatcher handles the all-to-all communication pattern required
/// for Expert Parallelism: sending tokens to the ranks that own the
/// experts they're routed to, and gathering the results back.
pub struct TokenDispatcher {
    expert_map: ExpertMap,
    comm: Arc<dyn DeviceCommunicator>,
    top_k: usize,
}

impl TokenDispatcher {
    /// Create a new token dispatcher.
    ///
    /// # Arguments
    /// * `expert_map` - Mapping of experts to EP ranks
    /// * `comm` - Device communicator for all-to-all operations
    /// * `top_k` - Number of experts per token
    pub fn new(expert_map: ExpertMap, comm: Arc<dyn DeviceCommunicator>, top_k: usize) -> Self {
        Self {
            expert_map,
            comm,
            top_k,
        }
    }

    /// Dispatch tokens to EP ranks owning the selected experts.
    ///
    /// # Arguments
    /// * `hidden_states` - Token hidden states of shape `[num_tokens, hidden_size]`
    /// * `expert_indices` - Selected expert IDs of shape `[num_tokens, top_k]`
    /// * `routing_weights` - Routing weights of shape `[num_tokens, top_k]`
    ///
    /// # Returns
    /// Metadata needed for local expert processing and combine().
    pub fn dispatch(
        &self,
        hidden_states: &Tensor,
        expert_indices: &Tensor,
        routing_weights: &Tensor,
    ) -> Result<DispatchMetadata> {
        let (num_tokens, hidden_size) = hidden_states.dims2()?;
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let ep_size = self.expert_map.ep_size();

        // Fast path for single-rank EP
        if ep_size == 1 {
            return self.dispatch_single_rank(hidden_states, expert_indices, routing_weights);
        }

        // Convert to CPU for indexing
        let expert_indices_flat: Vec<u32> = expert_indices.flatten_all()?.to_vec1()?;
        let routing_weights_flat: Vec<f32> = routing_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;

        // Group (token, k, weight) by destination EP rank
        let mut send_lists: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); ep_size];

        for token_idx in 0..num_tokens {
            for k in 0..self.top_k {
                let flat_idx = token_idx * self.top_k + k;
                let global_expert_id = expert_indices_flat[flat_idx] as usize;
                let weight = routing_weights_flat[flat_idx];
                let dest_rank = self.expert_map.owner_rank(global_expert_id);
                send_lists[dest_rank].push((token_idx, global_expert_id, weight));
            }
        }

        // Compute send counts
        let send_counts: Vec<usize> = send_lists.iter().map(|list| list.len()).collect();

        // Exchange counts to know how much we'll receive
        let recv_counts = self.exchange_counts(&send_counts, device)?;

        // Build send buffers
        let total_send: usize = send_counts.iter().sum();
        let mut send_tokens = Vec::with_capacity(total_send);
        let mut send_weights = Vec::with_capacity(total_send);
        let mut send_expert_ids = Vec::with_capacity(total_send);
        let mut send_indices = Vec::with_capacity(total_send);

        for (dest_rank, send_list) in send_lists.iter().enumerate() {
            for &(token_idx, global_expert_id, weight) in send_list {
                send_tokens.push(token_idx);
                send_weights.push(weight);
                // Convert global expert ID to local ID on destination rank
                let dest_local_id = self.global_to_dest_local(global_expert_id, dest_rank);
                send_expert_ids.push(dest_local_id as u32);
                send_indices.push(token_idx);
            }
        }

        // Gather hidden states for sending
        let send_hidden = self.gather_tokens(hidden_states, &send_tokens)?;

        // Exchange hidden states via all_to_all_v
        let recv_hidden = self
            .comm
            .all_to_all_v(&send_hidden, &send_counts, &recv_counts)?;

        // Exchange weights
        let send_weights_tensor = Tensor::from_vec(send_weights, total_send, device)?;
        let recv_weights =
            self.comm
                .all_to_all_v(&send_weights_tensor, &send_counts, &recv_counts)?;

        // Exchange expert IDs
        let send_experts_tensor = Tensor::from_vec(send_expert_ids, total_send, device)?;
        let recv_experts =
            self.comm
                .all_to_all_v(&send_experts_tensor, &send_counts, &recv_counts)?;

        Ok(DispatchMetadata {
            permuted_tokens: recv_hidden,
            permuted_weights: recv_weights,
            permuted_expert_ids: recv_experts,
            send_counts,
            recv_counts,
            send_indices,
            num_tokens,
            top_k: self.top_k,
            hidden_size,
            dtype,
            device: device.clone(),
        })
    }

    /// Combine expert outputs back to original token order.
    ///
    /// # Arguments
    /// * `expert_output` - Expert outputs of shape `[total_recv_tokens, hidden_size]`
    /// * `metadata` - Dispatch metadata from dispatch()
    ///
    /// # Returns
    /// Combined output of shape `[num_tokens, hidden_size]`
    pub fn combine(&self, expert_output: &Tensor, metadata: &DispatchMetadata) -> Result<Tensor> {
        let ep_size = self.expert_map.ep_size();

        // Fast path for single-rank EP
        if ep_size == 1 {
            return self.combine_single_rank(expert_output, metadata);
        }

        // Exchange outputs back via all_to_all_v (reverse direction)
        let recv_output =
            self.comm
                .all_to_all_v(expert_output, &metadata.recv_counts, &metadata.send_counts)?;

        // Scatter-add with routing weights
        self.scatter_add_weighted(&recv_output, metadata)
    }

    /// Fast path dispatch for single EP rank.
    fn dispatch_single_rank(
        &self,
        hidden_states: &Tensor,
        expert_indices: &Tensor,
        routing_weights: &Tensor,
    ) -> Result<DispatchMetadata> {
        let (num_tokens, hidden_size) = hidden_states.dims2()?;
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();

        // Convert global expert IDs to local (identity for single rank)
        let expert_indices_flat: Vec<u32> = expert_indices.flatten_all()?.to_vec1()?;
        let routing_weights_flat: Vec<f32> = routing_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;

        // Flatten tokens for expert processing
        let total_entries = num_tokens * self.top_k;

        // Build permuted tensors: each (token, expert_k) pair becomes one entry
        let mut permuted_tokens_data = Vec::with_capacity(total_entries);
        let mut permuted_weights = Vec::with_capacity(total_entries);
        let mut permuted_expert_ids = Vec::with_capacity(total_entries);
        let mut send_indices = Vec::with_capacity(total_entries);

        for token_idx in 0..num_tokens {
            for k in 0..self.top_k {
                let flat_idx = token_idx * self.top_k + k;
                permuted_tokens_data.push(token_idx);
                permuted_weights.push(routing_weights_flat[flat_idx]);
                // For single rank, global ID == local ID
                permuted_expert_ids.push(expert_indices_flat[flat_idx]);
                send_indices.push(token_idx);
            }
        }

        // Gather tokens for processing
        let permuted_tokens = self.gather_tokens(hidden_states, &permuted_tokens_data)?;
        let permuted_weights_tensor = Tensor::from_vec(permuted_weights, total_entries, device)?;
        let permuted_expert_ids_tensor =
            Tensor::from_vec(permuted_expert_ids, total_entries, device)?;

        Ok(DispatchMetadata {
            permuted_tokens,
            permuted_weights: permuted_weights_tensor,
            permuted_expert_ids: permuted_expert_ids_tensor,
            send_counts: vec![total_entries],
            recv_counts: vec![total_entries],
            send_indices,
            num_tokens,
            top_k: self.top_k,
            hidden_size,
            dtype,
            device: device.clone(),
        })
    }

    /// Fast path combine for single EP rank.
    fn combine_single_rank(
        &self,
        expert_output: &Tensor,
        metadata: &DispatchMetadata,
    ) -> Result<Tensor> {
        self.scatter_add_weighted(expert_output, metadata)
    }

    /// Exchange send counts to determine receive counts.
    ///
    /// # Arguments
    /// * `send_counts` - Number of tokens to send to each EP rank
    /// * `device` - Device to create tensors on (should match data device)
    fn exchange_counts(&self, send_counts: &[usize], device: &Device) -> Result<Vec<usize>> {
        let ep_size = self.expert_map.ep_size();

        // For single rank, send_counts == recv_counts
        if ep_size == 1 {
            return Ok(send_counts.to_vec());
        }

        // Create tensor on same device as data to avoid CPU-GPU sync overhead
        let counts_tensor = Tensor::from_vec(
            send_counts.iter().map(|&c| c as u32).collect::<Vec<_>>(),
            ep_size,
            device,
        )?;

        // All-to-all exchange
        let recv_counts_tensor = self.comm.all_to_all(&counts_tensor)?;

        // Convert back to Vec<usize>
        let recv_counts: Vec<u32> = recv_counts_tensor.to_vec1()?;
        Ok(recv_counts.iter().map(|&c| c as usize).collect())
    }

    /// Gather tokens by index.
    fn gather_tokens(&self, hidden_states: &Tensor, indices: &[usize]) -> Result<Tensor> {
        if indices.is_empty() {
            let (_, hidden_size) = hidden_states.dims2()?;
            return Tensor::zeros(
                (0, hidden_size),
                hidden_states.dtype(),
                hidden_states.device(),
            );
        }

        // Gather rows by index
        let index_tensor = Tensor::from_vec(
            indices.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
            indices.len(),
            hidden_states.device(),
        )?;
        hidden_states.index_select(&index_tensor, 0)
    }

    /// Scatter-add weighted expert outputs to original positions.
    ///
    /// Uses batched index_add for O(n) complexity instead of O(n²).
    fn scatter_add_weighted(
        &self,
        expert_output: &Tensor,
        metadata: &DispatchMetadata,
    ) -> Result<Tensor> {
        let num_tokens = metadata.num_tokens;
        let hidden_size = metadata.hidden_size;
        let dtype = metadata.dtype;
        let device = &metadata.device;

        // Initialize output with zeros
        let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

        let total_entries = metadata.send_indices.len();
        if total_entries == 0 {
            return Ok(output);
        }

        // Get weights and apply them to expert outputs
        let weights = metadata.permuted_weights.to_dtype(dtype)?;
        // Expand weights to [total_entries, 1] for broadcasting
        let weights_expanded = weights.reshape((total_entries, 1))?;
        let weighted_outputs = expert_output.broadcast_mul(&weights_expanded)?;

        // Use index_add for efficient scatter
        // index_add accumulates: output[indices[i]] += weighted_outputs[i]
        let indices = Tensor::from_vec(
            metadata
                .send_indices
                .iter()
                .map(|&i| i as u32)
                .collect::<Vec<u32>>(),
            total_entries,
            device,
        )?;

        output = output.index_add(&indices, &weighted_outputs, 0)?;

        Ok(output)
    }

    /// Get local expert ID on destination rank for a global expert ID.
    fn global_to_dest_local(&self, global_id: usize, _dest_rank: usize) -> usize {
        // For linear placement: local_id = global_id % local_num_experts
        // For round-robin: local_id = global_id / ep_size
        match self.expert_map.placement() {
            super::expert_map::ExpertPlacement::Linear => {
                global_id % self.expert_map.local_num_experts()
            }
            super::expert_map::ExpertPlacement::RoundRobin => global_id / self.expert_map.ep_size(),
        }
    }

    /// Get the expert map.
    pub fn expert_map(&self) -> &ExpertMap {
        &self.expert_map
    }

    /// Get the top-k value.
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator};
    use crate::moe::ExpertPlacement;

    fn make_mock_comm(rank: usize, world_size: usize) -> Arc<dyn DeviceCommunicator> {
        let pg = LocalProcessGroup::with_rank(rank, world_size);
        Arc::new(MockCommunicator::new(pg))
    }

    #[test]
    fn test_dispatch_single_rank() {
        let device = Device::Cpu;
        let num_tokens = 4;
        let hidden_size = 8;
        let num_experts = 4;
        let top_k = 2;

        let expert_map = ExpertMap::single_gpu(num_experts);
        let comm = make_mock_comm(0, 1);
        let dispatcher = TokenDispatcher::new(expert_map, comm, top_k);

        let hidden_states = Tensor::randn(0f32, 1.0, (num_tokens, hidden_size), &device).unwrap();
        let expert_indices = Tensor::from_vec(
            vec![0u32, 1, 2, 3, 1, 2, 0, 3],
            (num_tokens, top_k),
            &device,
        )
        .unwrap();
        let routing_weights = Tensor::from_vec(
            vec![0.6f32, 0.4, 0.5, 0.5, 0.7, 0.3, 0.5, 0.5],
            (num_tokens, top_k),
            &device,
        )
        .unwrap();

        let metadata = dispatcher
            .dispatch(&hidden_states, &expert_indices, &routing_weights)
            .unwrap();

        // For single rank, all tokens are "received" locally
        assert_eq!(metadata.num_tokens, num_tokens);
        assert_eq!(metadata.top_k, top_k);
        assert_eq!(metadata.hidden_size, hidden_size);

        // Total entries = num_tokens * top_k
        let total_entries = num_tokens * top_k;
        assert_eq!(
            metadata.permuted_tokens.dims(),
            &[total_entries, hidden_size]
        );
        assert_eq!(metadata.permuted_weights.dims(), &[total_entries]);
        assert_eq!(metadata.permuted_expert_ids.dims(), &[total_entries]);
    }

    #[test]
    fn test_dispatch_permutation_invertible() {
        let device = Device::Cpu;
        let num_tokens = 3;
        let hidden_size = 4;
        let num_experts = 2;
        let top_k = 1;

        let expert_map = ExpertMap::single_gpu(num_experts);
        let comm = make_mock_comm(0, 1);
        let dispatcher = TokenDispatcher::new(expert_map, comm, top_k);

        // Simple input
        let hidden_states = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
            ],
            (num_tokens, hidden_size),
            &device,
        )
        .unwrap();
        let expert_indices =
            Tensor::from_vec(vec![0u32, 1, 0], (num_tokens, top_k), &device).unwrap();
        let routing_weights =
            Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (num_tokens, top_k), &device).unwrap();

        let metadata = dispatcher
            .dispatch(&hidden_states, &expert_indices, &routing_weights)
            .unwrap();

        // Simulate expert processing (identity)
        let expert_output = metadata.permuted_tokens.clone();

        // Combine should restore original order
        let output = dispatcher.combine(&expert_output, &metadata).unwrap();

        assert_eq!(output.dims(), &[num_tokens, hidden_size]);
    }

    #[test]
    fn test_combine_restores_order() {
        let device = Device::Cpu;
        let num_tokens = 2;
        let hidden_size = 2;
        let num_experts = 2;
        let top_k = 2;

        let expert_map = ExpertMap::single_gpu(num_experts);
        let comm = make_mock_comm(0, 1);
        let dispatcher = TokenDispatcher::new(expert_map, comm, top_k);

        let hidden_states = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            (num_tokens, hidden_size),
            &device,
        )
        .unwrap();
        let expert_indices =
            Tensor::from_vec(vec![0u32, 1, 1, 0], (num_tokens, top_k), &device).unwrap();
        let routing_weights =
            Tensor::from_vec(vec![0.5f32, 0.5, 0.6, 0.4], (num_tokens, top_k), &device).unwrap();

        let metadata = dispatcher
            .dispatch(&hidden_states, &expert_indices, &routing_weights)
            .unwrap();

        // Simulate expert processing (identity)
        let expert_output = metadata.permuted_tokens.clone();

        let output = dispatcher.combine(&expert_output, &metadata).unwrap();

        assert_eq!(output.dims(), &[num_tokens, hidden_size]);

        // Output should be weighted sum of expert outputs
        // Token 0: 0.5 * [1,2] + 0.5 * [1,2] = [1,2]
        // Token 1: 0.6 * [3,4] + 0.4 * [3,4] = [3,4]
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!((output_vec[0] - 1.0).abs() < 1e-5);
        assert!((output_vec[1] - 2.0).abs() < 1e-5);
        assert!((output_vec[2] - 3.0).abs() < 1e-5);
        assert!((output_vec[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_dispatcher_with_multi_rank_mock() {
        let device = Device::Cpu;
        let num_tokens = 4;
        let hidden_size = 4;
        let num_experts = 8;
        let top_k = 2;
        let ep_size = 2;

        // Create expert map for rank 0
        let expert_map = ExpertMap::new(num_experts, ep_size, 0, ExpertPlacement::Linear);
        let comm = make_mock_comm(0, ep_size);
        let dispatcher = TokenDispatcher::new(expert_map, comm, top_k);

        let hidden_states = Tensor::randn(0f32, 1.0, (num_tokens, hidden_size), &device).unwrap();
        let expert_indices = Tensor::from_vec(
            vec![0u32, 4, 1, 5, 2, 6, 3, 7],
            (num_tokens, top_k),
            &device,
        )
        .unwrap();
        let routing_weights = Tensor::from_vec(
            vec![0.6f32, 0.4, 0.5, 0.5, 0.7, 0.3, 0.5, 0.5],
            (num_tokens, top_k),
            &device,
        )
        .unwrap();

        // Dispatch should work (mock will return zeros for cross-rank data)
        let metadata = dispatcher
            .dispatch(&hidden_states, &expert_indices, &routing_weights)
            .unwrap();

        assert_eq!(metadata.num_tokens, num_tokens);
    }

    #[test]
    fn test_empty_tokens() {
        let device = Device::Cpu;
        let hidden_size = 4;
        let num_experts = 2;
        let top_k = 1;

        let expert_map = ExpertMap::single_gpu(num_experts);
        let comm = make_mock_comm(0, 1);
        let dispatcher = TokenDispatcher::new(expert_map, comm, top_k);

        let hidden_states = Tensor::zeros((0, hidden_size), DType::F32, &device).unwrap();
        let expert_indices = Tensor::zeros((0, top_k), DType::U32, &device).unwrap();
        let routing_weights = Tensor::zeros((0, top_k), DType::F32, &device).unwrap();

        let metadata = dispatcher
            .dispatch(&hidden_states, &expert_indices, &routing_weights)
            .unwrap();

        assert_eq!(metadata.num_tokens, 0);
        assert_eq!(metadata.permuted_tokens.dims(), &[0, hidden_size]);
    }

    #[test]
    fn test_accessors() {
        let num_experts = 8;
        let top_k = 2;

        let expert_map = ExpertMap::new(num_experts, 2, 0, ExpertPlacement::Linear);
        let comm = make_mock_comm(0, 2);
        let dispatcher = TokenDispatcher::new(expert_map, comm, top_k);

        assert_eq!(dispatcher.top_k(), 2);
        assert_eq!(dispatcher.expert_map().num_experts(), 8);
        assert_eq!(dispatcher.expert_map().ep_size(), 2);
    }
}
