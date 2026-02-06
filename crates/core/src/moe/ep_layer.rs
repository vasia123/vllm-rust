//! Expert-Parallel MoE Layer implementation.
//!
//! This module provides an MoE layer that distributes experts across
//! multiple GPUs using Expert Parallelism (EP). Each GPU stores only
//! `num_experts / ep_size` experts, with all-to-all communication
//! used to route tokens to the correct ranks.
//!
//! ## Memory Efficiency
//!
//! Unlike replicated MoE (where all GPUs have all experts), EP divides
//! experts across ranks:
//! - 8 experts with EP=4: each GPU stores 2 experts
//! - Memory per GPU: ~1/4 of replicated approach
//!
//! ## Communication Pattern
//!
//! ```text
//! 1. Route (replicated): All ranks compute routing independently
//! 2. Dispatch: all_to_all_v sends tokens to expert-owning ranks
//! 3. Local compute: Each rank processes tokens with its local experts
//! 4. Combine: all_to_all_v gathers results back to source ranks
//! ```
//!
//! ## When to Use EP vs TP
//!
//! - EP: Large number of experts, memory-bound (Mixtral, Qwen2-MoE)
//! - TP: Few experts, compute-bound, sharding within experts

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::expert_layer::{MoEExpert, MoEExpertConfig};
use super::expert_map::{ExpertMap, ExpertPlacement};
use super::fused::FusedMoEBlockConfig;
use super::router::{MoERouter, RouterConfig, TopKRouter};
use super::token_dispatch::TokenDispatcher;
use crate::distributed::DeviceCommunicator;

/// Configuration for Expert-Parallel MoE layer.
#[derive(Debug, Clone)]
pub struct EPMoEConfig {
    /// Hidden size of input tokens.
    pub hidden_size: usize,
    /// Intermediate (FFN) size for each expert.
    pub intermediate_size: usize,
    /// Total number of experts globally.
    pub num_experts: usize,
    /// Number of experts activated per token.
    pub top_k: usize,
    /// Number of EP ranks (experts distributed across).
    pub ep_size: usize,
    /// This rank's position in EP group.
    pub ep_rank: usize,
    /// Expert placement strategy.
    pub placement: ExpertPlacement,
    /// Whether to renormalize routing weights.
    pub renormalize: bool,
}

impl EPMoEConfig {
    /// Create config for single-GPU execution (no EP).
    pub fn single_gpu(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            ep_size: 1,
            ep_rank: 0,
            placement: ExpertPlacement::Linear,
            renormalize: true,
        }
    }

    /// Get the number of experts stored on this rank.
    pub fn local_num_experts(&self) -> usize {
        self.num_experts / self.ep_size
    }
}

/// Expert-Parallel MoE Layer.
///
/// This layer loads only `num_experts / ep_size` experts, using
/// TokenDispatcher to route tokens to the correct EP ranks.
pub struct EPMoELayer {
    /// Router for computing expert assignments (replicated on all ranks).
    router: TopKRouter,
    /// Only local experts (num_experts / ep_size).
    local_experts: Vec<MoEExpert>,
    /// Token dispatcher for EP communication.
    dispatcher: TokenDispatcher,
    /// Expert placement map.
    expert_map: ExpertMap,
    /// Layer configuration.
    config: EPMoEConfig,
    /// Block configuration for fused kernels.
    /// NOTE: Currently unused - fused CUDA kernels would need custom EP-aware
    /// implementation that combines local expert computation.
    #[allow(dead_code)]
    block_config: FusedMoEBlockConfig,
}

impl EPMoELayer {
    /// Create a new Expert-Parallel MoE layer.
    ///
    /// # Arguments
    /// * `config` - EP configuration
    /// * `vb` - Variable builder for loading weights
    /// * `comm` - Device communicator for all-to-all operations
    ///
    /// # Weight Loading
    ///
    /// Only loads experts assigned to this EP rank. For example, with
    /// 8 experts and EP=4 on rank 0 with Linear placement:
    /// - Loads: experts.0, experts.1
    /// - Skips: experts.2 through experts.7
    pub fn new(
        config: EPMoEConfig,
        vb: VarBuilder,
        comm: Arc<dyn DeviceCommunicator>,
    ) -> Result<Self> {
        // Create expert map
        let expert_map = ExpertMap::new(
            config.num_experts,
            config.ep_size,
            config.ep_rank,
            config.placement,
        );

        // Create router (same on all ranks for consistent routing)
        let router_config = RouterConfig {
            hidden_size: config.hidden_size,
            num_experts: config.num_experts,
            top_k: config.top_k,
            renormalize: config.renormalize,
            ..Default::default()
        };
        let router = TopKRouter::new(router_config, vb.pp("gate"))?;

        // Load ONLY local experts
        let expert_config = MoEExpertConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        };

        let mut local_experts = Vec::with_capacity(expert_map.local_num_experts());
        for local_id in 0..expert_map.local_num_experts() {
            let global_id = expert_map.to_global(local_id);
            let expert = MoEExpert::new(&expert_config, vb.pp(format!("experts.{}", global_id)))?;
            local_experts.push(expert);
        }

        // Create token dispatcher
        let dispatcher = TokenDispatcher::new(expert_map.clone(), comm, config.top_k);

        // Auto-select block config for fused kernels
        let block_config =
            FusedMoEBlockConfig::auto_select(128, config.hidden_size, config.intermediate_size);

        Ok(Self {
            router,
            local_experts,
            dispatcher,
            expert_map,
            config,
            block_config,
        })
    }

    /// Forward pass through the Expert-Parallel MoE layer.
    ///
    /// 1. Compute routing (replicated on all ranks)
    /// 2. Dispatch tokens to EP ranks owning selected experts
    /// 3. Process tokens through local experts
    /// 4. Combine results back
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden_states.dims().to_vec();
        let hidden_size = *orig_shape.last().ok_or_else(|| {
            candle_core::Error::Msg("Tensor must have at least 1 dimension".to_string())
        })?;

        // Flatten to [num_tokens, hidden_size]
        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
        let flat_hidden = hidden_states.reshape((num_tokens, hidden_size))?;

        // 1. Route tokens to experts (replicated computation)
        let (routing_weights, expert_indices) = self.router.route(&flat_hidden)?;

        // 2. Dispatch tokens to EP ranks
        let metadata = self
            .dispatcher
            .dispatch(&flat_hidden, &expert_indices, &routing_weights)?;

        // 3. Process tokens through local experts
        let expert_output = self.forward_local(&metadata)?;

        // 4. Combine results back
        let output = self.dispatcher.combine(&expert_output, &metadata)?;

        // Reshape to original shape
        output.reshape(orig_shape)
    }

    /// Process tokens through local experts.
    ///
    /// Groups tokens by expert assignment and processes each group
    /// as a batch for efficiency. Uses index_add for O(n) scatter.
    fn forward_local(&self, metadata: &super::token_dispatch::DispatchMetadata) -> Result<Tensor> {
        let permuted_tokens = &metadata.permuted_tokens;
        let permuted_expert_ids = &metadata.permuted_expert_ids;
        let device = permuted_tokens.device();
        let dtype = permuted_tokens.dtype();
        let hidden_size = metadata.hidden_size;

        let total_tokens = permuted_tokens.dim(0)?;
        if total_tokens == 0 {
            return Tensor::zeros((0, hidden_size), dtype, device);
        }

        // Get expert IDs as vector
        let expert_ids: Vec<u32> = permuted_expert_ids.to_vec1()?;

        // Group tokens by local expert ID
        let mut expert_groups: Vec<Vec<usize>> = vec![Vec::new(); self.local_experts.len()];
        for (idx, &expert_id) in expert_ids.iter().enumerate() {
            let local_id = expert_id as usize;
            if local_id < self.local_experts.len() {
                expert_groups[local_id].push(idx);
            }
        }

        // Initialize output
        let mut output = Tensor::zeros((total_tokens, hidden_size), dtype, device)?;

        // Process each expert's tokens as a batch
        for (local_expert_id, token_indices) in expert_groups.iter().enumerate() {
            if token_indices.is_empty() {
                continue;
            }

            let expert = &self.local_experts[local_expert_id];

            // Gather input tokens for this expert using index_select
            let index_tensor = Tensor::from_vec(
                token_indices
                    .iter()
                    .map(|&i| i as u32)
                    .collect::<Vec<u32>>(),
                token_indices.len(),
                device,
            )?;
            let batch_input = permuted_tokens.index_select(&index_tensor, 0)?;

            // Batched expert forward
            let expert_output = expert.forward(&batch_input)?;

            // Scatter results back using index_add (O(n) instead of O(n²))
            output = output.index_add(&index_tensor, &expert_output, 0)?;
        }

        Ok(output)
    }

    /// Get the number of experts globally.
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    /// Get the number of local experts on this rank.
    pub fn local_num_experts(&self) -> usize {
        self.local_experts.len()
    }

    /// Get the top-k value.
    pub fn top_k(&self) -> usize {
        self.config.top_k
    }

    /// Get the EP size.
    pub fn ep_size(&self) -> usize {
        self.config.ep_size
    }

    /// Get the EP rank.
    pub fn ep_rank(&self) -> usize {
        self.config.ep_rank
    }

    /// Get the expert map.
    pub fn expert_map(&self) -> &ExpertMap {
        &self.expert_map
    }

    /// Get the block configuration for fused kernels.
    pub fn block_config(&self) -> &FusedMoEBlockConfig {
        &self.block_config
    }

    /// Set custom block configuration.
    pub fn set_block_config(&mut self, config: FusedMoEBlockConfig) {
        self.block_config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator};
    use candle_core::DType;

    fn make_mock_comm(rank: usize, world_size: usize) -> Arc<dyn DeviceCommunicator> {
        let pg = LocalProcessGroup::with_rank(rank, world_size);
        Arc::new(MockCommunicator::new(pg))
    }

    #[test]
    fn test_ep_layer_construction() {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = EPMoEConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_experts: 8,
            top_k: 2,
            ep_size: 2,
            ep_rank: 0,
            placement: ExpertPlacement::Linear,
            renormalize: true,
        };

        let comm = make_mock_comm(0, 2);
        let layer = EPMoELayer::new(config, vb, comm).unwrap();

        // Should have 4 local experts (8 / 2)
        assert_eq!(layer.local_num_experts(), 4);
        assert_eq!(layer.num_experts(), 8);
        assert_eq!(layer.top_k(), 2);
        assert_eq!(layer.ep_size(), 2);
        assert_eq!(layer.ep_rank(), 0);
    }

    #[test]
    fn test_ep_layer_construction_rank1() {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = EPMoEConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_experts: 8,
            top_k: 2,
            ep_size: 2,
            ep_rank: 1, // Rank 1
            placement: ExpertPlacement::Linear,
            renormalize: true,
        };

        let comm = make_mock_comm(1, 2);
        let layer = EPMoELayer::new(config, vb, comm).unwrap();

        assert_eq!(layer.local_num_experts(), 4);
        assert_eq!(layer.ep_rank(), 1);

        // Rank 1 should have experts 4-7 mapped to local 0-3
        assert_eq!(layer.expert_map().to_global(0), 4);
        assert_eq!(layer.expert_map().to_global(3), 7);
    }

    #[test]
    fn test_ep_layer_forward_shape() {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = EPMoEConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 4,
            top_k: 2,
            ep_size: 1, // Single rank for testing
            ep_rank: 0,
            placement: ExpertPlacement::Linear,
            renormalize: true,
        };

        let comm = make_mock_comm(0, 1);
        let layer = EPMoELayer::new(config, vb, comm).unwrap();

        // Test with 2D input
        let input_2d = Tensor::randn(0f32, 1.0, (4, 16), &device).unwrap();
        let output_2d = layer.forward(&input_2d).unwrap();
        assert_eq!(output_2d.dims(), &[4, 16]);

        // Test with 3D input
        let input_3d = Tensor::randn(0f32, 1.0, (2, 3, 16), &device).unwrap();
        let output_3d = layer.forward(&input_3d).unwrap();
        assert_eq!(output_3d.dims(), &[2, 3, 16]);
    }

    #[test]
    fn test_ep_layer_equivalence_ep1() {
        // EP=1 should behave like regular MoE layer
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = EPMoEConfig::single_gpu(16, 32, 4, 2);

        let comm = make_mock_comm(0, 1);
        let layer = EPMoELayer::new(config, vb, comm).unwrap();

        // All 4 experts should be local
        assert_eq!(layer.local_num_experts(), 4);
        assert_eq!(layer.ep_size(), 1);

        let input = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[3, 16]);
    }

    #[test]
    fn test_ep_layer_various_batch_sizes() {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = EPMoEConfig::single_gpu(16, 32, 4, 2);

        let comm = make_mock_comm(0, 1);
        let layer = EPMoELayer::new(config, vb, comm).unwrap();

        for batch_size in [1, 5, 16, 32] {
            let input = Tensor::randn(0f32, 1.0, (batch_size, 16), &device).unwrap();
            let output = layer.forward(&input).unwrap();
            assert_eq!(
                output.dims(),
                &[batch_size, 16],
                "Failed for batch_size={}",
                batch_size
            );
        }
    }

    #[test]
    fn test_ep_layer_with_round_robin() {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = EPMoEConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 8,
            top_k: 2,
            ep_size: 4,
            ep_rank: 1,
            placement: ExpertPlacement::RoundRobin,
            renormalize: true,
        };

        let comm = make_mock_comm(1, 4);
        let layer = EPMoELayer::new(config, vb, comm).unwrap();

        // Rank 1 with round-robin gets experts 1 and 5
        assert_eq!(layer.local_num_experts(), 2);
        assert_eq!(layer.expert_map().to_global(0), 1);
        assert_eq!(layer.expert_map().to_global(1), 5);
    }

    #[test]
    fn test_ep_config_single_gpu() {
        let config = EPMoEConfig::single_gpu(256, 512, 8, 2);

        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.intermediate_size, 512);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.ep_size, 1);
        assert_eq!(config.ep_rank, 0);
        assert_eq!(config.local_num_experts(), 8);
    }

    #[test]
    fn test_ep_config_local_num_experts() {
        let config = EPMoEConfig {
            hidden_size: 256,
            intermediate_size: 512,
            num_experts: 16,
            top_k: 2,
            ep_size: 4,
            ep_rank: 0,
            placement: ExpertPlacement::Linear,
            renormalize: true,
        };

        assert_eq!(config.local_num_experts(), 4);
    }
}
