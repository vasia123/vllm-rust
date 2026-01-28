//! MoE Router implementation.
//!
//! Routes tokens to experts using top-k selection with softmax normalization.
//!
//! ## CUDA Acceleration
//!
//! When the `cuda-moe` feature is enabled, the router uses a fused CUDA kernel
//! that combines softmax and top-k selection in a single pass for improved
//! performance. See [`crate::moe::topk_softmax`] for details.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

#[cfg(feature = "cuda-moe")]
use super::topk_softmax::{topk_softmax, TopKSoftmaxConfig};

/// Configuration for the MoE router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Hidden size of input tokens.
    pub hidden_size: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// Number of experts to route each token to.
    pub top_k: usize,
    /// Whether to renormalize routing weights after top-k selection.
    pub renormalize: bool,
}

/// Trait for MoE routers.
pub trait MoERouter: Send + Sync {
    /// Route tokens to experts.
    ///
    /// # Arguments
    /// * `hidden_states` - Token hidden states of shape `[num_tokens, hidden_size]`
    ///
    /// # Returns
    /// * `routing_weights` - Weights for each token-expert pair `[num_tokens, top_k]`
    /// * `selected_experts` - Expert indices for each token `[num_tokens, top_k]`
    fn route(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)>;

    /// Get the number of experts.
    fn num_experts(&self) -> usize;

    /// Get the top-k value.
    fn top_k(&self) -> usize;
}

/// Top-K router with softmax normalization.
///
/// Routes each token to the top-k experts based on a learned gating function.
pub struct TopKRouter {
    gate: Linear,
    config: RouterConfig,
}

impl TopKRouter {
    /// Create a new TopKRouter.
    pub fn new(config: RouterConfig, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(config.hidden_size, config.num_experts, vb)?;
        Ok(Self { gate, config })
    }

    /// Get the router logits (before softmax).
    pub fn get_router_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.gate.forward(hidden_states)
    }
}

impl MoERouter for TopKRouter {
    fn route(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute router logits: [num_tokens, num_experts]
        let router_logits = self.gate.forward(hidden_states)?;

        // Use fused CUDA kernel when available
        #[cfg(feature = "cuda-moe")]
        {
            let config = TopKSoftmaxConfig::new(self.config.top_k, self.config.renormalize);
            return topk_softmax(&router_logits, &config);
        }

        // CPU fallback path
        #[cfg(not(feature = "cuda-moe"))]
        {
            // Apply softmax to get routing probabilities
            let routing_probs = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;

            // Get top-k experts and their weights
            let (top_k_weights, top_k_indices) =
                top_k_with_indices(&routing_probs, self.config.top_k)?;

            // Optionally renormalize weights so they sum to 1
            let final_weights = if self.config.renormalize {
                let sum = top_k_weights.sum_keepdim(candle_core::D::Minus1)?;
                top_k_weights.broadcast_div(&sum)?
            } else {
                top_k_weights
            };

            Ok((final_weights, top_k_indices))
        }
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    fn top_k(&self) -> usize {
        self.config.top_k
    }
}

/// Get top-k values and their indices from a tensor.
///
/// # Arguments
/// * `tensor` - Input tensor of shape `[..., N]`
/// * `k` - Number of top values to return
///
/// # Returns
/// * `values` - Top-k values of shape `[..., k]`
/// * `indices` - Indices of top-k values of shape `[..., k]`
fn top_k_with_indices(tensor: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let dim = tensor.dims().len() - 1;
    let n = tensor.dim(dim)?;

    if k >= n {
        // If k >= n, return all values sorted
        let indices = tensor.arg_sort_last_dim(false)?;
        return Ok((tensor.contiguous()?, indices.contiguous()?));
    }

    // Sort in descending order
    let sorted_indices = tensor.arg_sort_last_dim(false)?;

    // Take top-k indices and make contiguous for gather
    let top_k_indices = sorted_indices.narrow(dim, 0, k)?.contiguous()?;

    // Make tensor contiguous for gather
    let tensor_contig = tensor.contiguous()?;

    // Gather top-k values
    let top_k_values = tensor_contig.gather(&top_k_indices, dim)?;

    Ok((top_k_values, top_k_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_top_k_with_indices() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[0.1f32, 0.4, 0.2, 0.3]], &device).unwrap();

        let (values, indices) = top_k_with_indices(&tensor, 2).unwrap();

        let values_vec: Vec<f32> = values.flatten_all().unwrap().to_vec1().unwrap();
        let indices_vec: Vec<u32> = indices.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(values_vec.len(), 2);
        assert_eq!(indices_vec.len(), 2);
        // Top 2 should be 0.4 (idx 1) and 0.3 (idx 3)
        assert!((values_vec[0] - 0.4).abs() < 1e-5);
        assert!((values_vec[1] - 0.3).abs() < 1e-5);
        assert_eq!(indices_vec[0], 1);
        assert_eq!(indices_vec[1], 3);
    }

    #[test]
    fn test_top_k_router_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = RouterConfig {
            hidden_size: 256,
            num_experts: 8,
            top_k: 2,
            renormalize: true,
        };

        let router = TopKRouter::new(config, vb).unwrap();
        assert_eq!(router.num_experts(), 8);
        assert_eq!(router.top_k(), 2);
    }

    #[test]
    fn test_top_k_router_route() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = RouterConfig {
            hidden_size: 16,
            num_experts: 4,
            top_k: 2,
            renormalize: true,
        };

        let router = TopKRouter::new(config, vb).unwrap();

        // Create some random hidden states
        let hidden_states = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();

        let (weights, indices) = router.route(&hidden_states).unwrap();

        assert_eq!(weights.dims(), &[3, 2]); // [num_tokens, top_k]
        assert_eq!(indices.dims(), &[3, 2]); // [num_tokens, top_k]

        // Check that weights sum to 1 (renormalized)
        let sums = weights.sum_keepdim(1).unwrap();
        let sums_vec: Vec<f32> = sums.flatten_all().unwrap().to_vec1().unwrap();
        for sum in sums_vec {
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_router_no_renormalize() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = RouterConfig {
            hidden_size: 16,
            num_experts: 4,
            top_k: 2,
            renormalize: false,
        };

        let router = TopKRouter::new(config, vb).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();

        let (weights, _) = router.route(&hidden_states).unwrap();

        // Weights should not necessarily sum to 1 without renormalization
        assert_eq!(weights.dims(), &[3, 2]);
    }
}
