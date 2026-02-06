//! MoE Router implementation.
//!
//! Routes tokens to experts using top-k selection with softmax/sigmoid normalization.
//!
//! ## Scoring Functions
//!
//! - **Softmax** (default): Standard softmax over expert logits
//! - **Sigmoid**: Per-expert sigmoid scores (used by GLM4-MoE)
//!
//! ## Grouped Top-K
//!
//! GLM4-MoE uses grouped top-k where experts are divided into groups
//! and top-k is selected within each group.
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

/// Scoring function for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoringFunc {
    /// Standard softmax normalization (default for most MoE models).
    #[default]
    Softmax,
    /// Per-expert sigmoid scores (used by GLM4-MoE).
    Sigmoid,
}

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
    /// Scoring function to use (Softmax or Sigmoid).
    pub scoring_func: ScoringFunc,
    /// Whether to use grouped top-k selection (GLM4-MoE).
    pub use_grouped_topk: bool,
    /// Number of expert groups (for grouped top-k).
    pub num_expert_groups: Option<usize>,
    /// Top-k per group (for grouped top-k).
    pub topk_per_group: Option<usize>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            num_experts: 0,
            top_k: 2,
            renormalize: true,
            scoring_func: ScoringFunc::Softmax,
            use_grouped_topk: false,
            num_expert_groups: None,
            topk_per_group: None,
        }
    }
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

/// Top-K router with softmax/sigmoid normalization.
///
/// Routes each token to the top-k experts based on a learned gating function.
/// Supports both standard softmax and sigmoid scoring (GLM4-MoE).
pub struct TopKRouter {
    gate: Linear,
    config: RouterConfig,
    /// Optional bias for score correction (GLM4-MoE e_score_correction_bias).
    e_score_correction_bias: Option<Tensor>,
}

impl TopKRouter {
    /// Create a new TopKRouter.
    pub fn new(config: RouterConfig, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(config.hidden_size, config.num_experts, vb)?;
        Ok(Self {
            gate,
            config,
            e_score_correction_bias: None,
        })
    }

    /// Create a TopKRouter with score correction bias (GLM4-MoE).
    pub fn new_with_bias(
        config: RouterConfig,
        vb: VarBuilder,
        e_score_correction_bias: Option<Tensor>,
    ) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(config.hidden_size, config.num_experts, vb)?;
        Ok(Self {
            gate,
            config,
            e_score_correction_bias,
        })
    }

    /// Set the score correction bias.
    pub fn set_e_score_correction_bias(&mut self, bias: Tensor) {
        self.e_score_correction_bias = Some(bias);
    }

    /// Get the router logits (before scoring function).
    pub fn get_router_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.gate.forward(hidden_states)
    }

    /// Get the scoring function used.
    pub fn scoring_func(&self) -> ScoringFunc {
        self.config.scoring_func
    }

    /// Apply the scoring function to logits.
    fn apply_scoring(&self, logits: &Tensor) -> Result<Tensor> {
        match self.config.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax(logits, candle_core::D::Minus1),
            ScoringFunc::Sigmoid => {
                // Sigmoid scoring with optional bias correction
                let scores = candle_nn::ops::sigmoid(logits)?;
                if let Some(ref bias) = self.e_score_correction_bias {
                    // Add bias: scores + bias
                    scores.broadcast_add(bias)
                } else {
                    Ok(scores)
                }
            }
        }
    }

    /// Perform grouped top-k selection.
    ///
    /// Experts are divided into groups, and top-k is selected from each group.
    fn grouped_topk(
        &self,
        scores: &Tensor,
        num_groups: usize,
        topk_per_group: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_num_tokens, num_experts) = scores.dims2()?;
        let experts_per_group = num_experts / num_groups;

        let mut all_weights = Vec::with_capacity(num_groups);
        let mut all_indices = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * experts_per_group;
            // Get scores for this group and make contiguous
            let group_scores = scores.narrow(1, start, experts_per_group)?.contiguous()?;
            // Get top-k within group
            let (group_weights, group_indices) = top_k_with_indices(&group_scores, topk_per_group)?;
            // Adjust indices to global expert IDs
            let offset = Tensor::new(&[start as u32], scores.device())?;
            let adjusted_indices = group_indices.broadcast_add(&offset)?;

            all_weights.push(group_weights);
            all_indices.push(adjusted_indices);
        }

        // Concatenate all groups: [num_tokens, num_groups * topk_per_group]
        let weights = Tensor::cat(&all_weights, 1)?.contiguous()?;
        let indices = Tensor::cat(&all_indices, 1)?.contiguous()?;

        // Re-sort by weight to get overall top-k
        let total_k = self.config.top_k;
        if weights.dim(1)? > total_k {
            // Sort and take top-k
            let (final_weights, final_indices) = top_k_with_indices(&weights, total_k)?;
            // Gather the actual expert indices - need contiguous for gather
            let expert_indices = indices.gather(&final_indices.contiguous()?, 1)?;
            Ok((final_weights, expert_indices))
        } else {
            Ok((weights, indices))
        }
    }
}

impl MoERouter for TopKRouter {
    fn route(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute router logits: [num_tokens, num_experts]
        let router_logits = self.gate.forward(hidden_states)?;

        // Use fused CUDA kernel when available (only for softmax without grouped top-k)
        #[cfg(feature = "cuda-moe")]
        if self.config.scoring_func == ScoringFunc::Softmax && !self.config.use_grouped_topk {
            let config = TopKSoftmaxConfig::new(self.config.top_k, self.config.renormalize);
            return topk_softmax(&router_logits, &config);
        }

        // CPU/generic path
        // Apply scoring function
        let routing_probs = self.apply_scoring(&router_logits)?;

        // Get top-k experts and their weights
        let (top_k_weights, top_k_indices) = if self.config.use_grouped_topk {
            let num_groups = self.config.num_expert_groups.unwrap_or(1);
            let topk_per_group = self.config.topk_per_group.unwrap_or(self.config.top_k);
            self.grouped_topk(&routing_probs, num_groups, topk_per_group)?
        } else {
            top_k_with_indices(&routing_probs, self.config.top_k)?
        };

        // Optionally renormalize weights so they sum to 1
        let final_weights = if self.config.renormalize {
            let sum = top_k_weights.sum_keepdim(candle_core::D::Minus1)?;
            // Avoid division by zero
            let sum = sum.clamp(1e-10, f64::INFINITY)?;
            top_k_weights.broadcast_div(&sum)?
        } else {
            top_k_weights
        };

        Ok((final_weights, top_k_indices))
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        };

        let router = TopKRouter::new(config, vb).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();

        let (weights, _) = router.route(&hidden_states).unwrap();

        // Weights should not necessarily sum to 1 without renormalization
        assert_eq!(weights.dims(), &[3, 2]);
    }

    #[test]
    fn test_sigmoid_scoring() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = RouterConfig {
            hidden_size: 16,
            num_experts: 4,
            top_k: 2,
            renormalize: true,
            scoring_func: ScoringFunc::Sigmoid,
            ..Default::default()
        };

        let router = TopKRouter::new(config, vb).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();

        let (weights, indices) = router.route(&hidden_states).unwrap();

        assert_eq!(weights.dims(), &[3, 2]);
        assert_eq!(indices.dims(), &[3, 2]);

        // Weights should be renormalized to sum to 1
        let sums = weights.sum_keepdim(1).unwrap();
        let sums_vec: Vec<f32> = sums.flatten_all().unwrap().to_vec1().unwrap();
        for sum in sums_vec {
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_grouped_topk() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = RouterConfig {
            hidden_size: 16,
            num_experts: 8, // 8 experts in 2 groups of 4
            top_k: 4,       // Total top-k
            renormalize: true,
            scoring_func: ScoringFunc::Softmax,
            use_grouped_topk: true,
            num_expert_groups: Some(2),
            topk_per_group: Some(2), // 2 from each group
        };

        let router = TopKRouter::new(config, vb).unwrap();
        let hidden_states = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();

        let (weights, indices) = router.route(&hidden_states).unwrap();

        assert_eq!(weights.dims(), &[3, 4]); // [num_tokens, top_k]
        assert_eq!(indices.dims(), &[3, 4]);

        // Check expert indices are valid (0-7)
        let indices_vec: Vec<u32> = indices.flatten_all().unwrap().to_vec1().unwrap();
        for idx in indices_vec {
            assert!(idx < 8, "Expert index should be < 8, got {idx}");
        }
    }

    #[test]
    fn test_scoring_func_default() {
        assert_eq!(ScoringFunc::default(), ScoringFunc::Softmax);
    }

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();
        assert_eq!(config.scoring_func, ScoringFunc::Softmax);
        assert!(!config.use_grouped_topk);
        assert!(config.num_expert_groups.is_none());
    }
}
