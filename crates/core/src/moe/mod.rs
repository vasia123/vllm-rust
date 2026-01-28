//! Mixture of Experts (MoE) module.
//!
//! This module provides infrastructure for sparse MoE models like Mixtral,
//! including expert routing and parallel expert execution.
//!
//! ## Submodules
//!
//! - [`expert_layer`]: Basic MoE expert and layer implementations
//! - [`router`]: Top-K routing with softmax normalization
//! - [`fused`]: Optimized fused MoE kernels (when enabled)
//! - [`topk_softmax`]: Fused top-k softmax for routing (CUDA accelerated)
//!
//! ## Feature Flags
//!
//! - `fused-moe`: Enable fused CUDA kernels for optimized MoE execution
//! - `cuda-moe`: Enable CUDA-accelerated top-k softmax routing

mod expert_layer;
pub mod fused;
mod router;
pub mod topk_softmax;

pub use expert_layer::{MoEExpert, MoEExpertConfig, MoELayer, MoELayerConfig};
pub use fused::{FusedMoEBlockConfig, FusedMoEConfig};
pub use router::{MoERouter, RouterConfig, TopKRouter};
pub use topk_softmax::{topk_softmax, TopKSoftmaxConfig};
