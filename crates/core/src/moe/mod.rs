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
//! - [`expert_map`]: Expert placement mapping for Expert Parallelism
//! - [`token_dispatch`]: Token dispatch/combine for Expert Parallelism
//! - [`ep_layer`]: Expert-parallel MoE layer implementation
//!
//! ## Feature Flags
//!
//! - `fused-moe`: Enable fused CUDA kernels for optimized MoE execution
//! - `cuda-moe`: Enable CUDA-accelerated top-k softmax routing

mod ep_layer;
mod expert_layer;
mod expert_map;
pub mod fused;
mod lora;
mod router;
mod token_dispatch;
pub mod topk_softmax;

pub use ep_layer::{EPMoEConfig, EPMoELayer};
pub use expert_layer::{
    MoEExpert, MoEExpertConfig, MoELayer, MoELayerConfig, MoELayerWithShared,
    MoELayerWithSharedConfig,
};
pub use expert_map::{ExpertMap, ExpertPlacement};
pub use fused::{FusedMoEBlockConfig, FusedMoEConfig};
pub use lora::MoELoraWeights;
pub use router::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};
pub use token_dispatch::{DispatchMetadata, TokenDispatcher};
pub use topk_softmax::{topk_softmax, TopKSoftmaxConfig};
