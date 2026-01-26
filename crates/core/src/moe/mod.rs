//! Mixture of Experts (MoE) module.
//!
//! This module provides infrastructure for sparse MoE models like Mixtral,
//! including expert routing and parallel expert execution.

mod expert_layer;
mod router;

pub use expert_layer::{MoEExpert, MoEExpertConfig, MoELayer, MoELayerConfig};
pub use router::{MoERouter, TopKRouter};
