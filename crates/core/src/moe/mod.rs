//! Mixture of Experts (MoE) shared infrastructure.
//!
//! This module is the canonical home for sparse MoE primitives used by
//! Mixtral, Qwen2/3-MoE, DeepSeek V2/V3, GLM4-MoE, ExaoneMoE,
//! MiniMax-Text01/M2, ERNIE 4.5-MoE, GraniteMoE, OLMoE, Bailing-MoE,
//! Pangu and the rest of the MoE zoo. The architectural rationale and
//! the migration roadmap are recorded in [ADR-0011].
//!
//! [ADR-0011]: ../../../../docs/adr/0011-moe-router-and-layer.md
//!
//! ## Pick the right primitive
//!
//! | Use case | Primitive |
//! |---|---|
//! | Routed-only MoE (Mixtral, Qwen2/3-MoE, Grok-1) | [`MoELayer`] |
//! | Routed + shared experts (DeepSeek family, MiniMax-M2 with shared, ExaoneMoE shared) | [`MoELayerWithShared`] |
//! | Quantized expert weights (GPTQ-style INT4/INT8) | [`QuantizedMoELayer`] |
//! | Expert-parallel sharding across ranks | [`EPMoELayer`] (+ [`EplbState`] for live load balancing) |
//! | Just the routing component (when expert dispatch must stay bespoke) | [`TopKRouter`] |
//!
//! All routed paths share [`RouterConfig`] / [`ScoringFunc`] knobs:
//! softmax vs. sigmoid scoring, plain vs. grouped top-k, optional
//! `e_score_correction_bias` (DeepSeek V3 / GLM4-MoE), `routed_scaling_factor`,
//! per-token renormalization.
//!
//! ## Why a single trait + single struct
//!
//! MoE routers across the zoo differ along ~6 declarative axes
//! (scoring, top-k flavor, score bias, routed scale, group routing,
//! renormalization). Encoding them as fields on `RouterConfig` collapses
//! ~150 LOC of bespoke routing per model into a 10-line `RouterConfig`
//! literal. The same applies to `MoELayer`'s expert dispatch: token
//! grouping, fused GEMM, all-to-all dispatch — written once, optimized
//! once.
//!
//! ## Submodules
//!
//! - [`expert_layer`]: [`MoEExpert`], [`MoELayer`], [`MoELayerWithShared`]
//! - [`router`]: [`MoERouter`] trait + [`TopKRouter`] impl
//! - [`quantized_experts`]: [`QuantizedMoELayer`] for INT4/INT8 weights
//! - [`ep_layer`]: [`EPMoELayer`] expert-parallel variant
//! - [`eplb`] / [`eplb_execute`]: live expert load balancing across ranks
//! - [`expert_map`]: logical-to-physical expert mapping (EP)
//! - [`fused`]: block-fused MoE GEMM kernels
//! - [`topk_softmax`]: fused top-k + softmax kernel (CUDA-accelerated)
//! - [`token_dispatch`]: token-to-expert dispatch metadata
//! - [`lora`]: per-expert LoRA adapters
//!
//! ## Feature Flags
//!
//! - `fused-moe`: Enable fused CUDA kernels for optimized MoE execution
//! - `cuda-moe`: Enable CUDA-accelerated top-k softmax routing

mod ep_layer;
mod eplb;
pub mod eplb_execute;
mod expert_layer;
mod expert_map;
pub mod fused;
mod lora;
mod quantized_experts;
mod router;
mod token_dispatch;
pub mod topk_softmax;

pub use ep_layer::{EPMoEConfig, EPMoELayer};
pub use eplb::{EplbConfig, EplbState, ExpertLoadStats};
pub use eplb_execute::{rearrange_expert_weights_inplace, LayerExpertPlacement};
pub use expert_layer::{
    MoEExpert, MoEExpertConfig, MoELayer, MoELayerConfig, MoELayerWithShared,
    MoELayerWithSharedConfig,
};
pub use expert_map::{ExpertMap, ExpertPlacement};
pub use fused::{FusedMoEBlockConfig, FusedMoEConfig};
pub use lora::MoELoraWeights;
pub use quantized_experts::{QuantizedMoEExpert, QuantizedMoELayer, QuantizedMoELayerConfig};
pub use router::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};
pub use token_dispatch::{DispatchMetadata, TokenDispatcher};
pub use topk_softmax::{topk_softmax, TopKSoftmaxConfig};
