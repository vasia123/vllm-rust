//! Fused MoE kernels for optimized expert execution.
//!
//! This module implements the permute-compute-unpermute pattern from vLLM:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      FusedMoELayer (Rust)                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐   ┌────────────────┐   ┌───────────────────┐  │
//! │  │   Router    │──▶│  Permute+Group │──▶│  Fused GEMM/Act   │  │
//! │  │ (existing)  │   │  (CUDA)        │   │  (CUDA)           │  │
//! │  └─────────────┘   └────────────────┘   └─────────┬─────────┘  │
//! │                                                    │            │
//! │  ┌─────────────────────────────────────────────────▼──────────┐ │
//! │  │                Unpermute + Reduce (CUDA)                   │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`config`]: Kernel configuration and tuning parameters
//! - [`token_grouper`]: Token alignment by expert for batched execution
//! - [`kernel_wrapper`]: CUDA kernel wrappers and CPU fallbacks
//!
//! ## Algorithm
//!
//! 1. **Token-to-Expert Alignment:** Sort tokens by expert, pad to block boundaries
//! 2. **Fused GEMM:** Single kernel processes all tokens grouped by expert
//! 3. **Output Reduction:** Apply router weights, sum across top-k, scatter back
//!
//! ## Performance
//!
//! Compared to naive per-token routing:
//! - **Naive:** O(tokens × top_k × expert_forward) sequential operations
//! - **Fused:** O(1 kernel launch) with batched expert execution
//!
//! Expected speedup: 5-10x for batch sizes > 64

mod config;
mod kernel_wrapper;
mod token_grouper;

pub use config::{FusedMoEBlockConfig, FusedMoEConfig};
pub use kernel_wrapper::{fused_moe_forward, moe_align_block_size, MoeAlignOutput};
pub use token_grouper::{AlignedTokens, MoETokenGrouper};
