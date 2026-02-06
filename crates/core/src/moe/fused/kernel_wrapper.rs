//! CUDA kernel wrappers for fused MoE operations.
//!
//! This module provides Rust bindings to CUDA kernels for:
//! - Token alignment (moe_align_block_size)
//! - Fused expert GEMM with activation
//!
//! # Implementation Status
//!
//! - **CPU path**: Fully implemented with batched expert execution.
//!   Groups tokens by expert assignment and processes each expert's tokens
//!   as a batch, providing significant speedup over naive per-token routing.
//!
//! - **CUDA path**: PTX kernels are compiled and ready, but tensor creation
//!   from raw GPU allocations requires better I32 support in candle.
//!   Currently falls back to optimized CPU path.
//!
//! # Available CUDA Kernels (fused_moe_align.ptx)
//!
//! - `moe_align_block_size_kernel` - main alignment for large batches
//! - `moe_align_block_size_small_kernel` - optimized for small batches
//! - `moe_sort_tokens_kernel` - second pass token sorting
//! - `moe_sum_kernel` / `moe_sum_bf16_kernel` - reduction kernels
//!
//! # Available CUDA Kernels (fused_moe_gemm.ptx)
//!
//! - `fused_moe_gemm_64_64_32_weighted` - tiled GEMM with routing weights
//! - `fused_moe_gemm_64_64_32_unweighted` - tiled GEMM without weights
//! - `fused_moe_gemm_simple_weighted` - simple GEMM for small batches
//! - `fused_moe_gemm_simple_unweighted` - simple GEMM without weights
//! - `fused_moe_gate_up_silu_kernel` - fused gate+up with SiLU activation
//! - `fused_moe_down_reduce_kernel` - down projection with reduction

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use super::config::FusedMoEBlockConfig;
use super::token_grouper::MoETokenGrouper;

/// PTX for moe_align_block_size kernel (compiled from fused_moe_align.cu)
#[cfg(feature = "cuda-kernels")]
#[allow(dead_code)]
const MOE_ALIGN_PTX: &str = include_str!("../../../kernels/fused_moe_align.ptx");

/// PTX for fused_moe_gemm kernel (compiled from fused_moe_gemm.cu)
#[cfg(feature = "cuda-kernels")]
#[allow(dead_code)]
const MOE_GEMM_PTX: &str = include_str!("../../../kernels/fused_moe_gemm.ptx");

// ============================================================================
// MoE Alignment Output
// ============================================================================

/// Output of MoE token alignment operation.
#[derive(Debug)]
pub struct MoeAlignOutput {
    /// Sorted token indices, shape [num_tokens_padded].
    /// Maps output position -> original (token_idx * top_k + k).
    pub sorted_token_ids: Tensor,
    /// Expert ID for each processing block, shape [num_blocks].
    pub expert_ids: Tensor,
    /// Total number of tokens after padding to block boundaries.
    pub num_tokens_post_padded: usize,
    /// Number of valid tokens (before padding).
    pub num_valid_tokens: usize,
}

// ============================================================================
// Token Alignment
// ============================================================================

/// Align tokens by expert assignment for batched GEMM execution.
///
/// Groups tokens by their assigned experts and pads each group to block boundaries.
/// This is the first step in the fused MoE pipeline.
///
/// # Arguments
/// * `topk_ids` - Expert indices [num_tokens, top_k] as U32
/// * `num_experts` - Total number of experts
/// * `block_size` - Block size for GEMM alignment
///
/// # Returns
/// * `MoeAlignOutput` with sorted indices and expert assignments
pub fn moe_align_block_size(
    topk_ids: &Tensor,
    num_experts: usize,
    block_size: usize,
) -> Result<MoeAlignOutput> {
    // CPU implementation handles all cases for now.
    // CUDA kernels are compiled and available for future use.
    moe_align_block_size_cpu(topk_ids, num_experts, block_size)
}

/// CPU implementation of token alignment.
fn moe_align_block_size_cpu(
    topk_ids: &Tensor,
    num_experts: usize,
    block_size: usize,
) -> Result<MoeAlignOutput> {
    let grouper = MoETokenGrouper::new(block_size, num_experts);
    let aligned = grouper.align_block_size(topk_ids)?;

    let num_tokens_post_padded: Vec<i64> = aligned.num_tokens_post_padded.to_vec1()?;
    let num_tokens_post_padded = num_tokens_post_padded[0];

    Ok(MoeAlignOutput {
        sorted_token_ids: aligned.sorted_token_ids,
        expert_ids: aligned.expert_ids,
        num_tokens_post_padded: num_tokens_post_padded as usize,
        num_valid_tokens: aligned.num_valid_tokens,
    })
}

// ============================================================================
// Fused MoE Forward Pass
// ============================================================================

/// Fused MoE forward pass that combines:
/// 1. Token alignment/grouping by expert
/// 2. Batched expert GEMM (gate + up projections)
/// 3. SiLU activation and element-wise multiply
/// 4. Batched down projection
/// 5. Unpermutation and weighted reduction
///
/// # Arguments
/// * `hidden_states` - Input tensor [num_tokens, hidden_size]
/// * `w13_weights` - Stacked gate+up weights [num_experts, 2*intermediate_size, hidden_size]
/// * `w2_weights` - Down projection weights [num_experts, hidden_size, intermediate_size]
/// * `routing_weights` - Router weights [num_tokens, top_k]
/// * `expert_indices` - Router selections [num_tokens, top_k]
/// * `config` - Block configuration
///
/// # Returns
/// * Output tensor [num_tokens, hidden_size]
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_forward(
    hidden_states: &Tensor,
    w13_weights: &Tensor,
    w2_weights: &Tensor,
    routing_weights: &Tensor,
    expert_indices: &Tensor,
    config: &FusedMoEBlockConfig,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    let device = hidden_states.device();

    match device {
        Device::Cpu => fused_moe_forward_cpu(
            hidden_states,
            w13_weights,
            w2_weights,
            routing_weights,
            expert_indices,
            config,
            num_experts,
            top_k,
        ),
        #[cfg(feature = "cuda-kernels")]
        Device::Cuda(_) => {
            // TODO: Full CUDA kernel implementation
            //
            // Current limitation: candle doesn't have I32 tensor support,
            // which is needed for the alignment kernel outputs (sorted_token_ids,
            // expert_ids are i32 in CUDA).
            //
            // The CUDA implementation would:
            // 1. Call moe_align_block_size_kernel to group tokens
            // 2. Launch fused_moe_gate_up_silu_kernel for SiLU(Wg @ x) * Wu @ x
            // 3. Launch fused_moe_gemm_simple_weighted for down projection
            // 4. Launch moe_sum_kernel for reduction across top_k
            //
            // PTX kernels are compiled and available in MOE_ALIGN_PTX and MOE_GEMM_PTX.
            // For now, use the optimized CPU path.
            fused_moe_forward_cpu(
                hidden_states,
                w13_weights,
                w2_weights,
                routing_weights,
                expert_indices,
                config,
                num_experts,
                top_k,
            )
        }
        #[cfg(not(feature = "cuda-kernels"))]
        Device::Cuda(_) => {
            candle_core::bail!("CUDA kernels not compiled. Enable 'cuda-kernels' feature.")
        }
        _ => candle_core::bail!("Unsupported device for fused MoE"),
    }
}

/// CPU fallback implementation using optimized batching.
/// Groups tokens by expert and processes each expert's batch together.
#[allow(clippy::too_many_arguments)]
fn fused_moe_forward_cpu(
    hidden_states: &Tensor,
    w13_weights: &Tensor,
    w2_weights: &Tensor,
    routing_weights: &Tensor,
    expert_indices: &Tensor,
    _config: &FusedMoEBlockConfig,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    let device = hidden_states.device();
    let dtype = hidden_states.dtype();
    let (num_tokens, hidden_size) = hidden_states.dims2()?;
    let intermediate_size = w2_weights.dim(2)?;

    // Get flat vectors for indexing
    let expert_indices_vec: Vec<u32> = expert_indices.flatten_all()?.to_vec1()?;
    let routing_weights_vec: Vec<f32> = routing_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1()?;

    // Initialize output
    let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

    // Group token indices by expert
    let mut expert_tokens: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); num_experts];

    for token_idx in 0..num_tokens {
        for k in 0..top_k {
            let flat_idx = token_idx * top_k + k;
            let expert_id = expert_indices_vec[flat_idx] as usize;
            let weight = routing_weights_vec[flat_idx];
            if expert_id < num_experts {
                expert_tokens[expert_id].push((token_idx, k, weight));
            }
        }
    }

    // Process each expert's batch
    for (expert_id, tokens) in expert_tokens.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }

        // Gather input tokens for this expert
        let token_indices: Vec<usize> = tokens.iter().map(|(idx, _, _)| *idx).collect();
        let weights: Vec<f32> = tokens.iter().map(|(_, _, w)| *w).collect();

        // Batch the input tokens
        let batch_size = token_indices.len();
        let mut input_rows = Vec::with_capacity(batch_size);
        for &idx in &token_indices {
            input_rows.push(hidden_states.i(idx)?.unsqueeze(0)?);
        }
        let batch_input = Tensor::cat(&input_rows, 0)?;

        // Get expert weights
        let expert_w13 = w13_weights.i(expert_id)?; // [2*intermediate, hidden]
        let expert_w2 = w2_weights.i(expert_id)?; // [hidden, intermediate]

        // Split w13 into gate and up projections
        let gate_proj = expert_w13.narrow(0, 0, intermediate_size)?;
        let up_proj = expert_w13.narrow(0, intermediate_size, intermediate_size)?;

        // Forward pass: SiLU(gate_proj(x)) * up_proj(x)
        let gate = batch_input.matmul(&gate_proj.t()?)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = batch_input.matmul(&up_proj.t()?)?;
        let hidden = gate.mul(&up)?;

        // Down projection
        let expert_output = hidden.matmul(&expert_w2.t()?)?;

        // Scatter weighted results back to output
        for (batch_idx, (&token_idx, &weight)) in
            token_indices.iter().zip(weights.iter()).enumerate()
        {
            let row_output = expert_output.i(batch_idx)?;
            let weight_tensor = Tensor::new(&[weight], device)?.to_dtype(dtype)?;
            let weighted = row_output.broadcast_mul(&weight_tensor)?;

            // Add to output
            let current = output.i(token_idx)?;
            let updated = current.add(&weighted)?;
            output = update_row(&output, token_idx, &updated)?;
        }
    }

    Ok(output)
}

/// Helper to update a single row in a 2D tensor.
fn update_row(tensor: &Tensor, row_idx: usize, values: &Tensor) -> Result<Tensor> {
    let (num_rows, _) = tensor.dims2()?;
    let mut rows = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        if i == row_idx {
            rows.push(values.unsqueeze(0)?);
        } else {
            rows.push(tensor.i(i)?.unsqueeze(0)?);
        }
    }
    Tensor::cat(&rows, 0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_align_block_size_cpu() {
        let device = Device::Cpu;
        let topk_ids = Tensor::new(&[[0u32, 1], [1, 2], [2, 3], [0, 3]], &device).unwrap();

        let aligned = moe_align_block_size(&topk_ids, 4, 4).unwrap();

        // 8 valid tokens (4 tokens * 2 top_k)
        assert_eq!(aligned.num_valid_tokens, 8);
        // Should have padded to block boundaries
        assert!(aligned.num_tokens_post_padded >= 8);
    }

    #[test]
    fn test_fused_moe_forward_cpu() {
        let device = Device::Cpu;

        let num_tokens = 4;
        let hidden_size = 16;
        let intermediate_size = 32;
        let num_experts = 4;
        let top_k = 2;

        // Create random inputs
        let hidden_states = Tensor::randn(0f32, 1.0, (num_tokens, hidden_size), &device).unwrap();

        // w13: [num_experts, 2*intermediate, hidden]
        let w13_weights = Tensor::randn(
            0f32,
            0.1,
            (num_experts, 2 * intermediate_size, hidden_size),
            &device,
        )
        .unwrap();

        // w2: [num_experts, hidden, intermediate]
        let w2_weights = Tensor::randn(
            0f32,
            0.1,
            (num_experts, hidden_size, intermediate_size),
            &device,
        )
        .unwrap();

        // Routing weights (normalized)
        let routing_weights = Tensor::new(
            &[[0.6f32, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]],
            &device,
        )
        .unwrap();

        // Expert indices
        let expert_indices = Tensor::new(&[[0u32, 1], [1, 2], [2, 3], [0, 3]], &device).unwrap();

        let config = FusedMoEBlockConfig::default();

        let output = fused_moe_forward(
            &hidden_states,
            &w13_weights,
            &w2_weights,
            &routing_weights,
            &expert_indices,
            &config,
            num_experts,
            top_k,
        )
        .unwrap();

        // Check output shape
        assert_eq!(output.dims(), &[num_tokens, hidden_size]);
    }

    #[test]
    fn test_fused_moe_single_expert() {
        let device = Device::Cpu;

        let num_tokens = 2;
        let hidden_size = 8;
        let intermediate_size = 16;
        let num_experts = 2;
        let top_k = 1;

        let hidden_states = Tensor::randn(0f32, 1.0, (num_tokens, hidden_size), &device).unwrap();

        let w13_weights = Tensor::randn(
            0f32,
            0.1,
            (num_experts, 2 * intermediate_size, hidden_size),
            &device,
        )
        .unwrap();

        let w2_weights = Tensor::randn(
            0f32,
            0.1,
            (num_experts, hidden_size, intermediate_size),
            &device,
        )
        .unwrap();

        // Each token goes to exactly one expert
        let routing_weights = Tensor::new(&[[1.0f32], [1.0]], &device).unwrap();
        let expert_indices = Tensor::new(&[[0u32], [1]], &device).unwrap();

        let config = FusedMoEBlockConfig::default();

        let output = fused_moe_forward(
            &hidden_states,
            &w13_weights,
            &w2_weights,
            &routing_weights,
            &expert_indices,
            &config,
            num_experts,
            top_k,
        )
        .unwrap();

        assert_eq!(output.dims(), &[num_tokens, hidden_size]);
    }
}
