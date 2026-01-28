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
//! - **CUDA path**: Currently uses CPU fallback. The full CUDA kernel
//!   implementation is planned but not yet complete. When implemented,
//!   it will use:
//!   1. `moe_align_block_size` kernel for token grouping
//!   2. `fused_moe_gemm` kernel for batched expert computation
//!
//! The PTX files are included but the kernel launch code is not yet wired up.

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use super::config::FusedMoEBlockConfig;
use super::token_grouper::MoETokenGrouper;

#[cfg(feature = "cuda-kernels")]
use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Shape, Storage,
};

/// PTX for moe_align_block_size kernel (compiled from fused_moe_align.cu)
#[cfg(feature = "cuda-kernels")]
const MOE_ALIGN_PTX: &str = include_str!("../../kernels/fused_moe_align.ptx");

/// PTX for fused_moe_gemm kernel (compiled from fused_moe_gemm.cu)
#[cfg(feature = "cuda-kernels")]
const MOE_GEMM_PTX: &str = include_str!("../../kernels/fused_moe_gemm.ptx");

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
        Device::Cuda(_) => fused_moe_forward_cuda(
            hidden_states,
            w13_weights,
            w2_weights,
            routing_weights,
            expert_indices,
            config,
            num_experts,
            top_k,
        ),
        #[cfg(not(feature = "cuda-kernels"))]
        Device::Cuda(_) => {
            candle_core::bail!("CUDA kernels not compiled. Enable 'cuda-kernels' feature.")
        }
        _ => candle_core::bail!("Unsupported device for fused MoE"),
    }
}

/// CPU fallback implementation using optimized batching.
/// Still more efficient than naive per-token routing due to batched operations.
#[allow(clippy::too_many_arguments)]
fn fused_moe_forward_cpu(
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
    let dtype = hidden_states.dtype();
    let (num_tokens, hidden_size) = hidden_states.dims2()?;
    let intermediate_size = w2_weights.dim(2)?;

    // Step 1: Align tokens by expert (used for CUDA path)
    let grouper = MoETokenGrouper::from_config(config, num_experts);
    let _aligned = grouper.align_block_size(expert_indices)?;

    // Step 2: Process each expert's tokens as a batch
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

/// CUDA implementation of fused MoE forward pass.
#[cfg(feature = "cuda-kernels")]
fn fused_moe_forward_cuda(
    hidden_states: &Tensor,
    w13_weights: &Tensor,
    w2_weights: &Tensor,
    routing_weights: &Tensor,
    expert_indices: &Tensor,
    config: &FusedMoEBlockConfig,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    // For initial implementation, use the CPU path which is already optimized
    // with batching. Full CUDA kernel implementation would replace this.
    //
    // TODO: Implement full CUDA kernel path:
    // 1. Call moe_align_block_size kernel
    // 2. Call fused_moe_gemm kernel for w13 (gate+up)
    // 3. Apply activation
    // 4. Call fused_moe_gemm kernel for w2 (down)
    // 5. Unpermute and reduce

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
