//! CUDA kernel wrappers for fused MoE operations.
//!
//! This module provides Rust bindings to CUDA kernels for:
//! - Token alignment (moe_align_block_size)
//! - Fused expert GEMM with activation
//!
//! # Implementation
//!
//! - **CPU path**: Fully implemented with batched expert execution.
//!   Groups tokens by expert assignment and processes each expert's tokens
//!   as a batch, providing significant speedup over naive per-token routing.
//!
//! - **CUDA path**: Uses PTX kernels for fused gate+up+SiLU and down
//!   projection. Alignment is performed on CPU (lightweight) and integer
//!   buffers (sorted_token_ids, expert_ids, num_tokens_post_padded) are
//!   uploaded to GPU via raw cudarc allocations, bypassing candle's
//!   tensor system which lacks I32 support.
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
// I32 Workaround Utilities
// ============================================================================

/// Convert a slice of i64 values to i32, validating range.
#[cfg(feature = "cuda-kernels")]
fn i64_to_i32_vec(values: &[i64]) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                candle_core::Error::Msg(format!("i64_to_i32: value {v} out of i32 range"))
            })
        })
        .collect()
}

// ============================================================================
// CUDA Fused MoE Custom Op
// ============================================================================

#[cfg(feature = "cuda-kernels")]
struct FusedMoECudaOp {
    w13_weights: Tensor,
    w2_weights: Tensor,
    routing_weights: Tensor,
    expert_indices: Tensor,
    config: FusedMoEBlockConfig,
    num_experts: usize,
    top_k: usize,
}

#[cfg(feature = "cuda-kernels")]
impl candle_core::CustomOp1 for FusedMoECudaOp {
    fn name(&self) -> &'static str {
        "fused_moe_cuda"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("FusedMoECudaOp: CPU path should not be reached via CustomOp")
    }

    #[allow(clippy::too_many_lines)]
    fn cuda_fwd(
        &self,
        hs_storage: &candle_core::CudaStorage,
        hs_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda::CudaStorageSlice;
        use candle_core::Storage;
        let dev = &hs_storage.device;
        if hs_layout.start_offset() != 0 {
            candle_core::bail!("fused_moe_cuda: hidden_states must be contiguous from offset 0");
        }
        let hs_slice = match &hs_storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("fused_moe_cuda: hidden_states must be BF16"),
        };
        let hs_dims = hs_layout.dims();
        let num_tokens = hs_dims[0];
        let hidden_size = hs_dims[1];
        let intermediate_size = self.w2_weights.dim(2)?;
        let expert_indices_cpu = self.expert_indices.to_device(&Device::Cpu)?;
        let alignment = moe_align_block_size_cpu(
            &expert_indices_cpu,
            self.num_experts,
            self.config.block_size_m,
        )?;
        let num_tokens_post_padded = alignment.num_tokens_post_padded;
        let num_valid_tokens = alignment.num_valid_tokens;
        // Upload i32 alignment data via raw cudarc (candle lacks I32 tensors)
        let sorted_ids_cpu: Vec<i64> = alignment.sorted_token_ids.to_vec1()?;
        let sorted_ids_i32 = i64_to_i32_vec(&sorted_ids_cpu)?;
        let expert_ids_cpu: Vec<i64> = alignment.expert_ids.to_vec1()?;
        let expert_ids_i32 = i64_to_i32_vec(&expert_ids_cpu)?;
        let num_tokens_pp_i32: Vec<i32> = vec![num_tokens_post_padded as i32];
        let sorted_ids_dev = dev
            .htod_copy(sorted_ids_i32)
            .map_err(|e| candle_core::Error::Msg(format!("htod_copy sorted_ids: {e}")))?;
        let expert_ids_dev = dev
            .htod_copy(expert_ids_i32)
            .map_err(|e| candle_core::Error::Msg(format!("htod_copy expert_ids: {e}")))?;
        let num_tokens_pp_dev = dev.htod_copy(num_tokens_pp_i32).map_err(|e| {
            candle_core::Error::Msg(format!("htod_copy num_tokens_post_padded: {e}"))
        })?;
        let (w13_guard, w13_layout) = self.w13_weights.storage_and_layout();
        let w13_slice = match &*w13_guard {
            Storage::Cuda(cs) => {
                if w13_layout.start_offset() != 0 {
                    candle_core::bail!("fused_moe_cuda: w13 must be contiguous from offset 0");
                }
                match &cs.slice {
                    CudaStorageSlice::BF16(s) => s,
                    _ => candle_core::bail!("fused_moe_cuda: w13 must be BF16"),
                }
            }
            _ => candle_core::bail!("fused_moe_cuda: w13 must be on CUDA"),
        };
        let (w2_guard, w2_layout) = self.w2_weights.storage_and_layout();
        let w2_slice = match &*w2_guard {
            Storage::Cuda(cs) => {
                if w2_layout.start_offset() != 0 {
                    candle_core::bail!("fused_moe_cuda: w2 must be contiguous from offset 0");
                }
                match &cs.slice {
                    CudaStorageSlice::BF16(s) => s,
                    _ => candle_core::bail!("fused_moe_cuda: w2 must be BF16"),
                }
            }
            _ => candle_core::bail!("fused_moe_cuda: w2 must be on CUDA"),
        };
        let (rw_guard, rw_layout) = self.routing_weights.storage_and_layout();
        let rw_slice = match &*rw_guard {
            Storage::Cuda(cs) => {
                if rw_layout.start_offset() != 0 {
                    candle_core::bail!(
                        "fused_moe_cuda: routing_weights must be contiguous from offset 0"
                    );
                }
                match &cs.slice {
                    CudaStorageSlice::F32(s) => s,
                    _ => candle_core::bail!("fused_moe_cuda: routing_weights must be F32"),
                }
            }
            _ => candle_core::bail!("fused_moe_cuda: routing_weights must be on CUDA"),
        };
        let hidden_elem_count = num_tokens_post_padded * intermediate_size;
        let hidden_intermediate = dev
            .alloc_zeros::<half::bf16>(hidden_elem_count)
            .map_err(|e| candle_core::Error::Msg(format!("alloc hidden_intermediate: {e}")))?;
        let block_size_m = self.config.block_size_m;
        let threads_per_block: u32 = 256;
        let num_warps = threads_per_block / 32;
        let shared_mem_bytes = num_warps * 4;
        {
            let gate_up_func = dev.get_or_load_custom_func(
                "fused_moe_gate_up_silu_kernel",
                "fused_moe_gemm",
                MOE_GEMM_PTX,
            )?;
            let gate_up_cfg = LaunchConfig {
                grid_dim: (num_tokens_post_padded as u32, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes,
            };
            let hidden_size_i32 = hidden_size as i32;
            let intermediate_size_i32 = intermediate_size as i32;
            let num_valid_tokens_i32 = num_valid_tokens as i32;
            let top_k_i32 = self.top_k as i32;
            let block_size_m_i32 = block_size_m as i32;
            let mut builder = gate_up_func.builder();
            builder.arg(hs_slice);
            builder.arg(w13_slice);
            builder.arg(&hidden_intermediate);
            builder.arg(&sorted_ids_dev);
            builder.arg(&expert_ids_dev);
            builder.arg(&num_tokens_pp_dev);
            builder.arg(&hidden_size_i32);
            builder.arg(&intermediate_size_i32);
            builder.arg(&num_valid_tokens_i32);
            builder.arg(&top_k_i32);
            builder.arg(&block_size_m_i32);
            // SAFETY: All buffer pointers are valid CUDA device allocations
            // with sizes matching kernel expectations. PTX is compiled from
            // verified CUDA source (fused_moe_gemm.cu).
            unsafe { builder.launch(gate_up_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("fused_moe_gate_up_silu launch: {e}"))
            })?;
        }
        let output_elem_count = num_tokens * hidden_size;
        let output_slice = dev
            .alloc_zeros::<half::bf16>(output_elem_count)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;
        {
            let down_func = dev.get_or_load_custom_func(
                "fused_moe_down_reduce_kernel",
                "fused_moe_gemm",
                MOE_GEMM_PTX,
            )?;
            let down_cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes,
            };
            let hidden_size_i32 = hidden_size as i32;
            let intermediate_size_i32 = intermediate_size as i32;
            let num_valid_tokens_i32 = num_valid_tokens as i32;
            let num_tokens_i32 = num_tokens as i32;
            let top_k_i32 = self.top_k as i32;
            let block_size_m_i32 = block_size_m as i32;
            let mut builder = down_func.builder();
            builder.arg(&hidden_intermediate);
            builder.arg(w2_slice);
            builder.arg(&output_slice);
            builder.arg(rw_slice);
            builder.arg(&sorted_ids_dev);
            builder.arg(&expert_ids_dev);
            builder.arg(&num_tokens_pp_dev);
            builder.arg(&hidden_size_i32);
            builder.arg(&intermediate_size_i32);
            builder.arg(&num_valid_tokens_i32);
            builder.arg(&num_tokens_i32);
            builder.arg(&top_k_i32);
            builder.arg(&block_size_m_i32);
            // SAFETY: All buffer pointers are valid CUDA device allocations.
            // hidden_intermediate was written by the preceding kernel launch.
            unsafe { builder.launch(down_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("fused_moe_down_reduce launch: {e}"))
            })?;
        }
        drop(w13_guard);
        drop(w2_guard);
        drop(rw_guard);
        let output_storage = candle_core::CudaStorage {
            slice: CudaStorageSlice::BF16(output_slice),
            device: dev.clone(),
        };
        let output_shape = candle_core::Shape::from_dims(&[num_tokens, hidden_size]);
        Ok((output_storage, output_shape))
    }
}

#[cfg(feature = "cuda-kernels")]
#[allow(clippy::too_many_arguments)]
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
    let dtype = hidden_states.dtype();
    if dtype != DType::BF16 {
        return fused_moe_forward_cpu(
            hidden_states,
            w13_weights,
            w2_weights,
            routing_weights,
            expert_indices,
            &FusedMoEBlockConfig::default(),
            num_experts,
            top_k,
        );
    }
    let hidden_states = hidden_states.contiguous()?;
    let w13_weights = w13_weights.contiguous()?;
    let w2_weights = w2_weights.contiguous()?;
    let routing_weights = routing_weights.contiguous()?.to_dtype(DType::F32)?;
    let op = FusedMoECudaOp {
        w13_weights,
        w2_weights,
        routing_weights,
        expert_indices: expert_indices.clone(),
        config: *config,
        num_experts,
        top_k,
    };
    hidden_states.apply_op1_no_bwd(&op)
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

    #[test]
    fn test_i64_to_i32_roundtrip() {
        let values: Vec<i64> = vec![0, 1, -1, 42, i32::MAX as i64, i32::MIN as i64];
        #[cfg(feature = "cuda-kernels")]
        {
            let result = i64_to_i32_vec(&values).unwrap();
            assert_eq!(result, vec![0i32, 1, -1, 42, i32::MAX, i32::MIN]);
        }
        #[cfg(not(feature = "cuda-kernels"))]
        {
            for v in &values {
                assert!(i32::try_from(*v).is_ok());
            }
        }
    }

    #[test]
    fn test_i64_to_i32_overflow() {
        #[cfg(feature = "cuda-kernels")]
        {
            let values: Vec<i64> = vec![i64::MAX];
            let result = i64_to_i32_vec(&values);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_fused_moe_output_nonzero() {
        let device = Device::Cpu;

        let num_tokens = 2;
        let hidden_size = 4;
        let intermediate_size = 8;
        let num_experts = 2;
        let top_k = 1;

        let hidden_states = Tensor::ones((num_tokens, hidden_size), DType::F32, &device).unwrap();

        let w13_weights = Tensor::full(
            0.1f32,
            (num_experts, 2 * intermediate_size, hidden_size),
            &device,
        )
        .unwrap();

        let w2_weights = Tensor::full(
            0.1f32,
            (num_experts, hidden_size, intermediate_size),
            &device,
        )
        .unwrap();

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

        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        let max_abs = output_vec.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-6,
            "Output should be non-zero, got max abs: {max_abs}"
        );
    }

    #[test]
    fn test_fused_moe_multiple_topk_accumulation() {
        let device = Device::Cpu;

        let num_tokens = 1;
        let hidden_size = 4;
        let intermediate_size = 8;
        let num_experts = 2;
        let top_k = 2;

        let hidden_states = Tensor::ones((num_tokens, hidden_size), DType::F32, &device).unwrap();

        let w13_weights = Tensor::full(
            0.1f32,
            (num_experts, 2 * intermediate_size, hidden_size),
            &device,
        )
        .unwrap();

        let w2_weights = Tensor::full(
            0.1f32,
            (num_experts, hidden_size, intermediate_size),
            &device,
        )
        .unwrap();

        let routing_weights = Tensor::new(&[[0.5f32, 0.5]], &device).unwrap();
        let expert_indices = Tensor::new(&[[0u32, 1]], &device).unwrap();

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

        // With symmetric experts, 2x0.5 should equal 1x1.0
        let routing_weights_single = Tensor::new(&[[1.0f32]], &device).unwrap();
        let expert_indices_single = Tensor::new(&[[0u32]], &device).unwrap();

        let output_single = fused_moe_forward(
            &hidden_states,
            &w13_weights,
            &w2_weights,
            &routing_weights_single,
            &expert_indices_single,
            &config,
            num_experts,
            1,
        )
        .unwrap();

        let out_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        let single_vec: Vec<f32> = output_single.flatten_all().unwrap().to_vec1().unwrap();

        for (a, b) in out_vec.iter().zip(single_vec.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "top_k=2 with 0.5 weights should match top_k=1: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_fused_moe_varying_token_counts() {
        let device = Device::Cpu;

        for num_tokens in [1, 2, 3, 7, 8, 15, 16, 33] {
            let hidden_size = 8;
            let intermediate_size = 16;
            let num_experts = 4;
            let top_k = 2;

            let hidden_states =
                Tensor::randn(0f32, 1.0, (num_tokens, hidden_size), &device).unwrap();

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

            let mut rw_data = Vec::with_capacity(num_tokens * top_k);
            let mut ei_data = Vec::with_capacity(num_tokens * top_k);
            for t in 0..num_tokens {
                for k in 0..top_k {
                    rw_data.push(1.0f32 / top_k as f32);
                    ei_data.push(((t + k) % num_experts) as u32);
                }
            }

            let routing_weights = Tensor::from_vec(rw_data, (num_tokens, top_k), &device).unwrap();
            let expert_indices = Tensor::from_vec(ei_data, (num_tokens, top_k), &device).unwrap();

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

            assert_eq!(
                output.dims(),
                &[num_tokens, hidden_size],
                "Shape mismatch for num_tokens={num_tokens}"
            );
        }
    }
}
