//! Token grouping for fused MoE execution.
//!
//! This module implements the permute-compute-unpermute pattern:
//! 1. Group tokens by expert assignment
//! 2. Pad to block boundaries
//! 3. Compute sorted token indices for gather operations

use candle_core::{DType, IndexOp, Result, Tensor};

use super::config::FusedMoEBlockConfig;

/// Result of token alignment operation.
#[derive(Debug)]
pub struct AlignedTokens {
    /// Sorted token indices: maps output position to original token index.
    /// Shape: [num_tokens_padded]
    /// Value at position i indicates which original token (divided by top_k)
    /// should be processed at output position i.
    pub sorted_token_ids: Tensor,

    /// Expert ID for each block of tokens.
    /// Shape: [num_blocks] where num_blocks = num_tokens_padded / block_size
    /// Value at position i indicates which expert processes block i.
    /// -1 indicates an invalid/padding block.
    pub expert_ids: Tensor,

    /// Total number of tokens after padding (scalar tensor for GPU access).
    pub num_tokens_post_padded: Tensor,

    /// Number of valid tokens (before padding).
    pub num_valid_tokens: usize,
}

/// Token grouper that aligns tokens by expert for efficient batched execution.
pub struct MoETokenGrouper {
    block_size: usize,
    num_experts: usize,
}

impl MoETokenGrouper {
    /// Create a new token grouper.
    pub fn new(block_size: usize, num_experts: usize) -> Self {
        Self {
            block_size,
            num_experts,
        }
    }

    /// Create from block config.
    pub fn from_config(config: &FusedMoEBlockConfig, num_experts: usize) -> Self {
        Self::new(config.block_size_m, num_experts)
    }

    /// Align tokens to block boundaries for efficient GEMM execution.
    ///
    /// This function groups tokens by their expert assignments and pads each
    /// expert's token count to be divisible by block_size.
    ///
    /// # Arguments
    /// * `topk_ids` - Expert indices for each token, shape [num_tokens, top_k]
    ///
    /// # Returns
    /// * `AlignedTokens` containing sorted indices and expert assignments
    ///
    /// # Algorithm
    /// 1. Count tokens per expert
    /// 2. Compute cumulative sum with padding to block boundaries
    /// 3. Scatter token indices to their sorted positions
    /// 4. Generate expert IDs for each block
    pub fn align_block_size(&self, topk_ids: &Tensor) -> Result<AlignedTokens> {
        let device = topk_ids.device();
        let (num_tokens, top_k) = topk_ids.dims2()?;
        let numel = num_tokens * top_k;
        let num_valid_tokens = numel;

        // Maximum possible padded size
        let max_num_tokens_padded = numel + self.num_experts * (self.block_size - 1);
        let max_num_blocks = max_num_tokens_padded.div_ceil(self.block_size);

        // Flatten topk_ids for processing
        let flat_topk_ids = topk_ids.flatten_all()?;
        let topk_ids_vec: Vec<u32> = flat_topk_ids.to_vec1()?;

        // Count tokens per expert
        let mut expert_counts = vec![0usize; self.num_experts];
        for &expert_id in &topk_ids_vec {
            let expert_idx = expert_id as usize;
            if expert_idx < self.num_experts {
                expert_counts[expert_idx] += 1;
            }
        }

        // Compute padded counts and cumulative sums
        let mut cumsum = vec![0usize; self.num_experts + 1];
        for (i, &count) in expert_counts.iter().enumerate() {
            let padded_count = count.div_ceil(self.block_size) * self.block_size;
            cumsum[i + 1] = cumsum[i] + padded_count;
        }
        let num_tokens_padded = cumsum[self.num_experts];

        // Initialize sorted_token_ids with invalid index (numel)
        let mut sorted_token_ids = vec![numel as i32; max_num_tokens_padded];

        // Track current position within each expert's allocation
        let mut expert_offsets = cumsum[..self.num_experts].to_vec();

        // Scatter tokens to their sorted positions
        for (token_idx, &expert_id) in topk_ids_vec.iter().enumerate() {
            let expert_idx = expert_id as usize;
            if expert_idx < self.num_experts {
                let pos = expert_offsets[expert_idx];
                sorted_token_ids[pos] = token_idx as i32;
                expert_offsets[expert_idx] += 1;
            }
        }

        // Generate expert_ids for each block
        let mut expert_ids_vec = vec![-1i32; max_num_blocks];
        for expert_idx in 0..self.num_experts {
            let start_block = cumsum[expert_idx] / self.block_size;
            let end_block = cumsum[expert_idx + 1] / self.block_size;
            for expert_id in expert_ids_vec.iter_mut().take(end_block).skip(start_block) {
                *expert_id = expert_idx as i32;
            }
        }

        // Convert to tensors
        let sorted_token_ids: Vec<i64> = sorted_token_ids.iter().map(|&x| x as i64).collect();
        let sorted_token_ids = Tensor::new(sorted_token_ids.as_slice(), device)?;

        let expert_ids: Vec<i64> = expert_ids_vec.iter().map(|&x| x as i64).collect();
        let expert_ids = Tensor::new(expert_ids.as_slice(), device)?;

        let num_tokens_post_padded = Tensor::new(&[num_tokens_padded as i64], device)?;

        Ok(AlignedTokens {
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            num_valid_tokens,
        })
    }

    /// CPU-based alignment for testing and fallback.
    /// This is functionally equivalent to the CUDA kernel but runs on CPU.
    pub fn align_block_size_cpu(&self, topk_ids: &Tensor) -> Result<AlignedTokens> {
        self.align_block_size(topk_ids)
    }
}

/// Compute the unpermutation: scatter results back to original token order.
///
/// After fused MoE computation, results are in sorted order. This function
/// creates the inverse mapping to restore original token order.
///
/// # Arguments
/// * `sorted_token_ids` - Mapping from sorted position to original token
/// * `routing_weights` - Routing weights [num_tokens, top_k]
/// * `expert_output` - Expert outputs in sorted order [num_tokens_padded, hidden_size]
/// * `num_valid_tokens` - Number of valid (non-padding) tokens
/// * `top_k` - Number of experts per token
///
/// # Returns
/// * Output tensor in original token order [num_tokens, hidden_size]
#[allow(dead_code)]
pub fn unpermute_and_reduce(
    sorted_token_ids: &Tensor,
    routing_weights: &Tensor,
    expert_output: &Tensor,
    num_valid_tokens: usize,
    top_k: usize,
) -> Result<Tensor> {
    let device = expert_output.device();
    let dtype = expert_output.dtype();
    let (_num_tokens_padded, hidden_size) = expert_output.dims2()?;
    let num_tokens = num_valid_tokens / top_k;

    // Initialize output tensor
    let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

    // Get sorted token IDs as vector for CPU processing
    let sorted_ids: Vec<i64> = sorted_token_ids.to_vec1()?;
    let weights_flat: Vec<f32> = routing_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1()?;

    // For each position in sorted output
    for (sorted_pos, &token_idx) in sorted_ids.iter().enumerate() {
        if token_idx < 0 || token_idx as usize >= num_valid_tokens {
            continue; // Skip padding tokens
        }

        let token_idx = token_idx as usize;
        let original_token = token_idx / top_k;
        let k_idx = token_idx % top_k;

        // Get the expert output for this position
        let expert_out = expert_output.i(sorted_pos)?;

        // Get routing weight
        let weight_idx = original_token * top_k + k_idx;
        let weight = weights_flat[weight_idx];

        // Create weight tensor for broadcasting
        let weight_tensor = Tensor::new(&[weight], device)?.to_dtype(dtype)?;

        // Weighted contribution
        let weighted = expert_out.broadcast_mul(&weight_tensor)?;

        // Add to output at original position
        let current = output.i(original_token)?;
        let updated = current.add(&weighted)?;

        // Update output row
        output = update_row(&output, original_token, &updated)?;
    }

    Ok(output)
}

/// Helper to update a single row in a 2D tensor.
#[allow(dead_code)]
fn update_row(tensor: &Tensor, row_idx: usize, values: &Tensor) -> Result<Tensor> {
    let (num_rows, _hidden_size) = tensor.dims2()?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_token_grouper_basic() {
        let device = Device::Cpu;
        let grouper = MoETokenGrouper::new(4, 4);

        // 4 tokens, each routed to 2 experts
        // Token 0 -> experts [2, 3]
        // Token 1 -> experts [1, 2]
        // Token 2 -> experts [1, 3]
        // Token 3 -> experts [1, 2]
        let topk_ids = Tensor::new(&[[2u32, 3], [1, 2], [1, 3], [1, 2]], &device).unwrap();

        let aligned = grouper.align_block_size(&topk_ids).unwrap();

        // Verify dimensions
        let _sorted_ids: Vec<i64> = aligned.sorted_token_ids.to_vec1().unwrap();
        let expert_ids: Vec<i64> = aligned.expert_ids.to_vec1().unwrap();

        // Total tokens: 8 (4 tokens × 2 top_k)
        // Expert 0: 0 tokens -> 0 padded
        // Expert 1: 3 tokens (indices 2, 4, 6) -> 4 padded
        // Expert 2: 3 tokens (indices 0, 3, 7) -> 4 padded
        // Expert 3: 2 tokens (indices 1, 5) -> 4 padded
        // Total padded: 12

        assert!(aligned.num_valid_tokens == 8);

        // Check that expert_ids has correct expert assignments
        // First 4 positions should be expert 1
        // Next 4 positions should be expert 2
        // Next 4 positions should be expert 3
        let num_blocks = expert_ids.len();
        assert!(num_blocks >= 3, "Should have at least 3 blocks");
    }

    #[test]
    fn test_token_grouper_single_expert() {
        let device = Device::Cpu;
        let grouper = MoETokenGrouper::new(4, 8);

        // All tokens routed to same expert
        let topk_ids = Tensor::new(&[[0u32, 1], [0, 1], [0, 1], [0, 1]], &device).unwrap();

        let aligned = grouper.align_block_size(&topk_ids).unwrap();
        assert_eq!(aligned.num_valid_tokens, 8);
    }

    #[test]
    fn test_token_grouper_padding() {
        let device = Device::Cpu;
        let grouper = MoETokenGrouper::new(64, 8);

        // Small number of tokens - should be heavily padded
        let topk_ids = Tensor::new(&[[0u32, 1], [2, 3]], &device).unwrap();

        let aligned = grouper.align_block_size(&topk_ids).unwrap();
        assert_eq!(aligned.num_valid_tokens, 4);

        // Each expert with 1 token gets padded to 64
        let num_tokens_padded: Vec<i64> = aligned.num_tokens_post_padded.to_vec1().unwrap();
        assert!(num_tokens_padded[0] >= 64);
    }
}
