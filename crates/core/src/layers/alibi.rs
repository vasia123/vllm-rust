//! ALiBi (Attention with Linear Biases) positional encoding.
//!
//! ALiBi is an alternative to sinusoidal or rotary positional encodings.
//! Instead of adding position information to embeddings, ALiBi adds a linear
//! bias to attention scores based on the distance between query and key positions.
//!
//! # Mathematical Formula
//!
//! For a query at position i attending to a key at position j:
//!
//! ```text
//! attention_score[h, i, j] = (Q[h,i] @ K[h,j].T) / sqrt(d) + alibi_bias[h, i, j]
//! ```
//!
//! where:
//!
//! ```text
//! alibi_bias[h, i, j] = m[h] * (j - i)
//! ```
//!
//! Note: For causal attention, j <= i, so (j - i) <= 0, making the bias non-positive.
//! This means tokens further in the past receive more negative bias (less attention).
//!
//! # Slope Computation
//!
//! The slopes m[h] are computed as geometric sequences:
//!
//! For n heads where n is a power of 2:
//! ```text
//! base = 2^(-(2^(-(log2(n) - 3))))
//! m[h] = base^(h+1)  for h in 0..n
//! ```
//!
//! For n heads where n is NOT a power of 2:
//! - Compute slopes for closest_power_of_2 heads using above formula
//! - Compute extra slopes for remaining heads with a tighter base
//!
//! # Usage
//!
//! ALiBi is used by:
//! - BLOOM (BigScience)
//! - MPT (MosaicML)
//! - Falcon (with alibi=true config)
//!
//! # References
//!
//! - Paper: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization"
//!   <https://arxiv.org/abs/2108.12409>
//! - vLLM implementation: reference/vllm/vllm/model_executor/models/bloom.py

use candle_core::{DType, Device, Result, Tensor};

/// Pre-computed ALiBi attention biases for efficient inference.
///
/// Stores per-head slopes that are used to compute position-dependent
/// attention biases at runtime.
#[derive(Debug, Clone)]
pub struct AlibiAttentionBias {
    /// Per-head slopes, shape [num_heads]
    slopes: Tensor,
    /// Number of attention heads
    num_heads: usize,
}

impl AlibiAttentionBias {
    /// Create ALiBi attention bias for the given number of heads.
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Total number of attention heads
    /// * `dtype` - Data type for the slopes tensor
    /// * `device` - Device to place the slopes tensor on
    ///
    /// # Returns
    ///
    /// `AlibiAttentionBias` with pre-computed slopes for each head.
    pub fn new(num_heads: usize, dtype: DType, device: &Device) -> Result<Self> {
        let slopes_vec = compute_alibi_slopes(num_heads);
        let slopes = Tensor::from_vec(slopes_vec, (num_heads,), device)?.to_dtype(dtype)?;
        Ok(Self { num_heads, slopes })
    }

    /// Create ALiBi attention bias for a subset of heads (tensor parallel).
    ///
    /// Used when heads are distributed across multiple devices.
    ///
    /// # Arguments
    ///
    /// * `total_num_heads` - Total number of heads across all devices
    /// * `head_start` - Starting head index for this device
    /// * `head_end` - Ending head index (exclusive) for this device
    /// * `dtype` - Data type for the slopes tensor
    /// * `device` - Device to place the slopes tensor on
    pub fn new_partial(
        total_num_heads: usize,
        head_start: usize,
        head_end: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let all_slopes = compute_alibi_slopes(total_num_heads);
        let slopes_vec: Vec<f32> = all_slopes[head_start..head_end].to_vec();
        let num_heads = head_end - head_start;
        let slopes = Tensor::from_vec(slopes_vec, (num_heads,), device)?.to_dtype(dtype)?;
        Ok(Self { num_heads, slopes })
    }

    /// Get the slopes tensor.
    pub fn slopes(&self) -> &Tensor {
        &self.slopes
    }

    /// Get the number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Build the full ALiBi bias matrix for a sequence.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Query sequence length
    /// * `kv_len` - Key/value sequence length (can be larger for decode with cache)
    ///
    /// # Returns
    ///
    /// Tensor of shape [1, num_heads, seq_len, kv_len] containing the ALiBi biases.
    ///
    /// The bias at position [0, h, i, j] = slopes[h] * (j - (kv_len - seq_len + i))
    /// which simplifies to slopes[h] * (j - kv_offset - i) where kv_offset = kv_len - seq_len.
    pub fn build_bias_matrix(&self, seq_len: usize, kv_len: usize) -> Result<Tensor> {
        let device = self.slopes.device();
        let dtype = self.slopes.dtype();

        // Position indices
        // For query position i (0..seq_len), the actual position in the full sequence is (kv_len - seq_len + i)
        // For key position j (0..kv_len), it's just j
        // Distance = j - (kv_len - seq_len + i) = j - kv_offset - i
        let kv_offset = kv_len - seq_len;

        // Build distance matrix [seq_len, kv_len]
        // distance[i, j] = j - (kv_offset + i)
        let distances: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    let query_pos = kv_offset + i;
                    (j as i64 - query_pos as i64) as f32
                })
            })
            .collect();

        let distance_matrix =
            Tensor::from_vec(distances, (1, 1, seq_len, kv_len), device)?.to_dtype(dtype)?;

        // Expand slopes to [1, num_heads, 1, 1] for broadcasting
        let slopes_expanded = self.slopes.reshape((1, self.num_heads, 1, 1))?;

        // Compute bias: slopes * distances
        // Result shape: [1, num_heads, seq_len, kv_len]
        distance_matrix.broadcast_mul(&slopes_expanded)
    }

    /// Apply ALiBi bias to attention scores.
    ///
    /// # Arguments
    ///
    /// * `attention_scores` - Tensor of shape [batch, num_heads, seq_len, kv_len]
    ///
    /// # Returns
    ///
    /// Attention scores with ALiBi bias added, same shape as input.
    pub fn apply(&self, attention_scores: &Tensor) -> Result<Tensor> {
        let dims = attention_scores.dims();
        if dims.len() != 4 {
            return Err(candle_core::Error::Msg(format!(
                "Expected 4D attention scores [batch, heads, seq, kv], got {:?}",
                dims
            )));
        }

        let seq_len = dims[2];
        let kv_len = dims[3];

        let bias = self.build_bias_matrix(seq_len, kv_len)?;
        attention_scores.broadcast_add(&bias)
    }

    /// Apply ALiBi bias to attention scores with explicit dimensions.
    ///
    /// More efficient than `apply` when called repeatedly with the same dimensions,
    /// as the caller can cache the bias matrix.
    ///
    /// # Arguments
    ///
    /// * `attention_scores` - Tensor of shape [batch, num_heads, seq_len, kv_len]
    /// * `alibi_bias` - Pre-computed bias from `build_bias_matrix`
    ///
    /// # Returns
    ///
    /// Attention scores with ALiBi bias added.
    pub fn apply_precomputed(attention_scores: &Tensor, alibi_bias: &Tensor) -> Result<Tensor> {
        attention_scores.broadcast_add(alibi_bias)
    }
}

/// Compute ALiBi slopes for the given number of heads.
///
/// This follows the algorithm from the ALiBi paper:
/// - For power-of-2 head counts: slopes are a geometric sequence
/// - For non-power-of-2: combine slopes from closest power of 2 with extra slopes
///
/// # Arguments
///
/// * `num_heads` - Number of attention heads
///
/// # Returns
///
/// Vector of slopes, one per head. Slopes are always positive and decrease
/// geometrically (head 0 has the largest slope).
pub fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
    // Find closest power of 2 <= num_heads
    let closest_power_of_2 = 1usize << ((num_heads as f64).log2().floor() as u32);

    // Base for the geometric sequence
    // base = 2^(-(2^(-(log2(n) - 3))))
    // For n=8: log2(8)-3 = 0, so 2^0 = 1, so 2^(-1) = 0.5, base = 2^(-0.5)
    let exponent = -((closest_power_of_2 as f64).log2() - 3.0);
    let base = 2.0_f64.powf(-(2.0_f64.powf(exponent)));

    // Compute slopes for the first closest_power_of_2 heads
    let mut slopes: Vec<f32> = (1..=closest_power_of_2)
        .map(|i| base.powi(i as i32) as f32)
        .collect();

    // If num_heads is not a power of 2, we need extra slopes
    if closest_power_of_2 != num_heads {
        // Extra base with tighter geometric ratio
        let extra_exponent = -((2 * closest_power_of_2) as f64).log2() + 3.0;
        let extra_base = 2.0_f64.powf(-(2.0_f64.powf(extra_exponent)));

        let num_remaining_heads = (num_heads - closest_power_of_2).min(closest_power_of_2);

        // Use odd powers: 1, 3, 5, ...
        let extra_slopes: Vec<f32> = (0..num_remaining_heads)
            .map(|i| {
                let power = 1 + 2 * i;
                extra_base.powi(power as i32) as f32
            })
            .collect();

        slopes.extend(extra_slopes);
    }

    slopes
}

/// Build ALiBi bias matrix for a single sequence.
///
/// Standalone function for cases where you don't need the full `AlibiAttentionBias` struct.
///
/// # Arguments
///
/// * `alibi_slopes` - Per-head slopes, shape [num_heads]
/// * `seq_len` - Query sequence length
/// * `kv_len` - Key/value sequence length
/// * `dtype` - Output data type
/// * `device` - Output device
///
/// # Returns
///
/// Bias tensor of shape [1, num_heads, seq_len, kv_len]
pub fn build_alibi_bias(
    alibi_slopes: &Tensor,
    seq_len: usize,
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let num_heads = alibi_slopes.dims()[0];
    let kv_offset = kv_len - seq_len;

    // Build distance matrix [seq_len, kv_len]
    let distances: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                let query_pos = kv_offset + i;
                (j as i64 - query_pos as i64) as f32
            })
        })
        .collect();

    let distance_matrix = Tensor::from_vec(distances, (1, 1, seq_len, kv_len), device)?;

    // Expand slopes to [1, num_heads, 1, 1]
    let slopes_expanded = alibi_slopes.reshape((1, num_heads, 1, 1))?;

    // Compute bias
    distance_matrix
        .broadcast_mul(&slopes_expanded)?
        .to_dtype(dtype)
}

/// Apply ALiBi bias to attention scores.
///
/// Convenience function that computes and applies the bias in one call.
///
/// # Arguments
///
/// * `attention_scores` - Tensor of shape [batch, num_heads, seq_len, kv_len]
/// * `alibi_slopes` - Per-head slopes, shape [num_heads]
///
/// # Returns
///
/// Attention scores with ALiBi bias added.
pub fn apply_alibi_bias(attention_scores: &Tensor, alibi_slopes: &Tensor) -> Result<Tensor> {
    let dims = attention_scores.dims();
    if dims.len() != 4 {
        return Err(candle_core::Error::Msg(format!(
            "Expected 4D attention scores [batch, heads, seq, kv], got {:?}",
            dims
        )));
    }

    let seq_len = dims[2];
    let kv_len = dims[3];

    let bias = build_alibi_bias(
        alibi_slopes,
        seq_len,
        kv_len,
        attention_scores.dtype(),
        attention_scores.device(),
    )?;

    attention_scores.broadcast_add(&bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Slope Computation Tests ────────────────────────────────────────────────

    #[test]
    fn test_alibi_slopes_power_of_2_heads() {
        // Test with 8 heads (power of 2)
        let slopes = compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);

        // Slopes should be a geometric sequence, all positive
        for slope in &slopes {
            assert!(*slope > 0.0, "Slope should be positive");
            assert!(*slope <= 1.0, "Slope should be <= 1.0");
        }

        // Verify geometric progression: each slope is base * previous
        // For 8 heads: base = 2^(-0.5) ≈ 0.7071, so ratio should be constant
        for i in 1..slopes.len() {
            let ratio = slopes[i] / slopes[i - 1];
            let expected_ratio = slopes[1] / slopes[0];
            assert!(
                (ratio - expected_ratio).abs() < 1e-5,
                "Slopes should form geometric sequence"
            );
        }
    }

    #[test]
    fn test_alibi_slopes_non_power_of_2_heads() {
        // Test with 12 heads (not a power of 2)
        let slopes = compute_alibi_slopes(12);
        assert_eq!(slopes.len(), 12);

        // All slopes should be positive
        for slope in &slopes {
            assert!(*slope > 0.0, "Slope should be positive");
        }
    }

    #[test]
    fn test_alibi_slopes_known_values_8_heads() {
        // Verify against known values from the ALiBi paper
        // For 8 heads: base = 2^(-(2^(-(log2(8)-3)))) = 2^(-(2^0)) = 2^(-1) = 0.5
        // Wait, let me recalculate:
        // log2(8) = 3, so exponent = -(3 - 3) = 0
        // 2^0 = 1, so base = 2^(-1) = 0.5
        // Hmm, but that gives slopes: 0.5, 0.25, 0.125, ...
        //
        // Actually the formula is: base = 2^(-(2^(-(log2(n) - 3))))
        // For n=8: -(log2(8) - 3) = -(3 - 3) = 0
        // 2^0 = 1
        // 2^(-1) = 0.5
        // So base = 0.5, and slopes are 0.5^1, 0.5^2, 0.5^3, ... = 0.5, 0.25, 0.125, ...

        let slopes = compute_alibi_slopes(8);

        let expected = [
            0.5_f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];

        for (i, (&actual, &expected)) in slopes.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Head {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_alibi_slopes_16_heads() {
        let slopes = compute_alibi_slopes(16);
        assert_eq!(slopes.len(), 16);

        // For 16 heads: log2(16) = 4
        // exponent = -(4 - 3) = -1
        // 2^(-1) = 0.5
        // base = 2^(-0.5) ≈ 0.7071
        let base: f32 = 2.0_f32.powf(-0.5);

        for (i, &slope) in slopes.iter().enumerate() {
            let expected = base.powi((i + 1) as i32);
            assert!(
                (slope - expected).abs() < 1e-5,
                "Head {i}: expected {expected}, got {slope}"
            );
        }
    }

    #[test]
    fn test_alibi_slopes_different_head_counts() {
        // Test various head counts used in real models
        for num_heads in [1, 2, 4, 8, 12, 16, 24, 32, 40, 64] {
            let slopes = compute_alibi_slopes(num_heads);
            assert_eq!(
                slopes.len(),
                num_heads,
                "Wrong number of slopes for {num_heads} heads"
            );

            // All slopes must be positive and finite
            for (i, &slope) in slopes.iter().enumerate() {
                assert!(
                    slope > 0.0 && slope.is_finite(),
                    "Invalid slope at head {i} for {num_heads} heads: {slope}"
                );
            }
        }
    }

    // ─── Bias Matrix Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_alibi_bias_shape() {
        let device = Device::Cpu;
        let num_heads = 8;
        let seq_len = 16;
        let kv_len = 16;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        let bias = alibi
            .build_bias_matrix(seq_len, kv_len)
            .expect("Failed to build bias matrix");

        assert_eq!(bias.dims(), &[1, num_heads, seq_len, kv_len]);
    }

    #[test]
    fn test_alibi_bias_causal_pattern() {
        // For causal attention, biases should be non-positive on the lower triangle
        // (key position <= query position means distance <= 0)
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 8;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        let bias = alibi
            .build_bias_matrix(seq_len, seq_len)
            .expect("Failed to build bias matrix");

        let bias_data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        // Check that diagonal is zero and lower triangle is non-positive
        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    let distance = j as i64 - i as i64;

                    if distance == 0 {
                        assert!(
                            bias_data[idx].abs() < 1e-6,
                            "Diagonal should be zero at head {h}, pos ({i},{j})"
                        );
                    } else if distance < 0 {
                        // j < i: past positions have negative bias
                        assert!(
                            bias_data[idx] < 0.0,
                            "Past positions should have negative bias at head {h}, pos ({i},{j})"
                        );
                    } else {
                        // distance > 0: future positions have positive bias
                        // (will be masked out by causal mask anyway)
                        assert!(
                            bias_data[idx] > 0.0,
                            "Future positions should have positive bias at head {h}, pos ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_alibi_bias_values_head_0() {
        // Verify specific values for head 0
        let device = Device::Cpu;
        let num_heads = 8;
        let seq_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        let slopes = compute_alibi_slopes(num_heads);
        let slope_0 = slopes[0]; // 0.5 for 8 heads

        let bias = alibi
            .build_bias_matrix(seq_len, seq_len)
            .expect("Failed to build bias matrix");

        // Extract head 0
        let bias_h0 = bias.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
        let bias_h0_data: Vec<f32> = bias_h0.flatten_all().unwrap().to_vec1().unwrap();

        // Verify: bias[i,j] = slope * (j - i)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                let expected = slope_0 * (j as i64 - i as i64) as f32;
                assert!(
                    (bias_h0_data[idx] - expected).abs() < 1e-5,
                    "Mismatch at ({i},{j}): expected {expected}, got {}",
                    bias_h0_data[idx]
                );
            }
        }
    }

    #[test]
    fn test_alibi_bias_with_kv_cache() {
        // Test when kv_len > seq_len (decode with cached KV)
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 1; // Decode: single query token
        let kv_len = 10; // 10 tokens in KV cache

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        let bias = alibi
            .build_bias_matrix(seq_len, kv_len)
            .expect("Failed to build bias matrix");

        assert_eq!(bias.dims(), &[1, num_heads, seq_len, kv_len]);

        let slopes = compute_alibi_slopes(num_heads);
        let bias_data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        // Query is at position kv_len - 1 = 9
        // Keys are at positions 0..10
        // bias[h, 0, j] = slope[h] * (j - 9)
        for h in 0..num_heads {
            for j in 0..kv_len {
                let idx = h * kv_len + j;
                let expected = slopes[h] * (j as i64 - 9) as f32;
                assert!(
                    (bias_data[idx] - expected).abs() < 1e-5,
                    "Head {h}, key {j}: expected {expected}, got {}",
                    bias_data[idx]
                );
            }
        }
    }

    // ─── Apply Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_alibi_bias() {
        let device = Device::Cpu;
        let batch = 2;
        let num_heads = 8;
        let seq_len = 4;
        let kv_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        // Create attention scores (all ones)
        let attn_scores =
            Tensor::ones((batch, num_heads, seq_len, kv_len), DType::F32, &device).unwrap();

        let biased_scores = alibi.apply(&attn_scores).expect("Failed to apply ALiBi");

        assert_eq!(biased_scores.dims(), &[batch, num_heads, seq_len, kv_len]);

        // Verify that diagonal entries are 1.0 (no bias on diagonal)
        let biased_data: Vec<f32> = biased_scores.flatten_all().unwrap().to_vec1().unwrap();
        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let idx =
                        b * num_heads * seq_len * kv_len + h * seq_len * kv_len + i * kv_len + i;
                    assert!(
                        (biased_data[idx] - 1.0).abs() < 1e-5,
                        "Diagonal should be 1.0 (original score), got {}",
                        biased_data[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_apply_alibi_bias_standalone() {
        let device = Device::Cpu;
        let num_heads = 8;
        let seq_len = 4;

        let slopes = compute_alibi_slopes(num_heads);
        let slopes_tensor = Tensor::from_vec(slopes.clone(), (num_heads,), &device)
            .expect("Failed to create slopes");

        let attn_scores =
            Tensor::zeros((1, num_heads, seq_len, seq_len), DType::F32, &device).unwrap();

        let biased = apply_alibi_bias(&attn_scores, &slopes_tensor).expect("Failed to apply ALiBi");

        // With zero scores, result should equal the bias
        let biased_data: Vec<f32> = biased.flatten_all().unwrap().to_vec1().unwrap();

        for h in 0..num_heads {
            for i in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + i;
                assert!(
                    biased_data[idx].abs() < 1e-6,
                    "Diagonal should be zero, got {}",
                    biased_data[idx]
                );
            }
        }
    }

    // ─── Integration Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_alibi_with_causal_mask() {
        // Simulate how ALiBi works with a causal mask
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        // Create uniform attention scores
        let attn_scores =
            Tensor::zeros((1, num_heads, seq_len, seq_len), DType::F32, &device).unwrap();

        // Apply ALiBi
        let biased = alibi.apply(&attn_scores).expect("Failed to apply ALiBi");

        // Create causal mask
        let causal = crate::layers::mask::causal_mask(seq_len, 0, DType::F32, &device)
            .expect("Failed to create causal mask");

        // Combine
        let masked = biased
            .broadcast_add(&causal)
            .expect("Failed to apply causal mask");

        // After softmax, future positions should be zero, past positions should decrease
        let masked_data: Vec<f32> = masked.flatten_all().unwrap().to_vec1().unwrap();

        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    if j > i {
                        // Future: should be -inf
                        assert!(
                            masked_data[idx] == f32::NEG_INFINITY,
                            "Future position ({i},{j}) should be -inf"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_alibi_dtype_preservation() {
        let device = Device::Cpu;
        let num_heads = 8;
        let seq_len = 4;

        // Test F32
        let alibi_f32 = AlibiAttentionBias::new(num_heads, DType::F32, &device).unwrap();
        let bias_f32 = alibi_f32.build_bias_matrix(seq_len, seq_len).unwrap();
        assert_eq!(bias_f32.dtype(), DType::F32);

        // Test F16
        let alibi_f16 = AlibiAttentionBias::new(num_heads, DType::F16, &device).unwrap();
        let bias_f16 = alibi_f16.build_bias_matrix(seq_len, seq_len).unwrap();
        assert_eq!(bias_f16.dtype(), DType::F16);

        // Test BF16
        let alibi_bf16 = AlibiAttentionBias::new(num_heads, DType::BF16, &device).unwrap();
        let bias_bf16 = alibi_bf16.build_bias_matrix(seq_len, seq_len).unwrap();
        assert_eq!(bias_bf16.dtype(), DType::BF16);
    }

    #[test]
    fn test_alibi_partial_heads() {
        // Test tensor parallel scenario
        let device = Device::Cpu;
        let total_heads = 16;
        let heads_per_device = 4;

        // Device 0: heads 0-3
        let alibi_0 = AlibiAttentionBias::new_partial(total_heads, 0, 4, DType::F32, &device)
            .expect("Failed to create partial ALiBi");

        // Device 1: heads 4-7
        let alibi_1 = AlibiAttentionBias::new_partial(total_heads, 4, 8, DType::F32, &device)
            .expect("Failed to create partial ALiBi");

        assert_eq!(alibi_0.num_heads(), heads_per_device);
        assert_eq!(alibi_1.num_heads(), heads_per_device);

        // Verify slopes are correct slices of full slopes
        let full_slopes = compute_alibi_slopes(total_heads);

        let slopes_0: Vec<f32> = alibi_0.slopes().to_vec1().unwrap();
        let slopes_1: Vec<f32> = alibi_1.slopes().to_vec1().unwrap();

        for i in 0..heads_per_device {
            assert!(
                (slopes_0[i] - full_slopes[i]).abs() < 1e-6,
                "Device 0 slope mismatch at {i}"
            );
            assert!(
                (slopes_1[i] - full_slopes[4 + i]).abs() < 1e-6,
                "Device 1 slope mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_alibi_bloom_config() {
        // BLOOM-176B uses 112 heads
        let num_heads = 112;
        let slopes = compute_alibi_slopes(num_heads);

        assert_eq!(slopes.len(), num_heads);
        assert!(slopes.iter().all(|&s| s > 0.0 && s.is_finite()));
    }

    #[test]
    fn test_alibi_mpt_config() {
        // MPT-7B uses 32 heads
        let num_heads = 32;
        let slopes = compute_alibi_slopes(num_heads);

        assert_eq!(slopes.len(), num_heads);
        assert!(slopes.iter().all(|&s| s > 0.0 && s.is_finite()));

        // 32 is a power of 2, so slopes should be a clean geometric sequence
        let base: f32 = 2.0_f32.powf(-0.25); // base for 32 heads
        for (i, &slope) in slopes.iter().enumerate() {
            let expected = base.powi((i + 1) as i32);
            assert!(
                (slope - expected).abs() < 1e-5,
                "MPT head {i}: expected {expected}, got {slope}"
            );
        }
    }

    #[test]
    fn test_alibi_attention_score_range() {
        // Verify that ALiBi biases don't cause numerical issues
        let device = Device::Cpu;
        let num_heads = 8;
        let seq_len = 2048; // Long sequence

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        let bias = alibi
            .build_bias_matrix(seq_len, seq_len)
            .expect("Failed to build bias matrix");

        let bias_data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        // All values should be finite
        assert!(
            bias_data.iter().all(|v| v.is_finite()),
            "ALiBi bias should be finite for long sequences"
        );

        // The most negative bias (first head, last row, first column)
        // should still be finite
        let slopes = compute_alibi_slopes(num_heads);
        let max_negative_bias = slopes[0] * (-(seq_len as i64 - 1)) as f32;
        assert!(
            max_negative_bias.is_finite(),
            "Most negative bias should be finite: {max_negative_bias}"
        );
    }

    #[test]
    fn test_alibi_precomputed_apply() {
        let device = Device::Cpu;
        let batch = 2;
        let num_heads = 8;
        let seq_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
            .expect("Failed to create AlibiAttentionBias");

        let attn_scores =
            Tensor::ones((batch, num_heads, seq_len, seq_len), DType::F32, &device).unwrap();

        // Method 1: Direct apply
        let result1 = alibi.apply(&attn_scores).expect("Failed to apply");

        // Method 2: Precomputed
        let bias = alibi.build_bias_matrix(seq_len, seq_len).unwrap();
        let result2 = AlibiAttentionBias::apply_precomputed(&attn_scores, &bias)
            .expect("Failed to apply precomputed");

        // Results should be identical
        let data1: Vec<f32> = result1.flatten_all().unwrap().to_vec1().unwrap();
        let data2: Vec<f32> = result2.flatten_all().unwrap().to_vec1().unwrap();

        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "Results should match: {v1} vs {v2}");
        }
    }
}
