//! Common attention operations shared across backends.

use crate::layers::RmsNorm;
use candle_core::{Result, Tensor};

/// Repeat KV heads for Grouped Query Attention.
///
/// When using GQA (Grouped Query Attention), the number of KV heads is less than
/// the number of query heads. This function repeats the KV heads so they can be
/// used with the full number of query heads.
///
/// # Arguments
/// * `x` - Input tensor `[batch, num_kv_heads, seq_len, head_dim]`
/// * `num_kv_groups` - Number of times to repeat each KV head (num_heads / num_kv_heads)
///
/// # Returns
/// Tensor with shape `[batch, num_heads, seq_len, head_dim]`
pub fn repeat_kv(x: Tensor, num_kv_groups: usize) -> Result<Tensor> {
    if num_kv_groups == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, s, d) = x.dims4()?;
    let num_heads = num_kv_heads * num_kv_groups;
    x.unsqueeze(2)?
        .expand((b, num_kv_heads, num_kv_groups, s, d))?
        .reshape((b, num_heads, s, d))
}

/// Apply RMSNorm per attention head (used by Qwen3).
///
/// Reshapes `[b, h, s, d]` → `[b*h*s, d]`, applies norm, reshapes back.
///
/// # Arguments
/// * `x` - Input tensor `[batch, num_heads, seq_len, head_dim]`
/// * `norm` - RmsNorm layer with dimension `head_dim`
///
/// # Returns
/// Normalized tensor with same shape as input
pub fn apply_per_head_norm(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let x = x.reshape((b * h * s, d))?;
    let x = candle_nn::Module::forward(norm, &x)?;
    x.reshape((b, h, s, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    // ==========================================================================
    // repeat_kv tests (GQA head repetition)
    // ==========================================================================

    #[test]
    fn test_repeat_kv_single_group_identity() {
        // When num_kv_groups=1, output should be identical to input
        let device = Device::Cpu;
        let x =
            Tensor::randn(0.0f32, 1.0, (1, 4, 16, 64), &device).expect("Failed to create tensor");

        let result = repeat_kv(x.clone(), 1).expect("repeat_kv failed");

        assert_eq!(result.dims(), x.dims());

        // Values should be identical
        let x_data: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in x_data.iter().zip(result_data.iter()) {
            assert!((a - b).abs() < 1e-6, "Values differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_repeat_kv_gqa_2groups() {
        // [1, 4, 16, 64] with num_kv_groups=2 → [1, 8, 16, 64]
        let device = Device::Cpu;
        let b = 1;
        let num_kv_heads = 4;
        let num_kv_groups = 2;
        let s = 16;
        let d = 64;

        let x = Tensor::randn(0.0f32, 1.0, (b, num_kv_heads, s, d), &device)
            .expect("Failed to create tensor");

        let result = repeat_kv(x.clone(), num_kv_groups).expect("repeat_kv failed");

        let expected_num_heads = num_kv_heads * num_kv_groups;
        assert_eq!(result.dims(), &[b, expected_num_heads, s, d]);
    }

    #[test]
    fn test_repeat_kv_gqa_4groups() {
        // [2, 2, 8, 32] with num_kv_groups=4 → [2, 8, 8, 32]
        let device = Device::Cpu;
        let b = 2;
        let num_kv_heads = 2;
        let num_kv_groups = 4;
        let s = 8;
        let d = 32;

        let x = Tensor::randn(0.0f32, 1.0, (b, num_kv_heads, s, d), &device)
            .expect("Failed to create tensor");

        let result = repeat_kv(x.clone(), num_kv_groups).expect("repeat_kv failed");

        let expected_num_heads = num_kv_heads * num_kv_groups;
        assert_eq!(result.dims(), &[b, expected_num_heads, s, d]);
    }

    #[test]
    fn test_repeat_kv_values_are_repeated_correctly() {
        // Verify that values are properly repeated for GQA
        let device = Device::Cpu;
        let b = 1;
        let num_kv_heads = 2;
        let num_kv_groups = 3;
        let s = 2;
        let d = 4;

        let x = Tensor::randn(0.0f32, 1.0, (b, num_kv_heads, s, d), &device)
            .expect("Failed to create tensor");

        let result = repeat_kv(x.clone(), num_kv_groups).expect("repeat_kv failed");

        // result should have shape [1, 6, 2, 4]
        assert_eq!(result.dims(), &[b, num_kv_heads * num_kv_groups, s, d]);

        // Heads 0,1,2 should be identical (from kv_head 0)
        // Heads 3,4,5 should be identical (from kv_head 1)
        let x_data: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Extract first kv_head's data (s*d elements)
        let kv_head_0: Vec<f32> = x_data[..s * d].to_vec();

        // Verify repetition for first group (heads 0,1,2)
        for group_idx in 0..num_kv_groups {
            let head_offset = group_idx * s * d;
            for i in 0..(s * d) {
                assert!(
                    (result_data[head_offset + i] - kv_head_0[i]).abs() < 1e-6,
                    "Head {group_idx} doesn't match kv_head 0 at position {i}"
                );
            }
        }
    }

    #[test]
    fn test_repeat_kv_mqa_case() {
        // MQA: single KV head shared across all Q heads
        // [1, 1, 16, 64] with num_kv_groups=8 → [1, 8, 16, 64]
        let device = Device::Cpu;
        let b = 1;
        let num_kv_heads = 1;
        let num_kv_groups = 8;
        let s = 16;
        let d = 64;

        let x = Tensor::randn(0.0f32, 1.0, (b, num_kv_heads, s, d), &device)
            .expect("Failed to create tensor");

        let result = repeat_kv(x.clone(), num_kv_groups).expect("repeat_kv failed");

        assert_eq!(result.dims(), &[b, 8, s, d]);

        // All 8 heads should have identical values
        let head_0 = result.narrow(1, 0, 1).unwrap();
        for h in 1..8 {
            let head_h = result.narrow(1, h, 1).unwrap();
            let h0_data: Vec<f32> = head_0.flatten_all().unwrap().to_vec1().unwrap();
            let hh_data: Vec<f32> = head_h.flatten_all().unwrap().to_vec1().unwrap();
            for (a, b) in h0_data.iter().zip(hh_data.iter()) {
                assert!((a - b).abs() < 1e-6, "MQA repetition failed at head {h}");
            }
        }
    }

    #[test]
    fn test_repeat_kv_batch_size() {
        // Test with larger batch size
        let device = Device::Cpu;
        let b = 4;
        let num_kv_heads = 4;
        let num_kv_groups = 2;
        let s = 32;
        let d = 128;

        let x = Tensor::randn(0.0f32, 1.0, (b, num_kv_heads, s, d), &device)
            .expect("Failed to create tensor");

        let result = repeat_kv(x, num_kv_groups).expect("repeat_kv failed");

        assert_eq!(result.dims(), &[b, num_kv_heads * num_kv_groups, s, d]);
    }

    // ==========================================================================
    // apply_per_head_norm tests
    // ==========================================================================

    #[test]
    fn test_apply_per_head_norm_shape() {
        let device = Device::Cpu;
        let b = 2;
        let h = 8;
        let s = 16;
        let d = 64;

        // Create RmsNorm
        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = crate::layers::rms_norm(d, 1e-6, vb).expect("Failed to create RmsNorm");

        let x = Tensor::randn(0.0f32, 1.0, (b, h, s, d), &device).expect("Failed to create tensor");

        let result = apply_per_head_norm(&x, &norm).expect("apply_per_head_norm failed");

        // Shape should be preserved
        assert_eq!(result.dims(), &[b, h, s, d]);
    }

    #[test]
    fn test_apply_per_head_norm_normalizes() {
        let device = Device::Cpu;
        let b = 1;
        let h = 4;
        let s = 8;
        let d = 32;

        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = crate::layers::rms_norm(d, 1e-6, vb).expect("Failed to create RmsNorm");

        // Create input with varying magnitudes
        let x = Tensor::randn(0.0f32, 5.0, (b, h, s, d), &device).expect("Failed to create tensor");

        let result = apply_per_head_norm(&x, &norm).expect("apply_per_head_norm failed");

        // Result should have finite values
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            result_data.iter().all(|v| v.is_finite()),
            "Normalized values should be finite"
        );
    }

    #[test]
    fn test_apply_per_head_norm_different_dimensions() {
        let device = Device::Cpu;

        // Test various dimension combinations
        let test_cases = [(1, 8, 16, 64), (2, 4, 32, 128), (4, 2, 8, 32)];

        for (b, h, s, d) in test_cases {
            let vb = VarBuilder::zeros(DType::F32, &device);
            let norm = crate::layers::rms_norm(d, 1e-6, vb).expect("Failed to create RmsNorm");

            let x =
                Tensor::randn(0.0f32, 1.0, (b, h, s, d), &device).expect("Failed to create tensor");

            let result = apply_per_head_norm(&x, &norm).expect("apply_per_head_norm failed");

            assert_eq!(
                result.dims(),
                &[b, h, s, d],
                "Shape mismatch for dims ({b}, {h}, {s}, {d})"
            );
        }
    }

    // ==========================================================================
    // Edge cases
    // ==========================================================================

    #[test]
    fn test_repeat_kv_seq_len_1() {
        // Single token case (decode phase)
        let device = Device::Cpu;
        let x =
            Tensor::randn(0.0f32, 1.0, (1, 4, 1, 64), &device).expect("Failed to create tensor");

        let result = repeat_kv(x, 2).expect("repeat_kv failed");

        assert_eq!(result.dims(), &[1, 8, 1, 64]);
    }

    #[test]
    fn test_repeat_kv_preserves_dtype() {
        let device = Device::Cpu;
        let x =
            Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).expect("Failed to create tensor");

        let result = repeat_kv(x.clone(), 3).expect("repeat_kv failed");

        assert_eq!(result.dtype(), x.dtype());
    }
}
