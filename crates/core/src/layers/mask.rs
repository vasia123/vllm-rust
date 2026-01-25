use candle_core::{DType, Device, Result, Tensor};

/// Generate a causal attention mask for decoder-only models.
/// Returns shape [1, 1, seq_len, seq_len + seqlen_offset].
pub fn causal_mask(
    seq_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_len = seq_len + seqlen_offset;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                if j > i + seqlen_offset {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?;
    mask.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;
        let seq_len = 8;
        let seqlen_offset = 0;

        let mask = causal_mask(seq_len, seqlen_offset, DType::F32, &device)
            .expect("Failed to create mask");

        // Shape should be [1, 1, seq_len, seq_len + offset]
        assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
    }

    #[test]
    fn test_causal_mask_shape_with_offset() {
        let device = Device::Cpu;
        let seq_len = 4;
        let seqlen_offset = 10;

        let mask = causal_mask(seq_len, seqlen_offset, DType::F32, &device)
            .expect("Failed to create mask");

        // Shape should be [1, 1, seq_len, seq_len + offset]
        let total_len = seq_len + seqlen_offset;
        assert_eq!(mask.dims(), &[1, 1, seq_len, total_len]);
    }

    #[test]
    fn test_causal_mask_values_no_offset() {
        // Test the classic causal mask pattern
        let device = Device::Cpu;
        let seq_len = 4;

        let mask = causal_mask(seq_len, 0, DType::F32, &device).expect("Failed to create mask");

        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Expected pattern (row-major, 4x4):
        // [0, -inf, -inf, -inf]  row 0: can only attend to position 0
        // [0,    0, -inf, -inf]  row 1: can attend to positions 0,1
        // [0,    0,    0, -inf]  row 2: can attend to positions 0,1,2
        // [0,    0,    0,    0]  row 3: can attend to all positions

        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if j > i {
                    // Upper triangle: should be -inf
                    assert!(
                        mask_data[idx] == f32::NEG_INFINITY,
                        "Position ({i},{j}) should be -inf, got {}",
                        mask_data[idx]
                    );
                } else {
                    // Lower triangle + diagonal: should be 0
                    assert!(
                        mask_data[idx] == 0.0,
                        "Position ({i},{j}) should be 0, got {}",
                        mask_data[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_values_with_offset() {
        // With offset, we're continuing generation from position `offset`
        let device = Device::Cpu;
        let seq_len = 2;
        let seqlen_offset = 3;

        let mask = causal_mask(seq_len, seqlen_offset, DType::F32, &device)
            .expect("Failed to create mask");

        let total_len = seq_len + seqlen_offset; // 5
        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Expected pattern (2x5):
        // Row 0 (position 3): can attend to positions 0,1,2,3 (indices 0-3), not 4
        //   [0, 0, 0, 0, -inf]
        // Row 1 (position 4): can attend to all positions 0-4
        //   [0, 0, 0, 0, 0]

        // Row 0: j=0..4 should be 0 (j <= 0 + 3 = 3), j=4 should be -inf
        for j in 0..total_len {
            let idx = j;
            if j > 0 + seqlen_offset {
                assert!(
                    mask_data[idx] == f32::NEG_INFINITY,
                    "Row 0, col {j} should be -inf"
                );
            } else {
                assert!(mask_data[idx] == 0.0, "Row 0, col {j} should be 0");
            }
        }

        // Row 1: all should be 0 (j <= 1 + 3 = 4, and total_len = 5, so j goes 0-4)
        for j in 0..total_len {
            let idx = total_len + j;
            assert!(
                mask_data[idx] == 0.0,
                "Row 1, col {j} should be 0, got {}",
                mask_data[idx]
            );
        }
    }

    #[test]
    fn test_causal_mask_dtype() {
        let device = Device::Cpu;

        let mask_f32 = causal_mask(4, 0, DType::F32, &device).expect("Failed to create F32 mask");
        assert_eq!(mask_f32.dtype(), DType::F32);

        let mask_f16 = causal_mask(4, 0, DType::F16, &device).expect("Failed to create F16 mask");
        assert_eq!(mask_f16.dtype(), DType::F16);

        let mask_bf16 =
            causal_mask(4, 0, DType::BF16, &device).expect("Failed to create BF16 mask");
        assert_eq!(mask_bf16.dtype(), DType::BF16);
    }

    #[test]
    fn test_causal_mask_single_token() {
        // Decode phase: seq_len=1
        let device = Device::Cpu;
        let seqlen_offset = 10;

        let mask =
            causal_mask(1, seqlen_offset, DType::F32, &device).expect("Failed to create mask");

        // Shape: [1, 1, 1, 11]
        assert_eq!(mask.dims(), &[1, 1, 1, 11]);

        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Single row (position 10): can attend to all previous positions 0-10
        assert!(
            mask_data.iter().all(|&v| v == 0.0),
            "All positions should be attendable for single decode token"
        );
    }

    #[test]
    fn test_causal_mask_large_sequence() {
        // Test with larger sequence to ensure no overflow
        let device = Device::Cpu;
        let seq_len = 128;
        let seqlen_offset = 512;

        let mask = causal_mask(seq_len, seqlen_offset, DType::F32, &device)
            .expect("Failed to create mask");

        let total_len = seq_len + seqlen_offset;
        assert_eq!(mask.dims(), &[1, 1, seq_len, total_len]);

        // Spot check: last row should have all zeros (can attend to everything)
        let last_row = mask
            .narrow(2, seq_len - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap();
        let last_row_data: Vec<f32> = last_row.to_vec1().unwrap();
        assert!(
            last_row_data.iter().all(|&v| v == 0.0),
            "Last row should have all zeros"
        );

        // First row should have -inf for positions > seqlen_offset
        let first_row = mask.narrow(2, 0, 1).unwrap().flatten_all().unwrap();
        let first_row_data: Vec<f32> = first_row.to_vec1().unwrap();
        for (j, &v) in first_row_data.iter().enumerate() {
            if j > seqlen_offset {
                assert!(v == f32::NEG_INFINITY, "First row, col {j} should be -inf");
            } else {
                assert!(v == 0.0, "First row, col {j} should be 0");
            }
        }
    }

    #[test]
    fn test_causal_mask_broadcast_shape() {
        // The mask has shape [1, 1, seq, total] for broadcasting with [batch, heads, seq, total]
        let device = Device::Cpu;
        let mask = causal_mask(8, 0, DType::F32, &device).expect("Failed to create mask");

        // First two dims should be 1 for broadcasting
        assert_eq!(mask.dims()[0], 1);
        assert_eq!(mask.dims()[1], 1);
    }

    #[test]
    fn test_causal_mask_attention_application() {
        // Test that mask correctly zeros out future attention
        let device = Device::Cpu;
        let seq_len = 4;

        let mask = causal_mask(seq_len, 0, DType::F32, &device).expect("Failed to create mask");

        // Create fake attention scores
        let attn_scores = Tensor::ones((1, 1, seq_len, seq_len), DType::F32, &device)
            .expect("Failed to create attn scores");

        // Apply mask
        let masked_scores = attn_scores
            .broadcast_add(&mask)
            .expect("broadcast_add failed");
        let masked_data: Vec<f32> = masked_scores.flatten_all().unwrap().to_vec1().unwrap();

        // After softmax, -inf positions become 0 probability
        // Here we just verify the mask was applied correctly
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if j > i {
                    assert!(
                        masked_data[idx] == f32::NEG_INFINITY,
                        "Future position ({i},{j}) should be masked"
                    );
                } else {
                    assert!(
                        (masked_data[idx] - 1.0).abs() < 1e-6,
                        "Attended position ({i},{j}) should be 1.0"
                    );
                }
            }
        }
    }
}
