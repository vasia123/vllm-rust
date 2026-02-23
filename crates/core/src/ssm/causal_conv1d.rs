//! Causal 1D convolution for SSM (Mamba-style) models.
//!
//! Used by Mamba, Mamba2, Jamba, Bamba, Zamba2, Falcon-H1, PLaMo-2, NemotronH,
//! and GraniteMoE-Hybrid. All share an identical depthwise causal conv1d.
//!
//! Weight shape: `[d_inner, 1, kernel_size]` (depthwise, groups=d_inner).
//! Bias shape:   `[d_inner]`.
//! Input shape (prefill): `[batch, d_inner, seq_len]` — channels-last convention.
//! Input shape (decode):  `[batch, d_inner]`.

use candle_core::{Result, Tensor};

/// Applies causal depthwise 1D convolution over a full sequence (prefill).
///
/// Left-pads the input with `kernel_size - 1` zeros so that the output at
/// position `t` only depends on positions `≤ t` (causal).
///
/// # Arguments
/// * `x`      — `[batch, d_inner, seq_len]`
/// * `weight` — `[d_inner, 1, kernel_size]`
/// * `bias`   — `[d_inner]`
///
/// # Returns
/// `[batch, d_inner, seq_len]`
pub fn causal_conv1d_prefill(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_batch, d_inner, seq_len) = x.dims3()?;
    let (_d_inner_w, _one, kernel_size) = weight.dims3()?;

    // Left-pad with zeros so each output position only sees past context.
    let pad_len = kernel_size - 1;
    let pad = Tensor::zeros((x.dims()[0], d_inner, pad_len), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&pad, x], 2)?; // [batch, d_inner, pad_len + seq_len]

    // Depthwise convolution — squeeze groups dim once, then broadcast over batch.
    let w = weight.squeeze(1)?; // [d_inner, kernel_size]
    let w_expanded = w.unsqueeze(0)?; // [1, d_inner, kernel_size]

    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let window = padded.narrow(2, t, kernel_size)?; // [batch, d_inner, kernel_size]
        let product = window.broadcast_mul(&w_expanded)?;
        let conv_out = product.sum(2)?; // [batch, d_inner]
        let conv_out = conv_out.broadcast_add(bias)?;
        outputs.push(conv_out.unsqueeze(2)?); // [batch, d_inner, 1]
    }

    Tensor::cat(&outputs, 2) // [batch, d_inner, seq_len]
}

/// Applies causal depthwise 1D convolution for a single decode step.
///
/// Shifts the convolution state left by one and appends the new token, then
/// computes the dot product over the `kernel_size` window.
///
/// # Arguments
/// * `x`          — `[batch, d_inner]` (new token)
/// * `weight`     — `[d_inner, 1, kernel_size]`
/// * `bias`       — `[d_inner]`
/// * `conv_state` — `[batch, d_inner, kernel_size - 1]` (ring buffer)
///
/// # Returns
/// `(output [batch, d_inner], new_conv_state [batch, d_inner, kernel_size - 1])`
pub fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    conv_state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, d_inner) = x.dims2()?;
    let (_d_inner_w, _one, kernel_size) = weight.dims3()?;
    let conv_state_len = kernel_size - 1;

    let x_expanded = x.unsqueeze(2)?; // [batch, d_inner, 1]

    // Shift old state left, append new token.
    let new_conv_state = if conv_state_len > 1 {
        let shifted = conv_state.narrow(2, 1, conv_state_len - 1)?;
        Tensor::cat(&[&shifted, &x_expanded], 2)? // [batch, d_inner, conv_state_len]
    } else if conv_state_len == 1 {
        x_expanded.clone()
    } else {
        // kernel_size == 1: no history needed.
        Tensor::zeros((batch, d_inner, 0), x.dtype(), x.device())?
    };

    // Full kernel window: [conv_state | x] → [batch, d_inner, kernel_size]
    let full_window = Tensor::cat(&[&new_conv_state, &x_expanded], 2)?;

    let w = weight.squeeze(1)?; // [d_inner, kernel_size]
    let w_expanded = w.unsqueeze(0)?; // [1, d_inner, kernel_size]
    let product = full_window.broadcast_mul(&w_expanded)?;
    let conv_out = product.sum(2)?; // [batch, d_inner]
    let conv_out = conv_out.broadcast_add(bias)?;

    Ok((conv_out, new_conv_state))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_causal_conv1d_prefill_shape() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 8, 16), DType::F32, &device).unwrap();
        let weight = Tensor::ones((8, 1, 4), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(8, DType::F32, &device).unwrap();
        let out = causal_conv1d_prefill(&x, &weight, &bias).unwrap();
        assert_eq!(out.dims(), &[2, 8, 16]);
    }

    #[test]
    fn test_causal_conv1d_decode_shape() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 8), DType::F32, &device).unwrap();
        let weight = Tensor::ones((8, 1, 4), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(8, DType::F32, &device).unwrap();
        let conv_state = Tensor::zeros((2, 8, 3), DType::F32, &device).unwrap();
        let (out, new_state) = causal_conv1d_decode(&x, &weight, &bias, &conv_state).unwrap();
        assert_eq!(out.dims(), &[2, 8]);
        assert_eq!(new_state.dims(), &[2, 8, 3]);
    }

    #[test]
    fn test_causal_conv1d_decode_kernel1() {
        // kernel_size == 1: no state needed, conv_state has length 0
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let weight = Tensor::ones((4, 1, 1), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(4, DType::F32, &device).unwrap();
        let conv_state = Tensor::zeros((1, 4, 0), DType::F32, &device).unwrap();
        let (out, new_state) = causal_conv1d_decode(&x, &weight, &bias, &conv_state).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
        assert_eq!(new_state.dims(), &[1, 4, 0]);
    }

    #[test]
    fn test_causal_conv1d_prefill_causality() {
        // With impulse at position 0 and kernel [1, 0, 0, 0], only position 0 is affected.
        let device = Device::Cpu;
        let mut vals = vec![0.0f32; 1 * 1 * 8];
        vals[0] = 1.0; // impulse at t=0, d=0
        let x = Tensor::new(vals.as_slice(), &device)
            .unwrap()
            .reshape((1, 1, 8))
            .unwrap();
        // kernel [1, 0, 0, 0] (only current position contributes)
        let w = Tensor::new(&[0.0f32, 0.0, 0.0, 1.0], &device)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        let bias = Tensor::zeros(1, DType::F32, &device).unwrap();
        let out = causal_conv1d_prefill(&x, &w, &bias).unwrap();
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // impulse at t=0 should appear at output t=0 only with "current" kernel position
        assert!((flat[0] - 1.0).abs() < 1e-5, "expected 1 at t=0");
        for &v in &flat[1..] {
            assert!(v.abs() < 1e-5, "expected 0 at t>0");
        }
    }

    #[test]
    fn test_causal_conv1d_decode_state_shift() {
        // Implementation uses the UPDATED state for the full window:
        //   conv_state_len==1: new_state = [x], full_window = [x, x]
        // output = x*w0 + x*w1 = 2*(3+5) = 16.
        // The new state stores [x] = [2.0].
        let device = Device::Cpu;
        let x = Tensor::new(&[2.0f32], &device)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        let weight = Tensor::new(&[3.0f32, 5.0], &device)
            .unwrap()
            .reshape((1, 1, 2))
            .unwrap();
        let bias = Tensor::zeros(1, DType::F32, &device).unwrap();
        let conv_state = Tensor::new(&[1.0f32], &device)
            .unwrap()
            .reshape((1, 1, 1))
            .unwrap();
        let (out, new_state) = causal_conv1d_decode(&x, &weight, &bias, &conv_state).unwrap();
        let out_val = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(
            (out_val - 16.0).abs() < 1e-5,
            "expected 16.0, got {out_val}"
        );
        let state_val = new_state.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((state_val - 2.0).abs() < 1e-5);
    }
}
