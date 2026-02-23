//! Gated RMS normalization for Mamba-2 (Mixer2RMSNormGated).
//!
//! Mamba-2 uses a post-SSM gated normalization where the SSM output is
//! combined with a gate from in_proj before being normalized.
//!
//! Two modes controlled by `norm_before_gate`:
//! - `false` (Mamba-2 default): gate first, then normalize — `rms_norm(x * silu(z))`
//! - `true`: normalize first, then gate — `rms_norm(x) * silu(z)`

use candle_core::{Result, Tensor};

/// Applies RMS normalization with an optional gate.
///
/// # Arguments
/// * `x`               — `[..., d]` tensor to normalize (SSM output)
/// * `z`               — `[..., d]` gate tensor
/// * `weight`          — `[d]` learnable scale
/// * `eps`             — numerical stability epsilon (typically 1e-6)
/// * `norm_before_gate` — if true: `rms_norm(x) * silu(z)`;
///   if false (Mamba-2 default): `rms_norm(x * silu(z))`
///
/// # Returns
/// Normalized output with same shape as `x`.
pub fn rms_norm_gated(
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
    norm_before_gate: bool,
) -> Result<Tensor> {
    let gate = silu(z)?;

    let y = if norm_before_gate {
        // normalize x first, then apply gate
        let normed = rms_norm_inplace(x, weight, eps)?;
        normed.broadcast_mul(&gate)?
    } else {
        // gate x first, then normalize the result
        let gated = x.broadcast_mul(&gate)?;
        rms_norm_inplace(&gated, weight, eps)?
    };

    Ok(y)
}

/// SiLU activation: `x * sigmoid(x)`.
fn silu(x: &Tensor) -> Result<Tensor> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let sig = candle_nn::ops::sigmoid(x)?;
    x.mul(&sig)
}

/// RMS normalization without bias: `x / rms(x) * weight`.
fn rms_norm_inplace(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let x_normed = x.broadcast_div(&rms)?;
    x_normed.broadcast_mul(weight)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_rms_norm_gated_norm_after_gate_shape() {
        let device = Device::Cpu;
        let x = Tensor::ones((2, 4, 8), DType::F32, &device).unwrap();
        let z = Tensor::ones((2, 4, 8), DType::F32, &device).unwrap();
        let weight = Tensor::ones(8, DType::F32, &device).unwrap();
        let out = rms_norm_gated(&x, &z, &weight, 1e-6, false).unwrap();
        assert_eq!(out.dims(), &[2, 4, 8]);
    }

    #[test]
    fn test_rms_norm_gated_norm_before_gate_shape() {
        let device = Device::Cpu;
        let x = Tensor::ones((2, 4, 8), DType::F32, &device).unwrap();
        let z = Tensor::ones((2, 4, 8), DType::F32, &device).unwrap();
        let weight = Tensor::ones(8, DType::F32, &device).unwrap();
        let out = rms_norm_gated(&x, &z, &weight, 1e-6, true).unwrap();
        assert_eq!(out.dims(), &[2, 4, 8]);
    }

    #[test]
    fn test_rms_norm_gated_ones_input() {
        // With all-ones input, x * silu(z) = 1 * 0.731 = 0.731 (approx)
        // After RMSNorm of a constant vector, each element = constant (rms = constant)
        // Output = constant / constant * 1.0 = 1.0
        let device = Device::Cpu;
        let x = Tensor::ones((1, 1, 4), DType::F32, &device).unwrap();
        let z = Tensor::ones((1, 1, 4), DType::F32, &device).unwrap();
        let weight = Tensor::ones(4, DType::F32, &device).unwrap();
        let out = rms_norm_gated(&x, &z, &weight, 1e-6, false).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // constant input → rms_norm outputs 1.0 everywhere
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_gated_zero_gate() {
        // z=0 → silu(0) = 0 → gate=0 → x*gate=0 → rms_norm of zeros returns 0
        let device = Device::Cpu;
        let x = Tensor::ones((1, 1, 4), DType::F32, &device).unwrap();
        let z = Tensor::zeros((1, 1, 4), DType::F32, &device).unwrap();
        let weight = Tensor::ones(4, DType::F32, &device).unwrap();
        let out = rms_norm_gated(&x, &z, &weight, 1e-6, false).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &v in &vals {
            assert!(v.abs() < 1e-4, "expected ~0.0 with zero gate, got {v}");
        }
    }

    #[test]
    fn test_silu_at_zero() {
        let device = Device::Cpu;
        let x = Tensor::zeros(4, DType::F32, &device).unwrap();
        let out = silu(&x).unwrap();
        let vals = out.to_vec1::<f32>().unwrap();
        for &v in &vals {
            assert!(v.abs() < 1e-6, "silu(0) should be 0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_gated_weight_scaling() {
        // With weight=2 and constant input, output should be scaled by 2.
        let device = Device::Cpu;
        let x = Tensor::ones((1, 1, 4), DType::F32, &device).unwrap();
        let z = Tensor::ones((1, 1, 4), DType::F32, &device).unwrap();
        let weight = (Tensor::ones(4, DType::F32, &device).unwrap() * 2.0f64).unwrap();
        let out = rms_norm_gated(&x, &z, &weight, 1e-6, false).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &v in &vals {
            assert!((v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
        }
    }
}
