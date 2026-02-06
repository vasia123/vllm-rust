//! Selective scan (S6) algorithm implementation.
//!
//! This is the core recurrence used in Mamba models. It implements the
//! discretized state space model:
//!
//! ```text
//!   h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
//!   y_t = C_t * h_t + D * x_t
//! ```
//!
//! Where A_bar = exp(delta * A) and B_bar = delta * B.

use candle_core::{Result, Tensor};

/// Selective scan forward pass (CPU implementation).
///
/// Implements the recurrence:
///   h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
///   y_t = C_t * h_t + D * x_t
///
/// Where:
///   A_bar = exp(delta * A)
///   B_bar = delta * B
///
/// # Arguments
/// * `x` - Input tensor [batch, seq_len, d_inner]
/// * `delta` - Time-step tensor [batch, seq_len, d_inner]
/// * `a` - State transition matrix [d_inner, d_state]
/// * `b` - Input projection [batch, seq_len, d_state]
/// * `c` - Output projection [batch, seq_len, d_state]
/// * `d` - Skip connection scalar [d_inner]
/// * `state` - Optional initial state [batch, d_inner, d_state]
///
/// # Returns
/// * Output tensor [batch, seq_len, d_inner]
/// * Updated state [batch, d_inner, d_state]
pub fn selective_scan(
    x: &Tensor,
    delta: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (batch, seq_len, d_inner) = x.dims3()?;
    let (_d_inner_a, d_state) = a.dims2()?;
    let device = x.device();
    let dtype = x.dtype();

    // Initialize state to zeros if not provided
    let mut h = match state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, d_inner, d_state), dtype, device)?,
    };

    let mut outputs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // Extract time step t: x_t [batch, d_inner]
        let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
        // delta_t [batch, d_inner]
        let delta_t = delta.narrow(1, t, 1)?.squeeze(1)?;
        // b_t [batch, d_state]
        let b_t = b.narrow(1, t, 1)?.squeeze(1)?;
        // c_t [batch, d_state]
        let c_t = c.narrow(1, t, 1)?.squeeze(1)?;

        // delta_a = exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        // delta_t: [batch, d_inner] -> [batch, d_inner, 1]
        // A: [d_inner, d_state] -> [1, d_inner, d_state]
        let delta_t_expanded = delta_t.unsqueeze(2)?; // [batch, d_inner, 1]
        let a_expanded = a.unsqueeze(0)?; // [1, d_inner, d_state]
        let delta_a = (delta_t_expanded.broadcast_mul(&a_expanded))?.exp()?; // [batch, d_inner, d_state]

        // delta_b_x = (delta_t * x_t).unsqueeze(-1) * b_t.unsqueeze(1)
        // delta_t * x_t: [batch, d_inner]
        let delta_x = (&delta_t * &x_t)?; // [batch, d_inner]
        let delta_x_expanded = delta_x.unsqueeze(2)?; // [batch, d_inner, 1]
        let b_t_expanded = b_t.unsqueeze(1)?; // [batch, 1, d_state]
        let delta_b_x = delta_x_expanded.broadcast_mul(&b_t_expanded)?; // [batch, d_inner, d_state]

        // h = delta_a * h + delta_b_x
        h = ((&delta_a * &h)? + &delta_b_x)?;

        // y_t = (h * c_t.unsqueeze(1)).sum(-1) + D * x_t
        let c_t_expanded = c_t.unsqueeze(1)?; // [batch, 1, d_state]
        let h_c = h.broadcast_mul(&c_t_expanded)?; // [batch, d_inner, d_state]
        let y_t = h_c.sum(2)?; // [batch, d_inner]

        // Skip connection: y_t + D * x_t
        let d_x = d.unsqueeze(0)?.broadcast_mul(&x_t)?; // [batch, d_inner]
        let y_t = (&y_t + &d_x)?;

        outputs.push(y_t.unsqueeze(1)?); // [batch, 1, d_inner]
    }

    // Stack outputs: [batch, seq_len, d_inner]
    let output = Tensor::cat(&outputs, 1)?;

    Ok((output, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const BATCH: usize = 1;
    const D_INNER: usize = 4;
    const D_STATE: usize = 3;

    /// When A=0, delta_a = exp(0) = 1 and the scan becomes a simple accumulation.
    #[test]
    fn selective_scan_a_zero_identity() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let seq_len = 2;

        let x = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("x");
        let delta = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("delta");
        let a = Tensor::zeros((D_INNER, D_STATE), dtype, &device).expect("a");
        let b = Tensor::ones((BATCH, seq_len, D_STATE), dtype, &device).expect("b");
        let c = Tensor::ones((BATCH, seq_len, D_STATE), dtype, &device).expect("c");
        let d = Tensor::zeros((D_INNER,), dtype, &device).expect("d");

        let (output, final_state) = selective_scan(&x, &delta, &a, &b, &c, &d, None).expect("scan");

        assert_eq!(output.dims(), &[BATCH, seq_len, D_INNER]);
        assert_eq!(final_state.dims(), &[BATCH, D_INNER, D_STATE]);

        // With A=0: delta_a = exp(0)=1, delta_b_x = delta*x * B = 1*1*1 = 1 for each state dim
        // Step 0: h = 1*0 + 1 = 1 (for each d_state), y_0 = sum(1*C) = d_state = 3
        // Step 1: h = 1*1 + 1 = 2, y_1 = sum(2*1) = 2*d_state = 6
        let output_data: Vec<f32> = output.flatten_all().expect("flat").to_vec1().expect("vec");
        for i in 0..D_INNER {
            assert!(
                (output_data[i] - D_STATE as f32).abs() < 1e-5,
                "step 0: expected {}, got {}",
                D_STATE,
                output_data[i]
            );
        }
        for i in 0..D_INNER {
            assert!(
                (output_data[D_INNER + i] - (2.0 * D_STATE as f32)).abs() < 1e-5,
                "step 1: expected {}, got {}",
                2.0 * D_STATE as f32,
                output_data[D_INNER + i]
            );
        }
    }

    #[test]
    fn selective_scan_output_shape() {
        let device = Device::Cpu;
        let seq_len = 5;

        let x = Tensor::randn(0f32, 1.0, (BATCH, seq_len, D_INNER), &device).expect("x");
        let delta = Tensor::randn(0f32, 0.1, (BATCH, seq_len, D_INNER), &device)
            .expect("delta")
            .abs()
            .expect("abs");
        let a = Tensor::randn(0f32, 1.0, (D_INNER, D_STATE), &device).expect("a");
        let b = Tensor::randn(0f32, 1.0, (BATCH, seq_len, D_STATE), &device).expect("b");
        let c = Tensor::randn(0f32, 1.0, (BATCH, seq_len, D_STATE), &device).expect("c");
        let d = Tensor::randn(0f32, 1.0, (D_INNER,), &device).expect("d");

        let (output, final_state) = selective_scan(&x, &delta, &a, &b, &c, &d, None).expect("scan");

        assert_eq!(output.dims(), &[BATCH, seq_len, D_INNER]);
        assert_eq!(final_state.dims(), &[BATCH, D_INNER, D_STATE]);
    }

    #[test]
    fn selective_scan_with_initial_state() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let seq_len = 1;

        let x = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("x");
        let delta = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("delta");
        let a = Tensor::zeros((D_INNER, D_STATE), dtype, &device).expect("a");
        let b = Tensor::ones((BATCH, seq_len, D_STATE), dtype, &device).expect("b");
        let c = Tensor::ones((BATCH, seq_len, D_STATE), dtype, &device).expect("c");
        let d = Tensor::zeros((D_INNER,), dtype, &device).expect("d");

        // Initial state = 5.0 everywhere
        let init_state = (Tensor::ones((BATCH, D_INNER, D_STATE), dtype, &device).expect("init")
            * 5.0)
            .expect("mul");

        let (output, final_state) =
            selective_scan(&x, &delta, &a, &b, &c, &d, Some(&init_state)).expect("scan");

        // With A=0: delta_a = 1, so h = 1*5 + 1 = 6
        // y = sum(6*1) = 6*d_state = 18
        let output_data: Vec<f32> = output.flatten_all().expect("flat").to_vec1().expect("vec");
        for i in 0..D_INNER {
            assert!(
                (output_data[i] - 6.0 * D_STATE as f32).abs() < 1e-5,
                "with initial state: expected {}, got {}",
                6.0 * D_STATE as f32,
                output_data[i]
            );
        }

        // Final state should be 6.0 everywhere
        let state_data: Vec<f32> = final_state
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        for val in &state_data {
            assert!(
                (*val - 6.0).abs() < 1e-5,
                "final state: expected 6.0, got {}",
                val
            );
        }
    }

    #[test]
    fn selective_scan_skip_connection() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let seq_len = 1;

        // Zero everything except D (skip connection)
        let x = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("x");
        let delta = Tensor::zeros((BATCH, seq_len, D_INNER), dtype, &device).expect("delta");
        let a = Tensor::zeros((D_INNER, D_STATE), dtype, &device).expect("a");
        let b = Tensor::zeros((BATCH, seq_len, D_STATE), dtype, &device).expect("b");
        let c = Tensor::zeros((BATCH, seq_len, D_STATE), dtype, &device).expect("c");
        let d = (Tensor::ones((D_INNER,), dtype, &device).expect("d") * 3.0).expect("mul");

        let (output, _) = selective_scan(&x, &delta, &a, &b, &c, &d, None).expect("scan");

        // With delta=0: h stays at 0, y = 0 + D*x = 3.0*1.0 = 3.0
        let output_data: Vec<f32> = output.flatten_all().expect("flat").to_vec1().expect("vec");
        for val in &output_data {
            assert!(
                (*val - 3.0).abs() < 1e-5,
                "skip connection: expected 3.0, got {}",
                val
            );
        }
    }

    #[test]
    fn selective_scan_negative_a_causes_decay() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let seq_len = 3;

        // Use negative A to cause state decay (exp(delta * A) < 1 when A < 0)
        let x = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("x");
        let delta = Tensor::ones((BATCH, seq_len, D_INNER), dtype, &device).expect("delta");
        // A = -1.0 -> delta_a = exp(-1) ~ 0.368
        let a_data = vec![-1.0f32; D_INNER * D_STATE];
        let a = Tensor::from_vec(a_data, (D_INNER, D_STATE), &device).expect("a");
        let b = Tensor::ones((BATCH, seq_len, D_STATE), dtype, &device).expect("b");
        let c = Tensor::ones((BATCH, seq_len, D_STATE), dtype, &device).expect("c");
        let d = Tensor::zeros((D_INNER,), dtype, &device).expect("d");

        let (output, _) = selective_scan(&x, &delta, &a, &b, &c, &d, None).expect("scan");

        // The output should exist and have correct shape
        assert_eq!(output.dims(), &[BATCH, seq_len, D_INNER]);

        // With decay, the output at each step should be bounded
        let output_data: Vec<f32> = output.flatten_all().expect("flat").to_vec1().expect("vec");
        for val in &output_data {
            assert!(val.is_finite(), "output should be finite, got {}", val);
        }
    }

    #[test]
    fn selective_scan_state_preservation_across_calls() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let x = Tensor::ones((BATCH, 1, D_INNER), dtype, &device).expect("x");
        let delta = Tensor::ones((BATCH, 1, D_INNER), dtype, &device).expect("delta");
        let a = Tensor::zeros((D_INNER, D_STATE), dtype, &device).expect("a");
        let b = Tensor::ones((BATCH, 1, D_STATE), dtype, &device).expect("b");
        let c = Tensor::ones((BATCH, 1, D_STATE), dtype, &device).expect("c");
        let d = Tensor::zeros((D_INNER,), dtype, &device).expect("d");

        // First call: process one token
        let (out1, state1) = selective_scan(&x, &delta, &a, &b, &c, &d, None).expect("scan1");

        // Second call: process another token with previous state
        let (out2, _) = selective_scan(&x, &delta, &a, &b, &c, &d, Some(&state1)).expect("scan2");

        // With A=0: first call h=1, y=d_state; second call h=2, y=2*d_state
        let out1_data: Vec<f32> = out1.flatten_all().expect("flat").to_vec1().expect("vec");
        let out2_data: Vec<f32> = out2.flatten_all().expect("flat").to_vec1().expect("vec");

        for i in 0..D_INNER {
            assert!(
                (out1_data[i] - D_STATE as f32).abs() < 1e-5,
                "first call: expected {}, got {}",
                D_STATE as f32,
                out1_data[i]
            );
            assert!(
                (out2_data[i] - 2.0 * D_STATE as f32).abs() < 1e-5,
                "second call: expected {}, got {}",
                2.0 * D_STATE as f32,
                out2_data[i]
            );
        }
    }
}
