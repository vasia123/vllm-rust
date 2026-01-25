use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// SwiGLU MLP used by Llama, Qwen3, Mistral, and others.
pub struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluMlp {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for SwiGluMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(xs)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    fn create_swiglu_mlp(
        hidden_size: usize,
        intermediate_size: usize,
        device: &Device,
    ) -> Result<SwiGluMlp> {
        let vb = VarBuilder::zeros(DType::F32, device);
        SwiGluMlp::new(hidden_size, intermediate_size, vb)
    }

    #[test]
    fn test_swiglu_forward_shape() {
        let device = Device::Cpu;
        let hidden_size = 256;
        let intermediate_size = 512;
        let batch_size = 4;
        let seq_len = 16;

        let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
            .expect("Failed to create SwiGluMlp");

        let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)
            .expect("Failed to create input");

        let output = mlp.forward(&x).expect("Forward failed");

        // Output should have same shape as input: [batch, seq, hidden_size]
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_swiglu_forward_shape_2d() {
        // Test with 2D input [batch, hidden]
        let device = Device::Cpu;
        let hidden_size = 128;
        let intermediate_size = 256;
        let batch_size = 8;

        let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
            .expect("Failed to create SwiGluMlp");

        let x = Tensor::randn(0.0f32, 1.0, (batch_size, hidden_size), &device)
            .expect("Failed to create input");

        let output = mlp.forward(&x).expect("Forward failed");

        assert_eq!(output.dims(), &[batch_size, hidden_size]);
    }

    #[test]
    fn test_silu_activation_formula() {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let device = Device::Cpu;

        let x = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], (5,), &device)
            .expect("Failed to create input");

        let silu_result = x.apply(&candle_nn::Activation::Silu).expect("SiLU failed");
        let silu_data: Vec<f32> = silu_result.to_vec1().expect("to_vec1 failed");
        let x_data: Vec<f32> = x.to_vec1().expect("to_vec1 failed");

        // Manually compute SiLU
        for (xi, silu_xi) in x_data.iter().zip(silu_data.iter()) {
            let expected = xi / (1.0 + (-xi).exp());
            assert!(
                (expected - silu_xi).abs() < 1e-5,
                "SiLU mismatch: input={xi}, expected={expected}, got={silu_xi}"
            );
        }
    }

    #[test]
    fn test_silu_properties() {
        // SiLU properties:
        // - silu(0) = 0
        // - silu(x) ≈ 0 for large negative x
        // - silu(x) ≈ x for large positive x
        let device = Device::Cpu;

        let x = Tensor::from_vec(vec![0.0f32, -10.0, 10.0], (3,), &device)
            .expect("Failed to create input");

        let silu_result = x.apply(&candle_nn::Activation::Silu).expect("SiLU failed");
        let silu_data: Vec<f32> = silu_result.to_vec1().expect("to_vec1 failed");

        // silu(0) ≈ 0
        assert!(
            silu_data[0].abs() < 1e-6,
            "silu(0) should be 0, got {}",
            silu_data[0]
        );

        // silu(-10) ≈ 0
        assert!(
            silu_data[1].abs() < 0.001,
            "silu(-10) should be ≈0, got {}",
            silu_data[1]
        );

        // silu(10) ≈ 10
        assert!(
            (silu_data[2] - 10.0).abs() < 0.001,
            "silu(10) should be ≈10, got {}",
            silu_data[2]
        );
    }

    #[test]
    fn test_swiglu_with_zero_weights() {
        // With zero weights, output should be zero
        let device = Device::Cpu;
        let hidden_size = 64;
        let intermediate_size = 128;

        let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
            .expect("Failed to create SwiGluMlp");

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, hidden_size), &device)
            .expect("Failed to create input");

        let output = mlp.forward(&x).expect("Forward failed");
        let output_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        // All outputs should be zero
        assert!(
            output_data.iter().all(|&v| v.abs() < 1e-6),
            "Output should be all zeros with zero weights"
        );
    }

    #[test]
    fn test_swiglu_different_intermediate_sizes() {
        // Test standard Llama-style ratio: intermediate_size ≈ 2.67 * hidden_size
        let device = Device::Cpu;

        let test_cases = [
            (256, 683),   // Small model
            (512, 1365),  // Medium
            (1024, 2731), // Larger
        ];

        for (hidden_size, intermediate_size) in test_cases {
            let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
                .expect("Failed to create SwiGluMlp");

            let x = Tensor::randn(0.0f32, 1.0, (1, 1, hidden_size), &device)
                .expect("Failed to create input");

            let output = mlp.forward(&x).expect("Forward failed");

            assert_eq!(
                output.dims(),
                &[1, 1, hidden_size],
                "Shape mismatch for hidden={hidden_size}, intermediate={intermediate_size}"
            );
        }
    }

    #[test]
    fn test_swiglu_output_finite() {
        let device = Device::Cpu;
        let hidden_size = 128;
        let intermediate_size = 256;

        let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
            .expect("Failed to create SwiGluMlp");

        // Test with various input magnitudes
        for scale in [0.1, 1.0, 10.0] {
            let x = Tensor::randn(0.0f32, scale, (2, 4, hidden_size), &device)
                .expect("Failed to create input");

            let output = mlp.forward(&x).expect("Forward failed");
            let output_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

            assert!(
                output_data.iter().all(|v| v.is_finite()),
                "Output should be finite for scale={scale}"
            );
        }
    }

    #[test]
    fn test_swiglu_preserves_dtype() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let intermediate_size = 128;

        let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
            .expect("Failed to create SwiGluMlp");

        let x = Tensor::randn(0.0f32, 1.0, (1, 1, hidden_size), &device)
            .expect("Failed to create input");

        let output = mlp.forward(&x).expect("Forward failed");

        assert_eq!(output.dtype(), x.dtype());
    }

    #[test]
    fn test_swiglu_single_token() {
        // Test decode phase: single token
        let device = Device::Cpu;
        let hidden_size = 256;
        let intermediate_size = 512;
        let batch_size = 4;

        let mlp = create_swiglu_mlp(hidden_size, intermediate_size, &device)
            .expect("Failed to create SwiGluMlp");

        // [batch, 1, hidden] for decode
        let x = Tensor::randn(0.0f32, 1.0, (batch_size, 1, hidden_size), &device)
            .expect("Failed to create input");

        let output = mlp.forward(&x).expect("Forward failed");

        assert_eq!(output.dims(), &[batch_size, 1, hidden_size]);
    }
}
