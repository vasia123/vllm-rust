use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// SwiGLU MLP used by Llama, Qwen3, Mistral, and others.
///
/// Implements the SwiGLU activation function: SwiGLU(x) = silu(gate_proj(x)) * up_proj(x)
/// followed by a down projection.
///
/// When the `cuda-fused-activations` feature is enabled and the input is on a CUDA device,
/// the silu activation and element-wise multiplication are fused into a single kernel,
/// saving memory bandwidth.
pub struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// Whether to use the fused CUDA kernel when available
    use_fused_kernel: bool,
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
            use_fused_kernel: true,
        })
    }

    /// Create a new SwiGluMlp with explicit control over fused kernel usage.
    ///
    /// # Arguments
    /// - `hidden_size`: Input/output dimension
    /// - `intermediate_size`: Intermediate (expanded) dimension
    /// - `vb`: Variable builder for loading weights
    /// - `use_fused_kernel`: Whether to use fused CUDA kernel when available
    pub fn new_with_config(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        use_fused_kernel: bool,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            use_fused_kernel,
        })
    }

    /// Enable or disable the fused CUDA kernel.
    pub fn set_use_fused_kernel(&mut self, use_fused: bool) {
        self.use_fused_kernel = use_fused;
    }

    /// Check if fused kernel is enabled and available for the given tensor.
    #[cfg(feature = "cuda-fused-activations")]
    fn should_use_fused(&self, tensor: &Tensor) -> bool {
        self.use_fused_kernel && crate::cuda_kernels::fused_swiglu_available(tensor)
    }

    #[cfg(not(feature = "cuda-fused-activations"))]
    #[allow(dead_code)]
    fn should_use_fused(&self, _tensor: &Tensor) -> bool {
        false
    }
}

impl Module for SwiGluMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;

        // Use fused kernel if available and enabled
        #[cfg(feature = "cuda-fused-activations")]
        let activation_result = if self.should_use_fused(&gate) {
            crate::cuda_kernels::fused_swiglu(&gate, &up)?
        } else {
            let gate_silu = gate.apply(&candle_nn::Activation::Silu)?;
            (gate_silu * up)?
        };

        #[cfg(not(feature = "cuda-fused-activations"))]
        let activation_result = {
            let gate_silu = gate.apply(&candle_nn::Activation::Silu)?;
            (gate_silu * up)?
        };

        activation_result.apply(&self.down_proj)
    }
}

/// Standalone fused SwiGLU function for use outside of the MLP module.
///
/// Computes: silu(gate) * up
///
/// When the `cuda-fused-activations` feature is enabled and tensors are on CUDA,
/// this uses a fused kernel. Otherwise falls back to standard operations.
#[cfg(feature = "cuda-fused-activations")]
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if crate::cuda_kernels::fused_swiglu_available(gate) {
        crate::cuda_kernels::fused_swiglu(gate, up)
    } else {
        let gate_silu = gate.apply(&candle_nn::Activation::Silu)?;
        gate_silu * up
    }
}

#[cfg(not(feature = "cuda-fused-activations"))]
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let gate_silu = gate.apply(&candle_nn::Activation::Silu)?;
    gate_silu * up
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

    #[test]
    fn test_fused_swiglu_correctness() {
        // Test that fused_swiglu produces the same result as unfused
        let device = Device::Cpu;
        let hidden_size = 128;
        let batch_size = 4;
        let seq_len = 8;

        let gate = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)
            .expect("Failed to create gate");
        let up = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)
            .expect("Failed to create up");

        // Fused version
        let fused_result = fused_swiglu(&gate, &up).expect("fused_swiglu failed");

        // Unfused version: silu(gate) * up
        let unfused_result = {
            let gate_silu = gate
                .apply(&candle_nn::Activation::Silu)
                .expect("silu failed");
            (gate_silu * &up).expect("mul failed")
        };

        // Compare results
        let fused_data: Vec<f32> = fused_result.flatten_all().unwrap().to_vec1().unwrap();
        let unfused_data: Vec<f32> = unfused_result.flatten_all().unwrap().to_vec1().unwrap();

        for (f, u) in fused_data.iter().zip(unfused_data.iter()) {
            assert!(
                (f - u).abs() < 1e-5,
                "Fused vs unfused mismatch: fused={f}, unfused={u}"
            );
        }
    }

    #[test]
    fn test_fused_swiglu_shapes() {
        let device = Device::Cpu;

        // Test various shapes as Vec<usize>
        let test_cases: Vec<Vec<usize>> = vec![
            vec![4, 128],      // 2D
            vec![2, 8, 256],   // 3D small
            vec![1, 512, 512], // 3D larger
        ];

        for dims in test_cases {
            let gate =
                Tensor::zeros(&dims[..], DType::F32, &device).expect("Failed to create gate");
            let up = Tensor::zeros(&dims[..], DType::F32, &device).expect("Failed to create up");

            let result = fused_swiglu(&gate, &up).expect("fused_swiglu failed");

            assert_eq!(result.dims(), &dims[..], "Shape mismatch for {dims:?}");
        }
    }

    #[test]
    fn test_mlp_fused_vs_unfused() {
        // Test that MLP produces same results with fused enabled/disabled
        let device = Device::Cpu;
        let hidden_size = 64;
        let intermediate_size = 128;

        let vb = VarBuilder::zeros(DType::F32, &device);

        // Create two MLPs with same weights but different fused settings
        let mut mlp_fused = SwiGluMlp::new(hidden_size, intermediate_size, vb.clone())
            .expect("Failed to create fused MLP");
        mlp_fused.set_use_fused_kernel(true);

        let mut mlp_unfused = SwiGluMlp::new(hidden_size, intermediate_size, vb)
            .expect("Failed to create unfused MLP");
        mlp_unfused.set_use_fused_kernel(false);

        let x = Tensor::randn(0.0f32, 1.0, (2, 4, hidden_size), &device)
            .expect("Failed to create input");

        let fused_out = mlp_fused.forward(&x).expect("Fused forward failed");
        let unfused_out = mlp_unfused.forward(&x).expect("Unfused forward failed");

        let fused_data: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();
        let unfused_data: Vec<f32> = unfused_out.flatten_all().unwrap().to_vec1().unwrap();

        for (f, u) in fused_data.iter().zip(unfused_data.iter()) {
            assert!(
                (f - u).abs() < 1e-5,
                "MLP fused vs unfused mismatch: fused={f}, unfused={u}"
            );
        }
    }
}

// ============================================================================
// Benchmarking utilities
// ============================================================================

/// Benchmark configuration for SwiGLU performance testing.
#[derive(Debug, Clone)]
pub struct SwiGluBenchConfig {
    /// Number of tokens (batch * seq_len)
    pub num_tokens: usize,
    /// Hidden size (intermediate dimension after projection)
    pub hidden_size: usize,
    /// Number of warmup iterations
    pub warmup_iters: usize,
    /// Number of benchmark iterations
    pub bench_iters: usize,
}

impl Default for SwiGluBenchConfig {
    fn default() -> Self {
        Self {
            num_tokens: 4096,
            hidden_size: 4096, // Typical for 7B models
            warmup_iters: 10,
            bench_iters: 100,
        }
    }
}

/// Benchmark result for SwiGLU operations.
#[derive(Debug, Clone)]
pub struct SwiGluBenchResult {
    /// Average time per iteration in microseconds
    pub avg_time_us: f64,
    /// Minimum time in microseconds
    pub min_time_us: f64,
    /// Maximum time in microseconds
    pub max_time_us: f64,
    /// Throughput in GB/s (based on memory traffic)
    pub throughput_gbps: f64,
}

/// Run a simple benchmark comparing fused vs unfused SwiGLU.
///
/// Returns (fused_result, unfused_result) if benchmarking was successful.
/// This is a basic timing utility - for production benchmarks use proper
/// profiling tools like Nsight Systems.
#[cfg(feature = "cuda-fused-activations")]
pub fn benchmark_swiglu(
    config: &SwiGluBenchConfig,
    device: &candle_core::Device,
) -> Result<(SwiGluBenchResult, SwiGluBenchResult)> {
    use std::time::Instant;

    let dtype = candle_core::DType::BF16;

    // Create test tensors
    let gate = Tensor::randn(0.0f32, 1.0, (config.num_tokens, config.hidden_size), device)?
        .to_dtype(dtype)?;
    let up = Tensor::randn(0.0f32, 1.0, (config.num_tokens, config.hidden_size), device)?
        .to_dtype(dtype)?;

    // Memory traffic: read gate + read up + write output = 3 * num_tokens * hidden_size * 2 bytes
    let bytes_per_iter = 3 * config.num_tokens * config.hidden_size * 2;

    // Benchmark fused version
    let mut fused_times = Vec::with_capacity(config.bench_iters);

    // Warmup
    for _ in 0..config.warmup_iters {
        let _ = crate::cuda_kernels::fused_swiglu(&gate, &up)?;
    }

    // Benchmark
    for _ in 0..config.bench_iters {
        let start = Instant::now();
        let _ = crate::cuda_kernels::fused_swiglu(&gate, &up)?;
        // NOTE: This doesn't sync CUDA - for accurate timing use cuda events
        fused_times.push(start.elapsed().as_micros() as f64);
    }

    let fused_result = compute_bench_result(&fused_times, bytes_per_iter);

    // Benchmark unfused version
    let mut unfused_times = Vec::with_capacity(config.bench_iters);

    // Warmup
    for _ in 0..config.warmup_iters {
        let gate_silu = gate.apply(&candle_nn::Activation::Silu)?;
        let _ = (&gate_silu * &up)?;
    }

    // Benchmark
    for _ in 0..config.bench_iters {
        let start = Instant::now();
        let gate_silu = gate.apply(&candle_nn::Activation::Silu)?;
        let _ = (&gate_silu * &up)?;
        unfused_times.push(start.elapsed().as_micros() as f64);
    }

    let unfused_result = compute_bench_result(&unfused_times, bytes_per_iter);

    Ok((fused_result, unfused_result))
}

#[cfg(feature = "cuda-fused-activations")]
fn compute_bench_result(times: &[f64], bytes_per_iter: usize) -> SwiGluBenchResult {
    let avg_time_us = times.iter().sum::<f64>() / times.len() as f64;
    let min_time_us = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time_us = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Throughput: bytes / time = GB/s
    // bytes_per_iter is in bytes, avg_time_us is in microseconds
    // GB/s = (bytes / 1e9) / (time_us / 1e6) = bytes / (time_us * 1000)
    let throughput_gbps = (bytes_per_iter as f64) / (avg_time_us * 1000.0);

    SwiGluBenchResult {
        avg_time_us,
        min_time_us,
        max_time_us,
        throughput_gbps,
    }
}
