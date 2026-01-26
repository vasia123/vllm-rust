//! Linear layer with LoRA adapter support.

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use super::types::LoraAdapter;

/// Linear layer wrapper that supports LoRA adapters.
///
/// The forward computation is:
/// ```text
/// output = base_linear(x) + scale * (x @ lora_a.T @ lora_b.T)
/// ```
///
/// Multiple adapters can be registered, and the adapter to use is selected
/// per-forward pass via the `adapter_name` parameter.
pub struct LinearWithLora {
    /// Base linear layer (frozen during LoRA fine-tuning).
    base: Linear,
    /// Registered LoRA adapters, keyed by adapter name.
    adapters: HashMap<String, LoraAdapter>,
}

impl LinearWithLora {
    /// Create a new LinearWithLora from an existing Linear layer.
    pub fn from_linear(base: Linear) -> Self {
        Self {
            base,
            adapters: HashMap::new(),
        }
    }

    /// Create a new LinearWithLora from VarBuilder.
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let base = linear_no_bias(in_features, out_features, vb)?;
        Ok(Self::from_linear(base))
    }

    /// Register a LoRA adapter.
    ///
    /// The adapter will be available for use in forward passes via its name.
    pub fn register_adapter(&mut self, name: impl Into<String>, adapter: LoraAdapter) {
        self.adapters.insert(name.into(), adapter);
    }

    /// Remove a registered adapter.
    pub fn remove_adapter(&mut self, name: &str) -> Option<LoraAdapter> {
        self.adapters.remove(name)
    }

    /// Check if an adapter is registered.
    pub fn has_adapter(&self, name: &str) -> bool {
        self.adapters.contains_key(name)
    }

    /// Get list of registered adapter names.
    pub fn adapter_names(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered adapters.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Forward pass with optional LoRA adapter.
    ///
    /// If `adapter_name` is provided and the adapter exists, applies LoRA.
    /// Otherwise, returns base linear output.
    pub fn forward_with_lora(&self, x: &Tensor, adapter_name: Option<&str>) -> Result<Tensor> {
        // Base linear output
        let output = self.base.forward(x)?;

        // Apply LoRA if adapter specified and exists
        if let Some(name) = adapter_name {
            if let Some(adapter) = self.adapters.get(name) {
                let lora_output = self.apply_lora(x, adapter)?;
                return output.add(&lora_output);
            }
        }

        Ok(output)
    }

    /// Apply LoRA computation: scale * (x @ lora_a.T @ lora_b.T)
    fn apply_lora(&self, x: &Tensor, adapter: &LoraAdapter) -> Result<Tensor> {
        // x: [..., in_features]
        // lora_a: [rank, in_features]
        // lora_b: [out_features, rank]
        //
        // Computation:
        // intermediate = x @ lora_a.T  -> [..., rank]
        // output = intermediate @ lora_b.T -> [..., out_features]
        // scaled_output = scale * output

        // Handle both 2D and 3D inputs
        let x_dims = x.dims();
        let is_3d = x_dims.len() == 3;

        let x_2d = if is_3d {
            // Reshape [batch, seq, features] to [batch*seq, features]
            let batch = x_dims[0];
            let seq = x_dims[1];
            let features = x_dims[2];
            x.reshape((batch * seq, features))?
        } else {
            x.clone()
        };

        // lora_a.T: [in_features, rank]
        let lora_a_t = adapter.lora_a.t()?;
        // lora_b.T: [rank, out_features]
        let lora_b_t = adapter.lora_b.t()?;

        // x_2d @ lora_a.T -> [batch*seq, rank]
        let intermediate = x_2d.matmul(&lora_a_t)?;
        // intermediate @ lora_b.T -> [batch*seq, out_features]
        let lora_output = intermediate.matmul(&lora_b_t)?;

        // Apply scaling
        let scaled = if (adapter.scale - 1.0).abs() > f32::EPSILON {
            lora_output.affine(adapter.scale as f64, 0.0)?
        } else {
            lora_output
        };

        // Reshape back to 3D if needed
        if is_3d {
            let batch = x_dims[0];
            let seq = x_dims[1];
            let out_features = scaled.dims()[1];
            scaled.reshape((batch, seq, out_features))
        } else {
            Ok(scaled)
        }
    }

    /// Get reference to the base linear layer.
    pub fn base(&self) -> &Linear {
        &self.base
    }

    /// Get the weight tensor from the base layer.
    pub fn weight(&self) -> &Tensor {
        self.base.weight()
    }

    /// Get the bias tensor from the base layer (if any).
    pub fn bias(&self) -> Option<&Tensor> {
        self.base.bias()
    }
}

impl Module for LinearWithLora {
    /// Standard forward pass (no LoRA).
    ///
    /// This implements the Module trait for compatibility.
    /// For LoRA-enabled forward, use `forward_with_lora`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.base.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_test_linear(
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> LinearWithLora {
        let vb = VarBuilder::zeros(DType::F32, device);
        LinearWithLora::new(in_features, out_features, vb).unwrap()
    }

    #[test]
    fn test_linear_with_lora_no_adapter() {
        let device = Device::Cpu;
        let in_features = 64;
        let out_features = 32;

        let layer = create_test_linear(in_features, out_features, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, in_features), &device).unwrap();
        let output = layer.forward_with_lora(&x, None).unwrap();

        assert_eq!(output.dims(), &[2, 8, out_features]);
    }

    #[test]
    fn test_linear_with_lora_adapter() {
        let device = Device::Cpu;
        let in_features = 64;
        let out_features = 32;
        let rank = 8;

        let mut layer = create_test_linear(in_features, out_features, &device);

        // Create LoRA adapter with non-zero weights
        let lora_a = Tensor::randn(0.0f32, 0.1, (rank, in_features), &device).unwrap();
        let lora_b = Tensor::randn(0.0f32, 0.1, (out_features, rank), &device).unwrap();
        let adapter = LoraAdapter::new(lora_a, lora_b, rank, 16.0);

        layer.register_adapter("test", adapter);
        assert!(layer.has_adapter("test"));
        assert_eq!(layer.num_adapters(), 1);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, in_features), &device).unwrap();

        // Forward with LoRA
        let output_with_lora = layer.forward_with_lora(&x, Some("test")).unwrap();
        assert_eq!(output_with_lora.dims(), &[2, 8, out_features]);

        // Forward without LoRA
        let output_without_lora = layer.forward_with_lora(&x, None).unwrap();
        assert_eq!(output_without_lora.dims(), &[2, 8, out_features]);

        // Outputs should be different (LoRA adds to base output)
        let diff = (&output_with_lora - &output_without_lora)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0, "LoRA should modify the output");
    }

    #[test]
    fn test_linear_with_lora_nonexistent_adapter() {
        let device = Device::Cpu;
        let layer = create_test_linear(64, 32, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();

        // Forward with non-existent adapter should return base output
        let output = layer.forward_with_lora(&x, Some("nonexistent")).unwrap();
        let base_output = layer.forward(&x).unwrap();

        let diff = (&output - &base_output)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < f32::EPSILON,
            "Non-existent adapter should return base output"
        );
    }

    #[test]
    fn test_linear_with_lora_multiple_adapters() {
        let device = Device::Cpu;
        let in_features = 64;
        let out_features = 32;
        let rank = 8;

        let mut layer = create_test_linear(in_features, out_features, &device);

        // Register two different adapters
        let lora_a1 = Tensor::randn(0.0f32, 0.1, (rank, in_features), &device).unwrap();
        let lora_b1 = Tensor::randn(0.0f32, 0.1, (out_features, rank), &device).unwrap();
        layer.register_adapter("adapter1", LoraAdapter::new(lora_a1, lora_b1, rank, 16.0));

        let lora_a2 = Tensor::randn(0.0f32, 0.2, (rank, in_features), &device).unwrap();
        let lora_b2 = Tensor::randn(0.0f32, 0.2, (out_features, rank), &device).unwrap();
        layer.register_adapter("adapter2", LoraAdapter::new(lora_a2, lora_b2, rank, 32.0));

        assert_eq!(layer.num_adapters(), 2);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, in_features), &device).unwrap();

        let output1 = layer.forward_with_lora(&x, Some("adapter1")).unwrap();
        let output2 = layer.forward_with_lora(&x, Some("adapter2")).unwrap();

        // Different adapters should produce different outputs
        let diff = (&output1 - &output2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 0.0,
            "Different adapters should produce different outputs"
        );
    }

    #[test]
    fn test_linear_with_lora_remove_adapter() {
        let device = Device::Cpu;
        let mut layer = create_test_linear(64, 32, &device);

        let lora_a = Tensor::zeros((8, 64), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((32, 8), DType::F32, &device).unwrap();
        layer.register_adapter("test", LoraAdapter::new(lora_a, lora_b, 8, 16.0));

        assert!(layer.has_adapter("test"));

        let removed = layer.remove_adapter("test");
        assert!(removed.is_some());
        assert!(!layer.has_adapter("test"));
        assert_eq!(layer.num_adapters(), 0);
    }

    #[test]
    fn test_linear_with_lora_adapter_names() {
        let device = Device::Cpu;
        let mut layer = create_test_linear(64, 32, &device);

        let lora_a = Tensor::zeros((8, 64), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((32, 8), DType::F32, &device).unwrap();

        layer.register_adapter(
            "alpha",
            LoraAdapter::new(lora_a.clone(), lora_b.clone(), 8, 16.0),
        );
        layer.register_adapter("beta", LoraAdapter::new(lora_a, lora_b, 8, 16.0));

        let names = layer.adapter_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn test_linear_with_lora_scale() {
        let device = Device::Cpu;
        let in_features = 16;
        let out_features = 8;
        let rank = 4;

        let mut layer = create_test_linear(in_features, out_features, &device);

        // Create adapter with scale = 2.0 (alpha=8, rank=4)
        let lora_a = Tensor::ones((rank, in_features), DType::F32, &device).unwrap();
        let lora_b = Tensor::ones((out_features, rank), DType::F32, &device).unwrap();
        let adapter = LoraAdapter::new(lora_a, lora_b, rank, 8.0);

        assert!((adapter.scale - 2.0).abs() < f32::EPSILON);
        layer.register_adapter("scaled", adapter);

        // With ones weights, output should be scaled
        let x = Tensor::ones((1, 1, in_features), DType::F32, &device).unwrap();
        let output = layer.forward_with_lora(&x, Some("scaled")).unwrap();

        // Expected LoRA contribution:
        // x @ lora_a.T @ lora_b.T * scale
        // = [1,1,...] @ [16,4] @ [4,8] * 2
        // = [1*16]*4 @ [4,8] * 2 = [64]*8 * 2 = [128]*8
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(output_vec.iter().all(|&v| (v - 128.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_linear_with_lora_2d_input() {
        let device = Device::Cpu;
        let in_features = 32;
        let out_features = 16;
        let rank = 4;

        let mut layer = create_test_linear(in_features, out_features, &device);

        let lora_a = Tensor::randn(0.0f32, 0.1, (rank, in_features), &device).unwrap();
        let lora_b = Tensor::randn(0.0f32, 0.1, (out_features, rank), &device).unwrap();
        layer.register_adapter("test", LoraAdapter::new(lora_a, lora_b, rank, 16.0));

        // 2D input: [batch, features]
        let x = Tensor::randn(0.0f32, 1.0, (4, in_features), &device).unwrap();
        let output = layer.forward_with_lora(&x, Some("test")).unwrap();

        assert_eq!(output.dims(), &[4, out_features]);
    }

    #[test]
    fn test_module_trait() {
        let device = Device::Cpu;
        let layer = create_test_linear(64, 32, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();

        // Module::forward should work and return base output
        let output = layer.forward(&x).unwrap();
        assert_eq!(output.dims(), &[2, 8, 32]);
    }
}
