//! MoE Expert Layer implementation.
//!
//! Provides expert FFN layers and the fused MoE execution layer.

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::router::{MoERouter, RouterConfig, TopKRouter};

/// Configuration for a single MoE expert.
#[derive(Debug, Clone)]
pub struct MoEExpertConfig {
    /// Hidden size.
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
}

/// A single MoE expert (FFN layer).
///
/// Uses SwiGLU activation: output = down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct MoEExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MoEExpert {
    /// Create a new MoE expert.
    pub fn new(config: &MoEExpertConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("w1"))?;
        let up_proj =
            candle_nn::linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("w3"))?;
        let down_proj =
            candle_nn::linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("w2"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass through the expert.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(xs)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Configuration for the MoE layer.
#[derive(Debug, Clone)]
pub struct MoELayerConfig {
    /// Hidden size.
    pub hidden_size: usize,
    /// Intermediate (FFN) size for each expert.
    pub intermediate_size: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// Number of experts to activate per token.
    pub top_k: usize,
    /// Whether to renormalize routing weights.
    pub renormalize: bool,
}

/// MoE Layer that combines routing and expert execution.
///
/// For each token, routes to top-k experts and combines their outputs
/// weighted by the routing probabilities.
pub struct MoELayer {
    router: TopKRouter,
    experts: Vec<MoEExpert>,
    config: MoELayerConfig,
}

impl MoELayer {
    /// Create a new MoE layer.
    pub fn new(config: MoELayerConfig, vb: VarBuilder) -> Result<Self> {
        let router_config = RouterConfig {
            hidden_size: config.hidden_size,
            num_experts: config.num_experts,
            top_k: config.top_k,
            renormalize: config.renormalize,
        };

        let router = TopKRouter::new(router_config, vb.pp("gate"))?;

        let expert_config = MoEExpertConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        };

        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            let expert = MoEExpert::new(&expert_config, vb.pp(format!("experts.{}", i)))?;
            experts.push(expert);
        }

        Ok(Self {
            router,
            experts,
            config,
        })
    }

    /// Forward pass through the MoE layer.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `[num_tokens, hidden_size]` or `[batch, seq, hidden_size]`
    ///
    /// # Returns
    /// Output tensor of same shape as input.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden_states.dims().to_vec();
        let hidden_size = *orig_shape.last().unwrap();

        // Flatten to [num_tokens, hidden_size] for routing
        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
        let flat_hidden = hidden_states.reshape((num_tokens, hidden_size))?;

        // Route tokens to experts
        let (routing_weights, expert_indices) = self.router.route(&flat_hidden)?;

        // Initialize output tensor
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

        // Process each token
        // This is a naive implementation - production would use fused kernels
        for token_idx in 0..num_tokens {
            let token_hidden = flat_hidden.i(token_idx)?;
            let token_weights: Vec<f32> = routing_weights
                .i(token_idx)?
                .to_dtype(DType::F32)?
                .to_vec1()?;
            let token_experts: Vec<u32> = expert_indices.i(token_idx)?.to_vec1()?;

            let mut token_output = Tensor::zeros(hidden_size, dtype, device)?;

            for (k, &expert_idx) in token_experts.iter().enumerate() {
                let expert = &self.experts[expert_idx as usize];
                let expert_out = expert.forward(&token_hidden.unsqueeze(0)?)?.squeeze(0)?;

                let weight = token_weights[k];
                let weight_tensor = Tensor::new(&[weight], device)?.to_dtype(dtype)?;
                let weighted = expert_out.broadcast_mul(&weight_tensor)?;
                token_output = token_output.add(&weighted)?;
            }

            // Update output tensor at token_idx
            output = scatter_add_1d(&output, token_idx, &token_output)?;
        }

        // Reshape back to original shape
        output.reshape(orig_shape)
    }

    /// Get number of experts.
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    /// Get top-k value.
    pub fn top_k(&self) -> usize {
        self.config.top_k
    }
}

/// Helper to update a single row in a 2D tensor.
fn scatter_add_1d(tensor: &Tensor, row_idx: usize, values: &Tensor) -> Result<Tensor> {
    let (num_rows, _hidden_size) = tensor.dims2()?;

    // Build a new tensor with the updated row
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
    fn test_moe_expert_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = MoEExpertConfig {
            hidden_size: 64,
            intermediate_size: 128,
        };

        let expert = MoEExpert::new(&config, vb).unwrap();
        let input = Tensor::randn(0f32, 1.0, (2, 64), &device).unwrap();
        let output = expert.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 64]);
    }

    #[test]
    fn test_moe_layer_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = MoELayerConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_experts: 4,
            top_k: 2,
            renormalize: true,
        };

        let moe = MoELayer::new(config, vb).unwrap();
        assert_eq!(moe.num_experts(), 4);
        assert_eq!(moe.top_k(), 2);
    }

    #[test]
    fn test_moe_layer_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = MoELayerConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 4,
            top_k: 2,
            renormalize: true,
        };

        let moe = MoELayer::new(config, vb).unwrap();

        // Test with 2D input [num_tokens, hidden_size]
        let input_2d = Tensor::randn(0f32, 1.0, (3, 16), &device).unwrap();
        let output_2d = moe.forward(&input_2d).unwrap();
        assert_eq!(output_2d.dims(), &[3, 16]);

        // Test with 3D input [batch, seq, hidden_size]
        let input_3d = Tensor::randn(0f32, 1.0, (2, 3, 16), &device).unwrap();
        let output_3d = moe.forward(&input_3d).unwrap();
        assert_eq!(output_3d.dims(), &[2, 3, 16]);
    }

    #[test]
    fn test_moe_layer_deterministic() {
        let device = Device::Cpu;

        // Create two MoE layers with same weights (zeros)
        let vb1 = VarBuilder::zeros(DType::F32, &device);
        let vb2 = VarBuilder::zeros(DType::F32, &device);

        let config = MoELayerConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 4,
            top_k: 2,
            renormalize: true,
        };

        let moe1 = MoELayer::new(config.clone(), vb1).unwrap();
        let moe2 = MoELayer::new(config, vb2).unwrap();

        // Same input should give same output
        let input = Tensor::new(&[[1.0f32; 16]], &device).unwrap();
        let out1 = moe1.forward(&input).unwrap();
        let out2 = moe2.forward(&input).unwrap();

        let diff: f32 = out1
            .sub(&out2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-5);
    }
}
