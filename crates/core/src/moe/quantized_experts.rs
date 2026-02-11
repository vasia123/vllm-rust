//! Quantized MoE expert and layer implementations.
//!
//! Provides `QuantizedMoEExpert` and `QuantizedMoELayer` that mirror the
//! standard `MoEExpert` and `MoELayer` but use `Box<dyn QuantizedLinear>`
//! for expert projections. This enables ExpertsInt8, MoeWNA16, FP8, and
//! any other quantized MoE expert format.
//!
//! Non-MoE components (router, shared expert gate) remain unquantized.

use candle_core::{DType, IndexOp, Result, Tensor};

use super::router::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};
use crate::quantization::QuantizedLinear;

/// A single quantized MoE expert (FFN layer).
///
/// Uses SwiGLU activation: output = down_proj(silu(gate_proj(x)) * up_proj(x))
/// All three projections are quantized via `Box<dyn QuantizedLinear>`.
pub struct QuantizedMoEExpert {
    gate_proj: Box<dyn QuantizedLinear>, // w1
    up_proj: Box<dyn QuantizedLinear>,   // w3
    down_proj: Box<dyn QuantizedLinear>, // w2
}

impl QuantizedMoEExpert {
    /// Create from pre-loaded quantized linear layers.
    pub fn new(
        gate_proj: Box<dyn QuantizedLinear>,
        up_proj: Box<dyn QuantizedLinear>,
        down_proj: Box<dyn QuantizedLinear>,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass through the quantized expert.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(xs)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Configuration for the quantized MoE layer.
#[derive(Debug, Clone)]
pub struct QuantizedMoELayerConfig {
    /// Hidden size.
    pub hidden_size: usize,
    /// Intermediate (FFN) size for each expert.
    pub intermediate_size: usize,
    /// Number of routed experts.
    pub num_experts: usize,
    /// Number of experts to activate per token.
    pub top_k: usize,
    /// Whether to renormalize routing weights.
    pub renormalize: bool,
    /// Scoring function (Softmax or Sigmoid).
    pub scoring_func: ScoringFunc,
    /// Whether to use grouped top-k.
    pub use_grouped_topk: bool,
    /// Number of expert groups for grouped top-k.
    pub num_expert_groups: Option<usize>,
    /// Top-k per group.
    pub topk_per_group: Option<usize>,
    /// Scaling factor for routed expert output.
    pub routed_scaling_factor: f64,
}

impl Default for QuantizedMoELayerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            intermediate_size: 0,
            num_experts: 0,
            top_k: 2,
            renormalize: true,
            scoring_func: ScoringFunc::Softmax,
            use_grouped_topk: false,
            num_expert_groups: None,
            topk_per_group: None,
            routed_scaling_factor: 1.0,
        }
    }
}

/// MoE Layer with quantized experts.
///
/// Mirrors `MoELayer` but expert projections are quantized. The router
/// remains at full precision. Supports:
/// - ExpertsInt8 (online INT8 quantization)
/// - MoeWNA16 (GPTQ/AWQ packed weights)
/// - Any other `Box<dyn QuantizedLinear>` expert implementation
pub struct QuantizedMoELayer {
    router: TopKRouter,
    experts: Vec<QuantizedMoEExpert>,
    config: QuantizedMoELayerConfig,
}

impl QuantizedMoELayer {
    /// Create a new quantized MoE layer.
    ///
    /// The router is loaded from `vb`, but expert weights must be provided
    /// as pre-loaded `QuantizedMoEExpert` instances (since quantized weights
    /// have different loading paths).
    pub fn new(
        config: QuantizedMoELayerConfig,
        router: TopKRouter,
        experts: Vec<QuantizedMoEExpert>,
    ) -> Self {
        Self {
            router,
            experts,
            config,
        }
    }

    /// Create the router from a VarBuilder.
    pub fn create_router(config: &QuantizedMoELayerConfig, vb: candle_nn::VarBuilder) -> Result<TopKRouter> {
        let router_config = RouterConfig {
            hidden_size: config.hidden_size,
            num_experts: config.num_experts,
            top_k: config.top_k,
            renormalize: config.renormalize,
            scoring_func: config.scoring_func,
            use_grouped_topk: config.use_grouped_topk,
            num_expert_groups: config.num_expert_groups,
            topk_per_group: config.topk_per_group,
        };
        TopKRouter::new(router_config, vb)
    }

    /// Forward pass through the quantized MoE layer.
    ///
    /// Automatically selects naive or fused mode based on batch size.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden_states.dims().to_vec();
        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();

        if num_tokens >= 16 {
            self.forward_fused(hidden_states)
        } else {
            self.forward_naive(hidden_states)
        }
    }

    /// Naive forward pass — per-token, per-expert computation.
    pub fn forward_naive(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden_states.dims().to_vec();
        let hidden_size = *orig_shape.last().ok_or_else(|| {
            candle_core::Error::Msg("Tensor must have at least 1 dimension".to_string())
        })?;

        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
        let flat_hidden = hidden_states.reshape((num_tokens, hidden_size))?;

        let (routing_weights, expert_indices) = self.router.route(&flat_hidden)?;

        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

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

            let indices = Tensor::new(&[token_idx as u32], device)?;
            let token_output_2d = token_output.unsqueeze(0)?;
            output = output.index_add(&indices, &token_output_2d, 0)?;
        }

        output.reshape(orig_shape)
    }

    /// Fused forward pass with batched expert execution.
    pub fn forward_fused(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let orig_shape = hidden_states.dims().to_vec();
        let hidden_size = *orig_shape.last().ok_or_else(|| {
            candle_core::Error::Msg("Tensor must have at least 1 dimension".to_string())
        })?;

        let num_tokens: usize = orig_shape.iter().take(orig_shape.len() - 1).product();
        let flat_hidden = hidden_states.reshape((num_tokens, hidden_size))?;

        let (routing_weights, expert_indices) = self.router.route(&flat_hidden)?;

        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

        let expert_indices_vec: Vec<u32> = expert_indices.flatten_all()?.to_vec1()?;
        let routing_weights_vec: Vec<f32> = routing_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;

        // Group tokens by expert
        let mut expert_tokens: Vec<Vec<(usize, usize, f32)>> =
            vec![Vec::new(); self.config.num_experts];

        for token_idx in 0..num_tokens {
            for k in 0..self.config.top_k {
                let flat_idx = token_idx * self.config.top_k + k;
                let expert_id = expert_indices_vec[flat_idx] as usize;
                let weight = routing_weights_vec[flat_idx];
                if expert_id < self.config.num_experts {
                    expert_tokens[expert_id].push((token_idx, k, weight));
                }
            }
        }

        // Process each expert's tokens as a batch
        for (expert_id, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let expert = &self.experts[expert_id];
            let batch_size = tokens.len();

            let mut input_rows = Vec::with_capacity(batch_size);
            for &(token_idx, _, _) in tokens {
                input_rows.push(flat_hidden.i(token_idx)?.unsqueeze(0)?);
            }
            let batch_input = Tensor::cat(&input_rows, 0)?;

            let expert_output = expert.forward(&batch_input)?;

            let weights_vec: Vec<f32> = tokens.iter().map(|(_, _, w)| *w).collect();
            let weights_tensor =
                Tensor::from_vec(weights_vec, batch_size, device)?.to_dtype(dtype)?;
            let weights_expanded = weights_tensor.reshape((batch_size, 1))?;
            let weighted_output = expert_output.broadcast_mul(&weights_expanded)?;

            let indices: Vec<u32> = tokens.iter().map(|(idx, _, _)| *idx as u32).collect();
            let index_tensor = Tensor::from_vec(indices, batch_size, device)?;
            output = output.index_add(&index_tensor, &weighted_output, 0)?;
        }

        // Apply routed scaling factor
        if (self.config.routed_scaling_factor - 1.0).abs() > 1e-9 {
            let scale =
                Tensor::new(&[self.config.routed_scaling_factor as f32], device)?.to_dtype(dtype)?;
            output = output.broadcast_mul(&scale)?;
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::UnquantizedLinear;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    /// Helper: create an unquantized expert (for testing the layer dispatch).
    fn make_unquantized_expert(
        hidden: usize,
        intermediate: usize,
        device: &Device,
    ) -> QuantizedMoEExpert {
        let gate = Box::new(
            UnquantizedLinear::new(hidden, intermediate, false, DType::F32, device).unwrap(),
        );
        let up = Box::new(
            UnquantizedLinear::new(hidden, intermediate, false, DType::F32, device).unwrap(),
        );
        let down = Box::new(
            UnquantizedLinear::new(intermediate, hidden, false, DType::F32, device).unwrap(),
        );
        QuantizedMoEExpert::new(gate, up, down)
    }

    #[test]
    fn test_quantized_moe_expert_forward() {
        let device = Device::Cpu;
        let expert = make_unquantized_expert(16, 32, &device);

        let input = Tensor::randn(0f32, 1.0, (2, 16), &device).unwrap();
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 16]);
    }

    #[test]
    fn test_quantized_moe_layer_creation() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();

        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);
        assert_eq!(layer.num_experts(), 4);
        assert_eq!(layer.top_k(), 2);
    }

    #[test]
    fn test_quantized_moe_layer_forward_naive() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);

        let input = Tensor::randn(0f32, 1.0, (3, hidden), &device).unwrap();
        let output = layer.forward_naive(&input).unwrap();
        assert_eq!(output.dims(), &[3, hidden]);
    }

    #[test]
    fn test_quantized_moe_layer_forward_fused() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);

        let input = Tensor::randn(0f32, 1.0, (8, hidden), &device).unwrap();
        let output = layer.forward_fused(&input).unwrap();
        assert_eq!(output.dims(), &[8, hidden]);
    }

    #[test]
    fn test_quantized_moe_naive_vs_fused_equivalence() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);

        let input = Tensor::randn(0f32, 1.0, (4, hidden), &device).unwrap();
        let naive_out = layer.forward_naive(&input).unwrap();
        let fused_out = layer.forward_fused(&input).unwrap();

        let diff: f32 = naive_out
            .sub(&fused_out)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-4, "Naive vs fused diff: {diff}");
    }

    #[test]
    fn test_quantized_moe_layer_3d_input() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);

        let input = Tensor::randn(0f32, 1.0, (2, 3, hidden), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 3, hidden]);
    }

    #[test]
    fn test_quantized_moe_layer_scaling_factor() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            routed_scaling_factor: 0.5,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);

        let input = Tensor::randn(0f32, 1.0, (4, hidden), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, hidden]);
    }

    #[test]
    fn test_quantized_moe_layer_various_batch_sizes() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let num_experts = 4;
        let top_k = 2;

        let config = QuantizedMoELayerConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_experts,
            top_k,
            renormalize: true,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let router = QuantizedMoELayer::create_router(&config, vb.pp("gate")).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|_| make_unquantized_expert(hidden, intermediate, &device))
            .collect();

        let layer = QuantizedMoELayer::new(config, router, experts);

        for batch_size in [1, 7, 16, 32, 64] {
            let input = Tensor::randn(0f32, 1.0, (batch_size, hidden), &device).unwrap();
            let output = layer.forward(&input).unwrap();
            assert_eq!(
                output.dims(),
                &[batch_size, hidden],
                "Failed for batch_size={batch_size}"
            );
        }
    }

    #[test]
    fn test_quantized_moe_layer_config_default() {
        let config = QuantizedMoELayerConfig::default();
        assert_eq!(config.routed_scaling_factor, 1.0);
        assert!(!config.use_grouped_topk);
        assert_eq!(config.top_k, 2);
    }

    #[test]
    fn test_quantized_moe_expert_with_int8() {
        use crate::quantization::experts_int8::ExpertsInt8Linear;
        use std::collections::HashMap;

        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;

        // Create INT8 expert
        let mut gate =
            ExpertsInt8Linear::new(hidden, intermediate, false, DType::F32, &device).unwrap();
        let mut up =
            ExpertsInt8Linear::new(hidden, intermediate, false, DType::F32, &device).unwrap();
        let mut down =
            ExpertsInt8Linear::new(intermediate, hidden, false, DType::F32, &device).unwrap();

        // Load weights (simulating online quantization)
        let w_gate = Tensor::randn(0f32, 1.0, (intermediate, hidden), &device).unwrap();
        let w_up = Tensor::randn(0f32, 1.0, (intermediate, hidden), &device).unwrap();
        let w_down = Tensor::randn(0f32, 1.0, (hidden, intermediate), &device).unwrap();

        gate.load_weights(&HashMap::from([("weight".into(), w_gate)]))
            .unwrap();
        up.load_weights(&HashMap::from([("weight".into(), w_up)]))
            .unwrap();
        down.load_weights(&HashMap::from([("weight".into(), w_down)]))
            .unwrap();

        let expert = QuantizedMoEExpert::new(Box::new(gate), Box::new(up), Box::new(down));

        let input = Tensor::randn(0f32, 1.0, (3, hidden), &device).unwrap();
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.dims(), &[3, hidden]);
    }
}
