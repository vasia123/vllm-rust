//! Per-expert LoRA weights for MoE layers.
//!
//! Supports "unpermute-aware" LoRA: LoRA computation happens on tokens already
//! grouped by expert assignment in the fused execution path. The existing
//! scatter-back (unpermute) step handles restoring original token order.
//!
//! ## Weight Layout
//!
//! Each projection (gate/up/down) stores LoRA weights as 3D tensors indexed
//! by expert ID:
//! - lora_a: `[num_experts, rank, input_dim]`
//! - lora_b: `[num_experts, output_dim, rank]`
//!
//! ## Reference
//!
//! Based on vLLM's `FusedMoEWithLoRA` (reference/vllm/vllm/lora/layers/fused_moe.py)
//! which applies per-expert LoRA on permuted tokens within the fused MoE kernel.

use candle_core::{IndexOp, Result, Tensor};

use crate::lora::LoraAdapter;

/// Per-expert LoRA weights for MoE layers (SwiGLU: gate + up + down).
///
/// Each expert has independent LoRA adapters for its three projections:
/// - w1 (gate_proj): hidden → intermediate
/// - w2 (down_proj): intermediate → hidden
/// - w3 (up_proj): hidden → intermediate
#[derive(Debug)]
pub struct MoELoraWeights {
    /// LoRA A for gate_proj: `[num_experts, rank, hidden_size]`
    pub w1_lora_a: Tensor,
    /// LoRA B for gate_proj: `[num_experts, intermediate_size, rank]`
    pub w1_lora_b: Tensor,
    /// LoRA A for down_proj: `[num_experts, rank, intermediate_size]`
    pub w2_lora_a: Tensor,
    /// LoRA B for down_proj: `[num_experts, hidden_size, rank]`
    pub w2_lora_b: Tensor,
    /// LoRA A for up_proj: `[num_experts, rank, hidden_size]`
    pub w3_lora_a: Tensor,
    /// LoRA B for up_proj: `[num_experts, intermediate_size, rank]`
    pub w3_lora_b: Tensor,
    /// Pre-computed scale (alpha / rank).
    pub scale: f32,
    /// Number of experts.
    pub num_experts: usize,
    /// LoRA rank.
    pub rank: usize,
}

impl MoELoraWeights {
    /// Create MoE LoRA weights from per-expert adapter triples.
    ///
    /// Each expert must provide adapters for gate_proj (w1), down_proj (w2),
    /// and up_proj (w3). All adapters must have the same rank and scale.
    ///
    /// # Arguments
    /// * `gate_adapters` - One `LoraAdapter` per expert for gate_proj (w1)
    /// * `down_adapters` - One `LoraAdapter` per expert for down_proj (w2)
    /// * `up_adapters` - One `LoraAdapter` per expert for up_proj (w3)
    pub fn from_adapters(
        gate_adapters: &[LoraAdapter],
        down_adapters: &[LoraAdapter],
        up_adapters: &[LoraAdapter],
    ) -> Result<Self> {
        let num_experts = gate_adapters.len();
        if num_experts == 0 {
            return Err(candle_core::Error::Msg(
                "MoELoraWeights requires at least one expert".to_string(),
            ));
        }
        if down_adapters.len() != num_experts || up_adapters.len() != num_experts {
            return Err(candle_core::Error::Msg(format!(
                "All adapter slices must have same length ({}), got down={}, up={}",
                num_experts,
                down_adapters.len(),
                up_adapters.len()
            )));
        }

        let rank = gate_adapters[0].rank;
        let scale = gate_adapters[0].scale;

        // Stack per-expert weights into 3D tensors
        let w1_a: Vec<Tensor> = gate_adapters.iter().map(|a| a.lora_a.clone()).collect();
        let w1_b: Vec<Tensor> = gate_adapters.iter().map(|a| a.lora_b.clone()).collect();
        let w2_a: Vec<Tensor> = down_adapters.iter().map(|a| a.lora_a.clone()).collect();
        let w2_b: Vec<Tensor> = down_adapters.iter().map(|a| a.lora_b.clone()).collect();
        let w3_a: Vec<Tensor> = up_adapters.iter().map(|a| a.lora_a.clone()).collect();
        let w3_b: Vec<Tensor> = up_adapters.iter().map(|a| a.lora_b.clone()).collect();

        Ok(Self {
            w1_lora_a: Tensor::stack(&w1_a, 0)?,
            w1_lora_b: Tensor::stack(&w1_b, 0)?,
            w2_lora_a: Tensor::stack(&w2_a, 0)?,
            w2_lora_b: Tensor::stack(&w2_b, 0)?,
            w3_lora_a: Tensor::stack(&w3_a, 0)?,
            w3_lora_b: Tensor::stack(&w3_b, 0)?,
            scale,
            num_experts,
            rank,
        })
    }

    /// Create MoE LoRA weights from raw 3D tensors.
    ///
    /// All tensors must have the same first dimension (num_experts).
    #[allow(clippy::too_many_arguments)]
    pub fn from_tensors(
        w1_lora_a: Tensor,
        w1_lora_b: Tensor,
        w2_lora_a: Tensor,
        w2_lora_b: Tensor,
        w3_lora_a: Tensor,
        w3_lora_b: Tensor,
        scale: f32,
        rank: usize,
    ) -> Result<Self> {
        let num_experts = w1_lora_a.dims()[0];

        // Validate all tensors have matching expert count
        for (name, tensor) in [
            ("w1_lora_b", &w1_lora_b),
            ("w2_lora_a", &w2_lora_a),
            ("w2_lora_b", &w2_lora_b),
            ("w3_lora_a", &w3_lora_a),
            ("w3_lora_b", &w3_lora_b),
        ] {
            if tensor.dims()[0] != num_experts {
                return Err(candle_core::Error::Msg(format!(
                    "{} has {} experts, expected {}",
                    name,
                    tensor.dims()[0],
                    num_experts
                )));
            }
        }

        Ok(Self {
            w1_lora_a,
            w1_lora_b,
            w2_lora_a,
            w2_lora_b,
            w3_lora_a,
            w3_lora_b,
            scale,
            num_experts,
            rank,
        })
    }
}

/// Apply LoRA to a single expert's projection on a batch of tokens.
///
/// Computes: `scale * (x @ lora_a[expert_id].T @ lora_b[expert_id].T)`
///
/// # Arguments
/// * `x` - Input tensor `[batch, in_dim]`
/// * `lora_a` - LoRA A weights `[num_experts, rank, in_dim]`
/// * `lora_b` - LoRA B weights `[num_experts, out_dim, rank]`
/// * `expert_id` - Which expert's weights to use
/// * `scale` - LoRA scaling factor
///
/// # Returns
/// LoRA output `[batch, out_dim]`
pub fn apply_expert_lora(
    x: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    expert_id: usize,
    scale: f32,
) -> Result<Tensor> {
    // Extract this expert's LoRA weights
    let a = lora_a.i(expert_id)?; // [rank, in_dim]
    let b = lora_b.i(expert_id)?; // [out_dim, rank]

    // x @ a.T -> [batch, rank]
    let intermediate = x.matmul(&a.t()?)?;
    // intermediate @ b.T -> [batch, out_dim]
    let result = intermediate.matmul(&b.t()?)?;

    // Apply scaling
    if (scale - 1.0).abs() > f32::EPSILON {
        result.affine(scale as f64, 0.0)
    } else {
        Ok(result)
    }
}

/// Forward pass through a single expert with LoRA applied to all projections.
///
/// Computes SwiGLU with per-projection LoRA:
/// ```text
/// gate = gate_proj(x) + lora_w1(x)
/// up   = up_proj(x)   + lora_w3(x)
/// hidden = silu(gate) * up
/// output = down_proj(hidden) + lora_w2(hidden)
/// ```
pub fn expert_forward_with_lora(
    x: &Tensor,
    expert: &super::expert_layer::MoEExpert,
    expert_id: usize,
    lora: &MoELoraWeights,
) -> Result<Tensor> {
    // Gate projection + LoRA
    let gate = candle_nn::Module::forward(&expert.gate_proj, x)?;
    let gate_lora = apply_expert_lora(x, &lora.w1_lora_a, &lora.w1_lora_b, expert_id, lora.scale)?;
    let gate = gate.add(&gate_lora)?;
    let gate = candle_nn::ops::silu(&gate)?;

    // Up projection + LoRA
    let up = candle_nn::Module::forward(&expert.up_proj, x)?;
    let up_lora = apply_expert_lora(x, &lora.w3_lora_a, &lora.w3_lora_b, expert_id, lora.scale)?;
    let up = up.add(&up_lora)?;

    // SwiGLU
    let hidden = gate.mul(&up)?;

    // Down projection + LoRA
    let down = candle_nn::Module::forward(&expert.down_proj, &hidden)?;
    let down_lora =
        apply_expert_lora(&hidden, &lora.w2_lora_a, &lora.w2_lora_b, expert_id, lora.scale)?;
    down.add(&down_lora)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_adapter(rank: usize, in_dim: usize, out_dim: usize, device: &Device) -> LoraAdapter {
        let lora_a = Tensor::randn(0f32, 0.1, (rank, in_dim), device).unwrap();
        let lora_b = Tensor::randn(0f32, 0.1, (out_dim, rank), device).unwrap();
        LoraAdapter::new(lora_a, lora_b, rank, 16.0)
    }

    #[test]
    fn test_moe_lora_weights_from_adapters() {
        let device = Device::Cpu;
        let num_experts = 4;
        let rank = 8;
        let hidden = 64;
        let intermediate = 128;

        let gate: Vec<_> = (0..num_experts)
            .map(|_| make_adapter(rank, hidden, intermediate, &device))
            .collect();
        let down: Vec<_> = (0..num_experts)
            .map(|_| make_adapter(rank, intermediate, hidden, &device))
            .collect();
        let up: Vec<_> = (0..num_experts)
            .map(|_| make_adapter(rank, hidden, intermediate, &device))
            .collect();

        let weights = MoELoraWeights::from_adapters(&gate, &down, &up).unwrap();

        assert_eq!(weights.num_experts, num_experts);
        assert_eq!(weights.rank, rank);
        assert_eq!(weights.w1_lora_a.dims(), &[num_experts, rank, hidden]);
        assert_eq!(weights.w1_lora_b.dims(), &[num_experts, intermediate, rank]);
        assert_eq!(weights.w2_lora_a.dims(), &[num_experts, rank, intermediate]);
        assert_eq!(weights.w2_lora_b.dims(), &[num_experts, hidden, rank]);
        assert_eq!(weights.w3_lora_a.dims(), &[num_experts, rank, hidden]);
        assert_eq!(weights.w3_lora_b.dims(), &[num_experts, intermediate, rank]);
    }

    #[test]
    fn test_moe_lora_weights_from_tensors() {
        let device = Device::Cpu;
        let ne = 4;
        let r = 8;
        let h = 64;
        let im = 128;

        let weights = MoELoraWeights::from_tensors(
            Tensor::zeros((ne, r, h), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, im, r), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, r, im), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, h, r), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, r, h), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, im, r), DType::F32, &device).unwrap(),
            2.0,
            r,
        )
        .unwrap();

        assert_eq!(weights.num_experts, ne);
        assert_eq!(weights.rank, r);
        assert!((weights.scale - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_moe_lora_weights_mismatched_experts() {
        let device = Device::Cpu;
        let rank = 8;
        let hidden = 64;
        let intermediate = 128;

        let gate: Vec<_> = (0..4)
            .map(|_| make_adapter(rank, hidden, intermediate, &device))
            .collect();
        let down: Vec<_> = (0..3) // Wrong count!
            .map(|_| make_adapter(rank, intermediate, hidden, &device))
            .collect();
        let up: Vec<_> = (0..4)
            .map(|_| make_adapter(rank, hidden, intermediate, &device))
            .collect();

        let result = MoELoraWeights::from_adapters(&gate, &down, &up);
        assert!(result.is_err());
    }

    #[test]
    fn test_moe_lora_weights_empty() {
        let gate: Vec<LoraAdapter> = vec![];
        let down: Vec<LoraAdapter> = vec![];
        let up: Vec<LoraAdapter> = vec![];

        let result = MoELoraWeights::from_adapters(&gate, &down, &up);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_expert_lora() {
        let device = Device::Cpu;
        let batch = 4;
        let in_dim = 64;
        let out_dim = 128;
        let rank = 8;
        let num_experts = 4;
        let scale = 2.0;

        let x = Tensor::randn(0f32, 1.0, (batch, in_dim), &device).unwrap();
        let lora_a = Tensor::randn(0f32, 0.1, (num_experts, rank, in_dim), &device).unwrap();
        let lora_b = Tensor::randn(0f32, 0.1, (num_experts, out_dim, rank), &device).unwrap();

        let result = apply_expert_lora(&x, &lora_a, &lora_b, 2, scale).unwrap();
        assert_eq!(result.dims(), &[batch, out_dim]);
    }

    #[test]
    fn test_apply_expert_lora_scale_one() {
        let device = Device::Cpu;
        let batch = 2;
        let in_dim = 16;
        let out_dim = 32;
        let rank = 4;
        let num_experts = 2;

        let x = Tensor::ones((batch, in_dim), DType::F32, &device).unwrap();
        let lora_a = Tensor::ones((num_experts, rank, in_dim), DType::F32, &device).unwrap();
        let lora_b = Tensor::ones((num_experts, out_dim, rank), DType::F32, &device).unwrap();

        // scale=1.0 path
        let result_s1 = apply_expert_lora(&x, &lora_a, &lora_b, 0, 1.0).unwrap();
        // scale=2.0 path
        let result_s2 = apply_expert_lora(&x, &lora_a, &lora_b, 0, 2.0).unwrap();

        // s2 should be 2x s1
        let ratio: Vec<f32> = result_s2
            .div(&result_s1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(ratio.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    }

    #[test]
    fn test_apply_expert_lora_different_experts_differ() {
        let device = Device::Cpu;
        let batch = 3;
        let in_dim = 32;
        let out_dim = 64;
        let rank = 4;
        let num_experts = 4;

        let x = Tensor::randn(0f32, 1.0, (batch, in_dim), &device).unwrap();
        let lora_a = Tensor::randn(0f32, 0.1, (num_experts, rank, in_dim), &device).unwrap();
        let lora_b = Tensor::randn(0f32, 0.1, (num_experts, out_dim, rank), &device).unwrap();

        let result_0 = apply_expert_lora(&x, &lora_a, &lora_b, 0, 1.0).unwrap();
        let result_1 = apply_expert_lora(&x, &lora_a, &lora_b, 1, 1.0).unwrap();

        let diff: f32 = result_0
            .sub(&result_1)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 0.0, "Different experts should produce different LoRA outputs");
    }

    #[test]
    fn test_expert_forward_with_lora() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let rank = 4;
        let num_experts = 4;

        let expert_config = super::super::expert_layer::MoEExpertConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
        };
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let expert = super::super::expert_layer::MoEExpert::new(&expert_config, vb).unwrap();

        let gate: Vec<_> = (0..num_experts)
            .map(|_| make_adapter(rank, hidden, intermediate, &device))
            .collect();
        let down: Vec<_> = (0..num_experts)
            .map(|_| make_adapter(rank, intermediate, hidden, &device))
            .collect();
        let up: Vec<_> = (0..num_experts)
            .map(|_| make_adapter(rank, hidden, intermediate, &device))
            .collect();
        let lora = MoELoraWeights::from_adapters(&gate, &down, &up).unwrap();

        let x = Tensor::randn(0f32, 1.0, (3, hidden), &device).unwrap();

        // Forward with LoRA
        let out_lora = expert_forward_with_lora(&x, &expert, 0, &lora).unwrap();
        assert_eq!(out_lora.dims(), &[3, hidden]);

        // Forward without LoRA (base only)
        let out_base = expert.forward(&x).unwrap();
        assert_eq!(out_base.dims(), &[3, hidden]);

        // LoRA should modify the output (non-zero LoRA weights)
        let diff: f32 = out_lora
            .sub(&out_base)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 0.0, "LoRA should modify expert output");
    }

    #[test]
    fn test_expert_forward_with_lora_zero_weights() {
        let device = Device::Cpu;
        let hidden = 16;
        let intermediate = 32;
        let rank = 4;
        let ne = 4;

        let expert_config = super::super::expert_layer::MoEExpertConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
        };
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let expert = super::super::expert_layer::MoEExpert::new(&expert_config, vb).unwrap();

        // Zero LoRA weights — should produce same output as base
        let lora = MoELoraWeights::from_tensors(
            Tensor::zeros((ne, rank, hidden), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, intermediate, rank), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, rank, intermediate), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, hidden, rank), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, rank, hidden), DType::F32, &device).unwrap(),
            Tensor::zeros((ne, intermediate, rank), DType::F32, &device).unwrap(),
            2.0,
            rank,
        )
        .unwrap();

        let x = Tensor::randn(0f32, 1.0, (3, hidden), &device).unwrap();
        let out_lora = expert_forward_with_lora(&x, &expert, 0, &lora).unwrap();
        let out_base = expert.forward(&x).unwrap();

        let diff: f32 = out_lora
            .sub(&out_base)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff < 1e-6,
            "Zero LoRA weights should not change output, diff={}",
            diff
        );
    }
}
