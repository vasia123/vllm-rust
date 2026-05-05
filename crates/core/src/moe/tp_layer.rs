//! Tensor-parallel MoE layers — TP variants of [`MoEExpert`],
//! [`MoELayer`], and [`MoELayerWithShared`] for models whose bespoke
//! MoE shards each expert across TP ranks (Bailing-MoE, ERNIE 4.5-MoE,
//! GPT-OSS, etc.).
//!
//! ## When to use
//!
//! - Use [`MoELayer`] when the model is single-GPU or expert weights are
//!   not sharded across ranks.
//! - Use [`EPMoELayer`] when each expert lives entirely on one rank
//!   (expert parallelism).
//! - Use [`TpMoELayer`] / [`TpMoELayerWithShared`] when each expert is
//!   itself sharded across ranks (intra-expert tensor parallelism).
//!
//! ## Sharding scheme
//!
//! Mirrors the dense-MLP TP sharding: `gate_proj` and `up_proj` are
//! column-parallel (split intermediate dim), `down_proj` is row-parallel
//! (reduces partial outputs across ranks). The router `gate` is small
//! (`hidden -> num_experts`) and stays replicated.
//!
//! ## Forward pass
//!
//! Mirrors [`MoELayer::forward_fused`]: route tokens, group by expert,
//! batch each expert's tokens through one TP-aware `gate_up`/`down`
//! pass, scatter results back. The naive per-token path is preserved
//! for `num_tokens < 16`.
//!
//! [`MoELayer`]: super::expert_layer::MoELayer
//! [`MoEExpert`]: super::expert_layer::MoEExpert
//! [`MoELayerWithShared`]: super::expert_layer::MoELayerWithShared
//! [`EPMoELayer`]: super::ep_layer::EPMoELayer

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::distributed::ProcessGroup;
use crate::models::tp_layers::{TpContext, TpLinear, TpSwiGluMlp};

use super::router::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};

/// Configuration for [`TpMoEExpert`].
#[derive(Debug, Clone)]
pub struct TpMoEExpertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

/// A single TP-aware MoE expert (`gate/up/down` SwiGLU MLP using TpLinear).
pub struct TpMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl TpMoEExpert {
    /// Construct an expert. Weight paths are Mixtral-style `w1/w2/w3`
    /// (matching [`super::expert_layer::MoEExpert`]); column-parallel
    /// for gate/up, row-parallel for down.
    pub fn new(config: &TpMoEExpertConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let gate_proj = TpLinear::column_parallel(
            config.hidden_size,
            config.intermediate_size,
            false,
            false,
            vb.pp("w1"),
            pg,
        )?;
        let up_proj = TpLinear::column_parallel(
            config.hidden_size,
            config.intermediate_size,
            false,
            false,
            vb.pp("w3"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            config.intermediate_size,
            config.hidden_size,
            false,
            true,
            vb.pp("w2"),
            pg,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward through the expert: `down(silu(gate(x)) * up(x))`.
    pub fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs, tp_ctx)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

/// Configuration for [`TpMoELayer`].
#[derive(Debug, Clone)]
pub struct TpMoELayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub renormalize: bool,
    /// Routing scoring function; default `Softmax`. Set `Sigmoid` for
    /// sigmoid-routed models (Bailing-MoE in sigmoid mode, ERNIE
    /// 4.5-MoE, GLM4-MoE, DeepSeek V3).
    pub scoring_func: ScoringFunc,
    pub use_grouped_topk: bool,
    pub num_expert_groups: Option<usize>,
    pub topk_per_group: Option<usize>,
    pub routed_scaling_factor: f64,
}

impl Default for TpMoELayerConfig {
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

/// TP-aware MoE layer.
///
/// Same forward shape as [`super::expert_layer::MoELayer`] but threads
/// `&TpContext` through expert calls and uses [`TpMoEExpert`] internally.
/// The router stays non-TP (it's a small `hidden -> num_experts` projection).
pub struct TpMoELayer {
    router: TopKRouter,
    experts: Vec<TpMoEExpert>,
    config: TpMoELayerConfig,
}

impl TpMoELayer {
    pub fn new(config: TpMoELayerConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        Self::new_inner(config, vb, pg, None)
    }

    /// Constructor that wires `e_score_correction_bias` into the router.
    /// Used by sigmoid-routed models (DeepSeek V3, GLM4-MoE, ERNIE
    /// 4.5-MoE) where the bias is loaded as a `[num_experts]` tensor
    /// from the checkpoint.
    pub fn new_with_router_bias(
        config: TpMoELayerConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        Self::new_inner(config, vb, pg, bias)
    }

    fn new_inner(
        config: TpMoELayerConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let router_cfg = RouterConfig {
            hidden_size: config.hidden_size,
            num_experts: config.num_experts,
            top_k: config.top_k,
            renormalize: config.renormalize,
            scoring_func: config.scoring_func,
            use_grouped_topk: config.use_grouped_topk,
            num_expert_groups: config.num_expert_groups,
            topk_per_group: config.topk_per_group,
            routed_scaling_factor: config.routed_scaling_factor,
        };
        let router = TopKRouter::new_with_bias(router_cfg, vb.pp("gate"), bias)?;

        let expert_cfg = TpMoEExpertConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        };
        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            experts.push(TpMoEExpert::new(
                &expert_cfg,
                vb.pp(format!("experts.{}", i)),
                pg,
            )?);
        }

        Ok(Self {
            router,
            experts,
            config,
        })
    }

    /// Forward: route → group tokens by expert → batched TP expert
    /// forward → scatter back, with optional `routed_scaling_factor`
    /// applied to the routed output.
    pub fn forward(&self, hidden_states: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
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

        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.config.num_experts];

        for token_idx in 0..num_tokens {
            for k in 0..self.config.top_k {
                let flat_idx = token_idx * self.config.top_k + k;
                let expert_id = expert_indices_vec[flat_idx] as usize;
                let weight = routing_weights_vec[flat_idx];
                if expert_id < self.config.num_experts {
                    expert_tokens[expert_id].push((token_idx, weight));
                }
            }
        }

        for (expert_id, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let batch_size = tokens.len();
            let mut input_rows = Vec::with_capacity(batch_size);
            for &(token_idx, _) in tokens {
                input_rows.push(flat_hidden.i(token_idx)?.unsqueeze(0)?);
            }
            let batch_input = Tensor::cat(&input_rows, 0)?;
            let expert_output = self.experts[expert_id].forward(&batch_input, tp_ctx)?;

            let weights_vec: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();
            let weights_tensor =
                Tensor::from_vec(weights_vec, batch_size, device)?.to_dtype(dtype)?;
            let weights_expanded = weights_tensor.reshape((batch_size, 1))?;
            let weighted_output = expert_output.broadcast_mul(&weights_expanded)?;

            let indices: Vec<u32> = tokens.iter().map(|(idx, _)| *idx as u32).collect();
            let index_tensor = Tensor::from_vec(indices, batch_size, device)?;
            output = output.index_add(&index_tensor, &weighted_output, 0)?;
        }

        if (self.config.routed_scaling_factor - 1.0).abs() > 1e-9 {
            let scale = Tensor::new(&[self.config.routed_scaling_factor as f32], device)?
                .to_dtype(dtype)?;
            output = output.broadcast_mul(&scale)?;
        }

        output.reshape(orig_shape)
    }

    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.config.top_k
    }
}

/// Configuration for [`TpMoELayerWithShared`].
#[derive(Debug, Clone)]
pub struct TpMoELayerWithSharedConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Intermediate size for the shared expert. When `Some`, a shared
    /// MLP is loaded under `vb.pp("shared_experts")` (Bailing-MoE,
    /// DeepSeek family naming) and added to the routed output.
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_experts: usize,
    pub top_k: usize,
    pub renormalize: bool,
    pub scoring_func: ScoringFunc,
    pub use_grouped_topk: bool,
    pub num_expert_groups: Option<usize>,
    pub topk_per_group: Option<usize>,
    pub routed_scaling_factor: f64,
    /// VarBuilder sub-path for the shared expert. Defaults to
    /// `"shared_experts"` (Bailing-MoE, DeepSeek). Some checkpoints use
    /// `"shared_expert"` (Qwen2-MoE, GLM4-MoE) — set this field to
    /// match the checkpoint.
    pub shared_expert_path: &'static str,
}

impl Default for TpMoELayerWithSharedConfig {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            intermediate_size: 0,
            shared_expert_intermediate_size: None,
            num_experts: 0,
            top_k: 2,
            renormalize: true,
            scoring_func: ScoringFunc::Softmax,
            use_grouped_topk: false,
            num_expert_groups: None,
            topk_per_group: None,
            routed_scaling_factor: 1.0,
            shared_expert_path: "shared_experts",
        }
    }
}

/// TP-aware MoE layer with optional shared expert.
///
/// Same shape as [`super::expert_layer::MoELayerWithShared`] but
/// threads `TpContext` through and uses [`TpMoEExpert`] +
/// [`TpSwiGluMlp`]-shaped shared expert.
pub struct TpMoELayerWithShared {
    inner: TpMoELayer,
    shared_expert: Option<TpSwiGluMlp>,
}

impl TpMoELayerWithShared {
    pub fn new(
        config: TpMoELayerWithSharedConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        Self::new_inner(config, vb, pg, None)
    }

    pub fn new_with_router_bias(
        config: TpMoELayerWithSharedConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        Self::new_inner(config, vb, pg, bias)
    }

    fn new_inner(
        config: TpMoELayerWithSharedConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let shared_expert =
            if let Some(shared_intermediate) = config.shared_expert_intermediate_size {
                Some(TpSwiGluMlp::new(
                    config.hidden_size,
                    shared_intermediate,
                    vb.pp(config.shared_expert_path),
                    pg,
                )?)
            } else {
                None
            };

        let inner_cfg = TpMoELayerConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_experts: config.num_experts,
            top_k: config.top_k,
            renormalize: config.renormalize,
            scoring_func: config.scoring_func,
            use_grouped_topk: config.use_grouped_topk,
            num_expert_groups: config.num_expert_groups,
            topk_per_group: config.topk_per_group,
            routed_scaling_factor: config.routed_scaling_factor,
        };
        let inner = TpMoELayer::new_with_router_bias(inner_cfg, vb, pg, bias)?;

        Ok(Self {
            inner,
            shared_expert,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let routed = self.inner.forward(hidden_states, tp_ctx)?;
        match &self.shared_expert {
            Some(shared) => {
                let orig_shape = hidden_states.dims().to_vec();
                let hidden_size = *orig_shape.last().unwrap();
                let flat = hidden_states.reshape(((), hidden_size))?;
                let shared_out = shared.forward(&flat, tp_ctx)?;
                let shared_out = shared_out.reshape(orig_shape)?;
                routed + shared_out
            }
            None => Ok(routed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::LocalProcessGroup;
    use candle_core::Device;

    #[test]
    fn tp_moe_expert_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let cfg = TpMoEExpertConfig {
            hidden_size: 64,
            intermediate_size: 128,
        };
        let expert = TpMoEExpert::new(&cfg, vb, &pg).unwrap();
        let xs = Tensor::zeros((4, 64), DType::F32, &device).unwrap();
        let out = expert.forward(&xs, &TpContext::single_gpu()).unwrap();
        assert_eq!(out.dims(), &[4, 64]);
    }

    #[test]
    fn tp_moe_layer_forward_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let cfg = TpMoELayerConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_experts: 4,
            top_k: 2,
            renormalize: true,
            scoring_func: ScoringFunc::Softmax,
            use_grouped_topk: false,
            num_expert_groups: None,
            topk_per_group: None,
            routed_scaling_factor: 1.0,
        };
        let layer = TpMoELayer::new(cfg, vb, &pg).unwrap();
        let xs = Tensor::zeros((1, 8, 64), DType::F32, &device).unwrap();
        let out = layer.forward(&xs, &TpContext::single_gpu()).unwrap();
        assert_eq!(out.dims(), &[1, 8, 64]);
    }

    #[test]
    fn tp_moe_with_shared_forward_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let cfg = TpMoELayerWithSharedConfig {
            hidden_size: 64,
            intermediate_size: 128,
            shared_expert_intermediate_size: Some(64),
            num_experts: 4,
            top_k: 2,
            renormalize: true,
            scoring_func: ScoringFunc::Sigmoid,
            use_grouped_topk: false,
            num_expert_groups: None,
            topk_per_group: None,
            routed_scaling_factor: 2.0,
            shared_expert_path: "shared_experts",
        };
        let layer = TpMoELayerWithShared::new(cfg, vb, &pg).unwrap();
        let xs = Tensor::zeros((1, 4, 64), DType::F32, &device).unwrap();
        let out = layer.forward(&xs, &TpContext::single_gpu()).unwrap();
        assert_eq!(out.dims(), &[1, 4, 64]);
    }
}
