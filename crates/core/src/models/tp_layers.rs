//! Tensor parallelism abstractions for model layers.
//!
//! This module provides unified wrappers for linear and embedding layers
//! that work in both single-GPU and tensor-parallel modes.
//!
//! # Design
//!
//! Following vLLM's pattern:
//! - Column-parallel layers split output dimension (QKV projections, gate/up in MLP)
//! - Row-parallel layers split input dimension (O projection, down in MLP)
//! - VocabParallel embeddings split vocabulary across GPUs
//!
//! # Usage
//!
//! ```ignore
//! use vllm_core::models::tp_layers::{TpContext, TpLinear};
//!
//! // Single GPU (backward compatible)
//! let tp_ctx = TpContext::single_gpu();
//!
//! // Multi-GPU
//! let tp_ctx = TpContext::new(communicator);
//! ```

use std::sync::Arc;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::distributed::{
    ColumnParallelLinear, DeviceCommunicator, LocalProcessGroup, MockCommunicator, ProcessGroup,
    RowParallelLinear, VocabParallelEmbedding,
};

// ─── TP Context ───────────────────────────────────────────────────────────────

/// Context for tensor parallelism.
///
/// Holds the communicator for collective operations and provides
/// convenience methods for checking TP configuration.
pub struct TpContext {
    /// Communicator for collective operations (all_reduce, all_gather, etc.)
    pub communicator: Arc<dyn DeviceCommunicator>,
    /// Total number of GPUs in the TP group
    pub world_size: usize,
    /// This GPU's rank in the TP group
    pub rank: usize,
}

impl TpContext {
    /// Create TP context from a communicator.
    pub fn new(communicator: Arc<dyn DeviceCommunicator>) -> Self {
        let pg = communicator.process_group();
        let world_size = pg.world_size();
        let rank = pg.rank();
        Self {
            communicator,
            world_size,
            rank,
        }
    }

    /// Create a single-GPU context (no actual parallelism).
    ///
    /// All collective operations become identity functions.
    pub fn single_gpu() -> Self {
        let pg = LocalProcessGroup::new();
        let comm = Arc::new(MockCommunicator::new(pg));
        Self {
            communicator: comm,
            world_size: 1,
            rank: 0,
        }
    }

    /// Create a mock multi-GPU context for testing.
    ///
    /// Uses MockCommunicator which simulates collective operations without
    /// actual multi-GPU communication. Useful for testing TP logic.
    ///
    /// # Arguments
    /// * `rank` - This "GPU's" rank
    /// * `world_size` - Total number of simulated GPUs
    #[cfg(test)]
    pub fn mock_multi_gpu(rank: usize, world_size: usize) -> Self {
        let pg = LocalProcessGroup::with_rank(rank, world_size);
        let comm = Arc::new(MockCommunicator::new(pg));
        Self {
            communicator: comm,
            world_size,
            rank,
        }
    }

    /// Whether this is single-GPU (no actual TP).
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Get a reference to the underlying communicator.
    pub fn comm(&self) -> &dyn DeviceCommunicator {
        self.communicator.as_ref()
    }
}

// ─── Linear Layers ────────────────────────────────────────────────────────────

/// Unified linear layer that can be either regular or tensor-parallel.
///
/// In TP mode:
/// - ColumnParallel: splits output dimension across GPUs
/// - RowParallel: splits input dimension across GPUs
pub enum TpLinear {
    /// Regular linear layer (single GPU).
    Regular(Linear),
    /// Column-parallel linear (splits output dim, used for QKV, gate/up).
    ColumnParallel(ColumnParallelLinear),
    /// Row-parallel linear (splits input dim, used for O, down).
    RowParallel(RowParallelLinear),
}

impl TpLinear {
    /// Create a TpLinear from an existing Linear (for tied weights).
    pub fn from_linear(linear: Linear) -> Self {
        TpLinear::Regular(linear)
    }

    /// Create a column-parallel linear layer.
    ///
    /// For single GPU, creates a regular linear layer.
    /// For multi-GPU, creates a ColumnParallelLinear.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension (not split)
    /// * `out_features` - Output dimension (split across TP)
    /// * `bias` - Whether to include bias
    /// * `gather_output` - Whether to all_gather output
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for TP
    pub fn column_parallel(
        in_features: usize,
        out_features: usize,
        bias: bool,
        gather_output: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        if pg.world_size() == 1 {
            let linear = if bias {
                candle_nn::linear(in_features, out_features, vb)?
            } else {
                candle_nn::linear_no_bias(in_features, out_features, vb)?
            };
            Ok(TpLinear::Regular(linear))
        } else {
            Ok(TpLinear::ColumnParallel(ColumnParallelLinear::new(
                in_features,
                out_features,
                bias,
                gather_output,
                vb,
                pg,
            )?))
        }
    }

    /// Create a row-parallel linear layer.
    ///
    /// For single GPU, creates a regular linear layer.
    /// For multi-GPU, creates a RowParallelLinear.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension (split across TP)
    /// * `out_features` - Output dimension (not split)
    /// * `bias` - Whether to include bias
    /// * `input_is_parallel` - Whether input comes from column-parallel
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for TP
    pub fn row_parallel(
        in_features: usize,
        out_features: usize,
        bias: bool,
        input_is_parallel: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        if pg.world_size() == 1 {
            let linear = if bias {
                candle_nn::linear(in_features, out_features, vb)?
            } else {
                candle_nn::linear_no_bias(in_features, out_features, vb)?
            };
            Ok(TpLinear::Regular(linear))
        } else {
            Ok(TpLinear::RowParallel(RowParallelLinear::new(
                in_features,
                out_features,
                bias,
                input_is_parallel,
                vb,
                pg,
            )?))
        }
    }

    /// Forward pass.
    ///
    /// For TP layers, uses the provided communicator for collective operations.
    pub fn forward(&self, input: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            TpLinear::Regular(linear) => linear.forward(input),
            TpLinear::ColumnParallel(layer) => layer
                .forward(input, tp_ctx.comm())
                .map_err(|e| candle_core::Error::Msg(format!("TP column parallel: {e}"))),
            TpLinear::RowParallel(layer) => layer
                .forward(input, tp_ctx.comm())
                .map_err(|e| candle_core::Error::Msg(format!("TP row parallel: {e}"))),
        }
    }
}

// ─── Embedding ────────────────────────────────────────────────────────────────

/// Unified embedding that can be regular or vocabulary-parallel.
///
/// In TP mode, the vocabulary is split across GPUs, with each GPU
/// handling a portion of the token embeddings.
pub enum TpEmbedding {
    /// Regular embedding (single GPU).
    Regular(Embedding),
    /// Vocabulary-parallel embedding (splits vocab across GPUs).
    VocabParallel(VocabParallelEmbedding),
}

impl TpEmbedding {
    /// Create an embedding layer.
    ///
    /// For single GPU, creates a regular embedding.
    /// For multi-GPU, creates a VocabParallelEmbedding.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        if pg.world_size() == 1 {
            Ok(TpEmbedding::Regular(candle_nn::embedding(
                vocab_size,
                hidden_size,
                vb,
            )?))
        } else {
            Ok(TpEmbedding::VocabParallel(VocabParallelEmbedding::new(
                vocab_size,
                hidden_size,
                vb,
                pg,
            )?))
        }
    }

    /// Forward pass: look up embeddings for token IDs.
    pub fn forward(&self, input_ids: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            TpEmbedding::Regular(emb) => emb.forward(input_ids),
            TpEmbedding::VocabParallel(emb) => emb
                .forward(input_ids, tp_ctx.comm())
                .map_err(|e| candle_core::Error::Msg(format!("TP vocab parallel: {e}"))),
        }
    }

    /// Get the embedding weights (for tied embeddings).
    ///
    /// Returns None for VocabParallel embeddings as tying is not supported in TP mode.
    pub fn embeddings(&self) -> Option<&Tensor> {
        match self {
            TpEmbedding::Regular(emb) => Some(emb.embeddings()),
            TpEmbedding::VocabParallel(_) => None,
        }
    }
}

// ─── MLP ──────────────────────────────────────────────────────────────────────

/// Tensor-parallel SwiGLU MLP.
///
/// In TP mode:
/// - gate_proj and up_proj are column-parallel (split intermediate dim)
/// - down_proj is row-parallel (reduce partial outputs)
pub struct TpSwiGluMlp {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

/// Weight-name aliases for TP MLP projections.
///
/// Most model families use the canonical Llama-style triple
/// (`gate_proj` / `up_proj` / `down_proj`). InternLM2 uses
/// `feed_forward.{w1, w3, w2}` instead — `w1`/`w3` are gate/up
/// (the order is *not* alphabetical), `w2` is down.
#[derive(Clone, Copy, Debug)]
pub struct GluProjNames {
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
}

impl GluProjNames {
    /// Llama / Mistral / Qwen / Gemma standard triple.
    pub const STANDARD: Self = Self {
        gate: "gate_proj",
        up: "up_proj",
        down: "down_proj",
    };
    /// InternLM2 / Yi `feed_forward.{w1,w3,w2}` aliases.
    pub const W1_W3_W2: Self = Self {
        gate: "w1",
        up: "w3",
        down: "w2",
    };
}

impl Default for GluProjNames {
    fn default() -> Self {
        Self::STANDARD
    }
}

impl TpSwiGluMlp {
    /// Create a SwiGLU MLP layer with the canonical Llama-style weight
    /// names (`gate_proj` / `up_proj` / `down_proj`).
    ///
    /// For single GPU, uses regular linear layers.
    /// For multi-GPU, uses column/row-parallel linear layers.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        Self::new_with_proj_names(
            hidden_size,
            intermediate_size,
            vb,
            pg,
            GluProjNames::STANDARD,
        )
    }

    /// Create a SwiGLU MLP layer with custom weight-name aliases.
    ///
    /// Used by InternLM2 / Yi (`w1` / `w3` / `w2`) and any future
    /// family that diverges from the Llama-style triple. The TP
    /// sharding is identical — only the VarBuilder lookup differs.
    pub fn new_with_proj_names(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        names: GluProjNames,
    ) -> Result<Self> {
        // Gate and Up are column-parallel (split intermediate)
        // Down is row-parallel (reduce outputs)
        let gate_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false, // no gather, goes to element-wise mul with up
            vb.pp(names.gate),
            pg,
        )?;
        let up_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp(names.up),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true, // input is parallel
            vb.pp(names.down),
            pg,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass: SwiGLU(x) = silu(gate(x)) * up(x), then down projection.
    pub fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs, tp_ctx)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        // SwiGLU: silu(gate) * up
        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

/// Tensor-parallel SwiGLU MLP with **fused** gate+up projection.
///
/// Phi-3 family stores `gate_proj` and `up_proj` as a single
/// `gate_up_proj` weight `[hidden, 2 * intermediate]`. After the
/// projection, the output is split: first half is gate, second
/// half is up. Saves one matmul of overhead at inference time.
///
/// In TP mode the fused weight is column-parallel — the `2 *
/// intermediate` dim is split across ranks, so each rank holds
/// its slice of (gate || up) interleaved. The split into gate/up
/// happens locally after the projection.
pub struct TpFusedSwiGluMlp {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
    /// Local intermediate size = `intermediate_size / world_size`.
    local_intermediate: usize,
}

impl TpFusedSwiGluMlp {
    /// Create a fused-gate-up SwiGLU MLP **without** bias.
    ///
    /// Weight layout: `gate_up_proj.weight` has shape
    /// `[2 * intermediate_size, hidden_size]` with `[gate || up]`
    /// row order (matching Phi-3 / vLLM convention).
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        Self::new_with_bias(hidden_size, intermediate_size, false, vb, pg)
    }

    /// Create a fused-gate-up SwiGLU MLP with optional bias.
    ///
    /// Used by Step1, Ernie 4.5-MoE, and other architectures whose
    /// config has an `mlp_bias` flag. The bias applies uniformly to
    /// both the fused `gate_up_proj` and the `down_proj` projections.
    pub fn new_with_bias(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            bias,
            false,
            vb.pp("gate_up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            bias,
            true,
            vb.pp("down_proj"),
            pg,
        )?;

        let local_intermediate = intermediate_size / pg.world_size();

        Ok(Self {
            gate_up_proj,
            down_proj,
            local_intermediate,
        })
    }

    /// Forward pass: fused projection → split → SwiGLU → down.
    pub fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let fused = self.gate_up_proj.forward(xs, tp_ctx)?;
        let gate = fused.narrow(candle_core::D::Minus1, 0, self.local_intermediate)?;
        let up = fused.narrow(
            candle_core::D::Minus1,
            self.local_intermediate,
            self.local_intermediate,
        )?;
        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

/// Tensor-parallel GeGLU MLP (GELU-gated, used by Gemma models).
///
/// GeGLU: GELU(gate(x)) * up(x), then down projection.
/// Different from SwiGLU which uses SiLU instead of GELU.
///
/// In TP mode:
/// - gate_proj and up_proj are column-parallel (split intermediate dim)
/// - down_proj is row-parallel (reduce partial outputs)
pub struct TpGeGluMlp {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl TpGeGluMlp {
    /// Create a GeGLU MLP layer.
    ///
    /// For single GPU, uses regular linear layers.
    /// For multi-GPU, uses column/row-parallel linear layers.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // Gate and Up are column-parallel (split intermediate)
        // Down is row-parallel (reduce outputs)
        let gate_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false, // no gather, goes to element-wise mul with up
            vb.pp("gate_proj"),
            pg,
        )?;
        let up_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true, // input is parallel
            vb.pp("down_proj"),
            pg,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass: GeGLU(x) = gelu(gate(x)) * up(x), then down projection.
    pub fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs, tp_ctx)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        // GeGLU: gelu(gate) * up
        let hidden = gate.gelu_erf()?.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

/// Tensor-parallel GeLU MLP (used by some models like Falcon, Phi).
///
/// In TP mode:
/// - fc1 is column-parallel (split intermediate dim)
/// - fc2 is row-parallel (reduce partial outputs)
pub struct TpGeluMlp {
    fc1: TpLinear,
    fc2: TpLinear,
}

impl TpGeluMlp {
    /// Create a GeLU MLP layer.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let fc1 = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("fc1"),
            pg,
        )?;
        let fc2 = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true,
            vb.pp("fc2"),
            pg,
        )?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward pass: gelu(fc1(x)), then fc2.
    pub fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let hidden = self.fc1.forward(xs, tp_ctx)?;
        let hidden = candle_nn::Activation::Gelu.forward(&hidden)?;
        self.fc2.forward(&hidden, tp_ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_vb(device: &Device) -> VarBuilder<'static> {
        VarBuilder::zeros(DType::F32, device)
    }

    #[test]
    fn tp_context_single_gpu() {
        let ctx = TpContext::single_gpu();
        assert!(ctx.is_single());
        assert_eq!(ctx.world_size, 1);
        assert_eq!(ctx.rank, 0);
    }

    #[test]
    fn tp_linear_single_gpu() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let linear = TpLinear::column_parallel(64, 128, false, false, vb.pp("test"), &pg).unwrap();

        let input = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
        let output = linear.forward(&input, &tp_ctx).unwrap();

        assert_eq!(output.dims(), &[2, 128]);
    }

    #[test]
    fn tp_embedding_single_gpu() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let emb = TpEmbedding::new(1000, 64, vb.pp("test"), &pg).unwrap();

        let input = Tensor::new(&[1u32, 5, 10], &Device::Cpu).unwrap();
        let output = emb.forward(&input, &tp_ctx).unwrap();

        assert_eq!(output.dims(), &[3, 64]);
    }

    #[test]
    fn tp_swiglu_mlp_single_gpu() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let mlp = TpSwiGluMlp::new(64, 256, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::ones(&[2, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input, &tp_ctx).unwrap();

        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn tp_gelu_mlp_single_gpu() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let mlp = TpGeluMlp::new(64, 256, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::ones(&[2, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input, &tp_ctx).unwrap();

        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn tp_geglu_mlp_single_gpu() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let mlp = TpGeGluMlp::new(64, 256, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::ones(&[2, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input, &tp_ctx).unwrap();

        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn tp_swiglu_mlp_with_proj_names_w123() {
        // InternLM2-style w1/w3/w2 naming. Sharding is identical to the
        // canonical triple — only VarBuilder lookup differs. Verify both
        // construction succeeds and output shape is unchanged.
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let mlp = TpSwiGluMlp::new_with_proj_names(
            64,
            256,
            vb.pp("feed_forward"),
            &pg,
            GluProjNames::W1_W3_W2,
        )
        .unwrap();

        let input = Tensor::ones(&[2, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input, &tp_ctx).unwrap();
        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn tp_fused_swiglu_mlp_single_gpu() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);
        let tp_ctx = TpContext::single_gpu();

        let mlp = TpFusedSwiGluMlp::new(64, 256, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::ones(&[2, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input, &tp_ctx).unwrap();
        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn glu_proj_names_constants() {
        assert_eq!(GluProjNames::STANDARD.gate, "gate_proj");
        assert_eq!(GluProjNames::STANDARD.up, "up_proj");
        assert_eq!(GluProjNames::STANDARD.down, "down_proj");
        assert_eq!(GluProjNames::W1_W3_W2.gate, "w1");
        assert_eq!(GluProjNames::W1_W3_W2.up, "w3");
        assert_eq!(GluProjNames::W1_W3_W2.down, "w2");
        assert_eq!(GluProjNames::default().gate, "gate_proj");
    }

    #[test]
    fn tp_embedding_weights_access() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);

        let emb = TpEmbedding::new(100, 32, vb.pp("test"), &pg).unwrap();

        // Single GPU should have accessible weights
        assert!(emb.embeddings().is_some());
    }
}
