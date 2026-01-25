//! Tensor-parallel attention implementation.
//!
//! In tensor parallelism, attention heads are split across GPUs:
//! - Each GPU handles `num_heads / tp_size` heads
//! - QKV projection is column-parallel (split output)
//! - Output projection is row-parallel (split input, reduce)
//!
//! This matches vLLM's attention parallelism strategy.

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::communicator::DeviceCommunicator;
use super::error::Result;
use super::parallel_layers::{ColumnParallelLinear, RowParallelLinear};
use super::process_group::ProcessGroup;

/// Tensor-parallel multi-head attention.
///
/// Each GPU handles a subset of attention heads.
pub struct TensorParallelAttention {
    /// QKV projection (column-parallel, no gather)
    qkv_proj: ColumnParallelLinear,
    /// Output projection (row-parallel, with reduce)
    o_proj: RowParallelLinear,
    /// Number of heads on this GPU
    num_heads_per_gpu: usize,
    /// Number of KV heads on this GPU (for GQA)
    num_kv_heads_per_gpu: usize,
    /// Head dimension
    head_dim: usize,
    /// Tensor parallel size
    tp_size: usize,
    /// This GPU's rank
    tp_rank: usize,
}

impl TensorParallelAttention {
    /// Create a new tensor-parallel attention layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden size
    /// * `num_heads` - Total number of query heads
    /// * `num_kv_heads` - Total number of KV heads (for GQA)
    /// * `vb` - VarBuilder for weights
    /// * `pg` - Process group for tensor parallelism
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> candle_core::Result<Self> {
        let tp_size = pg.world_size();
        let tp_rank = pg.rank();

        assert_eq!(
            num_heads % tp_size,
            0,
            "num_heads must be divisible by tp_size"
        );
        assert_eq!(
            num_kv_heads % tp_size,
            0,
            "num_kv_heads must be divisible by tp_size"
        );

        let num_heads_per_gpu = num_heads / tp_size;
        let num_kv_heads_per_gpu = num_kv_heads / tp_size;
        let head_dim = hidden_size / num_heads;

        // QKV projection: hidden -> (num_heads + 2*num_kv_heads) * head_dim per GPU
        // This is column-parallel, each GPU gets its portion of Q, K, V
        let qkv_size = (num_heads + 2 * num_kv_heads) * head_dim;
        let qkv_proj =
            ColumnParallelLinear::new(hidden_size, qkv_size, false, false, vb.pp("qkv_proj"), pg)?;

        // Output projection: (num_heads * head_dim) -> hidden
        // This is row-parallel, takes split input and reduces
        let o_proj = RowParallelLinear::new(
            num_heads * head_dim,
            hidden_size,
            false,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads_per_gpu,
            num_kv_heads_per_gpu,
            head_dim,
            tp_size,
            tp_rank,
        })
    }

    /// Forward pass (prefill mode, no KV cache).
    ///
    /// Returns (output, (keys, values)) for caching.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        comm: &dyn DeviceCommunicator,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // QKV projection (column-parallel)
        let qkv = self.qkv_proj.forward(hidden_states, comm)?;

        // Split into Q, K, V
        let q_size = self.num_heads_per_gpu * self.head_dim;
        let kv_size = self.num_kv_heads_per_gpu * self.head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape for attention: [batch, seq, heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads_per_gpu, self.head_dim))?;
        let k = k.reshape((
            batch_size,
            seq_len,
            self.num_kv_heads_per_gpu,
            self.head_dim,
        ))?;
        let v = v.reshape((
            batch_size,
            seq_len,
            self.num_kv_heads_per_gpu,
            self.head_dim,
        ))?;

        // Transpose to [batch, heads, seq, head_dim] and make contiguous for matmul
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Compute attention scores
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores * scale)?;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        // Attention output
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads_per_gpu * self.head_dim))?;

        // Output projection (row-parallel with all-reduce)
        let output = self.o_proj.forward(&attn_output, comm)?;

        Ok(output)
    }

    /// Get number of heads on this GPU.
    pub fn num_heads_per_gpu(&self) -> usize {
        self.num_heads_per_gpu
    }

    /// Get number of KV heads on this GPU.
    pub fn num_kv_heads_per_gpu(&self) -> usize {
        self.num_kv_heads_per_gpu
    }

    /// Get tensor parallel rank.
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    /// Get tensor parallel size.
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }
}

/// Tensor-parallel MLP (SwiGLU variant).
///
/// In tensor parallelism:
/// - Gate + Up projection: column-parallel (split intermediate)
/// - Down projection: row-parallel (reduce output)
pub struct TensorParallelMLP {
    /// Gate projection (column-parallel)
    gate_proj: ColumnParallelLinear,
    /// Up projection (column-parallel)
    up_proj: ColumnParallelLinear,
    /// Down projection (row-parallel)
    down_proj: RowParallelLinear,
}

impl TensorParallelMLP {
    /// Create a new tensor-parallel MLP.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> candle_core::Result<Self> {
        // Gate and Up are column-parallel (split intermediate dimension)
        let gate_proj = ColumnParallelLinear::new(
            hidden_size,
            intermediate_size,
            false,
            false, // no gather, output goes to down_proj
            vb.pp("gate_proj"),
            pg,
        )?;

        let up_proj = ColumnParallelLinear::new(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("up_proj"),
            pg,
        )?;

        // Down is row-parallel (split input, reduce output)
        let down_proj = RowParallelLinear::new(
            intermediate_size,
            hidden_size,
            false,
            true, // input is parallel from gate/up
            vb.pp("down_proj"),
            pg,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&self, x: &Tensor, comm: &dyn DeviceCommunicator) -> Result<Tensor> {
        // Column-parallel projections (no communication needed)
        let gate = self.gate_proj.forward(x, comm)?;
        let up = self.up_proj.forward(x, comm)?;

        // SiLU activation on gate, then multiply
        let gate = candle_nn::ops::silu(&gate)?;
        let hidden = (gate * up)?;

        // Row-parallel down projection (with all-reduce)
        let output = self.down_proj.forward(&hidden, comm)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator};
    use candle_core::{DType, Device};

    fn make_vb(device: &Device) -> VarBuilder<'static> {
        VarBuilder::zeros(DType::F32, device)
    }

    #[test]
    fn tp_attention_construction() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);

        let attn = TensorParallelAttention::new(
            256, // hidden_size
            8,   // num_heads
            8,   // num_kv_heads
            vb.pp("attn"),
            &pg,
        )
        .unwrap();

        assert_eq!(attn.num_heads_per_gpu(), 8);
        assert_eq!(attn.num_kv_heads_per_gpu(), 8);
        assert_eq!(attn.tp_size(), 1);
    }

    #[test]
    fn tp_attention_with_gqa() {
        // GQA: fewer KV heads than query heads
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);

        let attn = TensorParallelAttention::new(
            512, // hidden_size
            16,  // num_heads (Q)
            4,   // num_kv_heads (KV, GQA)
            vb.pp("attn"),
            &pg,
        )
        .unwrap();

        assert_eq!(attn.num_heads_per_gpu(), 16);
        assert_eq!(attn.num_kv_heads_per_gpu(), 4);
    }

    #[test]
    fn tp_attention_forward_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let attn = TensorParallelAttention::new(
            64, // hidden_size
            4,  // num_heads
            4,  // num_kv_heads
            vb.pp("attn"),
            &pg,
        )
        .unwrap();

        let input = Tensor::ones(&[2, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let output = attn.forward(&input, None, &comm).unwrap();

        assert_eq!(output.dims(), &[2, 8, 64]);
    }

    #[test]
    fn tp_attention_with_mask() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let attn = TensorParallelAttention::new(64, 4, 4, vb.pp("attn"), &pg).unwrap();

        let input = Tensor::ones(&[1, 4, 64], DType::F32, &Device::Cpu).unwrap();

        // Causal mask: [1, 1, 4, 4] - broadcasting over batch and heads
        let mask = Tensor::zeros(&[1, 1, 4, 4], DType::F32, &Device::Cpu).unwrap();

        let output = attn.forward(&input, Some(&mask), &comm).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn tp_mlp_construction() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);

        let _mlp = TensorParallelMLP::new(
            256,  // hidden_size
            1024, // intermediate_size
            vb.pp("mlp"),
            &pg,
        )
        .unwrap();
    }

    #[test]
    fn tp_mlp_forward() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let mlp = TensorParallelMLP::new(64, 256, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::ones(&[2, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input, &comm).unwrap();

        assert_eq!(output.dims(), &[2, 8, 64]);
    }

    #[test]
    fn tp_attention_multi_gpu_simulated() {
        // Simulate 4-GPU tensor parallelism
        let pg = LocalProcessGroup::with_rank(0, 4);
        let vb = make_vb(&Device::Cpu);

        let attn = TensorParallelAttention::new(
            256, // hidden_size (256 = 32 heads * 8 head_dim)
            32,  // num_heads (32 / 4 = 8 per GPU)
            8,   // num_kv_heads (8 / 4 = 2 per GPU)
            vb.pp("attn"),
            &pg,
        )
        .unwrap();

        assert_eq!(attn.num_heads_per_gpu(), 8);
        assert_eq!(attn.num_kv_heads_per_gpu(), 2);
        assert_eq!(attn.tp_size(), 4);
    }

    #[test]
    #[should_panic(expected = "num_heads must be divisible by tp_size")]
    fn tp_attention_invalid_head_count() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let vb = make_vb(&Device::Cpu);

        // 10 heads not divisible by 4 GPUs
        let _ = TensorParallelAttention::new(
            256,
            10, // not divisible by 4
            4,
            vb.pp("attn"),
            &pg,
        );
    }

    #[test]
    #[should_panic(expected = "num_kv_heads must be divisible by tp_size")]
    fn tp_attention_invalid_kv_head_count() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let vb = make_vb(&Device::Cpu);

        let _ = TensorParallelAttention::new(
            256,
            16,
            3, // not divisible by 4
            vb.pp("attn"),
            &pg,
        );
    }
}
