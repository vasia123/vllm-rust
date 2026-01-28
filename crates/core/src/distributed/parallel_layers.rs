//! Parallel linear layers for tensor parallelism.
//!
//! These layers split weights across multiple GPUs and use collective
//! operations to combine results.
//!
//! # Column Parallel Linear
//! Splits the output dimension: each GPU computes a portion of the output.
//! Used for: first linear in MLP, QKV projection.
//!
//! # Row Parallel Linear
//! Splits the input dimension: each GPU has a portion of the weight.
//! Used for: second linear in MLP, output projection.

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::communicator::{DeviceCommunicator, ReduceOp};
use super::error::Result;
use super::process_group::ProcessGroup;

/// Column-parallel linear layer.
///
/// Splits output features across GPUs. Each GPU computes:
/// `output_chunk = input @ weight_chunk + bias_chunk`
///
/// The outputs are later gathered (all_gather) or used directly
/// if followed by row-parallel.
pub struct ColumnParallelLinear {
    /// Weight: [out_features/tp_size, in_features]
    weight: Tensor,
    /// Optional bias: [out_features/tp_size]
    bias: Option<Tensor>,
    /// Tensor parallel size
    tp_size: usize,
    /// This GPU's rank in TP group
    tp_rank: usize,
    /// Whether to gather output across TP group
    gather_output: bool,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear layer.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension (not split)
    /// * `out_features` - Output dimension (split across TP)
    /// * `bias` - Whether to include bias
    /// * `gather_output` - Whether to all_gather output
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for TP
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        gather_output: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> candle_core::Result<Self> {
        let tp_size = pg.world_size();
        let tp_rank = pg.rank();

        // Each GPU gets out_features/tp_size columns
        let out_per_gpu = out_features / tp_size;
        assert_eq!(
            out_features % tp_size,
            0,
            "out_features must be divisible by tp_size"
        );

        // Weight shape: [out_per_gpu, in_features]
        let weight = vb.get((out_per_gpu, in_features), "weight")?;

        let bias = if bias {
            Some(vb.get(out_per_gpu, "bias")?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            tp_size,
            tp_rank,
            gather_output,
        })
    }

    /// Create from existing tensors (for testing).
    pub fn from_parts(
        weight: Tensor,
        bias: Option<Tensor>,
        tp_size: usize,
        tp_rank: usize,
        gather_output: bool,
    ) -> Self {
        Self {
            weight,
            bias,
            tp_size,
            tp_rank,
            gather_output,
        }
    }

    /// Forward pass.
    ///
    /// If `gather_output` is true, uses all_gather to combine outputs.
    /// Otherwise returns the local chunk.
    pub fn forward(&self, input: &Tensor, comm: &dyn DeviceCommunicator) -> Result<Tensor> {
        // Handle N-dimensional input by flattening to 2D for matmul
        let input_dims = input.dims();
        let in_features = *input_dims.last().ok_or_else(|| {
            super::error::DistributedError::TensorError(candle_core::Error::Msg(
                "Tensor must have at least 1 dimension".to_string(),
            ))
        })?;
        let batch_size: usize = input_dims.iter().rev().skip(1).product();

        // Flatten: [..., in_features] -> [batch, in_features]
        let flat_input = input.reshape((batch_size, in_features))?;

        // Local matmul: [batch, in_features] @ [out_per_gpu, in_features].T
        let mut output = flat_input.matmul(&self.weight.t()?)?;

        if let Some(ref bias) = self.bias {
            output = output.broadcast_add(bias)?;
        }

        if self.gather_output && self.tp_size > 1 {
            // Gather outputs from all GPUs: [batch, out_per_gpu] -> [batch, out_features]
            output = comm.all_gather(&output, 1)?;
        }

        // Reshape back to original dimensions with new last dim
        let out_features = output.dim(1)?;
        let mut out_shape: Vec<usize> = input_dims[..input_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        let output = output.reshape(out_shape.as_slice())?;

        Ok(output)
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

/// Row-parallel linear layer.
///
/// Splits input features across GPUs. Each GPU computes:
/// `output_partial = input_chunk @ weight_chunk`
///
/// The outputs are then reduced (all_reduce) to get final result.
pub struct RowParallelLinear {
    /// Weight: [out_features, in_features/tp_size]
    weight: Tensor,
    /// Optional bias (only on rank 0, or added after reduce)
    bias: Option<Tensor>,
    /// Tensor parallel size
    tp_size: usize,
    /// This GPU's rank in TP group
    tp_rank: usize,
    /// Whether input is already split (from column parallel)
    #[allow(dead_code)]
    input_is_parallel: bool,
}

impl RowParallelLinear {
    /// Create a new row-parallel linear layer.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension (split across TP if input_is_parallel)
    /// * `out_features` - Output dimension (not split)
    /// * `bias` - Whether to include bias
    /// * `input_is_parallel` - Whether input comes from column-parallel (already split)
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for TP
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        input_is_parallel: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> candle_core::Result<Self> {
        let tp_size = pg.world_size();
        let tp_rank = pg.rank();

        // Each GPU gets in_features/tp_size rows
        let in_per_gpu = in_features / tp_size;
        assert_eq!(
            in_features % tp_size,
            0,
            "in_features must be divisible by tp_size"
        );

        // Weight shape: [out_features, in_per_gpu]
        let weight = vb.get((out_features, in_per_gpu), "weight")?;

        let bias = if bias {
            Some(vb.get(out_features, "bias")?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            tp_size,
            tp_rank,
            input_is_parallel,
        })
    }

    /// Create from existing tensors (for testing).
    pub fn from_parts(
        weight: Tensor,
        bias: Option<Tensor>,
        tp_size: usize,
        tp_rank: usize,
        input_is_parallel: bool,
    ) -> Self {
        Self {
            weight,
            bias,
            tp_size,
            tp_rank,
            input_is_parallel,
        }
    }

    /// Forward pass.
    ///
    /// Uses all_reduce to sum partial outputs across GPUs.
    pub fn forward(&self, input: &Tensor, comm: &dyn DeviceCommunicator) -> Result<Tensor> {
        // If input is parallel, we need the right chunk
        // (handled externally: column parallel output goes directly to row parallel)

        // Handle N-dimensional input by flattening to 2D for matmul
        let input_dims = input.dims();
        let in_features = *input_dims.last().ok_or_else(|| {
            super::error::DistributedError::TensorError(candle_core::Error::Msg(
                "Tensor must have at least 1 dimension".to_string(),
            ))
        })?;
        let batch_size: usize = input_dims.iter().rev().skip(1).product();

        // Flatten: [..., in_features] -> [batch, in_features]
        let flat_input = input.reshape((batch_size, in_features))?;

        // Local matmul: [batch, in_per_gpu] @ [out_features, in_per_gpu].T
        let output = flat_input.matmul(&self.weight.t()?)?;

        // All-reduce to sum partial results
        let mut output = if self.tp_size > 1 {
            comm.all_reduce(&output, ReduceOp::Sum)?
        } else {
            output
        };

        // Add bias after reduce (to avoid duplicating)
        if let Some(ref bias) = self.bias {
            output = output.broadcast_add(bias)?;
        }

        // Reshape back to original dimensions with new last dim
        let out_features = output.dim(1)?;
        let mut out_shape: Vec<usize> = input_dims[..input_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        let output = output.reshape(out_shape.as_slice())?;

        Ok(output)
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

/// Vocabulary-parallel embedding.
///
/// Splits vocabulary across GPUs. Used for large vocabularies.
pub struct VocabParallelEmbedding {
    /// Embedding table: [vocab_size/tp_size, hidden_size]
    embeddings: Tensor,
    /// Full vocabulary size
    #[allow(dead_code)]
    vocab_size: usize,
    /// Tensor parallel size
    tp_size: usize,
    /// This GPU's rank
    #[allow(dead_code)]
    tp_rank: usize,
    /// Starting index of vocab partition on this GPU
    vocab_start: usize,
    /// Ending index (exclusive)
    vocab_end: usize,
}

impl VocabParallelEmbedding {
    /// Create a new vocabulary-parallel embedding.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> candle_core::Result<Self> {
        let tp_size = pg.world_size();
        let tp_rank = pg.rank();

        // Split vocabulary evenly
        let vocab_per_gpu = vocab_size.div_ceil(tp_size);
        let vocab_start = tp_rank * vocab_per_gpu;
        let vocab_end = ((tp_rank + 1) * vocab_per_gpu).min(vocab_size);
        let actual_size = vocab_end - vocab_start;

        let embeddings = vb.get((actual_size, hidden_size), "weight")?;

        Ok(Self {
            embeddings,
            vocab_size,
            tp_size,
            tp_rank,
            vocab_start,
            vocab_end,
        })
    }

    /// Create from existing tensors (for testing).
    pub fn from_parts(
        embeddings: Tensor,
        vocab_size: usize,
        tp_size: usize,
        tp_rank: usize,
    ) -> Self {
        let vocab_per_gpu = vocab_size.div_ceil(tp_size);
        let vocab_start = tp_rank * vocab_per_gpu;
        let vocab_end = ((tp_rank + 1) * vocab_per_gpu).min(vocab_size);

        Self {
            embeddings,
            vocab_size,
            tp_size,
            tp_rank,
            vocab_start,
            vocab_end,
        }
    }

    /// Forward pass: lookup embeddings for token IDs.
    ///
    /// Uses all_reduce to combine embeddings from different vocab partitions.
    pub fn forward(&self, input_ids: &Tensor, comm: &dyn DeviceCommunicator) -> Result<Tensor> {
        let device = self.embeddings.device();
        let dtype = self.embeddings.dtype();
        let hidden_size = self.embeddings.dim(1)?;

        // Get input shape for output
        let input_shape = input_ids.dims();
        let flat_input = input_ids.flatten_all()?;
        let seq_len = flat_input.dim(0)?;

        if self.tp_size == 1 {
            // Single GPU: direct lookup
            let output = self.embeddings.embedding(&flat_input)?;
            let mut out_shape: Vec<usize> = input_shape.to_vec();
            out_shape.push(hidden_size);
            return Ok(output.reshape(out_shape.as_slice())?);
        }

        // Multi-GPU: masked lookup + all_reduce
        let input_vec: Vec<u32> = flat_input.to_vec1()?;
        let mut output_data = vec![0f32; seq_len * hidden_size];

        for (i, &token_id) in input_vec.iter().enumerate() {
            let token_id = token_id as usize;
            if token_id >= self.vocab_start && token_id < self.vocab_end {
                // This token is on our partition
                let local_idx = token_id - self.vocab_start;
                let embedding = self.embeddings.get(local_idx)?;
                let emb_vec: Vec<f32> = embedding.to_vec1()?;
                output_data[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&emb_vec);
            }
            // Tokens not on this partition remain zeros
        }

        let output = Tensor::from_vec(output_data, (seq_len, hidden_size), device)?;
        let output = output.to_dtype(dtype)?;

        // All-reduce: sum embeddings across partitions
        // (each token only has non-zero embedding on one GPU)
        let output = comm.all_reduce(&output, ReduceOp::Sum)?;

        let mut out_shape: Vec<usize> = input_shape.to_vec();
        out_shape.push(hidden_size);
        Ok(output.reshape(out_shape.as_slice())?)
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
    fn column_parallel_construction() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);

        let layer = ColumnParallelLinear::new(128, 256, true, false, vb.pp("test"), &pg).unwrap();

        assert_eq!(layer.tp_size(), 1);
        assert_eq!(layer.tp_rank(), 0);
    }

    #[test]
    fn column_parallel_forward_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let layer = ColumnParallelLinear::new(64, 128, false, false, vb.pp("test"), &pg).unwrap();

        let input = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
        let output = layer.forward(&input, &comm).unwrap();

        assert_eq!(output.dims(), &[2, 128]);
    }

    #[test]
    fn column_parallel_with_gather() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let comm = MockCommunicator::new(pg.clone());

        // Create layer manually with correctly sized weight
        let weight = Tensor::ones(&[32, 64], DType::F32, &Device::Cpu).unwrap(); // 128/4 = 32
        let layer = ColumnParallelLinear::from_parts(weight, None, 4, 0, true);

        let input = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
        let output = layer.forward(&input, &comm).unwrap();

        // With gather_output=true, output should be gathered: 32*4 = 128
        assert_eq!(output.dims(), &[2, 128]);
    }

    #[test]
    fn row_parallel_construction() {
        let pg = LocalProcessGroup::new();
        let vb = make_vb(&Device::Cpu);

        let layer = RowParallelLinear::new(256, 128, true, true, vb.pp("test"), &pg).unwrap();

        assert_eq!(layer.tp_size(), 1);
        assert_eq!(layer.tp_rank(), 0);
    }

    #[test]
    fn row_parallel_forward_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let layer = RowParallelLinear::new(64, 32, false, false, vb.pp("test"), &pg).unwrap();

        let input = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
        let output = layer.forward(&input, &comm).unwrap();

        assert_eq!(output.dims(), &[2, 32]);
    }

    #[test]
    fn vocab_parallel_single_gpu() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let layer = VocabParallelEmbedding::new(1000, 64, vb.pp("test"), &pg).unwrap();

        let input = Tensor::new(&[1u32, 5, 10], &Device::Cpu).unwrap();
        let output = layer.forward(&input, &comm).unwrap();

        assert_eq!(output.dims(), &[3, 64]);
    }

    #[test]
    fn vocab_parallel_batch() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        let layer = VocabParallelEmbedding::new(100, 32, vb.pp("test"), &pg).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &Device::Cpu).unwrap();
        let output = layer.forward(&input, &comm).unwrap();

        assert_eq!(output.dims(), &[2, 3, 32]);
    }

    #[test]
    fn column_row_parallel_chain() {
        // Test column parallel -> row parallel pattern (like in MLP)
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg.clone());
        let vb = make_vb(&Device::Cpu);

        // Column parallel: 64 -> 256, no gather (output goes to row parallel)
        let col = ColumnParallelLinear::new(64, 256, false, false, vb.pp("col"), &pg).unwrap();

        // Row parallel: 256 -> 64, input is parallel
        let row = RowParallelLinear::new(256, 64, false, true, vb.pp("row"), &pg).unwrap();

        let input = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
        let hidden = col.forward(&input, &comm).unwrap();
        let output = row.forward(&hidden, &comm).unwrap();

        assert_eq!(output.dims(), &[2, 64]);
    }

    #[test]
    #[should_panic(expected = "out_features must be divisible by tp_size")]
    fn divisibility_check_column() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let vb = make_vb(&Device::Cpu);

        // 101 not divisible by 4 -> should panic
        let _ = ColumnParallelLinear::new(64, 101, false, false, vb.pp("test"), &pg);
    }

    #[test]
    #[should_panic(expected = "in_features must be divisible by tp_size")]
    fn divisibility_check_row() {
        let pg = LocalProcessGroup::with_rank(0, 4);
        let vb = make_vb(&Device::Cpu);

        // 101 not divisible by 4 -> should panic
        let _ = RowParallelLinear::new(101, 64, false, false, vb.pp("test"), &pg);
    }
}
