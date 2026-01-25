//! Pipeline Parallelism for distributed inference.
//!
//! Pipeline parallelism splits the model into stages, each on a different GPU.
//! Tokens flow through the pipeline: GPU0 -> GPU1 -> ... -> GPUn.
//!
//! # Architecture
//!
//! ```text
//! GPU 0 (Stage 0)     GPU 1 (Stage 1)     GPU 2 (Stage 2)
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │ Layers 0-7  │ --> │ Layers 8-15 │ --> │ Layers 16-23│
//! └─────────────┘     └─────────────┘     └─────────────┘
//!       │                   │                   │
//!       v                   v                   v
//!   Send to 1          Send to 2           Output
//! ```
//!
//! # Key Concepts
//!
//! - **Stage**: A subset of model layers on one GPU
//! - **Microbatch**: Small batch processed through the pipeline
//! - **Bubble**: Time when some GPUs are idle (minimized with more microbatches)

use candle_core::Tensor;

use super::communicator::DeviceCommunicator;
use super::error::Result;

/// Pipeline stage configuration.
#[derive(Debug, Clone)]
pub struct PipelineStageConfig {
    /// Index of this stage (0..num_stages)
    pub stage_id: usize,
    /// Total number of pipeline stages
    pub num_stages: usize,
    /// First layer index for this stage
    pub first_layer: usize,
    /// Number of layers in this stage
    pub num_layers: usize,
    /// Whether this is the first stage (receives input)
    pub is_first: bool,
    /// Whether this is the last stage (produces output)
    pub is_last: bool,
}

impl PipelineStageConfig {
    /// Create configuration for a pipeline stage.
    pub fn new(stage_id: usize, num_stages: usize, total_layers: usize) -> Self {
        assert!(num_stages > 0, "num_stages must be > 0");
        assert!(stage_id < num_stages, "stage_id must be < num_stages");
        assert!(
            total_layers >= num_stages,
            "total_layers must be >= num_stages"
        );

        // Distribute layers evenly
        let base_layers = total_layers / num_stages;
        let extra = total_layers % num_stages;

        // Earlier stages get extra layers if not evenly divisible
        let num_layers = if stage_id < extra {
            base_layers + 1
        } else {
            base_layers
        };

        let first_layer = if stage_id < extra {
            stage_id * (base_layers + 1)
        } else {
            extra * (base_layers + 1) + (stage_id - extra) * base_layers
        };

        Self {
            stage_id,
            num_stages,
            first_layer,
            num_layers,
            is_first: stage_id == 0,
            is_last: stage_id == num_stages - 1,
        }
    }

    /// Get layer indices for this stage.
    pub fn layer_range(&self) -> std::ops::Range<usize> {
        self.first_layer..self.first_layer + self.num_layers
    }
}

/// Pipeline schedule for micro-batching.
///
/// Defines the order of forward/backward passes across stages
/// to minimize pipeline bubbles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// Simple synchronous schedule (1F1B-like for inference)
    /// Each stage processes and sends immediately.
    Synchronous,
    /// Interleaved schedule for better utilization.
    /// Requires virtual pipeline stages.
    Interleaved,
}

/// Pipeline communicator for inter-stage data transfer.
pub struct PipelineCommunicator<'a, C: DeviceCommunicator> {
    /// Underlying device communicator
    comm: &'a C,
    /// Stage configuration
    stage_config: PipelineStageConfig,
}

impl<'a, C: DeviceCommunicator> PipelineCommunicator<'a, C> {
    /// Create a new pipeline communicator.
    pub fn new(comm: &'a C, stage_config: PipelineStageConfig) -> Self {
        Self { comm, stage_config }
    }

    /// Send hidden states to the next stage.
    ///
    /// Called by all stages except the last.
    pub fn send_to_next(&self, hidden_states: &Tensor) -> Result<()> {
        if self.stage_config.is_last {
            return Ok(()); // Last stage has no next
        }

        let next_rank = self.stage_config.stage_id + 1;
        self.comm.send(hidden_states, next_rank)
    }

    /// Receive hidden states from the previous stage.
    ///
    /// Called by all stages except the first.
    pub fn recv_from_prev(&self, shape: &[usize], dtype: candle_core::DType) -> Result<Tensor> {
        if self.stage_config.is_first {
            // First stage receives from input, not from another stage
            return Err(super::error::DistributedError::InvalidRank {
                rank: 0,
                world_size: self.stage_config.num_stages,
            });
        }

        let prev_rank = self.stage_config.stage_id - 1;
        self.comm.recv(shape, dtype, prev_rank)
    }

    /// Get the stage configuration.
    pub fn stage_config(&self) -> &PipelineStageConfig {
        &self.stage_config
    }
}

/// Helper for synchronous pipeline execution.
pub struct SyncPipelineExecutor {
    /// Stage configuration
    stage_config: PipelineStageConfig,
}

impl SyncPipelineExecutor {
    /// Create executor for a pipeline stage.
    pub fn new(stage_config: PipelineStageConfig) -> Self {
        Self { stage_config }
    }

    /// Execute one forward step.
    ///
    /// For inference, this processes a microbatch through this stage.
    pub fn forward_step<F>(&self, input: Tensor, stage_forward: F) -> Result<Tensor>
    where
        F: FnOnce(Tensor) -> Result<Tensor>,
    {
        // Apply stage's layers
        let output = stage_forward(input)?;
        Ok(output)
    }

    /// Whether this stage should receive input from previous.
    pub fn should_recv(&self) -> bool {
        !self.stage_config.is_first
    }

    /// Whether this stage should send output to next.
    pub fn should_send(&self) -> bool {
        !self.stage_config.is_last
    }

    /// Get stage ID.
    pub fn stage_id(&self) -> usize {
        self.stage_config.stage_id
    }

    /// Get number of stages.
    pub fn num_stages(&self) -> usize {
        self.stage_config.num_stages
    }
}

/// Calculate optimal number of microbatches for a given batch size.
///
/// More microbatches = smaller bubbles but more overhead.
pub fn optimal_microbatches(batch_size: usize, num_stages: usize) -> usize {
    // Rule of thumb: at least num_stages microbatches to keep pipeline full
    // But not more than batch_size
    (num_stages * 2).min(batch_size).max(1)
}

/// Split a batch into microbatches.
pub fn split_microbatches(
    batch: &Tensor,
    num_microbatches: usize,
) -> candle_core::Result<Vec<Tensor>> {
    let batch_size = batch.dim(0)?;
    let microbatch_size = batch_size.div_ceil(num_microbatches);

    let mut microbatches = Vec::with_capacity(num_microbatches);
    let mut offset = 0;

    for _ in 0..num_microbatches {
        let size = microbatch_size.min(batch_size - offset);
        if size == 0 {
            break;
        }
        microbatches.push(batch.narrow(0, offset, size)?);
        offset += size;
    }

    Ok(microbatches)
}

/// Merge microbatches back into a single batch.
pub fn merge_microbatches(microbatches: &[Tensor]) -> candle_core::Result<Tensor> {
    if microbatches.is_empty() {
        return Err(candle_core::Error::Msg(
            "No microbatches to merge".to_string(),
        ));
    }
    Tensor::cat(microbatches, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn pipeline_stage_config_single_stage() {
        let cfg = PipelineStageConfig::new(0, 1, 32);
        assert_eq!(cfg.stage_id, 0);
        assert_eq!(cfg.num_stages, 1);
        assert_eq!(cfg.first_layer, 0);
        assert_eq!(cfg.num_layers, 32);
        assert!(cfg.is_first);
        assert!(cfg.is_last);
    }

    #[test]
    fn pipeline_stage_config_two_stages() {
        let cfg0 = PipelineStageConfig::new(0, 2, 32);
        let cfg1 = PipelineStageConfig::new(1, 2, 32);

        assert_eq!(cfg0.first_layer, 0);
        assert_eq!(cfg0.num_layers, 16);
        assert!(cfg0.is_first);
        assert!(!cfg0.is_last);

        assert_eq!(cfg1.first_layer, 16);
        assert_eq!(cfg1.num_layers, 16);
        assert!(!cfg1.is_first);
        assert!(cfg1.is_last);
    }

    #[test]
    fn pipeline_stage_config_uneven() {
        // 32 layers, 3 stages: 11 + 11 + 10
        let cfg0 = PipelineStageConfig::new(0, 3, 32);
        let cfg1 = PipelineStageConfig::new(1, 3, 32);
        let cfg2 = PipelineStageConfig::new(2, 3, 32);

        assert_eq!(cfg0.num_layers, 11);
        assert_eq!(cfg1.num_layers, 11);
        assert_eq!(cfg2.num_layers, 10);

        assert_eq!(cfg0.first_layer, 0);
        assert_eq!(cfg1.first_layer, 11);
        assert_eq!(cfg2.first_layer, 22);

        // Verify all layers covered exactly once
        let total: usize = [&cfg0, &cfg1, &cfg2].iter().map(|c| c.num_layers).sum();
        assert_eq!(total, 32);
    }

    #[test]
    fn pipeline_stage_layer_range() {
        let cfg = PipelineStageConfig::new(1, 4, 24);
        let range = cfg.layer_range();
        assert_eq!(range, 6..12);
    }

    #[test]
    fn sync_executor_should_recv_send() {
        let cfg_first = PipelineStageConfig::new(0, 3, 24);
        let cfg_mid = PipelineStageConfig::new(1, 3, 24);
        let cfg_last = PipelineStageConfig::new(2, 3, 24);

        let exec_first = SyncPipelineExecutor::new(cfg_first);
        let exec_mid = SyncPipelineExecutor::new(cfg_mid);
        let exec_last = SyncPipelineExecutor::new(cfg_last);

        // First stage: doesn't recv, does send
        assert!(!exec_first.should_recv());
        assert!(exec_first.should_send());

        // Middle stage: recv and send
        assert!(exec_mid.should_recv());
        assert!(exec_mid.should_send());

        // Last stage: recv, doesn't send
        assert!(exec_last.should_recv());
        assert!(!exec_last.should_send());
    }

    #[test]
    fn optimal_microbatches_calculation() {
        assert_eq!(optimal_microbatches(32, 4), 8); // 4*2 = 8
        assert_eq!(optimal_microbatches(4, 8), 4); // min(16, 4) = 4
        assert_eq!(optimal_microbatches(1, 4), 1); // max(8, 1) but batch=1
    }

    #[test]
    fn split_merge_microbatches() {
        let batch = Tensor::ones(&[8, 64], DType::F32, &Device::Cpu).unwrap();

        let microbatches = split_microbatches(&batch, 4).unwrap();
        assert_eq!(microbatches.len(), 4);
        for mb in &microbatches {
            assert_eq!(mb.dims(), &[2, 64]);
        }

        let merged = merge_microbatches(&microbatches).unwrap();
        assert_eq!(merged.dims(), &[8, 64]);
    }

    #[test]
    fn split_microbatches_uneven() {
        let batch = Tensor::ones(&[7, 32], DType::F32, &Device::Cpu).unwrap();

        let microbatches = split_microbatches(&batch, 3).unwrap();
        assert_eq!(microbatches.len(), 3);

        let sizes: Vec<usize> = microbatches.iter().map(|t| t.dim(0).unwrap()).collect();
        assert_eq!(sizes, vec![3, 3, 1]); // 7 / 3 = 2.33, ceil to 3

        let total: usize = sizes.iter().sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn sync_executor_forward_step() {
        let cfg = PipelineStageConfig::new(0, 2, 16);
        let exec = SyncPipelineExecutor::new(cfg);

        let input = Tensor::ones(&[4, 64], DType::F32, &Device::Cpu).unwrap();

        let output = exec
            .forward_step(input, |x| {
                // Simulate layer processing (identity for test)
                Ok(x)
            })
            .unwrap();

        assert_eq!(output.dims(), &[4, 64]);
    }

    #[test]
    fn pipeline_schedule_variants() {
        // Ensure both schedule types exist
        let _sync = PipelineSchedule::Synchronous;
        let _inter = PipelineSchedule::Interleaved;
    }

    #[test]
    #[should_panic(expected = "num_stages must be > 0")]
    fn pipeline_config_zero_stages() {
        let _ = PipelineStageConfig::new(0, 0, 16);
    }

    #[test]
    #[should_panic(expected = "stage_id must be < num_stages")]
    fn pipeline_config_invalid_stage_id() {
        let _ = PipelineStageConfig::new(4, 4, 16);
    }
}
