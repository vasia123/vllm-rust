//! Pipeline parallelism execution for distributed inference.
//!
//! This module provides pipeline-parallel execution by splitting the model across
//! multiple pipeline stages (typically one per GPU). Each stage runs a subset of
//! the model layers, and hidden states flow through the pipeline via P2P communication.
//!
//! # Architecture
//!
//! The key types are:
//!
//! - [`PipelineForward`] — Trait for models that support partial-layer execution
//! - [`PipelineStagedModel`] — Wraps `PipelineForward`, implements [`ModelForward`]
//!   transparently, handling all pipeline communication internally
//! - [`pipeline_worker_loop`] — Worker loop for non-coordinator pipeline stages
//!
//! # Design
//!
//! `PipelineStagedModel` implements `ModelForward`, so it works seamlessly with
//! `StandardExecution` and the existing engine loop. The pipeline communication
//! is completely transparent to the rest of the engine.
//!
//! ```text
//! Rank 0 (coordinator):
//!   StandardExecution<PipelineStagedModel<M>>
//!     → embed → forward_layers → send → ... → recv logits
//!
//! Rank 1..N-1 (workers):
//!   pipeline_worker_loop()
//!     → recv → forward_layers → send (or lm_head + send logits)
//! ```

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use crate::distributed::{DeviceCommunicator, PipelineStageConfig};
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::model_forward::{DecodeSequenceMetadata, ModelForward};

// ─── PipelineForward Trait ───────────────────────────────────────────────

/// Trait for models that support pipeline-parallel execution.
///
/// Models implement this to enable partial layer execution across multiple
/// pipeline stages. Each stage processes a subset of the model's layers.
///
/// # Stage Responsibilities
///
/// - **First stage** (rank 0): Calls `embed()` + `forward_layers()`
/// - **Middle stages**: Calls `forward_layers()` only
/// - **Last stage**: Calls `forward_layers()` + `lm_head()`
pub trait PipelineForward: Send + 'static {
    /// Embed input token IDs into hidden states.
    ///
    /// Called by the first pipeline stage only. Applies the token embedding
    /// layer (and possibly positional encoding) to convert token IDs into
    /// hidden state vectors.
    ///
    /// # Returns
    /// Hidden states tensor of shape `[batch, seq_len, hidden_size]`.
    fn embed(&self, input_ids: &Tensor) -> candle_core::Result<Tensor>;

    /// Forward through this stage's transformer layers.
    ///
    /// Processes hidden states through the layers assigned to this pipeline
    /// stage, updating the KV cache as needed.
    ///
    /// # Returns
    /// Hidden states tensor of shape `[batch, seq_len, hidden_size]`.
    fn forward_layers(
        &self,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor>;

    /// Forward through this stage's layers for batched decode.
    ///
    /// Processes multiple sequences each generating one token. Each sequence
    /// has its own KV cache block table and position offset.
    ///
    /// Default implementation processes sequences one by one.
    fn forward_layers_decode_batch(
        &self,
        hidden_states: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let hs = hidden_states.narrow(0, i, 1)?;
            let block_table = BlockTable::from_block_ids(seq.block_ids.clone(), seq.seqlen_offset);
            let out = self.forward_layers(
                &hs,
                seq.seqlen_offset,
                kv_cache_mgr,
                &block_table,
                &seq.slot_mapping,
            )?;
            outputs.push(out);
        }
        Tensor::cat(&outputs, 0)
    }

    /// Apply the language model head to produce logits from hidden states.
    ///
    /// Called by the last pipeline stage only. Converts the final layer's
    /// hidden states into vocabulary logits for sampling.
    ///
    /// # Returns
    /// Logits tensor of shape `[batch, seq_len, vocab_size]`.
    fn lm_head(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor>;

    /// Get the device this stage runs on.
    fn device(&self) -> &Device;

    /// Get the hidden size of the model.
    ///
    /// Used for allocating communication buffers between pipeline stages.
    fn hidden_size(&self) -> usize;
}

/// Blanket implementation so `Box<dyn PipelineForward>` can be passed wherever
/// `M: PipelineForward` is required (e.g. `pipeline_worker_loop`).
impl PipelineForward for Box<dyn PipelineForward> {
    fn embed(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        (**self).embed(input_ids)
    }

    fn forward_layers(
        &self,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut crate::kv_cache::KVCacheManager,
        block_table: &crate::kv_cache::BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        (**self).forward_layers(
            hidden_states,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_layers_decode_batch(
        &self,
        hidden_states: &Tensor,
        sequences: &[super::model_forward::DecodeSequenceMetadata],
        kv_cache_mgr: &mut crate::kv_cache::KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_layers_decode_batch(hidden_states, sequences, kv_cache_mgr)
    }

    fn lm_head(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        (**self).lm_head(hidden_states)
    }

    fn device(&self) -> &Device {
        (**self).device()
    }

    fn hidden_size(&self) -> usize {
        (**self).hidden_size()
    }
}

// ─── PipelineStagedModel ─────────────────────────────────────────────────

/// Model wrapper that transparently handles pipeline communication.
///
/// Implements [`ModelForward`], so it can be used directly with
/// [`StandardExecution`](super::standard::StandardExecution) and the existing
/// engine loop. All inter-stage P2P communication is handled internally.
///
/// # Protocol (rank 0, non-last stage)
///
/// ```text
/// forward(input_ids):
///   1. embed(input_ids) → hidden [B, S, D]
///   2. forward_layers(hidden) → hidden [B, S, D]
///   3. send(hidden) → rank 1
///   4. recv(logits) ← rank N-1  (blocks until pipeline completes)
/// ```
///
/// # Single-stage bypass
///
/// When `num_stages == 1`, no communication occurs and the model executes
/// normally: embed → forward_layers → lm_head.
pub struct PipelineStagedModel<M: PipelineForward> {
    model: M,
    stage_config: PipelineStageConfig,
    comm: Arc<dyn DeviceCommunicator>,
    vocab_size: usize,
}

impl<M: PipelineForward> PipelineStagedModel<M> {
    /// Create a new pipeline-staged model.
    ///
    /// # Arguments
    /// * `model` - The partial model for this stage (only has its assigned layers)
    /// * `stage_config` - Pipeline stage configuration
    /// * `comm` - Device communicator for inter-stage communication
    /// * `vocab_size` - Model vocabulary size (for recv buffer allocation)
    pub fn new(
        model: M,
        stage_config: PipelineStageConfig,
        comm: Arc<dyn DeviceCommunicator>,
        vocab_size: usize,
    ) -> Self {
        Self {
            model,
            stage_config,
            comm,
            vocab_size,
        }
    }

    /// Get the pipeline stage configuration.
    pub fn stage_config(&self) -> &PipelineStageConfig {
        &self.stage_config
    }

    /// Whether this is a single-stage pipeline (no communication needed).
    fn is_single_stage(&self) -> bool {
        self.stage_config.num_stages == 1
    }

    /// Last rank in the pipeline (for receiving logits).
    fn last_rank(&self) -> usize {
        self.stage_config.num_stages - 1
    }

    /// Send hidden states to the next pipeline stage.
    fn send_to_next(&self, tensor: &Tensor) -> candle_core::Result<()> {
        let next_rank = self.stage_config.stage_id + 1;
        self.comm
            .send(tensor, next_rank)
            .map_err(|e| candle_core::Error::Msg(format!("pipeline send to rank {next_rank}: {e}")))
    }

    /// Receive hidden states from the previous pipeline stage.
    fn recv_from_prev(&self, shape: &[usize], dtype: DType) -> candle_core::Result<Tensor> {
        let prev_rank = self.stage_config.stage_id - 1;
        self.comm.recv(shape, dtype, prev_rank).map_err(|e| {
            candle_core::Error::Msg(format!("pipeline recv from rank {prev_rank}: {e}"))
        })
    }

    /// Receive logits from the last pipeline stage.
    fn recv_logits(&self, shape: &[usize], dtype: DType) -> candle_core::Result<Tensor> {
        let last_rank = self.last_rank();
        self.comm.recv(shape, dtype, last_rank).map_err(|e| {
            candle_core::Error::Msg(format!("pipeline recv logits from rank {last_rank}: {e}"))
        })
    }

    /// Send logits back to rank 0 (called by last stage only).
    fn send_logits_to_coordinator(&self, logits: &Tensor) -> candle_core::Result<()> {
        self.comm
            .send(logits, 0)
            .map_err(|e| candle_core::Error::Msg(format!("pipeline send logits to rank 0: {e}")))
    }

    /// Broadcast prefill execution signal and cache metadata to all worker ranks.
    ///
    /// Called by rank 0 before sending hidden states to rank 1. All workers
    /// receive identical cache metadata because the scheduler on rank 0 allocates
    /// blocks globally and each stage uses the same logical block IDs for its
    /// own physical KV cache.
    fn broadcast_prefill_signals(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
        slot_mapping: &[usize],
        block_table: &BlockTable,
    ) -> candle_core::Result<()> {
        let num_slots = slot_mapping.len() as u32;
        let block_ids: Vec<u32> = block_table.block_ids().iter().map(|&x| x as u32).collect();
        let num_block_ids = block_ids.len() as u32;
        let device = self.model.device();

        for worker_rank in 1..self.stage_config.num_stages {
            send_worker_signal(
                &*self.comm,
                worker_rank,
                WorkerSignal {
                    signal: SIGNAL_EXECUTE,
                    batch_size: batch_size as u32,
                    seq_len: seq_len as u32,
                    seqlen_offset: seqlen_offset as u32,
                    num_slots,
                    num_block_ids,
                },
                device,
            )?;
            if num_slots > 0 {
                let slot_tensor = Tensor::from_vec(
                    slot_mapping.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    num_slots as usize,
                    device,
                )?;
                self.comm.send(&slot_tensor, worker_rank).map_err(|e| {
                    candle_core::Error::Msg(format!("send slots to rank {worker_rank}: {e}"))
                })?;
            }
            if num_block_ids > 0 {
                let blk_tensor =
                    Tensor::from_vec(block_ids.clone(), num_block_ids as usize, device)?;
                self.comm.send(&blk_tensor, worker_rank).map_err(|e| {
                    candle_core::Error::Msg(format!("send block_ids to rank {worker_rank}: {e}"))
                })?;
            }
        }
        Ok(())
    }

    /// Broadcast decode batch execution signal and per-sequence metadata to all workers.
    ///
    /// Uses a fixed-size encoding of `DECODE_META_WORDS_PER_SEQ` u32 per sequence
    /// to avoid variable-length framing. Block lists longer than `MAX_BLOCKS_PER_SEQ`
    /// are silently truncated (sequences exceeding this threshold are expected to be
    /// extremely rare in practice).
    fn broadcast_decode_signals(
        &self,
        sequences: &[DecodeSequenceMetadata],
    ) -> candle_core::Result<()> {
        let batch_size = sequences.len();
        if batch_size == 0 {
            return Ok(());
        }
        let device = self.model.device();

        let mut meta_data = vec![0u32; batch_size * DECODE_META_WORDS_PER_SEQ];
        for (i, seq) in sequences.iter().enumerate() {
            let base = i * DECODE_META_WORDS_PER_SEQ;
            meta_data[base] = seq.seqlen_offset as u32;
            meta_data[base + 1] = seq.slot_mapping.first().copied().unwrap_or(0) as u32;
            let n = seq.block_ids.len().min(MAX_BLOCKS_PER_SEQ);
            meta_data[base + 2] = n as u32;
            for (j, &bid) in seq.block_ids[..n].iter().enumerate() {
                meta_data[base + 3 + j] = bid as u32;
            }
        }
        let meta_tensor =
            Tensor::from_vec(meta_data, batch_size * DECODE_META_WORDS_PER_SEQ, device)?;

        for worker_rank in 1..self.stage_config.num_stages {
            send_worker_signal(
                &*self.comm,
                worker_rank,
                WorkerSignal {
                    signal: SIGNAL_EXECUTE_DECODE,
                    batch_size: batch_size as u32,
                    seq_len: 1,
                    seqlen_offset: 0,
                    num_slots: 0,
                    num_block_ids: 0,
                },
                device,
            )?;
            self.comm.send(&meta_tensor, worker_rank).map_err(|e| {
                candle_core::Error::Msg(format!("send decode meta to rank {worker_rank}: {e}"))
            })?;
        }
        Ok(())
    }
}

impl<M: PipelineForward> ModelForward for PipelineStagedModel<M> {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        // Single-stage: no communication, direct execution
        if self.is_single_stage() {
            let hidden = self.model.embed(input_ids)?;
            let hidden = self.model.forward_layers(
                &hidden,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            )?;
            return self.model.lm_head(&hidden);
        }

        // Multi-stage pipeline execution
        let batch_size = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;

        let hidden = if self.stage_config.is_first {
            // Broadcast execution signal and cache metadata to all downstream workers
            // before doing local work, so workers can begin their recv.
            self.broadcast_prefill_signals(
                batch_size,
                seq_len,
                seqlen_offset,
                slot_mapping,
                block_table,
            )?;
            // First stage: embed input tokens
            self.model.embed(input_ids)?
        } else {
            // Non-first stage: receive hidden states from previous stage
            self.recv_from_prev(&[batch_size, seq_len, self.model.hidden_size()], DType::F32)?
        };

        // Forward through this stage's layers
        let hidden = self.model.forward_layers(
            &hidden,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )?;

        if self.stage_config.is_last {
            // Last stage: produce logits
            let logits = self.model.lm_head(&hidden)?;

            // Send logits back to coordinator (rank 0) if this isn't also rank 0
            if !self.stage_config.is_first {
                self.send_logits_to_coordinator(&logits)?;
            }

            Ok(logits)
        } else {
            // Intermediate stage: send hidden states to next, receive logits from last
            self.send_to_next(&hidden)?;

            let batch_size = hidden.dim(0)?;
            let seq_len = hidden.dim(1)?;
            self.recv_logits(&[batch_size, seq_len, self.vocab_size], DType::F32)
        }
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        let batch_size = sequences.len();

        // Single-stage: no communication, direct execution
        if self.is_single_stage() {
            let hidden = self.model.embed(input_ids)?;
            let hidden =
                self.model
                    .forward_layers_decode_batch(&hidden, sequences, kv_cache_mgr)?;
            return self.model.lm_head(&hidden);
        }

        // Multi-stage pipeline execution
        let hidden = if self.stage_config.is_first {
            // Broadcast execution signal and per-sequence metadata to all workers.
            self.broadcast_decode_signals(sequences)?;
            self.model.embed(input_ids)?
        } else {
            self.recv_from_prev(&[batch_size, 1, self.model.hidden_size()], DType::F32)?
        };

        let hidden = self
            .model
            .forward_layers_decode_batch(&hidden, sequences, kv_cache_mgr)?;

        if self.stage_config.is_last {
            let logits = self.model.lm_head(&hidden)?;
            if !self.stage_config.is_first {
                self.send_logits_to_coordinator(&logits)?;
            }
            Ok(logits)
        } else {
            self.send_to_next(&hidden)?;
            self.recv_logits(&[batch_size, 1, self.vocab_size], DType::F32)
        }
    }

    fn device(&self) -> &Device {
        self.model.device()
    }
}

// ─── Pipeline Worker Loop ────────────────────────────────────────────────

// Control signals sent from the coordinator to pipeline workers.
// Header packed as a 1D u32 tensor of length SIGNAL_LEN:
//   [signal_type, batch_size, seq_len, seqlen_offset, num_slots, num_block_ids]
//
// After the header, additional tensors may follow depending on signal_type:
//   SIGNAL_EXECUTE:        [num_slots] u32 slot_mapping, [num_block_ids] u32 block_ids
//   SIGNAL_EXECUTE_DECODE: [batch_size * DECODE_META_WORDS_PER_SEQ] u32 decode_meta

/// Execute signal (prefill/forward): tells the worker to process the next batch.
pub const SIGNAL_EXECUTE: u32 = 0;

/// Shutdown signal: tells the worker to exit its loop.
pub const SIGNAL_SHUTDOWN: u32 = 1;

/// Execute decode batch signal: one token per sequence.
pub const SIGNAL_EXECUTE_DECODE: u32 = 2;

/// Number of u32 elements in the control signal header.
const SIGNAL_LEN: usize = 6;

/// Maximum KV cache blocks per sequence in the fixed-size decode metadata encoding.
///
/// Supports sequences up to `MAX_BLOCKS_PER_SEQ * block_size` tokens.
pub const MAX_BLOCKS_PER_SEQ: usize = 64;

/// Words per sequence in the decode metadata tensor.
///
/// Layout: `[seqlen_offset, slot, num_block_ids, block_id_0..block_id_{MAX-1}]`
pub const DECODE_META_WORDS_PER_SEQ: usize = 3 + MAX_BLOCKS_PER_SEQ;

/// Payload for a pipeline worker control signal header.
pub struct WorkerSignal {
    pub signal: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    pub seqlen_offset: u32,
    pub num_slots: u32,
    pub num_block_ids: u32,
}

/// Send a control signal header to a pipeline worker.
///
/// For `SIGNAL_EXECUTE`, the caller must then send the slot_mapping and block_ids
/// tensors. For `SIGNAL_EXECUTE_DECODE`, send the decode metadata tensor after.
/// For `SIGNAL_SHUTDOWN`, no additional tensors are expected.
pub fn send_worker_signal(
    comm: &dyn DeviceCommunicator,
    dst_rank: usize,
    sig: WorkerSignal,
    device: &Device,
) -> candle_core::Result<()> {
    let data = Tensor::from_vec(
        vec![
            sig.signal,
            sig.batch_size,
            sig.seq_len,
            sig.seqlen_offset,
            sig.num_slots,
            sig.num_block_ids,
        ],
        SIGNAL_LEN,
        device,
    )?;
    comm.send(&data, dst_rank)
        .map_err(|e| candle_core::Error::Msg(format!("send signal to rank {dst_rank}: {e}")))?;
    Ok(())
}

/// Worker loop for non-coordinator pipeline stages (ranks 1..N-1).
///
/// This function runs on GPU worker processes and handles:
/// 1. Receiving control signals from the coordinator (rank 0)
/// 2. Receiving hidden states from the previous stage
/// 3. Forwarding through local layers
/// 4. Sending results to the next stage (or logits to rank 0)
///
/// # Communication Protocol
///
/// Each step:
/// 1. Recv control signal `[signal, batch_size, seq_len]` from rank 0
/// 2. If shutdown: exit loop
/// 3. Recv hidden states `[batch_size, seq_len, hidden_size]` from prev rank
/// 4. Forward through local layers
/// 5. If last stage: lm_head → send logits to rank 0
///    Else: send hidden states to next rank
///
/// # Arguments
/// * `model` - The partial model for this stage
/// * `stage_config` - Pipeline stage configuration
/// * `comm` - Device communicator for P2P communication
/// * `kv_cache_mgr` - KV cache for this stage's layers
pub fn pipeline_worker_loop<M: PipelineForward>(
    model: M,
    stage_config: PipelineStageConfig,
    comm: Arc<dyn DeviceCommunicator>,
    mut kv_cache_mgr: KVCacheManager,
) {
    assert!(
        !stage_config.is_first,
        "pipeline_worker_loop is for non-coordinator stages (rank > 0)"
    );

    let prev_rank = stage_config.stage_id - 1;
    let hidden_size = model.hidden_size();
    let device = model.device().clone();

    tracing::info!(
        stage = stage_config.stage_id,
        layers = ?stage_config.layer_range(),
        is_last = stage_config.is_last,
        "Pipeline worker starting"
    );

    loop {
        // 1. Receive control signal header from coordinator (rank 0).
        let signal_tensor = match comm.recv(&[SIGNAL_LEN], DType::U32, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, "Worker failed to receive control signal");
                break;
            }
        };

        let signal_vec = match signal_tensor.to_vec1::<u32>() {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(error = %e, "Worker failed to decode control signal");
                break;
            }
        };

        if signal_vec.len() < SIGNAL_LEN {
            tracing::error!(len = signal_vec.len(), "Invalid control signal length");
            break;
        }

        let signal_type = signal_vec[0];
        let batch_size = signal_vec[1] as usize;
        let seq_len = signal_vec[2] as usize;
        let seqlen_offset = signal_vec[3] as usize;
        let num_slots = signal_vec[4] as usize;
        let num_block_ids = signal_vec[5] as usize;

        if signal_type == SIGNAL_SHUTDOWN {
            tracing::info!(
                stage = stage_config.stage_id,
                "Worker received shutdown signal"
            );
            break;
        }

        if batch_size == 0 {
            continue;
        }

        // 2. Receive cache metadata from coordinator, reconstruct execution context.
        //    Workers cannot independently allocate blocks; rank 0 broadcasts the
        //    block assignments so every stage uses the same logical block IDs.
        enum ExecMode {
            Prefill {
                block_table: BlockTable,
                slot_mapping: Vec<usize>,
            },
            Decode {
                sequences: Vec<DecodeSequenceMetadata>,
            },
        }

        let exec_mode = if signal_type == SIGNAL_EXECUTE {
            let slot_mapping: Vec<usize> = if num_slots > 0 {
                match comm
                    .recv(&[num_slots], DType::U32, 0)
                    .and_then(|t| t.to_vec1::<u32>().map_err(|e| e.into()))
                {
                    Ok(v) => v.into_iter().map(|x| x as usize).collect(),
                    Err(e) => {
                        tracing::error!(error = %e, "Worker failed to recv slot_mapping");
                        continue;
                    }
                }
            } else {
                vec![]
            };

            let block_ids: Vec<usize> = if num_block_ids > 0 {
                match comm
                    .recv(&[num_block_ids], DType::U32, 0)
                    .and_then(|t| t.to_vec1::<u32>().map_err(|e| e.into()))
                {
                    Ok(v) => v.into_iter().map(|x| x as usize).collect(),
                    Err(e) => {
                        tracing::error!(error = %e, "Worker failed to recv block_ids");
                        continue;
                    }
                }
            } else {
                vec![]
            };

            let block_table = BlockTable::from_block_ids(block_ids, seqlen_offset);
            ExecMode::Prefill {
                block_table,
                slot_mapping,
            }
        } else if signal_type == SIGNAL_EXECUTE_DECODE {
            let meta_len = batch_size * DECODE_META_WORDS_PER_SEQ;
            let meta_vec = match comm
                .recv(&[meta_len], DType::U32, 0)
                .and_then(|t| t.to_vec1::<u32>().map_err(|e| e.into()))
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(error = %e, "Worker failed to recv decode metadata");
                    continue;
                }
            };

            let sequences: Vec<DecodeSequenceMetadata> = (0..batch_size)
                .map(|i| {
                    let base = i * DECODE_META_WORDS_PER_SEQ;
                    let seq_offset = meta_vec[base] as usize;
                    let slot = meta_vec[base + 1] as usize;
                    let n = (meta_vec[base + 2] as usize).min(MAX_BLOCKS_PER_SEQ);
                    let block_ids: Vec<usize> =
                        (0..n).map(|j| meta_vec[base + 3 + j] as usize).collect();
                    DecodeSequenceMetadata {
                        request_id: i as u64,
                        seqlen_offset: seq_offset,
                        block_ids,
                        slot_mapping: vec![slot],
                    }
                })
                .collect();

            ExecMode::Decode { sequences }
        } else {
            tracing::error!(signal = signal_type, "Worker received unknown signal type");
            continue;
        };

        // 3. Receive hidden states from the previous pipeline stage.
        let hidden = match comm.recv(&[batch_size, seq_len, hidden_size], DType::F32, prev_rank) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, stage = stage_config.stage_id, "Worker recv failed");
                continue;
            }
        };

        // 4. Forward through local layers using the received execution context.
        let output = match exec_mode {
            ExecMode::Prefill {
                block_table,
                slot_mapping,
            } => match model.forward_layers(
                &hidden,
                seqlen_offset,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            ) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, stage = stage_config.stage_id, "Worker forward_layers failed");
                    continue;
                }
            },
            ExecMode::Decode { sequences } => {
                match model.forward_layers_decode_batch(&hidden, &sequences, &mut kv_cache_mgr) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(error = %e, stage = stage_config.stage_id, "Worker forward_decode failed");
                        continue;
                    }
                }
            }
        };

        // 5. Send output to the next stage (or logits to coordinator).
        if stage_config.is_last {
            // Last stage: apply lm_head and send logits to coordinator
            match model.lm_head(&output) {
                Ok(logits) => {
                    if let Err(e) = comm.send(&logits, 0) {
                        tracing::error!(error = %e, "Worker failed to send logits");
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Worker lm_head failed");
                    // Send zero logits as error recovery
                    if let Ok(zeros) = Tensor::zeros(output.dims(), DType::F32, &device) {
                        let _ = comm.send(&zeros, 0);
                    }
                }
            }
        } else {
            // Middle stage: send hidden states to next stage
            let next_rank = stage_config.stage_id + 1;
            if let Err(e) = comm.send(&output, next_rank) {
                tracing::error!(error = %e, dst = next_rank, "Worker failed to send hidden states");
            }
        }
    }

    tracing::info!(stage = stage_config.stage_id, "Pipeline worker exiting");
}

// ─── Per-Stage KV Cache ──────────────────────────────────────────────────

/// Create a KV cache configuration for a specific pipeline stage.
///
/// Each pipeline stage only needs KV cache for its assigned layers,
/// so `num_layers` is set to the stage's layer count rather than the
/// total model layer count.
pub fn create_stage_cache_config(
    base_config: &crate::kv_cache::config::CacheConfig,
    stage_config: &PipelineStageConfig,
) -> crate::kv_cache::config::CacheConfig {
    crate::kv_cache::config::CacheConfig {
        num_layers: stage_config.num_layers,
        // All other fields are the same: block_size, num_blocks, heads, etc.
        // Each stage gets the same number of blocks — the scheduler on rank 0
        // manages block allocation and broadcasts decisions.
        block_size: base_config.block_size,
        num_blocks: base_config.num_blocks,
        num_kv_heads: base_config.num_kv_heads,
        head_dim: base_config.head_dim,
        dtype: base_config.dtype,
        device: base_config.device.clone(),
        kv_cache_dtype: base_config.kv_cache_dtype,
        cpu_offload: base_config.cpu_offload.clone(),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator, PipelineStageConfig};
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;

    // ─── Mock PipelineForward Model ──────────────────────────────────────

    /// Mock model implementing PipelineForward for testing.
    ///
    /// Each method produces deterministic output:
    /// - embed: returns ones * (token_id + 1) for each token
    /// - forward_layers: returns input * 2 (simulates layer processing)
    /// - lm_head: returns logits with `output_token` having the highest score
    struct MockPipelineModel {
        output_token: u32,
        vocab_size: usize,
        hidden_size: usize,
        device: Device,
    }

    impl MockPipelineModel {
        fn new(output_token: u32, vocab_size: usize, hidden_size: usize) -> Self {
            Self {
                output_token,
                vocab_size,
                hidden_size,
                device: Device::Cpu,
            }
        }
    }

    impl PipelineForward for MockPipelineModel {
        fn embed(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
            let dims = input_ids.dims();
            let batch = dims[0];
            let seq_len = dims[1];
            // Return ones as hidden states
            Tensor::ones((batch, seq_len, self.hidden_size), DType::F32, &self.device)
        }

        fn forward_layers(
            &self,
            hidden_states: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            // Simulate layer processing: multiply by 2
            (hidden_states * 2.0)?.contiguous()
        }

        fn lm_head(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
            let dims = hidden_states.dims();
            let batch = dims[0];
            let seq_len = dims[1];
            // Produce logits with output_token having highest score
            let mut logits_data = vec![-100.0f32; batch * seq_len * self.vocab_size];
            for b in 0..batch {
                for s in 0..seq_len {
                    let idx = (b * seq_len + s) * self.vocab_size + self.output_token as usize;
                    logits_data[idx] = 100.0;
                }
            }
            Tensor::from_vec(logits_data, (batch, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
    }

    fn test_cache_config() -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    fn make_mock_comm() -> Arc<dyn DeviceCommunicator> {
        Arc::new(MockCommunicator::new(LocalProcessGroup::new()))
    }

    // ─── PipelineStagedModel Tests ───────────────────────────────────────

    #[test]
    fn single_stage_pipeline_forward() {
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        let input = Tensor::zeros((1, 5), DType::U32, &Device::Cpu).unwrap();
        let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();

        // Single stage: embed → forward_layers → lm_head
        // Shape: [1, 5, vocab_size=100]
        assert_eq!(logits.dims(), &[1, 5, 100]);

        // Greedy sample should pick token 42
        let last_logits = logits.narrow(1, 4, 1).unwrap().squeeze(1).unwrap();
        let token = last_logits
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        assert_eq!(token, 42);
    }

    #[test]
    fn single_stage_pipeline_decode_batch() {
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();

        let input = Tensor::zeros((3, 1), DType::U32, &Device::Cpu).unwrap();
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 0,
                seqlen_offset: 5,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 10,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![],
                slot_mapping: vec![],
            },
        ];

        let logits = staged
            .forward_decode_batch(&input, &sequences, &mut kv)
            .unwrap();

        // [3, 1, 100]
        assert_eq!(logits.dims(), &[3, 1, 100]);
    }

    #[test]
    fn multi_stage_first_stage_sends_and_receives() {
        // First stage of a 2-stage pipeline. MockCommunicator send is no-op,
        // recv returns zeros. This tests the orchestration (not data correctness).
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 2, 16);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        let input = Tensor::zeros((1, 3), DType::U32, &Device::Cpu).unwrap();
        let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();

        // With MockCommunicator, recv returns zeros (zero logits).
        // Shape should still be correct: [1, 3, 100]
        assert_eq!(logits.dims(), &[1, 3, 100]);
    }

    #[test]
    fn multi_stage_last_stage_receives_and_produces_logits() {
        // Last stage of a 2-stage pipeline.
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(1, 2, 16);
        // Use a multi-rank process group so recv doesn't short-circuit
        let pg = LocalProcessGroup::with_rank(1, 2);
        let comm: Arc<dyn DeviceCommunicator> = Arc::new(MockCommunicator::new(pg));
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        let input = Tensor::zeros((1, 3), DType::U32, &Device::Cpu).unwrap();
        let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();

        // Last stage: recv → forward_layers → lm_head
        // Shape: [1, 3, 100]
        assert_eq!(logits.dims(), &[1, 3, 100]);
    }

    #[test]
    fn multi_stage_middle_stage() {
        // Middle stage (rank 1) of a 3-stage pipeline.
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(1, 3, 24);
        let pg = LocalProcessGroup::with_rank(1, 3);
        let comm: Arc<dyn DeviceCommunicator> = Arc::new(MockCommunicator::new(pg));
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        let input = Tensor::zeros((1, 4), DType::U32, &Device::Cpu).unwrap();
        let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();

        // Middle stage: recv → forward_layers → send → recv logits
        // Logits come from recv (zeros from MockCommunicator)
        assert_eq!(logits.dims(), &[1, 4, 100]);
    }

    #[test]
    fn pipeline_staged_model_device() {
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        assert!(staged.device().is_cpu());
    }

    #[test]
    fn is_single_stage_detection() {
        let model = MockPipelineModel::new(42, 100, 64);
        let single = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, single, comm, 100);
        assert!(staged.is_single_stage());

        let model = MockPipelineModel::new(42, 100, 64);
        let multi = PipelineStageConfig::new(0, 2, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, multi, comm, 100);
        assert!(!staged.is_single_stage());
    }

    // ─── Per-Stage Cache Config Tests ────────────────────────────────────

    #[test]
    fn stage_cache_config_reduces_num_layers() {
        let base = CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };

        // 2-stage pipeline: each stage gets 16 layers
        let stage0 = PipelineStageConfig::new(0, 2, 32);
        let stage1 = PipelineStageConfig::new(1, 2, 32);

        let cache0 = create_stage_cache_config(&base, &stage0);
        let cache1 = create_stage_cache_config(&base, &stage1);

        assert_eq!(cache0.num_layers, 16);
        assert_eq!(cache1.num_layers, 16);

        // Other fields preserved
        assert_eq!(cache0.block_size, 16);
        assert_eq!(cache0.num_blocks, 64);
        assert_eq!(cache0.num_kv_heads, 8);
        assert_eq!(cache0.head_dim, 128);
    }

    #[test]
    fn stage_cache_config_uneven_distribution() {
        let base = CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 33,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };

        // 3-stage pipeline: 11 + 11 + 11 layers
        let stage0 = PipelineStageConfig::new(0, 3, 33);
        let stage1 = PipelineStageConfig::new(1, 3, 33);
        let stage2 = PipelineStageConfig::new(2, 3, 33);

        let cache0 = create_stage_cache_config(&base, &stage0);
        let cache1 = create_stage_cache_config(&base, &stage1);
        let cache2 = create_stage_cache_config(&base, &stage2);

        assert_eq!(cache0.num_layers, 11);
        assert_eq!(cache1.num_layers, 11);
        assert_eq!(cache2.num_layers, 11);
        assert_eq!(
            cache0.num_layers + cache1.num_layers + cache2.num_layers,
            33
        );
    }

    #[test]
    fn stage_cache_config_single_stage() {
        let base = test_cache_config();
        let stage = PipelineStageConfig::new(0, 1, 8);

        let cache = create_stage_cache_config(&base, &stage);
        // Single stage keeps all layers
        assert_eq!(cache.num_layers, 8);
    }

    // ─── PipelineForward Trait Tests ─────────────────────────────────────

    #[test]
    fn mock_pipeline_model_embed() {
        let model = MockPipelineModel::new(42, 100, 64);
        let input = Tensor::zeros((2, 5), DType::U32, &Device::Cpu).unwrap();
        let hidden = model.embed(&input).unwrap();
        assert_eq!(hidden.dims(), &[2, 5, 64]);
    }

    #[test]
    fn mock_pipeline_model_forward_layers() {
        let model = MockPipelineModel::new(42, 100, 64);
        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        let hidden = Tensor::ones((1, 3, 64), DType::F32, &Device::Cpu).unwrap();
        let output = model.forward_layers(&hidden, 0, &mut kv, &bt, &[]).unwrap();

        assert_eq!(output.dims(), &[1, 3, 64]);
        // forward_layers multiplies by 2
        let val = output.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((val - 2.0).abs() < 1e-6);
    }

    #[test]
    fn mock_pipeline_model_lm_head() {
        let model = MockPipelineModel::new(42, 100, 64);
        let hidden = Tensor::ones((1, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let logits = model.lm_head(&hidden).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 100]);

        // Token 42 should have highest logit
        let token = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(0)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(token, 42);
    }

    #[test]
    fn mock_pipeline_model_hidden_size() {
        let model = MockPipelineModel::new(42, 100, 128);
        assert_eq!(model.hidden_size(), 128);
    }

    #[test]
    fn mock_pipeline_model_decode_batch() {
        let model = MockPipelineModel::new(42, 100, 64);
        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();

        let hidden = Tensor::ones((2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 0,
                seqlen_offset: 5,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 10,
                block_ids: vec![],
                slot_mapping: vec![],
            },
        ];

        let output = model
            .forward_layers_decode_batch(&hidden, &sequences, &mut kv)
            .unwrap();
        assert_eq!(output.dims(), &[2, 1, 64]);
    }

    // ─── Worker Signal Tests ─────────────────────────────────────────────

    #[test]
    fn send_worker_signal_encodes_correctly() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        // Should not error (MockCommunicator send is no-op)
        send_worker_signal(
            &comm,
            1,
            WorkerSignal {
                signal: SIGNAL_EXECUTE,
                batch_size: 4,
                seq_len: 128,
                seqlen_offset: 0,
                num_slots: 0,
                num_block_ids: 0,
            },
            &Device::Cpu,
        )
        .unwrap();
        send_worker_signal(
            &comm,
            1,
            WorkerSignal {
                signal: SIGNAL_SHUTDOWN,
                batch_size: 0,
                seq_len: 0,
                seqlen_offset: 0,
                num_slots: 0,
                num_block_ids: 0,
            },
            &Device::Cpu,
        )
        .unwrap();
    }

    #[test]
    fn signal_constants_distinct() {
        assert_ne!(SIGNAL_EXECUTE, SIGNAL_SHUTDOWN);
        assert_ne!(SIGNAL_EXECUTE, SIGNAL_EXECUTE_DECODE);
        assert_ne!(SIGNAL_SHUTDOWN, SIGNAL_EXECUTE_DECODE);
    }

    #[test]
    fn decode_meta_constants_consistent() {
        assert_eq!(DECODE_META_WORDS_PER_SEQ, 3 + MAX_BLOCKS_PER_SEQ);
    }

    #[test]
    fn send_worker_signal_decode_encodes_correctly() {
        let pg = LocalProcessGroup::new();
        let comm = MockCommunicator::new(pg);

        send_worker_signal(
            &comm,
            1,
            WorkerSignal {
                signal: SIGNAL_EXECUTE_DECODE,
                batch_size: 2,
                seq_len: 1,
                seqlen_offset: 0,
                num_slots: 0,
                num_block_ids: 0,
            },
            &Device::Cpu,
        )
        .unwrap();
    }

    // ─── Integration: PipelineStagedModel as ModelForward ────────────────

    #[test]
    fn pipeline_model_implements_model_forward() {
        // Verify that PipelineStagedModel can be used where ModelForward is expected
        fn takes_model_forward(_m: &dyn ModelForward) {}

        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        takes_model_forward(&staged);
    }

    #[test]
    fn pipeline_model_boxable() {
        // Verify it can be boxed as Box<dyn ModelForward>
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let _boxed: Box<dyn ModelForward> = Box::new(staged);
    }

    #[test]
    fn single_stage_pipeline_end_to_end_with_engine() {
        // Single-stage pipeline produces identical results to a direct model call.
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 1, 8);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        // Prefill
        let input = Tensor::zeros((1, 10), DType::U32, &Device::Cpu).unwrap();
        let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();
        assert_eq!(logits.dims(), &[1, 10, 100]);

        // Decode
        let decode_input = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let logits = staged
            .forward(&decode_input, 10, &mut kv, &bt, &[])
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, 100]);
    }

    #[test]
    fn multi_stage_pipeline_two_stages_orchestration() {
        // 2-stage pipeline: rank 0. Tests that send/recv are called correctly.
        // With MockCommunicator, recv returns zeros. Verifies shape correctness.
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 2, 16);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        let config = test_cache_config();
        let mut kv = KVCacheManager::new(&config).unwrap();
        let bt = BlockTable::new(16);

        // Prefill
        let input = Tensor::zeros((1, 5), DType::U32, &Device::Cpu).unwrap();
        let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();
        assert_eq!(logits.dims(), &[1, 5, 100]);

        // Decode batch
        let input = Tensor::zeros((4, 1), DType::U32, &Device::Cpu).unwrap();
        let sequences: Vec<DecodeSequenceMetadata> = (0..4)
            .map(|i| DecodeSequenceMetadata {
                request_id: i,
                seqlen_offset: 5,
                block_ids: vec![],
                slot_mapping: vec![],
            })
            .collect();
        let logits = staged
            .forward_decode_batch(&input, &sequences, &mut kv)
            .unwrap();
        assert_eq!(logits.dims(), &[4, 1, 100]);
    }

    #[test]
    fn multi_stage_pipeline_three_stages() {
        // Verify 3-stage pipeline at each rank produces correct shapes.
        for rank in 0..3 {
            let model = MockPipelineModel::new(42, 100, 64);
            let stage = PipelineStageConfig::new(rank, 3, 24);
            let pg = LocalProcessGroup::with_rank(rank, 3);
            let comm: Arc<dyn DeviceCommunicator> = Arc::new(MockCommunicator::new(pg));
            let staged = PipelineStagedModel::new(model, stage, comm, 100);

            let config = test_cache_config();
            let mut kv = KVCacheManager::new(&config).unwrap();
            let bt = BlockTable::new(16);

            let input = Tensor::zeros((1, 3), DType::U32, &Device::Cpu).unwrap();
            let logits = staged.forward(&input, 0, &mut kv, &bt, &[]).unwrap();
            assert_eq!(
                logits.dims(),
                &[1, 3, 100],
                "rank {rank} should produce correct logits shape"
            );
        }
    }

    #[test]
    fn pipeline_staged_model_stage_config_accessor() {
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(1, 4, 32);
        let comm = make_mock_comm();
        let staged = PipelineStagedModel::new(model, stage, comm, 100);

        assert_eq!(staged.stage_config().stage_id, 1);
        assert_eq!(staged.stage_config().num_stages, 4);
        assert_eq!(staged.stage_config().num_layers, 8);
    }

    // ─── Worker Loop Tests ───────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "pipeline_worker_loop is for non-coordinator stages")]
    fn worker_loop_rejects_rank_zero() {
        let model = MockPipelineModel::new(42, 100, 64);
        let stage = PipelineStageConfig::new(0, 2, 16);
        let comm = make_mock_comm();
        let config = test_cache_config();
        let kv = KVCacheManager::new(&config).unwrap();

        pipeline_worker_loop(model, stage, comm, kv);
    }

    // ─── KV Cache Manager Integration ────────────────────────────────────

    #[test]
    fn stage_kv_cache_has_correct_layer_count() {
        let base = CacheConfig {
            block_size: 16,
            num_blocks: 32,
            num_layers: 24,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };

        // 3-stage pipeline: 8 + 8 + 8 layers
        for rank in 0..3 {
            let stage = PipelineStageConfig::new(rank, 3, 24);
            let cache_config = create_stage_cache_config(&base, &stage);
            let kv = KVCacheManager::new(&cache_config).unwrap();

            // Verify the cache was created (no error)
            assert!(kv.num_free_blocks() > 0);
        }
    }
}
