//! Tensor parallelism execution for distributed inference.
//!
//! All TP ranks run the same model forward pass in lockstep. The coordinator
//! (rank 0) broadcasts the input tensors and cache metadata before each step;
//! all ranks then call the model forward, which internally executes NCCL
//! all-reduce operations inside every tensor-parallel layer. Rank 0 uses the
//! logits; workers discard their output and loop back.
//!
//! # Signal protocol
//!
//! Before each forward pass, rank 0 broadcasts a 6-element u32 header:
//! `[signal, batch_size, seq_len, seqlen_offset, num_slots, num_block_ids]`
//!
//! For `TP_SIGNAL_EXECUTE`:
//!   1. `input_ids` flattened as `[batch_size * seq_len]` u32
//!   2. `slot_mapping` as `[num_slots]` u32  (absent when `num_slots == 0`)
//!   3. `block_ids`   as `[num_block_ids]` u32  (absent when `num_block_ids == 0`)
//!
//! For `TP_SIGNAL_EXECUTE_DECODE`:
//!   1. `input_ids` flattened as `[batch_size]` u32
//!   2. decode metadata as `[batch_size * DECODE_META_WORDS_PER_SEQ]` u32
//!
//! For `TP_SIGNAL_SHUTDOWN`: no additional tensors.
//!
//! # Relationship to pipeline parallelism
//!
//! PP splits layers; TP splits weights. The two are conceptually orthogonal.
//! PP uses point-to-point send/recv between consecutive stages. TP uses
//! broadcast (single-source, all-receive) to distribute inputs and relies on
//! NCCL all-reduce inside each layer to synchronise partial results.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use crate::distributed::DeviceCommunicator;
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::model_forward::{DecodeSequenceMetadata, ModelForward};
use super::pipeline::{DECODE_META_WORDS_PER_SEQ, MAX_BLOCKS_PER_SEQ};

// ─── Signal constants ─────────────────────────────────────────────────────────

/// Prefill execute: broadcast `input_ids` + cache metadata, then run `forward`.
pub const TP_SIGNAL_EXECUTE: u32 = 0;

/// Decode execute: one token per sequence.
pub const TP_SIGNAL_EXECUTE_DECODE: u32 = 1;

/// Shutdown: workers break out of `tensor_worker_loop`.
pub const TP_SIGNAL_SHUTDOWN: u32 = 2;

/// Number of u32 elements in the broadcast header tensor.
const TP_SIGNAL_LEN: usize = 6;

// ─── TpStagedModel ────────────────────────────────────────────────────────────

/// Coordinator wrapper that broadcasts inputs to TP workers before each forward.
///
/// Implements [`ModelForward`] so it plugs transparently into
/// [`StandardExecution`](super::standard::StandardExecution).
///
/// When `tp_size == 1` the wrapper is a no-op pass-through: no NCCL calls
/// are made and the inner model runs as if TP were disabled.
pub struct TpStagedModel {
    model: Box<dyn ModelForward>,
    comm: Arc<dyn DeviceCommunicator>,
    tp_size: usize,
}

impl TpStagedModel {
    /// Wrap a model built with [`from_config_with_tp`](crate::models::from_config_with_tp).
    ///
    /// # Arguments
    /// * `model`   — coordinator's model (rank 0 weight shards)
    /// * `comm`    — NCCL communicator shared with all TP workers
    /// * `tp_size` — total number of TP ranks (== NCCL world size)
    pub fn new(
        model: Box<dyn ModelForward>,
        comm: Arc<dyn DeviceCommunicator>,
        tp_size: usize,
    ) -> Self {
        Self {
            model,
            comm,
            tp_size,
        }
    }

    // Broadcast a u32 tensor from rank 0.
    fn bcast_u32(&self, data: Vec<u32>, len: usize) -> candle_core::Result<()> {
        let t = Tensor::from_vec(data, len, self.model.device())?;
        self.comm
            .broadcast(&t, 0)
            .map_err(|e| candle_core::Error::Msg(format!("TP broadcast u32: {e}")))?;
        Ok(())
    }

    fn broadcast_prefill(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
        input_ids: &Tensor,
        slot_mapping: &[usize],
        block_table: &BlockTable,
    ) -> candle_core::Result<()> {
        let num_slots = slot_mapping.len();
        let block_ids: Vec<u32> = block_table.block_ids().iter().map(|&x| x as u32).collect();
        let num_block_ids = block_ids.len();

        // Header
        self.bcast_u32(
            vec![
                TP_SIGNAL_EXECUTE,
                batch_size as u32,
                seq_len as u32,
                seqlen_offset as u32,
                num_slots as u32,
                num_block_ids as u32,
            ],
            TP_SIGNAL_LEN,
        )?;

        // input_ids (cast to u32 for uniform encoding; token IDs always fit)
        let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
        self.comm
            .broadcast(&ids, 0)
            .map_err(|e| candle_core::Error::Msg(format!("TP broadcast input_ids: {e}")))?;

        if num_slots > 0 {
            self.bcast_u32(slot_mapping.iter().map(|&x| x as u32).collect(), num_slots)?;
        }
        if num_block_ids > 0 {
            self.bcast_u32(block_ids, num_block_ids)?;
        }
        Ok(())
    }

    fn broadcast_decode(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
    ) -> candle_core::Result<()> {
        let batch_size = sequences.len();

        // Header
        self.bcast_u32(
            vec![
                TP_SIGNAL_EXECUTE_DECODE,
                batch_size as u32,
                1u32,
                0u32,
                0u32,
                0u32,
            ],
            TP_SIGNAL_LEN,
        )?;

        // input_ids
        let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
        self.comm
            .broadcast(&ids, 0)
            .map_err(|e| candle_core::Error::Msg(format!("TP broadcast decode ids: {e}")))?;

        // Decode metadata: same fixed-size encoding as PP
        let mut meta = vec![0u32; batch_size * DECODE_META_WORDS_PER_SEQ];
        for (i, seq) in sequences.iter().enumerate() {
            let base = i * DECODE_META_WORDS_PER_SEQ;
            meta[base] = seq.seqlen_offset as u32;
            meta[base + 1] = seq.slot_mapping.first().copied().unwrap_or(0) as u32;
            let n = seq.block_ids.len().min(MAX_BLOCKS_PER_SEQ);
            meta[base + 2] = n as u32;
            for (j, &bid) in seq.block_ids[..n].iter().enumerate() {
                meta[base + 3 + j] = bid as u32;
            }
        }
        self.bcast_u32(meta, batch_size * DECODE_META_WORDS_PER_SEQ)?;
        Ok(())
    }
}

impl Drop for TpStagedModel {
    fn drop(&mut self) {
        // Best-effort shutdown broadcast. Errors are suppressed: NCCL may
        // already be torn down when this runs (e.g. during process exit).
        let device = self.model.device().clone();
        if let Ok(hdr) = Tensor::from_vec(
            vec![TP_SIGNAL_SHUTDOWN, 0u32, 0, 0, 0, 0],
            TP_SIGNAL_LEN,
            &device,
        ) {
            let _ = self.comm.broadcast(&hdr, 0);
        }
    }
}

impl ModelForward for TpStagedModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        if self.tp_size > 1 {
            let batch_size = input_ids.dim(0)?;
            let seq_len = input_ids.dim(1)?;
            self.broadcast_prefill(
                batch_size,
                seq_len,
                seqlen_offset,
                input_ids,
                slot_mapping,
                block_table,
            )?;
        }
        self.model.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        if self.tp_size > 1 {
            self.broadcast_decode(input_ids, sequences)?;
        }
        self.model
            .forward_decode_batch(input_ids, sequences, kv_cache_mgr)
    }

    fn device(&self) -> &Device {
        self.model.device()
    }
}

// ─── Worker loop ──────────────────────────────────────────────────────────────

/// Worker loop for non-coordinator TP ranks (ranks 1..N-1).
///
/// This function blocks, receiving broadcast signals from rank 0 and
/// participating in all-reduce operations inside the model forward pass.
/// It exits cleanly when it receives `TP_SIGNAL_SHUTDOWN` or when the
/// NCCL communicator errors (e.g. coordinator process exits).
///
/// # Arguments
/// * `model`       — TP-aware model for this rank's weight shards
/// * `comm`        — NCCL communicator (same process group as coordinator)
/// * `kv_cache_mgr` — KV cache for this rank (mirrors coordinator's layout)
pub fn tensor_worker_loop(
    model: Box<dyn ModelForward>,
    comm: Arc<dyn DeviceCommunicator>,
    mut kv_cache_mgr: KVCacheManager,
) {
    let device = model.device().clone();

    tracing::info!("TP worker entering loop");

    loop {
        // 1. Receive broadcast signal header from coordinator (rank 0).
        let hdr = match recv_broadcast_u32(&comm, TP_SIGNAL_LEN, &device) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(error = %e, "TP worker: failed to receive signal header");
                break;
            }
        };

        if hdr.len() < TP_SIGNAL_LEN {
            tracing::error!(len = hdr.len(), "TP worker: invalid header length");
            break;
        }

        let signal_type = hdr[0];
        let batch_size = hdr[1] as usize;
        let seq_len = hdr[2] as usize;
        let seqlen_offset = hdr[3] as usize;
        let num_slots = hdr[4] as usize;
        let num_block_ids = hdr[5] as usize;

        if signal_type == TP_SIGNAL_SHUTDOWN {
            tracing::info!("TP worker: received shutdown signal, exiting");
            break;
        }

        if batch_size == 0 {
            continue;
        }

        if signal_type == TP_SIGNAL_EXECUTE {
            // 2a. Receive input_ids [batch_size * seq_len] u32
            let ids_vec = match recv_broadcast_u32(&comm, batch_size * seq_len, &device) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(error = %e, "TP worker: recv input_ids failed");
                    continue;
                }
            };
            let input_ids = match Tensor::from_vec(ids_vec.clone(), (batch_size, seq_len), &device)
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, "TP worker: input_ids reshape failed");
                    continue;
                }
            };

            // 2b. Receive slot_mapping
            let slot_mapping: Vec<usize> = if num_slots > 0 {
                match recv_broadcast_u32(&comm, num_slots, &device) {
                    Ok(v) => v.into_iter().map(|x| x as usize).collect(),
                    Err(e) => {
                        tracing::error!(error = %e, "TP worker: recv slot_mapping failed");
                        continue;
                    }
                }
            } else {
                vec![]
            };

            // 2c. Receive block_ids
            let block_ids: Vec<usize> = if num_block_ids > 0 {
                match recv_broadcast_u32(&comm, num_block_ids, &device) {
                    Ok(v) => v.into_iter().map(|x| x as usize).collect(),
                    Err(e) => {
                        tracing::error!(error = %e, "TP worker: recv block_ids failed");
                        continue;
                    }
                }
            } else {
                vec![]
            };

            let block_table = BlockTable::from_block_ids(block_ids, seqlen_offset);

            // 3. Run forward — TP layers all-reduce internally; discard output.
            if let Err(e) = model.forward(
                &input_ids,
                seqlen_offset,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            ) {
                tracing::error!(error = %e, "TP worker: forward failed");
            }
        } else if signal_type == TP_SIGNAL_EXECUTE_DECODE {
            // 2a. Receive input_ids [batch_size] u32
            let ids_vec = match recv_broadcast_u32(&comm, batch_size, &device) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(error = %e, "TP worker: recv decode input_ids failed");
                    continue;
                }
            };
            let input_ids = match Tensor::from_vec(ids_vec, batch_size, &device) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, "TP worker: decode input_ids reshape failed");
                    continue;
                }
            };

            // 2b. Receive decode metadata
            let meta_len = batch_size * DECODE_META_WORDS_PER_SEQ;
            let meta_vec = match recv_broadcast_u32(&comm, meta_len, &device) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(error = %e, "TP worker: recv decode meta failed");
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

            // 3. Run decode forward; discard output.
            if let Err(e) = model.forward_decode_batch(&input_ids, &sequences, &mut kv_cache_mgr) {
                tracing::error!(error = %e, "TP worker: forward_decode_batch failed");
            }
        } else {
            tracing::error!(signal = signal_type, "TP worker: unknown signal type");
        }
    }

    tracing::info!("TP worker exiting");
}

// Receive a broadcast from rank 0 by allocating a correctly-shaped placeholder
// tensor and calling `broadcast`. The returned `Vec<u32>` is host-accessible.
fn recv_broadcast_u32(
    comm: &Arc<dyn DeviceCommunicator>,
    len: usize,
    device: &Device,
) -> candle_core::Result<Vec<u32>> {
    // Allocate a zero-filled placeholder; broadcast overwrites its content.
    let placeholder = Tensor::zeros(len, DType::U32, device)?;
    let received = comm
        .broadcast(&placeholder, 0)
        .map_err(|e| candle_core::Error::Msg(format!("TP recv broadcast: {e}")))?;
    received
        .to_vec1::<u32>()
        .map_err(|e| candle_core::Error::Msg(format!("TP broadcast to_vec1: {e}")))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator};

    struct PassthroughModel {
        device: Device,
    }

    impl ModelForward for PassthroughModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            // Return ones with the same batch/seq shape but vocab_size=8
            Tensor::ones(
                (input_ids.dim(0)?, input_ids.dim(1)?, 8),
                DType::F32,
                &self.device,
            )
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    #[test]
    fn tp_staged_single_gpu_passthrough() {
        let device = Device::Cpu;
        let model = Box::new(PassthroughModel {
            device: device.clone(),
        });
        let mock_comm = Arc::new(MockCommunicator::new(LocalProcessGroup::new()));
        let staged = TpStagedModel::new(model, mock_comm, 1);

        let ids = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let cache_cfg = crate::kv_cache::config::CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv = KVCacheManager::new(&cache_cfg).unwrap();
        let bt = BlockTable::from_block_ids(vec![], 0);

        let logits = staged.forward(&ids, 0, &mut kv, &bt, &[]).unwrap();
        assert_eq!(logits.dims(), [1, 4, 8]);
    }
}
