//! ModelForward trait for model inference.

use candle_core::{Device, Tensor};

use crate::kv_cache::{BlockId, BlockTable, KVCacheManager};

use super::cuda_graph::ForwardContext;

/// Per-sequence metadata for batched decode (one token per sequence).
pub struct DecodeSequenceMetadata {
    pub seqlen_offset: usize,
    pub block_ids: Vec<BlockId>,
    pub slot_mapping: Vec<usize>,
}

/// Trait for model forward pass, enabling different model implementations.
pub trait ModelForward: Send + 'static {
    /// Single-sequence forward pass for prefill or decode.
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor>;

    /// Batched decode: process multiple sequences each generating one token.
    /// Default implementation falls back to sequential forward calls.
    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let token = input_ids.narrow(0, i, 1)?;
            let block_table = BlockTable::from_block_ids(seq.block_ids.clone(), seq.seqlen_offset);
            let logits = self.forward(
                &token,
                seq.seqlen_offset,
                kv_cache_mgr,
                &block_table,
                &seq.slot_mapping,
            )?;
            outputs.push(logits);
        }
        Tensor::cat(&outputs, 0)
    }

    /// Batched decode with CUDA graph context.
    ///
    /// This method provides forward context for CUDA graph capture and replay.
    /// The default implementation ignores the context and delegates to
    /// `forward_decode_batch`.
    ///
    /// Models that support CUDA graph capture should override this method
    /// to handle the context appropriately.
    fn forward_decode_batch_with_ctx(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &KVCacheManager,
        ctx: &ForwardContext,
    ) -> candle_core::Result<Tensor> {
        // Default: ignore context, use eager path
        let _ = ctx;
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr)
    }

    /// Whether this model supports CUDA graph capture.
    ///
    /// Returns `true` if the model can be captured into a CUDA graph.
    /// This requires the model to use static memory allocation patterns
    /// and avoid operations that are incompatible with graph capture.
    fn supports_cuda_graphs(&self) -> bool {
        false
    }

    fn device(&self) -> &Device;
}

impl ModelForward for Box<dyn ModelForward> {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        (**self).forward(
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
        kv_cache_mgr: &KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_decode_batch(input_ids, sequences, kv_cache_mgr)
    }

    fn forward_decode_batch_with_ctx(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &KVCacheManager,
        ctx: &ForwardContext,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_decode_batch_with_ctx(input_ids, sequences, kv_cache_mgr, ctx)
    }

    fn supports_cuda_graphs(&self) -> bool {
        (**self).supports_cuda_graphs()
    }

    fn device(&self) -> &Device {
        (**self).device()
    }
}
