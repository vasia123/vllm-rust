//! ModelForward trait for model inference.

use candle_core::{Device, Tensor};

use crate::kv_cache::{BlockId, BlockTable, KVCacheManager};
use crate::lora::LoraContext;
use crate::multimodal::MultimodalInputs;

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
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor>;

    /// Batched decode: process multiple sequences each generating one token.
    /// Default implementation falls back to sequential forward calls.
    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
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
        kv_cache_mgr: &mut KVCacheManager,
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

    /// Whether this model supports LoRA adapters.
    ///
    /// Returns `true` if the model can use LoRA adapters for per-request
    /// fine-tuning behavior. LoRA-enabled models should override this and
    /// the `forward_with_lora` methods.
    fn supports_lora(&self) -> bool {
        false
    }

    /// Single-sequence forward pass with LoRA adapter support.
    ///
    /// For models that support LoRA, this method applies the specified adapter
    /// during inference. For models that don't support LoRA, this delegates to
    /// the standard forward pass and ignores the LoRA context.
    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> candle_core::Result<Tensor> {
        // Default: ignore LoRA context and use base forward
        let _ = lora_ctx;
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    /// Batched decode with LoRA adapter support.
    ///
    /// For models that support LoRA, this method applies the specified adapter
    /// during batched decode. For models that don't support LoRA, this delegates
    /// to the standard batched decode and ignores the LoRA context.
    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> candle_core::Result<Tensor> {
        // Default: ignore LoRA context and use base batched decode
        let _ = lora_ctx;
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr)
    }

    fn device(&self) -> &Device;

    /// Whether this model supports multimodal (image/video) inputs.
    ///
    /// Returns `true` if the model can process images along with text.
    /// Multimodal-enabled models should override this and the
    /// `forward_multimodal` method.
    fn supports_multimodal(&self) -> bool {
        false
    }

    /// Forward pass with multimodal inputs (images/video).
    ///
    /// For models that support multimodal inputs (like LLaVA), this method
    /// processes image embeddings and merges them with text embeddings at
    /// the appropriate positions.
    ///
    /// For models that don't support multimodal, this delegates to the
    /// standard forward pass and ignores the multimodal inputs.
    ///
    /// # Arguments
    /// * `input_ids` - Text token IDs [batch, seq_len]
    /// * `multimodal_inputs` - Optional multimodal data (images, positions)
    /// * `seqlen_offset` - Position offset for RoPE
    /// * `kv_cache_mgr` - KV cache manager
    /// * `block_table` - Block table for paged attention
    /// * `slot_mapping` - Slot mapping for cache
    #[allow(unused_variables)]
    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        // Default: ignore multimodal inputs and use base forward
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}

impl ModelForward for Box<dyn ModelForward> {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
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
        kv_cache_mgr: &mut KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_decode_batch(input_ids, sequences, kv_cache_mgr)
    }

    fn forward_decode_batch_with_ctx(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        ctx: &ForwardContext,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_decode_batch_with_ctx(input_ids, sequences, kv_cache_mgr, ctx)
    }

    fn supports_cuda_graphs(&self) -> bool {
        (**self).supports_cuda_graphs()
    }

    fn supports_lora(&self) -> bool {
        (**self).supports_lora()
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_with_lora(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_decode_batch_with_lora(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn device(&self) -> &Device {
        (**self).device()
    }

    fn supports_multimodal(&self) -> bool {
        (**self).supports_multimodal()
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        (**self).forward_multimodal(
            input_ids,
            multimodal_inputs,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}
