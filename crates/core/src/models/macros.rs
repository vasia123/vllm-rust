/// Generates `ModelForward` delegation for an enum whose variants each wrap
/// a concrete type that implements `ModelForward`.
///
/// Only the `ModelForward` trait methods are generated. Any additional
/// methods (e.g. LoRA registration) must be implemented manually on the
/// enum in a separate `impl` block.
macro_rules! delegate_model_forward {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $($variant:ident($type:ty)),* $(,)?
        }
    ) => {
        $(#[$meta])*
        pub enum $name {
            $($variant($type)),*
        }

        impl $crate::engine::ModelForward for $name {
            fn forward(
                &self,
                input_ids: &::candle_core::Tensor,
                seqlen_offset: usize,
                kv_cache_mgr: &mut $crate::kv_cache::KVCacheManager,
                block_table: &$crate::kv_cache::BlockTable,
                slot_mapping: &[usize],
            ) -> ::candle_core::Result<::candle_core::Tensor> {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::forward(
                            m, input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping,
                        )
                    }),*
                }
            }

            fn forward_decode_batch(
                &self,
                input_ids: &::candle_core::Tensor,
                sequences: &[$crate::engine::DecodeSequenceMetadata],
                kv_cache_mgr: &mut $crate::kv_cache::KVCacheManager,
            ) -> ::candle_core::Result<::candle_core::Tensor> {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::forward_decode_batch(
                            m, input_ids, sequences, kv_cache_mgr,
                        )
                    }),*
                }
            }

            fn forward_decode_batch_with_ctx(
                &self,
                input_ids: &::candle_core::Tensor,
                sequences: &[$crate::engine::DecodeSequenceMetadata],
                kv_cache_mgr: &mut $crate::kv_cache::KVCacheManager,
                ctx: &$crate::engine::cuda_graph::ForwardContext,
            ) -> ::candle_core::Result<::candle_core::Tensor> {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::forward_decode_batch_with_ctx(
                            m, input_ids, sequences, kv_cache_mgr, ctx,
                        )
                    }),*
                }
            }

            fn supports_cuda_graphs(&self) -> bool {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::supports_cuda_graphs(m)
                    }),*
                }
            }

            fn supports_lora(&self) -> bool {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::supports_lora(m)
                    }),*
                }
            }

            fn forward_with_lora(
                &self,
                input_ids: &::candle_core::Tensor,
                seqlen_offset: usize,
                kv_cache_mgr: &mut $crate::kv_cache::KVCacheManager,
                block_table: &$crate::kv_cache::BlockTable,
                slot_mapping: &[usize],
                lora_ctx: &$crate::lora::LoraContext,
            ) -> ::candle_core::Result<::candle_core::Tensor> {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::forward_with_lora(
                            m, input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping, lora_ctx,
                        )
                    }),*
                }
            }

            fn forward_decode_batch_with_lora(
                &self,
                input_ids: &::candle_core::Tensor,
                sequences: &[$crate::engine::DecodeSequenceMetadata],
                kv_cache_mgr: &mut $crate::kv_cache::KVCacheManager,
                lora_ctx: &$crate::lora::LoraContext,
            ) -> ::candle_core::Result<::candle_core::Tensor> {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::forward_decode_batch_with_lora(
                            m, input_ids, sequences, kv_cache_mgr, lora_ctx,
                        )
                    }),*
                }
            }

            fn device(&self) -> &::candle_core::Device {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::device(m)
                    }),*
                }
            }

            fn supports_multimodal(&self) -> bool {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::supports_multimodal(m)
                    }),*
                }
            }

            fn forward_multimodal(
                &self,
                input_ids: &::candle_core::Tensor,
                multimodal_inputs: Option<&$crate::multimodal::MultimodalInputs>,
                seqlen_offset: usize,
                kv_cache_mgr: &mut $crate::kv_cache::KVCacheManager,
                block_table: &$crate::kv_cache::BlockTable,
                slot_mapping: &[usize],
            ) -> ::candle_core::Result<::candle_core::Tensor> {
                match self {
                    $(Self::$variant(m) => {
                        $crate::engine::ModelForward::forward_multimodal(
                            m, input_ids, multimodal_inputs, seqlen_offset, kv_cache_mgr, block_table, slot_mapping,
                        )
                    }),*
                }
            }
        }
    };
}
