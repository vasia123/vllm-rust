//! Encoder-decoder model trait for sequence-to-sequence inference.
//!
//! This module provides the `ModelForEncoderDecoder` trait for models with
//! separate encoder and decoder components (T5, BART, mBART, etc.).
//!
//! # Architecture
//!
//! Encoder-decoder models have two distinct phases:
//! 1. **Encode**: Process the source input once (encoder is bidirectional)
//! 2. **Decode**: Autoregressive generation attending to encoder output via cross-attention
//!
//! The encoder output is computed once and cached for the entire decoding process,
//! which is the key difference from decoder-only models where all context is
//! processed through the same autoregressive mechanism.
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::engine::{ModelForEncoderDecoder, EncoderOutput};
//!
//! // Phase 1: Encode source once
//! let encoder_output = model.encode(&source_ids, Some(&source_mask))?;
//!
//! // Phase 2: Decode autoregressively
//! let logits = model.decode(
//!     &decoder_ids,
//!     &encoder_output,
//!     seqlen_offset,
//!     &mut kv_cache_mgr,
//!     &block_table,
//!     &slot_mapping,
//! )?;
//! ```

use candle_core::{Device, Result, Tensor};

use crate::kv_cache::{BlockTable, KVCacheManager};

/// Cached encoder output to avoid re-encoding the same input.
///
/// Encoder outputs are immutable once computed, so they can be shared
/// across decoder steps without recomputation. For workloads where many
/// requests share the same source text (e.g., document QA), the engine
/// can optionally cache these across requests.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Hidden states from the encoder `[batch, src_len, hidden_size]`.
    pub hidden_states: Tensor,
    /// Source sequence length, stored separately for cross-attention masking
    /// so callers don't need to inspect tensor dimensions.
    pub src_len: usize,
}

impl EncoderOutput {
    /// Create a new encoder output from hidden states.
    ///
    /// The `src_len` is extracted from the tensor's second dimension.
    pub fn new(hidden_states: Tensor) -> Result<Self> {
        let src_len = hidden_states.dim(1)?;
        Ok(Self {
            hidden_states,
            src_len,
        })
    }

    /// Create a new encoder output with an explicit source length.
    ///
    /// Useful when the tensor has already been validated or when
    /// constructing from pre-computed values in tests.
    pub fn with_src_len(hidden_states: Tensor, src_len: usize) -> Self {
        Self {
            hidden_states,
            src_len,
        }
    }
}

/// Trait for encoder-decoder models (T5, BART, mBART, etc.).
///
/// Encoder-decoder models have two distinct phases:
/// 1. Encode: Process the source input once (encoder is bidirectional)
/// 2. Decode: Autoregressive generation attending to encoder output via cross-attention
///
/// The encoder output is computed once and cached for the entire decoding process.
/// The decoder uses self-attention (with KV cache) and cross-attention (to encoder output).
pub trait ModelForEncoderDecoder: Send + 'static {
    /// Run the encoder on the source input.
    ///
    /// This should be called once per request. The result is cached and
    /// reused for all decoder steps.
    ///
    /// # Arguments
    /// * `input_ids` - Source token IDs `[batch, src_len]`
    /// * `attention_mask` - Optional source attention mask `[batch, src_len]`
    ///   where 1 = real token and 0 = padding
    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<EncoderOutput>;

    /// Run one step of the decoder with cross-attention to encoder output.
    ///
    /// Returns logits over the vocabulary `[batch, tgt_len, vocab_size]`.
    ///
    /// # Arguments
    /// * `decoder_input_ids` - Decoder token IDs `[batch, tgt_len]`
    /// * `encoder_output` - Cached encoder hidden states
    /// * `seqlen_offset` - Position offset for decoder (number of previously generated tokens)
    /// * `kv_cache_mgr` - KV cache manager for decoder self-attention cache
    /// * `block_table` - Block table for paged attention
    /// * `slot_mapping` - Slot mapping for cache writes
    fn decode(
        &self,
        decoder_input_ids: &Tensor,
        encoder_output: &EncoderOutput,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor>;

    /// Convenience method: encode + single decode step.
    ///
    /// Encodes the source and immediately runs one decode step.
    /// Returns both the logits and the encoder output (for caching across
    /// subsequent decode steps).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        encoder_input_ids: &Tensor,
        decoder_input_ids: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, EncoderOutput)> {
        let encoder_output = self.encode(encoder_input_ids, encoder_attention_mask)?;
        let logits = self.decode(
            decoder_input_ids,
            &encoder_output,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )?;
        Ok((logits, encoder_output))
    }

    /// Get the decoder start token ID.
    ///
    /// Most encoder-decoder models use a special token to begin decoding
    /// (e.g., `</s>` for T5, `<s>` for BART). This token is the first
    /// input to the decoder.
    fn decoder_start_token_id(&self) -> u32;

    /// Get the device this model is on.
    fn device(&self) -> &Device;

    /// Whether this model supports encoder-side caching.
    ///
    /// If true, the engine can cache encoder outputs for prefix matching,
    /// which is beneficial when many requests share the same source text
    /// (e.g., document QA, translation of the same source to multiple targets).
    fn supports_encoder_cache(&self) -> bool {
        false
    }

    /// Maximum source sequence length supported by the encoder.
    fn max_source_len(&self) -> usize {
        512
    }

    /// Maximum target sequence length supported by the decoder.
    fn max_target_len(&self) -> usize {
        512
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    // ─── EncoderOutput Tests ─────────────────────────────────────────────────

    #[test]
    fn encoder_output_new_extracts_src_len() {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((1, 10, 64), DType::F32, &device)
            .expect("tensor creation should not fail");

        let output = EncoderOutput::new(hidden).expect("EncoderOutput::new should succeed");
        assert_eq!(output.src_len, 10);
    }

    #[test]
    fn encoder_output_with_explicit_src_len() {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((2, 20, 128), DType::F32, &device)
            .expect("tensor creation should not fail");

        let output = EncoderOutput::with_src_len(hidden, 20);
        assert_eq!(output.src_len, 20);
        assert_eq!(output.hidden_states.dims(), &[2, 20, 128]);
    }

    #[test]
    fn encoder_output_clone_preserves_data() {
        let device = Device::Cpu;
        let hidden =
            Tensor::ones((1, 5, 32), DType::F32, &device).expect("tensor creation should not fail");

        let output = EncoderOutput::with_src_len(hidden, 5);
        let cloned = output.clone();

        assert_eq!(cloned.src_len, output.src_len);
        assert_eq!(cloned.hidden_states.dims(), output.hidden_states.dims());

        // Verify the data is the same after cloning
        let orig_data: Vec<f32> = output
            .hidden_states
            .flatten_all()
            .expect("flatten should work")
            .to_vec1()
            .expect("to_vec1 should work");
        let cloned_data: Vec<f32> = cloned
            .hidden_states
            .flatten_all()
            .expect("flatten should work")
            .to_vec1()
            .expect("to_vec1 should work");
        assert_eq!(orig_data, cloned_data);
    }

    #[test]
    fn encoder_output_batched() {
        let device = Device::Cpu;
        let batch_size = 4;
        let src_len = 15;
        let hidden_size = 256;
        let hidden = Tensor::zeros((batch_size, src_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should not fail");

        let output = EncoderOutput::new(hidden).expect("EncoderOutput::new should succeed");
        assert_eq!(output.src_len, src_len);
        assert_eq!(
            output.hidden_states.dims(),
            &[batch_size, src_len, hidden_size]
        );
    }

    // ─── Trait default implementations ───────────────────────────────────────

    /// Mock encoder-decoder model for testing trait defaults.
    struct MockEncoderDecoder {
        device: Device,
        vocab_size: usize,
        decoder_start_token: u32,
    }

    impl MockEncoderDecoder {
        fn new(vocab_size: usize, decoder_start_token: u32) -> Self {
            Self {
                device: Device::Cpu,
                vocab_size,
                decoder_start_token,
            }
        }
    }

    impl ModelForEncoderDecoder for MockEncoderDecoder {
        fn encode(
            &self,
            input_ids: &Tensor,
            _attention_mask: Option<&Tensor>,
        ) -> Result<EncoderOutput> {
            let (_batch, src_len) = input_ids.dims2()?;
            let hidden = Tensor::ones((1, src_len, 64), DType::F32, &self.device)?;
            Ok(EncoderOutput::with_src_len(hidden, src_len))
        }

        fn decode(
            &self,
            decoder_input_ids: &Tensor,
            _encoder_output: &EncoderOutput,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> Result<Tensor> {
            let (batch, tgt_len) = decoder_input_ids.dims2()?;
            Tensor::zeros((batch, tgt_len, self.vocab_size), DType::F32, &self.device)
        }

        fn decoder_start_token_id(&self) -> u32 {
            self.decoder_start_token
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    #[test]
    fn trait_default_supports_encoder_cache_is_false() {
        let model = MockEncoderDecoder::new(1000, 0);
        assert!(!model.supports_encoder_cache());
    }

    #[test]
    fn trait_default_max_source_len() {
        let model = MockEncoderDecoder::new(1000, 0);
        assert_eq!(model.max_source_len(), 512);
    }

    #[test]
    fn trait_default_max_target_len() {
        let model = MockEncoderDecoder::new(1000, 0);
        assert_eq!(model.max_target_len(), 512);
    }

    #[test]
    fn trait_decoder_start_token_id() {
        let model = MockEncoderDecoder::new(1000, 42);
        assert_eq!(model.decoder_start_token_id(), 42);
    }

    #[test]
    fn trait_device_returns_correct_device() {
        let model = MockEncoderDecoder::new(1000, 0);
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn trait_forward_default_calls_encode_then_decode() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;

        let model = MockEncoderDecoder::new(1000, 0);
        let device = Device::Cpu;

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr =
            KVCacheManager::new(&cache_config).expect("cache manager creation should not fail");
        let mut block_table = BlockTable::new(16);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocation should not fail");
        let slot_mapping = block_table.slot_mapping(0, 4);

        let encoder_ids =
            Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).expect("tensor creation should not fail");
        let decoder_ids = Tensor::new(&[[0u32]], &device).expect("tensor creation should not fail");

        let (logits, encoder_output) = model
            .forward(
                &encoder_ids,
                &decoder_ids,
                None,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward should succeed");

        // Encoder output should have source length 5
        assert_eq!(encoder_output.src_len, 5);
        assert_eq!(encoder_output.hidden_states.dims(), &[1, 5, 64]);

        // Logits should be [1, 1, vocab_size]
        assert_eq!(logits.dims(), &[1, 1, 1000]);
    }

    #[test]
    fn trait_encode_preserves_source_length() {
        let model = MockEncoderDecoder::new(1000, 0);
        let device = Device::Cpu;

        let input_ids = Tensor::new(&[[10u32, 20, 30, 40, 50, 60, 70]], &device)
            .expect("tensor creation should not fail");

        let output = model
            .encode(&input_ids, None)
            .expect("encode should succeed");

        assert_eq!(output.src_len, 7);
    }

    #[test]
    fn trait_encode_with_attention_mask() {
        let model = MockEncoderDecoder::new(1000, 0);
        let device = Device::Cpu;

        let input_ids =
            Tensor::new(&[[1u32, 2, 3, 0, 0]], &device).expect("tensor creation should not fail");
        let mask = Tensor::new(&[[1.0f32, 1.0, 1.0, 0.0, 0.0]], &device)
            .expect("tensor creation should not fail");

        let output = model
            .encode(&input_ids, Some(&mask))
            .expect("encode with mask should succeed");

        // Our mock ignores the mask but it should still pass through without error
        assert_eq!(output.src_len, 5);
    }

    // ─── Send + 'static bound verification ───────────────────────────────────

    #[test]
    fn trait_is_send() {
        // Compile-time check that the trait requires Send + 'static
        fn assert_send<T: Send + 'static>() {}
        assert_send::<MockEncoderDecoder>();
    }
}
