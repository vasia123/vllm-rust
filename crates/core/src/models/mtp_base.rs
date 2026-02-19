//! Multi-Token Prediction (MTP) draft model trait.
//!
//! MTP is a speculative decoding paradigm where the target model's hidden
//! states are fused with the draft token embeddings via a lightweight
//! transformer to predict the next token.
//!
//! # Key difference from Eagle
//!
//! Eagle chains its own hidden states between draft steps:
//!   `hidden_k = eagle_forward(draft_token_k, hidden_{k-1})`
//!
//! MTP uses FIXED target hidden states for ALL draft steps:
//!   `hidden_k = mtp_forward(draft_token_k, target_hs)` — same target_hs every step
//!
//! This makes MTP slightly simpler to implement but requires the target model
//! to provide its last hidden state at each decode round.
//!
//! # Architecture
//!
//! Each MTP layer processes:
//!   1. `inputs_embeds = enorm(embed(input_ids))` — embedding with BOS masking
//!   2. `previous_hidden_states = hnorm(target_hs)` — normalized target context
//!   3. `hidden = mtp_block(eh_proj(cat([inputs_embeds, previous_hidden_states])))`
//!   4. `logits = lm_head(norm(hidden))`
//!
//! Multiple MTP layers cycle: `layer_idx = spec_step_idx % num_mtp_layers`.
//! For DeepSeek-V3 `num_nextn_predict_layers = 1`, so no cycling occurs.

use candle_core::{Device, Tensor};

use crate::kv_cache::{BlockTable, KVCacheManager};

/// Trait for Multi-Token Prediction (MTP) draft models.
///
/// Implementations wrap a set of lightweight transformer layers that predict
/// draft tokens using target model hidden states as context.
///
/// Used by [`MtpProposer`](crate::engine::spec_decode::MtpProposer) for
/// speculative decoding.
pub trait MtpDraftModel: Send {
    /// Run one MTP forward pass.
    ///
    /// # Arguments
    /// - `input_ids`: `[1, seq_len]` — draft token(s) to process. During
    ///   prefill `seq_len = prompt_len + 1`; during decode `seq_len = 1`.
    /// - `previous_hidden_states`: `[1, seq_len, hidden_size]` — target model
    ///   hidden states. FIXED for all draft steps (not chained).
    /// - `seqlen_offset`: Position of the first token in `input_ids` within
    ///   the full sequence. Used for RoPE and KV cache addressing.
    /// - `spec_step_idx`: Draft step index (0-based). Determines which MTP
    ///   layer to use via `spec_step_idx % num_mtp_layers()`.
    /// - `kv_cache_mgr`: KV cache for the MTP transformer layers.
    /// - `block_table`: Block allocation for this request's MTP KV cache.
    /// - `slot_mapping`: Physical KV cache slots for `seq_len` tokens.
    ///
    /// Returns `hidden_states: [1, seq_len, hidden_size]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        seqlen_offset: usize,
        spec_step_idx: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor>;

    /// Compute logits from hidden states.
    ///
    /// Applies the per-layer shared_head normalization and LM head projection.
    /// `spec_step_idx` selects which layer's head to use.
    fn compute_logits(
        &self,
        hidden_states: &Tensor,
        spec_step_idx: usize,
    ) -> candle_core::Result<Tensor>;

    /// Number of MTP layers. Determines cycling: `layer = step % num_mtp_layers()`.
    fn num_mtp_layers(&self) -> usize;

    /// Device the model is loaded on.
    fn device(&self) -> &Device;
}
