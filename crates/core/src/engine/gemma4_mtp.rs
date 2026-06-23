//! Gemma 4 MTP speculative decoding execution strategy.
//!
//! Specialised speculative-decode strategy for the Gemma 4 backbone +
//! `gemma4_assistant` drafter. Unlike the generic [`SpeculativeExecution`]
//! (which delegates to a [`DraftProposer`] that owns its own KV cache), the
//! Gemma 4 assistant has NO KV cache of its own: it reads the TARGET's KV and
//! fuses the target's POST-final-norm hidden state. That needs direct access to
//! the concrete target + its KV cache, so this strategy owns both models.
//!
//! Per decode round (one target forward — the throughput win depends on it):
//! 1. **Draft** `k` tokens with the assistant. Each fuses the FIXED backbone
//!    hidden (captured one step behind from the last verify — the EAGLE/MTP
//!    convention) with the cycling draft-token embedding, reading the target's
//!    last sliding / last full layer KV for attention, all at the same position.
//! 2. **Verify** `[anchor, draft_1..k]` with ONE target forward; accept/reject
//!    by rejection sampling; capture the new fixed backbone hidden from the last
//!    accepted position for the next round.
//!
//! [`SpeculativeExecution`]: super::speculative::SpeculativeExecution
//! [`DraftProposer`]: super::spec_decode::DraftProposer

use std::collections::HashMap;

use candle_core::Tensor;
use rand::Rng;
use tracing::warn;

use crate::kv_cache::{BlockId, KVCacheManager};
use crate::models::{Gemma4Assistant, QuantizedGemma4ForCausalLM};
use crate::request::{RequestId, RequestStatus};
use crate::scheduler::SchedulerOutput;
use crate::tokenizer::TokenizerWrapper;

use super::context::OwnedExecutionState;
use super::helpers::{
    execute_decode, finish_request_with_error_deferred, handle_prefill_error, sample_token,
};
use super::model_forward::ModelForward;
use super::speculative::{compute_target_probs, sample_recovered_token, sample_speculative};
use super::strategy::ExecutionStrategy;
use super::types::{EngineError, SpecDecodingStats};

/// Speculative decoding with a Gemma 4 `gemma4_assistant` drafter.
pub struct Gemma4MtpExecution {
    target: Box<dyn ModelForward>,
    assistant: Gemma4Assistant,
    /// Backbone layer whose KV the assistant's sliding layers read.
    last_swa_layer: usize,
    /// Backbone layer whose KV the assistant's full layer reads.
    last_full_layer: usize,
    num_speculative_tokens: usize,
    stats: SpecDecodingStats,
    /// Per-request fixed backbone hidden `[1, 1, hidden]` (POST-final-norm),
    /// captured one step behind, fused by the assistant for the next round.
    backbone_hidden: HashMap<RequestId, Tensor>,
}

impl Gemma4MtpExecution {
    pub fn new(
        target: Box<dyn ModelForward>,
        assistant: Gemma4Assistant,
        num_speculative_tokens: usize,
    ) -> Self {
        let (last_swa_layer, last_full_layer) = target
            .as_gemma4_mtp()
            .expect("Gemma4MtpExecution requires a Gemma 4 target")
            .mtp_shared_kv_layers();
        Self {
            target,
            assistant,
            last_swa_layer,
            last_full_layer,
            num_speculative_tokens,
            stats: SpecDecodingStats::new(num_speculative_tokens),
            backbone_hidden: HashMap::new(),
        }
    }

    fn tgt(&self) -> &QuantizedGemma4ForCausalLM {
        self.target
            .as_gemma4_mtp()
            .expect("Gemma4MtpExecution requires a Gemma 4 target")
    }

    /// Read the target's per-layer KV for a request's context, narrowed to the
    /// assistant's `(n_head_kv, head_dim)` geometry → `[1, n_head_kv, ctx, head_dim]`.
    fn read_target_kv(
        &self,
        kv: &KVCacheManager,
        layer: usize,
        block_ids: &[BlockId],
        ctx: usize,
        n_head_kv: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor), EngineError> {
        let (k, v) = kv
            .engine(layer)
            .read(block_ids, ctx)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let narrow = |t: &Tensor| -> candle_core::Result<Tensor> {
            let t = if t.dim(1)? != n_head_kv {
                t.narrow(1, 0, n_head_kv)?
            } else {
                t.clone()
            };
            let t = if t.dim(3)? != head_dim {
                t.narrow(3, 0, head_dim)?
            } else {
                t
            };
            t.contiguous()
        };
        let k = narrow(&k).map_err(|e| EngineError::Model(e.to_string()))?;
        let v = narrow(&v).map_err(|e| EngineError::Model(e.to_string()))?;
        Ok((k, v))
    }

    /// Prefill that also captures the prompt's last POST-final-norm hidden as
    /// the initial fixed backbone hidden. Mirrors `helpers::execute_prefill`'s
    /// chunk/alloc bookkeeping; uses `forward_verify` on the final chunk.
    fn mtp_prefill(
        &mut self,
        req_id: RequestId,
        chunk_size: usize,
        kv: &mut KVCacheManager,
        state: &mut OwnedExecutionState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<(), EngineError> {
        let req = state
            .requests
            .get_mut(&req_id)
            .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;

        let total_len = req.state.total_len();
        let offset = req.state.num_computed_tokens;
        let chunk_end = (offset + chunk_size).min(total_len);
        let actual_chunk = chunk_end - offset;
        let is_final_chunk = chunk_end == total_len;

        kv.allocate_for_request(&mut req.state.block_table, actual_chunk)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = req.state.block_table.slot_mapping(offset, actual_chunk);
        let chunk_tokens = req.state.token_window(offset, chunk_end);
        let input = Tensor::from_vec(chunk_tokens, (1, actual_chunk), self.target.device())
            .map_err(|e| EngineError::Model(e.to_string()))?;

        if is_final_chunk {
            let (logits, hidden) = self
                .tgt()
                .forward_verify(&input, offset, kv, &req.state.block_table, &slot_mapping)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            req.state.block_table.advance(actual_chunk);
            req.state.num_computed_tokens = chunk_end;
            req.state.seqlen_offset = chunk_end;
            req.state.status = RequestStatus::Decoding;

            let last = actual_chunk - 1;
            let h = hidden
                .narrow(1, last, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            self.backbone_hidden.insert(req_id, h);

            let last_logits = logits
                .narrow(1, last, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let next = sample_token(&last_logits, &mut req.state, tokenizer)?;
            req.state.generated_token_ids.push(next);
        } else {
            self.target
                .forward(&input, offset, kv, &req.state.block_table, &slot_mapping)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            req.state.block_table.advance(actual_chunk);
            req.state.num_computed_tokens = chunk_end;
            req.state.seqlen_offset = chunk_end;
            req.state.status = RequestStatus::Prefilling;
        }
        Ok(())
    }

    /// One speculative decode round for a request (single target forward).
    fn mtp_decode(
        &mut self,
        req_id: RequestId,
        kv: &mut KVCacheManager,
        state: &mut OwnedExecutionState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<(), EngineError> {
        let backbone_hidden = match self.backbone_hidden.get(&req_id) {
            Some(h) => h.clone(),
            None => {
                return execute_decode(req_id, &self.target, kv, &mut state.requests, tokenizer)
            }
        };

        let (k, anchor, seqlen_offset, block_ids) = {
            let req = state
                .requests
                .get(&req_id)
                .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
            let remaining = req
                .state
                .max_new_tokens
                .saturating_sub(req.state.num_generated());
            let k = self.num_speculative_tokens.min(remaining.saturating_sub(1));
            let anchor = *req.state.generated_token_ids.last().ok_or_else(|| {
                EngineError::Model("no generated tokens for speculative decode".into())
            })?;
            (
                k,
                anchor,
                req.state.seqlen_offset,
                req.state.block_table.block_ids().to_vec(),
            )
        };

        // No draft budget, or the draft position would exceed the assistant's
        // RoPE table → plain decode this step.
        if k == 0 || seqlen_offset >= self.assistant.max_pos() {
            return execute_decode(req_id, &self.target, kv, &mut state.requests, tokenizer);
        }

        let device = self.target.device().clone();
        let cfg = self.assistant.config().clone();

        // ── Draft phase: read the target KV [0, seqlen_offset) once. The anchor
        //    (position seqlen_offset) is not yet cached; its content reaches the
        //    drafter through the fused embedding. ───────────────────────────
        let (k_swa, v_swa) = self.read_target_kv(
            kv,
            self.last_swa_layer,
            &block_ids,
            seqlen_offset,
            cfg.swa_kv_heads(),
            cfg.head_dim_swa,
        )?;
        let (k_full, v_full) = self.read_target_kv(
            kv,
            self.last_full_layer,
            &block_ids,
            seqlen_offset,
            cfg.full_kv_heads(),
            cfg.head_dim_full,
        )?;

        let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
        {
            let req = state
                .requests
                .get_mut(&req_id)
                .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
            for step in 0..k {
                let token = if step == 0 {
                    anchor
                } else {
                    draft_tokens[step - 1]
                };
                let ids = Tensor::new(&[[token]], &device)
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                let embed = self
                    .tgt()
                    .embed_for_mtp(&ids)
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                let (logits, _h_next) = self
                    .assistant
                    .forward(
                        &embed,
                        &backbone_hidden,
                        seqlen_offset,
                        (&k_swa, &v_swa),
                        (&k_full, &v_full),
                    )
                    .map_err(|e| EngineError::Model(e.to_string()))?;
                let t = sample_speculative(&logits, &mut req.state, tokenizer)?;
                draft_tokens.push(t);
            }
        }
        let actual_k = draft_tokens.len();

        // ── Verify phase: one target forward over [anchor, draft_1..k]. ────
        let req = state
            .requests
            .get_mut(&req_id)
            .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
        let k_plus_1 = actual_k + 1;
        kv.allocate_for_request(&mut req.state.block_table, k_plus_1)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot = req
            .state
            .block_table
            .slot_mapping(req.state.seqlen_offset, k_plus_1);
        let mut verify_input = vec![anchor];
        verify_input.extend_from_slice(&draft_tokens[..actual_k]);
        let input = Tensor::from_vec(verify_input, (1, k_plus_1), &device)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let (logits, hidden) = self
            .tgt()
            .forward_verify(
                &input,
                req.state.seqlen_offset,
                kv,
                &req.state.block_table,
                &slot,
            )
            .map_err(|e| EngineError::Model(e.to_string()))?;
        req.state.block_table.advance(k_plus_1);

        // ── Acceptance (rejection sampling). logits[i] is the distribution
        //    after the i-th verify token, i.e. it predicts draft i. ──────────
        let is_greedy = req.state.sampling_params.is_greedy();
        let mut accepted = 0usize;
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let pos_logits = logits
                .narrow(1, i, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            if is_greedy {
                let target_token = sample_speculative(&pos_logits, &mut req.state, tokenizer)?;
                if target_token == draft_token {
                    accepted += 1;
                    req.state.generated_token_ids.push(draft_token);
                } else {
                    req.state.generated_token_ids.push(target_token);
                    break;
                }
            } else {
                let target_probs = compute_target_probs(&pos_logits, &mut req.state, tokenizer)?;
                let p_draft = target_probs
                    .get(draft_token as usize)
                    .copied()
                    .unwrap_or(0.0);
                let u: f64 = req.state.sampler_state.rng_mut().gen();
                if (p_draft as f64) >= u {
                    accepted += 1;
                    req.state.generated_token_ids.push(draft_token);
                } else {
                    let recovered =
                        sample_recovered_token(&target_probs, draft_token, &mut req.state);
                    req.state.generated_token_ids.push(recovered);
                    break;
                }
            }
        }
        if accepted == actual_k {
            let bonus_logits = logits
                .narrow(1, actual_k, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let bonus = sample_speculative(&bonus_logits, &mut req.state, tokenizer)?;
            req.state.generated_token_ids.push(bonus);
        }

        // Next round's fixed hidden = the last PROCESSED accepted position
        // (index `accepted` of the verify sequence). The bonus/recovered token
        // becomes the next anchor (one step behind).
        let h_next = hidden
            .narrow(1, accepted, 1)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        self.backbone_hidden.insert(req_id, h_next);

        // Roll the target KV back to the accepted length.
        let original_offset = req.state.seqlen_offset;
        let target_total = original_offset + accepted + 1;
        let freed = req.state.block_table.trim_to(target_total);
        if !freed.is_empty() {
            kv.free_blocks(&freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        req.state.seqlen_offset = target_total;

        self.stats.observe_draft(actual_k, accepted);
        Ok(())
    }
}

impl ExecutionStrategy for Gemma4MtpExecution {
    fn execute_prefills(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        tokenizer: &TokenizerWrapper,
    ) {
        for schedule in &output.prefill_requests {
            let req_id = schedule.request_id;
            if let Err(e) =
                self.mtp_prefill(req_id, schedule.chunk_size, kv_cache_mgr, state, tokenizer)
            {
                match handle_prefill_error(req_id, e, &mut state.requests, kv_cache_mgr) {
                    Ok(preempted) => state.preempted_ids.push(preempted),
                    Err(Some(id)) => state.errored_ids.push(id),
                    Err(None) => {}
                }
            }
        }
    }

    fn execute_decodes(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        _multi_step_count: usize,
        tokenizer: &TokenizerWrapper,
    ) {
        let has_active_prefill = !output.prefill_requests.is_empty();
        for &req_id in &output.decode_requests {
            let result = if has_active_prefill {
                execute_decode(
                    req_id,
                    &self.target,
                    kv_cache_mgr,
                    &mut state.requests,
                    tokenizer,
                )
            } else {
                self.mtp_decode(req_id, kv_cache_mgr, state, tokenizer)
            };
            if let Err(e) = result {
                if let Some(id) =
                    finish_request_with_error_deferred(req_id, e, &mut state.requests, kv_cache_mgr)
                {
                    state.errored_ids.push(id);
                }
            }
        }
    }

    fn handle_preemptions(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
    ) {
        for &req_id in &output.preempted_requests {
            let Some(req) = state.requests.get_mut(&req_id) else {
                continue;
            };
            if let Err(e) = kv_cache_mgr.free_request(&mut req.state.block_table) {
                warn!(error = %e, request_id = req_id, "Gemma4 MTP: failed to free target cache on preemption");
            }
            self.backbone_hidden.remove(&req_id);
            req.state.num_computed_tokens = 0;
            req.state.seqlen_offset = 0;
            req.state.status = RequestStatus::Waiting;
        }
    }

    fn on_request_completed(&mut self, req_id: RequestId, _state: &mut OwnedExecutionState) {
        self.backbone_hidden.remove(&req_id);
    }

    fn spec_decode_stats(&self) -> Option<SpecDecodingStats> {
        Some(self.stats.clone())
    }
}
