//! Eagle-3 DraftProposer implementation.
//!
//! Wraps any [`Eagle3DraftModel`] with full lifecycle management. Unlike
//! [`DraftModelDraftProposer`], this proposer requires target model hidden
//! states via [`set_target_hidden_states`] before each proposal round.
//!
//! Draft token generation flow:
//! 1. Target model produces hidden states → engine calls `set_target_hidden_states`
//! 2. `propose_for_request` runs K sequential Eagle3 forward passes
//! 3. Each step chains the **prenorm** hidden states to the next step
//! 4. `compute_logits` uses the **post-norm** output for sampling
//!
//! On first call, the proposer runs a full prefill through Eagle3 to populate
//! its KV cache. Subsequent calls only do single-token decode steps.
//!
//! Reference: `reference/vllm/vllm/v1/spec_decode/eagle.py`

use std::collections::HashMap;

use candle_core::Tensor;
use tracing::warn;

use crate::engine::types::EngineError;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::models::Eagle3DraftModel;
use crate::request::{RequestId, SequenceState};
use crate::tokenizer::TokenizerWrapper;

use super::sample_speculative;
use super::DraftProposer;

/// Per-request Eagle3 draft state.
struct Eagle3RequestState {
    block_table: BlockTable,
    seqlen_offset: usize,
    prompt_tokens: Vec<u32>,
    /// Target model hidden states for the current proposal round.
    target_hidden_states: Option<Tensor>,
    /// Whether the Eagle3 KV cache needs an initial prefill.
    needs_prefill: bool,
}

/// [`DraftProposer`] backed by an Eagle-3 speculative decoding model.
///
/// Works with any model implementing [`Eagle3DraftModel`], including:
/// - [`Eagle3LlamaForCausalLM`](crate::models::Eagle3LlamaForCausalLM): Llama-based (layer-0 concatenation)
/// - [`Eagle3MistralLarge3ForCausalLM`](crate::models::Eagle3MistralLarge3ForCausalLM): DeepSeek-based (fc projection)
///
/// The engine must call [`set_target_hidden_states`] before each
/// `propose_for_request` call to provide the target model's context.
pub struct Eagle3DraftProposer {
    model: Box<dyn Eagle3DraftModel>,
    kv_cache: KVCacheManager,
    num_speculative_tokens: usize,
    requests: HashMap<RequestId, Eagle3RequestState>,
}

impl Eagle3DraftProposer {
    pub fn new(
        model: Box<dyn Eagle3DraftModel>,
        kv_cache: KVCacheManager,
        num_speculative_tokens: usize,
    ) -> Self {
        Self {
            model,
            kv_cache,
            num_speculative_tokens,
            requests: HashMap::new(),
        }
    }
}

impl DraftProposer for Eagle3DraftProposer {
    fn init_request(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
    ) -> Result<(), EngineError> {
        let block_table = BlockTable::new(self.kv_cache.block_size());

        self.requests.insert(
            request_id,
            Eagle3RequestState {
                block_table,
                seqlen_offset: 0,
                prompt_tokens: prompt_tokens.to_vec(),
                target_hidden_states: None,
                needs_prefill: true,
            },
        );

        Ok(())
    }

    fn set_target_hidden_states(
        &mut self,
        request_id: RequestId,
        hidden_states: Tensor,
    ) -> Result<(), EngineError> {
        let req_state = self
            .requests
            .get_mut(&request_id)
            .ok_or_else(|| {
                EngineError::Model(format!("Eagle3 state for {request_id} not found"))
            })?;

        // Project auxiliary hidden states (3×hs → hs) if needed
        let hidden_states = if self.model.use_aux_hidden_state() {
            self.model
                .combine_hidden_states(&hidden_states)
                .map_err(|e| EngineError::Model(e.to_string()))?
        } else {
            hidden_states
        };

        req_state.target_hidden_states = Some(hidden_states);
        Ok(())
    }

    fn propose_for_request(
        &mut self,
        request_id: RequestId,
        last_token: u32,
        state: &mut SequenceState,
        tokenizer: &TokenizerWrapper,
    ) -> Result<Vec<u32>, EngineError> {
        let k = self.num_speculative_tokens;
        let req_state = self
            .requests
            .get_mut(&request_id)
            .ok_or_else(|| {
                EngineError::Model(format!("Eagle3 state for {request_id} not found"))
            })?;

        let target_hs = req_state
            .target_hidden_states
            .take()
            .ok_or_else(|| {
                EngineError::Model("Eagle3: target hidden states not set".into())
            })?;

        // Prefill Eagle3 KV cache on first call
        let mut current_hs = if req_state.needs_prefill {
            let prompt_len = req_state.prompt_tokens.len();

            self.kv_cache
                .allocate_for_request(&mut req_state.block_table, prompt_len)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            let slot_mapping = req_state.block_table.slot_mapping(0, prompt_len);

            let input_ids = Tensor::from_vec(
                req_state.prompt_tokens.clone(),
                (1, prompt_len),
                self.model.device(),
            )
            .map_err(|e| EngineError::Model(e.to_string()))?;

            let (_hs, hs_prenorm) = self
                .model
                .forward(
                    &input_ids,
                    &target_hs,
                    0,
                    &mut self.kv_cache,
                    &req_state.block_table,
                    &slot_mapping,
                )
                .map_err(|e| EngineError::Model(e.to_string()))?;

            req_state.block_table.advance(prompt_len);
            req_state.seqlen_offset = prompt_len;
            req_state.needs_prefill = false;

            // Use prenorm from last position for chaining to decode steps
            let seq_dim = hs_prenorm.dims()[1];
            hs_prenorm
                .narrow(1, seq_dim - 1, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?
        } else {
            target_hs
        };

        // K single-token decode steps
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_input_token = last_token;

        for _ in 0..k {
            self.kv_cache
                .allocate_for_request(&mut req_state.block_table, 1)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
            let slot_mapping = req_state
                .block_table
                .slot_mapping(req_state.seqlen_offset, 1);

            let input = Tensor::new(&[[draft_input_token]], self.model.device())
                .map_err(|e| EngineError::Model(e.to_string()))?;

            let (hs, hs_prenorm) = self
                .model
                .forward(
                    &input,
                    &current_hs,
                    req_state.seqlen_offset,
                    &mut self.kv_cache,
                    &req_state.block_table,
                    &slot_mapping,
                )
                .map_err(|e| EngineError::Model(e.to_string()))?;

            req_state.block_table.advance(1);
            req_state.seqlen_offset += 1;

            let logits = self
                .model
                .compute_logits(&hs)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let seq_dim = logits.dims()[1];
            let logits = logits
                .narrow(1, seq_dim - 1, 1)
                .map_err(|e| EngineError::Model(e.to_string()))?;
            let token = sample_speculative(&logits, state, tokenizer)?;
            draft_tokens.push(token);
            draft_input_token = token;

            // Chain prenorm hidden states for next step
            current_hs = hs_prenorm;
        }

        Ok(draft_tokens)
    }

    fn on_tokens_verified(
        &mut self,
        request_id: RequestId,
        num_accepted: usize,
        original_offset: usize,
    ) -> Result<(), EngineError> {
        let req_state = self
            .requests
            .get_mut(&request_id)
            .ok_or_else(|| {
                EngineError::Model(format!("Eagle3 state for {request_id} not found"))
            })?;

        let draft_total = original_offset + num_accepted;
        let freed = req_state.block_table.trim_to(draft_total);
        if !freed.is_empty() {
            self.kv_cache
                .free_blocks(&freed)
                .map_err(|e| EngineError::Cache(e.to_string()))?;
        }
        req_state.seqlen_offset = draft_total;

        Ok(())
    }

    fn finish_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(mut rs) = self.requests.remove(&request_id) {
            let freed = rs.block_table.release();
            if !freed.is_empty() {
                if let Err(e) = self.kv_cache.free_blocks(&freed) {
                    warn!(
                        error = %e,
                        request_id = request_id,
                        "Failed to free Eagle3 cache blocks on request completion"
                    );
                }
            }
        }
        Ok(())
    }

    fn preempt_request(&mut self, request_id: RequestId) -> Result<(), EngineError> {
        if let Some(mut rs) = self.requests.remove(&request_id) {
            let freed = rs.block_table.release();
            if !freed.is_empty() {
                if let Err(e) = self.kv_cache.free_blocks(&freed) {
                    warn!(
                        error = %e,
                        request_id = request_id,
                        "Failed to free Eagle3 cache blocks during preemption"
                    );
                }
            }
        }
        Ok(())
    }

    fn num_speculative_tokens(&self) -> usize {
        self.num_speculative_tokens
    }

    fn name(&self) -> &str {
        "eagle3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use crate::models::Eagle3LlamaForCausalLM;
    use crate::request::SequenceState;
    use crate::tokenizer::TokenizerWrapper;
    use candle_core::{DType, Device};

    fn eagle3_model_config() -> ModelConfig {
        let mut cfg = ModelConfig {
            architectures: vec!["Eagle3LlamaForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        };
        // Disable aux hidden state for simpler testing
        let mut eagle = serde_json::Map::new();
        eagle.insert(
            "use_aux_hidden_state".to_string(),
            serde_json::Value::Bool(false),
        );
        cfg.extra.insert(
            "eagle_config".to_string(),
            serde_json::Value::Object(eagle),
        );
        cfg
    }

    fn eagle3_cache_config(cfg: &ModelConfig) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    fn make_eagle3_proposer(num_spec_tokens: usize) -> Eagle3DraftProposer {
        let cfg = eagle3_model_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");
        let cache_cfg = eagle3_cache_config(&cfg);
        let kv_cache = KVCacheManager::new(&cache_cfg).expect("cache creation");
        Eagle3DraftProposer::new(Box::new(model), kv_cache, num_spec_tokens)
    }

    fn make_state(prompt: &[u32]) -> SequenceState {
        SequenceState::new(0, prompt.to_vec(), 100, 255, 16, 0)
    }

    fn tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::for_testing(256)
    }

    fn hidden_states_for_prompt(prompt_len: usize, hidden_size: usize) -> Tensor {
        Tensor::zeros((1, prompt_len, hidden_size), DType::F32, &Device::Cpu)
            .expect("hidden states")
    }

    fn hidden_states_single(hidden_size: usize) -> Tensor {
        Tensor::zeros((1, 1, hidden_size), DType::F32, &Device::Cpu).expect("hidden states")
    }

    #[test]
    fn eagle3_proposer_init_and_propose() {
        let mut proposer = make_eagle3_proposer(3);
        let prompt = vec![1u32, 2, 3, 4, 5];
        proposer.init_request(0, &prompt).unwrap();

        // Set target hidden states for the full prompt
        let hs = hidden_states_for_prompt(prompt.len(), 64);
        proposer.set_target_hidden_states(0, hs).unwrap();

        let mut state = make_state(&prompt);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        assert_eq!(drafts.len(), 3);
    }

    #[test]
    fn eagle3_proposer_prefill_then_decode() {
        let mut proposer = make_eagle3_proposer(2);
        let prompt = vec![1u32, 2, 3];
        proposer.init_request(0, &prompt).unwrap();

        // First round: prefill + 2 draft tokens
        let hs = hidden_states_for_prompt(prompt.len(), 64);
        proposer.set_target_hidden_states(0, hs).unwrap();

        let mut state = make_state(&prompt);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let drafts1 = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        assert_eq!(drafts1.len(), 2);

        // Verify: 1 of 2 accepted
        let original_offset = prompt.len();
        proposer.on_tokens_verified(0, 1, original_offset).unwrap();

        // Second round: single-token hidden states (non-prefill path)
        let hs = hidden_states_single(64);
        proposer.set_target_hidden_states(0, hs).unwrap();

        let drafts2 = proposer
            .propose_for_request(0, drafts1[0], &mut state, &tok)
            .unwrap();
        assert_eq!(drafts2.len(), 2);
    }

    #[test]
    fn eagle3_proposer_on_verified_trims_cache() {
        let mut proposer = make_eagle3_proposer(3);
        let prompt = vec![1u32, 2, 3, 4, 5];
        proposer.init_request(0, &prompt).unwrap();

        let hs = hidden_states_for_prompt(prompt.len(), 64);
        proposer.set_target_hidden_states(0, hs).unwrap();

        let mut state = make_state(&prompt);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let original_offset = prompt.len();
        let _drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();

        // Only 1 of 3 accepted
        proposer.on_tokens_verified(0, 1, original_offset).unwrap();

        let rs = proposer.requests.get(&0).unwrap();
        assert_eq!(rs.seqlen_offset, original_offset + 1);
    }

    #[test]
    fn eagle3_proposer_finish_frees_blocks() {
        let mut proposer = make_eagle3_proposer(3);
        let initial_free = proposer.kv_cache.num_free_blocks();

        proposer.init_request(0, &[1, 2, 3, 4, 5]).unwrap();

        let hs = hidden_states_for_prompt(5, 64);
        proposer.set_target_hidden_states(0, hs).unwrap();

        let mut state = make_state(&[1, 2, 3, 4, 5]);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let _drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        assert!(proposer.kv_cache.num_free_blocks() < initial_free);

        proposer.finish_request(0).unwrap();
        assert_eq!(proposer.kv_cache.num_free_blocks(), initial_free);
        assert!(!proposer.requests.contains_key(&0));
    }

    #[test]
    fn eagle3_proposer_preempt_frees_blocks() {
        let mut proposer = make_eagle3_proposer(3);
        let initial_free = proposer.kv_cache.num_free_blocks();

        proposer.init_request(0, &[1, 2, 3, 4, 5]).unwrap();

        let hs = hidden_states_for_prompt(5, 64);
        proposer.set_target_hidden_states(0, hs).unwrap();

        let mut state = make_state(&[1, 2, 3, 4, 5]);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let _drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        proposer.preempt_request(0).unwrap();
        assert_eq!(proposer.kv_cache.num_free_blocks(), initial_free);
    }

    #[test]
    fn eagle3_proposer_finish_nonexistent_noop() {
        let mut proposer = make_eagle3_proposer(3);
        proposer.finish_request(999).unwrap();
    }

    #[test]
    fn eagle3_proposer_name() {
        let proposer = make_eagle3_proposer(3);
        assert_eq!(proposer.name(), "eagle3");
    }

    #[test]
    fn eagle3_proposer_num_speculative_tokens() {
        let proposer = make_eagle3_proposer(5);
        assert_eq!(proposer.num_speculative_tokens(), 5);
    }

    #[test]
    fn eagle3_proposer_propose_without_init_fails() {
        let mut proposer = make_eagle3_proposer(3);
        let mut state = make_state(&[1, 2, 3]);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let result = proposer.propose_for_request(999, 10, &mut state, &tok);
        assert!(result.is_err());
    }

    #[test]
    fn eagle3_proposer_propose_without_hidden_states_fails() {
        let mut proposer = make_eagle3_proposer(3);
        proposer.init_request(0, &[1, 2, 3]).unwrap();
        // Intentionally skip set_target_hidden_states

        let mut state = make_state(&[1, 2, 3]);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let result = proposer.propose_for_request(0, 10, &mut state, &tok);
        assert!(result.is_err());
    }

    #[test]
    fn eagle3_proposer_multiple_requests() {
        let mut proposer = make_eagle3_proposer(2);

        proposer.init_request(0, &[1, 2, 3]).unwrap();
        proposer.init_request(1, &[4, 5, 6, 7]).unwrap();

        let hs0 = hidden_states_for_prompt(3, 64);
        proposer.set_target_hidden_states(0, hs0).unwrap();
        let hs1 = hidden_states_for_prompt(4, 64);
        proposer.set_target_hidden_states(1, hs1).unwrap();

        let mut state0 = make_state(&[1, 2, 3]);
        state0.generated_token_ids.push(10);
        let mut state1 = make_state(&[4, 5, 6, 7]);
        state1.generated_token_ids.push(20);
        let tok = tokenizer();

        let d0 = proposer
            .propose_for_request(0, 10, &mut state0, &tok)
            .unwrap();
        let d1 = proposer
            .propose_for_request(1, 20, &mut state1, &tok)
            .unwrap();

        assert_eq!(d0.len(), 2);
        assert_eq!(d1.len(), 2);

        // Finish one, other still works
        proposer.finish_request(0).unwrap();
        assert!(!proposer.requests.contains_key(&0));
        assert!(proposer.requests.contains_key(&1));
        proposer.finish_request(1).unwrap();
    }

    #[test]
    fn eagle3_proposer_on_verified_without_init_fails() {
        let mut proposer = make_eagle3_proposer(3);
        let result = proposer.on_tokens_verified(999, 1, 5);
        assert!(result.is_err());
    }

    #[test]
    fn eagle3_proposer_set_hidden_states_without_init_fails() {
        let mut proposer = make_eagle3_proposer(3);
        let hs = hidden_states_single(64);
        let result = proposer.set_target_hidden_states(999, hs);
        assert!(result.is_err());
    }

    #[test]
    fn eagle3_proposer_with_aux_hidden_states() {
        let mut cfg = eagle3_model_config();
        // Enable aux hidden state
        let mut eagle = serde_json::Map::new();
        eagle.insert(
            "use_aux_hidden_state".to_string(),
            serde_json::Value::Bool(true),
        );
        cfg.extra.insert(
            "eagle_config".to_string(),
            serde_json::Value::Object(eagle),
        );

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");
        let cache_cfg = eagle3_cache_config(&cfg);
        let kv_cache = KVCacheManager::new(&cache_cfg).expect("cache creation");
        let mut proposer = Eagle3DraftProposer::new(Box::new(model), kv_cache, 2);

        proposer.init_request(0, &[1, 2, 3]).unwrap();

        // Aux hidden states: 3 * hidden_size = 192
        let aux_hs =
            Tensor::zeros((1, 3, 3 * 64), DType::F32, &Device::Cpu).expect("aux hidden states");
        proposer.set_target_hidden_states(0, aux_hs).unwrap();

        let mut state = make_state(&[1, 2, 3]);
        state.generated_token_ids.push(10);
        let tok = tokenizer();

        let drafts = proposer
            .propose_for_request(0, 10, &mut state, &tok)
            .unwrap();
        assert_eq!(drafts.len(), 2);
    }
}
