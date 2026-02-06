//! Draft-model based speculative token proposer.
//!
//! Wraps a smaller language model to generate greedy proposals. Unlike the
//! [`NGramProposer`], this requires VRAM for the draft model's weights and
//! KV cache, but can produce higher-quality proposals since it uses an
//! actual language model.
//!
//! NOTE: This proposer maintains its own KV cache, which is separate from
//! the target model's cache. The caller is responsible for managing the
//! lifecycle (prefill, rollback) when integrating with the engine loop.
//! Today the existing `SpeculativeExecution` strategy handles that directly;
//! this wrapper exists to unify the proposer interface for future refactoring.

use std::sync::Mutex;

use candle_core::Tensor;

use crate::engine::model_forward::ModelForward;
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::SpeculativeProposer;

/// Draft-model based speculative proposer.
///
/// Uses a small language model to greedily generate candidate tokens.
/// The draft model runs a full forward pass per proposed token, so the
/// cost scales linearly with `num_speculative_tokens`.
///
/// # Thread safety
///
/// The model and KV cache are wrapped in a `Mutex` because `SpeculativeProposer`
/// requires `Send + Sync`. The mutex is uncontended in practice since the
/// engine loop is single-threaded.
pub struct DraftModelProposer<M: ModelForward> {
    inner: Mutex<DraftModelInner<M>>,
    num_speculative_tokens: usize,
}

struct DraftModelInner<M: ModelForward> {
    model: M,
    kv_cache: KVCacheManager,
}

impl<M: ModelForward> DraftModelProposer<M> {
    /// Create a new draft model proposer.
    pub fn new(model: M, kv_cache: KVCacheManager, num_speculative_tokens: usize) -> Self {
        Self {
            inner: Mutex::new(DraftModelInner { model, kv_cache }),
            num_speculative_tokens,
        }
    }
}

impl<M: ModelForward> SpeculativeProposer for DraftModelProposer<M> {
    fn propose(&self, token_ids: &[u32], max_tokens: usize) -> Vec<u32> {
        let k = max_tokens.min(self.num_speculative_tokens);
        if k == 0 || token_ids.is_empty() {
            return Vec::new();
        }

        let Ok(mut inner) = self.inner.lock() else {
            return Vec::new();
        };

        // Destructure to get independent borrows of model and kv_cache.
        let DraftModelInner {
            ref model,
            ref mut kv_cache,
        } = *inner;

        let device = model.device().clone();

        // Allocate a fresh block table for this proposal.
        let mut block_table = BlockTable::new(kv_cache.block_size());

        // Prefill: run the draft model on all existing tokens to populate KV cache.
        let prompt_len = token_ids.len();
        if kv_cache
            .allocate_for_request(&mut block_table, prompt_len)
            .is_err()
        {
            return Vec::new();
        }
        let slot_mapping = block_table.slot_mapping(0, prompt_len);

        let input = match Tensor::from_vec(token_ids.to_vec(), (1, prompt_len), &device) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let logits = match model.forward(&input, 0, kv_cache, &block_table, &slot_mapping) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };

        block_table.advance(prompt_len);

        // Sample the first draft token from the prefill logits.
        let seq_dim = logits.dims()[1];
        let last_logits = match logits.narrow(1, seq_dim - 1, 1) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };
        let first_token = match greedy_sample_u32(&last_logits) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut draft_tokens = Vec::with_capacity(k);
        draft_tokens.push(first_token);
        let mut current_token = first_token;
        let mut seqlen_offset = prompt_len;

        // Autoregressive draft generation for remaining tokens.
        for _ in 1..k {
            if kv_cache.allocate_for_request(&mut block_table, 1).is_err() {
                break;
            }
            let slot_mapping = block_table.slot_mapping(seqlen_offset, 1);

            let input = match Tensor::new(&[[current_token]], &device) {
                Ok(t) => t,
                Err(_) => break,
            };

            let logits =
                match model.forward(&input, seqlen_offset, kv_cache, &block_table, &slot_mapping) {
                    Ok(l) => l,
                    Err(_) => break,
                };

            block_table.advance(1);
            seqlen_offset += 1;

            let seq_dim = logits.dims()[1];
            let last_logits = match logits.narrow(1, seq_dim - 1, 1) {
                Ok(l) => l,
                Err(_) => break,
            };
            match greedy_sample_u32(&last_logits) {
                Some(t) => {
                    draft_tokens.push(t);
                    current_token = t;
                }
                None => break,
            }
        }

        // Clean up: free all blocks allocated for this proposal.
        let _ = kv_cache.free_request(&mut block_table);

        draft_tokens
    }

    fn name(&self) -> &str {
        "draft_model"
    }
}

/// Greedy sample a single token from logits. Returns None on error.
fn greedy_sample_u32(logits: &Tensor) -> Option<u32> {
    let logits = logits.squeeze(0).ok()?.squeeze(0).ok()?;
    let token_id = logits.argmax(0).ok()?.to_scalar::<u32>().ok()?;
    Some(token_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use candle_core::{DType, Device};

    /// Deterministic mock model that always outputs a specific token.
    struct ConstantMockModel {
        output_token: u32,
        vocab_size: usize,
        device: Device,
    }

    impl ConstantMockModel {
        fn new(output_token: u32, vocab_size: usize) -> Self {
            Self {
                output_token,
                vocab_size,
                device: Device::Cpu,
            }
        }
    }

    impl ModelForward for ConstantMockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits_vec[pos * self.vocab_size + self.output_token as usize] = 100.0;
            }
            Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
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

    #[test]
    fn draft_model_proposes_correct_tokens() {
        let model = ConstantMockModel::new(42, 1000);
        let kv_cache = KVCacheManager::new(&test_cache_config()).expect("cache creation failed");
        let proposer = DraftModelProposer::new(model, kv_cache, 3);

        let tokens = [1u32, 2, 3, 4, 5];
        let result = proposer.propose(&tokens, 3);
        assert_eq!(result, vec![42, 42, 42]);
    }

    #[test]
    fn draft_model_respects_max_tokens() {
        let model = ConstantMockModel::new(42, 1000);
        let kv_cache = KVCacheManager::new(&test_cache_config()).expect("cache creation failed");
        let proposer = DraftModelProposer::new(model, kv_cache, 5);

        let tokens = [1u32, 2, 3];
        let result = proposer.propose(&tokens, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn draft_model_empty_input() {
        let model = ConstantMockModel::new(42, 1000);
        let kv_cache = KVCacheManager::new(&test_cache_config()).expect("cache creation failed");
        let proposer = DraftModelProposer::new(model, kv_cache, 3);

        let result = proposer.propose(&[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn draft_model_zero_k() {
        let model = ConstantMockModel::new(42, 1000);
        let kv_cache = KVCacheManager::new(&test_cache_config()).expect("cache creation failed");
        let proposer = DraftModelProposer::new(model, kv_cache, 3);

        let result = proposer.propose(&[1, 2, 3], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn draft_model_name() {
        let model = ConstantMockModel::new(42, 1000);
        let kv_cache = KVCacheManager::new(&test_cache_config()).expect("cache creation failed");
        let proposer = DraftModelProposer::new(model, kv_cache, 3);
        assert_eq!(proposer.name(), "draft_model");
    }

    #[test]
    fn draft_model_frees_cache_after_proposal() {
        let cache_config = test_cache_config();
        let kv_cache = KVCacheManager::new(&cache_config).expect("cache creation failed");
        let initial_free = kv_cache.num_free_blocks();

        let model = ConstantMockModel::new(42, 1000);
        let proposer = DraftModelProposer::new(model, kv_cache, 3);

        let _ = proposer.propose(&[1, 2, 3, 4, 5], 3);

        // After proposal, all blocks should be freed.
        let inner = proposer.inner.lock().expect("lock");
        assert_eq!(inner.kv_cache.num_free_blocks(), initial_free);
    }
}
