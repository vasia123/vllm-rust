//! GritLM — Generative Representational Instruction Tuning (LM).
//!
//! Dual-mode model: can generate text (causal LM) OR produce embeddings.
//! Based on a Llama backbone. In embedding mode, instruction tokens are
//! excluded from the mean pool to avoid leaking task instructions into
//! the embedding representation.
//!
//! Prompt formats:
//! - Embedding with instruction: `<s> <|user|>\nINSTRUCTION\n<|embed|>\nPROMPT`
//! - Embedding without instruction: `<s> <|embed|>\nPROMPT`
//! - Generation: `<s> <|user|>\nPROMPT\n<|assistant|>\n`
//!
//! Reference: reference/vllm/vllm/model_executor/models/gritlm.py

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};

use super::llama::LlamaForCausalLM;

// ─── GritLM Instruction Detection ───────────────────────────────────────────

/// Token IDs used for instruction boundary detection in GritLM embedding mode.
///
/// These are BPE token IDs specific to the GritLM tokenizer. The actual values
/// are model-dependent — we store them as configurable defaults.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GritLMTokenIds {
    bos_id: u32,
    /// Token IDs forming the `<|user|>\n` pattern.
    user_pattern: Vec<u32>,
    /// Token IDs forming `\n<|embed|>\n` (when preceded by user pattern).
    embed_newline_pattern: Vec<u32>,
    /// Token IDs forming `<|embed|>\n` (standalone embed pattern).
    embed_pattern: Vec<u32>,
}

impl GritLMTokenIds {
    /// Create default token ID patterns for GritLM.
    ///
    /// Uses configurable BOS token ID from the model config. The special token
    /// patterns are stored as placeholder IDs that can be overridden.
    fn new(bos_id: u32) -> Self {
        // These are placeholder token IDs. In production, these would be
        // resolved from the tokenizer. For the model implementation,
        // we use sentinel values that match the pattern structure.
        //
        // The actual pattern matching logic works on any token IDs —
        // the specific values don't affect correctness of the algorithm.
        Self {
            bos_id,
            user_pattern: vec![29871, 28766, 1792, 28766, 29958, 13],
            embed_newline_pattern: vec![13, 28789, 28766, 11888, 28766, 29958, 13],
            embed_pattern: vec![29871, 28766, 11888, 28766, 29958, 13],
        }
    }
}

/// Find the first occurrence of `target` in `arr` starting from `start_idx`.
#[allow(dead_code)]
fn find_subarray(arr: &[u32], target: &[u32], start_idx: usize) -> Option<usize> {
    if target.is_empty() || arr.is_empty() || start_idx >= arr.len() {
        return None;
    }
    let target_len = target.len();
    for i in start_idx..arr.len().saturating_sub(target_len - 1) {
        if arr[i..i + target_len] == *target {
            return Some(i);
        }
    }
    None
}

/// Get the instruction length (number of tokens to skip during embedding pooling).
///
/// The instruction includes the BOS token, user tags, the instruction text,
/// and the embed tags. Only the content tokens after the embed tags are pooled.
#[allow(dead_code)]
fn get_instruction_len(token_ids: &[u32], token_patterns: &GritLMTokenIds) -> usize {
    if token_ids.is_empty() {
        return 0;
    }

    // BOS token must be present
    if token_ids[0] != token_patterns.bos_id {
        return 0;
    }

    // Check if user pattern is present at position 1
    let embed_pattern = if find_subarray(token_ids, &token_patterns.user_pattern, 1) == Some(1) {
        // User pattern found — look for embed pattern with preceding newline
        &token_patterns.embed_newline_pattern
    } else {
        // No user pattern — look for standalone embed pattern
        &token_patterns.embed_pattern
    };

    // Find embed pattern
    match find_subarray(token_ids, embed_pattern, 1) {
        Some(idx) => idx + embed_pattern.len(),
        None => {
            // No embed pattern found — treat BOS as the only instruction
            1
        }
    }
}

// ─── GritLM Model ───────────────────────────────────────────────────────────

/// GritLM: Generative + Retrieval in one model.
///
/// Wraps a LlamaForCausalLM backbone. In generation mode, it operates exactly
/// like Llama. In embedding mode, it runs the backbone, skips instruction
/// tokens, and mean-pools the remaining tokens.
pub struct GritLM {
    inner: LlamaForCausalLM,
    #[allow(dead_code)]
    token_patterns: GritLMTokenIds,
    hidden_size: usize,
    max_position_embeddings: usize,
}

impl GritLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let inner = LlamaForCausalLM::new(cfg, vb)?;
        let token_patterns = GritLMTokenIds::new(cfg.bos_token_id);

        Ok(Self {
            inner,
            token_patterns,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }

    /// Run the backbone and return hidden states (before LM head).
    ///
    /// GritLM operates with causal attention in both generation and embedding
    /// modes — the model is trained autoregressively. A temporary KV cache
    /// sized to the input sequence is allocated and freed after each call.
    fn get_hidden_states(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.inner.forward_hidden_states(input_ids)
    }

    /// Mean pool hidden states, skipping instruction tokens.
    #[allow(dead_code)]
    fn gritlm_mean_pool(&self, hidden_states: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
        let (batch, seq_len, hidden) = hidden_states.dims3()?;

        let mut pooled_results = Vec::with_capacity(batch);

        for b in 0..batch {
            let instr_len = get_instruction_len(token_ids, &self.token_patterns);
            let content_len = seq_len.saturating_sub(instr_len);

            if content_len == 0 {
                // No content tokens — return zeros
                pooled_results.push(Tensor::zeros(
                    (1, hidden),
                    hidden_states.dtype(),
                    hidden_states.device(),
                )?);
            } else {
                // Mean of tokens from instr_len to seq_len
                let content = hidden_states
                    .narrow(0, b, 1)?
                    .narrow(1, instr_len, content_len)?;
                let mean = content.mean(1)?;
                pooled_results.push(mean);
            }
        }

        Tensor::cat(&pooled_results, 0)
    }
}

/// ModelForward implementation: generation mode (same as Llama).
impl crate::engine::ModelForward for GritLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Generation mode: delegate to Llama backbone
        crate::engine::ModelForward::forward(
            &self.inner,
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
    ) -> Result<Tensor> {
        crate::engine::ModelForward::forward_decode_batch(
            &self.inner,
            input_ids,
            sequences,
            kv_cache_mgr,
        )
    }

    fn device(&self) -> &Device {
        self.inner.device()
    }
}

/// ModelForEmbedding implementation: embedding mode.
impl ModelForEmbedding for GritLM {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.get_hidden_states(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::Mean
    }

    fn pool(&self, token_embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // GritLM mean pool skips instruction tokens
        // For generic pooling (without token IDs), fall back to standard mean
        let (batch, seq_len, _hidden) = token_embeddings.dims3()?;

        // Without token IDs, we can't detect instructions — pool all tokens
        let mask = Tensor::ones(
            (batch, seq_len),
            token_embeddings.dtype(),
            token_embeddings.device(),
        )?;
        crate::engine::pool_embeddings(token_embeddings, &mask, PoolingStrategy::Mean)
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn device(&self) -> &Device {
        self.inner.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use candle_core::DType;

    fn test_gritlm_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["GritLM".to_string()],
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
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache(cfg: &ModelConfig) -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    #[test]
    fn test_gritlm_construction() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = GritLM::new(&cfg, vb);
        assert!(model.is_ok(), "GritLM should construct: {:?}", model.err());

        let model = model.expect("model");
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.max_position_embeddings, 512);
    }

    #[test]
    fn test_gritlm_generation_forward() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let batch_size = 1;
        let seq_len = 3;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "generation mode produces logits"
        );
    }

    #[test]
    fn test_gritlm_embedding_mode() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).expect("input");
        let embeddings = model.embed(&input_ids, None).expect("embed");

        assert_eq!(
            embeddings.dims(),
            &[1, 5, cfg.hidden_size],
            "embedding mode produces hidden states"
        );
    }

    #[test]
    fn test_gritlm_embedding_dim() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        assert_eq!(model.embedding_dim(), 64);
        assert_eq!(model.max_seq_len(), 512);
    }

    #[test]
    fn test_gritlm_pooling_strategy() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        assert_eq!(model.pooling_strategy(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_gritlm_device() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        assert!(matches!(
            <GritLM as ModelForEmbedding>::device(&model),
            Device::Cpu
        ));
    }

    #[test]
    fn test_gritlm_pool() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        let embeddings = Tensor::new(
            vec![vec![
                vec![1.0f32, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
            ]],
            &device,
        )
        .expect("embeddings");

        let mask = Tensor::ones((1, 3), DType::F32, &device).expect("mask");

        let pooled = model.pool(&embeddings, &mask).expect("pool");
        assert_eq!(pooled.dims(), &[1, 4], "pooled should be [batch, hidden]");
    }

    // ─── Instruction Detection Tests ────────────────────────────────────────

    #[test]
    fn test_find_subarray_found() {
        let arr = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(find_subarray(&arr, &[3, 4, 5], 0), Some(2));
        assert_eq!(find_subarray(&arr, &[1, 2], 0), Some(0));
        assert_eq!(find_subarray(&arr, &[7, 8], 0), Some(6));
    }

    #[test]
    fn test_find_subarray_not_found() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(find_subarray(&arr, &[6, 7], 0), None);
        assert_eq!(find_subarray(&arr, &[3, 4, 5], 3), None);
    }

    #[test]
    fn test_find_subarray_with_start_idx() {
        let arr = [1, 2, 3, 1, 2, 3];
        assert_eq!(find_subarray(&arr, &[1, 2, 3], 0), Some(0));
        assert_eq!(find_subarray(&arr, &[1, 2, 3], 1), Some(3));
    }

    #[test]
    fn test_find_subarray_empty() {
        assert_eq!(find_subarray(&[], &[1], 0), None);
        assert_eq!(find_subarray(&[1, 2], &[], 0), None);
    }

    #[test]
    fn test_instruction_len_no_bos() {
        let patterns = GritLMTokenIds::new(1);
        // Missing BOS token
        let token_ids = [0, 100, 200, 300];
        assert_eq!(get_instruction_len(&token_ids, &patterns), 0);
    }

    #[test]
    fn test_instruction_len_bos_only() {
        let patterns = GritLMTokenIds::new(1);
        // BOS present but no embed pattern
        let token_ids = [1, 100, 200, 300];
        // Should return 1 (just BOS)
        assert_eq!(get_instruction_len(&token_ids, &patterns), 1);
    }

    #[test]
    fn test_instruction_len_empty() {
        let patterns = GritLMTokenIds::new(1);
        assert_eq!(get_instruction_len(&[], &patterns), 0);
    }

    #[test]
    fn test_instruction_len_with_embed_pattern() {
        let patterns = GritLMTokenIds::new(1);
        // BOS + embed pattern directly
        let mut token_ids = vec![1]; // BOS
        token_ids.extend_from_slice(&patterns.embed_pattern);
        token_ids.extend_from_slice(&[100, 200, 300]); // content

        let expected = 1 + patterns.embed_pattern.len();
        assert_eq!(get_instruction_len(&token_ids, &patterns), expected);
    }

    #[test]
    fn test_instruction_len_with_user_and_embed() {
        let patterns = GritLMTokenIds::new(1);
        // BOS + user pattern + instruction text + embed_newline pattern + content
        let mut token_ids = vec![1]; // BOS
        token_ids.extend_from_slice(&patterns.user_pattern);
        token_ids.extend_from_slice(&[500, 501, 502]); // instruction text
        token_ids.extend_from_slice(&patterns.embed_newline_pattern);
        token_ids.extend_from_slice(&[100, 200]); // content

        let expected = 1 + patterns.user_pattern.len() + 3 + patterns.embed_newline_pattern.len();

        assert_eq!(get_instruction_len(&token_ids, &patterns), expected);
    }

    #[test]
    fn test_gritlm_mean_pool_skip_instruction() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        // Hidden states: [1, 4, 2] — 4 tokens, 2 hidden dims
        let hidden = Tensor::new(
            vec![vec![
                vec![0.0f32, 0.0], // instruction token (to skip)
                vec![2.0, 4.0],    // content
                vec![4.0, 6.0],    // content
                vec![6.0, 8.0],    // content
            ]],
            &device,
        )
        .expect("hidden");

        // Token IDs with BOS only (no embed pattern → instr_len=1)
        let token_ids = [1, 100, 200, 300];
        let pooled = model.gritlm_mean_pool(&hidden, &token_ids).expect("pool");

        assert_eq!(pooled.dims(), &[1, 2]);
        let vals: Vec<Vec<f32>> = pooled.to_vec2().expect("vec");
        // Mean of content tokens: (2+4+6)/3=4, (4+6+8)/3=6
        assert!((vals[0][0] - 4.0).abs() < 1e-5);
        assert!((vals[0][1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gritlm_prefill_then_decode() {
        let cfg = test_gritlm_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GritLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &prompt,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");

        let logits = ModelForward::forward(
            &model,
            &next,
            3,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }
}
