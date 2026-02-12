//! Arcee (AFM) causal language model.
//!
//! Architecturally identical to Jais2 but uses RMSNorm instead of LayerNorm.
//! The MLP uses relu-squared activation with up_proj/down_proj (no gate_proj),
//! same weight naming as Llama family.
//!
//! See `jais2.rs` for the shared implementation.

pub use super::jais2::{new_arcee, new_arcee_with_tp};

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::new_arcee;
    use crate::config::ModelConfig;
    use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};

    fn arcee_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["ArceeForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "relu2".to_string(),
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

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_arcee_reexport_construction() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = new_arcee(&cfg, vb);
        assert!(
            model.is_ok(),
            "new_arcee from arcee module should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_arcee_reexport_forward() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = new_arcee(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_arcee_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = new_arcee(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 4);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("trait forward");

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }
}
