//! MusicFlamingo: identical architecture to AudioFlamingo3.
//!
//! Python implementation is an empty subclass:
//! `class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration): pass`
//!
//! Rust re-exports `AudioFlamingo3ForConditionalGeneration` under the
//! `MusicFlamingoForConditionalGeneration` name; both checkpoint formats
//! work with the same weight-loading logic.
//!
//! Reference: reference/vllm/vllm/model_executor/models/musicflamingo.py

pub use super::audioflamingo3::AudioFlamingo3ForConditionalGeneration as MusicFlamingoForConditionalGeneration;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::MusicFlamingoForConditionalGeneration;
    use crate::config::ModelConfig;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager};
    use crate::multimodal::{MultimodalInputs, ProcessedAudio};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    fn make_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "audio_config".into(),
            serde_json::json!({
                "hidden_size": 16,
                "num_mel_bins": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "max_source_positions": 64,
                "d_model": 16,
            }),
        );
        extra.insert(
            "text_config".into(),
            serde_json::json!({
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "hidden_act": "silu",
            }),
        );
        extra.insert("projector_bias".into(), serde_json::json!(true));
        extra.insert("projector_hidden_act".into(), serde_json::json!("gelu"));
        extra.insert("audio_token_id".into(), serde_json::json!(10u32));
        ModelConfig {
            architectures: vec!["MusicFlamingoForConditionalGeneration".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    fn make_cache(cfg: &ModelConfig, dev: &Device) -> KVCacheManager {
        let cc = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: dev.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        KVCacheManager::new(&cc).unwrap()
    }

    #[test]
    fn musicflamingo_construction() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MusicFlamingoForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "{:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn musicflamingo_text_forward() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MusicFlamingoForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut cache = make_cache(&cfg, &dev);
        let bt = crate::kv_cache::BlockTable::from_block_ids(vec![0, 1], 0);
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &dev).unwrap();
        let out = model
            .forward(&input, 0, &mut cache, &bt, &[0, 1, 2, 3])
            .unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn musicflamingo_audio_scatter() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MusicFlamingoForConditionalGeneration::new(&cfg, vb).unwrap();

        // Use an in-range audio token ID matching the test vocab_size=32
        let audio_token_id = 10u32;
        let seq_len = 4usize;
        let audio_tokens = 2usize;
        let hidden = cfg.hidden_size;

        // Sequence: [audio, audio, text, text]
        let token_ids = vec![audio_token_id, audio_token_id, 10u32, 11u32];
        let embedding = Tensor::zeros((audio_tokens, hidden), DType::F32, &dev).unwrap();
        let processed = ProcessedAudio::new(embedding, audio_tokens);
        let mm = MultimodalInputs::with_audio(token_ids.clone(), vec![(1, processed)]);

        let input_ids = Tensor::from_vec(token_ids, (1, seq_len), &dev).unwrap();
        let mut cache = make_cache(&cfg, &dev);
        let bt = crate::kv_cache::BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let out = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut cache, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(out.dims()[0..2], [1, seq_len]);
    }
}
