//! Integration tests for model construction with synthetic weights.
//!
//! These tests verify that model architectures can be constructed from
//! `ModelConfig` using `VarBuilder::zeros`, and that `from_config` dispatches
//! correctly. No real weights or GPU are required.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use vllm_core::{
    config::ModelConfig,
    engine::ModelForward,
    kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager},
    models::{from_config, ModelError},
};

// ─── Tiny Config Helpers ─────────────────────────────────────────────────────

/// Create a tiny model config suitable for CPU testing with zero weights.
/// All models using standard transformer architecture (RoPE, RMSNorm, SwiGLU)
/// can be tested with this base config.
fn tiny_config(arch: &str) -> ModelConfig {
    ModelConfig {
        architectures: vec![arch.to_string()],
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
        attention_bias: None,
        extra: serde_json::Map::new(),
    }
}

/// Create a tiny config for GPT-2 which needs LayerNorm bias and different MLP structure.
/// GPT-2 is MHA (num_kv_heads == num_attention_heads) and computes head_dim as
/// hidden_size / num_attention_heads.
fn tiny_gpt2_config() -> ModelConfig {
    let mut cfg = tiny_config("GPT2LMHeadModel");
    cfg.hidden_act = "gelu_new".to_string();
    cfg.attention_bias = Some(true);
    // GPT-2 is MHA: num_kv_heads must equal num_attention_heads
    cfg.num_key_value_heads = cfg.num_attention_heads;
    // GPT-2 computes head_dim from hidden_size / num_attention_heads
    cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    // GPT-2 uses layer_norm_epsilon from extra
    cfg.extra.insert(
        "layer_norm_epsilon".to_string(),
        serde_json::Value::from(1e-5),
    );
    cfg
}

/// Create a tiny config for Phi-3 which uses fused QKV and gate_up_proj.
fn tiny_phi3_config() -> ModelConfig {
    tiny_config("Phi3ForCausalLM")
}

/// Create a tiny config for Gemma which uses (1+weight) RMSNorm.
fn tiny_gemma_config() -> ModelConfig {
    tiny_config("GemmaForCausalLM")
}

/// Create a KV cache matching the given model config.
fn kv_cache_for_config(cfg: &ModelConfig) -> KVCacheManager {
    let config = CacheConfig {
        block_size: 16,
        num_blocks: 32,
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    };
    KVCacheManager::new(&config).unwrap()
}

// ─── from_config dispatch tests ──────────────────────────────────────────────

#[test]
fn test_from_config_llama() {
    let cfg = tiny_config("LlamaForCausalLM");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "LlamaForCausalLM construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_mistral() {
    let cfg = tiny_config("MistralForCausalLM");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "MistralForCausalLM construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_gemma() {
    let cfg = tiny_gemma_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "GemmaForCausalLM construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_qwen2() {
    let cfg = tiny_config("Qwen2ForCausalLM");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "Qwen2ForCausalLM construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_qwen3() {
    let cfg = tiny_config("Qwen3ForCausalLM");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "Qwen3ForCausalLM construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_gpt2() {
    let cfg = tiny_gpt2_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "GPT2LMHeadModel construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_phi3() {
    let cfg = tiny_phi3_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb);
    assert!(
        model.is_ok(),
        "Phi3ForCausalLM construction failed: {:?}",
        model.err()
    );
}

#[test]
fn test_from_config_unknown_architecture() {
    let cfg = tiny_config("UnknownForCausalLM");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = from_config(&cfg, vb);
    assert!(result.is_err());
    match result.err().unwrap() {
        ModelError::UnsupportedArchitecture(arch) => {
            assert_eq!(arch, "UnknownForCausalLM");
        }
        other => panic!("expected UnsupportedArchitecture, got {:?}", other),
    }
}

#[test]
fn test_from_config_empty_architectures() {
    let mut cfg = tiny_config("LlamaForCausalLM");
    cfg.architectures.clear();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = from_config(&cfg, vb);
    assert!(result.is_err());
    match result.err().unwrap() {
        ModelError::UnsupportedArchitecture(msg) => {
            assert!(
                msg.contains("empty"),
                "expected empty architectures error, got: {msg}"
            );
        }
        other => panic!("expected UnsupportedArchitecture, got {:?}", other),
    }
}

// ─── Forward pass shape tests ────────────────────────────────────────────────

/// Verify that the forward pass produces logits of the correct shape.
fn assert_forward_shape(arch: &str, cfg: &ModelConfig) {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(cfg, vb).unwrap_or_else(|e| {
        panic!("{arch} construction failed: {e:?}");
    });

    let mut kv_cache_mgr = kv_cache_for_config(cfg);
    let mut block_table = BlockTable::new(16);
    let prompt_len = 4;

    kv_cache_mgr
        .allocate_for_request(&mut block_table, prompt_len)
        .unwrap();
    let slot_mapping = block_table.slot_mapping(0, prompt_len);
    let input_ids = Tensor::zeros((1, prompt_len), DType::U32, &Device::Cpu).unwrap();

    let logits = model
        .forward(
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .unwrap_or_else(|e| {
            panic!("{arch} forward pass failed: {e:?}");
        });

    let dims = logits.dims();
    // Logits should be [batch=1, seq_len, vocab_size]
    // Some models return [batch, vocab_size] for decode (seq_len=1)
    assert!(
        dims.len() == 2 || dims.len() == 3,
        "{arch} logits should be 2D or 3D, got {dims:?}"
    );

    let vocab_dim = dims[dims.len() - 1];
    assert_eq!(
        vocab_dim, cfg.vocab_size,
        "{arch} vocab dimension mismatch: expected {}, got {}",
        cfg.vocab_size, vocab_dim
    );

    kv_cache_mgr.free_request(&mut block_table).unwrap();
}

#[test]
fn test_forward_shape_llama() {
    let cfg = tiny_config("LlamaForCausalLM");
    assert_forward_shape("LlamaForCausalLM", &cfg);
}

#[test]
fn test_forward_shape_mistral() {
    let cfg = tiny_config("MistralForCausalLM");
    assert_forward_shape("MistralForCausalLM", &cfg);
}

#[test]
fn test_forward_shape_gemma() {
    let cfg = tiny_gemma_config();
    assert_forward_shape("GemmaForCausalLM", &cfg);
}

#[test]
fn test_forward_shape_qwen2() {
    let cfg = tiny_config("Qwen2ForCausalLM");
    assert_forward_shape("Qwen2ForCausalLM", &cfg);
}

#[test]
fn test_forward_shape_qwen3() {
    let cfg = tiny_config("Qwen3ForCausalLM");
    assert_forward_shape("Qwen3ForCausalLM", &cfg);
}

#[test]
fn test_forward_shape_gpt2() {
    let cfg = tiny_gpt2_config();
    assert_forward_shape("GPT2LMHeadModel", &cfg);
}

#[test]
fn test_forward_shape_phi3() {
    let cfg = tiny_phi3_config();
    assert_forward_shape("Phi3ForCausalLM", &cfg);
}

// ─── Model device tests ─────────────────────────────────────────────────────

#[test]
fn test_model_device_is_cpu() {
    let cfg = tiny_config("LlamaForCausalLM");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = from_config(&cfg, vb).unwrap();
    assert!(
        matches!(model.device(), Device::Cpu),
        "model device should be CPU"
    );
}

// ─── Registry integration ────────────────────────────────────────────────────

#[test]
fn test_all_from_config_architectures_in_registry() {
    use vllm_core::models::find_architecture;

    let architectures = [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GemmaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "GPT2LMHeadModel",
        "Phi3ForCausalLM",
    ];

    for arch in &architectures {
        let info = find_architecture(arch);
        assert!(
            info.is_some(),
            "{arch} should be in the architecture registry"
        );
    }
}
