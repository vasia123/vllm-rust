use serde::Serialize;

use crate::config::ModelConfig;
use crate::perf_estimate::vram_fitness::WeightDtype;

/// Analyzed model profile for performance estimation.
/// Built from `ModelConfig` without loading weights.
#[derive(Debug, Clone, Serialize)]
pub struct ModelProfile {
    pub model_id: String,
    pub architecture: String,
    /// Total estimated parameters.
    pub total_params: u64,
    /// Active parameters per token (differs for MoE: only top-k experts active).
    pub active_params_per_token: u64,
    /// Total model weight size in bytes.
    pub model_weight_bytes: u64,
    /// KV cache bytes per token (all layers combined).
    pub kv_bytes_per_token: usize,
    /// FLOPs per token for decode (single token, all layers).
    pub flops_per_token_decode: u64,
    /// FLOPs per token for prefill (linear component, excluding quadratic attention).
    pub flops_per_token_prefill_linear: u64,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub is_moe: bool,
    pub num_experts: Option<usize>,
    pub experts_per_token: Option<usize>,
    pub tie_word_embeddings: bool,
}

impl ModelProfile {
    /// Build a model profile from ModelConfig.
    pub fn from_config(model_id: &str, config: &ModelConfig, weight_dtype: WeightDtype) -> Self {
        let hidden = config.hidden_size as u64;
        let kv_heads = config.num_key_value_heads as u64;
        let head_dim = config.head_dim as u64;
        let intermediate = config.intermediate_size as u64;
        let vocab = config.vocab_size as u64;
        let layers = config.num_hidden_layers as u64;

        // Embedding parameters
        let embedding_params = vocab * hidden;
        let lm_head_params = if config.tie_word_embeddings {
            0
        } else {
            vocab * hidden
        };

        // Attention parameters per layer:
        //   Q: hidden * (num_heads * head_dim)
        //   K: hidden * (kv_heads * head_dim)
        //   V: hidden * (kv_heads * head_dim)
        //   O: (num_heads * head_dim) * hidden
        let num_heads = config.num_attention_heads as u64;
        let attn_per_layer = hidden * (num_heads * head_dim)
            + 2 * hidden * (kv_heads * head_dim)
            + (num_heads * head_dim) * hidden;

        // FFN parameters per layer
        let is_gated = is_gated_activation(&config.hidden_act);
        let ffn_multiplier: u64 = if is_gated { 3 } else { 2 };
        let dense_ffn_per_layer = ffn_multiplier * hidden * intermediate;

        // MoE handling
        let is_moe = config.is_moe();
        let num_experts = config.num_routed_experts();
        let experts_per_token = config.num_experts_per_tok();
        let moe_intermediate = config
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(intermediate);

        // Norm parameters per layer (2 RMSNorm layers * hidden_size)
        let norm_per_layer = 2 * hidden;

        let (ffn_total_per_layer, ffn_active_per_layer) = if is_moe {
            let n_experts = num_experts.unwrap_or(8) as u64;
            let expert_params = ffn_multiplier * hidden * moe_intermediate;
            let total = n_experts * expert_params;
            let topk = experts_per_token.unwrap_or(2) as u64;
            let active = topk * expert_params;
            // Some MoE models also have shared experts
            let shared_expert = config
                .shared_expert_intermediate_size()
                .map(|s| ffn_multiplier * hidden * s as u64)
                .unwrap_or(0);
            (total + shared_expert, active + shared_expert)
        } else {
            (dense_ffn_per_layer, dense_ffn_per_layer)
        };

        // Sparse step: not all layers are MoE
        let sparse_step = config.decoder_sparse_step().unwrap_or(1);
        let moe_layers = if is_moe {
            layers / sparse_step as u64
        } else {
            0
        };
        let dense_layers = layers - moe_layers;

        let total_params = embedding_params
            + lm_head_params
            + dense_layers * (attn_per_layer + dense_ffn_per_layer + norm_per_layer)
            + moe_layers * (attn_per_layer + ffn_total_per_layer + norm_per_layer);

        let active_params_per_token = embedding_params
            + lm_head_params
            + dense_layers * (attn_per_layer + dense_ffn_per_layer + norm_per_layer)
            + moe_layers * (attn_per_layer + ffn_active_per_layer + norm_per_layer);

        let bytes_per_param = weight_dtype.bytes_per_param();
        let model_weight_bytes = (total_params as f64 * bytes_per_param) as u64;

        // KV cache per token (all layers)
        let is_mla = config.is_mla_model();
        let kv_bytes_per_token = if is_mla {
            let kv_lora_rank = config
                .extra
                .get("kv_lora_rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize;
            let qk_rope_head_dim = config
                .extra
                .get("qk_rope_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize;
            // MLA compresses KV to (kv_lora_rank + qk_rope_head_dim) per layer
            (kv_lora_rank + qk_rope_head_dim) * config.num_hidden_layers * 2 // BF16 = 2 bytes
        } else {
            // Standard: 2 (K+V) * kv_heads * head_dim * layers * elem_size
            2 * config.num_key_value_heads * config.head_dim * config.num_hidden_layers * 2
        };

        // FLOPs: 2 * params (one multiply + one add per matmul parameter)
        let flops_per_token_decode = 2 * active_params_per_token;
        let flops_per_token_prefill_linear = flops_per_token_decode;

        let architecture = config.architectures.first().cloned().unwrap_or_default();

        Self {
            model_id: model_id.to_string(),
            architecture,
            total_params,
            active_params_per_token,
            model_weight_bytes,
            kv_bytes_per_token,
            flops_per_token_decode,
            flops_per_token_prefill_linear,
            num_layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            is_moe,
            num_experts,
            experts_per_token,
            tie_word_embeddings: config.tie_word_embeddings,
        }
    }

    /// Quick estimate of total params from basic config values.
    /// Used when only approximate sizing is needed.
    pub fn estimate_params(
        hidden_size: usize,
        num_layers: usize,
        intermediate_size: usize,
        vocab_size: usize,
        tie_word_embeddings: bool,
    ) -> u64 {
        let h = hidden_size as u64;
        let l = num_layers as u64;
        let i = intermediate_size as u64;
        let v = vocab_size as u64;
        let embed = v * h;
        let lm_head = if tie_word_embeddings { 0 } else { v * h };
        let attn = 4 * h * h; // approximate: Q, K, V, O all h×h
        let ffn = 3 * h * i; // SwiGLU: gate + up + down
        let norms = 2 * h;
        embed + lm_head + l * (attn + ffn + norms)
    }
}

fn is_gated_activation(act: &str) -> bool {
    matches!(
        act,
        "silu" | "swiglu" | "geglu" | "gelu_pytorch_tanh" | "silu_and_mul" | "gelu_and_mul"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_8b_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["LlamaForCausalLM".to_string()],
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            intermediate_size: 14336,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            head_dim: 128,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            tie_word_embeddings: false,
            ..Default::default()
        }
    }

    fn qwen3_06b_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            num_hidden_layers: 28,
            intermediate_size: 3072,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            head_dim: 64,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_model_profile_llama_8b() {
        let config = llama_8b_config();
        let profile =
            ModelProfile::from_config("meta-llama/Llama-3.1-8B", &config, WeightDtype::Bf16);

        // Llama-3.1-8B has ~8B params
        let params_b = profile.total_params as f64 / 1e9;
        assert!(
            (6.0..=10.0).contains(&params_b),
            "Expected ~8B params, got {params_b:.1}B"
        );

        // BF16: ~16GB weights
        let weight_gb = profile.model_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (12.0..=20.0).contains(&weight_gb),
            "Expected ~16GB weights, got {weight_gb:.1}GB"
        );

        assert!(!profile.is_moe);
        assert!(!profile.tie_word_embeddings);
        assert_eq!(profile.active_params_per_token, profile.total_params);
    }

    #[test]
    fn test_model_profile_qwen3_06b() {
        let config = qwen3_06b_config();
        let profile = ModelProfile::from_config("Qwen/Qwen3-0.6B", &config, WeightDtype::Bf16);

        let params_b = profile.total_params as f64 / 1e9;
        assert!(
            (0.3..=1.0).contains(&params_b),
            "Expected ~0.6B params, got {params_b:.2}B"
        );

        assert!(profile.tie_word_embeddings);
    }

    #[test]
    fn test_model_profile_kv_bytes() {
        let config = llama_8b_config();
        let profile = ModelProfile::from_config("test", &config, WeightDtype::Bf16);

        // Standard KV: 2 * 8 kv_heads * 128 head_dim * 32 layers * 2 bytes = 131072 bytes/token
        let expected = 2 * 8 * 128 * 32 * 2;
        assert_eq!(profile.kv_bytes_per_token, expected);
    }

    #[test]
    fn test_model_profile_moe() {
        let mut config = llama_8b_config();
        config
            .extra
            .insert("n_routed_experts".to_string(), serde_json::json!(64));
        config
            .extra
            .insert("num_experts_per_tok".to_string(), serde_json::json!(6));
        config
            .extra
            .insert("moe_intermediate_size".to_string(), serde_json::json!(2048));

        let profile = ModelProfile::from_config("moe-test", &config, WeightDtype::Bf16);

        assert!(profile.is_moe);
        assert_eq!(profile.num_experts, Some(64));
        assert_eq!(profile.experts_per_token, Some(6));
        // Active params < total params because only top-k experts are used
        assert!(profile.active_params_per_token < profile.total_params);
    }

    #[test]
    fn test_model_profile_int4_weight_size() {
        let config = llama_8b_config();
        let bf16_profile = ModelProfile::from_config("test", &config, WeightDtype::Bf16);
        let int4_profile = ModelProfile::from_config("test", &config, WeightDtype::Int4);

        // INT4 should be ~4x smaller than BF16
        let ratio = bf16_profile.model_weight_bytes as f64 / int4_profile.model_weight_bytes as f64;
        assert!(
            (3.5..=4.5).contains(&ratio),
            "Expected ~4x ratio, got {ratio:.1}x"
        );
    }

    #[test]
    fn test_estimate_params_quick() {
        let params = ModelProfile::estimate_params(4096, 32, 14336, 128256, false);
        let params_b = params as f64 / 1e9;
        assert!(
            (6.0..=10.0).contains(&params_b),
            "Expected ~8B params, got {params_b:.1}B"
        );
    }

    #[test]
    fn test_flops_per_token() {
        let config = llama_8b_config();
        let profile = ModelProfile::from_config("test", &config, WeightDtype::Bf16);

        // FLOPs per token ≈ 2 * params for dense model
        assert_eq!(
            profile.flops_per_token_decode,
            2 * profile.active_params_per_token
        );
    }

    #[test]
    fn test_gated_activation_detection() {
        assert!(is_gated_activation("silu"));
        assert!(is_gated_activation("swiglu"));
        assert!(!is_gated_activation("gelu"));
        assert!(!is_gated_activation("relu"));
    }
}
