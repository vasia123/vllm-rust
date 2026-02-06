use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,

    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub attention_bias: Option<bool>,

    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["LlamaForCausalLM".to_string()],
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            head_dim: 128,
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
}

impl ModelConfig {
    /// Check if this model uses MLA (Multi-head Latent Attention).
    ///
    /// MLA is used by DeepSeek V2/V3 models and identified by the presence
    /// of `kv_lora_rank` in the config.
    pub fn is_mla_model(&self) -> bool {
        self.extra.contains_key("kv_lora_rank")
    }

    /// Extract MLA dimensions from config.
    ///
    /// Returns `None` if this is not an MLA model.
    pub fn mla_dims(&self) -> Option<crate::kv_cache::MLADims> {
        crate::kv_cache::MLADims::from_config_extra(&self.extra)
    }

    /// Check if this is a DeepSeek model.
    pub fn is_deepseek(&self) -> bool {
        self.architectures.iter().any(|a| a.contains("DeepSeek"))
    }

    /// Check if this is a MoE (Mixture of Experts) model.
    pub fn is_moe(&self) -> bool {
        self.extra.contains_key("n_routed_experts")
    }

    /// Get the number of routed experts (for MoE models).
    pub fn num_routed_experts(&self) -> Option<usize> {
        self.extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the top-k experts per token (for MoE models).
    pub fn num_experts_per_tok(&self) -> Option<usize> {
        self.extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    // ─── GLM Model Config Getters ───────────────────────────────────────────────

    /// Get partial rotary factor for GLM models (default 1.0 = full rotation).
    /// GLM models use 0.5 to only rotate half of head_dim.
    pub fn partial_rotary_factor(&self) -> f64 {
        self.extra
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
    }

    /// Check if neox-style rotation is used (interleaved vs split).
    /// GLM models use false (split style).
    pub fn is_neox_style(&self) -> bool {
        // "original_rope" = true means use standard (neox) style
        // GLM uses !original_rope which defaults to split style
        self.extra
            .get("original_rope")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Get rope ratio multiplier (ChatGLM specific).
    pub fn rope_ratio(&self) -> f64 {
        self.extra
            .get("rope_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
    }

    /// Get multi_query_group_num for ChatGLM MQA.
    pub fn multi_query_group_num(&self) -> Option<usize> {
        self.extra
            .get("multi_query_group_num")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    // ─── MoE Config Getters (Qwen MoE, GLM4 MoE) ────────────────────────────────

    /// Get shared expert intermediate size (Qwen2-MoE).
    pub fn shared_expert_intermediate_size(&self) -> Option<usize> {
        self.extra
            .get("shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get decoder sparse step - which layers are MoE (Qwen2-MoE, Qwen3-MoE).
    /// If decoder_sparse_step = 2, every 2nd layer is MoE.
    pub fn decoder_sparse_step(&self) -> Option<usize> {
        self.extra
            .get("decoder_sparse_step")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get first_k_dense_replace - first K layers are dense (GLM4-MoE).
    pub fn first_k_dense_replace(&self) -> Option<usize> {
        self.extra
            .get("first_k_dense_replace")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get MoE intermediate size (different from standard intermediate_size).
    pub fn moe_intermediate_size(&self) -> Option<usize> {
        self.extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get number of expert groups (GLM4-MoE grouped top-k).
    pub fn n_group(&self) -> Option<usize> {
        self.extra
            .get("n_group")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get top-k per expert group (GLM4-MoE grouped top-k).
    pub fn topk_group(&self) -> Option<usize> {
        self.extra
            .get("topk_group")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get number of shared experts (GLM4-MoE).
    pub fn n_shared_experts(&self) -> Option<usize> {
        self.extra
            .get("n_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get routed scaling factor (GLM4-MoE output scaling).
    pub fn routed_scaling_factor(&self) -> f64 {
        self.extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
    }

    /// Check if QK normalization is used (GLM4-MoE, Qwen3-MoE).
    pub fn use_qk_norm(&self) -> bool {
        self.extra
            .get("use_qk_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Check if top-k probabilities should be renormalized.
    pub fn norm_topk_prob(&self) -> bool {
        self.extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Get MLP-only layers list (Qwen MoE models).
    pub fn mlp_only_layers(&self) -> Vec<usize> {
        self.extra
            .get("mlp_only_layers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the number of experts for generic MoE (num_experts field).
    pub fn num_experts(&self) -> Option<usize> {
        self.extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEEPSEEK_V3_CONFIG: &str = r#"{
        "architectures": ["DeepSeekForCausalLM"],
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "num_hidden_layers": 60,
        "intermediate_size": 18432,
        "vocab_size": 129280,
        "max_position_embeddings": 163840,
        "head_dim": 192,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000,
        "tie_word_embeddings": false,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "q_lora_rank": 1536,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8
    }"#;

    const QWEN3_06B_CONFIG: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    #[test]
    fn parse_qwen3_06b_config() {
        let config: ModelConfig =
            serde_json::from_str(QWEN3_06B_CONFIG).expect("failed to parse config");

        assert_eq!(config.architectures, vec!["Qwen3ForCausalLM"]);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.max_position_embeddings, 40960);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.hidden_act, "silu");
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.bos_token_id, 151643);
        assert_eq!(config.eos_token_id, 151645);
    }

    #[test]
    fn gqa_ratio_is_correct() {
        let config: ModelConfig =
            serde_json::from_str(QWEN3_06B_CONFIG).expect("failed to parse config");

        let gqa_groups = config.num_attention_heads / config.num_key_value_heads;
        assert_eq!(gqa_groups, 2);
    }

    #[test]
    fn parse_deepseek_v3_config() {
        let config: ModelConfig =
            serde_json::from_str(DEEPSEEK_V3_CONFIG).expect("failed to parse config");

        assert!(config.is_deepseek());
        assert!(config.is_mla_model());
        assert!(config.is_moe());
    }

    #[test]
    fn mla_dims_extraction() {
        let config: ModelConfig =
            serde_json::from_str(DEEPSEEK_V3_CONFIG).expect("failed to parse config");

        let dims = config.mla_dims().expect("should have MLA dims");
        assert_eq!(dims.kv_lora_rank, 512);
        assert_eq!(dims.qk_rope_head_dim, 64);
        assert_eq!(dims.qk_nope_head_dim, 128);
        assert_eq!(dims.v_head_dim, 128);
    }

    #[test]
    fn moe_config_extraction() {
        let config: ModelConfig =
            serde_json::from_str(DEEPSEEK_V3_CONFIG).expect("failed to parse config");

        assert_eq!(config.num_routed_experts(), Some(256));
        assert_eq!(config.num_experts_per_tok(), Some(8));
    }

    #[test]
    fn non_mla_model() {
        let config: ModelConfig =
            serde_json::from_str(QWEN3_06B_CONFIG).expect("failed to parse config");

        assert!(!config.is_mla_model());
        assert!(config.mla_dims().is_none());
        assert!(!config.is_deepseek());
        assert!(!config.is_moe());
    }
}
