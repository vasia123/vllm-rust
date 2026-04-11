use serde::Deserialize;

/// Normalise a raw HuggingFace `config.json` payload so that models which
/// nest transformer hyperparameters under a sub-object (Gemma 4's
/// `text_config`, some VLMs' `llm_config`, etc.) still deserialise into the
/// flat `ModelConfig` shape.
///
/// For Gemma 4 specifically we also:
/// - Pull `rope_parameters.{full,sliding}_attention.rope_theta` up to the
///   flat `rope_theta` / `rope_local_base_freq` fields so the deserializer
///   finds them.
/// - Rename `hidden_activation` → `hidden_act` to match transformers'
///   canonical naming.
/// - Collapse `eos_token_id` arrays (some Gemma checkpoints list several)
///   to a single integer.
///
/// The architecture list is left intact — callers can still dispatch on
/// `Gemma4ForConditionalGeneration` when the checkpoint is a VLM.
pub fn flatten_hf_model_config(raw: &str) -> Result<String, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_str(raw)?;
    let Some(obj) = value.as_object_mut() else {
        return Ok(raw.to_string());
    };

    // Pull `text_config` up to the top level. We only do this when the
    // top-level object does NOT already expose the flat transformer
    // fields — otherwise we'd clobber deliberate overrides.
    let has_flat_hidden = obj.contains_key("hidden_size") && obj.contains_key("num_hidden_layers");
    if !has_flat_hidden {
        if let Some(text_cfg) = obj.remove("text_config") {
            if let Some(text_obj) = text_cfg.as_object() {
                for (k, v) in text_obj {
                    obj.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
    }

    // HF uses `hidden_activation` for Gemma family; we expect `hidden_act`.
    if !obj.contains_key("hidden_act") {
        if let Some(act) = obj.remove("hidden_activation") {
            obj.insert("hidden_act".to_string(), act);
        }
    }

    // Gemma 4 stores RoPE theta per layer type under `rope_parameters`.
    // Surface the full-attention theta as `rope_theta` (primary) and the
    // sliding-attention theta as `rope_local_base_freq` so the rest of
    // the engine keeps working with flat fields.
    if let Some(rope_params) = obj.get("rope_parameters").cloned() {
        if !obj.contains_key("rope_theta") {
            let full_theta = rope_params
                .get("full_attention")
                .and_then(|fa| fa.get("rope_theta"))
                .and_then(|v| v.as_f64());
            let sliding_theta = rope_params
                .get("sliding_attention")
                .and_then(|sa| sa.get("rope_theta"))
                .and_then(|v| v.as_f64());
            if let Some(t) = full_theta.or(sliding_theta) {
                obj.insert("rope_theta".to_string(), serde_json::json!(t));
            }
        }
        if !obj.contains_key("rope_local_base_freq") {
            if let Some(t) = rope_params
                .get("sliding_attention")
                .and_then(|sa| sa.get("rope_theta"))
                .and_then(|v| v.as_f64())
            {
                obj.insert("rope_local_base_freq".to_string(), serde_json::json!(t));
            }
        }
    }

    // `eos_token_id` can be an array (e.g. Gemma 4 has `[1, 106]`).
    // `ModelConfig::eos_token_id` is a single `u32`, so take the first.
    if let Some(eos) = obj.get("eos_token_id").cloned() {
        if let Some(first) = eos.as_array().and_then(|a| a.first().cloned()) {
            obj.insert("eos_token_id".to_string(), first);
        }
    }

    // Most pre-2024 HF Llama/Mistral/etc. configs omit `head_dim` —
    // transformers computes it as `hidden_size / num_attention_heads`.
    // Our `ModelConfig` has `head_dim` as a required field because
    // newer models (Gemma, some Qwen) store it explicitly, so we
    // back-fill here when it's missing.
    if !obj.contains_key("head_dim") {
        let hidden_size = obj.get("hidden_size").and_then(|v| v.as_u64());
        let num_heads = obj.get("num_attention_heads").and_then(|v| v.as_u64());
        if let (Some(h), Some(n)) = (hidden_size, num_heads) {
            if n > 0 {
                obj.insert("head_dim".to_string(), serde_json::json!(h / n));
            }
        }
    }

    // Similar defensive defaults: `num_key_value_heads` defaults to
    // `num_attention_heads` when missing (multi-head = GQA with groups=1).
    if !obj.contains_key("num_key_value_heads") {
        if let Some(n) = obj.get("num_attention_heads").cloned() {
            obj.insert("num_key_value_heads".to_string(), n);
        }
    }

    // `tie_word_embeddings` defaults to false if missing.
    if !obj.contains_key("tie_word_embeddings") {
        obj.insert("tie_word_embeddings".to_string(), serde_json::json!(false));
    }

    // `rms_norm_eps` default (transformers uses 1e-6).
    if !obj.contains_key("rms_norm_eps") {
        obj.insert("rms_norm_eps".to_string(), serde_json::json!(1e-6));
    }

    // `rope_theta` default (transformers uses 10000.0 if not specified).
    if !obj.contains_key("rope_theta") {
        obj.insert("rope_theta".to_string(), serde_json::json!(10_000.0));
    }

    serde_json::to_string(&value)
}

/// Handles sliding_window configs that are either null, an int, or a list of
/// ints (Mistral-style per-layer windows). Lists are collapsed to a single
/// value after validating all non-null entries are identical.
fn deserialize_sliding_window<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;
    match &value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_u64()
            .map(|v| Some(v as usize))
            .ok_or_else(|| serde::de::Error::custom("sliding_window must be a positive integer")),
        serde_json::Value::Array(arr) => {
            let values: Vec<usize> = arr
                .iter()
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64().map(|n| n as usize).ok_or_else(|| {
                        serde::de::Error::custom(
                            "sliding_window array must contain integers or null",
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            if values.is_empty() {
                return Ok(None);
            }
            let first = values[0];
            if values.iter().any(|&v| v != first) {
                return Err(serde::de::Error::custom(
                    "sliding_window array must contain identical non-null values",
                ));
            }
            Ok(Some(first))
        }
        _ => Err(serde::de::Error::custom(
            "sliding_window must be null, integer, or array",
        )),
    }
}

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

    #[serde(default, deserialize_with = "deserialize_sliding_window")]
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

    #[test]
    fn sliding_window_as_int() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "intermediate_size": 128, "vocab_size": 256,
            "max_position_embeddings": 512, "head_dim": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "tie_word_embeddings": true,
            "bos_token_id": 1, "eos_token_id": 2,
            "sliding_window": 4096
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn sliding_window_as_list() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "intermediate_size": 128, "vocab_size": 256,
            "max_position_embeddings": 512, "head_dim": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "tie_word_embeddings": true,
            "bos_token_id": 1, "eos_token_id": 2,
            "sliding_window": [4096, null, 4096, null]
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn sliding_window_all_null_list() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "intermediate_size": 128, "vocab_size": 256,
            "max_position_embeddings": 512, "head_dim": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "tie_word_embeddings": true,
            "bos_token_id": 1, "eos_token_id": 2,
            "sliding_window": [null, null]
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, None);
    }

    // ─── flatten_hf_model_config tests ──────────────────────────────

    #[test]
    fn flatten_gemma4_text_config_pulls_fields_up() {
        // Realistic slice of `google/gemma-4-E2B-it/config.json`.
        let raw = r#"{
            "architectures": ["Gemma4ForConditionalGeneration"],
            "tie_word_embeddings": true,
            "eos_token_id": [1, 106],
            "text_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "intermediate_size": 6144,
                "vocab_size": 262144,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "sliding_window": 512,
                "hidden_activation": "gelu_pytorch_tanh",
                "bos_token_id": 2,
                "eos_token_id": 1,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        }"#;

        let flattened = flatten_hf_model_config(raw).expect("flatten");
        let cfg: ModelConfig =
            serde_json::from_str(&flattened).expect("deserialise flattened config");

        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh");
        // Full-attention theta lands on the top-level `rope_theta`.
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        // Sliding theta surfaces as `rope_local_base_freq` in `extra`.
        assert_eq!(
            cfg.extra
                .get("rope_local_base_freq")
                .and_then(|v| v.as_f64()),
            Some(10_000.0)
        );
        // `rope_parameters` is kept so Gemma4ExtraConfig can read
        // partial_rotary_factor / rope_type per layer type.
        assert!(cfg.extra.get("rope_parameters").is_some());
        // `eos_token_id` at top level is an array `[1, 106]`; the
        // collapser picks the first element. (`text_config.eos_token_id`
        // is 1, so both paths agree here.)
        assert_eq!(cfg.eos_token_id, 1);
        // Architecture is preserved so the VLM dispatch still fires.
        assert_eq!(cfg.architectures, vec!["Gemma4ForConditionalGeneration"]);
    }

    #[test]
    fn flatten_preserves_flat_config() {
        // A flat config (no `text_config`) should round-trip to the same
        // logical shape.
        let raw = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "intermediate_size": 128, "vocab_size": 100,
            "max_position_embeddings": 256, "head_dim": 16, "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "tie_word_embeddings": false,
            "bos_token_id": 1, "eos_token_id": 2
        }"#;
        let flattened = flatten_hf_model_config(raw).expect("flatten");
        let cfg: ModelConfig = serde_json::from_str(&flattened).expect("deserialise");
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.rope_theta, 10_000.0);
    }
}
