use crate::config::ModelConfig;

/// Create a tiny Llama-like config for testing (4 layers, 64 hidden, 2 heads).
/// Uses < 1MB of parameters.
pub fn tiny_llama_config() -> ModelConfig {
    ModelConfig {
        architectures: vec!["LlamaForCausalLM".to_string()],
        hidden_size: 64,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        num_hidden_layers: 4,
        intermediate_size: 128,
        vocab_size: 256,
        max_position_embeddings: 512,
        head_dim: 32,
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

/// Create a tiny BERT-like config for testing encoder-only models.
/// Uses < 1MB of parameters.
pub fn tiny_bert_config() -> ModelConfig {
    ModelConfig {
        architectures: vec!["BertModel".to_string()],
        hidden_size: 64,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        num_hidden_layers: 2,
        intermediate_size: 128,
        vocab_size: 256,
        max_position_embeddings: 128,
        head_dim: 32,
        hidden_act: "gelu".to_string(),
        rms_norm_eps: 1e-12,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        bos_token_id: 101,
        eos_token_id: 102,
        sliding_window: None,
        attention_bias: Some(true),
        extra: serde_json::Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_llama_config_is_valid() {
        let config = tiny_llama_config();
        assert_eq!(config.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 2);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.intermediate_size, 128);
        assert_eq!(config.vocab_size, 256);
        assert_eq!(config.head_dim, 32);
        // GQA ratio should be 1 (MHA)
        assert_eq!(config.num_attention_heads / config.num_key_value_heads, 1);
    }

    #[test]
    fn tiny_bert_config_is_valid() {
        let config = tiny_bert_config();
        assert_eq!(config.architectures, vec!["BertModel"]);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 2);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.hidden_act, "gelu");
        assert_eq!(config.attention_bias, Some(true));
    }

    #[test]
    fn tiny_llama_not_moe() {
        let config = tiny_llama_config();
        assert!(!config.is_moe());
        assert!(!config.is_mla_model());
        assert!(!config.is_deepseek());
    }

    #[test]
    fn tiny_bert_not_moe() {
        let config = tiny_bert_config();
        assert!(!config.is_moe());
        assert!(!config.is_mla_model());
    }

    #[test]
    fn tiny_llama_default_extra_fields() {
        let config = tiny_llama_config();
        assert!(config.extra.is_empty());
        assert!(config.sliding_window.is_none());
        assert!(config.attention_bias.is_none());
    }

    #[test]
    fn tiny_bert_has_attention_bias() {
        let config = tiny_bert_config();
        assert_eq!(config.attention_bias, Some(true));
        assert!(config.sliding_window.is_none());
    }
}
