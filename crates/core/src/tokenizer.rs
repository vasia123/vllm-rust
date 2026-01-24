use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct TokenizerWrapper {
    inner: Tokenizer,
}

impl TokenizerWrapper {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner =
            Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        Ok(Self { inner })
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn for_testing(vocab_size: usize) -> Self {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let mut vocab = ahash::AHashMap::new();
        for i in 0..vocab_size {
            vocab.insert(format!("t{i}"), i as u32);
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("t0".into())
            .build()
            .expect("build test tokenizer model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        Self { inner: tokenizer }
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

// ─── Chat Template ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplateEngine {
    template_source: String,
    bos_token: String,
    eos_token: String,
}

#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    #[serde(default)]
    bos_token: SpecialToken,
    #[serde(default)]
    eos_token: SpecialToken,
}

#[derive(Deserialize, Default)]
#[serde(untagged)]
enum SpecialToken {
    Plain(String),
    Dict {
        content: String,
    },
    #[default]
    None,
}

impl SpecialToken {
    fn as_str(&self) -> &str {
        match self {
            SpecialToken::Plain(s) => s,
            SpecialToken::Dict { content } => content,
            SpecialToken::None => "",
        }
    }
}

impl ChatTemplateEngine {
    pub fn new(template_source: String, bos_token: String, eos_token: String) -> Self {
        Self {
            template_source,
            bos_token,
            eos_token,
        }
    }

    pub fn from_tokenizer_config(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: TokenizerConfig = serde_json::from_str(&content)?;
        let template_source = config
            .chat_template
            .ok_or_else(|| anyhow::anyhow!("no chat_template field in tokenizer_config.json"))?;
        Ok(Self {
            template_source,
            bos_token: config.bos_token.as_str().to_string(),
            eos_token: config.eos_token.as_str().to_string(),
        })
    }

    pub fn apply(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        let mut env = minijinja::Environment::new();
        env.add_template("chat", &self.template_source)?;
        let tmpl = env.get_template("chat")?;
        let rendered = tmpl.render(minijinja::context! {
            messages => messages,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => add_generation_prompt,
        })?;
        Ok(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader;

    #[test]
    #[ignore] // requires downloaded model
    fn encode_decode_roundtrip() {
        let files = loader::fetch_model("Qwen/Qwen3-0.6B").expect("fetch model");
        let tok = TokenizerWrapper::from_file(&files.tokenizer).expect("load tokenizer");

        let text = "Hello, world!";
        let ids = tok.encode(text).expect("encode");
        assert!(!ids.is_empty());

        let decoded = tok.decode(&ids).expect("decode");
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore] // requires downloaded model
    fn encode_produces_expected_token_count() {
        let files = loader::fetch_model("Qwen/Qwen3-0.6B").expect("fetch model");
        let tok = TokenizerWrapper::from_file(&files.tokenizer).expect("load tokenizer");

        let ids = tok.encode("The quick brown fox").expect("encode");
        // Qwen3 BPE: "The" "quick" "brown" "fox" → ~4 tokens (may vary with BPE merges)
        assert!(
            ids.len() >= 3 && ids.len() <= 6,
            "unexpected token count: {}",
            ids.len()
        );
    }

    // ─── Chat Template tests ─────────────────────────────────────────────

    const CHATML_TEMPLATE: &str = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

    #[test]
    fn chat_template_single_turn() {
        let engine = ChatTemplateEngine::new(
            CHATML_TEMPLATE.to_string(),
            "<|begin|>".to_string(),
            "<|end|>".to_string(),
        );
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = engine.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn chat_template_multi_turn() {
        let engine =
            ChatTemplateEngine::new(CHATML_TEMPLATE.to_string(), "".to_string(), "".to_string());
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "And 3+3?".to_string(),
            },
        ];
        let result = engine.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n4<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nAnd 3+3?<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chat_template_no_generation_prompt() {
        let engine =
            ChatTemplateEngine::new(CHATML_TEMPLATE.to_string(), "".to_string(), "".to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let result = engine.apply(&messages, false).unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn chat_template_with_special_tokens() {
        let template = r#"{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{{ eos_token }}"#;
        let engine =
            ChatTemplateEngine::new(template.to_string(), "<s>".to_string(), "</s>".to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Test".to_string(),
        }];
        let result = engine.apply(&messages, false).unwrap();
        assert!(result.starts_with("<s>"));
        assert!(result.ends_with("</s>"));
    }

    #[test]
    fn chat_template_from_json_config() {
        let config_json = r#"{
            "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}",
            "bos_token": "<s>",
            "eos_token": "</s>"
        }"#;
        let dir = std::env::temp_dir().join("vllm_test_chat_template");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer_config.json");
        std::fs::write(&path, config_json).unwrap();

        let engine = ChatTemplateEngine::from_tokenizer_config(&path).unwrap();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = engine.apply(&messages, false).unwrap();
        assert!(result.contains("user: Hello"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn chat_template_from_json_dict_tokens() {
        let config_json = r#"{
            "chat_template": "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}",
            "bos_token": {"content": "<bos>", "lstrip": false, "rstrip": false},
            "eos_token": {"content": "<eos>", "lstrip": false, "rstrip": false}
        }"#;
        let dir = std::env::temp_dir().join("vllm_test_chat_template_dict");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer_config.json");
        std::fs::write(&path, config_json).unwrap();

        let engine = ChatTemplateEngine::from_tokenizer_config(&path).unwrap();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let result = engine.apply(&messages, false).unwrap();
        assert!(result.starts_with("<bos>"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore] // requires downloaded model
    fn chat_template_qwen3_real() {
        let files = loader::fetch_model("Qwen/Qwen3-0.6B").expect("fetch model");
        let config_path = files
            .tokenizer_config
            .expect("should have tokenizer_config.json");
        let engine =
            ChatTemplateEngine::from_tokenizer_config(&config_path).expect("load template");

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];
        let result = engine.apply(&messages, true).unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("Hello!"));
    }
}
