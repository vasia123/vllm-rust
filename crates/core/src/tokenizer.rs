use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::multimodal::ContentPart;

pub struct TokenizerWrapper {
    inner: Tokenizer,
    /// Maximum character length of any single token in the vocabulary.
    /// Used for early-fail validation: a text of C characters produces
    /// at least ceil(C / max_chars_per_token) tokens.
    max_chars_per_token: usize,
}

impl TokenizerWrapper {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner =
            Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let max_chars_per_token = inner
            .get_vocab(true)
            .keys()
            .map(|k| k.len())
            .max()
            .unwrap_or(1);
        Ok(Self {
            inner,
            max_chars_per_token,
        })
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
        // Test tokens are "t0".."tN" — max length is the longest token string
        let max_chars_per_token = format!("t{}", vocab_size.saturating_sub(1)).len();
        Self {
            inner: tokenizer,
            max_chars_per_token,
        }
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

    /// Maximum character length of any single token in the vocabulary.
    pub fn max_chars_per_token(&self) -> usize {
        self.max_chars_per_token
    }
}

// ─── Chat Template ────────────────────────────────────────────────────────

/// Message content that can be either text or multimodal (text + images).
///
/// This type supports the OpenAI API format where content can be:
/// - A simple string: `"content": "Hello"`
/// - An array of content parts: `"content": [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content (most common case).
    Text(String),
    /// Array of content parts for multimodal messages.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Get the text content, joining all text parts if multimodal.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Check if this content has any images.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => {
                parts.iter().any(|p| matches!(p, ContentPart::Image { .. }))
            }
        }
    }

    /// Get all image URLs from this content.
    pub fn image_urls(&self) -> Vec<&str> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Image { image_url } => Some(image_url.url.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Convert to content parts (wraps string in a Text part).
    pub fn into_parts(self) -> Vec<ContentPart> {
        match self {
            MessageContent::Text(s) => vec![ContentPart::text(s)],
            MessageContent::Parts(parts) => parts,
        }
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    /// Reasoning/thinking content for interleaved thinking models (e.g. DeepSeek R1, Qwen3).
    /// Also accepts the deprecated `reasoning_content` field name.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_content"
    )]
    pub reasoning: Option<String>,
}

impl ChatMessage {
    /// Create a new text-only chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(content.into()),
            reasoning: None,
        }
    }

    /// Create a new multimodal chat message.
    pub fn multimodal(role: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Parts(parts),
            reasoning: None,
        }
    }

    /// Get the text content of this message.
    pub fn text(&self) -> String {
        self.content.as_text()
    }

    /// Check if this message has images.
    pub fn has_images(&self) -> bool {
        self.content.has_images()
    }
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
        self.apply_with_tools(messages, None, add_generation_prompt)
    }

    /// Apply the chat template with optional tool definitions.
    ///
    /// Many chat templates support a `tools` variable that includes function/tool
    /// definitions for function calling models.
    pub fn apply_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[crate::tool_parser::ToolDefinition]>,
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        let mut env = minijinja::Environment::new();
        // Add Python-compatible string methods (startswith, endswith, etc.)
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.template_source)?;
        let tmpl = env.get_template("chat")?;

        // Convert tools to JSON-serializable format
        let tools_json: Option<Vec<serde_json::Value>> = tools.map(|t| {
            t.iter()
                .filter_map(|tool| serde_json::to_value(tool).ok())
                .collect()
        });

        let rendered = tmpl.render(minijinja::context! {
            messages => messages,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => add_generation_prompt,
            tools => tools_json,
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
        let messages = vec![ChatMessage::new("user", "Hello")];
        let result = engine.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn chat_template_multi_turn() {
        let engine =
            ChatTemplateEngine::new(CHATML_TEMPLATE.to_string(), "".to_string(), "".to_string());
        let messages = vec![
            ChatMessage::new("user", "What is 2+2?"),
            ChatMessage::new("assistant", "4"),
            ChatMessage::new("user", "And 3+3?"),
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
        let messages = vec![ChatMessage::new("user", "Hi")];
        let result = engine.apply(&messages, false).unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn chat_template_with_special_tokens() {
        let template = r#"{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{{ eos_token }}"#;
        let engine =
            ChatTemplateEngine::new(template.to_string(), "<s>".to_string(), "</s>".to_string());
        let messages = vec![ChatMessage::new("user", "Test")];
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
        let messages = vec![ChatMessage::new("user", "Hello")];
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
        let messages = vec![ChatMessage::new("user", "Hi")];
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

        let messages = vec![ChatMessage::new("user", "Hello!")];
        let result = engine.apply(&messages, true).unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("Hello!"));
    }

    #[test]
    fn test_max_chars_per_token() {
        let tok = TokenizerWrapper::for_testing(100);
        // Tokens are "t0".."t99", longest is "t99" (3 chars)
        assert_eq!(tok.max_chars_per_token(), 3);
    }

    #[test]
    fn test_max_chars_per_token_small_vocab() {
        let tok = TokenizerWrapper::for_testing(5);
        // Tokens are "t0".."t4", longest is "t4" (2 chars)
        assert_eq!(tok.max_chars_per_token(), 2);
    }

    #[test]
    fn test_message_content_text() {
        let content = MessageContent::Text("Hello".to_string());
        assert_eq!(content.as_text(), "Hello");
        assert!(!content.has_images());
    }

    #[test]
    fn test_message_content_multimodal() {
        let content = MessageContent::Parts(vec![
            ContentPart::text("What is in this image?"),
            ContentPart::image_url("https://example.com/img.jpg"),
        ]);
        assert_eq!(content.as_text(), "What is in this image?");
        assert!(content.has_images());
        assert_eq!(content.image_urls(), vec!["https://example.com/img.jpg"]);
    }

    #[test]
    fn test_chat_message_new() {
        let msg = ChatMessage::new("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.text(), "Hello");
        assert!(!msg.has_images());
    }

    #[test]
    fn test_chat_message_multimodal() {
        let msg = ChatMessage::multimodal(
            "user",
            vec![
                ContentPart::text("Describe this:"),
                ContentPart::image_url("https://example.com/img.jpg"),
            ],
        );
        assert_eq!(msg.role, "user");
        assert!(msg.has_images());
    }

    #[test]
    fn test_chat_message_reasoning_field() {
        let json = r#"{"role": "assistant", "content": "The answer is 4.", "reasoning": "2+2=4"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.reasoning.as_deref(), Some("2+2=4"));
    }

    #[test]
    fn test_chat_message_reasoning_content_alias() {
        // Deprecated field name should be deserialized into `reasoning`
        let json =
            r#"{"role": "assistant", "content": "Yes.", "reasoning_content": "Let me think..."}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.reasoning.as_deref(), Some("Let me think..."));
    }

    #[test]
    fn test_chat_message_reasoning_defaults_to_none() {
        let json = r#"{"role": "user", "content": "Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(msg.reasoning.is_none());
    }

    #[test]
    fn test_chat_message_reasoning_serialization_omits_none() {
        let msg = ChatMessage::new("user", "Hello");
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("reasoning").is_none());
    }

    #[test]
    fn test_message_content_deserialization() {
        // Text-only (string)
        let json = r#"{"role": "user", "content": "Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.text(), "Hello");

        // Multimodal (array)
        let json = r#"{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}]}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.text(), "Hi");
        assert!(msg.has_images());
    }
}
