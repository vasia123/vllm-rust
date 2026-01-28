//! Multimodal input types for vision-language models.

use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// A part of multimodal content (text or image).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content.
    Text { text: String },
    /// Image content.
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
}

impl ContentPart {
    /// Create a text content part.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Create an image content part from a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create an image content part from base64 data.
    pub fn image_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        let data = data.into();
        let media_type = media_type.into();
        Self::Image {
            image_url: ImageUrl {
                url: format!("data:{};base64,{}", media_type, data),
                detail: None,
            },
        }
    }
}

/// Image URL structure matching OpenAI API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL or data URI of the image.
    pub url: String,
    /// Detail level for image processing (low, high, auto).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Image data that can be processed by a vision encoder.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Source of the image.
    pub source: ImageSource,
    /// Target size for resizing (width, height).
    pub target_size: Option<(usize, usize)>,
}

impl ImageData {
    /// Create image data from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Url(url.into()),
            target_size: None,
        }
    }

    /// Create image data from base64-encoded bytes.
    pub fn base64(data: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Base64(data.into()),
            target_size: None,
        }
    }

    /// Create image data from raw bytes.
    pub fn bytes(data: Vec<u8>) -> Self {
        Self {
            source: ImageSource::Bytes(data),
            target_size: None,
        }
    }

    /// Create image data from a pre-computed embedding tensor.
    pub fn embedding(tensor: Tensor) -> Self {
        Self {
            source: ImageSource::Embedding(tensor),
            target_size: None,
        }
    }

    /// Set the target size for resizing.
    pub fn with_target_size(mut self, width: usize, height: usize) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Check if this is a pre-computed embedding.
    pub fn is_embedding(&self) -> bool {
        matches!(self.source, ImageSource::Embedding(_))
    }
}

/// Source of image data.
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// URL to fetch the image from.
    Url(String),
    /// Base64-encoded image data.
    Base64(String),
    /// Raw image bytes.
    Bytes(Vec<u8>),
    /// Pre-computed embedding tensor.
    Embedding(Tensor),
}

/// A processed image ready for model input.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Image embedding tensor [num_patches, hidden_size].
    pub embedding: Tensor,
    /// Number of image tokens this embedding represents.
    pub num_tokens: usize,
    /// Original image dimensions (width, height).
    pub original_size: Option<(usize, usize)>,
}

impl ProcessedImage {
    /// Create a new processed image.
    pub fn new(embedding: Tensor, num_tokens: usize) -> Self {
        Self {
            embedding,
            num_tokens,
            original_size: None,
        }
    }

    /// Set the original image size.
    pub fn with_original_size(mut self, width: usize, height: usize) -> Self {
        self.original_size = Some((width, height));
        self
    }
}

/// Multimodal inputs ready for model forward pass.
#[derive(Debug)]
pub struct MultimodalInputs {
    /// Token IDs with placeholder tokens for images.
    pub token_ids: Vec<u32>,
    /// Image embeddings indexed by their position in token_ids.
    /// Key: start position in token_ids, Value: processed image.
    pub image_embeddings: Vec<(usize, ProcessedImage)>,
    /// Total number of image tokens.
    pub num_image_tokens: usize,
}

impl MultimodalInputs {
    /// Create inputs with only text (no images).
    pub fn text_only(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            image_embeddings: Vec::new(),
            num_image_tokens: 0,
        }
    }

    /// Create inputs with text and images.
    pub fn with_images(
        token_ids: Vec<u32>,
        image_embeddings: Vec<(usize, ProcessedImage)>,
    ) -> Self {
        let num_image_tokens = image_embeddings.iter().map(|(_, img)| img.num_tokens).sum();
        Self {
            token_ids,
            image_embeddings,
            num_image_tokens,
        }
    }

    /// Check if this input has any images.
    pub fn has_images(&self) -> bool {
        !self.image_embeddings.is_empty()
    }

    /// Get the effective sequence length (text tokens + image tokens).
    pub fn effective_seq_len(&self) -> usize {
        // Image placeholders are replaced by actual image tokens
        self.token_ids.len() + self.num_image_tokens - self.image_embeddings.len()
    }
}

/// Image placeholder token configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Infrastructure for future model implementations
pub struct ImagePlaceholderConfig {
    /// Token used as image placeholder in the prompt.
    pub placeholder_token: String,
    /// Token ID for the placeholder.
    pub placeholder_token_id: u32,
    /// Number of tokens each image is converted to.
    pub tokens_per_image: usize,
}

impl Default for ImagePlaceholderConfig {
    fn default() -> Self {
        Self {
            placeholder_token: "<image>".to_string(),
            placeholder_token_id: 32000, // Common default, model-specific
            tokens_per_image: 576,       // CLIP ViT-L/14 @ 336px: 24x24 patches
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_part_text() {
        let part = ContentPart::text("Hello, world!");
        match part {
            ContentPart::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_content_part_image_url() {
        let part = ContentPart::image_url("https://example.com/image.jpg");
        match part {
            ContentPart::Image { image_url } => {
                assert_eq!(image_url.url, "https://example.com/image.jpg");
            }
            _ => panic!("Expected image content"),
        }
    }

    #[test]
    fn test_content_part_image_base64() {
        let part = ContentPart::image_base64("abc123", "image/png");
        match part {
            ContentPart::Image { image_url } => {
                assert_eq!(image_url.url, "data:image/png;base64,abc123");
            }
            _ => panic!("Expected image content"),
        }
    }

    #[test]
    fn test_image_data_url() {
        let data = ImageData::url("https://example.com/image.jpg");
        assert!(!data.is_embedding());
        matches!(data.source, ImageSource::Url(_));
    }

    #[test]
    fn test_multimodal_inputs_text_only() {
        let inputs = MultimodalInputs::text_only(vec![1, 2, 3, 4, 5]);
        assert!(!inputs.has_images());
        assert_eq!(inputs.effective_seq_len(), 5);
    }

    #[test]
    fn test_content_part_serialization() {
        let text = ContentPart::text("Hello");
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"text\""));

        let image = ContentPart::image_url("https://example.com/img.jpg");
        let json = serde_json::to_string(&image).unwrap();
        assert!(json.contains("\"type\":\"image_url\""));
    }

    #[test]
    fn test_content_part_deserialization() {
        let json = r#"{"type": "text", "text": "Hello"}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match part {
            ContentPart::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected text"),
        }

        let json = r#"{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match part {
            ContentPart::Image { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.jpg")
            }
            _ => panic!("Expected image"),
        }
    }
}
