//! Multimodal input types for vision-language models.

use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// A part of multimodal content (text, image, or video).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content.
    Text { text: String },
    /// Image content.
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
    /// Video content (for video-language models).
    #[serde(rename = "video_url")]
    Video { video_url: VideoUrl },
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

    /// Create a video content part from a URL.
    pub fn video_url(url: impl Into<String>) -> Self {
        Self::Video {
            video_url: VideoUrl {
                url: url.into(),
                num_frames: None,
                fps: None,
            },
        }
    }

    /// Create a video content part from a URL with a specific frame count.
    pub fn video_url_with_frames(url: impl Into<String>, num_frames: usize) -> Self {
        Self::Video {
            video_url: VideoUrl {
                url: url.into(),
                num_frames: Some(num_frames),
                fps: None,
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

/// Video URL structure for video content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    /// URL or data URI of the video.
    pub url: String,
    /// Number of frames to sample from the video.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_frames: Option<usize>,
    /// Frames per second to sample.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f32>,
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

/// Video data that can be processed by a video encoder.
#[derive(Debug, Clone)]
pub struct VideoData {
    /// Source of the video.
    pub source: VideoSource,
    /// Number of frames to sample from the video.
    pub num_frames: Option<usize>,
    /// Frames per second to sample.
    pub fps: Option<f32>,
    /// Target size for each frame (width, height).
    pub target_size: Option<(usize, usize)>,
}

impl VideoData {
    /// Create video data from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            source: VideoSource::Url(url.into()),
            num_frames: None,
            fps: None,
            target_size: None,
        }
    }

    /// Create video data from base64-encoded bytes.
    pub fn base64(data: impl Into<String>) -> Self {
        Self {
            source: VideoSource::Base64(data.into()),
            num_frames: None,
            fps: None,
            target_size: None,
        }
    }

    /// Create video data from raw bytes.
    pub fn bytes(data: Vec<u8>) -> Self {
        Self {
            source: VideoSource::Bytes(data),
            num_frames: None,
            fps: None,
            target_size: None,
        }
    }

    /// Create video data from pre-extracted frame tensors.
    pub fn frames(frames: Vec<Tensor>) -> Self {
        Self {
            source: VideoSource::Frames(frames),
            num_frames: None,
            fps: None,
            target_size: None,
        }
    }

    /// Create video data from a pre-computed embedding tensor.
    pub fn embedding(tensor: Tensor) -> Self {
        Self {
            source: VideoSource::Embedding(tensor),
            num_frames: None,
            fps: None,
            target_size: None,
        }
    }

    /// Set the number of frames to sample.
    pub fn with_num_frames(mut self, num_frames: usize) -> Self {
        self.num_frames = Some(num_frames);
        self
    }

    /// Set the frames per second to sample.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Set the target size for each frame.
    pub fn with_target_size(mut self, width: usize, height: usize) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Check if this is a pre-computed embedding.
    pub fn is_embedding(&self) -> bool {
        matches!(self.source, VideoSource::Embedding(_))
    }
}

/// Source of video data.
#[derive(Debug, Clone)]
pub enum VideoSource {
    /// URL to fetch the video from.
    Url(String),
    /// Base64-encoded video data.
    Base64(String),
    /// Raw video bytes.
    Bytes(Vec<u8>),
    /// Pre-extracted frames as image tensors.
    Frames(Vec<Tensor>),
    /// Pre-computed video embedding tensor.
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
    /// Grid dimensions (height_patches, width_patches) after spatial merge.
    /// Used by models like Qwen2-VL for spatial position encoding.
    pub grid_size: Option<(usize, usize)>,
}

impl ProcessedImage {
    /// Create a new processed image.
    pub fn new(embedding: Tensor, num_tokens: usize) -> Self {
        Self {
            embedding,
            num_tokens,
            original_size: None,
            grid_size: None,
        }
    }

    /// Set the original image size.
    pub fn with_original_size(mut self, width: usize, height: usize) -> Self {
        self.original_size = Some((width, height));
        self
    }
}

/// A processed video ready for model input.
#[derive(Debug, Clone)]
pub struct ProcessedVideo {
    /// Video embedding tensor [num_frames * patches_per_frame, hidden_size].
    pub embedding: Tensor,
    /// Number of video tokens this embedding represents.
    pub num_tokens: usize,
    /// Number of frames extracted.
    pub num_frames: usize,
    /// Original video dimensions (width, height).
    pub original_size: Option<(usize, usize)>,
    /// Duration in seconds (if known).
    pub duration_secs: Option<f32>,
}

impl ProcessedVideo {
    /// Create a new processed video.
    pub fn new(embedding: Tensor, num_tokens: usize, num_frames: usize) -> Self {
        Self {
            embedding,
            num_tokens,
            num_frames,
            original_size: None,
            duration_secs: None,
        }
    }

    /// Set the original video dimensions.
    pub fn with_original_size(mut self, width: usize, height: usize) -> Self {
        self.original_size = Some((width, height));
        self
    }

    /// Set the video duration.
    pub fn with_duration(mut self, duration_secs: f32) -> Self {
        self.duration_secs = Some(duration_secs);
        self
    }
}

/// Multimodal inputs ready for model forward pass.
#[derive(Debug)]
pub struct MultimodalInputs {
    /// Token IDs with placeholder tokens for images/videos.
    pub token_ids: Vec<u32>,
    /// Image embeddings indexed by their position in token_ids.
    /// Key: start position in token_ids, Value: processed image.
    pub image_embeddings: Vec<(usize, ProcessedImage)>,
    /// Video embeddings indexed by their position in token_ids.
    /// Key: start position in token_ids, Value: processed video.
    pub video_embeddings: Vec<(usize, ProcessedVideo)>,
    /// Total number of image tokens.
    pub num_image_tokens: usize,
    /// Total number of video tokens.
    pub num_video_tokens: usize,
}

impl MultimodalInputs {
    /// Create inputs with only text (no images or videos).
    pub fn text_only(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            image_embeddings: Vec::new(),
            video_embeddings: Vec::new(),
            num_image_tokens: 0,
            num_video_tokens: 0,
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
            video_embeddings: Vec::new(),
            num_image_tokens,
            num_video_tokens: 0,
        }
    }

    /// Create inputs with text and videos.
    pub fn with_videos(
        token_ids: Vec<u32>,
        video_embeddings: Vec<(usize, ProcessedVideo)>,
    ) -> Self {
        let num_video_tokens = video_embeddings.iter().map(|(_, vid)| vid.num_tokens).sum();
        Self {
            token_ids,
            image_embeddings: Vec::new(),
            video_embeddings,
            num_image_tokens: 0,
            num_video_tokens,
        }
    }

    /// Create inputs with text, images, and videos.
    pub fn with_images_and_videos(
        token_ids: Vec<u32>,
        image_embeddings: Vec<(usize, ProcessedImage)>,
        video_embeddings: Vec<(usize, ProcessedVideo)>,
    ) -> Self {
        let num_image_tokens = image_embeddings.iter().map(|(_, img)| img.num_tokens).sum();
        let num_video_tokens = video_embeddings.iter().map(|(_, vid)| vid.num_tokens).sum();
        Self {
            token_ids,
            image_embeddings,
            video_embeddings,
            num_image_tokens,
            num_video_tokens,
        }
    }

    /// Check if this input has any images.
    pub fn has_images(&self) -> bool {
        !self.image_embeddings.is_empty()
    }

    /// Check if this input has any videos.
    pub fn has_videos(&self) -> bool {
        !self.video_embeddings.is_empty()
    }

    /// Get the effective sequence length (text tokens + image tokens + video tokens).
    pub fn effective_seq_len(&self) -> usize {
        // Placeholders are replaced by actual media tokens.
        // Each placeholder occupies 1 token in token_ids, replaced by num_tokens from the embedding.
        let num_placeholders = self.image_embeddings.len() + self.video_embeddings.len();
        self.token_ids.len() + self.num_image_tokens + self.num_video_tokens - num_placeholders
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

    #[test]
    fn test_content_part_video_url() {
        let part = ContentPart::video_url("https://example.com/video.mp4");
        match part {
            ContentPart::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
                assert!(video_url.num_frames.is_none());
                assert!(video_url.fps.is_none());
            }
            _ => panic!("Expected video content"),
        }
    }

    #[test]
    fn test_content_part_video_url_with_frames() {
        let part = ContentPart::video_url_with_frames("https://example.com/video.mp4", 16);
        match part {
            ContentPart::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
                assert_eq!(video_url.num_frames, Some(16));
                assert!(video_url.fps.is_none());
            }
            _ => panic!("Expected video content"),
        }
    }

    #[test]
    fn test_content_part_video_serialization() {
        let video = ContentPart::video_url("https://example.com/video.mp4");
        let json = serde_json::to_string(&video).unwrap();
        assert!(json.contains("\"type\":\"video_url\""));
        assert!(json.contains("\"url\":\"https://example.com/video.mp4\""));
    }

    #[test]
    fn test_content_part_video_deserialization() {
        let json =
            r#"{"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match part {
            ContentPart::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
                assert!(video_url.num_frames.is_none());
            }
            _ => panic!("Expected video"),
        }
    }

    #[test]
    fn test_content_part_video_deserialization_with_frames() {
        let json = r#"{"type": "video_url", "video_url": {"url": "https://example.com/video.mp4", "num_frames": 8, "fps": 2.0}}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match part {
            ContentPart::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
                assert_eq!(video_url.num_frames, Some(8));
                assert!((video_url.fps.unwrap() - 2.0).abs() < f32::EPSILON);
            }
            _ => panic!("Expected video"),
        }
    }

    #[test]
    fn test_video_data_url() {
        let data = VideoData::url("https://example.com/video.mp4");
        assert!(!data.is_embedding());
        assert!(data.num_frames.is_none());
        assert!(data.fps.is_none());
        assert!(data.target_size.is_none());
        assert!(matches!(data.source, VideoSource::Url(_)));
    }

    #[test]
    fn test_video_data_builder_methods() {
        let data = VideoData::url("https://example.com/video.mp4")
            .with_num_frames(16)
            .with_fps(2.0)
            .with_target_size(224, 224);
        assert_eq!(data.num_frames, Some(16));
        assert!((data.fps.unwrap() - 2.0).abs() < f32::EPSILON);
        assert_eq!(data.target_size, Some((224, 224)));
    }

    #[test]
    fn test_video_data_base64() {
        let data = VideoData::base64("abc123");
        assert!(matches!(data.source, VideoSource::Base64(_)));
    }

    #[test]
    fn test_video_data_bytes() {
        let data = VideoData::bytes(vec![1, 2, 3]);
        assert!(matches!(data.source, VideoSource::Bytes(_)));
    }

    #[test]
    fn test_processed_video() {
        let device = candle_core::Device::Cpu;
        let embedding = Tensor::zeros((128, 1024), candle_core::DType::F32, &device).unwrap();
        let video = ProcessedVideo::new(embedding, 128, 8)
            .with_original_size(1920, 1080)
            .with_duration(30.0);
        assert_eq!(video.num_tokens, 128);
        assert_eq!(video.num_frames, 8);
        assert_eq!(video.original_size, Some((1920, 1080)));
        assert!((video.duration_secs.unwrap() - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multimodal_inputs_with_videos() {
        let device = candle_core::Device::Cpu;
        let embedding = Tensor::zeros((64, 1024), candle_core::DType::F32, &device).unwrap();
        let video = ProcessedVideo::new(embedding, 64, 8);
        let inputs = MultimodalInputs::with_videos(vec![1, 2, 3, 4, 5], vec![(2, video)]);
        assert!(inputs.has_videos());
        assert!(!inputs.has_images());
        assert_eq!(inputs.num_video_tokens, 64);
        // effective_seq_len: 5 + 64 - 1 (placeholder) = 68
        assert_eq!(inputs.effective_seq_len(), 68);
    }

    #[test]
    fn test_multimodal_inputs_with_images_and_videos() {
        let device = candle_core::Device::Cpu;
        let img_emb = Tensor::zeros((10, 1024), candle_core::DType::F32, &device).unwrap();
        let vid_emb = Tensor::zeros((64, 1024), candle_core::DType::F32, &device).unwrap();
        let image = ProcessedImage::new(img_emb, 10);
        let video = ProcessedVideo::new(vid_emb, 64, 8);
        let inputs = MultimodalInputs::with_images_and_videos(
            vec![1, 2, 3, 4, 5, 6],
            vec![(1, image)],
            vec![(4, video)],
        );
        assert!(inputs.has_images());
        assert!(inputs.has_videos());
        assert_eq!(inputs.num_image_tokens, 10);
        assert_eq!(inputs.num_video_tokens, 64);
        // effective_seq_len: 6 + 10 + 64 - 2 (placeholders) = 78
        assert_eq!(inputs.effective_seq_len(), 78);
    }

    #[test]
    fn test_multimodal_inputs_text_only_no_videos() {
        let inputs = MultimodalInputs::text_only(vec![1, 2, 3]);
        assert!(!inputs.has_images());
        assert!(!inputs.has_videos());
        assert_eq!(inputs.num_video_tokens, 0);
        assert_eq!(inputs.effective_seq_len(), 3);
    }
}
