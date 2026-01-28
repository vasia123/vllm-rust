//! Multimodal processor for preparing inputs for vision-language models.
//!
//! The processor handles:
//! 1. Parsing content parts (text and images)
//! 2. Downloading/loading images
//! 3. Encoding images with a vision encoder
//! 4. Tokenizing text with image placeholders
//! 5. Preparing final inputs for the model

use candle_core::{DType, Device, Result, Tensor};
use thiserror::Error;

use super::inputs::{ContentPart, ImageData, ImageSource, MultimodalInputs, ProcessedImage};
use super::vision::VisionEncoder;

/// Error type for multimodal processing.
#[derive(Debug, Error)]
pub enum ProcessorError {
    #[error("Failed to load image: {0}")]
    ImageLoad(String),
    #[error("Failed to encode image: {0}")]
    ImageEncode(String),
    #[error("Failed to tokenize: {0}")]
    Tokenize(String),
    #[error("Invalid content: {0}")]
    InvalidContent(String),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Configuration for the multimodal processor.
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Image placeholder token string.
    pub image_placeholder: String,
    /// Token ID for the image placeholder.
    pub image_placeholder_id: u32,
    /// Number of tokens each image produces.
    pub num_image_tokens: usize,
    /// Mean values for image normalization (RGB).
    pub image_mean: [f32; 3],
    /// Std values for image normalization (RGB).
    pub image_std: [f32; 3],
    /// Target image size.
    pub image_size: usize,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        // Default values for CLIP ViT-L/14 @ 336px
        Self {
            image_placeholder: "<image>".to_string(),
            image_placeholder_id: 32000,
            num_image_tokens: 577, // CLIP: 24x24 patches + CLS
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.2686295, 0.2613026, 0.2757771],
            image_size: 336,
        }
    }
}

impl ProcessorConfig {
    /// Configuration for LLaVA 1.5 models.
    pub fn llava_1_5() -> Self {
        Self {
            image_placeholder: "<image>".to_string(),
            image_placeholder_id: 32000,
            num_image_tokens: 576, // 24x24 patches without CLS
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.2686295, 0.2613026, 0.2757771],
            image_size: 336,
        }
    }

    /// Configuration for Qwen-VL models.
    pub fn qwen_vl() -> Self {
        Self {
            image_placeholder: "<img>".to_string(),
            image_placeholder_id: 151655,
            num_image_tokens: 256,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.2686295, 0.2613026, 0.2757771],
            image_size: 448,
        }
    }
}

/// Multimodal processor that prepares inputs for vision-language models.
pub struct MultimodalProcessor {
    config: ProcessorConfig,
    vision_encoder: Option<VisionEncoder>,
    device: Device,
    dtype: DType,
}

impl MultimodalProcessor {
    /// Create a new processor without a vision encoder.
    ///
    /// Use this when images will be provided as pre-computed embeddings.
    pub fn new(config: ProcessorConfig, device: &Device, dtype: DType) -> Self {
        Self {
            config,
            vision_encoder: None,
            device: device.clone(),
            dtype,
        }
    }

    /// Create a new processor with a vision encoder.
    pub fn with_vision_encoder(
        config: ProcessorConfig,
        vision_encoder: VisionEncoder,
    ) -> Self {
        let device = vision_encoder.device().clone();
        let dtype = vision_encoder.dtype();
        Self {
            config,
            vision_encoder: Some(vision_encoder),
            device,
            dtype,
        }
    }

    /// Process content parts into model inputs.
    ///
    /// This method:
    /// 1. Extracts text and image content
    /// 2. Processes images through the vision encoder
    /// 3. Returns token IDs with placeholder positions and image embeddings
    pub fn process_content(
        &self,
        content: &[ContentPart],
        tokenize: impl Fn(&str) -> std::result::Result<Vec<u32>, String>,
    ) -> std::result::Result<MultimodalInputs, ProcessorError> {
        let mut text_parts = Vec::new();
        let mut images = Vec::new();

        // Collect text and images
        for part in content {
            match part {
                ContentPart::Text { text } => {
                    text_parts.push(text.clone());
                }
                ContentPart::Image { image_url } => {
                    // Insert placeholder for image
                    text_parts.push(self.config.image_placeholder.clone());
                    images.push(self.parse_image_url(&image_url.url)?);
                }
            }
        }

        // Join text with images replaced by placeholders
        let full_text = text_parts.join("");

        // Tokenize
        let token_ids = tokenize(&full_text).map_err(ProcessorError::Tokenize)?;

        // If no images, return text-only inputs
        if images.is_empty() {
            return Ok(MultimodalInputs::text_only(token_ids));
        }

        // Find placeholder positions in token_ids
        let placeholder_positions = self.find_placeholder_positions(&token_ids);

        if placeholder_positions.len() != images.len() {
            return Err(ProcessorError::InvalidContent(format!(
                "Found {} placeholders but {} images",
                placeholder_positions.len(),
                images.len()
            )));
        }

        // Process images
        let mut image_embeddings = Vec::with_capacity(images.len());
        for (pos, image) in placeholder_positions.into_iter().zip(images) {
            let processed = self.process_image(image)?;
            image_embeddings.push((pos, processed));
        }

        Ok(MultimodalInputs::with_images(token_ids, image_embeddings))
    }

    /// Process a single image into embeddings.
    pub fn process_image(&self, image: ImageData) -> std::result::Result<ProcessedImage, ProcessorError> {
        match image.source {
            ImageSource::Embedding(tensor) => {
                // Pre-computed embedding
                let num_tokens = tensor.dim(0).map_err(ProcessorError::Candle)?;
                Ok(ProcessedImage::new(tensor, num_tokens))
            }
            _ => {
                // Need to encode the image
                let encoder = self.vision_encoder.as_ref().ok_or_else(|| {
                    ProcessorError::ImageEncode("No vision encoder configured".to_string())
                })?;

                let pixel_values = self.load_and_preprocess_image(&image)?;
                encoder
                    .encode_image(&pixel_values)
                    .map_err(|e| ProcessorError::ImageEncode(e.to_string()))
            }
        }
    }

    /// Load and preprocess an image for the vision encoder.
    fn load_and_preprocess_image(&self, _image: &ImageData) -> std::result::Result<Tensor, ProcessorError> {
        // For now, create a dummy tensor as image loading requires additional dependencies
        // TODO: Integrate image loading library (e.g., image crate)
        //
        // The actual implementation would:
        // 1. Load image from URL/base64/bytes
        // 2. Resize to target size
        // 3. Convert to RGB tensor [1, 3, H, W]
        // 4. Normalize with mean/std

        let size = self.config.image_size;

        // Create placeholder tensor
        // In production, this would be the actual image data
        let pixel_values =
            Tensor::zeros((1, 3, size, size), self.dtype, &self.device)
                .map_err(ProcessorError::Candle)?;

        Ok(pixel_values)
    }

    /// Parse an image URL into ImageData.
    fn parse_image_url(&self, url: &str) -> std::result::Result<ImageData, ProcessorError> {
        if url.starts_with("data:") {
            // Data URI: data:image/png;base64,<data>
            let parts: Vec<&str> = url.splitn(2, ",").collect();
            if parts.len() != 2 {
                return Err(ProcessorError::ImageLoad("Invalid data URI".to_string()));
            }
            Ok(ImageData::base64(parts[1]))
        } else {
            // Regular URL
            Ok(ImageData::url(url))
        }
    }

    /// Find positions of image placeholder tokens in the token sequence.
    fn find_placeholder_positions(&self, token_ids: &[u32]) -> Vec<usize> {
        token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                if id == self.config.image_placeholder_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the number of tokens each image produces.
    pub fn num_image_tokens(&self) -> usize {
        self.config.num_image_tokens
    }

    /// Get the image placeholder token.
    pub fn image_placeholder(&self) -> &str {
        &self.config.image_placeholder
    }

    /// Get the image placeholder token ID.
    pub fn image_placeholder_id(&self) -> u32 {
        self.config.image_placeholder_id
    }

    /// Check if this processor has a vision encoder.
    pub fn has_vision_encoder(&self) -> bool {
        self.vision_encoder.is_some()
    }
}

/// Merge text token embeddings with image embeddings.
///
/// This function takes the token embeddings from the LLM's embedding layer
/// and replaces placeholder positions with image embeddings.
///
/// # Arguments
/// * `text_embeddings` - Token embeddings from LLM [batch, seq_len, hidden_size]
/// * `image_embeddings` - List of (position, ProcessedImage) pairs
/// * `hidden_size` - Hidden dimension of the LLM
///
/// # Returns
/// Merged embeddings with images inserted at placeholder positions
#[allow(dead_code)] // Infrastructure for model implementations
pub fn merge_embeddings(
    text_embeddings: &Tensor,
    image_embeddings: &[(usize, ProcessedImage)],
    hidden_size: usize,
) -> Result<Tensor> {
    if image_embeddings.is_empty() {
        return Ok(text_embeddings.clone());
    }

    let (batch_size, seq_len, _) = text_embeddings.dims3()?;
    let device = text_embeddings.device();
    let dtype = text_embeddings.dtype();

    // Calculate total sequence length after expansion
    let total_image_tokens: usize = image_embeddings.iter().map(|(_, img)| img.num_tokens).sum();
    let _new_seq_len = seq_len - image_embeddings.len() + total_image_tokens;

    // Build merged sequence
    // For simplicity, process batch=1 case
    // TODO: Support batch > 1
    if batch_size != 1 {
        return Err(candle_core::Error::Msg(
            "Batch size > 1 not yet supported for multimodal".to_string(),
        ));
    }

    let text_emb = text_embeddings.squeeze(0)?; // [seq_len, hidden_size]

    // Sort image positions
    let mut sorted_images = image_embeddings.to_vec();
    sorted_images.sort_by_key(|(pos, _)| *pos);

    let mut segments: Vec<Tensor> = Vec::new();
    let mut current_pos = 0;

    for (placeholder_pos, processed_image) in &sorted_images {
        // Add text segment before this image
        if *placeholder_pos > current_pos {
            let text_segment = text_emb.narrow(0, current_pos, placeholder_pos - current_pos)?;
            segments.push(text_segment);
        }

        // Add image embedding
        // Project to hidden_size if needed
        let img_emb = &processed_image.embedding;
        let img_hidden = img_emb.dim(1)?;
        let img_emb = if img_hidden != hidden_size {
            // NOTE: In production, this projection should be a learned linear layer
            // For now, we truncate or pad
            if img_hidden > hidden_size {
                img_emb.narrow(1, 0, hidden_size)?
            } else {
                let padding = Tensor::zeros(
                    (processed_image.num_tokens, hidden_size - img_hidden),
                    dtype,
                    device,
                )?;
                Tensor::cat(&[img_emb, &padding], 1)?
            }
        } else {
            img_emb.clone()
        };
        segments.push(img_emb);

        // Skip the placeholder token
        current_pos = placeholder_pos + 1;
    }

    // Add remaining text after last image
    if current_pos < seq_len {
        let text_segment = text_emb.narrow(0, current_pos, seq_len - current_pos)?;
        segments.push(text_segment);
    }

    // Concatenate all segments
    let merged = Tensor::cat(&segments, 0)?;
    merged.unsqueeze(0) // Add batch dimension back
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_config_default() {
        let cfg = ProcessorConfig::default();
        assert_eq!(cfg.image_placeholder, "<image>");
        assert_eq!(cfg.num_image_tokens, 577);
    }

    #[test]
    fn test_processor_creation() {
        let cfg = ProcessorConfig::default();
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);
        assert!(!processor.has_vision_encoder());
    }

    #[test]
    fn test_parse_image_url() {
        let cfg = ProcessorConfig::default();
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        // Regular URL
        let img = processor.parse_image_url("https://example.com/img.jpg").unwrap();
        matches!(img.source, ImageSource::Url(_));

        // Data URI
        let img = processor.parse_image_url("data:image/png;base64,abc123").unwrap();
        matches!(img.source, ImageSource::Base64(_));
    }

    #[test]
    fn test_find_placeholder_positions() {
        let cfg = ProcessorConfig {
            image_placeholder_id: 100,
            ..ProcessorConfig::default()
        };
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let tokens = vec![1, 2, 100, 3, 4, 100, 5];
        let positions = processor.find_placeholder_positions(&tokens);
        assert_eq!(positions, vec![2, 5]);
    }

    #[test]
    fn test_process_content_text_only() {
        let cfg = ProcessorConfig::default();
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let content = vec![ContentPart::text("Hello, world!")];

        let inputs = processor
            .process_content(&content, |text| Ok(text.chars().map(|c| c as u32).collect()))
            .unwrap();

        assert!(!inputs.has_images());
        assert_eq!(inputs.token_ids.len(), 13); // "Hello, world!" = 13 chars
    }

    #[test]
    fn test_merge_embeddings_no_images() {
        let device = Device::Cpu;
        let text_emb = Tensor::randn(0f32, 1.0, (1, 10, 64), &device).unwrap();

        let merged = merge_embeddings(&text_emb, &[], 64).unwrap();
        assert_eq!(merged.dims(), text_emb.dims());
    }

    #[test]
    fn test_merge_embeddings_with_image() {
        let device = Device::Cpu;

        // Text embeddings: [1, 5, 64] (5 tokens)
        let text_emb = Tensor::ones((1, 5, 64), DType::F32, &device).unwrap();

        // Image embedding at position 2, replacing 1 token with 3 image tokens
        let img_emb = Tensor::zeros((3, 64), DType::F32, &device).unwrap();
        let processed = ProcessedImage::new(img_emb, 3);

        let image_embeddings = vec![(2, processed)];

        let merged = merge_embeddings(&text_emb, &image_embeddings, 64).unwrap();

        // Expected: 5 - 1 (placeholder) + 3 (image tokens) = 7 tokens
        assert_eq!(merged.dims(), &[1, 7, 64]);
    }
}
