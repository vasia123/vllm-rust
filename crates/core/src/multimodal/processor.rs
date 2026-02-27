//! Multimodal processor for preparing inputs for vision-language models.
//!
//! The processor handles:
//! 1. Parsing content parts (text and images)
//! 2. Downloading/loading images
//! 3. Encoding images with a vision encoder
//! 4. Tokenizing text with image placeholders
//! 5. Preparing final inputs for the model

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor};
use thiserror::Error;

use super::inputs::{
    ContentPart, ImageData, ImageSource, MultimodalInputs, ProcessedImage, ProcessedVideo,
    VideoData, VideoSource,
};
use super::preprocessor_cache::{PreprocessorCache, DEFAULT_CACHE_CAPACITY};
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
    /// Video placeholder token string.
    pub video_placeholder: String,
    /// Token ID for the video placeholder.
    pub video_placeholder_id: u32,
    /// Default number of tokens each video produces (num_frames * tokens_per_frame).
    pub num_video_tokens: usize,
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
            video_placeholder: "<video>".to_string(),
            video_placeholder_id: 32001,
            num_video_tokens: 2048, // 8 frames × 256 patches
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
            video_placeholder: "<video>".to_string(),
            video_placeholder_id: 32001,
            num_video_tokens: 2048,
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
            video_placeholder: "<video>".to_string(),
            video_placeholder_id: 151656,
            num_video_tokens: 2048,
        }
    }
}

/// Multimodal processor that prepares inputs for vision-language models.
pub struct MultimodalProcessor {
    config: ProcessorConfig,
    vision_encoder: Option<VisionEncoder>,
    device: Device,
    dtype: DType,
    /// Optional LRU cache for vision encoder outputs.
    ///
    /// `None` when the cache is disabled via `--disable-mm-preprocessor-cache`.
    /// Wrapped in `Arc<Mutex<>>` so the processor can be shared across threads
    /// (e.g., multiple request handlers) while still mutating the cache.
    preprocessor_cache: Option<Arc<Mutex<PreprocessorCache>>>,
}

impl MultimodalProcessor {
    /// Create a new processor without a vision encoder.
    ///
    /// Use this when images will be provided as pre-computed embeddings.
    /// The preprocessor cache is enabled by default with `DEFAULT_CACHE_CAPACITY`
    /// entries. Call `.without_preprocessor_cache()` to disable it.
    pub fn new(config: ProcessorConfig, device: &Device, dtype: DType) -> Self {
        Self {
            config,
            vision_encoder: None,
            device: device.clone(),
            dtype,
            preprocessor_cache: Some(Arc::new(Mutex::new(PreprocessorCache::new(
                DEFAULT_CACHE_CAPACITY,
            )))),
        }
    }

    /// Create a new processor with a vision encoder.
    ///
    /// The preprocessor cache is enabled by default with `DEFAULT_CACHE_CAPACITY`
    /// entries. Call `.without_preprocessor_cache()` to disable it.
    pub fn with_vision_encoder(config: ProcessorConfig, vision_encoder: VisionEncoder) -> Self {
        let device = vision_encoder.device().clone();
        let dtype = vision_encoder.dtype();
        Self {
            config,
            vision_encoder: Some(vision_encoder),
            device,
            dtype,
            preprocessor_cache: Some(Arc::new(Mutex::new(PreprocessorCache::new(
                DEFAULT_CACHE_CAPACITY,
            )))),
        }
    }

    /// Disable the preprocessor cache for this processor instance.
    ///
    /// Called when `--disable-mm-preprocessor-cache` is set. Subsequent
    /// `process_image` calls will always invoke the vision encoder.
    pub fn without_preprocessor_cache(mut self) -> Self {
        self.preprocessor_cache = None;
        self
    }

    /// Process content parts into model inputs.
    ///
    /// This method:
    /// 1. Extracts text, image, and video content
    /// 2. Processes images/videos through the vision encoder
    /// 3. Returns token IDs with placeholder positions and media embeddings
    pub fn process_content(
        &self,
        content: &[ContentPart],
        tokenize: impl Fn(&str) -> std::result::Result<Vec<u32>, String>,
    ) -> std::result::Result<MultimodalInputs, ProcessorError> {
        let mut text_parts = Vec::new();
        let mut images = Vec::new();
        let mut videos = Vec::new();

        // Collect text, images, and videos
        for part in content {
            match part {
                ContentPart::Text { text } => {
                    text_parts.push(text.clone());
                }
                ContentPart::Image { image_url } => {
                    text_parts.push(self.config.image_placeholder.clone());
                    images.push(self.parse_image_url(&image_url.url)?);
                }
                ContentPart::Video { video_url } => {
                    text_parts.push(self.config.video_placeholder.clone());
                    videos.push(self.parse_video_url(video_url)?);
                }
            }
        }

        // Join text with media replaced by placeholders
        let full_text = text_parts.join("");

        // Tokenize
        let token_ids = tokenize(&full_text).map_err(ProcessorError::Tokenize)?;

        // Text-only fast path
        if images.is_empty() && videos.is_empty() {
            return Ok(MultimodalInputs::text_only(token_ids));
        }

        // Process images
        let mut image_embeddings = Vec::with_capacity(images.len());
        if !images.is_empty() {
            let image_positions = self.find_placeholder_positions(&token_ids);
            if image_positions.len() != images.len() {
                return Err(ProcessorError::InvalidContent(format!(
                    "Found {} image placeholders but {} images",
                    image_positions.len(),
                    images.len()
                )));
            }
            for (pos, image) in image_positions.into_iter().zip(images) {
                let processed = self.process_image(image)?;
                image_embeddings.push((pos, processed));
            }
        }

        // Process videos
        let mut video_embeddings = Vec::with_capacity(videos.len());
        if !videos.is_empty() {
            let video_positions = self.find_video_placeholder_positions(&token_ids);
            if video_positions.len() != videos.len() {
                return Err(ProcessorError::InvalidContent(format!(
                    "Found {} video placeholders but {} videos",
                    video_positions.len(),
                    videos.len()
                )));
            }
            for (pos, video) in video_positions.into_iter().zip(videos) {
                let processed = self.process_video(video)?;
                video_embeddings.push((pos, processed));
            }
        }

        if video_embeddings.is_empty() {
            Ok(MultimodalInputs::with_images(token_ids, image_embeddings))
        } else if image_embeddings.is_empty() {
            Ok(MultimodalInputs::with_videos(token_ids, video_embeddings))
        } else {
            Ok(MultimodalInputs::with_images_and_videos(
                token_ids,
                image_embeddings,
                video_embeddings,
            ))
        }
    }

    /// Process a single image into embeddings.
    ///
    /// Pre-computed `ImageSource::Embedding` values bypass the encoder and
    /// cache entirely. For all other sources the preprocessor cache is checked
    /// first; a hit returns a clone of the cached `ProcessedImage` without
    /// calling the vision encoder.
    pub fn process_image(
        &self,
        image: ImageData,
    ) -> std::result::Result<ProcessedImage, ProcessorError> {
        // Fast path: already encoded, no need for the vision encoder or cache.
        if let ImageSource::Embedding(tensor) = image.source {
            let num_tokens = tensor.dim(0).map_err(ProcessorError::Candle)?;
            return Ok(ProcessedImage::new(tensor, num_tokens));
        }

        let encoder = self.vision_encoder.as_ref().ok_or_else(|| {
            ProcessorError::ImageEncode("No vision encoder configured".to_string())
        })?;

        // Cache-enabled path: check for a hit before encoding.
        if let Some(cache) = &self.preprocessor_cache {
            if let Some(key) = PreprocessorCache::compute_key(&image) {
                {
                    let mut guard = cache.lock().unwrap();
                    if let Some(cached) = guard.get(&key) {
                        return Ok((*cached).clone());
                    }
                } // release lock before encoding

                let pixel_values = self.load_and_preprocess_image(&image)?;
                let encoded = encoder
                    .encode_image(&pixel_values)
                    .map_err(|e| ProcessorError::ImageEncode(e.to_string()))?;

                cache.lock().unwrap().insert(key, encoded.clone());
                return Ok(encoded);
            }
        }

        // Cache disabled or source has no cacheable key — encode unconditionally.
        let pixel_values = self.load_and_preprocess_image(&image)?;
        encoder
            .encode_image(&pixel_values)
            .map_err(|e| ProcessorError::ImageEncode(e.to_string()))
    }

    /// Load and preprocess an image for the vision encoder.
    ///
    /// When the `image-loading` feature is enabled, this loads actual image data.
    /// Otherwise, it returns a placeholder tensor.
    #[cfg(feature = "image-loading")]
    fn load_and_preprocess_image(
        &self,
        image: &ImageData,
    ) -> std::result::Result<Tensor, ProcessorError> {
        let size = self.config.image_size;

        // Load image bytes based on source
        let img = match &image.source {
            ImageSource::Base64(data) => {
                let bytes =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                        .map_err(|e| ProcessorError::ImageLoad(format!("base64 decode: {e}")))?;
                image::load_from_memory(&bytes)
                    .map_err(|e| ProcessorError::ImageLoad(format!("image load: {e}")))?
            }
            ImageSource::Bytes(bytes) => image::load_from_memory(bytes)
                .map_err(|e| ProcessorError::ImageLoad(format!("image load: {e}")))?,
            ImageSource::Url(_url) => {
                // NOTE: URL loading requires an HTTP client (e.g., reqwest) which adds
                // significant dependencies. For now, URLs should be pre-fetched by the
                // caller and provided as bytes or base64. This is consistent with
                // many inference frameworks that expect pre-fetched image data.
                return Err(ProcessorError::ImageLoad(
                    "URL image loading not supported. Pre-fetch the image and provide as bytes."
                        .to_string(),
                ));
            }
            ImageSource::Embedding(_) => {
                // Already an embedding, shouldn't reach here
                return Err(ProcessorError::InvalidContent(
                    "Cannot preprocess embedding as image".to_string(),
                ));
            }
        };

        // Convert to RGB
        let img = img.to_rgb8();

        // Resize to target size
        let target_size = image.target_size.unwrap_or((size, size));
        let img = image::imageops::resize(
            &img,
            target_size.0 as u32,
            target_size.1 as u32,
            image::imageops::FilterType::Triangle,
        );

        // Convert to tensor [1, 3, H, W]
        let (width, height) = (target_size.0, target_size.1);
        let mut pixel_data = vec![0f32; 3 * height * width];

        for (x, y, pixel) in img.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b] = pixel.0;

            // Normalize to [0, 1] then apply mean/std normalization
            let r = (r as f32 / 255.0 - self.config.image_mean[0]) / self.config.image_std[0];
            let g = (g as f32 / 255.0 - self.config.image_mean[1]) / self.config.image_std[1];
            let b = (b as f32 / 255.0 - self.config.image_mean[2]) / self.config.image_std[2];

            // Store in CHW format (channel, height, width)
            let hw = height * width;
            pixel_data[y * width + x] = r; // R channel
            pixel_data[hw + y * width + x] = g; // G channel
            pixel_data[2 * hw + y * width + x] = b; // B channel
        }

        let pixel_values = Tensor::from_vec(pixel_data, (1, 3, height, width), &self.device)
            .map_err(ProcessorError::Candle)?
            .to_dtype(self.dtype)
            .map_err(ProcessorError::Candle)?;

        Ok(pixel_values)
    }

    /// Load and preprocess an image for the vision encoder.
    ///
    /// This is a placeholder implementation when `image-loading` feature is disabled.
    /// Returns a dummy tensor of the correct shape.
    ///
    /// NOTE: Enable the `image-loading` feature to get actual image loading support.
    /// Without it, this returns zeros which is only suitable for testing.
    #[cfg(not(feature = "image-loading"))]
    fn load_and_preprocess_image(
        &self,
        _image: &ImageData,
    ) -> std::result::Result<Tensor, ProcessorError> {
        // NOTE: Enable `image-loading` feature for real image processing
        let size = self.config.image_size;

        let pixel_values = Tensor::zeros((1, 3, size, size), self.dtype, &self.device)
            .map_err(ProcessorError::Candle)?;

        Ok(pixel_values)
    }

    /// Process a video into frame embeddings.
    ///
    /// Handles two cases:
    /// - `VideoSource::Embedding` — returned as-is (pre-computed by the caller).
    /// - `VideoSource::Frames` — each frame is encoded with the vision encoder and
    ///   the per-frame embeddings are concatenated along the token dimension.
    /// - All other sources (URL/Base64/Bytes) — not supported without a video codec.
    pub fn process_video(
        &self,
        video: VideoData,
    ) -> std::result::Result<ProcessedVideo, ProcessorError> {
        match video.source {
            VideoSource::Embedding(tensor) => {
                let num_tokens = tensor.dim(0).map_err(ProcessorError::Candle)?;
                let num_frames = video.num_frames.unwrap_or(1);
                Ok(ProcessedVideo::new(tensor, num_tokens, num_frames))
            }
            VideoSource::Frames(frames) => {
                let encoder = self.vision_encoder.as_ref().ok_or_else(|| {
                    ProcessorError::ImageEncode(
                        "vision encoder required to encode video frames".to_string(),
                    )
                })?;
                let num_frames = frames.len();
                if num_frames == 0 {
                    return Err(ProcessorError::InvalidContent(
                        "video has no frames".to_string(),
                    ));
                }
                let mut frame_embeddings = Vec::with_capacity(num_frames);
                for frame in &frames {
                    let processed = encoder
                        .encode_image(frame)
                        .map_err(|e| ProcessorError::ImageEncode(e.to_string()))?;
                    frame_embeddings.push(processed.embedding);
                }
                let all_embeddings =
                    Tensor::cat(&frame_embeddings, 0).map_err(ProcessorError::Candle)?;
                let num_tokens = all_embeddings.dim(0).map_err(ProcessorError::Candle)?;
                Ok(ProcessedVideo::new(all_embeddings, num_tokens, num_frames))
            }
            VideoSource::Url(_) | VideoSource::Base64(_) | VideoSource::Bytes(_) => {
                // Video decoding requires a codec library (e.g., ffmpeg). Callers should
                // pre-extract frames (VideoSource::Frames) or pre-compute embeddings
                // (VideoSource::Embedding) before calling this method.
                Err(ProcessorError::InvalidContent(
                    "video decoding from URL/bytes is not supported; \
                     provide pre-extracted frames or a pre-computed embedding"
                        .to_string(),
                ))
            }
        }
    }

    /// Parse a `VideoUrl` into `VideoData`.
    fn parse_video_url(
        &self,
        video_url: &super::inputs::VideoUrl,
    ) -> std::result::Result<VideoData, ProcessorError> {
        let mut data = if video_url.url.starts_with("data:") {
            let parts: Vec<&str> = video_url.url.splitn(2, ',').collect();
            if parts.len() != 2 {
                return Err(ProcessorError::ImageLoad("Invalid data URI".to_string()));
            }
            VideoData::base64(parts[1])
        } else {
            VideoData::url(&video_url.url)
        };
        if let Some(n) = video_url.num_frames {
            data = data.with_num_frames(n);
        }
        if let Some(fps) = video_url.fps {
            data = data.with_fps(fps);
        }
        Ok(data)
    }

    /// Find positions of video placeholder tokens in the token sequence.
    fn find_video_placeholder_positions(&self, token_ids: &[u32]) -> Vec<usize> {
        token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                if id == self.config.video_placeholder_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
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

    /// Get the video placeholder token.
    pub fn video_placeholder(&self) -> &str {
        &self.config.video_placeholder
    }

    /// Get the video placeholder token ID.
    pub fn video_placeholder_id(&self) -> u32 {
        self.config.video_placeholder_id
    }

    /// Get the default number of video tokens.
    pub fn num_video_tokens(&self) -> usize {
        self.config.num_video_tokens
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
/// Merged embeddings with images inserted at placeholder positions.
/// For batch > 1, the image_embeddings should contain positions that are
/// within each batch item's sequence (i.e., positions are local to each item).
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

    if batch_size == 1 {
        // Fast path for single-batch case
        merge_embeddings_single(text_embeddings, image_embeddings, hidden_size, seq_len)
    } else {
        // Process each batch item separately, then stack
        // For batched multimodal, we assume all images belong to the first batch item.
        // A more general approach would require per-batch-item image lists.
        merge_embeddings_single(text_embeddings, image_embeddings, hidden_size, seq_len)
    }
}

/// Merge embeddings for a single batch item (or treat batch as single sequence).
fn merge_embeddings_single(
    text_embeddings: &Tensor,
    image_embeddings: &[(usize, ProcessedImage)],
    hidden_size: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let (batch_size, _, _) = text_embeddings.dims3()?;
    let device = text_embeddings.device();
    let dtype = text_embeddings.dtype();

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

    // Restore batch dimension(s)
    if batch_size == 1 {
        merged.unsqueeze(0)
    } else {
        // For batch > 1, the input was squeezed to [batch * seq_len, hidden_size]
        // We need to reshape back to [batch, new_seq_len, hidden_size]
        let new_seq_len = merged.dim(0)?;
        merged
            .unsqueeze(0)?
            .broadcast_as((batch_size, new_seq_len, hidden_size))?
            .contiguous()
    }
}

/// Merge embeddings for batched inputs with per-batch image positions.
///
/// This function handles the case where different batch items have images
/// at different positions. Each batch item is processed separately.
///
/// # Arguments
/// * `text_embeddings` - Token embeddings [batch, seq_len, hidden_size]
/// * `batch_image_embeddings` - Per-batch-item image embeddings
/// * `hidden_size` - Hidden dimension of the LLM
///
/// # Returns
/// Merged embeddings. All sequences are padded to the longest sequence length.
#[allow(dead_code)] // Infrastructure for future use
pub fn merge_embeddings_batched(
    text_embeddings: &Tensor,
    batch_image_embeddings: &[Vec<(usize, ProcessedImage)>],
    hidden_size: usize,
) -> Result<Tensor> {
    let (batch_size, seq_len, _) = text_embeddings.dims3()?;
    let device = text_embeddings.device();
    let dtype = text_embeddings.dtype();

    if batch_image_embeddings.len() != batch_size {
        return Err(candle_core::Error::Msg(format!(
            "Batch size mismatch: {} embeddings vs {} image lists",
            batch_size,
            batch_image_embeddings.len()
        )));
    }

    // Process each batch item
    let mut merged_items: Vec<Tensor> = Vec::with_capacity(batch_size);
    let mut max_merged_len = 0;

    for (i, image_embeddings) in batch_image_embeddings.iter().enumerate() {
        let text_item = text_embeddings.narrow(0, i, 1)?;

        if image_embeddings.is_empty() {
            // No images for this batch item
            merged_items.push(text_item.squeeze(0)?);
            if seq_len > max_merged_len {
                max_merged_len = seq_len;
            }
        } else {
            let merged =
                merge_embeddings_single(&text_item, image_embeddings, hidden_size, seq_len)?;
            let merged = merged.squeeze(0)?;
            let merged_len = merged.dim(0)?;
            if merged_len > max_merged_len {
                max_merged_len = merged_len;
            }
            merged_items.push(merged);
        }
    }

    // Pad all sequences to max_merged_len
    let mut padded_items: Vec<Tensor> = Vec::with_capacity(batch_size);
    for item in merged_items {
        let item_len = item.dim(0)?;
        if item_len < max_merged_len {
            let padding = Tensor::zeros((max_merged_len - item_len, hidden_size), dtype, device)?;
            padded_items.push(Tensor::cat(&[&item, &padding], 0)?);
        } else {
            padded_items.push(item);
        }
    }

    // Stack into batch dimension
    Tensor::stack(&padded_items, 0)
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
        let img = processor
            .parse_image_url("https://example.com/img.jpg")
            .unwrap();
        matches!(img.source, ImageSource::Url(_));

        // Data URI
        let img = processor
            .parse_image_url("data:image/png;base64,abc123")
            .unwrap();
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
            .process_content(&content, |text| {
                Ok(text.chars().map(|c| c as u32).collect())
            })
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
    fn test_processor_video_config() {
        let cfg = ProcessorConfig::default();
        assert_eq!(cfg.video_placeholder, "<video>");
        assert_eq!(cfg.video_placeholder_id, 32001);
        assert_eq!(cfg.num_video_tokens, 2048);
    }

    #[test]
    fn test_process_video_embedding() {
        let cfg = ProcessorConfig::default();
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let embedding = Tensor::zeros((64, 512), DType::F32, &device).unwrap();
        let video = VideoData::embedding(embedding).with_num_frames(8);
        let processed = processor.process_video(video).unwrap();

        assert_eq!(processed.num_tokens, 64);
        assert_eq!(processed.num_frames, 8);
        assert_eq!(processed.embedding.dims(), &[64, 512]);
    }

    #[test]
    fn test_process_video_url_not_supported() {
        let cfg = ProcessorConfig::default();
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let video = VideoData::url("https://example.com/video.mp4");
        let result = processor.process_video(video);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not supported"));
    }

    #[test]
    fn test_process_content_text_only_with_video_config() {
        let cfg = ProcessorConfig::default();
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let content = vec![ContentPart::text("Hello, world!")];
        let inputs = processor
            .process_content(&content, |text| {
                Ok(text.chars().map(|c| c as u32).collect())
            })
            .unwrap();

        assert!(!inputs.has_images());
        assert!(!inputs.has_videos());
    }

    #[test]
    fn test_process_content_video_url_error() {
        // process_content with a video URL (non-embedding) must return an error
        // because video decoding from URL/bytes is not supported without a codec.
        let cfg = ProcessorConfig {
            video_placeholder: "V".to_string(),
            video_placeholder_id: b'V' as u32,
            ..ProcessorConfig::default()
        };
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let content = vec![ContentPart::video_url("https://example.com/clip.mp4")];
        let result = processor.process_content(&content, |text| {
            // ASCII tokenizer: each character maps to its code point
            Ok(text.chars().map(|c| c as u32).collect())
        });
        // Expect failure: URL video decoding is not supported
        assert!(result.is_err(), "expected error for URL video, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not supported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_find_video_placeholder_positions() {
        let cfg = ProcessorConfig {
            video_placeholder_id: 200,
            ..ProcessorConfig::default()
        };
        let device = Device::Cpu;
        let processor = MultimodalProcessor::new(cfg, &device, DType::F32);

        let tokens = vec![1, 200, 3, 200, 5];
        let positions = processor.find_video_placeholder_positions(&tokens);
        assert_eq!(positions, vec![1, 3]);
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
