//! Multimodal support for vision-language models.
//!
//! This module provides infrastructure for handling multimodal inputs
//! (text + images/video/audio) in LLMs like LLaVA, Qwen-VL, Whisper, etc.
//!
//! # Architecture
//!
//! Multimodal processing follows these steps:
//! 1. Parse content parts (text, images, audio) from the request
//! 2. Encode media using appropriate encoders (CLIP/SigLIP for images, Whisper for audio)
//! 3. Tokenize text with special placeholder tokens for media
//! 4. Inject media embeddings at placeholder positions during forward pass
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::multimodal::{ContentPart, ImageData, MultimodalProcessor};
//!
//! let content = vec![
//!     ContentPart::Text("What is in this image?".to_string()),
//!     ContentPart::Image(ImageData::url("https://example.com/image.jpg")),
//! ];
//!
//! let processor = MultimodalProcessor::new(vision_encoder, tokenizer);
//! let inputs = processor.process(&content)?;
//! ```

pub mod audio;
mod inputs;
pub mod mel_spectrogram;
mod processor;
mod projector;
pub mod video;
mod vision;

pub use inputs::{
    ContentPart, ImageData, ImageSource, MultimodalInputs, ProcessedAudio, ProcessedImage,
    ProcessedVideo, VideoData, VideoSource, VideoUrl,
};
pub use mel_spectrogram::{
    build_mel_filterbank, hann_window, log_mel_spectrogram, stft_power_spectrum,
    MelSpectrogramConfig,
};
pub use processor::{MultimodalProcessor, ProcessorConfig};
pub use projector::{MultimodalProjector, ProjectorConfig, ProjectorType};
pub use video::{FrameSamplingStrategy, VideoEncoder, VideoEncoderConfig, VideoPlaceholderConfig};
pub use vision::{VisionEncoder, VisionEncoderConfig, VisionEncoderType};
