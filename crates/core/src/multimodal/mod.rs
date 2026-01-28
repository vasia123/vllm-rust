//! Multimodal support for vision-language models.
//!
//! This module provides infrastructure for handling multimodal inputs
//! (text + images/video) in LLMs like LLaVA, Qwen-VL, etc.
//!
//! # Architecture
//!
//! Multimodal processing follows these steps:
//! 1. Parse content parts (text, images) from the request
//! 2. Encode images using a vision encoder (CLIP/SigLIP)
//! 3. Tokenize text with special placeholder tokens for images
//! 4. Inject image embeddings at placeholder positions during forward pass
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

mod inputs;
mod processor;
mod vision;

pub use inputs::{ContentPart, ImageData, ImageSource, MultimodalInputs, ProcessedImage};
pub use processor::{MultimodalProcessor, ProcessorConfig};
pub use vision::{VisionEncoder, VisionEncoderConfig, VisionEncoderType};
