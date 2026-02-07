//! Video encoder infrastructure for video-language models.
//!
//! Provides traits and utilities for encoding video frames into embeddings
//! compatible with language model hidden states.
//!
//! # Architecture
//!
//! Video processing follows these steps:
//! 1. Sample frames from the video according to a [`FrameSamplingStrategy`]
//! 2. Encode each frame (or batch of frames) with a [`VideoEncoder`]
//! 3. Optionally add temporal position encoding via [`add_temporal_encoding`]
//! 4. Inject video embeddings at placeholder positions in the token sequence

use candle_core::{Device, Result, Tensor};

/// Configuration for video encoding.
#[derive(Debug, Clone)]
pub struct VideoEncoderConfig {
    /// Number of frames to sample per video.
    pub num_frames: usize,
    /// Image size for each frame (square).
    pub image_size: usize,
    /// Patch size for vision transformer.
    pub patch_size: usize,
    /// Hidden size of frame embeddings.
    pub hidden_size: usize,
    /// Whether to use temporal encoding.
    pub use_temporal_encoding: bool,
}

impl Default for VideoEncoderConfig {
    fn default() -> Self {
        Self {
            num_frames: 8,
            image_size: 224,
            patch_size: 14,
            hidden_size: 1024,
            use_temporal_encoding: true,
        }
    }
}

impl VideoEncoderConfig {
    /// Number of spatial patches per frame dimension.
    pub fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of spatial patches per frame.
    pub fn patches_per_frame(&self) -> usize {
        let n = self.patches_per_side();
        n * n
    }

    /// Total number of video tokens for the default frame count.
    pub fn total_tokens(&self) -> usize {
        self.num_frames * self.patches_per_frame()
    }
}

/// Trait for video encoding models.
///
/// Video encoders process video frames into embeddings that can be
/// consumed by video-language models. Implementations may use per-frame
/// vision encoders (e.g., CLIP applied to each frame) or native video
/// encoders (e.g., ViViT, TimeSformer).
pub trait VideoEncoder: Send + Sync + 'static {
    /// Encode video frames into embeddings.
    ///
    /// # Arguments
    /// * `frames` - Video frames tensor [num_frames, channels, height, width]
    ///
    /// # Returns
    /// Video embedding tensor [num_video_tokens, hidden_size]
    fn encode_frames(&self, frames: &Tensor) -> Result<Tensor>;

    /// Number of tokens produced per frame.
    fn tokens_per_frame(&self) -> usize;

    /// Total video tokens for a given number of frames.
    fn total_video_tokens(&self, num_frames: usize) -> usize {
        num_frames * self.tokens_per_frame()
    }

    /// Get the encoder configuration.
    fn config(&self) -> &VideoEncoderConfig;

    /// Get the device this encoder runs on.
    fn device(&self) -> &Device;
}

/// Frame sampling strategy for videos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameSamplingStrategy {
    /// Sample frames uniformly across the video duration.
    Uniform,
    /// Sample at a fixed frames-per-second rate.
    FixedFps(f32),
    /// Use keyframes only (I-frames).
    Keyframes,
}

impl Default for FrameSamplingStrategy {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Compute frame indices for uniform sampling.
///
/// Given a total number of frames in a video and a desired sample count,
/// returns indices spaced evenly across the video.
pub fn compute_uniform_frame_indices(total_frames: usize, num_samples: usize) -> Vec<usize> {
    if num_samples == 0 || total_frames == 0 {
        return Vec::new();
    }
    if num_samples >= total_frames {
        return (0..total_frames).collect();
    }
    (0..num_samples)
        .map(|i| {
            let pos = i as f64 * (total_frames - 1) as f64 / (num_samples - 1).max(1) as f64;
            pos.round() as usize
        })
        .collect()
}

/// Add temporal position encoding to frame embeddings.
///
/// Adds frame-level positional information so the model can distinguish
/// between frames at different points in the video. Uses a simple
/// linear position scaling.
///
/// # Arguments
/// * `frame_embeddings` - Tensor [total_tokens, hidden_size] where
///   total_tokens = num_frames * tokens_per_frame
/// * `num_frames` - Number of video frames
/// * `tokens_per_frame` - Number of tokens each frame produces
///
/// # Returns
/// Tensor with temporal encoding added, same shape as input.
pub fn add_temporal_encoding(
    frame_embeddings: &Tensor,
    num_frames: usize,
    tokens_per_frame: usize,
) -> Result<Tensor> {
    let (total_tokens, hidden_size) = frame_embeddings.dims2()?;

    if total_tokens != num_frames * tokens_per_frame {
        return Err(candle_core::Error::Msg(format!(
            "temporal encoding dimension mismatch: total_tokens={} but num_frames={} * tokens_per_frame={} = {}",
            total_tokens,
            num_frames,
            tokens_per_frame,
            num_frames * tokens_per_frame,
        )));
    }

    let device = frame_embeddings.device();
    let mut positions = Vec::with_capacity(total_tokens);
    for frame_idx in 0..num_frames {
        let pos = frame_idx as f32 / num_frames.max(1) as f32;
        for _ in 0..tokens_per_frame {
            positions.push(pos);
        }
    }

    let pos_tensor = Tensor::from_vec(positions, (total_tokens, 1), device)?;
    let pos_expanded = pos_tensor.broadcast_as((total_tokens, hidden_size))?;

    // Scale by a small factor to avoid dominating the frame features
    let scaled = (pos_expanded * 0.1)?;
    frame_embeddings + scaled
}

/// Video placeholder token configuration.
#[derive(Debug, Clone)]
pub struct VideoPlaceholderConfig {
    /// Token used as video placeholder in the prompt.
    pub placeholder_token: String,
    /// Token ID for the placeholder.
    pub placeholder_token_id: u32,
    /// Default number of frames to sample.
    pub default_num_frames: usize,
    /// Number of tokens produced per frame.
    pub tokens_per_frame: usize,
}

impl Default for VideoPlaceholderConfig {
    fn default() -> Self {
        Self {
            placeholder_token: "<video>".to_string(),
            placeholder_token_id: 32001,
            default_num_frames: 8,
            tokens_per_frame: 256,
        }
    }
}

impl VideoPlaceholderConfig {
    /// Total number of tokens for a video with the default frame count.
    pub fn total_tokens(&self) -> usize {
        self.default_num_frames * self.tokens_per_frame
    }

    /// Total number of tokens for a video with the given frame count.
    pub fn total_tokens_for_frames(&self, num_frames: usize) -> usize {
        num_frames * self.tokens_per_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_video_encoder_config_default() {
        let cfg = VideoEncoderConfig::default();
        assert_eq!(cfg.num_frames, 8);
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.hidden_size, 1024);
        assert!(cfg.use_temporal_encoding);
    }

    #[test]
    fn test_video_encoder_config_patches() {
        let cfg = VideoEncoderConfig::default();
        assert_eq!(cfg.patches_per_side(), 16); // 224 / 14 = 16
        assert_eq!(cfg.patches_per_frame(), 256); // 16 * 16 = 256
        assert_eq!(cfg.total_tokens(), 2048); // 8 * 256 = 2048
    }

    #[test]
    fn test_video_encoder_config_custom() {
        let cfg = VideoEncoderConfig {
            num_frames: 16,
            image_size: 336,
            patch_size: 14,
            hidden_size: 1024,
            use_temporal_encoding: false,
        };
        assert_eq!(cfg.patches_per_side(), 24);
        assert_eq!(cfg.patches_per_frame(), 576);
        assert_eq!(cfg.total_tokens(), 9216);
    }

    #[test]
    fn test_frame_sampling_strategy_default() {
        assert_eq!(
            FrameSamplingStrategy::default(),
            FrameSamplingStrategy::Uniform
        );
    }

    #[test]
    fn test_frame_sampling_strategy_variants() {
        let uniform = FrameSamplingStrategy::Uniform;
        let fixed_fps = FrameSamplingStrategy::FixedFps(2.0);
        let keyframes = FrameSamplingStrategy::Keyframes;

        assert_eq!(uniform, FrameSamplingStrategy::Uniform);
        assert_eq!(fixed_fps, FrameSamplingStrategy::FixedFps(2.0));
        assert_eq!(keyframes, FrameSamplingStrategy::Keyframes);
        assert_ne!(uniform, keyframes);
    }

    #[test]
    fn test_compute_uniform_frame_indices_basic() {
        // 100 frames, sample 8
        let indices = compute_uniform_frame_indices(100, 8);
        assert_eq!(indices.len(), 8);
        assert_eq!(indices[0], 0);
        assert_eq!(*indices.last().unwrap(), 99);

        // Indices should be monotonically increasing
        for window in indices.windows(2) {
            assert!(window[1] > window[0]);
        }
    }

    #[test]
    fn test_compute_uniform_frame_indices_exact() {
        // Fewer frames than samples: return all frames
        let indices = compute_uniform_frame_indices(5, 10);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_compute_uniform_frame_indices_equal() {
        let indices = compute_uniform_frame_indices(8, 8);
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_compute_uniform_frame_indices_single() {
        let indices = compute_uniform_frame_indices(100, 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_compute_uniform_frame_indices_empty() {
        assert!(compute_uniform_frame_indices(0, 8).is_empty());
        assert!(compute_uniform_frame_indices(8, 0).is_empty());
        assert!(compute_uniform_frame_indices(0, 0).is_empty());
    }

    #[test]
    fn test_add_temporal_encoding() {
        let device = Device::Cpu;
        let num_frames = 4;
        let tokens_per_frame = 16;
        let hidden_size = 64;
        let total_tokens = num_frames * tokens_per_frame;

        let embeddings = Tensor::ones((total_tokens, hidden_size), DType::F32, &device).unwrap();
        let result = add_temporal_encoding(&embeddings, num_frames, tokens_per_frame).unwrap();

        assert_eq!(result.dims(), &[total_tokens, hidden_size]);

        // Tokens from the first frame should have temporal position 0.0
        let first_frame_token: Vec<f32> = result
            .narrow(0, 0, 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        // 1.0 + 0.0 * 0.1 = 1.0
        assert!((first_frame_token[0] - 1.0).abs() < 1e-5);

        // Tokens from the last frame should have temporal position (3/4) * 0.1 = 0.075
        let last_start = (num_frames - 1) * tokens_per_frame;
        let last_frame_token: Vec<f32> = result
            .narrow(0, last_start, 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        // 1.0 + (3/4) * 0.1 = 1.075
        assert!((last_frame_token[0] - 1.075).abs() < 1e-5);
    }

    #[test]
    fn test_add_temporal_encoding_dimension_mismatch() {
        let device = Device::Cpu;
        let embeddings = Tensor::ones((100, 64), DType::F32, &device).unwrap();
        // 4 * 30 = 120 != 100 -> should error
        let result = add_temporal_encoding(&embeddings, 4, 30);
        assert!(result.is_err());
    }

    #[test]
    fn test_video_placeholder_config_default() {
        let cfg = VideoPlaceholderConfig::default();
        assert_eq!(cfg.placeholder_token, "<video>");
        assert_eq!(cfg.placeholder_token_id, 32001);
        assert_eq!(cfg.default_num_frames, 8);
        assert_eq!(cfg.tokens_per_frame, 256);
        assert_eq!(cfg.total_tokens(), 2048);
    }

    #[test]
    fn test_video_placeholder_config_custom() {
        let cfg = VideoPlaceholderConfig {
            placeholder_token: "<vid>".to_string(),
            placeholder_token_id: 50000,
            default_num_frames: 16,
            tokens_per_frame: 196,
        };
        assert_eq!(cfg.total_tokens(), 3136);
        assert_eq!(cfg.total_tokens_for_frames(8), 1568);
    }
}
