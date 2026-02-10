//! Audio preprocessing for multimodal models.
//!
//! This module provides infrastructure for handling audio inputs
//! in multimodal LLMs like Whisper, Qwen-Audio, etc.
//!
//! # Architecture
//!
//! Audio processing follows these steps:
//! 1. Load audio from file or raw bytes (WAV, MP3, FLAC, etc.)
//! 2. Normalize: convert to mono, resample, normalize amplitude
//! 3. Extract features (e.g., mel spectrogram) - to be added later
//! 4. Pass to audio encoder for embedding generation
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::multimodal::audio::{AudioData, AudioSpec, ChannelReduction, normalize_audio};
//!
//! let audio = AudioData {
//!     samples: vec![0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
//!     sample_rate: 16000,
//!     channels: 2,
//! };
//!
//! let spec = AudioSpec {
//!     sample_rate: 16000,
//!     target_channels: Some(1),
//!     reduction: ChannelReduction::Mean,
//! };
//!
//! let normalized = normalize_audio(&audio, &spec).unwrap();
//! assert_eq!(normalized.channels, 1);
//! ```

use thiserror::Error;

/// Error type for audio processing operations.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Cannot expand channels (e.g., mono to stereo).
    #[error("cannot expand {from} channels to {to} channels")]
    CannotExpandChannels { from: usize, to: usize },

    /// Invalid audio data.
    #[error("invalid audio data: {0}")]
    InvalidData(String),

    /// Sample rate conversion error.
    #[error("sample rate conversion failed: {0}")]
    ResampleError(String),

    /// File I/O error.
    #[cfg(feature = "audio")]
    #[error("audio file error: {0}")]
    FileError(String),

    /// Decoding error.
    #[cfg(feature = "audio")]
    #[error("audio decoding error: {0}")]
    DecodeError(String),

    /// Unsupported format.
    #[cfg(feature = "audio")]
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),
}

/// Method to reduce multi-channel audio to target channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelReduction {
    /// Average across channels (default, preserves energy balance).
    #[default]
    Mean,
    /// Take first channel only.
    First,
    /// Take max value across channels at each sample.
    Max,
    /// Sum across channels.
    Sum,
}

/// Specification for target audio format.
///
/// This struct defines the expected audio format for a model's feature
/// extractor. It is used to normalize audio data before processing.
#[derive(Debug, Clone)]
pub struct AudioSpec {
    /// Target sample rate in Hz. None means passthrough (no resampling).
    pub sample_rate: Option<u32>,
    /// Number of output channels. None means passthrough (no channel conversion).
    /// 1 = mono, 2 = stereo, etc.
    pub target_channels: Option<usize>,
    /// Method to reduce channels when input has more channels than target.
    pub reduction: ChannelReduction,
    /// Whether to normalize amplitude to [-1, 1] range.
    pub normalize_amplitude: bool,
}

impl Default for AudioSpec {
    fn default() -> Self {
        Self {
            sample_rate: None,
            target_channels: Some(1), // Mono by default
            reduction: ChannelReduction::Mean,
            normalize_amplitude: true,
        }
    }
}

impl AudioSpec {
    /// Create a spec for mono audio at a specific sample rate.
    pub fn mono(sample_rate: u32) -> Self {
        Self {
            sample_rate: Some(sample_rate),
            target_channels: Some(1),
            reduction: ChannelReduction::Mean,
            normalize_amplitude: true,
        }
    }

    /// Create a passthrough spec (no normalization).
    pub fn passthrough() -> Self {
        Self {
            sample_rate: None,
            target_channels: None,
            reduction: ChannelReduction::Mean,
            normalize_amplitude: false,
        }
    }

    /// Whisper-compatible spec: 16kHz mono.
    pub fn whisper() -> Self {
        Self::mono(16000)
    }

    /// Check if any normalization is needed.
    pub fn needs_normalization(&self) -> bool {
        self.sample_rate.is_some() || self.target_channels.is_some() || self.normalize_amplitude
    }
}

/// Pre-defined specs for common use cases.
pub const MONO_AUDIO_SPEC: AudioSpec = AudioSpec {
    sample_rate: None,
    target_channels: Some(1),
    reduction: ChannelReduction::Mean,
    normalize_amplitude: true,
};

pub const PASSTHROUGH_AUDIO_SPEC: AudioSpec = AudioSpec {
    sample_rate: None,
    target_channels: None,
    reduction: ChannelReduction::Mean,
    normalize_amplitude: false,
};

/// Raw audio data with metadata.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Interleaved audio samples in [-1, 1] range (or unnormalized).
    /// For stereo: [L0, R0, L1, R1, ...]
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: usize,
}

impl AudioData {
    /// Create new audio data.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: usize) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Create mono audio data.
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(samples, sample_rate, 1)
    }

    /// Create stereo audio data.
    pub fn stereo(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(samples, sample_rate, 2)
    }

    /// Get the number of samples per channel.
    pub fn num_samples(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels
        }
    }

    /// Get the duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.num_samples() as f64 / self.sample_rate as f64
        }
    }

    /// Check if this is mono audio.
    pub fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// Get sample at a specific frame and channel.
    pub fn get_sample(&self, frame: usize, channel: usize) -> Option<f32> {
        if channel >= self.channels {
            return None;
        }
        let idx = frame * self.channels + channel;
        self.samples.get(idx).copied()
    }
}

/// Normalize audio to the specified format.
///
/// This function handles:
/// - Channel reduction for multi-channel audio
/// - Sample rate conversion (basic linear interpolation)
/// - Amplitude normalization to [-1, 1] range
///
/// # Arguments
/// * `audio` - Input audio data
/// * `spec` - AudioSpec defining the target format
///
/// # Returns
/// Normalized audio data, or error if conversion is not possible.
///
/// # Errors
/// - `CannotExpandChannels` if channel expansion is requested (e.g., mono to stereo)
/// - `InvalidData` if audio data is malformed
pub fn normalize_audio(audio: &AudioData, spec: &AudioSpec) -> Result<AudioData, AudioError> {
    if !spec.needs_normalization() {
        return Ok(audio.clone());
    }

    // Validate input
    if audio.channels == 0 {
        return Err(AudioError::InvalidData("audio has 0 channels".to_string()));
    }
    if audio.samples.is_empty() {
        return Err(AudioError::InvalidData("audio has no samples".to_string()));
    }
    if !audio.samples.len().is_multiple_of(audio.channels) {
        return Err(AudioError::InvalidData(format!(
            "sample count {} is not divisible by channel count {}",
            audio.samples.len(),
            audio.channels
        )));
    }

    let mut result = audio.clone();

    // Step 1: Channel reduction
    if let Some(target_channels) = spec.target_channels {
        result = reduce_channels(&result, target_channels, spec.reduction)?;
    }

    // Step 2: Sample rate conversion
    if let Some(target_rate) = spec.sample_rate {
        if result.sample_rate != target_rate {
            result = resample_linear(&result, target_rate)?;
        }
    }

    // Step 3: Amplitude normalization
    if spec.normalize_amplitude {
        result = normalize_amplitude(&result);
    }

    Ok(result)
}

/// Reduce audio channels to target count.
fn reduce_channels(
    audio: &AudioData,
    target_channels: usize,
    reduction: ChannelReduction,
) -> Result<AudioData, AudioError> {
    if audio.channels == target_channels {
        return Ok(audio.clone());
    }

    if audio.channels < target_channels {
        return Err(AudioError::CannotExpandChannels {
            from: audio.channels,
            to: target_channels,
        });
    }

    let num_frames = audio.num_samples();

    // Currently only supporting reduction to mono
    if target_channels == 1 {
        let mut mono_samples = Vec::with_capacity(num_frames);

        for frame in 0..num_frames {
            let sample = match reduction {
                ChannelReduction::Mean => {
                    let mut sum = 0.0f32;
                    for ch in 0..audio.channels {
                        sum += audio.samples[frame * audio.channels + ch];
                    }
                    sum / audio.channels as f32
                }
                ChannelReduction::First => audio.samples[frame * audio.channels],
                ChannelReduction::Max => {
                    let mut max = f32::NEG_INFINITY;
                    for ch in 0..audio.channels {
                        let val = audio.samples[frame * audio.channels + ch];
                        if val > max {
                            max = val;
                        }
                    }
                    max
                }
                ChannelReduction::Sum => {
                    let mut sum = 0.0f32;
                    for ch in 0..audio.channels {
                        sum += audio.samples[frame * audio.channels + ch];
                    }
                    sum
                }
            };
            mono_samples.push(sample);
        }

        Ok(AudioData::mono(mono_samples, audio.sample_rate))
    } else {
        // For N > 1 target channels, take first N channels
        let mut reduced_samples = Vec::with_capacity(num_frames * target_channels);
        for frame in 0..num_frames {
            for ch in 0..target_channels {
                reduced_samples.push(audio.samples[frame * audio.channels + ch]);
            }
        }

        Ok(AudioData::new(
            reduced_samples,
            audio.sample_rate,
            target_channels,
        ))
    }
}

/// Resample audio using linear interpolation.
///
/// This is a basic resampling implementation. For production use with
/// critical audio quality requirements, consider using a more sophisticated
/// algorithm (e.g., sinc interpolation via the `rubato` crate).
fn resample_linear(audio: &AudioData, target_rate: u32) -> Result<AudioData, AudioError> {
    if audio.sample_rate == target_rate {
        return Ok(audio.clone());
    }

    if audio.sample_rate == 0 {
        return Err(AudioError::ResampleError(
            "source sample rate is 0".to_string(),
        ));
    }

    let ratio = target_rate as f64 / audio.sample_rate as f64;
    let num_input_frames = audio.num_samples();
    let num_output_frames = (num_input_frames as f64 * ratio).ceil() as usize;

    let mut output_samples = Vec::with_capacity(num_output_frames * audio.channels);

    for out_frame in 0..num_output_frames {
        // Map output frame to input frame position
        let in_pos = out_frame as f64 / ratio;
        let in_frame_low = in_pos.floor() as usize;
        let in_frame_high = (in_frame_low + 1).min(num_input_frames.saturating_sub(1));
        let frac = (in_pos - in_frame_low as f64) as f32;

        for ch in 0..audio.channels {
            let idx_low = in_frame_low * audio.channels + ch;
            let idx_high = in_frame_high * audio.channels + ch;

            let val_low = audio.samples.get(idx_low).copied().unwrap_or(0.0);
            let val_high = audio.samples.get(idx_high).copied().unwrap_or(0.0);

            // Linear interpolation
            let interpolated = val_low + frac * (val_high - val_low);
            output_samples.push(interpolated);
        }
    }

    Ok(AudioData::new(output_samples, target_rate, audio.channels))
}

/// Normalize audio amplitude to [-1, 1] range.
fn normalize_amplitude(audio: &AudioData) -> AudioData {
    // Find peak absolute value
    let peak = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    if peak == 0.0 || peak <= 1.0 {
        // Already normalized or silent
        return audio.clone();
    }

    let scale = 1.0 / peak;
    let normalized_samples: Vec<f32> = audio.samples.iter().map(|s| s * scale).collect();

    AudioData::new(normalized_samples, audio.sample_rate, audio.channels)
}

// ─── File Loading (requires `audio` feature) ─────────────────────────────────

#[cfg(feature = "audio")]
mod file_loading {
    use super::*;
    use std::fs::File;
    use std::path::Path;
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    /// Load audio from a file path.
    ///
    /// Supports common audio formats: WAV, MP3, FLAC, OGG, etc.
    /// (depending on symphonia's enabled codecs)
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    ///
    /// # Returns
    /// Decoded audio data with original sample rate and channel count.
    pub fn load_audio_file<P: AsRef<Path>>(path: P) -> Result<AudioData, AudioError> {
        let path = path.as_ref();

        let file = File::open(path).map_err(|e| AudioError::FileError(e.to_string()))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Provide hints about the file format
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe the format
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| AudioError::DecodeError(format!("probe failed: {}", e)))?;

        let mut format = probed.format;

        // Get the default track
        let track = format
            .default_track()
            .ok_or_else(|| AudioError::DecodeError("no audio track found".to_string()))?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| AudioError::DecodeError("unknown sample rate".to_string()))?;
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .ok_or_else(|| AudioError::DecodeError("unknown channel count".to_string()))?;

        // Create decoder
        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| AudioError::DecodeError(format!("decoder creation failed: {}", e)))?;

        let mut all_samples: Vec<f32> = Vec::new();

        // Decode all packets
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break; // End of stream
                }
                Err(e) => {
                    return Err(AudioError::DecodeError(format!(
                        "packet read failed: {}",
                        e
                    )));
                }
            };

            // Skip packets from other tracks
            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(e)) => {
                    // Skip decode errors for resilience
                    tracing::warn!("decode error (skipping): {}", e);
                    continue;
                }
                Err(e) => {
                    return Err(AudioError::DecodeError(format!("decode failed: {}", e)));
                }
            };

            // Get sample buffer
            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;

            let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
            sample_buf.copy_interleaved_ref(decoded);

            all_samples.extend_from_slice(sample_buf.samples());
        }

        if all_samples.is_empty() {
            return Err(AudioError::DecodeError(
                "no audio samples decoded".to_string(),
            ));
        }

        Ok(AudioData::new(all_samples, sample_rate, channels))
    }

    /// Load audio from raw bytes with format hint.
    ///
    /// # Arguments
    /// * `data` - Raw audio file bytes
    /// * `format_hint` - Optional format hint (e.g., "wav", "mp3")
    pub fn load_audio_bytes(
        data: &[u8],
        format_hint: Option<&str>,
    ) -> Result<AudioData, AudioError> {
        use std::io::Cursor;

        let cursor = Cursor::new(data.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = format_hint {
            hint.with_extension(ext);
        }

        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| AudioError::DecodeError(format!("probe failed: {}", e)))?;

        let mut format = probed.format;

        let track = format
            .default_track()
            .ok_or_else(|| AudioError::DecodeError("no audio track found".to_string()))?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| AudioError::DecodeError("unknown sample rate".to_string()))?;
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .ok_or_else(|| AudioError::DecodeError("unknown channel count".to_string()))?;

        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| AudioError::DecodeError(format!("decoder creation failed: {}", e)))?;

        let mut all_samples: Vec<f32> = Vec::new();

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    return Err(AudioError::DecodeError(format!(
                        "packet read failed: {}",
                        e
                    )));
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(e)) => {
                    tracing::warn!("decode error (skipping): {}", e);
                    continue;
                }
                Err(e) => {
                    return Err(AudioError::DecodeError(format!("decode failed: {}", e)));
                }
            };

            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;

            let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
            sample_buf.copy_interleaved_ref(decoded);

            all_samples.extend_from_slice(sample_buf.samples());
        }

        if all_samples.is_empty() {
            return Err(AudioError::DecodeError(
                "no audio samples decoded".to_string(),
            ));
        }

        Ok(AudioData::new(all_samples, sample_rate, channels))
    }
}

#[cfg(feature = "audio")]
pub use file_loading::{load_audio_bytes, load_audio_file};

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test audio data
    fn make_stereo_audio() -> AudioData {
        // Stereo audio: L and R channels with different patterns
        // L: [0.1, 0.2, 0.3, 0.4]
        // R: [0.5, 0.6, 0.7, 0.8]
        // Interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        AudioData::stereo(vec![0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8], 16000)
    }

    fn make_mono_audio() -> AudioData {
        AudioData::mono(vec![0.1, 0.2, 0.3, 0.4, 0.5], 16000)
    }

    #[test]
    fn test_audio_data_creation() {
        let audio = AudioData::mono(vec![0.1, 0.2, 0.3], 44100);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.num_samples(), 3);
        assert!(audio.is_mono());
    }

    #[test]
    fn test_audio_data_stereo() {
        let audio = make_stereo_audio();
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.num_samples(), 4);
        assert!(!audio.is_mono());
    }

    #[test]
    fn test_audio_duration() {
        let audio = AudioData::mono(vec![0.0; 16000], 16000);
        assert!((audio.duration_secs() - 1.0).abs() < 1e-6);

        let audio = AudioData::mono(vec![0.0; 8000], 16000);
        assert!((audio.duration_secs() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_get_sample() {
        let audio = make_stereo_audio();
        // Frame 0: L=0.1, R=0.5
        assert!((audio.get_sample(0, 0).unwrap() - 0.1).abs() < 1e-6);
        assert!((audio.get_sample(0, 1).unwrap() - 0.5).abs() < 1e-6);
        // Frame 1: L=0.2, R=0.6
        assert!((audio.get_sample(1, 0).unwrap() - 0.2).abs() < 1e-6);
        assert!((audio.get_sample(1, 1).unwrap() - 0.6).abs() < 1e-6);
        // Out of bounds
        assert!(audio.get_sample(0, 2).is_none());
        assert!(audio.get_sample(10, 0).is_none());
    }

    #[test]
    fn test_audio_spec_defaults() {
        let spec = AudioSpec::default();
        assert!(spec.target_channels.is_some());
        assert_eq!(spec.target_channels.unwrap(), 1);
        assert!(spec.normalize_amplitude);
        assert!(spec.needs_normalization());
    }

    #[test]
    fn test_audio_spec_passthrough() {
        let spec = AudioSpec::passthrough();
        assert!(spec.sample_rate.is_none());
        assert!(spec.target_channels.is_none());
        assert!(!spec.normalize_amplitude);
        assert!(!spec.needs_normalization());
    }

    #[test]
    fn test_audio_spec_whisper() {
        let spec = AudioSpec::whisper();
        assert_eq!(spec.sample_rate, Some(16000));
        assert_eq!(spec.target_channels, Some(1));
    }

    #[test]
    fn test_normalize_passthrough() {
        let audio = make_stereo_audio();
        let spec = AudioSpec::passthrough();
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.channels, audio.channels);
        assert_eq!(result.sample_rate, audio.sample_rate);
        assert_eq!(result.samples.len(), audio.samples.len());
    }

    #[test]
    fn test_channel_reduction_mean() {
        let audio = make_stereo_audio();
        let spec = AudioSpec {
            target_channels: Some(1),
            reduction: ChannelReduction::Mean,
            ..AudioSpec::passthrough()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.channels, 1);
        assert_eq!(result.num_samples(), 4);
        // Mean of [0.1, 0.5] = 0.3
        assert!((result.samples[0] - 0.3).abs() < 1e-6);
        // Mean of [0.2, 0.6] = 0.4
        assert!((result.samples[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_channel_reduction_first() {
        let audio = make_stereo_audio();
        let spec = AudioSpec {
            target_channels: Some(1),
            reduction: ChannelReduction::First,
            ..AudioSpec::passthrough()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.channels, 1);
        // First channel only: [0.1, 0.2, 0.3, 0.4]
        assert!((result.samples[0] - 0.1).abs() < 1e-6);
        assert!((result.samples[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_channel_reduction_max() {
        let audio = make_stereo_audio();
        let spec = AudioSpec {
            target_channels: Some(1),
            reduction: ChannelReduction::Max,
            ..AudioSpec::passthrough()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.channels, 1);
        // Max of [0.1, 0.5] = 0.5
        assert!((result.samples[0] - 0.5).abs() < 1e-6);
        // Max of [0.2, 0.6] = 0.6
        assert!((result.samples[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_channel_reduction_sum() {
        let audio = make_stereo_audio();
        let spec = AudioSpec {
            target_channels: Some(1),
            reduction: ChannelReduction::Sum,
            ..AudioSpec::passthrough()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.channels, 1);
        // Sum of [0.1, 0.5] = 0.6
        assert!((result.samples[0] - 0.6).abs() < 1e-6);
        // Sum of [0.2, 0.6] = 0.8
        assert!((result.samples[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cannot_expand_channels() {
        let audio = make_mono_audio();
        let spec = AudioSpec {
            target_channels: Some(2),
            ..AudioSpec::passthrough()
        };
        let result = normalize_audio(&audio, &spec);
        assert!(matches!(
            result,
            Err(AudioError::CannotExpandChannels { from: 1, to: 2 })
        ));
    }

    #[test]
    fn test_resample_upsample() {
        // 16kHz to 32kHz (2x)
        let audio = AudioData::mono(vec![0.0, 1.0, 0.0, -1.0], 16000);
        let spec = AudioSpec {
            sample_rate: Some(32000),
            target_channels: None,
            normalize_amplitude: false,
            ..Default::default()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.sample_rate, 32000);
        // Should have approximately 2x samples
        assert!(result.num_samples() >= 7);
    }

    #[test]
    fn test_resample_downsample() {
        // 32kHz to 16kHz (0.5x)
        let audio = AudioData::mono(vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5], 32000);
        let spec = AudioSpec {
            sample_rate: Some(16000),
            target_channels: None,
            normalize_amplitude: false,
            ..Default::default()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.sample_rate, 16000);
        // Should have approximately 0.5x samples
        assert!(result.num_samples() >= 3 && result.num_samples() <= 5);
    }

    #[test]
    fn test_amplitude_normalization() {
        // Audio with peak at 2.0 (needs normalization)
        let audio = AudioData::mono(vec![0.0, 1.0, 2.0, -1.0, 0.5], 16000);
        let spec = AudioSpec {
            normalize_amplitude: true,
            target_channels: None,
            sample_rate: None,
            ..Default::default()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        // Peak should now be 1.0
        let peak = result
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 1e-6);
        // Relative values should be preserved
        assert!((result.samples[2] - 1.0).abs() < 1e-6); // Was 2.0, now 1.0
        assert!((result.samples[1] - 0.5).abs() < 1e-6); // Was 1.0, now 0.5
    }

    #[test]
    fn test_amplitude_normalization_already_normalized() {
        let audio = AudioData::mono(vec![0.0, 0.5, 1.0, -0.5, 0.25], 16000);
        let spec = AudioSpec {
            normalize_amplitude: true,
            target_channels: None,
            sample_rate: None,
            ..Default::default()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        // Should be unchanged (already within [-1, 1])
        assert_eq!(result.samples, audio.samples);
    }

    #[test]
    fn test_full_pipeline() {
        // Stereo 44.1kHz -> Mono 16kHz, normalized
        let audio = AudioData::stereo(
            vec![0.2, 0.4, 0.4, 0.8, 0.6, 1.2, 0.8, 1.6], // 4 stereo frames
            44100,
        );
        let spec = AudioSpec::whisper();
        let result = normalize_audio(&audio, &spec).unwrap();

        assert_eq!(result.channels, 1);
        assert_eq!(result.sample_rate, 16000);
        // Amplitude should be normalized
        let peak = result
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(peak <= 1.0 + 1e-6);
    }

    #[test]
    fn test_invalid_empty_audio() {
        let audio = AudioData::mono(vec![], 16000);
        let spec = AudioSpec::default();
        let result = normalize_audio(&audio, &spec);
        assert!(matches!(result, Err(AudioError::InvalidData(_))));
    }

    #[test]
    fn test_invalid_zero_channels() {
        let audio = AudioData::new(vec![0.1, 0.2], 16000, 0);
        let spec = AudioSpec::default();
        let result = normalize_audio(&audio, &spec);
        assert!(matches!(result, Err(AudioError::InvalidData(_))));
    }

    #[test]
    fn test_invalid_misaligned_samples() {
        // 3 samples for stereo (should be even)
        let audio = AudioData::new(vec![0.1, 0.2, 0.3], 16000, 2);
        let spec = AudioSpec::default();
        let result = normalize_audio(&audio, &spec);
        assert!(matches!(result, Err(AudioError::InvalidData(_))));
    }

    #[test]
    fn test_predefined_specs() {
        // Test MONO_AUDIO_SPEC
        assert_eq!(MONO_AUDIO_SPEC.target_channels, Some(1));
        assert!(MONO_AUDIO_SPEC.normalize_amplitude);

        // Test PASSTHROUGH_AUDIO_SPEC
        assert!(PASSTHROUGH_AUDIO_SPEC.target_channels.is_none());
        assert!(!PASSTHROUGH_AUDIO_SPEC.normalize_amplitude);
    }

    #[test]
    fn test_channel_reduction_default() {
        assert_eq!(ChannelReduction::default(), ChannelReduction::Mean);
    }

    #[test]
    fn test_multi_channel_to_stereo() {
        // 4 channels to 2 channels (take first 2)
        let audio = AudioData::new(
            vec![
                0.1, 0.2, 0.3, 0.4, // Frame 0: ch0, ch1, ch2, ch3
                0.5, 0.6, 0.7, 0.8, // Frame 1
            ],
            16000,
            4,
        );
        let spec = AudioSpec {
            target_channels: Some(2),
            ..AudioSpec::passthrough()
        };
        let result = normalize_audio(&audio, &spec).unwrap();
        assert_eq!(result.channels, 2);
        assert_eq!(result.num_samples(), 2);
        // Should have first 2 channels only
        assert!((result.samples[0] - 0.1).abs() < 1e-6); // Frame 0, ch0
        assert!((result.samples[1] - 0.2).abs() < 1e-6); // Frame 0, ch1
        assert!((result.samples[2] - 0.5).abs() < 1e-6); // Frame 1, ch0
        assert!((result.samples[3] - 0.6).abs() < 1e-6); // Frame 1, ch1
    }
}
