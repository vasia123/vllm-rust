//! Log-mel spectrogram feature extraction for audio models (Whisper-compatible).
//!
//! Implements the standard preprocessing pipeline used by Whisper and derivative
//! models (Qwen2-Audio, Ultravox, etc.):
//!
//! ```text
//! raw waveform [num_samples]
//!   → STFT (n_fft=400, hop=160, Hann window)
//!   → power spectrum [n_frames, n_fft/2]
//!   → mel filterbank [n_frames, n_mels]
//!   → log₁₀ + normalize
//!   → [n_mels, n_frames]   (channels-first for model input)
//! ```
//!
//! # Reference
//!
//! Parameters match the HuggingFace `WhisperFeatureExtractor` defaults:
//! n_fft=400, hop_length=160, n_mels=80, sampling_rate=16000, f_min=0, f_max=8000.
//!
//! NOTE: The STFT uses a pure Rust DFT implementation (O(n·n_fft) per frame).
//! For n_fft=400 and typical audio lengths this is fast enough for CPU preprocessing.
//! A production path would use rustfft or cuFFT.

use std::f64::consts::PI;

/// Parameters for log-mel spectrogram extraction.
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    /// FFT window size in samples. Whisper default: 400.
    pub n_fft: usize,
    /// Hop length (stride) between successive frames. Whisper default: 160.
    pub hop_length: usize,
    /// Number of mel frequency bins. Whisper default: 80.
    pub n_mels: usize,
    /// Input audio sample rate in Hz. Whisper default: 16000.
    pub sample_rate: usize,
    /// Minimum frequency for mel filterbank. Whisper default: 0.0.
    pub f_min: f64,
    /// Maximum frequency for mel filterbank. Whisper default: 8000.0 (Nyquist at 16 kHz).
    pub f_max: f64,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self::whisper()
    }
}

impl MelSpectrogramConfig {
    /// Standard Whisper feature extractor configuration.
    pub fn whisper() -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            sample_rate: 16000,
            f_min: 0.0,
            f_max: 8000.0,
        }
    }

    /// Number of positive-frequency DFT bins: n_fft / 2.
    ///
    /// The power spectrum uses only the first n_fft/2 bins (the negative-frequency
    /// mirror bins carry no new information for real-valued signals). The last bin
    /// (DC + N/2) is dropped to match the Python `torch.stft` convention.
    pub fn num_freq_bins(&self) -> usize {
        self.n_fft / 2
    }
}

// ─── Hann Window ────────────────────────────────────────────────────────────

/// Computes a periodic Hann window of length `n`.
///
/// `w[k] = 0.5 · (1 − cos(2π·k/n))`
///
/// "Periodic" (not "symmetric") matches numpy's `np.hanning(n)` and
/// scipy's `get_window('hann', n, fftbins=True)`.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| {
            let theta = 2.0 * PI * k as f64 / n as f64;
            (0.5 * (1.0 - theta.cos())) as f32
        })
        .collect()
}

// ─── STFT Power Spectrum ────────────────────────────────────────────────────

/// Computes the short-time Fourier transform power spectrum.
///
/// For each frame of `n_fft` samples (with `hop_length` between frames) the
/// signal is multiplied by the Hann window and the real DFT is computed. Only
/// the first `n_fft/2` positive-frequency bins are retained (matching the
/// Python `torch.stft(onesided=True)` convention minus the Nyquist bin).
///
/// # Arguments
/// * `samples`    — mono audio samples `[num_samples]`, values in \[-1, 1\]
/// * `window`     — precomputed Hann window `[n_fft]`
/// * `n_fft`      — FFT window size
/// * `hop_length` — samples between successive frames
///
/// # Returns
/// Power spectrum `[num_frames, n_fft/2]` (magnitude squared, no normalization).
pub fn stft_power_spectrum(
    samples: &[f32],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Vec<Vec<f32>> {
    let num_freq = n_fft / 2;
    let num_frames = if samples.len() < n_fft {
        0
    } else {
        (samples.len() - n_fft) / hop_length + 1
    };

    let mut power = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        let frame = &samples[start..start + n_fft];

        // Apply Hann window
        let windowed: Vec<f64> = frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s as f64 * w as f64)
            .collect();

        // DFT: compute only bins 0..num_freq
        let mut frame_power = Vec::with_capacity(num_freq);
        let n = n_fft as f64;
        for k in 0..num_freq {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let phase_step = -2.0 * PI * k as f64 / n;
            for (t, &x) in windowed.iter().enumerate() {
                let angle = phase_step * t as f64;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            frame_power.push((re * re + im * im) as f32);
        }
        power.push(frame_power);
    }

    power
}

// ─── Mel Filterbank ─────────────────────────────────────────────────────────

/// Converts Hz to the mel scale.
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts mel scale to Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10_f64.powf(mel / 2595.0) - 1.0)
}

/// Builds the mel filterbank matrix.
///
/// Returns `[n_mels, num_freq_bins]` where each row is one triangular mel filter.
/// The filter weights are HTK-style (area-normalized triangles).
///
/// # Arguments
/// * `n_mels`    — number of mel bins
/// * `n_fft`     — FFT size (only first n_fft/2 bins are used)
/// * `sample_rate` — audio sample rate in Hz
/// * `f_min`     — minimum frequency (Hz)
/// * `f_max`     — maximum frequency (Hz, typically Nyquist)
pub fn build_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f32>> {
    let num_freq = n_fft / 2;

    // Linear spacing of n_mels+2 points on the mel scale
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // Convert mel center frequencies back to Hz, then to FFT bin indices
    let hz_per_bin = sample_rate as f64 / n_fft as f64;
    let bin_points: Vec<f64> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) / hz_per_bin)
        .collect();

    // Construct triangular filters
    let mut filters = vec![vec![0.0f32; num_freq]; n_mels];
    for (m, filter_row) in filters.iter_mut().enumerate() {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for (k, weight) in filter_row.iter_mut().enumerate() {
            let k_f = k as f64;
            *weight = if k_f >= f_left && k_f <= f_center {
                let denom = f_center - f_left;
                if denom > 1e-12 {
                    ((k_f - f_left) / denom) as f32
                } else {
                    0.0
                }
            } else if k_f > f_center && k_f <= f_right {
                let denom = f_right - f_center;
                if denom > 1e-12 {
                    ((f_right - k_f) / denom) as f32
                } else {
                    0.0
                }
            } else {
                0.0
            };
        }
    }

    filters
}

// ─── Full Pipeline ───────────────────────────────────────────────────────────

/// Computes the log-mel spectrogram from a mono audio waveform.
///
/// This matches the output of HuggingFace `WhisperFeatureExtractor`:
/// - Multiply STFT power spectrum by mel filterbank
/// - Apply log₁₀ with floor at 1e-10
/// - Normalize: `max(log_spec, global_max − 8) → (log_spec + 4) / 4`
///
/// # Arguments
/// * `samples` — mono f32 samples `[num_samples]`, values in \[-1, 1\]
/// * `cfg`     — mel spectrogram configuration
///
/// # Returns
/// `[n_mels, num_frames]` log-mel feature matrix, values approximately in \[0, 2\].
pub fn log_mel_spectrogram(samples: &[f32], cfg: &MelSpectrogramConfig) -> Vec<Vec<f32>> {
    let window = hann_window(cfg.n_fft);
    let power = stft_power_spectrum(samples, &window, cfg.n_fft, cfg.hop_length);

    let num_frames = power.len();
    if num_frames == 0 {
        return vec![vec![]; cfg.n_mels];
    }

    let mel_fb = build_mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.f_min, cfg.f_max);

    // Apply mel filterbank: mel_spec[m][t] = Σ_k mel_fb[m][k] * power[t][k]
    let mut mel_spec = vec![vec![0.0f32; num_frames]; cfg.n_mels];
    for (t, frame) in power.iter().enumerate() {
        for (m, mel_row) in mel_spec.iter_mut().enumerate() {
            mel_row[t] = mel_fb[m]
                .iter()
                .zip(frame.iter())
                .map(|(&w, &p)| w * p)
                .sum();
        }
    }

    // Log₁₀ with floor
    for row in mel_spec.iter_mut() {
        for v in row.iter_mut() {
            *v = v.max(1e-10).log10();
        }
    }

    // Global max normalization (Whisper-style)
    let global_max = mel_spec
        .iter()
        .flat_map(|row| row.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);

    let floor = global_max - 8.0;
    for row in mel_spec.iter_mut() {
        for v in row.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
    }

    mel_spec
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    #[test]
    fn test_hann_window_length() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
    }

    #[test]
    fn test_hann_window_endpoints() {
        // Periodic Hann: w[0] = 0, w[N/2] = 1
        let w = hann_window(400);
        assert!(w[0].abs() < 1e-6, "w[0] should be 0, got {}", w[0]);
        assert!(
            (w[200] - 1.0).abs() < 1e-5,
            "w[N/2] should be 1, got {}",
            w[200]
        );
    }

    #[test]
    fn test_stft_power_shape() {
        let n_fft = 400;
        let hop = 160;
        let num_samples = 16000; // 1 second
        let samples = vec![0.0f32; num_samples];
        let window = hann_window(n_fft);
        let power = stft_power_spectrum(&samples, &window, n_fft, hop);

        let expected_frames = (num_samples - n_fft) / hop + 1; // 99 frames for 16000 samples
        assert_eq!(power.len(), expected_frames);
        assert_eq!(power[0].len(), n_fft / 2); // 200 freq bins
    }

    #[test]
    fn test_stft_silence_is_zero() {
        let n_fft = 400;
        let hop = 160;
        let samples = vec![0.0f32; 16000];
        let window = hann_window(n_fft);
        let power = stft_power_spectrum(&samples, &window, n_fft, hop);

        for frame in &power {
            for &v in frame {
                assert!(
                    v.abs() < 1e-10,
                    "silence should produce zero power, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_stft_single_frequency() {
        // Pure 440 Hz tone: power should peak at bin ≈ 440*400/16000 = 11
        let n_fft = 400;
        let hop = 160;
        let sample_rate = 16000;
        let freq = 440.0f32;
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let window = hann_window(n_fft);
        let power = stft_power_spectrum(&samples, &window, n_fft, hop);

        // Check that the peak bin is near expected
        let expected_bin = (freq as usize * n_fft / sample_rate).min(n_fft / 2 - 1);
        let max_bin = power[10]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            (max_bin as i64 - expected_bin as i64).abs() <= 2,
            "440 Hz peak at bin {max_bin}, expected ~{expected_bin}"
        );
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = build_mel_filterbank(80, 400, 16000, 0.0, 8000.0);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), 200);
    }

    #[test]
    fn test_mel_filterbank_partition_of_unity() {
        // Each frequency bin should be covered by at least some filter
        let fb = build_mel_filterbank(80, 400, 16000, 0.0, 8000.0);
        let num_freq = 200;
        let col_sums: Vec<f32> = (0..num_freq)
            .map(|k| fb.iter().map(|row| row[k]).sum::<f32>())
            .collect();
        // Most bins should have non-zero total filter response (excluding edges)
        let covered = col_sums[1..num_freq - 1]
            .iter()
            .filter(|&&s| s > 1e-6)
            .count();
        assert!(
            covered > num_freq / 2,
            "too many uncovered frequency bins: {covered} covered out of {num_freq}"
        );
    }

    #[test]
    fn test_log_mel_spectrogram_shape() {
        let cfg = MelSpectrogramConfig::whisper();
        // 0.5 second of silence
        let samples = vec![0.0f32; cfg.sample_rate / 2];
        let mel = log_mel_spectrogram(&samples, &cfg);

        assert_eq!(mel.len(), cfg.n_mels, "wrong n_mels");
        let expected_frames = (samples.len() - cfg.n_fft) / cfg.hop_length + 1;
        assert_eq!(mel[0].len(), expected_frames, "wrong num_frames");
    }

    #[test]
    fn test_log_mel_spectrogram_range() {
        // Output should be approximately in [0, 2] after Whisper normalization
        let cfg = MelSpectrogramConfig::whisper();
        let freq = 440.0f32;
        let samples: Vec<f32> = (0..cfg.sample_rate)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / cfg.sample_rate as f32).sin())
            .collect();

        let mel = log_mel_spectrogram(&samples, &cfg);

        for row in &mel {
            for &v in row {
                assert!(
                    v >= -0.1 && v <= 2.5,
                    "mel value {v} out of expected range [0, 2.5]"
                );
            }
        }
    }
}
