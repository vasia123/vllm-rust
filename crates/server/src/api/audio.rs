//! Audio transcription and translation API handlers.
//!
//! Implements OpenAI-compatible `/v1/audio/transcriptions` and
//! `/v1/audio/translations` endpoints.
//!
//! # Request format
//!
//! Both endpoints accept `multipart/form-data` with the following fields:
//! - `file` (required): audio file bytes (WAV format currently supported)
//! - `model` (optional): model identifier (must match the loaded model)
//! - `language` (optional): ISO-639-1 language code of the source audio
//! - `prompt` (optional): text to guide transcription style
//! - `response_format` (optional): `json` (default) | `text` | `verbose_json`
//! - `temperature` (optional): sampling temperature (default 0.0)
//!
//! # Response format
//!
//! ```json
//! { "text": "transcribed text here" }
//! ```
//!
//! # Architecture
//!
//! The handler parses audio bytes into raw PCM samples (WAV via hound),
//! wraps them in `AudioData`, and submits a `GenerationRequest` to the
//! existing engine with `audio_inputs` populated.
//!
//! NOTE: The model's `forward_prefill_batch` must read `audio_inputs` from
//! the request, run the audio encoder + projector, and splice the resulting
//! hidden states into the token sequence at the audio placeholder position
//! (analogous to how `image_inputs` are handled for VLMs). This is the
//! remaining work to make transcription fully functional end-to-end.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Multipart, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use vllm_core::engine::{GenerationRequest, GenerationResult};
use vllm_core::multimodal::audio::{normalize_audio, AudioData, AudioSpec};
use vllm_core::sampling::SamplingParams;

use super::error::ApiError;
use super::AppState;

// ─── Response-format type ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioResponseFormat {
    #[default]
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

// ─── Response types ───────────────────────────────────────────────────────────

/// Response for `json` or `text` formats.
#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// Usage field for transcription (audio duration in seconds).
#[derive(Debug, Serialize)]
pub struct TranscriptionUsage {
    pub r#type: &'static str,
    pub seconds: u64,
}

/// Verbose transcription response with metadata.
#[derive(Debug, Serialize)]
pub struct TranscriptionResponseVerbose {
    pub text: String,
    pub language: Option<String>,
    pub duration: f64,
    pub usage: TranscriptionUsage,
}

/// Response for translation requests.
#[derive(Debug, Serialize)]
pub struct TranslationResponse {
    pub text: String,
}

// ─── Parsed form data ─────────────────────────────────────────────────────────

struct AudioFormData {
    /// Raw audio file bytes.
    file_bytes: Vec<u8>,
    /// Optional filename hint (used to detect format).
    file_name: Option<String>,
    model: Option<String>,
    language: Option<String>,
    to_language: Option<String>,
    prompt: String,
    response_format: AudioResponseFormat,
    temperature: f32,
}

impl Default for AudioFormData {
    fn default() -> Self {
        Self {
            file_bytes: Vec::new(),
            file_name: None,
            model: None,
            language: None,
            to_language: None,
            prompt: String::new(),
            response_format: AudioResponseFormat::Json,
            temperature: 0.0,
        }
    }
}

/// Parse a `multipart/form-data` request into `AudioFormData`.
async fn parse_audio_form(mut multipart: Multipart) -> Result<AudioFormData, ApiError> {
    let mut form = AudioFormData::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::InvalidRequest(format!("multipart parse error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                form.file_name = field.file_name().map(str::to_string);
                form.file_bytes = field
                    .bytes()
                    .await
                    .map_err(|e| {
                        ApiError::InvalidRequest(format!("failed to read audio file: {}", e))
                    })?
                    .to_vec();
            }
            "model" => {
                form.model = Some(field.text().await.map_err(|e| {
                    ApiError::InvalidRequest(format!("invalid model field: {}", e))
                })?);
            }
            "language" => {
                let v = field.text().await.map_err(|e| {
                    ApiError::InvalidRequest(format!("invalid language field: {}", e))
                })?;
                if !v.is_empty() {
                    form.language = Some(v);
                }
            }
            "to_language" => {
                let v = field.text().await.map_err(|e| {
                    ApiError::InvalidRequest(format!("invalid to_language field: {}", e))
                })?;
                if !v.is_empty() {
                    form.to_language = Some(v);
                }
            }
            "prompt" => {
                form.prompt = field.text().await.map_err(|e| {
                    ApiError::InvalidRequest(format!("invalid prompt field: {}", e))
                })?;
            }
            "response_format" => {
                let v = field.text().await.map_err(|e| {
                    ApiError::InvalidRequest(format!("invalid response_format field: {}", e))
                })?;
                form.response_format = match v.as_str() {
                    "json" => AudioResponseFormat::Json,
                    "text" => AudioResponseFormat::Text,
                    "verbose_json" => AudioResponseFormat::VerboseJson,
                    "srt" => AudioResponseFormat::Srt,
                    "vtt" => AudioResponseFormat::Vtt,
                    other => {
                        return Err(ApiError::InvalidRequest(format!(
                            "unsupported response_format '{}', use json/text/verbose_json",
                            other
                        )))
                    }
                };
            }
            "temperature" => {
                let v = field.text().await.map_err(|e| {
                    ApiError::InvalidRequest(format!("invalid temperature field: {}", e))
                })?;
                form.temperature = v.parse::<f32>().map_err(|_| {
                    ApiError::InvalidRequest("temperature must be a number".to_string())
                })?;
            }
            // Silently ignore unknown fields for forward-compatibility.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    if form.file_bytes.is_empty() {
        return Err(ApiError::InvalidRequest(
            "missing required field 'file'".to_string(),
        ));
    }

    Ok(form)
}

/// Parse audio bytes to `AudioData`.
///
/// Currently supports PCM WAV files via `hound`. For other formats (MP3, FLAC,
/// OGG, WEBM), returns an error with a TODO noting that symphonia integration
/// is needed for full OpenAI API format support.
fn parse_audio_bytes(bytes: &[u8], file_name: Option<&str>) -> Result<AudioData, ApiError> {
    // WAV detection: RIFF...WAVE magic bytes.
    let is_wav = bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE";

    if is_wav {
        return parse_wav_bytes(bytes);
    }

    // Hint from filename extension as fallback.
    let ext = file_name
        .and_then(|n| n.rfind('.').map(|i| &n[i + 1..]))
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "wav" => parse_wav_bytes(bytes),
        _ => Err(ApiError::InvalidRequest(
            // TODO: Add MP3/FLAC/OGG/WEBM support via the `symphonia` crate.
            // Only WAV (PCM) is supported. Use `ffmpeg -i input.mp3 output.wav`
            // to convert before uploading.
            "unsupported audio format — only PCM WAV is currently supported. \
             Convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav"
                .to_string(),
        )),
    }
}

/// Decode a WAV file to interleaved f32 PCM samples using `hound`.
fn parse_wav_bytes(bytes: &[u8]) -> Result<AudioData, ApiError> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| ApiError::InvalidRequest(format!("invalid WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ApiError::InvalidRequest(format!("WAV read error: {}", e)))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| ApiError::InvalidRequest(format!("WAV read error: {}", e)))?
        }
    };

    Ok(AudioData::new(samples, sample_rate, channels))
}

/// Duration of `AudioData` in seconds.
fn audio_duration_s(audio: &AudioData) -> f64 {
    if audio.sample_rate == 0 || audio.channels == 0 {
        return 0.0;
    }
    (audio.samples.len() / audio.channels) as f64 / audio.sample_rate as f64
}

/// Build the generation prompt for an audio task.
///
/// Audio-language models expect the user content to contain an audio
/// placeholder token (e.g. `<|AUDIO|>`, `<|audio|>`, `<audio>`) that the
/// tokenizer maps to a special ID. The model's `forward_prefill_batch` then
/// replaces those embeddings with the audio encoder output.
///
/// Since the placeholder token is model-specific and not available at the
/// server level, we fall back to a plain instruction. Models that require
/// an explicit placeholder (Qwen2-Audio, Ultravox) will need the
/// placeholder injected by the model-specific tokenizer/processor.
///
/// NOTE: A future improvement is to expose a `audio_placeholder` field in
/// `AppState` (read from model config) and inject it here.
fn build_audio_prompt(
    language: Option<&str>,
    to_language: Option<&str>,
    user_prompt: &str,
    task: AudioTask,
) -> String {
    let mut parts = Vec::new();

    // Model-specific audio placeholder would go here — left as a comment so
    // it is obvious when reading the prompt why audio doesn't show up:
    // parts.push("<|AUDIO|>".to_string());

    match task {
        AudioTask::Transcribe => {
            let lang_hint = language.map(|l| format!(" in {}", l)).unwrap_or_default();
            parts.push(format!("Transcribe the audio{}.", lang_hint));
        }
        AudioTask::Translate => {
            let from = language.map(|l| format!(" from {}", l)).unwrap_or_default();
            let to = to_language
                .map(|l| format!(" to {}", l))
                .unwrap_or(" to English".to_string());
            parts.push(format!("Translate the audio{}{}.", from, to));
        }
    }

    if !user_prompt.is_empty() {
        parts.push(user_prompt.to_string());
    }

    parts.join(" ")
}

#[derive(Clone, Copy)]
enum AudioTask {
    Transcribe,
    Translate,
}

/// Shared logic for both transcription and translation.
async fn run_audio_task(
    state: AppState,
    form: AudioFormData,
    task: AudioTask,
) -> Result<(GenerationResult, AudioData), ApiError> {
    // Model check.
    if let Some(ref m) = form.model {
        if m != &state.model_id {
            return Err(ApiError::ModelNotFound(format!("model '{}' not found", m)));
        }
    }

    // Parse audio bytes → raw PCM samples.
    let raw_audio = parse_audio_bytes(&form.file_bytes, form.file_name.as_deref())?;

    // Normalize to mono 16 kHz (de-facto standard for ASR models).
    // This mirrors Whisper's pre-processing pipeline.
    let audio_spec = AudioSpec::whisper();
    let audio = normalize_audio(&raw_audio, &audio_spec)
        .map_err(|e| ApiError::InvalidRequest(format!("audio normalization failed: {}", e)))?;

    let prompt = build_audio_prompt(
        form.language.as_deref(),
        form.to_language.as_deref(),
        &form.prompt,
        task,
    );

    let sampling_params = SamplingParams {
        temperature: form.temperature,
        ..SamplingParams::default()
    };

    let gen_req = GenerationRequest {
        prompt,
        max_new_tokens: state.max_model_len,
        eos_token_id: state.eos_token_id,
        sampling_params,
        audio_inputs: vec![audio.clone()],
        ..GenerationRequest::default()
    };

    let result = state
        .engine
        .get()
        .generate(gen_req)
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    Ok((result, audio))
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// `POST /v1/audio/transcriptions`
///
/// Transcribes audio into text in the language of the audio.
pub async fn create_transcription(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    let form = parse_audio_form(multipart).await?;
    let response_format = form.response_format;
    let language = form.language.clone();

    let (result, audio) = run_audio_task(state, form, AudioTask::Transcribe).await?;
    let text = result.generated_text;

    match response_format {
        AudioResponseFormat::Text => Ok(text.into_response()),
        AudioResponseFormat::VerboseJson => {
            let duration = audio_duration_s(&audio);
            let resp = TranscriptionResponseVerbose {
                text,
                language,
                duration,
                usage: TranscriptionUsage {
                    r#type: "duration",
                    seconds: duration.ceil() as u64,
                },
            };
            Ok(Json(resp).into_response())
        }
        AudioResponseFormat::Srt | AudioResponseFormat::Vtt => {
            // NOTE: SRT/VTT require timestamp tokens from the model output.
            // Models like Whisper produce `<|t0.00|>...<|t0.50|>` tokens that
            // encode segment boundaries. This parsing is not yet implemented.
            // Fall back to plain JSON until timestamp parsing is added.
            Ok(Json(TranscriptionResponse { text }).into_response())
        }
        _ => Ok(Json(TranscriptionResponse { text }).into_response()),
    }
}

/// `POST /v1/audio/translations`
///
/// Translates audio into English text (or another language if `to_language` is
/// specified and the model supports it).
pub async fn create_translation(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    let form = parse_audio_form(multipart).await?;
    let response_format = form.response_format;

    let (result, _audio) = run_audio_task(state, form, AudioTask::Translate).await?;
    let text = result.generated_text;

    match response_format {
        AudioResponseFormat::Text => Ok(text.into_response()),
        _ => Ok(Json(TranslationResponse { text }).into_response()),
    }
}

/// Timestamp of the current second (for response IDs).
#[allow(dead_code)]
fn timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wav_bytes(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let mut buf = Vec::new();
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        buf
    }

    #[test]
    fn parse_wav_mono_16bit() {
        let samples: Vec<i16> = (0..16_000).map(|i| (i % 100) as i16).collect();
        let bytes = make_wav_bytes(16_000, 1, &samples);
        let audio = parse_wav_bytes(&bytes).expect("should parse WAV");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 16_000);
        // i16 max = 32768; value 99 → 99/32768 ≈ 0.00302
        assert!((audio.samples[99] - 99.0 / 32768.0).abs() < 1e-4);
    }

    #[test]
    fn parse_wav_stereo() {
        let samples: Vec<i16> = vec![100, -100, 200, -200, 300, -300];
        let bytes = make_wav_bytes(44_100, 2, &samples);
        let audio = parse_wav_bytes(&bytes).expect("should parse stereo WAV");
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 6);
    }

    #[test]
    fn parse_audio_bytes_detects_wav_magic() {
        let samples: Vec<i16> = vec![0i16; 100];
        let bytes = make_wav_bytes(16_000, 1, &samples);
        // Should succeed even without a filename extension.
        let audio = parse_audio_bytes(&bytes, None).expect("WAV magic bytes detected");
        assert_eq!(audio.sample_rate, 16_000);
    }

    #[test]
    fn parse_audio_bytes_rejects_non_wav() {
        let fake_mp3 = b"\xff\xfb\x90\x00fake mp3 data";
        let err = parse_audio_bytes(fake_mp3, Some("audio.mp3")).unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("WAV") || msg.contains("wav") || msg.contains("unsupported"));
    }

    #[test]
    fn audio_duration_mono() {
        let audio = AudioData::mono(vec![0.0f32; 16_000], 16_000);
        let d = audio_duration_s(&audio);
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn build_transcribe_prompt_with_language() {
        let p = build_audio_prompt(Some("fr"), None, "", AudioTask::Transcribe);
        assert!(p.contains("Transcribe") && p.contains("fr"));
    }

    #[test]
    fn build_translate_prompt_defaults_to_english() {
        let p = build_audio_prompt(Some("de"), None, "", AudioTask::Translate);
        assert!(p.contains("Translate") && p.contains("English"));
    }
}
