//! WebSocket realtime transcription endpoint — `GET /v1/realtime`.
//!
//! # Protocol (OpenAI Realtime Audio API subset)
//!
//! 1. Client opens a WebSocket to `ws://host/v1/realtime`.
//! 2. Server sends `session.created`.
//! 3. Client may send `session.update` to validate/select a model.
//! 4. Client sends one or more `input_audio_buffer.append` events with
//!    base64-encoded raw PCM16 @ 16 kHz mono audio chunks.
//! 5. Client sends `input_audio_buffer.commit` to trigger a transcription run.
//!    - `final: true` signals that no more audio will arrive after this
//!      generation; the connection closes cleanly after the response.
//!    - `final: false` (default) runs transcription and stays connected for
//!      the next utterance.
//! 6. Server streams `transcription.delta` events (one per generated token).
//! 7. Server sends `transcription.done` with the full text and usage stats.
//! 8. Steps 4–7 may repeat for additional utterances.
//!
//! # Audio format
//!
//! Raw PCM16 little-endian @ 16 kHz mono, base64-encoded.  Equivalent to
//! `audio/pcm;rate=16000` with 16-bit signed samples.  This matches the
//! format produced by browser `AudioContext` and the OpenAI Realtime spec.
//!
//! # Reference
//!
//! `reference/vllm/vllm/entrypoints/openai/realtime/`

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use vllm_core::engine::{GenerationRequest, StreamEvent};
use vllm_core::multimodal::audio::{normalize_audio, AudioData, AudioSpec};
use vllm_core::sampling::SamplingParams;

use super::AppState;

// ─── Protocol: server → client ───────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ServerEvent {
    #[serde(rename = "session.created")]
    SessionCreated { id: String, created: u64 },
    #[serde(rename = "transcription.delta")]
    TranscriptionDelta { delta: String },
    #[serde(rename = "transcription.done")]
    TranscriptionDone { text: String, usage: UsageInfo },
    #[serde(rename = "error")]
    Error {
        error: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },
}

#[derive(Debug, Serialize)]
struct UsageInfo {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ─── Protocol: client → server ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientEvent {
    #[serde(rename = "session.update")]
    SessionUpdate { model: Option<String> },
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend { audio: String },
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit {
        #[serde(default)]
        r#final: bool,
    },
}

// ─── Connection state ─────────────────────────────────────────────────────────

/// Maximum accumulated PCM16 audio bytes (~4 MB → ~125 s @ 16 kHz mono).
const MAX_AUDIO_BYTES: usize = 4 * 1024 * 1024;

struct Connection {
    session_id: String,
    audio_buf: Vec<u8>,
    model_validated: bool,
    input_finished: bool,
}

impl Connection {
    fn new() -> Self {
        Self {
            session_id: format!("sess-{}", Uuid::new_v4()),
            audio_buf: Vec::new(),
            model_validated: false,
            input_finished: false,
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Encode a `ServerEvent` to a WebSocket text message.
fn encode_event(ev: ServerEvent) -> Message {
    Message::Text(serde_json::to_string(&ev).unwrap_or_default().into())
}

/// Decode PCM16 (little-endian int16) bytes to f32 samples in [-1, 1].
fn pcm16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let s = i16::from_le_bytes([c[0], c[1]]);
            s as f32 / 32768.0
        })
        .collect()
}

// ─── WebSocket handler ────────────────────────────────────────────────────────

/// `GET /v1/realtime` — upgrade to WebSocket and handle the realtime session.
pub async fn realtime_ws(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let mut conn = Connection::new();

    // Send session.created immediately on accept.
    let created_msg = encode_event(ServerEvent::SessionCreated {
        id: conn.session_id.clone(),
        created: timestamp_now(),
    });
    if socket.send(created_msg).await.is_err() {
        return;
    }

    loop {
        let msg = match socket.recv().await {
            Some(Ok(m)) => m,
            // Client closed or network error.
            _ => break,
        };

        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            // Ignore binary, ping, pong.
            _ => continue,
        };

        let event: ClientEvent = match serde_json::from_str(&text) {
            Ok(ev) => ev,
            Err(_) => {
                let _ = socket
                    .send(encode_event(ServerEvent::Error {
                        error: "invalid JSON or unknown event type".to_string(),
                        code: Some("invalid_event".to_string()),
                    }))
                    .await;
                continue;
            }
        };

        match event {
            ClientEvent::SessionUpdate { model } => {
                if let Some(ref m) = model {
                    if m != &state.model_id {
                        let _ = socket
                            .send(encode_event(ServerEvent::Error {
                                error: format!("model '{}' not found", m),
                                code: Some("model_not_found".to_string()),
                            }))
                            .await;
                        continue;
                    }
                }
                conn.model_validated = true;
            }

            ClientEvent::InputAudioBufferAppend { audio } => {
                let bytes = match base64_decode(&audio) {
                    Some(b) => b,
                    None => {
                        let _ = socket
                            .send(encode_event(ServerEvent::Error {
                                error: "invalid base64 audio data".to_string(),
                                code: Some("invalid_audio".to_string()),
                            }))
                            .await;
                        continue;
                    }
                };

                if conn.audio_buf.len() + bytes.len() > MAX_AUDIO_BYTES {
                    let _ = socket
                        .send(encode_event(ServerEvent::Error {
                            error: "audio buffer too large (max 4 MB)".to_string(),
                            code: Some("audio_too_large".to_string()),
                        }))
                        .await;
                    continue;
                }

                conn.audio_buf.extend_from_slice(&bytes);
            }

            ClientEvent::InputAudioBufferCommit { r#final } => {
                if r#final {
                    conn.input_finished = true;
                }

                if conn.audio_buf.is_empty() {
                    let _ = socket
                        .send(encode_event(ServerEvent::Error {
                            error: "audio buffer is empty".to_string(),
                            code: Some("empty_audio".to_string()),
                        }))
                        .await;
                    if conn.input_finished {
                        break;
                    }
                    continue;
                }

                // Run transcription for the buffered audio.
                let should_close = run_transcription(&mut socket, &state, &conn).await;
                conn.audio_buf.clear();

                if should_close || conn.input_finished {
                    break;
                }
            }
        }
    }

    // Send a close frame; ignore errors (client may have already disconnected).
    drop(socket);
}

/// Decode base64 audio, build a `GenerationRequest`, stream tokens back,
/// and send `transcription.done`.  Returns `true` if the socket should close.
async fn run_transcription(socket: &mut WebSocket, state: &AppState, conn: &Connection) -> bool {
    let samples = pcm16_bytes_to_f32(&conn.audio_buf);
    let raw_audio = AudioData::mono(samples, 16_000);

    let audio_spec = AudioSpec::whisper();
    let audio = match normalize_audio(&raw_audio, &audio_spec) {
        Ok(a) => a,
        Err(e) => {
            let _ = socket
                .send(encode_event(ServerEvent::Error {
                    error: format!("audio normalization failed: {}", e),
                    code: Some("audio_error".to_string()),
                }))
                .await;
            return false;
        }
    };

    let gen_req = GenerationRequest {
        prompt: "Transcribe the audio.".to_string(),
        max_new_tokens: state.max_model_len.min(4096),
        eos_token_id: state.eos_token_id,
        sampling_params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        audio_inputs: vec![audio],
        ..GenerationRequest::default()
    };

    let (_req_id, mut rx) = match state.engine.get().generate_stream(gen_req).await {
        Ok(pair) => pair,
        Err(e) => {
            let _ = socket
                .send(encode_event(ServerEvent::Error {
                    error: format!("engine error: {}", e),
                    code: Some("engine_error".to_string()),
                }))
                .await;
            return false;
        }
    };

    let mut full_text = String::new();
    let mut completion_tokens: usize = 0;

    loop {
        let ev = match rx.recv().await {
            Some(ev) => ev,
            None => break,
        };

        match ev {
            StreamEvent::Token { token_text, .. } => {
                full_text.push_str(&token_text);
                completion_tokens += 1;
                let _ = socket
                    .send(encode_event(ServerEvent::TranscriptionDelta {
                        delta: token_text,
                    }))
                    .await;
            }
            StreamEvent::Done { generated_text, .. } => {
                // `generated_text` is the full accumulated text from the engine.
                if !generated_text.is_empty() {
                    full_text = generated_text;
                }
                break;
            }
            StreamEvent::Error { error } => {
                let _ = socket
                    .send(encode_event(ServerEvent::Error {
                        error,
                        code: Some("generation_error".to_string()),
                    }))
                    .await;
                return false;
            }
        }
    }

    let _ = socket
        .send(encode_event(ServerEvent::TranscriptionDone {
            text: full_text,
            usage: UsageInfo {
                // NOTE: prompt_tokens not exposed by generate_stream; use 0 as
                // placeholder. A future improvement is to return the prompt
                // token count alongside the first StreamEvent::Token.
                prompt_tokens: 0,
                completion_tokens,
                total_tokens: completion_tokens,
            },
        }))
        .await;

    false
}

/// Simple base64 decode (standard alphabet, ignores padding errors).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    use std::collections::HashMap;

    // Build decode table once (per-call; OK since this is not a hot path).
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table: HashMap<u8, u8> = HashMap::with_capacity(64);
    for (i, &c) in alphabet.iter().enumerate() {
        table.insert(c, i as u8);
    }

    let bytes: Vec<u8> = s.bytes().filter(|b| *b != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);

    for chunk in bytes.chunks(4) {
        let mut vals = [0u8; 4];
        let n = chunk.len();
        for (i, &b) in chunk.iter().enumerate() {
            vals[i] = *table.get(&b)?;
        }
        match n {
            4 => {
                out.push((vals[0] << 2) | (vals[1] >> 4));
                out.push((vals[1] << 4) | (vals[2] >> 2));
                out.push((vals[2] << 6) | vals[3]);
            }
            3 => {
                out.push((vals[0] << 2) | (vals[1] >> 4));
                out.push((vals[1] << 4) | (vals[2] >> 2));
            }
            2 => {
                out.push((vals[0] << 2) | (vals[1] >> 4));
            }
            _ => return None,
        }
    }

    Some(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_decode_empty() {
        let out = pcm16_bytes_to_f32(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn pcm16_decode_zero() {
        let bytes = [0u8, 0u8, 0u8, 0u8];
        let out = pcm16_bytes_to_f32(&bytes);
        assert_eq!(out, &[0.0f32, 0.0f32]);
    }

    #[test]
    fn pcm16_decode_max_positive() {
        // i16::MAX = 32767 → 32767/32768 ≈ 0.9999…
        let bytes = [0xFF, 0x7F]; // 0x7FFF little-endian
        let out = pcm16_bytes_to_f32(&bytes);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 32767.0f32 / 32768.0).abs() < 1e-5);
    }

    #[test]
    fn pcm16_decode_min_negative() {
        // i16::MIN = -32768 → -1.0
        let bytes = [0x00, 0x80]; // 0x8000 little-endian
        let out = pcm16_bytes_to_f32(&bytes);
        assert_eq!(out.len(), 1);
        assert!((out[0] - (-1.0f32)).abs() < 1e-5);
    }

    #[test]
    fn base64_decode_known_value() {
        // "hello" → "aGVsbG8="
        let decoded = base64_decode("aGVsbG8=").expect("valid base64");
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn base64_decode_empty_string() {
        let decoded = base64_decode("").expect("empty is valid");
        assert!(decoded.is_empty());
    }

    #[test]
    fn base64_decode_invalid_char() {
        // '!' is not in the base64 alphabet → None
        let result = base64_decode("aG!sbG8=");
        assert!(result.is_none());
    }

    #[test]
    fn session_created_serialises_correctly() {
        let ev = ServerEvent::SessionCreated {
            id: "sess-abc".to_string(),
            created: 1_700_000_000,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"type\":\"session.created\""));
        assert!(json.contains("\"id\":\"sess-abc\""));
        assert!(json.contains("\"created\":1700000000"));
    }

    #[test]
    fn error_event_omits_null_code() {
        let ev = ServerEvent::Error {
            error: "oops".to_string(),
            code: None,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(!json.contains("code"));
    }
}
