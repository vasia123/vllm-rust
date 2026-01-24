use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use vllm_core::engine::StreamEvent;

use super::types::{
    finish_reason_str, timestamp_now, ChatCompletionChunk, ChatCompletionChunkChoice, ChatDelta,
    CompletionChunk, CompletionChunkChoice,
};

pub fn completion_sse_stream(
    request_id: String,
    model: String,
    rx: mpsc::Receiver<StreamEvent>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx).map(move |event| {
        let sse_event = match event {
            StreamEvent::Token { token_text, .. } => {
                let chunk = CompletionChunk {
                    id: request_id.clone(),
                    object: "text_completion",
                    created: timestamp_now(),
                    model: model.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: token_text,
                        index: 0,
                        finish_reason: None,
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
            StreamEvent::Done { finish_reason, .. } => {
                let chunk = CompletionChunk {
                    id: request_id.clone(),
                    object: "text_completion",
                    created: timestamp_now(),
                    model: model.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: String::new(),
                        index: 0,
                        finish_reason: Some(finish_reason_str(&finish_reason)),
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
            StreamEvent::Error { error } => {
                Event::default().data(format!("{{\"error\":\"{error}\"}}"))
            }
        };
        Ok(sse_event)
    });

    let done_stream =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let combined = stream.chain(done_stream);
    Sse::new(combined).keep_alive(KeepAlive::default())
}

pub fn chat_completion_sse_stream(
    request_id: String,
    model: String,
    rx: mpsc::Receiver<StreamEvent>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut first = true;
    let stream = ReceiverStream::new(rx).map(move |event| {
        let sse_event = match event {
            StreamEvent::Token { token_text, .. } => {
                let mut choices = Vec::new();
                if first {
                    choices.push(ChatCompletionChunkChoice {
                        delta: ChatDelta {
                            role: Some("assistant".to_string()),
                            content: Some(token_text),
                        },
                        index: 0,
                        finish_reason: None,
                    });
                    first = false;
                } else {
                    choices.push(ChatCompletionChunkChoice {
                        delta: ChatDelta {
                            role: None,
                            content: Some(token_text),
                        },
                        index: 0,
                        finish_reason: None,
                    });
                }
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created: timestamp_now(),
                    model: model.clone(),
                    choices,
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
            StreamEvent::Done { finish_reason, .. } => {
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created: timestamp_now(),
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        delta: ChatDelta {
                            role: None,
                            content: None,
                        },
                        index: 0,
                        finish_reason: Some(finish_reason_str(&finish_reason)),
                    }],
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            }
            StreamEvent::Error { error } => {
                Event::default().data(format!("{{\"error\":\"{error}\"}}"))
            }
        };
        Ok(sse_event)
    });

    let done_stream =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let combined = stream.chain(done_stream);
    Sse::new(combined).keep_alive(KeepAlive::default())
}
