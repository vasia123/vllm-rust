use std::collections::HashMap;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use vllm_core::engine::StreamEvent;
use vllm_core::tokenizer::TokenizerWrapper;

use super::types::{
    finish_reason_str, system_fingerprint, timestamp_now, ChatCompletionChunk,
    ChatCompletionChunkChoice, ChatDelta, ChatLogProbToken, ChatLogProbs, ChatTopLogProb,
    CompletionChunk, CompletionChunkChoice, CompletionLogProbs, Usage,
};

/// Options controlling what metadata to include in streaming responses.
#[derive(Clone, Default)]
pub struct StreamingOptions {
    /// If true, emit a final chunk with usage statistics before the [DONE] sentinel.
    pub include_usage: bool,
    /// Number of prompt tokens (needed for the usage chunk).
    pub prompt_tokens: usize,
    /// Whether the client requested logprobs in the streaming response.
    pub include_logprobs: bool,
    /// Tokenizer reference for decoding token IDs in logprobs to strings.
    pub tokenizer: Option<Arc<TokenizerWrapper>>,
}

/// Build a single `CompletionLogProbs` entry for one streaming token.
fn build_completion_token_logprobs(
    token_text: &str,
    text_offset: usize,
    logprob: Option<f32>,
    top_logprobs_raw: Option<&Vec<(u32, f32)>>,
    tokenizer: &TokenizerWrapper,
) -> CompletionLogProbs {
    let top = top_logprobs_raw.map(|entries| {
        entries
            .iter()
            .map(|&(tid, lp)| {
                let t = tokenizer
                    .decode(&[tid])
                    .unwrap_or_else(|_| format!("<unk:{}>", tid));
                (t, lp)
            })
            .collect::<HashMap<String, f32>>()
    });

    CompletionLogProbs {
        text_offset: vec![text_offset],
        token_logprobs: vec![logprob],
        tokens: vec![token_text.to_string()],
        top_logprobs: vec![top],
    }
}

/// Build a single `ChatLogProbs` entry for one streaming token.
fn build_chat_token_logprobs(
    token_text: &str,
    logprob: f32,
    top_logprobs_raw: Option<&Vec<(u32, f32)>>,
    tokenizer: &TokenizerWrapper,
) -> ChatLogProbs {
    let top = top_logprobs_raw.map(|entries| {
        entries
            .iter()
            .map(|&(tid, lp)| {
                let t = tokenizer
                    .decode(&[tid])
                    .unwrap_or_else(|_| format!("<unk:{}>", tid));
                ChatTopLogProb {
                    token: t.clone(),
                    logprob: lp,
                    bytes: Some(t.as_bytes().to_vec()),
                }
            })
            .collect::<Vec<_>>()
    });

    ChatLogProbs {
        content: vec![ChatLogProbToken {
            token: token_text.to_string(),
            logprob,
            bytes: Some(token_text.as_bytes().to_vec()),
            top_logprobs: top,
        }],
    }
}

pub fn completion_sse_stream(
    request_id: String,
    model: String,
    rx: mpsc::Receiver<StreamEvent>,
    options: StreamingOptions,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Convert to Arc<str> once — subsequent per-token clones are cheap
    // pointer bumps instead of heap allocations.
    let id: Arc<str> = request_id.into();
    let model: Arc<str> = model.into();

    let output_stream = async_stream::stream! {
        let mut completion_tokens: usize = 0;
        let mut text_offset: usize = 0;
        let mut rx_stream = ReceiverStream::new(rx);

        while let Some(event) = rx_stream.next().await {
            let sse_event = match event {
                StreamEvent::Token { token_text, logprob, top_logprobs: top_lps, .. } => {
                    completion_tokens += 1;

                    let chunk_logprobs = if options.include_logprobs {
                        if let Some(ref tokenizer) = options.tokenizer {
                            let lp = build_completion_token_logprobs(
                                &token_text,
                                text_offset,
                                logprob,
                                top_lps.as_ref(),
                                tokenizer,
                            );
                            Some(lp)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    text_offset += token_text.len();

                    let chunk = CompletionChunk {
                        id: Arc::clone(&id),
                        object: "text_completion",
                        created: timestamp_now(),
                        model: Arc::clone(&model),
                        system_fingerprint: system_fingerprint(),
                        choices: vec![CompletionChunkChoice {
                            text: token_text,
                            index: 0,
                            finish_reason: None,
                            logprobs: chunk_logprobs,
                        }],
                        usage: None,
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                }
                StreamEvent::Done { finish_reason, .. } => {
                    let chunk = CompletionChunk {
                        id: Arc::clone(&id),
                        object: "text_completion",
                        created: timestamp_now(),
                        model: Arc::clone(&model),
                        system_fingerprint: system_fingerprint(),
                        choices: vec![CompletionChunkChoice {
                            text: String::new(),
                            index: 0,
                            finish_reason: Some(finish_reason_str(&finish_reason)),
                            logprobs: None,
                        }],
                        usage: None,
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                }
                StreamEvent::Error { error } => {
                    Event::default().data(format!("{{\"error\":\"{error}\"}}"))
                }
            };
            yield Ok::<_, Infallible>(sse_event);
        }

        // Emit usage chunk if requested via stream_options.include_usage
        if options.include_usage {
            let usage_chunk = CompletionChunk {
                id: Arc::clone(&id),
                object: "text_completion",
                created: timestamp_now(),
                model: Arc::clone(&model),
                system_fingerprint: system_fingerprint(),
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens: options.prompt_tokens,
                    completion_tokens,
                    total_tokens: options.prompt_tokens + completion_tokens,
                }),
            };
            yield Ok(Event::default().data(serde_json::to_string(&usage_chunk).unwrap_or_default()));
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(output_stream).keep_alive(KeepAlive::default())
}

pub fn chat_completion_sse_stream(
    request_id: String,
    model: String,
    rx: mpsc::Receiver<StreamEvent>,
    options: StreamingOptions,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Convert to Arc<str> once — subsequent per-token clones are cheap
    // pointer bumps instead of heap allocations.
    let id: Arc<str> = request_id.into();
    let model: Arc<str> = model.into();

    let output_stream = async_stream::stream! {
        let mut first = true;
        let mut completion_tokens: usize = 0;
        let mut rx_stream = ReceiverStream::new(rx);

        while let Some(event) = rx_stream.next().await {
            let sse_event = match event {
                StreamEvent::Token { token_text, logprob, top_logprobs: top_lps, .. } => {
                    completion_tokens += 1;

                    let chunk_logprobs = if options.include_logprobs {
                        if let (Some(lp), Some(ref tokenizer)) = (logprob, &options.tokenizer) {
                            Some(build_chat_token_logprobs(
                                &token_text,
                                lp,
                                top_lps.as_ref(),
                                tokenizer,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let choices = if first {
                        first = false;
                        vec![ChatCompletionChunkChoice {
                            delta: ChatDelta {
                                role: Some("assistant".to_string()),
                                content: Some(token_text),
                                reasoning: None,
                                reasoning_content: None,
                            },
                            index: 0,
                            finish_reason: None,
                            logprobs: chunk_logprobs,
                        }]
                    } else {
                        vec![ChatCompletionChunkChoice {
                            delta: ChatDelta {
                                role: None,
                                content: Some(token_text),
                                reasoning: None,
                                reasoning_content: None,
                            },
                            index: 0,
                            finish_reason: None,
                            logprobs: chunk_logprobs,
                        }]
                    };
                    let chunk = ChatCompletionChunk {
                        id: Arc::clone(&id),
                        object: "chat.completion.chunk",
                        created: timestamp_now(),
                        model: Arc::clone(&model),
                        system_fingerprint: system_fingerprint(),
                        choices,
                        usage: None,
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                }
                StreamEvent::Done { finish_reason, .. } => {
                    let chunk = ChatCompletionChunk {
                        id: Arc::clone(&id),
                        object: "chat.completion.chunk",
                        created: timestamp_now(),
                        model: Arc::clone(&model),
                        system_fingerprint: system_fingerprint(),
                        choices: vec![ChatCompletionChunkChoice {
                            delta: ChatDelta {
                                role: None,
                                content: None,
                                reasoning: None,
                                reasoning_content: None,
                            },
                            index: 0,
                            finish_reason: Some(finish_reason_str(&finish_reason)),
                            logprobs: None,
                        }],
                        usage: None,
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                }
                StreamEvent::Error { error } => {
                    Event::default().data(format!("{{\"error\":\"{error}\"}}"))
                }
            };
            yield Ok::<_, Infallible>(sse_event);
        }

        // Emit usage chunk if requested via stream_options.include_usage
        if options.include_usage {
            let usage_chunk = ChatCompletionChunk {
                id: Arc::clone(&id),
                object: "chat.completion.chunk",
                created: timestamp_now(),
                model: Arc::clone(&model),
                system_fingerprint: system_fingerprint(),
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens: options.prompt_tokens,
                    completion_tokens,
                    total_tokens: options.prompt_tokens + completion_tokens,
                }),
            };
            yield Ok(Event::default().data(serde_json::to_string(&usage_chunk).unwrap_or_default()));
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(output_stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use tokio::sync::mpsc;
    use vllm_core::engine::StreamEvent;
    use vllm_core::request::FinishReason;

    /// Convert an SSE response body into a list of "data: ..." line payloads.
    async fn body_to_data_lines(response: axum::response::Response) -> Vec<String> {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        text.lines()
            .filter(|l| l.starts_with("data: "))
            .map(|l| l.to_string())
            .collect()
    }

    /// Collect all SSE data payloads from a completion stream.
    async fn collect_completion_sse(
        rx: mpsc::Receiver<StreamEvent>,
        options: StreamingOptions,
    ) -> Vec<String> {
        let sse =
            completion_sse_stream("test-id".to_string(), "test-model".to_string(), rx, options);
        body_to_data_lines(sse.into_response()).await
    }

    /// Collect all SSE data payloads from a chat completion stream.
    async fn collect_chat_sse(
        rx: mpsc::Receiver<StreamEvent>,
        options: StreamingOptions,
    ) -> Vec<String> {
        let sse = chat_completion_sse_stream(
            "test-id".to_string(),
            "test-model".to_string(),
            rx,
            options,
        );
        body_to_data_lines(sse.into_response()).await
    }

    /// Parse JSON from a "data: ..." line.
    fn parse_data_json(line: &str) -> serde_json::Value {
        let payload = line.trim_start_matches("data: ");
        serde_json::from_str(payload).unwrap()
    }

    #[tokio::test]
    async fn completion_stream_without_logprobs() {
        let (tx, rx) = mpsc::channel(16);
        tx.send(StreamEvent::Token {
            token_id: 1,
            token_text: "hello".to_string(),
            logprob: None,
            top_logprobs: None,
        })
        .await
        .unwrap();
        tx.send(StreamEvent::Done {
            finish_reason: FinishReason::Eos,
            generated_text: "hello".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        let options = StreamingOptions::default();
        let events = collect_completion_sse(rx, options).await;

        // Should have token chunk, done chunk, and [DONE]
        assert!(events.len() >= 3);

        // Token chunk should not have logprobs
        assert!(events[0].contains("hello"));
        let json = parse_data_json(&events[0]);
        assert!(json["choices"][0].get("logprobs").is_none());
    }

    #[tokio::test]
    async fn completion_stream_with_logprobs() {
        let tokenizer = Arc::new(TokenizerWrapper::for_testing(1000));
        let (tx, rx) = mpsc::channel(16);
        tx.send(StreamEvent::Token {
            token_id: 42,
            token_text: "hello".to_string(),
            logprob: Some(-0.5),
            top_logprobs: Some(vec![(42, -0.5), (43, -1.2)]),
        })
        .await
        .unwrap();
        tx.send(StreamEvent::Done {
            finish_reason: FinishReason::Eos,
            generated_text: "hello".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        let options = StreamingOptions {
            include_logprobs: true,
            tokenizer: Some(tokenizer),
            ..Default::default()
        };
        let events = collect_completion_sse(rx, options).await;

        let json = parse_data_json(&events[0]);
        let logprobs = &json["choices"][0]["logprobs"];
        assert!(logprobs.is_object());
        assert_eq!(logprobs["tokens"][0], "hello");
        assert!(logprobs["token_logprobs"][0].as_f64().unwrap() < 0.0);
        assert_eq!(logprobs["text_offset"][0], 0);
        assert!(logprobs["top_logprobs"][0].is_object());
    }

    #[tokio::test]
    async fn chat_stream_without_logprobs() {
        let (tx, rx) = mpsc::channel(16);
        tx.send(StreamEvent::Token {
            token_id: 1,
            token_text: "hi".to_string(),
            logprob: None,
            top_logprobs: None,
        })
        .await
        .unwrap();
        tx.send(StreamEvent::Done {
            finish_reason: FinishReason::Eos,
            generated_text: "hi".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        let options = StreamingOptions::default();
        let events = collect_chat_sse(rx, options).await;

        assert!(events.len() >= 3);
        let json = parse_data_json(&events[0]);
        assert!(json["choices"][0].get("logprobs").is_none());
    }

    #[tokio::test]
    async fn chat_stream_with_logprobs() {
        let tokenizer = Arc::new(TokenizerWrapper::for_testing(1000));
        let (tx, rx) = mpsc::channel(16);
        tx.send(StreamEvent::Token {
            token_id: 42,
            token_text: "world".to_string(),
            logprob: Some(-0.3),
            top_logprobs: Some(vec![(42, -0.3), (99, -2.1)]),
        })
        .await
        .unwrap();
        tx.send(StreamEvent::Done {
            finish_reason: FinishReason::Eos,
            generated_text: "world".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        let options = StreamingOptions {
            include_logprobs: true,
            tokenizer: Some(tokenizer),
            ..Default::default()
        };
        let events = collect_chat_sse(rx, options).await;

        let json = parse_data_json(&events[0]);
        let logprobs = &json["choices"][0]["logprobs"];
        assert!(logprobs.is_object());
        assert_eq!(logprobs["content"][0]["token"], "world");
        assert!(logprobs["content"][0]["logprob"].as_f64().unwrap() < 0.0);
        assert!(logprobs["content"][0]["bytes"].is_array());
        assert!(logprobs["content"][0]["top_logprobs"].is_array());
        assert_eq!(
            logprobs["content"][0]["top_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn completion_stream_text_offset_increments() {
        let tokenizer = Arc::new(TokenizerWrapper::for_testing(1000));
        let (tx, rx) = mpsc::channel(16);
        tx.send(StreamEvent::Token {
            token_id: 1,
            token_text: "ab".to_string(),
            logprob: Some(-0.1),
            top_logprobs: None,
        })
        .await
        .unwrap();
        tx.send(StreamEvent::Token {
            token_id: 2,
            token_text: "cde".to_string(),
            logprob: Some(-0.2),
            top_logprobs: None,
        })
        .await
        .unwrap();
        tx.send(StreamEvent::Done {
            finish_reason: FinishReason::Length,
            generated_text: "abcde".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        let options = StreamingOptions {
            include_logprobs: true,
            tokenizer: Some(tokenizer),
            ..Default::default()
        };
        let events = collect_completion_sse(rx, options).await;

        // First token offset = 0
        let json0 = parse_data_json(&events[0]);
        assert_eq!(json0["choices"][0]["logprobs"]["text_offset"][0], 0);

        // Second token offset = 2 (length of "ab")
        let json1 = parse_data_json(&events[1]);
        assert_eq!(json1["choices"][0]["logprobs"]["text_offset"][0], 2);
    }
}
