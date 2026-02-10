use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use vllm_core::engine::{GenerationRequest, GenerationResult};
use vllm_core::lora::LoraRequest;
use vllm_core::multimodal::{ContentPart, ImageData};
use vllm_core::sampling::{
    BeamSearchConfig, JsonSchemaConstraint, SamplingConstraint, SamplingParams,
};
use vllm_core::tokenizer::{MessageContent, TokenizerWrapper};

use super::admin::prometheus;
use super::error::ApiError;
use super::response_format::{inject_json_system_prompt, validate_response_format};
use super::streaming::{chat_completion_sse_stream, StreamingOptions};
use super::types::{
    finish_reason_str, system_fingerprint, timestamp_now, ChatCompletionChoice,
    ChatCompletionRequest, ChatCompletionResponse, ChatLogProbToken, ChatLogProbs,
    ChatMessageResponse, ChatTopLogProb, ResponseFormat, Usage,
};
use super::AppState;

pub async fn create_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let start_time = Instant::now();
    prometheus::inc_requests_total();

    if req.model != state.model_id {
        prometheus::inc_requests_error();
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    super::validation::validate_chat_completion_request(&req)?;

    let chat_template = state.chat_template.as_ref().ok_or_else(|| {
        prometheus::inc_requests_error();
        ApiError::TemplateError("no chat template available".to_string())
    })?;

    // Extract image inputs from multimodal messages before template application
    let image_inputs = extract_image_inputs(&req.messages).inspect_err(|_| {
        prometheus::inc_requests_error();
    })?;

    // Inject system prompt for JSON response format enforcement.
    // This must happen before template application so the system message
    // is included in the rendered prompt.
    let mut messages = req.messages.clone();
    inject_json_system_prompt(&mut messages, req.response_format.as_ref());

    // Apply template with tools if provided
    let prompt = chat_template
        .apply_with_tools(&messages, req.tools.as_deref(), true)
        .map_err(|e| ApiError::TemplateError(e.to_string()))?;

    // Early-fail: reject prompts that are clearly too long before tokenizing
    let max_tokens = req.effective_max_tokens();
    super::validation::validate_prompt_char_length(
        prompt.len(),
        max_tokens,
        state.max_model_len,
        state.tokenizer.max_chars_per_token(),
    )?;

    // Determine if we should parse tool calls from the output
    let has_tools = req.tools.is_some();

    // Convert lora_name to LoraRequest (references a pre-loaded adapter by name)
    let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);

    // Determine logprobs count: if logprobs=true, use top_logprobs (default 1, max 20)
    let logprobs_count = if req.logprobs.unwrap_or(false) {
        Some(req.top_logprobs.unwrap_or(1).min(20))
    } else {
        None
    };

    // Create constraint from response_format
    let constraint =
        create_constraint_from_response_format(req.response_format.as_ref(), &state.tokenizer);

    // Convert logit_bias once, before branching on stream vs non-stream
    let logit_bias = req.logit_bias.as_ref().map(|bias| {
        bias.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect::<Vec<_>>()
    });

    // Tokenize bad_words strings into token ID sequences
    let bad_words_token_ids = req
        .bad_words
        .as_ref()
        .map(|words| super::tokenize_bad_words(words, &state.tokenizer));

    if req.stream {
        // Streaming only supports n=1
        if req.n > 1 {
            prometheus::inc_requests_error();
            return Err(ApiError::InvalidRequest(
                "n > 1 is not supported with streaming".to_string(),
            ));
        }

        let beam_search = build_beam_config(req.beam_width, req.length_penalty, req.early_stopping);

        let gen_req = GenerationRequest {
            prompt: prompt.clone(),
            max_new_tokens: max_tokens,
            eos_token_id: state.eos_token_id,
            sampling_params: SamplingParams {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                repetition_penalty: req.repetition_penalty,
                frequency_penalty: req.frequency_penalty,
                presence_penalty: req.presence_penalty,
                min_p: req.min_p,
                seed: req.seed,
                beam_search,
                logit_bias,
                min_tokens: req.min_tokens,
                eos_token_id: Some(state.eos_token_id),
                banned_token_ids: req.banned_token_ids.clone(),
                allowed_token_ids: req.allowed_token_ids.clone(),
                bad_words_token_ids: bad_words_token_ids.clone(),
            },
            stop_strings: req.stop,
            stop_token_ids: req.stop_token_ids,
            include_stop_str_in_output: req.include_stop_str_in_output,
            ignore_eos: req.ignore_eos,
            logprobs: logprobs_count,
            echo: false,
            lora_request,
            constraint,
            image_inputs,
        };

        // Compute prompt tokens for streaming usage reporting
        let stream_opts_ref = req.stream_options.as_ref();
        let include_usage = stream_opts_ref.is_some_and(|opts| opts.include_usage);
        let continuous_usage_stats =
            stream_opts_ref.is_some_and(|opts| opts.continuous_usage_stats);
        let prompt_tokens = if include_usage || continuous_usage_stats {
            state
                .tokenizer
                .encode(&prompt)
                .map(|ids| ids.len())
                .unwrap_or(0)
        } else {
            0
        };

        let rx = state
            .engine
            .get()
            .generate_stream(gen_req)
            .await
            .map_err(|e| {
                prometheus::inc_requests_error();
                ApiError::EngineError(e.to_string())
            })?;

        let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let include_logprobs = logprobs_count.is_some();
        let streaming_opts = StreamingOptions {
            include_usage,
            continuous_usage_stats,
            prompt_tokens,
            include_logprobs,
            tokenizer: if include_logprobs {
                Some(state.tokenizer.clone())
            } else {
                None
            },
            return_token_ids: req.return_tokens_as_token_ids.unwrap_or(false),
        };
        Ok(
            chat_completion_sse_stream(request_id, state.model_id.clone(), rx, streaming_opts)
                .into_response(),
        )
    } else {
        let n = req.n;
        // best_of defaults to n when not specified
        let best_of = req.best_of.unwrap_or(n);

        let prompt_tokens = state
            .tokenizer
            .encode(&prompt)
            .map(|ids| ids.len())
            .unwrap_or(0);

        // When best_of > n, we need logprobs to score candidates even if the
        // user didn't ask for them. Request at least 1 top logprob internally.
        let internal_logprobs_count = if best_of > n && logprobs_count.is_none() {
            Some(1)
        } else {
            logprobs_count
        };

        // Generate `best_of` candidates
        let mut candidates: Vec<GenerationResult> = Vec::with_capacity(best_of);

        for _ in 0..best_of {
            // Recreate constraint for each iteration (constraints are stateful)
            let iter_constraint = create_constraint_from_response_format(
                req.response_format.as_ref(),
                &state.tokenizer,
            );

            let iter_beam =
                build_beam_config(req.beam_width, req.length_penalty, req.early_stopping);

            let iter_gen_req = GenerationRequest {
                prompt: prompt.clone(),
                max_new_tokens: max_tokens,
                eos_token_id: state.eos_token_id,
                sampling_params: SamplingParams {
                    temperature: req.temperature,
                    top_p: req.top_p,
                    top_k: req.top_k,
                    repetition_penalty: req.repetition_penalty,
                    frequency_penalty: req.frequency_penalty,
                    presence_penalty: req.presence_penalty,
                    min_p: req.min_p,
                    seed: req.seed,
                    beam_search: iter_beam,
                    logit_bias: logit_bias.clone(),
                    min_tokens: req.min_tokens,
                    eos_token_id: Some(state.eos_token_id),
                    banned_token_ids: req.banned_token_ids.clone(),
                    allowed_token_ids: req.allowed_token_ids.clone(),
                    bad_words_token_ids: bad_words_token_ids.clone(),
                },
                stop_strings: req.stop.clone(),
                stop_token_ids: req.stop_token_ids.clone(),
                include_stop_str_in_output: req.include_stop_str_in_output,
                ignore_eos: req.ignore_eos,
                logprobs: internal_logprobs_count,
                echo: false,
                lora_request: lora_request.clone(),
                constraint: iter_constraint,
                image_inputs: image_inputs.clone(),
            };

            let result = state
                .engine
                .get()
                .generate(iter_gen_req)
                .await
                .map_err(|e| {
                    prometheus::inc_requests_error();
                    ApiError::EngineError(e.to_string())
                })?;

            // Validate output against response_format after generation completes.
            if let Err(e) =
                validate_response_format(&result.generated_text, req.response_format.as_ref())
            {
                prometheus::inc_requests_error();
                return Err(ApiError::InvalidRequest(e.to_string()));
            }

            candidates.push(result);
        }

        // Select the top `n` candidates by cumulative log probability
        let selected = select_top_n_chat_candidates(candidates, n);

        let mut choices = Vec::with_capacity(n);
        let mut total_completion_tokens = 0;

        for (i, result) in selected.into_iter().enumerate() {
            total_completion_tokens += result.generated_token_ids.len();

            // Parse tool calls if tools were provided
            let (content, tool_calls, finish_reason) = if has_tools {
                let parser = &state.tool_call_parser;
                if let Ok(calls) = parser.parse(&result.generated_text) {
                    if !calls.is_empty() {
                        let content = parser.extract_content(&result.generated_text);
                        (content, Some(calls), "tool_calls".to_string())
                    } else {
                        (
                            Some(result.generated_text.clone()),
                            None,
                            finish_reason_str(&result.finish_reason),
                        )
                    }
                } else {
                    (
                        Some(result.generated_text.clone()),
                        None,
                        finish_reason_str(&result.finish_reason),
                    )
                }
            } else {
                (
                    Some(result.generated_text.clone()),
                    None,
                    finish_reason_str(&result.finish_reason),
                )
            };

            // Build logprobs if the user requested them
            let logprobs = if logprobs_count.is_some() {
                Some(build_chat_logprobs(&result, &state.tokenizer))
            } else {
                None
            };

            choices.push(ChatCompletionChoice {
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content,
                    refusal: None,
                    reasoning: None,
                    reasoning_content: None,
                    tool_calls,
                    annotations: None,
                    audio: None,
                },
                index: i as u32,
                finish_reason: Some(finish_reason),
                stop_reason: result.stop_reason.map(serde_json::Value::from),
                logprobs,
                token_ids: None,
            });
        }

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion",
            created: timestamp_now(),
            model: state.model_id.clone(),
            system_fingerprint: system_fingerprint(),
            choices,
            usage: Usage::new(prompt_tokens, total_completion_tokens),
            service_tier: req.service_tier.clone(),
            prompt_logprobs: None,
            prompt_token_ids: None,
        };

        // Record metrics
        let elapsed = start_time.elapsed().as_secs_f64();
        prometheus::observe_e2e_latency(elapsed);
        if total_completion_tokens > 0 && elapsed > 0.0 {
            prometheus::observe_tps(total_completion_tokens as f64 / elapsed);
        }
        prometheus::inc_requests_success();

        Ok(Json(response).into_response())
    }
}

/// Select the top `n` candidates from a list, ranked by cumulative log probability.
///
/// If `n >= candidates.len()`, returns all candidates in their original order.
/// If `n < candidates.len()`, scores each by cumulative logprob and returns the
/// top `n` sorted by descending score.
fn select_top_n_chat_candidates(
    mut candidates: Vec<GenerationResult>,
    n: usize,
) -> Vec<GenerationResult> {
    if n >= candidates.len() {
        return candidates;
    }

    // Score each candidate and sort by descending logprob score
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, result)| (i, sum_token_logprobs(result)))
        .collect();

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top n indices (sorted descending by index so swap_remove doesn't shift)
    let mut top_indices: Vec<usize> = scored.iter().take(n).map(|(i, _)| *i).collect();
    top_indices.sort_unstable_by(|a, b| b.cmp(a));

    let mut selected = Vec::with_capacity(n);
    for idx in top_indices {
        selected.push(candidates.swap_remove(idx));
    }
    // Reverse to restore original relative ordering (highest score first)
    selected.reverse();
    selected
}

/// Score a generation result by the sum of its token logprobs.
/// Higher (less negative) scores indicate more confident completions.
fn sum_token_logprobs(result: &GenerationResult) -> f64 {
    result
        .token_logprobs
        .as_ref()
        .map(|lps| lps.iter().map(|&lp| lp as f64).sum())
        .unwrap_or(f64::NEG_INFINITY)
}

/// Build a `BeamSearchConfig` from optional API parameters.
fn build_beam_config(
    beam_width: Option<usize>,
    length_penalty: Option<f32>,
    early_stopping: Option<bool>,
) -> Option<BeamSearchConfig> {
    beam_width.map(|bw| BeamSearchConfig {
        beam_width: bw,
        length_penalty: length_penalty.unwrap_or(1.0),
        early_stopping: early_stopping.unwrap_or(false),
        ..Default::default()
    })
}

/// Build ChatLogProbs from GenerationResult (OpenAI chat completion format).
fn build_chat_logprobs(result: &GenerationResult, tokenizer: &TokenizerWrapper) -> ChatLogProbs {
    let mut content = Vec::new();

    let gen_token_ids = &result.generated_token_ids;
    let gen_lps = result.token_logprobs.as_ref();
    let gen_top_lps = result.top_logprobs.as_ref();

    for (i, &token_id) in gen_token_ids.iter().enumerate() {
        let token_str = tokenizer
            .decode(&[token_id])
            .unwrap_or_else(|_| format!("<unk:{}>", token_id));

        // Get logprob for this token (default to 0.0 if not available)
        let logprob = gen_lps.and_then(|lps| lps.get(i).copied()).unwrap_or(0.0);

        // Get UTF-8 bytes of the token
        let bytes = Some(token_str.as_bytes().to_vec());

        // Get top-k logprobs and convert to ChatTopLogProb format
        let top_logprobs = gen_top_lps.and_then(|tops| {
            tops.get(i).map(|token_lps| {
                token_lps
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
            })
        });

        content.push(ChatLogProbToken {
            token: token_str,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs { content }
}

/// Create a sampling constraint from response_format.
pub(crate) fn create_constraint_from_response_format(
    response_format: Option<&ResponseFormat>,
    tokenizer: &Arc<TokenizerWrapper>,
) -> Option<Box<dyn SamplingConstraint>> {
    match response_format {
        None | Some(ResponseFormat::Text) | Some(ResponseFormat::StructuralTag { .. }) => None,
        Some(ResponseFormat::JsonObject) => {
            // Basic JSON object constraint
            let schema = serde_json::json!({"type": "object"});
            Some(Box::new(JsonSchemaConstraint::new(
                schema,
                tokenizer.clone(),
            )))
        }
        Some(ResponseFormat::JsonSchema { json_schema }) => Some(Box::new(
            JsonSchemaConstraint::new(json_schema.schema.clone(), tokenizer.clone()),
        )),
    }
}

/// Extract image inputs from chat messages with multimodal content.
///
/// Scans all messages for image content parts and converts them to `ImageData`
/// objects. Supports both data URIs (base64-encoded inline images) and regular
/// URLs (to be fetched by the model processor later).
fn extract_image_inputs(
    messages: &[vllm_core::tokenizer::ChatMessage],
) -> Result<Vec<ImageData>, ApiError> {
    let mut images = Vec::new();

    for message in messages {
        match &message.content {
            MessageContent::Text(_) => {}
            MessageContent::Parts(parts) => {
                for part in parts {
                    if let ContentPart::Image { image_url } = part {
                        let image_data = parse_image_url(&image_url.url)?;
                        images.push(image_data);
                    }
                }
            }
        }
    }

    Ok(images)
}

/// Parse an image URL into `ImageData`.
///
/// Handles two cases:
/// - Data URIs: `data:<media_type>;base64,<encoded_data>` - decoded to raw bytes
/// - Regular URLs: `https://...` - stored as-is for later fetching
fn parse_image_url(url: &str) -> Result<ImageData, ApiError> {
    if let Some(data_part) = url.strip_prefix("data:") {
        // Parse data URI: data:<media_type>;base64,<data>
        let (_, base64_data) = data_part.split_once(";base64,").ok_or_else(|| {
            ApiError::InvalidRequest(
                "invalid data URI: expected format 'data:<media_type>;base64,<data>'".to_string(),
            )
        })?;

        let decoded = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            base64_data.trim(),
        )
        .map_err(|e| ApiError::InvalidRequest(format!("invalid base64 image data: {e}")))?;

        Ok(ImageData::bytes(decoded))
    } else {
        // Regular URL - store for later fetching by the multimodal processor
        Ok(ImageData::url(url))
    }
}

// ─── Render endpoint ────────────────────────────────────────────────

/// Response type for the render endpoint.
#[derive(Debug, serde::Serialize)]
pub struct RenderResponse {
    /// The rendered prompt text after template application.
    pub prompt: String,
    /// Token IDs after tokenization.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub num_tokens: usize,
}

/// Render a chat completion request: apply the chat template and tokenize,
/// but don't generate. Useful for debugging and validation.
pub async fn render_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let chat_template = state
        .chat_template
        .as_ref()
        .ok_or_else(|| ApiError::TemplateError("no chat template available".to_string()))?;

    let mut messages = req.messages.clone();
    super::response_format::inject_json_system_prompt(&mut messages, req.response_format.as_ref());

    let prompt = chat_template
        .apply_with_tools(&messages, req.tools.as_deref(), true)
        .map_err(|e| ApiError::TemplateError(e.to_string()))?;

    let token_ids = state
        .tokenizer
        .encode(&prompt)
        .map_err(|e| ApiError::InvalidRequest(format!("tokenization failed: {e}")))?;

    let num_tokens = token_ids.len();

    Ok(Json(RenderResponse {
        prompt,
        token_ids,
        num_tokens,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_core::tokenizer::ChatMessage;

    #[test]
    fn extract_images_from_text_only_messages() {
        let messages = vec![ChatMessage::new("user", "Hello, world!")];
        let images = extract_image_inputs(&messages).expect("should succeed");
        assert!(images.is_empty());
    }

    #[test]
    fn extract_images_from_multimodal_message() {
        let messages = vec![ChatMessage::multimodal(
            "user",
            vec![
                ContentPart::text("What is in this image?"),
                ContentPart::image_url("https://example.com/cat.jpg"),
            ],
        )];
        let images = extract_image_inputs(&messages).expect("should succeed");
        assert_eq!(images.len(), 1);
        assert!(!images[0].is_embedding());
    }

    #[test]
    fn extract_images_from_base64_data_uri() {
        // Minimal valid base64 data (represents a few bytes)
        let b64 =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, b"fake-png-data");
        let data_uri = format!("data:image/png;base64,{b64}");

        let messages = vec![ChatMessage::multimodal(
            "user",
            vec![
                ContentPart::text("Describe this:"),
                ContentPart::image_url(&data_uri),
            ],
        )];
        let images = extract_image_inputs(&messages).expect("should succeed");
        assert_eq!(images.len(), 1);
        // Should have been decoded to bytes
        match &images[0].source {
            vllm_core::multimodal::ImageSource::Bytes(bytes) => {
                assert_eq!(bytes, b"fake-png-data");
            }
            other => panic!("expected Bytes, got {:?}", other),
        }
    }

    #[test]
    fn extract_images_invalid_base64_returns_error() {
        let messages = vec![ChatMessage::multimodal(
            "user",
            vec![ContentPart::image_url(
                "data:image/png;base64,not-valid-base64!!!",
            )],
        )];
        let result = extract_image_inputs(&messages);
        assert!(result.is_err());
    }

    #[test]
    fn extract_images_invalid_data_uri_format_returns_error() {
        let messages = vec![ChatMessage::multimodal(
            "user",
            vec![ContentPart::image_url("data:image/png,no-base64-marker")],
        )];
        let result = extract_image_inputs(&messages);
        assert!(result.is_err());
    }

    #[test]
    fn extract_multiple_images_from_multiple_messages() {
        let messages = vec![
            ChatMessage::multimodal(
                "user",
                vec![
                    ContentPart::text("Compare these images:"),
                    ContentPart::image_url("https://example.com/img1.jpg"),
                    ContentPart::image_url("https://example.com/img2.jpg"),
                ],
            ),
            ChatMessage::new("assistant", "They look similar."),
            ChatMessage::multimodal(
                "user",
                vec![
                    ContentPart::text("Now look at this one:"),
                    ContentPart::image_url("https://example.com/img3.jpg"),
                ],
            ),
        ];
        let images = extract_image_inputs(&messages).expect("should succeed");
        assert_eq!(images.len(), 3);
    }

    #[test]
    fn parse_image_url_regular_url() {
        let result = parse_image_url("https://example.com/image.jpg").expect("should succeed");
        match &result.source {
            vllm_core::multimodal::ImageSource::Url(url) => {
                assert_eq!(url, "https://example.com/image.jpg");
            }
            other => panic!("expected Url, got {:?}", other),
        }
    }

    #[test]
    fn parse_image_url_data_uri() {
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, b"\x89PNG");
        let uri = format!("data:image/png;base64,{b64}");
        let result = parse_image_url(&uri).expect("should succeed");
        match &result.source {
            vllm_core::multimodal::ImageSource::Bytes(bytes) => {
                assert_eq!(bytes, b"\x89PNG");
            }
            other => panic!("expected Bytes, got {:?}", other),
        }
    }
}
