use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use vllm_core::engine::{GenerationRequest, GenerationResult};
use vllm_core::lora::LoraRequest;
use vllm_core::sampling::{
    BeamSearchConfig, JsonSchemaConstraint, SamplingConstraint, SamplingParams,
};
use vllm_core::tokenizer::TokenizerWrapper;

use super::admin::prometheus;
use super::error::ApiError;
use super::response_format::validate_response_format;
use super::streaming::{completion_sse_stream, StreamingOptions};
use super::types::{
    finish_reason_str, system_fingerprint, timestamp_now, CompletionChoice, CompletionLogProbs,
    CompletionRequest, CompletionResponse, PromptInput, ResponseFormat, Usage,
};
use super::AppState;

pub async fn create_completion(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
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

    super::validation::validate_completion_request(&req)?;

    let inputs = req.prompt.clone().into_inputs();

    let logit_bias = req.logit_bias.as_ref().map(|bias| {
        bias.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect::<Vec<_>>()
    });

    // Tokenize bad_words strings into token ID sequences
    let bad_words_token_ids = req.bad_words.as_ref().map(|words| {
        super::tokenize_bad_words(words, &state.tokenizer)
    });

    if req.stream {
        // Streaming only supports n=1
        if req.n > 1 {
            prometheus::inc_requests_error();
            return Err(ApiError::InvalidRequest(
                "n > 1 is not supported with streaming".to_string(),
            ));
        }

        // Streaming supports single prompt
        let input = inputs
            .into_iter()
            .next()
            .unwrap_or(PromptInput::Text(String::new()));
        let (prompt, prompt_tokens) = resolve_prompt_input(&state, input, req.max_tokens)?;

        let stream_opts_ref = req.stream_options.as_ref();
        let include_usage = stream_opts_ref.is_some_and(|opts| opts.include_usage);
        let continuous_usage_stats =
            stream_opts_ref.is_some_and(|opts| opts.continuous_usage_stats);

        // Capture logprobs count before req fields are moved
        let logprobs_count = req.logprobs;

        // Convert lora_name to LoraRequest (references a pre-loaded adapter by name)
        let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);

        // Create constraint from response_format
        let constraint =
            create_constraint_from_response_format(req.response_format.as_ref(), &state.tokenizer);

        let beam_search = build_beam_config(req.beam_width, req.length_penalty, req.early_stopping);

        let gen_req = GenerationRequest {
            prompt,
            max_new_tokens: req.max_tokens,
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
                logit_bias: logit_bias.clone(),
                min_tokens: req.min_tokens,
                eos_token_id: Some(state.eos_token_id),
                banned_token_ids: req.banned_token_ids.clone(),
                allowed_token_ids: req.allowed_token_ids.clone(),
                bad_words_token_ids: bad_words_token_ids.clone(),
            },
            stop_strings: req.stop,
            stop_token_ids: req.stop_token_ids,
            include_stop_str_in_output: req.include_stop_str_in_output,
            logprobs: logprobs_count,
            echo: req.echo,
            lora_request,
            constraint,
            image_inputs: Vec::new(),
        };

        let rx = state
            .engine
            .get()
            .generate_stream(gen_req)
            .await
            .map_err(|e| ApiError::EngineError(e.to_string()))?;

        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
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
            completion_sse_stream(request_id, state.model_id.clone(), rx, streaming_opts)
                .into_response(),
        )
    } else {
        // Non-streaming supports batch of prompts with n and best_of
        let n = req.n;
        // best_of defaults to n when not specified
        let best_of = req.best_of.unwrap_or(n);
        let user_requested_logprobs = req.logprobs;

        // When best_of > n, we need logprobs to score candidates even if the
        // user didn't ask for them. Request at least 1 logprob internally.
        let internal_logprobs = if best_of > n && user_requested_logprobs.is_none() {
            Some(1)
        } else {
            user_requested_logprobs
        };

        let mut choices = Vec::new();
        let mut total_prompt_tokens = 0;
        let mut total_completion_tokens = 0;
        let mut choice_index = 0u32;

        // Convert lora_name to LoraRequest (for batch requests, same adapter for all)
        let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);

        // Check if response_format requires constraints (used to decide whether
        // to recreate per-iteration)
        let has_constraint =
            create_constraint_from_response_format(req.response_format.as_ref(), &state.tokenizer)
                .is_some();

        for input in inputs {
            let (prompt, prompt_tokens) = resolve_prompt_input(&state, input, req.max_tokens)?;
            total_prompt_tokens += prompt_tokens;

            // Generate `best_of` candidates for this prompt
            let mut candidates: Vec<(GenerationResult, Option<CompletionLogProbs>)> =
                Vec::with_capacity(best_of);

            for _ in 0..best_of {
                // Recreate constraint for each generation call (constraints are stateful)
                let request_constraint = if has_constraint {
                    create_constraint_from_response_format(
                        req.response_format.as_ref(),
                        &state.tokenizer,
                    )
                } else {
                    None
                };

                let iter_beam =
                    build_beam_config(req.beam_width, req.length_penalty, req.early_stopping);

                let gen_req = GenerationRequest {
                    prompt: prompt.clone(),
                    max_new_tokens: req.max_tokens,
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
                    logprobs: internal_logprobs,
                    echo: req.echo,
                    lora_request: lora_request.clone(),
                    constraint: request_constraint,
                    image_inputs: Vec::new(),
                };

                let result = state.engine.get().generate(gen_req).await.map_err(|e| {
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

                let logprobs_data = if internal_logprobs.is_some() {
                    Some(build_logprobs(&result, &state.tokenizer, req.echo))
                } else {
                    None
                };

                candidates.push((result, logprobs_data));
            }

            // Select the top `n` candidates by cumulative log probability
            let selected = select_top_n_candidates(candidates, n);

            for (result, logprobs_data) in selected {
                total_completion_tokens += result.generated_token_ids.len();

                // Strip logprobs from the response if the user didn't request them
                let final_logprobs = if user_requested_logprobs.is_some() {
                    logprobs_data
                } else {
                    None
                };

                choices.push(CompletionChoice {
                    text: result.generated_text,
                    index: choice_index,
                    finish_reason: Some(finish_reason_str(&result.finish_reason)),
                    stop_reason: None,
                    logprobs: final_logprobs,
                    token_ids: None,
                });
                choice_index += 1;
            }
        }

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created: timestamp_now(),
            model: state.model_id.clone(),
            system_fingerprint: system_fingerprint(),
            choices,
            usage: Usage::new(total_prompt_tokens, total_completion_tokens),
            service_tier: None,
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

fn resolve_prompt_input(
    state: &AppState,
    input: PromptInput,
    max_tokens: usize,
) -> Result<(String, usize), ApiError> {
    match input {
        PromptInput::Text(text) => {
            // Early-fail: reject prompts that are clearly too long before tokenizing
            super::validation::validate_prompt_char_length(
                text.len(),
                max_tokens,
                state.max_model_len,
                state.tokenizer.max_chars_per_token(),
            )?;
            let token_count = state
                .tokenizer
                .encode(&text)
                .map(|ids| ids.len())
                .unwrap_or(0);
            Ok((text, token_count))
        }
        PromptInput::TokenIds(ids) => {
            let token_count = ids.len();
            let text = state
                .tokenizer
                .decode(&ids)
                .map_err(|e| ApiError::EngineError(format!("Failed to decode token IDs: {}", e)))?;
            Ok((text, token_count))
        }
    }
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

/// Select the top `n` candidates from a list, ranked by cumulative log probability.
///
/// If `n >= candidates.len()`, returns all candidates in their original order.
/// If `n < candidates.len()`, scores each by cumulative logprob and returns the
/// top `n` sorted by descending score.
fn select_top_n_candidates(
    mut candidates: Vec<(GenerationResult, Option<CompletionLogProbs>)>,
    n: usize,
) -> Vec<(GenerationResult, Option<CompletionLogProbs>)> {
    if n >= candidates.len() {
        return candidates;
    }

    // Score each candidate and sort by descending logprob score
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, (result, _))| (i, sum_token_logprobs(result)))
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

/// Create a sampling constraint from response_format.
fn create_constraint_from_response_format(
    response_format: Option<&ResponseFormat>,
    tokenizer: &Arc<TokenizerWrapper>,
) -> Option<Box<dyn SamplingConstraint>> {
    match response_format {
        None | Some(ResponseFormat::Text) => None,
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

/// Build CompletionLogProbs from GenerationResult.
fn build_logprobs(
    result: &GenerationResult,
    tokenizer: &TokenizerWrapper,
    echo: bool,
) -> CompletionLogProbs {
    let mut text_offset = Vec::new();
    let mut token_logprobs = Vec::new();
    let mut tokens = Vec::new();
    let mut top_logprobs = Vec::new();

    let mut current_offset = 0usize;

    // If echo, include prompt tokens first
    if echo {
        if let Some(ref prompt_ids) = result.prompt_token_ids {
            let prompt_lps = result.prompt_logprobs.as_ref();
            for (i, &token_id) in prompt_ids.iter().enumerate() {
                let token_str = tokenizer
                    .decode(&[token_id])
                    .unwrap_or_else(|_| format!("<unk:{}>", token_id));

                text_offset.push(current_offset);
                current_offset += token_str.len();
                tokens.push(token_str.clone());

                // First prompt token has no logprob (nothing to condition on)
                let lp = if i == 0 {
                    None
                } else {
                    prompt_lps.and_then(|lps| lps.get(i).copied().flatten())
                };
                token_logprobs.push(lp);

                // For prompt tokens: first has None, rest have the token's logprob
                let top = if i == 0 {
                    None
                } else if let Some(logprob) = lp {
                    let mut map = HashMap::new();
                    map.insert(token_str, logprob);
                    Some(map)
                } else {
                    // No logprob available, return empty map
                    Some(HashMap::new())
                };
                top_logprobs.push(top);
            }
        }
    }

    // Include generated tokens
    let gen_token_ids = &result.generated_token_ids;
    let gen_lps = result.token_logprobs.as_ref();
    let gen_top_lps = result.top_logprobs.as_ref();

    for (i, &token_id) in gen_token_ids.iter().enumerate() {
        let token_str = tokenizer
            .decode(&[token_id])
            .unwrap_or_else(|_| format!("<unk:{}>", token_id));

        text_offset.push(current_offset);
        current_offset += token_str.len();
        tokens.push(token_str);

        // Get logprob for this token
        let lp = gen_lps.and_then(|lps| lps.get(i).copied());
        token_logprobs.push(lp);

        // Get top-k logprobs and convert token IDs to strings
        let top = gen_top_lps.and_then(|tops| {
            tops.get(i).map(|token_lps| {
                token_lps
                    .iter()
                    .map(|&(tid, lp)| {
                        let t = tokenizer
                            .decode(&[tid])
                            .unwrap_or_else(|_| format!("<unk:{}>", tid));
                        (t, lp)
                    })
                    .collect::<HashMap<String, f32>>()
            })
        });
        top_logprobs.push(top);
    }

    CompletionLogProbs {
        text_offset,
        token_logprobs,
        tokens,
        top_logprobs,
    }
}

// ─── Render endpoint ────────────────────────────────────────────────

/// Response type for the completions render endpoint.
#[derive(Debug, serde::Serialize)]
pub struct CompletionRenderResponse {
    /// The prompt text (decoded if token IDs were provided).
    pub prompt: String,
    /// Token IDs after tokenization.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub num_tokens: usize,
}

/// Render a completion request: tokenize the prompt but don't generate.
pub async fn render_completion(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let inputs = req.prompt.clone().into_inputs();
    let input = inputs
        .into_iter()
        .next()
        .unwrap_or(PromptInput::Text(String::new()));

    match input {
        PromptInput::Text(text) => {
            let token_ids = state
                .tokenizer
                .encode(&text)
                .map_err(|e| ApiError::InvalidRequest(format!("tokenization failed: {e}")))?;
            let num_tokens = token_ids.len();
            Ok(Json(CompletionRenderResponse {
                prompt: text,
                token_ids,
                num_tokens,
            }))
        }
        PromptInput::TokenIds(ids) => {
            let text = state
                .tokenizer
                .decode(&ids)
                .map_err(|e| ApiError::EngineError(format!("Failed to decode token IDs: {e}")))?;
            let num_tokens = ids.len();
            Ok(Json(CompletionRenderResponse {
                prompt: text,
                token_ids: ids,
                num_tokens,
            }))
        }
    }
}
