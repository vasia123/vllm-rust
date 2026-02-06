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

    if req.stream {
        // Streaming supports single prompt
        let input = inputs
            .into_iter()
            .next()
            .unwrap_or(PromptInput::Text(String::new()));
        let (prompt, prompt_tokens) = resolve_prompt_input(&state, input)?;

        let include_usage = req
            .stream_options
            .as_ref()
            .is_some_and(|opts| opts.include_usage);

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
            prompt_tokens,
            include_logprobs,
            tokenizer: if include_logprobs {
                Some(state.tokenizer.clone())
            } else {
                None
            },
        };
        Ok(
            completion_sse_stream(request_id, state.model_id.clone(), rx, streaming_opts)
                .into_response(),
        )
    } else {
        // Non-streaming supports batch of prompts
        let best_of = req.best_of;
        let user_requested_logprobs = req.logprobs;

        // When best_of > 1, we need logprobs to score candidates even if the
        // user didn't ask for them. Request at least 1 logprob internally.
        let internal_logprobs = if best_of > 1 && user_requested_logprobs.is_none() {
            Some(1)
        } else {
            user_requested_logprobs
        };

        let mut choices = Vec::with_capacity(inputs.len());
        let mut total_prompt_tokens = 0;
        let mut total_completion_tokens = 0;

        // Convert lora_name to LoraRequest (for batch requests, same adapter for all)
        let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);

        // Check if response_format requires constraints (used to decide whether
        // to recreate per-iteration)
        let has_constraint =
            create_constraint_from_response_format(req.response_format.as_ref(), &state.tokenizer)
                .is_some();

        for (index, input) in inputs.into_iter().enumerate() {
            let (prompt, prompt_tokens) = resolve_prompt_input(&state, input)?;
            total_prompt_tokens += prompt_tokens;

            // Generate `best_of` candidates for this prompt and pick the best
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

            // Select the best candidate by sum of token logprobs
            let best_idx = if best_of > 1 {
                candidates
                    .iter()
                    .enumerate()
                    .max_by(|(_, (a, _)), (_, (b, _))| {
                        let score_a = sum_token_logprobs(a);
                        let score_b = sum_token_logprobs(b);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                0
            };

            let (best_result, best_logprobs) = candidates.swap_remove(best_idx);

            total_completion_tokens += best_result.generated_token_ids.len();

            // Strip logprobs from the response if the user didn't request them
            let final_logprobs = if user_requested_logprobs.is_some() {
                best_logprobs
            } else {
                None
            };

            choices.push(CompletionChoice {
                text: best_result.generated_text,
                index: index as u32,
                finish_reason: Some(finish_reason_str(&best_result.finish_reason)),
                logprobs: final_logprobs,
            });
        }

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created: timestamp_now(),
            model: state.model_id.clone(),
            system_fingerprint: system_fingerprint(),
            choices,
            usage: Usage {
                prompt_tokens: total_prompt_tokens,
                completion_tokens: total_completion_tokens,
                total_tokens: total_prompt_tokens + total_completion_tokens,
            },
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

fn resolve_prompt_input(state: &AppState, input: PromptInput) -> Result<(String, usize), ApiError> {
    match input {
        PromptInput::Text(text) => {
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
