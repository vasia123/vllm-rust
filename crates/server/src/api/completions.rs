use std::collections::HashMap;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use vllm_core::engine::{GenerationRequest, GenerationResult};
use vllm_core::lora::LoraRequest;
use vllm_core::sampling::SamplingParams;
use vllm_core::tokenizer::TokenizerWrapper;

use super::error::ApiError;
use super::streaming::completion_sse_stream;
use super::types::{
    finish_reason_str, timestamp_now, CompletionChoice, CompletionLogProbs, CompletionRequest,
    CompletionResponse, PromptInput, Usage,
};
use super::AppState;

pub async fn create_completion(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.model != state.model_id {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found",
            req.model
        )));
    }

    let inputs = req.prompt.clone().into_inputs();

    if req.stream {
        // Streaming only supports single prompt (logprobs not yet supported in streaming)
        let input = inputs
            .into_iter()
            .next()
            .unwrap_or(PromptInput::Text(String::new()));
        let (prompt, prompt_tokens) = resolve_prompt_input(&state, input)?;

        // Convert lora_name to LoraRequest (references a pre-loaded adapter by name)
        let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);

        let gen_req = GenerationRequest {
            prompt,
            max_new_tokens: req.max_tokens,
            eos_token_id: state.eos_token_id,
            sampling_params: SamplingParams {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                repetition_penalty: req.repetition_penalty,
                min_p: req.min_p,
                seed: req.seed,
            },
            stop_strings: req.stop,
            stop_token_ids: req.stop_token_ids,
            include_stop_str_in_output: req.include_stop_str_in_output,
            logprobs: None, // Streaming doesn't support logprobs yet
            echo: false,
            lora_request,
        };

        let _ = prompt_tokens; // Used for non-streaming only

        let rx = state
            .engine
            .get()
            .generate_stream(gen_req)
            .await
            .map_err(|e| ApiError::EngineError(e.to_string()))?;

        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        Ok(completion_sse_stream(request_id, state.model_id.clone(), rx).into_response())
    } else {
        // Non-streaming supports batch of prompts
        let mut choices = Vec::with_capacity(inputs.len());
        let mut total_prompt_tokens = 0;
        let mut total_completion_tokens = 0;

        // Convert lora_name to LoraRequest (for batch requests, same adapter for all)
        let lora_request = req.lora_name.as_ref().map(LoraRequest::by_name);

        for (index, input) in inputs.into_iter().enumerate() {
            let (prompt, prompt_tokens) = resolve_prompt_input(&state, input)?;

            let gen_req = GenerationRequest {
                prompt,
                max_new_tokens: req.max_tokens,
                eos_token_id: state.eos_token_id,
                sampling_params: SamplingParams {
                    temperature: req.temperature,
                    top_p: req.top_p,
                    top_k: req.top_k,
                    repetition_penalty: req.repetition_penalty,
                    min_p: req.min_p,
                    seed: req.seed,
                },
                stop_strings: req.stop.clone(),
                stop_token_ids: req.stop_token_ids.clone(),
                include_stop_str_in_output: req.include_stop_str_in_output,
                logprobs: req.logprobs,
                echo: req.echo,
                lora_request: lora_request.clone(),
            };

            let result = state
                .engine
                .get()
                .generate(gen_req)
                .await
                .map_err(|e| ApiError::EngineError(e.to_string()))?;

            total_prompt_tokens += prompt_tokens;
            total_completion_tokens += result.generated_token_ids.len();

            // Build logprobs if requested
            let logprobs = if req.logprobs.is_some() {
                Some(build_logprobs(&result, &state.tokenizer, req.echo))
            } else {
                None
            };

            choices.push(CompletionChoice {
                text: result.generated_text,
                index: index as u32,
                finish_reason: Some(finish_reason_str(&result.finish_reason)),
                logprobs,
            });
        }

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created: timestamp_now(),
            model: state.model_id.clone(),
            choices,
            usage: Usage {
                prompt_tokens: total_prompt_tokens,
                completion_tokens: total_completion_tokens,
                total_tokens: total_prompt_tokens + total_completion_tokens,
            },
        };

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
