use super::error::ApiError;
use super::types::{ChatCompletionRequest, CompletionRequest, ResponseFormat};

pub fn validate_completion_request(req: &CompletionRequest) -> Result<(), ApiError> {
    validate_temperature(req.temperature)?;
    validate_top_p(req.top_p)?;
    validate_frequency_penalty(req.frequency_penalty)?;
    validate_presence_penalty(req.presence_penalty)?;
    validate_max_tokens(req.max_tokens)?;
    validate_n(req.n)?;
    validate_best_of_n(req.best_of, req.n)?;
    validate_beam_search(req.beam_width, req.response_format.as_ref(), req.stream)?;

    if let Some(logprobs) = req.logprobs {
        if logprobs > 5 {
            return Err(ApiError::InvalidRequest(format!(
                "logprobs must be between 0 and 5, got {logprobs}"
            )));
        }
    }

    Ok(())
}

pub fn validate_chat_completion_request(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    validate_temperature(req.temperature)?;
    validate_top_p(req.top_p)?;
    validate_frequency_penalty(req.frequency_penalty)?;
    validate_presence_penalty(req.presence_penalty)?;
    validate_max_tokens(req.max_tokens)?;
    validate_n(req.n)?;
    validate_best_of_n(req.best_of, req.n)?;
    validate_beam_search(req.beam_width, req.response_format.as_ref(), req.stream)?;

    if let Some(top_logprobs) = req.top_logprobs {
        if !req.logprobs.unwrap_or(false) {
            return Err(ApiError::InvalidRequest(
                "top_logprobs requires logprobs to be true".to_string(),
            ));
        }
        if top_logprobs > 20 {
            return Err(ApiError::InvalidRequest(format!(
                "top_logprobs must be between 0 and 20, got {top_logprobs}"
            )));
        }
    }

    Ok(())
}

fn validate_temperature(temperature: f32) -> Result<(), ApiError> {
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::InvalidRequest(format!(
            "temperature must be between 0 and 2, got {temperature}"
        )));
    }
    Ok(())
}

fn validate_top_p(top_p: f32) -> Result<(), ApiError> {
    if top_p <= 0.0 || top_p > 1.0 {
        return Err(ApiError::InvalidRequest(format!(
            "top_p must be between 0 (exclusive) and 1, got {top_p}"
        )));
    }
    Ok(())
}

fn validate_frequency_penalty(frequency_penalty: f32) -> Result<(), ApiError> {
    if !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(ApiError::InvalidRequest(format!(
            "frequency_penalty must be between -2 and 2, got {frequency_penalty}"
        )));
    }
    Ok(())
}

fn validate_presence_penalty(presence_penalty: f32) -> Result<(), ApiError> {
    if !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(ApiError::InvalidRequest(format!(
            "presence_penalty must be between -2 and 2, got {presence_penalty}"
        )));
    }
    Ok(())
}

fn validate_max_tokens(max_tokens: usize) -> Result<(), ApiError> {
    if max_tokens < 1 {
        return Err(ApiError::InvalidRequest(
            "max_tokens must be at least 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_best_of_n(best_of: Option<usize>, n: usize) -> Result<(), ApiError> {
    if let Some(bo) = best_of {
        if bo < 1 {
            return Err(ApiError::InvalidRequest(
                "best_of must be at least 1".to_string(),
            ));
        }
        if bo < n {
            return Err(ApiError::InvalidRequest(format!(
                "best_of ({bo}) must be greater than or equal to n ({n})"
            )));
        }
    }
    Ok(())
}

fn validate_n(n: usize) -> Result<(), ApiError> {
    if n < 1 {
        return Err(ApiError::InvalidRequest("n must be at least 1".to_string()));
    }
    Ok(())
}

fn validate_beam_search(
    beam_width: Option<usize>,
    response_format: Option<&ResponseFormat>,
    _stream: bool,
) -> Result<(), ApiError> {
    let Some(bw) = beam_width else {
        return Ok(());
    };

    if !(1..=16).contains(&bw) {
        return Err(ApiError::InvalidRequest(format!(
            "beam_width must be between 1 and 16, got {bw}"
        )));
    }

    // Beam search is incompatible with structured output constraints
    if let Some(fmt) = response_format {
        match fmt {
            ResponseFormat::Text => {}
            ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. } => {
                return Err(ApiError::InvalidRequest(
                    "beam_search is not compatible with response_format json constraints"
                        .to_string(),
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Helper constructors ──────────────────────────────────────────

    fn minimal_completion_request() -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "hello"
        }))
        .unwrap()
    }

    fn minimal_chat_request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap()
    }

    // ─── Valid defaults pass ──────────────────────────────────────────

    #[test]
    fn valid_defaults_pass_completion() {
        let req = minimal_completion_request();
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn valid_defaults_pass_chat() {
        let req = minimal_chat_request();
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    // ─── Temperature ─────────────────────────────────────────────────

    #[test]
    fn temperature_too_high() {
        let mut req = minimal_completion_request();
        req.temperature = 3.5;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("temperature")));
    }

    #[test]
    fn temperature_negative() {
        let mut req = minimal_completion_request();
        req.temperature = -0.1;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("temperature")));
    }

    #[test]
    fn temperature_zero_passes() {
        let mut req = minimal_completion_request();
        req.temperature = 0.0;
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn temperature_two_passes() {
        let mut req = minimal_completion_request();
        req.temperature = 2.0;
        assert!(validate_completion_request(&req).is_ok());
    }

    // ─── top_p ───────────────────────────────────────────────────────

    #[test]
    fn top_p_zero_fails() {
        let mut req = minimal_completion_request();
        req.top_p = 0.0;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("top_p")));
    }

    #[test]
    fn top_p_one_passes() {
        let mut req = minimal_completion_request();
        req.top_p = 1.0;
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn top_p_above_one_fails() {
        let mut req = minimal_completion_request();
        req.top_p = 1.1;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("top_p")));
    }

    #[test]
    fn top_p_negative_fails() {
        let mut req = minimal_completion_request();
        req.top_p = -0.5;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("top_p")));
    }

    #[test]
    fn top_p_small_positive_passes() {
        let mut req = minimal_completion_request();
        req.top_p = 0.01;
        assert!(validate_completion_request(&req).is_ok());
    }

    // ─── frequency_penalty ───────────────────────────────────────────

    #[test]
    fn frequency_penalty_too_high() {
        let mut req = minimal_completion_request();
        req.frequency_penalty = 5.0;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("frequency_penalty")));
    }

    #[test]
    fn frequency_penalty_too_low() {
        let mut req = minimal_completion_request();
        req.frequency_penalty = -3.0;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("frequency_penalty")));
    }

    #[test]
    fn frequency_penalty_boundary_passes() {
        let mut req = minimal_completion_request();
        req.frequency_penalty = -2.0;
        assert!(validate_completion_request(&req).is_ok());
        req.frequency_penalty = 2.0;
        assert!(validate_completion_request(&req).is_ok());
    }

    // ─── presence_penalty ────────────────────────────────────────────

    #[test]
    fn presence_penalty_too_high() {
        let mut req = minimal_chat_request();
        req.presence_penalty = 2.5;
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("presence_penalty")));
    }

    #[test]
    fn presence_penalty_too_low() {
        let mut req = minimal_chat_request();
        req.presence_penalty = -2.1;
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("presence_penalty")));
    }

    #[test]
    fn presence_penalty_boundary_passes() {
        let mut req = minimal_chat_request();
        req.presence_penalty = -2.0;
        assert!(validate_chat_completion_request(&req).is_ok());
        req.presence_penalty = 2.0;
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    // ─── max_tokens ──────────────────────────────────────────────────

    #[test]
    fn max_tokens_zero_fails() {
        let mut req = minimal_completion_request();
        req.max_tokens = 0;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("max_tokens")));
    }

    #[test]
    fn max_tokens_one_passes() {
        let mut req = minimal_completion_request();
        req.max_tokens = 1;
        assert!(validate_completion_request(&req).is_ok());
    }

    // ─── best_of ─────────────────────────────────────────────────────

    #[test]
    fn best_of_zero_fails() {
        let mut req = minimal_completion_request();
        req.best_of = Some(0);
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("best_of")));
    }

    #[test]
    fn best_of_one_passes() {
        let mut req = minimal_completion_request();
        req.best_of = Some(1);
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn best_of_less_than_n_fails() {
        let mut req = minimal_completion_request();
        req.n = 3;
        req.best_of = Some(2);
        let err = validate_completion_request(&req).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(msg) if msg.contains("best_of") && msg.contains("greater than or equal"))
        );
    }

    #[test]
    fn best_of_equal_to_n_passes() {
        let mut req = minimal_completion_request();
        req.n = 3;
        req.best_of = Some(3);
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn best_of_greater_than_n_passes() {
        let mut req = minimal_completion_request();
        req.n = 2;
        req.best_of = Some(5);
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn best_of_none_with_n_greater_than_one_passes() {
        let mut req = minimal_completion_request();
        req.n = 3;
        req.best_of = None;
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn chat_best_of_less_than_n_fails() {
        let mut req = minimal_chat_request();
        req.n = 4;
        req.best_of = Some(2);
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(msg) if msg.contains("best_of") && msg.contains("greater than or equal"))
        );
    }

    #[test]
    fn chat_best_of_equal_to_n_passes() {
        let mut req = minimal_chat_request();
        req.n = 3;
        req.best_of = Some(3);
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    // ─── n ──────────────────────────────────────────────────────────

    #[test]
    fn n_zero_fails_completion() {
        let mut req = minimal_completion_request();
        req.n = 0;
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("n must")));
    }

    #[test]
    fn n_zero_fails_chat() {
        let mut req = minimal_chat_request();
        req.n = 0;
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("n must")));
    }

    #[test]
    fn n_one_passes_completion() {
        let mut req = minimal_completion_request();
        req.n = 1;
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn n_one_passes_chat() {
        let mut req = minimal_chat_request();
        req.n = 1;
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    // ─── logprobs (completions) ──────────────────────────────────────

    #[test]
    fn logprobs_above_five_fails_completion() {
        let mut req = minimal_completion_request();
        req.logprobs = Some(10);
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("logprobs")));
    }

    #[test]
    fn logprobs_five_passes_completion() {
        let mut req = minimal_completion_request();
        req.logprobs = Some(5);
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn logprobs_none_passes_completion() {
        let mut req = minimal_completion_request();
        req.logprobs = None;
        assert!(validate_completion_request(&req).is_ok());
    }

    // ─── top_logprobs (chat) ─────────────────────────────────────────

    #[test]
    fn top_logprobs_without_logprobs_true_fails() {
        let mut req = minimal_chat_request();
        req.top_logprobs = Some(5);
        req.logprobs = Some(false);
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(msg) if msg.contains("top_logprobs requires logprobs to be true"))
        );
    }

    #[test]
    fn top_logprobs_without_logprobs_field_fails() {
        let mut req = minimal_chat_request();
        req.top_logprobs = Some(5);
        req.logprobs = None;
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(msg) if msg.contains("top_logprobs requires logprobs to be true"))
        );
    }

    #[test]
    fn top_logprobs_above_twenty_fails() {
        let mut req = minimal_chat_request();
        req.logprobs = Some(true);
        req.top_logprobs = Some(21);
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(
            matches!(err, ApiError::InvalidRequest(msg) if msg.contains("top_logprobs must be between 0 and 20"))
        );
    }

    #[test]
    fn top_logprobs_with_logprobs_true_passes() {
        let mut req = minimal_chat_request();
        req.logprobs = Some(true);
        req.top_logprobs = Some(5);
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    #[test]
    fn top_logprobs_zero_with_logprobs_true_passes() {
        let mut req = minimal_chat_request();
        req.logprobs = Some(true);
        req.top_logprobs = Some(0);
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    #[test]
    fn top_logprobs_twenty_with_logprobs_true_passes() {
        let mut req = minimal_chat_request();
        req.logprobs = Some(true);
        req.top_logprobs = Some(20);
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    #[test]
    fn no_top_logprobs_passes() {
        let mut req = minimal_chat_request();
        req.top_logprobs = None;
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    // ─── beam_width (completion) ────────────────────────────────────

    #[test]
    fn beam_width_zero_fails_completion() {
        let mut req = minimal_completion_request();
        req.beam_width = Some(0);
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("beam_width")));
    }

    #[test]
    fn beam_width_above_sixteen_fails_completion() {
        let mut req = minimal_completion_request();
        req.beam_width = Some(17);
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("beam_width")));
    }

    #[test]
    fn beam_width_one_passes_completion() {
        let mut req = minimal_completion_request();
        req.beam_width = Some(1);
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn beam_width_sixteen_passes_completion() {
        let mut req = minimal_completion_request();
        req.beam_width = Some(16);
        assert!(validate_completion_request(&req).is_ok());
    }

    #[test]
    fn beam_width_none_passes_completion() {
        let mut req = minimal_completion_request();
        req.beam_width = None;
        assert!(validate_completion_request(&req).is_ok());
    }

    // ─── beam_width (chat) ──────────────────────────────────────────

    #[test]
    fn beam_width_zero_fails_chat() {
        let mut req = minimal_chat_request();
        req.beam_width = Some(0);
        let err = validate_chat_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("beam_width")));
    }

    #[test]
    fn beam_width_valid_passes_chat() {
        let mut req = minimal_chat_request();
        req.beam_width = Some(4);
        assert!(validate_chat_completion_request(&req).is_ok());
    }

    // ─── beam_width + response_format ───────────────────────────────

    #[test]
    fn beam_width_with_json_object_fails() {
        let mut req = minimal_completion_request();
        req.beam_width = Some(2);
        req.response_format = Some(ResponseFormat::JsonObject);
        let err = validate_completion_request(&req).unwrap_err();
        assert!(matches!(err, ApiError::InvalidRequest(msg) if msg.contains("not compatible")));
    }

    #[test]
    fn beam_width_with_text_format_passes() {
        let mut req = minimal_completion_request();
        req.beam_width = Some(2);
        req.response_format = Some(ResponseFormat::Text);
        assert!(validate_completion_request(&req).is_ok());
    }
}
