use crate::kv_cache::BlockTable;
use crate::lora::LoraRequest;
use crate::prompt_adapter::PromptAdapterRequest;
use crate::sampling::{SamplerState, SamplingConstraint, SamplingParams};

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    Prefilling,
    Decoding,
    Preempted,
    FinishedEos,
    FinishedLength,
    /// Finished due to a stop token or stop string match.
    FinishedStopped,
}

impl RequestStatus {
    pub fn is_finished(self) -> bool {
        matches!(
            self,
            Self::FinishedEos | Self::FinishedLength | Self::FinishedStopped
        )
    }

    pub fn is_running(self) -> bool {
        matches!(self, Self::Prefilling | Self::Decoding)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    Length,
    Stop,
}

pub struct SequenceState {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub block_table: BlockTable,
    pub status: RequestStatus,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub seqlen_offset: usize,
    pub arrival_order: u64,
    pub sampling_params: SamplingParams,
    pub sampler_state: SamplerState,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub include_stop_str_in_output: bool,
    /// When true, EOS token does not trigger generation stop.
    pub ignore_eos: bool,
    /// Number of prompt tokens already processed (for chunked prefill).
    pub num_computed_tokens: usize,
    /// Number of top logprobs to return (None = no logprobs).
    pub num_top_logprobs: Option<usize>,
    /// Log probability of each generated token.
    pub token_logprobs: Vec<f32>,
    /// Top-k logprobs for each generated token.
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
    /// Log probability of each prompt token (for echo mode).
    pub prompt_logprobs: Vec<Option<f32>>,
    /// Whether to include prompt tokens and their logprobs in output.
    pub echo: bool,
    /// LoRA adapter to use for this request (None = no adapter).
    pub lora_request: Option<LoraRequest>,
    /// Prompt adapter (soft prompt tuning) for this request (None = no adapter).
    pub prompt_adapter_request: Option<PromptAdapterRequest>,
    /// Sampling constraint for structured output (JSON schema, regex, etc.).
    pub constraint: Option<Box<dyn SamplingConstraint>>,
    /// How many times this request has been preempted-and-requeued after a
    /// CUDA OOM during prefill. Bounds the retry loop so a request that
    /// genuinely cannot fit (even with no concurrent load) eventually fails
    /// cleanly instead of busy-looping. Reset implicitly on success.
    pub oom_retries: u32,
}

impl SequenceState {
    pub fn new(
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: u32,
        block_size: usize,
        arrival_order: u64,
    ) -> Self {
        Self {
            request_id,
            prompt_token_ids,
            generated_token_ids: Vec::new(),
            block_table: BlockTable::new(block_size),
            status: RequestStatus::Waiting,
            max_new_tokens,
            eos_token_id,
            seqlen_offset: 0,
            arrival_order,
            sampling_params: SamplingParams::default(),
            sampler_state: SamplerState::new(None),
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            include_stop_str_in_output: false,
            ignore_eos: false,
            num_computed_tokens: 0,
            num_top_logprobs: None,
            token_logprobs: Vec::new(),
            top_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            echo: false,
            lora_request: None,
            prompt_adapter_request: None,
            constraint: None,
            oom_retries: 0,
        }
    }

    pub fn num_generated(&self) -> usize {
        self.generated_token_ids.len()
    }

    /// Length of the full token sequence the model must hold KV for:
    /// prompt + everything generated so far. Recompute-preemption keeps
    /// `prompt_token_ids` and `generated_token_ids` SEPARATE (vLLM model)
    /// and re-prefills this full length, so scheduling, block sizing, and
    /// the prefill chunk all reason over `total_len()`, not `prompt` alone.
    /// For a fresh (not-yet-generated) request this equals the prompt
    /// length, so the common path is unchanged.
    pub fn total_len(&self) -> usize {
        self.prompt_token_ids.len() + self.generated_token_ids.len()
    }

    /// Tokens of the full sequence (prompt followed by generated) in the
    /// half-open range `[start, end)`. Used by prefill — including the
    /// re-prefill of a preempted-and-resumed request, whose chunk may span
    /// the prompt→generated boundary. Out-of-range indices are skipped.
    pub fn token_window(&self, start: usize, end: usize) -> Vec<u32> {
        let plen = self.prompt_token_ids.len();
        (start..end)
            .filter_map(|i| {
                if i < plen {
                    self.prompt_token_ids.get(i).copied()
                } else {
                    self.generated_token_ids.get(i - plen).copied()
                }
            })
            .collect()
    }

    pub fn num_blocks(&self) -> usize {
        self.block_table.block_ids().len()
    }

    /// Blocks needed for the next step (prefill or single decode token).
    ///
    /// For beam search requests in decode phase, each beam may independently
    /// need a new block, so the estimate is multiplied by `beam_width`.
    pub fn blocks_needed_for_step(&self) -> usize {
        if self.status == RequestStatus::Waiting || self.status == RequestStatus::Preempted {
            // Full sequence (prompt + already-generated) must be re-prefilled
            // for a resumed request; equals the prompt for a fresh one.
            self.block_table.blocks_needed(self.total_len())
        } else {
            let beam_width = self
                .sampling_params
                .beam_search
                .as_ref()
                .map(|b| b.beam_width)
                .unwrap_or(1);
            self.block_table.blocks_needed(1) * beam_width
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_is_finished() {
        assert!(!RequestStatus::Waiting.is_finished());
        assert!(!RequestStatus::Prefilling.is_finished());
        assert!(!RequestStatus::Decoding.is_finished());
        assert!(!RequestStatus::Preempted.is_finished());
        assert!(RequestStatus::FinishedEos.is_finished());
        assert!(RequestStatus::FinishedLength.is_finished());
        assert!(RequestStatus::FinishedStopped.is_finished());
    }

    #[test]
    fn status_is_running() {
        assert!(!RequestStatus::Waiting.is_running());
        assert!(RequestStatus::Prefilling.is_running());
        assert!(RequestStatus::Decoding.is_running());
        assert!(!RequestStatus::Preempted.is_running());
        assert!(!RequestStatus::FinishedEos.is_running());
        assert!(!RequestStatus::FinishedLength.is_running());
        assert!(!RequestStatus::FinishedStopped.is_running());
    }

    #[test]
    fn new_sequence_state() {
        let state = SequenceState::new(42, vec![1, 2, 3, 4, 5], 10, 151645, 16, 0);
        assert_eq!(state.request_id, 42);
        assert_eq!(state.status, RequestStatus::Waiting);
        assert_eq!(state.num_generated(), 0);
        assert_eq!(state.seqlen_offset, 0);
    }

    #[test]
    fn blocks_needed_for_prefill() {
        let state = SequenceState::new(0, vec![0; 20], 10, 0, 16, 0);
        // 20 tokens with block_size=16 needs ceil(20/16)=2 blocks
        assert_eq!(state.blocks_needed_for_step(), 2);
    }

    #[test]
    fn total_len_counts_prompt_plus_generated() {
        let mut state = SequenceState::new(0, vec![1, 2, 3], 10, 0, 16, 0);
        assert_eq!(state.total_len(), 3, "fresh: prompt only");
        state.generated_token_ids = vec![7, 8];
        assert_eq!(state.total_len(), 5, "resumed: prompt + generated");
    }

    #[test]
    fn token_window_spans_prompt_and_generated() {
        let mut state = SequenceState::new(0, vec![10, 11, 12], 10, 0, 16, 0);
        // Fresh: window is a prompt slice.
        assert_eq!(state.token_window(0, 3), vec![10, 11, 12]);
        assert_eq!(state.token_window(1, 3), vec![11, 12]);
        // Resumed: generated appended logically after the prompt.
        state.generated_token_ids = vec![20, 21];
        assert_eq!(state.token_window(0, 5), vec![10, 11, 12, 20, 21]);
        // A chunk crossing the prompt→generated boundary.
        assert_eq!(state.token_window(2, 5), vec![12, 20, 21]);
        // Generated-only tail.
        assert_eq!(state.token_window(3, 5), vec![20, 21]);
    }

    #[test]
    fn blocks_needed_for_resumed_request_uses_total_len() {
        // Preempted/resumed: prompt 10 + generated 12 = 22 tokens →
        // ceil(22/16) = 2 blocks (must reserve for the full re-prefill).
        let mut state = SequenceState::new(0, vec![0; 10], 64, 0, 16, 0);
        state.generated_token_ids = vec![0; 12];
        state.status = RequestStatus::Preempted;
        assert_eq!(state.blocks_needed_for_step(), 2);
    }

    #[test]
    fn blocks_needed_for_decode() {
        let mut state = SequenceState::new(0, vec![0; 5], 10, 0, 16, 0);
        state.status = RequestStatus::Decoding;
        state.block_table.append_blocks(&[0]);
        state.block_table.advance(5);
        state.seqlen_offset = 5;
        // 5 tokens in block of 16, decode needs 1 more → fits, 0 new blocks
        assert_eq!(state.blocks_needed_for_step(), 0);
    }

    #[test]
    fn blocks_needed_for_decode_boundary() {
        let mut state = SequenceState::new(0, vec![0; 16], 10, 0, 16, 0);
        state.status = RequestStatus::Decoding;
        state.block_table.append_blocks(&[0]);
        state.block_table.advance(16);
        state.seqlen_offset = 16;
        // Block is full, decode needs 1 new block
        assert_eq!(state.blocks_needed_for_step(), 1);
    }
}
