use crate::kv_cache::BlockTable;
use crate::sampling::{SamplerState, SamplingParams};

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    Prefilling,
    Decoding,
    Preempted,
    FinishedEos,
    FinishedLength,
}

impl RequestStatus {
    pub fn is_finished(self) -> bool {
        matches!(self, Self::FinishedEos | Self::FinishedLength)
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
    /// Number of prompt tokens already processed (for chunked prefill).
    pub num_computed_tokens: usize,
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
            num_computed_tokens: 0,
        }
    }

    pub fn num_generated(&self) -> usize {
        self.generated_token_ids.len()
    }

    pub fn num_blocks(&self) -> usize {
        self.block_table.block_ids().len()
    }

    /// Blocks needed for the next step (prefill or single decode token).
    pub fn blocks_needed_for_step(&self) -> usize {
        if self.status == RequestStatus::Waiting || self.status == RequestStatus::Preempted {
            self.block_table.blocks_needed(self.prompt_token_ids.len())
        } else {
            self.block_table.blocks_needed(1)
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
    }

    #[test]
    fn status_is_running() {
        assert!(!RequestStatus::Waiting.is_running());
        assert!(RequestStatus::Prefilling.is_running());
        assert!(RequestStatus::Decoding.is_running());
        assert!(!RequestStatus::Preempted.is_running());
        assert!(!RequestStatus::FinishedEos.is_running());
        assert!(!RequestStatus::FinishedLength.is_running());
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
