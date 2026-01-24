use std::collections::{HashMap, HashSet, VecDeque};

use crate::request::{RequestId, SequenceState};

pub struct SchedulerConfig {
    pub max_running_requests: usize,
    pub max_tokens_per_step: usize,
}

#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub prefill_requests: Vec<RequestId>,
    pub decode_requests: Vec<RequestId>,
    pub preempted_requests: Vec<RequestId>,
}

pub struct Scheduler {
    config: SchedulerConfig,
    waiting_queue: VecDeque<RequestId>,
    running_set: HashSet<RequestId>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running_set: HashSet::new(),
        }
    }

    pub fn add_request(&mut self, id: RequestId) {
        self.waiting_queue.push_back(id);
    }

    pub fn remove_request(&mut self, id: RequestId) {
        self.running_set.remove(&id);
        self.waiting_queue.retain(|&x| x != id);
    }

    pub fn is_idle(&self) -> bool {
        self.waiting_queue.is_empty() && self.running_set.is_empty()
    }

    pub fn num_running(&self) -> usize {
        self.running_set.len()
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Core scheduling logic. Called once per engine iteration.
    ///
    /// `states` provides access to per-request state for budget checks.
    /// `num_free_blocks` is the current number of free blocks in the cache.
    pub fn schedule(
        &mut self,
        states: &HashMap<RequestId, &SequenceState>,
        num_free_blocks: usize,
    ) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();
        let mut budget = self.config.max_tokens_per_step;
        let mut free_blocks = num_free_blocks;

        // Step 1: Schedule running requests for decode (oldest first for priority)
        let mut running_snapshot: Vec<RequestId> = self.running_set.iter().copied().collect();
        running_snapshot.sort_by_key(|id| states[id].arrival_order);
        let mut to_preempt: Vec<RequestId> = Vec::new();

        for req_id in &running_snapshot {
            let state = &states[req_id];
            let blocks_needed = state.blocks_needed_for_step();

            if blocks_needed <= free_blocks && budget > 0 {
                output.decode_requests.push(*req_id);
                free_blocks -= blocks_needed;
                budget -= 1;
            } else {
                to_preempt.push(*req_id);
            }
        }

        // Step 2: Preempt (newest-first to minimize wasted compute)
        to_preempt.sort_by(|a, b| {
            let order_a = states[a].arrival_order;
            let order_b = states[b].arrival_order;
            order_b.cmp(&order_a)
        });
        for req_id in to_preempt {
            let state = &states[&req_id];
            free_blocks += state.num_blocks();
            self.running_set.remove(&req_id);
            self.waiting_queue.push_front(req_id);
            output.preempted_requests.push(req_id);
        }

        // Step 3: Admit waiting requests (FCFS)
        // Skip requests preempted this step — engine hasn't reset their state yet
        let preempted_set: HashSet<RequestId> = output.preempted_requests.iter().copied().collect();
        let mut newly_admitted = Vec::new();
        while !self.waiting_queue.is_empty()
            && self.running_set.len() + newly_admitted.len() < self.config.max_running_requests
        {
            let &req_id = self.waiting_queue.front().unwrap();
            if preempted_set.contains(&req_id) {
                break;
            }
            let state = &states[&req_id];
            let blocks_needed = state.blocks_needed_for_step();
            let tokens_needed = state.prompt_token_ids.len();

            if blocks_needed <= free_blocks && tokens_needed <= budget {
                self.waiting_queue.pop_front();
                newly_admitted.push(req_id);
                free_blocks -= blocks_needed;
                budget -= tokens_needed;
                output.prefill_requests.push(req_id);
            } else {
                break;
            }
        }

        for req_id in newly_admitted {
            self.running_set.insert(req_id);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::RequestStatus;

    fn refs(states: &HashMap<RequestId, SequenceState>) -> HashMap<RequestId, &SequenceState> {
        states.iter().map(|(&id, s)| (id, s)).collect()
    }

    fn make_state(
        id: RequestId,
        prompt_len: usize,
        status: RequestStatus,
        block_size: usize,
        arrival_order: u64,
    ) -> SequenceState {
        let mut state =
            SequenceState::new(id, vec![0u32; prompt_len], 64, 0, block_size, arrival_order);
        state.status = status;
        state
    }

    fn make_decoding_state(
        id: RequestId,
        prompt_len: usize,
        generated: usize,
        block_size: usize,
        arrival_order: u64,
    ) -> SequenceState {
        let mut state =
            SequenceState::new(id, vec![0u32; prompt_len], 64, 0, block_size, arrival_order);
        state.status = RequestStatus::Decoding;
        let total = prompt_len + generated;
        let blocks_needed = total.div_ceil(block_size);
        let block_ids: Vec<usize> = (0..blocks_needed).collect();
        state.block_table.append_blocks(&block_ids);
        state.block_table.advance(total);
        state.seqlen_offset = total;
        state.generated_token_ids = vec![0u32; generated];
        state
    }

    #[test]
    fn single_request_admitted() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        let state = make_state(0, 5, RequestStatus::Waiting, 16, 0);
        let mut states = HashMap::new();
        states.insert(0, state);

        scheduler.add_request(0);
        assert!(!scheduler.is_idle());

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(output.prefill_requests, vec![0]);
        assert!(output.decode_requests.is_empty());
        assert!(output.preempted_requests.is_empty());
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn running_request_scheduled_for_decode() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        let state = make_decoding_state(0, 5, 3, 16, 0);
        let mut states = HashMap::new();
        states.insert(0, state);

        scheduler.add_request(0);
        // Simulate first schedule admitted it
        let _ = scheduler.schedule(&refs(&states), 64);
        // Now it's in running set, update status
        states.get_mut(&0).unwrap().status = RequestStatus::Decoding;

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(output.decode_requests, vec![0]);
        assert!(output.prefill_requests.is_empty());
    }

    #[test]
    fn max_running_requests_enforced() {
        let config = SchedulerConfig {
            max_running_requests: 2,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        for i in 0..4 {
            states.insert(i, make_state(i, 5, RequestStatus::Waiting, 16, i));
            scheduler.add_request(i);
        }

        let output = scheduler.schedule(&refs(&states), 64);
        // Only 2 admitted
        assert_eq!(output.prefill_requests.len(), 2);
        assert_eq!(scheduler.num_running(), 2);
        assert_eq!(scheduler.num_waiting(), 2);
    }

    #[test]
    fn token_budget_enforced() {
        let config = SchedulerConfig {
            max_running_requests: 10,
            max_tokens_per_step: 10, // tight budget
        };
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Request with 8 tokens fits, second with 8 doesn't
        states.insert(0, make_state(0, 8, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 8, RequestStatus::Waiting, 16, 1));
        scheduler.add_request(0);
        scheduler.add_request(1);

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(output.prefill_requests, vec![0]);
        assert_eq!(scheduler.num_waiting(), 1); // request 1 still waiting
    }

    #[test]
    fn block_budget_enforced() {
        let config = SchedulerConfig {
            max_running_requests: 10,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Request needs 2 blocks (20 tokens / 16 block_size), only 1 free
        states.insert(0, make_state(0, 20, RequestStatus::Waiting, 16, 0));
        scheduler.add_request(0);

        let output = scheduler.schedule(&refs(&states), 1); // only 1 free block
        assert!(output.prefill_requests.is_empty());
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn preemption_frees_blocks() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        // Running request owns 1 block, but 0 free blocks and decode needs a new block
        let mut states = HashMap::new();
        let state = make_decoding_state(0, 16, 0, 16, 0);
        states.insert(0, state);
        scheduler.running_set.insert(0);

        // 0 free blocks means decode needs 1 block but can't get it
        let output = scheduler.schedule(&refs(&states), 0);
        assert!(output.decode_requests.is_empty());
        assert_eq!(output.preempted_requests, vec![0]);
        assert_eq!(scheduler.num_running(), 0);
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn preemption_newest_first() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Two running requests, both need 1 new block, only 1 available
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0)); // older
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1)); // newer
        scheduler.running_set.insert(0);
        scheduler.running_set.insert(1);

        let output = scheduler.schedule(&refs(&states), 1);
        // Only 1 block available, one decode succeeds, newer gets preempted
        assert_eq!(output.decode_requests.len(), 1);
        assert_eq!(output.preempted_requests, vec![1]); // newest preempted
    }

    #[test]
    fn preempted_request_readmitted() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 5, 3, 16, 0));
        scheduler.running_set.insert(0);

        // Preempt (0 blocks available, needs new block at boundary)
        states.get_mut(&0).unwrap().block_table.advance(11); // fill to 16+3=19, needs 2nd block
        let output = scheduler.schedule(&refs(&states), 0);
        assert_eq!(output.preempted_requests, vec![0]);

        // Reset state as if preempted (engine would do this)
        let state = states.get_mut(&0).unwrap();
        state.status = RequestStatus::Preempted;
        state.block_table = crate::kv_cache::BlockTable::new(16);
        state.generated_token_ids.clear();
        state.seqlen_offset = 0;

        // Now schedule again with blocks available
        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(output.prefill_requests, vec![0]);
        assert_eq!(scheduler.num_running(), 1);
    }

    #[test]
    fn remove_request_from_running() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);
        scheduler.running_set.insert(0);

        scheduler.remove_request(0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn remove_request_from_waiting() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);
        scheduler.add_request(0);

        scheduler.remove_request(0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn decode_no_new_block_needed() {
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
        };
        let mut scheduler = Scheduler::new(config);

        // 5 tokens in block of 16, next decode fits without new block
        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 5, 0, 16, 0));
        scheduler.running_set.insert(0);

        let output = scheduler.schedule(&refs(&states), 0); // 0 free blocks but not needed
        assert_eq!(output.decode_requests, vec![0]);
        assert!(output.preempted_requests.is_empty());
    }
}
