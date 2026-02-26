use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::request::{RequestId, RequestStatus, SequenceState};

/// Scheduling policy for request ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingPolicy {
    /// First-come-first-served: requests are processed in arrival order.
    #[default]
    Fcfs,
    /// Priority-based: requests with lower priority values are processed first.
    /// Ties are broken by arrival time (earlier arrivals first).
    Priority,
}

/// How to handle in-flight requests when KV cache memory is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreemptionMode {
    /// Free the KV blocks and recompute the prefill on next scheduling.
    ///
    /// No CPU memory is required. Recomputation wastes GPU cycles on re-prefill
    /// but avoids the PCIe transfer latency of the Swap path.
    #[default]
    Recompute,
    /// Copy KV blocks to CPU swap space and restore them on resumption.
    ///
    /// Preserves computed KV state at the cost of CPU memory and PCIe bandwidth.
    /// Falls back to Recompute when CPU offload is not configured.
    Swap,
}

/// Priority level for a request. Lower values indicate higher priority.
pub type RequestPriority = i32;

/// Default priority for requests when none is specified.
pub const DEFAULT_PRIORITY: RequestPriority = 0;

/// Entry in the priority queue containing request metadata for ordering.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct PriorityEntry {
    /// Priority value (lower = higher priority).
    priority: RequestPriority,
    /// Arrival time as monotonic counter (lower = earlier).
    arrival_time: u64,
    /// The request identifier.
    request_id: RequestId,
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // For BinaryHeap (max-heap), we reverse the ordering to get min-heap behavior.
        // Lower priority value = higher priority = should come first.
        // For equal priorities, earlier arrival = should come first.
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Earlier arrival time should have higher priority (come first).
                // Reverse comparison for max-heap: smaller arrival_time -> Greater ordering.
                other.arrival_time.cmp(&self.arrival_time)
            }
            Ordering::Less => {
                // self has lower priority value = higher priority = should come first.
                // For max-heap: self should be "greater".
                Ordering::Greater
            }
            Ordering::Greater => {
                // self has higher priority value = lower priority = should come later.
                // For max-heap: self should be "less".
                Ordering::Less
            }
        }
    }
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Unified request queue that supports both FCFS and Priority scheduling.
/// Uses an enum internally to avoid trait object overhead.
pub enum RequestQueue {
    Fcfs(FcfsRequestQueue),
    Priority(PriorityRequestQueue),
}

impl RequestQueue {
    /// Create a new request queue based on the scheduling policy.
    pub fn new(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::Fcfs => Self::Fcfs(FcfsRequestQueue::new()),
            SchedulingPolicy::Priority => Self::Priority(PriorityRequestQueue::new()),
        }
    }

    /// Add a request to the queue with the given priority.
    pub fn add(&mut self, request_id: RequestId, priority: RequestPriority, arrival_time: u64) {
        match self {
            Self::Fcfs(q) => q.add(request_id, priority, arrival_time),
            Self::Priority(q) => q.add(request_id, priority, arrival_time),
        }
    }

    /// Remove and return the highest-priority request, or None if empty.
    pub fn pop(&mut self) -> Option<RequestId> {
        match self {
            Self::Fcfs(q) => q.pop(),
            Self::Priority(q) => q.pop(),
        }
    }

    /// Peek at the highest-priority request without removing it.
    pub fn peek(&mut self) -> Option<RequestId> {
        match self {
            Self::Fcfs(q) => q.peek(),
            Self::Priority(q) => q.peek(),
        }
    }

    /// Remove a specific request from the queue.
    /// Returns true if the request was found and removed.
    pub fn remove(&mut self, request_id: RequestId) -> bool {
        match self {
            Self::Fcfs(q) => q.remove(request_id),
            Self::Priority(q) => q.remove(request_id),
        }
    }

    /// Prepend a request to the front of the queue (for preempted requests).
    /// For FCFS, this goes to the front. For priority queues, this re-adds with
    /// the same priority (priority queues don't have a "front" concept).
    pub fn prepend(&mut self, request_id: RequestId, priority: RequestPriority, arrival_time: u64) {
        match self {
            Self::Fcfs(q) => q.prepend(request_id, priority, arrival_time),
            Self::Priority(q) => q.prepend(request_id, priority, arrival_time),
        }
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Fcfs(q) => q.is_empty(),
            Self::Priority(q) => q.is_empty(),
        }
    }

    /// Get the number of requests in the queue.
    pub fn len(&self) -> usize {
        match self {
            Self::Fcfs(q) => q.len(),
            Self::Priority(q) => q.len(),
        }
    }

    /// Get request IDs in scheduling order as a vector.
    pub fn iter_in_order(&self) -> Vec<RequestId> {
        match self {
            Self::Fcfs(q) => q.iter_in_order(),
            Self::Priority(q) => q.iter_in_order(),
        }
    }
}

/// First-come-first-served request queue.
/// Requests are processed in the order they arrive.
pub struct FcfsRequestQueue {
    queue: VecDeque<(RequestId, RequestPriority, u64)>,
}

impl FcfsRequestQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl Default for FcfsRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl FcfsRequestQueue {
    pub fn add(&mut self, request_id: RequestId, priority: RequestPriority, arrival_time: u64) {
        self.queue.push_back((request_id, priority, arrival_time));
    }

    pub fn pop(&mut self) -> Option<RequestId> {
        self.queue.pop_front().map(|(id, _, _)| id)
    }

    pub fn peek(&mut self) -> Option<RequestId> {
        self.queue.front().map(|(id, _, _)| *id)
    }

    pub fn remove(&mut self, request_id: RequestId) -> bool {
        if let Some(pos) = self.queue.iter().position(|(id, _, _)| *id == request_id) {
            self.queue.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn prepend(&mut self, request_id: RequestId, priority: RequestPriority, arrival_time: u64) {
        self.queue.push_front((request_id, priority, arrival_time));
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn iter_in_order(&self) -> Vec<RequestId> {
        self.queue.iter().map(|(id, _, _)| *id).collect()
    }
}

/// Priority-based request queue.
/// Requests with lower priority values are processed first.
/// Ties are broken by arrival time (earlier arrivals first).
pub struct PriorityRequestQueue {
    heap: BinaryHeap<PriorityEntry>,
    /// Track removed request IDs for lazy deletion from the heap.
    removed: HashSet<RequestId>,
    /// Set of active (non-removed) request IDs. Provides O(1) membership
    /// checks for remove() and O(1) len()/is_empty().
    active: HashSet<RequestId>,
}

impl PriorityRequestQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            removed: HashSet::new(),
            active: HashSet::new(),
        }
    }

    /// Clean up any removed entries from the top of the heap.
    fn cleanup_top(&mut self) {
        loop {
            let should_pop = self
                .heap
                .peek()
                .is_some_and(|entry| self.removed.contains(&entry.request_id));
            if should_pop {
                if let Some(entry) = self.heap.pop() {
                    self.removed.remove(&entry.request_id);
                }
            } else {
                break;
            }
        }
    }

    pub fn add(&mut self, request_id: RequestId, priority: RequestPriority, arrival_time: u64) {
        self.removed.remove(&request_id);
        self.active.insert(request_id);
        self.heap.push(PriorityEntry {
            priority,
            arrival_time,
            request_id,
        });
    }

    pub fn pop(&mut self) -> Option<RequestId> {
        self.cleanup_top();
        if let Some(entry) = self.heap.pop() {
            self.active.remove(&entry.request_id);
            Some(entry.request_id)
        } else {
            None
        }
    }

    pub fn peek(&mut self) -> Option<RequestId> {
        self.cleanup_top();
        self.heap.peek().map(|e| e.request_id)
    }

    pub fn remove(&mut self, request_id: RequestId) -> bool {
        // O(1): only succeed if the request is actually active.
        if !self.active.remove(&request_id) {
            return false;
        }
        self.removed.insert(request_id);
        true
    }

    pub fn prepend(&mut self, request_id: RequestId, priority: RequestPriority, arrival_time: u64) {
        // For priority queue, prepend is the same as add.
        // The priority determines position, not insertion order.
        self.add(request_id, priority, arrival_time);
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    pub fn len(&self) -> usize {
        self.active.len()
    }

    pub fn iter_in_order(&self) -> Vec<RequestId> {
        // Create a sorted copy for iteration.
        let mut entries: Vec<_> = self
            .heap
            .iter()
            .filter(|e| !self.removed.contains(&e.request_id))
            .copied()
            .collect();
        entries.sort_by(|a, b| b.cmp(a)); // Reverse because our Ord is reversed.
        entries.into_iter().map(|e| e.request_id).collect()
    }
}

impl Default for PriorityRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a request queue based on the scheduling policy.
pub fn create_request_queue(policy: SchedulingPolicy) -> RequestQueue {
    RequestQueue::new(policy)
}

#[derive(Clone, Copy)]
pub struct SchedulerConfig {
    pub max_running_requests: usize,
    pub max_tokens_per_step: usize,
    /// Enable chunked prefill: long prompts are processed in chunks across steps.
    pub enable_chunked_prefill: bool,
    /// Scheduling policy for request ordering.
    pub scheduling_policy: SchedulingPolicy,
    /// Maximum number of distinct LoRA adapters in a single decode batch.
    /// Requests exceeding this limit are deferred to the next step.
    /// 0 means unlimited.
    pub max_loras_per_batch: usize,
    /// Preemption strategy when KV cache is exhausted.
    pub preemption_mode: PreemptionMode,
    /// Maximum number of simultaneously active "long" chunked prefill requests.
    ///
    /// When chunked prefill is enabled and `long_prefill_token_threshold > 0`,
    /// at most this many requests whose remaining prompt tokens exceed the
    /// threshold are scheduled concurrently. Default 1. Must be ≥ 1.
    pub max_num_partial_prefills: usize,
    /// Remaining-token threshold for classifying a partial prefill as "long".
    ///
    /// 0 disables long-prefill throttling entirely. When non-zero, requests
    /// with more remaining tokens than this value compete for the shared
    /// `max_num_partial_prefills` budget instead of being admitted freely.
    pub long_prefill_token_threshold: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_running_requests: 256,
            max_tokens_per_step: 8192,
            enable_chunked_prefill: false,
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
            preemption_mode: PreemptionMode::Recompute,
            max_num_partial_prefills: 1,
            long_prefill_token_threshold: 0,
        }
    }
}

/// A scheduled prefill request with the number of tokens to process this step.
#[derive(Debug, Clone)]
pub struct PrefillSchedule {
    pub request_id: RequestId,
    /// Number of tokens to process in this step.
    pub chunk_size: usize,
}

#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub prefill_requests: Vec<PrefillSchedule>,
    pub decode_requests: Vec<RequestId>,
    pub preempted_requests: Vec<RequestId>,
}

/// The result of `compute_schedule`: a scheduling decision that can be
/// inspected without mutating the scheduler, then applied via `apply_schedule`.
///
/// This split enables optimistic pre-scheduling: the engine can compute the
/// next step's schedule before GPU execution finishes, then validate and
/// optionally discard the decision.
#[derive(Debug)]
pub struct ScheduleDecision {
    pub output: SchedulerOutput,
    /// Requests that should be moved from waiting → running.
    pub newly_admitted: Vec<RequestId>,
}

/// Metadata for requests tracked by the scheduler.
#[derive(Debug, Clone, Copy)]
struct RequestMetadata {
    priority: RequestPriority,
    arrival_time: u64,
}

pub struct Scheduler {
    config: SchedulerConfig,
    waiting_queue: RequestQueue,
    running_set: HashSet<RequestId>,
    /// Metadata for all tracked requests (waiting + running).
    request_metadata: HashMap<RequestId, RequestMetadata>,
    /// Monotonic counter for arrival time ordering.
    arrival_counter: u64,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            waiting_queue: create_request_queue(config.scheduling_policy),
            config,
            running_set: HashSet::new(),
            request_metadata: HashMap::new(),
            arrival_counter: 0,
        }
    }

    /// Add a request to the waiting queue with default priority.
    pub fn add_request(&mut self, id: RequestId) {
        self.add_request_with_priority(id, DEFAULT_PRIORITY);
    }

    /// Add a request to the waiting queue with a specified priority.
    /// Lower priority values indicate higher priority (processed first).
    pub fn add_request_with_priority(&mut self, id: RequestId, priority: RequestPriority) {
        let arrival_time = self.arrival_counter;
        self.arrival_counter += 1;

        self.request_metadata.insert(
            id,
            RequestMetadata {
                priority,
                arrival_time,
            },
        );
        self.waiting_queue.add(id, priority, arrival_time);
    }

    /// Remove a request from both waiting queue and running set.
    pub fn remove_request(&mut self, id: RequestId) {
        self.running_set.remove(&id);
        self.waiting_queue.remove(id);
        self.request_metadata.remove(&id);
    }

    /// Check if the scheduler has no pending or running requests.
    pub fn is_idle(&self) -> bool {
        self.waiting_queue.is_empty() && self.running_set.is_empty()
    }

    /// Get the number of currently running requests.
    pub fn num_running(&self) -> usize {
        self.running_set.len()
    }

    /// Get the number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Get the scheduling policy in use.
    pub fn scheduling_policy(&self) -> SchedulingPolicy {
        self.config.scheduling_policy
    }

    /// Compute a scheduling decision without mutating scheduler state.
    ///
    /// This is the pure read-only half of the scheduling logic. The returned
    /// [`ScheduleDecision`] can be inspected, validated, and optionally
    /// discarded before committing via [`apply_schedule`].
    ///
    /// `states` provides access to per-request state for budget checks.
    /// `num_free_blocks` is the current number of free blocks in the cache.
    pub fn compute_schedule(
        &self,
        states: &HashMap<RequestId, &SequenceState>,
        num_free_blocks: usize,
    ) -> ScheduleDecision {
        let mut output = SchedulerOutput::default();
        let mut budget = self.config.max_tokens_per_step;
        let mut free_blocks = num_free_blocks;

        // Step 1: Schedule running requests for decode or continued prefill
        let mut running_snapshot: Vec<RequestId> = self.running_set.iter().copied().collect();
        running_snapshot.sort_by_key(|id| {
            // Defensive: skip missing requests instead of panicking
            states.get(id).map(|s| s.arrival_order).unwrap_or(u64::MAX)
        });
        let mut to_preempt: Vec<RequestId> = Vec::new();

        for req_id in &running_snapshot {
            let Some(state) = states.get(req_id) else {
                continue;
            };

            if state.status == RequestStatus::Prefilling {
                // Continued prefill chunk (chunked prefill only)
                let remaining = state.prompt_token_ids.len() - state.num_computed_tokens;
                let chunk_size = if self.config.enable_chunked_prefill {
                    remaining.min(budget)
                } else {
                    remaining
                };
                if chunk_size == 0 {
                    continue;
                }
                let blocks_needed = state.block_table.blocks_needed(chunk_size);
                if blocks_needed <= free_blocks && chunk_size <= budget {
                    output.prefill_requests.push(PrefillSchedule {
                        request_id: *req_id,
                        chunk_size,
                    });
                    free_blocks -= blocks_needed;
                    budget -= chunk_size;
                } else {
                    to_preempt.push(*req_id);
                }
            } else {
                // Decode: 1 token
                let blocks_needed = state.blocks_needed_for_step();
                if blocks_needed <= free_blocks && budget > 0 {
                    output.decode_requests.push(*req_id);
                    free_blocks -= blocks_needed;
                    budget -= 1;
                } else {
                    to_preempt.push(*req_id);
                }
            }
        }

        // Step 2: Determine preemption order
        to_preempt.sort_by(|a, b| match self.config.scheduling_policy {
            SchedulingPolicy::Fcfs => {
                let order_a = states.get(a).map(|s| s.arrival_order).unwrap_or(0);
                let order_b = states.get(b).map(|s| s.arrival_order).unwrap_or(0);
                order_b.cmp(&order_a)
            }
            SchedulingPolicy::Priority => {
                let meta_a = self.request_metadata.get(a);
                let meta_b = self.request_metadata.get(b);
                match (meta_a, meta_b) {
                    (Some(ma), Some(mb)) => match mb.priority.cmp(&ma.priority) {
                        Ordering::Equal => mb.arrival_time.cmp(&ma.arrival_time),
                        other => other,
                    },
                    _ => {
                        let order_a = states.get(a).map(|s| s.arrival_order).unwrap_or(0);
                        let order_b = states.get(b).map(|s| s.arrival_order).unwrap_or(0);
                        order_b.cmp(&order_a)
                    }
                }
            }
        });
        for &req_id in &to_preempt {
            if let Some(state) = states.get(&req_id) {
                free_blocks += state.num_blocks();
            }
            output.preempted_requests.push(req_id);
        }

        // Step 3: Determine which waiting requests to admit
        let preempted_set: HashSet<RequestId> = output.preempted_requests.iter().copied().collect();
        let mut newly_admitted = Vec::new();

        // Simulate running_set size after preemptions
        let running_count = self.running_set.len() - to_preempt.len();

        // Count running chunked-prefill requests that exceed the long-prefill
        // threshold. These consume capacity from `max_num_partial_prefills`.
        // Preempted requests are excluded since they leave the running set.
        let threshold = self.config.long_prefill_token_threshold;
        let active_long_prefills: usize = if threshold > 0 {
            running_snapshot
                .iter()
                .filter(|id| !preempted_set.contains(id))
                .filter(|id| {
                    states.get(id).is_some_and(|s| {
                        s.status == RequestStatus::Prefilling
                            && (s.prompt_token_ids.len() - s.num_computed_tokens) > threshold
                    })
                })
                .count()
        } else {
            0
        };
        let mut long_prefill_count = active_long_prefills;

        let candidates: Vec<RequestId> = self.waiting_queue.iter_in_order();

        for req_id in candidates {
            if running_count + newly_admitted.len() >= self.config.max_running_requests {
                break;
            }
            if preempted_set.contains(&req_id) {
                break;
            }
            let Some(state) = states.get(&req_id) else {
                continue;
            };
            let remaining = state.prompt_token_ids.len() - state.num_computed_tokens;

            if remaining == 0 {
                if budget > 0 {
                    let blocks_needed = state.blocks_needed_for_step();
                    if blocks_needed <= free_blocks {
                        newly_admitted.push(req_id);
                        free_blocks -= blocks_needed;
                        budget -= 1;
                        output.decode_requests.push(req_id);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                continue;
            }

            // Long-prefill throttling: skip (not break) so shorter waiting
            // requests can still be admitted this step.
            let is_long_prefill = threshold > 0 && remaining > threshold;
            if is_long_prefill && long_prefill_count >= self.config.max_num_partial_prefills {
                continue;
            }

            let chunk_size = if self.config.enable_chunked_prefill {
                remaining.min(budget)
            } else {
                remaining
            };

            if chunk_size == 0 {
                break;
            }

            let blocks_needed = state.block_table.blocks_needed(chunk_size);
            if blocks_needed <= free_blocks && chunk_size <= budget {
                newly_admitted.push(req_id);
                free_blocks -= blocks_needed;
                budget -= chunk_size;
                if is_long_prefill {
                    long_prefill_count += 1;
                }
                output.prefill_requests.push(PrefillSchedule {
                    request_id: req_id,
                    chunk_size,
                });
            } else {
                break;
            }
        }

        ScheduleDecision {
            output,
            newly_admitted,
        }
    }

    /// Apply a previously computed schedule decision, mutating the scheduler.
    ///
    /// Moves preempted requests from running → waiting and newly admitted
    /// requests from waiting → running.
    pub fn apply_schedule(&mut self, decision: &ScheduleDecision) {
        // Move preempted requests back to waiting queue
        for &req_id in &decision.output.preempted_requests {
            self.running_set.remove(&req_id);
            if let Some(metadata) = self.request_metadata.get(&req_id) {
                self.waiting_queue
                    .prepend(req_id, metadata.priority, metadata.arrival_time);
            }
        }

        // Move newly admitted requests from waiting to running
        for &req_id in &decision.newly_admitted {
            self.waiting_queue.remove(req_id);
            self.running_set.insert(req_id);
        }
    }

    /// Convenience: compute + apply in one call (backward compatible).
    pub fn schedule(
        &mut self,
        states: &HashMap<RequestId, &SequenceState>,
        num_free_blocks: usize,
    ) -> SchedulerOutput {
        let decision = self.compute_schedule(states, num_free_blocks);
        self.apply_schedule(&decision);
        decision.output
    }

    /// Provides direct access to the running set for tests.
    #[cfg(test)]
    pub(crate) fn running_set_mut(&mut self) -> &mut HashSet<RequestId> {
        &mut self.running_set
    }

    /// Provides direct access to request metadata for tests.
    #[cfg(test)]
    pub(crate) fn insert_metadata(
        &mut self,
        id: RequestId,
        priority: RequestPriority,
        arrival_time: u64,
    ) {
        self.request_metadata.insert(
            id,
            RequestMetadata {
                priority,
                arrival_time,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::RequestStatus;

    fn refs(states: &HashMap<RequestId, SequenceState>) -> HashMap<RequestId, &SequenceState> {
        states.iter().map(|(&id, s)| (id, s)).collect()
    }

    fn prefill_ids(output: &SchedulerOutput) -> Vec<RequestId> {
        output
            .prefill_requests
            .iter()
            .map(|s| s.request_id)
            .collect()
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

    fn fcfs_config(max_running: usize, max_tokens: usize, chunked: bool) -> SchedulerConfig {
        SchedulerConfig {
            max_running_requests: max_running,
            max_tokens_per_step: max_tokens,
            enable_chunked_prefill: chunked,
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
            ..SchedulerConfig::default()
        }
    }

    fn priority_config(max_running: usize, max_tokens: usize, chunked: bool) -> SchedulerConfig {
        SchedulerConfig {
            max_running_requests: max_running,
            max_tokens_per_step: max_tokens,
            enable_chunked_prefill: chunked,
            scheduling_policy: SchedulingPolicy::Priority,
            max_loras_per_batch: 0,
            ..SchedulerConfig::default()
        }
    }

    // ==================== FCFS Request Queue Tests ====================

    #[test]
    fn fcfs_queue_basic_operations() {
        let mut queue = FcfsRequestQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.peek(), None);
        assert_eq!(queue.pop(), None);

        queue.add(1, 0, 0);
        queue.add(2, 0, 1);
        queue.add(3, 0, 2);

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.peek(), Some(1));

        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.pop(), Some(3));
        assert!(queue.is_empty());
    }

    #[test]
    fn fcfs_queue_prepend() {
        let mut queue = FcfsRequestQueue::new();

        queue.add(1, 0, 0);
        queue.add(2, 0, 1);
        queue.prepend(3, 0, 2); // Should go to front

        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
    }

    #[test]
    fn fcfs_queue_remove() {
        let mut queue = FcfsRequestQueue::new();

        queue.add(1, 0, 0);
        queue.add(2, 0, 1);
        queue.add(3, 0, 2);

        assert!(queue.remove(2));
        assert!(!queue.remove(2)); // Already removed
        assert!(!queue.remove(99)); // Never existed

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(3));
    }

    #[test]
    fn fcfs_queue_iter_in_order() {
        let mut queue = FcfsRequestQueue::new();

        queue.add(1, 0, 0);
        queue.add(2, 0, 1);
        queue.add(3, 0, 2);

        let order: Vec<_> = queue.iter_in_order();
        assert_eq!(order, vec![1, 2, 3]);
    }

    // ==================== Priority Request Queue Tests ====================

    #[test]
    fn priority_queue_basic_operations() {
        let mut queue = PriorityRequestQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.peek(), None);
        assert_eq!(queue.pop(), None);

        // Add with different priorities (lower = higher priority)
        queue.add(1, 10, 0); // Low priority
        queue.add(2, 5, 1); // Medium priority
        queue.add(3, 1, 2); // High priority

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 3);

        // Should pop in priority order (lowest value first)
        assert_eq!(queue.pop(), Some(3)); // priority 1
        assert_eq!(queue.pop(), Some(2)); // priority 5
        assert_eq!(queue.pop(), Some(1)); // priority 10
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_queue_same_priority_uses_arrival_time() {
        let mut queue = PriorityRequestQueue::new();

        // Same priority, different arrival times
        queue.add(1, 5, 2); // Later arrival
        queue.add(2, 5, 0); // Earlier arrival
        queue.add(3, 5, 1); // Middle arrival

        // Should pop in arrival order when priorities are equal
        assert_eq!(queue.pop(), Some(2)); // arrival 0
        assert_eq!(queue.pop(), Some(3)); // arrival 1
        assert_eq!(queue.pop(), Some(1)); // arrival 2
    }

    #[test]
    fn priority_queue_remove() {
        let mut queue = PriorityRequestQueue::new();

        queue.add(1, 1, 0);
        queue.add(2, 2, 1);
        queue.add(3, 3, 2);

        assert!(queue.remove(2));
        assert!(!queue.remove(2)); // Already removed
        assert!(!queue.remove(99)); // Never existed

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(3));
    }

    #[test]
    fn priority_queue_prepend_acts_as_add() {
        let mut queue = PriorityRequestQueue::new();

        queue.add(1, 5, 0);
        queue.prepend(2, 1, 1); // Should still respect priority

        // Even though 2 was "prepended", its lower priority value wins
        assert_eq!(queue.pop(), Some(2)); // priority 1
        assert_eq!(queue.pop(), Some(1)); // priority 5
    }

    #[test]
    fn priority_queue_iter_in_order() {
        let mut queue = PriorityRequestQueue::new();

        queue.add(1, 10, 0);
        queue.add(2, 5, 1);
        queue.add(3, 1, 2);
        queue.add(4, 5, 3); // Same priority as 2, later arrival

        let order: Vec<_> = queue.iter_in_order();
        assert_eq!(order, vec![3, 2, 4, 1]); // By priority, then arrival
    }

    #[test]
    fn priority_queue_negative_priorities() {
        let mut queue = PriorityRequestQueue::new();

        queue.add(1, 0, 0);
        queue.add(2, -10, 1); // Negative = even higher priority
        queue.add(3, 10, 2);

        assert_eq!(queue.pop(), Some(2)); // -10 (highest priority)
        assert_eq!(queue.pop(), Some(1)); // 0
        assert_eq!(queue.pop(), Some(3)); // 10 (lowest priority)
    }

    // ==================== Scheduler FCFS Tests ====================

    #[test]
    fn single_request_admitted() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let state = make_state(0, 5, RequestStatus::Waiting, 16, 0);
        let mut states = HashMap::new();
        states.insert(0, state);

        scheduler.add_request(0);
        assert!(!scheduler.is_idle());

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(prefill_ids(&output), vec![0]);
        assert!(output.decode_requests.is_empty());
        assert!(output.preempted_requests.is_empty());
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn running_request_scheduled_for_decode() {
        let config = fcfs_config(4, 512, false);
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
        let config = fcfs_config(2, 512, false);
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
        let config = fcfs_config(10, 10, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Request with 8 tokens fits, second with 8 doesn't
        states.insert(0, make_state(0, 8, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 8, RequestStatus::Waiting, 16, 1));
        scheduler.add_request(0);
        scheduler.add_request(1);

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(prefill_ids(&output), vec![0]);
        assert_eq!(scheduler.num_waiting(), 1); // request 1 still waiting
    }

    #[test]
    fn block_budget_enforced() {
        let config = fcfs_config(10, 512, false);
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
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        // Running request owns 1 block, but 0 free blocks and decode needs a new block
        let mut states = HashMap::new();
        let state = make_decoding_state(0, 16, 0, 16, 0);
        states.insert(0, state);
        scheduler.running_set_mut().insert(0);
        scheduler.insert_metadata(0, DEFAULT_PRIORITY, 0);

        // 0 free blocks means decode needs 1 block but can't get it
        let output = scheduler.schedule(&refs(&states), 0);
        assert!(output.decode_requests.is_empty());
        assert_eq!(output.preempted_requests, vec![0]);
        assert_eq!(scheduler.num_running(), 0);
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn preemption_newest_first() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Two running requests, both need 1 new block, only 1 available
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0)); // older
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1)); // newer
        scheduler.running_set_mut().insert(0);
        scheduler.running_set_mut().insert(1);
        scheduler.insert_metadata(0, DEFAULT_PRIORITY, 0);
        scheduler.insert_metadata(1, DEFAULT_PRIORITY, 1);

        let output = scheduler.schedule(&refs(&states), 1);
        // Only 1 block available, one decode succeeds, newer gets preempted
        assert_eq!(output.decode_requests.len(), 1);
        assert_eq!(output.preempted_requests, vec![1]); // newest preempted
    }

    #[test]
    fn preempted_request_readmitted() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 5, 3, 16, 0));
        scheduler.running_set_mut().insert(0);
        scheduler.insert_metadata(0, DEFAULT_PRIORITY, 0);

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
        assert_eq!(prefill_ids(&output), vec![0]);
        assert_eq!(scheduler.num_running(), 1);
    }

    #[test]
    fn remove_request_from_running() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);
        scheduler.running_set_mut().insert(0);

        scheduler.remove_request(0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn remove_request_from_waiting() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);
        scheduler.add_request(0);

        scheduler.remove_request(0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn decode_no_new_block_needed() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        // 5 tokens in block of 16, next decode fits without new block
        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 5, 0, 16, 0));
        scheduler.running_set_mut().insert(0);

        let output = scheduler.schedule(&refs(&states), 0); // 0 free blocks but not needed
        assert_eq!(output.decode_requests, vec![0]);
        assert!(output.preempted_requests.is_empty());
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let config = fcfs_config(4, 10, true);
        let mut scheduler = Scheduler::new(config);

        // 25-token prompt, budget 10 → first chunk = 10 tokens
        let mut states = HashMap::new();
        states.insert(0, make_state(0, 25, RequestStatus::Waiting, 16, 0));
        scheduler.add_request(0);

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(output.prefill_requests.len(), 1);
        assert_eq!(output.prefill_requests[0].request_id, 0);
        assert_eq!(output.prefill_requests[0].chunk_size, 10);
    }

    #[test]
    fn chunked_prefill_continued_chunk() {
        let config = fcfs_config(4, 10, true);
        let mut scheduler = Scheduler::new(config);

        // Simulate a request that already computed 10 tokens, has 15 remaining
        let mut state = make_state(0, 25, RequestStatus::Prefilling, 16, 0);
        state.num_computed_tokens = 10;
        // Allocate blocks for computed tokens
        let blocks_for_10 = 10usize.div_ceil(16);
        let block_ids: Vec<usize> = (0..blocks_for_10).collect();
        state.block_table.append_blocks(&block_ids);
        state.block_table.advance(10);

        let mut states = HashMap::new();
        states.insert(0, state);
        scheduler.running_set_mut().insert(0);

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(output.prefill_requests.len(), 1);
        assert_eq!(output.prefill_requests[0].request_id, 0);
        // 15 remaining, budget 10 → chunk_size = 10
        assert_eq!(output.prefill_requests[0].chunk_size, 10);
    }

    #[test]
    fn chunked_prefill_mixed_with_decode() {
        let config = fcfs_config(4, 20, true);
        let mut scheduler = Scheduler::new(config);

        // One decoding request + one new prefill
        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 5, 3, 16, 0));
        states.insert(1, make_state(1, 30, RequestStatus::Waiting, 16, 1));
        scheduler.running_set_mut().insert(0);
        scheduler.add_request(1);

        let output = scheduler.schedule(&refs(&states), 64);
        // Decode takes 1 token from budget → 19 remaining for prefill
        assert_eq!(output.decode_requests, vec![0]);
        assert_eq!(output.prefill_requests.len(), 1);
        assert_eq!(output.prefill_requests[0].request_id, 1);
        assert_eq!(output.prefill_requests[0].chunk_size, 19);
    }

    #[test]
    fn no_chunking_when_disabled() {
        let config = fcfs_config(4, 10, false);
        let mut scheduler = Scheduler::new(config);

        // 25-token prompt with budget 10: without chunking, doesn't fit
        let mut states = HashMap::new();
        states.insert(0, make_state(0, 25, RequestStatus::Waiting, 16, 0));
        scheduler.add_request(0);

        let output = scheduler.schedule(&refs(&states), 64);
        // Can't fit 25 tokens in budget of 10 without chunking
        assert!(output.prefill_requests.is_empty());
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn full_prefix_cached_admits_for_decode() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        // Simulate a request where prefix cache covered entire prompt
        let mut state = SequenceState::new(0, vec![0u32; 32], 64, 0, 16, 0);
        state.num_computed_tokens = 32; // all prompt tokens cached
        state.block_table.append_blocks(&[0, 1]); // 2 blocks for 32 tokens
        state.block_table.advance(32);
        state.seqlen_offset = 32;
        // Need at least one generated token for decode to work
        state.generated_token_ids.push(42);

        let mut states = HashMap::new();
        states.insert(0, state);
        scheduler.add_request(0);

        let output = scheduler.schedule(&refs(&states), 64);

        // Should be admitted directly for decode, not prefill
        assert!(output.prefill_requests.is_empty());
        assert_eq!(output.decode_requests, vec![0]);
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    // ==================== Scheduler Priority Tests ====================

    #[test]
    fn priority_scheduling_respects_priority() {
        let config = priority_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Add requests with different priorities
        states.insert(0, make_state(0, 5, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 5, RequestStatus::Waiting, 16, 1));
        states.insert(2, make_state(2, 5, RequestStatus::Waiting, 16, 2));

        // Request 1 has highest priority (lowest value), 0 has lowest
        scheduler.add_request_with_priority(0, 10); // Low priority
        scheduler.add_request_with_priority(1, 1); // High priority
        scheduler.add_request_with_priority(2, 5); // Medium priority

        // Only allow 1 request to be admitted
        let config = priority_config(1, 512, false);
        let mut scheduler = Scheduler::new(config);
        states.insert(0, make_state(0, 5, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 5, RequestStatus::Waiting, 16, 1));
        states.insert(2, make_state(2, 5, RequestStatus::Waiting, 16, 2));

        scheduler.add_request_with_priority(0, 10);
        scheduler.add_request_with_priority(1, 1);
        scheduler.add_request_with_priority(2, 5);

        let output = scheduler.schedule(&refs(&states), 64);

        // Request 1 should be admitted first (priority 1)
        assert_eq!(prefill_ids(&output), vec![1]);
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 2);

        // Schedule again to get next request
        // Only include requests 0 and 2 in states (request 1 was already processed)
        states.remove(&1);
        let config = priority_config(1, 512, false);
        let mut scheduler = Scheduler::new(config);

        scheduler.add_request_with_priority(0, 10);
        scheduler.add_request_with_priority(2, 5);

        let output = scheduler.schedule(&refs(&states), 64);

        // Request 2 should be next (priority 5 < 10)
        assert_eq!(prefill_ids(&output), vec![2]);
    }

    #[test]
    fn priority_scheduling_same_priority_uses_arrival_order() {
        let config = priority_config(1, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        states.insert(0, make_state(0, 5, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 5, RequestStatus::Waiting, 16, 1));
        states.insert(2, make_state(2, 5, RequestStatus::Waiting, 16, 2));

        // All have same priority, should use arrival order
        scheduler.add_request_with_priority(0, 5);
        scheduler.add_request_with_priority(1, 5);
        scheduler.add_request_with_priority(2, 5);

        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(prefill_ids(&output), vec![0]); // First arrival
    }

    #[test]
    fn priority_scheduling_with_multiple_admits() {
        let config = priority_config(3, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        for i in 0..5 {
            states.insert(i, make_state(i, 5, RequestStatus::Waiting, 16, i));
        }

        // Add with various priorities
        scheduler.add_request_with_priority(0, 3);
        scheduler.add_request_with_priority(1, 1); // Highest
        scheduler.add_request_with_priority(2, 2);
        scheduler.add_request_with_priority(3, 5);
        scheduler.add_request_with_priority(4, 4);

        let output = scheduler.schedule(&refs(&states), 64);

        // Should admit 3 requests in priority order: 1, 2, 0
        assert_eq!(prefill_ids(&output), vec![1, 2, 0]);
        assert_eq!(scheduler.num_running(), 3);
        assert_eq!(scheduler.num_waiting(), 2);
    }

    #[test]
    fn priority_scheduling_preempted_request_readded() {
        let config = priority_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0));
        scheduler.running_set_mut().insert(0);
        scheduler.insert_metadata(0, 5, 0);

        // Preempt due to no blocks
        let output = scheduler.schedule(&refs(&states), 0);
        assert_eq!(output.preempted_requests, vec![0]);

        // Request should be back in waiting queue with same metadata
        assert_eq!(scheduler.num_waiting(), 1);

        // Reset state
        let state = states.get_mut(&0).unwrap();
        state.status = RequestStatus::Preempted;
        state.block_table = crate::kv_cache::BlockTable::new(16);

        // Reschedule with blocks available
        let output = scheduler.schedule(&refs(&states), 64);
        assert_eq!(prefill_ids(&output), vec![0]);
    }

    #[test]
    fn default_scheduling_policy_is_fcfs() {
        let config = SchedulerConfig::default();
        assert_eq!(config.scheduling_policy, SchedulingPolicy::Fcfs);
    }

    #[test]
    fn scheduler_reports_policy() {
        let fcfs = Scheduler::new(fcfs_config(4, 512, false));
        assert_eq!(fcfs.scheduling_policy(), SchedulingPolicy::Fcfs);

        let priority = Scheduler::new(priority_config(4, 512, false));
        assert_eq!(priority.scheduling_policy(), SchedulingPolicy::Priority);
    }

    #[test]
    fn create_request_queue_returns_correct_type() {
        let fcfs = create_request_queue(SchedulingPolicy::Fcfs);
        let priority = create_request_queue(SchedulingPolicy::Priority);

        // Basic sanity check that they work
        let mut fcfs = fcfs;
        let mut priority = priority;

        fcfs.add(1, 0, 0);
        priority.add(1, 0, 0);

        assert!(!fcfs.is_empty());
        assert!(!priority.is_empty());
    }

    #[test]
    fn priority_preemption_evicts_lowest_priority_first() {
        // With priority scheduling, preemption should prefer to evict
        // lowest-priority requests (highest priority value) to protect
        // high-priority requests.
        let config = priority_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Three running requests with different priorities:
        // Request 0: priority 1 (high priority) - should be protected
        // Request 1: priority 10 (low priority) - should be preempted first
        // Request 2: priority 5 (medium priority) - should be preempted second if needed
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0));
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1));
        states.insert(2, make_decoding_state(2, 16, 0, 16, 2));

        scheduler.running_set_mut().insert(0);
        scheduler.running_set_mut().insert(1);
        scheduler.running_set_mut().insert(2);
        scheduler.insert_metadata(0, 1, 0); // High priority
        scheduler.insert_metadata(1, 10, 1); // Low priority
        scheduler.insert_metadata(2, 5, 2); // Medium priority

        // Only 1 block available, each request needs 1 block for decode
        // Two must be preempted, should preempt lowest priority first
        let output = scheduler.schedule(&refs(&states), 1);

        // Only 1 decode can succeed
        assert_eq!(output.decode_requests.len(), 1);

        // Two requests preempted: should be request 1 (priority 10) first,
        // then request 2 (priority 5). Request 0 (priority 1) should survive.
        assert_eq!(output.preempted_requests.len(), 2);
        assert_eq!(output.preempted_requests[0], 1); // Lowest priority (10)
        assert_eq!(output.preempted_requests[1], 2); // Next lowest (5)
        assert_eq!(output.decode_requests[0], 0); // Highest priority (1) survives
    }

    #[test]
    fn fcfs_preemption_evicts_newest_first() {
        // With FCFS scheduling, preemption should evict newest requests first
        // to minimize wasted compute on older requests.
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Three running requests with different arrival times
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0)); // Oldest
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1)); // Middle
        states.insert(2, make_decoding_state(2, 16, 0, 16, 2)); // Newest

        scheduler.running_set_mut().insert(0);
        scheduler.running_set_mut().insert(1);
        scheduler.running_set_mut().insert(2);
        scheduler.insert_metadata(0, DEFAULT_PRIORITY, 0);
        scheduler.insert_metadata(1, DEFAULT_PRIORITY, 1);
        scheduler.insert_metadata(2, DEFAULT_PRIORITY, 2);

        // Only 1 block available, each request needs 1 block for decode
        let output = scheduler.schedule(&refs(&states), 1);

        assert_eq!(output.decode_requests.len(), 1);
        assert_eq!(output.preempted_requests.len(), 2);

        // Preempted in newest-first order
        assert_eq!(output.preempted_requests[0], 2); // Newest (arrival 2)
        assert_eq!(output.preempted_requests[1], 1); // Next newest (arrival 1)
        assert_eq!(output.decode_requests[0], 0); // Oldest survives
    }

    #[test]
    fn priority_preemption_tiebreaker_is_arrival_time() {
        // When priorities are equal, preemption should prefer newer arrivals
        let config = priority_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Two requests with same priority, different arrival times
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0)); // Older
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1)); // Newer

        scheduler.running_set_mut().insert(0);
        scheduler.running_set_mut().insert(1);
        scheduler.insert_metadata(0, 5, 0); // Same priority, older
        scheduler.insert_metadata(1, 5, 1); // Same priority, newer

        // Only 1 block available
        let output = scheduler.schedule(&refs(&states), 1);

        assert_eq!(output.decode_requests.len(), 1);
        assert_eq!(output.preempted_requests.len(), 1);

        // Newer request should be preempted when priorities are equal
        assert_eq!(output.preempted_requests[0], 1); // Newer
        assert_eq!(output.decode_requests[0], 0); // Older survives
    }

    // ==================== compute_schedule / apply_schedule Tests ====================

    #[test]
    fn compute_schedule_does_not_mutate_scheduler() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        // Add 2 waiting requests
        scheduler.add_request(0);
        scheduler.add_request(1);

        let mut states = HashMap::new();
        states.insert(0, make_state(0, 8, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 8, RequestStatus::Waiting, 16, 1));

        // Snapshot scheduler state before
        let waiting_before = scheduler.num_waiting();
        let running_before = scheduler.num_running();

        // compute_schedule takes &self — should not mutate
        let _decision = scheduler.compute_schedule(&refs(&states), 100);

        // Verify state is identical
        assert_eq!(scheduler.num_waiting(), waiting_before);
        assert_eq!(scheduler.num_running(), running_before);
    }

    #[test]
    fn apply_schedule_moves_requests_correctly() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        // Add 2 waiting requests
        scheduler.add_request(0);
        scheduler.add_request(1);

        let mut states = HashMap::new();
        states.insert(0, make_state(0, 8, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 8, RequestStatus::Waiting, 16, 1));

        assert_eq!(scheduler.num_waiting(), 2);
        assert_eq!(scheduler.num_running(), 0);

        let decision = scheduler.compute_schedule(&refs(&states), 100);

        // Both should be admitted
        assert_eq!(decision.newly_admitted.len(), 2);

        // After apply, waiting should be 0 and running should be 2
        scheduler.apply_schedule(&decision);
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 2);
    }

    #[test]
    fn apply_schedule_handles_preemptions() {
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        let mut states = HashMap::new();
        // Two running requests, very tight block budget
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0));
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1));

        scheduler.running_set_mut().insert(0);
        scheduler.running_set_mut().insert(1);
        scheduler.insert_metadata(0, DEFAULT_PRIORITY, 0);
        scheduler.insert_metadata(1, DEFAULT_PRIORITY, 1);

        assert_eq!(scheduler.num_running(), 2);
        assert_eq!(scheduler.num_waiting(), 0);

        // Only 1 block: request 1 (newest) should be preempted
        let decision = scheduler.compute_schedule(&refs(&states), 1);
        assert_eq!(decision.output.preempted_requests.len(), 1);
        assert_eq!(decision.output.preempted_requests[0], 1);

        scheduler.apply_schedule(&decision);

        // Preempted request should be back in waiting
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn compute_then_apply_equals_schedule() {
        // Property test: compute + apply must produce identical SchedulerOutput
        // compared to the convenience schedule() method.
        let config = fcfs_config(4, 512, false);

        // Create two identical schedulers
        let mut scheduler_a = Scheduler::new(config.clone());
        let mut scheduler_b = Scheduler::new(config);

        // Add identical waiting requests
        scheduler_a.add_request(0);
        scheduler_a.add_request(1);
        scheduler_a.add_request(2);
        scheduler_b.add_request(0);
        scheduler_b.add_request(1);
        scheduler_b.add_request(2);

        let mut states = HashMap::new();
        states.insert(0, make_state(0, 8, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 8, RequestStatus::Waiting, 16, 1));
        states.insert(2, make_state(2, 16, RequestStatus::Waiting, 16, 2));

        // Method A: compute + apply
        let decision = scheduler_a.compute_schedule(&refs(&states), 10);
        scheduler_a.apply_schedule(&decision);
        let output_a = decision.output;

        // Method B: convenience schedule()
        let output_b = scheduler_b.schedule(&refs(&states), 10);

        // Outputs must be identical
        assert_eq!(prefill_ids(&output_a), prefill_ids(&output_b));
        assert_eq!(output_a.decode_requests, output_b.decode_requests);
        assert_eq!(output_a.preempted_requests, output_b.preempted_requests);

        // Scheduler state must be identical
        assert_eq!(scheduler_a.num_waiting(), scheduler_b.num_waiting());
        assert_eq!(scheduler_a.num_running(), scheduler_b.num_running());
    }

    #[test]
    fn compute_then_apply_equals_schedule_with_preemptions() {
        // Same property test but with running requests that get preempted
        let config = fcfs_config(4, 512, false);

        let mut scheduler_a = Scheduler::new(config.clone());
        let mut scheduler_b = Scheduler::new(config);

        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0));
        states.insert(1, make_decoding_state(1, 16, 0, 16, 1));
        states.insert(2, make_decoding_state(2, 16, 0, 16, 2));

        // Same running set in both
        for sched in [&mut scheduler_a, &mut scheduler_b] {
            sched.running_set_mut().insert(0);
            sched.running_set_mut().insert(1);
            sched.running_set_mut().insert(2);
            sched.insert_metadata(0, DEFAULT_PRIORITY, 0);
            sched.insert_metadata(1, DEFAULT_PRIORITY, 1);
            sched.insert_metadata(2, DEFAULT_PRIORITY, 2);
        }

        // Tight budget: only 2 blocks available
        let decision = scheduler_a.compute_schedule(&refs(&states), 2);
        scheduler_a.apply_schedule(&decision);
        let output_a = decision.output;

        let output_b = scheduler_b.schedule(&refs(&states), 2);

        assert_eq!(output_a.decode_requests, output_b.decode_requests);
        assert_eq!(output_a.preempted_requests, output_b.preempted_requests);
        assert_eq!(scheduler_a.num_waiting(), scheduler_b.num_waiting());
        assert_eq!(scheduler_a.num_running(), scheduler_b.num_running());
    }

    #[test]
    fn compute_schedule_tolerates_missing_requests_in_states() {
        // Scheduler has requests that are not in the states map (window between
        // completion and scheduler.remove_request). Should skip, not panic.
        let config = fcfs_config(4, 512, false);
        let mut scheduler = Scheduler::new(config);

        scheduler.running_set_mut().insert(0);
        scheduler.running_set_mut().insert(1);
        scheduler.insert_metadata(0, DEFAULT_PRIORITY, 0);
        scheduler.insert_metadata(1, DEFAULT_PRIORITY, 1);

        // Only provide state for request 0 — request 1 is "missing"
        let mut states = HashMap::new();
        states.insert(0, make_decoding_state(0, 16, 0, 16, 0));

        // Should not panic
        let decision = scheduler.compute_schedule(&refs(&states), 100);

        // Request 0 should still be scheduled for decode
        assert_eq!(decision.output.decode_requests.len(), 1);
        assert_eq!(decision.output.decode_requests[0], 0);
    }

    // ==================== PreemptionMode / Partial Prefill Tests ====================

    #[test]
    fn preemption_mode_default_is_recompute() {
        let cfg = SchedulerConfig::default();
        assert_eq!(cfg.preemption_mode, PreemptionMode::Recompute);
        assert_eq!(cfg.max_num_partial_prefills, 1);
        assert_eq!(cfg.long_prefill_token_threshold, 0);
    }

    #[test]
    fn long_prefill_threshold_zero_admits_all() {
        // When threshold == 0, long-prefill throttling is disabled and all
        // waiting requests are admitted according to the normal budget rules.
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 1024,
            enable_chunked_prefill: true,
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
            max_num_partial_prefills: 1, // limit = 1 but threshold = 0 so inactive
            long_prefill_token_threshold: 0,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut states = HashMap::new();
        // Two long requests (600 tokens each) — both admitted since threshold == 0
        states.insert(0, make_state(0, 600, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 600, RequestStatus::Waiting, 16, 1));
        scheduler.add_request(0);
        scheduler.add_request(1);

        let output = scheduler.schedule(&refs(&states), 1000);
        assert_eq!(output.prefill_requests.len(), 2);
    }

    #[test]
    fn long_prefill_limit_one_blocks_second_long_request() {
        // With threshold=100 and max_num_partial_prefills=1, only one long
        // request can be in-flight; the second is skipped this step.
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 2048,
            enable_chunked_prefill: true,
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
            max_num_partial_prefills: 1,
            long_prefill_token_threshold: 100,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut states = HashMap::new();
        // Two long requests (500 tokens each), both fit in budget/blocks
        states.insert(0, make_state(0, 500, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 500, RequestStatus::Waiting, 16, 1));
        scheduler.add_request(0);
        scheduler.add_request(1);

        let output = scheduler.schedule(&refs(&states), 1000);
        // Only one long prefill admitted
        assert_eq!(output.prefill_requests.len(), 1);
        assert_eq!(output.prefill_requests[0].request_id, 0);
    }

    #[test]
    fn long_prefill_limit_short_request_still_admitted() {
        // The long-prefill limit must not block short requests. A short request
        // that arrives after a long one must be admitted even when the long-
        // prefill slot is full.
        let config = SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 2048,
            enable_chunked_prefill: true,
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
            max_num_partial_prefills: 1,
            long_prefill_token_threshold: 100,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut states = HashMap::new();
        // long request (arrival 0), another long (arrival 1), short (arrival 2)
        states.insert(0, make_state(0, 500, RequestStatus::Waiting, 16, 0));
        states.insert(1, make_state(1, 500, RequestStatus::Waiting, 16, 1));
        states.insert(2, make_state(2, 50, RequestStatus::Waiting, 16, 2)); // short
        scheduler.add_request(0);
        scheduler.add_request(1);
        scheduler.add_request(2);

        let output = scheduler.schedule(&refs(&states), 1000);
        let ids: Vec<RequestId> = output
            .prefill_requests
            .iter()
            .map(|p| p.request_id)
            .collect();
        // First long + short admitted; second long skipped
        assert!(ids.contains(&0), "long request 0 should be admitted");
        assert!(
            ids.contains(&2),
            "short request 2 should be admitted despite long-prefill limit"
        );
        assert!(
            !ids.contains(&1),
            "second long request 1 should be throttled"
        );
    }
}
