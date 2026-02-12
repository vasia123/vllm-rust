# ADR-0008: Optimistic Pre-Scheduling (Batch Queue Overlap)

## Status
Implemented

## Context
ADR-0006 moved GPU execution into `spawn_blocking`, freeing the async runtime
during model forward passes. However, scheduling remained serial: the engine
computes the next step's schedule only after the current step returns. This
means the GPU sits idle during scheduling:

```
Step N:  [execute (GPU) ──────────] [schedule] [execute (GPU) ──────────]
```

For large batch sizes with priority-based policies, `schedule()` is non-trivial.
The opportunity: compute step N+1's schedule while step N's GPU execution runs.

## Decision
Implement optimistic pre-scheduling: compute the next step's schedule before
GPU execution finishes, validate it on return, and discard if invalidated.

### Architecture

```
Step N:  [schedule] [pre-schedule N+1] [execute (GPU) ──────────────────]
                                       [buffer commands via select! ────]
         After: [validate pre-schedule] [process buffered commands]
Step N+1:[use pre-schedule ─────────── or ─── recompute if invalid]
```

### Key design decisions

**1. Scheduler stays in the async loop (ADR-0006 Phase 1)**

The `Scheduler` was extracted from `OwnedExecutionState`. While `state` moves
into `spawn_blocking`, the scheduler remains in the async task. This enables
the pre-scheduling call without thread-safety concerns.

**2. schedule() split into compute + apply (Phase 2)**

```rust
pub fn compute_schedule(&self, ...) -> ScheduleDecision  // read-only
pub fn apply_schedule(&mut self, ...)                     // mutating
pub fn schedule(&mut self, ...) -> SchedulerOutput        // convenience wrapper
```

`compute_schedule` takes `&self` — it can be called speculatively without
committing state changes. If the pre-schedule is invalidated, we discard the
`ScheduleDecision` and recompute.

**3. Optimistic free block estimation (Phase 3)**

Pre-scheduling uses a conservative free block count:

```
optimistic_free = current_free - blocks_allocated_this_step
```

This is always a lower bound. Actual free blocks after execution will be >=
this value (completions release blocks). The pre-schedule therefore never
overcommits memory.

**4. Validation criteria**

A pre-computed schedule is valid if:
- No new commands arrived during GPU execution (adds/aborts/pauses change state)
- No scheduled request was completed or errored during execution

If invalid, the pre-schedule is discarded and a fresh schedule is computed.
Cost of invalidation: exactly one wasted `compute_schedule()` call — identical
to the non-optimistic path. **The optimization is never worse than baseline.**

**5. Deferred error handling**

Strategy methods (`execute_prefills`, `execute_decodes`) run inside
`spawn_blocking` without access to the scheduler. Errors are collected in
`state.errored_ids`, drained into `StepResult` at the end, and processed
by the async loop after reclaiming ownership.

**6. EngineConfig builder**

Added `EngineConfigBuilder` with required fields (`scheduler_config`,
`speculative_config`) and optional fields with defaults (`block_size=16`,
`multi_step_count=1`, `enable_prefix_caching=false`, etc.). The
`enable_optimistic_scheduling` flag (default: `true`) gates pre-scheduling.

## Consequences

### Performance
- Scheduling is overlapped with GPU execution when valid
- Hit rate depends on workload: steady decode-only batches → near 100%;
  frequent completions or command bursts → more invalidations
- Worst case matches current behavior (one `compute_schedule` per step)

### Correctness
- `compute_schedule` is pure (`&self`) — no state mutation until `apply_schedule`
- Conservative block estimation prevents OOM
- Validation is checked before any state changes from the pre-schedule

### Code impact
- `StepResult` struct for deferred cleanup (completed + errored request IDs)
- `ScheduleDecision` struct decouples scheduling computation from application
- `EngineConfigBuilder` simplifies adding future config fields
- `finish_request_with_error_deferred` pattern for error collection in strategies

## Alternatives considered

**Concurrent scheduling in a separate task**: More complex, requires shared
state or message passing for scheduler access. The current approach achieves
overlap without concurrency — scheduling runs between `apply_schedule` and
`spawn_blocking`, and the pre-computed result is a simple `Option`.

**Always-on without config flag**: Considered, but the flag costs nothing and
allows disabling for debugging or A/B testing without recompilation.
