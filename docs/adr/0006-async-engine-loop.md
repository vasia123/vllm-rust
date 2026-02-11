# ADR-0006: Async Engine Loop with spawn_blocking Execution

## Status
Implemented

## Context
The engine loop in `strategy.rs` was serial:
```
loop {
    drain_commands()      // CPU
    schedule()            // CPU
    execute_prefills()    // GPU (synchronous)
    execute_decodes()     // GPU (synchronous)
    check_completion()    // CPU
}
```

The GPU sits idle during CPU phases (command draining, scheduling, completion
checking). The CPU sits idle during GPU phases. More critically, the entire
engine runs in a single `tokio::spawn` task, so GPU execution blocks the async
runtime — preventing other async tasks (HTTP handlers, streaming delivery) from
making progress.

## Decision
Move GPU execution into `tokio::task::spawn_blocking` while the async task
buffers incoming commands via `tokio::select!`.

### Architecture

```
Step N:  [schedule] [execute (spawn_blocking) ─────────]
                    [buffer commands via select! ───────]
         After:     [process buffered commands]
Step N+1:[schedule] [execute (spawn_blocking) ─────────]
                    [buffer commands via select! ───────]
         After:     [process buffered commands]
```

### Implementation

The core change is a single `execute_engine_step()` function that encapsulates
all GPU work and post-processing (preemptions, prefills, stream tokens, decodes,
completion checks, finalization). This function runs inside `spawn_blocking`:

```rust
let exec_future = tokio::task::spawn_blocking(move || {
    execute_engine_step(
        &mut strategy, &mut state, &mut kv_cache_mgr,
        &tokenizer, &output, multi_step_count,
    );
    (strategy, state, kv_cache_mgr, tokenizer)
});
```

While the blocking task runs, the async task uses `select!` to buffer commands:

```rust
tokio::select! {
    biased;
    result = &mut exec_future => {
        // Reclaim ownership of strategy, state, kv_cache_mgr, tokenizer
        break;
    }
    Some(cmd) = cmd_rx.recv() => {
        pending_cmds.push(cmd);
    }
}
```

After the blocking task completes, buffered commands are processed normally.

### State ownership

All GPU-facing state moves into the blocking closure and is returned as a tuple:
- `strategy: S` (the model)
- `state: OwnedExecutionState` (scheduler + request map)
- `kv_cache_mgr: KVCacheManager` (block pools + cache engines)
- `tokenizer: TokenizerWrapper` (encoding/decoding)

The `cmd_rx` channel receiver stays in the async task for command buffering.

### Command handling during execution

Commands received while the GPU runs are buffered in a `Vec<EngineCommand>` and
processed synchronously when execution returns. This means:

- **New requests**: Admitted after execution completes; scheduled on next step
- **Aborts**: Processed after execution; one-step delay is acceptable
- **Stats/Pause/Resume**: Processed after execution; slight latency increase
- **Shutdown**: Processed after execution; current step completes before exit

### Channel close handling

If `cmd_rx.recv()` returns `None` (all senders dropped), the `Some(cmd)` pattern
fails and `select!` disables the branch. The loop then waits solely on the
execution future. After it completes, the next iteration's `try_recv()` returns
`Disconnected` and the loop exits cleanly.

## Consequences
- Tokio async runtime is free during GPU work (no blocked worker threads)
- HTTP handlers and streaming delivery remain responsive during model execution
- Commands are buffered during execution (at most one-step latency)
- `ExecutionStrategy: Send + 'static` required (was already `Send`)
- All other types (`KVCacheManager`, `TokenizerWrapper`, `OwnedExecutionState`)
  must be `Send` (already the case — required by `tokio::spawn`)
- Zero overhead for command-only iterations (frozen, idle)
- Foundation for future optimistic scheduling overlap (see ADR notes below)

## Future: Optimistic Scheduling Overlap

The current implementation buffers commands but does not overlap scheduling with
execution. A future enhancement could:

1. Snapshot scheduling-relevant state before execution
2. Run scheduling concurrently using the snapshot + optimistic free block count
3. When execution returns, validate and swap in the pre-computed schedule

The optimistic free block count formula:
```
optimistic_free = pre_execution_free - blocks_allocated_by_current_schedule
```

This is always a conservative estimate (execution may free more blocks via
completions), so it cannot cause OOM. This is left as a separate follow-up.
