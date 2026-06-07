# 0022 — Graceful degradation under a tiny KV pool (OOM, admission, mem-pool reclaim)

Status: accepted (partial — see Known Gaps)
Date: 2026-06-08

## Context

`scripts/repro-vllm-rust-kvpool-oom.sh` (against Gemma 4 12B EXL3 @2.00bpw on
an 8 GB GPU) exposed three failure modes when the KV pool is tiny — the model
weights are 6.23 GB, leaving ~1.5 GB for KV + activations, so the pool is only
**22 blocks = 352 tokens**:

1. **Over-long prompt → HTTP 000 / empty.** A prompt needing more blocks than
   the pool *has* can never satisfy `compute_schedule`'s
   `blocks_needed ≤ free_blocks`, so it sat in the waiting queue until the
   client timed out.
2. **Concurrent burst → CUDA OOM**, failing the *whole* decode batch (one
   allocation failure killed every concurrent request), and in the worst case
   crashing the server.
3. **Silent EMPTY** on some requests (client timeouts during preemption thrash).

Measurement established the key facts:
- Sequential / low-concurrency serving works perfectly (6/6 sequential).
- The "VRAM creep" within a run is the **live working set**, not reclaimable
  pooled-but-freed memory: a `cuMemPoolTrimTo` smoke test reclaims a dropped
  1 GB tensor fully (7084→6028→7084 MiB), but in-server the freed-pool is
  empty (`trim` reclaims 0) — free VRAM is steady, the limit is the model +
  per-step transients.
- The burst OOM is therefore a **transient working-set spike** exceeding the
  ~700 MiB headroom under concurrency, plus preemption thrash.

## Decision

Make the engine degrade gracefully on a tiny pool instead of hanging,
OOM-killing batches, or crashing:

1. **Admission guard** (`crates/core/src/engine/helpers.rs::admit_request`,
   plus server-side `validation::validate_prompt_token_length` /
   `completions.rs`): reject a prompt whose prefill needs more blocks than the
   pool total — engine returns a clear error, server returns HTTP 400 before
   admission. The prefill analogue of the decode-time recompute guard (0021
   follow-up) and the startup `clamp_max_model_len_to_capacity`.

2. **CUDA mem-pool reclaim** (`crates/core/src/engine/cuda_mem.rs`): set the
   default pool's `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` and `trim_to` under a
   free-VRAM watermark at the end of each engine step (on the spawn_blocking
   GPU thread, where the context is bound and the stream is synchronized — a
   trim on the async-loop thread is ineffective). Bounds genuine
   stream-ordered-pool creep on creep-prone workloads. Preventive; it is *not*
   the hero for this repro (the repro's footprint is live, not pooled).

3. **OOM backpressure, not OOM-kill**:
   - *Decode*: on a CUDA OOM, trim the pool and fail only the **newest**
     (highest `arrival_order`) sequence of the batch; survivors retry next
     step. The whole batch is never killed for one allocation failure.
   - *Prefill*: on a CUDA OOM, trim and **preempt-requeue** the request
     (bounded by `MAX_PREFILL_OOM_RETRIES = 8`) so concurrent prefills
     serialize under pressure and retry once peers free memory; a request that
     still OOMs alone after the budget fails cleanly.
   - OOM is detected by `is_cuda_oom` (string match — the structured
     `candle_core::Error` is lost at the model→engine stringification boundary).

## Consequences

- Over-long prompts: clean 400 (verified: "prompt too long: 429 tokens exceeds
  the usable context of 352 tokens").
- Concurrent burst: server stays **alive**, no whole-batch OOM-kill, no raw
  `DriverError(OUT_OF_MEMORY)` reaching the client, no hang.
- Sequential and modest concurrency: unchanged, fully working.

## Known Gaps (deliberately deferred — documented, not silently shipped)

Heavy concurrent bursts on a 22-block pool still **shed some requests** with a
clean `recompute after preemption can never be scheduled` 500. Root cause is a
**pre-existing recompute-preemption limitation**, not addressed here:

- KV-exhaustion preemption folds `generated_token_ids` into `prompt_token_ids`
  and **resets the generation budget**, so a repeatedly-preempted request
  regrows past `max_model_len` (the "353 tokens needs 23 blocks" guard then
  fails it), and for **non-streaming** requests the folded tokens are lost from
  the final output (would corrupt structured JSON).
- Proper fix (multi-day): track `original_prompt_len`, decrement
  `max_new_tokens` by the folded count on preempt, and assemble output from
  `prompt[original_prompt_len..] ++ generated` — so recompute neither regrows
  nor loses tokens. Until then, failing cleanly is preferred over returning
  corrupt output.

**Operational guidance for 8 GB + a 12B model:** the hardware is marginal.
Use `--kv-cache-dtype fp8` (roughly doubles the pool → far less preemption),
keep concurrency low, or use a larger GPU. `max_model_len` is auto-clamped to
pool capacity with a warning.
