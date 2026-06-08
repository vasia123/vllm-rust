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

## Recompute-preemption correctness (the former "Known Gap" — now FIXED)

The earlier version of this ADR deferred a recompute-preemption bug: KV-exhaustion
preemption **folded** `generated_token_ids` into `prompt_token_ids` and reset the
generation budget, so a repeatedly-preempted request regrew past `max_model_len`
(the "353 tokens needs 23 blocks" guard then failed it) and lost the folded
tokens from non-streaming output.

**Fixed** by mirroring vLLM's V1 model (no-fold recompute):
- Preemption keeps `prompt_token_ids` and `generated_token_ids` SEPARATE and
  resets only `num_computed_tokens`/`seqlen_offset` to 0 (all four preempt sites:
  the decode-time KV-exhaustion path in `helpers.rs`, and `handle_preemptions` in
  `standard.rs`/`speculative.rs`/`mod.rs` — the latter three previously *cleared*
  generated, losing output even without regrowth).
- `SequenceState::total_len()` (= prompt + generated) and `token_window()` drive
  scheduling, block sizing, and the prefill chunk, so a resumed request
  re-prefills its FULL sequence (prompt + generated) and continues generating.
- `check_finished` already counts `num_generated()`, which is now never reset →
  the generation budget is exact (no regrowth), and the final output
  (`generated_token_ids`) keeps every token. Bonus: `prompt_token_ids` stays the
  true prompt, so prompt-token counts and prefix-cache registration are correct.

Verified: `preemption_preserves_output_and_budget` integration test (tiny pool
forces preempt/resume → each request yields exactly `max_tokens`, all preserved);
the kvpool-OOM repro burst no longer produces "353 tokens" failures.

**Operational guidance for 8 GB + a 12B model:** the hardware is marginal.
Use `--kv-cache-dtype fp8` (roughly doubles the pool → far less preemption),
keep concurrency low, or use a larger GPU. `max_model_len` is auto-clamped to
pool capacity with a warning.
