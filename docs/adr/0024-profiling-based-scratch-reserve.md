# 0024 — Profiling-based non-KV memory reserve (replace the fixed 1.5 GiB scratch constant)

Date: 2026-06-17
Status: accepted

## Context

When `--gpu-memory-utilization` (or the no-flag auto-tune at 0.85) sizes the
KV cache, the server must withhold some VRAM for the non-KV working set:
transient activations plus the cuBLAS/dequant kernel workspace a forward
allocates. We did this with a **hardcoded constant** in
`estimate_kv_cache_budget` (`crates/server/src/main.rs`):

```rust
let scratch_reserve: usize = 1536 * 1024 * 1024; // 1.5 GiB, withheld unconditionally
```

The constant was calibrated for the worst case we hit: a 4 B AWQ-Marlin model
whose long-context decode — crossing the adaptive-attention boundary at 4096
tokens in `scripts/test_qwen3_awq_correctness.sh` case 6 — OOM'd mid-decode at
a 1 GiB reserve and only cleared at 1.5 GiB.

The problem: it is **one-size-fits-worst-case**. A small model (e.g. an
EmbeddingGemma embedder, ~300 M params) needs only a few hundred MiB of
working set, but the 1.5 GiB floor is withheld anyway. Worse, because the
reserve is subtracted from both the utilization target and free VRAM, it
imposes a hard floor of `util ≈ 1536/total ≈ 0.19` on an 8 GiB card and a
~1.85 GB minimum footprint — almost all of it the constant, not the weights.
Operators trying to pack an embedder alongside a chat model hit this wall.

vLLM does not guess. `determine_available_memory`
(`reference/vllm/vllm/v1/worker/gpu_worker.py:332`) runs a **profiling
forward** at `max_num_batched_tokens`, measures the peak activation + the
non-torch (driver/cuBLAS) increase, and sizes the KV budget as
`util*total − weights − measured_peak`. No magic constant; the reserve is
workload-aware by construction.

We considered exposing a manual `--scratch-reserve-mib` flag instead. Rejected:
it converts our miscalibration into an operator-facing tuning knob — the
opposite of professional. The right fix is to measure.

## Decision

Replace the constant with a startup **profiling run**, `profile_non_kv_memory`
(`crates/server/src/main.rs`, `#[cfg(feature = "cuda")]`):

1. `cuda_mem::trim(0)` + `synchronize()`; snapshot baseline free VRAM.
2. Build a **temporary** `KVCacheManager` (one prefill chunk + one
   long-context sequence) with the real cache's geometry / kv_dtype via the
   existing `build_kv_cache_manager`. Snapshot free-with-temp.
3. **Prefill probe** — dummy `forward` at `max_num_batched_tokens` tokens.
4. **Decode probe** — single-token `forward_decode_batch_with_ctx` at
   `seqlen_offset = profile_ctx − 1`, with `profile_ctx` clamped to cross the
   4096 boundary when the model's capacity allows (the reason the constant
   had to be 1.5 GiB, not 1.0). Capped at 8192 so the temp KV stays bounded.
5. `peak = free_with_temp − min(free_after_prefill, free_after_decode)`; free
   everything, `trim(0)`.
6. Reserve = `max(peak × margin + headroom, floor)` with `margin = 1.10`,
   `headroom = 128 MiB`, `floor = 256 MiB`, all overridable via
   `VLLM_PROFILE_{MARGIN,HEADROOM_MB,FLOOR_MB}` (`cuda_mem.rs`).

`estimate_kv_cache_budget` calls it and **falls back to the 1.5 GiB constant**
(`LEGACY_SCRATCH_RESERVE`) on any failure — OOM, forward error, an implausible
measurement, or when profiling is disabled. Profiling is skipped under
pipeline parallelism (`pp_stage.is_some()`): a PP stage holds only a layer
subset, but the profiler builds a full-model temp cache, so that combination
is untested and degrades to the safe constant.

## Measurement limitation (why margin + headroom, not a true peak counter)

candle's CUDA backend uses a **retaining** stream-ordered pool: freed buffers
(`cuMemFreeAsync`) stay in the pool and `mem_get_info` counts them as *used*
until `trim`. So a free-VRAM snapshot after a forward already reflects the
pool's high-water — convenient, no separate peak counter needed. The temp
cache's block-pool tensor is allocated up front, so it is already accounted
for in `free_with_temp`; the dip below it is pure transient scratch.

The catch: this measures *net* pool growth, which can **under-count** the true
instantaneous peak, because the pool reuses buffers freed *within* a single
forward. candle exposes no `allocated.peak` counter (torch's
`reset_peak_memory_stats` has no equivalent), and adding one means patching
cudarc — out of scope for a startup-only path. We compensate with:

- the **min-free across both probes** (whichever reserved more wins);
- a **multiplicative margin** (1.10) for scaling error;
- a **fixed additive headroom** (128 MiB) for the fixed-size cuBLAS workspace
  the proportional margin cannot cover on small models.

Note the cuBLAS handle's workspace, allocated on the profiler's first matmul,
persists after profiling (it is outside candle's pool, not reclaimed by
`trim`). It is therefore counted both in the profiled peak and in the reduced
post-profile free VRAM — a slight over-reserve. That is conservative (never an
under-reserve) and small; runtime mempool growth across many decode steps is
bounded separately by the existing per-step `trim_under_pressure`.

## Consequences

- Small models (embedders) get a reserve near their true working set, not
  1.5 GiB — the `util ≈ 0.19` floor and ~1.85 GB minimum footprint are gone.
  An embedder and a chat model can coexist on one 8 GiB card.
- Large models get a measured reserve that crosses the same 4096 boundary the
  constant protected, so case 6 must not regress.
- The 1.5 GiB constant remains as the documented fallback; behaviour under PP
  or any profiling failure is exactly as before.
- Explicit `--num-blocks` still bypasses budget sizing entirely (no profiling).

## Validation

- **GPU unit test** (`profiling_tests` in `main.rs`, `#[ignore]`): toy Qwen3,
  asserts the reserve is positive, below free VRAM, below the legacy constant,
  and stable across two runs. Validates the *mechanism*, not absolute
  calibration (toy weights don't exercise the dequant workspace).
- **E2E guard**: `scripts/test_qwen3_awq_correctness.sh` case 6 — the real
  long-context-decode OOM scenario the constant was calibrated for — must pass
  with profiled sizing. **Status: pending** — the `Qwen/Qwen3-4B-AWQ`
  checkpoint failed to download (flaky HF CDN), so this authoritative AWQ run
  was deferred. The GPU unit test (mechanism) and the EmbeddingGemma embedder
  (`--gpu-memory-utilization 0.15`: profiled reserve 256 MiB vs the 1.5 GiB
  constant that would have bailed, `/v1/embeddings` returns L2-normalized
  768-d vectors) are validated. Run case 6 before relying on profiled sizing
  for AWQ-Marlin in production.

No criterion bench: profiling is startup-only, not an engine-step hot path, so
a microbench would measure nothing actionable (CLAUDE.md scopes mandatory
benches to the engine step / forward / kernels).
