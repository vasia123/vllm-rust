# 0019 — PooledTensor: type-safety for CUDA Graph capture

Status: accepted (2026-05-12)

## Context

CUDA Graph capture requires that every device pointer encoded into a
captured kernel node remains valid (same VA range, same intent) for the
lifetime of every captured graph that references it. In practice this
means:

- pool-backed buffers (`OutputPool::reserve`-allocated tensors) — valid
  because the pool retains them in its `Vec<Tensor>` for the process'
  lifetime;
- model weights — valid because they outlive the engine;
- KV cache slots — valid because allocated once at engine startup;
- `runner.captured.buffers.{input_ids, output}` — valid because the
  `CapturedGraph` owns them for the lifetime of each graph.

Anything else — `Tensor::from_vec(host_data)`, `Tensor::zeros(shape)`,
candle's `Embedding::index_select` output, `dev.alloc[_zeros]` inside
an `InplaceOp::cuda_fwd` — produces a fresh device pointer with a
**reference-counted, Rust-lifetime** storage. When the wrapping
`Tensor` goes out of scope its storage is freed (returned to the
cuMemPool). At replay the captured kernel reads from the now-stale
pointer; the memory has been reassigned to something else and the
captured forward produces wrong (deterministic-but-wrong) results or
crashes with `CUDA_ERROR_ILLEGAL_ADDRESS`.

Through Phase 11 (CUDA-Graph capture for EXL3) we paid for this
implicit invariant in burnt cycles: each fresh-alloc was found by
runtime symptom (wrong tokens, ILLEGAL_ADDRESS), then traced to a
specific `.contiguous()` or `from_vec` site, then migrated to pool. The
invariant lived only in code review. Any future op added to the
captured forward could silently re-introduce a fresh-alloc and reopen
the wound.

## Decision

Promote the "pool-backed" invariant into the Rust type system. Introduce
a `PooledTensor` newtype around `candle_core::Tensor`. Construct it
only via the pool (or, via `unsafe from_pool_unchecked`, from another
explicitly stable-lifetime source). View methods on `PooledTensor`
(`reshape`, `narrow`, `unsqueeze`, `squeeze`, `transpose`, `flatten_all`)
return `PooledTensor` (storage is shared). `contiguous` materialises
into a fresh pool slot if the input view is non-contiguous, preserving
the invariant.

Captured-eligible kernel wrappers gain `*_typed` siblings that accept
`&PooledTensor` inputs and return `PooledTensor` outputs. Model decode
forward methods (`QuantizedLlamaForCausalLM::forward_decode_batch_with_ctx`,
the per-layer `forward_decode_batch_with_shared_pooled`) thread
`PooledTensor` through every intermediate.

The compiler now rejects:

```rust
let positions = Tensor::from_vec(positions, (n,), device)?; // fresh
self.rotary_emb.apply_varlen_with_pos_tensor_pooled(q_pt, k_pt, ..., &positions)
// ^ type error: expected &PooledTensor, got &Tensor
```

Three categories of `&Tensor` parameters remain on the public typed
APIs:

1. **Model weights** (`rms_norm`'s `weight`, `bf16_matmul`'s `weight`,
   `exl3_gemm`'s `trellis/suh/svh`) — stable for engine lifetime by
   design contract; explicit `&Tensor` documents this.
2. **KV cache slots** (`paged_attention_v2`'s `k_cache`/`v_cache`) —
   allocated once by `CacheEngine::with_layout`; explicit `&Tensor`.
3. **CudaGraphRunner-owned buffers** (`captured.input_ids`) — owned by
   the runner for graph lifetime; wrapped via `from_pool_unchecked` at
   the model boundary.

Future work could tighten (1) and (2) with `ModelWeight` / `KvCacheTensor`
newtypes, but the current design captures the most-frequent regression
target (intermediate activations).

## Migration phasing

Implemented in 4 sub-phases (Phase TS.1–TS.4):

- **TS.1** Foundation: `PooledTensor` + `OutputPool::reserve_pooled` +
  view methods + tests.
- **TS.2** 10 typed-sibling kernel wrappers (`rms_norm_cuda_pooled_typed`,
  `bf16_add_pooled_typed`, `paged_attention_v2_cuda_pooled_typed`,
  `exl3_gemm_pooled_typed`, etc.). Legacy `_pooled` siblings remain for
  trait-bound callers (`Module::forward`, eager non-captured paths).
- **TS.3** `DecodeBatchShared` fields (`block_tables`, `seq_lens`,
  `positions_device`, `slot_mapping_device`) migrated to `PooledTensor`.
- **TS.4** `QuantizedLlamaForCausalLM::forward_decode_batch_with_ctx`
  routes through a typed path when `shared.prefer_pooled_attention` is
  true (capture-eligible). Legacy untyped path remains for
  eager / non-capture / CPU / unsupported-dtype callers.

## Trade-offs

- **Compile-time cost**: zero (newtype is `repr(transparent)` over
  `Tensor`).
- **Runtime cost**: zero — `as_tensor()` and `from_pool_unchecked` are
  no-ops; view methods are the same calls as on `Tensor`.
- **API surface**: 10 new `_typed` functions + `forward_pooled`
  defaults on `QuantizedLinear` and `RmsNorm`. Mostly mechanical and
  documented as siblings of their legacy counterparts. Future cleanup
  can rename the typed versions to the canonical name once all callers
  migrate.
- **Sound-by-construction**: every `&PooledTensor` participating in a
  captured kernel is provably pool-backed. Fresh-alloc regressions in
  the captured decode path become compile errors.
- **Soundness gap (acknowledged)**: `from_pool_unchecked` exists for
  the eager-fallback inside `build_decode_batch_shared` (oversize
  batches that never enter captured paths) and for re-wrapping outputs
  of legacy `_pooled` functions whose internal `OutputPool::reserve`
  proves pool ownership at runtime. Both annotated with SAFETY rationales.

## What the type system does NOT cover

The Phase 11.2.D capture-replay correctness regression (replay produces
wrong tokens despite identical kernel args) was investigated against
this type-safety hypothesis and **disproved** for the fresh-alloc class:

- **Forced `prefer_pooled = true` in eager mode** proves V2-pool path
  is algorithmically correct (eager + V2 + pool = correct generation
  byte-identical to eager + V1).
- **Cross-test on Qwen3-4B-AWQ** under capture replay yields the same
  wrong-tokens pattern (`Paris Agreement oncoming of the the the the
  ...`) — the bug is **not EXL3-specific**, it affects all
  captured-eligible quantized models in this project.
- **Disproved hypotheses (Phase CR.1–CR.5):**
  1. Stale V2 scratch buffers (CR.1) — fixed via captured
     `cuMemsetD8Async` of `tmp_out/exp_sums/max_logits`; no behavior
     change.
  2. cudarc event tracking (CR.2) — already disabled by
     `create_cuda_device` in `crates/server/src/main.rs:2160-2186`.
  3. Vendored kernel `cudaMalloc` (CR.3) — only `exl3_devctx.cu`'s
     `DevCtx::get_ws`/`get_locks` allocate, and they are dead code in
     the Rust dispatch path.
  4. EXL3 locks state (CR.5 prerequisite) — decode path uses
     non-cooperative `exl3_gemm_decode_kernel` (Phase 11.1) which
     receives Rust-side `exl3_locks` cache buffer as a kernel arg.
  5. `CU_STREAM_CAPTURE_MODE_GLOBAL` vs `RELAXED` (CR.5) — captures
     succeed identically with both; same wrong-tokens output.

The remaining bug is in CUDA Graph capture-replay execution semantics
proper — not in kernel logic, not in allocation lifetime, not in
event/stream tracking, not in capture mode.

### CR.6 apples-to-apples localization

With `helpers.rs:1025` patched to `prefer_pooled = true`, eager mode
also uses V2-pool path. Pool-slot diff between eager+V2-pool forward 1
and capture+V2-pool forward 1 pinpoints divergent shapes/slots:

| Slot | Op | Eager vs Capture |
|---|---|---|
| F16 [1, 1, 2048] slot 0 | embed | MATCH |
| F16 [1, 1, 2048] slot 1 | layer 0 input_layernorm | MATCH |
| F16 [1, 2048] slot 0 | layer 0 q_proj | MATCH |
| F16 [1, 2048] slot 1 | layer 0 RoPE Q out | MATCH |
| F16 [1, 512] slot 0 | layer 0 k_proj | MATCH |
| F16 [1, 512] slot 1 | layer 0 v_proj | MATCH |
| F16 [1, 512] slot 2 | layer 0 **RoPE K out** | **DIVERGE** |
| F16 [1, 32, 64] slot 0 | layer 0 **paged_attn V2 output** | **DIVERGE** |
| F16 [1, 1, 2048] slot 2 | layer 0 first bf16_add (post-attn residual) | DIVERGE |
| ...layer 1+ all activations | (cascading from above) | DIVERGE |

`RoPE Q` (block_dim=512, num_heads=32) matches; `RoPE K` (block_dim=256,
num_heads=8) does not — same kernel template
(`rotary_embedding_neox_fp16`), same `dev.memcpy_dtod` + kernel
sequence, different launch config. paged_attention V2 main+reduce kernel
pair output also diverges. Most likely root cause: cuGraph replay
handles the `cuMemcpy_dtod` graph node + immediately-following kernel
node sequence incorrectly under specific block_dim or memory access
patterns (the smaller K kernel pattern, the V2 reduction pattern).

### CR.7 — compute-sanitizer enabled on WSL via CUDA 13 toolkit

After CUDA 13.0 toolkit install (`/usr/local/cuda-13.0/bin/compute-sanitizer`),
the injection library is present. Full-server `racecheck` run is impossible
on the 16 GB Windows host (sanitizer + server + warmup exceeds 10 GB WSL
RAM cap → Linux OOM killer fires → Hyper-V resets the VM; confirmed by
`journalctl --boot=-1 -p err` showing `init.scope: killed by OOM killer`
at the relevant timestamps).

### CR.9–CR.12 — Cross-capture pool-persistence bug (root cause)

Built increasingly faithful multi-kernel + multi-layer + multi-forward
mini-decoder repros (see `cuda_kernels::tests` for `mini_decoder_*`
families). Key finding: **single tests pass, sequential tests fail**.

`zz_mini_decoder_second_with_input_shift_after_prior` runs the same
function (`run_mini_decoder_capture_replay(16, true, false)`) after
`mini_decoder_16layer_capture_replay_with_input_shift`. First call
PASSES, second FAILS with 99.8% mismatch (2044/2048 BF16 elements).

Bug requires:
1. Sequential capture-replay invocations on the SAME `OutputPool::global()`.
2. `input_shift=true` (pre-replay `cuda_memcpy_inplace` into a persistent
   input tensor, then `cuGraphLaunch`) — exact production pattern from
   `runner.execute(captured.buffers.input_ids, padded_input)`.

Adding `OutputPool::clear_all()` at test start (drops all entries,
forces fresh allocations) FIXES the divergence. Confirms the bug is
in `OutputPool` slot persistence across captured-graph lifecycles.

**Compute-sanitizer findings (CUDA 13 toolkit):**

- `racecheck`: **0 hazards** — no data races inside any captured kernel.
- `initcheck`: **0 errors** — no uninitialized memory reads.

Both sanitizers ran cleanly while the divergence reproduced. The bug
is therefore NOT at the kernel-memory-access level — it is at the
**cuGraph node dependency / pool slot lifecycle** level.

### Two distinct bugs uncovered

**Bug 1 — cudarc cross-capture state pollution (NEW; reproducible)**

`zz_mini_decoder_second_with_input_shift_after_prior` runs the same function
after `mini_decoder_16layer_capture_replay_with_input_shift`. First capture
records a 353-node graph; second captures only 113 nodes — **240 ops
(160 KERNEL + 32 MEMCPY + 48 MEMSET) silently dropped**. Same Rust code
path, same invocation count of internal ops (confirmed by counting
`zero_pool_tensor_dtod` invocations: 480 = 240×2). Surviving 113 nodes
are exactly the cuBLAS GEMM kernels (which use a separate cublas handle).
Custom-kernel launches via cudarc's `stream.launch_builder` plus
`memcpy_dtod` and `cuMemsetD8Async` go missing.

Compute-sanitizer (CUDA 13.0) `racecheck`/`initcheck`: 0 hazards each
during reproduction. Workaround: `OutputPool::clear_all()` between
captures, OR a single capture per process. Documented in
`cuda_kernels::tests::zz_mini_decoder_second_with_input_shift_after_prior`
and supporting tests.

**Bug 2 — production EXL3 wrong-tokens (unresolved)**

The original Phase 11.2.D regression that motivated Phase TS. Production
captures a FULL graph for batch=1 (468 nodes = 387 KERNEL + 33 MEMCPY +
48 MEMSET — every expected op present), yet replay produces deterministic
wrong tokens (` Parisianus...` instead of ` Paris. The capital...`).
Bug 1 does NOT explain this — production's captured graph is complete.

Tested workarounds that do NOT fix Bug 2:
- `VLLM_CAPTURE_SINGLE_SIZE=1` (capture only batch=1, skip 2/4/8/16) —
  wrong tokens persist.
- `VLLM_CAPTURE_ONLY_WARMUP=1` (skip JIT warmups for non-captured
  sizes; eliminates any prior eager forwards before the capture) —
  wrong tokens persist.
- All prior CR.1-CR.5 hypotheses (zero scratches, event-tracking
  disable, cuMemAlloc audit, capture mode GLOBAL vs RELAXED) — disproved.

Bug 2 manifests at the captured-graph REPLAY level: every node present,
deterministic execution, but values diverge from eager invocation of
the SAME kernels with SAME inputs. Compute-sanitizer cannot reproduce
under WSL2 because full-server OOMs (16 GB Windows host insufficient).

### Production fix candidates

1. **Pre-allocate pool to steady-state size BEFORE any capture starts**
   (warmup with worst-case batch BEFORE any `cuStreamBeginCapture`).
   Then captures never trigger pool growth; addresses stable.
2. **Per-captured-graph pool** (separate `OutputPool` instance per
   `CapturedGraph`). Defeats global pool sharing but eliminates
   cross-capture pollution.
3. **`OutputPool::clear_all()` before mark_warmed_up** (after all
   captures complete). RISKY — captured graphs hold pointers to
   dropped slots; needs careful lifecycle audit.

Option (1) is the surgical fix matching the diagnostic. Production
warmup currently captures batch=1, 2, 4, 8, 16 sequentially via JIT,
each potentially growing pool. Pre-warming the pool with a dummy
`batch=16` (or all-batch) eager forward BEFORE any capture would
freeze the pool at maximum size; subsequent captures find existing
slots without growing.

### CR.8 — Isolated kernel-level capture-replay mini-repros

Three tests added in `crates/core/src/cuda_kernels.rs::tests` (`#[ignore]`,
`--test-threads=1`):

1. `rope_k_capture_replay_matches_eager` — RoPE on Q+K, K-side block_dim=256.
   **PASS** — RoPE K alone produces identical output under cuGraph replay.
2. `rope_k_pool_chained_capture_replay_matches_eager` — same but inputs
   are pool slots populated by a captured upstream memcpy (mimics
   production data flow where `q_proj`/`k_proj` write into pool slots
   that RoPE then reads). **PASS** — pool chaining is also clean.
3. `paged_attn_v2_pool_capture_replay_matches_eager` — V2 main+reduce
   kernels in isolation, Q from a captured memcpy into a pool slot.
   **PASS (release mode)** — V2 kernel pair alone is clean too.

These results invalidate the single-kernel-buggy hypothesis: **no
individual kernel mis-behaves under cuGraph replay**. The wrong-tokens
regression on the full server requires **multi-kernel/multi-layer
interaction**:

- Inferred inter-node dependencies in the captured graph (cuGraph
  analyses memory aliasing to add automatic edges; if any kernel's
  output buffer aliases another's input non-trivially, missing
  dependency causes incorrect ordering at replay).
- Pool slot lifecycle across 16 attention layers in one captured graph
  (~600+ captured kernel/memcpy nodes total).
- KV cache write-then-read race within a single layer not captured as
  an explicit dependency.

### Why we stop here

Further bisect needs either:

- A multi-layer mini-repro test (1-2 day investment to construct a
  faithful small decoder forward without the full vllm-server) — would
  enable lightweight sanitizer/cuda-gdb work.
- Or full-server access on a Linux native box (no WSL2 RAM cap), where
  `compute-sanitizer racecheck` and `cuda-gdb` can attach to the
  production server directly.

`VLLM_EXL3_CAPTURE_MAX_M=0` (force-eager default) stays. Type-safety
foundation (TS.1–TS.5) protects from accidental fresh-alloc regressions
while the capture-replay bug awaits a fuller investigation environment.

`VLLM_EXL3_CAPTURE_MAX_M=0` (force-eager default) stays. The
type-safety foundation laid by Phase TS guards against introducing
further allocation-lifetime regressions while the capture-mechanics
bug awaits a debugging environment that can step through replayed
graph nodes.

### Defensive code retained from CR.1

`zero_pool_tensor_dtod` (`crates/core/src/cuda_kernels.rs`) was added
to wipe `paged_attention_v2_cuda_pooled`'s scratch buffers each forward
via a captured `cuMemsetD8Async` node. The current root-cause analysis
shows this is not strictly necessary (V2 reduce kernel bound-checks
per-seq), but it is defensively sound and capture-safe; kept in place.

## Files

- `crates/core/src/engine/output_pool.rs` — `PooledTensor` + `reserve_pooled`.
- `crates/core/src/cuda_kernels.rs` — 7 typed siblings (Type-Safe
  PooledTensor Wrappers section).
- `crates/core/src/quantization/exl3_cuda.rs` — `exl3_gemm_pooled_typed`,
  `exl3_gemv_pooled_typed`.
- `crates/core/src/quantization/marlin_cuda.rs` — `marlin_gemm_pooled_typed`.
- `crates/core/src/layers/attention/block.rs` — `DecodeBatchShared`
  with `PooledTensor` fields; `build_decode_batch_shared_with_options`.
- `crates/core/src/layers/normalization.rs` — `RmsNorm::forward_pooled`.
- `crates/core/src/quantization/config.rs` — `QuantizedLinear::forward_pooled` (default).
- `crates/core/src/models/llama_quantized.rs` — TS.4 typed path.
