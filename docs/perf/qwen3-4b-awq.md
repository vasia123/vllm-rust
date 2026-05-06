# Performance: Qwen3-4B-AWQ

Tokens-per-second baseline for Qwen3-4B with `quant_method=awq` (bits=4,
group_size=128) running through the `cuda-full` build.

## Build

```bash
cargo build --release -p vllm-server --features cuda-full
```

Required GPU compute capability: ≥ 8.0 (Ampere) for Marlin INT4 GEMM.
Without `marlin` the AWQ path falls back to `gptq_cuda::gptq_gemm` —
still GPU-resident, but a layer slower than Marlin.

## Run server

```bash
./target/release/vllm-server serve \
  --model Qwen/Qwen3-4B-AWQ \
  --port 8000
```

CUDA graphs are auto-enabled for CUDA devices. Disable with
`--enforce-eager` if you suspect graph capture is interacting badly with
a kernel.

## Bench

```bash
python scripts/bench_decode.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-4B-AWQ \
  --label vllm-rust \
  --concurrency 1 4 8 \
  --prompt-len 256 \
  --max-tokens 512 \
  --runs 3
```

Run a Python vLLM container on the same machine and bench it the same way
(`--label python-vllm --base-url http://localhost:8001/v1`) to compare
backend-to-backend on identical hardware.

## Measured numbers

Hardware: NVIDIA GeForce RTX 4060 Laptop, 8 GiB VRAM, sm_89 (Ada).
Build: `cargo build --release -p vllm-server --features cuda-full`.
Server: `--num-blocks 96 --max-requests 4 --max-model-len 1024
--enforce-eager`. Warmup run discarded.

| Build state                                            | Startup | TTFT (ms) | decode tok/s | per-layer time |
| ------------------------------------------------------ | ------- | --------- | ------------ | -------------- |
| Stages 1–6 (initial baseline)                          | ~5 min  | ~17 000   | ~0.05        | not measured   |
| + Stage 7 (decode shared tensors) wired on Qwen3-AWQ¹  | ~5 min  | ~17 000   | ~0.06        | ~460 ms / 36   |
| + Stage 8 (CUDA `awq_to_gptq_qweight_int4` kernel)     | **21 s** | ~17 000  | ~0.06        | ~460 ms / 36   |
| + Stage 9 (AwqMarlin live + AWQ GEMV kernel for M ≤ 16) | ~21 s  | ~600     | **~11.6**    | ~10 ms / 36    |
| + Stage 10-1 (RoPE positions: zero-copy U32 → kernel)   | ~21 s  | ~600     | **~13.0**    | ~9 ms / 36     |
| + Stage 10-α (split-K GEMV: ×4 / ×8 thread-per-column)  | ~21 s  | ~600     | **~20.0**    | ~6 ms / 36     |
| + Stage 10-β (vec4 LDG GEMV on transposed qweight)      | ~21 s  | ~600     | **~40.5**    | ~3 ms / 36     |
| + Stage 10-γ (skip zero-init of GEMV output buffer)     | ~21 s  | ~600     | **~42 / 50** | ~3 ms / 36     |
| + Stage 10-δ (uninit alloc on RoPE/PA/RMSNorm/SwiGLU)   | ~21 s  | ~600     | **~44 / 50** | ~3 ms / 36     |

Cumulative speedup vs the Stages 1–6 baseline: **~733× steady (44 tok/s
on a 256-token sustained run), peaks ~50** — original ≥ 50 tok/s
target reached on warm-cache short prompts.

The remaining gap to a steady ≥ 50 tok/s (about ~10%) is now
quantified:

1. `start_engine` was wired into `start_engine_with_warmup` (it had
   been silently bypassing it). JIT precompiles batch sizes
   1, 2, 4, 8, 16, 32 at boot, smoothing first-token latency.
2. CUDA stream capture itself remains blocked one level lower (initially
   error 900 — `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` — because candle
   0.9 + cudarc 0.16 dispatch every kernel through the CudaContext's
   *default* stream, whose underlying handle is the legacy null stream).
3. Honest diagnostics now distinguish "graph captured" from "JIT
   fallback" in the warmup log, and `try_capture_decode_step` emits the
   exact CUDA driver error code on every failure path.

## Stage 11 — vendored candle fork: error 900 → error 905

`scripts/setup-vendor-candle.sh` clones candle 0.9.1 into `vendor/candle/`
and applies a one-line surgical patch (`CudaDevice::new` switches from
`context.default_stream()` to `context.new_stream().w()?`). Workspace
`[patch.crates-io]` redirects `candle-core`, `candle-nn`, and
`candle-flash-attn` to the vendored copies. Combined with switching
`cuStreamBeginCapture_v2` mode from GLOBAL to RELAXED:

- **Before fork:** every batch fails with CUDA error 900
  (`STREAM_CAPTURE_UNSUPPORTED`). Capture never starts.
- **After fork:** every batch progresses past `cuStreamBeginCapture_v2`
  and fails later with CUDA error 905 (`STREAM_CAPTURE_ISOLATION`,
  *"dependency created on uncaptured work in another stream"*).

Same eager fallback path runs in both cases — steady-state remains
~42 tok/s with no regression — but the failure mode has shifted from
*"can't even start"* to *"started, then aborted on a tensor
allocation"*, which is the actual upstream-blocked operation.

The new error is surfaced by candle's per-tensor allocations
(`Tensor::zeros`, intermediate `.contiguous()` results, KV cache slot
mapping rebuilds) which go through plain synchronous `cuMemAlloc`. CUDA
stream capture forbids any allocation primitive other than
`cuMemAllocAsync` from a captured memory pool — the moment a
captured stream sees a `cuMemAlloc` it logs an event-edge to the
allocator's internal stream and aborts the capture sequence.

Closing the gap therefore requires either:
- Teaching candle's hot path to use a memory pool
  (`cuMemAllocAsync`-only inside the captured stream), or
- Pre-allocating every intermediate tensor once into the
  `DecodeBatchShared` bundle that Stage 7 already plumbs through, and
  reusing those buffers for every decode forward.

The fork is the minimum viable infrastructure for either follow-up —
without it CUDA capture is unreachable. The eager path with the new
GEMV variants and uninit allocs already delivers ~733× of baseline on
warm hardware.

## Stage 12 — pool-backed GEMV output buffers

`crates/core/src/engine/output_pool.rs` introduces a process-global
shape-keyed buffer pool.  At the top of every decode forward
`OutputPool::global().reset_cursors()` rewinds the per-shape
round-robin cursors; `MarlinLinear::forward_marlin` reserves the
output tensor for AWQ INT4-ZP linears from the pool and routes the
launch through a new `InplaceOp2` (`MarlinGemmInplaceOp`) which writes
into the pool-owned storage in place.  The 252 per-token
`cuMemAlloc` round-trips the GEMV path used to make are eliminated;
the pool only allocates on first observation of a new shape, and
reuses the same `CudaSlice` thereafter.

| Build state                                              | tok/s steady (10-run median) | tok/s peak |
| -------------------------------------------------------- | ---------------------------- | ---------- |
| Stage 11 (eager, no pool)                                | ~42–44                       | ~50        |
| **Stage 12 (eager, pooled GEMV outputs)**                | **~43.7**                    | **~48.6**  |

The eager-path gain is within run-to-run noise — `cuMemAlloc` on
8 GB Ada laptops is fast (~3-6 µs) compared to the kernel itself, so
amortising it gives a modest improvement.  The strategic value is
infrastructure: CUDA Graph capture no longer trips on GEMV
allocation; the pool is reusable for any future op that wants the
same treatment.

Capture **still** aborts with `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`,
now traced to candle's intermediate tensor allocations:
`Tensor::cat`, `.contiguous()` after a transpose, arithmetic
operators (`xs + residual`), `Tensor::zeros` from
`build_decode_batch_shared`, and the per-call output buffers inside
RoPE, paged-attention, RMSNorm, SwiGLU, and the activation kernels.
Each of those falls outside the quantization layer and keeps using
plain `cuMemAlloc`.  Bringing them onto the pool requires either
porting each kernel to its own `InplaceOpN` variant (similar surgery
to what we did for GEMV) or a deeper candle fork that swaps the
default allocator out for `cuMemAllocAsync` from a captured pool.

Python vLLM was not installed on the test box, so no head-to-head
comparison was made on this run.

## Stage 13 — capture diagnosis: V3 (cross-stream event isolation)

After Stages 11–12 capture still aborted with
`CUDA_ERROR_STREAM_CAPTURE_ISOLATION` (905). Stage 13-A added one-shot
diagnostic logging in `try_capture_decode_step` to classify the root
cause among four hypotheses (V1 — no memory pools; V2 — alloc path
bypass; V3 — cross-stream event tracking; V4 — driver-level oddity).
The probes cover: querying
`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` directly via
`cuDeviceGetAttribute`; running an alloc-only mini-capture
(`Tensor::zeros` between `cuStreamBeginCapture_v2` and
`cuStreamEndCapture`); and snapshotting `cuStreamGetCaptureInfo_v2`
status after the forward returns.

Run on RTX 4060 Laptop, server boot without `--enforce-eager`:

```
stage="13-A"   stream_ptr_null=false  memory_pools_supported=1  attr_query_ok=true
stage="13-A.2" alloc_ok=true  status_after_alloc="ACTIVE"  end_result=0
               -> Alloc-only probe SURVIVED capture. cuMemAllocAsync is active.
forward          DriverError(CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
                 "dependency created on uncaptured work in another stream")
                 (×6 — every warmup batch size 1, 2, 4, 8, 16)
```

**Verdict — V3.** V1 falsified: the device supports memory pools.
V2 falsified: a single `Tensor::zeros` allocated *inside* the capture
window does not invalidate it, so cudarc's allocator is already going
through `cuMemAllocAsync`. The error text "dependency created on
uncaptured work in another stream" matches cudarc's automatic event
tracking — every cross-resource ordering edge cudarc records reaches
back to the warmup-time event chain on resources allocated before the
capture began (model weights, KV-cache pages, internal scratch
tensors). Even RELAXED capture mode rejects that.

There is no public switch to disable cudarc's event tracking, and the
forward path touches hundreds of cudarc-managed `CudaSlice`s per layer.
A surgical fix would require either bypassing cudarc's safe API for the
capture path (re-implement allocation/launch via raw FFI) or upstream
cudarc work to make event tracking capture-aware. Both are out of
scope for this push.

Per pre-approved fallback policy, capture is closed as **infra-ready
but not activated for AWQ on cudarc 0.16**: the diagnostic logging
remains, the `try_capture_decode_step` path is intact and will benefit
automatically if/when cudarc gains capture-aware event tracking. The
44 tok/s steady baseline holds; remaining headroom (≥50 tok/s) will
come from kernel-level optimizations (Stage 13-C).

¹ The first attempt at Stage 7 placed the override on
`Qwen3ForCausalLM`; the runtime path for AWQ models is actually
`QuantizedQwen3ForCausalLM` (registry's `build_quant`), so the override
was dormant. Fixed by mirroring the `_with_shared` plumbing into
`models/qwen3_quantized.rs` and overriding
`forward_decode_batch_with_ctx` on the quantized class. Confirmed live
via `tracing::info!` instrumentation: `decode_shared_present=true` on
every forward.

Stage 8's startup win is the headline so far: AWQ→GPTQ qweight repack
moved from a strided scalar Rust loop (5+ minutes for 36×7 = 252 linear
layers, ~2.85 G random-access reads) to a single CUDA kernel
(`awq_to_gptq_qweight_int4` in `crates/core/kernels/marlin_gemm.cu`).

## Diagnosis: the remaining bottleneck is per-linear kernel cadence

Per-forward timing instrumentation in
`QuantizedQwen3ForCausalLM::forward_decode_batch_with_ctx` shows:

```
total_ms        = 16720
embed_ms        =     0.05
first_layer_ms  =   591    (cold)
layers_total_ms = 16670    (36 layers × ~460 ms each)
norm_ms         =     0.02
lm_head_ms      =    50
```

So 99.7 % of the decode-step wall-clock lives in 36 decoder layers, and
embedding / final norm / lm_head are negligible. Each decoder layer
issues 7 quantized linear forwards (q/k/v/o + gate/up/down) plus
RMSNorm, RoPE, paged_attention. With ~460 ms / layer at batch=1 that's
roughly 60 ms per `MarlinLinear::forward_marlin` call.

Marlin's INT4 GEMM is tuned for M ≥ 64; at M = 1 (single-token decode,
batch=1) the kernel is heavily SM-underutilised. Combined with the
candle CustomOp1 wrapper boundaries — each `apply_op1` has small but
non-zero per-call overhead — this stacks into the ~60 ms per call we
observe.

vLLM hides this with two mechanisms we don't yet have on the AWQ path:

1. **CUDA graphs over the whole decoder layer** so the per-call overhead
   amortises to a single `cuGraphLaunch` per token. Our CUDA graph
   infrastructure exists but `--enforce-eager` was used here on
   purpose — and Qwen3-AWQ's quantized kernels haven't yet been
   verified compatible with capture.
2. **A specialised batch=1 GEMV kernel** (`marlin_gemv` /
   `gemv_awq_kernel`) that skips the full Marlin tile dispatch in favour
   of a thin GEMV path. The current Marlin kernel always enters its
   tile-mma loop even when M = 1.

Both are next-stage fixes; they are out of scope for the present
landed work and are tracked separately.

## Stage 9 — what changed (2026-05-06)

Two latent gating bugs blocked the entire Marlin/GEMV path on Qwen3-AWQ.
Both were undiagnosed until we wired one-shot `tracing::info!` markers
inside `AwqMarlinLinear::load_weights`, `MarlinLinear::forward_marlin`,
`forward_fallback`, and the new `cuda_fwd_awq_gemv`, then ran the server.

1. **`AwqWeightLoader::load_linear` always built `AwqLinear` directly**
   (`crates/core/src/quantization/weight_loader.rs:403-510`), bypassing
   `AwqConfig::create_linear` where Stage 1 had wired the Marlin route.
   Fix: mirror the Marlin gate inside the weight-loader entry path so
   AWQ checkpoints construct `AwqMarlinLinear` whenever
   `use_marlin && can_use_marlin_for_shape` clears.

2. **`MarlinLinear::process_weights` repacked qweight to a Marlin
   tile-MMA layout via a kernel we do not ship.** The PTX entry
   `repack_gptq_to_marlin_int4` is missing from `marlin_gemm.ptx`
   (`grep .entry`); all four live INT4/INT8 GEMM kernels plus the new
   `awq_gemv_int4_bf16` consume raw `[K/8, N]` GPTQ-style qweight and
   un-permuted `[num_groups, N]` scales. Until a real tile-MMA kernel
   ships, `process_weights` is a documented no-op.

3. **HF AWQ scales arrive in F16; the Marlin and GEMV kernels both
   need BF16.** `AwqMarlinLinear::load_weights` now converts at load
   time so the per-call cast is free.

4. **Decode-path GEMV kernel** (`crates/core/kernels/awq_gemv.cu`,
   `MarlinGemmOp::cuda_fwd_awq_gemv`): one warp per block, one thread
   per output column, FP32 accumulation, BF16 output, scale/zp cached
   per group. Routed for `M ≤ 16` AWQ-INT4 with zero points and
   without `g_idx`.

Net effect: **0.06 → 11.6 tok/s** on the same hardware and command line
(`--enforce-eager`, `--num-blocks 96`, `--max-requests 4`,
`--max-model-len 1024`). 193× single-stream throughput improvement.

Per-layer profile on the new path: ~10 ms / 36 layers
(7 quantized linear forwards each), down from ~460 ms / 36.
Effective memory bandwidth on the GEMV kernel is ~12 GB/s of the
~256 GB/s peak — leaving substantial headroom that Stage 10 (CUDA-graph
capture, output buffer pooling, GPU positions for RoPE) is expected
to close to ≥ 50 tok/s.

## Stage attribution (cumulative on this branch)

- **Stage 1**  AWQ→Marlin layout repack at load via `AwqMarlinLinear`,
  feeding the Marlin INT4 kernel correctly (was silently wrong before).
- **Stage 2A** Repaired the `marlin` / `flashinfer` /
  `cuda-layernorm` / `cuda-fused-activations` / `cuda-moe` builds
  against the bumped cudarc/candle stack.
- **Stage 3**  Folded additive logit modifiers into the GPU sampler
  (`gpu_apply_logits_diff`).
- **Stage 4**  Wired CUDA-graph capture into warmup (infra only;
  `--enforce-eager` for these measurements).
- **Stage 5**  Detokenize tail of `generated_token_ids` instead of all.
- **Stage 7**  `DecodeBatchShared` built once per forward; reused by
  every attention layer through `ForwardContext.decode_shared`.
  Wired on `Qwen3ForCausalLM` (eager path) and
  `QuantizedQwen3ForCausalLM` (AWQ runtime path).
- **Stage 8**  GPU `awq_to_gptq_qweight_int4` kernel — startup
  ~5 min → ~21 s.
- **Stage 9**  Three coordinated changes that activate every Marlin-path
  optimisation Stages 1, 7, 8 had silently been gated out of:
  (9-A) `AwqWeightLoader` routes through `AwqMarlinLinear`;
  (9-B) `MarlinLinear::process_weights` becomes a no-op (no tile-MMA
  kernel is shipped, so the old repack would have crashed at startup);
  (9-C) new `awq_gemv_int4_bf16` kernel takes over for `M ≤ 16` and
  scale/zp dtype is normalised at load time.
  Single-stream decode: 0.06 → 11.6 tok/s.
- **Stage 10-1** RoPE no longer round-trips position indices through
  the host. The U32 positions tensor is handed straight to the kernel
  (the bit pattern of u32 and i32 matches for non-negative position
  indices). 144 device↔host syncs per token eliminated; small numerical
  win (11.6 → 13.0 tok/s), but the bigger value is unblocking CUDA-graph
  capture of the RoPE op for a future stage.
- **Stage 10-α** Split-K GEMV. The serial kernel’s single warp per
  block left ~10% of Ada SM warp slots active at Qwen3 N = 2560.
  Two new variants fan each output column across 4 or 8 cooperating
  threads (one warp each), with a cascading dispatch that picks the
  largest split the shape allows. ~28% → ~56% theoretical occupancy
  doubling, ~50% wall-clock improvement: 13 → 20 tok/s. Beyond ×8
  the kernel is memory-bandwidth-bound — adding more threads no
  longer helps without restructuring qweight access.
- **Stage 10-β** vec4 LDG on transposed qweight. The previous
  layouts (`[K/8, N]`) put the K-axis on the strided dimension —
  every packed_k step landed in a fresh cache line. `AwqMarlinLinear`
  now stores qweight as `[N, K/8]` (one strided GPU copy at load
  time), and the new `awq_gemv_int4_kt_bf16` kernel issues one
  `uint4` LDG (16 B = 4 packed words = 32 packed int4 weights) per
  inner iteration, quartering the per-thread issue rate. AWQ INT4-ZP
  dispatch routes here for every M (decode and prefill); the older
  variants stay as fallbacks if a future shape misses the gate.
  20 → 40 tok/s.
- **Stage 10-γ** Skip zero-init on the GEMV output buffer. Every
  element is written exactly once by the kernel; the implicit memset
  inside `alloc_zeros::<bf16>` was dead work on each of the 252
  per-token GEMV launches. Switched to `unsafe alloc::<bf16>`. Pushed
  steady-state to 42 tok/s with peaks above 50.
- **Stage 10-δ** Same uninit-alloc swap on every other hot-path
  output buffer: RoPE (overwritten by `memcpy_dtod` before the
  kernel), paged-attention V1, RMSNorm BF16, SwiGLU (BF16 + F16),
  and the GeGLU activation. ~108 extra zero-init memsets per token
  removed; steady-state 42 → 44 tok/s.

Python vLLM was not present in the test environment (no apples-to-apples
comparison was made on this run). The vllm-rust numbers above are real
measurements from a streaming `/v1/chat/completions` request against the
running server; raw per-token timing showed pace was ~70 seconds between
4-token bursts (multi-step accumulation), giving an effective decode of
~0.03–0.05 tok/s.

## Diagnosis: where the time is going

The measured throughput is roughly three orders of magnitude below the
~50–80 tok/s that an Ada-class GPU should sustain on Qwen3-4B-AWQ via
Marlin INT4. With graph capture disabled and a proper INT4 path, this is
not a "small constant overhead" problem — something in the per-step path
is doing host-blocking work.

Strong signals from observed behaviour:

- GPU utilisation hovers at ~32 % during decode → there is a CPU/GPU
  sync per step, not a kernel-bound issue.
- Token output arrives in bursts of 4 (exactly `--multi-step-count 4`)
  separated by ~70 s gaps. So one decode "step" — four amortised
  forwards plus the host work between them — costs around 70 s. That
  is ~17 s per forward for a 36-layer 4B-param model: ~470 ms per
  decoder layer. A real Marlin-on-Ada layer should be sub-millisecond.

Most likely culprits, in order of probability:

1. **AwqMarlinLinear is reached but Marlin isn't actually used.**
   `MarlinLinear::forward_marlin` requires the `marlin` feature *and*
   a CUDA tensor; otherwise it goes through `forward_fallback` →
   `gptq_cuda::gptq_gemm`. If something inside the new
   `try_capture_decode_step` warmup path leaves the runner's
   `is_initialized` flag inconsistent, replays can silently fall back
   to a slow path. Add a one-shot `tracing::info!` at the first
   `forward_marlin` / `forward_fallback` call to confirm which is
   live.
2. **FlashInfer plan() metadata forces a `to_vec1::<u32>()` per
   step.** MEMORY.md flags exactly this: `plan()` ALL variants need
   indptr/len arrays as **host** pointers (copy from GPU each step).
   On a tiny 1-token decode this synchronisation can dominate the
   step.
3. **AWQ→GPTQ qweight repack at load is an unoptimised scalar
   loop with strided memory access** (~52 M random-access reads
   per layer × 252 layers). Loads "Building model …" took several
   minutes. Not the per-token cost, but it makes startup
   unpleasant. Should be moved to a CUDA kernel — vLLM's
   `awq_marlin_repack.cu` does this in ~10 ms total.

## Suggested next steps

1. Add `tracing::info_span!` once-per-process around the first
   `forward_decode_batch` and the first `forward_marlin` /
   `forward_fallback` invocation; rebuild, rerun, confirm which
   GEMM kernel is actually executing each layer. This is the
   first thing to do — every guess below is downstream of this
   answer.
2. If Marlin is *not* live: check `MarlinLinear::is_initialized`
   after `AwqMarlinLinear::load_weights` — the inner runner may
   need an explicit `mark_warmed_up()` when the weights are loaded
   *outside* the `try_capture_decode_step` flow (e.g. on the eager
   path).
3. If Marlin *is* live and per-layer is still ~470 ms: capture an
   `nsys profile` of one decode step. The trace will show whether
   FlashInfer plan(), `slot_mapping.to_vec()`, or some other
   host-sync path is dominating.
4. Move `awq_to_gptq_qweight` and `repack_awq_nibbles` for qzeros
   to a CUDA kernel (mirror `awq_marlin_repack.cu` from vLLM
   reference). One kernel call replaces 504 host↔device copies and
   ~13 G scalar Rust ops.
5. Get Python vLLM installed in a venv on the test box for a real
   side-by-side comparison; otherwise we're benching against an
   imagined target.

## What each stage of `perf/awq-cuda-fast-path` contributes

- **Stage 1**  AWQ→Marlin layout repack at load. Removes 252 host↔device
  round-trips per token (one per linear layer), the dominant cost.
- **Stage 2A** Repaired the marlin/flashinfer/cuda-layernorm/cuda-fused-
  activations/cuda-moe builds so the production feature set actually
  links — without it we were stuck on `cuda-kernels` only.
- **Stage 3**  Folded additive logit modifiers (logit_bias, freq+pres
  penalties, banned tokens, bad words) into the GPU sampler via a single
  `index_add`, so configurations with these no longer pull the full
  logits matrix to host.
- **Stage 4**  Wired `cuStreamBeginCapture_v2` / `cuStreamEndCapture` into
  warmup. Decode replays now skip per-step kernel-launch latency.
- **Stage 5**  Detokenize only the tail of `generated_token_ids` when
  checking stop strings (was O(N) per step).
