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

Python vLLM was not installed on the test box, so no head-to-head
comparison was made on this run.

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
