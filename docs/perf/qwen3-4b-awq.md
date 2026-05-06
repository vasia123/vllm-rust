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

Fill in after running on the target box. Format: median tokens/sec per
request and aggregate across the batch. Warmup run discarded.

| Backend             | Concurrency | TTFT (ms) | tok/s/req | Aggregate tok/s |
| ------------------- | ----------- | --------- | --------- | --------------- |
| vllm-rust (`cuda-full`) | 1       |           |           |                 |
| vllm-rust (`cuda-full`) | 4       |           |           |                 |
| vllm-rust (`cuda-full`) | 8       |           |           |                 |
| Python vLLM         | 1           |           |           |                 |
| Python vLLM         | 4           |           |           |                 |
| Python vLLM         | 8           |           |           |                 |

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
