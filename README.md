# vllm-rust

A Rust-native LLM inference engine — an alternative to vLLM. PagedAttention,
continuous batching, speculative decoding, a broad quantization stack, and an
OpenAI-compatible HTTP API, with custom CUDA kernels via
[Candle](https://github.com/huggingface/candle).

> **Status — research-grade.** A handful of model families are verified to run
> coherently on real weights (see [Tested models](#tested-models)); 190+ other
> architectures are *implemented with unit tests but not yet run end-to-end* —
> they may work as-is or may need fixing. The HTTP API is broad but pre-1.0:
> behaviour and flags can change between minor versions (see
> [docs/RELEASING.md](docs/RELEASING.md)).

## Features

- **Paged KV cache** — block-based memory management, zero fragmentation;
  heterogeneous per-layer caches (e.g. Gemma sliding/full attention), FP8/INT8
  KV cache.
- **Continuous batching** — dynamic scheduling, chunked prefill (on by
  default), preemption under memory pressure.
- **Quantization** — native GGUF (incl. **I-quants** IQ2/IQ3/IQ4 that candle
  cannot parse), AWQ, GPTQ/Marlin, EXL3, FP8, and more (see
  [the table](#quantization)).
- **Speculative decoding** — draft models, EAGLE, Medusa, MTP, n-gram.
- **OpenAI + Anthropic API** — `/v1/completions`, `/v1/chat/completions`
  (streaming), `/v1/embeddings`, `/v1/messages`, plus structured output
  (JSON-schema / grammar via xgrammar), tool calling, and reasoning parsers.
- **CUDA kernels** — RMSNorm, RoPE, paged attention, GPU sampling, fused MoE,
  q8_1 MMVQ for I-quant decode, and FlashInfer integration.

## Requirements

- Rust (edition 2021; a recent stable toolchain).
- NVIDIA GPU + CUDA toolkit **12.x** for GPU inference. Compute capability
  7.5+ (kernels target the detected arch; FP8 paths need 8.9+). An 8 GB card
  runs quantized 8–12B models.
- Disk for model weights (auto-downloaded from the HuggingFace Hub, or pass a
  local path).

## Build

```bash
# Full GPU build (recommended): CUDA kernels + FlashInfer + xgrammar
cargo build --release -p vllm-server --features cuda-full
```

The release profile pins `jobs = 1` (see `.cargo/config.toml`) because
FlashInfer's template-heavy kernels peak `ptxas` memory; on a low-RAM host
(e.g. WSL2) this avoids OOM during the build. `cargo build --release` with no
features produces a CPU-only build (limited; most kernels are gated behind
`cuda*` features).

## Run

### HTTP server

```bash
# A HuggingFace repo:
cargo run --release -p vllm-server --features cuda-full -- serve \
  --model Qwen/Qwen3-0.6B --port 8000

# A local GGUF file (pass the tokenizer.json that ships alongside it; the
# server also auto-detects one in the same directory):
cargo run --release -p vllm-server --features cuda-full -- serve \
  --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json \
  --max-model-len 4096 --num-blocks 96 --text-only --port 8000
```

Common flags: `--draft-model` / `--num-speculative-tokens` (speculative
decoding), `--num-blocks` (KV cache blocks), `--max-model-len` (bound context
to fit a small GPU), `--gpu-memory-utilization`, `--tokenizer`, `--text-only`
(skip vision/audio towers on multimodal checkpoints). `vllm-server --version`
prints the release.

> **Instruct models need the chat template.** Send instruction-tuned
> checkpoints (e.g. `gemma-…-it`) through `/v1/chat/completions`, not raw
> `/v1/completions` — a raw prompt is out-of-distribution and degenerates into
> garbage. This is the model, not the engine.

### CLI generation

```bash
cargo run --release -p vllm-server --features cuda-full -- generate \
  --model Qwen/Qwen3-0.6B --prompt "Hello, world" --max-tokens 64
```

(The `generate` subcommand takes a HuggingFace repo id; for a local GGUF file
use `serve` + the HTTP API.)

### API

```bash
# Chat (use this for instruct models)
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'

# Completions (+ "stream": true for SSE)
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id>","prompt":"Hello","max_tokens":32}'

# Embeddings
curl http://localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"<id>","input":"some text"}'
```

`<id>` is whatever you passed to `--model` (a repo id, or the GGUF file path);
`GET /v1/models` lists it.

## Tested models

These have been **run end-to-end on real weights in this environment** and
produce coherent output. Start here if you want something that just works.

| Model | Formats verified | Notes |
|---|---|---|
| **Qwen3** (0.6B / 4B / 8B) | bf16/fp16, AWQ, EXL3, FP8 KV cache | Primary dev model; heavily perf-tuned (EXL3 ~88% of ExLlamaV3 at 4.0bpw). |
| **Gemma 4** (12B, E4B) | GGUF: I-quants (Unsloth UD `IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_XS`) + K-quants; FP8 KV cache | Runs on 8 GB; `gemma-4-12b-it-UD-IQ3_XXS` ≈ 21.5 tok/s decode. Heterogeneous sliding/full KV + K-as-V handled. |
| **EmbeddingGemma** (gemma3 GGUF) | GGUF | `/v1/embeddings`; parity with ollama (cos ≈ 0.998). |
| **Qwen3-Embedding** | safetensors | `/v1/embeddings`. |

### Implemented but **not** verified end-to-end

The architecture registry has **190+** entries — Llama / Llama 4, Mistral,
DeepSeek-V2/V3 (MLA), Mixtral and other MoE, Phi, GLM, Mamba/SSM, many
vision-language models (CLIP/SigLIP/Qwen-VL/Gemma-VL/InternVL/…), and the
EAGLE/Medusa/MTP speculators. **These have implementations and unit tests but
have not been loaded and run on real weights here.** They may work out of the
box, or may need fixing — treat them as untested. Reports (and fixes) welcome.

## Quantization

| Format | Status |
|---|---|
| GGUF K-quants (`Q*_K`) | Tested (Gemma 4) |
| GGUF I-quants (`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_XS`) | Tested (Gemma 4) — native, candle cannot parse these (ADR 0025) |
| AWQ / AWQ-Marlin | Tested (Qwen3) |
| EXL3 | Tested (Qwen3) |
| FP8 (weights + KV cache) | Tested (Qwen3, Gemma 4 KV) |
| GPTQ / Marlin, ModelOpt, FBGEMM-FP8, BitsAndBytes, … | Implemented, unverified |

## Releases & versioning

[Semantic versioning](https://semver.org); see
[CHANGELOG.md](CHANGELOG.md) and [docs/RELEASING.md](docs/RELEASING.md).
Releases are tagged `vX.Y.Z`; CUDA binaries are built locally and attached to
the [GitHub release](https://github.com/vasia123/vllm-rust/releases) (they are
too heavy/environment-specific for free CI runners).

## Project structure

```
crates/
  core/    — inference engine (models, layers, KV cache, scheduler, sampling, quantization)
  server/  — HTTP server + CLI binary
docs/      — architecture, ADRs, perf history, release process
```

See [docs/architecture.md](docs/architecture.md) and the ADRs in `docs/adr/`.

## Tests & development

```bash
cargo test -p vllm-core --lib                 # core unit tests (CPU)
cargo test -p vllm-core --features cuda-kernels  # + GPU kernel tests
cargo fmt --all                               # format (enforced by pre-commit hook)
cargo clippy --all-targets --features cuda    # lint (zero warnings; enforced)
```

GPU integration tests are gated behind the `cuda*` features and a present GPU.
See [docs/adding-a-model.md](docs/adding-a-model.md) to add an architecture.

## License

MIT
