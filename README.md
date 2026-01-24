# vllm-rust

Rust-native LLM inference engine with PagedAttention, continuous batching, and speculative decoding. OpenAI-compatible API.

## Features

- **Paged KV Cache** — block-based memory management, zero fragmentation
- **Continuous Batching** — dynamic scheduling, preemption under memory pressure
- **Speculative Decoding** — draft model acceleration with automatic verification
- **Multi-Model** — pluggable architecture registry (Qwen3, Llama, extensible)
- **OpenAI API** — `/v1/completions` and `/v1/chat/completions` with streaming
- **CUDA** — GPU inference via Candle

## Requirements

- Rust 1.70+
- CUDA toolkit (for GPU inference)
- ~2GB disk for model weights (auto-downloaded from HuggingFace Hub)

## Build

```bash
cargo build --release
```

## Run

### HTTP Server

```bash
cargo run --release -p vllm-server -- serve \
  --model Qwen/Qwen3-0.6B \
  --port 8000
```

Options:
- `--model` — HuggingFace model ID (default: `Qwen/Qwen3-0.6B`)
- `--draft-model` — draft model for speculative decoding
- `--num-speculative-tokens` — speculative tokens per step (default: 3)
- `--port` — listen port (default: 8000)
- `--host` — bind address (default: `0.0.0.0`)
- `--num-blocks` — KV cache blocks (default: 512)
- `--max-requests` — max concurrent requests (default: 8)

### CLI Generation

```bash
cargo run --release -p vllm-server -- generate \
  --model Qwen/Qwen3-0.6B \
  --prompt "Hello, world" \
  --max-tokens 64
```

### API Usage

```bash
# Completions
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello", "max_tokens": 32}'

# Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32}'

# Streaming
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello", "max_tokens": 32, "stream": true}'
```

## Project Structure

```
crates/
  core/       — inference engine library (models, layers, KV cache, scheduler)
  server/     — HTTP server + CLI binary
docs/         — development guides
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture overview.

## Tests

```bash
cargo test --workspace
```

Tests requiring a downloaded model are `#[ignore]`-d. Run them with:

```bash
cargo test --workspace -- --ignored
```

## Development

```bash
cargo fmt                          # format
cargo clippy --workspace           # lint
cargo check --workspace            # type-check
```

See [docs/adding-a-model.md](docs/adding-a-model.md) for how to add new model architectures.

## License

MIT
