# Speculative Decoding

## Overview

Speculative decoding uses a small "draft" model to predict K tokens ahead, then verifies them all at once with the main "target" model. When predictions match, you get K+1 tokens per target forward pass instead of 1.

## How It Works

```
Draft model:   generate K tokens autoregressively (cheap, fast)
Target model:  verify K+1 positions in one forward pass (expensive, batched)
Compare:       accept matching prefix + bonus/correction token
Rollback:      trim KV caches to actual accepted length
```

### Example (K=3)

```
Draft produces:  [A, B, C]
Target verifies: position 0→A, 1→B, 2→X (mismatch at position 2)
Result:          accept [A, B] + correction token X = 3 tokens total
                 (instead of 3 separate decode steps)
```

If all K drafts match:
```
Draft produces:  [A, B, C]
Target verifies: position 0→A, 1→B, 2→C, bonus position 3→D
Result:          accept [A, B, C, D] = K+1 = 4 tokens total
```

## Usage

### CLI

```bash
cargo run --release -p vllm-server -- generate \
  --model Qwen/Qwen3-0.6B \
  --draft-model Qwen/Qwen3-0.6B \
  --num-speculative-tokens 3 \
  --prompt "Hello" \
  --max-tokens 64
```

### HTTP Server

```bash
cargo run --release -p vllm-server -- serve \
  --model Qwen/Qwen3-0.6B \
  --draft-model Qwen/Qwen3-0.6B \
  --num-speculative-tokens 3
```

The draft model can be any model that implements `ModelForward`. Typically a smaller variant of the same family (e.g., Qwen3-0.6B as draft for Qwen3-1.7B).

## Tuning

- **`--num-speculative-tokens` (K)** — more tokens = higher throughput when acceptance is high, but wastes compute when acceptance is low. Typical values: 3-5.
- **Draft model choice** — must share vocabulary with target. Smaller = faster draft phase, but lower acceptance rate.
- **Memory** — draft model needs its own KV cache (same `--num-blocks` allocation).

## Implementation Details

Key files:
- `engine.rs:start_engine_with_draft()` — launches speculative engine loop
- `engine.rs:engine_loop_speculative()` — the core speculative loop
- `engine.rs:execute_speculative_decode()` — draft phase + verify phase + rollback

Both target and draft models maintain independent KV caches and block tables. On acceptance mismatch, both caches are trimmed to the actual accepted sequence length via `BlockTable::trim_to()`.

## Streaming

Speculative decoding works with streaming (`"stream": true`). Multiple accepted tokens from a single speculative round are sent as individual `StreamEvent::Token` events.
