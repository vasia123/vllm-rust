# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│  vllm-server (binary)                                       │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │   CLI    │  │  HTTP API (axum)                         │ │
│  │  (clap)  │  │  /v1/completions, /v1/chat/completions   │ │
│  └────┬─────┘  └──────────────┬───────────────────────────┘ │
│       │                       │                              │
│       └───────────┬───────────┘                              │
│                   ▼                                          │
│            EngineHandle                                      │
└───────────────────┼──────────────────────────────────────────┘
                    │ (mpsc channel)
┌───────────────────┼──────────────────────────────────────────┐
│  vllm-core (library)                                         │
│                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Engine Loop (tokio task)                               │ │
│  │  ┌───────────┐  ┌───────────┐  ┌────────────────────┐  │ │
│  │  │ Scheduler │  │   Model   │  │  KV Cache Manager  │  │ │
│  │  │  (FCFS)   │  │ (Forward) │  │  (Paged Blocks)    │  │ │
│  │  └───────────┘  └─────┬─────┘  └────────────────────┘  │ │
│  │                        │                                 │ │
│  │              ┌─────────┴─────────┐                       │ │
│  │              ▼                   ▼                       │ │
│  │      Shared Layers          Model Impl                   │ │
│  │    (RoPE, GQA, MLP)     (Qwen3, Llama, ...)            │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Crates

### vllm-core

Core inference library. No I/O, no HTTP, no CLI — pure computation and scheduling.

| Module | Responsibility |
|--------|---------------|
| `models/` | Model architectures + factory registry |
| `layers/` | Shared ops: RoPE, paged attention, SwiGLU, masks |
| `engine.rs` | Inference loop, speculative decoding, `ModelForward` trait |
| `kv_cache/` | Block-based paged KV cache management |
| `scheduler.rs` | Request scheduling with preemption |
| `loader.rs` | HuggingFace Hub download + safetensors loading |
| `tokenizer.rs` | Tokenization + chat templates (Jinja2) |
| `config.rs` | Model configuration (parsed from `config.json`) |
| `request.rs` | Per-request state tracking |

### vllm-server

HTTP server and CLI binary. Thin layer over vllm-core.

| Module | Responsibility |
|--------|---------------|
| `main.rs` | CLI parsing, model loading, engine startup |
| `api/` | Axum router, OpenAI-compatible endpoints, SSE streaming |

## Key Abstractions

### ModelForward (trait)

```rust
pub trait ModelForward: Send + 'static {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor>;

    fn device(&self) -> &Device;
}
```

Every model implements this. The engine is generic over `M: ModelForward`.
`Box<dyn ModelForward>` also implements the trait (blanket impl), enabling the factory pattern.

### Engine Loop

The engine runs as a tokio task with these phases per iteration:

1. **Drain commands** — new requests from `EngineHandle`
2. **Block-wait if idle** — no active requests, await next command
3. **Schedule** — decide which requests to prefill/decode
4. **Handle preemptions** — free KV cache for preempted requests
5. **Execute prefills** — first forward pass for new prompts
6. **Execute decodes** — next-token generation for running requests
7. **Check completion** — EOS or max_tokens reached, notify callers

### KV Cache

Block-based paged attention:

- **BlockPool** — free list of cache blocks (fixed-size memory chunks)
- **BlockTable** — per-request mapping of logical positions to physical blocks
- **CacheEngine** — per-layer K/V tensor storage, write/read by slot index
- **KVCacheManager** — coordinates pool + per-layer engines

### Scheduler

FCFS with preemption:

- Admits requests up to `max_running_requests` and token budget
- Preempts newest-first when blocks run out
- Re-admits preempted requests when memory frees up

### Speculative Decoding

Draft-then-verify pipeline:

1. Draft model generates K candidate tokens autoregressively
2. Target model verifies all K+1 positions in a single forward pass
3. Greedy comparison accepts matching prefix + one bonus/correction token
4. KV caches trimmed to actual accepted length

## Data Flow

```
Request → Tokenize → Scheduler → Prefill (full prompt forward)
                         ↓
                      Decode (single token forward, loop)
                         ↓
                      EOS/MaxTokens → Detokenize → Response
```

With streaming, each generated token is sent as an SSE event before the next decode step.

## Design Principles

- **No premature abstraction** — shared layers earned through repetition (Qwen3 + Llama)
- **Per-model files** — each architecture owns its full structure, imports shared ops
- **Zero dynamic dispatch in hot path** — engine is generic `<M: ModelForward>`
- **Factory only at startup** — `Box<dyn ModelForward>` used once during model construction
