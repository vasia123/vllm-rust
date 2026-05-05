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
│  │    (AttentionBlock,      (Llama, Qwen3, Gemma2,          │ │
│  │     SwiGluMlp, RoPE,      Mixtral, DeepSeek-MLA,         │ │
│  │     RmsNorm, MoELayer)    Mamba, …)                      │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Crates

### vllm-core

Core inference library. No I/O, no HTTP, no CLI — pure computation and scheduling.

| Module | Responsibility |
|--------|---------------|
| `models/` | Model architectures + factory registry |
| `layers/` | Shared ops: `AttentionBlock` (config-driven attention), RoPE, paged attention, SwiGLU, RmsNorm, masks |
| `moe/` | Shared MoE infrastructure: `MoERouter`, `MoELayer`, fused experts, EPLB |
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

## Key Building Blocks

### AttentionBlock (`layers::attention::block`)

Config-driven, TP-aware reusable attention covering ~80% of decoder-only
architectures. Construction is declarative — start from
`AttentionConfig::gqa(...)` and chain `with_*` builders:

| Axis | Config knob | Used by |
|---|---|---|
| Per-head QK RMSNorm | `with_qk_norm(PerHead, eps)` | Qwen3, Olmo2, Cohere, Bailing-MoE |
| Per-head QK LayerNorm | `with_qk_norm(PerHeadLayerNorm, eps)` | Persimmon |
| QK norm after RoPE | `with_qk_norm_after_rope()` | Hunyuan |
| Attention-logit softcap | `with_softcap(cap)` | Gemma2 |
| Sliding-window mask | `with_sliding_window(w)` | Mistral, Gemma2 even layers |
| Bias on Q/K/V/O subset | `with_bias(AttentionBias::*)` | Qwen2, GPT-2, Falcon, Persimmon |
| Custom q-scale | `with_scale(s)` | Gemma2 query_pre_attn_scalar, Granite |
| Fused QKV (`[Q\|K\|V]`) | `with_qkv_fused()` | Phi3, GPT-2, GPT-BigCode, ChatGLM, DBRX, Falcon |
| Bypass RoPE | `with_bypass_rope()` | Llama4 alternating layers, GPT-2/OPT (abs. pos.) |
| Custom proj names | `with_proj_names(ProjNames { … })` | Falcon (`query_key_value`/`dense`), GPT-2 (`c_attn`/`c_proj`), Exaone (`out_proj`), … |
| Asymmetric QKV input | `with_qkv_input_size(n)` | Eagle3 layer 0 (concat embeds + hidden) |
| Partial RoPE | `RotaryEmbedding::new_partial` | GLM4, GPT-J/NeoX, Nemotron, ChatGLM, MiniMax-M2, … |

Architectures with genuinely exotic attention math (DeepSeek MLA, SSM
families, ALiBi, linear attention, gated/output-gated attention,
asymmetric V-dim, flat-dim QK norm, multi-modal/Fourier RoPE) stay
bespoke and are listed in `block.rs`.

See `docs/adding-a-model.md` for the full add-a-model walkthrough.

### MoE Infrastructure (`moe/`)

- `MoERouter` trait + `TopKRouter` (softmax/sigmoid scoring, optional
  grouped top-k for DeepSeek V3, optional `e_score_correction_bias`).
- `MoELayer` / `MoELayerWithShared` — config-driven expert layers, with or
  without shared experts.
- `EPMoELayer` — expert-parallel variant.
- `QuantizedMoELayer` — quantized expert weights.
- `eplb` / `eplb_execute` — expert load balancing across ranks.

## Design Principles

- **Earn abstraction through repetition.** Shared building blocks
  (`AttentionBlock`, `MoELayer`, `RmsNorm`, `SwiGluMlp`) consolidate
  patterns that appear in 3+ models; one-off architectures stay bespoke.
- **Declarative > imperative for structure.** Models describe attention
  shape via config knobs, not by reimplementing the forward kernel.
- **Per-model files for the rest.** Decoder-layer wiring, residual
  patterns, MoE specifics, position-encoding tweaks, and outer wrappers
  live in `models/<name>.rs`.
- **Zero dynamic dispatch in hot path.** Engine is generic
  `<M: ModelForward>`; `Box<dyn ModelForward>` is used only at startup.
- **Bit-exact migrations.** Refactoring a model to use shared blocks
  delegates to the same numerical primitives — never a behavior change.
  Latent bugs uncovered during migration (e.g. an `o_proj` accidentally
  not applied) are documented as fixes in the corresponding commit.
