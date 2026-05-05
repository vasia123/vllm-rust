# ADR-0010: Universal `AttentionBlock` for Decoder-Only Architectures

## Status

Accepted

## Context

Through the first 90+ supported architectures the project accumulated
~180 separate `XxxAttention` structs across `crates/core/src/models/`,
each ~200–300 LOC of mostly identical scaffolding around
`paged_attention()`. Variations between models came down to a small set of
declarative knobs:

- per-head Q/K RMSNorm (Qwen3, Olmo2, Cohere, Bailing-MoE),
- attention-logit softcap (Gemma2),
- per-layer alternating sliding window (Gemma2),
- bias on any subset of Q/K/V/O (Qwen2, GPT-2, OPT, Falcon, Persimmon),
- custom q-scale (Gemma2 `query_pre_attn_scalar`, Granite multiplier),
- fused QKV `[Q | K | V]` projection (Phi3, GPT-2/BigCode, ChatGLM,
  DBRX, Falcon, …),
- bypass RoPE for layers using absolute / no positional encoding
  (Llama4 alternating layers, GPT-2, OPT),
- partial RoPE (GLM4, GPT-J/NeoX, Nemotron, ChatGLM, MiniMax-M2, …),
- custom projection names (`out_proj`, `dense`, `c_attn`, `c_proj`,
  `query_key_value`, `wo`, …),
- per-head LayerNorm QK (Persimmon),
- QK-norm ordering after RoPE (Hunyuan),
- asymmetric Q/K/V input dim (Eagle3 layer 0).

Every new architecture re-implemented the same TP-aware projection
loading, RoPE application, paged-attention dispatch, and CUDA fast-path
decode — frequently with subtle copy-paste bugs (one of which, a missing
`o_proj` application in Molmo2, was uncovered during the migration).

Python vLLM addresses this with a config-driven `Attention` class.
We need the same in Rust without sacrificing TP correctness or the
zero-overhead direct dispatch model that the engine relies on.

## Decision

Introduce `crates/core/src/layers/attention/block.rs` containing:

- `AttentionConfig` — a plain-data builder describing the structural
  shape of the attention block (heads/kv_heads/head_dim/hidden_size +
  the declarative knobs above).
- `AttentionBlock` — a TP-aware module that owns the projections, the
  rotary embedding, optional QK-norm parameters, and dispatches
  forward/forward_decode_batch through one of two paths:
  - **Fast path** (no softcap, no sliding window, no custom scale,
    `qk_norm_after_rope = false`): delegates to existing
    `paged_attention()` and `paged_attention_cuda()` primitives —
    **bit-exact** to the legacy bespoke implementations.
  - **Manual path** (any of the above set): explicit
    `cache_engine.write` + `cache_engine.read` + matmul attention with
    custom mask and softcap — bit-exact to the Gemma2/Gemma3
    implementations it replaced.

Models migrate to the block as a thin shim:

```rust
struct LlamaAttention {
    inner: AttentionBlock,
}

impl LlamaAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, cfg.hidden_size,
        );
        let rope = RotaryEmbedding::new(/* … */)?;
        Ok(Self { inner: AttentionBlock::new(&attn_cfg, vb, pg, rope)? })
    }

    fn forward(&self, …) -> Result<Tensor> {
        self.inner.forward(…)
    }
}
```

### Key design choices

1. **Structural, not algorithmic.** The block owns the *shape* of
   attention (which projections exist, with what biases, what norms) but
   delegates all numerics to the same primitives the bespoke
   implementations used. This keeps migrations trivially bit-exact and
   avoids re-deriving correctness for ~90 models.

2. **Two execution paths, not N feature flags.** Once any non-default
   axis is set (softcap, sliding window, custom scale,
   `qk_norm_after_rope`), the block falls into a single manual path
   that handles all of them uniformly. We refused the temptation to
   build a kernel-of-everything: it produces O(2^N) configuration
   combinations and unreadable forward functions.

3. **Builder-style config with `with_*` methods.** This is the API
   contract every model file must satisfy. New axes are added as new
   builders rather than fields-with-defaults, so model files document
   what's non-default by what they call.

4. **Bespoke list, not "do everything".** The plan is **80% covered by
   `AttentionBlock`, 20% bespoke** — pragmatic, reviewable, performant.
   Architectures with genuinely exotic attention math (DeepSeek MLA,
   SSM families, ALiBi, linear attention, gated attention, asymmetric
   V-dim, flat-dim QK norm, multi-modal/Fourier RoPE) stay bespoke and
   are listed in `block.rs`. The list is the explicit complement of the
   block's coverage, not a "TODO" — these models will not migrate.

5. **TP transparency.** The block uses `TpLinear` and accepts a
   `&dyn ProcessGroup` at construction. Models that aren't yet TP-aware
   (rare bespoke MoE/VLM cases) hold an internal
   `TpContext::single_gpu()` and `LocalProcessGroup` so the public
   layer/model API stays non-TP. See `models/exaone_moe.rs` for the
   pattern.

6. **The CUDA fast-path stays in one place.** The bespoke
   implementations duplicated a ~90-line `cuda-kernels` cfg-gated
   batched-decode block. The migration removes ~10 copies of it; the
   AttentionBlock fast path is the single source of truth.

### Migration order

The plan-driven migration order was: Llama (calibration) →
Qwen2/Mistral/Olmo2 (vanilla GQA) → Qwen3/Cohere (QK norm) →
Gemma1/Starcoder2 (no-norm Gemma RmsNorm) → Phi3/Dots1/DBRX/GPT-OSS
(fused QKV + bias) → Llama4/InternLM2VE (bypass RoPE) → ChatGLM/GLM4/
GPT-J/NeoX/Phi/Nemotron/Glm (partial RoPE) → Hunyuan
(qk_norm_after_rope) → Bailing-MoE (custom proj names) → GPT-2/OPT
(absolute pos = bypass RoPE + ALL bias) → Persimmon (LayerNorm QK) →
Granite (output multiplier) → Gemma2 (softcap + sliding) → Eagle3
(asymmetric QKV input). Each cohort surfaced one new axis or validated
existing ones; per-PR functional tests ensured zero regression.

## Consequences

**Benefits:**

- ~1600 LOC of duplicated attention scaffolding removed from the model
  zoo across ~40 migrations, with similar savings remaining behind the
  bespoke list.
- One place to fix or extend attention behaviour. Adding a new axis
  (e.g. asymmetric QKV input for Eagle3) is a one-builder addition that
  every model that needs it picks up declaratively.
- Adding a new model becomes ~250 LOC: a config struct, an attention
  shim, a decoder layer, a wrapper, and a registry entry. Three places
  to touch instead of seven (see `docs/adding-a-model.md`).
- Latent bugs surface during migration. Cohort 29 (Molmo2) found that
  the bespoke attention never applied `o_proj` — a constant-factor error
  in production that AttentionBlock fixed automatically.

**Trade-offs:**

- The block is a non-trivial primitive (~700 LOC). Its module-level doc
  enumerates every axis and the bespoke list; the test suite exercises
  every combination through migrated models.
- Models that pass `with_scale`, `with_softcap`, or `with_sliding_window`
  fall off the CUDA fast-path. This matches the legacy behaviour
  (Gemma2 always took the manual path) but is a hot-path surprise for
  anyone enabling those knobs on a new model.
- Bespoke models still exist; the project is not a "single attention
  forward" codebase. The bespoke list is the contract — it documents
  what the block deliberately doesn't cover, and migrating any of those
  to the block would require either dropping their unique math or
  expanding the block beyond the design.

**Future work:**

- The plan's Phase 5 (MoE consolidation, `MoERouter` trait) reuses the
  same approach: declarative config, structural reuse, bespoke list.
  Most of the infrastructure (`TopKRouter`, `MoELayer`,
  `MoELayerWithShared`) already exists; the remaining work is migrating
  bespoke MoE implementations in 21 models. This work is gated on
  golden-logits tests for bit-exact validation across the differing
  topk algorithms.
- Phase 4c (`VisionTowerFactory`) applies the same pattern to vision
  encoders across ~35 VLMs.
- Phase 6 cleanup (registry-based dispatch in `models/mod.rs` to bring
  it from 2.6k LOC down to ~300 LOC) is independent of `AttentionBlock`
  but compounds nicely with it.
