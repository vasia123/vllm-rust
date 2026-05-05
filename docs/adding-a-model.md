# Adding a New Model

This guide shows how to add a new decoder-only architecture to vllm-rust.

The vast majority of models in the wild differ only in **declarative knobs**
on top of GQA + RoPE attention. After Phase 4 of the architecture refactor,
those knobs live on a single `AttentionConfig`, and a new model is usually
< 250 LOC: an attention shim, a decoder layer, a model wrapper, and a
registry entry.

## Prerequisites

Look in `crates/core/src/layers/` first. The reusable primitives cover most
common cases:

- `attention::AttentionBlock` — TP-aware attention with a config-driven
  shape. See `crates/core/src/layers/attention/block.rs` for the full list
  of supported axes.
- `RotaryEmbedding::new` / `RotaryEmbedding::new_partial` — full or partial
  RoPE.
- `RmsNorm` / `rms_norm()` — RMSNorm with the standard, `(1 + w)`, and
  unweighted variants.
- `SwiGluMlp` — gate/up/down GLU MLP, configurable activation (SiLU, GeLU
  PyTorch-tanh) and bias.
- `causal_mask()` — standard upper-triangle mask.
- The `tp_layers` module re-exported from `models::tp_layers`:
  `TpLinear`, `TpEmbedding`, `TpSwiGluMlp`, `TpGeGluMlp`, `TpContext`.

If your model needs a layer type that isn't there yet (e.g. a new
activation, a gated norm), add it to `layers/` first and add a unit test —
not in the model file.

## Where the new model fits

A model file usually contains four things, in this order:

1. A `<Name>Config` struct that pulls the model-specific fields out of
   `ModelConfig::extra` (vendor-specific keys go here, not into
   `ModelConfig` itself).
2. A `<Name>Attention` struct holding `inner: AttentionBlock`. The
   `AttentionConfig` is built declaratively in `new` and the forward
   methods are one-line shims over `inner`.
3. A `<Name>DecoderLayer` struct combining attention + MLP +
   layer norms.
4. A `<Name>ForCausalLM` struct (`embed_tokens`, `Vec<DecoderLayer>`,
   `norm`, `lm_head`) implementing `crate::engine::ModelForward`.

Three places need to be touched:

- `crates/core/src/models/<name>.rs` — the model file itself.
- `crates/core/src/models/mod.rs` — `pub mod <name>;` plus a match arm in
  the relevant `from_config*` dispatch fn (typically `from_config_with_quant`
  or `from_config_with_tp`).
- (optional) `crates/core/src/registry.rs` — alias the architecture name
  if HF uses several spellings.

That's it. **Three places, not seven.**

## Walkthrough

### 1. Configure attention declaratively

`AttentionBlock` covers the structural variations across 80%+ of decoder-only
models. The pattern is to start from `AttentionConfig::gqa(...)` and chain
`with_*` builders for any non-default behaviour:

```rust
use crate::layers::attention::{
    AttentionBias, AttentionBlock, AttentionConfig, ProjNames, QkNormVariant,
};

let attn_cfg = AttentionConfig::gqa(
    cfg.num_attention_heads,
    cfg.num_key_value_heads,
    cfg.head_dim,
    cfg.hidden_size,
)
.with_qk_norm(QkNormVariant::PerHead, cfg.rms_norm_eps)  // Qwen3 / Olmo2 / Cohere
.with_bias(AttentionBias::QKV_ONLY)                      // Qwen2 / Dots1
.with_qkv_fused()                                        // Phi3 / DBRX / Falcon
.with_softcap(50.0)                                      // Gemma2
.with_sliding_window(window)                             // Mistral / Gemma2 even layers
.with_proj_names(ProjNames {
    qkv: "query_key_value",
    o: "dense",
    ..Default::default()
});
```

The full set of axes is documented in `layers/attention/block.rs`. The
defaults match vanilla GQA (Llama-class), so most models only set one or
two axes.

### 2. Build the attention shim

```rust
struct MyModelAttention {
    inner: AttentionBlock,
}

impl MyModelAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let attn_cfg = /* …configured as above… */;

        let rotary_emb = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        let inner = AttentionBlock::new(&attn_cfg, vb, pg, rotary_emb)?;
        Ok(Self { inner })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.inner.forward(
            xs, attention_mask, seqlen_offset,
            cache_engine, block_table, slot_mapping, tp_ctx,
        )
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.inner.forward_decode_batch(xs, sequences, cache_engine, tp_ctx)
    }
}
```

If the model is not yet TP-aware (a single-GPU bespoke MoE, for example),
keep an internal `tp_ctx: TpContext::single_gpu()` and a
`pg = LocalProcessGroup::new()` so the public layer/model API can stay
non-TP. See `models/exaone_moe.rs` for the canonical example.

### 3. Decoder layer + model

The decoder layer wires attention to MLP through the normalization scheme
the model uses. Most models are pre-norm with RMSNorm + SwiGLU; some are
post-norm (Exaone4); some have per-block dual-norm (Gemma2's
`pre_feedforward_layernorm` + `post_feedforward_layernorm`). Use an
existing model file with the closest norm pattern as a template.

The model wrapper threads embeddings, layers, final norm and `lm_head`,
and implements `crate::engine::ModelForward` (and `forward_decode_batch`
when the model has decode batching).

### 4. Register

In `crates/core/src/models/mod.rs`:

```rust
pub mod my_model;
pub use my_model::MyModelForCausalLM;
```

Add the architecture name to the appropriate dispatch match. For a vanilla
text-only LM, that's `from_config_with_quant`. For models with TP support
expose them through `from_config_with_tp` as well (see `mistral.rs`,
`qwen3.rs` for the pattern).

If HuggingFace uses several names for the same architecture, add aliases
in `registry.rs`.

### 5. Tests

Add unit tests in the model file:

- `test_<name>_construction` — builds with zero-weight `VarBuilder`.
- `test_<name>_forward_shape` — single prefill produces logits of the right
  shape.
- `test_<name>_prefill_then_decode` — prefill + 1-token decode.
- (When applicable) `test_<name>_tp_forward_world_size_2` — TP=2 with a
  mock communicator.

Functional shape tests are sufficient for review. Use the existing
quantized-counterpart tests (e.g. `qwen3_quantized.rs`) when adding a new
quantization variant.

### 6. Verify

```bash
cargo check -p vllm-core
cargo test -p vllm-core --lib <name>
cargo test -p vllm-core --lib            # full suite
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all
```

The pre-commit hook runs fmt + clippy automatically; if either fails,
fix and re-stage rather than `--no-verify`.

## When AttentionBlock isn't enough

A handful of architectures need genuinely exotic attention math. They stay
bespoke and are documented in `layers/attention/block.rs`:

- **MLA** (DeepSeek V2/V3) — kv_b_proj absorption, low-rank latent KV.
- **SSM / hybrid SSM** (Mamba/Mamba2/Jamba/Bamba/Falcon-H1) — state space
  recurrence + conv1d.
- **Linear attention** (MiniMax-Text01 lightning, Kimi-Linear) — GDN-style
  recurrent state.
- **ALiBi** (MPT/Bloom/Jais/Baichuan) — additive position bias instead of
  RoPE.
- **Gated attention** (AfMoE, Qwen3-Next output gating, iquest_loopcoder
  loop > 0).
- **Asymmetric V-dim** (MiMoV2-Flash) — `v_head_dim != head_dim`.
- **Flat-dim QK norm** (OLMoE / FlexOlmo) — RMSNorm on the full `embed_dim`
  rather than per-head `head_dim`.
- **Multi-modal RoPE** (Qwen2-VL / Qwen3-VL) — 3D positional encoding.
- **Fourier RoPE** (InternS1-Pro) — learned frequency basis.

These models implement attention in their own file. The plan is **80%
covered by `AttentionBlock`, 20% bespoke** — pragmatic and reviewable.

The CI guardrail `crates/core/tests/no_local_attention.rs` enforces
this discipline: any new model file with a bespoke `Attention` struct
must either use `AttentionBlock` or be added to
`BESPOKE_ATTENTION_FILES` with a one-line reason. This catches drift
at PR time. See ADR-0010 for the full rationale.

## Common gotchas

- **`bias.q == bias.k == bias.v` for fused QKV.** A single fused weight
  cannot have heterogeneous bias; the constructor enforces this.
- **`o_proj` output is always `hidden_size`.** Use
  `with_qkv_input_size(N)` only to override the *input* dimension of
  Q/K/V (Eagle3 layer 0).
- **`with_scale` triggers the manual attention path** (no CUDA fast path).
  Only override scale when the model genuinely needs a non-standard one
  (Gemma2 query_pre_attn_scalar, Granite attention_multiplier).
- **`with_qk_norm_after_rope` is for Hunyuan-style ordering.** Default is
  pre-RoPE (Qwen3, Olmo2, Cohere).
- **Custom proj names must match the HF checkpoint.** GPT-2 uses
  `c_attn`/`c_proj`, Falcon uses `query_key_value`/`dense`, Exaone uses
  `out_proj`. See the `ProjNames` defaults in `block.rs`.

## Checklist

- [ ] Model file created in `models/<name>.rs`
- [ ] `AttentionConfig` built from declarative knobs; `inner: AttentionBlock`
- [ ] `pub mod` + (where appropriate) `pub use` in `models/mod.rs`
- [ ] Match arm added to `from_config_with_quant` (and TP variant if
      applicable)
- [ ] Aliases registered in `registry.rs` if HF uses multiple names
- [ ] Unit tests cover construction, prefill, decode, (optional) TP
- [ ] `cargo check`, `cargo test --lib`, `cargo clippy -- -D warnings`,
      `cargo fmt --all` all green
