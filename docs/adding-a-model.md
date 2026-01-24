# Adding a New Model

This guide shows how to add a new model architecture to vllm-rust.

## Prerequisites

Verify the model uses components already in `layers/`:
- **RoPE** — `layers::RotaryEmbedding`
- **Paged Attention with GQA** — `layers::paged_attention()`
- **SwiGLU MLP** — `layers::SwiGluMlp`
- **Causal Mask** — `layers::causal_mask()`
- **Per-head Norm** — `layers::apply_per_head_norm()`

If the model needs a new layer type (e.g., GeGLU, sliding window attention), add it to `layers/` first.

## Steps

### 1. Create the model file

Create `crates/core/src/models/<name>.rs`. Use an existing model as a template:
- `qwen3.rs` — has per-head Q/K RMSNorm (Qwen3-specific)
- `llama.rs` — standard attention without per-head norm

A model file contains:
- `<Name>Attention` struct — projections, RoPE, optional norms
- `<Name>DecoderLayer` struct — attention + MLP + layernorms
- `<Name>ForCausalLM` struct — embeddings + layers + lm_head
- `impl ModelForward for <Name>ForCausalLM`

### 2. Register in the module

Edit `crates/core/src/models/mod.rs`:

```rust
pub mod <name>;            // add module declaration
pub use <name>::<Name>ForCausalLM;  // add re-export
```

### 3. Add to factory

In the same file, add a match arm in `from_config()`:

```rust
match arch.as_str() {
    "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new(cfg, vb)?)),
    "LlamaForCausalLM" => Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?)),
    "<Name>ForCausalLM" => Ok(Box::new(<Name>ForCausalLM::new(cfg, vb)?)),  // add this
    other => Err(ModelError::UnsupportedArchitecture(other.into())),
}
```

The architecture string must match the value in the model's `config.json` on HuggingFace (`"architectures": ["<Name>ForCausalLM"]`).

### 4. Add config fields (if needed)

If the model requires config fields not yet in `ModelConfig`, add them as optional:

```rust
// crates/core/src/config.rs
#[serde(default)]
pub sliding_window: Option<usize>,
```

Use `#[serde(default)]` so existing models continue to parse. The `#[serde(flatten)] pub extra` field catches any remaining unknown fields.

### 5. Verify

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace
```

If you have the model downloaded, run the ignored integration tests:
```bash
cargo test --workspace -- --ignored
```

## Example: Minimal Attention Difference

The only difference between Llama and Qwen3 attention is per-head norm:

```rust
// Qwen3: applies RMSNorm to each head independently before RoPE
let q = apply_per_head_norm(&q, &self.q_norm)?;
let k = apply_per_head_norm(&k, &self.k_norm)?;
let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

// Llama: applies RoPE directly
let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;
```

Everything else (projections, paged attention, GQA, MLP, residual connections) is identical.

## Adding a New Layer Type

If your model needs a layer not in `layers/`:

1. Create `crates/core/src/layers/<layer>.rs`
2. Add `pub mod <layer>;` to `layers/mod.rs`
3. Add `pub use <layer>::<Type>;` to `layers/mod.rs`
4. Use it in your model file

Keep layers generic — accept dimensions as parameters, not `ModelConfig`.

## Checklist

- [ ] Model file created in `models/`
- [ ] `pub mod` + `pub use` added to `models/mod.rs`
- [ ] Match arm added to `from_config()`
- [ ] Any new config fields are `#[serde(default)]` optional
- [ ] `cargo check --workspace` passes
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace` — zero warnings
