# Gemma 4 heterogeneous-KV cache padding — baseline

Bench: `crates/core/benches/gemma4_hetero_kv_bench.rs` (CPU; WSL2 blocks GPU
profilers). Measures the cache write-side pad (`pad_last_dim` head_dim +
`pad_kv_heads` KV-head, both `Tensor::cat`) and read-side `narrow` added to
the Gemma 4 attention path for the released 12B/31B head geometry (full
layers: 1 KV head @ head_dim 512 → shared cache stride 8 KV heads @ 512).

Criterion, warm-up 1s / measure 3s, 100 samples.

| case                           | dtype | median |
|--------------------------------|-------|--------|
| write_pad_full_layer           | F32   | 411 ns |
| write_pad_full_layer           | BF16  | 388 ns |
| read_narrow_full_layer         | F32   |  69 ns |
| write_pad_homogeneous_noop     | F32   |  22 ns |

Key result: the **homogeneous / non-Gemma-4 / sliding-layer path is a
no-op** — `pad_*` early-returns a clone (~22 ns, just the `Arc` bump), so
the change costs nothing on the common path. The full-attention pad
(~0.4 µs/layer, 8 of 48 layers on 12B) is negligible next to the per-layer
EXL3 GEMMs. This is the *before* baseline; the padding is required for
correctness (shared cache stride), so there is no "after" to beat — the
snapshot guards against future regressions in the no-op fast path.

## Addendum 2026-06-07 — prefill length bucketing

Prefill activations are now padded to `PREFILL_LEN_BUCKET = 32` (capped at the
RoPE horizon), so candle's per-shape CUDA buffer cache holds at most
`max_model_len / 32` distinct prefill shape sets instead of one per prompt
length. K/V padding rows are dropped before the cache write; the KV read-back
is zero-padded to the bucket and masked (`bucketed_prefill_mask`, padding rows
keep column 0 open as a softmax-NaN guard).

Live measurement (gemma-4-12B-it-exl3 @ 2.00bpw, 8 GB, --enforce-eager):

| scenario                                   | before        | after  |
|--------------------------------------------|---------------|--------|
| VRAM after 8 requests of distinct lengths  | +~30 MB each  | flat (6201 → 6201 MiB) |
| coherence ("Paris", thinking "42")         | ok            | ok (unchanged) |

Compute overhead: ≤31 extra prefill tokens per request (bucket rounding);
decode path untouched. Mask construction is O(q·kv) per layer — same class as
the causal/sliding masks it replaces.
