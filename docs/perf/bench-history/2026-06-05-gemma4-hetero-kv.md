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
