# 0025 — Native I-quant (IQ) support for GGUF

Date: 2026-06-22
Status: accepted

## Context

We could not load `gemma-4-12b-it-UD-IQ3_XXS.gguf` (an Unsloth "UD" — Unsloth
Dynamic — checkpoint). The load failed at the header:

```
Error: GGUF load failed: Failed to parse GGUF header: unknown dtype for tensor 22
```

The file mixes five **I-quant** types across its 372 quantized tensors:

| GGML type | count | candle 0.10 support |
|-----------|------:|---------------------|
| IQ3_XXS (18) | 187 | ✗ |
| IQ2_S (22)   |  76 | ✗ |
| IQ3_S (21)   |  50 | ✗ |
| IQ2_XS (17)  |  10 | ✗ |
| IQ4_XS (23)  |   5 | ✗ |
| Q3_K (11)    |   1 | ✓ (token_embd) |
| F32 (0)      | 338 | ✓ (norms) |

candle's `GgmlDType` covers only the legacy quants and the K-quants — it has
**no** I-quant variants, so `gguf_file::Content::read` aborts the moment it
reads the first `IQ*` tensor's type id. The whole UD family is therefore
unloadable, and this is not candle-specific: mistral.rs (also candle-backed)
hits the identical wall (their issue #2100).

I-quants are not a niche: they are the quality-per-byte sweet spot at ≤4 bpw
and are what makes a 12 B model fit an 8 GB card (this file is 4.6 GB).

## Decision

Add **native I-quant support in our own crate**, leaving candle vanilla.

### Considered and rejected

1. **Transcode IQ → nearest K-quant at load** (dequant then re-quantize to
   Q2_K/Q3_K/Q4_K, cache a candle-loadable file). Fast to build and reuses
   candle's fast K-quant CUDA path, but llama.cpp's own docs are explicit that
   requantizing an already-quantized model **"can severely reduce quality"**,
   and you cannot avoid it on 8 GB: preserving quality needs a target with
   *more* bits than the source, which no longer fits. Rejected on quality.

2. **Fork candle to add the IQ types.** The project retired its candle fork
   (see `Cargo.toml` notes) and avoids committing a `[patch.crates-io]`.
   Adding IQ there would re-introduce exactly that. Rejected to keep candle
   vanilla; the work is the same either way (the dequant code) — only its
   home differs.

3. **Per-forward host-side dequant** (CPU dequant → upload → matmul each
   call). Preserves full IQ precision but ~10 s/token and needs the weights
   in host RAM (4.6 GB on a 3.9 GB-free host). Infeasible.

### What we built (`crates/core/src/quantization/gguf/`)

- `iq/tables.rs` + `kernels/iq_tables.cuh` — the GGML codebook tables
  (`iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `ksigns_iq2xs`,
  `kmask_iq2xs`, `kvalues_iq4nl`) extracted **byte-exact** from
  `reference/llama.cpp/ggml/src/ggml-common.h` by `scripts/extract_iq_tables.py`
  (deterministic; never hand-edited).
- `iq/mod.rs` — `IqType` + the scalar CPU dequant for all five types, a
  one-to-one port of ggml's `dequantize_row_iq*`. Plus `dequantize_iq`, a
  `CustomOp1` whose `cuda_fwd` launches the kernel and whose `cpu_fwd` runs
  the scalar port.
- `kernels/iq_dequant.cu` — two paths sharing the same device decoders: (1)
  `dequantize_iq*` (one thread per 256-element super-block) materializes a
  dense f32 weight for the prefill GEMM; (2) `quantize_q8_1_rows` + `mmvq_iq*`
  implement q8_1 MMVQ for decode (integer `__dp4a` dots, no dense weight). See
  "Consequences".
- `header.rs` — a minimal GGUF header parser that classifies each tensor's
  dtype as candle-native **or** I-quant. Replaces `Content::read` (which
  cannot get past the IQ tensors) while reusing candle's public
  `gguf_file::Value` for metadata, so every existing metadata reader is
  unchanged.
- `GgufTensor` / `IqLinear` — the loader keeps candle-native weights as
  `QTensor` (fast `QMatMul` path, untouched) and I-quant weights as raw block
  bytes resident on the GPU; `IqLinear::forward` runs q8_1 MMVQ for decode and
  dequant + GEMM for prefill (see "Consequences").

### Correctness pinning

Every layer is pinned against an independent reference:

- CPU dequant ↔ **gguf-py golden vectors** generated from the *real* model
  file (`scripts/gen_iq_fixtures.py`, gguf 0.19.0 — a separate numpy port of
  the same ggml algorithm). ≤ 1e-5.
- GPU kernel ↔ CPU dequant (same fixtures). ≤ 1e-4 (`--use_fast_math`).
- `IqLinear::forward` ↔ dense `x @ Wᵀ` over the same dequantized weight.
- `header.rs` ↔ the real file's tensor table (counts, dtypes, shapes).

## Consequences

- The UD Gemma-4-12B IQ3_XXS checkpoint (and any IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/
  IQ4_XS GGUF) loads and runs on 8 GB at full source-file fidelity — no
  re-quantization, weights stay I-quant-resident (4.6 GB).
- `IqLinear` is two-path, mirroring candle's `QMatMul` (vec-kernel for small M,
  GEMM otherwise):
  - **Decode (M ≤ `IQ_GEMV_MAX_M` = 16): q8_1 MMVQ** (`quantize_q8_1_rows` +
    `mmvq_*` in `kernels/iq_dequant.cu`). The activation row is quantized to
    q8_1 (int8 quants + a per-32 f32 scale) once, then each output row is the
    sum of integer `__dp4a` dots of the I-quant codebook values (sign-flipped
    on the fly) against the q8_1 quants, scaled by the per-block I-quant scale
    and the weight/activation deltas. **No dense weight is materialized and the
    inner loop has no float multiplies** — this is a one-to-one port of ggml's
    `vec_dot_iq*_q8_1`, the path llama.cpp itself runs for I-quant decode. One
    warp per output row, 32-element-sub-block striding so every lane stays busy
    on narrow rows. (This replaced an earlier float GEMV that dequantized each
    block to a 256-float local array per lane — a register/local-memory hog.)
  - **Prefill (M > 16) / CPU: dequant-then-matmul** — dequantize the whole
    weight to dense f32 once, then a single cuBLAS/candle GEMM amortizes it
    over all rows. One-shot, so the materialization cost is paid once per
    prompt, not per token.
  Correctness is doubly pinned: the CUDA `mmvq_*` kernels match a CPU port of
  the identical integer algorithm (`q8_1_mmvq_cpu`) bit-for-bit modulo f32
  reduction order (`*_mmvq_gpu_matches_cpu`), and that algorithm approximates a
  full-precision f32 matmul within int8-activation tolerance
  (`iq_matmul_matches_dequant_then_matmul`). `benches/iq_dequant_bench.rs`
  covers the before/after.
- candle stays at vanilla 0.10.2; no `[patch.crates-io]`.
- Regenerating the tables requires `reference/llama.cpp` present; the script
  is the single source of truth for both the Rust and CUDA tables.
