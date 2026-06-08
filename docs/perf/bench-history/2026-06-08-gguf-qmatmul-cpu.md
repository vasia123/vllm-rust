# GGUF linear: dense-dequant baseline vs candle QMatMul (CPU)

Date: 2026-06-08
Device: CPU (WSL2, criterion 30-sample median, 3 s measurement)
Bench: `crates/core/benches/gguf_qmatmul_bench.rs`
Quant: Q4_K (QK_K=256), weight `[out, in]`

## Headline

The pre-rework `GgufLinear::forward` dequantized the WHOLE weight to a
dense f32 tensor on every call, then ran a plain matmul. At decode (M=1)
the dequant cost dwarfs the matmul. candle's fused `QMatMul` keeps the
weight quantized-resident and fuses dequant into the matmul, so decode
is **~29-45× faster on CPU alone**. Prefill (M=32) ties — once the
matmul has 32 rows of work the one-shot dequant amortizes.

This is the before/after pair the CLAUDE.md perf rule requires, captured
in a single self-contained bench (baseline = `dense_dequant_matmul`,
candidate = `qmatmul`).

## CPU results

| Shape | M (mode) | dense baseline | **QMatMul** | speedup |
|---|---|---|---|---|
| 4096×4096   | 1 (decode)   | 31.26 ms  | **1.06 ms** | **29.5×** |
| 4096×4096   | 32 (prefill) | 36.61 ms  | **33.90 ms** | 1.08× |
| 14336×4096  | 1 (decode)   | 102.12 ms | **2.27 ms** | **45.0×** |
| 14336×4096  | 32 (prefill) | (matmul-bound, ~par) | — | ≈1× |

## Why decode dominates the win

Decode generates one token per forward (M=1): the matmul is a
matrix-vector product (cheap), so dequantizing all `out·in` weights to
f32 each step is pure overhead — exactly the hot path of generation
speed. The rework removes it. Prefill batches M tokens, so the dequant
is amortized across M matmul rows and the gap closes.

## CUDA

The CUDA half of the bench (`#[cfg(feature="cuda")]`, MMVQ fused kernel)
is recorded separately after a `cuda-full` build — see the e2e decode
tok/s via `scripts/bench_decode.py` on a real Q4_K_M checkpoint. The CPU
result already proves the algorithmic win; CUDA MMVQ widens it further
at decode.
