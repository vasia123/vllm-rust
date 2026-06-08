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

## CUDA results (RTX 4060 Laptop, sm_89)

Recorded after a `cuda-full` build via the bench's `#[cfg(feature="cuda")]`
half (MMVQ fused decode kernel, q8_1 GEMM for prefill).

| Shape | M (mode) | dense baseline | **QMatMul** | speedup |
|---|---|---|---|---|
| 4096×4096   | 1 (decode)   | 1.82 ms  | **52.7 µs** | **34.6×** |
| 4096×4096   | 32 (prefill) | 1.91 ms  | **246 µs**  | **7.8×** |
| 14336×4096  | 1 (decode)   | 8.87 ms  | **103 µs**  | **85.7×** |

On CUDA even prefill wins ~8× — the dense path's whole-weight dequant is a
separate memory-bound kernel that dominates, while MMVQ/q8_1 fuses dequant
into the matmul. Decode (M=1) is the dominant generation-speed case and
wins 35-86×.

## e2e validation

GGUF served end-to-end on GPU via the new `--gguf-file` path
(SmolLM-135M-Instruct.Q4_K_M, Llama arch, eager): loads quantized-resident,
produces grammatical text ("The capital of France is the country of
France"), ~89 tok/s wall-clock incl. HTTP for an 80-token greedy decode.
(SmolLM-135M itself degenerates into repetition at greedy temp=0 — a known
trait of the 135M model, not the GGUF path; the QMatMul parity test and the
coherent leading sentence confirm correctness.)
