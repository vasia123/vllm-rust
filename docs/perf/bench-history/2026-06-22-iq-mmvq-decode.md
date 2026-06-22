# Native I-quant decode: dequant-then-matmul vs q8_1 MMVQ (CUDA)

Date: 2026-06-22
Device: RTX 4060 Laptop (sm_89), criterion 20-sample median, 3 s measurement
Bench: `crates/core/benches/iq_dequant_bench.rs`
Quant: IQ3_XXS (QK_K=256), weight `[out, in]`, M=1 (decode)

## Headline

candle has no I-quant path, so the Unsloth "UD" GGUFs run through our own
`IqLinear`. The original decode path dequantized the WHOLE I-quant weight to
a dense f32 tensor on every token, then ran a matmul — `O(out·in)` of pure
overhead per step. The new path quantizes the activation to q8_1 once and
dots the I-quant weight against it with integer `__dp4a` (a port of ggml's
`vec_dot_iq*_q8_1`, the path llama.cpp itself runs for I-quant decode), so
decode is **~75-130× faster** at the kernel level.

This is the before/after pair the CLAUDE.md perf rule requires
(baseline = `decode_dequant_matmul`, candidate = `decode_fused_mmvq`),
captured in a single self-contained bench.

## CUDA results

| Shape | mode | dequant+matmul (BEFORE) | **q8_1 MMVQ (AFTER)** | speedup |
|---|---|---|---|---|
| 4096×4096   | 1 (decode)   | 5.256 ms  | **70.5 µs**  | **74.5×** |
| 14336×4096  | 1 (decode)   | 21.541 ms | **165.4 µs** | **130×**  |
| 4096×4096   | 32 (prefill) | 5.283 ms  | (dequant+GEMM path) | — |
| 14336×4096  | 32 (prefill) | 21.297 ms | (dequant+GEMM path) | — |

## Why decode dominates the win

Decode generates one token per forward (M=1): the matmul is a matrix-vector
product (cheap), so dequantizing all `out·in` weights to f32 each step is the
hot path of generation speed. MMVQ never materializes the dense weight — one
pass over the I-quant bytes, integer dots, no float multiplies in the inner
loop — so it runs at memory bandwidth. Prefill (M>16) keeps the dequant+GEMM
path: once the matmul has many rows, the one-shot dequant amortizes.

## Lineage

This replaced an earlier float GEMV (commit 0a0efb0) that dequantized each
256-element block to a per-lane local f32 array before dotting — a
register/local-memory hog. MMVQ's integer path is ~9-16× faster than that
float GEMV at the kernel level, and is the same algorithm llama.cpp uses.

Correctness is doubly pinned (see ADR 0025): the CUDA `mmvq_*` kernels match a
CPU port of the identical integer algorithm bit-for-bit (modulo f32 reduction
order), and that algorithm approximates a full-precision f32 matmul within
int8-activation tolerance — verified at the model's real widths (n_in up to
4096, nblk32 = 128) by tiling the fixture blocks.

## End-to-end (gemma-4-12b-it-UD-IQ3_XXS, 8 GB, --max-model-len 1024)

Differential decode rate (two chat completions, 16 vs 180 generated tokens,
same prompt so prefill+load cancels): **21.5 tok/s**.

| path | decode tok/s |
|---|---|
| naive dequant-then-matmul (original) | 0.41 |
| float GEMV (commit 0a0efb0) | 3.3 |
| **q8_1 MMVQ (this change)** | **21.5** |

That is ~6.5× over the float GEMV and ~52× over the naive path, at or above
llama.cpp parity for this model on this GPU. Output is coherent via the chat
endpoint ("capital of France" → "Paris"; a correct `square_number` Python
function). NOTE: raw `/v1/completions` (no chat template) degenerates — that is
the instruct model being prompted out-of-distribution, not the kernel; always
exercise this checkpoint through the chat template.
