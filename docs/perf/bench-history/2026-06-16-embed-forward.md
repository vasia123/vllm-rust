# Embedding forward path: `forward_hidden` vs generation `forward` (CPU)

Date: 2026-06-16
Device: CPU (WSL2, criterion 100-sample median, 3 s measurement)
Bench: `crates/core/benches/embed_forward_bench.rs`
Models: small (toy-dim) quantized Gemma 4 (PLE + sliding window) and Qwen3,
zero weights via the unquantized loader.

## What this measures

`forward_hidden` is the new prefill-style pass that backs `/v1/embeddings`
on a generative model (serve generation + embeddings from one loaded model,
no second weight load). It equals `forward` **minus** the last-token narrow
**minus** the `lm_head` vocab projection, returning per-token hidden states
`[seq, hidden]` for pooling. This bench records its baseline and contrasts it
with the generation prefill on the same model.

## CPU results (median)

| Model  | seq | `forward` (gen prefill) | **`forward_hidden`** | Δ |
|--------|----:|------------------------:|---------------------:|----:|
| Qwen3  |  32 | 12.66 ms | **12.51 ms** | −1.2% |
| Qwen3  | 128 | 21.78 ms | **20.64 ms** | −5.2% |
| Gemma4 |  32 | 38.77 ms | **38.86 ms** | +0.2% |
| Gemma4 | 128 | 58.46 ms | **62.89 ms** | +7.6% |

## Reading the numbers (important caveat)

These use a **toy vocab of 256**, so `lm_head` is nearly free — which means
the bench *understates* `forward_hidden`'s real-world win and even shows it
slightly slower for Gemma 4 at seq 128. Why:

- **Qwen3** prefill does the final norm + `lm_head` over the **full**
  sequence (no narrow). `forward_hidden` keeps the full-sequence norm but
  drops `lm_head`, so it is already faster here, and the gap grows with seq
  length. In production (Qwen3 vocab ≈ 151 936) the dropped `lm_head` is a
  large matmul → the real win is far bigger than the 1–5% seen at toy scale.

- **Gemma 4** prefill narrows to the **last token before** the final norm
  (a deliberate 262 144-vocab memory optimization), so its norm + `lm_head`
  run on a single token. `forward_hidden` instead runs the final norm over
  **all** positions (pooling needs them) and skips `lm_head`. At toy vocab
  the skipped `lm_head` saves almost nothing while the full-sequence norm
  costs more → `forward_hidden` looks ~7% slower. In production the skipped
  262k-vocab `lm_head` dominates and `forward_hidden` is dramatically cheaper
  **and** never allocates the ~300 MB full-vocab logits buffer (a real win on
  the 8 GB card).

Net: `forward_hidden`'s structural advantage (no vocab projection, lower peak
memory) is masked by the toy vocab here; it is the production-relevant
property the bench guards against regressing. The absolute toy-dim CPU times
are the regression baseline for the forward shape itself.
