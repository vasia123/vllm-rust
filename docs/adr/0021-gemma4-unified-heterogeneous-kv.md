# 0021 — Gemma 4 (released) EXL3 support: heterogeneous KV heads + unified arch

Status: Accepted (2026-06-05)

## Context

The released Gemma 4 family (12B `Gemma4UnifiedForConditionalGeneration`,
31B `Gemma4ForConditionalGeneration`, 26B-A4B MoE) differs from the early
E2B spec our `gemma4.rs` / `gemma4_quantized.rs` were written against. We
want to run the EXL3 quant `turboderp/gemma-4-12B-it-exl3` (quant version
0.0.39 = ExLlamaV3 v0.0.40+). Verifying against the real checkpoint
(`config.json` + `quantization_config.json` + `tensor_storage`) and
vLLM `gemma4.py` / ExLlamaV3 `architecture/gemma4.py` surfaced four gaps
that blocked loading or corrupted output:

1. **Heterogeneous KV heads.** Full-attention layers use both a larger
   `head_dim` (`global_head_dim`, 512) *and* fewer KV heads
   (`num_global_key_value_heads`, 1 for 12B / 4 for 31B) than sliding
   layers (8 / 16 @ 256). We already padded `head_dim` to a shared cache
   width but assumed a uniform KV-head count, so full-layer `k_proj`
   shapes (`1×512`) mismatched the `8×512` we built → load crash.
2. **EXL3 `codebook` parsed by name.** The quant records
   `"codebook": "mcg"` rather than an `mcg_multiplier` integer; we only
   read the latter, so the mcg codebook stayed off → garbage decode.
3. **Quantized `lm_head` under `tie_word_embeddings=true`.** The quant
   ships a separate 6-bit `lm_head.trellis` while `embed_tokens` stays
   bf16; tying to the embedding produced wrong logits.
4. **Unregistered arch / prefix.** `Gemma4UnifiedForConditionalGeneration`
   (`model_type = gemma4_unified`) was unknown to the registry, and its
   tensors live under `model.language_model.*` with a top-level `lm_head`.

## Decisions

- **Shared-cache padding for KV heads (mirrors the existing head_dim
  approach).** All layers share one paged KV cache, sized at
  `cache_num_kv_heads = max(num_key_value_heads, num_global_key_value_heads)`
  and `cache_head_dim = max(head_dim, global_head_dim)`. Full layers
  zero-pad K/V up to that stride on write (`pad_kv_heads`) and slice back
  on read (`narrow` on dim 1) before GQA expansion. Padding is a no-op
  (early-return clone) for homogeneous models and sliding layers, so
  non-Gemma-4 paths and the common case pay nothing. Rejected the
  per-layer-cache alternative (per-layer KV geometry in `CacheConfig`) as
  far more invasive across the engine for a single model family.
- **`ModelConfig::kv_cache_head_dim()` / `kv_cache_num_kv_heads()`** expose
  the max-adjusted geometry. The server cache builders (`main.rs`,
  `admin/restart.rs`, KV-budget + CPU-offload estimates) use these instead
  of the raw fields. Generic and inert for models without
  `global_head_dim` / `num_global_key_value_heads` in `extra`.
- **EXL3 codebook**: parse `codebook: "mcg"|"mul1"` as the global default
  in `Exl3Config::from_detected`, and additionally probe per-linear
  `.mcg` / `.mul1` flag tensors in `Exl3WeightLoader::load_linear`
  (presence → flag on) — robust to per-layer codebook mixing.
- **lm_head**: for EXL3 loaders, prefer a separately-quantized `lm_head`
  when its tensors exist even if `tie_word_embeddings=true`; fall back to
  the tied embedding when absent. Gated on `method() == Exl3` so other
  loaders (which may zero-fill) keep the tied path.
- **Unified arch is text-only.** The published EXL3 quant contains no
  vision tensors, so `Gemma4UnifiedForConditionalGeneration` /
  `Gemma4UnifiedTextModel` route to a text-only factory that builds the
  Gemma 4 backbone at the `model.language_model` root, reusing
  `RemappingWeightLoader` (`model.` → `model.language_model.`). Full
  unified multimodal (vision) is deliberately out of scope here — the
  text-only quant cannot exercise it.

## Consequences

- Correctly loads/decodes the released 12B/31B EXL3 quants.
- KV cache for Gemma 4 is over-allocated (full layers reserve the sliding
  stride): on the 8 GB GPU this roughly doubles per-token KV cost vs an
  ideal per-layer cache, reducing max context — acceptable for the
  shared-cache design; documented for future per-layer-cache work.
- Bench: `crates/core/benches/gemma4_hetero_kv_bench.rs` records the
  pad/narrow cost (full-layer vs homogeneous no-op).
