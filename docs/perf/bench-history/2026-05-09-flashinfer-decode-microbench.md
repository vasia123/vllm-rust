# FlashInfer DecodeWrapper vs paged_attention V2 — microbench

Date: 2026-05-09
Direction A of research roadmap (`/home/vasis/.claude/plans/curried-coalescing-tome.md`).
Bench: `cargo bench -p vllm-core --features cuda-default --bench flashinfer_decode_bench -- --quick`
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), 8 GiB, WSL2

## Shape (Qwen3-4B-AWQ realistic decode)

- batch_size  = 8 (c=8 from head-to-head with Python vLLM)
- num_q_heads = 32
- num_kv_heads = 8 (GQA 4:1)
- head_dim    = 128
- block_size  = 16
- KV layout   = NHD

## Results (median μs/call)

| seq_len | paged_attn V2 | FI full (build+run) | FI replay (run only) | V2 / FI replay | V2 / FI full |
|---------|---------------|---------------------|----------------------|----------------|--------------|
| 64      | 65            | 69                  | 33                   | **1.97×**      | 0.94×        |
| 256     | 120           | 80                  | 50                   | **2.40×**      | 1.50×        |
| 384     | 145           | 95                  | 61                   | **2.38×**      | 1.53×        |
| 512     | 191           | 105                 | 71                   | **2.69×**      | 1.82×        |
| 1024    | 335           | 152                 | 117                  | **2.86×**      | 2.20×        |

## Interpretation

**Production-relevant metric is FI replay** — `DecodePlan` is built **once
per forward batch** and replayed across every attention layer (36 for
Qwen3-4B). Per-layer attention cost is the replay column, plus a
once-per-forward `build_plan` (~36 μs amortised over 36 layers ≈ 1
μs/layer).

**FlashInfer replay beats V2 by ≥ 1.97× on every measured seq_len, and
2.4× at the realistic seq_len ≈ 384** (256 prompt + 128 max_tokens
midpoint). Decision gate from the roadmap was ≥ 1.3× — cleared with
huge margin.

Even one-shot (FI full = build_plan + run_with_plan, no amortisation):
FlashInfer wins at every seq_len except the smallest (64), where the
plan overhead barely matches V2.

## Estimated e2e impact

- Current paged_attn V2 slot: ~5.6 ms / 39 ms forward = 14 % of step.
- FI replay would cut this by ~58 % at seq_len ≈ 384 → save ~8 % step.
- Expected lift: 165 tps c=8 → ~178 tps c=8.

This is at the upper end of the plan's 5-15 % expected lift. Direction
A is **clearly worth implementing**. Next step: design the integration
plan — wire `DecodeWrapper::run_with_plan` into
`block.rs::cuda_decode_batch` (env-gated initially), with `DecodePlan`
cached per-forward in `DecodeBatchShared`.

## Why FlashInfer is so much faster

Hypothesis (not verified — would need ncu, blocked in WSL2):

1. **Better split-K work distribution.** FI's plan does work-estimation
   on the host (the host sync is the cost we pay once per forward) and
   selects an optimal partition-per-sequence based on `kv_indptr`. V2
   uses a single global partition_size for the whole batch.
2. **Tighter memory reduce kernel.** FI's reduce stage is a single
   warp-cooperative pass; V2's reduce launches a separate kernel.
3. **No padding waste.** V2's `block_tables` is `[batch, max_blocks_per_seq]`
   dense — sequences shorter than max waste GMEM bandwidth on padding.
   FI uses ragged `kv_indptr` / `kv_indices` — only valid blocks are
   touched.

## Bench file

`crates/core/benches/flashinfer_decode_bench.rs` — one-time setup,
warmup, criterion-driven measurement with explicit per-iteration sync
to prevent overlap-driven false numbers.
