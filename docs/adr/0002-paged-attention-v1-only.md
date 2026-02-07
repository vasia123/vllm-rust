# ADR-0002: Paged Attention V1 and V2

## Status

Updated. V2 implemented in Phase 1.3 with auto-selection.

## Context

vLLM implements multiple paged attention variants:

- **V1**: Single thread block per (head, sequence). Simple, good for short-to-medium sequences.
- **V2**: Split-reduce across multiple thread blocks per sequence. Better for long sequences (32k+).
- **V3**: Further optimized with FP8 KV cache support.

## Decision (Original)

We implemented **V1 only** for the initial release, with configurable `head_dim` and `block_size` parameters (as of Phase 0.2).

## Decision (Updated)

V2 has been implemented with a **PARTITION_SIZE of 512 tokens** and **auto-selection** based on `max_seq_len`:

- `max_seq_len <= 512`: V1 (lower overhead, single-pass)
- `max_seq_len > 512`: V2 (split-K with partitioned reduce)

### V2 Architecture

Two-stage kernel launch:

1. **paged_attention_v2_bf16**: Grid `(num_heads, num_seqs, max_num_partitions)`. Each thread block processes one partition of 512 tokens. Produces `tmp_out` (partial weighted V in f32), `max_logits`, and `exp_sums` per partition.

2. **paged_attention_v2_reduce_bf16**: Grid `(num_heads, num_seqs)`. Merges partition results using log-sum-exp for numerical stability. Converts final output to bf16.

Fast path: when `num_partitions == 1`, the reduce kernel just copies f32 → bf16 without reduction.

### Public API

- `paged_attention_auto()` — recommended entry point, auto-selects V1 vs V2
- `paged_attention_auto_alibi()` — same, with ALiBi support
- `paged_attention_cuda()` — V1 explicitly
- `paged_attention_v2_cuda()` — V2 explicitly

## Rationale

### Why 512 for PARTITION_SIZE

- Matches reference vLLM's default
- Each partition uses `512 * sizeof(float) = 2KB` of shared memory for logits — well within limits
- 512 tokens covers most decode batches in a single partition (fast path)

### Why auto-selection at 512 threshold

- Below 512, V1 avoids the overhead of intermediate buffer allocation and reduce kernel launch
- Above 512, V2 avoids shared memory pressure from V1's `logits[max_seq_len]` allocation

### Intermediate buffers (f32)

V2 uses f32 for `tmp_out` to avoid precision loss during the log-sum-exp merge. This costs `num_seqs * num_heads * max_num_partitions * head_dim * 4` bytes of GPU memory (temporary, freed after the operation).

## Consequences

- Decode performance scales to 128k+ token sequences
- 2-5x improvement for sequences > 32k tokens vs V1
- Slightly higher latency for short sequences if V2 is used directly (hence auto-selection)
- ALiBi models (Bloom, MPT) benefit equally from V2
