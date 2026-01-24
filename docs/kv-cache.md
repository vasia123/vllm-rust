# KV Cache System

## Overview

The KV cache stores key/value tensors from previous forward passes so they don't need to be recomputed. vllm-rust uses **paged attention** — the cache is divided into fixed-size blocks, allocated on demand, and freed when requests complete.

## Components

```
KVCacheManager
├── BlockPool          — free list of physical block IDs
├── CacheEngine[0]     — layer 0: K tensor + V tensor
├── CacheEngine[1]     — layer 1: K tensor + V tensor
└── ...

BlockTable (per request)
├── blocks: Vec<BlockId>   — allocated block IDs in sequence order
├── block_size: usize      — tokens per block
└── fill: usize            — tokens written to last block
```

## Memory Layout

Each `CacheEngine` holds two tensors:
- K cache: `[num_blocks, block_size, num_kv_heads, head_dim]`
- V cache: `[num_blocks, block_size, num_kv_heads, head_dim]`

Physical slots are addressed as `block_id * block_size + offset_within_block`.

## Lifecycle

### Allocation

```rust
// Request needs N tokens of cache space
kv_cache_mgr.allocate_for_request(&mut block_table, num_tokens)?;
```

This allocates enough blocks from the pool to hold `num_tokens` new tokens (accounting for partially-filled last block).

### Write (during forward pass)

```rust
let slot_mapping = block_table.slot_mapping(seqlen_offset, num_new_tokens);
// slot_mapping is passed to paged_attention() which writes K/V to these slots
```

### Advance

```rust
block_table.advance(num_new_tokens);
// Updates fill counter so next allocation knows where to continue
```

### Read (during decode)

The `paged_attention()` function reads all previously written K/V from the block table's blocks to compute attention over the full sequence.

### Free

```rust
kv_cache_mgr.free_request(&mut block_table)?;
// Returns all blocks to the pool
```

## Configuration

```rust
CacheConfig {
    block_size: 16,          // tokens per block
    num_blocks: 512,         // total blocks in pool
    num_layers: 28,          // one CacheEngine per layer
    num_kv_heads: 8,         // GQA head count
    head_dim: 128,           // dimension per head
    dtype: DType::BF16,      // storage precision
    device: Device::Cuda(0), // GPU device
}
```

**Memory usage**: `num_blocks * block_size * num_kv_heads * head_dim * 2 (K+V) * dtype_size * num_layers`

Example (Qwen3-0.6B, 512 blocks): `512 * 16 * 8 * 128 * 2 * 2 * 28 = ~7.5 GB`

## Preemption

When the block pool is exhausted, the scheduler preempts the newest running request:
1. Free its blocks back to the pool
2. Clear its generated tokens
3. Re-queue it for later re-prefill

This prevents OOM crashes and allows the system to make progress under memory pressure.

## Trim (Speculative Decoding)

After speculative verification, accepted tokens may be fewer than allocated:

```rust
let freed_blocks = block_table.trim_to(actual_sequence_length);
kv_cache_mgr.free_blocks(&freed_blocks)?;
```

This returns unused trailing blocks to the pool.
