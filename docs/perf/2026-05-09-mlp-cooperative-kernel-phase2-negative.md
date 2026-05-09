# Phase 2 — cooperative 4-warp kernel for AWQ-Marlin MLP — NEGATIVE

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), WSL2

## Hypothesis (refuted)

The legacy TC kernel `marlin_tile_mma_int4_bf16_tc_m16k16g16` runs
1 warp per block (block_dim=32), grid=(N/16, M_tiles). Each warp's
B-tile read pattern `b_ptr[... + lane*4 + warp_id]` has 16-byte
stride between consecutive lanes — 4 cache sectors per warp's LDG
vs. 1 ideal sector → 25 % coalescing efficiency.

Hypothesis: pack 4 warps into a single block (block_dim=128),
quartered grid (N/64). With all 4 warp_ids in the same block, the
cooperative LDG can be coalesced (each block-thread reads its own
u32, all 128 reads contiguous).

Predicted gain: ~4× B bandwidth → e2e MLP ~3-4× faster, c=8 lift
to ~150-180 tps.

## Result: NEGATIVE — slower in microbench, not faster

`cargo bench --features cuda-default,marlin -p vllm-core --bench
marlin_tile_mma_path_bench` after dispatching to the cooperative
kernel (`VLLM_AWQ_MARLIN_4W=1`):

### v1 (cooperative LDG + smem read)

| shape M=8 | legacy | 4w v1 (smem) | Δ |
|---|---|---|---|
| q_o N=2560 | 38.8 µs | 51.1 µs | **+32 % slower** |
| gate_up N=11008 | 103.8 µs | 127.2 µs | **+23 % slower** |
| down N=2560 | 156.0 µs | 212.8 µs | **+36 % slower** |

The smem read pattern `smem_b[lane*4 + warp_id]` for fixed warp_id
hits the same bank every 8 lanes (4-way bank conflict). The
coalesced LDG win (4 cache lines/block instead of 16) is more than
offset by 4× LDS latency under conflict.

### v2 (multi-warp blocks, no smem; just direct global LDG)

| shape M=8 | legacy | 4w v2 (no smem) | Δ |
|---|---|---|---|
| q_o N=2560 | 38.8 µs | 51.0 µs | **+31 % slower** |
| gate_up N=11008 | 103.8 µs | 115.9 µs | **+12 % slower** |
| down N=4 (sample) | 142 µs | 200 µs | **+41 % slower** |

Without smem, v2 isolates the "block packing" effect from coalescing.
Still slower → multi-warp blocks reduce SM-level parallelism more
than they save.

## Why it failed

The original analysis assumed **HBM bandwidth on B reads** was the
bottleneck. The microbench evidence contradicts this:

- Legacy isolated bench achieves 153 GB/s effective (per
  `2026-05-09-awq-marlin-mlp-bandwidth-finding.md`) — close to the
  HBM ceiling.
- Coalescing efficiency, even at 25 %, does not explain a 6×
  production gap. (At 25 %, 153 GB/s should fall to ~38 GB/s; we
  see 23 GB/s in production. So coalescing is a 1.6× factor at most.)

The real production-vs-microbench 6× gap is **L2 cache contention
state** between layers, not HBM bandwidth or coalescing. Neither
v1 nor v2 of Phase 2 changes the cache state — both just reorganise
the same number of bytes flowing through the same cache hierarchy.

In v1, smem bank conflicts add real latency. In v2, fewer larger
blocks (152 vs 608) give each block ~4× more work but reduce SM
occupancy proportionally — net worse on this M=8 shape where work
per block is already small enough that scheduling per-block has
overhead.

## Decision: revert, document, move on

Both v1 and v2 are reverted. Findings preserved here. The Phase 2
hypothesis was ruled out by microbench BEFORE integration —
working as intended (memory rule `feedback_perf_assumption_test.md`:
isolated bench prove first).

## What ACTUALLY would help

The remaining bottleneck is L2 cache contention between MLP and
attention blocks, NOT a kernel-level coalescing/occupancy issue.
Real fixes are architectural:

1. **Persistent fused MLP+attention block** running through whole
   multi-step in one launch — keeps weights pinned in L2 across
   layers. Requires major model refactor; bound by L2 size = 24 MB
   on this GPU (one MLP layer alone needs 37.5 MB, see prior
   analysis), so this path may be infeasible without quantising
   to ≤2-bit.
2. **Cache-aware layer reordering**: process all M tokens through
   layer 0 first, then layer 1 — invalidates current KV/state model.
3. **lm_head optimisation** (separate slot, 27 ms) — smaller scope,
   single matmul, maybe easier wins via cuBLAS routing.

## Verification

- `cargo bench --features cuda-default,marlin -p vllm-core
  --bench marlin_tile_mma_path_bench` shows v1 and v2 microbench
  numbers as documented.
- Legacy kernel restored on disk; PTX rebuilt; existing GPU tests
  still pass (40 marlin tests).
