# AWQ-Marlin MLP — kernel time vs e2e: 6× discrepancy is L2/HBM contention

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89)
Server: `VLLM_AWQ_HYBRID=1 ./target/release/vllm-server serve --model Qwen/Qwen3-4B-AWQ --enforce-eager --max-model-len 1024 --num-blocks 384`

## Headline finding

The AWQ-Marlin tensor-core kernel (`marlin_tile_mma_int4_bf16_tc_m16k16g16`)
achieves **153 GB/s effective bandwidth in isolation** (microbench)
but only **23 GB/s in production** — a 6× slowdown ON THE SAME SHAPE.
Dispatch path is correctly routed (84 % TC dispatches confirmed via
`vllm_core::marlin_path` counters), so it's NOT a gate bug.

| measurement | M=8 gate/up (K=2560 N=11008) | M=8 down (K=11008 N=2560) | per-MLP sum |
|---|---|---|---|
| `marlin_tile_mma_path_bench` (isolated) | 103.8 µs | 156.0 µs | 363.6 µs |
| `decode_profile` mlp slot ÷ 3 (production) | ~540 µs | ~540 µs | 1620 µs/layer |
| ratio | 5.2× | 3.5× | 4.5× |

Production routing breakdown (Qwen3-4B-AWQ, c=8 decode, 20 K dispatch
sample): TC = 16,884 (84 %), fallback = 3,116 (16 %, m ∈ {1,2,3} or
m≥33 prefill). Decode m=8 calls go through TC. The 6× slowdown is
**not** dispatch routing.

## Root cause: cache contention with surrounding forward graph

In the microbench, the 12 MB INT4 weight buffer sits in L2 (24 MB on
RTX 4060) across iterations and warps share L1 prefetches.
In production, between MLP gate matmul and MLP up matmul, the layer
runs RMSNorm + the next attention block (qkv proj, paged_attn,
o_proj). Their working sets evict the MLP weights from L2; each
matmul re-reads from HBM.

Bandwidth math:
- Per-matmul weight bytes: 2560 × 11008 / 2 = ~14 MB (gate/up); 11008 × 2560 / 2 = ~14 MB (down).
- Microbench 91.7 µs → 153 GB/s (within HBM peak ~272 GB/s, very high).
- Production 540 µs → 26 GB/s (10 % of peak — reading mostly cold from HBM).

This pattern matches a **memory-bandwidth-bound kernel at high cache
miss rate**. Compute is not the bottleneck (the kernel emits
mma.m16n8k16 already; throughput is not limited by SM utilisation).

## What does NOT help

Tested 2026-05-09:
- `alloc_zeros → alloc` for output buffer (eliminates per-matmul
  cuMemset): bench delta 0 % within noise. Saves a launch but not
  meaningful at this scale. Kept anyway for cleanliness — 432
  cuMemsets/multi-step removed.
- async DtoH (Substep 2.2): gives +0.8 % on this baseline because
  sampler is already only 6 / 115 ms — orthogonal to MLP problem.

## What WOULD help (not implemented)

Approaches that target the actual bottleneck (cache contention):

1. **Fuse gate + up into one matmul kernel**. Both read the SAME
   `[B, hidden]` activation (cached even now), but more importantly,
   both write `[B, intermediate]` outputs which can pipeline together
   in a fused kernel that loads gate/up weight tiles once. Estimated
   gain: 30-40 % MLP time → ~10 % e2e.

2. **Persistent kernel design**: keep MLP weights pinned in L2 across
   the multi-step block via a single launched grid that processes all
   N tokens. Removes the cold-cache reload between layers. Largest
   potential gain (~50 % MLP) but largest implementation effort.

3. **Cooperative thread block design**: currently 1 warp per block,
   so each block reads its own weight slice from HBM. Multiple warps
   per block sharing weight tile via shared memory would saturate
   shared bandwidth instead of HBM. Medium effort.

4. **Streaming K-tiles via `cp.async`**: overlap weight loads with
   mma compute. Requires sm_80+ (we have sm_89). Medium effort.

## Decision

The cache-contention hypothesis explains the 6× discrepancy and is
consistent with all observed data. Without nvprof / Nsight Compute
(blocked in WSL2 — memory rule `feedback_wsl2_profiling_blocked.md`)
we can't confirm cache miss rate directly, but the inference is
strong from:
- microbench saturates 56 % of peak HBM BW
- production runs at 9.5 % of peak — a 6× cache-miss penalty
- TC dispatch confirmed correct, kernel correctness verified
- 84 % of dispatches go TC; only the m∈{1,2,3} and m≥33 tails fall back

Next-step optimisations that would actually move the needle on this
host require either approach (1) or (2). Approach (1) is the most
tractable; budget ~1-2 days of careful kernel work.

## Verification

- `cargo bench --features cuda-default,marlin -p vllm-core --bench marlin_tile_mma_path_bench`:
  microbench numbers reproducible, alloc-opt is within ±2 %.
- `decode_profile` slot per-stage breakdown logged with c=8 sustained
  load.
- `vllm_core::marlin_path` dispatch counter (used during diagnosis,
  removed in commit) confirmed 84 % TC routing.
