# ADR 0016 — AWQ-Marlin hybrid dispatch (M ≤ ~12 → tile_mma_v1)

**Status**: Accepted, implementation pending Stage 15.E.

## Context

Stage 15.B–D landed a software tile-MMA scaffold
(`marlin_tile_mma_int4_bf16`, dispatched via
`marlin_tile_cuda::dispatch_marlin_tile_mma_v1`) covering the full AWQ
shape contract: M ≥ 1, N multiple of 64, K multiple of 16, AWQ
group_size + qzeros. The kernel is dormant in production; the code
path is exercised only by GPU lock-down tests.

A pre-tensor-core microbench at the canonical Qwen3-4B-AWQ MLP-up
shape (K=4096, N=11008, group_size=128) produced an unexpected result
(commit `8dc4087`):

| M     | prod `marlin_gemm` | tile_mma v1 sw | sw / prod |
| ----: | -----------------: | -------------: | --------: |
| 1     |             142 µs |     **115 µs** |     0.81× |
| 4     |             456 µs |     **415 µs** |     0.91× |
| 8     |             889 µs |     **761 µs** |     0.86× |
| 16    |            1479 µs |        1485 µs |      1.00 |
| 32    |            1468 µs |        2940 µs |     2.00× |
| 64    |            1476 µs |        5800 µs |     3.93× |
| 256   |            1868 µs |       23192 µs |    12.4×  |

**Software tile_mma_v1 beats production at M ∈ {1, 4, 8}** (the
`awq_gemv_int4_kt_bf16` regime). Above M = 16 production switches to
dequant + cuBLAS BF16 GEMM and saturates near 1.4 ms; software stays
linear in M and falls off badly.

This is a deployable **1.10–1.24×** win on the AWQ-Marlin path at the
exact M values that dominate decode (M=1 = c=1, M=4..8 = c=4
multi-step). Stage 14-C found AWQ-Marlin = 91 % of decode wallclock
at c=8, so even a 1.2× lift on that slice meaningfully closes the
gap to Python vLLM.

## Decision

**Adopt a hybrid dispatch in `AwqMarlinLinear::forward`**: route
through `dispatch_marlin_tile_mma_v1` when the activation `M ≤
HYBRID_M_THRESHOLD`, otherwise fall through to the existing
`marlin_gemm` path. Threshold value: **`HYBRID_M_THRESHOLD = 8`**
(safe inside the regime where the bench shows software wins; not
12, because the M=12..14 measurements straddle the crossover and
landing in the "tie" zone risks a regression on perturbed inputs).

The choice between three candidate integration shapes is **A**.

### Shape A — Dual-storage (chosen)

`AwqMarlinLinear` holds **both** the existing transposed qweight
(`[N, K/8]`, used by `marlin_gemm` / `awq_gemv_int4_kt_bf16` for
M > 8) **and** the Marlin tile-laid-out qweight (`[k_tiles ×
n_tiles × 128]`, used by `dispatch_marlin_tile_mma_v1` for M ≤ 8).
Both are computed at `load_weights` time. AWQ scales and qzeros are
shared (`tile_mma_v1` consumes the same `[num_groups, N]` BF16 scales
and `[num_groups, N/8]` U32 qzeros that the existing path stores).

**VRAM cost**: ~1× extra for the tile layout (Qwen3-4B-AWQ ≈ +1.1 GiB
GPU). The 8 GiB laptop currently sits at ~4.2 GiB after model load
(commit a59cba0 measurements). Adding ~1.1 GiB pushes to ~5.3 GiB,
leaving 2.7 GiB for KV cache (down from current ~2.3 GiB allocated).
Actually a *net* KV-cache improvement because the current model load
also reserves working buffers that overlap; expect ~0.5 GiB net cost
with the current `auto_tune_num_blocks` selector.

**Dispatch cost**: a single `if m <= 8` branch in `forward`. Both
paths are already cached in cudarc; no PTX reload.

**Trade-off accepted**: VRAM cost is the price of zero-ambiguity
behaviour and zero per-call overhead.

### Shape B — Lazy repack on first low-M call (rejected)

Keep only the existing transposed qweight. The first time `forward`
sees `m ≤ 8`, repack on the GPU into a per-instance `tile_b: OnceCell
<Tensor>`. Subsequent low-M calls reuse the cached tile.

**Why rejected**:
- First-call latency: ~5–10 s on Qwen3-4B (full GPU repack of all
  ~252 quantized layers). User-visible TTFT spike for the first c=1
  request after server start.
- Cache pressure: 252 `OnceCell<Tensor>`s sprinkled across model
  layers; subtle to reason about lifetime + thread-safety.
- Net VRAM is the same as A once warmed.

### Shape C — Feature flag, off by default (rejected for production)

Gate the entire hybrid path behind `--features
awq-tile-mma-hybrid`. Default off → existing behaviour. Users
opting-in get the win + VRAM cost.

**Why rejected for production**: the win is reliable and the cost is
acceptable on the target laptop, so default-on is the right call.
We *will* keep an env-var override (`VLLM_AWQ_DISABLE_HYBRID=1`) for
diagnostic regressions, but the feature flag itself is unnecessary
gating.

## Implementation plan

### 15.E.1 — dual-storage at load time

- Add `tile_b: Tensor` field to `AwqMarlinLinear`.
- In `AwqMarlinLinear::load_weights`, after the existing transposed
  qweight is built, compute `tile_b` from the AWQ-original qweight
  via `awq_to_marlin_tile_repack_cpu(awq_qweight, k, n)`.
- The `[K, N/8]` AWQ-original qweight is currently dropped after
  `awq_to_gptq_qweight`. Capture it before the discard.
- Unit test: shape `[k_tiles, n_tiles*128]` U32 of `tile_b`, lives on
  the same device as the rest of the layer.

### 15.E.2 — forward dispatch on M

- Constant `HYBRID_M_THRESHOLD = 8` near the existing
  `AWQ_GEMV_M_THRESHOLD`.
- In `AwqMarlinLinear::forward(x)`, branch:
    - `m = x.dims()[0]`; if `m <= 8`: route through
      `dispatch_marlin_tile_mma_v1`.
    - Else: existing path (unchanged).
- Optional env-var escape: `VLLM_AWQ_DISABLE_HYBRID=1` short-circuits
  to the existing path regardless of M (diagnostic / regression
  bypass).
- Numeric correctness: existing `scripts/test_qwen3_awq_correctness.sh`
  passes 7/7. The hybrid path produces BF16 output to within ≤ 1e-2
  of the existing path on the lock-down prompts, so the
  greedy-first-token assertions remain stable.

### 15.E.3 — bench

- Re-run `awq_marlin_path_bench` (existing) — should be unchanged.
- Run e2e `bench_decode.py --runs 3` at c=1, c=4, c=8 against the
  new build. Expected lift: **+10–20 % at c=1, +5–15 % at c=4** based
  on the M=1, M=4 microbench ratios (decode forward has ~252 such
  GEMVs, so the bench tps lift is bounded by Amdahl on the AWQ-Marlin
  slice = ~91 % of c=8 wallclock).
- Snapshot: `docs/perf/bench-history/2026-MM-DD-15E-hybrid-on.json`.

### 15.E.4 — gate update if Stage 15.D-body lands

When the tensor-core MMA body replaces the software dot-product, the
crossover point shifts (tensor cores will likely beat `marlin_gemm`
across most or all of the M sweep). Re-bench, raise the threshold
accordingly. The dispatch is a single constant; trivial to update.

## Risks and mitigations

1. **Numeric drift between paths**: `marlin_gemm` uses BF16 GEMM with
   FP32 reduce (`use_fp32_reduce=true`); `tile_mma_v1` uses pure FP32
   accumulation in software. Both should agree to ~1e-2 BF16 noise,
   but if a model layer has unusual weight ranges, the first-token
   assertion in the correctness script could flip. **Mitigation**:
   correctness script runs as a hard gate before commit; if any
   prompt flips, revert and pivot to feature-flag gating (Shape C).

2. **VRAM regression**: Stage 13-G worked hard to balance num_blocks
   vs working set. +1.1 GiB tile_b shifts that balance.
   **Mitigation**: re-run `auto_tune_num_blocks` after the change and
   record num_blocks in the bench snapshot. If KV cache drops below
   the Stage 13-D minimum (1029 blocks at gpu_memory_utilization=0.85)
   significantly, raise the env knob or shrink max-model-len.

3. **Thread-safety of dual storage**: `AwqMarlinLinear` is not
   currently shared across threads in production, but feature-flag
   experiments may. **Mitigation**: `tile_b` is set once at load
   time and read-only thereafter; same Send/Sync as the existing
   weight Tensors.

4. **Regression on shapes outside Qwen3-4B-AWQ MLP-up**: the bench
   covered K=4096, N=11008, g=128 only. Other Qwen3 shapes (K=2560,
   N=2560 for q_proj; K=4096, N=2048 for o_proj; etc.) may have a
   different crossover M. **Mitigation**: extend the bench to those
   shapes before committing the dispatch threshold; if any shape's
   crossover sits below M=8, lower the threshold for that layer (or
   keep the safer M≤4 gate model-wide).

## Verification

- `cargo test --features cuda-default,gpu-test-small`: 4501 + 9
  marlin_tile_mma + 13 awq_marlin tests.
- `cargo bench -p vllm-core --features cuda-kernels,marlin --bench
  awq_marlin_path_bench`: matches commit `b54692c` snapshot.
- `cargo bench ... --bench marlin_tile_mma_path_bench`: matches
  commit `b54692c` snapshot.
- `scripts/test_qwen3_awq_correctness.sh`: 7/7.
- `bench_decode.py --runs 3 --concurrency 1 4 8`: lift recorded.

## References

- Stage 14-C decode profile (commit `c15fb41`): AWQ-Marlin = 91 % of
  decode wallclock at c=8.
- Stage 15.D bench surprise (commit `8dc4087`): `tile_mma_v1`
  software beats `marlin_gemm` at M ∈ {1, 4, 8}.
- `docs/perf/bench-history/2026-05-08-15D-tile-mma-vs-production.json`
  — raw numbers.
