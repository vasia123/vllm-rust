# Stage 15 status — checkpoint after the 2026-05-08 session

22 commits, full software AWQ tile-MMA stack landed, hybrid dispatch
wired and gated, two negative-result analyses captured. Production
behaviour unchanged by default; tensor-core work (the remaining win
path) starts from a clean foundation.

## Landed

| Substep | Commit  | Outcome |
|---------|---------|---------|
| 14-C.1  | `8cd6dc1` | decode_profile → CUDA-event timing |
| 14-C.2  | `c15fb41` | decode breakdown c=1/4/8 (AWQ-Marlin = 91 % at c=8) |
| 15.A    | `c05a595` | Marlin reference notes (tile shapes, MMA, dequant) |
| 15.B    | `c1b2bef` `acc1dd5` | CPU AWQ→Marlin tile repack + 4 round-trip tests |
| 15.C    | `065be42` `95ba0a7` `c34fa82` | design + scaffold + dispatcher + 1 GPU test |
| 13-H FU | `a59cba0` `d8e4134` | candle 0.10.2 vanilla; vendor/ retired |
| 15.D.1  | `7013f24` | kernel multi-K |
| 15.D.2  | `265e2e8` | kernel multi-M |
| 15.D.3  | `3ad0538` | kernel multi-N |
| 15.D.4  | `27b0bef` | kernel + dispatcher: group_size + AWQ qzeros |
| 15.D bench | `b54692c` | software baseline microbench |
| 15.D vs prod | `8dc4087` | surprise: software beats marlin_gemm at M=1..8 (one shape) |
| ADR 0016 | `3f0f1ba` | hybrid-dispatch design |
| 15.E.1  | `a361afc` | tile_b storage at load |
| 15.E.2  | `c0ba543` | forward dispatch on M |
| 15.E.3  | `7dd5aac` | env-gate; e2e bench: **0 % lift** (negative) |
| 15.E.3 follow-up | `60ade04` | multi-shape sweep: software loses at M=1 on small-K layers |
| 15.D-body.1 | `87c060e` | INT4→BF16 LOP3 dequant primitive + standalone GPU test |
| 15.D-body.2a | `c286f0f` | bf16 mma.m16n8k16 single-tile probe (manual fragments, no ldmatrix) |
| 15.D-body.2b | `29cd5e8` | INT4 dequant + scale/zp + mma fused in one kernel (single-tile) |
| 15.D-body.3a | `52c7b15` | TC kernel for M=16,K=16,g=16 routed via dispatcher |
| 15.D-body.3b | `a837f80` | TC multi-K extension (group_size=K) |
| 15.D-body.3c | `63f97fa` | TC multi-group_size extension |
| 15.D-body.3d | `03e5896` | TC small-M / multi-m-tile via masking — full production-shape contract on TC |
| 15.D-body.4  | `b9bc05f` | TC microbench: 2-5× vs production marlin_gemm at M=4..8 |
| 15.D-body.5  | `ce162d6` | initial e2e bench: 0% lift (caused by 3D-shape gate bug) |
| 15.D-body.5 fix | (this commit) | **+74% c=8, +23% c=4** after 3D-shape fix in awq_marlin gate |

## Current production behaviour

- Default build: identical to commit `a59cba0` (post-13-H follow-up).
  `bench_decode --runs 3` matches baseline (43 / 65 / 64 tps at c=1/4/8).
- `VLLM_AWQ_HYBRID=1` opt-in: tile_b loaded at startup (+1.25 GiB
  VRAM); forward dispatches M ∈ [4, 8] through `tile_mma_v1` software
  path. e2e tps unchanged. Numeric correctness preserved (7/7
  `scripts/test_qwen3_awq_correctness.sh`).
- All 4502 unit tests + 9 marlin_tile_mma GPU tests + 13 awq_marlin
  tests pass.

## Why the win didn't materialise e2e

Per `docs/perf/bench-history/2026-05-08-15E-shape-sweep.json`:

- The 15.D microbench measured ONE shape (Qwen3 MLP-up K=4096
  N=11008). Multi-shape sweep reveals software loses at M=1 on
  small-K layers (q_o by 2.13×, down by 2.98×) — these layers
  outnumber the ones where software wins.
- Even with the corrected M ∈ [4, 8] threshold, e2e bench shows
  ≈0 % lift. Hypothesis 2 from 15.E.3 (per-call launch overhead
  amortisation differs in microbench vs real workload) remains
  unresolved; would need nvprof under production load to settle.

## Foundation in place for Stage 15.D-body (tensor-core MMA)

The path that *would* deliver e2e wins — a real tensor-core
`mma.m16n8k16` kernel — is now unblocked by everything below:

1. Producer: `awq_to_marlin_tile_repack_cpu` (Stage 15.B) emits the
   tile layout the kernel expects, with 4 round-trip tests.
2. Dispatcher: `dispatch_marlin_tile_mma_v1` (Stage 15.C/D) handles
   shape extraction, validation, cudarc launch. Same signature will
   be reused for the tensor-core kernel; only the PTX changes.
3. Storage: `AwqMarlinLinear.tile_b` (Stage 15.E.1) populated at load
   time, env-gated. No runtime repack.
4. Forward gate: M ∈ [4, 8] hybrid dispatch (Stage 15.E.2) — when
   tensor cores beat both software and marlin_gemm across the M
   sweep, this gate widens to "always tile_mma" (or stays at the
   measured crossover, whichever bench dictates).
5. Tests: 9 GPU correctness tests (M ∈ {1, 4, 16, 32, 64} × N ∈
   {64, 128, 256} × K ∈ {16..128}, group_size ∈ {16, 32, 64, 128}).
   Tensor-core swap-in must pass these untouched.
6. Bench: software baseline at canonical shapes; tensor-core variant
   diffs directly.

## Stage 15.D-body — concrete starting point for next iteration

Per `docs/perf/marlin-tile-mma-step-15c-design.md` (commit `065be42`),
the kernel body changes are:

- Replace the per-thread software dot-product loop with:
  - cooperative load of A into shared memory + `ldmatrix.x4` → FragA
  - INT4 dequant via `LOP3(MASK=0x000f000f, EX=0x43004300)` + bf16
    SUB (per `dequant.h:174-215`) → FragB
  - Two `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` calls
    (n=0..7 and n=8..15) with shared FragA, FP32 accumulation
  - Output FragC FP32 → BF16 store with the m16n8 thread-mapping
- Block dims switch from (256, 1, 1) to (32, 1, 1) (1 warp).
- Grid sizing changes accordingly.
- Kernel name unchanged (`marlin_tile_mma_int4_bf16`); dispatcher /
  Rust wrapper / tests / bench all keep working untouched.

Risk: PTX inline asm + fragment layout subtle. Single-tile (M=N=K=16,
g=K) variant first; multi-tile and group_size loop already in the
software K-loop and will inherit cleanly.

Estimated effort (per 15.A reference notes): 3-4 days focused work
(PTX, debug, correctness against software path, microbench).

## Lessons captured in memory + docs

- `feedback_bench_runs.md` — `bench_decode --runs >= 3` mandatory on
  laptop GPUs (avoids power-state artefacts; Stage 13-H 45 % "drop"
  was entirely this).
- `feedback_loop_wakeup.md` — /loop wakeup ≤ 60 s when no running
  process to wait on.
- `feedback_python_env.md` — never `pip install --user` /
  `--break-system-packages`; project venv at `/tmp/hf-venv/`.
- ADR 0016 — hybrid-dispatch design decisions + risks.
- This file — Stage 15 substep state-of-the-world.
