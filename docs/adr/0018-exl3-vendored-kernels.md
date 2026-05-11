# 0018 — EXL3 vendored CUDA kernels

Status: accepted (2026-05-11)

## Context

The EXL3 quantization format (ExLlamaV3 by Turboderp) uses QTIP-style
trellis-coded vector quantization with separate Hadamard pre/post
transforms. There is no public reference implementation outside the
ExLlamaV3 codebase — the trellis decoder and Hadamard butterflies are
hand-tuned CUDA with non-trivial template specialisation per
bits-per-weight (K=2..8). The format is GPU-only by design.

Implementing inference for `turboderp/Llama-3.1-8B-EXL3-3.0bpw` requires
those kernels. Three options were considered:

1. **Vendor `.cu`/`.cuh` from ExLlamaV3 (chosen).** ExLlamaV3 is
   MIT-licensed; copying source into this repo is legally clean.
   Effort: a few days of include-path / stub fixes to drop the libtorch
   dependency.
2. **FFI to `_exllamav3_ext.so`.** Faster to integrate but adds a
   Python/libtorch runtime dependency — contradicts the project's
   "Rust-native" goal and blocks distribution.
3. **Reimplement from scratch on `cudarc`.** Multi-month R&D against
   the QTIP paper, with high risk of underperforming Turboderp's
   hand-tuned PTX on every detail.

## Decision

Vendor the relevant `.cu` and `.cuh` files into
`crates/core/kernels/exl3/`. Compile to PTX via `build.rs` (10 targets:
`hadamard`, `reconstruct`, `exl3_gemv`, and seven per-K `comp_units`).
Drive launches from Rust through `cudarc`. Skip vendoring
quantization/training/MoE sources — inference-only scope.

To make vendored `.cu` files compile without libtorch we provide
`crates/core/kernels/exl3/exl3_torch_stub.h`: minimal no-op declarations
for `at::Tensor`, `c10::optional`, and the `TORCH_CHECK_*` macro family.
This lets the host-side launcher functions parse cleanly even though
they are never called — the device kernels are what we want, and PTX
only carries device code.

## Consequences

- **Upstream contract.** When ExLlamaV3 changes its trellis encoding
  (format version bumps, new bpw values), we must re-vendor and re-test.
  The vendored snapshot is recorded in `LICENSE-THIRD-PARTY` together
  with the upstream MIT licence.
- **No libtorch in the build.** `nvcc --ptx` succeeds without any torch
  headers thanks to the local stub. The `.so` artefact is correspondingly
  smaller and the runtime has no Python/torch surface.
- **PTX size.** Per-K comp_units are roughly 6 MB of PTX each due to the
  template fan-out (3 codebook variants × 4 shape specialisations × 2
  output dtypes). Total cost for 7 K-variants is ~45 MB of PTX shipped
  in the crate; cudarc JIT-loads only what's actually used.
- **Diagnostics suppressed.** `--diag-suppress=174` and `--diag-suppress=550`
  silence "expression has no effect" and "set but unused" warnings that
  trip on the vendored host launchers. Local code keeps strict warnings.
- **Per-K specialisation.** Each bits-per-weight has its own PTX file.
  We dispatch by reading the trellis tensor's last dim (`16*K`) at load
  time and picking the matching cubin (Phase 4).

## Alternatives rejected

- *Surgical strip of host launchers per file.* More auditable but
  vastly more edits — every TORCH_CHECK site, every host function body.
  Risk of typos. The stub approach gives the same result with one file
  of additions.
- *Recompile kernels using cust / a Rust CUDA frontend.* The kernels
  rely on inline PTX (`lop3.b32`, `vabsdiff4.u32`, `mma.m16n8k16`).
  Translating these to Rust-side intrinsics has zero upside and high
  bug surface.

## References

- ExLlamaV3 source: <https://github.com/turboderp-org/exllamav3>
- QTIP paper: <https://arxiv.org/abs/2406.11235>
- Phase plan: `~/.claude/plans/curried-coalescing-tome.md`
