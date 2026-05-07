# CLAUDE.md

## Project

Rust-native LLM inference engine. Alternative to vLLM.

## Stack

- Language: Rust (edition 2021)
- Build: Cargo
- GPU: CUDA via cudarc / custom kernels
- Async: tokio

## Reference

vLLM source: `reference/vllm/`. Consult before implementing any component.

- Baseline commit: `14d03b8` (2026-01-23) — initial version used during development
- Current commit: `fa9e6802` (2026-04-03) — latest synced version

## Principles

- TDD: test first, implement second, refactor third
- Performance over convenience
- Zero-copy where possible
- Unsafe only when measured and justified
- No premature abstraction — earn generality through repetition
- Errors: thiserror for libraries, anyhow for binaries
- No unwrap() in production paths

## Code Style

- `cargo fmt` — non-negotiable
- `cargo clippy` — zero warnings
- Names: descriptive, no abbreviations except domain-standard (KV, QKV, MHA, GQA, SwiGLU)
- Comments: only "why", never "what"
- Tests: unit tests in-module, integration tests in /tests

## Bench Coverage

Every change that touches a hot path **MUST** ship with a criterion bench
in `crates/core/benches/` that measures it. A "hot path" is anything in
the engine step (scheduler, KV alloc, attention, sampling), the
forward path of any model, or any quantization/MoE/SSM kernel.

- New optimisation → add the bench *first* (records baseline), then
  the optimisation (records improvement). Without the before/after
  pair the change cannot be reviewed honestly.
- Refactor that touches a hot path → confirm the bench is unchanged
  ≤ 2% by running `scripts/run_benches.sh --filter <bench> --label
  pre/post` and diffing the JSON snapshots.
- E2E behaviour (TTFT, batched throughput) → covered via
  `scripts/bench_decode.py` and `scripts/bench_prefill.py` against a
  running server. These are the user-visible regression guards.
- Snapshots land in `docs/perf/bench-history/` as part of the same
  commit as the perf-relevant change. Reviewers diff them.

A perf change without a bench is not done. A 50% speedup that surfaced
only via external comparison (e.g. Python vLLM) is a process failure —
the bench should have caught it months earlier.

## Architecture Decisions

Document in `/docs/adr/NNNN-title.md` when:
- Choosing between competing approaches
- Introducing unsafe
- Adding dependencies

## Quality Bar

Production-grade. Every line, every commit, every decision — as if it ships to thousands of GPUs tomorrow. No prototyping mindset, no "fix later", no shortcuts. Code reviews would pass at a top-tier infra team.

## Agent Expectations

- Read before write. Always.
- Verify assumptions with code, not guesses
- If unsure — ask, don't assume
- Run `cargo check` after every edit
- Run `cargo test` before declaring done
- No stubs. Either complete the implementation or document the gap with a detailed TODO.
- Workarounds require TODO with explanation of the proper fix.
- Use NOTE for non-obvious decisions or important context that isn't actionable.
- Think before generating. Less code that works > more code that might.
