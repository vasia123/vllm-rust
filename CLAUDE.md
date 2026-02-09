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
- Current commit: `3025b3c` (2026-02-09) — latest synced version

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
