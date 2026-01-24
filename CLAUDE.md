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
- One logical change per commit
- If unsure — ask, don't assume
- Run `cargo check` after every edit
- Run `cargo test` before declaring done
- No placeholder code. No TODOs without a linked issue.
- Think before generating. Less code that works > more code that might.
