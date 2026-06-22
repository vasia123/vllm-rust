# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, minor bumps may include breaking changes to the
HTTP API, CLI flags, or on-disk formats; see [docs/RELEASING.md](docs/RELEASING.md).

## [Unreleased]

## [0.2.0] - 2026-06-23

First tagged release. The version was previously pinned at `0.1.0` with no
tags; the first `scripts/release.sh` run starts the real release history.
Highlights:

### Added
- **Native I-quant (IQ) GGUF support** — IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S /
  IQ4_XS, which candle cannot parse. Byte-exact dequant (CPU + CUDA) pinned to
  gguf-py golden vectors, a dedicated GGUF header parser, and `IqLinear`. Runs
  `gemma-4-12b-it-UD-IQ3_XXS` on an 8 GB GPU. (ADR 0025)
- Real `/v1/embeddings` on the loaded model + EmbeddingGemma (gemma3 GGUF)
  parity with ollama.
- Chat-template resolution for `/v1/chat/completions` on GGUF checkpoints.
- `--version` flag on the `vllm-server` CLI.

### Changed
- **q8_1 MMVQ decode for I-quants** — the decode hot path now quantizes the
  activation to q8_1 once and dots the I-quant weight with integer `__dp4a`
  (a port of ggml's `vec_dot_iq*_q8_1`), no dense weight materialized. Decode
  on gemma-4-12B IQ3_XXS: **0.41 → 3.3 → 21.5 tok/s**; the matmul kernel is
  74–130× faster than dequant-then-matmul. (ADR 0025)
- KV-cache memory reserve is now **profiled** (dummy prefill+decode measures
  peak non-KV memory) instead of a fixed 1.5 GiB constant. (ADR 0024)
- Chunked prefill is **on by default** — no silent long-prompt starvation.
- Gemma-4 sliding-window layers use windowed paged decode — O(window) at any
  context length.

### Fixed
- Grammar-compile isolation: async path + single-flight + compile deadline
  (a pathological EBNF could pin a core until restart). (ADR 0023)
- Generation is capped at the effective context length — no single-sequence
  KV-cache wedge.

[Unreleased]: https://github.com/vasia123/vllm-rust/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vasia123/vllm-rust/releases/tag/v0.2.0
