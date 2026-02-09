# ADR-0004: vLLM Reference Update Strategy

## Status

Accepted

## Context

Our reference repository `reference/vllm/` fell 493 commits behind upstream (17 days, 2026-01-23 to 2026-02-09). The delta includes performance improvements, new models, bug fixes, quantization changes, and speculative decoding enhancements.

We need a systematic approach to triage and port relevant changes while skipping those that don't apply to a Rust-native engine.

## Decision

Adopt a **phased prioritization** strategy for syncing with upstream vLLM changes.

### Prioritization criteria

1. **High**: Performance improvements to existing features (attention, KV cache, spec decode, quantization kernels)
2. **High**: Bug fixes that affect correctness
3. **Medium**: New model architectures (incremental, follow existing patterns)
4. **Medium**: MoE and quantization method improvements
5. **Low**: Frontend/API additions
6. **Skip**: Changes not applicable to Rust-native engine

### Phases

| Phase | Scope | Rationale |
|-------|-------|-----------|
| 0 | Infrastructure (pin commits, ADR) | Foundation for tracking |
| 1 | Performance + Spec Decode | Highest impact on throughput |
| 2 | New models | User-facing feature parity |
| 3 | MoE + Quantization | Memory efficiency |
| 4 | Bug fixes + Frontend | Correctness + API completeness |

### What we skip

- **torch.compile optimizations**: No analog in Rust; we use compile-time optimization
- **TRTLLM-specific fixes**: Different backend entirely
- **ROCm/XPU/CPU-specific patches**: Not supported yet; tracked separately
- **CI/CD and Docker-only changes**: Infrastructure, not engine code
- **Python-specific refactors**: Code organization changes in Python don't map to Rust

### What we defer

- Native Weight Syncing API — only relevant if we add RL training support
- OTEL tracing — observability improvement, not critical path
- KV Connector rework — disaggregated inference, future work
- Fabric/MNNVL detection — NVLink mesh topology, hardware-specific

## Rationale

A phased approach allows us to:
- Ship correctness fixes quickly (Phase 4.1 bug fixes first)
- Validate performance changes with benchmarks before and after
- Add models incrementally without destabilizing core engine
- Track which upstream changes have been ported vs skipped

The alternative — a single large sync — risks introducing regressions across multiple subsystems simultaneously.

## Consequences

- CLAUDE.md tracks baseline and current reference commits
- Each significant port references the upstream PR number in commit messages
- Performance-critical changes (attention layout, spec decode) get their own ADRs
- Skipped changes are documented here to avoid re-evaluation
