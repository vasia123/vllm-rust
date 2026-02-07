# ADR-0003: PTX Compilation Strategy

## Status

Accepted (retroactive)

## Context

CUDA kernels can be compiled to:

1. **PTX** (Parallel Thread Execution) - NVIDIA intermediate representation, JIT-compiled by driver at load time
2. **CUBIN** - Native GPU binary for a specific SM version, no JIT needed
3. **fatbin** - Bundle of multiple CUBIN + PTX for different architectures

## Decision

We compile all kernels to **PTX** at build time via `nvcc --ptx`, loaded at runtime via `include_str!` and cudarc's `get_or_load_custom_func`.

## Rationale

- **Forward compatibility**: PTX compiled for sm_80 runs on sm_89, sm_90+ via driver JIT
- **No fatbin complexity**: Avoids maintaining multiple CUBIN variants and selecting at runtime
- **Simple build**: Single `nvcc` invocation per kernel, no CUDA driver API for module loading
- **candle integration**: cudarc expects PTX strings, not binary modules
- **Acceptable overhead**: JIT compilation adds ~10-50ms per kernel on first use, amortized over the runtime

### SM-aware compilation (Phase 0.1)

The build system (`build.rs`) now:
- Auto-detects GPU architecture via `nvidia-smi` or `CUDA_ARCH` env var
- Compiles each kernel at its minimum required SM version (sm_75 for GPTQ, sm_80 for BF16, sm_89 for FP8)
- Skips kernels that require higher SM than the target (e.g., FP8 kernels skipped on A100)
- Exports `CUDA_TARGET_SM` for runtime feature availability checks

### Trade-offs accepted

- First kernel launch incurs JIT overhead (~10-50ms per kernel)
- PTX may not exploit SM-specific instructions as aggressively as native CUBIN
- For production deployments with known GPU targets, CUBIN would give ~1-3% better performance

## Consequences

- All `.cu` files compile to `.ptx` in `crates/core/kernels/`
- PTX included as compile-time string constants via `include_str!`
- Runtime SM capability checks via `cuda_target_sm()` for feature gating
- Build requires `nvcc` in PATH and CUDA toolkit installed
