# ADR-0001: Candle as Tensor Backend

## Status

Accepted (retroactive)

## Context

A Rust-native LLM inference engine needs a tensor computation library with CUDA support. The main candidates were:

1. **Candle** (Hugging Face) - Pure Rust ML framework with CUDA via cudarc
2. **tch-rs** - Rust bindings to libtorch (PyTorch C++ backend)
3. **burn** - Rust ML framework with multiple backends
4. **Custom** - Direct cudarc usage without a tensor abstraction

## Decision

We chose **Candle** as the tensor backend.

## Rationale

- **Pure Rust**: No C++ build dependency (unlike tch-rs which requires libtorch ~2GB)
- **cudarc integration**: Clean CUDA interop via `CudaStorage`, `CustomOp` traits for custom kernels
- **HuggingFace ecosystem**: Native safetensors loading, tokenizer compatibility, model weight format alignment
- **CustomOp trait**: Allows injecting hand-written CUDA kernels (PTX) into the computation graph without forking the framework
- **Active maintenance**: Regular releases aligned with HF model releases

### Trade-offs accepted

- Candle is less mature than PyTorch — some operations are slower or missing
- No automatic differentiation needed (inference only), so candle's limited autograd is acceptable
- Custom CUDA kernels (paged attention, fused activations) compensate for generic op performance gaps

## Consequences

- All tensor operations go through `candle_core::Tensor`
- Custom CUDA kernels implement `CustomOp1`/`CustomOp2` traits
- Model weights loaded via `candle_nn::VarBuilder` from safetensors
- PTX kernels compiled at build time, loaded via `include_str!` + `get_or_load_custom_func`
