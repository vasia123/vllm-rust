# Feature Gap Analysis: vLLM vs vllm-rust

## Summary

vllm-rust implements ~85% of vLLM's core functionality. The project has grown from an initial prototype to a comprehensive inference engine with 55 model architectures, 9 custom CUDA kernels, and full quantization support.

---

## Current State (Implemented)

| Category | Status |
|----------|--------|
| **Core** | PagedAttention v1 (parameterized head_dim/block_size), continuous batching, FCFS/priority scheduler, chunked prefill, preemption |
| **Engine** | ExecutionStrategy trait, standard + speculative decoding, CUDA graph capture/replay, warmup system |
| **Attention** | 4 backends: FlashInfer, FlashAttention-2, Naive, MLA. Auto-selection by GPU capability |
| **Speculative** | Draft model, n-gram, suffix, EAGLE, Medusa proposers. Tree attention verification |
| **KV Cache** | Block management, prefix caching, COW, eviction, FP8/INT8 quantized KV, CPU offload, MLA cache |
| **Sampling** | Greedy, top-k, top-p, min-p, temperature, repetition/frequency/presence penalty, logprobs, beam search |
| **Constraints** | JSON schema, regex, choice constraints |
| **Quantization** | FP8, GPTQ, AWQ, BitsAndBytes (NF4/INT8), GGUF, Marlin |
| **Models** | 55 architectures including quantized + LoRA variants |
| **LoRA** | Per-request adapter loading, multi-adapter serving |
| **MoE** | Fused MoE kernels, expert parallelism, token dispatch |
| **Multimodal** | Vision (LLaVA), audio processing |
| **Distributed** | NCCL, tensor parallelism, pipeline parallelism |
| **Tool Calling** | JSON parser, Hermes format |
| **SSM** | Mamba/Mamba2 state-space models |
| **API** | OpenAI-compatible completions, chat, embeddings, tokenize. SSE streaming |
| **Admin** | Prometheus metrics, Vue.js panel, graceful restart |
| **CUDA Kernels** | 9 custom kernels: paged attention, FP8 quant/gemm, GPTQ dequant, fused MoE, SwiGLU, top-k softmax, BnB matmul |

---

## Remaining Gaps

### High Priority (Performance)

| # | Feature | Impact | Notes |
|---|---------|--------|-------|
| 1 | **RMSNorm CUDA kernel** | 5-15% throughput | Called 80+ times per forward pass |
| 2 | **RoPE CUDA kernel** | 3-8% throughput | Currently multi-launch through candle ops |
| 3 | **Paged Attention V2** | 2-5x for long contexts | V1 adequate for <32k, V2 needed for 32k+ |
| 4 | **Cache reshape CUDA kernel** | Reduced latency | Currently tensor ops, should be fused |
| 5 | **GPU-side sampling** | 10-20% latency (large batch) | Currently CPU-side, 32MB+ GPU->CPU transfer at batch_size=64 |
| 6 | **GELU/GeGLU CUDA kernel** | Moderate | Needed for BERT, T5 model families |

### Medium Priority (Functionality)

| # | Feature | Notes |
|---|---------|-------|
| 7 | **Request cancellation (abort)** | Client disconnect still consumes GPU |
| 8 | **Parallel sampling (n > 1)** | OpenAI API `n` parameter blocked |
| 9 | **Extensible logits processor pipeline** | Hard-coded sampling logic |
| 10 | **Marlin GEMM kernel** | GPTQ via basic path, Marlin would be 2-3x faster |
| 11 | **CUTLASS Scaled MM** | FP8 W8A8 via naive kernel, CUTLASS critical for H100 |

### Lower Priority (Future)

| # | Feature | Notes |
|---|---------|-------|
| 12 | **Custom All-Reduce** | P2P/NVLink for multi-GPU, 2-3x over NCCL for small tensors |
| 13 | **Encoder-decoder models** | T5, BART, cross-attention |
| 14 | **Video multimodal** | Video frame processing |
| 15 | **Fused QKNorm + RoPE** | DeepSeek V3 optimization |

---

## Model Support (55 architectures)

### Decoder-only (Causal LM)
Llama, Llama2, Llama3, Qwen2, Qwen3, Qwen2-MoE, Qwen3-MoE, Mistral, Mixtral, Gemma, Gemma2, Gemma3, Phi, Phi3, DeepSeek, DeepSeek V2, Jamba, GLM4, GLM4-MoE, Cohere, Baichuan, InternLM2, StarCoder2, GPT2, GPT-NeoX, Falcon, Bloom, MPT, ExaOne, OLMo2, Persimmon, DBRX

### Embedding
BERT, BertForSequenceClassification

### Multimodal
LLaVA

### State-Space
Mamba, Mamba2

### Quantized Variants
Most models have GPTQ/AWQ/FP8/BnB variants + LoRA variants

---

## Key Dependencies

```
RMSNorm kernel ─────────→ 5-15% overall throughput gain
RoPE kernel ────────────→ 3-8% throughput gain
Paged Attention V2 ────→ Long context performance (32k+)
GPU sampling ───────────→ Large batch latency
Marlin GEMM ────────────→ GPTQ performance (2-3x)
CUTLASS Scaled MM ──────→ H100 FP8 performance
```
