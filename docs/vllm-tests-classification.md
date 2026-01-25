# vLLM Tests Classification

A comprehensive classification of vLLM tests for prioritizing implementation in vllm-rust.

## Summary

| Metric | Count |
|--------|-------|
| **Total test files** | ~703 |
| **Functional (priority)** | ~230 files |
| **Specific (low priority)** | ~470 files |

### Classification Criteria

- **Functional**: Core inference functionality required for any vLLM-compatible implementation
- **Specific**: Bug fixes, model-specific features, vLLM-internal optimizations, hardware-specific paths

---

## FUNCTIONAL TESTS (Priority for Implementation)

Tests that verify core LLM inference functionality. These should be implemented in vllm-rust.

### 1. Core Engine & Scheduling (~25 tests)

Core scheduling, request management, and engine lifecycle.

| File | Description | Priority |
|------|-------------|----------|
| `v1/core/test_scheduler.py` | Main scheduler logic, async scheduling | P0 |
| `v1/core/test_prefix_caching.py` | Prefix caching in scheduler | P0 |
| `v1/core/test_kv_cache_utils.py` | KV cache utilities | P0 |
| `v1/core/test_request_state.py` | Request state management | P0 |
| `v1/engine/test_engine_core.py` | Engine core lifecycle | P0 |
| `v1/engine/test_async_llm.py` | Async LLM interface | P0 |
| `v1/engine/test_llm_engine.py` | LLM engine interface | P0 |
| `v1/engine/test_output_processor.py` | Output processing | P1 |
| `engine/test_arg_utils.py` | Argument parsing utilities | P1 |

**Key test functions to implement:**
- `test_scheduler_basic` - Basic scheduling flow
- `test_scheduler_preemption` - Request preemption
- `test_scheduler_priority` - Priority-based scheduling
- `test_prefix_caching_hit` - Prefix cache hits
- `test_kv_cache_allocation` - KV cache block allocation

### 2. KV Cache & Memory Management (~15 tests)

KV cache allocation, eviction, and prefix caching.

| File | Description | Priority |
|------|-------------|----------|
| `v1/core/test_kv_cache_utils.py` | KV cache block management | P0 |
| `v1/core/test_prefix_caching.py` | Prefix caching logic | P0 |
| `v1/kv_connector/unit/test_*.py` | KV cache connector tests | P1 |

**Key test functions:**
- `test_kv_block_allocation` - Block allocation
- `test_kv_block_eviction` - LRU/FIFO eviction
- `test_prefix_caching_basic` - Basic prefix matching
- `test_prefix_caching_eviction` - Cache eviction with prefix

### 3. Attention Mechanisms (~20 tests)

PagedAttention, FlashAttention, and attention backends.

| File | Description | Priority |
|------|-------------|----------|
| `kernels/attention/test_paged_attention.py` | PagedAttention kernel | P0 |
| `kernels/attention/test_flash_attn.py` | FlashAttention integration | P0 |
| `kernels/attention/test_attention_selector.py` | Backend selection | P1 |
| `kernels/attention/test_prefix_prefill.py` | Prefix prefill attention | P1 |
| `v1/attention/test_basic_attn.py` | Basic attention tests | P0 |

**Key test functions:**
- `test_paged_attention_decode` - Decode phase attention
- `test_paged_attention_prefill` - Prefill phase attention
- `test_flash_attention_basic` - FlashAttention correctness
- `test_flash_attention_varlen` - Variable-length sequences

### 4. Core Kernels (~15 tests)

Fundamental CUDA kernels for LLM operations.

| File | Description | Priority |
|------|-------------|----------|
| `kernels/core/test_activation.py` | Activation functions (SiLU, GELU) | P0 |
| `kernels/core/test_rotary_embedding.py` | RoPE embeddings | P0 |
| `kernels/core/test_layernorm.py` | LayerNorm/RMSNorm | P0 |
| `kernels/core/test_pos_encoding.py` | Positional encodings | P1 |

**Key test functions:**
- `test_silu_and_mul` - SiLU activation with mul fusion
- `test_gelu_fast` - Fast GELU approximation
- `test_rotary_embedding` - RoPE correctness
- `test_rms_norm` - RMSNorm kernel
- `test_layer_norm` - Standard LayerNorm

### 5. Sampling (~15 tests)

Sampling strategies, logits processing, and token selection.

| File | Description | Priority |
|------|-------------|----------|
| `v1/sample/test_sampler.py` | Core sampler | P0 |
| `v1/sample/test_logits_processor.py` | Logits processing | P0 |
| `samplers/test_sampler.py` | Sampling strategies | P0 |
| `samplers/test_rejection_sampler.py` | Rejection sampling | P1 |
| `samplers/test_logprobs.py` | Log probabilities | P0 |
| `samplers/test_typical_acceptance_sampler.py` | Typical acceptance | P1 |

**Key test functions:**
- `test_greedy_sampling` - Greedy decoding
- `test_top_k_sampling` - Top-k sampling
- `test_top_p_sampling` - Top-p (nucleus) sampling
- `test_temperature_scaling` - Temperature adjustment
- `test_repetition_penalty` - Repetition penalty
- `test_logprobs_computation` - Log probability extraction

### 6. Distributed Inference (~30 tests)

Tensor parallelism, pipeline parallelism, and collective operations.

| File | Description | Priority |
|------|-------------|----------|
| `distributed/test_comm_ops.py` | Communication primitives | P0 |
| `distributed/test_pynccl.py` | NCCL bindings | P0 |
| `distributed/test_custom_all_reduce.py` | Custom all-reduce | P1 |
| `distributed/test_pipeline_parallel.py` | Pipeline parallelism | P0 |
| `distributed/test_sequence_parallel.py` | Sequence parallelism | P1 |
| `distributed/test_shm_broadcast.py` | Shared memory broadcast | P1 |
| `distributed/test_shm_buffer.py` | Shared memory buffers | P1 |
| `distributed/test_pipeline_partition.py` | Pipeline partitioning | P0 |
| `distributed/test_expert_parallel.py` | Expert parallelism | P1 |
| `v1/distributed/test_*.py` | V1 distributed tests | P0 |

**Key test functions:**
- `test_tensor_parallel_linear` - TP linear layers
- `test_all_reduce` - All-reduce collective
- `test_all_gather` - All-gather collective
- `test_pipeline_stage_forward` - PP stage execution
- `test_nccl_init` - NCCL initialization

### 7. Quantization Kernels (~20 tests)

FP8, INT8, and scaled matrix operations.

| File | Description | Priority |
|------|-------------|----------|
| `kernels/quantization/test_fp8.py` | FP8 operations | P0 |
| `kernels/quantization/test_int8.py` | INT8 operations | P0 |
| `kernels/quantization/test_scaled_mm.py` | Scaled matmul | P0 |
| `kernels/quantization/test_qqq.py` | QQQ quantization | P1 |
| `kernels/quantization/test_nvfp4.py` | NVFP4 format | P2 |
| `models/quantization/test_fp8.py` | FP8 model inference | P0 |

**Key test functions:**
- `test_fp8_gemm` - FP8 GEMM
- `test_int8_gemm` - INT8 GEMM
- `test_scaled_mm_fp8` - Scaled FP8 matmul
- `test_dynamic_quantization` - Dynamic quantization

### 8. LoRA (~28 tests)

Low-Rank Adaptation support.

| File | Description | Priority |
|------|-------------|----------|
| `lora/test_lora.py` | Core LoRA functionality | P0 |
| `lora/test_layers.py` | LoRA layer implementations | P0 |
| `lora/test_lora_manager.py` | LoRA adapter management | P0 |
| `lora/test_multi_lora.py` | Multiple adapters | P1 |
| `lora/test_lora_tp.py` | LoRA with tensor parallelism | P1 |
| `lora/test_long_context.py` | Long context LoRA | P2 |
| `lora/test_punica.py` | Punica kernels | P1 |

**Key test functions:**
- `test_lora_linear` - LoRA linear layers
- `test_lora_merge` - Adapter merging
- `test_multi_adapter_forward` - Multi-adapter batching
- `test_lora_tp_linear` - TP-compatible LoRA

### 9. Speculative Decoding (~10 tests)

Draft model and speculative decoding strategies.

| File | Description | Priority |
|------|-------------|----------|
| `v1/spec_decode/test_eagle.py` | EAGLE speculation | P0 |
| `v1/spec_decode/test_ngram.py` | N-gram speculation | P0 |
| `v1/spec_decode/test_mtp.py` | Multi-token prediction | P1 |
| `spec_decode/test_spec_decode_worker.py` | Speculation worker | P0 |
| `spec_decode/test_metrics.py` | Speculation metrics | P1 |

**Key test functions:**
- `test_draft_generation` - Draft token generation
- `test_verification` - Draft verification
- `test_acceptance_rate` - Acceptance rate tracking
- `test_eagle_tree_attention` - EAGLE tree attention

### 10. OpenAI-Compatible API (~20 tests)

API compatibility layer.

| File | Description | Priority |
|------|-------------|----------|
| `entrypoints/openai/test_chat.py` | Chat completions | P0 |
| `entrypoints/openai/test_completion.py` | Text completions | P0 |
| `entrypoints/openai/test_serving_chat.py` | Chat serving | P0 |
| `entrypoints/openai/test_serving_completion.py` | Completion serving | P0 |
| `entrypoints/openai/test_tokenization.py` | Tokenization endpoint | P1 |
| `entrypoints/openai/test_models.py` | Models endpoint | P1 |
| `entrypoints/llm/test_generate.py` | Direct generation | P0 |
| `entrypoints/llm/test_generate_multiple.py` | Batch generation | P0 |

**Key test functions:**
- `test_chat_completion` - Basic chat completion
- `test_chat_streaming` - Streaming responses
- `test_function_calling` - Tool/function calling
- `test_completion_basic` - Basic completion
- `test_completion_logprobs` - Completion with logprobs

### 11. Tokenization & Detokenization (~10 tests)

Tokenizer handling and stop conditions.

| File | Description | Priority |
|------|-------------|----------|
| `detokenizer/test_stop_strings.py` | Stop string detection | P0 |
| `detokenizer/test_stop_reason.py` | Stop reasons | P0 |
| `detokenizer/test_min_tokens.py` | Minimum tokens | P1 |
| `tokenizers_/test_tokenizers.py` | Tokenizer loading | P0 |

**Key test functions:**
- `test_stop_string_detection` - Stop string matching
- `test_eos_token_stop` - EOS token handling
- `test_incremental_detokenization` - Incremental decode

### 12. Compile & Optimization (~15 tests)

Graph compilation and kernel fusion.

| File | Description | Priority |
|------|-------------|----------|
| `compile/test_fusion.py` | Operator fusion | P1 |
| `compile/test_graph_partition.py` | Graph partitioning | P1 |
| `compile/test_pass_manager.py` | Compilation passes | P1 |
| `compile/fullgraph/test_basic_correctness.py` | Full graph compile | P1 |

**Key test functions:**
- `test_silu_mul_fusion` - SiLU+Mul fusion
- `test_attention_fusion` - Attention fusion
- `test_graph_capture` - CUDA graph capture

### 13. Multimodal Core (~12 tests)

Basic multimodal processing infrastructure.

| File | Description | Priority |
|------|-------------|----------|
| `multimodal/test_processing.py` | Input processing | P1 |
| `multimodal/test_inputs.py` | Multimodal inputs | P1 |
| `multimodal/test_utils.py` | MM utilities | P1 |

**Key test functions:**
- `test_image_processing` - Image preprocessing
- `test_multimodal_input_merge` - Input merging
- `test_placeholder_replacement` - Placeholder handling

---

## SPECIFIC TESTS (Low Priority)

Tests specific to vLLM implementation details, specific models, or hardware.

### Model-Specific Tests (~150+ tests)

Tests for specific model architectures. Implement as models are added.

| Category | Files | Notes |
|----------|-------|-------|
| `models/language/generation/` | ~50 | Specific model generation tests |
| `models/language/pooling/` | ~20 | Embedding model tests |
| `models/multimodal/generation/` | ~40 | VLM-specific tests |
| `models/multimodal/processing/` | ~20 | Model-specific processors |
| `models/quantization/` | ~20 | Model quantization variants |

### Tool Parsers (~25 tests)

Model-specific function calling parsers.

| File | Model |
|------|-------|
| `tool_parsers/test_llama_tool_parser.py` | Llama 3.1+ |
| `tool_parsers/test_mistral_tool_parser.py` | Mistral |
| `tool_parsers/test_hermes_tool_parser.py` | Hermes |
| `tool_parsers/test_qwen_tool_parser.py` | Qwen |
| `tool_parsers/test_internlm_tool_parser.py` | InternLM |
| `tool_parsers/test_jamba_tool_parser.py` | Jamba |
| `tool_parsers/test_granite_tool_parser.py` | Granite |

### Reasoning Parsers (~16 tests)

Model-specific reasoning format parsers.

| File | Model |
|------|-------|
| `reasoning/test_deepseek_r1.py` | DeepSeek R1 |
| `reasoning/test_granite.py` | Granite |
| `reasoning/test_qwen3.py` | Qwen3 |

### Hardware-Specific Tests (~30 tests)

Tests for specific hardware platforms.

| Category | Files |
|----------|-------|
| `rocm/` | ROCm/AMD GPU specific |
| `cuda/test_cuda_context.py` | CUDA context management |
| `kernels/helion/` | Helion-specific kernels |

### MoE Kernels (~31 tests)

Mixture of Experts specific kernels.

| File | Description |
|------|-------------|
| `kernels/moe/test_batched_moe.py` | Batched MoE |
| `kernels/moe/test_fused_moe.py` | Fused MoE kernels |
| `kernels/moe/test_deepep.py` | DeepSpeed EP |
| `kernels/moe/test_pplx_moe.py` | PPLX MoE |
| `distributed/test_eplb_*.py` | Expert load balancing |

### vLLM-Specific Optimizations (~50 tests)

Internal vLLM optimizations and infrastructure.

| Category | Files | Notes |
|----------|-------|-------|
| `benchmarks/` | ~10 | Benchmark infrastructure |
| `plugins/` | ~10 | Plugin system tests |
| `standalone_tests/` | Various | E2E standalone tests |
| `vllm_test_utils/` | Various | Test utilities |

### Backend-Specific Tests (~40 tests)

Tests for specific attention/kernel backends.

| File | Backend |
|------|---------|
| `kernels/attention/test_flashinfer.py` | FlashInfer |
| `kernels/attention/test_triton_decode.py` | Triton |
| `kernels/attention/test_xformers.py` | xFormers |
| `v1/attention/test_flashinfer_backend.py` | FlashInfer v1 |

### Configuration & Utilities (~50 tests)

Configuration parsing and utility functions.

| File | Description |
|------|-------------|
| `config/test_config_utils.py` | Config utilities |
| `config/test_model_arch_config.py` | Model config |
| `utils_/test_*.py` | Various utilities |
| `transformers_utils/test_*.py` | HF transformers utils |

### Error Handling & Edge Cases (~30 tests)

Bug fixes and regression tests.

| Category | Notes |
|----------|-------|
| `basic_correctness/` | Basic error cases |
| `v1/e2e/` | End-to-end edge cases |
| `v1/shutdown/` | Shutdown handling |

---

## Implementation Priority Matrix

### Phase 1: Core Inference (P0)
Essential for basic LLM inference.

1. Core kernels (activation, layernorm, RoPE)
2. PagedAttention
3. KV cache management
4. Basic scheduler
5. Greedy/sampling
6. Detokenization

### Phase 2: Production Features (P1)
Required for production deployment.

1. Tensor parallelism
2. Prefix caching
3. OpenAI-compatible API
4. FlashAttention integration
5. FP8 quantization

### Phase 3: Advanced Features (P2)
Nice-to-have features.

1. Speculative decoding
2. LoRA support
3. Pipeline parallelism
4. CUDA graph optimization
5. Multimodal support

### Phase 4: Model Coverage
Add as models are needed.

1. Model-specific tests
2. Tool parsers
3. Reasoning parsers

---

## Test Count Summary

| Category | Functional | Specific | Total |
|----------|------------|----------|-------|
| Core Engine | 25 | 10 | 35 |
| KV Cache | 15 | 5 | 20 |
| Attention | 20 | 20 | 40 |
| Core Kernels | 15 | 5 | 20 |
| Sampling | 15 | 5 | 20 |
| Distributed | 30 | 15 | 45 |
| Quantization | 20 | 30 | 50 |
| LoRA | 28 | 0 | 28 |
| Speculative | 10 | 5 | 15 |
| OpenAI API | 20 | 65 | 85 |
| Tokenization | 10 | 5 | 15 |
| Compile | 15 | 10 | 25 |
| Multimodal | 12 | 60 | 72 |
| Models | 0 | 150 | 150 |
| Tools/Utils | 0 | 80 | 80 |
| **Total** | **~235** | **~465** | **~700** |

---

## Notes

1. **Test counts are approximate** - vLLM's test suite is actively developed
2. **Functional tests** define the compatibility surface we need to match
3. **Specific tests** can be implemented incrementally as features are added
4. **Model tests** should be added as we support each model architecture
5. **Hardware-specific tests** (ROCm, etc.) are out of scope for initial implementation
