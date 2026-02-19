# Feature Gap Analysis: vLLM-Rust vs Python vLLM

**Date:** 2026-02-19 (updated)
**Reference vLLM commit:** `3025b3c` (2026-02-09)
**Rust model files:** 185 | **Python model files:** 240

## Summary

vLLM-Rust implements core text generation, sampling, KV cache, speculative decoding, LoRA, MoE, and a full OpenAI-compatible serving layer. The primary remaining gaps are: VLM/audio models (~50+), MTP framework (11 models), distributed inference (PP/DP/CP/EP), and advanced quantization methods.

Many Python model files that appear "missing" by filename are actually handled via architecture aliases in Rust (e.g., Orion/Solar/TeleChat → Llama, Mistral-Large-3 → DeepSeek, ERNIE-4.5 → Ernie45Moe, Bee → LLaVA-OneVision, OpenCUA → Qwen2.5-VL).

| Category | Python | Rust | Coverage |
|----------|--------|------|----------|
| Model architectures (unique) | ~160 | ~140 | ~88% |
| Quantization methods | 20+ | 10 | 50% |
| Attention backends (V1) | 20+ | 5 | 25% |
| Tool call parsers | 31 | 30 | **97%** |
| Reasoning parsers | 14 | 15 | **100%+** |
| RoPE variants | 17 | 11 | 65% |
| Server API routes | 15+ | 13 | 87% |
| MoE infrastructure | 40+ files | ~10 | 25% |
| SSM/Mamba ops | 16+ | 2 | 12% |
| Distributed | Full TP/PP/DP/CP/EP | TP only | 20% |
| MTP models | 11 | 0 | 0% |

---

## Current State (Implemented)

| Category | Status |
|----------|--------|
| **Core** | PagedAttention (parameterized head_dim/block_size), continuous batching, FCFS/priority scheduler, chunked prefill, preemption |
| **Engine** | ExecutionStrategy trait, standard + speculative decoding, CUDA graph capture/replay, warmup system |
| **Attention** | 5 backends: FlashInfer (decode/prefill/MLA), FlashAttention-2, Naive, MLA, Auto-select |
| **Speculative** | Draft model, n-gram, suffix, EAGLE-1/3, Medusa, MLP Speculator proposers. Tree attention verification |
| **KV Cache** | Block management, prefix caching, COW, eviction, FP8/INT8 quantized KV, CPU offload, MLA cache (NHD/HND layouts) |
| **Sampling** | Greedy, top-k, top-p, min-p, typical, temperature, repetition/frequency/presence penalty, logprobs, beam search, GPU sampling |
| **Constraints** | JSON schema, regex, choice, EBNF/GBNF grammar, structured_outputs API |
| **Quantization** | FP8, GPTQ (+Marlin), AWQ, BitsAndBytes (NF4/INT8), GGUF, Marlin, MxFP8, CompressedTensors, ExpertsInt8, MoE-WNA16 |
| **Models** | 185 files: ~140 architectures via aliases + quantized/LoRA variants |
| **LoRA** | Per-request adapter loading, 11 model families, multi-adapter serving |
| **MoE** | Fused MoE, expert parallelism, token dispatch, quantized experts, top-k softmax router |
| **Multimodal** | 13+ VLMs (LLaVA, Qwen2-VL/2.5-VL/3-VL, Gemma3-VLM, InternVL, Pixtral, etc.), CLIP + SigLIP + InternViT vision encoders |
| **Distributed** | NCCL communicator, process group, TP layers (28 architectures with `new_with_tp`), PP design (not enabled) |
| **Tool Calling** | 30 parser names → 28 unique parsers (Hermes, Mistral, DeepSeek v3/v3.1/v3.2, LLaMA, Qwen3Coder/XML, GLM4/4.7/MoE, Granite/20b-FC, HunYuan, ERNIE, etc.) |
| **Reasoning** | 15 parsers (DeepSeek-R1/V3, Qwen3, Mistral, Step3/3.5, ERNIE-4.5, Granite, Olmo3, SeedOSS, MiniMax-M2, HunYuan-A13B, Identity) |
| **SSM** | Mamba/Mamba2, Zamba2, Bamba state-space models |
| **API** | OpenAI: chat completions, completions, embeddings, responses, batch, models, tokenize/detokenize. Score, rerank, pooling, classify. LoRA management. SSE streaming |
| **Admin** | Prometheus metrics, profiling, sleep/wake, prefix cache reset, health, version, server info |
| **CUDA Kernels** | RMSNorm, RoPE, PagedAttnV2, GPU sampling, reshape_and_cache, FP8 GEMM, GPTQ dequant, fused MoE, top-k softmax |

### Models handled via architecture aliases (no separate file needed)

These Python model files have separate `.py` files but map to existing Rust implementations:

| Python File | Architecture | Maps To (Rust) |
|-------------|-------------|----------------|
| `olmo.py` | `OlmoForCausalLM` | `LlamaForCausalLM` |
| `orion.py` | `OrionForCausalLM` | `LlamaForCausalLM` |
| `solar.py` | `SolarForCausalLM` | `LlamaForCausalLM` |
| `telechat2.py` | `TeleChat2ForCausalLM` | `LlamaForCausalLM` |
| `teleflm.py` | `TeleFLMForCausalLM` | `LlamaForCausalLM` |
| `fairseq2_llama.py` | `Fairseq2LlamaForCausalLM` | `LlamaForCausalLM` |
| `stablelm.py` | `StableLMEpochForCausalLM` | `GPTNeoXForCausalLM` |
| `nemotron_nas.py` | `DeciLMForCausalLM` | `LlamaForCausalLM` |
| `commandr.py` | `CohereForCausalLM` | `CohereForCausalLM` |
| `ernie45.py` | `Ernie4_5ForCausalLM` | `Ernie45MoeForCausalLM` |
| `mistral_large_3.py` | `MistralLarge3ForCausalLM` | `DeepSeekForCausalLM` |
| `deepseek_v2.py` | `DeepseekV2ForCausalLM` | `DeepSeekForCausalLM` |
| `minimax_text_01.py` | `MiniMaxText01ForCausalLM` | `MiniMaxText01ForCausalLM` (already exists) |
| `glm4_moe_lite.py` | `Glm4MoeLiteForCausalLM` | `Glm4MoeForCausalLM` |
| `hunyuan_v1.py` | `HunYuanDenseV1/MoEV1ForCausalLM` | `hunyuan.rs` |
| `openpangu.py` | `PanguEmbedded/UltraMoE/ProMoEV2ForCausalLM` | `pangu.rs` |
| `granitemoehybrid.py` | `GraniteMoeHybridForCausalLM` | `granitemoe_hybrid.rs` |
| `granitemoeshared.py` | `GraniteMoeSharedForCausalLM` | `granitemoe_shared.rs` |
| `mimo.py` | `MiMoForCausalLM` | `MiMoV2FlashForCausalLM` |
| `lfm2_moe.py` | `Lfm2MoeForCausalLM` | `lfm2.rs` |
| `bee.py` | `BeeForConditionalGeneration` | `LlavaOnevisionForConditionalGeneration` |
| `opencua.py` | `OpenCUAForConditionalGeneration` | `Qwen2_5_VLForConditionalGeneration` |
| `qwen2_rm.py` | `Qwen2ForRewardModel` | `qwen2_reward.rs` |
| `clip.py`, `siglip.py` | Vision encoders | `multimodal/vision.rs` |
| `intern_vit.py` | InternViT | inline in `internvl.rs` |

---

## 1. Missing Models

### 1.1 Vision Encoders

| Model | Python File | Status |
|-------|------------|--------|
| CLIP | `clip.py` | **DONE** (`multimodal/vision.rs`) |
| SigLIP | `siglip.py` | **DONE** (`multimodal/vision.rs`) |
| InternViT | `intern_vit.py` | **DONE** (inline in `internvl.rs`) |
| SigLIP2-NaViT | `siglip2navit.py` | P1 — needed by Gemma3-MM, LFM2-VL |
| LFM2-SigLIP2 | `lfm2_siglip2.py` | P2 — LFM2-specific SigLIP2 variant |
| Swin | `swin.py` | P2 |
| MoonViT | `moonvit.py` | P2 |
| IDEFICS2 vision | `idefics2_vision_model.py` | P3 |
| Radio | `radio.py` | P3 |
| AIMv2 | `aimv2.py` | P3 |

### 1.2 Missing VLMs (~45 models)

**P1 (high-demand):**

| Model | Python File | Blocked By |
|-------|------------|------------|
| LLaVA-Next | `llava_next.py` | Dynamic resolution patch merging |
| LLaVA-Next-Video | `llava_next_video.py` | Dynamic resolution patch merging |
| Kimi-VL | `kimi_vl.py` | — |
| Kimi-K2.5-VL | `kimi_k25.py`, `kimi_k25_vit.py` | — |
| Gemma3-MM | `gemma3_mm.py` | SigLIP2-NaViT |
| Gemma3n-MM | `gemma3n_mm.py` | — |
| MLLaMA4 | `mllama4.py` | — |
| InternS1 / Pro | `interns1.py`, `interns1_pro.py`, `interns1_vit.py` | — (InternViT done) |
| ERNIE4.5-VL | `ernie45_vl.py`, `ernie45_vl_moe.py` | ernie45_vl_rope |
| Step3-VL | `step3_vl.py`, `step_vl.py` | — |
| Qwen2.5-Omni-Thinker | `qwen2_5_omni_thinker.py` | — |
| Qwen3-Omni-MoE-Thinker | `qwen3_omni_moe_thinker.py` | — |
| Cohere2-Vision | `cohere2_vision.py` | — |
| HyperClovaX-Vision | `hyperclovax_vision.py` | — |

**P2 (established):**

| Model | Python File |
|-------|------------|
| H2O-VL | `h2ovl.py` |
| Ovis / Ovis2.5 | `ovis.py`, `ovis2_5.py` |
| Isaac | `isaac.py` |
| Keye / Keye-VL1.5 | `keye.py`, `keye_vl1_5.py` |
| Kanana-V | `kanana_v.py` |
| SkyWork-R1V | `skyworkr1v.py` |
| Tarsier | `tarsier.py` |
| SmolVLM | `smolvlm.py` |
| MiniMax-VL-01 | `minimax_vl_01.py` |
| Nemotron-VL / Nano | `nemotron_vl.py`, `nano_nemotron_vl.py` |
| Hunyuan-Vision | `hunyuan_vision.py` |
| OpenPangu-VL | `openpangu_vl.py` |
| LFM2-VL | `lfm2_vl.py` |
| DeepSeek-OCR 1/2 | `deepseek_ocr.py`, `deepseek_ocr2.py` |
| GLM-OCR | `glm_ocr.py` |
| GLM4-1V | `glm4_1v.py` |
| Aria | `aria.py` |
| MiniCPM-O | `minicpmo.py` |

**P3 (niche):**
PaddleOCR-VL, Jina-VL, RVL, LightonOCR, Dots-OCR, MidashenGLM, BLIP v1, Qwen-VL (legacy)

### 1.3 Missing Audio Models (12+)

| Model | Python File | Priority |
|-------|------------|----------|
| Whisper | `whisper.py`, `whisper_causal.py`, `whisper_utils.py` | P1 |
| Qwen2-Audio | `qwen2_audio.py` | P1 |
| Qwen3-ASR | `qwen3_asr.py` | P1 |
| Ultravox | `ultravox.py` | P1 |
| Granite-Speech | `granite_speech.py` | P2 |
| FunAudioChat | `funaudiochat.py` | P2 |
| AudioFlamingo3 | `audioflamingo3.py` | P2 |
| GLM-ASR | `glmasr.py` | P2 |
| Gemma3n-Audio | `gemma3n_audio_utils.py` | P2 |
| Voxtral / Voxtral-Realtime | `voxtral.py`, `voxtral_realtime.py` | P2 |
| Phi4MM-Audio | `phi4mm_audio.py` | P2 |
| MusicFlamingo | `musicflamingo.py` | P3 |

### 1.4 Missing Text Models

Most Python text models already work via aliases (see table above). Genuinely missing:

| Model | Python File | Priority | Notes |
|-------|------------|----------|-------|
| Nemotron-Parse | `nemotron_parse.py` | P2 | Bart-based encoder-decoder (OCR) |
| DeepEncoder 1/2 | `deepencoder.py`, `deepencoder2.py` | P3 | Encoder models |
| TerraTorch | `terratorch.py` | P3 | Geospatial wrapper, requires `terratorch` lib |

### 1.5 Missing Embedding/Encoder Models

| Model | Python File | Priority |
|-------|------------|----------|
| RoBERTa | `roberta.py` | P1 |
| BERT-with-RoPE (GTE-New, Nomic, Snowflake) | `bert_with_rope.py` | P1 |

### 1.6 Missing Speculative Decode Draft Models

| Model | Python File | Priority | Status |
|-------|------------|----------|--------|
| Eagle3-Mistral-Large3 | `mistral_large_3_eagle.py` | P1 | **DONE** (`eagle3_mistral_large3.rs`) |
| Medusa (model file) | `medusa.py` | P1 | Proposer exists; model loader missing |
| DeepSeek-Eagle | `deepseek_eagle.py` | P1 | — |
| LLaMA-Eagle | `llama_eagle.py` | P1 | — |
| LLaMA-Eagle3 | `llama_eagle3.py` | P1 | — |
| LLaMA4-Eagle | `llama4_eagle.py` | P1 | — |
| MiniCPM-Eagle | `minicpm_eagle.py` | P2 | — |

### 1.7 Missing MTP (Multi-Token Prediction) Models

MTP is a critical gap — no framework or models exist in Rust yet.

| Model | Python File | Priority |
|-------|------------|----------|
| DeepSeek-MTP | `deepseek_mtp.py` | **P0** |
| ERNIE-MTP | `ernie_mtp.py` | P2 |
| Exaone-MoE-MTP | `exaone_moe_mtp.py` | P2 |
| GLM4-MoE-MTP / Lite | `glm4_moe_mtp.py`, `glm4_moe_lite_mtp.py` | P2 |
| Longcat-Flash-MTP | `longcat_flash_mtp.py` | P2 |
| MIMO-MTP | `mimo_mtp.py` | P2 |
| OpenPangu-MTP | `openpangu_mtp.py` | P2 |
| Qwen3-Next-MTP | `qwen3_next_mtp.py` | P2 |
| Step3p5-MTP | `step3p5_mtp.py` | P2 |
| GLM-OCR-MTP | `glm_ocr_mtp.py` | P3 |

---

## 2. Missing/Incomplete Features

### 2.1 Server API Endpoints

**Implemented:** chat completions, completions, embeddings, responses (+ get/cancel), batch (+ get/cancel/list), models, tokenize, detokenize, score, rerank, pooling, classify, LoRA management, admin (health, version, metrics, prometheus, profiling, sleep/wake, prefix cache)

**Missing:**

| Endpoint | Python Location | Priority |
|----------|----------------|----------|
| POST /v1/audio/speech-to-text | `speech_to_text/` | P2 |
| POST /v1/audio/translations | `translations/` | P2 |
| WebSocket /v1/realtime | `realtime/` | P2 |
| POST /v1/images/generations | `images/` | P3 |
| gRPC server | separate binary | P3 |

### 2.2 Quantization Methods

**Implemented (10):** AWQ, BitsAndBytes (NF4/INT8), CompressedTensors, ExpertsInt8, FP8 (+CUDA), GPTQ (+CUDA, +Marlin), Marlin (+CUDA), MoE-WNA16, MxFP8, GGUF

**Missing (14):**

| Method | Python File | Priority |
|--------|------------|----------|
| AWQ-Marlin | `awq_marlin.py` | P2 |
| FBGEMM-FP8 | `fbgemm_fp8.py` | P2 |
| PTPC-FP8 | `ptpc_fp8.py` | P2 |
| Input-Quant-FP8 | `input_quant_fp8.py` | P2 |
| ModelOpt | `modelopt.py` | P2 |
| MxFP4 | `mxfp4.py` | P2 |
| KV-Cache quant | `kv_cache.py` | P2 |
| AWQ-Triton | `awq_triton.py` | P3 |
| INC (Intel) | `inc.py` | P3 |
| TorchAO | `torchao.py` | P3 |
| Petit (NvFP4) | `petit.py` | P3 |
| CPU-WNA16 | `cpu_wna16.py` | P3 |
| QUARK | `quark/` (6 files) | P3 |
| FP-Quant | `fp_quant.py` | P3 |

### 2.3 Attention Backends (V1)

**Implemented:** Naive, FlashAttention-2, FlashInfer (decode/prefill/MLA), Auto-select

**Missing (relevant to CUDA):**

| Backend | Python File | Priority |
|---------|------------|----------|
| CPU attention | `cpu_attn.py` | P2 |
| Tree attention | `tree_attn.py` | P2 — for spec decode verification |
| Linear attention | `linear_attn.py` | P2 — for linear attention models |
| Mamba1/2 attention | `mamba1_attn.py`, `mamba2_attn.py` | P2 |
| Short conv attention | `short_conv_attn.py` | P2 |
| FlashAttn-DiffKV | `flash_attn_diffkv.py` | P2 |
| MLA variants | `mla/` (11 files: CUTLASS, Triton, FlashMLA, sparse) | P2 |
| FlexAttention | `flex_attention.py` | P3 |
| GDN attention | `gdn_attn.py` | P3 |
| Triton attention | `triton_attn.py` | P3 |
| ROCm variants | 3 files | N/A — AMD-specific |

### 2.4 Tool Call Parsers

**Implemented (30 names → 28 unique parsers):** hermes, glm4/glm45/glm47/glm4_moe, json/openai, llama/llama3_json/llama4_json, llama4_pythonic, mistral, deepseek_v3/v31/v32, internlm/internlm2, jamba, pythonic, olmo3, granite, granite-20b-fc, kimi_k2, phi4mini, longcat, xlam, gigachat3, functiongemma, hunyuan/hunyuan_a13b, ernie45, seed_oss, minimax, minimax_m2, step3, step3p5/qwen3_xml/qwen3coder

**Missing (1):**

| Parser | Python File | Priority |
|--------|------------|----------|
| Qwen-VL (legacy) | `qwen_vl` in registry | P3 |

### 2.5 Reasoning Parsers

**Implemented (15):** deepseek_r1, deepseek_v3, qwen3, mistral, step3, step3p5, ernie45, granite, olmo3, seed_oss, minimax_m2, minimax_m2_append_think, hunyuan_a13b, identity

**Missing (1):**

| Parser | Python File | Priority |
|--------|------------|----------|
| GPT-OSS | `gptoss_reasoning_parser.py` | P2 |

### 2.6 RoPE Variants

**Implemented:** Standard (NeoX), linear scaling, dynamic NTK, YaRN, Llama3, Phi3 Long, MRoPE, DeepSeek scaling, dual chunk (partial)

**Missing (6):**

| Variant | Python File | Priority |
|---------|------------|----------|
| MRoPE Interleaved | `mrope_interleaved.py` | P2 |
| FoPE | `fope.py` | P2 |
| XDRoPE | `xdrope.py` | P2 |
| ERNIE4.5-VL RoPE | `ernie45_vl_rope.py` | P2 — blocks ERNIE4.5-VL |
| LLaMA4 Vision RoPE | `llama4_vision_rope.py` | P2 — blocks MLLaMA4 |
| NTK scaling (separate) | `ntk_scaling_rope.py` | P3 |

### 2.7 MoE Infrastructure

**Implemented:** Fused MoE kernel, token dispatch, top-k softmax router, quantized experts, shared experts

**Missing:**

| Component | Python Location | Priority |
|-----------|----------------|----------|
| Deep GEMM MoE | `fused_moe/deep_gemm_moe.py` | P2 |
| CUTLASS MoE | `fused_moe/cutlass_moe.py` | P2 |
| Batched MoE | `fused_moe/fused_batched_moe.py` | P2 |
| Fused Marlin MoE | `fused_moe/fused_marlin_moe.py` | P2 |
| Shared fused MoE | `fused_moe/shared_fused_moe.py` | P2 |
| Zero expert optimization | `fused_moe/zero_expert_fused_moe.py` | P2 |
| EPLB (expert load balancing) | `distributed/eplb/` | P2 |
| MoE router variants (7) | `fused_moe/router/` | P2 |
| All2All backends (7) | pplx, deepep, mori, flashinfer, etc. | P2 |

### 2.8 SSM/Mamba Infrastructure

**Implemented:** Mamba1/2 models, Zamba2, Bamba

**Missing ops:**

| Component | Python Location | Priority |
|-----------|----------------|----------|
| Mamba mixer v2 | `mamba/mamba_mixer2.py` | P2 |
| Causal conv1d | `mamba/ops/causal_conv1d.py` | P2 |
| SSD operations (6) | `mamba/ops/ssd_*.py` | P2 |
| Gated LayerNorm | `mamba/ops/layernorm_gated.py` | P2 |
| ShortConv | `mamba/short_conv.py` | P2 |
| Flash Linear Attention (FLA) | `fla/ops/` (16 files) | P3 |

---

## 3. Distributed Inference

### Current State
- `crates/core/src/distributed/` — ProcessGroup, NCCL communicator, launcher
- `tp_layers.rs` — TpLinear, TpEmbedding, TpGeGluMlp, TpSwiGluMlp
- `from_config_with_tp()` — 28 architectures (Llama, Mistral, Mixtral, Qwen2/3/MoE, GLM4/MoE, Gemma/2/3, Phi3, OLMo2, Baichuan, InternLM2, Cohere, GPTNeoX, StarCoder2, Bloom, Falcon, Phi, Yi, GPT2, Exaone, Persimmon, MPT, Dbrx, SeedOss, ERNIE45, HunYuan, Pangu, Lfm2, MiMo)
- `crates/core/src/engine/pipeline.rs` — PipelineForward trait, worker loop
- **PP returns error if world_size > 1.** TP verified for above architectures.

### Missing Components

| Component | Python Location | Priority |
|-----------|----------------|----------|
| Pipeline Parallelism (enable) | `distributed/parallel_state.py` | P1 |
| Communication ops (allreduce etc.) | `distributed/communication_op.py` | P1 |
| Data Parallelism | V1 executors | P2 |
| Context Parallelism | config + attention | P2 |
| Expert Parallelism | `distributed/eplb/` | P2 |
| KV transfer (disaggregated) | `distributed/kv_transfer/` | P3 |
| Weight distribution | `distributed/weight_transfer/` | P2 |
| Ray executor | `v1/executor/ray_executor.py` | P3 |

---

## 4. Known TODOs in Existing Code

| File | TODO | Priority |
|------|------|----------|
| `multimodal/processor.rs:146` | Video support not yet implemented | P1 |
| `phi4mm.rs:271` | HD transform + 2x2 compression + projection | P1 |
| `molmo2.rs:586` | Extract vision features for multimodal | P1 |
| `compressed_tensors.rs:485` | Use INT8 GEMM CUDA kernel | P2 |
| `gritlm.rs:151` | Expose Llama internal hidden states | P2 |
| `kimi_linear.rs:413` | KDA (linear attention kernels) | P2 |
| `opt.rs:503` | Heterogeneous offsets in batch | P2 |
| `mimo_v2_flash.rs:469` | MoE block instead of dense MLP | P2 |
| `chameleon.rs:501` | VQVAE image tokenization | P2 |
| `lfm2.rs:843` | ShortConv layer support | P2 |
| `longcat_flash.rs:297` | Full MoE with ZeroExpert routing | P2 |

---

## 5. Infrastructure & Config Gaps

### Configuration

| Config | Python File | Priority |
|--------|------------|----------|
| ParallelConfig (extended) | `parallel.py` | P1 |
| MultimodalConfig (extended) | `multimodal.py` | P2 |
| SpeechToTextConfig | `speech_to_text.py` | P2 |
| CompilationConfig | `compilation.py` | P3 |
| KVTransferConfig | `kv_transfer.py` | P3 |
| ProfilerConfig | `profiler.py` | P3 |
| ObservabilityConfig | `observability.py` | P3 |

### CUDA Graph Support
- Rust has `cuda_graph.rs` / `cuda_graph_runner.rs` — needs end-to-end verification
- **Priority: P1** (significant decode performance impact)

### Multi-Token Prediction (MTP)
- Python: 11 MTP model variants (DeepSeek-MTP critical)
- Rust: none implemented — framework + models both needed
- **Priority: P0**

---

## 6. Recommended Roadmap

### Phase 0: Critical Foundation (completed items struck through)
1. ~~CLIP vision encoder~~ — **DONE** (`multimodal/vision.rs`)
2. ~~SigLIP vision encoder~~ — **DONE** (`multimodal/vision.rs`)
3. ~~InternViT~~ — **DONE** (inline in `internvl.rs`)
4. ~~MRoPE~~ — **DONE** (`qwen2_vl.rs`)
5. ~~DeepSeek scaling RoPE~~ — **DONE** (`deepseek.rs`)
6. ~~Eagle3-Mistral-Large3~~ — **DONE** (`eagle3_mistral_large3.rs`)
7. ~~Tool parsers: OpenAI, Qwen3-XML, Step3p5, HunYuan, GLM4-MoE, Granite-20b-FC~~ — **DONE**
8. ~~Responses API~~ — **DONE** (`responses.rs`)

### Phase 1: MTP + High-Impact Models
1. **MTP framework** — proposer trait + verification loop in `speculative/`
2. **DeepSeek-MTP** model — most demanded MTP variant
3. **RoBERTa + BERT-with-RoPE** — widely used embeddings
4. **SigLIP2-NaViT** vision encoder → unblocks Gemma3-MM + other VLMs
5. **GPT-OSS reasoning parser** — single missing parser
6. CUDA graphs: end-to-end verification
7. Resolve P1 TODOs: phi4mm, molmo2, processor video

### Phase 2: VLM & Audio Expansion
1. Whisper + audio pipeline → unlocks Qwen2-Audio, Ultravox, Qwen3-ASR
2. P1 VLMs: LLaVA-Next, Kimi-VL, Gemma3-MM, MLLaMA4, InternS1, ERNIE4.5-VL, Step3-VL
3. RoPE: ernie45_vl, llama4_vision, mrope_interleaved, fope, xdrope
4. Spec decode models: DeepSeek-Eagle, LLaMA-Eagle/3, Medusa loader

### Phase 3: Infrastructure Hardening
1. Pipeline Parallelism — enable existing code, add tests
2. Quantization: AWQ-Marlin, FBGEMM-FP8, ModelOpt, MxFP4
3. Attention: CPU, tree, linear, Mamba1/2
4. MoE: router variants, batched MoE, Deep GEMM, EPLB
5. Server: speech-to-text, translations, realtime WebSocket

### Phase 4: Long Tail & Full Distributed
1. Data Parallelism + Context Parallelism
2. Expert Parallelism with load balancing
3. Remaining P2/P3 VLMs + audio models
4. Remaining MTP models (10)
5. All remaining quantization methods
6. KV transfer / disaggregated serving
7. gRPC server
8. Flash Linear Attention (FLA) ops
9. SSM ops: SSD, causal conv1d, gated layernorm
