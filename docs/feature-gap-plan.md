# Full Feature Parity Roadmap: vLLM-Rust

## Context

Bring vLLM-Rust to complete feature parity with Python vLLM `3025b3c`. Tasks ordered from hardest (new infrastructure) to easiest (incremental additions). Each tier builds on the previous.

---

## Tier 1: New Infrastructure (Hardest — Architectural)

### 1.1 Context Parallelism (CP) ✅ DONE
**Difficulty:** ★★★★★ | **Effort:** 4–6 weeks | **Status:** COMPLETE — commit `64ca787`

**Completed (2026-02-19):**
- `crates/core/src/distributed/context_parallel.rs` — `CpContext`, `CpConfig`, interleaved round-robin KV sharding ✅
- `get_dcp_local_seq_lens` / `get_dcp_local_slot_mapping` — slot filtering helpers ✅
- `lse_correct_and_reduce` — LSE-based attention output merging across CP ranks ✅
- `DcpAttentionWrapper` — transparent AttentionBackend wrapper for DCP ✅
- `AttentionBackend::batched_decode_attention_with_lse` — default trait method ✅
- `NaiveAttentionBackend` + FlashInfer: LSE output path ✅
- `decode_context_parallel_size` in `ParallelConfig` ✅
- 19 unit tests ✅

### 1.2 Data Parallelism (DP) ✅ DONE
**Difficulty:** ★★★★☆ | **Effort:** 2–3 weeks | **Status:** COMPLETE — commit `be29ab5`

**Completed (2026-02-19):**
- `crates/core/src/distributed/data_parallel.rs` — `DpContext`, `BatchCoordinationResult` ✅
- `coordinate_batch_across_dp()` — all_reduce max(num_tokens) before each forward pass ✅
- `request_belongs_to_rank()` — deterministic modulo routing for server layer ✅
- `data_parallel_size` in `ParallelConfig`; `world_size()` stays tp×pp ✅
- `EngineConfig::dp_context` + builder method ✅
- `run_engine_loop`: batch coordination call in `spawn_blocking` ✅
- 15 unit tests ✅

### 1.3 Expert Parallelism (EP) ✅ DONE
**Difficulty:** ★★★★☆ | **Effort:** 2–3 weeks | **Status:** COMPLETE

**Already implemented (prior to this PR):**
- `EPContext` (rank + ep_size) in `process_group.rs` — value type for model APIs ✅
- `ExpertPlacement` + `ExpertMap` in `moe/expert_map.rs` — linear/round-robin placement ✅
- `EPMoEConfig` + `EPMoELayer` in `moe/ep_layer.rs` — full EP MoE forward pass ✅
- `TokenDispatcher` + `DispatchMetadata` in `moe/token_dispatch.rs` — all-to-all dispatch ✅
- `Mixtral::new_with_ep()`, `Qwen2Moe::new_with_ep()` — model-level EP wiring ✅
- `expert_parallel_size` in `ParallelConfig` ✅

**Completed (this PR):**
- `crates/core/src/distributed/expert_parallel.rs` — `ExpertParallelContext` (EP coordinator with communicator) ✅
- `crates/core/src/moe/eplb.rs` — `EplbState`, load tracking, rebalance detection ✅
- `EngineConfig::ep_context` + builder method ✅
- 22 new unit tests ✅

**Out of scope:** Weight rearrangement (`rearrange_expert_weights_inplace`) — requires
NCCL weight migration; tracked as TODO in `moe/eplb.rs`.

### 1.4 MTP Framework + Models ✅ DONE
**Difficulty:** ★★★★☆ | **Effort:** 4–5 weeks | **Status:** FULLY COMPLETE — commit `dc0be81`

**Completed (2026-02-19):**
- `crates/core/src/models/mtp_base.rs` — `MtpDraftModel` trait ✅
- `crates/core/src/engine/spec_decode/mtp_proposer.rs` — `MtpProposer: DraftProposer` ✅
- `crates/core/src/models/deepseek_mtp.rs` — `DeepSeekMtpModel: MtpDraftModel` ✅ (9 tests)
- All 9 remaining MTP model files + `mtp_from_config()` ✅
  (ernie_mtp, mimo_mtp, longcat_flash_mtp, glm4_moe_mtp, openpangu_mtp, step3p5_mtp,
   glm_ocr_mtp, qwen3_next_mtp, exaone_moe_mtp)

**Architecture:** MTP uses FIXED target hidden states for ALL K draft steps. Each step:
`enorm(embed(token)) + hnorm(target_hs)` → `eh_proj` → `mtp_block` → `shared_head` → logits.
Layers cycle via `spec_step_idx % num_mtp_layers`.
Pattern B (Qwen3Next): `pre_fc_norm + fc + mtp_block + norm + shared lm_head`.

### 1.5 Audio Pipeline + Models
**Difficulty:** ★★★★☆ | **Effort:** 3–4 weeks
- `crates/core/src/multimodal/audio.rs` has `AudioData`/`AudioSpec` but NO encoders, NO mel spectrogram, NO file loading
- Need librosa-equivalent Rust: FFT, mel filterbank, STFT → ~150 LOC DSP
- Add WAV/MP3 loading (hound + symphonia crates)
- **Encoder models (12 total):**
  - `whisper.rs` — P1, gate-keeper for all audio; Conv1D + transformer, ~800 LOC
  - `qwen2_audio.rs`, `ultravox.rs` — P1
  - `qwen3_asr.rs`, `granite_speech.rs`, `funaudiochat.rs` — P2
  - `audioflamingo3.rs`, `glmasr.rs`, `voxtral.rs`, `phi4mm_audio.rs` — P2
  - `gemma3n_audio.rs`, `musicflamingo.rs` — P3
- **Files:** extend `multimodal/audio.rs`, new `multimodal/audio_encoder.rs`, per-model files

---

## Tier 2: Model Expansion (Hard — High Volume)

### 2.1 Missing VLMs (~20 models) ✅ InternS1, Step3-VL, ERNIE4.5-VL, InternS1Pro, MoonViT+Kimi-VL, Kimi-K2.5, HyperCLOVA-X, GLM-OCR, GLM4-1V, Kanana-V, Ovis, Aria, OpenPangu-VL DONE
**Difficulty:** ★★★☆☆ per model | **Effort:** 4–6 weeks total
- All follow Vision→Projector→LM pattern; 150–300 LOC per model
- **Blockers to resolve first (new vision encoders):**
  - `MoonViT` ✅ — `moonvit.rs` done; also unblocks Kimi-K2.5-VL
  - `SigLIP2-NaViT` — blocks LFM2-VL (~250 LOC, `multimodal/vision.rs`)

**Completed:**
- `interns1.rs` — `InternS1ForConditionalGeneration` ✅ (22 tests) — 2026-02-19
- `step3_vl.rs` — `Step3VLForConditionalGeneration` ✅ (5 tests) — commit 809b3ba
- `ernie45_vl.rs` — `Ernie4_5_VLForConditionalGeneration` ✅ (5 tests) — commit b74d083
- `interns1_pro.rs` — `InternS1ProForConditionalGeneration` ✅ (5 tests) — commit a3e5cd3
- `moonvit.rs` + `kimi_vl.rs` — `MoonVitPretrainedModel` + `KimiVLForConditionalGeneration` ✅ (10 tests) — commit 463a5a5
- `kimi_k25.rs` — `KimiK25ForConditionalGeneration` ✅ (5 tests)
- `hyperclovax_vision.rs` — `HCXVisionForCausalLM` ✅ (5 tests) — commit 767ebec
- `glm_ocr.rs` — `GlmOcrForConditionalGeneration` ✅ (5 tests)
- `glm4_1v.rs` — `Glm4vForConditionalGeneration` ✅ (5 tests)
- `kanana_v.rs` — `KananaVForConditionalGeneration` ✅ (5 tests)
- `ovis.rs` — `OvisForConditionalGeneration` ✅ (6 tests)
- `aria.rs` — `AriaForConditionalGeneration` ✅ (5 tests) — commit 3f285e8
- `openpangu_vl.rs` — `OpenPanguVLForConditionalGeneration` ✅ (5 tests) — commit 8e77a55

**Architecture:** InternS1ViT (separate Q/K/V, `layernorm_before/after`, `encoder.layer.{i}` singular) +
InternS1-specific pixel shuffle + `multi_modal_projector` (LN+Linear+GELU+Linear) + InternLM2 LLM.
Weight mapping: `model.vision_tower.*` → `vision_tower.*`, `model.multi_modal_projector.*` → `multi_modal_projector.*`.

Step3-VL: Step3VisionTransformer (63-layer ViT, QuickGELU) + Conv2d downsamplers + Linear projector
+ Step3TextForCausalLM. TP-padding: 3 extra CLS tokens prepended → 2708 total, drop first 4 → 2704 patches.

ERNIE4.5-VL: `Ernie4_5_VisionTransformer` (Linear patch embed, 2D spatial RoPE, QuickGELU) +
`VariableResolutionResamplerModel` (spatial_conv_size² pooling, 2-layer GELU MLP, RMSNorm) + Ernie45MoeForCausalLM.
Weight paths: `vision_model.*` (root), `model.resampler_model.spatial_linear.{0,2,3}.*`, `model.*` / `lm_head.*`.

MoonViT+KimiVL: `MoonVitPretrainedModel` (Conv2d patch embed + `Learnable2DInterpPosEmb` + 2D RoPE interleaved x/y
complex multiply + fused-QKV attention + bilinear pos-emb interpolation + `patch_merger`) +
`KimiVLMultiModalProjector` (pre_norm LN + flatten + linear_1 + GELU + linear_2) + DeepSeekForCausalLM.
Weight paths: `vision_tower.*`, `multi_modal_projector.*`, `language_model.*`.

Kimi-K2.5: `MoonViT3dPretrainedModel` (3D patch embed with temporal sincos pos emb; `Rope2DPosEmbRepeated`
repeats spatial freqs T times; `tpool_patch_merger` mean-pools temporal dim then spatial groups) +
`KimiK25MultiModalProjector` (pre_norm + flatten + linear_1 GELU + linear_2, weight `mm_projector.*`) +
DeepSeekForCausalLM. Reuses `MoonVitEncoderLayer` from moonvit.rs (pub(crate)).

HyperCLOVA-X Vision: CLIP/SigLIP vision encoder + `HCXCAbstractor` (timm RegNet bottleneck stages with
`LayerNorm2d` channel-norm, SE attention, adaptive avg pool) or MLP/Linear projector + LLaMA LLM.
`LayerNorm2d`: permute `[B,C,H,W]→[B,H,W,C]`, normalize last dim, permute back.
Weight paths: `vision_model.*`, `mm_projector.net.{0,2}.*`, `mm_projector.readout.{0,2}.*`.
Projector types: `linear`, `mlp`, `inverted_mlp`, `cabstractor` (default). `skip_cls=true` for CLIP (not SigLIP).

GLM-OCR: Linear patch embed (Conv3D equiv) + 2D block-tiled spatial RoPE (partial_factor=0.5, neox, sm²-grouped) +
per-head q_norm/k_norm (RMSNorm) + SwiGLU MLP (bias=True) + Conv2d downsample + PatchMerger (proj→LayerNorm→
GELU→SwiGLU, no bias) + Glm4ForCausalLM. `out_hidden_size` must match LLM `hidden_size`.
Weight paths: `visual.*` (vision), `model.*` / `lm_head.*` (LLM).

GLM-4.1V: Same block-tiled RoPE + PatchMerger as GLM-OCR but: no per-head q/k norms; qkv+proj bias=False;
MLP `hidden_dim = out_hidden_size` (not intermediate_size), bias=False; adds `post_conv_layernorm` (RmsNorm) +
`Glm4vVisionEmbeddings` (learnable 2D pos emb via bilinear interp at actual patch coords).
Merger `context_dim = intermediate_size`. Backbone: `Glm4ForCausalLM` (glm4v) or `Glm4MoeForCausalLM` (glm4v_moe).
Bilinear pixel coords: `src = (coord + 0.5) * orig_size / target - 0.5` (align_corners=False → border clamp).
Weight paths: `visual.*` (vision), `model.*` / `lm_head.*` (LLM).

Kanana-V: `Qwen2VLVisionEncoder` (Qwen2ViT without merger; new `pub(crate)` struct in qwen2_vl.rs) +
`DynamicCAbstractor` (bilinear pos-emb resampling + two timm `RegStage`s with `LayerNorm2d` channel-norm +
`PatchMerge` + `ReadoutMLP`) + `LlamaForCausalLM`. Requires `depth ≥ 1` (depth=0 has ambiguous dim layout).
`RegBottleneck`: conv1(1×1,act) → conv2(3×3,act) → conv3(1×1) + skip(1×1 if in≠out) → SiLU.
Weight paths: `vision_model.*`, `abstractor.net.{0,2}.blocks.block{k}.*`, `abstractor.readout.*`,
`model.model.*` / `model.lm_head.*`. `Qwen2VLVisionConfig::from_model_config()` added (reads `extra["vision_config"]`).

Ovis: `AIMv2Model` (Conv2d patch embed + RMSNorm + learnable pos_embed; depth SwiGLU pre-norm blocks; no CLS token) +
`VisualTokenizer` (AIMv2 → optional hidden_stride² spatial merge → Linear(no-bias) + LayerNorm head → softmax/st_argmax
→ pad to vocab_size) + `VisualEmbedding` (soft_tokens @ vte.weight; float path: matmul, int path: table lookup) +
LLM (LlamaForCausalLM or Qwen2ForCausalLM, enum `OvisLlm`).
Weight paths: `visual_tokenizer.backbone.*` (AIMv2), `visual_tokenizer.head.{0,1}.*`, `vte.weight`, `llm.model.*` / `llm.lm_head.*`.
Image pad token: qwen2→151655, llama→128002, gemma2→7.
AIMv2 fc1/fc3 loaded separately (not merged fc13); `qkv_bias`/`use_bias` from config.
`st_argmax`/`gumbel_argmax` → argmax + one-hot via broadcast_eq at inference (no noise).

OpenPangu-VL: Qwen2.5-VL-adapted ViT with window attention (get_window_index/build_window_mask) +
multiple intermediate mergers (`OpPatchMerger` at `mm_unit_vision_select_layer` indices, summed) +
`vision_projection` (linear_no_bias, under `vision_projection.linear`) + `PanguEmbeddedForCausalLM`.
Key difference vs Qwen2.5-VL: N mergers applied to N intermediate hidden states (with shared `final_layernorm`);
results summed rather than single merger at end. VL methods added to `PanguEmbeddedForCausalLM` (pangu.rs).
Weight paths: `visual.*` (ViT), `language_model.*` (LLM). `take_indices` = reversed `[depth+sl for sl in select_layer]`.

Aria: SigLIP ViT (no post_layernorm; `forward_no_post_norm`) + `AriaProjector` (learnable queries
[max_q, vis_hidden] + `AriaCrossAttention` + MLP) + `AriaTextModel` (LlamaAttention + per-layer MoE).
`AriaCrossAttention`: loads PyTorch MHA `in_proj_weight [3H, H]` split into q/k/v at load time.
`AriaTextMoELayer`: shared `TpSwiGluMlp` always active + top-K routed sparse experts using HF weight
format [E, H, 2I] (fc1) and [E, I, H] (fc2) — direct matmul `x @ fc[e]` (no transpose needed).
Weight paths: `vision_tower.*`, `multi_modal_projector.*`, `language_model.model.*` / `language_model.lm_head.*`.
`LlamaAttention` made `pub(crate)` to allow reuse from `AriaDecoderLayer`.

- **P1 models remaining (in order):**
  1. `qwen2_5_omni_thinker.rs` (after audio), `qwen3_omni_moe_thinker.rs` (after audio)
- **P2 models (~8 remaining):** MiniCPM-O, MiniMax-VL-01 (BLOCKED: Lightning Attention), Nemotron-VL (BLOCKED: dynamic AutoModel), DeepSeek-OCR, Hunyuan-Vision (BLOCKED), LFM2-VL (BLOCKED: SigLIP2-NaViT), Keye, Isaac
- **Pattern:** `crates/core/src/models/{name}.rs`, register in `mod.rs`, add alias if needed

### 2.2 MoE Infrastructure: Advanced
**Difficulty:** ★★★☆☆ | **Effort:** 3–4 weeks
- **DeepGEMM** — batched GEMM for dynamic expert shapes, 10–20% speedup
  - `crates/core/src/moe/deep_gemm.rs`, CUDA kernel wrapper
- **Router variants** — `crates/core/src/moe/router.rs` currently only `TopKRouter`
  - Add: grouped-topk (partial), renormalized-topk, expert-choice routing
- **EPLB** — requires EP infrastructure (§1.3 prerequisite)
  - `crates/core/src/moe/eplb.rs` — expert load balancing across ranks
- **Batched MoE** — for different token counts per expert
- **Reference:** `reference/vllm/model_executor/layers/fused_moe/`

### 2.3 Quantization P2 Methods
**Difficulty:** ★★★☆☆ per method | **Effort:** 3–4 weeks (6 methods)
- All follow `QuantizationConfig` + `QuantizedLinear` trait in `crates/core/src/quantization/`
- **In order of priority:**
  1. `awq_marlin.rs` — route AWQ weights to Marlin kernel; ~150 LOC, quick win
  2. `fbgemm_fp8.rs` — Meta's FP8 GEMM wrapper; ~300 LOC
  3. `input_quant_fp8.rs` — FP8 activation quantization; ~200 LOC
  4. `ptpc_fp8.rs` — per-tensor/per-channel FP8; ~250 LOC
  5. `mxfp4.rs` — MX floating-point 4-bit; ~400 LOC (new format)
  6. `modelopt.rs` — NVIDIA ModelOpt format; ~500 LOC
- **Reference:** `reference/vllm/model_executor/layers/quantization/`

### 2.4 Speculative Decode: Missing Eagle Variants
**Difficulty:** ★★★☆☆ | **Effort:** 2–3 weeks
- Pattern fully established: `eagle_llama.rs` (Eagle-1) and `eagle3.rs` (Eagle-3)
- **Files to create:**
  1. `deepseek_eagle.rs` — Eagle-1 for DeepSeek; ~600 LOC, adapt `eagle_llama.rs`
  2. `llama4_eagle.rs` — Eagle for LLaMA4; ~700 LOC
  3. `minicpm_eagle.rs` — Eagle for MiniCPM; ~500 LOC
  4. Medusa model loader — add `MedusaModel` match arm in `from_config()`; ~100 LOC
- **Proposers:** `crates/core/src/engine/spec_decode/`
- **Reference:** `reference/vllm/model_executor/models/deepseek_eagle.py`

### 2.5 SSM/Mamba2 Ops
**Difficulty:** ★★★☆☆ | **Effort:** 2–3 weeks
- **`mamba_mixer2`** — Mamba2's updated selective scan with state dimension splitting
  - `crates/core/src/ssm/mamba_mixer2.rs` (~350 LOC CPU; +600 LOC CUDA kernel)
- **Causal conv1d refactor** — currently duplicated inline in 8+ model files
  - Create `crates/core/src/ssm/causal_conv1d.rs` (~100 LOC); remove ~500 LOC of duplication
- **SSD ops** (6 Python files) — simplified state-space ops; required by future Mamba2 variants
  - `crates/core/src/ssm/ssd.rs` (~300 LOC)
- **Gated LayerNorm** — `crates/core/src/ssm/gated_layer_norm.rs` (~80 LOC)
- **ShortConv** — fix `lfm2.rs:843` TODO; `crates/core/src/ssm/short_conv.rs` (~150 LOC)

---

## Tier 3: Feature Completion (Medium)

### 3.1 Quantization P3 Methods
**Effort:** 4–6 weeks (8 methods) | Same trait pattern as P2
- AWQ-Triton, TorchAO, Intel INC, Petit (NvFP4), CPU-WNA16, QUARK (6 files), FP-Quant
- All in `crates/core/src/quantization/`

### 3.2 Server API — Speech & Realtime
**Effort:** 2–3 weeks
- `POST /v1/audio/transcriptions` + `POST /v1/audio/translations` — ~250 LOC handler
  - Multipart form upload → decode audio → audio encoder → text response
  - **Requires:** Audio pipeline (§1.5) complete
  - `crates/server/src/api/audio.rs` (new file)
- `WebSocket /v1/realtime` — ~500 LOC
  - Persistent connection, delta-streaming, session state machine
  - `crates/server/src/api/realtime.rs` (new file)
  - Add `tokio-tungstenite` dependency
- `POST /v1/images/generations` — P3, requires diffusion models

### 3.3 Attention Backend Variants
**Effort:** 2–3 weeks
- **CPU attention** — naive fallback for non-GPU inference; extend `naive.rs`
- **Linear attention backend** — for Kimi-Linear, Qwen3-Next GDN layers; new backend file
- **Mamba1/2 attention backends** — custom selective scan backend variants
- **FlashAttn-DiffKV** — differential KV (some research models)
- MLA variants (CUTLASS, sparse) — further optimization of existing MLA backend

---

## Tier 4: Easy Wins (Low Complexity, High Value)

### 4.1 Pipeline Parallelism — Enable
**Effort:** 2–3 days | **Difficulty:** ★★☆☆☆
- Framework is 90% designed: `PipelineForward` trait, `PipelineStagedModel`, P2P comms exist
- Wire `DeviceCommunicator.send()/recv()` calls in `crates/core/src/engine/pipeline.rs`
- Remove error gate in `crates/server/src/main.rs:815`
- Add 2-GPU integration test
- **Reference:** `crates/core/src/distributed/pipeline.rs` + `communicator.rs`

### 4.2 RoPE Variants
**Effort:** 1 week | All in `crates/core/src/layers/rotary.rs`
- **MRoPE Interleaved** — `mrope_interleaved.py`; ~200 LOC; interleaved frequency layout
- **FoPE** — `fope.py`; ~200 LOC; factored positional encoding
- **XDRoPE** — `xdrope.py`; ~250 LOC; cross-dimensional RoPE
- **ERNIE4.5-VL RoPE** — `ernie45_vl_rope.py`; ~150 LOC; unblocks ERNIE4.5-VL

### 4.3 Resolve P1 Code TODOs
**Effort:** 3–5 days
1. `phi4mm.rs:271` — HD transform + 2x2 compression + projection (~150 LOC)
2. `molmo2.rs:586` — extract vision features for multimodal path (~100 LOC)
3. `multimodal/processor.rs:146` — video support in processor (~200 LOC)
4. `mimo_v2_flash.rs:469` — MoE block instead of dense MLP (~50 LOC)

### 4.4 Tool & Reasoning Parsers
**Effort:** 2–3 days
- **GPT-OSS reasoning parser** — add to `crates/core/src/reasoning/mod.rs`; ~100 LOC
- **Qwen-VL tool parser (legacy)** — add to `crates/core/src/tool_parser/`; ~150 LOC

### 4.5 Medusa Model Loader
**Effort:** 0.5 day
- Add `MedusaModel` match arm in `from_config()` in `crates/core/src/models/mod.rs`; ~100 LOC
- Proposer (`medusa_proposer.rs`) already complete

---

## Summary: Effort by Tier

| Tier | Area | Estimated Effort |
|------|------|-----------------|
| 1 | Distributed (CP/DP/EP/MTP/Audio) | 16–21 weeks |
| 2 | Model expansion (VLMs/MoE/Quant/Eagle/SSM) | 14–18 weeks |
| 3 | Feature completion (Quant P3/API/Attention) | 8–12 weeks |
| 4 | Easy wins (PP enable/RoPE/Parsers/TODOs) | 2–3 weeks |
| **Total** | | **~40–54 weeks** |

## Dependency Graph (Critical Path)

```
MTP Framework → DeepSeek-MTP model
MoonViT encoder → Kimi-VL, Kimi-K2.5-VL
SigLIP2-NaViT → LFM2-VL
ERNIE4.5-VL RoPE → ERNIE4.5-VL
Audio Pipeline → Whisper → Qwen2-Audio, Ultravox → Qwen2.5-Omni, Qwen3-Omni
~~Data Parallelism → Expert Parallelism → EPLB~~ ✅ ALL DONE
Pipeline Parallelism (enable) → no blockers ← can do now
```

## Suggested Starting Point

1. **Now (no blockers):** PP enable, GPT-OSS parser, Medusa loader, phi4mm TODO
2. **Week 1–4:** ~~MTP framework + DeepSeek-MTP~~ ✅; ~~InternS1~~ ✅; ~~CP~~ ✅; ~~DP~~ ✅; ~~EP+EPLB~~ ✅; Step3-VL; AWQ-Marlin
3. **Week 4–8:** Audio pipeline + Whisper; Kimi-VL (after MoonViT); remaining VLMs
4. **Week 8–16:** Remaining VLMs; remaining Eagle variants; Quant P2
5. **Week 16+:** Quant P3; server audio API; attention backends
