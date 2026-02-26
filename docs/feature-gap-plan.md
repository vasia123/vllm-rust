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
- `crates/core/src/multimodal/mel_spectrogram.rs` ✅ DONE — log-mel spectrogram (pure Rust DFT STFT + mel filterbank + Whisper normalization); 9 tests; 3967 total — 2026-02-23
  - `MelSpectrogramConfig::whisper()`: n_fft=400, hop=160, n_mels=80, sr=16000, f_max=8000
  - `hann_window()`, `stft_power_spectrum()`, `build_mel_filterbank()`, `log_mel_spectrogram()`
  - Output `[n_mels, n_frames]` matches HuggingFace `WhisperFeatureExtractor`
- `crates/core/src/multimodal/audio.rs` has `AudioData`/`AudioSpec` + normalize/resample but NO audio file loading (see `audio` feature below)
- Add WAV/MP3 loading (hound + symphonia crates)
- **Encoder models (12 total):**
  - `whisper.rs` ✅ DONE — `WhisperForConditionalGeneration`, implements `ModelForEncoderDecoder`; Conv1d×2 encoder + sinusoidal pos + cross-attn decoder; 5 tests; 3972 total — 2026-02-23
  - `qwen2_audio.rs` ✅ DONE — `Qwen2AudioForConditionalGeneration`; Conv1d(s=1)+Conv1d(s=2)+AvgPool1d(k=2,s=2) encoder; 5 tests; 3977 total — 2026-02-23
  - `ultravox.rs` ✅ DONE — `UltravoxModel`; WhisperEncoder + StackAudioFrames + FFProjector(SwiGLU) + Llama LLM; 5 tests; 3982 total — 2026-02-23
  - `qwen2_5_omni_thinker.rs` ✅ DONE — `Qwen2_5OmniThinkerForConditionalGeneration`; Qwen2AudioEncoder + Qwen25VisionTransformer + Qwen2ForCausalLM; 5 tests; 3987 total — commit cc24c0e
  - `qwen3_omni_moe_thinker.rs` ✅ DONE — `Qwen3OmniMoeThinkerForConditionalGeneration`; Conv2d×3 audio encoder + Qwen3OmniVisionTransformer (LayerNorm + SiLU MLP) + Qwen3MoeForCausalLM; 5 tests; 3992 total — commit cc24c0e
  - `qwen3_asr.rs` ✅ DONE — `Qwen3ASRForConditionalGeneration`; Qwen3OmniMoeAudioEncoder + Qwen3ForCausalLM; 5 tests; commit 21fa570
  - `granite_speech.rs` ✅ DONE — `GraniteSpeechForConditionalGeneration`; Conformer CTC encoder (Shaw RPE + block attn) + BLIP2 QFormer projector + GraniteForCausalLM; 5 tests — commit 9297a5b
  - `funaudiochat.rs` ✅ DONE — `FunAudioChatForConditionalGeneration`; continuous Conv1d×2+transformer encoder + discrete group-avg encoder + Qwen3; 5 tests — commit 96aff3b
  - `audioflamingo3.rs` ✅ DONE — `AudioFlamingo3ForConditionalGeneration`; Qwen2AudioEncoder + 2-layer GELU MLP projector + Qwen2; 5 tests — commit 1c5dce2
  - `voxtral.rs` ✅ DONE — `VoxtralForConditionalGeneration`; WhisperEncoder + downsample reshape + AudioLanguageAdapter + MistralForCausalLM; 5 tests — commit 18a930e
  - `glmasr.rs` ✅ DONE — `GlmAsrForConditionalGeneration`; Conv1d×2+GELU + GQA+partial-RoPE encoder + MLP projector + LlamaForCausalLM; 6 tests — commit f0dc167
  - `phi4mm_audio.rs` ✅ DONE — `Phi4MMAudioEmbedding` (ConformerEncoder stub + MLP/linear AudioProjection); 11 tests — commit 727e7dc
  - `gemma3n_audio` ✅ DONE — extended `gemma3n_vlm.rs` with `Gemma3nAudioEncoder` stub + `embed_audio` projector; audio scatter in `forward_multimodal`; 1 new test — commit f5856bb
  - `musicflamingo.rs` ✅ DONE — `MusicFlamingoForConditionalGeneration` = type alias for `AudioFlamingo3ForConditionalGeneration`; 3 tests — commit f5856bb
- **Files:** extend `multimodal/audio.rs`, new `multimodal/audio_encoder.rs`, per-model files

---

## Tier 2: Model Expansion (Hard — High Volume)

### 2.1 Missing VLMs (~18 models) ✅ InternS1, Step3-VL, ERNIE4.5-VL, InternS1Pro, MoonViT+Kimi-VL, Kimi-K2.5, HyperCLOVA-X, GLM-OCR, GLM4-1V, Kanana-V, Ovis, Aria, OpenPangu-VL, Keye-VL, Isaac DONE
**Difficulty:** ★★★☆☆ per model | **Effort:** 4–6 weeks total
- All follow Vision→Projector→LM pattern; 150–300 LOC per model
- **Blockers to resolve first (new vision encoders):**
  - `MoonViT` ✅ — `moonvit.rs` done; also unblocks Kimi-K2.5-VL
  - `SigLIP2-NaViT` ✅ — implemented directly inside `lfm2_vl.rs` (no external NaViT needed; the LFM2-VL variant is simpler — standard attention, bilinear pos-emb resize, no windowed attention)

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
- `keye_vl.rs` — `KeyeVL1_5ForConditionalGeneration` ✅ (5 tests) — commit 0d3767d
- `isaac.rs` — `IsaacForConditionalGeneration` ✅ (5 tests) — commit 3641eee
- `deepseek_ocr2.rs` — `DeepseekOCR2ForCausalLM` ✅ (5 tests) — 2026-02-23
- `dots_ocr.rs` — `DotsOCRForCausalLM` ✅ (6 tests) — commit 6f8fbf4 — DotsVisionTransformer (Conv2d patchifier + VisionRoPE + SwiGLU blocks + PatchMerger) + Qwen2
- `mistral3.rs` — `LightOnOCRForConditionalGeneration` ✅ (1 test) — commit 6f8fbf4 — Mistral3 variant with HF weight path remapping

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
Keye-VL 1.5: `KeyeSiglipVisionTransformer` (Conv2d patch embed + `packing_position_embedding` (Embedding, 32768 entries) +
SigLIP encoder with 2D RoPE (`SigLipRotaryEmbedding`, split-half rotation, head_dim/2 freqs) + `post_layernorm`) +
`KeyeVL1_5Projector` (2×2 spatial merge → `pre_norm` LayerNorm → `linear_1`(bias) → GELU → `linear_2`(bias)) +
`Qwen3ForCausalLM`. qkv_proj has bias; out_proj has no bias.
Weight paths: `visual.vision_model.embeddings.*`, `visual.vision_model.encoder.layers.{i}.*`,
`mlp_AR.{pre_norm,linear_1,linear_2}.*`, `language_model.{model.*,lm_head.*}`.
VL helpers added to `Qwen3ForCausalLM` (embed_text, forward_with_embeddings, forward_decode_batch_with_embeddings).

Isaac: `Siglip2VisionTransformer` (Linear patch embed + bilinear-interp `position_embedding` (Embedding, `num_patches` entries, grid=√num_patches) +
N pre-norm encoder layers (GELU-erf MLP, separate q/k/v proj with bias, out_proj with bias, `layer_norm1/2`) + `post_layernorm` + `pixel_shuffle` spatial merge (scale²× channel expand)) +
`IsaacVisionEmbedding` projector (ViT at `transformer.*` + `linear_fc1` (SiLU, no bias) + `linear_fc2` (no bias)) + `Qwen3ForCausalLM`.
`bilinear_interp_pos_emb`: align_corners=False CPU loop (no antialias); fast path when tgt==src.
Weight paths: `vision_embedding.transformer.{embeddings,encoder,post_layernorm}.*`,
`vision_embedding.linear_fc{1,2}.weight`, `language_model.{model,lm_head}.*`.
HF mapper: `model.vision_embedding.{0→transformer,1→linear_fc1,3→linear_fc2}`, `model.text_model.→language_model.model.`.

DeepSeek-OCR2: `SamImageEncoderViT` (Conv2d patch embed + abs pos embed + 12 blocks with decomposed relative pos embeds,
window=14 for non-global, global attn at [2,5,8,11] → neck Conv2d×4 + net_2/net_3 stride-2 Conv2d → `[B, 896, H/64, W/64]`) +
`Ocr2Qwen2Encoder` (24 Qwen2 decoder layers as visual encoder: flatten SAM output → cat with learnable queries [144 or 256] →
custom dual-mode mask [image=full non-causal, query=causal] → return query tokens `[B, n_query, 896]`) +
`Ocr2MlpProjector` (spatial unfold dr² patches into channel dim + 2-layer GELU MLP) + `view_seperator` token +
`Qwen2ForCausalLM`. Weight paths: `model.sam_model.*` (SAM), `model.qwen2_model.*` (encoder), `model.projector.*`,
`model.view_seperator`, `model.*` / `lm_head.*` (Qwen2 LLM). GQA: 14 heads, 2 KV heads → ratio=7 broadcast.

- **P2 models (~2 remaining):** ~~MiniCPM-O~~ ✅ DONE (`minicpmo.rs`, 5 tests — commit 7763a99), MiniMax-VL-01 (BLOCKED: Lightning Attention), Nemotron-VL (BLOCKED: dynamic AutoModel), ~~Hunyuan-Vision~~ ✅ DONE (`hunyuan_vision.rs`, 5 tests + XDRoPE — commit 94ada76), ~~LFM2-VL~~ ✅ DONE (`lfm2_vl.rs`, 6 tests — commit f0ac8f7 — Siglip2VisionTransformer + pixel-shuffle projector + Lfm2ForCausalLM)
- **NemotronNAS / DeciLM** ✅ DONE (`nemotron_nas.rs`, 5 tests — commit cb28bb0): `DeciLMForCausalLM`/`NemotronNasForCausalLM`; per-layer `block_configs` JSON (no_op_attention, no_op_ffn, n_heads_in_group, intermediate_size/ffn_mult); `NasAttention` with explicit per-layer num_kv_heads; `NasDecoderLayer` with optional attn+FFN; separate `kv_layer_idx` counter for KV cache (num_kv_layers ≤ num_hidden_layers)
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
  1. `awq_marlin.rs` ✅ DONE — `AwqMarlinConfig`, `AwqMarlinWeightLoader`, CPU nibble deinterleave (`repack_awq_nibbles`); 7 tests; 3924 total
     - CPU path: AWQ nibble deinterleave → GPTQ ordering → `MarlinLinear::process_weights()` (GPTQ→Marlin tile repack)
     - GPU TODO: add `awq_marlin_repack_int4` PTX to `marlin_gemm.cu` for single-step AWQ→Marlin repack
     - GPU TODO: qzeros need `marlin_awq_repack_zp` conversion for Marlin format
  2. `fbgemm_fp8.rs` ✅ DONE — `FbgemmFp8Config` (modules_to_not_convert, activation_scale_ub), `FbgemmFp8Linear` (per-channel dequantize, F32 CPU path), `FbgemmFp8WeightLoader`; 6 tests; 3930 total
     - CPU/Ampere: per-channel FP8 dequantize → F32 matmul; CUDA cap≥89: fp8_gemm (hardware FP8)
     - Ampere TODO: wire `MarlinScalarType::Float8E4m3fn` path for `apply_fp8_marlin_linear`
  3. `gptq_marlin` detection alias ✅ DONE — added `"gptq_marlin"` → `QuantizationMethod::Marlin` in detection.rs; min_capability=80 (Marlin)
  4. `ptpc_fp8.rs` ✅ DONE — `PtpcFp8Config` wrapping `Fp8Config`, min_capability=94 (AMD MI300+), reuses `Fp8WeightLoader`; 6 tests; 3936 total
     - Detection: `"ptpc_fp8"` in both config.json and quantize_config.json paths
     - Weight format identical to standard FP8 — `is_checkpoint_fp8_serialized=false`, dynamic-only
  5. `mxfp4.rs` ✅ DONE — `MxFp4Config`, `MxFp4Linear` (FP4 E2M1 nibble unpack + E8M0 block scales, CPU emulation); 19 tests; 4011 total
     - FP4 E2M1 lookup table, lower-nibble-first packing, block_size=32, min_capability=80
     - MoE GPU dispatch (FlashInfer/Marlin/Triton) tracked as TODO
  6. `modelopt.rs` ✅ DONE — `ModelOptConfig` + `ModelOptLinear` covering FP8/FP8_PER_CHANNEL_PER_TOKEN/FP8_PB_WO/NVFP4/MXFP8; 20 tests; 4035 total — commit 80459fe
     - Detection routes FP8/NVFP4/FP8_PB_WO → `ModelOptFull`; MXFP8 stays → `ModelOpt` (existing MxFp8Config)
     - Both nested `{"quantization":{"quant_algo":...}}` and flat `{"quant_algo":...}` JSON layouts
  - `input_quant_fp8.py` — NOT a QuantizationConfig (it's a CustomOp utility for FP8 activation quantization); skip
- **Reference:** `reference/vllm/model_executor/layers/quantization/`

### 2.4 Speculative Decode: Missing Eagle Variants + Medusa ✅ DONE
**Difficulty:** ★★★☆☆ | **Effort:** 2–3 weeks
- Pattern fully established: `eagle_llama.rs` (Eagle-1) and `eagle3.rs` (Eagle-3)
- **Completed:**
  - `eagle_llama4.rs` — `EagleLlama4ForCausalLM` ✅ (5 tests) — commit 614ebc3
  - `eagle_minicpm.rs` — `EagleMiniCPMForCausalLM` ✅ (5 tests) — commit 614ebc3
  - `eagle_deepseek.rs` — `EagleDeepSeekForCausalLM` (arch: `EagleDeepSeekMTPModel`) ✅ (5 tests) — commit abd76b0
  - `eagle1_from_config()` factory ✅ (4 variants: Llama, Llama4, MiniCPM, DeepSeek)
  - Fix: `EagleDeepSeekMTPModel` moved from `mtp_from_config()` to `eagle1_from_config()`
  - `models/medusa.rs` — `MedusaModel` + `MedusaDraftModel` trait + `medusa_from_config()` ✅ (5 tests)
- **Architecture notes:**
  - DeepSeek Eagle: `fc(cat(enorm(embed), hnorm(hidden))) → MLA layers → norm`; MLA cache required
  - EagleLlama4: `fc(cat(embed, hidden)) → Llama4 layers → norm`; uses `TpContext::single_gpu()`
  - EagleMiniCPM: `fc(cat(norm1(embed), norm2(hidden))) → MiniCPM layers`; no final norm, divide by `scale_width`
  - Medusa: K independent residual blocks (`x += SiLU(Wx)`) + K lm_heads; weights at `blocks.{i}.layers.{j}` / `lm_heads.{i}`

### 2.5 SSM/Mamba2 Ops
**Difficulty:** ★★★☆☆ | **Effort:** 2–3 weeks
- **Causal conv1d refactor** ✅ DONE — `crates/core/src/ssm/causal_conv1d.rs` (5 tests); removed ~520 LOC duplication from 9 models (mamba, mamba2, jamba, bamba, zamba2, falcon_h1, plamo2, nemotron_h, granitemoe_hybrid); 3946 total
- **ShortConv** ✅ DONE — `ShortConvBlock` + `Lfm2ShortConvDecoderLayer` implemented inline in `lfm2.rs`; `Lfm2ForCausalLM` + `Lfm2MoeForCausalLM` now fully hybrid (SSMStateManager, attn_layer_cache_idx, forward_with_request_id); 24 tests (3951 total) — 2026-02-23
- **Gated LayerNorm** ✅ DONE — `crates/core/src/ssm/gated_layer_norm.rs`; `rms_norm_gated(x, z, weight, eps, norm_before_gate)`; 6 tests — 2026-02-23
- **`mamba2.rs` correctness fix** ✅ DONE — fixed `in_proj` size (added gate dim → `2*d_inner + 2*n*S + H`), conv on full `xBC`, post-SSM `Mixer2RMSNormGated` output norm, `Mamba2DecoderLayer` pre-norm at correct weight path, removed spurious `norm_b`/`norm_c`; `SSMStateManager::new_with_conv_channels` for `conv_dim` states; 11 tests (3958 total) — 2026-02-23
- **SSD ops** (6 Python files) — chunked parallel SSD scan; required for GPU-efficient Mamba2
  - `crates/core/src/ssm/ssd.rs` (~300 LOC); CPU sequential recurrence is already correct

---

## Tier 3: Feature Completion (Medium)

### 3.1 Quantization P3 Methods
**Effort:** 4–6 weeks (7 methods) | Same trait pattern as P2
- ✅ `cpu_wna16.rs` DONE — `CpuWna16Config` wraps `AwqConfig` (Marlin disabled); detection keys `"cpu_awq"`/`"cpu_wna16"`; min_capability=0; 5 tests
- ✅ `inc.rs` DONE — `IncConfig` routes to `GptqConfig`/`AwqConfig` based on `packing_format`; detection keys `"inc"`/`"auto-round"`; 7 tests
- ✅ `torchao.rs` DONE — `TorchaoConfig` wraps `NoQuantizationConfig` (standard BF16 matmul); detection already wired; 5 tests
- ✅ `fp_quant.rs` DONE — `FpQuantConfig`, `FpQuantLinear` (FP4 E2M1 + per-group E8M0/FP8 scales + global scale); MxFp4/NvFp4 variants; CPU dequant→F32 matmul; detection key `"fp_quant"`; 7 tests — commit `89b94e8`
- AWQ-Triton (kernel-only, no separate config), Petit (AMD-only external kernel — skip), QUARK (6 files, complex)
- All in `crates/core/src/quantization/`

### 3.2 Server API — Speech & Realtime
**Effort:** 2–3 weeks
- ✅ `POST /v1/audio/transcriptions` + `POST /v1/audio/translations` DONE — commit `3a5bd10`
  - `crates/server/src/api/audio.rs`: multipart/form-data upload (axum multipart), WAV PCM decoding (hound), normalize to 16 kHz mono, `build_audio_prompt`, `run_audio_task`, 7 tests; 4550 total
  - `GenerationRequest.audio_inputs: Vec<AudioData>` added; all struct literals updated
  - `AudioData` re-exported from `vllm_core::multimodal`
  - NOTE: model forward passes need to consume `audio_inputs` to embed via encoder+projector (see doc comment)
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

### 4.1 Pipeline Parallelism — Enable ✅ DONE
**Effort:** 2–3 days | **Difficulty:** ★★☆☆☆

**Completed (2026-02-23):**
- Extended `send_worker_signal` to 6-element header: `[signal, batch_size, seq_len, seqlen_offset, num_slots, num_block_ids]` + `WorkerSignal` struct ✅
- Added `SIGNAL_EXECUTE_DECODE = 2`, `MAX_BLOCKS_PER_SEQ = 64`, `DECODE_META_WORDS_PER_SEQ = 67` ✅
- `PipelineStagedModel::broadcast_prefill_signals()` — rank 0 sends control header + slot_mapping + block_ids to all workers before forwarding hidden states ✅
- `PipelineStagedModel::broadcast_decode_signals()` — rank 0 sends decode metadata (fixed `DECODE_META_WORDS_PER_SEQ` words per sequence) to all workers ✅
- `pipeline_worker_loop` — receives extended header, recvs slot_mapping/block_ids or decode_meta accordingly, reconstructs `BlockTable` + `DecodeSequenceMetadata`, dispatches to `forward_layers` or `forward_layers_decode_batch` ✅
- Removed error gate at `crates/server/src/main.rs` ✅
- 27 pipeline tests pass (2 new: `decode_meta_constants_consistent`, `send_worker_signal_decode_encodes_correctly`) ✅

**Remaining:** `pipeline_parallel_size` not yet wired through `ServerLaunchConfig` → `run_server` (no model registry hook for stage slicing yet).

### 4.2 RoPE Variants
**Effort:** 1 week | All in `crates/core/src/layers/rotary.rs`
- ~~**MRoPE Interleaved**~~ ✅ DONE — `mrope_interleaved.py`; `get_mrope_interleaved_id_list` + `MRoPEInterleaved` in `layers/rotary.rs`; 8 tests — commit pending
- **FoPE** — `fope.py`; ~200 LOC; factored positional encoding
- ~~**XDRoPE**~~ ✅ DONE — `xdrope.py`; `XDRotaryEmbedding` in `layers/rotary.rs`, 5 tests — commit 94ada76
- **ERNIE4.5-VL RoPE** — `ernie45_vl_rope.py`; ~150 LOC; unblocks ERNIE4.5-VL

### 4.3 Resolve P1 Code TODOs
**Effort:** 3–5 days
1. ~~`phi4mm.rs:271` — HD transform + 2x2 compression + projection~~  ✅ DONE
   - `avg_pool2x2_nhwc()`, `hd_transform()`, `glb_gn`/`sub_gn` learnable separators
   - Fixed projector input dim (`image_dim_out` not `*4`), 2 new tests (single + HD)
2. ~~`molmo2.rs:586` — extract vision features for multimodal path~~  ✅ DONE
   - `merge_vision_features()`, splices pre-projected `[np, hidden]` embeddings at patch positions; 1 new test
3. ~~`multimodal/processor.rs:146` — video support in processor~~  ✅ DONE
   - `process_video()` (Embedding/Frames paths), `parse_video_url()`, `find_video_placeholder_positions()`
   - `ProcessorConfig` extended with `video_placeholder`/`video_placeholder_id`/`num_video_tokens`
   - `process_content()` handles `ContentPart::Video` (mixed image+video supported); 6 new tests; 3915 total
4. ~~`mimo_v2_flash.rs:469` — MoE block instead of dense MLP~~  ✅ DONE
   - `MiMoV2Expert`, `MiMoV2MoEBlock` (sigmoid + grouped top-k, no shared experts), `MiMoV2Mlp` enum; 1 new test; 3909 total

### 4.4 Tool & Reasoning Parsers
**Effort:** 2–3 days
- **GPT-OSS reasoning parser** ✅ DONE — `GptOssReasoningParser` in `reasoning/mod.rs`; 5 tests; 3905 total
  - Format: `<|channel|>analysis<|message|>{reasoning}<|end|>` / `<|channel|>final<|message|>{content}<|end|>`
  - Streaming: re-parse + diff (mirrors Python `parse_chat_output` approach)
  - Aliases: `gpt_oss`, `gpt-oss`, `gptoss`
- **Qwen-VL tool parser (legacy)** — add to `crates/core/src/tool_parser/`; ~150 LOC

### 4.5 Medusa Model Loader ✅ DONE
**Effort:** 0.5 day
- `models/medusa.rs` — `MedusaModel` + `MedusaDraftModel` trait + `medusa_from_config()` ✅ (5 tests)
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
~~SigLIP2-NaViT → LFM2-VL~~ ✅ DONE
ERNIE4.5-VL RoPE → ERNIE4.5-VL
Audio Pipeline → Whisper → Qwen2-Audio, Ultravox → Qwen2.5-Omni, Qwen3-Omni
~~Data Parallelism → Expert Parallelism → EPLB~~ ✅ ALL DONE
Pipeline Parallelism (enable) → no blockers ← can do now
```

## Suggested Starting Point

1. **Now (no blockers):** PP enable, ~~GPT-OSS parser~~ ✅, ~~Medusa loader~~ ✅, ~~phi4mm HD transform~~ ✅, molmo2 TODO
2. **Week 1–4:** ~~MTP framework + DeepSeek-MTP~~ ✅; ~~InternS1~~ ✅; ~~CP~~ ✅; ~~DP~~ ✅; ~~EP+EPLB~~ ✅; ~~Step3-VL~~ ✅; ~~AWQ-Marlin~~ ✅; ~~FBGEMM FP8~~ ✅; ~~gptq_marlin alias~~ ✅; ~~PTPC FP8~~ ✅
3. **Week 4–8:** Audio pipeline + Whisper; Kimi-VL (after MoonViT); remaining VLMs
4. **Week 8–16:** Remaining VLMs; remaining Eagle variants; Quant P2
5. **Week 16+:** Quant P3; server audio API; attention backends
