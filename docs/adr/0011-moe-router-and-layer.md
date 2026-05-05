# ADR-0011: Shared MoE Routing and Expert Layer Infrastructure

## Status

Accepted (infrastructure already in place; this ADR documents the
canonical path for new MoE models and frames the remaining migration
work).

## Context

Mixture-of-Experts decoder layers in the wild differ along a small number
of declarative axes:

- **Scoring function**: softmax (Mixtral, Qwen2/3-MoE, DeepSeek)
  vs. sigmoid (GLM4-MoE, MiniMax-M2 with sigmoid mode, Dots1 with
  sigmoid mode).
- **Top-k flavor**: plain top-k (Mixtral, Grok-1) vs. grouped top-k
  (DeepSeek V2/V3 with `n_group` / `topk_group`, GLM4-MoE).
- **Score correction bias** (`e_score_correction_bias`): present in
  DeepSeek V3 and several derivatives.
- **Routed scaling factor**: post-renormalization multiplicative scale
  (DeepSeek V3 `routed_scaling_factor`).
- **Renormalization**: whether top-k weights are re-normalized after
  selection (most models: yes; some: no).
- **Shared experts**: shared/dense path that runs alongside the routed
  experts (DeepSeek family, Qwen2-MoE, MiniMax-M2's optional
  `shared_intermediate_size`, ExaoneMoE's `num_shared_experts`).
- **Expert layer cadence**: every layer (Mixtral, Qwen2/3-MoE) vs.
  per-layer flag list (`is_moe_layer`, ExaoneMoE) vs. `first_k_dense_replace`
  + `moe_layer_freq` schedule (DeepSeek, Dots1).
- **Activation**: SwiGLU (`is_act_and_mul = true`) vs. plain activation
  (Switch-style).
- **Quantization**: per-expert quantized weights (`QuantizedMoELayer`).
- **Expert parallelism**: experts sharded across ranks (`EPMoELayer`),
  with optional load balancing (`eplb`).

Without shared infrastructure each model duplicates the gating-softmax-
top-k-dispatch-aggregate pipeline (~150–200 LOC), and small numerical
differences in the bespoke top-k loops (Vec sort vs. tensor `arg_sort`
vs. CPU scalar) make per-model inference quality drift hard to audit.

## Decision

The canonical MoE path lives in `crates/core/src/moe/` and is composed
of small, single-purpose primitives that decoder-only model files
compose declaratively.

### Components

```
moe/
├── router.rs            ─ MoERouter trait + TopKRouter impl
├── expert_layer.rs      ─ MoEExpert + MoELayer + MoELayerWithShared
├── quantized_experts.rs ─ QuantizedMoELayer for INT4/INT8 expert weights
├── ep_layer.rs          ─ EPMoELayer for expert-parallel inference
├── eplb.rs              ─ Expert load tracking
├── eplb_execute.rs      ─ Live expert rearrangement across ranks
├── expert_map.rs        ─ Logical-to-physical expert mapping (EP)
├── topk_softmax.rs      ─ Fused top-k + softmax kernel
├── token_dispatch.rs    ─ Token-to-expert dispatch metadata
├── fused/               ─ Block-fused MoE GEMM kernels
└── lora.rs              ─ Per-expert LoRA adapters
```

### `MoERouter` trait

```rust
pub trait MoERouter: Send + Sync {
    fn route(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)>;
    fn num_experts(&self) -> usize;
    fn top_k(&self) -> usize;
}
```

The single canonical impl is `TopKRouter`, whose behavior is controlled
by `RouterConfig`:

```rust
pub struct RouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub renormalize: bool,
    pub scoring_func: ScoringFunc,        // Softmax | Sigmoid
    pub use_grouped_topk: bool,
    pub num_expert_groups: Option<usize>,
    pub topk_per_group: Option<usize>,
    pub routed_scaling_factor: f64,
}
```

Optional `e_score_correction_bias` is set via
`TopKRouter::new_with_bias`. This single struct + trait covers softmax,
sigmoid, grouped top-k, score correction, and routed scaling — the full
matrix used across Mixtral, Qwen2/3-MoE, DeepSeek V2/V3, GLM4-MoE,
ExaoneMoE, MiniMax-M2 in sigmoid mode, Dots1, ERNIE 4.5-MoE, GraniteMoE,
…

### `MoELayer` and `MoELayerWithShared`

`MoELayer` combines a `TopKRouter` with `Vec<MoEExpert>` and a fused
expert-dispatch kernel. The forward pass delegates to either:

- a naive per-token loop (`forward`), used in tests and for debugging;
- a batched fused kernel (`forward_fused`), used in production, with
  5–10× throughput over the naive path.

`MoELayerWithShared` extends `MoELayer` with an optional shared-expert
path: a separate gate + dense MLP whose output is added to the routed
mixture. Used by Qwen2-MoE, DeepSeek family, MiniMax-M2 with shared
experts, ExaoneMoE.

### Expert parallelism

`EPMoELayer` shards `num_experts` across ranks, dispatches tokens by
expert assignment, and reduces by all-to-all. The dynamic load balancer
(`eplb` + `eplb_execute`) tracks per-expert load per layer and rearranges
expert weight tensors across ranks at safe boundaries to minimize
imbalance, transparent to model code.

### Quantized experts

`QuantizedMoELayer` mirrors `MoELayer` but holds quantized expert
weights (GPTQ-style INT4 / INT8 packed with per-group scales) and
dispatches to a quantized GEMM kernel. The router is the same
`TopKRouter` — only expert math changes.

## Adoption status

**Already on the canonical path** (use `MoELayer` /
`MoELayerWithShared` / `TopKRouter` / `EPMoELayer` /
`QuantizedMoELayer`):

- Mixtral (text + quantized + LoRA)
- Qwen2-MoE (single + EP)
- Qwen3-MoE
- Step3-Text
- Step3.5
- Grok-1 (TopKRouter routing component)
- DeepSeek family (LoRA + quantized)
- DBRX (existing `MoELayer` consumer through `from_config_with_quant`)
- MLP Speculator (no MoE — listed for completeness)

**Bespoke today (Phase 5 candidates)**:

- ExaoneMoE — per-token CPU loop
- MiniMax-M2 — per-token CPU loop
- MiniMax-Text01 — shared expert mixing in softmax/sigmoid mode
- Grok-1 expert layer (router migrated; experts still bespoke)
- Dots1 — shared/routed mixing
- ERNIE 4.5-MoE
- GraniteMoE / GraniteMoE-Shared / GraniteMoE-Hybrid
- JetMoE
- OLMoE / Flex-OLMo
- Bailing-MoE
- GLM4-MoE
- Pangu (`pangu_pro_moe`)
- Hunyuan dense+MoE hybrids
- Arctic (alternates dense/MoE; uses `MoELayer` already for the MoE
  layers but the dense-vs-MoE dispatch is bespoke)

### Phase 5 migration discipline

Each migration must:

1. Be scoped to one model per PR.
2. Replace the bespoke routing+dispatch with `MoELayer` /
   `MoELayerWithShared`. The shared-expert variant is preferred when
   the model has an explicit shared/routed mixture.
3. Preserve the model's `attention_type` / `is_moe_layer` cadence
   logic — that lives in the decoder-layer code, not in the MoE
   primitive.
4. **Be gated on a per-PR throughput benchmark.** The plan's regression
   bound is ≤ 2% on Mixtral-8×7B FP16. Migrations that lose more than
   that to numerical changes (different top-k algorithm, different
   reduction order) are reverted and revisited.
5. Have functional tests covering: router shape, top-k weights sum
   ≈ 1.0 (when renormalize), shared-expert path adds correctly, and
   end-to-end model forward/decode shape.

The migration order parallels the attention work: dense-MoE
substitutions first (Bailing-MoE, GLM4-MoE — already partially routed
through `TopKRouter`), then shared-expert models, then the full bespoke
list.

### What stays bespoke

- **EPLB orchestration** (`eplb.rs`, `eplb_execute.rs`) is independent
  of the router/expert primitives and handled at the engine level.
  Models do not need to change to support it.
- **Per-LoRA expert adapters** (`lora.rs`) are an optional feature on
  `MoELayer`; bespoke LoRA on top of bespoke MoE is rare and remains
  per-model.
- **Hybrid models** that gate on a per-layer attention type (Jamba,
  Bamba, Falcon-H1, Nemotron-H, GraniteMoE-Hybrid) keep their
  decoder-layer dispatch but can still use `MoELayer` inside the
  MoE arms.

## Consequences

**Benefits:**

- New MoE models slot in declaratively. The model file owns config
  parsing + decoder layer wiring + per-layer cadence; the expert
  primitive is shared.
- One place to optimize. The fused dispatch kernel and EP/EPLB stack
  benefit every model that adopts `MoELayer`. Bespoke implementations
  do not.
- Quantized MoE works through the same shape: swap `MoELayer` for
  `QuantizedMoELayer`, keep the same router.

**Trade-offs:**

- Bespoke implementations exist for legitimate reasons (custom expert
  activation in Grok-1's `GeluAndMul`; custom shared/routed mixing in
  MiniMax-Text01). The migration is a per-model judgment about whether
  the architectural variation is worth a config knob on the shared
  primitive or warrants staying bespoke.
- Numerical bit-exactness across implementations is not guaranteed —
  top-k algorithms differ in tie-breaking. Per-PR benchmarks are the
  contract, not bit equality.

**Future work:**

- Phase 5 migrations as described above.
- A `no_local_moe.rs` CI guardrail analogous to
  `no_local_attention.rs`, listing the bespoke MoE models with reasons,
  once the migration cohort completes.
- Performance audit of `MoELayer::forward` (naive path) — the per-token
  loop is wasteful when the fused path is unavailable; consider a
  vectorized fallback that doesn't require the CUDA kernel.
