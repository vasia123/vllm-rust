# lm_head 3D → 2D flatten — +40 % at c=8 (the unexpected win)

Date: 2026-05-09
Device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89), WSL2
Server: `VLLM_AWQ_HYBRID=1 ./target/release/vllm-server serve --model Qwen/Qwen3-4B-AWQ --port 8765 --enforce-eager --max-model-len 1024 --num-blocks 384`
Bench: `python3 scripts/bench_decode.py --concurrency 1 4 8 16 --prompt-len 256 --max-tokens 128 --runs 3`

## Result

| c | baseline | + flatten | Δ |
|---|---|---|---|
| 1 | 45.9 | 43.2 | −6 % (within laptop GPU run-to-run variance) |
| 4 | 85.8 | 101.7 | **+18 %** |
| 8 | 118.8 | **166.0** | **+40 %** |
| 16 | — | 314.6 | (no prior baseline at this c) |

## What changed

Both `LoadedUnquantizedLinear::forward`
(`crates/core/src/quantization/weight_loader.rs`) and
`TiedEmbeddingHead::forward` (`crates/core/src/models/qwen3_quantized.rs`)
previously did:

```rust
let w = match x.dims().len() {
    3 => self.weight.broadcast_left(x.dim(0)?)?,   // [V,H] → [B,V,H]
    _ => self.weight.clone(),
};
let y = x.matmul(&w.t()?)?;
```

For decode `x = [B, 1, H]` this routes through cuBLAS's batched GEMM
with `stride_b = 0` (the broadcast view). cuBLAS does NOT pick the
plain-GEMM algorithm in that case — it falls into a batched path
optimised for medium M, which under-utilises bandwidth on the
small-M / large-N lm_head shape.

Replaced with:

```rust
3 => {
    let x_flat = x.reshape((B*S, H))?;
    let y_flat = x_flat.matmul(&self.weight.t()?)?;
    y_flat.reshape((B, S, V))
}
```

Flatten + 2D matmul lands on cuBLAS's plain `Hgemm` path. cuBLAS picks
a markedly different (and faster) algorithm for shape M=B, K=hidden,
N=vocab.

## Why this finding was a long time coming

The Phase 2 work earlier today set out to write a custom 4-warp
cooperative kernel believing the bottleneck was MLP HBM bandwidth.
Microbench refuted it (the cooperative kernel was slower). When the
focus pivoted to lm_head, an attempted custom GEMV kernel ALSO
turned negative — **and the flatten that was carried along as
"infrastructure" happened to be the actual win**.

Memory rule `feedback_perf_assumption_test.md`: "test perf-fix
hypothesis with profile, not theory." Both the cooperative-warp and
custom-GEMV theories were appealing on paper but lost in microbench.
Meanwhile the 5-line flatten refactor — initially landed as a
"perf-neutral cleanup" because the profile-on bench showed 0 % —
turned out to be +40 % in real e2e once the profile overhead was
removed.

Earlier reading was misleading because:

- **Profile env (VLLM_PROFILE_DECODE=1) hides the win.** It adds
  CUDA event overhead that scales with kernel count. The 3D batched
  path makes more calls; with profiling on, the extra overhead
  amortises across paths and the 3D vs 2D delta vanishes.
- **The cuBLAS 3D-batched-with-stride-0 vs 2D plain-GEMM
  difference on lm_head shape is large** in this driver / hardware
  combination. Independent of profile state.

## Verification

- `cargo test --features cuda-default,gpu-test-small -p vllm-core
  --lib`: 4532/4532 PASS.
- Bench above with `--runs 3` shows tight per-run variance on c=4
  and c=8 (single-digit %).
- This change applies to ANY 3D-input forward through
  `LoadedUnquantizedLinear` or `TiedEmbeddingHead`. It's a model-
  agnostic refactor; Qwen3-4B-AWQ uses tied embeddings hence
  `TiedEmbeddingHead` is the hot path here, but other models with
  separate lm_head weights (loaded via the unquantized loader)
  benefit identically.

## Net session position post-finding

c=8 baseline pre-session: 118.8 tps. Post-session: 166.0 tps.
**+40 % e2e improvement** on the c=8 target case. lm_head per-token
cost at c=8 was the largest single slot pre-fix; this refactor
collapses it to roughly the level we'd expect for an HBM-bound
single GEMM on this shape.

## Negative results that fed into this finding

- `2026-05-09-mlp-cooperative-kernel-phase2-negative.md` (Phase 2
  custom 4-warp kernel — slower in microbench, reverted).
- `2026-05-09-mlp-gate-up-fusion-phase1-negative.md` (Phase 1 fused
  gate+up — 0 % e2e, reverted).
- A custom lm_head GEMV kernel was written and microbench'd; cuBLAS's
  plain GEMM (post-flatten) beat it. Reverted; the kernel files are
  not in the tree.
