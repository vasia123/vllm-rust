# ADR 0017: Async on-device sampling pipeline

Date: 2026-05-08
Status: **PROPOSED** — foundation in tree, full implementation pending
Tracking commit (foundation): `<this commit>`
Related docs:
- `docs/perf/post-15D-nsys-finding.md` — nsys evidence
- `docs/perf/post-15D-decode-profile.md` — pre-fix profile

## Context

Post-Stage-15.D-body production decode at c=8 spends **~30 ms per
step in host-side stall** waiting for `gpu_sample_batch_with_diffs`'s
final `to_vec1()`. Confirmed via:

- `nsys profile`: 45 % of CUDA API time in `cuMemcpyDtoHAsync_v2`,
  54 calls × 41 ms avg, durations clustered 13-54 ms.
- Direct sync+Instant timer (env `VLLM_PROFILE_SAMPLER_TOVEC=1`):
  to_vec1 of an 8-element u32 tensor = **50 ms per call** under load.

The 50 ms is NOT the 32-byte DMA. It is GPU stream-drain wait — the
host blocks until the stream finishes 36 layers + lm_head + sampler
kernel. Each forward serializes on this. No multi-step or parallel
issuing helps because the next forward's `input_ids` needs the
sampled IDs (read host-side from generated_token_ids).

If eliminated, c=8 throughput projection: **105 → 140-180 tps**
(per-step wallclock 60 → ~40 ms; theoretical lower bound is GPU
compute ~38 ms).

## Decision

Build an async-pipelined sampling path that:

1. Keeps sampled token IDs ON GPU as a Tensor (no host trip).
2. Uses that Tensor directly as `input_ids` for the next forward
   (no host-vec → tensor rebuild, no DtoH+HtoD round-trip).
3. Launches the next forward IMMEDIATELY after the sampler kernel —
   without waiting for host-side anything.
4. Asynchronously DtoHs the sampler IDs on a SEPARATE CUDA stream
   for host-side finish-reason check + streaming output. Host
   blocks on the DtoH event only when it actually needs the IDs.
5. Allows finish-checks to lag by 1-N steps (amortise DtoH stall
   across multi-step decode block); request will potentially
   over-decode by 1-N tokens past EOS, trimmed in finalisation.

## Implementation phases

### Phase 1 (foundation, this commit)

- [x] `gpu_sample_batch_with_diffs_to_tensor`: returns `Tensor`
      instead of `Vec<u32>` — no `to_vec1()` on the hot path. Mixed
      greedy+multinomial batches still fall back to host trip; pure-
      multinomial and pure-greedy paths stay GPU-resident.
- [x] All 4503 tests pass; existing `gpu_sample_batch_with_diffs`
      kept as-is for back-compat.

### Phase 2 (engine wiring, ~1 day)

- [ ] In `execute_batched_decode_with_graph`, switch sampler call to
      `*_to_tensor`.
- [ ] Cache the resulting tensor as `last_sampled_ids: Option<Tensor>`
      in `OwnedExecutionState`.
- [ ] Build next step's `input_ids` from this tensor (concat / index)
      instead of from host `generated_token_ids`.
- [ ] DtoH on a SEPARATE stream — record CUDA event after enqueue,
      poll for completion at end of multi-step block.

### Phase 3 (deferred finish-check, ~0.5 day)

- [ ] Defer `check_finished` until end of multi-step block (host has
      IDs by then via DtoH event).
- [ ] Trim over-decoded tokens past EOS in finalisation.
- [ ] Update `streamed_text_len` accounting to handle
      retro-active token removal.

### Phase 4 (validation, ~0.5 day)

- [ ] `scripts/test_qwen3_awq_correctness.sh` — must remain 7/7 PASS.
- [ ] `bench_decode --runs 3` c=1/4/8/16 — record lift.
- [ ] Snapshot in `docs/perf/bench-history/`.

## Risks

| Risk | Mitigation |
|---|---|
| Mixed greedy/multinomial batches still need host trip | Phase 1 keeps that path; rare in practice |
| DtoH on second stream still blocks if pinned memory not used | Use `cudaMallocHost` for the host buffer (cudarc supports it) |
| Over-decode past EOS by N tokens | Trim in finalisation; minor quality impact ≤ multi_step_count tokens |
| Finish-reason latency increases | Streaming first-token TTFT may improve (less host work mid-decode) |

## Alternatives considered

1. **Pre-allocated pinned host buffer for sampler result.** Marginal
   — bottleneck is the stream-drain wait, not the DMA itself.
2. **Move full sampler to GPU including finish-check.** Full
   on-device EOS detection is feasible but requires reworking the
   max_tokens / stop_token_ids logic which currently lives in Rust.
   Phase 2+3 captures the wins without needing this.
3. **Speculative decoding.** Reduces total forward count (so total
   stall count) but doesn't fix per-step stall. Orthogonal; can
   stack with this fix.
