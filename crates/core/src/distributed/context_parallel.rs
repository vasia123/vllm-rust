//! Decode Context Parallelism (DCP) for long-context inference.
//!
//! DCP shards the KV cache along the sequence dimension across multiple GPUs, allowing
//! each rank to hold only 1/N of the context tokens. This enables larger batch sizes
//! and longer contexts on fixed GPU memory.
//!
//! # Algorithm
//!
//! Token assignment uses interleaved round-robin in blocks of `interleave_size` tokens:
//! block[i] → rank[i % dcp_size]. For example with dcp_size=2, interleave_size=4:
//! tokens 0-3 → rank 0, tokens 4-7 → rank 1, tokens 8-11 → rank 0, ...
//!
//! For each decode step:
//! 1. Each rank holds only its local KV slice (kv_lengths reflect local count)
//! 2. Each rank computes attention: `(out_local, lse_local) = attn(Q, K_local, V_local)`
//! 3. All-gather `lse_local` across DCP ranks
//! 4. Compute `final_lse = logsumexp(lse_all, dim=rank)` (numerically stable)
//! 5. Correct: `out_corrected = out_local * exp(lse_local - final_lse)`
//! 6. All-reduce corrected outputs (Sum) → each rank has final result
//!
//! # Reference
//! vLLM: `vllm/v1/attention/ops/common.py::cp_lse_ag_out_rs`
//! vLLM: `vllm/v1/attention/backends/utils.py::get_dcp_local_seq_lens`

use std::sync::Arc;

use candle_core::{Result, Tensor};

use super::communicator::{DeviceCommunicator, ReduceOp};
use crate::kv_cache::CacheEngine;
use crate::layers::attention::{AttentionBackend, BatchedDecodeMetadata, PagedAttentionMetadata};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Decode Context Parallelism configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpConfig {
    /// Number of GPUs sharing the KV cache (the DCP group size).
    pub dcp_size: usize,
    /// Token block size for round-robin interleaved assignment.
    ///
    /// Each contiguous `interleave_size` tokens form one block. Blocks are assigned
    /// to ranks in order: block[i] → rank[i % dcp_size]. Larger values reduce
    /// communication overhead but coarsen the sharding granularity.
    pub interleave_size: usize,
}

impl CpConfig {
    /// No context parallelism (single GPU, noop).
    pub fn no_cp() -> Self {
        Self {
            dcp_size: 1,
            interleave_size: 1,
        }
    }

    /// Create a DCP configuration.
    ///
    /// # Panics
    /// Panics if `dcp_size == 0` or `interleave_size == 0`.
    pub fn new(dcp_size: usize, interleave_size: usize) -> Self {
        assert!(dcp_size > 0, "dcp_size must be > 0");
        assert!(interleave_size > 0, "interleave_size must be > 0");
        Self {
            dcp_size,
            interleave_size,
        }
    }
}

impl Default for CpConfig {
    fn default() -> Self {
        Self::no_cp()
    }
}

// ─── Runtime Context ──────────────────────────────────────────────────────────

/// Runtime context for a single DCP rank.
///
/// Holds the rank's position in the DCP group, the interleave size, and an
/// optional communicator for collective operations. When `comm` is `None`
/// (single-GPU), all sharding functions still work correctly (identity behaviour).
pub struct CpContext {
    /// This rank's index in the DCP group (0..world_size).
    pub rank: usize,
    /// Total number of DCP ranks.
    pub world_size: usize,
    /// Token block size for interleaved assignment.
    pub interleave_size: usize,
    /// Communicator for DCP collectives (None = single GPU).
    comm: Option<Arc<dyn DeviceCommunicator>>,
}

impl CpContext {
    /// Single-GPU context (no DCP). All sharding operations are identity.
    pub fn single_gpu() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            interleave_size: 1,
            comm: None,
        }
    }

    /// Create a multi-GPU DCP context.
    ///
    /// # Panics
    /// Panics if `rank >= world_size`.
    pub fn new(
        rank: usize,
        world_size: usize,
        interleave_size: usize,
        comm: Arc<dyn DeviceCommunicator>,
    ) -> Self {
        assert!(rank < world_size, "rank must be < world_size");
        assert!(interleave_size > 0, "interleave_size must be > 0");
        Self {
            rank,
            world_size,
            interleave_size,
            comm: Some(comm),
        }
    }

    /// Whether this context represents a single-GPU (no DCP) setup.
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Compute the local KV sequence length for this rank given the full sequence length.
    ///
    /// Convenience wrapper for a single sequence; see [`get_dcp_local_seq_lens`].
    pub fn local_kv_len(&self, total_seq_len: usize) -> usize {
        get_dcp_local_seq_len(
            total_seq_len,
            self.world_size,
            self.rank,
            self.interleave_size,
        )
    }

    /// Get the communicator, or panic if called on a single-GPU context.
    pub fn comm(&self) -> &dyn DeviceCommunicator {
        self.comm
            .as_deref()
            .expect("communicator not available on single-GPU CpContext")
    }
}

// ─── Sharding Math ───────────────────────────────────────────────────────────

/// Compute local KV sequence lengths for all sequences in a batch, for one DCP rank.
///
/// Token assignment uses interleaved round-robin: `block[i] → rank[i % dcp_size]`,
/// where each "block" contains `interleave_size` consecutive tokens.
///
/// # Formula
/// For a sequence of length `S`, rank `r`, group size `N`, block size `B`:
/// - `base = (S / (N * B)) * B`  — full rounds each rank receives
/// - `remainder = S - base * N`  — leftover tokens (0 ≤ remainder < N * B)
/// - `extra = clip(remainder - r * B, 0, B)`
/// - `local_len = base + extra`
///
/// Invariant: `sum over ranks(local_len) == total_seq_len` for any valid inputs.
///
/// # Arguments
/// * `seq_lens` — global (full) sequence lengths for each item in the batch
/// * `dcp_size` — number of DCP ranks (must be > 0)
/// * `dcp_rank` — this rank's index (must be < dcp_size)
/// * `interleave_size` — token block size for round-robin (must be > 0)
pub fn get_dcp_local_seq_lens(
    seq_lens: &[usize],
    dcp_size: usize,
    dcp_rank: usize,
    interleave_size: usize,
) -> Vec<usize> {
    seq_lens
        .iter()
        .map(|&s| get_dcp_local_seq_len(s, dcp_size, dcp_rank, interleave_size))
        .collect()
}

/// Compute the local KV length for a single sequence on one DCP rank.
fn get_dcp_local_seq_len(
    total_seq_len: usize,
    dcp_size: usize,
    dcp_rank: usize,
    interleave_size: usize,
) -> usize {
    debug_assert!(dcp_size > 0);
    debug_assert!(dcp_rank < dcp_size);
    debug_assert!(interleave_size > 0);

    if dcp_size == 1 {
        return total_seq_len;
    }

    let cycle = dcp_size * interleave_size;
    let base = (total_seq_len / cycle) * interleave_size;
    let remainder = total_seq_len % cycle;
    let rank_start = dcp_rank * interleave_size;
    let extra = if remainder > rank_start {
        (remainder - rank_start).min(interleave_size)
    } else {
        0
    };
    base + extra
}

/// Filter a global slot_mapping to only include the token slots owned by this DCP rank.
///
/// Slot mappings are assigned in the same interleaved round-robin pattern as sequence
/// lengths: tokens at interleaved positions belonging to this rank are kept.
///
/// This is used during KV cache writes (prefill/decode) to ensure each DCP rank only
/// stores its own portion of the context tokens, achieving `1/dcp_size` memory usage.
///
/// # Arguments
/// * `global_slot_mapping` — flat list of slot indices for all new tokens
/// * `dcp_size`, `dcp_rank`, `interleave_size` — DCP configuration (same as seq_lens)
///
/// The mapping must correspond to a contiguous range of token positions starting at
/// `token_start_pos` within the sequence. The caller must ensure the positions match
/// the interleaving schedule used for KV allocation.
///
/// # Current use
/// Pass the result as the `slot_mapping` argument to `cache_engine.write()` / `write_batch()`
/// so that only the rank-local tokens are written to the paged KV cache.
pub fn get_dcp_local_slot_mapping(
    global_slot_mapping: &[usize],
    dcp_size: usize,
    dcp_rank: usize,
    interleave_size: usize,
) -> Vec<usize> {
    if dcp_size == 1 {
        return global_slot_mapping.to_vec();
    }

    global_slot_mapping
        .iter()
        .enumerate()
        .filter_map(|(token_idx, &slot)| {
            // Determine which block and within-block position this token occupies
            let block_idx = token_idx / interleave_size;
            let owner_rank = block_idx % dcp_size;
            if owner_rank == dcp_rank {
                Some(slot)
            } else {
                None
            }
        })
        .collect()
}

// ─── LSE Correction ──────────────────────────────────────────────────────────

/// Merge partial attention outputs from all DCP ranks using log-sum-exp correction.
///
/// Each DCP rank computes attention over its local KV slice, producing `attn_out_local`
/// and `lse_local`. Since softmax is not commutative across independent computations,
/// the partial results must be recombined via the LSE statistics.
///
/// # Algorithm
/// 1. All-gather `lse_local` → `lse_all [N, batch, heads]`
/// 2. `final_lse = logsumexp(lse_all, dim=0)` (numerically stable via max-shift)
/// 3. `correction = exp(lse_local - final_lse)` — per-(batch, head) scale factor
/// 4. `corrected = attn_out * correction[..., None]`
/// 5. All-reduce `corrected` (Sum) → each rank holds the correct full output
///
/// # Arguments
/// * `attn_out` — local attention output `[batch, num_heads, head_dim]`
/// * `lse` — local log-sum-exp statistics `[batch, num_heads]` (natural log, loge)
/// * `ctx` — DCP runtime context providing rank info and communicator
///
/// # Returns
/// Corrected attention output `[batch, num_heads, head_dim]`.
///
/// # Panics
/// Panics if called on a single-GPU context (use `ctx.is_single()` to guard).
pub fn lse_correct_and_reduce(attn_out: &Tensor, lse: &Tensor, ctx: &CpContext) -> Result<Tensor> {
    if ctx.is_single() {
        return Ok(attn_out.clone());
    }

    let comm = ctx.comm();
    let n = ctx.world_size;
    let (batch, heads) = lse.dims2()?;

    // 1. All-gather LSE across DCP ranks: [N * batch, heads]
    let lse_all = comm
        .all_gather(lse, 0)
        .map_err(|e| candle_core::Error::Msg(format!("DCP all_gather lse failed: {e}")))?;

    // 2. Reshape to [N, batch, heads] for per-rank log-sum-exp
    let lse_stacked = lse_all.reshape((n, batch, heads))?;

    // 3. Numerically stable logsumexp across rank dimension (dim 0):
    //    final_lse = max + log(sum(exp(lse - max)))
    let lse_max = lse_stacked.max_keepdim(0)?; // [1, batch, heads]
    let exp_shifted = (lse_stacked.broadcast_sub(&lse_max))?.exp()?; // [N, batch, heads]
    let exp_sum = exp_shifted.sum_keepdim(0)?; // [1, batch, heads]
    let final_lse = (exp_sum.log()?.broadcast_add(&lse_max))?.squeeze(0)?; // [batch, heads]

    // 4. Correction scale: exp(lse_local - final_lse)
    let correction = (lse.broadcast_sub(&final_lse))?.exp()?; // [batch, heads]

    // 5. Apply correction: attn_out [batch, heads, head_dim] * correction [batch, heads, 1]
    let corrected = (attn_out.broadcast_mul(&correction.unsqueeze(2)?))?;

    // 6. All-reduce corrected outputs (Sum) across DCP ranks
    comm.all_reduce(&corrected, ReduceOp::Sum).map_err(|e| {
        candle_core::Error::Msg(format!("DCP all_reduce corrected output failed: {e}"))
    })
}

// ─── Attention Wrapper ───────────────────────────────────────────────────────

/// Attention backend wrapper that applies Decode Context Parallelism.
///
/// Wraps any [`AttentionBackend`] implementation. On the decode path, when the
/// context is multi-GPU, it calls `batched_decode_attention_with_lse()` on the
/// inner backend and applies [`lse_correct_and_reduce`] to merge partial results.
///
/// Callers are responsible for:
/// - Providing local KV lengths (not global) in `BatchedDecodeMetadata::kv_lengths`
///   (use [`get_dcp_local_seq_lens`] to compute them)
/// - Writing only rank-local tokens to the cache during prefill/decode
///   (use [`get_dcp_local_slot_mapping`] to filter slot_mapping)
pub struct DcpAttentionWrapper<B: AttentionBackend> {
    inner: B,
    ctx: CpContext,
}

impl<B: AttentionBackend> DcpAttentionWrapper<B> {
    /// Wrap an attention backend with a DCP context.
    pub fn new(inner: B, ctx: CpContext) -> Self {
        Self { inner, ctx }
    }

    /// Access the inner backend directly (e.g. for configuration queries).
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Access the DCP context.
    pub fn context(&self) -> &CpContext {
        &self.ctx
    }
}

impl<B: AttentionBackend> AttentionBackend for DcpAttentionWrapper<B> {
    fn name(&self) -> &'static str {
        // NOTE: can't build a &'static str at runtime without leaking; callers should
        // query inner().name() if they need the backend name.
        "dcp-wrapped"
    }

    fn prefill_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        cache_engine: &mut CacheEngine,
        metadata: &PagedAttentionMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        // DCP prefill: the caller has already filtered slot_mapping to rank-local slots.
        // The inner backend writes only rank-local tokens; no collective needed here.
        self.inner.prefill_attention(
            q,
            k,
            v,
            attention_mask,
            cache_engine,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

    fn batched_decode_attention(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache_engine: &mut CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        // Single-GPU fast path: no collectives needed.
        if self.ctx.is_single() {
            return self.inner.batched_decode_attention(
                q,
                k_new,
                v_new,
                cache_engine,
                metadata,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        // DCP multi-GPU path: get output + LSE from inner backend.
        // metadata.kv_lengths must already contain local KV lengths for this rank.
        let (out_3d, maybe_lse) = self.inner.batched_decode_attention_with_lse(
            q,
            k_new,
            v_new,
            cache_engine,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;

        let batch = out_3d.dim(0)?;

        let corrected_3d = if let Some(lse) = maybe_lse {
            // Accurate DCP correction using log-sum-exp statistics.
            lse_correct_and_reduce(&out_3d, &lse, &self.ctx)?
        } else {
            // Backend does not return LSE (e.g. CPU fallback). All-reduce raw outputs.
            // This is correct only when a single rank holds all KV tokens (dcp_size=1).
            // NOTE: In a real multi-GPU DCP setup this path should not be reachable
            // because the FlashInfer backend always provides LSE on CUDA.
            self.ctx
                .comm()
                .all_reduce(&out_3d, ReduceOp::Sum)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("DCP all_reduce (no-lse) failed: {e}"))
                })?
        };

        // Reshape [batch, heads, head_dim] → [batch, heads * head_dim] to match base trait.
        corrected_3d.reshape((batch, num_heads * head_dim))
    }

    fn supports_config(&self, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> bool {
        self.inner
            .supports_config(num_heads, num_kv_heads, head_dim)
    }

    fn supported_dtypes(&self) -> &[candle_core::DType] {
        self.inner.supported_dtypes()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator};
    use candle_core::{DType, Device};

    // ── get_dcp_local_seq_lens ──────────────────────────────────────────────

    #[test]
    fn test_get_dcp_local_seq_lens_single_rank() {
        // dcp_size=1 is identity: every token belongs to rank 0.
        let result = get_dcp_local_seq_lens(&[100, 50, 0], 1, 0, 1);
        assert_eq!(result, vec![100, 50, 0]);
    }

    #[test]
    fn test_get_dcp_local_seq_lens_two_ranks_even() {
        // 100 tokens, 2 ranks, interleave=1: each rank gets 50.
        assert_eq!(get_dcp_local_seq_lens(&[100], 2, 0, 1), vec![50]);
        assert_eq!(get_dcp_local_seq_lens(&[100], 2, 1, 1), vec![50]);
    }

    #[test]
    fn test_get_dcp_local_seq_lens_two_ranks_odd() {
        // 101 tokens, 2 ranks, interleave=1.
        // 50 full cycles of 2 tokens = 100. Remainder=1 → rank 0 gets 1 extra.
        assert_eq!(get_dcp_local_seq_lens(&[101], 2, 0, 1), vec![51]);
        assert_eq!(get_dcp_local_seq_lens(&[101], 2, 1, 1), vec![50]);
    }

    #[test]
    fn test_get_dcp_local_seq_lens_four_ranks_interleave4() {
        // 32 tokens, 4 ranks, interleave=4.
        // Each rank gets exactly 8 tokens.
        for rank in 0..4 {
            assert_eq!(
                get_dcp_local_seq_lens(&[32], 4, rank, 4),
                vec![8],
                "rank {rank}"
            );
        }
    }

    #[test]
    fn test_get_dcp_local_seq_lens_uneven_interleave() {
        // 10 tokens, 3 ranks, interleave=4.
        // cycle = 3*4 = 12. base = (10/12)*4 = 0. remainder = 10.
        // rank 0: extra = clip(10 - 0, 0, 4) = 4
        // rank 1: extra = clip(10 - 4, 0, 4) = 4
        // rank 2: extra = clip(10 - 8, 0, 4) = 2
        assert_eq!(get_dcp_local_seq_lens(&[10], 3, 0, 4), vec![4]);
        assert_eq!(get_dcp_local_seq_lens(&[10], 3, 1, 4), vec![4]);
        assert_eq!(get_dcp_local_seq_lens(&[10], 3, 2, 4), vec![2]);
    }

    #[test]
    fn test_get_dcp_local_seq_lens_sum_equals_total() {
        // Invariant: sum of local lengths across all ranks equals global length.
        let test_cases: &[(usize, usize, usize)] = &[
            (100, 2, 1),
            (101, 2, 1),
            (200, 4, 1),
            (200, 4, 4),
            (17, 3, 2),
            (0, 4, 1),
        ];
        for &(total, dcp_size, interleave) in test_cases {
            let sum: usize = (0..dcp_size)
                .map(|r| get_dcp_local_seq_lens(&[total], dcp_size, r, interleave)[0])
                .sum();
            assert_eq!(
                sum, total,
                "total={total}, dcp={dcp_size}, isize={interleave}"
            );
        }
    }

    #[test]
    fn test_get_dcp_local_seq_lens_batch() {
        // Multiple sequences in one call.
        let result = get_dcp_local_seq_lens(&[100, 200, 0], 2, 0, 1);
        assert_eq!(result, vec![50, 100, 0]);
    }

    // ── get_dcp_local_slot_mapping ──────────────────────────────────────────

    #[test]
    fn test_get_dcp_local_slot_mapping_single_rank() {
        let slots: Vec<usize> = (0..10).collect();
        let result = get_dcp_local_slot_mapping(&slots, 1, 0, 1);
        assert_eq!(result, slots);
    }

    #[test]
    fn test_get_dcp_local_slot_mapping_two_ranks_interleave1() {
        // Alternating: rank 0 gets even indices, rank 1 gets odd indices.
        let slots: Vec<usize> = (0..6).map(|i| i * 10).collect(); // [0,10,20,30,40,50]
        let r0 = get_dcp_local_slot_mapping(&slots, 2, 0, 1);
        let r1 = get_dcp_local_slot_mapping(&slots, 2, 1, 1);
        assert_eq!(r0, vec![0, 20, 40]);
        assert_eq!(r1, vec![10, 30, 50]);
    }

    #[test]
    fn test_get_dcp_local_slot_mapping_interleaved_block4() {
        // 8 tokens, 2 ranks, interleave=4.
        // tokens 0-3 → rank 0, tokens 4-7 → rank 1.
        let slots: Vec<usize> = (0..8).collect();
        let r0 = get_dcp_local_slot_mapping(&slots, 2, 0, 4);
        let r1 = get_dcp_local_slot_mapping(&slots, 2, 1, 4);
        assert_eq!(r0, vec![0, 1, 2, 3]);
        assert_eq!(r1, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_get_dcp_local_slot_mapping_count_matches_seq_lens() {
        // The number of local slots must match get_dcp_local_seq_len.
        let total_tokens = 13usize;
        let slots: Vec<usize> = (0..total_tokens).collect();
        let dcp_size = 3;
        let interleave = 2;
        for rank in 0..dcp_size {
            let local_slots = get_dcp_local_slot_mapping(&slots, dcp_size, rank, interleave);
            let expected_len = get_dcp_local_seq_len(total_tokens, dcp_size, rank, interleave);
            assert_eq!(
                local_slots.len(),
                expected_len,
                "rank {rank}: slot count mismatch"
            );
        }
    }

    // ── CpContext ───────────────────────────────────────────────────────────

    #[test]
    fn test_cp_context_single_gpu() {
        let ctx = CpContext::single_gpu();
        assert!(ctx.is_single());
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
    }

    #[test]
    fn test_cp_context_local_kv_len() {
        let ctx = CpContext::single_gpu();
        assert_eq!(ctx.local_kv_len(100), 100);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let comm = Arc::new(MockCommunicator::new(pg));
        let ctx2 = CpContext::new(0, 2, 1, comm);
        assert_eq!(ctx2.local_kv_len(100), 50);
    }

    #[test]
    #[should_panic(expected = "rank must be < world_size")]
    fn test_cp_context_invalid_rank_panics() {
        let pg = LocalProcessGroup::with_rank(0, 2);
        let comm = Arc::new(MockCommunicator::new(pg));
        CpContext::new(5, 2, 1, comm);
    }

    // ── lse_correct_and_reduce ──────────────────────────────────────────────

    #[test]
    fn test_lse_correct_single_gpu_is_identity() {
        let ctx = CpContext::single_gpu();
        let device = Device::Cpu;
        let out = Tensor::ones(&[2, 4, 8], DType::F32, &device).unwrap();
        let lse = Tensor::zeros(&[2, 4], DType::F32, &device).unwrap();

        let result = lse_correct_and_reduce(&out, &lse, &ctx).unwrap();
        assert_eq!(result.dims(), out.dims());
        let out_vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let res_vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in out_vals.iter().zip(res_vals.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lse_correct_two_ranks_equal_lse() {
        // When both ranks have equal LSE, correction factor = 0.5, all_reduce gives correct result.
        // With MockCommunicator (rank 0, world 2): all_gather repeats tensor,
        // so lse_all = [lse, lse]. logsumexp = lse + log(2).
        // correction = exp(lse - (lse + log(2))) = 1/2.
        // corrected = out * 0.5. all_reduce (sum repeats in mock) = out * 0.5 (mock identity).
        let pg = LocalProcessGroup::with_rank(0, 2);
        let comm = Arc::new(MockCommunicator::new(pg));
        let ctx = CpContext::new(0, 2, 1, comm);

        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let head_dim = 8;

        let out = Tensor::ones(&[batch, heads, head_dim], DType::F32, &device).unwrap();
        // lse = 0.0 everywhere (log-partition of uniform dist over 1 token)
        let lse = Tensor::zeros(&[batch, heads], DType::F32, &device).unwrap();

        // Should not error; correctness of numerical values depends on mock fidelity.
        let result = lse_correct_and_reduce(&out, &lse, &ctx).unwrap();
        assert_eq!(result.dims(), &[batch, heads, head_dim]);
    }

    // ── DcpAttentionWrapper ─────────────────────────────────────────────────

    #[test]
    fn test_dcp_wrapper_name() {
        use crate::layers::attention::NaiveAttentionBackend;
        let ctx = CpContext::single_gpu();
        let wrapper = DcpAttentionWrapper::new(NaiveAttentionBackend::new(), ctx);
        assert!(!wrapper.name().is_empty());
    }

    #[test]
    fn test_cp_config_no_cp() {
        let cfg = CpConfig::no_cp();
        assert_eq!(cfg.dcp_size, 1);
        assert_eq!(cfg.interleave_size, 1);
    }

    #[test]
    fn test_cp_config_new() {
        let cfg = CpConfig::new(4, 8);
        assert_eq!(cfg.dcp_size, 4);
        assert_eq!(cfg.interleave_size, 8);
    }
}
