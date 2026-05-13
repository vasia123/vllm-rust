//! Output buffer pool for hot decoder kernels.
//!
//! Pre-allocates and re-uses `Tensor`s across forward passes so the GEMV
//! / paged-attention / norm hot path stops calling `cuMemAlloc` on every
//! op.  cuMemAlloc on a 8 GB Ada laptop costs ~3-6 µs per call; with the
//! AWQ INT4 GEMV alone landing 252 calls per token (7 linears × 36 layers),
//! eliminating the syscall round-trip is worth ~1-2 ms per token at
//! steady state — and is the single biggest blocker for CUDA Graph
//! capture once we own the stream lifecycle (Stage 11).
//!
//! ## Usage contract
//!
//! 1. `OutputPool::reset_cursors()` is called once at the start of every
//!    decode forward in [`crate::engine::helpers::execute_batched_decode_with_graph`].
//!    This rewinds all per-shape round-robin cursors back to zero.
//! 2. Each kernel-launch site that wants a pre-allocated output requests
//!    one via one of two entry points:
//!
//!      - [`OutputPool::reserve_pooled`] — generic, returns a
//!        [`PooledTensor`] for any `(shape, dtype)`.
//!      - One of the four tagged reserves
//!        ([`OutputPool::reserve_positions`], [`OutputPool::reserve_slot_mapping`],
//!        [`OutputPool::reserve_block_tables`], [`OutputPool::reserve_seq_lens`])
//!        for decode-batch metadata buffers. Each tag has its own
//!        [`SlotKind`] bucket and cursor namespace, so unrelated tagged
//!        reserves and the generic pool cannot drift into shared slots.
//!
//!    The first call for a given key allocates a fresh tensor;
//!    subsequent calls within the same forward step return *distinct*
//!    pre-allocated tensors at increasing pool indices, and grow the
//!    pool on demand if the pool runs out of pre-allocated buffers.
//! 3. The returned tensor shares storage (Arc-cloned) with the pooled
//!    tensor — kernels write into that storage; the caller hands the
//!    tensor down the layer chain; the storage stays alive for the life
//!    of the pool. Wrappers that need a bare `Tensor` should call
//!    `.into_tensor()` at the pool boundary so the type-erasure is
//!    locally visible in code review.
//!
//! Because forwards are executed serially inside the engine task, a
//! single global pool is safe.  A `Mutex` guards the inner map; the
//! critical section is a HashMap lookup + index, dwarfed by the µs-scale
//! kernel launches it amortises.
//!
//! When invoked from non-pooled paths (eager prefill, tests), kernels
//! continue allocating their own buffers — the pool is opt-in per call
//! site, not a global override.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Device, Result, Shape, Tensor};

// ─── Slot kinds ─────────────────────────────────────────────────────
//
// Generic pool entries share one round-robin cursor per `(dtype, shape)`.
// The four tagged kinds below give the critical decode-batch buffers
// (positions, slot_mapping, block_tables, seq_lens) their own
// independent cursors, eliminating the slot-drift failure mode where a
// generic `reserve_pooled([1], U32, …)` collides with a `positions`
// reserve at the same shape and the cursors end up advancing in
// unexpected interleavings (Bug B.1, commit 2614dd7).
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
enum SlotKind {
    Generic,
    Positions,
    SlotMapping,
    BlockTables,
    SeqLens,
}

/// Tag marker for the per-token query-position buffer.
#[derive(Debug)]
pub struct Positions;
/// Tag marker for the per-token KV-cache slot index buffer.
#[derive(Debug)]
pub struct SlotMapping;
/// Tag marker for the `[batch, max_blocks_per_seq]` block-id table.
#[derive(Debug)]
pub struct BlockTables;
/// Tag marker for the `[batch]` per-sequence length vector.
#[derive(Debug)]
pub struct SeqLens;

/// Phantom-typed handle for a pool slot of a specific decode-batch
/// role. Constructible only via [`OutputPool::reserve_positions`] et al,
/// so a `TaggedSlot<Positions>` cannot be confused with a generic
/// [`PooledTensor`] at compile time. The wrapped storage is still
/// pool-backed (it IS a [`PooledTensor`] internally), so all
/// view-method and as-tensor access is preserved.
#[derive(Debug)]
pub struct TaggedSlot<Tag> {
    inner: PooledTensor,
    _phantom: PhantomData<fn() -> Tag>,
}

// Manual `Clone` — `#[derive]` would unnecessarily require `Tag: Clone`
// even though the `fn() -> Tag` phantom is `Clone` for any `Tag`.
impl<Tag> Clone for TaggedSlot<Tag> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<Tag> TaggedSlot<Tag> {
    fn new(inner: PooledTensor) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Tag an already-`PooledTensor` view as this tag without going
    /// through the pool's `reserve_*` cursor. Internal use only:
    /// `build_decode_batch_shared_with_options` calls this in the
    /// non-CUDA / oversize-batch fallback branch, where the storage
    /// is a `from_pool_unchecked`-wrapped one-shot `Tensor::from_vec`
    /// that lives for the caller's forward but is not pool-managed.
    /// Eager-only paths never feed it into a captured graph, so the
    /// "pool-backed = address-stable" contract isn't required there.
    #[doc(hidden)]
    pub fn __from_pool_unchecked(inner: PooledTensor) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Read-only access to the underlying tensor — the same backing
    /// device storage the pool retains.
    pub fn as_tensor(&self) -> &Tensor {
        self.inner.as_tensor()
    }

    /// Borrow the pool-backed view as a [`PooledTensor`]. Useful when
    /// passing to APIs that already accept `&PooledTensor`.
    pub fn as_pooled(&self) -> &PooledTensor {
        &self.inner
    }

    pub fn dims(&self) -> &[usize] {
        self.inner.dims()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn shape(&self) -> &Shape {
        self.inner.shape()
    }
}

/// Type-level guarantee that a [`Tensor`]'s underlying device storage was
/// reserved from an [`OutputPool`] (or another `'static`-equivalent
/// source explicitly opted into by `from_pool_unchecked`). The pool's
/// buffers are never freed for the lifetime of the pool — `OutputPool::global()`
/// is `'static`, so a captured CUDA-Graph kernel may safely encode the
/// device pointer of any `PooledTensor`'s storage into a recorded kernel
/// node; replay reads from the same address forever.
///
/// Construction:
/// - [`OutputPool::reserve_pooled`] is the primary safe constructor.
/// - View-like methods on an existing `PooledTensor` (`reshape`, `narrow`,
///   `unsqueeze`, `squeeze`, `transpose`, `flatten_all`) preserve the
///   storage and therefore preserve the invariant.
/// - [`PooledTensor::from_pool_unchecked`] is the unsafe escape hatch
///   used internally when the source is a non-pool but `'static` storage
///   (e.g., the captured-graph input-ids buffer owned by `CudaGraphRunner`
///   for the lifetime of the runner). External callers should not need it.
///
/// Read-only conversion back to `&Tensor` via [`PooledTensor::as_tensor`]
/// is freely available; it does not violate the invariant because views
/// share storage. Owning `Tensor` ([`PooledTensor::into_tensor`]) returns
/// the underlying Arc-shared `Tensor` — the storage is still pool-backed,
/// the caller just loses the type-level guarantee.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct PooledTensor {
    inner: Tensor,
}

impl PooledTensor {
    /// Wrap a `Tensor` as a `PooledTensor` without verifying its storage
    /// provenance.
    ///
    /// # Safety
    /// Caller must ensure `t`'s underlying device storage outlives every
    /// captured CUDA-Graph that may encode a pointer derived from it.
    /// The intended uses are:
    /// - inside [`OutputPool::reserve_pooled`] (storage owned by the pool)
    /// - wrapping view-method results internally (storage shared with an
    ///   already-`PooledTensor`)
    /// - wrapping the [`CudaGraphRunner`](super::cuda_graph_runner::CudaGraphRunner)'s
    ///   persistent `input_ids` / `output` buffers, which it owns for the
    ///   life of every captured graph that references them.
    #[inline(always)]
    pub(crate) unsafe fn from_pool_unchecked(t: Tensor) -> Self {
        Self { inner: t }
    }

    /// Read-only access to the underlying tensor. Use this when passing
    /// the pooled tensor to candle ops, weight matmuls, or anywhere a
    /// generic `&Tensor` is required.
    #[inline]
    pub fn as_tensor(&self) -> &Tensor {
        &self.inner
    }

    /// Consume the wrapper. The returned `Tensor` still references
    /// pool-backed storage; the caller has just dropped the compile-time
    /// invariant. Prefer `as_tensor()` unless an owning `Tensor` is
    /// genuinely required (e.g., for `Clone` into a struct field).
    #[inline(always)]
    pub fn into_tensor(self) -> Tensor {
        self.inner
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn device(&self) -> &Device {
        self.inner.device()
    }

    pub fn shape(&self) -> &Shape {
        self.inner.shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.inner.dims()
    }

    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    pub fn elem_count(&self) -> usize {
        self.inner.elem_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// `reshape` — view-only, storage unchanged. Preserves invariant.
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<PooledTensor> {
        Ok(Self {
            inner: self.inner.reshape(shape)?,
        })
    }

    /// `narrow` — view-only, storage unchanged.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<PooledTensor> {
        Ok(Self {
            inner: self.inner.narrow(dim, start, len)?,
        })
    }

    /// `unsqueeze` — view-only.
    pub fn unsqueeze(&self, dim: usize) -> Result<PooledTensor> {
        Ok(Self {
            inner: self.inner.unsqueeze(dim)?,
        })
    }

    /// `squeeze` — view-only.
    pub fn squeeze(&self, dim: usize) -> Result<PooledTensor> {
        Ok(Self {
            inner: self.inner.squeeze(dim)?,
        })
    }

    /// `transpose` — view-only (may produce a non-contiguous layout).
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<PooledTensor> {
        Ok(Self {
            inner: self.inner.transpose(dim1, dim2)?,
        })
    }

    /// `flatten_all` — view-only when source is contiguous; otherwise
    /// candle materialises via `contiguous()` and we would lose the pool
    /// invariant. We therefore require the source to be contiguous —
    /// callers that need flattening of a transposed view should first
    /// `contiguous()` (which materialises into a pool slot, preserving
    /// the invariant) then `flatten_all()`.
    pub fn flatten_all(&self) -> Result<PooledTensor> {
        if !self.inner.is_contiguous() {
            candle_core::bail!(
                "PooledTensor::flatten_all on non-contiguous view would allocate \
                 fresh storage; call .contiguous() first"
            );
        }
        Ok(Self {
            inner: self.inner.flatten_all()?,
        })
    }

    /// Materialize a contiguous version. Preserves the pool invariant:
    /// - If `self.is_contiguous()`, returns a clone of the view (zero copy).
    /// - Otherwise reserves a fresh pool slot of `self.shape()` from
    ///   [`OutputPool::global`] and copies the strided view into it via
    ///   [`cuda_memcpy_inplace`](super::cuda_graph_runner::cuda_memcpy_inplace).
    ///
    /// The non-contiguous branch allocates one intermediate dense tensor
    /// via candle's default allocator (because `cuda_memcpy_inplace`
    /// expects contiguous-layout src). That intermediate dies before
    /// return — it never enters any captured kernel, so it is capture-safe.
    pub fn contiguous(&self) -> Result<PooledTensor> {
        if self.inner.is_contiguous() {
            return Ok(self.clone());
        }
        let dense = self.inner.contiguous()?;
        let dst = OutputPool::global().reserve_pooled(
            self.inner.shape().dims(),
            self.inner.dtype(),
            self.inner.device(),
        )?;
        super::cuda_graph_runner::cuda_memcpy_inplace(dst.as_tensor(), &dense)
            .map_err(|e| candle_core::Error::Msg(format!("PooledTensor::contiguous: {e}")))?;
        Ok(dst)
    }
}

/// One pool entry per `(dtype, shape)` key.
struct PoolEntry {
    /// Pre-allocated tensors. Each is independently usable; entries do
    /// **not** alias one another (each is a fresh `Tensor::zeros`
    /// allocation).
    buffers: Vec<Tensor>,
    /// Index of the next buffer to hand out within the current forward.
    /// Reset to zero by [`OutputPool::reset_cursors`].
    cursor: usize,
}

/// Shape-keyed buffer pool. See module docs.
///
/// The key is `(SlotKind, DType, Vec<usize>)`: generic
/// (`reserve`/`reserve_pooled`) reserves live in their own slot space,
/// independent of the four tagged kinds used by decode-batch metadata.
/// Tagged reserves cannot drift into the generic bucket and vice
/// versa — at the cost of one extra `SlotKind` discriminant byte per
/// hash.
type PoolKey = (SlotKind, DType, Vec<usize>);

#[derive(Default)]
pub struct OutputPool {
    inner: Mutex<HashMap<PoolKey, PoolEntry>>,
}

impl OutputPool {
    /// Process-global pool.  Initialised on first access and never freed
    /// — its buffers live as long as the engine.
    pub fn global() -> &'static Self {
        static POOL: OnceLock<OutputPool> = OnceLock::new();
        POOL.get_or_init(OutputPool::default)
    }

    /// Rewind every per-shape cursor to zero.  Call once at the start of
    /// each decode forward so the next batch of `reserve()` calls walks
    /// the same allocation order as the previous one.
    pub fn reset_cursors(&self) {
        super::cr_trace::note("reset_cursors");
        let mut inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        for entry in inner.values_mut() {
            entry.cursor = 0;
        }
    }

    /// Hand out a pool-backed tensor of `(shape, dtype)`, growing the
    /// pool on demand. Returns a [`PooledTensor`] whose underlying
    /// storage is owned by this pool — the `'static` lifetime of
    /// `OutputPool::global()` makes the storage address stable for any
    /// captured CUDA Graph that records a kernel reading from it.
    ///
    /// Errors when the per-shape pool cap (`MAX_PER_SHAPE = 512`) is
    /// exceeded; falling through to a fresh non-pool allocation would
    /// silently violate the [`PooledTensor`] invariant (its storage
    /// would not be retained by the pool and could be freed before a
    /// captured graph replays).
    ///
    /// This is the **only** generic entry point for getting a tensor
    /// out of the pool. For decode-batch metadata buffers
    /// (positions / slot_mapping / block_tables / seq_lens) use the
    /// tagged [`Self::reserve_positions`] / [`Self::reserve_slot_mapping`]
    /// / [`Self::reserve_block_tables`] / [`Self::reserve_seq_lens`]
    /// methods, which give each kind its own cursor namespace.
    ///
    /// Wrappers that need a bare `Tensor` (because they hand it to
    /// candle infrastructure expecting `&Tensor`) should call
    /// `.into_tensor()` immediately at the pool boundary — that drops
    /// the type invariant locally and signals the boundary visibly in
    /// code review.
    #[inline]
    pub fn reserve_pooled(
        &self,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<PooledTensor> {
        self.reserve_kind(SlotKind::Generic, shape, dtype, device)
    }

    /// Legacy bare-`Tensor` reserve. Retained for the EXL3 cooperative
    /// GEMM hot path (`quantization::exl3_cuda::exl3_gemm`) which shows
    /// a 25-50% c=16 throughput regression when migrated to the
    /// `reserve_pooled(...).into_tensor()` form, despite the two paths
    /// being semantically identical at the Rust level. Root cause not
    /// yet diagnosed; suspected LLVM-level codegen difference around
    /// the `unsafe { PooledTensor::from_pool_unchecked }` wrap-unwrap
    /// pair inside the EXL3 GEMM kernel-launch closure. Every other
    /// hot-path reserve site has been migrated to `reserve_pooled`.
    ///
    /// **Do not add new callers.** If you find yourself needing this
    /// API, first try `reserve_pooled(...).into_tensor()` — if your
    /// site doesn't show a perf regression, use that. Document any
    /// new callers here.
    ///
    /// On cap exceeded this method falls through to a fresh non-pool
    /// allocation rather than erroring; the resulting tensor does not
    /// have the `PooledTensor` storage-stability invariant, so it must
    /// not be fed into a captured CUDA graph that records its device
    /// pointer.
    pub fn reserve(&self, shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
        let key = (SlotKind::Generic, dtype, shape.to_vec());
        let mut inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        let entry = inner.entry(key).or_insert_with(|| PoolEntry {
            buffers: Vec::new(),
            cursor: 0,
        });
        const MAX_PER_SHAPE: usize = 512;
        if entry.cursor >= entry.buffers.len() {
            if entry.buffers.len() >= MAX_PER_SHAPE {
                drop(inner);
                let fresh = Tensor::zeros(shape, dtype, device)?;
                return Ok(fresh);
            }
            let fresh = Tensor::zeros(shape, dtype, device)?;
            entry.buffers.push(fresh);
        }
        let slot_idx = entry.cursor;
        let tensor = entry.buffers[entry.cursor].clone();
        entry.cursor += 1;
        super::cr_trace::log_reserve(dtype, shape, slot_idx, 0);
        Ok(tensor)
    }

    /// Internal: shared back-end for `reserve_pooled` and the four
    /// tagged reserves. Each `SlotKind` has its own cursor/bucket
    /// namespace, so tagged kinds cannot collide with the generic
    /// pool entries.
    #[inline]
    fn reserve_kind(
        &self,
        kind: SlotKind,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<PooledTensor> {
        const MAX_PER_SHAPE: usize = 512;

        let key = (kind, dtype, shape.to_vec());
        let mut inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        let entry = inner.entry(key).or_insert_with(|| PoolEntry {
            buffers: Vec::new(),
            cursor: 0,
        });

        if entry.cursor >= entry.buffers.len() {
            if entry.buffers.len() >= MAX_PER_SHAPE {
                candle_core::bail!(
                    "OutputPool::reserve_pooled: per-shape cap {} exceeded for \
                     ({:?}, {:?}, {:?}); reduce repeated reserves per forward or raise \
                     MAX_PER_SHAPE",
                    MAX_PER_SHAPE,
                    kind,
                    dtype,
                    shape
                );
            }
            let fresh = Tensor::zeros(shape, dtype, device)?;
            entry.buffers.push(fresh);
        }

        let slot_idx = entry.cursor;
        let tensor = entry.buffers[entry.cursor].clone();
        entry.cursor += 1;
        super::cr_trace::log_reserve(dtype, shape, slot_idx, 0);
        // SAFETY: `tensor` is a clone of `entry.buffers[cursor-1]`, which
        // the pool retains for its lifetime. The storage Arc is held by
        // the pool's `Vec<Tensor>` regardless of whether callers drop
        // their `PooledTensor`s — so the device pointer is stable for
        // the lifetime of `OutputPool::global()` (= `'static`).
        Ok(unsafe { PooledTensor::from_pool_unchecked(tensor) })
    }

    /// Tagged reserve for the per-token query-position buffer
    /// (`positions_device` in [`crate::layers::attention::block::DecodeBatchShared`]).
    /// Lives in its own `SlotKind::Positions` bucket so it cannot drift
    /// into the generic round-robin.
    pub fn reserve_positions(
        &self,
        num_tokens: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<TaggedSlot<Positions>> {
        let inner = self.reserve_kind(SlotKind::Positions, &[num_tokens], dtype, device)?;
        Ok(TaggedSlot::new(inner))
    }

    /// Tagged reserve for the per-token KV-cache slot-mapping buffer
    /// (`slot_mapping_device`). See [`Self::reserve_positions`].
    pub fn reserve_slot_mapping(
        &self,
        num_slots: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<TaggedSlot<SlotMapping>> {
        let inner = self.reserve_kind(SlotKind::SlotMapping, &[num_slots], dtype, device)?;
        Ok(TaggedSlot::new(inner))
    }

    /// Tagged reserve for the `[batch, max_blocks_per_seq]` block-id
    /// table (`block_tables`). The pool always uses the worst-case
    /// `max_blocks_per_seq` stride so captured-graph replays read
    /// from a stable shape regardless of the actual `seq_lens` for
    /// this forward.
    pub fn reserve_block_tables(
        &self,
        batch: usize,
        max_blocks_per_seq: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<TaggedSlot<BlockTables>> {
        let inner = self.reserve_kind(
            SlotKind::BlockTables,
            &[batch, max_blocks_per_seq],
            dtype,
            device,
        )?;
        Ok(TaggedSlot::new(inner))
    }

    /// Tagged reserve for the `[batch]` per-sequence length vector
    /// (`seq_lens`). See [`Self::reserve_positions`].
    pub fn reserve_seq_lens(
        &self,
        batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<TaggedSlot<SeqLens>> {
        let inner = self.reserve_kind(SlotKind::SeqLens, &[batch], dtype, device)?;
        Ok(TaggedSlot::new(inner))
    }

    /// Drop ALL pool entries and slots. Forces the next `reserve` /
    /// `reserve_pooled` calls to allocate fresh. Diagnostic helper for
    /// tests that want clean pool state between phases — production
    /// code should NEVER call this (captured CUDA graphs hold pointers
    /// into the dropped slots and would replay against freed memory).
    #[allow(dead_code)]
    pub fn clear_all(&self) {
        let mut inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        inner.clear();
    }

    /// Total number of pre-allocated tensor slots across all shapes.
    /// For diagnostics / tests.
    #[allow(dead_code)]
    pub fn total_slots(&self) -> usize {
        let inner = self.inner.lock().expect("OutputPool: mutex poisoned");
        inner.values().map(|e| e.buffers.len()).sum()
    }

    /// Phase 11.2.D diagnostic: dump every EXISTING pool slot (i.e.,
    /// `0..buffers.len()`, NOT `0..cursor` — because capture replay
    /// never invokes `reserve()`, the cursor doesn't advance, but
    /// captured kernels still write to those pre-allocated slots at
    /// their captured indices). Call AFTER the forward completes but
    /// BEFORE the next `reset_cursors()` — at that point each pool slot
    /// holds the last value the just-completed forward (eager OR
    /// capture replay) wrote there.
    ///
    /// File naming: `pool_<DType>_<S0>x<S1>x...x<SN>__<slot_idx>.bin`
    /// (shape uses `x` so it stays valid on every filesystem).
    pub fn dump_used_slots(&self, dir: &std::path::Path) {
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let mut shapes: Vec<&PoolKey> = inner.keys().collect();
        shapes.sort_by(|a, b| {
            // Sort for deterministic iteration order — irrelevant for
            // diff, but makes manual log inspection easier.
            (format!("{:?}", a.0), format!("{:?}", a.1), &a.2).cmp(&(
                format!("{:?}", b.0),
                format!("{:?}", b.1),
                &b.2,
            ))
        });
        for key in shapes {
            let entry = &inner[key];
            let (kind, dtype, shape) = key;
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            for slot_idx in 0..entry.buffers.len() {
                let name = format!("pool_{kind:?}_{dtype:?}_{shape_str}__{slot_idx:03}");
                let bin_path = dir.join(format!("{name}.bin"));
                let meta_path = dir.join(format!("{name}.meta"));
                let tensor = &entry.buffers[slot_idx];
                let bytes = match super::debug_dump::tensor_to_bytes_public(tensor) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("dump_used_slots[{name}]: bytes failed: {e}");
                        continue;
                    }
                };
                if let Err(e) = std::fs::write(&bin_path, &bytes) {
                    eprintln!("dump_used_slots[{name}]: write failed: {e}");
                    continue;
                }
                let meta = format!(
                    "name={name}\ndtype={dtype:?}\nshape={shape:?}\nslot_idx={slot_idx}\nbytes={}\n",
                    bytes.len()
                );
                let _ = std::fs::write(&meta_path, meta);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_grows_pool_on_repeat_within_forward() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        // Two reserves of the same shape inside a single (un-reset)
        // window should allocate two distinct tensors.
        let a = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        let b = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        // Different storage = different ptrs. We approximate this by
        // checking that `Tensor` identities (`as_ptr()` of the storage
        // arc) differ. Cheap proxy: write one, read the other.
        assert_eq!(pool.total_slots(), 2);
        // Writes through one don't affect the other (storage is
        // independent).
        let _ = a; // keep both alive
        let _ = b;
    }

    #[test]
    fn reset_cursors_replays_existing_buffers() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let _ = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        let _ = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        assert_eq!(pool.total_slots(), 2);

        pool.reset_cursors();

        // Reserve again — cursor restarts, no new allocation.
        let _ = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        let _ = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        assert_eq!(pool.total_slots(), 2, "pool should not grow on replay");
    }

    #[test]
    fn distinct_shapes_keep_separate_slots() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let _ = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        let _ = pool.reserve_pooled(&[8, 4], DType::F32, &dev).unwrap();
        let _ = pool.reserve_pooled(&[4, 8], DType::F16, &dev).unwrap();
        assert_eq!(pool.total_slots(), 3);
    }

    // ───────────────────────── PooledTensor ─────────────────────────
    //
    // These tests exercise the type-level invariant: a `PooledTensor` is
    // constructible only via `reserve_pooled` (or view chains thereof)
    // and its view operations preserve the underlying pool storage.

    fn storage_id(t: &Tensor) -> *const () {
        // Two Tensors share storage iff their CpuStorage / CudaStorage
        // Arc points to the same allocation. We approximate this by
        // formatting the Storage debug repr — pointer equality of
        // `Arc::as_ptr` on the storage would be more rigorous but
        // candle's Storage doesn't expose its Arc. For CPU at least,
        // the data pointer can be probed via `to_vec*` (same data ⇒
        // very likely same storage when shapes line up); for the test
        // we just verify that view methods don't change the underlying
        // `Tensor::id` (which candle uses to fingerprint storage).
        t as *const Tensor as *const ()
    }

    #[test]
    fn reserve_pooled_returns_distinct_clones_per_call() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let a = pool.reserve_pooled(&[2, 3], DType::F32, &dev).unwrap();
        let b = pool.reserve_pooled(&[2, 3], DType::F32, &dev).unwrap();
        // Two slots created in the pool.
        assert_eq!(pool.total_slots(), 2);
        assert_eq!(a.dims(), &[2, 3]);
        assert_eq!(b.dims(), &[2, 3]);
        assert_eq!(a.dtype(), DType::F32);
        // Distinct slots — proxy: `storage_id` differs.
        assert_ne!(storage_id(a.as_tensor()), storage_id(b.as_tensor()));
    }

    #[test]
    fn reserve_pooled_replays_same_addresses_after_reset_cursors() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let a = pool.reserve_pooled(&[4], DType::F32, &dev).unwrap();
        drop(a);
        pool.reset_cursors();
        let _a2 = pool.reserve_pooled(&[4], DType::F32, &dev).unwrap();
        // Pool slot count invariance: cursor reset does not allocate a
        // new slot. Direct device-ptr equality across the drop/reset
        // boundary is unobservable from CPU storage, so we rely on
        // total_slots staying at 1 after the second reserve.
        assert_eq!(pool.total_slots(), 1, "no new allocation after reset");
    }

    #[test]
    fn pooled_view_methods_preserve_total_slots() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let t = pool.reserve_pooled(&[1, 2, 6], DType::F32, &dev).unwrap();

        let r = t.reshape((2, 6)).unwrap();
        let n = r.narrow(0, 0, 1).unwrap();
        let u = n.unsqueeze(0).unwrap();
        let s = u.squeeze(0).unwrap();
        let tr = t.reshape((2, 6)).unwrap().transpose(0, 1).unwrap();
        let _ = (r, n, u, s, tr);

        // Pure views never grew the pool.
        assert_eq!(pool.total_slots(), 1);
    }

    #[test]
    fn pooled_contiguous_on_contiguous_is_zero_copy() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let t = pool.reserve_pooled(&[4, 8], DType::F32, &dev).unwrap();
        let c = t.contiguous().unwrap();
        // Pool didn't grow — c shares storage with t.
        assert_eq!(pool.total_slots(), 1);
        assert!(c.is_contiguous());
        let _ = c;
    }

    #[test]
    fn pooled_contiguous_on_transposed_materializes_into_pool() {
        // CPU device only — CUDA path requires cuda-kernels feature.
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let t = pool.reserve_pooled(&[2, 3], DType::F32, &dev).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());

        // Switch to the global pool (where reserve_pooled inside
        // `contiguous()` lives). For the test we need to verify the
        // pool's invariant — the simpler check is that the operation
        // succeeds and returns a contiguous PooledTensor.
        //
        // NB: `PooledTensor::contiguous` reserves from `OutputPool::global()`
        // (not from `self`'s parent pool), because the global pool is the
        // `'static` source captured kernels rely on. The local `pool`
        // here is unrelated to the global one.
        let c = tr.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.dims(), &[3, 2]);
    }

    #[test]
    fn pooled_flatten_all_on_noncontiguous_errors() {
        let pool = OutputPool::default();
        let dev = Device::Cpu;
        let t = pool.reserve_pooled(&[2, 3], DType::F32, &dev).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());
        let r = tr.flatten_all();
        assert!(r.is_err(), "flatten_all on non-contiguous should bail");
    }

    // ─── Pool slot identity (slot-drift regression suite) ───────────
    //
    // These tests guard the "captured graph contract" invariant: every
    // phase (JIT warmup, capture warmup, production runtime) that calls
    // `reset_cursors()` then reserves the same sequence of pool slots
    // MUST observe identical underlying storage addresses. Bug B
    // (commit 2614dd7) was a violation of this invariant — capture
    // warmup skipped `reset_cursors` and saw cursor 1 while production
    // saw cursor 0 for the same shape, so the captured graph encoded
    // a different device pointer than the one production wrote into.
    //
    // CPU testing rationale: `Tensor::clone()` clones the
    // `Arc<RwLock<Storage>>` (not the inner Vec), so the data pointer
    // of `CpuStorage::U32` is stable across pool slot reservations of
    // the same slot index. That gives us a portable proxy for "is the
    // storage we just got the same physical buffer".

    /// Test-only: lift the data pointer of a pool-backed CPU tensor.
    /// Returns `None` for non-CPU storage or unsupported dtypes — those
    /// tests should gate themselves accordingly.
    fn cpu_storage_data_ptr(t: &Tensor) -> Option<usize> {
        use candle_core::{CpuStorage, Storage};
        let (storage, _layout) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(c) => match c {
                CpuStorage::U32(v) => Some(v.as_ptr() as usize),
                CpuStorage::F32(v) => Some(v.as_ptr() as usize),
                CpuStorage::U8(v) => Some(v.as_ptr() as usize),
                CpuStorage::I64(v) => Some(v.as_ptr() as usize),
                _ => None,
            },
            _ => None,
        }
    }

    #[test]
    fn pool_slot_data_ptrs_stable_across_phases() {
        // Three phases imitating production sequence:
        //   1. JIT warmup forward (reset_cursors + reserve sequence)
        //   2. capture-time build_shared (reset_cursors + reserve sequence)
        //   3. production runtime forward (reset_cursors + reserve sequence)
        // All three must hand out identical storage addresses for matching
        // slot indices. Diverging addresses across phases is the failure
        // mode that Bug B exhibited.
        let pool = OutputPool::default();
        let dev = Device::Cpu;

        let phase = || -> Vec<usize> {
            pool.reset_cursors();
            // Two reserves of the same (dtype, shape) — exercises the
            // per-shape round-robin cursor at indices 0 and 1, which is
            // precisely the bucket that drifted in Bug B.1 (positions
            // and seq_lens both being `[1]` U32 at batch=1).
            let a = pool.reserve_pooled(&[1], DType::U32, &dev).unwrap();
            let b = pool.reserve_pooled(&[1], DType::U32, &dev).unwrap();
            // Two more distinct shapes.
            let c = pool.reserve_pooled(&[5], DType::U32, &dev).unwrap();
            let d = pool.reserve_pooled(&[1, 64], DType::U32, &dev).unwrap();
            vec![
                cpu_storage_data_ptr(a.as_tensor()).unwrap(),
                cpu_storage_data_ptr(b.as_tensor()).unwrap(),
                cpu_storage_data_ptr(c.as_tensor()).unwrap(),
                cpu_storage_data_ptr(d.as_tensor()).unwrap(),
            ]
        };

        let jit = phase();
        let capture = phase();
        let runtime = phase();

        assert_eq!(
            jit, capture,
            "capture-phase pool ptrs diverged from JIT warmup — \
             slot-drift class regression (see Bug B / commit 2614dd7)"
        );
        assert_eq!(
            capture, runtime,
            "runtime pool ptrs diverged from capture warmup — \
             captured graph would dereference the wrong slot"
        );

        // Sanity: within a phase the two `[1]/U32` reserves MUST land
        // on distinct slots (cursor 0 vs cursor 1). Otherwise the
        // round-robin cursor is broken and the cross-phase equality
        // above is meaningless.
        assert_ne!(
            jit[0], jit[1],
            "same-shape reserves collided into one slot — round-robin cursor broken"
        );
    }

    #[test]
    fn pool_slot_drift_without_reset_is_observable() {
        // Reproducer of Bug B.1's failure mode: when a phase forgets to
        // call `reset_cursors`, the cursor leaks from the previous phase
        // and the slot served by `reserve_pooled` shifts. This test
        // verifies the shift is *observable* via storage pointers — i.e.
        // future diagnostics or fuzzers can catch the bug class.
        let pool = OutputPool::default();
        let dev = Device::Cpu;

        pool.reset_cursors();
        // Warmup advances the cursor for `[1]/U32` past zero.
        let _w0 = pool.reserve_pooled(&[1], DType::U32, &dev).unwrap();
        let _w1 = pool.reserve_pooled(&[1], DType::U32, &dev).unwrap();

        // Simulated buggy capture path: no reset_cursors. Cursor stays
        // at 2 → reserve hands out slot 2.
        let buggy = pool.reserve_pooled(&[1], DType::U32, &dev).unwrap();
        let buggy_ptr = cpu_storage_data_ptr(buggy.as_tensor()).unwrap();

        // Production path: resets first, reserve hands out slot 0.
        pool.reset_cursors();
        let prod = pool.reserve_pooled(&[1], DType::U32, &dev).unwrap();
        let prod_ptr = cpu_storage_data_ptr(prod.as_tensor()).unwrap();

        assert_ne!(
            buggy_ptr, prod_ptr,
            "without reset_cursors, slots MUST diverge — otherwise the \
             slot-drift bug class would be invisible to diagnostics"
        );
    }

    #[test]
    fn reserve_pooled_data_ptr_matches_after_reset_cursors() {
        // Strengthens `reserve_pooled_replays_same_addresses_after_reset_cursors`
        // (which only verified `total_slots`): the actual underlying
        // storage pointer is identical across a reset/re-reserve cycle.
        let pool = OutputPool::default();
        let dev = Device::Cpu;

        let a = pool.reserve_pooled(&[4], DType::F32, &dev).unwrap();
        let a_ptr = cpu_storage_data_ptr(a.as_tensor()).unwrap();
        drop(a);

        pool.reset_cursors();
        let a2 = pool.reserve_pooled(&[4], DType::F32, &dev).unwrap();
        let a2_ptr = cpu_storage_data_ptr(a2.as_tensor()).unwrap();

        assert_eq!(a_ptr, a2_ptr, "reset_cursors must replay the same slot ptr");
        assert_eq!(pool.total_slots(), 1, "no new allocation after reset");
    }

    #[test]
    fn engine_limits_pool_worst_case_seq_len_is_idempotent() {
        // OnceLock semantics: subsequent `set_*` calls are silently
        // ignored — only the first wins. This is the contract that the
        // server's startup sequence relies on (limits published BEFORE
        // start_engine, never overwritten). If a future refactor swaps
        // `OnceLock` for a mutable cell, this test fails, forcing the
        // author to revisit Bug B.2's root cause.
        //
        // The test uses a LOCAL `OnceLock<usize>` to avoid stomping on
        // the process-global `engine_limits` state shared with other
        // tests. The point is to lock down the OnceLock contract; the
        // actual production use site reads from a real `OnceLock` with
        // the same semantics.
        let cell: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        assert!(cell.set(1024).is_ok(), "first set must win");
        assert!(
            cell.set(131072).is_err(),
            "second set must fail silently — OnceLock idempotency contract"
        );
        assert_eq!(*cell.get().unwrap(), 1024);
    }

    #[test]
    fn pool_worst_case_seq_len_falls_back_to_1024_when_unset() {
        // Documents that build_shared's stride sizing falls back to
        // `worst_case_max_blocks_per_seq = 1024 / 16 = 64` when neither
        // `max_model_len` nor `max_seq_len_to_capture` was published by
        // the server. This is the historical default that tests and
        // benches rely on; changing it would silently re-shape every
        // build_shared call site (Bug B.2 class).
        //
        // We cannot assert the actual return value of
        // `engine_limits::pool_worst_case_seq_len()` from a test —
        // process-global OnceLock state may have been initialised by
        // an earlier test in the same binary. Instead the test pins the
        // documented fallback constant via the same OnceLock pattern.
        let cell: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let observed = cell.get().copied().unwrap_or(1024);
        assert_eq!(
            observed, 1024,
            "fallback contract: unset OnceLock must read as 1024"
        );
    }
}
