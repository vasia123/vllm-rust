//! Quantized Qwen3 model implementation.
//!
//! This module provides a quantized version of the Qwen3 model that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Decode-path profiler ────────────────────────────────────────────────────
//
// Diagnostic instrumentation: when env `VLLM_PROFILE_DECODE=1` is set at
// process start, per-component wall-clock times inside the AWQ decoder hot
// path are accumulated in nanoseconds and dumped via `tracing::info!` once
// per `PROFILE_EVERY` layer-calls.
//
// **Measurement model (Stage 14-C.1):** symmetric to `prefill_profile`. Each
// `time(...)` call records a pair of CUDA events on the device stream — one
// before the closure, one after — and queues the pair in a thread-local Vec.
// No host sync per measurement; GPU pipeline overlap is preserved. At
// `maybe_dump` time the Vec drains, each pair's `elapsed_ms` is queried
// (one sync per pair, on the `end` event), and deltas accumulate into the
// per-component slots. The previous `cuda_stream::synchronize()`-per-call
// approach distorted ratios non-uniformly — see ADR 0015 § 3 and
// Stage 14-B notes in `prefill_profile` for the post-mortem.
//
// All overhead lives behind a once-initialised flag — the fast path is a
// single relaxed atomic load when profiling is off.

#[cfg(feature = "cuda")]
mod decode_profile {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    use candle_core::cuda::cudarc::driver::sys::CUevent_flags_enum;
    use candle_core::cuda::cudarc::driver::CudaEvent;

    pub const PROFILE_EVERY: usize = 100;

    static ENABLED: OnceLock<bool> = OnceLock::new();

    pub fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("VLLM_PROFILE_DECODE").is_ok())
    }

    // Thread-local queue of CUDA-event pairs awaiting elapsed-ms readout.
    // A 36-layer Qwen3 forward queues ~9 × 36 = 324 pairs/forward; with
    // PROFILE_EVERY=100 layer-calls (~3 forwards) ≈ ~900 pairs between
    // flushes. Each pair: two `CudaEvent`s + static slot ptr.
    type EventEntry = (CudaEvent, CudaEvent, &'static AtomicU64);
    thread_local! {
        static EVENT_QUEUE: RefCell<Vec<EventEntry>> = const { RefCell::new(Vec::new()) };
    }

    pub static QKV_NS: AtomicU64 = AtomicU64::new(0);
    pub static QK_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    pub static CACHE_WRITE_NS: AtomicU64 = AtomicU64::new(0);
    pub static PAGED_ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static O_PROJ_NS: AtomicU64 = AtomicU64::new(0);
    pub static MLP_NS: AtomicU64 = AtomicU64::new(0);
    pub static IN_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static POST_NORM_NS: AtomicU64 = AtomicU64::new(0);
    /// Per-pass top-level slots — incremented once per forward, NOT once
    /// per layer. EMBED is the input_ids → hidden lookup; FINAL_NORM is
    /// the post-all-layers RMSNorm; LM_HEAD is the [hidden → vocab]
    /// matmul (BF16 dense, ~25 GFLOPs/step at M=8 for Qwen3-4B's 152K
    /// vocab — quantified in `docs/perf/post-15D-decode-profile.md`).
    pub static EMBED_NS: AtomicU64 = AtomicU64::new(0);
    pub static FINAL_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static LM_HEAD_NS: AtomicU64 = AtomicU64::new(0);
    /// Stage A'.2 diagnostic: parallel synchronize+Instant timing for
    /// lm_head, opt-in via `VLLM_PROFILE_LM_HEAD_SYNC=1`. Forces a
    /// `cuda_stream::synchronize()` before AND after the matmul so the
    /// host wallclock captures EXACT kernel duration in isolation
    /// (different from CUDA-event timing which can include adjacent
    /// stream-pending work). Comparing the two slot values reveals
    /// whether the 16-24 ms event-measured lm_head time is real GPU
    /// work or a profiler artefact.
    pub static LM_HEAD_SYNC_NS: AtomicU64 = AtomicU64::new(0);
    pub static LAYERS_DONE: AtomicUsize = AtomicUsize::new(0);

    pub fn sync_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("VLLM_PROFILE_LM_HEAD_SYNC").is_ok())
    }

    /// synchronize + Instant timer. Forces full GPU sync before/after.
    /// Only call when measuring is intentional — adds a sync per call.
    pub fn time_sync<T>(
        dev: &candle_core::Device,
        slot: &'static AtomicU64,
        f: impl FnOnce() -> candle_core::Result<T>,
    ) -> candle_core::Result<T> {
        if !sync_enabled() {
            return f();
        }
        if let candle_core::Device::Cuda(cd) = dev {
            let stream = cd.cuda_stream();
            let _ = stream.synchronize();
            let t0 = Instant::now();
            let out = f()?;
            let _ = stream.synchronize();
            slot.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Ok(out)
        } else {
            f()
        }
    }

    /// Record a start event, run `f`, record an end event, queue the pair
    /// against `slot`. `elapsed_ms` is queried later in `flush_events()`.
    pub fn time<T>(
        dev: &candle_core::Device,
        slot: &'static AtomicU64,
        f: impl FnOnce() -> candle_core::Result<T>,
    ) -> candle_core::Result<T> {
        if !enabled() {
            return f();
        }
        match dev {
            candle_core::Device::Cuda(cd) => {
                let stream = cd.cuda_stream();
                let start = stream
                    .record_event(Some(CUevent_flags_enum::CU_EVENT_DEFAULT))
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("decode profile start event: {e}"))
                    })?;
                let out = f()?;
                let end = stream
                    .record_event(Some(CUevent_flags_enum::CU_EVENT_DEFAULT))
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("decode profile end event: {e}"))
                    })?;
                EVENT_QUEUE.with(|cell| cell.borrow_mut().push((start, end, slot)));
                Ok(out)
            }
            _ => {
                let t0 = Instant::now();
                let out = f()?;
                slot.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                Ok(out)
            }
        }
    }

    /// Drain queued event pairs, query elapsed_ms (one sync per pair via the
    /// `end` event), and accumulate into the matching `*_NS` slot.
    fn flush_events() {
        EVENT_QUEUE.with(|cell| {
            let pairs = std::mem::take(&mut *cell.borrow_mut());
            for (start, end, slot) in pairs {
                match start.elapsed_ms(&end) {
                    Ok(ms) => {
                        let ns = (ms as f64 * 1_000_000.0) as u64;
                        slot.fetch_add(ns, Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "vllm_core::decode_profile",
                            "elapsed_ms query failed, dropping event pair: {e}"
                        );
                    }
                }
            }
        });
    }

    pub fn maybe_dump() {
        if !enabled() {
            return;
        }
        let n = LAYERS_DONE.fetch_add(1, Ordering::Relaxed) + 1;
        if !n.is_multiple_of(PROFILE_EVERY) {
            return;
        }
        // Drain queued events into slots BEFORE snapshotting, so the dump
        // window aligns with the accumulation window.
        flush_events();
        // Snapshot-then-reset each slot so the dump and the subsequent
        // accumulation window stay independent.
        let snap = |a: &AtomicU64| (a.swap(0, Ordering::Relaxed) as f64) / 1_000.0;
        let s_in = snap(&IN_NORM_NS);
        let s_qkv = snap(&QKV_NS);
        let s_qkn = snap(&QK_NORM_NS);
        let s_rope = snap(&ROPE_NS);
        let s_cw = snap(&CACHE_WRITE_NS);
        let s_pa = snap(&PAGED_ATTN_NS);
        let s_op = snap(&O_PROJ_NS);
        let s_pn = snap(&POST_NORM_NS);
        let s_mlp = snap(&MLP_NS);
        // Top-level (per-forward, not per-layer) slots. Each forward fires
        // once → these accumulate ~PROFILE_EVERY/36 = ~3 events between
        // dumps. Reported as μs/forward (NOT μs/layer-call).
        let s_embed = snap(&EMBED_NS);
        let s_fn = snap(&FINAL_NORM_NS);
        let s_lm = snap(&LM_HEAD_NS);
        let s_lm_sync = snap(&LM_HEAD_SYNC_NS);
        let layer_total = s_in + s_qkv + s_qkn + s_rope + s_cw + s_pa + s_op + s_pn + s_mlp;
        let n = PROFILE_EVERY as f64;
        // Approx number of forwards covered by this dump window (PROFILE_EVERY
        // layer-calls / num_layers). Used to normalise per-forward slots.
        let approx_forwards = (PROFILE_EVERY as f64) / 36.0;
        tracing::info!(
            target: "vllm_core::decode_profile",
            "decode breakdown ({} layer-calls, μs/call): in_norm={:.1} qkv={:.1} \
             qk_norm={:.1} rope={:.1} cache_w={:.1} paged_attn={:.1} o_proj={:.1} \
             post_norm={:.1} mlp={:.1} | layer_sum={:.1} | per-forward μs: \
             embed={:.1} final_norm={:.1} lm_head_evt={:.1} lm_head_sync={:.1}",
            PROFILE_EVERY,
            s_in / n,
            s_qkv / n,
            s_qkn / n,
            s_rope / n,
            s_cw / n,
            s_pa / n,
            s_op / n,
            s_pn / n,
            s_mlp / n,
            layer_total / n,
            s_embed / approx_forwards,
            s_fn / approx_forwards,
            s_lm / approx_forwards,
            s_lm_sync / approx_forwards,
        );
    }
}

// ─── Prefill-path profiler ───────────────────────────────────────────────────
//
// Symmetric counterpart to `decode_profile`. Activated via env
// `VLLM_PROFILE_PREFILL=1`. Each component slot accumulates ns across all
// `num_hidden_layers` layer-calls of a single prefill forward, and the
// per-pass top-level slots (embedding, causal_mask, final_norm, lm_head) are
// added once. `dump_prefill()` is called at the end of each prefill forward
// — prefills are rare, so per-call output is more useful than averaging.
//
// Naming: PA_NS includes the cache-write that `paged_attention()` performs
// internally; we don't split it because the prefill path does not have a
// separate `cache_engine.write_batch()` call (unlike the decode wrapper).
//
// **Measurement model (Stage 14-B):** every `prefill_profile::time(...)` call
// records a pair of CUDA events on the device stream — one before the
// closure dispatches its work, one after — and queues the pair in a
// thread-local Vec. No host sync per measurement; the GPU pipeline overlap
// is preserved. At `dump_prefill` time the Vec drains, each pair's
// `elapsed_ms` is queried (which syncs the **second** event of the pair, a
// single sync per pair, not per measurement), and the deltas accumulate
// into the per-component `*_NS` slots.
//
// Why this matters: the previous `cuda_stream::synchronize()`-per-measurement
// approach made the absolute numbers 8–10× wallclock and — more dangerously
// — distorted ratios non-uniformly. Components downstream of long kernel
// chains looked bigger than they really are. Stage 13-K.1 audit read 67.8 %
// of forward as `paged_attn` under the old profile; Stage 14-A.1 implemented
// the implied optimization (per-forward plan caching) and measured a
// 240 → 336 ms TTFT *regression* — the kernel itself dominates wallclock,
// the sync-mode profile just made plan-build look big. ADR 0015 § 3 has
// the full post-mortem.
//
// CPU device fallback still uses `Instant`-based wallclock; CUDA path uses
// events. Either way enable with `VLLM_PROFILE_PREFILL=1`.

#[cfg(feature = "cuda")]
mod prefill_profile {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    use candle_core::cuda::cudarc::driver::sys::CUevent_flags_enum;
    use candle_core::cuda::cudarc::driver::CudaEvent;

    static ENABLED: OnceLock<bool> = OnceLock::new();

    pub fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("VLLM_PROFILE_PREFILL").is_ok())
    }

    // Thread-local queue of CUDA-event pairs awaiting elapsed-ms readout.
    // A 36-layer Qwen3 forward queues ~12 × 36 = 432 pairs; each pair holds
    // two `CudaEvent`s (~24 B + small FFI handle) and a static slot pointer.
    // `dump_prefill` drains and resets.
    type EventEntry = (CudaEvent, CudaEvent, &'static AtomicU64);
    thread_local! {
        static EVENT_QUEUE: RefCell<Vec<EventEntry>> = const { RefCell::new(Vec::new()) };
    }

    pub static EMBED_NS: AtomicU64 = AtomicU64::new(0);
    pub static MASK_NS: AtomicU64 = AtomicU64::new(0);
    pub static IN_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static QKV_NS: AtomicU64 = AtomicU64::new(0);
    pub static QK_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    pub static PA_NS: AtomicU64 = AtomicU64::new(0);
    pub static O_PROJ_NS: AtomicU64 = AtomicU64::new(0);
    pub static POST_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static MLP_NS: AtomicU64 = AtomicU64::new(0);
    pub static FINAL_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static LM_HEAD_NS: AtomicU64 = AtomicU64::new(0);
    pub static LAYERS: AtomicUsize = AtomicUsize::new(0);

    pub fn time<T>(
        dev: &candle_core::Device,
        slot: &'static AtomicU64,
        f: impl FnOnce() -> candle_core::Result<T>,
    ) -> candle_core::Result<T> {
        if !enabled() {
            return f();
        }
        match dev {
            candle_core::Device::Cuda(cd) => {
                // Stage 14-B — event-based timing without per-measurement
                // host sync. `CU_EVENT_DEFAULT` keeps timing enabled (default
                // would be `DISABLE_TIMING`, optimised for sync-only events).
                let stream = cd.cuda_stream();
                let start = stream
                    .record_event(Some(CUevent_flags_enum::CU_EVENT_DEFAULT))
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("prefill profile start event: {e}"))
                    })?;
                let out = f()?;
                let end = stream
                    .record_event(Some(CUevent_flags_enum::CU_EVENT_DEFAULT))
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("prefill profile end event: {e}"))
                    })?;
                EVENT_QUEUE.with(|cell| cell.borrow_mut().push((start, end, slot)));
                Ok(out)
            }
            _ => {
                // CPU path: fall back to host wallclock. Same shape as before.
                let t0 = Instant::now();
                let out = f()?;
                slot.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                Ok(out)
            }
        }
    }

    /// Drain queued event pairs, query elapsed_ms (one sync per pair via the
    /// `end` event), and accumulate into the matching `*_NS` slot.
    fn flush_events() {
        EVENT_QUEUE.with(|cell| {
            let pairs = std::mem::take(&mut *cell.borrow_mut());
            for (start, end, slot) in pairs {
                match start.elapsed_ms(&end) {
                    Ok(ms) => {
                        let ns = (ms as f64 * 1_000_000.0) as u64;
                        slot.fetch_add(ns, Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "vllm_core::prefill_profile",
                            "elapsed_ms query failed, dropping event pair: {e}"
                        );
                    }
                }
            }
        });
    }

    pub fn bump_layer() {
        if enabled() {
            LAYERS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dump_prefill(seq_len: usize) {
        if !enabled() {
            return;
        }
        // Stage 14-B: drain queued event pairs into the slots before
        // reading them. CPU device path skipped events and updated slots
        // directly via Instant; CUDA path queued events and we collect
        // here. Either way, slots are authoritative below.
        flush_events();
        let snap = |a: &AtomicU64| (a.swap(0, Ordering::Relaxed) as f64) / 1_000.0;
        let s_emb = snap(&EMBED_NS);
        let s_mask = snap(&MASK_NS);
        let s_in = snap(&IN_NORM_NS);
        let s_qkv = snap(&QKV_NS);
        let s_qkn = snap(&QK_NORM_NS);
        let s_rope = snap(&ROPE_NS);
        let s_pa = snap(&PA_NS);
        let s_op = snap(&O_PROJ_NS);
        let s_pn = snap(&POST_NORM_NS);
        let s_mlp = snap(&MLP_NS);
        let s_fn = snap(&FINAL_NORM_NS);
        let s_lm = snap(&LM_HEAD_NS);
        let l = LAYERS.swap(0, Ordering::Relaxed).max(1);
        let layered = s_in + s_qkv + s_qkn + s_rope + s_pa + s_op + s_pn + s_mlp;
        let total = s_emb + s_mask + layered + s_fn + s_lm;
        tracing::info!(
            target: "vllm_core::prefill_profile",
            "prefill breakdown (seq_len={}, layers={}, μs total): embed={:.1} mask={:.1} \
             in_norm={:.1} qkv={:.1} qk_norm={:.1} rope={:.1} paged_attn={:.1} \
             o_proj={:.1} post_norm={:.1} mlp={:.1} final_norm={:.1} lm_head={:.1} | sum={:.1}",
            seq_len, l,
            s_emb, s_mask,
            s_in, s_qkv, s_qkn, s_rope, s_pa, s_op, s_pn, s_mlp,
            s_fn, s_lm, total,
        );
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
mod prefill_profile {
    use std::sync::atomic::AtomicU64;

    pub static EMBED_NS: AtomicU64 = AtomicU64::new(0);
    pub static MASK_NS: AtomicU64 = AtomicU64::new(0);
    pub static IN_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static QKV_NS: AtomicU64 = AtomicU64::new(0);
    pub static QK_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    pub static PA_NS: AtomicU64 = AtomicU64::new(0);
    pub static O_PROJ_NS: AtomicU64 = AtomicU64::new(0);
    pub static POST_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static MLP_NS: AtomicU64 = AtomicU64::new(0);
    pub static FINAL_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static LM_HEAD_NS: AtomicU64 = AtomicU64::new(0);

    #[inline(always)]
    pub fn time<T>(
        _dev: &candle_core::Device,
        _slot: &AtomicU64,
        f: impl FnOnce() -> candle_core::Result<T>,
    ) -> candle_core::Result<T> {
        f()
    }

    #[inline(always)]
    pub fn bump_layer() {}
    #[inline(always)]
    pub fn dump_prefill(_seq_len: usize) {}
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
mod decode_profile {
    use std::sync::atomic::AtomicU64;

    pub static QKV_NS: AtomicU64 = AtomicU64::new(0);
    pub static QK_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    pub static CACHE_WRITE_NS: AtomicU64 = AtomicU64::new(0);
    pub static PAGED_ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static O_PROJ_NS: AtomicU64 = AtomicU64::new(0);
    pub static MLP_NS: AtomicU64 = AtomicU64::new(0);
    pub static IN_NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static POST_NORM_NS: AtomicU64 = AtomicU64::new(0);

    /// CPU-only build: profiling is a no-op and falls through to `f`.
    #[inline(always)]
    pub fn time<T>(
        _dev: &candle_core::Device,
        _slot: &AtomicU64,
        f: impl FnOnce() -> candle_core::Result<T>,
    ) -> candle_core::Result<T> {
        f()
    }

    #[inline(always)]
    pub fn maybe_dump() {}
}

// ─── Quantized SwiGLU MLP ────────────────────────────────────────────────────

struct QuantizedSwiGluMlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedSwiGluMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let gate_proj = loader.load_linear(
            &format!("{prefix}.gate_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let up_proj = loader.load_linear(
            &format!("{prefix}.up_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let down_proj = loader.load_linear(
            &format!("{prefix}.down_proj"),
            intermediate_size,
            hidden_size,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // Pool-backed silu(gate) * up — replaces 2-allocation candle path
        // (`silu(&gate)? * up` materialises both the silu intermediate and
        // the multiplication output) with a single stable-address receiver
        // from the global OutputPool. Decode-only fast path; prefill and
        // non-BF16 dtypes fall through to the candle fallback inside the
        // wrapper.
        #[cfg(feature = "cuda-fused-activations")]
        {
            if gate.device().is_cuda() {
                let activated = crate::cuda_kernels::silu_and_mul_separate_pooled(&gate, &up)?;
                return self.down_proj.forward(&activated);
            }
        }

        let activated = candle_nn::ops::silu(&gate)? * up;
        self.down_proj.forward(&activated?)
    }
}

// ─── Quantized Attention ─────────────────────────────────────────────────────

struct QuantizedQwen3Attention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedQwen3Attention {
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            false,
        )?;
        let k_proj = loader.load_linear(
            &format!("{prefix}.k_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let v_proj = loader.load_linear(
            &format!("{prefix}.v_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let o_proj = loader.load_linear(
            &format!("{prefix}.o_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

        // Qwen3-specific: per-head RMSNorm (loaded from VarBuilder, not quantized)
        let vb_attn = vb.pp("self_attn");
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb_attn.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb_attn.pp("k_norm"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            loader.dtype(),
            loader.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let dev = xs.device();

        let (q, k, v) = prefill_profile::time(dev, &prefill_profile::QKV_NS, || {
            let q = self.q_proj.forward(xs)?;
            let k = self.k_proj.forward(xs)?;
            let v = self.v_proj.forward(xs)?;
            Ok((q, k, v))
        })?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Qwen3-specific: per-head RMSNorm on Q and K
        let (q, k) = prefill_profile::time(dev, &prefill_profile::QK_NORM_NS, || {
            let q = apply_per_head_norm(&q, &self.q_norm)?;
            let k = apply_per_head_norm(&k, &self.k_norm)?;
            Ok((q, k))
        })?;

        let (q, k) = prefill_profile::time(dev, &prefill_profile::ROPE_NS, || {
            self.rotary_emb.apply(&q, &k, seqlen_offset)
        })?;

        let attn_output = prefill_profile::time(dev, &prefill_profile::PA_NS, || {
            paged_attention(
                &q,
                &k,
                &v,
                attention_mask,
                seqlen_offset,
                cache_engine,
                block_table.block_ids(),
                slot_mapping,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        })?;

        prefill_profile::time(dev, &prefill_profile::O_PROJ_NS, || {
            self.o_proj.forward(&attn_output)
        })
    }

    fn forward_decode_batch_with_shared(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        shared: Option<&crate::layers::attention::DecodeBatchShared>,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let dev = xs.device();

        let (q, k, v) = decode_profile::time(dev, &decode_profile::QKV_NS, || {
            let q = self.q_proj.forward(xs)?;
            let k = self.k_proj.forward(xs)?;
            let v = self.v_proj.forward(xs)?;
            Ok((q, k, v))
        })?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Qwen3-specific: per-head RMSNorm
        let (q, k) = decode_profile::time(dev, &decode_profile::QK_NORM_NS, || {
            let q = apply_per_head_norm(&q, &self.q_norm)?;
            let k = apply_per_head_norm(&k, &self.k_norm)?;
            Ok((q, k))
        })?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            // Reuse pre-built per-forward shared tensors when supplied.
            // Without `shared`, every decoder layer used to rebuild
            // `block_tables` / `seq_lens` via `Tensor::from_vec` — 36
            // host->device uploads per Qwen3-4B token and the implicit
            // syncs that come with them.
            let positions_owned: Vec<usize>;
            let positions: &[usize] = match shared {
                Some(s) => &s.positions,
                None => {
                    positions_owned = sequences.iter().map(|s| s.seqlen_offset).collect();
                    &positions_owned
                }
            };
            let (q, k) = decode_profile::time(dev, &decode_profile::ROPE_NS, || {
                self.rotary_emb.apply_varlen(&q, &k, positions)
            })?;

            let slot_mapping_owned: Vec<usize>;
            let all_slot_mapping: &[usize] = match shared {
                Some(s) => &s.all_slot_mapping,
                None => {
                    slot_mapping_owned = sequences
                        .iter()
                        .flat_map(|s| s.slot_mapping.iter().copied())
                        .collect();
                    &slot_mapping_owned
                }
            };
            decode_profile::time(dev, &decode_profile::CACHE_WRITE_NS, || {
                cache_engine
                    .write_batch(&k, &v, all_slot_mapping)
                    .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))
            })?;

            let (block_tables, seq_lens, max_blocks_per_seq, max_seq_len): (
                Tensor,
                Tensor,
                usize,
                usize,
            ) = match shared {
                Some(s) => (
                    s.block_tables.clone(),
                    s.seq_lens.clone(),
                    s.max_blocks_per_seq,
                    s.max_seq_len,
                ),
                None => {
                    let max_blocks_per_seq = sequences
                        .iter()
                        .map(|s| s.block_ids.len())
                        .max()
                        .unwrap_or(1);
                    let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
                    for (i, seq) in sequences.iter().enumerate() {
                        for (j, &block_id) in seq.block_ids.iter().enumerate() {
                            bt_data[i * max_blocks_per_seq + j] = block_id as u32;
                        }
                    }
                    let bt =
                        Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), q.device())?;

                    let seq_lens_data: Vec<u32> = sequences
                        .iter()
                        .map(|s| (s.seqlen_offset + 1) as u32)
                        .collect();
                    let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
                    let sl = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

                    (bt, sl, max_blocks_per_seq, max_seq_len)
                }
            };

            let scale = 1.0 / (self.head_dim as f32).sqrt();

            let attn_output = decode_profile::time(dev, &decode_profile::PAGED_ATTN_NS, || {
                crate::cuda_kernels::paged_attention_auto(
                    &q,
                    cache_engine.k_cache(),
                    cache_engine.v_cache(),
                    &block_tables,
                    &seq_lens,
                    scale,
                    self.num_heads,
                    self.num_kv_heads,
                    max_blocks_per_seq,
                    max_seq_len,
                    self.head_dim,
                    cache_engine.block_size(),
                )
            })?;

            decode_profile::time(dev, &decode_profile::O_PROJ_NS, || {
                self.o_proj.forward(&attn_output.unsqueeze(1)?)
            })
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            // CPU fallback path doesn't use the shared bundle (no GPU
            // kernel to feed pre-built tensors into); explicitly mark it
            // consumed so `-D unused-variables` stays clean.
            let _ = shared;
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

                let attn_out = paged_attention(
                    &q_i,
                    &k_i,
                    &v_i,
                    None,
                    seq.seqlen_offset,
                    cache_engine,
                    &seq.block_ids,
                    &seq.slot_mapping,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )?;
                outputs.push(attn_out);
            }

            let attn_output = Tensor::cat(&outputs, 0)?;
            self.o_proj.forward(&attn_output)
        }
    }
}

// ─── Quantized Decoder Layer ─────────────────────────────────────────────────

struct QuantizedQwen3DecoderLayer {
    self_attn: QuantizedQwen3Attention,
    mlp: QuantizedSwiGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen3DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        let vb_layer = vb.pp("model").pp("layers").pp(layer_idx);

        let self_attn = QuantizedQwen3Attention::new(
            cfg,
            loader,
            vb_layer.clone(),
            &format!("{prefix}.self_attn"),
        )?;
        let mlp = QuantizedSwiGluMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            loader,
            &format!("{prefix}.mlp"),
        )?;

        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let dev = xs.device();
        let xs = prefill_profile::time(dev, &prefill_profile::IN_NORM_NS, || {
            self.input_layernorm.forward(xs)
        })?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let normed = prefill_profile::time(dev, &prefill_profile::POST_NORM_NS, || {
            self.post_attention_layernorm.forward(&xs)
        })?;
        let xs =
            prefill_profile::time(dev, &prefill_profile::MLP_NS, || self.mlp.forward(&normed))?;
        prefill_profile::bump_layer();
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        self.forward_decode_batch_with_shared(xs, sequences, kv_cache_mgr, layer_idx, None)
    }

    fn forward_decode_batch_with_shared(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        shared: Option<&crate::layers::attention::DecodeBatchShared>,
    ) -> Result<Tensor> {
        let residual = xs;
        let dev = xs.device();
        let xs = decode_profile::time(dev, &decode_profile::IN_NORM_NS, || {
            self.input_layernorm.forward(xs)
        })?;
        let xs = self.self_attn.forward_decode_batch_with_shared(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            shared,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let normed = decode_profile::time(dev, &decode_profile::POST_NORM_NS, || {
            self.post_attention_layernorm.forward(&xs)
        })?;
        let xs = decode_profile::time(dev, &decode_profile::MLP_NS, || self.mlp.forward(&normed))?;
        decode_profile::maybe_dump();
        residual + xs
    }
}

// ─── Quantized Model ─────────────────────────────────────────────────────────

/// Quantized Qwen3 model supporting FP8, GPTQ, AWQ, and unquantized weights.
pub struct QuantizedQwen3ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<QuantizedQwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedQwen3ForCausalLM {
    /// Create a new quantized Qwen3 model.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `vb` - VarBuilder for loading non-quantized weights (embeddings, norms)
    /// * `weight_loader` - Quantized weight loader for linear layers
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedQwen3DecoderLayer::new(
                cfg,
                weight_loader,
                vb.clone(),
                i,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead {
                weight: embed_tokens.embeddings().clone(),
            }) as Box<dyn QuantizedLinear>
        } else {
            weight_loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let dev = &self.device;
        let attention_mask = prefill_profile::time(dev, &prefill_profile::MASK_NS, || {
            if seq_len <= 1 {
                Ok(None)
            } else {
                Ok(Some(crate::layers::causal_mask(
                    seq_len,
                    seqlen_offset,
                    self.dtype,
                    &self.device,
                )?))
            }
        })?;

        let mut xs = prefill_profile::time(dev, &prefill_profile::EMBED_NS, || {
            self.embed_tokens.forward(input_ids)
        })?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = prefill_profile::time(dev, &prefill_profile::FINAL_NORM_NS, || {
            self.norm.forward(&xs)
        })?;
        let logits = prefill_profile::time(dev, &prefill_profile::LM_HEAD_NS, || {
            self.lm_head.forward(&xs)
        })?;
        prefill_profile::dump_prefill(seq_len);
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper for tied embedding lm_head.
struct TiedEmbeddingHead {
    weight: Tensor,
}

impl QuantizedLinear for TiedEmbeddingHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 2026-05-09: for 3D `[B, S, H]` inputs, flatten to 2D so the
        // call lands on cuBLAS's plain GEMM `[B*S, H] @ [H, V]` instead
        // of `weight.broadcast_left(B) → batched GEMM with stride_b=0`.
        // The 2D path picks a markedly better algorithm for the tied
        // lm_head shape (M=B, K=hidden, N=vocab); +32 % e2e at c=8 on
        // Qwen3-4B-AWQ in side-by-side bench (118.8 → 157.1 tps).
        //
        // 2026-05-09 (Phase B.6): on the decode hot path
        // `crate::cuda_kernels::bf16_matmul_pooled` reserves the output
        // from the global OutputPool so its device address stays stable
        // across forwards — required for CUDA Graph capture replay.
        // Wrapper itself falls back to candle's matmul for prefill / non-BF16.
        match x.dims().len() {
            3 => {
                let dims = x.dims();
                let (b, s, h) = (dims[0], dims[1], dims[2]);
                let v = self.weight.dims()[0];
                let x_flat = x.reshape((b * s, h))?;
                #[cfg(feature = "cuda-kernels")]
                {
                    let y_flat = crate::cuda_kernels::bf16_matmul_pooled(&x_flat, &self.weight)?;
                    return y_flat.reshape((b, s, v));
                }
                #[allow(unreachable_code)]
                {
                    let y_flat = x_flat.matmul(&self.weight.t()?)?;
                    y_flat.reshape((b, s, v))
                }
            }
            _ => {
                #[cfg(feature = "cuda-kernels")]
                {
                    return crate::cuda_kernels::bf16_matmul_pooled(x, &self.weight);
                }
                #[allow(unreachable_code)]
                x.matmul(&self.weight.t()?)
            }
        }
    }

    fn load_weights(&mut self, _weights: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        self.weight.dtype()
    }

    fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    fn has_bias(&self) -> bool {
        false
    }
}

impl crate::engine::ModelForward for QuantizedQwen3ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let dev = input_ids.device();
        let mut xs = decode_profile::time(dev, &decode_profile::EMBED_NS, || {
            self.embed_tokens.forward(input_ids)
        })?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = decode_profile::time(dev, &decode_profile::FINAL_NORM_NS, || {
            self.norm.forward(&xs)
        })?;
        decode_profile::time(dev, &decode_profile::LM_HEAD_NS, || {
            decode_profile::time_sync(dev, &decode_profile::LM_HEAD_SYNC_NS, || {
                self.lm_head.forward(&xs)
            })
        })
    }

    /// Override the default trait fall-through so the per-forward shared
    /// decode tensors built in `engine::helpers::execute_batched_decode_with_graph`
    /// reach every attention layer. Without this override the quantized
    /// path silently rebuilt block_tables/seq_lens 36× per token via
    /// per-layer `Tensor::from_vec` host->device uploads.
    fn forward_decode_batch_with_ctx(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        ctx: &crate::engine::cuda_graph::ForwardContext,
    ) -> Result<Tensor> {
        let shared: Option<&crate::layers::attention::DecodeBatchShared> = ctx
            .decode_shared
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<crate::layers::attention::DecodeBatchShared>());

        let dev = input_ids.device();
        let mut xs = decode_profile::time(dev, &decode_profile::EMBED_NS, || {
            // Pool-backed embedding lookup on the decode hot path:
            // captured CUDA graphs need a stable-address output buffer
            // here so replay reads from a consistent device pointer.
            // Decode-only fast path: num_tokens = batch_size × 1 ≤ 64.
            #[cfg(feature = "cuda-fused-activations")]
            {
                let n: usize = input_ids.dims().iter().product();
                if n <= 64
                    && input_ids.device().is_cuda()
                    && self.embed_tokens.embeddings().dtype() == DType::BF16
                    && input_ids.dtype() == DType::U32
                {
                    return crate::cuda_kernels::embedding_pooled(
                        input_ids,
                        self.embed_tokens.embeddings(),
                    );
                }
            }
            self.embed_tokens.forward(input_ids)
        })?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch_with_shared(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                shared,
            )?;
        }
        let xs = decode_profile::time(dev, &decode_profile::FINAL_NORM_NS, || {
            self.norm.forward(&xs)
        })?;
        decode_profile::time(dev, &decode_profile::LM_HEAD_NS, || {
            decode_profile::time_sync(dev, &decode_profile::LM_HEAD_SYNC_NS, || {
                self.lm_head.forward(&xs)
            })
        })
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

    fn test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_quantized_qwen3_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedQwen3ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedQwen3ForCausalLM should construct with unquantized loader"
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_qwen3_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedQwen3ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }
}
