//! AWQ-Marlin quantization.
//!
//! AWQ-Marlin loads weights using the AWQ checkpoint format (qweight, scales,
//! qzeros with interleaved nibble ordering) and routes inference through the
//! Marlin INT4 kernel.  The weight loader deinterleaves AWQ's nibble ordering
//! to produce standard GPTQ sequential packing, then passes the result to
//! `MarlinLinear` which handles the final GPTQ → Marlin tile repack.
//!
//! AWQ nibble ordering (interleaved even-odd):
//!   u32 encodes [v0, v2, v4, v6, v1, v3, v5, v7] in nibbles 0..7
//!
//! GPTQ sequential ordering:
//!   u32 encodes [v0, v1, v2, v3, v4, v5, v6, v7] in nibbles 0..7
//!
//! Applies `undo_pack = {0, 4, 1, 5, 2, 6, 3, 7}` per u32 word.
//!
//! References:
//! - vLLM: `vllm/model_executor/layers/quantization/awq_marlin.py`
//! - CUDA kernel: `csrc/quantization/marlin/awq_marlin_repack.cu`

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "marlin")]
use super::marlin_cuda;

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::marlin::{MarlinConfig, MarlinLinear};

// ─── AwqMarlinConfig ─────────────────────────────────────────────────────────

/// AWQ-Marlin quantization configuration.
///
/// Asymmetric INT4 with group scales and zero points, accelerated by the
/// Marlin kernel.  Requires GPU compute capability ≥ 80 (Ampere).
#[derive(Debug, Clone)]
pub struct AwqMarlinConfig {
    inner: MarlinConfig,
    /// Whether lm_head is quantized (false = skip lm_head).
    lm_head_quantized: bool,
}

impl AwqMarlinConfig {
    /// Create from detected quantization fields.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        // AWQ-Marlin only supports 4-bit in vLLM; default to 4 for other bit widths.
        let _ = bits; // preserved for API symmetry
        let group_size = group_size.map(|g| g as i32).unwrap_or(128);
        let inner = MarlinConfig::awq_int4(group_size);

        let lm_head_quantized = raw_config
            .get("lm_head_quantized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            inner,
            lm_head_quantized,
        }
    }

    /// Access the underlying Marlin config.
    pub fn marlin_config(&self) -> &MarlinConfig {
        &self.inner
    }

    /// Group size used for quantization.
    pub fn group_size(&self) -> i32 {
        self.inner.group_size
    }
}

impl QuantizationConfig for AwqMarlinConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::AwqMarlin
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        80 // Marlin requires Ampere
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        !self.lm_head_quantized && layer_name.contains("lm_head")
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(AwqMarlinLinear::new(
            in_features,
            out_features,
            bias,
            self.inner.clone(),
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── AWQ nibble deinterleaving ────────────────────────────────────────────────

/// Repack `qweight` per-word: AWQ interleaved nibble order → GPTQ sequential.
///
/// Shape is preserved.  Suitable for `qzeros` (where AWQ and GPTQ share
/// the same `[num_groups, N/8]` layout) but **NOT** for `qweight`, where
/// AWQ packs along the output axis (`[K, N/8]`) and GPTQ packs along the
/// input axis (`[K/8, N]`) — for that, use [`awq_to_gptq_qweight`].
///
/// Applies `undo_pack = {0,4,1,5,2,6,3,7}` per 32-bit word.
pub(crate) fn repack_awq_nibbles(qweight: &Tensor) -> Result<Tensor> {
    let shape = qweight.shape().clone();
    let device = qweight.device().clone();

    // GPU fast path: parallel per-word transform via CUDA kernel.
    // Any sm_80+ device is supported (the kernel uses only integer ops).
    #[cfg(feature = "marlin")]
    if device.is_cuda() {
        let dims = shape.dims();
        if dims.len() == 2 {
            // qweight shape [K/8, N]: reconstruct size_k and size_n
            return marlin_cuda::awq_marlin_repack(qweight, dims[0] * 8, dims[1]);
        }
    }

    // CPU fallback: bring to host, transform word-by-word, return to original device.
    let flat = qweight
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let words = flat.to_vec1::<u32>()?;

    let reordered: Vec<u32> = words.iter().map(|&w| awq_to_gptq_u32(w)).collect();

    Tensor::from_vec(reordered, &shape, &Device::Cpu)?.to_device(&device)
}

/// Repack AWQ `qweight` (`[K, N/8]`, packed along the output axis with
/// interleaved nibble ordering) into GPTQ sequential layout (`[K/8, N]`,
/// packed along the input axis).
///
/// This is what vLLM's `awq_marlin_repack` CUDA kernel does; we provide
/// a CPU implementation here so the loader can run AWQ checkpoints
/// through the Marlin kernel (or GPTQ fallback) without any per-forward
/// dequantization.  The transform is one-shot at load time.
///
/// Steps:
/// 1. For each `(k, n)`, extract the int4 nibble from the AWQ word
///    `qweight_awq[k, n/8]` at position `undo_pack[n % 8]`.
/// 2. Pack 8 consecutive `k`-values (k = k_block * 8 + i, i in 0..8) into
///    a single u32 in **sequential** nibble order:
///    `gptq_word = sum_i nib(k_block*8 + i, n) << (i * 4)`.
///
/// Result lives on the same device as the input.
pub(crate) fn awq_to_gptq_qweight(
    awq_qweight: &Tensor,
    in_features: usize,
    out_features: usize,
) -> Result<Tensor> {
    const PACK_FACTOR: usize = 8;
    if !in_features.is_multiple_of(PACK_FACTOR) {
        candle_core::bail!(
            "awq_to_gptq_qweight: in_features ({in_features}) must be divisible by 8"
        );
    }
    if !out_features.is_multiple_of(PACK_FACTOR) {
        candle_core::bail!(
            "awq_to_gptq_qweight: out_features ({out_features}) must be divisible by 8"
        );
    }
    let packed_n = out_features / PACK_FACTOR;
    let packed_k = in_features / PACK_FACTOR;

    if awq_qweight.dtype() != DType::U32 {
        candle_core::bail!(
            "awq_to_gptq_qweight: qweight must be U32, got {:?}",
            awq_qweight.dtype()
        );
    }
    if awq_qweight.dims() != [in_features, packed_n] {
        candle_core::bail!(
            "awq_to_gptq_qweight: qweight shape {:?} != expected [{in_features}, {packed_n}]",
            awq_qweight.dims()
        );
    }

    // GPU fast path: a single CUDA kernel that does both the nibble
    // deinterleave and the packing-axis transpose. Replaces a strided
    // scalar Rust loop on the host that took 5+ minutes for Qwen3-4B
    // (252 linear layers × ~12M iterations each, every read on a fresh
    // cache line).
    #[cfg(feature = "marlin")]
    if awq_qweight.device().is_cuda() {
        return super::marlin_cuda::awq_to_gptq_qweight_cuda(
            awq_qweight,
            in_features,
            out_features,
        );
    }

    let original_device = awq_qweight.device().clone();
    let words: Vec<u32> = awq_qweight
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let mut out = vec![0u32; packed_k * out_features];
    for n in 0..out_features {
        let undo_shift = AWQ_UNDO_PACK_SHIFTS[n % PACK_FACTOR];
        let n_block = n / PACK_FACTOR;
        for kb in 0..packed_k {
            let k_base = kb * PACK_FACTOR;
            let mut word: u32 = 0;
            for i in 0..PACK_FACTOR {
                let k = k_base + i;
                let awq_word = words[k * packed_n + n_block];
                let nib = (awq_word >> undo_shift) & 0xF;
                word |= nib << (i * 4);
            }
            out[kb * out_features + n] = word;
        }
    }

    Tensor::from_vec(out, (packed_k, out_features), &Device::Cpu)?.to_device(&original_device)
}

/// CPU port of vLLM's `awq_marlin_repack_kernel` (INT4 path, `is_a_8bit=false`).
///
/// Reads AWQ qweight `[size_k, size_n / 8]` (u32, output-axis packed,
/// interleaved nibble ordering) and produces Marlin tile-laid-out qweight
/// `[size_k / 16, size_n * 16 / 8]` = `[size_k / 16, size_n * 2]` u32 ready
/// for the tensor-core `mma.m16n8k16` kernel.
///
/// **One-shot.** This function combines AWQ undo_pack and Marlin tile
/// permutation in a single pass; the caller does not need to invoke
/// [`repack_awq_nibbles`] or [`awq_to_gptq_qweight`] beforehand.
///
/// Tile sizes (from `reference/vllm/csrc/quantization/marlin/marlin.cuh`):
/// - `tile_k_size = 16`, `tile_n_size = 64`, `pack_factor = 8` (INT4)
/// - Each tile holds 16×64 = 1024 nibbles = 128 u32 in the output.
/// - Output stride per k-tile-row: `n_tiles × 128` = `(size_n / 64) × 128`
///   = `size_n × 2` u32.
///
/// Per-tile thread-block layout in the reference kernel: 4 warps × 32 threads
/// = 128 thread-output-positions, each writing one u32 at offset
/// `th_id * 4 + warp_id` within the tile. We replicate that mapping so the
/// output bytes match the kernel byte-for-byte.
///
/// Reference: `csrc/quantization/marlin/awq_marlin_repack.cu` lines 70–151.
///
/// # Arguments
/// * `awq_qweight` — u32 tensor with shape `[size_k, size_n / 8]`.
/// * `size_k` — input feature count (must be divisible by 16).
/// * `size_n` — output feature count (must be divisible by 64).
///
/// # Returns
/// u32 tensor with shape `[size_k / 16, size_n * 2]` on the same device as
/// the input. Currently CPU-only; the caller should `.to_device(...)` if a
/// CUDA copy is needed (a CUDA-side mirror lives in `marlin_cuda` for
/// load-time GPU repack and is wired separately at Stage 15.E).
pub fn awq_to_marlin_tile_repack_cpu(
    awq_qweight: &Tensor,
    size_k: usize,
    size_n: usize,
) -> Result<Tensor> {
    const PACK_FACTOR: usize = 8;
    const TILE_K: usize = 16;
    const TILE_N: usize = 64;
    const TILE_INTS: usize = TILE_K * TILE_N / PACK_FACTOR; // 128

    // Reference kernel: undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7}; pack_idx[8] =
    // {0, 2, 4, 6, 1, 3, 5, 7}. Hardcoded for INT4. tc_offsets[4] = {0, 1, 8, 9}
    // identifies the 4 k-rows each thread reads within the 16-row k-tile.
    const UNDO_PACK: [u32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    const PACK_IDX: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    const TC_OFFSETS: [usize; 4] = [0, 1, 8, 9];

    if !size_k.is_multiple_of(TILE_K) {
        candle_core::bail!(
            "awq_to_marlin_tile_repack_cpu: size_k ({size_k}) must be divisible by {TILE_K}"
        );
    }
    if !size_n.is_multiple_of(TILE_N) {
        candle_core::bail!(
            "awq_to_marlin_tile_repack_cpu: size_n ({size_n}) must be divisible by {TILE_N}"
        );
    }
    if awq_qweight.dtype() != DType::U32 {
        candle_core::bail!(
            "awq_to_marlin_tile_repack_cpu: qweight must be U32, got {:?}",
            awq_qweight.dtype()
        );
    }
    let packed_n = size_n / PACK_FACTOR;
    if awq_qweight.dims() != [size_k, packed_n] {
        candle_core::bail!(
            "awq_to_marlin_tile_repack_cpu: qweight shape {:?} != expected [{size_k}, {packed_n}]",
            awq_qweight.dims()
        );
    }

    let original_device = awq_qweight.device().clone();
    let words: Vec<u32> = awq_qweight
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let k_tiles = size_k / TILE_K;
    let n_tiles = size_n / TILE_N;
    let out_n_per_row = n_tiles * TILE_INTS; // = size_n * 2
    let mut out = vec![0u32; k_tiles * out_n_per_row];

    for k_tile_id in 0..k_tiles {
        let first_k = k_tile_id * TILE_K;
        for n_tile_id in 0..n_tiles {
            let first_n_packed = (n_tile_id * TILE_N) / PACK_FACTOR; // ×8 packed cols/tile
            let tile_out_base = k_tile_id * out_n_per_row + n_tile_id * TILE_INTS;

            // Replicate the reference kernel's 4 warps × 32 threads layout.
            for warp_id in 0..4 {
                for th_id in 0..32 {
                    let tc_col = th_id / 4; // 0..7
                    let tc_row = (th_id % 4) * 2; // 0,2,4,6
                    let cur_n = warp_id * 16 + tc_col;
                    let cur_n_packed = cur_n / PACK_FACTOR;
                    let cur_n_pos = cur_n % PACK_FACTOR;
                    let undo_shift = UNDO_PACK[cur_n_pos] * 4;

                    let mut vals = [0u32; 8];
                    for i in 0..4 {
                        let cur_elem = tc_row + TC_OFFSETS[i];
                        let row_base = (first_k + cur_elem) * packed_n + first_n_packed;
                        let packed_src_0 = words[row_base + cur_n_packed];
                        let packed_src_1 = words[row_base + cur_n_packed + 1];
                        vals[i] = (packed_src_0 >> undo_shift) & 0xF;
                        vals[4 + i] = (packed_src_1 >> undo_shift) & 0xF;
                    }

                    let mut res: u32 = 0;
                    for j in 0..8 {
                        res |= vals[PACK_IDX[j]] << (j * 4);
                    }

                    out[tile_out_base + th_id * 4 + warp_id] = res;
                }
            }
        }
    }

    Tensor::from_vec(out, (k_tiles, out_n_per_row), &Device::Cpu)?.to_device(&original_device)
}

/// Bit-shift positions per output lane in an AWQ packed word.
///
/// AWQ encodes 8 int4 outputs in a u32 with `undo_pack = {0,4,1,5,2,6,3,7}`,
/// so output `i` lives at bit offset `undo_pack[i] * 4`.
const AWQ_UNDO_PACK_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];

/// Convert one u32 word from AWQ interleaved nibble ordering to GPTQ
/// sequential nibble ordering.
///
/// `undo_pack = {0, 4, 1, 5, 2, 6, 3, 7}`:
/// output nibble `i` ← AWQ nibble `undo_pack[i]`.
#[inline]
fn awq_to_gptq_u32(w: u32) -> u32 {
    let n0 = (w) & 0xF;
    let n1 = (w >> 4) & 0xF;
    let n2 = (w >> 8) & 0xF;
    let n3 = (w >> 12) & 0xF;
    let n4 = (w >> 16) & 0xF;
    let n5 = (w >> 20) & 0xF;
    let n6 = (w >> 24) & 0xF;
    let n7 = (w >> 28) & 0xF;
    // output = [n0, n4, n1, n5, n2, n6, n3, n7]
    n0 | (n4 << 4) | (n1 << 8) | (n5 << 12) | (n2 << 16) | (n6 << 20) | (n3 << 24) | (n7 << 28)
}

// ─── AwqMarlinLinear ─────────────────────────────────────────────────────────

/// Linear layer that loads AWQ checkpoints and runs through Marlin
/// (or, when the `marlin` feature is off, through the GPTQ CUDA fallback).
///
/// On `load_weights`, this layer:
/// 1. Repacks AWQ `qweight` (`[K, N/8]`, output-axis interleaved) into
///    GPTQ sequential layout (`[K/8, N]`) via [`awq_to_gptq_qweight`].
/// 2. Repacks AWQ `qzeros` (`[num_groups, N/8]`) into sequential nibble
///    ordering via [`repack_awq_nibbles`] (shape preserved).
/// 3. Hands the GPTQ-style tensors to the inner [`MarlinLinear`], whose
///    own `load_weights` then performs the GPTQ → Marlin tile repack.
///
/// All forward passes delegate to `MarlinLinear::forward`, which uses
/// the Marlin INT4 kernel when `feature = "marlin"`, otherwise falls
/// back to `gptq_cuda::gptq_gemm` (via `cuda-kernels`).  Either way is
/// dramatically faster than `AwqLinear`'s per-forward CPU dequant.
///
/// **Stage 15.E.1 — hybrid dispatch dual storage.** When the
/// `marlin` feature and a CUDA device are available, `load_weights`
/// also computes a Marlin tile-laid-out qweight (`tile_b`) from the
/// AWQ-original `qweight` via [`awq_to_marlin_tile_repack_cpu`]. The
/// production forward path (`MarlinLinear::forward` →
/// `marlin_gemm`) ignores it; Stage 15.E.2 adds a forward-time
/// dispatch that routes M ≤ 8 calls to
/// `dispatch_marlin_tile_mma_v1` for a 1.1–1.24× win on the
/// AWQ-Marlin GEMV regime (see ADR 0016).
#[derive(Debug)]
pub struct AwqMarlinLinear {
    inner: MarlinLinear,
    in_features: usize,
    out_features: usize,
    /// Marlin tile-laid-out qweight, populated when the `marlin`
    /// feature is on and the load-device is CUDA. `None` otherwise
    /// (or on layers whose `(K, N)` violate the `awq_to_marlin_tile_
    /// repack_cpu` shape constraints — `K % 16 == 0`, `N % 64 == 0`).
    tile_b: Option<Tensor>,
}

impl AwqMarlinLinear {
    /// Create a new AWQ-Marlin linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        marlin_config: MarlinConfig,
        device: &Device,
    ) -> Result<Self> {
        let inner = MarlinLinear::new(in_features, out_features, has_bias, marlin_config, device)?;
        Ok(Self {
            inner,
            in_features,
            out_features,
            tile_b: None,
        })
    }

    /// Stage 15.E.2 hook: tile_b for the hybrid M ≤ 8 dispatch path.
    /// Returns `None` until `load_weights` has run on a hybrid-eligible
    /// (CUDA, `marlin` feature, valid shape) layer.
    #[allow(dead_code)]
    pub(crate) fn tile_b(&self) -> Option<&Tensor> {
        self.tile_b.as_ref()
    }
}

/// Stage 15.E.2 hybrid dispatch threshold (ADR 0016). At M ≤ this value the
/// software tile-MMA path beats production `marlin_gemm` by 1.10–1.24× on
/// the canonical Qwen3-4B-AWQ MLP-up shape (commit `8dc4087` bench).
#[cfg(feature = "marlin")]
pub(crate) const HYBRID_M_THRESHOLD: usize = 8;

/// Returns `true` if hybrid dispatch is enabled via env opt-in.
/// **Default is OFF** — turning it on doubles the qweight footprint at
/// load time (Stage 15.E.3 measured +2.2 GiB on Qwen3-4B-AWQ vs the
/// ADR 0016 prediction of +1.1 GiB; the laptop ran out of KV-cache room
/// at default settings). Users with VRAM headroom can opt in via
/// `VLLM_AWQ_HYBRID=1` for the 1.10-1.24× decode-side win.
///
/// Read once per process — re-checking on every forward call would cost
/// a syscall per layer per token at decode hot path.
#[cfg(feature = "marlin")]
fn hybrid_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VLLM_AWQ_HYBRID").is_ok())
}

impl QuantizedLinear for AwqMarlinLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Stage 15.E.2 hybrid dispatch:
        //   M ≤ 8 + tile_b populated → tile_mma_v1 software path
        //                              (1.10–1.24× win on the AWQ-Marlin slice)
        //   else → existing MarlinLinear → marlin_gemm path (saturated cuBLAS
        //          GEMM for M ≥ 16, dequant+matmul for the M=12-16 crossover)
        // tile_b is only populated when `VLLM_AWQ_HYBRID=1` is set at load
        // time, so this branch is dead under the default configuration —
        // production behaviour unchanged. ADR 0016 documents the rationale
        // and bench data; the env opt-in protects laptop / VRAM-tight
        // deployments from the +2.2 GiB load-time cost.
        #[cfg(feature = "marlin")]
        {
            if let Some(tile_b) = self.tile_b.as_ref() {
                if x.dims().len() == 2 {
                    let m = x.dims()[0];
                    if m > 0 && m <= HYBRID_M_THRESHOLD {
                        let group_size_i = self.inner.config_ref().group_size;
                        // Per-channel scales (group_size = -1) are
                        // disallowed by the tile_mma_v1 dispatcher; drop
                        // through to the existing path on those layers.
                        if group_size_i > 0 {
                            let group_size = group_size_i as usize;
                            return super::marlin_tile_cuda::dispatch_marlin_tile_mma_v1(
                                x,
                                tile_b,
                                self.inner.scales_ref(),
                                self.inner.qzeros_ref(),
                                self.out_features,
                                group_size,
                            );
                        }
                    }
                }
            }
        }

        self.inner.forward(x)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                target: "vllm_core::marlin_path",
                "AwqMarlinLinear::load_weights live (in={}, out={})",
                self.in_features, self.out_features
            );
        });

        let mut repacked: HashMap<String, Tensor> = HashMap::with_capacity(weights.len());

        if let Some(qw) = weights.get("qweight") {
            // Stage 15.E.1/3: compute the Marlin tile layout BEFORE the GPTQ
            // transpose consumes the AWQ-original `[K, N/8]` shape. Gated on
            // (a) the `marlin` feature, (b) the `VLLM_AWQ_HYBRID=1` env opt-in
            // (15.E.3 measured +2.2 GiB VRAM on Qwen3-4B vs the ADR 0016
            // prediction of +1.1 GiB; default is OFF to preserve KV-cache
            // budget on memory-tight devices), and (c) shape divisibility
            // (Marlin tile requires `K % 16 == 0`, `N % 64 == 0`).
            #[cfg(feature = "marlin")]
            {
                // `cfg(test)` keeps the unit tests below independent of
                // the env var (OnceLock would otherwise pin the first
                // observed state for the whole test process).
                let gate_open = cfg!(test) || hybrid_enabled();
                if gate_open
                    && self.in_features.is_multiple_of(16)
                    && self.out_features.is_multiple_of(64)
                {
                    let qw_cpu = qw.to_dtype(DType::U32)?.to_device(&Device::Cpu)?;
                    let tile = awq_to_marlin_tile_repack_cpu(
                        &qw_cpu,
                        self.in_features,
                        self.out_features,
                    )?;
                    let tile_dev = tile.to_device(qw.device())?;
                    self.tile_b = Some(tile_dev);
                }
            }

            // Step 1: AWQ interleaved nibbles → GPTQ sequential `[K/8, N]`.
            let gptq_qw = awq_to_gptq_qweight(qw, self.in_features, self.out_features)?;

            // Step 2: transpose to `[N, K/8]` so the K-axis sits in the
            // contiguous dimension. This unlocks vec4 (uint4) global loads
            // along K in the AWQ GEMV kernel — single packed_k step now
            // reads 4 u32s = 32 nibbles per LDG, quartering the number of
            // memory transactions per output column at the same coalesced
            // bandwidth budget.
            //
            // `t()?.contiguous()?` materialises the transposed layout once
            // at load time (a single GPU strided-copy); every subsequent
            // forward reads it directly. The decode kernel
            // `awq_gemv_int4_kt_bf16` consumes this layout.
            let qweight_kt = gptq_qw.t()?.contiguous()?;
            repacked.insert("qweight".to_string(), qweight_kt);
        }
        if let Some(qz) = weights.get("qzeros") {
            // qzeros share the same shape between AWQ and GPTQ
            // (`[num_groups, N/8]`); only the nibble order inside each
            // u32 word changes.
            let gptq_qz = repack_awq_nibbles(qz)?;
            repacked.insert("qzeros".to_string(), gptq_qz);
        }
        // HuggingFace AWQ checkpoints store `scales` in F16. The Marlin
        // and decode-GEMV kernels both consume BF16 scales (the activation
        // dtype on Ampere+); convert at load time so each forward avoids
        // a per-call cast.
        if let Some(s) = weights.get("scales") {
            let scales = if s.dtype() == DType::BF16 {
                s.clone()
            } else {
                s.to_dtype(DType::BF16)?
            };
            repacked.insert("scales".to_string(), scales);
        }
        if let Some(b) = weights.get("bias") {
            let bias = if b.dtype() == DType::BF16 {
                b.clone()
            } else {
                b.to_dtype(DType::BF16)?
            };
            repacked.insert("bias".to_string(), bias);
        }

        self.inner.load_weights(&repacked)
    }

    fn weight_dtype(&self) -> DType {
        self.inner.weight_dtype()
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.inner.has_bias()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_awq_marlin_config_default() {
        let config = AwqMarlinConfig::from_detected(Some(4), Some(128), &HashMap::new());
        assert_eq!(config.method(), QuantizationMethod::AwqMarlin);
        assert_eq!(config.group_size(), 128);
        assert_eq!(config.min_capability(), 80);
    }

    #[test]
    fn test_awq_marlin_config_lm_head_skip() {
        let config = AwqMarlinConfig::from_detected(Some(4), Some(128), &HashMap::new());
        // By default lm_head is skipped.
        assert!(config.is_layer_skipped("lm_head"));
        assert!(!config.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn test_awq_marlin_config_lm_head_quantized() {
        let mut raw = HashMap::new();
        raw.insert(
            "lm_head_quantized".to_string(),
            serde_json::Value::Bool(true),
        );
        let config = AwqMarlinConfig::from_detected(Some(4), Some(128), &raw);
        assert!(!config.is_layer_skipped("lm_head"));
    }

    #[test]
    fn test_awq_to_gptq_u32_identity_all_zero() {
        // All-zero word stays zero.
        assert_eq!(awq_to_gptq_u32(0x0000_0000), 0x0000_0000);
    }

    #[test]
    fn test_awq_to_gptq_u32_known_values() {
        // Pack 8 distinct nibbles in AWQ order: v=[1,2,3,4,5,6,7,8]
        // AWQ u32: n0=v[0]=1, n1=v[2]=3, n2=v[4]=5, n3=v[6]=7,
        //          n4=v[1]=2, n5=v[3]=4, n6=v[5]=6, n7=v[7]=8
        let awq: u32 =
            1 | (3 << 4) | (5 << 8) | (7 << 12) | (2 << 16) | (4 << 20) | (6 << 24) | (8 << 28);
        let gptq = awq_to_gptq_u32(awq);
        // GPTQ u32: [v0,v1,v2,v3,v4,v5,v6,v7] = [1,2,3,4,5,6,7,8]
        let expected: u32 =
            1 | (2 << 4) | (3 << 8) | (4 << 12) | (5 << 16) | (6 << 20) | (7 << 24) | (8 << 28);
        assert_eq!(gptq, expected);
    }

    #[test]
    fn test_repack_awq_nibbles_shape_preserved() {
        let device = Device::Cpu;
        // qweight shape for int4: [K/8, N] — use K=128, N=64 → [16, 64]
        let qweight = Tensor::zeros((16usize, 64usize), DType::U32, &device).unwrap();
        let result = repack_awq_nibbles(&qweight).unwrap();
        assert_eq!(result.dims(), &[16, 64]);
    }

    #[test]
    fn test_awq_to_gptq_qweight_shape_transposes_packing_axis() {
        // AWQ qweight shape: [K, N/8]; GPTQ output: [K/8, N].
        let device = Device::Cpu;
        let k = 64usize;
        let n = 32usize;
        let qw = Tensor::zeros((k, n / 8), DType::U32, &device).unwrap();
        let gptq = awq_to_gptq_qweight(&qw, k, n).unwrap();
        assert_eq!(gptq.dims(), &[k / 8, n]);
    }

    #[test]
    fn test_awq_to_gptq_qweight_known_value() {
        // Build a 1×1 (in nibble units) AWQ matrix where K=8, N=8.
        // AWQ shape: [K=8, N/8=1]. Each row k stores 8 nibbles for the
        // 8 outputs of column-block 0, in undo_pack order.
        // GPTQ shape after repack: [K/8=1, N=8]. Column n stores 8
        // nibbles for inputs k=0..7, in sequential order.
        //
        // Use a unique value per (k, n): nib(k, n) = (k + 1) (mod 16),
        // independent of n. After repack, every GPTQ word equals
        // sum_i (i+1) << (i*4).
        let device = Device::Cpu;
        let k = 8usize;
        let n = 8usize;
        let mut awq_words = Vec::with_capacity(k);
        for ki in 0..k {
            let v = ((ki + 1) & 0xF) as u32;
            // All 8 output lanes hold the same value v, so the AWQ word
            // is v repeated in every nibble — identical regardless of
            // undo_pack order.
            let mut word: u32 = 0;
            for shift in (0..32).step_by(4) {
                word |= v << shift;
            }
            awq_words.push(word);
        }
        let awq = Tensor::from_vec(awq_words, (k, n / 8), &device).unwrap();
        let gptq = awq_to_gptq_qweight(&awq, k, n).unwrap();
        let out: Vec<u32> = gptq.flatten_all().unwrap().to_vec1().unwrap();

        let mut expected_word: u32 = 0;
        for ki in 0..k {
            let v = ((ki + 1) & 0xF) as u32;
            expected_word |= v << (ki * 4);
        }
        assert_eq!(out.len(), n);
        for &w in &out {
            assert_eq!(w, expected_word);
        }
    }

    #[test]
    fn test_repack_awq_nibbles_idempotent_all_zero() {
        // All-zero input: deinterleave of zeros is zeros.
        let device = Device::Cpu;
        let qweight = Tensor::zeros((8usize, 32usize), DType::U32, &device).unwrap();
        let result = repack_awq_nibbles(&qweight).unwrap();
        let flat: Vec<u32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&v| v == 0));
    }

    /// Stage 15.E.1: `tile_b` is populated when the layer's shape clears
    /// the Marlin tile constraints (`K % 16 == 0`, `N % 64 == 0`) and the
    /// `marlin` feature is on. Output shape: `[K/16, N*2]` per the
    /// `awq_to_marlin_tile_repack_cpu` contract.
    ///
    /// Note: MarlinLinear's own constructor enforces `K % 128 == 0`,
    /// `N % 64 == 0`, and a supported group_size — so any AwqMarlinLinear
    /// that builds at all already satisfies the strictly weaker
    /// awq_to_marlin_tile_repack_cpu shape constraints. The
    /// `is_multiple_of` check inside load_weights is purely defensive.
    #[cfg(feature = "marlin")]
    #[test]
    fn test_awq_marlin_linear_load_populates_tile_b() {
        let device = Device::Cpu;
        let in_features = 128usize; // K, mult of 128 (Marlin min)
        let out_features = 128usize; // N, mult of 64
        let group_size = 128i32; // K = 1 group

        let cfg = MarlinConfig::awq_int4(group_size);
        let mut layer =
            AwqMarlinLinear::new(in_features, out_features, false, cfg, &device).unwrap();

        // Synthesize the AWQ-original weights with the right shapes.
        let qweight = Tensor::zeros((in_features, out_features / 8), DType::U32, &device).unwrap();
        let scales = Tensor::zeros(
            (in_features / group_size as usize, out_features),
            DType::F16,
            &device,
        )
        .unwrap();
        let qzeros = Tensor::zeros(
            (in_features / group_size as usize, out_features / 8),
            DType::U32,
            &device,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert("qweight".to_string(), qweight);
        weights.insert("scales".to_string(), scales);
        weights.insert("qzeros".to_string(), qzeros);

        layer.load_weights(&weights).unwrap();

        let tile_b = layer.tile_b().expect("tile_b populated");
        let k_tiles = in_features / 16;
        let n_tiles = out_features / 64;
        let expected_inner = n_tiles * 128;
        assert_eq!(tile_b.dims(), &[k_tiles, expected_inner]);
        assert_eq!(tile_b.dtype(), DType::U32);
    }

    // ─── awq_to_marlin_tile_repack_cpu ───────────────────────────────────

    /// Bit offset of logical nibble at column `n` in an AWQ-packed u32.
    /// Mirrors `AWQ_UNDO_PACK_SHIFTS` but keyed by logical column rather
    /// than output lane (they happen to be the same array since both are
    /// indexed by the AWQ undo_pack permutation).
    const AWQ_PACK_SHIFTS: [u32; 8] = AWQ_UNDO_PACK_SHIFTS;

    /// Predicted output (u32_idx_in_tile, bit_offset) for a logical nibble
    /// at (k_in_tile ∈ 0..16, n_in_tile ∈ 0..64) — derived independently from
    /// the kernel comments at `awq_marlin_repack.cu:75-150` so the round-trip
    /// test is not circular.
    fn marlin_tile_decode_position(k_in_tile: usize, n_in_tile: usize) -> (usize, u32) {
        const INV_PACK_IDX: [u32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
        let m_n = n_in_tile % 16;
        let warp_id = n_in_tile / 16;
        let (tc_col, val_offset_high) = if m_n < 8 { (m_n, 0) } else { (m_n - 8, 4) };
        let tc_row = ((k_in_tile / 2) % 4) * 2;
        let i_inner = (k_in_tile % 2) + (k_in_tile / 8) * 2;
        let th_id = tc_row / 2 + tc_col * 4;
        let val_idx = i_inner + val_offset_high;
        let u32_idx_in_tile = th_id * 4 + warp_id;
        let bit_offset = INV_PACK_IDX[val_idx] * 4;
        (u32_idx_in_tile, bit_offset)
    }

    #[test]
    fn test_awq_to_marlin_tile_repack_cpu_shape() {
        // K=32 (2 k-tiles), N=128 (2 n-tiles) → output [2, 256].
        let device = Device::Cpu;
        let q = Tensor::zeros((32usize, 16usize), DType::U32, &device).unwrap();
        let out = awq_to_marlin_tile_repack_cpu(&q, 32, 128).unwrap();
        assert_eq!(out.dims(), &[2, 256]);
        assert_eq!(out.dtype(), DType::U32);
        let flat: Vec<u32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_awq_to_marlin_tile_repack_cpu_rejects_bad_shape() {
        let device = Device::Cpu;
        // size_k not divisible by 16
        let q = Tensor::zeros((15usize, 8usize), DType::U32, &device).unwrap();
        assert!(awq_to_marlin_tile_repack_cpu(&q, 15, 64).is_err());
        // size_n not divisible by 64
        let q = Tensor::zeros((16usize, 5usize), DType::U32, &device).unwrap();
        assert!(awq_to_marlin_tile_repack_cpu(&q, 16, 40).is_err());
        // wrong dtype
        let q = Tensor::zeros((16usize, 8usize), DType::F32, &device).unwrap();
        assert!(awq_to_marlin_tile_repack_cpu(&q, 16, 64).is_err());
        // wrong shape
        let q = Tensor::zeros((16usize, 16usize), DType::U32, &device).unwrap();
        assert!(awq_to_marlin_tile_repack_cpu(&q, 16, 64).is_err());
    }

    #[test]
    fn test_awq_to_marlin_tile_repack_cpu_round_trip_single_tile() {
        // 1 tile (K=16, N=64). Pack a deterministic logical nibble at every
        // (k, n) into AWQ format, run the repack, then for each (k, n)
        // decode the predicted output position and assert the nibble matches.
        let device = Device::Cpu;
        let k: usize = 16;
        let n: usize = 64;
        let packed_n = n / 8;

        let nibble = |row: usize, col: usize| -> u32 { ((row * n + col) as u32 * 7 + 3) % 16 };

        // AWQ pack: awq[k][n/8] holds 8 nibbles at AWQ_PACK_SHIFTS positions.
        let mut awq = vec![0u32; k * packed_n];
        for kk in 0..k {
            for nn in 0..n {
                let shift = AWQ_PACK_SHIFTS[nn % 8];
                awq[kk * packed_n + nn / 8] |= nibble(kk, nn) << shift;
            }
        }
        let q = Tensor::from_vec(awq, (k, packed_n), &device).unwrap();
        let out = awq_to_marlin_tile_repack_cpu(&q, k, n).unwrap();
        assert_eq!(out.dims(), &[1, 128]);
        let out_flat: Vec<u32> = out.flatten_all().unwrap().to_vec1().unwrap();

        for kk in 0..k {
            for nn in 0..n {
                let (u32_idx, bit_off) = marlin_tile_decode_position(kk, nn);
                let actual = (out_flat[u32_idx] >> bit_off) & 0xF;
                let expected = nibble(kk, nn);
                assert_eq!(
                    actual, expected,
                    "mismatch at (k={kk}, n={nn}): u32_idx={u32_idx} bit_off={bit_off}, \
                     expected {expected} got {actual}, raw_u32=0x{:08x}",
                    out_flat[u32_idx]
                );
            }
        }
    }

    #[test]
    fn test_awq_to_marlin_tile_repack_cpu_round_trip_multi_tile() {
        // Multi-tile: K=32 (2 k-tiles), N=128 (2 n-tiles) → output [2, 256].
        // Same round-trip invariant; this tile addressing exercises both
        // tile-axes and the per-tile output stride.
        let device = Device::Cpu;
        let k: usize = 32;
        let n: usize = 128;
        let packed_n = n / 8;

        let nibble = |row: usize, col: usize| -> u32 { ((row * 1023 + col * 17 + 5) as u32) % 16 };

        let mut awq = vec![0u32; k * packed_n];
        for kk in 0..k {
            for nn in 0..n {
                awq[kk * packed_n + nn / 8] |= nibble(kk, nn) << AWQ_PACK_SHIFTS[nn % 8];
            }
        }
        let q = Tensor::from_vec(awq, (k, packed_n), &device).unwrap();
        let out = awq_to_marlin_tile_repack_cpu(&q, k, n).unwrap();
        assert_eq!(out.dims(), &[2, 256]);
        let out_flat: Vec<u32> = out.flatten_all().unwrap().to_vec1().unwrap();

        for kk in 0..k {
            for nn in 0..n {
                let k_tile = kk / 16;
                let k_in_tile = kk % 16;
                let n_tile = nn / 64;
                let n_in_tile = nn % 64;
                let (u32_idx_in_tile, bit_off) = marlin_tile_decode_position(k_in_tile, n_in_tile);
                // Output row = k_tile, col within row = n_tile * 128 + u32_idx_in_tile.
                let row = k_tile;
                let col = n_tile * 128 + u32_idx_in_tile;
                let raw = out_flat[row * 256 + col];
                let actual = (raw >> bit_off) & 0xF;
                let expected = nibble(kk, nn);
                assert_eq!(
                    actual, expected,
                    "mismatch at (k={kk}, n={nn}, k_tile={k_tile}, n_tile={n_tile})"
                );
            }
        }
    }

    /// Numeric equivalence between the GPU `awq_to_gptq_qweight_cuda`
    /// kernel (Stage 8) and the CPU reference implementation. Gated to
    /// the gpu-test-medium tier since it needs a real CUDA device.
    #[cfg(all(feature = "marlin", feature = "gpu-test-medium"))]
    #[test]
    fn test_awq_to_gptq_qweight_gpu_matches_cpu() {
        // Use a non-trivial shape that exercises both axes:
        // K=64 (8 packed_k), N=128 (16 packed_n). Random-ish nibble
        // pattern so we don't accidentally pass on a degenerate case.
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device available — skipping");
            return;
        };
        let k = 64usize;
        let n = 128usize;
        let packed_n = n / 8;
        let mut words: Vec<u32> = Vec::with_capacity(k * packed_n);
        let mut rng_state: u32 = 0x9E37_79B9;
        for _ in 0..(k * packed_n) {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            words.push(rng_state);
        }

        let awq_cpu = Tensor::from_vec(words.clone(), (k, packed_n), &Device::Cpu).unwrap();
        let awq_gpu = Tensor::from_vec(words, (k, packed_n), &cuda).unwrap();

        // CPU reference (force CPU path by passing CPU tensor) vs GPU.
        let gptq_cpu = awq_to_gptq_qweight(&awq_cpu, k, n).unwrap();
        let gptq_gpu = awq_to_gptq_qweight(&awq_gpu, k, n).unwrap();

        let cpu_vec: Vec<u32> = gptq_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let gpu_vec: Vec<u32> = gptq_gpu
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            cpu_vec, gpu_vec,
            "GPU awq_to_gptq_qweight kernel must be bit-exact with CPU reference"
        );
    }
}
