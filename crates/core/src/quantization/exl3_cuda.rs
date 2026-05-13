//! Low-level CUDA dispatchers for the vendored EXL3 kernels.
//!
//! Layout mirrors `marlin_tile_cuda.rs`: PTX is embedded via
//! `include_str!`, loaded once per device by cudarc, and exposed
//! through `CustomOp1` wrappers so candle handles the storage/shape
//! plumbing.
//!
//! Phase-3 scope: `had_r_128` (Hadamard butterfly on 128-element
//! groups, fp16). Reconstruct + the GEMM/GEMV dispatchers land in
//! Phase 4. The implementation here is capture-friendly — each launch
//! writes into a pool-reserved buffer or in-place, never via
//! ephemeral `dev.alloc()`.

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, InplaceOp2, Layout, Result,
    Shape, Tensor,
};

const HADAMARD_PTX: &str = include_str!("../../kernels/exl3/hadamard.ptx");
const RECONSTRUCT_PTX: &str = include_str!("../../kernels/exl3/reconstruct.ptx");
const EXL3_GEMV_PTX: &str = include_str!("../../kernels/exl3/exl3_gemv.ptx");

// Per-K GEMM comp_unit PTX. Each unit defines `exl3_gemm_kernel<K, ...>`
// across the full (c_fp32, codebook, shape) cross-product — 48 kernel
// entries per K = 336 mangled names total. We pick one of them at
// dispatch time based on the activation shape and codebook flag.
const EXL3_GEMM_PTX_K2: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_2.ptx");
const EXL3_GEMM_PTX_K3: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_3.ptx");
const EXL3_GEMM_PTX_K4: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_4.ptx");
const EXL3_GEMM_PTX_K5: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_5.ptx");
const EXL3_GEMM_PTX_K6: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_6.ptx");
const EXL3_GEMM_PTX_K7: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_7.ptx");
const EXL3_GEMM_PTX_K8: &str = include_str!("../../kernels/exl3/comp_units/exl3_comp_unit_8.ptx");

fn gemm_ptx_for_bpw(bpw: u32) -> Result<&'static str> {
    Ok(match bpw {
        2 => EXL3_GEMM_PTX_K2,
        3 => EXL3_GEMM_PTX_K3,
        4 => EXL3_GEMM_PTX_K4,
        5 => EXL3_GEMM_PTX_K5,
        6 => EXL3_GEMM_PTX_K6,
        7 => EXL3_GEMM_PTX_K7,
        8 => EXL3_GEMM_PTX_K8,
        _ => candle_core::bail!("exl3_gemm: bpw must be in 2..=8, got {}", bpw),
    })
}

/// Shape specialisation chosen at dispatch time. Values mirror the
/// `EXL3_GEMM_SHAPE_{1..4}` defines in `exl3_kernel_map.cuh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GemmShape {
    idx: u32,
    tilesize_m: u32,
    tilesize_k: u32,
    tilesize_n: u32,
    sh_stages: u32,
    frag_stages: u32,
    block_dim: u32,
}

impl GemmShape {
    const fn from_idx(idx: u32) -> Self {
        // EXL3_GEMM_SHAPE_1: 16, 16, 128, 6, 5      block_dim 256
        // EXL3_GEMM_SHAPE_2: 16, 32, 128, 4, 3      block_dim 512
        // EXL3_GEMM_SHAPE_3: 16, 32, 256, 4, 3      block_dim 512
        // EXL3_GEMM_SHAPE_4: 16, 16, 512, 4, 3      block_dim 256
        match idx {
            1 => Self {
                idx: 1,
                tilesize_m: 16,
                tilesize_k: 16,
                tilesize_n: 128,
                sh_stages: 6,
                frag_stages: 5,
                block_dim: 256,
            },
            2 => Self {
                idx: 2,
                tilesize_m: 16,
                tilesize_k: 32,
                tilesize_n: 128,
                sh_stages: 4,
                frag_stages: 3,
                block_dim: 512,
            },
            3 => Self {
                idx: 3,
                tilesize_m: 16,
                tilesize_k: 32,
                tilesize_n: 256,
                sh_stages: 4,
                frag_stages: 3,
                block_dim: 512,
            },
            4 => Self {
                idx: 4,
                tilesize_m: 16,
                tilesize_k: 16,
                tilesize_n: 512,
                sh_stages: 4,
                frag_stages: 3,
                block_dim: 256,
            },
            _ => Self {
                idx: 0,
                tilesize_m: 0,
                tilesize_k: 0,
                tilesize_n: 0,
                sh_stages: 0,
                frag_stages: 0,
                block_dim: 0,
            },
        }
    }
    const fn compatible(&self, size_k: usize, size_n: usize) -> bool {
        self.idx > 0
            && size_k % (self.tilesize_k as usize) == 0
            && size_n % (self.tilesize_n as usize) == 0
    }
}

/// Compute class — ports the C `select_gemm_shape` switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CcClass {
    Ampere, // sm_80, sm_86, sm_87
    Ada,    // sm_89, sm_90 (Hopper treated as ADA in the upstream switch)
}

impl CcClass {
    fn from_compute(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (8, 0..=7) => CcClass::Ampere,
            _ => CcClass::Ada, // ADA + Hopper + Blackwell all take this path
        }
    }
}

/// Pick the shape index based on (cc, m, k, n, bpw). Mirrors
/// `select_gemm_shape` in `quant/exl3_kernel_map.cu` (non-multi path).
fn select_gemm_shape(cc: CcClass, _size_m: usize, size_k: usize, size_n: usize, bpw: u32) -> u32 {
    let mod_256 = size_n.is_multiple_of(256);
    let mod_512 = size_n.is_multiple_of(512);
    match cc {
        CcClass::Ampere => {
            if mod_256 && bpw <= 4 {
                if size_n <= 2048 || size_k <= 2048 {
                    return 2;
                }
                return 3;
            }
            if mod_256 && size_n < 4096 {
                return if size_k > 8192 { 3 } else { 2 };
            }
            if mod_512 && (size_n * size_k) > (4096 * 4096) && bpw <= 6 {
                return 4;
            }
            if mod_256 {
                return 3;
            }
            2
        }
        CcClass::Ada => {
            if mod_256 && bpw <= 3 {
                if size_k <= 2048 {
                    return 2;
                }
                if size_n < 4096 && size_k <= 12288 {
                    return 2;
                }
                return 3;
            }
            if size_n <= 16384 {
                return 2;
            }
            if mod_512 && size_n >= 32768 {
                return 4;
            }
            if mod_256 {
                return 3;
            }
            2
        }
    }
}

/// Build the mangled PTX symbol for an exl3_gemm_kernel template instance.
///
/// Itanium mangling of the original signature:
/// ```text
/// _Z16exl3_gemm_kernelI Li<bpw> E Lb<c_fp32> E Li<cb> E
///     Li<TM> E Li<TK> E Li<TN> E Li<SH> E Li<FR> E E
///     vPK6__half PKt Pv iii Pi S2_ PS0_ S2_
/// ```
fn mangled_gemm_symbol(bpw: u32, c_fp32: bool, cb: u32, shape: GemmShape) -> String {
    format!(
        "_Z16exl3_gemm_kernelILi{}ELb{}ELi{}ELi{}ELi{}ELi{}ELi{}ELi{}EEvPK6__halfPKtPviiiPiS2_PS0_S2_",
        bpw,
        if c_fp32 { 1 } else { 0 },
        cb,
        shape.tilesize_m,
        shape.tilesize_k,
        shape.tilesize_n,
        shape.sh_stages,
        shape.frag_stages,
    )
}

/// Mangled PTX symbol for an `exl3_gemm_decode_kernel` template instance
/// (Phase 11.1). Same template args as `mangled_gemm_symbol`; the prefix
/// changes from `16exl3_gemm_kernel` to `23exl3_gemm_decode_kernel`
/// (Itanium length-prefixed identifiers — "exl3_gemm_decode_kernel" is
/// 23 characters).
fn mangled_gemm_decode_symbol(bpw: u32, c_fp32: bool, cb: u32, shape: GemmShape) -> String {
    format!(
        "_Z23exl3_gemm_decode_kernelILi{}ELb{}ELi{}ELi{}ELi{}ELi{}ELi{}ELi{}EEvPK6__halfPKtPviiiPiS2_PS0_S2_",
        bpw,
        if c_fp32 { 1 } else { 0 },
        cb,
        shape.tilesize_m,
        shape.tilesize_k,
        shape.tilesize_n,
        shape.sh_stages,
        shape.frag_stages,
    )
}

/// M-threshold below which `Exl3GemmInplaceOp` takes the split fast
/// path: standalone Hadamard launch + non-cooperative GEMM. At this
/// threshold the original kernel's M-tile loop runs at most once
/// anyway, so the trailing `grid.sync()` is dead weight and dropping
/// it costs nothing. Larger M values fall back to the cooperative
/// kernel where inter-iteration sync is load-bearing.
const EXL3_GEMM_DECODE_MAX_M: usize = 16;

/// EXL3 max dynamic shared memory request per kernel, mirroring
/// `SMEM_MAX` in `exl3_gemm_inner.cuh` (90 KiB).
const EXL3_GEMM_SMEM_MAX: i32 = 90 * 1024;

/// Dispatch the EXL3 GEMM kernel.
///
/// Computes
///   `out = had_r_128(had_r_128(a * suh) @ decode(trellis)) * svh`
/// — both Hadamard butterflies are **always** applied. The wrapper
/// kernel hardcodes `shmem_out_had=true`, so the output goes through
/// an `output_had_sh_gl` pass that reads `svh` unconditionally. Both
/// `suh` and `svh` are therefore required; pass all-ones vectors when
/// the sign-vector scaling step is mathematically a no-op (e.g. in
/// correctness tests).
///
/// # Arguments
/// * `a` — `[M, K]` fp16 row-major activations
/// * `trellis` — `[K/16, N/16, 16*bpw]` I16 packed weights
/// * `suh` — `[K]` fp16 input Hadamard sign vector (required)
/// * `svh` — `[N]` fp16 output Hadamard sign vector (required)
/// * `bpw` — bits-per-weight in `2..=8`
/// * `codebook` — `Default` / `Mcg` / `Mul1`
///
/// Returns `[M, N]` fp16 row-major output on the same device.
pub fn exl3_gemm(
    a: &Tensor,
    trellis: &Tensor,
    suh: Option<&Tensor>,
    svh: Option<&Tensor>,
    bpw: u32,
    codebook: Exl3Codebook,
) -> Result<Tensor> {
    if a.dtype() != DType::F16 {
        candle_core::bail!("exl3_gemm: A must be F16, got {:?}", a.dtype());
    }
    if a.dims().len() != 2 {
        candle_core::bail!("exl3_gemm: A must be 2D [M, K], got {:?}", a.dims());
    }
    if trellis.dtype() != DType::I16 {
        candle_core::bail!("exl3_gemm: trellis must be I16, got {:?}", trellis.dtype());
    }
    if trellis.dims().len() != 3 {
        candle_core::bail!("exl3_gemm: trellis must be 3D, got {:?}", trellis.dims());
    }
    let (m, k) = (a.dims()[0], a.dims()[1]);
    let (k_blocks, n_blocks, last) = (trellis.dims()[0], trellis.dims()[1], trellis.dims()[2]);
    if k_blocks * 16 != k {
        candle_core::bail!(
            "exl3_gemm: A.cols ({}) and trellis.dim0*16 ({}) disagree",
            k,
            k_blocks * 16
        );
    }
    let n = n_blocks * 16;
    if last != 16 * bpw as usize {
        candle_core::bail!(
            "exl3_gemm: trellis last dim ({}) must be 16*bpw ({})",
            last,
            16 * bpw
        );
    }
    if !k.is_multiple_of(16) || !n.is_multiple_of(128) {
        candle_core::bail!(
            "exl3_gemm: K ({}) must be %16 and N ({}) must be %128",
            k,
            n
        );
    }

    // The upstream kernel always reads `suh` — caller must provide.
    let suh_t = match suh {
        Some(s) => {
            if s.dtype() != DType::F16 {
                candle_core::bail!("exl3_gemm: suh must be F16");
            }
            let len: usize = s.dims().iter().product();
            if len != k {
                candle_core::bail!("exl3_gemm: suh len ({}) must equal K ({})", len, k);
            }
            s.clone()
        }
        None => candle_core::bail!(
            "exl3_gemm: suh is required (the kernel's `if (suh)` guard is \
             commented out — see exl3_gemm_kernel.cuh:14). Pass an \
             all-ones vector to mathematically disable the sign step."
        ),
    };
    let svh_t = match svh {
        Some(s) => {
            if s.dtype() != DType::F16 {
                candle_core::bail!("exl3_gemm: svh must be F16");
            }
            let len: usize = s.dims().iter().product();
            if len != n {
                candle_core::bail!("exl3_gemm: svh len ({}) must equal N ({})", len, n);
            }
            s.clone()
        }
        None => candle_core::bail!(
            "exl3_gemm: svh is required — the kernel hardcodes `shmem_out_had=true`, \
             so an output Hadamard pass reads svh unconditionally. Pass an all-ones \
             vector to mathematically disable the sign step."
        ),
    };

    // Pool-backed output + A_had scratch. The pool keeps device addresses
    // stable across forwards (Phase 8.2 — eliminates two of three per-call
    // `dev.alloc_zeros` allocations in the EXL3 hot path; the third was
    // killed in 8.1). The op writes through both — `output` via the
    // `&mut self_storage` receiver, `a_had` via the kernel's device-pointer
    // arg (immutable host ref, mutable device write — same pattern as
    // existing `silu_and_mul_separate_inplace`).
    let device = a.device();
    // PERF NOTE: this hot loop uses the legacy `reserve` (returning
    // `Result<Tensor>` directly) rather than `reserve_pooled` (returning
    // `Result<PooledTensor>` then `.into_tensor()` at the call site).
    // The two should be semantically identical, but the latter form
    // produces a measurable ~50% c=16 throughput regression here
    // (1685 → 600 tps on Llama-3.2-1B-EXL3, confirmed via same-session
    // stash/unstash). Attempts that did NOT restore perf:
    //   - `#[inline(always)]` on `reserve_pooled`, `reserve_kind`,
    //     `from_pool_unchecked`, `into_tensor`.
    //   - `#[repr(transparent)]` on `PooledTensor` so the wrapper has
    //     identical layout to `Tensor`.
    //   - bypassing the `reserve_kind` indirection via a direct-bodied
    //     `reserve_pooled_direct` variant.
    //   - `.map(|p| p.into_tensor())?` ordering vs `?.into_tensor()`.
    // What DID restore perf: returning `Result<Tensor>` from the pool
    // (either old `reserve` or a `reserve_via_pooled` that wraps then
    // unwraps the `PooledTensor` inside one function body, so the
    // function-return ABI is `Result<Tensor>`).
    // Suspected root cause: even with `#[repr(transparent)]`, rustc's
    // ABI treats `Result<PooledTensor>` as a distinct return type and
    // emits drop-handling unwind paths around the call site that
    // poison register allocation in the surrounding cooperative-GEMM
    // launch closure. Inlining is partial — the unwind tables stay
    // even when the body is inlined.
    let output =
        crate::engine::output_pool::OutputPool::global().reserve(&[m, n], DType::F16, device)?;
    let a_had =
        crate::engine::output_pool::OutputPool::global().reserve(&[m, k], DType::F16, device)?;

    let op = Exl3GemmInplaceOp {
        trellis: trellis.clone(),
        suh: suh_t,
        svh: svh_t,
        a_had,
        m,
        k,
        n,
        bpw,
        codebook,
    };
    output.inplace_op2(a, &op)?;
    Ok(output)
}

#[doc(hidden)]
/// Public helper for Exl3Linear::forward when the layer was loaded
/// without an explicit svh — we treat that as "no-op sign vector"
/// and allocate an ones-filled buffer on demand.
pub fn ones_f16_vec(len: usize, dev: &candle_core::Device) -> Result<Tensor> {
    Tensor::ones(len, DType::F16, dev)
}

/// Returns true if the M=1..=8 GEMV fast path is available for the
/// given bpw. Upstream ExLlamaV3 only instantiates the GEMV template
/// for K=4 (4-bit weights); other bpw values must fall through to the
/// cooperative GEMM kernel. See `quant/exl3_gemv.cu`.
pub const fn exl3_gemv_supports_bpw(bpw: u32) -> bool {
    bpw == 4
}

const EXL3_GEMV_TILESIZE_K: usize = 512;
const EXL3_GEMV_TILESIZE_N: usize = 32;

/// EXL3 GEMV fast path for M ≤ 8 and bpw=4.
///
/// Same `out = had(x*suh) @ decode(trellis); out = had(out) * svh`
/// math as `exl3_gemm`, but uses the simpler non-cooperative kernel
/// that doesn't need locks / A_had scratch. Only available for 4-bit
/// weights — call `exl3_gemv_supports_bpw(bpw)` to gate.
///
/// Requires `K % 512 == 0` and `N % 32 == 0`.
pub fn exl3_gemv(a: &Tensor, trellis: &Tensor, bpw: u32, codebook: Exl3Codebook) -> Result<Tensor> {
    if !exl3_gemv_supports_bpw(bpw) {
        candle_core::bail!("exl3_gemv: only bpw=4 is available upstream, got {}", bpw);
    }
    if a.dtype() != DType::F16 || trellis.dtype() != DType::I16 {
        candle_core::bail!("exl3_gemv: A must be F16, trellis must be I16");
    }
    if a.dims().len() != 2 {
        candle_core::bail!("exl3_gemv: A must be 2D [M, K]");
    }
    let (m, k) = (a.dims()[0], a.dims()[1]);
    let (_kb, n_blocks, last) = (trellis.dims()[0], trellis.dims()[1], trellis.dims()[2]);
    let n = n_blocks * 16;
    if m == 0 || m > 8 {
        candle_core::bail!("exl3_gemv: M must be in 1..=8, got {}", m);
    }
    if !k.is_multiple_of(EXL3_GEMV_TILESIZE_K) {
        candle_core::bail!(
            "exl3_gemv: K ({}) must be a multiple of {}",
            k,
            EXL3_GEMV_TILESIZE_K
        );
    }
    if !n.is_multiple_of(EXL3_GEMV_TILESIZE_N) {
        candle_core::bail!(
            "exl3_gemv: N ({}) must be a multiple of {}",
            n,
            EXL3_GEMV_TILESIZE_N
        );
    }
    if last != 16 * bpw as usize {
        candle_core::bail!(
            "exl3_gemv: trellis last dim ({}) must be 16*bpw ({})",
            last,
            16 * bpw
        );
    }
    let device = a.device();
    let output = crate::engine::output_pool::OutputPool::global()
        .reserve_pooled(&[m, n], DType::F16, device)?
        .into_tensor();
    let op = Exl3GemvInplaceOp {
        trellis: trellis.clone(),
        m,
        k,
        n,
        codebook,
    };
    output.inplace_op2(a, &op)?;
    Ok(output)
}

// ─── Type-Safe PooledTensor Wrappers (Phase TS.2) ───────────────────

/// Type-safe wrapper for [`exl3_gemm`]. `a` is pool-backed; trellis/suh/svh
/// are model weights (stable for engine lifetime, remain `&Tensor`).
pub fn exl3_gemm_pooled_typed(
    a: &crate::engine::output_pool::PooledTensor,
    trellis: &Tensor,
    suh: Option<&Tensor>,
    svh: Option<&Tensor>,
    bpw: u32,
    codebook: Exl3Codebook,
) -> Result<crate::engine::output_pool::PooledTensor> {
    let out = exl3_gemm(a.as_tensor(), trellis, suh, svh, bpw, codebook)?;
    // SAFETY: `exl3_gemm` reserves output from OutputPool::global() (line 349).
    Ok(unsafe { crate::engine::output_pool::PooledTensor::from_pool_unchecked(out) })
}

/// Type-safe wrapper for [`exl3_gemv`].
pub fn exl3_gemv_pooled_typed(
    a: &crate::engine::output_pool::PooledTensor,
    trellis: &Tensor,
    bpw: u32,
    codebook: Exl3Codebook,
) -> Result<crate::engine::output_pool::PooledTensor> {
    let out = exl3_gemv(a.as_tensor(), trellis, bpw, codebook)?;
    // SAFETY: `exl3_gemv` reserves output from OutputPool::global() (line 434).
    Ok(unsafe { crate::engine::output_pool::PooledTensor::from_pool_unchecked(out) })
}

struct Exl3GemvInplaceOp {
    trellis: Tensor,
    m: usize,
    k: usize,
    n: usize,
    codebook: Exl3Codebook,
}

impl InplaceOp2 for Exl3GemvInplaceOp {
    fn name(&self) -> &'static str {
        "exl3_gemv_inplace"
    }
    fn cpu_fwd(
        &self,
        _s1: &mut CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<()> {
        candle_core::bail!("exl3_gemv: CUDA only")
    }
    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        a_storage: &CudaStorage,
        _a_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let dev = &a_storage.device;
        // `_Z16exl3_gemv_kernelILi4ELb<c_fp32>ELi<cb>ELi1EEvPK6__halfPKtPviii`
        let kernel_name = format!(
            "_Z16exl3_gemv_kernelILi4ELb0ELi{}ELi1EEvPK6__halfPKtPviii",
            self.codebook.as_u32()
        );
        let func = dev.get_or_load_custom_func(
            crate::quantization::exl3_scratch::intern_kernel_name(kernel_name),
            "exl3_gemv",
            EXL3_GEMV_PTX,
        )?;

        let a = match &a_storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_gemv: A must be F16"),
        };
        let (tg, _) = self.trellis.storage_and_layout();
        let trellis = match &*tg {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::I16(s) => s,
                _ => candle_core::bail!("exl3_gemv: trellis must be I16"),
            },
            _ => candle_core::bail!("exl3_gemv: trellis must be on CUDA"),
        };

        let output = match &mut out_storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_gemv: output must be F16"),
        };

        // Grid + block per the kernel header comment:
        //   grid:  (1, size_n / TILESIZE_N, 1)
        //   block: (32, TILESIZE_N/16, TILESIZE_K/16) → 32 × 2 × 32 = 2048
        // but the implementation uses (1, N/TILESIZE_N) and block
        // (32, 1, TILESIZE_K / 16) = (32, 1, 32) = 1024 threads.
        let cfg = LaunchConfig {
            grid_dim: (1, (self.n / EXL3_GEMV_TILESIZE_N) as u32, 1),
            block_dim: (32, 1, (EXL3_GEMV_TILESIZE_K / 16) as u32),
            shared_mem_bytes: 0,
        };
        let size_m = self.m as i32;
        let size_k = self.k as i32;
        let size_n = self.n as i32;
        let mut builder = func.builder();
        builder.arg(a);
        builder.arg(trellis);
        builder.arg(output);
        builder.arg(&size_m);
        builder.arg(&size_k);
        builder.arg(&size_n);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemv launch: {e}")))?;

        Ok(())
    }
}

/// EXL3 GEMM op. Output is the InplaceOp2 receiver (pool-backed),
/// activations `A` are the second arg, and trellis/suh/svh/a_had are
/// captured in the struct.
struct Exl3GemmInplaceOp {
    trellis: Tensor,
    suh: Tensor, // required — see exl3_gemm() docs
    svh: Tensor, // required — kernel hardcodes shmem_out_had=true
    /// Pool-reserved `[M, K]` F16 scratch — Hadamard-transformed A. The
    /// kernel writes then reads it within a single launch; the device
    /// pointer is passed via `builder.arg(&CudaSlice<f16>)` (host-ref
    /// is immutable; device kernel writes through it freely).
    a_had: Tensor,
    m: usize,
    k: usize,
    n: usize,
    bpw: u32,
    codebook: Exl3Codebook,
}

impl InplaceOp2 for Exl3GemmInplaceOp {
    fn name(&self) -> &'static str {
        "exl3_gemm_inplace"
    }
    fn cpu_fwd(
        &self,
        _s1: &mut CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<()> {
        candle_core::bail!("exl3_gemm: CUDA only")
    }
    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        a_storage: &CudaStorage,
        _a_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys::{
            CUdevice_attribute, CUfunction_attribute_enum,
        };
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let dev = &a_storage.device;

        // Query device properties for shape selection + grid sizing.
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| candle_core::Error::Msg(format!("get cc-major: {e}")))?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| candle_core::Error::Msg(format!("get cc-minor: {e}")))?;
        let num_sms = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| candle_core::Error::Msg(format!("get num_sms: {e}")))?
            as u32;
        let cc = CcClass::from_compute(major, minor);

        // Pick a compatible shape: start with the upstream selection,
        // then fall back through {2, 3, 4, 1} if it isn't compatible.
        let mut shape =
            GemmShape::from_idx(select_gemm_shape(cc, self.m, self.k, self.n, self.bpw));
        if !shape.compatible(self.k, self.n) {
            for &fallback_idx in &[2u32, 3, 4, 1] {
                let s = GemmShape::from_idx(fallback_idx);
                if s.compatible(self.k, self.n) {
                    shape = s;
                    break;
                }
            }
        }
        if !shape.compatible(self.k, self.n) {
            candle_core::bail!(
                "exl3_gemm: no compatible shape for K={}, N={}",
                self.k,
                self.n
            );
        }

        // Bound num_sms so we don't launch CTAs with zero work.
        let max_slices = ((self.k / shape.tilesize_k as usize)
            * (self.n / shape.tilesize_n as usize))
            .max(1) as u32;
        let grid_sms = num_sms.min(max_slices).max(1);

        // Resolve the kernel. For M ≤ EXL3_GEMM_DECODE_MAX_M we use the
        // Phase-11 decode variant (no cooperative launch, capture-friendly,
        // split input-Hadamard into a separate prior launch). For larger
        // M (prefill batches) we keep the original cooperative kernel —
        // its inter-iteration `grid.sync()` is load-bearing there.
        let use_decode_fast_path = self.m <= EXL3_GEMM_DECODE_MAX_M;
        let kernel_name = if use_decode_fast_path {
            mangled_gemm_decode_symbol(
                self.bpw,
                false, /* c_fp32 = false → fp16 output */
                self.codebook.as_u32(),
                shape,
            )
        } else {
            mangled_gemm_symbol(self.bpw, false, self.codebook.as_u32(), shape)
        };
        let ptx = gemm_ptx_for_bpw(self.bpw)?;
        let func = dev.get_or_load_custom_func(
            crate::quantization::exl3_scratch::intern_kernel_name(kernel_name),
            if use_decode_fast_path {
                "exl3_gemm_decode"
            } else {
                "exl3_gemm"
            },
            ptx,
        )?;

        // Bump dynamic shared memory cap. cuFuncSetAttribute is idempotent
        // per-function; we set it on every dispatch for safety (low-cost).
        let _ = func.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            EXL3_GEMM_SMEM_MAX,
        );

        // ─── Buffers ──────────────────────────────────────────────────
        let a = match &a_storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_gemm: A must be F16"),
        };

        let (tg, _) = self.trellis.storage_and_layout();
        let trellis = match &*tg {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::I16(s) => s,
                _ => candle_core::bail!("exl3_gemm: trellis must be I16"),
            },
            _ => candle_core::bail!("exl3_gemm: trellis must be on CUDA"),
        };

        // Output [M, N] fp16 — receiver, pool-backed by the caller.
        let output = match &mut out_storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_gemm: output must be F16"),
        };

        // A_had scratch [M, K] fp16 — pool-backed by the caller. The
        // device pointer is shared via the immutable host ref; the
        // kernel writes/reads its memory within one launch.
        let a_had_storage_guard = self.a_had.storage_and_layout().0;
        let a_had = match &*a_had_storage_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F16(s) => s,
                _ => candle_core::bail!("exl3_gemm: a_had scratch must be F16"),
            },
            _ => candle_core::bail!("exl3_gemm: a_had scratch must be on CUDA"),
        };

        // Per-device locks workspace. Cached for the life of the process
        // — the barrier protocol is self-restoring across launches
        // (`barrier_release(reset=true)` zeroes locks, `group_barrier`
        // uses sense-reversal so the sense bit value is irrelevant
        // between launches). See `quantization::exl3_scratch` for the
        // ownership model, which mirrors ExLlamaV3's `DevCtx::get_locks`.
        let locks = crate::quantization::exl3_scratch::exl3_locks(dev)?;

        // Both `suh` and `svh` are mandatory — the kernel hardcodes
        // shmem_out_had=true and unconditionally dereferences both.
        let suh_storage_guard = self.suh.storage_and_layout().0;
        let suh_slice = match &*suh_storage_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F16(slice) => slice,
                _ => candle_core::bail!("exl3_gemm: suh must be F16"),
            },
            _ => candle_core::bail!("exl3_gemm: suh must be on CUDA"),
        };
        let svh_storage_guard = self.svh.storage_and_layout().0;
        let svh_slice = match &*svh_storage_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F16(slice) => slice,
                _ => candle_core::bail!("exl3_gemm: svh must be F16"),
            },
            _ => candle_core::bail!("exl3_gemm: svh must be on CUDA"),
        };
        let size_m = self.m as i32;
        let size_k = self.k as i32;
        let size_n = self.n as i32;

        let cfg = LaunchConfig {
            grid_dim: (grid_sms, 1, 1),
            block_dim: (shape.block_dim, 1, 1),
            shared_mem_bytes: EXL3_GEMM_SMEM_MAX as u32,
        };

        // Phase 11.1 fast path: pre-launch the standalone Hadamard kernel
        // into `a_had`, then dispatch the non-cooperative decode GEMM
        // which reads `a_had` as its A. Stream serialisation between
        // launches gives the same happens-before as the old in-kernel
        // `grid.sync()`. Both launches are capture-eligible.
        if use_decode_fast_path {
            let had_func = dev.get_or_load_custom_func(
                HadScale::Pre.fp16_kernel(),
                "exl3_hadamard",
                HADAMARD_PTX,
            )?;
            // had_hf_r_128_kernel grid layout: (rows, cols/128, 1), block (32, 1, 1).
            let had_cfg = LaunchConfig {
                grid_dim: (self.m as u32, (self.k as u32) / 128, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            // Match the in-kernel constant: r_scale = 1 / sqrt(128).
            let r_scale: f32 = 1.0_f32 / 11.313708498984761_f32;
            let mut hb = had_func.builder();
            hb.arg(a);
            hb.arg(a_had);
            hb.arg(suh_slice);
            hb.arg(&r_scale);
            unsafe { hb.launch(had_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("exl3_gemm_decode hadamard launch: {e}"))
            })?;

            // Decode GEMM reads `a_had` (Hadamard-transformed input) and
            // writes `output` directly. `suh`/A_had args are kept in the
            // ABI for signature parity with the cooperative kernel but
            // are ignored inside `exl3_gemm_decode_kernel`.
            let mut builder = func.builder();
            builder.arg(a_had); // A := a_had
            builder.arg(trellis);
            builder.arg(output);
            builder.arg(&size_m);
            builder.arg(&size_k);
            builder.arg(&size_n);
            builder.arg(&*locks);
            builder.arg(suh_slice); // unused inside the decode kernel
            builder.arg(a_had); // unused (A_had param) — kernel ignores
            builder.arg(svh_slice);
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm_decode launch: {e}")))?;
            return Ok(());
        }

        // Cooperative path (prefill, M > EXL3_GEMM_DECODE_MAX_M).
        let mut builder = func.builder();
        builder.arg(a);
        builder.arg(trellis);
        builder.arg(output);
        builder.arg(&size_m);
        builder.arg(&size_k);
        builder.arg(&size_n);
        builder.arg(&*locks);
        builder.arg(suh_slice);
        builder.arg(a_had);
        builder.arg(svh_slice);

        // Cooperative launch: the kernel uses `cg::this_grid().sync()`
        // for inter-CTA coordination of tile assignment.
        unsafe { builder.launch_cooperative(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm launch: {e}")))?;

        Ok(())
    }
}

/// Codebook variant used by the EXL3 trellis decoder.
///
/// - `Default` (cb=0): the canonical 3INST procedural codebook
/// - `Mcg` (cb=1): multiplied by `0xCBAC1FED`
/// - `Mul1` (cb=2): multiplied by `0x83DCD12D`
///
/// The variant is encoded in the trellis weights at quantization time and
/// is reported by the checkpoint's `mcg_multiplier` / `mul1_multiplier`
/// fields. See `quantization::exl3::EXL3_MCG_MULTIPLIER` for the constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3Codebook {
    Default,
    Mcg,
    Mul1,
}

impl Exl3Codebook {
    pub const fn from_flags(mcg: bool, mul1: bool) -> Self {
        if mcg {
            Self::Mcg
        } else if mul1 {
            Self::Mul1
        } else {
            Self::Default
        }
    }

    const fn as_u32(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::Mcg => 1,
            Self::Mul1 => 2,
        }
    }
}

/// Decode an EXL3-packed trellis tensor to a dense fp16 weight matrix.
///
/// This is the inverse of EXL3 quantization, used as a correctness anchor
/// against ExLlamaV3's reference Python implementation. It's also the
/// fallback path for non-aligned shapes that the GEMM dispatcher can't
/// handle.
///
/// # Arguments
/// * `trellis` — `[k/16, n/16, 16*bpw]` I16. Storage byte layout matches
///   the kernel's `uint16_t*` view (signed/unsigned share bits).
/// * `bpw` — bits-per-weight (2..=8).
/// * `codebook` — `Default` / `Mcg` / `Mul1`.
///
/// Returns `[k, n]` fp16 row-major dense weights on the same device.
pub fn exl3_reconstruct(trellis: &Tensor, bpw: u32, codebook: Exl3Codebook) -> Result<Tensor> {
    if trellis.dtype() != DType::I16 {
        candle_core::bail!(
            "exl3_reconstruct: trellis must be I16, got {:?}",
            trellis.dtype()
        );
    }
    let dims = trellis.dims();
    if dims.len() != 3 {
        candle_core::bail!("exl3_reconstruct: trellis must be 3D, got {:?}", dims);
    }
    let (k_blocks, n_blocks, last) = (dims[0], dims[1], dims[2]);
    if !(2..=8).contains(&bpw) {
        candle_core::bail!("exl3_reconstruct: bpw must be in 2..=8, got {}", bpw);
    }
    let expected_last = 16 * bpw as usize;
    if last != expected_last {
        candle_core::bail!(
            "exl3_reconstruct: trellis last dim ({}) must be 16*bpw ({})",
            last,
            expected_last
        );
    }

    let op = ReconstructOp {
        k: k_blocks * 16,
        n: n_blocks * 16,
        k_blocks,
        n_blocks,
        bpw,
        codebook,
    };
    trellis.apply_op1(op)
}

struct ReconstructOp {
    k: usize,
    n: usize,
    k_blocks: usize,
    n_blocks: usize,
    bpw: u32,
    codebook: Exl3Codebook,
}

impl CustomOp1 for ReconstructOp {
    fn name(&self) -> &'static str {
        "exl3_reconstruct"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("exl3_reconstruct: CUDA only")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, _l: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        // `_Z18reconstruct_kernelILi<K>ELi<cb>EEvP6__halfPKt`
        let kernel_name = format!(
            "_Z18reconstruct_kernelILi{}ELi{}EEvP6__halfPKt",
            self.bpw,
            self.codebook.as_u32()
        );
        let func = dev.get_or_load_custom_func(
            crate::quantization::exl3_scratch::intern_kernel_name(kernel_name),
            "exl3_reconstruct",
            RECONSTRUCT_PTX,
        )?;

        let trellis = match &storage.slice {
            CudaStorageSlice::I16(s) => s,
            _ => candle_core::bail!("exl3_reconstruct: trellis must be I16"),
        };

        let output = dev.alloc_zeros::<half::f16>(self.k * self.n)?;

        // Block 256, grid (n_blocks / 8, k_blocks). The kernel processes
        // an (8 n-blocks × 1 k-block) tile per CTA.
        if !self.n_blocks.is_multiple_of(8) {
            candle_core::bail!(
                "exl3_reconstruct: n_blocks ({}) must be a multiple of 8 (i.e. n must be a multiple of 128)",
                self.n_blocks
            );
        }
        let cfg = LaunchConfig {
            grid_dim: ((self.n_blocks / 8) as u32, self.k_blocks as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(trellis);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("exl3_reconstruct launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::F16(output),
            device: dev.clone(),
        };
        Ok((out_storage, Shape::from_dims(&[self.k, self.n])))
    }
}

/// Which side carries the Hadamard sign-vector scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadScale {
    /// `out = (x * scale) @ Had`
    Pre,
    /// `out = (x @ Had) * scale`
    Post,
    /// `out = x @ Had` (scale ignored — caller passes any tensor)
    None,
}

impl HadScale {
    /// Mangled PTX entry-point name for the fp16 hadamard kernel.
    /// Symbols come from `nvcc --ptx` over `hadamard.cu`; verified by
    /// `grep '.entry' hadamard.ptx`.
    const fn fp16_kernel(self) -> &'static str {
        match self {
            HadScale::Pre => "_Z19had_hf_r_128_kernelILb1ELb0EEvPK6__halfPS0_S2_f",
            HadScale::Post => "_Z19had_hf_r_128_kernelILb0ELb1EEvPK6__halfPS0_S2_f",
            HadScale::None => "_Z19had_hf_r_128_kernelILb0ELb0EEvPK6__halfPS0_S2_f",
        }
    }
}

/// Out-of-place 128-wise Hadamard transform over fp16 rows.
///
/// `input` must be 2D `[rows, cols]` with `cols % 128 == 0`. Returns
/// a freshly allocated `[rows, cols]` fp16 tensor. For capture-friendly
/// pooled output, prefer `had_r_128_fp16_inplace` once the pool path
/// is wired (Phase 4 — pending the OutputPool integration story).
///
/// Math: each contiguous 128-element block of a row is multiplied by
/// `had_128 / sqrt(128) * scale_factor`. If `mode == Pre`, the input is
/// first elementwise-multiplied by `scale[col / 128]`; if `Post`, the
/// output is elementwise-multiplied by `scale[col / 128]` afterwards.
pub fn had_r_128_fp16(input: &Tensor, scale: Option<&Tensor>, mode: HadScale) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        candle_core::bail!("had_r_128: input must be fp16, got {:?}", input.dtype());
    }
    let dims = input.dims();
    if dims.len() != 2 {
        candle_core::bail!("had_r_128: input must be 2D, got {:?}", dims);
    }
    let (rows, cols) = (dims[0], dims[1]);
    if !cols.is_multiple_of(128) {
        candle_core::bail!("had_r_128: cols ({}) must be a multiple of 128", cols);
    }
    if matches!(mode, HadScale::Pre | HadScale::Post) && scale.is_none() {
        candle_core::bail!("had_r_128: mode={:?} requires a scale tensor", mode);
    }
    if let Some(s) = scale {
        if s.dtype() != DType::F16 {
            candle_core::bail!("had_r_128: scale must be fp16, got {:?}", s.dtype());
        }
        let scale_len: usize = s.dims().iter().product();
        let expected = match mode {
            // pre  → scale spans the input cols (k-side Hadamard signs)
            // post → scale spans the output cols (n-side; same width here)
            HadScale::Pre | HadScale::Post => cols,
            HadScale::None => scale_len, // unused
        };
        if matches!(mode, HadScale::Pre | HadScale::Post) && scale_len != expected {
            candle_core::bail!(
                "had_r_128: scale len ({}) must equal cols ({})",
                scale_len,
                cols
            );
        }
    }

    let device = input.device();
    let output = crate::engine::output_pool::OutputPool::global()
        .reserve_pooled(&[rows, cols], DType::F16, device)?
        .into_tensor();
    let op = HadR128Fp16InplaceOp {
        scale: scale.cloned(),
        mode,
        rows: rows as i32,
        cols: cols as i32,
    };
    output.inplace_op2(input, &op)?;
    Ok(output)
}

struct HadR128Fp16InplaceOp {
    scale: Option<Tensor>,
    mode: HadScale,
    rows: i32,
    cols: i32,
}

impl InplaceOp2 for HadR128Fp16InplaceOp {
    fn name(&self) -> &'static str {
        "exl3_had_r_128_fp16_inplace"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<()> {
        candle_core::bail!("exl3_had_r_128_fp16: CUDA only")
    }

    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        in_storage: &CudaStorage,
        _in_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let dev = &in_storage.device;
        let kernel_name = self.mode.fp16_kernel();
        let func = dev.get_or_load_custom_func(kernel_name, "exl3_hadamard", HADAMARD_PTX)?;

        let input = match &in_storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_had_r_128: input must be F16"),
        };
        let output = match &mut out_storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_had_r_128: output must be F16"),
        };

        // Resolve `scale` to a device pointer. In scaled modes the
        // caller's tensor is forwarded. In `HadScale::None` mode the
        // <false,false> kernel template instance never dereferences
        // the pointer — we pass a cached per-device 1-element sentinel
        // (`exl3_had_null_scale`), eliminating what was a 2-byte
        // per-call `cudaMallocAsync`.
        let scale_storage_guard;
        let null_scale_sentinel;
        let scale_ref: &candle_core::cuda::cudarc::driver::CudaSlice<half::f16> =
            match (&self.scale, self.mode) {
                (Some(s), HadScale::Pre | HadScale::Post) => {
                    scale_storage_guard = s.storage_and_layout().0;
                    match &*scale_storage_guard {
                        Storage::Cuda(cs) => match &cs.slice {
                            CudaStorageSlice::F16(slice) => slice,
                            _ => candle_core::bail!("exl3_had_r_128: scale must be F16"),
                        },
                        _ => candle_core::bail!("exl3_had_r_128: scale must be on CUDA"),
                    }
                }
                _ => {
                    null_scale_sentinel =
                        crate::quantization::exl3_scratch::exl3_had_null_scale(dev)?;
                    &*null_scale_sentinel
                }
            };

        // 1/sqrt(128) — the kernel multiplies by this internally.
        let r_scale: f32 = 1.0_f32 / 11.313708498984761_f32; // = 0.088388347648f

        let grid = (self.rows as u32, (self.cols as u32) / 128, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(input);
        builder.arg(output);
        builder.arg(scale_ref);
        builder.arg(&r_scale);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("had_r_128 launch: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn had_scale_kernel_names_distinct() {
        assert_ne!(HadScale::Pre.fp16_kernel(), HadScale::Post.fp16_kernel());
        assert_ne!(HadScale::Pre.fp16_kernel(), HadScale::None.fp16_kernel());
        assert_ne!(HadScale::Post.fp16_kernel(), HadScale::None.fp16_kernel());
        // Sanity: every mangled name starts with the C++ Itanium prefix.
        for s in [HadScale::Pre, HadScale::Post, HadScale::None] {
            assert!(s.fp16_kernel().starts_with("_Z"));
        }
    }

    #[test]
    fn had_r_128_rejects_non_fp16_input() {
        let x = Tensor::zeros((1, 128), DType::F32, &Device::Cpu).unwrap();
        let err = had_r_128_fp16(&x, None, HadScale::None).err().unwrap();
        assert!(format!("{err}").contains("fp16"));
    }

    #[test]
    fn had_r_128_rejects_non_multiple_of_128_cols() {
        let x = Tensor::zeros((1, 64), DType::F16, &Device::Cpu).unwrap();
        let err = had_r_128_fp16(&x, None, HadScale::None).err().unwrap();
        assert!(format!("{err}").contains("multiple of 128"));
    }

    #[test]
    fn select_gemm_shape_ada_llama_3_1_8b_attn_proj() {
        // Llama 3.1 8B attention: q/k/v/o projections are (M, K=4096, N=4096),
        // bpw=3 on the 3.0bpw checkpoint.
        // ADA path with mod_256 && K<=3: size_k=4096 <= 2048? no; size_n<4096
        // && size_k<=12288? no; → shape 3.
        assert_eq!(select_gemm_shape(CcClass::Ada, 1, 4096, 4096, 3), 3);
    }

    #[test]
    fn select_gemm_shape_ada_llama_3_1_8b_mlp_up() {
        // MLP up/gate proj: (M, K=4096, N=14336) — mod_256 true, mod_512 false.
        // K=3 path: size_k=4096<=2048? no; size_n=14336<4096? no → shape 3.
        assert_eq!(select_gemm_shape(CcClass::Ada, 1, 4096, 14336, 3), 3);
    }

    #[test]
    fn select_gemm_shape_ada_llama_3_1_8b_mlp_down() {
        // MLP down proj: (M, K=14336, N=4096) — mod_256 true.
        // K=3 path: size_k=14336<=2048? no; size_n=4096<4096? no → shape 3.
        assert_eq!(select_gemm_shape(CcClass::Ada, 1, 14336, 4096, 3), 3);
    }

    #[test]
    fn mangled_gemm_symbol_matches_ptx_entry() {
        // Cross-check the symbol builder against a known entry from
        // comp_unit_3.ptx: K=3, c_fp32=true, cb=0, shape 2 (16,32,128,4,3).
        let s = mangled_gemm_symbol(3, true, 0, GemmShape::from_idx(2));
        assert_eq!(
            s,
            "_Z16exl3_gemm_kernelILi3ELb1ELi0ELi16ELi32ELi128ELi4ELi3EEvPK6__halfPKtPviiiPiS2_PS0_S2_"
        );
        // And shape 1 with c_fp32=false:
        let s2 = mangled_gemm_symbol(3, false, 0, GemmShape::from_idx(1));
        assert_eq!(
            s2,
            "_Z16exl3_gemm_kernelILi3ELb0ELi0ELi16ELi16ELi128ELi6ELi5EEvPK6__halfPKtPviiiPiS2_PS0_S2_"
        );
    }

    #[test]
    fn gemm_shape_compatibility() {
        let s2 = GemmShape::from_idx(2);
        assert!(s2.compatible(4096, 4096));
        assert!(s2.compatible(14336, 4096));
        assert!(!s2.compatible(48, 128)); // 48 % 32 != 0
        let s4 = GemmShape::from_idx(4);
        assert!(s4.compatible(4096, 4096));
        assert!(!s4.compatible(4096, 4096 - 256)); // N=3840 % 512 != 0
    }

    #[test]
    fn gemv_only_supports_bpw_4() {
        assert!(exl3_gemv_supports_bpw(4));
        for k in [2u32, 3, 5, 6, 7, 8] {
            assert!(!exl3_gemv_supports_bpw(k), "bpw {k} must be unsupported");
        }
    }

    #[test]
    fn gemv_rejects_non_4bit() {
        let a = Tensor::zeros((1, 512), DType::F16, &Device::Cpu).unwrap();
        let t = Tensor::zeros((32, 2, 48), DType::I16, &Device::Cpu).unwrap();
        let err = exl3_gemv(&a, &t, 3, Exl3Codebook::Default).err().unwrap();
        assert!(format!("{err}").contains("bpw=4"));
    }

    #[test]
    fn gemv_rejects_large_m() {
        let a = Tensor::zeros((16, 512), DType::F16, &Device::Cpu).unwrap();
        let t = Tensor::zeros((32, 2, 64), DType::I16, &Device::Cpu).unwrap();
        let err = exl3_gemv(&a, &t, 4, Exl3Codebook::Default).err().unwrap();
        assert!(format!("{err}").contains("M must be"));
    }

    #[test]
    fn gemm_rejects_wrong_dtypes() {
        let a = Tensor::zeros((1, 16), DType::F32, &Device::Cpu).unwrap();
        let t = Tensor::zeros((1, 1, 48), DType::I16, &Device::Cpu).unwrap();
        let suh = Tensor::zeros(16, DType::F16, &Device::Cpu).unwrap();
        let err = exl3_gemm(&a, &t, Some(&suh), None, 3, Exl3Codebook::Default)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("F16"));
    }

    #[test]
    fn gemm_requires_suh() {
        let a = Tensor::zeros((1, 4096), DType::F16, &Device::Cpu).unwrap();
        let t = Tensor::zeros((256, 256, 48), DType::I16, &Device::Cpu).unwrap();
        let err = exl3_gemm(&a, &t, None, None, 3, Exl3Codebook::Default)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("suh is required"));
    }

    #[test]
    fn reconstruct_rejects_wrong_trellis_dtype() {
        let t = Tensor::zeros((1, 1, 48), DType::F16, &Device::Cpu).unwrap();
        let err = exl3_reconstruct(&t, 3, Exl3Codebook::Default)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("I16"));
    }

    #[test]
    fn reconstruct_rejects_inconsistent_bpw() {
        // trellis last-dim = 16*bpw. last=48 → bpw=3 is fine; last=48 with bpw=4 is bad.
        let t = Tensor::zeros((1, 1, 48), DType::I16, &Device::Cpu).unwrap();
        let err = exl3_reconstruct(&t, 4, Exl3Codebook::Default)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("16*bpw"));
    }

    #[test]
    fn codebook_from_flags_priorities_mcg_over_mul1() {
        assert_eq!(Exl3Codebook::from_flags(true, true), Exl3Codebook::Mcg);
        assert_eq!(Exl3Codebook::from_flags(false, true), Exl3Codebook::Mul1);
        assert_eq!(
            Exl3Codebook::from_flags(false, false),
            Exl3Codebook::Default
        );
    }

    #[test]
    fn had_r_128_requires_scale_in_scaled_modes() {
        let x = Tensor::zeros((1, 128), DType::F16, &Device::Cpu).unwrap();
        let err = had_r_128_fp16(&x, None, HadScale::Pre).err().unwrap();
        assert!(format!("{err}").contains("requires a scale"));
    }

    #[cfg(feature = "gpu-test-small")]
    mod gpu_tests {
        use super::*;
        use candle_core::{Device, IndexOp};

        fn cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn reconstruct_zero_trellis_is_constant() {
            // Diagnostic: for zero trellis, decode_3inst<cb=0>(0) is a
            // specific constant v0; every element of `reconstruct` must
            // equal v0. If we get a non-constant result, the dq layout
            // shuffle is wrong — that's our bug, not the matmul kernel.
            let Some(dev) = cuda_device() else { return };
            let k = 32usize;
            let n = 128usize;
            let bpw = 4u32;
            let trellis =
                Tensor::zeros((k / 16, n / 16, 16 * bpw as usize), DType::I16, &dev).unwrap();
            let w = exl3_reconstruct(&trellis, bpw, Exl3Codebook::Default).unwrap();
            assert_eq!(w.dims(), &[k, n]);
            let w_f32 = w.to_dtype(DType::F32).unwrap();
            let min_v = w_f32
                .min(0)
                .unwrap()
                .min(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let max_v = w_f32
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mean = w_f32.mean_all().unwrap().to_scalar::<f32>().unwrap();
            eprintln!("RECONSTRUCT(0): min={min_v}, max={max_v}, mean={mean}");
            assert!(
                (max_v - min_v).abs() < 1e-3,
                "reconstruct(0) is not constant: min={min_v}, max={max_v}"
            );
        }

        #[test]
        fn exl3_gemm_matches_reconstruct_on_zero_trellis() {
            // Deterministic kernel-vs-reconstruct cross-check using
            // all-zero trellis. Both paths route through the same
            // codebook decoder; zero bytes give a constant decoded
            // value v0 per element, so reconstruct→[K,N] is uniformly
            // v0 and the matmul reduces to v0·sum(had(x)) per column.
            //
            // If this test PASSES but the random-trellis variant fails,
            // the bug is in tile-layout interpretation rather than the
            // decoder itself.
            let Some(dev) = cuda_device() else { return };

            let m = 2usize;
            let k = 512usize;
            let n = 128usize;
            let bpw = 4u32;

            let trellis =
                Tensor::zeros((k / 16, n / 16, 16 * bpw as usize), DType::I16, &dev).unwrap();
            let mut x_data = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.013).sin() * 0.05) as f32;
                x_data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&x_data, (m, k), &dev).unwrap();
            let ones_suh = Tensor::ones(k, DType::F16, &dev).unwrap();
            let ones_svh = Tensor::ones(n, DType::F16, &dev).unwrap();

            // Kernel does:
            //   xh = had_r_128(x * suh)
            //   y_pre = xh @ decode(trellis)
            //   y    = had_r_128(y_pre) * svh
            // With suh=svh=ones, that's:
            //   y = had(had(x) @ reconstruct(trellis))
            let xh = had_r_128_fp16(&x, Some(&ones_suh), HadScale::Pre).unwrap();
            let w_dense = exl3_reconstruct(&trellis, bpw, Exl3Codebook::Default).unwrap();
            let y_pre = xh.matmul(&w_dense).unwrap();
            let y_ref = had_r_128_fp16(&y_pre, Some(&ones_svh), HadScale::Post).unwrap();

            let y_act = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .unwrap();

            let diff = (&y_act - &y_ref).unwrap();
            let max_abs = diff
                .abs()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let ref_max = y_ref
                .abs()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            // Per-row sanity: every column of y_act must equal v0 *
            // sum(xh per row), where v0 = decode_3inst<0>(0) ≈ 0.768.
            let act_f32 = y_act.to_dtype(DType::F32).unwrap();
            let row0 = act_f32.i(0).unwrap();
            let vals: Vec<f32> = row0.to_vec1().unwrap();
            let col0_val = vals[0];
            let row0_uniform_max = vals.iter().cloned().fold(f32::MIN, f32::max);
            let row0_uniform_min = vals.iter().cloned().fold(f32::MAX, f32::min);
            // dump first 16 cols
            eprintln!("ACT row0 cols[0..16]: {:?}", &vals[0..16]);
            let nonzero_count = vals.iter().filter(|&&v| v.abs() > 1e-6).count();
            eprintln!("ACT row0 nonzero count: {}/128", nonzero_count);
            let xh_row0_sum = xh
                .i(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "ACT row0: col0={col0_val}, row_min={row0_uniform_min}, row_max={row0_uniform_max}; expected = 0.768 * sum(xh[0])= {}",
                0.768 * xh_row0_sum
            );
            eprintln!("ZERO-TRELLIS: ref_max={ref_max}, max_abs_diff={max_abs}");
            // Loose-but-meaningful: with deterministic all-zero
            // codebook decode, any layout/decode disagreement shows up
            // as a relative diff >> fp16 noise (~1e-3).
            assert!(
                max_abs < 1e-2 || (max_abs / ref_max.max(1e-6)) < 0.01,
                "zero-trellis kernel/reconstruct disagree: ref_max={ref_max}, diff={max_abs}"
            );
        }

        #[test]
        fn exl3_gemm_matches_hadamard_then_matmul() {
            // Cross-check: both `exl3_reconstruct` and the GEMM kernel
            // route through the same `dq_dispatch<bits, cb>` per-tile
            // decoder (see exl3_dq.cuh + exl3_gemm_inner.cuh:280 +
            // reconstruct.cu:37). With suh=ones the kernel computes
            //   y_kernel = had_r_128(x) @ decode(trellis)
            // We build the same on the Rust side:
            //   y_ref = had_r_128_fp16(x, ones, Pre) @ reconstruct(trellis)
            //
            // K=4 (bpw=4), shape_idx=2 compatible: M=2, K=512, N=128.
            let Some(dev) = cuda_device() else { return };

            let m = 2usize;
            let k = 512usize;
            let n = 128usize;
            let bpw = 4u32;

            // Deterministic pseudo-random trellis.
            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut t_data: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0x1234_5678;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                t_data.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&t_data, (k / 16, n / 16, 16 * bpw as usize), &dev).unwrap();

            // Small random x (avoid fp16 saturation).
            let mut x_data = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.013).sin() * 0.05) as f32;
                x_data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&x_data, (m, k), &dev).unwrap();
            let ones_suh = Tensor::ones(k, DType::F16, &dev).unwrap();
            let ones_svh = Tensor::ones(n, DType::F16, &dev).unwrap();

            // Reference math (matches the kernel pipeline):
            //   xh   = had_r_128(x * suh)
            //   ypre = xh @ reconstruct(trellis)
            //   y    = had_r_128(ypre) * svh
            let xh = had_r_128_fp16(&x, Some(&ones_suh), HadScale::Pre).expect("had ok");
            let w_dense =
                exl3_reconstruct(&trellis, bpw, Exl3Codebook::Default).expect("reconstruct ok");
            assert_eq!(w_dense.dims(), &[k, n]);
            let y_pre = xh.matmul(&w_dense).expect("ref matmul ok");
            let y_ref =
                had_r_128_fp16(&y_pre, Some(&ones_svh), HadScale::Post).expect("had post ok");

            let y_act = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("gemm ok");
            assert_eq!(y_act.dims(), &[m, n]);

            let diff = (&y_act - &y_ref).unwrap();
            let abs = diff.abs().unwrap().to_dtype(DType::F32).unwrap();
            let max_abs = abs
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let ref_abs_max = y_ref
                .abs()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            // Random-trellis stress test: outputs span large magnitudes
            // from accumulation over 512 fp16 mac ops + an additional
            // 128-point Hadamard butterfly. Tolerance is relative —
            // diff must be at most 10% of the reference norm.
            let rel = max_abs / ref_abs_max.max(1e-6);
            assert!(
                rel < 0.10,
                "exl3_gemm diverges: max_abs={max_abs}, ref_max={ref_abs_max}, rel={rel}"
            );
        }

        #[test]
        fn exl3_gemm_cooperative_path_matches_reference_when_m_above_threshold() {
            // Phase 11.1: exercises the M > EXL3_GEMM_DECODE_MAX_M branch
            // (cooperative kernel). Same reference pipeline as
            // `exl3_gemm_matches_hadamard_then_matmul` but with M=32 so we
            // skip the new fast path and validate that the unchanged
            // cooperative kernel still matches the reference. Catches
            // regressions if the dispatch gate or symbol resolution for
            // the legacy path silently breaks.
            let Some(dev) = cuda_device() else { return };

            let m = 32usize; // > EXL3_GEMM_DECODE_MAX_M (16)
            let k = 512usize;
            let n = 128usize;
            let bpw = 4u32;

            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut t_data: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0x9E37_79B1;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                t_data.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&t_data, (k / 16, n / 16, 16 * bpw as usize), &dev).unwrap();
            let mut x_data = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.013).sin() * 0.05) as f32;
                x_data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&x_data, (m, k), &dev).unwrap();
            let ones_suh = Tensor::ones(k, DType::F16, &dev).unwrap();
            let ones_svh = Tensor::ones(n, DType::F16, &dev).unwrap();

            let xh = had_r_128_fp16(&x, Some(&ones_suh), HadScale::Pre).expect("had ok");
            let w_dense =
                exl3_reconstruct(&trellis, bpw, Exl3Codebook::Default).expect("reconstruct ok");
            let y_pre = xh.matmul(&w_dense).expect("ref matmul ok");
            let y_ref =
                had_r_128_fp16(&y_pre, Some(&ones_svh), HadScale::Post).expect("had post ok");

            let y_act = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("cooperative gemm ok");
            let diff = (&y_act - &y_ref).unwrap();
            let abs = diff.abs().unwrap().to_dtype(DType::F32).unwrap();
            let max_abs = abs
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let ref_abs_max = y_ref
                .abs()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .max(0)
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let rel = max_abs / ref_abs_max.max(1e-6);
            assert!(
                rel < 0.10,
                "exl3_gemm cooperative path diverges: max_abs={max_abs}, ref_max={ref_abs_max}, rel={rel}"
            );
        }

        #[test]
        fn exl3_gemv_launches_on_synthetic_inputs() {
            // bpw=4 path: M=1, K=4096, N=4096 — Llama-style attention proj
            // dimensions. We expect the dedicated GEMV kernel to launch
            // and return a finite [1, 4096] fp16 output.
            let Some(dev) = cuda_device() else { return };

            let m = 1usize;
            let k = 4096usize;
            let n = 4096usize;
            let bpw = 4u32;

            let mut x_data = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.0017).sin() * 0.1).clamp(-1.0, 1.0);
                x_data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&x_data, (m, k), &dev).unwrap();
            let trellis =
                Tensor::zeros((k / 16, n / 16, 16 * bpw as usize), DType::I16, &dev).unwrap();
            let y = exl3_gemv(&x, &trellis, bpw, Exl3Codebook::Default).expect("gemv launch ok");
            assert_eq!(y.dims(), &[m, n]);
            let s = y
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(s.is_finite(), "exl3_gemv produced non-finite sum: {s}");
        }

        #[test]
        fn exl3_gemm_launches_on_synthetic_inputs() {
            // Smoke test: 4-bit trellis, Llama-3.1-8B attention proj shape.
            // We don't have real EXL3 weights here, so we use all-zero
            // trellis + ones suh/svh. The kernel must launch and return
            // the right shape (and not produce NaN/Inf).
            let Some(dev) = cuda_device() else { return };

            let m = 1usize;
            let k = 4096usize;
            let n = 4096usize;
            let bpw = 4u32;

            // x = small random fp16
            let mut x_data = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.0017).sin() * 0.1).clamp(-1.0, 1.0);
                x_data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&x_data, (m, k), &dev).unwrap();

            // trellis = zeros (synthetic), shape [k/16, n/16, 16*bpw] I16.
            let trellis =
                Tensor::zeros((k / 16, n / 16, 16 * bpw as usize), DType::I16, &dev).unwrap();
            // suh, svh = ones fp16
            let suh = Tensor::ones(k, DType::F16, &dev).unwrap();
            let svh = Tensor::ones(n, DType::F16, &dev).unwrap();

            let y = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("exl3_gemm launch ok");

            assert_eq!(y.dims(), &[m, n]);
            assert_eq!(y.dtype(), DType::F16);

            // Output should be finite (no NaN/Inf). Check sum is finite.
            let s = y
                .to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(s.is_finite(), "exl3_gemm produced non-finite sum: {s}");
        }

        #[test]
        fn had_r_128_fp16_no_scale_launches_and_produces_orthogonal_norm() {
            // The 128-point Hadamard matrix is orthogonal up to a factor
            // of √128, and the kernel divides by √128 internally.
            // Therefore ‖had_r_128(x)‖² ≈ ‖x‖² for any x. We use that
            // as a self-consistency check that doesn't require a Python
            // reference.
            let Some(dev) = cuda_device() else { return };

            let rows = 4usize;
            let cols = 256usize; // two Hadamard groups
            let n = rows * cols;
            let mut data = Vec::with_capacity(n);
            for i in 0..n {
                // Simple deterministic pseudo-random fp16 inputs.
                let v = ((i as f32 * 0.1234).sin() * 0.5).clamp(-1.0, 1.0);
                data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&data, (rows, cols), &dev).unwrap();
            let y = had_r_128_fp16(&x, None, HadScale::None).expect("launch ok");

            assert_eq!(y.dims(), &[rows, cols]);
            assert_eq!(y.dtype(), DType::F16);

            let x_norm = x
                .to_dtype(DType::F32)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let y_norm = y
                .to_dtype(DType::F32)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            // 1.5 % slack for fp16 + fast-math.
            let ratio = y_norm / x_norm;
            assert!(
                (ratio - 1.0).abs() < 0.015,
                "Hadamard norm-preservation broken: ‖y‖²/‖x‖² = {ratio}"
            );
        }

        /// Phase 11.2 diagnostic: drive `exl3_gemm` through the decode
        /// fast path inside an explicit `cuStreamBeginCapture_v2` /
        /// `cuStreamEndCapture` window and report whether the captured
        /// graph instantiates cleanly. The full-engine warmup at
        /// `engine/standard.rs` fails with `CUDA_ERROR_INVALID_VALUE`
        /// when capture is enabled for EXL3-Llama, but in that context
        /// it's hard to tell whether the EXL3 op itself is the offender
        /// or some downstream layer (attention / MLP / lm_head) corrupts
        /// the stream first. This test isolates the EXL3 path so a
        /// failure here unambiguously points at one of:
        ///   - kernel-internal allocation (locks / null-scale-buf cache
        ///     cold miss inside capture),
        ///   - pool-growth `cuMemAllocAsync` inside the captured stream,
        ///   - `cuFuncSetAttribute` interaction with capture state,
        ///   - the two-launch Hadamard+GEMM sequence itself.
        ///
        /// The test pre-warms every per-device EXL3 cache + the pool
        /// before BeginCapture. If `cuStreamEndCapture` returns SUCCESS
        /// here, EXL3 is capture-clean and the production failure is
        /// elsewhere in the forward.
        // Capture-window tests share the GPU's default cuMemPool, so
        // running them in parallel (cargo test default) deadlocks the
        // first BeginCapture against another test's in-flight kernels.
        // Ignored by default; run via
        //   cargo test --features cuda-default,gpu-test-small -- --ignored --test-threads=1 capture
        #[test]
        #[ignore]
        fn exl3_gemm_decode_path_inside_cuda_capture_window() {
            use candle_core::cuda::cudarc::driver::sys::{
                cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUresult,
                CUstreamCaptureMode,
            };

            // Production engine uses a NON-default stream so that
            // `cuStreamBeginCapture_v2` works. The default null stream
            // returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` and would
            // make this test useless. Match production by constructing
            // the device via `Device::new_cuda_with_stream`.
            let dev_t = match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return,
            };
            let cuda_dev = match &dev_t {
                Device::Cuda(c) => c.clone(),
                _ => unreachable!(),
            };
            // Production engine disables event tracking via `create_cuda_device`
            // in main.rs; without this every `CudaSlice` carries a
            // per-slice event chain that creates cross-stream dependencies
            // when touched inside capture (CUDA_ERROR_STREAM_CAPTURE_ISOLATION).
            // Mirror that here.
            // SAFETY: this test issues all work on a single stream — exactly
            // the constraint `disable_event_tracking` requires.
            unsafe {
                cuda_dev.disable_event_tracking();
            }

            // Llama-3.2-1B attention-proj shape, bpw=3 (production target).
            let m = 1usize;
            let k = 2048usize;
            let n = 2048usize;
            let bpw = 3u32;

            // Synthetic inputs.
            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut t_data: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0xDEAD_BEEF;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                t_data.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&t_data, (k / 16, n / 16, 16 * bpw as usize), &dev_t).unwrap();
            let mut x_data = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.0017).sin() * 0.05) as f32;
                x_data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&x_data, (m, k), &dev_t).unwrap();
            let ones_suh = Tensor::ones(k, DType::F16, &dev_t).unwrap();
            let ones_svh = Tensor::ones(n, DType::F16, &dev_t).unwrap();

            // ─── Pre-warm everything that lazy-initialises ──────────────
            // Per-device EXL3 scratch caches.
            let _ = crate::quantization::exl3_scratch::exl3_locks(&cuda_dev).expect("locks warm");
            let _ = crate::quantization::exl3_scratch::exl3_had_null_scale(&cuda_dev)
                .expect("null-scale warm");

            // Pool growth: run the SAME gemm twice eagerly so pool grows
            // to the steady-state slot count; reset cursors so the
            // captured forward starts from slot 0 of a fully-grown pool.
            let _y0 = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("warmup gemm 1");
            let _y1 = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("warmup gemm 2");
            crate::engine::output_pool::OutputPool::global().reset_cursors();

            // Sync so no warmup work is in-flight when we start capture.
            unsafe {
                let r = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
                assert_eq!(
                    r,
                    candle_core::cuda::cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "pre-capture cuCtxSynchronize failed"
                );
            }

            // ─── Capture window ──────────────────────────────────────
            let stream_ref = cuda_dev.cuda_stream();
            let stream = stream_ref.cu_stream();

            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(
                begin,
                CUresult::CUDA_SUCCESS,
                "cuStreamBeginCapture_v2 failed: {begin:?}"
            );

            // Run the actual decode path inside the capture window.
            let captured = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            );

            // Always end capture (even on error) to release the stream.
            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            if !graph.is_null() {
                unsafe {
                    cuGraphDestroy(graph);
                }
            }

            // ─── Assertions ──────────────────────────────────────────
            assert!(
                captured.is_ok(),
                "exl3_gemm errored inside capture: {:?}",
                captured.err()
            );
            assert_eq!(
                end,
                CUresult::CUDA_SUCCESS,
                "cuStreamEndCapture failed with {end:?} — EXL3 decode path \
                 is NOT capture-clean: a kernel did a non-captureable \
                 operation (cuMemAllocAsync inside the stream, host pull, \
                 sync, etc.). This is the smoking gun the full-engine \
                 warmup was hitting."
            );
            let _ = captured;
        }

        /// Phase 11.2 diagnostic: same as the M=1 capture test but at
        /// M=16, the boundary of `EXL3_GEMM_DECODE_MAX_M`. The engine
        /// warmup uses batch_size=16 (largest of the default capture
        /// sizes after EXL3 clipping), which means M=16 hits the
        /// decode fast path. The production INVALID_VALUE failure
        /// starts at (or after) this batch — covers the boundary case
        /// explicitly.
        #[test]
        #[ignore]
        fn exl3_gemm_decode_path_at_m_16_inside_cuda_capture_window() {
            use candle_core::cuda::cudarc::driver::sys::{
                cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUresult,
                CUstreamCaptureMode,
            };

            let dev_t = match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return,
            };
            let cuda_dev = match &dev_t {
                Device::Cuda(c) => c.clone(),
                _ => unreachable!(),
            };
            unsafe {
                cuda_dev.disable_event_tracking();
            }

            let m = 16usize; // boundary case for the fast path gate
            let k = 2048usize;
            let n = 2048usize;
            let bpw = 3u32;
            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut td: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0xBEEF_1234;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                td.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&td, (k / 16, n / 16, 16 * bpw as usize), &dev_t).unwrap();
            let mut xd = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                xd.push(half::f16::from_f32(
                    ((i as f32 * 0.0017).sin() * 0.05) as f32,
                ));
            }
            let x = Tensor::from_slice(&xd, (m, k), &dev_t).unwrap();
            let suh = Tensor::ones(k, DType::F16, &dev_t).unwrap();
            let svh = Tensor::ones(n, DType::F16, &dev_t).unwrap();

            let _ = crate::quantization::exl3_scratch::exl3_locks(&cuda_dev).unwrap();
            let _ = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("warmup gemm M=16");
            let _ = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("warmup gemm M=16 #2");
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            unsafe {
                let _ = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            let stream = cuda_dev.cuda_stream().cu_stream();
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(
                begin,
                CUresult::CUDA_SUCCESS,
                "BeginCapture M=16: {begin:?}"
            );

            let captured = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            );

            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            if !graph.is_null() {
                unsafe {
                    cuGraphDestroy(graph);
                }
            }

            assert!(
                captured.is_ok(),
                "exl3_gemm M=16 errored inside capture: {:?}",
                captured.err()
            );
            assert_eq!(
                end,
                CUresult::CUDA_SUCCESS,
                "cuStreamEndCapture failed at M=16 with {end:?} — \
                 the fast-path boundary case is the actual capture blocker"
            );
        }

        /// Phase 11.2 diagnostic: reproduce the engine warmup pattern.
        /// Engine does: capture for bs=16 (success), then eager for
        /// bs=8 (FAILS with INVALID_VALUE). This test mirrors that
        /// EXACT cross-shape transition: capture at M=16 then eager
        /// at M=8 (different shape — different pool slots, different
        /// kernel-tile selection).
        #[test]
        #[ignore]
        fn eager_at_m_8_after_capture_at_m_16_reproduces_engine_pattern() {
            use candle_core::cuda::cudarc::driver::sys::{
                cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUresult,
                CUstreamCaptureMode,
            };

            let dev_t = match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return,
            };
            let cuda_dev = match &dev_t {
                Device::Cuda(c) => c.clone(),
                _ => unreachable!(),
            };
            unsafe {
                cuda_dev.disable_event_tracking();
            }

            let k = 2048usize;
            let n = 2048usize;
            let bpw = 3u32;
            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut td: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0xACE0_FACE;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                td.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&td, (k / 16, n / 16, 16 * bpw as usize), &dev_t).unwrap();
            let suh = Tensor::ones(k, DType::F16, &dev_t).unwrap();
            let svh = Tensor::ones(n, DType::F16, &dev_t).unwrap();

            // Build input tensors for both M=16 and M=8.
            let mut xd16 = Vec::with_capacity(16 * k);
            for i in 0..(16 * k) {
                xd16.push(half::f16::from_f32(
                    ((i as f32 * 0.011).sin() * 0.05) as f32,
                ));
            }
            let x16 = Tensor::from_slice(&xd16, (16, k), &dev_t).unwrap();
            let mut xd8 = Vec::with_capacity(8 * k);
            for i in 0..(8 * k) {
                xd8.push(half::f16::from_f32(
                    ((i as f32 * 0.013).sin() * 0.05) as f32,
                ));
            }
            let x8 = Tensor::from_slice(&xd8, (8, k), &dev_t).unwrap();

            // Mimic engine: JIT warmup for bs=16 (eager) then bs=8 (eager).
            // This pre-grows pool for BOTH shapes.
            let _ = exl3_gemm(
                &x16,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("eager warm M=16");
            let _ = exl3_gemm(
                &x8,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("eager warm M=8");
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            unsafe {
                let _ = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            // Now: capture at M=16 (mimics engine's Phase 2 for bs=16).
            let stream = cuda_dev.cuda_stream().cu_stream();
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS);
            let cap = exl3_gemm(
                &x16,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("captured gemm M=16");
            drop(cap);
            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            if !graph.is_null() {
                unsafe {
                    cuGraphDestroy(graph);
                }
            }
            assert_eq!(end, CUresult::CUDA_SUCCESS, "EndCapture M=16: {end:?}");

            // Now: eager at M=8 (mimics engine's Phase 1 for bs=8).
            // This is the exact transition the engine warmup performs
            // and fails on with INVALID_VALUE.
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            let post = exl3_gemm(
                &x8,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            );
            assert!(
                post.is_ok(),
                "eager exl3_gemm M=8 AFTER capture cycle M=16 failed: {:?} — \
                 ROOT CAUSE FOUND: cross-shape eager-after-capture is the \
                 specific engine failure mode",
                post.err()
            );
        }

        /// Phase 11.2 diagnostic: verify that a complete capture cycle
        /// (BeginCapture → forward → EndCapture) does NOT corrupt the
        /// CUDA context for subsequent eager forwards. The engine
        /// warmup symptom was exactly this: capture for bs=N succeeded
        /// silently, then the next eager forward for bs=N-1 errored
        /// with INVALID_VALUE. If reproducible outside the engine,
        /// this test will reproduce it.
        #[test]
        #[ignore]
        fn eager_exl3_gemm_after_capture_cycle_still_works() {
            use candle_core::cuda::cudarc::driver::sys::{
                cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUresult,
                CUstreamCaptureMode,
            };

            let dev_t = match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return,
            };
            let cuda_dev = match &dev_t {
                Device::Cuda(c) => c.clone(),
                _ => unreachable!(),
            };
            unsafe {
                cuda_dev.disable_event_tracking();
            }

            let m = 1usize;
            let k = 2048usize;
            let n = 2048usize;
            let bpw = 3u32;
            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut td: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0xC0FF_EE;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                td.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&td, (k / 16, n / 16, 16 * bpw as usize), &dev_t).unwrap();
            let mut xd = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                xd.push(half::f16::from_f32(
                    ((i as f32 * 0.013).sin() * 0.05) as f32,
                ));
            }
            let x = Tensor::from_slice(&xd, (m, k), &dev_t).unwrap();
            let suh = Tensor::ones(k, DType::F16, &dev_t).unwrap();
            let svh = Tensor::ones(n, DType::F16, &dev_t).unwrap();

            // Pre-warm
            let _ = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("pre-warm");
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            unsafe {
                let _ = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            // Capture cycle
            let stream = cuda_dev.cuda_stream().cu_stream();
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS, "BeginCapture: {begin:?}");
            let cap_y = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            )
            .expect("captured gemm");
            // Drop captured output before EndCapture so its destructor's
            // potential cuMemFreeAsync doesn't land inside capture.
            drop(cap_y);
            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            if !graph.is_null() {
                unsafe {
                    cuGraphDestroy(graph);
                }
            }
            assert_eq!(end, CUresult::CUDA_SUCCESS, "EndCapture: {end:?}");

            // Smoking gun: after a clean capture cycle, can we still do
            // eager forwards? If this fails, capture leaves the stream
            // in a state hostile to subsequent eager kernels — same
            // symptom as the engine warmup.
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            let post_y = exl3_gemm(
                &x,
                &trellis,
                Some(&suh),
                Some(&svh),
                bpw,
                Exl3Codebook::Default,
            );
            assert!(
                post_y.is_ok(),
                "eager exl3_gemm AFTER capture cycle failed: {:?} — \
                 this is the engine warmup symptom",
                post_y.err()
            );
        }

        /// Phase 11.2 diagnostic: run the standalone Hadamard launch
        /// (`had_r_128_fp16`, which is exactly Launch 1 of the EXL3
        /// decode fast path) inside a capture window. Same setup as the
        /// gemm test but only the Hadamard step — narrows down further
        /// if the gemm test were to ever start failing.
        #[test]
        #[ignore]
        fn had_r_128_fp16_inside_cuda_capture_window() {
            use candle_core::cuda::cudarc::driver::sys::{
                cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUresult,
                CUstreamCaptureMode,
            };

            let dev_t = match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return,
            };
            let cuda_dev = match &dev_t {
                Device::Cuda(c) => c.clone(),
                _ => unreachable!(),
            };
            unsafe {
                cuda_dev.disable_event_tracking();
            }

            // Warm pool: one eager run + reset cursors.
            let m = 1usize;
            let k = 2048usize;
            let mut xd = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                xd.push(half::f16::from_f32(
                    ((i as f32 * 0.013).sin() * 0.05) as f32,
                ));
            }
            let x = Tensor::from_slice(&xd, (m, k), &dev_t).unwrap();
            let suh = Tensor::ones(k, DType::F16, &dev_t).unwrap();
            let _ = had_r_128_fp16(&x, Some(&suh), HadScale::Pre).expect("warmup had");
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            unsafe {
                let _ = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            let stream_ref = cuda_dev.cuda_stream();
            let stream = stream_ref.cu_stream();
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS);

            let captured = had_r_128_fp16(&x, Some(&suh), HadScale::Pre);

            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            if !graph.is_null() {
                unsafe {
                    cuGraphDestroy(graph);
                }
            }

            assert!(
                captured.is_ok(),
                "had_r_128_fp16 errored inside capture: {:?}",
                captured.err()
            );
            assert_eq!(
                end,
                CUresult::CUDA_SUCCESS,
                "cuStreamEndCapture after had_r_128_fp16 failed: {end:?}"
            );
        }

        /// Phase CR.10: capture + INSTANTIATE + REPLAY exl3_gemm and
        /// compare replay output to eager. Existing capture-window
        /// tests verify capture SUCCESS but never run the captured
        /// graph — replay correctness is the production-blocking
        /// regression (forward 1 logits differ between eager and capture
        /// replay).
        ///
        /// If this test fails, EXL3 GEMM under cuGraph replay produces
        /// different output than eager invocation — the root cause of
        /// the full-server wrong-tokens bug.
        #[test]
        #[ignore]
        fn exl3_gemm_capture_replay_matches_eager() {
            use candle_core::cuda::cudarc::driver::sys::{
                cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
                cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult,
                CUstreamCaptureMode,
            };

            let dev_t = match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return,
            };
            let cuda_dev = match &dev_t {
                Device::Cuda(c) => c.clone(),
                _ => unreachable!(),
            };
            unsafe {
                cuda_dev.disable_event_tracking();
            }

            // Llama-3.2-1B q_proj shape: m=1 (batch=1 decode), k=2048, n=2048.
            let m = 1usize;
            let k = 2048usize;
            let n = 2048usize;
            let bpw = 3u32;
            let trellis_elems = (k / 16) * (n / 16) * 16 * bpw as usize;
            let mut td: Vec<i16> = Vec::with_capacity(trellis_elems);
            let mut seed: u32 = 0xDEAD_BEEF;
            for _ in 0..trellis_elems {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                td.push((seed >> 16) as i16);
            }
            let trellis =
                Tensor::from_slice(&td, (k / 16, n / 16, 16 * bpw as usize), &dev_t).unwrap();
            let mut xd = Vec::with_capacity(m * k);
            for i in 0..(m * k) {
                let v = ((i as f32 * 0.0017).sin() * 0.05) as f32;
                xd.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&xd, (m, k), &dev_t).unwrap();
            let ones_suh = Tensor::ones(k, DType::F16, &dev_t).unwrap();
            let ones_svh = Tensor::ones(n, DType::F16, &dev_t).unwrap();

            // Pre-warm caches.
            let _ = crate::quantization::exl3_scratch::exl3_locks(&cuda_dev).unwrap();
            let _ = crate::quantization::exl3_scratch::exl3_had_null_scale(&cuda_dev).unwrap();
            // Warmup gemm calls (pool growth + kernel JIT).
            let _ = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .unwrap();
            let _ = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .unwrap();
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            unsafe {
                candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            // Eager baseline.
            let eager_out = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .unwrap();
            unsafe {
                candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }
            let eager_bytes: Vec<u16> = {
                let f: Vec<half::f16> = eager_out.flatten_all().unwrap().to_vec1().unwrap();
                f.iter().map(|v| v.to_bits()).collect()
            };
            drop(eager_out);

            // Capture warmup.
            crate::engine::output_pool::OutputPool::global().reset_cursors();
            let _ = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .unwrap();
            unsafe {
                candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            crate::engine::output_pool::OutputPool::global().reset_cursors();
            let stream = cuda_dev.cuda_stream().cu_stream();
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS);

            let captured_out = exl3_gemm(
                &x,
                &trellis,
                Some(&ones_suh),
                Some(&ones_svh),
                bpw,
                Exl3Codebook::Default,
            )
            .unwrap();

            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            assert_eq!(end, CUresult::CUDA_SUCCESS);

            let mut exec: CUgraphExec = std::ptr::null_mut();
            let inst = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
            assert_eq!(inst, CUresult::CUDA_SUCCESS);

            let launch = unsafe { cuGraphLaunch(exec, stream) };
            assert_eq!(launch, CUresult::CUDA_SUCCESS);
            unsafe {
                candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            }

            let replay_bytes: Vec<u16> = {
                let f: Vec<half::f16> = captured_out.flatten_all().unwrap().to_vec1().unwrap();
                f.iter().map(|v| v.to_bits()).collect()
            };

            unsafe {
                cuGraphExecDestroy(exec);
                cuGraphDestroy(graph);
            }

            if eager_bytes != replay_bytes {
                let mismatches: Vec<(usize, u16, u16)> = eager_bytes
                    .iter()
                    .zip(replay_bytes.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .take(8)
                    .map(|(i, (a, b))| (i, *a, *b))
                    .collect();
                let n_diff = eager_bytes
                    .iter()
                    .zip(replay_bytes.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                panic!(
                    "exl3_gemm diverges under capture-replay: {} elems, {} mismatches ({:.1}%), first 8: {:?}",
                    eager_bytes.len(),
                    n_diff,
                    100.0 * n_diff as f64 / eager_bytes.len() as f64,
                    mismatches
                );
            }
        }
    }
}
