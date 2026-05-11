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
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, InplaceOp1, Layout, Result,
    Shape, Tensor,
};

const HADAMARD_PTX: &str = include_str!("../../kernels/exl3/hadamard.ptx");
const RECONSTRUCT_PTX: &str = include_str!("../../kernels/exl3/reconstruct.ptx");

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

/// EXL3 max dynamic shared memory request per kernel, mirroring
/// `SMEM_MAX` in `exl3_gemm_inner.cuh` (90 KiB).
const EXL3_GEMM_SMEM_MAX: i32 = 90 * 1024;

/// Per-device locks workspace used by `exl3_gemm` for cooperative tile
/// scheduling. See `quant/exl3_devctx.cu::get_locks`:
/// `(MAX_TILES_C + MAX_BARRIERS * 2) * sizeof(int)`.
const EXL3_LOCKS_ELEMS: usize = (1024 * 1024) + (1024 * 2);

/// Dispatch the EXL3 GEMM kernel.
///
/// `out = had(a * suh) @ decode(trellis); out = had(out) * svh`
/// all fused into the cooperative kernel. The Hadamard scratch buffer
/// `a_had` is provided by the dispatcher; the kernel writes the
/// transformed activation into it before the matmul phase.
///
/// # Arguments
/// * `a` — `[M, K]` fp16 row-major activations
/// * `trellis` — `[K/16, N/16, 16*bpw]` I16 packed weights
/// * `suh` — optional `[K]` fp16 input Hadamard sign vector
/// * `svh` — optional `[N]` fp16 output Hadamard sign vector
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

    // Validate (and clone for capture inside the Op) the optional scales.
    let suh_t = match suh {
        Some(s) => {
            if s.dtype() != DType::F16 {
                candle_core::bail!("exl3_gemm: suh must be F16");
            }
            let len: usize = s.dims().iter().product();
            if len != k {
                candle_core::bail!("exl3_gemm: suh len ({}) must equal K ({})", len, k);
            }
            Some(s.clone())
        }
        None => None,
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
            Some(s.clone())
        }
        None => None,
    };

    let op = Exl3GemmOp {
        trellis: trellis.clone(),
        suh: suh_t,
        svh: svh_t,
        m,
        k,
        n,
        bpw,
        codebook,
    };
    a.apply_op1(op)
}

struct Exl3GemmOp {
    trellis: Tensor,
    suh: Option<Tensor>,
    svh: Option<Tensor>,
    m: usize,
    k: usize,
    n: usize,
    bpw: u32,
    codebook: Exl3Codebook,
}

impl CustomOp1 for Exl3GemmOp {
    fn name(&self) -> &'static str {
        "exl3_gemm"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("exl3_gemm: CUDA only")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, _l: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::sys::{
            CUdevice_attribute, CUfunction_attribute_enum,
        };
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let dev = &storage.device;

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

        // Resolve the kernel.
        let kernel_name = mangled_gemm_symbol(
            self.bpw,
            false, /* c_fp32 = false → fp16 output */
            self.codebook.as_u32(),
            shape,
        );
        let ptx = gemm_ptx_for_bpw(self.bpw)?;
        let func =
            dev.get_or_load_custom_func(Box::leak(kernel_name.into_boxed_str()), "exl3_gemm", ptx)?;

        // Bump dynamic shared memory cap. cuFuncSetAttribute is idempotent
        // per-function; we set it on every dispatch for safety (low-cost).
        let _ = func.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            EXL3_GEMM_SMEM_MAX,
        );

        // ─── Buffers ──────────────────────────────────────────────────
        let a = match &storage.slice {
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

        // Output [M, N] fp16. Pool integration follows in Phase 4b.6.
        let output = dev
            .alloc_zeros::<half::f16>(self.m * self.n)
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm output alloc: {e}")))?;

        // A_had scratch [M, K] fp16.
        let a_had = dev
            .alloc_zeros::<half::f16>(self.m * self.k)
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm A_had alloc: {e}")))?;

        // Per-device locks workspace, zero-initialised. Re-allocated
        // every call until a per-device cache lands (Phase 4b.6
        // alongside the OutputPool wiring).
        let locks = dev
            .alloc_zeros::<i32>(EXL3_LOCKS_ELEMS)
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm locks alloc: {e}")))?;

        // Optional Hadamard scale vectors. Use a 1-element placeholder
        // when not provided — the kernel branches on `nullptr`-equivalent
        // by checking the pointer, but in our case the template flag
        // `c_fp32` is unrelated and the kernel always reads `suh/svh`;
        // however, the runtime path only dereferences them when the
        // corresponding non-null path was selected at compile time, so
        // an unused dummy buffer is safe.
        let suh_storage_guard;
        let svh_storage_guard;
        let null_buf = dev
            .alloc_zeros::<half::f16>(1)
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm null buf alloc: {e}")))?;
        let suh_ref: &candle_core::cuda::cudarc::driver::CudaSlice<half::f16> = match &self.suh {
            Some(s) => {
                suh_storage_guard = s.storage_and_layout().0;
                match &*suh_storage_guard {
                    Storage::Cuda(cs) => match &cs.slice {
                        CudaStorageSlice::F16(slice) => slice,
                        _ => candle_core::bail!("exl3_gemm: suh must be F16"),
                    },
                    _ => candle_core::bail!("exl3_gemm: suh must be on CUDA"),
                }
            }
            None => &null_buf,
        };
        let svh_ref: &candle_core::cuda::cudarc::driver::CudaSlice<half::f16> = match &self.svh {
            Some(s) => {
                svh_storage_guard = s.storage_and_layout().0;
                match &*svh_storage_guard {
                    Storage::Cuda(cs) => match &cs.slice {
                        CudaStorageSlice::F16(slice) => slice,
                        _ => candle_core::bail!("exl3_gemm: svh must be F16"),
                    },
                    _ => candle_core::bail!("exl3_gemm: svh must be on CUDA"),
                }
            }
            None => &null_buf,
        };

        let size_m = self.m as i32;
        let size_k = self.k as i32;
        let size_n = self.n as i32;

        let cfg = LaunchConfig {
            grid_dim: (grid_sms, 1, 1),
            block_dim: (shape.block_dim, 1, 1),
            shared_mem_bytes: EXL3_GEMM_SMEM_MAX as u32,
        };

        let mut builder = func.builder();
        builder.arg(a);
        builder.arg(trellis);
        builder.arg(&output);
        builder.arg(&size_m);
        builder.arg(&size_k);
        builder.arg(&size_n);
        builder.arg(&locks);
        builder.arg(suh_ref);
        builder.arg(&a_had);
        builder.arg(svh_ref);

        // Cooperative launch: the kernel uses `cg::this_grid().sync()`
        // for inter-CTA coordination of tile assignment.
        unsafe { builder.launch_cooperative(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("exl3_gemm launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::F16(output),
            device: dev.clone(),
        };
        Ok((out_storage, Shape::from_dims(&[self.m, self.n])))
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
            // `get_or_load_custom_func` wants &'static str; leak the
            // small kernel-name string (one allocation per (bpw, cb)
            // pair per device, amortised across the full session).
            Box::leak(kernel_name.into_boxed_str()),
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

    let op = HadR128Fp16Op {
        scale: scale.cloned(),
        mode,
        rows: rows as i32,
        cols: cols as i32,
    };
    input.apply_op1(op)
}

struct HadR128Fp16Op {
    scale: Option<Tensor>,
    mode: HadScale,
    rows: i32,
    cols: i32,
}

impl CustomOp1 for HadR128Fp16Op {
    fn name(&self) -> &'static str {
        "exl3_had_r_128_fp16"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("exl3_had_r_128_fp16: CUDA only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _l: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let dev = &storage.device;
        let kernel_name = self.mode.fp16_kernel();
        let func = dev.get_or_load_custom_func(kernel_name, "exl3_hadamard", HADAMARD_PTX)?;

        let input = match &storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_had_r_128: input must be F16"),
        };

        // Output is freshly allocated (zeroed; the kernel overwrites
        // every element, so the zero init is wasted but cheap). When
        // we wire OutputPool in Phase 4, replace with `reserve()`.
        let elem_count = (self.rows as usize) * (self.cols as usize);
        let output = dev.alloc_zeros::<half::f16>(elem_count)?;

        // Build a "null" device pointer for the no-scale variant by
        // allocating a 1-element placeholder. Cleaner than nullable
        // pointers in the cudarc builder.
        let null_scale_buf;
        let scale_storage_guard;
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
                    null_scale_buf = dev.alloc_zeros::<half::f16>(1)?;
                    // Build a leaked reference: cudarc takes a slice ref;
                    // null_scale_buf lives for the rest of the function.
                    // Workaround: rebind via shadowing below.
                    // The kernel ignores `scale` in the <false,false> case.
                    #[allow(clippy::needless_borrow)]
                    {
                        &null_scale_buf
                    }
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
        builder.arg(&output);
        builder.arg(scale_ref);
        builder.arg(&r_scale);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("had_r_128 launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::F16(output),
            device: dev.clone(),
        };
        Ok((
            out_storage,
            Shape::from_dims(&[self.rows as usize, self.cols as usize]),
        ))
    }
}

// `InplaceOp1` variant used when the caller already owns the output
// buffer (e.g. pool-reserved tensor). Not used in Phase 3 yet — leave
// the type declared so the wire-up in Phase 4 is a one-line addition.
#[allow(dead_code)]
struct HadR128Fp16InplaceOp<'a> {
    input: &'a Tensor,
    scale: Option<&'a Tensor>,
    mode: HadScale,
    rows: i32,
    cols: i32,
}

#[allow(dead_code)]
impl<'a> InplaceOp1 for HadR128Fp16InplaceOp<'a> {
    fn name(&self) -> &'static str {
        "exl3_had_r_128_fp16_inplace"
    }
    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!("exl3_had_r_128_fp16_inplace: CUDA only")
    }
    fn cuda_fwd(&self, _s: &mut CudaStorage, _l: &Layout) -> Result<()> {
        // Will be implemented in Phase 4 alongside OutputPool wiring.
        candle_core::bail!("exl3_had_r_128_fp16_inplace: pending Phase 4")
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
    fn gemm_rejects_wrong_dtypes() {
        let a = Tensor::zeros((1, 16), DType::F32, &Device::Cpu).unwrap();
        let t = Tensor::zeros((1, 1, 48), DType::I16, &Device::Cpu).unwrap();
        let err = exl3_gemm(&a, &t, None, None, 3, Exl3Codebook::Default)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("F16"));
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
        use candle_core::Device;

        fn cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
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
    }
}
