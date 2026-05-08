//! Stage 15.C/D — Marlin tile-MMA dispatcher.
//!
//! Wires the `marlin_tile_mma_int4_bf16` kernel (in
//! `kernels/marlin_tile_mma.cu`) into a Rust-callable function.
//!
//! v1 supports M ≥ 1, N multiple of 64 (tile_n_size), K multiple of 16
//! (tile_k_size). BF16 activations, INT4 AWQ weights laid out per
//! [`super::awq_marlin::awq_to_marlin_tile_repack_cpu`] (full multi-tile
//! output, k_tiles × n_tiles × 128 u32), per-channel BF16 scales. See
//! `docs/perf/marlin-tile-mma-step-15c-design.md` for the full kernel
//! design and the upcoming tensor-core path.
//!
//! The current kernel body is a software dot-product scaffold (proves
//! the build + dispatch pipeline). Stage 15.D-body replaces it with a
//! tensor-core mma.m16n8k16 path; the Rust dispatcher does not change.

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape,
    Storage, Tensor,
};

const MARLIN_TILE_MMA_PTX: &str = include_str!("../../kernels/marlin_tile_mma.ptx");

/// CustomOp wrapping the tile-MMA dispatch. The activation `A` is the
/// Op's primary tensor (so we get its CudaStorage via the trait); the
/// repacked weight `B` and per-channel scales `S` are held as struct
/// fields and resolved through `storage_and_layout()` inside `cuda_fwd`.
struct MarlinTileMmaV1Op {
    b_tile: Tensor,
    scales: Tensor,
    m: i32,
    n: i32,
    k: i32,
}

impl CustomOp1 for MarlinTileMmaV1Op {
    fn name(&self) -> &'static str {
        "marlin_tile_mma_int4_bf16"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("marlin_tile_mma requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let func = dev.get_or_load_custom_func(
            "marlin_tile_mma_int4_bf16",
            "marlin_tile_mma",
            MARLIN_TILE_MMA_PTX,
        )?;

        let a = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("marlin_tile_mma: A must be BF16"),
        };

        let (b_guard, _) = self.b_tile.storage_and_layout();
        let b = match &*b_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("marlin_tile_mma: B must be U32"),
            },
            _ => candle_core::bail!("marlin_tile_mma: B must be on CUDA"),
        };

        let (s_guard, _) = self.scales.storage_and_layout();
        let scales = match &*s_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("marlin_tile_mma: S must be BF16"),
            },
            _ => candle_core::bail!("marlin_tile_mma: S must be on CUDA"),
        };

        // v1: M ≥ 1, N multiple of 64, K multiple of 16. Output [M, N] BF16.
        let m = self.m as usize;
        let n = self.n as usize;
        let elem_count = m * n;
        let output = dev.alloc_zeros::<half::bf16>(elem_count)?;

        // Block 256 threads; grid covers ceildiv(M*N, 256) blocks.
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (elem_count.div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(a);
        builder.arg(b);
        builder.arg(scales);
        builder.arg(&output);
        builder.arg(&self.m);
        builder.arg(&self.n);
        builder.arg(&self.k);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("marlin_tile_mma launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((out_storage, Shape::from_dims(&[m, n])))
    }
}

/// Dispatch the v1 Marlin tile-MMA kernel.
///
/// # Arguments
/// * `a` — `[M, K]` BF16 activations on CUDA. `M ≥ 1`, `K` multiple of 16.
/// * `b_tile` — `[k_tiles * n_tiles * 128]` u32, the full output of
///   `awq_to_marlin_tile_repack_cpu(K, N)` where `k_tiles = K / 16` and
///   `n_tiles = N / 64`. The producer's `[k_tiles, n_tiles*128]` shape is
///   accepted as either 1-D flat or 2-D — both are flattened internally.
/// * `scales` — `[N]` BF16 per-channel scale (group_size = K).
/// * `n` — output column count, must be a multiple of 64.
///
/// Returns `[M, N]` BF16 row-major output on the same device.
pub fn dispatch_marlin_tile_mma_v1(
    a: &Tensor,
    b_tile: &Tensor,
    scales: &Tensor,
    n: usize,
) -> Result<Tensor> {
    if a.dims().len() != 2 || a.dtype() != DType::BF16 {
        candle_core::bail!(
            "marlin_tile_mma v1: A must be 2D BF16, got {:?} {:?}",
            a.dims(),
            a.dtype()
        );
    }
    let m = a.dims()[0];
    let k = a.dims()[1];
    if m == 0 {
        candle_core::bail!("marlin_tile_mma v1: M must be ≥ 1");
    }
    if !k.is_multiple_of(16) {
        candle_core::bail!("marlin_tile_mma v1: K ({}) must be a multiple of 16", k);
    }
    if !n.is_multiple_of(64) {
        candle_core::bail!("marlin_tile_mma v1: N ({}) must be a multiple of 64", n);
    }
    let k_tiles = k / 16;
    let n_tiles = n / 64;
    let expected_b_len = k_tiles * n_tiles * 128;
    let actual_b_len: usize = b_tile.dims().iter().product();
    if actual_b_len != expected_b_len || b_tile.dtype() != DType::U32 {
        candle_core::bail!(
            "marlin_tile_mma v1: B must contain {} U32 elements, got {:?} {:?}",
            expected_b_len,
            b_tile.dims(),
            b_tile.dtype()
        );
    }
    if scales.dims() != [n] || scales.dtype() != DType::BF16 {
        candle_core::bail!(
            "marlin_tile_mma v1: S must be [{}] BF16, got {:?} {:?}",
            n,
            scales.dims(),
            scales.dtype()
        );
    }

    let op = MarlinTileMmaV1Op {
        b_tile: b_tile.clone(),
        scales: scales.clone(),
        m: m as i32,
        n: n as i32,
        k: k as i32,
    };
    a.apply_op1(op)
}

#[cfg(all(test, feature = "gpu-test-small"))]
mod gpu_tests {
    use super::*;
    use crate::quantization::awq_marlin::awq_to_marlin_tile_repack_cpu;
    use candle_core::Device;

    /// Numeric-correctness lock-down for arbitrary K (multiple of 16).
    ///
    /// Build a deterministic AWQ qweight + per-channel scales + activations,
    /// dispatch the v1 kernel, compare against a CPU reference that does the
    /// same dequant + multiply + matmul. BF16 reduction tolerance scales
    /// with K (each summand is up to ~16 in magnitude × bf16 mantissa noise).
    fn run_lock_down(m: usize, n: usize, k: usize, tolerance: f32) {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        assert!(k.is_multiple_of(16), "K must be multiple of 16");
        assert!(n.is_multiple_of(64), "N must be multiple of 64");
        assert!(m >= 1);

        // 1. Activations A[M, K] BF16: deterministic small floats in [-1, 1].
        let a_vals: Vec<half::bf16> = (0..m * k)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.137) % 2.0) - 1.0))
            .collect();
        let a = Tensor::from_vec(a_vals.clone(), (m, k), &cuda).unwrap();

        // 2. AWQ qweight: [K, N/8] u32. Pack a deterministic logical nibble.
        let nibble = |kk: usize, nn: usize| -> u32 { ((kk * 113 + nn * 29 + 7) as u32) % 16 };
        const AWQ_PACK_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
        let mut awq = vec![0u32; k * (n / 8)];
        for kk in 0..k {
            for nn in 0..n {
                awq[kk * (n / 8) + nn / 8] |= nibble(kk, nn) << AWQ_PACK_SHIFTS[nn % 8];
            }
        }

        // 3. Repack via Stage 15.B and pass the FULL multi-tile output to
        // the kernel (no manual quarter extraction — the kernel reads all
        // 4 warps' worth per tile).
        let awq_t = Tensor::from_vec(awq, (k, n / 8), &Device::Cpu).unwrap();
        let tile_full = awq_to_marlin_tile_repack_cpu(&awq_t, k, n).unwrap();
        let b_tile = tile_full.flatten_all().unwrap().to_device(&cuda).unwrap();

        // 4. Per-channel scales [N] BF16.
        let scales_vec: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.05 + (i as f32) * 0.0025))
            .collect();
        let scales = Tensor::from_vec(scales_vec.clone(), n, &cuda).unwrap();

        // 5. Dispatch.
        let c = dispatch_marlin_tile_mma_v1(&a, &b_tile, &scales, n).unwrap();
        let c_host: Vec<half::bf16> = c.flatten_all().unwrap().to_vec1().unwrap();

        // 6. CPU reference.
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let w = nibble(kk, col) as f32 * scales_vec[col].to_f32();
                    let av = a_vals[row * k + kk].to_f32();
                    acc += av * w;
                }
                let got = c_host[row * n + col].to_f32();
                let diff = (got - acc).abs();
                assert!(
                    diff < tolerance,
                    "M={m} N={n} K={k} mismatch at (row={row}, col={col}): cpu={acc:.6} gpu={got:.6} diff={diff:.6}"
                );
            }
        }
    }

    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k16() {
        run_lock_down(16, 64, 16, 5e-2);
    }

    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k32() {
        run_lock_down(16, 64, 32, 1e-1);
    }

    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k64() {
        run_lock_down(16, 64, 64, 2e-1);
    }

    /// M=1 — the canonical c=1 decode case.
    #[test]
    fn test_marlin_tile_mma_v1_n64_m1_k64() {
        run_lock_down(1, 64, 64, 2e-1);
    }

    /// M=4 — typical multi-step decode.
    #[test]
    fn test_marlin_tile_mma_v1_n64_m4_k64() {
        run_lock_down(4, 64, 64, 2e-1);
    }

    /// Multi-block M.
    #[test]
    fn test_marlin_tile_mma_v1_n64_m32_k32() {
        run_lock_down(32, 64, 32, 1e-1);
    }

    /// N=128 (2 n-tiles) — exercises the n_tile axis of the producer.
    #[test]
    fn test_marlin_tile_mma_v1_n128_m16_k64() {
        run_lock_down(16, 128, 64, 2e-1);
    }

    /// N=256 with M and K above 16. Tolerance 3e-1 to absorb bf16 mantissa
    /// noise on the larger absolute sums (output magnitudes ≈ 30+ at this
    /// shape; 0.4 % relative bf16 error → ~0.12 abs).
    #[test]
    fn test_marlin_tile_mma_v1_n256_m32_k32() {
        run_lock_down(32, 256, 32, 3e-1);
    }
}
