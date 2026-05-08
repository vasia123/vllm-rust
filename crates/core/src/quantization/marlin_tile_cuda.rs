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
    qzeros: Tensor,
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
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

        let (z_guard, _) = self.qzeros.storage_and_layout();
        let qzeros = match &*z_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("marlin_tile_mma: qzeros must be U32"),
            },
            _ => candle_core::bail!("marlin_tile_mma: qzeros must be on CUDA"),
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
        builder.arg(qzeros);
        builder.arg(&output);
        builder.arg(&self.m);
        builder.arg(&self.n);
        builder.arg(&self.k);
        builder.arg(&self.group_size);

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
/// * `scales` — `[num_groups, N]` BF16, per-group scale.
/// * `qzeros` — `[num_groups, N/8]` u32 — AWQ-packed (8 zp per word).
/// * `n` — output column count, must be a multiple of 64.
/// * `group_size` — must divide K and be a multiple of 16.
///
/// Returns `[M, N]` BF16 row-major output on the same device.
pub fn dispatch_marlin_tile_mma_v1(
    a: &Tensor,
    b_tile: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    n: usize,
    group_size: usize,
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
    if group_size == 0 || !group_size.is_multiple_of(16) || !k.is_multiple_of(group_size) {
        candle_core::bail!(
            "marlin_tile_mma v1: group_size ({}) must be a multiple of 16 that divides K ({})",
            group_size,
            k
        );
    }
    let num_groups = k / group_size;
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
    if scales.dims() != [num_groups, n] || scales.dtype() != DType::BF16 {
        candle_core::bail!(
            "marlin_tile_mma v1: S must be [{}, {}] BF16, got {:?} {:?}",
            num_groups,
            n,
            scales.dims(),
            scales.dtype()
        );
    }
    if qzeros.dims() != [num_groups, n / 8] || qzeros.dtype() != DType::U32 {
        candle_core::bail!(
            "marlin_tile_mma v1: qzeros must be [{}, {}] U32, got {:?} {:?}",
            num_groups,
            n / 8,
            qzeros.dims(),
            qzeros.dtype()
        );
    }

    let op = MarlinTileMmaV1Op {
        b_tile: b_tile.clone(),
        scales: scales.clone(),
        qzeros: qzeros.clone(),
        m: m as i32,
        n: n as i32,
        k: k as i32,
        group_size: group_size as i32,
    };
    a.apply_op1(op)
}

#[cfg(all(test, feature = "gpu-test-small"))]
mod gpu_tests {
    use super::*;
    use crate::quantization::awq_marlin::awq_to_marlin_tile_repack_cpu;
    use candle_core::Device;

    /// Stage 15.D-body.1 — CustomOp1 wrapping the standalone LOP3
    /// dequant kernel. Input `q` is U32 [count]; output is BF16
    /// [count, 4] with nibbles {#0, #4, #1, #5} per u32.
    struct DequantInt4Bf16Lo4Op {
        count: i32,
    }

    impl CustomOp1 for DequantInt4Bf16Lo4Op {
        fn name(&self) -> &'static str {
            "marlin_test_dequant_int4_bf16_lo4"
        }
        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("dequant_int4_bf16_lo4: CUDA only")
        }
        fn cuda_fwd(&self, storage: &CudaStorage, _l: &Layout) -> Result<(CudaStorage, Shape)> {
            use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
            let dev = &storage.device;
            let q = match &storage.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("dequant_int4_bf16_lo4: q must be U32"),
            };
            let func = dev.get_or_load_custom_func(
                "marlin_test_dequant_int4_bf16_lo4",
                "marlin_tile_mma",
                MARLIN_TILE_MMA_PTX,
            )?;
            let count = self.count as usize;
            let output = dev.alloc_zeros::<half::bf16>(count * 4)?;
            const BLOCK: u32 = 256;
            let cfg = LaunchConfig {
                grid_dim: (count.div_ceil(BLOCK as usize) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(q);
            builder.arg(&output);
            builder.arg(&self.count);
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("dequant_int4 launch: {e}")))?;
            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::BF16(output),
                    device: dev.clone(),
                },
                Shape::from_dims(&[count, 4]),
            ))
        }
    }

    fn dispatch_dequant_int4_bf16_lo4(q: &Tensor) -> Result<Tensor> {
        if q.dtype() != DType::U32 {
            candle_core::bail!("dequant_int4_bf16_lo4: q must be U32");
        }
        let count = q.dims().iter().product::<usize>();
        let q_flat = q.flatten_all()?.contiguous()?;
        q_flat.apply_op1(DequantInt4Bf16Lo4Op {
            count: count as i32,
        })
    }

    /// Stage 15.D-body.2a — single-tile bf16 mma.m16n8k16 probe.
    /// A: [16, 16] BF16 row-major; B: [8, 16] BF16 (col-major over [K=16, N=8],
    /// stored as `b[n*16 + k]`); output C: [16, 8] FP32 row-major.
    struct MmaM16N8K16Bf16Op {
        b: Tensor, // [8, 16] BF16, col-major K-N pack
    }

    impl CustomOp1 for MmaM16N8K16Bf16Op {
        fn name(&self) -> &'static str {
            "marlin_test_mma_m16n8k16_bf16"
        }
        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("mma_m16n8k16: CUDA only")
        }
        fn cuda_fwd(&self, storage: &CudaStorage, _l: &Layout) -> Result<(CudaStorage, Shape)> {
            use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
            let dev = &storage.device;
            let a = match &storage.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("mma_m16n8k16: A must be BF16"),
            };
            let (b_guard, _) = self.b.storage_and_layout();
            let b = match &*b_guard {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::BF16(s) => s,
                    _ => candle_core::bail!("mma_m16n8k16: B must be BF16"),
                },
                _ => candle_core::bail!("mma_m16n8k16: B must be on CUDA"),
            };
            let func = dev.get_or_load_custom_func(
                "marlin_test_mma_m16n8k16_bf16",
                "marlin_tile_mma",
                MARLIN_TILE_MMA_PTX,
            )?;
            let output = dev.alloc_zeros::<f32>(16 * 8)?;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(a);
            builder.arg(b);
            builder.arg(&output);
            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("mma_m16n8k16 launch: {e}")))?;
            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(output),
                    device: dev.clone(),
                },
                Shape::from_dims(&[16, 8]),
            ))
        }
    }

    fn dispatch_mma_m16n8k16_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.dims() != [16, 16] || a.dtype() != DType::BF16 {
            candle_core::bail!("mma_m16n8k16: A must be [16,16] BF16");
        }
        if b.dims() != [8, 16] || b.dtype() != DType::BF16 {
            candle_core::bail!("mma_m16n8k16: B must be [8,16] BF16 (col-major K,N pack)");
        }
        a.contiguous()?
            .apply_op1(MmaM16N8K16Bf16Op { b: b.contiguous()? })
    }

    #[test]
    fn test_mma_m16n8k16_bf16_correctness() {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        // A [16, 16] row-major; B [8, 16] col-major (B(k,n) = b[n*16+k]).
        // Use small deterministic floats so BF16 quantisation noise is
        // bounded. K=16 reduction → tolerance scales linearly: each
        // summand ≤ 1.0 × 1.0 = 1.0; bf16 has ~3-decimal mantissa, so
        // expect error per element ≲ 16 × 1.0 × 2^-7 ≈ 0.125.
        let a_vals: Vec<half::bf16> = (0..16 * 16)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.073) % 2.0) - 1.0))
            .collect();
        let b_vals: Vec<half::bf16> = (0..8 * 16)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.117) % 2.0) - 1.0))
            .collect();
        let a = Tensor::from_vec(a_vals.clone(), (16, 16), &cuda).unwrap();
        let b = Tensor::from_vec(b_vals.clone(), (8, 16), &cuda).unwrap();

        let c = dispatch_mma_m16n8k16_bf16(&a, &b).unwrap();
        let c_host: Vec<f32> = c.flatten_all().unwrap().to_vec1().unwrap();

        for m in 0..16 {
            for n in 0..8 {
                let mut acc = 0.0f32;
                for k in 0..16 {
                    let av = a_vals[m * 16 + k].to_f32();
                    // B is col-major [K=16, N=8]; we stored b[n*16+k]
                    let bv = b_vals[n * 16 + k].to_f32();
                    acc += av * bv;
                }
                let got = c_host[m * 8 + n];
                let diff = (got - acc).abs();
                assert!(
                    diff < 0.2,
                    "mma m16n8k16 mismatch at (m={m},n={n}): cpu={acc:.6} gpu={got:.6} diff={diff:.6}"
                );
            }
        }
    }

    #[test]
    fn test_dequant_int4_bf16_lo4_primitive() {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        // Deterministic test inputs covering the full byte range.
        let q_vals: Vec<u32> = (0..256u32).map(|i| i.wrapping_mul(0x9E3779B1)).collect();
        let q = Tensor::from_vec(q_vals.clone(), 256, &cuda).unwrap();

        let out = dispatch_dequant_int4_bf16_lo4(&q).unwrap();
        let out_host: Vec<half::bf16> = out.flatten_all().unwrap().to_vec1().unwrap();

        // CPU reference: same bit extraction as the kernel.
        for (i, &qv) in q_vals.iter().enumerate() {
            let expected = [
                (qv & 0xF) as f32,
                ((qv >> 16) & 0xF) as f32,
                ((qv >> 4) & 0xF) as f32,
                ((qv >> 20) & 0xF) as f32,
            ];
            for j in 0..4 {
                let got = out_host[i * 4 + j].to_f32();
                assert!(
                    (got - expected[j]).abs() < 1e-3,
                    "i={i} j={j} q=0x{qv:08x} expected={} got={got}",
                    expected[j]
                );
            }
        }
    }

    /// Numeric-correctness lock-down for arbitrary K (multiple of 16).
    ///
    /// Build a deterministic AWQ qweight + per-channel scales + activations,
    /// dispatch the v1 kernel, compare against a CPU reference that does the
    /// same dequant + multiply + matmul. BF16 reduction tolerance scales
    /// with K (each summand is up to ~16 in magnitude × bf16 mantissa noise).
    /// Run end-to-end lock-down for given (M, N, K, group_size).
    /// `group_size` must be a multiple of 16 that divides K.
    fn run_lock_down(m: usize, n: usize, k: usize, group_size: usize, tolerance: f32) {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        assert!(k.is_multiple_of(16), "K must be multiple of 16");
        assert!(n.is_multiple_of(64), "N must be multiple of 64");
        assert!(m >= 1);
        assert!(group_size.is_multiple_of(16) && k.is_multiple_of(group_size));
        let num_groups = k / group_size;

        // 1. Activations A[M, K] BF16: deterministic small floats in [-1, 1].
        let a_vals: Vec<half::bf16> = (0..m * k)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.137) % 2.0) - 1.0))
            .collect();
        let a = Tensor::from_vec(a_vals.clone(), (m, k), &cuda).unwrap();

        // 2. AWQ qweight: [K, N/8] u32 with deterministic logical nibbles.
        let nibble = |kk: usize, nn: usize| -> u32 { ((kk * 113 + nn * 29 + 7) as u32) % 16 };
        // Per-group, per-column zero point in [0, 15].
        let zp_val = |g: usize, nn: usize| -> u32 { ((g * 17 + nn * 11 + 3) as u32) % 16 };
        // Per-group, per-column scale.
        let scale_val =
            |g: usize, nn: usize| -> f32 { 0.05 + (g as f32) * 0.01 + (nn as f32) * 0.0025 };

        const AWQ_PACK_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
        let mut awq = vec![0u32; k * (n / 8)];
        for kk in 0..k {
            for nn in 0..n {
                awq[kk * (n / 8) + nn / 8] |= nibble(kk, nn) << AWQ_PACK_SHIFTS[nn % 8];
            }
        }

        // 3. Repack qweight via Stage 15.B.
        let awq_t = Tensor::from_vec(awq, (k, n / 8), &Device::Cpu).unwrap();
        let tile_full = awq_to_marlin_tile_repack_cpu(&awq_t, k, n).unwrap();
        let b_tile = tile_full.flatten_all().unwrap().to_device(&cuda).unwrap();

        // 4. Per-group scales [num_groups, N] BF16.
        let mut scales_vec: Vec<half::bf16> = Vec::with_capacity(num_groups * n);
        for g in 0..num_groups {
            for nn in 0..n {
                scales_vec.push(half::bf16::from_f32(scale_val(g, nn)));
            }
        }
        let scales = Tensor::from_vec(scales_vec.clone(), (num_groups, n), &cuda).unwrap();

        // 5. GPTQ-sequential qzeros [num_groups, N/8] u32 (matches
        // MarlinLinear's stored format after repack_awq_nibbles).
        let mut qz = vec![0u32; num_groups * (n / 8)];
        for g in 0..num_groups {
            for nn in 0..n {
                qz[g * (n / 8) + nn / 8] |= zp_val(g, nn) << ((nn % 8) * 4);
            }
        }
        let qzeros = Tensor::from_vec(qz, (num_groups, n / 8), &cuda).unwrap();

        // 6. Dispatch.
        let c = dispatch_marlin_tile_mma_v1(&a, &b_tile, &scales, &qzeros, n, group_size).unwrap();
        let c_host: Vec<half::bf16> = c.flatten_all().unwrap().to_vec1().unwrap();

        // 7. CPU reference.
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let g = kk / group_size;
                    let w = (nibble(kk, col) as f32 - zp_val(g, col) as f32) * scale_val(g, col);
                    let av = a_vals[row * k + kk].to_f32();
                    acc += av * w;
                }
                let got = c_host[row * n + col].to_f32();
                let diff = (got - acc).abs();
                assert!(
                    diff < tolerance,
                    "M={m} N={n} K={k} g={group_size} mismatch at (row={row}, col={col}): cpu={acc:.6} gpu={got:.6} diff={diff:.6}"
                );
            }
        }
    }

    // ─── group_size = K (per-channel, smallest case) ─────────────────────

    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k16_g16() {
        run_lock_down(16, 64, 16, 16, 5e-2);
    }

    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k64_g64() {
        run_lock_down(16, 64, 64, 64, 2e-1);
    }

    #[test]
    fn test_marlin_tile_mma_v1_n64_m1_k64_g64() {
        run_lock_down(1, 64, 64, 64, 2e-1);
    }

    #[test]
    fn test_marlin_tile_mma_v1_n64_m4_k64_g64() {
        run_lock_down(4, 64, 64, 64, 2e-1);
    }

    #[test]
    fn test_marlin_tile_mma_v1_n128_m16_k64_g64() {
        run_lock_down(16, 128, 64, 64, 2e-1);
    }

    // ─── group_size < K (multiple groups along K) ────────────────────────

    /// K=64 split into 4 groups of 16 each.
    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k64_g16() {
        run_lock_down(16, 64, 64, 16, 2e-1);
    }

    /// K=128 with group_size=32 (4 groups). Larger reduction depth.
    #[test]
    fn test_marlin_tile_mma_v1_n64_m16_k128_g32() {
        run_lock_down(16, 64, 128, 32, 4e-1);
    }

    /// K=128 with the AWQ default group_size=128 (single group).
    #[test]
    fn test_marlin_tile_mma_v1_n128_m4_k128_g128() {
        run_lock_down(4, 128, 128, 128, 4e-1);
    }

    /// Multi-block, multi-group, larger N.
    #[test]
    fn test_marlin_tile_mma_v1_n256_m32_k64_g32() {
        run_lock_down(32, 256, 64, 32, 4e-1);
    }
}
