//! Native I-quant (IQ) support for GGUF.
//!
//! candle 0.10 has no I-quant (`IQ*`) support — its `GgmlDType` covers only
//! the legacy quants and the K-quants. The Unsloth "UD" (Unsloth Dynamic)
//! checkpoints mix several I-quant types across tensors (e.g.
//! `gemma-4-12b-it-UD-IQ3_XXS.gguf` uses IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S /
//! IQ4_XS), so candle rejects the file at header parse. This module adds the
//! five types that checkpoint needs.
//!
//! The dequantization is ported byte-for-byte from ggml's
//! `dequantize_row_iq*` (`reference/llama.cpp/ggml/src/ggml-quants.c`), and the
//! codebook tables in [`tables`] are extracted verbatim from
//! `ggml-common.h` by `scripts/extract_iq_tables.py`. Correctness is pinned
//! against golden vectors generated from the real model file by the
//! `gguf` Python package (`scripts/gen_iq_fixtures.py`) — see the tests.
//!
//! Only **de**quantization is implemented: we read I-quant weights, never
//! write them. The weights stay I-quant-resident on the GPU; the matmul path
//! dequantizes on the fly (see the CUDA kernels and `IqLinear`).

mod tables;

#[cfg(all(test, feature = "cuda-kernels"))]
mod cuda_tests;

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use half::f16;
use tables::{
    IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

/// GGML super-block size (elements per block) for every K-/I-quant.
pub const QK_K: usize = 256;

/// The I-quant types present in the Unsloth Dynamic Gemma-4 checkpoints that
/// candle cannot parse. Each maps to a GGML type id (see ggml.h) and a fixed
/// on-disk block byte size for `QK_K = 256`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IqType {
    /// 2.3125 bpw, GGML type 17.
    Iq2Xs,
    /// 2.5625 bpw, GGML type 22.
    Iq2S,
    /// 3.0625 bpw, GGML type 18.
    Iq3Xxs,
    /// 3.4375 bpw, GGML type 21.
    Iq3S,
    /// 4.25 bpw, GGML type 23.
    Iq4Xs,
}

impl IqType {
    /// Map a GGML tensor type id to an [`IqType`], or `None` if the id is not
    /// one of the supported I-quants (candle handles the rest).
    pub fn from_ggml_type_id(id: u32) -> Option<Self> {
        Some(match id {
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            _ => return None,
        })
    }

    /// The GGML tensor type id for this I-quant.
    pub fn ggml_type_id(self) -> u32 {
        match self {
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
        }
    }

    /// Elements per block (always [`QK_K`] for these types).
    pub fn block_size(self) -> usize {
        QK_K
    }

    /// On-disk byte size of one block. Matches the `block_iq*` structs in
    /// `ggml-common.h` for `QK_K = 256` (verified by `static_assert` there):
    /// iq2_xs 74, iq2_s 82, iq3_xxs 98, iq3_s 110, iq4_xs 136.
    pub fn type_size(self) -> usize {
        match self {
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq3S => 110,
            Self::Iq4Xs => 136,
        }
    }

    /// Short human-readable name (matches llama.cpp's `IQ*` spelling).
    pub fn name(self) -> &'static str {
        match self {
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq2S => "IQ2_S",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq3S => "IQ3_S",
            Self::Iq4Xs => "IQ4_XS",
        }
    }
}

/// Read the little-endian f16 scale at the start of a block.
#[inline]
fn block_scale(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[0], block[1]]).to_f32()
}

/// Byte `j` (0-based) of an 8-byte grid entry stored as a little-endian u64.
#[inline]
fn grid_u64_byte(entry: u64, j: usize) -> f32 {
    ((entry >> (8 * j)) & 0xff) as u8 as f32
}

/// Byte `j` (0-based) of a 4-byte grid entry stored as a little-endian u32.
#[inline]
fn grid_u32_byte(entry: u32, j: usize) -> f32 {
    ((entry >> (8 * j)) & 0xff) as u8 as f32
}

/// Sign multiplier for output lane `j` given a packed sign byte.
#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Dequantize one IQ2_XS block (74 bytes → 256 f32).
fn dequant_iq2_xs(block: &[u8], y: &mut [f32]) {
    let d = block_scale(block);
    // qs: u16[QK_K/8 = 32] at byte 2; scales: u8[QK_K/32 = 8] at byte 66.
    let qs = |i: usize| u16::from_le_bytes([block[2 + 2 * i], block[2 + 2 * i + 1]]);
    let scales = |i: usize| block[66 + i];

    let mut yi = 0;
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + (scales(ib32) & 0xf) as f32) * 0.25,
            d * (0.5 + (scales(ib32) >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = qs(4 * ib32 + l);
            let grid = IQ2XS_GRID[(q & 511) as usize];
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                y[yi + j] = db[l / 2] * grid_u64_byte(grid, j) * sign(signs, j);
            }
            yi += 8;
        }
    }
}

/// Dequantize one IQ2_S block (82 bytes → 256 f32).
fn dequant_iq2_s(block: &[u8], y: &mut [f32]) {
    let d = block_scale(block);
    // qs: u8[QK_K/4 = 64] at byte 2 (the upper 32 bytes are the sign bytes:
    // ggml sets `signs = qs + QK_K/8`). qh: u8[8] at byte 66. scales: u8[8] at
    // byte 74.
    let qs = |i: usize| block[2 + i];
    let signs = |i: usize| block[2 + QK_K / 8 + i];
    let qh = |i: usize| block[66 + i];
    let scales = |i: usize| block[74 + i];

    let mut yi = 0;
    let mut qs_off = 0;
    let mut signs_off = 0;
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + (scales(ib32) & 0xf) as f32) * 0.25,
            d * (0.5 + (scales(ib32) >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx = (qs(qs_off + l) as usize) | (((qh(ib32) as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx];
            let sgn = signs(signs_off + l);
            for j in 0..8 {
                y[yi + j] = dl * grid_u64_byte(grid, j) * sign(sgn, j);
            }
            yi += 8;
        }
        qs_off += 4;
        signs_off += 4;
    }
}

/// Dequantize one IQ3_XXS block (98 bytes → 256 f32).
fn dequant_iq3_xxs(block: &[u8], y: &mut [f32]) {
    let d = block_scale(block);
    // qs: u8[3*QK_K/8 = 96] at byte 2. scales_and_signs (u32[8]) follow at
    // `qs + QK_K/4` = byte 66.
    let qs = |i: usize| block[2 + i] as usize;
    let sas_base = 66;

    let mut yi = 0;
    let mut qs_off = 0;
    for ib32 in 0..QK_K / 32 {
        let o = sas_base + 4 * ib32;
        let aux32 = u32::from_le_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]]);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let grid1 = IQ3XXS_GRID[qs(qs_off + 2 * l)];
            let grid2 = IQ3XXS_GRID[qs(qs_off + 2 * l + 1)];
            for j in 0..4 {
                y[yi + j] = db * grid_u32_byte(grid1, j) * sign(signs, j);
                y[yi + j + 4] = db * grid_u32_byte(grid2, j) * sign(signs, j + 4);
            }
            yi += 8;
        }
        qs_off += 8;
    }
}

/// Dequantize one IQ3_S block (110 bytes → 256 f32).
fn dequant_iq3_s(block: &[u8], y: &mut [f32]) {
    let d = block_scale(block);
    // qs: u8[QK_K/4 = 64] at byte 2. qh: u8[8] at byte 66. signs: u8[32] at
    // byte 74. scales: u8[IQ3S_N_SCALE = 4] at byte 106.
    let qs = |i: usize| block[2 + i] as usize;
    let qh = |i: usize| block[66 + i] as usize;
    let signs = |i: usize| block[74 + i];
    let scales = |i: usize| block[106 + i];

    let mut yi = 0;
    let mut qs_off = 0;
    let mut signs_off = 0;
    let mut qh_off = 0;
    // ggml iterates ib32 in steps of 2; each step handles two 32-lane groups
    // sharing one scale byte, the first using qh[qh_off] and the second
    // qh[qh_off + 1].
    for ib32 in (0..QK_K / 32).step_by(2) {
        let sc = scales(ib32 / 2);
        let db1 = d * (1 + 2 * (sc & 0xf) as i32) as f32;
        let db2 = d * (1 + 2 * (sc >> 4) as i32) as f32;

        for l in 0..4 {
            let g1 = IQ3S_GRID[qs(qs_off + 2 * l) | ((qh(qh_off) << (8 - 2 * l)) & 256)];
            let g2 = IQ3S_GRID[qs(qs_off + 2 * l + 1) | ((qh(qh_off) << (7 - 2 * l)) & 256)];
            let sgn = signs(signs_off + l);
            for j in 0..4 {
                y[yi + j] = db1 * grid_u32_byte(g1, j) * sign(sgn, j);
                y[yi + j + 4] = db1 * grid_u32_byte(g2, j) * sign(sgn, j + 4);
            }
            yi += 8;
        }
        qs_off += 8;
        signs_off += 4;

        for l in 0..4 {
            let g1 = IQ3S_GRID[qs(qs_off + 2 * l) | ((qh(qh_off + 1) << (8 - 2 * l)) & 256)];
            let g2 = IQ3S_GRID[qs(qs_off + 2 * l + 1) | ((qh(qh_off + 1) << (7 - 2 * l)) & 256)];
            let sgn = signs(signs_off + l);
            for j in 0..4 {
                y[yi + j] = db2 * grid_u32_byte(g1, j) * sign(sgn, j);
                y[yi + j + 4] = db2 * grid_u32_byte(g2, j) * sign(sgn, j + 4);
            }
            yi += 8;
        }
        qh_off += 2;
        qs_off += 8;
        signs_off += 4;
    }
}

/// Dequantize one IQ4_XS block (136 bytes → 256 f32).
fn dequant_iq4_xs(block: &[u8], y: &mut [f32]) {
    let d = block_scale(block);
    // scales_h: u16 at byte 2. scales_l: u8[QK_K/64 = 4] at byte 4.
    // qs: u8[QK_K/2 = 128] at byte 8.
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = |i: usize| block[4 + i];
    let qs = |i: usize| block[8 + i];

    let mut yi = 0;
    for ib in 0..QK_K / 32 {
        let ls = (((scales_l(ib / 2) >> (4 * (ib % 2))) & 0xf) as i32)
            | ((((scales_h >> (2 * ib)) & 3) << 4) as i32);
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            let q = qs(16 * ib + j);
            y[yi + j] = dl * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
            y[yi + 16 + j] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
        }
        yi += 32;
    }
}

/// Dequantize a whole I-quant tensor (`data` = `n_blocks` packed blocks) into
/// `out` (`n_blocks * QK_K` f32, row-major).
///
/// Returns an error if the buffer sizes are inconsistent with `iq`'s block
/// geometry. Element count must be a multiple of [`QK_K`] — always true for
/// GGUF weight tensors (the quantizer pads the last partial block).
pub fn dequantize(iq: IqType, data: &[u8], out: &mut [f32]) -> candle_core::Result<()> {
    let ts = iq.type_size();
    let bs = iq.block_size();
    if !out.len().is_multiple_of(bs) {
        candle_core::bail!(
            "{}: output length {} is not a multiple of block size {bs}",
            iq.name(),
            out.len()
        );
    }
    let n_blocks = out.len() / bs;
    let expected = n_blocks * ts;
    if data.len() != expected {
        candle_core::bail!(
            "{}: data length {} != expected {expected} ({n_blocks} blocks × {ts} B)",
            iq.name(),
            data.len()
        );
    }

    let deq = match iq {
        IqType::Iq2Xs => dequant_iq2_xs,
        IqType::Iq2S => dequant_iq2_s,
        IqType::Iq3Xxs => dequant_iq3_xxs,
        IqType::Iq3S => dequant_iq3_s,
        IqType::Iq4Xs => dequant_iq4_xs,
    };
    for b in 0..n_blocks {
        let block = &data[b * ts..(b + 1) * ts];
        let yb = &mut out[b * bs..(b + 1) * bs];
        deq(block, yb);
    }
    Ok(())
}

#[cfg(feature = "cuda-kernels")]
impl IqType {
    /// The `kernels/iq_dequant.cu` entry point that dequantizes this type.
    fn dequant_kernel(self) -> &'static str {
        match self {
            IqType::Iq2Xs => "dequantize_iq2_xs_f32",
            IqType::Iq2S => "dequantize_iq2_s_f32",
            IqType::Iq3Xxs => "dequantize_iq3_xxs_f32",
            IqType::Iq3S => "dequantize_iq3_s_f32",
            IqType::Iq4Xs => "dequantize_iq4_xs_f32",
        }
    }
}

#[cfg(feature = "cuda-kernels")]
const IQ_DEQUANT_PTX: &str = include_str!("../../../../kernels/iq_dequant.ptx");

/// `CustomOp1` over the I-quant weight bytes (U8) producing the dense
/// dequantized values `[n_elements]` (F32). The caller reshapes to
/// `[out, in]`.
///
/// On CUDA it launches `kernels/iq_dequant.cu` (weight stays I-quant-resident
/// in VRAM); on CPU it runs the scalar [`dequantize`] port. Both are pinned
/// against the same gguf-py golden vectors, so the two paths agree.
struct DequantIqOp {
    iq: IqType,
    n_elements: usize,
}

impl CustomOp1 for DequantIqOp {
    fn name(&self) -> &'static str {
        "dequantize_iq"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let bytes = match storage {
            CpuStorage::U8(b) => b,
            _ => candle_core::bail!("dequantize_iq: weight bytes must be U8"),
        };
        let mut out = vec![0f32; self.n_elements];
        dequantize(self.iq, bytes, &mut out)?;
        Ok((CpuStorage::F32(out), Shape::from_dims(&[self.n_elements])))
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda::CudaStorageSlice;

        let dev = &storage.device;
        let blocks = match &storage.slice {
            CudaStorageSlice::U8(s) => s,
            _ => candle_core::bail!("dequantize_iq: weight bytes must be U8"),
        };

        if !self.n_elements.is_multiple_of(QK_K) {
            candle_core::bail!(
                "dequantize_iq: n_elements {} not a multiple of {QK_K}",
                self.n_elements
            );
        }
        let n_blocks = self.n_elements / QK_K;
        let expected_bytes = n_blocks * self.iq.type_size();
        if blocks.len() < expected_bytes {
            candle_core::bail!(
                "dequantize_iq: have {} weight bytes, need {expected_bytes}",
                blocks.len()
            );
        }

        let output = dev.alloc_zeros::<f32>(self.n_elements)?;
        let func =
            dev.get_or_load_custom_func(self.iq.dequant_kernel(), "iq_dequant", IQ_DEQUANT_PTX)?;

        const THREADS: u32 = 256;
        let grid = (n_blocks as u32).div_ceil(THREADS);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_blocks_i32 = n_blocks as i32;
        let mut builder = func.builder();
        builder.arg(blocks);
        builder.arg(&output);
        builder.arg(&n_blocks_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("dequantize_iq launch: {e}")))?;

        let output_storage = candle_core::CudaStorage {
            slice: CudaStorageSlice::F32(output),
            device: dev.clone(),
        };
        Ok((output_storage, Shape::from_dims(&[self.n_elements])))
    }
}

/// Dequantize an I-quant weight to a flat dense f32 tensor `[n_elements]`.
///
/// * `bytes` — raw GGUF block bytes of the weight, U8 (CUDA in production,
///   CPU in tests / on a CPU device).
/// * `n_elements` — total dequantized element count (a multiple of [`QK_K`];
///   GGUF weight tensors always are).
pub fn dequantize_iq(bytes: &Tensor, iq: IqType, n_elements: usize) -> Result<Tensor> {
    bytes.apply_op1_no_bwd(&DequantIqOp { iq, n_elements })
}

/// Largest activation-row count the fused GEMV kernel handles in one launch
/// (matches `IQ_GEMV_MAX_M` in `kernels/iq_dequant.cu`). Decode batches are
/// far smaller; larger M (prefill) takes the dequant + cuBLAS path.
pub const IQ_GEMV_MAX_M: usize = 16;

#[cfg(feature = "cuda-kernels")]
impl IqType {
    /// The `kernels/iq_dequant.cu` MMVQ (q8_1 × I-quant) entry point.
    fn mmvq_kernel(self) -> &'static str {
        match self {
            IqType::Iq2Xs => "mmvq_iq2_xs_f32",
            IqType::Iq2S => "mmvq_iq2_s_f32",
            IqType::Iq3Xxs => "mmvq_iq3_xxs_f32",
            IqType::Iq3S => "mmvq_iq3_s_f32",
            IqType::Iq4Xs => "mmvq_iq4_xs_f32",
        }
    }
}

// ===========================================================================
// CPU q8_1 MMVQ reference — the exact integer algorithm the CUDA `mmvq_*`
// kernels run, used as `IqMatmulOp::cpu_fwd` and as the bit-level pin for the
// GPU kernels. A one-to-one port of ggml's `vec_dot_iq*_q8_1`
// (reference/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh). The SIMD primitives
// (`__dp4a`, `__vsub4`, `__vcmpne4`, `__byte_perm`) are reproduced with
// portable byte arithmetic so the integer results are identical to the kernel.
// ===========================================================================

/// SIMD int8x4 dot-product with accumulate: `c + Σ a_i·b_i` over the four
/// signed bytes of `a` and `b`.
#[inline]
fn dp4a(a: i32, b: i32, c: i32) -> i32 {
    let mut acc = c;
    for i in 0..4 {
        let av = ((a >> (8 * i)) & 0xff) as u8 as i8 as i32;
        let bv = ((b >> (8 * i)) & 0xff) as u8 as i8 as i32;
        acc += av * bv;
    }
    acc
}

/// Per-byte "set if not equal to zero" → 0xFF / 0x00 (mirrors `__vcmpne4(x,0)`).
#[inline]
fn vcmpne4(x: u32) -> i32 {
    let mut r = 0u32;
    for i in 0..4 {
        if (x >> (8 * i)) & 0xff != 0 {
            r |= 0xffu32 << (8 * i);
        }
    }
    r as i32
}

/// Per-byte wrapping subtract (mirrors `__vsub4`).
#[inline]
fn vsub4(a: i32, b: i32) -> i32 {
    let (a, b) = (a as u32, b as u32);
    let mut r = 0u32;
    for i in 0..4 {
        let av = ((a >> (8 * i)) & 0xff) as u8;
        let bv = ((b >> (8 * i)) & 0xff) as u8;
        r |= (av.wrapping_sub(bv) as u32) << (8 * i);
    }
    r as i32
}

/// Negate the four signed bytes of `g` where the matching byte of `signs`
/// (a `vcmpne4` mask) is set. Two's-complement: `(g ^ 0xFF) - 0xFF == -g`.
#[inline]
fn apply_signs(g: i32, signs: i32) -> i32 {
    vsub4(g ^ signs, signs)
}

/// Expand a 7-bit ggml sign index to a per-byte selector. The 8th sign is the
/// parity of the 7 bits; the xor cancels any stray high bit the caller leaves
/// in, so callers need not mask (mirrors `unpack_ksigns`).
#[inline]
fn unpack_ksigns(v: u8) -> u32 {
    let p = v.count_ones() & 1;
    let s = (v as u32) ^ (p << 7);
    s.wrapping_mul(0x0101_0101)
}

/// Look up eight 4-bit indices packed in `q4` against a 16-entry int8 table.
/// `.0` = even-nibble (low) lookups, `.1` = odd-nibble (high), each int8x4
/// (mirrors `get_int_from_table_16`).
#[inline]
fn table16(q4: i32, table: &[i8; 16]) -> (i32, i32) {
    let q4 = q4 as u32;
    let mut v0 = 0u32;
    let mut v1 = 0u32;
    for i in 0..4 {
        let lo = ((q4 >> (8 * i)) & 0xf) as usize;
        let hi = ((q4 >> (8 * i + 4)) & 0xf) as usize;
        v0 |= (table[lo] as u8 as u32) << (8 * i);
        v1 |= (table[hi] as u8 as u32) << (8 * i);
    }
    (v0 as i32, v1 as i32)
}

/// Low/high 32 bits of a u64 grid entry as the two int8x4 lanes (matches the
/// CUDA `(const uint2*)`/`(const int*)` reinterpret of the codebook).
#[inline]
fn grid_u64_halves(entry: u64) -> (i32, i32) {
    (entry as u32 as i32, (entry >> 32) as u32 as i32)
}

fn cpu_dot_iq2_xs(b: &[u8], ib: usize, u: &[i32; 8]) -> i32 {
    let qs = |i: usize| u16::from_le_bytes([b[2 + 2 * i], b[2 + 2 * i + 1]]);
    let scales = b[66 + ib];
    let (mut s0, mut s1) = (0i32, 0i32);
    for k in 0..4 {
        let q = qs(4 * ib + k);
        let (gx, gy) = grid_u64_halves(IQ2XS_GRID[(q & 0x1ff) as usize]);
        let signs = unpack_ksigns((q >> 9) as u8);
        let gl = apply_signs(gx, vcmpne4(signs & 0x0804_0201));
        let gh = apply_signs(gy, vcmpne4(signs & 0x8040_2010));
        if k < 2 {
            s0 = dp4a(gl, u[2 * k], s0);
            s0 = dp4a(gh, u[2 * k + 1], s0);
        } else {
            s1 = dp4a(gl, u[2 * k], s1);
            s1 = dp4a(gh, u[2 * k + 1], s1);
        }
    }
    let (ls0, ls1) = ((scales & 0xf) as i32, (scales >> 4) as i32);
    (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4
}

fn cpu_dot_iq2_s(b: &[u8], ib: usize, u: &[i32; 8]) -> i32 {
    let idx = |k: usize| b[2 + 4 * ib + k];
    let sgn = |k: usize| b[2 + QK_K / 8 + 4 * ib + k];
    let qh = b[66 + ib] as i32;
    let scales = b[74 + ib];
    let (mut s0, mut s1) = (0i32, 0i32);
    for k in 0..4 {
        let l0 = 2 * k;
        let gi = (idx(k) as i32 | ((qh << (8 - l0)) & 0x300)) as usize;
        let (g0, g1) = grid_u64_halves(IQ2S_GRID[gi]);
        let sb = sgn(k);
        let sg0 = vcmpne4((((sb & 0x03) as u32) << 7) | (((sb & 0x0C) as u32) << 21));
        let sg1 = vcmpne4((((sb & 0x30) as u32) << 3) | (((sb & 0xC0) as u32) << 17));
        let gl = apply_signs(g0, sg0);
        let gh = apply_signs(g1, sg1);
        if k < 2 {
            s0 = dp4a(gl, u[2 * k], s0);
            s0 = dp4a(gh, u[2 * k + 1], s0);
        } else {
            s1 = dp4a(gl, u[2 * k], s1);
            s1 = dp4a(gh, u[2 * k + 1], s1);
        }
    }
    let (ls0, ls1) = ((scales & 0xf) as i32, (scales >> 4) as i32);
    (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4
}

fn cpu_dot_iq3_xxs(b: &[u8], ib: usize, u: &[i32; 8]) -> i32 {
    let qs = |i: usize| b[2 + 8 * ib + i] as usize;
    let o = 2 + QK_K / 4 + 4 * ib;
    let aux32 = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut sumi = 0i32;
    for k in 0..4 {
        let g0 = IQ3XXS_GRID[qs(2 * k)] as i32;
        let g1 = IQ3XXS_GRID[qs(2 * k + 1)] as i32;
        let signs = unpack_ksigns((aux32 >> (7 * k)) as u8);
        let gl = apply_signs(g0, vcmpne4(signs & 0x0804_0201));
        let gh = apply_signs(g1, vcmpne4(signs & 0x8040_2010));
        sumi = dp4a(gl, u[2 * k], sumi);
        sumi = dp4a(gh, u[2 * k + 1], sumi);
    }
    let ls = (aux32 >> 28) as i32;
    (ls * sumi + sumi / 2) / 2
}

fn cpu_dot_iq3_s(b: &[u8], ib: usize, u: &[i32; 8]) -> i32 {
    let qs = |i: usize| b[2 + 8 * ib + i] as i32;
    let qh = b[66 + ib] as i32;
    let sgn = |k: usize| b[74 + 4 * ib + k];
    let scales = |i: usize| b[106 + i];
    let mut sumi = 0i32;
    for k in 0..4 {
        let l0 = 2 * k;
        let g0 = IQ3S_GRID[(qs(2 * k) | ((qh << (8 - l0)) & 0x100)) as usize] as i32;
        let g1 = IQ3S_GRID[(qs(2 * k + 1) | ((qh << (7 - l0)) & 0x100)) as usize] as i32;
        let sb = sgn(k);
        let sg0 = vcmpne4((((sb & 0x03) as u32) << 7) | (((sb & 0x0C) as u32) << 21));
        let sg1 = vcmpne4((((sb & 0x30) as u32) << 3) | (((sb & 0xC0) as u32) << 17));
        sumi = dp4a(apply_signs(g0, sg0), u[2 * k], sumi);
        sumi = dp4a(apply_signs(g1, sg1), u[2 * k + 1], sumi);
    }
    let nib = ((scales(ib / 2) >> (4 * (ib & 1))) & 0x0f) as i32;
    sumi * (1 + 2 * nib)
}

fn cpu_dot_iq4_xs(b: &[u8], ib: usize, u: &[i32; 8]) -> i32 {
    let scales_h = u16::from_le_bytes([b[2], b[3]]);
    let scales_l = |i: usize| b[4 + i];
    let mut sumi = 0i32;
    for j in 0..4 {
        let o = 8 + 16 * ib + 4 * j;
        let aux = i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (vx, vy) = table16(aux, &KVALUES_IQ4NL);
        sumi = dp4a(vx, u[j], sumi);
        sumi = dp4a(vy, u[j + 4], sumi);
    }
    let iqs = 4 * ib;
    let ls = (((scales_l(ib / 2) >> (iqs & 0x04)) & 0x0f) as i32)
        | ((((scales_h >> (iqs / 2)) & 0x03) as i32) << 4);
    sumi * (ls - 32)
}

/// Quantize a row-major `[m, n_in]` activation to q8_1: per 32-element block,
/// `d = amax/127` and `q = round(x/d)`. Returns `(qs i8 [m·n_in], d f32
/// [m·n_in/32])`. Identical to the `quantize_q8_1_rows` kernel.
fn quantize_q8_1_cpu(x: &[f32], n_in: usize, m: usize) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = m * (n_in / 32);
    let mut aq = vec![0i8; m * n_in];
    let mut ad = vec![0f32; n_blocks];
    for (blk, ad_blk) in ad.iter_mut().enumerate() {
        let base = blk * 32;
        let mut amax = 0f32;
        for i in 0..32 {
            amax = amax.max(x[base + i].abs());
        }
        let d = amax / 127.0;
        *ad_blk = d;
        if amax != 0.0 {
            for i in 0..32 {
                aq[base + i] = (x[base + i] / d).round() as i8;
            }
        }
    }
    (aq, ad)
}

/// CPU q8_1 MMVQ: `Y[m, n_out] = X[m, n_in] @ Wᵀ` over the I-quant weight
/// `w` (`n_out · n_in/QK_K` blocks). Quantizes X to q8_1, then for each output
/// row sums the per-sub-block integer dots scaled by the weight/activation
/// deltas. Bit-for-bit the algorithm the `mmvq_*` kernels run (modulo f32
/// reduction order).
fn q8_1_mmvq_cpu(w: &[u8], x: &[f32], iq: IqType, n_out: usize, n_in: usize, m: usize) -> Vec<f32> {
    let ts = iq.type_size();
    let nb256 = n_in / QK_K;
    let nblk32 = n_in / 32;
    let (aq, ad) = quantize_q8_1_cpu(x, n_in, m);
    let dot: fn(&[u8], usize, &[i32; 8]) -> i32 = match iq {
        IqType::Iq2Xs => cpu_dot_iq2_xs,
        IqType::Iq2S => cpu_dot_iq2_s,
        IqType::Iq3Xxs => cpu_dot_iq3_xxs,
        IqType::Iq3S => cpu_dot_iq3_s,
        IqType::Iq4Xs => cpu_dot_iq4_xs,
    };
    let mut y = vec![0f32; m * n_out];
    for o in 0..n_out {
        for mi in 0..m {
            let mut acc = 0f32;
            for g in 0..nblk32 {
                let (sb, ib) = (g >> 3, g & 7);
                let off = (o * nb256 + sb) * ts;
                let wb = &w[off..off + ts];
                let w_d = block_scale(wb);
                let u: [i32; 8] = core::array::from_fn(|j| {
                    let p = mi * n_in + g * 32 + 4 * j;
                    i32::from_le_bytes([
                        aq[p] as u8,
                        aq[p + 1] as u8,
                        aq[p + 2] as u8,
                        aq[p + 3] as u8,
                    ])
                });
                acc += w_d * ad[mi * nblk32 + g] * (dot(wb, ib, &u) as f32);
            }
            y[mi * n_out + o] = acc;
        }
    }
    y
}

/// Fused `Y = X @ Wᵀ` over an I-quant weight via q8_1 MMVQ — for small M
/// (decode). The activation row is quantized to q8_1 (int8 + per-32 scale)
/// once, then the I-quant weight is dotted against it with integer `__dp4a`;
/// the dense weight is never materialized and the inner loop has no float
/// multiplies. `m` must be `<= IQ_GEMV_MAX_M`.
struct IqMatmulOp {
    /// Activation `[m, n_in]`, f32, contiguous, same device as the weight.
    x: Tensor,
    iq: IqType,
    n_out: usize,
    n_in: usize,
    m: usize,
}

impl CustomOp1 for IqMatmulOp {
    fn name(&self) -> &'static str {
        "iq_matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // Reference path (CPU device / tests): the exact q8_1 MMVQ algorithm
        // the CUDA kernel runs (integer dots over q8_1-quantized activations).
        let bytes = match storage {
            CpuStorage::U8(b) => b,
            _ => candle_core::bail!("iq_matmul: weight bytes must be U8"),
        };
        let x = self.x.flatten_all()?.to_vec1::<f32>()?;
        let y = q8_1_mmvq_cpu(bytes, &x, self.iq, self.n_out, self.n_in, self.m);
        Ok((CpuStorage::F32(y), Shape::from_dims(&[self.m, self.n_out])))
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda::CudaStorageSlice;
        use candle_core::Storage;

        if self.m > IQ_GEMV_MAX_M {
            candle_core::bail!("iq_matmul: m {} exceeds IQ_GEMV_MAX_M", self.m);
        }
        if !self.n_in.is_multiple_of(QK_K) {
            candle_core::bail!("iq_matmul: n_in {} not a multiple of {QK_K}", self.n_in);
        }

        let dev = &storage.device;
        let w = match &storage.slice {
            CudaStorageSlice::U8(s) => s,
            _ => candle_core::bail!("iq_matmul: weight bytes must be U8"),
        };

        let (x_guard, x_layout) = self.x.storage_and_layout();
        let x = match &*x_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("iq_matmul: activation must be F32"),
            },
            _ => candle_core::bail!("iq_matmul: activation must be on CUDA"),
        };
        let x = match x_layout.contiguous_offsets() {
            Some((o1, o2)) => x.slice(o1..o2),
            None => candle_core::bail!("iq_matmul: activation must be contiguous"),
        };
        if !self.n_in.is_multiple_of(32) {
            candle_core::bail!("iq_matmul: n_in {} not a multiple of 32", self.n_in);
        }

        // Step 1: quantize the activation rows to q8_1 (int8 quants in `aq`,
        // per-32-block f32 deltas in `ad`). Done once, reused for every row.
        let total_blocks = self.m * (self.n_in / 32);
        let aq = dev.alloc_zeros::<u8>(self.m * self.n_in)?;
        let ad = dev.alloc_zeros::<f32>(total_blocks)?;
        let qfunc =
            dev.get_or_load_custom_func("quantize_q8_1_rows", "iq_dequant", IQ_DEQUANT_PTX)?;
        let total_blocks_i32 = total_blocks as i32;
        const QUANT_THREADS: u32 = 256;
        let qcfg = LaunchConfig {
            grid_dim: ((total_blocks as u32).div_ceil(QUANT_THREADS), 1, 1),
            block_dim: (QUANT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut qb = qfunc.builder();
        qb.arg(&x);
        qb.arg(&aq);
        qb.arg(&ad);
        qb.arg(&total_blocks_i32);
        unsafe { qb.launch(qcfg) }
            .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_1 launch: {e}")))?;

        // Step 2: MMVQ — one warp per output row, IQ_GEMV_ROWS_PER_BLOCK (=4)
        // rows per CUDA block. Must match kernels/iq_dequant.cu.
        let output = dev.alloc_zeros::<f32>(self.m * self.n_out)?;
        let func =
            dev.get_or_load_custom_func(self.iq.mmvq_kernel(), "iq_dequant", IQ_DEQUANT_PTX)?;
        const THREADS: u32 = 128;
        const ROWS_PER_BLOCK: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: ((self.n_out as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_out_i32 = self.n_out as i32;
        let n_in_i32 = self.n_in as i32;
        let m_i32 = self.m as i32;
        let mut builder = func.builder();
        builder.arg(w);
        builder.arg(&aq);
        builder.arg(&ad);
        builder.arg(&output);
        builder.arg(&n_out_i32);
        builder.arg(&n_in_i32);
        builder.arg(&m_i32);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("iq_matmul launch: {e}")))?;

        drop(x_guard);
        let output_storage = candle_core::CudaStorage {
            slice: CudaStorageSlice::F32(output),
            device: dev.clone(),
        };
        Ok((output_storage, Shape::from_dims(&[self.m, self.n_out])))
    }
}

/// Fused `Y = X @ Wᵀ` for an I-quant weight via q8_1 MMVQ (decode path).
///
/// * `w_bytes` — raw GGUF block bytes of the weight `[n_out, n_in]`, U8.
/// * `x` — activation `[m, n_in]`, f32, contiguous; `m <= IQ_GEMV_MAX_M`.
///
/// Returns `Y` `[m, n_out]` f32. The activation is quantized to q8_1 and
/// dotted against the I-quant weight with integer arithmetic; the dense weight
/// is never materialized.
pub fn iq_matmul(
    w_bytes: &Tensor,
    x: &Tensor,
    iq: IqType,
    n_out: usize,
    n_in: usize,
    m: usize,
) -> Result<Tensor> {
    w_bytes.apply_op1_no_bwd(&IqMatmulOp {
        x: x.clone(),
        iq,
        n_out,
        n_in,
        m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        #[allow(dead_code)]
        r#type: String,
        ggml_type_id: u32,
        block_size: usize,
        type_size: usize,
        n_blocks: usize,
        raw_hex: String,
        golden_f32: Vec<f32>,
    }

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Compare our port against gguf-py golden vectors generated from the real
    /// `gemma-4-12b-it-UD-IQ3_XXS.gguf` (`scripts/gen_iq_fixtures.py`).
    fn check(json: &str) {
        let f: Fixture = serde_json::from_str(json).unwrap();
        let iq = IqType::from_ggml_type_id(f.ggml_type_id).unwrap();
        assert_eq!(iq.block_size(), f.block_size);
        assert_eq!(iq.type_size(), f.type_size);

        let raw = hex_to_bytes(&f.raw_hex);
        assert_eq!(raw.len(), f.n_blocks * f.type_size);

        let mut out = vec![0f32; f.n_blocks * f.block_size];
        dequantize(iq, &raw, &mut out).unwrap();
        assert_eq!(out.len(), f.golden_f32.len());

        // f16→f32 is exact; the only divergence from gguf-py's numpy path is
        // f32 rounding order, well under 1e-5 for these magnitudes.
        let mut max_abs = 0f32;
        for (k, (a, b)) in out.iter().zip(&f.golden_f32).enumerate() {
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            assert!(
                diff <= 1e-5,
                "{} lane {k}: ours={a} golden={b} diff={diff}",
                iq.name()
            );
        }
        // Sanity: not all zeros (would mask a broken decode).
        assert!(out.iter().any(|v| *v != 0.0), "{} all-zero", iq.name());
        eprintln!("{} OK: max_abs_diff={max_abs:e}", iq.name());
    }

    #[test]
    fn iq2_xs_matches_ggml() {
        check(include_str!("testdata/iq2_xs.json"));
    }

    #[test]
    fn iq2_s_matches_ggml() {
        check(include_str!("testdata/iq2_s.json"));
    }

    #[test]
    fn iq3_xxs_matches_ggml() {
        check(include_str!("testdata/iq3_xxs.json"));
    }

    #[test]
    fn iq3_s_matches_ggml() {
        check(include_str!("testdata/iq3_s.json"));
    }

    #[test]
    fn iq4_xs_matches_ggml() {
        check(include_str!("testdata/iq4_xs.json"));
    }

    #[test]
    fn type_ids_round_trip() {
        for iq in [
            IqType::Iq2Xs,
            IqType::Iq2S,
            IqType::Iq3Xxs,
            IqType::Iq3S,
            IqType::Iq4Xs,
        ] {
            assert_eq!(IqType::from_ggml_type_id(iq.ggml_type_id()), Some(iq));
        }
        assert_eq!(IqType::from_ggml_type_id(12), None); // Q4_K is candle's
    }

    /// The fused `iq_matmul` (q8_1 MMVQ — the CUDA kernel on a CUDA device,
    /// the CPU port otherwise) must *approximate* `dequantize_iq` + a plain
    /// f32 matmul over the same weight. Both dequantize the SAME I-quant
    /// weight identically, so the only divergence is the int8 quantization of
    /// the activation — bounded by a few percent relative error. The IQ3_XXS
    /// fixture (1536 vals) is treated as a `[6, 256]` weight; X is `[m, 256]`.
    #[test]
    fn iq_matmul_matches_dequant_then_matmul() {
        use candle_core::{Device, Tensor};
        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(_) => Device::Cpu,
        };
        let f: Fixture = serde_json::from_str(include_str!("testdata/iq3_xxs.json")).unwrap();
        let iq = IqType::from_ggml_type_id(f.ggml_type_id).unwrap();
        let raw0 = hex_to_bytes(&f.raw_hex);
        let ts = iq.type_size();
        // Repeat valid IQ3_XXS blocks to reach the model's real widths.
        let tile = |n_blocks: usize| -> Vec<u8> {
            (0..n_blocks)
                .flat_map(|i| raw0[(i % f.n_blocks) * ts..(i % f.n_blocks + 1) * ts].to_vec())
                .collect()
        };

        // Single- and multi-super-block rows incl. the model's real n_in
        // (2048/4096 → nblk32 64/128) so the q8_1 MMVQ math is pinned to a
        // full-precision matmul at production sizes, not just one super-block.
        for &(n_out, n_in) in &[(6usize, 256usize), (3, 512), (2, 768), (1, 2048), (1, 4096)] {
            let raw = tile(n_out * n_in / QK_K);
            let nb = raw.len();
            let w_bytes = Tensor::from_vec(raw.clone(), (nb,), &dev).unwrap();
            let w = dequantize_iq(&w_bytes, iq, n_out * n_in)
                .unwrap()
                .reshape((n_out, n_in))
                .unwrap();

            for m in [1usize, 4, IQ_GEMV_MAX_M] {
                let xv: Vec<f32> = (0..m * n_in)
                    .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
                    .collect();
                let x = Tensor::from_vec(xv, (m, n_in), &dev).unwrap();

                let fused = iq_matmul(&w_bytes, &x, iq, n_out, n_in, m)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let reference = x
                    .matmul(&w.t().unwrap())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(fused.len(), m * n_out);

                // Relative L2 error: int8 activation quant, not exact equality.
                let mut num = 0f64;
                let mut den = 0f64;
                for (a, b) in fused.iter().zip(&reference) {
                    num += ((a - b) as f64).powi(2);
                    den += (*b as f64).powi(2);
                }
                let rel = (num / den.max(1e-12)).sqrt();
                assert!(den > 1e-6, "n_in={n_in} m={m}: reference ~zero, vacuous");
                assert!(
                    rel <= 0.05,
                    "n_in={n_in} m={m}: q8_1 MMVQ vs f32 matmul rel L2 {rel} > 5%"
                );
            }
        }
    }

    /// Per-32-block int8 (q8_1) activation quant must stay accurate even with
    /// Gemma-style outliers (a few huge values dwarfing the rest within a
    /// block): the per-block scale absorbs the outlier and the dot is
    /// dominated by it, so the relative error stays small (≤ a few %). Guards
    /// against the assumption that outliers would force a fallback to f32.
    #[test]
    fn q8_1_outlier_activation_diagnostic() {
        use candle_core::{Device, Tensor};
        let dev = Device::Cpu;
        let f: Fixture = serde_json::from_str(include_str!("testdata/iq3_xxs.json")).unwrap();
        let iq = IqType::from_ggml_type_id(f.ggml_type_id).unwrap();
        let raw = hex_to_bytes(&f.raw_hex);
        let (n_out, n_in) = (1usize, 1536usize);
        let w = dequantize_iq(
            &Tensor::from_vec(raw.clone(), (raw.len(),), &dev).unwrap(),
            iq,
            n_out * n_in,
        )
        .unwrap()
        .reshape((n_out, n_in))
        .unwrap();

        // One outlier per 32-block (~100×) over a small background — mimics
        // Gemma's massive-activation channels.
        let xv: Vec<f32> = (0..n_in)
            .map(|i| {
                if i % 32 == 0 {
                    120.0
                } else {
                    ((i % 7) as f32 - 3.0) * 0.2
                }
            })
            .collect();
        let x = Tensor::from_vec(xv.clone(), (1, n_in), &dev).unwrap();
        let fused = q8_1_mmvq_cpu(&raw, &xv, iq, n_out, n_in, 1);
        let reference = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in fused.iter().zip(&reference) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(
            rel <= 0.05,
            "q8_1 outlier-activation rel L2 {rel} > 5% (fused={fused:?} ref={reference:?})"
        );
    }
}
