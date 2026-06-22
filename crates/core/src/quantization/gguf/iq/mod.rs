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

        if self.n_elements % QK_K != 0 {
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
}
