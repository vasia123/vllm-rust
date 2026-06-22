//! GPU correctness gate for `kernels/iq_dequant.cu`: the CUDA dequant must
//! match the CPU port (itself pinned against gguf-py golden vectors), and the
//! CUDA q8_1 MMVQ kernels must match the CPU MMVQ reference bit-for-bit
//! (modulo f32 reduction order).

use super::{dequantize_iq, iq_matmul, q8_1_mmvq_cpu, IqType, QK_K};
use candle_core::{Device, Tensor};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    ggml_type_id: u32,
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

fn gpu_matches_cpu(json: &str) {
    let cuda = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skip: no CUDA device");
            return;
        }
    };
    let f: Fixture = serde_json::from_str(json).unwrap();
    let iq = IqType::from_ggml_type_id(f.ggml_type_id).unwrap();
    let raw = hex_to_bytes(&f.raw_hex);
    assert_eq!(raw.len(), f.n_blocks * f.type_size);
    let n_elements = f.n_blocks * QK_K;

    let bytes_gpu = Tensor::from_vec(raw, (f.n_blocks * f.type_size,), &cuda).unwrap();
    let deq = dequantize_iq(&bytes_gpu, iq, n_elements)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // GPU uses --use_fast_math; allow a small tolerance vs the golden.
    let mut max_abs = 0f32;
    for (k, (a, b)) in deq.iter().zip(&f.golden_f32).enumerate() {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        assert!(diff <= 1e-4, "{} lane {k}: gpu={a} golden={b}", iq.name());
    }
    eprintln!("{} GPU OK: max_abs_diff={max_abs:e}", iq.name());
}

/// The CUDA `mmvq_*` kernel must match the CPU `q8_1_mmvq_cpu` reference: both
/// run the identical integer algorithm (same q8_1 quants, same `__dp4a`
/// dots), so the only divergence is f32 reduction order — well under 1e-3.
/// The fixture blocks are treated as a `[n_blocks, 256]` weight; X is
/// `[m, 256]`. This is the exact-port gate for the decode hot path.
fn mmvq_gpu_matches_cpu(json: &str) {
    let cuda = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skip: no CUDA device");
            return;
        }
    };
    let f: Fixture = serde_json::from_str(json).unwrap();
    let iq = IqType::from_ggml_type_id(f.ggml_type_id).unwrap();
    let raw = hex_to_bytes(&f.raw_hex);
    assert_eq!(raw.len(), f.n_blocks * f.type_size);
    let (n_out, n_in) = (f.n_blocks, QK_K);

    let w_gpu = Tensor::from_vec(raw.clone(), (raw.len(),), &cuda).unwrap();

    for m in [1usize, 4, super::IQ_GEMV_MAX_M] {
        let xv: Vec<f32> = (0..m * n_in)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
            .collect();
        let x = Tensor::from_vec(xv.clone(), (m, n_in), &cuda).unwrap();

        let gpu = iq_matmul(&w_gpu, &x, iq, n_out, n_in, m)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cpu = q8_1_mmvq_cpu(&raw, &xv, iq, n_out, n_in, m);
        assert_eq!(gpu.len(), cpu.len());

        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in gpu.iter().zip(&cpu) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(den > 1e-6, "{} m={m}: cpu reference ~zero", iq.name());
        assert!(
            rel <= 1e-3,
            "{} m={m}: GPU MMVQ vs CPU MMVQ rel L2 {rel}",
            iq.name()
        );
    }
    eprintln!("{} GPU MMVQ == CPU MMVQ OK", iq.name());
}

/// Multi-super-block stride gate: the model's `n_in` is far larger than one
/// QK_K super-block, and rows wider than 1024 make a warp lane process more
/// than one 32-element sub-block (`nblk32 > 32`). The 1536-value IQ3_XXS
/// fixture is reshaped to cover nb256 = 1/2/3/6 (the last makes nblk32 = 48,
/// i.e. lanes that loop twice). GPU MMVQ must still match the CPU reference.
#[test]
fn iq3_xxs_mmvq_multi_superblock_gpu_matches_cpu() {
    let cuda = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skip: no CUDA device");
            return;
        }
    };
    let f: Fixture = serde_json::from_str(include_str!("testdata/iq3_xxs.json")).unwrap();
    let iq = IqType::from_ggml_type_id(f.ggml_type_id).unwrap();
    let raw = hex_to_bytes(&f.raw_hex);
    let ts = iq.type_size();

    // The 6-block fixture reshaped (nb256 1/2/3/6), plus tiled to the model's
    // real widths n_in 2048/4096 (nb256 8/16 → nblk32 64/128, i.e. a warp lane
    // looping 2× / 4×). Tiling repeats valid IQ3_XXS blocks, so the bytes stay
    // decodable.
    let tile = |n_blocks: usize| -> Vec<u8> {
        (0..n_blocks)
            .flat_map(|i| raw[(i % f.n_blocks) * ts..(i % f.n_blocks + 1) * ts].to_vec())
            .collect()
    };

    for &(n_out, n_in) in &[
        (6usize, 256usize),
        (3, 512),
        (2, 768),
        (1, 1536),
        (1, 2048),
        (1, 4096),
    ] {
        let n_blocks = n_out * n_in / QK_K;
        let raw = tile(n_blocks);
        let w_gpu = Tensor::from_vec(raw.clone(), (raw.len(),), &cuda).unwrap();
        for m in [1usize, 4, super::IQ_GEMV_MAX_M] {
            let xv: Vec<f32> = (0..m * n_in)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
                .collect();
            let x = Tensor::from_vec(xv.clone(), (m, n_in), &cuda).unwrap();
            let gpu = iq_matmul(&w_gpu, &x, iq, n_out, n_in, m)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let cpu = q8_1_mmvq_cpu(&raw, &xv, iq, n_out, n_in, m);
            let mut num = 0f64;
            let mut den = 0f64;
            for (a, b) in gpu.iter().zip(&cpu) {
                num += ((a - b) as f64).powi(2);
                den += (*b as f64).powi(2);
            }
            let rel = (num / den.max(1e-12)).sqrt();
            assert!(den > 1e-6, "n_in={n_in} m={m}: cpu ~zero");
            assert!(
                rel <= 1e-3,
                "n_in={n_in} m={m}: GPU vs CPU MMVQ rel L2 {rel}"
            );
        }
    }
    eprintln!("IQ3_XXS multi-super-block GPU MMVQ == CPU OK");
}

#[test]
fn iq2_xs_mmvq_gpu_matches_cpu() {
    mmvq_gpu_matches_cpu(include_str!("testdata/iq2_xs.json"));
}
#[test]
fn iq2_s_mmvq_gpu_matches_cpu() {
    mmvq_gpu_matches_cpu(include_str!("testdata/iq2_s.json"));
}
#[test]
fn iq3_xxs_mmvq_gpu_matches_cpu() {
    mmvq_gpu_matches_cpu(include_str!("testdata/iq3_xxs.json"));
}
#[test]
fn iq3_s_mmvq_gpu_matches_cpu() {
    mmvq_gpu_matches_cpu(include_str!("testdata/iq3_s.json"));
}
#[test]
fn iq4_xs_mmvq_gpu_matches_cpu() {
    mmvq_gpu_matches_cpu(include_str!("testdata/iq4_xs.json"));
}

#[test]
fn iq2_xs_gpu_matches_cpu() {
    gpu_matches_cpu(include_str!("testdata/iq2_xs.json"));
}
#[test]
fn iq2_s_gpu_matches_cpu() {
    gpu_matches_cpu(include_str!("testdata/iq2_s.json"));
}
#[test]
fn iq3_xxs_gpu_matches_cpu() {
    gpu_matches_cpu(include_str!("testdata/iq3_xxs.json"));
}
#[test]
fn iq3_s_gpu_matches_cpu() {
    gpu_matches_cpu(include_str!("testdata/iq3_s.json"));
}
#[test]
fn iq4_xs_gpu_matches_cpu() {
    gpu_matches_cpu(include_str!("testdata/iq4_xs.json"));
}
