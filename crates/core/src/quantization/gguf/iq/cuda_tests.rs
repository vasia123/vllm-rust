//! GPU correctness gate for `kernels/iq_dequant.cu`: the CUDA dequant must
//! match the CPU port (itself pinned against gguf-py golden vectors).

use super::{dequantize_iq, IqType, QK_K};
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
