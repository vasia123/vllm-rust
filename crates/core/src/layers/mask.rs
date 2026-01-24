use candle_core::{DType, Device, Result, Tensor};

/// Generate a causal attention mask for decoder-only models.
/// Returns shape [1, 1, seq_len, seq_len + seqlen_offset].
pub fn causal_mask(
    seq_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_len = seq_len + seqlen_offset;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                if j > i + seqlen_offset {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?;
    mask.to_dtype(dtype)
}
