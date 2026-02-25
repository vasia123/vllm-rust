//! State Space Dual (SSD) operations for Mamba2.
//!
//! Mamba2 reformulates the SSM as a "state space dual" model where `A` is a
//! scalar (rather than a matrix) per head.  The recurrence is:
//!
//! ```text
//!   decay_t  = exp(dt_t · A_h)                          [scalar per (batch, head, t)]
//!   h_t      = decay_t · h_{t-1}  +  dt_t · outer(x_t, B_t)
//!   y_t      = h_t · C_t  +  D_h · x_t
//! ```
//!
//! This module provides two equivalent implementations:
//!
//! * [`ssd_sequential`] — simple sequential loop; always correct, used as
//!   reference and for short sequences.
//! * [`ssd_chunk_scan`] — chunked scan that groups time-steps into blocks,
//!   enabling matrix-level parallelism within each chunk.  On CPU both produce
//!   the same numerics; the chunked structure maps cleanly to future CUDA
//!   kernel integration.
//!
//! ## Tensor layout
//!
//! All functions share the same layout convention (matching Mamba2 / HF):
//!
//! | Symbol  | Shape                              | Notes                       |
//! |---------|------------------------------------|-----------------------------|
//! | `x`     | `[B, L, H, P]`                    | P = head_dim after SiLU     |
//! | `dt`    | `[B, L, H]`                        | after softplus              |
//! | `a`     | `[H]`                              | negative scalar per head    |
//! | `b`     | `[B, L, G, N]`                    | G = n_groups, N = d_state   |
//! | `c`     | `[B, L, G, N]`                    |                             |
//! | `d`     | `[H]`                              | skip connection             |
//! | `state` | `[B, H, P, N]`                    | initial / carry state       |
//!
//! `heads_per_group = H / G` — how many heads share one (B, C) group.

use candle_core::{Result, Tensor};

// ─── Sequential SSD (reference) ──────────────────────────────────────────────

/// Sequential SSD recurrence for Mamba2 (all-heads-at-once).
///
/// Iterates over time steps; within each step all batch / head dimensions are
/// processed with vectorised tensor ops.  Correctness reference; prefer this
/// for short sequences or when no GPU kernel is available.
///
/// # Arguments
/// * `x`               — `[B, L, H, P]`  input after SiLU
/// * `dt`              — `[B, L, H]`     time-steps after softplus
/// * `a`               — `[H]`           per-head scalar decay (negative)
/// * `b`               — `[B, L, G, N]`  B projections (n_groups)
/// * `c`               — `[B, L, G, N]`  C projections (n_groups)
/// * `d`               — `[H]`           per-head skip-connection
/// * `state`           — `[B, H, P, N]`  initial SSM state
/// * `heads_per_group` — `H / G`
///
/// # Returns
/// `(output [B, L, H, P], new_state [B, H, P, N])`
#[allow(clippy::too_many_arguments)]
pub fn ssd_sequential(
    x: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    state: &Tensor,
    heads_per_group: usize,
) -> Result<(Tensor, Tensor)> {
    let x_dims = x.dims();
    let seq_len = x_dims[1];
    let b_dims = b.dims();
    let (n_groups, d_state) = (b_dims[2], b_dims[3]);

    let mut h = state.clone(); // [B, H, P, N]
    let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // ── Extract at time t ────────────────────────────────────────────────
        let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // [B, H, P]
        let dt_t = dt.narrow(1, t, 1)?.squeeze(1)?; // [B, H]
        let b_t = b.narrow(1, t, 1)?.squeeze(1)?; // [B, G, N]
        let c_t = c.narrow(1, t, 1)?.squeeze(1)?; // [B, G, N]

        // ── Decay: exp(dt_t · A_h) ───────────────────────────────────────────
        // a: [H] → broadcast with dt_t: [B, H]
        let da = dt_t.broadcast_mul(&a.unsqueeze(0)?)?; // [B, H]
        let decay = da.exp()?; // [B, H]

        // ── Expand B/C from n_groups → num_heads ────────────────────────────
        let b_exp = expand_groups(&b_t, n_groups, heads_per_group, d_state)?; // [B, H, N]
        let c_exp = expand_groups(&c_t, n_groups, heads_per_group, d_state)?; // [B, H, N]

        // ── State update: h = decay · h + dt_t · outer(x_t, B_exp) ─────────
        // outer(x_t, B_exp): [B, H, P] × [B, H, N] → [B, H, P, N]
        let dt_x = x_t.broadcast_mul(&dt_t.unsqueeze(2)?)?; // [B, H, P]
        let db = dt_x.unsqueeze(3)?.broadcast_mul(&b_exp.unsqueeze(2)?)?; // [B, H, P, N]

        let decay4d = decay.unsqueeze(2)?.unsqueeze(3)?; // [B, H, 1, 1]
        h = (h.broadcast_mul(&decay4d)? + &db)?; // [B, H, P, N]

        // ── Output: (h · C_exp).sum(-1) + D · x_t ───────────────────────────
        let hc = h.broadcast_mul(&c_exp.unsqueeze(2)?)?.sum(3)?; // [B, H, P]
        let dx = x_t.broadcast_mul(&d.unsqueeze(0)?.unsqueeze(2)?)?; // [B, H, P]
        let y_t = (&hc + &dx)?;

        outputs.push(y_t.unsqueeze(1)?); // [B, 1, H, P]
    }

    let output = Tensor::cat(&outputs, 1)?; // [B, L, H, P]
    Ok((output, h))
}

// ─── Chunked SSD scan ────────────────────────────────────────────────────────

/// Chunked SSD scan for Mamba2.
///
/// Partitions the sequence into chunks of `chunk_size` time-steps.  Within
/// each chunk the computation is expressed as two batched matrix operations:
///
/// 1. **State contribution** — how the carry state at the chunk boundary
///    decays into each output position.
/// 2. **Intra-chunk contribution** — lower-triangular matrix-vector multiply
///    using the causal decay matrix `L[t, s] = exp(Σ da[s+1..=t]) · dt[s]`.
///
/// The inter-chunk state is updated with a single matrix multiply after each
/// chunk.
///
/// On CPU this is numerically identical to [`ssd_sequential`] but structures
/// computation so that a future CUDA kernel can drop in with minimal interface
/// changes.
///
/// # Arguments
/// Same as [`ssd_sequential`] plus:
/// * `chunk_size` — number of time-steps per chunk (typically 64–256)
///
/// # Returns
/// `(output [B, L, H, P], new_state [B, H, P, N])`
#[allow(clippy::too_many_arguments)]
pub fn ssd_chunk_scan(
    x: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    state: &Tensor,
    chunk_size: usize,
    heads_per_group: usize,
) -> Result<(Tensor, Tensor)> {
    let x_dims = x.dims();
    let (batch, seq_len, num_heads, head_dim) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let _ = (num_heads, head_dim); // used via tensor ops
    let b_dims = b.dims();
    let (n_groups, d_state) = (b_dims[2], b_dims[3]);

    let mut h = state.clone(); // [B, H, P, N]
    let num_chunks = seq_len.div_ceil(chunk_size);
    let mut chunk_outputs: Vec<Tensor> = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let t_start = chunk_idx * chunk_size;
        let t_end = (t_start + chunk_size).min(seq_len);
        let cs = t_end - t_start; // actual chunk length (last chunk may be smaller)

        // Slice chunk tensors
        let x_c = x.narrow(1, t_start, cs)?; // [B, cs, H, P]
        let dt_c = dt.narrow(1, t_start, cs)?; // [B, cs, H]
        let b_c = b.narrow(1, t_start, cs)?; // [B, cs, G, N]
        let c_c = c.narrow(1, t_start, cs)?; // [B, cs, G, N]

        // da[b, t, h] = dt_c[b, t, h] * a[h]   [B, cs, H]
        let da = dt_c.broadcast_mul(&a.unsqueeze(0)?.unsqueeze(0)?)?;

        // Cumulative sum along the time dimension → decay factors
        // da_cumsum[b, t, h] = Σ_{s=0}^{t} da[b, s, h]
        let da_cumsum = cumsum_dim1(&da)?; // [B, cs, H]

        // Chunk-boundary decay: exp(da_cumsum[:, -1, :])  [B, H]
        let chunk_decay = da_cumsum.narrow(1, cs - 1, 1)?.squeeze(1)?.exp()?; // [B, H]

        // ── 1. State contribution ─────────────────────────────────────────────
        // For each position t in the chunk:
        //   y_state[b, t, h, i] = scale[b, t, h] · (h[b, h, i, :] · C[b, t, g, :])
        // where scale[b, t, h] = exp(da_cumsum[b, t, h])
        let scale = da_cumsum.exp()?; // [B, cs, H]

        // Expand C: [B, cs, G, N] → [B, cs, H, N]
        let c_exp = expand_groups_seq(&c_c, n_groups, heads_per_group, d_state, batch, cs)?;

        // h: [B, H, P, N] → [B, 1, H, P, N] for broadcast
        // c_exp: [B, cs, H, N] → [B, cs, H, 1, N]
        // h · c: sum over N → [B, cs, H, P]
        let h4 = h.unsqueeze(1)?; // [B, 1, H, P, N]
        let c5 = c_exp.unsqueeze(3)?; // [B, cs, H, 1, N]
        let hc_state = (h4.broadcast_mul(&c5)?).sum(4)?; // [B, cs, H, P]
                                                         // Scale by position decay
        let y_state = hc_state.broadcast_mul(&scale.unsqueeze(3)?)?; // [B, cs, H, P]

        // ── 2. Intra-chunk contribution ───────────────────────────────────────
        // For each position t: sum over s <= t of L[t,s] * x_c[s] * (B[s] · C[t])
        // L[t, s, h] = exp(da_cumsum[t, h] - da_cumsum[s, h]) * dt_c[s, h]  (lower-tri, s<=t)
        //
        // We compute this as a sequential mini-scan within the chunk, accumulating
        // an "inner state" that resets at the chunk start.
        let b_exp = expand_groups_seq(&b_c, n_groups, heads_per_group, d_state, batch, cs)?;

        // Sequential scan within the chunk using an inner state
        let dtype = x.dtype();
        let dev = x.device();
        let mut inner_h = Tensor::zeros((batch, num_heads, head_dim, d_state), dtype, dev)?;
        let mut intra_outputs: Vec<Tensor> = Vec::with_capacity(cs);

        for t in 0..cs {
            let x_t = x_c.narrow(1, t, 1)?.squeeze(1)?; // [B, H, P]
            let dt_t = dt_c.narrow(1, t, 1)?.squeeze(1)?; // [B, H]
            let b_t = b_exp.narrow(1, t, 1)?.squeeze(1)?; // [B, H, N]
            let c_t = c_exp.narrow(1, t, 1)?.squeeze(1)?; // [B, H, N]

            // da_t: use da (not da_cumsum) for inner state decay
            let da_t = da.narrow(1, t, 1)?.squeeze(1)?; // [B, H]
            let inner_decay = da_t.exp()?; // [B, H]
            let inner_decay4d = inner_decay.unsqueeze(2)?.unsqueeze(3)?;

            let dt_x = x_t.broadcast_mul(&dt_t.unsqueeze(2)?)?; // [B, H, P]
            let db = dt_x.unsqueeze(3)?.broadcast_mul(&b_t.unsqueeze(2)?)?; // [B, H, P, N]
            inner_h = (inner_h.broadcast_mul(&inner_decay4d)? + &db)?;

            let hc_inner = inner_h.broadcast_mul(&c_t.unsqueeze(2)?)?.sum(3)?; // [B, H, P]
            intra_outputs.push(hc_inner.unsqueeze(1)?);
        }

        let y_intra = Tensor::cat(&intra_outputs, 1)?; // [B, cs, H, P]

        // ── 3. Skip connection + combine ─────────────────────────────────────
        // d: [H] → [1, 1, H, 1] to broadcast with x_c: [B, cs, H, P]
        let d_skip = x_c.broadcast_mul(&d.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(3)?)?;
        let y_chunk = (&y_state + &y_intra + &d_skip)?; // [B, cs, H, P]
        chunk_outputs.push(y_chunk);

        // ── 4. Update carry state ─────────────────────────────────────────────
        // new_h = decay · h + Σ_s exp(da_cumsum[-1] - da_cumsum[s]) * dt[s] * outer(x[s], B[s])
        //
        // This equals: decay * h + inner_h_at_end * decay_factor_adjustment.
        // However, we already have inner_h computed with per-step decays.
        // The correct inter-chunk update is equivalent to running the full scan:
        // new_state = chunk_decay * h + intra_state_at_end_of_chunk
        //
        // NOTE: `inner_h` at this point holds the intra-chunk accumulated state
        // using the within-chunk cumulative decay (starting from 0). The inter-chunk
        // update must apply the full chunk decay to the carry, then add the intra part.
        let chunk_decay4d = chunk_decay.unsqueeze(2)?.unsqueeze(3)?;
        h = (h.broadcast_mul(&chunk_decay4d)? + &inner_h)?;
    }

    let output = Tensor::cat(&chunk_outputs, 1)?; // [B, L, H, P]
    Ok((output, h))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Cumulative sum along dim 1 (time dimension) using a sequential prefix sum.
///
/// Candle does not expose a direct `cumsum` op, so we implement it via a scan.
fn cumsum_dim1(x: &Tensor) -> Result<Tensor> {
    let seq_len = x.dims()[1];
    if seq_len == 0 {
        return Ok(x.clone());
    }
    let mut acc = x.narrow(1, 0, 1)?.clone();
    let mut slices = vec![acc.clone()];
    for t in 1..seq_len {
        let s = x.narrow(1, t, 1)?;
        acc = (&acc + &s)?;
        slices.push(acc.clone());
    }
    Tensor::cat(&slices, 1)
}

/// Expand B or C from `[B, G, N]` → `[B, H, N]` by repeating each group
/// `heads_per_group` times.
fn expand_groups(
    t: &Tensor,
    n_groups: usize,
    heads_per_group: usize,
    d_state: usize,
) -> Result<Tensor> {
    let batch = t.dims()[0];
    if heads_per_group == 1 {
        return Ok(t.clone());
    }
    // [B, G, N] → [B, G, 1, N] → [B, G, HPG, N] → [B, H, N]
    t.unsqueeze(2)?
        .broadcast_as((batch, n_groups, heads_per_group, d_state))?
        .contiguous()?
        .reshape((batch, n_groups * heads_per_group, d_state))
}

/// Expand B or C from `[B, L, G, N]` → `[B, L, H, N]`.
fn expand_groups_seq(
    t: &Tensor,
    n_groups: usize,
    heads_per_group: usize,
    d_state: usize,
    batch: usize,
    seq_len: usize,
) -> Result<Tensor> {
    if heads_per_group == 1 {
        return Ok(t.clone());
    }
    t.unsqueeze(3)?
        .broadcast_as((batch, seq_len, n_groups, heads_per_group, d_state))?
        .contiguous()?
        .reshape((batch, seq_len, n_groups * heads_per_group, d_state))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const BATCH: usize = 1;
    const SEQ: usize = 8;
    const NUM_HEADS: usize = 4;
    const HEAD_DIM: usize = 3;
    const N_GROUPS: usize = 2;
    const D_STATE: usize = 4;
    const HPG: usize = 2; // heads_per_group = NUM_HEADS / N_GROUPS

    fn make_inputs(device: &Device) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let dtype = DType::F32;
        let x = Tensor::randn(0f32, 0.1, (BATCH, SEQ, NUM_HEADS, HEAD_DIM), device).unwrap();
        let dt = Tensor::ones((BATCH, SEQ, NUM_HEADS), dtype, device).unwrap() * 0.1;
        let dt = dt.unwrap();
        // A negative — standard Mamba2 init
        let a = (Tensor::ones((NUM_HEADS,), dtype, device).unwrap() * -1.0).unwrap();
        let b = Tensor::randn(0f32, 0.1, (BATCH, SEQ, N_GROUPS, D_STATE), device).unwrap();
        let c = Tensor::randn(0f32, 0.1, (BATCH, SEQ, N_GROUPS, D_STATE), device).unwrap();
        let d = Tensor::zeros((NUM_HEADS,), dtype, device).unwrap();
        let state = Tensor::zeros((BATCH, NUM_HEADS, HEAD_DIM, D_STATE), dtype, device).unwrap();
        (x, dt, a, b, c, d, state)
    }

    #[test]
    fn ssd_sequential_output_shape() {
        let dev = Device::Cpu;
        let (x, dt, a, b, c, d, state) = make_inputs(&dev);
        let (out, new_state) = ssd_sequential(&x, &dt, &a, &b, &c, &d, &state, HPG).unwrap();
        assert_eq!(out.dims(), [BATCH, SEQ, NUM_HEADS, HEAD_DIM]);
        assert_eq!(new_state.dims(), [BATCH, NUM_HEADS, HEAD_DIM, D_STATE]);
    }

    #[test]
    fn ssd_chunk_scan_output_shape() {
        let dev = Device::Cpu;
        let (x, dt, a, b, c, d, state) = make_inputs(&dev);
        let (out, new_state) = ssd_chunk_scan(&x, &dt, &a, &b, &c, &d, &state, 4, HPG).unwrap();
        assert_eq!(out.dims(), [BATCH, SEQ, NUM_HEADS, HEAD_DIM]);
        assert_eq!(new_state.dims(), [BATCH, NUM_HEADS, HEAD_DIM, D_STATE]);
    }

    /// Sequential and chunked scans must agree within floating-point tolerance.
    #[test]
    fn sequential_chunked_agree() {
        let dev = Device::Cpu;
        let (x, dt, a, b, c, d, state) = make_inputs(&dev);
        let (out_seq, state_seq) = ssd_sequential(&x, &dt, &a, &b, &c, &d, &state, HPG).unwrap();
        let (out_chk, state_chk) = ssd_chunk_scan(&x, &dt, &a, &b, &c, &d, &state, 4, HPG).unwrap();

        let out_diff: f32 = ((&out_seq - &out_chk).unwrap())
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            out_diff < 1e-5,
            "output mismatch between sequential and chunked: max diff = {}",
            out_diff
        );

        let state_diff: f32 = ((&state_seq - &state_chk).unwrap())
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            state_diff < 1e-5,
            "state mismatch: max diff = {}",
            state_diff
        );
    }

    /// With A = 0 the decay is always 1, so the output at step t is the
    /// sum of all previous dt * x * B contributions, projected through C, plus D*x.
    #[test]
    fn ssd_a_zero_accumulation() {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let seq = 2;
        let h = 1;
        let p = 2;
        let g = 1;
        let n = 2;
        // A = 0 → no decay
        let a = Tensor::zeros((h,), dtype, &dev).unwrap();
        let x = Tensor::ones((1, seq, h, p), dtype, &dev).unwrap();
        let dt = Tensor::ones((1, seq, h), dtype, &dev).unwrap();
        let b = Tensor::ones((1, seq, g, n), dtype, &dev).unwrap();
        let c = Tensor::ones((1, seq, g, n), dtype, &dev).unwrap();
        let d = Tensor::zeros((h,), dtype, &dev).unwrap();
        let state = Tensor::zeros((1, h, p, n), dtype, &dev).unwrap();

        let (out, _) = ssd_sequential(&x, &dt, &a, &b, &c, &d, &state, 1).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // t=0: h = 1*outer([1,1],[1,1]) = [[1,1],[1,1]]
        //       y = h·C = [2,2]  (D=0 skip)
        // t=1: h += 1*outer([1,1],[1,1]) → [[2,2],[2,2]]
        //       y = [4,4]
        for &v in &vals[..p] {
            assert!((v - n as f32).abs() < 1e-5, "t=0 expected {}, got {}", n, v);
        }
        for &v in &vals[p..] {
            assert!(
                (v - 2.0 * n as f32).abs() < 1e-5,
                "t=1 expected {}, got {}",
                2 * n,
                v
            );
        }
    }

    /// State is propagated correctly across multiple calls (decode-like pattern).
    #[test]
    fn ssd_state_carry() {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let (x, dt, a, b, c, d, state) = make_inputs(&dev);

        // Run half the sequence at a time and carry state
        let x1 = x.narrow(1, 0, SEQ / 2).unwrap();
        let dt1 = dt.narrow(1, 0, SEQ / 2).unwrap();
        let b1 = b.narrow(1, 0, SEQ / 2).unwrap();
        let c1 = c.narrow(1, 0, SEQ / 2).unwrap();
        let (out1, state1) = ssd_sequential(&x1, &dt1, &a, &b1, &c1, &d, &state, HPG).unwrap();

        let x2 = x.narrow(1, SEQ / 2, SEQ / 2).unwrap();
        let dt2 = dt.narrow(1, SEQ / 2, SEQ / 2).unwrap();
        let b2 = b.narrow(1, SEQ / 2, SEQ / 2).unwrap();
        let c2 = c.narrow(1, SEQ / 2, SEQ / 2).unwrap();
        let (out2, _) = ssd_sequential(&x2, &dt2, &a, &b2, &c2, &d, &state1, HPG).unwrap();

        // Full run
        let (out_full, _) = ssd_sequential(&x, &dt, &a, &b, &c, &d, &state, HPG).unwrap();

        // Concatenate half outputs and compare
        let out_concat = Tensor::cat(&[&out1, &out2], 1).unwrap();
        let diff: f32 = ((&out_concat - &out_full).unwrap())
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-5, "chunked vs full mismatch: {}", diff);
    }

    /// Chunk boundary at seq_len not divisible by chunk_size.
    #[test]
    fn ssd_chunk_unaligned_length() {
        let dev = Device::Cpu;
        let (x, dt, a, b, c, d, state) = make_inputs(&dev);
        // chunk_size = 3, seq_len = 8 → chunks of [3, 3, 2]
        let (out, _) = ssd_chunk_scan(&x, &dt, &a, &b, &c, &d, &state, 3, HPG).unwrap();
        assert_eq!(out.dims(), [BATCH, SEQ, NUM_HEADS, HEAD_DIM]);
    }
}
