//! EPLB expert weight rearrangement via collective all-to-all.
//!
//! When [`EplbState::should_rebalance`] triggers, this module performs the actual
//! weight migration across EP ranks so the new logical-to-physical expert assignment
//! takes effect.
//!
//! ## Algorithm
//!
//! Each rank owns `num_local_experts` physical slots. Both the old and new placement
//! maps are known globally: `placement[rank * num_local + slot] = logical_expert_id`.
//!
//! For each weight tensor, the transfer proceeds in three steps:
//!
//! 1. **Routing**: compute which of my old experts other ranks need and which of my new
//!    experts come from remote ranks (pure local computation from the placement maps).
//! 2. **Transfer**: pack send rows into a flat buffer and exchange via `all_to_all_v`.
//!    Every rank participates in the collective even when it has nothing to send.
//! 3. **Assembly**: rebuild the weight tensor slot-by-slot, taking each row from either
//!    the local old tensor or the received buffer.
//!
//! ## Reference
//!
//! `reference/vllm/vllm/distributed/eplb/rebalance_execute.py::rearrange_expert_weights_inplace`
//! (Python uses `batch_isend_irecv`; we use `all_to_all_v` which is semantically
//! equivalent and maps more naturally onto the existing `DeviceCommunicator` trait.)

use std::collections::HashMap;

use candle_core::{Result, Tensor};

use crate::distributed::DeviceCommunicator;

/// Global expert placement for all layers:
/// `placement[layer][rank * num_local + slot] = logical_expert_id`.
pub type LayerExpertPlacement = Vec<Vec<usize>>;

/// Rearrange expert weights in-place across all EP ranks.
///
/// After this call `expert_weights[layer][weight_idx]` holds the experts
/// dictated by `new_placement` rather than `old_placement`.
///
/// # Arguments
///
/// * `old_placement` — current placement: `[layer_idx][phys_slot]` = logical_id.
///   Length `ep_size * num_local_experts`.
/// * `new_placement` — desired placement after rebalancing, same shape.
/// * `expert_weights` — `[layer_idx][weight_tensor_idx]`, each tensor has shape
///   `[num_local_experts, D1, D2, ...]`.  Modified in-place by replacing each
///   tensor with the rearranged version.
/// * `ep_rank` — this rank's index in the expert-parallel group.
/// * `ep_size` — total number of expert-parallel ranks.
/// * `comm` — collective communicator for the EP group.
///
/// # Panics (debug only)
///
/// Panics when placement lengths are inconsistent with `ep_size * num_local_experts`
/// or when layer counts disagree.
pub fn rearrange_expert_weights_inplace(
    old_placement: &LayerExpertPlacement,
    new_placement: &LayerExpertPlacement,
    expert_weights: &mut [Vec<Tensor>],
    ep_rank: usize,
    ep_size: usize,
    comm: &dyn DeviceCommunicator,
) -> Result<()> {
    let num_layers = old_placement.len();
    if num_layers == 0 || ep_size == 0 {
        return Ok(());
    }

    debug_assert_eq!(
        num_layers,
        new_placement.len(),
        "placement layer count mismatch"
    );
    debug_assert_eq!(
        num_layers,
        expert_weights.len(),
        "weight layer count mismatch"
    );

    let total_physical = old_placement[0].len();
    debug_assert_eq!(
        total_physical % ep_size,
        0,
        "total_physical must be divisible by ep_size"
    );
    let num_local = total_physical / ep_size;

    for layer_idx in 0..num_layers {
        let old = &old_placement[layer_idx];
        let new = &new_placement[layer_idx];

        for w in expert_weights[layer_idx].iter_mut() {
            // Clone here to avoid borrow conflict; the tensor itself is reference-counted
            // so clone is cheap (increments the Arc refcount).
            let old_w = w.clone();
            *w = rearrange_weight_tensor(&old_w, old, new, ep_rank, ep_size, num_local, comm)?;
        }
    }

    Ok(())
}

/// Rearrange a single weight tensor across EP ranks.
///
/// Returns the new weight tensor for `ep_rank` after the rearrangement.
fn rearrange_weight_tensor(
    weight: &Tensor,
    old_placement: &[usize], // length ep_size * num_local
    new_placement: &[usize],
    ep_rank: usize,
    ep_size: usize,
    num_local: usize,
    comm: &dyn DeviceCommunicator,
) -> Result<Tensor> {
    debug_assert_eq!(
        weight.dim(0)?,
        num_local,
        "weight first dim must equal num_local"
    );

    // ── Step 1: build routing tables ──────────────────────────────────────────

    // expert_location[logical_id] = (rank, local_slot) — tells us who currently holds each expert.
    let expert_location: HashMap<usize, (usize, usize)> = old_placement
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, (i / num_local, i % num_local)))
        .collect();

    // Desired experts for each of my new slots.
    let my_new_experts: Vec<usize> = (0..num_local)
        .map(|s| new_placement[ep_rank * num_local + s])
        .collect();

    // send_order[r] = my old local slot indices to send to rank r.
    // Canonical order: iterate rank r's new slots 0..num_local; include those sourced from me.
    // Both the sender and receiver compute this in the same order, so they agree without
    // extra coordination.
    let mut send_order: Vec<Vec<usize>> = vec![Vec::new(); ep_size];
    for r in 0..ep_size {
        if r == ep_rank {
            continue; // never send to self via the collective
        }
        for s in 0..num_local {
            let needed = new_placement[r * num_local + s];
            if let Some(&(src, src_local)) = expert_location.get(&needed) {
                if src == ep_rank {
                    send_order[r].push(src_local);
                }
            }
        }
    }

    // recv_order[r] = my new slot indices that come from rank r.
    // Canonical order: iterate my new slots 0..num_local; include those sourced from r.
    let mut recv_order: Vec<Vec<usize>> = vec![Vec::new(); ep_size];
    for (s, &needed) in my_new_experts.iter().enumerate() {
        if let Some(&(src, _)) = expert_location.get(&needed) {
            if src != ep_rank {
                recv_order[src].push(s);
            }
        }
    }

    // ── Step 2: pack send buffer and exchange via all_to_all_v ────────────────

    // Flatten the weight tensor so all_to_all_v works element-wise.
    // row_elems = product of all dims except the first.
    let orig_shape = weight.dims().to_vec();
    let flat = weight.flatten_all()?;
    let row_elems = if num_local > 0 {
        flat.elem_count() / num_local
    } else {
        0
    };

    // splits are in elements (scalar values), not rows
    let send_splits: Vec<usize> = (0..ep_size)
        .map(|r| send_order[r].len() * row_elems)
        .collect();
    let recv_splits: Vec<usize> = (0..ep_size)
        .map(|r| recv_order[r].len() * row_elems)
        .collect();

    let total_send_elems: usize = send_splits.iter().sum();

    // Build the packed send buffer: rows ordered by destination rank.
    let send_buffer = if total_send_elems > 0 {
        let mut pieces: Vec<Tensor> = Vec::new();
        for rank_slots in &send_order {
            for &slot in rank_slots {
                pieces.push(flat.narrow(0, slot * row_elems, row_elems)?);
            }
        }
        Tensor::cat(&pieces, 0)?
    } else {
        // Nothing to send; still need to participate in the collective.
        Tensor::zeros(&[0], weight.dtype(), weight.device())?
    };

    // All ranks must call all_to_all_v together (it is a collective op).
    // For ep_size == 1 there is no communication — skip the call and use an
    // empty buffer (all sources are guaranteed to be local when ep_size == 1).
    let recv_flat = if ep_size > 1 {
        comm.all_to_all_v(&send_buffer, &send_splits, &recv_splits)
            .map_err(candle_core::Error::from)?
    } else {
        Tensor::zeros(&[0], weight.dtype(), weight.device())?
    };

    // Pre-compute byte offsets into recv_flat for each source rank.
    let mut recv_base_elems = vec![0usize; ep_size];
    {
        let mut off = 0usize;
        for r in 0..ep_size {
            recv_base_elems[r] = off;
            off += recv_splits[r];
        }
    }

    // ── Step 3: assemble the rearranged weight tensor ─────────────────────────

    let mut recv_cursor = vec![0usize; ep_size]; // current element offset within each rank's recv data
    let mut new_rows: Vec<Tensor> = Vec::with_capacity(num_local);

    for &needed in &my_new_experts {
        let &(src_rank, src_local) = expert_location.get(&needed).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "EPLB: expert {needed} not found in old placement \
                 (ep_rank={ep_rank}, ep_size={ep_size})"
            ))
        })?;

        let row = if src_rank == ep_rank {
            // Local: take from the flat old-weight tensor.
            flat.narrow(0, src_local * row_elems, row_elems)?
        } else {
            // Remote: take from the received flat buffer.
            let elem_off = recv_base_elems[src_rank] + recv_cursor[src_rank];
            recv_cursor[src_rank] += row_elems;
            recv_flat.narrow(0, elem_off, row_elems)?
        };
        new_rows.push(row);
    }

    let new_flat = Tensor::cat(&new_rows, 0)?;
    new_flat.reshape(orig_shape.as_slice())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier, Mutex};

    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::distributed::{DistributedError, LocalProcessGroup, ProcessGroup, ReduceOp};

    type DistResult<T> = std::result::Result<T, DistributedError>;

    // ─── Simulated multi-rank communicator ───────────────────────────────────

    /// Shared state for a group of `SimComm` instances running in separate threads.
    struct SimState {
        ep_size: usize,
        // Submissions slot: each rank stores (send_flat, send_splits) before the barrier.
        sends: Mutex<Vec<Option<(Tensor, Vec<usize>)>>>,
        /// Reusable barrier — automatically resets after `ep_size` threads wait.
        /// Used twice per `all_to_all_v` call (submit phase + read phase).
        barrier: Barrier,
    }

    impl SimState {
        fn new(ep_size: usize) -> Arc<Self> {
            Arc::new(Self {
                ep_size,
                sends: Mutex::new(vec![None; ep_size]),
                barrier: Barrier::new(ep_size),
            })
        }
    }

    struct SimComm {
        rank: usize,
        pg: LocalProcessGroup,
        state: Arc<SimState>,
    }

    impl SimComm {
        /// Create a group of `ep_size` communicators that exchange data in-process.
        fn group(ep_size: usize) -> Vec<SimComm> {
            let state = SimState::new(ep_size);
            (0..ep_size)
                .map(|rank| SimComm {
                    rank,
                    pg: LocalProcessGroup::with_rank(rank, ep_size),
                    state: Arc::clone(&state),
                })
                .collect()
        }
    }

    impl crate::distributed::DeviceCommunicator for SimComm {
        fn process_group(&self) -> &dyn ProcessGroup {
            &self.pg
        }

        /// Simulated all_to_all_v: each rank stores its send buffer behind a barrier,
        /// then reads from other ranks' buffers.
        fn all_to_all_v(
            &self,
            tensor: &Tensor,
            send_splits: &[usize],
            recv_splits: &[usize],
        ) -> DistResult<Tensor> {
            // Phase 1 — submit
            {
                let mut sends = self.state.sends.lock().unwrap();
                sends[self.rank] = Some((tensor.clone(), send_splits.to_vec()));
            }
            self.state.barrier.wait(); // all ranks have submitted

            // Phase 2 — read
            let recv = {
                let sends = self.state.sends.lock().unwrap();
                let mut pieces: Vec<Tensor> = Vec::new();
                for r in 0..self.state.ep_size {
                    let (buf, other_splits) = sends[r].as_ref().unwrap();
                    let count = other_splits[self.rank]; // elements rank r sends to me
                    if count > 0 {
                        let offset: usize = other_splits[..self.rank].iter().sum();
                        pieces.push(buf.narrow(0, offset, count)?);
                    }
                }
                if pieces.is_empty() {
                    Tensor::zeros(&[0], tensor.dtype(), tensor.device())?
                } else {
                    Tensor::cat(&pieces, 0)?
                }
            };
            self.state.barrier.wait(); // all ranks have read — safe to clear

            // Phase 3 — clear this rank's slot so the next call starts clean
            {
                let mut sends = self.state.sends.lock().unwrap();
                sends[self.rank] = None;
            }

            Ok(recv)
        }

        // The remaining methods are unused in these tests; provide no-op impls.
        fn all_reduce(&self, t: &Tensor, _: ReduceOp) -> DistResult<Tensor> {
            Ok(t.clone())
        }
        fn all_gather(&self, t: &Tensor, _: usize) -> DistResult<Tensor> {
            Ok(t.clone())
        }
        fn reduce_scatter(&self, t: &Tensor, _: usize, _: ReduceOp) -> DistResult<Tensor> {
            Ok(t.clone())
        }
        fn broadcast(&self, t: &Tensor, _: usize) -> DistResult<Tensor> {
            Ok(t.clone())
        }
        fn send(&self, _: &Tensor, _: usize) -> DistResult<()> {
            Ok(())
        }
        fn recv(&self, s: &[usize], dt: DType, _: usize) -> DistResult<Tensor> {
            Ok(Tensor::zeros(s, dt, &Device::Cpu)?)
        }
        fn barrier(&self) -> DistResult<()> {
            Ok(())
        }
        fn all_to_all(&self, t: &Tensor) -> DistResult<Tensor> {
            Ok(t.clone())
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /// Build a [num_local, hidden] weight tensor from a flat row-major slice.
    fn make_weights(num_local: usize, hidden: usize, vals: &[f32]) -> Tensor {
        assert_eq!(vals.len(), num_local * hidden);
        Tensor::from_vec(vals.to_vec(), &[num_local, hidden], &Device::Cpu).unwrap()
    }

    fn as_vec(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    // ─── Test cases ──────────────────────────────────────────────────────────

    #[test]
    fn identity_placement_no_change() {
        // ep_size=1, old == new: the weight tensor must be unchanged.
        let ep_size = 1;
        let num_local = 3;
        let hidden = 4;
        let old = vec![vec![0usize, 1, 2]];
        let new_p = vec![vec![0usize, 1, 2]];
        let vals: Vec<f32> = (0..num_local * hidden).map(|x| x as f32).collect();
        let mut weights = vec![vec![make_weights(num_local, hidden, &vals)]];

        let comms = SimComm::group(ep_size);
        rearrange_expert_weights_inplace(&old, &new_p, &mut weights, 0, ep_size, &comms[0])
            .unwrap();

        assert_eq!(
            as_vec(&weights[0][0]),
            vals,
            "identity placement must not modify weights"
        );
    }

    #[test]
    fn local_reorder_single_rank() {
        // ep_size=1, reorder experts locally: [0,1,2] → [2,0,1].
        // Expert 0 = [0,1], expert 1 = [2,3], expert 2 = [4,5]
        let ep_size = 1;
        let old = vec![vec![0usize, 1, 2]];
        let new_p = vec![vec![2usize, 0, 1]]; // slot 0←expert 2, slot 1←expert 0, slot 2←expert 1
        let vals = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut weights = vec![vec![make_weights(3, 2, &vals)]];

        let comms = SimComm::group(ep_size);
        rearrange_expert_weights_inplace(&old, &new_p, &mut weights, 0, ep_size, &comms[0])
            .unwrap();

        // Expected: [expert 2 | expert 0 | expert 1] = [4,5, 0,1, 2,3]
        assert_eq!(
            as_vec(&weights[0][0]),
            vec![4.0f32, 5.0, 0.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn two_ranks_swap_one_expert() {
        // ep_size=2, num_local=2.
        // Old: rank 0 → {expert 0, expert 1}, rank 1 → {expert 2, expert 3}
        // New: rank 0 → {expert 0, expert 2}, rank 1 → {expert 1, expert 3}
        //      rank 0 sends expert 1 to rank 1, receives expert 2 from rank 1.
        let ep_size = 2;
        // Expert 0 = [0,1], expert 1 = [2,3], expert 2 = [4,5], expert 3 = [6,7]
        let old = vec![vec![0usize, 1, 2, 3]];
        let new_p = vec![vec![0usize, 2, 1, 3]];
        let vals0 = vec![0.0f32, 1.0, 2.0, 3.0]; // rank 0
        let vals1 = vec![4.0f32, 5.0, 6.0, 7.0]; // rank 1
        let mut weights0 = vec![vec![make_weights(2, 2, &vals0)]];
        let mut weights1 = vec![vec![make_weights(2, 2, &vals1)]];

        let comms = SimComm::group(ep_size);
        let mut it = comms.into_iter();
        let comm0 = it.next().unwrap();
        let comm1 = it.next().unwrap();
        let old1 = old.clone();
        let new1 = new_p.clone();

        let h = std::thread::spawn(move || {
            rearrange_expert_weights_inplace(&old1, &new1, &mut weights1, 1, ep_size, &comm1)
                .unwrap();
            weights1
        });

        rearrange_expert_weights_inplace(&old, &new_p, &mut weights0, 0, ep_size, &comm0).unwrap();
        let weights1 = h.join().unwrap();

        // rank 0: slot 0 ← expert 0 = [0,1], slot 1 ← expert 2 = [4,5]
        assert_eq!(as_vec(&weights0[0][0]), vec![0.0f32, 1.0, 4.0, 5.0]);
        // rank 1: slot 0 ← expert 1 = [2,3], slot 1 ← expert 3 = [6,7]
        assert_eq!(as_vec(&weights1[0][0]), vec![2.0f32, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn two_ranks_identity_no_communication() {
        // ep_size=2, old == new: no data should move.
        let ep_size = 2;
        let old = vec![vec![0usize, 1, 2, 3]];
        let new_p = old.clone();
        let vals0 = vec![0.0f32, 1.0, 2.0, 3.0];
        let vals1 = vec![4.0f32, 5.0, 6.0, 7.0];
        let mut weights0 = vec![vec![make_weights(2, 2, &vals0)]];
        let mut weights1 = vec![vec![make_weights(2, 2, &vals1)]];

        let comms = SimComm::group(ep_size);
        let mut it = comms.into_iter();
        let comm0 = it.next().unwrap();
        let comm1 = it.next().unwrap();
        let old1 = old.clone();
        let new1 = new_p.clone();

        let h = std::thread::spawn(move || {
            rearrange_expert_weights_inplace(&old1, &new1, &mut weights1, 1, ep_size, &comm1)
                .unwrap();
            weights1
        });

        rearrange_expert_weights_inplace(&old, &new_p, &mut weights0, 0, ep_size, &comm0).unwrap();
        let weights1 = h.join().unwrap();

        assert_eq!(as_vec(&weights0[0][0]), vals0);
        assert_eq!(as_vec(&weights1[0][0]), vals1);
    }

    #[test]
    fn two_ranks_full_swap() {
        // ep_size=2, all experts move: rank 0 gets rank 1's experts and vice versa.
        // Old: rank 0 → {0,1}, rank 1 → {2,3}
        // New: rank 0 → {2,3}, rank 1 → {0,1}
        let ep_size = 2;
        let old = vec![vec![0usize, 1, 2, 3]];
        let new_p = vec![vec![2usize, 3, 0, 1]];
        let vals0 = vec![0.0f32, 1.0, 2.0, 3.0];
        let vals1 = vec![4.0f32, 5.0, 6.0, 7.0];
        let mut weights0 = vec![vec![make_weights(2, 2, &vals0)]];
        let mut weights1 = vec![vec![make_weights(2, 2, &vals1)]];

        let comms = SimComm::group(ep_size);
        let mut it = comms.into_iter();
        let comm0 = it.next().unwrap();
        let comm1 = it.next().unwrap();
        let old1 = old.clone();
        let new1 = new_p.clone();

        let h = std::thread::spawn(move || {
            rearrange_expert_weights_inplace(&old1, &new1, &mut weights1, 1, ep_size, &comm1)
                .unwrap();
            weights1
        });

        rearrange_expert_weights_inplace(&old, &new_p, &mut weights0, 0, ep_size, &comm0).unwrap();
        let weights1 = h.join().unwrap();

        // rank 0 now has expert 2 = [4,5] and expert 3 = [6,7]
        assert_eq!(as_vec(&weights0[0][0]), vec![4.0f32, 5.0, 6.0, 7.0]);
        // rank 1 now has expert 0 = [0,1] and expert 1 = [2,3]
        assert_eq!(as_vec(&weights1[0][0]), vec![0.0f32, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn multiple_weight_tensors_per_layer() {
        // Each expert has two weight matrices (e.g., gate_proj and up_proj).
        // ep_size=2, swap one expert between ranks.
        let ep_size = 2;
        let old = vec![vec![0usize, 1, 2, 3]];
        let new_p = vec![vec![0usize, 2, 1, 3]];

        // gate_proj: expert i has row [10*i, 10*i+1]
        let gate0 = vec![0.0f32, 1.0, 10.0, 11.0]; // experts 0,1
        let gate1 = vec![20.0f32, 21.0, 30.0, 31.0]; // experts 2,3
                                                     // up_proj: expert i has row [100*i, 100*i+1]
        let up0 = vec![0.0f32, 1.0, 100.0, 101.0];
        let up1 = vec![200.0f32, 201.0, 300.0, 301.0];

        let mut weights0 = vec![vec![make_weights(2, 2, &gate0), make_weights(2, 2, &up0)]];
        let mut weights1 = vec![vec![make_weights(2, 2, &gate1), make_weights(2, 2, &up1)]];

        let comms = SimComm::group(ep_size);
        let mut it = comms.into_iter();
        let comm0 = it.next().unwrap();
        let comm1 = it.next().unwrap();
        let old1 = old.clone();
        let new1 = new_p.clone();

        let h = std::thread::spawn(move || {
            rearrange_expert_weights_inplace(&old1, &new1, &mut weights1, 1, ep_size, &comm1)
                .unwrap();
            weights1
        });

        rearrange_expert_weights_inplace(&old, &new_p, &mut weights0, 0, ep_size, &comm0).unwrap();
        let weights1 = h.join().unwrap();

        // rank 0: slot 0 ← expert 0, slot 1 ← expert 2
        assert_eq!(
            as_vec(&weights0[0][0]),
            vec![0.0f32, 1.0, 20.0, 21.0],
            "gate_proj rank 0"
        );
        assert_eq!(
            as_vec(&weights0[0][1]),
            vec![0.0f32, 1.0, 200.0, 201.0],
            "up_proj rank 0"
        );
        // rank 1: slot 0 ← expert 1, slot 1 ← expert 3
        assert_eq!(
            as_vec(&weights1[0][0]),
            vec![10.0f32, 11.0, 30.0, 31.0],
            "gate_proj rank 1"
        );
        assert_eq!(
            as_vec(&weights1[0][1]),
            vec![100.0f32, 101.0, 300.0, 301.0],
            "up_proj rank 1"
        );
    }

    #[test]
    fn multiple_layers() {
        // Two MoE layers, each with different placement changes.
        // Layer 0: identity (no move), Layer 1: full swap.
        let ep_size = 2;
        let old = vec![
            vec![0usize, 1, 2, 3], // layer 0
            vec![4usize, 5, 6, 7], // layer 1
        ];
        let new_p = vec![
            vec![0usize, 1, 2, 3], // layer 0: unchanged
            vec![6usize, 7, 4, 5], // layer 1: full swap
        ];

        let w00 = vec![0.0f32, 1.0, 2.0, 3.0]; // layer 0, rank 0
        let w01 = vec![4.0f32, 5.0, 6.0, 7.0]; // layer 0, rank 1
        let w10 = vec![40.0f32, 41.0, 50.0, 51.0]; // layer 1, rank 0 (experts 4,5)
        let w11 = vec![60.0f32, 61.0, 70.0, 71.0]; // layer 1, rank 1 (experts 6,7)

        let mut weights0 = vec![
            vec![make_weights(2, 2, &w00)],
            vec![make_weights(2, 2, &w10)],
        ];
        let mut weights1 = vec![
            vec![make_weights(2, 2, &w01)],
            vec![make_weights(2, 2, &w11)],
        ];

        let comms = SimComm::group(ep_size);
        let mut it = comms.into_iter();
        let comm0 = it.next().unwrap();
        let comm1 = it.next().unwrap();
        let old1 = old.clone();
        let new1 = new_p.clone();

        let h = std::thread::spawn(move || {
            rearrange_expert_weights_inplace(&old1, &new1, &mut weights1, 1, ep_size, &comm1)
                .unwrap();
            weights1
        });

        rearrange_expert_weights_inplace(&old, &new_p, &mut weights0, 0, ep_size, &comm0).unwrap();
        let weights1 = h.join().unwrap();

        // Layer 0: unchanged
        assert_eq!(as_vec(&weights0[0][0]), w00, "layer 0 rank 0 unchanged");
        assert_eq!(as_vec(&weights1[0][0]), w01, "layer 0 rank 1 unchanged");
        // Layer 1: rank 0 now has experts 6,7; rank 1 now has experts 4,5
        assert_eq!(
            as_vec(&weights0[1][0]),
            vec![60.0f32, 61.0, 70.0, 71.0],
            "layer 1 rank 0 after swap"
        );
        assert_eq!(
            as_vec(&weights1[1][0]),
            vec![40.0f32, 41.0, 50.0, 51.0],
            "layer 1 rank 1 after swap"
        );
    }

    #[test]
    fn empty_layers_no_op() {
        // Edge case: zero layers — should return Ok immediately.
        let ep_size = 2;
        let old: LayerExpertPlacement = vec![];
        let new_p: LayerExpertPlacement = vec![];
        let mut weights: Vec<Vec<Tensor>> = vec![];
        let comms = SimComm::group(ep_size);
        rearrange_expert_weights_inplace(&old, &new_p, &mut weights, 0, ep_size, &comms[0])
            .unwrap();
    }
}
