//! Expert placement mapping for Expert Parallelism (EP).
//!
//! This module provides mapping between global expert IDs and local expert IDs
//! when experts are distributed across multiple EP ranks.
//!
//! ## Placement Strategies
//!
//! - **Linear**: Rank 0 gets experts [0..n/ep], Rank 1 gets [n/ep..2n/ep], etc.
//!   Contiguous blocks, good for locality.
//!
//! - **RoundRobin**: Rank 0 gets experts [0, ep, 2ep, ...], Rank 1 gets [1, ep+1, 2ep+1, ...].
//!   Better load balancing for sequential expert access patterns.
//!
//! ## Example
//!
//! ```ignore
//! use vllm_core::moe::ExpertMap;
//!
//! // 8 experts distributed across 2 EP ranks
//! let map = ExpertMap::new(8, 2, 0, ExpertPlacement::Linear);
//!
//! // Rank 0 owns experts 0-3
//! assert!(map.is_local(0));
//! assert!(map.is_local(3));
//! assert!(!map.is_local(4));
//!
//! // Global ID 2 -> Local ID 2 on rank 0
//! assert_eq!(map.to_local(2), Some(2));
//! ```

/// Expert placement strategy across EP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertPlacement {
    /// Contiguous blocks: Rank i gets experts [i*n/ep .. (i+1)*n/ep).
    /// Good for locality when experts are accessed in order.
    #[default]
    Linear,
    /// Round-robin: Rank i gets experts [i, i+ep, i+2ep, ...].
    /// Better load balancing for varied access patterns.
    RoundRobin,
}

/// Maps global expert IDs to local expert IDs for a specific EP rank.
///
/// When using Expert Parallelism, each GPU only stores a subset of experts.
/// This struct handles the mapping between global expert IDs (used in routing)
/// and local expert IDs (used for weight storage and computation).
#[derive(Debug, Clone)]
pub struct ExpertMap {
    /// Mapping from global expert ID to local expert ID.
    /// Value is `Some(local_id)` if expert is local, `None` otherwise.
    global_to_local: Vec<Option<usize>>,
    /// Mapping from local expert ID to global expert ID.
    local_to_global: Vec<usize>,
    /// Number of experts stored on this rank.
    local_num_experts: usize,
    /// This rank's position in the EP group.
    ep_rank: usize,
    /// Total number of EP ranks.
    ep_size: usize,
    /// Total number of experts globally.
    num_experts: usize,
    /// Placement strategy used.
    placement: ExpertPlacement,
}

impl ExpertMap {
    /// Create a new expert map for the given EP configuration.
    ///
    /// # Arguments
    /// * `num_experts` - Total number of experts globally
    /// * `ep_size` - Number of EP ranks (must divide num_experts evenly)
    /// * `ep_rank` - This rank's position (0 <= ep_rank < ep_size)
    /// * `placement` - How to distribute experts across ranks
    ///
    /// # Panics
    /// Panics if:
    /// - `ep_size` is 0
    /// - `ep_rank >= ep_size`
    /// - `num_experts` is not evenly divisible by `ep_size`
    pub fn new(
        num_experts: usize,
        ep_size: usize,
        ep_rank: usize,
        placement: ExpertPlacement,
    ) -> Self {
        assert!(ep_size > 0, "ep_size must be > 0");
        assert!(ep_rank < ep_size, "ep_rank must be < ep_size");
        assert!(
            num_experts % ep_size == 0,
            "num_experts ({}) must be divisible by ep_size ({})",
            num_experts,
            ep_size
        );

        let local_num_experts = num_experts / ep_size;
        let mut global_to_local = vec![None; num_experts];
        let mut local_to_global = Vec::with_capacity(local_num_experts);

        match placement {
            ExpertPlacement::Linear => {
                // Rank i gets experts [i*local_num .. (i+1)*local_num)
                let start = ep_rank * local_num_experts;
                for local_id in 0..local_num_experts {
                    let global_id = start + local_id;
                    global_to_local[global_id] = Some(local_id);
                    local_to_global.push(global_id);
                }
            }
            ExpertPlacement::RoundRobin => {
                // Rank i gets experts [i, i+ep_size, i+2*ep_size, ...]
                for local_id in 0..local_num_experts {
                    let global_id = ep_rank + local_id * ep_size;
                    global_to_local[global_id] = Some(local_id);
                    local_to_global.push(global_id);
                }
            }
        }

        Self {
            global_to_local,
            local_to_global,
            local_num_experts,
            ep_rank,
            ep_size,
            num_experts,
            placement,
        }
    }

    /// Create an expert map for single-GPU execution (no EP).
    ///
    /// All experts are local, mapping is identity.
    pub fn single_gpu(num_experts: usize) -> Self {
        Self::new(num_experts, 1, 0, ExpertPlacement::Linear)
    }

    /// Check if a global expert ID is stored on this rank.
    #[inline]
    pub fn is_local(&self, global_id: usize) -> bool {
        global_id < self.num_experts && self.global_to_local[global_id].is_some()
    }

    /// Convert a global expert ID to a local expert ID.
    ///
    /// Returns `None` if the expert is not on this rank.
    #[inline]
    pub fn to_local(&self, global_id: usize) -> Option<usize> {
        if global_id < self.num_experts {
            self.global_to_local[global_id]
        } else {
            None
        }
    }

    /// Convert a local expert ID to a global expert ID.
    ///
    /// # Panics
    /// Panics if `local_id >= local_num_experts`.
    #[inline]
    pub fn to_global(&self, local_id: usize) -> usize {
        self.local_to_global[local_id]
    }

    /// Get the rank that owns a given global expert ID.
    #[inline]
    pub fn owner_rank(&self, global_id: usize) -> usize {
        debug_assert!(global_id < self.num_experts);
        match self.placement {
            ExpertPlacement::Linear => global_id / self.local_num_experts,
            ExpertPlacement::RoundRobin => global_id % self.ep_size,
        }
    }

    /// Get the number of experts stored on this rank.
    #[inline]
    pub fn local_num_experts(&self) -> usize {
        self.local_num_experts
    }

    /// Get the total number of experts globally.
    #[inline]
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get this rank's position in the EP group.
    #[inline]
    pub fn ep_rank(&self) -> usize {
        self.ep_rank
    }

    /// Get the total number of EP ranks.
    #[inline]
    pub fn ep_size(&self) -> usize {
        self.ep_size
    }

    /// Get the placement strategy.
    #[inline]
    pub fn placement(&self) -> ExpertPlacement {
        self.placement
    }

    /// Iterate over all local expert IDs.
    pub fn local_expert_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.local_num_experts
    }

    /// Iterate over all global expert IDs that are local.
    pub fn local_global_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.local_to_global.iter().copied()
    }

    /// Get the local-to-global mapping as a slice.
    #[inline]
    pub fn local_to_global_map(&self) -> &[usize] {
        &self.local_to_global
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_placement_basic() {
        // 8 experts, 2 EP ranks
        let map = ExpertMap::new(8, 2, 0, ExpertPlacement::Linear);

        // Rank 0 should have experts 0-3
        assert!(map.is_local(0));
        assert!(map.is_local(1));
        assert!(map.is_local(2));
        assert!(map.is_local(3));
        assert!(!map.is_local(4));
        assert!(!map.is_local(5));
        assert!(!map.is_local(6));
        assert!(!map.is_local(7));

        assert_eq!(map.local_num_experts(), 4);
        assert_eq!(map.to_local(0), Some(0));
        assert_eq!(map.to_local(3), Some(3));
        assert_eq!(map.to_local(4), None);
    }

    #[test]
    fn test_linear_placement_rank1() {
        // 8 experts, 2 EP ranks, rank 1
        let map = ExpertMap::new(8, 2, 1, ExpertPlacement::Linear);

        // Rank 1 should have experts 4-7
        assert!(!map.is_local(0));
        assert!(!map.is_local(3));
        assert!(map.is_local(4));
        assert!(map.is_local(7));

        assert_eq!(map.to_local(4), Some(0));
        assert_eq!(map.to_local(7), Some(3));
        assert_eq!(map.to_global(0), 4);
        assert_eq!(map.to_global(3), 7);
    }

    #[test]
    fn test_round_robin_placement() {
        // 8 experts, 4 EP ranks
        let map = ExpertMap::new(8, 4, 0, ExpertPlacement::RoundRobin);

        // Rank 0 gets experts 0, 4
        assert!(map.is_local(0));
        assert!(!map.is_local(1));
        assert!(!map.is_local(2));
        assert!(!map.is_local(3));
        assert!(map.is_local(4));
        assert!(!map.is_local(5));

        assert_eq!(map.local_num_experts(), 2);
        assert_eq!(map.to_local(0), Some(0));
        assert_eq!(map.to_local(4), Some(1));
        assert_eq!(map.to_global(0), 0);
        assert_eq!(map.to_global(1), 4);
    }

    #[test]
    fn test_round_robin_placement_rank1() {
        // 8 experts, 4 EP ranks, rank 1
        let map = ExpertMap::new(8, 4, 1, ExpertPlacement::RoundRobin);

        // Rank 1 gets experts 1, 5
        assert!(!map.is_local(0));
        assert!(map.is_local(1));
        assert!(!map.is_local(4));
        assert!(map.is_local(5));

        assert_eq!(map.to_local(1), Some(0));
        assert_eq!(map.to_local(5), Some(1));
    }

    #[test]
    fn test_owner_rank_linear() {
        let map = ExpertMap::new(8, 4, 0, ExpertPlacement::Linear);

        // Each rank owns 2 experts
        assert_eq!(map.owner_rank(0), 0);
        assert_eq!(map.owner_rank(1), 0);
        assert_eq!(map.owner_rank(2), 1);
        assert_eq!(map.owner_rank(3), 1);
        assert_eq!(map.owner_rank(4), 2);
        assert_eq!(map.owner_rank(5), 2);
        assert_eq!(map.owner_rank(6), 3);
        assert_eq!(map.owner_rank(7), 3);
    }

    #[test]
    fn test_owner_rank_round_robin() {
        let map = ExpertMap::new(8, 4, 0, ExpertPlacement::RoundRobin);

        // Round-robin: expert i is owned by rank (i % ep_size)
        assert_eq!(map.owner_rank(0), 0);
        assert_eq!(map.owner_rank(1), 1);
        assert_eq!(map.owner_rank(2), 2);
        assert_eq!(map.owner_rank(3), 3);
        assert_eq!(map.owner_rank(4), 0);
        assert_eq!(map.owner_rank(5), 1);
        assert_eq!(map.owner_rank(6), 2);
        assert_eq!(map.owner_rank(7), 3);
    }

    #[test]
    fn test_single_gpu() {
        let map = ExpertMap::single_gpu(8);

        // All experts should be local
        for i in 0..8 {
            assert!(map.is_local(i));
            assert_eq!(map.to_local(i), Some(i));
            assert_eq!(map.to_global(i), i);
            assert_eq!(map.owner_rank(i), 0);
        }

        assert_eq!(map.local_num_experts(), 8);
        assert_eq!(map.ep_size(), 1);
        assert_eq!(map.ep_rank(), 0);
    }

    #[test]
    fn test_boundary_ep_size_1() {
        // EP size 1 is equivalent to single GPU
        let map = ExpertMap::new(4, 1, 0, ExpertPlacement::Linear);

        for i in 0..4 {
            assert!(map.is_local(i));
            assert_eq!(map.to_local(i), Some(i));
        }
    }

    #[test]
    fn test_boundary_ep_size_equals_num_experts() {
        // Each rank gets exactly 1 expert
        let map0 = ExpertMap::new(4, 4, 0, ExpertPlacement::Linear);
        let map1 = ExpertMap::new(4, 4, 1, ExpertPlacement::Linear);
        let map2 = ExpertMap::new(4, 4, 2, ExpertPlacement::Linear);
        let map3 = ExpertMap::new(4, 4, 3, ExpertPlacement::Linear);

        assert_eq!(map0.local_num_experts(), 1);
        assert!(map0.is_local(0));
        assert!(map1.is_local(1));
        assert!(map2.is_local(2));
        assert!(map3.is_local(3));

        assert_eq!(map0.to_global(0), 0);
        assert_eq!(map1.to_global(0), 1);
        assert_eq!(map2.to_global(0), 2);
        assert_eq!(map3.to_global(0), 3);
    }

    #[test]
    fn test_local_expert_ids_iterator() {
        let map = ExpertMap::new(8, 2, 0, ExpertPlacement::Linear);

        let local_ids: Vec<usize> = map.local_expert_ids().collect();
        assert_eq!(local_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_local_global_ids_iterator() {
        let map = ExpertMap::new(8, 4, 1, ExpertPlacement::RoundRobin);

        let global_ids: Vec<usize> = map.local_global_ids().collect();
        // Rank 1 in round-robin gets experts 1 and 5
        assert_eq!(global_ids, vec![1, 5]);
    }

    #[test]
    fn test_to_local_out_of_bounds() {
        let map = ExpertMap::new(8, 2, 0, ExpertPlacement::Linear);

        // Global ID >= num_experts should return None
        assert_eq!(map.to_local(8), None);
        assert_eq!(map.to_local(100), None);
    }

    #[test]
    fn test_is_local_out_of_bounds() {
        let map = ExpertMap::new(8, 2, 0, ExpertPlacement::Linear);

        // Global ID >= num_experts should return false
        assert!(!map.is_local(8));
        assert!(!map.is_local(100));
    }

    #[test]
    fn test_accessors() {
        let map = ExpertMap::new(16, 4, 2, ExpertPlacement::RoundRobin);

        assert_eq!(map.num_experts(), 16);
        assert_eq!(map.ep_size(), 4);
        assert_eq!(map.ep_rank(), 2);
        assert_eq!(map.local_num_experts(), 4);
        assert_eq!(map.placement(), ExpertPlacement::RoundRobin);
    }

    #[test]
    #[should_panic(expected = "ep_size must be > 0")]
    fn test_invalid_ep_size_zero() {
        ExpertMap::new(8, 0, 0, ExpertPlacement::Linear);
    }

    #[test]
    #[should_panic(expected = "ep_rank must be < ep_size")]
    fn test_invalid_ep_rank() {
        ExpertMap::new(8, 2, 2, ExpertPlacement::Linear);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_num_experts_not_divisible() {
        ExpertMap::new(7, 2, 0, ExpertPlacement::Linear);
    }

    #[test]
    fn test_placement_default() {
        assert_eq!(ExpertPlacement::default(), ExpertPlacement::Linear);
    }

    #[test]
    fn test_local_to_global_map() {
        let map = ExpertMap::new(8, 2, 1, ExpertPlacement::Linear);

        let mapping = map.local_to_global_map();
        assert_eq!(mapping, &[4, 5, 6, 7]);
    }
}
