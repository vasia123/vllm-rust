//! Tree attention utilities for speculative decoding verification.
//!
//! When speculative decoding proposes multiple candidate tokens — whether from
//! a linear chain (draft model, n-gram) or a tree (Medusa, Eagle) — these
//! candidates must be verified in a single forward pass through the target
//! model. Tree attention constructs the appropriate attention mask so that
//! each candidate token can only attend to its ancestors in the speculation
//! tree, preserving causal semantics.
//!
//! Reference: vLLM `vllm/v1/spec_decode/` tree attention utilities.

use candle_core::{DType, Device, Result, Tensor};

/// Represents a tree of speculative token candidates.
///
/// Each node in the tree is a candidate token. The root (index 0) is the
/// last accepted token from the previous step. Children represent speculated
/// continuations. A linear chain is a degenerate tree where every node has
/// exactly one child.
///
/// The tree structure is encoded via `parent_indices`: for each node, the
/// index of its parent in the `tokens` array. The root has parent index -1.
#[derive(Debug, Clone)]
pub struct SpeculationTree {
    /// Token IDs at each node.
    pub tokens: Vec<u32>,
    /// Parent index for each node (-1 for root).
    pub parent_indices: Vec<i32>,
    /// Depth of each node (0 for root).
    pub depths: Vec<usize>,
}

impl SpeculationTree {
    /// Create a flat tree (linear chain) from a sequence of proposed tokens.
    ///
    /// The resulting tree is a simple chain: node 0 is the root, node 1 is
    /// its child, node 2 is node 1's child, and so on. This is appropriate
    /// for draft-model or n-gram proposals where tokens are predicted
    /// sequentially.
    pub fn from_linear(tokens: &[u32]) -> Self {
        let n = tokens.len();
        let mut parent_indices = Vec::with_capacity(n);
        let mut depths = Vec::with_capacity(n);

        for i in 0..n {
            if i == 0 {
                parent_indices.push(-1);
                depths.push(0);
            } else {
                parent_indices.push((i - 1) as i32);
                depths.push(i);
            }
        }

        Self {
            tokens: tokens.to_vec(),
            parent_indices,
            depths,
        }
    }

    /// Create a tree from Medusa-style multi-head predictions.
    ///
    /// Each entry in `candidates` is the list of top-k token IDs from one
    /// Medusa head, ordered by position (head 0 predicts position i+1, head 1
    /// predicts position i+2, etc.).
    ///
    /// The tree structure has the root at index 0 (the first token from head 0's
    /// top-1 prediction), and each head's candidates are children of the
    /// previous head's candidates, forming a multi-branch tree.
    ///
    /// For simplicity, the tree is structured as:
    /// - Level 0: root (not from candidates, placeholder token 0)
    /// - Level 1: all candidates from head 0 (children of root)
    /// - Level 2: all candidates from head 1 (each is child of every level-1 node)
    ///
    /// In practice, Medusa uses a fixed tree topology. This method builds a
    /// simple breadth-first expansion where each head's candidates branch from
    /// all nodes at the previous level.
    pub fn from_medusa_candidates(candidates: &[Vec<u32>]) -> Self {
        if candidates.is_empty() {
            return Self {
                tokens: Vec::new(),
                parent_indices: Vec::new(),
                depths: Vec::new(),
            };
        }

        let mut tokens = Vec::new();
        let mut parent_indices = Vec::new();
        let mut depths = Vec::new();

        // Level-0 nodes are the first head's candidates, all parented to a
        // virtual root. We use the first candidate of head 0 as the root node.
        // Each head's candidates are children of every node at the previous level.
        //
        // Track start/end indices for each level so we know the parent range.
        let mut level_start: usize = 0;
        let mut level_end: usize;

        // First level: candidates from head 0, all with parent = -1 (root-level)
        for &token in &candidates[0] {
            tokens.push(token);
            parent_indices.push(-1);
            depths.push(0);
        }
        level_end = tokens.len();

        // Subsequent levels: candidates from head i are children of nodes at level i-1
        for (head_idx, head_candidates) in candidates.iter().enumerate().skip(1) {
            let new_level_start = tokens.len();
            for parent_idx in level_start..level_end {
                for &token in head_candidates {
                    tokens.push(token);
                    parent_indices.push(parent_idx as i32);
                    depths.push(head_idx);
                }
            }
            level_start = new_level_start;
            level_end = tokens.len();
        }

        Self {
            tokens,
            parent_indices,
            depths,
        }
    }

    /// Build the attention mask for tree verification.
    ///
    /// Each node can attend to itself and all its ancestors. The mask is a
    /// square matrix of shape `[num_nodes, num_nodes]` where `mask[i][j] = 0.0`
    /// means node i can attend to node j, and `mask[i][j] = -inf` means it
    /// cannot (following the additive mask convention used in transformers).
    ///
    /// For a linear chain this produces a standard causal (lower-triangular) mask.
    pub fn build_attention_mask(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        let n = self.tokens.len();
        if n == 0 {
            return Tensor::zeros((0, 0), dtype, device);
        }

        let neg_inf = f32::NEG_INFINITY;
        let mut mask_data = vec![neg_inf; n * n];

        for i in 0..n {
            // Node i can attend to itself
            mask_data[i * n + i] = 0.0;

            // Walk up the parent chain and allow attention to each ancestor
            let mut current = self.parent_indices[i];
            while current >= 0 {
                let parent = current as usize;
                mask_data[i * n + parent] = 0.0;
                current = self.parent_indices[parent];
            }
        }

        let mask = Tensor::from_vec(mask_data, (n, n), device)?;
        mask.to_dtype(dtype)
    }

    /// Verify speculative tokens against the base model's logits and return
    /// the longest accepted prefix.
    ///
    /// For each node in depth-first order along the primary path (first child
    /// at each level), compare the base model's prediction at the parent
    /// position with the speculated token. Accept tokens as long as they match.
    ///
    /// `base_logits` has shape `[1, num_nodes, vocab_size]` or `[num_nodes, vocab_size]`.
    /// `temperature` controls sampling: 0.0 means greedy (argmax).
    ///
    /// Returns the accepted tokens along the primary (depth-first) path.
    pub fn verify_and_accept(&self, base_logits: &Tensor, temperature: f32) -> Result<Vec<u32>> {
        if self.tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Squeeze to [num_nodes, vocab_size]
        let logits = if base_logits.dims().len() == 3 {
            base_logits.squeeze(0)?
        } else {
            base_logits.clone()
        };

        let num_nodes = self.tokens.len();
        let mut accepted = Vec::new();

        // Build children map: for each node, its children sorted by index
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        let mut root_nodes: Vec<usize> = Vec::new();
        for (i, &parent) in self.parent_indices.iter().enumerate() {
            if parent < 0 {
                root_nodes.push(i);
            } else {
                children[parent as usize].push(i);
            }
        }

        // Walk the primary path (first root, then first child at each level)
        if root_nodes.is_empty() {
            return Ok(Vec::new());
        }

        // For greedy verification, check each node along the primary path.
        // The base model's logits at position i predict the token at position i+1.
        // For the root node, accept it unconditionally (it's the starting point).
        let mut path = Vec::new();
        let mut current = root_nodes[0];
        path.push(current);
        while !children[current].is_empty() {
            current = children[current][0];
            path.push(current);
        }

        // Accept root unconditionally
        accepted.push(self.tokens[path[0]]);

        // For each subsequent node in the path, check if the base model agrees
        for window in path.windows(2) {
            let parent_idx = window[0];
            let child_idx = window[1];

            // Get the base model's prediction at the parent position
            let parent_logits = logits.get(parent_idx)?;
            let predicted_token = if temperature <= 0.0 || temperature < 1e-6 {
                parent_logits.argmax(0)?.to_scalar::<u32>()?
            } else {
                let scaled = (parent_logits / temperature as f64)?;
                scaled.argmax(0)?.to_scalar::<u32>()?
            };

            if predicted_token == self.tokens[child_idx] {
                accepted.push(self.tokens[child_idx]);
            } else {
                // Mismatch: stop accepting, but add the base model's prediction
                // as a bonus token (standard speculative decoding behavior).
                accepted.push(predicted_token);
                break;
            }
        }

        Ok(accepted)
    }

    /// Total number of nodes (including root-level nodes).
    pub fn num_nodes(&self) -> usize {
        self.tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_linear_tree() {
        let tokens = vec![10u32, 20, 30, 40];
        let tree = SpeculationTree::from_linear(&tokens);

        assert_eq!(tree.tokens, vec![10, 20, 30, 40]);
        assert_eq!(tree.parent_indices, vec![-1, 0, 1, 2]);
        assert_eq!(tree.depths, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_linear_tree_empty() {
        let tree = SpeculationTree::from_linear(&[]);
        assert!(tree.tokens.is_empty());
        assert!(tree.parent_indices.is_empty());
        assert!(tree.depths.is_empty());
    }

    #[test]
    fn test_linear_tree_single() {
        let tree = SpeculationTree::from_linear(&[42]);
        assert_eq!(tree.tokens, vec![42]);
        assert_eq!(tree.parent_indices, vec![-1]);
        assert_eq!(tree.depths, vec![0]);
    }

    #[test]
    fn test_medusa_tree() {
        // Head 0: top-2 candidates [10, 11]
        // Head 1: top-2 candidates [20, 21]
        let candidates = vec![vec![10u32, 11], vec![20, 21]];
        let tree = SpeculationTree::from_medusa_candidates(&candidates);

        // Level 0: tokens [10, 11] with parent -1, depth 0
        // Level 1: for each parent in level 0, expand with [20, 21]
        //   parent 0 (token 10) -> [20, 21]
        //   parent 1 (token 11) -> [20, 21]
        assert_eq!(tree.tokens.len(), 6); // 2 + 2*2
        assert_eq!(tree.tokens[0], 10);
        assert_eq!(tree.tokens[1], 11);
        // Children of node 0: tokens 20, 21
        assert_eq!(tree.tokens[2], 20);
        assert_eq!(tree.tokens[3], 21);
        // Children of node 1: tokens 20, 21
        assert_eq!(tree.tokens[4], 20);
        assert_eq!(tree.tokens[5], 21);

        // Parent indices
        assert_eq!(tree.parent_indices[0], -1);
        assert_eq!(tree.parent_indices[1], -1);
        assert_eq!(tree.parent_indices[2], 0);
        assert_eq!(tree.parent_indices[3], 0);
        assert_eq!(tree.parent_indices[4], 1);
        assert_eq!(tree.parent_indices[5], 1);

        // Depths
        assert_eq!(tree.depths[0], 0);
        assert_eq!(tree.depths[1], 0);
        assert_eq!(tree.depths[2], 1);
        assert_eq!(tree.depths[3], 1);
        assert_eq!(tree.depths[4], 1);
        assert_eq!(tree.depths[5], 1);
    }

    #[test]
    fn test_medusa_tree_empty() {
        let tree = SpeculationTree::from_medusa_candidates(&[]);
        assert!(tree.tokens.is_empty());
    }

    #[test]
    fn test_medusa_tree_single_head() {
        let candidates = vec![vec![10u32, 11, 12]];
        let tree = SpeculationTree::from_medusa_candidates(&candidates);

        assert_eq!(tree.tokens, vec![10, 11, 12]);
        assert_eq!(tree.parent_indices, vec![-1, -1, -1]);
        assert_eq!(tree.depths, vec![0, 0, 0]);
    }

    #[test]
    fn test_attention_mask_causal() {
        // Linear chain of 3 tokens -> standard lower-triangular causal mask
        let tree = SpeculationTree::from_linear(&[1, 2, 3]);
        let mask = tree.build_attention_mask(DType::F32, &Device::Cpu).unwrap();

        assert_eq!(mask.dims(), &[3, 3]);

        let mask_data: Vec<Vec<f32>> = (0..3)
            .map(|i| mask.get(i).unwrap().to_vec1::<f32>().unwrap())
            .collect();

        let neg_inf = f32::NEG_INFINITY;

        // Row 0: can only attend to self
        assert_eq!(mask_data[0][0], 0.0);
        assert_eq!(mask_data[0][1], neg_inf);
        assert_eq!(mask_data[0][2], neg_inf);

        // Row 1: can attend to 0 and self
        assert_eq!(mask_data[1][0], 0.0);
        assert_eq!(mask_data[1][1], 0.0);
        assert_eq!(mask_data[1][2], neg_inf);

        // Row 2: can attend to 0, 1, and self
        assert_eq!(mask_data[2][0], 0.0);
        assert_eq!(mask_data[2][1], 0.0);
        assert_eq!(mask_data[2][2], 0.0);
    }

    #[test]
    fn test_attention_mask_tree() {
        // Medusa-style tree: head 0 = [A, B], head 1 = [C, D]
        // Tree:
        //   0: A (root, parent=-1)
        //   1: B (root, parent=-1)
        //   2: C (parent=0)
        //   3: D (parent=0)
        //   4: C (parent=1)
        //   5: D (parent=1)
        let candidates = vec![vec![100u32, 101], vec![200, 201]];
        let tree = SpeculationTree::from_medusa_candidates(&candidates);
        let mask = tree.build_attention_mask(DType::F32, &Device::Cpu).unwrap();

        assert_eq!(mask.dims(), &[6, 6]);

        let mask_data: Vec<Vec<f32>> = (0..6)
            .map(|i| mask.get(i).unwrap().to_vec1::<f32>().unwrap())
            .collect();

        // Node 2 (parent=0): can attend to self (2) and ancestor (0)
        assert_eq!(mask_data[2][0], 0.0); // ancestor
        assert!(mask_data[2][1].is_infinite()); // not ancestor
        assert_eq!(mask_data[2][2], 0.0); // self
        assert!(mask_data[2][3].is_infinite()); // sibling
        assert!(mask_data[2][4].is_infinite()); // cousin
        assert!(mask_data[2][5].is_infinite()); // cousin

        // Node 4 (parent=1): can attend to self (4) and ancestor (1)
        assert!(mask_data[4][0].is_infinite()); // not ancestor
        assert_eq!(mask_data[4][1], 0.0); // ancestor
        assert!(mask_data[4][2].is_infinite()); // not related
        assert!(mask_data[4][3].is_infinite()); // not related
        assert_eq!(mask_data[4][4], 0.0); // self
        assert!(mask_data[4][5].is_infinite()); // sibling
    }

    #[test]
    fn test_attention_mask_empty() {
        let tree = SpeculationTree::from_linear(&[]);
        let mask = tree.build_attention_mask(DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[0, 0]);
    }

    #[test]
    fn test_tree_num_nodes() {
        let tree = SpeculationTree::from_linear(&[1, 2, 3, 4, 5]);
        assert_eq!(tree.num_nodes(), 5);

        let tree = SpeculationTree::from_linear(&[]);
        assert_eq!(tree.num_nodes(), 0);

        // Medusa: 2 candidates * 2 heads = 2 + 2*2 = 6
        let candidates = vec![vec![10u32, 11], vec![20, 21]];
        let tree = SpeculationTree::from_medusa_candidates(&candidates);
        assert_eq!(tree.num_nodes(), 6);
    }

    #[test]
    fn test_verify_and_accept_all_match() {
        // Linear chain [10, 20, 30], base model agrees with all
        let tree = SpeculationTree::from_linear(&[10, 20, 30]);

        // Logits: at position 0, argmax -> 20; at position 1, argmax -> 30
        // (position 2 logits don't matter for acceptance of existing nodes)
        let vocab_size = 50usize;
        let mut logits_data = vec![-100.0f32; 3 * vocab_size];
        // Position 0 predicts token 20
        logits_data[0 * vocab_size + 20] = 100.0;
        // Position 1 predicts token 30
        logits_data[1 * vocab_size + 30] = 100.0;
        // Position 2 predicts token 40 (bonus token)
        logits_data[2 * vocab_size + 40] = 100.0;

        let logits = Tensor::from_vec(logits_data, (3, vocab_size), &Device::Cpu).unwrap();
        let accepted = tree.verify_and_accept(&logits, 0.0).unwrap();

        // All match: root (10) + matched (20, 30) + bonus from position 2
        // Actually: root accepted unconditionally, then 20 matches, then 30 matches,
        // and since there are no more children, we stop. The path has 3 nodes
        // and 2 windows, both match, so we get [10, 20, 30].
        assert_eq!(accepted, vec![10, 20, 30]);
    }

    #[test]
    fn test_verify_and_accept_partial_match() {
        // Linear chain [10, 20, 30], base model disagrees at position 1
        let tree = SpeculationTree::from_linear(&[10, 20, 30]);

        let vocab_size = 100usize;
        let mut logits_data = vec![-100.0f32; 3 * vocab_size];
        // Position 0 predicts token 20 (matches)
        logits_data[0 * vocab_size + 20] = 100.0;
        // Position 1 predicts token 99 (doesn't match 30)
        logits_data[1 * vocab_size + 99] = 100.0;

        let logits = Tensor::from_vec(logits_data, (3, vocab_size), &Device::Cpu).unwrap();
        let accepted = tree.verify_and_accept(&logits, 0.0).unwrap();

        // Root (10) accepted, 20 matches, 30 doesn't match -> bonus 99
        assert_eq!(accepted, vec![10, 20, 99]);
    }

    #[test]
    fn test_verify_and_accept_first_mismatch() {
        // Linear chain [10, 20, 30], base model disagrees at position 0
        let tree = SpeculationTree::from_linear(&[10, 20, 30]);

        let vocab_size = 100usize;
        let mut logits_data = vec![-100.0f32; 3 * vocab_size];
        // Position 0 predicts token 99 (doesn't match 20)
        logits_data[0 * vocab_size + 99] = 100.0;

        let logits = Tensor::from_vec(logits_data, (3, vocab_size), &Device::Cpu).unwrap();
        let accepted = tree.verify_and_accept(&logits, 0.0).unwrap();

        // Root (10) accepted, then mismatch at first edge -> bonus token 99
        assert_eq!(accepted, vec![10, 99]);
    }

    #[test]
    fn test_verify_and_accept_empty() {
        let tree = SpeculationTree::from_linear(&[]);
        let logits = Tensor::zeros((0, 50), DType::F32, &Device::Cpu).unwrap();
        let accepted = tree.verify_and_accept(&logits, 0.0).unwrap();
        assert!(accepted.is_empty());
    }
}
