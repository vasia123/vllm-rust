use super::block_pool::{BlockId, NULL_BLOCK};

/// Per-request mapping: logical block index → physical BlockId.
#[derive(Clone)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    num_tokens_stored: usize,
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens_stored: 0,
            block_size,
        }
    }

    /// Reconstruct a BlockTable from existing block IDs and stored token count.
    /// Used by the default `forward_decode_batch` fallback.
    pub fn from_block_ids(blocks: Vec<BlockId>, num_tokens_stored: usize) -> Self {
        let block_size = if blocks.is_empty() {
            16
        } else {
            // Infer block_size: at least ceil(num_tokens / num_blocks)
            num_tokens_stored.div_ceil(blocks.len()).max(1)
        };
        Self {
            blocks,
            num_tokens_stored,
            block_size,
        }
    }

    /// Total tokens currently stored.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens_stored
    }

    /// How many new blocks are needed to store `new_tokens` additional tokens.
    pub fn blocks_needed(&self, new_tokens: usize) -> usize {
        if new_tokens == 0 {
            return 0;
        }
        let total_after = self.num_tokens_stored + new_tokens;
        let blocks_required = total_after.div_ceil(self.block_size);
        blocks_required.saturating_sub(self.blocks.len())
    }

    /// Append newly allocated block IDs.
    pub fn append_blocks(&mut self, block_ids: &[BlockId]) {
        self.blocks.extend_from_slice(block_ids);
    }

    /// Advance fill by `n` tokens (after writing to cache).
    pub fn advance(&mut self, n: usize) {
        self.num_tokens_stored += n;
    }

    /// Compute slot_mapping for a range of token positions.
    /// Returns physical slot IDs for positions [start_pos..start_pos + n).
    pub fn slot_mapping(&self, start_pos: usize, n: usize) -> Vec<usize> {
        (start_pos..start_pos + n)
            .map(|pos| {
                let block_idx = pos / self.block_size;
                let offset = pos % self.block_size;
                self.blocks[block_idx] * self.block_size + offset
            })
            .collect()
    }

    /// Get the ordered list of physical block IDs.
    pub fn block_ids(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Release all blocks, returning their IDs for freeing.
    pub fn release(&mut self) -> Vec<BlockId> {
        self.num_tokens_stored = 0;
        std::mem::take(&mut self.blocks)
    }

    /// Reclaim leading blocks that are entirely outside a sliding window.
    ///
    /// Replaces the first `count` blocks with `NULL_BLOCK` and returns
    /// the old (non-null) block IDs for freeing. Already-null blocks are skipped.
    ///
    /// This preserves the block table's positional structure (length, token count)
    /// so slot_mapping for active positions remains correct.
    pub fn reclaim_leading_blocks(&mut self, count: usize) -> Vec<BlockId> {
        let count = count.min(self.blocks.len());
        let mut freed = Vec::new();
        for block in self.blocks[..count].iter_mut() {
            if *block != NULL_BLOCK {
                freed.push(*block);
                *block = NULL_BLOCK;
            }
        }
        freed
    }

    /// Get block IDs excluding null blocks (for attention backends that need clean lists).
    pub fn active_block_ids(&self) -> Vec<BlockId> {
        self.blocks
            .iter()
            .copied()
            .filter(|&id| id != NULL_BLOCK)
            .collect()
    }

    /// Number of leading null blocks (already reclaimed).
    pub fn num_null_blocks(&self) -> usize {
        self.blocks.iter().take_while(|&&id| id == NULL_BLOCK).count()
    }

    /// Trim the block table to hold exactly `target_tokens` tokens.
    /// Returns block IDs that are no longer needed (caller must free them).
    /// Panics if `target_tokens > num_tokens_stored`.
    pub fn trim_to(&mut self, target_tokens: usize) -> Vec<BlockId> {
        assert!(
            target_tokens <= self.num_tokens_stored,
            "cannot trim to {target_tokens}, only {} stored",
            self.num_tokens_stored
        );
        let blocks_required = if target_tokens == 0 {
            0
        } else {
            target_tokens.div_ceil(self.block_size)
        };
        self.num_tokens_stored = target_tokens;
        if self.blocks.len() > blocks_required {
            self.blocks.split_off(blocks_required)
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table() {
        let table = BlockTable::new(16);
        assert_eq!(table.num_tokens(), 0);
        assert!(table.block_ids().is_empty());
    }

    #[test]
    fn blocks_needed_first_token() {
        let table = BlockTable::new(16);
        assert_eq!(table.blocks_needed(1), 1);
    }

    #[test]
    fn blocks_needed_within_block() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[0]);
        table.advance(5);
        // 11 more tokens fit in the current block
        assert_eq!(table.blocks_needed(11), 0);
    }

    #[test]
    fn blocks_needed_cross_boundary() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[0]);
        table.advance(15);
        // 1 more fits, but 2 needs a new block
        assert_eq!(table.blocks_needed(1), 0);
        assert_eq!(table.blocks_needed(2), 1);
    }

    #[test]
    fn blocks_needed_multiple_blocks() {
        let table = BlockTable::new(16);
        // 33 tokens need ceil(33/16) = 3 blocks
        assert_eq!(table.blocks_needed(33), 3);
    }

    #[test]
    fn slot_mapping_sequential() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[5]); // physical block 5
        let slots = table.slot_mapping(0, 16);
        let expected: Vec<usize> = (0..16).map(|i| 5 * 16 + i).collect();
        assert_eq!(slots, expected);
    }

    #[test]
    fn slot_mapping_cross_block() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[3, 7]); // two blocks
        let slots = table.slot_mapping(14, 4); // positions 14,15,16,17
        assert_eq!(
            slots,
            vec![
                3 * 16 + 14, // pos 14 → block 3, offset 14
                3 * 16 + 15, // pos 15 → block 3, offset 15
                7 * 16,      // pos 16 → block 7, offset 0
                7 * 16 + 1,  // pos 17 → block 7, offset 1
            ]
        );
    }

    #[test]
    fn advance_updates_fill() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[0]);
        table.advance(5);
        assert_eq!(table.num_tokens(), 5);
    }

    #[test]
    fn advance_across_blocks() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[0, 1]);
        table.advance(20);
        assert_eq!(table.num_tokens(), 20);
    }

    #[test]
    fn release_returns_all() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[2, 5, 9]);
        table.advance(40);
        let released = table.release();
        assert_eq!(released, vec![2, 5, 9]);
        assert_eq!(table.num_tokens(), 0);
        assert!(table.block_ids().is_empty());
    }

    #[test]
    fn trim_to_within_same_block() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[3, 7]);
        table.advance(20);
        let freed = table.trim_to(18);
        assert!(freed.is_empty());
        assert_eq!(table.num_tokens(), 18);
        assert_eq!(table.block_ids(), &[3, 7]);
    }

    #[test]
    fn trim_to_frees_trailing_block() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[3, 7, 12]);
        table.advance(35);
        let freed = table.trim_to(20);
        assert_eq!(freed, vec![12]);
        assert_eq!(table.num_tokens(), 20);
        assert_eq!(table.block_ids(), &[3, 7]);
    }

    #[test]
    fn trim_to_frees_multiple_blocks() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[0, 1, 2, 3, 4]);
        table.advance(18);
        let freed = table.trim_to(5);
        assert_eq!(freed, vec![2, 3, 4]);
        assert_eq!(table.num_tokens(), 5);
        assert_eq!(table.block_ids(), &[0, 1]);
    }

    #[test]
    fn trim_to_zero_frees_all() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[5, 9]);
        table.advance(20);
        let freed = table.trim_to(0);
        assert_eq!(freed, vec![5, 9]);
        assert_eq!(table.num_tokens(), 0);
        assert!(table.block_ids().is_empty());
    }

    #[test]
    fn trim_to_same_is_noop() {
        let mut table = BlockTable::new(16);
        table.append_blocks(&[2]);
        table.advance(10);
        let freed = table.trim_to(10);
        assert!(freed.is_empty());
        assert_eq!(table.num_tokens(), 10);
    }

    // ---- null block / reclaim tests ----

    #[test]
    fn reclaim_leading_blocks_basic() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[10, 20, 30, 40]);
        table.advance(16);

        let freed = table.reclaim_leading_blocks(2);
        assert_eq!(freed, vec![10, 20]);
        assert_eq!(table.block_ids(), &[NULL_BLOCK, NULL_BLOCK, 30, 40]);
        assert_eq!(table.num_tokens(), 16); // unchanged
    }

    #[test]
    fn reclaim_leading_blocks_idempotent() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[10, 20, 30]);
        table.advance(12);

        // First reclaim
        let freed1 = table.reclaim_leading_blocks(2);
        assert_eq!(freed1, vec![10, 20]);

        // Second reclaim on same range — already null
        let freed2 = table.reclaim_leading_blocks(2);
        assert!(freed2.is_empty());
    }

    #[test]
    fn reclaim_leading_blocks_exceeds_length() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[5, 6]);
        table.advance(8);

        let freed = table.reclaim_leading_blocks(10); // more than block count
        assert_eq!(freed, vec![5, 6]);
        assert_eq!(table.block_ids(), &[NULL_BLOCK, NULL_BLOCK]);
    }

    #[test]
    fn reclaim_leading_blocks_zero_is_noop() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[1, 2]);
        table.advance(8);

        let freed = table.reclaim_leading_blocks(0);
        assert!(freed.is_empty());
        assert_eq!(table.block_ids(), &[1, 2]);
    }

    #[test]
    fn active_block_ids_filters_null() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[10, 20, 30]);
        table.advance(12);

        table.reclaim_leading_blocks(1);
        assert_eq!(table.active_block_ids(), vec![20, 30]);
    }

    #[test]
    fn num_null_blocks_counts_leading() {
        let mut table = BlockTable::new(4);
        table.append_blocks(&[1, 2, 3, 4]);
        table.advance(16);

        assert_eq!(table.num_null_blocks(), 0);

        table.reclaim_leading_blocks(2);
        assert_eq!(table.num_null_blocks(), 2);

        table.reclaim_leading_blocks(3);
        assert_eq!(table.num_null_blocks(), 3);
    }
}
