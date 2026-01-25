use super::error::CacheError;

pub type BlockId = usize;

/// Manages physical block allocation. Pure bookkeeping, no GPU awareness.
pub struct BlockPool {
    num_blocks: usize,
    free_list: Vec<BlockId>,
    allocated: Vec<bool>,
}

impl BlockPool {
    pub fn new(num_blocks: usize) -> Self {
        // LIFO: push 0..num_blocks so that pop gives low IDs first
        let free_list: Vec<BlockId> = (0..num_blocks).rev().collect();
        Self {
            num_blocks,
            free_list,
            allocated: vec![false; num_blocks],
        }
    }

    /// Allocate n blocks. Returns Err if insufficient free blocks.
    pub fn allocate(&mut self, n: usize) -> Result<Vec<BlockId>, CacheError> {
        if n > self.free_list.len() {
            return Err(CacheError::OutOfBlocks {
                requested: n,
                available: self.free_list.len(),
            });
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_list.pop().expect("checked above");
            self.allocated[id] = true;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Free a set of blocks back to the pool.
    pub fn free(&mut self, blocks: &[BlockId]) -> Result<(), CacheError> {
        for &id in blocks {
            if id >= self.num_blocks || !self.allocated[id] {
                return Err(CacheError::BlockNotAllocated { block_id: id });
            }
            self.allocated[id] = false;
            self.free_list.push(id);
        }
        Ok(())
    }

    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    pub fn num_total(&self) -> usize {
        self.num_blocks
    }

    #[cfg(test)]
    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_all_free() {
        let pool = BlockPool::new(64);
        assert_eq!(pool.num_free(), 64);
        assert_eq!(pool.num_used(), 0);
    }

    #[test]
    fn allocate_reduces_free() {
        let mut pool = BlockPool::new(64);
        let ids = pool.allocate(4).unwrap();
        assert_eq!(ids.len(), 4);
        assert_eq!(pool.num_free(), 60);
        assert_eq!(pool.num_used(), 4);
    }

    #[test]
    fn allocate_returns_unique_ids() {
        let mut pool = BlockPool::new(64);
        let ids = pool.allocate(10).unwrap();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
    }

    #[test]
    fn free_increases_free() {
        let mut pool = BlockPool::new(64);
        let ids = pool.allocate(4).unwrap();
        pool.free(&ids).unwrap();
        assert_eq!(pool.num_free(), 64);
        assert_eq!(pool.num_used(), 0);
    }

    #[test]
    fn allocate_oom_returns_error() {
        let mut pool = BlockPool::new(4);
        let result = pool.allocate(5);
        assert!(result.is_err());
        match result.unwrap_err() {
            CacheError::OutOfBlocks {
                requested,
                available,
            } => {
                assert_eq!(requested, 5);
                assert_eq!(available, 4);
            }
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn double_free_returns_error() {
        let mut pool = BlockPool::new(8);
        let ids = pool.allocate(2).unwrap();
        pool.free(&ids).unwrap();
        let result = pool.free(&ids);
        assert!(result.is_err());
        match result.unwrap_err() {
            CacheError::BlockNotAllocated { .. } => {}
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn allocate_after_free_reuses_blocks() {
        let mut pool = BlockPool::new(4);
        let ids1 = pool.allocate(4).unwrap();
        pool.free(&ids1).unwrap();
        let ids2 = pool.allocate(4).unwrap();
        assert_eq!(ids2.len(), 4);
    }
}
