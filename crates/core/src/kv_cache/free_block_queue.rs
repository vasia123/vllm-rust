//! O(1) free block queue using doubly-linked list.
//!
//! Unlike a simple Vec-based free list, this supports O(1) removal from
//! the middle of the queue (needed when a cached block is touched/reused).

use super::block_pool::BlockId;

/// Sentinel value for "no block"
const NONE: usize = usize::MAX;

/// Node in the doubly-linked list.
#[derive(Clone, Copy)]
struct Node {
    prev: usize,
    next: usize,
    /// Whether this node is currently in the queue
    in_queue: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            prev: NONE,
            next: NONE,
            in_queue: false,
        }
    }
}

/// O(1) free block queue with doubly-linked list.
///
/// Supports:
/// - `popleft()` - O(1) pop from front
/// - `append()` - O(1) append to back
/// - `remove()` - O(1) removal from anywhere (by block ID)
pub struct FreeBlockQueue {
    /// Per-block node data (indexed by BlockId)
    nodes: Vec<Node>,
    /// Index of first element (or NONE if empty)
    head: usize,
    /// Index of last element (or NONE if empty)
    tail: usize,
    /// Number of blocks in queue
    len: usize,
}

impl FreeBlockQueue {
    /// Create a new queue with all blocks initially free.
    ///
    /// Blocks are ordered by ID initially (0, 1, 2, ...).
    pub fn new(num_blocks: usize) -> Self {
        if num_blocks == 0 {
            return Self {
                nodes: Vec::new(),
                head: NONE,
                tail: NONE,
                len: 0,
            };
        }

        let mut nodes = vec![Node::default(); num_blocks];

        // Link all blocks in order
        for (i, node) in nodes.iter_mut().enumerate() {
            node.in_queue = true;
            node.prev = if i == 0 { NONE } else { i - 1 };
            node.next = if i == num_blocks - 1 { NONE } else { i + 1 };
        }

        Self {
            nodes,
            head: 0,
            tail: num_blocks - 1,
            len: num_blocks,
        }
    }

    /// Number of free blocks in the queue.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Pop the first free block from the front.
    ///
    /// Returns `None` if the queue is empty.
    pub fn popleft(&mut self) -> Option<BlockId> {
        if self.head == NONE {
            return None;
        }

        let block_id = self.head;
        self.remove_internal(block_id);
        Some(block_id)
    }

    /// Pop n blocks from the front.
    ///
    /// Returns fewer blocks if not enough available.
    pub fn popleft_n(&mut self, n: usize) -> Vec<BlockId> {
        let mut result = Vec::with_capacity(n.min(self.len));
        for _ in 0..n {
            match self.popleft() {
                Some(id) => result.push(id),
                None => break,
            }
        }
        result
    }

    /// Append a block to the back of the queue.
    ///
    /// # Panics
    /// Panics if the block is already in the queue.
    pub fn append(&mut self, block_id: BlockId) {
        assert!(
            block_id < self.nodes.len(),
            "block_id {} out of range",
            block_id
        );
        assert!(
            !self.nodes[block_id].in_queue,
            "block {} already in queue",
            block_id
        );

        let node = &mut self.nodes[block_id];
        node.in_queue = true;
        node.prev = self.tail;
        node.next = NONE;

        if self.tail != NONE {
            self.nodes[self.tail].next = block_id;
        } else {
            // Queue was empty
            self.head = block_id;
        }
        self.tail = block_id;
        self.len += 1;
    }

    /// Append multiple blocks to the back of the queue.
    pub fn append_n(&mut self, blocks: &[BlockId]) {
        for &block_id in blocks {
            self.append(block_id);
        }
    }

    /// Remove a specific block from anywhere in the queue.
    ///
    /// This is the key operation that makes this data structure useful:
    /// when a cached block is reused, it needs to be removed from the free queue.
    ///
    /// Returns `true` if the block was in the queue and removed.
    pub fn remove(&mut self, block_id: BlockId) -> bool {
        if block_id >= self.nodes.len() || !self.nodes[block_id].in_queue {
            return false;
        }
        self.remove_internal(block_id);
        true
    }

    /// Check if a block is currently in the queue.
    pub fn contains(&self, block_id: BlockId) -> bool {
        block_id < self.nodes.len() && self.nodes[block_id].in_queue
    }

    /// Internal removal logic.
    fn remove_internal(&mut self, block_id: BlockId) {
        let node = self.nodes[block_id];
        debug_assert!(node.in_queue);

        // Update prev's next pointer
        if node.prev != NONE {
            self.nodes[node.prev].next = node.next;
        } else {
            // Removing head
            self.head = node.next;
        }

        // Update next's prev pointer
        if node.next != NONE {
            self.nodes[node.next].prev = node.prev;
        } else {
            // Removing tail
            self.tail = node.prev;
        }

        // Clear node
        self.nodes[block_id].in_queue = false;
        self.nodes[block_id].prev = NONE;
        self.nodes[block_id].next = NONE;

        self.len -= 1;
    }

    /// Get all blocks in the queue (for testing).
    pub fn get_all(&self) -> Vec<BlockId> {
        let mut result = Vec::with_capacity(self.len);
        let mut current = self.head;
        while current != NONE {
            result.push(current);
            current = self.nodes[current].next;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_queue_has_all_blocks() {
        let q = FreeBlockQueue::new(4);
        assert_eq!(q.len(), 4);
        assert_eq!(q.get_all(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn empty_queue() {
        let q = FreeBlockQueue::new(0);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn popleft_returns_in_order() {
        let mut q = FreeBlockQueue::new(4);
        assert_eq!(q.popleft(), Some(0));
        assert_eq!(q.popleft(), Some(1));
        assert_eq!(q.popleft(), Some(2));
        assert_eq!(q.popleft(), Some(3));
        assert_eq!(q.popleft(), None);
        assert!(q.is_empty());
    }

    #[test]
    fn popleft_n() {
        let mut q = FreeBlockQueue::new(5);
        let blocks = q.popleft_n(3);
        assert_eq!(blocks, vec![0, 1, 2]);
        assert_eq!(q.len(), 2);
        assert_eq!(q.get_all(), vec![3, 4]);
    }

    #[test]
    fn popleft_n_more_than_available() {
        let mut q = FreeBlockQueue::new(2);
        let blocks = q.popleft_n(5);
        assert_eq!(blocks, vec![0, 1]);
        assert!(q.is_empty());
    }

    #[test]
    fn append_to_back() {
        let mut q = FreeBlockQueue::new(4);
        let _ = q.popleft(); // remove 0
        let _ = q.popleft(); // remove 1

        q.append(0);
        assert_eq!(q.get_all(), vec![2, 3, 0]);

        q.append(1);
        assert_eq!(q.get_all(), vec![2, 3, 0, 1]);
    }

    #[test]
    fn append_to_empty_queue() {
        let mut q = FreeBlockQueue::new(2);
        let _ = q.popleft();
        let _ = q.popleft();
        assert!(q.is_empty());

        q.append(1);
        assert_eq!(q.len(), 1);
        assert_eq!(q.get_all(), vec![1]);
    }

    #[test]
    fn append_n() {
        let mut q = FreeBlockQueue::new(4);
        // Pop all
        q.popleft_n(4);
        assert!(q.is_empty());

        q.append_n(&[3, 1, 2]);
        assert_eq!(q.get_all(), vec![3, 1, 2]);
    }

    #[test]
    fn remove_from_middle() {
        let mut q = FreeBlockQueue::new(5);
        // Queue: [0, 1, 2, 3, 4]

        assert!(q.remove(2));
        assert_eq!(q.get_all(), vec![0, 1, 3, 4]);
        assert_eq!(q.len(), 4);
    }

    #[test]
    fn remove_from_head() {
        let mut q = FreeBlockQueue::new(3);
        assert!(q.remove(0));
        assert_eq!(q.get_all(), vec![1, 2]);
    }

    #[test]
    fn remove_from_tail() {
        let mut q = FreeBlockQueue::new(3);
        assert!(q.remove(2));
        assert_eq!(q.get_all(), vec![0, 1]);
    }

    #[test]
    fn remove_only_element() {
        let mut q = FreeBlockQueue::new(1);
        assert!(q.remove(0));
        assert!(q.is_empty());
        assert_eq!(q.head, NONE);
        assert_eq!(q.tail, NONE);
    }

    #[test]
    fn remove_not_in_queue() {
        let mut q = FreeBlockQueue::new(3);
        let _ = q.popleft(); // remove 0

        assert!(!q.remove(0)); // already removed
        assert!(!q.remove(99)); // out of range
    }

    #[test]
    fn contains() {
        let mut q = FreeBlockQueue::new(3);
        assert!(q.contains(0));
        assert!(q.contains(1));
        assert!(q.contains(2));
        assert!(!q.contains(99));

        let _ = q.popleft();
        assert!(!q.contains(0));
        assert!(q.contains(1));
    }

    #[test]
    fn mixed_operations() {
        let mut q = FreeBlockQueue::new(5);
        // Start: [0, 1, 2, 3, 4]

        // Pop 2 from front
        assert_eq!(q.popleft(), Some(0));
        assert_eq!(q.popleft(), Some(1));
        // Now: [2, 3, 4]

        // Remove from middle
        q.remove(3);
        // Now: [2, 4]

        // Append freed blocks
        q.append(1);
        q.append(0);
        // Now: [2, 4, 1, 0]

        assert_eq!(q.get_all(), vec![2, 4, 1, 0]);
        assert_eq!(q.len(), 4);
    }

    #[test]
    #[should_panic(expected = "already in queue")]
    fn double_append_panics() {
        let mut q = FreeBlockQueue::new(3);
        let _ = q.popleft();
        q.append(0);
        q.append(0); // panic!
    }

    #[test]
    fn stress_test() {
        let n = 1000;
        let mut q = FreeBlockQueue::new(n);

        // Pop all
        let all = q.popleft_n(n);
        assert_eq!(all.len(), n);
        assert!(q.is_empty());

        // Append all in reverse
        for id in (0..n).rev() {
            q.append(id);
        }
        assert_eq!(q.len(), n);

        // Remove even numbers from middle
        for id in (0..n).step_by(2) {
            q.remove(id);
        }
        assert_eq!(q.len(), n / 2);

        // Verify only odd numbers remain
        let remaining = q.get_all();
        assert!(remaining.iter().all(|&id| id % 2 == 1));
    }
}
