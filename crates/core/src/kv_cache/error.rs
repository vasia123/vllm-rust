use thiserror::Error;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("out of blocks: requested {requested}, available {available}")]
    OutOfBlocks { requested: usize, available: usize },

    #[error("block {block_id} is not allocated")]
    BlockNotAllocated { block_id: usize },

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_out_of_blocks() {
        let e = CacheError::OutOfBlocks {
            requested: 10,
            available: 3,
        };
        assert_eq!(e.to_string(), "out of blocks: requested 10, available 3");
    }

    #[test]
    fn error_display_block_not_allocated() {
        let e = CacheError::BlockNotAllocated { block_id: 42 };
        assert_eq!(e.to_string(), "block 42 is not allocated");
    }
}
