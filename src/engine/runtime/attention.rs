//! Validated image-block metadata and physical attention visibility.

use crate::engine::types::{
    ImageAttentionMode, LanguageAttentionPolicy, MediaAttentionBlock, MediaAttentionPlan,
};
use std::ops::Range;

impl MediaAttentionPlan {
    pub(crate) fn new(
        prompt_tokens: usize,
        blocks: Vec<MediaAttentionBlock>,
    ) -> Result<Self, String> {
        let mut previous_end = 0;
        for block in &blocks {
            let end = block
                .token_start
                .checked_add(block.token_len)
                .ok_or_else(|| "image attention block range overflow".to_string())?;
            if block.token_len == 0 || block.token_start < previous_end || end > prompt_tokens {
                return Err("image attention blocks must be nonempty, ordered, disjoint, and within the prompt".to_string());
            }
            previous_end = end;
        }
        Ok(Self {
            prompt_tokens,
            blocks,
        })
    }

    pub(crate) fn block_at(&self, position: usize) -> Option<MediaAttentionBlock> {
        let index = self
            .blocks
            .partition_point(|block| block.token_start <= position);
        index
            .checked_sub(1)
            .and_then(|index| self.blocks.get(index))
            .copied()
            .filter(|block| position - block.token_start < block.token_len)
    }

    /// Adjust an ordinary chunk boundary so it never splits a complete image
    /// block. A block larger than the requested chunk is processed as one unit.
    pub(crate) fn chunk_end(
        &self,
        start: usize,
        limit: usize,
        chunk: usize,
    ) -> Result<usize, String> {
        if chunk == 0 || start >= limit || limit > self.prompt_tokens {
            return Err("invalid media prefill chunk range or size".to_string());
        }
        if let Some(block) = self.block_at(start) {
            if start != block.token_start {
                return Err("media prefill cannot resume inside an image block".to_string());
            }
            let end = block.token_start + block.token_len;
            if end > limit {
                return Err("media prefill limit splits an image block".to_string());
            }
            return Ok(end);
        }
        let end = start.saturating_add(chunk).min(limit);
        // Stop text before an image; evaluating the following image as its own
        // unit bounds the resident layer inputs to one view.
        Ok(self
            .blocks
            .iter()
            .find(|block| block.token_start >= start && block.token_start < end)
            .map_or(end, |block| block.token_start))
    }
}

impl LanguageAttentionPolicy {
    pub(crate) fn visible_keys(
        &self,
        plan: &MediaAttentionPlan,
        layer: usize,
        query: usize,
    ) -> Range<usize> {
        let end = query + 1;
        let start = self
            .layer_windows
            .get(layer)
            .copied()
            .flatten()
            .map_or(0, |window| end.saturating_sub(window));
        if self.image_mode == ImageAttentionMode::Bidirectional
            && let Some(block) = plan.block_at(query)
        {
            // The reference ORs the image group with the local causal mask:
            // every key in this block is visible, including those outside the
            // local window. Previous/future blocks receive no such override.
            return start.min(block.token_start)..block.token_start + block.token_len;
        }
        start..end
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::types::{LanguageAttentionPolicy, MediaAttentionBlock, MediaAttentionPlan};

    #[test]
    fn image_attention_chunks_never_split_blocks() {
        let plan = MediaAttentionPlan::new(
            12,
            vec![
                MediaAttentionBlock {
                    token_start: 2,
                    token_len: 5,
                    media_index: 0,
                },
                MediaAttentionBlock {
                    token_start: 8,
                    token_len: 4,
                    media_index: 1,
                },
            ],
        )
        .unwrap();
        for chunk in [1, 2, 3, 4, 8, usize::MAX] {
            let mut start = 0;
            while start < 12 {
                let end = plan.chunk_end(start, 12, chunk).unwrap();
                assert!(end > start);
                if let Some(block) = plan.block_at(start) {
                    assert_eq!(
                        (start, end),
                        (block.token_start, block.token_start + block.token_len)
                    );
                } else {
                    assert!((start..end).all(|position| plan.block_at(position).is_none()));
                }
                start = end;
            }
        }
        assert!(plan.chunk_end(3, 12, 2).is_err());
        assert!(plan.chunk_end(2, 6, 2).is_err());
        assert!(plan.chunk_end(0, 12, 0).is_err());
        assert!(plan.chunk_end(0, 13, 1).is_err());
        for blocks in [
            vec![MediaAttentionBlock {
                token_start: 1,
                token_len: 0,
                media_index: 0,
            }],
            vec![MediaAttentionBlock {
                token_start: 11,
                token_len: 2,
                media_index: 0,
            }],
            vec![MediaAttentionBlock {
                token_start: usize::MAX,
                token_len: 1,
                media_index: 0,
            }],
            vec![
                MediaAttentionBlock {
                    token_start: 2,
                    token_len: 5,
                    media_index: 0,
                },
                MediaAttentionBlock {
                    token_start: 6,
                    token_len: 2,
                    media_index: 1,
                },
            ],
        ] {
            assert!(MediaAttentionPlan::new(12, blocks).is_err());
        }
        let causal = LanguageAttentionPolicy::default();
        assert_eq!(causal.visible_keys(&plan, 0, 3), 0..4);
    }
}
