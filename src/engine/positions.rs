use crate::engine::types::{RopePositionLayout, RopePositionPlan};

impl RopePositionPlan {
    pub(crate) fn append_text(&mut self, count: usize) -> Result<(), String> {
        let end = self
            .next_text_position
            .checked_add(count)
            .ok_or_else(|| "text rotary position overflow".to_string())?;
        self.positions
            .extend((self.next_text_position..end).map(|position| [position; 3]));
        self.next_text_position = end;
        Ok(())
    }

    /// Grid dimensions are after spatial merging; tokens are in T/H/W order.
    pub(crate) fn append_grid(
        &mut self,
        grid: [usize; 3],
        token_count: usize,
    ) -> Result<(), String> {
        let [temporal, height, width] = grid;
        let count = temporal
            .checked_mul(height)
            .and_then(|count| count.checked_mul(width))
            .ok_or_else(|| "image rotary grid size overflow".to_string())?;
        if count == 0 || count != token_count {
            return Err(format!(
                "image rotary grid {grid:?} contains {count} positions for {token_count} embeddings"
            ));
        }
        let base = self.next_text_position;
        let next = base
            .checked_add(temporal.max(height).max(width))
            .ok_or_else(|| "image rotary position overflow".to_string())?;
        for t in 0..temporal {
            for h in 0..height {
                for w in 0..width {
                    self.positions.push([base + t, base + h, base + w]);
                }
            }
        }
        self.next_text_position = next;
        Ok(())
    }

    pub(crate) fn at(&self, physical_position: usize) -> [usize; 3] {
        self.positions
            .get(physical_position)
            .copied()
            .unwrap_or_else(|| {
                [self.next_text_position + (physical_position - self.positions.len()); 3]
            })
    }
}

impl RopePositionLayout {
    pub(crate) fn axis(self, frequency: usize, sections: [usize; 4]) -> usize {
        match self {
            Self::Sequential => 0,
            Self::Interleaved => {
                let axis = frequency % 3;
                // H and W occupy every third frequency up to their section
                // lengths. Remaining frequencies retain the temporal axis.
                if axis != 0 && frequency / 3 < sections[axis] {
                    axis
                } else {
                    0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::types::{RopePositionLayout, RopePositionPlan};

    #[test]
    fn rectangular_grid_and_following_text_match_reference_positions() {
        // Literal get_rope_index fixture: two text tokens, a merged 1x2x3
        // image, then two text tokens. Transformers v5.2.0 Qwen3-VL.
        let mut plan = RopePositionPlan::default();
        plan.append_text(2).unwrap();
        plan.append_grid([1, 2, 3], 6).unwrap();
        plan.append_text(2).unwrap();
        assert_eq!(
            plan.positions,
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4],
                [5, 5, 5],
                [6, 6, 6],
            ]
        );
        assert_eq!(plan.at(10), [7; 3]);
        assert_eq!(plan.at(12), [9; 3]);
        // An appended think-close suffix is ordinary text in the same plan.
        plan.append_text(3).unwrap();
        assert_eq!(plan.at(12), [9; 3]);
        assert_eq!(plan.at(13), [10; 3]);
    }

    #[test]
    fn interleaved_axes_match_reference_section_slices() {
        let layout = RopePositionLayout::Interleaved;
        // v5.2.0 Qwen3-VL: H is [1:60:3], W is [2:60:3],
        // and the four remaining frequencies use T.
        let axes: Vec<_> = (0..64).map(|i| layout.axis(i, [24, 20, 20, 0])).collect();
        assert_eq!(&axes[..9], &[0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(&axes[57..], &[0, 1, 2, 0, 0, 0, 0]);
        // v5.3.0 Qwen3.5: H has 11 entries, W has 10.
        let axes: Vec<_> = (0..32).map(|i| layout.axis(i, [11, 11, 10, 0])).collect();
        assert_eq!(&axes[27..], &[0, 1, 2, 0, 1]);
        assert!((0..64).all(|i| RopePositionLayout::Sequential.axis(i, [0; 4]) == 0));
    }

    #[test]
    fn invalid_grids_fail_before_modifying_positions() {
        for (grid, count) in [([0, 2, 3], 0), ([1, 2, 3], 5), ([usize::MAX, 2, 2], 1)] {
            let mut plan = RopePositionPlan::default();
            assert!(plan.append_grid(grid, count).is_err());
            assert_eq!(plan, RopePositionPlan::default());
        }
        let mut plan = RopePositionPlan {
            positions: Vec::new(),
            next_text_position: usize::MAX,
        };
        assert!(plan.append_grid([1, 1, 1], 1).is_err());
        assert!(plan.append_text(1).is_err());
    }
}
