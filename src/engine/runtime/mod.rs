mod inference;
mod parallel;

use crate::engine::types::Config;

pub(crate) use inference::{malloc_run_state, transformer};
pub(crate) use parallel::configure_rayon_threads;

pub(crate) fn apply_context_size_overrides(
    config: &mut Config,
    context_size: usize,
    debug_mode: bool,
) {
    if context_size > 0 {
        config.seq_len = context_size;
    } else if config.is_qwen3moe || config.is_qwen3next || config.is_deepseek2 {
        if debug_mode {
            eprintln!(
                "Using model native context length {} (model may require a large workspace)",
                config.seq_len
            );
        }
    }
}
