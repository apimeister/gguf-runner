mod inference;
mod parallel;

use crate::engine::types::Config;

pub(crate) use inference::{
    malloc_run_state, transformer, transformer_with_embedding,
    transformer_with_embedding_with_moe_executor, transformer_with_embedding_without_logits,
    transformer_with_embedding_without_logits_with_moe_executor, transformer_with_moe_executor,
    transformer_without_logits, transformer_without_logits_with_moe_executor,
};
pub(crate) use parallel::configure_rayon_threads;

pub(crate) fn apply_context_size_overrides(
    config: &mut Config,
    context_size: usize,
    _debug_mode: bool,
) {
    if context_size > 0 {
        config.seq_len = context_size;
    }
}
