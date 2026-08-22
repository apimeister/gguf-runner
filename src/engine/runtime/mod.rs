mod inference;
mod parallel;

use crate::engine::types::Config;

pub(crate) use inference::{
    PrefillInput, PrefillScratch, batch_prefill_supported, malloc_run_state,
    malloc_run_state_with_kv_cache_format, transformer, transformer_prefill_batch,
    transformer_with_embedding, transformer_with_embedding_without_logits,
    transformer_without_logits,
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
