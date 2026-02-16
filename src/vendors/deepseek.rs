use crate::engine::types::Config;

pub(super) fn validate_deepseek2(config: &mut Config) -> Result<(), String> {
    if config.deepseek_q_lora_rank == 0 || config.deepseek_kv_lora_rank == 0 {
        return Err(
            "deepseek2 model is missing MLA metadata (attention.q_lora_rank/kv_lora_rank)"
                .to_string(),
        );
    }
    if config.deepseek_qk_rope_head_dim == 0 || config.deepseek_qk_nope_head_dim == 0 {
        return Err(
            "deepseek2 invalid MLA head dims: rope/nope dimensions must both be > 0".to_string(),
        );
    }
    if config.deepseek_v_head_dim == 0 {
        return Err(
            "deepseek2 invalid MLA value head dim: value_length_mla must be > 0".to_string(),
        );
    }
    if config.deepseek_leading_dense_block_count > config.n_layers {
        return Err(format!(
            "deepseek2 invalid leading_dense_block_count {} for n_layers {}",
            config.deepseek_leading_dense_block_count, config.n_layers
        ));
    }

    // DeepSeek-MLA materializes per-head K/V projections in this runtime path.
    config.n_kv_heads = config.n_heads;
    Ok(())
}

pub(super) fn print_deepseek2_debug(config: &Config) {
    eprintln!(
        "DeepSeek2: q_lora_rank={}, kv_lora_rank={}, qk_nope_head_dim={}, qk_rope_head_dim={}, v_head_dim={}, leading_dense_blocks={}, experts={}, experts_used={}, expert_hidden_dim={}, shared_expert_hidden_dim={}",
        config.deepseek_q_lora_rank,
        config.deepseek_kv_lora_rank,
        config.deepseek_qk_nope_head_dim,
        config.deepseek_qk_rope_head_dim,
        config.deepseek_v_head_dim,
        config.deepseek_leading_dense_block_count,
        config.n_experts,
        config.n_experts_used,
        config.expert_hidden_dim,
        config.shared_expert_hidden_dim
    );
}
