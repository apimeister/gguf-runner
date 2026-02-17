use crate::engine::types::{Config, Tokenizer};

fn find_special_token_any(tokenizer: &Tokenizer, candidates: &[&str]) -> Option<i32> {
    for candidate in candidates {
        if let Some(tok) = tokenizer.find_special_token(candidate) {
            return Some(tok);
        }
    }
    None
}

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

pub(super) fn encode_deepseek_chat(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    if tokenizer.bos_token >= 0 {
        tokens.push(tokenizer.bos_token);
    }

    if !system_prompt.is_empty() {
        tokenizer.bpe_encode(system_prompt, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("\n\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    if let Some(user_tok) = find_special_token_any(
        tokenizer,
        &["<｜User｜>", "<|User|>", "<｜user｜>", "<|user|>"],
    ) {
        tokens.push(user_tok);
    } else {
        tokenizer.bpe_encode("<｜User｜>", &mut temp);
        tokens.extend_from_slice(&temp);
    }
    tokenizer.bpe_encode(prompt, &mut temp);
    tokens.extend_from_slice(&temp);

    if let Some(assistant_tok) = find_special_token_any(
        tokenizer,
        &[
            "<｜Assistant｜>",
            "<|Assistant|>",
            "<｜assistant｜>",
            "<|assistant|>",
        ],
    ) {
        tokens.push(assistant_tok);
    } else {
        tokenizer.bpe_encode("<｜Assistant｜>", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    // DeepSeek-V3 chat templates commonly append </think> before assistant output
    // when "thinking" mode is disabled.
    if let Some(close_think_tok) =
        find_special_token_any(tokenizer, &["</think>", "<|/think|>", "<｜/think｜>"])
    {
        tokens.push(close_think_tok);
    } else {
        tokenizer.bpe_encode("</think>", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    tokens
}
