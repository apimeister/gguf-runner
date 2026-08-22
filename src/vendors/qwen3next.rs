use super::{
    ChatMessage, VendorDecodePolicy, VendorMultimodalPolicy, VendorRuntimeDebugPolicy,
    VendorTokenizerPolicy, qwen_common,
};
use crate::engine::types::{Config, EncodedPrompt, GenerationRequest, ThinkMode, Tokenizer};

pub(super) fn validate_qwen3next(config: &mut Config) -> Result<(), String> {
    // Qwen3Next uses normalized top-k expert weights with softmax gating.
    config.moe_norm_topk_prob = true;
    config.moe_routed_scaling_factor = 1.0;

    if config.ssm_conv_kernel == 0
        || config.ssm_inner_size == 0
        || config.ssm_state_size == 0
        || config.ssm_time_step_rank == 0
        || config.ssm_group_count == 0
    {
        return Err(
            "qwen3next model is missing SSM metadata (ssm.conv_kernel/inner_size/state_size/time_step_rank/group_count)"
                .to_string(),
        );
    }

    if !config
        .ssm_inner_size
        .is_multiple_of(config.ssm_time_step_rank)
    {
        return Err(format!(
            "qwen3next invalid SSM metadata: inner_size {} not divisible by time_step_rank {}",
            config.ssm_inner_size, config.ssm_time_step_rank
        ));
    }
    if !config
        .ssm_time_step_rank
        .is_multiple_of(config.ssm_group_count)
    {
        return Err(format!(
            "qwen3next invalid SSM metadata: time_step_rank {} not divisible by group_count {}",
            config.ssm_time_step_rank, config.ssm_group_count
        ));
    }

    let head_v_dim = config.ssm_inner_size / config.ssm_time_step_rank;
    if head_v_dim != config.ssm_state_size {
        return Err(format!(
            "qwen3next unsupported SSM shape: state_size {} != inner_size/time_step_rank {}",
            config.ssm_state_size, head_v_dim
        ));
    }

    Ok(())
}

pub(super) fn print_qwen3next_debug(config: &Config) {
    eprintln!(
        "Qwen3Next: experts={}, experts_used={}, expert_hidden_dim={}, shared_expert_hidden_dim={}, ssm_inner={}, ssm_state={}, ssm_heads={}, ssm_groups={}, ssm_conv_kernel={}, rms_norm_eps={}",
        config.n_experts,
        config.n_experts_used,
        config.expert_hidden_dim,
        config.shared_expert_hidden_dim,
        config.ssm_inner_size,
        config.ssm_state_size,
        config.ssm_time_step_rank,
        config.ssm_group_count,
        config.ssm_conv_kernel,
        config.rms_norm_eps
    );
}

pub(super) fn decode_policy(config: &Config) -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: config.qwen_chat_template_contains_think,
        stop_token_literals: qwen_common::QWEN_STOP_TOKEN_LITERALS,
        stop_text_literals: qwen_common::QWEN_STOP_TEXT_LITERALS,
        deterministic_loop_guard: true,
        deterministic_loop_guard_min_generated_tokens: 192,
        unconditional_loop_guard: true,
        recover_early_endoftext_once: true,
        early_endoftext_recover_max_tokens: 192,
        hidden_think_token_cap_base: 384,
        visible_think_token_cap_base: 384,
        prefer_hidden_think_for_multimodal: false,
        retry_without_think_when_no_post_think_text: false,
        // Greedy JSON-only decoding is brittle on Qwen3Next/Qwen3.6 and can
        // collapse into malformed repeated prefixes like {"type":"{"{"...
        // Use the generic low-temperature agent profile instead.
        agent_force_deterministic: false,
        agent_protocol_max_failures: 3,
        agent_plain_chat_fallback_after_protocol_failures: true,
    }
}

pub(super) fn tokenizer_policy() -> VendorTokenizerPolicy {
    VendorTokenizerPolicy {
        disable_bos_fallback: true,
        end_turn_token_literals: qwen_common::QWEN_END_TURN_TOKEN_LITERALS,
    }
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy::default()
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    qwen_common::runtime_debug_policy()
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    config: &Config,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
    let effective_system_prompt = if config.qwen_chat_template_has_builtin_system
        && system_prompt.trim() == DEFAULT_SYSTEM_PROMPT
    {
        ""
    } else {
        system_prompt
    };

    if config.qwen_chat_template_contains_think {
        qwen_common::encode_qwen3_chat(
            tokenizer,
            prompt,
            effective_system_prompt,
            image_count,
            think_mode,
        )
    } else {
        qwen_common::encode_qwen3_chat_no_forced_think(
            tokenizer,
            prompt,
            effective_system_prompt,
            image_count,
            think_mode,
        )
    }
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    config: &Config,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
    let effective_system_prompt = if config.qwen_chat_template_has_builtin_system
        && system_prompt.trim() == DEFAULT_SYSTEM_PROMPT
    {
        ""
    } else {
        system_prompt
    };

    if config.qwen_chat_template_contains_think {
        qwen_common::encode_qwen3_messages(tokenizer, messages, effective_system_prompt, think_mode)
    } else {
        qwen_common::encode_qwen3_messages_no_forced_think(
            tokenizer,
            messages,
            effective_system_prompt,
            think_mode,
        )
    }
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    config: &Config,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
    let mut effective_request = request.clone();
    if config.qwen_chat_template_has_builtin_system
        && effective_request.system_prompt.trim() == DEFAULT_SYSTEM_PROMPT
    {
        effective_request.system_prompt.clear();
    }
    if config.qwen_chat_template_contains_think {
        qwen_common::encode_qwen3_request(tokenizer, &effective_request, think_mode)
    } else {
        qwen_common::encode_qwen3_request_no_forced_think(tokenizer, &effective_request, think_mode)
    }
}
