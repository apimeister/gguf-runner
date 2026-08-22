// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use super::{
    ChatMessage, VendorDecodePolicy, VendorMultimodalPolicy, VendorTokenizerPolicy, qwen_common,
};
use crate::engine::types::{Config, EncodedPrompt, GenerationRequest, ThinkMode, Tokenizer};

pub(super) fn finalize_moe_config(config: &mut Config) -> Result<(), String> {
    if config.expert_hidden_dim == 0 || config.n_experts == 0 {
        return Err(
            "qwen model is missing expert metadata (expert_count/expert_feed_forward_length)"
                .to_string(),
        );
    }

    if config.n_experts_used == 0 {
        config.n_experts_used = 1;
    }
    if config.n_experts_used > config.n_experts {
        config.n_experts_used = config.n_experts;
    }

    Ok(())
}

pub(super) fn apply_qwen3moe_defaults(config: &mut Config) {
    // Qwen3 MoE routing defaults from official config.json.
    config.moe_n_group = 8;
    config.moe_topk_group = 4;
    config.moe_norm_topk_prob = true;
    config.moe_routed_scaling_factor = 1.0;
}

pub(super) fn print_qwen3moe_debug(config: &Config) {
    eprintln!(
        "Qwen3MoE: experts={}, experts_used={}, n_group={}, topk_group={}, norm_topk_prob={}, routed_scaling_factor={}, rms_norm_eps={}",
        config.n_experts,
        config.n_experts_used,
        config.moe_n_group,
        config.moe_topk_group,
        config.moe_norm_topk_prob,
        config.moe_routed_scaling_factor,
        config.rms_norm_eps
    );
}

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: true,
        stop_token_literals: qwen_common::QWEN_STOP_TOKEN_LITERALS,
        stop_text_literals: qwen_common::QWEN_STOP_TEXT_LITERALS,
        deterministic_loop_guard: true,
        deterministic_loop_guard_min_generated_tokens: 96,
        unconditional_loop_guard: true,
        recover_early_endoftext_once: false,
        early_endoftext_recover_max_tokens: 0,
        hidden_think_token_cap_base: 256,
        visible_think_token_cap_base: 256,
        prefer_hidden_think_for_multimodal: false,
        retry_without_think_when_no_post_think_text: false,
        agent_force_deterministic: false,
        agent_protocol_max_failures: 3,
        agent_plain_chat_fallback_after_protocol_failures: false,
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

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    qwen_common::encode_qwen3_chat(tokenizer, prompt, system_prompt, image_count, think_mode)
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    qwen_common::encode_qwen3_messages(tokenizer, messages, system_prompt, think_mode)
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    qwen_common::encode_qwen3_request(tokenizer, request, think_mode)
}
