// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use super::{
    ChatMessage, VendorDecodePolicy, VendorMultimodalPolicy, VendorRuntimeDebugPolicy,
    VendorTokenizerPolicy, qwen_common,
};
use crate::engine::types::{Config, EncodedPrompt, GenerationRequest, ThinkMode, Tokenizer};

pub(super) fn decode_policy(_config: &Config) -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: true,
        stop_token_literals: qwen_common::QWEN_STOP_TOKEN_LITERALS,
        stop_text_literals: qwen_common::QWEN_STOP_TEXT_LITERALS,
        deterministic_loop_guard: false,
        deterministic_loop_guard_min_generated_tokens: 0,
        unconditional_loop_guard: true,
        recover_early_endoftext_once: false,
        early_endoftext_recover_max_tokens: 0,
        hidden_think_token_cap_base: 320,
        visible_think_token_cap_base: 224,
        prefer_hidden_think_for_multimodal: true,
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
    VendorMultimodalPolicy {
        mmproj_filename_score_hints: qwen_common::QWEN_MMPROJ_SCORE_HINTS,
        ..VendorMultimodalPolicy::default()
    }
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    qwen_common::runtime_debug_policy()
}

pub(super) fn configure_rope(config: &mut Config) -> Result<(), String> {
    qwen_common::configure_interleaved_rope(config, [24, 20, 20, 0])
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
