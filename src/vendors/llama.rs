// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use super::{
    ChatMessage, ChatRole, VendorDecodePolicy, VendorMultimodalPolicy, VendorRuntimeDebugPolicy,
};
use crate::engine::types::{
    LLAMA3_BOS_TOKEN, LLAMA3_END_HEADER, LLAMA3_EOT, LLAMA3_START_HEADER, Tokenizer,
    VendorTokenizerPolicy,
};

pub(super) fn default_rope_theta() -> f32 {
    500_000.0
}

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: false,
        stop_token_literals: &[],
        stop_text_literals: &[],
        deterministic_loop_guard: false,
        deterministic_loop_guard_min_generated_tokens: 0,
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
    VendorTokenizerPolicy::default()
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy::default()
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    VendorRuntimeDebugPolicy::default()
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    let bos = tokenizer
        .find_special_token("<|begin_of_text|>")
        .unwrap_or(LLAMA3_BOS_TOKEN);
    let start_header = tokenizer
        .find_special_token("<|start_header_id|>")
        .unwrap_or(LLAMA3_START_HEADER);
    let end_header = tokenizer
        .find_special_token("<|end_header_id|>")
        .unwrap_or(LLAMA3_END_HEADER);
    let eot = tokenizer
        .find_special_token("<|eot_id|>")
        .unwrap_or(LLAMA3_EOT);

    tokens.push(bos);

    if !system_prompt.is_empty() {
        tokens.push(start_header);
        tokenizer.bpe_encode("system", &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end_header);
        tokenizer.bpe_encode(&format!("\n\n{}", system_prompt), &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(eot);
    }

    tokens.push(start_header);
    tokenizer.bpe_encode("user", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode(&format!("\n\n{}", prompt), &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(eot);

    tokens.push(start_header);
    tokenizer.bpe_encode("assistant", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode("\n\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    let bos = tokenizer
        .find_special_token("<|begin_of_text|>")
        .unwrap_or(LLAMA3_BOS_TOKEN);
    let start_header = tokenizer
        .find_special_token("<|start_header_id|>")
        .unwrap_or(LLAMA3_START_HEADER);
    let end_header = tokenizer
        .find_special_token("<|end_header_id|>")
        .unwrap_or(LLAMA3_END_HEADER);
    let eot = tokenizer
        .find_special_token("<|eot_id|>")
        .unwrap_or(LLAMA3_EOT);

    tokens.push(bos);

    if !system_prompt.is_empty() {
        tokens.push(start_header);
        tokenizer.bpe_encode("system", &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end_header);
        tokenizer.bpe_encode(&format!("\n\n{}", system_prompt), &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(eot);
    }

    for message in messages {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        tokens.push(start_header);
        tokenizer.bpe_encode(role, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end_header);
        tokenizer.bpe_encode(&format!("\n\n{}", message.content), &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(eot);
    }

    tokens.push(start_header);
    tokenizer.bpe_encode("assistant", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode("\n\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}
