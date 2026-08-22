use super::{
    ChatMessage, ChatRole, VendorDecodePolicy, VendorMultimodalPolicy, VendorTokenizerPolicy,
    qwen_common,
};
use crate::engine::types::Tokenizer;

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: false,
        stop_token_literals: qwen_common::QWEN_STOP_TOKEN_LITERALS,
        stop_text_literals: qwen_common::QWEN_STOP_TEXT_LITERALS,
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
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = if system_prompt.is_empty() {
        "You are a helpful assistant."
    } else {
        system_prompt
    };

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

    if tokenizer.bos_token >= 0 && !tokenizer.skip_bos_token {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        tokens.push(start);
        tokenizer.bpe_encode("system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("user\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(prompt, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        return tokens;
    }

    let rendered = format!(
        "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    tokenizer.bpe_encode(&rendered, &mut tokens);
    tokens
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = if system_prompt.is_empty() {
        "You are a helpful assistant."
    } else {
        system_prompt
    };

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

    if tokenizer.bos_token >= 0 && !tokenizer.skip_bos_token {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        tokens.push(start);
        tokenizer.bpe_encode("system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        for message in messages {
            tokens.push(start);
            let role = match message.role {
                ChatRole::User => "user\n",
                ChatRole::Assistant => "assistant\n",
            };
            tokenizer.bpe_encode(role, &mut temp);
            tokens.extend_from_slice(&temp);
            tokenizer.bpe_encode(&message.content, &mut temp);
            tokens.extend_from_slice(&temp);
            tokens.push(end);
            tokenizer.bpe_encode("\n", &mut temp);
            tokens.extend_from_slice(&temp);
        }

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        return tokens;
    }

    let mut rendered = format!("<|im_start|>system\n{sys}<|im_end|>\n");
    for message in messages {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        rendered.push_str(&format!(
            "<|im_start|>{role}\n{}<|im_end|>\n",
            message.content
        ));
    }
    rendered.push_str("<|im_start|>assistant\n");
    tokenizer.bpe_encode(&rendered, &mut tokens);
    tokens
}
