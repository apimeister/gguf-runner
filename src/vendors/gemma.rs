// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use super::{
    ChatMessage, ChatRole, MmprojFilenameScoreHint, VendorDecodePolicy, VendorMultimodalPolicy,
    VendorRuntimeDebugPolicy,
};
use crate::engine::types::{
    Config, ContentPart, EncodedPrompt, GEMMA3_BOS_TOKEN, GEMMA3_END_TURN, GEMMA3_START_TURN,
    GenerationRequest, MultimodalBackend, PlaceholderSpan, Tokenizer, VendorTokenizerPolicy,
};

const GEMMA_MMPROJ_SCORE_HINTS: &[MmprojFilenameScoreHint] = &[
    MmprojFilenameScoreHint {
        token: "gemma3",
        backend: MultimodalBackend::Gemma3,
        match_score: 100,
        mismatch_score: -100,
    },
    MmprojFilenameScoreHint {
        token: "gemma",
        backend: MultimodalBackend::Gemma3,
        match_score: 25,
        mismatch_score: -25,
    },
];

pub(super) fn default_rope_theta() -> f32 {
    1_000_000.0
}

pub(super) fn print_config_debug(config: &Config) {
    eprintln!(
        "Gemma3: rms_norm_eps={}, final_logit_softcapping={}",
        config.rms_norm_eps, config.final_logit_softcapping
    );
}

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: false,
        stop_token_literals: &["<end_of_turn>"],
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
    VendorTokenizerPolicy {
        disable_bos_fallback: false,
        end_turn_token_literals: &["<end_of_turn>"],
    }
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy {
        mmproj_filename_score_hints: GEMMA_MMPROJ_SCORE_HINTS,
        missing_sidecar_hint: " hint: Gemma3 image inputs require a compatible Gemma3 mmproj sidecar from the same checkpoint family.",
        ..VendorMultimodalPolicy::default()
    }
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    VendorRuntimeDebugPolicy::default()
}

fn append_encoded_literal(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    literal: &str,
) -> (usize, usize) {
    let start = tokens.len();
    tokenizer.bpe_encode(literal, temp);
    tokens.extend_from_slice(temp);
    (start, tokens.len().saturating_sub(start))
}

fn append_image_placeholder(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    image_index: usize,
    image_spans: &mut Vec<PlaceholderSpan>,
) {
    let image_start = tokenizer.find_special_token("<start_of_image>");
    let image_end = tokenizer.find_special_token("<end_of_image>");
    let (token_start, token_len) = if let (Some(start), Some(end)) = (image_start, image_end) {
        let start_idx = tokens.len();
        tokens.push(start);
        tokens.push(end);
        (start_idx, 2)
    } else {
        append_encoded_literal(tokenizer, temp, tokens, "<start_of_image><end_of_image>")
    };

    image_spans.push(PlaceholderSpan {
        token_start,
        token_len,
        media_index: image_index,
        replace_marker: false,
    });
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
) -> EncodedPrompt {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let mut image_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut image_index = 0usize;

    let bos_token = tokenizer
        .find_special_token("<bos>")
        .unwrap_or(GEMMA3_BOS_TOKEN);
    let start_turn = tokenizer
        .find_special_token("<start_of_turn>")
        .unwrap_or(GEMMA3_START_TURN);
    let end_turn = tokenizer
        .find_special_token("<end_of_turn>")
        .unwrap_or(GEMMA3_END_TURN);

    tokens.push(bos_token);
    tokens.push(start_turn);
    tokenizer.bpe_encode("user\n", &mut temp);
    tokens.extend_from_slice(&temp);

    let system_prompt = request.system_prompt.trim();
    if !system_prompt.is_empty() {
        tokenizer.bpe_encode(system_prompt, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("\n\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }
    for part in &request.parts {
        match part {
            ContentPart::Text(text) => {
                tokenizer.bpe_encode(text, &mut temp);
                tokens.extend_from_slice(&temp);
            }
            ContentPart::Image(_) => {
                append_image_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    image_index,
                    &mut image_spans,
                );
                image_index += 1;
            }
            ContentPart::Video(_) | ContentPart::Audio(_) => {}
        }
    }

    tokens.push(end_turn);
    tokenizer.bpe_encode("\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens.push(start_turn);
    tokenizer.bpe_encode("model\n", &mut temp);
    tokens.extend_from_slice(&temp);

    EncodedPrompt {
        token_ids: tokens,
        image_spans,
        video_spans: Vec::new(),
        audio_spans: Vec::new(),
    }
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    let request = GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts: vec![ContentPart::Text(prompt.to_string())],
        include_empty_system_prompt: false,
        assistant_prefill: None,
    };
    encode_generation_request(tokenizer, &request).token_ids
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    let bos_token = tokenizer
        .find_special_token("<bos>")
        .unwrap_or(GEMMA3_BOS_TOKEN);
    let start_turn = tokenizer
        .find_special_token("<start_of_turn>")
        .unwrap_or(GEMMA3_START_TURN);
    let end_turn = tokenizer
        .find_special_token("<end_of_turn>")
        .unwrap_or(GEMMA3_END_TURN);

    tokens.push(bos_token);

    let mut first_user_turn = true;
    for message in messages {
        tokens.push(start_turn);
        let role = match message.role {
            ChatRole::User => "user\n",
            ChatRole::Assistant => "model\n",
        };
        tokenizer.bpe_encode(role, &mut temp);
        tokens.extend_from_slice(&temp);
        if first_user_turn
            && matches!(message.role, ChatRole::User)
            && !system_prompt.trim().is_empty()
        {
            tokenizer.bpe_encode(system_prompt.trim(), &mut temp);
            tokens.extend_from_slice(&temp);
            tokenizer.bpe_encode("\n\n", &mut temp);
            tokens.extend_from_slice(&temp);
            first_user_turn = false;
        }
        tokenizer.bpe_encode(&message.content, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end_turn);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    tokens.push(start_turn);
    tokenizer.bpe_encode("model\n", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens
}

#[cfg(test)]
mod tests {
    use super::encode_generation_request;
    use crate::engine::types::{ContentPart, GenerationRequest, MediaRef, Tokenizer};

    fn tokenizer_with_gemma_specials() -> Tokenizer {
        Tokenizer {
            vocab: vec![
                "<bos>".to_string(),
                "<start_of_turn>".to_string(),
                "<end_of_turn>".to_string(),
                "<start_of_image>".to_string(),
                "<end_of_image>".to_string(),
            ],
            ..Tokenizer::default()
        }
    }

    #[test]
    fn gemma_request_maps_image_placeholder_span() {
        let mut tokenizer = tokenizer_with_gemma_specials();
        let request = GenerationRequest {
            system_prompt: String::new(),
            parts: vec![
                ContentPart::Text("describe".to_string()),
                ContentPart::Image(MediaRef {
                    path: "img.png".to_string(),
                }),
            ],
            include_empty_system_prompt: false,
            assistant_prefill: None,
        };

        let encoded = encode_generation_request(&mut tokenizer, &request);
        assert_eq!(encoded.image_spans.len(), 1);

        let start = tokenizer
            .find_special_token("<start_of_image>")
            .expect("start_of_image");
        let end = tokenizer
            .find_special_token("<end_of_image>")
            .expect("end_of_image");
        let span = encoded.image_spans[0];
        assert_eq!(encoded.token_ids[span.token_start], start);
        assert_eq!(encoded.token_ids[span.token_start + 1], end);
        assert_eq!(encoded.image_spans[0].token_len, 2);
    }
}
