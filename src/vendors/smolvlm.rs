#![allow(dead_code)]

use super::{
    ChatMessage, ChatRole, MmprojFilenameScoreHint, VendorDecodePolicy, VendorMultimodalPolicy,
    VendorRuntimeDebugPolicy, VendorTokenizerPolicy,
};
use crate::engine::types::{
    ContentPart, EncodedPrompt, GenerationRequest, MultimodalBackend, PlaceholderSpan, Tokenizer,
};

static SMOLVLM_STOP_TOKEN_LITERALS: &[&str] = &["<end_of_utterance>"];
static SMOLVLM_END_TURN_TOKEN_LITERALS: &[&str] = &["<end_of_utterance>"];

static SMOLVLM_MMPROJ_SCORE_HINTS: &[MmprojFilenameScoreHint] = &[MmprojFilenameScoreHint {
    token: "smolvlm",
    backend: MultimodalBackend::Idefics3,
    match_score: 10,
    mismatch_score: -20,
}];

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: false,
        stop_token_literals: SMOLVLM_STOP_TOKEN_LITERALS,
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
        end_turn_token_literals: SMOLVLM_END_TURN_TOKEN_LITERALS,
    }
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy {
        image_prompt_suffix: "",
        detail_crop: Default::default(),
        mmproj_filename_score_hints: SMOLVLM_MMPROJ_SCORE_HINTS,
        missing_sidecar_hint: " hint: SmolVLM image inputs require a matching mmproj sidecar (mmproj-*.gguf) from the same checkpoint.",
    }
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    VendorRuntimeDebugPolicy::default()
}

fn push_im_start(tokenizer: &mut Tokenizer, tokens: &mut Vec<i32>, temp: &mut Vec<i32>) {
    if let Some(tok) = tokenizer.find_special_token("<|im_start|>") {
        tokens.push(tok);
    } else {
        tokenizer.bpe_encode("<|im_start|>", temp);
        tokens.extend_from_slice(temp);
    }
}

fn push_eou(tokenizer: &mut Tokenizer, tokens: &mut Vec<i32>, temp: &mut Vec<i32>) {
    if let Some(tok) = tokenizer.find_special_token("<end_of_utterance>") {
        tokens.push(tok);
    } else {
        tokenizer.bpe_encode("<end_of_utterance>", temp);
        tokens.extend_from_slice(temp);
    }
}

fn image_token_id(tokenizer: &mut Tokenizer, temp: &mut Vec<i32>) -> i32 {
    if let Some(tok) = tokenizer.find_special_token("<image>") {
        return tok;
    }
    tokenizer.bpe_encode("<image>", temp);
    *temp.first().unwrap_or(&0)
}

fn fake_image_token_id(tokenizer: &mut Tokenizer, temp: &mut Vec<i32>) -> i32 {
    if let Some(tok) = tokenizer.find_special_token("<fake_token_around_image>") {
        return tok;
    }
    tokenizer.bpe_encode("<fake_token_around_image>", temp);
    *temp.first().unwrap_or(&0)
}

fn global_img_token_id(tokenizer: &mut Tokenizer, temp: &mut Vec<i32>) -> i32 {
    if let Some(tok) = tokenizer.find_special_token("<global-img>") {
        return tok;
    }
    tokenizer.bpe_encode("<global-img>", temp);
    *temp.first().unwrap_or(&0)
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(256);
    let mut temp: Vec<i32> = Vec::with_capacity(64);

    push_im_start(tokenizer, &mut tokens, &mut temp);

    let system = system_prompt.trim();
    if !system.is_empty() {
        tokenizer.bpe_encode(system, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("\n\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    tokenizer.bpe_encode("User: ", &mut temp);
    tokens.extend_from_slice(&temp);
    tokenizer.bpe_encode(prompt, &mut temp);
    tokens.extend_from_slice(&temp);
    push_eou(tokenizer, &mut tokens, &mut temp);
    tokenizer.bpe_encode("\nAssistant:", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(512);
    let mut temp: Vec<i32> = Vec::with_capacity(64);

    push_im_start(tokenizer, &mut tokens, &mut temp);

    let system = system_prompt.trim();
    if !system.is_empty() {
        tokenizer.bpe_encode(system, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("\n\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    for message in messages {
        let role = match message.role {
            ChatRole::User => "User",
            ChatRole::Assistant => "Assistant",
        };
        tokenizer.bpe_encode(&format!("{role}: "), &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(&message.content, &mut temp);
        tokens.extend_from_slice(&temp);
        push_eou(tokenizer, &mut tokens, &mut temp);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    tokenizer.bpe_encode("Assistant:", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
) -> EncodedPrompt {
    let mut tokens: Vec<i32> = Vec::with_capacity(256);
    let mut temp: Vec<i32> = Vec::with_capacity(64);
    let mut image_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut image_index = 0usize;

    push_im_start(tokenizer, &mut tokens, &mut temp);

    let system = request.system_prompt.trim();
    if !system.is_empty() {
        tokenizer.bpe_encode(system, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("\n\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    // SmolVLM chat template: "User:" (no space) if first part is image, else "User: "
    let first_is_image = request
        .parts
        .first()
        .map(|p| matches!(p, ContentPart::Image(_)))
        .unwrap_or(false);
    tokenizer.bpe_encode(if first_is_image { "User:" } else { "User: " }, &mut temp);
    tokens.extend_from_slice(&temp);

    let img_tok = image_token_id(tokenizer, &mut temp);
    let fake_tok = fake_image_token_id(tokenizer, &mut temp);
    let global_img_tok = global_img_token_id(tokenizer, &mut temp);

    for part in &request.parts {
        match part {
            ContentPart::Text(text) => {
                tokenizer.bpe_encode(text, &mut temp);
                tokens.extend_from_slice(&temp);
            }
            ContentPart::Image(_) => {
                // SmolVLM idefics3 image wrapping (non-tiled / single global image):
                //   <fake_token_around_image><global-img><image>×N<fake_token_around_image>
                // The <image> marker is replaced by N embedding tokens via PlaceholderSpan.
                tokens.push(fake_tok);
                tokens.push(global_img_tok);
                let token_start = tokens.len();
                tokens.push(img_tok);
                image_spans.push(PlaceholderSpan {
                    token_start,
                    token_len: 1,
                    media_index: image_index,
                    replace_marker: true,
                });
                tokens.push(fake_tok);
                image_index += 1;
            }
            ContentPart::Video(_) | ContentPart::Audio(_) => {}
        }
    }

    push_eou(tokenizer, &mut tokens, &mut temp);
    tokenizer.bpe_encode("\nAssistant:", &mut temp);
    tokens.extend_from_slice(&temp);

    EncodedPrompt {
        token_ids: tokens,
        image_spans,
        video_spans: Vec::new(),
        audio_spans: Vec::new(),
    }
}
