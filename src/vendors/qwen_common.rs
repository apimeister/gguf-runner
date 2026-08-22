// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.
#![allow(dead_code)]

use super::{ChatMessage, ChatRole, MmprojFilenameScoreHint, VendorRuntimeDebugPolicy};
use crate::engine::types::{
    ContentPart, EncodedPrompt, GenerationRequest, MediaRef, MultimodalBackend, PlaceholderSpan,
    ThinkMode, Tokenizer,
};

pub(crate) const QWEN_STOP_TOKEN_LITERALS: &[&str] =
    &["<|im_end|>", "<|endoftext|>", "<|im_start|>"];
pub(crate) const QWEN_STOP_TEXT_LITERALS: &[&str] = &[
    "</response>",
    "</assistant_response>",
    "</user_request>",
    "</assistant>",
    "</user>",
    "</system>",
];
pub(crate) const QWEN_END_TURN_TOKEN_LITERALS: &[&str] = &["<|im_end|>", "<|endoftext|>"];
pub(crate) const QWEN_MMPROJ_SCORE_HINTS: &[MmprojFilenameScoreHint] = &[
    MmprojFilenameScoreHint {
        token: "qwen3vl",
        backend: MultimodalBackend::Qwen3Vl,
        match_score: 100,
        mismatch_score: -100,
    },
    MmprojFilenameScoreHint {
        token: "qwen35",
        backend: MultimodalBackend::Qwen35,
        match_score: 100,
        mismatch_score: -100,
    },
];

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    VendorRuntimeDebugPolicy {
        native_context_label: Some("qwen3"),
    }
}

fn encode_qwen3_chat_with_think_style(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
    inject_forced_think_prompt: bool,
) -> Vec<i32> {
    let mut parts = Vec::with_capacity(1 + image_count);
    parts.push(ContentPart::Text(prompt.to_string()));
    for _ in 0..image_count {
        parts.push(ContentPart::Image(MediaRef {
            path: String::new(),
        }));
    }
    let request = GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts,
        include_empty_system_prompt: false,
        assistant_prefill: None,
    };
    encode_qwen3_request_with_think_style(
        tokenizer,
        &request,
        think_mode,
        inject_forced_think_prompt,
    )
    .token_ids
}

pub(super) fn encode_qwen3_chat(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    encode_qwen3_chat_with_think_style(
        tokenizer,
        prompt,
        system_prompt,
        image_count,
        think_mode,
        true,
    )
}

pub(super) fn encode_qwen3_chat_no_forced_think(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    encode_qwen3_chat_with_think_style(
        tokenizer,
        prompt,
        system_prompt,
        image_count,
        think_mode,
        false,
    )
}

pub(super) fn encode_qwen3_messages_with_think_style(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
    inject_forced_think_prompt: bool,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = system_prompt.trim();

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

    if tokenizer.bos_token >= 0 && !tokenizer.skip_bos_token {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        if !sys.is_empty() {
            tokens.push(start);
            tokenizer.bpe_encode("system\n", &mut temp);
            tokens.extend_from_slice(&temp);
            tokenizer.bpe_encode(sys, &mut temp);
            tokens.extend_from_slice(&temp);
            tokens.push(end);
            tokenizer.bpe_encode("\n", &mut temp);
            tokens.extend_from_slice(&temp);
        }

        for message in messages {
            tokens.push(start);
            let role = match message.role {
                ChatRole::User => "user\n",
                ChatRole::Assistant => "assistant\n",
            };
            tokenizer.bpe_encode(role, &mut temp);
            tokens.extend_from_slice(&temp);
            // For Qwen3-style models that pre-fill an empty think block in the
            // assistant prefix, past assistant messages must also include that
            // empty block — otherwise the KV-cached representation of past turns
            // looks different from what was actually generated, which confuses
            // the model on subsequent turns.
            if matches!(message.role, ChatRole::Assistant)
                && inject_forced_think_prompt
                && think_mode == ThinkMode::No
                && !message.content.contains("</think>")
            {
                tokenizer.bpe_encode("<think>\n\n</think>\n\n", &mut temp);
                tokens.extend_from_slice(&temp);
            }
            tokenizer.bpe_encode(&message.content, &mut temp);
            tokens.extend_from_slice(&temp);
            tokens.push(end);
            tokenizer.bpe_encode("\n", &mut temp);
            tokens.extend_from_slice(&temp);
        }

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        if inject_forced_think_prompt {
            let think_prefix = if think_mode == ThinkMode::No {
                "<think>\n\n</think>\n\n"
            } else {
                "<think>\n"
            };
            tokenizer.bpe_encode(think_prefix, &mut temp);
            tokens.extend_from_slice(&temp);
        }
        return tokens;
    }

    if !sys.is_empty() {
        tokenizer.bpe_encode("<|im_start|>system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("<|im_end|>\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    for message in messages {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        tokenizer.bpe_encode(&format!("<|im_start|>{role}\n"), &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(&message.content, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("<|im_end|>\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }
    let assistant_suffix = if !inject_forced_think_prompt {
        "<|im_start|>assistant\n"
    } else if think_mode == ThinkMode::No {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    } else {
        "<|im_start|>assistant\n<think>\n"
    };
    tokenizer.bpe_encode(assistant_suffix, &mut temp);
    tokens.extend_from_slice(&temp);
    tokens
}

pub(super) fn encode_qwen3_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    encode_qwen3_messages_with_think_style(tokenizer, messages, system_prompt, think_mode, true)
}

pub(super) fn encode_qwen3_messages_no_forced_think(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    encode_qwen3_messages_with_think_style(tokenizer, messages, system_prompt, think_mode, false)
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

fn append_qwen_assistant_prefill(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    prefill: &str,
) {
    const ASR_TEXT_TAG: &str = "<asr_text>";

    let Some((before_tag, after_tag)) = prefill.split_once(ASR_TEXT_TAG) else {
        tokenizer.bpe_encode(prefill, temp);
        tokens.extend_from_slice(temp);
        return;
    };

    tokenizer.bpe_encode(before_tag, temp);
    tokens.extend_from_slice(temp);
    if let Some(asr_text) = tokenizer.find_special_token(ASR_TEXT_TAG) {
        tokens.push(asr_text);
    } else {
        tokenizer.bpe_encode(ASR_TEXT_TAG, temp);
        tokens.extend_from_slice(temp);
    }
    tokenizer.bpe_encode(after_tag, temp);
    tokens.extend_from_slice(temp);
}

fn append_vision_wrapped_placeholder(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    vision_start: Option<i32>,
    pad_token: Option<i32>,
    vision_end: Option<i32>,
    pad_literal: &str,
) -> (usize, usize) {
    if let (Some(vs), Some(pad), Some(ve)) = (vision_start, pad_token, vision_end) {
        let start = tokens.len();
        tokens.push(vs);
        tokens.push(pad);
        tokens.push(ve);
        (start, 3)
    } else {
        let rendered = format!("<|vision_start|>{pad_literal}<|vision_end|>");
        append_encoded_literal(tokenizer, temp, tokens, &rendered)
    }
}

fn append_audio_placeholder(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    audio_start: Option<i32>,
    audio_pad: Option<i32>,
    audio_end: Option<i32>,
) -> (usize, usize) {
    if let (Some(start_token), Some(pad_token), Some(end_token)) =
        (audio_start, audio_pad, audio_end)
    {
        let start = tokens.len();
        tokens.push(start_token);
        tokens.push(pad_token);
        tokens.push(end_token);
        return (start, 3);
    }
    append_encoded_literal(
        tokenizer,
        temp,
        tokens,
        "<|audio_start|><|audio_pad|><|audio_end|>",
    )
}

fn encode_qwen3_request_with_think_style(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
    inject_forced_think_prompt: bool,
) -> EncodedPrompt {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let mut image_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut video_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut audio_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut audio_index = 0usize;
    let sys = request.system_prompt.trim();

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");
    let vision_start = tokenizer.find_special_token("<|vision_start|>");
    let vision_end = tokenizer.find_special_token("<|vision_end|>");
    let image_pad = tokenizer.find_special_token("<|image_pad|>");
    let video_pad = tokenizer.find_special_token("<|video_pad|>");
    let audio_start = tokenizer.find_special_token("<|audio_start|>");
    let audio_pad = tokenizer.find_special_token("<|audio_pad|>");
    let audio_end = tokenizer.find_special_token("<|audio_end|>");

    if tokenizer.bos_token >= 0 && !tokenizer.skip_bos_token {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        if !sys.is_empty() || request.include_empty_system_prompt {
            tokens.push(start);
            tokenizer.bpe_encode("system\n", &mut temp);
            tokens.extend_from_slice(&temp);
            tokenizer.bpe_encode(sys, &mut temp);
            tokens.extend_from_slice(&temp);
            tokens.push(end);
            tokenizer.bpe_encode("\n", &mut temp);
            tokens.extend_from_slice(&temp);
        }

        tokens.push(start);
        tokenizer.bpe_encode("user\n", &mut temp);
        tokens.extend_from_slice(&temp);
        for part in &request.parts {
            match part {
                ContentPart::Text(text) => {
                    tokenizer.bpe_encode(text, &mut temp);
                    tokens.extend_from_slice(&temp);
                }
                ContentPart::Image(_) => {
                    let (token_start, token_len) = append_vision_wrapped_placeholder(
                        tokenizer,
                        &mut temp,
                        &mut tokens,
                        vision_start,
                        image_pad,
                        vision_end,
                        "<|image_pad|>",
                    );
                    image_spans.push(PlaceholderSpan {
                        token_start,
                        token_len,
                        media_index: image_index,
                        replace_marker: false,
                    });
                    image_index += 1;
                }
                ContentPart::Video(_) => {
                    let (token_start, token_len) = append_vision_wrapped_placeholder(
                        tokenizer,
                        &mut temp,
                        &mut tokens,
                        vision_start,
                        video_pad,
                        vision_end,
                        "<|video_pad|>",
                    );
                    video_spans.push(PlaceholderSpan {
                        token_start,
                        token_len,
                        media_index: video_index,
                        replace_marker: false,
                    });
                    video_index += 1;
                }
                ContentPart::Audio(_) => {
                    let (token_start, token_len) = append_audio_placeholder(
                        tokenizer,
                        &mut temp,
                        &mut tokens,
                        audio_start,
                        audio_pad,
                        audio_end,
                    );
                    audio_spans.push(PlaceholderSpan {
                        token_start,
                        token_len,
                        media_index: audio_index,
                        replace_marker: false,
                    });
                    audio_index += 1;
                }
            }
        }
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        if inject_forced_think_prompt && request.assistant_prefill.is_none() {
            let think_prefix = if think_mode == ThinkMode::No {
                "<think>\n\n</think>\n\n"
            } else {
                "<think>\n"
            };
            tokenizer.bpe_encode(think_prefix, &mut temp);
            tokens.extend_from_slice(&temp);
        }
        if let Some(prefill) = &request.assistant_prefill {
            append_qwen_assistant_prefill(tokenizer, &mut temp, &mut tokens, prefill);
        }

        return EncodedPrompt {
            token_ids: tokens,
            image_spans,
            video_spans,
            audio_spans,
        };
    }

    if !sys.is_empty() || request.include_empty_system_prompt {
        tokenizer.bpe_encode("<|im_start|>system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("<|im_end|>\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    tokenizer.bpe_encode("<|im_start|>user\n", &mut temp);
    tokens.extend_from_slice(&temp);
    for part in &request.parts {
        match part {
            ContentPart::Text(text) => {
                tokenizer.bpe_encode(text, &mut temp);
                tokens.extend_from_slice(&temp);
            }
            ContentPart::Image(_) => {
                let (token_start, token_len) = append_vision_wrapped_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    vision_start,
                    image_pad,
                    vision_end,
                    "<|image_pad|>",
                );
                image_spans.push(PlaceholderSpan {
                    token_start,
                    token_len,
                    media_index: image_index,
                    replace_marker: false,
                });
                image_index += 1;
            }
            ContentPart::Video(_) => {
                let (token_start, token_len) = append_vision_wrapped_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    vision_start,
                    video_pad,
                    vision_end,
                    "<|video_pad|>",
                );
                video_spans.push(PlaceholderSpan {
                    token_start,
                    token_len,
                    media_index: video_index,
                    replace_marker: false,
                });
                video_index += 1;
            }
            ContentPart::Audio(_) => {
                let (token_start, token_len) = append_audio_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    audio_start,
                    audio_pad,
                    audio_end,
                );
                audio_spans.push(PlaceholderSpan {
                    token_start,
                    token_len,
                    media_index: audio_index,
                    replace_marker: false,
                });
                audio_index += 1;
            }
        }
    }
    let assistant_suffix = if !inject_forced_think_prompt || request.assistant_prefill.is_some() {
        "<|im_end|>\n<|im_start|>assistant\n"
    } else if think_mode == ThinkMode::No {
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    } else {
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    };
    tokenizer.bpe_encode(assistant_suffix, &mut temp);
    tokens.extend_from_slice(&temp);
    if let Some(prefill) = &request.assistant_prefill {
        append_qwen_assistant_prefill(tokenizer, &mut temp, &mut tokens, prefill);
    }

    EncodedPrompt {
        token_ids: tokens,
        image_spans,
        video_spans,
        audio_spans,
    }
}

pub(super) fn encode_qwen3_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    encode_qwen3_request_with_think_style(tokenizer, request, think_mode, true)
}

pub(super) fn encode_qwen3_request_no_forced_think(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    encode_qwen3_request_with_think_style(tokenizer, request, think_mode, false)
}

#[cfg(test)]
mod tests {
    use super::encode_qwen3_request;
    use crate::engine::types::{ContentPart, GenerationRequest, MediaRef, ThinkMode, Tokenizer};

    fn tokenizer_with_qwen_specials() -> Tokenizer {
        Tokenizer {
            vocab: vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "<|vision_start|>".to_string(),
                "<|vision_end|>".to_string(),
                "<|image_pad|>".to_string(),
                "<|video_pad|>".to_string(),
                "<|audio_start|>".to_string(),
                "<|audio_pad|>".to_string(),
                "<|audio_end|>".to_string(),
                "<asr_text>".to_string(),
            ],
            ..Tokenizer::default()
        }
    }

    #[test]
    fn qwen3_request_maps_multimodal_placeholder_spans() {
        let mut tokenizer = tokenizer_with_qwen_specials();
        let request = GenerationRequest {
            system_prompt: String::new(),
            parts: vec![
                ContentPart::Text("analyze".to_string()),
                ContentPart::Image(MediaRef {
                    path: "a.png".to_string(),
                }),
                ContentPart::Video(MediaRef {
                    path: "b.mp4".to_string(),
                }),
                ContentPart::Audio(MediaRef {
                    path: "c.wav".to_string(),
                }),
            ],
            include_empty_system_prompt: false,
            assistant_prefill: None,
        };
        let encoded = encode_qwen3_request(&mut tokenizer, &request, ThinkMode::Yes);

        assert_eq!(encoded.image_spans.len(), 1);
        assert_eq!(encoded.video_spans.len(), 1);
        assert_eq!(encoded.audio_spans.len(), 1);

        let vision_start = tokenizer
            .find_special_token("<|vision_start|>")
            .expect("vision_start");
        let vision_end = tokenizer
            .find_special_token("<|vision_end|>")
            .expect("vision_end");
        let image_pad = tokenizer
            .find_special_token("<|image_pad|>")
            .expect("image_pad");
        let video_pad = tokenizer
            .find_special_token("<|video_pad|>")
            .expect("video_pad");
        let audio_pad = tokenizer
            .find_special_token("<|audio_pad|>")
            .expect("audio_pad");
        let audio_start = tokenizer
            .find_special_token("<|audio_start|>")
            .expect("audio_start");
        let audio_end = tokenizer
            .find_special_token("<|audio_end|>")
            .expect("audio_end");

        let image_span = encoded.image_spans[0];
        assert_eq!(image_span.token_len, 3);
        assert_eq!(
            &encoded.token_ids
                [image_span.token_start..image_span.token_start + image_span.token_len],
            &[vision_start, image_pad, vision_end]
        );

        let video_span = encoded.video_spans[0];
        assert_eq!(video_span.token_len, 3);
        assert_eq!(
            &encoded.token_ids
                [video_span.token_start..video_span.token_start + video_span.token_len],
            &[vision_start, video_pad, vision_end]
        );

        let audio_span = encoded.audio_spans[0];
        assert_eq!(audio_span.token_len, 3);
        assert_eq!(
            &encoded.token_ids
                [audio_span.token_start..audio_span.token_start + audio_span.token_len],
            &[audio_start, audio_pad, audio_end]
        );
    }

    #[test]
    fn qwen3_request_appends_asr_prefill_as_special_token_without_think_seed() {
        let mut tokenizer = tokenizer_with_qwen_specials();
        let request = GenerationRequest {
            system_prompt: String::new(),
            parts: vec![ContentPart::Audio(MediaRef {
                path: "speech.wav".to_string(),
            })],
            include_empty_system_prompt: true,
            assistant_prefill: Some("language English<asr_text>".to_string()),
        };

        let asr_text = tokenizer
            .find_special_token("<asr_text>")
            .expect("asr_text");
        tokenizer.bos_token = asr_text;
        tokenizer.skip_bos_token = true;

        let encoded = encode_qwen3_request(&mut tokenizer, &request, ThinkMode::Yes);
        let im_start = tokenizer
            .find_special_token("<|im_start|>")
            .expect("im_start");
        let im_end = tokenizer.find_special_token("<|im_end|>").expect("im_end");
        assert_eq!(&encoded.token_ids[..3], &[im_start, im_end, im_start]);
        assert_eq!(encoded.token_ids.last(), Some(&asr_text));
    }
}
