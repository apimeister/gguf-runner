use super::{
    PreparedAudioTranscriptionRequest, VendorAudioTranscriptionPolicy, VendorDecodePolicy,
};
use crate::engine::types::{
    AudioTranscriptionResult, ContentPart, GenerationRequest, KvCacheFormat, MediaRef,
};

const ASR_TEXT_TAG: &str = "<asr_text>";
const LANGUAGE_PREFIX: &str = "language ";
const STOP_TOKEN_LITERALS: &[&str] = &["<|im_end|>"];
const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
];

pub(super) fn transcription_policy() -> VendorAudioTranscriptionPolicy {
    VendorAudioTranscriptionPolicy {
        decode_policy: VendorDecodePolicy {
            parse_think_tags: false,
            stop_token_literals: STOP_TOKEN_LITERALS,
            stop_text_literals: &[],
            deterministic_loop_guard: false,
            deterministic_loop_guard_min_generated_tokens: 0,
            // Speech repeats. A transcript that says the same phrase three
            // times is correct output, so the generic repetition guards must
            // not cut it short; `normalize_repetitions` handles the degenerate
            // decoder loops instead.
            unconditional_loop_guard: false,
            recover_early_endoftext_once: false,
            early_endoftext_recover_max_tokens: 0,
            hidden_think_token_cap_base: 0,
            visible_think_token_cap_base: 0,
            prefer_hidden_think_for_multimodal: false,
            retry_without_think_when_no_post_think_text: false,
            agent_force_deterministic: true,
            agent_protocol_max_failures: 0,
            agent_plain_chat_fallback_after_protocol_failures: false,
        },
        // TurboQuant KV compression destroys the low-norm projected audio
        // signal during text-model prefill. Q8 reproduces the reference
        // transcript while retaining a quantized cache.
        kv_cache_format: KvCacheFormat::Q8,
        max_new_tokens: 512,
        build_request,
        parse_output,
    }
}

fn normalize_forced_language(language: Option<&str>) -> Result<Option<String>, String> {
    let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let Some(canonical) = SUPPORTED_LANGUAGES
        .iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(language))
    else {
        return Err(format!(
            "unsupported Qwen3-ASR language '{language}'; supported languages: {}",
            SUPPORTED_LANGUAGES.join(", ")
        ));
    };
    Ok(Some((*canonical).to_string()))
}

fn build_request(
    audio_path: &str,
    context: &str,
    language: Option<&str>,
) -> Result<PreparedAudioTranscriptionRequest, String> {
    if audio_path.is_empty() {
        return Err("audio transcription path must not be empty".to_string());
    }
    let forced_language = normalize_forced_language(language)?;
    let assistant_prefill = forced_language
        .as_ref()
        .map_or_else(String::new, |language| {
            format!("{LANGUAGE_PREFIX}{language}{ASR_TEXT_TAG}")
        });
    Ok(PreparedAudioTranscriptionRequest {
        generation_request: GenerationRequest {
            system_prompt: context.to_string(),
            parts: vec![ContentPart::Audio(MediaRef {
                path: audio_path.to_string(),
            })],
            include_empty_system_prompt: true,
            // Qwen3-ASR uses the plain ChatML assistant prefix. `Some("")`
            // is intentionally distinct from the normal Qwen think prefix.
            assistant_prefill: Some(assistant_prefill),
        },
        forced_language,
    })
}

fn normalize_language_name(language: &str) -> String {
    let mut chars = language.trim().chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first
        .to_uppercase()
        .chain(chars.flat_map(char::to_lowercase))
        .collect()
}

fn collapse_long_character_runs(chars: &[char], threshold: usize) -> Vec<char> {
    let mut output = Vec::with_capacity(chars.len());
    let mut index = 0usize;
    while index < chars.len() {
        let mut end = index + 1;
        while end < chars.len() && chars[end] == chars[index] {
            end += 1;
        }
        if end - index > threshold {
            output.push(chars[index]);
        } else {
            output.extend_from_slice(&chars[index..end]);
        }
        index = end;
    }
    output
}

fn collapse_long_pattern_runs(
    chars: &[char],
    threshold: usize,
    max_pattern_len: usize,
) -> Vec<char> {
    let min_repeat_chars = threshold.saturating_mul(2);
    if chars.len() < min_repeat_chars {
        return chars.to_vec();
    }

    let mut output = Vec::with_capacity(chars.len());
    let mut index = 0usize;
    while index <= chars.len() - min_repeat_chars {
        let mut matched = None;
        for pattern_len in 1..=max_pattern_len {
            let Some(required_end) = pattern_len
                .checked_mul(threshold)
                .and_then(|length| index.checked_add(length))
            else {
                break;
            };
            if required_end > chars.len() {
                break;
            }
            let pattern = &chars[index..index + pattern_len];
            if (1..threshold).all(|repeat| {
                let start = index + repeat * pattern_len;
                &chars[start..start + pattern_len] == pattern
            }) {
                matched = Some((pattern_len, required_end));
                break;
            }
        }

        if let Some((pattern_len, mut end)) = matched {
            let pattern = &chars[index..index + pattern_len];
            while end + pattern_len <= chars.len() && &chars[end..end + pattern_len] == pattern {
                end += pattern_len;
            }
            output.extend_from_slice(pattern);
            output.extend(collapse_long_pattern_runs(
                &chars[end..],
                threshold,
                max_pattern_len,
            ));
            return output;
        }

        output.push(chars[index]);
        index += 1;
    }
    output.extend_from_slice(&chars[index..]);
    output
}

fn normalize_repetitions(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let chars = collapse_long_character_runs(&chars, 20);
    collapse_long_pattern_runs(&chars, 20, 20)
        .into_iter()
        .collect()
}

fn parse_output(raw: &str, forced_language: Option<&str>) -> AudioTranscriptionResult {
    let raw_output = raw.to_string();
    let normalized = normalize_repetitions(raw.trim());
    if normalized.is_empty() {
        return AudioTranscriptionResult {
            raw_output,
            language: String::new(),
            transcript: String::new(),
        };
    }

    if let Some(language) = forced_language {
        return AudioTranscriptionResult {
            raw_output,
            language: language.to_string(),
            transcript: normalized,
        };
    }

    let Some((metadata, text)) = normalized.split_once(ASR_TEXT_TAG) else {
        return AudioTranscriptionResult {
            raw_output,
            language: String::new(),
            transcript: normalized.trim().to_string(),
        };
    };
    let transcript = text.trim().to_string();
    if metadata.to_ascii_lowercase().contains("language none") {
        return AudioTranscriptionResult {
            raw_output,
            language: String::new(),
            transcript,
        };
    }

    let language = metadata
        .lines()
        .map(str::trim)
        .find_map(|line| {
            let lower = line.to_ascii_lowercase();
            lower
                .starts_with(LANGUAGE_PREFIX)
                .then(|| normalize_language_name(&line[LANGUAGE_PREFIX.len()..]))
        })
        .unwrap_or_default();
    AudioTranscriptionResult {
        raw_output,
        language,
        transcript,
    }
}

#[cfg(test)]
mod tests {
    use super::{ASR_TEXT_TAG, build_request, normalize_repetitions, parse_output};
    use crate::engine::types::{AudioTranscriptionResult, ContentPart, KvCacheFormat};

    #[test]
    fn auto_language_request_uses_context_and_plain_assistant_prefix() {
        assert_eq!(
            super::transcription_policy().kv_cache_format,
            KvCacheFormat::Q8
        );
        let prepared = build_request("speech.wav", "Names: Ada, Linus", None).unwrap();

        assert_eq!(
            prepared.generation_request.system_prompt,
            "Names: Ada, Linus"
        );
        assert_eq!(
            prepared.generation_request.assistant_prefill.as_deref(),
            Some("")
        );
        assert_eq!(prepared.forced_language, None);
        assert!(matches!(
            &prepared.generation_request.parts[..],
            [ContentPart::Audio(audio)] if audio.path == "speech.wav"
        ));
    }

    #[test]
    fn forced_language_request_normalizes_and_prefills_output_contract() {
        let prepared = build_request("speech.wav", "", Some("eNGLISH")).unwrap();

        assert_eq!(prepared.forced_language.as_deref(), Some("English"));
        assert_eq!(
            prepared.generation_request.assistant_prefill.as_deref(),
            Some("language English<asr_text>")
        );
    }

    #[test]
    fn unsupported_forced_language_is_rejected() {
        let error = build_request("speech.wav", "", Some("Klingon")).unwrap_err();
        assert!(error.contains("unsupported Qwen3-ASR language 'Klingon'"));
    }

    #[test]
    fn auto_language_output_is_parsed_into_typed_result() {
        let raw = "language Chinese\nconfidence ignored<asr_text>  hello 世界  ";
        assert_eq!(
            parse_output(raw, None),
            AudioTranscriptionResult {
                raw_output: raw.to_string(),
                language: "Chinese".to_string(),
                transcript: "hello 世界".to_string(),
            }
        );
    }

    #[test]
    fn forced_language_output_is_plain_transcript_text() {
        let raw = "  hello world  ";
        assert_eq!(
            parse_output(raw, Some("English")),
            AudioTranscriptionResult {
                raw_output: raw.to_string(),
                language: "English".to_string(),
                transcript: "hello world".to_string(),
            }
        );
    }

    #[test]
    fn no_tag_falls_back_to_text_with_unknown_language() {
        let parsed = parse_output("fallback transcript", None);
        assert_eq!(parsed.language, "");
        assert_eq!(parsed.transcript, "fallback transcript");
    }

    #[test]
    fn language_none_with_no_text_represents_silence() {
        let parsed = parse_output(&format!("language None{ASR_TEXT_TAG}"), None);
        assert_eq!(parsed.language, "");
        assert_eq!(parsed.transcript, "");
    }

    #[test]
    fn official_repetition_cleanup_is_applied_to_parsed_text() {
        assert_eq!(normalize_repetitions(&"x".repeat(21)), "x");
        assert_eq!(normalize_repetitions(&"ab".repeat(20)), "ab");
    }
}
