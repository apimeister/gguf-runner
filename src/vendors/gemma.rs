// Items in this module are used by the binary crate. When the library crate is linted
// in isolation (cargo clippy without --bin) they appear unused because the lib only
// exports EmbeddedRuntime and does not re-export binary-only code.

#[path = "gemma_image_views.rs"]
mod image_views;

use super::{
    ChatMessage, ChatRole, MmprojFilenameScoreHint, VendorDecodePolicy, VendorMultimodalPolicy,
    VendorRuntimeDebugPolicy,
};
use crate::engine::types::{
    Config, ContentPart, EncodedPrompt, GEMMA3_BOS_TOKEN, GEMMA3_END_TURN, GEMMA3_START_TURN,
    GenerationRequest, GgufValue, ImageAttentionMode, ImagePromptSource, ImageStretchFilter,
    ImageViewPolicy, LanguageAttentionPolicy, MultimodalBackend, RopeScalingPolicy, Tokenizer,
    VendorTokenizerPolicy,
};
use std::collections::HashMap;

pub(super) fn rope_scaling_policy(
    metadata: &HashMap<String, GgufValue>,
) -> Result<RopeScalingPolicy, String> {
    let kind = match metadata.get("gemma3.rope.scaling.type") {
        None => "linear",
        Some(GgufValue::Str(value)) => value.as_str(),
        _ => return Err("invalid Gemma3 RoPE scaling type".into()),
    };
    if kind == "none" {
        return Ok(RopeScalingPolicy::default());
    }
    if kind != "linear" {
        return Err(format!("unsupported Gemma3 RoPE scaling type: {kind}"));
    }
    let factor = match metadata
        .get("gemma3.rope.scaling.factor")
        .or_else(|| metadata.get("gemma3.rope.scale_linear"))
    {
        None => 0.0,
        Some(GgufValue::F32(value)) => *value,
        Some(GgufValue::F64(value)) if *value == 0.0 || (*value as f32) != 0.0 => *value as f32,
        _ => return Err("invalid Gemma3 RoPE scaling factor: expected a float".into()),
    };
    // Match the GGUF zero/missing-factor default; scaling applies only to global layers.
    let global = if factor == 0.0 { 1.0 } else { factor.recip() };
    if !factor.is_finite() || factor < 0.0 || !global.is_finite() || global <= 0.0 {
        return Err("invalid Gemma3 RoPE scaling factor: expected zero or a finite positive factor with finite reciprocal".into());
    }
    Ok(RopeScalingPolicy { global, local: 1.0 })
}

pub(super) fn attention_policy(
    metadata: &HashMap<String, GgufValue>,
    n_layers: usize,
) -> Result<LanguageAttentionPolicy, String> {
    fn nonnegative(value: &GgufValue, key: &str) -> Result<usize, String> {
        match value {
            GgufValue::UInt(value) => usize::try_from(*value).ok(),
            GgufValue::Int(value) => usize::try_from(*value).ok(),
            _ => None,
        }
        .ok_or_else(|| format!("invalid Gemma3 {key}: expected a nonnegative integer"))
    }
    let window_key = "gemma3.attention.sliding_window";
    let window = metadata
        .get(window_key)
        .map(|value| nonnegative(value, window_key))
        .transpose()?
        .unwrap_or(0);
    let mut layer_windows = Vec::new();
    layer_windows
        .try_reserve_exact(n_layers)
        .map_err(|_| "unable to allocate Gemma3 attention policy".to_string())?;
    layer_windows.resize(n_layers, None);
    if window > 0 {
        let pattern_key = "gemma3.attention.sliding_window_pattern";
        match metadata.get(pattern_key) {
            Some(GgufValue::I64Array(pattern)) => {
                if pattern.len() != n_layers
                    || pattern.iter().any(|&value| value != 0 && value != 1)
                {
                    return Err(
                        "Gemma3 sliding-window pattern must contain one 0/1 entry per layer"
                            .to_string(),
                    );
                }
                for (slot, &local) in layer_windows.iter_mut().zip(pattern) {
                    *slot = (local == 1).then_some(window);
                }
            }
            value => {
                let period = value
                    .map(|value| nonnegative(value, pattern_key))
                    .transpose()?
                    .unwrap_or(6);
                for (layer, slot) in layer_windows.iter_mut().enumerate() {
                    // GGUF period zero denotes local attention in every layer.
                    *slot = (period == 0 || layer % period != period - 1).then_some(window);
                }
            }
        }
    }
    Ok(LanguageAttentionPolicy {
        layer_windows,
        image_mode: ImageAttentionMode::Bidirectional,
    })
}

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
        image_stretch_filter: ImageStretchFilter::PillowBilinear,
        image_view_prompt: Some(image_views::encode_request),
        // Pan and scan, using the Transformers Gemma3ImageProcessor defaults.
        // Images below `min_aspect_ratio` still plan a single overview view.
        image_view_policy: ImageViewPolicy::LongAxisCrops {
            min_crop_size: 256,
            max_crops: 4,
            min_aspect_ratio: 1.2,
        },
        mmproj_filename_score_hints: GEMMA_MMPROJ_SCORE_HINTS,
        missing_sidecar_hint: " hint: Gemma3 image inputs require a compatible Gemma3 mmproj sidecar from the same checkpoint family.",
        ..VendorMultimodalPolicy::default()
    }
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    VendorRuntimeDebugPolicy::default()
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
) -> Result<EncodedPrompt, String> {
    let count = request
        .parts
        .iter()
        .filter(|part| matches!(part, ContentPart::Image(_)))
        .count();
    if count > 0 {
        let sources = (0..count)
            .map(|source_index| ImagePromptSource {
                source_index,
                view_count: 1,
            })
            .collect::<Vec<_>>();
        return image_views::encode_request(tokenizer, request, &sources, usize::MAX);
    }
    Ok(encode_text_request(tokenizer, request))
}

fn encode_text_request(tokenizer: &mut Tokenizer, request: &GenerationRequest) -> EncodedPrompt {
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
            ContentPart::Image(_) | ContentPart::Video(_) | ContentPart::Audio(_) => {}
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
        image_spans: Vec::new(),
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
    encode_text_request(tokenizer, &request).token_ids
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
    use super::{
        attention_policy, encode_generation_request, multimodal_policy, rope_scaling_policy,
    };
    use crate::engine::types::ImageViewPolicy;

    #[test]
    fn gemma_rope_scaling_matches_gguf_defaults_and_global_only_linear_scaling() {
        let mut metadata = std::collections::HashMap::new();
        assert_eq!(rope_scaling_policy(&metadata).unwrap(), Default::default());
        let legacy = "gemma3.rope.scale_linear";
        let factor = "gemma3.rope.scaling.factor";
        let kind = "gemma3.rope.scaling.type";
        metadata.insert(legacy.into(), GgufValue::F32(4.0));
        assert_eq!(rope_scaling_policy(&metadata).unwrap().global, 0.25);
        metadata.insert(kind.into(), GgufValue::Str("linear".into()));
        metadata.insert(factor.into(), GgufValue::F32(8.0));
        let policy = rope_scaling_policy(&metadata).unwrap();
        assert_eq!((policy.global, policy.local), (0.125, 1.0));
        metadata.insert(factor.into(), GgufValue::F64(8.0));
        assert_eq!(rope_scaling_policy(&metadata).unwrap(), policy);
        metadata.insert(factor.into(), GgufValue::F32(0.0));
        assert_eq!(rope_scaling_policy(&metadata).unwrap(), Default::default());
        metadata.insert(factor.into(), GgufValue::F32(8.0));
        metadata.insert(kind.into(), GgufValue::Str("none".into()));
        assert_eq!(rope_scaling_policy(&metadata).unwrap(), Default::default());
    }

    #[test]
    fn gemma_rope_scaling_rejects_invalid_or_unsupported_metadata() {
        let mut metadata = std::collections::HashMap::new();
        for value in [
            GgufValue::F32(-1.0),
            GgufValue::F32(f32::NAN),
            GgufValue::F32(f32::INFINITY),
            GgufValue::F32(f32::from_bits(1)),
            GgufValue::F64(f64::MIN_POSITIVE),
            GgufValue::UInt(8),
        ] {
            metadata.insert("gemma3.rope.scaling.factor".into(), value);
            assert!(rope_scaling_policy(&metadata).is_err());
        }
        metadata.clear();
        for value in [GgufValue::UInt(1), GgufValue::Str("yarn".into())] {
            metadata.insert("gemma3.rope.scaling.type".into(), value);
            assert!(rope_scaling_policy(&metadata).is_err());
        }
    }
    use crate::engine::types::{
        ContentPart, GenerationRequest, GgufValue, ImageAttentionMode, MediaRef, Tokenizer,
    };
    use std::collections::HashMap;

    #[test]
    fn gemma_attention_reads_gguf_windows_and_scalar_or_array_patterns() {
        let mut metadata = HashMap::new();
        let policy = attention_policy(&metadata, 6).unwrap();
        assert_eq!(policy.layer_windows, vec![None; 6]);
        assert_eq!(policy.image_mode, ImageAttentionMode::Bidirectional);
        metadata.insert(
            "gemma3.attention.sliding_window".into(),
            GgufValue::UInt(1024),
        );
        // The unrelated hybrid-model key must not override the Gemma pattern.
        metadata.insert("gemma3.full_attention_interval".into(), GgufValue::UInt(2));
        assert_eq!(
            attention_policy(&metadata, 6).unwrap().layer_windows,
            vec![
                Some(1024),
                Some(1024),
                Some(1024),
                Some(1024),
                Some(1024),
                None
            ]
        );
        let pattern = "gemma3.attention.sliding_window_pattern";
        metadata.insert(pattern.into(), GgufValue::UInt(2));
        assert_eq!(
            attention_policy(&metadata, 4).unwrap().layer_windows,
            vec![Some(1024), None, Some(1024), None]
        );
        metadata.insert(pattern.into(), GgufValue::UInt(0));
        assert_eq!(
            attention_policy(&metadata, 4).unwrap().layer_windows,
            vec![Some(1024); 4]
        );
        metadata.insert(pattern.into(), GgufValue::I64Array(vec![0, 1, 1, 0]));
        assert_eq!(
            attention_policy(&metadata, 4).unwrap().layer_windows,
            vec![None, Some(1024), Some(1024), None]
        );
        for value in [
            GgufValue::Int(-1),
            GgufValue::F32(2.0),
            GgufValue::I64Array(vec![1, 0]),
            GgufValue::I64Array(vec![1, 0, 2, 0]),
        ] {
            metadata.insert(pattern.into(), value);
            assert!(attention_policy(&metadata, 4).is_err());
        }
        metadata.insert("gemma3.attention.sliding_window".into(), GgufValue::Int(-1));
        assert!(attention_policy(&metadata, 4).is_err());
    }

    fn tokenizer_with_gemma_specials() -> Tokenizer {
        Tokenizer {
            vocab: vec![
                "<bos>".to_string(),
                "<start_of_turn>".to_string(),
                "<end_of_turn>".to_string(),
                "<start_of_image>".to_string(),
                "<end_of_image>".to_string(),
                "<image_soft_token>".to_string(),
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

        let encoded = encode_generation_request(&mut tokenizer, &request).unwrap();
        assert_eq!(encoded.image_spans.len(), 1);

        let start = tokenizer
            .find_special_token("<start_of_image>")
            .expect("start_of_image");
        let end = tokenizer
            .find_special_token("<end_of_image>")
            .expect("end_of_image");
        let span = encoded.image_spans[0];
        assert_eq!(encoded.token_ids[span.token_start], start);
        assert_eq!(encoded.token_ids[span.token_start + 2], end);
        assert_eq!(encoded.image_spans[0].token_len, 3);
    }

    /// Gemma3 always plans pan-and-scan views, with the pinned processor
    /// parameters. The grouped prompt contract is required to execute them.
    #[test]
    fn gemma_selects_pan_and_scan_with_the_pinned_processor_parameters() {
        let policy = multimodal_policy();
        assert!(
            policy.image_view_prompt.is_some(),
            "grouped image prompts are required for pan and scan"
        );
        assert!(matches!(
            policy.image_view_policy,
            ImageViewPolicy::LongAxisCrops {
                min_crop_size: 256,
                max_crops: 4,
                min_aspect_ratio,
            } if (min_aspect_ratio - 1.2).abs() < f64::EPSILON
        ));
    }
}
