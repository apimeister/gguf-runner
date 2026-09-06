use crate::app::events::{
    RuntimeEvent, RuntimeEventCallback, RuntimePhase, RuntimeProgress, emit_runtime_event,
};
use crate::cli::CliOptions;
use crate::engine::audio::{
    load_audio_chunk_samples, prepare_audios_for_multimodal, probe_audios_for_multimodal,
};
use crate::engine::io::{get_gguf_string_from_map, parse_gguf_file};
use crate::engine::kernels::{TopKSampler, argmax, sample, softmax};
use crate::engine::multimodal::{
    AudioEncoder, MediaEmbeddingSequence, VisionEncoder, build_audio_encoder_from_mmproj,
    build_vision_encoder_from_mmproj, expand_prompt_with_media_embeddings, preflight_media_context,
};
use crate::engine::profiling::{PROF_TRANSFORMER_NS, prof_end, prof_start, record_forward_pass};
use crate::engine::types::{
    AudioEncoderBackend, AudioTranscriptionResult, Config, ContentPart, EncodedPrompt, GGUFFile,
    GenerationRequest, MediaRef, MultimodalBackend, MultimodalWeights, PlaceholderSpan,
    RopePositionPlan, ThinkMode, Tokenizer, TransformerWeights, XorShiftRng,
};
use crate::engine::vision::{
    ImageNormalization, ImagePreprocessProfile, ImageResizeMode, load_video_chunk_tensors,
    prepare_images_for_multimodal, prepare_videos_for_multimodal,
};
use crate::rag::{DocumentEncoder, RagIndex, prepend_rag_context};
use image::{ImageFormat, ImageReader};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Sampling parameters embedded in the GGUF file under `general.sampling.*`.
/// These are model-provided defaults; explicit CLI flags take precedence.
#[allow(dead_code)]
struct GgufSamplingHints {
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

#[allow(dead_code)]
fn read_gguf_sampling_hints(gguf: &GGUFFile) -> GgufSamplingHints {
    use crate::engine::types::GgufValue;
    let get_f32 = |key: &str| -> Option<f32> {
        match gguf.kv.get(key)? {
            GgufValue::F32(f) => Some(*f),
            GgufValue::F64(f) => Some(*f as f32),
            _ => None,
        }
    };
    let get_usize = |key: &str| -> Option<usize> {
        match gguf.kv.get(key)? {
            GgufValue::UInt(n) => Some(*n as usize),
            GgufValue::Int(n) if *n >= 0 => Some(*n as usize),
            GgufValue::F32(f) if *f >= 1.0 => Some(*f as usize),
            _ => None,
        }
    };
    GgufSamplingHints {
        temperature: get_f32("general.sampling.temp"),
        top_k: get_usize("general.sampling.top_k"),
        top_p: get_f32("general.sampling.top_p"),
    }
}

fn time_in_ms() -> i64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() * 1000 + (now.subsec_nanos() as u64 / 1_000_000)) as i64
}

fn repeated_cycle_period(tokens: &[i32]) -> Option<usize> {
    // Detect degenerate decode loops by checking whether the current token suffix
    // reappears several times in a recent lookback window (alignment-agnostic).
    // Smaller windows catch tight think-phase cycles (e.g. "I think." = 3 tokens).
    const WINDOWS: &[usize] = &[4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];
    for &window in WINDOWS {
        let need = window * 3;
        if tokens.len() < need {
            continue;
        }
        let n = tokens.len();
        let suffix = &tokens[n - window..n];
        let search_start = n.saturating_sub(window * 6);
        let search_end = n - window;
        let mut hits = 0usize;
        let mut start = search_start;
        while start <= search_end {
            if &tokens[start..start + window] == suffix {
                hits += 1;
                if hits >= 3 {
                    return Some(window);
                }
                // Ignore overlapping matches to avoid false positives on local repetition.
                start = start.saturating_add(window);
                continue;
            }
            start += 1;
        }
    }
    None
}

/// Detect inline phrase repetition — works without newlines and tolerates
/// slight token-level variations (e.g. "The The capital" vs "The capital").
/// Returns the repeated phrase if a substring of 8–80 bytes appears 4+ times
/// in the last 1024 bytes of output.
fn repeated_inline_phrase(output: &str) -> Option<String> {
    const MIN_PHRASE: usize = 8;
    const MAX_PHRASE: usize = 80;
    const MIN_REPS: usize = 3;
    const LOOKBACK: usize = 1024;

    let window = if output.len() > LOOKBACK {
        // `len - LOOKBACK` can land inside a multi-byte character; snap to a
        // char boundary instead of slicing (which would panic).
        split_at_char_boundary(output, output.len() - LOOKBACK).1
    } else {
        output
    };
    if window.len() < MIN_PHRASE * MIN_REPS {
        return None;
    }
    let bytes = window.as_bytes();
    let n = bytes.len();

    // Try each candidate phrase length from large to small (prefer longer matches).
    for phrase_len in (MIN_PHRASE..=MAX_PHRASE.min(n / MIN_REPS)).rev() {
        let suffix = &bytes[n - phrase_len..n];
        let mut count = 0usize;
        let mut i = 0;
        while i + phrase_len <= n {
            if &bytes[i..i + phrase_len] == suffix {
                count += 1;
                if count >= MIN_REPS {
                    return Some(String::from_utf8_lossy(suffix).replace('\n', "\\n"));
                }
                i += phrase_len; // non-overlapping: skip past this occurrence
            } else {
                i += 1;
            }
        }
    }
    None
}

fn repeated_long_line(output: &str) -> Option<(String, usize)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    const MIN_REPEAT_COUNT: usize = 3;
    for line in output.lines().rev().take(96) {
        let normalized = line.trim();
        if normalized.len() < 24 {
            continue;
        }
        if normalized.chars().all(|c| c.is_ascii_punctuation()) {
            continue;
        }
        let entry = counts.entry(normalized.to_string()).or_insert(0);
        *entry += 1;
        if *entry >= MIN_REPEAT_COUNT {
            return Some((normalized.to_string(), *entry));
        }
    }
    None
}

fn repeated_text_suffix_bytes(output: &str) -> Option<usize> {
    let bytes = output.as_bytes();
    const LENGTHS: &[usize] = &[64, 96, 128, 160, 192, 256];
    for &len in LENGTHS {
        if bytes.len() < len * 2 {
            continue;
        }
        let n = bytes.len();
        let a = &bytes[n - len..n];
        let b = &bytes[n - 2 * len..n - len];
        if a == b {
            return Some(len);
        }
    }
    None
}

const THINK_CLOSE_TAG: &str = "</think>";
const THINK_OPEN_TAG: &str = "<think>";
const THINK_OPEN_TAG_ALIASES: &[&str] = &[THINK_OPEN_TAG, "<thinking>", "<thought>"];
const THINK_CLOSE_TAG_ALIASES: &[&str] = &[THINK_CLOSE_TAG, "</thinking>", "</thought>"];
const KNOWN_PROTOCOL_MARKERS: &[&str] = &[
    THINK_OPEN_TAG,
    THINK_CLOSE_TAG,
    "<thinking>",
    "</thinking>",
    "<thought>",
    "</thought>",
    "<answer>",
    "</answer>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|im_start|>",
    "<|im_start|>assistant",
    "<|im_start|>user",
    "<|im_start|>system",
    "</assistant>",
    "</user>",
    "</system>",
];

fn is_simple_protocol_tag(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 3 || !trimmed.starts_with('<') || !trimmed.ends_with('>') {
        return false;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let inner = inner.strip_prefix('/').unwrap_or(inner);
    if inner.is_empty() || inner.contains(char::is_whitespace) {
        return false;
    }
    inner
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | ':'))
}

fn append_visible_text(
    decoded: &str,
    output: &mut String,
    pending_newline: &mut bool,
    stream_stdout: bool,
    callback: Option<&RuntimeEventCallback>,
) {
    if decoded.is_empty() {
        return;
    }
    let decoded = if output.is_empty() {
        decoded.trim_start_matches(['\n', '\r'])
    } else {
        decoded
    };
    if decoded.is_empty() {
        return;
    }
    if decoded == "\n" {
        *pending_newline = true;
        return;
    }
    if *pending_newline {
        if !output.is_empty() {
            output.push('\n');
            if callback.is_some() {
                emit_runtime_event(callback, RuntimeEvent::Output("\n".to_string()));
            } else if stream_stdout {
                println!();
            }
        }
        *pending_newline = false;
    }
    output.push_str(decoded);
    if callback.is_some() {
        emit_runtime_event(callback, RuntimeEvent::Output(decoded.to_string()));
    } else if stream_stdout {
        print!("{decoded}");
        let _ = io::stdout().flush();
    }
}

fn find_first_tag<'a>(text: &'a str, tags: &[&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|&tag| text.find(tag).map(|idx| (idx, tag)))
        .min_by_key(|(idx, _)| *idx)
}

fn find_last_tag<'a>(text: &'a str, tags: &[&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|&tag| text.rfind(tag).map(|idx| (idx, tag)))
        .max_by_key(|(idx, _)| *idx)
}

fn find_trailing_stop_text_literal(
    text: &str,
    stop_text_literals: &'static [&'static str],
) -> Option<(usize, &'static str)> {
    let trimmed = text.trim_end();
    for &literal in stop_text_literals {
        let Some(prefix) = trimmed.strip_suffix(literal) else {
            continue;
        };
        let last_line = prefix
            .rsplit_once('\n')
            .map(|(_, line)| line)
            .unwrap_or(prefix);
        if last_line.trim().is_empty() {
            return Some((prefix.len(), literal));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn append_visible_text_with_stop_literals(
    decoded: &str,
    output: &mut String,
    pending_newline: &mut bool,
    stream_stdout: bool,
    callback: Option<&RuntimeEventCallback>,
    stop_text_literals: &'static [&'static str],
    stop_text_tail: &mut String,
    matched_stop_text_literal: &mut Option<&'static str>,
) {
    if decoded.is_empty() || matched_stop_text_literal.is_some() {
        return;
    }
    if stop_text_literals.is_empty() {
        append_visible_text(decoded, output, pending_newline, stream_stdout, callback);
        return;
    }

    let mut combined = String::new();
    if !stop_text_tail.is_empty() {
        combined.push_str(stop_text_tail);
        stop_text_tail.clear();
    }
    combined.push_str(decoded);

    if let Some((idx, literal)) = find_trailing_stop_text_literal(&combined, stop_text_literals) {
        let visible = combined[..idx].trim_end_matches(['\n', '\r']);
        append_visible_text(visible, output, pending_newline, stream_stdout, callback);
        *matched_stop_text_literal = Some(literal);
        return;
    }

    let holdback = stop_text_literals
        .iter()
        .map(|literal| literal.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    if holdback == 0 || combined.len() <= holdback {
        stop_text_tail.push_str(&combined);
        return;
    }

    let emit_len = combined.len() - holdback;
    let (emit_part, tail_part) = split_at_char_boundary(&combined, emit_len);
    append_visible_text(emit_part, output, pending_newline, stream_stdout, callback);
    stop_text_tail.push_str(tail_part);
}

fn flush_visible_text_stop_tail(
    output: &mut String,
    pending_newline: &mut bool,
    stream_stdout: bool,
    callback: Option<&RuntimeEventCallback>,
    stop_text_tail: &mut String,
    matched_stop_text_literal: Option<&'static str>,
) {
    if matched_stop_text_literal.is_some() || stop_text_tail.is_empty() {
        stop_text_tail.clear();
        return;
    }
    let pending = std::mem::take(stop_text_tail);
    append_visible_text(&pending, output, pending_newline, stream_stdout, callback);
}

fn emit_debug_line(callback: Option<&RuntimeEventCallback>, text: impl Into<String>) {
    let text = text.into();
    if callback.is_some() {
        emit_runtime_event(
            callback,
            RuntimeEvent::Log(crate::app::events::RuntimeLog::debug(text)),
        );
    } else {
        eprintln!("{text}");
    }
}

fn emit_output_text(
    callback: Option<&RuntimeEventCallback>,
    text: impl Into<String>,
    stream_stdout: bool,
) {
    let text = text.into();
    if callback.is_some() {
        emit_runtime_event(callback, RuntimeEvent::Output(text));
    } else if stream_stdout && !text.is_empty() {
        print!("{text}");
        let _ = io::stdout().flush();
    }
}

fn emit_cli_info_line(
    callback: Option<&RuntimeEventCallback>,
    text: impl Into<String>,
    stream_stdout: bool,
    prefer_stdout: bool,
) {
    let text = text.into();
    if callback.is_some() {
        emit_runtime_event(
            callback,
            RuntimeEvent::Log(crate::app::events::RuntimeLog::debug(text)),
        );
    } else if prefer_stdout && stream_stdout {
        println!("{text}");
    } else {
        eprintln!("{text}");
    }
}

fn emit_progress_update(callback: Option<&RuntimeEventCallback>, progress: RuntimeProgress) {
    emit_runtime_event(callback, RuntimeEvent::Progress(progress));
}

fn decode_utf8_streaming(pending: &mut Vec<u8>, piece: &[u8]) -> String {
    pending.extend_from_slice(piece);
    let mut out = String::new();

    loop {
        match std::str::from_utf8(pending) {
            Ok(valid) => {
                out.push_str(valid);
                pending.clear();
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    let valid = std::str::from_utf8(&pending[..valid_up_to]).unwrap_or_default();
                    out.push_str(valid);
                    pending.drain(..valid_up_to);
                }
                if err.error_len().is_none() {
                    // Incomplete UTF-8 sequence at the end; wait for more token bytes.
                    break;
                }
                // Invalid UTF-8 sequence: emit replacement and advance one byte.
                out.push('\u{FFFD}');
                if !pending.is_empty() {
                    pending.drain(..1);
                } else {
                    break;
                }
            }
        }
    }

    out
}

fn flush_utf8_pending_lossy(pending: &mut Vec<u8>) -> String {
    if pending.is_empty() {
        return String::new();
    }
    let s = String::from_utf8_lossy(pending).to_string();
    pending.clear();
    s
}

fn has_post_think_response_text(output: &str) -> bool {
    if let Some((idx, close_tag)) = find_last_tag(output, THINK_CLOSE_TAG_ALIASES) {
        let rest = &output[idx + close_tag.len()..];
        return has_meaningful_retry_text(rest);
    }
    false
}

fn is_incomplete_think_tag_fragment(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.contains('>') || !trimmed.starts_with('<') {
        return false;
    }
    THINK_OPEN_TAG_ALIASES
        .iter()
        .any(|tag| tag.starts_with(trimmed))
        || THINK_CLOSE_TAG_ALIASES
            .iter()
            .any(|tag| tag.starts_with(trimmed))
        || trimmed.starts_with("<th")
        || trimmed.starts_with("</th")
}

fn finalize_visible_think_tail(tail: &mut String, is_thinking: bool) -> String {
    if tail.is_empty() {
        return String::new();
    }
    let raw = std::mem::take(tail);
    if !is_thinking {
        return raw;
    }
    if THINK_OPEN_TAG_ALIASES
        .iter()
        .any(|tag| tag.starts_with(&raw))
        || THINK_CLOSE_TAG_ALIASES
            .iter()
            .any(|tag| tag.starts_with(&raw))
    {
        return String::new();
    }
    if let Some(idx) = raw.rfind('<') {
        let suffix = &raw[idx..];
        if is_incomplete_think_tag_fragment(suffix) {
            return raw[..idx].to_string();
        }
    }
    raw
}

fn trim_trailing_protocol_markers(text: &str) -> String {
    let mut trimmed = text.trim().to_string();
    loop {
        let mut changed = false;
        for marker in KNOWN_PROTOCOL_MARKERS {
            if let Some(prefix) = trimmed.strip_suffix(marker) {
                trimmed = prefix.trim_end().to_string();
                changed = true;
                break;
            }
        }
        if !changed
            && let Some(last_line) = trimmed.lines().last()
            && is_simple_protocol_tag(last_line)
        {
            let new_len = trimmed.len().saturating_sub(last_line.len());
            trimmed.truncate(new_len);
            trimmed = trimmed.trim_end().to_string();
            changed = true;
        }
        if !changed
            && let Some(last_line) = trimmed.lines().last()
            && is_incomplete_think_tag_fragment(last_line)
        {
            let new_len = trimmed.len().saturating_sub(last_line.len());
            trimmed.truncate(new_len);
            trimmed = trimmed.trim_end().to_string();
            changed = true;
        }
        if !changed {
            break;
        }
    }
    trimmed
}

fn promote_think_only_content(output: &str) -> Option<String> {
    let mut promoted = trim_trailing_protocol_markers(output);
    for close_tag in THINK_CLOSE_TAG_ALIASES {
        if let Some(prefix) = promoted.strip_suffix(close_tag) {
            promoted = prefix.trim().to_string();
            break;
        }
    }
    for open_tag in THINK_OPEN_TAG_ALIASES {
        if let Some(rest) = promoted.strip_prefix(open_tag) {
            promoted = rest.trim().to_string();
            break;
        }
    }
    if promoted.is_empty() {
        None
    } else {
        Some(promoted)
    }
}

fn has_meaningful_retry_text(output: &str) -> bool {
    let mut stripped = output.to_string();
    for marker in KNOWN_PROTOCOL_MARKERS {
        stripped = stripped.replace(marker, "");
    }
    let stripped = stripped
        .lines()
        .filter(|line| !is_simple_protocol_tag(line.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    let stripped = stripped.trim().to_string();
    stripped.chars().any(|ch| ch.is_alphanumeric())
}

fn sanitize_final_response_text(output: &str) -> String {
    trim_trailing_protocol_markers(output)
}

/// Hidden-mode reasoning budget: `(think_cap, total_cap)` in decode tokens.
/// `think_cap` bounds the reasoning block, `total_cap` the whole generation.
/// `override_cap` (caller-set) replaces the vendor heuristic and its hard
/// ceilings; only the real decode budget still bounds it.
fn compute_think_caps(
    hidden_cap_base: usize,
    n_layers: usize,
    seq_len: usize,
    new_budget: usize,
    override_cap: Option<usize>,
) -> (usize, usize) {
    let hidden_vendor_base = hidden_cap_base.max(256usize);
    let layer_mult = if n_layers >= 64 {
        3usize
    } else if n_layers >= 32 {
        2usize
    } else {
        1usize
    };
    let ctx_mult = if seq_len >= 262_144 { 2usize } else { 1usize };
    let hard_think_cap_max = if n_layers >= 48 || seq_len >= 262_144 {
        1536usize
    } else {
        1024usize
    };
    let hard_total_cap_max = hard_think_cap_max.saturating_mul(4);
    let mut think_cap = hidden_vendor_base
        .saturating_mul(layer_mult)
        .saturating_mul(ctx_mult);
    if new_budget > 0 {
        let max_cap = hard_think_cap_max.min(new_budget.max(96));
        think_cap = think_cap.clamp(96, max_cap);
    } else {
        think_cap = 64;
    }
    if let Some(cap) = override_cap {
        think_cap = cap.max(1);
        if new_budget > 0 {
            think_cap = think_cap.min(new_budget);
        }
    }
    let mut total_cap = think_cap.saturating_mul(4);
    if override_cap.is_none() {
        total_cap = total_cap.min(hard_total_cap_max);
    }
    if new_budget > 0 {
        total_cap = total_cap.min(new_budget);
        let min_total = (think_cap + 64).min(new_budget).max(think_cap + 1);
        total_cap = total_cap.max(min_total);
    } else {
        total_cap = think_cap + 64;
    }
    (think_cap, total_cap)
}

/// Appends `\n</think>\n\n` to `prompt_tokens`, turning the chat template's
/// trailing `<think>\n` seed into the pre-closed empty think block that
/// `ThinkMode::No` prompts carry. Used when a hidden-mode generation is
/// retried without thinking on the already-encoded prompt.
fn append_think_close_suffix(tokenizer: &mut Tokenizer, prompt_tokens: &mut Vec<i32>) {
    let mut piece: Vec<i32> = Vec::new();
    if let Some(close_id) = tokenizer.find_special_token(THINK_CLOSE_TAG) {
        tokenizer.bpe_encode("\n", &mut piece);
        prompt_tokens.extend_from_slice(&piece);
        prompt_tokens.push(close_id);
        tokenizer.bpe_encode("\n\n", &mut piece);
        prompt_tokens.extend_from_slice(&piece);
    } else {
        tokenizer.bpe_encode("\n</think>\n\n", &mut piece);
        prompt_tokens.extend_from_slice(&piece);
    }
}

fn default_sampling_seed() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_secs() ^ ((now.subsec_nanos() as u64) << 32)
}

fn should_retry_without_think_for_output(
    think_mode: ThinkMode,
    decode_policy: crate::vendors::VendorDecodePolicy,
    output: &str,
) -> bool {
    think_mode == ThinkMode::Yes
        && decode_policy.retry_without_think_when_no_post_think_text
        && decode_policy.parse_think_tags
        && !has_post_think_response_text(output)
}

fn should_buffer_visible_think_stdout(
    stream_stdout: bool,
    has_event_callback: bool,
    think_mode: ThinkMode,
    decode_policy: crate::vendors::VendorDecodePolicy,
) -> bool {
    let _ = (stream_stdout, has_event_callback, think_mode, decode_policy);
    false
}

fn build_direct_answer_retry_system_prompt(system_prompt: &str) -> String {
    let directive = "Provide the final answer directly and briefly. Do not describe your reasoning or restate the user's question.";
    if system_prompt.trim().is_empty() {
        directive.to_string()
    } else {
        format!("{system_prompt}\n\n{directive}")
    }
}

fn find_first_complete_json_object_span(text: &str) -> Option<(usize, usize)> {
    for (idx, ch) in text.char_indices() {
        if ch != '{' {
            continue;
        }
        let mut stream = serde_json::Deserializer::from_str(&text[idx..]).into_iter::<Value>();
        let Some(Ok(value)) = stream.next() else {
            continue;
        };
        if !value.is_object() {
            continue;
        }
        let consumed = stream.byte_offset();
        if consumed == 0 {
            continue;
        }
        return Some((idx, idx + consumed));
    }
    None
}

fn extract_first_complete_json_object(text: &str) -> Option<String> {
    let (start, end) = find_first_complete_json_object_span(text)?;
    Some(text[start..end].trim().to_string())
}

fn mask_blocked_logits(logits: &mut [f32], blocked_token_ids: &HashSet<i32>) {
    for &token_id in blocked_token_ids {
        if token_id >= 0 && (token_id as usize) < logits.len() {
            logits[token_id as usize] = f32::NEG_INFINITY;
        }
    }
}

fn is_agent_json_safe_text(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    text.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(
                c,
                ' ' | '\n'
                    | '\r'
                    | '\t'
                    | '{'
                    | '}'
                    | '['
                    | ']'
                    | '('
                    | ')'
                    | ':'
                    | ','
                    | '.'
                    | '_'
                    | '-'
                    | '/'
                    | '\\'
                    | '"'
                    | '\''
                    | '`'
                    | '+'
                    | '='
                    | '*'
                    | '?'
                    | '&'
                    | '<'
                    | '>'
                    | '!'
                    | '@'
                    | '#'
                    | '$'
                    | '%'
                    | '^'
                    | '|'
                    | '~'
                    | ';'
            )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixMatch {
    Complete,
    Incomplete,
    Invalid,
}

fn match_agent_response_prefix(text: &str) -> PrefixMatch {
    let final_match = match_final_response_prefix(text);
    if final_match != PrefixMatch::Invalid {
        return final_match;
    }
    match_tool_call_response_prefix(text)
}

fn match_final_response_prefix(text: &str) -> PrefixMatch {
    let mut rest = text;
    rest = match strip_literal_prefix(rest, "{\"type\":\"") {
        Some(rest) => rest,
        None if "{\"type\":\"".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    rest = match strip_literal_prefix(rest, "final") {
        Some(rest) => rest,
        None if "final".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    if let Some(rest) = strip_literal_prefix(rest, "\"}") {
        return if rest.is_empty() {
            PrefixMatch::Complete
        } else {
            PrefixMatch::Invalid
        };
    }
    if "\"}".starts_with(rest) {
        return PrefixMatch::Incomplete;
    }
    rest = match strip_literal_prefix(rest, "\",\"content\":\"") {
        Some(rest) => rest,
        None if "\",\"content\":\"".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    match parse_json_string_prefix(rest) {
        PrefixMatch::Complete => {
            let quote_idx = find_json_string_end(rest).expect("json string end");
            let rest = &rest[quote_idx + 1..];
            if rest.is_empty() {
                return PrefixMatch::Incomplete;
            }
            match strip_literal_prefix(rest, "}") {
                Some("") => PrefixMatch::Complete,
                None if "}".starts_with(rest) => PrefixMatch::Incomplete,
                _ => PrefixMatch::Invalid,
            }
        }
        PrefixMatch::Incomplete => PrefixMatch::Incomplete,
        PrefixMatch::Invalid => PrefixMatch::Invalid,
    }
}

fn match_tool_call_response_prefix(text: &str) -> PrefixMatch {
    let mut rest = text;
    rest = match strip_literal_prefix(rest, "{\"type\":\"") {
        Some(rest) => rest,
        None if "{\"type\":\"".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    rest = match strip_literal_prefix(rest, "tool_call") {
        Some(rest) => rest,
        None if "tool_call".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    rest = match strip_literal_prefix(rest, "\",\"tool\":\"") {
        Some(rest) => rest,
        None if "\",\"tool\":\"".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    rest = match parse_json_string_prefix(rest) {
        PrefixMatch::Complete => {
            let quote_idx = find_json_string_end(rest).expect("json string end");
            &rest[quote_idx + 1..]
        }
        PrefixMatch::Incomplete => return PrefixMatch::Incomplete,
        PrefixMatch::Invalid => return PrefixMatch::Invalid,
    };
    rest = match strip_literal_prefix(rest, ",\"args\":") {
        Some(rest) => rest,
        None if ",\"args\":".starts_with(rest) => return PrefixMatch::Incomplete,
        None => return PrefixMatch::Invalid,
    };
    rest = match parse_json_object_prefix(rest) {
        PrefixMatch::Complete => {
            let len = find_json_object_end(rest).expect("json object end");
            &rest[len..]
        }
        PrefixMatch::Incomplete => return PrefixMatch::Incomplete,
        PrefixMatch::Invalid => return PrefixMatch::Invalid,
    };
    if rest.is_empty() {
        return PrefixMatch::Incomplete;
    }
    match strip_literal_prefix(rest, "}") {
        Some("") => PrefixMatch::Complete,
        None if "}".starts_with(rest) => PrefixMatch::Incomplete,
        _ => PrefixMatch::Invalid,
    }
}

fn strip_literal_prefix<'a>(input: &'a str, literal: &str) -> Option<&'a str> {
    input.strip_prefix(literal)
}

fn parse_json_string_prefix(input: &str) -> PrefixMatch {
    if input.is_empty() {
        return PrefixMatch::Incomplete;
    }
    let mut escaped = false;
    for ch in input.chars() {
        if escaped {
            if !matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') {
                return PrefixMatch::Invalid;
            }
            escaped = false;
            continue;
        }
        match ch {
            '"' => return PrefixMatch::Complete,
            '\\' => escaped = true,
            c if c.is_ascii() && !c.is_ascii_control() => {}
            _ => return PrefixMatch::Invalid,
        }
    }
    PrefixMatch::Incomplete
}

fn find_json_string_end(input: &str) -> Option<usize> {
    let mut escaped = false;
    for (idx, ch) in input.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Some(idx),
            _ => {}
        }
    }
    None
}

fn parse_json_object_prefix(input: &str) -> PrefixMatch {
    if input.is_empty() {
        return PrefixMatch::Incomplete;
    }
    let Some(first) = input.chars().next() else {
        return PrefixMatch::Incomplete;
    };
    if first != '{' {
        return PrefixMatch::Invalid;
    }
    if find_json_object_end(input).is_some() {
        PrefixMatch::Complete
    } else if is_json_object_lexically_valid_prefix(input) {
        PrefixMatch::Incomplete
    } else {
        PrefixMatch::Invalid
    }
}

fn find_json_object_end(input: &str) -> Option<usize> {
    let mut depth_brace = 0usize;
    let mut depth_bracket = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in input.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                c if c.is_ascii() && !c.is_ascii_control() => {}
                _ => return None,
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth_brace += 1,
            '}' => {
                if depth_brace == 0 {
                    return None;
                }
                depth_brace -= 1;
                if depth_brace == 0 && depth_bracket == 0 {
                    return Some(idx + ch.len_utf8());
                }
            }
            '[' => depth_bracket += 1,
            ']' => {
                if depth_bracket == 0 {
                    return None;
                }
                depth_bracket -= 1;
            }
            c if c.is_ascii_whitespace()
                || matches!(
                    c,
                    ':' | ',' | '-' | '.' | 't' | 'r' | 'u' | 'e' | 'f' | 'a' | 'l' | 's' | 'n'
                )
                || c.is_ascii_alphanumeric() => {}
            _ => return None,
        }
    }
    None
}

fn is_json_object_lexically_valid_prefix(input: &str) -> bool {
    let mut depth_brace = 0usize;
    let mut depth_bracket = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for ch in input.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                c if c.is_ascii() && !c.is_ascii_control() => {}
                _ => return false,
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth_brace += 1,
            '}' => {
                if depth_brace == 0 {
                    return false;
                }
                depth_brace -= 1;
            }
            '[' => depth_bracket += 1,
            ']' => {
                if depth_bracket == 0 {
                    return false;
                }
                depth_bracket -= 1;
            }
            c if c.is_ascii_whitespace()
                || matches!(
                    c,
                    ':' | ',' | '-' | '.' | 't' | 'r' | 'u' | 'e' | 'f' | 'a' | 'l' | 's' | 'n'
                )
                || c.is_ascii_alphanumeric() => {}
            _ => return false,
        }
    }
    depth_brace > 0 || depth_bracket > 0 || in_string
}

struct StructuredOutputSchema {
    seed_literal: &'static str,
    text_is_lexically_safe: fn(&str) -> bool,
    accepts_prefix: fn(&str) -> bool,
    extract_complete: fn(&str) -> Option<String>,
}

const AGENT_JSON_STRUCTURED_OUTPUT_SCHEMA: StructuredOutputSchema = StructuredOutputSchema {
    seed_literal: "{\"type\":\"",
    text_is_lexically_safe: is_agent_json_safe_text,
    accepts_prefix: is_valid_agent_json_prefix,
    extract_complete: extract_first_complete_json_object,
};

fn is_valid_agent_json_prefix(text: &str) -> bool {
    matches!(
        match_agent_response_prefix(text),
        PrefixMatch::Complete | PrefixMatch::Incomplete
    )
}

fn collect_structured_output_blocked_token_ids(
    tokenizer: &Tokenizer,
    stop_tokens: &[(i32, &str)],
    schema: Option<&StructuredOutputSchema>,
) -> HashSet<i32> {
    let Some(schema) = schema else {
        return HashSet::new();
    };
    let mut blocked = HashSet::new();
    let token_literals = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<bos>",
        "<eos>",
        "<start_of_turn>",
        "<end_of_turn>",
    ];
    for literal in token_literals {
        if let Some(id) = tokenizer.find_special_token(literal) {
            blocked.insert(id);
        }
    }
    if tokenizer.eos_token >= 0 {
        blocked.insert(tokenizer.eos_token);
    }
    if tokenizer.eot_token >= 0 {
        blocked.insert(tokenizer.eot_token);
    }
    for (id, _) in stop_tokens {
        blocked.insert(*id);
    }
    for token_id in 0..tokenizer.vocab_size.min(tokenizer.vocab.len()) {
        let token_id = token_id as i32;
        let Some(bytes) = tokenizer.decode_token_bytes(token_id) else {
            blocked.insert(token_id);
            continue;
        };
        let Ok(text) = std::str::from_utf8(&bytes) else {
            blocked.insert(token_id);
            continue;
        };
        if !(schema.text_is_lexically_safe)(text) {
            blocked.insert(token_id);
        }
    }
    blocked
}

fn build_structured_output_prefix_tokens(
    tokenizer: &mut Tokenizer,
    schema: Option<&StructuredOutputSchema>,
) -> Vec<i32> {
    let Some(schema) = schema else {
        return Vec::new();
    };
    let mut tokens = Vec::new();
    tokenizer.bpe_encode(schema.seed_literal, &mut tokens);
    if tokens.is_empty() {
        tokenizer.bpe_encode("{", &mut tokens);
    }
    tokens
}

fn mask_invalid_structured_output_logits(
    logits: &mut [f32],
    blocked_token_ids: &HashSet<i32>,
    tokenizer: &Tokenizer,
    current_output: &str,
    schema: &StructuredOutputSchema,
) {
    mask_blocked_logits(logits, blocked_token_ids);
    for (token_id, logit) in logits.iter_mut().enumerate() {
        if logit.is_infinite() && logit.is_sign_negative() {
            continue;
        }
        let token_id_i32 = token_id as i32;
        let Some(bytes) = tokenizer.decode_token_bytes(token_id_i32) else {
            *logit = f32::NEG_INFINITY;
            continue;
        };
        let Ok(text) = std::str::from_utf8(&bytes) else {
            *logit = f32::NEG_INFINITY;
            continue;
        };
        let mut candidate = String::with_capacity(current_output.len() + text.len());
        candidate.push_str(current_output);
        candidate.push_str(text);
        if !(schema.text_is_lexically_safe)(text) || !(schema.accepts_prefix)(&candidate) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn extract_first_complete_structured_output(
    text: &str,
    schema: Option<&StructuredOutputSchema>,
) -> Option<String> {
    let schema = schema?;
    (schema.extract_complete)(text)
}

fn split_at_char_boundary(s: &str, byte_idx: usize) -> (&str, &str) {
    let mut i = byte_idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    s.split_at(i)
}

fn hidden_strip_visible_chunk(decoded: &str, tail: &mut String) -> String {
    let mut combined = String::new();
    if !tail.is_empty() {
        combined.push_str(tail);
        tail.clear();
    }
    combined.push_str(decoded);

    let mut keep_len = 0usize;
    for tag in THINK_OPEN_TAG_ALIASES
        .iter()
        .chain(THINK_CLOSE_TAG_ALIASES.iter())
    {
        let max_prefix = tag.len().saturating_sub(1);
        for k in 1..=max_prefix {
            if combined.ends_with(&tag[..k]) {
                keep_len = keep_len.max(k);
            }
        }
    }
    let split_at = combined.len().saturating_sub(keep_len);
    let (emit_part, tail_part) = split_at_char_boundary(&combined, split_at);
    tail.push_str(tail_part);
    let mut cleaned = emit_part.to_string();
    for tag in THINK_OPEN_TAG_ALIASES {
        cleaned = cleaned.replace(tag, "");
    }
    for tag in THINK_CLOSE_TAG_ALIASES {
        cleaned = cleaned.replace(tag, "");
    }
    cleaned
}

fn hidden_finalize_tail(tail: &mut String) -> String {
    if tail.is_empty() {
        return String::new();
    }
    let raw = std::mem::take(tail);
    if THINK_OPEN_TAG_ALIASES
        .iter()
        .any(|tag| tag.starts_with(&raw))
        || THINK_CLOSE_TAG_ALIASES
            .iter()
            .any(|tag| tag.starts_with(&raw))
    {
        return String::new();
    }
    let mut cleaned = raw;
    for tag in THINK_OPEN_TAG_ALIASES {
        cleaned = cleaned.replace(tag, "");
    }
    for tag in THINK_CLOSE_TAG_ALIASES {
        cleaned = cleaned.replace(tag, "");
    }
    cleaned
}

fn trim_leading_line_breaks_for_first_visible<'a>(text: &'a str, output: &str) -> &'a str {
    if output.is_empty() {
        text.trim_start_matches(['\n', '\r'])
    } else {
        text
    }
}

#[allow(clippy::too_many_arguments)]
fn process_decoded_with_think(
    decoded: &str,
    parse_think_tags: bool,
    think_mode: ThinkMode,
    is_thinking: &mut bool,
    think_tail: &mut String,
    hidden_visible_tail: &mut String,
    output: &mut String,
    pending_newline: &mut bool,
    stream_stdout: bool,
    suppress_visible_think_stdout: bool,
    callback: Option<&RuntimeEventCallback>,
    stop_text_literals: &'static [&'static str],
    stop_text_tail: &mut String,
    matched_stop_text_literal: &mut Option<&'static str>,
) {
    if decoded.is_empty() {
        return;
    }

    if think_mode == ThinkMode::No {
        if parse_think_tags {
            let cleaned = hidden_strip_visible_chunk(decoded, hidden_visible_tail);
            let cleaned = trim_leading_line_breaks_for_first_visible(&cleaned, output);
            append_visible_text_with_stop_literals(
                cleaned,
                output,
                pending_newline,
                stream_stdout,
                callback,
                stop_text_literals,
                stop_text_tail,
                matched_stop_text_literal,
            );
        } else {
            append_visible_text_with_stop_literals(
                decoded,
                output,
                pending_newline,
                stream_stdout,
                callback,
                stop_text_literals,
                stop_text_tail,
                matched_stop_text_literal,
            );
        }
        return;
    }
    if !parse_think_tags {
        append_visible_text_with_stop_literals(
            decoded,
            output,
            pending_newline,
            stream_stdout,
            callback,
            stop_text_literals,
            stop_text_tail,
            matched_stop_text_literal,
        );
        return;
    }

    let mut combined = String::new();
    if !think_tail.is_empty() {
        combined.push_str(think_tail);
        think_tail.clear();
    }
    combined.push_str(decoded);

    if !*is_thinking {
        if think_mode == ThinkMode::Hidden {
            let cleaned = hidden_strip_visible_chunk(&combined, hidden_visible_tail);
            let cleaned = trim_leading_line_breaks_for_first_visible(&cleaned, output);
            append_visible_text_with_stop_literals(
                cleaned,
                output,
                pending_newline,
                stream_stdout,
                callback,
                stop_text_literals,
                stop_text_tail,
                matched_stop_text_literal,
            );
        } else {
            append_visible_text_with_stop_literals(
                &combined,
                output,
                pending_newline,
                stream_stdout,
                callback,
                stop_text_literals,
                stop_text_tail,
                matched_stop_text_literal,
            );
        }
        return;
    }

    if let Some((close_idx, close_tag)) = find_first_tag(&combined, THINK_CLOSE_TAG_ALIASES) {
        if think_mode == ThinkMode::Yes {
            let end = close_idx + close_tag.len();
            append_visible_text_with_stop_literals(
                &combined[..end],
                output,
                pending_newline,
                stream_stdout && !suppress_visible_think_stdout,
                callback,
                stop_text_literals,
                stop_text_tail,
                matched_stop_text_literal,
            );
        } else {
            *pending_newline = false;
        }
        *is_thinking = false;
        let rest = &combined[close_idx + close_tag.len()..];
        if !rest.is_empty() {
            if think_mode == ThinkMode::Hidden || think_mode == ThinkMode::No {
                let cleaned = hidden_strip_visible_chunk(rest, hidden_visible_tail);
                let cleaned = trim_leading_line_breaks_for_first_visible(&cleaned, output);
                append_visible_text_with_stop_literals(
                    cleaned,
                    output,
                    pending_newline,
                    stream_stdout,
                    callback,
                    stop_text_literals,
                    stop_text_tail,
                    matched_stop_text_literal,
                );
            } else {
                append_visible_text_with_stop_literals(
                    rest,
                    output,
                    pending_newline,
                    stream_stdout,
                    callback,
                    stop_text_literals,
                    stop_text_tail,
                    matched_stop_text_literal,
                );
            }
        }
        return;
    }

    let keep = THINK_CLOSE_TAG_ALIASES
        .iter()
        .map(|tag| tag.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    if think_mode == ThinkMode::Yes {
        if combined.len() > keep {
            let emit_len = combined.len() - keep;
            let (emit_part, tail_part) = split_at_char_boundary(&combined, emit_len);
            append_visible_text_with_stop_literals(
                emit_part,
                output,
                pending_newline,
                stream_stdout && !suppress_visible_think_stdout,
                callback,
                stop_text_literals,
                stop_text_tail,
                matched_stop_text_literal,
            );
            think_tail.push_str(tail_part);
        } else {
            think_tail.push_str(&combined);
        }
    } else {
        *pending_newline = false;
        if combined.len() > keep {
            let split_at = combined.len().saturating_sub(keep);
            let (_, tail_part) = split_at_char_boundary(&combined, split_at);
            think_tail.push_str(tail_part);
        } else {
            think_tail.push_str(&combined);
        }
    }
}

pub(crate) struct GenerationSettings {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) sampling_seed: Option<u64>,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) profiling_mode: bool,
    pub(crate) show_timings: bool,
    pub(crate) show_tokens: bool,
    pub(crate) debug_mode: bool,
    pub(crate) think_mode: ThinkMode,
    /// Caller-set cap on hidden-mode reasoning tokens per generation. `None`
    /// selects the vendor-derived heuristic in [`compute_think_caps`].
    pub(crate) hidden_think_token_cap: Option<usize>,
    pub(crate) structured_output_mode: StructuredOutputMode,
    pub(crate) vendor_decode_policy: crate::vendors::VendorDecodePolicy,
    pub(crate) vendor_multimodal_policy: crate::vendors::VendorMultimodalPolicy,
    pub(crate) runtime_event_callback: Option<RuntimeEventCallback>,
    pub(crate) rag_top_k: usize,
    pub(crate) rag_max_chars_per_chunk: usize,
    pub(crate) rag_max_tokens_per_chunk: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StructuredOutputMode {
    #[default]
    None,
    AgentJson,
}

impl StructuredOutputMode {
    fn schema(self) -> Option<&'static StructuredOutputSchema> {
        match self {
            StructuredOutputMode::None => None,
            StructuredOutputMode::AgentJson => Some(&AGENT_JSON_STRUCTURED_OUTPUT_SCHEMA),
        }
    }
}

#[derive(Clone, Debug)]
struct MmprojSidecarProbe {
    path: String,
    has_vision_encoder: bool,
    has_vision_projector: bool,
    audio_backend: Option<AudioEncoderBackend>,
    n_tensors: u64,
}

fn load_rag_components(
    cli: &CliOptions,
    debug_mode: bool,
) -> Result<(Option<DocumentEncoder>, Option<RagIndex>), String> {
    // Resolve encoder path: explicit flag > auto-discovery.
    let encoder_path = cli
        .rag_encoder
        .clone()
        .or_else(|| crate::rag::encoder::discover_embedding_sidecar(&cli.model));

    let Some(enc_path) = encoder_path else {
        if cli.rag_index.is_some() || cli.rag_source.is_some() {
            return Err(
                "RAG index/source specified but no embedding encoder found. \
                 Provide --rag-encoder <embed.gguf> or place an embed*.gguf next to the model."
                    .to_string(),
            );
        }
        return Ok((None, None));
    };

    let mut encoder = DocumentEncoder::load(&enc_path, debug_mode)?;

    // Resolve index: load from file if it exists, otherwise build from source dir.
    let index = if let Some(ref idx_path) = cli.rag_index {
        let p = std::path::Path::new(idx_path);
        if p.exists() {
            if debug_mode {
                eprintln!("RAG: loading pre-built index from '{idx_path}'");
            }
            let idx = RagIndex::load(p)?;
            eprintln!(
                "RAG: loaded {} chunks (dim={}) from '{idx_path}'",
                idx.len(),
                encoder.dim()
            );
            Some(idx)
        } else if let Some(ref src) = cli.rag_source {
            let src_path = std::path::Path::new(src);
            let idx = RagIndex::build_from_dir(
                src_path,
                &mut encoder,
                cli.rag_max_chars_per_chunk,
                cli.rag_max_tokens_per_chunk,
                None,
                debug_mode,
            )?;
            eprintln!("RAG: saving index to '{idx_path}'…");
            idx.save(p)?;
            Some(idx)
        } else {
            return Err(format!(
                "RAG index file '{idx_path}' does not exist and --rag-source was not provided"
            ));
        }
    } else if let Some(ref src) = cli.rag_source {
        // No index file; build on-the-fly (not persisted).
        let src_path = std::path::Path::new(src);
        let idx = RagIndex::build_from_dir(
            src_path,
            &mut encoder,
            cli.rag_max_chars_per_chunk,
            cli.rag_max_tokens_per_chunk,
            None,
            debug_mode,
        )?;
        Some(idx)
    } else {
        None
    };

    Ok((Some(encoder), index))
}

pub(crate) struct ModelRuntime {
    checkpoint_path: String,
    gguf: GGUFFile,
    config: Config,
    tokenizer: Tokenizer,
    weights: TransformerWeights,
    settings: GenerationSettings,
    multimodal_weights: Option<MultimodalWeights>,
    mmproj_sidecar: Option<MmprojSidecarProbe>,
    mmproj_candidates: Vec<String>,
    vision_encoder: Option<VisionEncoder>,
    preencoded_image_embeddings: HashMap<String, MediaEmbeddingSequence>,
    audio_encoder: Option<AudioEncoder>,
    document_encoder: Option<DocumentEncoder>,
    rag_index: Option<RagIndex>,
    kv_cache_format_override: Option<crate::engine::types::KvCacheFormat>,
    kv_cache_format_logged: bool,
    /// Whether the last decode consumed its entire token budget, which means it
    /// stopped at the limit instead of at a stop token.
    last_decode_used_full_token_budget: bool,
    prefill_cache: Option<crate::app::prefill_cache::PrefixCache>,
    prefill_cache_warned: bool,
}

impl ModelRuntime {
    const DEFAULT_VIDEO_SAMPLED_FPS: u32 = 1;
    const MAX_VIDEO_DECODED_FRAMES: usize = 3600;
    const VIDEO_CHUNK_SIZE_FRAMES: usize = 32;
    const MEDIA_DECODE_RESERVE_TOKENS: usize = 256;
    /// Upper bound of the decode budget reserved for audio requests. It covers
    /// the vendor transcription cap so a request that passes context preflight
    /// cannot lose transcript text to the context limit instead.
    const AUDIO_DECODE_RESERVE_TOKENS: usize = 512;
    const VISION_ENCODER_TENSOR_PREFIXES: &'static [&'static str] = &[
        "v.",
        "vision.",
        "visual.",
        "vision_tower.",
        "vision_encoder.",
        "model.vision.",
        "model.visual.",
        "model.vision_tower.",
    ];
    const VISION_PROJECTOR_TENSOR_PREFIXES: &'static [&'static str] = &[
        "mm.",
        "mmproj.",
        "multi_modal_projector.",
        "projector.",
        "model.mmproj.",
        "model.projector.",
        "vision_language_adapter.",
        "model.vision_language_adapter.",
    ];
    const AUDIO_TENSOR_PREFIXES: &'static [&'static str] = &[
        "a.",
        "mm.a.",
        "audio.",
        "aud.",
        "speech.",
        "whisper.",
        "model.audio.",
    ];

    fn model_architecture(&self) -> &str {
        get_gguf_string_from_map(&self.gguf.kv, "general.architecture").unwrap_or("unknown")
    }

    fn gguf_has_tensor_with_any_prefix(gguf: &GGUFFile, prefixes: &[&str]) -> bool {
        gguf.tensors.iter().any(|tensor| {
            prefixes
                .iter()
                .any(|prefix| tensor.name.starts_with(prefix))
        })
    }

    fn has_vocab_token(&self, token: &str) -> bool {
        self.gguf.vocab_tokens.iter().any(|entry| entry == token)
    }

    fn has_tensor_with_any_prefix(&self, prefixes: &[&str]) -> bool {
        Self::gguf_has_tensor_with_any_prefix(&self.gguf, prefixes)
    }

    fn has_image_tokens(&self) -> bool {
        match self.config.capabilities.multimodal_backend {
            MultimodalBackend::Gemma3 => {
                self.has_vocab_token("<start_of_image>") && self.has_vocab_token("<end_of_image>")
            }
            MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
                self.has_vocab_token("<|vision_start|>")
                    && self.has_vocab_token("<|vision_end|>")
                    && self.has_vocab_token("<|image_pad|>")
            }
            MultimodalBackend::Idefics3 => self.has_vocab_token("<image>"),
            MultimodalBackend::None => false,
        }
    }

    fn has_video_tokens(&self) -> bool {
        match self.config.capabilities.multimodal_backend {
            MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
                self.has_vocab_token("<|vision_start|>")
                    && self.has_vocab_token("<|vision_end|>")
                    && self.has_vocab_token("<|video_pad|>")
            }
            MultimodalBackend::Gemma3 | MultimodalBackend::Idefics3 | MultimodalBackend::None => {
                false
            }
        }
    }

    fn has_audio_tokens(&self) -> bool {
        match self.config.capabilities.multimodal_backend {
            MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
                self.has_vocab_token("<|audio_start|>")
                    && self.has_vocab_token("<|audio_pad|>")
                    && self.has_vocab_token("<|audio_end|>")
            }
            MultimodalBackend::Gemma3 | MultimodalBackend::Idefics3 | MultimodalBackend::None => {
                false
            }
        }
    }

    fn supports_external_vision(&self) -> bool {
        // Vision encoder already loaded from embedded bytes (load_mmproj_from_bytes).
        if self.vision_encoder.is_some() {
            return true;
        }
        self.mmproj_sidecar
            .as_ref()
            .map(|probe| probe.has_vision_encoder && probe.has_vision_projector)
            .unwrap_or(false)
    }

    fn ensure_external_multimodal_initialized(
        &mut self,
        image_count: usize,
        video_count: usize,
        audio_count: usize,
    ) -> Result<(), String> {
        if image_count == 0 && video_count == 0 && audio_count == 0 {
            return Ok(());
        }
        if self.config.capabilities.multimodal_backend == MultimodalBackend::None {
            return Ok(());
        }
        let vision_requested = image_count > 0 || video_count > 0;
        let audio_requested = audio_count > 0;
        if self.mmproj_sidecar.is_some()
            && (!vision_requested || self.vision_encoder.is_some())
            && (!audio_requested || self.audio_encoder.is_some())
        {
            return Ok(());
        }

        let debug_mode = self.settings.debug_mode;
        let event_callback = self.settings.runtime_event_callback.as_ref();
        let (mmproj_sidecar, mmproj_candidates) =
            Self::probe_mmproj_sidecar(&self.checkpoint_path, &self.config, debug_mode)?;
        self.mmproj_candidates = mmproj_candidates;
        self.mmproj_sidecar = mmproj_sidecar;

        if let Some(probe) = &self.mmproj_sidecar {
            if debug_mode {
                emit_debug_line(
                    event_callback,
                    format!(
                        "Detected llama-style mmproj sidecar: path='{}', tensors={}, vision_encoder={}, vision_projector={}, audio_backend={}",
                        probe.path,
                        probe.n_tensors,
                        probe.has_vision_encoder,
                        probe.has_vision_projector,
                        probe
                            .audio_backend
                            .map(AudioEncoderBackend::as_str)
                            .unwrap_or("none")
                    ),
                );
            }
            if self.vision_encoder.is_none()
                && vision_requested
                && probe.has_vision_encoder
                && probe.has_vision_projector
            {
                let mmproj = parse_gguf_file(&probe.path, debug_mode).map_err(|e| {
                    format!(
                        "failed to load llama-style mmproj sidecar '{}' for multimodal backend initialization: {e}",
                        probe.path
                    )
                })?;
                self.vision_encoder = build_vision_encoder_from_mmproj(&self.config, mmproj)?;
            }
            if self.audio_encoder.is_none()
                && audio_requested
                && let Some(audio_backend) = probe.audio_backend
            {
                let mmproj = parse_gguf_file(&probe.path, debug_mode).map_err(|e| {
                    format!(
                        "failed to load llama-style audio mmproj sidecar '{}': {e}",
                        probe.path
                    )
                })?;
                let encoder = build_audio_encoder_from_mmproj(
                    audio_backend,
                    self.config.input_embedding_dim,
                    mmproj,
                )?;
                if debug_mode {
                    emit_debug_line(
                        event_callback,
                        format!(
                            "Loaded audio sidecar weight contract: {} (execution_ready={})",
                            encoder.contract_summary(),
                            encoder.execution_ready()
                        ),
                    );
                }
                self.audio_encoder = Some(encoder);
            }
        } else if debug_mode && !self.mmproj_candidates.is_empty() {
            emit_debug_line(
                event_callback,
                format!(
                    "No llama-style mmproj sidecar found. searched candidates: [{}]",
                    self.mmproj_candidates.join(", ")
                ),
            );
        }

        Ok(())
    }

    fn effective_supports_image(&self) -> bool {
        self.config.capabilities.supports_native_image
            || (self.has_image_tokens() && self.supports_external_vision())
    }

    fn effective_supports_video(&self) -> bool {
        self.config.capabilities.supports_native_video
            || (self.has_video_tokens() && self.supports_external_vision())
    }

    fn effective_supports_audio(&self) -> bool {
        self.has_audio_tokens()
            && (self.config.capabilities.supports_native_audio
                || self
                    .audio_encoder
                    .as_ref()
                    .map(AudioEncoder::execution_ready)
                    .unwrap_or(false))
    }

    fn mmproj_summary(&self) -> String {
        if let Some(probe) = &self.mmproj_sidecar {
            format!(
                "mmproj(path='{}', n_tensors={}, vision_encoder={}, vision_projector={}, audio_backend={}, audio_encoder_loaded={}, audio_execution_ready={})",
                probe.path,
                probe.n_tensors,
                probe.has_vision_encoder,
                probe.has_vision_projector,
                probe
                    .audio_backend
                    .map(AudioEncoderBackend::as_str)
                    .unwrap_or("none"),
                self.audio_encoder.is_some(),
                self.audio_encoder
                    .as_ref()
                    .map(AudioEncoder::execution_ready)
                    .unwrap_or(false),
            )
        } else if self.mmproj_candidates.is_empty() {
            "mmproj(path=not-searched)".to_string()
        } else {
            format!(
                "mmproj(path=not-found, searched=[{}])",
                self.mmproj_candidates.join(", ")
            )
        }
    }

    fn discover_mmproj_candidates(model_path: &str) -> Vec<PathBuf> {
        fn push_unique(candidates: &mut Vec<PathBuf>, path: PathBuf) {
            if !candidates.iter().any(|existing| existing == &path) {
                candidates.push(path);
            }
        }

        let model = Path::new(model_path);
        let parent = model.parent().unwrap_or_else(|| Path::new("."));
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        let file_name = model
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let stem = model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        let mut candidates: Vec<PathBuf> = Vec::new();
        if !file_name.is_empty() {
            push_unique(&mut candidates, parent.join(format!("mmproj-{file_name}")));
        }
        if !stem.is_empty() {
            push_unique(&mut candidates, parent.join(format!("mmproj-{stem}.gguf")));
            push_unique(&mut candidates, parent.join(format!("{stem}.mmproj.gguf")));
        }
        push_unique(&mut candidates, parent.join("mmproj.gguf"));

        if let Ok(entries) = fs::read_dir(parent) {
            let mut discovered: Vec<PathBuf> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .map(|name| {
                            let lowered = name.to_ascii_lowercase();
                            lowered.starts_with("mmproj") && lowered.ends_with(".gguf")
                        })
                        .unwrap_or(false)
                })
                .collect();
            discovered.sort();
            for path in discovered {
                push_unique(&mut candidates, path);
            }
        }

        candidates
    }

    fn strip_quant_suffix(stem: &str) -> String {
        let mut kept: Vec<&str> = Vec::new();
        for token in stem.split('-') {
            let t = token.trim().to_ascii_lowercase();
            let q_prefix_is_quant = t
                .strip_prefix('q')
                .and_then(|rest| rest.chars().next())
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);
            let is_quant_token = q_prefix_is_quant || t == "f16" || t == "f32" || t == "bf16";
            if is_quant_token {
                break;
            }
            kept.push(token);
        }
        if kept.is_empty() {
            stem.to_string()
        } else {
            kept.join("-")
        }
    }

    fn normalize_alnum_lower(input: &str) -> String {
        input
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
    }

    fn split_alnum_tokens(input: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut cur = String::new();
        for ch in input.chars() {
            if ch.is_ascii_alphanumeric() {
                cur.push(ch.to_ascii_lowercase());
            } else if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            tokens.push(cur);
        }
        tokens
    }

    fn is_size_token(token: &str) -> bool {
        token
            .strip_suffix('b')
            .map(|prefix| !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    }

    fn is_active_size_token(token: &str) -> bool {
        token
            .strip_prefix('a')
            .and_then(|t| t.strip_suffix('b'))
            .map(|middle| !middle.is_empty() && middle.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    }

    fn required_model_variant_tokens(checkpoint: &str) -> Vec<String> {
        let stem = Path::new(checkpoint)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let mut required = Vec::new();
        for token in Self::split_alnum_tokens(stem) {
            if Self::is_size_token(&token) || Self::is_active_size_token(&token) {
                required.push(token);
            }
        }
        required
    }

    fn sidecar_descriptor_tokens(sidecar_path: &str, sidecar: &GGUFFile) -> HashSet<String> {
        let mut combined = String::new();
        combined.push_str(sidecar_path);
        combined.push(' ');
        for key in [
            "general.name",
            "general.basename",
            "general.finetune",
            "general.base_model.0.name",
            "general.base_model.0.repo_url",
            "general.repo_url",
        ] {
            if let Some(value) = get_gguf_string_from_map(&sidecar.kv, key) {
                combined.push_str(value);
                combined.push(' ');
            }
        }
        Self::split_alnum_tokens(&combined).into_iter().collect()
    }

    fn validate_mmproj_variant_match(
        checkpoint: &str,
        sidecar_path: &str,
        sidecar: &GGUFFile,
    ) -> Result<(), String> {
        let required_tokens = Self::required_model_variant_tokens(checkpoint);
        if required_tokens.is_empty() {
            return Ok(());
        }
        let sidecar_tokens = Self::sidecar_descriptor_tokens(sidecar_path, sidecar);
        for token in required_tokens {
            if !sidecar_tokens.contains(&token) {
                return Err(format!(
                    "mmproj/model variant mismatch: model requires token '{token}' (derived from checkpoint name), but sidecar metadata/name does not contain it"
                ));
            }
        }
        Ok(())
    }

    fn score_mmproj_candidate(
        path: &Path,
        normalized_model_key: &str,
        backend: MultimodalBackend,
        policy: crate::vendors::VendorMultimodalPolicy,
    ) -> i32 {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let normalized_file = Self::normalize_alnum_lower(file_name);
        let mut score = 0i32;
        if !normalized_model_key.is_empty() && normalized_file.contains(normalized_model_key) {
            score += 1_000;
        }
        for hint in policy.mmproj_filename_score_hints {
            if normalized_file.contains(hint.token) {
                if backend == hint.backend {
                    score += hint.match_score;
                } else {
                    score += hint.mismatch_score;
                }
            }
        }
        if normalized_file.contains("mmproj") {
            score += 10;
        }
        score
    }

    fn probe_mmproj_sidecar(
        checkpoint: &str,
        cfg: &Config,
        debug_mode: bool,
    ) -> Result<(Option<MmprojSidecarProbe>, Vec<String>), String> {
        let candidates = Self::discover_mmproj_candidates(checkpoint);
        let candidate_strings = candidates
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let model_stem = Path::new(checkpoint)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let model_key = Self::normalize_alnum_lower(&Self::strip_quant_suffix(
            model_stem.to_lowercase().as_str(),
        ));
        let multimodal_policy = crate::vendors::multimodal_policy(cfg);
        let mut existing_candidates = candidates
            .into_iter()
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        existing_candidates.sort_by(|a, b| {
            let sa = Self::score_mmproj_candidate(
                a,
                &model_key,
                cfg.capabilities.multimodal_backend,
                multimodal_policy,
            );
            let sb = Self::score_mmproj_candidate(
                b,
                &model_key,
                cfg.capabilities.multimodal_backend,
                multimodal_policy,
            );
            sb.cmp(&sa)
                .then_with(|| a.to_string_lossy().cmp(&b.to_string_lossy()))
        });
        for path in existing_candidates {
            let sidecar_path = path.to_string_lossy().into_owned();
            let sidecar = match parse_gguf_file(&sidecar_path, debug_mode) {
                Ok(sidecar) => sidecar,
                Err(e) => {
                    if debug_mode {
                        eprintln!(
                            "Skipping mmproj sidecar '{}': failed to parse GGUF ({e})",
                            sidecar_path
                        );
                    }
                    continue;
                }
            };
            let audio_backend = match crate::vendors::validate_mmproj_for_backend(cfg, &sidecar) {
                Ok(audio_backend) => audio_backend,
                Err(e) => {
                    if debug_mode {
                        eprintln!(
                            "Skipping mmproj sidecar '{}': not compatible with backend '{}' ({e})",
                            sidecar_path,
                            cfg.capabilities.multimodal_backend.as_str()
                        );
                    }
                    continue;
                }
            };
            if let Err(e) = Self::validate_mmproj_variant_match(checkpoint, &sidecar_path, &sidecar)
            {
                if debug_mode {
                    eprintln!(
                        "Skipping mmproj sidecar '{}': checkpoint-variant mismatch ({e})",
                        sidecar_path
                    );
                }
                continue;
            }
            // Qwen3-ASR projector tensors use `mm.a.*`, which also matches the legacy
            // broad `mm.*` vision-projector prefix. The validated audio contract is
            // audio-only, so do not misreport that overlap as a vision component.
            let probe = MmprojSidecarProbe {
                path: sidecar_path.clone(),
                has_vision_encoder: audio_backend.is_none()
                    && Self::gguf_has_tensor_with_any_prefix(
                        &sidecar,
                        Self::VISION_ENCODER_TENSOR_PREFIXES,
                    ),
                has_vision_projector: audio_backend.is_none()
                    && Self::gguf_has_tensor_with_any_prefix(
                        &sidecar,
                        Self::VISION_PROJECTOR_TENSOR_PREFIXES,
                    ),
                audio_backend,
                n_tensors: sidecar.n_tensors,
            };
            return Ok((Some(probe), candidate_strings));
        }
        Ok((None, candidate_strings))
    }

    fn summarize_tensor_prefixes(&self, max_items: usize) -> String {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for tensor in &self.gguf.tensors {
            let prefix = tensor
                .name
                .split('.')
                .next()
                .unwrap_or("unknown")
                .to_string();
            *counts.entry(prefix).or_insert(0) += 1;
        }
        let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries
            .into_iter()
            .take(max_items)
            .map(|(name, count)| format!("{name}={count}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn native_media_probe_details(&self) -> String {
        let (has_image_start, has_image_end, has_image_pad, has_video_pad, has_audio_pad) =
            match self.config.capabilities.multimodal_backend {
                MultimodalBackend::Gemma3 => (
                    self.has_vocab_token("<start_of_image>"),
                    self.has_vocab_token("<end_of_image>"),
                    false,
                    false,
                    false,
                ),
                MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => (
                    self.has_vocab_token("<|vision_start|>"),
                    self.has_vocab_token("<|vision_end|>"),
                    self.has_vocab_token("<|image_pad|>"),
                    self.has_vocab_token("<|video_pad|>"),
                    self.has_vocab_token("<|audio_pad|>"),
                ),
                MultimodalBackend::Idefics3 => {
                    (false, false, self.has_vocab_token("<image>"), false, false)
                }
                MultimodalBackend::None => (false, false, false, false, false),
            };
        let has_vision_encoder =
            self.has_tensor_with_any_prefix(Self::VISION_ENCODER_TENSOR_PREFIXES);
        let has_vision_projector =
            self.has_tensor_with_any_prefix(Self::VISION_PROJECTOR_TENSOR_PREFIXES);
        let has_audio_encoder = self.has_tensor_with_any_prefix(Self::AUDIO_TENSOR_PREFIXES);
        let top_prefixes = self.summarize_tensor_prefixes(8);
        let vision_placeholders =
            has_image_start && has_image_end && (self.has_image_tokens() || has_video_pad);
        let likely_text_only_export =
            vision_placeholders && !has_vision_encoder && !has_vision_projector;
        let text_only_hint = if likely_text_only_export {
            " hint: chat-template vision placeholders exist, but no vision/projector tensor groups were found in this GGUF; this artifact is likely text-only."
        } else {
            ""
        };
        let sidecar_hint = if self.mmproj_sidecar.is_none()
            && !self
                .settings
                .vendor_multimodal_policy
                .missing_sidecar_hint
                .is_empty()
        {
            self.settings.vendor_multimodal_policy.missing_sidecar_hint
        } else {
            ""
        };

        format!(
            "probe details: arch='{}', n_tensors={}, tokens(image_start={} image_end={} image_pad={} video_pad={} audio_pad={}), tensor_groups_main(vision_encoder={} vision_projector={} audio={}), effective_support(image={} video={} audio={}), {}, top_tensor_prefixes=[{}].{}{}",
            self.model_architecture(),
            self.gguf.n_tensors,
            has_image_start,
            has_image_end,
            has_image_pad,
            has_video_pad,
            has_audio_pad,
            has_vision_encoder,
            has_vision_projector,
            has_audio_encoder,
            self.effective_supports_image(),
            self.effective_supports_video(),
            self.effective_supports_audio(),
            self.mmproj_summary(),
            top_prefixes,
            text_only_hint,
            sidecar_hint,
        )
    }

    fn ensure_native_media_support(
        &self,
        image_count: usize,
        video_count: usize,
        audio_count: usize,
    ) -> Result<(), String> {
        if image_count == 0 && video_count == 0 && audio_count == 0 {
            return Ok(());
        }

        let capabilities = self.config.capabilities;
        if capabilities.multimodal_backend == MultimodalBackend::None {
            return Err(format!(
                "media inputs require a native multimodal backend, but model architecture '{}' is text-only in this runner",
                self.model_architecture()
            ));
        }

        let backend = capabilities.multimodal_backend.as_str();
        if image_count > 0 && !self.effective_supports_image() {
            return Err(format!(
                "image inputs require native multimodal tensors/components for backend '{backend}', but capability probe reports image=false for this GGUF (image={} video={} audio={}). {}",
                capabilities.supports_native_image,
                capabilities.supports_native_video,
                capabilities.supports_native_audio,
                self.native_media_probe_details(),
            ));
        }
        if video_count > 0 && !self.effective_supports_video() {
            return Err(format!(
                "video inputs require native multimodal tensors/components for backend '{backend}', but capability probe reports video=false for this GGUF (image={} video={} audio={}). {}",
                capabilities.supports_native_image,
                capabilities.supports_native_video,
                capabilities.supports_native_audio,
                self.native_media_probe_details(),
            ));
        }
        if audio_count > 0 && !self.effective_supports_audio() {
            return Err(format!(
                "audio inputs require native multimodal tensors/components for backend '{backend}', but capability probe reports audio=false for this GGUF (image={} video={} audio={}). {}",
                capabilities.supports_native_image,
                capabilities.supports_native_video,
                capabilities.supports_native_audio,
                self.native_media_probe_details(),
            ));
        }

        Ok(())
    }

    fn media_decode_reserve(&self) -> usize {
        self.settings
            .max_tokens
            .clamp(1, Self::MEDIA_DECODE_RESERVE_TOKENS)
    }

    fn audio_decode_reserve(&self) -> usize {
        self.settings
            .max_tokens
            .clamp(1, Self::AUDIO_DECODE_RESERVE_TOKENS)
    }

    fn image_preprocess_profile(&self) -> ImagePreprocessProfile {
        let fallback_norm = ImageNormalization::MeanStd {
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.2757771],
        };
        if let Some(encoder) = &self.vision_encoder {
            let (mean, std) = encoder.recommended_image_normalization();
            let clip_norm = ImageNormalization::MeanStd { mean, std };
            let base_size = encoder.recommended_image_size().max(224);
            let align_to = encoder.recommended_image_alignment().max(1);
            if self.config.capabilities.multimodal_backend == MultimodalBackend::Qwen35 {
                // Scale image resolution with model embedding dim. At dim=2048 (2B) this
                // yields ~2/3 of the mmproj base_size; at dim=3072 (7B) the full base_size;
                // beyond that the resolution continues to grow for OCR and fine-detail tasks,
                // capped at 2× base_size where bilinear position-embedding interpolation still
                // produces reliable results.
                let balanced_size = ((base_size as f32 * self.config.dim as f32 / 3072.0) as usize)
                    .clamp(align_to.max(224), base_size * 2);
                let aligned = if align_to > 1 {
                    (balanced_size / align_to) * align_to
                } else {
                    balanced_size
                };
                let target = aligned.max(align_to).max(224);
                return ImagePreprocessProfile::new_with_mode(
                    target,
                    target,
                    clip_norm,
                    ImageResizeMode::FitWithin,
                    align_to,
                );
            }
            if self.config.capabilities.multimodal_backend == MultimodalBackend::Gemma3
                || self.config.capabilities.multimodal_backend == MultimodalBackend::Idefics3
            {
                // SigLIP-derived encoders (Gemma3, SmolVLM/idefics3) use a direct
                // bilinear stretch to the encoder's fixed square input size.
                return ImagePreprocessProfile::new_with_mode(
                    base_size,
                    base_size,
                    clip_norm,
                    ImageResizeMode::Stretch,
                    1,
                );
            }
            return ImagePreprocessProfile::new_with_mode(
                base_size,
                base_size,
                clip_norm,
                ImageResizeMode::CenterCrop,
                align_to,
            );
        }

        match self.config.capabilities.multimodal_backend {
            MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
                ImagePreprocessProfile::new(224, 224, fallback_norm)
            }
            MultimodalBackend::Gemma3 => ImagePreprocessProfile::new_with_mode(
                896,
                896,
                fallback_norm,
                ImageResizeMode::Stretch,
                1,
            ),
            MultimodalBackend::Idefics3 => ImagePreprocessProfile::new(384, 384, fallback_norm),
            MultimodalBackend::None => {
                ImagePreprocessProfile::new(448, 448, ImageNormalization::UnitRange)
            }
        }
    }

    fn create_center_square_crop_file(
        image_path: &str,
        temp_file_prefix: &str,
    ) -> Result<Option<String>, String> {
        let reader = ImageReader::open(image_path)
            .map_err(|e| format!("cannot open image '{image_path}' for detail crop: {e}"))?;
        let decoded = reader
            .decode()
            .map_err(|e| format!("cannot decode image '{image_path}' for detail crop: {e}"))?;
        let rgb = decoded.to_rgb8();
        let src_w = rgb.width() as usize;
        let src_h = rgb.height() as usize;
        if src_w == 0 || src_h == 0 {
            return Ok(None);
        }
        if src_w == src_h {
            return Ok(None);
        }

        let side = src_w.min(src_h);
        if side < 64 {
            return Ok(None);
        }
        let crop_x = (src_w - side) / 2;
        let crop_y = (src_h - side) / 2;
        let cropped =
            image::imageops::crop_imm(&rgb, crop_x as u32, crop_y as u32, side as u32, side as u32)
                .to_image();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let file_prefix = if temp_file_prefix.is_empty() {
            "gguf-runner-detail"
        } else {
            temp_file_prefix
        };
        let file_name = format!("{file_prefix}-{}-{now}.png", std::process::id());
        let out_path = std::env::temp_dir().join(file_name);
        cropped
            .save_with_format(&out_path, ImageFormat::Png)
            .map_err(|e| {
                format!(
                    "cannot save detail crop for image '{image_path}' to '{}': {e}",
                    out_path.display()
                )
            })?;
        Ok(Some(out_path.to_string_lossy().into_owned()))
    }

    fn expand_request_for_vendor_detail_crop(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationRequest, String> {
        let detail_crop_policy = self.settings.vendor_multimodal_policy.detail_crop;
        if !detail_crop_policy.enabled {
            return Ok(request.clone());
        }
        if self.config.n_layers > detail_crop_policy.max_layers {
            // Keep large variants on single-image path unless explicitly requested otherwise.
            return Ok(request.clone());
        }
        if request
            .parts
            .iter()
            .any(|part| matches!(part, ContentPart::Video(_)))
            || request
                .parts
                .iter()
                .any(|part| matches!(part, ContentPart::Audio(_)))
        {
            return Ok(request.clone());
        }

        let mut image_indices: Vec<(usize, String)> = Vec::new();
        for (idx, part) in request.parts.iter().enumerate() {
            if let ContentPart::Image(img) = part {
                image_indices.push((idx, img.path.clone()));
            }
        }
        if image_indices.len() != 1 {
            return Ok(request.clone());
        }
        let (image_part_idx, image_path) = &image_indices[0];
        let Some(crop_path) =
            Self::create_center_square_crop_file(image_path, detail_crop_policy.temp_file_prefix)?
        else {
            return Ok(request.clone());
        };

        let mut parts: Vec<ContentPart> = Vec::with_capacity(request.parts.len() + 2);
        for (idx, part) in request.parts.iter().enumerate() {
            parts.push(part.clone());
            if idx == *image_part_idx {
                if !detail_crop_policy.note_text.is_empty() {
                    parts.push(ContentPart::Text(detail_crop_policy.note_text.to_string()));
                }
                parts.push(ContentPart::Image(MediaRef {
                    path: crop_path.clone(),
                }));
            }
        }

        Ok(GenerationRequest {
            system_prompt: request.system_prompt.clone(),
            parts,
            include_empty_system_prompt: request.include_empty_system_prompt,
            assistant_prefill: request.assistant_prefill.clone(),
        })
    }

    fn validate_placeholder_spans(
        &self,
        spans: &[PlaceholderSpan],
        label: &str,
    ) -> Result<(), String> {
        let mut prev_end = 0usize;
        for span in spans {
            if span.token_len == 0 {
                return Err(format!(
                    "prompt placeholder span for {label}[{}] has zero token length",
                    span.media_index
                ));
            }
            if span.token_start < prev_end {
                return Err(format!(
                    "prompt placeholder spans for {label} overlap or are out of order around media index {}",
                    span.media_index
                ));
            }
            prev_end = span.token_start.saturating_add(span.token_len);
        }
        Ok(())
    }

    fn validate_encoded_prompt_media_alignment(
        &self,
        encoded: &EncodedPrompt,
        image_count: usize,
        video_count: usize,
        audio_count: usize,
    ) -> Result<(), String> {
        self.validate_placeholder_spans(&encoded.image_spans, "image")?;
        self.validate_placeholder_spans(&encoded.video_spans, "video")?;
        self.validate_placeholder_spans(&encoded.audio_spans, "audio")?;

        if encoded.image_spans.len() != image_count {
            return Err(format!(
                "prompt/media mismatch: encoded {} image placeholder span(s), but request contains {image_count} image input(s)",
                encoded.image_spans.len()
            ));
        }
        if encoded.video_spans.len() != video_count {
            return Err(format!(
                "prompt/media mismatch: encoded {} video placeholder span(s), but request contains {video_count} video input(s)",
                encoded.video_spans.len()
            ));
        }
        if encoded.audio_spans.len() != audio_count {
            return Err(format!(
                "prompt/media mismatch: encoded {} audio placeholder span(s), but request contains {audio_count} audio input(s)",
                encoded.audio_spans.len()
            ));
        }
        Ok(())
    }

    pub(crate) fn load(cli: &CliOptions) -> Result<Self, String> {
        Self::load_with_debug_mode(cli, cli.debug)
    }

    pub(crate) fn load_for_repl(cli: &CliOptions) -> Result<Self, String> {
        Self::load_with_debug_mode(cli, false)
    }

    /// Load a model from a filesystem path.
    ///
    /// Prefer this over `load_from_bytes` when the model is too large to embed
    /// at compile time (e.g. vision models).  The real `checkpoint_path` is
    /// recorded so that mmproj sidecar discovery works when the first media
    /// request arrives — the sidecar is initialised lazily on demand.
    pub(crate) fn load_from_file(path: &std::path::Path, debug_mode: bool) -> Result<Self, String> {
        use crate::engine::io::parse_gguf_file;

        let checkpoint = path
            .to_str()
            .ok_or("model path contains non-UTF8 characters")?;
        let gguf = parse_gguf_file(checkpoint, debug_mode)?;
        let mut config = crate::vendors::build_config_from_gguf(&gguf, debug_mode)?;
        let tokenizer_policy = crate::vendors::tokenizer_policy(&config);
        let vendor_multimodal_policy = crate::vendors::multimodal_policy(&config);
        let mut tokenizer = crate::engine::tokenizer::init_tokenizer_from_gguf(
            &gguf,
            &mut config,
            tokenizer_policy,
            debug_mode,
        )?;
        tokenizer.use_sentencepiece = config.is_gemma3;

        crate::engine::runtime::apply_context_size_overrides(&mut config, 0, debug_mode);
        let max_tokens = config.seq_len;

        let weights = crate::engine::weights::init_weights_from_gguf(&gguf, &config, debug_mode)?;
        let multimodal_weights =
            crate::engine::weights::init_multimodal_weights_from_gguf(&gguf, &config, debug_mode)?;
        let vendor_decode_policy = crate::vendors::decode_policy(&config);

        let hints = read_gguf_sampling_hints(&gguf);
        let settings = GenerationSettings {
            temperature: 0.0,
            top_k: hints.top_k.unwrap_or(0),
            top_p: hints.top_p.unwrap_or(0.9),
            sampling_seed: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_tokens,
            profiling_mode: false,
            show_timings: false,
            show_tokens: false,
            debug_mode,
            think_mode: if config.qwen_chat_template_uses_empty_think {
                crate::engine::types::ThinkMode::No
            } else {
                crate::engine::types::ThinkMode::Hidden
            },
            hidden_think_token_cap: None,
            structured_output_mode: StructuredOutputMode::None,
            vendor_decode_policy,
            vendor_multimodal_policy,
            runtime_event_callback: None,
            rag_top_k: 5,
            rag_max_chars_per_chunk: 1800,
            rag_max_tokens_per_chunk: 0,
        };

        Ok(Self {
            checkpoint_path: checkpoint.to_string(),
            gguf,
            config,
            tokenizer,
            weights,
            settings,
            multimodal_weights,
            // mmproj sidecar is probed lazily the first time media is provided.
            mmproj_sidecar: None,
            mmproj_candidates: Vec::new(),
            vision_encoder: None,
            preencoded_image_embeddings: HashMap::new(),
            audio_encoder: None,
            document_encoder: None,
            rag_index: None,
            kv_cache_format_override: None,
            kv_cache_format_logged: false,
            last_decode_used_full_token_budget: false,
            prefill_cache: None,
            prefill_cache_warned: false,
        })
    }

    /// Load a multimodal projector from bytes embedded in the binary.
    ///
    /// Call this once after [`load_from_bytes`](Self::load_from_bytes) to enable
    /// sidecar-backed media without needing a file on disk. Audio weight-contract
    /// loading is supported, but audio execution remains unavailable.
    pub(crate) fn load_mmproj_from_bytes(&mut self, data: &'static [u8]) -> Result<(), String> {
        if self.vision_encoder.is_some() || self.audio_encoder.is_some() {
            return Ok(());
        }
        use crate::engine::io::parse_gguf_from_bytes;
        let mmproj = parse_gguf_from_bytes(data, false)
            .map_err(|e| format!("failed to parse embedded mmproj: {e}"))?;
        let audio_backend = crate::vendors::validate_mmproj_for_backend(&self.config, &mmproj)
            .map_err(|e| format!("embedded mmproj is not compatible with the text model: {e}"))?;
        let n_tensors = mmproj.n_tensors;
        let has_vision_encoder = audio_backend.is_none()
            && Self::gguf_has_tensor_with_any_prefix(&mmproj, Self::VISION_ENCODER_TENSOR_PREFIXES);
        let has_vision_projector = audio_backend.is_none()
            && Self::gguf_has_tensor_with_any_prefix(
                &mmproj,
                Self::VISION_PROJECTOR_TENSOR_PREFIXES,
            );
        if let Some(backend) = audio_backend {
            self.audio_encoder = Some(build_audio_encoder_from_mmproj(
                backend,
                self.config.input_embedding_dim,
                mmproj,
            )?);
        } else {
            self.vision_encoder = build_vision_encoder_from_mmproj(&self.config, mmproj)?;
        }
        self.mmproj_sidecar = Some(MmprojSidecarProbe {
            path: "<embedded>".to_string(),
            has_vision_encoder,
            has_vision_projector,
            audio_backend,
            n_tensors,
        });
        Ok(())
    }

    /// Load a model from bytes embedded in the binary (e.g. via `include_bytes!`).
    /// Skips multimodal, RAG, and mmproj discovery. Uses conservative defaults
    /// suitable for an interactive assistant.
    pub(crate) fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        use crate::engine::io::parse_gguf_from_bytes;

        let gguf = parse_gguf_from_bytes(data, false)?;
        let mut config = crate::vendors::build_config_from_gguf(&gguf, false)?;
        let tokenizer_policy = crate::vendors::tokenizer_policy(&config);
        let vendor_multimodal_policy = crate::vendors::multimodal_policy(&config);
        let mut tokenizer = crate::engine::tokenizer::init_tokenizer_from_gguf(
            &gguf,
            &mut config,
            tokenizer_policy,
            false,
        )?;
        tokenizer.use_sentencepiece = config.is_gemma3;

        crate::engine::runtime::apply_context_size_overrides(&mut config, 0, false);
        let max_tokens = config.seq_len;

        let weights = crate::engine::weights::init_weights_from_gguf(&gguf, &config, false)?;
        let multimodal_weights =
            crate::engine::weights::init_multimodal_weights_from_gguf(&gguf, &config, false)?;
        let vendor_decode_policy = crate::vendors::decode_policy(&config);

        // Embedded usage defaults to greedy decoding (temperature=0). Interactive
        // assistants need predictable, reproducible output across sessions; stochastic
        // sampling on noisy 1-bit models produces inconsistent answers even with the
        // model's own recommended top_k/top_p. Greedy is also seed-independent, so
        // every call returns the model's highest-probability answer. Callers who want
        // variability can use the setters after loading.
        let hints = read_gguf_sampling_hints(&gguf);
        let settings = GenerationSettings {
            temperature: 0.0,
            top_k: hints.top_k.unwrap_or(0),
            top_p: hints.top_p.unwrap_or(0.9),
            sampling_seed: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_tokens,
            profiling_mode: false,
            show_timings: false,
            show_tokens: false,
            debug_mode: false,
            think_mode: if config.qwen_chat_template_uses_empty_think {
                crate::engine::types::ThinkMode::No
            } else {
                crate::engine::types::ThinkMode::Hidden
            },
            hidden_think_token_cap: None,
            structured_output_mode: StructuredOutputMode::None,
            vendor_decode_policy,
            vendor_multimodal_policy,
            runtime_event_callback: None,
            rag_top_k: 5,
            rag_max_chars_per_chunk: 1800,
            rag_max_tokens_per_chunk: 0,
        };

        Ok(Self {
            checkpoint_path: "<embedded>".to_string(),
            gguf,
            config,
            tokenizer,
            weights,
            settings,
            multimodal_weights,
            mmproj_sidecar: None,
            mmproj_candidates: Vec::new(),
            vision_encoder: None,
            preencoded_image_embeddings: HashMap::new(),
            audio_encoder: None,
            document_encoder: None,
            rag_index: None,
            kv_cache_format_override: None,
            kv_cache_format_logged: false,
            last_decode_used_full_token_budget: false,
            prefill_cache: None,
            prefill_cache_warned: false,
        })
    }

    fn load_with_debug_mode(cli: &CliOptions, debug_mode: bool) -> Result<Self, String> {
        let mut max_tokens = cli.max_tokens;
        let checkpoint = &cli.model;
        if debug_mode {
            eprintln!("Loading GGUF model: {checkpoint}");
            eprintln!(
                "Sampling (CLI): temperature={}, top_k={}, top_p={}, repeat_penalty={}, repeat_last_n={}, seed={}",
                cli.temperature
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "model/default".to_string()),
                cli.top_k
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "model/default".to_string()),
                cli.top_p
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "model/default".to_string()),
                cli.repeat_penalty,
                cli.repeat_last_n,
                cli.seed
                    .map(|seed| seed.to_string())
                    .unwrap_or_else(|| "time".to_string())
            );
        }

        let gguf = parse_gguf_file(checkpoint, debug_mode)?;

        if debug_mode {
            eprintln!(
                "GGUF metadata: version={}, tensors={}, kv={}, tensor_data_start={} bytes",
                gguf.version, gguf.n_tensors, gguf.n_kv, gguf.tensor_data_start
            );
        }

        let mut config = crate::vendors::build_config_from_gguf(&gguf, debug_mode)?;
        let tokenizer_policy = crate::vendors::tokenizer_policy(&config);
        let vendor_multimodal_policy = crate::vendors::multimodal_policy(&config);
        let vendor_runtime_debug_policy = crate::vendors::runtime_debug_policy(&config);
        let mut tokenizer = crate::engine::tokenizer::init_tokenizer_from_gguf(
            &gguf,
            &mut config,
            tokenizer_policy,
            debug_mode,
        )?;
        tokenizer.use_sentencepiece = config.is_gemma3;
        let vision_requested =
            !cli.images.is_empty() || !cli.videos.is_empty() || cli.image_batch.is_some();
        let media_requested = vision_requested || !cli.audios.is_empty();
        let (mmproj_sidecar, mmproj_candidates) = if media_requested
            && config.capabilities.multimodal_backend != MultimodalBackend::None
        {
            Self::probe_mmproj_sidecar(checkpoint, &config, debug_mode)?
        } else {
            (None, Vec::new())
        };
        if debug_mode
            && media_requested
            && config.capabilities.multimodal_backend != MultimodalBackend::None
        {
            if let Some(probe) = &mmproj_sidecar {
                eprintln!(
                    "Detected llama-style mmproj sidecar: path='{}', tensors={}, vision_encoder={}, vision_projector={}, audio_backend={}",
                    probe.path,
                    probe.n_tensors,
                    probe.has_vision_encoder,
                    probe.has_vision_projector,
                    probe
                        .audio_backend
                        .map(AudioEncoderBackend::as_str)
                        .unwrap_or("none")
                );
            } else if !mmproj_candidates.is_empty() {
                eprintln!(
                    "No llama-style mmproj sidecar found. searched candidates: [{}]",
                    mmproj_candidates.join(", ")
                );
            }
        }

        let vision_encoder = if vision_requested {
            if let Some(probe) = &mmproj_sidecar {
                if probe.has_vision_encoder && probe.has_vision_projector {
                    let mmproj = parse_gguf_file(&probe.path, debug_mode).map_err(|e| {
                        format!(
                            "failed to load llama-style mmproj sidecar '{}' for multimodal backend initialization: {e}",
                            probe.path
                        )
                    })?;
                    build_vision_encoder_from_mmproj(&config, mmproj)?
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let audio_encoder = if !cli.audios.is_empty() {
            if let Some(probe) = &mmproj_sidecar {
                if let Some(audio_backend) = probe.audio_backend {
                    let mmproj = parse_gguf_file(&probe.path, debug_mode).map_err(|e| {
                        format!(
                            "failed to load llama-style audio mmproj sidecar '{}': {e}",
                            probe.path
                        )
                    })?;
                    let encoder = build_audio_encoder_from_mmproj(
                        audio_backend,
                        config.input_embedding_dim,
                        mmproj,
                    )?;
                    if debug_mode {
                        eprintln!(
                            "Loaded audio sidecar weight contract: {} (execution_ready={})",
                            encoder.contract_summary(),
                            encoder.execution_ready()
                        );
                    }
                    Some(encoder)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        crate::engine::runtime::apply_context_size_overrides(
            &mut config,
            cli.context_size,
            debug_mode,
        );
        if cli.context_size == 0
            && debug_mode
            && let Some(label) = vendor_runtime_debug_policy.native_context_label
        {
            eprintln!(
                "Using {label} native context length {} (model may require a large workspace)",
                config.seq_len
            );
        }
        if max_tokens == 0 || max_tokens > config.seq_len {
            max_tokens = config.seq_len;
        }

        if let Some(n_threads) = cli.threads {
            crate::engine::runtime::configure_rayon_threads(n_threads, debug_mode);
        }

        let weights = crate::engine::weights::init_weights_from_gguf(&gguf, &config, debug_mode)?;
        if debug_mode && config.is_gemma3 && config.dim > 0 {
            let sample_rows = config.vocab_size.min(2048);
            if sample_rows > 0 {
                let mut min_norm = f32::INFINITY;
                let mut max_norm = 0.0f32;
                let mut sum_norm = 0.0f32;
                let scale = (config.dim as f32).sqrt();
                for row in 0..sample_rows {
                    let start = row * config.dim;
                    let end = start + config.dim;
                    let norm = weights.token_embedding_table[start..end]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        .sqrt()
                        * scale;
                    min_norm = min_norm.min(norm);
                    max_norm = max_norm.max(norm);
                    sum_norm += norm;
                }
                eprintln!(
                    "Gemma token embedding norms (scaled, sample={}): min/avg/max={:.4}/{:.4}/{:.4}",
                    sample_rows,
                    min_norm,
                    sum_norm / sample_rows as f32,
                    max_norm
                );
            }
        }
        let multimodal_weights =
            crate::engine::weights::init_multimodal_weights_from_gguf(&gguf, &config, debug_mode)?;
        let vendor_decode_policy = crate::vendors::decode_policy(&config);

        // For Qwen3-style chat templates that pre-fill an empty `<think>\n\n</think>\n\n`
        // block, the model was trained to see the closed block already in the prompt and
        // produce the answer directly. Forcing No-mode here ensures the encoder emits the
        // closed block (matching the model's training distribution) and the runtime's
        // think-tag parser starts in the "outside think" state so visible output isn't
        // suppressed. Any user-supplied --think yes/hidden is ignored for these models
        // because they don't actually reason — the flag would just confuse the model.
        let effective_think_mode = if config.qwen_chat_template_uses_empty_think {
            if debug_mode && cli.think_mode != crate::engine::types::ThinkMode::No {
                eprintln!(
                    "Note: chat template pre-fills empty <think>/</think>; overriding --think={:?} to no",
                    cli.think_mode
                );
            }
            crate::engine::types::ThinkMode::No
        } else {
            cli.think_mode
        };

        let hints = read_gguf_sampling_hints(&gguf);
        let temperature = cli.temperature.or(hints.temperature).unwrap_or(0.9);
        let top_k = cli.top_k.or(hints.top_k).unwrap_or(0);
        let top_p = cli.top_p.or(hints.top_p).unwrap_or(1.0);
        if debug_mode {
            let source = if hints.temperature.is_some() || hints.top_k.is_some() {
                " (model hints applied where not overridden)"
            } else {
                ""
            };
            eprintln!(
                "Sampling{source}: temperature={temperature}, top_k={top_k}, top_p={top_p}, repeat_penalty={}, repeat_last_n={}, seed={}",
                cli.repeat_penalty,
                cli.repeat_last_n,
                cli.seed
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "time".to_string())
            );
        }
        let settings = GenerationSettings {
            temperature,
            top_k,
            top_p,
            sampling_seed: cli.seed,
            repeat_penalty: cli.repeat_penalty,
            repeat_last_n: cli.repeat_last_n,
            max_tokens,
            profiling_mode: cli.profiling,
            show_timings: cli.show_timings,
            show_tokens: cli.show_tokens,
            debug_mode,
            think_mode: effective_think_mode,
            hidden_think_token_cap: if cli.hidden_think_token_cap == 0 {
                None
            } else {
                Some(cli.hidden_think_token_cap)
            },
            structured_output_mode: StructuredOutputMode::None,
            vendor_decode_policy,
            vendor_multimodal_policy,
            runtime_event_callback: None,
            rag_top_k: cli.rag_top_k,
            rag_max_chars_per_chunk: cli.rag_max_chars_per_chunk,
            rag_max_tokens_per_chunk: cli.rag_max_tokens_per_chunk,
        };

        // --- RAG sidecar ---
        let (document_encoder, rag_index) = load_rag_components(cli, debug_mode)?;

        Ok(Self {
            checkpoint_path: checkpoint.to_string(),
            gguf,
            config,
            tokenizer,
            weights,
            settings,
            multimodal_weights,
            mmproj_sidecar,
            mmproj_candidates,
            vision_encoder,
            preencoded_image_embeddings: HashMap::new(),
            audio_encoder,
            document_encoder,
            rag_index,
            kv_cache_format_override: None,
            kv_cache_format_logged: false,
            last_decode_used_full_token_budget: false,
            prefill_cache: None,
            prefill_cache_warned: false,
        })
    }

    /// Load a serialized prefill cache. Validates the model fingerprint; the
    /// per-request token-prefix match provides the actual correctness guard.
    pub(crate) fn set_prefill_cache_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let pc = crate::app::prefill_cache::PrefixCache::parse(bytes)?;
        let (len, fnv) = crate::app::prefill_cache::model_fingerprint(self.gguf.mapped.as_slice());
        if pc.model_len != len || pc.model_head_fnv != fnv {
            return Err("prefill cache: model fingerprint mismatch".to_string());
        }
        self.prefill_cache = Some(pc);
        Ok(())
    }

    /// Render a prefill-cache blob for `system_prompt`.
    ///
    /// The static prefix is discovered template-agnostically: encode two
    /// prompts that differ only in user text and take their token LCP —
    /// exactly the tokens every future prompt with this system prompt shares.
    /// That prefix is prefilled once and the resulting state serialized.
    pub(crate) fn render_prefill_cache_blob(
        &mut self,
        system_prompt: &str,
    ) -> Result<Vec<u8>, String> {
        use crate::vendors::{ChatMessage, ChatRole};
        let think_mode = self.settings.think_mode;
        let a = crate::vendors::encode_chat_messages(
            &mut self.tokenizer,
            &self.config,
            &[ChatMessage {
                role: ChatRole::User,
                content: "A".to_string(),
            }],
            system_prompt,
            think_mode,
        );
        let b = crate::vendors::encode_chat_messages(
            &mut self.tokenizer,
            &self.config,
            &[ChatMessage {
                role: ChatRole::User,
                content: "zq 9".to_string(),
            }],
            system_prompt,
            think_mode,
        );
        let mut k = 0usize;
        while k < a.len().min(b.len()) && a[k] == b[k] {
            k += 1;
        }
        if k == 0 {
            return Err("prefill cache: no shared static prefix".to_string());
        }
        if k >= self.config.seq_len {
            return Err("prefill cache: static prefix exceeds context size".to_string());
        }
        let tokens = a[..k].to_vec();

        let mut state = crate::engine::runtime::malloc_run_state(&self.config)?;
        let use_batch = crate::engine::switches::use_batch_prefill()
            && crate::engine::runtime::batch_prefill_supported(&self.config);
        if use_batch {
            let chunk = crate::engine::switches::batch_prefill_chunk();
            let mut scratch = crate::engine::runtime::PrefillScratch::new();
            let mut base = 0usize;
            while base < tokens.len() {
                let take = chunk.min(tokens.len() - base);
                let toks: Vec<crate::engine::runtime::PrefillInput<'_>> = tokens[base..base + take]
                    .iter()
                    .map(|&t| crate::engine::runtime::PrefillInput::Token(t as usize))
                    .collect();
                crate::engine::runtime::transformer_prefill_batch(
                    &toks,
                    base,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                    &mut scratch,
                )?;
                base += take;
            }
        } else {
            for (pos, &t) in tokens.iter().enumerate() {
                crate::engine::runtime::transformer_without_logits(
                    t as usize,
                    pos,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                )?;
            }
        }

        let (len, fnv) = crate::app::prefill_cache::model_fingerprint(self.gguf.mapped.as_slice());
        crate::app::prefill_cache::snapshot(&self.config, &state, &tokens, len, fnv)
    }

    pub(crate) fn generate_text(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        stream_stdout: bool,
    ) -> Result<String, String> {
        self.generate_text_with_images(prompt, system_prompt, &[], stream_stdout)
    }

    pub(crate) fn generate_text_without_think(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        stream_stdout: bool,
    ) -> Result<String, String> {
        let original_think_mode = self.settings.think_mode;
        self.settings.think_mode = ThinkMode::No;
        let result = self.generate_text(prompt, system_prompt, stream_stdout);
        self.settings.think_mode = original_think_mode;
        result
    }

    pub(crate) fn generate_text_for_agent(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        stream_stdout: bool,
    ) -> Result<String, String> {
        let original_temperature = self.settings.temperature;
        let original_top_k = self.settings.top_k;
        let original_top_p = self.settings.top_p;
        let original_max_tokens = self.settings.max_tokens;
        let original_think_mode = self.settings.think_mode;
        let original_show_tokens = self.settings.show_tokens;
        let original_debug_mode = self.settings.debug_mode;
        let original_structured_output_mode = self.settings.structured_output_mode;
        let decode_policy = self.settings.vendor_decode_policy;

        // Agent protocol adherence is more stable with no think tags and a bounded
        // decode budget. Pure argmax can degenerate on some models, so prefer a
        // conservative low-temperature/top-k profile instead.
        if decode_policy.agent_force_deterministic {
            self.settings.temperature = 0.0;
            self.settings.top_k = 1;
            self.settings.top_p = 1.0;
            self.settings.max_tokens = original_max_tokens.clamp(96, 256);
        } else {
            self.settings.temperature = original_temperature.clamp(0.15, 0.35);
            self.settings.top_k = if original_top_k == 0 {
                40
            } else {
                original_top_k.clamp(8, 64)
            };
            self.settings.top_p = original_top_p.clamp(0.8, 0.95);
            self.settings.max_tokens = original_max_tokens.clamp(128, 384);
        }
        self.settings.think_mode = ThinkMode::No;
        self.settings.show_tokens = false;
        self.settings.structured_output_mode = StructuredOutputMode::AgentJson;
        let result = self.generate_text(prompt, system_prompt, stream_stdout);

        self.settings.temperature = original_temperature;
        self.settings.top_k = original_top_k;
        self.settings.top_p = original_top_p;
        self.settings.max_tokens = original_max_tokens;
        self.settings.think_mode = original_think_mode;
        self.settings.show_tokens = original_show_tokens;
        self.settings.structured_output_mode = original_structured_output_mode;
        self.settings.debug_mode = original_debug_mode;

        result
    }

    pub(crate) fn generate_chat_messages_for_repl(
        &mut self,
        messages: &[crate::vendors::ChatMessage],
        system_prompt: &str,
    ) -> Result<String, String> {
        let debug_mode = self.settings.debug_mode;

        // RAG: augment system_prompt with retrieved context, using the last user message as query.
        let rag_system: String;
        let system_prompt = if let (Some(enc), Some(rag_index)) =
            (self.document_encoder.as_mut(), self.rag_index.as_ref())
        {
            let query_text = messages
                .iter()
                .rev()
                .find(|m| m.role == crate::vendors::ChatRole::User)
                .map(|m| m.content.as_str())
                .unwrap_or("");
            let prefixed = if enc.query_prefix().is_empty() {
                query_text.to_string()
            } else {
                format!("{}{query_text}", enc.query_prefix())
            };
            let query_emb = enc.embed(&prefixed)?;
            let chunks = rag_index.query(&query_emb, query_text, self.settings.rag_top_k);
            rag_system = prepend_rag_context(&chunks, system_prompt);
            &rag_system
        } else {
            system_prompt
        };

        let mut active_messages = messages.to_vec();
        let mut prompt_tokens: Vec<i32> = crate::vendors::encode_chat_messages(
            &mut self.tokenizer,
            &self.config,
            &active_messages,
            system_prompt,
            self.settings.think_mode,
        );

        while prompt_tokens.len() > self.config.seq_len && active_messages.len() > 1 {
            // Preserve the newest exchange when the rolling chat transcript outgrows context.
            active_messages.remove(0);
            prompt_tokens = crate::vendors::encode_chat_messages(
                &mut self.tokenizer,
                &self.config,
                &active_messages,
                system_prompt,
                self.settings.think_mode,
            );
        }

        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token);
        }
        if prompt_tokens.len() > self.config.seq_len {
            prompt_tokens =
                prompt_tokens.split_off(prompt_tokens.len().saturating_sub(self.config.seq_len));
        }
        if debug_mode {
            emit_debug_line(
                self.settings.runtime_event_callback.as_ref(),
                format!("Prompt tokens: {}", prompt_tokens.len()),
            );
            if active_messages.len() != messages.len() {
                emit_debug_line(
                    self.settings.runtime_event_callback.as_ref(),
                    format!(
                        "Trimmed chat history to {} message(s) to fit context window",
                        active_messages.len()
                    ),
                );
            }
            let preview = prompt_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            emit_debug_line(
                self.settings.runtime_event_callback.as_ref(),
                format!("Prompt token ids: [{preview}]"),
            );
        }

        let prefill_injected_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
        let output =
            self.generate_from_prefill(prompt_tokens, prefill_injected_embeddings, None, false)?;
        let request = GenerationRequest {
            system_prompt: system_prompt.to_string(),
            parts: vec![ContentPart::Text(
                active_messages
                    .last()
                    .map(|msg| msg.content.clone())
                    .unwrap_or_default(),
            )],
            include_empty_system_prompt: false,
            assistant_prefill: None,
        };
        self.retry_without_think_for_request(output, &request, false)
    }

    pub(crate) fn generate_chat_messages_without_think_for_repl(
        &mut self,
        messages: &[crate::vendors::ChatMessage],
        system_prompt: &str,
    ) -> Result<String, String> {
        let original_think_mode = self.settings.think_mode;
        self.settings.think_mode = ThinkMode::No;
        let result = self.generate_chat_messages_for_repl(messages, system_prompt);
        self.settings.think_mode = original_think_mode;
        result
    }

    pub(crate) fn set_runtime_event_callback(&mut self, callback: Option<RuntimeEventCallback>) {
        self.settings.runtime_event_callback = callback;
    }

    pub(crate) fn runtime_event_callback(&self) -> Option<RuntimeEventCallback> {
        self.settings.runtime_event_callback.clone()
    }

    pub(crate) fn vendor_decode_policy(&self) -> crate::vendors::VendorDecodePolicy {
        self.settings.vendor_decode_policy
    }

    pub(crate) fn settings(&self) -> &GenerationSettings {
        &self.settings
    }

    pub(crate) fn set_debug_mode(&mut self, enabled: bool) {
        self.settings.debug_mode = enabled;
    }

    pub(crate) fn set_show_tokens(&mut self, enabled: bool) {
        self.settings.show_tokens = enabled;
    }

    pub(crate) fn set_temperature(&mut self, temperature: f32) {
        self.settings.temperature = temperature;
    }

    pub(crate) fn set_top_k(&mut self, top_k: usize) {
        self.settings.top_k = top_k;
    }

    pub(crate) fn set_top_p(&mut self, top_p: f32) {
        self.settings.top_p = top_p;
    }

    pub(crate) fn set_repeat_penalty(&mut self, repeat_penalty: f32) {
        self.settings.repeat_penalty = repeat_penalty;
    }

    pub(crate) fn set_sampling_seed(&mut self, seed: Option<u64>) {
        self.settings.sampling_seed = seed;
    }

    pub(crate) fn set_think_mode_hidden(&mut self) {
        self.settings.think_mode = ThinkMode::Hidden;
    }

    pub(crate) fn set_hidden_think_token_cap(&mut self, cap: usize) {
        self.settings.hidden_think_token_cap = if cap == 0 { None } else { Some(cap) };
    }

    pub(crate) fn set_think_mode_no(&mut self) {
        self.settings.think_mode = ThinkMode::No;
    }

    /// Override the model's effective context length.
    ///
    /// Clamps `config.seq_len` (and `settings.max_tokens` when it would
    /// exceed the new limit) so that subsequent generations allocate a
    /// smaller KV cache and truncate to the override on prefill. Passing
    /// `0` is a no-op, matching `apply_context_size_overrides`.
    pub(crate) fn set_context_size(&mut self, context_size: usize) {
        if context_size == 0 {
            return;
        }
        self.config.seq_len = context_size;
        if self.settings.max_tokens == 0 || self.settings.max_tokens > context_size {
            self.settings.max_tokens = context_size;
        }
    }

    /// Load (or reload) an embedding encoder and build a RAG index from `source_dir`.
    /// Returns a human-readable status string on success.
    pub(crate) fn load_rag_from_dir(
        &mut self,
        encoder_path: Option<&str>,
        source_dir: &str,
        progress: Option<std::sync::Arc<dyn Fn(String) + Send + Sync>>,
    ) -> Result<String, String> {
        let debug = self.settings.debug_mode;
        let enc_path = encoder_path
            .map(|s| s.to_string())
            .or_else(|| crate::rag::encoder::discover_embedding_sidecar(&self.checkpoint_path))
            .ok_or_else(|| {
                "no embedding encoder found — use /rag-encoder <path> first".to_string()
            })?;

        let mut encoder = DocumentEncoder::load(&enc_path, debug)?;
        let src = std::path::Path::new(source_dir);
        let index = RagIndex::build_from_dir(
            src,
            &mut encoder,
            self.settings.rag_max_chars_per_chunk,
            self.settings.rag_max_tokens_per_chunk,
            progress,
            debug,
        )?;
        let summary = format!(
            "RAG: {} chunks loaded from '{}' (encoder: '{}')",
            index.len(),
            source_dir,
            enc_path
        );
        self.document_encoder = Some(encoder);
        self.rag_index = Some(index);
        Ok(summary)
    }

    /// Build a RAG index from in-memory `(source_name, markdown_content)` pairs
    /// using an embedding encoder loaded from bytes embedded in the binary.
    pub(crate) fn load_rag_from_embedded_docs(
        &mut self,
        encoder_bytes: &'static [u8],
        docs: &[(&str, &str)],
    ) -> Result<String, String> {
        let mut encoder = crate::rag::encoder::DocumentEncoder::load_from_bytes(encoder_bytes)?;
        let index = crate::rag::RagIndex::build_from_text_slices(
            docs,
            &mut encoder,
            self.settings.rag_max_chars_per_chunk,
            self.settings.rag_max_tokens_per_chunk,
        )?;
        let chunk_count = index.len();
        self.document_encoder = Some(encoder);
        self.rag_index = Some(index);
        Ok(format!(
            "RAG: {chunk_count} chunks indexed from embedded docs"
        ))
    }

    /// Load a precomputed serialized RAG index plus an embedding encoder, both
    /// from bytes embedded in the binary.  Avoids the CPU-heavy per-chunk
    /// embedding work that `load_rag_from_embedded_docs` does at startup —
    /// the chunks were already embedded at build time.
    pub(crate) fn load_rag_from_serialized_bytes(
        &mut self,
        encoder_bytes: &'static [u8],
        index_bytes: &[u8],
    ) -> Result<String, String> {
        let encoder = crate::rag::encoder::DocumentEncoder::load_from_bytes(encoder_bytes)?;
        let index = crate::rag::RagIndex::load_from_bytes(index_bytes)?;
        let chunk_count = index.len();
        self.document_encoder = Some(encoder);
        self.rag_index = Some(index);
        Ok(format!(
            "RAG: {chunk_count} chunks loaded from precomputed index"
        ))
    }

    /// Drop the active RAG encoder and index.
    pub(crate) fn clear_rag(&mut self) {
        self.document_encoder = None;
        self.rag_index = None;
    }

    /// Returns true if a RAG index and encoder are both loaded.
    pub(crate) fn has_rag_index(&self) -> bool {
        self.rag_index.is_some() && self.document_encoder.is_some()
    }

    /// Query the RAG index and return a formatted knowledge block string.
    /// Returns an error string on failure (embedding error, etc.).
    pub(crate) fn search_rag(&mut self, query: &str, top_k: usize) -> Result<String, String> {
        let (Some(index), Some(enc)) = (self.rag_index.as_ref(), self.document_encoder.as_mut())
        else {
            return Err("no RAG index loaded".to_string());
        };
        let prefixed = if enc.query_prefix().is_empty() {
            query.to_string()
        } else {
            format!("{}{query}", enc.query_prefix())
        };
        let query_emb = enc.embed(&prefixed)?;
        let chunks = index.query(&query_emb, query, top_k);
        if chunks.is_empty() {
            return Ok("No relevant knowledge found.".to_string());
        }
        let mut out = String::new();
        for chunk in chunks {
            out.push('[');
            out.push_str(&chunk.source);
            out.push_str("]\n");
            out.push_str(&chunk.text);
            out.push_str("\n\n");
        }
        Ok(out.trim_end().to_string())
    }

    pub(crate) fn generate_request(
        &mut self,
        request: &GenerationRequest,
        stream_stdout: bool,
    ) -> Result<String, String> {
        self.generate_request_with_rag(request, stream_stdout, true)
    }

    /// Prepare and jointly encode a bounded set of independent image jobs.
    /// The resulting embeddings are consumed by subsequent normal generation
    /// requests whose image paths match this chunk.
    pub(crate) fn preencode_image_batch(
        &mut self,
        image_paths: &[String],
    ) -> Vec<Result<(), String>> {
        self.preencoded_image_embeddings.clear();
        if image_paths.is_empty() {
            return Vec::new();
        }

        let mut unique_paths = Vec::new();
        let mut unique_by_path = HashMap::new();
        let mut request_to_unique = Vec::with_capacity(image_paths.len());
        for path in image_paths {
            let unique_index = if let Some(&index) = unique_by_path.get(path) {
                index
            } else {
                let index = unique_paths.len();
                unique_paths.push(path.clone());
                unique_by_path.insert(path.clone(), index);
                index
            };
            request_to_unique.push(unique_index);
        }

        if let Err(error) = self.ensure_external_multimodal_initialized(unique_paths.len(), 0, 0) {
            return vec![Err(error); image_paths.len()];
        }
        if let Err(error) = self.ensure_native_media_support(unique_paths.len(), 0, 0) {
            return vec![
                Err(format!(
                    "native multimodal execution required (fallback disabled): {error}"
                ));
                image_paths.len()
            ];
        }
        let Some(encoder) = self.vision_encoder.as_ref() else {
            return vec![
                Err(format!(
                    "image batch requires a compatible native vision encoder; {}",
                    self.mmproj_summary()
                ));
                image_paths.len()
            ];
        };

        let image_profile = self.image_preprocess_profile();
        let mut unique_errors = vec![None; unique_paths.len()];
        let mut prepared_unique_indices = Vec::new();
        let mut prepared_images = Vec::new();
        for (unique_index, path) in unique_paths.iter().enumerate() {
            match prepare_images_for_multimodal(std::slice::from_ref(path), image_profile) {
                Ok(mut prepared) => {
                    prepared_unique_indices.push(unique_index);
                    prepared_images.push(prepared.remove(0));
                }
                Err(error) => unique_errors[unique_index] = Some(error),
            }
        }

        if !prepared_images.is_empty() {
            match encoder.encode_images(&prepared_images) {
                Ok(sequences) if sequences.len() == prepared_images.len() => {
                    for ((&unique_index, prepared), sequence) in prepared_unique_indices
                        .iter()
                        .zip(prepared_images.iter())
                        .zip(sequences)
                    {
                        self.preencoded_image_embeddings
                            .insert(prepared.path.clone(), sequence);
                        unique_errors[unique_index] = None;
                    }
                }
                batch_result => {
                    let batch_error = match batch_result {
                        Ok(_) => "vision microbatch returned the wrong result count".to_string(),
                        Err(error) => error,
                    };
                    if self.settings.debug_mode {
                        emit_debug_line(
                            self.settings.runtime_event_callback.as_ref(),
                            format!(
                                "Image vision microbatch failed ({batch_error}); retrying each image independently"
                            ),
                        );
                    }
                    for (&unique_index, prepared) in
                        prepared_unique_indices.iter().zip(prepared_images.iter())
                    {
                        match encoder.encode_images(std::slice::from_ref(prepared)) {
                            Ok(mut sequence) if sequence.len() == 1 => {
                                self.preencoded_image_embeddings
                                    .insert(prepared.path.clone(), sequence.remove(0));
                                unique_errors[unique_index] = None;
                            }
                            Ok(_) => {
                                unique_errors[unique_index] = Some(
                                    "vision encoder returned the wrong result count".to_string(),
                                );
                            }
                            Err(error) => unique_errors[unique_index] = Some(error),
                        }
                    }
                }
            }
        }

        if self.settings.debug_mode {
            emit_debug_line(
                self.settings.runtime_event_callback.as_ref(),
                format!(
                    "Image vision microbatch: records={}, unique={}, encoded={}, failed={}",
                    image_paths.len(),
                    unique_paths.len(),
                    self.preencoded_image_embeddings.len(),
                    unique_errors.iter().filter(|error| error.is_some()).count()
                ),
            );
        }

        request_to_unique
            .into_iter()
            .map(|unique_index| {
                if let Some(error) = &unique_errors[unique_index] {
                    Err(error.clone())
                } else if self
                    .preencoded_image_embeddings
                    .contains_key(&unique_paths[unique_index])
                {
                    Ok(())
                } else {
                    Err("vision microbatch did not retain an embedding".to_string())
                }
            })
            .collect()
    }

    pub(crate) fn transcribe_audio(
        &mut self,
        audio_path: &str,
        context: &str,
        language: Option<&str>,
    ) -> Result<AudioTranscriptionResult, String> {
        self.ensure_external_multimodal_initialized(0, 0, 1)?;
        let audio_backend = self
            .audio_encoder
            .as_ref()
            .map(AudioEncoder::backend)
            .ok_or_else(|| {
                format!(
                    "audio transcription requires a supported audio encoder sidecar; {}",
                    self.mmproj_summary()
                )
            })?;
        let policy = crate::vendors::audio_transcription_policy(audio_backend);
        let prepared = policy.build_request(audio_path, context, language)?;

        let original_temperature = self.settings.temperature;
        let original_top_k = self.settings.top_k;
        let original_top_p = self.settings.top_p;
        let original_repeat_penalty = self.settings.repeat_penalty;
        let original_repeat_last_n = self.settings.repeat_last_n;
        let original_max_tokens = self.settings.max_tokens;
        let original_think_mode = self.settings.think_mode;
        let original_show_tokens = self.settings.show_tokens;
        let original_structured_output_mode = self.settings.structured_output_mode;
        let original_decode_policy = self.settings.vendor_decode_policy;
        let original_kv_cache_format_override = self.kv_cache_format_override;

        self.settings.temperature = 0.0;
        self.settings.top_k = 0;
        self.settings.top_p = 1.0;
        self.settings.repeat_penalty = 1.0;
        self.settings.repeat_last_n = 0;
        let decode_budget = original_max_tokens.clamp(1, policy.max_new_tokens);
        self.settings.max_tokens = decode_budget;
        self.settings.think_mode = ThinkMode::No;
        self.settings.show_tokens = false;
        self.settings.structured_output_mode = StructuredOutputMode::None;
        self.settings.vendor_decode_policy = policy.decode_policy;
        self.kv_cache_format_override = Some(policy.kv_cache_format);

        // Transcription context is model input, not a retrieval query. Keep the
        // official system-context/audio-only message shape even when RAG is loaded.
        let output = self.generate_request_with_rag(&prepared.generation_request, false, false);

        self.settings.temperature = original_temperature;
        self.settings.top_k = original_top_k;
        self.settings.top_p = original_top_p;
        self.settings.repeat_penalty = original_repeat_penalty;
        self.settings.repeat_last_n = original_repeat_last_n;
        self.settings.max_tokens = original_max_tokens;
        self.settings.think_mode = original_think_mode;
        self.settings.show_tokens = original_show_tokens;
        self.settings.structured_output_mode = original_structured_output_mode;
        self.settings.vendor_decode_policy = original_decode_policy;
        self.kv_cache_format_override = original_kv_cache_format_override;

        if output.is_ok() && self.last_decode_used_full_token_budget {
            emit_cli_info_line(
                self.settings.runtime_event_callback.as_ref(),
                format!(
                    "Warning: transcript stopped at the {decode_budget}-token transcription limit for '{audio_path}'; it covers only the beginning of this audio. Split long recordings into shorter files.",
                ),
                false,
                false,
            );
        }

        output.map(|raw| policy.parse_output(&raw, prepared.forced_language.as_deref()))
    }

    fn generate_request_with_rag(
        &mut self,
        request: &GenerationRequest,
        stream_stdout: bool,
        allow_rag: bool,
    ) -> Result<String, String> {
        let buffered_visible_think_stdout = should_buffer_visible_think_stdout(
            stream_stdout,
            self.settings.runtime_event_callback.is_some(),
            self.settings.think_mode,
            self.settings.vendor_decode_policy,
        );
        let effective_stream_stdout = stream_stdout && !buffered_visible_think_stdout;
        // RAG: augment system_prompt with retrieved context if an index is loaded.
        let rag_augmented = if allow_rag {
            if let (Some(enc), Some(idx)) =
                (self.document_encoder.as_mut(), self.rag_index.as_ref())
            {
                let user_text: String = request
                    .parts
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::Text(t) = p {
                            Some(t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let top_k = self.settings.rag_top_k;
                let prefixed = if enc.query_prefix().is_empty() {
                    user_text.clone()
                } else {
                    format!("{}{user_text}", enc.query_prefix())
                };
                let query_emb = enc.embed(&prefixed)?;
                let chunks = idx.query(&query_emb, &user_text, top_k);
                let new_system = prepend_rag_context(&chunks, &request.system_prompt);
                Some(GenerationRequest {
                    system_prompt: new_system,
                    parts: request.parts.clone(),
                    include_empty_system_prompt: request.include_empty_system_prompt,
                    assistant_prefill: request.assistant_prefill.clone(),
                })
            } else {
                None
            }
        } else {
            None
        };
        let request = rag_augmented.as_ref().unwrap_or(request);

        let effective_request = self.expand_request_for_vendor_detail_crop(request)?;
        let media_requested = effective_request.parts.iter().any(|part| {
            matches!(
                part,
                ContentPart::Image(_) | ContentPart::Video(_) | ContentPart::Audio(_)
            )
        });
        let original_think_mode = self.settings.think_mode;
        let override_hidden_think = media_requested
            && self
                .settings
                .vendor_decode_policy
                .prefer_hidden_think_for_multimodal
            && self.settings.think_mode == ThinkMode::Yes;
        if override_hidden_think {
            if self.settings.debug_mode {
                emit_debug_line(
                    self.settings.runtime_event_callback.as_ref(),
                    "Note: using hidden think mode for multimodal turn per vendor decode policy",
                );
            }
            self.settings.think_mode = ThinkMode::Hidden;
        }
        let result = (|| -> Result<String, String> {
            let mut prompt_parts: Vec<&str> = Vec::new();
            let mut images: Vec<String> = Vec::new();
            let mut videos: Vec<String> = Vec::new();
            let mut audios: Vec<String> = Vec::new();

            for part in &effective_request.parts {
                match part {
                    ContentPart::Text(text) => prompt_parts.push(text),
                    ContentPart::Image(image) => images.push(image.path.clone()),
                    ContentPart::Video(video) => videos.push(video.path.clone()),
                    ContentPart::Audio(audio) => audios.push(audio.path.clone()),
                }
            }

            let prompt = prompt_parts.join("\n");
            if prompt.trim().is_empty()
                && images.is_empty()
                && videos.is_empty()
                && audios.is_empty()
            {
                return Err("generation request has no text content".to_string());
            }

            self.ensure_external_multimodal_initialized(images.len(), videos.len(), audios.len())?;
            self.ensure_native_media_support(images.len(), videos.len(), audios.len())
                .map_err(|e| {
                    format!("native multimodal execution required (fallback disabled): {e}")
                })?;
            let image_profile = self.image_preprocess_profile();
            let event_callback = self.settings.runtime_event_callback.as_ref();
            if (!images.is_empty() || !videos.is_empty() || !audios.is_empty())
                && self.multimodal_weights.is_none()
                && self.vision_encoder.is_none()
                && self.audio_encoder.is_none()
            {
                return Err(format!(
                    "native media path selected but multimodal weights for backend '{}' were not initialized",
                    self.config.capabilities.multimodal_backend.as_str()
                ));
            }

            let encoded_prompt = crate::vendors::encode_generation_request(
                &mut self.tokenizer,
                &self.config,
                &effective_request,
                self.settings.think_mode,
            );
            self.validate_encoded_prompt_media_alignment(
                &encoded_prompt,
                images.len(),
                videos.len(),
                audios.len(),
            )?;
            if self.settings.debug_mode {
                emit_debug_line(
                    event_callback,
                    format!(
                        "Encoded prompt: tokens={}, image_spans={}, video_spans={}, audio_spans={}",
                        encoded_prompt.token_ids.len(),
                        encoded_prompt.image_spans.len(),
                        encoded_prompt.video_spans.len(),
                        encoded_prompt.audio_spans.len()
                    ),
                );
                // Dump full prompt token list so we can verify the chat template.
                let prompt_preview: Vec<String> = encoded_prompt
                    .token_ids
                    .iter()
                    .map(|&id| {
                        let text = self
                            .tokenizer
                            .decode_token(id)
                            .unwrap_or_else(|| format!("?{id}"))
                            .replace('\n', "\\n")
                            .replace('\r', "\\r");
                        format!("{id}(\"{text}\")")
                    })
                    .collect();
                emit_debug_line(
                    event_callback,
                    format!("Prompt tokens: [{}]", prompt_preview.join(", ")),
                );
            }

            let mut planned_audio_token_counts = Vec::new();
            if !audios.is_empty() {
                let audio_encoder = self.audio_encoder.as_ref().ok_or_else(|| {
                    "audio context preflight requires a loaded audio encoder sidecar".to_string()
                })?;
                let preprocess_config =
                    crate::vendors::audio_preprocess_config(audio_encoder.backend());
                let audio_plans = probe_audios_for_multimodal(&audios, preprocess_config)?;
                planned_audio_token_counts = audio_plans
                    .iter()
                    .map(|plan| audio_encoder.planned_embedding_token_count(&plan.feature_windows))
                    .collect::<Result<Vec<_>, _>>()?;
                let total_audio_tokens =
                    planned_audio_token_counts
                        .iter()
                        .try_fold(0usize, |total, count| {
                            total
                                .checked_add(*count)
                                .ok_or_else(|| "planned audio embedding count overflow".to_string())
                        })?;
                let total_target_samples = audio_plans.iter().try_fold(0usize, |total, plan| {
                    total
                        .checked_add(plan.target_sample_count)
                        .ok_or_else(|| "planned audio sample count overflow".to_string())
                })?;
                let duration_seconds = total_target_samples as f64
                    / f64::from(preprocess_config.decode.target_sample_rate);

                if images.is_empty() && videos.is_empty() {
                    preflight_media_context(
                        &encoded_prompt,
                        &[],
                        &planned_audio_token_counts,
                        self.config.seq_len,
                        self.audio_decode_reserve(),
                    )
                    .map_err(|error| {
                        format!(
                            "audio context preflight failed for {} file(s), {:.2}s total, {total_audio_tokens} projected token(s): {error}",
                            audio_plans.len(), duration_seconds
                        )
                    })?;
                }
                if self.settings.debug_mode {
                    emit_debug_line(
                        event_callback,
                        format!(
                            "Audio context plan: files={}, duration={duration_seconds:.2}s, projected_tokens={total_audio_tokens}, decode_reserve={}",
                            audio_plans.len(),
                            self.audio_decode_reserve()
                        ),
                    );
                }
            }

            let mut preprocess_summary: Vec<String> = Vec::new();
            let mut prepared_images = Vec::new();
            let mut prepared_audios = Vec::new();

            if !images.is_empty() {
                prepared_images = prepare_images_for_multimodal(&images, image_profile)?;
                if self.settings.debug_mode {
                    let first = &prepared_images[0];
                    emit_debug_line(
                        event_callback,
                        format!(
                            "Prepared {} image tensor(s); first image: path='{}', width={}, height={}, elements={}",
                            prepared_images.len(),
                            first.path,
                            first.width,
                            first.height,
                            first.element_count()
                        ),
                    );
                }
                preprocess_summary.push(format!("images={}", prepared_images.len()));
            }
            if !videos.is_empty() {
                let prepared = prepare_videos_for_multimodal(
                    &videos,
                    image_profile,
                    Self::DEFAULT_VIDEO_SAMPLED_FPS,
                    Self::MAX_VIDEO_DECODED_FRAMES,
                    Self::VIDEO_CHUNK_SIZE_FRAMES,
                )?;
                if self.settings.debug_mode {
                    let first = &prepared[0];
                    let (chunk_start, chunk_frames, decoded_chunk_frames) =
                        if let Some(chunk0) = first.chunks.first() {
                            let decoded = load_video_chunk_tensors(first, 0, image_profile)?;
                            (chunk0.start_frame, chunk0.frame_paths.len(), decoded.len())
                        } else {
                            (0, 0, 0)
                        };
                    emit_debug_line(
                        event_callback,
                        format!(
                            "Prepared {} video tensor(s); first video: path='{}', fps={}, size={}x{}, frames={}, chunks={}, first_chunk_start={}, first_chunk_frames={}, first_chunk_decoded={}",
                            prepared.len(),
                            first.path,
                            first.sampled_fps,
                            first.frame_width,
                            first.frame_height,
                            first.frame_count,
                            first.chunks.len(),
                            chunk_start,
                            chunk_frames,
                            decoded_chunk_frames
                        ),
                    );
                }
                preprocess_summary.push(format!("videos={}", prepared.len()));
            }
            if !audios.is_empty() {
                let audio_encoder = self.audio_encoder.as_ref().ok_or_else(|| {
                    "audio preprocessing requires a loaded audio encoder sidecar".to_string()
                })?;
                let preprocess_config =
                    crate::vendors::audio_preprocess_config(audio_encoder.backend());
                prepared_audios = prepare_audios_for_multimodal(&audios, preprocess_config)?;
                if self.settings.debug_mode {
                    let first = &prepared_audios[0];
                    let first_chunk_samples = if !first.chunks.is_empty() {
                        load_audio_chunk_samples(first, 0)?.len()
                    } else {
                        0
                    };
                    let (
                        first_feature_start,
                        first_feature_valid,
                        first_feature_padded,
                        mel_bins,
                        first_feature_elements,
                    ) = first
                        .feature_windows
                        .first()
                        .map_or((0, 0, 0, 0, 0), |window| {
                            (
                                window.start_frame,
                                window.valid_frames,
                                window.padded_frames,
                                window.mel_bins,
                                window.data_mel_major.len(),
                            )
                        });
                    emit_debug_line(
                        event_callback,
                        format!(
                            "Prepared {} audio tensor(s); first audio: path='{}', source_rate={}, source_channels={}, sample_rate={}, channels={}, samples={}, chunks={}, feature_windows={}, first_feature_start={}, first_feature_valid={}, first_feature_padded={}, mel_bins={}, first_feature_elements={}, first_chunk_samples={}",
                            prepared_audios.len(),
                            first.path,
                            first.source_sample_rate,
                            first.source_channels,
                            first.sample_rate,
                            first.channels,
                            first.total_samples,
                            first.chunks.len(),
                            first.feature_windows.len(),
                            first_feature_start,
                            first_feature_valid,
                            first_feature_padded,
                            mel_bins,
                            first_feature_elements,
                            first_chunk_samples
                        ),
                    );
                }
                preprocess_summary.push(format!("audios={}", prepared_audios.len()));
            }

            let mut prefill_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
            let mut rope_position_plan = None;
            let mut prompt_tokens = encoded_prompt.token_ids.clone();
            let mut image_embeddings = Vec::new();

            if !prepared_images.is_empty() {
                let encoder = self.vision_encoder.as_ref().ok_or_else(|| {
                    let mmproj_note = self
                        .mmproj_sidecar
                        .as_ref()
                        .map(|probe| {
                            format!(" (llama-style mmproj sidecar loaded: '{}')", probe.path)
                        })
                        .unwrap_or_default();
                    format!(
                        "native image preprocessing succeeded ({}), but no compatible native vision encoder is initialized for backend '{}'{}",
                        preprocess_summary.join(", "),
                        self.config.capabilities.multimodal_backend.as_str(),
                        mmproj_note
                    )
                })?;
                let mut resolved_embeddings = vec![None; prepared_images.len()];
                let mut missing_indices = Vec::new();
                let mut missing_images = Vec::new();
                for (index, prepared) in prepared_images.iter().enumerate() {
                    if let Some(sequence) = self.preencoded_image_embeddings.get(&prepared.path) {
                        resolved_embeddings[index] = Some(sequence.clone());
                    } else {
                        missing_indices.push(index);
                        missing_images.push(prepared.clone());
                    }
                }
                if !missing_images.is_empty() {
                    let encoded_missing = encoder.encode_images(&missing_images)?;
                    if encoded_missing.len() != missing_indices.len() {
                        return Err(
                            "vision encoder returned the wrong number of image embeddings"
                                .to_string(),
                        );
                    }
                    for (index, sequence) in missing_indices.into_iter().zip(encoded_missing) {
                        resolved_embeddings[index] = Some(sequence);
                    }
                }
                image_embeddings = resolved_embeddings
                    .into_iter()
                    .map(|sequence| {
                        sequence.ok_or_else(|| {
                            "vision encoder did not return an image embedding".to_string()
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if self.settings.debug_mode
                    && let Some(first) = image_embeddings.first()
                {
                    let mut min_norm = f32::INFINITY;
                    let mut max_norm = 0.0f32;
                    let mut sum_norm = 0.0f32;
                    for token in &first.tokens {
                        let norm = token.iter().map(|v| v * v).sum::<f32>().sqrt();
                        min_norm = min_norm.min(norm);
                        max_norm = max_norm.max(norm);
                        sum_norm += norm;
                    }
                    let avg_norm = if first.tokens.is_empty() {
                        0.0
                    } else {
                        sum_norm / first.tokens.len() as f32
                    };
                    let backend = self.config.capabilities.multimodal_backend.as_str();
                    emit_debug_line(
                        event_callback,
                        format!(
                            "{backend} image embeddings: tokens={} norm(min/avg/max)={:.4}/{:.4}/{:.4}",
                            first.tokens.len(),
                            min_norm,
                            avg_norm,
                            max_norm
                        ),
                    );
                }
            }

            if !videos.is_empty() {
                return Err(format!(
                    "native media preprocessing completed ({}), but video embedding execution is not implemented yet",
                    preprocess_summary.join(", ")
                ));
            }

            let mut audio_embeddings = Vec::new();
            if !prepared_audios.is_empty() {
                let encoder = self.audio_encoder.as_ref().ok_or_else(|| {
                    "audio preprocessing succeeded, but no audio encoder is initialized".to_string()
                })?;
                let debug_mode = self.settings.debug_mode;
                audio_embeddings = prepared_audios
                    .iter()
                    .map(|audio| {
                        let window_count = audio.feature_windows.len();
                        // One encoder window covers eight seconds, so long files
                        // spend minutes here. Report progress at most ten times.
                        let progress_step = window_count.div_ceil(10).max(1);
                        let mut tokens = Vec::new();
                        for (index, window) in audio.feature_windows.iter().enumerate() {
                            let encoded = encoder.encode_feature_window(window)?;
                            tokens.extend(encoded.tokens);
                            if debug_mode
                                && (index + 1) % progress_step == 0
                                && index + 1 < window_count
                            {
                                emit_debug_line(
                                    event_callback,
                                    format!(
                                        "Encoding audio '{}': window {}/{window_count}, {} embedding token(s)",
                                        audio.path,
                                        index + 1,
                                        tokens.len()
                                    ),
                                );
                            }
                        }
                        Ok(MediaEmbeddingSequence { tokens, grid: None })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let actual_audio_token_counts = audio_embeddings
                    .iter()
                    .map(|sequence| sequence.tokens.len())
                    .collect::<Vec<_>>();
                if actual_audio_token_counts != planned_audio_token_counts {
                    return Err(format!(
                        "audio encoder token count differs from preflight plan: planned={planned_audio_token_counts:?}, actual={actual_audio_token_counts:?}"
                    ));
                }
                if self.settings.debug_mode
                    && let Some(first) = audio_embeddings.first()
                {
                    let mut min_norm = f32::INFINITY;
                    let mut max_norm = 0.0f32;
                    let mut sum_norm = 0.0f32;
                    for token in &first.tokens {
                        let norm = token.iter().map(|value| value * value).sum::<f32>().sqrt();
                        min_norm = min_norm.min(norm);
                        max_norm = max_norm.max(norm);
                        sum_norm += norm;
                    }
                    let average_norm = if first.tokens.is_empty() {
                        0.0
                    } else {
                        sum_norm / first.tokens.len() as f32
                    };
                    emit_debug_line(
                        event_callback,
                        format!(
                            "audio embeddings: tokens={} norm(min/avg/max)={min_norm:.4}/{average_norm:.4}/{max_norm:.4}",
                            first.tokens.len()
                        ),
                    );
                }
            }

            if !image_embeddings.is_empty() || !audio_embeddings.is_empty() {
                let image_token_counts = image_embeddings
                    .iter()
                    .map(|sequence| sequence.tokens.len())
                    .collect::<Vec<_>>();
                let audio_token_counts = audio_embeddings
                    .iter()
                    .map(|sequence| sequence.tokens.len())
                    .collect::<Vec<_>>();
                let decode_reserve = if audio_token_counts.is_empty() {
                    self.media_decode_reserve()
                } else {
                    self.audio_decode_reserve()
                };
                preflight_media_context(
                    &encoded_prompt,
                    &image_token_counts,
                    &audio_token_counts,
                    self.config.seq_len,
                    decode_reserve,
                )?;
                let expanded = expand_prompt_with_media_embeddings(
                    &encoded_prompt,
                    &image_embeddings,
                    &audio_embeddings,
                    self.config.input_embedding_dim,
                )?;
                prompt_tokens = expanded.token_ids;
                prefill_embeddings = expanded.embeddings;
                rope_position_plan = expanded.position_plan;
            }

            if self.settings.debug_mode {
                if let Some(mm) = &self.multimodal_weights {
                    emit_debug_line(
                        event_callback,
                        format!(
                            "Multimodal weights ready: backend={}, total_tensors={}, vision={}, projector={}, audio={}",
                            mm.backend.as_str(),
                            mm.total_tensor_count(),
                            mm.vision_tensor_names.len(),
                            mm.projector_tensor_names.len(),
                            mm.audio_tensor_names.len()
                        ),
                    );
                }
                emit_debug_line(
                    event_callback,
                    format!(
                        "Prepared multimodal prefill: tokens={} injected_embeddings={} images={} videos={} audios={}",
                        prompt_tokens.len(),
                        prefill_embeddings.len(),
                        encoded_prompt.image_spans.len(),
                        encoded_prompt.video_spans.len(),
                        encoded_prompt.audio_spans.len()
                    ),
                );
            }

            if prompt_tokens.is_empty() {
                prompt_tokens.push(self.tokenizer.bos_token);
            }
            if prompt_tokens.len() > self.config.seq_len {
                if media_requested {
                    return Err(format!(
                        "expanded media prompt has {} token(s), exceeding context limit {}; refusing to truncate through media embeddings",
                        prompt_tokens.len(),
                        self.config.seq_len
                    ));
                }
                prompt_tokens.truncate(self.config.seq_len);
            }

            let output = self.generate_from_prefill(
                prompt_tokens,
                prefill_embeddings,
                rope_position_plan,
                effective_stream_stdout,
            )?;
            self.retry_without_think_for_request(
                output,
                &effective_request,
                effective_stream_stdout,
            )
        })();
        self.settings.think_mode = original_think_mode;
        if buffered_visible_think_stdout
            && let Ok(text) = &result
            && !text.trim().is_empty()
        {
            println!("{text}");
        }
        result
    }

    fn generate_from_prefill(
        &mut self,
        prompt_tokens: Vec<i32>,
        prefill_injected_embeddings: HashMap<usize, Vec<f32>>,
        rope_position_plan: Option<RopePositionPlan>,
        stream_stdout: bool,
    ) -> Result<String, String> {
        let event_callback = self.settings.runtime_event_callback.clone();
        let hidden_retry_enabled = self.settings.think_mode == ThinkMode::Hidden;
        let retry_position_plan = hidden_retry_enabled
            .then(|| rope_position_plan.clone())
            .flatten();
        let retry_prompt_tokens = if hidden_retry_enabled {
            Some(prompt_tokens.clone())
        } else {
            None
        };
        let retry_prefill_embeddings = if hidden_retry_enabled {
            Some(prefill_injected_embeddings.clone())
        } else {
            None
        };

        let temperature = self.settings.temperature;
        let top_k = self.settings.top_k;
        let top_p = self.settings.top_p;
        let repetition_penalty = self.settings.repeat_penalty;
        let repeat_last_n = self.settings.repeat_last_n;
        let max_new_tokens = self.settings.max_tokens;
        let profiling_mode = self.settings.profiling_mode;
        let show_tokens = self.settings.show_tokens;
        let debug_mode = self.settings.debug_mode;
        let structured_output_mode = self.settings.structured_output_mode;

        let mut token = prompt_tokens[0];
        let mut next: i32;
        let mut pos = 0usize;
        let mut start = 0i64;

        let mut state = crate::engine::runtime::malloc_run_state_with_kv_cache_format(
            &self.config,
            self.kv_cache_format_override,
        )?;
        state.rope_position_plan = rope_position_plan;
        if debug_mode && !self.kv_cache_format_logged {
            emit_debug_line(
                event_callback.as_ref(),
                format!("KV cache format: {:?}", state.kv_cache_format),
            );
            self.kv_cache_format_logged = true;
        }

        // Prefill cache: when the encoded prompt starts with the cached static
        // prefix, restore its KV rows + SSM states instead of prefilling them.
        // Any mismatch (drift, config, format) falls back to a cold prefill.
        if prefill_injected_embeddings.is_empty() {
            let inject_result =
                self.prefill_cache
                    .as_ref()
                    .and_then(|pc| match pc.match_len(&prompt_tokens) {
                        Some(kp) if kp < self.config.seq_len => {
                            Some((kp, pc.inject(&self.config, &mut state)))
                        }
                        _ => None,
                    });
            match inject_result {
                Some((kp, Ok(()))) => {
                    pos = kp;
                    token = prompt_tokens[kp];
                    if !self.prefill_cache_warned {
                        eprintln!(
                            "prefill cache: restored {kp} prefix tokens (skipping their prefill)"
                        );
                        self.prefill_cache_warned = true;
                    }
                }
                Some((_, Err(e))) if !self.prefill_cache_warned => {
                    eprintln!("Warning: prefill cache disabled: {e}");
                    self.prefill_cache_warned = true;
                }
                None if self.prefill_cache.is_some() && !self.prefill_cache_warned => {
                    // Pinpoint the divergence so drift is diagnosable from logs.
                    if let Some(pc) = self.prefill_cache.as_ref() {
                        let lcp = pc
                            .tokens
                            .iter()
                            .zip(prompt_tokens.iter())
                            .take_while(|(a, b)| a == b)
                            .count();
                        let window = |toks: &[i32]| -> String {
                            let lo = lcp.saturating_sub(8);
                            let hi = (lcp + 8).min(toks.len());
                            toks[lo..hi]
                                .iter()
                                .filter_map(|&t| self.tokenizer.decode_token(t))
                                .collect::<String>()
                        };
                        eprintln!(
                            "Warning: prefill cache mismatch — diverges at token {lcp}/{} \
                             (cached={:?} prompt={:?}, prompt_len={}); cold prefill\n\
                             cached text around divergence: {:?}\n\
                             prompt text around divergence: {:?}",
                            pc.tokens.len(),
                            pc.tokens.get(lcp),
                            prompt_tokens.get(lcp),
                            prompt_tokens.len(),
                            window(&pc.tokens),
                            window(&prompt_tokens),
                        );
                    }
                    self.prefill_cache_warned = true;
                }
                _ => {}
            }
        }
        let injected_prefix = pos;
        let sampling_seed = self
            .settings
            .sampling_seed
            .unwrap_or_else(default_sampling_seed);
        if debug_mode && pos == prompt_tokens.len().saturating_sub(1) {
            emit_debug_line(
                event_callback.as_ref(),
                format!("Sampling seed: {sampling_seed}"),
            );
        }
        let mut rng = XorShiftRng::new(sampling_seed);
        let mut topk_sampler = TopKSampler::new();
        let mut warned_top_p_without_top_k = false;

        let use_repetition_penalty = repetition_penalty != 1.0 && repeat_last_n > 0;
        let mut recent_tokens = if use_repetition_penalty {
            VecDeque::with_capacity(repeat_last_n)
        } else {
            VecDeque::new()
        };
        let mut unique_recent_tokens = if use_repetition_penalty {
            HashSet::with_capacity(repeat_last_n)
        } else {
            HashSet::new()
        };
        let mut pending_newline = false;
        let mut output = String::new();
        let mut generated_tokens = Vec::new();
        let mut utf8_pending: Vec<u8> = Vec::new();
        let mut think_tail = String::new();
        let mut hidden_visible_tail = String::new();
        let mut stop_text_tail = String::new();
        let mut hidden_think_token_count = 0usize;
        let mut terminal_recovery_used = false;
        let mut early_terminal_recovery_used = false;
        let mut matched_stop_text_literal: Option<&'static str> = None;

        // Think mode state: track whether we're currently inside a <think>...</think> block.
        // The prompt already ends with "<think>\n" for Yes/Hidden modes, so generation starts
        // inside the thinking block. For No mode the prompt closes it immediately.
        let think_mode = self.settings.think_mode;
        let decode_policy = self.settings.vendor_decode_policy;
        let stop_text_literals = decode_policy.stop_text_literals;
        let suppress_visible_think_stdout = stream_stdout
            && event_callback.is_none()
            && think_mode == ThinkMode::Yes
            && decode_policy.parse_think_tags;
        let thinking_active = decode_policy.parse_think_tags && think_mode != ThinkMode::No;
        let mut is_thinking = thinking_active;
        if thinking_active && think_mode == ThinkMode::Yes && !suppress_visible_think_stdout {
            emit_output_text(event_callback.as_ref(), "<think>", stream_stdout);
        }
        let stop_tokens = decode_policy
            .stop_token_literals
            .iter()
            .filter_map(|literal| {
                self.tokenizer
                    .find_special_token(literal)
                    .map(|id| (id, *literal))
            })
            .collect::<Vec<_>>();
        let stop_token_ids = stop_tokens
            .iter()
            .map(|(id, _)| *id)
            .collect::<HashSet<_>>();
        let structured_output_schema = structured_output_mode.schema();
        let structured_output_blocked_token_ids = collect_structured_output_blocked_token_ids(
            &self.tokenizer,
            &stop_tokens,
            structured_output_schema,
        );
        let mut structured_output_prefix_tokens =
            build_structured_output_prefix_tokens(&mut self.tokenizer, structured_output_schema);
        let mut structured_output_completed = false;

        let total_limit = prompt_tokens
            .len()
            .saturating_add(max_new_tokens)
            .min(self.config.seq_len);
        emit_progress_update(
            event_callback.as_ref(),
            RuntimeProgress {
                phase: RuntimePhase::Prefill,
                prefill_tokens: prompt_tokens.len(),
                decode_tokens: 0,
                hidden_thinking: false,
                hidden_think_tokens: 0,
                tokens_per_second: None,
                context_used: prompt_tokens.len(),
                context_limit: self.config.seq_len,
            },
        );
        let think_caps = if decode_policy.parse_think_tags {
            let new_budget = total_limit.saturating_sub(prompt_tokens.len());
            Some(compute_think_caps(
                decode_policy.hidden_think_token_cap_base,
                self.config.n_layers,
                self.config.seq_len,
                new_budget,
                self.settings.hidden_think_token_cap,
            ))
        } else {
            None
        };
        let hidden_mode_caps = if think_mode == ThinkMode::Hidden {
            think_caps
        } else {
            None
        };
        let visible_yes_think_cap = if think_mode == ThinkMode::Yes {
            think_caps.map(|(_, _)| {
                let new_budget = total_limit.saturating_sub(prompt_tokens.len());
                if new_budget == 0 {
                    return 0usize;
                }
                let visible_vendor_base = decode_policy.visible_think_token_cap_base.max(96usize);
                let layer_mult = if self.config.n_layers >= 64 {
                    3usize
                } else if self.config.n_layers >= 32 {
                    2usize
                } else {
                    1usize
                };
                let ctx_mult = if self.config.seq_len >= 262_144 {
                    2usize
                } else {
                    1usize
                };
                let mut visible_cap = visible_vendor_base
                    .saturating_mul(layer_mult)
                    .saturating_mul(ctx_mult);
                // Reserve some decode budget for the answer after </think>. Without this,
                // short generations can spend the entire budget inside thinking and then
                // require a second pass to produce the visible answer.
                let answer_reserve = if new_budget >= 96 {
                    48usize
                } else if new_budget >= 48 {
                    24usize
                } else {
                    (new_budget / 3).max(8usize)
                };
                let visible_budget_cap = new_budget.saturating_sub(answer_reserve).max(8usize);
                visible_cap = visible_cap.clamp(8usize, visible_budget_cap.max(8usize));
                visible_cap.min(visible_budget_cap)
            })
        } else {
            None
        };
        let mut visible_yes_think_token_count = 0usize;
        let mut last_progress_emit_ms = 0i64;
        if let Some((think_cap, total_cap)) = hidden_mode_caps
            && debug_mode
        {
            emit_debug_line(
                event_callback.as_ref(),
                format!(
                    "Hidden-think policy: think_cap={} total_cap={} (n_layers={} seq_len={} budget={})",
                    think_cap,
                    total_cap,
                    self.config.n_layers,
                    self.config.seq_len,
                    total_limit.saturating_sub(prompt_tokens.len())
                ),
            );
        }

        // Batched prefill: process all but the last prompt token in chunks so
        // each weight tensor streams from memory once per chunk instead of
        // once per token. Exact batched matmuls mirror the sequential kernels;
        // the default fast K-quant kernel can differ in floating-point rounding.
        // Both modes share the rope/KV/attention helpers. The last
        // prompt token stays on the sequential path so the logits and
        // sampling flow are untouched.
        //
        // Media prompts take this path too: injected audio/image embeddings
        // ride along as `PrefillInput::Embedding`. Deepstack models are the
        // exception — their embeddings carry a per-layer tail that only the
        // sequential path replays, so they keep the per-token loop.
        let embeddings_fit_batch_prefill = prefill_injected_embeddings.is_empty()
            || self.config.input_embedding_dim == self.config.dim;
        if crate::engine::switches::use_batch_prefill()
            && crate::engine::runtime::batch_prefill_supported(&self.config)
            && embeddings_fit_batch_prefill
            && prompt_tokens.len() > 8
        {
            let prefill_end = prompt_tokens.len() - 1;
            let chunk = crate::engine::switches::batch_prefill_chunk();
            let mut scratch = crate::engine::runtime::PrefillScratch::new();
            let mut base = pos;
            while base < prefill_end {
                let take = chunk.min(prefill_end - base);
                let toks: Vec<crate::engine::runtime::PrefillInput<'_>> = (base..base + take)
                    .map(|position| {
                        prefill_injected_embeddings.get(&position).map_or_else(
                            || {
                                crate::engine::runtime::PrefillInput::Token(
                                    prompt_tokens[position] as usize,
                                )
                            },
                            |embedding| {
                                crate::engine::runtime::PrefillInput::Embedding(
                                    embedding.as_slice(),
                                )
                            },
                        )
                    })
                    .collect();
                crate::engine::runtime::transformer_prefill_batch(
                    &toks,
                    base,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                    &mut scratch,
                )?;
                base += take;
                emit_progress_update(
                    event_callback.as_ref(),
                    RuntimeProgress {
                        phase: RuntimePhase::Prefill,
                        prefill_tokens: base,
                        decode_tokens: 0,
                        hidden_thinking: false,
                        hidden_think_tokens: 0,
                        tokens_per_second: None,
                        context_used: base,
                        context_limit: self.config.seq_len,
                    },
                );
            }
            pos = prefill_end;
            token = prompt_tokens[prefill_end];
        }

        while pos < total_limit {
            if token < 0 || token as usize >= self.config.vocab_size {
                return Err(format!("token id out of bounds: {token}"));
            }

            let prof_t0 = prof_start();
            let needs_logits = pos >= prompt_tokens.len().saturating_sub(1);
            if let Some(embedding) = prefill_injected_embeddings.get(&pos) {
                if needs_logits {
                    crate::engine::runtime::transformer_with_embedding(
                        embedding,
                        pos,
                        &self.config,
                        &mut state,
                        &self.weights,
                        self.gguf.mapped.as_slice(),
                    )?;
                } else {
                    crate::engine::runtime::transformer_with_embedding_without_logits(
                        embedding,
                        pos,
                        &self.config,
                        &mut state,
                        &self.weights,
                        self.gguf.mapped.as_slice(),
                    )?;
                }
            } else if needs_logits {
                crate::engine::runtime::transformer(
                    token as usize,
                    pos,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                )?;
            } else {
                crate::engine::runtime::transformer_without_logits(
                    token as usize,
                    pos,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                )?;
            }
            prof_end(&PROF_TRANSFORMER_NS, prof_t0);
            if profiling_mode {
                record_forward_pass();
            }

            if debug_mode && pos >= prompt_tokens.len().saturating_sub(1) {
                let mut ranked: Vec<(usize, f32)> = state.logits[..self.config.vocab_size]
                    .iter()
                    .copied()
                    .enumerate()
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let mut line = format!("[DEBUG pos={pos}] Top 5: ");
                for (id, v) in ranked.iter().take(5) {
                    let decoded = self
                        .tokenizer
                        .decode_token(*id as i32)
                        .unwrap_or_else(|| "?".to_string())
                        .replace('\n', "\\n")
                        .replace('\r', "\\r");
                    line.push_str(&format!("{id}({v:.2},\"{decoded}\") "));
                }
                // Always show where every stop token ranks, so we can see if the model
                // ever wants to stop but is prevented by sampling or a missing check.
                let stop_info: Vec<String> = stop_tokens
                    .iter()
                    .chain(std::iter::once(&(self.tokenizer.eos_token, "<eos>")))
                    .map(|(sid, slit)| {
                        let logit = if (*sid as usize) < self.config.vocab_size {
                            state.logits[*sid as usize]
                        } else {
                            f32::NEG_INFINITY
                        };
                        let rank = ranked
                            .iter()
                            .position(|(id, _)| *id == *sid as usize)
                            .map(|r| r + 1)
                            .unwrap_or(0);
                        format!("{slit}(id={sid},rank={rank},logit={logit:.2})")
                    })
                    .collect();
                line.push_str(&format!(" | stop: {}", stop_info.join(" ")));
                emit_debug_line(event_callback.as_ref(), line);
            }

            if pos < prompt_tokens.len().saturating_sub(1) {
                next = prompt_tokens[pos + 1];
            } else {
                if structured_output_mode != StructuredOutputMode::None
                    && output.trim().is_empty()
                    && !structured_output_prefix_tokens.is_empty()
                {
                    next = structured_output_prefix_tokens.remove(0);
                } else {
                    if use_repetition_penalty {
                        unique_recent_tokens.clear();
                        for &tok in &recent_tokens {
                            unique_recent_tokens.insert(tok);
                        }
                        for tok in unique_recent_tokens.iter().copied() {
                            if tok >= 0 && (tok as usize) < self.config.vocab_size {
                                let idx = tok as usize;
                                if state.logits[idx] > 0.0 {
                                    state.logits[idx] /= repetition_penalty;
                                } else {
                                    state.logits[idx] *= repetition_penalty;
                                }
                            }
                        }
                    }

                    if let Some(schema) = structured_output_schema {
                        mask_invalid_structured_output_logits(
                            &mut state.logits[..self.config.vocab_size],
                            &structured_output_blocked_token_ids,
                            &self.tokenizer,
                            &output,
                            schema,
                        );
                    }

                    if temperature == 0.0 {
                        next = argmax(&state.logits[..self.config.vocab_size]) as i32;
                    } else if top_k == 1 {
                        // temperature scaling is monotonic for temperature>0, so k=1 equals argmax.
                        next = argmax(&state.logits[..self.config.vocab_size]) as i32;
                    } else if top_k > 0 {
                        next = topk_sampler.sample_top_k_top_p(
                            &state.logits[..self.config.vocab_size],
                            temperature,
                            top_k,
                            top_p,
                            &mut rng,
                        ) as i32;
                    } else {
                        if top_p < 1.0 && debug_mode && !warned_top_p_without_top_k {
                            emit_debug_line(
                                event_callback.as_ref(),
                                "Note: -top_p is ignored unless -top_k > 0",
                            );
                            warned_top_p_without_top_k = true;
                        }
                        for q in 0..self.config.vocab_size {
                            state.logits[q] /= temperature;
                        }
                        softmax(
                            &mut state.logits[..self.config.vocab_size],
                            self.config.vocab_size,
                        );
                        next = sample(&state.logits[..self.config.vocab_size], &mut rng) as i32;
                    }

                    if use_repetition_penalty {
                        if recent_tokens.len() == repeat_last_n {
                            recent_tokens.pop_front();
                        }
                        recent_tokens.push_back(next);
                    }
                }
            }

            let is_vendor_stop_token = stop_token_ids.contains(&next);
            let is_endoftext_stop = next == self.tokenizer.eos_token
                || stop_tokens
                    .iter()
                    .any(|(id, literal)| *id == next && *literal == "<|endoftext|>");
            let should_recover_hidden_think_terminal = think_mode == ThinkMode::Hidden
                && output.trim().is_empty()
                && !early_terminal_recovery_used
                && pos >= prompt_tokens.len().saturating_sub(1)
                && hidden_mode_caps
                    .map(|(_, total_cap)| generated_tokens.len() < total_cap)
                    .unwrap_or(false)
                && (is_vendor_stop_token
                    || next == self.tokenizer.eos_token
                    || next == self.tokenizer.eot_token);
            let should_recover_early_terminal = (decode_policy.recover_early_endoftext_once
                && !early_terminal_recovery_used
                && pos >= prompt_tokens.len().saturating_sub(1)
                && generated_tokens.len() < decode_policy.early_endoftext_recover_max_tokens
                && is_endoftext_stop)
                || should_recover_hidden_think_terminal;
            if should_recover_early_terminal {
                if debug_mode {
                    let mut ranked: Vec<(usize, f32)> = state.logits[..self.config.vocab_size]
                        .iter()
                        .copied()
                        .enumerate()
                        .collect();
                    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                    emit_debug_line(
                        event_callback.as_ref(),
                        format!(
                            "Note: {} token id={} at generated_tokens={}; top alternatives:",
                            if should_recover_hidden_think_terminal {
                                "hidden-think terminal"
                            } else {
                                "early terminal"
                            },
                            next,
                            generated_tokens.len(),
                        ),
                    );
                    let mut shown = 0usize;
                    for (idx, logit) in ranked {
                        let candidate = idx as i32;
                        if candidate == self.tokenizer.eos_token
                            || candidate == self.tokenizer.eot_token
                            || stop_token_ids.contains(&candidate)
                            || structured_output_blocked_token_ids.contains(&candidate)
                        {
                            continue;
                        }
                        let decoded = self
                            .tokenizer
                            .decode_token(candidate)
                            .unwrap_or_else(|| "?".to_string())
                            .replace('\n', "\\n")
                            .replace('\r', "\\r");
                        emit_debug_line(
                            event_callback.as_ref(),
                            format!(
                                "  cand id={} logit={:.3} tok=\"{}\"",
                                candidate, logit, decoded
                            ),
                        );
                        shown += 1;
                        if shown >= 8 {
                            break;
                        }
                    }
                }
                let mut alt_token: Option<i32> = None;
                let mut alt_logit = f32::NEG_INFINITY;
                let mut fallback_token: Option<i32> = None;
                let mut fallback_logit = f32::NEG_INFINITY;
                for (idx, &logit) in state.logits[..self.config.vocab_size].iter().enumerate() {
                    let candidate = idx as i32;
                    if candidate == next
                        || candidate == self.tokenizer.eos_token
                        || candidate == self.tokenizer.eot_token
                        || stop_token_ids.contains(&candidate)
                        || structured_output_blocked_token_ids.contains(&candidate)
                    {
                        continue;
                    }
                    if logit > fallback_logit {
                        fallback_logit = logit;
                        fallback_token = Some(candidate);
                    }
                    let decoded = self.tokenizer.decode_token(candidate).unwrap_or_default();
                    let trimmed = decoded.trim();
                    let looks_like_short_language_marker = !trimmed.is_empty()
                        && trimmed.len() <= 16
                        && trimmed.chars().all(|c| c.is_ascii_alphabetic());
                    if looks_like_short_language_marker {
                        continue;
                    }
                    if logit > alt_logit {
                        alt_logit = logit;
                        alt_token = Some(candidate);
                    }
                }
                if let Some(candidate) = if should_recover_hidden_think_terminal {
                    fallback_token.or(alt_token)
                } else {
                    alt_token.or(fallback_token)
                } {
                    early_terminal_recovery_used = true;
                    if debug_mode {
                        emit_debug_line(
                            event_callback.as_ref(),
                            format!(
                                "Note: recovered once from early terminal token id={} at generated_tokens={}; using alternative token id={}",
                                next,
                                generated_tokens.len(),
                                candidate
                            ),
                        );
                    }
                    next = candidate;
                }
            }

            if pos >= prompt_tokens.len().saturating_sub(1)
                && next != self.tokenizer.eot_token
                && next != self.tokenizer.eos_token
                && !is_vendor_stop_token
                && let Some(bytes) = self.tokenizer.decode_token_bytes(next)
            {
                let decoded = decode_utf8_streaming(&mut utf8_pending, &bytes);
                process_decoded_with_think(
                    &decoded,
                    decode_policy.parse_think_tags,
                    think_mode,
                    &mut is_thinking,
                    &mut think_tail,
                    &mut hidden_visible_tail,
                    &mut output,
                    &mut pending_newline,
                    stream_stdout,
                    suppress_visible_think_stdout,
                    event_callback.as_ref(),
                    stop_text_literals,
                    &mut stop_text_tail,
                    &mut matched_stop_text_literal,
                );
                if let Some(structured_output) =
                    extract_first_complete_structured_output(&output, structured_output_schema)
                {
                    output = structured_output;
                    pending_newline = false;
                    think_tail.clear();
                    hidden_visible_tail.clear();
                    utf8_pending.clear();
                    structured_output_completed = true;
                }
            }

            token = next;
            pos += 1;

            if start == 0 {
                start = time_in_ms();
            }

            // Post-increment `pos`, as above: decode progress starts once the
            // prompt is fully consumed.
            if pos >= prompt_tokens.len() {
                let now_ms = time_in_ms();
                if now_ms.saturating_sub(last_progress_emit_ms) >= 200 {
                    let decode_tokens = generated_tokens.len();
                    let tok_s = if decode_tokens > 0 {
                        let elapsed_ms = (now_ms - start).max(1) as f64;
                        Some(decode_tokens as f64 / elapsed_ms * 1000.0)
                    } else {
                        None
                    };
                    emit_progress_update(
                        event_callback.as_ref(),
                        RuntimeProgress {
                            phase: RuntimePhase::Decode,
                            prefill_tokens: prompt_tokens.len(),
                            decode_tokens,
                            hidden_thinking: think_mode == ThinkMode::Hidden && is_thinking,
                            hidden_think_tokens: hidden_think_token_count,
                            tokens_per_second: tok_s,
                            context_used: prompt_tokens.len().saturating_add(decode_tokens),
                            context_limit: self.config.seq_len,
                        },
                    );
                    last_progress_emit_ms = now_ms;
                }
            }

            // `pos` and `token` were advanced just above, so `token` is the
            // token *at* `pos`. It is generated only once `pos` has passed the
            // whole prompt — testing `len() - 1` here counted the final prompt
            // token as generated, which shifted every periodic loop-guard check
            // (`% 4`, `% 16`) and the decode-budget accounting by one. Batched
            // prefill enters this loop already past that position, so the two
            // prefill paths disagreed on `generated_tokens.len()`.
            if pos >= prompt_tokens.len() {
                generated_tokens.push(token);
                if let Some(literal) = matched_stop_text_literal {
                    if debug_mode {
                        emit_debug_line(
                            event_callback.as_ref(),
                            format!("Stopping on decoded text stop literal '{literal}'"),
                        );
                    }
                    break;
                }
                if let Some((hidden_think_cap, hidden_total_cap)) = hidden_mode_caps {
                    if is_thinking {
                        hidden_think_token_count = hidden_think_token_count.saturating_add(1);
                        if hidden_think_token_count == hidden_think_cap && debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Note: hidden thinking block exceeded {} tokens without </think>, continuing to suppress until total cap",
                                    hidden_think_cap
                                ),
                            );
                        }
                    }
                    if generated_tokens.len() >= hidden_total_cap {
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Stopping hidden mode after {} generated tokens to avoid unbounded run",
                                    hidden_total_cap
                                ),
                            );
                        }
                        break;
                    }
                }
                if let Some(visible_think_cap) = visible_yes_think_cap
                    && is_thinking
                {
                    visible_yes_think_token_count = visible_yes_think_token_count.saturating_add(1);
                    if visible_yes_think_token_count >= visible_think_cap {
                        is_thinking = false;
                        think_tail.clear();
                        append_visible_text_with_stop_literals(
                            THINK_CLOSE_TAG,
                            &mut output,
                            &mut pending_newline,
                            stream_stdout,
                            event_callback.as_ref(),
                            stop_text_literals,
                            &mut stop_text_tail,
                            &mut matched_stop_text_literal,
                        );
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Note: forced close of visible thinking block after {} tokens without </think>",
                                    visible_think_cap
                                ),
                            );
                        }
                    }
                }
                if structured_output_completed {
                    if debug_mode {
                        emit_debug_line(
                            event_callback.as_ref(),
                            "Stopping after first complete structured output object",
                        );
                    }
                    break;
                }
                if token == self.tokenizer.eos_token || token == self.tokenizer.eot_token {
                    if decode_policy.parse_think_tags
                        && think_mode != ThinkMode::No
                        && is_thinking
                        && !terminal_recovery_used
                    {
                        terminal_recovery_used = true;
                        is_thinking = false;
                        think_tail.clear();
                        if think_mode == ThinkMode::Yes {
                            append_visible_text_with_stop_literals(
                                THINK_CLOSE_TAG,
                                &mut output,
                                &mut pending_newline,
                                stream_stdout,
                                event_callback.as_ref(),
                                stop_text_literals,
                                &mut stop_text_tail,
                                &mut matched_stop_text_literal,
                            );
                        } else {
                            pending_newline = false;
                        }
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Note: forced close of thinking block on terminal token id={token}; retrying decode once"
                                ),
                            );
                        }
                        continue;
                    }
                    if debug_mode {
                        emit_debug_line(
                            event_callback.as_ref(),
                            format!("Stopping on terminal token id={token}"),
                        );
                    }
                    break;
                }
                if let Some((_, literal)) = stop_tokens.iter().find(|(id, _)| *id == token) {
                    if decode_policy.parse_think_tags && think_mode != ThinkMode::No && is_thinking
                    {
                        is_thinking = false;
                        think_tail.clear();
                        if think_mode == ThinkMode::Yes {
                            append_visible_text_with_stop_literals(
                                THINK_CLOSE_TAG,
                                &mut output,
                                &mut pending_newline,
                                stream_stdout,
                                event_callback.as_ref(),
                                stop_text_literals,
                                &mut stop_text_tail,
                                &mut matched_stop_text_literal,
                            );
                        } else {
                            pending_newline = false;
                        }
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Note: forced close of thinking block on stop token '{}' (id={token})",
                                    literal
                                ),
                            );
                        }
                        continue;
                    } else {
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!("Stopping on vendor stop token '{}' (id={token})", literal),
                            );
                        }
                        break;
                    }
                }
                // Existing gated checks (newline-based or exact suffix).
                if self.settings.vendor_decode_policy.deterministic_loop_guard
                    && generated_tokens.len()
                        >= self
                            .settings
                            .vendor_decode_policy
                            .deterministic_loop_guard_min_generated_tokens
                    && generated_tokens.len() % 16 == 0
                {
                    if let Some(len) = repeated_text_suffix_bytes(&output) {
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Stopping due to repeated output suffix block (len={len} bytes)"
                                ),
                            );
                        }
                        break;
                    }
                    if let Some((line, repeats)) = repeated_long_line(&output) {
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!(
                                    "Stopping due to repeated output line (repeats={repeats}): {line}"
                                ),
                            );
                        }
                        break;
                    }
                }
                // Policy-gated checks that ignore temperature and think mode.
                // Fires every 4 tokens after the first 8 generated tokens.
                if decode_policy.unconditional_loop_guard
                    && generated_tokens.len() >= 8
                    && generated_tokens.len() % 4 == 0
                {
                    // Text-level: inline phrase repetition in visible output.
                    if !output.is_empty()
                        && let Some(phrase) = repeated_inline_phrase(&output)
                    {
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!("Stopping due to repeated inline phrase: \"{phrase}\""),
                            );
                        }
                        break;
                    }
                    // Token-level: tight cycle in all generated tokens (catches hidden think loops).
                    if let Some(period) = repeated_cycle_period(&generated_tokens) {
                        if debug_mode {
                            emit_debug_line(
                                event_callback.as_ref(),
                                format!("Stopping due to repeated token cycle (window={period})"),
                            );
                        }
                        break;
                    }
                }
            }
        }

        // Decoding that consumes its whole budget ended at the limit rather than
        // at a stop token, so the visible output is cut off.
        self.last_decode_used_full_token_budget =
            generated_tokens.len() >= total_limit.saturating_sub(prompt_tokens.len());

        let pending_decoded = flush_utf8_pending_lossy(&mut utf8_pending);
        process_decoded_with_think(
            &pending_decoded,
            decode_policy.parse_think_tags,
            think_mode,
            &mut is_thinking,
            &mut think_tail,
            &mut hidden_visible_tail,
            &mut output,
            &mut pending_newline,
            stream_stdout,
            suppress_visible_think_stdout,
            event_callback.as_ref(),
            stop_text_literals,
            &mut stop_text_tail,
            &mut matched_stop_text_literal,
        );
        if let Some(structured_output) =
            extract_first_complete_structured_output(&output, structured_output_schema)
        {
            output = structured_output;
        }
        if !think_tail.is_empty() {
            if think_mode == ThinkMode::Yes {
                let trailing = finalize_visible_think_tail(&mut think_tail, is_thinking);
                append_visible_text_with_stop_literals(
                    &trailing,
                    &mut output,
                    &mut pending_newline,
                    stream_stdout && !suppress_visible_think_stdout,
                    event_callback.as_ref(),
                    stop_text_literals,
                    &mut stop_text_tail,
                    &mut matched_stop_text_literal,
                );
            }
            think_tail.clear();
        }
        if think_mode == ThinkMode::Hidden || think_mode == ThinkMode::No {
            let trailing = hidden_finalize_tail(&mut hidden_visible_tail);
            let trailing = trim_leading_line_breaks_for_first_visible(&trailing, &output);
            append_visible_text_with_stop_literals(
                trailing,
                &mut output,
                &mut pending_newline,
                stream_stdout,
                event_callback.as_ref(),
                stop_text_literals,
                &mut stop_text_tail,
                &mut matched_stop_text_literal,
            );
        }
        if decode_policy.parse_think_tags && think_mode == ThinkMode::Yes && is_thinking {
            append_visible_text_with_stop_literals(
                THINK_CLOSE_TAG,
                &mut output,
                &mut pending_newline,
                stream_stdout && !suppress_visible_think_stdout,
                event_callback.as_ref(),
                stop_text_literals,
                &mut stop_text_tail,
                &mut matched_stop_text_literal,
            );
            if debug_mode {
                emit_debug_line(
                    event_callback.as_ref(),
                    "Note: auto-closed missing </think> at end of generation",
                );
            }
        }
        flush_visible_text_stop_tail(
            &mut output,
            &mut pending_newline,
            stream_stdout,
            event_callback.as_ref(),
            &mut stop_text_tail,
            matched_stop_text_literal,
        );
        if decode_policy.parse_think_tags
            && think_mode == ThinkMode::Yes
            && !decode_policy.retry_without_think_when_no_post_think_text
            && !has_post_think_response_text(&output)
            && let Some(promoted) = promote_think_only_content(&output)
        {
            if event_callback.is_some() || stream_stdout {
                emit_output_text(
                    event_callback.as_ref(),
                    format!("\n{promoted}"),
                    stream_stdout,
                );
            }
            output = promoted;
            if debug_mode {
                emit_debug_line(
                    event_callback.as_ref(),
                    "Note: promoted think-only content to final response text",
                );
            }
        }
        if let Some(structured_output) =
            extract_first_complete_structured_output(&output, structured_output_schema)
        {
            output = structured_output;
        }

        let end = time_in_ms();
        emit_progress_update(
            event_callback.as_ref(),
            RuntimeProgress {
                phase: RuntimePhase::Ready,
                prefill_tokens: prompt_tokens.len().saturating_sub(injected_prefix),
                decode_tokens: generated_tokens.len(),
                hidden_thinking: false,
                hidden_think_tokens: hidden_think_token_count,
                tokens_per_second: if pos > 1 {
                    let elapsed_ms = (end - start).max(1) as f64;
                    Some((pos - 1) as f64 / elapsed_ms * 1000.0)
                } else {
                    None
                },
                context_used: prompt_tokens.len().saturating_add(generated_tokens.len()),
                context_limit: self.config.seq_len,
            },
        );
        let will_retry_without_think =
            should_retry_without_think_for_output(think_mode, decode_policy, &output);
        if (debug_mode || show_tokens) && pos > 1 && !will_retry_without_think {
            if stream_stdout && !output.is_empty() && !output.ends_with('\n') {
                println!();
            }
            let elapsed_ms = (end - start).max(1) as f64;
            emit_cli_info_line(
                event_callback.as_ref(),
                format!(
                    "achieved tok/s: {:.3}",
                    (pos - 1) as f64 / elapsed_ms * 1000.0
                ),
                stream_stdout,
                show_tokens && !debug_mode,
            );
        } else if stream_stdout && !output.is_empty() {
            println!();
        }

        if hidden_retry_enabled && output.trim().is_empty() {
            const HIDDEN_EMPTY_OUTPUT_RETRY_MAX_TOKENS: usize = 512;
            if debug_mode {
                emit_debug_line(
                    event_callback.as_ref(),
                    format!(
                        "Note: hidden think mode produced no visible output; retrying with think=no and max_tokens={}",
                        self.settings
                            .max_tokens
                            .min(HIDDEN_EMPTY_OUTPUT_RETRY_MAX_TOKENS)
                    ),
                );
            }
            if let (Some(mut retry_prompt_tokens), Some(retry_prefill_embeddings)) =
                (retry_prompt_tokens, retry_prefill_embeddings)
            {
                if decode_policy.parse_think_tags {
                    // The encoded prompt still ends with the template's
                    // "<think>\n" seed. Close the block so the think=no retry
                    // decodes in the answer region; otherwise the model keeps
                    // reasoning and No-mode parsing streams that reasoning as
                    // the visible response.
                    append_think_close_suffix(&mut self.tokenizer, &mut retry_prompt_tokens);
                }
                let original_think_mode = self.settings.think_mode;
                let original_max_tokens = self.settings.max_tokens;
                self.settings.think_mode = ThinkMode::No;
                self.settings.max_tokens = self
                    .settings
                    .max_tokens
                    .min(HIDDEN_EMPTY_OUTPUT_RETRY_MAX_TOKENS);
                let retry_result = self.generate_from_prefill(
                    retry_prompt_tokens,
                    retry_prefill_embeddings,
                    retry_position_plan,
                    stream_stdout,
                );
                self.settings.think_mode = original_think_mode;
                self.settings.max_tokens = original_max_tokens;
                return retry_result;
            }
        }

        Ok(sanitize_final_response_text(&output))
    }

    fn retry_without_think_for_request(
        &mut self,
        output: String,
        request: &GenerationRequest,
        stream_stdout: bool,
    ) -> Result<String, String> {
        if !should_retry_without_think_for_output(
            self.settings.think_mode,
            self.settings.vendor_decode_policy,
            &output,
        ) {
            return Ok(sanitize_final_response_text(&output));
        }

        if self.settings.debug_mode {
            emit_debug_line(
                self.settings.runtime_event_callback.as_ref(),
                "Note: no post-think answer detected; retrying once with think=no",
            );
        }
        if self.settings.runtime_event_callback.is_none()
            && stream_stdout
            && !output.ends_with('\n')
        {
            println!();
        }

        let mut retry_request = request.clone();
        retry_request.system_prompt =
            build_direct_answer_retry_system_prompt(&retry_request.system_prompt);
        let promoted_original = promote_think_only_content(&output);

        let original_think_mode = self.settings.think_mode;
        let original_temperature = self.settings.temperature;
        let original_top_k = self.settings.top_k;
        let original_top_p = self.settings.top_p;

        self.settings.think_mode = ThinkMode::No;
        self.settings.temperature = 0.0;
        self.settings.top_k = 1;
        self.settings.top_p = 1.0;

        let retry_result = self.generate_request(&retry_request, stream_stdout);

        self.settings.think_mode = original_think_mode;
        self.settings.temperature = original_temperature;
        self.settings.top_k = original_top_k;
        self.settings.top_p = original_top_p;

        match retry_result {
            Ok(retry_output) => {
                let retry_output = retry_output.trim();
                if retry_output.is_empty() || !has_meaningful_retry_text(retry_output) {
                    return Ok(sanitize_final_response_text(
                        &promoted_original.unwrap_or(output),
                    ));
                }
                let mut combined = sanitize_final_response_text(&output);
                if !combined.is_empty() {
                    combined.push_str("\n\n");
                }
                combined.push_str(&sanitize_final_response_text(retry_output));
                Ok(sanitize_final_response_text(&combined))
            }
            Err(err) => {
                if self.settings.debug_mode {
                    emit_debug_line(
                        self.settings.runtime_event_callback.as_ref(),
                        format!("Note: think=no retry failed: {err}"),
                    );
                }
                Ok(sanitize_final_response_text(
                    &promoted_original.unwrap_or(output),
                ))
            }
        }
    }

    pub(crate) fn generate_text_with_images(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        images: &[String],
        stream_stdout: bool,
    ) -> Result<String, String> {
        if !images.is_empty() {
            let mut parts = Vec::with_capacity(1 + images.len());
            // Keep image placeholders ahead of text so multimodal templates can bind media spans first.
            for image in images {
                parts.push(ContentPart::Image(crate::engine::types::MediaRef {
                    path: image.clone(),
                }));
            }
            let mut effective_prompt = prompt.to_string();
            if !self
                .settings
                .vendor_multimodal_policy
                .image_prompt_suffix
                .is_empty()
            {
                effective_prompt
                    .push_str(self.settings.vendor_multimodal_policy.image_prompt_suffix);
            }
            if !prompt.trim().is_empty() {
                parts.push(ContentPart::Text(effective_prompt));
            }
            let req = GenerationRequest {
                system_prompt: system_prompt.to_string(),
                parts,
                include_empty_system_prompt: false,
                assistant_prefill: None,
            };
            return self.generate_request(&req, stream_stdout);
        }

        let debug_mode = self.settings.debug_mode;
        let mut prompt_tokens: Vec<i32> = crate::vendors::encode_chat_prompt(
            &mut self.tokenizer,
            &self.config,
            prompt,
            system_prompt,
            images.len(),
            self.settings.think_mode,
        );

        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token);
        }
        if prompt_tokens.len() > self.config.seq_len {
            prompt_tokens.truncate(self.config.seq_len);
        }
        if debug_mode {
            emit_debug_line(
                self.settings.runtime_event_callback.as_ref(),
                format!("Prompt tokens: {}", prompt_tokens.len()),
            );
            let preview = prompt_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            emit_debug_line(
                self.settings.runtime_event_callback.as_ref(),
                format!("Prompt token ids: [{preview}]"),
            );
        }

        let prefill_injected_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
        let output = self.generate_from_prefill(
            prompt_tokens,
            prefill_injected_embeddings,
            None,
            stream_stdout,
        )?;
        let request = GenerationRequest {
            system_prompt: system_prompt.to_string(),
            parts: vec![ContentPart::Text(prompt.to_string())],
            include_empty_system_prompt: false,
            assistant_prefill: None,
        };
        self.retry_without_think_for_request(output, &request, stream_stdout)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ModelRuntime, PrefixMatch, append_visible_text_with_stop_literals, compute_think_caps,
        extract_first_complete_json_object, finalize_visible_think_tail,
        find_first_complete_json_object_span, flush_visible_text_stop_tail,
        has_meaningful_retry_text, is_agent_json_safe_text, match_agent_response_prefix,
        promote_think_only_content, repeated_inline_phrase, sanitize_final_response_text,
        should_buffer_visible_think_stdout,
    };
    use crate::app::CWD_TEST_LOCK;
    use crate::engine::types::ThinkMode;
    use crate::vendors::VendorDecodePolicy;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn repeated_inline_phrase_lookback_snaps_to_char_boundary() {
        // 600 two-byte 'ä' + one ascii byte = 1201 bytes; the 1024-byte
        // lookback start (byte 177) falls inside an 'ä'.
        let mid_char_cut = "ä".repeat(600) + "x";
        let _ = repeated_inline_phrase(&mid_char_cut);

        // Detection still works on multi-byte text past the lookback size.
        let padding = "ä".repeat(600);
        let repeated = format!("{padding}{}", "Grüße aus Köln! ".repeat(4));
        assert!(repeated_inline_phrase(&repeated).is_some());
    }

    #[test]
    #[ignore = "requires local Qwen3.5-2B-Q4_K_M.gguf, its F16 mmproj, and regression/IMG_0138.jpg; run in release mode"]
    fn lazy_image_requests_match_warm_and_preencoded_requests() {
        use crate::app::events::RuntimeEvent;
        use crate::engine::vision::ImageNormalization;
        use std::path::Path;
        use std::sync::{Arc, Mutex};

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut runtime =
            ModelRuntime::load_from_file(&root.join("Qwen3.5-2B-Q4_K_M.gguf"), false).unwrap();
        assert!(
            runtime.vision_encoder.is_none(),
            "fixture must start with lazy sidecar loading"
        );
        runtime.set_context_size(512);
        runtime.set_think_mode_no();
        runtime.set_sampling_seed(Some(1));
        runtime.set_debug_mode(true);
        runtime.settings.max_tokens = 128;
        runtime
            .settings
            .vendor_multimodal_policy
            .detail_crop
            .enabled = false;
        let messages = Arc::new(Mutex::new(Vec::new()));
        let captured = messages.clone();
        runtime.set_runtime_event_callback(Some(Arc::new(move |event| {
            if let RuntimeEvent::Log(log) = event {
                captured.lock().unwrap().push(log.message);
            }
        })));
        let image_path = root
            .join("regression/IMG_0138.jpg")
            .to_str()
            .unwrap()
            .to_string();
        let mut outputs = Vec::new();
        for run in 0..3 {
            if run == 2 {
                let results = runtime.preencode_image_batch(std::slice::from_ref(&image_path));
                assert_eq!(results, [Ok(())]);
                assert_eq!(
                    runtime.preencoded_image_embeddings[&image_path].grid,
                    Some([1, 12, 16])
                );
            }
            messages.lock().unwrap().clear();
            // Use the same request construction as EmbeddedRuntime::generate_with_image,
            // including image-first ordering and the vendor's prompt suffix.
            outputs.push(
                runtime
                    .generate_text_with_images(
                        "Answer with one word: does this image show water?",
                        "Be concise.",
                        std::slice::from_ref(&image_path),
                        false,
                    )
                    .unwrap(),
            );
            let logs = messages.lock().unwrap();
            assert!(
                logs.iter()
                    .any(|line| line.contains("width=512, height=384, elements=589824")),
                "run {run}: {logs:?}"
            );
            assert!(
                logs.iter()
                    .any(|line| line.contains("injected_embeddings=192 images=1")),
                "run {run}: {logs:?}"
            );
            assert_eq!(
                runtime.image_preprocess_profile().normalization,
                ImageNormalization::MeanStd {
                    mean: [0.5; 3],
                    std: [0.5; 3]
                }
            );
        }
        assert!(
            !outputs[0].trim().is_empty(),
            "outputs={outputs:?}; logs={:?}",
            messages.lock().unwrap()
        );
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[0], outputs[2]);
        eprintln!("lazy/warm/preencoded image responses: {outputs:?}");
    }

    #[test]
    fn finds_first_complete_json_object_after_prefix_junk() {
        let text = "noise before {\"type\":\"final\",\"content\":\"ok\"} trailing";
        let span = find_first_complete_json_object_span(text).expect("json span");
        assert_eq!(
            &text[span.0..span.1],
            "{\"type\":\"final\",\"content\":\"ok\"}"
        );
    }

    #[test]
    fn extracts_first_complete_json_object_only() {
        let text = "  {\"type\":\"tool_call\",\"tool\":\"read_file\",\"args\":{\"path\":\"Cargo.toml\"}} extra";
        let extracted = extract_first_complete_json_object(text).expect("json object");
        assert_eq!(
            extracted,
            "{\"type\":\"tool_call\",\"tool\":\"read_file\",\"args\":{\"path\":\"Cargo.toml\"}}"
        );
    }

    #[test]
    fn incomplete_json_object_is_not_reported_complete() {
        let text = "{\"type\":\"final\",\"content\":\"unterminated\"";
        assert!(find_first_complete_json_object_span(text).is_none());
        assert!(extract_first_complete_json_object(text).is_none());
    }

    #[test]
    fn agent_json_safe_text_rejects_non_ascii_gibberish() {
        assert!(is_agent_json_safe_text("{\"type\":\"tool_call\"}"));
        assert!(!is_agent_json_safe_text("った"));
        assert!(!is_agent_json_safe_text("\u{0000}"));
    }

    #[test]
    fn agent_json_prefix_accepts_partial_final_response() {
        assert_eq!(
            match_agent_response_prefix("{\"type\":\"final\",\"content\":\"Bei"),
            PrefixMatch::Incomplete
        );
    }

    #[test]
    fn agent_json_prefix_accepts_compact_final_decision() {
        assert_eq!(
            match_agent_response_prefix("{\"type\":\"final\"}"),
            PrefixMatch::Complete
        );
    }

    #[test]
    fn agent_json_prefix_accepts_complete_tool_call_response() {
        assert_eq!(
            match_agent_response_prefix(
                "{\"type\":\"tool_call\",\"tool\":\"read_file\",\"args\":{\"path\":\"Cargo.toml\"}}"
            ),
            PrefixMatch::Complete
        );
    }

    #[test]
    fn agent_json_prefix_rejects_protocol_gibberish() {
        assert_eq!(
            match_agent_response_prefix("{\"type\":\"tool_call\" 0000"),
            PrefixMatch::Invalid
        );
    }

    #[test]
    fn compute_think_caps_uses_vendor_heuristic_without_override() {
        // qwen35 base 384, small model (28 layers), 16K context, ample budget.
        assert_eq!(
            compute_think_caps(384, 28, 16_384, 8_000, None),
            (384, 1536)
        );
        // 36 layers doubles the base.
        assert_eq!(
            compute_think_caps(384, 36, 16_384, 8_000, None),
            (768, 3072)
        );
    }

    #[test]
    fn compute_think_caps_override_replaces_heuristic_and_hard_ceiling() {
        // Override wins over the vendor base and the 1024-token hard cap.
        assert_eq!(
            compute_think_caps(384, 28, 16_384, 8_000, Some(1024)),
            (1024, 4096)
        );
        assert_eq!(
            compute_think_caps(384, 28, 16_384, 16_000, Some(2048)),
            (2048, 8192)
        );
    }

    #[test]
    fn compute_think_caps_override_is_bounded_by_decode_budget() {
        let (think_cap, _) = compute_think_caps(384, 28, 16_384, 500, Some(2048));
        assert_eq!(think_cap, 500);
    }

    #[test]
    fn promote_think_only_content_strips_wrapping_tags() {
        assert_eq!(
            promote_think_only_content("<think>\nHello there. How can I help?\n</think>"),
            Some("Hello there. How can I help?".to_string())
        );
        assert_eq!(
            promote_think_only_content("<think>\nHello there.\n</think>\n</user>"),
            Some("Hello there.".to_string())
        );
    }

    #[test]
    fn retry_text_rejects_tag_only_noise() {
        assert!(!has_meaningful_retry_text("</think>\n\n<|im_end|>"));
        assert!(!has_meaningful_retry_text("</think>\n</user>"));
        assert!(!has_meaningful_retry_text("</response>"));
        assert!(has_meaningful_retry_text("Final answer."));
    }

    #[test]
    fn sanitize_final_response_text_strips_trailing_protocol_markers() {
        assert_eq!(
            sanitize_final_response_text("Hello!\n</think>\n</user>"),
            "Hello!"
        );
        assert_eq!(
            sanitize_final_response_text("Hello!\n</think>\n\n</response>"),
            "Hello!"
        );
        assert_eq!(
            sanitize_final_response_text("During execution...\n</think>\n</thn"),
            "During execution..."
        );
        assert_eq!(
            sanitize_final_response_text("The capital is Paris.\n</thinking>\n</response>"),
            "The capital is Paris."
        );
    }

    #[test]
    fn visible_think_tail_drops_incomplete_terminal_think_fragment() {
        let mut tail = "</thn".to_string();
        assert_eq!(finalize_visible_think_tail(&mut tail, true), "");

        let mut tail = "ution\n</thn".to_string();
        assert_eq!(finalize_visible_think_tail(&mut tail, true), "ution\n");
    }

    #[test]
    fn promote_think_only_content_supports_thinking_alias_tags() {
        assert_eq!(
            promote_think_only_content("<thinking>\nHello there.\n</thinking>"),
            Some("Hello there.".to_string())
        );
    }

    #[test]
    fn stop_text_literals_strip_split_protocol_closer_during_streaming() {
        let mut output = String::new();
        let mut pending_newline = false;
        let mut stop_text_tail = String::new();
        let mut matched = None;

        append_visible_text_with_stop_literals(
            "Hello! I'm ready to help.\n</res",
            &mut output,
            &mut pending_newline,
            false,
            None,
            &["</response>"],
            &mut stop_text_tail,
            &mut matched,
        );
        append_visible_text_with_stop_literals(
            "ponse>",
            &mut output,
            &mut pending_newline,
            false,
            None,
            &["</response>"],
            &mut stop_text_tail,
            &mut matched,
        );
        flush_visible_text_stop_tail(
            &mut output,
            &mut pending_newline,
            false,
            None,
            &mut stop_text_tail,
            matched,
        );

        assert_eq!(matched, Some("</response>"));
        assert_eq!(output, "Hello! I'm ready to help.");
    }

    const TRANSCRIPTION_BACKENDS: &[crate::engine::types::AudioEncoderBackend] =
        &[crate::engine::types::AudioEncoderBackend::Qwen3Asr];

    #[test]
    fn audio_decode_reserve_covers_every_transcription_decode_cap() {
        for backend in TRANSCRIPTION_BACKENDS.iter().copied() {
            let policy = crate::vendors::audio_transcription_policy(backend);
            assert!(
                policy.max_new_tokens <= ModelRuntime::AUDIO_DECODE_RESERVE_TOKENS,
                "{} reserves {} decode token(s) but may generate {}",
                backend.as_str(),
                ModelRuntime::AUDIO_DECODE_RESERVE_TOKENS,
                policy.max_new_tokens
            );
        }
    }

    #[test]
    fn transcription_policy_disables_the_generic_repetition_guards() {
        for backend in TRANSCRIPTION_BACKENDS.iter().copied() {
            let policy = crate::vendors::audio_transcription_policy(backend);
            assert!(!policy.decode_policy.unconditional_loop_guard);
            assert!(!policy.decode_policy.deterministic_loop_guard);
        }
    }

    #[test]
    fn visible_think_stdout_is_not_buffered_for_cli_streaming() {
        let policy = VendorDecodePolicy {
            parse_think_tags: true,
            retry_without_think_when_no_post_think_text: true,
            ..VendorDecodePolicy::default()
        };
        assert!(!should_buffer_visible_think_stdout(
            true,
            false,
            ThinkMode::Yes,
            policy
        ));
        assert!(!should_buffer_visible_think_stdout(
            true,
            false,
            ThinkMode::No,
            policy
        ));
        assert!(!should_buffer_visible_think_stdout(
            true,
            true,
            ThinkMode::Yes,
            policy
        ));
    }

    #[test]
    fn mmproj_discovery_includes_family_sidecar_files_in_model_directory() {
        let temp = tempdir().expect("tempdir");
        let model_path = temp.path().join("Qwen3.5-35B-A3B-Q4_K_M.gguf");
        let sidecar_path = temp.path().join("mmproj-Qwen3.5-35B-A3B-F16.gguf");
        fs::write(&model_path, []).expect("write model placeholder");
        fs::write(&sidecar_path, []).expect("write sidecar placeholder");

        let candidates =
            ModelRuntime::discover_mmproj_candidates(model_path.to_str().expect("utf8 path"));
        assert!(
            candidates.iter().any(|path| path == &sidecar_path),
            "expected discovered candidates to include {}",
            sidecar_path.display()
        );
    }

    #[test]
    fn mmproj_discovery_uses_current_directory_for_relative_model_path() {
        let _cwd_guard = CWD_TEST_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let temp = tempdir().expect("tempdir");
        let old_cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(temp.path()).expect("chdir temp");

        let model_name = "Qwen3.5-35B-A3B-Q4_K_M.gguf";
        let sidecar_name = "mmproj-Qwen3.5-35B-A3B-F16.gguf";
        fs::write(model_name, []).expect("write model placeholder");
        fs::write(sidecar_name, []).expect("write sidecar placeholder");

        let candidates = ModelRuntime::discover_mmproj_candidates(model_name);
        let contains = candidates.iter().any(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name == sidecar_name)
                .unwrap_or(false)
        });

        std::env::set_current_dir(old_cwd).expect("restore cwd");

        assert!(
            contains,
            "expected discovered candidates to include {}",
            sidecar_name
        );
    }
}
