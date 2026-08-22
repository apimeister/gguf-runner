use crate::engine::types::{EncodedPrompt, PlaceholderSpan};
use std::collections::HashMap;

pub(crate) type PrefillEmbeddingMap = HashMap<usize, Vec<f32>>;

#[derive(Clone, Debug)]
pub(crate) struct MediaEmbeddingSequence {
    pub(crate) tokens: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MediaKind {
    Image,
    Audio,
}

impl MediaKind {
    fn label(self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Audio => "audio",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MediaSpanPlan {
    kind: MediaKind,
    span: PlaceholderSpan,
    embedding_tokens: usize,
}

fn append_modality_plans(
    encoded: &EncodedPrompt,
    kind: MediaKind,
    spans: &[PlaceholderSpan],
    embedding_token_counts: &[usize],
    plans: &mut Vec<MediaSpanPlan>,
) -> Result<(), String> {
    let label = kind.label();
    if spans.len() != embedding_token_counts.len() {
        return Err(format!(
            "{label} embedding expansion mismatch: {} prompt {label} span(s) but {} embedding group(s)",
            spans.len(),
            embedding_token_counts.len()
        ));
    }

    let mut seen_media_indices = vec![false; embedding_token_counts.len()];
    let mut previous_end = 0usize;
    for span in spans {
        let minimum_len = if span.replace_marker { 1 } else { 2 };
        if span.token_len < minimum_len {
            return Err(format!(
                "{label} placeholder span[{}] is too short: token_len={} (expected at least {minimum_len})",
                span.media_index, span.token_len
            ));
        }
        let span_end = span
            .token_start
            .checked_add(span.token_len)
            .ok_or_else(|| format!("{label} placeholder span range overflow"))?;
        if span_end > encoded.token_ids.len() {
            return Err(format!(
                "{label} placeholder span[{}] exceeds prompt token range",
                span.media_index
            ));
        }
        if span.token_start < previous_end {
            return Err(format!(
                "{label} placeholder spans overlap or are out of order around media index {}",
                span.media_index
            ));
        }
        previous_end = span_end;

        let embedding_tokens = *embedding_token_counts
            .get(span.media_index)
            .ok_or_else(|| {
                format!(
                    "{label} placeholder span references missing media index {}",
                    span.media_index
                )
            })?;
        if seen_media_indices[span.media_index] {
            return Err(format!(
                "duplicate {label} placeholder span for media index {}",
                span.media_index
            ));
        }
        if embedding_tokens == 0 {
            return Err(format!(
                "{label} embedding sequence[{}] is empty; at least one embedding token is required",
                span.media_index
            ));
        }
        seen_media_indices[span.media_index] = true;
        plans.push(MediaSpanPlan {
            kind,
            span: *span,
            embedding_tokens,
        });
    }
    Ok(())
}

fn ordered_media_plan(
    encoded: &EncodedPrompt,
    image_embedding_token_counts: &[usize],
    audio_embedding_token_counts: &[usize],
) -> Result<Vec<MediaSpanPlan>, String> {
    if !encoded.video_spans.is_empty() {
        return Err(
            "ordered media embedding expansion does not support video spans yet".to_string(),
        );
    }

    let capacity = encoded
        .image_spans
        .len()
        .checked_add(encoded.audio_spans.len())
        .ok_or_else(|| "media placeholder count overflow".to_string())?;
    let mut plans = Vec::with_capacity(capacity);
    append_modality_plans(
        encoded,
        MediaKind::Image,
        &encoded.image_spans,
        image_embedding_token_counts,
        &mut plans,
    )?;
    append_modality_plans(
        encoded,
        MediaKind::Audio,
        &encoded.audio_spans,
        audio_embedding_token_counts,
        &mut plans,
    )?;
    plans.sort_by_key(|plan| plan.span.token_start);

    let mut previous_end = 0usize;
    for plan in &plans {
        if plan.span.token_start < previous_end {
            return Err(format!(
                "{} placeholder span[{}] overlaps another media span",
                plan.kind.label(),
                plan.span.media_index
            ));
        }
        previous_end = plan.span.token_start + plan.span.token_len;
    }
    Ok(plans)
}

fn expanded_len_from_plan(
    original_token_count: usize,
    plans: &[MediaSpanPlan],
) -> Result<usize, String> {
    let mut expanded = original_token_count;
    for plan in plans {
        let replacement_tokens = if plan.span.replace_marker {
            plan.embedding_tokens
        } else {
            plan.embedding_tokens
                .checked_add(2)
                .ok_or_else(|| "media embedding token count overflow".to_string())?
        };
        expanded = expanded
            .checked_sub(plan.span.token_len)
            .and_then(|value| value.checked_add(replacement_tokens))
            .ok_or_else(|| "expanded media prompt token count overflow".to_string())?;
    }
    Ok(expanded)
}

pub(crate) fn expanded_media_prompt_token_count(
    encoded: &EncodedPrompt,
    image_embedding_token_counts: &[usize],
    audio_embedding_token_counts: &[usize],
) -> Result<usize, String> {
    let plans = ordered_media_plan(
        encoded,
        image_embedding_token_counts,
        audio_embedding_token_counts,
    )?;
    expanded_len_from_plan(encoded.token_ids.len(), &plans)
}

pub(crate) fn preflight_media_context(
    encoded: &EncodedPrompt,
    image_embedding_token_counts: &[usize],
    audio_embedding_token_counts: &[usize],
    context_limit: usize,
    decode_reserve: usize,
) -> Result<usize, String> {
    let expanded_tokens = expanded_media_prompt_token_count(
        encoded,
        image_embedding_token_counts,
        audio_embedding_token_counts,
    )?;
    let required = expanded_tokens
        .checked_add(decode_reserve)
        .ok_or_else(|| "media prompt context requirement overflow".to_string())?;
    if required > context_limit {
        return Err(format!(
            "expanded media prompt requires {expanded_tokens} prefill token(s) plus {decode_reserve} reserved decode token(s), exceeding context limit {context_limit}"
        ));
    }
    Ok(expanded_tokens)
}

fn validate_embedding_dimensions(
    kind: MediaKind,
    sequences: &[MediaEmbeddingSequence],
    expected_embedding_dim: usize,
) -> Result<(), String> {
    let label = kind.label();
    for (media_index, sequence) in sequences.iter().enumerate() {
        for (token_index, embedding) in sequence.tokens.iter().enumerate() {
            if embedding.len() != expected_embedding_dim {
                return Err(format!(
                    "{label} embedding dim mismatch for {label} {media_index} token {token_index}: got {}, expected {expected_embedding_dim}",
                    embedding.len()
                ));
            }
        }
    }
    Ok(())
}

fn marker_tokens(encoded: &EncodedPrompt, span: PlaceholderSpan) -> (i32, i32, i32) {
    let span_tokens = &encoded.token_ids[span.token_start..span.token_start + span.token_len];
    let begin = span_tokens[0];
    let end = span_tokens[span_tokens.len() - 1];
    let placeholder = if span.token_len >= 3 {
        span_tokens[1]
    } else {
        begin
    };
    (begin, placeholder, end)
}

pub(crate) fn expand_prompt_with_media_embeddings(
    encoded: &EncodedPrompt,
    image_embeddings: &[MediaEmbeddingSequence],
    audio_embeddings: &[MediaEmbeddingSequence],
    expected_embedding_dim: usize,
) -> Result<(Vec<i32>, PrefillEmbeddingMap), String> {
    validate_embedding_dimensions(MediaKind::Image, image_embeddings, expected_embedding_dim)?;
    validate_embedding_dimensions(MediaKind::Audio, audio_embeddings, expected_embedding_dim)?;
    let image_token_counts = image_embeddings
        .iter()
        .map(|sequence| sequence.tokens.len())
        .collect::<Vec<_>>();
    let audio_token_counts = audio_embeddings
        .iter()
        .map(|sequence| sequence.tokens.len())
        .collect::<Vec<_>>();
    let plans = ordered_media_plan(encoded, &image_token_counts, &audio_token_counts)?;
    let expanded_len = expanded_len_from_plan(encoded.token_ids.len(), &plans)?;
    let mut out_tokens = Vec::with_capacity(expanded_len);
    let mut injected_embeddings = PrefillEmbeddingMap::new();
    let mut source_cursor = 0usize;

    for plan in plans {
        out_tokens.extend_from_slice(&encoded.token_ids[source_cursor..plan.span.token_start]);
        let sequence = match plan.kind {
            MediaKind::Image => &image_embeddings[plan.span.media_index],
            MediaKind::Audio => &audio_embeddings[plan.span.media_index],
        };

        if plan.span.replace_marker {
            let placeholder = encoded.token_ids[plan.span.token_start];
            for embedding in &sequence.tokens {
                let destination = out_tokens.len();
                out_tokens.push(placeholder);
                injected_embeddings.insert(destination, embedding.clone());
            }
        } else {
            let (begin, placeholder, end) = marker_tokens(encoded, plan.span);
            out_tokens.push(begin);
            for embedding in &sequence.tokens {
                let destination = out_tokens.len();
                out_tokens.push(placeholder);
                injected_embeddings.insert(destination, embedding.clone());
            }
            out_tokens.push(end);
        }
        source_cursor = plan.span.token_start + plan.span.token_len;
    }

    out_tokens.extend_from_slice(&encoded.token_ids[source_cursor..]);
    debug_assert_eq!(out_tokens.len(), expanded_len);
    Ok((out_tokens, injected_embeddings))
}

#[cfg(test)]
mod tests {
    use super::{
        MediaEmbeddingSequence, expand_prompt_with_media_embeddings,
        expanded_media_prompt_token_count, preflight_media_context,
    };
    use crate::engine::types::{EncodedPrompt, PlaceholderSpan};

    fn sequence(values: &[f32]) -> MediaEmbeddingSequence {
        MediaEmbeddingSequence {
            tokens: values.iter().map(|value| vec![*value, -*value]).collect(),
        }
    }

    fn span(token_start: usize, media_index: usize) -> PlaceholderSpan {
        PlaceholderSpan {
            token_start,
            token_len: 3,
            media_index,
            replace_marker: false,
        }
    }

    #[test]
    fn expands_image_and_audio_in_source_token_order() {
        let encoded = EncodedPrompt {
            token_ids: vec![10, 20, 21, 22, 30, 40, 41, 42, 50],
            image_spans: vec![span(5, 0)],
            video_spans: Vec::new(),
            audio_spans: vec![span(1, 0)],
        };

        let (tokens, embeddings) = expand_prompt_with_media_embeddings(
            &encoded,
            &[sequence(&[3.0])],
            &[sequence(&[1.0, 2.0])],
            2,
        )
        .unwrap();

        assert_eq!(tokens, [10, 20, 21, 21, 22, 30, 40, 41, 42, 50]);
        assert_eq!(embeddings[&2], [1.0, -1.0]);
        assert_eq!(embeddings[&3], [2.0, -2.0]);
        assert_eq!(embeddings[&7], [3.0, -3.0]);
    }

    #[test]
    fn replace_marker_span_emits_only_embedding_slots() {
        let encoded = EncodedPrompt {
            token_ids: vec![10, 99, 20],
            image_spans: vec![PlaceholderSpan {
                token_start: 1,
                token_len: 1,
                media_index: 0,
                replace_marker: true,
            }],
            video_spans: Vec::new(),
            audio_spans: Vec::new(),
        };

        let (tokens, embeddings) =
            expand_prompt_with_media_embeddings(&encoded, &[sequence(&[1.0, 2.0])], &[], 2)
                .unwrap();

        assert_eq!(tokens, [10, 99, 99, 20]);
        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn rejects_cross_modality_overlap() {
        let encoded = EncodedPrompt {
            token_ids: vec![10, 20, 21, 22, 23, 30],
            image_spans: vec![span(1, 0)],
            video_spans: Vec::new(),
            audio_spans: vec![span(2, 0)],
        };

        let error = expanded_media_prompt_token_count(&encoded, &[1], &[1]).unwrap_err();

        assert!(error.contains("overlaps another media span"));
    }

    #[test]
    fn rejects_duplicate_media_index() {
        let encoded = EncodedPrompt {
            token_ids: vec![10, 20, 21, 22, 30, 40, 41, 42, 50],
            image_spans: vec![span(1, 0), span(5, 0)],
            video_spans: Vec::new(),
            audio_spans: Vec::new(),
        };

        let error = expanded_media_prompt_token_count(&encoded, &[1, 1], &[]).unwrap_err();

        assert!(error.contains("duplicate image placeholder"));
    }

    #[test]
    fn rejects_out_of_order_and_out_of_bounds_spans() {
        let mut encoded = EncodedPrompt {
            token_ids: vec![10, 20, 21, 22, 30, 40, 41, 42, 50],
            image_spans: vec![span(5, 1), span(1, 0)],
            video_spans: Vec::new(),
            audio_spans: Vec::new(),
        };

        let error = expanded_media_prompt_token_count(&encoded, &[1, 1], &[]).unwrap_err();
        assert!(error.contains("overlap or are out of order"));

        encoded.image_spans = vec![span(8, 0)];
        let error = expanded_media_prompt_token_count(&encoded, &[1], &[]).unwrap_err();
        assert!(error.contains("exceeds prompt token range"));
    }

    #[test]
    fn rejects_embedding_dimension_mismatch() {
        let encoded = EncodedPrompt {
            token_ids: vec![10, 20, 21, 22, 30],
            image_spans: Vec::new(),
            video_spans: Vec::new(),
            audio_spans: vec![span(1, 0)],
        };
        let audio = [MediaEmbeddingSequence {
            tokens: vec![vec![1.0]],
        }];

        let error = expand_prompt_with_media_embeddings(&encoded, &[], &audio, 2).unwrap_err();

        assert!(error.contains("audio embedding dim mismatch"));
    }

    #[test]
    fn context_preflight_reserves_decode_tokens_without_truncating_media() {
        let encoded = EncodedPrompt {
            token_ids: vec![10, 20, 21, 22, 30],
            image_spans: Vec::new(),
            video_spans: Vec::new(),
            audio_spans: vec![span(1, 0)],
        };

        assert_eq!(preflight_media_context(&encoded, &[], &[4], 10, 2), Ok(8));
        let error = preflight_media_context(&encoded, &[], &[4], 9, 2).unwrap_err();
        assert!(error.contains("exceeding context limit 9"));
    }
}
