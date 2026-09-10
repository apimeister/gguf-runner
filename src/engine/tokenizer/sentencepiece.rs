//! Score-based UTF-8 symbol merging for GGUF's explicit no-prefix normalizer.
//! User-defined pieces are recognized before spaces are escaped. Unknown UTF-8
//! symbols remain merge candidates and fall back to byte tokens only at the end.

use super::SentencepieceCandidate;
use crate::engine::types::Tokenizer;
use std::collections::BinaryHeap;

pub(super) fn byte_token(raw: &str) -> Option<u8> {
    (raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>'))
        .then(|| u8::from_str_radix(&raw[3..5], 16).ok())
        .flatten()
}

pub(super) fn encode_without_prefix(tokenizer: &Tokenizer, text: &str, output: &mut Vec<i32>) {
    let mut cursor = 0;
    let mut start = 0;
    while cursor < text.len() {
        let special = tokenizer.sentencepiece_user_defined.iter().find(|&&id| {
            let token = &tokenizer.vocab[id as usize];
            !token.is_empty() && text[cursor..].starts_with(token)
        });
        if let Some(&id) = special {
            encode_piece(tokenizer, &text[start..cursor], output);
            output.push(id);
            cursor += tokenizer.vocab[id as usize].len();
            start = cursor;
        } else {
            cursor += text[cursor..].chars().next().unwrap().len_utf8();
        }
    }
    encode_piece(tokenizer, &text[start..], output);
}

fn encode_piece(tokenizer: &Tokenizer, text: &str, output: &mut Vec<i32>) {
    let text = text.replace(' ', "▁");
    let starts = text
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let count = starts.len();
    if count == 0 {
        return;
    }
    let mut ends = starts
        .iter()
        .skip(1)
        .copied()
        .chain([text.len()])
        .collect::<Vec<_>>();
    let mut next = (1..=count).collect::<Vec<_>>();
    let mut previous = (0..count)
        .map(|index| index.checked_sub(1))
        .collect::<Vec<_>>();
    let mut active = vec![true; count];
    let mut versions = vec![0u32; count];
    let mut heap = BinaryHeap::new();
    let candidate = |left: usize, right: usize, end: usize, version| {
        if right >= count {
            return None;
        }
        let id = *tokenizer.token_to_id.get(&text[starts[left]..end])?;
        Some(SentencepieceCandidate {
            score: tokenizer
                .vocab_scores
                .get(id as usize)
                .copied()
                .unwrap_or(0.0),
            merged_id: id,
            left,
            right,
            version,
        })
    };
    for left in 0..count - 1 {
        if let Some(pair) = candidate(left, left + 1, ends[left + 1], 0) {
            heap.push(pair);
        }
    }
    while let Some(pair) = heap.pop() {
        let left = pair.left;
        let right = pair.right;
        if !active[left] || !active[right] || next[left] != right || versions[left] != pair.version
        {
            continue;
        }
        // A right-hand symbol can grow while the left link remains unchanged.
        // Reject that stale candidate using the original merged token's length.
        if tokenizer.vocab[pair.merged_id as usize].len() != ends[right] - starts[left] {
            continue;
        }
        ends[left] = ends[right];
        next[left] = next[right];
        active[right] = false;
        versions[left] = versions[left].wrapping_add(1);
        if next[left] < count {
            previous[next[left]] = Some(left);
        }
        if let Some(before) = previous[left]
            && let Some(pair) = candidate(before, left, ends[left], versions[before])
        {
            heap.push(pair);
        }
        if next[left] < count
            && let Some(pair) = candidate(left, next[left], ends[next[left]], versions[left])
        {
            heap.push(pair);
        }
    }
    let mut index = 0;
    while index < count {
        let piece = &text[starts[index]..ends[index]];
        if let Some(&id) = tokenizer.token_to_id.get(piece) {
            output.push(id);
        } else {
            for byte in piece.as_bytes() {
                if let Some(&id) = tokenizer.token_to_id.get(&format!("<0x{byte:02X}>")) {
                    output.push(id);
                }
            }
        }
        index = next[index];
    }
}
