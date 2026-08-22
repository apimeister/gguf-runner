use crate::engine::io::{get_gguf_bool_from_map, get_gguf_int_from_map, get_gguf_string_from_map};
use crate::engine::types::{
    Config, GGUFFile, GgufValue, LLAMA3_BOS_TOKEN, LLAMA3_END_HEADER, LLAMA3_EOS_TOKEN, LLAMA3_EOT,
    LLAMA3_START_HEADER, Tokenizer, TokenizerPreType, VendorTokenizerPolicy,
};
use fancy_regex::Regex;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

fn tiktoken_decode_map() -> [i16; 512] {
    let mut map = [-1i16; 512];
    let mut n = 0i16;
    for b in 0..=255u16 {
        let b8 = b as u8;
        if (33..=126).contains(&b8) || (161..=172).contains(&b8) || (174..=255).contains(&b8) {
            map[b as usize] = b as i16;
        } else {
            map[(256 + n as u16) as usize] = b as i16;
            n += 1;
        }
    }
    map
}

fn tiktoken_encode_map() -> [u32; 256] {
    let mut map = [0u32; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let b8 = b as u8;
        if (33..=126).contains(&b8) || (161..=172).contains(&b8) || (174..=255).contains(&b8) {
            map[b as usize] = b;
        } else {
            map[b as usize] = 256 + n;
            n += 1;
        }
    }
    map
}

fn decode_sentencepiece(s: &str) -> String {
    s.replace('\u{2581}', " ")
}

fn decode_tiktoken_internal(s: &str) -> String {
    let out = decode_tiktoken_bytes(s);
    String::from_utf8_lossy(&out).to_string()
}

fn decode_tiktoken_bytes(s: &str) -> Vec<u8> {
    let map = tiktoken_decode_map();
    let mut out: Vec<u8> = Vec::with_capacity(s.len());

    for ch in s.chars() {
        let cp = ch as u32;
        if cp < 512 {
            let v = map[cp as usize];
            if v >= 0 {
                out.push(v as u8);
                continue;
            }
        }
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        out.extend_from_slice(encoded.as_bytes());
    }

    out
}

fn text_to_tiktoken(text: &str) -> String {
    let map = tiktoken_encode_map();
    let mut out = String::with_capacity(text.len() * 2);
    for b in text.as_bytes() {
        let cp = map[*b as usize];
        if let Some(ch) = char::from_u32(cp) {
            out.push(ch);
        }
    }
    out
}

fn text_to_sentencepiece(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    let mut need_prefix = true;

    for b in text.bytes() {
        match b {
            b' ' => {
                out.push('\u{2581}');
                need_prefix = false;
            }
            b'\n' | b'\t' | b'\r' => {
                out.push(b as char);
                need_prefix = true;
            }
            _ => {
                if need_prefix && (b as char).is_ascii_alphanumeric() {
                    out.push('\u{2581}');
                }
                out.push(b as char);
                need_prefix = false;
            }
        }
    }

    out
}

#[cfg(test)]
fn split_gpt2_pieces(text: &str) -> Vec<String> {
    fn contraction_len(s: &str, idx: usize) -> usize {
        let rest = &s[idx..];
        for pat in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
            if rest.starts_with(pat) {
                return pat.len();
            }
        }
        0
    }

    fn next_char(s: &str, idx: usize) -> Option<(char, usize)> {
        s[idx..].chars().next().map(|c| (c, c.len_utf8()))
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    enum Kind {
        Alpha,
        Numeric,
        Other,
    }

    fn char_kind(c: char) -> Kind {
        if c.is_alphabetic() {
            Kind::Alpha
        } else if c.is_numeric() {
            Kind::Numeric
        } else {
            Kind::Other
        }
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    let len = text.len();

    while i < len {
        let (c0, c0_len) = match next_char(text, i) {
            Some(v) => v,
            None => break,
        };

        if c0.is_whitespace() && c0 != ' ' {
            let start = i;
            i += c0_len;
            while i < len {
                if let Some((c, clen)) = next_char(text, i) {
                    if c.is_whitespace() && c != ' ' {
                        i += clen;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            out.push(text[start..i].to_string());
            continue;
        }

        if c0 == ' ' {
            let mut j = i + c0_len;
            if j >= len {
                out.push(" ".to_string());
                break;
            }
            if let Some((c1, _)) = next_char(text, j)
                && c1.is_whitespace()
            {
                let start = i;
                while j < len {
                    if let Some((c, clen)) = next_char(text, j) {
                        if c.is_whitespace() {
                            j += clen;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                out.push(text[start..j].to_string());
                i = j;
                continue;
            }

            let start = i;
            i = j;
            let contr = contraction_len(text, i);
            if contr > 0 {
                i += contr;
                out.push(text[start..i].to_string());
                continue;
            }
            if let Some((c1, clen1)) = next_char(text, i) {
                let kind = char_kind(c1);
                i += clen1;
                while i < len {
                    let contr2 = contraction_len(text, i);
                    if contr2 > 0 {
                        break;
                    }
                    if let Some((c, clen)) = next_char(text, i) {
                        if c.is_whitespace() {
                            break;
                        }
                        if char_kind(c) != kind {
                            break;
                        }
                        i += clen;
                    } else {
                        break;
                    }
                }
                out.push(text[start..i].to_string());
                continue;
            }
        }

        let contr = contraction_len(text, i);
        if contr > 0 {
            let start = i;
            i += contr;
            out.push(text[start..i].to_string());
            continue;
        }

        let start = i;
        let kind = char_kind(c0);
        i += c0_len;
        while i < len {
            let contr2 = contraction_len(text, i);
            if contr2 > 0 {
                break;
            }
            if let Some((c, clen)) = next_char(text, i) {
                if c.is_whitespace() {
                    break;
                }
                if char_kind(c) != kind {
                    break;
                }
                i += clen;
            } else {
                break;
            }
        }
        out.push(text[start..i].to_string());
    }

    out
}

fn for_each_gpt2_piece<F>(text: &str, mut f: F)
where
    F: FnMut(&str),
{
    fn contraction_len(s: &str, idx: usize) -> usize {
        let rest = &s[idx..];
        for pat in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
            if rest.starts_with(pat) {
                return pat.len();
            }
        }
        0
    }

    fn next_char(s: &str, idx: usize) -> Option<(char, usize)> {
        s[idx..].chars().next().map(|c| (c, c.len_utf8()))
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    enum Kind {
        Alpha,
        Numeric,
        Other,
    }

    fn char_kind(c: char) -> Kind {
        if c.is_alphabetic() {
            Kind::Alpha
        } else if c.is_numeric() {
            Kind::Numeric
        } else {
            Kind::Other
        }
    }

    let mut i = 0usize;
    let len = text.len();

    while i < len {
        let (c0, c0_len) = match next_char(text, i) {
            Some(v) => v,
            None => break,
        };

        if c0.is_whitespace() && c0 != ' ' {
            let start = i;
            i += c0_len;
            while i < len {
                if let Some((c, clen)) = next_char(text, i) {
                    if c.is_whitespace() && c != ' ' {
                        i += clen;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            f(&text[start..i]);
            continue;
        }

        if c0 == ' ' {
            let mut j = i + c0_len;
            if j >= len {
                f(" ");
                break;
            }
            if let Some((c1, _)) = next_char(text, j)
                && c1.is_whitespace()
            {
                let start = i;
                while j < len {
                    if let Some((c, clen)) = next_char(text, j) {
                        if c.is_whitespace() {
                            j += clen;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                f(&text[start..j]);
                i = j;
                continue;
            }

            let start = i;
            i = j;
            let contr = contraction_len(text, i);
            if contr > 0 {
                i += contr;
                f(&text[start..i]);
                continue;
            }
            if let Some((c1, clen1)) = next_char(text, i) {
                let kind = char_kind(c1);
                i += clen1;
                while i < len {
                    let contr2 = contraction_len(text, i);
                    if contr2 > 0 {
                        break;
                    }
                    if let Some((c, clen)) = next_char(text, i) {
                        if c.is_whitespace() {
                            break;
                        }
                        if char_kind(c) != kind {
                            break;
                        }
                        i += clen;
                    } else {
                        break;
                    }
                }
                f(&text[start..i]);
                continue;
            }
        }

        let contr = contraction_len(text, i);
        if contr > 0 {
            let start = i;
            i += contr;
            f(&text[start..i]);
            continue;
        }

        let start = i;
        let kind = char_kind(c0);
        i += c0_len;
        while i < len {
            let contr2 = contraction_len(text, i);
            if contr2 > 0 {
                break;
            }
            if let Some((c, clen)) = next_char(text, i) {
                if c.is_whitespace() {
                    break;
                }
                if char_kind(c) != kind {
                    break;
                }
                i += clen;
            } else {
                break;
            }
        }
        f(&text[start..i]);
    }
}

#[cfg(test)]
fn split_with_regex(text: &str, re: &Regex) -> Option<Vec<String>> {
    let mut out = Vec::new();
    let mut covered = 0usize;
    let mut had_match = false;
    for m in re.find_iter(text) {
        let m = match m {
            Ok(v) => v,
            Err(_) => return None,
        };
        had_match = true;
        if m.start() > covered {
            out.push(text[covered..m.start()].to_string());
        }
        out.push(m.as_str().to_string());
        covered = m.end();
    }
    if !had_match {
        return Some(vec![text.to_string()]);
    }
    if covered < text.len() {
        out.push(text[covered..].to_string());
    }
    Some(out)
}

fn for_each_regex_piece<F>(text: &str, re: &Regex, mut f: F) -> Result<(), ()>
where
    F: FnMut(&str),
{
    let mut covered = 0usize;
    let mut had_match = false;
    for m in re.find_iter(text) {
        let m = match m {
            Ok(v) => v,
            Err(_) => return Err(()),
        };
        had_match = true;
        if m.start() > covered {
            f(&text[covered..m.start()]);
        }
        f(m.as_str());
        covered = m.end();
    }
    if !had_match {
        f(text);
        return Ok(());
    }
    if covered < text.len() {
        f(&text[covered..]);
    }
    Ok(())
}

#[cfg(test)]
fn split_qwen2_pieces(text: &str) -> Vec<String> {
    static QWEN2_RE: OnceLock<Regex> = OnceLock::new();
    let re = QWEN2_RE.get_or_init(|| {
        Regex::new(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )
        .expect("valid qwen2 pre-tokenizer regex")
    });
    split_with_regex(text, re).unwrap_or_else(|| split_gpt2_pieces(text))
}

#[cfg(test)]
fn split_qwen35_pieces(text: &str) -> Vec<String> {
    static QWEN35_RE: OnceLock<Regex> = OnceLock::new();
    let re = QWEN35_RE.get_or_init(|| {
        Regex::new(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )
        .expect("valid qwen35 pre-tokenizer regex")
    });
    split_with_regex(text, re).unwrap_or_else(|| split_gpt2_pieces(text))
}

fn for_each_qwen2_piece<F>(text: &str, f: F)
where
    F: FnMut(&str),
{
    static QWEN2_RE: OnceLock<Regex> = OnceLock::new();
    let re = QWEN2_RE.get_or_init(|| {
        Regex::new(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )
        .expect("valid qwen2 pre-tokenizer regex")
    });
    let mut f = f;
    if for_each_regex_piece(text, re, &mut f).is_err() {
        for_each_gpt2_piece(text, &mut f);
    }
}

fn for_each_qwen35_piece<F>(text: &str, f: F)
where
    F: FnMut(&str),
{
    static QWEN35_RE: OnceLock<Regex> = OnceLock::new();
    let re = QWEN35_RE.get_or_init(|| {
        Regex::new(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )
        .expect("valid qwen35 pre-tokenizer regex")
    });
    let mut f = f;
    if for_each_regex_piece(text, re, &mut f).is_err() {
        for_each_gpt2_piece(text, &mut f);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SentencepieceCandidate {
    score: f32,
    merged_id: i32,
    left: usize,
    right: usize,
    version: u32,
}

impl Eq for SentencepieceCandidate {}

impl Ord for SentencepieceCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| self.version.cmp(&other.version))
    }
}

impl PartialOrd for SentencepieceCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BpeCandidate {
    rank: usize,
    merged_id: i32,
    left: usize,
    right: usize,
    version: u32,
}

impl Ord for BpeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| self.version.cmp(&other.version))
    }
}

impl PartialOrd for BpeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Tokenizer {
    fn sentencepiece_candidate(
        &self,
        work: &[i32],
        left: usize,
        right: usize,
        merged: &mut String,
        version: u32,
    ) -> Option<SentencepieceCandidate> {
        let left_token = &self.vocab[work[left] as usize];
        let right_token = &self.vocab[work[right] as usize];
        merged.clear();
        merged.reserve(left_token.len() + right_token.len());
        merged.push_str(left_token);
        merged.push_str(right_token);
        let &merged_id = self.token_to_id.get(merged.as_str())?;
        let score = self
            .vocab_scores
            .get(merged_id as usize)
            .copied()
            .unwrap_or(0.0);
        Some(SentencepieceCandidate {
            score,
            merged_id,
            left,
            right,
            version,
        })
    }

    fn bpe_candidate(
        &self,
        work: &[i32],
        left: usize,
        right: usize,
        pair: &mut String,
        merged: &mut String,
        version: u32,
    ) -> Option<BpeCandidate> {
        let left_token = &self.vocab[work[left] as usize];
        let right_token = &self.vocab[work[right] as usize];
        pair.clear();
        pair.reserve(left_token.len() + 1 + right_token.len());
        pair.push_str(left_token);
        pair.push(' ');
        pair.push_str(right_token);
        let &rank = self.merge_ranks.get(pair.as_str())?;
        merged.clear();
        merged.reserve(left_token.len() + right_token.len());
        merged.push_str(left_token);
        merged.push_str(right_token);
        let &merged_id = self.token_to_id.get(merged.as_str())?;
        Some(BpeCandidate {
            rank,
            merged_id,
            left,
            right,
            version,
        })
    }

    fn merge_sentencepiece_work(&self, work: &mut [i32], out: &mut Vec<i32>) {
        if work.is_empty() {
            return;
        }
        if work.len() == 1 {
            out.push(work[0]);
            return;
        }

        const NONE: usize = usize::MAX;
        let mut prev = vec![NONE; work.len()];
        let mut next = vec![NONE; work.len()];
        let mut active = vec![true; work.len()];
        for (i, slot) in next.iter_mut().enumerate() {
            if i > 0 {
                prev[i] = i - 1;
            }
            if i + 1 < work.len() {
                *slot = i + 1;
            }
        }

        let head = 0usize;
        let mut live = work.len();
        let mut merged = String::new();
        let mut versions = vec![0u32; work.len()];
        let mut heap = BinaryHeap::new();

        let mut pos = head;
        while pos != NONE {
            let right_pos = next[pos];
            if right_pos == NONE {
                break;
            }
            if let Some(candidate) =
                self.sentencepiece_candidate(work, pos, right_pos, &mut merged, versions[pos])
            {
                heap.push(candidate);
            }
            pos = right_pos;
        }

        while live > 1 {
            let Some(candidate) = heap.pop() else {
                break;
            };
            if !active[candidate.left]
                || !active[candidate.right]
                || versions[candidate.left] != candidate.version
                || next[candidate.left] != candidate.right
            {
                continue;
            }

            let best_pos = candidate.left;
            let removed = candidate.right;
            debug_assert_ne!(removed, NONE);
            work[best_pos] = candidate.merged_id;
            let after_removed = next[removed];
            next[best_pos] = after_removed;
            active[removed] = false;
            prev[removed] = NONE;
            next[removed] = NONE;
            if after_removed != NONE {
                prev[after_removed] = best_pos;
            }
            versions[best_pos] = versions[best_pos].wrapping_add(1);
            live -= 1;

            let left_neighbor = prev[best_pos];
            if left_neighbor != NONE {
                versions[left_neighbor] = versions[left_neighbor].wrapping_add(1);
                if let Some(new_candidate) = self.sentencepiece_candidate(
                    work,
                    left_neighbor,
                    best_pos,
                    &mut merged,
                    versions[left_neighbor],
                ) {
                    heap.push(new_candidate);
                }
            }
            if after_removed != NONE
                && let Some(new_candidate) = self.sentencepiece_candidate(
                    work,
                    best_pos,
                    after_removed,
                    &mut merged,
                    versions[best_pos],
                )
            {
                heap.push(new_candidate);
            }
        }

        let mut pos = head;
        while pos != NONE {
            out.push(work[pos]);
            pos = next[pos];
        }
    }

    fn merge_bpe_work(&self, work: &mut [i32], out: &mut Vec<i32>) {
        if work.is_empty() {
            return;
        }
        if work.len() == 1 {
            out.push(work[0]);
            return;
        }

        const NONE: usize = usize::MAX;
        let mut prev = vec![NONE; work.len()];
        let mut next = vec![NONE; work.len()];
        let mut active = vec![true; work.len()];
        for (i, slot) in next.iter_mut().enumerate() {
            if i > 0 {
                prev[i] = i - 1;
            }
            if i + 1 < work.len() {
                *slot = i + 1;
            }
        }

        let head = 0usize;
        let mut live = work.len();
        let mut pair = String::new();
        let mut merged = String::new();
        let mut versions = vec![0u32; work.len()];
        let mut heap = BinaryHeap::new();

        let mut pos = head;
        while pos != NONE {
            let right_pos = next[pos];
            if right_pos == NONE {
                break;
            }
            if let Some(candidate) =
                self.bpe_candidate(work, pos, right_pos, &mut pair, &mut merged, versions[pos])
            {
                heap.push(candidate);
            }
            pos = right_pos;
        }

        while live > 1 {
            let Some(candidate) = heap.pop() else {
                break;
            };
            if !active[candidate.left]
                || !active[candidate.right]
                || versions[candidate.left] != candidate.version
                || next[candidate.left] != candidate.right
            {
                continue;
            }

            let best_pos = candidate.left;
            let removed = candidate.right;
            debug_assert_ne!(removed, NONE);
            work[best_pos] = candidate.merged_id;
            let after_removed = next[removed];
            next[best_pos] = after_removed;
            active[removed] = false;
            prev[removed] = NONE;
            next[removed] = NONE;
            if after_removed != NONE {
                prev[after_removed] = best_pos;
            }
            versions[best_pos] = versions[best_pos].wrapping_add(1);
            live -= 1;

            let left_neighbor = prev[best_pos];
            if left_neighbor != NONE {
                versions[left_neighbor] = versions[left_neighbor].wrapping_add(1);
                if let Some(new_candidate) = self.bpe_candidate(
                    work,
                    left_neighbor,
                    best_pos,
                    &mut pair,
                    &mut merged,
                    versions[left_neighbor],
                ) {
                    heap.push(new_candidate);
                }
            }
            if after_removed != NONE
                && let Some(new_candidate) = self.bpe_candidate(
                    work,
                    best_pos,
                    after_removed,
                    &mut pair,
                    &mut merged,
                    versions[best_pos],
                )
            {
                heap.push(new_candidate);
            }
        }

        let mut pos = head;
        while pos != NONE {
            out.push(work[pos]);
            pos = next[pos];
        }
    }

    pub(crate) fn prepare_for_encode(&mut self) {
        self.build_token_lookup();
        if !self.use_sentencepiece {
            self.build_merge_ranks();
        }
    }

    pub(crate) fn find_special_token(&self, token_str: &str) -> Option<i32> {
        self.vocab
            .iter()
            .position(|s| s == token_str)
            .map(|i| i as i32)
    }

    fn build_token_lookup(&mut self) {
        if !self.token_to_id.is_empty() {
            return;
        }
        let mut map = HashMap::with_capacity(self.vocab.len() * 2);
        for (id, tok) in self.vocab.iter().enumerate() {
            map.entry(tok.clone()).or_insert(id as i32);
        }
        self.token_to_id = map;
    }

    fn build_merge_ranks(&mut self) {
        if !self.merge_ranks.is_empty() {
            return;
        }
        let mut ranks = HashMap::with_capacity(self.merges.len() * 2);
        for (rank, m) in self.merges.iter().enumerate() {
            ranks.entry(m.clone()).or_insert(rank);
        }
        self.merge_ranks = ranks;
    }

    pub(crate) fn encode_prepared(&self, text: &str, tokens: &mut Vec<i32>) {
        tokens.clear();
        if text.is_empty() {
            return;
        }

        if self.use_sentencepiece {
            let encoded_text = text_to_sentencepiece(text);

            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                if let Some(&id) = self.token_to_id.get(s) {
                    work.push(id);
                }
            }
            if work.is_empty() {
                return;
            }

            self.merge_sentencepiece_work(&mut work, tokens);
            return;
        }

        let mut encode_piece = |piece: &str| {
            let encoded_text = text_to_tiktoken(piece);
            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                if let Some(&id) = self.token_to_id.get(s) {
                    work.push(id);
                }
            }
            if work.is_empty() {
                return;
            }

            self.merge_bpe_work(&mut work, tokens);
        };
        match self.pre_tokenizer {
            TokenizerPreType::Qwen2 => for_each_qwen2_piece(text, &mut encode_piece),
            TokenizerPreType::Qwen35 => for_each_qwen35_piece(text, &mut encode_piece),
            TokenizerPreType::Gpt2 => for_each_gpt2_piece(text, &mut encode_piece),
        }
    }

    pub(crate) fn bpe_encode(&mut self, text: &str, tokens: &mut Vec<i32>) {
        self.prepare_for_encode();
        self.encode_prepared(text, tokens);
    }

    pub(crate) fn decode_token(&self, token_id: i32) -> Option<String> {
        if token_id < 0 || token_id as usize >= self.vocab.len() {
            return None;
        }
        let raw = &self.vocab[token_id as usize];
        if self.use_sentencepiece {
            Some(decode_sentencepiece(raw))
        } else {
            Some(decode_tiktoken_internal(raw))
        }
    }

    pub(crate) fn decode_token_bytes(&self, token_id: i32) -> Option<Vec<u8>> {
        if token_id < 0 || token_id as usize >= self.vocab.len() {
            return None;
        }
        let raw = &self.vocab[token_id as usize];
        if self.use_sentencepiece {
            Some(decode_sentencepiece(raw).into_bytes())
        } else {
            Some(decode_tiktoken_bytes(raw))
        }
    }
}

pub(crate) fn init_tokenizer_from_gguf(
    gguf: &GGUFFile,
    config: &mut Config,
    policy: VendorTokenizerPolicy,
    debug_mode: bool,
) -> Result<Tokenizer, String> {
    if gguf.vocab_tokens.is_empty() {
        return Err("no vocabulary found in GGUF file".to_string());
    }

    let mut tokenizer = Tokenizer::default();
    tokenizer.pre_tokenizer = match get_gguf_string_from_map(&gguf.kv, "tokenizer.ggml.pre") {
        Some("qwen2") | Some("megrez") => TokenizerPreType::Qwen2,
        Some("qwen35") => TokenizerPreType::Qwen35,
        _ => TokenizerPreType::Gpt2,
    };
    tokenizer.bos_token = match gguf.kv.get("tokenizer.ggml.bos_token_id") {
        Some(GgufValue::UInt(v)) => *v as i32,
        Some(GgufValue::Int(v)) => *v as i32,
        _ => -1,
    };
    tokenizer.skip_bos_token =
        !get_gguf_bool_from_map(&gguf.kv, "tokenizer.ggml.add_bos_token", true);
    tokenizer.eos_token = get_gguf_int_from_map(
        &gguf.kv,
        "tokenizer.ggml.eos_token_id",
        LLAMA3_EOS_TOKEN as i64,
    ) as i32;
    tokenizer.start_header_token = LLAMA3_START_HEADER;
    tokenizer.end_header_token = LLAMA3_END_HEADER;
    // Resolve end-of-turn token via vendor policy first, then fallback to Llama-style `<|eot_id|>`.
    tokenizer.eot_token = policy
        .end_turn_token_literals
        .iter()
        .find_map(|token| gguf.vocab_tokens.iter().position(|s| s == *token))
        .map(|i| i as i32)
        .or_else(|| {
            gguf.vocab_tokens
                .iter()
                .position(|s| s == "<|eot_id|>")
                .map(|i| i as i32)
        })
        .unwrap_or(LLAMA3_EOT);

    tokenizer.vocab = gguf.vocab_tokens.clone();
    tokenizer.vocab_size = tokenizer.vocab.len();
    tokenizer.max_token_length = tokenizer
        .vocab
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(256)
        .max(1);
    tokenizer.vocab_scores = if gguf.vocab_scores.is_empty() {
        vec![0.0; tokenizer.vocab_size]
    } else {
        gguf.vocab_scores.clone()
    };
    tokenizer.merges = gguf.vocab_merges.clone();
    if tokenizer.bos_token < 0 {
        if policy.disable_bos_fallback {
            tokenizer.bos_token = -1;
        } else {
            tokenizer.bos_token = tokenizer
                .vocab
                .iter()
                .position(|s| s == "<|begin_of_text|>")
                .map(|i| i as i32)
                .or_else(|| {
                    tokenizer
                        .vocab
                        .iter()
                        .position(|s| s == "<s>")
                        .map(|i| i as i32)
                })
                .unwrap_or(LLAMA3_BOS_TOKEN);
        }
    }

    if debug_mode {
        eprintln!(
            "Using vocabulary from GGUF file ({} tokens), pre-tokenizer={:?}",
            tokenizer.vocab_size, tokenizer.pre_tokenizer
        );
    }

    if config.vocab_size != tokenizer.vocab_size {
        if debug_mode {
            eprintln!(
                "Note: Updating vocab_size from {} to {} based on GGUF",
                config.vocab_size, tokenizer.vocab_size
            );
        }
        config.vocab_size = tokenizer.vocab_size;
    }

    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::env;
    use std::fs;
    use std::hint::black_box;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    fn legacy_encode_prepared(tokenizer: &Tokenizer, text: &str, tokens: &mut Vec<i32>) {
        tokens.clear();
        if text.is_empty() {
            return;
        }

        if tokenizer.use_sentencepiece {
            let encoded_text = text_to_sentencepiece(text);
            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let s = ch.to_string();
                if let Some(&id) = tokenizer.token_to_id.get(&s) {
                    work.push(id);
                }
            }
            if work.is_empty() {
                return;
            }

            while work.len() > 1 {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_id = -1i32;
                let mut best_pos = 0usize;

                for i in 0..work.len() - 1 {
                    let left = &tokenizer.vocab[work[i] as usize];
                    let right = &tokenizer.vocab[work[i + 1] as usize];
                    let merged = format!("{left}{right}");
                    if let Some(&id) = tokenizer.token_to_id.get(&merged) {
                        let score = tokenizer
                            .vocab_scores
                            .get(id as usize)
                            .copied()
                            .unwrap_or(0.0);
                        if score > best_score {
                            best_score = score;
                            best_id = id;
                            best_pos = i;
                        }
                    }
                }

                if best_id < 0 {
                    break;
                }

                work[best_pos] = best_id;
                work.remove(best_pos + 1);
            }

            tokens.extend(work);
            return;
        }

        let pieces = match tokenizer.pre_tokenizer {
            TokenizerPreType::Qwen2 => split_qwen2_pieces(text),
            TokenizerPreType::Qwen35 => split_qwen35_pieces(text),
            TokenizerPreType::Gpt2 => split_gpt2_pieces(text),
        };
        for piece in pieces {
            let encoded_text = text_to_tiktoken(&piece);
            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let s = ch.to_string();
                if let Some(&id) = tokenizer.token_to_id.get(&s) {
                    work.push(id);
                }
            }
            if work.is_empty() {
                continue;
            }

            while work.len() > 1 {
                let mut best_rank = usize::MAX;
                let mut best_id = -1i32;
                let mut best_pos = 0usize;

                for i in 0..work.len() - 1 {
                    let left = &tokenizer.vocab[work[i] as usize];
                    let right = &tokenizer.vocab[work[i + 1] as usize];
                    let pair = format!("{left} {right}");
                    let merged = format!("{left}{right}");
                    if let Some(&rank) = tokenizer.merge_ranks.get(&pair)
                        && let Some(&id) = tokenizer.token_to_id.get(&merged)
                        && rank < best_rank
                    {
                        best_rank = rank;
                        best_id = id;
                        best_pos = i;
                    }
                }

                if best_id < 0 {
                    break;
                }

                work[best_pos] = best_id;
                work.remove(best_pos + 1);
            }

            tokens.extend(work);
        }
    }

    fn synthetic_markdown(section_count: usize) -> String {
        let mut out = String::new();
        for idx in 0..section_count {
            out.push_str("# Service Runbook\n\n");
            out.push_str("## Deployment\n");
            out.push_str("- release: 2026.");
            out.push_str(&(idx % 10).to_string());
            out.push('\n');
            out.push_str("- owner: platform\n");
            out.push_str("- action: restart api and reindex docs cache\n\n");
            out.push_str("When indexing markdown for retrieval, prefer smaller token windows and preserve headings so search_document prefixes retain semantic structure.\n\n");
            out.push_str("```rust\n");
            out.push_str("fn deploy(region: &str) { println!(\"deploy {region}\"); }\n");
            out.push_str("fn rollback(region: &str) { println!(\"rollback {region}\"); }\n");
            out.push_str("```\n\n");
            out.push_str("### Notes\n");
            out.push_str("Synthetic corpus for tokenizer benchmarking. Repeat stable prose, identifiers, bullet points, and code spans to exercise repeated merges.\n\n");
        }
        out
    }

    fn corpus_shaped_markdown(target_bytes: usize, include_code: bool) -> String {
        let mut out = String::new();
        let mut section = 0usize;
        while out.len() < target_bytes {
            out.push_str("# Incident ");
            out.push_str(&(section % 17).to_string());
            out.push_str("\n\n");
            out.push_str("## Summary\n");
            out.push_str("Service owners documented rollout state, rollback instructions, pager escalation, and retrieval-specific wording for exact search hits.\n\n");
            out.push_str("## Checklist\n");
            out.push_str("- verify deployment window\n");
            out.push_str("- compare staging metrics against production baseline\n");
            out.push_str("- record shard identifiers, release hashes, and wiki backlinks\n\n");
            out.push_str("## Notes\n");
            out.push_str("Chunk-oriented indexing should preserve headings, lists, inline identifiers like api_gateway_v2, and brief prose paragraphs so retrieval stays stable across operational docs.\n\n");
            out.push_str("| Field | Value |\n| --- | --- |\n");
            out.push_str(
                "| owner | platform |\n| priority | high |\n| runbook | docs/ops/rollback.md |\n\n",
            );
            if include_code {
                out.push_str("```yaml\n");
                out.push_str("service: api-gateway\n");
                out.push_str("rollout: canary\n");
                out.push_str("regions: [eu-central-1, us-east-1]\n");
                out.push_str("alerts:\n  - 5xx_rate\n  - p95_latency\n");
                out.push_str("```\n\n");
                out.push_str("```rust\n");
                out.push_str("fn reindex_chunk(doc_id: &str, shard: usize) { println!(\"reindex {doc_id}:{shard}\"); }\n");
                out.push_str(
                    "fn rollback_release(version: &str) { println!(\"rollback {version}\"); }\n",
                );
                out.push_str("```\n\n");
            }
            out.push_str("### Follow-up\n");
            out.push_str("Repeat exact identifiers, timestamps, and command snippets across sections to mimic realistic markdown corpora with partial duplication and stable terminology.\n\n");
            section += 1;
        }
        out.truncate(target_bytes);
        out
    }

    fn collect_markdown_paths(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("cannot read benchmark source dir '{}': {e}", dir.display()))?;
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|v| v.path()))
            .collect();
        paths.sort();
        for path in paths {
            if path.is_dir() {
                collect_markdown_paths(&path, out)?;
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                out.push(path);
            }
        }
        Ok(())
    }

    fn load_markdown_documents_from_dir(dir: &Path) -> Result<Vec<String>, String> {
        let mut paths = Vec::new();
        collect_markdown_paths(dir, &mut paths)?;
        let mut docs = Vec::new();
        for path in paths {
            let content = fs::read_to_string(&path).map_err(|e| {
                format!(
                    "cannot read benchmark source file '{}': {e}",
                    path.display()
                )
            })?;
            if !content.trim().is_empty() {
                docs.push(content);
            }
        }
        if docs.is_empty() {
            return Err(format!(
                "no markdown benchmark files found under '{}'",
                dir.display()
            ));
        }
        Ok(docs)
    }

    fn chunk_text_by_paragraph(text: &str, target_bytes: usize) -> Vec<String> {
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        let mut out = Vec::new();
        let mut current = String::new();
        for paragraph in paragraphs {
            if paragraph.len() > target_bytes {
                if !current.trim().is_empty() {
                    out.push(current.trim().to_string());
                    current.clear();
                }
                let mut start = 0usize;
                while start < paragraph.len() {
                    let mut end = (start + target_bytes).min(paragraph.len());
                    while end > start && !paragraph.is_char_boundary(end) {
                        end -= 1;
                    }
                    out.push(paragraph[start..end].trim().to_string());
                    start = end;
                }
                continue;
            }
            let joiner = if current.is_empty() { 0 } else { 2 };
            if !current.is_empty() && current.len() + joiner + paragraph.len() > target_bytes {
                out.push(current.trim().to_string());
                current.clear();
            }
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(paragraph);
        }
        if !current.trim().is_empty() {
            out.push(current.trim().to_string());
        }
        out
    }

    fn benchmark_source_chunks(target_bytes: usize) -> Result<Vec<String>, String> {
        let source_dir = env::var("TOKENIZER_BENCH_SOURCE_DIR")
            .map_err(|_| "TOKENIZER_BENCH_SOURCE_DIR is not set".to_string())?;
        let docs = load_markdown_documents_from_dir(Path::new(&source_dir))?;
        let mut chunks = Vec::new();
        for doc in docs {
            chunks.extend(chunk_text_by_paragraph(&doc, target_bytes));
        }
        if chunks.is_empty() {
            return Err(format!(
                "no benchmark chunks produced from '{}' at target {} bytes",
                source_dir, target_bytes
            ));
        }
        Ok(chunks)
    }

    fn top_adjacent_pairs(chars: &[char], limit: usize) -> Vec<(String, String)> {
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for window in chars.windows(2) {
            let left = window[0].to_string();
            let right = window[1].to_string();
            *counts.entry((left, right)).or_default() += 1;
        }
        let mut pairs: Vec<_> = counts.into_iter().collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs
            .into_iter()
            .take(limit)
            .map(|(pair, _)| pair)
            .collect()
    }

    fn build_gpt2_synthetic_tokenizer(corpus: &str) -> Tokenizer {
        let encoded = text_to_tiktoken(corpus);
        let encoded_chars: Vec<char> = encoded.chars().collect();
        let mut vocab = Vec::new();
        let mut seen = HashSet::new();
        for ch in &encoded_chars {
            let s = ch.to_string();
            if seen.insert(s.clone()) {
                vocab.push(s);
            }
        }

        let merges = top_adjacent_pairs(&encoded_chars, 512);
        for (left, right) in &merges {
            let merged = format!("{left}{right}");
            if seen.insert(merged.clone()) {
                vocab.push(merged);
            }
        }

        let mut tokenizer = Tokenizer {
            vocab,
            vocab_scores: vec![0.0; seen.len()],
            vocab_size: seen.len(),
            max_token_length: 8,
            bos_token: -1,
            skip_bos_token: false,
            eos_token: -1,
            start_header_token: -1,
            end_header_token: -1,
            eot_token: -1,
            pre_tokenizer: TokenizerPreType::Gpt2,
            use_sentencepiece: false,
            token_to_id: HashMap::new(),
            merges: merges
                .iter()
                .map(|(left, right)| format!("{left} {right}"))
                .collect(),
            merge_ranks: HashMap::new(),
        };
        tokenizer.prepare_for_encode();
        tokenizer
    }

    fn build_sentencepiece_synthetic_tokenizer(corpus: &str) -> Tokenizer {
        let encoded = text_to_sentencepiece(corpus);
        let encoded_chars: Vec<char> = encoded.chars().collect();
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut seen = HashSet::new();

        for ch in &encoded_chars {
            let s = ch.to_string();
            if seen.insert(s.clone()) {
                vocab.push(s);
                scores.push(0.0);
            }
        }

        let merges = top_adjacent_pairs(&encoded_chars, 512);
        for (idx, (left, right)) in merges.iter().enumerate() {
            let merged = format!("{left}{right}");
            if seen.insert(merged.clone()) {
                vocab.push(merged);
                scores.push(10_000.0 - idx as f32);
            }
        }

        let mut tokenizer = Tokenizer {
            vocab,
            vocab_scores: scores,
            vocab_size: seen.len(),
            max_token_length: 8,
            bos_token: -1,
            skip_bos_token: false,
            eos_token: -1,
            start_header_token: -1,
            end_header_token: -1,
            eot_token: -1,
            pre_tokenizer: TokenizerPreType::Gpt2,
            use_sentencepiece: true,
            token_to_id: HashMap::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
        };
        tokenizer.prepare_for_encode();
        tokenizer
    }

    #[test]
    fn synthetic_reference_matches_optimized_gpt2() {
        let corpus = synthetic_markdown(8);
        let tokenizer = build_gpt2_synthetic_tokenizer(&corpus);
        let sample = synthetic_markdown(3);
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        legacy_encode_prepared(&tokenizer, &sample, &mut expected);
        tokenizer.encode_prepared(&sample, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn synthetic_reference_matches_optimized_sentencepiece() {
        let corpus = synthetic_markdown(8);
        let tokenizer = build_sentencepiece_synthetic_tokenizer(&corpus);
        let sample = synthetic_markdown(3);
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        legacy_encode_prepared(&tokenizer, &sample, &mut expected);
        tokenizer.encode_prepared(&sample, &mut actual);
        assert_eq!(actual, expected);
    }

    #[derive(Clone, Copy)]
    struct BenchSummary {
        min: Duration,
        median: Duration,
        max: Duration,
    }

    fn benchmark_loop<F>(mut f: F) -> BenchSummary
    where
        F: FnMut(),
    {
        const WARMUP_RUNS: usize = 1;
        const MEASURED_RUNS: usize = 7;

        for _ in 0..WARMUP_RUNS {
            f();
        }

        let mut samples = Vec::with_capacity(MEASURED_RUNS);
        for _ in 0..MEASURED_RUNS {
            let t0 = Instant::now();
            f();
            samples.push(t0.elapsed());
        }
        samples.sort_unstable();
        BenchSummary {
            min: samples[0],
            median: samples[samples.len() / 2],
            max: samples[samples.len() - 1],
        }
    }

    fn run_synthetic_benchmark(
        label: &str,
        tokenizer: &Tokenizer,
        docs: &[String],
        bytes_per_doc: usize,
    ) {
        let mut reference_sink = Vec::new();
        let mut optimized_sink = Vec::new();

        let reference = benchmark_loop(|| {
            reference_sink.clear();
            for doc in docs {
                legacy_encode_prepared(tokenizer, black_box(doc), &mut reference_sink);
                black_box(&reference_sink);
            }
        });

        let optimized = benchmark_loop(|| {
            optimized_sink.clear();
            for doc in docs {
                tokenizer.encode_prepared(black_box(doc), &mut optimized_sink);
                black_box(&optimized_sink);
            }
        });

        assert_eq!(reference_sink, optimized_sink);
        eprintln!(
            "TOKENIZER_BENCH mode={label} docs={} bytes_per_doc={} reference_min_us={} reference_median_us={} reference_max_us={} optimized_min_us={} optimized_median_us={} optimized_max_us={} median_speedup_x={:.4}",
            docs.len(),
            bytes_per_doc,
            reference.min.as_micros(),
            reference.median.as_micros(),
            reference.max.as_micros(),
            optimized.min.as_micros(),
            optimized.median.as_micros(),
            optimized.max.as_micros(),
            reference.median.as_secs_f64() / optimized.median.as_secs_f64().max(1e-9),
        );
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_gpt2_speedup() {
        let corpus = synthetic_markdown(12);
        let docs: Vec<String> = (0..8).map(|_| corpus.clone()).collect();
        let tokenizer = build_gpt2_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("gpt2", &tokenizer, &docs, corpus.len());
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_sentencepiece_speedup() {
        let corpus = synthetic_markdown(4);
        let docs: Vec<String> = (0..2).map(|_| corpus.clone()).collect();
        let tokenizer = build_sentencepiece_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("sentencepiece", &tokenizer, &docs, corpus.len());
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_gpt2_chunk_1k_speedup() {
        let corpus = corpus_shaped_markdown(1024, false);
        let docs: Vec<String> = (0..24).map(|_| corpus.clone()).collect();
        let tokenizer = build_gpt2_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("gpt2_chunk_1k", &tokenizer, &docs, corpus.len());
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_gpt2_chunk_2k_code_speedup() {
        let corpus = corpus_shaped_markdown(2048, true);
        let docs: Vec<String> = (0..16).map(|_| corpus.clone()).collect();
        let tokenizer = build_gpt2_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("gpt2_chunk_2k_code", &tokenizer, &docs, corpus.len());
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_sentencepiece_chunk_1k_speedup() {
        let corpus = corpus_shaped_markdown(1024, false);
        let docs: Vec<String> = (0..6).map(|_| corpus.clone()).collect();
        let tokenizer = build_sentencepiece_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("sentencepiece_chunk_1k", &tokenizer, &docs, corpus.len());
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_sentencepiece_chunk_2k_code_speedup() {
        let corpus = corpus_shaped_markdown(2048, true);
        let docs: Vec<String> = (0..4).map(|_| corpus.clone()).collect();
        let tokenizer = build_sentencepiece_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark(
            "sentencepiece_chunk_2k_code",
            &tokenizer,
            &docs,
            corpus.len(),
        );
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_gpt2_source_chunk_1200_speedup() {
        let docs = match benchmark_source_chunks(1200) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("skipping source benchmark: {msg}");
                return;
            }
        };
        let corpus = docs.join("\n\n");
        let tokenizer = build_gpt2_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("gpt2_source_chunk_1200", &tokenizer, &docs, 1200);
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_gpt2_source_chunk_1800_speedup() {
        let docs = match benchmark_source_chunks(1800) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("skipping source benchmark: {msg}");
                return;
            }
        };
        let corpus = docs.join("\n\n");
        let tokenizer = build_gpt2_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("gpt2_source_chunk_1800", &tokenizer, &docs, 1800);
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_sentencepiece_source_chunk_1200_speedup() {
        let docs = match benchmark_source_chunks(1200) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("skipping source benchmark: {msg}");
                return;
            }
        };
        let corpus = docs.join("\n\n");
        let tokenizer = build_sentencepiece_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("sentencepiece_source_chunk_1200", &tokenizer, &docs, 1200);
    }

    #[test]
    #[ignore]
    fn synthetic_benchmark_reports_sentencepiece_source_chunk_1800_speedup() {
        let docs = match benchmark_source_chunks(1800) {
            Ok(v) => v,
            Err(msg) => {
                eprintln!("skipping source benchmark: {msg}");
                return;
            }
        };
        let corpus = docs.join("\n\n");
        let tokenizer = build_sentencepiece_synthetic_tokenizer(&corpus);
        run_synthetic_benchmark("sentencepiece_source_chunk_1800", &tokenizer, &docs, 1800);
    }
}
