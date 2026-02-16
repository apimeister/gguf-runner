use crate::engine::io::get_gguf_int_from_map;
use crate::engine::types::{
    Config, GGUFFile, GgufValue, Tokenizer, LLAMA3_BOS_TOKEN, LLAMA3_END_HEADER, LLAMA3_EOS_TOKEN,
    LLAMA3_EOT, LLAMA3_START_HEADER,
};
use std::collections::HashMap;

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

    String::from_utf8_lossy(&out).to_string()
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
            if let Some((c1, _)) = next_char(text, j) {
                if c1.is_whitespace() {
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

impl Tokenizer {
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

    pub(crate) fn bpe_encode(&mut self, text: &str, tokens: &mut Vec<i32>) {
        tokens.clear();
        if text.is_empty() {
            return;
        }

        if self.use_sentencepiece {
            let encoded_text = text_to_sentencepiece(text);
            self.build_token_lookup();

            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let s = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&s) {
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
                    let left = &self.vocab[work[i] as usize];
                    let right = &self.vocab[work[i + 1] as usize];
                    let merged = format!("{left}{right}");
                    if let Some(&id) = self.token_to_id.get(&merged) {
                        let score = self.vocab_scores.get(id as usize).copied().unwrap_or(0.0);
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

        self.build_token_lookup();
        self.build_merge_ranks();

        let pieces = split_gpt2_pieces(text);
        for piece in pieces {
            let encoded_text = text_to_tiktoken(&piece);
            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let s = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&s) {
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
                    let left = &self.vocab[work[i] as usize];
                    let right = &self.vocab[work[i + 1] as usize];
                    let pair = format!("{left} {right}");
                    let merged = format!("{left}{right}");
                    if let Some(&rank) = self.merge_ranks.get(&pair) {
                        if let Some(&id) = self.token_to_id.get(&merged) {
                            if rank < best_rank {
                                best_rank = rank;
                                best_id = id;
                                best_pos = i;
                            }
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
        }

        return;
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
}

pub(crate) fn init_tokenizer_from_gguf(
    gguf: &GGUFFile,
    config: &mut Config,
    debug_mode: bool,
) -> Result<Tokenizer, String> {
    if gguf.vocab_tokens.is_empty() {
        return Err("no vocabulary found in GGUF file".to_string());
    }

    let mut tokenizer = Tokenizer::default();
    tokenizer.bos_token = match gguf.kv.get("tokenizer.ggml.bos_token_id") {
        Some(GgufValue::UInt(v)) => *v as i32,
        Some(GgufValue::Int(v)) => *v as i32,
        _ => -1,
    };
    tokenizer.eos_token = get_gguf_int_from_map(
        &gguf.kv,
        "tokenizer.ggml.eos_token_id",
        LLAMA3_EOS_TOKEN as i64,
    ) as i32;
    tokenizer.start_header_token = LLAMA3_START_HEADER;
    tokenizer.end_header_token = LLAMA3_END_HEADER;
    tokenizer.eot_token = LLAMA3_EOT;

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
        if config.is_qwen2 || config.is_qwen3moe || config.is_qwen3next || config.is_deepseek2 {
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
            "Using vocabulary from GGUF file ({} tokens)",
            tokenizer.vocab_size
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
