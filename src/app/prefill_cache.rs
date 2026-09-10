//! Prefill cache: a serialized snapshot of the inference state (KV-cache rows
//! + SSM states) after prefilling a static prompt prefix.
//!
//! Rendered once (at a consumer's build time or via `--render-prefill-cache`)
//! and injected at generation start when the encoded prompt begins with the
//! cached token sequence — replacing the prefix prefill with a memcpy. The
//! stored token IDs make the mechanism self-guarding: any drift in prompt
//! text, template, or tokenizer produces a prefix mismatch and the caller
//! falls back to a cold prefill.
//!
//! Format (little-endian, versioned):
//! `magic "GPFC" | version u16 | kv_format u8 (0=q8,1=turbo) | q8_block u8 |
//!  n_layers u32 | kv_dim u32 | n_kv_heads u32 | k u32 | seq_len_at_render u32 |
//!  model_len u64 | model_head_fnv u64 | tokens i32×k |
//!  sections (each: u64 byte len + raw bytes)`.
//! Sections in order — Q8: key rows, value rows, key scales, value scales;
//! Turbo: key base, value base, key sign, value sign, key scales, value
//! scales, key residual norms, value residual norms; then always:
//! ssm_conv_state, ssm_state (full arrays; empty for non-SSM models).

use crate::engine::types::{Config, KvCacheFormat, RunState};

/// Mirror of the private helpers in `engine::runtime::inference` — semantics
/// must match (`GGUF_Q8_BLOCK_SCALES` gating and the Q8 block size) so the
/// snapshot layout agrees with what the KV-write path produced.
const Q4_BLOCK_SIZE: usize = 32;

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(crate) const PREFILL_CACHE_MAGIC: &[u8; 4] = b"GPFC";
pub(crate) const PREFILL_CACHE_VERSION: u16 = 1;

/// Parsed, validated prefill cache ready for injection.
pub(crate) struct PrefixCache {
    pub(crate) tokens: Vec<i32>,
    kv_format: KvCacheFormat,
    q8_block_scales: bool,
    n_layers: usize,
    kv_dim: usize,
    n_kv_heads: usize,
    pub(crate) model_len: u64,
    pub(crate) model_head_fnv: u64,
    sections: Vec<Vec<u8>>,
}

/// FNV-1a over `bytes` — cheap model-identity fingerprint (the token-prefix
/// match is the real correctness guard; this catches gross mismatches early).
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub(crate) fn model_fingerprint(model_bytes: &[u8]) -> (u64, u64) {
    let head = &model_bytes[..model_bytes.len().min(1 << 20)];
    (model_bytes.len() as u64, fnv1a(head))
}

struct Writer(Vec<u8>);
impl Writer {
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn section(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.0.extend_from_slice(bytes);
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.at + n > self.b.len() {
            return Err("prefill cache truncated".to_string());
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn section(&mut self) -> Result<Vec<u8>, String> {
        let len = self.u64()? as usize;
        Ok(self.take(len)?.to_vec())
    }
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(b: &[u8]) -> Result<Vec<f32>, String> {
    if !b.len().is_multiple_of(4) {
        return Err("prefill cache: bad f32 section length".to_string());
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn i8s_to_bytes(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

/// Copy `k` leading rows of each layer out of a flat `[n_layers*seq_len]`-row
/// array with `elem_per_row` elements (or packed bytes) per row.
fn gather_rows<T: Copy>(
    src: &[T],
    per_row: usize,
    seq_len: usize,
    n_layers: usize,
    k: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(n_layers * k * per_row);
    for l in 0..n_layers {
        let start = l * seq_len * per_row;
        out.extend_from_slice(&src[start..start + k * per_row]);
    }
    out
}

/// Inverse of [`gather_rows`]: scatter per-layer leading rows into `dst`
/// using the destination's own `seq_len` stride.
fn scatter_rows<T: Copy>(
    dst: &mut [T],
    packed: &[T],
    per_row: usize,
    seq_len: usize,
    n_layers: usize,
    k: usize,
) -> Result<(), String> {
    if packed.len() != n_layers * k * per_row {
        return Err("prefill cache: section size mismatch".to_string());
    }
    for l in 0..n_layers {
        let dst_start = l * seq_len * per_row;
        let src_start = l * k * per_row;
        dst[dst_start..dst_start + k * per_row]
            .copy_from_slice(&packed[src_start..src_start + k * per_row]);
    }
    Ok(())
}

/// Serialize the first `k` positions of `state` (all layers) into a blob.
pub(crate) fn snapshot(
    p: &Config,
    s: &RunState,
    tokens: &[i32],
    model_len: u64,
    model_head_fnv: u64,
) -> Result<Vec<u8>, String> {
    let k = tokens.len();
    let n_layers = p.n_layers;
    let seq_len = p.seq_len;
    let kv_dim = s.kv_dim;
    let n_kv_heads = p.n_kv_heads.max(1);
    if !kv_dim.is_multiple_of(8) {
        return Err("prefill cache requires kv_dim % 8 == 0".to_string());
    }
    let q8_block_scales = env_flag("GGUF_Q8_BLOCK_SCALES");

    let mut w = Writer(Vec::new());
    w.0.extend_from_slice(PREFILL_CACHE_MAGIC);
    w.u16(PREFILL_CACHE_VERSION);
    w.0.push(match s.kv_cache_format {
        KvCacheFormat::Q8 => 0,
        KvCacheFormat::Turbo => 1,
    });
    w.0.push(q8_block_scales as u8);
    w.u32(n_layers as u32);
    w.u32(kv_dim as u32);
    w.u32(n_kv_heads as u32);
    w.u32(k as u32);
    w.u32(seq_len as u32);
    w.u64(model_len);
    w.u64(model_head_fnv);
    for &t in tokens {
        w.0.extend_from_slice(&t.to_le_bytes());
    }

    match s.kv_cache_format {
        KvCacheFormat::Q8 => {
            let scale_per_row = if q8_block_scales {
                (kv_dim / Q4_BLOCK_SIZE).max(1)
            } else {
                1
            };
            w.section(&i8s_to_bytes(&gather_rows(
                &s.key_cache_q8,
                kv_dim,
                seq_len,
                n_layers,
                k,
            )));
            w.section(&i8s_to_bytes(&gather_rows(
                &s.value_cache_q8,
                kv_dim,
                seq_len,
                n_layers,
                k,
            )));
            w.section(&f32s_to_bytes(&gather_rows(
                &s.key_cache_scale,
                scale_per_row,
                seq_len,
                n_layers,
                k,
            )));
            w.section(&f32s_to_bytes(&gather_rows(
                &s.value_cache_scale,
                scale_per_row,
                seq_len,
                n_layers,
                k,
            )));
        }
        KvCacheFormat::Turbo => {
            w.section(&gather_rows(
                &s.key_cache_turbo_base,
                kv_dim / 4,
                seq_len,
                n_layers,
                k,
            ));
            w.section(&gather_rows(
                &s.value_cache_turbo_base,
                kv_dim / 4,
                seq_len,
                n_layers,
                k,
            ));
            w.section(&gather_rows(
                &s.key_cache_turbo_sign,
                kv_dim / 8,
                seq_len,
                n_layers,
                k,
            ));
            w.section(&gather_rows(
                &s.value_cache_turbo_sign,
                kv_dim / 8,
                seq_len,
                n_layers,
                k,
            ));
            w.section(&f32s_to_bytes(&gather_rows(
                &s.key_cache_scale,
                n_kv_heads,
                seq_len,
                n_layers,
                k,
            )));
            w.section(&f32s_to_bytes(&gather_rows(
                &s.value_cache_scale,
                n_kv_heads,
                seq_len,
                n_layers,
                k,
            )));
            w.section(&f32s_to_bytes(&gather_rows(
                &s.key_cache_residual_norm,
                n_kv_heads,
                seq_len,
                n_layers,
                k,
            )));
            w.section(&f32s_to_bytes(&gather_rows(
                &s.value_cache_residual_norm,
                n_kv_heads,
                seq_len,
                n_layers,
                k,
            )));
        }
    }
    w.section(&f32s_to_bytes(&s.ssm_conv_state));
    w.section(&f32s_to_bytes(&s.ssm_state));

    Ok(w.0)
}

impl PrefixCache {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { b: bytes, at: 0 };
        if r.take(4)? != PREFILL_CACHE_MAGIC {
            return Err("prefill cache: bad magic".to_string());
        }
        let version = r.u16()?;
        if version != PREFILL_CACHE_VERSION {
            return Err(format!("prefill cache: unsupported version {version}"));
        }
        let kv_format = match r.take(1)?[0] {
            0 => KvCacheFormat::Q8,
            1 => KvCacheFormat::Turbo,
            other => return Err(format!("prefill cache: bad kv format {other}")),
        };
        let q8_block_scales = r.take(1)?[0] != 0;
        let n_layers = r.u32()? as usize;
        let kv_dim = r.u32()? as usize;
        let n_kv_heads = r.u32()? as usize;
        let k = r.u32()? as usize;
        let _seq_len_at_render = r.u32()?;
        let model_len = r.u64()?;
        let model_head_fnv = r.u64()?;
        let mut tokens = Vec::with_capacity(k);
        for _ in 0..k {
            tokens.push(i32::from_le_bytes(r.take(4)?.try_into().unwrap()));
        }
        let n_sections = match kv_format {
            KvCacheFormat::Q8 => 4,
            KvCacheFormat::Turbo => 8,
        } + 2;
        let mut sections = Vec::with_capacity(n_sections);
        for _ in 0..n_sections {
            sections.push(r.section()?);
        }
        Ok(Self {
            tokens,
            kv_format,
            q8_block_scales,
            n_layers,
            kv_dim,
            n_kv_heads,
            model_len,
            model_head_fnv,
            sections,
        })
    }

    /// Longest usable prefix: the full cached token sequence when the prompt
    /// starts with it and is strictly longer (the remainder goes through the
    /// normal prefill/decode path). `None` = no match, cold prefill.
    pub(crate) fn match_len(&self, prompt_tokens: &[i32]) -> Option<usize> {
        let k = self.tokens.len();
        if k == 0 || prompt_tokens.len() <= k {
            return None;
        }
        if prompt_tokens[..k] == self.tokens[..] {
            Some(k)
        } else {
            None
        }
    }

    /// Copy the cached rows/states into a freshly allocated `state`. The
    /// destination may have a different `seq_len`; rows are re-strided.
    pub(crate) fn inject(&self, p: &Config, s: &mut RunState) -> Result<(), String> {
        let k = self.tokens.len();
        if self.n_layers != p.n_layers
            || self.kv_dim != s.kv_dim
            || self.n_kv_heads != p.n_kv_heads.max(1)
            || self.kv_format != s.kv_cache_format
            || self.q8_block_scales != env_flag("GGUF_Q8_BLOCK_SCALES")
        {
            return Err("prefill cache: model/config mismatch".to_string());
        }
        if k > p.seq_len {
            return Err("prefill cache: prefix longer than context".to_string());
        }
        let (n_layers, seq_len, kv_dim, n_kv_heads) =
            (self.n_layers, p.seq_len, self.kv_dim, self.n_kv_heads);

        match self.kv_format {
            KvCacheFormat::Q8 => {
                let scale_per_row = if self.q8_block_scales {
                    (kv_dim / Q4_BLOCK_SIZE).max(1)
                } else {
                    1
                };
                let kq: Vec<i8> = self.sections[0].iter().map(|&b| b as i8).collect();
                let vq: Vec<i8> = self.sections[1].iter().map(|&b| b as i8).collect();
                scatter_rows(&mut s.key_cache_q8, &kq, kv_dim, seq_len, n_layers, k)?;
                scatter_rows(&mut s.value_cache_q8, &vq, kv_dim, seq_len, n_layers, k)?;
                let ks = bytes_to_f32s(&self.sections[2])?;
                let vs = bytes_to_f32s(&self.sections[3])?;
                scatter_rows(
                    &mut s.key_cache_scale,
                    &ks,
                    scale_per_row,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.value_cache_scale,
                    &vs,
                    scale_per_row,
                    seq_len,
                    n_layers,
                    k,
                )?;
            }
            KvCacheFormat::Turbo => {
                scatter_rows(
                    &mut s.key_cache_turbo_base,
                    &self.sections[0],
                    kv_dim / 4,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.value_cache_turbo_base,
                    &self.sections[1],
                    kv_dim / 4,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.key_cache_turbo_sign,
                    &self.sections[2],
                    kv_dim / 8,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.value_cache_turbo_sign,
                    &self.sections[3],
                    kv_dim / 8,
                    seq_len,
                    n_layers,
                    k,
                )?;
                let ks = bytes_to_f32s(&self.sections[4])?;
                let vs = bytes_to_f32s(&self.sections[5])?;
                let kr = bytes_to_f32s(&self.sections[6])?;
                let vr = bytes_to_f32s(&self.sections[7])?;
                scatter_rows(
                    &mut s.key_cache_scale,
                    &ks,
                    n_kv_heads,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.value_cache_scale,
                    &vs,
                    n_kv_heads,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.key_cache_residual_norm,
                    &kr,
                    n_kv_heads,
                    seq_len,
                    n_layers,
                    k,
                )?;
                scatter_rows(
                    &mut s.value_cache_residual_norm,
                    &vr,
                    n_kv_heads,
                    seq_len,
                    n_layers,
                    k,
                )?;
            }
        }

        let n_kv_sections = match self.kv_format {
            KvCacheFormat::Q8 => 4,
            KvCacheFormat::Turbo => 8,
        };
        let conv = bytes_to_f32s(&self.sections[n_kv_sections])?;
        let ssm = bytes_to_f32s(&self.sections[n_kv_sections + 1])?;
        if conv.len() != s.ssm_conv_state.len() || ssm.len() != s.ssm_state.len() {
            return Err("prefill cache: ssm state size mismatch".to_string());
        }
        s.ssm_conv_state.copy_from_slice(&conv);
        s.ssm_state.copy_from_slice(&ssm);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reader_roundtrip_primitives() {
        let mut w = Writer(Vec::new());
        w.u16(7);
        w.u32(1234);
        w.u64(0xdead_beef);
        w.section(&[1, 2, 3]);
        let mut r = Reader { b: &w.0, at: 0 };
        assert_eq!(r.u16().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 1234);
        assert_eq!(r.u64().unwrap(), 0xdead_beef);
        assert_eq!(r.section().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn gather_scatter_roundtrip_restrides() {
        // 2 layers, source seq_len 4, dest seq_len 6, k=2 rows of 3 elems.
        let mut src = vec![0i8; 2 * 4 * 3];
        for (i, v) in src.iter_mut().enumerate() {
            *v = i as i8;
        }
        let packed = gather_rows(&src, 3, 4, 2, 2);
        assert_eq!(packed.len(), 2 * 2 * 3);
        let mut dst = vec![0i8; 2 * 6 * 3];
        scatter_rows(&mut dst, &packed, 3, 6, 2, 2).unwrap();
        // layer 0 rows 0..2 match source layer 0 rows 0..2
        assert_eq!(&dst[0..6], &src[0..6]);
        // layer 1 begins at 6*3 in dst, 4*3 in src
        assert_eq!(&dst[18..24], &src[12..18]);
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a(b"everlock"), fnv1a(b"everlock"));
        assert_ne!(fnv1a(b"everlock"), fnv1a(b"everloch"));
    }
}
