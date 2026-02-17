use crate::engine::types::{
    GGUFFile, GgmlType, GgufValue, Gguftensor, LazyModelLoader, MappedFile, GGUF_MAGIC,
    GGUF_TYPE_ARRAY, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64, GGUF_TYPE_INT16,
    GGUF_TYPE_INT32, GGUF_TYPE_INT64, GGUF_TYPE_INT8, GGUF_TYPE_STRING, GGUF_TYPE_UINT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_UINT64, GGUF_TYPE_UINT8, LAZY_BOOTSTRAP_MAX_BYTES,
    LAZY_BOOTSTRAP_START_BYTES, LAZY_MODEL_LOADER,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;
use std::sync::Arc;

pub(crate) fn read_u16_le(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off + 1]])
}

#[inline]
pub(crate) fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline]
pub(crate) fn read_f32_le(data: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline]
pub(crate) fn fp16_to_fp32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1f) as i32;
    let mut mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            exp += 1;
            mant &= !0x0400;
            let exp32 = (exp + (127 - 15)) as u32;
            sign | (exp32 << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let exp32 = (exp + (127 - 15)) as u32;
        sign | (exp32 << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

#[inline]
pub(crate) fn bf16_to_fp32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn read_exact_array<const N: usize>(r: &mut File) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_u8(r: &mut File) -> io::Result<u8> {
    Ok(read_exact_array::<1>(r)?[0])
}

fn read_i8(r: &mut File) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut File) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_exact_array::<2>(r)?))
}

fn read_i16(r: &mut File) -> io::Result<i16> {
    Ok(i16::from_le_bytes(read_exact_array::<2>(r)?))
}

fn read_u32(r: &mut File) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_i32(r: &mut File) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_u64(r: &mut File) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_i64(r: &mut File) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_f32(r: &mut File) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_f64(r: &mut File) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_bool(r: &mut File) -> io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_gguf_string(r: &mut File) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    match String::from_utf8(buf) {
        Ok(s) => Ok(s),
        Err(e) => Ok(String::from_utf8_lossy(e.as_bytes()).into_owned()),
    }
}

fn skip_gguf_value(r: &mut File, value_type: u32) -> io::Result<()> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => {
            let _ = read_u8(r)?;
        }
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => {
            let _ = read_u16(r)?;
        }
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => {
            let _ = read_u32(r)?;
        }
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => {
            let _ = read_u64(r)?;
        }
        GGUF_TYPE_STRING => {
            let _ = read_gguf_string(r)?;
        }
        GGUF_TYPE_ARRAY => {
            let arr_type = read_u32(r)?;
            let arr_len = read_u64(r)?;
            for _ in 0..arr_len {
                skip_gguf_value(r, arr_type)?;
            }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported gguf value type: {value_type}"),
            ));
        }
    }
    Ok(())
}

fn read_gguf_scalar(r: &mut File, value_type: u32) -> io::Result<GgufValue> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::UInt(read_u8(r)? as u64)),
        GGUF_TYPE_INT8 => Ok(GgufValue::Int(read_i8(r)? as i64)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::UInt(read_u16(r)? as u64)),
        GGUF_TYPE_INT16 => Ok(GgufValue::Int(read_i16(r)? as i64)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::UInt(read_u32(r)? as u64)),
        GGUF_TYPE_INT32 => Ok(GgufValue::Int(read_i32(r)? as i64)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::UInt(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::Int(read_i64(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_bool(r)?)),
        GGUF_TYPE_STRING => Ok(GgufValue::Str(read_gguf_string(r)?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported scalar gguf type: {value_type}"),
        )),
    }
}

fn parse_gguf_file_local(filename: &str, debug_mode: bool) -> Result<GGUFFile, String> {
    let mut file = File::open(filename).map_err(|e| format!("cannot open file {filename}: {e}"))?;

    let magic = read_u32(&mut file).map_err(|e| format!("failed to read magic number: {e}"))?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "invalid GGUF magic: expected 0x{GGUF_MAGIC:X}, got 0x{magic:X}"
        ));
    }

    let version = read_u32(&mut file).map_err(|e| format!("failed to read version: {e}"))?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version: {version}"));
    }

    let n_tensors = read_u64(&mut file).map_err(|e| format!("failed to read n_tensors: {e}"))?;
    let n_kv = read_u64(&mut file).map_err(|e| format!("failed to read n_kv: {e}"))?;

    if debug_mode {
        eprintln!("GGUF version: {version}, tensors: {n_tensors}, kv pairs: {n_kv}");
    }

    let mut kv: HashMap<String, GgufValue> = HashMap::new();
    let mut vocab_tokens: Vec<String> = Vec::new();
    let mut vocab_scores: Vec<f32> = Vec::new();
    let mut vocab_merges: Vec<String> = Vec::new();

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut file).map_err(|e| format!("failed to read key: {e}"))?;
        let value_type =
            read_u32(&mut file).map_err(|e| format!("failed to read value type: {e}"))?;

        if value_type == GGUF_TYPE_ARRAY {
            let arr_type =
                read_u32(&mut file).map_err(|e| format!("failed to read array type: {e}"))?;
            let arr_len =
                read_u64(&mut file).map_err(|e| format!("failed to read array len: {e}"))?;

            if key == "tokenizer.ggml.tokens" && arr_type == GGUF_TYPE_STRING {
                vocab_tokens.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let tok = read_gguf_string(&mut file)
                        .map_err(|e| format!("failed to read token: {e}"))?;
                    vocab_tokens.push(tok);
                }
            } else if key == "tokenizer.ggml.scores" && arr_type == GGUF_TYPE_FLOAT32 {
                vocab_scores.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let score =
                        read_f32(&mut file).map_err(|e| format!("failed to read score: {e}"))?;
                    vocab_scores.push(score);
                }
            } else if key == "tokenizer.ggml.merges" && arr_type == GGUF_TYPE_STRING {
                vocab_merges.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let merge = read_gguf_string(&mut file)
                        .map_err(|e| format!("failed to read merge: {e}"))?;
                    vocab_merges.push(merge);
                }
            } else {
                for _ in 0..arr_len {
                    skip_gguf_value(&mut file, arr_type)
                        .map_err(|e| format!("failed to skip array value for key {key}: {e}"))?;
                }
            }
        } else {
            let value = read_gguf_scalar(&mut file, value_type)
                .map_err(|e| format!("failed to read scalar for key {key}: {e}"))?;
            kv.insert(key, value);
        }
    }

    let mut tensors: Vec<Gguftensor> = Vec::with_capacity(n_tensors as usize);

    for _ in 0..n_tensors {
        let name =
            read_gguf_string(&mut file).map_err(|e| format!("failed to read tensor name: {e}"))?;
        let n_dims = read_u32(&mut file).map_err(|e| format!("failed to read n_dims: {e}"))?;
        if n_dims > 4 {
            return Err(format!("tensor {name} has unsupported n_dims={n_dims}"));
        }

        let mut ne = [1u64; 4];
        for i in 0..n_dims as usize {
            ne[i] = read_u64(&mut file).map_err(|e| format!("failed reading tensor dims: {e}"))?;
        }

        let ttype = GgmlType::from_u32(
            read_u32(&mut file).map_err(|e| format!("failed reading tensor type: {e}"))?,
        );
        let offset =
            read_u64(&mut file).map_err(|e| format!("failed reading tensor offset: {e}"))?;

        tensors.push(Gguftensor {
            name,
            n_dims,
            ne,
            ttype,
            offset,
            data_offset: 0,
        });
    }

    let header_end = file
        .stream_position()
        .map_err(|e| format!("failed to query header end: {e}"))?;

    let alignment = get_gguf_int_from_map(&kv, "general.alignment", 32) as u64;
    let tensor_data_offset = ((header_end + alignment - 1) / alignment) * alignment;

    let mapped = MappedFile::map(&file).map_err(|e| format!("mmap failed: {e}"))?;
    let mapped_len = mapped.len;

    let mut tensor_lookup = HashMap::new();
    for (idx, t) in tensors.iter_mut().enumerate() {
        let abs_off = tensor_data_offset as usize + t.offset as usize;
        if abs_off >= mapped_len {
            return Err(format!("tensor {} points outside mapped file", t.name));
        }
        t.data_offset = abs_off;
        tensor_lookup.insert(t.name.clone(), idx);
    }

    if !vocab_tokens.is_empty() && vocab_scores.is_empty() {
        vocab_scores = vec![0.0; vocab_tokens.len()];
    }

    Ok(GGUFFile {
        version,
        n_tensors,
        n_kv,
        kv,
        tensors,
        tensor_lookup,
        tensor_data_start: tensor_data_offset as usize,
        vocab_tokens,
        vocab_scores,
        vocab_merges,
        mapped,
        lazy_loader: None,
    })
}

pub(crate) fn parse_gguf_file(
    filename: &str,
    model_url: Option<&str>,
    debug_mode: bool,
) -> Result<GGUFFile, String> {
    let model_path = Path::new(filename);
    let local_exists = model_path.exists();

    if local_exists {
        match parse_gguf_file_local(filename, debug_mode) {
            Ok(gguf) => {
                if debug_mode {
                    eprintln!("Using local model file: {filename}");
                }
                return Ok(gguf);
            }
            Err(e) => {
                if model_url.is_none() {
                    return Err(e);
                }
                eprintln!(
                    "Warning: local model file parse failed ({}). Falling back to remote lazy load.",
                    e
                );
            }
        }
    } else if model_url.is_none() {
        return Err(format!(
            "model file not found: {} (provide -url to lazily fetch it)",
            filename
        ));
    }

    let url = model_url.ok_or_else(|| "missing model url".to_string())?;
    let loader = Arc::new(LazyModelLoader::new(url, filename, debug_mode)?);
    let _ = LAZY_MODEL_LOADER.set(Arc::clone(&loader));

    let mut bootstrap = LAZY_BOOTSTRAP_START_BYTES.min(loader.file_len.max(1));
    let max_bootstrap = LAZY_BOOTSTRAP_MAX_BYTES.min(loader.file_len.max(1));

    loop {
        if debug_mode {
            eprintln!(
                "Lazy model bootstrap: ensuring first {} bytes of {}",
                bootstrap, loader.file_len
            );
        }
        loader.ensure_range(0, bootstrap)?;
        match parse_gguf_file_local(filename, debug_mode) {
            Ok(mut gguf) => {
                gguf.lazy_loader = Some(Arc::clone(&loader));
                loader.start_background_download();
                if debug_mode {
                    eprintln!(
                        "Lazy model mode enabled: url={}, local_cache={}, bytes_bootstrapped={}",
                        url, filename, bootstrap
                    );
                }
                return Ok(gguf);
            }
            Err(e) => {
                if bootstrap >= max_bootstrap {
                    return Err(format!(
                        "failed to parse GGUF metadata after bootstrapping {} bytes: {}",
                        bootstrap, e
                    ));
                }
                let next = (bootstrap.saturating_mul(2)).min(max_bootstrap);
                if next == bootstrap {
                    return Err(format!(
                        "failed to parse GGUF metadata at {} bytes: {}",
                        bootstrap, e
                    ));
                }
                if debug_mode {
                    eprintln!(
                        "Lazy model bootstrap parse retry: {} -> {} bytes ({})",
                        bootstrap, next, e
                    );
                }
                bootstrap = next;
            }
        }
    }
}

pub(crate) fn get_gguf_int_from_map(
    kv: &HashMap<String, GgufValue>,
    key: &str,
    default_val: i64,
) -> i64 {
    match kv.get(key) {
        Some(GgufValue::UInt(v)) => *v as i64,
        Some(GgufValue::Int(v)) => *v,
        _ => default_val,
    }
}

pub(crate) fn get_gguf_float_from_map(
    kv: &HashMap<String, GgufValue>,
    key: &str,
    default_val: f32,
) -> f32 {
    match kv.get(key) {
        Some(GgufValue::F32(v)) => *v,
        Some(GgufValue::F64(v)) => *v as f32,
        _ => default_val,
    }
}

pub(crate) fn get_gguf_string_from_map<'a>(
    kv: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Option<&'a str> {
    match kv.get(key) {
        Some(GgufValue::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

pub(crate) fn get_gguf_bool_from_map(
    kv: &HashMap<String, GgufValue>,
    key: &str,
    default_val: bool,
) -> bool {
    match kv.get(key) {
        Some(GgufValue::Bool(v)) => *v,
        _ => default_val,
    }
}

pub(crate) fn find_gguf_tensor<'a>(gguf: &'a GGUFFile, name: &str) -> Option<&'a Gguftensor> {
    gguf.tensor_lookup
        .get(name)
        .and_then(|idx| gguf.tensors.get(*idx))
}
