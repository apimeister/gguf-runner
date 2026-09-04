use crate::engine::types::{
    GGUF_MAGIC, GGUF_TYPE_ARRAY, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64,
    GGUF_TYPE_INT8, GGUF_TYPE_INT16, GGUF_TYPE_INT32, GGUF_TYPE_INT64, GGUF_TYPE_STRING,
    GGUF_TYPE_UINT8, GGUF_TYPE_UINT16, GGUF_TYPE_UINT32, GGUF_TYPE_UINT64, GGUFFile, GgmlType,
    GgufValue, Gguftensor, MappedFile,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;

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

/// Convert F32 to F16 with round-to-nearest, ties-to-even. NaNs retain a
/// non-zero payload so the conversion cannot turn them into infinities.
#[inline]
pub(crate) fn fp32_to_fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ff_ff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mantissa >> 13) as u16) | 0x0200;
        return sign | 0x7c00 | payload;
    }

    let unbiased_exponent = exponent - 127;
    if unbiased_exponent > 15 {
        return sign | 0x7c00;
    }
    if unbiased_exponent < -25 {
        return sign;
    }

    if unbiased_exponent < -14 {
        let significand = mantissa | 0x80_00_00;
        let shift = (-1 - unbiased_exponent) as u32;
        let mut rounded = significand >> shift;
        let remainder_mask = (1u32 << shift) - 1;
        let remainder = significand & remainder_mask;
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }

    let half_exponent = ((unbiased_exponent + 15) as u32) << 10;
    let mut rounded_mantissa = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded_mantissa & 1 != 0) {
        rounded_mantissa += 1;
    }
    sign | (half_exponent + rounded_mantissa) as u16
}

/// Convert F32 to BF16 with round-to-nearest, ties-to-even. NaNs retain a
/// non-zero payload so the conversion cannot turn them into infinities.
#[inline]
pub(crate) fn fp32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        ((bits >> 16) | 64) as u16
    } else {
        ((bits + (0x7fff + ((bits >> 16) & 1))) >> 16) as u16
    }
}

fn read_exact_array<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    Ok(read_exact_array::<1>(r)?[0])
}

fn read_i8(r: &mut impl Read) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_exact_array::<2>(r)?))
}

fn read_i16(r: &mut impl Read) -> io::Result<i16> {
    Ok(i16::from_le_bytes(read_exact_array::<2>(r)?))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_bool(r: &mut impl Read) -> io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_gguf_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    match String::from_utf8(buf) {
        Ok(s) => Ok(s),
        Err(e) => Ok(String::from_utf8_lossy(e.as_bytes()).into_owned()),
    }
}

fn skip_gguf_value(r: &mut impl Read, value_type: u32) -> io::Result<()> {
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

fn read_gguf_scalar(r: &mut impl Read, value_type: u32) -> io::Result<GgufValue> {
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

fn read_gguf_integer_array_value(r: &mut impl Read, value_type: u32) -> io::Result<i64> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(read_u8(r)? as i64),
        GGUF_TYPE_INT8 => Ok(read_i8(r)? as i64),
        GGUF_TYPE_UINT16 => Ok(read_u16(r)? as i64),
        GGUF_TYPE_INT16 => Ok(read_i16(r)? as i64),
        GGUF_TYPE_UINT32 => Ok(read_u32(r)? as i64),
        GGUF_TYPE_INT32 => Ok(read_i32(r)? as i64),
        GGUF_TYPE_UINT64 => Ok(read_u64(r)?.min(i64::MAX as u64) as i64),
        GGUF_TYPE_INT64 => Ok(read_i64(r)?),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported integer gguf array type: {value_type}"),
        )),
    }
}

/// Core GGUF parse logic shared by file-backed and bytes-backed loading.
/// `reader` is used for the sequential header; `mapped` holds the full file
/// bytes for tensor data access (already created by the caller).
fn parse_gguf_inner<R: Read + Seek>(
    reader: &mut R,
    debug_mode: bool,
    mapped: MappedFile,
) -> Result<GGUFFile, String> {
    let magic = read_u32(reader).map_err(|e| format!("failed to read magic number: {e}"))?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "invalid GGUF magic: expected 0x{GGUF_MAGIC:X}, got 0x{magic:X}"
        ));
    }

    let version = read_u32(reader).map_err(|e| format!("failed to read version: {e}"))?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version: {version}"));
    }

    let n_tensors = read_u64(reader).map_err(|e| format!("failed to read n_tensors: {e}"))?;
    let n_kv = read_u64(reader).map_err(|e| format!("failed to read n_kv: {e}"))?;

    if debug_mode {
        eprintln!("GGUF version: {version}, tensors: {n_tensors}, kv pairs: {n_kv}");
    }

    let mut kv: HashMap<String, GgufValue> = HashMap::new();
    let mut vocab_tokens: Vec<String> = Vec::new();
    let mut vocab_scores: Vec<f32> = Vec::new();
    let mut vocab_merges: Vec<String> = Vec::new();

    for _ in 0..n_kv {
        let key = read_gguf_string(reader).map_err(|e| format!("failed to read key: {e}"))?;
        let value_type = read_u32(reader).map_err(|e| format!("failed to read value type: {e}"))?;

        if value_type == GGUF_TYPE_ARRAY {
            let arr_type =
                read_u32(reader).map_err(|e| format!("failed to read array type: {e}"))?;
            let arr_len = read_u64(reader).map_err(|e| format!("failed to read array len: {e}"))?;

            if key == "tokenizer.ggml.tokens" && arr_type == GGUF_TYPE_STRING {
                vocab_tokens.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let tok = read_gguf_string(reader)
                        .map_err(|e| format!("failed to read token: {e}"))?;
                    vocab_tokens.push(tok);
                }
            } else if key == "tokenizer.ggml.scores" && arr_type == GGUF_TYPE_FLOAT32 {
                vocab_scores.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let score =
                        read_f32(reader).map_err(|e| format!("failed to read score: {e}"))?;
                    vocab_scores.push(score);
                }
            } else if key == "tokenizer.ggml.merges" && arr_type == GGUF_TYPE_STRING {
                vocab_merges.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let merge = read_gguf_string(reader)
                        .map_err(|e| format!("failed to read merge: {e}"))?;
                    vocab_merges.push(merge);
                }
            } else if arr_type == GGUF_TYPE_FLOAT32 {
                let mut values = Vec::with_capacity(arr_len as usize);
                for _ in 0..arr_len {
                    let value = read_f32(reader).map_err(|e| {
                        format!("failed to read float32 array value for key {key}: {e}")
                    })?;
                    values.push(value);
                }
                kv.insert(key, GgufValue::F32Array(values));
            } else if arr_type == GGUF_TYPE_FLOAT64 {
                let mut values = Vec::with_capacity(arr_len as usize);
                for _ in 0..arr_len {
                    let value = read_f64(reader).map_err(|e| {
                        format!("failed to read float64 array value for key {key}: {e}")
                    })?;
                    values.push(value as f32);
                }
                kv.insert(key, GgufValue::F32Array(values));
            } else if matches!(
                arr_type,
                GGUF_TYPE_UINT8
                    | GGUF_TYPE_INT8
                    | GGUF_TYPE_UINT16
                    | GGUF_TYPE_INT16
                    | GGUF_TYPE_UINT32
                    | GGUF_TYPE_INT32
                    | GGUF_TYPE_UINT64
                    | GGUF_TYPE_INT64
            ) {
                let mut values = Vec::with_capacity(arr_len as usize);
                for _ in 0..arr_len {
                    let value = read_gguf_integer_array_value(reader, arr_type).map_err(|e| {
                        format!("failed to read integer array value for key {key}: {e}")
                    })?;
                    values.push(value);
                }
                kv.insert(key, GgufValue::I64Array(values));
            } else {
                for _ in 0..arr_len {
                    skip_gguf_value(reader, arr_type)
                        .map_err(|e| format!("failed to skip array value for key {key}: {e}"))?;
                }
            }
        } else {
            let value = read_gguf_scalar(reader, value_type)
                .map_err(|e| format!("failed to read scalar for key {key}: {e}"))?;
            kv.insert(key, value);
        }
    }

    let mut tensors: Vec<Gguftensor> = Vec::with_capacity(n_tensors as usize);

    for _ in 0..n_tensors {
        let name =
            read_gguf_string(reader).map_err(|e| format!("failed to read tensor name: {e}"))?;
        let n_dims = read_u32(reader).map_err(|e| format!("failed to read n_dims: {e}"))?;
        if n_dims > 4 {
            return Err(format!("tensor {name} has unsupported n_dims={n_dims}"));
        }

        let mut ne = [1u64; 4];
        for n in ne.iter_mut().take(n_dims as usize) {
            *n = read_u64(reader).map_err(|e| format!("failed reading tensor dims: {e}"))?;
        }

        let ttype = GgmlType::from_u32(
            read_u32(reader).map_err(|e| format!("failed reading tensor type: {e}"))?,
        );
        let offset = read_u64(reader).map_err(|e| format!("failed reading tensor offset: {e}"))?;

        tensors.push(Gguftensor {
            name,
            n_dims,
            ne,
            ttype,
            offset,
            data_offset: 0,
        });
    }

    let header_end = reader
        .stream_position()
        .map_err(|e| format!("failed to query header end: {e}"))?;

    let alignment = get_gguf_int_from_map(&kv, "general.alignment", 32) as u64;
    let tensor_data_offset = header_end.div_ceil(alignment) * alignment;

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
    })
}

fn parse_gguf_file_local(filename: &str, debug_mode: bool) -> Result<GGUFFile, String> {
    let mut file = File::open(filename).map_err(|e| format!("cannot open file {filename}: {e}"))?;
    let mapped = MappedFile::map(&file).map_err(|e| format!("mmap failed: {e}"))?;
    parse_gguf_inner(&mut file, debug_mode, mapped)
}

/// Parse a GGUF model from a static byte slice (e.g. embedded via `include_bytes!`).
pub(crate) fn parse_gguf_from_bytes(
    data: &'static [u8],
    debug_mode: bool,
) -> Result<GGUFFile, String> {
    let mapped = MappedFile::from_static(data).map_err(|e| format!("static map failed: {e}"))?;
    let mut cursor = std::io::Cursor::new(data);
    parse_gguf_inner(&mut cursor, debug_mode, mapped)
}

pub(crate) fn parse_gguf_file(filename: &str, debug_mode: bool) -> Result<GGUFFile, String> {
    let model_path = Path::new(filename);
    if !model_path.exists() {
        return Err(format!("model file not found: {filename}"));
    }

    let gguf = parse_gguf_file_local(filename, debug_mode)?;
    if debug_mode {
        eprintln!("Using local model file: {filename}");
    }
    Ok(gguf)
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

pub(crate) fn get_gguf_f32_array_from_map<'a>(
    kv: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Option<&'a [f32]> {
    match kv.get(key) {
        Some(GgufValue::F32Array(values)) => Some(values.as_slice()),
        _ => None,
    }
}

pub(crate) fn get_gguf_i64_array_from_map<'a>(
    kv: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Option<&'a [i64]> {
    match kv.get(key) {
        Some(GgufValue::I64Array(values)) => Some(values.as_slice()),
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

pub(crate) fn find_gguf_tensor_names_with_any_prefix(
    gguf: &GGUFFile,
    prefixes: &[&str],
) -> Vec<String> {
    gguf.tensors
        .iter()
        .filter(|tensor| {
            prefixes
                .iter()
                .any(|prefix| tensor.name.starts_with(prefix))
        })
        .map(|tensor| tensor.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::fp32_to_fp16;

    #[test]
    fn fp32_to_fp16_matches_ieee_ties_and_subnormal_boundaries() {
        assert_eq!(fp32_to_fp16(0.0), 0x0000);
        assert_eq!(fp32_to_fp16(-0.0), 0x8000);
        assert_eq!(fp32_to_fp16(1.0), 0x3c00);
        assert_eq!(fp32_to_fp16(-2.0), 0xc000);
        assert_eq!(fp32_to_fp16(65_504.0), 0x7bff);
        assert_eq!(fp32_to_fp16(f32::INFINITY), 0x7c00);
        assert_eq!(fp32_to_fp16(f32::from_bits(0x3f80_1000)), 0x3c00);
        assert_eq!(fp32_to_fp16(f32::from_bits(0x3f80_3000)), 0x3c02);
        assert_eq!(fp32_to_fp16(f32::from_bits(0x3300_0000)), 0x0000);
        assert_eq!(fp32_to_fp16(f32::from_bits(0x3300_0001)), 0x0001);
        assert_eq!(fp32_to_fp16(f32::from_bits(0x3380_0000)), 0x0001);
    }
}
