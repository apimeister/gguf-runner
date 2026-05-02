#![allow(unsafe_op_in_unsafe_fn)]

/// SIMD encoding helpers for bf16/fp16 activation transport.
///
/// On x86_64 with AVX-2, processes 8 f32 values per iteration using
/// SIMD intrinsics. On other architectures, uses a scalar loop.
///
/// Both paths produce bit-exact IEEE 754 round-to-nearest-even encoding.

#[cfg(target_arch = "x86_64")]
use crate::engine::switches::use_x86_avx2_bf16_enc;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Encode f32 values to bf16 wire bytes.
#[inline]
pub(crate) fn encode_bf16_vector(values: &[f32], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if use_x86_avx2_bf16_enc() && values.len() >= 8 {
        return unsafe { encode_bf16_avx2(values, out) };
    }

    // Scalar fallback: encode each value individually
    for (i, &v) in values.iter().enumerate() {
        let bits = encode_fp32_scalar_to_bf16_bits(v);
        out[i * 2] = bits as u8;
        out[i * 2 + 1] = (bits >> 8) as u8;
    }
}

/// Encode f32 values to fp16 wire bytes.
#[inline]
pub(crate) fn encode_fp16_vector(values: &[f32], out: &mut [u8]) {
    for (i, &v) in values.iter().enumerate() {
        let bits = encode_fp32_scalar_to_fp16_bits(v);
        out[i * 2] = bits as u8;
        out[i * 2 + 1] = (bits >> 8) as u8;
    }
}

// ─── x86_64 AVX-2 SIMD paths ───

/// Encode bf16 activation using AVX-2 intrinsics.
/// bf16 = upper 16 bits of f32 IEEE 754 with round-to-nearest-even.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn encode_bf16_avx2(values: &[f32], out: &mut [u8]) {
    let n = values.len();
    let mut i = 0;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let v32 = _mm256_castps_si256(v);

        // bf16 = upper 16 bits of f32 IEEE 754 bits with round-to-nearest-even.
        // bias = ((bits >> 16) & 1) + 0x7FFF, then result = (bits + bias) >> 16.
        let shifted16 = _mm256_srli_epi32(v32, 16);
        let bias_bit = _mm256_and_si256(shifted16, _mm256_set1_epi32(1));
        let rte_bias = _mm256_add_epi32(bias_bit, _mm256_set1_epi32(0x7FFF));
        let bf16_bits = _mm256_srli_epi32(_mm256_add_epi32(v32, rte_bias), 16);

        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, bf16_bits);
        for j in 0..8 {
            let b = tmp[j] as u16;
            out[(i + j) * 2] = b as u8;
            out[(i + j) * 2 + 1] = (b >> 8) as u8;
        }
        i += 8;
    }

    // Tail: scalar encoding for remaining elements
    while i < n {
        let bits = encode_fp32_scalar_to_bf16_bits(values[i]);
        out[i * 2] = bits as u8;
        out[i * 2 + 1] = (bits >> 8) as u8;
        i += 1;
    }
}

// fp16 encoding is left to the scalar path; a correct SIMD fp16 encoder
// requires exponent rebasing (f32 bias 127 → fp16 bias 15) plus subnormal
// and overflow handling that the scalar already does correctly.

/// Scalar bf16 encoding (used by the scalar path and as fallback).
/// Inline within protocol.rs, but we need access to the scalar helper.
#[inline]
fn encode_fp32_scalar_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Scalar fp16 encoding (used by the scalar path and as fallback).
#[inline]
fn encode_fp32_scalar_to_fp16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ff_ff;

    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7c00 | ((mant >> 13) as u16).max(1);
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        if (mantissa & round_bit) != 0
            && ((mantissa & (round_bit - 1)) != 0 || (half_mant & 1) != 0)
        {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half_mant = (mant >> 13) as u16;
    let round_bits = mant & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (half_mant & 1) != 0) {
        half_mant = half_mant.wrapping_add(1);
        if half_mant == 0x0400 {
            return sign | (((half_exp + 1) as u16) << 10);
        }
    }

    sign | ((half_exp as u16) << 10) | half_mant
}
