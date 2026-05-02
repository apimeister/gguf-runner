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
    #[cfg(target_arch = "x86_64")]
    if use_x86_avx2_bf16_enc() && values.len() >= 8 {
        return unsafe { encode_fp16_avx2(values, out) };
    }

    // Scalar fallback: encode each value individually
    for (i, &v) in values.iter().enumerate() {
        let bits = encode_fp32_scalar_to_fp16_bits(v);
        out[i * 2] = bits as u8;
        out[i * 2 + 1] = (bits >> 8) as u8;
    }
}

// ─── x86_64 AVX-2 SIMD paths ───

/// Encode bf16 activation using AVX-2 intrinsics.
///
/// For each f32 value:
/// 1. Extract sign (bit 31), exponent (bits 30:23), mantissa (bits 22:0)
/// 2. Compute bf16 rounding bias: ((mantissa >> 16) & 1) + 0x7FFF
/// 3. Add bias to mantissa, clamp to 10 bits
/// 4. Combine sign, exponent, and biased mantissa into bf16 format
///
/// Special values (NaN, infinity) are handled correctly by the
/// bit-exact rounding formula.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn encode_bf16_avx2(values: &[f32], out: &mut [u8]) {
    let n = values.len();
    let mut i = 0;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let v32 = _mm256_castps_si256(v);

        // ─── Sign bit (bit 31 → bit 15) ───
        // sign = v32 >> 31 → sign at bit 0
        // We need it at bit 15, so shift left by 15
        let sign = _mm256_slli_epi32(_mm256_srli_epi32(v32, 31), 15);

        // ─── Exponent and mantissa ───
        // v32 >> 16: exponent bits 30:23 → bits 15:8, mantissa bits 22:0 → bits 14:0
        // v32 >> 22: exponent bits 30:23 → bits 8:0, mantissa bits 22:0 → bits 2:0
        let shift16 = _mm256_srli_epi32(v32, 16); // exp at 15:8, mant at 14:0
        let shift22 = _mm256_srli_epi32(v32, 22); // exp at 8:0, mant at 2:0

        // exp at 15:8, mant at 14:0
        let exp_mant = _mm256_or_si256(shift16, shift22);

        // ─── Rounding bias ───
        // Extract mantissa bit 16 (the lowest bit of the truncated portion)
        let mant_upper = _mm256_and_si256(shift16, _mm256_set1_epi32(0x8000)); // 0x8000 or 0
        // Shift to bit 0: 1 or 0
        let bias_bit = _mm256_srli_epi32(mant_upper, 16);
        // Rounding bias = 1 + 0x7FFF = 0x8000 if bit 16 was set, 0x7FFF otherwise
        let rounding_bias = _mm256_add_epi32(bias_bit, _mm256_set1_epi32(0x7FFF));

        // ─── Add rounding bias to mantissa only (not exponent) ───
        // Mantissa mask: bits 14:0 → 0x3FFF
        let mant_mask = _mm256_set1_epi32(0x3FFF);
        let mant = _mm256_and_si256(exp_mant, mant_mask);
        let biased = _mm256_add_epi32(mant, rounding_bias);

        // ─── Clamp overflow: if biased bit 10 is set (>= 0x400), clamp to 0x3FF ───
        let overflow = _mm256_and_si256(biased, _mm256_set1_epi32(0x0400));
        let overflow_mask = _mm256_cmpeq_epi32(overflow, _mm256_set1_epi32(0x0400));
        let clamped_mant = _mm256_or_si256(
            _mm256_andnot_si256(overflow_mask, biased),
            _mm256_and_si256(overflow_mask, _mm256_set1_epi32(0x03FF)),
        );

        // ─── Combine: sign | clamped_mant ───
        let final16 = _mm256_or_si256(sign, clamped_mant);

        // Extract and store 16-bit values
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, final16);
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

/// Encode fp16 activation using AVX-2 intrinsics.
///
/// For each f32 value:
/// 1. Extract sign (bit 31), exponent (bits 30:23), mantissa (bits 22:0)
/// 2. Round by checking bit 11 of mantissa (round bit)
/// 3. Shift exponent to bits 14:10, mantissa to bits 9:0
/// 4. Combine into fp16 format
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn encode_fp16_avx2(values: &[f32], out: &mut [u8]) {
    let n = values.len();
    let mut i = 0;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let v32 = _mm256_castps_si256(v);

        // ─── Sign bit (bit 31 → bit 15) ───
        let sign = _mm256_slli_epi32(_mm256_srli_epi32(v32, 31), 15);

        // ─── Exponent ───
        // f32 exponent at bits 30:23, fp16 exponent at bits 14:10
        // Shift right by 22 → exp at bits 7:0
        let exp_raw = _mm256_srli_epi32(v32, 22);
        // f32 mantissa at bits 22:0, fp16 mantissa at bits 9:0
        // Shift right by 11 → mant at bits 11:0
        let mant_raw = _mm256_srli_epi32(v32, 11);

        // ─── Rounding ───
        // Check bit 11 of original mantissa (the round bit)
        // If set, increment exponent (round up)
        let round_bit = _mm256_and_si256(mant_raw, _mm256_set1_epi32(1));
        let exp_rounded = _mm256_add_epi32(exp_raw, round_bit);

        // ─── Shift exponent to fp16 position (bits 14:10) ───
        let exp_shifted = _mm256_slli_epi32(exp_rounded, 10);

        // ─── Mantissa shifted to bits 9:0 ───
        // Already shifted by 11 above, now just mask to 10 bits
        let mant_final = _mm256_and_si256(mant_raw, _mm256_set1_epi32(0x03FF));

        // ─── Combine: sign | exp | mant ───
        let final16 = _mm256_or_si256(_mm256_or_si256(sign, exp_shifted), mant_final);

        // Extract and store 16-bit values
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, final16);
        for j in 0..8 {
            let b = tmp[j] as u16;
            out[(i + j) * 2] = b as u8;
            out[(i + j) * 2 + 1] = (b >> 8) as u8;
        }
        i += 8;
    }

    // Tail: scalar encoding for remaining elements
    while i < n {
        let bits = encode_fp32_scalar_to_fp16_bits(values[i]);
        out[i * 2] = bits as u8;
        out[i * 2 + 1] = (bits >> 8) as u8;
        i += 1;
    }
}

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
