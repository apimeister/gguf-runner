use crate::engine::io::{bf16_to_fp32, fp16_to_fp32, read_f32_le, read_u16_le, read_u32_le};
use crate::engine::profiling::{prof_end, prof_start, PROF_MATMUL_NS};
use crate::engine::switches::{par_matmul_chunk_rows, par_matmul_min_rows};
#[cfg(target_arch = "aarch64")]
use crate::engine::switches::{
    use_aarch64_dotprod_q8, use_aarch64_qk_mr4, AARCH64_Q4K_MR4_STATUS, AARCH64_Q5K_MR4_STATUS,
    AARCH64_Q6K_MR4_STATUS,
};
#[cfg(target_arch = "x86_64")]
use crate::engine::switches::{
    use_x86_avx2_fma, use_x86_avx512_vnni_q8, use_x86_avx_vnni, use_x86_f16c, use_x86_qk_mr4,
    X86_Q4K_MR4_STATUS, X86_Q5K_MR4_STATUS, X86_Q6K_MR4_STATUS,
};
use crate::engine::types::{
    ensure_model_range, GgmlType, QuantizedTensor, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32,
    GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    KVALUES_IQ4NL, QK4_0, QK4_1, QK4_NL, QK5_0, QK5_1, QK8_0, QK_K,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};

#[cfg(target_arch = "x86_64")]
const X86_MATMUL_PREFETCH_ROWS: usize = 6;
pub(crate) fn get_block_size(ttype: GgmlType) -> usize {
    match ttype.0 {
        GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16 | 30 => 1,
        GGML_TYPE_Q4_0 => QK4_0,
        GGML_TYPE_Q4_1 => QK4_1,
        GGML_TYPE_Q5_0 => QK5_0,
        GGML_TYPE_Q5_1 => QK5_1,
        GGML_TYPE_Q8_0 => QK8_0,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => QK_K,
        GGML_TYPE_IQ4_NL => QK4_NL,
        _ => 1,
    }
}

pub(crate) fn get_type_size(ttype: GgmlType) -> usize {
    match ttype.0 {
        GGML_TYPE_F32 => 4,
        GGML_TYPE_F16 | GGML_TYPE_BF16 | 30 => 2,
        GGML_TYPE_Q4_0 => 2 + QK4_0 / 2,
        GGML_TYPE_Q4_1 => 2 + 2 + QK4_1 / 2,
        GGML_TYPE_Q5_0 => 2 + 4 + QK5_0 / 2,
        GGML_TYPE_Q5_1 => 2 + 2 + 4 + QK5_1 / 2,
        GGML_TYPE_Q8_0 => 2 + QK8_0,
        GGML_TYPE_Q2_K => QK_K / 16 + QK_K / 4 + 2 + 2,
        GGML_TYPE_Q3_K => QK_K / 8 + QK_K / 4 + 12 + 2,
        GGML_TYPE_Q4_K => 2 + 2 + 12 + QK_K / 2,
        GGML_TYPE_Q5_K => 2 + 2 + 12 + QK_K / 8 + QK_K / 2,
        GGML_TYPE_Q6_K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
        GGML_TYPE_IQ4_NL => 2 + QK4_NL / 2,
        _ => 0,
    }
}

#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn x86_prefetch_row(
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    row_idx: usize,
    total_rows: usize,
) {
    let pf_row = row_idx + X86_MATMUL_PREFETCH_ROWS;
    if pf_row >= total_rows {
        return;
    }
    let Some(pf_off) = pf_row
        .checked_mul(row_size)
        .and_then(|off| data_offset.checked_add(off))
    else {
        return;
    };
    if pf_off < mapped.len() {
        unsafe {
            _mm_prefetch(mapped.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
        }
    }
}

pub(crate) fn dequantize_row_q4_0(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
    let nb = k / QK4_0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let qs = &src[off + 2..off + 2 + QK4_0 / 2];
        for j in 0..QK4_0 / 2 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            dst[i * QK4_0 + j] = x0 as f32 * d;
            dst[i * QK4_0 + j + QK4_0 / 2] = x1 as f32 * d;
        }
    }
}

pub(crate) fn dequantize_row_q4_1(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
    let nb = k / QK4_1;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let m = fp16_to_fp32(read_u16_le(src, off + 2));
        let qs = &src[off + 4..off + 4 + QK4_1 / 2];
        for j in 0..QK4_1 / 2 {
            let x0 = (qs[j] & 0x0f) as f32;
            let x1 = (qs[j] >> 4) as f32;
            dst[i * QK4_1 + j] = x0 * d + m;
            dst[i * QK4_1 + j + QK4_1 / 2] = x1 * d + m;
        }
    }
}

pub(crate) fn dequantize_row_q5_0(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
    let nb = k / QK5_0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let qh = read_u32_le(src, off + 2);
        let qs = &src[off + 6..off + 6 + QK5_0 / 2];
        for j in 0..QK5_0 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as i32 - 16;
            dst[i * QK5_0 + j] = x0 as f32 * d;
            dst[i * QK5_0 + j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
}

pub(crate) fn dequantize_row_q5_1(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
    let nb = k / QK5_1;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let m = fp16_to_fp32(read_u16_le(src, off + 2));
        let qh = read_u32_le(src, off + 4);
        let qs = &src[off + 8..off + 8 + QK5_1 / 2];
        for j in 0..QK5_1 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as f32;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as f32;
            dst[i * QK5_1 + j] = x0 * d + m;
            dst[i * QK5_1 + j + QK5_1 / 2] = x1 * d + m;
        }
    }
}

pub(crate) fn dequantize_row_q8_0(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let nb = k / QK8_0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        for j in 0..QK8_0 {
            let q = src[off + 2 + j] as i8;
            dst[i * QK8_0 + j] = q as f32 * d;
        }
    }
}

pub(crate) fn dequantize_row_q4_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let nb = k / QK_K;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let dmin = fp16_to_fp32(read_u16_le(src, off + 2));
        let scales = &src[off + 4..off + 16];
        let mut q_off = off + 16;
        let mut y_idx = i * QK_K;
        let mut is = 0usize;
        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let q = &src[q_off..q_off + 32];
            for l in 0..32 {
                dst[y_idx] = d1 * (q[l] & 0x0f) as f32 - m1f;
                y_idx += 1;
            }
            for l in 0..32 {
                dst[y_idx] = d2 * (q[l] >> 4) as f32 - m2f;
                y_idx += 1;
            }
            q_off += 32;
            is += 2;
        }
    }
}

pub(crate) fn dequantize_row_q2_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let nb = k / QK_K;
    for i in 0..nb {
        let off = i * block_sz;
        let scales = &src[off..off + QK_K / 16];
        let mut q_off = off + QK_K / 16;
        let d = fp16_to_fp32(read_u16_le(src, off + QK_K / 16 + QK_K / 4));
        let dmin = fp16_to_fp32(read_u16_le(src, off + QK_K / 16 + QK_K / 4 + 2));

        let mut is = 0usize;
        let mut y_idx = i * QK_K;

        for _ in (0..QK_K).step_by(128) {
            let q = &src[q_off..q_off + 32];
            let mut shift = 0;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let mut dl = d * (sc & 0x0f) as f32;
                let mut ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    dst[y_idx] = dl * ((q[l] >> shift) & 0x03) as f32 - ml;
                    y_idx += 1;
                }

                let sc2 = scales[is];
                is += 1;
                dl = d * (sc2 & 0x0f) as f32;
                ml = dmin * (sc2 >> 4) as f32;
                for l in 0..16 {
                    dst[y_idx] = dl * ((q[l + 16] >> shift) & 0x03) as f32 - ml;
                    y_idx += 1;
                }

                shift += 2;
            }
            q_off += 32;
        }
    }
}

pub(crate) fn q3_scales(scales12: &[u8]) -> [i8; 16] {
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    for i in 0..12 {
        let idx = i / 4;
        aux[idx] |= (scales12[i] as u32) << ((i % 4) * 8);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    let mut out = [0i8; 16];
    for i in 0..4 {
        let b = aux[i].to_le_bytes();
        out[i * 4] = b[0] as i8;
        out[i * 4 + 1] = b[1] as i8;
        out[i * 4 + 2] = b[2] as i8;
        out[i * 4 + 3] = b[3] as i8;
    }
    out
}

pub(crate) fn dequantize_row_q3_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let nb = k / QK_K;

    for i in 0..nb {
        let off = i * block_sz;
        let hmask = &src[off..off + QK_K / 8];
        let mut q_off = off + QK_K / 8;
        let scales = q3_scales(&src[off + QK_K / 8 + QK_K / 4..off + QK_K / 8 + QK_K / 4 + 12]);
        let d_all = fp16_to_fp32(read_u16_le(src, off + QK_K / 8 + QK_K / 4 + 12));

        let mut is = 0usize;
        let mut y_idx = i * QK_K;
        let mut m: u8 = 1;

        for _ in (0..QK_K).step_by(128) {
            let q = &src[q_off..q_off + 32];
            let mut shift = 0usize;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let w = ((q[l] >> shift) & 3) as i8 - if (hmask[l] & m) != 0 { 0 } else { 4 };
                    dst[y_idx] = dl * w as f32;
                    y_idx += 1;
                }

                let dl2 = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let w = ((q[l + 16] >> shift) & 3) as i8
                        - if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    dst[y_idx] = dl2 * w as f32;
                    y_idx += 1;
                }

                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    }
}

pub(crate) fn dequantize_row_q5_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let nb = k / QK_K;

    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let dmin = fp16_to_fp32(read_u16_le(src, off + 2));
        let scales = &src[off + 4..off + 16];
        let qh = &src[off + 16..off + 16 + QK_K / 8];
        let mut ql_off = off + 16 + QK_K / 8;

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut y_idx = i * QK_K;

        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            let ql = &src[ql_off..ql_off + 32];

            for l in 0..32 {
                let v = (ql[l] & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                dst[y_idx] = d1 * v as f32 - m1f;
                y_idx += 1;
            }
            for l in 0..32 {
                let v = (ql[l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                dst[y_idx] = d2 * v as f32 - m2f;
                y_idx += 1;
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

pub(crate) fn dequantize_row_q6_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let nb = k / QK_K;

    for i in 0..nb {
        let off = i * block_sz;
        let mut ql_off = off;
        let mut qh_off = off + QK_K / 2;
        let mut sc_off = off + QK_K / 2 + QK_K / 4;
        let d = fp16_to_fp32(read_u16_le(src, off + QK_K / 2 + QK_K / 4 + QK_K / 16));

        let mut y_idx = i * QK_K;
        for _ in (0..QK_K).step_by(128) {
            let ql = &src[ql_off..ql_off + 64];
            let qh = &src[qh_off..qh_off + 32];
            let sc = &src[sc_off..sc_off + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                dst[y_idx + l] = d * sc[is] as i8 as f32 * q1 as f32;
                dst[y_idx + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                dst[y_idx + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                dst[y_idx + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            y_idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

pub(crate) fn dequantize_row_f16(src: &[u8], dst: &mut [f32], k: usize) {
    for i in 0..k {
        dst[i] = fp16_to_fp32(read_u16_le(src, i * 2));
    }
}

pub(crate) fn dequantize_row_bf16(src: &[u8], dst: &mut [f32], k: usize) {
    for i in 0..k {
        dst[i] = bf16_to_fp32(read_u16_le(src, i * 2));
    }
}

pub(crate) fn dequantize_row_iq4_nl(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
    let nb = k / QK4_NL;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let qs = &src[off + 2..off + 2 + QK4_NL / 2];
        for j in 0..QK4_NL / 2 {
            dst[i * QK4_NL + j] = d * KVALUES_IQ4NL[(qs[j] & 0x0f) as usize] as f32;
            dst[i * QK4_NL + j + QK4_NL / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

pub(crate) fn dequantize_tensor(
    src: &[u8],
    n_elements: usize,
    ttype: GgmlType,
) -> Result<Vec<f32>, String> {
    let mut dst = vec![0.0; n_elements];
    match ttype.0 {
        GGML_TYPE_F32 => {
            for i in 0..n_elements {
                dst[i] = read_f32_le(src, i * 4);
            }
        }
        GGML_TYPE_F16 => dequantize_row_f16(src, &mut dst, n_elements),
        GGML_TYPE_Q4_0 => dequantize_row_q4_0(src, &mut dst, n_elements),
        GGML_TYPE_Q4_1 => dequantize_row_q4_1(src, &mut dst, n_elements),
        GGML_TYPE_Q5_0 => dequantize_row_q5_0(src, &mut dst, n_elements),
        GGML_TYPE_Q5_1 => dequantize_row_q5_1(src, &mut dst, n_elements),
        GGML_TYPE_Q8_0 => dequantize_row_q8_0(src, &mut dst, n_elements),
        GGML_TYPE_Q2_K => dequantize_row_q2_k(src, &mut dst, n_elements),
        GGML_TYPE_Q3_K => dequantize_row_q3_k(src, &mut dst, n_elements),
        GGML_TYPE_Q4_K => dequantize_row_q4_k(src, &mut dst, n_elements),
        GGML_TYPE_Q5_K => dequantize_row_q5_k(src, &mut dst, n_elements),
        GGML_TYPE_Q6_K => dequantize_row_q6_k(src, &mut dst, n_elements),
        GGML_TYPE_IQ4_NL => dequantize_row_iq4_nl(src, &mut dst, n_elements),
        GGML_TYPE_BF16 | 30 => dequantize_row_bf16(src, &mut dst, n_elements),
        _ => return Err(format!("unsupported quantization type: {}", ttype.0)),
    }
    Ok(dst)
}

#[inline(always)]
pub(crate) fn dot_f32_scalar_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    while i < n {
        unsafe {
            sum += *a.add(i) * *b.add(i);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_f32_simd_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i + 16 <= n {
        let a0 = vld1q_f32(a.add(i));
        let b0 = vld1q_f32(b.add(i));
        let a1 = vld1q_f32(a.add(i + 4));
        let b1 = vld1q_f32(b.add(i + 4));
        let a2 = vld1q_f32(a.add(i + 8));
        let b2 = vld1q_f32(b.add(i + 8));
        let a3 = vld1q_f32(a.add(i + 12));
        let b3 = vld1q_f32(b.add(i + 12));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
        i += 16;
    }
    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    while i + 4 <= n {
        let av = vld1q_f32(a.add(i));
        let bv = vld1q_f32(b.add(i));
        acc = vfmaq_f32(acc, av, bv);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(a.add(i));
        let b0 = _mm256_loadu_ps(b.add(i));
        let a1 = _mm256_loadu_ps(a.add(i + 8));
        let b1 = _mm256_loadu_ps(b.add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().copied().sum::<f32>();
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn vec_dot_f16_f16c_prefix(x: *const f32, w: *const u8, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let hv = _mm_loadu_si128(w.add(i * 2) as *const __m128i);
        let wv = _mm256_cvtph_ps(hv);
        acc = _mm256_fmadd_ps(xv, wv, acc);
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    tmp.iter().copied().sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_bf16_avx2_prefix(x: *const f32, w: *const u8, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let hv = _mm_loadu_si128(w.add(i * 2) as *const __m128i);
        let wv_i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hv), 16);
        let wv = _mm256_castsi256_ps(wv_i32);
        acc = _mm256_fmadd_ps(xv, wv, acc);
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    tmp.iter().copied().sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().copied().sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cvt_u8x8_to_f32x8(v8: __m128i) -> __m256 {
    let zero = _mm_setzero_si128();
    let lo16 = _mm_unpacklo_epi8(v8, zero);
    let lo32 = _mm256_cvtepu16_epi32(lo16);
    _mm256_cvtepi32_ps(lo32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cvt_i8x8_to_f32x8(v8: __m128i) -> __m256 {
    let lo32 = _mm256_cvtepi8_epi32(v8);
    _mm256_cvtepi32_ps(lo32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_u8_vals_avx2_ptr(x: *const f32, q: *const u8, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let q8 = _mm_loadl_epi64(q.add(i) as *const __m128i);
        let qf = cvt_u8x8_to_f32x8(q8);
        acc = _mm256_fmadd_ps(xv, qf, acc);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        sum += *x.add(i) * *q.add(i) as f32;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_vals_avx2_ptr(x: *const f32, q: *const i8, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let q8 = _mm_loadl_epi64(q.add(i) as *const __m128i);
        let qf = cvt_i8x8_to_f32x8(q8);
        acc = _mm256_fmadd_ps(xv, qf, acc);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        sum += *x.add(i) * *q.add(i) as f32;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_nibbles_pair_avx2_ptr(
    x_lo: *const f32,
    x_hi: *const f32,
    q: *const u8,
    n: usize,
) -> (f32, f32) {
    let nib_mask = _mm_set1_epi8(0x0f);
    let mut i = 0usize;
    let mut acc_lo = _mm256_setzero_ps();
    let mut acc_hi = _mm256_setzero_ps();

    while i + 8 <= n {
        let xv_lo = _mm256_loadu_ps(x_lo.add(i));
        let xv_hi = _mm256_loadu_ps(x_hi.add(i));
        let q8 = _mm_loadl_epi64(q.add(i) as *const __m128i);
        let lo8 = _mm_and_si128(q8, nib_mask);
        let hi8 = _mm_and_si128(_mm_srli_epi16(q8, 4), nib_mask);
        let q_lo_f = cvt_u8x8_to_f32x8(lo8);
        let q_hi_f = cvt_u8x8_to_f32x8(hi8);
        acc_lo = _mm256_fmadd_ps(xv_lo, q_lo_f, acc_lo);
        acc_hi = _mm256_fmadd_ps(xv_hi, q_hi_f, acc_hi);
        i += 8;
    }

    let mut sum_lo = hsum256_ps(acc_lo);
    let mut sum_hi = hsum256_ps(acc_hi);
    while i < n {
        let qv = *q.add(i);
        sum_lo += *x_lo.add(i) * (qv & 0x0f) as f32;
        sum_hi += *x_hi.add(i) * (qv >> 4) as f32;
        i += 1;
    }
    (sum_lo, sum_hi)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_f32_simd_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    if use_x86_avx2_fma() {
        return dot_f32_avx2_ptr(a, b, n);
    }
    let mut i = 0usize;
    let mut acc = _mm_setzero_ps();
    while i + 4 <= n {
        let av = _mm_loadu_ps(a.add(i));
        let bv = _mm_loadu_ps(b.add(i));
        acc = _mm_add_ps(acc, _mm_mul_ps(av, bv));
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[inline(always)]
pub(crate) fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        return dot_f32_simd_ptr(a.as_ptr(), b.as_ptr(), a.len());
    }
    #[allow(unreachable_code)]
    dot_f32_scalar_ptr(a.as_ptr(), b.as_ptr(), a.len())
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn axpy_simd_ptr(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    let mut i = 0usize;
    let av = vdupq_n_f32(a);
    while i + 16 <= n {
        let dv0 = vld1q_f32(dst.add(i));
        let sv0 = vld1q_f32(src.add(i));
        let dv1 = vld1q_f32(dst.add(i + 4));
        let sv1 = vld1q_f32(src.add(i + 4));
        let dv2 = vld1q_f32(dst.add(i + 8));
        let sv2 = vld1q_f32(src.add(i + 8));
        let dv3 = vld1q_f32(dst.add(i + 12));
        let sv3 = vld1q_f32(src.add(i + 12));
        vst1q_f32(dst.add(i), vfmaq_f32(dv0, sv0, av));
        vst1q_f32(dst.add(i + 4), vfmaq_f32(dv1, sv1, av));
        vst1q_f32(dst.add(i + 8), vfmaq_f32(dv2, sv2, av));
        vst1q_f32(dst.add(i + 12), vfmaq_f32(dv3, sv3, av));
        i += 16;
    }
    while i + 4 <= n {
        let dv = vld1q_f32(dst.add(i));
        let sv = vld1q_f32(src.add(i));
        let out = vfmaq_f32(dv, sv, av);
        vst1q_f32(dst.add(i), out);
        i += 4;
    }
    while i < n {
        *dst.add(i) += a * *src.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2_ptr(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    let mut i = 0usize;
    let av = _mm256_set1_ps(a);
    while i + 16 <= n {
        let dv0 = _mm256_loadu_ps(dst.add(i));
        let sv0 = _mm256_loadu_ps(src.add(i));
        let dv1 = _mm256_loadu_ps(dst.add(i + 8));
        let sv1 = _mm256_loadu_ps(src.add(i + 8));
        _mm256_storeu_ps(dst.add(i), _mm256_fmadd_ps(sv0, av, dv0));
        _mm256_storeu_ps(dst.add(i + 8), _mm256_fmadd_ps(sv1, av, dv1));
        i += 16;
    }
    while i + 8 <= n {
        let dv = _mm256_loadu_ps(dst.add(i));
        let sv = _mm256_loadu_ps(src.add(i));
        _mm256_storeu_ps(dst.add(i), _mm256_fmadd_ps(sv, av, dv));
        i += 8;
    }
    while i < n {
        *dst.add(i) += a * *src.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn axpy_simd_ptr(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    if use_x86_avx2_fma() {
        return axpy_avx2_ptr(dst, src, a, n);
    }
    let mut i = 0usize;
    let av = _mm_set1_ps(a);
    while i + 4 <= n {
        let dv = _mm_loadu_ps(dst.add(i));
        let sv = _mm_loadu_ps(src.add(i));
        let out = _mm_add_ps(dv, _mm_mul_ps(sv, av));
        _mm_storeu_ps(dst.add(i), out);
        i += 4;
    }
    while i < n {
        *dst.add(i) += a * *src.add(i);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn axpy_inplace(dst: &mut [f32], a: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        axpy_simd_ptr(dst.as_mut_ptr(), src.as_ptr(), a, dst.len());
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..dst.len() {
        dst[i] += a * src[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scale_simd_inplace(x: *mut f32, alpha: f32, n: usize) {
    let mut i = 0usize;
    let av = vdupq_n_f32(alpha);
    while i + 16 <= n {
        let xv0 = vld1q_f32(x.add(i));
        let xv1 = vld1q_f32(x.add(i + 4));
        let xv2 = vld1q_f32(x.add(i + 8));
        let xv3 = vld1q_f32(x.add(i + 12));
        vst1q_f32(x.add(i), vmulq_f32(xv0, av));
        vst1q_f32(x.add(i + 4), vmulq_f32(xv1, av));
        vst1q_f32(x.add(i + 8), vmulq_f32(xv2, av));
        vst1q_f32(x.add(i + 12), vmulq_f32(xv3, av));
        i += 16;
    }
    while i + 4 <= n {
        let xv = vld1q_f32(x.add(i));
        let out = vmulq_f32(xv, av);
        vst1q_f32(x.add(i), out);
        i += 4;
    }
    while i < n {
        *x.add(i) *= alpha;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2_inplace(x: *mut f32, alpha: f32, n: usize) {
    let mut i = 0usize;
    let av = _mm256_set1_ps(alpha);
    while i + 16 <= n {
        let xv0 = _mm256_loadu_ps(x.add(i));
        let xv1 = _mm256_loadu_ps(x.add(i + 8));
        _mm256_storeu_ps(x.add(i), _mm256_mul_ps(xv0, av));
        _mm256_storeu_ps(x.add(i + 8), _mm256_mul_ps(xv1, av));
        i += 16;
    }
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        _mm256_storeu_ps(x.add(i), _mm256_mul_ps(xv, av));
        i += 8;
    }
    while i < n {
        *x.add(i) *= alpha;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scale_simd_inplace(x: *mut f32, alpha: f32, n: usize) {
    if use_x86_avx2_fma() {
        return scale_avx2_inplace(x, alpha, n);
    }
    let mut i = 0usize;
    let av = _mm_set1_ps(alpha);
    while i + 4 <= n {
        let xv = _mm_loadu_ps(x.add(i));
        let out = _mm_mul_ps(xv, av);
        _mm_storeu_ps(x.add(i), out);
        i += 4;
    }
    while i < n {
        *x.add(i) *= alpha;
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn scale_slice_inplace(x: &mut [f32], alpha: f32) {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        scale_simd_inplace(x.as_mut_ptr(), alpha, x.len());
        return;
    }
    #[allow(unreachable_code)]
    for v in x {
        *v *= alpha;
    }
}

#[inline(always)]
pub(crate) fn vec_dot_f32(x: &[f32], w: &[u8], n: usize) -> f32 {
    let w_ptr = w.as_ptr() as *const f32;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        return dot_f32_simd_ptr(x.as_ptr(), w_ptr, n);
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i] * read_f32_le(w, i * 4);
        }
        sum
    }
}

#[inline(always)]
pub(crate) fn vec_dot_f16(x: &[f32], w: &[u8], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = [
                fp16_to_fp32(read_u16_le(w, i * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = vld1q_f32(wv.as_ptr());
            acc = vfmaq_f32(acc, xv, wq);
            i += 4;
        }
        sum += vaddvq_f32(acc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if use_x86_f16c() {
            let n8 = n & !7;
            if n8 > 0 {
                sum += vec_dot_f16_f16c_prefix(x.as_ptr(), w.as_ptr(), n8);
                i = n8;
            }
        }
        let mut acc = _mm_setzero_ps();
        while i + 4 <= n {
            let xv = _mm_loadu_ps(x.as_ptr().add(i));
            let wv = [
                fp16_to_fp32(read_u16_le(w, i * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = _mm_loadu_ps(wv.as_ptr());
            acc = _mm_add_ps(acc, _mm_mul_ps(xv, wq));
            i += 4;
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    while i < n {
        sum += x[i] * fp16_to_fp32(read_u16_le(w, i * 2));
        i += 1;
    }
    sum
}

#[inline(always)]
pub(crate) fn vec_dot_bf16(x: &[f32], w: &[u8], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = [
                bf16_to_fp32(read_u16_le(w, i * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = vld1q_f32(wv.as_ptr());
            acc = vfmaq_f32(acc, xv, wq);
            i += 4;
        }
        sum += vaddvq_f32(acc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if use_x86_avx2_fma() {
            let n8 = n & !7;
            if n8 > 0 {
                sum += vec_dot_bf16_avx2_prefix(x.as_ptr(), w.as_ptr(), n8);
                i = n8;
            }
        }
        let mut acc = _mm_setzero_ps();
        while i + 4 <= n {
            let xv = _mm_loadu_ps(x.as_ptr().add(i));
            let wv = [
                bf16_to_fp32(read_u16_le(w, i * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = _mm_loadu_ps(wv.as_ptr());
            acc = _mm_add_ps(acc, _mm_mul_ps(xv, wq));
            i += 4;
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    while i < n {
        sum += x[i] * bf16_to_fp32(read_u16_le(w, i * 2));
        i += 1;
    }
    sum
}

pub(crate) fn vec_dot_q4_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK4_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let qs = &w[off + 2..off + 2 + QK4_0 / 2];
        let xb = &x[i * QK4_0..(i + 1) * QK4_0];
        let mut block_sum = 0.0;
        for j in 0..QK4_0 / 2 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            block_sum += xb[j] * x0 as f32 + xb[j + QK4_0 / 2] * x1 as f32;
        }
        sum += block_sum * d;
    }
    sum
}

pub(crate) fn vec_dot_q4_1(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK4_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let m = fp16_to_fp32(read_u16_le(w, off + 2));
        let qs = &w[off + 4..off + 4 + QK4_1 / 2];
        let xb = &x[i * QK4_1..(i + 1) * QK4_1];
        let mut block_sum = 0.0;
        let mut x_sum = 0.0;
        for j in 0..QK4_1 / 2 {
            let x0 = (qs[j] & 0x0f) as f32;
            let x1 = (qs[j] >> 4) as f32;
            block_sum += xb[j] * x0 + xb[j + QK4_1 / 2] * x1;
            x_sum += xb[j] + xb[j + QK4_1 / 2];
        }
        sum += block_sum * d + x_sum * m;
    }
    sum
}

pub(crate) fn vec_dot_q5_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK5_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let qh = read_u32_le(w, off + 2);
        let qs = &w[off + 6..off + 6 + QK5_0 / 2];
        let xb = &x[i * QK5_0..(i + 1) * QK5_0];
        let mut block_sum = 0.0;
        for j in 0..QK5_0 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as i32 - 16;
            block_sum += xb[j] * x0 as f32 + xb[j + QK5_0 / 2] * x1 as f32;
        }
        sum += block_sum * d;
    }
    sum
}

pub(crate) fn vec_dot_q5_1(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK5_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let m = fp16_to_fp32(read_u16_le(w, off + 2));
        let qh = read_u32_le(w, off + 4);
        let qs = &w[off + 8..off + 8 + QK5_1 / 2];
        let xb = &x[i * QK5_1..(i + 1) * QK5_1];
        let mut block_sum = 0.0;
        let mut x_sum = 0.0;
        for j in 0..QK5_1 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as f32;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as f32;
            block_sum += xb[j] * x0 + xb[j + QK5_1 / 2] * x1;
            x_sum += xb[j] + xb[j + QK5_1 / 2];
        }
        sum += block_sum * d + x_sum * m;
    }
    sum
}

pub(crate) fn vec_dot_q8_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if use_aarch64_dotprod_q8() {
        unsafe {
            return vec_dot_q8_0_dotprod(x, w, n);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if use_x86_avx512_vnni_q8() {
        unsafe {
            return vec_dot_q8_0_x86_avx512vnni(x, w, n);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if use_x86_avx_vnni() {
        unsafe {
            return vec_dot_q8_0_x86_avxvnni(x, w, n);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q8_0_x86_avx2(x, w, n);
        }
    }

    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut qf = [0.0f32; QK8_0];
        for j in 0..QK8_0 {
            qf[j] = w[off + 2 + j] as i8 as f32;
        }
        let block_sum = dot_f32_simd(xb, &qf);
        sum += block_sum * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q8_0_x86_avx2(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let q = &w[off + 2..off + 2 + QK8_0];
        let block_sum = dot_f32_i8_vals_avx2_ptr(xb.as_ptr(), q.as_ptr() as *const i8, QK8_0);
        sum += block_sum * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sum_i8_32_ptr(v: *const i8) -> i32 {
    let mut sum = 0i32;
    for i in 0..QK8_0 {
        sum += *v.add(i) as i32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn dot_u8_i8_32_x86_avxvnni(a_u8: *const u8, b_i8: *const i8) -> i32 {
    let src = _mm256_setzero_si256();
    let a = _mm256_loadu_si256(a_u8 as *const __m256i);
    let b = _mm256_loadu_si256(b_i8 as *const __m256i);
    let acc = _mm256_dpbusd_avx_epi32(src, a, b);
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn vec_dot_q8_0_x86_avxvnni(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    let mut xq_u8 = [0u8; QK8_0];

    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut abs_max = 0.0f32;
        for &v in xb {
            let a = v.abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        if abs_max == 0.0 {
            continue;
        }
        let x_scale = abs_max / 127.0;
        let inv_x_scale = 1.0 / x_scale;
        for j in 0..QK8_0 {
            let q = (xb[j] * inv_x_scale).round().clamp(-127.0, 127.0) as i32;
            xq_u8[j] = (q + 128) as u8;
        }
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let dot_u = dot_u8_i8_32_x86_avxvnni(xq_u8.as_ptr(), q_ptr);
        let sum_q = sum_i8_32_ptr(q_ptr);
        let dot_s = dot_u - 128 * sum_q;
        sum += dot_s as f32 * x_scale * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl")]
unsafe fn dot_u8_i8_32_x86_avx512vnni(a_u8: *const u8, b_i8: *const i8) -> i32 {
    let src = _mm256_setzero_si256();
    let a = _mm256_loadu_si256(a_u8 as *const __m256i);
    let b = _mm256_loadu_si256(b_i8 as *const __m256i);
    let acc = _mm256_dpbusd_epi32(src, a, b);
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl")]
unsafe fn vec_dot_q8_0_x86_avx512vnni(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    let mut xq_u8 = [0u8; QK8_0];

    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut abs_max = 0.0f32;
        for &v in xb {
            let a = v.abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        if abs_max == 0.0 {
            continue;
        }
        let x_scale = abs_max / 127.0;
        let inv_x_scale = 1.0 / x_scale;
        for j in 0..QK8_0 {
            let q = (xb[j] * inv_x_scale).round().clamp(-127.0, 127.0) as i32;
            xq_u8[j] = (q + 128) as u8;
        }
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let dot_u = dot_u8_i8_32_x86_avx512vnni(xq_u8.as_ptr(), q_ptr);
        let sum_q = sum_i8_32_ptr(q_ptr);
        let dot_s = dot_u - 128 * sum_q;
        sum += dot_s as f32 * x_scale * d;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_16_neon(a: *const i8, b: *const i8) -> i32 {
    let av = vld1q_s8(a);
    let bv = vld1q_s8(b);
    let prod0 = vmull_s8(vget_low_s8(av), vget_low_s8(bv));
    let prod1 = vmull_s8(vget_high_s8(av), vget_high_s8(bv));
    let sum0 = vpaddlq_s16(prod0);
    let sum1 = vpaddlq_s16(prod1);
    vaddvq_s32(vaddq_s32(sum0, sum1))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_32_dotprod(a: *const i8, b: *const i8) -> i32 {
    dot_i8_16_neon(a, b) + dot_i8_16_neon(a.add(16), b.add(16))
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_dot_q8_0_dotprod(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    let mut xq = [0i8; QK8_0];

    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut abs_max = 0.0f32;
        for &v in xb {
            let a = v.abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        if abs_max == 0.0 {
            continue;
        }
        let x_scale = abs_max / 127.0;
        let inv_x_scale = 1.0 / x_scale;
        for j in 0..QK8_0 {
            let q = (xb[j] * inv_x_scale).round().clamp(-127.0, 127.0);
            xq[j] = q as i8;
        }
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let dot_i32 = dot_i8_32_dotprod(xq.as_ptr(), q_ptr);
        sum += dot_i32 as f32 * x_scale * d;
    }
    sum
}

pub(crate) fn vec_dot_q2_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mut sum = 0.0;

    for i in 0..nb {
        let off = i * block_sz;
        let scales = &w[off..off + QK_K / 16];
        let mut q_off = off + QK_K / 16;
        let d = fp16_to_fp32(read_u16_le(w, off + QK_K / 16 + QK_K / 4));
        let dmin = fp16_to_fp32(read_u16_le(w, off + QK_K / 16 + QK_K / 4 + 2));
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut block_sum = 0.0;

        for n_outer in (0..QK_K).step_by(128) {
            let q = &w[q_off..q_off + 32];
            let mut shift = 0;
            for j in 0..4 {
                let sc = scales[is];
                is += 1;
                let mut dl = d * (sc & 0x0f) as f32;
                let mut ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + l;
                    let wv = dl * ((q[l] >> shift) & 0x03) as f32 - ml;
                    block_sum += xb[idx] * wv;
                }
                let sc2 = scales[is];
                is += 1;
                dl = d * (sc2 & 0x0f) as f32;
                ml = dmin * (sc2 >> 4) as f32;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + 16 + l;
                    let wv = dl * ((q[l + 16] >> shift) & 0x03) as f32 - ml;
                    block_sum += xb[idx] * wv;
                }
                shift += 2;
            }
            q_off += 32;
        }
        sum += block_sum;
    }
    sum
}

pub(crate) fn vec_dot_q3_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let mut sum = 0.0;

    for i in 0..nb {
        let off = i * block_sz;
        let hmask = &w[off..off + QK_K / 8];
        let mut q_off = off + QK_K / 8;
        let scales = q3_scales(&w[off + QK_K / 8 + QK_K / 4..off + QK_K / 8 + QK_K / 4 + 12]);
        let d_all = fp16_to_fp32(read_u16_le(w, off + QK_K / 8 + QK_K / 4 + 12));
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut m: u8 = 1;
        let mut block_sum = 0.0;

        for n_outer in (0..QK_K).step_by(128) {
            let q = &w[q_off..q_off + 32];
            let mut shift = 0usize;
            for j in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + l;
                    let wv = ((q[l] >> shift) & 3) as i8 - if (hmask[l] & m) != 0 { 0 } else { 4 };
                    block_sum += xb[idx] * dl * wv as f32;
                }
                let dl2 = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + 16 + l;
                    let wv = ((q[l + 16] >> shift) & 3) as i8
                        - if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    block_sum += xb[idx] * dl2 * wv as f32;
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
        sum += block_sum;
    }
    sum
}

pub(crate) fn vec_dot_q4_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sum = 0.0;

    #[cfg(target_arch = "aarch64")]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let mut q_off = off + 16;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut block_sum = 0.0f32;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let q = &w[q_off..q_off + 32];
            for l in 0..32 {
                let qv = q[l];
                let w0 = d1 * (qv & 0x0f) as f32 - m1f;
                let w1 = d2 * (qv >> 4) as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }
            q_off += 32;
            is += 2;
        }
        sum += block_sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let mut q_off = off + 16;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut block_sum = 0.0;

        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let q = &w[q_off..q_off + 32];
            for l in 0..32 {
                let qv = q[l];
                let w0 = d1 * (qv & 0x0f) as f32 - m1f;
                let w1 = d2 * (qv >> 4) as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }
            q_off += 32;
            is += 2;
        }
        sum += block_sum;
    }
    sum
}

pub(crate) fn vec_dot_q5_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sum = 0.0;

    #[cfg(target_arch = "aarch64")]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let qh = &w[off + 16..off + 16 + QK_K / 8];
        let mut ql_off = off + 16 + QK_K / 8;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut block_sum = 0.0f32;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let ql = &w[ql_off..ql_off + 32];
            for l in 0..32 {
                let qv = ql[l];
                let lo = (qv & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi = (qv >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                let w0 = d1 * lo as f32 - m1f;
                let w1 = d2 * hi as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += block_sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let qh = &w[off + 16..off + 16 + QK_K / 8];
        let mut ql_off = off + 16 + QK_K / 8;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut block_sum = 0.0;

        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            let ql = &w[ql_off..ql_off + 32];

            for l in 0..32 {
                let qv = ql[l];
                let lo = (qv & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi = (qv >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                let w0 = d1 * lo as f32 - m1f;
                let w1 = d2 * hi as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += block_sum;
    }
    sum
}

pub(crate) fn vec_dot_q6_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sum = 0.0;

    #[cfg(target_arch = "aarch64")]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off + QK_K / 2 + QK_K / 4 + QK_K / 16));
        let mut ql_off = off;
        let mut qh_off = off + QK_K / 2;
        let mut sc_off = off + QK_K / 2 + QK_K / 4;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut block_sum = 0.0f32;
        for n_outer in (0..QK_K).step_by(128) {
            let ql = &w[ql_off..ql_off + 64];
            let qh = &w[qh_off..qh_off + 32];
            let sc = &w[sc_off..sc_off + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                let s0 = d * sc[is] as i8 as f32;
                let s1 = d * sc[is + 2] as i8 as f32;
                let s2 = d * sc[is + 4] as i8 as f32;
                let s3 = d * sc[is + 6] as i8 as f32;
                block_sum += xb[n_outer + l] * (s0 * q1 as f32);
                block_sum += xb[n_outer + 32 + l] * (s1 * q2 as f32);
                block_sum += xb[n_outer + 64 + l] * (s2 * q3 as f32);
                block_sum += xb[n_outer + 96 + l] * (s3 * q4 as f32);
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
        sum += block_sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off + QK_K / 2 + QK_K / 4 + QK_K / 16));
        let mut ql_off = off;
        let mut qh_off = off + QK_K / 2;
        let mut sc_off = off + QK_K / 2 + QK_K / 4;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut block_sum = 0.0;
        for n_outer in (0..QK_K).step_by(128) {
            let ql = &w[ql_off..ql_off + 64];
            let qh = &w[qh_off..qh_off + 32];
            let sc = &w[sc_off..sc_off + 8];

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                let s0 = d * sc[is] as i8 as f32;
                let s1 = d * sc[is + 2] as i8 as f32;
                let s2 = d * sc[is + 4] as i8 as f32;
                let s3 = d * sc[is + 6] as i8 as f32;
                block_sum += xb[n_outer + l] * (s0 * q1 as f32);
                block_sum += xb[n_outer + 32 + l] * (s1 * q2 as f32);
                block_sum += xb[n_outer + 64 + l] * (s2 * q3 as f32);
                block_sum += xb[n_outer + 96 + l] * (s3 * q4 as f32);
            }

            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
        sum += block_sum;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn vec_dot_q4_k_4rows(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut q_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            q_off[r] = off + 16;
        }

        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][q_off[r] + l];
                    sums[r] += x0 * (a_lo[r] * (qv & 0x0f) as f32 - b_lo[r])
                        + x1 * (a_hi[r] * (qv >> 4) as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn vec_dot_q5_k_4rows(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut qh = [&[][..]; 4];
        let mut ql_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            qh[r] = &rows[r][off + 16..off + 16 + QK_K / 8];
            ql_off[r] = off + 16 + QK_K / 8;
        }

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][ql_off[r] + l];
                    let lo = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    let hi = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                    sums[r] +=
                        x0 * (a_lo[r] * lo as f32 - b_lo[r]) + x1 * (a_hi[r] * hi as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                ql_off[r] += 32;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn vec_dot_q6_k_4rows(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut ql_off = [0usize; 4];
        let mut qh_off = [0usize; 4];
        let mut sc_off = [0usize; 4];
        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 2 + QK_K / 4 + QK_K / 16));
            ql_off[r] = off;
            qh_off[r] = off + QK_K / 2;
            sc_off[r] = off + QK_K / 2 + QK_K / 4;
        }

        for n_outer in (0..QK_K).step_by(128) {
            let mut ql = [&[][..]; 4];
            let mut qh = [&[][..]; 4];
            let mut sc = [&[][..]; 4];
            for r in 0..4 {
                ql[r] = &rows[r][ql_off[r]..ql_off[r] + 64];
                qh[r] = &rows[r][qh_off[r]..qh_off[r] + 32];
                sc[r] = &rows[r][sc_off[r]..sc_off[r] + 8];
            }

            for l in 0..32 {
                let is = l / 16;
                let x0 = xb[n_outer + l];
                let x1 = xb[n_outer + 32 + l];
                let x2 = xb[n_outer + 64 + l];
                let x3 = xb[n_outer + 96 + l];
                for r in 0..4 {
                    let ql0 = ql[r][l];
                    let ql1 = ql[r][l + 32];
                    let qh0 = qh[r][l];
                    let q1 = (((ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4)) as i8) - 32;
                    let q2 = (((ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4)) as i8) - 32;
                    let q3 = (((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)) as i8) - 32;
                    let q4 = (((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)) as i8) - 32;
                    let s0 = d[r] * sc[r][is] as i8 as f32;
                    let s1 = d[r] * sc[r][is + 2] as i8 as f32;
                    let s2 = d[r] * sc[r][is + 4] as i8 as f32;
                    let s3 = d[r] * sc[r][is + 6] as i8 as f32;
                    sums[r] += x0 * (s0 * q1 as f32)
                        + x1 * (s1 * q2 as f32)
                        + x2 * (s2 * q3 as f32)
                        + x3 * (s3 * q4 as f32);
                }
            }
            for r in 0..4 {
                ql_off[r] += 64;
                qh_off[r] += 32;
                sc_off[r] += 8;
            }
        }
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn matmul_qk_mr4_chunk(
    out: &mut [f32],
    base_row: usize,
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) {
    let mut i = 0usize;
    while i + 4 <= out.len() {
        let row0_off = data_offset + (base_row + i) * row_size;
        let row1_off = row0_off + row_size;
        let row2_off = row1_off + row_size;
        let row3_off = row2_off + row_size;
        let r0 = &mapped[row0_off..row0_off + row_size];
        let r1 = &mapped[row1_off..row1_off + row_size];
        let r2 = &mapped[row2_off..row2_off + row_size];
        let r3 = &mapped[row3_off..row3_off + row_size];
        let sums = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
        out[i] = sums[0];
        out[i + 1] = sums[1];
        out[i + 2] = sums[2];
        out[i + 3] = sums[3];
        i += 4;
    }
    while i < out.len() {
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn mr4_status(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q4_K => &AARCH64_Q4K_MR4_STATUS,
        GGML_TYPE_Q5_K => &AARCH64_Q5K_MR4_STATUS,
        GGML_TYPE_Q6_K => &AARCH64_Q6K_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn validate_qk_mr4_once(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    let status = mr4_status(ttype);
    match status.load(AtomicOrdering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }

    let r0 = &mapped[data_offset..data_offset + row_size];
    let r1 = &mapped[data_offset + row_size..data_offset + 2 * row_size];
    let r2 = &mapped[data_offset + 2 * row_size..data_offset + 3 * row_size];
    let r3 = &mapped[data_offset + 3 * row_size..data_offset + 4 * row_size];

    let mr4 = match ttype {
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q4_K => [
            vec_dot_q4_k(x, r0, n),
            vec_dot_q4_k(x, r1, n),
            vec_dot_q4_k(x, r2, n),
            vec_dot_q4_k(x, r3, n),
        ],
        GGML_TYPE_Q5_K => [
            vec_dot_q5_k(x, r0, n),
            vec_dot_q5_k(x, r1, n),
            vec_dot_q5_k(x, r2, n),
            vec_dot_q5_k(x, r3, n),
        ],
        GGML_TYPE_Q6_K => [
            vec_dot_q6_k(x, r0, n),
            vec_dot_q6_k(x, r1, n),
            vec_dot_q6_k(x, r2, n),
            vec_dot_q6_k(x, r3, n),
        ],
        _ => unreachable!(),
    };

    let mut ok = true;
    for i in 0..4 {
        let a = mr4[i];
        let b = scalar[i];
        let tol = 1e-4f32 * b.abs().max(1.0);
        if (a - b).abs() > tol {
            ok = false;
            break;
        }
    }

    status.store(if ok { 1 } else { 2 }, AtomicOrdering::Relaxed);
    if !ok {
        eprintln!(
            "Warning: disabling aarch64 MR4 kernel for type {} due to validation mismatch",
            ttype
        );
    }
    ok
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn try_matmul_qk_mr4(
    xout: &mut [f32],
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    if !use_aarch64_qk_mr4() {
        return false;
    }
    if !matches!(ttype, GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K) {
        return false;
    }
    if n < QK_K || n % QK_K != 0 {
        return false;
    }

    let d = xout.len();
    if d < 4 {
        return false;
    }
    if !validate_qk_mr4_once(x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }
    let chunk_rows = par_matmul_chunk_rows();
    if d >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_qk_mr4_chunk(chunk, base_row, x, mapped, data_offset, row_size, n, ttype);
            });
    } else {
        matmul_qk_mr4_chunk(xout, 0, x, mapped, data_offset, row_size, n, ttype);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn vec_dot_q4_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q4_k_4rows_x86_avx2(x, r0, r1, r2, r3, n);
        }
    }

    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut q_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            q_off[r] = off + 16;
        }

        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][q_off[r] + l];
                    sums[r] += x0 * (a_lo[r] * (qv & 0x0f) as f32 - b_lo[r])
                        + x1 * (a_hi[r] * (qv >> 4) as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn vec_dot_q5_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q5_k_4rows_x86_avx2(x, r0, r1, r2, r3, n);
        }
    }

    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut qh = [&[][..]; 4];
        let mut ql_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            qh[r] = &rows[r][off + 16..off + 16 + QK_K / 8];
            ql_off[r] = off + 16 + QK_K / 8;
        }

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][ql_off[r] + l];
                    let lo = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    let hi = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                    sums[r] +=
                        x0 * (a_lo[r] * lo as f32 - b_lo[r]) + x1 * (a_hi[r] * hi as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                ql_off[r] += 32;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn vec_dot_q6_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q6_k_4rows_x86_avx2(x, r0, r1, r2, r3, n);
        }
    }

    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut ql_off = [0usize; 4];
        let mut qh_off = [0usize; 4];
        let mut sc_off = [0usize; 4];
        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 2 + QK_K / 4 + QK_K / 16));
            ql_off[r] = off;
            qh_off[r] = off + QK_K / 2;
            sc_off[r] = off + QK_K / 2 + QK_K / 4;
        }

        for n_outer in (0..QK_K).step_by(128) {
            let mut ql = [&[][..]; 4];
            let mut qh = [&[][..]; 4];
            let mut sc = [&[][..]; 4];
            for r in 0..4 {
                ql[r] = &rows[r][ql_off[r]..ql_off[r] + 64];
                qh[r] = &rows[r][qh_off[r]..qh_off[r] + 32];
                sc[r] = &rows[r][sc_off[r]..sc_off[r] + 8];
            }

            for l in 0..32 {
                let is = l / 16;
                let x0 = xb[n_outer + l];
                let x1 = xb[n_outer + 32 + l];
                let x2 = xb[n_outer + 64 + l];
                let x3 = xb[n_outer + 96 + l];
                for r in 0..4 {
                    let ql0 = ql[r][l];
                    let ql1 = ql[r][l + 32];
                    let qh0 = qh[r][l];
                    let q1 = (((ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4)) as i8) - 32;
                    let q2 = (((ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4)) as i8) - 32;
                    let q3 = (((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)) as i8) - 32;
                    let q4 = (((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)) as i8) - 32;
                    let s0 = d[r] * sc[r][is] as i8 as f32;
                    let s1 = d[r] * sc[r][is + 2] as i8 as f32;
                    let s2 = d[r] * sc[r][is + 4] as i8 as f32;
                    let s3 = d[r] * sc[r][is + 6] as i8 as f32;
                    sums[r] += x0 * (s0 * q1 as f32)
                        + x1 * (s1 * q2 as f32)
                        + x2 * (s2 * q3 as f32)
                        + x3 * (s3 * q4 as f32);
                }
            }
            for r in 0..4 {
                ql_off[r] += 64;
                qh_off[r] += 32;
                sc_off[r] += 8;
            }
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_k_4rows_x86_avx2(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut q_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            q_off[r] = off + 16;
        }

        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let x0 = &xb[j..j + 32];
            let x1 = &xb[j + 32..j + 64];
            let x0_sum = x0.iter().copied().sum::<f32>();
            let x1_sum = x1.iter().copied().sum::<f32>();
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
                let q = &rows[r][q_off[r]..q_off[r] + 32];
                let (dot_lo, dot_hi) =
                    dot_q4_nibbles_pair_avx2_ptr(x0.as_ptr(), x1.as_ptr(), q.as_ptr(), 32);
                sums[r] +=
                    a_lo[r] * dot_lo - b_lo[r] * x0_sum + a_hi[r] * dot_hi - b_hi[r] * x1_sum;
                q_off[r] += 32;
            }
            is += 2;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5_k_4rows_x86_avx2(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut qh = [&[][..]; 4];
        let mut ql_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            qh[r] = &rows[r][off + 16..off + 16 + QK_K / 8];
            ql_off[r] = off + 16 + QK_K / 8;
        }

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let x0 = &xb[j..j + 32];
            let x1 = &xb[j + 32..j + 64];
            let x0_sum = x0.iter().copied().sum::<f32>();
            let x1_sum = x1.iter().copied().sum::<f32>();
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;

                let ql = &rows[r][ql_off[r]..ql_off[r] + 32];
                let mut lo_vals = [0u8; 32];
                let mut hi_vals = [0u8; 32];
                for l in 0..32 {
                    let qv = ql[l];
                    lo_vals[l] = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    hi_vals[l] = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                }
                let dot_lo = dot_f32_u8_vals_avx2_ptr(x0.as_ptr(), lo_vals.as_ptr(), 32);
                let dot_hi = dot_f32_u8_vals_avx2_ptr(x1.as_ptr(), hi_vals.as_ptr(), 32);
                sums[r] +=
                    a_lo[r] * dot_lo - b_lo[r] * x0_sum + a_hi[r] * dot_hi - b_hi[r] * x1_sum;
                ql_off[r] += 32;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q6_k_4rows_x86_avx2(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut ql_off = [0usize; 4];
        let mut qh_off = [0usize; 4];
        let mut sc_off = [0usize; 4];
        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 2 + QK_K / 4 + QK_K / 16));
            ql_off[r] = off;
            qh_off[r] = off + QK_K / 2;
            sc_off[r] = off + QK_K / 2 + QK_K / 4;
        }

        for n_outer in (0..QK_K).step_by(128) {
            let x0 = &xb[n_outer..n_outer + 32];
            let x1 = &xb[n_outer + 32..n_outer + 64];
            let x2 = &xb[n_outer + 64..n_outer + 96];
            let x3 = &xb[n_outer + 96..n_outer + 128];
            for r in 0..4 {
                let ql = &rows[r][ql_off[r]..ql_off[r] + 64];
                let qh = &rows[r][qh_off[r]..qh_off[r] + 32];
                let sc = &rows[r][sc_off[r]..sc_off[r] + 8];
                let mut q1 = [0i8; 32];
                let mut q2 = [0i8; 32];
                let mut q3 = [0i8; 32];
                let mut q4 = [0i8; 32];

                for l in 0..32 {
                    let ql0 = ql[l];
                    let ql1 = ql[l + 32];
                    let qh0 = qh[l];
                    q1[l] = ((ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4)) as i8 - 32;
                    q2[l] = ((ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4)) as i8 - 32;
                    q3[l] = ((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)) as i8 - 32;
                    q4[l] = ((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)) as i8 - 32;
                }

                let dot1_lo = dot_f32_i8_vals_avx2_ptr(x0.as_ptr(), q1.as_ptr(), 16);
                let dot1_hi =
                    dot_f32_i8_vals_avx2_ptr(x0.as_ptr().add(16), q1.as_ptr().add(16), 16);
                let dot2_lo = dot_f32_i8_vals_avx2_ptr(x1.as_ptr(), q2.as_ptr(), 16);
                let dot2_hi =
                    dot_f32_i8_vals_avx2_ptr(x1.as_ptr().add(16), q2.as_ptr().add(16), 16);
                let dot3_lo = dot_f32_i8_vals_avx2_ptr(x2.as_ptr(), q3.as_ptr(), 16);
                let dot3_hi =
                    dot_f32_i8_vals_avx2_ptr(x2.as_ptr().add(16), q3.as_ptr().add(16), 16);
                let dot4_lo = dot_f32_i8_vals_avx2_ptr(x3.as_ptr(), q4.as_ptr(), 16);
                let dot4_hi =
                    dot_f32_i8_vals_avx2_ptr(x3.as_ptr().add(16), q4.as_ptr().add(16), 16);

                let s00 = d[r] * sc[0] as i8 as f32;
                let s01 = d[r] * sc[1] as i8 as f32;
                let s10 = d[r] * sc[2] as i8 as f32;
                let s11 = d[r] * sc[3] as i8 as f32;
                let s20 = d[r] * sc[4] as i8 as f32;
                let s21 = d[r] * sc[5] as i8 as f32;
                let s30 = d[r] * sc[6] as i8 as f32;
                let s31 = d[r] * sc[7] as i8 as f32;

                sums[r] += s00 * dot1_lo
                    + s01 * dot1_hi
                    + s10 * dot2_lo
                    + s11 * dot2_hi
                    + s20 * dot3_lo
                    + s21 * dot3_hi
                    + s30 * dot4_lo
                    + s31 * dot4_hi;
            }
            for r in 0..4 {
                ql_off[r] += 64;
                qh_off[r] += 32;
                sc_off[r] += 8;
            }
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn matmul_qk_mr4_chunk_x86(
    out: &mut [f32],
    base_row: usize,
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) {
    let total_rows = out.len();
    let mut i = 0usize;
    while i + 4 <= out.len() {
        x86_prefetch_row(
            mapped,
            data_offset,
            row_size,
            base_row + i,
            base_row.saturating_add(total_rows),
        );
        let row0_off = data_offset + (base_row + i) * row_size;
        let row1_off = row0_off + row_size;
        let row2_off = row1_off + row_size;
        let row3_off = row2_off + row_size;
        let r0 = &mapped[row0_off..row0_off + row_size];
        let r1 = &mapped[row1_off..row1_off + row_size];
        let r2 = &mapped[row2_off..row2_off + row_size];
        let r3 = &mapped[row3_off..row3_off + row_size];
        let sums = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
        out[i] = sums[0];
        out[i + 1] = sums[1];
        out[i + 2] = sums[2];
        out[i + 3] = sums[3];
        i += 4;
    }
    while i < out.len() {
        x86_prefetch_row(
            mapped,
            data_offset,
            row_size,
            base_row + i,
            base_row.saturating_add(total_rows),
        );
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn mr4_status_x86(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q4_K => &X86_Q4K_MR4_STATUS,
        GGML_TYPE_Q5_K => &X86_Q5K_MR4_STATUS,
        GGML_TYPE_Q6_K => &X86_Q6K_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn validate_qk_mr4_once_x86(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    let status = mr4_status_x86(ttype);
    match status.load(AtomicOrdering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }

    let r0 = &mapped[data_offset..data_offset + row_size];
    let r1 = &mapped[data_offset + row_size..data_offset + 2 * row_size];
    let r2 = &mapped[data_offset + 2 * row_size..data_offset + 3 * row_size];
    let r3 = &mapped[data_offset + 3 * row_size..data_offset + 4 * row_size];

    let mr4 = match ttype {
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q4_K => [
            vec_dot_q4_k(x, r0, n),
            vec_dot_q4_k(x, r1, n),
            vec_dot_q4_k(x, r2, n),
            vec_dot_q4_k(x, r3, n),
        ],
        GGML_TYPE_Q5_K => [
            vec_dot_q5_k(x, r0, n),
            vec_dot_q5_k(x, r1, n),
            vec_dot_q5_k(x, r2, n),
            vec_dot_q5_k(x, r3, n),
        ],
        GGML_TYPE_Q6_K => [
            vec_dot_q6_k(x, r0, n),
            vec_dot_q6_k(x, r1, n),
            vec_dot_q6_k(x, r2, n),
            vec_dot_q6_k(x, r3, n),
        ],
        _ => unreachable!(),
    };

    let mut ok = true;
    for i in 0..4 {
        let a = mr4[i];
        let b = scalar[i];
        let tol = 1e-4f32 * b.abs().max(1.0);
        if (a - b).abs() > tol {
            ok = false;
            break;
        }
    }

    status.store(if ok { 1 } else { 2 }, AtomicOrdering::Relaxed);
    if !ok {
        eprintln!(
            "Warning: disabling x86_64 MR4 kernel for type {} due to validation mismatch",
            ttype
        );
    }
    ok
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn try_matmul_qk_mr4_x86(
    xout: &mut [f32],
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    if !use_x86_qk_mr4() {
        return false;
    }
    if !matches!(ttype, GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K) {
        return false;
    }
    if n < QK_K || n % QK_K != 0 {
        return false;
    }

    let d = xout.len();
    if d < 4 {
        return false;
    }
    if !validate_qk_mr4_once_x86(x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }
    let chunk_rows = par_matmul_chunk_rows();
    if d >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_qk_mr4_chunk_x86(
                    chunk,
                    base_row,
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    ttype,
                );
            });
    } else {
        matmul_qk_mr4_chunk_x86(xout, 0, x, mapped, data_offset, row_size, n, ttype);
    }
    true
}

pub(crate) fn vec_dot_iq4_nl(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK4_NL;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let qs = &w[off + 2..off + 2 + QK4_NL / 2];
        let xb = &x[i * QK4_NL..(i + 1) * QK4_NL];
        let mut block_sum = 0.0;
        for j in 0..QK4_NL / 2 {
            block_sum += xb[j] * KVALUES_IQ4NL[(qs[j] & 0x0f) as usize] as f32;
            block_sum += xb[j + QK4_NL / 2] * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
        sum += block_sum * d;
    }
    sum
}

pub(crate) fn get_row_size(n_cols: usize, ttype: GgmlType) -> usize {
    let block_size = get_block_size(ttype);
    let type_size = get_type_size(ttype);
    (n_cols / block_size) * type_size
}

pub(crate) fn matmul_quantized(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
) -> Result<(), String> {
    let prof_t0 = prof_start();
    let d = qw.rows;
    let n = qw.cols;
    let row_size = get_row_size(n, qw.ttype);
    if xout.len() < d || x.len() < n {
        return Err("matmul shape mismatch".to_string());
    }
    let data_size = d
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = qw
        .data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    let data_offset = qw.data_offset;
    ensure_model_range(data_offset, data_size)?;
    macro_rules! run_rows {
        ($dot:path) => {{
            if d >= par_matmul_min_rows() {
                let chunk_rows = par_matmul_chunk_rows();
                xout[..d].par_chunks_mut(chunk_rows).enumerate().for_each(
                    |(chunk_idx, out_chunk)| {
                        let base_row = chunk_idx * chunk_rows;
                        for (j, out) in out_chunk.iter_mut().enumerate() {
                            #[cfg(target_arch = "x86_64")]
                            x86_prefetch_row(mapped, data_offset, row_size, base_row + j, d);
                            let row_off = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_off..row_off + row_size];
                            *out = $dot(x, row, n);
                        }
                    },
                );
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    #[cfg(target_arch = "x86_64")]
                    x86_prefetch_row(mapped, data_offset, row_size, i, d);
                    let row_off = data_offset + i * row_size;
                    let row = &mapped[row_off..row_off + row_size];
                    *out = $dot(x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => run_rows!(vec_dot_q4_0),
        GGML_TYPE_Q4_1 => run_rows!(vec_dot_q4_1),
        GGML_TYPE_Q5_0 => run_rows!(vec_dot_q5_0),
        GGML_TYPE_Q5_1 => run_rows!(vec_dot_q5_1),
        GGML_TYPE_Q8_0 => run_rows!(vec_dot_q8_0),
        GGML_TYPE_Q2_K => run_rows!(vec_dot_q2_k),
        GGML_TYPE_Q3_K => run_rows!(vec_dot_q3_k),
        GGML_TYPE_Q4_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_k);
            }
        }
        GGML_TYPE_Q5_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_k);
            }
        }
        GGML_TYPE_Q6_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => run_rows!(vec_dot_iq4_nl),
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 | 30 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ))
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

pub(crate) fn matmul_quantized_rows(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    row_start: usize,
    n_rows: usize,
    mapped: &[u8],
) -> Result<(), String> {
    let prof_t0 = prof_start();
    let d = n_rows;
    let n = qw.cols;
    let row_size = get_row_size(n, qw.ttype);
    if row_start + n_rows > qw.rows {
        return Err("matmul row window exceeds tensor rows".to_string());
    }
    if xout.len() < d || x.len() < n {
        return Err("matmul shape mismatch".to_string());
    }
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = d
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;
    macro_rules! run_rows {
        ($dot:path) => {{
            if d >= par_matmul_min_rows() {
                let chunk_rows = par_matmul_chunk_rows();
                xout[..d].par_chunks_mut(chunk_rows).enumerate().for_each(
                    |(chunk_idx, out_chunk)| {
                        let base_row = chunk_idx * chunk_rows;
                        for (j, out) in out_chunk.iter_mut().enumerate() {
                            #[cfg(target_arch = "x86_64")]
                            x86_prefetch_row(mapped, data_offset, row_size, base_row + j, d);
                            let row_start = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_start..row_start + row_size];
                            *out = $dot(x, row, n);
                        }
                    },
                );
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    #[cfg(target_arch = "x86_64")]
                    x86_prefetch_row(mapped, data_offset, row_size, i, d);
                    let row_start = data_offset + i * row_size;
                    let row = &mapped[row_start..row_start + row_size];
                    *out = $dot(x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => run_rows!(vec_dot_q4_0),
        GGML_TYPE_Q4_1 => run_rows!(vec_dot_q4_1),
        GGML_TYPE_Q5_0 => run_rows!(vec_dot_q5_0),
        GGML_TYPE_Q5_1 => run_rows!(vec_dot_q5_1),
        GGML_TYPE_Q8_0 => run_rows!(vec_dot_q8_0),
        GGML_TYPE_Q2_K => run_rows!(vec_dot_q2_k),
        GGML_TYPE_Q3_K => run_rows!(vec_dot_q3_k),
        GGML_TYPE_Q4_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_k);
            }
        }
        GGML_TYPE_Q5_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_k);
            }
        }
        GGML_TYPE_Q6_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => run_rows!(vec_dot_iq4_nl),
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 | 30 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ))
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

pub(crate) fn select_topk_softmax(
    logits: &[f32],
    k: usize,
    n_group: usize,
    topk_group: usize,
    normalize_topk: bool,
    scale: f32,
    scores_scratch: &mut Vec<f32>,
    selected_group_scratch: &mut Vec<bool>,
    group_scores_scratch: &mut Vec<f32>,
    rank_scratch: &mut Vec<usize>,
    out_indices: &mut [usize],
    out_weights: &mut [f32],
) -> usize {
    let top_k = k.max(1).min(logits.len());
    if scores_scratch.len() < logits.len() {
        scores_scratch.resize(logits.len(), 0.0);
    }
    let scores = &mut scores_scratch[..logits.len()];
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max_logit).exp();
        scores[i] = e;
        sum += e;
    }
    let inv_sum = 1.0 / sum.max(f32::MIN_POSITIVE);
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }

    let use_grouped = n_group > 1 && topk_group < n_group && logits.len() % n_group == 0;
    let group_size = if use_grouped {
        logits.len() / n_group
    } else {
        logits.len()
    };

    let selected_group_len = n_group.max(1);
    if selected_group_scratch.len() < selected_group_len {
        selected_group_scratch.resize(selected_group_len, true);
    }
    let selected_group = &mut selected_group_scratch[..selected_group_len];
    selected_group.fill(true);

    if use_grouped {
        if group_scores_scratch.len() < n_group {
            group_scores_scratch.resize(n_group, 0.0);
        }
        let group_scores = &mut group_scores_scratch[..n_group];
        for g in 0..n_group {
            let start = g * group_size;
            let end = start + group_size;
            let mut best1 = f32::NEG_INFINITY;
            let mut best2 = f32::NEG_INFINITY;
            for &s in &scores[start..end] {
                if s > best1 {
                    best2 = best1;
                    best1 = s;
                } else if s > best2 {
                    best2 = s;
                }
            }
            group_scores[g] = best1 + if best2.is_finite() { best2 } else { 0.0 };
        }

        selected_group.fill(false);
        if rank_scratch.len() < n_group {
            rank_scratch.resize(n_group, 0);
        }
        let rank = &mut rank_scratch[..n_group];
        for (i, r) in rank.iter_mut().enumerate() {
            *r = i;
        }
        rank.sort_by(|&a, &b| {
            group_scores[b]
                .partial_cmp(&group_scores[a])
                .unwrap_or(Ordering::Equal)
        });
        for &g in rank.iter().take(topk_group.max(1).min(n_group)) {
            selected_group[g] = true;
        }
    }

    for i in 0..top_k {
        out_weights[i] = f32::NEG_INFINITY;
        out_indices[i] = 0;
    }
    let mut count = 0usize;

    for (idx, &v) in scores.iter().enumerate() {
        if use_grouped {
            let g = idx / group_size;
            if !selected_group[g] {
                continue;
            }
        }
        if count < top_k {
            let mut ins = count;
            while ins > 0 && v > out_weights[ins - 1] {
                out_weights[ins] = out_weights[ins - 1];
                out_indices[ins] = out_indices[ins - 1];
                ins -= 1;
            }
            out_weights[ins] = v;
            out_indices[ins] = idx;
            count += 1;
            continue;
        }

        if v <= out_weights[top_k - 1] {
            continue;
        }

        out_weights[top_k - 1] = v;
        out_indices[top_k - 1] = idx;
        let mut pos = top_k - 1;
        while pos > 0 && out_weights[pos] > out_weights[pos - 1] {
            out_weights.swap(pos, pos - 1);
            out_indices.swap(pos, pos - 1);
            pos -= 1;
        }
    }

    if count == 0 {
        return 0;
    }

    if top_k > 1 && normalize_topk {
        let mut sum_selected = 0.0f32;
        for i in 0..count {
            sum_selected += out_weights[i];
        }
        let inv = 1.0 / sum_selected.max(f32::MIN_POSITIVE);
        for i in 0..count {
            out_weights[i] *= inv;
        }
    }

    for i in 0..count {
        out_weights[i] *= scale;
    }

    count
}

pub(crate) fn select_topk_sigmoid(
    logits: &[f32],
    k: usize,
    n_group: usize,
    topk_group: usize,
    normalize_topk: bool,
    scale: f32,
    scores_scratch: &mut Vec<f32>,
    selected_group_scratch: &mut Vec<bool>,
    group_scores_scratch: &mut Vec<f32>,
    rank_scratch: &mut Vec<usize>,
    out_indices: &mut [usize],
    out_weights: &mut [f32],
) -> usize {
    let top_k = k.max(1).min(logits.len());
    if scores_scratch.len() < logits.len() {
        scores_scratch.resize(logits.len(), 0.0);
    }
    let scores = &mut scores_scratch[..logits.len()];
    for (i, &v) in logits.iter().enumerate() {
        scores[i] = 1.0 / (1.0 + (-v).exp());
    }

    let use_grouped = n_group > 1 && topk_group < n_group && logits.len() % n_group == 0;
    let group_size = if use_grouped {
        logits.len() / n_group
    } else {
        logits.len()
    };

    let selected_group_len = n_group.max(1);
    if selected_group_scratch.len() < selected_group_len {
        selected_group_scratch.resize(selected_group_len, true);
    }
    let selected_group = &mut selected_group_scratch[..selected_group_len];
    selected_group.fill(true);

    if use_grouped {
        if group_scores_scratch.len() < n_group {
            group_scores_scratch.resize(n_group, 0.0);
        }
        let group_scores = &mut group_scores_scratch[..n_group];
        for g in 0..n_group {
            let start = g * group_size;
            let end = start + group_size;
            let mut best1 = f32::NEG_INFINITY;
            let mut best2 = f32::NEG_INFINITY;
            for &s in &scores[start..end] {
                if s > best1 {
                    best2 = best1;
                    best1 = s;
                } else if s > best2 {
                    best2 = s;
                }
            }
            group_scores[g] = best1 + if best2.is_finite() { best2 } else { 0.0 };
        }

        selected_group.fill(false);
        if rank_scratch.len() < n_group {
            rank_scratch.resize(n_group, 0);
        }
        let rank = &mut rank_scratch[..n_group];
        for (i, r) in rank.iter_mut().enumerate() {
            *r = i;
        }
        rank.sort_by(|&a, &b| {
            group_scores[b]
                .partial_cmp(&group_scores[a])
                .unwrap_or(Ordering::Equal)
        });
        for &g in rank.iter().take(topk_group.max(1).min(n_group)) {
            selected_group[g] = true;
        }
    }

    for i in 0..top_k {
        out_weights[i] = f32::NEG_INFINITY;
        out_indices[i] = 0;
    }
    let mut count = 0usize;

    for (idx, &v) in scores.iter().enumerate() {
        if use_grouped {
            let g = idx / group_size;
            if !selected_group[g] {
                continue;
            }
        }
        if count < top_k {
            let mut ins = count;
            while ins > 0 && v > out_weights[ins - 1] {
                out_weights[ins] = out_weights[ins - 1];
                out_indices[ins] = out_indices[ins - 1];
                ins -= 1;
            }
            out_weights[ins] = v;
            out_indices[ins] = idx;
            count += 1;
            continue;
        }

        if v <= out_weights[top_k - 1] {
            continue;
        }

        out_weights[top_k - 1] = v;
        out_indices[top_k - 1] = idx;
        let mut pos = top_k - 1;
        while pos > 0 && out_weights[pos] > out_weights[pos - 1] {
            out_weights.swap(pos, pos - 1);
            out_indices.swap(pos, pos - 1);
            pos -= 1;
        }
    }

    if count == 0 {
        return 0;
    }

    if top_k > 1 && normalize_topk {
        let mut sum_selected = 0.0f32;
        for i in 0..count {
            sum_selected += out_weights[i];
        }
        let inv = 1.0 / sum_selected.max(f32::MIN_POSITIVE);
        for i in 0..count {
            out_weights[i] *= inv;
        }
    }

    for i in 0..count {
        out_weights[i] *= scale;
    }

    count
}
