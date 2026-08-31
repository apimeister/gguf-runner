#![allow(clippy::needless_range_loop)]
#![allow(unsafe_op_in_unsafe_fn)]

use crate::engine::io::{
    bf16_to_fp32, fp16_to_fp32, fp32_to_bf16, read_f32_le, read_u16_le, read_u32_le,
};
use crate::engine::profiling::{PROF_MATMUL_NS, prof_end, prof_start};
#[cfg(target_arch = "aarch64")]
use crate::engine::switches::{
    AARCH64_IQ4_NL_MR4_STATUS, AARCH64_Q2K_MR4_STATUS, AARCH64_Q3K_MR4_STATUS,
    AARCH64_Q4_0_MR4_STATUS, AARCH64_Q4_1_MR4_STATUS, AARCH64_Q4K_MR4_STATUS,
    AARCH64_Q5_0_MR4_STATUS, AARCH64_Q5_1_MR4_STATUS, AARCH64_Q5K_MR4_STATUS,
    AARCH64_Q6K_MR4_STATUS, AARCH64_Q8_0_MR2_STATUS, aarch64_matmul_prefetch_rows,
    use_aarch64_bf16, use_aarch64_dotprod_q8, use_aarch64_i8mm_q8, use_aarch64_qk_mr4,
};
#[cfg(target_arch = "x86_64")]
use crate::engine::switches::{
    X86_IQ4_NL_MR4_STATUS, X86_Q2K_MR4_STATUS, X86_Q3K_MR4_STATUS, X86_Q4_0_MR4_STATUS,
    X86_Q4_1_MR4_STATUS, X86_Q4K_MR4_STATUS, X86_Q5_0_MR4_STATUS, X86_Q5_1_MR4_STATUS,
    X86_Q5K_MR4_STATUS, X86_Q6K_MR4_STATUS, use_x86_avx_vnni, use_x86_avx2_fma,
    use_x86_avx512_vnni_q8, use_x86_f16c, use_x86_qk_mr4,
};
use crate::engine::switches::{par_matmul_chunk_rows, par_matmul_min_rows};
use crate::engine::types::{
    GGML_TYPE_BF16, GGML_TYPE_BIN1_40, GGML_TYPE_BIN1_41, GGML_TYPE_F16, GGML_TYPE_F32,
    GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    GgmlType, KVALUES_IQ4NL, QK_BIN1, QK_K, QK4_0, QK4_1, QK4_NL, QK5_0, QK5_1, QK8_0,
    QuantizedTensor, ensure_model_range,
};

/// type_size for BIN1 (types 40/41): f16 scale (2 bytes) + 16 bytes packed 1-bit values
/// for 128 elements. Layout follows all other GGML types: scale first, data after.
const BIN1_TYPE_SIZE: usize = 18;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSliceMut,
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::Ordering;
#[cfg(target_os = "macos")]
use std::ffi::c_int;
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

#[cfg(target_arch = "x86_64")]
const X86_MATMUL_PREFETCH_ROWS: usize = 6;

fn kernel_validation_warnings_enabled() -> bool {
    std::env::var("GGUF_KERNEL_VALIDATION_WARNINGS")
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(crate) fn get_block_size(ttype: GgmlType) -> usize {
    match ttype.0 {
        GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16 => 1,
        GGML_TYPE_Q4_0 => QK4_0,
        GGML_TYPE_Q4_1 => QK4_1,
        GGML_TYPE_Q5_0 => QK5_0,
        GGML_TYPE_Q5_1 => QK5_1,
        GGML_TYPE_Q8_0 => QK8_0,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => QK_K,
        GGML_TYPE_IQ4_NL => QK4_NL,
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => QK_BIN1,
        _ => 1,
    }
}

pub(crate) fn get_type_size(ttype: GgmlType) -> usize {
    match ttype.0 {
        GGML_TYPE_F32 => 4,
        GGML_TYPE_F16 | GGML_TYPE_BF16 => 2,
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
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => BIN1_TYPE_SIZE,
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn aarch64_prefetch_row(
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    row_idx: usize,
    total_rows: usize,
) {
    let dist = aarch64_matmul_prefetch_rows();
    if dist == 0 {
        return;
    }
    let Some(pf_row) = row_idx.checked_add(dist) else {
        return;
    };
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
        let ptr = unsafe { mapped.as_ptr().add(pf_off) };
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, readonly, preserves_flags)
            );
        }
    }
}

#[inline(always)]
fn matmul_prefetch_row(
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    row_idx: usize,
    total_rows: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        x86_prefetch_row(mapped, data_offset, row_size, row_idx, total_rows);
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64_prefetch_row(mapped, data_offset, row_size, row_idx, total_rows);
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
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
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
                let q1 = (((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4)) as i8) - 32;
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

/// Dequantise a 1-bit binary quantisation row (GGML types 40/41).
/// Block layout (128 elements): [2 bytes f16 scale][16 bytes packed bits (LSB-first)].
/// Each bit maps to +scale (1) or -scale (0). Scale is f16, consistent with all GGML types.
pub(crate) fn dequantize_row_bin1(src: &[u8], dst: &mut [f32], k: usize) {
    let nb = k / QK_BIN1;
    for i in 0..nb {
        let off = i * BIN1_TYPE_SIZE;
        let scale = fp16_to_fp32(read_u16_le(src, off));
        let bits = &src[off + 2..off + 18];
        let base = i * QK_BIN1;
        for j in 0..QK_BIN1 {
            let bit = (bits[j >> 3] >> (j & 7)) & 1;
            dst[base + j] = if bit != 0 { scale } else { -scale };
        }
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
        GGML_TYPE_BF16 => dequantize_row_bf16(src, &mut dst, n_elements),
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => dequantize_row_bin1(src, &mut dst, n_elements),
        _ => return Err(format!("unsupported quantization type: {}", ttype.0)),
    }
    Ok(dst)
}

#[inline]
fn dequantize_row_into(
    ttype: GgmlType,
    src: &[u8],
    dst: &mut [f32],
    k: usize,
) -> Result<(), String> {
    match ttype.0 {
        GGML_TYPE_F32 => {
            for (i, slot) in dst[..k].iter_mut().enumerate() {
                *slot = read_f32_le(src, i * 4);
            }
        }
        GGML_TYPE_F16 => dequantize_row_f16(src, dst, k),
        GGML_TYPE_Q4_0 => dequantize_row_q4_0(src, dst, k),
        GGML_TYPE_Q4_1 => dequantize_row_q4_1(src, dst, k),
        GGML_TYPE_Q5_0 => dequantize_row_q5_0(src, dst, k),
        GGML_TYPE_Q5_1 => dequantize_row_q5_1(src, dst, k),
        GGML_TYPE_Q8_0 => dequantize_row_q8_0(src, dst, k),
        GGML_TYPE_Q2_K => dequantize_row_q2_k(src, dst, k),
        GGML_TYPE_Q3_K => dequantize_row_q3_k(src, dst, k),
        GGML_TYPE_Q4_K => dequantize_row_q4_k(src, dst, k),
        GGML_TYPE_Q5_K => dequantize_row_q5_k(src, dst, k),
        GGML_TYPE_Q6_K => dequantize_row_q6_k(src, dst, k),
        GGML_TYPE_IQ4_NL => dequantize_row_iq4_nl(src, dst, k),
        GGML_TYPE_BF16 => dequantize_row_bf16(src, dst, k),
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => dequantize_row_bin1(src, dst, k),
        _ => {
            return Err(format!(
                "unsupported quantization type in batched matmul: {}",
                ttype.0
            ));
        }
    }
    Ok(())
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
        let packed = _mm_loadu_si128(w.add(i * 2) as *const __m128i);
        let widened = _mm256_cvtepu16_epi32(packed);
        let shifted = _mm256_slli_epi32::<16>(widened);
        let wv = _mm256_castsi256_ps(shifted);
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
#[target_feature(enable = "ssse3,avx2,fma")]
unsafe fn dot_iq4_nl_nibbles_pair_avx2_ptr(
    x_lo: *const f32,
    x_hi: *const f32,
    q: *const u8,
    n: usize,
) -> (f32, f32) {
    let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nib_mask = _mm_set1_epi8(0x0f);
    let mut i = 0usize;
    let mut acc_lo = _mm256_setzero_ps();
    let mut acc_hi = _mm256_setzero_ps();

    while i + 8 <= n {
        let xv_lo = _mm256_loadu_ps(x_lo.add(i));
        let xv_hi = _mm256_loadu_ps(x_hi.add(i));
        let q8 = _mm_loadl_epi64(q.add(i) as *const __m128i);
        let lo_idx = _mm_and_si128(q8, nib_mask);
        let hi_idx = _mm_and_si128(_mm_srli_epi16(q8, 4), nib_mask);
        let lo_vals = _mm_shuffle_epi8(table, lo_idx);
        let hi_vals = _mm_shuffle_epi8(table, hi_idx);
        let q_lo_f = cvt_i8x8_to_f32x8(lo_vals);
        let q_hi_f = cvt_i8x8_to_f32x8(hi_vals);
        acc_lo = _mm256_fmadd_ps(xv_lo, q_lo_f, acc_lo);
        acc_hi = _mm256_fmadd_ps(xv_hi, q_hi_f, acc_hi);
        i += 8;
    }

    let mut sum_lo = hsum256_ps(acc_lo);
    let mut sum_hi = hsum256_ps(acc_hi);
    while i < n {
        let qv = *q.add(i);
        sum_lo += *x_lo.add(i) * KVALUES_IQ4NL[(qv & 0x0f) as usize] as f32;
        sum_hi += *x_hi.add(i) * KVALUES_IQ4NL[(qv >> 4) as usize] as f32;
        i += 1;
    }
    (sum_lo, sum_hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32_32_avx2_ptr(x: *const f32) -> f32 {
    let mut acc = _mm256_setzero_ps();
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(x));
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(x.add(8)));
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(x.add(16)));
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(x.add(24)));
    hsum256_ps(acc)
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

/// Load 4 fp16 values from `ptr` and convert to float32x4_t using the FCVTL instruction.
/// FCVTL is base AArch64 NEON (ARMv8-A), no extra CPU feature required.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_f16x4_as_f32x4(ptr: *const u8) -> float32x4_t {
    let result: float32x4_t;
    core::arch::asm!(
        "ld1 {{v8.4h}}, [{ptr}]",
        "fcvtl v8.4s, v8.4h",
        ptr = in(reg) ptr,
        out("v8") result,
        options(nostack, pure, readonly),
    );
    result
}

/// Load 4 bf16 values and expand them to f32 by widening and shifting their bit patterns.
/// USHLL/SHL are base AArch64 NEON; native FEAT_BF16 arithmetic is not required.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_bf16x4_as_f32x4(ptr: *const u8) -> float32x4_t {
    let result: float32x4_t;
    core::arch::asm!(
        "ld1 {{v8.4h}}, [{ptr}]",
        "ushll v8.4s, v8.4h, #0",
        "shl v8.4s, v8.4s, #16",
        ptr = in(reg) ptr,
        out("v8") result,
        options(nostack, pure, readonly),
    );
    result
}

#[inline(always)]
pub(crate) fn vec_dot_f16(x: &[f32], w: &[u8], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Process 8 fp16 weights per iteration: two FCVTL loads + two FMLA
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let xv0 = vld1q_f32(x.as_ptr().add(i));
            let xv1 = vld1q_f32(x.as_ptr().add(i + 4));
            let wv0 = load_f16x4_as_f32x4(w.as_ptr().add(i * 2));
            let wv1 = load_f16x4_as_f32x4(w.as_ptr().add((i + 4) * 2));
            acc0 = vfmaq_f32(acc0, xv0, wv0);
            acc1 = vfmaq_f32(acc1, xv1, wv1);
            i += 8;
        }
        let mut acc = vaddq_f32(acc0, acc1);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = load_f16x4_as_f32x4(w.as_ptr().add(i * 2));
            acc = vfmaq_f32(acc, xv, wv);
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
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let xv0 = vld1q_f32(x.as_ptr().add(i));
            let xv1 = vld1q_f32(x.as_ptr().add(i + 4));
            let wv0 = load_bf16x4_as_f32x4(w.as_ptr().add(i * 2));
            let wv1 = load_bf16x4_as_f32x4(w.as_ptr().add((i + 4) * 2));
            acc0 = vfmaq_f32(acc0, xv0, wv0);
            acc1 = vfmaq_f32(acc1, xv1, wv1);
            i += 8;
        }
        let mut acc = vaddq_f32(acc0, acc1);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = load_bf16x4_as_f32x4(w.as_ptr().add(i * 2));
            acc = vfmaq_f32(acc, xv, wv);
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
    }
    while i < n {
        sum += x[i] * bf16_to_fp32(read_u16_le(w, i * 2));
        i += 1;
    }
    sum
}

/// Native Arm BF16 dot for an activation vector that has already been packed
/// to BF16. Packing is deliberately outside this function so a token's
/// activation can be reused across every weight row in a matrix.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "bf16")]
unsafe fn vec_dot_bf16_bfdot(x: &[u16], w: &[u8], n: usize) -> f32 {
    debug_assert!(x.len() >= n);
    debug_assert!(w.len() >= n * 2);

    let n16 = n & !15;
    let mut sum = 0.0f32;
    if n16 > 0 {
        let result: float32x4_t;
        let x_ptr = x.as_ptr();
        let w_ptr = w.as_ptr();
        let remaining = n16;
        core::arch::asm!(
            "movi v8.4s, #0",
            "movi v11.4s, #0",
            "2:",
            "ld1 {{v9.8h}}, [{x_ptr}], #16",
            "ld1 {{v10.8h}}, [{w_ptr}], #16",
            "bfdot v8.4s, v9.8h, v10.8h",
            "ld1 {{v9.8h}}, [{x_ptr}], #16",
            "ld1 {{v10.8h}}, [{w_ptr}], #16",
            "bfdot v11.4s, v9.8h, v10.8h",
            "subs {remaining}, {remaining}, #16",
            "b.ne 2b",
            "fadd v8.4s, v8.4s, v11.4s",
            x_ptr = inout(reg) x_ptr => _,
            w_ptr = inout(reg) w_ptr => _,
            remaining = inout(reg) remaining => _,
            out("v8") result,
            lateout("v9") _,
            lateout("v10") _,
            lateout("v11") _,
            options(nostack, readonly),
        );
        sum = vaddvq_f32(result);
    }

    for i in n16..n {
        sum += bf16_to_fp32(x[i]) * bf16_to_fp32(read_u16_le(w, i * 2));
    }
    sum
}

/// Dot product for 1-bit binary quantisation (types 40/41).
/// Block layout (128 elements): [2 bytes f16 scale][16 bytes packed bits].
/// Each bit 1 → +scale, bit 0 → -scale.
pub(crate) fn vec_dot_bin1(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_BIN1;
    let mut sum = 0.0f32;
    for i in 0..nb {
        let off = i * BIN1_TYPE_SIZE;
        let scale = fp16_to_fp32(read_u16_le(w, off));
        let bits = &w[off + 2..off + 18];
        let base = i * QK_BIN1;
        for j in 0..QK_BIN1 {
            let bit = (bits[j >> 3] >> (j & 7)) & 1;
            let weight = if bit != 0 { scale } else { -scale };
            sum += x[base + j] * weight;
        }
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sum_f32_32_neon_ptr(x: *const f32) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    acc = vaddq_f32(acc, vld1q_f32(x));
    acc = vaddq_f32(acc, vld1q_f32(x.add(4)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(8)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(12)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(16)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(20)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(24)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(28)));
    vaddvq_f32(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q4_nibbles_pair_16_neon_ptr(
    x_lo: *const f32,
    x_hi: *const f32,
    q: *const u8,
) -> (f32, f32) {
    let qv = vld1q_u8(q);
    let lo8 = vandq_u8(qv, vdupq_n_u8(0x0f));
    let hi8 = vshrq_n_u8(qv, 4);

    let lo16_lo = vmovl_u8(vget_low_u8(lo8));
    let lo16_hi = vmovl_u8(vget_high_u8(lo8));
    let hi16_lo = vmovl_u8(vget_low_u8(hi8));
    let hi16_hi = vmovl_u8(vget_high_u8(hi8));

    let lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_lo)));
    let lo_f1 = vcvtq_f32_u32(vmovl_high_u16(lo16_lo));
    let lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_hi)));
    let lo_f3 = vcvtq_f32_u32(vmovl_high_u16(lo16_hi));

    let hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_lo)));
    let hi_f1 = vcvtq_f32_u32(vmovl_high_u16(hi16_lo));
    let hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_hi)));
    let hi_f3 = vcvtq_f32_u32(vmovl_high_u16(hi16_hi));

    let mut acc_lo = vdupq_n_f32(0.0);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo), lo_f0);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo.add(4)), lo_f1);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo.add(8)), lo_f2);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo.add(12)), lo_f3);

    let mut acc_hi = vdupq_n_f32(0.0);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi), hi_f0);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi.add(4)), hi_f1);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi.add(8)), hi_f2);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi.add(12)), hi_f3);

    (vaddvq_f32(acc_lo), vaddvq_f32(acc_hi))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_iq4_nl_nibbles_16_neon_ptr(
    x_lo: *const f32,
    x_hi: *const f32,
    q: *const u8,
) -> (f32, f32) {
    let table = vld1q_s8(KVALUES_IQ4NL.as_ptr());
    let qv = vld1q_u8(q);
    let lo_idx = vandq_u8(qv, vdupq_n_u8(0x0f));
    let hi_idx = vshrq_n_u8(qv, 4);
    let lo_vals = vqtbl1q_s8(table, lo_idx);
    let hi_vals = vqtbl1q_s8(table, hi_idx);

    let lo16_lo = vmovl_s8(vget_low_s8(lo_vals));
    let lo16_hi = vmovl_s8(vget_high_s8(lo_vals));
    let hi16_lo = vmovl_s8(vget_low_s8(hi_vals));
    let hi16_hi = vmovl_s8(vget_high_s8(hi_vals));

    let lo_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_lo)));
    let lo_f1 = vcvtq_f32_s32(vmovl_high_s16(lo16_lo));
    let lo_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_hi)));
    let lo_f3 = vcvtq_f32_s32(vmovl_high_s16(lo16_hi));

    let hi_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_lo)));
    let hi_f1 = vcvtq_f32_s32(vmovl_high_s16(hi16_lo));
    let hi_f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_hi)));
    let hi_f3 = vcvtq_f32_s32(vmovl_high_s16(hi16_hi));

    let mut acc_lo = vdupq_n_f32(0.0);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo), lo_f0);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo.add(4)), lo_f1);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo.add(8)), lo_f2);
    acc_lo = vfmaq_f32(acc_lo, vld1q_f32(x_lo.add(12)), lo_f3);

    let mut acc_hi = vdupq_n_f32(0.0);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi), hi_f0);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi.add(4)), hi_f1);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi.add(8)), hi_f2);
    acc_hi = vfmaq_f32(acc_hi, vld1q_f32(x_hi.add(12)), hi_f3);

    (vaddvq_f32(acc_lo), vaddvq_f32(acc_hi))
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn vec_dot_q4_0_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK4_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK4_0;
        let x_ptr = unsafe { x.as_ptr().add(x_base) };
        let x_sum = unsafe { sum_f32_32_neon_ptr(x_ptr) };
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let q = rows[r][off + 2..off + 2 + QK4_0 / 2].as_ptr();
            let (lo, hi) =
                unsafe { dot_q4_nibbles_pair_16_neon_ptr(x_ptr, x_ptr.add(QK4_0 / 2), q) };
            sums[r] += (lo + hi - 8.0 * x_sum) * d;
        }
    }

    sums
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn vec_dot_q4_1_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK4_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK4_1;
        let x_ptr = unsafe { x.as_ptr().add(x_base) };
        let x_sum = unsafe { sum_f32_32_neon_ptr(x_ptr) };
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let m = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            let q = rows[r][off + 4..off + 4 + QK4_1 / 2].as_ptr();
            let (lo, hi) =
                unsafe { dot_q4_nibbles_pair_16_neon_ptr(x_ptr, x_ptr.add(QK4_1 / 2), q) };
            sums[r] += (lo + hi) * d + x_sum * m;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_q4_0_4rows_x86(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q4_0_4rows_x86_avx2(x, w0, w1, w2, w3, n);
        }
    }
    [
        vec_dot_q4_0(x, w0, n),
        vec_dot_q4_0(x, w1, n),
        vec_dot_q4_0(x, w2, n),
        vec_dot_q4_0(x, w3, n),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_4rows_x86_avx2(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK4_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK4_0;
        let x_ptr = x.as_ptr().add(x_base);
        let x_sum = sum_f32_32_avx2_ptr(x_ptr);
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let q = rows[r][off + 2..off + 2 + QK4_0 / 2].as_ptr();
            let (lo, hi) = dot_q4_nibbles_pair_avx2_ptr(x_ptr, x_ptr.add(QK4_0 / 2), q, QK4_0 / 2);
            sums[r] += (lo + hi - 8.0 * x_sum) * d;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_q4_1_4rows_x86(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q4_1_4rows_x86_avx2(x, w0, w1, w2, w3, n);
        }
    }
    [
        vec_dot_q4_1(x, w0, n),
        vec_dot_q4_1(x, w1, n),
        vec_dot_q4_1(x, w2, n),
        vec_dot_q4_1(x, w3, n),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_1_4rows_x86_avx2(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK4_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK4_1;
        let x_ptr = x.as_ptr().add(x_base);
        let x_sum = sum_f32_32_avx2_ptr(x_ptr);
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let m = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            let q = rows[r][off + 4..off + 4 + QK4_1 / 2].as_ptr();
            let (lo, hi) = dot_q4_nibbles_pair_avx2_ptr(x_ptr, x_ptr.add(QK4_1 / 2), q, QK4_1 / 2);
            sums[r] += (lo + hi) * d + x_sum * m;
        }
    }

    sums
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

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn dot_q5_high_bits_32_ptr(x: *const f32, qh: u32) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..16 {
        if ((qh >> j) & 1) != 0 {
            sum += *x.add(j);
        }
        if ((qh >> (j + 16)) & 1) != 0 {
            sum += *x.add(j + 16);
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn vec_dot_q5_0_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK5_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK5_0;
        let x_ptr = unsafe { x.as_ptr().add(x_base) };
        let x_sum = unsafe { sum_f32_32_neon_ptr(x_ptr) };
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let qh = read_u32_le(rows[r], off + 2);
            let q = rows[r][off + 6..off + 6 + QK5_0 / 2].as_ptr();
            let (lo, hi) =
                unsafe { dot_q4_nibbles_pair_16_neon_ptr(x_ptr, x_ptr.add(QK5_0 / 2), q) };
            let high = unsafe { dot_q5_high_bits_32_ptr(x_ptr, qh) };
            sums[r] += (lo + hi + 16.0 * high - 16.0 * x_sum) * d;
        }
    }

    sums
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn vec_dot_q5_1_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK5_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK5_1;
        let x_ptr = unsafe { x.as_ptr().add(x_base) };
        let x_sum = unsafe { sum_f32_32_neon_ptr(x_ptr) };
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let m = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            let qh = read_u32_le(rows[r], off + 4);
            let q = rows[r][off + 8..off + 8 + QK5_1 / 2].as_ptr();
            let (lo, hi) =
                unsafe { dot_q4_nibbles_pair_16_neon_ptr(x_ptr, x_ptr.add(QK5_1 / 2), q) };
            let high = unsafe { dot_q5_high_bits_32_ptr(x_ptr, qh) };
            sums[r] += (lo + hi + 16.0 * high) * d + x_sum * m;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_q5_0_4rows_x86(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q5_0_4rows_x86_avx2(x, w0, w1, w2, w3, n);
        }
    }
    [
        vec_dot_q5_0(x, w0, n),
        vec_dot_q5_0(x, w1, n),
        vec_dot_q5_0(x, w2, n),
        vec_dot_q5_0(x, w3, n),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5_0_4rows_x86_avx2(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK5_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK5_0;
        let x_ptr = x.as_ptr().add(x_base);
        let x_sum = sum_f32_32_avx2_ptr(x_ptr);
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let qh = read_u32_le(rows[r], off + 2);
            let q = rows[r][off + 6..off + 6 + QK5_0 / 2].as_ptr();
            let (lo, hi) = dot_q4_nibbles_pair_avx2_ptr(x_ptr, x_ptr.add(QK5_0 / 2), q, QK5_0 / 2);
            let high = dot_q5_high_bits_32_ptr(x_ptr, qh);
            sums[r] += (lo + hi + 16.0 * high - 16.0 * x_sum) * d;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_q5_1_4rows_x86(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q5_1_4rows_x86_avx2(x, w0, w1, w2, w3, n);
        }
    }
    [
        vec_dot_q5_1(x, w0, n),
        vec_dot_q5_1(x, w1, n),
        vec_dot_q5_1(x, w2, n),
        vec_dot_q5_1(x, w3, n),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5_1_4rows_x86_avx2(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK5_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK5_1;
        let x_ptr = x.as_ptr().add(x_base);
        let x_sum = sum_f32_32_avx2_ptr(x_ptr);
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let m = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            let qh = read_u32_le(rows[r], off + 4);
            let q = rows[r][off + 8..off + 8 + QK5_1 / 2].as_ptr();
            let (lo, hi) = dot_q4_nibbles_pair_avx2_ptr(x_ptr, x_ptr.add(QK5_1 / 2), q, QK5_1 / 2);
            let high = dot_q5_high_bits_32_ptr(x_ptr, qh);
            sums[r] += (lo + hi + 16.0 * high) * d + x_sum * m;
        }
    }

    sums
}

pub(crate) fn vec_dot_q8_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if use_aarch64_dotprod_q8() {
        unsafe {
            return vec_dot_q8_0_dotprod(x, w, n);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q8_0_x86_avx2(x, w, n);
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
#[inline(always)]
unsafe fn sum_i8_16_ptr(v: *const i8) -> i32 {
    let mut sum = 0i32;
    for i in 0..16 {
        sum += *v.add(i) as i32;
    }
    sum
}

#[inline(always)]
fn quantize_f32_block_i8_32(src: &[f32], dst: &mut [i8; QK8_0]) -> f32 {
    debug_assert_eq!(src.len(), QK8_0);
    let mut abs_max = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > abs_max {
            abs_max = a;
        }
    }
    if abs_max == 0.0 {
        dst.fill(0);
        return 0.0;
    }
    let inv_scale = 127.0 / abs_max;
    for i in 0..QK8_0 {
        dst[i] = (src[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    abs_max / 127.0
}

pub(crate) struct MatmulActivationScratch {
    q8_scales: Vec<f32>,
    #[cfg(target_arch = "x86_64")]
    q8_f32_sums: Vec<f32>,
    #[cfg(target_arch = "x86_64")]
    q8_f32_sums_lo: Vec<f32>,
    #[cfg(target_arch = "x86_64")]
    q8_f32_sums_hi: Vec<f32>,
    #[cfg(target_arch = "x86_64")]
    q8_i8_sums_lo: Vec<i32>,
    #[cfg(target_arch = "x86_64")]
    q8_i8_sums_hi: Vec<i32>,
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    q8_x_i8: Vec<i8>,
    #[cfg(target_arch = "x86_64")]
    q8_x_u8: Vec<u8>,
}

impl MatmulActivationScratch {
    pub(crate) fn new() -> Self {
        Self {
            q8_scales: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            q8_f32_sums: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            q8_f32_sums_lo: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            q8_f32_sums_hi: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            q8_i8_sums_lo: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            q8_i8_sums_hi: Vec::new(),
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            q8_x_i8: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            q8_x_u8: Vec::new(),
        }
    }
}

impl Default for MatmulActivationScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct Q8ActivationPrequant<'a> {
    scales: &'a [f32],
    #[cfg(target_arch = "x86_64")]
    f32_sums: &'a [f32],
    #[cfg(target_arch = "x86_64")]
    f32_sums_lo: &'a [f32],
    #[cfg(target_arch = "x86_64")]
    f32_sums_hi: &'a [f32],
    #[cfg(target_arch = "x86_64")]
    i8_sums_lo: &'a [i32],
    #[cfg(target_arch = "x86_64")]
    i8_sums_hi: &'a [i32],
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    xq_i8: &'a [i8],
    #[cfg(target_arch = "x86_64")]
    xq_u8: &'a [u8],
}

pub(crate) struct MatmulPreparedActivation<'a> {
    x: &'a [f32],
    n: usize,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    q8: Q8ActivationPrequant<'a>,
}

fn prequantize_activation_q8<'a>(
    x: &[f32],
    n: usize,
    scratch: &'a mut MatmulActivationScratch,
) -> Q8ActivationPrequant<'a> {
    let nb = n / QK8_0;
    if scratch.q8_scales.len() < nb {
        scratch.q8_scales.resize(nb, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if scratch.q8_f32_sums.len() < nb {
        scratch.q8_f32_sums.resize(nb, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if scratch.q8_f32_sums_lo.len() < nb {
        scratch.q8_f32_sums_lo.resize(nb, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if scratch.q8_f32_sums_hi.len() < nb {
        scratch.q8_f32_sums_hi.resize(nb, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if scratch.q8_i8_sums_lo.len() < nb {
        scratch.q8_i8_sums_lo.resize(nb, 0);
    }
    #[cfg(target_arch = "x86_64")]
    if scratch.q8_i8_sums_hi.len() < nb {
        scratch.q8_i8_sums_hi.resize(nb, 0);
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if scratch.q8_x_i8.len() < n {
        scratch.q8_x_i8.resize(n, 0);
    }
    #[cfg(target_arch = "x86_64")]
    if scratch.q8_x_u8.len() < n {
        scratch.q8_x_u8.resize(n, 0);
    }
    let mut xq_block = [0i8; QK8_0];
    for i in 0..nb {
        let base = i * QK8_0;
        let xb = &x[base..base + QK8_0];
        let scale = quantize_f32_block_i8_32(xb, &mut xq_block);
        scratch.q8_scales[i] = scale;
        #[cfg(target_arch = "x86_64")]
        {
            scratch.q8_f32_sums[i] = xb.iter().copied().sum::<f32>();
            scratch.q8_f32_sums_lo[i] = xb[..16].iter().copied().sum::<f32>();
            scratch.q8_f32_sums_hi[i] = xb[16..].iter().copied().sum::<f32>();
            scratch.q8_i8_sums_lo[i] = xq_block[..16].iter().map(|&v| v as i32).sum::<i32>();
            scratch.q8_i8_sums_hi[i] = xq_block[16..].iter().map(|&v| v as i32).sum::<i32>();
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        scratch.q8_x_i8[base..base + QK8_0].copy_from_slice(&xq_block);
        #[cfg(target_arch = "x86_64")]
        for j in 0..QK8_0 {
            scratch.q8_x_u8[base + j] = (xq_block[j] as i32 + 128) as u8;
        }
    }
    Q8ActivationPrequant {
        scales: &scratch.q8_scales[..nb],
        #[cfg(target_arch = "x86_64")]
        f32_sums: &scratch.q8_f32_sums[..nb],
        #[cfg(target_arch = "x86_64")]
        f32_sums_lo: &scratch.q8_f32_sums_lo[..nb],
        #[cfg(target_arch = "x86_64")]
        f32_sums_hi: &scratch.q8_f32_sums_hi[..nb],
        #[cfg(target_arch = "x86_64")]
        i8_sums_lo: &scratch.q8_i8_sums_lo[..nb],
        #[cfg(target_arch = "x86_64")]
        i8_sums_hi: &scratch.q8_i8_sums_hi[..nb],
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        xq_i8: &scratch.q8_x_i8[..n],
        #[cfg(target_arch = "x86_64")]
        xq_u8: &scratch.q8_x_u8[..n],
    }
}

pub(crate) fn matmul_prepared_activation_supported(qw: &QuantizedTensor) -> bool {
    let n = qw.cols;
    #[cfg(target_arch = "x86_64")]
    {
        match qw.ttype.0 {
            GGML_TYPE_Q8_0 => {
                !use_x86_avx2_fma()
                    && n >= QK8_0
                    && n.is_multiple_of(QK8_0)
                    && (use_x86_avx512_vnni_q8() || use_x86_avx_vnni())
            }
            GGML_TYPE_Q2_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => {
                n >= QK_K && n.is_multiple_of(QK_K) && x86_qk_mr4_uses_prequant(qw.ttype.0)
            }
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = n;
        false
    }
}

pub(crate) fn prepare_matmul_activation<'a>(
    x: &'a [f32],
    qw: &QuantizedTensor,
    scratch: &'a mut MatmulActivationScratch,
) -> Option<MatmulPreparedActivation<'a>> {
    let n = qw.cols;
    if x.len() < n || !matmul_prepared_activation_supported(qw) {
        return None;
    }
    let q8 = prequantize_activation_q8(&x[..n], n, scratch);
    Some(MatmulPreparedActivation { x: &x[..n], n, q8 })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn vec_dot_q8_0_dotprod_prequant(preq: &Q8ActivationPrequant, w: &[u8], n: usize) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    for i in 0..nb {
        let x_scale = preq.scales[i];
        if x_scale == 0.0 {
            continue;
        }
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let xq_ptr = preq.xq_i8.as_ptr().add(i * QK8_0);
        let dot_i32 = dot_i8_32_dotprod(xq_ptr, q_ptr);
        sum += dot_i32 as f32 * x_scale * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn vec_dot_q8_0_x86_avxvnni_prequant(
    preq: &Q8ActivationPrequant,
    w: &[u8],
    n: usize,
) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    for i in 0..nb {
        let x_scale = preq.scales[i];
        if x_scale == 0.0 {
            continue;
        }
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let xq_u8_ptr = preq.xq_u8.as_ptr().add(i * QK8_0);
        let dot_u = dot_u8_i8_32_x86_avxvnni(xq_u8_ptr, q_ptr);
        let sum_q = sum_i8_32_ptr(q_ptr);
        let dot_s = dot_u - 128 * sum_q;
        sum += dot_s as f32 * x_scale * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl")]
unsafe fn vec_dot_q8_0_x86_avx512vnni_prequant(
    preq: &Q8ActivationPrequant,
    w: &[u8],
    n: usize,
) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    for i in 0..nb {
        let x_scale = preq.scales[i];
        if x_scale == 0.0 {
            continue;
        }
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let xq_u8_ptr = preq.xq_u8.as_ptr().add(i * QK8_0);
        let dot_u = dot_u8_i8_32_x86_avx512vnni(xq_u8_ptr, q_ptr);
        let sum_q = sum_i8_32_ptr(q_ptr);
        let dot_s = dot_u - 128 * sum_q;
        sum += dot_s as f32 * x_scale * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn unpack_q4_nibbles_32(q: &[u8], lo: &mut [u8; QK8_0], hi: &mut [u8; QK8_0]) {
    debug_assert_eq!(q.len(), QK8_0);
    for i in 0..QK8_0 {
        let qv = q[i];
        lo[i] = qv & 0x0f;
        hi[i] = qv >> 4;
    }
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
unsafe fn dot_u8_i8_16_x86_avxvnni(a_u8: *const u8, b_i8: *const i8) -> i32 {
    let src = _mm_setzero_si128();
    let a = _mm_loadu_si128(a_u8 as *const __m128i);
    let b = _mm_loadu_si128(b_i8 as *const __m128i);
    let acc = _mm_dpbusd_avx_epi32(src, a, b);
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
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
unsafe fn dot_u8_i8_16_x86_avx512vnni(a_u8: *const u8, b_i8: *const i8) -> i32 {
    let src = _mm_setzero_si128();
    let a = _mm_loadu_si128(a_u8 as *const __m128i);
    let b = _mm_loadu_si128(b_i8 as *const __m128i);
    let acc = _mm_dpbusd_epi32(src, a, b);
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
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
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_32_dotprod(a: *const i8, b: *const i8) -> i32 {
    // Uses SDOT (ARMv8.2 dotprod) via inline asm to avoid unstable stdarch_neon_dotprod gate.
    // Caller must ensure dotprod is available at runtime.
    let result: i32;
    core::arch::asm!(
        "movi v8.4s, #0",
        "ldr q9, [{a}]",
        "ldr q10, [{b}]",
        "sdot v8.4s, v9.16b, v10.16b",
        "ldr q9, [{a}, #16]",
        "ldr q10, [{b}, #16]",
        "sdot v8.4s, v9.16b, v10.16b",
        "addv s8, v8.4s",
        "fmov {res:w}, s8",
        a = in(reg) a,
        b = in(reg) b,
        res = out(reg) result,
        out("v8") _,
        out("v9") _,
        out("v10") _,
        options(nostack, pure, readonly),
    );
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
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

/// Compute dot products of quantized x (32 i8 values) against two weight rows using SMMLA.
/// Each 8-element x chunk is duplicated into both A-matrix rows so that after 4 SMMLA calls:
///   v8[0] == v8[2] == dot(x, w0),  v8[1] == v8[3] == dot(x, w1)
/// Caller must ensure i8mm is available at runtime.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "i8mm")]
unsafe fn dot2_q8_32_i8mm(xq: *const i8, w0: *const i8, w1: *const i8) -> (i32, i32) {
    // Uses SMMLA (ARMv8.6 i8mm) via inline asm to avoid unstable stdarch_neon_i8mm gate.
    let dot0: i32;
    let dot1: i32;
    core::arch::asm!(
        "movi v8.4s, #0",
        // k=0: x[0..8] × (w0[0..8] | w1[0..8])
        "ldr d9, [{xq}]",
        "ins v9.d[1], v9.d[0]",
        "ldr d10, [{w0}]",
        "ldr d11, [{w1}]",
        "ins v10.d[1], v11.d[0]",
        "smmla v8.4s, v9.16b, v10.16b",
        // k=1: x[8..16]
        "ldr d9, [{xq}, #8]",
        "ins v9.d[1], v9.d[0]",
        "ldr d10, [{w0}, #8]",
        "ldr d11, [{w1}, #8]",
        "ins v10.d[1], v11.d[0]",
        "smmla v8.4s, v9.16b, v10.16b",
        // k=2: x[16..24]
        "ldr d9, [{xq}, #16]",
        "ins v9.d[1], v9.d[0]",
        "ldr d10, [{w0}, #16]",
        "ldr d11, [{w1}, #16]",
        "ins v10.d[1], v11.d[0]",
        "smmla v8.4s, v9.16b, v10.16b",
        // k=3: x[24..32]
        "ldr d9, [{xq}, #24]",
        "ins v9.d[1], v9.d[0]",
        "ldr d10, [{w0}, #24]",
        "ldr d11, [{w1}, #24]",
        "ins v10.d[1], v11.d[0]",
        "smmla v8.4s, v9.16b, v10.16b",
        // v8[0] = dot(x,w0), v8[1] = dot(x,w1) — extract both lanes
        "fmov {dot0:w}, s8",
        "mov {dot1:w}, v8.s[1]",
        xq = in(reg) xq,
        w0 = in(reg) w0,
        w1 = in(reg) w1,
        dot0 = out(reg) dot0,
        dot1 = out(reg) dot1,
        out("v8") _,
        out("v9") _,
        out("v10") _,
        out("v11") _,
        options(nostack, pure, readonly),
    );
    (dot0, dot1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn vec_dot_q8_0_2rows_i8mm_prequant(
    preq: &Q8ActivationPrequant,
    w0: &[u8],
    w1: &[u8],
    n: usize,
) -> (f32, f32) {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;

    for i in 0..nb {
        let x_scale = preq.scales[i];
        if x_scale == 0.0 {
            continue;
        }
        let off = i * block_sz;
        let d0 = fp16_to_fp32(read_u16_le(w0, off));
        let d1 = fp16_to_fp32(read_u16_le(w1, off));
        let w0_ptr = w0[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let w1_ptr = w1[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let xq_ptr = preq.xq_i8.as_ptr().add(i * QK8_0);
        let (dot0, dot1) = dot2_q8_32_i8mm(xq_ptr, w0_ptr, w1_ptr);
        sum0 += dot0 as f32 * x_scale * d0;
        sum1 += dot1 as f32 * x_scale * d1;
    }
    (sum0, sum1)
}

#[cfg(target_arch = "aarch64")]
fn matmul_q8_mr2_chunk_prequant(
    out: &mut [f32],
    base_row: usize,
    preq: &Q8ActivationPrequant,
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
) {
    let mut i = 0usize;
    while i + 2 <= out.len() {
        let row0_off = data_offset + (base_row + i) * row_size;
        let row1_off = row0_off + row_size;
        let r0 = &mapped[row0_off..row0_off + row_size];
        let r1 = &mapped[row1_off..row1_off + row_size];
        let (s0, s1) = unsafe { vec_dot_q8_0_2rows_i8mm_prequant(preq, r0, r1, n) };
        out[i] = s0;
        out[i + 1] = s1;
        i += 2;
    }
    if i < out.len() {
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = unsafe { vec_dot_q8_0_2rows_i8mm_prequant(preq, row, row, n).0 };
    }
}

/// Scalar reference for i8mm validation: quantizes x to int8 the same way as the i8mm
/// kernel, then accumulates with scalar i32 arithmetic. Integer arithmetic is exact, so
/// results must match the SMMLA output precisely (only the final f32 scale multiply can
/// differ by at most 1 ULP). Comparing against vec_dot_q8_0 (float x) would introduce
/// ~1% quantization error and always fail the tolerance check — that was the original bug.
#[cfg(target_arch = "aarch64")]
fn vec_dot_q8_0_quantized_ref(x: &[f32], w: &[u8], n: usize) -> f32 {
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
        let inv = 1.0 / x_scale;
        for j in 0..QK8_0 {
            xq[j] = (xb[j] * inv).round().clamp(-127.0, 127.0) as i8;
        }
        let wq = &w[off + 2..off + 2 + QK8_0];
        let mut dot = 0i32;
        for j in 0..QK8_0 {
            dot += xq[j] as i32 * wq[j] as i8 as i32;
        }
        sum += dot as f32 * x_scale * d;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
fn validate_q8_mr2_once(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
) -> bool {
    use std::sync::atomic::Ordering as AtomicOrdering;
    match AARCH64_Q8_0_MR2_STATUS.load(AtomicOrdering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }

    let r0 = &mapped[data_offset..data_offset + row_size];
    let r1 = &mapped[data_offset + row_size..data_offset + 2 * row_size];
    let mut scratch = MatmulActivationScratch::new();
    let preq = prequantize_activation_q8(x, n, &mut scratch);
    let (mr2_0, mr2_1) = unsafe { vec_dot_q8_0_2rows_i8mm_prequant(&preq, r0, r1, n) };
    // Use quantized scalar reference (same algorithm as i8mm) so integer dot products match.
    let ref_0 = vec_dot_q8_0_quantized_ref(x, r0, n);
    let ref_1 = vec_dot_q8_0_quantized_ref(x, r1, n);

    let tol0 = 1e-4f32 * ref_0.abs().max(1.0);
    let tol1 = 1e-4f32 * ref_1.abs().max(1.0);
    let ok = (mr2_0 - ref_0).abs() <= tol0 && (mr2_1 - ref_1).abs() <= tol1;

    AARCH64_Q8_0_MR2_STATUS.store(if ok { 1 } else { 2 }, AtomicOrdering::Relaxed);
    if !ok && kernel_validation_warnings_enabled() {
        eprintln!("Warning: disabling aarch64 i8mm MR2 Q8_0 kernel due to validation mismatch");
    }
    ok
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn try_matmul_q8_mr2_with_scratch(
    xout: &mut [f32],
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    scratch: &mut MatmulActivationScratch,
) -> bool {
    if !use_aarch64_i8mm_q8() {
        return false;
    }
    if n < QK8_0 || !n.is_multiple_of(QK8_0) {
        return false;
    }
    let d = xout.len();
    if d < 2 {
        return false;
    }
    if !validate_q8_mr2_once(x, mapped, data_offset, row_size, n) {
        return false;
    }
    let preq = prequantize_activation_q8(x, n, scratch);
    let chunk_rows = par_matmul_chunk_rows();
    if d >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_q8_mr2_chunk_prequant(
                    chunk,
                    base_row,
                    &preq,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                );
            });
    } else {
        matmul_q8_mr2_chunk_prequant(xout, 0, &preq, mapped, data_offset, row_size, n);
    }
    true
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

/// Four-row Q2_K dot product. This mirrors [`vec_dot_q2_k`] accumulation order
/// per row, but decodes four rows while sharing activation loads and row
/// scheduling through the MR4 matmul path.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn vec_dot_q2_k_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let scales: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][off..off + QK_K / 16]);
        let mut q_off = [off + QK_K / 16; 4];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4 + 2));
        }

        let mut is = 0usize;
        for n_outer in (0..QK_K).step_by(128) {
            let q: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][q_off[r]..q_off[r] + 32]);
            let mut shift = 0;
            for j in 0..4 {
                for r in 0..4 {
                    let sc = scales[r][is];
                    let dl = d[r] * (sc & 0x0f) as f32;
                    let ml = dmin[r] * (sc >> 4) as f32;
                    for l in 0..16 {
                        let idx = n_outer + j * 32 + l;
                        let wv = dl * ((q[r][l] >> shift) & 0x03) as f32 - ml;
                        sums[r] += xb[idx] * wv;
                    }

                    let sc2 = scales[r][is + 1];
                    let dl2 = d[r] * (sc2 & 0x0f) as f32;
                    let ml2 = dmin[r] * (sc2 >> 4) as f32;
                    for l in 0..16 {
                        let idx = n_outer + j * 32 + 16 + l;
                        let wv = dl2 * ((q[r][l + 16] >> shift) & 0x03) as f32 - ml2;
                        sums[r] += xb[idx] * wv;
                    }
                }
                is += 2;
                shift += 2;
            }
            for q_row_off in &mut q_off {
                *q_row_off += 32;
            }
        }
    }

    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q2_k_16_neon(x: *const f32, q: *const u8, shift: usize) -> f32 {
    let low2 = vdupq_n_u8(0x03);
    let qv = vld1q_u8(q);
    let q2 = match shift {
        0 => vandq_u8(qv, low2),
        2 => vandq_u8(vshrq_n_u8(qv, 2), low2),
        4 => vandq_u8(vshrq_n_u8(qv, 4), low2),
        6 => vandq_u8(vshrq_n_u8(qv, 6), low2),
        _ => unreachable!(),
    };

    let q16lo = vmovl_u8(vget_low_u8(q2));
    let q16hi = vmovl_u8(vget_high_u8(q2));
    let q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16lo)));
    let q1 = vcvtq_f32_u32(vmovl_high_u16(q16lo));
    let q2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16hi)));
    let q3 = vcvtq_f32_u32(vmovl_high_u16(q16hi));

    let mut acc = vfmaq_f32(vdupq_n_f32(0.0), vld1q_f32(x), q0);
    acc = vfmaq_f32(acc, vld1q_f32(x.add(4)), q1);
    acc = vfmaq_f32(acc, vld1q_f32(x.add(8)), q2);
    acc = vfmaq_f32(acc, vld1q_f32(x.add(12)), q3);
    vaddvq_f32(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sum_f32_16_neon_ptr(x: *const f32) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    acc = vaddq_f32(acc, vld1q_f32(x));
    acc = vaddq_f32(acc, vld1q_f32(x.add(4)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(8)));
    acc = vaddq_f32(acc, vld1q_f32(x.add(12)));
    vaddvq_f32(acc)
}

/// Q2_K 4-row dot for aarch64.
///
/// Decodes each 16-value two-bit group with NEON and applies the Q2_K
/// scale/min metadata exactly as the scalar [`vec_dot_q2_k`] path does. The
/// MR4 runtime gate validates this against four scalar rows before use.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn vec_dot_q2_k_4rows_aarch64(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = i * QK_K;
        let scales: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][off..off + QK_K / 16]);
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4 + 2));
        }

        let mut is = 0usize;
        for outer in 0..2usize {
            let q_base = off + QK_K / 16 + outer * 32;
            for j in 0..4usize {
                let shift = j * 2;
                for g in 0..2usize {
                    let x_base = xb + outer * 128 + j * 32 + g * 16;
                    let x_ptr = unsafe { x.as_ptr().add(x_base) };
                    let x_sum = unsafe { sum_f32_16_neon_ptr(x_ptr) };
                    let q_group = q_base + g * 16;
                    for r in 0..4usize {
                        let dot = unsafe {
                            dot_q2_k_16_neon(x_ptr, rows[r].as_ptr().add(q_group), shift)
                        };
                        let sc = scales[r][is + g];
                        let dl = d[r] * (sc & 0x0f) as f32;
                        let ml = dmin[r] * (sc >> 4) as f32;
                        sums[r] += dl * dot - ml * x_sum;
                    }
                }
                is += 2;
            }
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_q2_k_4rows_x86(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q2_k_4rows_x86_avx2(x, w0, w1, w2, w3, n);
        }
    }
    vec_dot_q2_k_4rows(x, w0, w1, w2, w3, n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q2_k_4rows_x86_avx2(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mask = _mm_set1_epi8(0x03);
    let mut acc = [_mm256_setzero_ps(); 4];

    for i in 0..nb {
        let off = i * block_sz;
        let x_base_block = i * QK_K;
        let scales: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][off..off + QK_K / 16]);
        let mut q_off = [off + QK_K / 16; 4];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4 + 2));
        }

        let mut is = 0usize;
        for n_outer in (0..QK_K).step_by(128) {
            let q: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][q_off[r]..q_off[r] + 32]);
            let mut shift = 0i32;
            for j in 0..4usize {
                let shift_count = _mm_cvtsi32_si128(shift);
                for g in 0..2usize {
                    let x_base = x_base_block + n_outer + j * 32 + g * 16;
                    let xv0 = _mm256_loadu_ps(x.as_ptr().add(x_base));
                    let xv1 = _mm256_loadu_ps(x.as_ptr().add(x_base + 8));
                    for r in 0..4usize {
                        let sc = scales[r][is + g];
                        let dl = _mm256_set1_ps(d[r] * (sc & 0x0f) as f32);
                        let ml = _mm256_set1_ps(dmin[r] * (sc >> 4) as f32);
                        let qv = _mm_loadu_si128(q[r].as_ptr().add(g * 16) as *const __m128i);
                        let q2 = _mm_and_si128(_mm_srl_epi16(qv, shift_count), mask);
                        let q_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q2));
                        let q_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(q2, 8)));
                        let w_lo = _mm256_sub_ps(_mm256_mul_ps(q_lo, dl), ml);
                        let w_hi = _mm256_sub_ps(_mm256_mul_ps(q_hi, dl), ml);
                        acc[r] = _mm256_fmadd_ps(xv0, w_lo, acc[r]);
                        acc[r] = _mm256_fmadd_ps(xv1, w_hi, acc[r]);
                    }
                }
                is += 2;
                shift += 2;
            }
            for q_row_off in &mut q_off {
                *q_row_off += 32;
            }
        }
    }

    let mut sums = [0.0f32; 4];
    for r in 0..4 {
        sums[r] = hsum256_ps(acc[r]);
    }
    sums
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

/// Four-row Q3_K dot product: computes `x · row_r` for four contiguous weight
/// rows sharing the activation `x`. Numerically identical to calling
/// [`vec_dot_q3_k`] on each row (same accumulation order), but the shared
/// activation element `xv` is loaded once and applied to all four rows with four
/// register accumulators. This is the portable MR4 microkernel for Q3_K; the
/// runtime self-check
/// (`validate_qk_mr4_once*`) verifies it against the scalar path on real tensors
/// before enabling it where architecture-specific wrappers use this fallback.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn vec_dot_q3_k_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let mut sums = [0.0f32; 4];

    let sc_off = QK_K / 8 + QK_K / 4;
    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let hmask: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][off..off + QK_K / 8]);
        let scales: [[i8; 16]; 4] =
            std::array::from_fn(|r| q3_scales(&rows[r][off + sc_off..off + sc_off + 12]));
        let d_all: [f32; 4] =
            std::array::from_fn(|r| fp16_to_fp32(read_u16_le(rows[r], off + sc_off + 12)));

        let mut q_off = off + QK_K / 8;
        let mut is = 0usize;
        let mut m: u8 = 1;
        let mut block_sum = [0.0f32; 4];

        for n_outer in (0..QK_K).step_by(128) {
            let q: [&[u8]; 4] = std::array::from_fn(|r| &rows[r][q_off..q_off + 32]);
            let mut shift = 0usize;
            for j in 0..4 {
                let dl: [f32; 4] =
                    std::array::from_fn(|r| d_all[r] * (scales[r][is] as i32 - 32) as f32);
                is += 1;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + l;
                    let xv = xb[idx];
                    for r in 0..4 {
                        let wv = ((q[r][l] >> shift) & 3) as i8
                            - if (hmask[r][l] & m) != 0 { 0 } else { 4 };
                        block_sum[r] += xv * dl[r] * wv as f32;
                    }
                }
                let dl2: [f32; 4] =
                    std::array::from_fn(|r| d_all[r] * (scales[r][is] as i32 - 32) as f32);
                is += 1;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + 16 + l;
                    let xv = xb[idx];
                    for r in 0..4 {
                        let wv = ((q[r][l + 16] >> shift) & 3) as i8
                            - if (hmask[r][l + 16] & m) != 0 { 0 } else { 4 };
                        block_sum[r] += xv * dl2[r] * wv as f32;
                    }
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
        for r in 0..4 {
            sums[r] += block_sum[r];
        }
    }
    sums
}

/// NEON Q3_K 16-element dot helper.
///
/// Decodes one 16-byte Q3_K low-bit lane with the matching hmask lane:
/// `(q >> shift) & 3` minus 4 when the current hmask bit is clear, then
/// accumulates `x[0..16] · decoded`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q3_k_16_neon(
    x: *const f32,
    q: *const u8,
    hmask: *const u8,
    shift: usize,
    m: u8,
) -> f32 {
    let low2 = vdupq_n_u8(0x03);
    let qv = vld1q_u8(q);
    let q2 = match shift {
        0 => vandq_u8(qv, low2),
        2 => vandq_u8(vshrq_n_u8(qv, 2), low2),
        4 => vandq_u8(vshrq_n_u8(qv, 4), low2),
        6 => vandq_u8(vshrq_n_u8(qv, 6), low2),
        _ => unreachable!(),
    };

    let hbit = vandq_u8(vld1q_u8(hmask), vdupq_n_u8(m));
    let not_set = vceqq_u8(hbit, vdupq_n_u8(0));
    let sub4 = vandq_u8(not_set, vdupq_n_u8(4));
    let wv = vsubq_s8(vreinterpretq_s8_u8(q2), vreinterpretq_s8_u8(sub4));

    let s16lo = vmovl_s8(vget_low_s8(wv));
    let s16hi = vmovl_s8(vget_high_s8(wv));
    let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16lo)));
    let w1 = vcvtq_f32_s32(vmovl_high_s16(s16lo));
    let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16hi)));
    let w3 = vcvtq_f32_s32(vmovl_high_s16(s16hi));

    let mut acc = vfmaq_f32(vdupq_n_f32(0.0), vld1q_f32(x), w0);
    acc = vfmaq_f32(acc, vld1q_f32(x.add(4)), w1);
    acc = vfmaq_f32(acc, vld1q_f32(x.add(8)), w2);
    acc = vfmaq_f32(acc, vld1q_f32(x.add(12)), w3);
    vaddvq_f32(acc)
}

/// Q3_K 4-row dot for aarch64.
///
/// This is the NEON counterpart to the x86 AVX2 kernel: it follows the scalar
/// Q3_K block walk exactly, but decodes each 16-value group with NEON and shares
/// the activation stream across four contiguous rows. The runtime MR4 gate
/// validates it against four scalar [`vec_dot_q3_k`] calls before use.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn vec_dot_q3_k_4rows_aarch64(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let sc_off = QK_K / 8 + QK_K / 4;
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = i * QK_K;

        let mut scales = [[0i8; 16]; 4];
        let mut d_all = [0.0f32; 4];
        for r in 0..4 {
            scales[r] = q3_scales(&rows[r][off + sc_off..off + sc_off + 12]);
            d_all[r] = fp16_to_fp32(read_u16_le(rows[r], off + sc_off + 12));
        }

        let mut is = 0usize;
        let mut m = 1u8;
        for outer in 0..2usize {
            let q_base = off + QK_K / 8 + outer * 32;
            for j in 0..4usize {
                let shift = j * 2;
                for g in 0..2usize {
                    let x_base = xb + outer * 128 + j * 32 + g * 16;
                    let q_group = q_base + g * 16;
                    let h_group = off + g * 16;
                    for r in 0..4usize {
                        let dot = unsafe {
                            dot_q3_k_16_neon(
                                x.as_ptr().add(x_base),
                                rows[r].as_ptr().add(q_group),
                                rows[r].as_ptr().add(h_group),
                                shift,
                                m,
                            )
                        };
                        let dl = d_all[r] * (scales[r][is] as i32 - 32) as f32;
                        sums[r] += dl * dot;
                    }
                    is += 1;
                }
                m <<= 1;
            }
        }
    }
    sums
}

/// Q3_K 4-row dot for x86_64: AVX2/FMA kernel when available, portable
/// fallback otherwise. Tolerance-level equivalent to four [`vec_dot_q3_k`]
/// calls (SIMD lane accumulation reorders the sums); gated at runtime by
/// `validate_qk_mr4_once_x86` like the other MR4 kernels.
#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_q3_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_q3_k_4rows_x86_avx2(x, r0, r1, r2, r3, n);
        }
    }
    vec_dot_q3_k_4rows(x, r0, r1, r2, r3, n)
}

/// AVX2/FMA Q3_K 4-row kernel.
///
/// Mirrors the scalar [`vec_dot_q3_k`] control flow exactly — same
/// per-superblock layout walk (hmask 32B, 2-bit planes in two 32B chunks,
/// 16 six-bit scales, fp16 `d`), same `(q >> shift) & 3` minus
/// `hmask-bit ? 0 : 4` weight decode — vectorized 16 elements at a time and
/// sharing the activation loads across the four rows. Weight decode is
/// integer-exact; only the FMA accumulation order differs from scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q3_k_4rows_x86_avx2(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let sc_off = QK_K / 8 + QK_K / 4; // 96: hmask(32) + qs(64)... qs follows hmask

    let zero = _mm_setzero_si128();
    let low2 = _mm_set1_epi8(0x03);
    let four = _mm_set1_epi8(4);
    let mut acc = [_mm256_setzero_ps(); 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = i * QK_K;

        // Per-row block constants: scales, d, hmask halves, all four 16-byte
        // 2-bit chunks (chunk layout: [outer0 A, outer0 B, outer1 A, outer1 B]).
        let mut scales = [[0i8; 16]; 4];
        let mut d_all = [0.0f32; 4];
        let mut hm = [[zero; 2]; 4];
        let mut qc = [[zero; 4]; 4];
        for r in 0..4 {
            let row = rows[r];
            scales[r] = q3_scales(&row[off + sc_off..off + sc_off + 12]);
            d_all[r] = fp16_to_fp32(read_u16_le(row, off + sc_off + 12));
            hm[r][0] = _mm_loadu_si128(row.as_ptr().add(off) as *const __m128i);
            hm[r][1] = _mm_loadu_si128(row.as_ptr().add(off + 16) as *const __m128i);
            qc[r][0] = _mm_loadu_si128(row.as_ptr().add(off + 32) as *const __m128i);
            qc[r][1] = _mm_loadu_si128(row.as_ptr().add(off + 48) as *const __m128i);
            qc[r][2] = _mm_loadu_si128(row.as_ptr().add(off + 64) as *const __m128i);
            qc[r][3] = _mm_loadu_si128(row.as_ptr().add(off + 80) as *const __m128i);
        }

        let mut is = 0usize;
        let mut mbit: u32 = 1;
        for outer in 0..2usize {
            for j in 0..4usize {
                let shift_count = _mm_cvtsi32_si128((2 * j) as i32);
                let mvec = _mm_set1_epi8(mbit as i8);
                for g in 0..2usize {
                    let x_base = xb + outer * 128 + j * 32 + g * 16;
                    let xv0 = _mm256_loadu_ps(x.as_ptr().add(x_base));
                    let xv1 = _mm256_loadu_ps(x.as_ptr().add(x_base + 8));
                    for r in 0..4usize {
                        // (q >> shift) & 3; cross-byte shift contamination
                        // lands at bit >= 8-shift >= 2, masked off by 0x03.
                        let q3 =
                            _mm_and_si128(_mm_srl_epi16(qc[r][outer * 2 + g], shift_count), low2);
                        // Subtract 4 where the hmask bit is NOT set.
                        let hbit = _mm_and_si128(hm[r][g], mvec);
                        let not_set = _mm_cmpeq_epi8(hbit, zero);
                        let sub4 = _mm_and_si128(not_set, four);
                        let wv = _mm_sub_epi8(q3, sub4);
                        let w_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wv));
                        let w_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(wv, 8)));
                        let dl = d_all[r] * (scales[r][is] as i32 - 32) as f32;
                        let dlv = _mm256_set1_ps(dl);
                        acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(w_lo, dlv), xv0, acc[r]);
                        acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(w_hi, dlv), xv1, acc[r]);
                    }
                    is += 1;
                }
                mbit <<= 1;
            }
        }
    }

    let mut sums = [0.0f32; 4];
    for r in 0..4 {
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[r]);
        sums[r] = tmp.iter().sum();
    }
    sums
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
                let q1 = (((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4)) as i8) - 32;
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

/// NEON helper: dot product of x0[0..32] with lo nibbles and x1[0..32] with hi nibbles
/// from 32 packed nibble bytes at `q`. Returns (dot_lo, dot_hi).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q4_nibbles_pair_neon(x0: *const f32, x1: *const f32, q: *const u8) -> (f32, f32) {
    let mask = vdupq_n_u8(0x0f);
    let mut acc_lo = vdupq_n_f32(0.0);
    let mut acc_hi = vdupq_n_f32(0.0);
    // Two chunks of 16 nibble-bytes = 32 lo values + 32 hi values
    for chunk in 0..2usize {
        let off = chunk * 16;
        let qv = vld1q_u8(q.add(off));
        let lo8 = vandq_u8(qv, mask);
        let hi8 = vshrq_n_u8(qv, 4);
        // Widen u8 → u16 → u32 → f32 (lo nibbles)
        let lo16_lo = vmovl_u8(vget_low_u8(lo8));
        let lo16_hi = vmovl_u8(vget_high_u8(lo8));
        let lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_lo)));
        let lo_f1 = vcvtq_f32_u32(vmovl_high_u16(lo16_lo));
        let lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_hi)));
        let lo_f3 = vcvtq_f32_u32(vmovl_high_u16(lo16_hi));
        // Widen u8 → u16 → u32 → f32 (hi nibbles)
        let hi16_lo = vmovl_u8(vget_low_u8(hi8));
        let hi16_hi = vmovl_u8(vget_high_u8(hi8));
        let hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_lo)));
        let hi_f1 = vcvtq_f32_u32(vmovl_high_u16(hi16_lo));
        let hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_hi)));
        let hi_f3 = vcvtq_f32_u32(vmovl_high_u16(hi16_hi));
        // Load x floats and accumulate
        let x0_0 = vld1q_f32(x0.add(off));
        let x0_1 = vld1q_f32(x0.add(off + 4));
        let x0_2 = vld1q_f32(x0.add(off + 8));
        let x0_3 = vld1q_f32(x0.add(off + 12));
        let x1_0 = vld1q_f32(x1.add(off));
        let x1_1 = vld1q_f32(x1.add(off + 4));
        let x1_2 = vld1q_f32(x1.add(off + 8));
        let x1_3 = vld1q_f32(x1.add(off + 12));
        acc_lo = vfmaq_f32(acc_lo, x0_0, lo_f0);
        acc_lo = vfmaq_f32(acc_lo, x0_1, lo_f1);
        acc_lo = vfmaq_f32(acc_lo, x0_2, lo_f2);
        acc_lo = vfmaq_f32(acc_lo, x0_3, lo_f3);
        acc_hi = vfmaq_f32(acc_hi, x1_0, hi_f0);
        acc_hi = vfmaq_f32(acc_hi, x1_1, hi_f1);
        acc_hi = vfmaq_f32(acc_hi, x1_2, hi_f2);
        acc_hi = vfmaq_f32(acc_hi, x1_3, hi_f3);
    }
    (vaddvq_f32(acc_lo), vaddvq_f32(acc_hi))
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
            let x0 = &xb[j..j + 32];
            let x1 = &xb[j + 32..j + 64];
            let x0_sum: f32 = x0.iter().copied().sum();
            let x1_sum: f32 = x1.iter().copied().sum();
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                let a_lo = d[r] * sc1 as f32;
                let b_lo = dmin[r] * m1 as f32;
                let a_hi = d[r] * sc2 as f32;
                let b_hi = dmin[r] * m2 as f32;
                let q = &rows[r][q_off[r]..q_off[r] + 32];
                let (dot_lo, dot_hi) =
                    unsafe { dot_q4_nibbles_pair_neon(x0.as_ptr(), x1.as_ptr(), q.as_ptr()) };
                sums[r] += a_lo * dot_lo - b_lo * x0_sum + a_hi * dot_hi - b_hi * x1_sum;
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

/// NEON helper: dot product for Q5_K — handles 5th bit from qh using vtstq_u8.
/// `ql` points to 32 bytes of lower 4-bit packed values, `qh` to 32 bytes of high-bit flags.
/// `u1`/`u2` are the current bit masks for lo/hi 5th-bit extraction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q5_nibbles_pair_neon(
    x0: *const f32,
    x1: *const f32,
    ql: *const u8,
    qh: *const u8,
    u1: u8,
    u2: u8,
) -> (f32, f32) {
    let mask_lo = vdupq_n_u8(0x0f);
    let add16 = vdupq_n_u8(16);
    let u1_mask = vdupq_n_u8(u1);
    let u2_mask = vdupq_n_u8(u2);
    let mut acc_lo = vdupq_n_f32(0.0);
    let mut acc_hi = vdupq_n_f32(0.0);

    for chunk in 0..2usize {
        let off = chunk * 16;
        let qv = vld1q_u8(ql.add(off));
        let qhv = vld1q_u8(qh.add(off));
        // lo nibble + 5th bit: vtstq_u8 gives 0xFF where bit set, AND with 16 → 0 or 16
        let lo8 = vaddq_u8(
            vandq_u8(qv, mask_lo),
            vandq_u8(vtstq_u8(qhv, u1_mask), add16),
        );
        let hi8 = vaddq_u8(vshrq_n_u8(qv, 4), vandq_u8(vtstq_u8(qhv, u2_mask), add16));
        // Widen u8 → u16 → u32 → f32
        let lo16_lo = vmovl_u8(vget_low_u8(lo8));
        let lo16_hi = vmovl_u8(vget_high_u8(lo8));
        let lo_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_lo)));
        let lo_f1 = vcvtq_f32_u32(vmovl_high_u16(lo16_lo));
        let lo_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16_hi)));
        let lo_f3 = vcvtq_f32_u32(vmovl_high_u16(lo16_hi));
        let hi16_lo = vmovl_u8(vget_low_u8(hi8));
        let hi16_hi = vmovl_u8(vget_high_u8(hi8));
        let hi_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_lo)));
        let hi_f1 = vcvtq_f32_u32(vmovl_high_u16(hi16_lo));
        let hi_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16_hi)));
        let hi_f3 = vcvtq_f32_u32(vmovl_high_u16(hi16_hi));
        let x0_0 = vld1q_f32(x0.add(off));
        let x0_1 = vld1q_f32(x0.add(off + 4));
        let x0_2 = vld1q_f32(x0.add(off + 8));
        let x0_3 = vld1q_f32(x0.add(off + 12));
        let x1_0 = vld1q_f32(x1.add(off));
        let x1_1 = vld1q_f32(x1.add(off + 4));
        let x1_2 = vld1q_f32(x1.add(off + 8));
        let x1_3 = vld1q_f32(x1.add(off + 12));
        acc_lo = vfmaq_f32(acc_lo, x0_0, lo_f0);
        acc_lo = vfmaq_f32(acc_lo, x0_1, lo_f1);
        acc_lo = vfmaq_f32(acc_lo, x0_2, lo_f2);
        acc_lo = vfmaq_f32(acc_lo, x0_3, lo_f3);
        acc_hi = vfmaq_f32(acc_hi, x1_0, hi_f0);
        acc_hi = vfmaq_f32(acc_hi, x1_1, hi_f1);
        acc_hi = vfmaq_f32(acc_hi, x1_2, hi_f2);
        acc_hi = vfmaq_f32(acc_hi, x1_3, hi_f3);
    }
    (vaddvq_f32(acc_lo), vaddvq_f32(acc_hi))
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
            let x0 = &xb[j..j + 32];
            let x1 = &xb[j + 32..j + 64];
            let x0_sum: f32 = x0.iter().copied().sum();
            let x1_sum: f32 = x1.iter().copied().sum();
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                let a_lo = d[r] * sc1 as f32;
                let b_lo = dmin[r] * m1 as f32;
                let a_hi = d[r] * sc2 as f32;
                let b_hi = dmin[r] * m2 as f32;
                let (dot_lo, dot_hi) = unsafe {
                    dot_q5_nibbles_pair_neon(
                        x0.as_ptr(),
                        x1.as_ptr(),
                        rows[r][ql_off[r]..].as_ptr(),
                        qh[r].as_ptr(),
                        u1,
                        u2,
                    )
                };
                sums[r] += a_lo * dot_lo - b_lo * x0_sum + a_hi * dot_hi - b_hi * x1_sum;
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

        // Q6_K: two 128-element outer blocks per QK_K=256 block
        for n_outer in (0..QK_K).step_by(128) {
            // x layout within this 128-elem window:
            //   x0 = xb[n_outer..n_outer+32]   → q1 (ql[0..32] lo nibbles)
            //   x1 = xb[n_outer+32..n_outer+64] → q2 (ql[32..64] lo nibbles)
            //   x2 = xb[n_outer+64..n_outer+96] → q3 (ql[0..32] hi nibbles)
            //   x3 = xb[n_outer+96..n_outer+128]→ q4 (ql[32..64] hi nibbles)
            let x0 = &xb[n_outer..n_outer + 32];
            let x1 = &xb[n_outer + 32..n_outer + 64];
            let x2 = &xb[n_outer + 64..n_outer + 96];
            let x3 = &xb[n_outer + 96..n_outer + 128];

            for r in 0..4 {
                let sc = &rows[r][sc_off[r]..sc_off[r] + 8];
                // Scales: sc[0/1] for first half, sc[2/3] for second half (pairs per 16 elements)
                // The scalar uses `is = l/16`, giving sc[0],sc[2],sc[4],sc[6] for l=0..15
                // and sc[1],sc[3],sc[5],sc[7] for l=16..31. We split into two 16-elem halves.
                let s1a = d[r] * sc[0] as i8 as f32;
                let s1b = d[r] * sc[1] as i8 as f32;
                let s2a = d[r] * sc[2] as i8 as f32;
                let s2b = d[r] * sc[3] as i8 as f32;
                let s3a = d[r] * sc[4] as i8 as f32;
                let s3b = d[r] * sc[5] as i8 as f32;
                let s4a = d[r] * sc[6] as i8 as f32;
                let s4b = d[r] * sc[7] as i8 as f32;

                let ql_ptr = rows[r][ql_off[r]..].as_ptr();
                let qh_ptr = rows[r][qh_off[r]..].as_ptr();

                // Split at l=16: each 16-element half uses a different scale pair
                let (h0_d1, h0_d2, h0_d3, h0_d4) = unsafe {
                    dot_q6_half_neon(
                        x0.as_ptr(),
                        x1.as_ptr(),
                        x2.as_ptr(),
                        x3.as_ptr(),
                        ql_ptr,
                        qh_ptr,
                    )
                };
                let (h1_d1, h1_d2, h1_d3, h1_d4) = unsafe {
                    dot_q6_half_neon(
                        x0.as_ptr().add(16),
                        x1.as_ptr().add(16),
                        x2.as_ptr().add(16),
                        x3.as_ptr().add(16),
                        ql_ptr.add(16),
                        qh_ptr.add(16),
                    )
                };

                sums[r] += s1a * h0_d1
                    + s1b * h1_d1
                    + s2a * h0_d2
                    + s2b * h1_d2
                    + s3a * h0_d3
                    + s3b * h1_d3
                    + s4a * h0_d4
                    + s4b * h1_d4;
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

/// NEON helper: 16-element Q6_K dot product for one half-segment.
/// Computes dot(x0,q1), dot(x1,q2), dot(x2,q3), dot(x3,q4) for l=0..16.
/// `ql` points to start of the 16-elem window (ql[0..16] and ql[32..48] for the two sub-vectors).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q6_half_neon(
    x0: *const f32,
    x1: *const f32,
    x2: *const f32,
    x3: *const f32,
    ql: *const u8,
    qh: *const u8,
) -> (f32, f32, f32, f32) {
    let mask_lo4 = vdupq_n_u8(0x0f);
    let mask_03 = vdupq_n_u8(0x03);
    let bias = vdupq_n_s8(32);

    let ql0v = vld1q_u8(ql); // ql[0..16]
    let ql1v = vld1q_u8(ql.add(32)); // ql[32..48]
    let qhv = vld1q_u8(qh); // qh[0..16]

    let top1 = vshlq_n_u8(vandq_u8(qhv, mask_03), 4);
    let q1u = vorrq_u8(vandq_u8(ql0v, mask_lo4), top1);
    let q1s = vsubq_s8(vreinterpretq_s8_u8(q1u), bias);

    let top2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), mask_03), 4);
    let q2u = vorrq_u8(vandq_u8(ql1v, mask_lo4), top2);
    let q2s = vsubq_s8(vreinterpretq_s8_u8(q2u), bias);

    let top3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), mask_03), 4);
    let q3u = vorrq_u8(vshrq_n_u8(ql0v, 4), top3);
    let q3s = vsubq_s8(vreinterpretq_s8_u8(q3u), bias);

    let top4 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), mask_03), 4);
    let q4u = vorrq_u8(vshrq_n_u8(ql1v, 4), top4);
    let q4s = vsubq_s8(vreinterpretq_s8_u8(q4u), bias);

    macro_rules! s8_to_f32x4 {
        ($v:expr) => {{
            let s16lo = vmovl_s8(vget_low_s8($v));
            let s16hi = vmovl_s8(vget_high_s8($v));
            (
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16lo))),
                vcvtq_f32_s32(vmovl_high_s16(s16lo)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16hi))),
                vcvtq_f32_s32(vmovl_high_s16(s16hi)),
            )
        }};
    }
    let (q1f0, q1f1, q1f2, q1f3) = s8_to_f32x4!(q1s);
    let (q2f0, q2f1, q2f2, q2f3) = s8_to_f32x4!(q2s);
    let (q3f0, q3f1, q3f2, q3f3) = s8_to_f32x4!(q3s);
    let (q4f0, q4f1, q4f2, q4f3) = s8_to_f32x4!(q4s);

    let x0_0 = vld1q_f32(x0);
    let x0_1 = vld1q_f32(x0.add(4));
    let x0_2 = vld1q_f32(x0.add(8));
    let x0_3 = vld1q_f32(x0.add(12));
    let x1_0 = vld1q_f32(x1);
    let x1_1 = vld1q_f32(x1.add(4));
    let x1_2 = vld1q_f32(x1.add(8));
    let x1_3 = vld1q_f32(x1.add(12));
    let x2_0 = vld1q_f32(x2);
    let x2_1 = vld1q_f32(x2.add(4));
    let x2_2 = vld1q_f32(x2.add(8));
    let x2_3 = vld1q_f32(x2.add(12));
    let x3_0 = vld1q_f32(x3);
    let x3_1 = vld1q_f32(x3.add(4));
    let x3_2 = vld1q_f32(x3.add(8));
    let x3_3 = vld1q_f32(x3.add(12));

    let mut a1 = vfmaq_f32(vdupq_n_f32(0.0), x0_0, q1f0);
    a1 = vfmaq_f32(a1, x0_1, q1f1);
    a1 = vfmaq_f32(a1, x0_2, q1f2);
    a1 = vfmaq_f32(a1, x0_3, q1f3);
    let mut a2 = vfmaq_f32(vdupq_n_f32(0.0), x1_0, q2f0);
    a2 = vfmaq_f32(a2, x1_1, q2f1);
    a2 = vfmaq_f32(a2, x1_2, q2f2);
    a2 = vfmaq_f32(a2, x1_3, q2f3);
    let mut a3 = vfmaq_f32(vdupq_n_f32(0.0), x2_0, q3f0);
    a3 = vfmaq_f32(a3, x2_1, q3f1);
    a3 = vfmaq_f32(a3, x2_2, q3f2);
    a3 = vfmaq_f32(a3, x2_3, q3f3);
    let mut a4 = vfmaq_f32(vdupq_n_f32(0.0), x3_0, q4f0);
    a4 = vfmaq_f32(a4, x3_1, q4f1);
    a4 = vfmaq_f32(a4, x3_2, q4f2);
    a4 = vfmaq_f32(a4, x3_3, q4f3);

    (
        vaddvq_f32(a1),
        vaddvq_f32(a2),
        vaddvq_f32(a3),
        vaddvq_f32(a4),
    )
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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
    let total_rows = out.len();
    let mut i = 0usize;
    while i + 4 <= out.len() {
        aarch64_prefetch_row(
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
            GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_aarch64(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q3_K => vec_dot_q3_k_4rows_aarch64(x, r0, r1, r2, r3, n),
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
        aarch64_prefetch_row(
            mapped,
            data_offset,
            row_size,
            base_row + i,
            base_row.saturating_add(total_rows),
        );
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = match ttype {
            GGML_TYPE_Q2_K => vec_dot_q2_k(x, row, n),
            GGML_TYPE_Q3_K => vec_dot_q3_k(x, row, n),
            GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_legacy_q4_q5_iq4_mr4_chunk_aarch64(
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
        aarch64_prefetch_row(
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
            GGML_TYPE_Q4_0 => vec_dot_q4_0_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_1 => vec_dot_q4_1_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_0 => vec_dot_q5_0_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_1 => vec_dot_q5_1_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_IQ4_NL => vec_dot_iq4_nl_4rows(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
        out[i] = sums[0];
        out[i + 1] = sums[1];
        out[i + 2] = sums[2];
        out[i + 3] = sums[3];
        i += 4;
    }
    while i < out.len() {
        aarch64_prefetch_row(
            mapped,
            data_offset,
            row_size,
            base_row + i,
            base_row.saturating_add(total_rows),
        );
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = match ttype {
            GGML_TYPE_Q4_0 => vec_dot_q4_0(x, row, n),
            GGML_TYPE_Q4_1 => vec_dot_q4_1(x, row, n),
            GGML_TYPE_Q5_0 => vec_dot_q5_0(x, row, n),
            GGML_TYPE_Q5_1 => vec_dot_q5_1(x, row, n),
            GGML_TYPE_IQ4_NL => vec_dot_iq4_nl(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn legacy_q4_q5_iq4_mr4_status_aarch64(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q4_0 => &AARCH64_Q4_0_MR4_STATUS,
        GGML_TYPE_Q4_1 => &AARCH64_Q4_1_MR4_STATUS,
        GGML_TYPE_Q5_0 => &AARCH64_Q5_0_MR4_STATUS,
        GGML_TYPE_Q5_1 => &AARCH64_Q5_1_MR4_STATUS,
        GGML_TYPE_IQ4_NL => &AARCH64_IQ4_NL_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn validate_legacy_q4_q5_iq4_mr4_once_aarch64(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    let status = legacy_q4_q5_iq4_mr4_status_aarch64(ttype);
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
        GGML_TYPE_Q4_0 => vec_dot_q4_0_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q4_1 => vec_dot_q4_1_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_0 => vec_dot_q5_0_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_1 => vec_dot_q5_1_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_IQ4_NL => vec_dot_iq4_nl_4rows(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q4_0 => [
            vec_dot_q4_0(x, r0, n),
            vec_dot_q4_0(x, r1, n),
            vec_dot_q4_0(x, r2, n),
            vec_dot_q4_0(x, r3, n),
        ],
        GGML_TYPE_Q4_1 => [
            vec_dot_q4_1(x, r0, n),
            vec_dot_q4_1(x, r1, n),
            vec_dot_q4_1(x, r2, n),
            vec_dot_q4_1(x, r3, n),
        ],
        GGML_TYPE_Q5_0 => [
            vec_dot_q5_0(x, r0, n),
            vec_dot_q5_0(x, r1, n),
            vec_dot_q5_0(x, r2, n),
            vec_dot_q5_0(x, r3, n),
        ],
        GGML_TYPE_Q5_1 => [
            vec_dot_q5_1(x, r0, n),
            vec_dot_q5_1(x, r1, n),
            vec_dot_q5_1(x, r2, n),
            vec_dot_q5_1(x, r3, n),
        ],
        GGML_TYPE_IQ4_NL => [
            vec_dot_iq4_nl(x, r0, n),
            vec_dot_iq4_nl(x, r1, n),
            vec_dot_iq4_nl(x, r2, n),
            vec_dot_iq4_nl(x, r3, n),
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
    if !ok && kernel_validation_warnings_enabled() {
        eprintln!(
            "Warning: disabling aarch64 legacy Q4/Q5/IQ4_NL MR4 kernel for type {} due to validation mismatch",
            ttype
        );
    }
    ok
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
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
    if !matches!(
        ttype,
        GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 | GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 | GGML_TYPE_IQ4_NL
    ) {
        return false;
    }
    if n < QK4_0 || !n.is_multiple_of(QK4_0) || xout.len() < 4 {
        return false;
    }
    if !validate_legacy_q4_q5_iq4_mr4_once_aarch64(x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }

    let chunk_rows = par_matmul_chunk_rows();
    if xout.len() >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_legacy_q4_q5_iq4_mr4_chunk_aarch64(
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
        matmul_legacy_q4_q5_iq4_mr4_chunk_aarch64(
            xout,
            0,
            x,
            mapped,
            data_offset,
            row_size,
            n,
            ttype,
        );
    }
    true
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn mr4_status(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q2_K => &AARCH64_Q2K_MR4_STATUS,
        GGML_TYPE_Q3_K => &AARCH64_Q3K_MR4_STATUS,
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
        GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_aarch64(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q3_K => vec_dot_q3_k_4rows_aarch64(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q2_K => [
            vec_dot_q2_k(x, r0, n),
            vec_dot_q2_k(x, r1, n),
            vec_dot_q2_k(x, r2, n),
            vec_dot_q2_k(x, r3, n),
        ],
        GGML_TYPE_Q3_K => [
            vec_dot_q3_k(x, r0, n),
            vec_dot_q3_k(x, r1, n),
            vec_dot_q3_k(x, r2, n),
            vec_dot_q3_k(x, r3, n),
        ],
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
    if !ok && kernel_validation_warnings_enabled() {
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
    if !matches!(
        ttype,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
    ) {
        return false;
    }
    if n < QK_K || !n.is_multiple_of(QK_K) {
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
    if use_x86_avx512_vnni_q8() {
        unsafe {
            return vec_dot_q4_k_4rows_x86_avx512vnni(x, r0, r1, r2, r3, n);
        }
    }
    if use_x86_avx_vnni() {
        unsafe {
            return vec_dot_q4_k_4rows_x86_avxvnni(x, r0, r1, r2, r3, n);
        }
    }
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
    if use_x86_avx512_vnni_q8() {
        unsafe {
            return vec_dot_q5_k_4rows_x86_avx512vnni(x, r0, r1, r2, r3, n);
        }
    }
    if use_x86_avx_vnni() {
        unsafe {
            return vec_dot_q5_k_4rows_x86_avxvnni(x, r0, r1, r2, r3, n);
        }
    }
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
    if use_x86_avx512_vnni_q8() {
        unsafe {
            return vec_dot_q6_k_4rows_x86_avx512vnni(x, r0, r1, r2, r3, n);
        }
    }
    if use_x86_avx_vnni() {
        unsafe {
            return vec_dot_q6_k_4rows_x86_avxvnni(x, r0, r1, r2, r3, n);
        }
    }
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
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q6_k_4rows_x86_avxvnni(
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

            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let mut x2_q = [0i8; QK8_0];
            let mut x3_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
            let x2_scale = quantize_f32_block_i8_32(x2, &mut x2_q);
            let x3_scale = quantize_f32_block_i8_32(x3, &mut x3_q);

            let x0_sum_lo = if x0_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x0_q.as_ptr())
            };
            let x0_sum_hi = if x0_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x0_q.as_ptr().add(16))
            };
            let x1_sum_lo = if x1_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x1_q.as_ptr())
            };
            let x1_sum_hi = if x1_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x1_q.as_ptr().add(16))
            };
            let x2_sum_lo = if x2_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x2_q.as_ptr())
            };
            let x2_sum_hi = if x2_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x2_q.as_ptr().add(16))
            };
            let x3_sum_lo = if x3_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x3_q.as_ptr())
            };
            let x3_sum_hi = if x3_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x3_q.as_ptr().add(16))
            };

            for r in 0..4 {
                let ql = &rows[r][ql_off[r]..ql_off[r] + 64];
                let qh = &rows[r][qh_off[r]..qh_off[r] + 32];
                let sc = &rows[r][sc_off[r]..sc_off[r] + 8];
                let mut q1_u = [0u8; QK8_0];
                let mut q2_u = [0u8; QK8_0];
                let mut q3_u = [0u8; QK8_0];
                let mut q4_u = [0u8; QK8_0];

                for l in 0..QK8_0 {
                    let ql0 = ql[l];
                    let ql1 = ql[l + 32];
                    let qh0 = qh[l];
                    let v1 = (ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4);
                    let v2 = (ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4);
                    let v3 = (ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4);
                    let v4 = (ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4);
                    // Map signed q6 range [-32, 31] to unsigned [96, 159] for VNNI:
                    // q_u = (q_signed + 128) = ((v - 32) + 128) = v + 96.
                    q1_u[l] = v1 + 96;
                    q2_u[l] = v2 + 96;
                    q3_u[l] = v3 + 96;
                    q4_u[l] = v4 + 96;
                }

                let s00 = d[r] * sc[0] as i8 as f32;
                let s01 = d[r] * sc[1] as i8 as f32;
                let s10 = d[r] * sc[2] as i8 as f32;
                let s11 = d[r] * sc[3] as i8 as f32;
                let s20 = d[r] * sc[4] as i8 as f32;
                let s21 = d[r] * sc[5] as i8 as f32;
                let s30 = d[r] * sc[6] as i8 as f32;
                let s31 = d[r] * sc[7] as i8 as f32;

                let mut acc = 0.0f32;
                if x0_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avxvnni(q1_u.as_ptr(), x0_q.as_ptr()) - 128 * x0_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avxvnni(q1_u.as_ptr().add(16), x0_q.as_ptr().add(16))
                            - 128 * x0_sum_hi;
                    acc += x0_scale * (s00 * dot_lo as f32 + s01 * dot_hi as f32);
                }
                if x1_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avxvnni(q2_u.as_ptr(), x1_q.as_ptr()) - 128 * x1_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avxvnni(q2_u.as_ptr().add(16), x1_q.as_ptr().add(16))
                            - 128 * x1_sum_hi;
                    acc += x1_scale * (s10 * dot_lo as f32 + s11 * dot_hi as f32);
                }
                if x2_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avxvnni(q3_u.as_ptr(), x2_q.as_ptr()) - 128 * x2_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avxvnni(q3_u.as_ptr().add(16), x2_q.as_ptr().add(16))
                            - 128 * x2_sum_hi;
                    acc += x2_scale * (s20 * dot_lo as f32 + s21 * dot_hi as f32);
                }
                if x3_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avxvnni(q4_u.as_ptr(), x3_q.as_ptr()) - 128 * x3_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avxvnni(q4_u.as_ptr().add(16), x3_q.as_ptr().add(16))
                            - 128 * x3_sum_hi;
                    acc += x3_scale * (s30 * dot_lo as f32 + s31 * dot_hi as f32);
                }
                sums[r] += acc;
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
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q6_k_4rows_x86_avx512vnni(
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

            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let mut x2_q = [0i8; QK8_0];
            let mut x3_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
            let x2_scale = quantize_f32_block_i8_32(x2, &mut x2_q);
            let x3_scale = quantize_f32_block_i8_32(x3, &mut x3_q);

            let x0_sum_lo = if x0_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x0_q.as_ptr())
            };
            let x0_sum_hi = if x0_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x0_q.as_ptr().add(16))
            };
            let x1_sum_lo = if x1_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x1_q.as_ptr())
            };
            let x1_sum_hi = if x1_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x1_q.as_ptr().add(16))
            };
            let x2_sum_lo = if x2_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x2_q.as_ptr())
            };
            let x2_sum_hi = if x2_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x2_q.as_ptr().add(16))
            };
            let x3_sum_lo = if x3_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x3_q.as_ptr())
            };
            let x3_sum_hi = if x3_scale == 0.0 {
                0
            } else {
                sum_i8_16_ptr(x3_q.as_ptr().add(16))
            };

            for r in 0..4 {
                let ql = &rows[r][ql_off[r]..ql_off[r] + 64];
                let qh = &rows[r][qh_off[r]..qh_off[r] + 32];
                let sc = &rows[r][sc_off[r]..sc_off[r] + 8];
                let mut q1_u = [0u8; QK8_0];
                let mut q2_u = [0u8; QK8_0];
                let mut q3_u = [0u8; QK8_0];
                let mut q4_u = [0u8; QK8_0];

                for l in 0..QK8_0 {
                    let ql0 = ql[l];
                    let ql1 = ql[l + 32];
                    let qh0 = qh[l];
                    let v1 = (ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4);
                    let v2 = (ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4);
                    let v3 = (ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4);
                    let v4 = (ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4);
                    q1_u[l] = v1 + 96;
                    q2_u[l] = v2 + 96;
                    q3_u[l] = v3 + 96;
                    q4_u[l] = v4 + 96;
                }

                let s00 = d[r] * sc[0] as i8 as f32;
                let s01 = d[r] * sc[1] as i8 as f32;
                let s10 = d[r] * sc[2] as i8 as f32;
                let s11 = d[r] * sc[3] as i8 as f32;
                let s20 = d[r] * sc[4] as i8 as f32;
                let s21 = d[r] * sc[5] as i8 as f32;
                let s30 = d[r] * sc[6] as i8 as f32;
                let s31 = d[r] * sc[7] as i8 as f32;

                let mut acc = 0.0f32;
                if x0_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avx512vnni(q1_u.as_ptr(), x0_q.as_ptr()) - 128 * x0_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avx512vnni(q1_u.as_ptr().add(16), x0_q.as_ptr().add(16))
                            - 128 * x0_sum_hi;
                    acc += x0_scale * (s00 * dot_lo as f32 + s01 * dot_hi as f32);
                }
                if x1_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avx512vnni(q2_u.as_ptr(), x1_q.as_ptr()) - 128 * x1_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avx512vnni(q2_u.as_ptr().add(16), x1_q.as_ptr().add(16))
                            - 128 * x1_sum_hi;
                    acc += x1_scale * (s10 * dot_lo as f32 + s11 * dot_hi as f32);
                }
                if x2_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avx512vnni(q3_u.as_ptr(), x2_q.as_ptr()) - 128 * x2_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avx512vnni(q3_u.as_ptr().add(16), x2_q.as_ptr().add(16))
                            - 128 * x2_sum_hi;
                    acc += x2_scale * (s20 * dot_lo as f32 + s21 * dot_hi as f32);
                }
                if x3_scale != 0.0 {
                    let dot_lo =
                        dot_u8_i8_16_x86_avx512vnni(q4_u.as_ptr(), x3_q.as_ptr()) - 128 * x3_sum_lo;
                    let dot_hi =
                        dot_u8_i8_16_x86_avx512vnni(q4_u.as_ptr().add(16), x3_q.as_ptr().add(16))
                            - 128 * x3_sum_hi;
                    acc += x3_scale * (s30 * dot_lo as f32 + s31 * dot_hi as f32);
                }
                sums[r] += acc;
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
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q4_k_4rows_x86_avxvnni(
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
            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
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

                let q = &rows[r][q_off[r]..q_off[r] + QK8_0];
                let mut q_lo = [0u8; QK8_0];
                let mut q_hi = [0u8; QK8_0];
                unpack_q4_nibbles_32(q, &mut q_lo, &mut q_hi);
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avxvnni(q_lo.as_ptr(), x0_q.as_ptr()) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avxvnni(q_hi.as_ptr(), x1_q.as_ptr()) as f32 * x1_scale
                };
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
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q5_k_4rows_x86_avxvnni(
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
            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
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

                let ql = &rows[r][ql_off[r]..ql_off[r] + QK8_0];
                let mut lo_vals = [0u8; QK8_0];
                let mut hi_vals = [0u8; QK8_0];
                for l in 0..QK8_0 {
                    let qv = ql[l];
                    lo_vals[l] = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    hi_vals[l] = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                }
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avxvnni(lo_vals.as_ptr(), x0_q.as_ptr()) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avxvnni(hi_vals.as_ptr(), x1_q.as_ptr()) as f32 * x1_scale
                };
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
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q4_k_4rows_x86_avx512vnni(
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
            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
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

                let q = &rows[r][q_off[r]..q_off[r] + QK8_0];
                let mut q_lo = [0u8; QK8_0];
                let mut q_hi = [0u8; QK8_0];
                unpack_q4_nibbles_32(q, &mut q_lo, &mut q_hi);
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avx512vnni(q_lo.as_ptr(), x0_q.as_ptr()) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avx512vnni(q_hi.as_ptr(), x1_q.as_ptr()) as f32 * x1_scale
                };
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
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q5_k_4rows_x86_avx512vnni(
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
            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
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

                let ql = &rows[r][ql_off[r]..ql_off[r] + QK8_0];
                let mut lo_vals = [0u8; QK8_0];
                let mut hi_vals = [0u8; QK8_0];
                for l in 0..QK8_0 {
                    let qv = ql[l];
                    lo_vals[l] = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    hi_vals[l] = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                }
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avx512vnni(lo_vals.as_ptr(), x0_q.as_ptr()) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    dot_u8_i8_32_x86_avx512vnni(hi_vals.as_ptr(), x1_q.as_ptr()) as f32 * x1_scale
                };
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
type X86DotU8I8x32 = unsafe fn(*const u8, *const i8) -> i32;

#[cfg(target_arch = "x86_64")]
type X86DotU8I8x16 = unsafe fn(*const u8, *const i8) -> i32;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn q8_prequant_block(preq: &Q8ActivationPrequant<'_>, block: usize) -> (*const i8, f32, f32) {
    let ptr = unsafe { preq.xq_i8.as_ptr().add(block * QK8_0) };
    (ptr, preq.scales[block], preq.f32_sums[block])
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vec_dot_q2_k_4rows_x86_vnni_prequant_impl(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
    dot16: X86DotU8I8x16,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let block_base = i * (QK_K / QK8_0);
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut q_off = [0usize; 4];

        for r in 0..4 {
            scales[r] = &rows[r][off..off + QK_K / 16];
            q_off[r] = off + QK_K / 16;
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + QK_K / 16 + QK_K / 4 + 2));
        }

        let mut is = 0usize;
        for n_outer in (0..QK_K).step_by(128) {
            for j in 0..4usize {
                let shift = j * 2;
                for g in 0..2usize {
                    let elem_base = n_outer + j * 32 + g * 16;
                    let x_block = block_base + elem_base / QK8_0;
                    let x_half_hi = (elem_base % QK8_0) == 16;
                    let xq = preq
                        .xq_i8
                        .as_ptr()
                        .add(x_block * QK8_0 + if x_half_hi { 16 } else { 0 });
                    let x_scale = preq.scales[x_block];
                    let x_f32_sum = if x_half_hi {
                        preq.f32_sums_hi[x_block]
                    } else {
                        preq.f32_sums_lo[x_block]
                    };

                    for r in 0..4 {
                        let sc = scales[r][is + g];
                        let dl = d[r] * (sc & 0x0f) as f32;
                        let ml = dmin[r] * (sc >> 4) as f32;
                        let q = &rows[r][q_off[r] + g * 16..q_off[r] + g * 16 + 16];
                        let mut q2 = [0u8; 16];
                        for l in 0..16 {
                            q2[l] = (q[l] >> shift) & 0x03;
                        }
                        let dot = if x_scale == 0.0 {
                            0.0
                        } else {
                            dot16(q2.as_ptr(), xq) as f32 * x_scale
                        };
                        sums[r] += dl * dot - ml * x_f32_sum;
                    }
                }
                is += 2;
            }
            for q_row_off in &mut q_off {
                *q_row_off += 32;
            }
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vec_dot_q4_k_4rows_x86_vnni_prequant_impl(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
    dot32: X86DotU8I8x32,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let block_base = i * (QK_K / QK8_0);
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
            let (x0_q, x0_scale, x0_sum) = q8_prequant_block(preq, block_base + j / QK8_0);
            let (x1_q, x1_scale, x1_sum) = q8_prequant_block(preq, block_base + j / QK8_0 + 1);

            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                let a_lo = d[r] * sc1 as f32;
                let b_lo = dmin[r] * m1 as f32;
                let a_hi = d[r] * sc2 as f32;
                let b_hi = dmin[r] * m2 as f32;

                let q = &rows[r][q_off[r]..q_off[r] + QK8_0];
                let mut q_lo = [0u8; QK8_0];
                let mut q_hi = [0u8; QK8_0];
                unpack_q4_nibbles_32(q, &mut q_lo, &mut q_hi);
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    dot32(q_lo.as_ptr(), x0_q) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    dot32(q_hi.as_ptr(), x1_q) as f32 * x1_scale
                };
                sums[r] += a_lo * dot_lo - b_lo * x0_sum + a_hi * dot_hi - b_hi * x1_sum;
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vec_dot_q5_k_4rows_x86_vnni_prequant_impl(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
    dot32: X86DotU8I8x32,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let block_base = i * (QK_K / QK8_0);
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
            let (x0_q, x0_scale, x0_sum) = q8_prequant_block(preq, block_base + j / QK8_0);
            let (x1_q, x1_scale, x1_sum) = q8_prequant_block(preq, block_base + j / QK8_0 + 1);

            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                let a_lo = d[r] * sc1 as f32;
                let b_lo = dmin[r] * m1 as f32;
                let a_hi = d[r] * sc2 as f32;
                let b_hi = dmin[r] * m2 as f32;

                let ql = &rows[r][ql_off[r]..ql_off[r] + QK8_0];
                let mut lo_vals = [0u8; QK8_0];
                let mut hi_vals = [0u8; QK8_0];
                for l in 0..QK8_0 {
                    let qv = ql[l];
                    lo_vals[l] = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    hi_vals[l] = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                }
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    dot32(lo_vals.as_ptr(), x0_q) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    dot32(hi_vals.as_ptr(), x1_q) as f32 * x1_scale
                };
                sums[r] += a_lo * dot_lo - b_lo * x0_sum + a_hi * dot_hi - b_hi * x1_sum;
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
unsafe fn vec_dot_q6_k_4rows_x86_vnni_prequant_impl(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
    dot16: X86DotU8I8x16,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let block_base = i * (QK_K / QK8_0);
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
            let x_blocks = [
                block_base + n_outer / QK8_0,
                block_base + n_outer / QK8_0 + 1,
                block_base + n_outer / QK8_0 + 2,
                block_base + n_outer / QK8_0 + 3,
            ];

            for r in 0..4 {
                let ql = &rows[r][ql_off[r]..ql_off[r] + 64];
                let qh = &rows[r][qh_off[r]..qh_off[r] + 32];
                let sc = &rows[r][sc_off[r]..sc_off[r] + 8];
                let mut q_u = [[0u8; QK8_0]; 4];

                for l in 0..QK8_0 {
                    let ql0 = ql[l];
                    let ql1 = ql[l + 32];
                    let qh0 = qh[l];
                    let v1 = (ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4);
                    let v2 = (ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4);
                    let v3 = (ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4);
                    let v4 = (ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4);
                    q_u[0][l] = v1 + 96;
                    q_u[1][l] = v2 + 96;
                    q_u[2][l] = v3 + 96;
                    q_u[3][l] = v4 + 96;
                }

                let scales = [
                    d[r] * sc[0] as i8 as f32,
                    d[r] * sc[1] as i8 as f32,
                    d[r] * sc[2] as i8 as f32,
                    d[r] * sc[3] as i8 as f32,
                    d[r] * sc[4] as i8 as f32,
                    d[r] * sc[5] as i8 as f32,
                    d[r] * sc[6] as i8 as f32,
                    d[r] * sc[7] as i8 as f32,
                ];

                let mut acc = 0.0f32;
                for b in 0..4 {
                    let x_block = x_blocks[b];
                    let x_scale = preq.scales[x_block];
                    if x_scale == 0.0 {
                        continue;
                    }
                    let xq = preq.xq_i8.as_ptr().add(x_block * QK8_0);
                    let dot_lo = dot16(q_u[b].as_ptr(), xq) - 128 * preq.i8_sums_lo[x_block];
                    let dot_hi =
                        dot16(q_u[b].as_ptr().add(16), xq.add(16)) - 128 * preq.i8_sums_hi[x_block];
                    acc += x_scale
                        * (scales[2 * b] * dot_lo as f32 + scales[2 * b + 1] * dot_hi as f32);
                }
                sums[r] += acc;
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
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q2_k_4rows_x86_avxvnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q2_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_16_x86_avxvnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q2_k_4rows_x86_avx512vnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q2_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_16_x86_avx512vnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q4_k_4rows_x86_avxvnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q4_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_32_x86_avxvnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q4_k_4rows_x86_avx512vnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q4_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_32_x86_avx512vnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q5_k_4rows_x86_avxvnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q5_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_32_x86_avxvnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q5_k_4rows_x86_avx512vnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q5_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_32_x86_avx512vnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn vec_dot_q6_k_4rows_x86_avxvnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q6_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_16_x86_avxvnni)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vnni,avx512vl,fma")]
unsafe fn vec_dot_q6_k_4rows_x86_avx512vnni_prequant(
    preq: &Q8ActivationPrequant<'_>,
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    vec_dot_q6_k_4rows_x86_vnni_prequant_impl(preq, r0, r1, r2, r3, n, dot_u8_i8_16_x86_avx512vnni)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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
            GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q3_K => vec_dot_q3_k_4rows_x86(x, r0, r1, r2, r3, n),
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
            GGML_TYPE_Q2_K => vec_dot_q2_k(x, row, n),
            GGML_TYPE_Q3_K => vec_dot_q3_k(x, row, n),
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
fn x86_qk_mr4_uses_prequant(ttype: i32) -> bool {
    if !matches!(
        ttype,
        GGML_TYPE_Q2_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
    ) {
        return false;
    }
    use_x86_avx_vnni() || use_x86_avx512_vnni_q8()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn vec_dot_qk_4rows_x86_prequant(
    preq: &Q8ActivationPrequant<'_>,
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
    ttype: i32,
) -> [f32; 4] {
    if use_x86_avx512_vnni_q8() {
        return match ttype {
            GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_x86_avx512vnni_prequant(preq, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86_avx512vnni_prequant(preq, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86_avx512vnni_prequant(preq, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86_avx512vnni_prequant(preq, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
    }
    if use_x86_avx_vnni() {
        return match ttype {
            GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_x86_avxvnni_prequant(preq, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86_avxvnni_prequant(preq, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86_avxvnni_prequant(preq, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86_avxvnni_prequant(preq, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
    }

    match ttype {
        GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_qk_mr4_chunk_x86_prequant(
    out: &mut [f32],
    base_row: usize,
    preq: &Q8ActivationPrequant<'_>,
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
        let sums = unsafe { vec_dot_qk_4rows_x86_prequant(preq, x, r0, r1, r2, r3, n, ttype) };
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
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul_legacy_mr4_chunk_x86(
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
            GGML_TYPE_Q4_0 => vec_dot_q4_0_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_1 => vec_dot_q4_1_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_0 => vec_dot_q5_0_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_1 => vec_dot_q5_1_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_IQ4_NL => vec_dot_iq4_nl_4rows_x86(x, r0, r1, r2, r3, n),
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
            GGML_TYPE_Q4_0 => vec_dot_q4_0(x, row, n),
            GGML_TYPE_Q4_1 => vec_dot_q4_1(x, row, n),
            GGML_TYPE_Q5_0 => vec_dot_q5_0(x, row, n),
            GGML_TYPE_Q5_1 => vec_dot_q5_1(x, row, n),
            GGML_TYPE_IQ4_NL => vec_dot_iq4_nl(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn legacy_mr4_status_x86(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q4_0 => &X86_Q4_0_MR4_STATUS,
        GGML_TYPE_Q4_1 => &X86_Q4_1_MR4_STATUS,
        GGML_TYPE_Q5_0 => &X86_Q5_0_MR4_STATUS,
        GGML_TYPE_Q5_1 => &X86_Q5_1_MR4_STATUS,
        GGML_TYPE_IQ4_NL => &X86_IQ4_NL_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn validate_legacy_mr4_once_x86(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    let status = legacy_mr4_status_x86(ttype);
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
        GGML_TYPE_Q4_0 => vec_dot_q4_0_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q4_1 => vec_dot_q4_1_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_0 => vec_dot_q5_0_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_1 => vec_dot_q5_1_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_IQ4_NL => vec_dot_iq4_nl_4rows_x86(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q4_0 => [
            vec_dot_q4_0(x, r0, n),
            vec_dot_q4_0(x, r1, n),
            vec_dot_q4_0(x, r2, n),
            vec_dot_q4_0(x, r3, n),
        ],
        GGML_TYPE_Q4_1 => [
            vec_dot_q4_1(x, r0, n),
            vec_dot_q4_1(x, r1, n),
            vec_dot_q4_1(x, r2, n),
            vec_dot_q4_1(x, r3, n),
        ],
        GGML_TYPE_Q5_0 => [
            vec_dot_q5_0(x, r0, n),
            vec_dot_q5_0(x, r1, n),
            vec_dot_q5_0(x, r2, n),
            vec_dot_q5_0(x, r3, n),
        ],
        GGML_TYPE_Q5_1 => [
            vec_dot_q5_1(x, r0, n),
            vec_dot_q5_1(x, r1, n),
            vec_dot_q5_1(x, r2, n),
            vec_dot_q5_1(x, r3, n),
        ],
        GGML_TYPE_IQ4_NL => [
            vec_dot_iq4_nl(x, r0, n),
            vec_dot_iq4_nl(x, r1, n),
            vec_dot_iq4_nl(x, r2, n),
            vec_dot_iq4_nl(x, r3, n),
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
    if !ok && kernel_validation_warnings_enabled() {
        eprintln!(
            "Warning: disabling x86_64 legacy Q4/Q5/IQ4_NL MR4 kernel for type {} due to validation mismatch",
            ttype
        );
    }
    ok
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn try_matmul_legacy_mr4_x86(
    xout: &mut [f32],
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    if !use_x86_qk_mr4() || !use_x86_avx2_fma() {
        return false;
    }
    if !matches!(
        ttype,
        GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 | GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 | GGML_TYPE_IQ4_NL
    ) {
        return false;
    }
    if n < QK4_0 || n % QK4_0 != 0 || xout.len() < 4 {
        return false;
    }
    if !validate_legacy_mr4_once_x86(x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }

    let chunk_rows = par_matmul_chunk_rows();
    if xout.len() >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_legacy_mr4_chunk_x86(
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
        matmul_legacy_mr4_chunk_x86(xout, 0, x, mapped, data_offset, row_size, n, ttype);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn mr4_status_x86(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q2_K => &X86_Q2K_MR4_STATUS,
        GGML_TYPE_Q3_K => &X86_Q3K_MR4_STATUS,
        GGML_TYPE_Q4_K => &X86_Q4K_MR4_STATUS,
        GGML_TYPE_Q5_K => &X86_Q5K_MR4_STATUS,
        GGML_TYPE_Q6_K => &X86_Q6K_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn scalar_dot_u8_i8(a: &[u8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0i32;
    for l in 0..a.len() {
        dot += a[l] as i32 * b[l] as i32;
    }
    dot
}

/// Whether the non-prequant MR4 dispatch for `ttype` resolves to an int8
/// (VNNI) kernel. Validation must then compare against the int8-aware
/// references below: int8 kernels deviate from the float reference by the
/// activation-quantization error alone, which always exceeds the tolerance
/// (same pitfall as [`vec_dot_q8_0_quantized_ref`] documents for aarch64).
#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_qk_mr4_dispatch_is_int8(ttype: i32) -> bool {
    matches!(ttype, GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K)
        && (use_x86_avx512_vnni_q8() || use_x86_avx_vnni())
}

/// Int8-aware scalar reference for the Q4_K MR4 VNNI kernels: quantizes
/// activations per 32-lane block exactly like the kernels and accumulates
/// exact integer dots, so kernel and reference differ only in float rounding
/// of the final per-block scaling.
#[cfg(target_arch = "x86_64")]
fn vec_dot_q4_k_4rows_i8_ref(
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
            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                let a_lo = d[r] * sc1 as f32;
                let b_lo = dmin[r] * m1 as f32;
                let a_hi = d[r] * sc2 as f32;
                let b_hi = dmin[r] * m2 as f32;

                let q = &rows[r][q_off[r]..q_off[r] + QK8_0];
                let mut q_lo = [0u8; QK8_0];
                let mut q_hi = [0u8; QK8_0];
                unpack_q4_nibbles_32(q, &mut q_lo, &mut q_hi);
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    scalar_dot_u8_i8(&q_lo, &x0_q) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    scalar_dot_u8_i8(&q_hi, &x1_q) as f32 * x1_scale
                };
                sums[r] += a_lo * dot_lo - b_lo * x0_sum + a_hi * dot_hi - b_hi * x1_sum;
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

/// Int8-aware scalar reference for the Q5_K MR4 VNNI kernels; see
/// [`vec_dot_q4_k_4rows_i8_ref`].
#[cfg(target_arch = "x86_64")]
fn vec_dot_q5_k_4rows_i8_ref(
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
            let mut x0_q = [0i8; QK8_0];
            let mut x1_q = [0i8; QK8_0];
            let x0_scale = quantize_f32_block_i8_32(x0, &mut x0_q);
            let x1_scale = quantize_f32_block_i8_32(x1, &mut x1_q);
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                let a_lo = d[r] * sc1 as f32;
                let b_lo = dmin[r] * m1 as f32;
                let a_hi = d[r] * sc2 as f32;
                let b_hi = dmin[r] * m2 as f32;

                let ql = &rows[r][ql_off[r]..ql_off[r] + QK8_0];
                let mut lo_vals = [0u8; QK8_0];
                let mut hi_vals = [0u8; QK8_0];
                for l in 0..QK8_0 {
                    let qv = ql[l];
                    lo_vals[l] = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    hi_vals[l] = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                }
                let dot_lo = if x0_scale == 0.0 {
                    0.0
                } else {
                    scalar_dot_u8_i8(&lo_vals, &x0_q) as f32 * x0_scale
                };
                let dot_hi = if x1_scale == 0.0 {
                    0.0
                } else {
                    scalar_dot_u8_i8(&hi_vals, &x1_q) as f32 * x1_scale
                };
                sums[r] += a_lo * dot_lo - b_lo * x0_sum + a_hi * dot_hi - b_hi * x1_sum;
                ql_off[r] += 32;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    sums
}

/// Int8-aware scalar reference for the Q6_K MR4 VNNI kernels; see
/// [`vec_dot_q4_k_4rows_i8_ref`]. Mirrors the kernels' +96 bias trick:
/// `(v + 96) · xq − 128 · Σxq = (v − 32) · xq` with all lanes kept unsigned
/// for the u8×i8 dot.
#[cfg(target_arch = "x86_64")]
fn vec_dot_q6_k_4rows_i8_ref(
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

    let half_sum = |xq: &[i8; QK8_0], hi: bool| -> i32 {
        let base = if hi { 16 } else { 0 };
        xq[base..base + 16].iter().map(|&v| v as i32).sum()
    };

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
            let mut xq = [[0i8; QK8_0]; 4];
            let mut xs = [0.0f32; 4];
            for (b, half) in xq.iter_mut().enumerate() {
                let src = &xb[n_outer + b * 32..n_outer + b * 32 + 32];
                xs[b] = quantize_f32_block_i8_32(src, half);
            }

            for r in 0..4 {
                let ql = &rows[r][ql_off[r]..ql_off[r] + 64];
                let qh = &rows[r][qh_off[r]..qh_off[r] + 32];
                let sc = &rows[r][sc_off[r]..sc_off[r] + 8];
                let mut q_u = [[0u8; QK8_0]; 4];
                for l in 0..QK8_0 {
                    let ql0 = ql[l];
                    let ql1 = ql[l + 32];
                    let qh0 = qh[l];
                    q_u[0][l] = ((ql0 & 0x0f) | ((qh0 & 0x03) << 4)) + 96;
                    q_u[1][l] = ((ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4)) + 96;
                    q_u[2][l] = ((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)) + 96;
                    q_u[3][l] = ((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)) + 96;
                }

                let mut acc = 0.0f32;
                for b in 0..4 {
                    if xs[b] == 0.0 {
                        continue;
                    }
                    let s_lo = d[r] * sc[b * 2] as i8 as f32;
                    let s_hi = d[r] * sc[b * 2 + 1] as i8 as f32;
                    let dot_lo = scalar_dot_u8_i8(&q_u[b][..16], &xq[b][..16])
                        - 128 * half_sum(&xq[b], false);
                    let dot_hi = scalar_dot_u8_i8(&q_u[b][16..], &xq[b][16..])
                        - 128 * half_sum(&xq[b], true);
                    acc += xs[b] * (s_lo * dot_lo as f32 + s_hi * dot_hi as f32);
                }
                sums[r] += acc;
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
        GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q3_K => vec_dot_q3_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    // Int8 (VNNI) kernels are compared against int8-aware references; the
    // float references below would fail on activation-quantization error
    // alone and disable the kernels (that gate is why AMD ran AVX2-float).
    if x86_qk_mr4_dispatch_is_int8(ttype) {
        let i8_ref = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_i8_ref(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_i8_ref(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_i8_ref(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
        let mut ok = true;
        for i in 0..4 {
            let a = mr4[i];
            let b = i8_ref[i];
            let tol = 1e-4f32 * b.abs().max(1.0);
            if (a - b).abs() > tol {
                ok = false;
                break;
            }
        }
        status.store(if ok { 1 } else { 2 }, AtomicOrdering::Relaxed);
        if !ok && kernel_validation_warnings_enabled() {
            eprintln!(
                "Warning: disabling x86_64 MR4 kernel for type {} due to validation mismatch",
                ttype
            );
        }
        return ok;
    }
    let scalar = match ttype {
        GGML_TYPE_Q2_K => [
            vec_dot_q2_k(x, r0, n),
            vec_dot_q2_k(x, r1, n),
            vec_dot_q2_k(x, r2, n),
            vec_dot_q2_k(x, r3, n),
        ],
        GGML_TYPE_Q3_K => [
            vec_dot_q3_k(x, r0, n),
            vec_dot_q3_k(x, r1, n),
            vec_dot_q3_k(x, r2, n),
            vec_dot_q3_k(x, r3, n),
        ],
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
    if !ok && kernel_validation_warnings_enabled() {
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
    scratch: &mut MatmulActivationScratch,
) -> bool {
    if !use_x86_qk_mr4() {
        return false;
    }
    if !matches!(
        ttype,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
    ) {
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
    if x86_qk_mr4_uses_prequant(ttype) {
        let preq = prequantize_activation_q8(&x[..n], n, scratch);
        if d >= par_matmul_min_rows() {
            xout.par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_row = chunk_idx * chunk_rows;
                    matmul_qk_mr4_chunk_x86_prequant(
                        chunk,
                        base_row,
                        &preq,
                        x,
                        mapped,
                        data_offset,
                        row_size,
                        n,
                        ttype,
                    );
                });
        } else {
            matmul_qk_mr4_chunk_x86_prequant(
                xout,
                0,
                &preq,
                x,
                mapped,
                data_offset,
                row_size,
                n,
                ttype,
            );
        }
    } else {
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
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn try_matmul_qk_mr4_x86_prepared(
    xout: &mut [f32],
    prepared: &MatmulPreparedActivation<'_>,
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    ttype: i32,
) -> bool {
    if !use_x86_qk_mr4() || !x86_qk_mr4_uses_prequant(ttype) {
        return false;
    }
    let n = prepared.n;
    if n < QK_K || n % QK_K != 0 {
        return false;
    }

    let d = xout.len();
    if d < 4 {
        return false;
    }
    if !validate_qk_mr4_once_x86(prepared.x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }

    let chunk_rows = par_matmul_chunk_rows();
    if d >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_qk_mr4_chunk_x86_prequant(
                    chunk,
                    base_row,
                    &prepared.q8,
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    ttype,
                );
            });
    } else {
        matmul_qk_mr4_chunk_x86_prequant(
            xout,
            0,
            &prepared.q8,
            prepared.x,
            mapped,
            data_offset,
            row_size,
            n,
            ttype,
        );
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

#[cfg(target_arch = "aarch64")]
pub(crate) fn vec_dot_iq4_nl_4rows(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK4_NL;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK4_NL;
        let x_ptr = unsafe { x.as_ptr().add(x_base) };
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let q = rows[r][off + 2..off + 2 + QK4_NL / 2].as_ptr();
            let (lo, hi) =
                unsafe { dot_iq4_nl_nibbles_16_neon_ptr(x_ptr, x_ptr.add(QK4_NL / 2), q) };
            sums[r] += (lo + hi) * d;
        }
    }

    sums
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn vec_dot_iq4_nl_4rows_x86(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    if use_x86_avx2_fma() {
        unsafe {
            return vec_dot_iq4_nl_4rows_x86_avx2(x, w0, w1, w2, w3, n);
        }
    }
    [
        vec_dot_iq4_nl(x, w0, n),
        vec_dot_iq4_nl(x, w1, n),
        vec_dot_iq4_nl(x, w2, n),
        vec_dot_iq4_nl(x, w3, n),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3,avx2,fma")]
unsafe fn vec_dot_iq4_nl_4rows_x86_avx2(
    x: &[f32],
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [w0, w1, w2, w3];
    let nb = n / QK4_NL;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let x_base = i * QK4_NL;
        let x_ptr = x.as_ptr().add(x_base);
        for r in 0..4 {
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(rows[r], off));
            let q = rows[r][off + 2..off + 2 + QK4_NL / 2].as_ptr();
            let (lo, hi) =
                dot_iq4_nl_nibbles_pair_avx2_ptr(x_ptr, x_ptr.add(QK4_NL / 2), q, QK4_NL / 2);
            sums[r] += (lo + hi) * d;
        }
    }

    sums
}

pub(crate) fn get_row_size(n_cols: usize, ttype: GgmlType) -> usize {
    let block_size = get_block_size(ttype);
    let type_size = get_type_size(ttype);
    (n_cols / block_size) * type_size
}

#[inline]
fn run_q8_rows_prequant<F>(
    xout: &mut [f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    d: usize,
    dot: F,
) where
    F: Fn(&[u8]) -> f32 + Sync,
{
    if d >= par_matmul_min_rows() {
        let chunk_rows = par_matmul_chunk_rows();
        xout[..d]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_row = chunk_idx * chunk_rows;
                for (j, out) in out_chunk.iter_mut().enumerate() {
                    matmul_prefetch_row(mapped, data_offset, row_size, base_row + j, d);
                    let row_off = data_offset + (base_row + j) * row_size;
                    let row = &mapped[row_off..row_off + row_size];
                    *out = dot(row);
                }
            });
    } else {
        for (i, out) in xout[..d].iter_mut().enumerate() {
            matmul_prefetch_row(mapped, data_offset, row_size, i, d);
            let row_off = data_offset + i * row_size;
            let row = &mapped[row_off..row_off + row_size];
            *out = dot(row);
        }
    }
}

pub(crate) fn matmul_quantized(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
) -> Result<(), String> {
    let mut scratch = MatmulActivationScratch::new();
    matmul_quantized_with_scratch(xout, x, qw, mapped, &mut scratch)
}

pub(crate) fn matmul_quantized_with_prepared_activation(
    xout: &mut [f32],
    prepared: &MatmulPreparedActivation<'_>,
    qw: &QuantizedTensor,
    mapped: &[u8],
) -> Result<(), String> {
    matmul_quantized_rows_with_prepared_activation(xout, prepared, qw, 0, qw.rows, mapped)
}

pub(crate) fn matmul_quantized_with_scratch(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    scratch: &mut MatmulActivationScratch,
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
                            matmul_prefetch_row(mapped, data_offset, row_size, base_row + j, d);
                            let row_off = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_off..row_off + row_size];
                            *out = $dot(x, row, n);
                        }
                    },
                );
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    matmul_prefetch_row(mapped, data_offset, row_size, i, d);
                    let row_off = data_offset + i * row_size;
                    let row = &mapped[row_off..row_off + row_size];
                    *out = $dot(x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_0,
                ) {
                    run_rows!(vec_dot_q4_0);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_0,
                ) {
                    run_rows!(vec_dot_q4_0);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_0);
            }
        }
        GGML_TYPE_Q4_1 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_1,
                ) {
                    run_rows!(vec_dot_q4_1);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_1,
                ) {
                    run_rows!(vec_dot_q4_1);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_1);
            }
        }
        GGML_TYPE_Q5_0 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_0,
                ) {
                    run_rows!(vec_dot_q5_0);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_0,
                ) {
                    run_rows!(vec_dot_q5_0);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_0);
            }
        }
        GGML_TYPE_Q5_1 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_1,
                ) {
                    run_rows!(vec_dot_q5_1);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_1,
                ) {
                    run_rows!(vec_dot_q5_1);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_1);
            }
        }
        GGML_TYPE_Q8_0 => {
            let mut handled = false;
            #[cfg(target_arch = "aarch64")]
            {
                if try_matmul_q8_mr2_with_scratch(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    scratch,
                ) {
                    handled = true;
                } else if n >= QK8_0 && n.is_multiple_of(QK8_0) && use_aarch64_dotprod_q8() {
                    let preq = prequantize_activation_q8(&x[..n], n, scratch);
                    run_q8_rows_prequant(xout, mapped, data_offset, row_size, d, |row| unsafe {
                        vec_dot_q8_0_dotprod_prequant(&preq, row, n)
                    });
                    handled = true;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                // Prefer the exact AVX2/scalar Q8 path on x86. The VNNI kernels
                // re-quantize activations to int8, which is faster but can perturb
                // logits enough to change greedy decoding.
                if !use_x86_avx2_fma() && n >= QK8_0 && n.is_multiple_of(QK8_0) {
                    if use_x86_avx512_vnni_q8() {
                        let preq = prequantize_activation_q8(&x[..n], n, scratch);
                        run_q8_rows_prequant(
                            xout,
                            mapped,
                            data_offset,
                            row_size,
                            d,
                            |row| unsafe { vec_dot_q8_0_x86_avx512vnni_prequant(&preq, row, n) },
                        );
                        handled = true;
                    } else if use_x86_avx_vnni() {
                        let preq = prequantize_activation_q8(&x[..n], n, scratch);
                        run_q8_rows_prequant(
                            xout,
                            mapped,
                            data_offset,
                            row_size,
                            d,
                            |row| unsafe { vec_dot_q8_0_x86_avxvnni_prequant(&preq, row, n) },
                        );
                        handled = true;
                    }
                }
            }
            if !handled {
                run_rows!(vec_dot_q8_0);
            }
        }
        GGML_TYPE_Q2_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q2_K,
                ) {
                    run_rows!(vec_dot_q2_k);
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
                    GGML_TYPE_Q2_K,
                    scratch,
                ) {
                    run_rows!(vec_dot_q2_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q2_k);
            }
        }
        GGML_TYPE_Q3_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q3_K,
                ) {
                    run_rows!(vec_dot_q3_k);
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
                    GGML_TYPE_Q3_K,
                    scratch,
                ) {
                    run_rows!(vec_dot_q3_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q3_k);
            }
        }
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
                    scratch,
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
                    scratch,
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
                    scratch,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_IQ4_NL,
                ) {
                    run_rows!(vec_dot_iq4_nl);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_IQ4_NL,
                ) {
                    run_rows!(vec_dot_iq4_nl);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_iq4_nl);
            }
        }
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => run_rows!(vec_dot_bin1),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ));
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
    let mut scratch = MatmulActivationScratch::new();
    matmul_quantized_rows_with_scratch(xout, x, qw, row_start, n_rows, mapped, &mut scratch)
}

pub(crate) fn matmul_quantized_rows_with_prepared_activation(
    xout: &mut [f32],
    prepared: &MatmulPreparedActivation<'_>,
    qw: &QuantizedTensor,
    row_start: usize,
    n_rows: usize,
    mapped: &[u8],
) -> Result<(), String> {
    let prof_t0 = prof_start();
    let d = n_rows;
    let n = qw.cols;
    let row_size = get_row_size(n, qw.ttype);
    if prepared.n != n {
        return Err("matmul prepared activation shape mismatch".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul row window exceeds tensor rows".to_string());
    }
    if xout.len() < d {
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
                            matmul_prefetch_row(mapped, data_offset, row_size, base_row + j, d);
                            let row_start = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_start..row_start + row_size];
                            *out = $dot(prepared.x, row, n);
                        }
                    },
                );
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    matmul_prefetch_row(mapped, data_offset, row_size, i, d);
                    let row_start = data_offset + i * row_size;
                    let row = &mapped[row_start..row_start + row_size];
                    *out = $dot(prepared.x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_0,
                ) {
                    run_rows!(vec_dot_q4_0);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_0,
                ) {
                    run_rows!(vec_dot_q4_0);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_0);
            }
        }
        GGML_TYPE_Q4_1 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_1,
                ) {
                    run_rows!(vec_dot_q4_1);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_1,
                ) {
                    run_rows!(vec_dot_q4_1);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_1);
            }
        }
        GGML_TYPE_Q5_0 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_0,
                ) {
                    run_rows!(vec_dot_q5_0);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_0,
                ) {
                    run_rows!(vec_dot_q5_0);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_0);
            }
        }
        GGML_TYPE_Q5_1 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_1,
                ) {
                    run_rows!(vec_dot_q5_1);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_1,
                ) {
                    run_rows!(vec_dot_q5_1);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_1);
            }
        }
        GGML_TYPE_Q8_0 => {
            #[cfg(target_arch = "x86_64")]
            {
                let mut handled = false;
                if !use_x86_avx2_fma() && n >= QK8_0 && n.is_multiple_of(QK8_0) {
                    if use_x86_avx512_vnni_q8() {
                        run_q8_rows_prequant(
                            xout,
                            mapped,
                            data_offset,
                            row_size,
                            d,
                            |row| unsafe {
                                vec_dot_q8_0_x86_avx512vnni_prequant(&prepared.q8, row, n)
                            },
                        );
                        handled = true;
                    } else if use_x86_avx_vnni() {
                        run_q8_rows_prequant(
                            xout,
                            mapped,
                            data_offset,
                            row_size,
                            d,
                            |row| unsafe {
                                vec_dot_q8_0_x86_avxvnni_prequant(&prepared.q8, row, n)
                            },
                        );
                        handled = true;
                    }
                }
                if !handled {
                    run_rows!(vec_dot_q8_0);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                run_rows!(vec_dot_q8_0);
            }
        }
        GGML_TYPE_Q2_K => run_rows!(vec_dot_q2_k),
        GGML_TYPE_Q3_K => run_rows!(vec_dot_q3_k),
        GGML_TYPE_Q4_K => {
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86_prepared(
                    &mut xout[..d],
                    prepared,
                    mapped,
                    data_offset,
                    row_size,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                run_rows!(vec_dot_q4_k);
            }
        }
        GGML_TYPE_Q5_K => {
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86_prepared(
                    &mut xout[..d],
                    prepared,
                    mapped,
                    data_offset,
                    row_size,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                run_rows!(vec_dot_q5_k);
            }
        }
        GGML_TYPE_Q6_K => {
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86_prepared(
                    &mut xout[..d],
                    prepared,
                    mapped,
                    data_offset,
                    row_size,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    prepared.n,
                    GGML_TYPE_IQ4_NL,
                ) {
                    run_rows!(vec_dot_iq4_nl);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    prepared.x,
                    mapped,
                    data_offset,
                    row_size,
                    prepared.n,
                    GGML_TYPE_IQ4_NL,
                ) {
                    run_rows!(vec_dot_iq4_nl);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_iq4_nl);
            }
        }
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => run_rows!(vec_dot_bin1),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ));
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

pub(crate) fn matmul_quantized_rows_with_scratch(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    row_start: usize,
    n_rows: usize,
    mapped: &[u8],
    scratch: &mut MatmulActivationScratch,
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
                            matmul_prefetch_row(mapped, data_offset, row_size, base_row + j, d);
                            let row_start = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_start..row_start + row_size];
                            *out = $dot(x, row, n);
                        }
                    },
                );
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    matmul_prefetch_row(mapped, data_offset, row_size, i, d);
                    let row_start = data_offset + i * row_size;
                    let row = &mapped[row_start..row_start + row_size];
                    *out = $dot(x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_0,
                ) {
                    run_rows!(vec_dot_q4_0);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_0,
                ) {
                    run_rows!(vec_dot_q4_0);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_0);
            }
        }
        GGML_TYPE_Q4_1 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_1,
                ) {
                    run_rows!(vec_dot_q4_1);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_1,
                ) {
                    run_rows!(vec_dot_q4_1);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_1);
            }
        }
        GGML_TYPE_Q5_0 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_0,
                ) {
                    run_rows!(vec_dot_q5_0);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_0,
                ) {
                    run_rows!(vec_dot_q5_0);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_0);
            }
        }
        GGML_TYPE_Q5_1 => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_1,
                ) {
                    run_rows!(vec_dot_q5_1);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_1,
                ) {
                    run_rows!(vec_dot_q5_1);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_1);
            }
        }
        GGML_TYPE_Q8_0 => {
            let mut handled = false;
            #[cfg(target_arch = "aarch64")]
            {
                if try_matmul_q8_mr2_with_scratch(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    scratch,
                ) {
                    handled = true;
                } else if n >= QK8_0 && n.is_multiple_of(QK8_0) && use_aarch64_dotprod_q8() {
                    let preq = prequantize_activation_q8(&x[..n], n, scratch);
                    run_q8_rows_prequant(xout, mapped, data_offset, row_size, d, |row| unsafe {
                        vec_dot_q8_0_dotprod_prequant(&preq, row, n)
                    });
                    handled = true;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                // Prefer the exact AVX2/scalar Q8 path on x86. The VNNI kernels
                // re-quantize activations to int8, which is faster but can perturb
                // logits enough to change greedy decoding.
                if !use_x86_avx2_fma() && n >= QK8_0 && n.is_multiple_of(QK8_0) {
                    if use_x86_avx512_vnni_q8() {
                        let preq = prequantize_activation_q8(&x[..n], n, scratch);
                        run_q8_rows_prequant(
                            xout,
                            mapped,
                            data_offset,
                            row_size,
                            d,
                            |row| unsafe { vec_dot_q8_0_x86_avx512vnni_prequant(&preq, row, n) },
                        );
                        handled = true;
                    } else if use_x86_avx_vnni() {
                        let preq = prequantize_activation_q8(&x[..n], n, scratch);
                        run_q8_rows_prequant(
                            xout,
                            mapped,
                            data_offset,
                            row_size,
                            d,
                            |row| unsafe { vec_dot_q8_0_x86_avxvnni_prequant(&preq, row, n) },
                        );
                        handled = true;
                    }
                }
            }
            if !handled {
                run_rows!(vec_dot_q8_0);
            }
        }
        GGML_TYPE_Q2_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q2_K,
                ) {
                    run_rows!(vec_dot_q2_k);
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
                    GGML_TYPE_Q2_K,
                    scratch,
                ) {
                    run_rows!(vec_dot_q2_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q2_k);
            }
        }
        GGML_TYPE_Q3_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q3_K,
                ) {
                    run_rows!(vec_dot_q3_k);
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
                    GGML_TYPE_Q3_K,
                    scratch,
                ) {
                    run_rows!(vec_dot_q3_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q3_k);
            }
        }
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
                    scratch,
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
                    scratch,
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
                    scratch,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_legacy_q4_q5_iq4_mr4_aarch64(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_IQ4_NL,
                ) {
                    run_rows!(vec_dot_iq4_nl);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_legacy_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_IQ4_NL,
                ) {
                    run_rows!(vec_dot_iq4_nl);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_iq4_nl);
            }
        }
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        GGML_TYPE_BIN1_40 | GGML_TYPE_BIN1_41 => run_rows!(vec_dot_bin1),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ));
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
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

    let use_grouped = n_group > 1 && topk_group < n_group && logits.len().is_multiple_of(n_group);
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

/// Batch matmul: `out[m × n_rows] = inp[m × n_cols] × qw[n_rows × n_cols]ᵀ`
///
/// For true batches (`m > 1`), each quantized weight row is dequantized once into a reusable
/// scratch buffer and then dotted against all `m` input rows. This avoids re-reading and
/// re-decoding the same quantized row for every token in the batch, which is critical for
/// BERT-style embedding prefills.
///
/// For the single-row case (`m <= 1`), this falls back to `matmul_quantized_rows` so the
/// architecture-specific micro-kernels remain in use for standard inference paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_quantized_batch_with_scratch(
    out: &mut [f32],      // [m × n_rows]
    inp: &[f32],          // [m × n_cols]
    qw: &QuantizedTensor, // [qw.rows × n_cols]
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    dequantized_row: &mut Vec<f32>,
) -> Result<(), String> {
    let n = qw.cols;
    if row_start + n_rows > qw.rows {
        return Err(format!(
            "matmul_quantized_batch: row window [{row_start},{}) out of bounds (qw.rows={})",
            row_start + n_rows,
            qw.rows
        ));
    }
    if out.len() < m * n_rows || inp.len() < m * n {
        return Err(format!(
            "matmul_quantized_batch: shape mismatch (out={}, need {}; inp={}, need {})",
            out.len(),
            m * n_rows,
            inp.len(),
            m * n
        ));
    }

    if m <= 1 {
        for j in 0..m {
            let x = &inp[j * n..(j + 1) * n];
            let out_row = &mut out[j * n_rows..(j + 1) * n_rows];
            matmul_quantized_rows(out_row, x, qw, row_start, n_rows, mapped)?;
        }
        return Ok(());
    }

    let prof_t0 = prof_start();
    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = n_rows
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    if dequantized_row.len() < n {
        dequantized_row.resize(n, 0.0);
    }
    for row_idx in 0..n_rows {
        matmul_prefetch_row(mapped, data_offset, row_size, row_idx, n_rows);
        let row_start = data_offset + row_idx * row_size;
        let row = &mapped[row_start..row_start + row_size];
        dequantize_row_into(qw.ttype, row, &mut dequantized_row[..n], n)?;
        for batch_idx in 0..m {
            let x = &inp[batch_idx * n..(batch_idx + 1) * n];
            out[batch_idx * n_rows + row_idx] = dot_f32_simd(x, &dequantized_row[..n]);
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn matmul_quantized_batch(
    out: &mut [f32],      // [m × n_rows]
    inp: &[f32],          // [m × n_cols]
    qw: &QuantizedTensor, // [qw.rows × n_cols]
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    let mut dequantized_row = Vec::new();
    matmul_quantized_batch_with_scratch(
        out,
        inp,
        qw,
        mapped,
        m,
        row_start,
        n_rows,
        &mut dequantized_row,
    )
}

// ---------------------------------------------------------------------------
// Exact batched matmul (batched prefill)
// ---------------------------------------------------------------------------

/// Whether [`matmul_quantized_batch_exact`] supports `ttype`. Types with
/// arch-dependent activation-prequant paths (Q8_0) or float weights must use
/// per-token [`matmul_quantized`] instead — the caller falls back per tensor.
pub(crate) fn batch_exact_supported(ttype: GgmlType) -> bool {
    matches!(
        ttype.0,
        GGML_TYPE_Q4_0
            | GGML_TYPE_Q4_1
            | GGML_TYPE_Q5_0
            | GGML_TYPE_Q5_1
            | GGML_TYPE_Q2_K
            | GGML_TYPE_Q3_K
            | GGML_TYPE_Q4_K
            | GGML_TYPE_Q5_K
            | GGML_TYPE_Q6_K
            | GGML_TYPE_IQ4_NL
    )
}

#[inline]
fn batch_exact_vec_dot(ttype: i32, x: &[f32], row: &[u8], n: usize) -> f32 {
    match ttype {
        GGML_TYPE_Q4_0 => vec_dot_q4_0(x, row, n),
        GGML_TYPE_Q4_1 => vec_dot_q4_1(x, row, n),
        GGML_TYPE_Q5_0 => vec_dot_q5_0(x, row, n),
        GGML_TYPE_Q5_1 => vec_dot_q5_1(x, row, n),
        GGML_TYPE_Q2_K => vec_dot_q2_k(x, row, n),
        GGML_TYPE_Q3_K => vec_dot_q3_k(x, row, n),
        GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
        GGML_TYPE_IQ4_NL => vec_dot_iq4_nl(x, row, n),
        _ => unreachable!("batch_exact_vec_dot: unsupported type {ttype}"),
    }
}

/// First exact tiled K-quant batch kernel.
///
/// This Q2_K implementation keeps the scalar [`vec_dot_q2_k`] accumulation
/// order for each `(token,row)` output while moving the batch dimension inside
/// the packed-weight decode loop. Each 16-value Q2 segment is decoded once per
/// row and reused across all `m` input tokens, avoiding full-row F32 dequant
/// scratch and repeated row scans.
///
/// It is intentionally not the default `batch_exact` dispatch yet: the current
/// exact prefill path mirrors architecture-specific sequential MR4 dispatch for
/// bit identity, while this helper is bit-identical to the scalar Q2_K oracle.
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_q2_k_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_Q2_K {
        return Err("matmul_quantized_batch_tiled_q2_k_exact: unsupported type".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_tiled_q2_k_exact row window out of bounds".to_string());
    }
    if out.len() < m * n_rows || inp.len() < m * n {
        return Err("matmul_quantized_batch_tiled_q2_k_exact shape mismatch".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if n < QK_K || !n.is_multiple_of(QK_K) {
        return Err("matmul_quantized_batch_tiled_q2_k_exact unsupported column count".to_string());
    }

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = n_rows
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    out[..m * n_rows].fill(0.0);

    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mut block_sums = vec![0.0f32; m];
    for row_idx in 0..n_rows {
        matmul_prefetch_row(mapped, data_offset, row_size, row_idx, n_rows);
        let row_base = data_offset + row_idx * row_size;
        let row = &mapped[row_base..row_base + row_size];

        for i in 0..nb {
            block_sums.fill(0.0);
            let off = i * block_sz;
            let scales = &row[off..off + QK_K / 16];
            let mut q_off = off + QK_K / 16;
            let d = fp16_to_fp32(read_u16_le(row, off + QK_K / 16 + QK_K / 4));
            let dmin = fp16_to_fp32(read_u16_le(row, off + QK_K / 16 + QK_K / 4 + 2));
            let x_block_base = i * QK_K;

            let mut is = 0usize;
            for n_outer in (0..QK_K).step_by(128) {
                let q = &row[q_off..q_off + 32];
                let mut shift = 0usize;
                for j in 0..4usize {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0f) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for l in 0..16usize {
                        let col = x_block_base + n_outer + j * 32 + l;
                        let wv = dl * ((q[l] >> shift) & 0x03) as f32 - ml;
                        for b in 0..m {
                            block_sums[b] += inp[b * n + col] * wv;
                        }
                    }

                    let sc2 = scales[is];
                    is += 1;
                    let dl2 = d * (sc2 & 0x0f) as f32;
                    let ml2 = dmin * (sc2 >> 4) as f32;
                    for l in 0..16usize {
                        let col = x_block_base + n_outer + j * 32 + 16 + l;
                        let wv = dl2 * ((q[l + 16] >> shift) & 0x03) as f32 - ml2;
                        for b in 0..m {
                            block_sums[b] += inp[b * n + col] * wv;
                        }
                    }
                    shift += 2;
                }
                q_off += 32;
            }
            for b in 0..m {
                out[b * n_rows + row_idx] += block_sums[b];
            }
        }
    }

    Ok(())
}

/// Exact tiled Q3_K batch kernel.
///
/// Mirrors [`vec_dot_q3_k`] for each `(token,row)` output, including the
/// per-superblock accumulation boundary, while reusing each decoded 16-value
/// Q3 segment across all batch tokens. Kept internal until tiled exact dispatch
/// policy is settled for all K-quants.
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_q3_k_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_Q3_K {
        return Err("matmul_quantized_batch_tiled_q3_k_exact: unsupported type".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_tiled_q3_k_exact row window out of bounds".to_string());
    }
    if out.len() < m * n_rows || inp.len() < m * n {
        return Err("matmul_quantized_batch_tiled_q3_k_exact shape mismatch".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if n < QK_K || !n.is_multiple_of(QK_K) {
        return Err("matmul_quantized_batch_tiled_q3_k_exact unsupported column count".to_string());
    }

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = n_rows
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    out[..m * n_rows].fill(0.0);

    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let sc_off = QK_K / 8 + QK_K / 4;
    let mut block_sums = vec![0.0f32; m];
    for row_idx in 0..n_rows {
        matmul_prefetch_row(mapped, data_offset, row_size, row_idx, n_rows);
        let row_base = data_offset + row_idx * row_size;
        let row = &mapped[row_base..row_base + row_size];

        for i in 0..nb {
            block_sums.fill(0.0);
            let off = i * block_sz;
            let hmask = &row[off..off + QK_K / 8];
            let mut q_off = off + QK_K / 8;
            let scales = q3_scales(&row[off + sc_off..off + sc_off + 12]);
            let d_all = fp16_to_fp32(read_u16_le(row, off + sc_off + 12));
            let x_block_base = i * QK_K;

            let mut is = 0usize;
            let mut mbit: u8 = 1;
            for n_outer in (0..QK_K).step_by(128) {
                let q = &row[q_off..q_off + 32];
                let mut shift = 0usize;
                for j in 0..4usize {
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16usize {
                        let col = x_block_base + n_outer + j * 32 + l;
                        let wv = ((q[l] >> shift) & 3) as i8
                            - if (hmask[l] & mbit) != 0 { 0 } else { 4 };
                        for b in 0..m {
                            block_sums[b] += inp[b * n + col] * dl * wv as f32;
                        }
                    }

                    let dl2 = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16usize {
                        let col = x_block_base + n_outer + j * 32 + 16 + l;
                        let wv = ((q[l + 16] >> shift) & 3) as i8
                            - if (hmask[l + 16] & mbit) != 0 { 0 } else { 4 };
                        for b in 0..m {
                            block_sums[b] += inp[b * n + col] * dl2 * wv as f32;
                        }
                    }
                    shift += 2;
                    mbit <<= 1;
                }
                q_off += 32;
            }
            for b in 0..m {
                out[b * n_rows + row_idx] += block_sums[b];
            }
        }
    }

    Ok(())
}

/// Exact tiled Q4_K batch kernel.
///
/// Reuses each decoded 32-value nibble pair across all batch tokens while
/// preserving [`vec_dot_q4_k`] multiply/add grouping for every output.
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_q4_k_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_Q4_K {
        return Err("matmul_quantized_batch_tiled_q4_k_exact: unsupported type".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_tiled_q4_k_exact row window out of bounds".to_string());
    }
    if out.len() < m * n_rows || inp.len() < m * n {
        return Err("matmul_quantized_batch_tiled_q4_k_exact shape mismatch".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if n < QK_K || !n.is_multiple_of(QK_K) {
        return Err("matmul_quantized_batch_tiled_q4_k_exact unsupported column count".to_string());
    }

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = n_rows
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    out[..m * n_rows].fill(0.0);

    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut block_sums = vec![0.0f32; m];
    for row_idx in 0..n_rows {
        matmul_prefetch_row(mapped, data_offset, row_size, row_idx, n_rows);
        let row_base = data_offset + row_idx * row_size;
        let row = &mapped[row_base..row_base + row_size];

        for i in 0..nb {
            block_sums.fill(0.0);
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(row, off));
            let dmin = fp16_to_fp32(read_u16_le(row, off + 2));
            let scales = &row[off + 4..off + 16];
            let mut q_off = off + 16;
            let x_block_base = i * QK_K;

            let mut is = 0usize;
            for j in (0..QK_K).step_by(64) {
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let d1 = d * sc1 as f32;
                let m1f = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let m2f = dmin * m2 as f32;
                let q = &row[q_off..q_off + 32];
                for l in 0..32usize {
                    let qv = q[l];
                    let w0 = d1 * (qv & 0x0f) as f32 - m1f;
                    let w1 = d2 * (qv >> 4) as f32 - m2f;
                    let col0 = x_block_base + j + l;
                    let col1 = x_block_base + j + 32 + l;
                    for b in 0..m {
                        block_sums[b] += inp[b * n + col0] * w0 + inp[b * n + col1] * w1;
                    }
                }
                q_off += 32;
                is += 2;
            }
            for b in 0..m {
                out[b * n_rows + row_idx] += block_sums[b];
            }
        }
    }

    Ok(())
}

/// Exact tiled Q5_K batch kernel.
///
/// Keeps [`vec_dot_q5_k`] bit semantics while sharing low/high-bit decode work
/// across every token in the prefill batch.
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_q5_k_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_Q5_K {
        return Err("matmul_quantized_batch_tiled_q5_k_exact: unsupported type".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_tiled_q5_k_exact row window out of bounds".to_string());
    }
    if out.len() < m * n_rows || inp.len() < m * n {
        return Err("matmul_quantized_batch_tiled_q5_k_exact shape mismatch".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if n < QK_K || !n.is_multiple_of(QK_K) {
        return Err("matmul_quantized_batch_tiled_q5_k_exact unsupported column count".to_string());
    }

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = n_rows
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    out[..m * n_rows].fill(0.0);

    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut block_sums = vec![0.0f32; m];
    for row_idx in 0..n_rows {
        matmul_prefetch_row(mapped, data_offset, row_size, row_idx, n_rows);
        let row_base = data_offset + row_idx * row_size;
        let row = &mapped[row_base..row_base + row_size];

        for i in 0..nb {
            block_sums.fill(0.0);
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(row, off));
            let dmin = fp16_to_fp32(read_u16_le(row, off + 2));
            let scales = &row[off + 4..off + 16];
            let qh = &row[off + 16..off + 16 + QK_K / 8];
            let mut ql_off = off + 16 + QK_K / 8;
            let x_block_base = i * QK_K;

            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for j in (0..QK_K).step_by(64) {
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let d1 = d * sc1 as f32;
                let m1f = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let m2f = dmin * m2 as f32;
                let ql = &row[ql_off..ql_off + 32];

                for l in 0..32usize {
                    let qv = ql[l];
                    let lo = (qv & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                    let hi = (qv >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                    let w0 = d1 * lo as f32 - m1f;
                    let w1 = d2 * hi as f32 - m2f;
                    let col0 = x_block_base + j + l;
                    let col1 = x_block_base + j + 32 + l;
                    for b in 0..m {
                        block_sums[b] += inp[b * n + col0] * w0 + inp[b * n + col1] * w1;
                    }
                }

                ql_off += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            for b in 0..m {
                out[b * n_rows + row_idx] += block_sums[b];
            }
        }
    }

    Ok(())
}

/// Exact tiled Q6_K batch kernel.
///
/// Preserves [`vec_dot_q6_k`] per-lane accumulation order and shares the
/// four-way Q6 decode for all tokens in the current batch.
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_q6_k_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_Q6_K {
        return Err("matmul_quantized_batch_tiled_q6_k_exact: unsupported type".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_tiled_q6_k_exact row window out of bounds".to_string());
    }
    if out.len() < m * n_rows || inp.len() < m * n {
        return Err("matmul_quantized_batch_tiled_q6_k_exact shape mismatch".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if n < QK_K || !n.is_multiple_of(QK_K) {
        return Err("matmul_quantized_batch_tiled_q6_k_exact unsupported column count".to_string());
    }

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = n_rows
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    out[..m * n_rows].fill(0.0);

    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut block_sums = vec![0.0f32; m];
    for row_idx in 0..n_rows {
        matmul_prefetch_row(mapped, data_offset, row_size, row_idx, n_rows);
        let row_base = data_offset + row_idx * row_size;
        let row = &mapped[row_base..row_base + row_size];

        for i in 0..nb {
            block_sums.fill(0.0);
            let off = i * block_sz;
            let d = fp16_to_fp32(read_u16_le(row, off + QK_K / 2 + QK_K / 4 + QK_K / 16));
            let mut ql_off = off;
            let mut qh_off = off + QK_K / 2;
            let mut sc_off = off + QK_K / 2 + QK_K / 4;
            let x_block_base = i * QK_K;

            for n_outer in (0..QK_K).step_by(128) {
                let ql = &row[ql_off..ql_off + 64];
                let qh = &row[qh_off..qh_off + 32];
                let sc = &row[sc_off..sc_off + 8];

                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = (((ql[l] & 0x0f) | (((qh[l]) & 0x03) << 4)) as i8) - 32;
                    let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                    let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                    let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                    let s0 = d * sc[is] as i8 as f32;
                    let s1 = d * sc[is + 2] as i8 as f32;
                    let s2 = d * sc[is + 4] as i8 as f32;
                    let s3 = d * sc[is + 6] as i8 as f32;
                    let col0 = x_block_base + n_outer + l;
                    let col1 = x_block_base + n_outer + 32 + l;
                    let col2 = x_block_base + n_outer + 64 + l;
                    let col3 = x_block_base + n_outer + 96 + l;
                    for b in 0..m {
                        block_sums[b] += inp[b * n + col0] * (s0 * q1 as f32);
                        block_sums[b] += inp[b * n + col1] * (s1 * q2 as f32);
                        block_sums[b] += inp[b * n + col2] * (s2 * q3 as f32);
                        block_sums[b] += inp[b * n + col3] * (s3 * q4 as f32);
                    }
                }

                ql_off += 64;
                qh_off += 32;
                sc_off += 8;
            }
            for b in 0..m {
                out[b * n_rows + row_idx] += block_sums[b];
            }
        }
    }

    Ok(())
}

#[inline]
fn tiled_k_exact_supported(ttype: GgmlType) -> bool {
    matches!(
        ttype.0,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
    )
}

/// Scalar-exact tiled K-quant dispatcher.
///
/// This is currently used only for small exact-mode row windows when the
/// architecture-specific MR4 mirror is inactive. That avoids changing default
/// fast-prefill policy while still eliminating per-token scalar row scans for
/// the exact fallback/tail shape.
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_k_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    match qw.ttype.0 {
        GGML_TYPE_Q2_K => {
            matmul_quantized_batch_tiled_q2_k_exact(out, inp, qw, mapped, m, row_start, n_rows)
        }
        GGML_TYPE_Q3_K => {
            matmul_quantized_batch_tiled_q3_k_exact(out, inp, qw, mapped, m, row_start, n_rows)
        }
        GGML_TYPE_Q4_K => {
            matmul_quantized_batch_tiled_q4_k_exact(out, inp, qw, mapped, m, row_start, n_rows)
        }
        GGML_TYPE_Q5_K => {
            matmul_quantized_batch_tiled_q5_k_exact(out, inp, qw, mapped, m, row_start, n_rows)
        }
        GGML_TYPE_Q6_K => {
            matmul_quantized_batch_tiled_q6_k_exact(out, inp, qw, mapped, m, row_start, n_rows)
        }
        ttype => Err(format!(
            "matmul_quantized_batch_tiled_k_exact: unsupported type {ttype}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_k_exact_row_major(
    out_row_major: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
) -> Result<(), String> {
    if out_row_major.len() < m * n_rows {
        return Err("matmul_quantized_batch_tiled_k_exact_row_major shape mismatch".to_string());
    }

    let mut token_major = vec![0.0f32; m * n_rows];
    matmul_quantized_batch_tiled_k_exact(&mut token_major, inp, qw, mapped, m, row_start, n_rows)?;
    for b in 0..m {
        for r in 0..n_rows {
            out_row_major[r * m + b] = token_major[b * n_rows + r];
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_tiled_k_exact_with_scratch(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    tmp: &mut Vec<f32>,
) -> Result<(), String> {
    let d = n_rows;
    if d >= par_matmul_min_rows() {
        if tmp.len() < d * m {
            tmp.resize(d * m, 0.0);
        }
        let chunk_rows = par_matmul_chunk_rows();
        let results: Vec<Result<(), String>> = tmp[..d * m]
            .par_chunks_mut(chunk_rows * m)
            .enumerate()
            .map(|(chunk_idx, block)| {
                let base_row = chunk_idx * chunk_rows;
                let rows_here = block.len() / m;
                matmul_quantized_batch_tiled_k_exact_row_major(
                    block,
                    inp,
                    qw,
                    mapped,
                    m,
                    row_start + base_row,
                    rows_here,
                )
            })
            .collect();
        for result in results {
            result?;
        }
        for b in 0..m {
            for r in 0..d {
                out[b * d + r] = tmp[r * m + b];
            }
        }
        return Ok(());
    }

    matmul_quantized_batch_tiled_k_exact(out, inp, qw, mapped, m, row_start, n_rows)
}

/// Same 4-row kernel selection as the sequential MR4 chunk functions, so the
/// batched path is bit-identical per row block on each architecture.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
fn batch_exact_vec_dot_4rows(
    ttype: i32,
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        match ttype {
            GGML_TYPE_Q4_0 => vec_dot_q4_0_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_1 => vec_dot_q4_1_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_0 => vec_dot_q5_0_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_1 => vec_dot_q5_1_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_IQ4_NL => vec_dot_iq4_nl_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q3_K => vec_dot_q3_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match ttype {
            GGML_TYPE_Q4_0 => vec_dot_q4_0_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_1 => vec_dot_q4_1_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_0 => vec_dot_q5_0_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_1 => vec_dot_q5_1_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_IQ4_NL => vec_dot_iq4_nl_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q2_K => vec_dot_q2_k_4rows_aarch64(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q3_K => vec_dot_q3_k_4rows_aarch64(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        }
    }
}

/// Batched quantized matmul with numerics **bit-identical** to calling
/// [`matmul_quantized`] once per token.
///
/// It mirrors the per-token kernel dispatch exactly — including the MR4
/// 4-row kernels, their per-type runtime validation gate, and the singles
/// tail — so a greedy decode over a batched prefill matches the sequential
/// path bit for bit. The bandwidth win comes from moving tokens into the
/// inner loop: each weight row (or 4-row block) is streamed from memory once
/// per batch instead of once per token.
///
/// `out` is token-major `[m × qw.rows]`, `inp` token-major `[m × qw.cols]`.
/// `tmp` is caller-owned scratch (`qw.rows × m` floats), kept across calls.
/// Callers must check [`batch_exact_supported`] first.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_quantized_batch_exact(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    tmp: &mut Vec<f32>,
) -> Result<(), String> {
    let d = n_rows;
    let n = qw.cols;
    let ttype = qw.ttype.0;
    if !batch_exact_supported(qw.ttype) {
        return Err(format!(
            "matmul_quantized_batch_exact: unsupported type {ttype}"
        ));
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_exact row window out of bounds".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if m == 1 {
        // Mirrors the sequential per-token dispatch exactly for one token.
        return if row_start == 0 && n_rows == qw.rows {
            matmul_quantized(out, inp, qw, mapped)
        } else {
            matmul_quantized_rows(out, inp, qw, row_start, n_rows, mapped)
        };
    }
    if out.len() < m * d || inp.len() < m * n {
        return Err("matmul_quantized_batch_exact shape mismatch".to_string());
    }
    let row_size = get_row_size(n, qw.ttype);
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

    // Same MR4 eligibility as `matmul_quantized` (incl. the shared validation
    // status), so row -> kernel assignment matches the sequential path.
    #[cfg(target_arch = "x86_64")]
    let use_mr4 = if matches!(
        ttype,
        GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 | GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 | GGML_TYPE_IQ4_NL
    ) {
        use_x86_qk_mr4()
            && use_x86_avx2_fma()
            && n >= QK4_0
            && n.is_multiple_of(QK4_0)
            && d >= 4
            && validate_legacy_mr4_once_x86(&inp[..n], mapped, data_offset, row_size, n, ttype)
    } else {
        matches!(
            ttype,
            GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
        ) && use_x86_qk_mr4()
            && n >= QK_K
            && n.is_multiple_of(QK_K)
            && d >= 4
            && validate_qk_mr4_once_x86(&inp[..n], mapped, data_offset, row_size, n, ttype)
    };
    #[cfg(target_arch = "aarch64")]
    let use_mr4 = if matches!(
        ttype,
        GGML_TYPE_Q4_0 | GGML_TYPE_Q4_1 | GGML_TYPE_Q5_0 | GGML_TYPE_Q5_1 | GGML_TYPE_IQ4_NL
    ) {
        use_aarch64_qk_mr4()
            && n >= QK4_0
            && n.is_multiple_of(QK4_0)
            && d >= 4
            && validate_legacy_q4_q5_iq4_mr4_once_aarch64(
                &inp[..n],
                mapped,
                data_offset,
                row_size,
                n,
                ttype,
            )
    } else {
        matches!(
            ttype,
            GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
        ) && use_aarch64_qk_mr4()
            && n >= QK_K
            && n.is_multiple_of(QK_K)
            && d >= 4
            && validate_qk_mr4_once(&inp[..n], mapped, data_offset, row_size, n, ttype)
    };
    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    let use_mr4 = false;

    if !use_mr4 && tiled_k_exact_supported(qw.ttype) {
        return matmul_quantized_batch_tiled_k_exact_with_scratch(
            out, inp, qw, mapped, m, row_start, n_rows, tmp,
        );
    }

    if tmp.len() < d * m {
        tmp.resize(d * m, 0.0);
    }

    // Row-major scratch (`tmp[r * m + b]`) so rayon splits it into contiguous
    // per-row-chunk blocks; transposed to token-major `out` at the end.
    let fill = |base_row: usize, block: &mut [f32]| {
        let rows_here = block.len() / m;
        let mut i = 0usize;
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if use_mr4 {
            while i + 4 <= rows_here {
                matmul_prefetch_row(mapped, data_offset, row_size, base_row + i, d);
                let r0_off = data_offset + (base_row + i) * row_size;
                let r1_off = r0_off + row_size;
                let r2_off = r1_off + row_size;
                let r3_off = r2_off + row_size;
                let r0 = &mapped[r0_off..r0_off + row_size];
                let r1 = &mapped[r1_off..r1_off + row_size];
                let r2 = &mapped[r2_off..r2_off + row_size];
                let r3 = &mapped[r3_off..r3_off + row_size];
                for b in 0..m {
                    let x = &inp[b * n..(b + 1) * n];
                    let sums = batch_exact_vec_dot_4rows(ttype, x, r0, r1, r2, r3, n);
                    block[i * m + b] = sums[0];
                    block[(i + 1) * m + b] = sums[1];
                    block[(i + 2) * m + b] = sums[2];
                    block[(i + 3) * m + b] = sums[3];
                }
                i += 4;
            }
        }
        #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
        let _ = use_mr4;
        while i < rows_here {
            matmul_prefetch_row(mapped, data_offset, row_size, base_row + i, d);
            let row_off = data_offset + (base_row + i) * row_size;
            let row = &mapped[row_off..row_off + row_size];
            for b in 0..m {
                block[i * m + b] = batch_exact_vec_dot(ttype, &inp[b * n..(b + 1) * n], row, n);
            }
            i += 1;
        }
    };

    if d >= par_matmul_min_rows() {
        let chunk_rows = par_matmul_chunk_rows();
        tmp[..d * m]
            .par_chunks_mut(chunk_rows * m)
            .enumerate()
            .for_each(|(chunk_idx, block)| fill(chunk_idx * chunk_rows, block));
    } else {
        fill(0, &mut tmp[..d * m]);
    }

    for b in 0..m {
        for r in 0..d {
            out[b * d + r] = tmp[r * m + b];
        }
    }
    Ok(())
}

/// Fast batched quantized matmul: dequantizes each weight row **once** and
/// dots it against all `m` tokens with the SIMD f32 kernel — amortizing the
/// (dominant, for K-quants) bit-unpack cost across the batch.
///
/// Numerics are tolerance-level equivalent to [`matmul_quantized`], not
/// bit-identical: the f32 dot uses a different accumulation order than the
/// per-quant dots (the same class of difference the tolerance-validated MR4
/// VNNI kernels already exhibit on x86). Use
/// [`matmul_quantized_batch_exact`] when a bitwise sequential mirror is
/// needed (debugging / structural validation).
///
/// `out` is token-major `[m × qw.rows]`, `inp` token-major `[m × qw.cols]`;
/// `tmp` is caller-owned row-major scratch (`qw.rows × m` floats).
/// Whether [`matmul_quantized_batch_fast`] handles `ttype`. Restricted to the
/// K-quants, whose sequential dots compute the same per-element products
/// (`x * scale * q`) — the fast kernel then only reorders the summation
/// (~1e-6 relative). Types with specialized sequential kernels (Q8_0
/// int8-prequant, F16 conversion kernels) diverge far more and are excluded;
/// they take the bit-exact or per-token path instead.
pub(crate) fn batch_fast_supported(ttype: GgmlType) -> bool {
    matches!(
        ttype.0,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K
    )
}

/// Whether a floating-point tensor can use the dequantize-once batch kernel.
///
/// This is kept separate from [`batch_fast_supported`] because language-model
/// prefill intentionally preserves the existing per-token F16/BF16 numerics,
/// while multimodal encoders can opt into tolerance-equivalent batched dots.
pub(crate) fn float_batch_supported(ttype: GgmlType) -> bool {
    matches!(ttype.0, GGML_TYPE_F16 | GGML_TYPE_BF16)
}

/// Caller-owned storage for floating-point token batches. BF16 activation
/// packing is retained across calls so multimodal encoders do not allocate a
/// temporary buffer for every projection.
#[derive(Default)]
pub(crate) struct FloatBatchMatmulScratch {
    row_major: Vec<f32>,
    #[cfg(target_os = "macos")]
    weights_f32: Vec<f32>,
    bf16_input: Vec<u16>,
    bf16_input_f32: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_quantized_batch_fast(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    tmp: &mut Vec<f32>,
) -> Result<(), String> {
    if !batch_fast_supported(qw.ttype) {
        return Err(format!(
            "matmul_quantized_batch_fast: unsupported type {}",
            qw.ttype.0
        ));
    }
    matmul_quantized_batch_dequantized(out, inp, qw, mapped, m, row_start, n_rows, tmp)
}

/// F16/BF16 batch matmul for multimodal encoders.
///
/// Each stored weight row is expanded once and reused across every token in
/// the batch. The caller-owned row-major scratch also lets large matrices run
/// in parallel before the result is transposed back to token-major storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_quantized_batch_float(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    scratch: &mut FloatBatchMatmulScratch,
) -> Result<(), String> {
    if !float_batch_supported(qw.ttype) {
        return Err(format!(
            "matmul_quantized_batch_float: unsupported type {}",
            qw.ttype.0
        ));
    }

    #[cfg(target_os = "macos")]
    if qw.ttype.0 == GGML_TYPE_F16 && m >= 8 {
        return matmul_quantized_batch_f16_accelerate(
            out,
            inp,
            qw,
            mapped,
            m,
            row_start,
            n_rows,
            &mut scratch.weights_f32,
        );
    }

    if qw.ttype.0 == GGML_TYPE_BF16 {
        scratch.bf16_input.resize(inp.len(), 0);
        scratch
            .bf16_input
            .par_iter_mut()
            .zip(inp.par_iter())
            .for_each(|(destination, source)| *destination = fp32_to_bf16(*source));

        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if use_aarch64_bf16() {
            return matmul_quantized_batch_bf16_bfdot(
                out,
                &scratch.bf16_input,
                qw,
                mapped,
                m,
                row_start,
                n_rows,
                &mut scratch.row_major,
            );
        }

        scratch.bf16_input_f32.resize(inp.len(), 0.0);
        scratch
            .bf16_input_f32
            .par_iter_mut()
            .zip(scratch.bf16_input.par_iter())
            .for_each(|(destination, source)| *destination = bf16_to_fp32(*source));
        return matmul_quantized_batch_dequantized(
            out,
            &scratch.bf16_input_f32,
            qw,
            mapped,
            m,
            row_start,
            n_rows,
            &mut scratch.row_major,
        );
    }

    matmul_quantized_batch_dequantized(
        out,
        inp,
        qw,
        mapped,
        m,
        row_start,
        n_rows,
        &mut scratch.row_major,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_f16_accelerate(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    weights_f32: &mut Vec<f32>,
) -> Result<(), String> {
    let d = n_rows;
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_F16 {
        return Err("Accelerate float batch requires F16 weights".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("Accelerate F16 row window out of bounds".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if out.len() < m * d || inp.len() < m * n {
        return Err("Accelerate F16 batch shape mismatch".to_string());
    }
    let m_i32 = c_int::try_from(m)
        .map_err(|_| "Accelerate F16 batch token count exceeds c_int".to_string())?;
    let d_i32 = c_int::try_from(d)
        .map_err(|_| "Accelerate F16 batch row count exceeds c_int".to_string())?;
    let n_i32 = c_int::try_from(n)
        .map_err(|_| "Accelerate F16 batch column count exceeds c_int".to_string())?;

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "Accelerate F16 row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "Accelerate F16 tensor offset overflow".to_string())?;
    let data_size = d
        .checked_mul(row_size)
        .ok_or_else(|| "Accelerate F16 tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "Accelerate F16 tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("Accelerate F16 row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    weights_f32.resize(d * n, 0.0);
    weights_f32
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, destination)| {
            let source_offset = data_offset + row * row_size;
            dequantize_row_f16(
                &mapped[source_offset..source_offset + row_size],
                destination,
                n,
            );
        });

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;
    let prof_t0 = prof_start();
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m_i32,
            d_i32,
            n_i32,
            1.0,
            inp.as_ptr(),
            n_i32,
            weights_f32.as_ptr(),
            n_i32,
            0.0,
            out.as_mut_ptr(),
            d_i32,
        );
    }
    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_bf16_bfdot(
    out: &mut [f32],
    inp: &[u16],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    tmp: &mut Vec<f32>,
) -> Result<(), String> {
    let d = n_rows;
    let n = qw.cols;
    if qw.ttype.0 != GGML_TYPE_BF16 {
        return Err("matmul_quantized_batch_bf16_bfdot requires BF16 weights".to_string());
    }
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_bf16_bfdot row window out of bounds".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if out.len() < m * d || inp.len() < m * n {
        return Err("matmul_quantized_batch_bf16_bfdot shape mismatch".to_string());
    }

    let row_size = get_row_size(n, qw.ttype);
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "BF16 row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "BF16 tensor offset overflow".to_string())?;
    let data_size = d
        .checked_mul(row_size)
        .ok_or_else(|| "BF16 tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "BF16 tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("BF16 row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;

    if tmp.len() < d * m {
        tmp.resize(d * m, 0.0);
    }
    let prof_t0 = prof_start();
    let fill = |base_row: usize, block: &mut [f32]| {
        let rows_here = block.len() / m;
        for i in 0..rows_here {
            matmul_prefetch_row(mapped, data_offset, row_size, base_row + i, d);
            let row_off = data_offset + (base_row + i) * row_size;
            let row = &mapped[row_off..row_off + row_size];
            for token in 0..m {
                let activation = &inp[token * n..(token + 1) * n];
                // Runtime dispatch checked FEAT_BF16 before entering this kernel.
                block[i * m + token] = unsafe { vec_dot_bf16_bfdot(activation, row, n) };
            }
        }
    };

    if d >= par_matmul_min_rows() {
        let chunk_rows = par_matmul_chunk_rows();
        tmp[..d * m]
            .par_chunks_mut(chunk_rows * m)
            .enumerate()
            .for_each(|(chunk_idx, block)| fill(chunk_idx * chunk_rows, block));
    } else {
        fill(0, &mut tmp[..d * m]);
    }

    for token in 0..m {
        for row in 0..d {
            out[token * d + row] = tmp[row * m + token];
        }
    }
    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_quantized_batch_dequantized(
    out: &mut [f32],
    inp: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
    m: usize,
    row_start: usize,
    n_rows: usize,
    tmp: &mut Vec<f32>,
) -> Result<(), String> {
    let d = n_rows;
    let n = qw.cols;
    if row_start + n_rows > qw.rows {
        return Err("matmul_quantized_batch_dequantized row window out of bounds".to_string());
    }
    if m == 0 {
        return Ok(());
    }
    if m == 1 {
        return if row_start == 0 && n_rows == qw.rows {
            matmul_quantized(out, inp, qw, mapped)
        } else {
            matmul_quantized_rows(out, inp, qw, row_start, n_rows, mapped)
        };
    }
    if out.len() < m * d || inp.len() < m * n {
        return Err("matmul_quantized_batch_dequantized shape mismatch".to_string());
    }
    let row_size = get_row_size(n, qw.ttype);
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

    if tmp.len() < d * m {
        tmp.resize(d * m, 0.0);
    }

    let prof_t0 = prof_start();
    let fill = |dequant: &mut Vec<f32>, base_row: usize, block: &mut [f32]| -> Result<(), String> {
        if dequant.len() < n {
            dequant.resize(n, 0.0);
        }
        let rows_here = block.len() / m;
        for i in 0..rows_here {
            matmul_prefetch_row(mapped, data_offset, row_size, base_row + i, d);
            let row_off = data_offset + (base_row + i) * row_size;
            let row = &mapped[row_off..row_off + row_size];
            dequantize_row_into(qw.ttype, row, &mut dequant[..n], n)?;
            for b in 0..m {
                block[i * m + b] = dot_f32_simd(&inp[b * n..(b + 1) * n], &dequant[..n]);
            }
        }
        Ok(())
    };

    if d >= par_matmul_min_rows() {
        let chunk_rows = par_matmul_chunk_rows();
        let results: Vec<Result<(), String>> = tmp[..d * m]
            .par_chunks_mut(chunk_rows * m)
            .enumerate()
            .map_init(Vec::new, |dequant, (chunk_idx, block)| {
                fill(dequant, chunk_idx * chunk_rows, block)
            })
            .collect();
        for r in results {
            r?;
        }
    } else {
        let mut dequant = Vec::new();
        fill(&mut dequant, 0, &mut tmp[..d * m])?;
    }

    for b in 0..m {
        for r in 0..d {
            out[b * d + r] = tmp[r * m + b];
        }
    }
    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_dot_matches_scalar_for_vector_and_tail_lengths() {
        for n in [1usize, 3, 4, 7, 8, 15, 33, 127] {
            let x = (0..n)
                .map(|i| ((i * 29 % 41) as f32 - 20.0) * 0.125)
                .collect::<Vec<_>>();
            let bf16_bits = (0..n)
                .map(|i| {
                    let value = ((i * 17 % 37) as f32 - 18.0) * 0.25;
                    (value.to_bits() >> 16) as u16
                })
                .collect::<Vec<_>>();
            let weights = bf16_bits
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect::<Vec<_>>();
            let want = x
                .iter()
                .zip(bf16_bits.iter())
                .map(|(activation, bits)| activation * bf16_to_fp32(*bits))
                .sum::<f32>();
            let got = vec_dot_bf16(&x, &weights, n);
            let tolerance = 1e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "n={n}: SIMD={got}, scalar={want}, tolerance={tolerance}"
            );
        }
    }

    // Activations that the kernels' per-32-block Q8 quantization represents
    // exactly: integers in [-127, 127] with abs-max 127 in every block, so
    // scale = 1.0 and round(x * 127 / abs_max) == x. Under these inputs the
    // int8-aware references must agree with the float scalar references to
    // float-rounding precision, proving they implement the K-quant semantics
    // rather than merely mirroring the SIMD kernels.
    #[cfg(target_arch = "x86_64")]
    fn i8_ref_test_activations(n: usize) -> Vec<f32> {
        let mut x: Vec<f32> = (0..n)
            .map(|i| (((i as i32 * 37 + 11) % 255) - 127) as f32)
            .collect();
        for b in 0..n / 32 {
            x[b * 32] = 127.0;
        }
        x
    }

    #[cfg(target_arch = "x86_64")]
    fn i8_ref_test_row(len: usize, seed: usize) -> Vec<u8> {
        (0..len)
            .map(|k| ((k * 7 + 13 + seed * 31) % 251) as u8)
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4_k_i8_ref_matches_float_ref_on_exact_activations() {
        let n = 2 * QK_K;
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
        let rows: Vec<Vec<u8>> = (0..4)
            .map(|r| {
                let mut row = i8_ref_test_row(2 * block_sz, r);
                for blk in 0..2 {
                    let off = blk * block_sz;
                    // d = 0.25, dmin = 0.5: powers of two keep the float
                    // reference exact so the comparison isolates semantics.
                    row[off..off + 2].copy_from_slice(&0x3400u16.to_le_bytes());
                    row[off + 2..off + 4].copy_from_slice(&0x3800u16.to_le_bytes());
                }
                row
            })
            .collect();
        let x = i8_ref_test_activations(n);
        let got = vec_dot_q4_k_4rows_i8_ref(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
        for r in 0..4 {
            let want = vec_dot_q4_k(&x, &rows[r], n);
            let tol = 1e-4f32 * want.abs().max(1.0);
            assert!(
                (got[r] - want).abs() <= tol,
                "q4_k row {r}: i8_ref={} float_ref={want}",
                got[r]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q5_k_i8_ref_matches_float_ref_on_exact_activations() {
        let n = 2 * QK_K;
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
        let rows: Vec<Vec<u8>> = (0..4)
            .map(|r| {
                let mut row = i8_ref_test_row(2 * block_sz, r);
                for blk in 0..2 {
                    let off = blk * block_sz;
                    row[off..off + 2].copy_from_slice(&0x3400u16.to_le_bytes());
                    row[off + 2..off + 4].copy_from_slice(&0x3800u16.to_le_bytes());
                }
                row
            })
            .collect();
        let x = i8_ref_test_activations(n);
        let got = vec_dot_q5_k_4rows_i8_ref(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
        for r in 0..4 {
            let want = vec_dot_q5_k(&x, &rows[r], n);
            let tol = 1e-4f32 * want.abs().max(1.0);
            assert!(
                (got[r] - want).abs() <= tol,
                "q5_k row {r}: i8_ref={} float_ref={want}",
                got[r]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q6_k_i8_ref_matches_float_ref_on_exact_activations() {
        let n = 2 * QK_K;
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        let rows: Vec<Vec<u8>> = (0..4)
            .map(|r| {
                let mut row = i8_ref_test_row(2 * block_sz, r);
                for blk in 0..2 {
                    let off = blk * block_sz + d_off;
                    // d = 0.0625: signed i8 scales reach ±128, so a smaller
                    // power of two keeps the accumulation exactly representable.
                    row[off..off + 2].copy_from_slice(&0x2C00u16.to_le_bytes());
                }
                row
            })
            .collect();
        let x = i8_ref_test_activations(n);
        let got = vec_dot_q6_k_4rows_i8_ref(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
        for r in 0..4 {
            let want = vec_dot_q6_k(&x, &rows[r], n);
            let tol = 1e-4f32 * want.abs().max(1.0);
            assert!(
                (got[r] - want).abs() <= tol,
                "q6_k row {r}: i8_ref={} float_ref={want}",
                got[r]
            );
        }
    }

    #[test]
    fn matmul_quantized_batch_matches_repeated_rows_for_f32_weights() {
        let weights: [f32; 12] = [
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0, 0.25, -0.75, 1.5, 2.5,
        ];
        let mapped: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qw = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_F32),
            rows: 3,
            cols: 4,
        };

        let inp: [f32; 8] = [1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 3.0];

        let mut batch_out = vec![0.0f32; 2 * qw.rows];
        matmul_quantized_batch(&mut batch_out, &inp, &qw, &mapped, 2, 0, qw.rows).unwrap();

        let mut repeated_out = vec![0.0f32; 2 * qw.rows];
        for batch_idx in 0..2 {
            let x = &inp[batch_idx * qw.cols..(batch_idx + 1) * qw.cols];
            let out_row = &mut repeated_out[batch_idx * qw.rows..(batch_idx + 1) * qw.rows];
            matmul_quantized_rows(out_row, x, &qw, 0, qw.rows, &mapped).unwrap();
        }

        for (got, want) in batch_out.iter().zip(repeated_out.iter()) {
            assert!((got - want).abs() < 1e-6, "got={got} want={want}");
        }
    }

    #[test]
    fn bf16_batch_matmul_matches_repeated_rows() {
        let rows = par_matmul_min_rows() + 1;
        let cols = 17usize;
        let batch = 5usize;
        let mapped = (0..rows * cols)
            .flat_map(|i| {
                let value = ((i * 19 % 53) as f32 - 26.0) * 0.125;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect::<Vec<_>>();
        let qw = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_BF16),
            rows,
            cols,
        };
        let input = (0..batch * cols)
            .map(|i| ((i * 23 % 47) as f32 - 23.0) * 0.0625)
            .collect::<Vec<_>>();

        let mut batch_out = vec![0.0f32; batch * rows];
        let mut scratch = FloatBatchMatmulScratch::default();
        matmul_quantized_batch_float(
            &mut batch_out,
            &input,
            &qw,
            &mapped,
            batch,
            0,
            rows,
            &mut scratch,
        )
        .unwrap();

        let mut repeated_out = vec![0.0f32; batch * rows];
        for batch_idx in 0..batch {
            matmul_quantized_rows(
                &mut repeated_out[batch_idx * rows..(batch_idx + 1) * rows],
                &input[batch_idx * cols..(batch_idx + 1) * cols],
                &qw,
                0,
                rows,
                &mapped,
            )
            .unwrap();
        }

        for (index, (got, want)) in batch_out.iter().zip(repeated_out.iter()).enumerate() {
            let tolerance = 1e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "index={index}: batch={got}, repeated={want}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn f16_batch_matmul_matches_repeated_rows() {
        let rows = 11usize;
        let cols = 37usize;
        let batch = 8usize;
        let mapped = (0..rows * cols)
            .flat_map(|i| {
                let magnitude = 0x3000u16 + (i as u16 * 29 % 0x0800);
                let bits = if i % 3 == 0 {
                    magnitude | 0x8000
                } else {
                    magnitude
                };
                bits.to_le_bytes()
            })
            .collect::<Vec<_>>();
        let qw = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_F16),
            rows,
            cols,
        };
        let input = (0..batch * cols)
            .map(|i| ((i * 23 % 47) as f32 - 23.0) * 0.0625)
            .collect::<Vec<_>>();

        let mut batch_out = vec![0.0f32; batch * rows];
        matmul_quantized_batch_float(
            &mut batch_out,
            &input,
            &qw,
            &mapped,
            batch,
            0,
            rows,
            &mut FloatBatchMatmulScratch::default(),
        )
        .unwrap();

        let mut repeated_out = vec![0.0f32; batch * rows];
        for batch_idx in 0..batch {
            matmul_quantized_rows(
                &mut repeated_out[batch_idx * rows..(batch_idx + 1) * rows],
                &input[batch_idx * cols..(batch_idx + 1) * cols],
                &qw,
                0,
                rows,
                &mapped,
            )
            .unwrap();
        }

        for (index, (got, want)) in batch_out.iter().zip(repeated_out.iter()).enumerate() {
            let tolerance = 1e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "index={index}: batch={got}, repeated={want}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn bf16_batch_matmul_packs_activations_with_ties_to_even() {
        let rows = 3usize;
        let cols = 19usize;
        let batch = 2usize;
        let weight_bits = (0..rows * cols)
            .map(|i| fp32_to_bf16(((i * 13 % 31) as f32 - 15.0) * 0.2))
            .collect::<Vec<_>>();
        let mapped = weight_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let qw = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_BF16),
            rows,
            cols,
        };
        let input = (0..batch * cols)
            .map(|i| f32::from_bits(0x3f80_8001 + (i as u32 % 11) * 0x1_0101))
            .collect::<Vec<_>>();

        let mut got = vec![0.0f32; batch * rows];
        matmul_quantized_batch_float(
            &mut got,
            &input,
            &qw,
            &mapped,
            batch,
            0,
            rows,
            &mut FloatBatchMatmulScratch::default(),
        )
        .unwrap();

        for token in 0..batch {
            for row in 0..rows {
                let want = (0..cols)
                    .map(|col| {
                        bf16_to_fp32(fp32_to_bf16(input[token * cols + col]))
                            * bf16_to_fp32(weight_bits[row * cols + col])
                    })
                    .sum::<f32>();
                let value = got[token * rows + row];
                let tolerance = 1e-5 * want.abs().max(1.0);
                assert!(
                    (value - want).abs() <= tolerance,
                    "token={token} row={row}: native={value}, scalar={want}, tolerance={tolerance}"
                );
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn prepared_q8_activation_can_differ_from_exact_q8_dot() {
        let n = QK8_0;
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
        let mut row = vec![0u8; block_sz];
        row[0] = 0x00;
        row[1] = 0x3c; // fp16 1.0 scale
        for q in &mut row[2..2 + QK8_0] {
            *q = 127u8;
        }

        let mut x = vec![0.01f32; n];
        x[0] = 100.0;

        let exact = x
            .iter()
            .zip(&row[2..2 + QK8_0])
            .map(|(&xv, &q)| xv * q as i8 as f32)
            .sum::<f32>();
        let mut scratch = MatmulActivationScratch::new();
        let preq = prequantize_activation_q8(&x, n, &mut scratch);
        let mut dot = 0i32;
        for j in 0..QK8_0 {
            dot += preq.xq_i8[j] as i32 * row[2 + j] as i8 as i32;
        }
        let fast_semantics = dot as f32 * preq.scales[0];

        assert!(
            (exact - fast_semantics).abs() > 1.0,
            "fixture should expose activation-quantization drift: exact={exact}, fast={fast_semantics}"
        );
        assert!(
            (exact - fast_semantics).abs() < exact.abs() * 0.01,
            "drift should remain bounded for this fixture: exact={exact}, fast={fast_semantics}"
        );
    }
}

#[cfg(test)]
mod qk_mr4_tests {
    use super::*;
    use crate::engine::types::GgmlType;

    fn xorshift(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    #[test]
    fn q2k_4rows_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
        let mut st = 0xD1B5_4A32_9977_4401u64;

        for nb in [1usize, 2, 5] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 16 + QK_K / 4;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let dmin16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                    row[d_at + 2] = (dmin16 & 0xff) as u8;
                    row[d_at + 3] = (dmin16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q2_k_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q2_k(&x, &rows[0], n),
                vec_dot_q2_k(&x, &rows[1], n),
                vec_dot_q2_k(&x, &rows[2], n),
                vec_dot_q2_k(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q2k_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
        let mut st = 0xBADC_0FFE_E15E_A5E5u64;

        for nb in [1usize, 3, 7] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 16 + QK_K / 4;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let dmin16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                    row[d_at + 2] = (dmin16 & 0xff) as u8;
                    row[d_at + 3] = (dmin16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q2_k_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q2_k(&x, &rows[0], n),
                vec_dot_q2_k(&x, &rows[1], n),
                vec_dot_q2_k(&x, &rows[2], n),
                vec_dot_q2_k(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q2k_4rows_x86_vnni_prequant_matches_single_row_when_available() {
        if !use_x86_avx_vnni() && !use_x86_avx512_vnni_q8() {
            return;
        }

        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
        let mut st = 0xC001_D00D_2220_0001u64;

        for nb in [1usize, 3, 7] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 16 + QK_K / 4;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let dmin16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                    row[d_at + 2] = (dmin16 & 0xff) as u8;
                    row[d_at + 3] = (dmin16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();
            let mut scratch = MatmulActivationScratch::new();
            let preq = prequantize_activation_q8(&x, n, &mut scratch);

            let got = unsafe {
                if use_x86_avx512_vnni_q8() {
                    vec_dot_q2_k_4rows_x86_avx512vnni_prequant(
                        &preq, &rows[0], &rows[1], &rows[2], &rows[3], n,
                    )
                } else {
                    vec_dot_q2_k_4rows_x86_avxvnni_prequant(
                        &preq, &rows[0], &rows[1], &rows[2], &rows[3], n,
                    )
                }
            };
            for r in 0..4 {
                let want = vec_dot_q2_k(&x, &rows[r], n);
                let tol = 5e-4f32 * want.abs().max(1.0);
                assert!(
                    (got[r] - want).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {want}, tol={tol}",
                    got[r],
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q2k_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
        let mut st = 0xA64A_C0DE_2220_0001u64;

        for nb in [1usize, 3, 7] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 16 + QK_K / 4;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let dmin16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                    row[d_at + 2] = (dmin16 & 0xff) as u8;
                    row[d_at + 3] = (dmin16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q2_k_4rows_aarch64(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q2_k(&x, &rows[0], n),
                vec_dot_q2_k(&x, &rows[1], n),
                vec_dot_q2_k(&x, &rows[2], n),
                vec_dot_q2_k(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4_0_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
        let mut st = 0xC001_C0DE_4440_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK4_0;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[i * block_sz] = (d16 & 0xff) as u8;
                    row[i * block_sz + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q4_0_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q4_0(&x, &rows[0], n),
                vec_dot_q4_0(&x, &rows[1], n),
                vec_dot_q4_0(&x, &rows[2], n),
                vec_dot_q4_0(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4_1_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
        let mut st = 0xC001_C0DE_4441_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK4_1;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let m16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                    row[off + 2] = (m16 & 0xff) as u8;
                    row[off + 3] = (m16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q4_1_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q4_1(&x, &rows[0], n),
                vec_dot_q4_1(&x, &rows[1], n),
                vec_dot_q4_1(&x, &rows[2], n),
                vec_dot_q4_1(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4_0_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
        let mut st = 0xA64A_C0DE_4440_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK4_0;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[i * block_sz] = (d16 & 0xff) as u8;
                    row[i * block_sz + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q4_0_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q4_0(&x, &rows[0], n),
                vec_dot_q4_0(&x, &rows[1], n),
                vec_dot_q4_0(&x, &rows[2], n),
                vec_dot_q4_0(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4_1_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
        let mut st = 0xA64A_C0DE_4441_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK4_1;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let m16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                    row[off + 2] = (m16 & 0xff) as u8;
                    row[off + 3] = (m16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q4_1_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q4_1(&x, &rows[0], n),
                vec_dot_q4_1(&x, &rows[1], n),
                vec_dot_q4_1(&x, &rows[2], n),
                vec_dot_q4_1(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q5_0_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
        let mut st = 0xA64A_C0DE_5550_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK5_0;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q5_0_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q5_0(&x, &rows[0], n),
                vec_dot_q5_0(&x, &rows[1], n),
                vec_dot_q5_0(&x, &rows[2], n),
                vec_dot_q5_0(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q5_1_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
        let mut st = 0xA64A_C0DE_5551_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK5_1;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let m16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                    row[off + 2] = (m16 & 0xff) as u8;
                    row[off + 3] = (m16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q5_1_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q5_1(&x, &rows[0], n),
                vec_dot_q5_1(&x, &rows[1], n),
                vec_dot_q5_1(&x, &rows[2], n),
                vec_dot_q5_1(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q5_0_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
        let mut st = 0xC001_C0DE_5550_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK5_0;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q5_0_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q5_0(&x, &rows[0], n),
                vec_dot_q5_0(&x, &rows[1], n),
                vec_dot_q5_0(&x, &rows[2], n),
                vec_dot_q5_0(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q5_1_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
        let mut st = 0xC001_C0DE_5551_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK5_1;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    let m16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                    row[off + 2] = (m16 & 0xff) as u8;
                    row[off + 3] = (m16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q5_1_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q5_1(&x, &rows[0], n),
                vec_dot_q5_1(&x, &rows[1], n),
                vec_dot_q5_1(&x, &rows[2], n),
                vec_dot_q5_1(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn iq4_nl_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
        let mut st = 0xC001_C0DE_1040_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK4_NL;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_iq4_nl_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_iq4_nl(&x, &rows[0], n),
                vec_dot_iq4_nl(&x, &rows[1], n),
                vec_dot_iq4_nl(&x, &rows[2], n),
                vec_dot_iq4_nl(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn iq4_nl_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
        let mut st = 0xA64A_C0DE_1040_0001u64;

        for nb in [1usize, 5, 13] {
            let n = nb * QK4_NL;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let off = i * block_sz;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[off] = (d16 & 0xff) as u8;
                    row[off + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_iq4_nl_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_iq4_nl(&x, &rows[0], n),
                vec_dot_iq4_nl(&x, &rows[1], n),
                vec_dot_iq4_nl(&x, &rows[2], n),
                vec_dot_iq4_nl(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    /// The 4-row Q3_K kernel must equal the reference single-row `vec_dot_q3_k`
    /// on every row — this is exactly what the runtime self-check enforces, and
    /// it de-risks the MR4 wiring without needing the model.
    /// On x86 the MR4 dispatch uses `vec_dot_q3_k_4rows_x86` (AVX2 kernel
    /// when the CPU has it). Same oracle as the portable test: must match the
    /// scalar reference within the MR4 gate tolerance. On non-AVX2 hosts the
    /// wrapper falls back to the portable kernel and this reduces to that case.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q3k_4rows_x86_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
        let mut st = 0xA5A5_5A5A_1234_8765u64;
        for nb in [1usize, 3] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 8 + QK_K / 4 + 12;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();
            let got = vec_dot_q3_k_4rows_x86(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            for r in 0..4 {
                let want = vec_dot_q3_k(&x, &rows[r], n);
                let tol = 1e-4f32 * want.abs().max(1.0);
                assert!(
                    (got[r] - want).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {want}",
                    got[r],
                );
            }
        }
    }

    /// On ARM the MR4 dispatch uses the explicit NEON kernel, validated with the
    /// same scalar-reference oracle as the portable and x86 variants.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q3k_4rows_aarch64_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
        let mut st = 0x6C8E_9CF5_5709_32BDu64;
        for nb in [1usize, 3] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 8 + QK_K / 4 + 12;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();
            let got = vec_dot_q3_k_4rows_aarch64(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            for r in 0..4 {
                let want = vec_dot_q3_k(&x, &rows[r], n);
                let tol = 1e-4f32 * want.abs().max(1.0);
                assert!(
                    (got[r] - want).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {want}",
                    got[r],
                );
            }
        }
    }

    #[test]
    fn q3k_4rows_matches_single_row() {
        let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
        let mut st = 0x9E3779B97F4A7C15u64;

        for nb in [1usize, 2, 5] {
            let n = nb * QK_K;
            let mut rows: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; nb * block_sz]);
            for row in rows.iter_mut() {
                for b in row.iter_mut() {
                    *b = (xorshift(&mut st) & 0xff) as u8;
                }
                // Force each super-block scale `d` (fp16 at block end) finite and
                // positive so random bytes can't yield NaN/Inf and break the compare.
                for i in 0..nb {
                    let d_at = i * block_sz + QK_K / 8 + QK_K / 4 + 12;
                    let d16: u16 = 0x3000 | (xorshift(&mut st) as u16 & 0x03ff);
                    row[d_at] = (d16 & 0xff) as u8;
                    row[d_at + 1] = (d16 >> 8) as u8;
                }
            }
            let x: Vec<f32> = (0..n)
                .map(|_| (xorshift(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();

            let got = vec_dot_q3_k_4rows(&x, &rows[0], &rows[1], &rows[2], &rows[3], n);
            let want = [
                vec_dot_q3_k(&x, &rows[0], n),
                vec_dot_q3_k(&x, &rows[1], n),
                vec_dot_q3_k(&x, &rows[2], n),
                vec_dot_q3_k(&x, &rows[3], n),
            ];
            for r in 0..4 {
                let tol = 1e-4f32 * want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "nb={nb} row={r}: got {}, want {}",
                    got[r],
                    want[r]
                );
            }
        }
    }
}

#[cfg(test)]
mod batch_exact_tests {
    use super::*;
    use crate::engine::types::{GgmlType, QuantizedTensor};

    fn xs(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// fp16 scale byte offsets within one super-block, per type, so random
    /// test weights can't produce Inf/NaN scales (which would break bitwise
    /// comparison via NaN != NaN).
    fn d16_offsets(ttype: i32, block_sz: usize) -> Vec<usize> {
        match ttype {
            GGML_TYPE_Q2_K => vec![QK_K / 16 + QK_K / 4, QK_K / 16 + QK_K / 4 + 2],
            GGML_TYPE_Q3_K => vec![QK_K / 8 + QK_K / 4 + 12],
            GGML_TYPE_Q4_K | GGML_TYPE_Q5_K => vec![0, 2],
            GGML_TYPE_Q6_K => vec![block_sz - 2],
            _ => unreachable!(),
        }
    }

    fn synth(ttype: i32, d_rows: usize, n: usize, st: &mut u64) -> (QuantizedTensor, Vec<u8>) {
        let gt = GgmlType(ttype);
        let block_sz = get_type_size(gt);
        let row_size = get_row_size(n, gt);
        let nb = n / QK_K;
        let mut mapped = vec![0u8; d_rows * row_size];
        for b in mapped.iter_mut() {
            *b = (xs(st) & 0xff) as u8;
        }
        for r in 0..d_rows {
            for blk in 0..nb {
                for off in d16_offsets(ttype, block_sz) {
                    let at = r * row_size + blk * block_sz + off;
                    let d16: u16 = 0x3000 | (xs(st) as u16 & 0x03ff);
                    mapped[at] = (d16 & 0xff) as u8;
                    mapped[at + 1] = (d16 >> 8) as u8;
                }
            }
        }
        (
            QuantizedTensor {
                data_offset: 0,
                ttype: gt,
                rows: d_rows,
                cols: n,
            },
            mapped,
        )
    }

    /// The whole point of `matmul_quantized_batch_exact`: bitwise equality
    /// with per-token `matmul_quantized`, across the MR4 4-block path, the
    /// singles tail (rows % 4), the serial path (small d), and the rayon
    /// path (large d).
    #[test]
    fn batch_exact_is_bitwise_equal_to_per_token() {
        let mut st = 0xfeed_beef_dead_cafeu64;
        let n = 2 * QK_K;
        for &ttype in &[
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
        ] {
            for &d_rows in &[3usize, 6, 400] {
                let (qw, mapped) = synth(ttype, d_rows, n, &mut st);
                for &m in &[1usize, 2, 5] {
                    let inp: Vec<f32> = (0..m * n)
                        .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                        .collect();
                    let mut batch = vec![0.0f32; m * d_rows];
                    let mut tmp = Vec::new();
                    matmul_quantized_batch_exact(
                        &mut batch, &inp, &qw, &mapped, m, 0, d_rows, &mut tmp,
                    )
                    .unwrap();
                    for b in 0..m {
                        let mut single = vec![0.0f32; d_rows];
                        matmul_quantized(&mut single, &inp[b * n..(b + 1) * n], &qw, &mapped)
                            .unwrap();
                        for r in 0..d_rows {
                            assert_eq!(
                                batch[b * d_rows + r].to_bits(),
                                single[r].to_bits(),
                                "ttype={ttype} d={d_rows} m={m} tok={b} row={r}: {} vs {}",
                                batch[b * d_rows + r],
                                single[r],
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_q2k_exact_matches_scalar_vec_dot_bitwise() {
        let mut st = 0x711e_d222_cafe_f00du64;
        let n = 3 * QK_K;
        let d_rows = 7usize;
        let row_start = 1usize;
        let n_rows = 5usize;
        let m = 4usize;
        let (qw, mapped) = synth(GGML_TYPE_Q2_K, d_rows, n, &mut st);
        let row_size = get_row_size(n, qw.ttype);
        let inp: Vec<f32> = (0..m * n)
            .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
            .collect();
        let mut tiled = vec![0.0f32; m * n_rows];

        matmul_quantized_batch_tiled_q2_k_exact(
            &mut tiled, &inp, &qw, &mapped, m, row_start, n_rows,
        )
        .unwrap();

        for b in 0..m {
            let x = &inp[b * n..(b + 1) * n];
            for r in 0..n_rows {
                let global_row = row_start + r;
                let row_off = global_row * row_size;
                let row = &mapped[row_off..row_off + row_size];
                let want = vec_dot_q2_k(x, row, n);
                let got = tiled[b * n_rows + r];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "token={b} row={r}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn tiled_q3k_exact_matches_scalar_vec_dot_bitwise() {
        let mut st = 0x711e_d333_cafe_f00du64;
        let n = 3 * QK_K;
        let d_rows = 7usize;
        let row_start = 1usize;
        let n_rows = 5usize;
        let m = 4usize;
        let (qw, mapped) = synth(GGML_TYPE_Q3_K, d_rows, n, &mut st);
        let row_size = get_row_size(n, qw.ttype);
        let inp: Vec<f32> = (0..m * n)
            .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
            .collect();
        let mut tiled = vec![0.0f32; m * n_rows];

        matmul_quantized_batch_tiled_q3_k_exact(
            &mut tiled, &inp, &qw, &mapped, m, row_start, n_rows,
        )
        .unwrap();

        for b in 0..m {
            let x = &inp[b * n..(b + 1) * n];
            for r in 0..n_rows {
                let global_row = row_start + r;
                let row_off = global_row * row_size;
                let row = &mapped[row_off..row_off + row_size];
                let want = vec_dot_q3_k(x, row, n);
                let got = tiled[b * n_rows + r];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "token={b} row={r}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn tiled_q4k_exact_matches_scalar_vec_dot_bitwise() {
        let mut st = 0x711e_d444_cafe_f00du64;
        let n = 3 * QK_K;
        let d_rows = 7usize;
        let row_start = 1usize;
        let n_rows = 5usize;
        let m = 4usize;
        let (qw, mapped) = synth(GGML_TYPE_Q4_K, d_rows, n, &mut st);
        let row_size = get_row_size(n, qw.ttype);
        let inp: Vec<f32> = (0..m * n)
            .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
            .collect();
        let mut tiled = vec![0.0f32; m * n_rows];

        matmul_quantized_batch_tiled_q4_k_exact(
            &mut tiled, &inp, &qw, &mapped, m, row_start, n_rows,
        )
        .unwrap();

        for b in 0..m {
            let x = &inp[b * n..(b + 1) * n];
            for r in 0..n_rows {
                let global_row = row_start + r;
                let row_off = global_row * row_size;
                let row = &mapped[row_off..row_off + row_size];
                let want = vec_dot_q4_k(x, row, n);
                let got = tiled[b * n_rows + r];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "token={b} row={r}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn tiled_q5k_exact_matches_scalar_vec_dot_bitwise() {
        let mut st = 0x711e_d555_cafe_f00du64;
        let n = 3 * QK_K;
        let d_rows = 7usize;
        let row_start = 1usize;
        let n_rows = 5usize;
        let m = 4usize;
        let (qw, mapped) = synth(GGML_TYPE_Q5_K, d_rows, n, &mut st);
        let row_size = get_row_size(n, qw.ttype);
        let inp: Vec<f32> = (0..m * n)
            .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
            .collect();
        let mut tiled = vec![0.0f32; m * n_rows];

        matmul_quantized_batch_tiled_q5_k_exact(
            &mut tiled, &inp, &qw, &mapped, m, row_start, n_rows,
        )
        .unwrap();

        for b in 0..m {
            let x = &inp[b * n..(b + 1) * n];
            for r in 0..n_rows {
                let global_row = row_start + r;
                let row_off = global_row * row_size;
                let row = &mapped[row_off..row_off + row_size];
                let want = vec_dot_q5_k(x, row, n);
                let got = tiled[b * n_rows + r];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "token={b} row={r}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn tiled_q6k_exact_matches_scalar_vec_dot_bitwise() {
        let mut st = 0x711e_d666_cafe_f00du64;
        let n = 3 * QK_K;
        let d_rows = 7usize;
        let row_start = 1usize;
        let n_rows = 5usize;
        let m = 4usize;
        let (qw, mapped) = synth(GGML_TYPE_Q6_K, d_rows, n, &mut st);
        let row_size = get_row_size(n, qw.ttype);
        let inp: Vec<f32> = (0..m * n)
            .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
            .collect();
        let mut tiled = vec![0.0f32; m * n_rows];

        matmul_quantized_batch_tiled_q6_k_exact(
            &mut tiled, &inp, &qw, &mapped, m, row_start, n_rows,
        )
        .unwrap();

        for b in 0..m {
            let x = &inp[b * n..(b + 1) * n];
            for r in 0..n_rows {
                let global_row = row_start + r;
                let row_off = global_row * row_size;
                let row = &mapped[row_off..row_off + row_size];
                let want = vec_dot_q6_k(x, row, n);
                let got = tiled[b * n_rows + r];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "token={b} row={r}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn tiled_k_exact_parallel_chunks_match_scalar_vec_dot_bitwise() {
        let mut st = 0x711e_da7a_cafe_f00du64;
        let n = QK_K;
        let row_start = 1usize;
        let n_rows = par_matmul_min_rows() + par_matmul_chunk_rows() + 3;
        let d_rows = n_rows + 2;
        let m = 3usize;

        for &ttype in &[
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
        ] {
            let (qw, mapped) = synth(ttype, d_rows, n, &mut st);
            let row_size = get_row_size(n, qw.ttype);
            let inp: Vec<f32> = (0..m * n)
                .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();
            let mut tiled = vec![0.0f32; m * n_rows];
            let mut tmp = Vec::new();

            matmul_quantized_batch_tiled_k_exact_with_scratch(
                &mut tiled, &inp, &qw, &mapped, m, row_start, n_rows, &mut tmp,
            )
            .unwrap();

            assert!(
                tmp.len() >= m * n_rows,
                "parallel tiled path should use caller scratch for ttype={ttype}"
            );
            for b in 0..m {
                let x = &inp[b * n..(b + 1) * n];
                for r in 0..n_rows {
                    let global_row = row_start + r;
                    let row_off = global_row * row_size;
                    let row = &mapped[row_off..row_off + row_size];
                    let want = match ttype {
                        GGML_TYPE_Q2_K => vec_dot_q2_k(x, row, n),
                        GGML_TYPE_Q3_K => vec_dot_q3_k(x, row, n),
                        GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
                        GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
                        GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
                        _ => unreachable!(),
                    };
                    let got = tiled[b * n_rows + r];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "ttype={ttype} token={b} row={r}: got {got}, want {want}",
                    );
                }
            }
        }
    }

    #[test]
    fn batch_fast_row_window_matches_exact_with_tolerance() {
        let mut st = 0x1234_f00d_cafe_5678u64;
        let n = 2 * QK_K;
        let d_rows = 9usize;
        let row_start = 2usize;
        let n_rows = 5usize;
        let m = 3usize;

        for &ttype in &[
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
        ] {
            let (qw, mapped) = synth(ttype, d_rows, n, &mut st);
            let inp: Vec<f32> = (0..m * n)
                .map(|_| (xs(&mut st) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32)
                .collect();
            let mut exact = vec![0.0f32; m * n_rows];
            let mut fast = vec![0.0f32; m * n_rows];
            let mut exact_tmp = Vec::new();
            let mut fast_tmp = Vec::new();

            matmul_quantized_batch_exact(
                &mut exact,
                &inp,
                &qw,
                &mapped,
                m,
                row_start,
                n_rows,
                &mut exact_tmp,
            )
            .unwrap();
            matmul_quantized_batch_fast(
                &mut fast,
                &inp,
                &qw,
                &mapped,
                m,
                row_start,
                n_rows,
                &mut fast_tmp,
            )
            .unwrap();

            for i in 0..m * n_rows {
                let tol = 5e-4f32 * exact[i].abs().max(1.0);
                assert!(
                    (fast[i] - exact[i]).abs() <= tol,
                    "ttype={ttype} i={i}: fast={}, exact={}, tol={tol}",
                    fast[i],
                    exact[i]
                );
            }
        }
    }
}
