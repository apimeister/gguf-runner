#![allow(unsafe_op_in_unsafe_fn)]

use crate::engine::distributed::{MoeExpertExecutor, compute_local_selected_experts};
use crate::engine::kernels::{
    accum, dot_f32_simd, get_row_size, l2_norm, layernorm_inplace, matmul_f32_embeddings,
    matmul_quantized, matmul_quantized_rows, qwen3next_linear_attention_autoregressive, rmsnorm,
    rmsnorm_gemma, rmsnorm_inplace, rmsnorm_per_head_gemma_inplace, sanitize_finite_inplace,
    scale_slice_inplace, select_topk_softmax, sigmoid_mul_inplace, silu_and_mul_inplace, softmax,
};
use crate::engine::profiling::{PROF_ATTN_NS, PROF_FFN_NS, PROF_MOE_NS, prof_end, prof_start};
use crate::engine::switches::{
    KvCacheMode as SwitchKvCacheMode, kv_cache_mode, layer_debug_enabled, layer_debug_pos,
    par_attn_min_heads,
};
use crate::engine::types::{
    Config, GGML_TYPE_BF16, KvCacheFormat, QuantizedTensor, RunState, TransformerWeights,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

#[inline]
fn bf16_le_to_f32(lo: u8, hi: u8) -> f32 {
    let u = u16::from_le_bytes([lo, hi]) as u32;
    f32::from_bits(u << 16)
}

#[allow(clippy::too_many_arguments)]
fn validate_bf16_projection_rows(
    p: &Config,
    pos: usize,
    l: usize,
    tag: &str,
    qw: &QuantizedTensor,
    x: &[f32],
    got: &[f32],
    mapped: &[u8],
) {
    if !p.is_qwen35 || qw.ttype.0 != GGML_TYPE_BF16 || !env_flag("GGUF_QWEN35_VALIDATE_BF16_KV") {
        return;
    }
    let target_layer = env_usize("GGUF_QWEN35_VALIDATE_LAYER").unwrap_or(3);
    let target_pos = env_usize("GGUF_QWEN35_VALIDATE_POS").unwrap_or(23);
    if l != target_layer || pos != target_pos {
        return;
    }
    let row_size = get_row_size(qw.cols, qw.ttype);
    if row_size != qw.cols * 2 {
        eprintln!(
            "[BF16CHK] unexpected row_size for {tag}: row_size={} cols={} ttype={}",
            row_size, qw.cols, qw.ttype.0
        );
        return;
    }

    let probes = [0usize, 1, 2, 7, 15, 31, 63, 127, 255, 511];
    let mut max_abs = 0.0f32;
    let mut worst_row = 0usize;
    let mut samples = 0usize;
    for &r in &probes {
        if r >= qw.rows || r >= got.len() {
            continue;
        }
        let row_off = match qw.data_offset.checked_add(r.saturating_mul(row_size)) {
            Some(v) => v,
            None => continue,
        };
        let row_end = match row_off.checked_add(row_size) {
            Some(v) => v,
            None => continue,
        };
        if row_end > mapped.len() {
            continue;
        }
        let row = &mapped[row_off..row_end];
        let mut ref_dot = 0.0f32;
        for (i, &xv) in x.iter().enumerate().take(qw.cols) {
            let b = i * 2;
            ref_dot += xv * bf16_le_to_f32(row[b], row[b + 1]);
        }
        let diff = (got[r] - ref_dot).abs();
        if diff > max_abs {
            max_abs = diff;
            worst_row = r;
        }
        samples += 1;
    }

    if samples > 0 {
        eprintln!(
            "[BF16CHK] pos={} layer={} tensor={} samples={} max_abs_diff={:.6e} worst_row={}",
            pos, l, tag, samples, max_abs, worst_row
        );
    }
}

fn alloc_f32(len: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        format!(
            "unable to allocate {label} ({bytes} bytes). Try reducing --context-size and --max-tokens."
        )
    })?;
    out.resize(len, 0.0);
    Ok(out)
}

fn alloc_i8(len: usize, label: &str) -> Result<Vec<i8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        format!(
            "unable to allocate {label} ({} bytes). Try reducing --context-size and --max-tokens.",
            len
        )
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn alloc_u8(len: usize, label: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        format!(
            "unable to allocate {label} ({} bytes). Try reducing --context-size and --max-tokens.",
            len
        )
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn quantize_row_q8(src: &[f32], dst: &mut [i8], scale_out: &mut f32) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = src.len();
        // Vectorized abs-max scan
        let max_abs = unsafe {
            let mut vmax = vdupq_n_f32(0.0f32);
            let mut i = 0usize;
            while i + 16 <= n {
                let v0 = vabsq_f32(vld1q_f32(src.as_ptr().add(i)));
                let v1 = vabsq_f32(vld1q_f32(src.as_ptr().add(i + 4)));
                let v2 = vabsq_f32(vld1q_f32(src.as_ptr().add(i + 8)));
                let v3 = vabsq_f32(vld1q_f32(src.as_ptr().add(i + 12)));
                vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
                i += 16;
            }
            while i + 4 <= n {
                vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(src.as_ptr().add(i))));
                i += 4;
            }
            let mut m = vmaxvq_f32(vmax);
            while i < n {
                let a = src[i].abs();
                if a > m {
                    m = a;
                }
                i += 1;
            }
            m
        };
        if max_abs == 0.0 {
            *scale_out = 1.0;
            dst.fill(0);
            return;
        }
        *scale_out = max_abs / 127.0;
        let inv = 127.0 / max_abs;
        // Vectorized quantize
        unsafe {
            let vinv = vdupq_n_f32(inv);
            let vmin = vdupq_n_f32(-127.0);
            let vmax = vdupq_n_f32(127.0);
            let mut i = 0usize;
            let dst_ptr = dst.as_mut_ptr();
            while i + 16 <= n {
                let f0 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i)), vinv)),
                    ),
                );
                let f1 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i + 4)), vinv)),
                    ),
                );
                let f2 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i + 8)), vinv)),
                    ),
                );
                let f3 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i + 12)), vinv)),
                    ),
                );
                let i0 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(f0)), vmovn_s32(vcvtq_s32_f32(f1)));
                let i1 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(f2)), vmovn_s32(vcvtq_s32_f32(f3)));
                vst1q_s8(dst_ptr.add(i), vcombine_s8(vmovn_s16(i0), vmovn_s16(i1)));
                i += 16;
            }
            while i < n {
                dst[i] = (src[i] * inv).round().clamp(-127.0, 127.0) as i8;
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut max_abs = 0.0f32;
        for &x in src {
            max_abs = max_abs.max(x.abs());
        }
        if max_abs == 0.0 {
            *scale_out = 1.0;
            dst.fill(0);
            return;
        }
        let inv = 127.0 / max_abs;
        *scale_out = max_abs / 127.0;
        for (i, &x) in src.iter().enumerate() {
            dst[i] = (x * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q8_row_neon(q: &[f32], cache: &[i8], row_offset: usize, scale: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = q.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let q_ptr = q.as_ptr();
    let k_ptr = cache.as_ptr().add(row_offset);
    let mut i = 0usize;
    while i + 16 <= n {
        let kv = vld1q_s8(k_ptr.add(i));
        let k_lo = vmovl_s8(vget_low_s8(kv));
        let k_hi = vmovl_s8(vget_high_s8(kv));
        let k0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_lo)));
        let k1 = vcvtq_f32_s32(vmovl_high_s16(k_lo));
        let k2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_hi)));
        let k3 = vcvtq_f32_s32(vmovl_high_s16(k_hi));
        acc0 = vfmaq_f32(acc0, vld1q_f32(q_ptr.add(i)), k0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(q_ptr.add(i + 4)), k1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(q_ptr.add(i + 8)), k2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(q_ptr.add(i + 12)), k3);
        i += 16;
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        sum += q[i] * cache[row_offset + i] as f32;
        i += 1;
    }
    sum * scale
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn axpy_q8_row_neon(dst: &mut [f32], a: f32, cache: &[i8], row_offset: usize, scale: f32) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let scaled = vdupq_n_f32(a * scale);
    let dst_ptr = dst.as_mut_ptr();
    let k_ptr = cache.as_ptr().add(row_offset);
    let mut i = 0usize;
    while i + 16 <= n {
        let kv = vld1q_s8(k_ptr.add(i));
        let k_lo = vmovl_s8(vget_low_s8(kv));
        let k_hi = vmovl_s8(vget_high_s8(kv));
        let k0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_lo)));
        let k1 = vcvtq_f32_s32(vmovl_high_s16(k_lo));
        let k2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_hi)));
        let k3 = vcvtq_f32_s32(vmovl_high_s16(k_hi));
        vst1q_f32(
            dst_ptr.add(i),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i)), scaled, k0),
        );
        vst1q_f32(
            dst_ptr.add(i + 4),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i + 4)), scaled, k1),
        );
        vst1q_f32(
            dst_ptr.add(i + 8),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i + 8)), scaled, k2),
        );
        vst1q_f32(
            dst_ptr.add(i + 12),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i + 12)), scaled, k3),
        );
        i += 16;
    }
    let scalar = a * scale;
    while i < n {
        dst[i] += scalar * cache[row_offset + i] as f32;
        i += 1;
    }
}

/// Block size used by the optional per-row Q8 block-scale path. Smaller blocks preserve more
/// precision by limiting the impact of outlier activations.
const Q4_BLOCK_SIZE: usize = 32;
const TURBOQUANT_CENTROIDS: [f32; 4] = [-1.510_417, -0.452_78, 0.452_78, 1.510_417];
const TURBOQUANT_THRESHOLD_LO: f32 = -0.981_598_5;
const TURBOQUANT_THRESHOLD_HI: f32 = 0.981_598_5;
const TURBOQUANT_ROTATE_PRE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const TURBOQUANT_ROTATE_POST_SALT: u64 = 0xC2B2_AE3D_27D4_EB4F;
const TURBOQUANT_RESIDUAL_PRE_SALT: u64 = 0x1656_67B1_9E37_79F9;
const TURBOQUANT_RESIDUAL_POST_SALT: u64 = 0xD6E8_FEB8_6659_FD93;

const TURBO_SIGN_SLOT_ROT_PRE: usize = 0;
const TURBO_SIGN_SLOT_ROT_POST: usize = 1;
const TURBO_SIGN_SLOT_RES_PRE: usize = 2;
const TURBO_SIGN_SLOT_RES_POST: usize = 3;
const TURBO_SIGN_SLOTS: usize = 4;

fn turboquant_sign_bytes_per_head(head_size: usize) -> usize {
    head_size.div_ceil(8)
}

/// Precomputes all sign bit patterns for every (slot, layer, kv_head) combination.
/// Layout: [slot * n_layers * n_kv_heads + layer * n_kv_heads + kv_head] * bytes_per_head
/// bit=1 means multiply by -1 (matches turboquant_sign convention: splitmix64 & 1 == 1 → -1.0)
fn turboquant_build_sign_table(n_layers: usize, n_kv_heads: usize, head_size: usize) -> Vec<u8> {
    let bytes = turboquant_sign_bytes_per_head(head_size);
    let mut table = vec![0u8; TURBO_SIGN_SLOTS * n_layers * n_kv_heads * bytes];
    let salts = [
        (TURBO_SIGN_SLOT_ROT_PRE, TURBOQUANT_ROTATE_PRE_SALT),
        (TURBO_SIGN_SLOT_ROT_POST, TURBOQUANT_ROTATE_POST_SALT),
        (TURBO_SIGN_SLOT_RES_PRE, TURBOQUANT_RESIDUAL_PRE_SALT),
        (TURBO_SIGN_SLOT_RES_POST, TURBOQUANT_RESIDUAL_POST_SALT),
    ];
    let stride = n_layers * n_kv_heads * bytes;
    for (slot, salt) in salts {
        for layer in 0..n_layers {
            for kv_head in 0..n_kv_heads {
                let seed = splitmix64(((layer as u64) << 32) ^ kv_head as u64 ^ salt);
                let base = slot * stride + (layer * n_kv_heads + kv_head) * bytes;
                let bits = &mut table[base..base + bytes];
                for i in 0..head_size {
                    if (splitmix64(seed ^ i as u64) & 1) != 0 {
                        bits[i / 8] |= 1 << (i & 7);
                    }
                }
            }
        }
    }
    table
}

/// Four precomputed sign-bit slices for one (layer, kv_head) pair.
#[derive(Clone, Copy)]
struct TurboSignRef<'a> {
    rot_pre: &'a [u8],
    rot_post: &'a [u8],
    res_pre: &'a [u8],
    res_post: &'a [u8],
}

impl<'a> TurboSignRef<'a> {
    fn from_table(
        table: &'a [u8],
        layer: usize,
        kv_head: usize,
        n_layers: usize,
        n_kv_heads: usize,
        bytes_per_head: usize,
    ) -> Self {
        let stride = n_layers * n_kv_heads * bytes_per_head;
        let off = (layer * n_kv_heads + kv_head) * bytes_per_head;
        let slot = |s: usize| &table[s * stride + off..][..bytes_per_head];
        TurboSignRef {
            rot_pre: slot(TURBO_SIGN_SLOT_ROT_PRE),
            rot_post: slot(TURBO_SIGN_SLOT_ROT_POST),
            res_pre: slot(TURBO_SIGN_SLOT_RES_PRE),
            res_post: slot(TURBO_SIGN_SLOT_RES_POST),
        }
    }
}

fn turboquant_apply_signs_bits(values: &mut [f32], sign_bits: &[u8]) {
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let table = &TURBOQUANT_NEG_SIGN_DECODE_TABLE;
        let ptr = values.as_mut_ptr();
        // Process 8 elements at a time: one byte → 8 sign multipliers
        while i + 8 <= values.len() {
            let byte = sign_bits[i / 8];
            let signs = &table[byte as usize];
            unsafe {
                let v0 = vld1q_f32(ptr.add(i));
                let v1 = vld1q_f32(ptr.add(i + 4));
                let s0 = vld1q_f32(signs.as_ptr());
                let s1 = vld1q_f32(signs.as_ptr().add(4));
                vst1q_f32(ptr.add(i), vmulq_f32(v0, s0));
                vst1q_f32(ptr.add(i + 4), vmulq_f32(v1, s1));
            }
            i += 8;
        }
    }
    // Scalar tail
    while i < values.len() {
        if ((sign_bits[i / 8] >> (i & 7)) & 1) != 0 {
            values[i] = -values[i];
        }
        i += 1;
    }
}

/// Fused sign1 → FWHT → sign2 transform. On aarch64 with power-of-two n≥8,
/// sign1 is folded into the first butterfly stage and sign2 into the final
/// scaling pass, eliminating two full data passes.
#[cfg(target_arch = "aarch64")]
unsafe fn turboquant_transform_fused_neon(
    values: &mut [f32],
    first_bits: &[u8],
    second_bits: &[u8],
) {
    use std::arch::aarch64::*;
    let n = values.len();
    let ptr = values.as_mut_ptr();
    let neg_table = &TURBOQUANT_NEG_SIGN_DECODE_TABLE;

    // Stage width=1 fused with sign1: apply sign1 then butterfly on adjacent pairs
    {
        let mut i = 0usize;
        while i + 8 <= n {
            let byte = first_bits[i / 8];
            let sp = neg_table[byte as usize].as_ptr();
            // Load values and apply sign1 in one step
            let v0 = vmulq_f32(vld1q_f32(ptr.add(i)), vld1q_f32(sp));
            let v1 = vmulq_f32(vld1q_f32(ptr.add(i + 4)), vld1q_f32(sp.add(4)));
            // Butterfly width=1
            let even = vuzp1q_f32(v0, v1);
            let odd = vuzp2q_f32(v0, v1);
            let sum = vaddq_f32(even, odd);
            let diff = vsubq_f32(even, odd);
            vst1q_f32(ptr.add(i), vzip1q_f32(sum, diff));
            vst1q_f32(ptr.add(i + 4), vzip2q_f32(sum, diff));
            i += 8;
        }
        // Scalar tail: apply sign1 then butterfly
        while i + 2 <= n {
            let s0 = if ((first_bits[i / 8] >> (i & 7)) & 1) != 0 {
                -1.0f32
            } else {
                1.0
            };
            let s1 = if ((first_bits[(i + 1) / 8] >> ((i + 1) & 7)) & 1) != 0 {
                -1.0f32
            } else {
                1.0
            };
            let a = *ptr.add(i) * s0;
            let b = *ptr.add(i + 1) * s1;
            *ptr.add(i) = a + b;
            *ptr.add(i + 1) = a - b;
            i += 2;
        }
    }

    // Stage width=2: butterfly on (i, i+2) within groups of 4
    {
        let mut i = 0usize;
        while i + 8 <= n {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            let lo = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
            let hi = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
            let sum = vaddq_f32(lo, hi);
            let diff = vsubq_f32(lo, hi);
            vst1q_f32(
                ptr.add(i),
                vcombine_f32(vget_low_f32(sum), vget_low_f32(diff)),
            );
            vst1q_f32(
                ptr.add(i + 4),
                vcombine_f32(vget_high_f32(sum), vget_high_f32(diff)),
            );
            i += 8;
        }
        while i + 4 <= n {
            let a0 = *ptr.add(i);
            let a1 = *ptr.add(i + 1);
            let a2 = *ptr.add(i + 2);
            let a3 = *ptr.add(i + 3);
            *ptr.add(i) = a0 + a2;
            *ptr.add(i + 1) = a1 + a3;
            *ptr.add(i + 2) = a0 - a2;
            *ptr.add(i + 3) = a1 - a3;
            i += 4;
        }
    }

    // Stages width=4,8,...: half-block butterfly
    let mut width = 4usize;
    while width < n {
        let stride = width * 2;
        let mut base = 0usize;
        while base < n {
            let mut i = 0usize;
            while i + 4 <= width {
                let lo = vld1q_f32(ptr.add(base + i));
                let hi = vld1q_f32(ptr.add(base + width + i));
                vst1q_f32(ptr.add(base + i), vaddq_f32(lo, hi));
                vst1q_f32(ptr.add(base + width + i), vsubq_f32(lo, hi));
                i += 4;
            }
            while i < width {
                let a = *ptr.add(base + i);
                let b = *ptr.add(base + width + i);
                *ptr.add(base + i) = a + b;
                *ptr.add(base + width + i) = a - b;
                i += 1;
            }
            base += stride;
        }
        width = stride;
    }

    // Final scaling fused with sign2: multiply by (1/sqrt(n)) * sign2
    let scale = 1.0 / (n as f32).sqrt();
    let vscale = vdupq_n_f32(scale);
    {
        let mut i = 0usize;
        while i + 8 <= n {
            let byte = second_bits[i / 8];
            let sp = neg_table[byte as usize].as_ptr();
            let ss0 = vmulq_f32(vscale, vld1q_f32(sp));
            let ss1 = vmulq_f32(vscale, vld1q_f32(sp.add(4)));
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), ss0));
            vst1q_f32(ptr.add(i + 4), vmulq_f32(vld1q_f32(ptr.add(i + 4)), ss1));
            i += 8;
        }
        while i < n {
            let s = if ((second_bits[i / 8] >> (i & 7)) & 1) != 0 {
                -scale
            } else {
                scale
            };
            *ptr.add(i) *= s;
            i += 1;
        }
    }
}

#[inline]
fn turboquant_transform_with_bits(values: &mut [f32], first_bits: &[u8], second_bits: &[u8]) {
    #[cfg(target_arch = "aarch64")]
    if values.len().is_power_of_two() && values.len() >= 8 {
        unsafe { turboquant_transform_fused_neon(values, first_bits, second_bits) };
        return;
    }
    turboquant_apply_signs_bits(values, first_bits);
    turboquant_fwht_inplace(values);
    turboquant_apply_signs_bits(values, second_bits);
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
const fn turboquant_build_q2_decode_table() -> [[f32; 4]; 256] {
    let mut table = [[0.0f32; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let b = byte as u8;
        table[byte] = [
            TURBOQUANT_CENTROIDS[(b & 0b11) as usize],
            TURBOQUANT_CENTROIDS[((b >> 2) & 0b11) as usize],
            TURBOQUANT_CENTROIDS[((b >> 4) & 0b11) as usize],
            TURBOQUANT_CENTROIDS[((b >> 6) & 0b11) as usize],
        ];
        byte += 1;
    }
    table
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
const fn turboquant_build_sign_decode_table() -> [[f32; 8]; 256] {
    let mut table = [[0.0f32; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let b = byte as u8;
        let mut bit = 0usize;
        while bit < 8 {
            table[byte][bit] = if ((b >> bit) & 1) != 0 { 1.0 } else { -1.0 };
            bit += 1;
        }
        byte += 1;
    }
    table
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
static TURBOQUANT_Q2_DECODE_TABLE: [[f32; 4]; 256] = turboquant_build_q2_decode_table();
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
static TURBOQUANT_SIGN_DECODE_TABLE: [[f32; 8]; 256] = turboquant_build_sign_decode_table();

/// Like TURBOQUANT_SIGN_DECODE_TABLE but with inverted convention:
/// bit=1 → -1.0 (negate), bit=0 → +1.0 (keep).
/// Used by turboquant_apply_signs_bits where bit=1 means "negate this element".
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
const fn turboquant_build_neg_sign_decode_table() -> [[f32; 8]; 256] {
    let mut table = [[0.0f32; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let b = byte as u8;
        let mut bit = 0usize;
        while bit < 8 {
            table[byte][bit] = if ((b >> bit) & 1) != 0 { -1.0 } else { 1.0 };
            bit += 1;
        }
        byte += 1;
    }
    table
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
static TURBOQUANT_NEG_SIGN_DECODE_TABLE: [[f32; 8]; 256] = turboquant_build_neg_sign_decode_table();
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn turboquant_q2_decode_table() -> &'static [[f32; 4]; 256] {
    &TURBOQUANT_Q2_DECODE_TABLE
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn turboquant_sign_decode_table() -> &'static [[f32; 8]; 256] {
    &TURBOQUANT_SIGN_DECODE_TABLE
}

#[inline]
fn turboquant_aux_index(row_index: usize, kv_head: usize, n_kv_heads: usize) -> usize {
    row_index * n_kv_heads + kv_head
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn turboquant_fwht_inplace_scalar(values: &mut [f32]) {
    let mut width = 1usize;
    while width < values.len() {
        let stride = width * 2;
        let mut base = 0usize;
        while base < values.len() {
            for i in 0..width {
                let a = values[base + i];
                let b = values[base + i + width];
                values[base + i] = a + b;
                values[base + i + width] = a - b;
            }
            base += stride;
        }
        width = stride;
    }
    let scale = 1.0 / (values.len() as f32).sqrt();
    for value in values.iter_mut() {
        *value *= scale;
    }
}

/// NEON FWHT for power-of-two lengths ≥ 8.
///
/// Stage width=1 — adjacent pairs (i, i+1):
///   [a,b,c,d, e,f,g,h] → vuzp deinterleaves even/odd, vadd/vsub, vzip re-interleaves.
///
/// Stage width=2 — pairs (i, i+2) within groups of 4:
///   [a0,a1,a2,a3, b0,b1,b2,b3]: split each vec4 into low/high, add/sub, reassemble.
///
/// Stages width≥4 — half-block butterfly:
///   lo = first half, hi = second half → sum/diff, straightforward 4-wide.
#[cfg(target_arch = "aarch64")]
unsafe fn turboquant_fwht_inplace_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = values.len();
    let ptr = values.as_mut_ptr();

    // width=1: butterfly on adjacent pairs, 8 at a time
    {
        let mut i = 0usize;
        while i + 8 <= n {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            // even = [v[0],v[2],v[4],v[6]], odd = [v[1],v[3],v[5],v[7]]
            let even = vuzp1q_f32(v0, v1);
            let odd = vuzp2q_f32(v0, v1);
            let sum = vaddq_f32(even, odd);
            let diff = vsubq_f32(even, odd);
            // re-interleave: [sum0,diff0,sum1,diff1] and [sum2,diff2,sum3,diff3]
            vst1q_f32(ptr.add(i), vzip1q_f32(sum, diff));
            vst1q_f32(ptr.add(i + 4), vzip2q_f32(sum, diff));
            i += 8;
        }
        // scalar tail (n < 8 or n not multiple of 8)
        while i + 2 <= n {
            let a = *ptr.add(i);
            let b = *ptr.add(i + 1);
            *ptr.add(i) = a + b;
            *ptr.add(i + 1) = a - b;
            i += 2;
        }
    }

    // width=2: butterfly on pairs (i, i+2) within groups of 4, 8 at a time
    {
        let mut i = 0usize;
        while i + 8 <= n {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            // lo = [v0[0..2], v1[0..2]], hi = [v0[2..4], v1[2..4]]
            let lo = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
            let hi = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
            let sum = vaddq_f32(lo, hi);
            let diff = vsubq_f32(lo, hi);
            // reassemble: each group gets [sum_lo, diff_lo]
            vst1q_f32(
                ptr.add(i),
                vcombine_f32(vget_low_f32(sum), vget_low_f32(diff)),
            );
            vst1q_f32(
                ptr.add(i + 4),
                vcombine_f32(vget_high_f32(sum), vget_high_f32(diff)),
            );
            i += 8;
        }
        // scalar tail for groups of 4
        while i + 4 <= n {
            let a0 = *ptr.add(i);
            let a1 = *ptr.add(i + 1);
            let a2 = *ptr.add(i + 2);
            let a3 = *ptr.add(i + 3);
            *ptr.add(i) = a0 + a2;
            *ptr.add(i + 1) = a1 + a3;
            *ptr.add(i + 2) = a0 - a2;
            *ptr.add(i + 3) = a1 - a3;
            i += 4;
        }
    }

    // width=4,8,...: half-block butterfly, 4-wide
    let mut width = 4usize;
    while width < n {
        let stride = width * 2;
        let mut base = 0usize;
        while base < n {
            let mut i = 0usize;
            while i + 4 <= width {
                let lo = vld1q_f32(ptr.add(base + i));
                let hi = vld1q_f32(ptr.add(base + width + i));
                vst1q_f32(ptr.add(base + i), vaddq_f32(lo, hi));
                vst1q_f32(ptr.add(base + width + i), vsubq_f32(lo, hi));
                i += 4;
            }
            // scalar tail (when width is not a multiple of 4)
            while i < width {
                let a = *ptr.add(base + i);
                let b = *ptr.add(base + width + i);
                *ptr.add(base + i) = a + b;
                *ptr.add(base + width + i) = a - b;
                i += 1;
            }
            base += stride;
        }
        width = stride;
    }

    // Final scaling: multiply all by 1/sqrt(n)
    let vscale = vdupq_n_f32(1.0 / (n as f32).sqrt());
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(ptr.add(i));
        vst1q_f32(ptr.add(i), vmulq_f32(v, vscale));
        i += 4;
    }
    while i < n {
        *ptr.add(i) *= 1.0 / (n as f32).sqrt();
        i += 1;
    }
}

fn turboquant_fwht_inplace(values: &mut [f32]) {
    if !values.len().is_power_of_two() {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if values.len() >= 8 {
        unsafe { turboquant_fwht_inplace_neon(values) };
        return;
    }
    turboquant_fwht_inplace_scalar(values);
}

#[inline]
fn turboquant_quantize_code(x: f32) -> u8 {
    if x < TURBOQUANT_THRESHOLD_LO {
        0
    } else if x < 0.0 {
        1
    } else if x < TURBOQUANT_THRESHOLD_HI {
        2
    } else {
        3
    }
}

#[inline]
fn get_q2_at(src: &[u8], elem_idx: usize) -> u8 {
    (src[elem_idx / 4] >> ((elem_idx & 3) * 2)) & 0b11
}

#[inline]
fn set_sign_bit(dst: &mut [u8], elem_idx: usize, positive: bool) {
    let shift = elem_idx & 7;
    let byte = &mut dst[elem_idx / 8];
    if positive {
        *byte |= 1 << shift;
    } else {
        *byte &= !(1 << shift);
    }
}

#[inline]
fn get_sign_bit(src: &[u8], elem_idx: usize) -> bool {
    ((src[elem_idx / 8] >> (elem_idx & 7)) & 1) != 0
}

struct TurboquantHeadWrite<'a> {
    base: &'a mut [u8],
    sign: &'a mut [u8],
    elem_offset: usize,
    scale_out: &'a mut f32,
    residual_norm_out: &'a mut f32,
}

#[derive(Clone, Copy)]
struct TurboquantHeadRead<'a> {
    base: &'a [u8],
    sign: &'a [u8],
    elem_offset: usize,
    scale: f32,
    residual_norm: f32,
}

fn quantize_turboquant_head(
    src: &[f32],
    signs: &TurboSignRef<'_>,
    rotated: &mut [f32],
    residual: &mut [f32],
    dst: TurboquantHeadWrite<'_>,
) {
    debug_assert!(rotated.len() >= src.len());
    debug_assert!(residual.len() >= src.len());
    let rotated = &mut rotated[..src.len()];
    let residual = &mut residual[..src.len()];
    rotated.copy_from_slice(src);
    turboquant_transform_with_bits(rotated, signs.rot_pre, signs.rot_post);

    let sigma = l2_norm(rotated) / (rotated.len() as f32).sqrt();
    *dst.scale_out = sigma;
    let q2_base = dst.elem_offset / 4;
    if sigma == 0.0 {
        *dst.residual_norm_out = 0.0;
        let byte_count = src.len().div_ceil(4);
        dst.base[q2_base..q2_base + byte_count].fill(0);
        for i in 0..src.len() {
            set_sign_bit(dst.sign, dst.elem_offset + i, true);
        }
        return;
    }

    let inv_sigma = 1.0 / sigma;
    let mut i = 0usize;
    while i + 4 <= rotated.len() {
        let mut packed = 0u8;
        for k in 0..4 {
            let value = rotated[i + k];
            let code = turboquant_quantize_code(value * inv_sigma);
            residual[i + k] = value - TURBOQUANT_CENTROIDS[code as usize] * sigma;
            packed |= code << (k * 2);
        }
        dst.base[q2_base + i / 4] = packed;
        i += 4;
    }
    if i < rotated.len() {
        let mut packed = 0u8;
        for k in 0..(rotated.len() - i) {
            let value = rotated[i + k];
            let code = turboquant_quantize_code(value * inv_sigma);
            residual[i + k] = value - TURBOQUANT_CENTROIDS[code as usize] * sigma;
            packed |= code << (k * 2);
        }
        dst.base[q2_base + i / 4] = packed;
    }

    let gamma = l2_norm(residual);
    *dst.residual_norm_out = gamma;
    if gamma == 0.0 {
        for i in 0..src.len() {
            set_sign_bit(dst.sign, dst.elem_offset + i, true);
        }
        return;
    }

    for value in residual.iter_mut() {
        *value /= gamma;
    }
    turboquant_transform_with_bits(residual, signs.res_pre, signs.res_post);
    for (i, &value) in residual.iter().enumerate() {
        set_sign_bit(dst.sign, dst.elem_offset + i, value >= 0.0);
    }
}

fn turboquant_prepare_query(
    q_head: &[f32],
    signs: &TurboSignRef<'_>,
    rotated: &mut [f32],
    projected: &mut [f32],
) {
    debug_assert!(rotated.len() >= q_head.len());
    debug_assert!(projected.len() >= q_head.len());
    let rotated = &mut rotated[..q_head.len()];
    let projected = &mut projected[..q_head.len()];
    rotated.copy_from_slice(q_head);
    turboquant_transform_with_bits(rotated, signs.rot_pre, signs.rot_post);
    projected.copy_from_slice(rotated);
    turboquant_transform_with_bits(projected, signs.res_pre, signs.res_post);
}

fn turboquant_reset_residual_accum(residual_accum: &mut [f32], len: usize) {
    debug_assert!(residual_accum.len() >= len);
    residual_accum[..len].fill(0.0);
}

#[inline]
fn turboquant_residual_scale(head_size: usize, residual_norm: f32) -> f32 {
    if residual_norm == 0.0 {
        0.0
    } else {
        residual_norm * (std::f32::consts::FRAC_PI_2 / head_size as f32).sqrt()
    }
}

/// Unpack 16 Q2 elements from packed bytes into 4 NEON registers,
/// pre-scaled by `scale`. Eliminates per-iteration vmulq_f32 in the inner loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_q2x16_scaled_neon(
    base: &[u8],
    elem_offset: usize,
    vscale: std::arch::aarch64::float32x4_t,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    debug_assert_eq!(elem_offset & 3, 0);
    let packed = &base[elem_offset / 4..elem_offset / 4 + 4];
    let table = turboquant_q2_decode_table();
    (
        vmulq_f32(vld1q_f32(table[packed[0] as usize].as_ptr()), vscale),
        vmulq_f32(vld1q_f32(table[packed[1] as usize].as_ptr()), vscale),
        vmulq_f32(vld1q_f32(table[packed[2] as usize].as_ptr()), vscale),
        vmulq_f32(vld1q_f32(table[packed[3] as usize].as_ptr()), vscale),
    )
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_sign16_scaled_neon(
    sign: &[u8],
    elem_offset: usize,
    vscale: std::arch::aarch64::float32x4_t,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    debug_assert_eq!(elem_offset & 7, 0);
    let packed = &sign[elem_offset / 8..elem_offset / 8 + 2];
    let table = turboquant_sign_decode_table();
    let p0 = table[packed[0] as usize].as_ptr();
    let p1 = table[packed[1] as usize].as_ptr();
    (
        vmulq_f32(vld1q_f32(p0), vscale),
        vmulq_f32(vld1q_f32(p0.add(4)), vscale),
        vmulq_f32(vld1q_f32(p1), vscale),
        vmulq_f32(vld1q_f32(p1.add(4)), vscale),
    )
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_turboquant_head_neon(
    q_rotated: &[f32],
    q_residual_proj: &[f32],
    cache: TurboquantHeadRead<'_>,
) -> f32 {
    use std::arch::aarch64::*;

    let n = q_rotated.len();
    let has_base = cache.scale != 0.0;
    let residual_scale = turboquant_residual_scale(n, cache.residual_norm);
    let has_residual = residual_scale != 0.0;

    let mut vacc0 = vdupq_n_f32(0.0);
    let mut vacc1 = vdupq_n_f32(0.0);
    let mut vacc2 = vdupq_n_f32(0.0);
    let mut vacc3 = vdupq_n_f32(0.0);
    let mut racc0 = vdupq_n_f32(0.0);
    let mut racc1 = vdupq_n_f32(0.0);
    let mut racc2 = vdupq_n_f32(0.0);
    let mut racc3 = vdupq_n_f32(0.0);
    let mut i = 0usize;

    if has_base && has_residual {
        let vscale = vdupq_n_f32(cache.scale);
        let vresidual_scale = vdupq_n_f32(residual_scale);
        while i + 16 <= n {
            let (c0, c1, c2, c3) =
                unpack_q2x16_scaled_neon(cache.base, cache.elem_offset + i, vscale);
            let (s0, s1, s2, s3) =
                unpack_sign16_scaled_neon(cache.sign, cache.elem_offset + i, vresidual_scale);
            let q0 = vld1q_f32(q_rotated.as_ptr().add(i));
            let q1 = vld1q_f32(q_rotated.as_ptr().add(i + 4));
            let q2 = vld1q_f32(q_rotated.as_ptr().add(i + 8));
            let q3 = vld1q_f32(q_rotated.as_ptr().add(i + 12));
            let r0 = vld1q_f32(q_residual_proj.as_ptr().add(i));
            let r1 = vld1q_f32(q_residual_proj.as_ptr().add(i + 4));
            let r2 = vld1q_f32(q_residual_proj.as_ptr().add(i + 8));
            let r3 = vld1q_f32(q_residual_proj.as_ptr().add(i + 12));
            vacc0 = vfmaq_f32(vacc0, q0, c0);
            vacc1 = vfmaq_f32(vacc1, q1, c1);
            vacc2 = vfmaq_f32(vacc2, q2, c2);
            vacc3 = vfmaq_f32(vacc3, q3, c3);
            racc0 = vfmaq_f32(racc0, r0, s0);
            racc1 = vfmaq_f32(racc1, r1, s1);
            racc2 = vfmaq_f32(racc2, r2, s2);
            racc3 = vfmaq_f32(racc3, r3, s3);
            i += 16;
        }
    } else if has_base {
        let vscale = vdupq_n_f32(cache.scale);
        while i + 16 <= n {
            let (c0, c1, c2, c3) =
                unpack_q2x16_scaled_neon(cache.base, cache.elem_offset + i, vscale);
            vacc0 = vfmaq_f32(vacc0, vld1q_f32(q_rotated.as_ptr().add(i)), c0);
            vacc1 = vfmaq_f32(vacc1, vld1q_f32(q_rotated.as_ptr().add(i + 4)), c1);
            vacc2 = vfmaq_f32(vacc2, vld1q_f32(q_rotated.as_ptr().add(i + 8)), c2);
            vacc3 = vfmaq_f32(vacc3, vld1q_f32(q_rotated.as_ptr().add(i + 12)), c3);
            i += 16;
        }
    } else if has_residual {
        let vresidual_scale = vdupq_n_f32(residual_scale);
        while i + 16 <= n {
            let (s0, s1, s2, s3) =
                unpack_sign16_scaled_neon(cache.sign, cache.elem_offset + i, vresidual_scale);
            racc0 = vfmaq_f32(racc0, vld1q_f32(q_residual_proj.as_ptr().add(i)), s0);
            racc1 = vfmaq_f32(racc1, vld1q_f32(q_residual_proj.as_ptr().add(i + 4)), s1);
            racc2 = vfmaq_f32(racc2, vld1q_f32(q_residual_proj.as_ptr().add(i + 8)), s2);
            racc3 = vfmaq_f32(racc3, vld1q_f32(q_residual_proj.as_ptr().add(i + 12)), s3);
            i += 16;
        }
    }

    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(vacc0, vacc1), vaddq_f32(vacc2, vacc3)))
        + vaddvq_f32(vaddq_f32(vaddq_f32(racc0, racc1), vaddq_f32(racc2, racc3)));

    // scalar tail
    while i < n {
        if has_base {
            let code = get_q2_at(cache.base, cache.elem_offset + i) as usize;
            sum += q_rotated[i] * TURBOQUANT_CENTROIDS[code] * cache.scale;
        }
        if has_residual {
            sum += if get_sign_bit(cache.sign, cache.elem_offset + i) {
                residual_scale * q_residual_proj[i]
            } else {
                -residual_scale * q_residual_proj[i]
            };
        }
        i += 1;
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn axpy_turboquant_head_neon(
    base_accum: &mut [f32],
    residual_accum: &mut [f32],
    a: f32,
    cache: TurboquantHeadRead<'_>,
) {
    use std::arch::aarch64::*;

    let n = base_accum.len();
    let has_base = cache.scale != 0.0;
    let residual_scale = a * turboquant_residual_scale(n, cache.residual_norm);
    let has_residual = residual_scale != 0.0;

    let base_ptr = base_accum.as_mut_ptr();
    let res_ptr = residual_accum.as_mut_ptr();
    let mut i = 0usize;

    if has_base && has_residual {
        let vscale = vdupq_n_f32(a * cache.scale);
        let vresidual_scale = vdupq_n_f32(residual_scale);
        while i + 16 <= n {
            let (c0, c1, c2, c3) =
                unpack_q2x16_scaled_neon(cache.base, cache.elem_offset + i, vscale);
            let (s0, s1, s2, s3) =
                unpack_sign16_scaled_neon(cache.sign, cache.elem_offset + i, vresidual_scale);
            vst1q_f32(base_ptr.add(i), vaddq_f32(vld1q_f32(base_ptr.add(i)), c0));
            vst1q_f32(
                base_ptr.add(i + 4),
                vaddq_f32(vld1q_f32(base_ptr.add(i + 4)), c1),
            );
            vst1q_f32(
                base_ptr.add(i + 8),
                vaddq_f32(vld1q_f32(base_ptr.add(i + 8)), c2),
            );
            vst1q_f32(
                base_ptr.add(i + 12),
                vaddq_f32(vld1q_f32(base_ptr.add(i + 12)), c3),
            );
            vst1q_f32(res_ptr.add(i), vaddq_f32(vld1q_f32(res_ptr.add(i)), s0));
            vst1q_f32(
                res_ptr.add(i + 4),
                vaddq_f32(vld1q_f32(res_ptr.add(i + 4)), s1),
            );
            vst1q_f32(
                res_ptr.add(i + 8),
                vaddq_f32(vld1q_f32(res_ptr.add(i + 8)), s2),
            );
            vst1q_f32(
                res_ptr.add(i + 12),
                vaddq_f32(vld1q_f32(res_ptr.add(i + 12)), s3),
            );
            i += 16;
        }
    } else if has_base {
        let vscale = vdupq_n_f32(a * cache.scale);
        while i + 16 <= n {
            let (c0, c1, c2, c3) =
                unpack_q2x16_scaled_neon(cache.base, cache.elem_offset + i, vscale);
            vst1q_f32(base_ptr.add(i), vaddq_f32(vld1q_f32(base_ptr.add(i)), c0));
            vst1q_f32(
                base_ptr.add(i + 4),
                vaddq_f32(vld1q_f32(base_ptr.add(i + 4)), c1),
            );
            vst1q_f32(
                base_ptr.add(i + 8),
                vaddq_f32(vld1q_f32(base_ptr.add(i + 8)), c2),
            );
            vst1q_f32(
                base_ptr.add(i + 12),
                vaddq_f32(vld1q_f32(base_ptr.add(i + 12)), c3),
            );
            i += 16;
        }
    } else if has_residual {
        let vresidual_scale = vdupq_n_f32(residual_scale);
        while i + 16 <= n {
            let (s0, s1, s2, s3) =
                unpack_sign16_scaled_neon(cache.sign, cache.elem_offset + i, vresidual_scale);
            vst1q_f32(res_ptr.add(i), vaddq_f32(vld1q_f32(res_ptr.add(i)), s0));
            vst1q_f32(
                res_ptr.add(i + 4),
                vaddq_f32(vld1q_f32(res_ptr.add(i + 4)), s1),
            );
            vst1q_f32(
                res_ptr.add(i + 8),
                vaddq_f32(vld1q_f32(res_ptr.add(i + 8)), s2),
            );
            vst1q_f32(
                res_ptr.add(i + 12),
                vaddq_f32(vld1q_f32(res_ptr.add(i + 12)), s3),
            );
            i += 16;
        }
    }

    // scalar tail
    while i < n {
        if has_base {
            let code = get_q2_at(cache.base, cache.elem_offset + i) as usize;
            base_accum[i] += a * TURBOQUANT_CENTROIDS[code] * cache.scale;
        }
        if has_residual {
            residual_accum[i] += if get_sign_bit(cache.sign, cache.elem_offset + i) {
                residual_scale
            } else {
                -residual_scale
            };
        }
        i += 1;
    }
}

fn dot_turboquant_head_scalar(
    q_rotated: &[f32],
    q_residual_proj: &[f32],
    cache: TurboquantHeadRead<'_>,
) -> f32 {
    let mut acc = 0.0f32;
    if cache.scale != 0.0 {
        for (i, &qv) in q_rotated.iter().enumerate() {
            let code = get_q2_at(cache.base, cache.elem_offset + i) as usize;
            acc += qv * (TURBOQUANT_CENTROIDS[code] * cache.scale);
        }
    }
    let residual_scale = turboquant_residual_scale(q_rotated.len(), cache.residual_norm);
    if residual_scale != 0.0 {
        for (i, &qv) in q_residual_proj.iter().enumerate() {
            acc += if get_sign_bit(cache.sign, cache.elem_offset + i) {
                residual_scale * qv
            } else {
                -residual_scale * qv
            };
        }
    }
    acc
}

fn dot_turboquant_head(
    q_rotated: &[f32],
    q_residual_proj: &[f32],
    cache: TurboquantHeadRead<'_>,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_turboquant_head_neon(q_rotated, q_residual_proj, cache);
    }
    #[allow(unreachable_code)]
    dot_turboquant_head_scalar(q_rotated, q_residual_proj, cache)
}

fn axpy_turboquant_head_scalar(
    base_accum: &mut [f32],
    residual_accum: &mut [f32],
    a: f32,
    cache: TurboquantHeadRead<'_>,
) {
    if cache.scale != 0.0 {
        for (i, dst) in base_accum.iter_mut().enumerate() {
            let code = get_q2_at(cache.base, cache.elem_offset + i) as usize;
            *dst += a * TURBOQUANT_CENTROIDS[code] * cache.scale;
        }
    }
    let residual_scale = a * turboquant_residual_scale(base_accum.len(), cache.residual_norm);
    if residual_scale != 0.0 {
        for (i, dst) in residual_accum.iter_mut().enumerate() {
            *dst += if get_sign_bit(cache.sign, cache.elem_offset + i) {
                residual_scale
            } else {
                -residual_scale
            };
        }
    }
}

fn axpy_turboquant_head(
    base_accum: &mut [f32],
    residual_accum: &mut [f32],
    a: f32,
    cache: TurboquantHeadRead<'_>,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        axpy_turboquant_head_neon(base_accum, residual_accum, a, cache);
        return;
    }
    #[allow(unreachable_code)]
    axpy_turboquant_head_scalar(base_accum, residual_accum, a, cache)
}

fn finalize_turboquant_value_head(
    base_accum: &mut [f32],
    residual_accum: &mut [f32],
    signs: &TurboSignRef<'_>,
) {
    // inverse residual transform: swap pre/post salts
    turboquant_transform_with_bits(residual_accum, signs.res_post, signs.res_pre);
    let len = base_accum.len();
    accum(base_accum, residual_accum, len);
    // inverse rotate transform: swap pre/post salts
    turboquant_transform_with_bits(base_accum, signs.rot_post, signs.rot_pre);
}

#[inline]
fn dot_q8_row(q: &[f32], cache: &[i8], row_offset: usize, scale: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q8_row_neon(q, cache, row_offset, scale);
    }
    #[allow(unreachable_code)]
    {
        let mut acc = 0.0f32;
        for (i, &qv) in q.iter().enumerate() {
            acc += qv * (cache[row_offset + i] as f32 * scale);
        }
        acc
    }
}

#[inline]
fn axpy_q8_row(dst: &mut [f32], a: f32, cache: &[i8], row_offset: usize, scale: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        axpy_q8_row_neon(dst, a, cache, row_offset, scale);
        return;
    }
    #[allow(unreachable_code)]
    {
        let scaled = a * scale;
        for (i, d) in dst.iter_mut().enumerate() {
            *d += scaled * cache[row_offset + i] as f32;
        }
    }
}

#[inline]
unsafe fn dot_q8_row_blocks_ptr(
    q: &[f32],
    cache: &[i8],
    row_offset: usize,
    scales_ptr: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let n_full = n_blocks.min(q.len() / Q4_BLOCK_SIZE);
    let mut bi = 0usize;

    while bi + 4 <= n_full {
        let s0 = *scales_ptr.add(bi);
        let s1 = *scales_ptr.add(bi + 1);
        let s2 = *scales_ptr.add(bi + 2);
        let s3 = *scales_ptr.add(bi + 3);
        let b0 = bi * Q4_BLOCK_SIZE;
        let b1 = b0 + Q4_BLOCK_SIZE;
        let b2 = b1 + Q4_BLOCK_SIZE;
        let b3 = b2 + Q4_BLOCK_SIZE;
        acc += dot_q8_row(&q[b0..b1], cache, row_offset + b0, s0);
        acc += dot_q8_row(&q[b1..b2], cache, row_offset + b1, s1);
        acc += dot_q8_row(&q[b2..b3], cache, row_offset + b2, s2);
        acc += dot_q8_row(&q[b3..b3 + Q4_BLOCK_SIZE], cache, row_offset + b3, s3);
        bi += 4;
    }
    while bi < n_full {
        let start = bi * Q4_BLOCK_SIZE;
        let end = start + Q4_BLOCK_SIZE;
        acc += dot_q8_row(
            &q[start..end],
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    while bi < n_blocks {
        let start = bi * Q4_BLOCK_SIZE;
        if start >= q.len() {
            break;
        }
        let end = (start + Q4_BLOCK_SIZE).min(q.len());
        acc += dot_q8_row(
            &q[start..end],
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    acc
}

#[inline]
unsafe fn axpy_q8_row_blocks_ptr(
    dst: &mut [f32],
    a: f32,
    cache: &[i8],
    row_offset: usize,
    scales_ptr: *const f32,
    n_blocks: usize,
) {
    let n_full = n_blocks.min(dst.len() / Q4_BLOCK_SIZE);
    let mut bi = 0usize;

    while bi + 4 <= n_full {
        let s0 = *scales_ptr.add(bi);
        let s1 = *scales_ptr.add(bi + 1);
        let s2 = *scales_ptr.add(bi + 2);
        let s3 = *scales_ptr.add(bi + 3);
        let b0 = bi * Q4_BLOCK_SIZE;
        let b1 = b0 + Q4_BLOCK_SIZE;
        let b2 = b1 + Q4_BLOCK_SIZE;
        let b3 = b2 + Q4_BLOCK_SIZE;
        axpy_q8_row(&mut dst[b0..b1], a, cache, row_offset + b0, s0);
        axpy_q8_row(&mut dst[b1..b2], a, cache, row_offset + b1, s1);
        axpy_q8_row(&mut dst[b2..b3], a, cache, row_offset + b2, s2);
        axpy_q8_row(
            &mut dst[b3..b3 + Q4_BLOCK_SIZE],
            a,
            cache,
            row_offset + b3,
            s3,
        );
        bi += 4;
    }
    while bi < n_full {
        let start = bi * Q4_BLOCK_SIZE;
        let end = start + Q4_BLOCK_SIZE;
        axpy_q8_row(
            &mut dst[start..end],
            a,
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    while bi < n_blocks {
        let start = bi * Q4_BLOCK_SIZE;
        if start >= dst.len() {
            break;
        }
        let end = (start + Q4_BLOCK_SIZE).min(dst.len());
        axpy_q8_row(
            &mut dst[start..end],
            a,
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
}

#[inline]
fn qwen35_uses_mrope(p: &Config) -> bool {
    p.is_qwen35 && p.rope_sections[0] > 0 && p.rope_sections[1] > 0
}

fn rebuild_rope_cache(p: &Config, s: &mut RunState, pos: usize, is_swa_layer: bool) {
    let current_is_swa = if is_swa_layer { 1 } else { 0 };
    if s.rope_cache_pos == pos as isize && s.rope_cache_is_swa == current_is_swa {
        return;
    }

    let rope_freqs = if p.is_gemma3 && is_swa_layer {
        &s.rope_freqs_swa
    } else {
        &s.rope_freqs
    };
    let rope_half = s.rope_cos.len();

    if qwen35_uses_mrope(p) {
        // llama.cpp M-RoPE text path expands scalar position into [t,h,w,e] = [pos,pos,pos,0].
        let pos_streams = [pos as f32, pos as f32, pos as f32, 0.0f32];
        let section_total = p.rope_sections.iter().sum::<usize>();
        let section_h = p.rope_sections[0];
        let section_w = section_h + p.rope_sections[1];
        let section_e = section_w + p.rope_sections[2];

        for (i, ((cos, sin), &freq)) in s
            .rope_cos
            .iter_mut()
            .zip(s.rope_sin.iter_mut())
            .zip(rope_freqs.iter())
            .take(rope_half)
            .enumerate()
        {
            let sector = i % section_total;
            let pos_value = if sector < section_h {
                pos_streams[0]
            } else if sector < section_w {
                pos_streams[1]
            } else if sector < section_e {
                pos_streams[2]
            } else {
                pos_streams[3]
            };
            let val = pos_value * freq;
            *cos = val.cos();
            *sin = val.sin();
        }
    } else {
        for ((cos, sin), &freq) in s
            .rope_cos
            .iter_mut()
            .zip(s.rope_sin.iter_mut())
            .zip(rope_freqs.iter())
            .take(rope_half)
        {
            let val = pos as f32 * freq;
            *cos = val.cos();
            *sin = val.sin();
        }
    }

    s.rope_cache_pos = pos as isize;
    s.rope_cache_is_swa = current_is_swa;
}

pub(crate) fn malloc_run_state(p: &Config) -> Result<RunState, String> {
    let head_size = if p.head_dim > 0 {
        p.head_dim
    } else {
        p.dim / p.n_heads
    };
    let kv_dim = p.n_kv_heads * head_size;
    let q_dim = p.n_heads * head_size;
    let ssm_inner = p.ssm_inner_size;
    let ssm_k_heads = p.ssm_group_count;
    let ssm_v_heads = p.ssm_time_step_rank;
    let ssm_head_dim = p.ssm_state_size;
    let ssm_conv_dim = if p.is_qwen3next {
        ssm_inner + 2 * ssm_k_heads * ssm_head_dim
    } else {
        0
    };
    let ssm_conv_hist = if p.is_qwen3next {
        p.ssm_conv_kernel.saturating_sub(1)
    } else {
        0
    };
    let ssm_state_stride = if p.is_qwen3next {
        ssm_v_heads * ssm_head_dim * ssm_head_dim
    } else {
        0
    };
    let ssm_conv_stride = if p.is_qwen3next {
        ssm_conv_hist * ssm_conv_dim
    } else {
        0
    };
    let max_dim = p.dim.max(q_dim);
    let ffn_dim = p
        .hidden_dim
        .max(p.expert_hidden_dim)
        .max(p.shared_expert_hidden_dim);
    let scratch_dim = ffn_dim.max(ssm_conv_dim).max(ssm_inner).max(ssm_head_dim);
    let rope_dim = if p.rope_dim > 0 {
        p.rope_dim
    } else {
        head_size
    };
    let rope_size = rope_dim / 2;
    let mut rope_freqs = vec![0.0f32; rope_size];
    for (i, freq) in rope_freqs.iter_mut().enumerate() {
        *freq = 1.0 / p.rope_theta.powf((i * 2) as f32 / rope_dim as f32);
    }

    let swa_theta = if p.rope_theta_swa > 0.0 {
        p.rope_theta_swa
    } else {
        10_000.0
    };
    let mut rope_freqs_swa = vec![0.0f32; rope_size];
    for (i, freq) in rope_freqs_swa.iter_mut().enumerate() {
        *freq = 1.0 / swa_theta.powf((i * 2) as f32 / rope_dim as f32);
    }

    let att_len = p
        .n_heads
        .checked_mul(p.seq_len)
        .ok_or_else(|| "overflow while computing attention buffer size".to_string())?;
    let kv_cache_rows = p
        .n_layers
        .checked_mul(p.seq_len)
        .ok_or_else(|| "overflow while computing kv cache rows".to_string())?;
    let kv_cache_len = kv_cache_rows
        .checked_mul(kv_dim)
        .ok_or_else(|| "overflow while computing kv cache size".to_string())?;
    let kv_cache_turbo_base_len = kv_cache_len.div_ceil(4);
    let kv_cache_turbo_sign_len = kv_cache_len.div_ceil(8);
    let kv_cache_block_scale_len = kv_cache_rows
        .checked_mul((kv_dim / Q4_BLOCK_SIZE).max(1))
        .ok_or_else(|| "overflow while computing kv scale buffer size".to_string())?;
    let kv_cache_head_aux_len = kv_cache_rows
        .checked_mul(p.n_kv_heads.max(1))
        .ok_or_else(|| "overflow while computing turbo kv aux size".to_string())?;
    let kv_cache_scale_len = kv_cache_block_scale_len.max(kv_cache_head_aux_len);

    let requested_mode = kv_cache_mode();
    let (
        kv_cache_format,
        key_cache_q8,
        value_cache_q8,
        key_cache_turbo_base,
        value_cache_turbo_base,
        key_cache_turbo_sign,
        value_cache_turbo_sign,
    ) = match requested_mode {
        SwitchKvCacheMode::Q8 => {
            let key = alloc_i8(kv_cache_len, "Q8 key cache")?;
            let value = alloc_i8(kv_cache_len, "Q8 value cache")?;
            (
                KvCacheFormat::Q8,
                key,
                value,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        }
        SwitchKvCacheMode::Turbo => {
            let key_base = alloc_u8(kv_cache_turbo_base_len, "Turbo key base cache")?;
            let value_base = alloc_u8(kv_cache_turbo_base_len, "Turbo value base cache")?;
            let key_sign = alloc_u8(kv_cache_turbo_sign_len, "Turbo key residual cache")?;
            let value_sign = alloc_u8(kv_cache_turbo_sign_len, "Turbo value residual cache")?;
            (
                KvCacheFormat::Turbo,
                Vec::new(),
                Vec::new(),
                key_base,
                value_base,
                key_sign,
                value_sign,
            )
        }
    };

    Ok(RunState {
        x: vec![0.0; p.dim],
        xb: vec![0.0; max_dim],
        xb2: vec![0.0; p.dim],
        hb: vec![0.0; scratch_dim],
        hb2: vec![0.0; scratch_dim],
        moe_tmp: vec![0.0; p.dim],
        moe_logits: vec![0.0; p.n_experts],
        moe_topk_indices: vec![0usize; p.n_experts_used.max(1)],
        moe_topk_weights: vec![0.0f32; p.n_experts_used.max(1)],
        moe_scores: vec![0.0; p.n_experts],
        moe_selected_group: vec![true; p.moe_n_group.max(1)],
        moe_group_scores: vec![0.0; p.moe_n_group.max(1)],
        moe_group_rank: vec![0usize; p.moe_n_group.max(1)],
        q: vec![0.0; q_dim],
        k: vec![0.0; kv_dim],
        v: vec![0.0; kv_dim],
        ssm_qkv: vec![0.0; ssm_conv_dim],
        ssm_conv: vec![0.0; ssm_conv_dim],
        ssm_q: vec![0.0; ssm_inner],
        ssm_k: vec![0.0; ssm_inner],
        ssm_v: vec![0.0; ssm_inner],
        ssm_z: vec![0.0; ssm_inner],
        ssm_ba: vec![0.0; 2 * ssm_v_heads],
        ssm_gate_exp: vec![0.0; ssm_v_heads],
        ssm_beta: vec![0.0; ssm_v_heads],
        ssm_proj: vec![0.0; ssm_inner],
        ssm_kv_mem: vec![0.0; ssm_inner],
        ssm_delta: vec![0.0; ssm_inner],
        ssm_conv_state: vec![0.0; p.n_layers * ssm_conv_stride],
        ssm_state: vec![0.0; p.n_layers * ssm_state_stride],
        att: alloc_f32(att_len, "attention buffer")?,
        logits: vec![0.0; p.vocab_size],
        kv_cache_format,
        key_cache_q8,
        value_cache_q8,
        key_cache_turbo_base,
        value_cache_turbo_base,
        key_cache_turbo_sign,
        value_cache_turbo_sign,
        // Q8 block-scale mode uses one scale per Q4_BLOCK_SIZE-element block per row, plain Q8
        // uses one scale per row, and Turbo uses one scale per KV head and row.
        key_cache_scale: alloc_f32(kv_cache_scale_len, "KV key scale buffer")?,
        value_cache_scale: alloc_f32(kv_cache_scale_len, "KV value scale buffer")?,
        key_cache_residual_norm: alloc_f32(kv_cache_head_aux_len, "KV key residual norm buffer")?,
        value_cache_residual_norm: alloc_f32(
            kv_cache_head_aux_len,
            "KV value residual norm buffer",
        )?,
        turbo_sign_table: if kv_cache_format == KvCacheFormat::Turbo {
            turboquant_build_sign_table(p.n_layers, p.n_kv_heads.max(1), head_size)
        } else {
            Vec::new()
        },
        turbo_scratch0: vec![0.0; head_size],
        turbo_scratch1: vec![0.0; head_size],
        turbo_scratch2: vec![0.0; head_size],
        rope_freqs,
        rope_freqs_swa,
        rope_cos: vec![0.0; rope_size],
        rope_sin: vec![0.0; rope_size],
        rope_cache_pos: -1,
        rope_cache_is_swa: -1,
        head_size,
        kv_dim,
        q_dim,
        kv_mul: p.n_heads / p.n_kv_heads,
        attn_scale: 1.0 / (head_size as f32).sqrt(),
        embed_scale: (p.dim as f32).sqrt(),
    })
}

pub(crate) fn transformer(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Token(token),
        pos,
        true,
        p,
        s,
        w,
        mapped,
        None,
    )
}

pub(crate) fn transformer_without_logits(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Token(token),
        pos,
        false,
        p,
        s,
        w,
        mapped,
        None,
    )
}

pub(crate) fn transformer_with_embedding(
    embedding: &[f32],
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Embedding(embedding),
        pos,
        true,
        p,
        s,
        w,
        mapped,
        None,
    )
}

pub(crate) fn transformer_with_embedding_without_logits(
    embedding: &[f32],
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Embedding(embedding),
        pos,
        false,
        p,
        s,
        w,
        mapped,
        None,
    )
}

pub(crate) fn transformer_with_moe_executor(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
    moe_executor: &mut dyn MoeExpertExecutor,
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Token(token),
        pos,
        true,
        p,
        s,
        w,
        mapped,
        Some(moe_executor),
    )
}

pub(crate) fn transformer_without_logits_with_moe_executor(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
    moe_executor: &mut dyn MoeExpertExecutor,
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Token(token),
        pos,
        false,
        p,
        s,
        w,
        mapped,
        Some(moe_executor),
    )
}

pub(crate) fn transformer_with_embedding_with_moe_executor(
    embedding: &[f32],
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
    moe_executor: &mut dyn MoeExpertExecutor,
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Embedding(embedding),
        pos,
        true,
        p,
        s,
        w,
        mapped,
        Some(moe_executor),
    )
}

pub(crate) fn transformer_with_embedding_without_logits_with_moe_executor(
    embedding: &[f32],
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
    moe_executor: &mut dyn MoeExpertExecutor,
) -> Result<(), String> {
    transformer_inner(
        TransformerInput::Embedding(embedding),
        pos,
        false,
        p,
        s,
        w,
        mapped,
        Some(moe_executor),
    )
}

#[derive(Clone, Copy)]
enum TransformerInput<'a> {
    Token(usize),
    Embedding(&'a [f32]),
}

#[allow(clippy::too_many_arguments)]
fn transformer_inner(
    input: TransformerInput<'_>,
    pos: usize,
    compute_logits: bool,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
    mut moe_executor: Option<&mut dyn MoeExpertExecutor>,
) -> Result<(), String> {
    let dim = p.dim;
    let hidden_dim = p.hidden_dim;
    let head_size = s.head_size;
    let kv_dim = s.kv_dim;
    let q_dim = s.q_dim;
    let kv_mul = s.kv_mul;
    let eps = if p.rms_norm_eps > 0.0 {
        p.rms_norm_eps
    } else {
        1e-5
    };
    let do_layer_debug =
        layer_debug_enabled() && layer_debug_pos().map_or(pos == 0, |p0| pos == p0);
    let mut deepstack_embedding: Option<&[f32]> = None;

    if let TransformerInput::Embedding(input_embedding) = input {
        if input_embedding.len() == dim {
            s.x[..dim].copy_from_slice(input_embedding);
        } else if p.n_deepstack_layers > 0 && input_embedding.len() == p.input_embedding_dim {
            s.x[..dim].copy_from_slice(&input_embedding[..dim]);
            deepstack_embedding = Some(&input_embedding[dim..]);
        } else {
            return Err(format!(
                "embedding input length mismatch: got {}, expected {} or {}",
                input_embedding.len(),
                dim,
                p.input_embedding_dim
            ));
        }
    } else {
        let TransformerInput::Token(token) = input else {
            return Err("missing token input for transformer step".to_string());
        };
        let emb_row = &w.token_embedding_table[token * dim..(token + 1) * dim];
        s.x[..dim].copy_from_slice(emb_row);

        if p.is_gemma3 {
            scale_slice_inplace(&mut s.x[..dim], s.embed_scale);
        }
    }

    for l in 0..p.n_layers {
        if p.is_bert_family {
            // Post-norm architecture: no pre-attention norm — feed x directly into attention.
            s.xb[..dim].copy_from_slice(&s.x[..dim]);
        } else if p.is_gemma3 {
            rmsnorm_gemma(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        let is_qwen3next_ssm_layer = p.is_qwen3next && w.attn_qkv[l].rows > 0;
        if is_qwen3next_ssm_layer {
            let offloaded = if let Some(exec) = moe_executor.as_deref_mut() {
                exec.try_compute_ssm_layer(l, &s.xb[..dim], &mut s.xb2[..dim])?
            } else {
                false
            };
            if !offloaded {
                qwen3next_linear_attention_autoregressive(l, p, s, w, mapped, eps)?;
            }
        } else {
            let attn_prof = prof_start();
            let mut qwen3next_packed_q_gate = false;
            if p.is_qwen3next {
                if w.wq[l].rows >= 2 * q_dim {
                    // Qwen3Next full-attn packs Q and gate interleaved per head:
                    // [q_head0, gate_head0, q_head1, gate_head1, ...]
                    matmul_quantized_rows(
                        &mut s.hb[..2 * q_dim],
                        &s.xb[..dim],
                        &w.wq[l],
                        0,
                        2 * q_dim,
                        mapped,
                    )?;
                    if p.n_heads >= par_attn_min_heads() {
                        let hb_src = &s.hb[..2 * q_dim];
                        s.q[..q_dim].par_chunks_mut(head_size).enumerate().for_each(
                            |(h, q_dst)| {
                                let src_base = h * 2 * head_size;
                                q_dst.copy_from_slice(&hb_src[src_base..src_base + head_size]);
                            },
                        );
                    } else {
                        for h in 0..p.n_heads {
                            let src_base = h * 2 * head_size;
                            let dst_base = h * head_size;
                            s.q[dst_base..dst_base + head_size]
                                .copy_from_slice(&s.hb[src_base..src_base + head_size]);
                        }
                    }
                    qwen3next_packed_q_gate = true;
                } else if w.wq[l].rows == q_dim {
                    matmul_quantized(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.q[..q_dim],
                        &s.xb[..dim],
                        &w.wq[l],
                        0,
                        q_dim,
                        mapped,
                    )?;
                }
                if w.wk[l].rows == kv_dim {
                    matmul_quantized(&mut s.k[..kv_dim], &s.xb[..dim], &w.wk[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.k[..kv_dim],
                        &s.xb[..dim],
                        &w.wk[l],
                        0,
                        kv_dim,
                        mapped,
                    )?;
                }
                if w.wv[l].rows == kv_dim {
                    matmul_quantized(&mut s.v[..kv_dim], &s.xb[..dim], &w.wv[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.v[..kv_dim],
                        &s.xb[..dim],
                        &w.wv[l],
                        0,
                        kv_dim,
                        mapped,
                    )?;
                }
            } else if p.is_bert_family && w.wq[l].rows == q_dim + 2 * kv_dim {
                // Fused QKV: Q rows [0, q_dim), K rows [q_dim, q_dim+kv_dim), V rows after.
                matmul_quantized_rows(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], 0, q_dim, mapped)?;
                matmul_quantized_rows(
                    &mut s.k[..kv_dim],
                    &s.xb[..dim],
                    &w.wq[l],
                    q_dim,
                    kv_dim,
                    mapped,
                )?;
                matmul_quantized_rows(
                    &mut s.v[..kv_dim],
                    &s.xb[..dim],
                    &w.wq[l],
                    q_dim + kv_dim,
                    kv_dim,
                    mapped,
                )?;
            } else {
                matmul_quantized(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], mapped)?;
                matmul_quantized(&mut s.k[..kv_dim], &s.xb[..dim], &w.wk[l], mapped)?;
                matmul_quantized(&mut s.v[..kv_dim], &s.xb[..dim], &w.wv[l], mapped)?;
            }

            validate_bf16_projection_rows(
                p,
                pos,
                l,
                "attn_k",
                &w.wk[l],
                &s.xb[..dim],
                &s.k[..kv_dim],
                mapped,
            );
            validate_bf16_projection_rows(
                p,
                pos,
                l,
                "attn_v",
                &w.wv[l],
                &s.xb[..dim],
                &s.v[..kv_dim],
                mapped,
            );

            if p.is_qwen2 && !w.attn_q_bias.is_empty() {
                let qb = &w.attn_q_bias[l * q_dim..(l + 1) * q_dim];
                let kb = &w.attn_k_bias[l * kv_dim..(l + 1) * kv_dim];
                let vb = &w.attn_v_bias[l * kv_dim..(l + 1) * kv_dim];
                for (q, &b) in s.q[..q_dim].iter_mut().zip(qb.iter()) {
                    *q += b;
                }
                for ((k, v), (&kbv, &vbv)) in s.k[..kv_dim]
                    .iter_mut()
                    .zip(s.v[..kv_dim].iter_mut())
                    .zip(kb.iter().zip(vb.iter()))
                {
                    *k += kbv;
                    *v += vbv;
                }
            }

            if !w.attn_q_norm.is_empty()
                && !w.attn_k_norm.is_empty()
                && !w.attn_qk_norm_present.is_empty()
                && w.attn_qk_norm_present[l]
            {
                let q_norm = &w.attn_q_norm[l * head_size..(l + 1) * head_size];
                let k_norm = &w.attn_k_norm[l * head_size..(l + 1) * head_size];
                rmsnorm_per_head_gemma_inplace(
                    &mut s.q[..q_dim],
                    q_norm,
                    p.n_heads,
                    head_size,
                    eps,
                );
                rmsnorm_per_head_gemma_inplace(
                    &mut s.k[..kv_dim],
                    k_norm,
                    p.n_kv_heads,
                    head_size,
                    eps,
                );
            }

            let is_swa_layer = p.swa_pattern > 0 && (l % p.swa_pattern < p.swa_pattern - 1);
            let rope_half = s.rope_cos.len();
            rebuild_rope_cache(p, s, pos, is_swa_layer);

            if p.is_gemma3 || p.is_qwen2 || p.is_qwen3vl || p.is_qwen3moe || p.is_qwen3next {
                let pair_offset = rope_half;
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    for i in 0..rope_half {
                        let fcr = s.rope_cos[i];
                        let fci = s.rope_sin[i];
                        let v0 = s.q[hs + i];
                        let v1 = s.q[hs + i + pair_offset];
                        s.q[hs + i] = v0 * fcr - v1 * fci;
                        s.q[hs + i + pair_offset] = v0 * fci + v1 * fcr;
                    }
                }
                for h in 0..p.n_kv_heads {
                    let hs = h * head_size;
                    for i in 0..rope_half {
                        let fcr = s.rope_cos[i];
                        let fci = s.rope_sin[i];
                        let v0 = s.k[hs + i];
                        let v1 = s.k[hs + i + pair_offset];
                        s.k[hs + i] = v0 * fcr - v1 * fci;
                        s.k[hs + i + pair_offset] = v0 * fci + v1 * fcr;
                    }
                }
            } else {
                let mut i = 0;
                while i < q_dim {
                    let head_dim_idx = (i % head_size) / 2;
                    let fcr = s.rope_cos[head_dim_idx];
                    let fci = s.rope_sin[head_dim_idx];
                    let rotn = if i < kv_dim { 2 } else { 1 };
                    for v in 0..rotn {
                        let vec = if v == 0 { &mut s.q } else { &mut s.k };
                        let v0 = vec[i];
                        let v1 = vec[i + 1];
                        vec[i] = v0 * fcr - v1 * fci;
                        vec[i + 1] = v0 * fci + v1 * fcr;
                    }
                    i += 2;
                }
            }

            if p.is_gemma3 {
                scale_slice_inplace(&mut s.q[..q_dim], s.attn_scale);
            }

            let layer_row_base = l * p.seq_len;
            let row_index = layer_row_base + pos;
            let row_elem_offset = row_index * kv_dim;
            let q8_block_scales = env_flag("GGUF_Q8_BLOCK_SCALES");
            let n_blocks_per_row = kv_dim / Q4_BLOCK_SIZE;

            match s.kv_cache_format {
                KvCacheFormat::Q8 => {
                    if q8_block_scales {
                        let scale_base = row_index * n_blocks_per_row;
                        for b in 0..n_blocks_per_row {
                            let src_start = b * Q4_BLOCK_SIZE;
                            let src_end = src_start + Q4_BLOCK_SIZE;
                            let dst_start = row_elem_offset + src_start;
                            let dst_end = dst_start + Q4_BLOCK_SIZE;
                            quantize_row_q8(
                                &s.k[src_start..src_end],
                                &mut s.key_cache_q8[dst_start..dst_end],
                                &mut s.key_cache_scale[scale_base + b],
                            );
                            quantize_row_q8(
                                &s.v[src_start..src_end],
                                &mut s.value_cache_q8[dst_start..dst_end],
                                &mut s.value_cache_scale[scale_base + b],
                            );
                        }
                    } else {
                        quantize_row_q8(
                            &s.k[..kv_dim],
                            &mut s.key_cache_q8[row_elem_offset..row_elem_offset + kv_dim],
                            &mut s.key_cache_scale[row_index],
                        );
                        quantize_row_q8(
                            &s.v[..kv_dim],
                            &mut s.value_cache_q8[row_elem_offset..row_elem_offset + kv_dim],
                            &mut s.value_cache_scale[row_index],
                        );
                    }
                }
                KvCacheFormat::Turbo => {
                    let sign_table_slice = s.turbo_sign_table.as_slice();
                    let sign_bytes = turboquant_sign_bytes_per_head(head_size);
                    let turbo_scratch0 = &mut s.turbo_scratch0[..head_size];
                    let turbo_scratch1 = &mut s.turbo_scratch1[..head_size];
                    for kv_head in 0..p.n_kv_heads {
                        let head_start = kv_head * head_size;
                        let head_end = head_start + head_size;
                        let aux_idx = turboquant_aux_index(row_index, kv_head, p.n_kv_heads);
                        let signs = TurboSignRef::from_table(
                            sign_table_slice,
                            l,
                            kv_head,
                            p.n_layers,
                            p.n_kv_heads,
                            sign_bytes,
                        );
                        quantize_turboquant_head(
                            &s.k[head_start..head_end],
                            &signs,
                            turbo_scratch0,
                            turbo_scratch1,
                            TurboquantHeadWrite {
                                base: &mut s.key_cache_turbo_base,
                                sign: &mut s.key_cache_turbo_sign,
                                elem_offset: row_elem_offset + head_start,
                                scale_out: &mut s.key_cache_scale[aux_idx],
                                residual_norm_out: &mut s.key_cache_residual_norm[aux_idx],
                            },
                        );
                        quantize_turboquant_head(
                            &s.v[head_start..head_end],
                            &signs,
                            turbo_scratch0,
                            turbo_scratch1,
                            TurboquantHeadWrite {
                                base: &mut s.value_cache_turbo_base,
                                sign: &mut s.value_cache_turbo_sign,
                                elem_offset: row_elem_offset + head_start,
                                scale_out: &mut s.value_cache_scale[aux_idx],
                                residual_norm_out: &mut s.value_cache_residual_norm[aux_idx],
                            },
                        );
                    }
                }
            }

            let attn_scale_score = s.attn_scale;
            let apply_attn_scale = !p.is_gemma3;
            let q_all = &s.q[..q_dim];
            let kv_format = s.kv_cache_format;
            let key_cache_q8 = &s.key_cache_q8;
            let value_cache_q8 = &s.value_cache_q8;
            let key_cache_turbo_base = &s.key_cache_turbo_base;
            let value_cache_turbo_base = &s.value_cache_turbo_base;
            let key_cache_turbo_sign = &s.key_cache_turbo_sign;
            let value_cache_turbo_sign = &s.value_cache_turbo_sign;
            let turbo_sign_table = &s.turbo_sign_table;
            let turbo_sign_bytes = turboquant_sign_bytes_per_head(head_size);
            let key_scales = &s.key_cache_scale;
            let value_scales = &s.value_cache_scale;
            let key_residual_norms = &s.key_cache_residual_norm;
            let value_residual_norms = &s.value_cache_residual_norm;
            // Number of Q4 scale blocks per full kv_dim row; head_size/Q4_BLOCK_SIZE per KV head.
            let blocks_per_head = head_size / Q4_BLOCK_SIZE;
            let (att_all, xb_all) = (&mut s.att[..p.n_heads * p.seq_len], &mut s.xb[..q_dim]);
            let fuse_qwen35_online_attn = p.online_attn_fusion;

            if p.n_heads >= par_attn_min_heads() {
                att_all
                    .par_chunks_mut(p.seq_len)
                    .zip(xb_all.par_chunks_mut(head_size))
                    .enumerate()
                    .for_each_init(
                        || {
                            (
                                vec![0.0f32; head_size],
                                vec![0.0f32; head_size],
                                vec![0.0f32; head_size],
                            )
                        },
                        |(turbo_query_rotated, turbo_query_residual_proj, turbo_residual_accum),
                         (h, (att_head_full, xb_head))| {
                            let hs = h * head_size;
                            let q_head = &q_all[hs..hs + head_size];
                            let kv_head = h / kv_mul;
                            let kv_head_offset = kv_head * head_size;

                            let att_head = &mut att_head_full[..=pos];
                            // Scale slice for this KV head: blocks_per_head blocks starting at
                            // the head's block offset within the row.
                            let head_block_off = kv_head * blocks_per_head;
                            let key_head_scales_ptr =
                                unsafe { key_scales.as_ptr().add(head_block_off) };
                            let value_head_scales_ptr =
                                unsafe { value_scales.as_ptr().add(head_block_off) };
                            let turbo_signs = if kv_format == KvCacheFormat::Turbo {
                                let s = TurboSignRef::from_table(
                                    turbo_sign_table,
                                    l,
                                    kv_head,
                                    p.n_layers,
                                    p.n_kv_heads,
                                    turbo_sign_bytes,
                                );
                                turboquant_prepare_query(
                                    q_head,
                                    &s,
                                    turbo_query_rotated,
                                    turbo_query_residual_proj,
                                );
                                Some(s)
                            } else {
                                None
                            };
                            if fuse_qwen35_online_attn {
                                xb_head.fill(0.0);
                                if kv_format == KvCacheFormat::Turbo {
                                    turboquant_reset_residual_accum(
                                        turbo_residual_accum,
                                        head_size,
                                    );
                                }
                                let mut max_score = f32::NEG_INFINITY;
                                let mut score_sum = 0.0f32;
                                for t in 0..=pos {
                                    let t_row = layer_row_base + t;
                                    let row_offset = t_row * kv_dim + kv_head_offset;
                                    let mut score = match kv_format {
                                        KvCacheFormat::Q8 => {
                                            if q8_block_scales {
                                                let sb = t_row * n_blocks_per_row;
                                                unsafe {
                                                    dot_q8_row_blocks_ptr(
                                                        q_head,
                                                        key_cache_q8,
                                                        row_offset,
                                                        key_head_scales_ptr.add(sb),
                                                        blocks_per_head,
                                                    )
                                                }
                                            } else {
                                                dot_q8_row(
                                                    q_head,
                                                    key_cache_q8,
                                                    row_offset,
                                                    key_scales[t_row],
                                                )
                                            }
                                        }
                                        KvCacheFormat::Turbo => {
                                            let aux_idx =
                                                turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                            dot_turboquant_head(
                                                &turbo_query_rotated[..head_size],
                                                &turbo_query_residual_proj[..head_size],
                                                TurboquantHeadRead {
                                                    base: key_cache_turbo_base,
                                                    sign: key_cache_turbo_sign,
                                                    elem_offset: row_offset,
                                                    scale: key_scales[aux_idx],
                                                    residual_norm: key_residual_norms[aux_idx],
                                                },
                                            )
                                        }
                                    };
                                    if apply_attn_scale {
                                        score *= attn_scale_score;
                                    }

                                    if score > max_score {
                                        if score_sum > 0.0 {
                                            let rescale = (max_score - score).exp();
                                            scale_slice_inplace(xb_head, rescale);
                                            if kv_format == KvCacheFormat::Turbo {
                                                scale_slice_inplace(
                                                    &mut turbo_residual_accum[..head_size],
                                                    rescale,
                                                );
                                            }
                                            score_sum *= rescale;
                                        }
                                        max_score = score;
                                    }

                                    let weight = (score - max_score).exp();
                                    score_sum += weight;
                                    match kv_format {
                                        KvCacheFormat::Q8 => {
                                            if q8_block_scales {
                                                let sb = t_row * n_blocks_per_row;
                                                unsafe {
                                                    axpy_q8_row_blocks_ptr(
                                                        xb_head,
                                                        weight,
                                                        value_cache_q8,
                                                        row_offset,
                                                        value_head_scales_ptr.add(sb),
                                                        blocks_per_head,
                                                    );
                                                }
                                            } else {
                                                axpy_q8_row(
                                                    xb_head,
                                                    weight,
                                                    value_cache_q8,
                                                    row_offset,
                                                    value_scales[t_row],
                                                );
                                            }
                                        }
                                        KvCacheFormat::Turbo => {
                                            let aux_idx =
                                                turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                            axpy_turboquant_head(
                                                xb_head,
                                                &mut turbo_residual_accum[..head_size],
                                                weight,
                                                TurboquantHeadRead {
                                                    base: value_cache_turbo_base,
                                                    sign: value_cache_turbo_sign,
                                                    elem_offset: row_offset,
                                                    scale: value_scales[aux_idx],
                                                    residual_norm: value_residual_norms[aux_idx],
                                                },
                                            );
                                        }
                                    }
                                }
                                if score_sum > 0.0 {
                                    scale_slice_inplace(xb_head, 1.0 / score_sum);
                                    if kv_format == KvCacheFormat::Turbo {
                                        scale_slice_inplace(
                                            &mut turbo_residual_accum[..head_size],
                                            1.0 / score_sum,
                                        );
                                    }
                                }
                                if let Some(ref s) = turbo_signs {
                                    finalize_turboquant_value_head(
                                        xb_head,
                                        &mut turbo_residual_accum[..head_size],
                                        s,
                                    );
                                }
                            } else {
                                for (t, slot) in att_head.iter_mut().enumerate() {
                                    let t_row = layer_row_base + t;
                                    let row_offset = t_row * kv_dim + kv_head_offset;
                                    let mut score = match kv_format {
                                        KvCacheFormat::Q8 => {
                                            if q8_block_scales {
                                                let sb = t_row * n_blocks_per_row;
                                                unsafe {
                                                    dot_q8_row_blocks_ptr(
                                                        q_head,
                                                        key_cache_q8,
                                                        row_offset,
                                                        key_head_scales_ptr.add(sb),
                                                        blocks_per_head,
                                                    )
                                                }
                                            } else {
                                                dot_q8_row(
                                                    q_head,
                                                    key_cache_q8,
                                                    row_offset,
                                                    key_scales[t_row],
                                                )
                                            }
                                        }
                                        KvCacheFormat::Turbo => {
                                            let aux_idx =
                                                turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                            dot_turboquant_head(
                                                &turbo_query_rotated[..head_size],
                                                &turbo_query_residual_proj[..head_size],
                                                TurboquantHeadRead {
                                                    base: key_cache_turbo_base,
                                                    sign: key_cache_turbo_sign,
                                                    elem_offset: row_offset,
                                                    scale: key_scales[aux_idx],
                                                    residual_norm: key_residual_norms[aux_idx],
                                                },
                                            )
                                        }
                                    };
                                    if apply_attn_scale {
                                        score *= attn_scale_score;
                                    }
                                    *slot = score;
                                }

                                softmax(att_head, pos + 1);

                                xb_head.fill(0.0);
                                if kv_format == KvCacheFormat::Turbo {
                                    turboquant_reset_residual_accum(
                                        turbo_residual_accum,
                                        head_size,
                                    );
                                }
                                for (t, &a) in att_head.iter().enumerate() {
                                    let t_row = layer_row_base + t;
                                    let row_offset = t_row * kv_dim + kv_head_offset;
                                    match kv_format {
                                        KvCacheFormat::Q8 => {
                                            if q8_block_scales {
                                                let sb = t_row * n_blocks_per_row;
                                                unsafe {
                                                    axpy_q8_row_blocks_ptr(
                                                        xb_head,
                                                        a,
                                                        value_cache_q8,
                                                        row_offset,
                                                        value_head_scales_ptr.add(sb),
                                                        blocks_per_head,
                                                    );
                                                }
                                            } else {
                                                axpy_q8_row(
                                                    xb_head,
                                                    a,
                                                    value_cache_q8,
                                                    row_offset,
                                                    value_scales[t_row],
                                                );
                                            }
                                        }
                                        KvCacheFormat::Turbo => {
                                            let aux_idx =
                                                turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                            axpy_turboquant_head(
                                                xb_head,
                                                &mut turbo_residual_accum[..head_size],
                                                a,
                                                TurboquantHeadRead {
                                                    base: value_cache_turbo_base,
                                                    sign: value_cache_turbo_sign,
                                                    elem_offset: row_offset,
                                                    scale: value_scales[aux_idx],
                                                    residual_norm: value_residual_norms[aux_idx],
                                                },
                                            );
                                        }
                                    }
                                }
                                if let Some(ref s) = turbo_signs {
                                    finalize_turboquant_value_head(
                                        xb_head,
                                        &mut turbo_residual_accum[..head_size],
                                        s,
                                    );
                                }
                            }
                        },
                    );
            } else {
                let turbo_query_rotated = &mut s.turbo_scratch0[..head_size];
                let turbo_query_residual_proj = &mut s.turbo_scratch1[..head_size];
                let turbo_residual_accum = &mut s.turbo_scratch2[..head_size];
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    let q_head = &q_all[hs..hs + head_size];
                    let kv_head = h / kv_mul;
                    let kv_head_offset = kv_head * head_size;
                    let head_block_off = kv_head * blocks_per_head;
                    let key_head_scales_ptr = unsafe { key_scales.as_ptr().add(head_block_off) };
                    let value_head_scales_ptr =
                        unsafe { value_scales.as_ptr().add(head_block_off) };
                    let att_head_full = &mut att_all[h * p.seq_len..(h + 1) * p.seq_len];
                    let att_head = &mut att_head_full[..=pos];
                    let xb_head = &mut xb_all[hs..hs + head_size];
                    let turbo_signs = if kv_format == KvCacheFormat::Turbo {
                        let s = TurboSignRef::from_table(
                            turbo_sign_table,
                            l,
                            kv_head,
                            p.n_layers,
                            p.n_kv_heads,
                            turbo_sign_bytes,
                        );
                        turboquant_prepare_query(
                            q_head,
                            &s,
                            turbo_query_rotated,
                            turbo_query_residual_proj,
                        );
                        Some(s)
                    } else {
                        None
                    };
                    if fuse_qwen35_online_attn {
                        xb_head.fill(0.0);
                        if kv_format == KvCacheFormat::Turbo {
                            turboquant_reset_residual_accum(turbo_residual_accum, head_size);
                        }
                        let mut max_score = f32::NEG_INFINITY;
                        let mut score_sum = 0.0f32;
                        for t in 0..=pos {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            let mut score = match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            dot_q8_row_blocks_ptr(
                                                q_head,
                                                key_cache_q8,
                                                row_offset,
                                                key_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            )
                                        }
                                    } else {
                                        dot_q8_row(
                                            q_head,
                                            key_cache_q8,
                                            row_offset,
                                            key_scales[t_row],
                                        )
                                    }
                                }
                                KvCacheFormat::Turbo => {
                                    let aux_idx =
                                        turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                    dot_turboquant_head(
                                        &turbo_query_rotated[..head_size],
                                        &turbo_query_residual_proj[..head_size],
                                        TurboquantHeadRead {
                                            base: key_cache_turbo_base,
                                            sign: key_cache_turbo_sign,
                                            elem_offset: row_offset,
                                            scale: key_scales[aux_idx],
                                            residual_norm: key_residual_norms[aux_idx],
                                        },
                                    )
                                }
                            };
                            if apply_attn_scale {
                                score *= attn_scale_score;
                            }

                            if score > max_score {
                                if score_sum > 0.0 {
                                    let rescale = (max_score - score).exp();
                                    scale_slice_inplace(xb_head, rescale);
                                    if kv_format == KvCacheFormat::Turbo {
                                        scale_slice_inplace(turbo_residual_accum, rescale);
                                    }
                                    score_sum *= rescale;
                                }
                                max_score = score;
                            }

                            let weight = (score - max_score).exp();
                            score_sum += weight;
                            match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            axpy_q8_row_blocks_ptr(
                                                xb_head,
                                                weight,
                                                value_cache_q8,
                                                row_offset,
                                                value_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            );
                                        }
                                    } else {
                                        axpy_q8_row(
                                            xb_head,
                                            weight,
                                            value_cache_q8,
                                            row_offset,
                                            value_scales[t_row],
                                        );
                                    }
                                }
                                KvCacheFormat::Turbo => {
                                    let aux_idx =
                                        turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                    axpy_turboquant_head(
                                        xb_head,
                                        turbo_residual_accum,
                                        weight,
                                        TurboquantHeadRead {
                                            base: value_cache_turbo_base,
                                            sign: value_cache_turbo_sign,
                                            elem_offset: row_offset,
                                            scale: value_scales[aux_idx],
                                            residual_norm: value_residual_norms[aux_idx],
                                        },
                                    );
                                }
                            }
                        }
                        if score_sum > 0.0 {
                            scale_slice_inplace(xb_head, 1.0 / score_sum);
                            if kv_format == KvCacheFormat::Turbo {
                                scale_slice_inplace(turbo_residual_accum, 1.0 / score_sum);
                            }
                        }
                        if let Some(ref s) = turbo_signs {
                            finalize_turboquant_value_head(xb_head, turbo_residual_accum, s);
                        }
                    } else {
                        for (t, slot) in att_head.iter_mut().enumerate() {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            let mut score = match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            dot_q8_row_blocks_ptr(
                                                q_head,
                                                key_cache_q8,
                                                row_offset,
                                                key_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            )
                                        }
                                    } else {
                                        dot_q8_row(
                                            q_head,
                                            key_cache_q8,
                                            row_offset,
                                            key_scales[t_row],
                                        )
                                    }
                                }
                                KvCacheFormat::Turbo => {
                                    let aux_idx =
                                        turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                    dot_turboquant_head(
                                        &turbo_query_rotated[..head_size],
                                        &turbo_query_residual_proj[..head_size],
                                        TurboquantHeadRead {
                                            base: key_cache_turbo_base,
                                            sign: key_cache_turbo_sign,
                                            elem_offset: row_offset,
                                            scale: key_scales[aux_idx],
                                            residual_norm: key_residual_norms[aux_idx],
                                        },
                                    )
                                }
                            };
                            if apply_attn_scale {
                                score *= attn_scale_score;
                            }
                            *slot = score;
                        }

                        softmax(att_head, pos + 1);

                        xb_head.fill(0.0);
                        if kv_format == KvCacheFormat::Turbo {
                            turboquant_reset_residual_accum(turbo_residual_accum, head_size);
                        }
                        for (t, &a) in att_head.iter().enumerate() {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            axpy_q8_row_blocks_ptr(
                                                xb_head,
                                                a,
                                                value_cache_q8,
                                                row_offset,
                                                value_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            );
                                        }
                                    } else {
                                        axpy_q8_row(
                                            xb_head,
                                            a,
                                            value_cache_q8,
                                            row_offset,
                                            value_scales[t_row],
                                        );
                                    }
                                }
                                KvCacheFormat::Turbo => {
                                    let aux_idx =
                                        turboquant_aux_index(t_row, kv_head, p.n_kv_heads);
                                    axpy_turboquant_head(
                                        xb_head,
                                        turbo_residual_accum,
                                        a,
                                        TurboquantHeadRead {
                                            base: value_cache_turbo_base,
                                            sign: value_cache_turbo_sign,
                                            elem_offset: row_offset,
                                            scale: value_scales[aux_idx],
                                            residual_norm: value_residual_norms[aux_idx],
                                        },
                                    );
                                }
                            }
                        }
                        if let Some(ref s) = turbo_signs {
                            finalize_turboquant_value_head(xb_head, turbo_residual_accum, s);
                        }
                    }
                }
            }

            if qwen3next_packed_q_gate {
                for h in 0..p.n_heads {
                    let src_base = h * 2 * head_size + head_size;
                    let dst_base = h * head_size;
                    sigmoid_mul_inplace(
                        &mut s.xb[dst_base..dst_base + head_size],
                        &s.hb[src_base..src_base + head_size],
                    );
                }
                sanitize_finite_inplace(&mut s.xb[..q_dim]);
            }
            matmul_quantized(&mut s.xb2[..dim], &s.xb[..q_dim], &w.wo[l], mapped)?;
            sanitize_finite_inplace(&mut s.xb2[..dim]);
            prof_end(&PROF_ATTN_NS, attn_prof);
        }

        if do_layer_debug {
            eprintln!(
                "[LAYERDBG pos={pos} l={l}] post_attn_norm={:.4} x_norm={:.4}",
                l2_norm(&s.xb2[..dim]),
                l2_norm(&s.x[..dim]),
            );
        }

        if p.is_bert_family {
            // Post-norm: residual first, then LayerNorm in-place on x.
            accum(&mut s.x[..dim], &s.xb2[..dim], dim);
            layernorm_inplace(
                &mut s.x[..dim],
                &w.attn_post_norm[l * dim..(l + 1) * dim],
                &w.attn_post_norm_bias[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            if p.is_gemma3 && !w.attn_post_norm.is_empty() {
                rmsnorm_inplace(
                    &mut s.xb2[..dim],
                    &w.attn_post_norm[l * dim..(l + 1) * dim],
                    dim,
                    eps,
                );
            }
            accum(&mut s.x[..dim], &s.xb2[..dim], dim);
        }

        if p.is_bert_family {
            // Post-norm architecture: no pre-FFN norm — feed x directly into FFN.
            s.xb[..dim].copy_from_slice(&s.x[..dim]);
        } else if p.is_gemma3 {
            rmsnorm_gemma(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        if p.is_qwen3moe || (p.is_qwen3next && p.n_experts > 0) {
            let moe_prof = prof_start();
            let disable_routed = p.is_qwen35 && env_flag("GGUF_QWEN35_DISABLE_ROUTED_EXPERTS");
            let disable_shared = p.is_qwen35 && env_flag("GGUF_QWEN35_DISABLE_SHARED_EXPERT");
            let force_serial_routed =
                p.is_qwen3next && env_flag("GGUF_QWEN3NEXT_SERIAL_ROUTED_EXPERTS");
            s.xb2[..dim].copy_from_slice(&s.xb[..dim]);
            matmul_quantized(
                &mut s.moe_logits[..p.n_experts],
                &s.xb2[..dim],
                &w.moe_gate_inp[l],
                mapped,
            )?;
            let n_selected = select_topk_softmax(
                &s.moe_logits[..p.n_experts],
                p.n_experts_used,
                p.moe_n_group,
                p.moe_topk_group,
                p.moe_norm_topk_prob,
                p.moe_routed_scaling_factor,
                &mut s.moe_scores,
                &mut s.moe_selected_group,
                &mut s.moe_group_scores,
                &mut s.moe_group_rank,
                &mut s.moe_topk_indices,
                &mut s.moe_topk_weights,
            );
            s.xb[..dim].fill(0.0);

            if !disable_routed {
                let mut routed_selected = Vec::with_capacity(n_selected);
                for j in 0..n_selected {
                    let route_weight = s.moe_topk_weights[j];
                    if route_weight != 0.0 {
                        routed_selected.push((s.moe_topk_indices[j], route_weight));
                    }
                }

                if let Some(executor) = moe_executor.as_deref_mut() {
                    executor.compute_selected_experts(
                        l,
                        &s.xb2[..dim],
                        &routed_selected,
                        &mut s.xb[..dim],
                        p,
                        w,
                        mapped,
                    )?;
                } else {
                    compute_local_selected_experts(
                        l,
                        &s.xb2[..dim],
                        &routed_selected,
                        &mut s.xb[..dim],
                        p,
                        w,
                        mapped,
                        !force_serial_routed,
                    )?;
                }
            }

            if p.is_qwen3next && !disable_shared && !w.moe_shared_gate_inp.is_empty() {
                let shared_hidden = if p.shared_expert_hidden_dim > 0 {
                    p.shared_expert_hidden_dim
                } else {
                    p.expert_hidden_dim
                };
                let shared_gate = &w.moe_shared_gate_inp[l * dim..(l + 1) * dim];
                let gate_logit = dot_f32_simd(&s.xb2[..dim], shared_gate);
                let gate = 1.0 / (1.0 + (-gate_logit).exp());

                matmul_quantized(&mut s.hb[..shared_hidden], &s.xb2[..dim], &w.w1[l], mapped)?;
                matmul_quantized(&mut s.hb2[..shared_hidden], &s.xb2[..dim], &w.w3[l], mapped)?;
                silu_and_mul_inplace(&mut s.hb[..shared_hidden], &s.hb2[..shared_hidden]);
                matmul_quantized(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..shared_hidden],
                    &w.w2[l],
                    mapped,
                )?;
                crate::engine::kernels::axpy_inplace(&mut s.xb[..dim], gate, &s.moe_tmp[..dim]);
            }
            prof_end(&PROF_MOE_NS, moe_prof);
        } else {
            let ffn_prof = prof_start();
            matmul_quantized(&mut s.hb[..hidden_dim], &s.xb[..dim], &w.w1[l], mapped)?;
            matmul_quantized(&mut s.hb2[..hidden_dim], &s.xb[..dim], &w.w3[l], mapped)?;

            if p.is_gemma3 {
                for i in 0..hidden_dim {
                    let x = s.hb[i];
                    let gelu =
                        0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh());
                    s.hb[i] = gelu * s.hb2[i];
                }
            } else {
                silu_and_mul_inplace(&mut s.hb[..hidden_dim], &s.hb2[..hidden_dim]);
            }

            matmul_quantized(&mut s.xb[..dim], &s.hb[..hidden_dim], &w.w2[l], mapped)?;
            prof_end(&PROF_FFN_NS, ffn_prof);
        }

        if p.is_bert_family {
            // Post-norm: residual first, then LayerNorm in-place on x.
            accum(&mut s.x[..dim], &s.xb[..dim], dim);
            layernorm_inplace(
                &mut s.x[..dim],
                &w.ffn_post_norm[l * dim..(l + 1) * dim],
                &w.ffn_post_norm_bias[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            if p.is_gemma3 && !w.ffn_post_norm.is_empty() {
                rmsnorm_inplace(
                    &mut s.xb[..dim],
                    &w.ffn_post_norm[l * dim..(l + 1) * dim],
                    dim,
                    eps,
                );
            }
            accum(&mut s.x[..dim], &s.xb[..dim], dim);
        }

        if do_layer_debug {
            eprintln!(
                "[LAYERDBG pos={pos} l={l}] post_ffn_norm={:.4} x_norm={:.4}",
                l2_norm(&s.xb[..dim]),
                l2_norm(&s.x[..dim]),
            );
        }

        if let Some(ds) = deepstack_embedding
            && l < p.n_deepstack_layers
        {
            let ds_off = l * dim;
            let ds_end = ds_off + dim;
            if ds_end > ds.len() {
                return Err(format!(
                    "deepstack embedding length mismatch at layer {l}: need {} values, have {}",
                    ds_end,
                    ds.len()
                ));
            }
            accum(&mut s.x[..dim], &ds[ds_off..ds_end], dim);
        }
    }

    if !compute_logits {
        return Ok(());
    }

    if !p.is_bert_family {
        rmsnorm_inplace(&mut s.x[..dim], &w.rms_final_weight[..dim], dim, eps);
    }
    sanitize_finite_inplace(&mut s.x[..dim]);

    if w.wcls_is_embed {
        matmul_f32_embeddings(
            &mut s.logits[..p.vocab_size],
            &s.x[..dim],
            &w.token_embedding_table,
            p.vocab_size,
            dim,
        );
    } else {
        matmul_quantized(&mut s.logits[..p.vocab_size], &s.x[..dim], &w.wcls, mapped)?;
    }
    sanitize_finite_inplace(&mut s.logits[..p.vocab_size]);

    if p.is_gemma3 && p.final_logit_softcapping > 0.0 {
        let cap = p.final_logit_softcapping;
        for i in 0..p.vocab_size {
            s.logits[i] = cap * (s.logits[i] / cap).tanh();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        TURBOQUANT_RESIDUAL_POST_SALT, TURBOQUANT_RESIDUAL_PRE_SALT, TURBOQUANT_ROTATE_POST_SALT,
        TURBOQUANT_ROTATE_PRE_SALT, TurboquantHeadRead, axpy_turboquant_head_scalar,
        dot_turboquant_head_scalar, get_q2_at, set_sign_bit, splitmix64, turboquant_fwht_inplace,
    };
    #[cfg(target_arch = "aarch64")]
    use super::{axpy_turboquant_head_neon, dot_turboquant_head_neon};

    #[inline]
    fn turboquant_seed(layer: usize, kv_head: usize, salt: u64) -> u64 {
        splitmix64(((layer as u64) << 32) ^ kv_head as u64 ^ salt)
    }

    #[inline]
    fn turboquant_sign(seed: u64, idx: usize) -> f32 {
        if (splitmix64(seed ^ idx as u64) & 1) == 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn turboquant_apply_signs(values: &mut [f32], seed: u64) {
        for (idx, value) in values.iter_mut().enumerate() {
            *value *= turboquant_sign(seed, idx);
        }
    }

    fn turboquant_transform_in_place(
        values: &mut [f32],
        layer: usize,
        kv_head: usize,
        residual: bool,
        inverse: bool,
    ) {
        let (pre_salt, post_salt) = if residual {
            (TURBOQUANT_RESIDUAL_PRE_SALT, TURBOQUANT_RESIDUAL_POST_SALT)
        } else {
            (TURBOQUANT_ROTATE_PRE_SALT, TURBOQUANT_ROTATE_POST_SALT)
        };
        let (first_salt, second_salt) = if inverse {
            (post_salt, pre_salt)
        } else {
            (pre_salt, post_salt)
        };
        turboquant_apply_signs(values, turboquant_seed(layer, kv_head, first_salt));
        turboquant_fwht_inplace(values);
        turboquant_apply_signs(values, turboquant_seed(layer, kv_head, second_salt));
    }

    fn set_q2_at(dst: &mut [u8], elem_idx: usize, code: u8) {
        let shift = (elem_idx & 3) * 2;
        let byte = &mut dst[elem_idx / 4];
        *byte = (*byte & !(0b11 << shift)) | ((code & 0b11) << shift);
    }

    #[test]
    fn turboquant_q2_pack_roundtrip() {
        let values = [0u8, 1, 2, 3, 3, 2, 1, 0, 2];
        let mut packed = vec![0u8; values.len().div_ceil(4)];
        for (idx, &value) in values.iter().enumerate() {
            set_q2_at(&mut packed, idx, value);
        }
        for (idx, &value) in values.iter().enumerate() {
            assert_eq!(get_q2_at(&packed, idx), value);
        }
    }

    #[test]
    fn turboquant_transform_is_self_inverse_for_power_of_two_heads() {
        let mut values = vec![0.25f32, -0.5, 1.0, -1.5, 0.75, 0.125, -0.25, 0.5];
        let original = values.clone();
        turboquant_transform_in_place(&mut values, 2, 3, false, false);
        turboquant_transform_in_place(&mut values, 2, 3, false, true);
        for (got, want) in values.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1e-5);
        }
    }

    #[test]
    fn turboquant_transform_is_self_inverse_for_non_power_of_two_heads() {
        let mut values = vec![0.25f32, -0.5, 1.0, -1.5, 0.75, 0.125];
        let original = values.clone();
        turboquant_transform_in_place(&mut values, 1, 7, true, false);
        turboquant_transform_in_place(&mut values, 1, 7, true, true);
        for (got, want) in values.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn turboquant_neon_matches_scalar() {
        let head_size = 128usize;
        let elem_offset = 128usize;
        let mut base = vec![0u8; (elem_offset + head_size).div_ceil(4)];
        let mut sign = vec![0u8; (elem_offset + head_size).div_ceil(8)];
        let q_rotated = (0..head_size)
            .map(|i| ((i as f32 * 0.17).sin() * 0.75) + 0.1)
            .collect::<Vec<_>>();
        let q_residual = (0..head_size)
            .map(|i| ((i as f32 * 0.11).cos() * 0.5) - 0.2)
            .collect::<Vec<_>>();

        for i in 0..head_size {
            set_q2_at(&mut base, elem_offset + i, ((i * 13 + 7) & 0b11) as u8);
            set_sign_bit(&mut sign, elem_offset + i, (i % 3) != 0);
        }

        let cache = TurboquantHeadRead {
            base: &base,
            sign: &sign,
            elem_offset,
            scale: 0.37,
            residual_norm: 0.21,
        };

        let scalar_dot = dot_turboquant_head_scalar(&q_rotated, &q_residual, cache);
        let neon_dot = unsafe { dot_turboquant_head_neon(&q_rotated, &q_residual, cache) };
        assert!((scalar_dot - neon_dot).abs() < 1e-4);

        let mut scalar_base = vec![0.0f32; head_size];
        let mut scalar_residual = vec![0.0f32; head_size];
        let mut neon_base = vec![0.0f32; head_size];
        let mut neon_residual = vec![0.0f32; head_size];

        axpy_turboquant_head_scalar(&mut scalar_base, &mut scalar_residual, 0.63, cache);
        unsafe {
            axpy_turboquant_head_neon(&mut neon_base, &mut neon_residual, 0.63, cache);
        }

        for (got, want) in neon_base.iter().zip(scalar_base.iter()) {
            assert!((got - want).abs() < 1e-5);
        }
        for (got, want) in neon_residual.iter().zip(scalar_residual.iter()) {
            assert!((got - want).abs() < 1e-5);
        }
    }
}
