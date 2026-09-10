//! Gemma's SigLIP CPU arithmetic boundaries, following the local GGML graph.
//! Keep these separate from other encoders and from language-model kernels.
//! GGML source: llama.cpp 1744c6bd, ggml/src/ggml-cpu/. See the MIT notice in
//! the sibling gemma3_attention.rs module.

use crate::engine::io::{bf16_to_fp32, fp16_to_fp32, fp32_to_bf16, fp32_to_fp16};
use crate::engine::types::{GGML_TYPE_BF16, QuantizedTensor};
#[cfg(not(target_os = "macos"))]
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use std::sync::OnceLock;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_dgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
    fn vDSP_sve(input: *const f32, stride: isize, sum: *mut f32, count: usize);
    fn vDSP_vsadd(
        input: *const f32,
        stride: isize,
        value: *const f32,
        output: *mut f32,
        output_stride: isize,
        count: usize,
    );
    fn vDSP_measqv(input: *const f32, stride: isize, mean: *mut f32, count: usize);
}

#[derive(Default)]
pub(super) struct Bf16MatmulScratch {
    weights: Vec<f64>,
    input: Vec<f64>,
    #[cfg(target_os = "macos")]
    output: Vec<f64>,
}

/// The pinned GGML ARM BF16 CPU dot widens its products into a double sum.
/// F32 BFMMLA reductions perturb subsequent BF16 rounding in deep SigLIP layers.
/// Use bounded F64 tiles with the same BF16 operands, without changing weights.
pub(super) fn matmul_bf16(
    dst: &mut [f32],
    src: &[f32],
    weight: &QuantizedTensor,
    mapped: &[u8],
    tokens: usize,
    scratch: &mut Bf16MatmulScratch,
) -> Result<(), String> {
    let n = weight.cols;
    let d = weight.rows;
    let bytes = n
        .checked_mul(d)
        .and_then(|count| count.checked_mul(2))
        .ok_or("BF16 encoder matrix size overflow")?;
    let end = weight
        .data_offset
        .checked_add(bytes)
        .ok_or("BF16 encoder offset overflow")?;
    if weight.ttype.0 != GGML_TYPE_BF16
        || n == 0
        || d == 0
        || end > mapped.len()
        || n.checked_mul(tokens) != Some(src.len())
        || d.checked_mul(tokens) != Some(dst.len())
    {
        return Err("invalid BF16 encoder matrix buffers".into());
    }
    scratch.weights.resize(n * d, 0.0);
    for (out, bytes) in scratch
        .weights
        .iter_mut()
        .zip(mapped[weight.data_offset..end].as_chunks::<2>().0)
    {
        *out = f64::from(bf16_to_fp32(u16::from_le_bytes([bytes[0], bytes[1]])));
    }
    #[cfg(target_os = "macos")]
    let (ni, di) = (
        i32::try_from(n).map_err(|_| "BF16 matrix width exceeds c_int")?,
        i32::try_from(d).map_err(|_| "BF16 matrix height exceeds c_int")?,
    );
    for (input, output) in src.chunks(128 * n).zip(dst.chunks_mut(128 * d)) {
        let count = input.len() / n;
        scratch.input.resize(input.len(), 0.0);
        for (out, &x) in scratch.input.iter_mut().zip(input) {
            *out = f64::from(bf16_to_fp32(fp32_to_bf16(x)));
        }
        #[cfg(target_os = "macos")]
        {
            scratch.output.resize(count * d, 0.0);
            // Row-major input times transposed stored rows. The checks above
            // validate matrix extents and BLAS integer dimensions.
            unsafe {
                cblas_dgemm(
                    101,
                    111,
                    112,
                    count as i32,
                    di,
                    ni,
                    1.0,
                    scratch.input.as_ptr(),
                    ni,
                    scratch.weights.as_ptr(),
                    ni,
                    0.0,
                    scratch.output.as_mut_ptr(),
                    di,
                );
            }
            for (out, &value) in output.iter_mut().zip(&scratch.output) {
                *out = value as f32;
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = count;
            output
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(token, row)| {
                    let x = &scratch.input[token * n..(token + 1) * n];
                    for (out, w) in row.iter_mut().zip(scratch.weights.chunks_exact(n)) {
                        *out = x.iter().zip(w).map(|(&a, &b)| a * b).sum::<f64>() as f32;
                    }
                });
        }
    }
    Ok(())
}

pub(super) fn layer_norm_affine(dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32], eps: f32) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(w.len(), src.len());
    debug_assert_eq!(b.len(), src.len());
    let variance;
    #[cfg(target_os = "macos")]
    {
        let mut sum = 0.0;
        let mut mean_square = 0.0;
        // GGML's Accelerate norm uses these same reductions and stores the
        // centered row before computing its variance. All buffers have n elements.
        unsafe {
            vDSP_sve(src.as_ptr(), 1, &mut sum, src.len());
            let negative_mean = -(sum / src.len() as f32);
            vDSP_vsadd(
                src.as_ptr(),
                1,
                &negative_mean,
                dst.as_mut_ptr(),
                1,
                src.len(),
            );
            vDSP_measqv(dst.as_ptr(), 1, &mut mean_square, dst.len());
        }
        variance = mean_square;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let sum = src.iter().map(|&x| f64::from(x)).sum::<f64>() as f32;
        let mean = sum / src.len() as f32;
        let mut squared = 0.0f64;
        for (out, &x) in dst.iter_mut().zip(src) {
            *out = x - mean;
            squared += f64::from(*out * *out);
        }
        variance = (squared / src.len() as f64) as f32;
    }
    let scale = 1.0 / (variance + eps).sqrt();
    for ((out, &weight), &bias) in dst.iter_mut().zip(w).zip(b) {
        // ggml_norm -> ggml_mul -> ggml_add: do not fuse the affine step.
        *out = ((*out * scale) * weight) + bias;
    }
}

pub(super) fn rms_norm_weight(dst: &mut [f32], src: &[f32], weight: &[f32], eps: f32) {
    let sum = src.iter().map(|&x| f64::from(x * x)).sum::<f64>();
    let mean = (sum / src.len() as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for ((out, &x), &w) in dst.iter_mut().zip(src).zip(weight) {
        *out = x * scale * w;
    }
}

pub(super) fn gelu(value: f32) -> f32 {
    // GGML_GELU_FP16 uses a half-input/half-output table for the tanh GELU,
    // with full-precision tails. This is also used for F32 GGML tensors.
    if value <= -10.0 {
        return 0.0;
    }
    if value >= 10.0 {
        return value;
    }
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|bits| {
                let x = fp16_to_fp32(bits);
                let cubic = (0.044_715 * x).mul_add(x, 1.0);
                let y = 0.5 * x * (1.0 + (0.797_884_6 * x * cubic).tanh());
                fp32_to_fp16(y)
            })
            .collect()
    });
    fp16_to_fp32(table[usize::from(fp32_to_fp16(value))])
}

/// GGML's native ARM half dot uses four half-precision accumulators, then
/// widens for the horizontal reduction and the scalar tail. Other CPUs retain
/// a portable widened reduction; their native GGML SIMD reduction may differ.
pub(super) fn dot_f16(x: &[u16], y: &[u16]) -> f32 {
    assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        // SAFETY: the runtime feature check guards half arithmetic; the helper
        // only loads complete 32-element blocks from equally sized slices.
        return unsafe { dot_f16_neon(x, y) };
    }
    x.iter()
        .zip(y)
        .map(|(&a, &b)| f64::from(fp16_to_fp32(a) * fp16_to_fp32(b)))
        .sum::<f64>() as f32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn dot_f16_neon(x: &[u16], y: &[u16]) -> f32 {
    let blocks = x.len() / 32;
    let mut partial = 0.0f32;
    if blocks > 0 {
        // Rust's half-vector intrinsics are unstable. Fixed-register assembly
        // expresses the same four accumulators and reduction as simd-mappings.h.
        unsafe {
            std::arch::asm!(
                "movi v0.8h, #0", "movi v1.8h, #0",
                "movi v2.8h, #0", "movi v3.8h, #0",
                "2:",
                "ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{x}], #64",
                "ld1 {{v8.8h, v9.8h, v10.8h, v11.8h}}, [{y}], #64",
                "fmla v0.8h, v4.8h, v8.8h", "fmla v1.8h, v5.8h, v9.8h",
                "fmla v2.8h, v6.8h, v10.8h", "fmla v3.8h, v7.8h, v11.8h",
                "subs {blocks}, {blocks}, #1", "b.ne 2b",
                "fadd v0.8h, v0.8h, v2.8h", "fadd v1.8h, v1.8h, v3.8h",
                "fadd v0.8h, v0.8h, v1.8h",
                "fcvtl v1.4s, v0.4h", "fcvtl2 v2.4s, v0.8h",
                "fadd v1.4s, v1.4s, v2.4s",
                "faddp v1.4s, v1.4s, v1.4s", "faddp s1, v1.2s",
                "str s1, [{out}]",
                x = inout(reg) x.as_ptr() => _, y = inout(reg) y.as_ptr() => _,
                blocks = inout(reg) blocks => _, out = in(reg) &mut partial,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                options(nostack),
            );
        }
    }
    let mut sum = f64::from(partial);
    for (&a, &b) in x[blocks * 32..].iter().zip(&y[blocks * 32..]) {
        sum += f64::from(fp16_to_fp32(a) * fp16_to_fp32(b));
    }
    sum as f32
}
