//! Noncausal SigLIP attention with the pinned GGML CPU reduction order.
//! Source: llama.cpp 1744c6bd, ggml/src/ggml-cpu/{vec.cpp,vec.h} (MIT).

// The GGML-derived arithmetic in this module and gemma3_math.rs is distributed
// under the following notice:
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // AArch64 guarantees NEON. Only complete four-vector blocks are loaded.
        unsafe {
            let mut sums = [vdupq_n_f32(0.0); 4];
            let end = x.len() / 16 * 16;
            for i in (0..end).step_by(16) {
                for (j, sum) in sums.iter_mut().enumerate() {
                    *sum = vfmaq_f32(
                        *sum,
                        vld1q_f32(x.as_ptr().add(i + j * 4)),
                        vld1q_f32(y.as_ptr().add(i + j * 4)),
                    );
                }
            }
            let mut sum = vaddvq_f32(vaddq_f32(
                vaddq_f32(sums[0], sums[2]),
                vaddq_f32(sums[1], sums[3]),
            ));
            let vector_end = x.len() / 4 * 4;
            // Clang vectorizes complete tail products without contraction;
            // the final 1-3 scalar products use FMA in the pinned CPU build.
            for (&a, &b) in x[end..vector_end].iter().zip(&y[end..vector_end]) {
                sum += a * b;
            }
            for (&a, &b) in x[vector_end..].iter().zip(&y[vector_end..]) {
                sum = a.mul_add(b, sum);
            }
            sum
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter()
            .zip(y)
            .map(|(&x, &y)| f64::from(x * y))
            .sum::<f64>() as f32
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let constant = |bits| vdupq_n_f32(f32::from_bits(bits));
        let r = constant(0x4b400000);
        let z = vfmaq_f32(r, x, constant(0x3fb8aa3b));
        let n = vsubq_f32(z, r);
        let b = vfmsq_f32(
            vfmsq_f32(x, n, constant(0x3f317200)),
            n,
            constant(0x35bfbe8e),
        );
        let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
        let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
        let c = vcagtq_f32(n, vdupq_n_f32(126.0));
        let u = vmulq_f32(b, b);
        let j = vfmaq_f32(
            vmulq_f32(constant(0x3f7ffff6), b),
            vfmaq_f32(
                vfmaq_f32(constant(0x3efffedb), constant(0x3e2aaf33), b),
                vfmaq_f32(constant(0x3d2b9f17), constant(0x3c072010), b),
                u,
            ),
            u,
        );
        if vaddvq_u64(vreinterpretq_u64_u32(c)) == 0 {
            return vfmaq_f32(k, j, k);
        }
        let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
        let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
        let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
        vbslq_f32(
            vcagtq_f32(n, vdupq_n_f32(192.0)),
            vmulq_f32(s1, s1),
            vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
        )
    }
}

fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    #[cfg(target_arch = "aarch64")]
    let mut start = 0;
    #[cfg(not(target_arch = "aarch64"))]
    let start = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // Process only complete vectors, then use expf for the tail like GGML.
        unsafe {
            while start + 4 <= values.len() {
                let ptr = values.as_mut_ptr().add(start);
                let v = exp4(vsubq_f32(vld1q_f32(ptr), vdupq_n_f32(maximum)));
                vst1q_f32(ptr, v);
                sum += f64::from(vaddvq_f32(v));
                start += 4;
            }
        }
    }
    for x in &mut values[start..] {
        *x = (*x - maximum).exp();
        sum += f64::from(*x);
    }
    let scale = (1.0 / sum) as f32;
    for x in values {
        *x *= scale;
    }
}

#[derive(Default)]
pub(super) struct AttentionScratch {
    transposed_values: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    scratch: &mut AttentionScratch,
) {
    let dim = heads * head_dim;
    assert_eq!(output.len(), tokens * dim);
    assert_eq!(q.len(), output.len());
    assert_eq!(k.len(), output.len());
    assert_eq!(v.len(), output.len());
    scratch.transposed_values.resize(v.len(), 0.0);
    scratch
        .transposed_values
        .par_chunks_mut(tokens)
        .enumerate()
        .for_each(|(channel, out)| {
            for (dst, row) in out.iter_mut().zip(v.chunks_exact(dim)) {
                *dst = row[channel];
            }
        });
    let values = &scratch.transposed_values;
    let scale = 1.0 / (head_dim as f32).sqrt();
    output.par_chunks_mut(dim).enumerate().for_each_init(
        || vec![0.0; tokens],
        |scores, (token, out)| {
            for head in 0..heads {
                let offset = head * head_dim;
                let query = &q[token * dim + offset..token * dim + offset + head_dim];
                for (score, key) in scores.iter_mut().zip(k.chunks_exact(dim)) {
                    *score = dot(query, &key[offset..offset + head_dim]) * scale;
                }
                softmax(scores);
                for (channel, dst) in out[offset..offset + head_dim].iter_mut().enumerate() {
                    let base = (offset + channel) * tokens;
                    *dst = dot(&values[base..base + tokens], scores);
                }
            }
        },
    );
}
