use crate::engine::io::{
    find_gguf_tensor, get_gguf_f32_array_from_map, get_gguf_float_from_map, get_gguf_int_from_map,
};
use crate::engine::kernels::{
    axpy_inplace, dequantize_tensor, dot_f32_simd, get_block_size, get_type_size,
};
use crate::engine::multimodal::injection::MediaEmbeddingSequence;
use crate::engine::types::{GGUFFile, Gguftensor, QuantizedTensor};
use crate::engine::vision::PreparedImageTensor;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};

use super::{
    EncoderAttentionScratch, FloatBatchMatmulScratch, encoder_self_attention, matmul_encoder_batch,
};

type PatchTokenGrid = (Vec<f32>, Vec<(usize, usize)>, usize, usize);

const MAX_VISION_MICROBATCH_IMAGES: usize = 4;
const MAX_VISION_MICROBATCH_PATCH_TOKENS: usize = 4096;

fn vision_microbatch_image_limit(tokens_per_image: usize) -> usize {
    if tokens_per_image == 0 {
        return 1;
    }
    (MAX_VISION_MICROBATCH_PATCH_TOKENS / tokens_per_image).clamp(1, MAX_VISION_MICROBATCH_IMAGES)
}

fn vision_rope_coefficients(head_dim: usize, positions: &[(usize, usize)]) -> Vec<(f32, f32)> {
    let half_head = head_dim / 2;
    let quarter_head = head_dim / 4;
    if half_head == 0 || quarter_head == 0 {
        return Vec::new();
    }

    let frequencies = (0..half_head)
        .map(|ic| {
            let local_ic = if ic < quarter_head {
                ic
            } else {
                ic - quarter_head
            };
            1.0 / 10_000.0f32.powf((2 * local_ic) as f32 / head_dim as f32)
        })
        .collect::<Vec<_>>();

    let mut coefficients = Vec::with_capacity(positions.len() * half_head);
    for &(py, px) in positions {
        for (ic, &frequency) in frequencies.iter().enumerate() {
            let position = if ic < quarter_head { py } else { px };
            let theta = position as f32 * frequency;
            coefficients.push((theta.cos(), theta.sin()));
        }
    }
    coefficients
}

fn apply_vision_rope_cached(
    q: &mut [f32],
    k: &mut [f32],
    dim: usize,
    head_count: usize,
    head_dim: usize,
    tokens_per_sequence: usize,
    coefficients: &[(f32, f32)],
) {
    let half_head = head_dim / 2;
    if half_head == 0 || tokens_per_sequence == 0 {
        return;
    }
    debug_assert_eq!(dim, head_count * head_dim);
    debug_assert_eq!(q.len(), k.len());
    debug_assert_eq!(q.len() % dim, 0);
    debug_assert_eq!(coefficients.len(), tokens_per_sequence * half_head);

    q.par_chunks_mut(dim)
        .zip(k.par_chunks_mut(dim))
        .enumerate()
        .for_each(|(token, (q_token, k_token))| {
            let local_token = token % tokens_per_sequence;
            let coefficient_row =
                &coefficients[local_token * half_head..(local_token + 1) * half_head];
            for head in 0..head_count {
                let head_offset = head * head_dim;
                for (ic, &(cos, sin)) in coefficient_row.iter().enumerate() {
                    let a = head_offset + ic;
                    let b = a + half_head;

                    let q0 = q_token[a];
                    let q1 = q_token[b];
                    q_token[a] = q0 * cos - q1 * sin;
                    q_token[b] = q0 * sin + q1 * cos;

                    let k0 = k_token[a];
                    let k1 = k_token[b];
                    k_token[a] = k0 * cos - k1 * sin;
                    k_token[b] = k0 * sin + k1 * cos;
                }
            }
        });
}

fn tensor_n_elements(tensor: &Gguftensor) -> usize {
    let mut n_elements = 1usize;
    for i in 0..tensor.n_dims as usize {
        n_elements = n_elements.saturating_mul(tensor.ne[i] as usize);
    }
    n_elements
}

fn load_tensor_float(
    gguf: &GGUFFile,
    name: &str,
    expected_elements: Option<usize>,
) -> Result<Vec<f32>, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);
    if let Some(expected) = expected_elements
        && n_elements != expected
    {
        return Err(format!(
            "tensor {name} has {n_elements} elements, expected {expected}"
        ));
    }

    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if block_size == 0 || type_size == 0 {
        return Err(format!(
            "unsupported tensor type {} for {name}",
            tensor.ttype.0
        ));
    }
    if !n_elements.is_multiple_of(block_size) {
        return Err(format!(
            "tensor {name} element count {n_elements} not divisible by block size {block_size}"
        ));
    }

    let src_size = (n_elements / block_size) * type_size;
    let mapped = gguf.mapped.as_slice();
    let end = tensor
        .data_offset
        .checked_add(src_size)
        .ok_or_else(|| format!("tensor {name} offset overflow"))?;
    if end > mapped.len() {
        return Err(format!("tensor {name} exceeds mapped bounds"));
    }
    gguf.ensure_range(tensor.data_offset, src_size)?;
    dequantize_tensor(
        &mapped[tensor.data_offset..tensor.data_offset + src_size],
        n_elements,
        tensor.ttype,
    )
}

fn load_tensor_quantized(
    gguf: &GGUFFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("shape overflow while loading {name}"))?;
    if n_elements != expected {
        return Err(format!(
            "tensor {name} shape mismatch: got {} elements, expected {} (rows={rows}, cols={cols})",
            n_elements, expected
        ));
    }
    Ok(QuantizedTensor {
        data_offset: tensor.data_offset,
        ttype: tensor.ttype,
        rows,
        cols,
    })
}

fn tensor_matrix_shape(tensor: &Gguftensor, name: &str) -> Result<(usize, usize), String> {
    if tensor.n_dims < 2 {
        return Err(format!(
            "tensor {name} has n_dims={}, expected at least 2 for matrix weight",
            tensor.n_dims
        ));
    }
    let cols = tensor.ne[0] as usize;
    let rows = tensor.ne[1] as usize;
    if rows == 0 || cols == 0 {
        return Err(format!(
            "tensor {name} has invalid matrix shape rows={rows} cols={cols}"
        ));
    }
    Ok((rows, cols))
}

#[inline]
fn layer_norm_affine(dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32], eps: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = src.len();
        let x_ptr = src.as_ptr();
        let w_ptr = w.as_ptr();
        let b_ptr = b.as_ptr();
        let y_ptr = dst.as_mut_ptr();

        let mut i = 0usize;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            acc0 = vaddq_f32(acc0, vld1q_f32(x_ptr.add(i)));
            acc1 = vaddq_f32(acc1, vld1q_f32(x_ptr.add(i + 4)));
            acc2 = vaddq_f32(acc2, vld1q_f32(x_ptr.add(i + 8)));
            acc3 = vaddq_f32(acc3, vld1q_f32(x_ptr.add(i + 12)));
            i += 16;
        }
        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= n {
            acc = vaddq_f32(acc, vld1q_f32(x_ptr.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while i < n {
            sum += *x_ptr.add(i);
            i += 1;
        }
        let mean = sum / n as f32;
        let meanv = vdupq_n_f32(mean);

        i = 0;
        acc0 = vdupq_n_f32(0.0);
        acc1 = vdupq_n_f32(0.0);
        acc2 = vdupq_n_f32(0.0);
        acc3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            let d0 = vsubq_f32(vld1q_f32(x_ptr.add(i)), meanv);
            let d1 = vsubq_f32(vld1q_f32(x_ptr.add(i + 4)), meanv);
            let d2 = vsubq_f32(vld1q_f32(x_ptr.add(i + 8)), meanv);
            let d3 = vsubq_f32(vld1q_f32(x_ptr.add(i + 12)), meanv);
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
            i += 16;
        }
        acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(x_ptr.add(i)), meanv);
            acc = vfmaq_f32(acc, d, d);
            i += 4;
        }
        let mut var = vaddvq_f32(acc);
        while i < n {
            let d = *x_ptr.add(i) - mean;
            var += d * d;
            i += 1;
        }
        var /= n as f32;
        let inv = 1.0f32 / (var + eps).sqrt();
        let invv = vdupq_n_f32(inv);

        i = 0;
        while i + 4 <= n {
            let xv = vld1q_f32(x_ptr.add(i));
            let wv = vld1q_f32(w_ptr.add(i));
            let bv = vld1q_f32(b_ptr.add(i));
            let norm = vmulq_f32(vsubq_f32(xv, meanv), invv);
            let out = vfmaq_f32(bv, norm, wv);
            vst1q_f32(y_ptr.add(i), out);
            i += 4;
        }
        while i < n {
            *y_ptr.add(i) = ((*x_ptr.add(i) - mean) * inv) * *w_ptr.add(i) + *b_ptr.add(i);
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        let mut mean = 0.0f32;
        for &v in src {
            mean += v;
        }
        mean /= src.len() as f32;

        let mut var = 0.0f32;
        for &v in src {
            let d = v - mean;
            var += d * d;
        }
        var /= src.len() as f32;
        let inv = 1.0f32 / (var + eps).sqrt();
        for i in 0..src.len() {
            dst[i] = ((src[i] - mean) * inv) * w[i] + b[i];
        }
    }
}

#[derive(Clone)]
struct VisionLayerWeights {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    attn_qkv_w: QuantizedTensor,
    attn_qkv_b: Vec<f32>,
    attn_out_w: QuantizedTensor,
    attn_out_b: Vec<f32>,
    ffn_up_w: QuantizedTensor,
    ffn_up_b: Vec<f32>,
    ffn_down_w: QuantizedTensor,
    ffn_down_b: Vec<f32>,
}

#[derive(Clone)]
struct DeepstackLayerWeights {
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    fc1_w: QuantizedTensor,
    fc1_b: Vec<f32>,
    fc2_w: QuantizedTensor,
    fc2_b: Vec<f32>,
}

pub(crate) struct Qwen3VlVisionEncoder {
    gguf: GGUFFile,
    dim: usize,
    head_count: usize,
    head_dim: usize,
    ff_dim: usize,
    n_layers: usize,
    eps: f32,
    patch_size: usize,
    base_image_size: usize,
    spatial_merge: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    patch_embd_w: Vec<f32>,
    patch_embd_b: Vec<f32>,
    position_embd: Vec<f32>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm0_w: QuantizedTensor,
    mm0_b: Vec<f32>,
    mm2_w: QuantizedTensor,
    mm2_b: Vec<f32>,
    layers: Vec<VisionLayerWeights>,
    deepstack_layers: Vec<DeepstackLayerWeights>,
    deepstack_by_layer: Vec<Option<usize>>,
}

impl Qwen3VlVisionEncoder {
    fn parse_rgb_triplet(
        kv_values: Option<&[f32]>,
        default: [f32; 3],
        key: &str,
    ) -> Result<[f32; 3], String> {
        let Some(values) = kv_values else {
            return Ok(default);
        };
        if values.len() < 3 {
            return Err(format!(
                "invalid {key} metadata: expected at least 3 values, got {}",
                values.len()
            ));
        }
        Ok([values[0], values[1], values[2]])
    }

    pub(crate) fn recommended_image_size(&self) -> usize {
        self.base_image_size
    }

    pub(crate) fn recommended_image_alignment(&self) -> usize {
        self.patch_size.saturating_mul(self.spatial_merge).max(1)
    }

    pub(crate) fn recommended_image_normalization(&self) -> ([f32; 3], [f32; 3]) {
        (self.image_mean, self.image_std)
    }

    pub(crate) fn new(
        gguf: GGUFFile,
        target_dim: usize,
        expected_deepstack_layers: usize,
    ) -> Result<Self, String> {
        let dim = get_gguf_int_from_map(&gguf.kv, "clip.vision.embedding_length", 0) as usize;
        let head_count =
            get_gguf_int_from_map(&gguf.kv, "clip.vision.attention.head_count", 0) as usize;
        let ff_dim = get_gguf_int_from_map(&gguf.kv, "clip.vision.feed_forward_length", 0) as usize;
        let n_layers = get_gguf_int_from_map(&gguf.kv, "clip.vision.block_count", 0) as usize;
        let eps =
            get_gguf_float_from_map(&gguf.kv, "clip.vision.attention.layer_norm_epsilon", 1e-6);
        let patch_size = get_gguf_int_from_map(&gguf.kv, "clip.vision.patch_size", 16) as usize;
        let base_image_size =
            get_gguf_int_from_map(&gguf.kv, "clip.vision.image_size", 768) as usize;
        let spatial_merge =
            get_gguf_int_from_map(&gguf.kv, "clip.vision.spatial_merge_size", 2) as usize;
        let image_mean = Self::parse_rgb_triplet(
            get_gguf_f32_array_from_map(&gguf.kv, "clip.vision.image_mean"),
            [0.48145466, 0.4578275, 0.40821073],
            "clip.vision.image_mean",
        )?;
        let image_std = Self::parse_rgb_triplet(
            get_gguf_f32_array_from_map(&gguf.kv, "clip.vision.image_std"),
            [0.26862954, 0.261_302_6, 0.2757771],
            "clip.vision.image_std",
        )?;

        if dim == 0 || head_count == 0 || ff_dim == 0 || n_layers == 0 || patch_size == 0 {
            return Err(
                "invalid qwen3vl mmproj metadata: one or more required clip.vision.* keys are missing/zero"
                    .to_string(),
            );
        }
        if !dim.is_multiple_of(head_count) {
            return Err(format!(
                "invalid qwen3vl mmproj metadata: dim {} is not divisible by head_count {}",
                dim, head_count
            ));
        }
        let head_dim = dim / head_count;

        let patch_kernel_elems = patch_size
            .checked_mul(patch_size)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "patch kernel element count overflow".to_string())?;
        let base_patch_grid = base_image_size
            .checked_div(patch_size)
            .ok_or_else(|| "invalid base image/patch ratio".to_string())?;
        let base_pos_tokens = base_patch_grid
            .checked_mul(base_patch_grid)
            .ok_or_else(|| "position token count overflow".to_string())?;

        let mut patch_embd_w =
            load_tensor_float(&gguf, "v.patch_embd.weight", Some(patch_kernel_elems))?;
        let patch_embd_w1 =
            load_tensor_float(&gguf, "v.patch_embd.weight.1", Some(patch_kernel_elems))?;
        for (dst, src) in patch_embd_w.iter_mut().zip(patch_embd_w1.iter()) {
            *dst += *src;
        }
        let patch_embd_b = load_tensor_float(&gguf, "v.patch_embd.bias", Some(dim))?;
        let position_embd =
            load_tensor_float(&gguf, "v.position_embd.weight", Some(base_pos_tokens * dim))?;
        let post_ln_w = load_tensor_float(&gguf, "v.post_ln.weight", Some(dim))?;
        let post_ln_b = load_tensor_float(&gguf, "v.post_ln.bias", Some(dim))?;

        let mm0_w = load_tensor_quantized(
            &gguf,
            "mm.0.weight",
            dim * spatial_merge * spatial_merge,
            dim * spatial_merge * spatial_merge,
        )?;
        let mm0_b = load_tensor_float(
            &gguf,
            "mm.0.bias",
            Some(dim * spatial_merge * spatial_merge),
        )?;
        let mm2_w = load_tensor_quantized(
            &gguf,
            "mm.2.weight",
            target_dim,
            dim * spatial_merge * spatial_merge,
        )?;
        let mm2_b = load_tensor_float(&gguf, "mm.2.bias", Some(target_dim))?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let prefix = format!("v.blk.{l}");
            layers.push(VisionLayerWeights {
                ln1_w: load_tensor_float(&gguf, &format!("{prefix}.ln1.weight"), Some(dim))?,
                ln1_b: load_tensor_float(&gguf, &format!("{prefix}.ln1.bias"), Some(dim))?,
                ln2_w: load_tensor_float(&gguf, &format!("{prefix}.ln2.weight"), Some(dim))?,
                ln2_b: load_tensor_float(&gguf, &format!("{prefix}.ln2.bias"), Some(dim))?,
                attn_qkv_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.attn_qkv.weight"),
                    dim * 3,
                    dim,
                )?,
                attn_qkv_b: load_tensor_float(
                    &gguf,
                    &format!("{prefix}.attn_qkv.bias"),
                    Some(dim * 3),
                )?,
                attn_out_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.attn_out.weight"),
                    dim,
                    dim,
                )?,
                attn_out_b: load_tensor_float(
                    &gguf,
                    &format!("{prefix}.attn_out.bias"),
                    Some(dim),
                )?,
                ffn_up_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.ffn_up.weight"),
                    ff_dim,
                    dim,
                )?,
                ffn_up_b: load_tensor_float(&gguf, &format!("{prefix}.ffn_up.bias"), Some(ff_dim))?,
                ffn_down_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.ffn_down.weight"),
                    dim,
                    ff_dim,
                )?,
                ffn_down_b: load_tensor_float(
                    &gguf,
                    &format!("{prefix}.ffn_down.bias"),
                    Some(dim),
                )?,
            });
        }

        let merged_dim = dim
            .checked_mul(spatial_merge)
            .and_then(|v| v.checked_mul(spatial_merge))
            .ok_or_else(|| "deepstack merged dimension overflow".to_string())?;
        let mut deepstack_layers = Vec::new();
        let mut deepstack_by_layer = vec![None; n_layers];
        for (l, ds_slot) in deepstack_by_layer.iter_mut().enumerate() {
            if expected_deepstack_layers == 0 {
                continue;
            }
            if deepstack_layers.len() >= expected_deepstack_layers {
                continue;
            }
            let norm_w_name = format!("v.deepstack.{l}.norm.weight");
            if find_gguf_tensor(&gguf, &norm_w_name).is_none() {
                continue;
            }

            let norm_b_name = format!("v.deepstack.{l}.norm.bias");
            let fc1_w_name = format!("v.deepstack.{l}.fc1.weight");
            let fc1_b_name = format!("v.deepstack.{l}.fc1.bias");
            let fc2_w_name = format!("v.deepstack.{l}.fc2.weight");
            let fc2_b_name = format!("v.deepstack.{l}.fc2.bias");

            let fc1_t = find_gguf_tensor(&gguf, &fc1_w_name)
                .ok_or_else(|| format!("tensor not found: {fc1_w_name}"))?;
            let (fc1_rows, fc1_cols) = tensor_matrix_shape(fc1_t, &fc1_w_name)?;
            if fc1_cols != merged_dim {
                return Err(format!(
                    "deepstack fc1 shape mismatch at layer {l}: cols={} expected merged_dim={}",
                    fc1_cols, merged_dim
                ));
            }

            let fc2_t = find_gguf_tensor(&gguf, &fc2_w_name)
                .ok_or_else(|| format!("tensor not found: {fc2_w_name}"))?;
            let (fc2_rows, fc2_cols) = tensor_matrix_shape(fc2_t, &fc2_w_name)?;
            if fc2_cols != fc1_rows {
                return Err(format!(
                    "deepstack fc2 shape mismatch at layer {l}: cols={} expected fc1_rows={}",
                    fc2_cols, fc1_rows
                ));
            }
            if fc2_rows != target_dim {
                return Err(format!(
                    "deepstack fc2 output dim mismatch at layer {l}: rows={} expected target_dim={}",
                    fc2_rows, target_dim
                ));
            }

            *ds_slot = Some(deepstack_layers.len());
            deepstack_layers.push(DeepstackLayerWeights {
                norm_w: load_tensor_float(&gguf, &norm_w_name, Some(merged_dim))?,
                norm_b: load_tensor_float(&gguf, &norm_b_name, Some(merged_dim))?,
                fc1_w: load_tensor_quantized(&gguf, &fc1_w_name, fc1_rows, fc1_cols)?,
                fc1_b: load_tensor_float(&gguf, &fc1_b_name, Some(fc1_rows))?,
                fc2_w: load_tensor_quantized(&gguf, &fc2_w_name, fc2_rows, fc2_cols)?,
                fc2_b: load_tensor_float(&gguf, &fc2_b_name, Some(fc2_rows))?,
            });
        }
        if deepstack_layers.len() != expected_deepstack_layers {
            return Err(format!(
                "mmproj/text deepstack mismatch: mmproj provides {} usable deepstack layer(s), but text model expects {}. hint: use an mmproj from the exact same Qwen3-VL checkpoint",
                deepstack_layers.len(),
                expected_deepstack_layers
            ));
        }

        Ok(Self {
            gguf,
            dim,
            head_count,
            head_dim,
            ff_dim,
            n_layers,
            eps,
            patch_size,
            base_image_size,
            spatial_merge,
            image_mean,
            image_std,
            patch_embd_w,
            patch_embd_b,
            position_embd,
            post_ln_w,
            post_ln_b,
            mm0_w,
            mm0_b,
            mm2_w,
            mm2_b,
            layers,
            deepstack_layers,
            deepstack_by_layer,
        })
    }

    fn gelu(x: f32) -> f32 {
        // Approximation used widely in inference runtimes.
        0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
    }

    fn layer_norm(&self, dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32]) {
        layer_norm_affine(dst, src, w, b, self.eps);
    }

    fn add_bias(v: &mut [f32], b: &[f32]) {
        for i in 0..v.len() {
            v[i] += b[i];
        }
    }

    fn position_embedding_interp(
        &self,
        y: usize,
        x: usize,
        out_h: usize,
        out_w: usize,
        dst: &mut [f32],
    ) {
        let base_grid = self.base_image_size / self.patch_size;
        let fy = if out_h <= 1 {
            0.0
        } else {
            y as f32 * (base_grid as f32 - 1.0) / (out_h as f32 - 1.0)
        };
        let fx = if out_w <= 1 {
            0.0
        } else {
            x as f32 * (base_grid as f32 - 1.0) / (out_w as f32 - 1.0)
        };
        let y0 = fy.floor() as usize;
        let x0 = fx.floor() as usize;
        let y1 = (y0 + 1).min(base_grid - 1);
        let x1 = (x0 + 1).min(base_grid - 1);
        let wy = fy - y0 as f32;
        let wx = fx - x0 as f32;

        let idx00 = (y0 * base_grid + x0) * self.dim;
        let idx01 = (y0 * base_grid + x1) * self.dim;
        let idx10 = (y1 * base_grid + x0) * self.dim;
        let idx11 = (y1 * base_grid + x1) * self.dim;

        for (c, d) in dst.iter_mut().enumerate().take(self.dim) {
            let v00 = self.position_embd[idx00 + c];
            let v01 = self.position_embd[idx01 + c];
            let v10 = self.position_embd[idx10 + c];
            let v11 = self.position_embd[idx11 + c];
            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            *d += top * (1.0 - wy) + bot * wy;
        }
    }

    fn patch_embed_and_reorder(
        &self,
        image: &PreparedImageTensor,
    ) -> Result<PatchTokenGrid, String> {
        if !image.width.is_multiple_of(self.patch_size)
            || !image.height.is_multiple_of(self.patch_size)
        {
            return Err(format!(
                "image '{}' size {}x{} is not divisible by patch_size {}",
                image.path, image.width, image.height, self.patch_size
            ));
        }
        let pw = image.width / self.patch_size;
        let ph = image.height / self.patch_size;
        if pw == 0 || ph == 0 {
            return Err(format!(
                "image '{}' produced empty patch grid (pw={} ph={})",
                image.path, pw, ph
            ));
        }
        if !pw.is_multiple_of(self.spatial_merge) || !ph.is_multiple_of(self.spatial_merge) {
            return Err(format!(
                "image '{}' patch grid {}x{} is not divisible by spatial_merge {}",
                image.path, pw, ph, self.spatial_merge
            ));
        }

        let patch_count = pw * ph;
        let mut row_major = vec![0.0f32; patch_count * self.dim];
        let chw = &image.data_chw;
        let image_plane = image.width * image.height;
        let kernel_elems = 3 * self.patch_size * self.patch_size;
        let dim = self.dim;
        let patch_size = self.patch_size;
        let image_width = image.width;
        let patch_embd_b = &self.patch_embd_b;
        let patch_embd_w = &self.patch_embd_w;

        row_major.par_chunks_mut(dim).enumerate().for_each_init(
            || vec![0.0f32; kernel_elems],
            |patch_buf, (patch_idx, out)| {
                let py = patch_idx / pw;
                let px = patch_idx % pw;
                out.copy_from_slice(patch_embd_b);

                let mut patch_off = 0usize;
                for ch in 0..3 {
                    let ch_base = ch * image_plane;
                    let y_base = py * patch_size;
                    let x_base = px * patch_size;
                    for ky in 0..patch_size {
                        let src_row = ch_base + (y_base + ky) * image_width + x_base;
                        let src = &chw[src_row..src_row + patch_size];
                        let dst = &mut patch_buf[patch_off..patch_off + patch_size];
                        dst.copy_from_slice(src);
                        patch_off += patch_size;
                    }
                }

                let mut woff = 0usize;
                for outv in out.iter_mut().take(dim) {
                    *outv += dot_f32_simd(patch_buf, &patch_embd_w[woff..woff + kernel_elems]);
                    woff += kernel_elems;
                }
            },
        );

        // Match Qwen-VL grouping order: iterate 2x2 merged cells, emit dy/dx within each cell.
        let mut tokens = Vec::with_capacity(row_major.len());
        let mut positions = Vec::with_capacity(patch_count);
        for y in (0..ph).step_by(self.spatial_merge) {
            for x in (0..pw).step_by(self.spatial_merge) {
                for dy in 0..self.spatial_merge {
                    for dx in 0..self.spatial_merge {
                        let yy = y + dy;
                        let xx = x + dx;
                        let src = (yy * pw + xx) * self.dim;
                        tokens.extend_from_slice(&row_major[src..src + self.dim]);
                        positions.push((yy, xx));
                    }
                }
            }
        }

        let n_tokens = tokens.len() / self.dim;
        for (tok, (yy, xx)) in positions.iter().enumerate().take(n_tokens) {
            let dst = &mut tokens[tok * self.dim..(tok + 1) * self.dim];
            self.position_embedding_interp(*yy, *xx, ph, pw, dst);
        }

        Ok((tokens, positions, pw, ph))
    }

    fn encode_same_shape_images(
        &self,
        images: &[&PreparedImageTensor],
    ) -> Result<Vec<MediaEmbeddingSequence>, String> {
        let Some(first_image) = images.first() else {
            return Ok(Vec::new());
        };
        let mapped = self.gguf.mapped.as_slice();
        let (mut x, positions, pw, ph) = self.patch_embed_and_reorder(first_image)?;
        let n_tokens = x.len() / self.dim;
        for image in &images[1..] {
            let (tokens, image_positions, image_pw, image_ph) =
                self.patch_embed_and_reorder(image)?;
            if image_pw != pw
                || image_ph != ph
                || tokens.len() != n_tokens * self.dim
                || image_positions != positions
            {
                return Err("qwen3vl vision microbatch requires matching patch grids".to_string());
            }
            x.extend(tokens);
        }
        let image_count = images.len();
        let total_tokens = n_tokens * image_count;
        let merge = self.spatial_merge * self.spatial_merge;
        let n_out = (pw / self.spatial_merge) * (ph / self.spatial_merge);
        if n_tokens != n_out * merge {
            return Err(format!(
                "unexpected qwen3vl token layout: n_tokens={} n_out={} merge={}",
                n_tokens, n_out, merge
            ));
        }
        let merged_dim = self.dim * merge;
        let dim = self.dim;
        let ff_dim = self.ff_dim;
        let eps = self.eps;

        let mut x_norm = vec![0.0f32; total_tokens * dim];
        let mut q = vec![0.0f32; total_tokens * dim];
        let mut k = vec![0.0f32; total_tokens * dim];
        let mut v = vec![0.0f32; total_tokens * dim];
        let mut qkv_batch = vec![0.0f32; total_tokens * dim * 3];
        let mut attn_out = vec![0.0f32; total_tokens * dim];
        let mut proj_out = vec![0.0f32; total_tokens * dim];
        let mut ffn_up_batch = vec![0.0f32; total_tokens * ff_dim];
        let mut ffn_down_batch = vec![0.0f32; total_tokens * dim];
        let mut batch_scratch = FloatBatchMatmulScratch::default();
        let mut attention_scratch = EncoderAttentionScratch::default();
        // Qwen vision M-RoPE depends only on the shared patch grid, so compute
        // its trigonometric table once and reuse it across every vision layer
        // and every image in this compatible microbatch.
        let vision_rope = vision_rope_coefficients(self.head_dim, &positions);
        let token_dim = self.mm2_b.len();
        let total_output_tokens = n_out * image_count;
        let mut deepstack_tokens = if self.deepstack_layers.is_empty() {
            Vec::new()
        } else {
            vec![0.0f32; total_output_tokens * self.deepstack_layers.len() * token_dim]
        };
        let deepstack_token_stride = self.deepstack_layers.len() * token_dim;
        let mut deepstack_merged_batch = Vec::new();
        let mut deepstack_normed_batch = Vec::new();
        let mut deepstack_hidden_batch = Vec::new();
        let mut deepstack_output_batch = Vec::new();

        for l in 0..self.n_layers {
            let layer = &self.layers[l];

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &x[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln1_w, &layer.ln1_b, eps);
            });

            matmul_encoder_batch(
                &mut qkv_batch,
                &x_norm,
                &layer.attn_qkv_w,
                mapped,
                total_tokens,
                &mut batch_scratch,
            )?;
            qkv_batch
                .par_chunks_mut(dim * 3)
                .zip(q.par_chunks_mut(dim))
                .zip(k.par_chunks_mut(dim))
                .zip(v.par_chunks_mut(dim))
                .for_each(|(((tok_qkv, q_dst), k_dst), v_dst)| {
                    Self::add_bias(tok_qkv, &layer.attn_qkv_b);
                    q_dst.copy_from_slice(&tok_qkv[..dim]);
                    k_dst.copy_from_slice(&tok_qkv[dim..2 * dim]);
                    v_dst.copy_from_slice(&tok_qkv[2 * dim..3 * dim]);
                });

            apply_vision_rope_cached(
                &mut q,
                &mut k,
                dim,
                self.head_count,
                self.head_dim,
                n_tokens,
                &vision_rope,
            );
            encoder_self_attention(
                &mut attn_out,
                &q,
                &k,
                &v,
                image_count,
                n_tokens,
                self.head_count,
                self.head_dim,
                &mut attention_scratch,
            )?;

            matmul_encoder_batch(
                &mut proj_out,
                &attn_out,
                &layer.attn_out_w,
                mapped,
                total_tokens,
                &mut batch_scratch,
            )?;
            proj_out
                .par_chunks_mut(dim)
                .for_each(|dst| Self::add_bias(dst, &layer.attn_out_b));
            for i in 0..x.len() {
                x[i] += proj_out[i];
            }

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &x[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln2_w, &layer.ln2_b, eps);
            });

            matmul_encoder_batch(
                &mut ffn_up_batch,
                &x_norm,
                &layer.ffn_up_w,
                mapped,
                total_tokens,
                &mut batch_scratch,
            )?;
            ffn_up_batch.par_chunks_mut(ff_dim).for_each(|ffn_up| {
                Self::add_bias(ffn_up, &layer.ffn_up_b);
                for value in ffn_up {
                    *value = Self::gelu(*value);
                }
            });
            matmul_encoder_batch(
                &mut ffn_down_batch,
                &ffn_up_batch,
                &layer.ffn_down_w,
                mapped,
                total_tokens,
                &mut batch_scratch,
            )?;
            x.par_chunks_mut(dim)
                .zip(ffn_down_batch.par_chunks_mut(dim))
                .for_each(|(destination, ffn_down)| {
                    Self::add_bias(ffn_down, &layer.ffn_down_b);
                    axpy_inplace(destination, 1.0, ffn_down);
                });

            if let Some(ds_idx) = self.deepstack_by_layer[l] {
                let deepstack = &self.deepstack_layers[ds_idx];
                let ds_layer_off = ds_idx * token_dim;
                let fc1_size = deepstack.fc1_b.len();
                let fc2_size = deepstack.fc2_b.len();

                deepstack_merged_batch.resize(total_output_tokens * merged_dim, 0.0);
                deepstack_merged_batch
                    .par_chunks_mut(merged_dim)
                    .enumerate()
                    .for_each(|(out_idx, destination)| {
                        for m in 0..merge {
                            let source =
                                &x[(out_idx * merge + m) * dim..(out_idx * merge + m + 1) * dim];
                            destination[m * dim..(m + 1) * dim].copy_from_slice(source);
                        }
                    });

                deepstack_normed_batch.resize(total_output_tokens * merged_dim, 0.0);
                deepstack_normed_batch
                    .par_chunks_mut(merged_dim)
                    .zip(deepstack_merged_batch.par_chunks(merged_dim))
                    .for_each(|(destination, source)| {
                        layer_norm_affine(
                            destination,
                            source,
                            &deepstack.norm_w,
                            &deepstack.norm_b,
                            eps,
                        );
                    });

                deepstack_hidden_batch.resize(total_output_tokens * fc1_size, 0.0);
                matmul_encoder_batch(
                    &mut deepstack_hidden_batch,
                    &deepstack_normed_batch,
                    &deepstack.fc1_w,
                    mapped,
                    total_output_tokens,
                    &mut batch_scratch,
                )?;
                deepstack_hidden_batch
                    .par_chunks_mut(fc1_size)
                    .for_each(|hidden| {
                        Self::add_bias(hidden, &deepstack.fc1_b);
                        for value in hidden {
                            *value = Self::gelu(*value);
                        }
                    });

                deepstack_output_batch.resize(total_output_tokens * fc2_size, 0.0);
                matmul_encoder_batch(
                    &mut deepstack_output_batch,
                    &deepstack_hidden_batch,
                    &deepstack.fc2_w,
                    mapped,
                    total_output_tokens,
                    &mut batch_scratch,
                )?;
                deepstack_tokens
                    .par_chunks_mut(deepstack_token_stride)
                    .zip(deepstack_output_batch.par_chunks_mut(fc2_size))
                    .for_each(|(deepstack_token, output)| {
                        Self::add_bias(output, &deepstack.fc2_b);
                        deepstack_token[ds_layer_off..ds_layer_off + token_dim]
                            .copy_from_slice(&output[..token_dim]);
                    });
            }
        }

        for t in 0..total_tokens {
            let src = &x[t * dim..(t + 1) * dim];
            let dst = &mut x_norm[t * dim..(t + 1) * dim];
            self.layer_norm(dst, src, &self.post_ln_w, &self.post_ln_b);
        }
        std::mem::swap(&mut x, &mut x_norm);

        let mut merged_batch = vec![0.0f32; total_output_tokens * merged_dim];
        for out_idx in 0..total_output_tokens {
            for m in 0..merge {
                let src = &x[(out_idx * merge + m) * dim..(out_idx * merge + m + 1) * dim];
                let dst = &mut merged_batch
                    [out_idx * merged_dim + m * dim..out_idx * merged_dim + (m + 1) * dim];
                dst.copy_from_slice(src);
            }
        }
        let mut hidden_batch = vec![0.0f32; total_output_tokens * merged_dim];
        matmul_encoder_batch(
            &mut hidden_batch,
            &merged_batch,
            &self.mm0_w,
            mapped,
            total_output_tokens,
            &mut batch_scratch,
        )?;
        for hidden in hidden_batch.chunks_mut(merged_dim) {
            Self::add_bias(hidden, &self.mm0_b);
            for value in hidden {
                *value = Self::gelu(*value);
            }
        }
        let mut output_batch = vec![0.0f32; total_output_tokens * token_dim];
        matmul_encoder_batch(
            &mut output_batch,
            &hidden_batch,
            &self.mm2_w,
            mapped,
            total_output_tokens,
            &mut batch_scratch,
        )?;
        for output in output_batch.chunks_mut(token_dim) {
            Self::add_bias(output, &self.mm2_b);
        }
        let mut sequences = Vec::with_capacity(image_count);
        for image_index in 0..image_count {
            let mut tokens = Vec::with_capacity(n_out);
            for image_out_idx in 0..n_out {
                let out_idx = image_index * n_out + image_out_idx;
                let mut token = Vec::with_capacity(token_dim * (1 + self.deepstack_layers.len()));
                token.extend_from_slice(
                    &output_batch[out_idx * token_dim..(out_idx + 1) * token_dim],
                );
                if !deepstack_tokens.is_empty() {
                    let base = out_idx * self.deepstack_layers.len() * token_dim;
                    for ds_idx in 0..self.deepstack_layers.len() {
                        let ds_off = base + ds_idx * token_dim;
                        token.extend_from_slice(&deepstack_tokens[ds_off..ds_off + token_dim]);
                    }
                }
                tokens.push(token);
            }
            sequences.push(MediaEmbeddingSequence { tokens });
        }

        Ok(sequences)
    }

    pub(crate) fn encode_images(
        &self,
        images: &[PreparedImageTensor],
    ) -> Result<Vec<MediaEmbeddingSequence>, String> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![None; images.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (index, image) in images.iter().enumerate() {
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = &images[group[0]];
                first.width == image.width && first.height == image.height
            }) {
                group.push(index);
            } else {
                groups.push(vec![index]);
            }
        }
        for group in groups {
            let first = &images[group[0]];
            let tokens_per_image = (first.width / self.patch_size)
                .checked_mul(first.height / self.patch_size)
                .ok_or_else(|| "qwen3vl vision patch token count overflow".to_string())?;
            let image_limit = vision_microbatch_image_limit(tokens_per_image);
            for group_chunk in group.chunks(image_limit) {
                let group_images = group_chunk
                    .iter()
                    .map(|&index| &images[index])
                    .collect::<Vec<_>>();
                let sequences = self.encode_same_shape_images(&group_images)?;
                if sequences.len() != group_chunk.len() {
                    return Err(
                        "qwen3vl vision microbatch returned the wrong result count".to_string()
                    );
                }
                for (&index, sequence) in group_chunk.iter().zip(sequences) {
                    out[index] = Some(sequence);
                }
            }
        }
        out.into_iter()
            .map(|sequence| {
                sequence.ok_or_else(|| "qwen3vl vision microbatch lost an image".to_string())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_VISION_MICROBATCH_IMAGES, apply_vision_rope_cached, vision_microbatch_image_limit,
        vision_rope_coefficients,
    };

    #[test]
    fn vision_microbatch_limit_bounds_images_and_patch_tokens() {
        assert_eq!(vision_microbatch_image_limit(0), 1);
        assert_eq!(vision_microbatch_image_limit(512), 4);
        assert_eq!(vision_microbatch_image_limit(1_024), 4);
        assert_eq!(vision_microbatch_image_limit(1_365), 3);
        assert_eq!(vision_microbatch_image_limit(2_048), 2);
        assert_eq!(vision_microbatch_image_limit(4_097), 1);
        assert_eq!(MAX_VISION_MICROBATCH_IMAGES, 4);
    }

    #[test]
    fn cached_parallel_vision_rope_matches_per_layer_scalar_reference() {
        let head_dim = 8usize;
        let head_count = 2usize;
        let dim = head_dim * head_count;
        let positions = [(0usize, 0usize), (0, 1), (2, 0), (2, 3), (4, 1)];
        let sequence_count = 2usize;
        let value_count = sequence_count * positions.len() * dim;
        let mut q = (0..value_count)
            .map(|i| ((i as f32 + 0.5) * 0.071).sin())
            .collect::<Vec<_>>();
        let mut k = (0..value_count)
            .map(|i| ((i as f32 + 1.25) * 0.043).cos())
            .collect::<Vec<_>>();
        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        let half_head = head_dim / 2;
        let quarter_head = head_dim / 4;

        for sequence in 0..sequence_count {
            for (token, &(py, px)) in positions.iter().enumerate() {
                let token_offset = (sequence * positions.len() + token) * dim;
                for head in 0..head_count {
                    let head_offset = token_offset + head * head_dim;
                    for ic in 0..half_head {
                        let (position, local_ic) = if ic < quarter_head {
                            (py as f32, ic)
                        } else {
                            (px as f32, ic - quarter_head)
                        };
                        let frequency =
                            1.0 / 10_000.0f32.powf((2 * local_ic) as f32 / head_dim as f32);
                        let theta = position * frequency;
                        let (cos, sin) = (theta.cos(), theta.sin());
                        let a = head_offset + ic;
                        let b = a + half_head;

                        let q0 = expected_q[a];
                        let q1 = expected_q[b];
                        expected_q[a] = q0 * cos - q1 * sin;
                        expected_q[b] = q0 * sin + q1 * cos;

                        let k0 = expected_k[a];
                        let k1 = expected_k[b];
                        expected_k[a] = k0 * cos - k1 * sin;
                        expected_k[b] = k0 * sin + k1 * cos;
                    }
                }
            }
        }

        let coefficients = vision_rope_coefficients(head_dim, &positions);
        apply_vision_rope_cached(
            &mut q,
            &mut k,
            dim,
            head_count,
            head_dim,
            positions.len(),
            &coefficients,
        );

        assert_eq!(q, expected_q);
        assert_eq!(k, expected_k);
    }
}
