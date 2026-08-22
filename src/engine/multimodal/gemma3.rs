use crate::engine::io::{
    find_gguf_tensor, get_gguf_bool_from_map, get_gguf_f32_array_from_map, get_gguf_float_from_map,
    get_gguf_int_from_map,
};
use crate::engine::kernels::{
    axpy_inplace, dequantize_tensor, dot_f32_simd, get_block_size, get_type_size, matmul_quantized,
    scale_slice_inplace,
};
use crate::engine::multimodal::injection::MediaEmbeddingSequence;
use crate::engine::types::{GGUFFile, Gguftensor, QuantizedTensor};
use crate::engine::vision::PreparedImageTensor;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

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
    attn_q_w: QuantizedTensor,
    attn_q_b: Vec<f32>,
    attn_k_w: QuantizedTensor,
    attn_k_b: Vec<f32>,
    attn_v_w: QuantizedTensor,
    attn_v_b: Vec<f32>,
    attn_out_w: QuantizedTensor,
    attn_out_b: Vec<f32>,
    ffn_up_w: QuantizedTensor,
    ffn_up_b: Vec<f32>,
    ffn_down_w: QuantizedTensor,
    ffn_down_b: Vec<f32>,
}

pub(crate) struct Gemma3VisionEncoder {
    gguf: GGUFFile,
    dim: usize,
    head_count: usize,
    head_dim: usize,
    ff_dim: usize,
    n_layers: usize,
    eps: f32,
    patch_size: usize,
    base_image_size: usize,
    merge_factor: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    use_gelu: bool,
    patch_embd_w: Vec<f32>,
    patch_embd_b: Vec<f32>,
    position_embd: Vec<f32>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm_input_proj_w: Vec<f32>,
    mm_input_proj_ne0: usize,
    mm_input_proj_ne1: usize,
    mm_soft_emb_norm_w: Vec<f32>,
    layers: Vec<VisionLayerWeights>,
}

impl Gemma3VisionEncoder {
    const FAST_PRE_ATTENTION_POOL_THRESHOLD: usize = 2_048;

    fn fast_pooling_enabled() -> bool {
        matches!(
            std::env::var("GGUF_GEMMA3_ENABLE_FAST_POOL"),
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        )
    }

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
        self.patch_size.saturating_mul(self.merge_factor).max(1)
    }

    pub(crate) fn recommended_image_normalization(&self) -> ([f32; 3], [f32; 3]) {
        (self.image_mean, self.image_std)
    }

    pub(crate) fn new(gguf: GGUFFile, target_dim: usize) -> Result<Self, String> {
        let dim = get_gguf_int_from_map(&gguf.kv, "clip.vision.embedding_length", 0) as usize;
        let head_count =
            get_gguf_int_from_map(&gguf.kv, "clip.vision.attention.head_count", 0) as usize;
        let ff_dim = get_gguf_int_from_map(&gguf.kv, "clip.vision.feed_forward_length", 0) as usize;
        let n_layers = get_gguf_int_from_map(&gguf.kv, "clip.vision.block_count", 0) as usize;
        let eps =
            get_gguf_float_from_map(&gguf.kv, "clip.vision.attention.layer_norm_epsilon", 1e-6);
        let patch_size = get_gguf_int_from_map(&gguf.kv, "clip.vision.patch_size", 14) as usize;
        let base_image_size =
            get_gguf_int_from_map(&gguf.kv, "clip.vision.image_size", 896) as usize;
        let merge_factor =
            get_gguf_int_from_map(&gguf.kv, "clip.vision.projector.scale_factor", 4) as usize;
        let image_mean = Self::parse_rgb_triplet(
            get_gguf_f32_array_from_map(&gguf.kv, "clip.vision.image_mean"),
            [0.5, 0.5, 0.5],
            "clip.vision.image_mean",
        )?;
        let image_std = Self::parse_rgb_triplet(
            get_gguf_f32_array_from_map(&gguf.kv, "clip.vision.image_std"),
            [0.5, 0.5, 0.5],
            "clip.vision.image_std",
        )?;
        let use_gelu = get_gguf_bool_from_map(&gguf.kv, "clip.use_gelu", true);

        if dim == 0
            || head_count == 0
            || ff_dim == 0
            || n_layers == 0
            || patch_size == 0
            || merge_factor == 0
        {
            return Err(
                "invalid gemma3 mmproj metadata: one or more required clip.vision.* keys are missing/zero"
                    .to_string(),
            );
        }
        if !dim.is_multiple_of(head_count) {
            return Err(format!(
                "invalid gemma3 mmproj metadata: dim {} is not divisible by head_count {}",
                dim, head_count
            ));
        }
        if !base_image_size.is_multiple_of(patch_size) {
            return Err(format!(
                "invalid gemma3 mmproj metadata: image_size {} is not divisible by patch_size {}",
                base_image_size, patch_size
            ));
        }
        let head_dim = dim / head_count;

        let patch_kernel_elems = patch_size
            .checked_mul(patch_size)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "patch kernel element count overflow".to_string())?;
        let base_patch_grid = base_image_size / patch_size;
        let base_pos_tokens = base_patch_grid
            .checked_mul(base_patch_grid)
            .ok_or_else(|| "position token count overflow".to_string())?;

        let patch_embd_w =
            load_tensor_float(&gguf, "v.patch_embd.weight", Some(patch_kernel_elems))?;
        let patch_embd_b = load_tensor_float(&gguf, "v.patch_embd.bias", Some(dim))?;
        let position_embd =
            load_tensor_float(&gguf, "v.position_embd.weight", Some(base_pos_tokens * dim))?;
        let post_ln_w = load_tensor_float(&gguf, "v.post_ln.weight", Some(dim))?;
        let post_ln_b = load_tensor_float(&gguf, "v.post_ln.bias", Some(dim))?;
        let mm_input_proj_t = find_gguf_tensor(&gguf, "mm.input_projection.weight")
            .ok_or_else(|| "tensor not found: mm.input_projection.weight".to_string())?;
        if mm_input_proj_t.n_dims < 2 {
            return Err("tensor mm.input_projection.weight must be at least 2D".to_string());
        }
        let mm_input_proj_ne0 = mm_input_proj_t.ne[0] as usize;
        let mm_input_proj_ne1 = mm_input_proj_t.ne[1] as usize;
        // llama.cpp gemma3 path multiplies transpose(mm.input_projection.weight),
        // so we expect stored shape [text_dim, vision_dim] and project to [vision_dim, text_dim].
        if mm_input_proj_ne0 != target_dim || mm_input_proj_ne1 != dim {
            return Err(format!(
                "unexpected mm.input_projection.weight shape: got {}x{}, expected {}x{} (text_dim x vision_dim)",
                mm_input_proj_ne0, mm_input_proj_ne1, target_dim, dim
            ));
        }
        let mm_input_proj_w =
            load_tensor_float(&gguf, "mm.input_projection.weight", Some(target_dim * dim))?;
        let mm_soft_emb_norm_w = load_tensor_float(&gguf, "mm.soft_emb_norm.weight", Some(dim))?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let prefix = format!("v.blk.{l}");
            layers.push(VisionLayerWeights {
                ln1_w: load_tensor_float(&gguf, &format!("{prefix}.ln1.weight"), Some(dim))?,
                ln1_b: load_tensor_float(&gguf, &format!("{prefix}.ln1.bias"), Some(dim))?,
                ln2_w: load_tensor_float(&gguf, &format!("{prefix}.ln2.weight"), Some(dim))?,
                ln2_b: load_tensor_float(&gguf, &format!("{prefix}.ln2.bias"), Some(dim))?,
                attn_q_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.attn_q.weight"),
                    dim,
                    dim,
                )?,
                attn_q_b: load_tensor_float(&gguf, &format!("{prefix}.attn_q.bias"), Some(dim))?,
                attn_k_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.attn_k.weight"),
                    dim,
                    dim,
                )?,
                attn_k_b: load_tensor_float(&gguf, &format!("{prefix}.attn_k.bias"), Some(dim))?,
                attn_v_w: load_tensor_quantized(
                    &gguf,
                    &format!("{prefix}.attn_v.weight"),
                    dim,
                    dim,
                )?,
                attn_v_b: load_tensor_float(&gguf, &format!("{prefix}.attn_v.bias"), Some(dim))?,
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
            merge_factor,
            image_mean,
            image_std,
            use_gelu,
            patch_embd_w,
            patch_embd_b,
            position_embd,
            post_ln_w,
            post_ln_b,
            mm_input_proj_w,
            mm_input_proj_ne0,
            mm_input_proj_ne1,
            mm_soft_emb_norm_w,
            layers,
        })
    }

    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
    }

    fn quick_gelu(x: f32) -> f32 {
        let z = 1.702 * x;
        x / (1.0 + (-z).exp())
    }

    fn layer_norm(&self, dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32]) {
        layer_norm_affine(dst, src, w, b, self.eps);
    }

    fn rms_norm_mul_weight(&self, dst: &mut [f32], src: &[f32], w: &[f32]) {
        let mut mean_sq = 0.0f32;
        for &v in src {
            mean_sq += v * v;
        }
        mean_sq /= src.len() as f32;
        let inv = 1.0f32 / (mean_sq + self.eps).sqrt();
        for i in 0..src.len() {
            dst[i] = src[i] * inv * w[i];
        }
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
    ) -> Result<(), String> {
        let base_grid = self.base_image_size / self.patch_size;
        let expected = base_grid
            .checked_mul(base_grid)
            .and_then(|v| v.checked_mul(self.dim))
            .ok_or_else(|| "position embedding shape overflow".to_string())?;
        if self.position_embd.len() != expected {
            return Err(format!(
                "invalid gemma3 position embedding tensor size: got {}, expected {} (grid={} dim={})",
                self.position_embd.len(),
                expected,
                base_grid,
                self.dim
            ));
        }

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
        Ok(())
    }

    fn patch_embed_and_add_position(
        &self,
        image: &PreparedImageTensor,
    ) -> Result<(Vec<f32>, usize, usize), String> {
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

        let patch_count = pw
            .checked_mul(ph)
            .ok_or_else(|| "patch grid overflow".to_string())?;

        let mut tokens = vec![0.0f32; patch_count * self.dim];
        let chw = &image.data_chw;
        let image_plane = image.width * image.height;
        let kernel_elems = 3 * self.patch_size * self.patch_size;
        let dim = self.dim;
        let patch_size = self.patch_size;
        let image_width = image.width;
        let patch_embd_b = &self.patch_embd_b;
        let patch_embd_w = &self.patch_embd_w;

        tokens.par_chunks_mut(dim).enumerate().for_each_init(
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

        for py in 0..ph {
            for px in 0..pw {
                let tok = py * pw + px;
                let tok_off = tok * self.dim;
                let dst = &mut tokens[tok_off..tok_off + self.dim];
                self.position_embedding_interp(py, px, ph, pw, dst)?;
            }
        }

        Ok((tokens, pw, ph))
    }

    fn pool_patch_grid(
        &self,
        tokens: &[f32],
        patch_w: usize,
        patch_h: usize,
    ) -> Result<Vec<f32>, String> {
        if !patch_w.is_multiple_of(self.merge_factor) || !patch_h.is_multiple_of(self.merge_factor)
        {
            return Err(format!(
                "gemma3 pooling requires patch grid divisible by merge factor {} (got {}x{})",
                self.merge_factor, patch_w, patch_h
            ));
        }
        let out_w = patch_w / self.merge_factor;
        let out_h = patch_h / self.merge_factor;
        let out_tokens = out_w
            .checked_mul(out_h)
            .ok_or_else(|| "pooled token count overflow".to_string())?;
        let mut pooled = vec![0.0f32; out_tokens * self.dim];
        let inv = 1.0f32 / (self.merge_factor * self.merge_factor) as f32;

        for oy in 0..out_h {
            for ox in 0..out_w {
                let out_idx = oy * out_w + ox;
                let dst = &mut pooled[out_idx * self.dim..(out_idx + 1) * self.dim];
                for my in 0..self.merge_factor {
                    for mx in 0..self.merge_factor {
                        let iy = oy * self.merge_factor + my;
                        let ix = ox * self.merge_factor + mx;
                        let in_idx = iy * patch_w + ix;
                        let src = &tokens[in_idx * self.dim..(in_idx + 1) * self.dim];
                        axpy_inplace(dst, 1.0, src);
                    }
                }
                scale_slice_inplace(dst, inv);
            }
        }

        Ok(pooled)
    }

    fn encode_single_image(
        &self,
        image: &PreparedImageTensor,
    ) -> Result<MediaEmbeddingSequence, String> {
        let mapped = self.gguf.mapped.as_slice();
        let (mut x, patch_w, patch_h) = self.patch_embed_and_add_position(image)?;
        let mut pre_pooled_for_speed = false;
        let mut n_tokens = x.len() / self.dim;

        // Optional speed mode: pre-pool before ViT to reduce attention cost.
        // This is disabled by default because it noticeably reduces vision quality.
        if n_tokens > Self::FAST_PRE_ATTENTION_POOL_THRESHOLD && Self::fast_pooling_enabled() {
            x = self.pool_patch_grid(&x, patch_w, patch_h)?;
            n_tokens = x.len() / self.dim;
            pre_pooled_for_speed = true;
        }

        let mut x_norm = vec![0.0f32; n_tokens * self.dim];
        let mut q = vec![0.0f32; n_tokens * self.dim];
        let mut k = vec![0.0f32; n_tokens * self.dim];
        let mut v = vec![0.0f32; n_tokens * self.dim];
        let head_token_stride = n_tokens * self.head_dim;
        let mut attn_head_major = vec![0.0f32; self.head_count * head_token_stride];
        let mut attn_out = vec![0.0f32; n_tokens * self.dim];
        let mut proj_out = vec![0.0f32; n_tokens * self.dim];
        let dim = self.dim;
        let ff_dim = self.ff_dim;
        let eps = self.eps;
        let use_gelu = self.use_gelu;

        for l in 0..self.n_layers {
            let layer = &self.layers[l];

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &x[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln1_w, &layer.ln1_b, eps);
            });

            q.par_chunks_mut(dim)
                .zip(k.par_chunks_mut(dim))
                .zip(v.par_chunks_mut(dim))
                .enumerate()
                .try_for_each(|(t, ((q_dst, k_dst), v_dst))| -> Result<(), String> {
                    let src = &x_norm[t * dim..(t + 1) * dim];
                    matmul_quantized(q_dst, src, &layer.attn_q_w, mapped)?;
                    Self::add_bias(q_dst, &layer.attn_q_b);
                    matmul_quantized(k_dst, src, &layer.attn_k_w, mapped)?;
                    Self::add_bias(k_dst, &layer.attn_k_b);
                    matmul_quantized(v_dst, src, &layer.attn_v_w, mapped)?;
                    Self::add_bias(v_dst, &layer.attn_v_b);
                    Ok(())
                })?;

            let inv_scale = 1.0 / (self.head_dim as f32).sqrt();
            let head_dim = self.head_dim;
            attn_head_major
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(row_idx, out)| {
                    let h = row_idx / n_tokens;
                    let i = row_idx % n_tokens;
                    let h_off = h * head_dim;
                    let qi = &q[i * dim + h_off..i * dim + h_off + head_dim];

                    out.fill(0.0);
                    let mut max_score = f32::NEG_INFINITY;
                    let mut score_sum = 0.0f32;
                    for j in 0..n_tokens {
                        let kj = &k[j * dim + h_off..j * dim + h_off + head_dim];
                        let score = dot_f32_simd(qi, kj) * inv_scale;

                        if score > max_score {
                            if score_sum > 0.0 {
                                let rescale = (max_score - score).exp();
                                scale_slice_inplace(out, rescale);
                                score_sum *= rescale;
                            }
                            max_score = score;
                        }

                        let weight = (score - max_score).exp();
                        score_sum += weight;
                        let vj = &v[j * dim + h_off..j * dim + h_off + head_dim];
                        axpy_inplace(out, weight, vj);
                    }

                    if score_sum > 0.0 {
                        let inv = 1.0 / score_sum;
                        scale_slice_inplace(out, inv);
                    }
                });
            for t in 0..n_tokens {
                let dst = &mut attn_out[t * self.dim..(t + 1) * self.dim];
                for h in 0..self.head_count {
                    let src = &attn_head_major[h * head_token_stride + t * self.head_dim
                        ..h * head_token_stride + (t + 1) * self.head_dim];
                    let off = h * self.head_dim;
                    dst[off..off + self.head_dim].copy_from_slice(src);
                }
            }

            proj_out.par_chunks_mut(dim).enumerate().try_for_each(
                |(t, dst)| -> Result<(), String> {
                    let src = &attn_out[t * dim..(t + 1) * dim];
                    matmul_quantized(dst, src, &layer.attn_out_w, mapped)?;
                    Self::add_bias(dst, &layer.attn_out_b);
                    Ok(())
                },
            )?;
            for i in 0..x.len() {
                x[i] += proj_out[i];
            }

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &x[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln2_w, &layer.ln2_b, eps);
            });

            x.par_chunks_mut(dim).enumerate().try_for_each_init(
                || (vec![0.0f32; ff_dim], vec![0.0f32; dim]),
                |(ffn_up, ffn_down), (t, dst)| -> Result<(), String> {
                    let src = &x_norm[t * dim..(t + 1) * dim];
                    matmul_quantized(ffn_up, src, &layer.ffn_up_w, mapped)?;
                    Self::add_bias(ffn_up, &layer.ffn_up_b);
                    for v in ffn_up.iter_mut() {
                        *v = if use_gelu {
                            Self::gelu(*v)
                        } else {
                            Self::quick_gelu(*v)
                        };
                    }
                    matmul_quantized(ffn_down, ffn_up, &layer.ffn_down_w, mapped)?;
                    Self::add_bias(ffn_down, &layer.ffn_down_b);
                    axpy_inplace(dst, 1.0, ffn_down);
                    Ok(())
                },
            )?;
        }

        for t in 0..n_tokens {
            let src = &x[t * dim..(t + 1) * dim];
            let dst = &mut x_norm[t * dim..(t + 1) * dim];
            self.layer_norm(dst, src, &self.post_ln_w, &self.post_ln_b);
        }
        std::mem::swap(&mut x, &mut x_norm);

        let pooled = if pre_pooled_for_speed {
            x
        } else {
            self.pool_patch_grid(&x, patch_w, patch_h)?
        };
        let n_out = pooled.len() / self.dim;
        let out_dim = self.mm_input_proj_ne0;
        if self.mm_input_proj_ne1 != self.dim {
            return Err(format!(
                "mm.input_projection.weight shape mismatch at runtime: ne1={} dim={}",
                self.mm_input_proj_ne1, self.dim
            ));
        }
        let mut normed = vec![0.0f32; self.dim];
        let mut projected = vec![0.0f32; out_dim];
        let mut tokens: Vec<Vec<f32>> = Vec::with_capacity(n_out);

        for out_idx in 0..n_out {
            let src = &pooled[out_idx * self.dim..(out_idx + 1) * self.dim];
            self.rms_norm_mul_weight(&mut normed, src, &self.mm_soft_emb_norm_w);
            // Match llama.cpp gemma3 path:
            // projected = transpose(mm.input_projection.weight) * normed
            for (out, dst) in projected.iter_mut().enumerate().take(out_dim) {
                let mut acc = 0.0f32;
                for (inp, &v) in normed.iter().enumerate() {
                    acc += v * self.mm_input_proj_w[out + self.mm_input_proj_ne0 * inp];
                }
                *dst = acc;
            }
            tokens.push(projected.clone());
        }

        Ok(MediaEmbeddingSequence { tokens })
    }

    pub(crate) fn encode_images(
        &self,
        images: &[PreparedImageTensor],
    ) -> Result<Vec<MediaEmbeddingSequence>, String> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(images.len());
        for image in images {
            out.push(self.encode_single_image(image)?);
        }
        Ok(out)
    }
}
