#[path = "gemma3_attention.rs"]
mod attention;
#[path = "gemma3_math.rs"]
mod math;

use crate::engine::io::{
    bf16_to_fp32, find_gguf_tensor, fp32_to_bf16, fp32_to_fp16, get_gguf_bool_from_map,
    get_gguf_f32_array_from_map, get_gguf_float_from_map, get_gguf_int_from_map,
};
use crate::engine::kernels::{
    axpy_inplace, dequantize_tensor, dot_f32_simd, get_block_size, get_type_size,
    matmul_quantized_batch_dequantized, scale_slice_inplace,
};
use crate::engine::multimodal::injection::MediaEmbeddingSequence;
use crate::engine::types::{GGML_TYPE_BF16, GGML_TYPE_F16, GGUFFile, Gguftensor, QuantizedTensor};
use crate::engine::vision::PreparedImageTensor;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

use super::{FloatBatchMatmulScratch, matmul_encoder_batch};

type StageObserver<'a> = dyn FnMut(&str, &[f32]) -> Result<(), String> + 'a;

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

use math::{layer_norm_affine, rms_norm_weight};

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
    patch_embd_f16: Option<Vec<u16>>,
    patch_embd_b: Vec<f32>,
    position_embd: Vec<f32>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm_input_proj_w: Vec<f32>,
    mm_input_proj_f16: Option<Vec<u16>>,
    mm_input_proj_bf16: bool,
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

    /// Projected token count for one prepared view, known before any encode.
    /// Patch grid divided by the projector's pooling factor, as `pool_patch_grid`
    /// computes it; the same alignment `recommended_image_alignment` reports.
    pub(crate) fn planned_view_tokens(&self, width: usize, height: usize) -> Result<usize, String> {
        let align = self.recommended_image_alignment();
        if width == 0
            || height == 0
            || !width.is_multiple_of(align)
            || !height.is_multiple_of(align)
        {
            return Err(format!(
                "gemma3 view {width}x{height} must be a positive multiple of {align}"
            ));
        }
        (width / align)
            .checked_mul(height / align)
            .ok_or_else(|| "gemma3 view token count overflow".to_string())
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
        // ggml_conv_2d uses F16 im2col (including the F32-weight case).
        // BF16 patch weights are the exception: that graph uses F32 im2col.
        let patch_embd_f16 = (find_gguf_tensor(&gguf, "v.patch_embd.weight")
            .expect("patch tensor was loaded")
            .ttype
            .0
            != GGML_TYPE_BF16)
            .then(|| patch_embd_w.iter().copied().map(fp32_to_fp16).collect());
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
        let mm_input_proj_bf16 = mm_input_proj_t.ttype.0 == GGML_TYPE_BF16;
        let mm_input_proj_f16 = (mm_input_proj_t.ttype.0 == GGML_TYPE_F16).then(|| {
            // Store the transposed rows once; projected outputs are text-major.
            (0..target_dim)
                .flat_map(|out| {
                    let weights = &mm_input_proj_w;
                    (0..dim).map(move |inp| fp32_to_fp16(weights[out + target_dim * inp]))
                })
                .collect()
        });
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
            patch_embd_f16,
            patch_embd_b,
            position_embd,
            post_ln_w,
            post_ln_b,
            mm_input_proj_w,
            mm_input_proj_f16,
            mm_input_proj_bf16,
            mm_input_proj_ne0,
            mm_input_proj_ne1,
            mm_soft_emb_norm_w,
            layers,
        })
    }

    fn gelu(x: f32) -> f32 {
        math::gelu(x)
    }

    fn quick_gelu(x: f32) -> f32 {
        let z = 1.702 * x;
        x / (1.0 + (-z).exp())
    }

    fn layer_norm(&self, dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32]) {
        layer_norm_affine(dst, src, w, b, self.eps);
    }

    fn rms_norm_mul_weight(&self, dst: &mut [f32], src: &[f32], w: &[f32]) {
        rms_norm_weight(dst, src, w, self.eps);
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
        let patch_embd_f16 = &self.patch_embd_f16;

        tokens.par_chunks_mut(dim).enumerate().for_each_init(
            || (vec![0.0f32; kernel_elems], vec![0u16; kernel_elems]),
            |(patch_buf, patch_half), (patch_idx, out)| {
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

                if let Some(weights) = patch_embd_f16 {
                    for (dst, &src) in patch_half.iter_mut().zip(patch_buf.iter()) {
                        *dst = fp32_to_fp16(src);
                    }
                    for (outv, row) in out.iter_mut().zip(weights.chunks_exact(kernel_elems)) {
                        *outv += math::dot_f16(patch_half, row);
                    }
                } else {
                    for (outv, row) in out.iter_mut().zip(patch_embd_w.chunks_exact(kernel_elems)) {
                        *outv += dot_f32_simd(patch_buf, row);
                    }
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
        self.encode_single_image_observed::<false>(image, &[], None)
    }

    /// Diagnostic hook: the observer borrows one token-major stage at a time.
    /// Ordinary inference neither retains nor copies intermediate stages.
    // Diagnostic stage API: called from examples/, which compile the engine separately.
    #[allow(dead_code)]
    pub(crate) fn encode_image_with_stages(
        &self,
        image: &PreparedImageTensor,
        detailed_layers: &[usize],
        f32_activations: bool,
        observer: &mut StageObserver<'_>,
    ) -> Result<MediaEmbeddingSequence, String> {
        if Self::fast_pooling_enabled() {
            return Err("encoder stage validation requires Gemma fast pooling disabled".into());
        }
        if detailed_layers.iter().any(|&layer| layer >= self.n_layers) {
            return Err("encoder diagnostic layer index is out of range".into());
        }
        if f32_activations {
            self.encode_single_image_observed::<true>(image, detailed_layers, Some(observer))
        } else {
            self.encode_single_image_observed::<false>(image, detailed_layers, Some(observer))
        }
    }

    fn encode_single_image_observed<const F32_ACTIVATIONS: bool>(
        &self,
        image: &PreparedImageTensor,
        detailed_layers: &[usize],
        mut observer: Option<&mut StageObserver<'_>>,
    ) -> Result<MediaEmbeddingSequence, String> {
        let mapped = self.gguf.mapped.as_slice();
        let (mut x, patch_w, patch_h) = self.patch_embed_and_add_position(image)?;
        if let Some(visit) = observer.as_mut() {
            visit("patch_embeddings", &x)?;
        }
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
        let mut attn_out = vec![0.0f32; n_tokens * self.dim];
        let mut proj_out = vec![0.0f32; n_tokens * self.dim];
        let dim = self.dim;
        let ff_dim = self.ff_dim;
        let mut ffn_up_batch = vec![0.0f32; n_tokens * ff_dim];
        let mut ffn_down_batch = vec![0.0f32; n_tokens * dim];
        let mut batch_scratch = FloatBatchMatmulScratch::default();
        let mut bf16_scratch = math::Bf16MatmulScratch::default();
        let mut f32_scratch = Vec::new();
        // The F32 control removes activation narrowing to distinguish graph
        // errors from rounding propagation. Normal inference always uses false.
        let mut matmul = |dst: &mut [f32], src: &[f32], weight: &QuantizedTensor| {
            if F32_ACTIVATIONS {
                matmul_quantized_batch_dequantized(
                    dst,
                    src,
                    weight,
                    mapped,
                    n_tokens,
                    0,
                    weight.rows,
                    &mut f32_scratch,
                )
            } else if weight.ttype.0 == GGML_TYPE_BF16 {
                math::matmul_bf16(dst, src, weight, mapped, n_tokens, &mut bf16_scratch)
            } else {
                matmul_encoder_batch(dst, src, weight, mapped, n_tokens, &mut batch_scratch)
            }
        };
        let mut attention_scratch = attention::AttentionScratch::default();
        let eps = self.eps;
        let use_gelu = self.use_gelu;

        for l in 0..self.n_layers {
            let layer = &self.layers[l];
            let mut observe_operation = |name: &str, values: &[f32]| -> Result<(), String> {
                if detailed_layers.contains(&l)
                    && let Some(visit) = observer.as_mut()
                {
                    visit(&format!("layer_{l:02}_{name}"), values)?;
                }
                Ok(())
            };

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &x[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln1_w, &layer.ln1_b, eps);
            });
            observe_operation("ln1", &x_norm)?;

            matmul(&mut q, &x_norm, &layer.attn_q_w)?;
            matmul(&mut k, &x_norm, &layer.attn_k_w)?;
            matmul(&mut v, &x_norm, &layer.attn_v_w)?;
            q.par_chunks_mut(dim)
                .zip(k.par_chunks_mut(dim))
                .zip(v.par_chunks_mut(dim))
                .for_each(|((q_dst, k_dst), v_dst)| {
                    Self::add_bias(q_dst, &layer.attn_q_b);
                    Self::add_bias(k_dst, &layer.attn_k_b);
                    Self::add_bias(v_dst, &layer.attn_v_b);
                });
            observe_operation("query", &q)?;
            observe_operation("key", &k)?;
            observe_operation("value", &v)?;

            attention::attention(
                &mut attn_out,
                &q,
                &k,
                &v,
                n_tokens,
                self.head_count,
                self.head_dim,
                &mut attention_scratch,
            );
            observe_operation("attention", &attn_out)?;

            matmul(&mut proj_out, &attn_out, &layer.attn_out_w)?;
            proj_out
                .par_chunks_mut(dim)
                .for_each(|dst| Self::add_bias(dst, &layer.attn_out_b));
            observe_operation("attention_projected", &proj_out)?;
            for i in 0..x.len() {
                x[i] += proj_out[i];
            }
            observe_operation("attention_residual", &x)?;

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &x[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln2_w, &layer.ln2_b, eps);
            });
            observe_operation("ln2", &x_norm)?;

            matmul(&mut ffn_up_batch, &x_norm, &layer.ffn_up_w)?;
            if detailed_layers.contains(&l) {
                let mut up_with_bias = ffn_up_batch.clone();
                up_with_bias
                    .par_chunks_mut(ff_dim)
                    .for_each(|values| Self::add_bias(values, &layer.ffn_up_b));
                observe_operation("ffn_up", &up_with_bias)?;
            }
            ffn_up_batch.par_chunks_mut(ff_dim).for_each(|values| {
                Self::add_bias(values, &layer.ffn_up_b);
                for value in values {
                    *value = if use_gelu {
                        Self::gelu(*value)
                    } else {
                        Self::quick_gelu(*value)
                    };
                }
            });
            observe_operation("ffn_activated", &ffn_up_batch)?;
            matmul(&mut ffn_down_batch, &ffn_up_batch, &layer.ffn_down_w)?;
            x.par_chunks_mut(dim)
                .zip(ffn_down_batch.par_chunks_mut(dim))
                .for_each(|(destination, down)| {
                    Self::add_bias(down, &layer.ffn_down_b);
                    axpy_inplace(destination, 1.0, down);
                });
            observe_operation("ffn_down", &ffn_down_batch)?;
            if let Some(visit) = observer.as_mut() {
                visit(&format!("layer_{l:02}"), &x)?;
            }
        }

        for t in 0..n_tokens {
            let src = &x[t * dim..(t + 1) * dim];
            let dst = &mut x_norm[t * dim..(t + 1) * dim];
            self.layer_norm(dst, src, &self.post_ln_w, &self.post_ln_b);
        }
        std::mem::swap(&mut x, &mut x_norm);
        if let Some(visit) = observer.as_mut() {
            visit("post_layernorm", &x)?;
        }

        let pooled = if pre_pooled_for_speed {
            x
        } else {
            self.pool_patch_grid(&x, patch_w, patch_h)?
        };
        let n_out = pooled.len() / self.dim;
        if let Some(visit) = observer.as_mut() {
            visit("pooled", &pooled)?;
        }
        let out_dim = self.mm_input_proj_ne0;
        if self.mm_input_proj_ne1 != self.dim {
            return Err(format!(
                "mm.input_projection.weight shape mismatch at runtime: ne1={} dim={}",
                self.mm_input_proj_ne1, self.dim
            ));
        }
        let mut normed = vec![0.0f32; self.dim];
        let mut projected = vec![0.0f32; out_dim];
        let mut normed_half = vec![0u16; self.dim];

        let mut tokens: Vec<Vec<f32>> = Vec::with_capacity(n_out);
        let mut normed_stage = observer.as_ref().map(|_| Vec::with_capacity(pooled.len()));

        for out_idx in 0..n_out {
            let src = &pooled[out_idx * self.dim..(out_idx + 1) * self.dim];
            self.rms_norm_mul_weight(&mut normed, src, &self.mm_soft_emb_norm_w);
            if let Some(stage) = normed_stage.as_mut() {
                stage.extend_from_slice(&normed);
            }
            // The transposed projector keeps its GGUF storage dtype in GGML.
            // Apply its activation conversion after observing the F32 RMS output.
            if let Some(weights) = &self.mm_input_proj_f16 {
                for (dst, &src) in normed_half.iter_mut().zip(&normed) {
                    *dst = fp32_to_fp16(src);
                }
                for (dst, row) in projected.iter_mut().zip(weights.chunks_exact(self.dim)) {
                    *dst = math::dot_f16(row, &normed_half);
                }
            } else {
                if self.mm_input_proj_bf16 {
                    for value in &mut normed {
                        *value = bf16_to_fp32(fp32_to_bf16(*value));
                    }
                }
                for (out, dst) in projected.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for (inp, &v) in normed.iter().enumerate() {
                        acc += v * self.mm_input_proj_w[out + out_dim * inp];
                    }
                    *dst = acc;
                }
            }
            tokens.push(projected.clone());
        }

        if let Some(visit) = observer.as_mut() {
            visit("normalized", normed_stage.as_deref().unwrap_or_default())?;
            let projected_stage: Vec<f32> = tokens.iter().flatten().copied().collect();
            visit("projected", &projected_stage)?;
        }

        Ok(MediaEmbeddingSequence { tokens, grid: None })
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
