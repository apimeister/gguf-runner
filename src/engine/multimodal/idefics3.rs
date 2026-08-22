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
    let mut n = 1usize;
    for i in 0..tensor.n_dims as usize {
        n = n.saturating_mul(tensor.ne[i] as usize);
    }
    n
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

// Load a quantized tensor for matmul_quantized.
// For idefics3 projection weights (F16, ne[0]=input_dim, ne[1]=output_dim in GGML convention):
//   pass rows=output_dim=ne[1], cols=input_dim=ne[0]  (transposed storage, see gemma3.rs rationale)
// For ViT transformer weights (Q4_0/F16, ne[0]=output_dim, ne[1]=input_dim):
//   pass rows=output_dim=ne[0], cols=input_dim=ne[1]
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
        .ok_or_else(|| format!("shape overflow loading {name}"))?;
    if n_elements != expected {
        return Err(format!(
            "tensor {name} shape mismatch: {n_elements} elements, expected {rows}×{cols}={expected}"
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
    let n = src.len();
    let mut mean = 0.0f32;
    let mut var = 0.0f32;
    for &v in src {
        mean += v;
    }
    mean /= n as f32;
    for &v in src {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f32;
    let inv = 1.0f32 / (var + eps).sqrt();
    for i in 0..n {
        dst[i] = (src[i] - mean) * inv * w[i] + b[i];
    }
}

struct VisionLayer {
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

pub(crate) struct Idefics3VisionEncoder {
    gguf: GGUFFile,
    dim: usize,
    head_count: usize,
    head_dim: usize,
    ff_dim: usize,
    n_layers: usize,
    eps: f32,
    patch_size: usize,
    image_size: usize,
    scale_factor: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    use_gelu: bool,
    patch_embd_w: Vec<f32>,
    patch_embd_b: Vec<f32>,
    position_embd: Vec<f32>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    // Idefics3 projector: pixel-shuffle + linear fc
    // Weight stored as [input_dim=dim*sf^2, output_dim=target_dim] (GGML column-major).
    // Loaded as QuantizedTensor with rows=target_dim, cols=dim*sf^2.
    proj_fc_w: QuantizedTensor,
    proj_target_dim: usize,
    layers: Vec<VisionLayer>,
}

impl Idefics3VisionEncoder {
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
        self.image_size
    }

    pub(crate) fn recommended_image_alignment(&self) -> usize {
        self.patch_size.max(1)
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
        let patch_size = get_gguf_int_from_map(&gguf.kv, "clip.vision.patch_size", 16) as usize;
        let image_size = get_gguf_int_from_map(&gguf.kv, "clip.vision.image_size", 512) as usize;
        let scale_factor =
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
            || scale_factor == 0
        {
            return Err(
                "invalid idefics3 mmproj metadata: one or more required clip.vision.* keys are missing/zero"
                    .to_string(),
            );
        }
        if !dim.is_multiple_of(head_count) {
            return Err(format!(
                "invalid idefics3 mmproj: dim {dim} not divisible by head_count {head_count}"
            ));
        }
        if !image_size.is_multiple_of(patch_size) {
            return Err(format!(
                "invalid idefics3 mmproj: image_size {image_size} not divisible by patch_size {patch_size}"
            ));
        }
        let head_dim = dim / head_count;

        // Patch embedding: patch_size×patch_size×3 → dim
        let patch_kernel_elems = patch_size
            .checked_mul(patch_size)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "patch kernel element count overflow".to_string())?;

        // Position embedding: (image_size/patch_size)^2 positions × dim
        let base_patch_grid = image_size / patch_size;
        let base_pos_tokens = base_patch_grid * base_patch_grid;

        let patch_embd_w =
            load_tensor_float(&gguf, "v.patch_embd.weight", Some(patch_kernel_elems))?;
        let patch_embd_b = load_tensor_float(&gguf, "v.patch_embd.bias", Some(dim))?;
        let position_embd =
            load_tensor_float(&gguf, "v.position_embd.weight", Some(base_pos_tokens * dim))?;

        // Projection FC: maps dim*scale_factor^2 → target_dim.
        // GGUF stores it as ne[0]=dim*sf^2 (input, fast), ne[1]=target_dim (output, slow).
        // Load as rows=target_dim (output), cols=dim*sf^2 (input).
        let expanded_dim = dim
            .checked_mul(scale_factor)
            .and_then(|v| v.checked_mul(scale_factor))
            .ok_or_else(|| "idefics3 expanded dim overflow".to_string())?;

        let fc_tensor = find_gguf_tensor(&gguf, "mm.model.fc.weight")
            .ok_or_else(|| "tensor not found: mm.model.fc.weight".to_string())?;
        let fc_ne0 = fc_tensor.ne[0] as usize;
        let fc_ne1 = fc_tensor.ne[1] as usize;
        if fc_ne0 != expanded_dim {
            return Err(format!(
                "mm.model.fc.weight ne[0]={fc_ne0}, expected expanded_dim={expanded_dim} (vision_dim={dim}×scale_factor^2={sf_sq})",
                sf_sq = scale_factor * scale_factor
            ));
        }
        if fc_ne1 != target_dim {
            return Err(format!(
                "mm.model.fc.weight ne[1]={fc_ne1}, expected target_dim={target_dim} (text embedding dim)"
            ));
        }
        // rows=ne[1]=target_dim, cols=ne[0]=expanded_dim
        let proj_fc_w =
            load_tensor_quantized(&gguf, "mm.model.fc.weight", target_dim, expanded_dim)?;

        let post_ln_w = load_tensor_float(&gguf, "v.post_ln.weight", Some(dim))?;
        let post_ln_b = load_tensor_float(&gguf, "v.post_ln.bias", Some(dim))?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("v.blk.{l}");
            layers.push(VisionLayer {
                ln1_w: load_tensor_float(&gguf, &format!("{p}.ln1.weight"), Some(dim))?,
                ln1_b: load_tensor_float(&gguf, &format!("{p}.ln1.bias"), Some(dim))?,
                ln2_w: load_tensor_float(&gguf, &format!("{p}.ln2.weight"), Some(dim))?,
                ln2_b: load_tensor_float(&gguf, &format!("{p}.ln2.bias"), Some(dim))?,
                attn_q_w: load_tensor_quantized(&gguf, &format!("{p}.attn_q.weight"), dim, dim)?,
                attn_q_b: load_tensor_float(&gguf, &format!("{p}.attn_q.bias"), Some(dim))?,
                attn_k_w: load_tensor_quantized(&gguf, &format!("{p}.attn_k.weight"), dim, dim)?,
                attn_k_b: load_tensor_float(&gguf, &format!("{p}.attn_k.bias"), Some(dim))?,
                attn_v_w: load_tensor_quantized(&gguf, &format!("{p}.attn_v.weight"), dim, dim)?,
                attn_v_b: load_tensor_float(&gguf, &format!("{p}.attn_v.bias"), Some(dim))?,
                attn_out_w: load_tensor_quantized(
                    &gguf,
                    &format!("{p}.attn_out.weight"),
                    dim,
                    dim,
                )?,
                attn_out_b: load_tensor_float(&gguf, &format!("{p}.attn_out.bias"), Some(dim))?,
                // In this GGUF convention ffn_down is the expansion (dim→ff_dim)
                // and ffn_up is the contraction (ff_dim→dim).
                ffn_up_w: load_tensor_quantized(&gguf, &format!("{p}.ffn_up.weight"), dim, ff_dim)?,
                ffn_up_b: load_tensor_float(&gguf, &format!("{p}.ffn_up.bias"), Some(dim))?,
                ffn_down_w: load_tensor_quantized(
                    &gguf,
                    &format!("{p}.ffn_down.weight"),
                    ff_dim,
                    dim,
                )?,
                ffn_down_b: load_tensor_float(&gguf, &format!("{p}.ffn_down.bias"), Some(ff_dim))?,
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
            image_size,
            scale_factor,
            image_mean,
            image_std,
            use_gelu,
            patch_embd_w,
            patch_embd_b,
            position_embd,
            post_ln_w,
            post_ln_b,
            proj_fc_w,
            proj_target_dim: target_dim,
            layers,
        })
    }

    #[inline]
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
    }

    #[inline]
    fn quick_gelu(x: f32) -> f32 {
        let z = 1.702 * x;
        x / (1.0 + (-z).exp())
    }

    fn add_bias(v: &mut [f32], b: &[f32]) {
        for i in 0..v.len() {
            v[i] += b[i];
        }
    }

    fn patch_embed_and_add_position(
        &self,
        image: &PreparedImageTensor,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        if !image.width.is_multiple_of(self.patch_size)
            || !image.height.is_multiple_of(self.patch_size)
        {
            return Err(format!(
                "image '{}' size {}×{} not divisible by patch_size {}",
                image.path, image.width, image.height, self.patch_size
            ));
        }
        let pw = image.width / self.patch_size;
        let ph = image.height / self.patch_size;
        if pw == 0 || ph == 0 {
            return Err(format!(
                "image '{}' produced empty patch grid ({pw}×{ph})",
                image.path
            ));
        }
        let patch_count = pw
            .checked_mul(ph)
            .ok_or_else(|| "patch count overflow".to_string())?;

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

        // Add positional embeddings — SmolVLM uses fixed (non-interpolated) positional embeddings.
        // The mmproj stores positions for the canonical grid (image_size/patch_size)^2.
        // When the actual image matches (width==height==image_size), positions map 1:1.
        // For other sizes we bilinearly interpolate as a graceful fallback.
        let base_grid = self.image_size / self.patch_size;
        for py in 0..ph {
            for px in 0..pw {
                let tok = py * pw + px;
                let tok_off = tok * self.dim;
                let dst = &mut tokens[tok_off..tok_off + self.dim];
                self.add_position_embd(py, px, ph, pw, base_grid, dst);
            }
        }

        Ok((tokens, pw, ph))
    }

    fn add_position_embd(
        &self,
        y: usize,
        x: usize,
        out_h: usize,
        out_w: usize,
        base_grid: usize,
        dst: &mut [f32],
    ) {
        if out_h == base_grid && out_w == base_grid {
            // Exact match — direct lookup.
            let off = (y * base_grid + x) * self.dim;
            for (d, &v) in dst.iter_mut().zip(&self.position_embd[off..off + self.dim]) {
                *d += v;
            }
            return;
        }
        // Bilinear interpolation for non-canonical sizes.
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

    fn vit_forward(&self, tokens: &mut [f32], n_tokens: usize) -> Result<(), String> {
        let mapped = self.gguf.mapped.as_slice();
        let dim = self.dim;
        let ff_dim = self.ff_dim;
        let eps = self.eps;
        let use_gelu = self.use_gelu;

        let mut x_norm = vec![0.0f32; n_tokens * dim];
        let mut q = vec![0.0f32; n_tokens * dim];
        let mut k = vec![0.0f32; n_tokens * dim];
        let mut v = vec![0.0f32; n_tokens * dim];
        let head_token_stride = n_tokens * self.head_dim;
        let mut attn_head_major = vec![0.0f32; self.head_count * head_token_stride];
        let mut attn_out = vec![0.0f32; n_tokens * dim];
        let mut proj_out = vec![0.0f32; n_tokens * dim];

        for l in 0..self.n_layers {
            let layer = &self.layers[l];

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &tokens[t * dim..(t + 1) * dim];
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
                        scale_slice_inplace(out, 1.0 / score_sum);
                    }
                });

            for t in 0..n_tokens {
                let dst = &mut attn_out[t * dim..(t + 1) * dim];
                for h in 0..self.head_count {
                    let src = &attn_head_major[h * head_token_stride + t * head_dim
                        ..h * head_token_stride + (t + 1) * head_dim];
                    let off = h * head_dim;
                    dst[off..off + head_dim].copy_from_slice(src);
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
            for i in 0..tokens.len() {
                tokens[i] += proj_out[i];
            }

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                let src = &tokens[t * dim..(t + 1) * dim];
                layer_norm_affine(dst, src, &layer.ln2_w, &layer.ln2_b, eps);
            });

            tokens.par_chunks_mut(dim).enumerate().try_for_each_init(
                // intermediate=ff_dim (expand stage), output=dim (contract stage)
                || (vec![0.0f32; ff_dim], vec![0.0f32; dim]),
                |(intermediate, output), (t, dst)| -> Result<(), String> {
                    let src = &x_norm[t * dim..(t + 1) * dim];
                    // Expand: dim → ff_dim (ffn_down in this GGUF's naming)
                    matmul_quantized(intermediate, src, &layer.ffn_down_w, mapped)?;
                    Self::add_bias(intermediate, &layer.ffn_down_b);
                    for v in intermediate.iter_mut() {
                        *v = if use_gelu {
                            Self::gelu(*v)
                        } else {
                            Self::quick_gelu(*v)
                        };
                    }
                    // Contract: ff_dim → dim (ffn_up in this GGUF's naming)
                    matmul_quantized(output, intermediate, &layer.ffn_up_w, mapped)?;
                    Self::add_bias(output, &layer.ffn_up_b);
                    axpy_inplace(dst, 1.0, output);
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    // Idefics3 pixel shuffle: group scale_factor×scale_factor patches and concatenate.
    // Input:  [ph×pw, dim]  (1024 patches × 768 for 512-image with patch_size=16)
    // Output: [(ph/sf)×(pw/sf), dim*sf*sf]  (64 tokens × 12288)
    fn pixel_shuffle(&self, tokens: &[f32], ph: usize, pw: usize) -> Result<Vec<f32>, String> {
        let sf = self.scale_factor;
        if !ph.is_multiple_of(sf) || !pw.is_multiple_of(sf) {
            return Err(format!(
                "idefics3 pixel_shuffle requires patch grid divisible by scale_factor {sf} (got {ph}×{pw})"
            ));
        }
        let out_h = ph / sf;
        let out_w = pw / sf;
        let n_out = out_h * out_w;
        let in_dim = self.dim;
        let out_dim = in_dim * sf * sf;
        let mut out = vec![0.0f32; n_out * out_dim];

        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(out_idx, dst)| {
                let oy = out_idx / out_w;
                let ox = out_idx % out_w;
                let mut off = 0usize;
                for dy in 0..sf {
                    for dx in 0..sf {
                        let iy = oy * sf + dy;
                        let ix = ox * sf + dx;
                        let src_idx = iy * pw + ix;
                        let src = &tokens[src_idx * in_dim..(src_idx + 1) * in_dim];
                        dst[off..off + in_dim].copy_from_slice(src);
                        off += in_dim;
                    }
                }
            });

        Ok(out)
    }

    fn encode_single_image(
        &self,
        image: &PreparedImageTensor,
    ) -> Result<MediaEmbeddingSequence, String> {
        let mapped = self.gguf.mapped.as_slice();

        // 1. Patch embed + positional embeddings
        let (mut tokens, pw, ph) = self.patch_embed_and_add_position(image)?;
        let n_tokens = tokens.len() / self.dim;

        // 2. ViT transformer blocks
        self.vit_forward(&mut tokens, n_tokens)?;

        // 2b. Post-ViT layer norm
        let mut normed = vec![0.0f32; tokens.len()];
        let dim = self.dim;
        let post_ln_w = &self.post_ln_w;
        let post_ln_b = &self.post_ln_b;
        let eps = self.eps;
        normed.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
            let src = &tokens[t * dim..(t + 1) * dim];
            layer_norm_affine(dst, src, post_ln_w, post_ln_b, eps);
        });
        let tokens = normed;

        // 3. Idefics3 pixel shuffle: [n_patches, dim] → [n_out, dim*sf^2]
        let shuffled = self.pixel_shuffle(&tokens, ph, pw)?;
        // n_out_tokens = shuffled.len() / expanded_dim where expanded_dim = proj_fc_w.cols
        let expanded_dim = self.proj_fc_w.cols;
        let n_out_tokens = shuffled.len() / expanded_dim;

        // 4. Linear projection: [n_out, expanded_dim] → [n_out, target_dim]
        let target_dim = self.proj_target_dim;
        let mut projected = vec![0.0f32; target_dim];
        let mut result_tokens: Vec<Vec<f32>> = Vec::with_capacity(n_out_tokens);

        for t in 0..n_out_tokens {
            let src = &shuffled[t * expanded_dim..(t + 1) * expanded_dim];
            projected.fill(0.0);
            matmul_quantized(&mut projected, src, &self.proj_fc_w, mapped)?;
            result_tokens.push(projected.clone());
        }

        Ok(MediaEmbeddingSequence {
            tokens: result_tokens,
        })
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
