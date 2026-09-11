//! MiniCPM-V 4.6 vision encoder (`clip.projector_type = "minicpmv4_6"`).
//!
//! The tower is SigLIP-so400m, the same geometry the Gemma3 and Idefics3
//! encoders run. What earns this checkpoint its own backend is the merge path.
//! With `clip.vision.projector.scale_factor = 4` a windowed merger is spliced
//! into the middle of the tower: the stack runs at full patch resolution through
//! `clip.vision.wa_layer_indexes[0]`, merges 2x2, runs the remaining layers at
//! quarter resolution, and merges 2x2 once more on the way into the projector.
//! A 448x448 view therefore leaves 64 embeddings.
//!
//! Reference: `llama.cpp/tools/mtmd/models/minicpmv.cpp`,
//! `clip_graph_minicpmv4_6::build()`.

use crate::engine::io::{
    find_gguf_tensor, get_gguf_bool_from_map, get_gguf_f32_array_from_map, get_gguf_float_from_map,
    get_gguf_i64_array_from_map, get_gguf_int_from_map,
};
use crate::engine::kernels::{
    axpy_inplace, dequantize_tensor, dot_f32_simd, get_block_size, get_type_size,
};
use crate::engine::multimodal::injection::MediaEmbeddingSequence;
use crate::engine::types::{GGUFFile, Gguftensor, QuantizedTensor};
use crate::engine::vision::PreparedImageTensor;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};

use super::qwen3_asr::gelu_erf;
use super::{
    EncoderAttentionScratch, FloatBatchMatmulScratch, encoder_self_attention, matmul_encoder_batch,
};

/// Tokens merged per stage, in each spatial dimension.
const MERGE_WINDOW: usize = 2;
/// Tokens merged per stage, in total.
const MERGE_GROUP: usize = MERGE_WINDOW * MERGE_WINDOW;
/// The merge factor this encoder implements: two 2x2 stages.
const SUPPORTED_MERGE_FACTOR: usize = 4;
/// Upper bound on tokens carried through one batched tower pass, which is what
/// caps peak activation memory when many views share a shape.
const MAX_BATCH_TOKENS: usize = 16_384;

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

/// Load a projection matrix by its intended direction.
///
/// GGUF stores `ne[0]` as the input width and `ne[1]` as the output width, while
/// `QuantizedTensor` wants rows=output, cols=input. Checking both extents here
/// matters more than usual: sidecars disagree about which of `ffn_up`/`ffn_down`
/// is the expansion — SmolVLM names the expansion `ffn_down`, MiniCPM-V names it
/// `ffn_up` — and a swapped pair has a matching element count, so only an extent
/// check turns it into a load error instead of silently transposed math.
fn load_projection(
    gguf: &GGUFFile,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<QuantizedTensor, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    if tensor.n_dims < 2 {
        return Err(format!(
            "tensor {name} has {} dimension(s), expected a 2-D projection",
            tensor.n_dims
        ));
    }
    let ne0 = tensor.ne[0] as usize;
    let ne1 = tensor.ne[1] as usize;
    if ne0 != in_dim || ne1 != out_dim {
        return Err(format!(
            "tensor {name} has ne=[{ne0}, {ne1}] but this backend reads it as input={in_dim} output={out_dim}"
        ));
    }
    Ok(QuantizedTensor {
        data_offset: tensor.data_offset,
        ttype: tensor.ttype,
        rows: out_dim,
        cols: in_dim,
    })
}

#[inline]
fn layer_norm_affine(dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32], eps: f32) {
    let n = src.len();
    let mut mean = 0.0f32;
    for &v in src {
        mean += v;
    }
    mean /= n as f32;
    let mut var = 0.0f32;
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

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn quick_gelu(x: f32) -> f32 {
    let z = 1.702 * x;
    x / (1.0 + (-z).exp())
}

#[inline]
fn add_bias(v: &mut [f32], b: &[f32]) {
    for i in 0..v.len() {
        v[i] += b[i];
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
    /// Expansion, dim -> ff_dim. Named `ffn_up` in this sidecar.
    ffn_expand_w: QuantizedTensor,
    ffn_expand_b: Vec<f32>,
    /// Contraction, ff_dim -> dim. Named `ffn_down` in this sidecar.
    ffn_contract_w: QuantizedTensor,
    ffn_contract_b: Vec<f32>,
}

/// The merger spliced into the tower: windowed self-attention over each 2x2
/// block, then a 2x2 downsample MLP with a mean residual.
struct VitMerger {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_q_w: QuantizedTensor,
    attn_q_b: Vec<f32>,
    attn_k_w: QuantizedTensor,
    attn_k_b: Vec<f32>,
    attn_v_w: QuantizedTensor,
    attn_v_b: Vec<f32>,
    attn_out_w: QuantizedTensor,
    attn_out_b: Vec<f32>,
    ds_ln_w: Vec<f32>,
    ds_ln_b: Vec<f32>,
    ds_expand_w: QuantizedTensor,
    ds_expand_b: Vec<f32>,
    ds_contract_w: QuantizedTensor,
    ds_contract_b: Vec<f32>,
    ds_ff_dim: usize,
}

/// The final merger: a second 2x2 downsample straight into text embedding width.
struct Projector {
    input_norm_w: Vec<f32>,
    input_norm_b: Vec<f32>,
    up_w: QuantizedTensor,
    up_b: Vec<f32>,
    down_w: QuantizedTensor,
    down_b: Vec<f32>,
    hidden_dim: usize,
    target_dim: usize,
}

pub(crate) struct MiniCpmVVisionEncoder {
    gguf: GGUFFile,
    dim: usize,
    head_count: usize,
    head_dim: usize,
    ff_dim: usize,
    n_layers: usize,
    eps: f32,
    patch_size: usize,
    image_size: usize,
    /// Width of one merged group, `dim * MERGE_GROUP`. Both merge stages and both
    /// of their layer norms operate at this width.
    merged_dim: usize,
    /// Side of the square learned position table, 70 for this checkpoint.
    pos_grid: usize,
    /// Last tower layer that runs before the merger is applied.
    insert_layer_id: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    use_gelu: bool,
    patch_embd_w: Vec<f32>,
    patch_embd_b: Vec<f32>,
    position_embd: Vec<f32>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    layers: Vec<VisionLayer>,
    merger: VitMerger,
    projector: Projector,
}

impl MiniCpmVVisionEncoder {
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

    /// Both merge stages halve the patch grid, so a view's pixel extents must be
    /// multiples of `patch_size * 4` for an integer token count to come out.
    pub(crate) fn recommended_image_alignment(&self) -> usize {
        (self.patch_size * SUPPORTED_MERGE_FACTOR).max(1)
    }

    pub(crate) fn recommended_image_normalization(&self) -> ([f32; 3], [f32; 3]) {
        (self.image_mean, self.image_std)
    }

    /// Embeddings one prepared view projects to, without decoding its pixels.
    pub(crate) fn planned_view_tokens(&self, width: usize, height: usize) -> Result<usize, String> {
        let (pw, ph) = self.patch_grid(width, height, "<planned>")?;
        Ok((pw / SUPPORTED_MERGE_FACTOR) * (ph / SUPPORTED_MERGE_FACTOR))
    }

    fn patch_grid(
        &self,
        width: usize,
        height: usize,
        path: &str,
    ) -> Result<(usize, usize), String> {
        let align = self.recommended_image_alignment();
        if !width.is_multiple_of(align) || !height.is_multiple_of(align) {
            return Err(format!(
                "image '{path}' size {width}x{height} is not a multiple of patch_size*{SUPPORTED_MERGE_FACTOR}={align}; both 2x2 merge stages need an even patch grid"
            ));
        }
        let pw = width / self.patch_size;
        let ph = height / self.patch_size;
        if pw == 0 || ph == 0 {
            return Err(format!(
                "image '{path}' produced an empty patch grid ({pw}x{ph})"
            ));
        }
        Ok((pw, ph))
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
        let image_size = get_gguf_int_from_map(&gguf.kv, "clip.vision.image_size", 448) as usize;
        let merge_factor = get_gguf_int_from_map(&gguf.kv, "clip.vision.projector.scale_factor", 4)
            .max(0) as usize;
        let use_gelu = get_gguf_bool_from_map(&gguf.kv, "clip.use_gelu", true);
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

        if dim == 0 || head_count == 0 || ff_dim == 0 || n_layers == 0 || patch_size == 0 {
            return Err(
                "invalid minicpmv mmproj metadata: one or more required clip.vision.* keys are missing or zero"
                    .to_string(),
            );
        }
        if merge_factor != SUPPORTED_MERGE_FACTOR {
            return Err(format!(
                "unsupported MiniCPM-V merge factor {merge_factor}: this backend implements clip.vision.projector.scale_factor={SUPPORTED_MERGE_FACTOR}"
            ));
        }
        if !dim.is_multiple_of(head_count) {
            return Err(format!(
                "invalid minicpmv mmproj: dim {dim} not divisible by head_count {head_count}"
            ));
        }
        let head_dim = dim / head_count;
        let merged_dim = dim
            .checked_mul(MERGE_GROUP)
            .ok_or_else(|| "minicpmv merged dim overflow".to_string())?;

        // `clip.vision.wa_layer_indexes` names window-attention layers for other
        // families; this projector reuses the key to carry the single layer the
        // merger is spliced after.
        let insert_layer_id = get_gguf_i64_array_from_map(&gguf.kv, "clip.vision.wa_layer_indexes")
            .and_then(|values| values.first().copied())
            .ok_or_else(|| {
                "minicpmv mmproj is missing clip.vision.wa_layer_indexes, which carries the ViT merger insertion point"
                    .to_string()
            })?;
        let insert_layer_id = usize::try_from(insert_layer_id)
            .map_err(|_| format!("invalid minicpmv merger insertion point {insert_layer_id}"))?;
        if insert_layer_id + 1 >= n_layers {
            return Err(format!(
                "minicpmv merger insertion point {insert_layer_id} leaves no tower layers after the merge (block_count={n_layers})"
            ));
        }

        let patch_kernel_elems = patch_size
            .checked_mul(patch_size)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "patch kernel element count overflow".to_string())?;
        let patch_embd_w =
            load_tensor_float(&gguf, "v.patch_embd.weight", Some(patch_kernel_elems))?;
        let patch_embd_b = load_tensor_float(&gguf, "v.patch_embd.bias", Some(dim))?;

        // The learned position table is a square bucket grid, not the canonical
        // patch grid: positions are sampled from it by nearest bucket.
        let position_embd = load_tensor_float(&gguf, "v.position_embd.weight", None)?;
        if !position_embd.len().is_multiple_of(dim) {
            return Err(format!(
                "v.position_embd.weight holds {} values, not a multiple of dim {dim}",
                position_embd.len()
            ));
        }
        let pos_tokens = position_embd.len() / dim;
        let pos_grid = (pos_tokens as f64).sqrt().round() as usize;
        if pos_grid == 0 || pos_grid * pos_grid != pos_tokens {
            return Err(format!(
                "v.position_embd.weight holds {pos_tokens} positions, which is not a square bucket grid"
            ));
        }

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
                attn_q_w: load_projection(&gguf, &format!("{p}.attn_q.weight"), dim, dim)?,
                attn_q_b: load_tensor_float(&gguf, &format!("{p}.attn_q.bias"), Some(dim))?,
                attn_k_w: load_projection(&gguf, &format!("{p}.attn_k.weight"), dim, dim)?,
                attn_k_b: load_tensor_float(&gguf, &format!("{p}.attn_k.bias"), Some(dim))?,
                attn_v_w: load_projection(&gguf, &format!("{p}.attn_v.weight"), dim, dim)?,
                attn_v_b: load_tensor_float(&gguf, &format!("{p}.attn_v.bias"), Some(dim))?,
                attn_out_w: load_projection(&gguf, &format!("{p}.attn_out.weight"), dim, dim)?,
                attn_out_b: load_tensor_float(&gguf, &format!("{p}.attn_out.bias"), Some(dim))?,
                ffn_expand_w: load_projection(&gguf, &format!("{p}.ffn_up.weight"), dim, ff_dim)?,
                ffn_expand_b: load_tensor_float(&gguf, &format!("{p}.ffn_up.bias"), Some(ff_dim))?,
                ffn_contract_w: load_projection(
                    &gguf,
                    &format!("{p}.ffn_down.weight"),
                    ff_dim,
                    dim,
                )?,
                ffn_contract_b: load_tensor_float(&gguf, &format!("{p}.ffn_down.bias"), Some(dim))?,
            });
        }

        let ds_up = find_gguf_tensor(&gguf, "v.vit_merger.ds_ffn_up.weight")
            .ok_or_else(|| "tensor not found: v.vit_merger.ds_ffn_up.weight".to_string())?;
        let ds_ff_dim = ds_up.ne[1] as usize;
        if ds_ff_dim == 0 {
            return Err("v.vit_merger.ds_ffn_up.weight has an empty output width".to_string());
        }

        let merger = VitMerger {
            ln1_w: load_tensor_float(&gguf, "v.vit_merger.ln1.weight", Some(dim))?,
            ln1_b: load_tensor_float(&gguf, "v.vit_merger.ln1.bias", Some(dim))?,
            attn_q_w: load_projection(&gguf, "v.vit_merger.attn_q.weight", dim, dim)?,
            attn_q_b: load_tensor_float(&gguf, "v.vit_merger.attn_q.bias", Some(dim))?,
            attn_k_w: load_projection(&gguf, "v.vit_merger.attn_k.weight", dim, dim)?,
            attn_k_b: load_tensor_float(&gguf, "v.vit_merger.attn_k.bias", Some(dim))?,
            attn_v_w: load_projection(&gguf, "v.vit_merger.attn_v.weight", dim, dim)?,
            attn_v_b: load_tensor_float(&gguf, "v.vit_merger.attn_v.bias", Some(dim))?,
            attn_out_w: load_projection(&gguf, "v.vit_merger.attn_out.weight", dim, dim)?,
            attn_out_b: load_tensor_float(&gguf, "v.vit_merger.attn_out.bias", Some(dim))?,
            ds_ln_w: load_tensor_float(&gguf, "v.vit_merger.ds_ln.weight", Some(merged_dim))?,
            ds_ln_b: load_tensor_float(&gguf, "v.vit_merger.ds_ln.bias", Some(merged_dim))?,
            ds_expand_w: load_projection(
                &gguf,
                "v.vit_merger.ds_ffn_up.weight",
                merged_dim,
                ds_ff_dim,
            )?,
            ds_expand_b: load_tensor_float(&gguf, "v.vit_merger.ds_ffn_up.bias", Some(ds_ff_dim))?,
            ds_contract_w: load_projection(
                &gguf,
                "v.vit_merger.ds_ffn_down.weight",
                ds_ff_dim,
                dim,
            )?,
            ds_contract_b: load_tensor_float(&gguf, "v.vit_merger.ds_ffn_down.bias", Some(dim))?,
            ds_ff_dim,
        };

        let mm_up = find_gguf_tensor(&gguf, "mm.up.weight")
            .ok_or_else(|| "tensor not found: mm.up.weight".to_string())?;
        let mm_hidden_dim = mm_up.ne[1] as usize;
        if mm_hidden_dim == 0 {
            return Err("mm.up.weight has an empty output width".to_string());
        }
        let projector = Projector {
            input_norm_w: load_tensor_float(&gguf, "mm.input_norm.weight", Some(merged_dim))?,
            input_norm_b: load_tensor_float(&gguf, "mm.input_norm.bias", Some(merged_dim))?,
            up_w: load_projection(&gguf, "mm.up.weight", merged_dim, mm_hidden_dim)?,
            up_b: load_tensor_float(&gguf, "mm.up.bias", Some(mm_hidden_dim))?,
            down_w: load_projection(&gguf, "mm.down.weight", mm_hidden_dim, target_dim)?,
            down_b: load_tensor_float(&gguf, "mm.down.bias", Some(target_dim))?,
            hidden_dim: mm_hidden_dim,
            target_dim,
        };

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
            merged_dim,
            pos_grid,
            insert_layer_id,
            image_mean,
            image_std,
            use_gelu,
            patch_embd_w,
            patch_embd_b,
            position_embd,
            post_ln_w,
            post_ln_b,
            layers,
            merger,
            projector,
        })
    }

    /// Patch-embed one view into `dst`, which must hold `pw * ph` token rows.
    fn patch_embed_into(&self, dst: &mut [f32], image: &PreparedImageTensor, pw: usize, ph: usize) {
        let chw = &image.data_chw;
        let image_plane = image.width * image.height;
        let kernel_elems = 3 * self.patch_size * self.patch_size;
        let dim = self.dim;
        let patch_size = self.patch_size;
        let image_width = image.width;
        let patch_embd_b = &self.patch_embd_b;
        let patch_embd_w = &self.patch_embd_w;
        let grid = self.pos_grid;
        let position_embd = &self.position_embd;

        dst.par_chunks_mut(dim).enumerate().for_each_init(
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
                        patch_buf[patch_off..patch_off + patch_size].copy_from_slice(src);
                        patch_off += patch_size;
                    }
                }

                let mut woff = 0usize;
                for outv in out.iter_mut().take(dim) {
                    *outv += dot_f32_simd(patch_buf, &patch_embd_w[woff..woff + kernel_elems]);
                    woff += kernel_elems;
                }

                // Positions are sampled from the square bucket table by nearest
                // bucket, matching `bucket_coords` upstream. There is no
                // interpolation: neighbouring patches can share a bucket.
                let by = (grid * py) / ph;
                let bx = (grid * px) / pw;
                let off = (by * grid + bx) * dim;
                for (d, &v) in out.iter_mut().zip(&position_embd[off..off + dim]) {
                    *d += v;
                }
            },
        );
    }

    /// Run tower layers `range` over a batch of `sequences` views, each holding
    /// `tokens_per_sequence` tokens. Attention is isolated per view, so batching
    /// changes only how much work each matmul call carries.
    fn tower_forward(
        &self,
        tokens: &mut [f32],
        sequences: usize,
        tokens_per_sequence: usize,
        range: std::ops::Range<usize>,
    ) -> Result<(), String> {
        let mapped = self.gguf.mapped.as_slice();
        let dim = self.dim;
        let ff_dim = self.ff_dim;
        let eps = self.eps;
        let use_gelu = self.use_gelu;
        let n_tokens = sequences * tokens_per_sequence;

        let mut x_norm = vec![0.0f32; n_tokens * dim];
        let mut q = vec![0.0f32; n_tokens * dim];
        let mut k = vec![0.0f32; n_tokens * dim];
        let mut v = vec![0.0f32; n_tokens * dim];
        let mut attn_out = vec![0.0f32; n_tokens * dim];
        let mut proj_out = vec![0.0f32; n_tokens * dim];
        let mut intermediate = vec![0.0f32; n_tokens * ff_dim];
        let mut ffn_out = vec![0.0f32; n_tokens * dim];
        let mut batch_scratch = FloatBatchMatmulScratch::default();
        let mut attention_scratch = EncoderAttentionScratch::default();

        for l in range {
            let layer = &self.layers[l];

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                layer_norm_affine(
                    dst,
                    &tokens[t * dim..(t + 1) * dim],
                    &layer.ln1_w,
                    &layer.ln1_b,
                    eps,
                );
            });

            matmul_encoder_batch(
                &mut q,
                &x_norm,
                &layer.attn_q_w,
                mapped,
                n_tokens,
                &mut batch_scratch,
            )?;
            matmul_encoder_batch(
                &mut k,
                &x_norm,
                &layer.attn_k_w,
                mapped,
                n_tokens,
                &mut batch_scratch,
            )?;
            matmul_encoder_batch(
                &mut v,
                &x_norm,
                &layer.attn_v_w,
                mapped,
                n_tokens,
                &mut batch_scratch,
            )?;
            q.par_chunks_mut(dim)
                .zip(k.par_chunks_mut(dim))
                .zip(v.par_chunks_mut(dim))
                .for_each(|((q_dst, k_dst), v_dst)| {
                    add_bias(q_dst, &layer.attn_q_b);
                    add_bias(k_dst, &layer.attn_k_b);
                    add_bias(v_dst, &layer.attn_v_b);
                });

            encoder_self_attention(
                &mut attn_out,
                &q,
                &k,
                &v,
                sequences,
                tokens_per_sequence,
                self.head_count,
                self.head_dim,
                &mut attention_scratch,
            )?;

            matmul_encoder_batch(
                &mut proj_out,
                &attn_out,
                &layer.attn_out_w,
                mapped,
                n_tokens,
                &mut batch_scratch,
            )?;
            proj_out
                .par_chunks_mut(dim)
                .for_each(|dst| add_bias(dst, &layer.attn_out_b));
            tokens
                .par_chunks_mut(dim)
                .zip(proj_out.par_chunks(dim))
                .for_each(|(destination, residual)| {
                    axpy_inplace(destination, 1.0, residual);
                });

            x_norm.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
                layer_norm_affine(
                    dst,
                    &tokens[t * dim..(t + 1) * dim],
                    &layer.ln2_w,
                    &layer.ln2_b,
                    eps,
                );
            });

            matmul_encoder_batch(
                &mut intermediate,
                &x_norm,
                &layer.ffn_expand_w,
                mapped,
                n_tokens,
                &mut batch_scratch,
            )?;
            intermediate.par_chunks_mut(ff_dim).for_each(|hidden| {
                add_bias(hidden, &layer.ffn_expand_b);
                for value in hidden {
                    *value = if use_gelu {
                        gelu_tanh(*value)
                    } else {
                        quick_gelu(*value)
                    };
                }
            });
            matmul_encoder_batch(
                &mut ffn_out,
                &intermediate,
                &layer.ffn_contract_w,
                mapped,
                n_tokens,
                &mut batch_scratch,
            )?;
            tokens
                .par_chunks_mut(dim)
                .zip(ffn_out.par_chunks_mut(dim))
                .for_each(|(destination, output)| {
                    add_bias(output, &layer.ffn_contract_b);
                    axpy_inplace(destination, 1.0, output);
                });
        }
        Ok(())
    }

    /// Token indices in window-major order across the batch: the four tokens of
    /// each 2x2 block land contiguously, which is what lets
    /// `encoder_self_attention` treat every block as its own sequence. Upstream
    /// reaches the same result with this reorder plus an n x n block-diagonal
    /// mask; isolating sequences computes the identical softmax over the
    /// identical four keys without building the mask.
    fn window_order(views: usize, ph: usize, pw: usize) -> Vec<usize> {
        let half_h = ph / MERGE_WINDOW;
        let half_w = pw / MERGE_WINDOW;
        let per_view = ph * pw;
        let mut order = Vec::with_capacity(views * half_h * half_w * MERGE_GROUP);
        for view in 0..views {
            let base = view * per_view;
            for wi in 0..half_h {
                for wj in 0..half_w {
                    let top = base + (MERGE_WINDOW * wi) * pw + MERGE_WINDOW * wj;
                    let bottom = base + (MERGE_WINDOW * wi + 1) * pw + MERGE_WINDOW * wj;
                    order.push(top);
                    order.push(top + 1);
                    order.push(bottom);
                    order.push(bottom + 1);
                }
            }
        }
        order
    }

    /// Windowed self-attention over each 2x2 block, added back as a residual.
    fn merger_attention(
        &self,
        tokens: &mut [f32],
        views: usize,
        ph: usize,
        pw: usize,
    ) -> Result<(), String> {
        let mapped = self.gguf.mapped.as_slice();
        let dim = self.dim;
        let order = Self::window_order(views, ph, pw);
        let n_windowed = order.len();
        let n_windows = n_windowed / MERGE_GROUP;
        let merger = &self.merger;
        let eps = self.eps;

        // Normalise and gather into window-major order in one pass.
        let mut x_norm = vec![0.0f32; n_windowed * dim];
        x_norm
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(slot, dst)| {
                let token = order[slot];
                layer_norm_affine(
                    dst,
                    &tokens[token * dim..(token + 1) * dim],
                    &merger.ln1_w,
                    &merger.ln1_b,
                    eps,
                );
            });

        let mut q = vec![0.0f32; n_windowed * dim];
        let mut k = vec![0.0f32; n_windowed * dim];
        let mut v = vec![0.0f32; n_windowed * dim];
        let mut batch_scratch = FloatBatchMatmulScratch::default();
        matmul_encoder_batch(
            &mut q,
            &x_norm,
            &merger.attn_q_w,
            mapped,
            n_windowed,
            &mut batch_scratch,
        )?;
        matmul_encoder_batch(
            &mut k,
            &x_norm,
            &merger.attn_k_w,
            mapped,
            n_windowed,
            &mut batch_scratch,
        )?;
        matmul_encoder_batch(
            &mut v,
            &x_norm,
            &merger.attn_v_w,
            mapped,
            n_windowed,
            &mut batch_scratch,
        )?;
        q.par_chunks_mut(dim)
            .zip(k.par_chunks_mut(dim))
            .zip(v.par_chunks_mut(dim))
            .for_each(|((q_dst, k_dst), v_dst)| {
                add_bias(q_dst, &merger.attn_q_b);
                add_bias(k_dst, &merger.attn_k_b);
                add_bias(v_dst, &merger.attn_v_b);
            });

        let mut attn_out = vec![0.0f32; n_windowed * dim];
        let mut attention_scratch = EncoderAttentionScratch::default();
        encoder_self_attention(
            &mut attn_out,
            &q,
            &k,
            &v,
            n_windows,
            MERGE_GROUP,
            self.head_count,
            self.head_dim,
            &mut attention_scratch,
        )?;

        let mut proj_out = vec![0.0f32; n_windowed * dim];
        matmul_encoder_batch(
            &mut proj_out,
            &attn_out,
            &merger.attn_out_w,
            mapped,
            n_windowed,
            &mut batch_scratch,
        )?;

        // Scatter back to row-major order, adding the residual in place.
        for (slot, &token) in order.iter().enumerate() {
            let src = &proj_out[slot * dim..(slot + 1) * dim];
            let dst = &mut tokens[token * dim..(token + 1) * dim];
            for i in 0..dim {
                dst[i] += src[i] + merger.attn_out_b[i];
            }
        }
        Ok(())
    }

    /// Gather each 2x2 block into one concatenated row across the batch, and
    /// return the mean of the four source tokens alongside it.
    fn gather_merge_groups(
        &self,
        tokens: &[f32],
        views: usize,
        ph: usize,
        pw: usize,
        want_mean: bool,
    ) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let dim = self.dim;
        let merged_dim = self.merged_dim;
        let out_h = ph / MERGE_WINDOW;
        let out_w = pw / MERGE_WINDOW;
        let per_view_out = out_h * out_w;
        let per_view_in = ph * pw;
        let n_out = views * per_view_out;

        let mut merged = vec![0.0f32; n_out * merged_dim];
        let mut mean = vec![0.0f32; if want_mean { n_out * dim } else { 0 }];

        let sources = |out_idx: usize| {
            let view = out_idx / per_view_out;
            let local = out_idx % per_view_out;
            let oy = local / out_w;
            let ox = local % out_w;
            let base = view * per_view_in;
            let top = base + (MERGE_WINDOW * oy) * pw + MERGE_WINDOW * ox;
            let bottom = base + (MERGE_WINDOW * oy + 1) * pw + MERGE_WINDOW * ox;
            [top, top + 1, bottom, bottom + 1]
        };

        if want_mean {
            merged
                .par_chunks_mut(merged_dim)
                .zip(mean.par_chunks_mut(dim))
                .enumerate()
                .for_each(|(out_idx, (cat, avg))| {
                    for (slot, token) in sources(out_idx).into_iter().enumerate() {
                        let src = &tokens[token * dim..(token + 1) * dim];
                        cat[slot * dim..(slot + 1) * dim].copy_from_slice(src);
                        for (a, &s) in avg.iter_mut().zip(src) {
                            *a += s;
                        }
                    }
                    for a in avg.iter_mut() {
                        *a *= 1.0 / MERGE_GROUP as f32;
                    }
                });
        } else {
            merged
                .par_chunks_mut(merged_dim)
                .enumerate()
                .for_each(|(out_idx, cat)| {
                    for (slot, token) in sources(out_idx).into_iter().enumerate() {
                        let src = &tokens[token * dim..(token + 1) * dim];
                        cat[slot * dim..(slot + 1) * dim].copy_from_slice(src);
                    }
                });
        }

        (merged, mean, out_h, out_w)
    }

    /// 2x2 downsample MLP with the mean of the merged tokens as its residual.
    fn merger_downsample(
        &self,
        tokens: &[f32],
        views: usize,
        ph: usize,
        pw: usize,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        let mapped = self.gguf.mapped.as_slice();
        let dim = self.dim;
        let merged_dim = self.merged_dim;
        let merger = &self.merger;
        let eps = self.eps;

        let (merged, mean, out_h, out_w) = self.gather_merge_groups(tokens, views, ph, pw, true);
        let n_out = views * out_h * out_w;

        let mut normed = vec![0.0f32; n_out * merged_dim];
        normed
            .par_chunks_mut(merged_dim)
            .enumerate()
            .for_each(|(out_idx, dst)| {
                layer_norm_affine(
                    dst,
                    &merged[out_idx * merged_dim..(out_idx + 1) * merged_dim],
                    &merger.ds_ln_w,
                    &merger.ds_ln_b,
                    eps,
                );
            });

        let mut batch_scratch = FloatBatchMatmulScratch::default();
        let mut hidden = vec![0.0f32; n_out * merger.ds_ff_dim];
        matmul_encoder_batch(
            &mut hidden,
            &normed,
            &merger.ds_expand_w,
            mapped,
            n_out,
            &mut batch_scratch,
        )?;
        // The downsample MLP is gelu_pytorch_tanh upstream, unlike the final
        // merger below.
        hidden.par_chunks_mut(merger.ds_ff_dim).for_each(|chunk| {
            add_bias(chunk, &merger.ds_expand_b);
            for value in chunk {
                *value = gelu_tanh(*value);
            }
        });

        let mut out = vec![0.0f32; n_out * dim];
        matmul_encoder_batch(
            &mut out,
            &hidden,
            &merger.ds_contract_w,
            mapped,
            n_out,
            &mut batch_scratch,
        )?;
        out.par_chunks_mut(dim)
            .zip(mean.par_chunks(dim))
            .for_each(|(dst, avg)| {
                add_bias(dst, &merger.ds_contract_b);
                for (d, &m) in dst.iter_mut().zip(avg) {
                    *d += m;
                }
            });

        Ok((out, out_h, out_w))
    }

    /// The final 2x2 merge straight into text embedding width. No residual.
    fn project(
        &self,
        tokens: &[f32],
        views: usize,
        ph: usize,
        pw: usize,
    ) -> Result<Vec<f32>, String> {
        let mapped = self.gguf.mapped.as_slice();
        let merged_dim = self.merged_dim;
        let projector = &self.projector;
        let eps = self.eps;

        let (merged, _, out_h, out_w) = self.gather_merge_groups(tokens, views, ph, pw, false);
        let n_out = views * out_h * out_w;

        let mut normed = vec![0.0f32; n_out * merged_dim];
        normed
            .par_chunks_mut(merged_dim)
            .enumerate()
            .for_each(|(out_idx, dst)| {
                layer_norm_affine(
                    dst,
                    &merged[out_idx * merged_dim..(out_idx + 1) * merged_dim],
                    &projector.input_norm_w,
                    &projector.input_norm_b,
                    eps,
                );
            });

        let mut batch_scratch = FloatBatchMatmulScratch::default();
        let mut hidden = vec![0.0f32; n_out * projector.hidden_dim];
        matmul_encoder_batch(
            &mut hidden,
            &normed,
            &projector.up_w,
            mapped,
            n_out,
            &mut batch_scratch,
        )?;
        // `nn.GELU()` upstream, so the erf form rather than the tanh form used
        // by the tower and the downsample MLP.
        hidden
            .par_chunks_mut(projector.hidden_dim)
            .for_each(|chunk| {
                add_bias(chunk, &projector.up_b);
                for value in chunk {
                    *value = gelu_erf(*value);
                }
            });

        let mut out = vec![0.0f32; n_out * projector.target_dim];
        matmul_encoder_batch(
            &mut out,
            &hidden,
            &projector.down_w,
            mapped,
            n_out,
            &mut batch_scratch,
        )?;
        out.par_chunks_mut(projector.target_dim)
            .for_each(|dst| add_bias(dst, &projector.down_b));

        Ok(out)
    }

    /// Encode views that share a patch grid as one batch.
    fn encode_batch(
        &self,
        images: &[&PreparedImageTensor],
        pw: usize,
        ph: usize,
    ) -> Result<Vec<MediaEmbeddingSequence>, String> {
        let views = images.len();
        let per_view = pw * ph;
        let dim = self.dim;

        let mut tokens = vec![0.0f32; views * per_view * dim];
        for (index, image) in images.iter().enumerate() {
            let start = index * per_view * dim;
            self.patch_embed_into(&mut tokens[start..start + per_view * dim], image, pw, ph);
        }

        self.tower_forward(&mut tokens, views, per_view, 0..self.insert_layer_id + 1)?;
        self.merger_attention(&mut tokens, views, ph, pw)?;
        let (mut tokens, ph, pw) = self.merger_downsample(&tokens, views, ph, pw)?;
        self.tower_forward(
            &mut tokens,
            views,
            ph * pw,
            self.insert_layer_id + 1..self.n_layers,
        )?;

        let eps = self.eps;
        let post_ln_w = &self.post_ln_w;
        let post_ln_b = &self.post_ln_b;
        let mut normed = vec![0.0f32; tokens.len()];
        normed.par_chunks_mut(dim).enumerate().for_each(|(t, dst)| {
            layer_norm_affine(
                dst,
                &tokens[t * dim..(t + 1) * dim],
                post_ln_w,
                post_ln_b,
                eps,
            );
        });

        let projected = self.project(&normed, views, ph, pw)?;
        let target_dim = self.projector.target_dim;
        let per_view_out = projected.len() / views.max(1) / target_dim;
        Ok((0..views)
            .map(|view| {
                let start = view * per_view_out * target_dim;
                let end = start + per_view_out * target_dim;
                MediaEmbeddingSequence {
                    grid: None,
                    tokens: projected[start..end]
                        .chunks_exact(target_dim)
                        .map(<[f32]>::to_vec)
                        .collect(),
                }
            })
            .collect())
    }

    pub(crate) fn encode_images(
        &self,
        images: &[PreparedImageTensor],
    ) -> Result<Vec<MediaEmbeddingSequence>, String> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // Views of one shape encode together. Every matmul dequantises its whole
        // weight matrix, so one pass over ten views does that work once instead
        // of ten times, and hands Accelerate a correspondingly larger GEMM.
        let mut groups: Vec<((usize, usize), Vec<usize>)> = Vec::new();
        for (index, image) in images.iter().enumerate() {
            let shape = (image.width, image.height);
            match groups.iter_mut().find(|(key, _)| *key == shape) {
                Some((_, members)) => members.push(index),
                None => groups.push((shape, vec![index])),
            }
        }

        let mut encoded: Vec<Option<MediaEmbeddingSequence>> =
            (0..images.len()).map(|_| None).collect();
        for (shape, members) in groups {
            let (pw, ph) = self.patch_grid(shape.0, shape.1, &images[members[0]].path)?;
            let per_view = pw.checked_mul(ph).filter(|&n| n > 0).ok_or_else(|| {
                format!(
                    "image '{}' produced an empty patch grid",
                    images[members[0]].path
                )
            })?;
            // Bound peak activation memory rather than the view count, so a
            // larger grid shrinks the batch instead of growing the buffers.
            let per_batch = (MAX_BATCH_TOKENS / per_view).max(1);
            for chunk in members.chunks(per_batch) {
                let batch: Vec<&PreparedImageTensor> =
                    chunk.iter().map(|&index| &images[index]).collect();
                for (slot, sequence) in chunk.iter().zip(self.encode_batch(&batch, pw, ph)?) {
                    encoded[*slot] = Some(sequence);
                }
            }
        }

        encoded
            .into_iter()
            .map(|sequence| sequence.ok_or_else(|| "image view was not encoded".to_string()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{MERGE_GROUP, MiniCpmVVisionEncoder};

    #[test]
    fn window_order_groups_each_2x2_block_contiguously() {
        // A 4x4 patch grid holds four 2x2 windows.
        let order = MiniCpmVVisionEncoder::window_order(1, 4, 4);

        assert_eq!(order.len(), 16);
        // First window: the top-left 2x2 block, row-major inside the block.
        assert_eq!(&order[0..MERGE_GROUP], &[0, 1, 4, 5]);
        // Second window sits beside it, not below.
        assert_eq!(&order[MERGE_GROUP..2 * MERGE_GROUP], &[2, 3, 6, 7]);
        // Last window is the bottom-right block.
        assert_eq!(&order[12..16], &[10, 11, 14, 15]);
    }

    #[test]
    fn window_order_offsets_each_view_in_a_batch() {
        // Two views of a 4x4 grid: the second view repeats the first view's
        // pattern shifted by one view's worth of tokens, so batching cannot
        // leak attention across views.
        let single = MiniCpmVVisionEncoder::window_order(1, 4, 4);
        let batched = MiniCpmVVisionEncoder::window_order(2, 4, 4);

        assert_eq!(batched.len(), single.len() * 2);
        assert_eq!(&batched[..single.len()], &single[..]);
        let shifted: Vec<usize> = single.iter().map(|index| index + 16).collect();
        assert_eq!(&batched[single.len()..], &shifted[..]);
    }

    #[test]
    fn window_order_visits_every_token_exactly_once() {
        let order = MiniCpmVVisionEncoder::window_order(1, 8, 6);
        let mut seen = order.clone();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(order.len(), 48);
        assert_eq!(seen.len(), 48);
        assert_eq!(seen.first().copied(), Some(0));
        assert_eq!(seen.last().copied(), Some(47));
    }
}
