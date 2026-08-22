use crate::engine::audio::{PreparedAudioFeatureWindow, PreparedAudioFeatureWindowPlan};
use crate::engine::io::{
    find_gguf_tensor, fp16_to_fp32, get_gguf_bool_from_map, get_gguf_float_from_map,
    get_gguf_int_from_map, get_gguf_string_from_map,
};
use crate::engine::kernels::{
    axpy_inplace, dequantize_tensor, dot_f32_simd, get_block_size, get_type_size, matmul_quantized,
    scale_slice_inplace,
};
use crate::engine::types::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGUFFile, GgmlType, Gguftensor, QuantizedTensor,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};

use super::{AudioEncoderFrontendOutput, MediaEmbeddingSequence};

const CONV_KERNEL: usize = 3;
const CONV_STRIDE: usize = 2;
const CONV_PADDING: usize = 1;
const CONV_LAYER_COUNT: usize = 3;
const MEL_FRAMES_PER_CHUNK: usize = 100;
const TOKENS_PER_CHUNK: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Qwen3AsrAudioConfig {
    pub(crate) embedding_dim: usize,
    pub(crate) convolution_channel_count: usize,
    pub(crate) feed_forward_dim: usize,
    pub(crate) layer_count: usize,
    pub(crate) head_count: usize,
    pub(crate) head_dim: usize,
    pub(crate) layer_norm_epsilon: f32,
    pub(crate) mel_bin_count: usize,
    pub(crate) projection_dim: usize,
    pub(crate) position_count: usize,
    pub(crate) projector_hidden_dim: usize,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct TensorWeight {
    name: String,
    data_offset: usize,
    ttype: GgmlType,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct AudioTransformerLayerWeights {
    ln1_weight: TensorWeight,
    ln1_bias: Option<TensorWeight>,
    ln2_weight: TensorWeight,
    ln2_bias: Option<TensorWeight>,
    query_weight: TensorWeight,
    query_bias: Option<TensorWeight>,
    key_weight: TensorWeight,
    key_bias: Option<TensorWeight>,
    value_weight: TensorWeight,
    value_bias: Option<TensorWeight>,
    output_weight: TensorWeight,
    output_bias: Option<TensorWeight>,
    feed_forward_up_weight: TensorWeight,
    feed_forward_up_bias: Option<TensorWeight>,
    feed_forward_down_weight: TensorWeight,
    feed_forward_down_bias: Option<TensorWeight>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Qwen3AsrWeights {
    position_embeddings: TensorWeight,
    conv2d_weights: [TensorWeight; CONV_LAYER_COUNT],
    conv2d_biases: [TensorWeight; CONV_LAYER_COUNT],
    conv_output_weight: TensorWeight,
    layers: Vec<AudioTransformerLayerWeights>,
    post_layer_norm_weight: TensorWeight,
    post_layer_norm_bias: TensorWeight,
    projector_up_weight: TensorWeight,
    projector_up_bias: TensorWeight,
    projector_down_weight: TensorWeight,
    projector_down_bias: TensorWeight,
}

#[derive(Clone, Debug)]
struct Conv2dLayerWeights {
    kernel: Vec<f32>,
    bias: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel_type: GgmlType,
}

#[derive(Clone)]
struct ConvFrontendWeights {
    layers: [Conv2dLayerWeights; CONV_LAYER_COUNT],
    output_projection: QuantizedTensor,
}

#[derive(Clone)]
struct AudioTransformerLayer {
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
    query_weight: QuantizedTensor,
    query_bias: Vec<f32>,
    key_weight: QuantizedTensor,
    key_bias: Vec<f32>,
    value_weight: QuantizedTensor,
    value_bias: Vec<f32>,
    output_weight: QuantizedTensor,
    output_bias: Vec<f32>,
    feed_forward_up_weight: QuantizedTensor,
    feed_forward_up_bias: Vec<f32>,
    feed_forward_down_weight: QuantizedTensor,
    feed_forward_down_bias: Vec<f32>,
}

#[derive(Clone)]
struct AudioTransformerWeights {
    position_embeddings: Vec<f32>,
    layers: Vec<AudioTransformerLayer>,
}

#[derive(Clone)]
struct AudioPostLayerNormWeights {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone)]
struct AudioProjectorWeights {
    up_weight: QuantizedTensor,
    up_bias: Vec<f32>,
    down_weight: QuantizedTensor,
    down_bias: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ConvFeatureMap {
    batch_count: usize,
    width: usize,
    height: usize,
    channels: usize,
    /// Patch-major storage: `[batch][height][width][channels]`.
    data: Vec<f32>,
}

pub(crate) struct Qwen3AsrAudioEncoder {
    // Keep the mapped sidecar alive for quantized runtime weights and descriptor diagnostics.
    gguf: GGUFFile,
    config: Qwen3AsrAudioConfig,
    #[allow(dead_code)]
    weights: Qwen3AsrWeights,
    conv_frontend: ConvFrontendWeights,
    transformer: AudioTransformerWeights,
    post_layer_norm: AudioPostLayerNormWeights,
    projector: AudioProjectorWeights,
}

#[cfg(unix)]
#[cfg_attr(not(target_vendor = "apple"), link(name = "m"))]
unsafe extern "C" {
    #[link_name = "erff"]
    fn system_erff(value: f32) -> f32;
}

#[inline]
fn erf_f32(value: f32) -> f32 {
    #[cfg(unix)]
    {
        // SAFETY: `erff` is the platform C math function and accepts every finite/non-finite f32.
        unsafe { system_erff(value) }
    }
    #[cfg(not(unix))]
    {
        // Abramowitz and Stegun 7.1.26 fallback for platforms without a C `erff` symbol.
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let x = value.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let polynomial =
            (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
                + 0.254_829_6)
                * t;
        sign * (1.0 - polynomial * (-x * x).exp())
    }
}

#[inline]
fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_f32(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ff_ff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mantissa >> 13) as u16) | 0x0200;
        return sign | 0x7c00 | payload;
    }

    let unbiased_exponent = exponent - 127;
    if unbiased_exponent > 15 {
        return sign | 0x7c00;
    }
    if unbiased_exponent < -24 {
        return sign;
    }

    if unbiased_exponent < -14 {
        let significand = mantissa | 0x80_00_00;
        let shift = (-1 - unbiased_exponent) as u32;
        let mut rounded = significand >> shift;
        let remainder_mask = (1u32 << shift) - 1;
        let remainder = significand & remainder_mask;
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }

    let half_exponent = ((unbiased_exponent + 15) as u32) << 10;
    let mut rounded_mantissa = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded_mantissa & 1 != 0) {
        rounded_mantissa += 1;
    }
    sign | (half_exponent + rounded_mantissa) as u16
}

#[inline]
fn round_to_f16(value: f32) -> f32 {
    fp16_to_fp32(f32_to_f16_bits(value))
}

#[inline]
fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let bf16_bits = if bits & 0x7fff_ffff > 0x7f80_0000 {
        (bits >> 16) | 64
    } else {
        (bits + (0x7fff + ((bits >> 16) & 1))) >> 16
    };
    f32::from_bits(bf16_bits << 16)
}

fn matmul_with_ggml_activation_type(
    output: &mut [f32],
    input: &[f32],
    weight: &QuantizedTensor,
    mapped: &[u8],
) -> Result<(), String> {
    let rounded_input = match weight.ttype.0 {
        GGML_TYPE_F16 => Some(input.iter().copied().map(round_to_f16).collect::<Vec<_>>()),
        GGML_TYPE_BF16 => Some(input.iter().copied().map(round_to_bf16).collect::<Vec<_>>()),
        _ => None,
    };
    matmul_quantized(
        output,
        rounded_input.as_deref().unwrap_or(input),
        weight,
        mapped,
    )
}

fn positive_metadata_usize(gguf: &GGUFFile, key: &str) -> Result<usize, String> {
    usize::try_from(get_gguf_int_from_map(&gguf.kv, key, 0))
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("qwen3_asr mmproj is missing required positive metadata '{key}'"))
}

fn tensor_shape(tensor: &Gguftensor) -> Result<Vec<usize>, String> {
    if tensor.n_dims == 0 || tensor.n_dims > 4 {
        return Err(format!(
            "tensor {} has invalid rank {}",
            tensor.name, tensor.n_dims
        ));
    }
    tensor.ne[..tensor.n_dims as usize]
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                format!(
                    "tensor {} dimension {} does not fit usize",
                    tensor.name, dim
                )
            })
        })
        .collect()
}

fn validate_tensor_storage(gguf: &GGUFFile, tensor: &Gguftensor) -> Result<(), String> {
    let shape = tensor_shape(tensor)?;
    let n_elements = shape.iter().try_fold(1usize, |total, dim| {
        total.checked_mul(*dim).ok_or_else(|| {
            format!(
                "tensor {} element count overflows usize for shape {:?}",
                tensor.name, shape
            )
        })
    })?;
    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if type_size == 0 {
        return Err(format!(
            "tensor {} uses unsupported GGML type {}",
            tensor.name, tensor.ttype.0
        ));
    }
    if shape[0] % block_size != 0 {
        return Err(format!(
            "tensor {} first dimension {} is not divisible by GGML block size {} for type {}",
            tensor.name, shape[0], block_size, tensor.ttype.0
        ));
    }
    let byte_len = n_elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| format!("tensor {} storage size overflows usize", tensor.name))?;
    let end = tensor
        .data_offset
        .checked_add(byte_len)
        .ok_or_else(|| format!("tensor {} data range overflows usize", tensor.name))?;
    if end > gguf.mapped.as_slice().len() {
        return Err(format!(
            "tensor {} data range [{}..{}) exceeds mapped sidecar length {}",
            tensor.name,
            tensor.data_offset,
            end,
            gguf.mapped.as_slice().len()
        ));
    }
    Ok(())
}

fn tensor_weight(gguf: &GGUFFile, tensor: &Gguftensor) -> Result<TensorWeight, String> {
    validate_tensor_storage(gguf, tensor)?;
    Ok(TensorWeight {
        name: tensor.name.clone(),
        data_offset: tensor.data_offset,
        ttype: tensor.ttype,
        shape: tensor_shape(tensor)?,
    })
}

fn require_shape(gguf: &GGUFFile, name: &str, expected: &[usize]) -> Result<TensorWeight, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let actual = tensor_shape(tensor)?;
    if actual != expected {
        return Err(format!(
            "qwen3_asr tensor {name} shape mismatch: got {:?}, expected {:?} (GGML dimension order)",
            actual, expected
        ));
    }
    tensor_weight(gguf, tensor)
}

fn optional_shape(
    gguf: &GGUFFile,
    name: &str,
    expected: &[usize],
) -> Result<Option<TensorWeight>, String> {
    let Some(tensor) = find_gguf_tensor(gguf, name) else {
        return Ok(None);
    };
    let actual = tensor_shape(tensor)?;
    if actual != expected {
        return Err(format!(
            "qwen3_asr tensor {name} shape mismatch: got {:?}, expected {:?} (GGML dimension order)",
            actual, expected
        ));
    }
    tensor_weight(gguf, tensor).map(Some)
}

fn require_matrix(gguf: &GGUFFile, name: &str) -> Result<TensorWeight, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let shape = tensor_shape(tensor)?;
    if shape.len() != 2 {
        return Err(format!(
            "qwen3_asr tensor {name} has rank {}, expected a matrix",
            shape.len()
        ));
    }
    tensor_weight(gguf, tensor)
}

fn tensor_element_count(weight: &TensorWeight) -> Result<usize, String> {
    weight.shape.iter().try_fold(1usize, |total, dimension| {
        total.checked_mul(*dimension).ok_or_else(|| {
            format!(
                "qwen3_asr tensor {} element count overflows usize",
                weight.name
            )
        })
    })
}

fn dequantize_weight(gguf: &GGUFFile, weight: &TensorWeight) -> Result<Vec<f32>, String> {
    let n_elements = tensor_element_count(weight)?;
    let block_size = get_block_size(weight.ttype);
    let type_size = get_type_size(weight.ttype);
    let byte_len = n_elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| {
            format!(
                "qwen3_asr tensor {} storage size overflows usize",
                weight.name
            )
        })?;
    let end = weight
        .data_offset
        .checked_add(byte_len)
        .ok_or_else(|| format!("qwen3_asr tensor {} data range overflows", weight.name))?;
    let mapped = gguf.mapped.as_slice();
    if end > mapped.len() {
        return Err(format!(
            "qwen3_asr tensor {} exceeds mapped sidecar bounds",
            weight.name
        ));
    }
    gguf.ensure_range(weight.data_offset, byte_len)?;
    dequantize_tensor(&mapped[weight.data_offset..end], n_elements, weight.ttype)
}

fn load_conv_frontend_weights(
    gguf: &GGUFFile,
    conv_weights: &[TensorWeight; CONV_LAYER_COUNT],
    conv_biases: &[TensorWeight; CONV_LAYER_COUNT],
    output_weight: &TensorWeight,
) -> Result<ConvFrontendWeights, String> {
    let mut loaded_layers = Vec::with_capacity(CONV_LAYER_COUNT);
    for index in 0..CONV_LAYER_COUNT {
        let weight = &conv_weights[index];
        if !matches!(
            weight.ttype.0,
            GGML_TYPE_F16 | GGML_TYPE_F32 | GGML_TYPE_BF16
        ) {
            return Err(format!(
                "qwen3_asr convolution tensor {} uses unsupported GGML type {}; expected F16, F32, or BF16",
                weight.name, weight.ttype.0
            ));
        }
        loaded_layers.push(Conv2dLayerWeights {
            kernel: dequantize_weight(gguf, weight)?,
            bias: dequantize_weight(gguf, &conv_biases[index])?,
            input_channels: weight.shape[2],
            output_channels: weight.shape[3],
            kernel_type: weight.ttype,
        });
    }
    let layers: [Conv2dLayerWeights; CONV_LAYER_COUNT] = loaded_layers
        .try_into()
        .map_err(|_| "internal qwen3_asr convolution layer count mismatch".to_string())?;
    let output_projection = QuantizedTensor {
        data_offset: output_weight.data_offset,
        ttype: output_weight.ttype,
        rows: output_weight.shape[1],
        cols: output_weight.shape[0],
    };
    Ok(ConvFrontendWeights {
        layers,
        output_projection,
    })
}

fn quantized_matrix(
    weight: &TensorWeight,
    rows: usize,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    if weight.shape != [cols, rows] {
        return Err(format!(
            "qwen3_asr tensor {} runtime matrix shape mismatch: got {:?}, expected [{cols}, {rows}]",
            weight.name, weight.shape
        ));
    }
    Ok(QuantizedTensor {
        data_offset: weight.data_offset,
        ttype: weight.ttype,
        rows,
        cols,
    })
}

fn load_optional_vector(
    gguf: &GGUFFile,
    weight: Option<&TensorWeight>,
    length: usize,
) -> Result<Vec<f32>, String> {
    match weight {
        Some(weight) => dequantize_weight(gguf, weight),
        None => Ok(vec![0.0; length]),
    }
}

fn load_audio_transformer_weights(
    gguf: &GGUFFile,
    position_embeddings: &TensorWeight,
    layer_descriptors: &[AudioTransformerLayerWeights],
    dim: usize,
    feed_forward_dim: usize,
) -> Result<AudioTransformerWeights, String> {
    let mut layers = Vec::with_capacity(layer_descriptors.len());
    for layer in layer_descriptors {
        layers.push(AudioTransformerLayer {
            ln1_weight: dequantize_weight(gguf, &layer.ln1_weight)?,
            ln1_bias: load_optional_vector(gguf, layer.ln1_bias.as_ref(), dim)?,
            ln2_weight: dequantize_weight(gguf, &layer.ln2_weight)?,
            ln2_bias: load_optional_vector(gguf, layer.ln2_bias.as_ref(), dim)?,
            query_weight: quantized_matrix(&layer.query_weight, dim, dim)?,
            query_bias: load_optional_vector(gguf, layer.query_bias.as_ref(), dim)?,
            key_weight: quantized_matrix(&layer.key_weight, dim, dim)?,
            key_bias: load_optional_vector(gguf, layer.key_bias.as_ref(), dim)?,
            value_weight: quantized_matrix(&layer.value_weight, dim, dim)?,
            value_bias: load_optional_vector(gguf, layer.value_bias.as_ref(), dim)?,
            output_weight: quantized_matrix(&layer.output_weight, dim, dim)?,
            output_bias: load_optional_vector(gguf, layer.output_bias.as_ref(), dim)?,
            feed_forward_up_weight: quantized_matrix(
                &layer.feed_forward_up_weight,
                feed_forward_dim,
                dim,
            )?,
            feed_forward_up_bias: load_optional_vector(
                gguf,
                layer.feed_forward_up_bias.as_ref(),
                feed_forward_dim,
            )?,
            feed_forward_down_weight: quantized_matrix(
                &layer.feed_forward_down_weight,
                dim,
                feed_forward_dim,
            )?,
            feed_forward_down_bias: load_optional_vector(
                gguf,
                layer.feed_forward_down_bias.as_ref(),
                dim,
            )?,
        });
    }
    Ok(AudioTransformerWeights {
        position_embeddings: dequantize_weight(gguf, position_embeddings)?,
        layers,
    })
}

fn load_audio_projector_weights(
    gguf: &GGUFFile,
    up_weight: &TensorWeight,
    up_bias: &TensorWeight,
    down_weight: &TensorWeight,
    down_bias: &TensorWeight,
    config: &Qwen3AsrAudioConfig,
) -> Result<AudioProjectorWeights, String> {
    Ok(AudioProjectorWeights {
        up_weight: quantized_matrix(up_weight, config.projector_hidden_dim, config.embedding_dim)?,
        up_bias: dequantize_weight(gguf, up_bias)?,
        down_weight: quantized_matrix(
            down_weight,
            config.projection_dim,
            config.projector_hidden_dim,
        )?,
        down_bias: dequantize_weight(gguf, down_bias)?,
    })
}

fn conv_output_size(input: usize) -> Result<usize, String> {
    input
        .checked_add(CONV_PADDING * 2)
        .and_then(|padded| padded.checked_sub(CONV_KERNEL))
        .map(|value| value / CONV_STRIDE + 1)
        .ok_or_else(|| format!("invalid qwen3_asr convolution input size {input}"))
}

fn checked_feature_map_len(
    batch_count: usize,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<usize, String> {
    batch_count
        .checked_mul(width)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| "qwen3_asr convolution feature-map size overflow".to_string())
}

fn feature_map_from_window(
    window: &PreparedAudioFeatureWindow,
    expected_mel_bins: usize,
) -> Result<ConvFeatureMap, String> {
    if window.mel_bins != expected_mel_bins {
        return Err(format!(
            "qwen3_asr feature window at frame {} has {} mel bins, expected {expected_mel_bins}",
            window.start_frame, window.mel_bins
        ));
    }
    if window.padded_frames == 0 || !window.padded_frames.is_multiple_of(MEL_FRAMES_PER_CHUNK) {
        return Err(format!(
            "qwen3_asr feature window at frame {} has {} padded frames; expected a non-zero multiple of {MEL_FRAMES_PER_CHUNK}",
            window.start_frame, window.padded_frames
        ));
    }
    if window.valid_frames > window.padded_frames {
        return Err(format!(
            "qwen3_asr feature window at frame {} has {} valid frames but only {} padded frames",
            window.start_frame, window.valid_frames, window.padded_frames
        ));
    }
    let expected_len = expected_mel_bins
        .checked_mul(window.padded_frames)
        .ok_or_else(|| "qwen3_asr feature-window size overflow".to_string())?;
    if window.data_mel_major.len() != expected_len {
        return Err(format!(
            "qwen3_asr feature window at frame {} has {} values, expected {expected_len}",
            window.start_frame,
            window.data_mel_major.len()
        ));
    }

    let batch_count = window.padded_frames / MEL_FRAMES_PER_CHUNK;
    let output_len =
        checked_feature_map_len(batch_count, MEL_FRAMES_PER_CHUNK, expected_mel_bins, 1)?;
    let mut data = vec![0.0f32; output_len];
    for batch in 0..batch_count {
        for mel in 0..expected_mel_bins {
            for frame in 0..MEL_FRAMES_PER_CHUNK {
                let source = mel * window.padded_frames + batch * MEL_FRAMES_PER_CHUNK + frame;
                let destination = (batch * expected_mel_bins + mel) * MEL_FRAMES_PER_CHUNK + frame;
                data[destination] = window.data_mel_major[source];
            }
        }
    }

    Ok(ConvFeatureMap {
        batch_count,
        width: MEL_FRAMES_PER_CHUNK,
        height: expected_mel_bins,
        channels: 1,
        data,
    })
}

fn conv2d_stride2_gelu_erf(
    input: &ConvFeatureMap,
    weights: &Conv2dLayerWeights,
) -> Result<ConvFeatureMap, String> {
    if input.channels != weights.input_channels {
        return Err(format!(
            "qwen3_asr convolution channel mismatch: input has {}, weight expects {}",
            input.channels, weights.input_channels
        ));
    }
    let kernel_elements = weights
        .input_channels
        .checked_mul(CONV_KERNEL * CONV_KERNEL)
        .ok_or_else(|| "qwen3_asr convolution kernel size overflow".to_string())?;
    let expected_kernel_len = weights
        .output_channels
        .checked_mul(kernel_elements)
        .ok_or_else(|| "qwen3_asr convolution weight size overflow".to_string())?;
    if weights.kernel.len() != expected_kernel_len || weights.bias.len() != weights.output_channels
    {
        return Err("qwen3_asr convolution runtime weight shape mismatch".to_string());
    }

    let output_width = conv_output_size(input.width)?;
    let output_height = conv_output_size(input.height)?;
    let output_len = checked_feature_map_len(
        input.batch_count,
        output_width,
        output_height,
        weights.output_channels,
    )?;

    // llama.cpp's ggml_conv_2d im2col path uses F16 activations for F16/F32 kernels and
    // F32 activations for BF16 kernels. Round the source map once before extracting patches.
    let activation_data = if weights.kernel_type.0 == GGML_TYPE_BF16 {
        input.data.clone()
    } else {
        input.data.iter().copied().map(round_to_f16).collect()
    };
    let input_width = input.width;
    let input_height = input.height;
    let input_channels = input.channels;
    let output_spatial = output_width * output_height;
    let mut output = vec![0.0f32; output_len];

    output
        .par_chunks_mut(weights.output_channels)
        .enumerate()
        .for_each_init(
            || vec![0.0f32; kernel_elements],
            |patch, (patch_index, output_channels)| {
                let batch = patch_index / output_spatial;
                let spatial = patch_index % output_spatial;
                let output_y = spatial / output_width;
                let output_x = spatial % output_width;

                let mut patch_offset = 0usize;
                for input_channel in 0..input_channels {
                    for kernel_y in 0..CONV_KERNEL {
                        let padded_y = output_y * CONV_STRIDE + kernel_y;
                        for kernel_x in 0..CONV_KERNEL {
                            let padded_x = output_x * CONV_STRIDE + kernel_x;
                            patch[patch_offset] = if padded_y < CONV_PADDING
                                || padded_x < CONV_PADDING
                                || padded_y - CONV_PADDING >= input_height
                                || padded_x - CONV_PADDING >= input_width
                            {
                                0.0
                            } else {
                                let input_y = padded_y - CONV_PADDING;
                                let input_x = padded_x - CONV_PADDING;
                                let input_index =
                                    (((batch * input_height + input_y) * input_width + input_x)
                                        * input_channels)
                                        + input_channel;
                                activation_data[input_index]
                            };
                            patch_offset += 1;
                        }
                    }
                }

                for (output_channel, value) in output_channels.iter_mut().enumerate() {
                    let weight_offset = output_channel * kernel_elements;
                    let sum = dot_f32_simd(
                        patch,
                        &weights.kernel[weight_offset..weight_offset + kernel_elements],
                    );
                    *value = gelu_erf(sum + weights.bias[output_channel]);
                }
            },
        );

    Ok(ConvFeatureMap {
        batch_count: input.batch_count,
        width: output_width,
        height: output_height,
        channels: weights.output_channels,
        data: output,
    })
}

fn flatten_conv_features(features: &ConvFeatureMap) -> Result<Vec<f32>, String> {
    let flattened_dim = features
        .channels
        .checked_mul(features.height)
        .ok_or_else(|| "qwen3_asr flattened convolution dimension overflow".to_string())?;
    let token_count = features
        .batch_count
        .checked_mul(features.width)
        .ok_or_else(|| "qwen3_asr convolution token count overflow".to_string())?;
    let output_len = token_count
        .checked_mul(flattened_dim)
        .ok_or_else(|| "qwen3_asr flattened convolution output overflow".to_string())?;
    let mut output = vec![0.0f32; output_len];

    for batch in 0..features.batch_count {
        for x in 0..features.width {
            let token = batch * features.width + x;
            for channel in 0..features.channels {
                for y in 0..features.height {
                    let source = (((batch * features.height + y) * features.width + x)
                        * features.channels)
                        + channel;
                    let destination = token * flattened_dim + channel * features.height + y;
                    output[destination] = features.data[source];
                }
            }
        }
    }
    Ok(output)
}

fn project_conv_features(
    features: &ConvFeatureMap,
    projection: &QuantizedTensor,
    mapped: &[u8],
) -> Result<AudioEncoderFrontendOutput, String> {
    let flattened_dim = features
        .channels
        .checked_mul(features.height)
        .ok_or_else(|| "qwen3_asr flattened convolution dimension overflow".to_string())?;
    if projection.cols != flattened_dim {
        return Err(format!(
            "qwen3_asr convolution projection expects {} inputs, got {flattened_dim}",
            projection.cols
        ));
    }
    let token_count = features
        .batch_count
        .checked_mul(features.width)
        .ok_or_else(|| "qwen3_asr convolution token count overflow".to_string())?;
    let flattened = flatten_conv_features(features)?;
    let output_len = token_count
        .checked_mul(projection.rows)
        .ok_or_else(|| "qwen3_asr convolution projection output overflow".to_string())?;
    let mut data_token_major = vec![0.0f32; output_len];

    for token in 0..token_count {
        let input = &flattened[token * flattened_dim..(token + 1) * flattened_dim];
        let output = &mut data_token_major[token * projection.rows..(token + 1) * projection.rows];
        matmul_with_ggml_activation_type(output, input, projection, mapped)?;
    }

    Ok(AudioEncoderFrontendOutput {
        token_count,
        embedding_dim: projection.rows,
        data_token_major,
    })
}

fn add_chunk_position_embeddings(
    output: &mut AudioEncoderFrontendOutput,
    position_embeddings: &[f32],
    position_count: usize,
) -> Result<(), String> {
    let expected_output_values = output
        .token_count
        .checked_mul(output.embedding_dim)
        .ok_or_else(|| "qwen3_asr positional input size overflow".to_string())?;
    if output.embedding_dim == 0 || output.data_token_major.len() != expected_output_values {
        return Err("qwen3_asr positional input shape mismatch".to_string());
    }
    if output.token_count == 0 || !output.token_count.is_multiple_of(TOKENS_PER_CHUNK) {
        return Err(format!(
            "qwen3_asr positional token count {} is not a non-zero multiple of {TOKENS_PER_CHUNK}",
            output.token_count
        ));
    }
    if position_count < TOKENS_PER_CHUNK {
        return Err(format!(
            "qwen3_asr position table has {position_count} rows, expected at least {TOKENS_PER_CHUNK}"
        ));
    }
    let expected_position_values = position_count
        .checked_mul(output.embedding_dim)
        .ok_or_else(|| "qwen3_asr position table size overflow".to_string())?;
    if position_embeddings.len() != expected_position_values {
        return Err(format!(
            "qwen3_asr position table has {} values, expected {expected_position_values}",
            position_embeddings.len()
        ));
    }

    for token in 0..output.token_count {
        let position = token % TOKENS_PER_CHUNK;
        let source = &position_embeddings
            [position * output.embedding_dim..(position + 1) * output.embedding_dim];
        let destination = &mut output.data_token_major
            [token * output.embedding_dim..(token + 1) * output.embedding_dim];
        axpy_inplace(destination, 1.0, source);
    }
    Ok(())
}

fn layer_norm_affine(
    destination: &mut [f32],
    source: &[f32],
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) {
    let mut mean = 0.0f32;
    for value in source {
        mean += *value;
    }
    mean /= source.len() as f32;

    let mut variance = 0.0f32;
    for value in source {
        let centered = *value - mean;
        variance += centered * centered;
    }
    variance /= source.len() as f32;
    let inverse_standard_deviation = 1.0 / (variance + epsilon).sqrt();

    for index in 0..source.len() {
        destination[index] =
            ((source[index] - mean) * inverse_standard_deviation) * weight[index] + bias[index];
    }
}

fn add_bias(values: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(values.len(), bias.len());
    for (value, bias) in values.iter_mut().zip(bias) {
        *value += *bias;
    }
}

fn project_to_language_embeddings(
    frontend: &AudioEncoderFrontendOutput,
    weights: &AudioProjectorWeights,
    expected_input_dim: usize,
    mapped: &[u8],
) -> Result<MediaEmbeddingSequence, String> {
    let expected_input_values = frontend
        .token_count
        .checked_mul(expected_input_dim)
        .ok_or_else(|| "qwen3_asr projector input size overflow".to_string())?;
    if expected_input_dim == 0
        || frontend.token_count == 0
        || frontend.embedding_dim != expected_input_dim
        || frontend.data_token_major.len() != expected_input_values
        || weights.up_weight.cols != expected_input_dim
        || weights.up_weight.rows == 0
        || weights.up_weight.rows != weights.up_bias.len()
        || weights.down_weight.cols != weights.up_weight.rows
        || weights.down_weight.rows == 0
        || weights.down_weight.rows != weights.down_bias.len()
    {
        return Err("qwen3_asr projector input or runtime weight shape mismatch".to_string());
    }

    let hidden_dim = weights.up_weight.rows;
    let output_dim = weights.down_weight.rows;
    let output_values = frontend
        .token_count
        .checked_mul(output_dim)
        .ok_or_else(|| "qwen3_asr projector output size overflow".to_string())?;
    let mut projected = vec![0.0f32; output_values];
    projected
        .par_chunks_mut(output_dim)
        .enumerate()
        .try_for_each_init(
            || vec![0.0f32; hidden_dim],
            |hidden, (token, output)| -> Result<(), String> {
                let input = &frontend.data_token_major
                    [token * expected_input_dim..(token + 1) * expected_input_dim];
                matmul_with_ggml_activation_type(hidden, input, &weights.up_weight, mapped)?;
                add_bias(hidden, &weights.up_bias);
                for value in hidden.iter_mut() {
                    *value = gelu_erf(*value);
                }
                matmul_with_ggml_activation_type(output, hidden, &weights.down_weight, mapped)?;
                add_bias(output, &weights.down_bias);
                Ok(())
            },
        )?;

    if let Some((index, value)) = projected
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "qwen3_asr projector produced non-finite value {value} at flat output index {index}"
        ));
    }

    Ok(MediaEmbeddingSequence {
        tokens: projected
            .chunks_exact(output_dim)
            .map(<[f32]>::to_vec)
            .collect(),
    })
}

fn run_audio_transformer(
    output: &mut AudioEncoderFrontendOutput,
    weights: &AudioTransformerWeights,
    config: &Qwen3AsrAudioConfig,
    mapped: &[u8],
) -> Result<(), String> {
    run_audio_transformer_inner(output, weights, config, mapped, None)
}

fn run_audio_transformer_inner(
    output: &mut AudioEncoderFrontendOutput,
    weights: &AudioTransformerWeights,
    config: &Qwen3AsrAudioConfig,
    mapped: &[u8],
    mut layer_outputs: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), String> {
    let token_count = output.token_count;
    let dim = config.embedding_dim;
    let feed_forward_dim = config.feed_forward_dim;
    let token_values = token_count
        .checked_mul(dim)
        .ok_or_else(|| "qwen3_asr transformer input size overflow".to_string())?;
    if output.embedding_dim != dim
        || token_count == 0
        || output.data_token_major.len() != token_values
    {
        return Err("qwen3_asr transformer input shape mismatch".to_string());
    }
    if weights.layers.len() != config.layer_count {
        return Err(format!(
            "qwen3_asr transformer has {} loaded layers, expected {}",
            weights.layers.len(),
            config.layer_count
        ));
    }

    let mut normalized = vec![0.0f32; token_values];
    let mut query = vec![0.0f32; token_values];
    let mut key = vec![0.0f32; token_values];
    let mut value = vec![0.0f32; token_values];
    let head_token_stride = token_count
        .checked_mul(config.head_dim)
        .ok_or_else(|| "qwen3_asr attention head scratch overflow".to_string())?;
    let attention_values = config
        .head_count
        .checked_mul(head_token_stride)
        .ok_or_else(|| "qwen3_asr attention scratch size overflow".to_string())?;
    let mut attention_head_major = vec![0.0f32; attention_values];
    let mut attention_token_major = vec![0.0f32; token_values];
    let mut projected_attention = vec![0.0f32; token_values];
    let epsilon = config.layer_norm_epsilon;

    for layer in &weights.layers {
        normalized
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(token, destination)| {
                let source = &output.data_token_major[token * dim..(token + 1) * dim];
                layer_norm_affine(
                    destination,
                    source,
                    &layer.ln1_weight,
                    &layer.ln1_bias,
                    epsilon,
                );
            });

        query
            .par_chunks_mut(dim)
            .zip(key.par_chunks_mut(dim))
            .zip(value.par_chunks_mut(dim))
            .enumerate()
            .try_for_each(
                |(token, ((query_out, key_out), value_out))| -> Result<(), String> {
                    let source = &normalized[token * dim..(token + 1) * dim];
                    matmul_with_ggml_activation_type(
                        query_out,
                        source,
                        &layer.query_weight,
                        mapped,
                    )?;
                    add_bias(query_out, &layer.query_bias);
                    matmul_with_ggml_activation_type(key_out, source, &layer.key_weight, mapped)?;
                    add_bias(key_out, &layer.key_bias);
                    matmul_with_ggml_activation_type(
                        value_out,
                        source,
                        &layer.value_weight,
                        mapped,
                    )?;
                    add_bias(value_out, &layer.value_bias);
                    Ok(())
                },
            )?;

        let attention_scale = 1.0 / (config.head_dim as f32).sqrt();
        attention_head_major
            .par_chunks_mut(config.head_dim)
            .enumerate()
            .for_each(|(row, destination)| {
                let head = row / token_count;
                let query_token = row % token_count;
                let head_offset = head * config.head_dim;
                let query_values = &query[query_token * dim + head_offset
                    ..query_token * dim + head_offset + config.head_dim];

                destination.fill(0.0);
                let mut maximum_score = f32::NEG_INFINITY;
                let mut score_sum = 0.0f32;
                for key_token in 0..token_count {
                    let key_values = &key[key_token * dim + head_offset
                        ..key_token * dim + head_offset + config.head_dim];
                    let score = dot_f32_simd(query_values, key_values) * attention_scale;
                    if score > maximum_score {
                        if score_sum > 0.0 {
                            let rescale = (maximum_score - score).exp();
                            scale_slice_inplace(destination, rescale);
                            score_sum *= rescale;
                        }
                        maximum_score = score;
                    }
                    let attention_weight = (score - maximum_score).exp();
                    score_sum += attention_weight;
                    let values = &value[key_token * dim + head_offset
                        ..key_token * dim + head_offset + config.head_dim];
                    axpy_inplace(destination, attention_weight, values);
                }
                if score_sum > 0.0 {
                    scale_slice_inplace(destination, 1.0 / score_sum);
                }
            });

        for token in 0..token_count {
            let destination = &mut attention_token_major[token * dim..(token + 1) * dim];
            for head in 0..config.head_count {
                let source = &attention_head_major[head * head_token_stride
                    + token * config.head_dim
                    ..head * head_token_stride + (token + 1) * config.head_dim];
                let head_offset = head * config.head_dim;
                destination[head_offset..head_offset + config.head_dim].copy_from_slice(source);
            }
        }

        projected_attention
            .par_chunks_mut(dim)
            .enumerate()
            .try_for_each(|(token, destination)| -> Result<(), String> {
                let source = &attention_token_major[token * dim..(token + 1) * dim];
                matmul_with_ggml_activation_type(
                    destination,
                    source,
                    &layer.output_weight,
                    mapped,
                )?;
                add_bias(destination, &layer.output_bias);
                Ok(())
            })?;
        for (residual, attention) in output.data_token_major.iter_mut().zip(&projected_attention) {
            *residual += *attention;
        }

        normalized
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(token, destination)| {
                let source = &output.data_token_major[token * dim..(token + 1) * dim];
                layer_norm_affine(
                    destination,
                    source,
                    &layer.ln2_weight,
                    &layer.ln2_bias,
                    epsilon,
                );
            });

        output
            .data_token_major
            .par_chunks_mut(dim)
            .enumerate()
            .try_for_each_init(
                || (vec![0.0f32; feed_forward_dim], vec![0.0f32; dim]),
                |(feed_forward_up, feed_forward_down), (token, residual)| -> Result<(), String> {
                    let source = &normalized[token * dim..(token + 1) * dim];
                    matmul_with_ggml_activation_type(
                        feed_forward_up,
                        source,
                        &layer.feed_forward_up_weight,
                        mapped,
                    )?;
                    add_bias(feed_forward_up, &layer.feed_forward_up_bias);
                    for value in feed_forward_up.iter_mut() {
                        *value = gelu_erf(*value);
                    }
                    matmul_with_ggml_activation_type(
                        feed_forward_down,
                        feed_forward_up,
                        &layer.feed_forward_down_weight,
                        mapped,
                    )?;
                    add_bias(feed_forward_down, &layer.feed_forward_down_bias);
                    axpy_inplace(residual, 1.0, feed_forward_down);
                    Ok(())
                },
            )?;
        if let Some(outputs) = layer_outputs.as_mut() {
            outputs.push(output.data_token_major.clone());
        }
    }

    if let Some((index, value)) = output
        .data_token_major
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "qwen3_asr transformer produced non-finite value {value} at flat index {index}"
        ));
    }
    Ok(())
}

fn apply_post_layer_norm(
    output: &mut AudioEncoderFrontendOutput,
    weights: &AudioPostLayerNormWeights,
    epsilon: f32,
) -> Result<(), String> {
    let dim = output.embedding_dim;
    if dim == 0
        || weights.weight.len() != dim
        || weights.bias.len() != dim
        || output.data_token_major.len() != output.token_count.saturating_mul(dim)
    {
        return Err("qwen3_asr post-layer-norm shape mismatch".to_string());
    }
    let mut normalized = vec![0.0f32; output.data_token_major.len()];
    normalized
        .par_chunks_mut(dim)
        .zip(output.data_token_major.par_chunks(dim))
        .for_each(|(destination, source)| {
            layer_norm_affine(destination, source, &weights.weight, &weights.bias, epsilon);
        });
    output.data_token_major = normalized;
    if let Some((index, value)) = output
        .data_token_major
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "qwen3_asr post-layer norm produced non-finite value {value} at flat index {index}"
        ));
    }
    Ok(())
}

fn parse_metadata(
    gguf: &GGUFFile,
    expected_projection_dim: usize,
) -> Result<Qwen3AsrAudioConfig, String> {
    let architecture =
        get_gguf_string_from_map(&gguf.kv, "general.architecture").unwrap_or_default();
    if architecture != "clip" {
        return Err(format!(
            "unsupported qwen3_asr sidecar architecture '{architecture}'; expected 'clip'"
        ));
    }
    if !get_gguf_bool_from_map(&gguf.kv, "clip.has_audio_encoder", false) {
        return Err("qwen3_asr sidecar is missing clip.has_audio_encoder=true".to_string());
    }
    let projector =
        get_gguf_string_from_map(&gguf.kv, "clip.audio.projector_type").unwrap_or_default();
    if projector != "qwen3a" {
        return Err(format!(
            "unsupported audio projector '{projector}'; expected 'qwen3a'"
        ));
    }

    let embedding_dim = positive_metadata_usize(gguf, "clip.audio.embedding_length")?;
    let feed_forward_dim = positive_metadata_usize(gguf, "clip.audio.feed_forward_length")?;
    let layer_count = positive_metadata_usize(gguf, "clip.audio.block_count")?;
    let head_count = positive_metadata_usize(gguf, "clip.audio.attention.head_count")?;
    let mel_bin_count = positive_metadata_usize(gguf, "clip.audio.num_mel_bins")?;
    let projection_dim = positive_metadata_usize(gguf, "clip.audio.projection_dim")?;
    let layer_norm_epsilon =
        get_gguf_float_from_map(&gguf.kv, "clip.audio.attention.layer_norm_epsilon", 0.0);

    if projection_dim != expected_projection_dim {
        return Err(format!(
            "qwen3_asr projection mismatch: sidecar outputs {projection_dim}, text model expects {expected_projection_dim}"
        ));
    }
    if mel_bin_count != 128 {
        return Err(format!(
            "unsupported qwen3_asr mel-bin count {mel_bin_count}; expected 128"
        ));
    }
    if !embedding_dim.is_multiple_of(head_count) {
        return Err(format!(
            "invalid qwen3_asr attention shape: embedding_dim={embedding_dim} is not divisible by head_count={head_count}"
        ));
    }
    if !layer_norm_epsilon.is_finite() || layer_norm_epsilon <= 0.0 {
        return Err(
            "qwen3_asr sidecar has invalid clip.audio.attention.layer_norm_epsilon".to_string(),
        );
    }

    Ok(Qwen3AsrAudioConfig {
        embedding_dim,
        convolution_channel_count: 0,
        feed_forward_dim,
        layer_count,
        head_count,
        head_dim: embedding_dim / head_count,
        layer_norm_epsilon,
        mel_bin_count,
        projection_dim,
        position_count: 0,
        projector_hidden_dim: 0,
    })
}

impl Qwen3AsrAudioEncoder {
    pub(crate) fn new(gguf: GGUFFile, expected_projection_dim: usize) -> Result<Self, String> {
        let mut config = parse_metadata(&gguf, expected_projection_dim)?;
        let dim = config.embedding_dim;
        let ff_dim = config.feed_forward_dim;

        let position_tensor = find_gguf_tensor(&gguf, "a.position_embd.weight")
            .ok_or_else(|| "tensor not found: a.position_embd.weight".to_string())?;
        let position_shape = tensor_shape(position_tensor)?;
        if position_shape.len() != 2
            || position_shape[0] != dim
            || position_shape[1] < TOKENS_PER_CHUNK
        {
            return Err(format!(
                "qwen3_asr tensor a.position_embd.weight shape mismatch: got {:?}, expected [{dim}, positions>={TOKENS_PER_CHUNK}]",
                position_shape
            ));
        }
        config.position_count = position_shape[1];
        let position_embeddings = tensor_weight(&gguf, position_tensor)?;

        let first_conv_tensor = find_gguf_tensor(&gguf, "a.conv2d.1.weight")
            .ok_or_else(|| "tensor not found: a.conv2d.1.weight".to_string())?;
        let first_conv_shape = tensor_shape(first_conv_tensor)?;
        if first_conv_shape.len() != 4
            || first_conv_shape[0] != CONV_KERNEL
            || first_conv_shape[1] != CONV_KERNEL
            || first_conv_shape[2] != 1
            || first_conv_shape[3] == 0
        {
            return Err(format!(
                "qwen3_asr tensor a.conv2d.1.weight shape mismatch: got {:?}, expected [{CONV_KERNEL}, {CONV_KERNEL}, 1, channels>0] (GGML dimension order)",
                first_conv_shape
            ));
        }
        let convolution_channel_count = first_conv_shape[3];
        config.convolution_channel_count = convolution_channel_count;
        let conv2d_weights = [
            tensor_weight(&gguf, first_conv_tensor)?,
            require_shape(
                &gguf,
                "a.conv2d.2.weight",
                &[
                    CONV_KERNEL,
                    CONV_KERNEL,
                    convolution_channel_count,
                    convolution_channel_count,
                ],
            )?,
            require_shape(
                &gguf,
                "a.conv2d.3.weight",
                &[
                    CONV_KERNEL,
                    CONV_KERNEL,
                    convolution_channel_count,
                    convolution_channel_count,
                ],
            )?,
        ];
        let conv_bias_shape = [1, 1, convolution_channel_count];
        let conv2d_biases = [
            require_shape(&gguf, "a.conv2d.1.bias", &conv_bias_shape)?,
            require_shape(&gguf, "a.conv2d.2.bias", &conv_bias_shape)?,
            require_shape(&gguf, "a.conv2d.3.bias", &conv_bias_shape)?,
        ];

        let mut reduced_mel_bins = config.mel_bin_count;
        let mut reduced_frames = MEL_FRAMES_PER_CHUNK;
        for _ in 0..CONV_LAYER_COUNT {
            reduced_mel_bins = conv_output_size(reduced_mel_bins)?;
            reduced_frames = conv_output_size(reduced_frames)?;
        }
        if reduced_frames != TOKENS_PER_CHUNK {
            return Err(format!(
                "internal qwen3_asr convolution contract mismatch: {MEL_FRAMES_PER_CHUNK} frames reduce to {reduced_frames}, expected {TOKENS_PER_CHUNK}"
            ));
        }
        let flattened_conv_dim = convolution_channel_count
            .checked_mul(reduced_mel_bins)
            .ok_or_else(|| "qwen3_asr flattened convolution dimension overflow".to_string())?;
        let conv_output_weight =
            require_shape(&gguf, "a.conv_out.weight", &[flattened_conv_dim, dim])?;

        let mut layers = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            let prefix = format!("a.blk.{layer}");
            layers.push(AudioTransformerLayerWeights {
                ln1_weight: require_shape(&gguf, &format!("{prefix}.ln1.weight"), &[dim])?,
                ln1_bias: optional_shape(&gguf, &format!("{prefix}.ln1.bias"), &[dim])?,
                ln2_weight: require_shape(&gguf, &format!("{prefix}.ln2.weight"), &[dim])?,
                ln2_bias: optional_shape(&gguf, &format!("{prefix}.ln2.bias"), &[dim])?,
                query_weight: require_shape(
                    &gguf,
                    &format!("{prefix}.attn_q.weight"),
                    &[dim, dim],
                )?,
                query_bias: optional_shape(&gguf, &format!("{prefix}.attn_q.bias"), &[dim])?,
                key_weight: require_shape(&gguf, &format!("{prefix}.attn_k.weight"), &[dim, dim])?,
                key_bias: optional_shape(&gguf, &format!("{prefix}.attn_k.bias"), &[dim])?,
                value_weight: require_shape(
                    &gguf,
                    &format!("{prefix}.attn_v.weight"),
                    &[dim, dim],
                )?,
                value_bias: optional_shape(&gguf, &format!("{prefix}.attn_v.bias"), &[dim])?,
                output_weight: require_shape(
                    &gguf,
                    &format!("{prefix}.attn_out.weight"),
                    &[dim, dim],
                )?,
                output_bias: optional_shape(&gguf, &format!("{prefix}.attn_out.bias"), &[dim])?,
                feed_forward_up_weight: require_shape(
                    &gguf,
                    &format!("{prefix}.ffn_up.weight"),
                    &[dim, ff_dim],
                )?,
                feed_forward_up_bias: optional_shape(
                    &gguf,
                    &format!("{prefix}.ffn_up.bias"),
                    &[ff_dim],
                )?,
                feed_forward_down_weight: require_shape(
                    &gguf,
                    &format!("{prefix}.ffn_down.weight"),
                    &[ff_dim, dim],
                )?,
                feed_forward_down_bias: optional_shape(
                    &gguf,
                    &format!("{prefix}.ffn_down.bias"),
                    &[dim],
                )?,
            });
        }
        let post_layer_norm_weight = require_shape(&gguf, "a.post_ln.weight", &[dim])?;
        let post_layer_norm_bias = require_shape(&gguf, "a.post_ln.bias", &[dim])?;

        let projector_up_weight = require_matrix(&gguf, "mm.a.mlp.1.weight")?;
        if projector_up_weight.shape[0] != dim {
            return Err(format!(
                "qwen3_asr tensor mm.a.mlp.1.weight input mismatch: got {}, expected audio embedding dim {dim}",
                projector_up_weight.shape[0]
            ));
        }
        let projector_hidden_dim = projector_up_weight.shape[1];
        if projector_hidden_dim == 0 {
            return Err("qwen3_asr projector hidden dimension is zero".to_string());
        }
        let projector_up_bias = require_shape(&gguf, "mm.a.mlp.1.bias", &[projector_hidden_dim])?;

        let projector_down_weight = require_matrix(&gguf, "mm.a.mlp.2.weight")?;
        let expected_down_shape = [projector_hidden_dim, config.projection_dim];
        if projector_down_weight.shape != expected_down_shape {
            return Err(format!(
                "qwen3_asr tensor mm.a.mlp.2.weight shape mismatch: got {:?}, expected {:?} (projector continuity and text output dimension)",
                projector_down_weight.shape, expected_down_shape
            ));
        }
        let projector_down_bias =
            require_shape(&gguf, "mm.a.mlp.2.bias", &[config.projection_dim])?;
        config.projector_hidden_dim = projector_hidden_dim;

        let conv_frontend = load_conv_frontend_weights(
            &gguf,
            &conv2d_weights,
            &conv2d_biases,
            &conv_output_weight,
        )?;
        let transformer =
            load_audio_transformer_weights(&gguf, &position_embeddings, &layers, dim, ff_dim)?;
        let post_layer_norm = AudioPostLayerNormWeights {
            weight: dequantize_weight(&gguf, &post_layer_norm_weight)?,
            bias: dequantize_weight(&gguf, &post_layer_norm_bias)?,
        };
        let projector = load_audio_projector_weights(
            &gguf,
            &projector_up_weight,
            &projector_up_bias,
            &projector_down_weight,
            &projector_down_bias,
            &config,
        )?;

        Ok(Self {
            gguf,
            config,
            weights: Qwen3AsrWeights {
                position_embeddings,
                conv2d_weights,
                conv2d_biases,
                conv_output_weight,
                layers,
                post_layer_norm_weight,
                post_layer_norm_bias,
                projector_up_weight,
                projector_up_bias,
                projector_down_weight,
                projector_down_bias,
            },
            conv_frontend,
            transformer,
            post_layer_norm,
            projector,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn encode_conv_frontend(
        &self,
        window: &PreparedAudioFeatureWindow,
    ) -> Result<AudioEncoderFrontendOutput, String> {
        let mut features = feature_map_from_window(window, self.config.mel_bin_count)?;
        for layer in &self.conv_frontend.layers {
            features = conv2d_stride2_gelu_erf(&features, layer)?;
        }
        if features.width != TOKENS_PER_CHUNK {
            return Err(format!(
                "internal qwen3_asr convolution token mismatch: got {} per chunk, expected {TOKENS_PER_CHUNK}",
                features.width
            ));
        }
        project_conv_features(
            &features,
            &self.conv_frontend.output_projection,
            self.gguf.mapped.as_slice(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn encode_transformer_frontend(
        &self,
        window: &PreparedAudioFeatureWindow,
    ) -> Result<AudioEncoderFrontendOutput, String> {
        self.transform_conv_frontend(self.encode_conv_frontend(window)?)
    }

    fn transform_conv_frontend(
        &self,
        mut output: AudioEncoderFrontendOutput,
    ) -> Result<AudioEncoderFrontendOutput, String> {
        add_chunk_position_embeddings(
            &mut output,
            &self.transformer.position_embeddings,
            self.config.position_count,
        )?;
        run_audio_transformer(
            &mut output,
            &self.transformer,
            &self.config,
            self.gguf.mapped.as_slice(),
        )?;
        apply_post_layer_norm(
            &mut output,
            &self.post_layer_norm,
            self.config.layer_norm_epsilon,
        )?;
        Ok(output)
    }

    #[allow(dead_code)]
    pub(crate) fn encode_from_conv_output(
        &self,
        data_token_major: Vec<f32>,
    ) -> Result<(AudioEncoderFrontendOutput, MediaEmbeddingSequence), String> {
        if data_token_major.is_empty()
            || !data_token_major
                .len()
                .is_multiple_of(self.config.embedding_dim)
        {
            return Err(format!(
                "qwen3_asr convolution-output fixture has {} values; expected a non-zero multiple of {}",
                data_token_major.len(),
                self.config.embedding_dim
            ));
        }
        let token_count = data_token_major.len() / self.config.embedding_dim;
        if !token_count.is_multiple_of(TOKENS_PER_CHUNK) {
            return Err(format!(
                "qwen3_asr convolution-output fixture has {token_count} tokens; expected a multiple of {TOKENS_PER_CHUNK}"
            ));
        }
        let transformer = self.transform_conv_frontend(AudioEncoderFrontendOutput {
            token_count,
            embedding_dim: self.config.embedding_dim,
            data_token_major,
        })?;
        let projected = project_to_language_embeddings(
            &transformer,
            &self.projector,
            self.config.embedding_dim,
            self.gguf.mapped.as_slice(),
        )?;
        Ok((transformer, projected))
    }

    #[allow(dead_code)]
    pub(crate) fn encode_from_conv_output_with_layers(
        &self,
        data_token_major: Vec<f32>,
    ) -> Result<
        (
            Vec<Vec<f32>>,
            AudioEncoderFrontendOutput,
            MediaEmbeddingSequence,
        ),
        String,
    > {
        if data_token_major.is_empty()
            || !data_token_major
                .len()
                .is_multiple_of(self.config.embedding_dim)
        {
            return Err(format!(
                "qwen3_asr convolution-output fixture has {} values; expected a non-zero multiple of {}",
                data_token_major.len(),
                self.config.embedding_dim
            ));
        }
        let token_count = data_token_major.len() / self.config.embedding_dim;
        if !token_count.is_multiple_of(TOKENS_PER_CHUNK) {
            return Err(format!(
                "qwen3_asr convolution-output fixture has {token_count} tokens; expected a multiple of {TOKENS_PER_CHUNK}"
            ));
        }
        let mut transformer = AudioEncoderFrontendOutput {
            token_count,
            embedding_dim: self.config.embedding_dim,
            data_token_major,
        };
        add_chunk_position_embeddings(
            &mut transformer,
            &self.transformer.position_embeddings,
            self.config.position_count,
        )?;
        let mut layer_outputs = Vec::with_capacity(self.config.layer_count);
        run_audio_transformer_inner(
            &mut transformer,
            &self.transformer,
            &self.config,
            self.gguf.mapped.as_slice(),
            Some(&mut layer_outputs),
        )?;
        apply_post_layer_norm(
            &mut transformer,
            &self.post_layer_norm,
            self.config.layer_norm_epsilon,
        )?;
        let projected = project_to_language_embeddings(
            &transformer,
            &self.projector,
            self.config.embedding_dim,
            self.gguf.mapped.as_slice(),
        )?;
        Ok((layer_outputs, transformer, projected))
    }

    #[allow(dead_code)]
    pub(crate) fn encode_feature_window(
        &self,
        window: &PreparedAudioFeatureWindow,
    ) -> Result<MediaEmbeddingSequence, String> {
        let frontend = self.encode_transformer_frontend(window)?;
        project_to_language_embeddings(
            &frontend,
            &self.projector,
            self.config.embedding_dim,
            self.gguf.mapped.as_slice(),
        )
    }

    pub(crate) fn planned_embedding_token_count(
        &self,
        windows: &[PreparedAudioFeatureWindowPlan],
    ) -> Result<usize, String> {
        if windows.is_empty() {
            return Err("qwen3_asr audio plan contains no feature windows".to_string());
        }
        let mut token_count = 0usize;
        for window in windows {
            if window.valid_frames == 0
                || window.valid_frames > window.padded_frames
                || window.padded_frames == 0
                || !window.padded_frames.is_multiple_of(MEL_FRAMES_PER_CHUNK)
            {
                return Err(format!(
                    "invalid qwen3_asr feature-window plan at frame {}: valid={}, padded={}",
                    window.start_frame, window.valid_frames, window.padded_frames
                ));
            }
            let window_tokens = (window.padded_frames / MEL_FRAMES_PER_CHUNK)
                .checked_mul(TOKENS_PER_CHUNK)
                .ok_or_else(|| "qwen3_asr planned embedding token count overflow".to_string())?;
            token_count = token_count
                .checked_add(window_tokens)
                .ok_or_else(|| "qwen3_asr planned embedding token count overflow".to_string())?;
        }
        Ok(token_count)
    }

    pub(crate) fn contract_summary(&self) -> String {
        format!(
            "qwen3_asr(dim={}, conv_channels={}, ff={}, layers={}, heads={}, mel_bins={}, positions={}, projector_hidden={}, output_dim={})",
            self.config.embedding_dim,
            self.config.convolution_channel_count,
            self.config.feed_forward_dim,
            self.config.layer_count,
            self.config.head_count,
            self.config.mel_bin_count,
            self.config.position_count,
            self.config.projector_hidden_dim,
            self.config.projection_dim,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        AudioProjectorWeights, AudioTransformerLayer, AudioTransformerWeights, Conv2dLayerWeights,
        ConvFeatureMap, Qwen3AsrAudioConfig, Qwen3AsrAudioEncoder, add_chunk_position_embeddings,
        conv2d_stride2_gelu_erf, f32_to_f16_bits, flatten_conv_features, gelu_erf,
        matmul_with_ggml_activation_type, project_to_language_embeddings, round_to_bf16,
        run_audio_transformer,
    };
    use crate::engine::audio::{PreparedAudioFeatureWindow, PreparedAudioFeatureWindowPlan};
    use crate::engine::multimodal::AudioEncoderFrontendOutput;
    use crate::engine::types::{
        GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGUFFile, GgmlType,
        GgufValue, Gguftensor, MappedFile, QuantizedTensor,
    };

    fn add_tensor(tensors: &mut Vec<Gguftensor>, name: impl Into<String>, shape: &[u64]) {
        let name = name.into();
        let mut ne = [1u64; 4];
        ne[..shape.len()].copy_from_slice(shape);
        tensors.push(Gguftensor {
            name,
            n_dims: shape.len() as u32,
            ne,
            ttype: GgmlType(GGML_TYPE_F16),
            offset: 0,
            data_offset: 0,
        });
    }

    fn synthetic_qwen3_asr_gguf() -> GGUFFile {
        const DIM: u64 = 8;
        const CONV_CHANNELS: u64 = 6;
        const FF_DIM: u64 = 16;
        const OUTPUT_DIM: u64 = 12;
        const PROJECTOR_HIDDEN: u64 = 10;
        let mut kv = HashMap::new();
        kv.insert(
            "general.architecture".to_string(),
            GgufValue::Str("clip".to_string()),
        );
        kv.insert("clip.has_audio_encoder".to_string(), GgufValue::Bool(true));
        kv.insert(
            "clip.audio.projector_type".to_string(),
            GgufValue::Str("qwen3a".to_string()),
        );
        for (key, value) in [
            ("clip.audio.embedding_length", DIM),
            ("clip.audio.feed_forward_length", FF_DIM),
            ("clip.audio.block_count", 2),
            ("clip.audio.attention.head_count", 2),
            ("clip.audio.num_mel_bins", 128),
            ("clip.audio.projection_dim", OUTPUT_DIM),
        ] {
            kv.insert(key.to_string(), GgufValue::UInt(value));
        }
        kv.insert(
            "clip.audio.attention.layer_norm_epsilon".to_string(),
            GgufValue::F32(1e-5),
        );

        let mut tensors = Vec::new();
        add_tensor(&mut tensors, "a.position_embd.weight", &[DIM, 32]);
        add_tensor(&mut tensors, "a.conv2d.1.weight", &[3, 3, 1, CONV_CHANNELS]);
        add_tensor(&mut tensors, "a.conv2d.1.bias", &[1, 1, CONV_CHANNELS]);
        for layer in 2..=3 {
            add_tensor(
                &mut tensors,
                format!("a.conv2d.{layer}.weight"),
                &[3, 3, CONV_CHANNELS, CONV_CHANNELS],
            );
            add_tensor(
                &mut tensors,
                format!("a.conv2d.{layer}.bias"),
                &[1, 1, CONV_CHANNELS],
            );
        }
        add_tensor(
            &mut tensors,
            "a.conv_out.weight",
            &[CONV_CHANNELS * 16, DIM],
        );
        for layer in 0..2 {
            let prefix = format!("a.blk.{layer}");
            add_tensor(&mut tensors, format!("{prefix}.ln1.weight"), &[DIM]);
            add_tensor(&mut tensors, format!("{prefix}.ln2.weight"), &[DIM]);
            for projection in ["attn_q", "attn_k", "attn_v", "attn_out"] {
                add_tensor(
                    &mut tensors,
                    format!("{prefix}.{projection}.weight"),
                    &[DIM, DIM],
                );
            }
            add_tensor(
                &mut tensors,
                format!("{prefix}.ffn_up.weight"),
                &[DIM, FF_DIM],
            );
            add_tensor(
                &mut tensors,
                format!("{prefix}.ffn_down.weight"),
                &[FF_DIM, DIM],
            );
        }
        add_tensor(&mut tensors, "a.post_ln.weight", &[DIM]);
        add_tensor(&mut tensors, "a.post_ln.bias", &[DIM]);
        add_tensor(&mut tensors, "mm.a.mlp.1.weight", &[DIM, PROJECTOR_HIDDEN]);
        add_tensor(&mut tensors, "mm.a.mlp.1.bias", &[PROJECTOR_HIDDEN]);
        add_tensor(
            &mut tensors,
            "mm.a.mlp.2.weight",
            &[PROJECTOR_HIDDEN, OUTPUT_DIM],
        );
        add_tensor(&mut tensors, "mm.a.mlp.2.bias", &[OUTPUT_DIM]);

        let tensor_lookup = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect::<HashMap<_, _>>();
        let mapped_bytes = Box::leak(vec![0u8; 16 * 1024].into_boxed_slice());
        GGUFFile {
            version: 3,
            n_tensors: tensors.len() as u64,
            n_kv: kv.len() as u64,
            kv,
            tensors,
            tensor_lookup,
            tensor_data_start: 0,
            vocab_tokens: Vec::new(),
            vocab_scores: Vec::new(),
            vocab_merges: Vec::new(),
            mapped: MappedFile::from_static(mapped_bytes).expect("static mapped bytes"),
        }
    }

    #[test]
    fn loads_and_types_complete_qwen3_asr_weight_contract() {
        let encoder = Qwen3AsrAudioEncoder::new(synthetic_qwen3_asr_gguf(), 12).unwrap();
        let config = encoder.config;

        assert_eq!(config.embedding_dim, 8);
        assert_eq!(config.convolution_channel_count, 6);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.position_count, 32);
        assert_eq!(config.projector_hidden_dim, 10);
        assert!(encoder.contract_summary().contains("output_dim=12"));
    }

    #[test]
    fn f16_rounding_matches_known_ieee_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
    }

    #[test]
    fn bf16_activation_rounding_matches_ggml_ties_to_even() {
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3f80_8001)).to_bits(),
            0x3f81_0000
        );
        assert!(round_to_bf16(f32::NAN).is_nan());
    }

    #[test]
    fn bf16_matmul_rounds_activations_before_the_dot_product() {
        let mapped = 0x3f80u16.to_le_bytes();
        let weight = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_BF16),
            rows: 1,
            cols: 1,
        };
        let input = [f32::from_bits(0x3f80_8001)];
        let mut output = [0.0f32];

        matmul_with_ggml_activation_type(&mut output, &input, &weight, &mapped).unwrap();

        assert_eq!(output[0].to_bits(), 0x3f81_0000);
    }

    #[test]
    fn gelu_erf_matches_reference_values() {
        assert_eq!(gelu_erf(0.0).to_bits(), 0.0f32.to_bits());
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-7);
        assert!((gelu_erf(-1.0) + 0.158_655_26).abs() < 1e-7);
    }

    #[test]
    fn convolution_uses_ggml_kernel_order_stride_and_padding() {
        let input = ConvFeatureMap {
            batch_count: 1,
            width: 3,
            height: 3,
            channels: 1,
            data: (1..=9).map(|value| value as f32).collect(),
        };
        let mut kernel = vec![0.0f32; 9];
        kernel[0] = 1.0;
        let weights = Conv2dLayerWeights {
            kernel,
            bias: vec![0.0],
            input_channels: 1,
            output_channels: 1,
            kernel_type: GgmlType(GGML_TYPE_F32),
        };

        let output = conv2d_stride2_gelu_erf(&input, &weights).unwrap();

        assert_eq!((output.batch_count, output.width, output.height), (1, 2, 2));
        assert_eq!(output.data[..3], [0.0, 0.0, 0.0]);
        assert!((output.data[3] - gelu_erf(5.0)).abs() < 1e-6);
    }

    #[test]
    fn flattening_matches_qwen3_asr_channel_then_frequency_order() {
        let mut data = Vec::new();
        for batch in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    for channel in 0..2 {
                        data.push((batch * 100 + y * 10 + x * 2 + channel) as f32);
                    }
                }
            }
        }
        let features = ConvFeatureMap {
            batch_count: 2,
            width: 2,
            height: 2,
            channels: 2,
            data,
        };

        let flattened = flatten_conv_features(&features).unwrap();

        assert_eq!(
            flattened,
            [
                0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 100.0, 110.0, 101.0, 111.0, 102.0,
                112.0, 103.0, 113.0,
            ]
        );
    }

    #[test]
    fn convolution_frontend_emits_thirteen_tokens_per_padded_chunk() {
        let encoder = Qwen3AsrAudioEncoder::new(synthetic_qwen3_asr_gguf(), 12).unwrap();
        let window = PreparedAudioFeatureWindow {
            start_frame: 0,
            valid_frames: 137,
            padded_frames: 200,
            mel_bins: 128,
            data_mel_major: vec![0.25; 128 * 200],
        };

        let output = encoder.encode_conv_frontend(&window).unwrap();

        assert_eq!(output.token_count, 26);
        assert_eq!(output.embedding_dim, 8);
        assert_eq!(output.data_token_major.len(), 26 * 8);
        assert!(output.data_token_major.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn positional_embeddings_reset_for_each_hundred_frame_chunk() {
        let mut position_embeddings = Vec::new();
        for position in 0..16 {
            position_embeddings.push(position as f32);
            position_embeddings.push(-(position as f32));
        }
        let mut output = AudioEncoderFrontendOutput {
            token_count: 26,
            embedding_dim: 2,
            data_token_major: vec![0.0; 52],
        };

        add_chunk_position_embeddings(&mut output, &position_embeddings, 16).unwrap();

        assert_eq!(&output.data_token_major[0..2], [0.0, 0.0]);
        assert_eq!(&output.data_token_major[24..26], [12.0, -12.0]);
        assert_eq!(&output.data_token_major[26..28], [0.0, 0.0]);
        assert_eq!(&output.data_token_major[50..52], [12.0, -12.0]);
    }

    #[test]
    fn zero_transformer_branches_preserve_the_residual_stream() {
        let encoder = Qwen3AsrAudioEncoder::new(synthetic_qwen3_asr_gguf(), 12).unwrap();
        let original = (0..24)
            .map(|index| index as f32 * 0.125 - 1.0)
            .collect::<Vec<_>>();
        let mut output = AudioEncoderFrontendOutput {
            token_count: 3,
            embedding_dim: 8,
            data_token_major: original.clone(),
        };

        run_audio_transformer(
            &mut output,
            &encoder.transformer,
            &encoder.config,
            encoder.gguf.mapped.as_slice(),
        )
        .unwrap();

        assert_eq!(output.data_token_major, original);
    }

    #[test]
    fn transformer_attention_is_bidirectional_across_the_full_window() {
        let identity = [1.0f32, 0.0, 0.0, 1.0];
        let zeros = [0.0f32; 4];
        let mut mapped = Vec::new();
        for value in identity.into_iter().chain(zeros) {
            mapped.extend_from_slice(&value.to_le_bytes());
        }
        let identity_weight = crate::engine::types::QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_F32),
            rows: 2,
            cols: 2,
        };
        let zero_weight = crate::engine::types::QuantizedTensor {
            data_offset: 16,
            ttype: GgmlType(GGML_TYPE_F32),
            rows: 2,
            cols: 2,
        };
        let layer = AudioTransformerLayer {
            ln1_weight: vec![1.0, 1.0],
            ln1_bias: vec![0.0, 0.0],
            ln2_weight: vec![1.0, 1.0],
            ln2_bias: vec![0.0, 0.0],
            query_weight: identity_weight.clone(),
            query_bias: vec![0.0, 0.0],
            key_weight: identity_weight.clone(),
            key_bias: vec![0.0, 0.0],
            value_weight: identity_weight.clone(),
            value_bias: vec![0.0, 0.0],
            output_weight: identity_weight,
            output_bias: vec![0.0, 0.0],
            feed_forward_up_weight: zero_weight.clone(),
            feed_forward_up_bias: vec![0.0, 0.0],
            feed_forward_down_weight: zero_weight,
            feed_forward_down_bias: vec![0.0, 0.0],
        };
        let weights = AudioTransformerWeights {
            position_embeddings: Vec::new(),
            layers: vec![layer],
        };
        let config = Qwen3AsrAudioConfig {
            embedding_dim: 2,
            convolution_channel_count: 1,
            feed_forward_dim: 2,
            layer_count: 1,
            head_count: 1,
            head_dim: 2,
            layer_norm_epsilon: 1e-5,
            mel_bin_count: 128,
            projection_dim: 2,
            position_count: 13,
            projector_hidden_dim: 2,
        };
        let mut output = AudioEncoderFrontendOutput {
            token_count: 2,
            embedding_dim: 2,
            data_token_major: vec![1.0, -1.0, -1.0, 1.0],
        };

        run_audio_transformer(&mut output, &weights, &config, &mapped).unwrap();

        let normalized_amplitude = 1.0 / (1.0f32 + config.layer_norm_epsilon).sqrt();
        let score = std::f32::consts::SQRT_2 * normalized_amplitude * normalized_amplitude;
        let context = normalized_amplitude * score.tanh();
        let expected = [1.0 + context, -1.0 - context, -1.0 - context, 1.0 + context];
        for (actual, expected) in output.data_token_major.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn transformer_frontend_keeps_chunk_token_shape() {
        let encoder = Qwen3AsrAudioEncoder::new(synthetic_qwen3_asr_gguf(), 12).unwrap();
        let window = PreparedAudioFeatureWindow {
            start_frame: 0,
            valid_frames: 137,
            padded_frames: 200,
            mel_bins: 128,
            data_mel_major: vec![0.25; 128 * 200],
        };

        let output = encoder.encode_transformer_frontend(&window).unwrap();

        assert_eq!(output.token_count, 26);
        assert_eq!(output.embedding_dim, 8);
        assert_eq!(output.data_token_major.len(), 26 * 8);
        assert!(output.data_token_major.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn projector_applies_up_gelu_down_with_ggml_matrix_orientation() {
        let up = [1.0f32, 0.0, 0.0, 1.0];
        let down = [1.0f32, 0.0, 0.0, 1.0, 1.0, -1.0];
        let mut mapped = Vec::new();
        for value in up.into_iter().chain(down) {
            mapped.extend_from_slice(&value.to_le_bytes());
        }
        let weights = AudioProjectorWeights {
            up_weight: QuantizedTensor {
                data_offset: 0,
                ttype: GgmlType(GGML_TYPE_F32),
                rows: 2,
                cols: 2,
            },
            up_bias: vec![0.5, 0.25],
            down_weight: QuantizedTensor {
                data_offset: 16,
                ttype: GgmlType(GGML_TYPE_F32),
                rows: 3,
                cols: 2,
            },
            down_bias: vec![0.1, -0.2, 0.3],
        };
        let frontend = AudioEncoderFrontendOutput {
            token_count: 1,
            embedding_dim: 2,
            data_token_major: vec![1.0, -2.0],
        };

        let output = project_to_language_embeddings(&frontend, &weights, 2, &mapped).unwrap();

        let hidden_up = gelu_erf(1.5);
        let hidden_down = gelu_erf(-1.75);
        let expected = [
            hidden_up + 0.1,
            hidden_down - 0.2,
            hidden_up - hidden_down + 0.3,
        ];
        assert_eq!(output.tokens.len(), 1);
        for (actual, expected) in output.tokens[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn complete_audio_sidecar_graph_emits_language_space_tokens() {
        let encoder = Qwen3AsrAudioEncoder::new(synthetic_qwen3_asr_gguf(), 12).unwrap();
        let window = PreparedAudioFeatureWindow {
            start_frame: 0,
            valid_frames: 137,
            padded_frames: 200,
            mel_bins: 128,
            data_mel_major: vec![0.25; 128 * 200],
        };

        let output = encoder.encode_feature_window(&window).unwrap();

        assert_eq!(output.tokens.len(), 26);
        assert!(output.tokens.iter().all(|token| token.len() == 12));
        assert!(output.tokens.iter().flatten().all(|value| *value == 0.0));
    }

    #[test]
    fn planned_embedding_count_matches_thirteen_tokens_per_padded_chunk() {
        let encoder = Qwen3AsrAudioEncoder::new(synthetic_qwen3_asr_gguf(), 12).unwrap();
        let windows = [
            PreparedAudioFeatureWindowPlan {
                start_frame: 0,
                valid_frames: 800,
                padded_frames: 800,
            },
            PreparedAudioFeatureWindowPlan {
                start_frame: 800,
                valid_frames: 1,
                padded_frames: 100,
            },
        ];

        assert_eq!(encoder.planned_embedding_token_count(&windows), Ok(9 * 13));
    }

    #[test]
    fn rejects_qwen3_asr_convolution_shape_mismatch() {
        let mut gguf = synthetic_qwen3_asr_gguf();
        let index = gguf.tensor_lookup["a.conv2d.2.weight"];
        gguf.tensors[index].ne[2] = 7;

        let error = Qwen3AsrAudioEncoder::new(gguf, 12)
            .err()
            .expect("shape mismatch should fail");

        assert!(error.contains("a.conv2d.2.weight shape mismatch"));
    }

    #[test]
    fn rejects_qwen3_asr_projector_discontinuity() {
        let mut gguf = synthetic_qwen3_asr_gguf();
        let index = gguf.tensor_lookup["mm.a.mlp.2.weight"];
        gguf.tensors[index].ne[0] = 9;

        let error = Qwen3AsrAudioEncoder::new(gguf, 12)
            .err()
            .expect("projector discontinuity should fail");

        assert!(error.contains("projector continuity"));
    }

    #[test]
    fn rejects_unsupported_qwen3_asr_tensor_type() {
        let mut gguf = synthetic_qwen3_asr_gguf();
        let index = gguf.tensor_lookup["a.position_embd.weight"];
        gguf.tensors[index].ttype = GgmlType(999);

        let error = Qwen3AsrAudioEncoder::new(gguf, 12)
            .err()
            .expect("unsupported tensor type should fail");

        assert!(error.contains("unsupported GGML type 999"));
    }

    #[test]
    fn rejects_misaligned_quantized_qwen3_asr_tensor_rows() {
        let mut gguf = synthetic_qwen3_asr_gguf();
        let index = gguf.tensor_lookup["a.blk.0.attn_q.weight"];
        gguf.tensors[index].ttype = GgmlType(GGML_TYPE_Q4_0);

        let error = Qwen3AsrAudioEncoder::new(gguf, 12)
            .err()
            .expect("misaligned quantization blocks should fail");

        assert!(error.contains("not divisible by GGML block size 32"));
    }
}
