mod gemma3;
mod idefics3;
mod injection;
mod qwen3_asr;
mod qwen3vl;

use crate::engine::audio::{PreparedAudioFeatureWindow, PreparedAudioFeatureWindowPlan};
use crate::engine::io::{bf16_to_fp32, fp16_to_fp32, fp32_to_bf16, fp32_to_fp16};
use crate::engine::kernels::{
    FloatBatchMatmulScratch, float_batch_supported, matmul_quantized, matmul_quantized_batch_float,
};
use crate::engine::switches::use_mm_float_batch;
use crate::engine::types::{
    AudioEncoderBackend, Config, GGML_TYPE_BF16, GGML_TYPE_F16, GGUFFile, MultimodalBackend,
    QuantizedTensor,
};
use crate::engine::vision::PreparedImageTensor;
pub(crate) use injection::{
    MediaEmbeddingSequence, expand_prompt_with_media_embeddings, preflight_media_context,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
#[cfg(target_os = "macos")]
use std::ffi::c_int;

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
    fn vvexpf(output: *mut f32, input: *const f32, count: *const c_int);
}

#[cfg(any(target_os = "macos", test))]
const ATTENTION_QUERY_BLOCK: usize = 128;

#[inline]
fn round_to_f16(value: f32) -> f32 {
    fp16_to_fp32(fp32_to_fp16(value))
}

#[inline]
fn round_to_bf16(value: f32) -> f32 {
    bf16_to_fp32(fp32_to_bf16(value))
}

/// Match GGML's activation dot type for floating-point encoder matrices.
/// Other tensor types retain the existing F32 activation path.
fn matmul_encoder(
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

/// Run a full token batch when an encoder matrix is stored as F16/BF16.
/// Returns `false` for other tensor types so the backend can preserve its
/// architecture-specific per-token path.
fn try_matmul_float_batch(
    output: &mut [f32],
    input: &[f32],
    weight: &QuantizedTensor,
    mapped: &[u8],
    token_count: usize,
    scratch: &mut FloatBatchMatmulScratch,
) -> Result<bool, String> {
    if token_count <= 1 || !use_mm_float_batch() || !float_batch_supported(weight.ttype) {
        return Ok(false);
    }
    matmul_quantized_batch_float(
        output,
        input,
        weight,
        mapped,
        token_count,
        0,
        weight.rows,
        scratch,
    )?;
    Ok(true)
}

/// Token-major encoder matmul with a dequantize-once F16/BF16 path and the
/// existing parallel per-token behavior as the fallback for other types.
/// Both paths narrow floating-point activations to GGML's matrix dot type.
fn matmul_encoder_batch(
    output: &mut [f32],
    input: &[f32],
    weight: &QuantizedTensor,
    mapped: &[u8],
    token_count: usize,
    scratch: &mut FloatBatchMatmulScratch,
) -> Result<(), String> {
    if try_matmul_float_batch(output, input, weight, mapped, token_count, scratch)? {
        return Ok(());
    }
    output
        .par_chunks_mut(weight.rows)
        .enumerate()
        .try_for_each(|(token, destination)| {
            let source = &input[token * weight.cols..(token + 1) * weight.cols];
            matmul_encoder(destination, source, weight, mapped)
        })
}

/// Reusable storage for the blocked multimodal encoder-attention kernel.
#[derive(Default)]
struct EncoderAttentionScratch {
    #[cfg(target_os = "macos")]
    scores: Vec<f32>,
    #[cfg(target_os = "macos")]
    probabilities: Vec<f32>,
    #[cfg(target_os = "macos")]
    head_output: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn encoder_self_attention(
    output: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence_count: usize,
    tokens_per_sequence: usize,
    head_count: usize,
    head_dim: usize,
    scratch: &mut EncoderAttentionScratch,
) -> Result<(), String> {
    let dim = head_count
        .checked_mul(head_dim)
        .ok_or_else(|| "encoder attention dimension overflow".to_string())?;
    let total_tokens = sequence_count
        .checked_mul(tokens_per_sequence)
        .ok_or_else(|| "encoder attention token count overflow".to_string())?;
    let required = total_tokens
        .checked_mul(dim)
        .ok_or_else(|| "encoder attention buffer size overflow".to_string())?;
    if output.len() < required
        || query.len() < required
        || key.len() < required
        || value.len() < required
    {
        return Err(format!(
            "encoder attention shape mismatch: sequences={sequence_count} tokens={tokens_per_sequence} heads={head_count} head_dim={head_dim}"
        ));
    }
    if required == 0 {
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    if tokens_per_sequence > 1 {
        return encoder_self_attention_accelerate(
            &mut output[..required],
            &query[..required],
            &key[..required],
            &value[..required],
            sequence_count,
            tokens_per_sequence,
            head_count,
            head_dim,
            scratch,
        );
    }

    let _ = scratch;
    encoder_self_attention_portable(
        &mut output[..required],
        &query[..required],
        &key[..required],
        &value[..required],
        tokens_per_sequence,
        head_count,
        head_dim,
    );
    Ok(())
}

fn encoder_self_attention_portable(
    output: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens_per_sequence: usize,
    head_count: usize,
    head_dim: usize,
) {
    let dim = head_count * head_dim;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    output
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(global_token, token_output)| {
            let sequence_start = (global_token / tokens_per_sequence) * tokens_per_sequence;
            for head in 0..head_count {
                let head_offset = head * head_dim;
                let destination = &mut token_output[head_offset..head_offset + head_dim];
                let query_start = global_token * dim + head_offset;
                let query_values = &query[query_start..query_start + head_dim];

                destination.fill(0.0);
                let mut maximum_score = f32::NEG_INFINITY;
                let mut score_sum = 0.0f32;
                for token in 0..tokens_per_sequence {
                    let source_start = (sequence_start + token) * dim + head_offset;
                    let key_values = &key[source_start..source_start + head_dim];
                    let score =
                        crate::engine::kernels::dot_f32_simd(query_values, key_values) * inv_scale;
                    if score > maximum_score {
                        if score_sum > 0.0 {
                            let rescale = (maximum_score - score).exp();
                            crate::engine::kernels::scale_slice_inplace(destination, rescale);
                            score_sum *= rescale;
                        }
                        maximum_score = score;
                    }
                    let attention_weight = (score - maximum_score).exp();
                    score_sum += attention_weight;
                    let values = &value[source_start..source_start + head_dim];
                    crate::engine::kernels::axpy_inplace(destination, attention_weight, values);
                }
                if score_sum > 0.0 {
                    crate::engine::kernels::scale_slice_inplace(destination, 1.0 / score_sum);
                }
            }
        });
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encoder_self_attention_accelerate(
    output: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence_count: usize,
    tokens_per_sequence: usize,
    head_count: usize,
    head_dim: usize,
    scratch: &mut EncoderAttentionScratch,
) -> Result<(), String> {
    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    let dim = head_count * head_dim;
    let tokens_i32 = c_int::try_from(tokens_per_sequence)
        .map_err(|_| "encoder attention token count exceeds c_int".to_string())?;
    let head_dim_i32 = c_int::try_from(head_dim)
        .map_err(|_| "encoder attention head dimension exceeds c_int".to_string())?;
    let dim_i32 =
        c_int::try_from(dim).map_err(|_| "encoder attention stride exceeds c_int".to_string())?;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let block_capacity = ATTENTION_QUERY_BLOCK.min(tokens_per_sequence);
    let score_capacity = block_capacity
        .checked_mul(tokens_per_sequence)
        .ok_or_else(|| "encoder attention score scratch overflow".to_string())?;
    let head_output_capacity = block_capacity
        .checked_mul(head_dim)
        .ok_or_else(|| "encoder attention output scratch overflow".to_string())?;
    scratch.scores.resize(score_capacity, 0.0);
    scratch.probabilities.resize(score_capacity, 0.0);
    scratch.head_output.resize(head_output_capacity, 0.0);

    for sequence in 0..sequence_count {
        let sequence_start = sequence * tokens_per_sequence;
        for head in 0..head_count {
            let head_offset = head * head_dim;
            let key_ptr = unsafe { key.as_ptr().add(sequence_start * dim + head_offset) };
            let value_ptr = unsafe { value.as_ptr().add(sequence_start * dim + head_offset) };

            for query_start in (0..tokens_per_sequence).step_by(ATTENTION_QUERY_BLOCK) {
                let rows = (tokens_per_sequence - query_start).min(ATTENTION_QUERY_BLOCK);
                let rows_i32 = c_int::try_from(rows)
                    .map_err(|_| "encoder attention query block exceeds c_int".to_string())?;
                let score_len = rows * tokens_per_sequence;
                let score_len_i32 = c_int::try_from(score_len)
                    .map_err(|_| "encoder attention score block exceeds c_int".to_string())?;
                let head_output_len = rows * head_dim;
                let scores = &mut scratch.scores[..score_len];
                let probabilities = &mut scratch.probabilities[..score_len];
                let head_output = &mut scratch.head_output[..head_output_len];
                let query_ptr = unsafe {
                    query
                        .as_ptr()
                        .add((sequence_start + query_start) * dim + head_offset)
                };

                unsafe {
                    cblas_sgemm(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_TRANS,
                        rows_i32,
                        tokens_i32,
                        head_dim_i32,
                        inv_scale,
                        query_ptr,
                        dim_i32,
                        key_ptr,
                        dim_i32,
                        0.0,
                        scores.as_mut_ptr(),
                        tokens_i32,
                    );
                }

                scores
                    .par_chunks_mut(tokens_per_sequence)
                    .for_each(|score_row| {
                        let mut maximum_score = f32::NEG_INFINITY;
                        for &score in score_row.iter() {
                            if score > maximum_score {
                                maximum_score = score;
                            }
                        }
                        for score in score_row.iter_mut() {
                            *score -= maximum_score;
                        }
                    });
                unsafe {
                    vvexpf(probabilities.as_mut_ptr(), scores.as_ptr(), &score_len_i32);
                }
                probabilities
                    .par_chunks_mut(tokens_per_sequence)
                    .for_each(|probability_row| {
                        let score_sum = probability_row.iter().sum::<f32>();
                        if score_sum > 0.0 {
                            let inverse = 1.0 / score_sum;
                            for probability in probability_row {
                                *probability *= inverse;
                            }
                        }
                    });

                unsafe {
                    cblas_sgemm(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_NO_TRANS,
                        rows_i32,
                        head_dim_i32,
                        tokens_i32,
                        1.0,
                        probabilities.as_ptr(),
                        tokens_i32,
                        value_ptr,
                        dim_i32,
                        0.0,
                        head_output.as_mut_ptr(),
                        head_dim_i32,
                    );
                }

                for row in 0..rows {
                    let source = &head_output[row * head_dim..(row + 1) * head_dim];
                    let destination_start =
                        (sequence_start + query_start + row) * dim + head_offset;
                    output[destination_start..destination_start + head_dim].copy_from_slice(source);
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct AudioEncoderFrontendOutput {
    pub(crate) token_count: usize,
    pub(crate) embedding_dim: usize,
    /// Token-major storage: `[token_count][embedding_dim]`.
    pub(crate) data_token_major: Vec<f32>,
}

pub(crate) enum AudioEncoder {
    Qwen3Asr(qwen3_asr::Qwen3AsrAudioEncoder),
}

impl AudioEncoder {
    pub(crate) fn backend(&self) -> AudioEncoderBackend {
        match self {
            AudioEncoder::Qwen3Asr(_) => AudioEncoderBackend::Qwen3Asr,
        }
    }

    pub(crate) fn execution_ready(&self) -> bool {
        true
    }

    pub(crate) fn contract_summary(&self) -> String {
        match self {
            AudioEncoder::Qwen3Asr(encoder) => encoder.contract_summary(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn encode_conv_frontend(
        &self,
        window: &PreparedAudioFeatureWindow,
    ) -> Result<AudioEncoderFrontendOutput, String> {
        match self {
            AudioEncoder::Qwen3Asr(encoder) => encoder.encode_conv_frontend(window),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn encode_transformer_frontend(
        &self,
        window: &PreparedAudioFeatureWindow,
    ) -> Result<AudioEncoderFrontendOutput, String> {
        match self {
            AudioEncoder::Qwen3Asr(encoder) => encoder.encode_transformer_frontend(window),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn encode_feature_window(
        &self,
        window: &PreparedAudioFeatureWindow,
    ) -> Result<MediaEmbeddingSequence, String> {
        match self {
            AudioEncoder::Qwen3Asr(encoder) => encoder.encode_feature_window(window),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn encode_from_conv_output(
        &self,
        data_token_major: Vec<f32>,
    ) -> Result<(AudioEncoderFrontendOutput, MediaEmbeddingSequence), String> {
        match self {
            AudioEncoder::Qwen3Asr(encoder) => encoder.encode_from_conv_output(data_token_major),
        }
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
        match self {
            AudioEncoder::Qwen3Asr(encoder) => {
                encoder.encode_from_conv_output_with_layers(data_token_major)
            }
        }
    }

    pub(crate) fn planned_embedding_token_count(
        &self,
        windows: &[PreparedAudioFeatureWindowPlan],
    ) -> Result<usize, String> {
        match self {
            AudioEncoder::Qwen3Asr(encoder) => encoder.planned_embedding_token_count(windows),
        }
    }
}

pub(crate) enum VisionEncoder {
    Gemma3(gemma3::Gemma3VisionEncoder),
    Qwen3Vl(qwen3vl::Qwen3VlVisionEncoder),
    Idefics3(idefics3::Idefics3VisionEncoder),
}

impl VisionEncoder {
    pub(crate) fn recommended_image_size(&self) -> usize {
        match self {
            VisionEncoder::Gemma3(enc) => enc.recommended_image_size(),
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_size(),
            VisionEncoder::Idefics3(enc) => enc.recommended_image_size(),
        }
    }

    pub(crate) fn recommended_image_alignment(&self) -> usize {
        match self {
            VisionEncoder::Gemma3(enc) => enc.recommended_image_alignment(),
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_alignment(),
            VisionEncoder::Idefics3(enc) => enc.recommended_image_alignment(),
        }
    }

    pub(crate) fn recommended_image_normalization(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            VisionEncoder::Gemma3(enc) => enc.recommended_image_normalization(),
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_normalization(),
            VisionEncoder::Idefics3(enc) => enc.recommended_image_normalization(),
        }
    }

    pub(crate) fn encode_images(
        &self,
        images: &[PreparedImageTensor],
    ) -> Result<Vec<MediaEmbeddingSequence>, String> {
        match self {
            VisionEncoder::Gemma3(enc) => enc.encode_images(images),
            VisionEncoder::Qwen3Vl(enc) => enc.encode_images(images),
            VisionEncoder::Idefics3(enc) => enc.encode_images(images),
        }
    }
}

pub(crate) fn build_vision_encoder_from_mmproj(
    cfg: &Config,
    mmproj: GGUFFile,
) -> Result<Option<VisionEncoder>, String> {
    match cfg.capabilities.multimodal_backend {
        MultimodalBackend::Gemma3 => {
            let encoder = gemma3::Gemma3VisionEncoder::new(mmproj, cfg.dim)?;
            Ok(Some(VisionEncoder::Gemma3(encoder)))
        }
        MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
            let encoder =
                qwen3vl::Qwen3VlVisionEncoder::new(mmproj, cfg.dim, cfg.n_deepstack_layers)?;
            Ok(Some(VisionEncoder::Qwen3Vl(encoder)))
        }
        MultimodalBackend::Idefics3 => {
            let encoder = idefics3::Idefics3VisionEncoder::new(mmproj, cfg.dim)?;
            Ok(Some(VisionEncoder::Idefics3(encoder)))
        }
        _ => Ok(None),
    }
}

pub(crate) fn build_audio_encoder_from_mmproj(
    backend: AudioEncoderBackend,
    text_embedding_dim: usize,
    mmproj: GGUFFile,
) -> Result<AudioEncoder, String> {
    match backend {
        AudioEncoderBackend::Qwen3Asr => {
            qwen3_asr::Qwen3AsrAudioEncoder::new(mmproj, text_embedding_dim)
                .map(AudioEncoder::Qwen3Asr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EncoderAttentionScratch, FloatBatchMatmulScratch, encoder_self_attention,
        encoder_self_attention_portable, matmul_encoder_batch,
    };
    use crate::engine::types::{GGML_TYPE_BF16, GGML_TYPE_F16, GgmlType, QuantizedTensor};

    #[test]
    fn f16_encoder_fallback_rounds_activation_to_matrix_dot_type() {
        let weight = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_F16),
            rows: 1,
            cols: 1,
        };
        let input = [f32::from_bits(0x3f80_1000)];
        let mut output = [0.0f32];

        matmul_encoder_batch(
            &mut output,
            &input,
            &weight,
            &0x3c00u16.to_le_bytes(),
            1,
            &mut FloatBatchMatmulScratch::default(),
        )
        .unwrap();

        assert_eq!(output[0].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn bf16_encoder_fallback_rounds_activation_to_matrix_dot_type() {
        let weight = QuantizedTensor {
            data_offset: 0,
            ttype: GgmlType(GGML_TYPE_BF16),
            rows: 1,
            cols: 1,
        };
        let input = [f32::from_bits(0x3f80_8001)];
        let mut output = [0.0f32];

        matmul_encoder_batch(
            &mut output,
            &input,
            &weight,
            &0x3f80u16.to_le_bytes(),
            1,
            &mut FloatBatchMatmulScratch::default(),
        )
        .unwrap();

        assert_eq!(output[0].to_bits(), 0x3f81_0000);
    }

    #[test]
    fn encoder_attention_matches_reference_and_isolates_sequences() {
        let sequence_count = 2usize;
        let tokens_per_sequence = 3usize;
        let head_count = 2usize;
        let head_dim = 2usize;
        let dim = head_count * head_dim;
        let total_tokens = sequence_count * tokens_per_sequence;
        let query = (0..total_tokens * dim)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.125)
            .collect::<Vec<_>>();
        let key = (0..total_tokens * dim)
            .map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.1)
            .collect::<Vec<_>>();
        let value = (0..total_tokens * dim)
            .map(|i| {
                let sequence = i / (tokens_per_sequence * dim);
                sequence as f32 * 100.0 + (i % (tokens_per_sequence * dim)) as f32 * 0.25
            })
            .collect::<Vec<_>>();

        let mut got = vec![0.0f32; total_tokens * dim];
        encoder_self_attention(
            &mut got,
            &query,
            &key,
            &value,
            sequence_count,
            tokens_per_sequence,
            head_count,
            head_dim,
            &mut EncoderAttentionScratch::default(),
        )
        .unwrap();

        let inv_scale = 1.0 / (head_dim as f32).sqrt();
        let mut want = vec![0.0f32; got.len()];
        for sequence in 0..sequence_count {
            let sequence_start = sequence * tokens_per_sequence;
            for query_token in 0..tokens_per_sequence {
                for head in 0..head_count {
                    let head_offset = head * head_dim;
                    let query_start = (sequence_start + query_token) * dim + head_offset;
                    let mut scores = vec![0.0f32; tokens_per_sequence];
                    for (key_token, score) in scores.iter_mut().enumerate() {
                        let key_start = (sequence_start + key_token) * dim + head_offset;
                        *score = (0..head_dim)
                            .map(|i| query[query_start + i] * key[key_start + i])
                            .sum::<f32>()
                            * inv_scale;
                    }
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights = scores
                        .iter()
                        .map(|score| (*score - maximum).exp())
                        .collect::<Vec<_>>();
                    let sum = weights.iter().sum::<f32>();
                    for (key_token, weight) in weights.iter().enumerate() {
                        let value_start = (sequence_start + key_token) * dim + head_offset;
                        for i in 0..head_dim {
                            want[query_start + i] += *weight / sum * value[value_start + i];
                        }
                    }
                }
            }
        }

        for (index, (&actual, &expected)) in got.iter().zip(&want).enumerate() {
            let tolerance = 2e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
        assert!(
            got[..tokens_per_sequence * dim]
                .iter()
                .all(|value| *value < 10.0)
        );
        assert!(
            got[tokens_per_sequence * dim..]
                .iter()
                .all(|value| *value > 90.0)
        );
    }

    #[test]
    fn encoder_attention_handles_partial_query_block() {
        let tokens = super::ATTENTION_QUERY_BLOCK + 1;
        let heads = 2usize;
        let head_dim = 4usize;
        let dim = heads * head_dim;
        let query = (0..tokens * dim)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let key = (0..tokens * dim)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) * 0.025)
            .collect::<Vec<_>>();
        let value = (0..tokens * dim)
            .map(|i| ((i * 19 % 37) as f32 - 18.0) * 0.05)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; tokens * dim];
        encoder_self_attention_portable(
            &mut expected,
            &query,
            &key,
            &value,
            tokens,
            heads,
            head_dim,
        );
        let mut actual = vec![0.0f32; tokens * dim];
        encoder_self_attention(
            &mut actual,
            &query,
            &key,
            &value,
            1,
            tokens,
            heads,
            head_dim,
            &mut EncoderAttentionScratch::default(),
        )
        .unwrap();

        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 3e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "index={index}: actual={got} expected={want} tolerance={tolerance}"
            );
        }
    }
}
