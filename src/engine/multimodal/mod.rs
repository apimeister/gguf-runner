mod gemma3;
mod idefics3;
mod injection;
mod qwen3_asr;
mod qwen3vl;

use crate::engine::audio::{PreparedAudioFeatureWindow, PreparedAudioFeatureWindowPlan};
use crate::engine::kernels::{
    FloatBatchMatmulScratch, float_batch_supported, matmul_quantized, matmul_quantized_batch_float,
};
use crate::engine::switches::use_mm_float_batch;
use crate::engine::types::{
    AudioEncoderBackend, Config, GGUFFile, MultimodalBackend, QuantizedTensor,
};
use crate::engine::vision::PreparedImageTensor;
pub(crate) use injection::{
    MediaEmbeddingSequence, expand_prompt_with_media_embeddings, preflight_media_context,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

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
            matmul_quantized(destination, source, weight, mapped)
        })
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
