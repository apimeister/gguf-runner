use crate::engine::audio::{
    AudioDecodeConfig, DecodedAudio, WhisperLogMelConfig, decode_audio_file,
    extract_whisper_log_mel_windows,
};
use crate::engine::io::find_gguf_tensor;
use crate::engine::kernels::{
    dequantize_tensor, get_block_size, get_type_size, matmul_quantized, matmul_quantized_batch,
};
use crate::engine::types::{GGUFFile, Gguftensor, QuantizedTensor};

const MAX_SPEAKER_AUDIO_FILE_BYTES: usize = 1024 * 1024 * 1024;
const MAX_SPEAKER_AUDIO_CHANNELS: u16 = 64;
const MAX_SPEAKER_SOURCE_SAMPLE_RATE: u32 = 384_000;
const MAX_SPEAKER_FFT_LENGTH: usize = 4_096;
const MAX_SPEAKER_MEL_BINS: usize = 1_024;
const MAX_SPEAKER_FEATURE_WINDOW_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_SPEAKER_EMBEDDING_DIMENSION: usize = 8_192;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeakerThresholdPolicy {
    pub(crate) verification: f32,
    pub(crate) identification_margin: f32,
    pub(crate) enrollment: f32,
    pub(crate) auto_learning: f32,
    pub(crate) diarization_cluster: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpeakerTdnnLayerPolicy {
    pub(crate) weight_name: String,
    pub(crate) bias_name: String,
    pub(crate) contexts: Vec<isize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpeakerSegmentLayerPolicy {
    pub(crate) weight_name: String,
    pub(crate) bias_name: String,
    pub(crate) apply_relu: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeakerModelPolicy {
    pub(crate) architecture: String,
    pub(crate) model_name: String,
    pub(crate) sample_rate: u32,
    pub(crate) fft_length: usize,
    pub(crate) window_length: usize,
    pub(crate) hop_length: usize,
    pub(crate) mel_bins: usize,
    pub(crate) mel_floor: f32,
    pub(crate) max_window_frames: usize,
    pub(crate) min_audio_seconds: f32,
    pub(crate) max_audio_seconds: f32,
    pub(crate) tdnn_layers: Vec<SpeakerTdnnLayerPolicy>,
    pub(crate) segment_layers: Vec<SpeakerSegmentLayerPolicy>,
    pub(crate) thresholds: SpeakerThresholdPolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeakerAudioQuality {
    pub(crate) duration_seconds: f32,
    pub(crate) rms: f32,
    pub(crate) clipping_fraction: f32,
    pub(crate) active_fraction: f32,
    pub(crate) score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeakerEmbeddingOutput {
    pub(crate) vector: Vec<f32>,
    pub(crate) quality: SpeakerAudioQuality,
}

struct TdnnLayer {
    weight: QuantizedTensor,
    bias: Vec<f32>,
    contexts: Vec<isize>,
}

struct SegmentLayer {
    weight: QuantizedTensor,
    bias: Vec<f32>,
    apply_relu: bool,
}

pub(crate) struct SpeakerEncoder {
    gguf: GGUFFile,
    policy: SpeakerModelPolicy,
    fingerprint: String,
    tdnn_layers: Vec<TdnnLayer>,
    segment_layers: Vec<SegmentLayer>,
    embedding_dim: usize,
}

impl SpeakerEncoder {
    pub(crate) fn new(gguf: GGUFFile, policy: SpeakerModelPolicy) -> Result<Self, String> {
        validate_policy(&policy)?;
        let fingerprint = model_fingerprint(gguf.mapped.as_slice());
        let mut tdnn_layers = Vec::with_capacity(policy.tdnn_layers.len());
        let mut expected_input_channels = policy.mel_bins;
        for (index, layer_policy) in policy.tdnn_layers.iter().enumerate() {
            let tensor = find_gguf_tensor(&gguf, &layer_policy.weight_name)
                .ok_or_else(|| format!("speaker tensor not found: {}", layer_policy.weight_name))?;
            let (rows, cols) = matrix_shape(tensor, &layer_policy.weight_name)?;
            let expected_cols = expected_input_channels
                .checked_mul(layer_policy.contexts.len())
                .ok_or_else(|| format!("speaker TDNN layer {index} input shape overflow"))?;
            if cols != expected_cols {
                return Err(format!(
                    "speaker TDNN layer {index} '{}' has {cols} input columns, expected {expected_cols} (channels={expected_input_channels}, contexts={})",
                    layer_policy.weight_name,
                    layer_policy.contexts.len()
                ));
            }
            let bias = load_float_tensor(&gguf, &layer_policy.bias_name, Some(rows))?;
            tdnn_layers.push(TdnnLayer {
                weight: quantized_tensor(tensor, rows, cols),
                bias,
                contexts: layer_policy.contexts.clone(),
            });
            expected_input_channels = rows;
        }

        let mut expected_segment_input = expected_input_channels
            .checked_mul(2)
            .ok_or_else(|| "speaker statistics-pooling shape overflow".to_string())?;
        let mut segment_layers = Vec::with_capacity(policy.segment_layers.len());
        for (index, layer_policy) in policy.segment_layers.iter().enumerate() {
            let tensor = find_gguf_tensor(&gguf, &layer_policy.weight_name)
                .ok_or_else(|| format!("speaker tensor not found: {}", layer_policy.weight_name))?;
            let (rows, cols) = matrix_shape(tensor, &layer_policy.weight_name)?;
            if cols != expected_segment_input {
                return Err(format!(
                    "speaker segment layer {index} '{}' has {cols} input columns, expected {expected_segment_input}",
                    layer_policy.weight_name
                ));
            }
            let bias = load_float_tensor(&gguf, &layer_policy.bias_name, Some(rows))?;
            segment_layers.push(SegmentLayer {
                weight: quantized_tensor(tensor, rows, cols),
                bias,
                apply_relu: layer_policy.apply_relu,
            });
            expected_segment_input = rows;
        }
        let embedding_dim = expected_segment_input;
        Ok(Self {
            gguf,
            policy,
            fingerprint,
            tdnn_layers,
            segment_layers,
            embedding_dim,
        })
    }

    pub(crate) fn model_name(&self) -> &str {
        &self.policy.model_name
    }

    pub(crate) fn architecture(&self) -> &str {
        &self.policy.architecture
    }

    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub(crate) fn thresholds(&self) -> &SpeakerThresholdPolicy {
        &self.policy.thresholds
    }

    pub(crate) fn min_audio_seconds(&self) -> f32 {
        self.policy.min_audio_seconds
    }

    pub(crate) fn decode_file(&self, path: &str) -> Result<DecodedAudio, String> {
        let max_output_samples = seconds_to_samples(
            self.policy.max_audio_seconds,
            self.policy.sample_rate,
            "speaker maximum audio duration",
        )?;
        decode_audio_file(
            path,
            AudioDecodeConfig {
                target_sample_rate: self.policy.sample_rate,
                max_output_samples,
                chunk_size_samples: max_output_samples,
                max_file_bytes: MAX_SPEAKER_AUDIO_FILE_BYTES,
                max_channels: MAX_SPEAKER_AUDIO_CHANNELS,
                max_source_sample_rate: MAX_SPEAKER_SOURCE_SAMPLE_RATE,
            },
        )
    }

    pub(crate) fn embed_file(&self, path: &str) -> Result<SpeakerEmbeddingOutput, String> {
        let decoded = self.decode_file(path)?;
        self.embed_samples(&decoded.samples_mono_f32)
            .map_err(|error| format!("cannot create speaker embedding for '{path}': {error}"))
    }

    pub(crate) fn embed_samples(&self, samples: &[f32]) -> Result<SpeakerEmbeddingOutput, String> {
        let quality = analyze_quality(samples, self.policy.sample_rate)?;
        validate_quality(&quality, self.policy.min_audio_seconds)?;
        let feature_config = WhisperLogMelConfig {
            sample_rate: self.policy.sample_rate,
            fft_length: self.policy.fft_length,
            window_length: self.policy.window_length,
            hop_length: self.policy.hop_length,
            mel_bins: self.policy.mel_bins,
            mel_floor: self.policy.mel_floor,
            max_window_frames: self.policy.max_window_frames,
            frame_chunk_size: 1,
        };
        let windows = extract_whisper_log_mel_windows(samples, feature_config)?;
        let minimum_frames = seconds_to_frames(
            self.policy.min_audio_seconds,
            self.policy.sample_rate,
            self.policy.hop_length,
        )?;
        let mut combined = vec![0.0f32; self.embedding_dim];
        let mut total_weight = 0.0f32;
        for window in windows {
            if window.valid_frames < minimum_frames {
                continue;
            }
            let embedding = self.embed_feature_window(
                &window.data_mel_major,
                window.valid_frames,
                window.padded_frames,
            )?;
            let weight = window.valid_frames as f32;
            for (destination, value) in combined.iter_mut().zip(embedding) {
                *destination += value * weight;
            }
            total_weight += weight;
        }
        if total_weight == 0.0 {
            return Err(format!(
                "audio contains no speaker window of at least {:.2} seconds",
                self.policy.min_audio_seconds
            ));
        }
        for value in &mut combined {
            *value /= total_weight;
        }
        l2_normalize(&mut combined)?;
        Ok(SpeakerEmbeddingOutput {
            vector: combined,
            quality,
        })
    }

    fn embed_feature_window(
        &self,
        mel_major: &[f32],
        valid_frames: usize,
        padded_frames: usize,
    ) -> Result<Vec<f32>, String> {
        let expected_mel_elements = self
            .policy
            .mel_bins
            .checked_mul(padded_frames)
            .ok_or_else(|| "speaker feature-window shape overflow".to_string())?;
        if mel_major.len() != expected_mel_elements {
            return Err("speaker feature-window shape mismatch".to_string());
        }
        let feature_elements = valid_frames
            .checked_mul(self.policy.mel_bins)
            .ok_or_else(|| "speaker feature-buffer shape overflow".to_string())?;
        let mut features = vec![0.0f32; feature_elements];
        for mel in 0..self.policy.mel_bins {
            let source = &mel_major[mel * padded_frames..mel * padded_frames + valid_frames];
            let mean = source.iter().sum::<f32>() / valid_frames as f32;
            for frame in 0..valid_frames {
                features[frame * self.policy.mel_bins + mel] = source[frame] - mean;
            }
        }

        let mapped = self.gguf.mapped.as_slice();
        let mut frame_count = valid_frames;
        let mut channels = self.policy.mel_bins;
        let mut activations = features;
        for layer in &self.tdnn_layers {
            let context_width = channels
                .checked_mul(layer.contexts.len())
                .ok_or_else(|| "speaker TDNN context shape overflow".to_string())?;
            let context_elements = frame_count
                .checked_mul(context_width)
                .ok_or_else(|| "speaker TDNN context-buffer shape overflow".to_string())?;
            let mut context = vec![0.0f32; context_elements];
            for frame in 0..frame_count {
                for (context_index, offset) in layer.contexts.iter().enumerate() {
                    let source_frame = frame
                        .saturating_add_signed(*offset)
                        .min(frame_count.saturating_sub(1));
                    let source =
                        &activations[source_frame * channels..(source_frame + 1) * channels];
                    let destination_start = frame * context_width + context_index * channels;
                    context[destination_start..destination_start + channels]
                        .copy_from_slice(source);
                }
            }
            let output_channels = layer.weight.rows;
            let output_elements = frame_count
                .checked_mul(output_channels)
                .ok_or_else(|| "speaker TDNN output-buffer shape overflow".to_string())?;
            let mut output = vec![0.0f32; output_elements];
            matmul_quantized_batch(
                &mut output,
                &context,
                &layer.weight,
                mapped,
                frame_count,
                0,
                output_channels,
            )?;
            for row in output.chunks_mut(output_channels) {
                for (value, bias) in row.iter_mut().zip(&layer.bias) {
                    *value = (*value + *bias).max(0.0);
                }
            }
            activations = output;
            channels = output_channels;
            frame_count = activations.len() / channels;
        }

        let pooled_elements = channels
            .checked_mul(2)
            .ok_or_else(|| "speaker statistics-pooling shape overflow".to_string())?;
        let mut pooled = vec![0.0f32; pooled_elements];
        for channel in 0..channels {
            let mut mean = 0.0f32;
            for frame in 0..frame_count {
                mean += activations[frame * channels + channel];
            }
            mean /= frame_count as f32;
            let mut variance = 0.0f32;
            for frame in 0..frame_count {
                let delta = activations[frame * channels + channel] - mean;
                variance += delta * delta;
            }
            variance /= frame_count as f32;
            pooled[channel] = mean;
            pooled[channels + channel] = (variance + 1e-7).sqrt();
        }

        let mut vector = pooled;
        for layer in &self.segment_layers {
            let mut output = vec![0.0f32; layer.weight.rows];
            matmul_quantized(&mut output, &vector, &layer.weight, mapped)?;
            for (value, bias) in output.iter_mut().zip(&layer.bias) {
                *value += *bias;
                if layer.apply_relu {
                    *value = value.max(0.0);
                }
            }
            vector = output;
        }
        l2_normalize(&mut vector)?;
        Ok(vector)
    }
}

fn validate_policy(policy: &SpeakerModelPolicy) -> Result<(), String> {
    if policy.sample_rate == 0
        || policy.sample_rate > MAX_SPEAKER_SOURCE_SAMPLE_RATE
        || policy.fft_length < 2
        || policy.fft_length > MAX_SPEAKER_FFT_LENGTH
        || policy.window_length == 0
        || policy.hop_length == 0
        || policy.mel_bins == 0
        || policy.mel_bins > MAX_SPEAKER_MEL_BINS
        || policy.max_window_frames == 0
    {
        return Err("speaker model contains invalid feature dimensions".to_string());
    }
    if policy.window_length > policy.fft_length || !policy.fft_length.is_multiple_of(2) {
        return Err("speaker model FFT/window configuration is invalid".to_string());
    }
    if !policy.mel_floor.is_finite() || policy.mel_floor <= 0.0 {
        return Err("speaker model mel floor must be finite and positive".to_string());
    }
    if !policy.min_audio_seconds.is_finite()
        || !policy.max_audio_seconds.is_finite()
        || policy.min_audio_seconds <= 0.0
        || policy.max_audio_seconds < policy.min_audio_seconds
    {
        return Err("speaker model audio-duration limits are invalid".to_string());
    }
    let window_elements = policy
        .mel_bins
        .checked_mul(policy.max_window_frames)
        .ok_or_else(|| "speaker model feature-window shape overflow".to_string())?;
    if window_elements > MAX_SPEAKER_FEATURE_WINDOW_ELEMENTS {
        return Err(format!(
            "speaker model feature window has {window_elements} elements, exceeding {MAX_SPEAKER_FEATURE_WINDOW_ELEMENTS}"
        ));
    }
    let minimum_frames = seconds_to_frames(
        policy.min_audio_seconds,
        policy.sample_rate,
        policy.hop_length,
    )?;
    if minimum_frames > policy.max_window_frames {
        return Err(format!(
            "speaker model minimum duration needs {minimum_frames} frames, exceeding its {}-frame window",
            policy.max_window_frames
        ));
    }
    if policy.tdnn_layers.is_empty() || policy.segment_layers.is_empty() {
        return Err("speaker model requires TDNN and segment layers".to_string());
    }
    if policy
        .tdnn_layers
        .iter()
        .any(|layer| layer.contexts.is_empty())
    {
        return Err("speaker TDNN contexts must not be empty".to_string());
    }
    let thresholds = &policy.thresholds;
    for (name, value) in [
        ("verification", thresholds.verification),
        ("enrollment", thresholds.enrollment),
        ("auto learning", thresholds.auto_learning),
        ("diarization cluster", thresholds.diarization_cluster),
    ] {
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(format!(
                "speaker model {name} threshold must be within -1..=1"
            ));
        }
    }
    if !thresholds.identification_margin.is_finite()
        || !(0.0..=2.0).contains(&thresholds.identification_margin)
    {
        return Err("speaker model identification margin must be within 0..=2".to_string());
    }
    if thresholds.auto_learning < thresholds.verification {
        return Err(
            "speaker model auto-learning threshold must be at least the verification threshold"
                .to_string(),
        );
    }
    Ok(())
}

fn matrix_shape(tensor: &Gguftensor, name: &str) -> Result<(usize, usize), String> {
    if tensor.n_dims != 2 {
        return Err(format!(
            "speaker tensor {name} has rank {}, expected 2",
            tensor.n_dims
        ));
    }
    let cols = usize::try_from(tensor.ne[0])
        .map_err(|_| format!("speaker tensor {name} column count does not fit this platform"))?;
    let rows = usize::try_from(tensor.ne[1])
        .map_err(|_| format!("speaker tensor {name} row count does not fit this platform"))?;
    if cols == 0 || rows == 0 {
        return Err(format!("speaker tensor {name} has an empty dimension"));
    }
    if rows > MAX_SPEAKER_EMBEDDING_DIMENSION && name.starts_with("speaker.segment.") {
        return Err(format!(
            "speaker segment tensor {name} output dimension {rows} exceeds {MAX_SPEAKER_EMBEDDING_DIMENSION}"
        ));
    }
    Ok((rows, cols))
}

fn quantized_tensor(tensor: &Gguftensor, rows: usize, cols: usize) -> QuantizedTensor {
    QuantizedTensor {
        data_offset: tensor.data_offset,
        ttype: tensor.ttype,
        rows,
        cols,
    }
}

fn tensor_element_count(tensor: &Gguftensor) -> Result<usize, String> {
    (0..tensor.n_dims as usize).try_fold(1usize, |total, index| {
        total
            .checked_mul(tensor.ne[index] as usize)
            .ok_or_else(|| format!("speaker tensor {} element count overflow", tensor.name))
    })
}

fn load_float_tensor(
    gguf: &GGUFFile,
    name: &str,
    expected_elements: Option<usize>,
) -> Result<Vec<f32>, String> {
    let tensor =
        find_gguf_tensor(gguf, name).ok_or_else(|| format!("speaker tensor not found: {name}"))?;
    let elements = tensor_element_count(tensor)?;
    if let Some(expected) = expected_elements
        && elements != expected
    {
        return Err(format!(
            "speaker tensor {name} has {elements} elements, expected {expected}"
        ));
    }
    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if block_size == 0 || type_size == 0 || !elements.is_multiple_of(block_size) {
        return Err(format!(
            "speaker tensor {name} uses unsupported or invalid GGML type {}",
            tensor.ttype.0
        ));
    }
    let byte_len = elements / block_size * type_size;
    let end = tensor
        .data_offset
        .checked_add(byte_len)
        .ok_or_else(|| format!("speaker tensor {name} data range overflow"))?;
    let mapped = gguf.mapped.as_slice();
    if end > mapped.len() {
        return Err(format!("speaker tensor {name} exceeds mapped model data"));
    }
    dequantize_tensor(&mapped[tensor.data_offset..end], elements, tensor.ttype)
}

fn analyze_quality(samples: &[f32], sample_rate: u32) -> Result<SpeakerAudioQuality, String> {
    if samples.is_empty() {
        return Err("speaker audio is empty".to_string());
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("speaker audio contains non-finite samples".to_string());
    }
    let duration_seconds = samples.len() as f32 / sample_rate as f32;
    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    let rms = mean_square.sqrt();
    let clipping_fraction = samples
        .iter()
        .filter(|sample| sample.abs() >= 0.999)
        .count() as f32
        / samples.len() as f32;
    let frame_samples = ((sample_rate as usize) * 20 / 1000).max(1);
    let active_threshold = (rms * 0.25).max(1e-4);
    let mut active = 0usize;
    let mut frames = 0usize;
    for frame in samples.chunks(frame_samples) {
        let frame_rms =
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32).sqrt();
        active += usize::from(frame_rms >= active_threshold);
        frames += 1;
    }
    let active_fraction = active as f32 / frames as f32;
    let level_score = (rms / 0.05).clamp(0.0, 1.0);
    let clipping_score = (1.0 - clipping_fraction / 0.05).clamp(0.0, 1.0);
    let activity_score = (active_fraction / 0.6).clamp(0.0, 1.0);
    let score = (level_score * clipping_score * activity_score).cbrt();
    Ok(SpeakerAudioQuality {
        duration_seconds,
        rms,
        clipping_fraction,
        active_fraction,
        score,
    })
}

fn validate_quality(quality: &SpeakerAudioQuality, min_seconds: f32) -> Result<(), String> {
    if quality.duration_seconds < min_seconds {
        return Err(format!(
            "speaker audio is too short: {:.2}s; model requires at least {min_seconds:.2}s",
            quality.duration_seconds
        ));
    }
    if quality.rms < 1e-4 {
        return Err("speaker audio is silent or too quiet".to_string());
    }
    if quality.clipping_fraction > 0.20 {
        return Err(format!(
            "speaker audio is heavily clipped ({:.1}% of samples)",
            quality.clipping_fraction * 100.0
        ));
    }
    if quality.active_fraction < 0.10 {
        return Err(format!(
            "speaker audio contains too little active signal ({:.1}%)",
            quality.active_fraction * 100.0
        ));
    }
    Ok(())
}

fn seconds_to_samples(seconds: f32, sample_rate: u32, field: &str) -> Result<usize, String> {
    let samples = f64::from(seconds) * f64::from(sample_rate);
    if !samples.is_finite() || samples < 1.0 || samples > usize::MAX as f64 {
        return Err(format!("{field} does not fit this platform"));
    }
    Ok(samples.ceil() as usize)
}

fn seconds_to_frames(seconds: f32, sample_rate: u32, hop_length: usize) -> Result<usize, String> {
    let samples = seconds_to_samples(seconds, sample_rate, "speaker minimum audio duration")?;
    Ok(samples.div_ceil(hop_length).max(1))
}

fn l2_normalize(vector: &mut [f32]) -> Result<(), String> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 1e-8 {
        return Err("speaker encoder produced a zero or non-finite embedding".to_string());
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn model_fingerprint(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("fnv1a64:{hash:016x}:{}", bytes.len())
}

pub(crate) fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.len() != right.len() || left.is_empty() {
        return Err(format!(
            "speaker embedding dimension mismatch: left={}, right={}",
            left.len(),
            right.len()
        ));
    }
    let score = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    if score.is_finite() {
        Ok(score.clamp(-1.0, 1.0))
    } else {
        Err("speaker similarity is non-finite".to_string())
    }
}

pub(crate) fn normalize_embedding(vector: &mut [f32]) -> Result<(), String> {
    l2_normalize(vector)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        SpeakerAudioQuality, SpeakerEncoder, SpeakerModelPolicy, SpeakerSegmentLayerPolicy,
        SpeakerTdnnLayerPolicy, SpeakerThresholdPolicy, cosine_similarity, model_fingerprint,
        validate_quality,
    };
    use crate::engine::types::{GGML_TYPE_F32, GGUFFile, GgmlType, Gguftensor, MappedFile};

    fn synthetic_encoder() -> SpeakerEncoder {
        let tensors_data = [
            1.0f32, 0.0, 0.0, 1.0, // TDNN identity weight.
            0.0, 0.0, // TDNN bias.
            1.0, 0.0, 0.0, 0.0, // Segment row 0 selects mean channel 0.
            0.0, 0.0, 1.0, 0.0, // Segment row 1 selects std channel 0.
            0.0, 0.0, // Segment bias.
        ];
        let bytes = tensors_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mapped = MappedFile::from_static(Box::leak(bytes)).unwrap();
        let specifications = [
            ("speaker.tdnn.0.weight", 2, [2, 2, 1, 1], 0),
            ("speaker.tdnn.0.bias", 1, [2, 1, 1, 1], 16),
            ("speaker.segment.0.weight", 2, [4, 2, 1, 1], 24),
            ("speaker.segment.0.bias", 1, [2, 1, 1, 1], 56),
        ];
        let tensors = specifications
            .iter()
            .map(|(name, n_dims, ne, offset)| Gguftensor {
                name: (*name).to_string(),
                n_dims: *n_dims,
                ne: *ne,
                ttype: GgmlType(GGML_TYPE_F32),
                offset: *offset as u64,
                data_offset: *offset,
            })
            .collect::<Vec<_>>();
        let tensor_lookup = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        let gguf = GGUFFile {
            version: 3,
            n_tensors: tensors.len() as u64,
            n_kv: 0,
            kv: HashMap::new(),
            tensors,
            tensor_lookup,
            tensor_data_start: 0,
            vocab_tokens: Vec::new(),
            vocab_scores: Vec::new(),
            vocab_merges: Vec::new(),
            mapped,
        };
        let policy = SpeakerModelPolicy {
            architecture: "speaker_xvector".to_string(),
            model_name: "synthetic".to_string(),
            sample_rate: 16_000,
            fft_length: 400,
            window_length: 400,
            hop_length: 160,
            mel_bins: 2,
            mel_floor: 1e-8,
            max_window_frames: 100,
            min_audio_seconds: 0.01,
            max_audio_seconds: 10.0,
            tdnn_layers: vec![SpeakerTdnnLayerPolicy {
                weight_name: "speaker.tdnn.0.weight".to_string(),
                bias_name: "speaker.tdnn.0.bias".to_string(),
                contexts: vec![0],
            }],
            segment_layers: vec![SpeakerSegmentLayerPolicy {
                weight_name: "speaker.segment.0.weight".to_string(),
                bias_name: "speaker.segment.0.bias".to_string(),
                apply_relu: false,
            }],
            thresholds: SpeakerThresholdPolicy {
                verification: 0.6,
                identification_margin: 0.05,
                enrollment: 0.4,
                auto_learning: 0.8,
                diarization_cluster: 0.5,
            },
        };
        SpeakerEncoder::new(gguf, policy).unwrap()
    }

    #[test]
    fn cosine_similarity_handles_normalized_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap()).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0], &[1.0, 0.0]).is_err());
    }

    #[test]
    fn fingerprint_is_stable_and_content_sensitive() {
        assert_eq!(model_fingerprint(b"speaker"), model_fingerprint(b"speaker"));
        assert_ne!(model_fingerprint(b"speaker"), model_fingerprint(b"speakeq"));
    }

    #[test]
    fn quality_rejects_short_and_silent_audio() {
        let short = SpeakerAudioQuality {
            duration_seconds: 0.2,
            rms: 0.1,
            clipping_fraction: 0.0,
            active_fraction: 1.0,
            score: 1.0,
        };
        assert!(validate_quality(&short, 1.0).is_err());
        let silent = SpeakerAudioQuality {
            duration_seconds: 2.0,
            rms: 0.0,
            clipping_fraction: 0.0,
            active_fraction: 0.0,
            score: 0.0,
        };
        assert!(validate_quality(&silent, 1.0).is_err());
    }

    #[test]
    fn synthetic_xvector_runs_tdnn_stats_pooling_and_projection() {
        let encoder = synthetic_encoder();
        let embedding = encoder
            .embed_feature_window(&[1.0, 3.0, 2.0, 2.0], 2, 2)
            .unwrap();
        assert_eq!(embedding.len(), 2);
        assert!((embedding[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        assert!((embedding[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }
}
