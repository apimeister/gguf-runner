use std::fs;

use super::features::{extract_whisper_log_mel_windows, plan_whisper_log_mel_windows};
use super::{
    AudioDecodeConfig, AudioPreprocessConfig, DecodedAudio, PreparedAudioChunk,
    PreparedAudioInputPlan, PreparedAudioTensor, WhisperLogMelConfig,
};

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xfffe;
/// Fixed tail of the `KSDATAFORMAT_SUBTYPE_*` GUIDs, shared by the PCM and
/// IEEE-float sub-formats that WAVE_FORMAT_EXTENSIBLE wraps.
const SUBFORMAT_GUID_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];
const CONVERSION_HINT: &str =
    "convert it first, for example `ffmpeg -i input -ar 16000 -ac 1 -c:a pcm_s16le output.wav`";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PcmSampleFormat {
    Integer,
    Float,
}

#[derive(Clone, Copy, Debug)]
struct PcmWaveFormat {
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    sample_format: PcmSampleFormat,
}

fn read_u16_le(bytes: &[u8], offset: usize, field: &str) -> Result<u16, String> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| format!("truncated WAV {field}"))?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize, field: &str) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("truncated WAV {field}"))?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

/// Reads the sample format out of a `WAVE_FORMAT_EXTENSIBLE` fmt chunk, which
/// is what most writers emit for more than two channels or more than 16 bits.
fn parse_extensible_sample_format(bytes: &[u8]) -> Result<PcmSampleFormat, String> {
    if bytes.len() < 40 {
        return Err(format!(
            "invalid WAVE_FORMAT_EXTENSIBLE fmt chunk: expected at least 40 bytes, found {}",
            bytes.len()
        ));
    }
    let extension_size = read_u16_le(bytes, 16, "fmt extension size")?;
    if extension_size < 22 {
        return Err(format!(
            "invalid WAVE_FORMAT_EXTENSIBLE fmt extension size {extension_size}; expected at least 22"
        ));
    }
    let bits_per_sample = read_u16_le(bytes, 14, "bits per sample")?;
    let valid_bits_per_sample = read_u16_le(bytes, 18, "valid bits per sample")?;
    if valid_bits_per_sample != 0 && valid_bits_per_sample != bits_per_sample {
        return Err(format!(
            "unsupported WAVE_FORMAT_EXTENSIBLE sample packing: {valid_bits_per_sample} valid bits in a {bits_per_sample}-bit container"
        ));
    }
    if bytes[26..40] != SUBFORMAT_GUID_TAIL {
        return Err(
            "unsupported WAVE_FORMAT_EXTENSIBLE sub-format GUID; expected PCM or IEEE float"
                .to_string(),
        );
    }
    match read_u16_le(bytes, 24, "sub-format")? {
        WAVE_FORMAT_PCM => Ok(PcmSampleFormat::Integer),
        WAVE_FORMAT_IEEE_FLOAT => Ok(PcmSampleFormat::Float),
        other => Err(format!(
            "unsupported WAVE_FORMAT_EXTENSIBLE sub-format {other}; expected PCM (1) or IEEE float (3)"
        )),
    }
}

fn parse_pcm_format(bytes: &[u8], config: AudioPreprocessConfig) -> Result<PcmWaveFormat, String> {
    if bytes.len() < 16 {
        return Err("invalid WAV fmt chunk: expected at least 16 bytes".to_string());
    }
    let encoding = read_u16_le(bytes, 0, "encoding")?;
    let sample_format = match encoding {
        WAVE_FORMAT_PCM => PcmSampleFormat::Integer,
        WAVE_FORMAT_IEEE_FLOAT => PcmSampleFormat::Float,
        WAVE_FORMAT_EXTENSIBLE => parse_extensible_sample_format(bytes)?,
        other => {
            return Err(format!(
                "unsupported WAV encoding {other}; supported encodings are integer PCM (1), IEEE float (3), and extensible (65534) wrapping either; {CONVERSION_HINT}"
            ));
        }
    };
    let channels = read_u16_le(bytes, 2, "channel count")?;
    if channels == 0 || channels > config.decode.max_channels {
        return Err(format!(
            "unsupported WAV channel count {channels}; expected 1..={}",
            config.decode.max_channels
        ));
    }
    let sample_rate = read_u32_le(bytes, 4, "sample rate")?;
    if sample_rate == 0 || sample_rate > config.decode.max_source_sample_rate {
        return Err(format!(
            "unsupported WAV sample rate {sample_rate}; expected 1..={}",
            config.decode.max_source_sample_rate
        ));
    }
    let block_align = read_u16_le(bytes, 12, "block alignment")?;
    let bits_per_sample = read_u16_le(bytes, 14, "bits per sample")?;
    match sample_format {
        PcmSampleFormat::Integer if !matches!(bits_per_sample, 8 | 16 | 24 | 32) => {
            return Err(format!(
                "unsupported integer PCM WAV bit depth {bits_per_sample}; expected 8, 16, 24, or 32"
            ));
        }
        PcmSampleFormat::Float if !matches!(bits_per_sample, 32 | 64) => {
            return Err(format!(
                "unsupported IEEE float WAV bit depth {bits_per_sample}; expected 32 or 64"
            ));
        }
        _ => {}
    }
    let bytes_per_sample = bits_per_sample / 8;
    let expected_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| "WAV block alignment overflow".to_string())?;
    if block_align != expected_align {
        return Err(format!(
            "invalid PCM WAV block alignment {block_align}; expected {expected_align}"
        ));
    }
    let byte_rate = read_u32_le(bytes, 8, "byte rate")?;
    let expected_byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| "WAV byte rate overflow".to_string())?;
    if byte_rate != expected_byte_rate {
        return Err(format!(
            "invalid PCM WAV byte rate {byte_rate}; expected {expected_byte_rate}"
        ));
    }

    Ok(PcmWaveFormat {
        channels,
        sample_rate,
        block_align,
        bits_per_sample,
        sample_format,
    })
}

fn parse_wave_chunks(
    bytes: &[u8],
    config: AudioPreprocessConfig,
) -> Result<(PcmWaveFormat, &[u8]), String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!(
            "unsupported audio container; expected little-endian RIFF/WAVE; {CONVERSION_HINT}"
        ));
    }

    // Writers that cannot seek, such as `ffmpeg -f wav -`, leave the RIFF and
    // data sizes at 0xffffffff and rely on the reader stopping at the end of
    // the stream. Bound every chunk by the bytes actually present.
    let declared_size = usize::try_from(read_u32_le(bytes, 4, "RIFF size")?)
        .map_err(|_| "WAV RIFF size does not fit this platform".to_string())?;
    let declared_end = declared_size.saturating_add(8).min(bytes.len());

    let mut format = None;
    let mut data = None;
    let mut data_reaches_end_of_file = false;
    let mut cursor = 12usize;
    while cursor + 8 <= declared_end {
        let id = &bytes[cursor..cursor + 4];
        let declared_chunk_size = usize::try_from(read_u32_le(bytes, cursor + 4, "chunk size")?)
            .map_err(|_| "WAV chunk size does not fit this platform".to_string())?;
        let payload_start = cursor + 8;
        let available = declared_end - payload_start;
        let chunk_size = declared_chunk_size.min(available);
        let payload_end = payload_start + chunk_size;

        if id == b"fmt " && format.is_none() {
            format = Some(parse_pcm_format(
                &bytes[payload_start..payload_end],
                config,
            )?);
        } else if id == b"data" && data.is_none() {
            data = Some(&bytes[payload_start..payload_end]);
            data_reaches_end_of_file = chunk_size < declared_chunk_size;
        }

        if chunk_size < declared_chunk_size {
            break;
        }
        cursor = payload_end
            .checked_add(chunk_size & 1)
            .ok_or_else(|| "WAV chunk padding overflow".to_string())?;
    }

    let format = format.ok_or_else(|| "WAV is missing a fmt chunk".to_string())?;
    let mut data = data.ok_or_else(|| "WAV is missing a data chunk".to_string())?;
    let frame_bytes = usize::from(format.block_align);
    if data_reaches_end_of_file {
        // The declared length ran past the file, so keep the complete frames.
        data = &data[..data.len() - data.len() % frame_bytes];
    } else if data.len() % frame_bytes != 0 {
        return Err("PCM WAV data length is not aligned to complete sample frames".to_string());
    }
    if data.is_empty() {
        return Err("WAV data chunk is empty".to_string());
    }
    Ok((format, data))
}

fn decode_integer_sample(bytes: &[u8], bits_per_sample: u16) -> f32 {
    match bits_per_sample {
        8 => (f32::from(bytes[0]) - 128.0) / 128.0,
        16 => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0,
        24 => {
            let raw =
                i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
            let signed = if raw & 0x0080_0000 != 0 {
                raw | !0x00ff_ffff
            } else {
                raw
            };
            signed as f32 / 8_388_608.0
        }
        32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32 / 2_147_483_648.0,
        _ => unreachable!("bit depth validated before sample decode"),
    }
}

/// IEEE float samples already use the runtime's `[-1.0, 1.0]` convention, so
/// they pass through at their stored value like the pinned miniaudio path.
fn decode_float_sample(bytes: &[u8], bits_per_sample: u16) -> f32 {
    match bits_per_sample {
        32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        64 => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f32,
        _ => unreachable!("bit depth validated before sample decode"),
    }
}

fn decode_sample(bytes: &[u8], format: PcmWaveFormat) -> f32 {
    match format.sample_format {
        PcmSampleFormat::Integer => decode_integer_sample(bytes, format.bits_per_sample),
        PcmSampleFormat::Float => decode_float_sample(bytes, format.bits_per_sample),
    }
}

fn decode_and_downmix(data: &[u8], format: PcmWaveFormat) -> Vec<f32> {
    let bytes_per_sample = usize::from(format.bits_per_sample / 8);
    let block_align = usize::from(format.block_align);
    let channels = usize::from(format.channels);
    let mut mono = Vec::with_capacity(data.len() / block_align);
    for frame in data.chunks_exact(block_align) {
        let mut sum = 0.0f64;
        for channel in 0..channels {
            let start = channel * bytes_per_sample;
            sum += f64::from(decode_sample(
                &frame[start..start + bytes_per_sample],
                format,
            ));
        }
        mono.push((sum / f64::from(format.channels)) as f32);
    }
    mono
}

fn resampled_length(input_len: usize, source_rate: u32, target_rate: u32) -> Result<usize, String> {
    let numerator = (input_len as u128)
        .checked_mul(u128::from(target_rate))
        .ok_or_else(|| "resampled audio length overflow".to_string())?;
    let rounded = numerator
        .checked_add(u128::from(source_rate / 2))
        .ok_or_else(|| "resampled audio length overflow".to_string())?
        / u128::from(source_rate);
    usize::try_from(rounded.max(1)).map_err(|_| "resampled audio is too long".to_string())
}

#[derive(Clone, Copy)]
struct BiquadLowPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    r1: f32,
    r2: f32,
}

impl BiquadLowPass {
    fn new(sample_rate: f64, cutoff_frequency: f64, q: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * cutoff_frequency / sample_rate;
        let sin = w.sin();
        let cos = w.cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: (((1.0 - cos) / 2.0) / a0) as f32,
            b1: ((1.0 - cos) / a0) as f32,
            b2: (((1.0 - cos) / 2.0) / a0) as f32,
            a1: ((-2.0 * cos) / a0) as f32,
            a2: ((1.0 - alpha) / a0) as f32,
            r1: 0.0,
            r2: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        // Direct form II transposed, matching the f32 filter used by miniaudio.
        let output = self.b0 * input + self.r1;
        self.r1 = self.b1 * input - self.a1 * output + self.r2;
        self.r2 = self.b2 * input - self.a2 * output;
        output
    }
}

struct ButterworthLowPass4 {
    sections: [BiquadLowPass; 2],
}

impl ButterworthLowPass4 {
    fn new(source_rate: u32, target_rate: u32) -> Self {
        let sample_rate = f64::from(source_rate.max(target_rate));
        let cutoff_frequency = f64::from(source_rate.min(target_rate)) * 0.5;
        let sections = std::array::from_fn(|index| {
            let angle = (1 + index * 2) as f64 * std::f64::consts::PI / 8.0;
            let q = 1.0 / (2.0 * angle.cos());
            BiquadLowPass::new(sample_rate, cutoff_frequency, q)
        });
        Self { sections }
    }

    fn process(&mut self, mut input: f32) -> f32 {
        for section in &mut self.sections {
            input = section.process(input);
        }
        input
    }
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn resample_linear_low_pass(
    input: &[f32],
    source_rate: u32,
    target_rate: u32,
    output_len: usize,
) -> Vec<f32> {
    if source_rate == target_rate {
        return input.to_vec();
    }
    let divisor = greatest_common_divisor(source_rate, target_rate);
    let rate_in = source_rate / divisor;
    let rate_out = target_rate / divisor;
    let advance_integer = rate_in / rate_out;
    let advance_fraction = rate_in % rate_out;
    let mut time_integer = 1u32;
    let mut time_fraction = 0u32;
    let mut input_index = 0usize;
    let mut previous = 0.0f32;
    let mut next = 0.0f32;
    let mut low_pass = ButterworthLowPass4::new(rate_in, rate_out);
    let downsampling = rate_in > rate_out;
    let mut output = Vec::with_capacity(output_len);

    while output.len() < output_len {
        while time_integer > 0 && input_index < input.len() {
            previous = next;
            next = input[input_index];
            input_index += 1;
            if downsampling {
                next = low_pass.process(next);
            }
            time_integer -= 1;
        }
        if time_integer > 0 {
            break;
        }

        let fraction = time_fraction as f32 / rate_out as f32;
        let mut value = previous + (next - previous) * fraction;
        if !downsampling {
            value = low_pass.process(value);
        }
        output.push(value);

        time_integer += advance_integer;
        time_fraction += advance_fraction;
        if time_fraction >= rate_out {
            time_fraction -= rate_out;
            time_integer += 1;
        }
    }
    output
}

fn plan_chunks(total_samples: usize, chunk_size: usize) -> Vec<PreparedAudioChunk> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total_samples {
        let sample_count = (total_samples - start).min(chunk_size);
        chunks.push(PreparedAudioChunk {
            start_sample: start,
            sample_count,
        });
        start += sample_count;
    }
    chunks
}

fn validate_config(config: AudioPreprocessConfig) -> Result<(), String> {
    if config.decode.target_sample_rate == 0 {
        return Err("audio target_sample_rate must be > 0".to_string());
    }
    if config.decode.max_output_samples == 0 {
        return Err("audio max_output_samples must be > 0".to_string());
    }
    if config.decode.chunk_size_samples == 0 {
        return Err("audio chunk_size_samples must be > 0".to_string());
    }
    if config.decode.max_file_bytes < 12 {
        return Err("audio max_file_bytes must be at least 12".to_string());
    }
    if config.decode.max_channels == 0 {
        return Err("audio max_channels must be > 0".to_string());
    }
    if config.decode.max_source_sample_rate == 0 {
        return Err("audio max_source_sample_rate must be > 0".to_string());
    }
    if config.features.sample_rate != config.decode.target_sample_rate {
        return Err(format!(
            "audio feature sample rate {} does not match decode target {}",
            config.features.sample_rate, config.decode.target_sample_rate
        ));
    }
    Ok(())
}

fn read_bounded_audio_file(path: &str, config: AudioPreprocessConfig) -> Result<Vec<u8>, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("cannot read audio file '{path}': {error}"))?;
    let file_len = usize::try_from(metadata.len())
        .map_err(|_| format!("audio file '{path}' is too large for this platform"))?;
    if file_len > config.decode.max_file_bytes {
        return Err(format!(
            "audio file '{path}' is too large: {file_len} bytes exceeds limit {}",
            config.decode.max_file_bytes
        ));
    }
    fs::read(path).map_err(|error| format!("cannot read audio file '{path}': {error}"))
}

fn checked_output_sample_count(
    path: &str,
    source_sample_count: usize,
    source_sample_rate: u32,
    config: AudioPreprocessConfig,
) -> Result<usize, String> {
    let output_len = resampled_length(
        source_sample_count,
        source_sample_rate,
        config.decode.target_sample_rate,
    )?;
    if output_len > config.decode.max_output_samples {
        return Err(format!(
            "audio file '{path}' is too long after resampling: {output_len} samples exceeds limit {}",
            config.decode.max_output_samples
        ));
    }
    Ok(output_len)
}

fn probe_one(path: &str, config: AudioPreprocessConfig) -> Result<PreparedAudioInputPlan, String> {
    let bytes = read_bounded_audio_file(path, config)?;
    let (format, data) = parse_wave_chunks(&bytes, config)
        .map_err(|error| format!("cannot decode audio file '{path}': {error}"))?;
    let source_sample_count = data.len() / usize::from(format.block_align);
    let target_sample_count =
        checked_output_sample_count(path, source_sample_count, format.sample_rate, config)?;
    let feature_windows = plan_whisper_log_mel_windows(target_sample_count, config.features)
        .map_err(|error| format!("cannot plan audio features for '{path}': {error}"))?;

    Ok(PreparedAudioInputPlan {
        path: path.to_string(),
        source_sample_rate: format.sample_rate,
        source_channels: format.channels,
        target_sample_rate: config.decode.target_sample_rate,
        target_sample_count,
        feature_windows,
    })
}

fn prepare_one(path: &str, config: AudioPreprocessConfig) -> Result<PreparedAudioTensor, String> {
    let decoded = decode_audio_file(path, config.decode)?;
    let data_mono_f32 = decoded.samples_mono_f32;
    let feature_windows = extract_whisper_log_mel_windows(&data_mono_f32, config.features)
        .map_err(|error| format!("cannot extract audio features from '{path}': {error}"))?;
    let total_samples = data_mono_f32.len();
    let chunks = plan_chunks(total_samples, config.decode.chunk_size_samples);

    Ok(PreparedAudioTensor {
        path: path.to_string(),
        source_sample_rate: decoded.source_sample_rate,
        source_channels: decoded.source_channels,
        sample_rate: decoded.sample_rate,
        channels: 1,
        total_samples,
        chunks,
        feature_windows,
        data_mono_f32,
    })
}

pub(crate) fn decode_audio_file(
    path: &str,
    decode_config: AudioDecodeConfig,
) -> Result<DecodedAudio, String> {
    let placeholder_features = WhisperLogMelConfig {
        sample_rate: decode_config.target_sample_rate,
        fft_length: 400,
        window_length: 400,
        hop_length: 160,
        mel_bins: 1,
        mel_floor: f32::MIN_POSITIVE,
        max_window_frames: 1,
        frame_chunk_size: 1,
    };
    let config = AudioPreprocessConfig {
        decode: decode_config,
        features: placeholder_features,
    };
    validate_config(config)?;
    let bytes = read_bounded_audio_file(path, config)?;
    let (format, data) = parse_wave_chunks(&bytes, config)
        .map_err(|error| format!("cannot decode audio file '{path}': {error}"))?;
    let source_sample_count = data.len() / usize::from(format.block_align);
    let output_len =
        checked_output_sample_count(path, source_sample_count, format.sample_rate, config)?;
    let source_samples = decode_and_downmix(data, format);
    let samples_mono_f32 = resample_linear_low_pass(
        &source_samples,
        format.sample_rate,
        decode_config.target_sample_rate,
        output_len,
    );
    Ok(DecodedAudio {
        source_sample_rate: format.sample_rate,
        source_channels: format.channels,
        sample_rate: decode_config.target_sample_rate,
        samples_mono_f32,
    })
}

pub(crate) fn probe_audios_for_multimodal(
    audio_paths: &[String],
    config: AudioPreprocessConfig,
) -> Result<Vec<PreparedAudioInputPlan>, String> {
    validate_config(config)?;
    audio_paths
        .iter()
        .map(|path| probe_one(path, config))
        .collect()
}

pub(crate) fn prepare_audios_for_multimodal(
    audio_paths: &[String],
    config: AudioPreprocessConfig,
) -> Result<Vec<PreparedAudioTensor>, String> {
    validate_config(config)?;
    audio_paths
        .iter()
        .map(|path| prepare_one(path, config))
        .collect()
}

pub(crate) fn load_audio_chunk_samples(
    audio: &PreparedAudioTensor,
    chunk_idx: usize,
) -> Result<Vec<f32>, String> {
    let chunk = audio
        .chunks
        .get(chunk_idx)
        .ok_or_else(|| format!("audio chunk index out of range: {chunk_idx}"))?;
    let end = chunk
        .start_sample
        .checked_add(chunk.sample_count)
        .ok_or_else(|| "audio chunk range overflow".to_string())?;
    if end > audio.data_mono_f32.len() {
        return Err(format!(
            "audio chunk range out of bounds: end={} > total_samples={}",
            end,
            audio.data_mono_f32.len()
        ));
    }
    Ok(audio.data_mono_f32[chunk.start_sample..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::{
        SUBFORMAT_GUID_TAIL, WAVE_FORMAT_EXTENSIBLE, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM,
        decode_integer_sample, load_audio_chunk_samples, prepare_audios_for_multimodal,
        probe_audios_for_multimodal, resample_linear_low_pass,
    };
    use crate::engine::audio::{AudioDecodeConfig, AudioPreprocessConfig, WhisperLogMelConfig};
    use tempfile::TempDir;

    fn test_config() -> AudioPreprocessConfig {
        AudioPreprocessConfig {
            decode: AudioDecodeConfig {
                target_sample_rate: 16_000,
                max_output_samples: 64_000,
                chunk_size_samples: 3,
                max_file_bytes: 1_000_000,
                max_channels: 8,
                max_source_sample_rate: 192_000,
            },
            features: WhisperLogMelConfig {
                sample_rate: 16_000,
                fft_length: 400,
                window_length: 400,
                hop_length: 160,
                mel_bins: 128,
                mel_floor: 5.960_464_5e-8,
                max_window_frames: 800,
                frame_chunk_size: 100,
            },
        }
    }

    /// Byte offset of the `data` chunk size field in a canonical 44-byte
    /// RIFF/WAVE header, as written by `pcm16_wave`.
    const DATA_SIZE_OFFSET: usize = 40;

    fn pcm16_wave(sample_rate: u32, channels: u16, interleaved: &[i16]) -> Vec<u8> {
        assert_eq!(interleaved.len() % usize::from(channels), 0);
        let data_len = interleaved.len() * 2;
        let block_align = channels * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
        for sample in interleaved {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    /// Builds the header shape that writers emit for more than two channels or
    /// more than 16 bits: encoding 65534 plus a sub-format GUID.
    fn extensible_wave(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        sub_format: u16,
        guid_tail: [u8; 14],
        payload: &[u8],
    ) -> Vec<u8> {
        let block_align = channels * (bits_per_sample / 8);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(60u32 + payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&WAVE_FORMAT_EXTENSIBLE.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
        bytes.extend_from_slice(&22u16.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&sub_format.to_le_bytes());
        bytes.extend_from_slice(&guid_tail);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn float32_wave(sample_rate: u32, channels: u16, interleaved: &[f32]) -> Vec<u8> {
        let payload = interleaved
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let block_align = channels * 4;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&WAVE_FORMAT_IEEE_FLOAT.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&32u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn write_temp_wave(temp: &TempDir, name: &str, bytes: &[u8]) -> String {
        let path = temp.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn integer_pcm_scaling_covers_supported_bit_depths() {
        assert_eq!(decode_integer_sample(&[0], 8), -1.0);
        assert_eq!(decode_integer_sample(&[128], 8), 0.0);
        assert_eq!(decode_integer_sample(&[0x00, 0x80], 16), -1.0);
        assert_eq!(decode_integer_sample(&[0x00, 0x00, 0x80], 24), -1.0);
        assert_eq!(decode_integer_sample(&[0x00, 0x00, 0x00, 0x80], 32), -1.0);
        assert!((decode_integer_sample(&[0xff, 0xff, 0x7f], 24) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn downsampling_matches_miniaudio_linear_lpf4_reference() {
        let mut input = vec![0.0f32; 18];
        input[0] = 1.0;

        let output = resample_linear_low_pass(&input, 48_000, 16_000, 6);

        let expected = [
            0.0,
            0.310_407_88,
            0.039_493_32,
            -0.019_835_327,
            0.008_888_276,
            -0.003_842_225,
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn decodes_downmixes_resamples_and_chunks_pcm_wav() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("stereo-8k.wav");
        let wave = pcm16_wave(8_000, 2, &[16_384, 0, 0, -16_384, 32_767, 32_767]);
        std::fs::write(&path, wave).unwrap();

        let prepared =
            prepare_audios_for_multimodal(&[path.to_string_lossy().into_owned()], test_config())
                .unwrap();
        let audio = &prepared[0];
        assert_eq!(audio.source_sample_rate, 8_000);
        assert_eq!(audio.source_channels, 2);
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.total_samples, 6);
        assert_eq!(audio.chunks.len(), 2);
        let expected = [
            0.0,
            0.011_747_607,
            0.070_485_644,
            0.158_756_82,
            0.130_208_45,
            -0.030_378_915,
        ];
        for (actual, expected) in audio.data_mono_f32.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert_eq!(load_audio_chunk_samples(audio, 0).unwrap().len(), 3);
        assert_eq!(load_audio_chunk_samples(audio, 1).unwrap().len(), 3);
        assert_eq!(audio.feature_windows.len(), 1);
    }

    #[test]
    fn probe_predicts_resampled_shape_and_feature_windows_before_decode() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("stereo-8k.wav");
        let wave = pcm16_wave(8_000, 2, &[16_384, 0, 0, -16_384, 32_767, 32_767]);
        std::fs::write(&path, wave).unwrap();
        let paths = [path.to_string_lossy().into_owned()];

        let plans = probe_audios_for_multimodal(&paths, test_config()).unwrap();
        let prepared = prepare_audios_for_multimodal(&paths, test_config()).unwrap();

        assert_eq!(plans[0].source_sample_rate, 8_000);
        assert_eq!(plans[0].source_channels, 2);
        assert_eq!(plans[0].target_sample_rate, 16_000);
        assert_eq!(plans[0].target_sample_count, 6);
        assert_eq!(plans[0].feature_windows.len(), 1);
        assert_eq!(plans[0].feature_windows[0].valid_frames, 1);
        assert_eq!(plans[0].feature_windows[0].padded_frames, 100);
        assert_eq!(
            plans[0].feature_windows[0].padded_frames,
            prepared[0].feature_windows[0].padded_frames
        );
    }

    #[test]
    fn rejects_unsupported_wav_encoding_with_a_conversion_hint() {
        let temp = TempDir::new().unwrap();
        let mut wave = pcm16_wave(16_000, 1, &[0, 1]);
        // A-law, which this decoder does not support.
        wave[20..22].copy_from_slice(&6u16.to_le_bytes());
        let path = write_temp_wave(&temp, "alaw.wav", &wave);
        let error = prepare_audios_for_multimodal(&[path], test_config()).unwrap_err();
        assert!(error.contains("unsupported WAV encoding 6"), "{error}");
        assert!(error.contains("ffmpeg"), "{error}");
    }

    #[test]
    fn decodes_extensible_integer_pcm_wav() {
        let temp = TempDir::new().unwrap();
        let payload = [0x00u8, 0x40, 0x00, 0xc0];
        let wave = extensible_wave(
            16_000,
            1,
            16,
            WAVE_FORMAT_PCM,
            SUBFORMAT_GUID_TAIL,
            &payload,
        );
        let path = write_temp_wave(&temp, "extensible.wav", &wave);

        let prepared = prepare_audios_for_multimodal(&[path], test_config()).unwrap();

        assert_eq!(prepared[0].total_samples, 2);
        assert_eq!(prepared[0].sample_rate, 16_000);
        assert_eq!(prepared[0].data_mono_f32, vec![0.5, -0.5]);
    }

    #[test]
    fn rejects_extensible_wav_with_a_foreign_sub_format_guid() {
        let temp = TempDir::new().unwrap();
        let mut guid_tail = SUBFORMAT_GUID_TAIL;
        guid_tail[13] = 0x00;
        let wave = extensible_wave(
            16_000,
            1,
            16,
            WAVE_FORMAT_PCM,
            guid_tail,
            &[0x00, 0x40, 0x00, 0xc0],
        );
        let path = write_temp_wave(&temp, "foreign-guid.wav", &wave);

        let error = prepare_audios_for_multimodal(&[path], test_config()).unwrap_err();

        assert!(error.contains("sub-format GUID"), "{error}");
    }

    #[test]
    fn decodes_ieee_float_wav_at_its_stored_amplitude() {
        let temp = TempDir::new().unwrap();
        let wave = float32_wave(16_000, 2, &[0.25, 0.75, -1.0, -0.5]);
        let path = write_temp_wave(&temp, "float32.wav", &wave);

        let prepared = prepare_audios_for_multimodal(&[path], test_config()).unwrap();

        assert_eq!(prepared[0].source_channels, 2);
        assert_eq!(prepared[0].data_mono_f32, vec![0.5, -0.75]);
    }

    #[test]
    fn decodes_streamed_wav_with_unknown_chunk_sizes() {
        let temp = TempDir::new().unwrap();
        let mut wave = pcm16_wave(16_000, 1, &[16_384, -16_384, 0]);
        // `ffmpeg -f wav -` cannot seek back, so it leaves both sizes unknown.
        wave[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        wave[DATA_SIZE_OFFSET..DATA_SIZE_OFFSET + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        // A partial trailing frame, as produced by an interrupted writer.
        wave.push(0x11);
        let path = write_temp_wave(&temp, "streamed.wav", &wave);

        let prepared = prepare_audios_for_multimodal(&[path], test_config()).unwrap();

        assert_eq!(prepared[0].total_samples, 3);
        assert_eq!(prepared[0].data_mono_f32[0], 0.5);
    }

    #[test]
    fn rejects_misaligned_data_when_the_declared_size_fits() {
        let temp = TempDir::new().unwrap();
        let mut wave = pcm16_wave(16_000, 2, &[1, 2, 3, 4]);
        wave[DATA_SIZE_OFFSET..DATA_SIZE_OFFSET + 4].copy_from_slice(&6u32.to_le_bytes());
        let path = write_temp_wave(&temp, "misaligned.wav", &wave);

        let error = prepare_audios_for_multimodal(&[path], test_config()).unwrap_err();

        assert!(error.contains("complete sample frames"), "{error}");
    }

    #[test]
    fn rejects_output_beyond_sample_limit_before_resampling() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("long.wav");
        std::fs::write(&path, pcm16_wave(8_000, 1, &[0; 8])).unwrap();
        let mut config = test_config();
        config.decode.max_output_samples = 15;
        let error = prepare_audios_for_multimodal(&[path.to_string_lossy().into_owned()], config)
            .unwrap_err();
        assert!(error.contains("exceeds limit 15"));
    }

    #[test]
    fn rejects_mismatched_decode_and_feature_rates() {
        let mut config = test_config();
        config.features.sample_rate = 8_000;
        let error = prepare_audios_for_multimodal(&[], config).unwrap_err();
        assert!(error.contains("does not match decode target"));
    }
}
