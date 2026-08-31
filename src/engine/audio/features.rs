use std::f64::consts::PI;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSliceMut,
};

use super::{PreparedAudioFeatureWindow, PreparedAudioFeatureWindowPlan, WhisperLogMelConfig};

#[derive(Clone, Copy, Debug, Default)]
struct Complex32 {
    re: f32,
    im: f32,
}

fn validate_config(config: WhisperLogMelConfig) -> Result<(), String> {
    if config.sample_rate == 0 {
        return Err("audio feature sample_rate must be > 0".to_string());
    }
    if config.fft_length < 2 || !config.fft_length.is_multiple_of(2) || config.fft_length > 4096 {
        return Err("audio fft_length must be even and within 2..=4096".to_string());
    }
    if config.window_length == 0 || config.window_length > config.fft_length {
        return Err("audio window_length must be within 1..=fft_length".to_string());
    }
    if config.hop_length == 0 {
        return Err("audio hop_length must be > 0".to_string());
    }
    if config.mel_bins == 0 {
        return Err("audio mel_bins must be > 0".to_string());
    }
    if !config.mel_floor.is_finite() || config.mel_floor <= 0.0 {
        return Err("audio mel_floor must be finite and > 0".to_string());
    }
    if config.max_window_frames == 0 {
        return Err("audio max_window_frames must be > 0".to_string());
    }
    if config.frame_chunk_size == 0 {
        return Err("audio frame_chunk_size must be > 0".to_string());
    }
    Ok(())
}

fn periodic_hann(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            // llama.cpp computes the angle in f64, calls cosf, then performs the
            // surrounding literal arithmetic in f64 before storing an f32.
            let angle = (2.0 * PI * index as f64 / length as f64) as f32;
            (0.5f64 * (1.0 - f64::from(angle.cos()))) as f32
        })
        .collect()
}

fn slaney_hz_to_mel(frequency_hz: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const LINEAR_SLOPE: f64 = 3.0 / 200.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ * LINEAR_SLOPE;
    let log_step = 6.4f64.ln() / 27.0;
    if frequency_hz < MIN_LOG_HZ {
        frequency_hz * LINEAR_SLOPE
    } else {
        MIN_LOG_MEL + (frequency_hz / MIN_LOG_HZ).ln() / log_step
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const LINEAR_SLOPE: f64 = 3.0 / 200.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ * LINEAR_SLOPE;
    let log_step = 6.4f64.ln() / 27.0;
    if mel < MIN_LOG_MEL {
        mel / LINEAR_SLOPE
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * log_step).exp()
    }
}

fn build_slaney_mel_filters(config: WhisperLogMelConfig) -> Result<Vec<f32>, String> {
    let fft_bins = config.fft_length / 2 + 1;
    let element_count = config
        .mel_bins
        .checked_mul(fft_bins)
        .ok_or_else(|| "audio mel filter element count overflow".to_string())?;
    let mel_min = slaney_hz_to_mel(0.0);
    let mel_max = slaney_hz_to_mel(f64::from(config.sample_rate) / 2.0);
    let mut hz_points = Vec::with_capacity(config.mel_bins + 2);
    for index in 0..config.mel_bins + 2 {
        let mel = mel_min + (mel_max - mel_min) * (index as f64 / (config.mel_bins + 1) as f64);
        hz_points.push(slaney_mel_to_hz(mel));
    }

    let mut filters = vec![0.0; element_count];
    let bin_hz_step = f64::from(config.sample_rate) / config.fft_length as f64;
    for mel_index in 0..config.mel_bins {
        let left = hz_points[mel_index];
        let center = hz_points[mel_index + 1];
        let right = hz_points[mel_index + 2];
        let left_width = (center - left).max(1e-30);
        let right_width = (right - center).max(1e-30);
        let area_normalization = 2.0 / (right - left).max(1e-30);
        for bin in 0..fft_bins {
            let frequency = bin as f64 * bin_hz_step;
            let weight = if frequency >= left && frequency <= center {
                (frequency - left) / left_width
            } else if frequency > center && frequency <= right {
                (right - frequency) / right_width
            } else {
                0.0
            };
            filters[mel_index * fft_bins + bin] = (weight * area_normalization) as f32;
        }
    }
    Ok(filters)
}

/// The bins one Mel filter can actually contribute to, split the same way the
/// projection loop consumes them: whole groups of four followed by the ragged
/// tail. Slaney filters are triangular, so all but roughly 1.5% of the weights
/// are exactly zero and the surviving weights sit in one contiguous run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MelFilterSpan {
    group_start: usize,
    group_end: usize,
    tail_start: usize,
    tail_end: usize,
}

/// Skipping a group is exact rather than approximate: every weight in it is
/// `0.0`, so the group's f32 sum is `±0.0` and adding it to the f64 accumulator
/// is a no-op. The surviving groups keep their original order and boundaries,
/// which is what preserves the upstream summation semantics.
fn build_mel_filter_spans(filters: &[f32], mel_bins: usize, fft_bins: usize) -> Vec<MelFilterSpan> {
    let group_count = fft_bins / 4;
    let tail_base = group_count * 4;
    (0..mel_bins)
        .map(|mel| {
            let row = &filters[mel * fft_bins..mel * fft_bins + fft_bins];
            let first = row.iter().position(|weight| *weight != 0.0);
            let Some(first) = first else {
                return MelFilterSpan::default();
            };
            let last = row
                .iter()
                .rposition(|weight| *weight != 0.0)
                .expect("a first nonzero weight implies a last one");
            let group_start = (first / 4).min(group_count);
            let group_end = (last / 4 + 1).clamp(group_start, group_count);
            let tail_start = first.max(tail_base).min(fft_bins);
            let tail_end = (last + 1).clamp(tail_start, fft_bins);
            MelFilterSpan {
                group_start,
                group_end,
                tail_start,
                tail_end,
            }
        })
        .collect()
}

/// Precomputed constants for one FFT length.
///
/// The recursive form this replaces re-derived its twiddles from a modulo on
/// every inner iteration and allocated three `Vec`s per recursion node, which
/// for a 400-point transform meant 10_000 integer divisions and ~93 allocations
/// per frame. Both are hoisted here and shared across every frame.
struct FftPlan {
    length: usize,
    /// The odd factor left after removing every power of two, transformed
    /// directly by a DFT the way the recursion's base case did.
    leaf_length: usize,
    leaf_count: usize,
    levels: usize,
    sin_values: Vec<f32>,
    cos_values: Vec<f32>,
    /// `leaf_length * leaf_length` twiddles, indexed `frequency * leaf_length +
    /// sample`, holding exactly the values the modulo used to select.
    leaf_cos: Vec<f32>,
    leaf_sin: Vec<f32>,
    /// Decimation-in-time places leaf block `j` on the input residue class
    /// `bit_reverse(j)`, mirroring the recursion's even/odd split order.
    leaf_offsets: Vec<usize>,
}

impl FftPlan {
    fn new(length: usize) -> Self {
        let mut sin_values = Vec::with_capacity(length);
        let mut cos_values = Vec::with_capacity(length);
        for index in 0..length {
            let angle = (2.0 * PI * index as f64 / length as f64) as f32;
            sin_values.push(angle.sin());
            cos_values.push(angle.cos());
        }

        let mut leaf_length = length;
        let mut levels = 0usize;
        while leaf_length.is_multiple_of(2) && leaf_length > 1 {
            leaf_length /= 2;
            levels += 1;
        }
        let leaf_count = 1usize << levels;

        let leaf_step = length / leaf_length;
        let mut leaf_cos = vec![0.0f32; leaf_length * leaf_length];
        let mut leaf_sin = vec![0.0f32; leaf_length * leaf_length];
        for frequency in 0..leaf_length {
            for sample in 0..leaf_length {
                let table_index = ((frequency * sample) % leaf_length) * leaf_step;
                leaf_cos[frequency * leaf_length + sample] = cos_values[table_index];
                leaf_sin[frequency * leaf_length + sample] = sin_values[table_index];
            }
        }

        let leaf_offsets = (0..leaf_count)
            .map(|block| block.reverse_bits() >> (usize::BITS as usize - levels.max(1)))
            .map(|reversed| if levels == 0 { 0 } else { reversed })
            .collect();

        Self {
            length,
            leaf_length,
            leaf_count,
            levels,
            sin_values,
            cos_values,
            leaf_cos,
            leaf_sin,
            leaf_offsets,
        }
    }
}

/// Per-thread buffers reused across every frame, so the transform itself never
/// touches the allocator.
struct FftScratch {
    windowed: Vec<f32>,
    power: Vec<f32>,
    primary: Vec<Complex32>,
    secondary: Vec<Complex32>,
}

impl FftScratch {
    fn new(plan: &FftPlan, fft_bins: usize) -> Self {
        Self {
            windowed: vec![0.0f32; plan.length],
            power: vec![0.0f32; fft_bins],
            primary: vec![Complex32::default(); plan.length],
            secondary: vec![Complex32::default(); plan.length],
        }
    }

    /// Transforms `self.windowed` in place, leaving the spectrum in
    /// `self.primary`.
    fn transform(&mut self, plan: &FftPlan) {
        let leaf_length = plan.leaf_length;
        for (block, offset) in plan.leaf_offsets.iter().copied().enumerate() {
            let output = &mut self.primary[block * leaf_length..(block + 1) * leaf_length];
            if leaf_length == 1 {
                // The recursion's length-1 base case returned a real value, so
                // keep the imaginary part at +0.0 rather than the -0.0 a
                // multiply by sin(0) would produce.
                output[0] = Complex32 {
                    re: self.windowed[offset],
                    im: 0.0,
                };
                continue;
            }
            for (frequency, value) in output.iter_mut().enumerate() {
                let table_row = frequency * leaf_length;
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for sample_index in 0..leaf_length {
                    let sample = self.windowed[offset + sample_index * plan.leaf_count];
                    re += sample * plan.leaf_cos[table_row + sample_index];
                    im -= sample * plan.leaf_sin[table_row + sample_index];
                }
                *value = Complex32 { re, im };
            }
        }

        let mut half = leaf_length;
        for _ in 0..plan.levels {
            let size = half * 2;
            let table_step = plan.length / size;
            for (source, destination) in self
                .primary
                .chunks_exact(size)
                .zip(self.secondary.chunks_exact_mut(size))
            {
                let (even, odd) = source.split_at(half);
                for frequency in 0..half {
                    let table_index = frequency * table_step;
                    let twiddle_re = plan.cos_values[table_index];
                    let twiddle_im = -plan.sin_values[table_index];
                    // Preserve the upstream butterfly expression association.
                    // Re-grouping these products measurably increases
                    // cross-toolchain feature drift.
                    destination[frequency] = Complex32 {
                        re: even[frequency].re + twiddle_re * odd[frequency].re
                            - twiddle_im * odd[frequency].im,
                        im: even[frequency].im
                            + twiddle_re * odd[frequency].im
                            + twiddle_im * odd[frequency].re,
                    };
                    destination[frequency + half] = Complex32 {
                        re: even[frequency].re - twiddle_re * odd[frequency].re
                            + twiddle_im * odd[frequency].im,
                        im: even[frequency].im
                            - twiddle_re * odd[frequency].im
                            - twiddle_im * odd[frequency].re,
                    };
                }
            }
            std::mem::swap(&mut self.primary, &mut self.secondary);
            half = size;
        }
    }
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = vec![0.0; samples.len() + 2 * pad];
    for (index, destination) in padded[..pad].iter_mut().enumerate() {
        let source = pad - index;
        if source < samples.len() {
            *destination = samples[source];
        }
    }
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    for index in 0..pad {
        if let Some(source) = samples.len().checked_sub(2 + index) {
            padded[samples.len() + pad + index] = samples[source];
        }
    }
    padded
}

/// Transposes the frame-major log-Mel buffer into the mel-major windows the
/// encoder consumes. Fusing the transpose here avoids materializing a separate
/// mel-major copy of the whole clip.
fn split_feature_windows(
    frame_major: &[f32],
    mel_bins: usize,
    frame_count: usize,
    max_window_frames: usize,
    frame_chunk_size: usize,
) -> Result<Vec<PreparedAudioFeatureWindow>, String> {
    let plans = plan_feature_windows(frame_count, max_window_frames, frame_chunk_size)?;
    let mut windows = Vec::with_capacity(plans.len());
    for plan in plans {
        let element_count = mel_bins
            .checked_mul(plan.padded_frames)
            .ok_or_else(|| "audio feature window size overflow".to_string())?;
        let source_base = plan
            .start_frame
            .checked_mul(mel_bins)
            .ok_or_else(|| "audio feature source offset overflow".to_string())?;
        let source_end = plan
            .valid_frames
            .checked_mul(mel_bins)
            .and_then(|span| source_base.checked_add(span))
            .ok_or_else(|| "audio feature source range overflow".to_string())?;
        if source_end > frame_major.len() {
            return Err("audio feature window exceeds the log-Mel buffer".to_string());
        }
        let mut data = vec![0.0; element_count];
        data.par_chunks_mut(plan.padded_frames)
            .enumerate()
            .for_each(|(mel, row)| {
                for (frame, destination) in row[..plan.valid_frames].iter_mut().enumerate() {
                    *destination = frame_major[source_base + frame * mel_bins + mel];
                }
            });
        windows.push(PreparedAudioFeatureWindow {
            start_frame: plan.start_frame,
            valid_frames: plan.valid_frames,
            padded_frames: plan.padded_frames,
            mel_bins,
            data_mel_major: data,
        });
    }
    Ok(windows)
}

fn plan_feature_windows(
    frame_count: usize,
    max_window_frames: usize,
    frame_chunk_size: usize,
) -> Result<Vec<PreparedAudioFeatureWindowPlan>, String> {
    let mut windows = Vec::new();
    for start_frame in (0..frame_count).step_by(max_window_frames) {
        let valid_frames = (frame_count - start_frame).min(max_window_frames);
        let padded_frames = valid_frames
            .div_ceil(frame_chunk_size)
            .checked_mul(frame_chunk_size)
            .ok_or_else(|| "audio padded frame count overflow".to_string())?;
        windows.push(PreparedAudioFeatureWindowPlan {
            start_frame,
            valid_frames,
            padded_frames,
        });
    }
    Ok(windows)
}

pub(super) fn plan_whisper_log_mel_windows(
    sample_count: usize,
    config: WhisperLogMelConfig,
) -> Result<Vec<PreparedAudioFeatureWindowPlan>, String> {
    validate_config(config)?;
    if sample_count == 0 {
        return Err("audio sample count is zero".to_string());
    }
    let frame_count = sample_count
        .checked_div(config.hop_length)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| "audio feature frame count overflow".to_string())?;
    plan_feature_windows(
        frame_count,
        config.max_window_frames,
        config.frame_chunk_size,
    )
}

pub(crate) fn extract_whisper_log_mel_windows(
    samples: &[f32],
    config: WhisperLogMelConfig,
) -> Result<Vec<PreparedAudioFeatureWindow>, String> {
    validate_config(config)?;
    if samples.is_empty() {
        return Err("audio sample buffer is empty".to_string());
    }
    if samples.par_iter().any(|sample| !sample.is_finite()) {
        return Err("audio sample buffer contains non-finite values".to_string());
    }

    let pad = config.fft_length / 2;
    let padded = reflect_pad(samples, pad);
    let frame_count = (padded.len() - config.fft_length) / config.hop_length + 1;
    let effective_frame_count = frame_count.min(samples.len() / config.hop_length + 1);
    let element_count = config
        .mel_bins
        .checked_mul(frame_count)
        .ok_or_else(|| "audio log-Mel element count overflow".to_string())?;

    let mut hann = periodic_hann(config.window_length);
    if config.window_length < config.fft_length {
        let mut centered = vec![0.0; config.fft_length];
        let offset = (config.fft_length - config.window_length) / 2;
        centered[offset..offset + config.window_length].copy_from_slice(&hann);
        hann = centered;
    }
    let filters = build_slaney_mel_filters(config)?;
    let fft_bins = config.fft_length / 2 + 1;
    let spans = build_mel_filter_spans(&filters, config.mel_bins, fft_bins);
    let plan = FftPlan::new(config.fft_length);

    // Frame-major (`[frame][mel]`) so frames can be filled independently; the
    // transpose to mel-major happens once, fused into the window split.
    let mut frame_major = vec![0.0f32; element_count];
    frame_major
        .par_chunks_mut(config.mel_bins)
        .enumerate()
        .for_each_init(
            || FftScratch::new(&plan, fft_bins),
            |scratch, (frame, mel_row)| {
                let offset = frame * config.hop_length;
                for index in 0..config.fft_length {
                    scratch.windowed[index] = padded[offset + index] * hann[index];
                }
                scratch.transform(&plan);
                for (bin, power) in scratch.power.iter_mut().enumerate() {
                    let value = scratch.primary[bin];
                    *power = value.re * value.re + value.im * value.im;
                }
                for (mel, output) in mel_row.iter_mut().enumerate() {
                    let filter_start = mel * fft_bins;
                    let span = spans[mel];
                    let mut sum = 0.0f64;
                    // The pinned source sums four f32 products before promoting
                    // the group to f64. This is observable around near-zero mel
                    // values, so the grouping and its boundaries are preserved.
                    for group in span.group_start..span.group_end {
                        let bin = group * 4;
                        let group_sum = scratch.power[bin] * filters[filter_start + bin]
                            + scratch.power[bin + 1] * filters[filter_start + bin + 1]
                            + scratch.power[bin + 2] * filters[filter_start + bin + 2]
                            + scratch.power[bin + 3] * filters[filter_start + bin + 3];
                        sum += f64::from(group_sum);
                    }
                    for bin in span.tail_start..span.tail_end {
                        sum += f64::from(scratch.power[bin] * filters[filter_start + bin]);
                    }
                    *output = sum.max(f64::from(config.mel_floor)).log10() as f32;
                }
            },
        );

    let max_log_mel = frame_major
        .par_iter()
        .map(|value| f64::from(*value))
        .reduce(|| f64::NEG_INFINITY, f64::max);
    let dynamic_floor = max_log_mel - 8.0;
    frame_major.par_iter_mut().for_each(|value| {
        *value = ((f64::from(*value).max(dynamic_floor) + 4.0) / 4.0) as f32;
    });

    split_feature_windows(
        &frame_major,
        config.mel_bins,
        effective_frame_count,
        config.max_window_frames,
        config.frame_chunk_size,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        extract_whisper_log_mel_windows, plan_whisper_log_mel_windows, reflect_pad,
        split_feature_windows,
    };
    use crate::engine::audio::WhisperLogMelConfig;

    fn config() -> WhisperLogMelConfig {
        WhisperLogMelConfig {
            sample_rate: 16_000,
            fft_length: 400,
            window_length: 400,
            hop_length: 160,
            mel_bins: 128,
            mel_floor: 5.960_464_5e-8,
            max_window_frames: 800,
            frame_chunk_size: 100,
        }
    }

    #[test]
    fn reflection_padding_matches_qwen3_asr_edge_order() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn short_audio_is_padded_to_one_hundred_frames() {
        let windows = extract_whisper_log_mel_windows(&vec![0.0; 160], config()).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_frame, 0);
        assert_eq!(windows[0].valid_frames, 2);
        assert_eq!(windows[0].padded_frames, 100);
        assert_eq!(windows[0].mel_bins, 128);
        assert_eq!(windows[0].data_mel_major.len(), 12_800);
        let expected_silence = (config().mel_floor.log10() + 4.0) / 4.0;
        assert!((windows[0].data_mel_major[0] - expected_silence).abs() < 1e-5);
        assert_eq!(windows[0].data_mel_major[2], 0.0);
    }

    #[test]
    fn window_split_uses_eight_hundred_frames_and_hundred_frame_padding() {
        let mel_bins = 2;
        let frame_count = 851;
        // Frame-major input carrying the mel-major value `mel * frame_count +
        // frame`, so the transposed expectations below read the same as the
        // logical (mel, frame) grid.
        let mut mel = vec![0.0f32; mel_bins * frame_count];
        for frame in 0..frame_count {
            for (mel_index, cell) in mel[frame * mel_bins..(frame + 1) * mel_bins]
                .iter_mut()
                .enumerate()
            {
                *cell = (mel_index * frame_count + frame) as f32;
            }
        }
        let windows = split_feature_windows(&mel, mel_bins, frame_count, 800, 100).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!((windows[0].start_frame, windows[0].valid_frames), (0, 800));
        assert_eq!(windows[0].padded_frames, 800);
        assert_eq!((windows[1].start_frame, windows[1].valid_frames), (800, 51));
        assert_eq!(windows[1].padded_frames, 100);
        assert_eq!(windows[1].data_mel_major[0], 800.0);
        assert_eq!(windows[1].data_mel_major[50], 850.0);
        assert_eq!(windows[1].data_mel_major[51], 0.0);
        assert_eq!(windows[1].data_mel_major[100], 1651.0);
    }

    #[test]
    fn feature_plan_matches_extracted_window_shapes() {
        let samples = vec![0.0; 128_000];
        let plan = plan_whisper_log_mel_windows(samples.len(), config()).unwrap();
        let windows = extract_whisper_log_mel_windows(&samples, config()).unwrap();

        assert_eq!(plan.len(), windows.len());
        for (planned, actual) in plan.iter().zip(&windows) {
            assert_eq!(planned.start_frame, actual.start_frame);
            assert_eq!(planned.valid_frames, actual.valid_frames);
            assert_eq!(planned.padded_frames, actual.padded_frames);
        }
    }

    #[test]
    fn feature_extraction_is_deterministic_for_non_silent_input() {
        let samples = (0..640)
            .map(|index| ((index as f32 * 0.03125).sin() * 0.5).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        let first = extract_whisper_log_mel_windows(&samples, config()).unwrap();
        let second = extract_whisper_log_mel_windows(&samples, config()).unwrap();
        assert_eq!(first[0].valid_frames, 5);
        assert_eq!(first[0].data_mel_major, second[0].data_mel_major);
        assert!(
            first[0]
                .data_mel_major
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn long_fixture_matches_pinned_llama_cpp_qwen3_asr_features() {
        // Generated by invoking mtmd_audio_preprocessor_qwen3a directly at llama.cpp
        // d83f72d463287ab9c50b4bc18ee332104a963889. Cross-toolchain arithmetic can
        // differ slightly when Clang contracts multiply-adds, so this guards the
        // measured portable maximum absolute error rather than bit equality.
        let samples = (0usize..128_000)
            .map(|index| {
                let raw = ((index * 73 + (index / 97) * 19) % 2001) as i32 - 1000;
                raw as f32 / 1000.0
            })
            .collect::<Vec<_>>();
        let windows = extract_whisper_log_mel_windows(&samples, config()).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(
            (
                windows[0].start_frame,
                windows[0].valid_frames,
                windows[0].padded_frames,
            ),
            (0, 800, 800)
        );
        assert_eq!(
            (
                windows[1].start_frame,
                windows[1].valid_frames,
                windows[1].padded_frames,
            ),
            (800, 1, 100)
        );

        let golden = [
            (0, 0, 0, 0x3f80_ed67),
            (0, 0, 1, 0x3f06_3fac),
            (0, 0, 799, 0x3e9b_d808),
            (0, 17, 0, 0x3f8e_d1f3),
            (0, 57, 724, 0x3948_8000),
            (0, 57, 785, 0xbedd_25b4),
            (0, 64, 399, 0x3f8d_c854),
            (0, 127, 799, 0x3f4f_394c),
            (1, 0, 0, 0x3f20_1800),
            (1, 0, 1, 0x0000_0000),
            (1, 127, 0, 0x3f5c_5380),
            (1, 127, 1, 0x0000_0000),
        ];
        for (window, mel, frame, expected_bits) in golden {
            let actual =
                windows[window].data_mel_major[mel * windows[window].padded_frames + frame];
            let expected = f32::from_bits(expected_bits);
            assert!(
                (actual - expected).abs() <= 3.0e-5,
                "feature mismatch at window={window}, mel={mel}, frame={frame}: actual={actual}, expected={expected}"
            );
        }
    }
}
