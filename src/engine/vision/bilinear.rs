//! RGB8 bilinear stretching with Pillow's separable fixed-point rounding.
//!
//! Contract: Pillow 12.1.1, commit 5158d98c807e719c5938aa3886913ef0ea6814e9,
//! src/libImaging/Resample.c. Pixel-center sampling, widened downsampling support,
//! normalized 22-bit coefficients, and an RGB8 rounding step after each axis.
//! The independent Gemma processor fixtures exercise the complete operation.

use image::RgbImage;

const PRECISION: u32 = 22;
const UNITY: f64 = (1u32 << PRECISION) as f64;

struct Weights {
    start: usize,
    coefficients: Vec<u32>,
}

/// Conservative peak payload of the allocations in `resize_rgb8`, excluding
/// the borrowed source and allocator overhead. Used before a group is decoded.
pub(super) fn storage_bound(
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Option<usize> {
    fn weights(source: usize, target: usize) -> Option<usize> {
        if source == target {
            return Some(0);
        }
        let support = (f64::from(source as f32) / target as f64).max(1.0).ceil() as usize;
        let taps = support.checked_mul(2)?.checked_add(1)?.min(source);
        target
            .checked_mul(std::mem::size_of::<Weights>().checked_add(taps.checked_mul(4)?)?)?
            .checked_add(taps.checked_mul(8)?)
    }
    if width == 0 || height == 0 || target_width == 0 || target_height == 0 {
        return None;
    }
    let output = target_width.checked_mul(target_height)?.checked_mul(3)?;
    let horizontal = if width == target_width {
        0
    } else {
        target_width.checked_mul(height)?.checked_mul(3)?
    };
    output
        .checked_add(horizontal)?
        .checked_add(weights(width, target_width)?.max(weights(height, target_height)?))
}

fn reserve<T>(len: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| "unable to allocate bilinear resize storage".to_string())?;
    Ok(values)
}

fn axis_weights(source: usize, target: usize) -> Result<Vec<Weights>, String> {
    let mut result = reserve(target)?;
    // Pillow represents the source box endpoints as F32 before coefficient
    // generation in F64. Preserve that boundary even for long, thin images.
    let scale = f64::from(source as f32) / target as f64;
    let support = scale.max(1.0);
    let reciprocal = 1.0 / support;
    for output in 0..target {
        let center = (output as f64 + 0.5) * scale;
        let start = ((center - support + 0.5).max(0.0) as usize).min(source);
        let end = ((center + support + 0.5) as usize).min(source);
        let mut unscaled = reserve(end - start)?;
        let mut sum = 0.0;
        for input in start..end {
            let distance = ((input as f64 - center + 0.5) * reciprocal).abs();
            let weight = (1.0 - distance).max(0.0);
            unscaled.push(weight);
            sum += weight;
        }
        if sum <= 0.0 {
            return Err("bilinear resize has an empty sampling interval".to_string());
        }
        let mut coefficients = reserve(unscaled.len())?;
        coefficients.extend(
            unscaled
                .iter()
                .map(|weight| (weight / sum * UNITY + 0.5) as u32),
        );
        result.push(Weights {
            start,
            coefficients,
        });
    }
    Ok(result)
}

fn rgb_storage(width: usize, height: usize) -> Result<Vec<u8>, String> {
    let count = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "bilinear resize dimensions overflow".to_string())?;
    let mut values = reserve(count)?;
    values.resize(count, 0);
    Ok(values)
}

fn sample(input: &[u8], offset: usize, stride: usize, weights: &[u32]) -> u8 {
    // Positive bilinear weights allow a wider accumulator without changing
    // rounding. This also avoids signed overflow for extreme reduction ratios.
    let mut sum = 1u64 << (PRECISION - 1);
    for (tap, &weight) in weights.iter().enumerate() {
        sum += u64::from(input[offset + tap * stride]) * u64::from(weight);
    }
    (sum >> PRECISION).min(255) as u8
}

pub(super) fn resize_rgb8(
    source: &RgbImage,
    target_width: usize,
    target_height: usize,
) -> Result<RgbImage, String> {
    let width = source.width() as usize;
    let height = source.height() as usize;
    if [width, height, target_width, target_height]
        .iter()
        .any(|&dimension| dimension == 0 || dimension > i32::MAX as usize)
    {
        return Err("bilinear resize dimensions must be positive signed 32-bit values".to_string());
    }
    // Check the final buffer even when only the other axis needs resampling.
    let mut output = rgb_storage(target_width, target_height)?;
    let horizontal;
    let rows = if width == target_width {
        source.as_raw().as_slice()
    } else {
        let weights = axis_weights(width, target_width)?;
        let mut values = rgb_storage(target_width, height)?;
        for y in 0..height {
            for (x, weights) in weights.iter().enumerate() {
                for channel in 0..3 {
                    values[(y * target_width + x) * 3 + channel] = sample(
                        source.as_raw(),
                        (y * width + weights.start) * 3 + channel,
                        3,
                        &weights.coefficients,
                    );
                }
            }
        }
        horizontal = values;
        horizontal.as_slice()
    };
    if height == target_height {
        output.copy_from_slice(rows);
    } else {
        let weights = axis_weights(height, target_height)?;
        for (y, weights) in weights.iter().enumerate() {
            for x in 0..target_width {
                for channel in 0..3 {
                    output[(y * target_width + x) * 3 + channel] = sample(
                        rows,
                        (weights.start * target_width + x) * 3 + channel,
                        target_width * 3,
                        &weights.coefficients,
                    );
                }
            }
        }
    }
    RgbImage::from_raw(target_width as u32, target_height as u32, output)
        .ok_or_else(|| "bilinear resize output buffer size mismatch".to_string())
}

#[cfg(test)]
mod tests {
    use super::resize_rgb8;
    use image::RgbImage;

    #[test]
    fn rejects_invalid_resize_dimensions_before_allocation() {
        let image = RgbImage::new(1, 1);
        for dimensions in [[0, 1], [1, 0], [usize::MAX, 1], [1, usize::MAX]] {
            assert!(resize_rgb8(&image, dimensions[0], dimensions[1]).is_err());
        }
        assert!(resize_rgb8(&RgbImage::new(0, 1), 1, 1).is_err());
    }
}
