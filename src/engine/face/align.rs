//! Landmark-based similarity alignment to the encoder's canonical crop.

use image::RgbImage;

use crate::engine::face::FACE_LANDMARK_COUNT;

/// Forward 2x3 affine matrix `[a, b, tx; c, d, ty]` mapping source pixels to crop pixels.
pub(crate) type FaceAffine = [[f32; 3]; 2];

/// Least-squares similarity transform (rotation, uniform scale, translation, no reflection)
/// taking `source` landmarks onto `reference`.
pub(crate) fn similarity_transform(
    source: &[[f32; 2]; FACE_LANDMARK_COUNT],
    reference: &[[f32; 2]; FACE_LANDMARK_COUNT],
) -> Result<FaceAffine, String> {
    let count = FACE_LANDMARK_COUNT as f64;
    let mean = |points: &[[f32; 2]; FACE_LANDMARK_COUNT], axis: usize| {
        points.iter().map(|p| f64::from(p[axis])).sum::<f64>() / count
    };
    let (source_x, source_y) = (mean(source, 0), mean(source, 1));
    let (reference_x, reference_y) = (mean(reference, 0), mean(reference, 1));
    let (mut dot, mut cross, mut spread) = (0.0f64, 0.0f64, 0.0f64);
    for (s, r) in source.iter().zip(reference) {
        let (sx, sy) = (f64::from(s[0]) - source_x, f64::from(s[1]) - source_y);
        let (rx, ry) = (f64::from(r[0]) - reference_x, f64::from(r[1]) - reference_y);
        dot += sx * rx + sy * ry;
        cross += sx * ry - sy * rx;
        spread += sx * sx + sy * sy;
    }
    if !spread.is_finite() || spread <= 1e-9 {
        return Err("face landmarks are degenerate".to_string());
    }
    let (a, b) = (dot / spread, cross / spread);
    if !(a.is_finite() && b.is_finite()) || a * a + b * b <= 1e-12 {
        return Err("face landmarks do not define an alignment".to_string());
    }
    Ok([
        [
            a as f32,
            -b as f32,
            (reference_x - (a * source_x - b * source_y)) as f32,
        ],
        [
            b as f32,
            a as f32,
            (reference_y - (b * source_x + a * source_y)) as f32,
        ],
    ])
}

/// Samples a `size` x `size` crop through the inverse of `forward` with bilinear
/// interpolation. Pixels outside the source are black.
pub(crate) fn warp_affine(
    image: &RgbImage,
    forward: &FaceAffine,
    size: u32,
) -> Result<RgbImage, String> {
    let [[a, b, tx], [c, d, ty]] = forward.map(|row| row.map(f64::from));
    let det = a * d - b * c;
    if !det.is_finite() || det.abs() <= 1e-12 {
        return Err("face alignment transform is not invertible".to_string());
    }
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    let (width, height) = (i64::from(image.width()), i64::from(image.height()));
    let raw = image.as_raw();
    let sample = |x: i64, y: i64, channel: usize| -> f64 {
        if x < 0 || y < 0 || x >= width || y >= height {
            0.0
        } else {
            f64::from(raw[((y * width + x) * 3) as usize + channel])
        }
    };
    let mut out = RgbImage::new(size, size);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let (dx, dy) = (f64::from(x) - tx, f64::from(y) - ty);
        let (source_x, source_y) = (ia * dx + ib * dy, ic * dx + id * dy);
        let (x0, y0) = (source_x.floor(), source_y.floor());
        let (fx, fy) = (source_x - x0, source_y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        for (channel, value) in pixel.0.iter_mut().enumerate() {
            let top = sample(x0, y0, channel) * (1.0 - fx) + sample(x0 + 1, y0, channel) * fx;
            let bottom =
                sample(x0, y0 + 1, channel) * (1.0 - fx) + sample(x0 + 1, y0 + 1, channel) * fx;
            *value = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::{similarity_transform, warp_affine};

    const REFERENCE: [[f32; 2]; 5] = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ];

    #[test]
    fn recovers_a_known_rotation_scale_and_shift() {
        // source = inverse of (scale 0.5, rotate 30 degrees, shift) applied to the reference.
        let (scale, angle) = (0.5f32, 30.0f32.to_radians());
        let (a, b) = (scale * angle.cos(), scale * angle.sin());
        let (tx, ty) = (12.0f32, -7.0f32);
        let det = a * a + b * b;
        let source = REFERENCE.map(|[x, y]| {
            let (dx, dy) = (x - tx, y - ty);
            [(a * dx + b * dy) / det, (-b * dx + a * dy) / det]
        });
        let m = similarity_transform(&source, &REFERENCE).unwrap();
        for (got, want) in [
            (m[0][0], a),
            (m[0][1], -b),
            (m[0][2], tx),
            (m[1][0], b),
            (m[1][1], a),
            (m[1][2], ty),
        ] {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn degenerate_landmarks_are_rejected() {
        assert!(similarity_transform(&[[5.0, 5.0]; 5], &REFERENCE).is_err());
    }

    #[test]
    fn translation_warp_copies_pixels_and_blacks_out_of_bounds() {
        let mut image = RgbImage::new(4, 4);
        image.put_pixel(2, 1, Rgb([200, 100, 50]));
        // Crop pixel = source pixel - (1, 1).
        let crop = warp_affine(&image, &[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], 4).unwrap();
        assert_eq!(crop.get_pixel(1, 0), &Rgb([200, 100, 50]));
        assert_eq!(crop.get_pixel(3, 3), &Rgb([0, 0, 0]));
    }

    #[test]
    fn half_pixel_shift_interpolates() {
        let mut image = RgbImage::new(2, 1);
        image.put_pixel(0, 0, Rgb([0, 0, 0]));
        image.put_pixel(1, 0, Rgb([100, 100, 100]));
        let crop = warp_affine(&image, &[[1.0, 0.0, -0.5], [0.0, 1.0, 0.0]], 1).unwrap();
        assert_eq!(crop.get_pixel(0, 0), &Rgb([50, 50, 50]));
    }
}
