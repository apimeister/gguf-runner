use image::imageops::FilterType;
use image::{ImageReader, RgbImage};
use std::fs;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ImageNormalization {
    UnitRange,
    MeanStd { mean: [f32; 3], std: [f32; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageResizeMode {
    CenterCrop,
    FitWithin,
    Stretch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ImagePreprocessProfile {
    pub(crate) target_width: usize,
    pub(crate) target_height: usize,
    pub(crate) normalization: ImageNormalization,
    pub(crate) resize_mode: ImageResizeMode,
    pub(crate) align_to: usize,
}

impl ImagePreprocessProfile {
    pub(crate) fn new(
        target_width: usize,
        target_height: usize,
        normalization: ImageNormalization,
    ) -> Self {
        Self {
            target_width,
            target_height,
            normalization,
            resize_mode: ImageResizeMode::CenterCrop,
            align_to: 1,
        }
    }

    pub(crate) fn new_with_mode(
        target_width: usize,
        target_height: usize,
        normalization: ImageNormalization,
        resize_mode: ImageResizeMode,
        align_to: usize,
    ) -> Self {
        Self {
            target_width,
            target_height,
            normalization,
            resize_mode,
            align_to: align_to.max(1),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedImageTensor {
    pub(crate) path: String,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) data_chw: Vec<f32>,
}

impl PreparedImageTensor {
    #[inline]
    pub(crate) fn element_count(&self) -> usize {
        self.data_chw.len()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedVideoChunk {
    pub(crate) start_frame: usize,
    pub(crate) frame_paths: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct PreparedVideoTensor {
    pub(crate) path: String,
    pub(crate) sampled_fps: u32,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) frame_count: usize,
    pub(crate) chunks: Vec<PreparedVideoChunk>,
}

fn normalize_channel(
    value_unit_range: f32,
    channel: usize,
    normalization: ImageNormalization,
) -> Result<f32, String> {
    match normalization {
        ImageNormalization::UnitRange => Ok(value_unit_range),
        ImageNormalization::MeanStd { mean, std } => {
            let std_c = std[channel];
            if std_c <= 0.0 {
                return Err(format!(
                    "invalid image normalization std for channel {channel}: {std_c}"
                ));
            }
            Ok((value_unit_range - mean[channel]) / std_c)
        }
    }
}

fn rgb_u8_to_chw_f32(
    rgb: &[u8],
    width: usize,
    height: usize,
    normalization: ImageNormalization,
) -> Result<Vec<f32>, String> {
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| "image dimensions overflow".to_string())?;
    let expected_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| "image tensor element count overflow".to_string())?;

    if rgb.len() != expected_len {
        return Err(format!(
            "unexpected RGB buffer size: got {}, expected {}",
            rgb.len(),
            expected_len
        ));
    }

    let mut out = vec![0.0f32; expected_len];
    for y in 0..height {
        for x in 0..width {
            let pix = y * width + x;
            let src = pix * 3;
            for c in 0..3 {
                let unit = rgb[src + c] as f32 / 255.0;
                out[c * pixel_count + pix] = normalize_channel(unit, c, normalization)?;
            }
        }
    }

    Ok(out)
}

fn resize_and_center_crop(
    rgb: &RgbImage,
    target_width: usize,
    target_height: usize,
) -> Result<RgbImage, String> {
    if target_width == 0 || target_height == 0 {
        return Err("invalid image target size: width/height must be > 0".to_string());
    }

    let src_w = rgb.width() as usize;
    let src_h = rgb.height() as usize;
    if src_w == target_width && src_h == target_height {
        return Ok(rgb.clone());
    }

    let scale_x = target_width as f64 / src_w as f64;
    let scale_y = target_height as f64 / src_h as f64;
    let scale = scale_x.max(scale_y);

    let resized_w = (src_w as f64 * scale).ceil().max(target_width as f64) as usize;
    let resized_h = (src_h as f64 * scale).ceil().max(target_height as f64) as usize;

    let resized = image::imageops::resize(
        rgb,
        resized_w as u32,
        resized_h as u32,
        FilterType::CatmullRom,
    );

    let crop_x = (resized_w - target_width) / 2;
    let crop_y = (resized_h - target_height) / 2;
    let cropped = image::imageops::crop_imm(
        &resized,
        crop_x as u32,
        crop_y as u32,
        target_width as u32,
        target_height as u32,
    )
    .to_image();
    Ok(cropped)
}

fn round_down_to_multiple(value: usize, multiple: usize) -> usize {
    if multiple <= 1 {
        value
    } else {
        value / multiple * multiple
    }
}

fn align_within_limit(value: usize, align_to: usize, limit: usize) -> usize {
    if align_to <= 1 {
        return value.clamp(1, limit.max(1));
    }

    let mut aligned = round_down_to_multiple(value, align_to);
    if aligned == 0 {
        // A thin image may scale below one patch-merge unit. Use the
        // smallest representable dimension, not the full target edge.
        aligned = align_to.min(limit.max(1));
    }
    aligned.min(limit.max(1)).max(1)
}

fn resize_fit_within(
    rgb: &RgbImage,
    target_width: usize,
    target_height: usize,
    align_to: usize,
) -> Result<RgbImage, String> {
    if target_width == 0 || target_height == 0 {
        return Err("invalid image target size: width/height must be > 0".to_string());
    }
    let src_w = rgb.width() as usize;
    let src_h = rgb.height() as usize;
    if src_w == 0 || src_h == 0 {
        return Err("invalid source image dimensions".to_string());
    }

    let scale_x = target_width as f64 / src_w as f64;
    let scale_y = target_height as f64 / src_h as f64;
    let scale = scale_x.min(scale_y);

    let mut resized_w = (src_w as f64 * scale).round() as usize;
    let mut resized_h = (src_h as f64 * scale).round() as usize;
    resized_w = resized_w.clamp(1, target_width);
    resized_h = resized_h.clamp(1, target_height);

    let out_w = align_within_limit(resized_w, align_to, target_width);
    let out_h = align_within_limit(resized_h, align_to, target_height);

    if src_w == out_w && src_h == out_h {
        return Ok(rgb.clone());
    }
    Ok(image::imageops::resize(
        rgb,
        out_w as u32,
        out_h as u32,
        FilterType::CatmullRom,
    ))
}

fn resize_stretch_exact(
    rgb: &RgbImage,
    target_width: usize,
    target_height: usize,
) -> Result<RgbImage, String> {
    if target_width == 0 || target_height == 0 {
        return Err("invalid image target size: width/height must be > 0".to_string());
    }
    let src_w = rgb.width() as usize;
    let src_h = rgb.height() as usize;
    if src_w == target_width && src_h == target_height {
        return Ok(rgb.clone());
    }
    // Match llama.cpp clip preprocess behavior for Gemma3 projector path:
    // direct bilinear resize to the model's fixed image size.
    Ok(image::imageops::resize(
        rgb,
        target_width as u32,
        target_height as u32,
        FilterType::Triangle,
    ))
}

fn resize_for_profile(rgb: &RgbImage, profile: ImagePreprocessProfile) -> Result<RgbImage, String> {
    match profile.resize_mode {
        ImageResizeMode::CenterCrop => {
            resize_and_center_crop(rgb, profile.target_width, profile.target_height)
        }
        ImageResizeMode::FitWithin => resize_fit_within(
            rgb,
            profile.target_width,
            profile.target_height,
            profile.align_to,
        ),
        ImageResizeMode::Stretch => {
            resize_stretch_exact(rgb, profile.target_width, profile.target_height)
        }
    }
}

pub(crate) fn prepare_images_for_multimodal(
    image_paths: &[String],
    profile: ImagePreprocessProfile,
) -> Result<Vec<PreparedImageTensor>, String> {
    let mut prepared = Vec::with_capacity(image_paths.len());
    for path in image_paths {
        let reader =
            ImageReader::open(path).map_err(|e| format!("cannot open image '{path}': {e}"))?;
        let decoded = reader
            .decode()
            .map_err(|e| format!("cannot decode image '{path}': {e}"))?;
        let rgb = decoded.to_rgb8();
        let processed = resize_for_profile(&rgb, profile)?;
        let width = processed.width() as usize;
        let height = processed.height() as usize;
        let data_chw = rgb_u8_to_chw_f32(processed.as_raw(), width, height, profile.normalization)?;
        prepared.push(PreparedImageTensor {
            path: path.clone(),
            width,
            height,
            data_chw,
        });
    }
    Ok(prepared)
}

pub(crate) fn prepare_videos_for_multimodal(
    video_paths: &[String],
    _profile: ImagePreprocessProfile,
    sampled_fps: u32,
    max_frames: usize,
    _chunk_size_frames: usize,
) -> Result<Vec<PreparedVideoTensor>, String> {
    if sampled_fps == 0 {
        return Err("video sampled_fps must be > 0".to_string());
    }
    if max_frames == 0 {
        return Err("video max_frames must be > 0".to_string());
    }
    for path in video_paths {
        if fs::metadata(path).is_err() {
            return Err(format!("cannot read video file '{path}'"));
        }
    }
    Err(
        "native video preprocessing is unavailable in no-external-dependency mode (external decoder path removed)"
            .to_string(),
    )
}

pub(crate) fn load_video_chunk_tensors(
    video: &PreparedVideoTensor,
    chunk_idx: usize,
    profile: ImagePreprocessProfile,
) -> Result<Vec<PreparedImageTensor>, String> {
    let chunk = video
        .chunks
        .get(chunk_idx)
        .ok_or_else(|| format!("video chunk index out of range: {chunk_idx}"))?;
    prepare_images_for_multimodal(&chunk.frame_paths, profile)
}

#[cfg(test)]
mod tests {
    use super::{
        ImageNormalization, ImagePreprocessProfile, ImageResizeMode, prepare_images_for_multimodal,
        prepare_videos_for_multimodal, resize_and_center_crop, resize_for_profile,
        rgb_u8_to_chw_f32,
    };
    use image::{ImageFormat, RgbImage};
    use tempfile::TempDir;

    #[test]
    fn rgb_to_chw_unit_range_layout() {
        let rgb = vec![255u8, 0u8, 0u8, 0u8, 255u8, 0u8];
        let out = rgb_u8_to_chw_f32(&rgb, 2, 1, ImageNormalization::UnitRange).unwrap();
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 1.0);
        assert_eq!(out[4], 0.0);
        assert_eq!(out[5], 0.0);
    }

    #[test]
    fn rgb_to_chw_mean_std_normalization() {
        let rgb = vec![128u8, 64u8, 32u8];
        let out = rgb_u8_to_chw_f32(
            &rgb,
            1,
            1,
            ImageNormalization::MeanStd {
                mean: [0.5, 0.25, 0.125],
                std: [0.25, 0.25, 0.25],
            },
        )
        .unwrap();

        let expected0 = ((128.0 / 255.0) - 0.5) / 0.25;
        let expected1 = ((64.0 / 255.0) - 0.25) / 0.25;
        let expected2 = ((32.0 / 255.0) - 0.125) / 0.25;
        assert!((out[0] - expected0).abs() < 1e-6);
        assert!((out[1] - expected1).abs() < 1e-6);
        assert!((out[2] - expected2).abs() < 1e-6);
    }

    #[test]
    fn normalization_rejects_non_positive_std() {
        let rgb = vec![1u8, 2u8, 3u8];
        let err = rgb_u8_to_chw_f32(
            &rgb,
            1,
            1,
            ImageNormalization::MeanStd {
                mean: [0.0, 0.0, 0.0],
                std: [1.0, 0.0, 1.0],
            },
        )
        .unwrap_err();
        assert!(err.contains("invalid image normalization std"));
    }

    #[test]
    fn resize_and_crop_to_target() {
        let src = RgbImage::from_fn(32, 16, |_, _| image::Rgb([255, 0, 0]));
        let out = resize_and_center_crop(&src, 24, 24).unwrap();
        assert_eq!(out.width(), 24);
        assert_eq!(out.height(), 24);
    }

    #[test]
    fn resize_fit_within_preserves_aspect_ratio_and_alignment() {
        let src = RgbImage::from_fn(864, 1152, |_, _| image::Rgb([255, 255, 255]));
        let profile = ImagePreprocessProfile::new_with_mode(
            1024,
            1024,
            ImageNormalization::UnitRange,
            ImageResizeMode::FitWithin,
            32,
        );
        let out = resize_for_profile(&src, profile).unwrap();
        assert_eq!(out.width(), 768);
        assert_eq!(out.height(), 1024);
        assert_eq!(out.width() % 32, 0);
        assert_eq!(out.height() % 32, 0);
    }

    #[test]
    fn fit_within_extreme_aspect_ratios_use_minimum_patch_alignment() {
        let profile = ImagePreprocessProfile::new_with_mode(
            512,
            512,
            ImageNormalization::UnitRange,
            ImageResizeMode::FitWithin,
            32,
        );
        for (source, expected) in [
            ((4096, 64), (512, 32)),
            ((64, 4096), (32, 512)),
            ((512, 31), (512, 32)),
            ((512, 32), (512, 32)),
            ((512, 63), (512, 32)),
            ((512, 64), (512, 64)),
        ] {
            let image = RgbImage::new(source.0, source.1);
            assert_eq!(
                resize_for_profile(&image, profile).unwrap().dimensions(),
                expected
            );
        }
    }

    #[test]
    fn resize_stretch_forces_target_dimensions() {
        let src = RgbImage::from_fn(100, 50, |_, _| image::Rgb([255, 255, 255]));
        let profile = ImagePreprocessProfile::new_with_mode(
            64,
            64,
            ImageNormalization::UnitRange,
            ImageResizeMode::Stretch,
            1,
        );
        let out = resize_for_profile(&src, profile).unwrap();
        assert_eq!(out.width(), 64);
        assert_eq!(out.height(), 64);
    }

    fn write_fixture_image(path: &std::path::Path, format: ImageFormat) {
        let img = RgbImage::from_fn(8, 6, |x, y| image::Rgb([x as u8, y as u8, 128]));
        img.save_with_format(path, format).unwrap();
    }

    #[test]
    fn decode_png_jpeg_webp() {
        let temp = TempDir::new().unwrap();
        let png = temp.path().join("a.png");
        let jpg = temp.path().join("a.jpg");
        let webp = temp.path().join("a.webp");
        write_fixture_image(&png, ImageFormat::Png);
        write_fixture_image(&jpg, ImageFormat::Jpeg);
        write_fixture_image(&webp, ImageFormat::WebP);

        let profile = ImagePreprocessProfile::new(8, 8, ImageNormalization::UnitRange);
        for path in [&png, &jpg, &webp] {
            let prepared =
                prepare_images_for_multimodal(&[path.to_string_lossy().into_owned()], profile)
                    .unwrap();
            assert_eq!(prepared.len(), 1);
            assert_eq!(prepared[0].width, 8);
            assert_eq!(prepared[0].height, 8);
            assert_eq!(prepared[0].element_count(), 8 * 8 * 3);
        }
    }

    #[test]
    fn video_preprocess_returns_no_external_dependency_error() {
        let temp = TempDir::new().unwrap();
        let video_path = temp.path().join("dummy.mp4");
        std::fs::write(&video_path, b"not-a-real-video").unwrap();
        let profile = ImagePreprocessProfile::new(64, 64, ImageNormalization::UnitRange);
        let err = prepare_videos_for_multimodal(
            &[video_path.to_string_lossy().into_owned()],
            profile,
            1,
            4,
            2,
        )
        .expect_err("expected unsupported decoder error");
        assert!(err.contains("no-external-dependency mode"));
    }
}
