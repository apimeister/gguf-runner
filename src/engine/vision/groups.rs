//! Snapshot a source, preflight fixed-view resources, then decode/prepare lazily.
//! Request-wide prompt/context checks belong to orchestration before encoding.

use super::bilinear;
use super::preprocess::{
    ImageNormalization, ImagePreprocessProfile, ImageResizeMode, PreparedImageTensor,
    rgb_u8_to_chw_f32,
};
use super::views::plan_image_views;
use crate::engine::types::{
    ImageOrientationPolicy, ImageSourceLimits, ImageSourcePlan, ImageStretchFilter,
    ImageViewEncoding, ImageViewLimits, ImageViewPolicy, ImageViewSpec,
};
use image::{
    ColorType, DynamicImage, ImageDecoder, ImageReader, Limits, RgbImage, metadata::Orientation,
};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

#[cfg(test)]
#[path = "groups_tests.rs"]
mod tests;

#[derive(Debug)]
pub(crate) struct ImageViewGroupPlan {
    pub(crate) source_path: String,
    pub(crate) geometry: ImageSourcePlan,
    pub(crate) profile: ImagePreprocessProfile,
    pub(crate) encoding: ImageViewEncoding,
    pub(crate) preparation_bytes: usize,
    pub(crate) embedding_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct PlannedImageSource {
    plan: ImageViewGroupPlan,
    bytes: Vec<u8>,
    encoded_size: (u32, u32),
    orientation: Orientation,
    limits: ImageSourceLimits,
}

#[derive(Debug)]
pub(crate) struct DecodedImageSource {
    plan: ImageViewGroupPlan,
    pixels: RgbImage,
}

#[derive(Debug)]
pub(crate) struct PreparedImageView {
    pub(crate) spec: ImageViewSpec,
    pub(crate) encoding: ImageViewEncoding,
    pub(crate) tensor: PreparedImageTensor,
}

fn overflow() -> String {
    "image group resource size overflow".to_string()
}

fn validate_contract(
    profile: ImagePreprocessProfile,
    encoding: ImageViewEncoding,
    limits: ImageSourceLimits,
) -> Result<(), String> {
    if [
        limits.max_file_bytes,
        limits.max_source_pixels,
        limits.max_decoder_bytes,
        limits.max_prepare_bytes,
        limits.max_embedding_bytes,
        limits.max_views,
    ]
    .contains(&0)
        || limits.max_file_bytes >= isize::MAX as usize
    {
        return Err(
            "image source limits must be positive and file storage addressable".to_string(),
        );
    }
    if profile.resize_mode != ImageResizeMode::Stretch
        || profile.stretch_filter != ImageStretchFilter::PillowBilinear
        || profile.align_to != 1
    {
        return Err(
            "grouped image preparation requires fixed Pillow-bilinear stretching".to_string(),
        );
    }
    if encoding.width == 0
        || encoding.height == 0
        || encoding.width > i32::MAX as u32
        || encoding.height > i32::MAX as u32
        || profile.target_width != encoding.width as usize
        || profile.target_height != encoding.height as usize
        || encoding.tokens == 0
        || encoding.dimension == 0
    {
        return Err("invalid or inconsistent fixed-view encoding contract".to_string());
    }
    encoding
        .tokens
        .checked_mul(encoding.dimension)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(overflow)?;
    if let Some(grid) = encoding.grid
        && (grid.contains(&0)
            || grid
                .into_iter()
                .try_fold(1usize, |n, axis| n.checked_mul(axis))
                != Some(encoding.tokens))
    {
        return Err("image view grid does not match the embedding token count".to_string());
    }
    if let ImageNormalization::MeanStd { mean, std } = profile.normalization {
        for channel in 0..3 {
            if !mean[channel].is_finite()
                || !std[channel].is_finite()
                || std[channel] <= 0.0
                || !((0.0 - mean[channel]) / std[channel]).is_finite()
                || !((1.0 - mean[channel]) / std[channel]).is_finite()
            {
                return Err(
                    "image normalization must produce finite F32 values for RGB8 input".to_string(),
                );
            }
        }
    }
    Ok(())
}

fn codec_limits(limits: ImageSourceLimits) -> Limits {
    let mut codec = Limits::default();
    // An axis cannot exceed the permitted number of source pixels. The product
    // is checked separately before calling read_image via from_decoder.
    let max_axis = limits.max_source_pixels.min(i32::MAX as usize) as u32;
    codec.max_image_width = Some(max_axis);
    codec.max_image_height = Some(max_axis);
    codec.max_alloc = Some(limits.max_decoder_bytes as u64);
    codec
}

fn decoder(bytes: &[u8], limits: ImageSourceLimits) -> Result<impl ImageDecoder + '_, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    reader.limits(codec_limits(limits));
    reader
        .into_decoder()
        .map_err(|e| format!("cannot probe image source: {e}"))
}

fn oriented_size(size: (u32, u32), orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (size.1, size.0),
        _ => size,
    }
}

impl PlannedImageSource {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open(
        path: &Path,
        source_index: usize,
        policy: ImageViewPolicy,
        profile: ImagePreprocessProfile,
        encoding: ImageViewEncoding,
        orientation_policy: ImageOrientationPolicy,
        limits: ImageSourceLimits,
    ) -> Result<Self, String> {
        validate_contract(profile, encoding, limits)?;
        if !std::fs::metadata(path)
            .map_err(|e| format!("cannot inspect image '{}': {e}", path.display()))?
            .is_file()
        {
            return Err("image source must be a regular file".to_string());
        }
        let mut file =
            File::open(path).map_err(|e| format!("cannot open image '{}': {e}", path.display()))?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() > limits.max_file_bytes as u64 {
            return Err(
                "image source must be a regular file within the encoded-byte limit".to_string(),
            );
        }
        let file_len = usize::try_from(metadata.len()).map_err(|_| overflow())?;
        if file_len > limits.max_prepare_bytes {
            return Err("image snapshot exceeds the preparation-buffer byte limit".to_string());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|_| "unable to allocate image source snapshot".to_string())?;
        bytes.resize(file_len, 0);
        file.read_exact(&mut bytes)
            .map_err(|e| format!("cannot read image source snapshot: {e}"))?;
        let mut extra = [0u8; 1];
        if file.read(&mut extra).map_err(|e| e.to_string())? != 0 {
            return Err("image source size changed while taking its snapshot".to_string());
        }
        let mut decoder = decoder(&bytes, limits)?;
        let encoded_size = decoder.dimensions();
        if !matches!(
            decoder.color_type(),
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
        ) {
            return Err(
                "grouped image decoding currently requires 8-bit image channels".to_string(),
            );
        }
        let pixels = (encoded_size.0 as usize)
            .checked_mul(encoded_size.1 as usize)
            .ok_or_else(overflow)?;
        if pixels == 0 || pixels > limits.max_source_pixels {
            return Err("image source exceeds the decoded-pixel limit".to_string());
        }
        let decoded_bytes = usize::try_from(decoder.total_bytes()).map_err(|_| overflow())?;
        if decoded_bytes > limits.max_decoder_bytes {
            return Err("image source exceeds the decoded-byte limit".to_string());
        }
        let orientation = match orientation_policy {
            ImageOrientationPolicy::EncodedPixels => Orientation::NoTransforms,
            ImageOrientationPolicy::ApplyExif => decoder
                .orientation()
                .map_err(|e| format!("cannot read image orientation: {e}"))?,
        };
        drop(decoder);
        let size = oriented_size(encoded_size, orientation);
        let geometry = plan_image_views(
            source_index,
            size.0,
            size.1,
            policy,
            ImageViewLimits {
                max_source_pixels: limits.max_source_pixels,
                max_views: limits.max_views,
            },
        )
        .map_err(|e| e.to_string())?;
        let embedding_bytes = geometry
            .views
            .len()
            .checked_mul(encoding.tokens)
            .and_then(|n| n.checked_mul(encoding.dimension))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(overflow)?;
        if embedding_bytes > limits.max_embedding_bytes {
            return Err("image group exceeds the retained-embedding byte limit".to_string());
        }
        let rgb_bytes = pixels.checked_mul(3).ok_or_else(overflow)?;
        // Decode and conversion can briefly hold two full images. Include the
        // immutable compressed snapshot; codec scratch is governed separately.
        let decode_peak = bytes
            .capacity()
            .checked_add(
                decoded_bytes
                    .checked_mul(2)
                    .ok_or_else(overflow)?
                    .max(decoded_bytes.checked_add(rgb_bytes).ok_or_else(overflow)?),
            )
            .ok_or_else(overflow)?;
        let target_pixels = profile
            .target_width
            .checked_mul(profile.target_height)
            .ok_or_else(overflow)?;
        let mut preparation_bytes = decode_peak;
        for view in &geometry.views {
            let width = (view.rect.right - view.rect.left) as usize;
            let height = (view.rect.bottom - view.rect.top) as usize;
            let crop_bytes = if view.view_index == 0 {
                0
            } else {
                width
                    .checked_mul(height)
                    .and_then(|n| n.checked_mul(3))
                    .ok_or_else(overflow)?
            };
            let resize =
                bilinear::storage_bound(width, height, profile.target_width, profile.target_height)
                    .ok_or_else(overflow)?;
            let peak = rgb_bytes
                .checked_add(crop_bytes)
                .and_then(|n| n.checked_add(resize))
                .and_then(|n| n.checked_add(target_pixels.checked_mul(12)?))
                .ok_or_else(overflow)?;
            preparation_bytes = preparation_bytes.max(peak);
        }
        if preparation_bytes > limits.max_prepare_bytes {
            return Err("image group exceeds the preparation-buffer byte limit".to_string());
        }
        Ok(Self {
            plan: ImageViewGroupPlan {
                source_path: path.to_string_lossy().into_owned(),
                geometry,
                profile,
                encoding,
                preparation_bytes,
                embedding_bytes,
            },
            bytes,
            encoded_size,
            orientation,
            limits,
        })
    }

    pub(crate) fn plan(&self) -> &ImageViewGroupPlan {
        &self.plan
    }

    pub(crate) fn snapshot_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    pub(crate) fn decode(self) -> Result<DecodedImageSource, String> {
        let mut decoder = decoder(&self.bytes, self.limits)?;
        if decoder.dimensions() != self.encoded_size {
            return Err("decoded image geometry differs from its plan".to_string());
        }
        let mut remaining = codec_limits(self.limits);
        remaining
            .reserve(decoder.total_bytes())
            .map_err(|e| format!("image decode allocation rejected: {e}"))?;
        decoder.set_limits(remaining).map_err(|e| e.to_string())?;
        let mut decoded = DynamicImage::from_decoder(decoder)
            .map_err(|e| format!("cannot decode image source: {e}"))?;
        decoded.apply_orientation(self.orientation);
        // RGB conversion drops alpha, matching PIL convert("RGB"); it does not
        // composite transparent samples onto a fabricated background.
        let pixels = decoded.into_rgb8();
        if pixels.dimensions()
            != (
                self.plan.geometry.source_width,
                self.plan.geometry.source_height,
            )
        {
            return Err("oriented image geometry differs from its plan".to_string());
        }
        Ok(DecodedImageSource {
            plan: self.plan,
            pixels,
        })
    }
}

impl DecodedImageSource {
    pub(crate) fn plan(&self) -> &ImageViewGroupPlan {
        &self.plan
    }
    pub(crate) fn into_plan(self) -> ImageViewGroupPlan {
        self.plan
    }

    /// The caller releases the returned view before requesting the next one.
    pub(crate) fn prepare_view(&self, index: usize) -> Result<PreparedImageView, String> {
        let spec = *self
            .plan
            .geometry
            .views
            .get(index)
            .ok_or_else(|| "image view index is out of range".to_string())?;
        let crop;
        let source = if index == 0 {
            &self.pixels
        } else {
            let width = spec.rect.right - spec.rect.left;
            let height = spec.rect.bottom - spec.rect.top;
            let count = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(overflow)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(count)
                .map_err(|_| "unable to allocate source crop".to_string())?;
            for row in spec.rect.top..spec.rect.bottom {
                let start =
                    (row as usize * self.pixels.width() as usize + spec.rect.left as usize) * 3;
                bytes.extend_from_slice(&self.pixels.as_raw()[start..start + width as usize * 3]);
            }
            crop = RgbImage::from_raw(width, height, bytes)
                .ok_or_else(|| "source crop buffer shape mismatch".to_string())?;
            &crop
        };
        let profile = self.plan.profile;
        let rgb = bilinear::resize_rgb8(source, profile.target_width, profile.target_height)?;
        let data_chw = rgb_u8_to_chw_f32(
            rgb.as_raw(),
            profile.target_width,
            profile.target_height,
            profile.normalization,
        )?;
        if data_chw.iter().any(|value| !value.is_finite()) {
            return Err("image view contains nonfinite normalized values".to_string());
        }
        Ok(PreparedImageView {
            spec,
            encoding: self.plan.encoding,
            tensor: PreparedImageTensor {
                path: self.plan.source_path.clone(),
                width: profile.target_width,
                height: profile.target_height,
                data_chw,
            },
        })
    }
}
