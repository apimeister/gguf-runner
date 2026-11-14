//! Bounded image decoding for the face path. Unlike the vision-language single-view path,
//! EXIF orientation is always applied: the detector does not find sideways faces.

use std::io::Cursor;
use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageReader, Limits, RgbImage};

use crate::engine::types::ImageOrientationPolicy;

const MAX_FACE_IMAGE_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_FACE_IMAGE_PIXELS: u64 = 256 * 1024 * 1024;
const MAX_FACE_IMAGE_DECODER_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Orientation contract of every image the face path decodes.
pub(crate) const FACE_IMAGE_ORIENTATION: ImageOrientationPolicy = ImageOrientationPolicy::ApplyExif;

pub(crate) fn decode_face_image_file(path: &Path) -> Result<RgbImage, String> {
    let display = path.display();
    let metadata =
        std::fs::metadata(path).map_err(|e| format!("cannot open image '{display}': {e}"))?;
    if metadata.len() > MAX_FACE_IMAGE_FILE_BYTES {
        return Err(format!("image '{display}' exceeds the file-size limit"));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read image '{display}': {e}"))?;
    decode_face_image_bytes(&bytes).map_err(|e| format!("image '{display}': {e}"))
}

pub(crate) fn decode_face_image_bytes(bytes: &[u8]) -> Result<RgbImage, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_FACE_IMAGE_DECODER_BYTES);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("cannot probe image: {e}"))?;
    let (width, height) = decoder.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > MAX_FACE_IMAGE_PIXELS {
        return Err("image exceeds the decoded-pixel limit".to_string());
    }
    let orientation = match FACE_IMAGE_ORIENTATION {
        ImageOrientationPolicy::ApplyExif => decoder
            .orientation()
            .map_err(|e| format!("cannot read image orientation: {e}"))?,
        ImageOrientationPolicy::EncodedPixels => image::metadata::Orientation::NoTransforms,
    };
    let mut decoded =
        DynamicImage::from_decoder(decoder).map_err(|e| format!("cannot decode image: {e}"))?;
    decoded.apply_orientation(orientation);
    // Alpha is dropped, matching the vision path; it is not composited onto a background.
    Ok(decoded.into_rgb8())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{ImageFormat, Rgb, RgbImage};

    use super::decode_face_image_bytes;

    #[test]
    fn decodes_png_to_rgb() {
        let mut source = RgbImage::new(3, 2);
        source.put_pixel(2, 1, Rgb([9, 8, 7]));
        let mut bytes = Vec::new();
        source
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        let decoded = decode_face_image_bytes(&bytes).unwrap();
        assert_eq!(decoded.dimensions(), (3, 2));
        assert_eq!(decoded.get_pixel(2, 1), &Rgb([9, 8, 7]));
    }

    #[test]
    fn rejects_non_image_bytes() {
        assert!(decode_face_image_bytes(b"not an image").is_err());
    }
}
