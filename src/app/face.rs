use std::io::Write;
use std::path::Path;

use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::cli::{CliFaceAction, CliOptions};
use crate::engine::face::{
    FACE_LANDMARK_COUNT, FaceDetection, FaceDetectorModel, FaceEncoderModel,
    decode_face_image_bytes, decode_face_image_file, face_cosine_similarity,
    quantize_face_embedding,
};
use crate::engine::io::{parse_gguf_file, parse_gguf_from_bytes};
use crate::vendors;

/// Margin added around a caller-supplied box before searching it for landmarks, as a
/// fraction of the box size on each side.
const LOCATE_MARGIN: f32 = 0.5;

/// A decoded image in upright orientation (EXIF applied). All face coordinates refer to
/// this pixel grid. Decode once and pass it to both detection and embedding.
pub struct FaceImage {
    pixels: RgbImage,
}

impl FaceImage {
    /// Decode a PNG, JPEG, or WebP file.
    pub fn open(path: &Path) -> Result<Self, String> {
        Ok(Self {
            pixels: decode_face_image_file(path)?,
        })
    }

    /// Decode an encoded PNG, JPEG, or WebP image held in memory.
    pub fn from_encoded_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            pixels: decode_face_image_bytes(bytes)?,
        })
    }

    /// Wrap already-upright interleaved RGB8 pixels.
    pub fn from_rgb8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("face image is empty".to_string());
        }
        RgbImage::from_raw(width, height, pixels)
            .map(|pixels| Self { pixels })
            .ok_or_else(|| "RGB8 buffer does not match the image dimensions".to_string())
    }

    pub fn width(&self) -> u32 {
        self.pixels.width()
    }

    pub fn height(&self) -> u32 {
        self.pixels.height()
    }
}

/// Axis-aligned box in upright image pixels. Detector boxes may extend past the image edge.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl FaceBox {
    fn as_array(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }
}

/// One face in an image. Landmark order is right eye, left eye, nose tip, right mouth
/// corner, left mouth corner, from the subject's point of view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectedFace {
    pub bbox: FaceBox,
    /// Absent for a box that did not come from the detector.
    pub landmarks: Option<[[f32; 2]; FACE_LANDMARK_COUNT]>,
    /// Detector confidence in 0..=1; absent for a box that did not come from the detector.
    pub score: Option<f32>,
}

impl DetectedFace {
    /// A face region known only as a box, for example one drawn by a user. Pass it to
    /// `FaceDetector::locate` to obtain landmarks before embedding.
    pub fn from_box(bbox: FaceBox) -> Self {
        Self {
            bbox,
            landmarks: None,
            score: None,
        }
    }

    fn from_detection(detection: &FaceDetection) -> Self {
        let [x, y, width, height] = detection.bbox;
        Self {
            bbox: FaceBox {
                x,
                y,
                width,
                height,
            },
            landmarks: Some(detection.landmarks),
            score: Some(detection.score),
        }
    }
}

/// L2-normalized face vector. Vectors are comparable only when produced by the same encoder
/// model.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceEmbedding {
    pub vector: Vec<f32>,
    /// False when the face had no landmarks and was embedded from a plain box crop; such
    /// vectors match less reliably.
    pub aligned: bool,
}

impl FaceEmbedding {
    /// Compact form: each component multiplied by 127 and rounded to a signed byte.
    pub fn quantized(&self) -> Vec<i8> {
        quantize_face_embedding(&self.vector)
    }

    /// `quantized()` as standard padded base64.
    pub fn quantized_base64(&self) -> String {
        let bytes: Vec<u8> = self.quantized().iter().map(|value| *value as u8).collect();
        base64_encode(&bytes)
    }

    /// Cosine similarity in -1..=1; higher means more likely the same person.
    pub fn cosine_similarity(&self, other: &FaceEmbedding) -> Result<f32, String> {
        face_cosine_similarity(&self.vector, &other.vector)
    }

    /// Cosine similarity of two stored `quantized()` vectors, without restoring floats. It
    /// stays within about 0.02 of the float result.
    pub fn quantized_cosine_similarity(left: &[i8], right: &[i8]) -> Result<f32, String> {
        if left.len() != right.len() || left.is_empty() {
            return Err(format!(
                "face embedding dimension mismatch: left={}, right={}",
                left.len(),
                right.len()
            ));
        }
        let (mut dot, mut left_norm, mut right_norm) = (0i64, 0i64, 0i64);
        for (a, b) in left.iter().zip(right) {
            let (a, b) = (i64::from(*a), i64::from(*b));
            dot += a * b;
            left_norm += a * a;
            right_norm += b * b;
        }
        if left_norm == 0 || right_norm == 0 {
            return Err("stored face embedding is all zero".to_string());
        }
        let score = dot as f64 / ((left_norm as f64).sqrt() * (right_norm as f64).sqrt());
        Ok(score.clamp(-1.0, 1.0) as f32)
    }

    /// Decodes the `vector_q8` / `quantized_base64()` text form back to signed bytes.
    pub fn decode_quantized_base64(text: &str) -> Result<Vec<i8>, String> {
        Ok(base64_decode(text)?
            .into_iter()
            .map(|value| value as i8)
            .collect())
    }
}

/// One face of a processed image, as written by the CLI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceRecord {
    pub bbox: FaceBox,
    pub landmarks: Option<[[f32; 2]; FACE_LANDMARK_COUNT]>,
    pub score: Option<f32>,
    /// Present only when embedding was requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aligned: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Base64 of the signed-byte form of `vector`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_q8: Option<String>,
}

/// All faces of one image, best detector score first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceImageResult {
    pub source: String,
    /// Upright dimensions that the face coordinates refer to.
    pub image_width: u32,
    pub image_height: u32,
    pub faces: Vec<FaceRecord>,
}

/// Face detector loaded from a dedicated GGUF. This does not load a language model.
pub struct FaceDetector {
    model: FaceDetectorModel,
}

impl FaceDetector {
    /// Load a face detector GGUF embedded in the binary.
    pub fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        let gguf = parse_gguf_from_bytes(data, false)
            .map_err(|error| format!("failed to load embedded face detector GGUF: {error}"))?;
        let policy = vendors::face_detector_policy(&gguf)?;
        Ok(Self {
            model: FaceDetectorModel::new(&gguf, policy)?,
        })
    }

    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        Self::load_from_file_with_debug(path, false)
    }

    pub(crate) fn load_from_file_with_debug(path: &Path, debug: bool) -> Result<Self, String> {
        let path_string = path
            .to_str()
            .ok_or_else(|| "face detector path contains non-UTF8 characters".to_string())?;
        let gguf = parse_gguf_file(path_string, debug).map_err(|error| {
            format!("failed to load face detector GGUF '{path_string}': {error}")
        })?;
        let policy = vendors::face_detector_policy(&gguf)?;
        Ok(Self {
            model: FaceDetectorModel::new(&gguf, policy)?,
        })
    }

    pub fn model_name(&self) -> &str {
        self.model.model_name()
    }

    pub fn model_architecture(&self) -> &str {
        self.model.architecture()
    }

    pub fn model_fingerprint(&self) -> &str {
        self.model.fingerprint()
    }

    /// The model's default minimum detection score.
    pub fn score_threshold(&self) -> f32 {
        self.model.score_threshold()
    }

    /// Detect faces with the model's default score threshold, best score first.
    pub fn detect(&self, image: &FaceImage) -> Result<Vec<DetectedFace>, String> {
        self.detect_with_threshold(image, self.model.score_threshold())
    }

    /// Detect faces scoring above `score_threshold` (0..=1), best score first.
    pub fn detect_with_threshold(
        &self,
        image: &FaceImage,
        score_threshold: f32,
    ) -> Result<Vec<DetectedFace>, String> {
        Ok(self
            .model
            .detect(&image.pixels, Some(score_threshold))?
            .iter()
            .map(DetectedFace::from_detection)
            .collect())
    }

    /// Find landmarks for a face known only as a box by running the detector on the box
    /// plus a margin. Returns the detection that overlaps the box most, or `None`.
    pub fn locate(
        &self,
        image: &FaceImage,
        bbox: &FaceBox,
    ) -> Result<Option<DetectedFace>, String> {
        if bbox.as_array().iter().any(|value| !value.is_finite())
            || bbox.width <= 0.0
            || bbox.height <= 0.0
        {
            return Err("face box must be finite with positive size".to_string());
        }
        let (image_w, image_h) = (image.width() as f32, image.height() as f32);
        let left = (bbox.x - bbox.width * LOCATE_MARGIN)
            .clamp(0.0, image_w)
            .floor();
        let top = (bbox.y - bbox.height * LOCATE_MARGIN)
            .clamp(0.0, image_h)
            .floor();
        let right = (bbox.x + bbox.width * (1.0 + LOCATE_MARGIN))
            .clamp(0.0, image_w)
            .ceil();
        let bottom = (bbox.y + bbox.height * (1.0 + LOCATE_MARGIN))
            .clamp(0.0, image_h)
            .ceil();
        if right - left < 1.0 || bottom - top < 1.0 {
            return Ok(None);
        }
        let region = image::imageops::crop_imm(
            &image.pixels,
            left as u32,
            top as u32,
            (right - left) as u32,
            (bottom - top) as u32,
        )
        .to_image();
        let best = self
            .model
            .detect(&region, None)?
            .into_iter()
            .map(|mut detection| {
                detection.bbox[0] += left;
                detection.bbox[1] += top;
                for point in &mut detection.landmarks {
                    *point = [point[0] + left, point[1] + top];
                }
                (overlap(&detection.bbox, &bbox.as_array()), detection)
            })
            .filter(|(overlap, _)| *overlap > 0.0)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        Ok(best.map(|(_, detection)| DetectedFace::from_detection(&detection)))
    }
}

/// Intersection over union of two left/top/width/height boxes.
fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let width = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
    let height = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = width * height;
    let union = a[2] * a[3] + b[2] * b[3] - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Face embedding model loaded from a dedicated GGUF. This does not load a language model.
pub struct FaceEncoder {
    model: FaceEncoderModel,
}

impl FaceEncoder {
    /// Load a face encoder GGUF embedded in the binary.
    pub fn load_from_bytes(data: &'static [u8]) -> Result<Self, String> {
        let gguf = parse_gguf_from_bytes(data, false)
            .map_err(|error| format!("failed to load embedded face encoder GGUF: {error}"))?;
        let policy = vendors::face_encoder_policy(&gguf)?;
        Ok(Self {
            model: FaceEncoderModel::new(&gguf, policy)?,
        })
    }

    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        Self::load_from_file_with_debug(path, false)
    }

    pub(crate) fn load_from_file_with_debug(path: &Path, debug: bool) -> Result<Self, String> {
        let path_string = path
            .to_str()
            .ok_or_else(|| "face encoder path contains non-UTF8 characters".to_string())?;
        let gguf = parse_gguf_file(path_string, debug).map_err(|error| {
            format!("failed to load face encoder GGUF '{path_string}': {error}")
        })?;
        let policy = vendors::face_encoder_policy(&gguf)?;
        Ok(Self {
            model: FaceEncoderModel::new(&gguf, policy)?,
        })
    }

    pub fn model_name(&self) -> &str {
        self.model.model_name()
    }

    pub fn model_architecture(&self) -> &str {
        self.model.architecture()
    }

    pub fn model_fingerprint(&self) -> &str {
        self.model.fingerprint()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.model.embedding_dim()
    }

    /// Side length of the square crop the model consumes.
    pub fn crop_size(&self) -> usize {
        self.model.input_size()
    }

    /// The cosine similarity above which the model's publisher treats two faces as the same
    /// person. A starting point only; the runner never applies it.
    pub fn published_match_threshold(&self) -> f32 {
        self.model.cosine_threshold()
    }

    /// Embed one face of `image`. A face with landmarks is aligned first; a face without
    /// landmarks is embedded from its plain box crop and reported as not aligned.
    pub fn embed(&self, image: &FaceImage, face: &DetectedFace) -> Result<FaceEmbedding, String> {
        let (crop, aligned) = match &face.landmarks {
            Some(landmarks) => (self.model.align(&image.pixels, landmarks)?, true),
            None => (
                self.model.crop_box(&image.pixels, face.bbox.as_array())?,
                false,
            ),
        };
        Ok(FaceEmbedding {
            vector: self.model.embed_crop(&crop)?,
            aligned,
        })
    }

    /// Embed a crop that is already aligned to the model's reference landmarks and has
    /// `crop_size()` on both sides.
    pub fn embed_aligned_crop(&self, crop: &FaceImage) -> Result<FaceEmbedding, String> {
        Ok(FaceEmbedding {
            vector: self.model.embed_crop(&crop.pixels)?,
            aligned: true,
        })
    }

    /// Detect and embed every face of an already decoded image, best detector score first.
    pub fn embed_image(
        &self,
        detector: &FaceDetector,
        image: &FaceImage,
    ) -> Result<Vec<FaceRecord>, String> {
        detector
            .detect(image)?
            .into_iter()
            .map(|face| {
                let embedding = self.embed(image, &face)?;
                Ok(FaceRecord {
                    bbox: face.bbox,
                    landmarks: face.landmarks,
                    score: face.score,
                    aligned: Some(embedding.aligned),
                    vector_q8: Some(embedding.quantized_base64()),
                    vector: Some(embedding.vector),
                })
            })
            .collect()
    }

    /// Detect and embed every face of one image file.
    pub fn embed_all(
        &self,
        detector: &FaceDetector,
        image_path: &Path,
    ) -> Result<FaceImageResult, String> {
        let image = FaceImage::open(image_path)?;
        let faces = self.embed_image(detector, &image)?;
        Ok(image_result(image_path, &image, faces))
    }
}

fn image_result(path: &Path, image: &FaceImage, faces: Vec<FaceRecord>) -> FaceImageResult {
    FaceImageResult {
        source: path.to_string_lossy().into_owned(),
        image_width: image.width(),
        image_height: image.height(),
        faces,
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let block = u32::from(chunk[0]) << 16
            | u32::from(*chunk.get(1).unwrap_or(&0)) << 8
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for position in 0..4 {
            if position <= chunk.len() {
                let index = (block >> (18 - 6 * position)) & 0x3f;
                out.push(ALPHABET[index as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let invalid = || "invalid base64 face embedding".to_string();
    let bytes = text.trim().as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid());
    }
    let padding = bytes.iter().rev().take_while(|byte| **byte == b'=').count();
    if padding > 2 {
        return Err(invalid());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (index, chunk) in bytes.chunks(4).enumerate() {
        let last = (index + 1) * 4 == bytes.len();
        let mut block = 0u32;
        for (position, byte) in chunk.iter().enumerate() {
            let value = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' if last && position >= 4 - padding => 0,
                _ => return Err(invalid()),
            };
            block = block << 6 | u32::from(value);
        }
        let keep = if last { 3 - padding } else { 3 };
        out.extend_from_slice(&block.to_be_bytes()[1..1 + keep]);
    }
    Ok(out)
}

fn write_stdout_json(value: &impl Serialize) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, value)
        .map_err(|error| format!("cannot serialize face result: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("cannot write face result: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("cannot flush face result: {error}"))
}

/// Runs a `--face-*` action: one JSON line per `--image`, in input order.
pub(crate) fn run_cli(cli: &CliOptions) -> Result<(), String> {
    let action = cli
        .face_action
        .ok_or_else(|| "internal error: face mode has no action".to_string())?;
    let detector_path = cli
        .face_detector
        .as_deref()
        .ok_or_else(|| "face actions require --face-detector <detector.gguf>".to_string())?;
    let detector = FaceDetector::load_from_file_with_debug(Path::new(detector_path), cli.debug)?;
    let encoder = match action {
        CliFaceAction::Detect => None,
        CliFaceAction::Embed => {
            let path = cli
                .face_model
                .as_deref()
                .ok_or_else(|| "--face-embed requires --face-model <encoder.gguf>".to_string())?;
            Some(FaceEncoder::load_from_file_with_debug(
                Path::new(path),
                cli.debug,
            )?)
        }
    };
    if cli.debug {
        eprintln!(
            "face detector: {} ({})",
            detector.model_name(),
            detector.model_architecture()
        );
        if let Some(encoder) = &encoder {
            eprintln!(
                "face encoder: {} ({}), {} dimensions",
                encoder.model_name(),
                encoder.model_architecture(),
                encoder.embedding_dimension()
            );
        }
    }
    let threshold = cli
        .face_score_threshold
        .unwrap_or_else(|| detector.score_threshold());
    for path in &cli.images {
        let path = Path::new(path);
        let image = FaceImage::open(path)?;
        let mut faces = Vec::new();
        for face in detector.detect_with_threshold(&image, threshold)? {
            let embedding = match &encoder {
                Some(encoder) => Some(encoder.embed(&image, &face)?),
                None => None,
            };
            faces.push(FaceRecord {
                bbox: face.bbox,
                landmarks: face.landmarks,
                score: face.score,
                aligned: embedding.as_ref().map(|e| e.aligned),
                vector_q8: embedding.as_ref().map(FaceEmbedding::quantized_base64),
                vector: embedding.map(|e| e.vector),
            });
        }
        write_stdout_json(&image_result(path, &image, faces))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DetectedFace, FaceBox, FaceDetector, FaceEmbedding, FaceEncoder, FaceImage,
        FaceImageResult, FaceRecord, base64_decode, base64_encode, overlap,
    };

    #[test]
    fn base64_matches_rfc_4648_vectors() {
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), expected);
        }
    }

    #[test]
    fn quantized_form_is_one_signed_byte_per_dimension() {
        let embedding = FaceEmbedding {
            vector: vec![1.0, -1.0, 0.0],
            aligned: true,
        };
        assert_eq!(embedding.quantized(), vec![127, -127, 0]);
        assert_eq!(embedding.quantized_base64(), "f4EA");
        assert_eq!(embedding.cosine_similarity(&embedding).unwrap(), 1.0);
    }

    #[test]
    fn base64_round_trips_and_rejects_malformed_text() {
        for input in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
            let encoded = base64_encode(input.as_bytes());
            assert_eq!(base64_decode(&encoded).unwrap(), input.as_bytes());
        }
        for bad in ["Zg=", "Z===", "Zg==Zg==", "Zm9v!A==", "=m9v"] {
            assert!(base64_decode(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn stored_vectors_compare_without_restoring_floats() {
        let a = FaceEmbedding {
            vector: vec![0.6, 0.8, 0.0],
            aligned: true,
        };
        let b = FaceEmbedding {
            vector: vec![0.0, 0.6, 0.8],
            aligned: true,
        };
        let stored_a = FaceEmbedding::decode_quantized_base64(&a.quantized_base64()).unwrap();
        assert_eq!(stored_a, a.quantized());
        let float = a.cosine_similarity(&b).unwrap();
        let stored = FaceEmbedding::quantized_cosine_similarity(&stored_a, &b.quantized()).unwrap();
        assert!((float - stored).abs() < 0.01, "{float} vs {stored}");
        assert!(FaceEmbedding::quantized_cosine_similarity(&stored_a, &[1, 2]).is_err());
        assert!(FaceEmbedding::quantized_cosine_similarity(&[0, 0], &[1, 2]).is_err());
    }

    #[test]
    fn results_round_trip_through_json() {
        let result = FaceImageResult {
            source: "a.jpg".to_string(),
            image_width: 4,
            image_height: 3,
            faces: vec![FaceRecord {
                bbox: FaceBox {
                    x: 1.0,
                    y: 2.0,
                    width: 3.0,
                    height: 4.0,
                },
                landmarks: Some([[0.5; 2]; 5]),
                score: Some(0.9),
                aligned: None,
                vector: None,
                vector_q8: None,
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("vector"));
        assert_eq!(
            serde_json::from_str::<FaceImageResult>(&json).unwrap(),
            result
        );
    }

    #[test]
    fn loaded_models_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FaceDetector>();
        assert_send_sync::<FaceEncoder>();
        assert_send_sync::<FaceImage>();
    }

    #[test]
    fn rgb8_images_validate_their_buffer() {
        assert!(FaceImage::from_rgb8(2, 2, vec![0; 12]).is_ok());
        assert!(FaceImage::from_rgb8(2, 2, vec![0; 11]).is_err());
        assert!(FaceImage::from_rgb8(0, 2, Vec::new()).is_err());
    }

    #[test]
    fn manual_boxes_have_no_landmarks_or_score() {
        let face = DetectedFace::from_box(FaceBox {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        });
        assert!(face.landmarks.is_none() && face.score.is_none());
    }

    #[test]
    fn overlap_is_intersection_over_union() {
        assert_eq!(overlap(&[0.0, 0.0, 2.0, 2.0], &[0.0, 0.0, 2.0, 2.0]), 1.0);
        assert_eq!(overlap(&[0.0, 0.0, 2.0, 2.0], &[5.0, 5.0, 2.0, 2.0]), 0.0);
    }
}
