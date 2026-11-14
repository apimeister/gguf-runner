//! Face embedding: aligned crop in, L2-normalized vector out.

use image::RgbImage;
use image::imageops::FilterType;

use crate::engine::face::align::{similarity_transform, warp_affine};
use crate::engine::face::conv::FaceTensor;
use crate::engine::face::graph::{FaceGraph, FaceGraphPolicy};
use crate::engine::face::{FACE_LANDMARK_COUNT, FaceInputPolicy, model_fingerprint};
use crate::engine::types::GGUFFile;

const MAX_ENCODER_INPUT_SIZE: usize = 1_024;
const MAX_FACE_EMBEDDING_DIMENSION: usize = 8_192;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FaceEncoderPolicy {
    pub(crate) architecture: String,
    pub(crate) model_name: String,
    pub(crate) graph: FaceGraphPolicy,
    pub(crate) input: FaceInputPolicy,
    /// Side length of the square aligned crop.
    pub(crate) input_size: usize,
    /// Landmark positions inside the aligned crop, in detector landmark order.
    pub(crate) reference: [[f32; 2]; FACE_LANDMARK_COUNT],
    pub(crate) output_slot: usize,
    pub(crate) embedding_dim: usize,
    /// Published same-person cosine threshold; informational, the runner does not match.
    pub(crate) cosine_threshold: f32,
}

pub(crate) struct FaceEncoderModel {
    policy: FaceEncoderPolicy,
    graph: FaceGraph,
    fingerprint: String,
}

impl FaceEncoderModel {
    pub(crate) fn new(gguf: &GGUFFile, policy: FaceEncoderPolicy) -> Result<Self, String> {
        if policy.input_size == 0 || policy.input_size > MAX_ENCODER_INPUT_SIZE {
            return Err(format!(
                "face encoder input size must be within 1..={MAX_ENCODER_INPUT_SIZE}"
            ));
        }
        if policy.embedding_dim == 0 || policy.embedding_dim > MAX_FACE_EMBEDDING_DIMENSION {
            return Err(format!(
                "face embedding dimension must be within 1..={MAX_FACE_EMBEDDING_DIMENSION}"
            ));
        }
        let limit = policy.input_size as f32;
        if policy
            .reference
            .iter()
            .flatten()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > limit)
        {
            return Err("face alignment reference points must lie inside the crop".to_string());
        }
        if !(-1.0..=1.0).contains(&policy.cosine_threshold) {
            return Err("face cosine threshold must be within -1..=1".to_string());
        }
        policy.input.validate()?;
        let graph = FaceGraph::load(gguf, &policy.graph)?;
        if graph.input_channels() != 3 {
            return Err("face encoder must take a three-channel input".to_string());
        }
        if !graph.has_slot(policy.output_slot) {
            return Err("face encoder output slot is not produced by the graph".to_string());
        }
        Ok(Self {
            fingerprint: model_fingerprint(gguf.mapped.as_slice()),
            policy,
            graph,
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
        self.policy.embedding_dim
    }

    pub(crate) fn input_size(&self) -> usize {
        self.policy.input_size
    }

    pub(crate) fn cosine_threshold(&self) -> f32 {
        self.policy.cosine_threshold
    }

    /// Warps the face onto the encoder's canonical crop using its landmarks.
    pub(crate) fn align(
        &self,
        image: &RgbImage,
        landmarks: &[[f32; 2]; FACE_LANDMARK_COUNT],
    ) -> Result<RgbImage, String> {
        let transform = similarity_transform(landmarks, &self.policy.reference)?;
        warp_affine(image, &transform, self.policy.input_size as u32)
    }

    /// Unaligned fallback: stretches the box to the crop size. `bbox` is left, top, width,
    /// height and is clipped to the image.
    pub(crate) fn crop_box(&self, image: &RgbImage, bbox: [f32; 4]) -> Result<RgbImage, String> {
        if bbox.iter().any(|value| !value.is_finite()) {
            return Err("face box is not finite".to_string());
        }
        let (width, height) = (image.width() as f32, image.height() as f32);
        let left = bbox[0].clamp(0.0, width).floor();
        let top = bbox[1].clamp(0.0, height).floor();
        let right = (bbox[0] + bbox[2]).clamp(0.0, width).ceil();
        let bottom = (bbox[1] + bbox[3]).clamp(0.0, height).ceil();
        if right - left < 1.0 || bottom - top < 1.0 {
            return Err("face box does not overlap the image".to_string());
        }
        let view = image::imageops::crop_imm(
            image,
            left as u32,
            top as u32,
            (right - left) as u32,
            (bottom - top) as u32,
        );
        let size = self.policy.input_size as u32;
        Ok(image::imageops::resize(
            &*view,
            size,
            size,
            FilterType::Triangle,
        ))
    }

    /// Embeds a crop that already has the encoder's input size.
    pub(crate) fn embed_crop(&self, crop: &RgbImage) -> Result<Vec<f32>, String> {
        let size = self.policy.input_size;
        if (crop.width() as usize, crop.height() as usize) != (size, size) {
            return Err(format!(
                "face crop is {}x{}, expected {size}x{size}",
                crop.width(),
                crop.height()
            ));
        }
        let plane = size * size;
        let input = self.policy.input;
        let order = input.channel_order.rgb_indices();
        let mut data = vec![0.0f32; 3 * plane];
        for (index, pixel) in crop.as_raw().as_chunks::<3>().0.iter().enumerate() {
            for (channel, source) in order.iter().enumerate() {
                data[channel * plane + index] = input.apply(pixel[*source]);
            }
        }
        let tensor = FaceTensor::new(3, size, size, data)?;
        let mut outputs = self.graph.run(tensor, &[self.policy.output_slot])?;
        let mut vector = outputs
            .pop()
            .ok_or_else(|| "face encoder produced no output".to_string())?
            .data;
        if vector.len() != self.policy.embedding_dim {
            return Err(format!(
                "face encoder produced {} values, expected {}",
                vector.len(),
                self.policy.embedding_dim
            ));
        }
        l2_normalize(&mut vector)?;
        Ok(vector)
    }
}

fn l2_normalize(vector: &mut [f32]) -> Result<(), String> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 1e-8 {
        return Err("face encoder produced a zero or non-finite embedding".to_string());
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

/// Cosine similarity of two L2-normalized embeddings.
pub(crate) fn face_cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.len() != right.len() || left.is_empty() {
        return Err(format!(
            "face embedding dimension mismatch: left={}, right={}",
            left.len(),
            right.len()
        ));
    }
    let score = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    if score.is_finite() {
        Ok(score.clamp(-1.0, 1.0))
    } else {
        Err("face similarity is non-finite".to_string())
    }
}

/// Compact storage form: each normalized component scaled to a signed byte.
pub(crate) fn quantize_face_embedding(vector: &[f32]) -> Vec<i8> {
    vector
        .iter()
        .map(|value| (value * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{face_cosine_similarity, l2_normalize, quantize_face_embedding};

    #[test]
    fn normalization_and_cosine() {
        let mut vector = vec![3.0, 4.0];
        l2_normalize(&mut vector).unwrap();
        assert_eq!(vector, vec![0.6, 0.8]);
        assert!((face_cosine_similarity(&vector, &vector).unwrap() - 1.0).abs() < 1e-6);
        assert!(face_cosine_similarity(&vector, &[1.0]).is_err());
        assert!(l2_normalize(&mut [0.0, 0.0]).is_err());
    }

    #[test]
    fn quantization_rounds_to_signed_bytes() {
        assert_eq!(
            quantize_face_embedding(&[1.0, -1.0, 0.0, 0.5, -0.004]),
            vec![127, -127, 0, 64, -1]
        );
    }
}
