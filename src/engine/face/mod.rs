//! Face detection and face embedding from dedicated GGUF models. Nothing here loads a
//! language model.

mod align;
mod conv;
mod decode;
mod detect;
mod embed;
mod graph;

pub(crate) use conv::FaceActivation;
pub(crate) use decode::{decode_face_image_bytes, decode_face_image_file};
pub(crate) use detect::{
    FaceDetectLevelPolicy, FaceDetection, FaceDetectorModel, FaceDetectorPolicy,
};
pub(crate) use embed::{
    FaceEncoderModel, FaceEncoderPolicy, face_cosine_similarity, quantize_face_embedding,
};
pub(crate) use graph::{FACE_GRAPH_INPUT_SLOT, FaceGraphOpPolicy, FaceGraphPolicy};

/// Right eye, left eye, nose tip, right mouth corner, left mouth corner (subject's view).
pub(crate) const FACE_LANDMARK_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaceChannelOrder {
    Rgb,
    Bgr,
}

impl FaceChannelOrder {
    /// For each network input channel, the index into an RGB pixel.
    pub(crate) fn rgb_indices(self) -> [usize; 3] {
        match self {
            FaceChannelOrder::Rgb => [0, 1, 2],
            FaceChannelOrder::Bgr => [2, 1, 0],
        }
    }
}

/// Network input value for an 8-bit sample: `(sample - mean) * scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FaceInputPolicy {
    pub(crate) channel_order: FaceChannelOrder,
    pub(crate) mean: f32,
    pub(crate) scale: f32,
}

impl FaceInputPolicy {
    #[inline]
    pub(crate) fn apply(&self, sample: u8) -> f32 {
        (f32::from(sample) - self.mean) * self.scale
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.mean.is_finite() || !self.scale.is_finite() || self.scale == 0.0 {
            return Err("face model input normalization is invalid".to_string());
        }
        Ok(())
    }
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
