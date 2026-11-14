//! Anchor-free multi-scale face detector: preprocessing, head decoding, suppression.

use image::RgbImage;
use image::imageops::FilterType;

use crate::engine::face::conv::FaceTensor;
use crate::engine::face::graph::{FaceGraph, FaceGraphPolicy};
use crate::engine::face::{FACE_LANDMARK_COUNT, FaceInputPolicy, model_fingerprint};
use crate::engine::types::GGUFFile;

const MAX_DETECTOR_SIDE: usize = 4_096;
const MAX_DETECTOR_LEVELS: usize = 8;

/// Head output slots for one feature-map stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FaceDetectLevelPolicy {
    pub(crate) stride: usize,
    pub(crate) cls_slot: usize,
    pub(crate) obj_slot: usize,
    pub(crate) bbox_slot: usize,
    pub(crate) kps_slot: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FaceDetectorPolicy {
    pub(crate) architecture: String,
    pub(crate) model_name: String,
    pub(crate) graph: FaceGraphPolicy,
    pub(crate) input: FaceInputPolicy,
    /// Longest image side fed to the network; larger images are downscaled first.
    pub(crate) max_side: usize,
    /// The padded input is a multiple of this value in both axes.
    pub(crate) align: usize,
    pub(crate) levels: Vec<FaceDetectLevelPolicy>,
    pub(crate) score_threshold: f32,
    pub(crate) nms_threshold: f32,
    pub(crate) top_k: usize,
}

/// One detection in the pixel coordinates of the image passed to `detect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FaceDetection {
    /// Left, top, width, height. May extend past the image edge.
    pub(crate) bbox: [f32; 4],
    pub(crate) landmarks: [[f32; 2]; FACE_LANDMARK_COUNT],
    pub(crate) score: f32,
}

pub(crate) struct FaceDetectorModel {
    policy: FaceDetectorPolicy,
    graph: FaceGraph,
    fingerprint: String,
}

impl FaceDetectorModel {
    pub(crate) fn new(gguf: &GGUFFile, policy: FaceDetectorPolicy) -> Result<Self, String> {
        validate_policy(&policy)?;
        let graph = FaceGraph::load(gguf, &policy.graph)?;
        if graph.input_channels() != 3 {
            return Err("face detector must take a three-channel input".to_string());
        }
        for level in &policy.levels {
            for slot in [
                level.cls_slot,
                level.obj_slot,
                level.bbox_slot,
                level.kps_slot,
            ] {
                if !graph.has_slot(slot) {
                    return Err(format!(
                        "face detector head slot {slot} is not produced by the graph"
                    ));
                }
            }
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

    pub(crate) fn score_threshold(&self) -> f32 {
        self.policy.score_threshold
    }

    /// Detects faces, best score first. `score_threshold` overrides the model default.
    pub(crate) fn detect(
        &self,
        image: &RgbImage,
        score_threshold: Option<f32>,
    ) -> Result<Vec<FaceDetection>, String> {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err("face detector input image is empty".to_string());
        }
        let threshold = score_threshold.unwrap_or(self.policy.score_threshold);
        if !(0.0..=1.0).contains(&threshold) {
            return Err("face score threshold must be within 0..=1".to_string());
        }
        let longest = width.max(height) as f32;
        let scale = (self.policy.max_side as f32 / longest).min(1.0);
        let resized;
        let network_image = if scale < 1.0 {
            let target_w = ((width as f32 * scale).round() as u32).max(1);
            let target_h = ((height as f32 * scale).round() as u32).max(1);
            resized = image::imageops::resize(image, target_w, target_h, FilterType::Triangle);
            &resized
        } else {
            image
        };
        let (net_w, net_h) = network_image.dimensions();
        let input = self.input_tensor(network_image)?;
        let (padded_h, padded_w) = (input.height, input.width);

        let slots: Vec<usize> = self
            .policy
            .levels
            .iter()
            .flat_map(|l| [l.cls_slot, l.obj_slot, l.bbox_slot, l.kps_slot])
            .collect();
        let outputs = self.graph.run(input, &slots)?;

        let mut candidates = Vec::new();
        for (level, heads) in self.policy.levels.iter().zip(outputs.as_chunks::<4>().0) {
            decode_level(
                level.stride,
                padded_h,
                padded_w,
                heads,
                threshold,
                &mut candidates,
            )?;
        }
        let mut kept = suppress(candidates, self.policy.nms_threshold, self.policy.top_k);

        // Map from network pixels back to the caller's image.
        let sx = width as f32 / net_w as f32;
        let sy = height as f32 / net_h as f32;
        for face in &mut kept {
            face.bbox = [
                face.bbox[0] * sx,
                face.bbox[1] * sy,
                face.bbox[2] * sx,
                face.bbox[3] * sy,
            ];
            for point in &mut face.landmarks {
                *point = [point[0] * sx, point[1] * sy];
            }
        }
        Ok(kept)
    }

    /// Converts to the model's channel order and pads right/bottom up to the alignment.
    fn input_tensor(&self, image: &RgbImage) -> Result<FaceTensor, String> {
        let (width, height) = (image.width() as usize, image.height() as usize);
        let align = self.policy.align;
        let padded_w = width.div_ceil(align) * align;
        let padded_h = height.div_ceil(align) * align;
        let plane = padded_w * padded_h;
        let input = self.policy.input;
        let mut data = vec![input.apply(0); 3 * plane];
        let order = input.channel_order.rgb_indices();
        for (y, row) in image.as_raw().chunks_exact(width * 3).enumerate() {
            for (x, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
                for (channel, source) in order.iter().enumerate() {
                    data[channel * plane + y * padded_w + x] = input.apply(pixel[*source]);
                }
            }
        }
        FaceTensor::new(3, padded_h, padded_w, data)
    }
}

fn validate_policy(policy: &FaceDetectorPolicy) -> Result<(), String> {
    if policy.max_side == 0 || policy.max_side > MAX_DETECTOR_SIDE {
        return Err(format!(
            "face detector max side must be within 1..={MAX_DETECTOR_SIDE}"
        ));
    }
    if policy.levels.is_empty() || policy.levels.len() > MAX_DETECTOR_LEVELS {
        return Err("face detector has an invalid number of head levels".to_string());
    }
    if policy.align == 0
        || policy
            .levels
            .iter()
            .any(|level| level.stride == 0 || !policy.align.is_multiple_of(level.stride))
    {
        return Err("face detector alignment must be a multiple of every stride".to_string());
    }
    if !(0.0..=1.0).contains(&policy.score_threshold)
        || !(0.0..=1.0).contains(&policy.nms_threshold)
    {
        return Err("face detector thresholds must be within 0..=1".to_string());
    }
    if policy.top_k == 0 {
        return Err("face detector top-k must be positive".to_string());
    }
    policy.input.validate()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Decodes one stride. `heads` holds class, objectness, box, and landmark maps in CHW order;
/// the class and objectness maps are logits.
fn decode_level(
    stride: usize,
    padded_h: usize,
    padded_w: usize,
    heads: &[FaceTensor; 4],
    threshold: f32,
    out: &mut Vec<FaceDetection>,
) -> Result<(), String> {
    let (rows, cols) = (padded_h / stride, padded_w / stride);
    let expected = [1, 1, 4, 2 * FACE_LANDMARK_COUNT];
    for (head, channels) in heads.iter().zip(expected) {
        if (head.channels, head.height, head.width) != (channels, rows, cols) {
            return Err(format!(
                "face detector head for stride {stride} has shape {}x{}x{}, expected {channels}x{rows}x{cols}",
                head.channels, head.height, head.width
            ));
        }
    }
    let (cls, obj, bbox, kps) = (&heads[0], &heads[1], &heads[2], &heads[3]);
    let plane = rows * cols;
    let stride = stride as f32;
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            let score = (sigmoid(cls.data[index]) * sigmoid(obj.data[index])).sqrt();
            if score.is_nan() || score <= threshold {
                continue;
            }
            let center_x = (col as f32 + bbox.data[index]) * stride;
            let center_y = (row as f32 + bbox.data[plane + index]) * stride;
            let width = bbox.data[2 * plane + index].exp() * stride;
            let height = bbox.data[3 * plane + index].exp() * stride;
            if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
                continue;
            }
            let mut landmarks = [[0.0f32; 2]; FACE_LANDMARK_COUNT];
            for (point, landmark) in landmarks.iter_mut().enumerate() {
                *landmark = [
                    (kps.data[2 * point * plane + index] + col as f32) * stride,
                    (kps.data[(2 * point + 1) * plane + index] + row as f32) * stride,
                ];
            }
            out.push(FaceDetection {
                bbox: [
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    width,
                    height,
                ],
                landmarks,
                score,
            });
        }
    }
    Ok(())
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    let right = (a[0] + a[2]).min(b[0] + b[2]);
    let bottom = (a[1] + a[3]).min(b[1] + b[3]);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Greedy non-maximum suppression over the `top_k` best candidates.
fn suppress(
    mut candidates: Vec<FaceDetection>,
    nms_threshold: f32,
    top_k: usize,
) -> Vec<FaceDetection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(top_k);
    let mut kept: Vec<FaceDetection> = Vec::new();
    for candidate in candidates {
        if kept
            .iter()
            .all(|face| iou(&face.bbox, &candidate.bbox) <= nms_threshold)
        {
            kept.push(candidate);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::{FaceDetection, decode_level, iou, suppress};
    use crate::engine::face::FACE_LANDMARK_COUNT;
    use crate::engine::face::conv::FaceTensor;

    fn face(bbox: [f32; 4], score: f32) -> FaceDetection {
        FaceDetection {
            bbox,
            landmarks: [[0.0; 2]; FACE_LANDMARK_COUNT],
            score,
        }
    }

    #[test]
    fn iou_of_half_overlapping_boxes() {
        let value = iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 10.0, 10.0]);
        assert!((value - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(iou(&[0.0, 0.0, 1.0, 1.0], &[5.0, 5.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn suppression_keeps_best_of_overlapping_group() {
        let kept = suppress(
            vec![
                face([0.0, 0.0, 10.0, 10.0], 0.8),
                face([1.0, 0.0, 10.0, 10.0], 0.95),
                face([50.0, 50.0, 10.0, 10.0], 0.7),
            ],
            0.3,
            10,
        );
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.95);
        assert_eq!(kept[1].score, 0.7);
    }

    #[test]
    fn decode_places_boxes_and_landmarks_on_the_stride_grid() {
        // 2x2 grid at stride 8; only cell (row 1, col 0) is confident.
        let mut cls = vec![-20.0f32; 4];
        cls[2] = 20.0;
        let obj = vec![20.0f32; 4];
        let mut bbox = vec![0.0f32; 16];
        bbox[2] = 0.5; // dx
        bbox[4 + 2] = 0.25; // dy
        bbox[8 + 2] = 2.0f32.ln(); // width = 2 strides
        bbox[12 + 2] = 0.0; // height = 1 stride
        let mut kps = vec![0.0f32; 40];
        kps[2] = 1.0; // first landmark x offset
        kps[4 + 2] = -1.0; // first landmark y offset
        let heads = [
            FaceTensor::new(1, 2, 2, cls).unwrap(),
            FaceTensor::new(1, 2, 2, obj).unwrap(),
            FaceTensor::new(4, 2, 2, bbox).unwrap(),
            FaceTensor::new(10, 2, 2, kps).unwrap(),
        ];
        let mut out = Vec::new();
        decode_level(8, 16, 16, &heads, 0.5, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        let face = out[0];
        assert!(face.score > 0.99);
        assert_eq!(face.bbox, [4.0 - 8.0, 10.0 - 4.0, 16.0, 8.0]);
        assert_eq!(face.landmarks[0], [8.0, 0.0]);
        assert_eq!(face.landmarks[1], [0.0, 8.0]);
    }

    #[test]
    fn decode_rejects_mismatched_head_shape() {
        let heads = [
            FaceTensor::new(1, 2, 2, vec![0.0; 4]).unwrap(),
            FaceTensor::new(1, 2, 2, vec![0.0; 4]).unwrap(),
            FaceTensor::new(4, 2, 2, vec![0.0; 16]).unwrap(),
            FaceTensor::new(10, 1, 2, vec![0.0; 20]).unwrap(),
        ];
        assert!(decode_level(8, 16, 16, &heads, 0.5, &mut Vec::new()).is_err());
    }
}
