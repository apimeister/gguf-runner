//! `face_yunet` GGUF contract: YuNet anchor-free face detector with five landmarks.

use std::collections::HashMap;

use super::face_common::{
    graph_policy, input_policy, model_name, required_float, required_string, required_usize,
    required_usize_array,
};
use crate::engine::face::{FACE_LANDMARK_COUNT, FaceDetectLevelPolicy, FaceDetectorPolicy};
use crate::engine::io::get_gguf_i64_array_from_map;
use crate::engine::types::{GGUFFile, GgufValue};

const PREFIX: &str = "face_yunet";

pub(super) fn policy(gguf: &GGUFFile) -> Result<FaceDetectorPolicy, String> {
    policy_from_metadata(&gguf.kv)
}

fn policy_from_metadata(kv: &HashMap<String, GgufValue>) -> Result<FaceDetectorPolicy, String> {
    let architecture = required_string(kv, "general.architecture")?;
    if architecture != PREFIX {
        return Err(format!(
            "unsupported face detector GGUF architecture '{architecture}'; expected '{PREFIX}'"
        ));
    }
    let landmark_count = required_usize(kv, &format!("{PREFIX}.detect.landmark_count"))?;
    if landmark_count != FACE_LANDMARK_COUNT {
        return Err(format!(
            "face detector reports {landmark_count} landmarks, expected {FACE_LANDMARK_COUNT}"
        ));
    }
    let strides_key = format!("{PREFIX}.detect.strides");
    let level_count = get_gguf_i64_array_from_map(kv, &strides_key)
        .ok_or_else(|| format!("face model metadata is missing integer array '{strides_key}'"))?
        .len();
    let strides = required_usize_array(kv, &strides_key, level_count)?;
    let head = |name: &str| {
        required_usize_array(kv, &format!("{PREFIX}.detect.{name}_slots"), level_count)
    };
    let (cls, obj, bbox, kps) = (head("cls")?, head("obj")?, head("bbox")?, head("kps")?);
    let levels = (0..level_count)
        .map(|index| FaceDetectLevelPolicy {
            stride: strides[index],
            cls_slot: cls[index],
            obj_slot: obj[index],
            bbox_slot: bbox[index],
            kps_slot: kps[index],
        })
        .collect();
    Ok(FaceDetectorPolicy {
        architecture: architecture.to_string(),
        model_name: model_name(kv, "unnamed YuNet face detector"),
        graph: graph_policy(kv, PREFIX)?,
        input: input_policy(kv, PREFIX)?,
        max_side: required_usize(kv, &format!("{PREFIX}.input.max_side"))?,
        align: required_usize(kv, &format!("{PREFIX}.input.align"))?,
        levels,
        score_threshold: required_float(kv, &format!("{PREFIX}.detect.score_threshold"))?,
        nms_threshold: required_float(kv, &format!("{PREFIX}.detect.nms_threshold"))?,
        top_k: required_usize(kv, &format!("{PREFIX}.detect.top_k"))?,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::engine::types::GgufValue;

    use super::policy_from_metadata;

    fn metadata() -> HashMap<String, GgufValue> {
        let ints = |values: &[i64]| GgufValue::I64Array(values.to_vec());
        let text = |value: &str| GgufValue::Str(value.to_string());
        HashMap::from(
            [
                ("general.architecture", text("face_yunet")),
                ("general.name", text("test detector")),
                ("face_yunet.graph.ops", ints(&[0, 0, 0, 1, 1, 1, 1, 0])),
                ("face_yunet.graph.slot_count", GgufValue::UInt(2)),
                ("face_yunet.input.channel_order", text("bgr")),
                ("face_yunet.input.mean", GgufValue::F32(0.0)),
                ("face_yunet.input.scale", GgufValue::F32(1.0)),
                ("face_yunet.input.max_side", GgufValue::UInt(640)),
                ("face_yunet.input.align", GgufValue::UInt(32)),
                ("face_yunet.detect.strides", ints(&[8, 16])),
                ("face_yunet.detect.cls_slots", ints(&[1, 1])),
                ("face_yunet.detect.obj_slots", ints(&[1, 1])),
                ("face_yunet.detect.bbox_slots", ints(&[1, 1])),
                ("face_yunet.detect.kps_slots", ints(&[1, 1])),
                ("face_yunet.detect.landmark_count", GgufValue::UInt(5)),
                ("face_yunet.detect.score_threshold", GgufValue::F32(0.9)),
                ("face_yunet.detect.nms_threshold", GgufValue::F32(0.3)),
                ("face_yunet.detect.top_k", GgufValue::UInt(5000)),
            ]
            .map(|(key, value)| (key.to_string(), value)),
        )
    }

    #[test]
    fn reads_levels_and_thresholds() {
        let policy = policy_from_metadata(&metadata()).unwrap();
        assert_eq!(policy.model_name, "test detector");
        assert_eq!(policy.levels.len(), 2);
        assert_eq!(policy.levels[1].stride, 16);
        assert_eq!(policy.max_side, 640);
        assert_eq!(policy.score_threshold, 0.9);
    }

    #[test]
    fn rejects_mismatched_head_arrays_and_landmark_count() {
        let mut kv = metadata();
        kv.insert(
            "face_yunet.detect.kps_slots".to_string(),
            GgufValue::I64Array(vec![1]),
        );
        assert!(policy_from_metadata(&kv).is_err());
        let mut kv = metadata();
        kv.insert(
            "face_yunet.detect.landmark_count".to_string(),
            GgufValue::UInt(68),
        );
        assert!(policy_from_metadata(&kv).is_err());
    }
}
