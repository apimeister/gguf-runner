//! `face_sface` GGUF contract: SFace embedding model over landmark-aligned crops.

use std::collections::HashMap;

use super::face_common::{
    graph_policy, input_policy, model_name, required_float, required_string, required_usize,
};
use crate::engine::face::{FACE_LANDMARK_COUNT, FaceEncoderPolicy, lower_face_onnx};
use crate::engine::io::{GgufWriter, OnnxModel, get_gguf_f32_array_from_map};
use crate::engine::types::{GGUFFile, GgufValue};

const PREFIX: &str = "face_sface";

/// The image input and the embedding output of the OpenCV model-zoo ONNX file.
const ONNX_INPUT: &str = "data";
const ONNX_OUTPUT: &str = "fc1";

/// OpenCV `FaceRecognizerSF` alignment targets for a 112x112 crop, in detector landmark
/// order: right eye, left eye, nose tip, right mouth corner, left mouth corner (subject's
/// view).
const ALIGN_REFERENCE: [f32; 2 * FACE_LANDMARK_COUNT] = [
    38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041,
];

/// Converts the OpenCV model-zoo file `face_recognition_sface_2021dec.onnx` to a
/// `face_sface` GGUF. The cosine threshold is the one OpenCV publishes for the model.
pub(super) fn gguf_from_onnx(model: &OnnxModel) -> Result<Vec<u8>, String> {
    let graph = lower_face_onnx(model, ONNX_INPUT)?;
    let output_slot = graph.slot(ONNX_OUTPUT)?;
    let dimension = graph
        .last_output_size()
        .ok_or_else(|| "the ONNX graph does not end in a linear layer".to_string())?;

    let mut writer = GgufWriter::new(PREFIX);
    writer.add_string("general.name", "SFace 2021dec");
    writer.add_string("general.license", "Apache-2.0");
    writer.add_string(
        "general.source.url",
        "https://huggingface.co/opencv/face_recognition_sface",
    );
    graph.write(&mut writer, PREFIX);
    writer.add_string(&format!("{PREFIX}.input.channel_order"), "rgb");
    writer.add_u32(&format!("{PREFIX}.input.size"), 112);
    writer.add_f32_array(&format!("{PREFIX}.align.reference"), &ALIGN_REFERENCE);
    writer.add_u32(
        &format!("{PREFIX}.embedding.output_slot"),
        output_slot as u32,
    );
    writer.add_u32(&format!("{PREFIX}.embedding.dimension"), dimension as u32);
    writer.add_f32(&format!("{PREFIX}.threshold.cosine"), 0.363);
    Ok(writer.finish())
}

pub(super) fn policy(gguf: &GGUFFile) -> Result<FaceEncoderPolicy, String> {
    policy_from_metadata(&gguf.kv)
}

fn policy_from_metadata(kv: &HashMap<String, GgufValue>) -> Result<FaceEncoderPolicy, String> {
    let architecture = required_string(kv, "general.architecture")?;
    if architecture != PREFIX {
        return Err(format!(
            "unsupported face encoder GGUF architecture '{architecture}'; expected '{PREFIX}'"
        ));
    }
    let reference_key = format!("{PREFIX}.align.reference");
    let raw = get_gguf_f32_array_from_map(kv, &reference_key)
        .ok_or_else(|| format!("face model metadata is missing float array '{reference_key}'"))?;
    if raw.len() != 2 * FACE_LANDMARK_COUNT {
        return Err(format!(
            "face model metadata '{reference_key}' must hold {} values",
            2 * FACE_LANDMARK_COUNT
        ));
    }
    let mut reference = [[0.0f32; 2]; FACE_LANDMARK_COUNT];
    for (point, pair) in reference.iter_mut().zip(raw.as_chunks::<2>().0) {
        *point = *pair;
    }
    Ok(FaceEncoderPolicy {
        architecture: architecture.to_string(),
        model_name: model_name(kv, "unnamed SFace face encoder"),
        graph: graph_policy(kv, PREFIX)?,
        input: input_policy(kv, PREFIX)?,
        input_size: required_usize(kv, &format!("{PREFIX}.input.size"))?,
        reference,
        output_slot: required_usize(kv, &format!("{PREFIX}.embedding.output_slot"))?,
        embedding_dim: required_usize(kv, &format!("{PREFIX}.embedding.dimension"))?,
        cosine_threshold: required_float(kv, &format!("{PREFIX}.threshold.cosine"))?,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::engine::types::GgufValue;

    use super::policy_from_metadata;

    fn metadata() -> HashMap<String, GgufValue> {
        let text = |value: &str| GgufValue::Str(value.to_string());
        HashMap::from(
            [
                ("general.architecture", text("face_sface")),
                (
                    "face_sface.graph.ops",
                    GgufValue::I64Array(vec![0, 0, 0, 1, 1, 1, 1, 2]),
                ),
                ("face_sface.graph.slot_count", GgufValue::UInt(2)),
                ("face_sface.input.channel_order", text("rgb")),
                ("face_sface.input.mean", GgufValue::F32(127.5)),
                ("face_sface.input.scale", GgufValue::F32(0.007_812_5)),
                ("face_sface.input.size", GgufValue::UInt(112)),
                (
                    "face_sface.align.reference",
                    GgufValue::F32Array((0..10).map(|v| v as f32).collect()),
                ),
                ("face_sface.embedding.output_slot", GgufValue::UInt(1)),
                ("face_sface.embedding.dimension", GgufValue::UInt(128)),
                ("face_sface.threshold.cosine", GgufValue::F32(0.363)),
            ]
            .map(|(key, value)| (key.to_string(), value)),
        )
    }

    #[test]
    fn reads_reference_points_and_dimensions() {
        let policy = policy_from_metadata(&metadata()).unwrap();
        assert_eq!(policy.model_name, "unnamed SFace face encoder");
        assert_eq!(policy.reference[2], [4.0, 5.0]);
        assert_eq!(policy.embedding_dim, 128);
        assert_eq!(policy.input.mean, 127.5);
    }

    #[test]
    fn rejects_wrong_architecture_and_short_reference() {
        let mut kv = metadata();
        kv.insert(
            "general.architecture".to_string(),
            GgufValue::Str("face_yunet".to_string()),
        );
        assert!(policy_from_metadata(&kv).is_err());
        let mut kv = metadata();
        kv.insert(
            "face_sface.align.reference".to_string(),
            GgufValue::F32Array(vec![0.0; 8]),
        );
        assert!(policy_from_metadata(&kv).is_err());
    }
}
