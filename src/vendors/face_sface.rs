//! `face_sface` GGUF contract: SFace embedding model over landmark-aligned crops.

use std::collections::HashMap;

use super::face_common::{
    graph_policy, input_policy, model_name, required_float, required_string, required_usize,
};
use crate::engine::face::{FACE_LANDMARK_COUNT, FaceEncoderPolicy};
use crate::engine::io::get_gguf_f32_array_from_map;
use crate::engine::types::{GGUFFile, GgufValue};

const PREFIX: &str = "face_sface";

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
