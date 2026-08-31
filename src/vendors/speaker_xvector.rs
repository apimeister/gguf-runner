use std::collections::HashMap;

use crate::engine::io::{
    get_gguf_float_from_map, get_gguf_i64_array_from_map, get_gguf_int_from_map,
    get_gguf_string_from_map,
};
use crate::engine::speaker::{
    SpeakerModelPolicy, SpeakerSegmentLayerPolicy, SpeakerTdnnLayerPolicy, SpeakerThresholdPolicy,
};
use crate::engine::types::{GGUFFile, GgufValue};

const PREFIX: &str = "speaker_xvector";
const MAX_LAYER_COUNT: usize = 16;
const MAX_CONTEXT_COUNT: usize = 33;
const MAX_CONTEXT_OFFSET: i64 = 64;

pub(super) fn policy(gguf: &GGUFFile) -> Result<SpeakerModelPolicy, String> {
    policy_from_metadata(&gguf.kv)
}

fn policy_from_metadata(kv: &HashMap<String, GgufValue>) -> Result<SpeakerModelPolicy, String> {
    let architecture = required_string(kv, "general.architecture")?;
    if architecture != PREFIX {
        return Err(format!(
            "unsupported speaker GGUF architecture '{architecture}'; expected '{PREFIX}'"
        ));
    }
    let model_name = get_gguf_string_from_map(kv, "general.name")
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("unnamed speaker x-vector")
        .to_string();
    let sample_rate = required_usize(kv, &format!("{PREFIX}.audio.sample_rate"))?;
    let fft_length = required_usize(kv, &format!("{PREFIX}.audio.fft_length"))?;
    let window_length = required_usize(kv, &format!("{PREFIX}.audio.window_length"))?;
    let hop_length = required_usize(kv, &format!("{PREFIX}.audio.hop_length"))?;
    let mel_bins = required_usize(kv, &format!("{PREFIX}.audio.mel_bins"))?;
    let max_window_frames = required_usize(kv, &format!("{PREFIX}.audio.max_window_frames"))?;
    let mel_floor = required_float(kv, &format!("{PREFIX}.audio.mel_floor"))?;
    let min_audio_seconds = required_float(kv, &format!("{PREFIX}.audio.min_seconds"))?;
    let max_audio_seconds = required_float(kv, &format!("{PREFIX}.audio.max_seconds"))?;
    let tdnn_layer_count = bounded_layer_count(kv, &format!("{PREFIX}.tdnn.layer_count"), "TDNN")?;
    let segment_layer_count =
        bounded_layer_count(kv, &format!("{PREFIX}.segment.layer_count"), "segment")?;

    let mut tdnn_layers = Vec::with_capacity(tdnn_layer_count);
    for index in 0..tdnn_layer_count {
        let context_key = format!("{PREFIX}.tdnn.{index}.context");
        let raw_contexts = get_gguf_i64_array_from_map(kv, &context_key)
            .ok_or_else(|| format!("speaker model metadata is missing '{context_key}'"))?;
        if raw_contexts.is_empty() || raw_contexts.len() > MAX_CONTEXT_COUNT {
            return Err(format!(
                "speaker model '{context_key}' must contain 1..={MAX_CONTEXT_COUNT} offsets"
            ));
        }
        let contexts = raw_contexts
            .iter()
            .map(|offset| {
                if offset.abs() > MAX_CONTEXT_OFFSET {
                    return Err(format!(
                        "speaker model '{context_key}' offset {offset} exceeds +/-{MAX_CONTEXT_OFFSET}"
                    ));
                }
                isize::try_from(*offset).map_err(|_| {
                    format!("speaker model '{context_key}' offset {offset} does not fit this platform")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        tdnn_layers.push(SpeakerTdnnLayerPolicy {
            weight_name: format!("speaker.tdnn.{index}.weight"),
            bias_name: format!("speaker.tdnn.{index}.bias"),
            contexts,
        });
    }

    let segment_layers = (0..segment_layer_count)
        .map(|index| SpeakerSegmentLayerPolicy {
            weight_name: format!("speaker.segment.{index}.weight"),
            bias_name: format!("speaker.segment.{index}.bias"),
            apply_relu: index + 1 < segment_layer_count,
        })
        .collect();

    let sample_rate = u32::try_from(sample_rate)
        .map_err(|_| "speaker model sample rate exceeds u32".to_string())?;
    Ok(SpeakerModelPolicy {
        architecture: architecture.to_string(),
        model_name,
        sample_rate,
        fft_length,
        window_length,
        hop_length,
        mel_bins,
        mel_floor,
        max_window_frames,
        min_audio_seconds,
        max_audio_seconds,
        tdnn_layers,
        segment_layers,
        thresholds: SpeakerThresholdPolicy {
            verification: required_float(kv, &format!("{PREFIX}.threshold.verification"))?,
            identification_margin: required_float(
                kv,
                &format!("{PREFIX}.threshold.identification_margin"),
            )?,
            enrollment: required_float(kv, &format!("{PREFIX}.threshold.enrollment"))?,
            auto_learning: required_float(kv, &format!("{PREFIX}.threshold.auto_learning"))?,
            diarization_cluster: required_float(
                kv,
                &format!("{PREFIX}.threshold.diarization_cluster"),
            )?,
        },
    })
}

fn required_string<'a>(kv: &'a HashMap<String, GgufValue>, key: &str) -> Result<&'a str, String> {
    get_gguf_string_from_map(kv, key)
        .ok_or_else(|| format!("speaker model metadata is missing string '{key}'"))
}

fn required_usize(kv: &HashMap<String, GgufValue>, key: &str) -> Result<usize, String> {
    if !kv.contains_key(key) {
        return Err(format!("speaker model metadata is missing integer '{key}'"));
    }
    let value = get_gguf_int_from_map(kv, key, -1);
    if value <= 0 {
        return Err(format!(
            "speaker model metadata '{key}' must be a positive integer"
        ));
    }
    usize::try_from(value)
        .map_err(|_| format!("speaker model metadata '{key}' does not fit this platform"))
}

fn required_float(kv: &HashMap<String, GgufValue>, key: &str) -> Result<f32, String> {
    if !kv.contains_key(key) {
        return Err(format!("speaker model metadata is missing float '{key}'"));
    }
    let value = get_gguf_float_from_map(kv, key, f32::NAN);
    if !value.is_finite() {
        return Err(format!(
            "speaker model metadata '{key}' must be a finite float"
        ));
    }
    Ok(value)
}

fn bounded_layer_count(
    kv: &HashMap<String, GgufValue>,
    key: &str,
    kind: &str,
) -> Result<usize, String> {
    let count = required_usize(kv, key)?;
    if count > MAX_LAYER_COUNT {
        return Err(format!(
            "speaker model {kind} layer count {count} exceeds {MAX_LAYER_COUNT}"
        ));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::engine::types::GgufValue;

    use super::policy_from_metadata;

    fn metadata() -> HashMap<String, GgufValue> {
        let mut kv = HashMap::new();
        kv.insert(
            "general.architecture".to_string(),
            GgufValue::Str("speaker_xvector".to_string()),
        );
        for (key, value) in [
            ("speaker_xvector.audio.sample_rate", 16_000),
            ("speaker_xvector.audio.fft_length", 400),
            ("speaker_xvector.audio.window_length", 400),
            ("speaker_xvector.audio.hop_length", 160),
            ("speaker_xvector.audio.mel_bins", 80),
            ("speaker_xvector.audio.max_window_frames", 1_500),
            ("speaker_xvector.tdnn.layer_count", 2),
            ("speaker_xvector.segment.layer_count", 2),
        ] {
            kv.insert(key.to_string(), GgufValue::Int(value));
        }
        for (key, value) in [
            ("speaker_xvector.audio.mel_floor", 1e-8),
            ("speaker_xvector.audio.min_seconds", 1.0),
            ("speaker_xvector.audio.max_seconds", 3_600.0),
            ("speaker_xvector.threshold.verification", 0.6),
            ("speaker_xvector.threshold.identification_margin", 0.05),
            ("speaker_xvector.threshold.enrollment", 0.45),
            ("speaker_xvector.threshold.auto_learning", 0.8),
            ("speaker_xvector.threshold.diarization_cluster", 0.55),
        ] {
            kv.insert(key.to_string(), GgufValue::F32(value));
        }
        kv.insert(
            "speaker_xvector.tdnn.0.context".to_string(),
            GgufValue::I64Array(vec![-2, -1, 0, 1, 2]),
        );
        kv.insert(
            "speaker_xvector.tdnn.1.context".to_string(),
            GgufValue::I64Array(vec![-2, 0, 2]),
        );
        kv
    }

    #[test]
    fn parses_xvector_policy_and_tensor_names() {
        let policy = policy_from_metadata(&metadata()).unwrap();
        assert_eq!(policy.sample_rate, 16_000);
        assert_eq!(policy.tdnn_layers[0].contexts, [-2, -1, 0, 1, 2]);
        assert_eq!(policy.tdnn_layers[1].weight_name, "speaker.tdnn.1.weight");
        assert!(policy.segment_layers[0].apply_relu);
        assert!(!policy.segment_layers[1].apply_relu);
    }

    #[test]
    fn requires_model_calibrated_thresholds() {
        let mut kv = metadata();
        kv.remove("speaker_xvector.threshold.verification");
        let error = policy_from_metadata(&kv).unwrap_err();
        assert!(error.contains("threshold.verification"));
    }
}
