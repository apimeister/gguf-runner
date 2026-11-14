//! Metadata helpers shared by the face GGUF contracts: the flat operation list and the input
//! normalization keys.

use std::collections::HashMap;

use crate::engine::face::{
    FACE_GRAPH_INPUT_SLOT, FaceActivation, FaceChannelOrder, FaceGraphOpPolicy, FaceGraphPolicy,
    FaceInputPolicy,
};
use crate::engine::io::{
    get_gguf_float_from_map, get_gguf_i64_array_from_map, get_gguf_int_from_map,
    get_gguf_string_from_map,
};
use crate::engine::types::GgufValue;

/// Integers per operation record: kind, input a, input b, output, stride, pad, groups,
/// activation.
const OP_WIDTH: usize = 8;
const MAX_GRAPH_OPS: usize = 1_024;
const MAX_GRAPH_SLOTS: usize = 2_048;
const MAX_CONV_STRIDE: usize = 8;
const MAX_CONV_PAD: usize = 8;
const MAX_CONV_GROUPS: usize = 65_536;

pub(super) fn required_string<'a>(
    kv: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str, String> {
    get_gguf_string_from_map(kv, key)
        .ok_or_else(|| format!("face model metadata is missing string '{key}'"))
}

pub(super) fn required_usize(kv: &HashMap<String, GgufValue>, key: &str) -> Result<usize, String> {
    if !kv.contains_key(key) {
        return Err(format!("face model metadata is missing integer '{key}'"));
    }
    let value = get_gguf_int_from_map(kv, key, -1);
    if value <= 0 {
        return Err(format!(
            "face model metadata '{key}' must be a positive integer"
        ));
    }
    usize::try_from(value)
        .map_err(|_| format!("face model metadata '{key}' does not fit this platform"))
}

pub(super) fn required_float(kv: &HashMap<String, GgufValue>, key: &str) -> Result<f32, String> {
    if !kv.contains_key(key) {
        return Err(format!("face model metadata is missing float '{key}'"));
    }
    let value = get_gguf_float_from_map(kv, key, f32::NAN);
    if !value.is_finite() {
        return Err(format!(
            "face model metadata '{key}' must be a finite float"
        ));
    }
    Ok(value)
}

/// A non-negative integer array of exactly `expected` entries.
pub(super) fn required_usize_array(
    kv: &HashMap<String, GgufValue>,
    key: &str,
    expected: usize,
) -> Result<Vec<usize>, String> {
    let raw = get_gguf_i64_array_from_map(kv, key)
        .ok_or_else(|| format!("face model metadata is missing integer array '{key}'"))?;
    if raw.len() != expected {
        return Err(format!(
            "face model metadata '{key}' has {} entries, expected {expected}",
            raw.len()
        ));
    }
    raw.iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| format!("face model metadata '{key}' holds invalid value {value}"))
        })
        .collect()
}

pub(super) fn model_name(kv: &HashMap<String, GgufValue>, fallback: &str) -> String {
    get_gguf_string_from_map(kv, "general.name")
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

pub(super) fn input_policy(
    kv: &HashMap<String, GgufValue>,
    prefix: &str,
) -> Result<FaceInputPolicy, String> {
    let order_key = format!("{prefix}.input.channel_order");
    let channel_order = match required_string(kv, &order_key)? {
        "rgb" => FaceChannelOrder::Rgb,
        "bgr" => FaceChannelOrder::Bgr,
        other => {
            return Err(format!(
                "face model metadata '{order_key}' must be 'rgb' or 'bgr', got '{other}'"
            ));
        }
    };
    Ok(FaceInputPolicy {
        channel_order,
        mean: required_float(kv, &format!("{prefix}.input.mean"))?,
        scale: required_float(kv, &format!("{prefix}.input.scale"))?,
    })
}

/// Reads `<prefix>.graph.ops`. Convolution and linear tensors are numbered in operation
/// order: `face.conv.N.{weight,bias,slope}` and `face.linear.N.{weight,bias}`.
pub(super) fn graph_policy(
    kv: &HashMap<String, GgufValue>,
    prefix: &str,
) -> Result<FaceGraphPolicy, String> {
    let slot_count = required_usize(kv, &format!("{prefix}.graph.slot_count"))?;
    if slot_count > MAX_GRAPH_SLOTS {
        return Err(format!(
            "face graph slot count {slot_count} exceeds {MAX_GRAPH_SLOTS}"
        ));
    }
    let ops_key = format!("{prefix}.graph.ops");
    let raw = get_gguf_i64_array_from_map(kv, &ops_key)
        .ok_or_else(|| format!("face model metadata is missing integer array '{ops_key}'"))?;
    if raw.is_empty() || !raw.len().is_multiple_of(OP_WIDTH) || raw.len() / OP_WIDTH > MAX_GRAPH_OPS
    {
        return Err(format!(
            "face model metadata '{ops_key}' must hold 1..={MAX_GRAPH_OPS} records of {OP_WIDTH} integers"
        ));
    }
    let mut ops = Vec::with_capacity(raw.len() / OP_WIDTH);
    let (mut conv_index, mut linear_index) = (0usize, 0usize);
    for (index, record) in raw.as_chunks::<OP_WIDTH>().0.iter().enumerate() {
        let field = |position: usize, name: &str, max: usize| -> Result<usize, String> {
            usize::try_from(record[position])
                .ok()
                .filter(|value| *value <= max)
                .ok_or_else(|| {
                    format!(
                        "face graph operation {index} has invalid {name} {}",
                        record[position]
                    )
                })
        };
        let slot = |position: usize, name: &str| field(position, name, slot_count - 1);
        let input = slot(1, "input slot")?;
        let output = slot(3, "output slot")?;
        if output == FACE_GRAPH_INPUT_SLOT {
            return Err(format!(
                "face graph operation {index} overwrites the input slot"
            ));
        }
        ops.push(match record[0] {
            0 => {
                let activation = match record[7] {
                    0 => FaceActivation::None,
                    1 => FaceActivation::Relu,
                    2 => FaceActivation::PRelu,
                    other => {
                        return Err(format!(
                            "face graph operation {index} has unknown activation {other}"
                        ));
                    }
                };
                let base = format!("face.conv.{conv_index}");
                conv_index += 1;
                let (stride, groups) = (
                    field(4, "stride", MAX_CONV_STRIDE)?,
                    field(6, "group count", MAX_CONV_GROUPS)?,
                );
                if stride == 0 || groups == 0 {
                    return Err(format!(
                        "face graph operation {index} has a zero stride or group count"
                    ));
                }
                FaceGraphOpPolicy::Conv {
                    input,
                    output,
                    stride,
                    pad: field(5, "padding", MAX_CONV_PAD)?,
                    groups,
                    activation,
                    weight_name: format!("{base}.weight"),
                    bias_name: format!("{base}.bias"),
                    slope_name: format!("{base}.slope"),
                }
            }
            1 => FaceGraphOpPolicy::MaxPool { input, output },
            2 => FaceGraphOpPolicy::Upsample { input, output },
            3 => FaceGraphOpPolicy::Add {
                left: input,
                right: slot(2, "second input slot")?,
                output,
            },
            4 => {
                let base = format!("face.linear.{linear_index}");
                linear_index += 1;
                FaceGraphOpPolicy::Linear {
                    input,
                    output,
                    weight_name: format!("{base}.weight"),
                    bias_name: format!("{base}.bias"),
                }
            }
            other => {
                return Err(format!(
                    "face graph operation {index} has unknown kind {other}"
                ));
            }
        });
    }
    Ok(FaceGraphPolicy { slot_count, ops })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::engine::face::{FaceActivation, FaceGraphOpPolicy};
    use crate::engine::types::GgufValue;

    use super::graph_policy;

    fn metadata(ops: Vec<i64>, slots: u64) -> HashMap<String, GgufValue> {
        HashMap::from([
            ("p.graph.ops".to_string(), GgufValue::I64Array(ops)),
            ("p.graph.slot_count".to_string(), GgufValue::UInt(slots)),
        ])
    }

    #[test]
    fn parses_operation_records_and_numbers_tensors() {
        let kv = metadata(
            vec![
                0, 0, 0, 1, 2, 1, 1, 1, // conv 0, stride 2, relu
                1, 1, 0, 2, 0, 0, 0, 0, // max pool
                0, 2, 0, 3, 1, 1, 16, 2, // conv 1, depthwise, prelu
                3, 3, 2, 4, 0, 0, 0, 0, // add
                4, 4, 0, 5, 0, 0, 0, 0, // linear 0
            ],
            6,
        );
        let graph = graph_policy(&kv, "p").unwrap();
        assert_eq!(graph.ops.len(), 5);
        assert!(matches!(
            &graph.ops[2],
            FaceGraphOpPolicy::Conv { groups: 16, activation: FaceActivation::PRelu, weight_name, .. }
                if weight_name == "face.conv.1.weight"
        ));
        assert_eq!(
            graph.ops[3],
            FaceGraphOpPolicy::Add {
                left: 3,
                right: 2,
                output: 4
            }
        );
        assert!(matches!(
            &graph.ops[4],
            FaceGraphOpPolicy::Linear { bias_name, .. } if bias_name == "face.linear.0.bias"
        ));
    }

    #[test]
    fn rejects_malformed_records() {
        assert!(graph_policy(&metadata(vec![0, 0, 0, 1], 2), "p").is_err());
        assert!(graph_policy(&metadata(vec![9, 0, 0, 1, 0, 0, 0, 0], 2), "p").is_err());
        assert!(graph_policy(&metadata(vec![1, 0, 0, 5, 0, 0, 0, 0], 2), "p").is_err());
        assert!(graph_policy(&metadata(vec![1, 0, 0, 0, 0, 0, 0, 0], 2), "p").is_err());
        assert!(graph_policy(&metadata(vec![0, 0, 0, 1, 0, 1, 1, 0], 2), "p").is_err());
    }
}
