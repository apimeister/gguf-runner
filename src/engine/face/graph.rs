//! Operation-list executor shared by the face detector and the face encoder.

use crate::engine::face::conv::{
    FaceActivation, FaceConvLayer, FaceLinearLayer, FaceTensor, add, conv2d, linear, max_pool2,
    upsample2,
};
use crate::engine::io::find_gguf_tensor;
use crate::engine::kernels::{dequantize_tensor, get_block_size, get_type_size};
use crate::engine::types::{GGUFFile, Gguftensor};

/// Slot 0 always holds the network input.
pub(crate) const FACE_GRAPH_INPUT_SLOT: usize = 0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FaceGraphOpPolicy {
    Conv {
        input: usize,
        output: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        activation: FaceActivation,
        weight_name: String,
        bias_name: String,
        slope_name: String,
    },
    MaxPool {
        input: usize,
        output: usize,
    },
    Upsample {
        input: usize,
        output: usize,
    },
    Add {
        left: usize,
        right: usize,
        output: usize,
    },
    Linear {
        input: usize,
        output: usize,
        weight_name: String,
        bias_name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FaceGraphPolicy {
    pub(crate) slot_count: usize,
    pub(crate) ops: Vec<FaceGraphOpPolicy>,
}

enum Op {
    Conv {
        input: usize,
        output: usize,
        layer: FaceConvLayer,
    },
    MaxPool {
        input: usize,
        output: usize,
    },
    Upsample {
        input: usize,
        output: usize,
    },
    Add {
        left: usize,
        right: usize,
        output: usize,
    },
    Linear {
        input: usize,
        output: usize,
        layer: FaceLinearLayer,
    },
}

impl Op {
    fn inputs(&self) -> [usize; 2] {
        match self {
            Op::Conv { input, .. }
            | Op::MaxPool { input, .. }
            | Op::Upsample { input, .. }
            | Op::Linear { input, .. } => [*input, *input],
            Op::Add { left, right, .. } => [*left, *right],
        }
    }

    fn output(&self) -> usize {
        match self {
            Op::Conv { output, .. }
            | Op::MaxPool { output, .. }
            | Op::Upsample { output, .. }
            | Op::Add { output, .. }
            | Op::Linear { output, .. } => *output,
        }
    }
}

pub(crate) struct FaceGraph {
    slot_count: usize,
    ops: Vec<Op>,
    /// Index of the last operation reading each slot; intermediate maps are freed after it.
    last_use: Vec<usize>,
    input_channels: usize,
}

impl FaceGraph {
    pub(crate) fn load(gguf: &GGUFFile, policy: &FaceGraphPolicy) -> Result<Self, String> {
        if policy.slot_count < 2 || policy.ops.is_empty() {
            return Err("face graph is empty".to_string());
        }
        let mut ops = Vec::with_capacity(policy.ops.len());
        for op in &policy.ops {
            ops.push(match op {
                FaceGraphOpPolicy::Conv {
                    input,
                    output,
                    stride,
                    pad,
                    groups,
                    activation,
                    weight_name,
                    bias_name,
                    slope_name,
                } => {
                    let tensor = required_tensor(gguf, weight_name)?;
                    if tensor.n_dims != 4 || tensor.ne[0] != tensor.ne[1] {
                        return Err(format!(
                            "face tensor {weight_name} is not a square 4-D convolution kernel"
                        ));
                    }
                    let out_channels = dimension(tensor, 3)?;
                    let layer = FaceConvLayer {
                        out_channels,
                        in_per_group: dimension(tensor, 2)?,
                        kernel: dimension(tensor, 0)?,
                        stride: *stride,
                        pad: *pad,
                        groups: *groups,
                        activation: *activation,
                        weight: load_float_tensor(gguf, weight_name, None)?,
                        bias: load_float_tensor(gguf, bias_name, Some(out_channels))?,
                        slope: match activation {
                            FaceActivation::PRelu => {
                                load_float_tensor(gguf, slope_name, Some(out_channels))?
                            }
                            _ => Vec::new(),
                        },
                    };
                    layer
                        .validate()
                        .map_err(|error| format!("{weight_name}: {error}"))?;
                    Op::Conv {
                        input: *input,
                        output: *output,
                        layer,
                    }
                }
                FaceGraphOpPolicy::MaxPool { input, output } => Op::MaxPool {
                    input: *input,
                    output: *output,
                },
                FaceGraphOpPolicy::Upsample { input, output } => Op::Upsample {
                    input: *input,
                    output: *output,
                },
                FaceGraphOpPolicy::Add {
                    left,
                    right,
                    output,
                } => Op::Add {
                    left: *left,
                    right: *right,
                    output: *output,
                },
                FaceGraphOpPolicy::Linear {
                    input,
                    output,
                    weight_name,
                    bias_name,
                } => {
                    let tensor = required_tensor(gguf, weight_name)?;
                    if tensor.n_dims != 2 {
                        return Err(format!("face tensor {weight_name} is not a matrix"));
                    }
                    let outputs = dimension(tensor, 1)?;
                    Op::Linear {
                        input: *input,
                        output: *output,
                        layer: FaceLinearLayer {
                            outputs,
                            inputs: dimension(tensor, 0)?,
                            weight: load_float_tensor(gguf, weight_name, None)?,
                            bias: load_float_tensor(gguf, bias_name, Some(outputs))?,
                        },
                    }
                }
            });
        }

        // Every operation reads slots written earlier and writes a fresh one.
        let mut written = vec![false; policy.slot_count];
        written[FACE_GRAPH_INPUT_SLOT] = true;
        let mut last_use = vec![0usize; policy.slot_count];
        for (index, op) in ops.iter().enumerate() {
            for slot in op.inputs() {
                if !written.get(slot).copied().unwrap_or(false) {
                    return Err(format!(
                        "face graph operation {index} reads unwritten slot {slot}"
                    ));
                }
                last_use[slot] = index;
            }
            let output = op.output();
            if written.get(output).copied().unwrap_or(true) {
                return Err(format!(
                    "face graph operation {index} writes invalid or reused slot {output}"
                ));
            }
            written[output] = true;
        }
        let input_channels = ops
            .iter()
            .find_map(|op| match op {
                Op::Conv { input, layer, .. } if *input == FACE_GRAPH_INPUT_SLOT => {
                    Some(layer.in_per_group * layer.groups)
                }
                _ => None,
            })
            .ok_or_else(|| "face graph input must feed a convolution".to_string())?;
        Ok(Self {
            slot_count: policy.slot_count,
            ops,
            last_use,
            input_channels,
        })
    }

    pub(crate) fn input_channels(&self) -> usize {
        self.input_channels
    }

    pub(crate) fn has_slot(&self, slot: usize) -> bool {
        slot != FACE_GRAPH_INPUT_SLOT
            && slot < self.slot_count
            && self.ops.iter().any(|op| op.output() == slot)
    }

    /// Runs the network and returns the requested slots in `keep` order.
    pub(crate) fn run(&self, input: FaceTensor, keep: &[usize]) -> Result<Vec<FaceTensor>, String> {
        let mut slots: Vec<Option<FaceTensor>> = (0..self.slot_count).map(|_| None).collect();
        slots[FACE_GRAPH_INPUT_SLOT] = Some(input);
        for (index, op) in self.ops.iter().enumerate() {
            let get = |slot: usize| {
                slots[slot]
                    .as_ref()
                    .ok_or_else(|| format!("face graph slot {slot} was released too early"))
            };
            let result = match op {
                Op::Conv { input, layer, .. } => conv2d(get(*input)?, layer),
                Op::MaxPool { input, .. } => max_pool2(get(*input)?),
                Op::Upsample { input, .. } => upsample2(get(*input)?),
                Op::Add { left, right, .. } => add(get(*left)?, get(*right)?),
                Op::Linear { input, layer, .. } => linear(get(*input)?, layer),
            }
            .map_err(|error| format!("face graph operation {index}: {error}"))?;
            slots[op.output()] = Some(result);
            for slot in op.inputs() {
                if self.last_use[slot] == index && !keep.contains(&slot) {
                    slots[slot] = None;
                }
            }
        }
        keep.iter()
            .map(|slot| {
                slots
                    .get_mut(*slot)
                    .and_then(Option::take)
                    .ok_or_else(|| format!("face graph did not produce slot {slot}"))
            })
            .collect()
    }
}

fn required_tensor<'a>(gguf: &'a GGUFFile, name: &str) -> Result<&'a Gguftensor, String> {
    find_gguf_tensor(gguf, name).ok_or_else(|| format!("face tensor not found: {name}"))
}

fn dimension(tensor: &Gguftensor, axis: usize) -> Result<usize, String> {
    usize::try_from(tensor.ne[axis])
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("face tensor {} has an invalid dimension", tensor.name))
}

fn load_float_tensor(
    gguf: &GGUFFile,
    name: &str,
    expected_elements: Option<usize>,
) -> Result<Vec<f32>, String> {
    let tensor = required_tensor(gguf, name)?;
    let elements = (0..tensor.n_dims as usize).try_fold(1usize, |total, axis| {
        total
            .checked_mul(dimension(tensor, axis)?)
            .ok_or_else(|| format!("face tensor {name} element count overflow"))
    })?;
    if let Some(expected) = expected_elements
        && elements != expected
    {
        return Err(format!(
            "face tensor {name} has {elements} elements, expected {expected}"
        ));
    }
    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if block_size == 0 || type_size == 0 || !elements.is_multiple_of(block_size) {
        return Err(format!(
            "face tensor {name} uses unsupported or invalid GGML type {}",
            tensor.ttype.0
        ));
    }
    let byte_len = elements / block_size * type_size;
    let end = tensor
        .data_offset
        .checked_add(byte_len)
        .ok_or_else(|| format!("face tensor {name} data range overflow"))?;
    let mapped = gguf.mapped.as_slice();
    if end > mapped.len() {
        return Err(format!("face tensor {name} exceeds mapped model data"));
    }
    dequantize_tensor(&mapped[tensor.data_offset..end], elements, tensor.ttype)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{FaceGraph, FaceGraphOpPolicy, FaceGraphPolicy};
    use crate::engine::face::conv::{FaceActivation, FaceTensor};
    use crate::engine::types::{GGML_TYPE_F32, GGUFFile, GgmlType, Gguftensor, MappedFile};

    /// Builds an in-memory GGUF from `(name, GGML dimensions, values)` triples.
    fn synthetic_gguf(specifications: &[(&str, &[u64], &[f32])]) -> GGUFFile {
        let mut bytes = Vec::new();
        let mut tensors = Vec::new();
        for (name, dims, values) in specifications {
            let mut ne = [1u64; 4];
            ne[..dims.len()].copy_from_slice(dims);
            tensors.push(Gguftensor {
                name: (*name).to_string(),
                n_dims: dims.len() as u32,
                ne,
                ttype: GgmlType(GGML_TYPE_F32),
                offset: bytes.len() as u64,
                data_offset: bytes.len(),
            });
            bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        }
        let tensor_lookup = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        GGUFFile {
            version: 3,
            n_tensors: tensors.len() as u64,
            n_kv: 0,
            kv: HashMap::new(),
            tensors,
            tensor_lookup,
            tensor_data_start: 0,
            vocab_tokens: Vec::new(),
            vocab_scores: Vec::new(),
            vocab_merges: Vec::new(),
            mapped: MappedFile::from_static(Box::leak(bytes.into_boxed_slice())).unwrap(),
        }
    }

    fn conv(input: usize, output: usize, index: usize) -> FaceGraphOpPolicy {
        FaceGraphOpPolicy::Conv {
            input,
            output,
            stride: 1,
            pad: 0,
            groups: 1,
            activation: FaceActivation::Relu,
            weight_name: format!("face.conv.{index}.weight"),
            bias_name: format!("face.conv.{index}.bias"),
            slope_name: format!("face.conv.{index}.slope"),
        }
    }

    /// conv(1x1, x2, relu) -> max pool -> upsample -> add with the conv output -> linear sum.
    fn policy() -> FaceGraphPolicy {
        FaceGraphPolicy {
            slot_count: 6,
            ops: vec![
                conv(0, 1, 0),
                FaceGraphOpPolicy::MaxPool {
                    input: 1,
                    output: 2,
                },
                FaceGraphOpPolicy::Upsample {
                    input: 2,
                    output: 3,
                },
                FaceGraphOpPolicy::Add {
                    left: 1,
                    right: 3,
                    output: 4,
                },
                FaceGraphOpPolicy::Linear {
                    input: 4,
                    output: 5,
                    weight_name: "face.linear.0.weight".to_string(),
                    bias_name: "face.linear.0.bias".to_string(),
                },
            ],
        }
    }

    fn gguf() -> GGUFFile {
        synthetic_gguf(&[
            ("face.conv.0.weight", &[1, 1, 1, 1], &[2.0]),
            ("face.conv.0.bias", &[1], &[0.0]),
            ("face.linear.0.weight", &[4, 1], &[1.0; 4]),
            ("face.linear.0.bias", &[1], &[0.5]),
        ])
    }

    #[test]
    fn runs_operations_in_order_and_returns_kept_slots() {
        let graph = FaceGraph::load(&gguf(), &policy()).unwrap();
        assert_eq!(graph.input_channels(), 1);
        assert!(graph.has_slot(5) && !graph.has_slot(0) && !graph.has_slot(6));
        let input = FaceTensor::new(1, 2, 2, vec![1.0, -2.0, 3.0, 4.0]).unwrap();
        let outputs = graph.run(input, &[5, 1]).unwrap();
        // conv+relu: [2, 0, 6, 8]; pooled max 8; sum = 16 + 4 * 8 = 48.
        assert_eq!(outputs[0].data, vec![48.5]);
        assert_eq!(outputs[1].data, vec![2.0, 0.0, 6.0, 8.0]);
    }

    #[test]
    fn rejects_missing_tensors_and_bad_slot_wiring() {
        let mut missing = policy();
        missing.ops[0] = conv(0, 1, 7);
        assert!(FaceGraph::load(&gguf(), &missing).is_err());

        let mut unwritten = policy();
        unwritten.ops[1] = FaceGraphOpPolicy::MaxPool {
            input: 4,
            output: 2,
        };
        assert!(FaceGraph::load(&gguf(), &unwritten).is_err());

        let mut reused = policy();
        reused.ops[2] = FaceGraphOpPolicy::Upsample {
            input: 2,
            output: 1,
        };
        assert!(FaceGraph::load(&gguf(), &reused).is_err());
    }

    #[test]
    fn shape_errors_name_the_failing_operation() {
        let graph = FaceGraph::load(&gguf(), &policy()).unwrap();
        let input = FaceTensor::new(1, 4, 4, vec![0.0; 16]).unwrap();
        let error = graph.run(input, &[5]).unwrap_err();
        assert!(error.contains("operation 4"), "{error}");
    }
}
