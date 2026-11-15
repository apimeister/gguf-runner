//! Lowers an ONNX face model to the flat operation list of the face GGUF contracts
//! (`docs/face-recognition.md`): batch normalization is folded into the neighbouring
//! convolution or linear layer, an activation joins its convolution, and nodes that only
//! change the layout disappear. Weights stay float32; the folding itself is computed in
//! float64.

use std::collections::HashMap;

use crate::engine::io::{GgufWriter, OnnxAttribute, OnnxModel, OnnxNode};

const OP_CONV: i32 = 0;
const OP_MAX_POOL: i32 = 1;
const OP_UPSAMPLE: i32 = 2;
const OP_ADD: i32 = 3;
const OP_LINEAR: i32 = 4;

const ACT_NONE: i32 = 0;
const ACT_RELU: i32 = 1;
const ACT_PRELU: i32 = 2;

/// ONNX's default for the `epsilon` of a batch normalization.
const DEFAULT_EPSILON: f64 = 1e-5;

struct LoweredOp {
    kind: i32,
    a: usize,
    b: usize,
    out: usize,
    stride: usize,
    pad: usize,
    groups: usize,
    act: i32,
    /// `[out, in / groups, kernel, kernel]` for a convolution, `[outputs, inputs]` for a
    /// linear layer, empty for every other operation.
    weight_shape: Vec<usize>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    slope: Vec<f32>,
}

/// Per-channel scale and shift of a batch normalization.
type Affine = (Vec<f64>, Vec<f64>);

/// A face model as its GGUF holds it: operations over numbered slots, slot 0 being the
/// network input.
pub(crate) struct LoweredFaceGraph {
    ops: Vec<LoweredOp>,
    slots: HashMap<String, usize>,
    slot_count: usize,
    input_mean: f32,
    input_scale: f32,
}

impl LoweredFaceGraph {
    /// The slot that holds the ONNX value `name`.
    pub(crate) fn slot(&self, name: &str) -> Result<usize, String> {
        self.slots
            .get(name)
            .copied()
            .ok_or_else(|| format!("ONNX graph has no value '{name}'"))
    }

    /// How many values the last operation writes per position: the output channels of a
    /// convolution or the outputs of a linear layer.
    pub(crate) fn last_output_size(&self) -> Option<usize> {
        self.ops.last()?.weight_shape.first().copied()
    }

    /// Writes the operation list, the input normalization and the tensors. Convolution and
    /// linear tensors are numbered in operation order.
    pub(crate) fn write(self, writer: &mut GgufWriter, prefix: &str) {
        let mut flat: Vec<i32> = Vec::with_capacity(self.ops.len() * 8);
        let (mut convs, mut linears) = (0usize, 0usize);
        for op in self.ops {
            flat.extend([
                op.kind,
                op.a as i32,
                op.b as i32,
                op.out as i32,
                op.stride as i32,
                op.pad as i32,
                op.groups as i32,
                op.act,
            ]);
            match op.kind {
                OP_CONV => {
                    let base = format!("face.conv.{convs}");
                    convs += 1;
                    let outputs = op.bias.len();
                    writer.add_f32_tensor(&format!("{base}.weight"), &op.weight_shape, op.weight);
                    writer.add_f32_tensor(&format!("{base}.bias"), &[outputs], op.bias);
                    if op.act == ACT_PRELU {
                        let channels = op.slope.len();
                        writer.add_f32_tensor(&format!("{base}.slope"), &[channels], op.slope);
                    }
                }
                OP_LINEAR => {
                    let base = format!("face.linear.{linears}");
                    linears += 1;
                    let outputs = op.bias.len();
                    writer.add_f32_tensor(&format!("{base}.weight"), &op.weight_shape, op.weight);
                    writer.add_f32_tensor(&format!("{base}.bias"), &[outputs], op.bias);
                }
                _ => {}
            }
        }
        writer.add_i32_array(&format!("{prefix}.graph.ops"), &flat);
        writer.add_u32(
            &format!("{prefix}.graph.slot_count"),
            self.slot_count as u32,
        );
        writer.add_f32(&format!("{prefix}.input.mean"), self.input_mean);
        writer.add_f32(&format!("{prefix}.input.scale"), self.input_scale);
    }
}

/// Lowers `model`, whose image input is the value `input_name`.
pub(crate) fn lower_face_onnx(
    model: &OnnxModel,
    input_name: &str,
) -> Result<LoweredFaceGraph, String> {
    let mut lowering = Lowering {
        model,
        graph: LoweredFaceGraph {
            ops: Vec::new(),
            slots: HashMap::from([(input_name.to_string(), 0)]),
            slot_count: 1,
            input_mean: 0.0,
            input_scale: 1.0,
        },
        open: HashMap::new(),
        pending: HashMap::new(),
    };
    if !model
        .nodes
        .iter()
        .any(|node| node.inputs.iter().any(|name| name == input_name))
    {
        return Err(format!(
            "the ONNX graph has no image input '{input_name}'; it is not the model this conversion is for"
        ));
    }
    for node in &model.nodes {
        lowering.lower(node).map_err(|error| {
            format!("ONNX node {} ({}): {error}", node_label(node), node.op_type)
        })?;
    }
    if let Some(name) = lowering.pending.keys().next() {
        return Err(format!(
            "the batch normalization that writes '{name}' has no layer to fold into"
        ));
    }
    Ok(lowering.graph)
}

fn node_label(node: &OnnxNode) -> &str {
    node.outputs
        .first()
        .map_or("without output", String::as_str)
}

struct Lowering<'a> {
    model: &'a OnnxModel,
    graph: LoweredFaceGraph,
    /// Value name to the operation that may still absorb a following batch normalization
    /// or activation.
    open: HashMap<String, usize>,
    /// Value name to the batch normalization waiting for the next linear layer.
    pending: HashMap<String, Affine>,
}

impl Lowering<'_> {
    fn lower(&mut self, node: &OnnxNode) -> Result<(), String> {
        match node.op_type.as_str() {
            "Sub" => {
                self.graph.input_mean = self.input_scalar(node)?;
                self.pass_through(node)
            }
            "Mul" => {
                self.graph.input_scale = self.input_scalar(node)?;
                self.pass_through(node)
            }
            "Conv" => self.conv(node),
            "BatchNormalization" => self.batch_norm(node),
            "Relu" => self.activation(node, ACT_RELU, Vec::new()),
            "PRelu" => {
                let slope = self.model.floats(input(node, 1)?)?.0.to_vec();
                self.activation(node, ACT_PRELU, slope)
            }
            "MaxPool" => {
                let two = [2i64, 2];
                if ints(node, "kernel_shape") != Some(&two[..])
                    || ints(node, "strides") != Some(&two[..])
                    || ints(node, "pads").is_some_and(|pads| pads.iter().any(|pad| *pad != 0))
                {
                    return Err("only 2x2 max pooling with stride 2 is supported".to_string());
                }
                self.simple(node, OP_MAX_POOL, 1)
            }
            "Resize" => {
                if !matches!(node.attributes.get("mode"), Some(OnnxAttribute::Bytes(mode)) if mode == b"nearest")
                    || self.model.floats(input(node, 2)?)?.0 != [1.0, 1.0, 2.0, 2.0]
                {
                    return Err("only 2x nearest upsampling is supported".to_string());
                }
                self.simple(node, OP_UPSAMPLE, 1)
            }
            "Add" => self.simple(node, OP_ADD, 2),
            // Layout-only nodes, and nodes that do nothing at inference. The runner reads
            // head outputs in channel-major order and applies the sigmoid while decoding.
            "Transpose" | "Reshape" | "Sigmoid" | "Dropout" | "Flatten" => self.pass_through(node),
            "Gemm" => self.gemm(node),
            _ => Err("unsupported operation".to_string()),
        }
    }

    fn slot(&self, name: &str) -> Result<usize, String> {
        self.graph
            .slots
            .get(name)
            .copied()
            .ok_or_else(|| format!("reads '{name}', which no earlier node wrote"))
    }

    fn new_slot(&mut self, name: &str) -> usize {
        let slot = self.graph.slot_count;
        self.graph.slots.insert(name.to_string(), slot);
        self.graph.slot_count += 1;
        slot
    }

    /// Makes the node's output another name for its first input.
    fn pass_through(&mut self, node: &OnnxNode) -> Result<(), String> {
        let (old, new) = (input(node, 0)?, output(node)?);
        let slot = self.slot(old)?;
        self.graph.slots.insert(new.to_string(), slot);
        if let Some(op) = self.open.remove(old) {
            self.open.insert(new.to_string(), op);
        }
        if let Some(affine) = self.pending.remove(old) {
            self.pending.insert(new.to_string(), affine);
        }
        Ok(())
    }

    /// The scalar a `Sub` or `Mul` applies to the network input.
    fn input_scalar(&self, node: &OnnxNode) -> Result<f32, String> {
        if self.slot(input(node, 0)?)? != 0 || !self.graph.ops.is_empty() {
            return Err("is only supported on the network input".to_string());
        }
        match self.model.floats(input(node, 1)?)?.0 {
            [scalar] => Ok(*scalar),
            _ => Err("needs a scalar".to_string()),
        }
    }

    fn conv(&mut self, node: &OnnxNode) -> Result<(), String> {
        let (weight, shape) = self.model.floats(input(node, 1)?)?;
        let [outputs, _, kernel_height, kernel_width] = shape[..] else {
            return Err("its weight is not a 4-D kernel".to_string());
        };
        let bias = match node.inputs.get(2) {
            Some(name) => self.model.floats(name)?.0.to_vec(),
            None => vec![0.0; outputs],
        };
        if bias.len() != outputs {
            return Err("its bias does not match its weight".to_string());
        }
        let kernel = ints(node, "kernel_shape").ok_or("has no kernel_shape")?;
        let strides = ints(node, "strides").unwrap_or(&[1, 1]);
        let pads = ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
        if kernel != [kernel_height as i64, kernel_width as i64]
            || kernel_height != kernel_width
            || strides.len() != 2
            || strides[0] != strides[1]
        {
            return Err("non-square convolution".to_string());
        }
        if pads.is_empty()
            || pads.iter().any(|pad| *pad != pads[0])
            || ints(node, "dilations").is_some_and(|dilations| dilations != [1, 1])
        {
            return Err("unsupported padding or dilation".to_string());
        }
        let source = input(node, 0)?;
        if self.pending.contains_key(source) {
            return Err("a batch normalization before a convolution is not supported".to_string());
        }
        let op = LoweredOp {
            kind: OP_CONV,
            a: self.slot(source)?,
            b: 0,
            out: 0,
            stride: non_negative(strides[0])?,
            pad: non_negative(pads[0])?,
            groups: non_negative(int(node, "group").unwrap_or(1))?,
            act: ACT_NONE,
            weight_shape: shape.to_vec(),
            weight: weight.to_vec(),
            bias,
            slope: Vec::new(),
        };
        self.push_open(node, op)
    }

    fn batch_norm(&mut self, node: &OnnxNode) -> Result<(), String> {
        let parameter = |position: usize| -> Result<Vec<f64>, String> {
            let values = self.model.floats(input(node, position)?)?.0;
            Ok(values.iter().map(|value| f64::from(*value)).collect())
        };
        let (gamma, beta, mean, variance) =
            (parameter(1)?, parameter(2)?, parameter(3)?, parameter(4)?);
        let channels = gamma.len();
        if [beta.len(), mean.len(), variance.len()] != [channels; 3] {
            return Err("its parameters differ in length".to_string());
        }
        let epsilon = match node.attributes.get("epsilon") {
            Some(OnnxAttribute::Float(epsilon)) => f64::from(*epsilon),
            _ => DEFAULT_EPSILON,
        };
        let scale: Vec<f64> = (0..channels)
            .map(|channel| gamma[channel] / (variance[channel] + epsilon).sqrt())
            .collect();
        let shift: Vec<f64> = (0..channels)
            .map(|channel| beta[channel] - mean[channel] * scale[channel])
            .collect();

        let source = input(node, 0)?;
        let open = self
            .open
            .get(source)
            .copied()
            .filter(|op| self.graph.ops[*op].act == ACT_NONE);
        match open {
            Some(op) => {
                let op = &mut self.graph.ops[op];
                if op.bias.len() != channels {
                    return Err("does not match the layer it follows".to_string());
                }
                let per_output = op.weight.len() / channels;
                for (output, weights) in op.weight.chunks_mut(per_output).enumerate() {
                    for weight in weights {
                        *weight = (f64::from(*weight) * scale[output]) as f32;
                    }
                }
                for (output, bias) in op.bias.iter_mut().enumerate() {
                    *bias = (f64::from(*bias) * scale[output] + shift[output]) as f32;
                }
            }
            None => {
                if self.pending.contains_key(source) {
                    return Err("two batch normalizations in a row are not supported".to_string());
                }
                self.pending.insert(source.to_string(), (scale, shift));
            }
        }
        self.pass_through(node)
    }

    fn activation(&mut self, node: &OnnxNode, act: i32, slope: Vec<f32>) -> Result<(), String> {
        let op = self
            .open
            .get(input(node, 0)?)
            .map(|op| &mut self.graph.ops[*op])
            .filter(|op| op.kind == OP_CONV && op.act == ACT_NONE)
            .ok_or("an activation has to follow a convolution directly")?;
        if act == ACT_PRELU && slope.len() != op.bias.len() {
            return Err("its slope does not match the convolution".to_string());
        }
        op.act = act;
        op.slope = slope;
        self.pass_through(node)
    }

    /// An operation without weights over one or two inputs.
    fn simple(&mut self, node: &OnnxNode, kind: i32, inputs: usize) -> Result<(), String> {
        let mut slots = [0usize; 2];
        for (position, slot) in slots.iter_mut().enumerate().take(inputs) {
            let name = input(node, position)?;
            if self.pending.contains_key(name) {
                return Err("a batch normalization before it is not supported".to_string());
            }
            *slot = self.slot(name)?;
        }
        let out = self.new_slot(output(node)?);
        self.graph.ops.push(LoweredOp {
            kind,
            a: slots[0],
            b: slots[1],
            out,
            stride: 0,
            pad: 0,
            groups: 0,
            act: ACT_NONE,
            weight_shape: Vec::new(),
            weight: Vec::new(),
            bias: Vec::new(),
            slope: Vec::new(),
        });
        Ok(())
    }

    fn gemm(&mut self, node: &OnnxNode) -> Result<(), String> {
        let one = |name: &str| match node.attributes.get(name) {
            Some(OnnxAttribute::Float(value)) => *value == 1.0,
            None => true,
            _ => false,
        };
        if int(node, "transB") != Some(1) || !one("alpha") || !one("beta") {
            return Err("only y = x * W^T + b is supported".to_string());
        }
        let (weight, shape) = self.model.floats(input(node, 1)?)?;
        let [outputs, inputs] = shape[..] else {
            return Err("its weight is not a matrix".to_string());
        };
        let mut weight: Vec<f64> = weight.iter().map(|value| f64::from(*value)).collect();
        let mut bias: Vec<f64> = self
            .model
            .floats(input(node, 2)?)?
            .0
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        if bias.len() != outputs {
            return Err("its bias does not match its weight".to_string());
        }

        let source = input(node, 0)?;
        if let Some((scale, shift)) = self.pending.remove(source) {
            // The flattened input is channel-major: each channel covers `per_channel`
            // consecutive inputs.
            let channels = scale.len();
            if channels == 0 || !inputs.is_multiple_of(channels) {
                return Err("the batch normalization before it does not match".to_string());
            }
            let per_channel = inputs / channels;
            for (row, bias) in weight.chunks(inputs).zip(bias.iter_mut()) {
                let mut shifted = 0.0f64;
                for (position, weight) in row.iter().enumerate() {
                    shifted += weight * shift[position / per_channel];
                }
                *bias += shifted;
            }
            for row in weight.chunks_mut(inputs) {
                for (position, weight) in row.iter_mut().enumerate() {
                    *weight *= scale[position / per_channel];
                }
            }
        }
        let op = LoweredOp {
            kind: OP_LINEAR,
            a: self.slot(source)?,
            b: 0,
            out: 0,
            stride: 0,
            pad: 0,
            groups: 0,
            act: ACT_NONE,
            weight_shape: shape.to_vec(),
            weight: weight.into_iter().map(|value| value as f32).collect(),
            bias: bias.into_iter().map(|value| value as f32).collect(),
            slope: Vec::new(),
        };
        self.push_open(node, op)
    }

    /// Adds a convolution or linear layer, which a following batch normalization or
    /// activation may still join.
    fn push_open(&mut self, node: &OnnxNode, mut op: LoweredOp) -> Result<(), String> {
        let name = output(node)?;
        op.out = self.new_slot(name);
        self.graph.ops.push(op);
        self.open.insert(name.to_string(), self.graph.ops.len() - 1);
        Ok(())
    }
}

fn input(node: &OnnxNode, position: usize) -> Result<&str, String> {
    node.inputs
        .get(position)
        .map(String::as_str)
        .ok_or_else(|| format!("has no input {position}"))
}

fn output(node: &OnnxNode) -> Result<&str, String> {
    node.outputs
        .first()
        .map(String::as_str)
        .ok_or_else(|| "has no output".to_string())
}

fn ints<'a>(node: &'a OnnxNode, name: &str) -> Option<&'a [i64]> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Ints(values)) => Some(values),
        _ => None,
    }
}

fn int(node: &OnnxNode, name: &str) -> Option<i64> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Int(value)) => Some(*value),
        _ => None,
    }
}

fn non_negative(value: i64) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("has invalid attribute value {value}"))
}

#[cfg(test)]
mod tests {
    use super::{ACT_PRELU, ACT_RELU, LoweredFaceGraph, lower_face_onnx};
    use crate::engine::io::OnnxModel;
    use crate::engine::io::onnx_test_support::{
        Message, float_attribute, float_tensor, int_attribute, ints_attribute, model, node,
        text_attribute,
    };

    fn conv(
        inputs: &[&str],
        output: &str,
        kernel: i64,
        stride: i64,
        pad: i64,
        group: i64,
    ) -> Message {
        node("Conv", inputs, &[output])
            .message(5, ints_attribute("dilations", &[1, 1]))
            .message(5, int_attribute("group", group))
            .message(5, ints_attribute("kernel_shape", &[kernel, kernel]))
            .message(5, ints_attribute("pads", &[pad; 4]))
            .message(5, ints_attribute("strides", &[stride, stride]))
    }

    /// A batch normalization whose parameters are the initializers `<name>_gamma`,
    /// `_beta`, `_mean` and `_var`, with an epsilon of 1.
    fn batch_norm(input: &str, name: &str) -> Message {
        let parameter = |suffix: &str| format!("{name}_{suffix}");
        node(
            "BatchNormalization",
            &[
                input,
                &parameter("gamma"),
                &parameter("beta"),
                &parameter("mean"),
                &parameter("var"),
            ],
            &[name],
        )
        .message(5, float_attribute("epsilon", 1.0))
    }

    fn batch_norm_parameters(
        name: &str,
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        variance: &[f32],
    ) -> Vec<Message> {
        let dims = [gamma.len() as i64];
        vec![
            float_tensor(&format!("{name}_gamma"), &dims, gamma, false),
            float_tensor(&format!("{name}_beta"), &dims, beta, false),
            float_tensor(&format!("{name}_mean"), &dims, mean, false),
            float_tensor(&format!("{name}_var"), &dims, variance, false),
        ]
    }

    fn lowered(
        nodes: Vec<Message>,
        initializers: Vec<Message>,
    ) -> Result<LoweredFaceGraph, String> {
        lower_face_onnx(&OnnxModel::parse(&model(nodes, initializers))?, "input")
    }

    #[test]
    fn batch_normalization_and_activation_join_their_convolution() {
        let mut initializers = vec![
            float_tensor("w", &[2, 1, 1, 1], &[2.0, 3.0], true),
            float_tensor("b", &[2], &[1.0, -1.0], true),
        ];
        // Scale 0.5 for both channels, shift 0 and 2.
        initializers.extend(batch_norm_parameters(
            "bn",
            &[1.0, 2.0],
            &[0.5, 3.0],
            &[1.0, 2.0],
            &[3.0, 15.0],
        ));
        let graph = lowered(
            vec![
                conv(&["input", "w", "b"], "conv", 1, 2, 0, 1),
                batch_norm("conv", "bn"),
                node("Relu", &["bn"], &["out"]),
            ],
            initializers,
        )
        .unwrap();

        assert_eq!(graph.ops.len(), 1);
        let op = &graph.ops[0];
        assert_eq!(
            (op.kind, op.a, op.out, op.stride, op.pad, op.groups, op.act),
            (0, 0, 1, 2, 0, 1, ACT_RELU)
        );
        assert_eq!(op.weight, [1.0, 1.5]);
        assert_eq!(op.bias, [0.5, 1.5]);
        assert_eq!(graph.slot("out"), Ok(1));
        assert_eq!(graph.slot_count, 2);
    }

    #[test]
    fn a_batch_normalization_after_an_activation_joins_the_linear_layer() {
        let mut initializers = vec![
            float_tensor("mean", &[1], &[127.5], false),
            float_tensor("scale", &[1], &[0.5], false),
            float_tensor("w", &[1, 3, 1, 1], &[1.0, 1.0, 1.0], false),
            // A slope per channel, shaped as the published file shapes it.
            float_tensor("slope", &[1, 1, 1], &[0.25], false),
            float_tensor(
                "fc_w",
                &[2, 4],
                &[1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5],
                false,
            ),
            float_tensor("fc_b", &[2], &[1.0, -1.0], false),
        ];
        // Scale 2 and shift 0.5 for the one channel in front of the linear layer.
        initializers.extend(batch_norm_parameters("bn", &[4.0], &[1.0], &[0.25], &[3.0]));
        // Scale 0.5 and 1, shift 0 and 1 behind it.
        initializers.extend(batch_norm_parameters(
            "fc1",
            &[1.0, 3.0],
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[3.0, 8.0],
        ));
        let graph = lower_face_onnx(
            &OnnxModel::parse(&model(
                vec![
                    node("Sub", &["data", "mean"], &["centered"]),
                    node("Mul", &["centered", "scale"], &["scaled"]),
                    conv(&["scaled", "w"], "conv", 1, 1, 0, 1),
                    node("PRelu", &["conv", "slope"], &["relu"]),
                    batch_norm("relu", "bn"),
                    node("Dropout", &["bn"], &["dropout"]),
                    node("Flatten", &["dropout"], &["flat"]),
                    node("Gemm", &["flat", "fc_w", "fc_b"], &["pre_fc1"])
                        .message(5, float_attribute("alpha", 1.0))
                        .message(5, float_attribute("beta", 1.0))
                        .message(5, int_attribute("transA", 0))
                        .message(5, int_attribute("transB", 1)),
                    batch_norm("pre_fc1", "fc1"),
                ],
                initializers,
            ))
            .unwrap(),
            "data",
        )
        .unwrap();

        assert_eq!((graph.input_mean, graph.input_scale), (127.5, 0.5));
        assert_eq!(graph.ops.len(), 2);
        let conv = &graph.ops[0];
        assert_eq!(conv.act, ACT_PRELU);
        assert_eq!(conv.slope, [0.25]);
        assert_eq!(conv.bias, [0.0]);
        // The linear layer took the normalization in front of it and the one behind it.
        let linear = &graph.ops[1];
        assert_eq!((linear.kind, linear.a, linear.out), (4, 1, 2));
        assert_eq!(linear.weight, [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(linear.bias, [3.0, 1.0]);
        assert_eq!(graph.slot("fc1"), Ok(2));
        assert_eq!(graph.last_output_size(), Some(2));
    }

    #[test]
    fn layout_nodes_disappear_and_pool_upsample_and_add_become_operations() {
        let graph = lowered(
            vec![
                conv(&["input", "w1", "b1"], "c1", 1, 1, 0, 1),
                node("Relu", &["c1"], &["r1"]),
                node("MaxPool", &["r1"], &["pooled"])
                    .message(5, int_attribute("ceil_mode", 0))
                    .message(5, ints_attribute("kernel_shape", &[2, 2]))
                    .message(5, ints_attribute("pads", &[0; 4]))
                    .message(5, ints_attribute("strides", &[2, 2])),
                conv(&["pooled", "w2", "b2"], "c2", 3, 1, 1, 2),
                node("Relu", &["c2"], &["r2"]),
                node("Resize", &["r2", "roi", "scales"], &["up"])
                    .message(5, text_attribute("mode", "nearest")),
                node("Add", &["r1", "up"], &["sum"]),
                node("Transpose", &["sum"], &["moved"])
                    .message(5, ints_attribute("perm", &[0, 2, 3, 1])),
                node("Reshape", &["moved", "shape"], &["flat"]),
                node("Sigmoid", &["flat"], &["cls_8"]),
            ],
            vec![
                float_tensor("w1", &[2, 3, 1, 1], &[0.5; 6], true),
                float_tensor("b1", &[2], &[0.0; 2], true),
                float_tensor("w2", &[2, 1, 3, 3], &[0.25; 18], true),
                float_tensor("b2", &[2], &[0.0; 2], true),
                float_tensor("roi", &[0], &[], false),
                float_tensor("scales", &[4], &[1.0, 1.0, 2.0, 2.0], true),
            ],
        )
        .unwrap();

        let records: Vec<[usize; 8]> = graph
            .ops
            .iter()
            .map(|op| {
                [
                    op.kind as usize,
                    op.a,
                    op.b,
                    op.out,
                    op.stride,
                    op.pad,
                    op.groups,
                    op.act as usize,
                ]
            })
            .collect();
        assert_eq!(
            records,
            [
                [0, 0, 0, 1, 1, 0, 1, 1], // convolution with ReLU
                [1, 1, 0, 2, 0, 0, 0, 0], // max pool
                [0, 2, 0, 3, 1, 1, 2, 1], // grouped convolution with ReLU
                [2, 3, 0, 4, 0, 0, 0, 0], // upsample
                [3, 1, 4, 5, 0, 0, 0, 0], // add
            ]
        );
        assert_eq!(graph.slot("cls_8"), Ok(5));
        assert_eq!(graph.slot_count, 6);
    }

    #[test]
    fn a_graph_the_contracts_cannot_hold_is_refused_with_its_node() {
        let weights = || {
            vec![
                float_tensor("w", &[1, 3, 1, 1], &[1.0; 3], true),
                float_tensor("b", &[1], &[0.0], true),
            ]
        };
        let error = |nodes: Vec<Message>, initializers: Vec<Message>| {
            lowered(nodes, initializers).err().expect("refused")
        };

        let unsupported = error(vec![node("Softmax", &["input"], &["out"])], vec![]);
        assert_eq!(
            unsupported,
            "ONNX node out (Softmax): unsupported operation"
        );

        let stray_activation = error(vec![node("Relu", &["input"], &["out"])], vec![]);
        assert!(stray_activation.contains("has to follow a convolution"));

        let another_model = error(vec![node("Relu", &["data"], &["out"])], vec![]);
        assert!(another_model.contains("no image input 'input'"));

        let unknown_value = error(
            vec![
                conv(&["input", "w", "b"], "conv", 1, 1, 0, 1),
                node("Add", &["conv", "nowhere"], &["out"]),
            ],
            weights(),
        );
        assert!(unknown_value.contains("reads 'nowhere', which no earlier node wrote"));

        // A normalization behind an activation with no linear layer after it.
        let mut initializers = weights();
        initializers.extend(batch_norm_parameters("bn", &[1.0], &[0.0], &[0.0], &[3.0]));
        let nothing_to_fold_into = error(
            vec![
                conv(&["input", "w", "b"], "conv", 1, 1, 0, 1),
                node("Relu", &["conv"], &["relu"]),
                batch_norm("relu", "bn"),
            ],
            initializers,
        );
        assert!(nothing_to_fold_into.contains("no layer to fold into"));

        let dilated = error(
            vec![
                node("Conv", &["input", "w", "b"], &["conv"])
                    .message(5, ints_attribute("dilations", &[2, 2]))
                    .message(5, ints_attribute("kernel_shape", &[1, 1])),
            ],
            weights(),
        );
        assert!(dilated.contains("unsupported padding or dilation"));

        let mut later_scaling = weights();
        later_scaling.push(float_tensor("scale", &[1], &[2.0], true));
        let scaled_too_late = error(
            vec![
                conv(&["input", "w", "b"], "conv", 1, 1, 0, 1),
                node("Mul", &["conv", "scale"], &["out"]),
            ],
            later_scaling,
        );
        assert!(scaled_too_late.contains("only supported on the network input"));
    }
}
