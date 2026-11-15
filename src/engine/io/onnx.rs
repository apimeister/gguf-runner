//! Reader for the part of an ONNX file the face converters use: the nodes of the graph with
//! their attributes, and the initializers that hold floats. An ONNX file is a protocol
//! buffer; this reads the wire format directly.

use std::collections::HashMap;

/// ONNX `TensorProto.DataType` of 32-bit floats.
const DATA_TYPE_FLOAT: u64 = 1;
/// ONNX `TensorProto.DataLocation` of a tensor whose data sits in another file.
const DATA_LOCATION_EXTERNAL: u64 = 1;

#[derive(Debug, Default)]
pub(crate) struct OnnxModel {
    /// The nodes of the graph in file order, which is execution order.
    pub(crate) nodes: Vec<OnnxNode>,
    pub(crate) initializers: HashMap<String, OnnxTensor>,
}

#[derive(Debug, Default)]
pub(crate) struct OnnxNode {
    pub(crate) op_type: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
    pub(crate) attributes: HashMap<String, OnnxAttribute>,
}

/// The attribute values this reader keeps. An attribute of another kind is kept as `Other`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OnnxAttribute {
    Float(f32),
    Int(i64),
    Bytes(Vec<u8>),
    Ints(Vec<i64>),
    Other,
}

#[derive(Debug, Default)]
pub(crate) struct OnnxTensor {
    pub(crate) dims: Vec<usize>,
    /// The values of a float tensor in row-major order; `None` for another data type.
    pub(crate) floats: Option<Vec<f32>>,
}

impl OnnxModel {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, String> {
        const MODEL_GRAPH: u32 = 7;
        const GRAPH_NODE: u32 = 1;
        const GRAPH_INITIALIZER: u32 = 5;

        let mut graph = None;
        let mut fields = Fields::new(bytes);
        while let Some((number, value)) = fields.next()? {
            if number == MODEL_GRAPH {
                graph = Some(value.bytes()?);
            }
        }
        let graph = graph.ok_or_else(|| "ONNX file holds no graph".to_string())?;

        let mut model = OnnxModel::default();
        let mut fields = Fields::new(graph);
        while let Some((number, value)) = fields.next()? {
            match number {
                GRAPH_NODE => model.nodes.push(parse_node(value.bytes()?)?),
                GRAPH_INITIALIZER => {
                    let (name, tensor) = parse_tensor(value.bytes()?)?;
                    model.initializers.insert(name, tensor);
                }
                _ => {}
            }
        }
        Ok(model)
    }

    /// The float values and dimensions of an initializer.
    pub(crate) fn floats(&self, name: &str) -> Result<(&[f32], &[usize]), String> {
        let tensor = self
            .initializers
            .get(name)
            .ok_or_else(|| format!("ONNX graph has no initializer '{name}'"))?;
        let floats = tensor
            .floats
            .as_deref()
            .ok_or_else(|| format!("ONNX initializer '{name}' holds no 32-bit floats"))?;
        Ok((floats, &tensor.dims))
    }
}

fn parse_node(bytes: &[u8]) -> Result<OnnxNode, String> {
    const NODE_INPUT: u32 = 1;
    const NODE_OUTPUT: u32 = 2;
    const NODE_OP_TYPE: u32 = 4;
    const NODE_ATTRIBUTE: u32 = 5;

    let mut node = OnnxNode::default();
    let mut fields = Fields::new(bytes);
    while let Some((number, value)) = fields.next()? {
        match number {
            NODE_INPUT => node.inputs.push(value.string()?),
            NODE_OUTPUT => node.outputs.push(value.string()?),
            NODE_OP_TYPE => node.op_type = value.string()?,
            NODE_ATTRIBUTE => {
                let (name, attribute) = parse_attribute(value.bytes()?)?;
                node.attributes.insert(name, attribute);
            }
            _ => {}
        }
    }
    Ok(node)
}

fn parse_attribute(bytes: &[u8]) -> Result<(String, OnnxAttribute), String> {
    const ATTRIBUTE_NAME: u32 = 1;
    const ATTRIBUTE_F: u32 = 2;
    const ATTRIBUTE_I: u32 = 3;
    const ATTRIBUTE_S: u32 = 4;
    const ATTRIBUTE_INTS: u32 = 8;
    const ATTRIBUTE_TYPE: u32 = 20;
    // `AttributeProto.AttributeType`.
    const TYPE_FLOAT: u64 = 1;
    const TYPE_INT: u64 = 2;
    const TYPE_STRING: u64 = 3;
    const TYPE_INTS: u64 = 7;

    let mut name = String::new();
    let (mut float, mut int, mut text) = (None, None, None);
    let mut ints: Vec<i64> = Vec::new();
    let mut kind = None;
    let mut fields = Fields::new(bytes);
    while let Some((number, value)) = fields.next()? {
        match number {
            ATTRIBUTE_NAME => name = value.string()?,
            ATTRIBUTE_F => float = Some(value.float()?),
            ATTRIBUTE_I => int = Some(value.varint()? as i64),
            ATTRIBUTE_S => text = Some(value.bytes()?.to_vec()),
            ATTRIBUTE_INTS => value.varints(&mut ints)?,
            ATTRIBUTE_TYPE => kind = Some(value.varint()?),
            _ => {}
        }
    }
    // A file written before attributes carried their type says it through the field in use.
    let attribute = match kind {
        Some(TYPE_FLOAT) => OnnxAttribute::Float(float.unwrap_or(0.0)),
        Some(TYPE_INT) => OnnxAttribute::Int(int.unwrap_or(0)),
        Some(TYPE_STRING) => OnnxAttribute::Bytes(text.unwrap_or_default()),
        Some(TYPE_INTS) => OnnxAttribute::Ints(ints),
        Some(_) => OnnxAttribute::Other,
        None => match (float, int, text) {
            (Some(float), _, _) => OnnxAttribute::Float(float),
            (_, Some(int), _) => OnnxAttribute::Int(int),
            (_, _, Some(text)) => OnnxAttribute::Bytes(text),
            _ if !ints.is_empty() => OnnxAttribute::Ints(ints),
            _ => OnnxAttribute::Other,
        },
    };
    Ok((name, attribute))
}

fn parse_tensor(bytes: &[u8]) -> Result<(String, OnnxTensor), String> {
    const TENSOR_DIMS: u32 = 1;
    const TENSOR_DATA_TYPE: u32 = 2;
    const TENSOR_FLOAT_DATA: u32 = 4;
    const TENSOR_NAME: u32 = 8;
    const TENSOR_RAW_DATA: u32 = 9;
    const TENSOR_DATA_LOCATION: u32 = 14;

    let mut name = String::new();
    let mut dims: Vec<i64> = Vec::new();
    let mut data_type = 0;
    let mut float_data: Vec<f32> = Vec::new();
    let mut raw_data: &[u8] = &[];
    let mut external = false;
    let mut fields = Fields::new(bytes);
    while let Some((number, value)) = fields.next()? {
        match number {
            TENSOR_DIMS => value.varints(&mut dims)?,
            TENSOR_DATA_TYPE => data_type = value.varint()?,
            TENSOR_FLOAT_DATA => value.floats(&mut float_data)?,
            TENSOR_NAME => name = value.string()?,
            TENSOR_RAW_DATA => raw_data = value.bytes()?,
            TENSOR_DATA_LOCATION => external = value.varint()? == DATA_LOCATION_EXTERNAL,
            _ => {}
        }
    }
    if external {
        return Err(format!(
            "ONNX initializer '{name}' keeps its data in another file"
        ));
    }
    let dims = dims
        .into_iter()
        .map(|dim| {
            usize::try_from(dim)
                .map_err(|_| format!("ONNX initializer '{name}' has invalid dimension {dim}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let floats = if data_type == DATA_TYPE_FLOAT {
        let values = if raw_data.is_empty() {
            float_data
        } else {
            if !raw_data.len().is_multiple_of(4) {
                return Err(format!(
                    "ONNX initializer '{name}' has {} bytes of float data",
                    raw_data.len()
                ));
            }
            raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        };
        let expected = dims
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| format!("ONNX initializer '{name}' is too large"))?;
        if values.len() != expected {
            return Err(format!(
                "ONNX initializer '{name}' holds {} values for dimensions {dims:?}",
                values.len()
            ));
        }
        Some(values)
    } else {
        None
    };
    Ok((name, OnnxTensor { dims, floats }))
}

/// One field of a protocol buffer message as it is on the wire.
enum Wire<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32([u8; 4]),
}

impl<'a> Wire<'a> {
    fn mismatch<T>(expected: &str) -> Result<T, String> {
        Err(format!("malformed ONNX file: expected {expected}"))
    }

    fn varint(&self) -> Result<u64, String> {
        match self {
            Wire::Varint(value) => Ok(*value),
            _ => Self::mismatch("an integer"),
        }
    }

    fn float(&self) -> Result<f32, String> {
        match self {
            Wire::Fixed32(bytes) => Ok(f32::from_le_bytes(*bytes)),
            _ => Self::mismatch("a float"),
        }
    }

    fn bytes(&self) -> Result<&'a [u8], String> {
        match self {
            Wire::Bytes(bytes) => Ok(bytes),
            _ => Self::mismatch("a length-delimited field"),
        }
    }

    fn string(&self) -> Result<String, String> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| "malformed ONNX file: a name is not UTF-8".to_string())
    }

    /// Appends a repeated integer field, which arrives one value at a time or packed.
    fn varints(&self, into: &mut Vec<i64>) -> Result<(), String> {
        match self {
            Wire::Varint(value) => into.push(*value as i64),
            Wire::Bytes(packed) => {
                let mut fields = Fields::new(packed);
                while !fields.is_empty() {
                    into.push(fields.varint()? as i64);
                }
            }
            _ => return Self::mismatch("integers"),
        }
        Ok(())
    }

    /// Appends a repeated float field, which arrives one value at a time or packed.
    fn floats(&self, into: &mut Vec<f32>) -> Result<(), String> {
        match self {
            Wire::Fixed32(bytes) => into.push(f32::from_le_bytes(*bytes)),
            Wire::Bytes(packed) => {
                if !packed.len().is_multiple_of(4) {
                    return Self::mismatch("packed floats");
                }
                into.extend(
                    packed
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| f32::from_le_bytes(*chunk)),
                );
            }
            _ => return Self::mismatch("floats"),
        }
        Ok(())
    }
}

/// Walks the fields of one protocol buffer message.
struct Fields<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Fields<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn is_empty(&self) -> bool {
        self.position >= self.bytes.len()
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| "malformed ONNX file: a field reaches past its message".to_string())?;
        let taken = &self.bytes[self.position..end];
        self.position = end;
        Ok(taken)
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.take(1)?[0];
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("malformed ONNX file: an integer is longer than 10 bytes".to_string())
    }

    /// The next field as its number and its value, or `None` at the end of the message.
    fn next(&mut self) -> Result<Option<(u32, Wire<'a>)>, String> {
        if self.is_empty() {
            return Ok(None);
        }
        let key = self.varint()?;
        let number = u32::try_from(key >> 3)
            .map_err(|_| "malformed ONNX file: invalid field number".to_string())?;
        let value = match key & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed64
            }
            2 => {
                let length = usize::try_from(self.varint()?)
                    .map_err(|_| "malformed ONNX file: a field is too long".to_string())?;
                Wire::Bytes(self.take(length)?)
            }
            5 => Wire::Fixed32(self.take(4)?.try_into().expect("four bytes")),
            other => {
                return Err(format!(
                    "malformed ONNX file: unsupported wire type {other}"
                ));
            }
        };
        Ok(Some((number, value)))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{OnnxAttribute, OnnxModel};

    /// Builds protocol buffer messages for the tests of the ONNX reader and its users.
    #[derive(Default)]
    pub(crate) struct Message(Vec<u8>);

    impl Message {
        fn key(&mut self, number: u32, wire: u8) {
            self.raw_varint(u64::from(number) << 3 | u64::from(wire));
        }

        fn raw_varint(&mut self, mut value: u64) {
            loop {
                let byte = (value & 0x7f) as u8;
                value >>= 7;
                if value == 0 {
                    self.0.push(byte);
                    return;
                }
                self.0.push(byte | 0x80);
            }
        }

        pub(crate) fn varint(mut self, number: u32, value: i64) -> Self {
            self.key(number, 0);
            self.raw_varint(value as u64);
            self
        }

        pub(crate) fn float(mut self, number: u32, value: f32) -> Self {
            self.key(number, 5);
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub(crate) fn bytes(mut self, number: u32, value: &[u8]) -> Self {
            self.key(number, 2);
            self.raw_varint(value.len() as u64);
            self.0.extend_from_slice(value);
            self
        }

        pub(crate) fn text(self, number: u32, value: &str) -> Self {
            self.bytes(number, value.as_bytes())
        }

        pub(crate) fn message(self, number: u32, value: Message) -> Self {
            self.bytes(number, &value.0)
        }

        pub(crate) fn packed_varints(self, number: u32, values: &[i64]) -> Self {
            let mut packed = Message::default();
            for value in values {
                packed.raw_varint(*value as u64);
            }
            self.bytes(number, &packed.0)
        }

        pub(crate) fn packed_floats(self, number: u32, values: &[f32]) -> Self {
            let packed: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            self.bytes(number, &packed)
        }

        pub(crate) fn finish(self) -> Vec<u8> {
            self.0
        }
    }

    /// An initializer of 32-bit floats, stored as `raw_data` or as `float_data`.
    pub(crate) fn float_tensor(name: &str, dims: &[i64], values: &[f32], raw: bool) -> Message {
        let tensor = Message::default()
            .packed_varints(1, dims)
            .varint(2, 1)
            .text(8, name);
        if raw {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            tensor.bytes(9, &bytes)
        } else {
            tensor.packed_floats(4, values)
        }
    }

    pub(crate) fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> Message {
        let mut node = Message::default();
        for input in inputs {
            node = node.text(1, input);
        }
        for output in outputs {
            node = node.text(2, output);
        }
        node.text(4, op_type)
    }

    pub(crate) fn ints_attribute(name: &str, values: &[i64]) -> Message {
        Message::default()
            .text(1, name)
            .packed_varints(8, values)
            .varint(20, 7)
    }

    pub(crate) fn int_attribute(name: &str, value: i64) -> Message {
        Message::default()
            .text(1, name)
            .varint(3, value)
            .varint(20, 2)
    }

    pub(crate) fn float_attribute(name: &str, value: f32) -> Message {
        Message::default()
            .text(1, name)
            .float(2, value)
            .varint(20, 1)
    }

    pub(crate) fn text_attribute(name: &str, value: &str) -> Message {
        Message::default()
            .text(1, name)
            .text(4, value)
            .varint(20, 3)
    }

    /// A model whose graph holds the given nodes and initializers.
    pub(crate) fn model(nodes: Vec<Message>, initializers: Vec<Message>) -> Vec<u8> {
        let mut graph = Message::default();
        for node in nodes {
            graph = graph.message(1, node);
        }
        for initializer in initializers {
            graph = graph.message(5, initializer);
        }
        Message::default()
            .varint(1, 6)
            .text(2, "test")
            .message(7, graph)
            .finish()
    }

    #[test]
    fn reads_nodes_attributes_and_both_float_storages() {
        let bytes = model(
            vec![
                node("Conv", &["input", "w"], &["conv"])
                    .message(5, ints_attribute("kernel_shape", &[3, 3]))
                    .message(5, int_attribute("group", 16))
                    .message(5, float_attribute("epsilon", 0.5))
                    .message(5, text_attribute("mode", "nearest")),
                node("Relu", &["conv"], &["out"]),
            ],
            vec![
                float_tensor("w", &[1, 2], &[1.5, -2.0], true),
                float_tensor("b", &[2], &[0.25, 4.0], false),
                // An integer tensor, as a Reshape carries its target shape.
                super::tests::Message::default()
                    .packed_varints(1, &[2])
                    .varint(2, 7)
                    .text(8, "shape")
                    .bytes(9, &[0; 16]),
            ],
        );

        let model = OnnxModel::parse(&bytes).unwrap();

        assert_eq!(model.nodes.len(), 2);
        let conv = &model.nodes[0];
        assert_eq!(conv.op_type, "Conv");
        assert_eq!(conv.inputs, ["input", "w"]);
        assert_eq!(conv.outputs, ["conv"]);
        assert_eq!(
            conv.attributes["kernel_shape"],
            OnnxAttribute::Ints(vec![3, 3])
        );
        assert_eq!(conv.attributes["group"], OnnxAttribute::Int(16));
        assert_eq!(conv.attributes["epsilon"], OnnxAttribute::Float(0.5));
        assert_eq!(
            conv.attributes["mode"],
            OnnxAttribute::Bytes(b"nearest".to_vec())
        );
        assert_eq!(model.floats("w").unwrap(), (&[1.5, -2.0][..], &[1, 2][..]));
        assert_eq!(model.floats("b").unwrap(), (&[0.25, 4.0][..], &[2][..]));
        assert!(model.floats("shape").is_err());
        assert!(model.floats("missing").is_err());
    }

    #[test]
    fn reads_repeated_fields_that_are_not_packed() {
        // dims 2 and 3 as two separate fields, float_data as six separate fields.
        let mut tensor = Message::default().varint(1, 2).varint(1, 3).varint(2, 1);
        for value in 0..6 {
            tensor = tensor.float(4, value as f32);
        }
        let bytes = model(vec![], vec![tensor.text(8, "t")]);

        let model = OnnxModel::parse(&bytes).unwrap();

        assert_eq!(
            model.floats("t").unwrap(),
            (&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0][..], &[2, 3][..])
        );
    }

    #[test]
    fn refuses_files_that_are_cut_short_or_inconsistent() {
        let bytes = model(
            vec![node("Relu", &["a"], &["b"])],
            vec![float_tensor("w", &[2], &[1.0, 2.0], true)],
        );
        assert!(OnnxModel::parse(&bytes[..bytes.len() - 3]).is_err());
        assert!(OnnxModel::parse(b"not an onnx file").is_err());
        assert!(OnnxModel::parse(&[]).is_err());

        // Three values for dimensions that hold two.
        let wrong_count = model(
            vec![],
            vec![float_tensor("w", &[2], &[1.0, 2.0, 3.0], true)],
        );
        assert!(OnnxModel::parse(&wrong_count).is_err());

        // Data kept in another file.
        let external = model(
            vec![],
            vec![float_tensor("w", &[0], &[], true).varint(14, 1)],
        );
        assert!(OnnxModel::parse(&external).is_err());
    }
}
