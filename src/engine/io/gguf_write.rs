//! Writer for GGUF files that hold metadata and 32-bit float tensors, as the face
//! converters produce them. Metadata keeps the order it is added in, and the file is laid
//! out the way the reference `gguf` Python package lays it out.

use crate::engine::types::{
    GGML_TYPE_F32, GGUF_MAGIC, GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32, GGUF_TYPE_INT32,
    GGUF_TYPE_STRING, GGUF_TYPE_UINT32,
};

const GGUF_VERSION: u32 = 3;
/// Tensor data starts on a multiple of this, and so does every tensor inside it.
const ALIGNMENT: usize = 32;

struct Tensor {
    name: String,
    /// Outermost dimension first, as the array is indexed.
    shape: Vec<usize>,
    values: Vec<f32>,
}

pub(crate) struct GgufWriter {
    metadata: Vec<u8>,
    metadata_count: u64,
    tensors: Vec<Tensor>,
}

impl GgufWriter {
    /// A file whose first metadata entry is `general.architecture`.
    pub(crate) fn new(architecture: &str) -> Self {
        let mut writer = Self {
            metadata: Vec::new(),
            metadata_count: 0,
            tensors: Vec::new(),
        };
        writer.add_string("general.architecture", architecture);
        writer
    }

    fn key(&mut self, key: &str, value_type: u32) {
        push_string(&mut self.metadata, key);
        self.metadata.extend_from_slice(&value_type.to_le_bytes());
        self.metadata_count += 1;
    }

    pub(crate) fn add_string(&mut self, key: &str, value: &str) {
        self.key(key, GGUF_TYPE_STRING);
        push_string(&mut self.metadata, value);
    }

    pub(crate) fn add_u32(&mut self, key: &str, value: u32) {
        self.key(key, GGUF_TYPE_UINT32);
        self.metadata.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn add_f32(&mut self, key: &str, value: f32) {
        self.key(key, GGUF_TYPE_FLOAT32);
        self.metadata.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn add_i32_array(&mut self, key: &str, values: &[i32]) {
        self.key(key, GGUF_TYPE_ARRAY);
        self.metadata
            .extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
        self.metadata
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            self.metadata.extend_from_slice(&value.to_le_bytes());
        }
    }

    pub(crate) fn add_f32_array(&mut self, key: &str, values: &[f32]) {
        self.key(key, GGUF_TYPE_ARRAY);
        self.metadata
            .extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
        self.metadata
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            self.metadata.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Adds a tensor of 32-bit floats. `shape` names the outermost dimension first, and
    /// `values` holds the array in row-major order.
    pub(crate) fn add_f32_tensor(&mut self, name: &str, shape: &[usize], values: Vec<f32>) {
        debug_assert_eq!(shape.iter().product::<usize>(), values.len());
        self.tensors.push(Tensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            values,
        });
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        let mut file = Vec::new();
        file.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        file.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        file.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        file.extend_from_slice(&self.metadata_count.to_le_bytes());
        file.extend_from_slice(&self.metadata);

        let mut offset = 0usize;
        for tensor in &self.tensors {
            push_string(&mut file, &tensor.name);
            file.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            // GGUF lists the innermost dimension first.
            for dim in tensor.shape.iter().rev() {
                file.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            file.extend_from_slice(&(GGML_TYPE_F32 as u32).to_le_bytes());
            file.extend_from_slice(&(offset as u64).to_le_bytes());
            offset = (offset + tensor.values.len() * 4).next_multiple_of(ALIGNMENT);
        }

        file.resize(file.len().next_multiple_of(ALIGNMENT), 0);
        for tensor in &self.tensors {
            for value in &tensor.values {
                file.extend_from_slice(&value.to_le_bytes());
            }
            file.resize(file.len().next_multiple_of(ALIGNMENT), 0);
        }
        file
    }
}

fn push_string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::GgufWriter;
    use crate::engine::io::{
        find_gguf_tensor, get_gguf_f32_array_from_map, get_gguf_float_from_map,
        get_gguf_i64_array_from_map, get_gguf_int_from_map, get_gguf_string_from_map,
        parse_gguf_from_bytes,
    };

    #[test]
    fn a_written_file_is_read_back_by_the_parser() {
        let mut writer = GgufWriter::new("face_test");
        writer.add_string("general.name", "Test");
        writer.add_u32("face_test.count", 7);
        writer.add_f32("face_test.scale", 0.5);
        writer.add_i32_array("face_test.ops", &[1, -2, 3]);
        writer.add_f32_array("face_test.points", &[1.5, 2.5]);
        writer.add_f32_tensor("first", &[2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        writer.add_f32_tensor("second", &[1], vec![9.0]);
        let bytes: &'static [u8] = Box::leak(writer.finish().into_boxed_slice());

        let file = parse_gguf_from_bytes(bytes, false).unwrap();

        assert_eq!(
            get_gguf_string_from_map(&file.kv, "general.architecture"),
            Some("face_test")
        );
        assert_eq!(
            get_gguf_string_from_map(&file.kv, "general.name"),
            Some("Test")
        );
        assert_eq!(get_gguf_int_from_map(&file.kv, "face_test.count", 0), 7);
        assert_eq!(
            get_gguf_float_from_map(&file.kv, "face_test.scale", 0.0),
            0.5
        );
        assert_eq!(
            get_gguf_i64_array_from_map(&file.kv, "face_test.ops"),
            Some(&[1, -2, 3][..])
        );
        assert_eq!(
            get_gguf_f32_array_from_map(&file.kv, "face_test.points"),
            Some(&[1.5, 2.5][..])
        );
        let first = find_gguf_tensor(&file, "first").unwrap();
        assert_eq!(first.n_dims, 2);
        assert_eq!(first.ne[..2], [3, 2]);
        assert!(find_gguf_tensor(&file, "second").is_some());
    }

    #[test]
    fn tensor_data_starts_aligned_and_each_tensor_is_padded() {
        let mut writer = GgufWriter::new("a");
        writer.add_f32_tensor("t", &[3], vec![1.0, 2.0, 3.0]);
        writer.add_f32_tensor("u", &[1], vec![4.0]);
        let bytes = writer.finish();

        assert_eq!(bytes.len() % 32, 0);
        // Two tensors of 12 and 4 bytes, each padded to 32.
        let data = &bytes[bytes.len() - 64..];
        assert_eq!(data[..4], 1.0f32.to_le_bytes());
        assert_eq!(data[12..32], [0; 20]);
        assert_eq!(data[32..36], 4.0f32.to_le_bytes());
    }
}
