use crate::engine::io::{bf16_to_fp32, fp16_to_fp32};

pub(crate) const DISTRIBUTED_PROTOCOL_MAGIC: u32 = 0x444D_4F45;
pub(crate) const DISTRIBUTED_PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum FrameKind {
    Hello = 1,
    Ready = 2,
    ExpertBatchRequest = 3,
    ExpertBatchResponse = 4,
    Error = 5,
    Shutdown = 6,
}

impl FrameKind {
    pub(crate) fn from_u16(value: u16) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Hello),
            2 => Ok(Self::Ready),
            3 => Ok(Self::ExpertBatchRequest),
            4 => Ok(Self::ExpertBatchResponse),
            5 => Ok(Self::Error),
            6 => Ok(Self::Shutdown),
            _ => Err(format!("unknown distributed frame kind {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum ActivationDtype {
    Fp16 = 1,
    Bf16 = 2,
}

impl ActivationDtype {
    pub(crate) fn from_u16(value: u16) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            _ => Err(format!("unknown activation dtype {value}")),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HelloFrame {
    pub(crate) dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) activation_dtype: ActivationDtype,
}

#[derive(Clone, Debug)]
pub(crate) struct ReadyFrame {
    pub(crate) dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) activation_dtype: ActivationDtype,
}

#[derive(Clone, Debug)]
pub(crate) struct ExpertBatchRequest {
    pub(crate) token_pos: usize,
    pub(crate) layer: usize,
    pub(crate) activation_dtype: ActivationDtype,
    pub(crate) dim: usize,
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) activation: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExpertBatchResponse {
    pub(crate) layer: usize,
    pub(crate) output_dtype: ActivationDtype,
    pub(crate) dim: usize,
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) outputs: Vec<Vec<f32>>,
}

fn write_u16_le(dst: &mut Vec<u8>, value: u16) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn write_u32_le(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn write_u64_le(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn read_u16_le(src: &[u8], offset: &mut usize) -> Result<u16, String> {
    if *offset + 2 > src.len() {
        return Err("truncated u16 payload".to_string());
    }
    let value = u16::from_le_bytes([src[*offset], src[*offset + 1]]);
    *offset += 2;
    Ok(value)
}

fn read_u32_le(src: &[u8], offset: &mut usize) -> Result<u32, String> {
    if *offset + 4 > src.len() {
        return Err("truncated u32 payload".to_string());
    }
    let value = u32::from_le_bytes([
        src[*offset],
        src[*offset + 1],
        src[*offset + 2],
        src[*offset + 3],
    ]);
    *offset += 4;
    Ok(value)
}

fn read_u64_le(src: &[u8], offset: &mut usize) -> Result<u64, String> {
    if *offset + 8 > src.len() {
        return Err("truncated u64 payload".to_string());
    }
    let value = u64::from_le_bytes([
        src[*offset],
        src[*offset + 1],
        src[*offset + 2],
        src[*offset + 3],
        src[*offset + 4],
        src[*offset + 5],
        src[*offset + 6],
        src[*offset + 7],
    ]);
    *offset += 8;
    Ok(value)
}

fn encode_fp32_scalar_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn encode_fp32_scalar_to_fp16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ff_ff;

    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7c00 | ((mant >> 13) as u16).max(1);
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        if (mantissa & round_bit) != 0
            && ((mantissa & (round_bit - 1)) != 0 || (half_mant & 1) != 0)
        {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half_mant = (mant >> 13) as u16;
    let round_bits = mant & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (half_mant & 1) != 0) {
        half_mant = half_mant.wrapping_add(1);
        if half_mant == 0x0400 {
            return sign | (((half_exp + 1) as u16) << 10);
        }
    }

    sign | ((half_exp as u16) << 10) | half_mant
}

pub(crate) fn encode_activation_vector(values: &[f32], dtype: ActivationDtype) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        let bits = match dtype {
            ActivationDtype::Fp16 => encode_fp32_scalar_to_fp16_bits(value),
            ActivationDtype::Bf16 => encode_fp32_scalar_to_bf16_bits(value),
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

pub(crate) fn decode_activation_vector(
    bytes: &[u8],
    dtype: ActivationDtype,
    dim: usize,
) -> Result<Vec<f32>, String> {
    if bytes.len() != dim.saturating_mul(2) {
        return Err(format!(
            "activation payload size mismatch: got {} bytes, expected {}",
            bytes.len(),
            dim.saturating_mul(2)
        ));
    }
    let mut out = Vec::with_capacity(dim);
    let mut offset = 0usize;
    while offset < bytes.len() {
        let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let value = match dtype {
            ActivationDtype::Fp16 => fp16_to_fp32(bits),
            ActivationDtype::Bf16 => bf16_to_fp32(bits),
        };
        out.push(value);
        offset += 2;
    }
    Ok(out)
}

pub(crate) fn encode_hello_frame(frame: &HelloFrame) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(16);
    write_u32_le(
        &mut out,
        frame
            .dim
            .try_into()
            .map_err(|_| "dim overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .n_layers
            .try_into()
            .map_err(|_| "n_layers overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .n_experts
            .try_into()
            .map_err(|_| "n_experts overflow".to_string())?,
    );
    write_u16_le(&mut out, frame.activation_dtype as u16);
    Ok(out)
}

pub(crate) fn decode_hello_frame(payload: &[u8]) -> Result<HelloFrame, String> {
    let mut offset = 0usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let n_layers = read_u32_le(payload, &mut offset)? as usize;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let activation_dtype = ActivationDtype::from_u16(read_u16_le(payload, &mut offset)?)?;
    Ok(HelloFrame {
        dim,
        n_layers,
        n_experts,
        activation_dtype,
    })
}

pub(crate) fn encode_ready_frame(frame: &ReadyFrame) -> Result<Vec<u8>, String> {
    encode_hello_frame(&HelloFrame {
        dim: frame.dim,
        n_layers: frame.n_layers,
        n_experts: frame.n_experts,
        activation_dtype: frame.activation_dtype,
    })
}

pub(crate) fn decode_ready_frame(payload: &[u8]) -> Result<ReadyFrame, String> {
    let hello = decode_hello_frame(payload)?;
    Ok(ReadyFrame {
        dim: hello.dim,
        n_layers: hello.n_layers,
        n_experts: hello.n_experts,
        activation_dtype: hello.activation_dtype,
    })
}

pub(crate) fn encode_expert_batch_request(frame: &ExpertBatchRequest) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write_u64_le(
        &mut out,
        frame
            .token_pos
            .try_into()
            .map_err(|_| "token_pos overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .layer
            .try_into()
            .map_err(|_| "layer overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .dim
            .try_into()
            .map_err(|_| "dim overflow".to_string())?,
    );
    write_u16_le(&mut out, frame.activation_dtype as u16);
    write_u32_le(
        &mut out,
        frame
            .expert_ids
            .len()
            .try_into()
            .map_err(|_| "expert count overflow".to_string())?,
    );
    for &expert_id in &frame.expert_ids {
        write_u32_le(
            &mut out,
            expert_id
                .try_into()
                .map_err(|_| "expert id overflow".to_string())?,
        );
    }
    out.extend_from_slice(&encode_activation_vector(
        &frame.activation,
        frame.activation_dtype,
    ));
    Ok(out)
}

pub(crate) fn decode_expert_batch_request(payload: &[u8]) -> Result<ExpertBatchRequest, String> {
    let mut offset = 0usize;
    let token_pos = read_u64_le(payload, &mut offset)? as usize;
    let layer = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let activation_dtype = ActivationDtype::from_u16(read_u16_le(payload, &mut offset)?)?;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let mut expert_ids = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        expert_ids.push(read_u32_le(payload, &mut offset)? as usize);
    }
    let activation = decode_activation_vector(&payload[offset..], activation_dtype, dim)?;
    Ok(ExpertBatchRequest {
        token_pos,
        layer,
        activation_dtype,
        dim,
        expert_ids,
        activation,
    })
}

pub(crate) fn encode_expert_batch_response(frame: &ExpertBatchResponse) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    if frame.outputs.len() != frame.expert_ids.len() {
        return Err("expert response output count does not match expert id count".to_string());
    }
    write_u32_le(
        &mut out,
        frame
            .layer
            .try_into()
            .map_err(|_| "layer overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .dim
            .try_into()
            .map_err(|_| "dim overflow".to_string())?,
    );
    write_u16_le(&mut out, frame.output_dtype as u16);
    write_u32_le(
        &mut out,
        frame
            .expert_ids
            .len()
            .try_into()
            .map_err(|_| "expert count overflow".to_string())?,
    );
    for &expert_id in &frame.expert_ids {
        write_u32_le(
            &mut out,
            expert_id
                .try_into()
                .map_err(|_| "expert id overflow".to_string())?,
        );
    }
    for output in &frame.outputs {
        if output.len() != frame.dim {
            return Err(format!(
                "expert response vector length mismatch: got {}, expected {}",
                output.len(),
                frame.dim
            ));
        }
        out.extend_from_slice(&encode_activation_vector(output, frame.output_dtype));
    }
    Ok(out)
}

pub(crate) fn decode_expert_batch_response(payload: &[u8]) -> Result<ExpertBatchResponse, String> {
    let mut offset = 0usize;
    let layer = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let output_dtype = ActivationDtype::from_u16(read_u16_le(payload, &mut offset)?)?;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let mut expert_ids = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        expert_ids.push(read_u32_le(payload, &mut offset)? as usize);
    }
    let bytes_per_output = dim.saturating_mul(2);
    let mut outputs = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        let end = offset
            .checked_add(bytes_per_output)
            .ok_or_else(|| "response payload overflow".to_string())?;
        if end > payload.len() {
            return Err("truncated expert response payload".to_string());
        }
        outputs.push(decode_activation_vector(
            &payload[offset..end],
            output_dtype,
            dim,
        )?);
        offset = end;
    }
    Ok(ExpertBatchResponse {
        layer,
        output_dtype,
        dim,
        expert_ids,
        outputs,
    })
}

pub(crate) fn encode_error_frame(message: &str) -> Result<Vec<u8>, String> {
    let bytes = message.as_bytes();
    let len: u32 = bytes
        .len()
        .try_into()
        .map_err(|_| "distributed error message too large".to_string())?;
    let mut out = Vec::with_capacity(4 + bytes.len());
    write_u32_le(&mut out, len);
    out.extend_from_slice(bytes);
    Ok(out)
}

pub(crate) fn decode_error_frame(payload: &[u8]) -> Result<String, String> {
    let mut offset = 0usize;
    let len = read_u32_le(payload, &mut offset)? as usize;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| "distributed error payload overflow".to_string())?;
    if end > payload.len() {
        return Err("truncated distributed error payload".to_string());
    }
    String::from_utf8(payload[offset..end].to_vec())
        .map_err(|e| format!("distributed error payload was not valid utf-8: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip_has_expected_length() {
        let values = [0.0f32, 1.5, -2.25, 7.0];
        let encoded = encode_activation_vector(&values, ActivationDtype::Bf16);
        assert_eq!(encoded.len(), values.len() * 2);
        let decoded = decode_activation_vector(&encoded, ActivationDtype::Bf16, values.len())
            .expect("decode failed");
        assert_eq!(decoded.len(), values.len());
    }

    #[test]
    fn expert_batch_request_round_trip() {
        let request = ExpertBatchRequest {
            token_pos: 17,
            layer: 3,
            activation_dtype: ActivationDtype::Bf16,
            dim: 4,
            expert_ids: vec![1, 9],
            activation: vec![0.25, -0.5, 1.0, 2.0],
        };
        let payload = encode_expert_batch_request(&request).expect("encode failed");
        let decoded = decode_expert_batch_request(&payload).expect("decode failed");
        assert_eq!(decoded.token_pos, request.token_pos);
        assert_eq!(decoded.layer, request.layer);
        assert_eq!(decoded.dim, request.dim);
        assert_eq!(decoded.expert_ids, request.expert_ids);
        assert_eq!(decoded.activation.len(), request.activation.len());
    }
}
