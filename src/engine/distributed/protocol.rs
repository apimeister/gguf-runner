use crate::engine::io::{bf16_to_fp32, fp16_to_fp32};
use crate::engine::kernels::{encode_bf16_vector, encode_fp16_vector};

pub(crate) const DISTRIBUTED_PROTOCOL_MAGIC: u32 = 0x444D_4F45;
pub(crate) const DISTRIBUTED_PROTOCOL_VERSION: u16 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum FrameKind {
    Hello = 1,
    Ready = 2,
    ExpertBatchRequest = 3,
    ExpertBatchResponse = 4,
    Error = 5,
    Shutdown = 6,
    DiscoverRequest = 7,
    DiscoverResponse = 8,
    ModelShardHeader = 9,
    SsmLayerRequest = 10,
    SsmLayerResponse = 11,
    SpeculationAdvance = 12,
    ProposedExpertBatch = 13,
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
            7 => Ok(Self::DiscoverRequest),
            8 => Ok(Self::DiscoverResponse),
            9 => Ok(Self::ModelShardHeader),
            10 => Ok(Self::SsmLayerRequest),
            11 => Ok(Self::SsmLayerResponse),
            12 => Ok(Self::SpeculationAdvance),
            13 => Ok(Self::ProposedExpertBatch),
            _ => Err(format!("unknown distributed frame kind {value}")),
        }
    }
}

pub(crate) const SHARD_KIND_SSM_QKV_IN: u8 = 3;
pub(crate) const SHARD_KIND_SSM_GATE: u8 = 4;
pub(crate) const SHARD_KIND_SSM_OUT: u8 = 5;
pub(crate) const SHARD_KIND_SSM_BA: u8 = 6;
pub(crate) const SHARD_KIND_SSM_ALPHA: u8 = 7;
pub(crate) const SHARD_KIND_SSM_BETA: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum ActivationDtype {
    Fp16 = 1,
    Bf16 = 2,
    Q8 = 3,
}

impl ActivationDtype {
    pub(crate) fn from_u16(value: u16) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Q8),
            _ => Err(format!("unknown activation dtype {value}")),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HelloFrame {
    pub(crate) node_address: String,
    pub(crate) dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) activation_dtype: ActivationDtype,
    pub(crate) assigned_experts: Vec<(usize, usize)>,
    pub(crate) speculative_push: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ReadyFrame {
    pub(crate) node_address: String,
    pub(crate) dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) activation_dtype: ActivationDtype,
    pub(crate) speculative_push: bool,
    pub(crate) logical_cpu_count: usize,
    pub(crate) memory_bytes: usize,
    pub(crate) loaded_expert_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoverResponseFrame {
    pub(crate) node_address: String,
    pub(crate) dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) logical_cpu_count: usize,
    pub(crate) memory_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ExpertBatchRequest {
    pub(crate) token_pos: usize,
    pub(crate) layer: usize,
    pub(crate) activation_dtype: ActivationDtype,
    pub(crate) dim: usize,
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) activation: Vec<f32>,
    /// Coordinator-predicted expert IDs for layer+1 (empty = no hint).
    pub(crate) hint_expert_ids: Vec<usize>,
}

/// Coordinator -> worker: legacy speculative-control frame.
/// Runtime code no longer sends this because future expert outputs require exact
/// future-layer activations.
#[derive(Clone, Debug)]
pub(crate) struct SpeculationAdvanceFrame {
    pub(crate) layer: usize,
    pub(crate) activation_dtype: ActivationDtype,
    pub(crate) dim: usize,
    pub(crate) activation: Vec<f32>,
    pub(crate) hint_expert_ids: Vec<usize>,
}

/// Worker -> coordinator: legacy proactive expert-output frame.
/// Runtime code ignores these frames because stale-activation outputs are unsafe.
#[derive(Clone, Debug)]
pub(crate) struct ProposedExpertBatchFrame {
    pub(crate) layer: usize,
    pub(crate) output_dtype: ActivationDtype,
    pub(crate) dim: usize,
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) outputs: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExpertBatchResponse {
    pub(crate) layer: usize,
    pub(crate) output_dtype: ActivationDtype,
    pub(crate) dim: usize,
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) outputs: Vec<Vec<f32>>,
    pub(crate) spec_hit: bool,
}

fn write_f32_le(dst: &mut Vec<u8>, value: f32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn read_f32_le(src: &[u8], offset: &mut usize) -> Result<f32, String> {
    if *offset + 4 > src.len() {
        return Err("truncated f32 payload".to_string());
    }
    let value = f32::from_le_bytes([
        src[*offset],
        src[*offset + 1],
        src[*offset + 2],
        src[*offset + 3],
    ]);
    *offset += 4;
    Ok(value)
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

fn write_string(dst: &mut Vec<u8>, value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    let len: u16 = bytes
        .len()
        .try_into()
        .map_err(|_| "string payload too large".to_string())?;
    write_u16_le(dst, len);
    dst.extend_from_slice(bytes);
    Ok(())
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

fn read_string(src: &[u8], offset: &mut usize) -> Result<String, String> {
    let len = read_u16_le(src, offset)? as usize;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| "string payload overflow".to_string())?;
    if end > src.len() {
        return Err("truncated string payload".to_string());
    }
    let value = String::from_utf8(src[*offset..end].to_vec())
        .map_err(|e| format!("string payload was not valid utf-8: {e}"))?;
    *offset = end;
    Ok(value)
}

pub(crate) fn encode_activation_vector(values: &[f32], dtype: ActivationDtype) -> Vec<u8> {
    match dtype {
        ActivationDtype::Fp16 | ActivationDtype::Bf16 => {
            let mut out = vec![0u8; values.len() * 2];
            match dtype {
                ActivationDtype::Fp16 => encode_fp16_vector(values, &mut out),
                ActivationDtype::Bf16 => encode_bf16_vector(values, &mut out),
                ActivationDtype::Q8 => unreachable!(),
            }
            out
        }
        ActivationDtype::Q8 => {
            let max_abs = values
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let mut out = Vec::with_capacity(4 + values.len());
            out.extend_from_slice(&scale.to_le_bytes());
            for &value in values {
                let quantized = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
                out.push(quantized as u8);
            }
            out
        }
    }
}

pub(crate) fn activation_vector_encoded_len(dim: usize, dtype: ActivationDtype) -> usize {
    match dtype {
        ActivationDtype::Fp16 | ActivationDtype::Bf16 => dim.saturating_mul(2),
        ActivationDtype::Q8 => 4usize.saturating_add(dim),
    }
}

pub(crate) fn decode_activation_vector(
    bytes: &[u8],
    dtype: ActivationDtype,
    dim: usize,
) -> Result<Vec<f32>, String> {
    let expected_len = activation_vector_encoded_len(dim, dtype);
    if bytes.len() != expected_len {
        return Err(format!(
            "activation payload size mismatch: got {} bytes, expected {}",
            bytes.len(),
            expected_len
        ));
    }
    match dtype {
        ActivationDtype::Fp16 | ActivationDtype::Bf16 => {
            let mut out = Vec::with_capacity(dim);
            let mut offset = 0usize;
            while offset < bytes.len() {
                let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                let value = match dtype {
                    ActivationDtype::Fp16 => fp16_to_fp32(bits),
                    ActivationDtype::Bf16 => bf16_to_fp32(bits),
                    ActivationDtype::Q8 => unreachable!(),
                };
                out.push(value);
                offset += 2;
            }
            Ok(out)
        }
        ActivationDtype::Q8 => {
            let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let mut out = Vec::with_capacity(dim);
            for &quantized in &bytes[4..] {
                out.push((quantized as i8) as f32 * scale);
            }
            Ok(out)
        }
    }
}

pub(crate) fn encode_hello_frame(frame: &HelloFrame) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write_string(&mut out, &frame.node_address)?;
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
    write_u32_le(
        &mut out,
        frame
            .assigned_experts
            .len()
            .try_into()
            .map_err(|_| "assigned expert pair count overflow".to_string())?,
    );
    for &(layer, expert_idx) in &frame.assigned_experts {
        write_u32_le(
            &mut out,
            layer
                .try_into()
                .map_err(|_| "assigned expert layer overflow".to_string())?,
        );
        write_u32_le(
            &mut out,
            expert_idx
                .try_into()
                .map_err(|_| "assigned expert index overflow".to_string())?,
        );
    }
    out.push(frame.speculative_push as u8);
    Ok(out)
}

pub(crate) fn decode_hello_frame(payload: &[u8]) -> Result<HelloFrame, String> {
    let mut offset = 0usize;
    let node_address = read_string(payload, &mut offset)?;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let n_layers = read_u32_le(payload, &mut offset)? as usize;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let activation_dtype = ActivationDtype::from_u16(read_u16_le(payload, &mut offset)?)?;
    let pair_count = read_u32_le(payload, &mut offset)? as usize;
    let mut assigned_experts = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        assigned_experts.push((
            read_u32_le(payload, &mut offset)? as usize,
            read_u32_le(payload, &mut offset)? as usize,
        ));
    }
    let speculative_push = if offset < payload.len() {
        payload[offset] != 0
    } else {
        false
    };
    Ok(HelloFrame {
        node_address,
        dim,
        n_layers,
        n_experts,
        activation_dtype,
        assigned_experts,
        speculative_push,
    })
}

pub(crate) fn encode_ready_frame(frame: &ReadyFrame) -> Result<Vec<u8>, String> {
    let mut out = encode_hello_frame(&HelloFrame {
        node_address: frame.node_address.clone(),
        dim: frame.dim,
        n_layers: frame.n_layers,
        n_experts: frame.n_experts,
        activation_dtype: frame.activation_dtype,
        assigned_experts: Vec::new(),
        speculative_push: frame.speculative_push,
    })?;
    write_u32_le(
        &mut out,
        frame
            .logical_cpu_count
            .try_into()
            .map_err(|_| "logical_cpu_count overflow".to_string())?,
    );
    write_u64_le(
        &mut out,
        frame
            .memory_bytes
            .try_into()
            .map_err(|_| "memory_bytes overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .loaded_expert_count
            .try_into()
            .map_err(|_| "loaded_expert_count overflow".to_string())?,
    );
    Ok(out)
}

pub(crate) fn decode_ready_frame(payload: &[u8]) -> Result<ReadyFrame, String> {
    let hello = decode_hello_frame(payload)?;
    let mut offset = 0usize;
    let _ = read_string(payload, &mut offset)?;
    let _ = read_u32_le(payload, &mut offset)?;
    let _ = read_u32_le(payload, &mut offset)?;
    let _ = read_u32_le(payload, &mut offset)?;
    let _ = read_u16_le(payload, &mut offset)?;
    let pair_count = read_u32_le(payload, &mut offset)? as usize;
    for _ in 0..pair_count {
        let _ = read_u32_le(payload, &mut offset)?;
        let _ = read_u32_le(payload, &mut offset)?;
    }
    let speculative_push = if offset < payload.len() {
        let value = payload[offset] != 0;
        offset += 1;
        value
    } else {
        false
    };
    let logical_cpu_count = read_u32_le(payload, &mut offset)? as usize;
    let memory_bytes = read_u64_le(payload, &mut offset)? as usize;
    let loaded_expert_count = read_u32_le(payload, &mut offset)? as usize;
    Ok(ReadyFrame {
        node_address: hello.node_address,
        dim: hello.dim,
        n_layers: hello.n_layers,
        n_experts: hello.n_experts,
        activation_dtype: hello.activation_dtype,
        speculative_push,
        logical_cpu_count,
        memory_bytes,
        loaded_expert_count,
    })
}

pub(crate) fn encode_discover_response_frame(
    frame: &DiscoverResponseFrame,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write_string(&mut out, &frame.node_address)?;
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
    write_u32_le(
        &mut out,
        frame
            .logical_cpu_count
            .try_into()
            .map_err(|_| "logical_cpu_count overflow".to_string())?,
    );
    write_u64_le(
        &mut out,
        frame
            .memory_bytes
            .try_into()
            .map_err(|_| "memory_bytes overflow".to_string())?,
    );
    Ok(out)
}

pub(crate) fn decode_discover_response_frame(
    payload: &[u8],
) -> Result<DiscoverResponseFrame, String> {
    let mut offset = 0usize;
    let node_address = read_string(payload, &mut offset)?;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let n_layers = read_u32_le(payload, &mut offset)? as usize;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let logical_cpu_count = read_u32_le(payload, &mut offset)? as usize;
    let memory_bytes = read_u64_le(payload, &mut offset)? as usize;
    Ok(DiscoverResponseFrame {
        node_address,
        dim,
        n_layers,
        n_experts,
        logical_cpu_count,
        memory_bytes,
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
    // Append hint (u32 count + u32 expert ids); count=0 means no hint
    write_u32_le(
        &mut out,
        frame
            .hint_expert_ids
            .len()
            .try_into()
            .map_err(|_| "hint count overflow".to_string())?,
    );
    for &id in &frame.hint_expert_ids {
        write_u32_le(
            &mut out,
            id.try_into()
                .map_err(|_| "hint expert id overflow".to_string())?,
        );
    }
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
    let act_len = activation_vector_encoded_len(dim, activation_dtype);
    let act_end = offset
        .checked_add(act_len)
        .ok_or_else(|| "activation overflow".to_string())?;
    if act_end > payload.len() {
        return Err("truncated expert batch request activation".to_string());
    }
    let activation = decode_activation_vector(&payload[offset..act_end], activation_dtype, dim)?;
    offset = act_end;
    // Hint (optional, backward compat: empty if no bytes remain)
    let hint_expert_ids = if offset + 4 <= payload.len() {
        let n_hint = read_u32_le(payload, &mut offset)? as usize;
        let mut ids = Vec::with_capacity(n_hint);
        for _ in 0..n_hint {
            ids.push(read_u32_le(payload, &mut offset)? as usize);
        }
        ids
    } else {
        Vec::new()
    };
    Ok(ExpertBatchRequest {
        token_pos,
        layer,
        activation_dtype,
        dim,
        expert_ids,
        activation,
        hint_expert_ids,
    })
}

#[cfg(test)]
pub(crate) fn encode_speculation_advance(
    frame: &SpeculationAdvanceFrame,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
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
            .hint_expert_ids
            .len()
            .try_into()
            .map_err(|_| "hint count overflow".to_string())?,
    );
    for &id in &frame.hint_expert_ids {
        write_u32_le(
            &mut out,
            id.try_into()
                .map_err(|_| "hint expert id overflow".to_string())?,
        );
    }
    out.extend_from_slice(&encode_activation_vector(
        &frame.activation,
        frame.activation_dtype,
    ));
    Ok(out)
}

pub(crate) fn decode_speculation_advance(
    payload: &[u8],
) -> Result<SpeculationAdvanceFrame, String> {
    let mut offset = 0usize;
    let layer = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let activation_dtype = ActivationDtype::from_u16(read_u16_le(payload, &mut offset)?)?;
    let n_hint = read_u32_le(payload, &mut offset)? as usize;
    let mut hint_expert_ids = Vec::with_capacity(n_hint);
    for _ in 0..n_hint {
        hint_expert_ids.push(read_u32_le(payload, &mut offset)? as usize);
    }
    let activation = decode_activation_vector(&payload[offset..], activation_dtype, dim)?;
    Ok(SpeculationAdvanceFrame {
        layer,
        activation_dtype,
        dim,
        activation,
        hint_expert_ids,
    })
}

#[cfg(test)]
pub(crate) fn encode_proposed_expert_batch(
    frame: &ProposedExpertBatchFrame,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    if frame.outputs.len() != frame.expert_ids.len() {
        return Err("proposed expert output count does not match expert id count".to_string());
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
                "proposed expert vector length mismatch: got {}, expected {}",
                output.len(),
                frame.dim
            ));
        }
        out.extend_from_slice(&encode_activation_vector(output, frame.output_dtype));
    }
    Ok(out)
}

pub(crate) fn decode_proposed_expert_batch(
    payload: &[u8],
) -> Result<ProposedExpertBatchFrame, String> {
    let mut offset = 0usize;
    let layer = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let output_dtype = ActivationDtype::from_u16(read_u16_le(payload, &mut offset)?)?;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let mut expert_ids = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        expert_ids.push(read_u32_le(payload, &mut offset)? as usize);
    }
    let bytes_per_output = activation_vector_encoded_len(dim, output_dtype);
    let mut outputs = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        let end = offset
            .checked_add(bytes_per_output)
            .ok_or_else(|| "proposed expert payload overflow".to_string())?;
        if end > payload.len() {
            return Err("truncated proposed expert payload".to_string());
        }
        outputs.push(decode_activation_vector(
            &payload[offset..end],
            output_dtype,
            dim,
        )?);
        offset = end;
    }
    Ok(ProposedExpertBatchFrame {
        layer,
        output_dtype,
        dim,
        expert_ids,
        outputs,
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
    out.push(frame.spec_hit as u8);
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
    let bytes_per_output = activation_vector_encoded_len(dim, output_dtype);
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
    let spec_hit = if offset < payload.len() {
        payload[offset] != 0
    } else {
        false
    };
    Ok(ExpertBatchResponse {
        layer,
        output_dtype,
        dim,
        expert_ids,
        outputs,
        spec_hit,
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

pub(crate) struct ShardTensorEntry {
    pub(crate) layer: usize,
    pub(crate) expert_idx: usize,
    pub(crate) kind: u8, // 0=gate, 1=up, 2=down
    pub(crate) ttype: i32,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) byte_offset: u64,
    pub(crate) byte_len: u64,
}

pub(crate) struct ModelShardHeaderFrame {
    pub(crate) total_bytes: u64,
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) dim: usize,
    pub(crate) expert_hidden_dim: usize,
    pub(crate) entries: Vec<ShardTensorEntry>,
    pub(crate) ssm_inner_size: usize,
    pub(crate) ssm_group_count: usize,
    pub(crate) ssm_time_step_rank: usize,
    pub(crate) ssm_state_size: usize,
    pub(crate) ssm_conv_kernel: usize,
    pub(crate) ssm_eps: f32,
    pub(crate) ssm_float_payload: Vec<u8>,
    pub(crate) n_experts_per_tok: usize,
    pub(crate) moe_n_group: usize,
    pub(crate) moe_topk_group: usize,
}

pub(crate) fn encode_model_shard_header(frame: &ModelShardHeaderFrame) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write_u64_le(&mut out, frame.total_bytes);
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
            .expert_hidden_dim
            .try_into()
            .map_err(|_| "expert_hidden_dim overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .entries
            .len()
            .try_into()
            .map_err(|_| "entries count overflow".to_string())?,
    );
    for entry in &frame.entries {
        write_u32_le(
            &mut out,
            entry
                .layer
                .try_into()
                .map_err(|_| "entry layer overflow".to_string())?,
        );
        write_u32_le(
            &mut out,
            entry
                .expert_idx
                .try_into()
                .map_err(|_| "entry expert_idx overflow".to_string())?,
        );
        out.push(entry.kind);
        out.extend_from_slice(&entry.ttype.to_le_bytes());
        write_u32_le(
            &mut out,
            entry
                .rows
                .try_into()
                .map_err(|_| "entry rows overflow".to_string())?,
        );
        write_u32_le(
            &mut out,
            entry
                .cols
                .try_into()
                .map_err(|_| "entry cols overflow".to_string())?,
        );
        write_u64_le(&mut out, entry.byte_offset);
        write_u64_le(&mut out, entry.byte_len);
    }
    // SSM extension fields
    write_u32_le(
        &mut out,
        frame
            .ssm_inner_size
            .try_into()
            .map_err(|_| "ssm_inner_size overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .ssm_group_count
            .try_into()
            .map_err(|_| "ssm_group_count overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .ssm_time_step_rank
            .try_into()
            .map_err(|_| "ssm_time_step_rank overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .ssm_state_size
            .try_into()
            .map_err(|_| "ssm_state_size overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .ssm_conv_kernel
            .try_into()
            .map_err(|_| "ssm_conv_kernel overflow".to_string())?,
    );
    write_f32_le(&mut out, frame.ssm_eps);
    write_u64_le(
        &mut out,
        frame
            .ssm_float_payload
            .len()
            .try_into()
            .map_err(|_| "ssm_float_payload length overflow".to_string())?,
    );
    out.extend_from_slice(&frame.ssm_float_payload);
    write_u32_le(
        &mut out,
        frame
            .n_experts_per_tok
            .try_into()
            .map_err(|_| "n_experts_per_tok overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .moe_n_group
            .try_into()
            .map_err(|_| "moe_n_group overflow".to_string())?,
    );
    write_u32_le(
        &mut out,
        frame
            .moe_topk_group
            .try_into()
            .map_err(|_| "moe_topk_group overflow".to_string())?,
    );
    Ok(out)
}

pub(crate) fn decode_model_shard_header(payload: &[u8]) -> Result<ModelShardHeaderFrame, String> {
    let mut offset = 0usize;
    let total_bytes = read_u64_le(payload, &mut offset)?;
    let n_layers = read_u32_le(payload, &mut offset)? as usize;
    let n_experts = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let expert_hidden_dim = read_u32_le(payload, &mut offset)? as usize;
    let n_entries = read_u32_le(payload, &mut offset)? as usize;
    let mut entries = Vec::with_capacity(n_entries);
    for _ in 0..n_entries {
        let layer = read_u32_le(payload, &mut offset)? as usize;
        let expert_idx = read_u32_le(payload, &mut offset)? as usize;
        if offset >= payload.len() {
            return Err("truncated shard header entry kind".to_string());
        }
        let kind = payload[offset];
        offset += 1;
        if offset + 4 > payload.len() {
            return Err("truncated shard header entry ttype".to_string());
        }
        let ttype = i32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        offset += 4;
        let rows = read_u32_le(payload, &mut offset)? as usize;
        let cols = read_u32_le(payload, &mut offset)? as usize;
        let byte_offset = read_u64_le(payload, &mut offset)?;
        let byte_len = read_u64_le(payload, &mut offset)?;
        entries.push(ShardTensorEntry {
            layer,
            expert_idx,
            kind,
            ttype,
            rows,
            cols,
            byte_offset,
            byte_len,
        });
    }
    // SSM extension fields (optional for backward compat)
    let (
        ssm_inner_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_state_size,
        ssm_conv_kernel,
        ssm_eps,
        ssm_float_payload,
        n_experts_per_tok,
        moe_n_group,
        moe_topk_group,
    ) = if offset + 4 <= payload.len() {
        let ssm_inner_size = read_u32_le(payload, &mut offset)? as usize;
        let ssm_group_count = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        let ssm_time_step_rank = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        let ssm_state_size = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        let ssm_conv_kernel = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        let ssm_eps = if offset + 4 <= payload.len() {
            read_f32_le(payload, &mut offset)?
        } else {
            1e-6
        };
        let ssm_float_payload = if offset + 8 <= payload.len() {
            let payload_len = read_u64_le(payload, &mut offset)? as usize;
            let end = offset
                .checked_add(payload_len)
                .ok_or_else(|| "ssm_float_payload overflow".to_string())?;
            if end > payload.len() {
                return Err("truncated ssm_float_payload".to_string());
            }
            let data = payload[offset..end].to_vec();
            offset = end;
            data
        } else {
            Vec::new()
        };
        let n_experts_per_tok = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        let moe_n_group = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        let moe_topk_group = if offset + 4 <= payload.len() {
            read_u32_le(payload, &mut offset)? as usize
        } else {
            0
        };
        (
            ssm_inner_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_state_size,
            ssm_conv_kernel,
            ssm_eps,
            ssm_float_payload,
            n_experts_per_tok,
            moe_n_group,
            moe_topk_group,
        )
    } else {
        (0, 0, 0, 0, 0, 1e-6f32, Vec::new(), 0, 0, 0)
    };

    Ok(ModelShardHeaderFrame {
        total_bytes,
        n_layers,
        n_experts,
        dim,
        expert_hidden_dim,
        entries,
        ssm_inner_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_state_size,
        ssm_conv_kernel,
        ssm_eps,
        ssm_float_payload,
        n_experts_per_tok,
        moe_n_group,
        moe_topk_group,
    })
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

pub(crate) struct SsmLayerRequest {
    pub(crate) layer: usize,
    pub(crate) xb: Vec<f32>,
}

pub(crate) struct SsmLayerResponse {
    pub(crate) layer: usize,
    pub(crate) xb2: Vec<f32>,
}

pub(crate) fn encode_ssm_layer_request(frame: &SsmLayerRequest) -> Vec<u8> {
    let dim = frame.xb.len();
    let mut out = Vec::with_capacity(8 + dim * 4);
    write_u32_le(&mut out, frame.layer as u32);
    write_u32_le(&mut out, dim as u32);
    for &v in &frame.xb {
        write_f32_le(&mut out, v);
    }
    out
}

pub(crate) fn decode_ssm_layer_request(payload: &[u8]) -> Result<SsmLayerRequest, String> {
    let mut offset = 0usize;
    let layer = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let expected = offset + dim * 4;
    if expected > payload.len() {
        return Err("truncated SSM layer request xb".to_string());
    }
    let mut xb = Vec::with_capacity(dim);
    for _ in 0..dim {
        xb.push(read_f32_le(payload, &mut offset)?);
    }
    Ok(SsmLayerRequest { layer, xb })
}

pub(crate) fn encode_ssm_layer_response(frame: &SsmLayerResponse) -> Vec<u8> {
    let dim = frame.xb2.len();
    let mut out = Vec::with_capacity(8 + dim * 4);
    write_u32_le(&mut out, frame.layer as u32);
    write_u32_le(&mut out, dim as u32);
    for &v in &frame.xb2 {
        write_f32_le(&mut out, v);
    }
    out
}

pub(crate) fn decode_ssm_layer_response(payload: &[u8]) -> Result<SsmLayerResponse, String> {
    let mut offset = 0usize;
    let layer = read_u32_le(payload, &mut offset)? as usize;
    let dim = read_u32_le(payload, &mut offset)? as usize;
    let expected = offset + dim * 4;
    if expected > payload.len() {
        return Err("truncated SSM layer response xb2".to_string());
    }
    let mut xb2 = Vec::with_capacity(dim);
    for _ in 0..dim {
        xb2.push(read_f32_le(payload, &mut offset)?);
    }
    Ok(SsmLayerResponse { layer, xb2 })
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
            hint_expert_ids: vec![3, 7],
        };
        let payload = encode_expert_batch_request(&request).expect("encode failed");
        let decoded = decode_expert_batch_request(&payload).expect("decode failed");
        assert_eq!(decoded.token_pos, request.token_pos);
        assert_eq!(decoded.layer, request.layer);
        assert_eq!(decoded.dim, request.dim);
        assert_eq!(decoded.expert_ids, request.expert_ids);
        assert_eq!(decoded.activation.len(), request.activation.len());
        assert_eq!(decoded.hint_expert_ids, request.hint_expert_ids);
    }

    #[test]
    fn speculation_advance_round_trip() {
        let frame = SpeculationAdvanceFrame {
            layer: 5,
            activation_dtype: ActivationDtype::Bf16,
            dim: 4,
            activation: vec![0.25, -0.5, 1.0, 2.0],
            hint_expert_ids: vec![2, 6, 9],
        };
        let payload = encode_speculation_advance(&frame).expect("encode failed");
        let decoded = decode_speculation_advance(&payload).expect("decode failed");
        assert_eq!(decoded.layer, frame.layer);
        assert_eq!(decoded.dim, frame.dim);
        assert_eq!(decoded.hint_expert_ids, frame.hint_expert_ids);
        assert_eq!(decoded.activation.len(), frame.activation.len());
    }

    #[test]
    fn proposed_expert_batch_round_trip() {
        let frame = ProposedExpertBatchFrame {
            layer: 7,
            output_dtype: ActivationDtype::Bf16,
            dim: 4,
            expert_ids: vec![3, 11],
            outputs: vec![vec![0.25, -0.5, 1.0, 2.0], vec![1.5, 2.5, -3.5, 4.5]],
        };
        let payload = encode_proposed_expert_batch(&frame).expect("encode failed");
        let decoded = decode_proposed_expert_batch(&payload).expect("decode failed");
        assert_eq!(decoded.layer, frame.layer);
        assert_eq!(decoded.dim, frame.dim);
        assert_eq!(decoded.expert_ids, frame.expert_ids);
        assert_eq!(decoded.outputs.len(), frame.outputs.len());
    }

    #[test]
    fn hello_frame_round_trip_with_assignments() {
        let frame = HelloFrame {
            node_address: "192.168.10.11:7000".to_string(),
            dim: 8,
            n_layers: 4,
            n_experts: 16,
            activation_dtype: ActivationDtype::Bf16,
            assigned_experts: vec![(0, 1), (3, 7)],
            speculative_push: true,
        };
        let payload = encode_hello_frame(&frame).expect("encode failed");
        let decoded = decode_hello_frame(&payload).expect("decode failed");
        assert_eq!(decoded.node_address, frame.node_address);
        assert_eq!(decoded.assigned_experts, frame.assigned_experts);
        assert_eq!(decoded.speculative_push, frame.speculative_push);
    }

    #[test]
    fn q8_round_trip_preserves_shape_and_small_error() {
        let values = [0.0f32, 1.5, -2.25, 7.0, -0.125, 0.333];
        let encoded = encode_activation_vector(&values, ActivationDtype::Q8);
        assert_eq!(
            encoded.len(),
            activation_vector_encoded_len(values.len(), ActivationDtype::Q8)
        );
        let decoded = decode_activation_vector(&encoded, ActivationDtype::Q8, values.len())
            .expect("decode failed");
        let max_abs_err = values
            .iter()
            .zip(decoded.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_err < 0.05, "q8 max_abs_err={max_abs_err}");
    }
}
