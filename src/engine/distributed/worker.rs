use crate::engine::distributed::protocol::{
    ActivationDtype, DiscoverResponseFrame, ExpertBatchResponse, FrameKind, HelloFrame,
    ProposedExpertBatchFrame, ReadyFrame, SHARD_KIND_GATE_INP, SsmLayerResponse,
    decode_error_frame, decode_expert_batch_request, decode_hello_frame, decode_model_shard_header,
    decode_speculation_advance, decode_ssm_layer_request, encode_discover_response_frame,
    encode_error_frame, encode_expert_batch_response, encode_proposed_expert_batch,
    encode_ready_frame, encode_ssm_layer_response,
};
use crate::engine::distributed::resources::{NodeResourceSnapshot, detect_local_node_resources};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::kernels::{matmul_quantized, matmul_quantized_rows, silu_and_mul_inplace};
use crate::engine::switches::par_qwen3next_min_heads;
use crate::engine::types::{
    GgmlType, MappedFile, QuantizedTensor, WorkerExpertTensors, WorkerExpertWeights,
    WorkerSsmLayerWeights, WorkerSsmSession,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSlice,
    ParallelSliceMut,
};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

struct WorkerRuntime {
    bind_address: String,
    resources: NodeResourceSnapshot,
}

struct WorkerGateLayerData {
    gate_inp: QuantizedTensor,
}

struct SpeculationCache {
    layer: usize,
    outputs: std::collections::HashMap<usize, Vec<f32>>,
    push_expert_ids: Vec<usize>,
}

type SsmFloatPayloadEntry = (usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
type SsmLayerFloatData = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

struct WorkerSession {
    weights: WorkerExpertWeights,
    assigned_experts: Vec<Vec<bool>>,
    _shard: MappedFile,
    dim: usize,
    n_layers: usize,
    n_experts: usize,
    expert_hidden_dim: usize,
    ssm_session: Option<WorkerSsmSession>,
    gate_layers: Vec<Option<WorkerGateLayerData>>,
    moe_layer_indices: Vec<usize>,
    n_experts_per_tok: usize,
    moe_n_group: usize,
    moe_topk_group: usize,
    // Non-blocking deep speculation: one in-flight rayon task per future MoE layer.
    spec_receivers: std::sync::Mutex<
        std::collections::HashMap<usize, std::sync::mpsc::Receiver<SpeculationCache>>,
    >,
    last_activation: Vec<f32>,
    spec_hits: u64,
    spec_misses: u64,
    /// Results already pushed to the coordinator, retained as fallback for a timing miss.
    pushed_cache: std::collections::HashMap<usize, SpeculationCache>,
    /// Coordinator-provided routing hint for a specific future layer.
    hint_for_layer: Option<(usize, Vec<usize>)>,
    activation_dtype: ActivationDtype,
    speculative_push: bool,
}

fn shard_temp_path(bind_address: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "gguf_worker_shard_{}.bin",
        bind_address.replace(['.', ':'], "_")
    ))
}

impl WorkerRuntime {
    fn discovery_frame(&self) -> DiscoverResponseFrame {
        DiscoverResponseFrame {
            node_address: self.bind_address.clone(),
            dim: 0,
            n_layers: 0,
            n_experts: 0,
            logical_cpu_count: self.resources.logical_cpu_count,
            memory_bytes: self.resources.memory_bytes,
        }
    }

    fn prepare_session(
        &self,
        hello: &HelloFrame,
        connection: &mut FramedConnection,
    ) -> Result<(ReadyFrame, WorkerSession), String> {
        if hello.node_address != self.bind_address {
            return Err(format!(
                "coordinator targeted worker '{}' but this worker is '{}'",
                hello.node_address, self.bind_address
            ));
        }

        // Receive ModelShardHeader frame
        let shard_header_msg = connection.recv_message()?;
        if shard_header_msg.kind != FrameKind::ModelShardHeader {
            return Err(format!(
                "expected ModelShardHeader frame, got {:?}",
                shard_header_msg.kind
            ));
        }
        let shard_header = decode_model_shard_header(&shard_header_msg.payload)?;

        // Validate header dimensions match hello
        if shard_header.dim != hello.dim
            || shard_header.n_layers != hello.n_layers
            || shard_header.n_experts != hello.n_experts
        {
            return Err("shard header dimensions do not match coordinator HELLO".to_string());
        }

        // Set long timeout for shard receive
        connection.set_timeout(Duration::from_secs(600))?;

        // Write shard to temp file
        let temp_path = shard_temp_path(&self.bind_address);
        let mut temp_file = std::fs::File::create(&temp_path).map_err(|e| {
            format!(
                "failed to create shard temp file '{}': {e}",
                temp_path.display()
            )
        })?;

        let received_bytes = connection.recv_raw_stream_to_file(&mut temp_file)?;

        // Restore normal timeout
        connection.set_timeout(Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS))?;

        println!(
            "Worker '{}': received {:.1} MiB expert shard ({} experts)",
            self.bind_address,
            received_bytes as f64 / (1024.0 * 1024.0),
            hello.assigned_experts.len()
        );

        // Re-open for mmap (need read-only after writing)
        drop(temp_file);
        let mmap_file = std::fs::File::open(&temp_path)
            .map_err(|e| format!("failed to open shard temp file for mmap: {e}"))?;
        let shard_mmap = MappedFile::map(&mmap_file)
            .map_err(|e| format!("failed to mmap shard temp file: {e}"))?;

        // Build WorkerExpertWeights from shard header entries
        let mut expert_slots: Vec<Vec<Option<WorkerExpertTensors>>> = (0..shard_header.n_layers)
            .map(|_| (0..shard_header.n_experts).map(|_| None).collect())
            .collect();
        let mut assigned_mask = vec![vec![false; shard_header.n_experts]; shard_header.n_layers];

        let mut tensor_map: std::collections::HashMap<
            (usize, usize),
            [Option<QuantizedTensor>; 3],
        > = Default::default();
        for entry in &shard_header.entries {
            if entry.kind > 2 {
                continue; // SSM entries (kind 3-8) are handled by build_ssm_session
            }
            let t = QuantizedTensor {
                data_offset: entry.byte_offset as usize,
                ttype: GgmlType(entry.ttype),
                rows: entry.rows,
                cols: entry.cols,
            };
            let slot = tensor_map
                .entry((entry.layer, entry.expert_idx))
                .or_insert([None, None, None]);
            slot[entry.kind as usize] = Some(t);
        }
        for ((layer, expert_idx), [gate_opt, up_opt, down_opt]) in tensor_map {
            let gate = gate_opt
                .ok_or_else(|| format!("missing gate for layer {layer} expert {expert_idx}"))?;
            let up = up_opt
                .ok_or_else(|| format!("missing up for layer {layer} expert {expert_idx}"))?;
            let down = down_opt
                .ok_or_else(|| format!("missing down for layer {layer} expert {expert_idx}"))?;
            expert_slots[layer][expert_idx] = Some(WorkerExpertTensors { gate, up, down });
            assigned_mask[layer][expert_idx] = true;
        }

        let weights = WorkerExpertWeights {
            experts: expert_slots,
        };
        let loaded_expert_count = hello.assigned_experts.len();

        // Build SSM session if the shard header contains SSM data
        let ssm_session = if shard_header.ssm_inner_size > 0 {
            Some(build_ssm_session(&shard_header, shard_mmap.as_slice())?)
        } else {
            None
        };

        // Build gate layer data for speculative routing
        let gate_layers = build_gate_layers(&shard_header);
        let moe_layer_indices: Vec<usize> = (0..shard_header.n_layers)
            .filter(|&l| gate_layers[l].is_some())
            .collect();
        let n_experts_per_tok = shard_header.n_experts_per_tok;
        let moe_n_group = shard_header.moe_n_group;
        let moe_topk_group = shard_header.moe_topk_group;
        let dim = shard_header.dim;
        let n_experts = shard_header.n_experts;

        let ready = ReadyFrame {
            node_address: self.bind_address.clone(),
            dim,
            n_layers: shard_header.n_layers,
            n_experts,
            activation_dtype: hello.activation_dtype,
            speculative_push: hello.speculative_push,
            logical_cpu_count: self.resources.logical_cpu_count,
            memory_bytes: self.resources.memory_bytes,
            loaded_expert_count,
        };
        Ok((
            ready,
            WorkerSession {
                weights,
                assigned_experts: assigned_mask,
                _shard: shard_mmap,
                dim,
                n_layers: shard_header.n_layers,
                n_experts,
                expert_hidden_dim: shard_header.expert_hidden_dim,
                ssm_session,
                gate_layers,
                moe_layer_indices,
                n_experts_per_tok,
                moe_n_group,
                moe_topk_group,
                spec_receivers: std::sync::Mutex::new(std::collections::HashMap::new()),
                last_activation: vec![0.0f32; dim],
                spec_hits: 0,
                spec_misses: 0,
                pushed_cache: std::collections::HashMap::new(),
                hint_for_layer: None,
                activation_dtype: hello.activation_dtype,
                speculative_push: hello.speculative_push,
            },
        ))
    }

    fn handle_request(
        &self,
        session: &WorkerSession,
        request: crate::engine::distributed::protocol::ExpertBatchRequest,
    ) -> Result<ExpertBatchResponse, String> {
        if request.layer >= session.n_layers {
            return Err(format!(
                "invalid layer {} for worker request",
                request.layer
            ));
        }
        if request.dim != session.dim {
            return Err(format!(
                "invalid activation dim {} for worker request, expected {}",
                request.dim, session.dim
            ));
        }
        let mapped = session._shard.as_slice();
        let layer = request.layer;
        let dim = session.dim;
        let hidden = session.expert_hidden_dim;

        // Compute all experts in parallel — each gets its own scratch buffers.
        let outputs: Vec<Vec<f32>> = request
            .expert_ids
            .par_iter()
            .map(|&expert_idx| {
                if expert_idx >= session.n_experts
                    || !session.assigned_experts[layer][expert_idx]
                {
                    return Err(format!(
                        "expert {} in layer {} is not assigned to worker '{}'",
                        expert_idx, layer, self.bind_address
                    ));
                }
                let tensors = session
                    .weights
                    .experts
                    .get(layer)
                    .and_then(|l| l.get(expert_idx))
                    .and_then(|slot| slot.as_ref())
                    .ok_or_else(|| {
                        format!(
                            "expert {} in layer {} was assigned to worker '{}' but its sliced tensors are missing",
                            expert_idx, layer, self.bind_address
                        )
                    })?;
                let mut gate_scratch = vec![0.0f32; hidden];
                let mut up_scratch = vec![0.0f32; hidden];
                let mut output = vec![0.0f32; dim];
                matmul_quantized_rows(
                    &mut gate_scratch,
                    &request.activation,
                    &tensors.gate,
                    0,
                    hidden,
                    mapped,
                )?;
                matmul_quantized_rows(
                    &mut up_scratch,
                    &request.activation,
                    &tensors.up,
                    0,
                    hidden,
                    mapped,
                )?;
                silu_and_mul_inplace(&mut gate_scratch, &up_scratch);
                matmul_quantized_rows(
                    &mut output,
                    &gate_scratch,
                    &tensors.down,
                    0,
                    dim,
                    mapped,
                )?;
                Ok(output)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(ExpertBatchResponse {
            layer: request.layer,
            output_dtype: request.activation_dtype,
            dim: session.dim,
            expert_ids: request.expert_ids,
            outputs,
            spec_hit: false,
        })
    }
}

fn parse_ssm_float_payload(
    payload: &[u8],
    n_layers: usize,
) -> Result<Vec<SsmFloatPayloadEntry>, String> {
    let mut offset = 0usize;
    let mut result = Vec::new();

    let read_u32 = |buf: &[u8], off: &mut usize| -> Result<u32, String> {
        if *off + 4 > buf.len() {
            return Err("truncated u32 in float payload".to_string());
        }
        let v = u32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
        *off += 4;
        Ok(v)
    };
    let read_f32s = |buf: &[u8], off: &mut usize, count: usize| -> Result<Vec<f32>, String> {
        let end = off.checked_add(count * 4).ok_or("float payload overflow")?;
        if end > buf.len() {
            return Err("truncated f32s in float payload".to_string());
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let v = f32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
            *off += 4;
            out.push(v);
        }
        Ok(out)
    };

    let _ = n_layers; // informational
    while offset < payload.len() {
        let layer = read_u32(payload, &mut offset)? as usize;
        let conv1d_len = read_u32(payload, &mut offset)? as usize;
        let conv1d = read_f32s(payload, &mut offset, conv1d_len)?;
        let a_len = read_u32(payload, &mut offset)? as usize;
        let a = read_f32s(payload, &mut offset, a_len)?;
        let dt_bias = read_f32s(payload, &mut offset, a_len)?; // same len
        let norm_len = read_u32(payload, &mut offset)? as usize;
        let norm = read_f32s(payload, &mut offset, norm_len)?;
        result.push((layer, conv1d, a, dt_bias, norm));
    }
    Ok(result)
}

fn build_ssm_session(
    shard_header: &crate::engine::distributed::protocol::ModelShardHeaderFrame,
    mapped: &[u8],
) -> Result<WorkerSsmSession, String> {
    let n_layers = shard_header.n_layers;
    let d_inner = shard_header.ssm_inner_size;
    let n_k_heads = shard_header.ssm_group_count;
    let n_v_heads = shard_header.ssm_time_step_rank;
    let head_dim = shard_header.ssm_state_size;
    let conv_kernel = shard_header.ssm_conv_kernel;
    let eps = shard_header.ssm_eps;
    let dim = shard_header.dim;

    let conv_dim = d_inner + 2 * n_k_heads * head_dim;
    let hist_steps = if conv_kernel > 0 { conv_kernel - 1 } else { 0 };
    let conv_stride = hist_steps * conv_dim;
    let state_stride = n_v_heads * head_dim * head_dim;

    // Build layer weight slots indexed by layer index
    let mut layer_map: std::collections::HashMap<usize, [Option<QuantizedTensor>; 6]> =
        std::collections::HashMap::new();
    for entry in &shard_header.entries {
        let kind = entry.kind as usize;
        if !(3..=8).contains(&kind) {
            continue; // not an SSM entry
        }
        let t = QuantizedTensor {
            data_offset: entry.byte_offset as usize,
            ttype: GgmlType(entry.ttype),
            rows: entry.rows,
            cols: entry.cols,
        };
        let slot = layer_map
            .entry(entry.layer)
            .or_insert([None, None, None, None, None, None]);
        slot[kind - 3] = Some(t);
    }

    // Parse float payload
    let float_data = parse_ssm_float_payload(&shard_header.ssm_float_payload, n_layers)?;
    let mut float_by_layer: std::collections::HashMap<usize, SsmLayerFloatData> =
        std::collections::HashMap::new();
    for (layer, conv1d, a, dt_bias, norm) in float_data {
        float_by_layer.insert(layer, (conv1d, a, dt_bias, norm));
    }

    // Build layers vec
    let mut layers: Vec<Option<WorkerSsmLayerWeights>> = (0..n_layers).map(|_| None).collect();
    for (layer, tensors) in layer_map {
        if layer >= n_layers {
            continue;
        }
        let [qkv_opt, gate_opt, out_opt, ba_opt, alpha_opt, beta_opt] = tensors;
        let qkv_in = qkv_opt.ok_or_else(|| format!("SSM layer {layer}: missing qkv_in tensor"))?;
        let gate = gate_opt.ok_or_else(|| format!("SSM layer {layer}: missing gate tensor"))?;
        let out_proj =
            out_opt.ok_or_else(|| format!("SSM layer {layer}: missing out_proj tensor"))?;

        let (conv1d, a, dt_bias, norm) = float_by_layer
            .remove(&layer)
            .ok_or_else(|| format!("SSM layer {layer}: missing float payload"))?;

        layers[layer] = Some(WorkerSsmLayerWeights {
            qkv_in,
            gate,
            out_proj,
            ba: ba_opt,
            alpha: alpha_opt,
            beta: beta_opt,
            conv1d,
            a,
            dt_bias,
            norm,
        });
    }

    let _ = mapped; // SSM tensors are referenced by byte_offset into the shard mmap

    let conv_state = vec![0.0f32; n_layers * conv_stride];
    let ssm_state = vec![0.0f32; n_layers * state_stride];

    Ok(WorkerSsmSession {
        layers,
        conv_state,
        ssm_state,
        dim,
        d_inner,
        n_k_heads,
        n_v_heads,
        head_dim,
        conv_kernel,
        n_layers,
        eps,
        ssm_qkv: vec![0.0f32; conv_dim],
        ssm_z: vec![0.0f32; d_inner],
        ssm_ba_scratch: vec![0.0f32; 2 * n_v_heads],
        ssm_gate_exp: vec![0.0f32; n_v_heads],
        ssm_beta_scratch: vec![0.0f32; n_v_heads],
        ssm_conv: vec![0.0f32; conv_dim],
        ssm_q: vec![0.0f32; d_inner],
        ssm_k: vec![0.0f32; d_inner],
        ssm_v: vec![0.0f32; d_inner],
        ssm_proj: vec![0.0f32; d_inner],
        ssm_kv_mem: vec![0.0f32; d_inner],
        ssm_delta: vec![0.0f32; d_inner],
    })
}

fn build_gate_layers(
    shard_header: &crate::engine::distributed::protocol::ModelShardHeaderFrame,
) -> Vec<Option<WorkerGateLayerData>> {
    let mut layers = (0..shard_header.n_layers).map(|_| None).collect::<Vec<_>>();
    for entry in &shard_header.entries {
        if entry.kind != SHARD_KIND_GATE_INP || entry.layer >= layers.len() {
            continue;
        }
        layers[entry.layer] = Some(WorkerGateLayerData {
            gate_inp: QuantizedTensor {
                data_offset: entry.byte_offset as usize,
                ttype: GgmlType(entry.ttype),
                rows: entry.rows,
                cols: entry.cols,
            },
        });
    }
    layers
}

#[inline]
fn siluf(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplusf(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

fn ssm_norm_silu_head(out_h: &mut [f32], z_h: &[f32], norm: &[f32], eps: f32) {
    let head_dim = out_h.len();
    let ss: f32 = out_h.iter().map(|value| value * value).sum();
    let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
    for i in 0..head_dim {
        let gate = z_h[i] / (1.0 + (-z_h[i]).exp());
        out_h[i] = out_h[i] * inv * norm[i] * gate;
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_ssm_state_head_step(
    state_h: &mut [f32],
    out_h: &mut [f32],
    kv_mem: &mut [f32],
    delta: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
) {
    let head_dim = q.len();
    // Scale state by gate
    for s in state_h.iter_mut() {
        *s *= g;
    }
    // kv_mem = state^T @ k  (linear attention retrieval)
    kv_mem.fill(0.0);
    for j in 0..head_dim {
        let kj = k[j];
        if kj == 0.0 {
            continue;
        }
        let col = &state_h[j * head_dim..(j + 1) * head_dim];
        for i in 0..head_dim {
            kv_mem[i] += kj * col[i];
        }
    }
    for i in 0..head_dim {
        kv_mem[i] = finite_or_zero(kv_mem[i]);
        delta[i] = finite_or_zero((v[i] - kv_mem[i]) * beta);
    }
    // state update: state += k outer delta
    for j in 0..head_dim {
        let kj = k[j];
        if kj == 0.0 {
            continue;
        }
        let col = &mut state_h[j * head_dim..(j + 1) * head_dim];
        for i in 0..head_dim {
            col[i] += kj * delta[i];
        }
    }
    // output: out = state^T @ q
    out_h.fill(0.0);
    for j in 0..head_dim {
        let qj = q[j];
        if qj == 0.0 {
            continue;
        }
        let col = &state_h[j * head_dim..(j + 1) * head_dim];
        for i in 0..head_dim {
            out_h[i] += qj * col[i];
        }
    }
    for v in out_h.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
}

fn worker_ssm_step(
    ssm: &mut WorkerSsmSession,
    layer: usize,
    xb: &[f32],
    mapped: &[u8],
) -> Result<Vec<f32>, String> {
    let d_inner = ssm.d_inner;
    let n_k_heads = ssm.n_k_heads;
    let n_v_heads = ssm.n_v_heads;
    let head_dim = ssm.head_dim;
    let conv_kernel = ssm.conv_kernel;
    let eps = ssm.eps;
    let dim = ssm.dim;
    let conv_dim = d_inner + 2 * n_k_heads * head_dim;

    if layer >= ssm.n_layers {
        return Err(format!(
            "SSM layer {layer} out of range (n_layers={})",
            ssm.n_layers
        ));
    }

    let layer_w = ssm.layers[layer]
        .as_ref()
        .ok_or_else(|| format!("SSM layer {layer} has no weights on this worker"))?;

    // Input projections
    matmul_quantized_rows(
        &mut ssm.ssm_qkv[..conv_dim],
        &xb[..dim],
        &layer_w.qkv_in,
        0,
        conv_dim,
        mapped,
    )?;
    matmul_quantized_rows(
        &mut ssm.ssm_z[..d_inner],
        &xb[..dim],
        &layer_w.gate,
        0,
        d_inner,
        mapped,
    )?;

    let has_fused_ba = layer_w.ba.is_some();

    if has_fused_ba {
        let ba_t = layer_w.ba.as_ref().unwrap();
        matmul_quantized_rows(
            &mut ssm.ssm_ba_scratch[..2 * n_v_heads],
            &xb[..dim],
            ba_t,
            0,
            2 * n_v_heads,
            mapped,
        )?;
    } else {
        let alpha_t = layer_w
            .alpha
            .as_ref()
            .ok_or_else(|| format!("SSM layer {layer}: missing alpha tensor"))?;
        let beta_t = layer_w
            .beta
            .as_ref()
            .ok_or_else(|| format!("SSM layer {layer}: missing beta tensor"))?;
        matmul_quantized_rows(
            &mut ssm.ssm_gate_exp[..n_v_heads],
            &xb[..dim],
            alpha_t,
            0,
            n_v_heads,
            mapped,
        )?;
        matmul_quantized_rows(
            &mut ssm.ssm_beta_scratch[..n_v_heads],
            &xb[..dim],
            beta_t,
            0,
            n_v_heads,
            mapped,
        )?;
    }

    // Convolution
    let hist_steps = conv_kernel - 1;
    let conv_stride = hist_steps * conv_dim;
    let conv_hist_off = layer * conv_stride;

    {
        let conv_hist = &mut ssm.conv_state[conv_hist_off..conv_hist_off + conv_stride];
        let conv_w = &layer_w.conv1d;
        let qkv_snap: Vec<f32> = ssm.ssm_qkv[..conv_dim].to_vec();

        for c in 0..conv_dim {
            let mut acc = qkv_snap[c] * conv_w[c * conv_kernel + hist_steps];
            for t in 0..hist_steps {
                acc += conv_hist[t * conv_dim + c] * conv_w[c * conv_kernel + t];
            }
            if !acc.is_finite() {
                acc = 0.0;
            }
            ssm.ssm_conv[c] = siluf(acc);
        }

        if hist_steps > 0 {
            if hist_steps > 1 {
                conv_hist.copy_within(conv_dim.., 0);
            }
            let tail = (hist_steps - 1) * conv_dim;
            conv_hist[tail..tail + conv_dim].copy_from_slice(&qkv_snap);
        }
    }

    // Q/K/V split + norm
    let q_off = 0usize;
    let k_off = n_k_heads * head_dim;
    let v_off = 2 * n_k_heads * head_dim;
    let inv_scale_q = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_v_heads {
        let src_h = h % n_k_heads;
        let q_src_start = q_off + src_h * head_dim;
        let k_src_start = k_off + src_h * head_dim;
        let v_src_start = v_off + h * head_dim;

        let mut q_ss = 0.0f32;
        let mut k_ss = 0.0f32;
        for i in 0..head_dim {
            let q_v = ssm.ssm_conv[q_src_start + i];
            let k_v = ssm.ssm_conv[k_src_start + i];
            q_ss += q_v * q_v;
            k_ss += k_v * k_v;
        }
        let q_inv = 1.0 / (q_ss + eps).sqrt();
        let k_inv = 1.0 / (k_ss + eps).sqrt();
        for i in 0..head_dim {
            ssm.ssm_q[h * head_dim + i] =
                finite_or_zero(ssm.ssm_conv[q_src_start + i] * q_inv * inv_scale_q);
            ssm.ssm_k[h * head_dim + i] = finite_or_zero(ssm.ssm_conv[k_src_start + i] * k_inv);
            ssm.ssm_v[h * head_dim + i] = finite_or_zero(ssm.ssm_conv[v_src_start + i]);
        }
    }

    // Compute gate/beta for each head
    let heads_per_group = n_v_heads / n_k_heads;
    for h in 0..n_v_heads {
        let (beta_val, alpha_val) = if has_fused_ba {
            let group = h / heads_per_group;
            let idx = h % heads_per_group;
            let base = group * (2 * heads_per_group);
            (
                sigmoidf(ssm.ssm_ba_scratch[base + idx]),
                ssm.ssm_ba_scratch[base + heads_per_group + idx] + layer_w.dt_bias[h],
            )
        } else {
            (
                sigmoidf(ssm.ssm_beta_scratch[h]),
                ssm.ssm_gate_exp[h] + layer_w.dt_bias[h],
            )
        };
        let mut gate = softplusf(alpha_val) * layer_w.a[h];
        if !gate.is_finite() {
            gate = 0.0;
        }
        ssm.ssm_beta_scratch[h] = finite_or_zero(beta_val);
        ssm.ssm_gate_exp[h] = finite_or_zero(gate.exp());
    }

    // State update
    let state_stride = n_v_heads * head_dim * head_dim;
    let state_off = layer * state_stride;

    let gate_all: Vec<f32> = ssm.ssm_gate_exp[..n_v_heads].to_vec();
    let beta_all: Vec<f32> = ssm.ssm_beta_scratch[..n_v_heads].to_vec();

    if n_v_heads >= par_qwen3next_min_heads() {
        let state = &mut ssm.ssm_state[state_off..state_off + state_stride];
        let proj_all = &mut ssm.ssm_proj[..d_inner];
        let kv_mem_all = &mut ssm.ssm_kv_mem[..d_inner];
        let delta_all = &mut ssm.ssm_delta[..d_inner];
        let q_all = &ssm.ssm_q[..d_inner];
        let k_all = &ssm.ssm_k[..d_inner];
        let v_all = &ssm.ssm_v[..d_inner];

        state
            .par_chunks_mut(head_dim * head_dim)
            .zip(proj_all.par_chunks_mut(head_dim))
            .zip(kv_mem_all.par_chunks_mut(head_dim))
            .zip(delta_all.par_chunks_mut(head_dim))
            .enumerate()
            .for_each(|(h, (((state_h, out_h), kv_mem), delta))| {
                let q = &q_all[h * head_dim..(h + 1) * head_dim];
                let k = &k_all[h * head_dim..(h + 1) * head_dim];
                let v = &v_all[h * head_dim..(h + 1) * head_dim];
                worker_ssm_state_head_step(
                    state_h,
                    out_h,
                    kv_mem,
                    delta,
                    q,
                    k,
                    v,
                    gate_all[h],
                    beta_all[h],
                );
            });
    } else {
        for h in 0..n_v_heads {
            let q: Vec<f32> = ssm.ssm_q[h * head_dim..(h + 1) * head_dim].to_vec();
            let k: Vec<f32> = ssm.ssm_k[h * head_dim..(h + 1) * head_dim].to_vec();
            let v: Vec<f32> = ssm.ssm_v[h * head_dim..(h + 1) * head_dim].to_vec();
            let mut kv_mem = vec![0.0f32; head_dim];
            let mut delta = vec![0.0f32; head_dim];
            let state_h = &mut ssm.ssm_state
                [state_off + h * head_dim * head_dim..state_off + (h + 1) * head_dim * head_dim];
            let out_h = &mut ssm.ssm_proj[h * head_dim..(h + 1) * head_dim];
            worker_ssm_state_head_step(
                state_h,
                out_h,
                &mut kv_mem,
                &mut delta,
                &q,
                &k,
                &v,
                gate_all[h],
                beta_all[h],
            );
        }
    }

    // Norm + gate
    let norm = &layer_w.norm.clone();
    if n_v_heads >= par_qwen3next_min_heads() {
        ssm.ssm_proj[..d_inner]
            .par_chunks_mut(head_dim)
            .zip(ssm.ssm_z[..d_inner].par_chunks(head_dim))
            .for_each(|(out_h, z_h)| {
                ssm_norm_silu_head(out_h, z_h, norm, eps);
            });
    } else {
        let z_snap: Vec<f32> = ssm.ssm_z[..d_inner].to_vec();
        for h in 0..n_v_heads {
            let out_h = &mut ssm.ssm_proj[h * head_dim..(h + 1) * head_dim];
            let z_h = &z_snap[h * head_dim..(h + 1) * head_dim];
            ssm_norm_silu_head(out_h, z_h, norm, eps);
        }
    }

    // Output projection
    let layer_w = ssm.layers[layer].as_ref().unwrap();
    let proj_snap: Vec<f32> = ssm.ssm_proj[..d_inner].to_vec();
    let mut xb2 = vec![0.0f32; dim];
    matmul_quantized(&mut xb2[..dim], &proj_snap, &layer_w.out_proj, mapped)?;
    for v in &mut xb2[..dim] {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    Ok(xb2)
}

/// Thin wrapper to ship the mmap read-only pointer across thread boundaries.
/// SAFETY: the mmap lives for the duration of the WorkerSession, which outlives any
/// speculation task spawned from it.
struct SendRawSlice(usize, usize); // (ptr as usize, len)
unsafe impl Send for SendRawSlice {}

/// How many MoE layers ahead to speculatively pre-compute.
const SPEC_DEPTH: usize = 2;

/// Predict routing and launch a sequential background chain for uncovered future layers.
///
/// Path 1: if `session.hint_for_layer` matches the nearest future layer, uses the
/// coordinator-provided expert IDs (computed from exact x_L) instead of the stale
/// self-predicted IDs (from x_{L-1}).  The hint is consumed on use.
fn speculate_ahead(session: &mut WorkerSession, completed_layer: usize) {
    if session.n_experts_per_tok == 0 {
        session.spec_receivers.lock().unwrap().clear();
        return;
    }

    // Next SPEC_DEPTH MoE layers we need to cover, in ascending order.
    let future_layers: Vec<usize> = session
        .moe_layer_indices
        .iter()
        .filter(|&&l| l > completed_layer)
        .take(SPEC_DEPTH)
        .copied()
        .collect();

    if future_layers.is_empty() {
        return;
    }
    session
        .pushed_cache
        .retain(|&layer, _| future_layers.contains(&layer));

    let n_experts = session.n_experts;
    let n_top = (session.n_experts_per_tok * 2).min(n_experts);
    let dim = session.dim;
    let hidden = session.expert_hidden_dim;
    let moe_n_group = session.moe_n_group;
    let moe_topk_group = session.moe_topk_group;
    let shard_ptr = session._shard.ptr as usize;
    let shard_len = session._shard.len;
    let mapped: &[u8] = unsafe { std::slice::from_raw_parts(shard_ptr as *const u8, shard_len) };
    let act = session.last_activation[..dim].to_vec();

    // Consume hint if it targets the nearest future layer.
    let hint_consumed_layer = session
        .hint_for_layer
        .as_ref()
        .filter(|(hl, _)| future_layers.first() == Some(hl))
        .map(|(hl, _)| *hl);
    let hint_ids: Vec<usize> = if hint_consumed_layer.is_some() {
        session
            .hint_for_layer
            .take()
            .map(|(_, ids)| ids)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    type WorkItem = (
        usize,
        Vec<(usize, WorkerExpertTensors)>,
        Vec<usize>,
        std::sync::mpsc::SyncSender<SpeculationCache>,
    );
    let mut work_items: Vec<WorkItem> = Vec::new();
    let mut new_receivers: std::collections::HashMap<
        usize,
        std::sync::mpsc::Receiver<SpeculationCache>,
    > = std::collections::HashMap::new();

    {
        let mut receivers = session.spec_receivers.lock().unwrap();
        // Drop entries outside the current window.
        receivers.retain(|&layer, _| future_layers.contains(&layer));

        for &next_layer in &future_layers {
            if receivers.contains_key(&next_layer) {
                continue; // already in flight; retain it
            }
            if session.pushed_cache.contains_key(&next_layer) {
                continue;
            }
            // Determine which expert IDs to compute for next_layer.
            let predicted: Vec<usize> = if hint_consumed_layer == Some(next_layer)
                && !hint_ids.is_empty()
            {
                // Path 1: use coordinator hint (exact current activation, 0-stale).
                hint_ids
                    .iter()
                    .filter(|&&id| {
                        session
                            .assigned_experts
                            .get(next_layer)
                            .and_then(|v| v.get(id))
                            .copied()
                            .unwrap_or(false)
                    })
                    .copied()
                    .collect()
            } else {
                // Self-predict using last_activation (stale by 1 step).
                let gate_inp = match session.gate_layers.get(next_layer).and_then(|g| g.as_ref()) {
                    Some(g) => g.gate_inp.clone(),
                    None => continue,
                };
                let mut logits = vec![0.0f32; n_experts];
                if matmul_quantized_rows(&mut logits, &act, &gate_inp, 0, n_experts, mapped)
                    .is_err()
                {
                    continue;
                }
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut scores: Vec<f32> = logits.iter().map(|&v| (v - max_l).exp()).collect();
                let sum: f32 = scores.iter().sum();
                let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
                for s in &mut scores {
                    *s *= inv;
                }

                let use_groups = moe_n_group > 1
                    && moe_topk_group < moe_n_group
                    && n_experts.is_multiple_of(moe_n_group);
                let group_size = if use_groups {
                    n_experts / moe_n_group
                } else {
                    n_experts
                };
                let mut selected_group = vec![true; moe_n_group.max(1)];

                if use_groups {
                    let mut group_scores = vec![0.0f32; moe_n_group];
                    for (g, group_score) in group_scores.iter_mut().enumerate().take(moe_n_group) {
                        let start = g * group_size;
                        let mut top1 = f32::NEG_INFINITY;
                        let mut top2 = f32::NEG_INFINITY;
                        for &s in &scores[start..start + group_size] {
                            if s > top1 {
                                top2 = top1;
                                top1 = s;
                            } else if s > top2 {
                                top2 = s;
                            }
                        }
                        *group_score = top1 + top2.max(0.0);
                    }
                    let mut g_indices: Vec<usize> = (0..moe_n_group).collect();
                    g_indices.select_nth_unstable_by(moe_topk_group - 1, |&a, &b| {
                        group_scores[b]
                            .partial_cmp(&group_scores[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for g in &g_indices[moe_topk_group..] {
                        selected_group[*g] = false;
                        let start = g * group_size;
                        for s in &mut scores[start..start + group_size] {
                            *s = 0.0;
                        }
                    }
                }

                let mut candidates: Vec<(f32, usize)> = scores
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| selected_group[i / group_size])
                    .map(|(i, &s)| (s, i))
                    .collect();
                let n_take = n_top.min(candidates.len());
                if n_take > 0 {
                    candidates.select_nth_unstable_by(n_take - 1, |&a, &b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                let mut top = candidates[..n_take].to_vec();
                top.sort_unstable_by(|&a, &b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                top.into_iter().map(|(_, i)| i).collect()
            };

            let my_experts: Vec<(usize, WorkerExpertTensors)> = predicted
                .iter()
                .filter(|&&id| {
                    session
                        .assigned_experts
                        .get(next_layer)
                        .and_then(|v| v.get(id))
                        .copied()
                        .unwrap_or(false)
                })
                .filter_map(|&id| {
                    session
                        .weights
                        .experts
                        .get(next_layer)
                        .and_then(|l| l.get(id))
                        .and_then(|slot| slot.as_ref())
                        .map(|t| (id, t.clone()))
                })
                .collect();

            if my_experts.is_empty() {
                continue;
            }
            let push_expert_ids = predicted
                .iter()
                .take(session.n_experts_per_tok)
                .filter(|&&id| {
                    session
                        .assigned_experts
                        .get(next_layer)
                        .and_then(|v| v.get(id))
                        .copied()
                        .unwrap_or(false)
                })
                .copied()
                .collect();

            let (tx, rx) = std::sync::mpsc::sync_channel::<SpeculationCache>(1);
            work_items.push((next_layer, my_experts, push_expert_ids, tx));
            new_receivers.insert(next_layer, rx);
        }

        receivers.extend(new_receivers);
    } // drop receivers lock

    if work_items.is_empty() {
        return;
    }

    let raw = SendRawSlice(shard_ptr, shard_len);
    rayon::spawn(move || {
        let mapped: &[u8] = unsafe { std::slice::from_raw_parts(raw.0 as *const u8, raw.1) };
        for (next_layer, my_experts, push_expert_ids, tx) in work_items {
            let outputs: Result<Vec<(usize, Vec<f32>)>, String> = my_experts
                .par_iter()
                .map(|(expert_idx, tensors)| {
                    let mut gate_scratch = vec![0.0f32; hidden];
                    let mut up_scratch = vec![0.0f32; hidden];
                    let mut output = vec![0.0f32; dim];
                    matmul_quantized_rows(
                        &mut gate_scratch,
                        &act,
                        &tensors.gate,
                        0,
                        hidden,
                        mapped,
                    )?;
                    matmul_quantized_rows(&mut up_scratch, &act, &tensors.up, 0, hidden, mapped)?;
                    silu_and_mul_inplace(&mut gate_scratch, &up_scratch);
                    matmul_quantized_rows(
                        &mut output,
                        &gate_scratch,
                        &tensors.down,
                        0,
                        dim,
                        mapped,
                    )?;
                    Ok((*expert_idx, output))
                })
                .collect::<Result<Vec<_>, _>>();
            if let Ok(pairs) = outputs {
                let mut map = std::collections::HashMap::with_capacity(pairs.len());
                for (id, out) in pairs {
                    map.insert(id, out);
                }
                let _ = tx.send(SpeculationCache {
                    layer: next_layer,
                    outputs: map,
                    push_expert_ids,
                });
            }
        }
    });
}

fn drain_and_push_ready_specs(
    session: &mut WorkerSession,
    connection: &mut FramedConnection,
    completed_layer: usize,
) {
    if !session.speculative_push {
        return;
    }
    let push_layer = session
        .moe_layer_indices
        .iter()
        .copied()
        .find(|&layer| layer > completed_layer);

    if let Some(layer) = push_layer
        && let Some(mut cache) = session.pushed_cache.remove(&layer)
    {
        push_speculation_cache(session, connection, &cache);
        cache.push_expert_ids.clear();
        session.pushed_cache.insert(layer, cache);
    }

    let mut ready = Vec::new();
    {
        let mut receivers = session.spec_receivers.lock().unwrap();
        let ready_layers: Vec<usize> = receivers.keys().copied().collect();
        for layer in ready_layers {
            let Some(rx) = receivers.remove(&layer) else {
                continue;
            };
            match rx.try_recv() {
                Ok(cache) => ready.push(cache),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    receivers.insert(layer, rx);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }
        }
    }

    for cache in ready {
        let layer = cache.layer;
        if Some(layer) != push_layer {
            session.pushed_cache.insert(layer, cache);
            continue;
        }
        push_speculation_cache(session, connection, &cache);
        session.pushed_cache.insert(
            layer,
            SpeculationCache {
                layer,
                outputs: cache.outputs,
                push_expert_ids: Vec::new(),
            },
        );
    }
}

fn push_speculation_cache(
    session: &WorkerSession,
    connection: &mut FramedConnection,
    cache: &SpeculationCache,
) {
    let expert_ids = cache.push_expert_ids.clone();
    if expert_ids.is_empty() {
        return;
    }
    let outputs: Vec<Vec<f32>> = expert_ids
        .iter()
        .filter_map(|id| cache.outputs.get(id).cloned())
        .collect();
    if outputs.len() != expert_ids.len() {
        return;
    }
    let frame = ProposedExpertBatchFrame {
        layer: cache.layer,
        output_dtype: session.activation_dtype,
        dim: session.dim,
        expert_ids,
        outputs,
    };
    if let Ok(payload) = encode_proposed_expert_batch(&frame) {
        let _ = connection.send_message(FrameKind::ProposedExpertBatch, 0, &payload);
    }
}

pub(crate) fn run_worker_server(bind_address: &str) -> Result<(), String> {
    let runtime = WorkerRuntime {
        bind_address: bind_address.to_string(),
        resources: detect_local_node_resources()?,
    };
    let listener = TcpListener::bind(bind_address)
        .map_err(|e| format!("failed to bind worker listener '{}': {e}", bind_address))?;
    println!(
        "Distributed worker listening on {} cpu={} mem_bytes={}",
        bind_address, runtime.resources.logical_cpu_count, runtime.resources.memory_bytes
    );

    for stream in listener.incoming() {
        let stream = stream.map_err(|e| format!("worker accept failed: {e}"))?;
        let mut connection = FramedConnection::from_stream(
            stream,
            Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS),
        )?;
        if let Err(err) = handle_worker_connection(&runtime, &mut connection) {
            eprintln!("distributed worker connection error: {err}");
        }
    }

    Ok(())
}

fn handle_worker_connection(
    runtime: &WorkerRuntime,
    connection: &mut FramedConnection,
) -> Result<(), String> {
    let first_message = connection.recv_message()?;
    match first_message.kind {
        FrameKind::DiscoverRequest => {
            let payload = encode_discover_response_frame(&runtime.discovery_frame())?;
            connection.send_message(
                FrameKind::DiscoverResponse,
                first_message.request_id,
                &payload,
            )?;
        }
        FrameKind::Hello => {
            let hello = decode_hello_frame(&first_message.payload)?;
            match runtime.prepare_session(&hello, connection) {
                Ok((ready, mut session)) => {
                    let payload = encode_ready_frame(&ready)?;
                    connection.send_message(
                        FrameKind::Ready,
                        first_message.request_id,
                        &payload,
                    )?;

                    loop {
                        let message = match connection.recv_message() {
                            Ok(message) => message,
                            Err(err)
                                if err.contains(
                                    "failed to read distributed frame header: failed to fill whole buffer",
                                ) =>
                            {
                                break;
                            }
                            Err(err) => return Err(err),
                        };
                        match message.kind {
                            FrameKind::ExpertBatchRequest => {
                                let request = decode_expert_batch_request(&message.payload)?;
                                let completed_layer = request.layer;
                                let dim = session.dim;

                                // Store coordinator routing hint for layer+1 (Path 1).
                                if !request.hint_expert_ids.is_empty() {
                                    session.hint_for_layer = Some((
                                        completed_layer + 1,
                                        request.hint_expert_ids.clone(),
                                    ));
                                }

                                // Check already-pushed results first, then poll the background
                                // speculation task for this layer (non-blocking).
                                let spec_cache: Option<SpeculationCache> =
                                    session.pushed_cache.remove(&request.layer).or_else(|| {
                                        session
                                            .spec_receivers
                                            .lock()
                                            .unwrap()
                                            .remove(&request.layer)
                                            .and_then(|rx| rx.try_recv().ok())
                                    });

                                // Superset hit check: all requested experts must be in the cache.
                                let mut req_ids_sorted = request.expert_ids.clone();
                                req_ids_sorted.sort_unstable();

                                let spec_hit = spec_cache.as_ref().is_some_and(|spec| {
                                    spec.layer == request.layer
                                        && req_ids_sorted
                                            .iter()
                                            .all(|id| spec.outputs.contains_key(id))
                                });

                                let response = if spec_hit {
                                    session.spec_hits += 1;
                                    let spec = spec_cache.unwrap();
                                    let outputs: Result<Vec<Vec<f32>>, String> = request
                                        .expert_ids
                                        .iter()
                                        .map(|id| {
                                            spec.outputs.get(id).cloned().ok_or_else(|| {
                                                format!("spec cache missing expert {id}")
                                            })
                                        })
                                        .collect();
                                    match outputs {
                                        Ok(outs) => Ok(ExpertBatchResponse {
                                            layer: request.layer,
                                            output_dtype: request.activation_dtype,
                                            dim,
                                            expert_ids: request.expert_ids.clone(),
                                            outputs: outs,
                                            spec_hit: true,
                                        }),
                                        Err(_) => {
                                            session.spec_hits -= 1;
                                            session.spec_misses += 1;
                                            runtime.handle_request(&session, request.clone())
                                        }
                                    }
                                } else {
                                    session.spec_misses += 1;
                                    runtime.handle_request(&session, request.clone())
                                };

                                // Update last_activation for next speculation
                                if request.activation.len() >= dim {
                                    session.last_activation[..dim]
                                        .copy_from_slice(&request.activation[..dim]);
                                }

                                match response {
                                    Ok(resp) => {
                                        let payload = encode_expert_batch_response(&resp)?;
                                        connection.send_message(
                                            FrameKind::ExpertBatchResponse,
                                            message.request_id,
                                            &payload,
                                        )?;
                                        // Speculate next layers while coordinator does attention/norm.
                                        speculate_ahead(&mut session, completed_layer);
                                        drain_and_push_ready_specs(
                                            &mut session,
                                            connection,
                                            completed_layer,
                                        );
                                    }
                                    Err(err) => {
                                        let payload = encode_error_frame(&err)?;
                                        connection.send_message(
                                            FrameKind::Error,
                                            message.request_id,
                                            &payload,
                                        )?;
                                    }
                                }
                            }
                            FrameKind::SpeculationAdvance => {
                                match decode_speculation_advance(&message.payload) {
                                    Ok(advance) => {
                                        let dim = session.dim;
                                        if advance.activation.len() >= dim {
                                            session.last_activation[..dim]
                                                .copy_from_slice(&advance.activation[..dim]);
                                        }
                                        if !advance.hint_expert_ids.is_empty() {
                                            session.hint_for_layer =
                                                Some((advance.layer + 1, advance.hint_expert_ids));
                                        }
                                        speculate_ahead(&mut session, advance.layer);
                                        drain_and_push_ready_specs(
                                            &mut session,
                                            connection,
                                            advance.layer,
                                        );
                                    }
                                    Err(err) => {
                                        let payload = encode_error_frame(&err)?;
                                        connection.send_message(
                                            FrameKind::Error,
                                            message.request_id,
                                            &payload,
                                        )?;
                                    }
                                }
                            }
                            FrameKind::SsmLayerRequest => {
                                let mapped = session._shard.as_slice();
                                if let Some(ssm) = &mut session.ssm_session {
                                    let req = decode_ssm_layer_request(&message.payload)?;
                                    match worker_ssm_step(ssm, req.layer, &req.xb, mapped) {
                                        Ok(xb2) => {
                                            let resp = SsmLayerResponse {
                                                layer: req.layer,
                                                xb2,
                                            };
                                            let payload = encode_ssm_layer_response(&resp);
                                            connection.send_message(
                                                FrameKind::SsmLayerResponse,
                                                message.request_id,
                                                &payload,
                                            )?;
                                        }
                                        Err(e) => {
                                            let payload = encode_error_frame(&e)?;
                                            connection.send_message(
                                                FrameKind::Error,
                                                message.request_id,
                                                &payload,
                                            )?;
                                        }
                                    }
                                } else {
                                    let payload = encode_error_frame("worker has no SSM session")?;
                                    connection.send_message(
                                        FrameKind::Error,
                                        message.request_id,
                                        &payload,
                                    )?;
                                }
                            }
                            FrameKind::Shutdown => break,
                            FrameKind::Error => {
                                return Err(format!(
                                    "worker received coordinator error: {}",
                                    decode_error_frame(&message.payload)?
                                ));
                            }
                            other => {
                                return Err(format!(
                                    "worker received unexpected frame {:?}",
                                    other
                                ));
                            }
                        }
                    }
                    let total = session.spec_hits + session.spec_misses;
                    let hit_pct = if total > 0 {
                        100.0 * session.spec_hits as f64 / total as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[SPEC] speculation hits={} misses={} total={} hit_rate={:.1}%",
                        session.spec_hits, session.spec_misses, total, hit_pct
                    );
                }
                Err(err) => {
                    let payload = encode_error_frame(&err)?;
                    connection.send_message(
                        FrameKind::Error,
                        first_message.request_id,
                        &payload,
                    )?;
                }
            }
        }
        other => {
            return Err(format!(
                "worker expected DISCOVER_REQUEST or HELLO as first distributed frame, got {:?}",
                other
            ));
        }
    }
    Ok(())
}
