use crate::engine::distributed::protocol::{
    DiscoverResponseFrame, ExpertBatchResponse, FrameKind, HelloFrame,
    ReadyFrame, decode_error_frame, decode_expert_batch_request, decode_hello_frame,
    decode_model_shard_header, encode_discover_response_frame, encode_error_frame,
    encode_expert_batch_response, encode_ready_frame,
};
use crate::engine::distributed::resources::{NodeResourceSnapshot, detect_local_node_resources};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::kernels::{matmul_quantized_rows, silu_and_mul_inplace};
use crate::engine::types::{GgmlType, MappedFile, QuantizedTensor, WorkerExpertTensors, WorkerExpertWeights};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

struct WorkerRuntime {
    bind_address: String,
    resources: NodeResourceSnapshot,
}

struct WorkerSession {
    weights: WorkerExpertWeights,
    assigned_experts: Vec<Vec<bool>>,
    _shard: MappedFile,
    dim: usize,
    n_layers: usize,
    n_experts: usize,
    expert_hidden_dim: usize,
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
            return Err(
                "shard header dimensions do not match coordinator HELLO".to_string(),
            );
        }

        // Set long timeout for shard receive
        connection.set_timeout(Duration::from_secs(600))?;

        // Write shard to temp file
        let temp_path = shard_temp_path(&self.bind_address);
        let mut temp_file = std::fs::File::create(&temp_path)
            .map_err(|e| format!("failed to create shard temp file '{}': {e}", temp_path.display()))?;

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
        let mut assigned_mask =
            vec![vec![false; shard_header.n_experts]; shard_header.n_layers];

        let mut tensor_map: std::collections::HashMap<(usize, usize), [Option<QuantizedTensor>; 3]> =
            Default::default();
        for entry in &shard_header.entries {
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
            let gate = gate_opt.ok_or_else(|| {
                format!("missing gate for layer {layer} expert {expert_idx}")
            })?;
            let up = up_opt
                .ok_or_else(|| format!("missing up for layer {layer} expert {expert_idx}"))?;
            let down = down_opt.ok_or_else(|| {
                format!("missing down for layer {layer} expert {expert_idx}")
            })?;
            expert_slots[layer][expert_idx] = Some(WorkerExpertTensors { gate, up, down });
            assigned_mask[layer][expert_idx] = true;
        }

        let weights = WorkerExpertWeights {
            experts: expert_slots,
        };
        let loaded_expert_count = hello.assigned_experts.len();

        let ready = ReadyFrame {
            node_address: self.bind_address.clone(),
            dim: shard_header.dim,
            n_layers: shard_header.n_layers,
            n_experts: shard_header.n_experts,
            activation_dtype: hello.activation_dtype,
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
                dim: shard_header.dim,
                n_layers: shard_header.n_layers,
                n_experts: shard_header.n_experts,
                expert_hidden_dim: shard_header.expert_hidden_dim,
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
        })
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
                Ok((ready, session)) => {
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
                                match runtime.handle_request(&session, request) {
                                    Ok(response) => {
                                        let payload = encode_expert_batch_response(&response)?;
                                        connection.send_message(
                                            FrameKind::ExpertBatchResponse,
                                            message.request_id,
                                            &payload,
                                        )?;
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
