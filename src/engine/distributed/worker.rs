use crate::engine::distributed::protocol::{
    DiscoverResponseFrame, ExpertBatchResponse, FrameKind, HelloFrame, ReadyFrame,
    decode_error_frame, decode_expert_batch_request, decode_hello_frame,
    encode_discover_response_frame, encode_error_frame, encode_expert_batch_response,
    encode_ready_frame,
};
use crate::engine::distributed::resources::{NodeResourceSnapshot, detect_local_node_resources};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::kernels::{matmul_quantized_rows, silu_and_mul_inplace};
use crate::engine::types::{Config, GGUFFile, WorkerExpertWeights};
use std::net::TcpListener;
use std::time::Duration;

const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

struct WorkerRuntime {
    gguf: GGUFFile,
    config: Config,
    bind_address: String,
    resources: NodeResourceSnapshot,
}

struct WorkerSession {
    weights: WorkerExpertWeights,
    assigned_experts: Vec<Vec<bool>>,
}

impl WorkerRuntime {
    fn discovery_frame(&self) -> DiscoverResponseFrame {
        DiscoverResponseFrame {
            node_address: self.bind_address.clone(),
            dim: self.config.dim,
            n_layers: self.config.n_layers,
            n_experts: self.config.n_experts,
            logical_cpu_count: self.resources.logical_cpu_count,
            memory_bytes: self.resources.memory_bytes,
        }
    }

    fn prepare_session(&self, hello: &HelloFrame) -> Result<(ReadyFrame, WorkerSession), String> {
        if hello.node_address != self.bind_address {
            return Err(format!(
                "coordinator targeted worker '{}' but this worker is '{}'",
                hello.node_address, self.bind_address
            ));
        }
        if hello.dim != self.config.dim
            || hello.n_layers != self.config.n_layers
            || hello.n_experts != self.config.n_experts
        {
            return Err("coordinator HELLO does not match worker model metadata".to_string());
        }

        let mut assigned_lists = vec![Vec::new(); self.config.n_layers];
        let mut assigned_mask = vec![vec![false; self.config.n_experts]; self.config.n_layers];
        for &(layer, expert_idx) in &hello.assigned_experts {
            if layer >= self.config.n_layers || expert_idx >= self.config.n_experts {
                return Err(format!(
                    "coordinator assigned invalid expert pair ({layer}, {expert_idx})"
                ));
            }
            if !assigned_mask[layer][expert_idx] {
                assigned_lists[layer].push(expert_idx);
                assigned_mask[layer][expert_idx] = true;
            }
        }

        let weights = crate::engine::weights::init_worker_expert_weights_from_gguf(
            &self.gguf,
            &self.config,
            &assigned_lists,
        )?;
        let loaded_expert_count = assigned_lists.iter().map(Vec::len).sum::<usize>();
        let ready = ReadyFrame {
            node_address: self.bind_address.clone(),
            dim: self.config.dim,
            n_layers: self.config.n_layers,
            n_experts: self.config.n_experts,
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
            },
        ))
    }

    fn handle_request(
        &self,
        session: &WorkerSession,
        request: crate::engine::distributed::protocol::ExpertBatchRequest,
    ) -> Result<ExpertBatchResponse, String> {
        if request.layer >= self.config.n_layers {
            return Err(format!(
                "invalid layer {} for worker request",
                request.layer
            ));
        }
        if request.dim != self.config.dim {
            return Err(format!(
                "invalid activation dim {} for worker request, expected {}",
                request.dim, self.config.dim
            ));
        }
        let mapped = self.gguf.mapped.as_slice();
        let mut outputs = Vec::with_capacity(request.expert_ids.len());
        let mut gate_scratch = vec![0.0f32; self.config.expert_hidden_dim];
        let mut up_scratch = vec![0.0f32; self.config.expert_hidden_dim];
        for &expert_idx in &request.expert_ids {
            if expert_idx >= self.config.n_experts
                || !session.assigned_experts[request.layer][expert_idx]
            {
                return Err(format!(
                    "expert {} in layer {} is not assigned to worker '{}'",
                    expert_idx, request.layer, self.bind_address
                ));
            }
            let tensors = session
                .weights
                .experts
                .get(request.layer)
                .and_then(|layer| layer.get(expert_idx))
                .and_then(|slot| slot.as_ref())
                .ok_or_else(|| {
                    format!(
                        "expert {} in layer {} was assigned to worker '{}' but its sliced tensors are missing",
                        expert_idx, request.layer, self.bind_address
                    )
                })?;
            let mut output = vec![0.0f32; self.config.dim];
            matmul_quantized_rows(
                &mut gate_scratch[..self.config.expert_hidden_dim],
                &request.activation,
                &tensors.gate,
                0,
                self.config.expert_hidden_dim,
                mapped,
            )?;
            matmul_quantized_rows(
                &mut up_scratch[..self.config.expert_hidden_dim],
                &request.activation,
                &tensors.up,
                0,
                self.config.expert_hidden_dim,
                mapped,
            )?;
            silu_and_mul_inplace(
                &mut gate_scratch[..self.config.expert_hidden_dim],
                &up_scratch[..self.config.expert_hidden_dim],
            );
            matmul_quantized_rows(
                &mut output,
                &gate_scratch[..self.config.expert_hidden_dim],
                &tensors.down,
                0,
                self.config.dim,
                mapped,
            )?;
            outputs.push(output);
        }
        Ok(ExpertBatchResponse {
            layer: request.layer,
            output_dtype: request.activation_dtype,
            dim: self.config.dim,
            expert_ids: request.expert_ids,
            outputs,
        })
    }
}

fn build_worker_runtime(
    gguf: GGUFFile,
    config: Config,
    bind_address: &str,
) -> Result<WorkerRuntime, String> {
    Ok(WorkerRuntime {
        gguf,
        config,
        bind_address: bind_address.to_string(),
        resources: detect_local_node_resources()?,
    })
}

pub(crate) fn run_worker_server(
    gguf: GGUFFile,
    config: Config,
    bind_address: &str,
) -> Result<(), String> {
    let runtime = build_worker_runtime(gguf, config, bind_address)?;
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
            match runtime.prepare_session(&hello) {
                Ok((ready, session)) => {
                    let payload = encode_ready_frame(&ready)?;
                    connection.send_message(
                        FrameKind::Ready,
                        first_message.request_id,
                        &payload,
                    )?;

                    loop {
                        let message = connection.recv_message()?;
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
