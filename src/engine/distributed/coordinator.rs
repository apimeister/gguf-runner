use crate::engine::distributed::placement::{
    ClusterConfig, ClusterNodeConfig, ClusterNodeRole, MoePlacementPlan,
};
use crate::engine::distributed::protocol::{
    ActivationDtype, DiscoverResponseFrame, ExpertBatchRequest, FrameKind, HelloFrame, ReadyFrame,
    decode_discover_response_frame, decode_error_frame, decode_expert_batch_response,
    decode_ready_frame, encode_expert_batch_request, encode_hello_frame,
};
use crate::engine::distributed::resources::{NodeResourceSnapshot, detect_local_node_resources};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::kernels::{axpy_inplace, matmul_quantized_rows, silu_and_mul_inplace};
use crate::engine::profiling::{
    profiling_enabled, record_distributed_local_experts, record_distributed_remote_request,
    record_distributed_transport_bytes,
};
use crate::engine::types::{Config, QuantizedTensor, TransformerWeights, WorkerExpertTensors};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use std::collections::HashMap;
use std::thread::sleep;
use std::time::{Duration, Instant};

const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;
const DISTRIBUTED_FRAME_HEADER_LEN: usize = 20;
const DEFAULT_REMOTE_RETRY_ATTEMPTS: usize = 3;
const DEFAULT_REMOTE_RETRY_BACKOFF_MS: u64 = 200;

fn discover_remote_node_resources(address: &str) -> Result<DiscoverResponseFrame, String> {
    let timeout = Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS);
    let mut last_transport_error = None;
    for attempt in 0..DEFAULT_REMOTE_RETRY_ATTEMPTS {
        let mut connection = match FramedConnection::connect(address, timeout) {
            Ok(connection) => connection,
            Err(err) => {
                last_transport_error = Some(err);
                if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                    sleep(Duration::from_millis(
                        DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                    ));
                    continue;
                }
                break;
            }
        };
        if let Err(err) = connection.send_message(FrameKind::DiscoverRequest, 0, &[]) {
            last_transport_error = Some(err);
            if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                sleep(Duration::from_millis(
                    DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                ));
                continue;
            }
            break;
        }
        let message = match connection.recv_message() {
            Ok(message) => message,
            Err(err) => {
                last_transport_error = Some(err);
                if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                    sleep(Duration::from_millis(
                        DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                    ));
                    continue;
                }
                break;
            }
        };
        return match message.kind {
            FrameKind::DiscoverResponse => decode_discover_response_frame(&message.payload),
            FrameKind::Error => Err(format!(
                "worker discovery at '{}' returned error: {}",
                address,
                decode_error_frame(&message.payload)?
            )),
            other => Err(format!(
                "worker discovery at '{}' returned unexpected frame {:?}",
                address, other
            )),
        };
    }
    Err(format!(
        "worker discovery at '{}' failed after {} attempt(s): {}",
        address,
        DEFAULT_REMOTE_RETRY_ATTEMPTS,
        last_transport_error.unwrap_or_else(|| "unknown transport failure".to_string())
    ))
}

pub(crate) fn discover_cluster_resources(
    cluster: &ClusterConfig,
    config: &Config,
) -> Result<ClusterConfig, String> {
    let local_resources = detect_local_node_resources()?;
    let mut discovered_nodes = Vec::with_capacity(cluster.node.len());
    for node in &cluster.node {
        let resources = if node.role == ClusterNodeRole::Coordinator {
            NodeResourceSnapshot {
                logical_cpu_count: local_resources.logical_cpu_count,
                memory_bytes: local_resources.memory_bytes,
            }
        } else {
            let response = discover_remote_node_resources(&node.address)?;
            if response.node_address != node.address {
                return Err(format!(
                    "worker discovery address mismatch for '{}': worker reported '{}'",
                    node.address, response.node_address
                ));
            }
            if response.dim != config.dim
                || response.n_layers != config.n_layers
                || response.n_experts != config.n_experts
            {
                return Err(format!(
                    "worker '{}' reported mismatched model metadata during discovery",
                    node.id
                ));
            }
            NodeResourceSnapshot {
                logical_cpu_count: response.logical_cpu_count,
                memory_bytes: response.memory_bytes,
            }
        };
        discovered_nodes.push(ClusterNodeConfig {
            id: node.id.clone(),
            address: node.address.clone(),
            role: node.role,
            logical_cpu_count: Some(resources.logical_cpu_count),
            discovered_memory_bytes: Some(resources.memory_bytes),
        });
    }
    let discovered = ClusterConfig {
        node: discovered_nodes,
    };
    discovered.validate()?;
    Ok(discovered)
}

pub(crate) trait MoeExpertExecutor {
    #[allow(clippy::too_many_arguments)]
    fn compute_selected_experts(
        &mut self,
        layer: usize,
        input: &[f32],
        selected: &[(usize, f32)],
        output: &mut [f32],
        config: &Config,
        weights: &TransformerWeights,
        mapped: &[u8],
    ) -> Result<(), String>;
}

#[allow(clippy::too_many_arguments)]
fn compute_single_expert_output(
    input: &[f32],
    expert_idx: usize,
    output: &mut [f32],
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    gate_exps: &QuantizedTensor,
    up_exps: &QuantizedTensor,
    down_exps: &QuantizedTensor,
    expert_hidden: usize,
    dim: usize,
    mapped: &[u8],
) -> Result<(), String> {
    let row_start_ffn = expert_idx * expert_hidden;
    matmul_quantized_rows(
        &mut gate_scratch[..expert_hidden],
        input,
        gate_exps,
        row_start_ffn,
        expert_hidden,
        mapped,
    )?;
    matmul_quantized_rows(
        &mut up_scratch[..expert_hidden],
        input,
        up_exps,
        row_start_ffn,
        expert_hidden,
        mapped,
    )?;
    silu_and_mul_inplace(
        &mut gate_scratch[..expert_hidden],
        &up_scratch[..expert_hidden],
    );
    let row_start_down = expert_idx * dim;
    matmul_quantized_rows(
        output,
        &gate_scratch[..expert_hidden],
        down_exps,
        row_start_down,
        dim,
        mapped,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_single_sliced_expert_output(
    input: &[f32],
    output: &mut [f32],
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    tensors: &WorkerExpertTensors,
    expert_hidden: usize,
    dim: usize,
    mapped: &[u8],
) -> Result<(), String> {
    matmul_quantized_rows(
        &mut gate_scratch[..expert_hidden],
        input,
        &tensors.gate,
        0,
        expert_hidden,
        mapped,
    )?;
    matmul_quantized_rows(
        &mut up_scratch[..expert_hidden],
        input,
        &tensors.up,
        0,
        expert_hidden,
        mapped,
    )?;
    silu_and_mul_inplace(
        &mut gate_scratch[..expert_hidden],
        &up_scratch[..expert_hidden],
    );
    matmul_quantized_rows(
        output,
        &gate_scratch[..expert_hidden],
        &tensors.down,
        0,
        dim,
        mapped,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_local_selected_experts(
    layer: usize,
    input: &[f32],
    selected: &[(usize, f32)],
    output: &mut [f32],
    config: &Config,
    weights: &TransformerWeights,
    mapped: &[u8],
    allow_parallel: bool,
) -> Result<(), String> {
    let expert_hidden = config.expert_hidden_dim;
    let dim = config.dim;
    output[..dim].fill(0.0);
    if selected.is_empty() {
        return Ok(());
    }

    let gate_exps = &weights.moe_gate_exps[layer];
    let up_exps = &weights.moe_up_exps[layer];
    let down_exps = &weights.moe_down_exps[layer];
    let sliced_layer = weights
        .local_moe_experts
        .as_ref()
        .and_then(|local| local.experts.get(layer));

    if selected.len() >= 2 && allow_parallel {
        let mut contribs = vec![0.0f32; selected.len() * dim];
        let results = contribs
            .par_chunks_mut(dim)
            .zip(selected.par_iter())
            .map_init(
                || (vec![0.0f32; expert_hidden], vec![0.0f32; expert_hidden]),
                |(gate_scratch, up_scratch), (contrib, &(expert_idx, route_weight))| {
                    if let Some(tensors) = sliced_layer
                        .and_then(|layer_slots| layer_slots.get(expert_idx))
                        .and_then(|slot| slot.as_ref())
                    {
                        compute_single_sliced_expert_output(
                            input,
                            contrib,
                            gate_scratch,
                            up_scratch,
                            tensors,
                            expert_hidden,
                            dim,
                            mapped,
                        )?;
                    } else {
                        compute_single_expert_output(
                            input,
                            expert_idx,
                            contrib,
                            gate_scratch,
                            up_scratch,
                            gate_exps,
                            up_exps,
                            down_exps,
                            expert_hidden,
                            dim,
                            mapped,
                        )?;
                    }
                    for value in contrib {
                        *value *= route_weight;
                    }
                    Ok::<(), String>(())
                },
            )
            .collect::<Vec<_>>();
        for result in results {
            result?;
        }
        for contrib in contribs.chunks(dim) {
            axpy_inplace(&mut output[..dim], 1.0, contrib);
        }
        return Ok(());
    }

    let mut gate_scratch = vec![0.0f32; expert_hidden];
    let mut up_scratch = vec![0.0f32; expert_hidden];
    let mut expert_output = vec![0.0f32; dim];
    for &(expert_idx, route_weight) in selected {
        if let Some(tensors) = sliced_layer
            .and_then(|layer_slots| layer_slots.get(expert_idx))
            .and_then(|slot| slot.as_ref())
        {
            compute_single_sliced_expert_output(
                input,
                &mut expert_output,
                &mut gate_scratch,
                &mut up_scratch,
                tensors,
                expert_hidden,
                dim,
                mapped,
            )?;
        } else {
            compute_single_expert_output(
                input,
                expert_idx,
                &mut expert_output,
                &mut gate_scratch,
                &mut up_scratch,
                gate_exps,
                up_exps,
                down_exps,
                expert_hidden,
                dim,
                mapped,
            )?;
        }
        axpy_inplace(&mut output[..dim], route_weight, &expert_output);
    }
    Ok(())
}

struct RemoteWorkerClient {
    node_index: usize,
    address: String,
    hello: HelloFrame,
    timeout: Duration,
    connection: FramedConnection,
    stats: RemoteWorkerStats,
}

impl RemoteWorkerClient {
    fn shutdown(&mut self) {
        let _ = self.connection.send_message(FrameKind::Shutdown, 0, &[]);
        record_distributed_transport_bytes(DISTRIBUTED_FRAME_HEADER_LEN, 0);
    }

    fn reconnect(&mut self) -> Result<(), String> {
        let (connection, bytes_sent, bytes_received) =
            connect_worker_with_hello(&self.address, self.timeout, &self.hello, &self.address)?;
        self.connection = connection;
        record_distributed_transport_bytes(bytes_sent, bytes_received);
        Ok(())
    }
}

#[derive(Default)]
struct RemoteWorkerStats {
    request_batches: u64,
    expert_count: u64,
    bytes_sent: u64,
    bytes_received: u64,
    wait_ns: u64,
}

impl RemoteWorkerStats {
    fn record_request(
        &mut self,
        experts: usize,
        bytes_sent: usize,
        bytes_received: usize,
        wait_ns: u64,
    ) {
        self.request_batches += 1;
        self.expert_count += experts as u64;
        self.bytes_sent += bytes_sent as u64;
        self.bytes_received += bytes_received as u64;
        self.wait_ns += wait_ns;
    }
}

pub(crate) struct DistributedMoeCoordinator {
    plan: MoePlacementPlan,
    activation_dtype: ActivationDtype,
    next_request_id: u64,
    remote_workers: Vec<RemoteWorkerClient>,
}

fn connect_worker_with_hello(
    address: &str,
    timeout: Duration,
    hello: &HelloFrame,
    worker_address: &str,
) -> Result<(FramedConnection, usize, usize), String> {
    let hello_payload = encode_hello_frame(hello)?;
    let mut last_transport_error = None;
    for attempt in 0..DEFAULT_REMOTE_RETRY_ATTEMPTS {
        let mut connection = match FramedConnection::connect(address, timeout) {
            Ok(connection) => connection,
            Err(err) => {
                last_transport_error = Some(err);
                if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                    sleep(Duration::from_millis(
                        DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                    ));
                    continue;
                }
                break;
            }
        };
        if let Err(err) = connection.send_message(FrameKind::Hello, 0, &hello_payload) {
            last_transport_error = Some(err);
            if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                sleep(Duration::from_millis(
                    DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                ));
                continue;
            }
            break;
        }
        let message = match connection.recv_message() {
            Ok(message) => message,
            Err(err) => {
                last_transport_error = Some(err);
                if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                    sleep(Duration::from_millis(
                        DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                    ));
                    continue;
                }
                break;
            }
        };
        match message.kind {
            FrameKind::Ready => {
                let ready = decode_ready_frame(&message.payload)?;
                DistributedMoeCoordinator::validate_ready_frame(hello, &ready, worker_address)?;
                return Ok((
                    connection,
                    DISTRIBUTED_FRAME_HEADER_LEN + hello_payload.len(),
                    DISTRIBUTED_FRAME_HEADER_LEN + message.payload.len(),
                ));
            }
            FrameKind::Error => {
                return Err(format!(
                    "worker '{}' rejected hello: {}",
                    worker_address,
                    decode_error_frame(&message.payload)?
                ));
            }
            other => {
                return Err(format!(
                    "worker '{}' returned unexpected frame {:?} during handshake",
                    worker_address, other
                ));
            }
        }
    }
    Err(format!(
        "worker '{}' handshake at '{}' failed after {} attempt(s): {}",
        worker_address,
        address,
        DEFAULT_REMOTE_RETRY_ATTEMPTS,
        last_transport_error.unwrap_or_else(|| "unknown transport failure".to_string())
    ))
}

impl DistributedMoeCoordinator {
    pub(crate) fn connect(
        plan: MoePlacementPlan,
        activation_dtype: ActivationDtype,
    ) -> Result<Self, String> {
        let timeout = Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS);
        let mut remote_workers = Vec::new();
        for (node_index, node) in plan.nodes.iter().enumerate() {
            if node.role != ClusterNodeRole::Worker || node.assigned_expert_count == 0 {
                continue;
            }
            let assigned_experts = plan
                .assigned_experts_for_node(node_index)
                .collect::<Vec<_>>();
            let hello = HelloFrame {
                node_address: node.address.clone(),
                dim: plan.inventory.dim,
                n_layers: plan.inventory.n_layers,
                n_experts: plan.inventory.n_experts,
                activation_dtype,
                assigned_experts,
            };
            let (connection, bytes_sent, bytes_received) =
                connect_worker_with_hello(&node.address, timeout, &hello, &node.address)?;
            record_distributed_transport_bytes(bytes_sent, bytes_received);
            remote_workers.push(RemoteWorkerClient {
                node_index,
                address: node.address.clone(),
                hello,
                timeout,
                connection,
                stats: RemoteWorkerStats::default(),
            });
        }
        Ok(Self {
            plan,
            activation_dtype,
            next_request_id: 1,
            remote_workers,
        })
    }

    fn validate_ready_frame(
        hello: &HelloFrame,
        ready: &ReadyFrame,
        worker_address: &str,
    ) -> Result<(), String> {
        if hello.node_address != ready.node_address
            || hello.dim != ready.dim
            || hello.n_layers != ready.n_layers
            || hello.n_experts != ready.n_experts
            || hello.activation_dtype != ready.activation_dtype
        {
            return Err(format!(
                "worker '{}' READY frame does not match coordinator HELLO",
                worker_address
            ));
        }
        if ready.loaded_expert_count != hello.assigned_experts.len() {
            return Err(format!(
                "worker '{}' loaded {} experts but coordinator assigned {}",
                worker_address,
                ready.loaded_expert_count,
                hello.assigned_experts.len()
            ));
        }
        Ok(())
    }
}

impl Drop for DistributedMoeCoordinator {
    fn drop(&mut self) {
        for worker in &mut self.remote_workers {
            worker.shutdown();
        }
        if profiling_enabled() {
            for worker in &self.remote_workers {
                if worker.stats.request_batches == 0 {
                    continue;
                }
                eprintln!(
                    "[PROFILE] distributed_moe worker='{}' batches={} experts={} sent={:.3} MiB recv={:.3} MiB wait={:.3} ms ({:.3} ms/batch)",
                    self.plan.nodes[worker.node_index].address,
                    worker.stats.request_batches,
                    worker.stats.expert_count,
                    worker.stats.bytes_sent as f64 / (1024.0 * 1024.0),
                    worker.stats.bytes_received as f64 / (1024.0 * 1024.0),
                    worker.stats.wait_ns as f64 / 1_000_000.0,
                    if worker.stats.request_batches == 0 {
                        0.0
                    } else {
                        (worker.stats.wait_ns as f64 / 1_000_000.0)
                            / worker.stats.request_batches as f64
                    }
                );
            }
        }
    }
}

impl MoeExpertExecutor for DistributedMoeCoordinator {
    fn compute_selected_experts(
        &mut self,
        layer: usize,
        input: &[f32],
        selected: &[(usize, f32)],
        output: &mut [f32],
        config: &Config,
        weights: &TransformerWeights,
        mapped: &[u8],
    ) -> Result<(), String> {
        let dim = config.dim;
        output[..dim].fill(0.0);
        if selected.is_empty() {
            return Ok(());
        }

        let mut local_selected = Vec::new();
        let mut remote_selected: HashMap<usize, Vec<usize>> = HashMap::new();
        let n_experts = self.plan.inventory.n_experts;
        for &(expert_idx, _) in selected {
            let plan_index = layer
                .checked_mul(n_experts)
                .and_then(|value| value.checked_add(expert_idx))
                .ok_or_else(|| "expert plan index overflow".to_string())?;
            let node_index = *self
                .plan
                .expert_node_indices
                .get(plan_index)
                .ok_or_else(|| "expert plan index out of bounds".to_string())?;
            if self.plan.nodes[node_index].role
                == crate::engine::distributed::placement::ClusterNodeRole::Coordinator
            {
                if let Some((_, route_weight)) = selected.iter().find(|(idx, _)| *idx == expert_idx)
                {
                    local_selected.push((expert_idx, *route_weight));
                }
            } else {
                remote_selected
                    .entry(node_index)
                    .or_default()
                    .push(expert_idx);
            }
        }

        if !local_selected.is_empty() {
            record_distributed_local_experts(local_selected.len());
            compute_local_selected_experts(
                layer,
                input,
                &local_selected,
                output,
                config,
                weights,
                mapped,
                true,
            )?;
        }

        let mut remote_outputs: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut next_request_id = self.next_request_id;
        for client in &mut self.remote_workers {
            let Some(expert_ids) = remote_selected.get(&client.node_index) else {
                continue;
            };
            let request_id = next_request_id;
            next_request_id = next_request_id.wrapping_add(1).max(1);
            let request = ExpertBatchRequest {
                token_pos: 0,
                layer,
                activation_dtype: self.activation_dtype,
                dim,
                expert_ids: expert_ids.clone(),
                activation: input.to_vec(),
            };
            let payload = encode_expert_batch_request(&request)?;
            let (message, wait_ns, wire_sent, wire_received) = {
                let mut last_transport_error = None;
                let mut result = None;
                for attempt in 0..DEFAULT_REMOTE_RETRY_ATTEMPTS {
                    let wire_sent = DISTRIBUTED_FRAME_HEADER_LEN + payload.len();
                    let wait_start = Instant::now();
                    match client.connection.send_message(
                        FrameKind::ExpertBatchRequest,
                        request_id,
                        &payload,
                    ) {
                        Ok(()) => {}
                        Err(err) => {
                            last_transport_error = Some(err);
                            if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                                client.reconnect()?;
                                sleep(Duration::from_millis(
                                    DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                                ));
                                continue;
                            }
                            break;
                        }
                    }
                    match client.connection.recv_message() {
                        Ok(message) => {
                            let wait_ns = wait_start.elapsed().as_nanos() as u64;
                            let wire_received =
                                DISTRIBUTED_FRAME_HEADER_LEN + message.payload.len();
                            result = Some((message, wait_ns, wire_sent, wire_received));
                            break;
                        }
                        Err(err) => {
                            last_transport_error = Some(err);
                            if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                                client.reconnect()?;
                                sleep(Duration::from_millis(
                                    DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                                ));
                                continue;
                            }
                            break;
                        }
                    }
                }
                result.ok_or_else(|| {
                    format!(
                        "worker '{}' request failed after {} attempt(s): {}",
                        client.address,
                        DEFAULT_REMOTE_RETRY_ATTEMPTS,
                        last_transport_error
                            .unwrap_or_else(|| "unknown transport failure".to_string())
                    )
                })?
            };
            client
                .stats
                .record_request(expert_ids.len(), wire_sent, wire_received, wait_ns);
            record_distributed_remote_request(expert_ids.len(), wire_sent, wire_received, wait_ns);
            if message.request_id != request_id {
                return Err(format!(
                    "worker '{}' returned mismatched request id: got {}, expected {}",
                    self.plan.nodes[client.node_index].address, message.request_id, request_id
                ));
            }
            match message.kind {
                FrameKind::ExpertBatchResponse => {
                    let response = decode_expert_batch_response(&message.payload)?;
                    if response.layer != layer || response.dim != dim {
                        return Err(format!(
                            "worker '{}' returned invalid response shape for layer {}",
                            self.plan.nodes[client.node_index].address, layer
                        ));
                    }
                    for (expert_idx, values) in
                        response.expert_ids.into_iter().zip(response.outputs)
                    {
                        remote_outputs.insert(expert_idx, values);
                    }
                }
                FrameKind::Error => {
                    return Err(format!(
                        "worker '{}' returned error: {}",
                        self.plan.nodes[client.node_index].address,
                        decode_error_frame(&message.payload)?
                    ));
                }
                other => {
                    return Err(format!(
                        "worker '{}' returned unexpected frame {:?}",
                        self.plan.nodes[client.node_index].address, other
                    ));
                }
            }
        }
        self.next_request_id = next_request_id;

        for &(expert_idx, route_weight) in selected {
            if let Some(values) = remote_outputs.get(&expert_idx) {
                axpy_inplace(&mut output[..dim], route_weight, values);
            }
        }
        Ok(())
    }
}
