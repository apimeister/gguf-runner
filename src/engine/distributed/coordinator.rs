use crate::engine::distributed::placement::{
    ClusterConfig, ClusterNodeConfig, ClusterNodeRole, MoePlacementPlan,
};
use crate::engine::distributed::protocol::{
    ActivationDtype, DiscoverResponseFrame, ExpertBatchRequest, FrameKind, HelloFrame, ReadyFrame,
    SHARD_KIND_GATE_INP, SsmLayerRequest, decode_discover_response_frame, decode_error_frame,
    decode_expert_batch_response, decode_proposed_expert_batch, decode_ready_frame,
    decode_ssm_layer_response, encode_expert_batch_request, encode_hello_frame,
    encode_model_shard_header, encode_speculation_advance, encode_ssm_layer_request,
};
use crate::engine::distributed::resources::{NodeResourceSnapshot, detect_local_node_resources};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::kernels::{axpy_inplace, matmul_quantized_rows, silu_and_mul_inplace};
use crate::engine::profiling::{
    profiling_enabled, record_distributed_coordinator_timing, record_distributed_local_experts,
    record_distributed_remote_request, record_distributed_transport_bytes,
};
use crate::engine::types::{
    Config, GGUFFile, QuantizedTensor, TransformerWeights, WorkerExpertTensors,
};
use crate::engine::weights::{
    collect_expert_shard_ranges, collect_gate_routing_data, collect_ssm_shard_data,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, sleep};
use std::time::{Duration, Instant};

const BATCH_RETRY_ATTEMPTS: usize = 3;

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
            if response.dim != 0
                && (response.dim != config.dim
                    || response.n_layers != config.n_layers
                    || response.n_experts != config.n_experts)
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

    fn try_compute_ssm_layer(
        &mut self,
        _layer: usize,
        _xb: &[f32],
        _xb2: &mut [f32],
    ) -> Result<bool, String> {
        Ok(false)
    }
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

#[allow(dead_code)]
struct RemoteWorkerClient {
    node_index: usize,
    address: String,
    hello: HelloFrame,
    timeout: Duration,
    connection: FramedConnection,
    inbound: SharedWorkerInbound,
    next_request_id: u64,
    stats: RemoteWorkerStats,
    shard_header_payload: Vec<u8>,
    shard_data: Vec<u8>,
}

type SharedWorkerInbound = Arc<(Mutex<WorkerInboundState>, Condvar)>;

struct PendingBatchResponse {
    response: crate::engine::distributed::protocol::ExpertBatchResponse,
    wire_received: usize,
}

struct PendingSsmResponse {
    response: crate::engine::distributed::protocol::SsmLayerResponse,
    wire_received: usize,
}

#[derive(Default)]
struct WorkerInboundState {
    responses: HashMap<u64, Result<PendingBatchResponse, String>>,
    ssm_responses: HashMap<u64, Result<PendingSsmResponse, String>>,
    push_buffer: HashMap<usize, (Vec<usize>, Vec<Vec<f32>>)>,
    transport_error: Option<String>,
}

fn new_worker_inbound() -> SharedWorkerInbound {
    Arc::new((Mutex::new(WorkerInboundState::default()), Condvar::new()))
}

fn spawn_worker_reader(
    address: String,
    mut connection: FramedConnection,
    inbound: SharedWorkerInbound,
) {
    thread::spawn(move || {
        loop {
            let message = match connection.recv_message() {
                Ok(message) => message,
                Err(err) => {
                    let (lock, cvar) = &*inbound;
                    let mut state = lock.lock().unwrap();
                    state.transport_error = Some(format!("worker '{address}' recv failed: {err}"));
                    cvar.notify_all();
                    break;
                }
            };
            let wire_received = DISTRIBUTED_FRAME_HEADER_LEN + message.payload.len();
            let (lock, cvar) = &*inbound;
            let mut state = lock.lock().unwrap();
            match message.kind {
                FrameKind::ExpertBatchResponse => {
                    let decoded = decode_expert_batch_response(&message.payload).map(|response| {
                        PendingBatchResponse {
                            response,
                            wire_received,
                        }
                    });
                    state.responses.insert(message.request_id, decoded);
                    cvar.notify_all();
                }
                FrameKind::ProposedExpertBatch => {
                    if let Ok(frame) = decode_proposed_expert_batch(&message.payload) {
                        record_distributed_transport_bytes(0, wire_received);
                        state
                            .push_buffer
                            .insert(frame.layer, (frame.expert_ids, frame.outputs));
                        cvar.notify_all();
                    }
                }
                FrameKind::SsmLayerResponse => {
                    let decoded = decode_ssm_layer_response(&message.payload).map(|response| {
                        PendingSsmResponse {
                            response,
                            wire_received,
                        }
                    });
                    state.ssm_responses.insert(message.request_id, decoded);
                    cvar.notify_all();
                }
                FrameKind::Error => {
                    let err = decode_error_frame(&message.payload).unwrap_or_else(|decode_err| {
                        format!("failed to decode worker error: {decode_err}")
                    });
                    if message.request_id == 0 {
                        state.transport_error = Some(err);
                    } else {
                        state.responses.insert(message.request_id, Err(err.clone()));
                        state.ssm_responses.insert(message.request_id, Err(err));
                    }
                    cvar.notify_all();
                }
                other => {
                    state.transport_error = Some(format!(
                        "worker '{address}' sent unexpected frame {other:?}"
                    ));
                    cvar.notify_all();
                    break;
                }
            }
        }
    });
}

#[allow(dead_code)]
impl RemoteWorkerClient {
    fn shutdown(&mut self) {
        let _ = self.connection.send_message(FrameKind::Shutdown, 0, &[]);
        record_distributed_transport_bytes(DISTRIBUTED_FRAME_HEADER_LEN, 0);
    }

    fn reconnect(&mut self) -> Result<(), String> {
        let (connection, bytes_sent, bytes_received) = connect_worker_with_hello(
            &self.address,
            self.timeout,
            &self.hello,
            &self.address,
            &self.shard_header_payload,
            &self.shard_data,
        )?;
        let inbound = new_worker_inbound();
        let reader = connection.try_clone(self.timeout)?;
        spawn_worker_reader(self.address.clone(), reader, inbound.clone());
        self.connection = connection;
        self.inbound = inbound;
        self.next_request_id = 1;
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
    spec_hits: u64,
    spec_misses: u64,
}

type RemoteBatchReply = (Vec<Vec<f32>>, usize, usize, u64, bool);

impl RemoteWorkerStats {
    fn record_request(
        &mut self,
        experts: usize,
        bytes_sent: usize,
        bytes_received: usize,
        wait_ns: u64,
        spec_hit: bool,
    ) {
        self.request_batches += 1;
        self.expert_count += experts as u64;
        self.bytes_sent += bytes_sent as u64;
        self.bytes_received += bytes_received as u64;
        self.wait_ns += wait_ns;
        if spec_hit {
            self.spec_hits += 1;
        } else {
            self.spec_misses += 1;
        }
    }
}

pub(crate) struct DistributedMoeCoordinator {
    plan: MoePlacementPlan,
    activation_dtype: ActivationDtype,
    remote_workers: Vec<RemoteWorkerClient>,
    activation_buffer_pool: ActivationBufferPool,
    ssm_worker_index: Option<usize>,
    /// Gate routing tensors (indexed by layer) for coordinator-side routing prediction.
    /// These reference the original GGUF mmap via data_offset.
    gate_routing: Vec<Option<QuantizedTensor>>,
    n_experts_per_tok: usize,
    moe_n_group: usize,
    moe_topk_group: usize,
}

/// Pre-allocated activation buffers to avoid per-batch `input.to_vec()` allocations.
struct ActivationBufferPool {
    /// Buffer size = dim (model dimension, e.g., 14336 for Qwen3.5-122B)
    buffer_size: usize,
    /// Pool of reusable buffers (up to max_concurrent for thread safety)
    buffers: Vec<Vec<f32>>,
}

impl ActivationBufferPool {
    fn new(dim: usize, max_concurrent: usize) -> Self {
        let mut buffers = Vec::with_capacity(max_concurrent);
        for _ in 0..max_concurrent {
            buffers.push(vec![0.0f32; dim]);
        }
        Self {
            buffer_size: dim,
            buffers,
        }
    }

    /// Get a buffer by value (moves it into the caller's thread).
    /// This eliminates per-batch allocation overhead.
    fn get_buffer(&mut self) -> Vec<f32> {
        self.buffers
            .pop()
            .unwrap_or_else(|| vec![0.0f32; self.buffer_size])
    }

    /// Return a buffer to the pool for reuse.
    fn put_buffer(&mut self, buffer: Vec<f32>) {
        if buffer.len() == self.buffer_size {
            self.buffers.push(buffer);
        }
    }
}

fn connect_worker_with_hello(
    address: &str,
    timeout: Duration,
    hello: &HelloFrame,
    worker_address: &str,
    shard_header_payload: &[u8],
    shard_data: &[u8],
) -> Result<(FramedConnection, usize, usize), String> {
    let hello_payload = encode_hello_frame(hello)?;
    let total_mb = shard_data.len() as f64 / (1024.0 * 1024.0);
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
        // Send shard header frame and raw shard data before waiting for Ready
        if let Err(err) =
            connection.send_message(FrameKind::ModelShardHeader, 0, shard_header_payload)
        {
            last_transport_error = Some(err);
            if attempt + 1 < DEFAULT_REMOTE_RETRY_ATTEMPTS {
                sleep(Duration::from_millis(
                    DEFAULT_REMOTE_RETRY_BACKOFF_MS * (attempt as u64 + 1),
                ));
                continue;
            }
            break;
        }
        // Switch to long timeout for the raw data transfer
        if let Err(err) = connection.set_timeout(Duration::from_secs(600)) {
            last_transport_error = Some(err);
            break;
        }
        println!(
            "Sending model shard to worker '{}': {:.1} MiB",
            worker_address, total_mb
        );
        // Send raw bytes: write u64 length then data
        if let Err(err) = connection.send_raw_slices(shard_data, &[(0, shard_data.len())]) {
            last_transport_error = Some(err);
            break;
        }
        // Restore normal timeout
        if let Err(err) = connection.set_timeout(timeout) {
            last_transport_error = Some(err);
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
        speculative_push: bool,
        gguf: &GGUFFile,
        mapped: &[u8],
        config: &Config,
    ) -> Result<Self, String> {
        let timeout = Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS);

        struct WorkerSetup {
            node_index: usize,
            address: String,
            hello: HelloFrame,
            shard_header_payload: Vec<u8>,
            shard_data: Vec<u8>,
        }

        // Phase 1: prepare shard data for every worker (sequential mmap reads).
        let mut setups: Vec<WorkerSetup> = Vec::new();
        for (node_index, node) in plan.nodes.iter().enumerate() {
            if node.role != ClusterNodeRole::Worker || node.assigned_expert_count == 0 {
                continue;
            }
            let assigned_experts = plan
                .assigned_experts_for_node(node_index)
                .collect::<Vec<_>>();
            let mut assigned_by_layer = vec![Vec::new(); config.n_layers];
            for &(layer, expert_idx) in &assigned_experts {
                assigned_by_layer[layer].push(expert_idx);
            }
            let (shard_header, shard_ranges) =
                collect_expert_shard_ranges(gguf, config, &assigned_by_layer)?;
            let mut shard_data: Vec<u8> = Vec::with_capacity(shard_header.total_bytes as usize);
            for (offset, len) in &shard_ranges {
                shard_data.extend_from_slice(&mapped[*offset..*offset + *len]);
            }
            let shard_header_payload = encode_model_shard_header(&shard_header)?;
            let hello = HelloFrame {
                node_address: node.address.clone(),
                dim: plan.inventory.dim,
                n_layers: plan.inventory.n_layers,
                n_experts: plan.inventory.n_experts,
                activation_dtype,
                assigned_experts,
                speculative_push,
            };
            setups.push(WorkerSetup {
                node_index,
                address: node.address.clone(),
                hello,
                shard_header_payload,
                shard_data,
            });
        }

        // SSM shard for first worker only
        if let Some(first_setup) = setups.first_mut()
            && let Some(ssm_data) = collect_ssm_shard_data(gguf, config, 0)?
        {
            // Parse the existing header so we can update it
            let mut existing_header =
                crate::engine::distributed::protocol::decode_model_shard_header(
                    &first_setup.shard_header_payload,
                )
                .map_err(|e| format!("failed to re-decode shard header: {e}"))?;
            let expert_total_bytes = existing_header.total_bytes;

            // Shift SSM entry byte_offsets by the expert shard size and append
            for mut entry in ssm_data.entries {
                entry.byte_offset += expert_total_bytes;
                existing_header.entries.push(entry);
            }

            // Append SSM ranges data
            for (offset, len) in &ssm_data.ranges {
                first_setup
                    .shard_data
                    .extend_from_slice(&mapped[*offset..*offset + *len]);
            }

            // Update total_bytes and SSM fields
            existing_header.total_bytes = expert_total_bytes + ssm_data.total_bytes;
            existing_header.ssm_inner_size = ssm_data.ssm_inner_size;
            existing_header.ssm_group_count = ssm_data.ssm_group_count;
            existing_header.ssm_time_step_rank = ssm_data.ssm_time_step_rank;
            existing_header.ssm_state_size = ssm_data.ssm_state_size;
            existing_header.ssm_conv_kernel = ssm_data.ssm_conv_kernel;
            existing_header.ssm_eps = ssm_data.eps;
            existing_header.ssm_float_payload = ssm_data.float_payload;

            // Re-encode
            first_setup.shard_header_payload = encode_model_shard_header(&existing_header)?;
        }

        // Gate routing tensors for all workers (enables speculative routing)
        // Collect once; byte offsets are shifted per-worker relative to each worker's current shard end.
        let gate_data_opt = collect_gate_routing_data(gguf, config, 0)?;
        if let Some(ref gate_data) = gate_data_opt {
            for setup in setups.iter_mut() {
                let mut hdr = crate::engine::distributed::protocol::decode_model_shard_header(
                    &setup.shard_header_payload,
                )
                .map_err(|e| format!("failed to re-decode shard header for gate merge: {e}"))?;
                let base = hdr.total_bytes;
                for entry in &gate_data.entries {
                    hdr.entries
                        .push(crate::engine::distributed::protocol::ShardTensorEntry {
                            layer: entry.layer,
                            expert_idx: entry.expert_idx,
                            kind: entry.kind,
                            ttype: entry.ttype,
                            rows: entry.rows,
                            cols: entry.cols,
                            byte_offset: entry.byte_offset + base,
                            byte_len: entry.byte_len,
                        });
                }
                for (offset, len) in &gate_data.ranges {
                    setup
                        .shard_data
                        .extend_from_slice(&mapped[*offset..*offset + *len]);
                }
                hdr.total_bytes = base + gate_data.total_bytes;
                hdr.n_experts_per_tok = config.n_experts_used;
                hdr.moe_n_group = config.moe_n_group;
                hdr.moe_topk_group = config.moe_topk_group;
                setup.shard_header_payload = encode_model_shard_header(&hdr)?;
            }
        }

        // Phase 2: connect and transfer shards to all workers in parallel.
        // Each worker has its own dedicated network link so transfers overlap fully.
        let connections: Vec<Result<(FramedConnection, usize, usize), String>> =
            std::thread::scope(|s| {
                let handles: Vec<_> = setups
                    .iter()
                    .map(|setup| {
                        s.spawn(|| {
                            connect_worker_with_hello(
                                &setup.address,
                                timeout,
                                &setup.hello,
                                &setup.address,
                                &setup.shard_header_payload,
                                &setup.shard_data,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .unwrap_or_else(|_| Err("worker connect thread panicked".to_string()))
                    })
                    .collect()
            });

        let mut remote_workers = Vec::new();
        for (setup, conn_result) in setups.into_iter().zip(connections) {
            let (connection, bytes_sent, bytes_received) = conn_result?;
            let inbound = new_worker_inbound();
            let reader = connection.try_clone(timeout)?;
            spawn_worker_reader(setup.address.clone(), reader, inbound.clone());
            record_distributed_transport_bytes(bytes_sent, bytes_received);
            remote_workers.push(RemoteWorkerClient {
                node_index: setup.node_index,
                address: setup.address,
                hello: setup.hello,
                timeout,
                connection,
                inbound,
                next_request_id: 1,
                stats: RemoteWorkerStats::default(),
                shard_header_payload: setup.shard_header_payload,
                shard_data: setup.shard_data,
            });
        }

        // Build coordinator-local gate routing tensors for routing prediction (Path 1).
        // gate_data entries use byte_offset=0 base, so offsets are directly into the GGUF mmap.
        let mut gate_routing: Vec<Option<QuantizedTensor>> = vec![None; config.n_layers];
        if let Some(ref gate_data) = gate_data_opt {
            for entry in &gate_data.entries {
                if entry.kind == SHARD_KIND_GATE_INP && entry.layer < config.n_layers {
                    gate_routing[entry.layer] = Some(QuantizedTensor {
                        data_offset: entry.byte_offset as usize,
                        ttype: crate::engine::types::GgmlType(entry.ttype),
                        rows: entry.rows,
                        cols: entry.cols,
                    });
                }
            }
        }

        let dim = plan.inventory.dim;
        let max_concurrent = remote_workers.len().max(1);
        let ssm_worker_index = if remote_workers.is_empty() {
            None
        } else {
            Some(0)
        };
        Ok(Self {
            plan,
            activation_dtype,
            remote_workers,
            activation_buffer_pool: ActivationBufferPool::new(dim, max_concurrent),
            ssm_worker_index,
            gate_routing,
            n_experts_per_tok: config.n_experts_used,
            moe_n_group: config.moe_n_group,
            moe_topk_group: config.moe_topk_group,
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
            || hello.speculative_push != ready.speculative_push
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
                let spec_total = worker.stats.spec_hits + worker.stats.spec_misses;
                let spec_pct = if spec_total > 0 {
                    100.0 * worker.stats.spec_hits as f64 / spec_total as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[PROFILE] distributed_moe worker='{}' batches={} experts={} sent={:.3} MiB recv={:.3} MiB wait={:.3} ms ({:.3} ms/batch) spec_hit_rate={:.1}% ({}/{})",
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
                    },
                    spec_pct,
                    worker.stats.spec_hits,
                    spec_total,
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
        let total_start = Instant::now();
        let dim = config.dim;
        output[..dim].fill(0.0);
        if selected.is_empty() {
            return Ok(());
        }

        let mut local_selected = Vec::new();
        let mut remote_selected: Vec<(usize, Vec<usize>)> = Vec::new();
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
            } else if let Some(pos) = remote_selected.iter().position(|(n, _)| *n == node_index) {
                remote_selected[pos].1.push(expert_idx);
            } else {
                remote_selected.push((node_index, vec![expert_idx]));
            }
        }

        // Compute routing hint for layer+1 using the exact current activation (Path 1).
        // This hint will be sent to workers so they can restart speculation with 0-stale data.
        let next_layer = layer + 1;
        let hint_for_next: Vec<usize> = if next_layer < config.n_layers {
            if let Some(Some(gate)) = self.gate_routing.get(next_layer) {
                let gate = gate.clone();
                let hint_count = self.n_experts_per_tok.saturating_mul(2).min(n_experts);
                predict_moe_routing(
                    &input[..dim],
                    &gate,
                    n_experts,
                    hint_count,
                    self.moe_n_group,
                    self.moe_topk_group,
                    mapped,
                )
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let remote_start = Instant::now();
        let mut remote_outputs: HashMap<usize, Vec<f32>> = HashMap::new();

        // Phase 1: consume any asynchronously pushed results, otherwise fire requests
        // at all workers first so they start computing immediately.
        let mut push_hit_nodes: Vec<(usize, Vec<usize>, Vec<Vec<f32>>)> = Vec::new();
        let mut send_info: Vec<Option<(u64, usize, Instant)>> =
            Vec::with_capacity(remote_selected.len());

        for (node_index, expert_ids) in &remote_selected {
            let node_address = self.plan.nodes[*node_index].address.clone();
            let worker = self
                .remote_workers
                .iter_mut()
                .find(|w| w.node_index == *node_index)
                .ok_or_else(|| {
                    format!(
                        "distributed worker '{}' is missing an active coordinator connection",
                        node_address
                    )
                })?;

            if let Some(outputs) = take_pushed_outputs(worker, layer, expert_ids) {
                worker.stats.record_request(expert_ids.len(), 0, 0, 0, true);
                push_hit_nodes.push((*node_index, expert_ids.clone(), outputs));
                send_info.push(None);
            } else {
                let mut activation = self.activation_buffer_pool.get_buffer();
                activation[..dim].copy_from_slice(&input[..dim]);
                let send_result = send_batch_request_with_hint(
                    worker,
                    layer,
                    self.activation_dtype,
                    dim,
                    expert_ids,
                    &activation,
                    &hint_for_next,
                );
                self.activation_buffer_pool.put_buffer(activation);
                let (request_id, wire_sent, wait_start) = send_result
                    .map_err(|err| format!("worker '{}' send failed: {}", node_address, err))?;
                send_info.push(Some((request_id, wire_sent, wait_start)));
            }
        }

        for (_node_index, expert_ids, outputs) in &push_hit_nodes {
            for (expert_idx, values) in expert_ids.iter().copied().zip(outputs.iter()) {
                remote_outputs.insert(expert_idx, values.clone());
            }
        }

        for (node_index, _expert_ids, _) in &push_hit_nodes {
            let node_address = self.plan.nodes[*node_index].address.clone();
            let worker = self
                .remote_workers
                .iter_mut()
                .find(|w| w.node_index == *node_index)
                .ok_or_else(|| {
                    format!(
                        "distributed worker '{}' is missing an active coordinator connection",
                        node_address
                    )
                })?;
            send_speculation_advance(
                worker,
                layer,
                self.activation_dtype,
                dim,
                &input[..dim],
                &hint_for_next,
            )
            .map_err(|err| {
                format!(
                    "worker '{}' speculation advance send failed: {}",
                    node_address, err
                )
            })?;
        }

        // Overlap: compute local experts while workers run their remote batches.
        let local_start = Instant::now();
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
        let local_ns = local_start.elapsed().as_nanos() as u64;

        // Phase 2: collect responses from workers that have been computing since phase 1.
        for ((node_index, expert_ids), send_info_opt) in
            remote_selected.iter().zip(send_info.iter())
        {
            let Some((request_id, wire_sent, wait_start)) = send_info_opt else {
                continue;
            };
            let node_address = self.plan.nodes[*node_index].address.clone();
            let worker = self
                .remote_workers
                .iter_mut()
                .find(|w| w.node_index == *node_index)
                .ok_or_else(|| {
                    format!(
                        "distributed worker '{}' is missing an active coordinator connection",
                        node_address
                    )
                })?;
            let (outputs, bytes_sent, bytes_received, wait_ns, _spec_hit) = recv_batch_response(
                worker,
                *request_id,
                layer,
                dim,
                expert_ids,
                *wire_sent,
                *wait_start,
            )
            .map_err(|err| format!("worker '{}' batch recv failed: {}", node_address, err))?;
            if bytes_sent > 0 {
                record_distributed_remote_request(
                    expert_ids.len(),
                    bytes_sent,
                    bytes_received,
                    wait_ns,
                );
            }
            for (expert_idx, values) in expert_ids.iter().copied().zip(outputs) {
                remote_outputs.insert(expert_idx, values);
            }
        }
        let remote_ns = remote_start.elapsed().as_nanos() as u64;

        for &(expert_idx, route_weight) in selected {
            if let Some(values) = remote_outputs.get(&expert_idx) {
                axpy_inplace(&mut output[..dim], route_weight, values);
            }
        }
        let total_ns = total_start.elapsed().as_nanos() as u64;
        record_distributed_coordinator_timing(total_ns, local_ns, remote_ns);
        if profiling_enabled() {
            eprintln!(
                "[PROFILE] distributed_moe layer={} local_experts={} remote_nodes={} remote_experts={} push_hits={} total={:.3} ms local={:.3} ms remote={:.3} ms",
                layer,
                local_selected.len(),
                remote_selected.len(),
                remote_selected
                    .iter()
                    .map(|(_, ids)| ids.len())
                    .sum::<usize>(),
                push_hit_nodes.len(),
                total_ns as f64 / 1_000_000.0,
                local_ns as f64 / 1_000_000.0,
                remote_ns as f64 / 1_000_000.0
            );
        }
        Ok(())
    }

    fn try_compute_ssm_layer(
        &mut self,
        layer: usize,
        xb: &[f32],
        xb2: &mut [f32],
    ) -> Result<bool, String> {
        let Some(ssm_idx) = self.ssm_worker_index else {
            return Ok(false);
        };
        let dim = xb.len();
        let request = SsmLayerRequest {
            layer,
            xb: xb.to_vec(),
        };
        let payload = encode_ssm_layer_request(&request);
        let worker = &mut self.remote_workers[ssm_idx];
        let request_id = worker.next_request_id;
        worker.next_request_id = worker.next_request_id.wrapping_add(1).max(1);
        worker
            .connection
            .send_message(FrameKind::SsmLayerRequest, request_id, &payload)
            .map_err(|e| format!("SSM send to worker '{}' failed: {e}", worker.address))?;
        let pending = wait_for_ssm_response(worker, request_id)
            .map_err(|e| format!("SSM recv from worker '{}' failed: {e}", worker.address))?;
        let _ = pending.wire_received;
        if pending.response.xb2.len() != dim {
            Err(format!(
                "SSM response dim mismatch: got {}, expected {dim}",
                pending.response.xb2.len()
            ))
        } else {
            xb2[..dim].copy_from_slice(&pending.response.xb2);
            Ok(true)
        }
    }
}

/// Predict which experts will be selected for a future layer using the coordinator's
/// local copy of the gate tensor, matching the worker's group top-k softmax logic.
fn predict_moe_routing(
    activation: &[f32],
    gate: &QuantizedTensor,
    n_experts: usize,
    n_top: usize,
    moe_n_group: usize,
    moe_topk_group: usize,
    mapped: &[u8],
) -> Vec<usize> {
    let mut logits = vec![0.0f32; n_experts];
    if matmul_quantized_rows(&mut logits, activation, gate, 0, n_experts, mapped).is_err() {
        return Vec::new();
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut scores: Vec<f32> = logits.iter().map(|&v| (v - max_l).exp()).collect();
    let sum: f32 = scores.iter().sum();
    let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
    for s in &mut scores {
        *s *= inv;
    }

    let use_groups =
        moe_n_group > 1 && moe_topk_group < moe_n_group && n_experts.is_multiple_of(moe_n_group);
    let group_size = if use_groups {
        n_experts / moe_n_group
    } else {
        n_experts
    };
    let n_groups = if use_groups { moe_n_group } else { 1 };
    let mut selected_group = vec![true; n_groups];

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
    if n_take == 0 {
        return Vec::new();
    }
    candidates.select_nth_unstable_by(n_take - 1, |&a, &b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top = candidates[..n_take].to_vec();
    top.sort_unstable_by(|&a, &b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    top.into_iter().map(|(_, i)| i).collect()
}

/// Encode and send an expert batch request with an optional routing hint for the next layer.
/// Retries with reconnect on send failure. Does NOT wait for the response.
fn send_batch_request_with_hint(
    worker: &mut RemoteWorkerClient,
    layer: usize,
    activation_dtype: ActivationDtype,
    dim: usize,
    expert_ids: &[usize],
    activation: &[f32],
    hint_expert_ids: &[usize],
) -> Result<(u64, usize, Instant), String> {
    let request = ExpertBatchRequest {
        token_pos: 0,
        layer,
        activation_dtype,
        dim,
        expert_ids: expert_ids.to_vec(),
        activation: activation.to_vec(),
        hint_expert_ids: hint_expert_ids.to_vec(),
    };
    let payload = encode_expert_batch_request(&request).map_err(|err| format!("encode: {err}"))?;
    let wire_sent = DISTRIBUTED_FRAME_HEADER_LEN + payload.len();
    let mut last_error: Option<String> = None;
    for attempt in 0..BATCH_RETRY_ATTEMPTS {
        let request_id = worker.next_request_id;
        worker.next_request_id = worker.next_request_id.wrapping_add(1).max(1);
        match worker
            .connection
            .send_message(FrameKind::ExpertBatchRequest, request_id, &payload)
        {
            Ok(()) => return Ok((request_id, wire_sent, Instant::now())),
            Err(err) => {
                last_error = Some(err);
                if attempt + 1 < BATCH_RETRY_ATTEMPTS {
                    worker.reconnect()?;
                    continue;
                }
                break;
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        format!(
            "worker '{}' send failed without a specific transport error",
            worker.address
        )
    }))
}

fn send_speculation_advance(
    worker: &mut RemoteWorkerClient,
    layer: usize,
    activation_dtype: ActivationDtype,
    dim: usize,
    activation: &[f32],
    hint_expert_ids: &[usize],
) -> Result<(), String> {
    let frame = crate::engine::distributed::protocol::SpeculationAdvanceFrame {
        layer,
        activation_dtype,
        dim,
        activation: activation.to_vec(),
        hint_expert_ids: hint_expert_ids.to_vec(),
    };
    let payload = encode_speculation_advance(&frame)
        .map_err(|err| format!("encode speculation advance: {err}"))?;
    let wire_sent = DISTRIBUTED_FRAME_HEADER_LEN + payload.len();
    worker
        .connection
        .send_message(FrameKind::SpeculationAdvance, 0, &payload)?;
    worker.stats.bytes_sent += wire_sent as u64;
    record_distributed_transport_bytes(wire_sent, 0);
    Ok(())
}

fn take_pushed_outputs(
    worker: &RemoteWorkerClient,
    layer: usize,
    expert_ids: &[usize],
) -> Option<Vec<Vec<f32>>> {
    let (lock, _) = &*worker.inbound;
    let mut state = lock.lock().unwrap();
    state
        .push_buffer
        .retain(|&l, _| l >= layer.saturating_sub(1));
    let (buf_ids, buf_outputs) = state.push_buffer.get(&layer)?;
    if !expert_ids.iter().all(|id| buf_ids.contains(id)) {
        return None;
    }
    let outputs = expert_ids
        .iter()
        .map(|id| {
            let pos = buf_ids.iter().position(|buf_id| buf_id == id).unwrap();
            buf_outputs[pos].clone()
        })
        .collect();
    state.push_buffer.remove(&layer);
    Some(outputs)
}

fn wait_for_batch_response(
    worker: &RemoteWorkerClient,
    request_id: u64,
) -> Result<PendingBatchResponse, String> {
    let (lock, cvar) = &*worker.inbound;
    let mut state = lock.lock().unwrap();
    loop {
        if let Some(response) = state.responses.remove(&request_id) {
            return response;
        }
        if let Some(err) = &state.transport_error {
            return Err(err.clone());
        }
        state = cvar.wait(state).unwrap();
    }
}

fn wait_for_ssm_response(
    worker: &RemoteWorkerClient,
    request_id: u64,
) -> Result<PendingSsmResponse, String> {
    let (lock, cvar) = &*worker.inbound;
    let mut state = lock.lock().unwrap();
    loop {
        if let Some(response) = state.ssm_responses.remove(&request_id) {
            return response;
        }
        if let Some(err) = &state.transport_error {
            return Err(err.clone());
        }
        state = cvar.wait(state).unwrap();
    }
}

/// Phase 2: receive and decode the expert batch response from a worker.
/// `wire_sent` and `wait_start` come from the matching `send_batch_request` call.
fn recv_batch_response(
    worker: &mut RemoteWorkerClient,
    request_id: u64,
    layer: usize,
    dim: usize,
    expert_ids: &[usize],
    wire_sent: usize,
    wait_start: Instant,
) -> Result<RemoteBatchReply, String> {
    let pending = wait_for_batch_response(worker, request_id)?;
    let wait_ns = wait_start.elapsed().as_nanos() as u64;
    let wire_received = pending.wire_received;
    let response = pending.response;
    if response.layer != layer || response.dim != dim {
        Err(format!(
            "invalid response shape: layer {} dim {}",
            response.layer, response.dim
        ))
    } else {
        worker.stats.record_request(
            expert_ids.len(),
            wire_sent,
            wire_received,
            wait_ns,
            response.spec_hit,
        );
        Ok((
            response.outputs,
            wire_sent,
            wire_received,
            wait_ns,
            response.spec_hit,
        ))
    }
}
