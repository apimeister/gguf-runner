#![allow(unsafe_op_in_unsafe_fn)]

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
use std::sync::mpsc;
use std::thread;
use std::thread::sleep;
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

/// Results sent back from a concurrent batch worker task.
#[derive(Debug, Clone)]
struct BatchTaskResult {
    node_address: String,
    expert_ids: Vec<usize>,
    outputs: Vec<Vec<f32>>,
    /// Activation buffer returned to pool after use.
    activation: Vec<f32>,
    bytes_sent: usize,
    bytes_received: usize,
    wait_ns: u64,
    error: Option<String>,
}

#[allow(dead_code)]
struct RemoteWorkerClient {
    node_index: usize,
    address: String,
    hello: HelloFrame,
    timeout: Duration,
    connection: FramedConnection,
    stats: RemoteWorkerStats,
}

#[allow(dead_code)]
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
    activation_buffer_pool: ActivationBufferPool,
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
    fn put_buffer(&mut self, mut buffer: Vec<f32>) {
        if buffer.len() == self.buffer_size {
            // Clear the buffer to prevent stale data from leaking between batches
            fill_simd_inplace(&mut buffer);
            self.buffers.push(buffer);
        }
    }
}

/// Fill buffer with zeros using SIMD (AVX-2/AVX-512) when size permits.
/// Avoids per-element branch overhead for large activations.
#[inline]
fn fill_simd_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 8 {
        return unsafe {
            fill_simd_x86_64(x);
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        x.fill(0.0);
    }
}

/// x86_64 dispatch for SIMD fill.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn fill_simd_x86_64(x: &mut [f32]) {
    use crate::engine::switches::use_x86_avx512f;
    if use_x86_avx512f() {
        fill_simd_avx512(x);
    } else {
        fill_simd_avx2(x);
    }
}

/// AVX-2 (8-wide) SIMD fill with zeros.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn fill_simd_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let ptr = x.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= n {
        _mm256_storeu_ps(ptr.add(i), zero);
        i += 8;
    }

    // Scalar tail
    while i < n {
        ptr.add(i).write(0.0);
        i += 1;
    }
}

/// AVX-512 (16-wide) SIMD fill with zeros.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn fill_simd_avx512(x: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let ptr = x.as_mut_ptr();
    let zero = _mm512_setzero_ps();
    let mut i = 0usize;

    while i + 16 <= n {
        _mm512_storeu_ps(ptr.add(i), zero);
        i += 16;
    }

    // Scalar tail
    while i < n {
        ptr.add(i).write(0.0);
        i += 1;
    }
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
        let dim = plan.inventory.dim;
        let max_concurrent = plan
            .nodes
            .iter()
            .filter(|n| n.role == ClusterNodeRole::Worker)
            .count()
            .max(1);

        Ok(Self {
            plan,
            activation_dtype,
            next_request_id: 1,
            remote_workers,
            activation_buffer_pool: ActivationBufferPool::new(dim, max_concurrent),
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

        // ─── Concurrent batch: send to all workers in parallel, then collect ───
        // Each spawned thread creates its own fresh connection and sends the batch.
        // Results come back via mpsc channels.
        let activation_dtype = self.activation_dtype;
        let plan = &self.plan;
        let mut batch_handles: Vec<(mpsc::Receiver<BatchTaskResult>, usize)> = Vec::new();

        for (node_index, expert_ids) in &remote_selected {
            let node_address = plan.nodes[*node_index].address.clone();
            let address = self
                .remote_workers
                .iter()
                .find(|worker| worker.node_index == *node_index)
                .ok_or_else(|| {
                    format!(
                        "distributed worker '{}' is missing an active coordinator connection",
                        node_address
                    )
                })?
                .address
                .clone();
            let stats = RemoteWorkerStats::default();
            // Use pre-allocated buffer instead of allocating a new vector each time
            let activation = self.activation_buffer_pool.get_buffer();
            let expert_ids = expert_ids.clone();
            let n_layers = plan.inventory.n_layers;
            let n_experts = plan.inventory.n_experts;
            let assigned_experts = plan
                .assigned_experts_for_node(*node_index)
                .collect::<Vec<_>>();

            let (tx, rx) = mpsc::channel::<BatchTaskResult>();
            let _handle = thread::spawn(move || {
                let input = BatchTaskInput {
                    address,
                    activation_dtype,
                    layer,
                    expert_ids,
                    activation,
                    dim,
                    n_layers,
                    n_experts,
                    assigned_experts,
                    stats,
                    node_address,
                };
                run_batch(input, tx);
            });
            batch_handles.push((rx, *node_index));
        }

        // Collect results from all concurrent batches
        let mut remote_outputs: HashMap<usize, Vec<f32>> = HashMap::new();
        for (rx, node_index) in batch_handles {
            let task_result = match rx.recv() {
                Ok(result) => result,
                Err(_) => {
                    return Err(format!(
                        "worker '{}' batch channel disconnected",
                        plan.nodes[node_index].address
                    ));
                }
            };

            // Return activation buffer to pool for reuse on next token
            self.activation_buffer_pool
                .put_buffer(task_result.activation.clone());

            if let Some(ref err) = task_result.error {
                return Err(format!(
                    "worker '{}' batch failed: {}",
                    task_result.node_address, err
                ));
            }

            // Record profiling counters
            if task_result.bytes_sent > 0 {
                record_distributed_remote_request(
                    task_result.expert_ids.len(),
                    task_result.bytes_sent,
                    task_result.bytes_received,
                    task_result.wait_ns,
                );
            }

            // Populate remote_outputs with decoded expert outputs
            for (expert_idx, values) in task_result.expert_ids.into_iter().zip(task_result.outputs)
            {
                remote_outputs.insert(expert_idx, values);
            }
        }
        self.next_request_id = 1; // Reset per-layer

        for &(expert_idx, route_weight) in selected {
            if let Some(values) = remote_outputs.get(&expert_idx) {
                axpy_inplace(&mut output[..dim], route_weight, values);
            }
        }
        Ok(())
    }
}

/// Runs a single batch task: connects to worker, sends request, collects response.
fn run_batch(input: BatchTaskInput, tx: mpsc::Sender<BatchTaskResult>) {
    let BatchTaskInput {
        address,
        activation_dtype,
        layer,
        expert_ids,
        activation,
        dim,
        n_layers,
        n_experts,
        assigned_experts,
        mut stats,
        node_address,
    } = input;

    // Build the HelloFrame for the worker handshake
    let hello = HelloFrame {
        node_address: address.clone(),
        dim,
        n_layers,
        n_experts,
        activation_dtype,
        assigned_experts,
    };

    // Connect to the worker
    let conn_result = connect_worker_with_hello(
        &address,
        Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS),
        &hello,
        &address,
    );
    let mut connection = match conn_result {
        Ok((conn, _bsent, _brecv)) => conn,
        Err(err) => {
            let err_msg = err.clone();
            let addr = node_address.clone();
            let exp_ids = expert_ids.clone();
            let result_clone = BatchTaskResult {
                node_address: addr,
                expert_ids: exp_ids,
                outputs: vec![],
                activation: activation.clone(),
                bytes_sent: 0,
                bytes_received: 0,
                wait_ns: 0,
                error: Some(err_msg),
            };
            tx.send(result_clone).expect("channel dropped");
            return;
        }
    };

    // Send request, retry on failure, decode response
    let mut last_error: Option<String> = None;
    let mut decoded_outputs: Vec<Vec<f32>> = Vec::new();
    let mut wire_sent = 0usize;
    let mut wire_received = 0usize;
    let mut wait_ns = 0u64;

    for attempt in 0..BATCH_RETRY_ATTEMPTS {
        let request = ExpertBatchRequest {
            token_pos: 0,
            layer,
            activation_dtype,
            dim,
            expert_ids: expert_ids.clone(),
            activation: activation.clone(),
        };
        let payload = match encode_expert_batch_request(&request) {
            Ok(p) => p,
            Err(err) => {
                last_error = Some(format!("encode: {}", err));
                continue;
            }
        };

        wire_sent = DISTRIBUTED_FRAME_HEADER_LEN + payload.len();
        let wait_start = Instant::now();
        match connection.send_message(FrameKind::ExpertBatchRequest, 0, &payload) {
            Ok(()) => {}
            Err(err) => {
                last_error = Some(err);
                if attempt + 1 < BATCH_RETRY_ATTEMPTS {
                    continue;
                }
                break;
            }
        }
        match connection.recv_message() {
            Ok(message) => {
                wait_ns = wait_start.elapsed().as_nanos() as u64;
                wire_received = DISTRIBUTED_FRAME_HEADER_LEN + message.payload.len();
                stats.record_request(expert_ids.len(), wire_sent, wire_received, wait_ns);

                match message.kind {
                    FrameKind::ExpertBatchResponse => {
                        match decode_expert_batch_response(&message.payload) {
                            Ok(response) => {
                                if response.layer == layer && response.dim == dim {
                                    decoded_outputs = response.outputs;
                                } else {
                                    last_error = Some(format!(
                                        "invalid response shape: layer {} dim {}",
                                        response.layer, response.dim
                                    ));
                                }
                            }
                            Err(err) => last_error = Some(err),
                        }
                    }
                    FrameKind::Error => {
                        if let Ok(msg) = decode_error_frame(&message.payload) {
                            last_error = Some(msg);
                        } else {
                            last_error = Some("unknown error frame".to_string());
                        }
                    }
                    other => {
                        last_error = Some(format!("unexpected frame: {:?}", other));
                    }
                }
                break;
            }
            Err(err) => {
                last_error = Some(err);
                if attempt + 1 < BATCH_RETRY_ATTEMPTS {
                    continue;
                }
                break;
            }
        }
    }

    let result_clone = BatchTaskResult {
        node_address,
        expert_ids: expert_ids.clone(),
        outputs: decoded_outputs,
        activation: activation.clone(),
        bytes_sent: wire_sent,
        bytes_received: wire_received,
        wait_ns,
        error: last_error,
    };

    tx.send(result_clone).expect("channel dropped");
}
/// Input data for a concurrent batch worker task.
/// Spawned via `std::thread::spawn`.
struct BatchTaskInput {
    address: String,
    activation_dtype: ActivationDtype,
    layer: usize,
    expert_ids: Vec<usize>,
    activation: Vec<f32>,
    dim: usize,
    n_layers: usize,
    n_experts: usize,
    assigned_experts: Vec<(usize, usize)>,
    stats: RemoteWorkerStats,
    node_address: String,
}
