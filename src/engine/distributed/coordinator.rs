use crate::engine::distributed::placement::{ClusterNodeRole, MoePlacementPlan};
use crate::engine::distributed::protocol::{
    ActivationDtype, ExpertBatchRequest, FrameKind, HelloFrame, ReadyFrame, decode_error_frame,
    decode_expert_batch_response, decode_ready_frame, encode_expert_batch_request,
    encode_hello_frame,
};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::kernels::{axpy_inplace, matmul_quantized_rows, silu_and_mul_inplace};
use crate::engine::types::{Config, QuantizedTensor, TransformerWeights};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use std::collections::HashMap;
use std::time::Duration;

const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

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

    if selected.len() >= 2 && allow_parallel {
        let mut contribs = vec![0.0f32; selected.len() * dim];
        let results = contribs
            .par_chunks_mut(dim)
            .zip(selected.par_iter())
            .map_init(
                || (vec![0.0f32; expert_hidden], vec![0.0f32; expert_hidden]),
                |(gate_scratch, up_scratch), (contrib, &(expert_idx, route_weight))| {
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
        axpy_inplace(&mut output[..dim], route_weight, &expert_output);
    }
    Ok(())
}

struct RemoteWorkerClient {
    node_index: usize,
    connection: FramedConnection,
}

impl RemoteWorkerClient {
    fn shutdown(&mut self) {
        let _ = self.connection.send_message(FrameKind::Shutdown, 0, &[]);
    }
}

pub(crate) struct DistributedMoeCoordinator {
    plan: MoePlacementPlan,
    activation_dtype: ActivationDtype,
    next_request_id: u64,
    remote_workers: Vec<RemoteWorkerClient>,
}

impl DistributedMoeCoordinator {
    pub(crate) fn connect(
        plan: MoePlacementPlan,
        activation_dtype: ActivationDtype,
    ) -> Result<Self, String> {
        let timeout = Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS);
        let mut remote_workers = Vec::new();
        let hello = HelloFrame {
            dim: plan.inventory.dim,
            n_layers: plan.inventory.n_layers,
            n_experts: plan.inventory.n_experts,
            activation_dtype,
        };
        let hello_payload = encode_hello_frame(&hello)?;
        for (node_index, node) in plan.nodes.iter().enumerate() {
            if node.role != ClusterNodeRole::Worker || node.assigned_expert_count == 0 {
                continue;
            }
            let mut connection = FramedConnection::connect(&node.address, timeout)?;
            connection.send_message(FrameKind::Hello, 0, &hello_payload)?;
            let message = connection.recv_message()?;
            match message.kind {
                FrameKind::Ready => {
                    let ready = decode_ready_frame(&message.payload)?;
                    Self::validate_ready_frame(&hello, &ready, &node.node_id)?;
                }
                FrameKind::Error => {
                    return Err(format!(
                        "worker '{}' rejected hello: {}",
                        node.node_id,
                        decode_error_frame(&message.payload)?
                    ));
                }
                other => {
                    return Err(format!(
                        "worker '{}' returned unexpected frame {:?} during handshake",
                        node.node_id, other
                    ));
                }
            }
            remote_workers.push(RemoteWorkerClient {
                node_index,
                connection,
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
        node_id: &str,
    ) -> Result<(), String> {
        if hello.dim != ready.dim
            || hello.n_layers != ready.n_layers
            || hello.n_experts != ready.n_experts
            || hello.activation_dtype != ready.activation_dtype
        {
            return Err(format!(
                "worker '{}' READY frame does not match coordinator HELLO",
                node_id
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
            client
                .connection
                .send_message(FrameKind::ExpertBatchRequest, request_id, &payload)?;
            let message = client.connection.recv_message()?;
            if message.request_id != request_id {
                return Err(format!(
                    "worker '{}' returned mismatched request id: got {}, expected {}",
                    self.plan.nodes[client.node_index].node_id, message.request_id, request_id
                ));
            }
            match message.kind {
                FrameKind::ExpertBatchResponse => {
                    let response = decode_expert_batch_response(&message.payload)?;
                    if response.layer != layer || response.dim != dim {
                        return Err(format!(
                            "worker '{}' returned invalid response shape for layer {}",
                            self.plan.nodes[client.node_index].node_id, layer
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
                        self.plan.nodes[client.node_index].node_id,
                        decode_error_frame(&message.payload)?
                    ));
                }
                other => {
                    return Err(format!(
                        "worker '{}' returned unexpected frame {:?}",
                        self.plan.nodes[client.node_index].node_id, other
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
