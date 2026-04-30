use crate::engine::distributed::coordinator::compute_local_selected_experts;
use crate::engine::distributed::placement::{ClusterNodeRole, MoePlacementPlan};
use crate::engine::distributed::protocol::{
    ExpertBatchResponse, FrameKind, HelloFrame, ReadyFrame, decode_error_frame,
    decode_expert_batch_request, decode_hello_frame, encode_error_frame,
    encode_expert_batch_response, encode_ready_frame,
};
use crate::engine::distributed::transport::FramedConnection;
use crate::engine::types::{Config, GGUFFile, TransformerWeights, WorkerExpertWeights};
use std::net::TcpListener;
use std::time::Duration;

const DEFAULT_REMOTE_TIMEOUT_SECS: u64 = 30;

struct WorkerRuntime {
    gguf: GGUFFile,
    config: Config,
    weights: TransformerWeights,
    plan: MoePlacementPlan,
    node_index: usize,
}

impl WorkerRuntime {
    fn validate_hello(&self, hello: &HelloFrame) -> Result<ReadyFrame, String> {
        if hello.dim != self.config.dim
            || hello.n_layers != self.config.n_layers
            || hello.n_experts != self.config.n_experts
        {
            return Err("coordinator HELLO does not match worker model metadata".to_string());
        }
        Ok(ReadyFrame {
            dim: self.config.dim,
            n_layers: self.config.n_layers,
            n_experts: self.config.n_experts,
            activation_dtype: hello.activation_dtype,
        })
    }

    fn handle_request(
        &self,
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
        let n_experts = self.plan.inventory.n_experts;
        let mapped = self.gguf.mapped.as_slice();
        let mut outputs = Vec::with_capacity(request.expert_ids.len());
        for &expert_idx in &request.expert_ids {
            let plan_index = request
                .layer
                .checked_mul(n_experts)
                .and_then(|value| value.checked_add(expert_idx))
                .ok_or_else(|| "worker expert plan index overflow".to_string())?;
            let assigned_node = *self
                .plan
                .expert_node_indices
                .get(plan_index)
                .ok_or_else(|| "worker expert plan index out of bounds".to_string())?;
            if assigned_node != self.node_index {
                return Err(format!(
                    "expert {} in layer {} is not assigned to worker '{}'",
                    expert_idx, request.layer, self.plan.nodes[self.node_index].node_id
                ));
            }
            let mut output = vec![0.0f32; self.config.dim];
            compute_local_selected_experts(
                request.layer,
                &request.activation,
                &[(expert_idx, 1.0)],
                &mut output,
                &self.config,
                &self.weights,
                mapped,
                false,
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
    plan: MoePlacementPlan,
    node_id: &str,
) -> Result<WorkerRuntime, String> {
    let node_index = plan
        .nodes
        .iter()
        .position(|node| node.node_id == node_id)
        .ok_or_else(|| format!("node id '{}' was not found in cluster config", node_id))?;
    let node = &plan.nodes[node_index];
    if node.role != ClusterNodeRole::Worker {
        return Err(format!(
            "node '{}' has role '{}' but worker mode requires a worker node",
            node.node_id,
            node.role.as_str()
        ));
    }

    let worker_weights =
        crate::engine::weights::init_worker_expert_weights_from_gguf(&gguf, &config)?;
    let weights = inflate_worker_weights(worker_weights)?;

    Ok(WorkerRuntime {
        gguf,
        config,
        weights,
        plan,
        node_index,
    })
}

fn inflate_worker_weights(
    worker_weights: WorkerExpertWeights,
) -> Result<TransformerWeights, String> {
    Ok(TransformerWeights {
        token_embedding_table: Vec::new(),
        rms_att_weight: Vec::new(),
        rms_ffn_weight: Vec::new(),
        wq: Vec::new(),
        wk: Vec::new(),
        wv: Vec::new(),
        wo: Vec::new(),
        w1: Vec::new(),
        w2: Vec::new(),
        w3: Vec::new(),
        attn_qkv: Vec::new(),
        ssm_ba: Vec::new(),
        ssm_alpha: Vec::new(),
        ssm_beta: Vec::new(),
        ssm_conv1d: Vec::new(),
        ssm_a: Vec::new(),
        ssm_dt_bias: Vec::new(),
        ssm_norm: Vec::new(),
        moe_gate_inp: Vec::new(),
        moe_gate_exps: worker_weights.moe_gate_exps,
        moe_up_exps: worker_weights.moe_up_exps,
        moe_down_exps: worker_weights.moe_down_exps,
        moe_shared_gate_inp: Vec::new(),
        rms_final_weight: Vec::new(),
        wcls: Default::default(),
        wcls_is_embed: false,
        attn_q_bias: Vec::new(),
        attn_k_bias: Vec::new(),
        attn_v_bias: Vec::new(),
        attn_q_norm: Vec::new(),
        attn_k_norm: Vec::new(),
        attn_qk_norm_present: Vec::new(),
        attn_post_norm: Vec::new(),
        ffn_post_norm: Vec::new(),
        attn_post_norm_bias: Vec::new(),
        ffn_post_norm_bias: Vec::new(),
    })
}

pub(crate) fn run_worker_server(
    gguf: GGUFFile,
    config: Config,
    plan: MoePlacementPlan,
    node_id: &str,
) -> Result<(), String> {
    let runtime = build_worker_runtime(gguf, config, plan, node_id)?;
    let node = &runtime.plan.nodes[runtime.node_index];
    let listener = TcpListener::bind(&node.address)
        .map_err(|e| format!("failed to bind worker listener '{}': {e}", node.address))?;
    println!(
        "Distributed worker listening on {} with {} assigned experts",
        node.address, runtime.plan.nodes[runtime.node_index].assigned_expert_count
    );

    for stream in listener.incoming() {
        let stream = stream.map_err(|e| format!("worker accept failed: {e}"))?;
        let mut connection = FramedConnection::from_stream(
            stream,
            Duration::from_secs(DEFAULT_REMOTE_TIMEOUT_SECS),
        )?;
        let hello_message = connection.recv_message()?;
        if hello_message.kind != FrameKind::Hello {
            return Err("worker expected HELLO as first distributed frame".to_string());
        }
        let hello = decode_hello_frame(&hello_message.payload)?;
        match runtime.validate_hello(&hello) {
            Ok(ready) => {
                let payload = encode_ready_frame(&ready)?;
                connection.send_message(FrameKind::Ready, hello_message.request_id, &payload)?;
            }
            Err(err) => {
                let payload = encode_error_frame(&err)?;
                connection.send_message(FrameKind::Error, hello_message.request_id, &payload)?;
                continue;
            }
        }

        loop {
            let message = connection.recv_message()?;
            match message.kind {
                FrameKind::ExpertBatchRequest => {
                    let request = decode_expert_batch_request(&message.payload)?;
                    match runtime.handle_request(request) {
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
                    return Err(format!("worker received unexpected frame {:?}", other));
                }
            }
        }
    }

    Ok(())
}
