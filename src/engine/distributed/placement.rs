use crate::engine::io::find_gguf_tensor;
use crate::engine::kernels::{get_block_size, get_type_size};
use crate::engine::types::{Config, GGUFFile, Gguftensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClusterNodeRole {
    Coordinator,
    Worker,
}

impl ClusterNodeRole {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::Worker => "worker",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ClusterNodeConfig {
    pub(crate) id: String,
    pub(crate) address: String,
    pub(crate) role: ClusterNodeRole,
    pub(crate) logical_cpu_count: Option<usize>,
    pub(crate) discovered_memory_bytes: Option<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct ClusterConfig {
    pub(crate) node: Vec<ClusterNodeConfig>,
}

impl ClusterConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.node.is_empty() {
            return Err("cluster config must contain at least one [[node]] entry".to_string());
        }
        let coordinator_count = self
            .node
            .iter()
            .filter(|node| node.role == ClusterNodeRole::Coordinator)
            .count();
        if coordinator_count != 1 {
            return Err(format!(
                "cluster config must contain exactly one coordinator node, found {coordinator_count}"
            ));
        }
        for node in &self.node {
            if node.id.trim().is_empty() {
                return Err("cluster config contains a node with an empty id".to_string());
            }
            if node.address.trim().is_empty() {
                return Err(format!("cluster node '{}' has an empty address", node.id));
            }
        }
        Ok(())
    }

    pub(crate) fn coordinator_index(&self) -> Result<usize, String> {
        self.node
            .iter()
            .position(|node| node.role == ClusterNodeRole::Coordinator)
            .ok_or_else(|| "cluster config has no coordinator node".to_string())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MoePlacementInventory {
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) n_experts_used: usize,
    pub(crate) dim: usize,
    pub(crate) expert_hidden_dim: usize,
    pub(crate) checkpoint_bytes: usize,
    pub(crate) non_expert_checkpoint_bytes: usize,
    pub(crate) gate_bytes_per_expert: usize,
    pub(crate) up_bytes_per_expert: usize,
    pub(crate) down_bytes_per_expert: usize,
    pub(crate) total_bytes_per_expert: usize,
    pub(crate) total_bytes_all_experts: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct MoePlacementNode {
    pub(crate) node_id: String,
    pub(crate) role: ClusterNodeRole,
    pub(crate) address: String,
    pub(crate) logical_cpu_count: usize,
    pub(crate) capacity_bytes: usize,
    pub(crate) reserved_bytes: usize,
    pub(crate) assigned_expert_count: usize,
    pub(crate) assigned_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct MoePlacementPlan {
    pub(crate) inventory: MoePlacementInventory,
    pub(crate) coordinator_node_id: String,
    pub(crate) nodes: Vec<MoePlacementNode>,
    pub(crate) expert_node_indices: Vec<usize>,
}

impl MoePlacementPlan {
    pub(crate) fn assigned_experts_for_node(
        &self,
        node_index: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let n_experts = self.inventory.n_experts;
        self.expert_node_indices
            .iter()
            .enumerate()
            .filter(move |(_, assigned)| **assigned == node_index)
            .map(move |(flat_idx, _)| (flat_idx / n_experts, flat_idx % n_experts))
    }
}

fn node_assignable_capacity_bytes(node: &MoePlacementNode) -> usize {
    node.capacity_bytes.saturating_sub(node.reserved_bytes)
}

fn node_weight(node: &MoePlacementNode) -> u128 {
    (node_assignable_capacity_bytes(node) as u128)
        .saturating_mul(node.logical_cpu_count.max(1) as u128)
}

fn tensor_n_elements(tensor: &Gguftensor) -> usize {
    let mut n_elements = 1usize;
    for i in 0..tensor.n_dims as usize {
        n_elements = n_elements.saturating_mul(tensor.ne[i] as usize);
    }
    n_elements
}

fn tensor_storage_bytes(tensor: &Gguftensor) -> Result<usize, String> {
    let n_elements = tensor_n_elements(tensor);
    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if block_size == 0 || type_size == 0 {
        return Err(format!(
            "unsupported GGUF tensor type {} for tensor '{}'",
            tensor.ttype.0, tensor.name
        ));
    }
    if !n_elements.is_multiple_of(block_size) {
        return Err(format!(
            "tensor '{}' element count {} is not divisible by block size {}",
            tensor.name, n_elements, block_size
        ));
    }
    Ok((n_elements / block_size) * type_size)
}

fn layer_expert_tensor_bytes(gguf: &GGUFFile, layer: usize, suffix: &str) -> Result<usize, String> {
    let name = format!("blk.{layer}.{suffix}");
    let tensor =
        find_gguf_tensor(gguf, &name).ok_or_else(|| format!("tensor not found: {name}"))?;
    tensor_storage_bytes(tensor)
}

fn available_capacity_bytes(node: &MoePlacementNode) -> Option<usize> {
    let used = node.reserved_bytes.checked_add(node.assigned_bytes)?;
    node.capacity_bytes.checked_sub(used)
}

fn can_fit_additional_expert(node: &MoePlacementNode, expert_bytes: usize) -> bool {
    available_capacity_bytes(node)
        .map(|remaining| remaining >= expert_bytes)
        .unwrap_or(false)
}

fn pick_next_node(
    nodes: &[MoePlacementNode],
    candidate_indices: &[usize],
    ideal_assigned_bytes: &[usize],
    expert_bytes: usize,
) -> Option<usize> {
    let mut best_idx: Option<usize> = None;
    for &idx in candidate_indices {
        let candidate = &nodes[idx];
        if !can_fit_additional_expert(candidate, expert_bytes) {
            continue;
        }
        let candidate_deficit = ideal_assigned_bytes[idx].saturating_sub(candidate.assigned_bytes);
        match best_idx {
            None => best_idx = Some(idx),
            Some(current_best_idx) => {
                let best = &nodes[current_best_idx];
                let best_deficit =
                    ideal_assigned_bytes[current_best_idx].saturating_sub(best.assigned_bytes);
                if candidate_deficit > best_deficit
                    || (candidate_deficit == best_deficit
                        && (candidate.assigned_bytes < best.assigned_bytes
                            || (candidate.assigned_bytes == best.assigned_bytes
                                && candidate.assigned_expert_count < best.assigned_expert_count)))
                {
                    best_idx = Some(idx);
                }
            }
        }
    }
    best_idx
}

fn assign_expert_to_node(
    nodes: &mut [MoePlacementNode],
    expert_node_indices: &mut [usize],
    flat_idx: usize,
    node_index: usize,
    expert_bytes: usize,
) {
    expert_node_indices[flat_idx] = node_index;
    nodes[node_index].assigned_expert_count += 1;
    nodes[node_index].assigned_bytes += expert_bytes;
}

fn ideal_assigned_bytes_by_node(
    nodes: &[MoePlacementNode],
    total_expert_bytes: usize,
) -> Result<Vec<usize>, String> {
    let total_assignable_capacity = nodes
        .iter()
        .map(node_weight)
        .fold(0u128, |acc, weight| acc.saturating_add(weight));
    if total_assignable_capacity == 0 {
        return Err("cluster has zero assignable expert capacity after reservations".to_string());
    }

    let mut ideals = Vec::with_capacity(nodes.len());
    let mut assigned_total = 0usize;
    for node in nodes {
        let weighted = ((total_expert_bytes as u128)
            .checked_mul(node_weight(node))
            .ok_or_else(|| "ideal placement weighting overflow".to_string())?)
            / total_assignable_capacity;
        let ideal = usize::try_from(weighted)
            .map_err(|_| "ideal placement weighting conversion overflow".to_string())?;
        ideals.push(ideal.min(node_assignable_capacity_bytes(node)));
        assigned_total = assigned_total.saturating_add(*ideals.last().unwrap_or(&0));
    }

    let mut remaining = total_expert_bytes.saturating_sub(assigned_total);
    while remaining > 0 {
        let mut best_idx = None;
        let mut best_slack = 0usize;
        for (idx, node) in nodes.iter().enumerate() {
            let slack = node_assignable_capacity_bytes(node).saturating_sub(ideals[idx]);
            if slack > best_slack {
                best_slack = slack;
                best_idx = Some(idx);
            }
        }
        let Some(idx) = best_idx else {
            break;
        };
        if best_slack == 0 {
            break;
        }
        let add = remaining.min(best_slack);
        ideals[idx] = ideals[idx].saturating_add(add);
        remaining -= add;
    }
    Ok(ideals)
}

fn build_moe_placement_plan_from_inventory(
    inventory: MoePlacementInventory,
    cluster: &ClusterConfig,
) -> Result<MoePlacementPlan, String> {
    const UNASSIGNED_NODE_INDEX: usize = usize::MAX;
    let coordinator_index = cluster.coordinator_index()?;
    let mut nodes = cluster
        .node
        .iter()
        .map(|node| MoePlacementNode {
            node_id: node.id.clone(),
            role: node.role,
            address: node.address.clone(),
            logical_cpu_count: node.logical_cpu_count.unwrap_or(1),
            capacity_bytes: 0,
            reserved_bytes: 0,
            assigned_expert_count: 0,
            assigned_bytes: 0,
        })
        .collect::<Vec<_>>();

    for node in &mut nodes {
        node.capacity_bytes = cluster
            .node
            .iter()
            .find(|candidate| candidate.id == node.node_id)
            .and_then(|candidate| candidate.discovered_memory_bytes)
            .ok_or_else(|| {
                format!(
                    "node '{}' is missing discovered memory bytes for placement",
                    node.node_id
                )
            })?;
        node.reserved_bytes = if node.role == ClusterNodeRole::Coordinator {
            inventory.non_expert_checkpoint_bytes
        } else {
            0
        };
        if node.reserved_bytes > node.capacity_bytes {
            return Err(format!(
                "node '{}' cannot fit its baseline reservation: reserved_bytes={} capacity_bytes={}",
                node.node_id, node.reserved_bytes, node.capacity_bytes
            ));
        }
    }

    let total_assignable_capacity = nodes
        .iter()
        .map(node_assignable_capacity_bytes)
        .fold(0usize, |acc, bytes| acc.saturating_add(bytes));
    if total_assignable_capacity < inventory.total_bytes_all_experts {
        return Err(format!(
            "cluster expert capacity {} is smaller than required routed-expert bytes {}",
            total_assignable_capacity, inventory.total_bytes_all_experts
        ));
    }

    let ideal_bytes = ideal_assigned_bytes_by_node(&nodes, inventory.total_bytes_all_experts)?;
    let total_assignments = inventory
        .n_layers
        .checked_mul(inventory.n_experts)
        .ok_or_else(|| "expert assignment count overflow".to_string())?;
    let mut expert_node_indices = vec![UNASSIGNED_NODE_INDEX; total_assignments];
    let worker_indices = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.role == ClusterNodeRole::Worker)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    let all_node_indices = (0..nodes.len()).collect::<Vec<_>>();
    let mut worker_order = worker_indices.clone();

    for layer in 0..inventory.n_layers {
        let local_target = inventory.n_experts_used.max(1).min(inventory.n_experts);
        let local_offset = if inventory.n_experts == 0 {
            0
        } else {
            layer
                .checked_mul(local_target)
                .map(|value| value % inventory.n_experts)
                .unwrap_or(0)
        };
        let mut placed_local = 0usize;
        for local_rank in 0..local_target {
            if !can_fit_additional_expert(
                &nodes[coordinator_index],
                inventory.total_bytes_per_expert,
            ) {
                break;
            }
            let expert_idx = (local_offset + local_rank) % inventory.n_experts;
            let flat_idx = layer * inventory.n_experts + expert_idx;
            assign_expert_to_node(
                &mut nodes,
                &mut expert_node_indices,
                flat_idx,
                coordinator_index,
                inventory.total_bytes_per_expert,
            );
            placed_local += 1;
        }

        if !worker_order.is_empty() {
            let rotate_by = (layer + placed_local) % worker_order.len();
            worker_order.rotate_left(rotate_by);
        }

        for expert in 0..inventory.n_experts {
            let flat_idx = layer * inventory.n_experts + expert;
            if expert_node_indices[flat_idx] != UNASSIGNED_NODE_INDEX {
                continue;
            }
            let worker_node_index = if !worker_order.is_empty() {
                pick_next_node(
                    &nodes,
                    &worker_order,
                    &ideal_bytes,
                    inventory.total_bytes_per_expert,
                )
            } else {
                None
            };
            let node_index = worker_node_index
                .or_else(|| {
                    pick_next_node(
                        &nodes,
                        &all_node_indices,
                        &ideal_bytes,
                        inventory.total_bytes_per_expert,
                    )
                })
                .ok_or_else(|| {
                    format!(
                        "unable to place expert {} in layer {} within declared node memory budgets (expert_bytes={})",
                        expert, layer, inventory.total_bytes_per_expert
                    )
                })?;
            assign_expert_to_node(
                &mut nodes,
                &mut expert_node_indices,
                flat_idx,
                node_index,
                inventory.total_bytes_per_expert,
            );
        }
    }

    Ok(MoePlacementPlan {
        inventory,
        coordinator_node_id: cluster.node[coordinator_index].id.clone(),
        nodes,
        expert_node_indices,
    })
}

pub(crate) fn build_moe_placement_plan(
    gguf: &GGUFFile,
    config: &Config,
    cluster: &ClusterConfig,
) -> Result<MoePlacementPlan, String> {
    cluster.validate()?;
    if config.n_experts == 0 {
        return Err(
            "distributed planning requires a routed-MoE model with n_experts > 0".to_string(),
        );
    }
    if config.expert_hidden_dim == 0 {
        return Err(
            "distributed planning requires expert_hidden_dim > 0 for the current model".to_string(),
        );
    }

    let gate_layer_bytes = layer_expert_tensor_bytes(gguf, 0, "ffn_gate_exps.weight")?;
    let up_layer_bytes = layer_expert_tensor_bytes(gguf, 0, "ffn_up_exps.weight")?;
    let down_layer_bytes = layer_expert_tensor_bytes(gguf, 0, "ffn_down_exps.weight")?;

    if gate_layer_bytes % config.n_experts != 0
        || up_layer_bytes % config.n_experts != 0
        || down_layer_bytes % config.n_experts != 0
    {
        return Err(
            "expert tensor bytes are not evenly divisible by n_experts; packed expert layout assumption failed"
                .to_string(),
        );
    }

    let gate_bytes_per_expert = gate_layer_bytes / config.n_experts;
    let up_bytes_per_expert = up_layer_bytes / config.n_experts;
    let down_bytes_per_expert = down_layer_bytes / config.n_experts;
    let total_bytes_per_expert =
        gate_bytes_per_expert + up_bytes_per_expert + down_bytes_per_expert;
    let total_bytes_all_experts = total_bytes_per_expert
        .checked_mul(config.n_experts)
        .and_then(|value| value.checked_mul(config.n_layers))
        .ok_or_else(|| "expert byte count overflow while building placement plan".to_string())?;
    let checkpoint_bytes = gguf.mapped.len;
    let non_expert_checkpoint_bytes = checkpoint_bytes.saturating_sub(total_bytes_all_experts);

    let inventory = MoePlacementInventory {
        n_layers: config.n_layers,
        n_experts: config.n_experts,
        n_experts_used: config.n_experts_used,
        dim: config.dim,
        expert_hidden_dim: config.expert_hidden_dim,
        checkpoint_bytes,
        non_expert_checkpoint_bytes,
        gate_bytes_per_expert,
        up_bytes_per_expert,
        down_bytes_per_expert,
        total_bytes_per_expert,
        total_bytes_all_experts,
    };

    build_moe_placement_plan_from_inventory(inventory, cluster)
}

#[cfg(test)]
mod tests {
    use super::{
        ClusterConfig, ClusterNodeConfig, ClusterNodeRole, MoePlacementInventory,
        build_moe_placement_plan_from_inventory,
    };

    fn test_cluster(worker_a_gb: u64, worker_b_gb: u64) -> ClusterConfig {
        ClusterConfig {
            node: vec![
                ClusterNodeConfig {
                    id: "coordinator".to_string(),
                    address: "127.0.0.1:7000".to_string(),
                    role: ClusterNodeRole::Coordinator,
                    logical_cpu_count: Some(4),
                    discovered_memory_bytes: Some(4096),
                },
                ClusterNodeConfig {
                    id: "127.0.0.1:7001".to_string(),
                    address: "127.0.0.1:7001".to_string(),
                    role: ClusterNodeRole::Worker,
                    logical_cpu_count: Some(4),
                    discovered_memory_bytes: Some((worker_a_gb as usize) * 1024 * 1024 * 1024),
                },
                ClusterNodeConfig {
                    id: "127.0.0.1:7002".to_string(),
                    address: "127.0.0.1:7002".to_string(),
                    role: ClusterNodeRole::Worker,
                    logical_cpu_count: Some(8),
                    discovered_memory_bytes: Some((worker_b_gb as usize) * 1024 * 1024 * 1024),
                },
            ],
        }
    }

    fn test_inventory() -> MoePlacementInventory {
        MoePlacementInventory {
            n_layers: 4,
            n_experts: 8,
            n_experts_used: 2,
            dim: 4096,
            expert_hidden_dim: 1024,
            checkpoint_bytes: 10_000,
            non_expert_checkpoint_bytes: 1_000,
            gate_bytes_per_expert: 10,
            up_bytes_per_expert: 10,
            down_bytes_per_expert: 10,
            total_bytes_per_expert: 30,
            total_bytes_all_experts: 4 * 8 * 30,
        }
    }

    #[test]
    fn placement_keeps_a_local_expert_window_when_capacity_allows() {
        let plan = build_moe_placement_plan_from_inventory(test_inventory(), &test_cluster(1, 1))
            .expect("plan");
        let coordinator_index = plan
            .nodes
            .iter()
            .position(|node| node.role == ClusterNodeRole::Coordinator)
            .expect("coordinator");
        for layer in 0..plan.inventory.n_layers {
            let local_count = (0..plan.inventory.n_experts)
                .filter(|expert| {
                    plan.expert_node_indices[layer * plan.inventory.n_experts + *expert]
                        == coordinator_index
                })
                .count();
            assert!(
                local_count >= plan.inventory.n_experts_used,
                "expected at least {} local experts in layer {}, got {}",
                plan.inventory.n_experts_used,
                layer,
                local_count
            );
        }
    }

    #[test]
    fn placement_biases_more_experts_to_larger_workers() {
        let plan = build_moe_placement_plan_from_inventory(test_inventory(), &test_cluster(1, 3))
            .expect("plan");
        let worker_a = plan
            .nodes
            .iter()
            .find(|node| node.node_id == "127.0.0.1:7001")
            .expect("worker-a");
        let worker_b = plan
            .nodes
            .iter()
            .find(|node| node.node_id == "127.0.0.1:7002")
            .expect("worker-b");
        assert!(
            worker_b.assigned_expert_count > worker_a.assigned_expert_count,
            "expected larger worker to get more experts: worker-a={} worker-b={}",
            worker_a.assigned_expert_count,
            worker_b.assigned_expert_count
        );
    }
}
