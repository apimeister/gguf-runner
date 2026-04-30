use crate::engine::io::find_gguf_tensor;
use crate::engine::kernels::{get_block_size, get_type_size};
use crate::engine::types::{Config, GGUFFile, Gguftensor};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
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

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ClusterNodeConfig {
    pub(crate) id: String,
    pub(crate) address: String,
    pub(crate) role: ClusterNodeRole,
    pub(crate) memory_gb: u64,
}

#[derive(Clone, Debug, Deserialize)]
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
            if node.memory_gb == 0 {
                return Err(format!(
                    "cluster node '{}' must declare memory_gb > 0",
                    node.id
                ));
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

    pub(crate) fn node_index_by_id(&self, node_id: &str) -> Option<usize> {
        self.node.iter().position(|node| node.id == node_id)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MoePlacementInventory {
    pub(crate) n_layers: usize,
    pub(crate) n_experts: usize,
    pub(crate) n_experts_used: usize,
    pub(crate) dim: usize,
    pub(crate) expert_hidden_dim: usize,
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
    pub(crate) memory_gb: u64,
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

fn pick_next_node(nodes: &[MoePlacementNode]) -> usize {
    let mut best_idx = 0usize;
    for idx in 1..nodes.len() {
        let best = &nodes[best_idx];
        let candidate = &nodes[idx];
        if candidate.assigned_bytes < best.assigned_bytes
            || (candidate.assigned_bytes == best.assigned_bytes
                && candidate.assigned_expert_count < best.assigned_expert_count)
        {
            best_idx = idx;
        }
    }
    best_idx
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

    let inventory = MoePlacementInventory {
        n_layers: config.n_layers,
        n_experts: config.n_experts,
        n_experts_used: config.n_experts_used,
        dim: config.dim,
        expert_hidden_dim: config.expert_hidden_dim,
        gate_bytes_per_expert,
        up_bytes_per_expert,
        down_bytes_per_expert,
        total_bytes_per_expert,
        total_bytes_all_experts,
    };

    let coordinator_index = cluster.coordinator_index()?;
    let mut nodes = cluster
        .node
        .iter()
        .map(|node| MoePlacementNode {
            node_id: node.id.clone(),
            role: node.role,
            address: node.address.clone(),
            memory_gb: node.memory_gb,
            assigned_expert_count: 0,
            assigned_bytes: 0,
        })
        .collect::<Vec<_>>();

    let total_assignments = config
        .n_layers
        .checked_mul(config.n_experts)
        .ok_or_else(|| "expert assignment count overflow".to_string())?;
    let mut expert_node_indices = vec![0usize; total_assignments];

    for layer in 0..config.n_layers {
        for expert in 0..config.n_experts {
            let node_index = if expert == 0 {
                coordinator_index
            } else {
                pick_next_node(&nodes)
            };
            let flat_idx = layer * config.n_experts + expert;
            expert_node_indices[flat_idx] = node_index;
            nodes[node_index].assigned_expert_count += 1;
            nodes[node_index].assigned_bytes += total_bytes_per_expert;
        }
    }

    Ok(MoePlacementPlan {
        inventory,
        coordinator_node_id: cluster.node[coordinator_index].id.clone(),
        nodes,
        expert_node_indices,
    })
}
