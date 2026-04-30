# Distributed Cluster Usage

This document explains how to try the current distributed MoE flow in `gguf-runner`.

The implementation is aimed at routed-MoE GGUF models where the coordinator keeps the dense
runtime path local and forwards selected experts to worker nodes over TCP.

Current status:

- experimental
- fail-fast on worker/network/protocol errors
- intended for routed-MoE models such as Qwen3 MoE / Qwen3.5 A3B/A10B style checkpoints
- worker loading still uses full expert tensors per layer rather than row-range slicing, so memory
  planning is still important

## What Runs Where

- Coordinator:
  - normal `gguf-runner` generation process
  - owns prompt handling, tokenization, sampling, attention, KV cache, dense weights, routing, and logits
  - uses `--cluster <cluster.toml>` during generation
- Worker:
  - dedicated expert-serving process
  - validates cluster assignment, listens on its configured address, receives expert batch requests,
    and returns expert outputs
  - started with `--distributed-worker`

## Prerequisites

- the same GGUF model file must be available on coordinator and workers
- all nodes must use a compatible `gguf-runner` build
- the cluster file must describe exactly one coordinator and one or more workers
- worker addresses must be reachable from the coordinator

## Cluster File

Example `cluster.toml`:

```toml
[[node]]
id = "coordinator"
address = "192.168.10.10:7000"
role = "coordinator"
memory_gb = 32

[[node]]
id = "worker-a"
address = "192.168.10.11:7000"
role = "worker"
memory_gb = 32

[[node]]
id = "worker-b"
address = "192.168.10.12:7000"
role = "worker"
memory_gb = 32

[[node]]
id = "worker-c"
address = "192.168.10.13:7000"
role = "worker"
memory_gb = 32
```

Fields:

- `id`: stable node identifier used by `--node-id`
- `address`: TCP bind/connect address for worker service traffic
- `role`: `coordinator` or `worker`
- `memory_gb`: operator-declared memory budget used by planning output

## Step 1: Inspect Placement

Before starting workers, inspect the current placement plan:

```bash
gguf-runner \
  --model ./Qwen3.5-122B-A10B-Q4.gguf \
  --cluster ./cluster.toml \
  --distributed-plan
```

This prints:

- model MoE inventory
- bytes per expert per layer
- total routed-expert bytes
- per-node assigned expert counts and byte estimates

Use this as a sanity check before booting the cluster.

## Step 2: Start Worker Processes

Start one worker per `role = "worker"` entry.

Example for `worker-a`:

```bash
gguf-runner \
  --model ./Qwen3.5-122B-A10B-Q4.gguf \
  --cluster ./cluster.toml \
  --node-id worker-a \
  --distributed-worker
```

Repeat on each worker host with its own `--node-id`.

What happens at startup:

- the worker validates that `--node-id` exists and is marked as a worker
- it loads model metadata and expert tensors
- it computes the same placement plan locally
- it binds to the configured `address`
- it waits for the coordinator handshake

## Step 3: Run the Coordinator

Run normal generation on the coordinator, but include `--cluster`.

Example:

```bash
gguf-runner \
  --model ./Qwen3.5-122B-A10B-Q4.gguf \
  --cluster ./cluster.toml \
  --prompt "Write a concise explanation of Rust ownership." \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --max-tokens 128 \
  --show-timings
```

Coordinator behavior:

- loads the model normally
- builds the placement plan
- opens persistent TCP connections to worker nodes with assigned experts
- performs a HELLO/READY handshake
- routes selected remote experts during decode/prefill

## Typical Startup Sequence

1. Copy the same model to all hosts.
2. Copy the same `cluster.toml` to all hosts.
3. Run `--distributed-plan` once to inspect assignments.
4. Start all worker processes.
5. Run the coordinator generation command with `--cluster`.

## Failure Model

Current behavior is fail-fast:

- worker disconnect: generation aborts
- handshake mismatch: generation aborts
- request/response shape mismatch: generation aborts
- timeout while waiting for a worker: generation aborts

There is no automatic retry or local fallback today.

## Current Limitations

- worker loading is not yet sliced by assigned row ranges
- no localhost correctness test harness is wired yet
- no distributed performance counter summary is exposed yet
- transport dtype is currently fixed by the coordinator implementation
- only routed experts are distributed; shared experts stay local

## Troubleshooting

If `--distributed-plan` fails:

- verify the model is a routed-MoE GGUF
- verify `cluster.toml` contains exactly one coordinator
- verify tensor names/layout match the expected Qwen-style MoE tensor groups

If a worker does not start:

- verify `--node-id` matches a worker entry exactly
- verify the configured `address` can be bound on that host
- verify the model path points to a readable GGUF file

If the coordinator fails during startup:

- verify all workers are already listening
- verify the coordinator can reach each worker `address`
- verify all nodes use the same model checkpoint and cluster file

## Related Docs

- [Distributed MoE Plan](./distributed-moe-plan.md)
- [Features](./features.md)
- [Performance Notes](./performance.md)
