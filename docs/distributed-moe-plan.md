# Distributed MoE Execution Plan

This document sketches an implementation path for running large MoE GGUF models across multiple
machines, with an initial target of:

- Model: Qwen3.5-122B-A10B class MoE checkpoint
- Quantization: 4-bit GGUF, roughly 76 GB on disk
- Cluster: 4 Intel Ultra 125H hosts, 32 GB RAM each
- Network: 10 Gbit Ethernet
- Transport tensors: fp16 or bf16 activations over TCP

The goal is not generic tensor parallelism. The first useful target is expert offload: keep dense
runtime flow on one coordinator, shard routed expert weights across worker nodes, and send only the
current token/layer activation to remote expert workers.

## Design Summary

Use a coordinator/worker layout:

- Coordinator:
  - owns CLI, tokenization, sampling, decode loop, attention, KV cache, dense weights, routing gate,
    and final logits
  - computes MoE routing for each layer
  - runs local experts assigned to itself
  - sends selected remote expert requests to workers
  - accumulates weighted expert outputs back into the residual path
- Expert workers:
  - load only assigned expert tensors, plus minimal model metadata needed to validate shapes
  - wait for per-token expert requests from the coordinator
  - run `gate_proj`, `up_proj`, activation, and `down_proj` for requested experts
  - return fp16/bf16 expert output vectors

This keeps attention KV cache local to the coordinator and avoids synchronizing every attention
read/write across the network.

## Why Not Tensor Parallelism First

Tensor parallelism communicates after many dense matmul slices. On 10GbE this is still too chatty
for single-token decode latency. Expert offload communicates only at routed MoE boundaries and only
for selected experts. That maps better to the sparse structure of A10B-style active MoE.

Layer pipeline parallelism can help fit memory but gives limited speedup for a single decode stream,
because token generation is sequential. It may be useful later for batched serving.

## Transport Precision

Initial transport should use fp16 or bf16, not TurboQuant KV format.

Rationale:

- KV-cache quantization is designed for repeated attention key/value reads.
- MoE activation transport sends short-lived hidden vectors and expert outputs.
- The expert output is added into the residual stream, so very low-bit transport can visibly affect
  quality.
- fp16/bf16 is simple and should be cheap enough on 10GbE.

Later optional formats:

- int8 per-vector scale for lower network traffic
- int8 per-block scale for better accuracy
- 3-4 bit transport only after quality benchmarking

## Expected Network Shape

For each selected remote expert:

1. coordinator sends one hidden vector of length `dim`
2. worker returns one expert output vector of length `dim`

For `dim = 8192`:

- fp16/bf16 request: 16 KB
- fp16/bf16 response: 16 KB
- total per remote expert: 32 KB, plus framing

If a layer selects 8 experts and 6 are remote on average:

- about 192 KB per MoE layer
- with 60-80 MoE layers, about 12-15 MB per generated token

At practical 10GbE throughput, raw bandwidth cost is around 10-20 ms/token. Real cost will be
higher due to latency, serialization, scheduling, and worker compute. The implementation must batch
all selected experts for a layer into as few network round trips as possible.

## Module Boundaries

Keep the existing dependency direction:

- CLI/env flags in `src/cli.rs`
- orchestration in `src/app/`
- runtime protocol/client/server and execution mechanics in `src/engine/`
- model-family behavior through vendor policies in `src/vendors/`

Suggested module additions:

```text
src/engine/distributed/
  mod.rs
  protocol.rs
  coordinator.rs
  worker.rs
  placement.rs
  transport.rs
```

No model-family branches should be added to generic app or engine flow. Qwen-specific defaults
should be expressed through existing vendor config/policy plumbing, or through generic MoE metadata
already present in `Config`.

## Implementation Checklist

- [x] Capture a repo-specific implementation checklist tied to current module boundaries.
- [x] Add initial CLI/app scaffolding for distributed planning and worker bootstrap modes.
- [x] Create `src/engine/distributed/` as the home for protocol, placement, transport, and worker/coordinator logic.
- [x] Add cluster config domain types and metadata-only placement planning.
- [x] Add a `distributed-worker` bootstrap path that validates cluster role, model metadata, and assigned experts.
- [x] Add GGUF-backed expert inventory helpers for routed MoE layers.
- [x] Extract the routed expert path in `src/engine/runtime/inference.rs` behind a generic executor trait.
- [x] Preserve existing local behavior via a `LocalMoeExpertExecutor`.
- [x] Add a distributed executor implementation that partitions selected experts by host.
- [x] Define the binary distributed protocol with explicit framing, versioning, and request ids.
- [x] Implement fp16/bf16 activation encode/decode helpers for transport.
- [x] Make distributed activation transport dtype configurable (`bf16`, `fp16`, `q8`).
- [x] Add persistent TCP transport and request/response matching.
- [x] Implement the long-running worker server loop.
- [x] Add fail-fast distributed error handling and validation.
- [x] Add partial expert tensor loading by assigned row ranges.
- [x] Add an initial localhost correctness harness for local vs routed expert execution.
- [x] Add distributed profiling counters and benchmark reporting.
- [x] Reduce coordinator expert loading to dense/runtime tensors plus coordinator-assigned expert slices.
- [ ] Update docs and architecture notes as each phase lands.

## Phase 0: Measurements And Baseline

Before distributed execution, collect repeatable single-host numbers.

Use fixed prompt, context, model, and deterministic decoding:

```bash
./target/release/gguf-runner \
  --model Qwen3.5-122B-A10B-Q4.gguf \
  --prompt "Write a concise explanation of Rust ownership." \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --max-tokens 128 \
  --show-timings \
  --profiling
```

Record:

- prefill tokens/sec
- decode tokens/sec
- total wall time
- max RSS
- profile percentages for matmul, attention, MoE, and FFN
- page fault behavior if the run is overcommitted

This is the comparison point for distributed work.

## Phase 1: Expert Inventory And Placement

Add a placement planner that reads model metadata and tensor names, then assigns experts to hosts.

Inputs:

- `n_layers`
- `n_experts`
- `n_experts_used`
- `dim`
- `expert_hidden_dim`
- tensor byte sizes for:
  - `moe_gate_exps`
  - `moe_up_exps`
  - `moe_down_exps`

Output:

- layer/expert -> host assignment
- expected bytes per host
- coordinator-local expert set
- worker expert sets

Initial placement policy:

- spread experts evenly by total tensor bytes
- keep some experts local to the coordinator to reduce network traffic
- avoid requiring any worker to load dense attention or final logits

Current implementation note:

- placement now uses discovered per-node memory/CPU resources to reject impossible assignments
- the coordinator reserves a baseline equal to non-expert checkpoint bytes before expert assignment
- placement keeps a coordinator-local expert window per layer based on `n_experts_used` when capacity allows
- remaining experts are distributed with capacity-weighted worker targets rather than simple least-bytes assignment

Current implementation note:

- the coordinator now performs a lightweight discovery pass to read worker CPU and memory before planning
- placement config is now supplied via CLI flags and env vars rather than a TOML cluster file
- placement summary is printed during distributed coordinator/worker startup instead of a separate plan mode

## Phase 2: Worker Process

Add a worker mode:

```bash
gguf-runner \
  --model Qwen3.5-122B-A10B-Q4.gguf \
  --distributed-worker \
  --distributed-bind-address 192.168.10.11:7000 \
  --distributed-transport-dtype bf16
```

Worker responsibilities:

1. parse GGUF metadata
2. answer coordinator discovery with local CPU/memory
3. load only assigned expert tensors after HELLO assignment
4. validate tensor shapes and quantization types
5. listen for coordinator connections
6. process expert batches
7. report health and loaded tensor summary

The worker should not initialize tokenizer, sampling, KV cache, or CLI decode state.

## Phase 3: Wire Protocol

Use a small binary protocol over persistent TCP connections.

Initial requirements:

- explicit magic/version
- little-endian fixed-width headers
- request ids for matching responses
- shape and dtype fields
- CRC or length validation for frame sanity
- no per-request TCP connection setup

Frame types:

- `HELLO`
- `READY`
- `EXPERT_BATCH_REQUEST`
- `EXPERT_BATCH_RESPONSE`
- `ERROR`
- `SHUTDOWN`

`EXPERT_BATCH_REQUEST` fields:

- request id
- token position
- layer index
- number of experts in batch
- activation dtype: fp16 or bf16
- `dim`
- repeated expert ids
- repeated route weights, optional
- activation payload

The coordinator can send one activation vector plus a list of expert ids for the same layer. This
avoids resending the same hidden vector for every selected expert on the same worker.

`EXPERT_BATCH_RESPONSE` fields:

- request id
- layer index
- number of returned expert outputs
- output dtype
- `dim`
- repeated expert ids
- repeated output payloads

Route weights can be applied on either side. Initial implementation should apply them on the
coordinator so worker outputs are easier to validate against local execution.

## Phase 4: Activation Encoding

Implement local conversion helpers in `src/engine/distributed/protocol.rs`:

- f32 -> fp16 bytes
- fp16 bytes -> f32
- f32 -> bf16 bytes
- bf16 bytes -> f32

Prefer bf16 when worker and coordinator are both CPU-only because conversion is simple and preserves
range. Prefer fp16 only if measured faster or if it gives better compatibility with future GPU paths.

Do not reuse TurboQuant KV encoding in the first version.

## Phase 5: Coordinator Runtime Integration

Add an optional distributed expert executor behind a generic runtime interface:

```rust
trait MoeExpertExecutor {
    fn compute_selected_experts(
        &mut self,
        layer: usize,
        input: &[f32],
        selected: &[(usize, f32)],
        output: &mut [f32],
    ) -> Result<(), String>;
}
```

Implementations:

- `LocalMoeExpertExecutor`: existing in-process path
- `DistributedMoeExpertExecutor`: partitions selected experts by host, sends remote batches, runs
  local experts, then accumulates results in selected-expert order

Integration point:

- replace the routed expert block in `src/engine/runtime/inference.rs` with calls into the generic
  executor
- preserve current math for non-distributed mode
- preserve deterministic accumulation order where feasible

This likely requires adding executor state to `RunState` or passing a runtime execution context into
`transformer(...)`. Keep CLI types out of `engine`.

## Phase 6: Failure Handling

Start with fail-fast semantics:

- transport discovery/connect/send/recv failures retry a small number of times first
- if any worker still disconnects after retries, abort the generation
- if a response has wrong shape/dtype/request id, abort
- if a worker still times out after retries, abort

Later resilience:

- retry idempotent expert requests
- optional fallback to local execution if coordinator has the expert loaded
- worker warm restart

Avoid silent fallback to page-cache overcommit; it will make performance unpredictable.

## Phase 7: Correctness Validation

Add a test mode that runs the same prompt with:

1. all experts local
2. selected experts routed through localhost worker processes

Current status:

- a real localhost expert-equivalence test now exists under `src/engine/distributed/worker.rs`
- it is `#[ignore]` and expects `GGUF_RUNNER_TEST_MOE_MODEL` to point to a routed-MoE GGUF
- it validates one real routed expert path end-to-end through the worker/coordinator protocol and compares the remote result against the local expert executor
- full greedy token parity and logits-level comparison are still pending

Compare:

- selected expert ids per layer/token
- per-layer MoE output max absolute error
- final logits max absolute error
- generated token sequence under greedy decode

Acceptance targets for fp16/bf16 transport:

- per-layer MoE output max absolute error documented and stable
- greedy token sequence matches for short prompts, or differences are explainable and bounded
- no NaN/Inf in returned expert outputs

## Phase 8: Performance Validation

Benchmark configurations:

- single host overcommitted with mmap/page cache
- 4 hosts, 10GbE, fp16 transport
- 4 hosts, 10GbE, bf16 transport
- optional 4 hosts, 1GbE, bf16 transport for contrast

Measure separately:

- model load time
- prefill tokens/sec
- decode tokens/sec
- network MB/token
- remote expert latency per layer
- worker CPU utilization
- coordinator CPU utilization
- worker memory residency

Add debug counters:

- remote expert requests
- remote expert batches
- bytes sent/received
- per-worker average and p95 latency
- local vs remote expert count

Current status:

- `--profiling` now reports distributed MoE aggregate counters for remote batches, remote experts, local experts, transport bytes, and aggregate remote wait time
- the coordinator prints per-worker batch, expert, transport, and average wait summaries on shutdown
- benchmark automation and p95 latency tracking are still pending

## Phase 9: Optimization Passes

Only after correctness and baseline performance:

1. Batch all same-worker experts for one layer into one request.
2. Overlap remote expert requests with local expert computation.
3. Use one persistent worker thread per remote connection.
4. Reuse request/response buffers.
5. Add bf16/fp16 SIMD conversion helpers.
6. Add optional int8 activation transport.
7. Add placement policy that learns frequently selected experts and keeps them local.
8. Add multi-request batching for serving workloads.

## Open Questions

- Does the coordinator fit dense weights, KV cache, and its assigned experts in 32 GB?
- How many experts per layer does the target checkpoint use, and how many are selected per token?
- Are expert tensors stored as large packed tensors or separate per-expert tensors in this GGUF?
- Does routing show locality that allows a hot expert subset to stay local?
- Is decode latency or aggregate throughput the priority?
- Should workers bind to specific performance cores on Intel hybrid CPUs?

## Initial Success Criteria

The distributed implementation is useful if:

- the 76 GB model can run without memory-thrashing on the 4-node cluster
- decode throughput beats single-host page-cache overcommit
- p95 token latency is stable
- greedy output quality is close to local execution
- workers keep assigned expert weights resident after warmup

If 10GbE fp16/bf16 expert offload does not beat overcommitted mmap, do not pursue lower-bit network
formats first. Profile whether the bottleneck is worker compute, coordinator dense layers, or network
round trips.
