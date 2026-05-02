# Distributed Cluster Usage

This is the shortest path to run distributed routed-MoE mode.

## Prerequisites

- use the same `gguf-runner` build on all hosts
- put the same routed-MoE `.gguf` on coordinator and workers
- choose one coordinator address and one address per worker

## Coordinator Flags

Only the coordinator needs the worker list:

```text
--distributed-transport-dtype bf16
--distributed-worker-node '192.168.10.11:7000'
--distributed-worker-node '192.168.10.12:7000'
--distributed-worker-node '192.168.10.13:7000'
```

`--distributed-coordinator-address` is optional and only affects coordinator node metadata in the placement plan.

## Start Workers

Run one worker per worker host.

On `worker-a`:

```bash
gguf-runner \
  --model ./Qwen3.5-122B-A10B-Q4.gguf \
  --distributed-worker \
  --distributed-bind-address 192.168.10.11:7000 \
  --distributed-transport-dtype bf16
```

Repeat on `worker-b`, `worker-c`, changing only `--distributed-bind-address`.

## Start Coordinator

After all workers are listening, run normal generation on the coordinator:

```bash
gguf-runner \
  --model ./Qwen3.5-122B-A10B-Q4.gguf \
  --distributed-transport-dtype bf16 \
  --distributed-worker-node '192.168.10.11:7000' \
  --distributed-worker-node '192.168.10.12:7000' \
  --distributed-worker-node '192.168.10.13:7000' \
  --prompt "Write a concise explanation of Rust ownership." \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --max-tokens 128
```

## What Happens

- workers listen, answer discovery, then receive expert assignment during `HELLO`
- coordinator discovers worker CPU/memory, builds placement, prints the plan, and runs inference

## Useful Notes

- `--distributed-transport-dtype` supports `bf16`, `fp16`, and `q8`
- `q8` uses lossy int8 transport; validate it on your model before broad use
- transport failures are retried a few times; protocol/model mismatches still fail immediately
- there is no separate plan mode

## Troubleshooting

- worker won’t start: check `--distributed-bind-address` and that the port is free on that host
- coordinator won’t start: make sure workers are already listening and reachable
- startup mismatch: verify all nodes use the same model file and node declarations
