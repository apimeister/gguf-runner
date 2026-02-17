# Module Structure Reference

This project is currently a binary crate (`src/main.rs`) with internal modules (`mod ...`), not a separate `lib.rs`.

## Maintenance Rule

- When architecture-related code changes, update this file in the same change.
- Keep this file as the current-state snapshot.

## Top-Level Layout

```text
src/
  main.rs
  app/
    mod.rs
    generation.rs
    agent.rs
    tools.rs
  cli.rs
  engine/
    mod.rs
    types.rs
    io/
      mod.rs
      gguf.rs
    tokenizer/
      mod.rs
    weights.rs
    kernels/
      mod.rs
      math.rs
      quant.rs
      sampling.rs
    runtime/
      mod.rs
      inference.rs
      parallel.rs
    switches.rs
    profiling.rs
  vendors/
    mod.rs
    llama.rs
    gemma.rs
    qwen.rs
    deepseek.rs
```

## Module Responsibilities

### `src/main.rs`

- Binary entrypoint (`fn main()`) and crate root wiring.
- Delegates runtime orchestration to `app::run()`.

### `src/app/mod.rs`

- Application orchestration entrypoint (`run()`).
- Executes end-to-end run pipeline:
  - parse CLI options
  - map CLI tuning flags into `engine::switches::RuntimeSwitchConfig`
  - initialize runtime switches via `engine::switches::init_runtime_config(...)`
  - initialize profiling
  - load runtime/model context via `app::generation`
  - route to standard generation mode or tool-agent mode
  - print profiling/timing summaries

### `src/app/generation.rs`

- Model/runtime bootstrap for inference:
  - GGUF parse/load
  - vendor config + tokenizer + weights initialization
  - context/thread overrides
- Token generation loop implementation.
- Exposes a reusable `generate_text(...)` API used by both normal and agent flows.

### `src/app/agent.rs`

- Tool-agent orchestration loop for multi-step runs.
- Builds turn prompt transcript, requests one model response per turn, and parses JSON outputs.
- Builds system prompt with tool catalog metadata (description / when-to-use) and optional allowed-shell-command descriptions supplied by `cli`.
- Executes tool calls through `app::tools::ToolExecutor` and appends tool results back into transcript.
- Terminates on `final` response or configured tool-call limit.

### `src/app/tools.rs`

- Host-side tool execution for agent mode.
- Provides safe file tools:
  - `read_file`
  - `write_file`
  - `list_dir`
  - `mkdir` (recursive create)
  - `rmdir` (recursive delete; never removes `tool_root`)
  - `shell_list_allowed`
- Provides restricted external command tools:
  - `shell_exec` (only for allowed command names)
  - `shell_request_allowed` (structured request to operator)
- Enforces tool root path constraints and per-call payload limits.

### `src/cli.rs`

- All clap parsing lives here.
- Public parser result type: `CliOptions`.
- Parses user-facing flags plus hidden tuning/debug options.
- Env var integration is here (via clap `env = ...`), currently `GGUF_*` variables.
- Loads optional layered TOML config for agent shell allowed commands:
  - `~/.gguf-runner/config.toml`
  - `./.gguf-runner/config.toml` (overrides home config)
- Supports single-source command metadata in `[shell.cmd]` (key=command, value=description); `[shell.md]` and older formats remain accepted for compatibility.
- Supports optional tool config in `[tools]`:
  - internal tool toggles (all default enabled)
- Includes agent-related switches:
  - `--agent`
  - `--tool-root`
  - `--allow-shell-command`
  - `--max-tool-calls`

### `src/engine/mod.rs`

- Aggregates engine submodules:
  - `io`, `kernels`, `profiling`, `runtime`, `switches`, `tokenizer`, `types`, `weights`.

### `src/engine/types.rs`

- Core data model and shared constants.
- Defines:
  - GGUF constants and ggml quantization constants
  - Core structs like `Config`, `GGUFFile`, `TransformerWeights`, `RunState`, `Tokenizer`, `QuantizedTensor`
  - Lazy loader implementation (`LazyModelLoader`)
  - Unix mmap wrapper (`MappedFile`) including Linux memory advice hints for model mappings
  - Global lazy loader state:
    - `LAZY_MODEL_LOADER: OnceLock<Arc<LazyModelLoader>>`
  - `ensure_model_range(...)` helper used by quantized matmul paths.

### `src/engine/io/*`

- GGUF parsing and low-level read helpers.
- `io/gguf.rs`:
  - Parses GGUF metadata/tensors.
  - Maps model file.
  - Provides metadata access helpers:
    - `get_gguf_int_from_map`, `get_gguf_float_from_map`, `get_gguf_string_from_map`, `find_gguf_tensor`.

### `src/engine/tokenizer/mod.rs`

- Tokenizer initialization and encode/decode logic.
- Handles sentencepiece/tiktoken-ish paths and special token resolution.
- Exposes `init_tokenizer_from_gguf(...)`.

### `src/engine/weights.rs`

- Loads and validates model tensors from GGUF into `TransformerWeights`.
- Handles per-family tensor layout differences and optional tensors, including DeepSeek-MLA tensors and DeepSeek MoE expert-bias tensors (`exp_probs_b.bias`).
- Exposes `init_weights_from_gguf(...)`.

### `src/engine/kernels/*`

- Numerical and sampling kernels used by inference.
- `math.rs`: normalization, softmax, vector math, Qwen3Next SSM linear attention helpers.
- `quant.rs`: quantized dequant/dot/matmul paths, architecture-specific fast paths (including x86 AVX-VNNI/AVX512-VNNI Q8 paths), MR4 validation.
- `sampling.rs`: token selection helpers (`argmax`, multinomial sample, top-k/top-p sampler).

### `src/engine/runtime/*`

- Runtime-specific execution and threading config.
- `runtime/inference.rs`:
  - `malloc_run_state(...)`
  - `transformer(...)`
  - includes dedicated DeepSeek2 MLA attention path (`attn_q_a/q_b`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`) using absorbed queries + compressed MLA K/V cache, plus mixed dense/MoE FFN handling
  - DeepSeek2 MLA query layout follows model-native split order (`q_rope` then `q_nope`) for RoPE and absorb-projection paths
  - DeepSeek2 YaRN/rope scaling applies extrapolation scaling only beyond original context length (`rope_orig_ctx`)
  - DeepSeek2 MoE routing supports model-declared gating mode (including sigmoid routing) and per-layer expert routing bias application
  - quantized KV cache storage for attention state:
    - default Q8 cache
    - automatic Q4 fallback when Q8 allocation fails
- `runtime/parallel.rs`:
  - `configure_rayon_threads(...)`
- `runtime/mod.rs`:
  - Re-exports runtime helpers.
  - `apply_context_size_overrides(...)`.

### `src/engine/switches.rs`

- Runtime tuning and feature switches.
- Keeps `OnceLock` / atomic-backed switch state as system-of-record.
- Includes:
  - `RuntimeSwitchConfig` (engine-owned overrides struct)
  - Parallel thresholds (`par_matmul_min_rows`, `par_matmul_chunk_rows`, `par_attn_min_heads`, `par_qwen3next_min_heads`)
  - KV cache selection switch (`kv_cache_mode`: `auto` / `q8` / `q4`)
  - Arch feature toggles (`use_x86_*`, `use_aarch64_*`, including x86 AVX2/F16C/QK-MR4/AVX-VNNI/AVX512VNNI-Q8 switches)
  - Layer debug toggles
  - MR4 status atomics
  - `init_runtime_config(&RuntimeSwitchConfig)`.

### `src/engine/profiling.rs`

- Profiling counters and helper functions.
- Contains all profiling atomics and report formatting:
  - `set_profiling_enabled`, `prof_start`, `prof_end`, `profiling_reset`, `record_forward_pass`, `print_profile_report`.

### `src/vendors/*`

- Vendor/model-family specific config parsing and prompt templating.
- `vendors/mod.rs`:
  - Detects model family from GGUF metadata.
  - Builds `Config` from family-specific key conventions (including DeepSeek2 MLA metadata, DeepSeek expert gating metadata, and RoPE scaling metadata).
  - Routes chat prompt encoding to family-specific implementation.
- `vendors/llama.rs`, `vendors/gemma.rs`, `vendors/qwen.rs`, `vendors/deepseek.rs`:
  - Family-specific defaults, validations, and prompt rendering (including Qwen MoE routing defaults/scaling).

## Runtime Data Flow

1. `main.rs` invokes `app::run()`.
2. `app::run()` parses CLI (`CliOptions`).
3. `app::run()` builds `RuntimeSwitchConfig` and calls `engine::switches::init_runtime_config(...)`.
4. GGUF parsed via `engine::io::parse_gguf_file(...)`.
5. Vendor config built with `vendors::build_config_from_gguf(...)`.
6. Tokenizer initialized (`engine::tokenizer::init_tokenizer_from_gguf(...)`).
7. Runtime overrides applied (`engine::runtime::apply_context_size_overrides(...)`).
8. Weights loaded (`engine::weights::init_weights_from_gguf(...)`).
9. Standard mode:
  - prompt encoded once and token loop executes forward passes (`engine::runtime::transformer(...)`) + sampling (`engine::kernels`).
10. Agent mode:
  - tool transcript prompt encoded per turn
  - model emits JSON `tool_call` / `final`
  - host executes tool call (`app::tools`) and loops until final response or limit.
  - `shell_exec` calls are restricted to CLI/env-provided allowed commands; model can request missing commands with `shell_request_allowed`.
11. Profiling/timings printed from `engine::profiling` + `app::run()`.

## Placement Rules For Future Changes

- New CLI flags or env vars: `src/cli.rs`.
- End-to-end run orchestration: `src/app/mod.rs`.
- New runtime tuning switches or arch toggles: `src/engine/switches.rs`.
- New profiling counters/reporting: `src/engine/profiling.rs`.
- New math/quant/sampling primitive: `src/engine/kernels/*`.
- New model-family metadata or prompt format: `src/vendors/<family>.rs` + dispatch in `src/vendors/mod.rs`.
- New GGUF parsing logic: `src/engine/io/gguf.rs`.
- New tensor loading logic: `src/engine/weights.rs`.
- Keep `src/main.rs` focused on entrypoint + crate wiring.

## Validation Workflow

- Run these checks for refactor validation:
  - `cargo check`
  - `rg -n "^use crate::\\*;" src/engine -g'*.rs'`
  - `rg -n "^use crate::engine::[A-Za-z0-9_:]+::\\*;" src/engine -g'*.rs'`
  - `rg -n "crate::cli::" src/engine -g'*.rs'`
- Expected result for all `rg` checks above: no matches.

## Known Coupling (Current State)

- `engine` no longer depends on `cli` types directly; runtime switch wiring happens in `app`.
- `main.rs` no longer re-exports `engine::types::*`; `vendors` and `engine` import from `engine::*` modules directly.
- `src/engine/*` no longer uses wildcard imports for internal crate modules.
- Remaining wildcard imports in `engine` are architecture-intrinsics (`std::arch::*`) for SIMD code paths.
