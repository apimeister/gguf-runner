# gguf-runner

`gguf-runner` is a small command-line tool that lets you run AI models on your own machine.

The idea behind this project is simple: local AI should feel like a normal Unix-style tool.
You point it to a `.gguf` model file, ask a question, and stream the answer in your terminal.
No cloud API, no GPU setup maze, and no heavy platform around it.

It is built for people who want to:
- run models fully offline
- keep data local
- script prompts in shell workflows
- experiment with different model sizes on regular hardware

Under the hood, `gguf-runner` uses memory mapping (`mmap`) and CPU-only inference.
This means execution is not constrained by GPU availability or fixed GPU memory (VRAM) limits.
In theory, the upper bound shifts toward storage capacity, with the tradeoff that larger working sets become slower.
In practice, performance is often as good as your filesystem caching behavior allows, so warm-cache runs can feel much faster than cold starts.

If you are new to the project, start with the quick steps below and you should get your first response in a few minutes.

## Getting Started

1. Install `gguf-runner` (choose one):

Option A: prebuilt binary from [GitHub Releases](https://github.com/apimeister/gguf-runner/releases)

```bash
tar -xzf gguf-runner-<tag>-linux-amd64.tar.gz
```

Option B: install from source with Cargo

```bash
# default (portable)
cargo install --git https://github.com/apimeister/gguf-runner

# optimized for this machine (recommended)
RUSTFLAGS="-C target-cpu=native" cargo install --git https://github.com/apimeister/gguf-runner
```

On AMD Ryzen 7 PRO 8700GE, `target-cpu=native` improved:
- tok/s: `5.668` -> `6.848` (`+20.8%`)
- runtime: `215.522s` -> `178.041s` (`-17.4%`)

Note: `target-cpu=native` binaries are tuned for the build machine and are less portable across different CPUs.

2. Verify installation and CPU feature detection:

```bash
gguf-runner --show-features
```

If you used a release archive and did not move the binary into your `PATH`, run:

```bash
./gguf-runner --show-features
```

3. Download `Qwen3.5-0.8B`:

```bash
wget https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf
```

4. Run a first text prompt:

```bash
gguf-runner \
  --model ./Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "hello"
```

5. (Optional) Run a vision prompt with Qwen3.5:

```bash
wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf
wget -O mmproj-Qwen3.5-2B-F16.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf
gguf-runner \
  --model ./Qwen3.5-2B-Q4_K_M.gguf \
  --image sample-image.jpg \
  --prompt "Describe that image."
```

More model download examples:
- `docs/downloading-models.md`
- Distributed cluster usage:
  - `docs/distributed-cluster-usage.md`

## Working Models

Known-good status from `docs/performance.md` (text benchmarks) and local model/mmproj availability.

| Model | Text | Vision |
|---|---|---|
| `gemma-3-4b-it-Q4_K_M.gguf` | ✅ | ✅ |
| `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-0.6B-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf` | ✅ | ❌ |
| `Qwen3-Coder-Next-Q4_K_M.gguf` | ✅ | ❌ |
| `Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | ⚪ | ✅ |
| `Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | ⚪ | ✅ |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | ✅ | ✅ |
| `Qwen3.5-2B-Q4_K_M.gguf` | ✅ | ✅ |
| `Qwen3.5-35B-A3B-UD-Q4_K_M.gguf` | ✅ | ✅ |

## Distributed MoE Cluster Mode

`gguf-runner` now includes an experimental distributed routed-MoE mode for multi-host execution.

- `--distributed-plan` prints a metadata-based expert placement summary for a `cluster.toml`
- `--distributed-worker` starts a worker process bound to one cluster node id
- normal generation with `--cluster <cluster.toml>` runs the coordinator path

Start here:

- [docs/distributed-cluster-usage.md](./docs/distributed-cluster-usage.md)
- [docs/distributed-moe-plan.md](./docs/distributed-moe-plan.md)

## What You Need

- A local `.gguf` model file.
- Enough RAM for the model you choose.
- Rust toolchain (only if you build from source).

## Operation Modes

`gguf-runner` has two distinct operation modes selected with `--mode`:

### Oneshot mode (default)

Oneshot mode runs a single prompt and exits. The model loads, generates a response, prints it to stdout, then terminates. This is the default when `--mode` is not specified.

```bash
gguf-runner \
  --model ./Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "What is the capital of France?"
```

**When to use oneshot:**
- Scripting and automation — pipe the output to other tools
- One-off questions where you do not need a follow-up
- CI pipelines, cron jobs, or any non-interactive context

**Key characteristics:**
- `--prompt` is required
- Tools are **disabled** by default (pass `--allowed-tools all` to enable)
- No persistent chat history — each invocation starts fresh
- Output goes to stdout, making it easy to capture or pipe

**Scripting examples:**

```bash
# Capture output to a variable
SUMMARY=$(gguf-runner --model model.gguf --prompt "Summarize: $(cat notes.txt)")

# Pipe into another command
gguf-runner --model model.gguf --prompt "List five ideas for a project name" | fzf

# Use in a shell script
for file in *.md; do
  gguf-runner --model model.gguf --prompt "Summarize this: $(cat $file)" > "${file%.md}.summary"
done
```

### REPL mode

REPL mode starts an interactive terminal session. The model loads once and stays in memory. You type prompts and get responses in a continuous loop, with full chat history carried across turns.

```bash
gguf-runner \
  --model ./Qwen3.5-0.8B-Q4_K_M.gguf \
  --mode repl
```

**When to use REPL:**
- Multi-turn conversations where context matters
- Exploratory sessions — ask follow-up questions
- Agentic work with file and shell tools
- Any time you want the model to remember what you said earlier in the session

**Key characteristics:**
- Tools are **enabled** by default (pass `--allowed-tools none` to disable)
- Chat history accumulates within the session
- The model loads once — subsequent prompts pay no load cost
- A status bar shows token count, speed, and context usage

**Slash commands** (type `/help` inside the REPL for the current list):

| Command | Effect |
|---|---|
| `/help` | Show available commands |
| `/model` | Print the active model path |
| `/image <path>` | Attach an image to the next prompt (vision models) |
| `/images` | List currently attached images |
| `/clear-images` | Remove all image attachments |
| `/clear` | Reset chat history and image attachments |
| `/exit` or `/quit` | Exit the REPL |

Tab completion works for slash commands: type `/e` and press Tab to expand to `/exit`.

Use `Ctrl+C` or `Esc` to exit at any time.

**Starting with an initial prompt:**

You can pass `--prompt` alongside `--mode repl` to send a first message automatically once the model is ready:

```bash
gguf-runner \
  --model ./Qwen3.5-0.8B-Q4_K_M.gguf \
  --mode repl \
  --prompt "Hello, let's start by you telling me your name."
```

## Tools and Agent Capabilities

Both modes support an agent layer that lets the model read files, list directories, write files, and run approved shell commands. Tools are off by default in oneshot and on by default in REPL.

### Enabling and restricting tools

```bash
# Enable all tools (oneshot — disabled by default)
gguf-runner --model model.gguf --prompt "..." --allowed-tools all

# Enable only specific tools
gguf-runner --model model.gguf --mode repl --allowed-tools read_file,list_dir

# Disable all tools (REPL — enabled by default)
gguf-runner --model model.gguf --mode repl --allowed-tools none
```

Available tool names: `read_file`, `list_dir`, `write_file`, `mkdir`, `rmdir`, `shell_list_allowed`, `shell_exec`, `shell_request_allowed`

### Restricting file access

Use `--tool-root` to confine file operations to a specific directory. Without it, the current working directory is used as the root.

```bash
gguf-runner \
  --model model.gguf \
  --mode repl \
  --tool-root ./my-project
```

### Allowing shell commands

Shell execution is sandboxed: the model can only run commands you explicitly allowlist.

```bash
# Allow specific commands on the command line
gguf-runner --model model.gguf --mode repl \
  --allow-shell-command cargo \
  --allow-shell-command git

# Or via environment variable (comma-separated)
GGUF_ALLOW_SHELL_COMMANDS=cargo,git gguf-runner --model model.gguf --mode repl
```

### Config file

Persistent tool and shell settings can be stored in a TOML config file. `gguf-runner` checks two locations in order, with the project-local file taking precedence:

1. `~/.gguf-runner/config.toml` (user-wide defaults)
2. `./.gguf-runner/config.toml` (per-project overrides)

```toml
# .gguf-runner/config.toml

[tools]
# Disable individual tools if needed
write_file = false
rmdir = false

[shell]
# Allowlist shell commands with optional descriptions
# Descriptions help the model understand when to use each command
[shell.md]
cargo = "Rust build and test tool"
git = "Version control"
rg = "Fast grep (ripgrep)"
```

### How tool routing works in REPL

In REPL mode with tools enabled, `gguf-runner` inspects each prompt before sending it to the model. Plain conversational questions go directly to a fast chat path. Prompts that clearly need file or shell access (mentioning files, directories, `cargo`, `git`, etc.) go through the full agent loop. This means you can mix plain chat and tool-assisted requests in the same session without configuration changes.

## Basic Command Pattern

```bash
gguf-runner \
  --model ./your-model.gguf \
  --prompt "Your question"
```

Most common options (and what they do):
- `--mode oneshot|repl`: `oneshot` runs one request and exits. `repl` keeps an interactive prompt loop until you type `/exit` or `/quit`.
- `--allowed-tools <list>`: Comma-separated tool allowlist, or `all` / `none` (`none` disables all tools).
  - defaults: `oneshot => none`, `repl => all`
- `--max-tokens 256`: Maximum number of generated output tokens. Use lower values for short answers and faster test runs.
- `--context-size 4096`: Sets how much conversation/history the model can keep in context.
- `--temperature 0.7`: Controls randomness. Lower is more deterministic, higher is more creative/variable.
- `--threads 8`: Number of CPU threads to use. Usually set this near your available CPU cores.
- `--think yes|no|hidden`: Controls thinking output for reasoning models (Qwen3, Qwen3.5, etc.).
  - `yes` — show the model's thinking steps (default for oneshot)
  - `hidden` — suppress thinking, show only the final answer (default for REPL)
  - `no` — skip the thinking phase entirely for faster, shorter responses
- `--show-features`: Prints detected CPU features (compiled vs runtime) and exits.
- `--show-tokens`: Streams token-level output/diagnostics while generating.
- `--show-timings`: Prints timing breakdowns so you can inspect performance bottlenecks.
- `--profiling`: Enables deeper profiling output for performance analysis.
- `--debug`: Enables additional debug logging/details during execution.

## Vision Example (Image Input)

For vision-capable models (for example Qwen3-VL / Qwen3.5 multimodal variants):

```bash
gguf-runner \
  --model ./Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
  --image ./regression/IMG_0138.jpg \
  --prompt "Describe this image."
```

In REPL mode, attach images with the `/image` command before your prompt:

```
[you] > /image ./screenshot.png
[you] > What does this error message say?
```

If required multimodal tensors/components are missing, the runner fails fast with a clear error.

## Project Scope

- CPU inference only
- GGUF model files only
- Focus on clear, readable implementation

## Useful Docs

- Feature coverage: `docs/features.md`
- Performance history: `docs/performance.md`
- Performance ideas/tuning notes: `docs/performance-improvement-suggestions.md`
- Module layout: `docs/module-structure.md`

## GGUF Metadata Dump (No Inference)

```bash
cargo run --example gguf_dump -- --model ./model.gguf --dump-kv --dump-tensors
```

Suggestions and PRs are welcome.
