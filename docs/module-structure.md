# Module Structure Reference

This project provides a binary entrypoint (`src/main.rs`) plus a small library facade (`src/lib.rs`)
for `EmbeddedRuntime`; both share the same internal app, vendor, and engine modules.

## Maintenance Rule

- When architecture-related code changes, update this file in the same change.
- Keep this file as the current-state snapshot.

## Top-Level Layout

```text
src/
  main.rs
  lib.rs
  app/
    mod.rs
    audio_batch.rs
    embed.rs
    events.rs
    generation.rs
    agent.rs
    repl.rs
  cli.rs
  rag/
    mod.rs
    chunker.rs
    encoder.rs
    index_io.rs
  tools.rs
  engine/
    mod.rs
    types.rs
    audio/
      mod.rs
      decode.rs
      features.rs
    io/
      mod.rs
      gguf.rs
    multimodal/
      mod.rs
      gemma3.rs
      injection.rs
      qwen3_asr.rs
      qwen3vl.rs
    vision/
      mod.rs
      preprocess.rs
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
    qwen2.rs
    qwen3.rs
    qwen35.rs
    qwen3vl.rs
    qwen3next.rs
    qwen3_asr.rs
    qwen_common.rs
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
  - validate non-empty media file paths for standard generation mode:
    - `--image` (`png/jpg/jpeg/webp`)
    - `--video` (`mp4`)
    - one `--audio` path for offline Qwen3-ASR transcription (extension-agnostic validation)
    - optional `--audio-language` forced-language policy input
  - route by operation mode:
    - `oneshot`: single prompt execution
    - `repl`: interactive question/answer loop with slash commands (`/help`, `/model`, `/exit`, `/quit`)
  - routes plain text-only non-agent turns through `generate_text(...)`; uses structured `GenerationRequest` execution only when media inputs are present
  - routes one-shot audio-only requests through `transcribe_audio(...)`, treating `--prompt` as
    context/hotword text and keeping transcript-only stdout separate from stderr diagnostics
  - validates a complete `--audio-batch` manifest before model load, retains one runtime for the
    serial job, and routes per-record file checks/transcription through `app::audio_batch`
  - tool-agent support is orthogonal to operation mode:
    - default `allowed-tools=none` in `oneshot`
    - default `allowed-tools=all` in `repl`
    - when tools are enabled, the runner chooses between standard generation and `app::agent`
      based on prompt/tool eligibility rather than forcing every REPL turn through the agent planner
    - when tools are disabled, requests are handled via standard generation
  - print profiling/timing summaries

### `src/app/audio_batch.rs`

- Owns the offline JSONL batch schema and serial job loop.
- Performs a constant-space validation pass before model load and rewinds the manifest for
  processing instead of retaining all records in memory.
- Resolves relative paths from the manifest directory, preserves input order, writes one structured
  success/error record per item, and flushes every result.
- Keeps batch orchestration out of `engine`; each item uses the normal transcription request path
  and therefore receives fresh inference state while reusing loaded model and sidecar weights.

### `src/app/repl.rs`

- `crossterm` footer-style terminal UI for REPL mode.
- Owns:
  - main-screen terminal lifecycle with a fixed footer area
  - scroll-region management so model/debug output prints into the terminal's native scrollback above the footer
  - raw-mode key handling
  - visible input buffer editing
  - slash-command tab-completion at edit time
  - slash-command managed REPL state, including persistent image attachments (`/image`, `/images`, `/clear-images`, `/clear`)
  - prompt history navigation
  - rolling non-agent chat history for continuous multi-turn conversations
  - status line updates during turn execution:
    - standard generation progress uses prefill/decode counters, throughput, and a context gauge
    - agent/tool turns consume typed `RunnerStatus` values and render them as explicit phase labels such as planning, tool execution, finalizing, and recovery
  - a single background worker thread that owns the only `ModelRuntime` instance in `repl` mode
  - command/event channels between UI thread and runtime worker
  - streamed assistant output plus typed runner logs printed directly into terminal scrollback
- Uses native multimodal generation for turns with active image attachments, rather than routing image access through tool-agent calls.
- When tools are enabled, only tool-likely prompts are routed into the agent runner; ordinary chat remains on the standard text-generation path.
- Plain text REPL chat runs with `think=no` to avoid exposing raw chain-of-thought or losing the visible answer on reasoning-model chat templates; oneshot behavior remains unchanged.
- Active REPL media attachments can initialize external multimodal support lazily on first use, including local `mmproj` sidecar discovery/loading for sidecar-backed vision families. Audio-only sidecars can be recognized by the shared lazy probe, but REPL audio attachments and audio encoder execution remain unavailable.
- Loads the runtime inside the worker thread so the REPL owns runtime lifecycle end-to-end and enforces one active runtime.

### `src/app/events.rs`

- Shared app-level runtime event types for streamed REPL output.
- Defines the event callback used by:
  - `app::generation` for visible token chunks and debug lines
  - `app::agent` for typed runner logs, final output, and explicit footer status updates during agent turns
- Keeps event transport in `app/` so `engine/` stays independent of UI/orchestration concerns.
- The runner/UI boundary is typed:
  - `RuntimeEvent::Output(String)` for visible assistant text
  - `RuntimeEvent::Log(RuntimeLog)` for debug/system/error messages
  - `RuntimeEvent::Status(RunnerStatus)` for orchestration state such as planning, tool execution, finalizing, and recovery
  - `RuntimeEvent::Progress(RuntimeProgress)` for numeric decode/prefill progress

### `src/app/generation.rs`

- Model/runtime bootstrap for inference:
  - GGUF parse/load
  - llama-style local `mmproj*.gguf` sidecar discovery/probe for multimodal models (no extra CLI switch)
  - sidecar probe enforces checkpoint variant token matching (for example `2b`, `35b`, `a3b`) to prevent silent cross-size pairing
  - metadata-first Qwen3-ASR audio-only sidecar recognition records the validated `qwen3_asr` backend independently from vision capability
  - eager and lazy loading construct `VisionEncoder` only for image/video requests when the selected sidecar exposes both vision encoder and projector tensors
  - eager and lazy audio requests construct a dedicated Qwen3-ASR `AudioEncoder`; diagnostics
    report loaded weights and effective execution readiness
  - embedded mmproj loading supports either the existing vision encoders or the Qwen3-ASR audio weight contract
  - applies vendor multimodal/runtime debug policies to sidecar scoring, request shaping, and context-length debug logging
  - vendor config + tokenizer + weights initialization
  - multimodal weight-group probe/initialization for multimodal backends
  - context/thread overrides
- Token generation loop implementation.
- Exposes reusable generation APIs:
  - `generate_text(...)` for text-only prompts
  - `generate_chat_messages_for_repl(...)` for multi-turn REPL chat encoded through vendor-native chat templates
  - `generate_text_with_images(...)` (image path routes through structured request execution)
    - qwen35 image route appends a non-hallucination guard instruction for unreadable text regions
  - `generate_text_for_agent(...)` for strict agent JSON turns; uses vendor decode policy generically for deterministic agent-mode settings instead of family branches in app logic
  - `generate_request(...)` for structured multimodal requests (`GenerationRequest`) with:
    - structured prompt encoding via `vendors::encode_generation_request(...)`
    - placeholder-span/media alignment checks for image/video/audio inputs
    - fail-fast native capability checks when required multimodal tensors/components are missing
      - capability-probe details (`image/video/audio`) are included in diagnostics, including sidecar search/probe status
    - native image/video/audio preprocessing execution prior to embedding path
      - images: decode + resize/crop + normalize + CHW tensor conversion
      - for external mmproj vision backends, image resize target is taken from encoder metadata (for example `clip.vision.image_size`) instead of a fixed 224x224 fallback
      - qwen35 backend uses fit-within preprocessing (aspect-ratio preserving, patch-aligned) at a balanced intermediate scale to retain full-frame text regions while limiting visual-token overload
      - gemma3 backend uses fixed-size stretch resize to encoder image_size (llama.cpp-compatible SigLIP preprocess semantics)
      - videos: currently unavailable in no-external-dependency mode
      - audio: bounded RIFF/WAVE integer PCM and IEEE float decode, deterministic channel averaging and
        miniaudio-compatible LPF4 linear resampling, followed by model-policy-driven log-Mel window
        construction; header/shape probing predicts feature-window and embedding counts before the
        expensive path
    - native image embedding execution via in-engine multimodal backend (`qwen3vl`/`qwen35` mmproj sidecar path)
      - Gemma3 sidecar path uses `<start_of_image>`/`<end_of_image>` prompt markers and SigLIP-style projector pooling
    - think-tag decode safeguards:
      - hidden-think mode enforces vendor-bounded think/total token caps
      - visible-think mode reserves answer budget after `</think>` and enforces a vendor-bounded visible think phase
      - multimodal turns can be forced to hidden-think by vendor decode policy without changing the global CLI think setting
    - structured-output decode safeguards:
      - generic structured-output mode selector in generation settings
      - current `agent-json` mode seeds a compact JSON prefix for tool-agent turns
      - lexical/schema-aware token masking for `tool_call` / `final` responses
      - stop after first complete top-level JSON object
    - ordered image/audio language-space embedding injection with a context preflight that reserves up to 256 requested decode tokens for image requests, the full 512-token vendor transcription budget for audio requests, and never truncates media embeddings
    - debug-mode audio encoder progress at most ten times per file, since one window covers eight seconds of input
    - explicit current-state errors for unimplemented native video execution
  - `transcribe_audio(...)` orchestration builds the vendor-owned audio-only request,
    disables RAG/reasoning/structured output, applies greedy transcription decode settings, restores
    caller settings on success or failure, applies the vendor-required Q8 KV-cache override, and
    returns a typed parsed result
  - shared decode core `generate_from_prefill(...)` for text + multimodal routes (supports per-position embedding overrides during prefill)
    - prompts of more than eight tokens prefill in chunks through `transformer_prefill_batch(...)`,
      including media prompts: injected embeddings are passed as `PrefillInput::Embedding`. The
      batch gate falls back to the sequential loop only when the model is unsupported or when
      injected embeddings are wider than `dim` (deepstack). The final prompt token always stays on
      the sequential path so the logits and sampling flow are untouched
    - `generated_tokens` counts only positions past the prompt. Both bookkeeping checks after the
      `pos`/`token` advance test `pos >= prompt_tokens.len()`, not `len() - 1`; the off-by-one
      counted the last prompt token as generated, which shifted every periodic loop-guard check
      (`% 4`, `% 16`) and the decode-budget accounting, and made the two prefill paths disagree
- Supports optional app-level runtime event callbacks:
  - streamed visible output chunks for TUI REPL
  - debug-line emission without writing directly to terminal during REPL turns
- For REPL multi-turn chat, trims oldest encoded turns when chat history outgrows the model context window, preserving the newest exchange.

### `src/rag/mod.rs`

- RAG module entrypoint and shared index types.
- Owns:
  - index build orchestration from chunked source documents
  - exact duplicate-chunk elimination ahead of tokenisation/embedding
  - tokenizer priming plus parallel chunk tokenisation ahead of embedding
  - dynamically scheduled chunk embedding/progress reporting
  - in-memory embedding matrix assembly for retrieval
  - cosine-similarity retrieval plus keyword rescue
  - prompt-context injection helpers
- Keeps RAG-specific indexing and retrieval flow outside `app/` and `engine/`, while consuming generic engine/runtime primitives and the sidecar encoder exposed by `src/rag/encoder.rs`.

### `src/rag/chunker.rs`

- Source-to-chunk preprocessing for RAG indexing.
- Walks document trees, splits markdown by heading/paragraph structure, and applies lightweight language-aware chunking for supported code files.

### `src/rag/encoder.rs`

- Embedding sidecar GGUF loader and document/query embedding runtime.
- Owns tokenizer use for the sidecar model, pooling policy, reusable BERT prefill scratch buffers, fused-QKV staging, head-major attention staging, cached RoPE tables, and the embedding fast path used during index builds and retrieval.

### `src/rag/index_io.rs`

- Binary `.ragidx` serialization/deserialization for persisted RAG indexes.
- Encodes chunk metadata and embedding vectors for save/load without coupling persistence details into the retrieval path.

### `src/app/agent.rs`

- Engine-owned agent runner/tool loop:
  - one agent turn = one structured model decision (`tool_call` or `final`)
  - agent JSON is used only for control flow; final user-facing prose is generated in a separate plain-text step so long answers are not forced into JSON strings
  - retries malformed agent JSON up to a vendor-policy-defined budget
  - can fall back to plain chat for weak models after repeated malformed non-tool turns, using clean chat history instead of the internal protocol-repair transcript
- Internal runner roles are split explicitly:
  - `AgentPlanner` handles structured control decisions
  - `ToolRunner` executes host tools and produces transcript entries
  - `FinalAnswerGenerator` produces the final plain-text answer after planning completes
- Emits typed runner logs/status into `app::events` so UI code does not infer runner state from raw model text or free-form status strings.

- Tool-agent orchestration loop for multi-step runs.
- Entrypoint accepts per-turn prompt text, so the same agent loop can be used from both `oneshot` and `repl` modes.
- Builds turn prompt transcript, requests one model response per turn, and parses JSON outputs.
- Agent replies are expected as compact single-object JSON with fixed key order; runtime constrains decode to the two supported schemas.
- Builds system prompt with tool catalog metadata (description / when-to-use) and optional allowed-shell-command descriptions supplied by `cli`.
- Executes tool calls through `tools::ToolExecutor` and appends tool results back into transcript.
- Exposes both:
  - direct stdout/stderr execution for `oneshot`
  - collected event output for TUI-driven `repl`
  - immediate app-level runtime event emission during tool-agent turns when a callback is installed
- Terminates on `final` response or configured tool-call limit.

### `src/tools.rs`

- Host-side tool execution + canonical tool-name catalog.
- Defines stable tool-name constants and `ALL_TOOL_NAMES` used by both `cli` and tool dispatch.
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
- Includes operation/tool-mode switches:
  - `--mode oneshot|repl`
  - `--allowed-tools <list>` (comma-separated tool names, or `all` / `none`, validated against `tools::ALL_TOOL_NAMES`)
  - hidden compatibility alias `--agent` (maps to tools-enabled behavior)
- Includes agent/tool related switches:
  - `--tool-root`
  - `--allow-shell-command`
  - `--max-tool-calls`
- Includes multimodal switch:
  - repeatable `--image <path>` for image inputs (standard generation mode)
  - repeatable `--video <path>` for video inputs (standard generation mode)
  - `--audio <path>` for one-file offline transcription
  - `--audio-language <language>` for optional forced-language transcription
  - `--audio-batch <manifest.jsonl>` for serial offline jobs with structured JSONL output

### `src/engine/mod.rs`

- Aggregates engine submodules:
  - `audio`, `io`, `kernels`, `multimodal`, `profiling`, `runtime`, `switches`, `tokenizer`,
    `types`, `vision`, `weights`.

### `src/engine/types.rs`

- Core data model and shared constants.
- Defines:
  - GGUF constants and ggml quantization constants
  - Core structs like `Config`, `GGUFFile`, `TransformerWeights`, `RunState`, `Tokenizer`, `QuantizedTensor`
  - Qwen3-VL input embedding shaping metadata on `Config`:
    - `input_embedding_dim` (language dim plus optional deepstack lanes)
    - `n_deepstack_layers`
  - Multimodal request domain types used by app/runtime boundary:
    - `GenerationRequest`
      - can preserve an empty system turn and request a plain assistant prefill for exact
        processor contracts without adding family checks to app code
    - `ContentPart`
    - `MediaRef`
    - `EncodedPrompt`
    - `PlaceholderSpan`
    - `AudioTranscriptionResult` (raw model output, detected/forced language, transcript)
    - `KvCacheFormat`, also accepted as a generic run-state allocation override so app/vendor policy
      can select a quality-safe cache without engine family branches
  - Vendor tokenizer policy type used by tokenizer init:
    - `VendorTokenizerPolicy` (`disable_bos_fallback`, `end_turn_token_literals`)
  - Model multimodal capability metadata:
    - `MultimodalBackend`
    - `AudioEncoderBackend` (currently `Qwen3Asr`; selects the dedicated audio encoder constructor without adding family branches to app flow)
    - `ModelCapabilities`
  - Extended model identity flags:
    - `Config::is_qwen35`
    - `Config::rope_sections` for Qwen3.5 M-RoPE section metadata
    - `Config::online_attn_fusion` for vendor-selected dense-attention fast paths
  - Unix mmap wrapper (`MappedFile`) including Linux memory advice hints for model mappings
  - `ensure_model_range(...)` helper used by quantized matmul paths (currently a no-op in local-file mode).
  - GGUF metadata value variants include integer arrays (`I64Array`) for keys such as `*.rope.dimension_sections`.

### `src/engine/io/*`

- GGUF parsing and low-level read helpers.
- `io/gguf.rs`:
  - Parses GGUF metadata/tensors.
  - Maps model file.
  - Provides metadata access helpers:
    - `get_gguf_int_from_map`, `get_gguf_float_from_map`, `get_gguf_i64_array_from_map`, `get_gguf_string_from_map`, `find_gguf_tensor`.

### `src/engine/vision/*`

- Image/video preprocessing utilities.
- `vision/preprocess.rs` currently provides deterministic preprocessing:
  - images:
    - decode (`png`/`jpeg`/`webp` via `image` crate)
    - resize to profile target using mode (`CenterCrop` / `FitWithin` / `Stretch`)
    - CHW float tensor conversion
    - configurable normalization profile (`UnitRange` / `MeanStd`)
  - videos:
    - currently unavailable in no-external-dependency mode (native decode path removed)

### `src/engine/audio/*`

- Offline audio decode and feature preprocessing, separate from the vision namespace.
- `audio/decode.rs`:
  - validates RIFF/WAVE structure with checked chunk and buffer arithmetic
  - accepts integer PCM at 8, 16, 24, or 32 bits and IEEE float at 32 or 64 bits, in plain or
    `WAVE_FORMAT_EXTENSIBLE` fmt layouts, and names the `ffmpeg` conversion when rejecting anything
    else
  - bounds every chunk by the bytes present, so streams written without seeking keep their complete
    frames while a file whose declared data size fits must be frame-aligned
  - enforces vendor-provided file, channel, source-rate, and decoded-sample limits
  - averages channels deterministically into mono and uses a miniaudio-compatible linear converter
    with a fourth-order Butterworth low-pass filter for sample-rate conversion
  - plans bounded sample chunks while retaining one contiguous target-rate mono buffer
  - probes validated WAV shape and exact resampled sample count before sample decode, feature extraction, or encoder execution
- `audio/features.rs`:
  - implements periodic Hann windowing, centered reflection padding, exact-length FFT, power spectrum, Slaney filters with area normalization, base-10 log, and Whisper dynamic-range normalization
  - creates mel-major feature windows with valid-frame metadata and zero-pads each window to the configured frame-chunk multiple
  - exposes the same checked feature-window plan independently for context preflight
  - the frame loop is rayon-parallel over a frame-major intermediate; the mel-major transpose is
    fused into the window split, so no whole-clip mel-major copy is materialized
  - `FftPlan` hoists the twiddle tables, the decimation-in-time leaf order, and a
    `leaf_length * leaf_length` DFT table out of the per-frame path; `FftScratch` holds the
    ping-pong buffers so the transform never allocates. This replaced a recursive form that
    re-derived twiddles from a modulo per inner iteration and allocated three `Vec`s per recursion
    node — for the 400-point transform, 10,000 integer divisions and ~93 allocations per frame
  - `build_mel_filter_spans` restricts the projection to each filter's nonzero bin span. Slaney
    filters are ~1.5% dense (394 of 25,728 weights), so this touches ~33x fewer groups of four
  - all of the above is bit-identical to the previous implementation: the butterfly expression
    association, the group-of-four-then-promote summation order, and its group boundaries are
    preserved, and skipped groups are exactly zero. Verified byte-for-byte over 4.25M feature
    values before and after
- `audio/mod.rs` owns the shared preprocessing profiles and prepared PCM/feature domain types consumed by vendor policy and future audio encoder execution.
- The initial Qwen3-ASR profile is 16 kHz, 400-point FFT/window, 160-sample hop, 128 mel bins, at most 800 effective frames per window, and padding to 100-frame boundaries.
- A deterministic 128,000-sample regression is derived from direct execution of pinned llama.cpp `mtmd_audio_preprocessor_qwen3a`: shapes match exactly and the measured cross-toolchain maximum absolute log-Mel error is `2.288818359e-5` (or `1.192092896e-7` with contraction disabled).
- The official 48-kHz/24-bit speech sample matches miniaudio across 240,820 resampled samples with
  maximum/mean absolute error `1.788139343e-7/3.564497433e-9`; short upsample and downsample impulse
  regressions lock the converter's filter transient and latency semantics.

### `src/engine/multimodal/*`

- Native multimodal embedding and prompt-injection subsystem.
- `multimodal/injection.rs`:
  - defines a modality-neutral language-space `MediaEmbeddingSequence`
  - merges image/audio placeholder spans in source-token order and validates per-modality order, global overlap, bounds, marker shape, media-index uniqueness, embedding dimensions, and checked expanded length
  - preserves begin/end markers, replaces each placeholder with variable-length embedding slots, and builds the token-aligned prefill injection map
  - preflights expanded context plus decode reserve and rejects overflow rather than truncating through media embeddings
- `multimodal/qwen3vl.rs`:
  - Qwen3-VL CLIP/mmproj image encoder path (`qwen3vl_merger`)
  - loads mmproj tensors, runs patch embedding + vision transformer + projector in Rust
  - reads `clip.vision.image_mean/std` normalization metadata from mmproj GGUF when available
  - loads and applies optional `v.deepstack.*` layer branches, fused into projected media embeddings
  - uses SIMD dot products + rayon head-parallel attention for the vision self-attention hot path
  - emits language-space image token embeddings for prompt injection
- `multimodal/gemma3.rs`:
  - Gemma3 CLIP/mmproj image encoder path (`clip.projector_type='gemma3'`)
  - runs ViT layers with separate q/k/v projections, patch-grid average pooling, RMS normalization, and `mm.input_projection` into text embedding space
  - full-resolution ViT path is default; optional pre-attention fast-pooling shortcut is opt-in via `GGUF_GEMMA3_ENABLE_FAST_POOL=1`
  - emits language-space image token embeddings for prompt injection
- `multimodal/qwen3_asr.rs`:
  - parses Qwen3-ASR `clip.audio.*` metadata into `Qwen3AsrAudioConfig`
  - validates all `a.*` convolution/transformer and `mm.a.*` projector shapes in GGML dimension order
  - derives the distinct 480-channel convolution width from the official tensor contract while the
    transformer width remains 896
  - infers and cross-checks the projector hidden dimension from its two matrix weights
  - validates supported GGML storage, quantization block alignment, overflow, and mapped-file bounds
  - executes the isolated encoder front end over prepared feature windows: three stride-2/padding-1 Conv2D stages, GELU-ERF, Qwen channel/frequency reorder, and quantized `a.conv_out` projection
  - mirrors llama.cpp's F16 im2col activation rounding for F16/F32 convolution kernels and resets token order at each padded 100-frame chunk boundary
  - adds the first 13 positional rows repeatedly for each chunk, then executes every pre-LayerNorm audio transformer block with full-window bidirectional multi-head attention, GELU-ERF FFNs, optional biases, and residual connections
  - applies required `a.post_ln.*` normal LayerNorm after the transformer and before the audio projector
  - rounds F16/BF16 activations to GGML's matrix dot type; BF16 ties-to-even and matmul behavior have focused regressions
  - executes `mm.a.mlp.1` -> GELU-ERF -> `mm.a.mlp.2` through quantized matrix kernels and returns 13 text-model-width audio embeddings per padded 100-frame chunk
  - retains every padding-derived output token, matching the pinned source graph until model-backed fixtures establish whether a later trim is required
- `multimodal/mod.rs`:
  - backend construction (`build_vision_encoder_from_mmproj`, `build_audio_encoder_from_mmproj`)
  - enables external vision `mmproj` construction for `gemma3`, `qwen3vl`, and `qwen35`, plus Qwen3-ASR audio weight-contract construction
  - encoder abstractions (`VisionEncoder`, `AudioEncoder`), including internal typed dispatch for isolated audio convolution, transformer, and language-space projector execution
  - predicts Qwen3-ASR output counts from planned padded feature windows, dispatches each window through the complete sidecar graph, and concatenates adjacent chunks inside one audio marker envelope
  - `examples/audio_encoder_dump.rs` probes official sidecars and dumps convolution, per-layer,
    post-transformer, and projected F32 tensors from synthetic input, PCM WAV, or a captured
    convolution output
  - `examples/audio_prep_bench.rs` times `prepare_audios_for_multimodal` over one or more WAV files
    (`REPS` controls repetitions) and, with `DUMP_DIR` set, writes each window's raw f32 bits so a
    front-end change can be diffed for bit equality against a baseline build
  - audio readiness is enabled after Q8_0 automatic-language, forced-language, and silence
    transcription gates

### `src/engine/tokenizer/mod.rs`

- Tokenizer initialization and encode/decode logic.
- Handles sentencepiece/tiktoken-ish paths and special token resolution.
- Applies vendor-provided tokenizer policy for BOS fallback and end-of-turn token lookup.
- Exposes `init_tokenizer_from_gguf(...)`.

### `src/engine/weights.rs`

- Loads and validates model tensors from GGUF into `TransformerWeights`.
- Handles per-family tensor layout differences and optional tensors.
- Handles Qwen3.5 split-SMM gate tensor compatibility (`ssm_alpha.weight` / `ssm_beta.weight`) alongside fused `ssm_ba.weight`.
- Provides multimodal tensor-group initialization/probe (`init_multimodal_weights_from_gguf`) with explicit missing-group diagnostics for native multimodal backends.
- Exposes `init_weights_from_gguf(...)`.

### `src/engine/kernels/*`

- Numerical and sampling kernels used by inference.
- `math.rs`: normalization, softmax, vector math, Qwen3Next SSM linear attention helpers.
- `quant.rs`: quantized dequant/dot/matmul paths, reusable activation scratch and explicit prepared-activation matmul entry points for allocation-free Q8 prequantized fast paths and x86 Q2_K/Q4_K/Q5_K/Q6_K VNNI activation reuse (including half-block activation sums for Q2_K min correction), architecture-specific fast paths (including pre-quantized activation reuse for Q8 matmul on aarch64, x86 AVX2/FMA-preferred Q8 paths with optional fallback to lossy VNNI Q8 kernels, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K MR4 dispatch, ARM Q2_K/Q3_K and legacy Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL MR4 NEON coverage, x86 Q2_K/Q3_K plus legacy Q4_0/Q4_1/Q5_0/Q5_1 and IQ4_NL MR4 AVX2/FMA table-shuffle coverage, and x86 Q2_K/Q4_K/Q5_K/Q6_K MR4 AVX-VNNI/AVX512-VNNI paths), MR4 validation (int8-aware scalar references for the VNNI dispatch paths), architecture-specific matmul row prefetch helpers (x86 + aarch64), exact batched-prefill helpers including serial and parallel tiled Q2_K/Q3_K/Q4_K/Q5_K/Q6_K scalar-exact dispatch for non-MR4 row windows, and the batched quantized matmul helper used by the RAG BERT embed path with caller-owned dequant scratch reuse.
  - one-time kernel self-check disable warnings are now quiet by default and can be re-enabled with `GGUF_KERNEL_VALIDATION_WARNINGS=1`
- `sampling.rs`: token selection helpers (`argmax`, multinomial sample, top-k/top-p sampler).

### `src/engine/runtime/*`

- Runtime-specific execution and threading config.
- `runtime/inference.rs`:
  - `malloc_run_state(...)`
  - `malloc_run_state_with_kv_cache_format(...)` for a generic request-scoped Q8/Turbo override;
    Qwen3-ASR selects Q8 in vendor policy because Turbo fails the audio quality gate
  - `transformer(...)`
  - `transformer_without_logits(...)` for prompt-prefill steps that only need KV/cache state
  - `transformer_with_embedding(...)` (prefill hook for external embedding vectors)
  - `transformer_with_embedding_without_logits(...)` for embedded prefill steps that only need KV/cache state
  - `transformer_prefill_batch(...)` runs a chunk of prompt positions through every layer, batching
    the dense FFN while the attention half stays per token; `batch_prefill_supported(...)` gates it
    (post-norm BERT, Gemma3 ordering, and per-token MoE routing fall back to the sequential loop)
  - `PrefillInput::{Token, Embedding}` lets that chunk mix vocabulary tokens with encoder-produced
    embeddings, so audio and image prompts prefill in batches instead of one position at a time.
    Embeddings must be `dim` wide; deepstack prompts, whose embeddings carry a per-layer tail that
    only the sequential path replays, stay on the per-token loop
  - accepts multimodal prefill vectors at either `dim` or `input_embedding_dim`
  - applies per-layer deepstack residual injection for Qwen3-VL-style expanded embeddings
  - reuses kernel activation scratch across the high-frequency sequential projection calls in a token step, including prepared-activation reuse across compatible dense, BERT fused, Qwen3Next full-attention, and FFN gate/up projection groups
  - applies llama-style Qwen3.5 M-RoPE cache reconstruction for text decode (`[t,h,w,e]=[pos,pos,pos,0]`)
  - owns reusable per-token scratch buffers for routed MoE experts, including selected-expert contribution staging in `RunState`
  - quantized KV cache storage for attention state:
    - default TurboQuant-style cache mode with head-wise signed-Hadamard rotation, packed 2-bit base codes, and packed 1-bit residual sketches
    - optional Q8 cache mode
  - aarch64 attention helpers include NEON-accelerated Q8 and Turbo KV dot+axpy paths for per-head cached key/value reads
- `runtime/parallel.rs`:
  - `configure_rayon_threads(...)`
- `runtime/mod.rs`:
  - Re-exports runtime helpers.
  - `apply_context_size_overrides(...)` (applies explicit `--context-size` override only).

### `src/engine/switches.rs`

- Runtime tuning and feature switches.
- Keeps `OnceLock` / atomic-backed switch state as system-of-record.
- Includes:
  - `RuntimeSwitchConfig` (engine-owned overrides struct)
  - Parallel thresholds (`par_matmul_min_rows`, `par_matmul_chunk_rows`, `par_attn_min_heads`, `par_qwen3next_min_heads`)
  - AArch64 matmul row prefetch distance switch (`aarch64_matmul_prefetch_rows`)
    - default values for non-x86 matmul thresholds and aarch64 prefetch distance are now derived from `available_parallelism()` heuristics (with CLI/env overrides preserved)
  - KV cache selection switch (`kv_cache_mode`: `q8` / `turbo`, default `turbo`)
  - Arch feature toggles (`use_x86_*`, `use_aarch64_*`, including x86 AVX2/F16C/QK-MR4/AVX-VNNI/AVX512VNNI-Q8 switches)
    - default behavior uses runtime CPU feature detection for architecture fast paths (for example aarch64 `dotprod` Q8 and x86 `AVX512VNNI` Q8), while `RuntimeSwitchConfig`/CLI/env can still force-disable paths
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
  - Rejects unsupported DeepSeek architectures (`deepseek*`) with a clear config error.
  - Builds `Config` from family-specific key conventions.
  - Detects `qwen35` explicitly and maps it onto the Qwen3Next-style runtime path.
  - Keeps `qwen35*` checkpoints on `qwen35` vendor prompt/decode policies even when the runtime executes their recurrent/SSM layers through the Qwen3Next-style engine path.
  - Sets generic runtime feature flags on `Config` from vendor/model metadata so engine code can consume runtime behavior without new family branches.
  - Probes multimodal capability from tokenizer special tokens + GGUF tensor prefixes for `gemma3`, `qwen3vl`, and `qwen35`.
  - Performs vendor-specific mmproj sidecar compatibility checks (`validate_mmproj_for_backend(...)`):
    - vision projector type, projection dim matching, and Qwen family/deepstack guards
    - Qwen3-ASR audio-only metadata (`clip.audio.*`), text/audio dimension compatibility, and mandatory `a.*`/`mm.a.*` tensor-name validation
    - rejects combined vision/audio sidecars until both encoder paths can be constructed independently
  - Dispatches vendor policies used by app/tokenizer decode paths:
    - `decode_policy(...)` returning `VendorDecodePolicy` (`parse_think_tags`, `stop_token_literals`, `deterministic_loop_guard`, hidden/visible think budgets, multimodal think preference, think-retry toggles)
    - `tokenizer_policy(...)` returning `VendorTokenizerPolicy`
    - `multimodal_policy(...)` returning `VendorMultimodalPolicy` (image prompt suffix, detail-crop behavior, mmproj candidate scoring hints, sidecar diagnostics hint)
    - `audio_preprocess_config(...)` returning the backend-specific decode, resource-limit, log-Mel, and feature-window profile without placing Qwen3-ASR constants in app/engine branches
    - `audio_transcription_policy(...)` returning backend-owned request construction, decode/stopping,
      forced-language validation, typed output parsing behavior, and required KV-cache format
    - `runtime_debug_policy(...)` returning `VendorRuntimeDebugPolicy` (family-specific native-context debug label)
  - Routes both simple chat prompt encoding and structured `GenerationRequest` encoding to family-specific implementation.
- `vendors/llama.rs`, `vendors/gemma.rs`, `vendors/qwen*.rs`:
  - Family-specific defaults, validations, prompt rendering, and family-owned policy constructors.
  - Qwen family is split by variant:
    - `qwen2.rs`: Qwen2 chat template + baseline decode/tokenizer policies.
    - `qwen3.rs`: Qwen3-MoE defaults/validation helpers and Qwen3 prompt wrappers.
    - `qwen3_asr.rs`: Qwen3-ASR context/audio request contract, supported forced languages,
      `<asr_text>` parser, silence/fallback handling, upstream-compatible repetition cleanup, and
      Q8 KV-cache requirement.
    - `qwen35.rs`: Qwen3.5 decode/tokenizer/multimodal policies (detail-crop opt-in and sidecar hints).
    - `qwen3vl.rs`: Qwen3-VL decode/tokenizer/multimodal policies.
    - `qwen3next.rs`: Qwen3-Next SSM validation/debug + decode/tokenizer policies.
    - `qwen_common.rs`: shared Qwen stop-token constants, runtime debug policy, and Qwen3 structured prompt encoder with image/video/audio placeholder-span mapping; audio uses the exact `<|audio_start|><|audio_pad|><|audio_end|>` marker sequence, and exact assistant prefills preserve `<asr_text>` as one vocabulary token.
  - Tokenizer initialization honors `tokenizer.ggml.add_bos_token`; this prevents Qwen3-ASR's
    converted `<|im_end|>` BOS ID from being inserted when its source tokenizer says
    `add_bos_token=false`.

## Runtime Data Flow

1. `main.rs` invokes `app::run()`.
2. `app::run()` parses CLI (`CliOptions`).
3. `app::run()` builds `RuntimeSwitchConfig` and calls `engine::switches::init_runtime_config(...)`.
4. GGUF parsed via `engine::io::parse_gguf_file(...)`.
5. Vendor config built with `vendors::build_config_from_gguf(...)`.
6. Vendor tokenizer, multimodal, and runtime debug policies are built (`vendors::{tokenizer_policy,multimodal_policy,runtime_debug_policy}(...)`) and tokenizer is initialized (`engine::tokenizer::init_tokenizer_from_gguf(...)`).
7. Runtime overrides applied (`engine::runtime::apply_context_size_overrides(...)`), with vendor runtime-context debug logging handled in app.
8. Weights loaded (`engine::weights::init_weights_from_gguf(...)`).
9. Standard mode:
  - CLI media inputs are normalized into `engine::types::GenerationRequest` with `ContentPart` items.
  - prompt encoded via `vendors::encode_generation_request(...)`, including placeholder spans for image/video/audio on Qwen multimodal paths and image spans on Gemma multimodal path.
  - runtime validates prompt/media alignment before starting preprocessing.
  - if native multimodal tensors are unavailable, runtime fails with a qualified native-capability error that includes architecture/token/tensor probe details.
  - vendor decode policy built (`vendors::decode_policy(...)`) and applied by the token loop for think-tag parsing, phase-bounded visible/hidden think decoding, stop-token matching, deterministic loop-guard behavior, and vendor-enabled think-recovery retries.
  - both repetition guards are policy-owned: `deterministic_loop_guard` covers the suffix/line checks and `unconditional_loop_guard` covers the repeated-phrase and token-cycle checks, so families whose valid output repeats verbatim, such as transcription, keep their full output.
  - token loop executes forward passes (`engine::runtime::transformer(...)`) + sampling (`engine::kernels`); image injection is active, while audio injection is wired but held behind the model-parity readiness gate.
  - the internal transcription route obtains its policy from the loaded `AudioEncoderBackend`, keeps
    context as the exact system turn (without RAG augmentation), uses a plain ChatML assistant prefix,
    and parses the raw continuation into `AudioTranscriptionResult` after generation.
10. Agent mode:
  - tool transcript prompt encoded per turn
  - model emits JSON `tool_call` / `final`
  - host executes tool call (`tools`) and loops until final response or limit.
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
