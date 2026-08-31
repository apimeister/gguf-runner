# Features and Platform Support

This document summarizes the current runtime capabilities of `gguf-runner`.

## Model Family Support

Model-family handling is selected from GGUF metadata (`general.architecture`) and family-specific keys.

Supported families:
- Llama-style architectures
- Gemma (`gemma`, `gemma2`, `gemma3`)
- Qwen / Qwen2
- Qwen3.5 (`qwen35`, loaded through the Qwen3Next-style recurrent/full-attention path with dense FFN)
- Qwen3-VL (`qwen3vl`)
- Qwen3 MoE (`qwen3moe`)
- Qwen3 Next (`qwen3next`, including SSM-related tensors)

Currently unsupported:
- DeepSeek architectures (`deepseek*` GGUF metadata)

## Speaker Recognition

- Dedicated `speaker_xvector` GGUF runtime; no language model or ASR-language embeddings are used.
- Stateless, versioned JSONL embedding output with model fingerprint and recording-quality
  metadata; exported embeddings can be reused for enrollment, verification, or identification
  without another encoder pass.
- Versioned local `.spkidx` profiles with append-only observations and derived quality-weighted
  centroids.
- Additive enrollment, 1:1 verification, open-set 1:N identification, explicit unknown results,
  top-one/top-two margins, profile/observation listing and removal.
- Conservative refinement modes: off by default, editable JSONL candidates, or explicitly enabled
  high-confidence automatic updates.
- In-process energy VAD, known-speaker association, and unknown-speaker clustering for finite
  meeting WAVs, with an optional profile index; overlapping-speaker separation is not implemented.
- File and embedded-byte `SpeakerRuntime` entrypoints plus a retained, model-free
  `SpeakerIndexRuntime` for exported-embedding and profile-management operations.
- Model/index fingerprints prevent cross-model embedding comparisons. Model thresholds are required
  metadata rather than universal runner constants.
- Speaker association is not authentication or liveness detection.

See `docs/speaker-recognition.md` for usage, privacy guidance, limitations, and the GGUF contract.

## Quantization / Tensor Type Support

Supported tensor data paths include:
- `F32`, `F16`, `BF16`
- F16/BF16 multimodal encoder matrices reuse each expanded weight row across a token batch; BF16
  batches pack activations once and use native `BFDOT` on compatible AArch64 hosts (including
  Apple M5), with portable NEON widening/F32 FMA as the fallback; x86_64 uses AVX2 widening/FMA
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- `IQ4_NL`

## Runtime Features

- GGUF parsing from local files
- Linux mmap memory-advice hints for mapped model pages (best-effort)
- tokenizer initialization from GGUF vocab/metadata
- model-family-specific chat prompt rendering
- multimodal request/model capability scaffolding for Gemma3, Qwen3-VL, and Qwen3.5:
  - startup capability probe for native image/video/audio support (token + tensor checks)
  - llama-style local `mmproj*.gguf` sidecar auto-discovery/probe (no extra CLI flag)
  - strict native-only multimodal execution (no metadata fallback path)
  - multimodal tensor-group probe during runtime load:
    - vision encoder tensor groups
    - multimodal projector tensor groups
    - audio tensor groups
    - explicit missing-group errors when backend is marked native-capable
- multimodal request scaffolding:
  - repeatable `--video <path>` input parsing/validation (`mp4`)
  - repeatable `--audio <path>` input parsing/validation (extension-agnostic)
  - structured prompt encoding for multimodal requests with placeholder span mapping:
    - Gemma3 image placeholders (`<start_of_image>` / `<end_of_image>`)
    - Qwen image/video/audio placeholders
  - runtime prompt/media alignment validation before preprocessing
  - generation loop prefill hook for external embeddings (`transformer_with_embedding`) supports ordered image/audio embedding injection internally
  - media prompts prefill in batches: injected embeddings travel through `transformer_prefill_batch`
    as `PrefillInput::Embedding` rather than forcing the per-position loop. Deepstack models, whose
    embeddings carry a per-layer tail, keep the sequential path
  - media expansion validates spans globally and refuses context overflow instead of truncating through embedding sequences
  - clearer media capability diagnostics when GGUF is missing native multimodal tensor groups
    - includes sidecar search results and effective support status
  - native preprocessing foundation:
    - image:
      - PNG/JPEG/WebP decode
      - deterministic resize modes (`CenterCrop`, `FitWithin`, `Stretch`) selected per backend/profile
      - RGB -> CHW tensor conversion
      - normalization profiles (`UnitRange` and `MeanStd`)
    - video:
      - currently unavailable in no-external-dependency mode
    - audio:
      - bounded RIFF/WAVE decode of integer PCM (8/16/24/32-bit) and IEEE float (32/64-bit),
        including the `WAVE_FORMAT_EXTENSIBLE` header that writers emit for more than two channels
        or more than 16 bits
      - chunk sizes are bounded by the bytes present, so WAV streams written without seeking
        (`0xffffffff` RIFF/data sizes) decode down to their last complete frame
      - deterministic channel averaging and miniaudio-compatible LPF4 linear resampling to a
        vendor target rate
      - Qwen3-ASR source-contract log-Mel extraction and 800/100-frame windowing
      - pinned llama.cpp synthetic log-Mel parity regression with a measured `3e-5` portable tolerance
      - log-Mel extraction runs rayon-parallel per frame over a table-driven, allocation-free FFT
        and a sparsity-aware mel projection, bit-identical to the previous serial implementation
        (see `docs/performance.md` for the measured effect)
      - isolated Qwen3-ASR sidecar execution through the 480-channel convolution front end,
        repeated per-chunk positional embeddings, all 18 audio transformer blocks, required post
        LayerNorm, and the two-layer `mm.a.*` GELU-ERF projector
      - official Q8_0/BF16 sidecar probe support through `examples/audio_encoder_dump.rs`, including
        stable stage dumps for model-backed numerical comparison
      - `examples/audio_prep_bench.rs` times the decode/log-Mel front end and, with `DUMP_DIR` set,
        writes raw feature bits so front-end changes can be checked for bit equality
      - produces 13 text-model-width audio embeddings per padded 100-frame chunk while preserving padding-derived outputs
      - WAV shape probing predicts audio embedding counts before decode/FFT, reserves the full
        vendor transcription decode budget (512 tokens), and reports duration/context overflow
        before encoder work
      - Qwen3-ASR transcription contract is implemented internally: context-only system turn,
        audio-only user turn, optional canonical forced-language assistant prefill, greedy decode,
        `<|im_end|>` stopping, and typed raw/language/transcript parsing
      - auto-language parsing accepts `language X<asr_text>...`, text-only fallback, and
        `language None<asr_text>` silence output; forced-language continuations are treated as plain text
      - transcription decoding keeps repeated speech: the vendor decode policy switches off the
        generic repeated-phrase and token-cycle guards through `unconditional_loop_guard`, and
        degenerate decoder loops are collapsed by the vendor output normalizer instead
      - transcription prefill uses the vendor-required Q8 KV cache even when the general runtime
        default is Turbo; an exact projected-embedding A/B showed Turbo corrupting the low-norm audio
        signal while Q8 reproduced the reference transcript
      - request-level Qwen3-ASR execution passed automatic-language, forced-English, and silence
        quality gates with the official Q8_0 text/sidecar pair
  - one-shot CLI accepts exactly one `--audio`, treats `--prompt` as transcription context, accepts
    optional `--audio-language`, and writes transcript-only output to stdout
  - `--audio-batch <manifest.jsonl>` validates a strict `{id,path,language?,context?}` JSONL schema
    before model load, resolves relative audio paths from the manifest directory, retains one
    runtime, processes records serially, and flushes ordered structured success/error results
  - `--image-batch <manifest.jsonl>` provides the same retained-runtime, prevalidated, ordered,
    per-record-error contract for strict `{id,path,prompt,system_prompt?}` image jobs; bounded
    four-record chunks jointly encode compatible same-shape Qwen3-VL/Qwen3.5 images while preserving
    image-isolated attention, then run independent text generation in input order
  - `EmbeddedRuntime::transcribe_audio(...)` exposes the typed raw/language/transcript result and
    supports serial offline batch loops with one retained model/runtime
  - current runtime returns explicit errors for native video execution, mixed audio/image/video
    input, and repeated `--audio` attachments
- autoregressive generation loop
- quantized KV cache for attention state:
  - default TurboQuant-style `turbo` KV cache mode:
    - head-wise signed-Hadamard rotation before scalar quantization
    - 2-bit rotated-domain base codebook plus 1-bit residual sketch per channel
    - per-head scale and residual-norm metadata used during cached key/value reads
  - optional `Q8` KV cache mode
- optional tool-agent loop (`--agent`) with host-side file tools:
  - `read_file`
  - `list_dir`
  - `write_file`
  - `mkdir` (recursive directory creation)
  - `rmdir` (recursive directory removal)
  - `shell_list_allowed` (reports currently enabled tools + allowed shell commands)
  - `shell_exec` (restricted to operator-defined allowed commands)
  - `shell_request_allowed` (asks operator to allow a specific shell command)
- sampling modes:
  - greedy (`--temperature 0`)
  - stochastic temperature sampling
  - top-k / top-p (note: `top-p` is applied when `top-k > 0`)
  - repetition control (`--repeat-penalty`, `--repeat-last-n`)
- runtime diagnostics:
  - `--debug`
  - `--show-tokens`
  - `--show-timings`
  - `--profiling`

## CLI + Environment Configuration

User-facing CLI options are defined in `src/cli.rs`.

Media batch and audio-specific options:

- `--audio <path>`: one finite file for offline Qwen3-ASR transcription
- `--audio-language <language>`: optional canonical forced language; requires `--audio`
- `--prompt <text>`: transcription context/hotword hint when `--audio` is present
- `--audio-batch <manifest.jsonl>`: serial offline transcription with one model load and JSONL output
- `--image-batch <manifest.jsonl>`: bounded vision microbatches plus ordered independent image
  generation with one model/sidecar load and JSONL output

Agent config file (optional):
- `~/.gguf-runner/config.toml`
- `./.gguf-runner/config.toml` (loaded after home config and overrides it)

Shell allowed-commands config schema:
```toml
[tools]
read_file = true
list_dir = true
write_file = true
mkdir = true
rmdir = true
shell_list_allowed = true
shell_exec = true
shell_request_allowed = true

[shell.cmd]
rg = "Fast recursive text search."
ls = "List directory entries."
cat = "Read file content."
cwd = "Show current working directory (shell_exec built-in helper)."
```

Exposed env var:
- `GGUF_RAYON_THREADS` (same as `--threads`)
- `GGUF_ALLOW_SHELL_COMMANDS` (comma-separated allowed commands for `shell_exec`)

Hidden runtime tuning env vars (advanced use):
- `GGUF_PAR_MATMUL_MIN_ROWS`
- `GGUF_PAR_MATMUL_CHUNK_ROWS`
- `GGUF_PAR_ATTN_MIN_HEADS`
- `GGUF_PAR_QWEN3NEXT_MIN_HEADS`
- `GGUF_KV_CACHE_MODE` (`q8`, `turbo`)
- `GGUF_MM_FLOAT_BATCH` (`0` restores per-token F16/BF16 multimodal encoder matmul)
- `GGUF_KERNEL_VALIDATION_WARNINGS` (`1`/`true` to print one-time kernel self-check disable warnings)
- `GGUF_LAYER_DEBUG`
- `GGUF_LAYER_DEBUG_POS`
- `GGUF_AARCH64_DOTPROD_Q8` (aarch64 only)
- `GGUF_AARCH64_QK_MR4` (aarch64 only)
- `GGUF_AARCH64_BF16` (`0` disables native `BFDOT` BF16 batch matmul; aarch64 only)
- `GGUF_X86_AVX2` (x86_64 only)
- `GGUF_X86_F16C` (x86_64 only)
- `GGUF_X86_QK_MR4` (x86_64 only)
- `GGUF_X86_AVXVNNI` (x86_64 only)
- `GGUF_X86_AVX512VNNI_Q8` (x86_64 only; optional lossy `Q8_0` VNNI path used only when the exact AVX2/FMA path is unavailable or disabled)

## Supported Platforms

Current target platforms:
- macOS (aarch64 and x86_64)
- Linux (aarch64 and x86_64)

Notes:
- runtime uses Unix memory-mapping paths for GGUF loading
- platform-specific SIMD paths are implemented for `aarch64` and `x86_64`
- non-Unix platforms (for example Windows) are not currently the primary target

## Current Boundaries

- CPU-only runtime (no GPU backend)
- GGUF-only model format
- model compatibility depends on expected tensor layout and metadata presence
- native video decode remains unavailable; native audio transcription reads RIFF/WAVE integer PCM
  and IEEE float input, so compressed containers such as MP3, FLAC, and M4A need external conversion
