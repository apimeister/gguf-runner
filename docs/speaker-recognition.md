# Speaker Recognition and Additive Profiles

`gguf-runner` can create speaker embeddings, maintain additive speaker profiles, verify a claimed
speaker, identify a speaker from a local index, and perform basic meeting diarization. All runtime
work stays inside the process: audio decode, resampling, feature extraction, GGUF inference, cosine
matching, clustering, and index updates.

This is speaker **association**, not authentication. A recording can be replayed or synthesized,
and the current runtime does not implement liveness detection. Do not use a positive match as the
only factor for access control, signatures, payments, or another security decision.

## Requirements and current limits

For operations that consume audio, you need:

- one GGUF implementing the `speaker_xvector` contract described below;
- RIFF/WAVE input containing integer PCM or IEEE float samples;
- recordings with one dominant speaker for embedding, enrollment, verification, and identification.

Operations on an existing `.spkidx` and previously exported embeddings do not load the speaker
GGUF. This includes embedding-only enrollment, verification and identification, candidate
acceptance, profile listing, and removals.

The repository does not bundle speaker model weights. A compatible GGUF is model data, like the
text and multimodal GGUFs used elsewhere by the runner; it is not a separate runtime or service.
ONNX and PyTorch checkpoints cannot be passed directly to `--speaker-model`.

Current input and diarization limits:

- WAV is decoded in-process at any supported source sample rate and channel count. MP3, AAC/M4A,
  Opus, and other containers still need to be converted to WAV before invocation.
- Meeting diarization uses an internal energy VAD and non-overlapping 2.5-second speaker windows.
  It does not separate simultaneous speakers, and change boundaries are approximate.
- Thresholds are model- and deployment-specific. The speaker GGUF must carry calibrated starting
  thresholds; the runner does not hide a universal `0.7` constant behind the CLI.

## Embed versus enroll

Embedding is stateless. It returns a normalized vector and never opens or changes an index:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-embed \
  --audio ./alice-clean.wav
```

The JSON output contains the exact model fingerprint, quality measurements, and vector. Repeating
`--audio` emits one JSON object per line while keeping the model loaded:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-embed \
  --audio ./alice-1.wav \
  --audio ./alice-2.wav \
  > embeddings.jsonl
```

Embedding output is versioned JSONL and can be reused without decoding or running the encoder
again. For example, enroll every record in a saved file:

```bash
gguf-runner \
  --speaker-index ./team.spkidx \
  --speaker-enroll alice \
  --speaker-embedding-input ./alice-embeddings.jsonl
```

`--speaker-enroll` accepts multiple audio and embedding inputs in one atomic update. Verification
and identification accept either one `--audio` or one embedding record:

```bash
gguf-runner \
  --speaker-index ./team.spkidx \
  --speaker-identify \
  --speaker-embedding-input ./one-embedding.jsonl
```

The existing index validates the embedding format, version, exact model fingerprint, architecture,
dimension, normalization, and quality metadata before use. A verification or identification input
file must therefore contain exactly one record; enrollment files may contain many. Creating a new
index from exported embeddings still requires `--speaker-model` once, because calibrated model
thresholds are not duplicated into every embedding record.

Enrollment is stateful. The first observation creates a profile, and later observations refine it:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-enroll alice \
  --audio ./alice-clean.wav \
  --audio ./alice-headset.wav
```

All recordings in one enrollment invocation are embedded and checked before the index is replaced.
Every accepted observation remains in the index. The profile centroid is derived from all of them,
weighted by usable duration and recording quality; it is not an irreversible moving average.

When an existing profile and a new recording fall below the model's enrollment-consistency
threshold, the command fails without writing that invocation. Check that the identity and recording
are correct before overriding this guard:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-enroll alice \
  --speaker-force-enroll \
  --audio ./alice-unusual-channel.wav
```

## Verification and identification

Verification answers a 1:1 question: does this recording match the claimed profile?

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-verify alice \
  --audio ./claim.wav
```

The JSON result includes `verified`, the raw cosine `score`, and the applied `threshold`.

Identification answers a 1:N, open-set question:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-identify \
  --audio ./unknown.wav
```

The result is recognized only when the best score reaches the index threshold and, when there is a
runner-up, the top-one/top-two margin reaches the index margin. Otherwise `recognized` is false and
`speaker_id` is null. Raw candidates and scores remain in `matches` for audit and tuning.

## Refining profiles from later recordings

The safe default is no implicit learning. To collect high-confidence matches for review:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-identify \
  --speaker-learning candidates \
  --speaker-candidates ./review.jsonl \
  --audio ./later-recording.wav
```

A candidate is written only when it passes the model's stricter auto-learning threshold and the
identification margin. New candidate records contain `"accepted":false`. Review the identity,
source, score, and margin, then change only the records you approve to `"accepted":true` and run:

```bash
gguf-runner \
  --speaker-index ./team.spkidx \
  --speaker-accept ./review.jsonl
```

Records that remain false are skipped. Accepted embeddings must have the same model fingerprint and
dimension as the index and still pass profile consistency before the index is replaced.

Automatic refinement is available but deliberately explicit:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-identify \
  --speaker-learning auto \
  --audio ./later-recording.wav
```

Automatic learning uses the stricter model threshold, margin gate, audio-quality checks, duplicate
protection, and profile consistency check. Candidate review is still recommended: an incorrect
prediction can otherwise reinforce itself.

## Meetings and diarization

An index is optional. Without one, all detected voices are clustered as invocation-local unknown
speakers:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-diarize \
  --audio ./meeting.wav \
  > meeting-clusters.json
```

Index-free diarization supports `--speaker-learning off` only because there is no known profile to
refine.

Run local meeting segmentation and association with:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-diarize \
  --audio ./meeting.wav \
  > meeting-speakers.json
```

Known speakers use their profile id. Speech that does not pass open-set identification is clustered
as `unknown-1`, `unknown-2`, and so on for that invocation. Consecutive windows with the same label
are merged in the output.

Meeting recordings can also produce reviewed updates:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-diarize \
  --speaker-learning candidates \
  --speaker-candidates ./meeting-review.jsonl \
  --audio ./meeting.wav
```

High-confidence segments for the same known speaker are combined into one candidate for that
meeting. This limits correlated observations and index growth. The `auto` mode applies the same
combined candidate directly.

Meeting diarization is not speaker separation. A segment containing overlapping voices can produce
an unreliable embedding and should not be accepted for profile refinement.

## Inspecting and undoing profile changes

List profiles, observation ids, sources, durations, and quality scores:

```bash
gguf-runner \
  --speaker-index ./team.spkidx \
  --speaker-list
```

Remove one bad observation and rebuild the remaining centroid:

```bash
gguf-runner \
  --speaker-index ./team.spkidx \
  --speaker-remove-observation obs-0123456789abcdef
```

Delete a complete speaker profile:

```bash
gguf-runner \
  --speaker-index ./team.spkidx \
  --speaker-remove alice
```

## Threshold calibration

The GGUF supplies verification, identification-margin, enrollment, auto-learning, and unknown
cluster thresholds. When creating or deliberately recalibrating an index, verification and margin
can be overridden during enrollment:

```bash
gguf-runner \
  --speaker-model ./speaker-xvector.gguf \
  --speaker-index ./team.spkidx \
  --speaker-enroll alice \
  --speaker-threshold 0.63 \
  --speaker-margin 0.08 \
  --audio ./alice.wav
```

These values are persisted in the index. Calibrate them with representative same-speaker,
different-speaker, microphone, room, language, and noise conditions. Do not copy a threshold from a
different model or deployment. Cosine thresholds range from `-1` to `1`; the nonnegative
top-one/top-two margin ranges from `0` to `2`.

## Index behavior and privacy

`.spkidx` is strict, versioned JSON written through a same-directory temporary file and atomic
rename. It contains:

- the exact speaker-model fingerprint, architecture, and embedding dimension;
- stored thresholds;
- each speaker id and derived centroid;
- every enrolled embedding with source, duration, quality, and observation id.

Embeddings are biometric data. Restrict access to exported embedding, index, and candidate files;
avoid logging vectors, obtain consent, define retention/deletion rules, and encrypt the containing
storage where appropriate. The FNV model fingerprint prevents accidental model mixing; it is an
identity guard, not a cryptographic signature for an untrusted model file.

Index replacement is atomic, but multiple writer processes are not locked against one another.
Serialize enrollment, accepted-candidate, automatic-learning, and removal commands for a given
index. The same applies when multiple processes append to one candidate JSONL file.

## Embedded Rust API

The library facade exposes a separate runtime so a caller does not load an LLM for speaker work:

```rust,no_run
use std::path::Path;
use gguf_runner::{SpeakerIndexRuntime, SpeakerRuntime};

let runtime = SpeakerRuntime::load_from_file(Path::new("speaker-xvector.gguf"))?;
let embedding = runtime.embed_file(Path::new("voice.wav"))?;
println!("{} dimensions", embedding.dimension);

runtime.enroll_embedding(
    Path::new("team.spkidx"),
    "alice",
    &embedding,
    false,
)?;

let identification = runtime.identify_embedding(
    Path::new("team.spkidx"),
    &embedding,
)?;

// Later, retain only the index for repeated vector operations.
let index = SpeakerIndexRuntime::load(Path::new("team.spkidx"))?;
let identification = index.identify_embedding(&embedding)?;
# Ok::<(), String>(())
```

`SpeakerRuntime::load_from_bytes(include_bytes!(...))` is also available for applications that
compile the GGUF into their binary. File-based `enroll_file`, `verify_file`, and `identify_file`
methods remain available when caching or transporting embeddings is unnecessary.
`SpeakerIndexRuntime` retains a validated index in memory and supports model-free enrollment,
verification, identification, listing, and removals for exported embeddings.

## `speaker_xvector` GGUF contract

This section is for model publishers and converter authors. Normal users should obtain an already
compatible model.

Required metadata:

| Key | Type | Meaning |
|---|---:|---|
| `general.architecture` | string | exactly `speaker_xvector` |
| `general.name` | string | display name; optional but recommended |
| `speaker_xvector.audio.sample_rate` | integer | encoder input rate |
| `speaker_xvector.audio.fft_length` | integer | even FFT length |
| `speaker_xvector.audio.window_length` | integer | Hann window length |
| `speaker_xvector.audio.hop_length` | integer | frame hop in samples |
| `speaker_xvector.audio.mel_bins` | integer | input feature count |
| `speaker_xvector.audio.mel_floor` | float | positive log floor |
| `speaker_xvector.audio.max_window_frames` | integer | independent-window limit |
| `speaker_xvector.audio.min_seconds` | float | minimum accepted speech clip |
| `speaker_xvector.audio.max_seconds` | float | decoded-file safety limit |
| `speaker_xvector.tdnn.layer_count` | integer | number of frame layers |
| `speaker_xvector.tdnn.N.context` | integer array | signed frame offsets for layer N |
| `speaker_xvector.segment.layer_count` | integer | number of post-pooling affine layers |
| `speaker_xvector.threshold.verification` | float | initial open-set/verification threshold |
| `speaker_xvector.threshold.identification_margin` | float | required top-one/top-two gap |
| `speaker_xvector.threshold.enrollment` | float | profile consistency threshold |
| `speaker_xvector.threshold.auto_learning` | float | stricter automatic-update threshold |
| `speaker_xvector.threshold.diarization_cluster` | float | unknown-cluster threshold |

Required tensors for every TDNN layer N:

- `speaker.tdnn.N.weight`, matrix `[output_channels, input_channels * context_count]`;
- `speaker.tdnn.N.bias`, vector `[output_channels]`.

Required tensors for every segment layer N:

- `speaker.segment.N.weight`, matrix `[output_channels, input_channels]`;
- `speaker.segment.N.bias`, vector `[output_channels]`.

GGUF matrix dimensions follow GGML order: `ne[0]` is the input-column count and `ne[1]` is the
output-row count. TDNN layers use edge-replicated context frames followed by affine + ReLU.
Statistics pooling concatenates the population mean and `sqrt(variance + 1e-7)`. Segment layers use
affine + ReLU except for the final layer, and the final vector is L2-normalized.

The audio frontend uses the runner's deterministic Slaney log-Mel pipeline: channel averaging,
model-rate resampling, reflection padding, periodic Hann window, power spectrum, Slaney filters,
`log10`, eight-decade dynamic-floor normalization, and per-window/per-Mel mean centering. A model
converted from another speaker stack must match this frontend and the TDNN padding/pooling contract;
renaming arbitrary ECAPA, WeSpeaker, or ONNX tensors is not sufficient.

F32, F16, BF16, and the quantized matrix types already supported by the runner can be used for
weights when their GGML row-alignment requirements are satisfied. Bias tensors are dequantized at
load time.
