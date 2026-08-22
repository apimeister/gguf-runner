# Performance Notes (Historical)

This document summarizes benchmark snippets that previously lived in `README.md` and `test.md`.

These numbers are useful as historical reference points, not as strict apples-to-apples benchmarks across machines.

## Benchmark Table Template

Use this section as a starting point for structured performance collection.

### Host Profiles

| host_id | os | cpu | cores | memory gb | notes |
|---|---|---|---|---:|---|
| mac-m2-24g | macOS 15.3 | Apple M2 | 8 | 24 | laptop |
| mac-m4-32g | macOS 15.3 | Apple M4 | 10 | 32 | laptop |
| mac-m5-32g | macOS 26.5 | Apple M5 | 10 | 32 | laptop |
| lnx-n150-12g | Gentoo Linux | Intel N150 | 4 | 12 | Beelink ME mini |
| lnx-1340p-32g | Fedora 14 | Intel i5-1340P | 16 | 32 | Framework 13 |
| lnx-13600k-8g | Ubuntu 24.04 | Intel i5-13600K | 20 | 8 | |
| lnx-125h-32g | Gentoo Linux | Intel Ultra 125h | 18 | 32 | Minisforum M1 Pro-125H |
| lnx-9700-64g | Ubuntu 24.04 | AMD Ryzen 7 PRO 8700GE | 16 | 64 | Hetzner AX42 |

### Prompts

#### png_to_jpeg_v1
  "Can you write me a programm in Rust that can convert PNG images to JPEG"

```bash
gguf-runner --model Qwen3-4B-Instruct-2507-Q4_K_M.gguf --prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" --temperature=0 --show-tokens --show-timings
```

#### image_v1
  "Describe the content of this image."

```bash
gguf-runner --model ./Qwen3.5-2B-Q4_K_M.gguf --image ./regression/IMG_0138.jpg --prompt 'Describe the content of this image.' --show-tokens --show-timings
```

### Benchmark Runs - Current Models

| date | model | host_id | prompts | tokens/sec | runtime sec | notes |
|---|---|---|---|---:|---:|---|
| 2026-02-15 | gemma-3-4b-it-Q4_K_M.gguf | lnx-13600k-8g | png_to_jpeg_v1 | 3.106 | 317.936 | |
| 2026-02-15 | gemma-3-4b-it-Q4_K_M.gguf | lnx-1340p-32g | png_to_jpeg_v1 | 3.522 | 275.898 | |
| 2026-03-07 | gemma-3-4b-it-Q4_K_M.gguf | mac-m2-24g | png_to_jpeg_v1 | 5.483 | 186.410 | |
| 2026-03-08 | gemma-3-4b-it-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 6.402 | 117.833 | |
| 2026-02-15 | gemma-3-4b-it-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 6.894 | 147.734 | |
| 2026-02-15 | gemma-3-4b-it-Q4_K_M.gguf | mac-m4-32g | image_v1 | 7.469 | 136.642 | |
| 2026-02-15 | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 2.770 | 135.304 | |
| 2026-02-15 | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf | lnx-13600k-8g | png_to_jpeg_v1 | 3.109 | 124.928 |
| 2026-02-15 | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf | lnx-1340p-32g | png_to_jpeg_v1 | 3.292 | 111.207 | |
| 2026-03-08 | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 4.731 | 88.306 | |
| 2026-02-15 | Qwen3-Coder-Next-Q4_K_M.gguf | lnx-n150-12g | png_to_jpeg_v1 | 0.409 | 2240.847 | |
| 2026-02-15 | Qwen3-Coder-Next-Q4_K_M.gguf | lnx-125h-32g | png_to_jpeg_v1 | 2.228 | 369.767 | |
| 2026-03-08 | Qwen3-Coder-Next-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 4.981 | 253.543 | |
| 2026-03-08 | Qwen3-Coder-Next-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 6.848 | 178.041 | |
| 2026-02-16 | Qwen3-235B-A22B-Instruct-2507-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 0.652 | 1125.015 | |
| 2026-03-07 | Qwen3.5-0.8B-Q4_K_M.gguf | lnx-n150-12g | png_to_jpeg_v1 | 4.456 | 110.072 | |
| 2026-03-07 | Qwen3.5-0.8B-Q4_K_M.gguf | mac-m2-24g | png_to_jpeg_v1 | 22.116 | 22.764 | |
| 2026-03-07 | Qwen3.5-0.8B-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 22.156 | 101.068 | |
| 2026-03-07 | Qwen3.5-0.8B-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 34.652 | 14.941 | |
| 2026-07-11 | Qwen3.5-0.8B-Q4_K_M.gguf | mac-m5-local | png_to_jpeg_v1 | 51.508 | 4.467 | local release build |
| 2026-03-07 | Qwen3.5-2B-Q4_K_M.gguf | lnx-n150-12g | png_to_jpeg_v1 | 1.936 | 239.441 | |
| 2026-03-08 | Qwen3.5-2B-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 10.333 | 163.418 | |
| 2026-03-07 | Qwen3.5-2B-Q4_K_M.gguf | mac-m2-24g | png_to_jpeg_v1 | 10.774 | 47.773 | |
| 2026-03-07 | Qwen3.5-2B-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 15.915 | 33.569 | |
| 2026-03-07 | Qwen3.5-2B-Q4_K_M.gguf | mac-m4-32g | image_v1 | 16.418 | 62.619 | |
| 2026-07-11 | Qwen3.5-2B-Q4_K_M.gguf | mac-m5-local | png_to_jpeg_v1 | 28.079 | 7.472 | local release build |
| 2026-07-11 | Qwen3.5-2B-Q4_K_M.gguf | mac-m5-local | image_v1 | 22.608 | 23.738 | local release build |
| 2026-03-11 | Qwen3.5-35B-A3B-Q4_K_M.gguf | mac-m4-32g | image_v1 | 7.210 | 103.316 | |


### Benchmark Runs - Older Models

| date | model | host_id | prompts | tokens/sec | runtime sec | notes |
|---|---|---|---|---:|---:|---|
| 2026-02-15 | Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 1.251 | 421.389 | |
| 2026-02-15 | Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf | lnx-1340p-32g | png_to_jpeg_v1 | 1.798 | 289.223 | |
| 2026-02-15 | Qwen3-0.6B-Q4_K_M.gguf | lnx-n150-12g | png_to_jpeg_v1 | 6.236 | 179.751 | |
| 2026-02-15 | Qwen3-0.6B-Q4_K_M.gguf | lnx-1340p-32g | png_to_jpeg_v1 | 11.510 | 97.513 | |
| 2026-02-16 | Qwen3-0.6B-Q4_K_M.gguf | lnx-125h-32g | png_to_jpeg_v1 | 15.763 | 54.392 | |
| 2026-02-15 | Qwen3-0.6B-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 24.575 | 46.232 | |
| 2026-02-15 | Qwen3-0.6B-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 27.721 | 41.037 | |
| 2026-02-15 | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | lnx-n150-12g | png_to_jpeg_v1 | 1.607 | 528.286 | |
| 2026-02-15 | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | lnx-13600k-8g | png_to_jpeg_v1 | 3.836 | 221.583 | |
| 2026-02-15 | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | lnx-1340p-32g | png_to_jpeg_v1 | 4.237 | 200.740 | |
| 2026-02-15 | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | mac-m4-32g | png_to_jpeg_v1 | 4.881 | 175.791 | |
| 2026-02-15 | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | lnx-125h-32g | png_to_jpeg_v1 | 5.020 | 169.513 | |
| 2026-02-15 | Qwen3-4B-Instruct-2507-Q4_K_M.gguf | lnx-9700-64g | png_to_jpeg_v1 | 6.462 | 132.128 | |
| 2026-02-15 | Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf | lnx-n150-12g | png_to_jpeg_v1 | 1.602 | 609.450 | |
| 2026-02-15 | Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf | mac-m4-32g | png_to_jpeg_v1 | 3.625 | 268.448 | |
| 2026-02-15 | Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf | lnx-125h-32g | png_to_jpeg_v1 | 5.010 | 256.944 | |
| 2026-02-15 | Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf | lnx-9700-64g | png_to_jpeg_v1 | 7.287 | 154.820 | |
| 2026-03-11 | Qwen3-VL-2B-Instruct-Q4_K_M.gguf | mac-m4-32g | image_v1 | 15.784 | 71.829 | |
| 2026-03-11 | Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf | mac-m4-32g | image_v1 | 6.952 | 228.771 | |

## Audio Pipeline (2026-08-22, mac-m5-32g)

Host `mac-m5-32g` (Apple M5, 6 performance + 4 efficiency cores). Model: official
`Qwen3-ASR-1.7B-Q8_0.gguf` with its `mmproj` sidecar. Speech generated with `say` at 16 kHz mono.
All model-run figures are min-of-N with the model's page cache pre-warmed.

### Log-Mel front end

`extract_whisper_log_mel_windows` A/B, both implementations built from the same tree and measured
back to back, 15 repetitions, min. Output is byte-identical between the two (verified over 4.25M
feature values).

| audio | serial recursive FFT | parallel table FFT | speedup |
|---|---:|---:|---:|
| 1 s | 2.11 ms | 0.79 ms | 2.7x |
| 10 s | 11.28 ms | 1.65 ms | 6.8x |
| 30 s | 34.21 ms | 4.28 ms | 8.0x |
| 300 s | 346.4 ms | 40.25 ms | 8.6x |

Short clips gain least because 101 frames cannot fill ten cores. The pre-change profile was 58.8%
`fft_real`, 30.7% mel projection, 6.8% allocator; after the change the front end is ~0.03% of
audio-dependent work and no longer worth optimizing.

### Where transcription time goes

Phase split, measured by differencing a `--max-tokens 1` run against a full run:

| audio | prefill | decode |
|---|---:|---:|
| 19.1 s | 78% | 22% |
| 57.2 s | 79% | 21% |

Sampling profile of a full run: `vec_dot_q8_0_2rows_i8mm_prequant` 112,477 samples (~61%), idle
spin ~38%, audio encoder `erff` 415 (0.2%). The front end and the ASR encoder are both negligible;
the cost is language-model matmul.

### Prefill scales quadratically with audio duration

Qwen3-ASR emits ~13.05 audio embedding tokens per second of audio.

| audio | tokens | prefill | ms/token |
|---|---:|---:|---:|
| 19.1 s | 249 | 6.77 s | 27.2 |
| 57.2 s | 746 | 23.00 s | 30.8 |
| 183.4 s | 2393 | 102.40 s | 42.8 |

These fit `cost_per_token ~= 25.3 + 0.0073 * n` ms (consecutive slopes 0.00734 and 0.00727), so
total prefill is O(n^2). The constant term is weight streaming; the linear term is attention over
the KV cache. They cross near 3,470 tokens, about 4.4 minutes of audio: shorter files are
matmul-bound, longer files are attention-bound. A separately measured 33.9-minute recording
(26,468 tokens, 2004 s) is consistent with the quadratic term dominating at length.

### Batched prefill depends on the weight quantization

Interleaved on/off runs, ~480-token prompt, `--max-tokens 1`:

| model | min | median |
|---|---:|---:|
| Qwen3-ASR-1.7B-Q8_0 | 1.10x | 1.15x |
| Qwen3.5-2B-Q4_K_M | 1.50x | 1.46x |

Q8_0 appears in neither `batch_fast_supported` (Q2_K..Q6_K) nor `batch_exact_supported`
(Q4_0/Q4_1/Q5_0/Q5_1/K-quants/IQ4_NL), so `bmm_prefill` falls through to its per-token loop and only
cache locality remains. A K-quant build of the same model is the cheapest way to get the larger
win; a batched Q8_0 kernel is the open work item.

### Threads

30 s audio, warm cache:

| threads | real | user |
|---|---:|---:|
| 4 | 17.59 s | 61.97 s |
| 6 | 16.42 s | 78.48 s |
| 8 | 15.64 s | 92.83 s |
| 10 | 15.65 s | 103.15 s |

`--threads 8` matches the default on wall clock while using ~11% less CPU.

### Operational notes

- The KV cache is sized from the model's `seq_len`, not the prompt, and is allocated per generation
  call. For this model at the default 65,536 context that is roughly 1.9 GB per call, re-done for
  every `--audio-batch` record. `--context-size` caps it; a 30-second segment needs about 610
  tokens.
- Measurements on a thermally saturated machine are unusable: an identical batch command measured
  105.5 s and then 78.2 s (+/-30%). Interleave the variants being compared, take min-of-N, and let
  the machine cool between long runs.
- Highly repetitive synthetic speech is a poor benchmark input: it trips the repetition guards and
  truncates transcripts, which silently changes how much decode work a run does.

## Benchmark Caveats

- Results come from different dates, machines, and code revisions.
- Some runs include profiling or debug behavior that affects runtime.

## Legacy README Snapshots

### Llama 3 8B prompt run progression

Prompt: `Tell me in 1 line what is Microsoft.`

| Variant | Reported wall time |
|---|---:|
| C version (`llama3pure`) | 2:41.17 |
| Rust (early baseline) | 4:48.39 |
| Rust + SIMD | 2:02.36 |
| Rust + Rayon | 15.553s |
| Rust + Rayon + `RUSTFLAGS="-C target-cpu=native"` | 14.758s |

### Legacy comparison: `llama-cli` vs `llama3pure`

Same Qwen3-Coder-Next prompt workload (`/usr/bin/time -l`):

| Tool | real | user | sys | max RSS |
|---|---:|---:|---:|---:|
| `llama-cli` | 840.84s | 723.82s | 271.35s | 23,993,057,280 |
| `llama3pure` | 402.09s | 1471.08s | 615.37s | 24,622,071,808 |

## `test.md` Optimization Timeline (2026-02-10)

Workload used repeatedly:

```bash
/usr/bin/time -l ./target/release/llama3pure -model Qwen3-Coder-Next-Q4_K_M.gguf \
  -prompt "Can you write me a programm in Rust that can convert PNG images to JPEG" \
  -max_tokens 50000 -context_size 250000
```

| Label in notes | real | user | sys | max RSS |
|---|---:|---:|---:|---:|
| Baseline reference | 402.09s | 1471.08s | 615.37s | 24,622,071,808 |
| `updates (2026-02-10)` | 329.40s | 863.52s | 503.91s | 17,901,813,760 |
| `deep optimization pass` | 327.78s | 884.33s | 499.50s | 15,154,610,176 |
| `arm kernels + profiling` | 505.45s | 1881.64s | 565.64s | 14,985,953,280 |
| `full run after matmul 1/2/3` | 427.90s | 1384.12s | 639.31s | 14,742,552,576 |

Notes:
- The profiling-enabled run is expected to be slower.
- Memory footprint trends downward across most optimization passes.

## Reproducibility Guidance

From the original notes:
- keep model, prompt, `max_tokens`, and `context_size` fixed
- use deterministic decoding for comparisons:
  - `-temperature 0 -top_k 1 -top_p 1`
- compare both wall time and token throughput
