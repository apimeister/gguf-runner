# Image Scaling

gguf-runner automatically selects image resolution based on the loaded model and its mmproj sidecar.
The goal is to balance token budget, visual detail, and the spatial resolution the vision encoder
was trained at.

## Resolution source

When a mmproj sidecar is present the vision encoder exposes two values read directly from the
mmproj GGUF metadata:

| Key | Meaning | Typical value |
|---|---|---|
| `clip.vision.image_size` | Base training resolution (`base_size`) | 768 px |
| `clip.vision.patch_size × spatial_merge_size` | Alignment unit (`align_to`) | 28 px (14 × 2) |

Aligned fit-within dimensions are rounded down to a multiple of `align_to`, with a minimum of
one alignment unit. An extremely short edge uses that minimum instead of expanding to the full
target size. The profile is selected after sidecar initialization, including on the first request
through `EmbeddedRuntime`.

## Qwen3.5 — dimension-scaled resolution

Qwen3.5 multimodal models use a dynamic formula that grows the input resolution with the
language-model embedding dimension (`dim`), because larger models can usefully attend to more
visual detail without running out of context.

```
balanced_size = base_size × dim / 3072
balanced_size = clamp(balanced_size, max(align_to, 224), base_size × 2)
target        = floor(balanced_size / align_to) × align_to
```

The anchor point `3072` is the embedding dimension of a 7B-class model, which receives exactly
`base_size` pixels. Smaller models get proportionally less; larger models get proportionally more,
up to 2 × base_size.

### Concrete examples (base_size = 768, align_to = 28)

| Model size | dim  | Raw formula | Aligned target |
|---|---|---|---|
| 2B | 2048 | 768 × 2048 / 3072 = 512 | **504 px** |
| 7B (anchor) | 3072 | 768 × 3072 / 3072 = 768 | **756 px** |
| 14B | 5120 | 768 × 5120 / 3072 = 1280 | **1260 px** |
| ≥ 19B | ≥ 6144 | ≥ 1536 — hits 2 × cap | **1512 px** |

The 2 × cap (1536 px with base_size = 768) is where bilinear interpolation of the vision
encoder's learned position embeddings is still reliable.

The resize mode is **FitWithin**: the image is scaled to fit the target without cropping, then
each edge is aligned to the patch-merge unit. Alignment can change the aspect ratio slightly;
for extreme shapes the short edge is clamped to one unit. For example, a 4096 × 64 image with
target 512 and alignment 32 becomes 512 × 32.

## Qwen3-VL — fixed base resolution

Qwen3-VL models always use exactly `base_size` (768 px) with a **CenterCrop** resize mode.
The image is first scaled so the shorter edge equals the target, then center-cropped to a square.

## Gemma3 — fixed SigLIP resolution

Gemma3 multimodal models use the mmproj encoder `base_size` (typically 896 px) with a
**Stretch** resize mode, matching llama.cpp Gemma3 preprocessing: direct bilinear resize to
`base_size × base_size` without aspect-preserving fit or center-crop.

The [Gemma3 pan-and-scan implementation plan](gemma3-pan-and-scan-plan.md) describes
proposed cropped-view support, prerequisite correctness work, reference material,
and validation gates for later execution.

## Detail crop (Qwen3.5 small models only)

When `GGUF_QWEN35_DETAIL_CROP=1` is enabled, Qwen3.5 models with 24 or fewer transformer layers
(e.g. the 2B variant) append a second image to eligible single-image requests. This option is off
by default. The secondary image is a center-square
crop of the original, giving the model a higher-detail zoomed-in view of the central region —
useful when fine print, logos, or other small details appear near the center.

The crop is skipped when:
- The model has more than 24 layers (larger Qwen3.5 variants).
- The input already contains more than one image, a video, or audio.
- The source image is already square, or its shorter side is less than 64 px.

The temporary crop file is written to the system temp directory and referenced by the prompt
with the caption `(Second image: centered close-up crop of the same source.)`.

## Fallback (no mmproj sidecar)

The unloaded fallback profile is 224 × 224 for Qwen models and 896 × 896 with **Stretch** for
Gemma3. Native image requests require a compatible encoder and select its actual profile after
initialization; an unavailable sidecar produces an error.

## Image positions and correctness checks

Qwen3-VL and Qwen3.5 carry the processed image's merged T/H/W grid into language-model M-RoPE.
Image embeddings use spatial coordinates; subsequent text starts at the largest image coordinate
plus one. Physical KV-cache indices remain sequential. Their vision encoder independently uses
axial RoPE with half the attention head dimension per spatial axis.

Regression fixtures cover the formulas and position layouts in pinned
[Transformers v5.2.0 Qwen3-VL](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
and [v5.3.0 Qwen3.5](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
rectangular/multiple image grids, partial rotary dimensions, chunked versus sequential prefill,
and continuation into text decode. These are focused correctness checks, not full external-model
logit parity or an OCR quality evaluation.

With `Qwen3.5-2B-Q4_K_M.gguf`, its F16 mmproj sidecar, and `regression/IMG_0138.jpg` in the
checkout, run the model-backed lazy/warm/preencoded request regression with:

```bash
cargo test --release --lib lazy_image_requests_match_warm_and_preencoded_requests -- --ignored
```

This compares fresh, warm, and preencoded requests within one runtime mode. For a sequential
comparison, run with `GGUF_BATCH_PREFILL=0`; use `GGUF_BATCH_PREFILL_EXACT=1` to compare the
batched kernels that mirror sequential arithmetic. The default fast K-quant kernel changes
floating-point accumulation and can change generated text. In the local 2B/F16 regression,
sequential and exact batching both returned `yes`; default fast batching returned a longer
answer with the same conclusion. All three modes used 512 × 384 pixels, 192 image tokens, and
identical fresh/warm/preencoded responses within each mode. This observation does not establish
general output-quality equivalence between kernels.
