# Face Recognition Plan (YuNet + SFace)

This document is a handoff plan for adding face embedding to `gguf-runner`: given an image, return
one vector per detected face that can be compared across images to tell whether two faces belong to
the same person.

It captures:

- the goal and what is deliberately left out
- what was verified locally and what is still assumed
- the implementation phases, step by step
- the expected file touch points
- open questions that need a decision before or during implementation

## Current Conclusion

Person identity cannot come from the existing vision-language path. It needs two small dedicated
models, loaded without any language model, in the same way the speaker x-vector model is:

1. A face **detector** that returns a box, a score, and five landmarks per face (YuNet).
2. A face **embedding** model that turns an aligned 112x112 crop into a vector (SFace, 128-dim).

No model returns a person identifier directly. The identifier is derived by comparing vectors
(cosine similarity) against other vectors or against an index. This plan covers producing the
vectors; the index is a later, optional phase.

## What We Verified

Tested with `MiniCPM-V-4_6-Thinking-Q6_K.gguf` on four images from the parent directory, downscaled
to 1024 px, `--think no --temperature 0`:

- It counts foreground people correctly and answers "no faces" correctly on images without people.
- Its boxes cover bodies or head-and-shoulders, not faces, and it misses background faces.
- It takes 8-13 seconds per image and the output format is unstable.

So the VLM is not a usable detector for cropping, and it has no per-person output at all.

Verified in the repository:

- The speaker pipeline (`vendors/speaker_xvector.rs`, `engine/speaker/`, `app/speaker.rs`,
  `app/speaker_index.rs`) is a working template for a dedicated, language-model-free GGUF.
- The only 2D convolution in the engine is the specialised stride-2 GELU one in
  `engine/multimodal/qwen3_asr.rs`; a generic one does not exist.
- `engine/vision/groups.rs` already decodes images with size limits and reads EXIF orientation.
- `onnx` is not installed for the system Python; `uv` is available.

## Upstream Files (checked 2026-11-14)

Hugging Face repos `opencv/face_detection_yunet` and `opencv/face_recognition_sface`:

| File | Size | Precision |
|---|---:|---|
| `face_detection_yunet_2023mar.onnx` | 232,589 bytes | float32 |
| `face_detection_yunet_2023mar_int8.onnx` | 100,416 bytes | int8, fully quantized |
| `face_detection_yunet_2023mar_int8bq.onnx` | 122,489 bytes | int8, block-quantized weights |
| `face_recognition_sface_2021dec.onnx` | 38,696,353 bytes | float32 |
| `face_recognition_sface_2021dec_int8.onnx` | 9,896,933 bytes | int8, fully quantized |
| `face_recognition_sface_2021dec_int8bq.onnx` | 10,667,852 bytes | int8, block-quantized weights |

- Licences per the repos' `LICENSE` files and READMEs: YuNet is MIT, SFace is Apache-2.0.
- Upstream accuracy for SFace: 0.9940 float32, 0.9942 block-quantized, 0.9932 fully quantized.
- Upstream accuracy for YuNet (easy/medium/hard AP): 0.8844/0.8656/0.7503 float32, essentially
  unchanged block-quantized, 0.8810/0.8629/0.7503 fully quantized.
- No f16 files are published.

## Confirmed in Phase 0

From the inspected ONNX graphs:

- YuNet: 53 convolutions (regular and depthwise, batch norm already folded), ReLU, four 2x2
  max-pools, two nearest 2x upsamples with adds, and per-stride class, objectness, box, and
  landmark heads for strides 8, 16, 32. The published graph has a fixed 640x640 input but is fully
  convolutional. Input is BGR 0-255 without normalisation.
- SFace: a plain MobileNet-v1 stack, not residual: 27 convolution + batch norm + PReLU blocks, a
  batch norm, a 50176-to-128 linear layer, and a final batch norm. Input is an aligned 112x112 RGB
  crop, normalised inside the graph as `(x - 127.5) / 128`. Output is 128-dim.
- Alignment uses OpenCV's five reference points for 112x112 crops and a similarity transform.
- OpenCV's cosine match threshold for SFace is 0.363.

## Scope

In scope:

- Converting both models to GGUF with a documented contract.
- Detecting faces, aligning them, and producing L2-normalised vectors.
- A CLI action and a library facade that return one record per face.
- A compact stored form of the vector (128 bytes, int8).

Out of scope for the first version:

- Persistent face index, enrolment, identification, clustering (Phase 6, optional).
- Writing vectors or regions into image metadata; that belongs to the calling application.
- Video input, liveness detection, age/emotion attributes.

## Pipeline

```text
image file
  -> decode, apply EXIF orientation            (engine/vision, existing `ApplyExif`)
  -> resize/pad to detector input              (engine/face/detect)
  -> YuNet forward, decode, suppression        (engine/face/detect)
  -> per face: similarity transform + warp     (engine/face/align)
  -> SFace forward, L2 normalise               (engine/face/embed)
  -> JSON record per face                      (app/face)
```

## Implementation Phases

### Phase 0 - Model inspection, conversion, reference fixtures - DONE

1. Download the YuNet and SFace ONNX files and record their source URLs, hashes, and licences.
2. Dump both graphs: operator list, tensor names and shapes, input/output names.
3. Write `scripts/convert-face-onnx.py` (run via `uv run --with onnx --with numpy`):
   - fold batch norm into the preceding convolution;
   - write tensors under stable names, plus the metadata listed under "GGUF Contracts".
4. With ONNX Runtime, produce reference fixtures for a few images: detector raw outputs, decoded
   faces, aligned crops, final vectors.

Exit: two GGUF files and fixtures exist; every item under "Still Assumed" is confirmed or corrected.

### Phase 1 - Convolution kernels - DONE

1. Add `src/engine/face/conv.rs`: 2D convolution on CHW float tensors with kernel, stride, padding
   and groups; ReLU, PReLU, max-pool, nearest upsample, residual add, fully connected.
2. Parallelise over output channels with rayon.
3. Unit tests against hand-computed values, including depthwise and stride-2 cases.

Exit: kernels reproduce single-layer ONNX outputs from the fixtures.

### Phase 2 - Detector - DONE

1. Add `src/vendors/face_yunet.rs` and `face_detector_policy(gguf)` in `vendors/mod.rs`.
2. Add `src/engine/face/detect.rs`: preprocessing, forward pass, decoding of the three scales,
   non-maximum suppression, mapping back to upright source pixel coordinates.
3. Tests: decoded boxes and landmarks match the fixtures within a small tolerance.

Exit: `examples/face_dump.rs` prints the same faces as the reference for the sample images.

### Phase 3 - Alignment and embedding - DONE

1. Add `src/engine/face/align.rs`: least-squares similarity transform from five landmarks to the
   reference points, bilinear warp to 112x112.
2. Add `src/vendors/face_sface.rs` and `face_encoder_policy(gguf)`.
3. Add `src/engine/face/embed.rs`: forward pass and L2 normalisation.
4. Tests: aligned crops match the fixtures closely; vectors reach cosine similarity above 0.999 to
   the ONNX reference.

Exit: vectors for the sample images match the reference.

### Phase 4 - App, CLI, library - DONE

1. Add `src/app/face.rs` with public `FaceImage`, `FaceDetector`, and `FaceEncoder` (each model
   with `load_from_file` and `load_from_bytes`), the separate `detect` and `embed` calls, the
   combined `embed_all`, and result types.
2. Add CLI flags in `src/cli.rs`: `--face-detector`, `--face-model`, `--face-detect` (detector
   only), `--face-embed` (both).
3. Route face actions in `src/app/mod.rs` before language-model construction.
4. Export the facade and result types from `src/lib.rs`.
5. Output one JSON line per image holding its upright dimensions and a list of faces: box,
   landmarks, detector score, aligned flag, float vector, 128-byte int8 vector (base64).

Exit: `gguf-runner --face-detector ... --face-model ... --face-embed --image a.jpg` works.

### Phase 5 - Quality gates, validation, documentation - DONE

1. No faces are dropped: confirm every record carries the detector score, face size, and the
   aligned/not-aligned flag so the caller can filter.
2. Sanity checks on real photos: different people in one image give clearly different vectors; the
   same person in two images gives clearly similar ones.
3. Measure the accuracy cost of the int8 form against float on the same pairs.
4. Measure speed per image and per face.
5. Update `docs/module-structure.md`; add `docs/face-recognition.md` with usage and both GGUF
   contracts.
6. Run the validation commands required by `AGENTS.md`.

### Phase 6 - Index and identification (deferred, not planned)

A `.faceidx` file with enrolment, verification, identification, and clustering of unknown faces,
modelled on `app/speaker_index.rs`. Decided against for now: the runner outputs vectors only.

## GGUF Contracts (draft)

Detector, `general.architecture = face_yunet`:

- input width/height or size constraints, colour order, scale/mean
- strides, score threshold, suppression threshold, maximum detections
- convolution tensors in forward order

Encoder, `general.architecture = face_sface`:

- input size, colour order, scale/mean
- the five alignment reference points
- embedding dimension
- match threshold(s)
- convolution, PReLU, and fully connected tensors in forward order

Final key and tensor names are fixed in Phase 0, once the graphs are known.

## File Touch Points

New:

- `scripts/convert-face-onnx.py`
- `src/engine/face/{mod,conv,detect,align,embed}.rs`
- `src/vendors/face_yunet.rs`, `src/vendors/face_sface.rs`
- `src/app/face.rs`
- `examples/face_dump.rs`
- `docs/face-recognition.md`

Changed:

- `src/engine/mod.rs`, `src/vendors/mod.rs`, `src/app/mod.rs`
- `src/cli.rs`, `src/lib.rs`
- `docs/module-structure.md`

## Decisions

1. **Where does identity live?** The runner outputs vectors only. No index, enrolment,
   identification, or clustering for now; the calling application compares vectors. Phase 6 stays
   deferred.

2. **One GGUF or two?** Two files and two runtime types: `FaceDetector` loads the YuNet GGUF,
   `FaceEncoder` loads the SFace GGUF. Detection and embedding are separate calls that share one
   decoded image, so the source is decoded once:

   ```rust
   let image = FaceImage::open(path)?;             // decode once, EXIF applied
   let faces = detector.detect(&image)?;           // boxes, scores, landmarks
   let vector = encoder.embed(&image, &faces[0])?; // one face -> one vector
   ```

   - A combined convenience call (`embed_all(path)`) sits on top and backs the CLI action.
   - Detection works without loading the encoder.
   - Stored vectors depend on the encoder file alone, so a detector update does not invalidate
     them.
   - A face given as a box without landmarks (for example drawn by hand in the calling application)
     gets landmarks by re-running the detector inside the box; if none are found, the encoder falls
     back to a plain crop and flags the vector as not aligned.

3. **Model shipping.**
   - The `gguf-runner` binary always loads both face models from disk (`--face-detector`,
     `--face-model`).
   - Library consumers hand the models over exactly like the other GGUF models:
     `load_from_bytes(data: &'static [u8])` and `load_from_file(path)` on `FaceDetector` and
     `FaceEncoder`, matching `EmbeddedRuntime`, `load_mmproj_from_bytes`, and `SpeakerRuntime`.
   - The converter script stays in the repo so the files are reproducible. The converted GGUFs are
     published outside the git repo and linked from `docs/downloading-models.md`, provided the
     licence check in Phase 0 allows redistribution.

4. **Weight precision.** float32 for both models, as published upstream (about 38.7 MB for SFace).
   No precision conversion in the first version, so the GGUF weights are bit-identical to the
   reference and no f16 comparison is needed in Phase 0. f16 (about 19 MB) or upstream's
   block-quantized int8 variant (10.7 MB, same reported accuracy) remain possible later size
   reductions; either would count as a new model for stored vectors.

5. **Vector output format.** Every face record carries the L2-normalised float vector and a compact
   form: each value multiplied by 127, rounded, stored as one signed byte (128 bytes, base64 in
   JSON). The runner attaches no model tag to records, because the caller chose the model file.
   `FaceEncoder::model_name()` exposes the GGUF's `general.name` for logging. Labelling stored
   vectors so that different models are never compared is the calling application's concern.

9. **Licences.** Checked: YuNet is MIT and SFace is Apache-2.0 (see "Upstream Files"). Converted
   files can be redistributed with the licence texts attached.

6. **Coordinate space.** The face path decodes with `ImageOrientationPolicy::ApplyExif`, because
   the detector misses faces in images stored sideways. Boxes and landmarks are float pixel
   coordinates in that upright image; every record carries the upright width and height.
7. **Detector input size.** The image is shrunk so its longest side is at most 640 (never
   enlarged) and padded right/bottom to a multiple of 32. Feeding larger inputs finds more small
   background faces but loses faces that fill much of the frame, so 640 stays the single default.
8. **Quality gate.** The runner drops nothing: every detected face gets a vector, and its record
   carries the detector score and an aligned/not-aligned flag. Filtering is the caller's job.
10. **Input formats.** PNG, JPEG, and WebP, as supported by the runner today. No HEIC.
11. **Batching.** `--image` already accepts several paths; output is one JSON line per image with
    a `faces` list, so an image without faces is still reported. No manifest mode.
12. **Match thresholds.** Out of scope, since the runner does not compare vectors.
    `docs/face-recognition.md` mentions the published SFace value for reference.

## Implementation Notes

Deviations from the phase descriptions above, and why:

- **Operation-list contract.** Instead of hard-coding the two architectures, the converter writes
  each network as a flat operation list (`<prefix>.graph.ops`) that a small generic executor in
  `engine/face/graph.rs` runs. Both models share it, and the vendor modules only describe heads,
  preprocessing, and thresholds. `vendors/face_common.rs` holds the shared metadata readers.
- **Batch norm folding.** YuNet ships folded. For SFace the converter folds each batch norm into
  its convolution, and the two around the linear layer into that layer. Input normalisation is
  kept as metadata because zero padding applies after it.
- **Output shape.** One JSON line per image rather than per face (decision 11).
- **Model fingerprint.** `model_fingerprint()` is exposed next to `model_name()`, as for the
  speaker runtime.
- **Extra entry points.** `FaceImage::from_encoded_bytes`/`from_rgb8`,
  `FaceEncoder::embed_aligned_crop`, `FaceEmbedding::cosine_similarity`, and
  `--face-score-threshold`.

## Results

Reference: OpenCV 5.0.0 `FaceDetectorYN`/`FaceRecognizerSF` on the original ONNX files.

- Detector on identical 640x480 inputs: boxes, landmarks, scores equal within 0.001 px.
- Encoder on identical aligned crops: cosine 1.000000, largest component difference about 2e-6.
- `IMG_0369-preview.jpg`: 2 faces. `IMG_0455.png`: 4 faces. A photo without people: 0 faces.
- End-to-end vectors versus the reference: cosine 0.95 to 0.999; the gap comes from the
  downscale filter in front of the detector moving landmarks by a few source pixels.
- int8 form versus float: cosine differs by at most 0.006 over 15 pairs.
- Time including process start and model load: about 15 ms detection at 640x480, about 25 ms per
  embedded face, 150 to 250 ms for 12 to 18-megapixel files.

### No shared face between the two test photos

No face of `IMG_0369-preview.jpg` matches a face of `IMG_0455.png`. The eight cross-image
similarities range from -0.14 to 0.25; the published same-person threshold is 0.363. OpenCV's
reference pipeline gives the same picture (highest 0.26).

Checked before accepting that result:

- Detector input 1280 and 2560 instead of 640: finds up to two more background faces in
  `IMG_0455.png` and one in `IMG_0369-preview.jpg`; none match across the photos. One pair reaches
  0.376 at 1280, but only because the large left face of `IMG_0369-preview.jpg` gets visibly
  tilted landmarks at that size; with the well-aligned 640 landmarks the same pair is 0.25.
- Antialiased alignment (shrinking the source before warping large faces): changes similarities
  by less than 0.02.
- Exhaustive search in the reference pipeline: detection over overlapping 640-pixel tiles at four
  scales with the score threshold lowered to 0.8, which yields 9 and 8 candidate faces including
  small background ones and some partial detections. The highest cross-image similarity is 0.35,
  again for the pair below; every other pair is at most 0.28.
- For scale: the two clearly different foreground faces within `IMG_0369-preview.jpg` score 0.28
  against each other, so values around 0.25 to 0.35 are not evidence of the same person.

The highest-scoring pair is consistently the left face of `IMG_0369-preview.jpg` and the face at
x=2089 in `IMG_0455.png`.

- Diagnostic with a stronger encoder (ArcFace ResNet-50, research-only weights, not shipped) on
  the same aligned crops: the highest cross-photo pair scores 0.15, against 0.12 for known
  different people in the same photo. It confirms the two photos share no person.

### Shared faces with two further photos

Adding `IMG_0263-preview.jpg` (4 faces) and `3c5fd91c-b4a4-41a5-911e-8e1195320d73-preview.jpg`
(3 faces) gives 13 faces and 62 cross-image pairs:

| Pair | Similarity |
|---|---:|
| `IMG_0369` face 0 and `IMG_0263` face 0 | 0.718 |
| `IMG_0369` face 0 and `3c5fd91c` face 0 | 0.757 |
| `IMG_0263` face 0 and `3c5fd91c` face 0 | 0.828 |
| `IMG_0369` face 1 and `3c5fd91c` face 2 | 0.669 |
| highest of the other 58 pairs (`IMG_0455` face 0 and `3c5fd91c` face 0) | 0.327 |

The first three rows are one person seen in three photos and agree with each other. Matches
(0.67 to 0.83) and non-matches (at most 0.33) are clearly separated, though the strongest
non-match sits close to the published threshold of 0.363. `IMG_0455.png` shares no face with any
of the other three photos.

### Library embedding mode

`examples/face_embedded_test.rs` on the four test photos: models loaded from static bytes and
images passed as in-memory bytes give results identical to file-based loading; four threads with
separate rayon pools sharing one loaded detector and encoder give results identical to the serial
run; similarity computed from the stored signed bytes stays within 0.011 of the float value over
78 pairs. `FaceDetector`, `FaceEncoder`, and `FaceImage` are `Send + Sync`.

## Open Questions

- Where the converted GGUFs are hosted (decision 3).
