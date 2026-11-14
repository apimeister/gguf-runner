# Face Detection and Face Embeddings

`gguf-runner` can find the faces in an image and turn each one into a vector. Two vectors from the
same encoder model can be compared with cosine similarity to judge whether two faces belong to the
same person. All work stays inside the process: image decode, detection, alignment, and embedding.

The runner outputs vectors only. It keeps no index of people, assigns no identifiers, and applies
no match threshold; comparing and grouping vectors is the caller's job.

This is face **association**, not authentication. A photo of a photo matches as well as the
person, and there is no liveness detection. Face vectors are biometric data: treat stored vectors
with the same care as the photos they came from.

## Requirements

Two dedicated GGUF files, neither of which is a language model:

- a detector implementing the `face_yunet` contract (boxes, scores, five landmarks);
- an encoder implementing the `face_sface` contract (aligned 112x112 crop to a 128-dim vector).

Both are produced from the OpenCV model-zoo ONNX files with `scripts/convert-face-onnx.py`; see
[Downloading models](downloading-models.md). Weights stay float32, as published.

Input images are PNG, JPEG, or WebP. EXIF orientation is always applied before detection, and all
reported coordinates refer to that upright image.

## Command line

```bash
# Boxes, landmarks, and scores only; the encoder is not loaded
gguf-runner \
  --face-detector ./face-yunet-2023mar-f32.gguf \
  --face-detect \
  --image ./a.jpg

# One vector per face
gguf-runner \
  --face-detector ./face-yunet-2023mar-f32.gguf \
  --face-model ./face-sface-2021dec-f32.gguf \
  --face-embed \
  --image ./a.jpg --image ./b.png
```

`--face-score-threshold <0..1>` overrides the detector's default minimum score (0.9 for the
converted YuNet).

Output is one JSON line per `--image`, in input order. An image without faces yields an empty
`faces` array, so every input is accounted for:

```json
{"source":"./a.jpg","image_width":4896,"image_height":3672,"faces":[
  {"bbox":{"x":858.6,"y":1185.8,"width":960.6,"height":1231.3},
   "landmarks":[[1150.2,1606.5],[1615.1,1599.5],[1404.7,1847.3],[1175.4,2040.6],[1605.0,2036.8]],
   "score":0.939,
   "aligned":true,
   "vector":[-0.0198,0.1383,0.0587, "... 128 values"],
   "vector_q8":"/RIHB/75BvUC... 172 characters"}]}
```

| Field | Meaning |
|---|---|
| `image_width`, `image_height` | Upright dimensions the coordinates refer to |
| `bbox` | Left, top, width, height in pixels; may extend past the image edge |
| `landmarks` | Right eye, left eye, nose tip, right mouth corner, left mouth corner, from the subject's point of view |
| `score` | Detector confidence, 0..1 |
| `aligned` | `true` when the crop was aligned from landmarks (always, for detector output) |
| `vector` | L2-normalized float embedding |
| `vector_q8` | Base64 of the compact form: each component multiplied by 127 and rounded to a signed byte |

`aligned`, `vector`, and `vector_q8` are present only with `--face-embed`. Faces are ordered by
detector score, best first. The runner drops no detected face; filter on `score` and box size.

## Library

Detection and embedding are separate calls that share one decoded image:

```rust
use gguf_runner::{FaceDetector, FaceEncoder, FaceImage};

let detector = FaceDetector::load_from_file(detector_path)?;   // or load_from_bytes(&'static [u8])
let encoder = FaceEncoder::load_from_file(encoder_path)?;

let image = FaceImage::open(image_path)?;       // decoded once, EXIF applied
let faces = detector.detect(&image)?;           // boxes, landmarks, scores
let a = encoder.embed(&image, &faces[0])?;      // one face, one vector
let b = encoder.embed(&image, &faces[1])?;
let similarity = a.cosine_similarity(&b)?;      // -1..=1
let stored: Vec<i8> = a.quantized();            // 128 bytes
```

- `FaceEncoder::embed_all(&detector, path)` does both steps for one file and returns the same
  structure the CLI prints; `embed_image(&detector, &image)` does the same for a decoded image.
- `FaceDetector` works without an encoder.
- `FaceImage::from_encoded_bytes` and `FaceImage::from_rgb8` accept images that are not files.
- Result types implement `Serialize` and `Deserialize`.
- `model_name()`, `model_architecture()`, and `model_fingerprint()` identify the loaded model.
  Vectors from different encoder models are not comparable; a caller that stores vectors should
  store which encoder produced them.
- `FaceEncoder::published_match_threshold()` returns the publisher's same-person cosine threshold
  (0.363 for SFace). It is a starting point, not something the runner applies.

### Embedding the models in an application

An application that links its models into the binary hands the face models over exactly like
the other GGUFs, as `&'static [u8]`:

```rust
use gguf_runner::{FaceDetector, FaceEmbedding, FaceEncoder, FaceImage};

static DETECTOR: &[u8] = include_bytes!("../models/face-yunet-2023mar-f32.gguf");
static ENCODER: &[u8] = include_bytes!("../models/face-sface-2021dec-f32.gguf");

let detector = FaceDetector::load_from_bytes(DETECTOR)?;
let encoder = FaceEncoder::load_from_bytes(ENCODER)?;

// Images do not have to be files.
let image = FaceImage::from_encoded_bytes(&upload)?;        // PNG, JPEG, or WebP bytes
// let image = FaceImage::from_rgb8(width, height, pixels)?; // pixels decoded elsewhere, upright

for face in encoder.embed_image(&detector, &image)? {
    store(face.bbox, face.vector_q8.unwrap());              // base64 of 128 signed bytes
}

// Later: compare two stored vectors without the models.
let a = FaceEmbedding::decode_quantized_base64(&stored_a)?;
let b = FaceEmbedding::decode_quantized_base64(&stored_b)?;
let similarity = FaceEmbedding::quantized_cosine_similarity(&a, &b)?;
```

- **No setup call.** The face types need no runtime-switch or CLI initialisation and no language
  model.
- **Sharing across threads.** `FaceDetector` and `FaceEncoder` are `Send + Sync` and every call
  takes `&self`: load each once and share it, for example in an `Arc`. Nothing is forked or locked.
- **Thread pools.** Convolutions use the rayon pool the call runs in. Give a caller its own pool
  with `rayon::ThreadPool::install`, as with `EmbeddedRuntime`.
- **Memory.** Weights are expanded into owned floats at load time (about 38 MB for SFace) and the
  GGUF bytes are not read again afterwards. With `load_from_bytes` the static bytes stay part of
  the binary; with `load_from_file` the mapping is released after loading.
- **Other image formats.** For formats the runner does not decode, such as HEIC, decode in the
  application, rotate upright, and use `FaceImage::from_rgb8`.
- **Comparing stored vectors needs no model.** `quantized_cosine_similarity` works on the stored
  bytes alone.

`examples/face_embedded_test.rs` exercises this path: it loads both models from bytes, checks the
results against file-based loading, runs four threads with separate pools against one loaded pair
of models, and checks the stored form against the float vectors.

### Faces known only by a box

A region that did not come from the detector, such as a box drawn by a user, has no landmarks:

```rust
let manual = FaceBox { x, y, width, height };
let face = detector
    .locate(&image, &manual)?                              // detector run on the box plus margin
    .unwrap_or_else(|| DetectedFace::from_box(manual));    // fallback: no landmarks
let embedding = encoder.embed(&image, &face)?;
assert!(embedding.aligned || face.landmarks.is_none());
```

Without landmarks the encoder stretches the plain box to its input size and reports
`aligned = false`. Such vectors match much less reliably and should not be mixed blindly with
aligned ones.

## How it works

1. **Decode.** The image is decoded to RGB8 and rotated according to EXIF.
2. **Detect.** The image is shrunk so its longest side is at most the detector's `max_side`
   (640), padded at the right and bottom to a multiple of 32, and passed through YuNet. Each of the
   three output scales (strides 8, 16, 32) is decoded into boxes and landmarks, followed by
   non-maximum suppression. Coordinates are scaled back to the upright source.
3. **Align.** A least-squares similarity transform maps the five landmarks onto the encoder's
   reference positions; the 112x112 crop is sampled bilinearly from the full-resolution source.
4. **Embed.** SFace maps the crop to 128 values, which are L2-normalized.

Small faces are lost in the downscale to the detector size. In a 12-megapixel test photo,
background faces about 100 pixels wide (about 16 pixels at detector scale) were not found.

## Verification against the reference

Measured against OpenCV 5.0.0 `FaceDetectorYN` and `FaceRecognizerSF` running the original ONNX
files, on two photos with six faces:

- Detector, identical 640x480 input: boxes, landmarks, and scores equal to within 0.001 pixels.
- Encoder, identical aligned crops: cosine similarity 1.000000, largest component difference
  about 2e-6.
- Whole pipeline from the original files: 2 and 4 faces found, as in the reference. Vectors
  have cosine similarity 0.95 to 0.999 to the reference. The remaining difference comes from the
  downscale filter in front of the detector, which moves landmarks by a few source pixels.
- Compact form: cosine similarity computed from the signed bytes differs from the float result by
  at most 0.011 over 78 face pairs from four photos.
- Embedded loading: results from `load_from_bytes` and in-memory images are identical to
  file-based loading, and identical across four concurrent threads sharing one loaded pair of
  models.

Vectors react to landmark jitter: the same face embedded from two slightly different landmark
estimates scored 0.90. All 15 pairs of distinct faces in those photos scored between -0.14 and
0.28, below the published threshold of 0.363.

Wall-clock time on an Apple-silicon laptop, including process start and model load: about 15 ms
to detect in a 640x480 image, about 25 ms per embedded face, and 150 to 250 ms for a 12 to
18-megapixel file, where decoding and downscaling dominate.

## Limits

- Children, the same person decades apart, strong profile views, sunglasses, and small or blurry
  faces produce weaker matches.
- The published threshold was calibrated on a public benchmark, not on your photos.
- Very large faces (most of the frame) get less precise landmarks than mid-sized ones.
- No HEIC input, no video, no batching manifest.

## `face_yunet` GGUF contract

Required metadata:

| Key | Type | Meaning |
|---|---:|---|
| `general.architecture` | string | exactly `face_yunet` |
| `general.name` | string | display name; optional but recommended |
| `face_yunet.graph.ops` | integer array | operation list, see below |
| `face_yunet.graph.slot_count` | integer | number of value slots, including the input |
| `face_yunet.input.channel_order` | string | `rgb` or `bgr` |
| `face_yunet.input.mean` | float | subtracted from each 8-bit sample |
| `face_yunet.input.scale` | float | multiplied after the subtraction |
| `face_yunet.input.max_side` | integer | longest image side fed to the network |
| `face_yunet.input.align` | integer | padded input is a multiple of this; a multiple of every stride |
| `face_yunet.detect.strides` | integer array | one entry per output scale |
| `face_yunet.detect.cls_slots` | integer array | class-logit slot per scale, 1 channel |
| `face_yunet.detect.obj_slots` | integer array | objectness-logit slot per scale, 1 channel |
| `face_yunet.detect.bbox_slots` | integer array | box slot per scale, 4 channels |
| `face_yunet.detect.kps_slots` | integer array | landmark slot per scale, 10 channels |
| `face_yunet.detect.landmark_count` | integer | exactly 5 |
| `face_yunet.detect.score_threshold` | float | default minimum score |
| `face_yunet.detect.nms_threshold` | float | overlap above which the weaker box is dropped |
| `face_yunet.detect.top_k` | integer | candidates kept before suppression |

Decoding for cell `(row, col)` at stride `s`: score is `sqrt(sigmoid(cls) * sigmoid(obj))`; the
box centre is `((col + dx) * s, (row + dy) * s)` with size `(exp(w) * s, exp(h) * s)`; landmark
`n` is `((kx_n + col) * s, (ky_n + row) * s)`.

## `face_sface` GGUF contract

Required metadata:

| Key | Type | Meaning |
|---|---:|---|
| `general.architecture` | string | exactly `face_sface` |
| `general.name` | string | display name; optional but recommended |
| `face_sface.graph.ops` | integer array | operation list, see below |
| `face_sface.graph.slot_count` | integer | number of value slots, including the input |
| `face_sface.input.channel_order` | string | `rgb` or `bgr` |
| `face_sface.input.mean` | float | subtracted from each 8-bit sample |
| `face_sface.input.scale` | float | multiplied after the subtraction |
| `face_sface.input.size` | integer | side of the square aligned crop |
| `face_sface.align.reference` | float array | ten values: x, y of the five landmarks inside the crop |
| `face_sface.embedding.output_slot` | integer | slot holding the embedding |
| `face_sface.embedding.dimension` | integer | embedding length |
| `face_sface.threshold.cosine` | float | published same-person threshold; informational |

## Operation list

Both contracts describe their network as a flat list of records in `<prefix>.graph.ops`, eight
integers each:

| Position | Field |
|---:|---|
| 0 | kind: 0 convolution, 1 max-pool 2x2 stride 2, 2 nearest upsample x2, 3 add, 4 linear |
| 1 | input slot |
| 2 | second input slot (add only) |
| 3 | output slot |
| 4 | stride (convolution only) |
| 5 | symmetric zero padding (convolution only) |
| 6 | groups (convolution only) |
| 7 | activation: 0 none, 1 ReLU, 2 PReLU (convolution only) |

Slot 0 is the network input. Every operation reads slots written earlier and writes a slot not
written before.

Tensors are numbered in operation order, separately for convolutions and linear layers:

- `face.conv.N.weight`, written from a `[out, in / groups, kernel, kernel]` array, square kernel;
- `face.conv.N.bias`, `[out]`;
- `face.conv.N.slope`, `[out]`, only for PReLU;
- `face.linear.N.weight`, written from an `[outputs, inputs]` array over the flattened
  channel-major input;
- `face.linear.N.bias`, `[outputs]`.

Batch normalization is not an operation: the converter folds it into the neighbouring convolution
or linear layer. Input normalization is not folded, because zero padding is applied after it.
F32 and F16 tensors are accepted and expanded to float at load time.
