To download models from Hugging Face, use the `resolve/main/<filename>` pattern.

Examples:
- `wget https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
- `wget https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M.gguf`

## Face detector and face encoder

Face actions use two dedicated GGUFs passed with `--face-detector` and `--face-model`; they do not
use a language model. The project does not bundle them. Convert the OpenCV model-zoo ONNX files
(YuNet is MIT-licensed, SFace is Apache-2.0); `uv` installs the converter's dependencies on the fly:

```bash
wget https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
wget https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx

uv run scripts/convert-face-onnx.py yunet face_detection_yunet_2023mar.onnx face-yunet-2023mar-f32.gguf
uv run scripts/convert-face-onnx.py sface face_recognition_sface_2021dec.onnx face-sface-2021dec-f32.gguf
```

The converted files are about 0.2 MB and 38.5 MB. Weights stay float32. The files used for
verification had these SHA-256 hashes:

- `face_detection_yunet_2023mar.onnx`: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- `face_recognition_sface_2021dec.onnx`: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

Renaming an ONNX file to `.gguf` is not sufficient; see [Face Recognition](face-recognition.md) for
the contracts.

## Speaker embedding model

Speaker recognition actions that consume audio use a dedicated `speaker_xvector` GGUF passed with
`--speaker-model`; they do not use the language model or a multimodal sidecar. Existing indexes and
exported embeddings can be queried or updated without loading that GGUF. The project does not
currently bundle or name a pretrained checkpoint. A compatible model must implement the metadata,
tensor, and audio-frontend contract documented in [Speaker Recognition](speaker-recognition.md).

Renaming an ONNX, ECAPA-TDNN, or other speaker model to `.gguf` is not sufficient. Its weights must
be converted to the documented `speaker_xvector` layout, and its training frontend must match the
runner's frontend. Keep the exact GGUF file used for enrollment: the profile index records its
fingerprint and rejects a different model.

## Qwen3-VL `mmproj` (vision encoder)

Qwen3-VL GGUF is split into two artifacts:
- LLM (`Qwen3VL-...*.gguf`)
- vision encoder sidecar (`mmproj-...*.gguf`)

For multimodal image/video use, download both.

Official Qwen repos:
- `https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF`
- `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF`

Example downloads (30B Instruct):
- `wget https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF/resolve/main/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`
- `wget https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf`

Example downloads (2B Instruct):
- `wget https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- `wget https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`

Placement:
- Place the `mmproj` file in the same directory as the model file.
- `gguf-runner` auto-discovers local `mmproj*.gguf` sidecars (no extra CLI flag).

## Qwen3.5 `mmproj` (vision encoder)

Qwen3.5 GGUF multimodal use also requires a sidecar:
- LLM (`Qwen3.5-...*.gguf`)
- vision encoder sidecar (`mmproj-Qwen3.5-...*.gguf`)

Example repo:
- `https://huggingface.co/unsloth/Qwen3.5-2B-GGUF`

Example downloads:
- `wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf`
- `wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-Qwen3.5-2B-F16.gguf`

Notes:
- Use a same-family pair (`Qwen3.5` text model with `Qwen3.5` mmproj). Do not mix `Qwen3-VL` and `Qwen3.5` sidecars.
- Place the sidecar in the same directory as the model; discovery is automatic.

## Gemma3 `mmproj` (vision encoder)

Gemma3 GGUF multimodal use also requires a sidecar:
- LLM (`gemma-3-...*.gguf`)
- vision encoder sidecar (`mmproj-gemma-3-...*.gguf`, `clip.projector_type=gemma3`)

Notes:
- Use a same-checkpoint pair (for example, `gemma-3-4b-it` text model with `mmproj-gemma-3-4b-it` sidecar).
- Place the sidecar in the same directory as the model; discovery is automatic.
