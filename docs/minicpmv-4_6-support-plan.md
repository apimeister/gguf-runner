# MiniCPM-V 4.6 Support Plan

Handoff plan for adding support for the OpenBMB MiniCPM-V 4.6 GGUF pair:

- `MiniCPM-V-4_6-Q4_K_M.gguf` (language model, 529 MB)
- `mmproj-model-f16.gguf` (vision sidecar, 1109 MB)

Local copies live in the Hugging Face cache under
`models--openbmb--MiniCPM-V-4.6-gguf/snapshots/afe9accb78d2995d214cd912920c9c92f4015faa/`.

## Current Conclusion

The text half already works. The vision half is one new mmproj backend.

Unlike the LFM2-VL case, no new text family, weight-loader path, or runtime block is
needed: the language model reports `general.architecture = qwen35` and runs correctly
through the existing Qwen3.5 vendor today. The work is confined to the multimodal
layer: sidecar acceptance, a vision encoder, LLaVA-UHD slicing, and a slice-structured
prompt contract.

## What We Verified

### The language model already runs

```
gguf-runner --model .../MiniCPM-V-4_6-Q4_K_M.gguf \
  --prompt "Nenne drei Farben." --think no --temperature 0 --max-tokens 120
```

produces coherent German in 2.5 s. No tensor-shape mismatches, no fallback warnings.

Text-side metadata:

- `general.architecture = qwen35`, `general.name = MiniCPM V 4_6`, `general.size_label = 752M`
- `qwen35.block_count = 24`, `qwen35.embedding_length = 1024`, `qwen35.context_length = 262144`
- hybrid SSM/attention keys (`qwen35.ssm.*`, `qwen35.full_attention_interval = 4`) — all already handled
- ChatML chat template with `<think>` blocks, identical in shape to other Qwen3.5 checkpoints
- vocab carries both Qwen vision tokens (`<|vision_start|>`, `<|image_pad|>`, ids 248053–248057)
  and MiniCPM slice tokens (`<image>`, `</image>`, `<slice>`, `</slice>`, ids 248078–248089)

### The current failure is sidecar rejection

```
Skipping mmproj sidecar 'mmproj-model-f16.gguf': not compatible with backend 'qwen35'
(unsupported mmproj for this runner: expected clip.projector_type='qwen3vl_merger')
```

The sidecar is found by `discover_mmproj_candidates` and parses cleanly. It is dropped in
`validate_vision_mmproj_contract` (`src/vendors/mod.rs:433`), after which the probe reports
`mmproj(path=not-found)` — the path is on disk, the contract is what fails.

### Sidecar metadata and tensors

- `general.architecture = clip`, `clip.projector_type = minicpmv4_6`
- `clip.vision.image_size = 448`, `clip.vision.patch_size = 14`, `clip.vision.projection_dim = 1024`
- `clip.vision.embedding_length = 1152`, `clip.vision.feed_forward_length = 4304`
- `clip.vision.block_count = 27`, `clip.vision.attention.head_count = 16`
- `clip.vision.attention.layer_norm_epsilon = 1e-6`, `clip.use_gelu = true`
- `clip.vision.image_mean = clip.vision.image_std = [0.5, 0.5, 0.5]`
- `clip.vision.projector.scale_factor = 4`
- `clip.vision.wa_layer_indexes = [6]`

459 tensors in three groups:

- vision tower — `v.patch_embd.*`, `v.position_embd.weight` `[1152, 4900]`, `v.blk.0..26.*`, `v.post_ln.*`
- ViT merger — `v.vit_merger.ln1.*`, `v.vit_merger.attn_{q,k,v,out}.*`,
  `v.vit_merger.ds_ln.*` `[4608]`, `v.vit_merger.ds_ffn_up.*` `[4608→17216]`,
  `v.vit_merger.ds_ffn_down.*` `[17216→1152]`
- final projector — `mm.input_norm.*` `[4608]`, `mm.up.*` `[4608→4608]`, `mm.down.*` `[4608→1024]`

The tower is SigLIP-so400m: the same 1152/4304/27/16 geometry the Gemma3 and Idefics3
encoders already run.

## Reference Implementation

A llama.cpp checkout sits at `/Users/jens/tmp/everlock/llama.cpp` (commit `1744c6bde`) and
implements this projector. Authoritative sources:

- `tools/mtmd/models/minicpmv.cpp:116` — `clip_graph_minicpmv4_6::build()`, the full graph
- `tools/mtmd/clip.cpp:1447` — hparams: `n_merge` from `scale_factor`, `insert_layer_id` from `wa_layer_indexes[0]`
- `tools/mtmd/clip.cpp:4140` — output token count: `n_patches /= n_merge * n_merge`
- `tools/mtmd/clip.cpp:4690` — SigLIP position-bucket input construction
- `tools/mtmd/mtmd-image.cpp:502` — `llava_uhd::get_slice_instructions`, the slicing policy
- `tools/mtmd/mtmd-image.cpp:820` — `minicpmv::get_slice_instructions`, the `n_merge == 2` override
- `tools/mtmd/mtmd.cpp:682` — prompt/slice template selection

## Architecture

### Merge factor selects the graph

`n_merge` defaults to 4 and is overridden by `clip.vision.projector.scale_factor`; it is
either 2 or 4. This checkpoint carries 4, which selects the 16× path: a merger inserted
mid-tower plus a final merger. `n_merge == 2` selects a simpler 4× path with the final
merger only. Supporting 4 alone covers this checkpoint.

`insert_layer_id` comes from `clip.vision.wa_layer_indexes[0]` — the key name says window
attention, the value is the merger insertion point. Here it is 6.

### Forward pass (n_merge = 4)

1. Patch embed: 14×14 conv, stride 14, to 1152 channels.
2. Learned position embeddings, gathered by bucket from a 70×70 table:
   `bucket_h[i] = floor(70 * i / pos_h)`, `bucket_w[j] = floor(70 * j / pos_w)`,
   index `bucket_h[i] * 70 + bucket_w[j]`. Nearest-neighbour bucketing, not interpolation.
3. ViT layers 0..=6 at full patch resolution. Pre-norm, affine LayerNorm, GELU FFN.
4. ViT merger attention:
   - residual save, LayerNorm `vit_merger.ln1`
   - reorder tokens to window-major so each 2×2 window is 4 contiguous tokens
   - q/k/v/out attention under a block-diagonal mask — each window attends only to itself
   - inverse reorder, add residual
5. ViT merger downsample, 4 tokens to 1:
   - `mean_res = (p0 + p1 + p2 + p3) * 0.25`
   - `cat = concat(p0, p1, p2, p3)` → 4608
   - LayerNorm `vit_merger.ds_ln`
   - FFN `ds_ffn_up` → **GELU tanh** → `ds_ffn_down` → 1152
   - output `= ffn_out + mean_res`
6. ViT layers 7..=26 at quarter resolution.
7. `post_ln`.
8. Final merger, 4 tokens to 1:
   - `cat = concat(p0, p1, p2, p3)` → 4608
   - LayerNorm `mm.input_norm`
   - FFN `mm.up` → **GELU erf** → `mm.down` → 1024
   - no residual
9. Output width 1024 matches `qwen35.embedding_length`.

The two GELU variants differ by design: `gelu_pytorch_tanh` in the ViT merger,
`nn.GELU` (erf) in the final merger.

### Token budget

Each prepared view yields `(pw * ph) / 16` embeddings. A 448×448 view is 32×32 patches,
so **64 tokens per view**.

### Preprocessing — LLaVA-UHD

`slice_size = 448`, alignment `patch_size * n_merge = 56`.

An image within 448 on both axes produces a single 448×448 overview and no slices. Larger
images produce an overview plus a grid:

- `ratio = w * h / 448²`, `multiple = min(ceil(ratio), 9)` (`max_slice_nums = 9`)
- candidate slice counts `{multiple - 1, multiple, multiple + 1}`, dropping 1 and anything above 9
- every factorization `m × n` of each candidate is scored by `|log(w/h) - log(m/n)|`; lowest wins
- the source is resized to a refined size that divides evenly by the grid, then cropped into equal slices

Worked example — `ocr1.png` at 1444×1116: `ratio ≈ 8.03`, `multiple = 9`, candidates `{8, 9}`,
best grid **3×3**. Ten views (overview + 9 slices) at 64 tokens each = **640 image tokens**.

### Prompt structure

Overview first, then slices row by row:

```
<image>(64 embeddings)</image><slice>(64)</slice><slice>(64)</slice><slice>(64)</slice>\n<slice>…
```

`\n` terminates each row, with no trailing newline after the last row. The chat template's
`<|image_pad|>` marker is the insertion point; the whole structure replaces it.

## Gaps in gguf-runner

1. **Backend selection is text-driven** (`src/vendors/mod.rs:548`). `detect_model_capabilities`
   picks the vision backend from the text family, so this checkpoint resolves to
   `MultimodalBackend::Qwen35` before any sidecar is read. llama.cpp inverts this — see
   Design Decisions.
2. **Backend enum** (`src/engine/types.rs:266`) has no MiniCPM-V variant.
3. **Sidecar validation** (`src/vendors/mod.rs:460`) accepts only `gemma3`,
   `qwen3vl_merger`, and `idefics3`.
4. **No vision encoder** — `build_vision_encoder_from_mmproj` (`src/engine/multimodal/mod.rs:549`)
   has no MiniCPM arm.
5. **View planning** (`src/engine/vision/views.rs:40`) offers `OverviewOnly` and
   `LongAxisCrops`. LLaVA-UHD needs a 2D grid policy.
6. **View geometry is single-size.** `PreparedImageTensor` already carries per-image
   `width`/`height` and the encoders derive their patch grid from it, so the encoder side is
   ready. The constraint is upstream: `prepare_images_for_multimodal`
   (`src/engine/vision/preprocess.rs:303`) applies one `ImagePreprocessProfile` to every view,
   and `ImageViewSpec` (`src/engine/types.rs:182`) carries a source rect with no target size.
7. **Prompt encoding** — no vendor emits the `<image>`/`<slice>` structure with per-row
   newlines.

Injection itself needs no new concepts: `PlaceholderSpan` plus `MediaEmbeddingSequence`
already express "this literal token, then N embeddings, then that literal token", which is
what the slice template is made of. The Gemma multi-view encoder
(`src/vendors/gemma_image_views.rs`) is the closest working precedent.

## Implementation Phases

### Phase 1 — Sidecar-driven backend selection — DONE

- add `MultimodalBackend::MiniCpmV`
- resolve the vision backend from the sidecar's `clip.projector_type` inside
  `probe_mmproj_sidecar`, keeping `projection_dim == cfg.dim` as the pairing check
- leave text-family selection alone; only the vision backend moves off the text architecture
- accept `minicpmv4_6`; reject `n_merge == 2` explicitly until the 4× path exists
- reduce the backend argument in `score_mmproj_candidate` to filename ranking only

Success: the probe reports `backend=minicpmv` from the sidecar alone, and a `--debug` run
shows the sidecar paired rather than skipped.

Landed as `MultimodalBackend::MiniCpmV`, `vision_backends_for_projector`, and
`resolve_vision_backend` in `src/vendors/mod.rs`; the resolved backend travels on
`MmprojSidecarProbe.vision_backend` and reaches `build_vision_encoder_from_mmproj`.
Verified: `vision_backend=minicpmv` in the debug line, and SmolVLM still resolves to
`idefics3` through the same path.

### Phase 2 — Vision encoder — DONE

- add `src/engine/multimodal/minicpmv.rs` and `VisionEncoder::MiniCpmV`
- start from `idefics3.rs`: patch embed, ViT block loop, and post-LN are structurally identical
- add the position-bucket gather (70×70 table)
- add the windowed merger attention, the 2×2 mean-residual downsample MLP, and the split
  layer loop around `insert_layer_id`
- add the final merger with `mm.input_norm` / `mm.up` / `mm.down`
- keep the two GELU variants distinct

Success: a single 448×448 view encodes to 64 embeddings of width 1024.

Landed as `src/engine/multimodal/minicpmv.rs`. Verified against `llama-mtmd-cli` on a
448×448 crop, where both runners take the single-view path: the reference transcribes
`3 · "Was ist eine Order?" / Feurles Drift-Erkennung (Service / Ich glaube, dass das Konzept
das jetzt gerade versucht, in viel Logik auf eine ID basieren` and this encoder returns the
same text and line structure. Remaining differences trace to the prompt wrapper, which
Phase 4 replaces.

Two traps worth remembering:

- **The sidecars disagree about `ffn_up`/`ffn_down`.** SmolVLM names the expansion
  `ffn_down`; MiniCPM-V names it `ffn_up`. A swapped pair has a matching element count, so
  `load_projection` checks both extents and the fields are named `ffn_expand`/`ffn_contract`
  rather than by tensor name.
- **Three activation sites, two GELUs.** The tower and the downsample MLP use the tanh form,
  the final merger uses the erf form (`nn.GELU`). `gelu_erf` is shared from `qwen3_asr`.

### Phase 3 — LLaVA-UHD view planning — DONE

- add an `ImageViewPolicy::UhdGrid { slice_size, max_slices, align }` variant
- implement the grid search and refined-size arithmetic in `views.rs`
- add a per-view target size to `ImageViewSpec`, populated by the UHD planner and defaulting
  to current behaviour for existing backends
- carry `grid_x` / `grid_y` forward so prompt construction can place row breaks

Success: `ocr1.png` plans one overview plus a 3×3 grid, and the per-view target sizes are
patch-aligned.

Landed as `ImageViewPolicy::UhdGrid` plus `plan_uhd_grid` in `src/engine/vision/views.rs`.
`ImageViewSpec` now carries `target_width`/`target_height`, `ImageSourcePlan` carries the
grid, and `PlannedImageSource::open` takes a per-view token function so byte accounting and
preflight sum actual counts instead of multiplying one. Verified: `ocr1.png` plans
`views=10, image_tokens=630` — overview plus 3×3, each 504×392 for 63 tokens — matching the
ten image encodings `llama-mtmd-cli` reports for the same file.

Upstream resizes the whole source to a grid-divisible shape and then cuts equal tiles.
Views here hold source-space rects, so each tile is cut at the matching source fraction and
scaled to the same target; tile boundaries can land a sub-pixel apart from upstream's.

### Phase 4 — Prompt construction — DONE

- add `src/vendors/minicpmv.rs` request encoding that emits the overview/slice structure
  with row newlines
- reuse the ChatML helpers in `qwen_common`; only the media expansion differs
- drive placeholder spans from the view plan so token counts and embeddings agree

Success: the prompt token stream matches the mtmd layout for both the no-slice and the
gridded case.

Landed as `src/vendors/minicpmv.rs`. The policy inversion from Phase 1 had to reach the
prompt as well: `settings.vendor_multimodal_policy` came from the text vendor, but upstream
picks the slice template from the projector alongside the encoder graph. It is now derived
through `multimodal_policy_for_vision` wherever a sidecar resolves — at construction and in
the lazy media path — so MiniCPM-V brings its own contract onto a `qwen35` language model.

`ImageViewPromptEncoder` also gained a `ThinkMode` argument. Gemma3 ignores it; MiniCPM-V
needs it because its ChatML assistant turn seeds `<think>` differently per mode.

### Phase 5 — Validation — DONE

Both paths were compared against `llama-mtmd-cli` on the same checkpoint pair.

Single view, a 448×448 crop where neither runner slices: the reference transcribes
`3 · "Was ist eine Order?" / Feurles Drift-Erkennung (Service / Ich glaube, dass das Konzept
das jetzt gerade versucht, in viel Logik auf eine ID basieren`, and this runner returns the
same text and line structure.

Gridded, the full 1444×1116 page: the transcriptions agree line for line, and this runner is
closer to the page in two places — `„noch lange nicht so fest"` against the reference's
`„noch lange nicht fest"`, and `Update/Delete` against `Update/Delekte`. Both produce the
same slip, `„Ich would vermeiden"` for `„Ich wollte vermeiden"`, which is the clearest
evidence the two pipelines agree.

Thinking must be on for transcription quality. `llama-mtmd-cli` leaves it enabled and does
its transcription inside a `<think>` block; with `--think no` this model narrates instead of
transcribing, in both runners.

Regressions checked rather than assumed: SmolVLM still resolves to `idefics3` and reads the
same crop, and Qwen3.5-2B still runs its own prompt and encoder path.

## File Touch Points

| Area | Files |
| --- | --- |
| Detection / policy | `src/vendors/mod.rs`, `src/vendors/minicpmv.rs` (new) |
| Types | `src/engine/types.rs` |
| Vision encoder | `src/engine/multimodal/mod.rs`, `src/engine/multimodal/minicpmv.rs` (new) |
| View planning | `src/engine/vision/views.rs`, `src/engine/vision/groups.rs`, `src/engine/vision/preprocess.rs` |
| Docs | `docs/module-structure.md`, `docs/features.md`, `docs/image-scaling.md` |

## Design Decisions

Three questions were resolved by following the llama.cpp reference.

### The sidecar selects the vision backend

llama.cpp never infers the vision family from the text architecture. `init_vision()`
(`tools/mtmd/mtmd.cpp:631`) switches on `clip_get_projector_type(ctx_v)` and that one value
selects the encoder graph, the preprocessor, and the slice template together. The only
text-side input is `n_embd`, checked once against `clip_n_mmproj_embd`
(`tools/mtmd/mtmd.cpp:604`) to catch a mismatched pairing: *"hint: you may be using wrong
mmproj"*.

gguf-runner adopts the same direction. `probe_mmproj_sidecar` already parses each candidate
GGUF before anything is committed, so `clip.projector_type` maps to the vision backend and the
existing `projection_dim == cfg.dim` check becomes the pairing validation — the same check
llama.cpp makes. No vocab sniffing, and any future projector is one match arm.

`score_mmproj_candidate` (`src/app/generation.rs:2001`) uses the backend only for filename
hints when ranking several candidates, and the model-key match already dominates at +1000.
Ranking stays; the correctness decision moves to the sidecar.

Which tokens the prompt uses — `<image>`, `<slice>` — is projector-driven upstream too, then
looked up in the text model's vocab. The same split applies: the sidecar decides the
structure, the tokenizer resolves the ids.

### Views keep their own sizes

llama.cpp is variable-resolution per view. `mtmd_image_preproc_out::append` converts u8 to f32
and normalizes — it never resizes. Each slice enters the graph at its own `nx`/`ny`, and
`pos_h`/`pos_w` are derived per image.

`PreparedImageTensor` already carries per-image `width`/`height` and the encoders already read
their patch grid from it, so the encoder side needs no change. The work is in the planning
layer: a per-view target size on `ImageViewSpec`, populated by the UHD planner, replacing the
single profile-wide resize applied in `prepare_images_for_multimodal`.

### Merger attention becomes a per-window loop

The upstream reorder and mask are exactly equivalent to independent 4-token attentions.
`window_idx` (`tools/mtmd/clip.cpp:4726`) groups the four tokens `(2wi, 2wj)`,
`(2wi, 2wj+1)`, `(2wi+1, 2wj)`, `(2wi+1, 2wj+1)` — the same 2×2 block `make_ds_idx` merges in
the next step — and the mask is 0 inside each aligned 4-token block, `float::lowest()`
everywhere else.

A direct loop over the `(pos_h / 2) * (pos_w / 2)` windows computes the same softmax over the
same four keys, so no precision is lost. It also avoids materializing an `n_pos × n_pos` mask
— 4 MB per 448×448 view, ten views for `ocr1.png` — and replaces O(n²) score computation with
O(4n). The reorder exists upstream so flash-attention kernels can run on contiguous blocks,
which does not apply on this CPU path.

## Open Questions

- **Slice geometry is nominal in the vendor policy.** `VendorMultimodalPolicy` is static, so
  `slice_size = 448` and `align = 56` are constants in `src/vendors/minicpmv.rs` rather than
  read from the sidecar. A checkpoint that disagrees fails in the encoder's patch-grid check
  instead of slicing incorrectly, but wiring the sidecar's own values through would be better.
- **Encode time is bounded by CPU BLAS.** The 1 s figure the reference reports is Metal.
  Forced onto the CPU with `--no-mmproj-offload`, `llama-mtmd-cli` spends about 5,000 ms per
  view against this encoder's ~1,400 ms, so the CPU-to-CPU comparison already favours this
  runner by roughly 3.5x. Closing the remaining gap to Metal means a GPU backend, not a
  kernel change.

  Three optimisations landed: NEON `FCVTL`/`FCVTN` for F16 conversion in
  `dequantize_row_f16` and `round_slice_to_f16_precision`, batching views that share a target
  size through one tower pass, and dropping redundant zero-fills from the matmul scratch
  buffers. Together they take a ten-view page from 16.15 s to 14.3 s. All three are bit-exact:
  batched and unbatched runs produce identical output, as do pre- and post-optimisation runs
  at the same flags.

  `cblas_sgemm` accounts for 6.2 s of what remains, so roughly 43 % of the work is already
  inside Accelerate.
- **`n_merge == 2`.** No checkpoint on hand uses it. Rejecting it explicitly costs one line
  and keeps the encoder honest about what it implements.
- **Position-bucket table size.** Resolved: derived from `v.position_embd.weight` rows and
  required to be square, rather than hardcoding 70.
- **Capability reporting before pairing.** `supports_native_image` currently derives from
  text-vocab tokens plus vision tensors. Once the sidecar selects the backend, the honest
  signal is "a compatible sidecar was found and validated", which changes what the probe can
  say about a text GGUF on its own.

## Validation Checklist

From the repo root:

1. `cargo fmt --all --check`
2. `cargo clippy --all-targets --all-features`
3. `cargo check`
4. `rg -n "^use crate::\*;" src/engine -g'*.rs'`
5. `rg -n "^use crate::engine::[A-Za-z0-9_:]+::\*;" src/engine -g'*.rs'`
6. `rg -n "crate::cli::" src/engine -g'*.rs'`

## Quick Resume Summary

- the text model already runs; only the vision half is missing
- the sidecar is `clip.projector_type = minicpmv4_6`, rejected today at contract validation
- the sidecar, not the text architecture, selects the vision backend — as llama.cpp does
- the tower is SigLIP-so400m, the same geometry Gemma3 and Idefics3 already run
- `scale_factor = 4` selects the 16× path: a windowed merger after ViT layer 6, then a final merger
- `wa_layer_indexes[0]` is the merger insertion point, not a window-attention list
- 64 embeddings per view; preprocessing is LLaVA-UHD overview + grid, max 9 slices
- the prompt is `<image>…</image>` then `<slice>…</slice>` runs with `\n` after each row
- the reference graph is `llama.cpp/tools/mtmd/models/minicpmv.cpp:116`
- all five phases have landed; `--think yes` is required for transcription quality
