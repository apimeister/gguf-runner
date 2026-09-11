//! MiniCPM-V's image replacement contract, separate from chat templating.
//!
//! Each image part expands into an aspect-preserving overview followed by a grid
//! of slices, mirroring `MTMD_SLICE_TMPL_MINICPMV_2_6` in llama.cpp's `mtmd.cpp`:
//!
//! ```text
//! <image>(overview)</image><slice>(0,0)</slice><slice>(0,1)</slice>\n<slice>(1,0)</slice>…
//! ```
//!
//! A newline closes every slice row except the last. A source small enough to
//! need no slices contributes the overview alone.

use super::{MmprojFilenameScoreHint, VendorDetailCropPolicy, VendorMultimodalPolicy, qwen_common};
use crate::engine::types::{
    ContentPart, EncodedPrompt, GenerationRequest, ImagePromptSource, ImageStretchFilter,
    ImageViewPolicy, MultimodalBackend, PlaceholderSpan, ThinkMode, Tokenizer,
};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const IMAGE_START: &str = "<image>";
const IMAGE_END: &str = "</image>";
const SLICE_START: &str = "<slice>";
const SLICE_END: &str = "</slice>";
/// Stands in for one view's embeddings until injection expands it. Its identity
/// never reaches the model: injection replaces it with the projected rows.
const VIEW_PLACEHOLDER: &str = "<|image_pad|>";

const CONTROL_TOKENS: [&str; 6] = [
    IM_START,
    IM_END,
    IMAGE_START,
    IMAGE_END,
    SLICE_START,
    SLICE_END,
];

/// Nominal slice geometry. `slice_size` mirrors `clip.vision.image_size` and
/// `align` mirrors `patch_size * clip.vision.projector.scale_factor` for the
/// checkpoints this backend accepts; `max_slices` is llava-uhd's fixed cap. A
/// sidecar that disagrees fails in the encoder's patch-grid check rather than
/// slicing incorrectly.
const SLICE_SIZE: u32 = 448;
const SLICE_ALIGN: u32 = 56;
const MAX_SLICES: u32 = 9;

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy {
        image_view_prompt: Some(encode_request),
        image_view_policy: ImageViewPolicy::UhdGrid {
            slice_size: SLICE_SIZE,
            align: SLICE_ALIGN,
            max_slices: MAX_SLICES,
        },
        image_stretch_filter: ImageStretchFilter::PillowBilinear,
        image_prompt_suffix: "",
        detail_crop: VendorDetailCropPolicy::default(),
        mmproj_filename_score_hints: &[MmprojFilenameScoreHint {
            token: "minicpm",
            backend: MultimodalBackend::MiniCpmV,
            match_score: 100,
            mismatch_score: -100,
        }],
        missing_sidecar_hint: "MiniCPM-V image input requires the matching mmproj sidecar from the same checkpoint.",
    }
}

struct PromptBuilder<'a> {
    tokenizer: &'a mut Tokenizer,
    pending: String,
    temp: Vec<i32>,
    encoded: EncodedPrompt,
    bytes: usize,
    limit: usize,
}

impl PromptBuilder<'_> {
    fn account(&mut self, text: &str) -> Result<(), String> {
        self.bytes = self
            .bytes
            .checked_add(text.len())
            .filter(|&bytes| bytes <= self.limit)
            .ok_or_else(|| {
                "grouped image prompt exceeds the compact text byte limit".to_string()
            })?;
        Ok(())
    }

    fn text(&mut self, text: &str) -> Result<(), String> {
        self.account(text)?;
        self.pending
            .try_reserve(text.len())
            .map_err(|_| "unable to allocate grouped image prompt text".to_string())?;
        self.pending.push_str(text);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        if self.pending.is_empty() {
            return Ok(());
        }
        // Adjacent text parts can jointly spell a reserved literal too.
        if CONTROL_TOKENS
            .iter()
            .any(|token| self.pending.contains(token))
        {
            return Err(
                "grouped MiniCPM-V prompt text contains a reserved control token".to_string(),
            );
        }
        self.tokenizer.bpe_encode(&self.pending, &mut self.temp);
        self.encoded.token_ids.extend_from_slice(&self.temp);
        self.pending.clear();
        Ok(())
    }

    fn control(&mut self, literal: &str) -> Result<(), String> {
        self.account(literal)?;
        let token = self.tokenizer.find_special_token(literal).ok_or_else(|| {
            format!("grouped MiniCPM-V image prompt requires tokenizer token {literal}")
        })?;
        self.flush()?;
        self.encoded.token_ids.push(token);
        Ok(())
    }

    /// One view between its own markers. Injection keeps the markers and swaps
    /// the middle token for the view's embeddings.
    fn view(&mut self, start: &str, end: &str) -> Result<(), String> {
        self.control(start)?;
        let token_start = self.encoded.token_ids.len() - 1;
        self.control(VIEW_PLACEHOLDER)?;
        self.control(end)?;
        self.encoded.image_spans.push(PlaceholderSpan {
            token_start,
            token_len: 3,
            media_index: self.encoded.image_spans.len(),
            replace_marker: false,
        });
        Ok(())
    }

    fn image(&mut self, source: &ImagePromptSource) -> Result<(), String> {
        self.view(IMAGE_START, IMAGE_END)?;
        let slices = source.view_count - 1;
        if slices == 0 {
            return Ok(());
        }
        let (cols, rows) = source.grid.ok_or_else(|| {
            "MiniCPM-V slice prompt requires the grid layout for its view group".to_string()
        })?;
        if cols == 0 || rows == 0 || cols * rows != slices {
            return Err(format!(
                "MiniCPM-V slice grid {cols}x{rows} does not match {slices} slice view(s)"
            ));
        }
        for row in 0..rows {
            for _ in 0..cols {
                self.view(SLICE_START, SLICE_END)?;
            }
            // Rows are newline-separated, with none trailing the last row.
            if row + 1 < rows {
                self.text("\n")?;
            }
        }
        Ok(())
    }
}

pub(super) fn encode_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    sources: &[ImagePromptSource],
    max_prompt_bytes: usize,
    think_mode: ThinkMode,
) -> Result<EncodedPrompt, String> {
    let image_count = request
        .parts
        .iter()
        .filter(|part| matches!(part, ContentPart::Image(_)))
        .count();
    if image_count != sources.len()
        || sources
            .iter()
            .enumerate()
            .any(|(index, source)| source.source_index != index || source.view_count == 0)
    {
        return Err(
            "grouped MiniCPM-V prompt requires one nonempty, ordered view group per image occurrence"
                .to_string(),
        );
    }
    if request
        .parts
        .iter()
        .any(|part| matches!(part, ContentPart::Audio(_) | ContentPart::Video(_)))
    {
        return Err("grouped MiniCPM-V image prompts do not support audio or video".to_string());
    }
    // Typed image parts own the image placeholders. A literal control token in
    // user text must not create an extra view, turn, or ambiguous span.
    for text in std::iter::once(request.system_prompt.as_str())
        .chain(request.assistant_prefill.as_deref())
        .chain(request.parts.iter().filter_map(|part| match part {
            ContentPart::Text(text) => Some(text.as_str()),
            _ => None,
        }))
    {
        if CONTROL_TOKENS.iter().any(|token| text.contains(token)) {
            return Err(
                "grouped MiniCPM-V prompt text contains a reserved control token".to_string(),
            );
        }
    }

    let mut builder = PromptBuilder {
        tokenizer,
        pending: String::new(),
        temp: Vec::new(),
        encoded: EncodedPrompt::from_token_ids(Vec::new()),
        bytes: 0,
        limit: max_prompt_bytes,
    };

    let system = request.system_prompt.trim();
    if !system.is_empty() {
        builder.control(IM_START)?;
        builder.text("system\n")?;
        builder.text(system)?;
        builder.control(IM_END)?;
        builder.text("\n")?;
    }

    builder.control(IM_START)?;
    builder.text("user\n")?;
    let mut source_index = 0;
    for part in &request.parts {
        match part {
            ContentPart::Text(text) => builder.text(text)?,
            ContentPart::Image(_) => {
                builder.image(&sources[source_index])?;
                source_index += 1;
            }
            ContentPart::Video(_) | ContentPart::Audio(_) => unreachable!(),
        }
    }
    builder.control(IM_END)?;
    builder.text("\n")?;

    builder.control(IM_START)?;
    builder.text("assistant\n")?;
    match &request.assistant_prefill {
        Some(prefill) => builder.text(prefill)?,
        None => builder.text(qwen_common::assistant_think_seed(think_mode))?,
    }
    builder.flush()?;
    Ok(builder.encoded)
}
