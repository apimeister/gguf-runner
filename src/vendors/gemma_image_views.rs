//! Gemma3Processor's image replacement contract, separate from chat templating.
//! Text between control tokens is encoded together, including adjacent request
//! text parts and the whitespace surrounding views. Image rows stay compact
//! until generic embedding injection expands them.

use crate::engine::types::{
    ContentPart, EncodedPrompt, GenerationRequest, ImagePromptSource, PlaceholderSpan, Tokenizer,
};

const BOS: &str = "<bos>";
const START_TURN: &str = "<start_of_turn>";
const END_TURN: &str = "<end_of_turn>";
const START_IMAGE: &str = "<start_of_image>";
const IMAGE: &str = "<image_soft_token>";
const END_IMAGE: &str = "<end_of_image>";
const CONTROL_TOKENS: [&str; 6] = [BOS, START_TURN, END_TURN, START_IMAGE, IMAGE, END_IMAGE];

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
        // Adjacent text parts can jointly spell a reserved literal too.
        if CONTROL_TOKENS
            .iter()
            .any(|token| self.pending.contains(token))
        {
            return Err("grouped Gemma prompt text contains a reserved control token".to_string());
        }
        self.tokenizer.bpe_encode(&self.pending, &mut self.temp);
        self.encoded.token_ids.extend_from_slice(&self.temp);
        self.pending.clear();
        Ok(())
    }

    fn control(&mut self, literal: &str) -> Result<(), String> {
        self.account(literal)?;
        let token = self.tokenizer.find_special_token(literal).ok_or_else(|| {
            format!("grouped Gemma image prompt requires tokenizer token {literal}")
        })?;
        self.flush()?;
        self.encoded.token_ids.push(token);
        Ok(())
    }

    fn image(&mut self) -> Result<(), String> {
        // Pinned processing_gemma3.py: full_image_sequence surrounds every
        // view, including the overview-only case, with two newlines per side.
        self.text("\n\n")?;
        self.control(START_IMAGE)?;
        let token_start = self.encoded.token_ids.len() - 1;
        self.control(IMAGE)?;
        self.control(END_IMAGE)?;
        self.encoded.image_spans.push(PlaceholderSpan {
            token_start,
            token_len: 3,
            media_index: self.encoded.image_spans.len(),
            replace_marker: false,
        });
        self.text("\n\n")
    }
}

pub(super) fn encode_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    sources: &[ImagePromptSource],
    max_prompt_bytes: usize,
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
            "grouped Gemma prompt requires one nonempty, ordered view group per image occurrence"
                .to_string(),
        );
    }
    if request
        .parts
        .iter()
        .any(|part| matches!(part, ContentPart::Audio(_) | ContentPart::Video(_)))
    {
        return Err("grouped Gemma image prompts do not support audio or video".to_string());
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
            return Err("grouped Gemma prompt text contains a reserved control token".to_string());
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
    builder.control(BOS)?;
    builder.control(START_TURN)?;
    builder.text("user\n")?;
    let system = request.system_prompt.trim();
    if !system.is_empty() {
        builder.text(system)?;
        builder.text("\n\n")?;
    }
    let mut source_index = 0;
    for part in &request.parts {
        match part {
            ContentPart::Text(text) => builder.text(text)?,
            ContentPart::Image(_) => {
                let crops = sources[source_index].view_count - 1;
                if crops > 0 {
                    builder.text("Here is the original image ")?;
                }
                builder.image()?;
                if crops > 0 {
                    builder.text(" and here are some crops to help you see better ")?;
                    for crop in 0..crops {
                        if crop > 0 {
                            builder.text(" ")?;
                        }
                        builder.image()?;
                    }
                }
                source_index += 1;
            }
            ContentPart::Video(_) | ContentPart::Audio(_) => unreachable!(),
        }
    }
    builder.control(END_TURN)?;
    builder.text("\n")?;
    builder.control(START_TURN)?;
    builder.text("model\n")?;
    if let Some(prefill) = &request.assistant_prefill {
        builder.text(prefill)?;
    }
    builder.flush()?;
    Ok(builder.encoded)
}
