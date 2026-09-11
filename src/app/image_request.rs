//! Complete-request planning for fixed image views. No pixel decode or encoder
//! call can occur until every source, the vendor prompt, and aggregate budgets
//! have passed preflight. Public entry-point routing remains a separate gate.

use super::image_views::encode_image_source_with;
use crate::engine::multimodal::{
    ExpandedMediaPrompt, MediaEmbeddingSequence, VisionEncoder,
    expand_prompt_with_owned_media_embeddings, preflight_media_context,
};
use crate::engine::types::{
    ContentPart, EncodedPrompt, GenerationRequest, ImageOrientationPolicy, ImagePromptSource,
    ImageSourceLimits, ImageViewEncoding, ImageViewPolicy, ImageViewSpec, ThinkMode, Tokenizer,
};
use crate::engine::vision::ImagePreprocessProfile;
use crate::engine::vision::groups::{ImageViewGroupPlan, PlannedImageSource, PreparedImageView};
use crate::vendors::VendorMultimodalPolicy;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImageRequestLimits {
    pub(crate) max_sources: usize,
    pub(crate) max_views: usize,
    pub(crate) max_snapshot_bytes: usize,
    /// All compressed snapshots plus the largest single-source preparation
    /// bound. Conservative: includes that source's snapshot twice during decode.
    /// Excludes allocator/codec scratch, encoder working memory, and embeddings.
    pub(crate) max_prepare_bytes: usize,
    pub(crate) max_embedding_bytes: usize,
    /// Compact vendor text/markers, before expansion to embedding row tokens.
    pub(crate) max_prompt_bytes: usize,
    pub(crate) context_tokens: usize,
    pub(crate) decode_reserve: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImageRequestSettings {
    pub(crate) view_policy: ImageViewPolicy,
    pub(crate) profile: ImagePreprocessProfile,
    pub(crate) encoding: ImageViewEncoding,
    pub(crate) orientation: ImageOrientationPolicy,
    pub(crate) source_limits: ImageSourceLimits,
    pub(crate) limits: ImageRequestLimits,
}

#[derive(Debug, Default)]
pub(crate) struct ImageRequestResources {
    pub(crate) sources: usize,
    pub(crate) views: usize,
    pub(crate) snapshot_bytes: usize,
    pub(crate) preparation_bytes: usize,
    pub(crate) embedding_bytes: usize,
    pub(crate) image_tokens: usize,
    pub(crate) prompt_tokens: usize,
}

#[derive(Debug)]
pub(crate) struct PlannedImageRequest {
    sources: Vec<PlannedImageSource>,
    encoded: EncodedPrompt,
    view_order: Vec<ImageViewSpec>,
    resources: ImageRequestResources,
    dimension: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedImageRequest {
    pub(crate) prompt: ExpandedMediaPrompt,
    pub(crate) groups: Vec<ImageViewGroupPlan>,
    /// Flattened media index -> logical source/view, also indexing image_blocks.
    pub(crate) view_order: Vec<ImageViewSpec>,
    pub(crate) resources: ImageRequestResources,
}

fn bounded_add(total: usize, amount: usize, limit: usize, label: &str) -> Result<usize, String> {
    total
        .checked_add(amount)
        .filter(|&value| value <= limit)
        .ok_or_else(|| format!("image request exceeds {label} limit ({limit})"))
}

impl PlannedImageRequest {
    pub(crate) fn plan(
        tokenizer: &mut Tokenizer,
        vendor: VendorMultimodalPolicy,
        request: &GenerationRequest,
        settings: ImageRequestSettings,
        tokens_for: &dyn Fn(u32, u32) -> Result<usize, String>,
        think_mode: ThinkMode,
    ) -> Result<Self, String> {
        let encode_prompt = vendor
            .image_view_prompt
            .ok_or_else(|| "vendor does not support grouped image request prompts".to_string())?;
        let limits = settings.limits;
        if [
            limits.max_sources,
            limits.max_views,
            limits.max_snapshot_bytes,
            limits.max_prepare_bytes,
            limits.max_embedding_bytes,
            limits.max_prompt_bytes,
            limits.context_tokens,
        ]
        .contains(&0)
        {
            return Err("image request limits must be positive".to_string());
        }
        let mut resources = ImageRequestResources::default();
        for part in &request.parts {
            match part {
                ContentPart::Image(_) => {
                    resources.sources = bounded_add(
                        resources.sources,
                        1,
                        limits.max_sources,
                        "logical source count",
                    )?;
                }
                ContentPart::Audio(_) | ContentPart::Video(_) => {
                    return Err("grouped image requests do not support audio or video".to_string());
                }
                ContentPart::Text(_) => {}
            }
        }
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(resources.sources)
            .map_err(|_| "unable to allocate image source plans".to_string())?;
        let mut prompt_sources = Vec::new();
        let mut view_order = Vec::new();
        let mut counts = Vec::new();
        let mut largest_prepare = 0;
        for part in &request.parts {
            let ContentPart::Image(media) = part else {
                continue;
            };
            let source_index = sources.len();
            let context = |error| format!("image source {source_index}: {error}");
            // Limit snapshot allocation to the request's remaining capacity,
            // before opening the next file, as well as checking aggregate totals.
            let mut source_limits = settings.source_limits;
            source_limits.max_file_bytes = source_limits
                .max_file_bytes
                .min(limits.max_snapshot_bytes - resources.snapshot_bytes)
                .min(limits.max_prepare_bytes - resources.snapshot_bytes);
            source_limits.max_prepare_bytes = source_limits
                .max_prepare_bytes
                .min(limits.max_prepare_bytes - resources.snapshot_bytes);
            source_limits.max_views = source_limits
                .max_views
                .min(limits.max_views - resources.views);
            source_limits.max_embedding_bytes = source_limits
                .max_embedding_bytes
                .min(limits.max_embedding_bytes - resources.embedding_bytes);
            let source = PlannedImageSource::open(
                Path::new(&media.path),
                source_index,
                settings.view_policy,
                settings.profile,
                settings.encoding,
                settings.orientation,
                source_limits,
                tokens_for,
            )
            .map_err(&context)?;
            let plan = source.plan();
            resources.views = bounded_add(
                resources.views,
                plan.geometry.views.len(),
                limits.max_views,
                "view count",
            )?;
            resources.snapshot_bytes = bounded_add(
                resources.snapshot_bytes,
                source.snapshot_bytes(),
                limits.max_snapshot_bytes,
                "snapshot bytes",
            )?;
            resources.embedding_bytes = bounded_add(
                resources.embedding_bytes,
                plan.embedding_bytes,
                limits.max_embedding_bytes,
                "projected embedding bytes",
            )?;
            largest_prepare = largest_prepare.max(plan.preparation_bytes);
            resources.preparation_bytes = bounded_add(
                resources.snapshot_bytes,
                largest_prepare,
                limits.max_prepare_bytes,
                "preparation bytes",
            )?;
            prompt_sources.push(ImagePromptSource {
                source_index,
                view_count: plan.geometry.views.len(),
                grid: plan.geometry.grid,
            });
            view_order.extend_from_slice(&plan.geometry.views);
            counts.extend_from_slice(&plan.view_tokens);
            sources.push(source);
        }
        resources.image_tokens = counts.iter().try_fold(0usize, |sum, &count| {
            sum.checked_add(count)
                .ok_or_else(|| "image request token count overflow".to_string())
        })?;
        let encoded = encode_prompt(
            tokenizer,
            request,
            &prompt_sources,
            limits.max_prompt_bytes,
            think_mode,
        )?;
        resources.prompt_tokens = preflight_media_context(
            &encoded, &counts, &[], limits.context_tokens, limits.decode_reserve,
        ).map_err(|error| format!(
            "image request preflight failed: {} source(s), {} view(s), {} image token(s): {error}",
            resources.sources, resources.views, resources.image_tokens,
        ))?;
        // The vendor must use the same flattened source/view order as execution.
        if encoded
            .image_spans
            .iter()
            .enumerate()
            .any(|(index, span)| span.media_index != index)
        {
            return Err("vendor image prompt changed the planned view order".to_string());
        }
        Ok(Self {
            sources,
            encoded,
            view_order,
            resources,
            dimension: settings.encoding.dimension,
        })
    }

    pub(crate) fn resources(&self) -> &ImageRequestResources {
        &self.resources
    }

    pub(crate) fn encode(self, encoder: &VisionEncoder) -> Result<PreparedImageRequest, String> {
        self.encode_with(|view| encoder.encode_images(std::slice::from_ref(&view.tensor)))
    }

    fn encode_with(
        self,
        mut encode: impl FnMut(&PreparedImageView) -> Result<Vec<MediaEmbeddingSequence>, String>,
    ) -> Result<PreparedImageRequest, String> {
        let mut groups = Vec::new();
        let mut embeddings = Vec::new();
        groups
            .try_reserve_exact(self.resources.sources)
            .map_err(|_| "unable to allocate encoded source groups".to_string())?;
        embeddings
            .try_reserve_exact(self.resources.views)
            .map_err(|_| "unable to allocate encoded request views".to_string())?;
        for source in self.sources {
            let group = encode_image_source_with(source, &mut encode)?;
            for view in group.views {
                if self.view_order.get(embeddings.len()) != Some(&view.spec) {
                    return Err("encoded image view differs from its request plan".to_string());
                }
                embeddings.push(view.embeddings);
            }
            groups.push(group.plan);
        }
        let prompt = expand_prompt_with_owned_media_embeddings(
            &self.encoded,
            embeddings,
            Vec::new(),
            self.dimension,
        )?;
        if prompt.token_ids.len() != self.resources.prompt_tokens
            || prompt.image_blocks.len() != self.resources.views
        {
            return Err("expanded image request differs from its context plan".to_string());
        }
        // No shared cache/state was modified. Any failure above drops all rows
        // from this request, including earlier completed source groups.
        Ok(PreparedImageRequest {
            prompt,
            groups,
            view_order: self.view_order,
            resources: self.resources,
        })
    }
}
