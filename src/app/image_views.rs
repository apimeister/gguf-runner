//! Atomic execution of one preflighted source group. The request planner must
//! preflight all source groups and the exact prompt before invoking this stage.

use crate::engine::multimodal::MediaEmbeddingSequence;
use crate::engine::types::ImageViewSpec;
use crate::engine::vision::PreparedImageTensor;
use crate::engine::vision::groups::{ImageViewGroupPlan, PlannedImageSource};

#[derive(Debug)]
pub(crate) struct EncodedImageView {
    pub(crate) spec: ImageViewSpec,
    pub(crate) embeddings: MediaEmbeddingSequence,
}

#[derive(Debug)]
pub(crate) struct EncodedImageViews {
    pub(crate) plan: ImageViewGroupPlan,
    pub(crate) views: Vec<EncodedImageView>,
}

/// Views prepared and encoded together. Every matmul in a vision encoder
/// dequantises its whole weight matrix, so one batched pass does that work once
/// for the group instead of once per view. Only views sharing a target size can
/// batch, and the cap keeps the resident prepared tensors bounded.
const MAX_VIEW_BATCH: usize = 16;

pub(super) fn encode_image_source_with(
    source: PlannedImageSource,
    mut encode: impl FnMut(&[PreparedImageTensor]) -> Result<Vec<MediaEmbeddingSequence>, String>,
) -> Result<EncodedImageViews, String> {
    let source_index = source.plan().geometry.source_index;
    let decoded = source
        .decode()
        .map_err(|error| format!("image source {source_index}: {error}"))?;
    let count = decoded.plan().geometry.views.len();
    let mut views = Vec::new();
    views
        .try_reserve_exact(count)
        .map_err(|_| "unable to allocate encoded image group".to_string())?;

    let mut index = 0usize;
    while index < count {
        // Read the batch extent off the plan, so a view of a different target
        // size is never prepared just to be rejected.
        let planned = &decoded.plan().geometry.views;
        let target = (planned[index].target_width, planned[index].target_height);
        let mut end = index + 1;
        while end < count
            && end - index < MAX_VIEW_BATCH
            && (planned[end].target_width, planned[end].target_height) == target
        {
            end += 1;
        }

        // Only this batch and the previously completed projected rows are
        // resident. Errors drop the entire local group before returning.
        let mut specs = Vec::new();
        let mut tensors = Vec::new();
        for view_index in index..end {
            let context =
                |error| format!("image source {source_index}, view {view_index}: {error}");
            let prepared = decoded.prepare_view(view_index).map_err(&context)?;
            specs.push((prepared.spec, prepared.encoding));
            tensors.push(prepared.tensor);
        }

        let context = |error| format!("image source {source_index}, views {index}..{end}: {error}");
        let sequences = encode(&tensors).map_err(&context)?;
        drop(tensors);
        if sequences.len() != specs.len() {
            return Err(context(
                "vision encoder must return one sequence per view".to_string(),
            ));
        }
        for ((spec, expected), sequence) in specs.into_iter().zip(sequences) {
            let context = |error| {
                format!(
                    "image source {source_index}, view {}: {error}",
                    spec.view_index
                )
            };
            if sequence.tokens.len() != expected.tokens
                || sequence.tokens.iter().any(|token| {
                    token.len() != expected.dimension
                        || token.iter().any(|value| !value.is_finite())
                })
                || sequence.grid != expected.grid
            {
                return Err(context("vision embeddings differ from the planned count, dimension, grid, or finiteness contract".to_string()));
            }
            views.push(EncodedImageView {
                spec,
                embeddings: sequence,
            });
        }
        index = end;
    }
    Ok(EncodedImageViews {
        plan: decoded.into_plan(),
        views,
    })
}
