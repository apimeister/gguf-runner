//! Atomic execution of one preflighted source group. The request planner must
//! preflight all source groups and the exact prompt before invoking this stage.

use crate::engine::multimodal::MediaEmbeddingSequence;
use crate::engine::types::ImageViewSpec;
use crate::engine::vision::groups::{ImageViewGroupPlan, PlannedImageSource, PreparedImageView};

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

pub(super) fn encode_image_source_with(
    source: PlannedImageSource,
    mut encode: impl FnMut(&PreparedImageView) -> Result<Vec<MediaEmbeddingSequence>, String>,
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
    for index in 0..count {
        let context = |error| format!("image source {source_index}, view {index}: {error}");
        // Only this prepared view and the previously completed projected rows
        // are resident. Errors drop the entire local group before returning.
        let prepared = decoded.prepare_view(index).map_err(&context)?;
        let mut sequences = encode(&prepared).map_err(&context)?;
        if sequences.len() != 1 {
            return Err(context(
                "vision encoder must return exactly one sequence per view".to_string(),
            ));
        }
        let sequence = sequences.pop().unwrap();
        let expected = prepared.encoding;
        if sequence.tokens.len() != expected.tokens
            || sequence.tokens.iter().any(|token| {
                token.len() != expected.dimension || token.iter().any(|value| !value.is_finite())
            })
            || sequence.grid != expected.grid
        {
            return Err(context("vision embeddings differ from the planned count, dimension, grid, or finiteness contract".to_string()));
        }
        views.push(EncodedImageView {
            spec: prepared.spec,
            embeddings: sequence,
        });
    }
    Ok(EncodedImageViews {
        plan: decoded.into_plan(),
        views,
    })
}
