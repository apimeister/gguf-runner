use super::{ImageNormalization, ImagePreprocessProfile, ImageResizeMode, PlannedImageSource};
use crate::engine::types::{
    ImageOrientationPolicy, ImageSourceLimits, ImageStretchFilter, ImageViewEncoding,
    ImageViewPolicy,
};
use std::path::Path;

fn profile() -> ImagePreprocessProfile {
    ImagePreprocessProfile::new_with_mode(
        17,
        17,
        ImageNormalization::MeanStd {
            mean: [0.5; 3],
            std: [0.5; 3],
        },
        ImageResizeMode::Stretch,
        1,
    )
    .with_stretch_filter(ImageStretchFilter::PillowBilinear)
}
fn encoding() -> ImageViewEncoding {
    ImageViewEncoding {
        width: 17,
        height: 17,
        tokens: 2,
        dimension: 4,
        grid: None,
    }
}
fn policy() -> ImageViewPolicy {
    ImageViewPolicy::LongAxisCrops {
        min_crop_size: 8,
        max_crops: 4,
        min_aspect_ratio: 1.2,
    }
}
fn limits() -> ImageSourceLimits {
    ImageSourceLimits {
        max_file_bytes: 1 << 20,
        max_source_pixels: 1 << 20,
        max_decoder_bytes: 1 << 20,
        max_prepare_bytes: 1 << 20,
        max_embedding_bytes: 1 << 20,
        max_views: 5,
    }
}

#[test]
fn grouped_source_contract_validation_rejects_nonfinite_and_overflow() {
    let missing = Path::new("missing-image-for-contract-validation");
    for bad in [
        ImageViewEncoding {
            tokens: 0,
            ..encoding()
        },
        ImageViewEncoding {
            tokens: usize::MAX,
            ..encoding()
        },
        ImageViewEncoding {
            width: 0,
            ..encoding()
        },
        ImageViewEncoding {
            grid: Some([1, 1, 3]),
            ..encoding()
        },
    ] {
        let error = PlannedImageSource::open(
            missing,
            0,
            policy(),
            profile(),
            bad,
            ImageOrientationPolicy::ApplyExif,
            limits(),
            |_, _| Ok(1),
        )
        .unwrap_err();
        assert!(!error.contains("cannot inspect"), "{error}");
    }
    for normalization in [
        ImageNormalization::MeanStd {
            mean: [f32::NAN; 3],
            std: [1.0; 3],
        },
        ImageNormalization::MeanStd {
            mean: [0.0; 3],
            std: [0.0; 3],
        },
        ImageNormalization::MeanStd {
            mean: [f32::MAX; 3],
            std: [0.1; 3],
        },
    ] {
        let error = PlannedImageSource::open(
            missing,
            0,
            policy(),
            ImagePreprocessProfile {
                normalization,
                ..profile()
            },
            encoding(),
            ImageOrientationPolicy::ApplyExif,
            limits(),
            |_, _| Ok(1),
        )
        .unwrap_err();
        assert!(error.contains("finite"), "{error}");
    }
    assert!(
        super::bilinear::storage_bound(usize::MAX, usize::MAX, usize::MAX, usize::MAX).is_none()
    );
}
