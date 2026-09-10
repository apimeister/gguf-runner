//! Source-space geometry only: no decoding, model loading, or vendor defaults.

use crate::engine::types::{
    ImageRect, ImageSourcePlan, ImageViewKind, ImageViewLimits, ImageViewPolicy, ImageViewSpec,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageViewPlanError {
    InvalidDimensions,
    InvalidPolicy,
    InvalidLimits,
    SizeOverflow,
    SourcePixelLimit,
    ViewLimit,
    EmptyCrop,
    AllocationFailed,
}

impl std::fmt::Display for ImageViewPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => "image source dimensions must be positive",
            Self::InvalidPolicy => {
                "image crop policy requires a positive minimum size, at least two crops, and a finite aspect ratio greater than one"
            }
            Self::InvalidLimits => "image source-pixel and view limits must be positive",
            Self::SizeOverflow => "image source or view geometry size overflow",
            Self::SourcePixelLimit => "image source exceeds the decoded-pixel limit",
            Self::ViewLimit => "image source exceeds the view limit including its overview",
            Self::EmptyCrop => "image crop policy produces an empty edge crop",
            Self::AllocationFailed => "unable to allocate image view plan",
        })
    }
}

impl std::error::Error for ImageViewPlanError {}

/// Plan from dimensions in the caller's decoded coordinate system. Orientation
/// handling belongs to decoding; never apply a plan to differently oriented pixels.
pub(crate) fn plan_image_views(
    source_index: usize,
    width: u32,
    height: u32,
    policy: ImageViewPolicy,
    limits: ImageViewLimits,
) -> Result<ImageSourcePlan, ImageViewPlanError> {
    if width == 0 || height == 0 {
        return Err(ImageViewPlanError::InvalidDimensions);
    }
    if limits.max_source_pixels == 0 || limits.max_views == 0 {
        return Err(ImageViewPlanError::InvalidLimits);
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(ImageViewPlanError::SizeOverflow)?;
    pixels
        .checked_mul(3)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(ImageViewPlanError::SizeOverflow)?;
    if pixels > limits.max_source_pixels {
        return Err(ImageViewPlanError::SourcePixelLimit);
    }

    let mut num_crops = 0u32;
    let landscape = width >= height;
    let long = width.max(height);
    let short = width.min(height);
    let mut crop_extent = long;
    if let ImageViewPolicy::LongAxisCrops {
        min_crop_size,
        max_crops,
        min_aspect_ratio,
    } = policy
    {
        if min_crop_size == 0
            || max_crops < 2
            || !min_aspect_ratio.is_finite()
            || min_aspect_ratio <= 1.0
        {
            return Err(ImageViewPlanError::InvalidPolicy);
        }
        // Use the reference's binary64 division and half-up rounding. Inputs are
        // u32, so the rounded ratio and the subsequent count fit in u32.
        let ratio = f64::from(long) / f64::from(short);
        if ratio >= min_aspect_ratio {
            num_crops = ((ratio + 0.5).floor() as u32)
                .min(long / min_crop_size)
                .max(2)
                .min(max_crops);
            crop_extent = long.div_ceil(num_crops);
            // Check the nominal extent, not each clipped rectangle: a final
            // crop may be narrower than min_crop_size in the external oracle.
            if short.min(crop_extent) < min_crop_size {
                num_crops = 0;
            }
        }
    }

    let view_count = (num_crops as usize)
        .checked_add(1)
        .ok_or(ImageViewPlanError::SizeOverflow)?;
    if view_count > limits.max_views {
        return Err(ImageViewPlanError::ViewLimit);
    }
    if num_crops > 0 && u64::from(crop_extent) * u64::from(num_crops - 1) >= u64::from(long) {
        // Pathological custom parameters can make the reference return an empty
        // ndarray slice. Such a view cannot be resized or encoded.
        return Err(ImageViewPlanError::EmptyCrop);
    }
    let mut views = Vec::new();
    views
        .try_reserve_exact(view_count)
        .map_err(|_| ImageViewPlanError::AllocationFailed)?;
    views.push(ImageViewSpec {
        source_index,
        view_index: 0,
        kind: ImageViewKind::Overview,
        rect: ImageRect {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        },
    });
    for crop_index in 0..num_crops {
        let start = crop_extent
            .checked_mul(crop_index)
            .ok_or(ImageViewPlanError::SizeOverflow)?;
        let end = u64::from(start)
            .checked_add(u64::from(crop_extent))
            .ok_or(ImageViewPlanError::SizeOverflow)?
            .min(u64::from(long)) as u32;
        if start >= end {
            return Err(ImageViewPlanError::EmptyCrop);
        }
        let rect = if landscape {
            ImageRect {
                left: start,
                top: 0,
                right: end,
                bottom: height,
            }
        } else {
            ImageRect {
                left: 0,
                top: start,
                right: width,
                bottom: end,
            }
        };
        views.push(ImageViewSpec {
            source_index,
            view_index: views.len(),
            kind: ImageViewKind::Crop,
            rect,
        });
    }
    Ok(ImageSourcePlan {
        source_index,
        source_width: width,
        source_height: height,
        views,
    })
}

#[cfg(test)]
mod tests {
    use super::{ImageViewPlanError, plan_image_views};
    use crate::engine::types::{ImageViewKind, ImageViewLimits, ImageViewPolicy};

    const LIMITS: ImageViewLimits = ImageViewLimits {
        max_source_pixels: 4096 * 4096,
        max_views: 5,
    };

    #[test]
    fn image_view_overview_policy_and_limits() {
        let plan = plan_image_views(7, 4001, 1000, ImageViewPolicy::OverviewOnly, LIMITS).unwrap();
        assert_eq!(plan.views.len(), 1);
        assert_eq!(plan.views[0].source_index, 7);
        assert_eq!(plan.views[0].kind, ImageViewKind::Overview);
        let policy = ImageViewPolicy::LongAxisCrops {
            min_crop_size: 256,
            max_crops: 4,
            min_aspect_ratio: 1.2,
        };
        // An exact budget succeeds. One view or source pixel less fails the
        // whole plan, without silently changing the requested crop geometry.
        assert!(
            plan_image_views(
                0,
                4001,
                1000,
                policy,
                ImageViewLimits {
                    max_source_pixels: 4_001_000,
                    max_views: 5,
                }
            )
            .is_ok()
        );
        assert_eq!(
            plan_image_views(
                0,
                4001,
                1000,
                policy,
                ImageViewLimits {
                    max_source_pixels: 4_001_000,
                    max_views: 4,
                }
            ),
            Err(ImageViewPlanError::ViewLimit)
        );
        assert_eq!(
            plan_image_views(
                0,
                4001,
                1000,
                policy,
                ImageViewLimits {
                    max_source_pixels: 4_000_999,
                    max_views: 5,
                }
            ),
            Err(ImageViewPlanError::SourcePixelLimit)
        );
        // A cap exceeding the budget is harmless when geometry adds no crops.
        assert!(
            plan_image_views(
                0,
                896,
                896,
                policy,
                ImageViewLimits {
                    max_views: 1,
                    ..LIMITS
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn image_view_invalid_inputs_fail_before_allocation() {
        for (width, height) in [(0, 1), (1, 0), (0, 0)] {
            assert_eq!(
                plan_image_views(0, width, height, ImageViewPolicy::OverviewOnly, LIMITS),
                Err(ImageViewPlanError::InvalidDimensions)
            );
        }
        for (min_crop_size, max_crops, min_aspect_ratio) in [
            (0, 4, 1.2),
            (256, 0, 1.2),
            (256, 1, 1.2),
            (256, 4, 1.0),
            (256, 4, -1.0),
            (256, 4, f64::NAN),
            (256, 4, f64::INFINITY),
        ] {
            assert_eq!(
                plan_image_views(
                    0,
                    896,
                    896,
                    ImageViewPolicy::LongAxisCrops {
                        min_crop_size,
                        max_crops,
                        min_aspect_ratio,
                    },
                    LIMITS
                ),
                Err(ImageViewPlanError::InvalidPolicy)
            );
        }
        for limits in [
            ImageViewLimits {
                max_views: 0,
                ..LIMITS
            },
            ImageViewLimits {
                max_source_pixels: 0,
                ..LIMITS
            },
        ] {
            assert_eq!(
                plan_image_views(0, 1, 1, ImageViewPolicy::OverviewOnly, limits),
                Err(ImageViewPlanError::InvalidLimits)
            );
        }
        assert_eq!(
            plan_image_views(
                0,
                u32::MAX,
                u32::MAX,
                ImageViewPolicy::OverviewOnly,
                ImageViewLimits {
                    max_source_pixels: usize::MAX,
                    max_views: usize::MAX
                }
            ),
            Err(ImageViewPlanError::SizeOverflow)
        );
        assert_eq!(
            plan_image_views(
                0,
                5,
                1,
                ImageViewPolicy::LongAxisCrops {
                    min_crop_size: 1,
                    max_crops: 4,
                    min_aspect_ratio: 1.2,
                },
                LIMITS
            ),
            Err(ImageViewPlanError::EmptyCrop)
        );
    }
}
