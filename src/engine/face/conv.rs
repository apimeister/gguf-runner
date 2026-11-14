//! Dense CHW float kernels for the small convolutional face networks.

use rayon::prelude::*;

/// One activation map in channel-major order: `data[(c * height + y) * width + x]`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FaceTensor {
    pub(crate) channels: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) data: Vec<f32>,
}

impl FaceTensor {
    pub(crate) fn new(
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self, String> {
        let expected = channels
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| "face tensor shape overflow".to_string())?;
        if expected == 0 || data.len() != expected {
            return Err(format!(
                "face tensor has {} values, expected {channels}x{height}x{width}",
                data.len()
            ));
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
    }

    pub(crate) fn plane_len(&self) -> usize {
        self.height * self.width
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaceActivation {
    None,
    Relu,
    PRelu,
}

/// Weights in `[out, in / groups, kernel, kernel]` order with square kernels and symmetric
/// zero padding.
pub(crate) struct FaceConvLayer {
    pub(crate) out_channels: usize,
    pub(crate) in_per_group: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    pub(crate) pad: usize,
    pub(crate) groups: usize,
    pub(crate) activation: FaceActivation,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    /// Per-channel negative slope, present only for `FaceActivation::PRelu`.
    pub(crate) slope: Vec<f32>,
}

impl FaceConvLayer {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.out_channels == 0
            || self.in_per_group == 0
            || self.kernel == 0
            || self.stride == 0
            || self.groups == 0
            || !self.out_channels.is_multiple_of(self.groups)
        {
            return Err("face convolution has an invalid shape".to_string());
        }
        let weights = self
            .out_channels
            .checked_mul(self.in_per_group)
            .and_then(|n| n.checked_mul(self.kernel * self.kernel));
        if weights != Some(self.weight.len()) || self.bias.len() != self.out_channels {
            return Err("face convolution weights do not match its shape".to_string());
        }
        let slope_len = match self.activation {
            FaceActivation::PRelu => self.out_channels,
            _ => 0,
        };
        if self.slope.len() != slope_len {
            return Err("face convolution slope does not match its activation".to_string());
        }
        Ok(())
    }
}

pub(crate) fn conv2d(input: &FaceTensor, layer: &FaceConvLayer) -> Result<FaceTensor, String> {
    if input.channels != layer.in_per_group * layer.groups {
        return Err(format!(
            "face convolution expects {} input channels, got {}",
            layer.in_per_group * layer.groups,
            input.channels
        ));
    }
    let (kernel, stride, pad) = (layer.kernel, layer.stride, layer.pad);
    if input.height + 2 * pad < kernel || input.width + 2 * pad < kernel {
        return Err("face convolution input is smaller than its kernel".to_string());
    }
    let out_h = (input.height + 2 * pad - kernel) / stride + 1;
    let out_w = (input.width + 2 * pad - kernel) / stride + 1;
    let in_plane = input.plane_len();
    let out_plane = out_h * out_w;
    let out_per_group = layer.out_channels / layer.groups;
    let pointwise = kernel == 1 && stride == 1 && pad == 0;
    let mut data = vec![0.0f32; layer.out_channels * out_plane];
    data.par_chunks_mut(out_plane)
        .enumerate()
        .for_each(|(out_channel, plane)| {
            plane.fill(layer.bias[out_channel]);
            let group = out_channel / out_per_group;
            for in_index in 0..layer.in_per_group {
                let in_channel = group * layer.in_per_group + in_index;
                let source = &input.data[in_channel * in_plane..(in_channel + 1) * in_plane];
                let taps = (out_channel * layer.in_per_group + in_index) * kernel * kernel;
                if pointwise {
                    axpy(plane, source, layer.weight[taps]);
                    continue;
                }
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let weight = layer.weight[taps + ky * kernel + kx];
                        accumulate_tap(
                            plane,
                            source,
                            weight,
                            TapGeometry {
                                in_h: input.height,
                                in_w: input.width,
                                out_h,
                                out_w,
                                stride,
                                // Offset of this tap relative to the padded origin.
                                dy: ky as isize - pad as isize,
                                dx: kx as isize - pad as isize,
                            },
                        );
                    }
                }
            }
            match layer.activation {
                FaceActivation::None => {}
                FaceActivation::Relu => plane.iter_mut().for_each(|v| *v = v.max(0.0)),
                FaceActivation::PRelu => {
                    let slope = layer.slope[out_channel];
                    plane.iter_mut().for_each(|v| {
                        if *v < 0.0 {
                            *v *= slope;
                        }
                    });
                }
            }
        });
    FaceTensor::new(layer.out_channels, out_h, out_w, data)
}

#[derive(Clone, Copy)]
struct TapGeometry {
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    stride: usize,
    dy: isize,
    dx: isize,
}

#[inline]
fn axpy(dst: &mut [f32], src: &[f32], weight: f32) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += weight * s;
    }
}

/// First and one-past-last output index whose source coordinate `o * stride + offset` is inside
/// `0..limit`.
fn valid_range(out_len: usize, limit: usize, stride: usize, offset: isize) -> (usize, usize) {
    let stride = stride as isize;
    let first = if offset >= 0 {
        0
    } else {
        (-offset + stride - 1) / stride
    };
    let last = (limit as isize - 1 - offset).div_euclid(stride) + 1;
    let first = first.clamp(0, out_len as isize) as usize;
    let last = last.clamp(first as isize, out_len as isize) as usize;
    (first, last)
}

fn accumulate_tap(plane: &mut [f32], source: &[f32], weight: f32, g: TapGeometry) {
    let (y0, y1) = valid_range(g.out_h, g.in_h, g.stride, g.dy);
    let (x0, x1) = valid_range(g.out_w, g.in_w, g.stride, g.dx);
    if x0 >= x1 {
        return;
    }
    for oy in y0..y1 {
        let iy = (oy * g.stride) as isize + g.dy;
        let row = &source[iy as usize * g.in_w..(iy as usize + 1) * g.in_w];
        let out = &mut plane[oy * g.out_w + x0..oy * g.out_w + x1];
        let start = ((x0 * g.stride) as isize + g.dx) as usize;
        if g.stride == 1 {
            axpy(out, &row[start..start + (x1 - x0)], weight);
        } else {
            for (index, value) in out.iter_mut().enumerate() {
                *value += weight * row[start + index * g.stride];
            }
        }
    }
}

/// 2x2 max pooling with stride 2; a trailing odd row or column is dropped.
pub(crate) fn max_pool2(input: &FaceTensor) -> Result<FaceTensor, String> {
    let (out_h, out_w) = (input.height / 2, input.width / 2);
    if out_h == 0 || out_w == 0 {
        return Err("face max-pool input is smaller than its window".to_string());
    }
    let mut data = Vec::with_capacity(input.channels * out_h * out_w);
    for plane in input.data.chunks_exact(input.plane_len()) {
        for y in 0..out_h {
            let top = &plane[2 * y * input.width..];
            let bottom = &plane[(2 * y + 1) * input.width..];
            for x in 0..out_w {
                data.push(
                    top[2 * x]
                        .max(top[2 * x + 1])
                        .max(bottom[2 * x])
                        .max(bottom[2 * x + 1]),
                );
            }
        }
    }
    FaceTensor::new(input.channels, out_h, out_w, data)
}

/// Nearest-neighbour upsampling by two in both axes.
pub(crate) fn upsample2(input: &FaceTensor) -> Result<FaceTensor, String> {
    let (out_h, out_w) = (input.height * 2, input.width * 2);
    let mut data = Vec::with_capacity(input.channels * out_h * out_w);
    for plane in input.data.chunks_exact(input.plane_len()) {
        for y in 0..out_h {
            let row = &plane[(y / 2) * input.width..(y / 2 + 1) * input.width];
            for value in row {
                data.push(*value);
                data.push(*value);
            }
        }
    }
    FaceTensor::new(input.channels, out_h, out_w, data)
}

pub(crate) fn add(left: &FaceTensor, right: &FaceTensor) -> Result<FaceTensor, String> {
    if (left.channels, left.height, left.width) != (right.channels, right.height, right.width) {
        return Err("face add operands have different shapes".to_string());
    }
    let data = left.data.iter().zip(&right.data).map(|(a, b)| a + b);
    FaceTensor::new(left.channels, left.height, left.width, data.collect())
}

/// Row-major `[outputs, inputs]` weights applied to the flattened CHW input.
pub(crate) struct FaceLinearLayer {
    pub(crate) outputs: usize,
    pub(crate) inputs: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

pub(crate) fn linear(input: &FaceTensor, layer: &FaceLinearLayer) -> Result<FaceTensor, String> {
    if input.data.len() != layer.inputs {
        return Err(format!(
            "face linear layer expects {} inputs, got {}",
            layer.inputs,
            input.data.len()
        ));
    }
    let data = layer
        .weight
        .par_chunks(layer.inputs)
        .zip(&layer.bias)
        .map(|(row, bias)| bias + row.iter().zip(&input.data).map(|(w, x)| w * x).sum::<f32>())
        .collect();
    FaceTensor::new(layer.outputs, 1, 1, data)
}

#[cfg(test)]
mod tests {
    use super::{
        FaceActivation, FaceConvLayer, FaceLinearLayer, FaceTensor, add, conv2d, linear, max_pool2,
        upsample2,
    };

    fn ramp(channels: usize, height: usize, width: usize) -> FaceTensor {
        let data = (0..channels * height * width).map(|v| v as f32).collect();
        FaceTensor::new(channels, height, width, data).unwrap()
    }

    fn layer(
        out_channels: usize,
        in_per_group: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        weight: Vec<f32>,
    ) -> FaceConvLayer {
        FaceConvLayer {
            out_channels,
            in_per_group,
            kernel,
            stride,
            pad,
            groups,
            activation: FaceActivation::None,
            weight,
            bias: vec![0.0; out_channels],
            slope: Vec::new(),
        }
    }

    /// Straightforward reference used to cross-check the row-wise implementation.
    fn naive(input: &FaceTensor, l: &FaceConvLayer) -> Vec<f32> {
        let out_h = (input.height + 2 * l.pad - l.kernel) / l.stride + 1;
        let out_w = (input.width + 2 * l.pad - l.kernel) / l.stride + 1;
        let out_per_group = l.out_channels / l.groups;
        let mut out = Vec::new();
        for o in 0..l.out_channels {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = l.bias[o];
                    for i in 0..l.in_per_group {
                        let c = (o / out_per_group) * l.in_per_group + i;
                        for ky in 0..l.kernel {
                            for kx in 0..l.kernel {
                                let y = (oy * l.stride + ky) as isize - l.pad as isize;
                                let x = (ox * l.stride + kx) as isize - l.pad as isize;
                                if y < 0
                                    || x < 0
                                    || y >= input.height as isize
                                    || x >= input.width as isize
                                {
                                    continue;
                                }
                                let w = l.weight
                                    [((o * l.in_per_group + i) * l.kernel + ky) * l.kernel + kx];
                                sum += w * input.data
                                    [(c * input.height + y as usize) * input.width + x as usize];
                            }
                        }
                    }
                    out.push(sum);
                }
            }
        }
        out
    }

    #[test]
    fn box_filter_sums_the_padded_neighbourhood() {
        let input = FaceTensor::new(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = conv2d(&input, &layer(1, 1, 3, 1, 1, 1, vec![1.0; 9])).unwrap();
        assert_eq!(output.data, vec![10.0; 4]);
    }

    #[test]
    fn convolution_matches_naive_reference_across_shapes() {
        for (kernel, stride, pad, groups, height, width) in [
            (3, 1, 1, 1, 5, 7),
            (3, 2, 1, 1, 6, 6),
            (3, 2, 1, 1, 7, 5),
            (3, 2, 1, 4, 7, 9),
            (3, 1, 1, 4, 4, 4),
            (1, 1, 0, 1, 3, 5),
            (1, 1, 0, 2, 3, 5),
        ] {
            let input = ramp(4, height, width);
            let in_per_group = 4 / groups;
            let out_channels = 4;
            let weights = out_channels * in_per_group * kernel * kernel;
            let mut l = layer(
                out_channels,
                in_per_group,
                kernel,
                stride,
                pad,
                groups,
                (0..weights)
                    .map(|v| ((v * 7 % 11) as f32 - 5.0) * 0.25)
                    .collect(),
            );
            l.bias = vec![0.5, -1.0, 2.0, 0.0];
            l.validate().unwrap();
            let output = conv2d(&input, &l).unwrap();
            let expected = naive(&input, &l);
            assert_eq!(output.data.len(), expected.len());
            for (got, want) in output.data.iter().zip(&expected) {
                assert!(
                    (got - want).abs() <= 1e-3,
                    "k={kernel} s={stride} g={groups}"
                );
            }
        }
    }

    #[test]
    fn activations_apply_per_channel() {
        let input = FaceTensor::new(1, 1, 2, vec![-2.0, 3.0]).unwrap();
        let mut l = layer(2, 1, 1, 1, 0, 1, vec![1.0, 1.0]);
        l.activation = FaceActivation::Relu;
        assert_eq!(conv2d(&input, &l).unwrap().data, vec![0.0, 3.0, 0.0, 3.0]);
        l.activation = FaceActivation::PRelu;
        l.slope = vec![0.5, 0.25];
        assert_eq!(conv2d(&input, &l).unwrap().data, vec![-1.0, 3.0, -0.5, 3.0]);
    }

    #[test]
    fn convolution_rejects_channel_mismatch() {
        let l = layer(1, 3, 1, 1, 0, 1, vec![1.0; 3]);
        assert!(conv2d(&ramp(2, 2, 2), &l).is_err());
    }

    #[test]
    fn pooling_upsampling_and_add() {
        let input = ramp(1, 2, 4);
        let pooled = max_pool2(&input).unwrap();
        assert_eq!(pooled.data, vec![5.0, 7.0]);
        let up = upsample2(&pooled).unwrap();
        assert_eq!((up.height, up.width), (2, 4));
        assert_eq!(up.data, vec![5.0, 5.0, 7.0, 7.0, 5.0, 5.0, 7.0, 7.0]);
        assert_eq!(add(&up, &input).unwrap().data[7], 14.0);
        assert!(add(&up, &pooled).is_err());
    }

    #[test]
    fn linear_flattens_channel_major() {
        let l = FaceLinearLayer {
            outputs: 2,
            inputs: 4,
            weight: vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            bias: vec![0.5, 0.0],
        };
        let output = linear(&ramp(1, 2, 2), &l).unwrap();
        assert_eq!(output.data, vec![0.5, 6.0]);
    }
}
