#![warn(missing_docs)]
//! Statically ranked, contiguous f32 tensors and eager CPU reference operators.

use std::marker::PhantomData;
use titan_hal::Backend;

/// A supported tensor element type.
pub trait DType: Copy + Default + Send + Sync + 'static {}
impl DType for f32 {}

/// A contiguous row-major tensor whose storage is allocated by `B`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<B: Backend, T: DType, const D: usize> {
    data: B::Storage<T>,
    shape: [usize; D],
    _backend: PhantomData<B>,
}

/// Parameters for NCHW convolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dOptions {
    /// Vertical and horizontal stride.
    pub stride: [usize; 2],
    /// Symmetric vertical and horizontal padding.
    pub padding: [usize; 2],
    /// Vertical and horizontal kernel dilation.
    pub dilation: [usize; 2],
    /// Number of channel groups.
    pub groups: usize,
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 }
    }
}

impl<B: Backend, const D: usize> Tensor<B, f32, D> {
    /// Constructs a contiguous tensor from row-major data.
    pub fn from_vec(backend: B, shape: [usize; D], data: Vec<f32>) -> Result<Self, TensorError> {
        let expected = element_count(&shape);
        if data.len() != expected {
            return Err(TensorError::ElementCount { expected, actual: data.len() });
        }
        Ok(Self { data: backend.allocate(data), shape, _backend: PhantomData })
    }

    /// Allocates a zero-filled contiguous tensor.
    pub fn zeros(backend: B, shape: [usize; D]) -> Self {
        Self::from_vec(backend, shape, vec![0.0; element_count(&shape)]).expect("shape controls allocation length")
    }

    /// Returns the fixed-rank shape.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }
    /// Returns the contiguous row-major storage.
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_ref()
    }
    /// Returns mutable contiguous row-major storage.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data.as_mut()
    }

    /// Reshapes contiguous storage without changing element order.
    pub fn reshape<const ND: usize>(&self, backend: B, shape: [usize; ND]) -> Result<Tensor<B, f32, ND>, TensorError> {
        let actual = self.as_slice().len();
        let expected = element_count(&shape);
        if actual != expected {
            return Err(TensorError::ElementCount { expected, actual });
        }
        Tensor::from_vec(backend, shape, self.as_slice().to_vec())
    }

    /// Returns a contiguous tensor with axes reordered by `order`.
    pub fn permute(&self, backend: B, order: [usize; D]) -> Result<Self, TensorError> {
        validate_permutation(&order)?;
        let output_shape = std::array::from_fn(|axis| self.shape[order[axis]]);
        let input_strides = strides(&self.shape);
        let output_strides = strides(&output_shape);
        let mut output = vec![0.0; self.as_slice().len()];
        for (output_index, value) in output.iter_mut().enumerate() {
            let mut input_index = 0;
            for output_axis in 0..D {
                let coordinate = if output_shape[output_axis] == 0 {
                    0
                }
                else {
                    output_index / output_strides[output_axis] % output_shape[output_axis]
                };
                input_index += coordinate * input_strides[order[output_axis]];
            }
            *value = self.as_slice()[input_index];
        }
        Self::from_vec(backend, output_shape, output)
    }

    /// Adds two tensors using NumPy-style broadcasting for equal ranks.
    pub fn add(&self, rhs: &Self, backend: B) -> Result<Self, TensorError> {
        self.elementwise(rhs, backend, "add", |left, right| left + right)
    }
    /// Multiplies two tensors using NumPy-style broadcasting for equal ranks.
    pub fn mul(&self, rhs: &Self, backend: B) -> Result<Self, TensorError> {
        self.elementwise(rhs, backend, "mul", |left, right| left * right)
    }
    /// Applies the SiLU activation, `x / (1 + exp(-x))`.
    pub fn silu(&self, backend: B) -> Result<Self, TensorError> {
        Self::from_vec(backend, self.shape, self.as_slice().iter().map(|value| value / (1.0 + (-value).exp())).collect())
    }

    /// Applies softmax over an axis.
    pub fn softmax(&self, backend: B, axis: usize) -> Result<Self, TensorError> {
        if axis >= D {
            return Err(TensorError::AxisOutOfBounds { operation: "softmax", axis, rank: D });
        }
        let axis_len = self.shape[axis];
        if axis_len == 0 {
            return Err(TensorError::ZeroDimension { operation: "softmax", axis });
        }
        let inner = self.shape[axis + 1..].iter().product::<usize>();
        let outer = self.shape[..axis].iter().product::<usize>();
        let mut output = vec![0.0; self.as_slice().len()];
        for group in 0..outer * inner {
            let outer_index = group / inner;
            let inner_index = group % inner;
            let base = outer_index * axis_len * inner + inner_index;
            let maximum = (0..axis_len).map(|index| self.as_slice()[base + index * inner]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..axis_len).map(|index| (self.as_slice()[base + index * inner] - maximum).exp()).sum();
            for index in 0..axis_len {
                output[base + index * inner] = (self.as_slice()[base + index * inner] - maximum).exp() / sum;
            }
        }
        Self::from_vec(backend, self.shape, output)
    }

    /// Normalizes the final dimension and optionally applies per-feature affine values.
    pub fn layer_norm(
        &self,
        backend: B,
        epsilon: f32,
        weight: Option<&Tensor<B, f32, 1>>,
        bias: Option<&Tensor<B, f32, 1>>,
    ) -> Result<Self, TensorError> {
        let features = self.shape[D - 1];
        if features == 0 {
            return Err(TensorError::ZeroDimension { operation: "layer_norm", axis: D - 1 });
        }
        validate_epsilon(epsilon, "layer_norm")?;
        validate_affine("layer_norm weight", weight, features)?;
        validate_affine("layer_norm bias", bias, features)?;
        let mut output = vec![0.0; self.as_slice().len()];
        for (input, output) in self.as_slice().chunks_exact(features).zip(output.chunks_exact_mut(features)) {
            let mean = input.iter().sum::<f32>() / features as f32;
            let variance = input.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / features as f32;
            for (index, value) in input.iter().enumerate() {
                output[index] = (value - mean) / (variance + epsilon).sqrt()
                    * weight.map_or(1.0, |tensor| tensor.as_slice()[index])
                    + bias.map_or(0.0, |tensor| tensor.as_slice()[index]);
            }
        }
        Self::from_vec(backend, self.shape, output)
    }

    fn elementwise(
        &self,
        rhs: &Self,
        backend: B,
        operation: &'static str,
        apply: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let mut shape = [0; D];
        for axis in 0..D {
            let (left, right) = (self.shape[axis], rhs.shape[axis]);
            if left != right && left != 1 && right != 1 {
                return Err(TensorError::BroadcastShape { operation, left: self.shape.to_vec(), right: rhs.shape.to_vec() });
            }
            shape[axis] = left.max(right);
        }
        let output_strides = strides(&shape);
        let left_strides = strides(&self.shape);
        let right_strides = strides(&rhs.shape);
        let output = (0..element_count(&shape))
            .map(|index| {
                let mut left_index = 0;
                let mut right_index = 0;
                for axis in 0..D {
                    let coordinate = if shape[axis] == 0 { 0 } else { index / output_strides[axis] % shape[axis] };
                    left_index += if self.shape[axis] == 1 { 0 } else { coordinate * left_strides[axis] };
                    right_index += if rhs.shape[axis] == 1 { 0 } else { coordinate * right_strides[axis] };
                }
                apply(self.as_slice()[left_index], rhs.as_slice()[right_index])
            })
            .collect();
        Self::from_vec(backend, shape, output)
    }
}

impl<B: Backend, const D: usize> Tensor<B, f32, D> {
    /// Concatenates tensors along `axis`.
    pub fn concat(backend: B, axis: usize, tensors: &[&Self]) -> Result<Self, TensorError> {
        let first = tensors.first().ok_or(TensorError::EmptyInput { operation: "concat" })?;
        if axis >= D {
            return Err(TensorError::AxisOutOfBounds { operation: "concat", axis, rank: D });
        }
        let mut shape = first.shape;
        shape[axis] = 0;
        for tensor in tensors {
            for dimension in 0..D {
                if dimension != axis && tensor.shape[dimension] != first.shape[dimension] {
                    return Err(TensorError::ConcatShape {
                        axis,
                        expected: first.shape.to_vec(),
                        actual: tensor.shape.to_vec(),
                    });
                }
            }
            shape[axis] += tensor.shape[axis];
        }
        let outer = first.shape[..axis].iter().product::<usize>();
        let inner = first.shape[axis + 1..].iter().product::<usize>();
        let mut output = Vec::with_capacity(element_count(&shape));
        for outer_index in 0..outer {
            for tensor in tensors {
                let width = tensor.shape[axis] * inner;
                let start = outer_index * width;
                output.extend_from_slice(&tensor.as_slice()[start..start + width]);
            }
        }
        Self::from_vec(backend, shape, output)
    }
}

impl<B: Backend> Tensor<B, f32, 4> {
    /// Performs NCHW cross-correlation with OIHW weights.
    pub fn conv2d(
        &self,
        backend: B,
        weight: &Self,
        bias: Option<&Tensor<B, f32, 1>>,
        options: Conv2dOptions,
    ) -> Result<Self, TensorError> {
        let [batch, channels, height, width] = self.shape;
        let [output_channels, weight_channels, kernel_height, kernel_width] = weight.shape;
        if options.stride.contains(&0) || options.dilation.contains(&0) || options.groups == 0 {
            return Err(TensorError::InvalidParameter {
                operation: "conv2d",
                detail: "stride, dilation, and groups must be non-zero",
            });
        }
        if channels % options.groups != 0
            || output_channels % options.groups != 0
            || weight_channels != channels / options.groups
        {
            return Err(TensorError::Conv2dShape { input: self.shape, weight: weight.shape, groups: options.groups });
        }
        validate_affine("conv2d bias", bias, output_channels)?;
        let extent_height = options.dilation[0] * (kernel_height.saturating_sub(1)) + 1;
        let extent_width = options.dilation[1] * (kernel_width.saturating_sub(1)) + 1;
        if height + 2 * options.padding[0] < extent_height || width + 2 * options.padding[1] < extent_width {
            return Err(TensorError::InvalidParameter { operation: "conv2d", detail: "kernel exceeds padded input" });
        }
        let output_height = (height + 2 * options.padding[0] - extent_height) / options.stride[0] + 1;
        let output_width = (width + 2 * options.padding[1] - extent_width) / options.stride[1] + 1;
        let mut output = vec![0.0; batch * output_channels * output_height * output_width];
        for n in 0..batch {
            for oc in 0..output_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let group = oc / (output_channels / options.groups);
                        let mut value = bias.map_or(0.0, |tensor| tensor.as_slice()[oc]);
                        for ic in 0..weight_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let input_y = oh * options.stride[0] + kh * options.dilation[0];
                                    let input_x = ow * options.stride[1] + kw * options.dilation[1];
                                    if input_y >= options.padding[0] && input_x >= options.padding[1] {
                                        let y = input_y - options.padding[0];
                                        let x = input_x - options.padding[1];
                                        if y < height && x < width {
                                            value += self.as_slice()
                                                [((n * channels + group * weight_channels + ic) * height + y) * width + x]
                                                * weight.as_slice()
                                                    [((oc * weight_channels + ic) * kernel_height + kh) * kernel_width + kw];
                                        }
                                    }
                                }
                            }
                        }
                        output[((n * output_channels + oc) * output_height + oh) * output_width + ow] = value;
                    }
                }
            }
        }
        Self::from_vec(backend, [batch, output_channels, output_height, output_width], output)
    }

    /// Applies GroupNorm over each NCHW sample and channel group.
    pub fn group_norm(
        &self,
        backend: B,
        groups: usize,
        epsilon: f32,
        weight: Option<&Tensor<B, f32, 1>>,
        bias: Option<&Tensor<B, f32, 1>>,
    ) -> Result<Self, TensorError> {
        let [batch, channels, height, width] = self.shape;
        if groups == 0 || channels % groups != 0 {
            return Err(TensorError::GroupNormShape { channels, groups });
        }
        validate_epsilon(epsilon, "group_norm")?;
        validate_affine("group_norm weight", weight, channels)?;
        validate_affine("group_norm bias", bias, channels)?;
        let per_group = channels / groups * height * width;
        let mut output = vec![0.0; self.as_slice().len()];
        for n in 0..batch {
            for group in 0..groups {
                let start = (n * channels + group * (channels / groups)) * height * width;
                let input = &self.as_slice()[start..start + per_group];
                let mean = input.iter().sum::<f32>() / per_group as f32;
                let variance = input.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / per_group as f32;
                for local in 0..per_group {
                    let channel = group * (channels / groups) + local / (height * width);
                    output[start + local] = (input[local] - mean) / (variance + epsilon).sqrt()
                        * weight.map_or(1.0, |tensor| tensor.as_slice()[channel])
                        + bias.map_or(0.0, |tensor| tensor.as_slice()[channel]);
                }
            }
        }
        Self::from_vec(backend, self.shape, output)
    }

    /// Resizes NCHW spatial dimensions by nearest-neighbor sampling.
    pub fn resize_nearest2d(&self, backend: B, output_height: usize, output_width: usize) -> Result<Self, TensorError> {
        let [batch, channels, height, width] = self.shape;
        if height == 0 || width == 0 || output_height == 0 || output_width == 0 {
            return Err(TensorError::InvalidParameter {
                operation: "resize_nearest2d",
                detail: "input and output spatial dimensions must be non-zero",
            });
        }
        let mut output = vec![0.0; batch * channels * output_height * output_width];
        for n in 0..batch {
            for c in 0..channels {
                for y in 0..output_height {
                    for x in 0..output_width {
                        output[((n * channels + c) * output_height + y) * output_width + x] = self.as_slice()
                            [((n * channels + c) * height + y * height / output_height) * width + x * width / output_width];
                    }
                }
            }
        }
        Self::from_vec(backend, [batch, channels, output_height, output_width], output)
    }
}

impl<B: Backend> Tensor<B, f32, 2> {
    /// Computes a tiled matrix product.
    pub fn matmul(&self, rhs: &Self, backend: B, tile: usize) -> Result<Self, TensorError> {
        let [m, k] = self.shape;
        let [rk, n] = rhs.shape;
        if k != rk {
            return Err(TensorError::MatmulShape { left: self.shape, right: rhs.shape });
        }
        let mut output = vec![0.0; m * n];
        let tile = tile.max(1);
        for ii in (0..m).step_by(tile) {
            for kk in (0..k).step_by(tile) {
                for jj in (0..n).step_by(tile) {
                    for i in ii..(ii + tile).min(m) {
                        for p in kk..(kk + tile).min(k) {
                            let left = self.as_slice()[i * k + p];
                            for j in jj..(jj + tile).min(n) {
                                output[i * n + j] += left * rhs.as_slice()[p * n + j];
                            }
                        }
                    }
                }
            }
        }
        Self::from_vec(backend, [m, n], output)
    }
}

/// Reports invalid tensor shapes or scalar operator parameters.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Input storage does not match its requested shape.
    ElementCount { expected: usize, actual: usize },
    /// An axis is invalid for the tensor rank.
    AxisOutOfBounds { operation: &'static str, axis: usize, rank: usize },
    /// An axis that must be non-empty has length zero.
    ZeroDimension { operation: &'static str, axis: usize },
    /// A permutation repeats or omits an axis.
    InvalidPermutation { order: Vec<usize> },
    /// Equal-rank shapes cannot be broadcast.
    BroadcastShape { operation: &'static str, left: Vec<usize>, right: Vec<usize> },
    /// Concatenation shapes differ outside its selected axis.
    ConcatShape { axis: usize, expected: Vec<usize>, actual: Vec<usize> },
    /// An input collection is empty.
    EmptyInput { operation: &'static str },
    /// Matrices have incompatible inner dimensions.
    MatmulShape { left: [usize; 2], right: [usize; 2] },
    /// Convolution channels, grouping, or OIHW weight channels are incompatible.
    Conv2dShape { input: [usize; 4], weight: [usize; 4], groups: usize },
    /// GroupNorm channel count is incompatible with group count.
    GroupNormShape { channels: usize, groups: usize },
    /// An affine tensor has an unexpected feature count.
    AffineShape { operation: &'static str, expected: usize, actual: usize },
    /// An operator scalar parameter is invalid.
    InvalidParameter { operation: &'static str, detail: &'static str },
}
impl std::fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "tensor error: {self:?}")
    }
}
impl std::error::Error for TensorError {}

/// Computes gradients for `sum((x * w)^2)`.
pub fn squared_matmul_grad<B: Backend>(
    x: &Tensor<B, f32, 2>,
    w: &Tensor<B, f32, 2>,
    backend: B,
) -> Result<Tensor<B, f32, 2>, TensorError> {
    let y = x.matmul(w, backend.clone(), 16)?;
    let [batch, input] = x.shape();
    let [_, output] = w.shape();
    let mut grad = vec![0.0; input * output];
    for b in 0..batch {
        for i in 0..input {
            for o in 0..output {
                grad[i * output + o] += 2.0 * x.as_slice()[b * input + i] * y.as_slice()[b * output + o];
            }
        }
    }
    Tensor::from_vec(backend, [input, output], grad)
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}
fn strides<const D: usize>(shape: &[usize; D]) -> [usize; D] {
    let mut result = [1; D];
    for axis in (0..D).rev().skip(1) {
        result[axis] = result[axis + 1] * shape[axis + 1];
    }
    result
}
fn validate_permutation<const D: usize>(order: &[usize; D]) -> Result<(), TensorError> {
    let mut seen = [false; D];
    for &axis in order {
        if axis >= D || seen[axis] {
            return Err(TensorError::InvalidPermutation { order: order.to_vec() });
        }
        seen[axis] = true;
    }
    Ok(())
}
fn validate_epsilon(epsilon: f32, operation: &'static str) -> Result<(), TensorError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(TensorError::InvalidParameter { operation, detail: "epsilon must be finite and non-negative" });
    }
    Ok(())
}
fn validate_affine<B: Backend>(
    operation: &'static str,
    tensor: Option<&Tensor<B, f32, 1>>,
    expected: usize,
) -> Result<(), TensorError> {
    if let Some(tensor) = tensor {
        if tensor.shape[0] != expected {
            return Err(TensorError::AffineShape { operation, expected, actual: tensor.shape[0] });
        }
    }
    Ok(())
}
