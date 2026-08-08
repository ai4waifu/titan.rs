use titan_hal::Cpu;
use titan_tensor::{Conv2dOptions, Tensor, TensorError, squared_matmul_grad};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "actual={actual}, expected={expected}");
    }
}

#[test]
fn matmul_and_grad_work() {
    let x = Tensor::from_vec(Cpu, [1, 2], vec![1., 2.]).unwrap();
    let weights = Tensor::from_vec(Cpu, [2, 1], vec![3., 4.]).unwrap();
    assert_eq!(x.matmul(&weights, Cpu, 2).unwrap().as_slice(), &[11.]);
    assert_eq!(squared_matmul_grad(&x, &weights, Cpu).unwrap().as_slice(), &[22., 44.]);
}

#[test]
fn structural_and_elementwise_ops_are_contiguous_and_broadcast() {
    let input = Tensor::from_vec(Cpu, [2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let reshaped = input.reshape(Cpu, [3, 2]).unwrap();
    assert_eq!(reshaped.shape(), [3, 2]);
    let transposed = input.permute(Cpu, [1, 0]).unwrap();
    assert_eq!(transposed.as_slice(), &[1., 4., 2., 5., 3., 6.]);
    let bias = Tensor::from_vec(Cpu, [1, 3], vec![10., 20., 30.]).unwrap();
    assert_eq!(input.add(&bias, Cpu).unwrap().as_slice(), &[11., 22., 33., 14., 25., 36.]);
    assert_eq!(input.mul(&bias, Cpu).unwrap().as_slice(), &[10., 40., 90., 40., 100., 180.]);
    let joined = Tensor::concat(Cpu, 0, &[&input, &input]).unwrap();
    assert_eq!(joined.shape(), [4, 3]);
    assert_eq!(joined.as_slice(), &[1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.]);
    assert_close(&input.silu(Cpu).unwrap().as_slice()[..2], &[0.7310586, 1.761594]);
}

#[test]
fn normalization_and_softmax_match_reference_values() {
    let input = Tensor::from_vec(Cpu, [2, 2], vec![1., 3., 2., 4.]).unwrap();
    assert_close(input.softmax(Cpu, 1).unwrap().as_slice(), &[0.11920292, 0.880797, 0.11920292, 0.880797]);
    assert_close(input.layer_norm(Cpu, 0.0, None, None).unwrap().as_slice(), &[-1., 1., -1., 1.]);
    let weight = Tensor::from_vec(Cpu, [2], vec![2., 3.]).unwrap();
    let bias = Tensor::from_vec(Cpu, [2], vec![1., -1.]).unwrap();
    assert_close(input.layer_norm(Cpu, 0.0, Some(&weight), Some(&bias)).unwrap().as_slice(), &[-1., 2., -1., 2.]);
    let group_input = Tensor::from_vec(Cpu, [1, 2, 1, 2], vec![1., 3., 2., 4.]).unwrap();
    assert_close(group_input.group_norm(Cpu, 2, 0.0, None, None).unwrap().as_slice(), &[-1., 1., -1., 1.]);
}

#[test]
fn nchw_conv_and_nearest_resize_match_reference_values() {
    let input = Tensor::from_vec(Cpu, [1, 1, 3, 3], (1..=9).map(|value| value as f32).collect()).unwrap();
    let weight = Tensor::from_vec(Cpu, [1, 1, 2, 2], vec![1., 0., 0., 1.]).unwrap();
    let bias = Tensor::from_vec(Cpu, [1], vec![1.]).unwrap();
    let output = input.conv2d(Cpu, &weight, Some(&bias), Conv2dOptions::default()).unwrap();
    assert_eq!(output.shape(), [1, 1, 2, 2]);
    assert_eq!(output.as_slice(), &[7., 9., 13., 15.]);
    let small = Tensor::from_vec(Cpu, [1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
    assert_eq!(small.resize_nearest2d(Cpu, 3, 3).unwrap().as_slice(), &[1., 1., 2., 1., 1., 2., 3., 3., 4.]);
}

#[test]
fn invalid_shapes_report_the_operation_and_values() {
    let left = Tensor::zeros(Cpu, [2, 3]);
    let right = Tensor::zeros(Cpu, [2, 2]);
    assert_eq!(
        left.add(&right, Cpu),
        Err(TensorError::BroadcastShape { operation: "add", left: vec![2, 3], right: vec![2, 2] })
    );
    assert_eq!(left.permute(Cpu, [0, 0]), Err(TensorError::InvalidPermutation { order: vec![0, 0] }));
    assert_eq!(left.softmax(Cpu, 2), Err(TensorError::AxisOutOfBounds { operation: "softmax", axis: 2, rank: 2 }));
    let input = Tensor::zeros(Cpu, [1, 3, 1, 1]);
    let weight = Tensor::zeros(Cpu, [2, 2, 1, 1]);
    assert_eq!(
        input.conv2d(Cpu, &weight, None, Conv2dOptions { groups: 2, ..Default::default() }),
        Err(TensorError::Conv2dShape { input: [1, 3, 1, 1], weight: [2, 2, 1, 1], groups: 2 })
    );
    let incompatible_concat = Tensor::zeros(Cpu, [1, 4]);
    assert_eq!(
        Tensor::concat(Cpu, 0, &[&left, &incompatible_concat]),
        Err(TensorError::ConcatShape { axis: 0, expected: vec![2, 3], actual: vec![1, 4] })
    );
    assert_eq!(input.group_norm(Cpu, 2, 1e-5, None, None), Err(TensorError::GroupNormShape { channels: 3, groups: 2 }));
    assert_eq!(
        input.resize_nearest2d(Cpu, 0, 1),
        Err(TensorError::InvalidParameter {
            operation: "resize_nearest2d",
            detail: "input and output spatial dimensions must be non-zero"
        })
    );
    let bad_affine = Tensor::zeros(Cpu, [2]);
    assert_eq!(
        left.layer_norm(Cpu, 1e-5, Some(&bad_affine), None),
        Err(TensorError::AffineShape { operation: "layer_norm weight", expected: 3, actual: 2 })
    );
}
