use std::{collections::BTreeMap, sync::Arc};

use titan_backend_cpu::CpuDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor, TensorHandle};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn device() -> Device {
    let driver = Arc::new(CpuDriver);
    Device::from_session(driver.open(driver.enumerate().unwrap()[0].device).unwrap())
}

fn strides(shape: &[u64]) -> Vec<i64> {
    let mut result = vec![0; shape.len()];
    let mut step = 1i64;
    for axis in (0..shape.len()).rev() {
        result[axis] = step;
        step *= shape[axis] as i64;
    }
    result
}

fn request(operator: &str, inputs: Vec<TensorHandle>, shape: Vec<u64>, attrs: AttrMap) -> OpRequest {
    OpRequest {
        operator: OperatorId(operator.into()),
        inputs,
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(strides(&shape)),
            shape: Shape(shape),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: SourceSpan { file: "cpu_reference_ops.rs".into(), line: 1, column: 1 },
    }
}

fn attrs(values: &[(&str, AttrValue)]) -> AttrMap {
    values.iter().map(|(key, value)| ((*key).into(), value.clone())).collect::<BTreeMap<_, _>>()
}

fn float(value: f64) -> AttrValue {
    AttrValue::Float(value.to_bits())
}

fn run(request: OpRequest) -> Result<Vec<f32>, titan_runtime::ExecutionError> {
    Runtime::open(std::env::temp_dir().join("titan-cpu-reference-ops"))
        .execute(request)?
        .wait()
        .map(|result| result.outputs[0].to_vec_f32().unwrap())
}

#[test]
fn cpu_reference_gemm_is_numerically_correct_and_reads_back_a_tensor_buffer() {
    let d = device();
    let lhs = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let rhs = Tensor::<f32, 2>::from_slice(&d, [3, 2], &[7., 8., 9., 10., 11., 12.]).unwrap();
    assert_eq!(
        run(request("gemm", vec![lhs.handle(), rhs.handle()], vec![2, 2], AttrMap::new())).unwrap(),
        vec![58., 64., 139., 154.]
    );

    let transposed_lhs = Tensor::<f32, 2>::from_slice(&d, [3, 2], &[1., 4., 2., 5., 3., 6.]).unwrap();
    let transposed_rhs = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[7., 9., 11., 8., 10., 12.]).unwrap();
    assert_eq!(
        run(request(
            "gemm",
            vec![transposed_lhs.handle(), transposed_rhs.handle()],
            vec![2, 2],
            attrs(&[("transpose_lhs", AttrValue::Bool(true)), ("transpose_rhs", AttrValue::Bool(true))]),
        ))
        .unwrap(),
        vec![58., 64., 139., 154.]
    );
}

#[test]
fn cpu_reference_gemm_rejects_invalid_contracts() {
    let d = device();
    let lhs = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let rhs = Tensor::<f32, 2>::from_slice(&d, [2, 2], &[1., 2., 3., 4.]).unwrap();
    assert!(
        run(request("gemm", vec![lhs.handle(), rhs.handle()], vec![2, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("inner dimensions")
    );
    let vector = Tensor::<f32, 1>::from_slice(&d, [3], &[1., 2., 3.]).unwrap();
    assert!(
        run(request("gemm", vec![vector.handle(), rhs.handle()], vec![2, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("rank-2")
    );
    let compatible_rhs = Tensor::<f32, 2>::from_slice(&d, [3, 2], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert!(
        run(request("gemm", vec![lhs.handle(), compatible_rhs.handle()], vec![3, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("output shape")
    );
    let mut wrong_dtype = request("gemm", vec![lhs.handle(), compatible_rhs.handle()], vec![2, 2], AttrMap::new());
    wrong_dtype.outputs[0].dtype = DType::I32;
    assert!(run(wrong_dtype).unwrap_err().message.contains("F32"));
    let mut wrong_layout = request("gemm", vec![lhs.handle(), compatible_rhs.handle()], vec![2, 2], AttrMap::new());
    wrong_layout.outputs[0].layout = Layout::Strided;
    assert!(run(wrong_layout).unwrap_err().message.contains("contiguous"));
    assert!(run(request("gemm", vec![lhs.handle()], vec![2, 2], AttrMap::new())).unwrap_err().message.contains("exactly two"));
}

fn conv2d_attrs(values: &[(&str, AttrValue)]) -> AttrMap {
    let mut result = attrs(&[
        ("stride_h", AttrValue::Int(1)),
        ("stride_w", AttrValue::Int(1)),
        ("pad_h", AttrValue::Int(0)),
        ("pad_w", AttrValue::Int(0)),
        ("dilation_h", AttrValue::Int(1)),
        ("dilation_w", AttrValue::Int(1)),
        ("groups", AttrValue::Int(1)),
    ]);
    result.extend(attrs(values));
    result
}

#[test]
fn cpu_reference_conv2d_is_numerically_correct_and_reads_back_a_tensor_buffer() {
    let d = device();
    let input = Tensor::<f32, 4>::from_slice(&d, [1, 1, 3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
    let weight = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1., 0., 0., -1.]).unwrap();
    let bias = Tensor::<f32, 1>::from_slice(&d, [1], &[1.]).unwrap();
    assert_eq!(
        run(request("conv2d", vec![input.handle(), weight.handle(), bias.handle()], vec![1, 1, 2, 2], conv2d_attrs(&[]),))
            .unwrap(),
        vec![-3., -3., -3., -3.]
    );

    let grouped_input = Tensor::<f32, 4>::from_slice(&d, [1, 2, 1, 1], &[2., 3.]).unwrap();
    let grouped_weight = Tensor::<f32, 4>::from_slice(&d, [2, 1, 1, 1], &[10., 20.]).unwrap();
    assert_eq!(
        run(request(
            "conv2d",
            vec![grouped_input.handle(), grouped_weight.handle()],
            vec![1, 2, 1, 1],
            conv2d_attrs(&[("groups", AttrValue::Int(2))]),
        ))
        .unwrap(),
        vec![20., 60.]
    );
}

#[test]
fn cpu_reference_conv2d_rejects_invalid_contracts() {
    let d = device();
    let input = Tensor::<f32, 4>::from_slice(&d, [1, 1, 3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
    let weight = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1., 0., 0., -1.]).unwrap();
    let rank_three = Tensor::<f32, 3>::from_slice(&d, [1, 3, 3], &[1.; 9]).unwrap();
    assert!(
        run(request("conv2d", vec![rank_three.handle(), weight.handle()], vec![1, 1, 2, 2], conv2d_attrs(&[])))
            .unwrap_err()
            .message
            .contains("rank-4")
    );
    assert!(
        run(request(
            "conv2d",
            vec![input.handle(), weight.handle()],
            vec![1, 1, 2, 2],
            conv2d_attrs(&[("groups", AttrValue::Int(0))]),
        ))
        .unwrap_err()
        .message
        .contains("non-zero")
    );
    assert!(
        run(request("conv2d", vec![input.handle(), weight.handle()], vec![1, 1, 1, 2], conv2d_attrs(&[]),))
            .unwrap_err()
            .message
            .contains("output shape")
    );
    let invalid_weight = Tensor::<f32, 4>::from_slice(&d, [1, 2, 2, 2], &[1.; 8]).unwrap();
    assert!(
        run(request("conv2d", vec![input.handle(), invalid_weight.handle()], vec![1, 1, 2, 2], conv2d_attrs(&[])))
            .unwrap_err()
            .message
            .contains("channels")
    );
    let wrong_bias = Tensor::<f32, 1>::from_slice(&d, [2], &[1., 2.]).unwrap();
    assert!(
        run(
            request("conv2d", vec![input.handle(), weight.handle(), wrong_bias.handle()], vec![1, 1, 2, 2], conv2d_attrs(&[]),)
        )
        .unwrap_err()
        .message
        .contains("bias")
    );
    let mut wrong_dtype = request("conv2d", vec![input.handle(), weight.handle()], vec![1, 1, 2, 2], conv2d_attrs(&[]));
    wrong_dtype.outputs[0].dtype = DType::I32;
    assert!(run(wrong_dtype).unwrap_err().message.contains("F32"));
    let mut wrong_layout = request("conv2d", vec![input.handle(), weight.handle()], vec![1, 1, 2, 2], conv2d_attrs(&[]));
    wrong_layout.outputs[0].layout = Layout::Strided;
    assert!(run(wrong_layout).unwrap_err().message.contains("contiguous"));
}

#[test]
fn cpu_reference_scaled_dot_product_attention_supports_distinct_query_and_key_lengths() {
    let d = device();
    let query = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1., 0., 0., 1.]).unwrap();
    let key = Tensor::<f32, 4>::from_slice(&d, [1, 1, 3, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
    let value = Tensor::<f32, 4>::from_slice(&d, [1, 1, 3, 2], &[10., 0., 0., 20., 10., 10.]).unwrap();
    let output = run(request(
        "scaled_dot_product_attention",
        vec![query.handle(), key.handle(), value.handle()],
        vec![1, 1, 2, 2],
        AttrMap::new(),
    ))
    .unwrap();
    let high = std::f32::consts::FRAC_1_SQRT_2.exp();
    let high_weight = high / (2.0 * high + 1.0);
    let low_weight = 1.0 / (2.0 * high + 1.0);
    let expected =
        [20.0 * high_weight, 20.0 * low_weight + 10.0 * high_weight, 10.0 * (low_weight + high_weight), 30.0 * high_weight];
    assert!(output.iter().zip(expected).all(|(actual, expected)| (actual - expected).abs() < 1e-6));
}

#[test]
fn cpu_reference_scaled_dot_product_attention_rejects_invalid_contracts() {
    let d = device();
    let query = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1., 0., 0., 1.]).unwrap();
    let key = Tensor::<f32, 4>::from_slice(&d, [1, 1, 3, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
    let value = Tensor::<f32, 4>::from_slice(&d, [1, 1, 3, 2], &[10., 0., 0., 20., 10., 10.]).unwrap();
    assert!(
        run(request("scaled_dot_product_attention", vec![query.handle(), key.handle()], vec![1, 1, 2, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("exactly three")
    );
    let rank_three = Tensor::<f32, 3>::from_slice(&d, [1, 2, 2], &[1.; 4]).unwrap();
    assert!(
        run(request(
            "scaled_dot_product_attention",
            vec![rank_three.handle(), key.handle(), value.handle()],
            vec![1, 1, 2, 2],
            AttrMap::new(),
        ))
        .unwrap_err()
        .message
        .contains("rank-4")
    );
    let mismatched_value = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1.; 4]).unwrap();
    assert!(
        run(request(
            "scaled_dot_product_attention",
            vec![query.handle(), key.handle(), mismatched_value.handle()],
            vec![1, 1, 2, 2],
            AttrMap::new(),
        ))
        .unwrap_err()
        .message
        .contains("K/V sequence")
    );
    assert!(
        run(request(
            "scaled_dot_product_attention",
            vec![query.handle(), key.handle(), value.handle()],
            vec![1, 1, 3, 2],
            AttrMap::new(),
        ))
        .unwrap_err()
        .message
        .contains("output shape")
    );
    assert!(
        run(request(
            "scaled_dot_product_attention",
            vec![query.handle(), key.handle(), value.handle()],
            vec![1, 1, 2, 2],
            attrs(&[("causal", AttrValue::Bool(true))]),
        ))
        .unwrap_err()
        .message
        .contains("not implemented")
    );
    let second_device = device();
    let foreign_key = Tensor::<f32, 4>::from_slice(&second_device, [1, 1, 3, 2], &[1., 0., 0., 1., 1., 1.]).unwrap();
    let foreign_value = Tensor::<f32, 4>::from_slice(&second_device, [1, 1, 3, 2], &[10., 0., 0., 20., 10., 10.]).unwrap();
    assert!(
        run(request(
            "scaled_dot_product_attention",
            vec![query.handle(), foreign_key.handle(), foreign_value.handle()],
            vec![1, 1, 2, 2],
            AttrMap::new(),
        ))
        .unwrap_err()
        .message
        .contains("same session")
    );
    let mut wrong_dtype = request(
        "scaled_dot_product_attention",
        vec![query.handle(), key.handle(), value.handle()],
        vec![1, 1, 2, 2],
        AttrMap::new(),
    );
    wrong_dtype.outputs[0].dtype = DType::I32;
    assert!(run(wrong_dtype).unwrap_err().message.contains("F32"));
    let mut wrong_layout = request(
        "scaled_dot_product_attention",
        vec![query.handle(), key.handle(), value.handle()],
        vec![1, 1, 2, 2],
        AttrMap::new(),
    );
    wrong_layout.outputs[0].layout = Layout::Strided;
    assert!(run(wrong_layout).unwrap_err().message.contains("contiguous"));
}

#[test]
fn cpu_reference_reshape_transpose_slice_concat_reduction_and_softmax_are_numerically_correct() {
    let d = device();
    let matrix = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert_eq!(
        run(request("reshape", vec![matrix.handle()], vec![3, 2], AttrMap::new())).unwrap(),
        vec![1., 2., 3., 4., 5., 6.]
    );
    assert_eq!(
        run(request("transpose", vec![matrix.handle()], vec![3, 2], attrs(&[("permutation", AttrValue::Ints(vec![1, 0]))])))
            .unwrap(),
        vec![1., 4., 2., 5., 3., 6.]
    );
    assert_eq!(
        run(request(
            "slice",
            vec![matrix.handle()],
            vec![2, 2],
            attrs(&[
                ("starts", AttrValue::Ints(vec![1])),
                ("ends", AttrValue::Ints(vec![3])),
                ("axes", AttrValue::Ints(vec![1]))
            ])
        ))
        .unwrap(),
        vec![2., 3., 5., 6.]
    );
    let left = Tensor::<f32, 2>::from_slice(&d, [1, 2], &[1., 2.]).unwrap();
    let right = Tensor::<f32, 2>::from_slice(&d, [1, 2], &[3., 4.]).unwrap();
    assert_eq!(
        run(request("concat", vec![left.handle(), right.handle()], vec![2, 2], attrs(&[("axis", AttrValue::Int(0))]))).unwrap(),
        vec![1., 2., 3., 4.]
    );
    assert_eq!(
        run(request("reduction.sum", vec![matrix.handle()], vec![2], attrs(&[("axes", AttrValue::Ints(vec![1]))]))).unwrap(),
        vec![6., 15.]
    );
    let softmax = run(request("softmax", vec![matrix.handle()], vec![2, 3], attrs(&[("axis", AttrValue::Int(1))]))).unwrap();
    assert!(
        (softmax[0] - 0.09003057).abs() < 1e-6
            && (softmax[2] - 0.66524094).abs() < 1e-6
            && (softmax[3] - 0.09003057).abs() < 1e-6
    );
}

#[test]
fn cpu_reference_operators_reject_invalid_shape_contracts() {
    let d = device();
    let matrix = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert!(
        run(request("reshape", vec![matrix.handle()], vec![4, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("element count")
    );
    assert!(
        run(request("transpose", vec![matrix.handle()], vec![3, 2], attrs(&[("permutation", AttrValue::Ints(vec![0, 0]))])))
            .unwrap_err()
            .message
            .contains("permutation")
    );
    assert!(
        run(request(
            "slice",
            vec![matrix.handle()],
            vec![2, 2],
            attrs(&[
                ("starts", AttrValue::Ints(vec![0])),
                ("ends", AttrValue::Ints(vec![4])),
                ("axes", AttrValue::Ints(vec![1]))
            ])
        ))
        .unwrap_err()
        .message
        .contains("bounds")
    );
    let incompatible = Tensor::<f32, 2>::from_slice(&d, [1, 2], &[1., 2.]).unwrap();
    assert!(
        run(request("concat", vec![matrix.handle(), incompatible.handle()], vec![3, 3], attrs(&[("axis", AttrValue::Int(0))])))
            .unwrap_err()
            .message
            .contains("dimensions")
    );
    assert!(
        run(request("reduction.sum", vec![matrix.handle()], vec![2], attrs(&[("axes", AttrValue::Ints(vec![2]))])))
            .unwrap_err()
            .message
            .contains("axes")
    );
    assert!(
        run(request("softmax", vec![matrix.handle()], vec![3, 2], attrs(&[("axis", AttrValue::Int(1))])))
            .unwrap_err()
            .message
            .contains("shape")
    );
}

#[test]
fn cpu_reference_operators_reject_non_f32_and_non_contiguous_outputs() {
    let d = device();
    let input = Tensor::<f32, 1>::from_slice(&d, [2], &[1., 2.]).unwrap();
    let mut wrong_dtype = request("reshape", vec![input.handle()], vec![2], AttrMap::new());
    wrong_dtype.outputs[0].dtype = DType::I32;
    assert!(run(wrong_dtype).unwrap_err().message.contains("F32"));
    let mut wrong_layout = request("reshape", vec![input.handle()], vec![2], AttrMap::new());
    wrong_layout.outputs[0].layout = Layout::Strided;
    assert!(run(wrong_layout).unwrap_err().message.contains("contiguous"));
}

#[test]
fn cpu_reference_broadcast_add_silu_gelu_and_nearest_resize_are_numerically_correct() {
    let d = device();
    let lhs = Tensor::<f32, 2>::from_slice(&d, [2, 1], &[1., 10.]).unwrap();
    let rhs = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[2., 3., 4., 5., 6., 7.]).unwrap();
    assert_eq!(
        run(request("broadcast.add", vec![lhs.handle(), rhs.handle()], vec![2, 3], AttrMap::new())).unwrap(),
        vec![3., 4., 5., 15., 16., 17.]
    );

    let activations = Tensor::<f32, 1>::from_slice(&d, [3], &[-1., 0., 1.]).unwrap();
    let silu = run(request("silu", vec![activations.handle()], vec![3], AttrMap::new())).unwrap();
    assert!((silu[0] + 0.268_941_43).abs() < 1e-6 && silu[1] == 0.0 && (silu[2] - 0.731_058_6).abs() < 1e-6);
    let gelu = run(request("gelu", vec![activations.handle()], vec![3], AttrMap::new())).unwrap();
    assert!((gelu[0] + 0.158_655_26).abs() < 2e-6 && gelu[1] == 0.0 && (gelu[2] - 0.841_344_7).abs() < 2e-6);

    let image = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    assert_eq!(
        run(request("resize.nearest2d", vec![image.handle()], vec![1, 1, 3, 5], AttrMap::new())).unwrap(),
        vec![1., 1., 1., 2., 2., 1., 1., 1., 2., 2., 3., 3., 3., 4., 4.]
    );
}

#[test]
fn cpu_reference_broadcast_activation_and_resize_reject_invalid_contracts() {
    let d = device();
    let lhs = Tensor::<f32, 2>::from_slice(&d, [2, 2], &[1., 2., 3., 4.]).unwrap();
    let rhs = Tensor::<f32, 2>::from_slice(&d, [2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert!(
        run(request("broadcast.add", vec![lhs.handle(), rhs.handle()], vec![2, 3], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("dimensions")
    );
    assert!(
        run(request("broadcast.add", vec![lhs.handle()], vec![2, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("exactly two")
    );

    let activation = Tensor::<f32, 1>::from_slice(&d, [2], &[1., 2.]).unwrap();
    assert!(run(request("silu", vec![activation.handle()], vec![1, 2], AttrMap::new())).unwrap_err().message.contains("shape"));
    let mut gelu_dtype = request("gelu", vec![activation.handle()], vec![2], AttrMap::new());
    gelu_dtype.outputs[0].dtype = DType::I32;
    assert!(run(gelu_dtype).unwrap_err().message.contains("F32"));
    let mut gelu_layout = request("gelu", vec![activation.handle()], vec![2], AttrMap::new());
    gelu_layout.outputs[0].layout = Layout::Strided;
    assert!(run(gelu_layout).unwrap_err().message.contains("contiguous"));

    let rank_three = Tensor::<f32, 3>::from_slice(&d, [1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    assert!(
        run(request("resize.nearest2d", vec![rank_three.handle()], vec![1, 1, 2, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("rank-4")
    );
    let image = Tensor::<f32, 4>::from_slice(&d, [1, 1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    assert!(
        run(request("resize_nearest2d", vec![image.handle()], vec![2, 1, 2, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("preserve N and C")
    );
}

#[test]
fn cpu_reference_layer_and_group_norm_are_numerically_correct() {
    let d = device();
    let layer_input = Tensor::<f32, 2>::from_slice(&d, [2, 2], &[1., 3., 2., 4.]).unwrap();
    let layer_weight = Tensor::<f32, 1>::from_slice(&d, [2], &[2., 0.5]).unwrap();
    let layer_bias = Tensor::<f32, 1>::from_slice(&d, [2], &[1., -1.]).unwrap();
    let layer = run(request(
        "layer_norm",
        vec![layer_input.handle(), layer_weight.handle(), layer_bias.handle()],
        vec![2, 2],
        attrs(&[("epsilon", float(0.0))]),
    ))
    .unwrap();
    assert_eq!(layer, vec![-1., -0.5, -1., -0.5]);

    let group_input = Tensor::<f32, 4>::from_slice(&d, [1, 2, 1, 2], &[1., 3., 2., 4.]).unwrap();
    let group_weight = Tensor::<f32, 1>::from_slice(&d, [2], &[1., 2.]).unwrap();
    let group_bias = Tensor::<f32, 1>::from_slice(&d, [2], &[0., 0.5]).unwrap();
    let group = run(request(
        "group.norm",
        vec![group_input.handle(), group_weight.handle(), group_bias.handle()],
        vec![1, 2, 1, 2],
        attrs(&[("groups", AttrValue::Int(1)), ("epsilon", float(0.0))]),
    ))
    .unwrap();
    let expected = [-1.341_640_8, 0.447_213_6, -0.394_427_2, 3.183_281_7];
    assert!(group.iter().zip(expected).all(|(actual, expected)| (actual - expected).abs() < 1e-6));
}

#[test]
fn cpu_reference_layer_and_group_norm_reject_invalid_contracts() {
    let d = device();
    let scalar = Tensor::<f32, 0>::from_slice(&d, [], &[1.]).unwrap();
    assert!(
        run(request("layer.norm", vec![scalar.handle()], vec![], attrs(&[("epsilon", float(1e-5))])))
            .unwrap_err()
            .message
            .contains("non-scalar")
    );
    let layer_input = Tensor::<f32, 2>::from_slice(&d, [1, 2], &[1., 3.]).unwrap();
    assert!(
        run(request("layer_norm", vec![layer_input.handle()], vec![1, 2], AttrMap::new()))
            .unwrap_err()
            .message
            .contains("missing float attribute epsilon")
    );
    let wrong_weight = Tensor::<f32, 1>::from_slice(&d, [3], &[1., 1., 1.]).unwrap();
    assert!(
        run(request(
            "layer_norm",
            vec![layer_input.handle(), wrong_weight.handle()],
            vec![1, 2],
            attrs(&[("epsilon", float(1e-5))]),
        ))
        .unwrap_err()
        .message
        .contains("weight")
    );

    let group_input = Tensor::<f32, 4>::from_slice(&d, [1, 2, 1, 2], &[1., 2., 3., 4.]).unwrap();
    assert!(
        run(request(
            "group_norm",
            vec![group_input.handle()],
            vec![1, 2, 1, 2],
            attrs(&[("groups", AttrValue::Int(3)), ("epsilon", float(1e-5))]),
        ))
        .unwrap_err()
        .message
        .contains("groups")
    );
    assert!(
        run(request(
            "group_norm",
            vec![group_input.handle()],
            vec![1, 2, 1, 2],
            attrs(&[("groups", AttrValue::Int(1)), ("epsilon", float(-1.0))]),
        ))
        .unwrap_err()
        .message
        .contains("epsilon")
    );
    let rank_three = Tensor::<f32, 3>::from_slice(&d, [1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    assert!(
        run(request(
            "group.norm",
            vec![rank_three.handle()],
            vec![1, 1, 2, 2],
            attrs(&[("groups", AttrValue::Int(1)), ("epsilon", float(1e-5))]),
        ))
        .unwrap_err()
        .message
        .contains("rank-4")
    );
    let mut group_dtype = request(
        "group_norm",
        vec![group_input.handle()],
        vec![1, 2, 1, 2],
        attrs(&[("groups", AttrValue::Int(1)), ("epsilon", float(1e-5))]),
    );
    group_dtype.outputs[0].dtype = DType::I32;
    assert!(run(group_dtype).unwrap_err().message.contains("F32"));
    let mut layer_layout = request("layer_norm", vec![layer_input.handle()], vec![1, 2], attrs(&[("epsilon", float(1e-5))]));
    layer_layout.outputs[0].layout = Layout::Strided;
    assert!(run(layer_layout).unwrap_err().message.contains("contiguous"));
}
