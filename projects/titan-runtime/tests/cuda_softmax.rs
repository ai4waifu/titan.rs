use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_softmax.rs".into(), line: 1, column: 1 }
}

fn softmax_request(
    input: titan_tensor::TensorHandle,
    output_shape: Vec<u64>,
    output_dtype: DType,
    output_layout: Layout,
    output_strides: Vec<i64>,
    attrs: AttrMap,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("softmax".into()),
        inputs: vec![input],
        outputs: vec![TensorSpec {
            dtype: output_dtype,
            strides: Strides(output_strides),
            shape: Shape(output_shape),
            layout: output_layout,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: source(),
    }
}

fn no_cuda_environment(error: &titan_hal::HalError) -> bool {
    error.operation == "load_driver"
        || ((error.operation == "cuInit" || error.operation == "cuDeviceGetCount") && error.detail.contains("status 100"))
}

fn cuda_device() -> Option<Device> {
    let cuda_driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("CUDA unavailable: {error}");
            return None;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = cuda_driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("CUDA unavailable: Driver API reported no GPU");
        return None;
    };
    Some(Device::from_session(cuda_driver.open(fingerprint.device).expect("open CUDA primary context")))
}

fn cpu_softmax(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        let start = row * cols;
        let maximum = values[start..start + cols].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = values[start..start + cols].iter().map(|value| (value - maximum).exp()).sum();
        for column in 0..cols {
            output[start + column] = (values[start + column] - maximum).exp() / denominator;
        }
    }
    output
}

#[test]
fn runtime_cuda_softmax_matches_cpu_stable_reference_and_row_sums() {
    let Some(cuda_device) = cuda_device()
    else {
        return;
    };
    let cpu_driver = Arc::new(CpuDriver);
    let cpu_device =
        Device::from_session(cpu_driver.open(cpu_driver.enumerate().expect("enumerate CPU")[0].device).expect("open CPU"));
    let values = [1001.0f32, 1000.0, 999.0, 1.0, 2.0, 3.0];
    let expected = cpu_softmax(&values, 2, 3);
    let cuda_input = Tensor::<f32, 2>::from_slice(&cuda_device, [2, 3], &values).expect("upload CUDA input");
    let cpu_input = Tensor::<f32, 2>::from_slice(&cpu_device, [2, 3], &values).expect("upload CPU input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-softmax"));
    let attrs = AttrMap::new();

    let cuda = runtime
        .execute(softmax_request(cuda_input.handle(), vec![2, 3], DType::F32, Layout::Contiguous, vec![3, 1], attrs.clone()))
        .expect("Runtime CUDA softmax dispatch")
        .wait()
        .expect("Runtime CUDA softmax completion");
    let cpu = runtime
        .execute(softmax_request(cpu_input.handle(), vec![2, 3], DType::F32, Layout::Contiguous, vec![3, 1], attrs))
        .expect("Runtime CPU softmax dispatch")
        .wait()
        .expect("Runtime CPU softmax completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
    let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
    for (index, ((cuda_value, cpu_value), expected_value)) in cuda_values.iter().zip(&cpu_values).zip(&expected).enumerate() {
        assert!(
            (cuda_value - expected_value).abs() <= 2e-4,
            "index {index} mismatch: cuda={cuda_value} expected={expected_value}"
        );
        assert!(
            (cpu_value - expected_value).abs() <= 2e-6,
            "CPU index {index} mismatch: cpu={cpu_value} expected={expected_value}"
        );
    }
    for row in 0..2 {
        let sum: f32 = cuda_values[row * 3..(row + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() <= 2e-4, "CUDA row {row} sum {sum} is not close to 1");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_softmax_rejects_non_last_axis_dtype_layout_shape_and_attributes() {
    let Some(cuda_device) = cuda_device()
    else {
        return;
    };
    let input =
        Tensor::<f32, 2>::from_slice(&cuda_device, [2, 3], &[1001.0, 1000.0, 999.0, 1.0, 2.0, 3.0]).expect("upload CUDA input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-softmax-negative"));

    let mut axis_attrs = AttrMap::new();
    axis_attrs.insert("axis".into(), titan_types::AttrValue::Int(0));
    let bad_axis = runtime
        .execute(softmax_request(input.handle(), vec![2, 3], DType::F32, Layout::Contiguous, vec![3, 1], axis_attrs))
        .expect_err("softmax must reject non-last axis");
    assert_eq!(bad_axis.phase, "contract");
    assert!(bad_axis.message.contains("last axis"), "{}", bad_axis.message);

    let bad_dtype = runtime
        .execute(softmax_request(input.handle(), vec![2, 3], DType::F16, Layout::Contiguous, vec![3, 1], AttrMap::new()))
        .expect_err("softmax must reject non-F32 output");
    assert_eq!(bad_dtype.phase, "contract");
    assert!(bad_dtype.message.contains("F32"), "{}", bad_dtype.message);

    let bad_layout = runtime
        .execute(softmax_request(input.handle(), vec![2, 3], DType::F32, Layout::Strided, vec![3, 2], AttrMap::new()))
        .expect_err("softmax must reject non-contiguous output");
    assert_eq!(bad_layout.phase, "contract");
    assert!(bad_layout.message.contains("contiguous"), "{}", bad_layout.message);

    let bad_shape = runtime
        .execute(softmax_request(input.handle(), vec![2, 2], DType::F32, Layout::Contiguous, vec![2, 1], AttrMap::new()))
        .expect_err("softmax must reject output shape mismatch");
    assert_eq!(bad_shape.phase, "contract");
    assert!(bad_shape.message.contains("same shape"), "{}", bad_shape.message);

    let mut unsupported = AttrMap::new();
    unsupported.insert("temperature".into(), titan_types::AttrValue::Int(1));
    let bad_attrs = runtime
        .execute(softmax_request(input.handle(), vec![2, 3], DType::F32, Layout::Contiguous, vec![3, 1], unsupported))
        .expect_err("softmax must reject unsupported attributes");
    assert_eq!(bad_attrs.phase, "contract");
    assert!(bad_attrs.message.contains("axis"), "{}", bad_attrs.message);
}
