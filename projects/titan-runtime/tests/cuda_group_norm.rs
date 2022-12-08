use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor, TensorHandle};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_group_norm.rs".into(), line: 1, column: 1 }
}
fn request(
    inputs: Vec<TensorHandle>,
    shape: Vec<u64>,
    dtype: DType,
    layout: Layout,
    strides: Vec<i64>,
    attrs: AttrMap,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("group_norm".into()),
        inputs,
        outputs: vec![TensorSpec {
            dtype,
            shape: Shape(shape),
            strides: Strides(strides),
            layout,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: source(),
    }
}
fn no_cuda(error: &titan_hal::HalError) -> bool {
    error.operation == "load_driver"
        || ((error.operation == "cuInit" || error.operation == "cuDeviceGetCount") && error.detail.contains("status 100"))
}
fn cuda_device() -> Option<Device> {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda(&error) => return None,
        Err(error) => panic!("CUDA Driver API open: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA");
    devices.first().map(|f| Device::from_session(driver.open(f.device).expect("primary context")))
}
fn cpu_device() -> Device {
    let driver = Arc::new(CpuDriver);
    Device::from_session(driver.open(driver.enumerate().unwrap()[0].device).unwrap())
}
fn float(value: f32) -> AttrValue {
    AttrValue::Float((value as f64).to_bits())
}
fn attrs(groups: i64, epsilon: f32) -> AttrMap {
    let mut values = AttrMap::new();
    values.insert("groups".into(), AttrValue::Int(groups));
    values.insert("epsilon".into(), float(epsilon));
    values
}

#[test]
fn runtime_cuda_group_norm_driver_jit_d2h_matches_cpu_reference_with_affine() {
    let Some(cuda) = cuda_device()
    else {
        return;
    };
    let cpu = cpu_device();
    let values = [-2.0f32, 0.5, 4.0, 7.0, 3.0, -1.0, 2.0, 5.0, 1.0, 3.0, -4.0, 2.0, 8.0, 0.0, -3.0, 6.0];
    let gamma = [1.5f32, -0.5, 0.25, 2.0];
    let beta = [0.2f32, -0.3, 2.0, -1.0];
    let cuda_input = Tensor::<f32, 4>::from_slice(&cuda, [2, 4, 1, 2], &values).unwrap();
    let cuda_gamma = Tensor::<f32, 1>::from_slice(&cuda, [4], &gamma).unwrap();
    let cuda_beta = Tensor::<f32, 1>::from_slice(&cuda, [4], &beta).unwrap();
    let cpu_input = Tensor::<f32, 4>::from_slice(&cpu, [2, 4, 1, 2], &values).unwrap();
    let cpu_gamma = Tensor::<f32, 1>::from_slice(&cpu, [4], &gamma).unwrap();
    let cpu_beta = Tensor::<f32, 1>::from_slice(&cpu, [4], &beta).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-group-norm"));
    let cuda_output = runtime
        .execute(request(
            vec![cuda_input.handle(), cuda_gamma.handle(), cuda_beta.handle()],
            vec![2, 4, 1, 2],
            DType::F32,
            Layout::Contiguous,
            vec![8, 2, 2, 1],
            attrs(2, 1e-4),
        ))
        .unwrap()
        .wait()
        .unwrap();
    let cpu_output = runtime
        .execute(request(
            vec![cpu_input.handle(), cpu_gamma.handle(), cpu_beta.handle()],
            vec![2, 4, 1, 2],
            DType::F32,
            Layout::Contiguous,
            vec![8, 2, 2, 1],
            attrs(2, 1e-4),
        ))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(cuda_output.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let actual = cuda_output.outputs[0].to_vec_f32().expect("D2H CUDA output");
    let expected = cpu_output.outputs[0].to_vec_f32().expect("CPU output");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!((actual - expected).abs() <= 4e-4, "index {index}: CUDA {actual}, CPU {expected}");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_group_norm_rejects_dtype_layout_rank_channels_groups_affine_and_epsilon() {
    let Some(cuda) = cuda_device()
    else {
        return;
    };
    let input = Tensor::<f32, 4>::from_slice(&cuda, [1, 4, 1, 2], &[1.0; 8]).unwrap();
    let short = Tensor::<f32, 1>::from_slice(&cuda, [3], &[1.0; 3]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-group-norm-negative"));
    let error = runtime
        .execute(request(
            vec![input.handle()],
            vec![1, 4, 1, 2],
            DType::F16,
            Layout::Contiguous,
            vec![8, 2, 2, 1],
            attrs(2, 1e-5),
        ))
        .unwrap_err();
    assert!(error.message.contains("F32"));
    let error = runtime
        .execute(request(vec![input.handle()], vec![1, 4, 1, 2], DType::F32, Layout::Strided, vec![8, 2, 2, 2], attrs(2, 1e-5)))
        .unwrap_err();
    assert!(error.message.contains("contiguous"));
    let error = runtime
        .execute(request(vec![input.handle()], vec![1, 4, 1], DType::F32, Layout::Contiguous, vec![4, 1, 1], attrs(2, 1e-5)))
        .unwrap_err();
    assert!(error.message.contains("same NCHW"));
    let error = runtime
        .execute(request(
            vec![input.handle()],
            vec![1, 4, 1, 2],
            DType::F32,
            Layout::Contiguous,
            vec![8, 2, 2, 1],
            attrs(3, 1e-5),
        ))
        .unwrap_err();
    assert!(error.message.contains("divide"));
    let error = runtime
        .execute(request(
            vec![input.handle(), short.handle()],
            vec![1, 4, 1, 2],
            DType::F32,
            Layout::Contiguous,
            vec![8, 2, 2, 1],
            attrs(2, 1e-5),
        ))
        .unwrap_err();
    assert!(error.message.contains("affine"));
    let error = runtime
        .execute(request(
            vec![input.handle()],
            vec![1, 4, 1, 2],
            DType::F32,
            Layout::Contiguous,
            vec![8, 2, 2, 1],
            attrs(2, -1.0),
        ))
        .unwrap_err();
    assert!(error.message.contains("epsilon"));
}
