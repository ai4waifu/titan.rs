use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor, TensorHandle};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_layer_norm.rs".into(), line: 1, column: 1 }
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
        operator: OperatorId("layer_norm".into()),
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

#[test]
fn runtime_cuda_layer_norm_matches_cpu_reference_with_affine() {
    let Some(cuda) = cuda_device()
    else {
        return;
    };
    let cpu = cpu_device();
    let values = [-2.0f32, 0.5, 4.0, 7.0, 3.0, -1.0];
    let gamma = [1.5f32, -0.5, 0.25];
    let beta = [0.2f32, -0.3, 2.0];
    let cuda_input = Tensor::<f32, 2>::from_slice(&cuda, [2, 3], &values).unwrap();
    let cuda_gamma = Tensor::<f32, 1>::from_slice(&cuda, [3], &gamma).unwrap();
    let cuda_beta = Tensor::<f32, 1>::from_slice(&cuda, [3], &beta).unwrap();
    let cpu_input = Tensor::<f32, 2>::from_slice(&cpu, [2, 3], &values).unwrap();
    let cpu_gamma = Tensor::<f32, 1>::from_slice(&cpu, [3], &gamma).unwrap();
    let cpu_beta = Tensor::<f32, 1>::from_slice(&cpu, [3], &beta).unwrap();
    let mut attrs = AttrMap::new();
    attrs.insert("epsilon".into(), float(1e-4));
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-layer-norm"));
    let cuda_output = runtime
        .execute(request(
            vec![cuda_input.handle(), cuda_gamma.handle(), cuda_beta.handle()],
            vec![2, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
            attrs.clone(),
        ))
        .unwrap()
        .wait()
        .unwrap();
    let cpu_output = runtime
        .execute(request(
            vec![cpu_input.handle(), cpu_gamma.handle(), cpu_beta.handle()],
            vec![2, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
            attrs,
        ))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(cuda_output.outputs[0].device().backend, titan_types::BackendId::Cuda);
    for (index, (actual, expected)) in
        cuda_output.outputs[0].to_vec_f32().unwrap().iter().zip(cpu_output.outputs[0].to_vec_f32().unwrap()).enumerate()
    {
        assert!((actual - expected).abs() <= 3e-4, "index {index}: CUDA {actual}, CPU {expected}");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_layer_norm_rejects_invalid_contracts() {
    let Some(cuda) = cuda_device()
    else {
        return;
    };
    let input = Tensor::<f32, 2>::from_slice(&cuda, [2, 3], &[1.0; 6]).unwrap();
    let short = Tensor::<f32, 1>::from_slice(&cuda, [2], &[1.0; 2]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-layer-norm-negative"));
    let mut axis = AttrMap::new();
    axis.insert("axis".into(), AttrValue::Int(0));
    let error = runtime
        .execute(request(vec![input.handle()], vec![2, 3], DType::F32, Layout::Contiguous, vec![3, 1], axis))
        .unwrap_err();
    assert_eq!(error.phase, "contract");
    assert!(error.message.contains("last axis"));
    let error = runtime
        .execute(request(vec![input.handle()], vec![2, 3], DType::F16, Layout::Contiguous, vec![3, 1], AttrMap::new()))
        .unwrap_err();
    assert!(error.message.contains("F32"));
    let error = runtime
        .execute(request(vec![input.handle()], vec![2, 3], DType::F32, Layout::Strided, vec![3, 2], AttrMap::new()))
        .unwrap_err();
    assert!(error.message.contains("contiguous"));
    let error = runtime
        .execute(request(
            vec![input.handle(), short.handle()],
            vec![2, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
            AttrMap::new(),
        ))
        .unwrap_err();
    assert!(error.message.contains("affine"));
    let mut invalid_epsilon = AttrMap::new();
    invalid_epsilon.insert("epsilon".into(), float(-1.0));
    let error = runtime
        .execute(request(vec![input.handle()], vec![2, 3], DType::F32, Layout::Contiguous, vec![3, 1], invalid_epsilon))
        .unwrap_err();
    assert!(error.message.contains("epsilon"));
}
