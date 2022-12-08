use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_gelu.rs".into(), line: 1, column: 1 }
}

fn gelu_request(
    input: titan_tensor::TensorHandle,
    output_shape: Vec<u64>,
    output_dtype: DType,
    output_layout: Layout,
    output_strides: Vec<i64>,
    attrs: AttrMap,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("gelu".into()),
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

#[test]
fn runtime_cuda_gelu_matches_cpu_erf_reference_and_returns_readable_tensor_handle() {
    let Some(cuda_device) = cuda_device()
    else {
        return;
    };
    let cpu_driver = Arc::new(CpuDriver);
    let cpu_device =
        Device::from_session(cpu_driver.open(cpu_driver.enumerate().expect("enumerate CPU")[0].device).expect("open CPU"));
    let values = [-1.0f32, 0.0, 1.0];
    let cuda_input = Tensor::<f32, 1>::from_slice(&cuda_device, [3], &values).expect("upload CUDA input");
    let cpu_input = Tensor::<f32, 1>::from_slice(&cpu_device, [3], &values).expect("upload CPU input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-gelu"));
    let attrs = AttrMap::new();

    let cuda = runtime
        .execute(gelu_request(cuda_input.handle(), vec![3], DType::F32, Layout::Contiguous, vec![1], attrs.clone()))
        .expect("Runtime CUDA GELU dispatch")
        .wait()
        .expect("Runtime CUDA GELU completion");
    let cpu = runtime
        .execute(gelu_request(cpu_input.handle(), vec![3], DType::F32, Layout::Contiguous, vec![1], attrs))
        .expect("Runtime CPU GELU dispatch")
        .wait()
        .expect("Runtime CPU GELU completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
    let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
    for (index, (cuda_value, cpu_value)) in cuda_values.iter().zip(&cpu_values).enumerate() {
        let error = (cuda_value - cpu_value).abs();
        assert!(error <= 5e-4, "index {index} mismatch: cuda={cuda_value} cpu={cpu_value} error={error}");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_gelu_rejects_invalid_shape_dtype_layout_and_attributes() {
    let Some(cuda_device) = cuda_device()
    else {
        return;
    };
    let input = Tensor::<f32, 1>::from_slice(&cuda_device, [3], &[-1.0, 0.0, 1.0]).expect("upload CUDA input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-gelu-negative"));

    let bad_shape = runtime
        .execute(gelu_request(input.handle(), vec![2], DType::F32, Layout::Contiguous, vec![1], AttrMap::new()))
        .expect_err("GELU must reject output shape mismatch");
    assert_eq!(bad_shape.phase, "contract");
    assert!(bad_shape.message.contains("same shape"), "{}", bad_shape.message);

    let bad_dtype = runtime
        .execute(gelu_request(input.handle(), vec![3], DType::F16, Layout::Contiguous, vec![1], AttrMap::new()))
        .expect_err("GELU must reject non-F32 output");
    assert_eq!(bad_dtype.phase, "contract");
    assert!(bad_dtype.message.contains("F32"), "{}", bad_dtype.message);

    let bad_layout = runtime
        .execute(gelu_request(input.handle(), vec![3], DType::F32, Layout::Strided, vec![2], AttrMap::new()))
        .expect_err("GELU must reject non-contiguous output");
    assert_eq!(bad_layout.phase, "contract");
    assert!(bad_layout.message.contains("contiguous"), "{}", bad_layout.message);

    let mut attrs = AttrMap::new();
    attrs.insert("approximation".into(), titan_types::AttrValue::String("tanh".into()));
    let bad_attrs = runtime
        .execute(gelu_request(input.handle(), vec![3], DType::F32, Layout::Contiguous, vec![1], attrs))
        .expect_err("GELU must reject unsupported attributes");
    assert_eq!(bad_attrs.phase, "contract");
    assert!(bad_attrs.message.contains("attributes"), "{}", bad_attrs.message);
}
