use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_silu.rs".into(), line: 1, column: 1 }
}

fn silu_request(input: titan_tensor::TensorHandle) -> OpRequest {
    OpRequest {
        operator: OperatorId("silu".into()),
        inputs: vec![input],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(vec![1]),
            shape: Shape(vec![3]),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs: AttrMap::new(),
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: source(),
    }
}

fn no_cuda_environment(error: &titan_hal::HalError) -> bool {
    error.operation == "load_driver"
        || ((error.operation == "cuInit" || error.operation == "cuDeviceGetCount") && error.detail.contains("status 100"))
}

#[test]
fn runtime_cuda_silu_matches_cpu_reference_and_returns_readable_tensor_handle() {
    let cuda_driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("CUDA unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = cuda_driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("CUDA unavailable: Driver API reported no GPU");
        return;
    };
    let cuda_device = Device::from_session(cuda_driver.open(fingerprint.device).expect("open CUDA primary context"));
    let cpu_driver = Arc::new(CpuDriver);
    let cpu_device =
        Device::from_session(cpu_driver.open(cpu_driver.enumerate().expect("enumerate CPU")[0].device).expect("open CPU"));
    let values = [-1.0f32, 0.0, 1.0];
    let cuda_input = Tensor::<f32, 1>::from_slice(&cuda_device, [3], &values).expect("upload CUDA input");
    let cpu_input = Tensor::<f32, 1>::from_slice(&cpu_device, [3], &values).expect("upload CPU input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-silu"));

    let cuda = runtime
        .execute(silu_request(cuda_input.handle()))
        .expect("Runtime CUDA SiLU dispatch")
        .wait()
        .expect("Runtime CUDA SiLU completion");
    let cpu = runtime
        .execute(silu_request(cpu_input.handle()))
        .expect("Runtime CPU SiLU dispatch")
        .wait()
        .expect("Runtime CPU SiLU completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
    let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
    for (index, (cuda_value, cpu_value)) in cuda_values.iter().zip(&cpu_values).enumerate() {
        let error = (cuda_value - cpu_value).abs();
        assert!(error <= 5e-5, "index {index} mismatch: cuda={cuda_value} cpu={cpu_value} error={error}");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}
