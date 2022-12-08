use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_gemm.rs".into(), line: 1, column: 1 }
}

fn gemm_request(lhs: titan_tensor::TensorHandle, rhs: titan_tensor::TensorHandle, output_shape: [u64; 2]) -> OpRequest {
    OpRequest {
        operator: OperatorId("gemm".into()),
        inputs: vec![lhs, rhs],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(vec![output_shape[1] as i64, 1]),
            shape: Shape(output_shape.to_vec()),
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
fn runtime_cuda_gemm_matches_cpu_reference_and_returns_readable_tensor_handle() {
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
    let lhs = [1., 2., 3., 4., 5., 6., 7., 8.];
    let rhs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let cuda_lhs = Tensor::<f32, 2>::from_slice(&cuda_device, [2, 4], &lhs).expect("upload CUDA lhs");
    let cuda_rhs = Tensor::<f32, 2>::from_slice(&cuda_device, [4, 3], &rhs).expect("upload CUDA rhs");
    let cpu_lhs = Tensor::<f32, 2>::from_slice(&cpu_device, [2, 4], &lhs).expect("upload CPU lhs");
    let cpu_rhs = Tensor::<f32, 2>::from_slice(&cpu_device, [4, 3], &rhs).expect("upload CPU rhs");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-gemm"));

    let cuda = runtime
        .execute(gemm_request(cuda_lhs.handle(), cuda_rhs.handle(), [2, 3]))
        .expect("Runtime CUDA GEMM dispatch")
        .wait()
        .expect("Runtime CUDA GEMM completion");
    let cpu = runtime
        .execute(gemm_request(cpu_lhs.handle(), cpu_rhs.handle(), [2, 3]))
        .expect("Runtime CPU GEMM dispatch")
        .wait()
        .expect("Runtime CPU GEMM completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    assert_eq!(cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output"), cpu.outputs[0].to_vec_f32().unwrap());
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}
