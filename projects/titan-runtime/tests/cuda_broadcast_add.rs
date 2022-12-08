use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_broadcast_add.rs".into(), line: 1, column: 1 }
}

fn broadcast_add_request(
    lhs: titan_tensor::TensorHandle,
    rhs: titan_tensor::TensorHandle,
    output_shape: Vec<u64>,
    output_dtype: DType,
    output_layout: Layout,
    output_strides: Vec<i64>,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("broadcast.add".into()),
        inputs: vec![lhs, rhs],
        outputs: vec![TensorSpec {
            dtype: output_dtype,
            strides: Strides(output_strides),
            shape: Shape(output_shape),
            layout: output_layout,
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
fn runtime_cuda_broadcast_add_matches_cpu_reference_and_returns_readable_tensor_handle() {
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
    let lhs = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    let rhs = [0.5, -2.0];
    let cuda_lhs = Tensor::<f32, 2>::from_slice(&cuda_device, [2, 3], &lhs).expect("upload CUDA lhs");
    let cuda_rhs = Tensor::<f32, 2>::from_slice(&cuda_device, [2, 1], &rhs).expect("upload CUDA rhs");
    let cpu_lhs = Tensor::<f32, 2>::from_slice(&cpu_device, [2, 3], &lhs).expect("upload CPU lhs");
    let cpu_rhs = Tensor::<f32, 2>::from_slice(&cpu_device, [2, 1], &rhs).expect("upload CPU rhs");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-broadcast-add"));

    let cuda = runtime
        .execute(broadcast_add_request(
            cuda_lhs.handle(),
            cuda_rhs.handle(),
            vec![2, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
        ))
        .expect("Runtime CUDA broadcast add dispatch")
        .wait()
        .expect("Runtime CUDA broadcast add completion");
    let cpu = runtime
        .execute(broadcast_add_request(
            cpu_lhs.handle(),
            cpu_rhs.handle(),
            vec![2, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
        ))
        .expect("Runtime CPU broadcast add dispatch")
        .wait()
        .expect("Runtime CPU broadcast add completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
    let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
    assert_eq!(cuda_values, vec![1.5, 2.5, 3.5, 8.0, 18.0, 28.0]);
    assert_eq!(cuda_values, cpu_values);
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_broadcast_add_rejects_invalid_shape_dtype_layout_and_session_requests() {
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
    let second_cuda_device = Device::from_session(cuda_driver.open(fingerprint.device).expect("open second CUDA session"));
    let lhs = Tensor::<f32, 2>::from_slice(&cuda_device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("upload CUDA lhs");
    let rhs = Tensor::<f32, 2>::from_slice(&cuda_device, [2, 1], &[0.5, -1.5]).expect("upload CUDA rhs");
    let foreign_rhs =
        Tensor::<f32, 2>::from_slice(&second_cuda_device, [2, 1], &[0.5, -1.5]).expect("upload CUDA rhs in second session");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-broadcast-add-negative"));

    let bad_shape = runtime
        .execute(broadcast_add_request(lhs.handle(), rhs.handle(), vec![2, 2], DType::F32, Layout::Contiguous, vec![2, 1]))
        .expect_err("Runtime must reject invalid broadcast output shape");
    assert_eq!(bad_shape.phase, "contract");
    assert!(bad_shape.message.contains("output shape mismatch"), "{}", bad_shape.message);

    let bad_dtype = runtime
        .execute(broadcast_add_request(lhs.handle(), rhs.handle(), vec![2, 3], DType::F16, Layout::Contiguous, vec![3, 1]))
        .expect_err("Runtime must reject non-F32 broadcast add output dtype");
    assert_eq!(bad_dtype.phase, "contract");
    assert!(bad_dtype.message.contains("F32"), "{}", bad_dtype.message);

    let bad_layout = runtime
        .execute(broadcast_add_request(lhs.handle(), rhs.handle(), vec![2, 3], DType::F32, Layout::Strided, vec![1, 2]))
        .expect_err("Runtime must reject non-contiguous broadcast add output layout");
    assert_eq!(bad_layout.phase, "contract");
    assert!(bad_layout.message.contains("contiguous"), "{}", bad_layout.message);

    let bad_session = runtime
        .execute(broadcast_add_request(
            lhs.handle(),
            foreign_rhs.handle(),
            vec![2, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
        ))
        .expect_err("Runtime must reject cross-session CUDA broadcast add inputs");
    assert_eq!(bad_session.phase, "contract");
    assert!(bad_session.message.contains("same session"), "{}", bad_session.message);
}
