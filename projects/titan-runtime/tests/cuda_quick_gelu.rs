use std::{collections::BTreeMap, sync::Arc};

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn request(input: titan_tensor::TensorHandle, attrs: AttrMap) -> OpRequest {
    OpRequest {
        operator: OperatorId("quick_gelu".into()),
        inputs: vec![input],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(vec![1]),
            shape: Shape(vec![5]),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: SourceSpan { file: "cuda_quick_gelu.rs".into(), line: 1, column: 1 },
    }
}

fn slope_attr(slope: f64) -> AttrMap {
    BTreeMap::from([("slope".into(), AttrValue::Float(slope.to_bits()))])
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
fn runtime_cuda_quick_gelu_matches_cpu_and_formula_for_default_and_custom_slopes() {
    let Some(cuda_device) = cuda_device()
    else {
        return;
    };
    let cpu_driver = Arc::new(CpuDriver);
    let cpu_device =
        Device::from_session(cpu_driver.open(cpu_driver.enumerate().expect("enumerate CPU")[0].device).expect("open CPU"));
    let values = [-3.0f32, -1.0, 0.0, 1.0, 3.0];
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-quick-gelu"));

    for (attrs, slope) in [(AttrMap::new(), 1.702f32), (slope_attr(1.25), 1.25f32)] {
        let cuda_input = Tensor::<f32, 1>::from_slice(&cuda_device, [5], &values).expect("upload CUDA input");
        let cpu_input = Tensor::<f32, 1>::from_slice(&cpu_device, [5], &values).expect("upload CPU input");
        let cuda_handle =
            runtime.execute(request(cuda_input.handle(), attrs.clone())).expect("Runtime CUDA QuickGELU dispatch");
        assert_eq!(cuda_handle.kernel_id().0, "quick_gelu.f32");
        let cuda = cuda_handle.wait().expect("Runtime CUDA QuickGELU completion");
        let cpu = runtime
            .execute(request(cpu_input.handle(), attrs))
            .expect("Runtime CPU QuickGELU dispatch")
            .wait()
            .expect("Runtime CPU QuickGELU completion");

        assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
        let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
        let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
        for (index, ((cuda_value, cpu_value), input)) in cuda_values.iter().zip(&cpu_values).zip(values).enumerate() {
            let expected = input * (1.0 / (1.0 + (-slope * input).exp()));
            assert!((cpu_value - expected).abs() <= 1e-6, "CPU index {index}: actual={cpu_value} expected={expected}");
            assert!((cuda_value - expected).abs() <= 5e-5, "CUDA index {index}: actual={cuda_value} expected={expected}");
        }
    }

    assert_eq!(runtime.artifact_cache_stats(), (1, 1));
}

#[test]
fn runtime_cuda_quick_gelu_rejects_invalid_attributes() {
    let Some(cuda_device) = cuda_device()
    else {
        return;
    };
    let input = Tensor::<f32, 1>::from_slice(&cuda_device, [5], &[-3.0, -1.0, 0.0, 1.0, 3.0]).expect("upload CUDA input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-quick-gelu-negative"));

    let zero = runtime.execute(request(input.handle(), slope_attr(0.0))).expect_err("zero QuickGELU slope must fail");
    assert_eq!(zero.phase, "contract");
    assert!(zero.message.contains("positive"), "{}", zero.message);

    let unsupported = BTreeMap::from([("approximation".into(), AttrValue::String("tanh".into()))]);
    let unsupported = runtime.execute(request(input.handle(), unsupported)).expect_err("unsupported QuickGELU attr must fail");
    assert_eq!(unsupported.phase, "contract");
    assert!(unsupported.message.contains("only accepts"), "{}", unsupported.message);
}
