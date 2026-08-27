use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_attention.rs".into(), line: 1, column: 1 }
}

fn attention_request(
    query: titan_tensor::TensorHandle,
    key: titan_tensor::TensorHandle,
    value: titan_tensor::TensorHandle,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("scaled_dot_product_attention".into()),
        inputs: vec![query, key, value],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(vec![4, 4, 2, 1]),
            shape: Shape(vec![1, 1, 2, 2]),
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
fn runtime_cuda_attention_matches_cpu_reference_with_distinct_query_and_key_lengths() {
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
    let query = [1.0, -0.5, 0.25, 1.5];
    let key = [0.5, 1.0, -1.0, 0.25, 0.75, -0.5];
    let value = [2.0, -1.0, 0.5, 3.0, -2.0, 1.5];
    let cuda_query = Tensor::<f32, 4>::from_slice(&cuda_device, [1, 1, 2, 2], &query).expect("upload CUDA Q");
    let cuda_key = Tensor::<f32, 4>::from_slice(&cuda_device, [1, 1, 3, 2], &key).expect("upload CUDA K");
    let cuda_value = Tensor::<f32, 4>::from_slice(&cuda_device, [1, 1, 3, 2], &value).expect("upload CUDA V");
    let cpu_query = Tensor::<f32, 4>::from_slice(&cpu_device, [1, 1, 2, 2], &query).expect("upload CPU Q");
    let cpu_key = Tensor::<f32, 4>::from_slice(&cpu_device, [1, 1, 3, 2], &key).expect("upload CPU K");
    let cpu_value = Tensor::<f32, 4>::from_slice(&cpu_device, [1, 1, 3, 2], &value).expect("upload CPU V");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-attention"));

    let cuda = runtime
        .execute(attention_request(cuda_query.handle(), cuda_key.handle(), cuda_value.handle()))
        .expect("Runtime CUDA attention dispatch")
        .wait()
        .expect("Runtime CUDA attention completion");
    let cpu = runtime
        .execute(attention_request(cpu_query.handle(), cpu_key.handle(), cpu_value.handle()))
        .expect("Runtime CPU attention dispatch")
        .wait()
        .expect("Runtime CPU attention completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
    let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
    for (index, (cuda_value, cpu_value)) in cuda_values.iter().zip(cpu_values.iter()).enumerate() {
        assert!((cuda_value - cpu_value).abs() <= 2e-4, "output {index}: CUDA={cuda_value}, CPU={cpu_value}");
    }

    for attribute in ["mask", "causal"] {
        let mut unsupported = attention_request(cuda_query.handle(), cuda_key.handle(), cuda_value.handle());
        unsupported.attrs.insert(attribute.into(), AttrValue::Bool(true));
        let error = runtime.execute(unsupported).expect_err("Runtime must reject unsupported CUDA attention attributes");
        assert_eq!(error.phase, "contract");
        assert!(error.message.contains("does not implement"), "{attribute}: {}", error.message);
    }
}
