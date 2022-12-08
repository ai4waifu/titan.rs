use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cpu_add.rs".into(), line: 1, column: 1 }
}

fn request(lhs: titan_tensor::TensorHandle, rhs: titan_tensor::TensorHandle, dtype: DType, shape: Vec<u64>) -> OpRequest {
    OpRequest {
        operator: OperatorId("elementwise.add.f32".into()),
        inputs: vec![lhs, rhs],
        outputs: vec![TensorSpec {
            dtype,
            strides: Strides(vec![1; shape.len()]),
            shape: Shape(shape),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs: AttrMap::new(),
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: source(),
    }
}

fn device() -> Device {
    let driver = Arc::new(CpuDriver);
    Device::from_session(driver.open(driver.enumerate().unwrap()[0].device).unwrap())
}

#[test]
fn cpu_add_returns_real_multi_element_values() {
    let device = device();
    let lhs = Tensor::<f32, 2>::from_slice(&device, [2, 3], &[1.0, -2.0, 3.5, 4.0, 0.0, 8.0]).unwrap();
    let rhs = Tensor::<f32, 2>::from_slice(&device, [2, 3], &[2.0, 5.0, -0.5, 1.0, 7.0, 2.0]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cpu-add-test"));
    let result = runtime.execute(request(lhs.handle(), rhs.handle(), DType::F32, vec![2, 3])).unwrap().wait().unwrap();
    assert_eq!(result.outputs[0].to_vec_f32().unwrap(), vec![3.0, 3.0, 3.0, 5.0, 7.0, 10.0]);
}

#[test]
fn cpu_add_reuses_the_compiled_artifact_for_an_identical_abi_and_device() {
    let device = device();
    let lhs = Tensor::<f32, 1>::from_slice(&device, [3], &[1.0, 2.0, 3.0]).unwrap();
    let rhs = Tensor::<f32, 1>::from_slice(&device, [3], &[4.0, 5.0, 6.0]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cpu-add-cache-test"));

    runtime.execute(request(lhs.handle(), rhs.handle(), DType::F32, vec![3])).unwrap().wait().unwrap();
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));

    let result = runtime.execute(request(lhs.handle(), rhs.handle(), DType::F32, vec![3])).unwrap().wait().unwrap();
    assert_eq!(result.outputs[0].to_vec_f32().unwrap(), vec![5.0, 7.0, 9.0]);
    assert_eq!(runtime.artifact_cache_stats(), (1, 1));
}

#[test]
fn cpu_add_allows_zero_length() {
    let device = device();
    let lhs = Tensor::<f32, 1>::from_slice(&device, [0], &[]).unwrap();
    let rhs = Tensor::<f32, 1>::from_slice(&device, [0], &[]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cpu-add-zero-test"));
    let result = runtime.execute(request(lhs.handle(), rhs.handle(), DType::F32, vec![0])).unwrap().wait().unwrap();
    assert!(result.outputs[0].to_vec_f32().unwrap().is_empty());
}

#[test]
fn cpu_add_rejects_shape_mismatch() {
    let device = device();
    let lhs = Tensor::<f32, 1>::from_slice(&device, [2], &[1.0, 2.0]).unwrap();
    let rhs = Tensor::<f32, 1>::from_slice(&device, [3], &[1.0, 2.0, 3.0]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cpu-add-shape-test"));
    let error = runtime.execute(request(lhs.handle(), rhs.handle(), DType::F32, vec![2])).unwrap_err();
    assert_eq!(error.phase, "contract");
    assert!(error.message.contains("shapes"));
}

#[test]
fn cpu_add_rejects_wrong_dtype() {
    let device = device();
    let lhs = Tensor::<f32, 1>::from_slice(&device, [2], &[1.0, 2.0]).unwrap();
    let rhs = Tensor::<f32, 1>::from_slice(&device, [2], &[1.0, 2.0]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cpu-add-dtype-test"));
    let error = runtime.execute(request(lhs.handle(), rhs.handle(), DType::I32, vec![2])).unwrap_err();
    assert_eq!(error.phase, "contract");
    assert!(error.message.contains("F32"));
}

#[test]
fn cpu_add_rejects_cross_session_inputs() {
    let lhs_device = device();
    let rhs_device = device();
    let lhs = Tensor::<f32, 1>::from_slice(&lhs_device, [2], &[1.0, 2.0]).unwrap();
    let rhs = Tensor::<f32, 1>::from_slice(&rhs_device, [2], &[1.0, 2.0]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cpu-add-session-test"));
    let error = runtime.execute(request(lhs.handle(), rhs.handle(), DType::F32, vec![2])).unwrap_err();
    assert_eq!(error.phase, "contract");
    assert!(error.message.contains("session"));
}
