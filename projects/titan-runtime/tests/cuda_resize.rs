use std::sync::Arc;
use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn cuda() -> Option<Device> {
    let driver = CudaDriver::open().ok()?;
    let fp = driver.enumerate().ok()?.first()?.clone();
    Some(Device::from_session(driver.open(fp.device).ok()?))
}
fn request(input: titan_tensor::TensorHandle, shape: Vec<u64>, dtype: DType, layout: Layout, strides: Vec<i64>) -> OpRequest {
    OpRequest {
        operator: OperatorId("resize.nearest2d".into()),
        inputs: vec![input],
        outputs: vec![TensorSpec {
            dtype,
            strides: Strides(strides),
            shape: Shape(shape),
            layout,
            alias: AliasContract::NoAlias,
        }],
        attrs: AttrMap::new(),
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: SourceSpan { file: "cuda_resize.rs".into(), line: 1, column: 1 },
    }
}

#[test]
fn runtime_cuda_nearest_resize_matches_cpu_reference() {
    let Some(cuda) = cuda()
    else {
        return;
    };
    let cpu_driver = Arc::new(CpuDriver);
    let cpu = Device::from_session(cpu_driver.open(cpu_driver.enumerate().unwrap()[0].device).unwrap());
    let values = [1., 2., 3., 4., 5., 6.];
    let ci = Tensor::<f32, 4>::from_slice(&cuda, [1, 1, 2, 3], &values).unwrap();
    let pi = Tensor::<f32, 4>::from_slice(&cpu, [1, 1, 2, 3], &values).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cuda-resize"));
    let gpu = runtime
        .execute(request(ci.handle(), vec![1, 1, 3, 5], DType::F32, Layout::Contiguous, vec![15, 15, 5, 1]))
        .unwrap()
        .wait()
        .unwrap();
    let host = runtime
        .execute(request(pi.handle(), vec![1, 1, 3, 5], DType::F32, Layout::Contiguous, vec![15, 15, 5, 1]))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(gpu.outputs[0].device().backend, titan_types::BackendId::Cuda);
    assert_eq!(gpu.outputs[0].to_vec_f32().unwrap(), host.outputs[0].to_vec_f32().unwrap());
    assert_eq!(gpu.outputs[0].to_vec_f32().unwrap(), vec![1., 1., 2., 2., 3., 1., 1., 2., 2., 3., 4., 4., 5., 5., 6.]);
}

#[test]
fn runtime_cuda_nearest_resize_rejects_bad_contract() {
    let Some(cuda) = cuda()
    else {
        return;
    };
    let input = Tensor::<f32, 4>::from_slice(&cuda, [1, 1, 2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cuda-resize-negative"));
    let e = runtime
        .execute(request(input.handle(), vec![2, 1, 3, 5], DType::F32, Layout::Contiguous, vec![15, 15, 5, 1]))
        .unwrap_err();
    assert_eq!(e.phase, "contract");
    let e = runtime
        .execute(request(input.handle(), vec![1, 1, 3, 5], DType::F16, Layout::Contiguous, vec![15, 15, 5, 1]))
        .unwrap_err();
    assert_eq!(e.phase, "contract");
    let e = runtime
        .execute(request(input.handle(), vec![1, 1, 3, 5], DType::F32, Layout::Strided, vec![15, 15, 5, 2]))
        .unwrap_err();
    assert_eq!(e.phase, "contract");
}
