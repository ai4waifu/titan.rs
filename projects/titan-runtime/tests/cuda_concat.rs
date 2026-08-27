use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_concat.rs".into(), line: 1, column: 1 }
}

fn cuda_device() -> Option<Device> {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if error.operation == "load_driver" || error.detail.contains("status 100") => return None,
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let Some(fingerprint) = driver.enumerate().expect("enumerate CUDA devices").first().cloned()
    else {
        return None;
    };
    Some(Device::from_session(driver.open(fingerprint.device).expect("open CUDA primary context")))
}

fn request(
    inputs: Vec<titan_tensor::TensorHandle>,
    shape: Vec<u64>,
    dtype: DType,
    layout: Layout,
    strides: Vec<i64>,
    attrs: AttrMap,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("concat".into()),
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

fn axis(value: i64) -> AttrMap {
    let mut attrs = AttrMap::new();
    attrs.insert("axis".into(), AttrValue::Int(value));
    attrs
}

#[test]
fn runtime_cuda_concat_jits_launches_and_matches_cpu_reference() {
    let Some(cuda) = cuda_device()
    else {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    };
    let lhs = Tensor::<f32, 2>::from_slice(&cuda, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("upload lhs");
    let rhs = Tensor::<f32, 2>::from_slice(&cuda, [1, 3], &[7.0, 8.0, 9.0]).expect("upload rhs");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-concat"));
    let result = runtime
        .execute(request(vec![lhs.handle(), rhs.handle()], vec![3, 3], DType::F32, Layout::Contiguous, vec![3, 1], axis(0)))
        .expect("Driver JIT and CUDA launch")
        .wait()
        .expect("CUDA event completion and D2H readback");
    let actual = result.outputs[0].to_vec_f32().expect("D2H output readback");
    assert_eq!(actual, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_concat_rejects_dtype_layout_rank_axis_and_shape() {
    let Some(cuda) = cuda_device()
    else {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    };
    let lhs = Tensor::<f32, 2>::from_slice(&cuda, [2, 3], &[1.0; 6]).expect("upload lhs");
    let rhs = Tensor::<f32, 2>::from_slice(&cuda, [1, 3], &[2.0; 3]).expect("upload rhs");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-concat-negative"));
    let error = runtime
        .execute(request(vec![lhs.handle(), rhs.handle()], vec![3, 3], DType::F16, Layout::Contiguous, vec![3, 1], axis(0)))
        .unwrap_err();
    assert!(error.message.contains("F32"));
    let error = runtime
        .execute(request(vec![lhs.handle(), rhs.handle()], vec![3, 3], DType::F32, Layout::Strided, vec![1, 1], axis(0)))
        .unwrap_err();
    assert!(error.message.contains("contiguous"));
    let error = runtime
        .execute(request(vec![lhs.handle(), rhs.handle()], vec![3, 3], DType::F32, Layout::Contiguous, vec![3, 1], axis(1)))
        .unwrap_err();
    assert!(error.message.contains("axis=0"));
    let error = runtime
        .execute(request(vec![lhs.handle(), rhs.handle()], vec![4, 3], DType::F32, Layout::Contiguous, vec![3, 1], axis(0)))
        .unwrap_err();
    assert!(error.message.contains("exact"));
    let rank_three = Tensor::<f32, 3>::from_slice(&cuda, [1, 2, 3], &[1.0; 6]).expect("upload rank-three");
    let error = runtime
        .execute(request(
            vec![rank_three.handle(), rhs.handle()],
            vec![3, 3],
            DType::F32,
            Layout::Contiguous,
            vec![3, 1],
            axis(0),
        ))
        .unwrap_err();
    assert!(error.message.contains("rank-2"));
}
