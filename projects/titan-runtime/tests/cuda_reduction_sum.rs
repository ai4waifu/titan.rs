use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_reduction_sum.rs".into(), line: 1, column: 1 }
}

fn request(
    input: titan_tensor::TensorHandle,
    output_shape: Vec<u64>,
    dtype: DType,
    layout: Layout,
    strides: Vec<i64>,
    attrs: AttrMap,
) -> OpRequest {
    OpRequest {
        operator: OperatorId("reduction.sum".into()),
        inputs: vec![input],
        outputs: vec![TensorSpec {
            dtype,
            shape: Shape(output_shape),
            strides: Strides(strides),
            layout,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: source(),
    }
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

fn last_axis_attrs() -> AttrMap {
    let mut attrs = AttrMap::new();
    attrs.insert("axes".into(), AttrValue::Ints(vec![1]));
    attrs
}

#[test]
fn runtime_cuda_reduction_sum_jits_launches_and_matches_cpu_reference() {
    let Some(cuda) = cuda_device()
    else {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    };
    let input =
        Tensor::<f32, 2>::from_slice(&cuda, [2, 4], &[1.25, -2.0, 3.5, 0.0, 10.0, -8.0, 0.5, 2.25]).expect("upload CUDA input");
    let expected = [2.75_f32, 4.75];
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-reduction-sum"));

    let result = runtime
        .execute(request(input.handle(), vec![2], DType::F32, Layout::Contiguous, vec![1], last_axis_attrs()))
        .expect("Driver JIT and CUDA launch")
        .wait()
        .expect("CUDA event completion and D2H readback");
    let actual = result.outputs[0].to_vec_f32().expect("D2H output readback");
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!((actual - expected).abs() <= 1e-5, "index {index}: cuda={actual} cpu={expected}");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_reduction_sum_rejects_dtype_layout_axis_and_output_shape() {
    let Some(cuda) = cuda_device()
    else {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    };
    let input = Tensor::<f32, 2>::from_slice(&cuda, [2, 4], &[1.0; 8]).expect("upload CUDA input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-reduction-sum-negative"));

    let dtype = runtime
        .execute(request(input.handle(), vec![2], DType::F16, Layout::Contiguous, vec![1], last_axis_attrs()))
        .unwrap_err();
    assert_eq!(dtype.phase, "contract");
    assert!(dtype.message.contains("F32"));

    let layout =
        runtime.execute(request(input.handle(), vec![2], DType::F32, Layout::Strided, vec![2], last_axis_attrs())).unwrap_err();
    assert_eq!(layout.phase, "contract");
    assert!(layout.message.contains("contiguous"));

    let mut bad_axis = AttrMap::new();
    bad_axis.insert("axes".into(), AttrValue::Ints(vec![0]));
    let axis =
        runtime.execute(request(input.handle(), vec![4], DType::F32, Layout::Contiguous, vec![1], bad_axis)).unwrap_err();
    assert_eq!(axis.phase, "contract");
    assert!(axis.message.contains("last axis"));

    let shape = runtime
        .execute(request(input.handle(), vec![2, 1], DType::F32, Layout::Contiguous, vec![1, 1], last_axis_attrs()))
        .unwrap_err();
    assert_eq!(shape.phase, "contract");
    assert!(shape.message.contains("output shape"));
}
