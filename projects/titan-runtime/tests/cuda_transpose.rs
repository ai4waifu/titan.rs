use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

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
    input: titan_tensor::TensorHandle,
    output_shape: Vec<u64>,
    dtype: DType,
    layout: Layout,
    strides: Vec<i64>,
    permutation: Vec<i64>,
) -> OpRequest {
    let mut attrs = AttrMap::new();
    attrs.insert("permutation".into(), AttrValue::Ints(permutation));
    OpRequest {
        operator: OperatorId("transpose".into()),
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
        source: SourceSpan { file: "cuda_transpose.rs".into(), line: 1, column: 1 },
    }
}

#[test]
fn runtime_cuda_transpose_jits_launches_and_matches_cpu_reference() {
    let Some(cuda) = cuda_device()
    else {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    };
    let input = Tensor::<f32, 2>::from_slice(&cuda, [2, 3], &[1.5, -2.0, 3.25, 4.0, 0.5, -6.0]).expect("upload CUDA input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-transpose"));
    let result = runtime
        .execute(request(input.handle(), vec![3, 2], DType::F32, Layout::Contiguous, vec![2, 1], vec![1, 0]))
        .expect("Driver JIT and CUDA launch")
        .wait()
        .expect("CUDA event completion and D2H readback");
    let actual = result.outputs[0].to_vec_f32().expect("D2H output readback");
    let expected = [1.5, 4.0, -2.0, 0.5, 3.25, -6.0];
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!((actual - expected).abs() <= 1e-6, "index {index}: cuda={actual} cpu={expected}");
    }
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn runtime_cuda_transpose_rejects_dtype_layout_permutation_and_shape() {
    let Some(cuda) = cuda_device()
    else {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    };
    let input = Tensor::<f32, 2>::from_slice(&cuda, [2, 3], &[1.0; 6]).expect("upload CUDA input");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-transpose-negative"));
    for (dtype, layout, strides, shape, perm, needle) in [
        (DType::F16, Layout::Contiguous, vec![2, 1], vec![3, 2], vec![1, 0], "F32"),
        (DType::F32, Layout::Strided, vec![1, 2], vec![3, 2], vec![1, 0], "contiguous"),
        (DType::F32, Layout::Contiguous, vec![2, 1], vec![3, 2], vec![0, 1], "permutation"),
        (DType::F32, Layout::Contiguous, vec![2, 1], vec![2, 3], vec![1, 0], "output shape"),
    ] {
        let error = runtime.execute(request(input.handle(), shape, dtype, layout, strides, perm)).unwrap_err();
        assert_eq!(error.phase, "contract");
        assert!(error.message.contains(needle), "{}", error.message);
    }
}
