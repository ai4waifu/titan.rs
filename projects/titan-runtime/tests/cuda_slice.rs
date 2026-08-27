use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn device() -> Option<Device> {
    let driver = CudaDriver::open().ok()?;
    let fp = driver.enumerate().ok()?.first()?.clone();
    Some(Device::from_session(driver.open(fp.device).ok()?))
}

fn request(
    input: titan_tensor::TensorHandle,
    dtype: DType,
    layout: Layout,
    shape: Vec<u64>,
    axes: Vec<i64>,
    starts: Vec<i64>,
    stops: Vec<i64>,
    steps: Vec<i64>,
) -> OpRequest {
    let mut attrs = AttrMap::new();
    attrs.insert("axes".into(), AttrValue::Ints(axes));
    attrs.insert("starts".into(), AttrValue::Ints(starts));
    attrs.insert("stops".into(), AttrValue::Ints(stops));
    attrs.insert("steps".into(), AttrValue::Ints(steps));
    OpRequest {
        operator: OperatorId("slice".into()),
        inputs: vec![input],
        outputs: vec![TensorSpec {
            dtype,
            shape: Shape(shape),
            strides: Strides(vec![1]),
            layout,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: SourceSpan { file: "cuda_slice.rs".into(), line: 1, column: 1 },
    }
}

#[test]
fn cuda_slice_matches_cpu_reference_and_caches() {
    let Some(cuda) = device()
    else {
        eprintln!("SKIP: CUDA unavailable");
        return;
    };
    let input = Tensor::<f32, 1>::from_slice(&cuda, [6], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-slice"));
    let result = runtime
        .execute(request(input.handle(), DType::F32, Layout::Contiguous, vec![3], vec![0], vec![1], vec![4], vec![1]))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(result.outputs[0].to_vec_f32().unwrap(), vec![2., 3., 4.]);
    assert_eq!(runtime.artifact_cache_stats(), (0, 1));
}

#[test]
fn cuda_slice_rejects_dtype_layout_axis_step_and_shape() {
    let Some(cuda) = device()
    else {
        return;
    };
    let input = Tensor::<f32, 1>::from_slice(&cuda, [4], &[1., 2., 3., 4.]).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-slice-negative"));
    for (dtype, layout, axes, steps, shape) in [
        (DType::F16, Layout::Contiguous, vec![0], vec![1], vec![2]),
        (DType::F32, Layout::Strided, vec![0], vec![1], vec![2]),
        (DType::F32, Layout::Contiguous, vec![1], vec![1], vec![2]),
        (DType::F32, Layout::Contiguous, vec![0], vec![2], vec![2]),
        (DType::F32, Layout::Contiguous, vec![0], vec![1], vec![3]),
    ] {
        let error = runtime
            .execute(request(
                input.handle(),
                dtype,
                layout,
                shape.into_iter().map(|v| v as u64).collect(),
                axes,
                vec![0],
                vec![2],
                steps,
            ))
            .unwrap_err();
        assert_eq!(error.phase, "contract");
    }
}
