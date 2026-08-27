use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_conv2d.rs".into(), line: 1, column: 1 }
}

fn conv2d_request(
    input: titan_tensor::TensorHandle,
    weight: titan_tensor::TensorHandle,
    bias: titan_tensor::TensorHandle,
) -> OpRequest {
    let mut attrs = AttrMap::new();
    for (name, value) in
        [("stride_h", 2), ("stride_w", 2), ("pad_h", 1), ("pad_w", 1), ("dilation_h", 2), ("dilation_w", 2), ("groups", 2)]
    {
        attrs.insert(name.into(), AttrValue::Int(value));
    }
    OpRequest {
        operator: OperatorId("conv2d".into()),
        inputs: vec![input, weight, bias],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(vec![18, 9, 3, 1]),
            shape: Shape(vec![1, 2, 3, 3]),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs,
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: source(),
    }
}

fn no_cuda_environment(error: &titan_hal::HalError) -> bool {
    error.operation == "load_driver"
        || ((error.operation == "cuInit" || error.operation == "cuDeviceGetCount") && error.detail.contains("status 100"))
}

#[test]
fn runtime_cuda_conv2d_matches_cpu_reference_and_returns_readable_tensor_handle() {
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
    let input = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
        -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., -13., -14., -15., -16., -17., -18., -19., -20., -21.,
        -22., -23., -24., -25.,
    ];
    let weight = [1., 0.5, -1., 2., -0.5, 1.5, 0.25, -1.];
    let bias = [0.75, -1.25];
    let cuda_input = Tensor::<f32, 4>::from_slice(&cuda_device, [1, 2, 5, 5], &input).expect("upload CUDA input");
    let cuda_weight = Tensor::<f32, 4>::from_slice(&cuda_device, [2, 1, 2, 2], &weight).expect("upload CUDA weight");
    let cuda_bias = Tensor::<f32, 1>::from_slice(&cuda_device, [2], &bias).expect("upload CUDA bias");
    let cpu_input = Tensor::<f32, 4>::from_slice(&cpu_device, [1, 2, 5, 5], &input).expect("upload CPU input");
    let cpu_weight = Tensor::<f32, 4>::from_slice(&cpu_device, [2, 1, 2, 2], &weight).expect("upload CPU weight");
    let cpu_bias = Tensor::<f32, 1>::from_slice(&cpu_device, [2], &bias).expect("upload CPU bias");
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-runtime-cuda-conv2d"));

    let cuda = runtime
        .execute(conv2d_request(cuda_input.handle(), cuda_weight.handle(), cuda_bias.handle()))
        .expect("Runtime CUDA Conv2D dispatch")
        .wait()
        .expect("Runtime CUDA Conv2D completion");
    let cpu = runtime
        .execute(conv2d_request(cpu_input.handle(), cpu_weight.handle(), cpu_bias.handle()))
        .expect("Runtime CPU Conv2D dispatch")
        .wait()
        .expect("Runtime CPU Conv2D completion");

    assert_eq!(cuda.outputs[0].device().backend, titan_types::BackendId::Cuda);
    let cuda_values = cuda.outputs[0].to_vec_f32().expect("read CUDA TensorHandle output");
    let cpu_values = cpu.outputs[0].to_vec_f32().expect("read CPU TensorHandle output");
    assert_eq!(cuda_values.len(), cpu_values.len());
    for (index, (cuda_value, cpu_value)) in cuda_values.iter().zip(cpu_values.iter()).enumerate() {
        assert!((cuda_value - cpu_value).abs() < 1e-5, "output {index}: CUDA={cuda_value}, CPU={cpu_value}");
    }
}
