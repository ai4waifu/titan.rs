use std::sync::Arc;

use titan_backend_cpu::CpuDriver;
use titan_backend_cuda::{CudaCompiler, CudaDriver, elementwise_add_f32_abi};
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::{BackendDriver, EncodedLaunchArgs, LaunchGeometry};
use titan_kernel::{AddressSpace, BasicBlock, BlockId, Instruction, IrType, KernelModule, ValueId};
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{
    AliasContract, AttrMap, BackendId, DType, DeviceFingerprint, DeviceId, Layout, MemoryEffect, OperatorId, Shape, SourceSpan,
    Strides,
};

fn source() -> SourceSpan {
    SourceSpan { file: "cuda_add.rs".into(), line: 1, column: 1 }
}

fn request(lhs: titan_tensor::TensorHandle, rhs: titan_tensor::TensorHandle, shape: Vec<u64>) -> OpRequest {
    OpRequest {
        operator: OperatorId("elementwise.add.f32".into()),
        inputs: vec![lhs, rhs],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
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

fn no_cuda_environment(error: &titan_hal::HalError) -> bool {
    error.operation == "load_driver"
        || ((error.operation == "cuInit" || error.operation == "cuDeviceGetCount") && error.detail.contains("status 100"))
}

fn cuda_add_ir(abi: titan_kernel::KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: titan_types::KernelId("elementwise.add.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock {
            id: BlockId(0),
            params: vec![],
            instructions: vec![
                (
                    ValueId(0),
                    Instruction::Parameter {
                        index: 0,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (
                    ValueId(1),
                    Instruction::Parameter {
                        index: 1,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (
                    ValueId(2),
                    Instruction::Parameter {
                        index: 2,
                        ty: IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
                    },
                ),
                (ValueId(3), Instruction::Parameter { index: 3, ty: IrType::I32 }),
                (ValueId(4), Instruction::Load { ptr: ValueId(0), ty: IrType::F32 }),
                (ValueId(5), Instruction::Load { ptr: ValueId(1), ty: IrType::F32 }),
                (ValueId(6), Instruction::Add { lhs: ValueId(4), rhs: ValueId(5) }),
                (ValueId(7), Instruction::Store { ptr: ValueId(2), value: ValueId(6) }),
            ],
        }],
        abi,
    }
}

#[test]
fn cuda_add_matches_the_cpu_reference() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let cuda_device = Device::from_session(driver.open(fingerprint.device).expect("open CUDA primary context"));
    let cpu_driver = Arc::new(CpuDriver);
    let cpu_device = Device::from_session(cpu_driver.open(cpu_driver.enumerate().unwrap()[0].device).unwrap());
    let lhs = [1.25_f32, -2.0, 3.5, 0.0, 99.0, -16.25, 8.0, 0.75, 4.0, -7.0, 2.5, 12.0, -1.0];
    let rhs = [-0.25_f32, 4.0, 1.5, 8.0, -9.0, 0.25, -3.0, 5.25, -4.0, 7.5, 0.5, -2.0, 11.0];
    let cuda_lhs = Tensor::<f32, 1>::from_slice(&cuda_device, [lhs.len()], &lhs).unwrap();
    let cuda_rhs = Tensor::<f32, 1>::from_slice(&cuda_device, [rhs.len()], &rhs).unwrap();
    let cpu_lhs = Tensor::<f32, 1>::from_slice(&cpu_device, [lhs.len()], &lhs).unwrap();
    let cpu_rhs = Tensor::<f32, 1>::from_slice(&cpu_device, [rhs.len()], &rhs).unwrap();
    let mut runtime = Runtime::open(std::env::temp_dir().join("titan-cuda-add-test"));

    let cuda = runtime.execute(request(cuda_lhs.handle(), cuda_rhs.handle(), vec![lhs.len() as u64])).unwrap().wait().unwrap();
    let cpu = runtime.execute(request(cpu_lhs.handle(), cpu_rhs.handle(), vec![lhs.len() as u64])).unwrap().wait().unwrap();

    assert_eq!(cuda.outputs[0].to_vec_f32().unwrap(), cpu.outputs[0].to_vec_f32().unwrap());
    assert_eq!(runtime.artifact_cache_stats(), (0, 2));
}

#[test]
fn cuda_compiler_rejects_an_abi_that_does_not_match_the_module() {
    let abi = elementwise_add_f32_abi();
    let mut incompatible = abi.clone();
    incompatible.launch.block_size = 64;
    let fingerprint = DeviceFingerprint {
        device: DeviceId { backend: BackendId::Cuda, ordinal: 0 },
        model: "test GPU".into(),
        driver: "test".into(),
        capability_revision: "sm_86".into(),
    };
    let error = CudaCompiler.compile_artifact(&cuda_add_ir(abi), &incompatible, &fingerprint).unwrap_err();
    assert!(matches!(error, titan_kernel::KernelError::InvalidAbi(_)));
}

#[test]
fn cuda_driver_reports_load_and_launch_contract_errors() {
    let driver = match CudaDriver::open() {
        Ok(driver) => driver,
        Err(error) if no_cuda_environment(&error) => {
            eprintln!("SKIP: CUDA Driver API unavailable: {error}");
            return;
        }
        Err(error) => panic!("opening CUDA Driver API failed: {error}"),
    };
    let devices = driver.enumerate().expect("enumerate CUDA devices");
    let Some(fingerprint) = devices.first()
    else {
        eprintln!("SKIP: CUDA Driver API reported no GPU");
        return;
    };
    let session = driver.open(fingerprint.device).expect("open CUDA primary context");
    let abi = elementwise_add_f32_abi();
    let metadata = abi.launch_metadata(&titan_types::KernelId("elementwise.add.f32".into())).unwrap();

    let load_error = session.load(b"not PTX\0", &abi.abi_hash(), metadata.clone()).unwrap_err();
    assert_eq!(load_error.operation, "load");

    let artifact = CudaCompiler.compile_artifact(&cuda_add_ir(abi.clone()), &abi, session.fingerprint()).unwrap();
    let kernel = session.load(artifact.ptx(), artifact.abi_hash(), artifact.metadata().clone()).unwrap();
    let stream = session.create_stream().unwrap();
    let arguments = EncodedLaunchArgs::try_new(Vec::new(), b"wrong ABI".to_vec(), []).unwrap();
    let launch_error = session
        .launch(
            stream.as_ref(),
            kernel.as_ref(),
            &arguments,
            &LaunchGeometry { grid: [1, 1, 1], block: [128, 1, 1], shared_bytes: 0 },
        )
        .unwrap_err();
    assert_eq!(launch_error.operation, "launch");
}
