use titan_backend_cuda::{CudaCompiler, elementwise_add_f32_abi};
use titan_kernel::{AddressSpace, BasicBlock, BlockId, Instruction, IrType, KernelAbi, KernelError, KernelModule, TargetCompiler, ValueId};
use titan_types::{BackendId, DType, DeviceFingerprint, DeviceId, KernelId};

fn fingerprint(capability_revision: &str) -> DeviceFingerprint {
    DeviceFingerprint {
        device: DeviceId { backend: BackendId::Cuda, ordinal: 0 },
        model: "test GPU".into(),
        driver: "test".into(),
        capability_revision: capability_revision.into(),
    }
}

fn add_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("elementwise.add.f32".into()),
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
fn direct_compiler_lowers_add_ir_to_nul_terminated_typed_ptx() {
    let abi = elementwise_add_f32_abi();
    let bytes = CudaCompiler.compile(&add_ir(abi.clone()), &abi, &fingerprint("sm_86")).unwrap();
    assert_eq!(bytes.last(), Some(&0));
    let source = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
    assert!(source.contains(".target sm_86"));
    assert!(source.contains(".entry titan_elementwise_add_f32"));
    assert!(source.contains("add.rn.f32"));
}

#[test]
fn lowering_rejects_invalid_abi_non_global_pointer_and_old_target() {
    let abi = elementwise_add_f32_abi();
    let mut invalid_abi = abi.clone();
    invalid_abi.launch.block_size = 64;
    assert!(matches!(
        CudaCompiler.compile(&add_ir(abi.clone()), &invalid_abi, &fingerprint("sm_86")),
        Err(KernelError::InvalidAbi(_))
    ));

    let mut non_global = add_ir(abi.clone());
    non_global.blocks[0].instructions[0].1 =
        Instruction::Parameter { index: 0, ty: IrType::Pointer { address_space: AddressSpace::Shared, dtype: DType::F32 } };
    assert!(matches!(CudaCompiler.compile(&non_global, &abi, &fingerprint("sm_86")), Err(KernelError::Unsupported(_))));
    assert!(matches!(CudaCompiler.compile(&add_ir(abi.clone()), &abi, &fingerprint("sm_61")), Err(KernelError::Unsupported(_))));
}
