use titan_backend_cuda::{
    CudaCompiler, concat_f32_abi, conv2d_f32_abi, elementwise_add_f32_abi, gemm_f32_abi, layer_norm_f32_abi,
    reduction_sum_f32_abi, resize_nearest2d_f32_abi, silu_f32_abi, softmax_f32_abi, transpose_f32_abi,
};
use titan_kernel::{
    AddressSpace, BasicBlock, BlockId, Instruction, IrType, KernelAbi, KernelError, KernelModule, TargetCompiler, ValueId,
};
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

fn empty_macro_ir(kernel_id: &str, abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId(kernel_id.into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

fn compile_source(kernel_id: &str, abi: KernelAbi) -> String {
    let bytes = CudaCompiler.compile(&empty_macro_ir(kernel_id, abi.clone()), &abi, &fingerprint("sm_86")).unwrap();
    assert_eq!(bytes.last(), Some(&0));
    std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap().to_owned()
}

fn assert_atomic_prologue(source: &str) {
    assert!(source.contains("mov.u32"), "expected tid/cta moves");
    assert!(source.contains("mad.lo"), "expected linear index mad");
    assert!(source.contains("setp.ge.u32"), "expected bounds predicate");
    assert!(source.contains("bra "), "expected bounds branch");
    assert!(source.matches(";\n").count() > 8, "expected many atomic instruction lines");
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
fn silu_and_gemm_emit_atomic_ptx_lines_not_single_macro() {
    let silu_source = compile_source("silu.f32", silu_f32_abi());
    assert!(silu_source.contains("ex2.approx.f32"));
    assert!(silu_source.contains("div.rn.f32"));
    assert_atomic_prologue(&silu_source);

    let gemm_source = compile_source("gemm.f32", gemm_f32_abi());
    assert!(gemm_source.contains("fma.rn.f32"));
    assert!(gemm_source.contains("mad.lo.u32"));
    assert_atomic_prologue(&gemm_source);
}

#[test]
fn softmax_concat_transpose_emit_atomic_prologues() {
    let softmax = compile_source("softmax.f32", softmax_f32_abi());
    assert!(softmax.contains("max.f32"));
    assert!(softmax.contains("ex2.approx.f32"));
    assert_atomic_prologue(&softmax);

    let concat = compile_source("concat.f32", concat_f32_abi());
    assert!(concat.contains("ld.global.f32"));
    assert!(concat.contains("st.global.f32"));
    assert_atomic_prologue(&concat);

    let transpose = compile_source("transpose.f32", transpose_f32_abi());
    assert!(transpose.contains("div.u32"));
    assert!(transpose.contains("rem.u32"));
    assert_atomic_prologue(&transpose);
}

#[test]
fn reduction_resize_layernorm_conv_emit_shared_prologues() {
    let reduction = compile_source("reduction.sum.f32", reduction_sum_f32_abi());
    assert!(reduction.contains("add.rn.f32"));
    assert_atomic_prologue(&reduction);

    let resize = compile_source("resize.nearest2d.f32", resize_nearest2d_f32_abi());
    assert!(resize.contains("div.u32"));
    assert_atomic_prologue(&resize);

    let layer_norm = compile_source("layer_norm.f32", layer_norm_f32_abi());
    assert!(layer_norm.contains("rsqrt.approx.f32"));
    assert_atomic_prologue(&layer_norm);

    let conv = compile_source("conv2d.f32", conv2d_f32_abi());
    assert!(conv.contains("fma.rn.f32"));
    assert_atomic_prologue(&conv);
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
    assert!(matches!(
        CudaCompiler.compile(&add_ir(abi.clone()), &abi, &fingerprint("sm_61")),
        Err(KernelError::Unsupported(_))
    ));
}
