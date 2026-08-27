//! ABI and IR validation for CUDA PTX lowering.

use titan_kernel::{AddressSpace, Instruction as IrInstruction, IrType, KernelAbi, KernelError, KernelModule};
use titan_types::{BackendId, DType, DeviceFingerprint};

use super::ast::{ElementwiseOperation, FmaAddend};

fn require_matching_module_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    Ok(())
}

fn require_expected_abi(abi: &KernelAbi, expected: &KernelAbi, message: impl Into<String>) -> Result<(), KernelError> {
    if abi != expected {
        return Err(KernelError::InvalidAbi(message.into()));
    }
    Ok(())
}

fn require_empty_entry_block(ir: &KernelModule, message: impl Into<String>) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(message.into()));
    }
    Ok(())
}

fn require_single_empty_block(ir: &KernelModule, message: impl Into<String>) -> Result<(), KernelError> {
    if ir.blocks.len() != 1 || !ir.blocks[0].instructions.is_empty() {
        return Err(KernelError::Unsupported(message.into()));
    }
    Ok(())
}

fn require_kernel_abi(
    ir: &KernelModule,
    abi: &KernelAbi,
    expected: &KernelAbi,
    message: impl Into<String>,
) -> Result<(), KernelError> {
    require_matching_module_abi(ir, abi)?;
    require_expected_abi(abi, expected, message)
}

pub(super) fn validate_slice_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(ir, abi, &crate::slice_f32_abi(), "slice.f32 ABI mismatch")
}

pub(super) fn validate_slice_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_single_empty_block(ir, "slice.f32 requires canonical empty IR entry block")
}

pub(super) fn validate_gemm_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::gemm_f32_abi(),
        "CUDA GEMM lowering requires three aligned f32 buffers and i32 M, N, K scalars",
    )
}

pub(super) fn validate_gemm_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA GEMM lowering requires the canonical empty gemm.f32 IR entry block")
}

pub(super) fn validate_conv2d_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::conv2d_f32_abi(),
        "CUDA Conv2D lowering requires four aligned f32 buffers and fixed i32 geometry scalars",
    )
}

pub(super) fn validate_conv2d_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA Conv2D lowering requires the canonical empty conv2d.f32 IR entry block")
}

pub(super) fn validate_attention_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::scaled_dot_product_attention_f32_abi(),
        "CUDA attention lowering requires four aligned f32 buffers and i32 B, H, Tq, Tk, D scalars",
    )
}

pub(super) fn validate_attention_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(
        ir,
        "CUDA attention lowering requires the canonical empty scaled_dot_product_attention.f32 IR entry block",
    )
}

pub(super) fn validate_broadcast_add_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::broadcast_add_f32_abi(),
        "CUDA broadcast add lowering requires three aligned f32 buffers, output count, and padded shape scalars",
    )
}

pub(super) fn validate_broadcast_add_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA broadcast add lowering requires the canonical empty broadcast.add.f32 IR entry block")
}

pub(super) fn validate_silu_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::silu_f32_abi(),
        "CUDA SiLU lowering requires one aligned f32 input buffer, one aligned f32 output buffer, and one i32 element-count scalar",
    )
}

pub(super) fn validate_silu_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA SiLU lowering requires the canonical empty silu.f32 IR entry block")
}

pub(super) fn validate_gelu_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::gelu_f32_abi(),
        "CUDA GELU lowering requires one aligned f32 input buffer, one aligned f32 output buffer, and one i32 element-count scalar",
    )
}

pub(super) fn validate_gelu_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA GELU lowering requires the canonical empty gelu.f32 IR entry block")
}

pub(super) fn validate_quick_gelu_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::quick_gelu_f32_abi(),
        "CUDA QuickGELU lowering requires one aligned f32 input buffer, one aligned f32 output buffer, one i32 element-count scalar, and one f32 slope scalar",
    )
}

pub(super) fn validate_quick_gelu_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA QuickGELU lowering requires the canonical empty quick_gelu.f32 IR entry block")
}

pub(super) fn validate_softmax_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::softmax_f32_abi(),
        "CUDA softmax lowering requires one aligned f32 input buffer, one aligned f32 output buffer, and i32 row/column scalars",
    )
}

pub(super) fn validate_softmax_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA softmax lowering requires the canonical empty softmax.f32 IR entry block")
}

pub(super) fn validate_reduction_sum_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::reduction_sum_f32_abi(),
        "CUDA reduction.sum requires two aligned f32 buffers and i32 row/axis scalars",
    )
}

pub(super) fn validate_reduction_sum_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA reduction.sum lowering requires the canonical empty reduction.sum.f32 IR entry block")
}

pub(super) fn validate_concat_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::concat_f32_abi(),
        "CUDA concat requires three aligned f32 buffers and i32 lhs/total element scalars",
    )
}

pub(super) fn validate_concat_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA concat lowering requires the canonical empty concat.f32 IR entry block")
}

pub(super) fn validate_transpose_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::transpose_f32_abi(),
        "CUDA transpose requires two aligned f32 buffers and i32 row/column scalars",
    )
}

pub(super) fn validate_transpose_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA transpose lowering requires the canonical empty transpose.f32 IR entry block")
}

pub(super) fn validate_resize_nearest2d_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::resize_nearest2d_f32_abi(),
        "CUDA nearest resize requires two f32 buffers and N,C,input/output H,W i32 scalars",
    )
}

pub(super) fn validate_resize_nearest2d_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA nearest resize requires the canonical empty resize.nearest2d.f32 IR entry block")
}

pub(super) fn validate_layer_norm_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::layer_norm_f32_abi(),
        "CUDA LayerNorm lowering requires four aligned f32 buffers, rows/cols/flags i32 scalars, and one f32 epsilon scalar",
    )
}

pub(super) fn validate_layer_norm_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA LayerNorm lowering requires the canonical empty layer_norm.f32 IR entry block")
}

pub(super) fn validate_group_norm_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::group_norm_f32_abi(),
        "CUDA GroupNorm lowering requires four aligned f32 buffers; N, C, H, W, groups, gamma/beta flags i32 scalars; and one f32 epsilon scalar",
    )
}

pub(super) fn validate_group_norm_ir(ir: &KernelModule) -> Result<(), KernelError> {
    require_empty_entry_block(ir, "CUDA GroupNorm lowering requires the canonical empty group_norm.f32 IR entry block")
}

pub(super) fn validate_device(fingerprint: &DeviceFingerprint) -> Result<(), KernelError> {
    if fingerprint.device.backend != BackendId::Cuda {
        return Err(KernelError::Unsupported("CUDA lowering requires a CUDA device fingerprint".into()));
    }
    Ok(())
}

pub(super) fn validate_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    require_kernel_abi(
        ir,
        abi,
        &crate::elementwise_add_f32_abi(),
        "CUDA elementwise lowering requires three aligned f32 buffers and one i32 element-count scalar",
    )
}

pub(super) fn reject_non_global_pointers(ir: &KernelModule) -> Result<(), KernelError> {
    for block in &ir.blocks {
        for (_, instruction) in &block.instructions {
            if let IrInstruction::Parameter { ty: IrType::Pointer { address_space, .. }, .. } = instruction
                && *address_space != AddressSpace::Global
            {
                return Err(KernelError::Unsupported("CUDA lowering supports only global pointer parameters".into()));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_elementwise_ir(ir: &KernelModule) -> Result<ElementwiseOperation, KernelError> {
    if ir.blocks.len() != 1 || ir.blocks[0].id != ir.entry || !ir.blocks[0].params.is_empty() {
        return Err(KernelError::Unsupported("CUDA lowering requires one parameter-free entry block".into()));
    }
    let instructions = &ir.blocks[0].instructions;
    if instructions.len() != 8 {
        return Err(KernelError::Unsupported(
            "CUDA lowering requires Parameter, Parameter, Parameter, Parameter, Load, Load, arithmetic, Store".into(),
        ));
    }
    let parameter_values = [instructions[0].0, instructions[1].0, instructions[2].0, instructions[3].0];
    for (position, (value, instruction)) in instructions[..4].iter().enumerate() {
        if *value != parameter_values[position] || !matches_parameter(instruction, position as u32) {
            return Err(KernelError::Unsupported("CUDA lowering received an unsupported parameter declaration".into()));
        }
    }
    let (left, left_instruction) = &instructions[4];
    let (right, right_instruction) = &instructions[5];
    if !matches!(left_instruction, IrInstruction::Load { ptr, ty: IrType::F32 } if *ptr == parameter_values[0])
        || !matches!(right_instruction, IrInstruction::Load { ptr, ty: IrType::F32 } if *ptr == parameter_values[1])
    {
        return Err(KernelError::Unsupported(
            "CUDA lowering requires f32 loads from the first two global buffer parameters".into(),
        ));
    }

    let (result, arithmetic) = &instructions[6];
    let operation = match arithmetic {
        IrInstruction::Add { lhs, rhs } if *lhs == *left && *rhs == *right => ElementwiseOperation::Add,
        IrInstruction::Mul { lhs, rhs } if *lhs == *left && *rhs == *right => ElementwiseOperation::Mul,
        IrInstruction::Fma { a, b, c } if *a == *left && *b == *right && (*c == *left || *c == *right) => {
            ElementwiseOperation::Fma { addend: if *c == *left { FmaAddend::Left } else { FmaAddend::Right } }
        }
        _ => {
            return Err(KernelError::Unsupported(
                "CUDA lowering supports only f32 Add, Mul, or Fma over the two loaded values".into(),
            ));
        }
    };
    if !matches!(instructions[7].1, IrInstruction::Store { ptr, value } if ptr == parameter_values[2] && value == *result) {
        return Err(KernelError::Unsupported(
            "CUDA lowering requires storing the arithmetic result to the third global buffer parameter".into(),
        ));
    }
    Ok(operation)
}

fn matches_parameter(instruction: &IrInstruction, index: u32) -> bool {
    let expected = match index {
        0..=2 => IrType::Pointer { address_space: AddressSpace::Global, dtype: DType::F32 },
        3 => IrType::I32,
        _ => return false,
    };
    matches!(instruction, IrInstruction::Parameter { index: actual, ty } if *actual == index && *ty == expected)
}
