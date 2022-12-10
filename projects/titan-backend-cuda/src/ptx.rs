//! Private typed PTX AST and lowering for the CUDA backend.

use std::{fmt, num::NonZeroU8};

use titan_kernel::{AddressSpace, Instruction as IrInstruction, IrType, KernelAbi, KernelError, KernelModule};
use titan_types::{BackendId, DType, DeviceFingerprint};

const MINIMUM_SM: u16 = 70;

/// PTX bytes that are valid to pass to `cuModuleLoadDataEx`.
pub(super) struct PtxArtifact(Vec<u8>);

impl PtxArtifact {
    pub(super) fn from_driver_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut ptx = bytes.to_vec();
        if ptx.is_empty() {
            return Err("PTX artifact is empty");
        }
        if let Some(nul) = ptx.iter().position(|byte| *byte == 0) {
            if nul + 1 != ptx.len() {
                return Err("PTX artifact contains an interior NUL byte");
            }
        }
        else {
            ptx.push(0);
        }
        if !ptx[..ptx.len() - 1].windows(b".version".len()).any(|window| window == b".version") {
            return Err("artifact is not PTX source");
        }
        Ok(Self(ptx))
    }

    pub(super) fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

pub(super) struct LoweredPtx {
    artifact: PtxArtifact,
    entry: Identifier,
}

impl LoweredPtx {
    pub(super) fn entry(&self) -> &str {
        self.entry.as_str()
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.artifact.0
    }
}

/// Lowers the currently supported single-block global-f32 elementwise subset.
pub(super) fn lower(ir: &KernelModule, abi: &KernelAbi, fingerprint: &DeviceFingerprint) -> Result<LoweredPtx, KernelError> {
    validate_device(fingerprint)?;
    ir.verify()?;
    let target = Target::from_fingerprint(fingerprint)?;
    let entry_name = Identifier::from_kernel_id(&ir.kernel_id.0)?;
    let entry = if ir.kernel_id.0 == "gemm.f32" {
        validate_gemm_abi(ir, abi)?;
        validate_gemm_ir(ir)?;
        Entry::gemm_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "conv2d.f32" {
        validate_conv2d_abi(ir, abi)?;
        validate_conv2d_ir(ir)?;
        Entry::conv2d_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "scaled_dot_product_attention.f32" {
        validate_attention_abi(ir, abi)?;
        validate_attention_ir(ir)?;
        Entry::scaled_dot_product_attention_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "broadcast.add.f32" {
        validate_broadcast_add_abi(ir, abi)?;
        validate_broadcast_add_ir(ir)?;
        Entry::broadcast_add_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "silu.f32" {
        validate_silu_abi(ir, abi)?;
        validate_silu_ir(ir)?;
        Entry::silu_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "gelu.f32" {
        validate_gelu_abi(ir, abi)?;
        validate_gelu_ir(ir)?;
        Entry::gelu_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "quick_gelu.f32" {
        validate_quick_gelu_abi(ir, abi)?;
        validate_quick_gelu_ir(ir)?;
        Entry::quick_gelu_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "softmax.f32" {
        validate_softmax_abi(ir, abi)?;
        validate_softmax_ir(ir)?;
        Entry::softmax_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "reduction.sum.f32" {
        validate_reduction_sum_abi(ir, abi)?;
        validate_reduction_sum_ir(ir)?;
        Entry::reduction_sum_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "concat.f32" {
        validate_concat_abi(ir, abi)?;
        validate_concat_ir(ir)?;
        Entry::concat_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "transpose.f32" {
        validate_transpose_abi(ir, abi)?;
        validate_transpose_ir(ir)?;
        Entry::transpose_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "slice.f32" {
        validate_slice_abi(ir, abi)?;
        validate_slice_ir(ir)?;
        Entry::slice_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "resize.nearest2d.f32" {
        validate_resize_nearest2d_abi(ir, abi)?;
        validate_resize_nearest2d_ir(ir)?;
        Entry::resize_nearest2d_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "layer_norm.f32" {
        validate_layer_norm_abi(ir, abi)?;
        validate_layer_norm_ir(ir)?;
        Entry::layer_norm_f32(entry_name.clone())
    }
    else if ir.kernel_id.0 == "group_norm.f32" {
        validate_group_norm_abi(ir, abi)?;
        validate_group_norm_ir(ir)?;
        Entry::group_norm_f32(entry_name.clone())
    }
    else {
        validate_abi(ir, abi)?;
        reject_non_global_pointers(ir)?;
        let operation = validate_elementwise_ir(ir)?;
        Entry::elementwise_f32(entry_name.clone(), operation)
    };
    let module = PtxModule { version: PtxVersion::V80, target, address_size: AddressSize::Bits64, entry };
    let source = module.to_string();
    let artifact = PtxArtifact::from_driver_bytes(source.as_bytes())
        .map_err(|detail| KernelError::Compile(format!("typed PTX emitter produced invalid artifact: {detail}")))?;
    Ok(LoweredPtx { artifact, entry: entry_name })
}

fn validate_slice_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::slice_f32_abi() {
        return Err(KernelError::InvalidAbi("slice.f32 ABI mismatch".into()));
    }
    Ok(())
}

fn validate_slice_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1 || !ir.blocks[0].instructions.is_empty() {
        return Err(KernelError::Unsupported("slice.f32 requires canonical empty IR entry block".into()));
    }
    Ok(())
}

fn validate_gemm_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::gemm_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA GEMM lowering requires three aligned f32 buffers and i32 M, N, K scalars".into(),
        ));
    }
    Ok(())
}

fn validate_gemm_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported("CUDA GEMM lowering requires the canonical empty gemm.f32 IR entry block".into()));
    }
    Ok(())
}

fn validate_conv2d_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::conv2d_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA Conv2D lowering requires four aligned f32 buffers and fixed i32 geometry scalars".into(),
        ));
    }
    Ok(())
}

fn validate_conv2d_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA Conv2D lowering requires the canonical empty conv2d.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_attention_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::scaled_dot_product_attention_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA attention lowering requires four aligned f32 buffers and i32 B, H, Tq, Tk, D scalars".into(),
        ));
    }
    Ok(())
}

fn validate_attention_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA attention lowering requires the canonical empty scaled_dot_product_attention.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_broadcast_add_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::broadcast_add_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA broadcast add lowering requires three aligned f32 buffers, output count, and padded shape scalars".into(),
        ));
    }
    Ok(())
}

fn validate_broadcast_add_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA broadcast add lowering requires the canonical empty broadcast.add.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_silu_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::silu_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA SiLU lowering requires one aligned f32 input buffer, one aligned f32 output buffer, and one i32 element-count scalar"
                .into(),
        ));
    }
    Ok(())
}

fn validate_silu_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported("CUDA SiLU lowering requires the canonical empty silu.f32 IR entry block".into()));
    }
    Ok(())
}

fn validate_gelu_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::gelu_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA GELU lowering requires one aligned f32 input buffer, one aligned f32 output buffer, and one i32 element-count scalar"
                .into(),
        ));
    }
    Ok(())
}

fn validate_gelu_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported("CUDA GELU lowering requires the canonical empty gelu.f32 IR entry block".into()));
    }
    Ok(())
}

fn validate_quick_gelu_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::quick_gelu_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA QuickGELU lowering requires one aligned f32 input buffer, one aligned f32 output buffer, one i32 element-count scalar, and one f32 slope scalar"
                .into(),
        ));
    }
    Ok(())
}

fn validate_quick_gelu_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA QuickGELU lowering requires the canonical empty quick_gelu.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_softmax_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::softmax_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA softmax lowering requires one aligned f32 input buffer, one aligned f32 output buffer, and i32 row/column scalars"
                .into(),
        ));
    }
    Ok(())
}

fn validate_softmax_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA softmax lowering requires the canonical empty softmax.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_reduction_sum_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::reduction_sum_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA reduction.sum requires two aligned f32 buffers and i32 row/axis scalars".into(),
        ));
    }
    Ok(())
}

fn validate_reduction_sum_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA reduction.sum lowering requires the canonical empty reduction.sum.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_concat_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::concat_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA concat requires three aligned f32 buffers and i32 lhs/total element scalars".into(),
        ));
    }
    Ok(())
}

fn validate_concat_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA concat lowering requires the canonical empty concat.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_transpose_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::transpose_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA transpose requires two aligned f32 buffers and i32 row/column scalars".into(),
        ));
    }
    Ok(())
}

fn validate_transpose_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA transpose lowering requires the canonical empty transpose.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_resize_nearest2d_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::resize_nearest2d_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA nearest resize requires two f32 buffers and N,C,input/output H,W i32 scalars".into(),
        ));
    }
    Ok(())
}

fn validate_layer_norm_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::layer_norm_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA LayerNorm lowering requires four aligned f32 buffers, rows/cols/flags i32 scalars, and one f32 epsilon scalar"
                .into(),
        ));
    }
    Ok(())
}

fn validate_group_norm_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    if abi != &super::group_norm_f32_abi() {
        return Err(KernelError::InvalidAbi(
            "CUDA GroupNorm lowering requires four aligned f32 buffers; N, C, H, W, groups, gamma/beta flags i32 scalars; and one f32 epsilon scalar"
                .into(),
        ));
    }
    Ok(())
}

fn validate_resize_nearest2d_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA nearest resize requires the canonical empty resize.nearest2d.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_layer_norm_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA LayerNorm lowering requires the canonical empty layer_norm.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_group_norm_ir(ir: &KernelModule) -> Result<(), KernelError> {
    if ir.blocks.len() != 1
        || ir.blocks[0].id != ir.entry
        || !ir.blocks[0].params.is_empty()
        || !ir.blocks[0].instructions.is_empty()
    {
        return Err(KernelError::Unsupported(
            "CUDA GroupNorm lowering requires the canonical empty group_norm.f32 IR entry block".into(),
        ));
    }
    Ok(())
}

fn validate_device(fingerprint: &DeviceFingerprint) -> Result<(), KernelError> {
    if fingerprint.device.backend != BackendId::Cuda {
        return Err(KernelError::Unsupported("CUDA lowering requires a CUDA device fingerprint".into()));
    }
    Ok(())
}

fn validate_abi(ir: &KernelModule, abi: &KernelAbi) -> Result<(), KernelError> {
    if &ir.abi != abi {
        return Err(KernelError::InvalidAbi("KernelModule ABI and compile ABI differ".into()));
    }
    let expected = super::elementwise_add_f32_abi();
    if abi != &expected {
        return Err(KernelError::InvalidAbi(
            "CUDA elementwise lowering requires three aligned f32 buffers and one i32 element-count scalar".into(),
        ));
    }
    Ok(())
}

fn reject_non_global_pointers(ir: &KernelModule) -> Result<(), KernelError> {
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

fn validate_elementwise_ir(ir: &KernelModule) -> Result<ElementwiseOperation, KernelError> {
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

#[derive(Clone, Copy)]
enum ElementwiseOperation {
    Add,
    Mul,
    Fma { addend: FmaAddend },
}

#[derive(Clone, Copy)]
enum FmaAddend {
    Left,
    Right,
}

#[derive(Clone, Debug)]
struct Identifier(String);

impl Identifier {
    fn from_kernel_id(kernel_id: &str) -> Result<Self, KernelError> {
        let mut name = String::from("titan_");
        for character in kernel_id.bytes() {
            match character {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => name.push(character as char),
                b'.' | b'-' => name.push('_'),
                _ => return Err(KernelError::Unsupported("kernel ID cannot be represented as a PTX identifier".into())),
            }
        }
        if kernel_id.is_empty() {
            return Err(KernelError::Unsupported("kernel ID cannot be empty".into()));
        }
        Ok(Self(name))
    }

    fn parameter(&self, index: ParameterIndex) -> Self {
        Self(format!("{}_param_{}", self.0, index.0))
    }

    fn suffix(&self, suffix: &str) -> Self {
        debug_assert!(suffix.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'));
        Self(format!("{}{}", self.0, suffix))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Copy)]
enum PtxVersion {
    V80,
}

impl fmt::Display for PtxVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::V80 => "8.0",
        })
    }
}

#[derive(Clone, Copy)]
struct Target(u16);

impl Target {
    fn from_fingerprint(fingerprint: &DeviceFingerprint) -> Result<Self, KernelError> {
        let capability = fingerprint
            .capability_revision
            .strip_prefix("sm_")
            .ok_or_else(|| KernelError::Unsupported("CUDA capability must have the sm_XX form".into()))?;
        let sm = capability
            .parse::<u16>()
            .map_err(|_| KernelError::Unsupported("CUDA capability must have numeric sm_XX suffix".into()))?;
        if sm < MINIMUM_SM {
            return Err(KernelError::Unsupported(format!("CUDA target sm_{sm} is below the supported sm_{MINIMUM_SM}")));
        }
        Ok(Self(sm))
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "sm_{}", self.0)
    }
}

#[derive(Clone, Copy)]
enum AddressSize {
    Bits64,
}

impl fmt::Display for AddressSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("64")
    }
}

struct PtxModule {
    version: PtxVersion,
    target: Target,
    address_size: AddressSize,
    entry: Entry,
}

impl fmt::Display for PtxModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, ".version {}", self.version)?;
        writeln!(formatter, ".target {}", self.target)?;
        writeln!(formatter, ".address_size {}", self.address_size)?;
        writeln!(formatter)?;
        write!(formatter, "{}", self.entry)
    }
}

#[derive(Clone, Copy)]
struct ParameterIndex(u8);

struct Parameter {
    name: Identifier,
    kind: ParameterKind,
}

#[derive(Clone, Copy)]
enum ParameterKind {
    GlobalF32Pointer,
    U32,
    F32,
}

impl fmt::Display for Parameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = match self.kind {
            ParameterKind::GlobalF32Pointer => ".u64",
            ParameterKind::U32 => ".u32",
            ParameterKind::F32 => ".f32",
        };
        write!(formatter, ".param {ty} {}", self.name)
    }
}

#[derive(Clone, Copy)]
enum RegisterClass {
    Predicate,
    B32,
    B64,
    F32,
}

impl RegisterClass {
    fn prefix(self) -> &'static str {
        match self {
            Self::Predicate => "%p",
            Self::B32 => "%r",
            Self::B64 => "%rd",
            Self::F32 => "%f",
        }
    }

    fn ptx_type(self) -> &'static str {
        match self {
            Self::Predicate => ".pred",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
            Self::F32 => ".f32",
        }
    }
}

#[derive(Clone, Copy)]
struct Register {
    class: RegisterClass,
    index: NonZeroU8,
}

impl Register {
    fn new(class: RegisterClass, index: u8) -> Self {
        Self { class, index: NonZeroU8::new(index).expect("PTX registers are one-indexed") }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", self.class.prefix(), self.index)
    }
}

struct RegisterDeclaration {
    class: RegisterClass,
    count: NonZeroU8,
}

impl fmt::Display for RegisterDeclaration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, ".reg {} {}<{}>;", self.class.ptx_type(), self.class.prefix(), self.count)
    }
}

#[derive(Clone)]
struct Label(Identifier);

impl fmt::Display for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

enum PtxInstruction {
    LoadParameterU64 {
        destination: Register,
        parameter: Identifier,
    },
    LoadParameterU32 {
        destination: Register,
        parameter: Identifier,
    },
    MoveCtaIdX {
        destination: Register,
    },
    MoveNtidX {
        destination: Register,
    },
    MoveTidX {
        destination: Register,
    },
    MultiplyAddLoS32 {
        destination: Register,
        left: Register,
        right: Register,
        addend: Register,
    },
    SetPredicateGeU32 {
        destination: Register,
        left: Register,
        right: Register,
    },
    BranchIf {
        predicate: Register,
        target: Label,
    },
    MultiplyWideU32 {
        destination: Register,
        left: Register,
        right: u8,
    },
    AddS64 {
        destination: Register,
        left: Register,
        right: Register,
    },
    LoadGlobalF32 {
        destination: Register,
        pointer: Register,
    },
    ArithmeticF32 {
        destination: Register,
        operation: ElementwiseOperation,
        left: Register,
        right: Register,
    },
    StoreGlobalF32 {
        pointer: Register,
        value: Register,
    },
    DefineLabel(Label),
    Return,
    GemmF32 {
        parameters: [Identifier; 6],
        done: Label,
        loop_label: Label,
    },
    Conv2dF32 {
        parameters: [Identifier; 21],
        done: Label,
        no_bias: Label,
        input_channel_loop: Label,
        kernel_h_loop: Label,
        kernel_w_loop: Label,
        next_kernel_w: Label,
        kernel_w_done: Label,
        kernel_h_done: Label,
        input_channel_done: Label,
    },
    ScaledDotProductAttentionF32 {
        parameters: [Identifier; 9],
        done: Label,
        max_loop: Label,
        max_inner_loop: Label,
        max_inner_done: Label,
        max_next: Label,
        max_done: Label,
        sum_loop: Label,
        sum_inner_loop: Label,
        sum_inner_done: Label,
        sum_next: Label,
        sum_done: Label,
        value_loop: Label,
        value_inner_loop: Label,
        value_inner_done: Label,
        value_next: Label,
        value_done: Label,
    },
    BroadcastAddF32 {
        parameters: [Identifier; 16],
        done: Label,
        lhs_dim_done: [Label; 4],
        rhs_dim_done: [Label; 4],
    },
    SiluF32 {
        parameters: [Identifier; 3],
        done: Label,
    },
    GeluF32 {
        parameters: [Identifier; 3],
        done: Label,
        negative: Label,
        signed_done: Label,
    },
    QuickGeluF32 {
        parameters: [Identifier; 4],
        done: Label,
    },
    SoftmaxF32 {
        parameters: [Identifier; 4],
        done: Label,
        max_loop: Label,
        max_done: Label,
        sum_loop: Label,
        sum_done: Label,
        normalize_loop: Label,
    },
    ReductionSumF32 {
        parameters: [Identifier; 4],
        done: Label,
        loop_label: Label,
    },
    ConcatF32 {
        parameters: [Identifier; 5],
        done: Label,
        right: Label,
    },
    TransposeF32 {
        parameters: [Identifier; 4],
        done: Label,
    },
    SliceF32 {
        parameters: [Identifier; 5],
        done: Label,
    },
    ResizeNearest2dF32 {
        parameters: [Identifier; 8],
        done: Label,
    },
    LayerNormF32 {
        parameters: [Identifier; 9],
        done: Label,
        mean_loop: Label,
        mean_done: Label,
        var_loop: Label,
        var_done: Label,
        store_loop: Label,
        no_gamma: Label,
        no_beta: Label,
    },
    GroupNormF32 {
        parameters: [Identifier; 12],
        done: Label,
        mean_loop: Label,
        mean_done: Label,
        var_loop: Label,
        var_done: Label,
        store_loop: Label,
        no_gamma: Label,
        no_beta: Label,
    },
}

impl fmt::Display for PtxInstruction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadParameterU64 { destination, parameter } => {
                write!(formatter, "ld.param.u64 {destination}, [{parameter}];")
            }
            Self::LoadParameterU32 { destination, parameter } => {
                write!(formatter, "ld.param.u32 {destination}, [{parameter}];")
            }
            Self::MoveCtaIdX { destination } => write!(formatter, "mov.u32 {destination}, %ctaid.x;"),
            Self::MoveNtidX { destination } => write!(formatter, "mov.u32 {destination}, %ntid.x;"),
            Self::MoveTidX { destination } => write!(formatter, "mov.u32 {destination}, %tid.x;"),
            Self::MultiplyAddLoS32 { destination, left, right, addend } => {
                write!(formatter, "mad.lo.s32 {destination}, {left}, {right}, {addend};")
            }
            Self::SetPredicateGeU32 { destination, left, right } => {
                write!(formatter, "setp.ge.u32 {destination}, {left}, {right};")
            }
            Self::BranchIf { predicate, target } => write!(formatter, "@{predicate} bra {target};"),
            Self::MultiplyWideU32 { destination, left, right } => {
                write!(formatter, "mul.wide.u32 {destination}, {left}, {right};")
            }
            Self::AddS64 { destination, left, right } => write!(formatter, "add.s64 {destination}, {left}, {right};"),
            Self::LoadGlobalF32 { destination, pointer } => write!(formatter, "ld.global.f32 {destination}, [{pointer}];"),
            Self::ArithmeticF32 { destination, operation, left, right } => match operation {
                ElementwiseOperation::Add => write!(formatter, "add.rn.f32 {destination}, {left}, {right};"),
                ElementwiseOperation::Mul => write!(formatter, "mul.rn.f32 {destination}, {left}, {right};"),
                ElementwiseOperation::Fma { addend } => {
                    let addend = match addend {
                        FmaAddend::Left => left,
                        FmaAddend::Right => right,
                    };
                    write!(formatter, "fma.rn.f32 {destination}, {left}, {right}, {addend};")
                }
            },
            Self::StoreGlobalF32 { pointer, value } => write!(formatter, "st.global.f32 [{pointer}], {value};"),
            Self::DefineLabel(label) => write!(formatter, "{label}:"),
            Self::Return => formatter.write_str("ret;"),
            Self::GemmF32 { parameters, done, loop_label } => write!(
                formatter,
                "ld.param.u64 %rd1, [{a}];\n\
                 ld.param.u64 %rd2, [{b}];\n\
                 ld.param.u64 %rd3, [{c}];\n\
                 ld.param.u32 %r1, [{m}];\n\
                 ld.param.u32 %r2, [{n}];\n\
                 ld.param.u32 %r3, [{k}];\n\
                 mov.u32 %r4, %ctaid.x;\n\
                 mov.u32 %r5, %ntid.x;\n\
                 mov.u32 %r6, %tid.x;\n\
                 mad.lo.u32 %r7, %r4, %r5, %r6;\n\
                 mul.lo.u32 %r8, %r1, %r2;\n\
                 setp.ge.u32 %p1, %r7, %r8;\n\
                 @%p1 bra {done};\n\
                 div.u32 %r9, %r7, %r2;\n\
                 rem.u32 %r10, %r7, %r2;\n\
                 mov.u32 %r11, 0;\n\
                 mov.f32 %f1, 0f00000000;\n\
                 {loop_label}:\n\
                 setp.ge.u32 %p2, %r11, %r3;\n\
                 @%p2 bra {done}_store;\n\
                 mad.lo.u32 %r12, %r9, %r3, %r11;\n\
                 mad.lo.u32 %r13, %r11, %r2, %r10;\n\
                 mul.wide.u32 %rd4, %r12, 4;\n\
                 mul.wide.u32 %rd5, %r13, 4;\n\
                 add.s64 %rd6, %rd1, %rd4;\n\
                 add.s64 %rd7, %rd2, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 ld.global.f32 %f3, [%rd7];\n\
                 fma.rn.f32 %f1, %f2, %f3, %f1;\n\
                 add.u32 %r11, %r11, 1;\n\
                 bra {loop_label};\n\
                 {done}_store:\n\
                 mul.wide.u32 %rd8, %r7, 4;\n\
                 add.s64 %rd9, %rd3, %rd8;\n\
                 st.global.f32 [%rd9], %f1;\n\
                 {done}:\n\
                 ret;",
                a = parameters[0],
                b = parameters[1],
                c = parameters[2],
                m = parameters[3],
                n = parameters[4],
                k = parameters[5],
            ),
            Self::Conv2dF32 {
                parameters,
                done,
                no_bias,
                input_channel_loop,
                kernel_h_loop,
                kernel_w_loop,
                next_kernel_w,
                kernel_w_done,
                kernel_h_done,
                input_channel_done,
            } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{weight}];\n\
                 ld.param.u64 %rd3, [{bias}];\n\
                 ld.param.u64 %rd4, [{output}];\n\
                 ld.param.u32 %r1, [{batch}];\n\
                 ld.param.u32 %r2, [{channels}];\n\
                 ld.param.u32 %r3, [{input_h}];\n\
                 ld.param.u32 %r4, [{input_w}];\n\
                 ld.param.u32 %r5, [{output_channels}];\n\
                 ld.param.u32 %r6, [{kernel_h}];\n\
                 ld.param.u32 %r7, [{kernel_w}];\n\
                 ld.param.u32 %r8, [{output_h}];\n\
                 ld.param.u32 %r9, [{output_w}];\n\
                 ld.param.u32 %r10, [{stride_h}];\n\
                 ld.param.u32 %r11, [{stride_w}];\n\
                 ld.param.u32 %r12, [{pad_h}];\n\
                 ld.param.u32 %r13, [{pad_w}];\n\
                 ld.param.u32 %r14, [{dilation_h}];\n\
                 ld.param.u32 %r15, [{dilation_w}];\n\
                 ld.param.u32 %r16, [{groups}];\n\
                 ld.param.u32 %r17, [{has_bias}];\n\
                 mov.u32 %r18, %ctaid.x;\n\
                 mov.u32 %r19, %ntid.x;\n\
                 mov.u32 %r20, %tid.x;\n\
                 mad.lo.u32 %r18, %r18, %r19, %r20;\n\
                 mul.lo.u32 %r19, %r1, %r5;\n\
                 mul.lo.u32 %r19, %r19, %r8;\n\
                 mul.lo.u32 %r19, %r19, %r9;\n\
                 setp.ge.u32 %p1, %r18, %r19;\n\
                 @%p1 bra {done};\n\
                 div.u32 %r20, %r18, %r9;\n\
                 rem.u32 %r21, %r18, %r9;\n\
                 div.u32 %r22, %r20, %r8;\n\
                 rem.u32 %r23, %r20, %r8;\n\
                 div.u32 %r24, %r22, %r5;\n\
                 rem.u32 %r25, %r22, %r5;\n\
                 div.u32 %r26, %r5, %r16;\n\
                 div.u32 %r27, %r25, %r26;\n\
                 div.u32 %r28, %r2, %r16;\n\
                 mul.lo.u32 %r29, %r27, %r28;\n\
                 mov.f32 %f1, 0f00000000;\n\
                 setp.eq.u32 %p1, %r17, 0;\n\
                 @%p1 bra {no_bias};\n\
                 mul.wide.u32 %rd5, %r25, 4;\n\
                 add.s64 %rd6, %rd3, %rd5;\n\
                 ld.global.f32 %f1, [%rd6];\n\
                 {no_bias}:\n\
                 mov.u32 %r30, 0;\n\
                 {input_channel_loop}:\n\
                 setp.ge.u32 %p1, %r30, %r28;\n\
                 @%p1 bra {input_channel_done};\n\
                 mov.u32 %r31, 0;\n\
                 {kernel_h_loop}:\n\
                 setp.ge.u32 %p1, %r31, %r6;\n\
                 @%p1 bra {kernel_h_done};\n\
                 mov.u32 %r32, 0;\n\
                 {kernel_w_loop}:\n\
                 setp.ge.u32 %p1, %r32, %r7;\n\
                 @%p1 bra {kernel_w_done};\n\
                 mul.lo.u32 %r33, %r31, %r14;\n\
                 mad.lo.u32 %r33, %r23, %r10, %r33;\n\
                 sub.s32 %r33, %r33, %r12;\n\
                 mul.lo.u32 %r34, %r32, %r15;\n\
                 mad.lo.u32 %r34, %r21, %r11, %r34;\n\
                 sub.s32 %r34, %r34, %r13;\n\
                 setp.lt.s32 %p2, %r33, 0;\n\
                 @%p2 bra {next_kernel_w};\n\
                 setp.lt.s32 %p2, %r34, 0;\n\
                 @%p2 bra {next_kernel_w};\n\
                 setp.ge.s32 %p2, %r33, %r3;\n\
                 @%p2 bra {next_kernel_w};\n\
                 setp.ge.s32 %p2, %r34, %r4;\n\
                 @%p2 bra {next_kernel_w};\n\
                 add.u32 %r35, %r29, %r30;\n\
                 mad.lo.u32 %r36, %r24, %r2, %r35;\n\
                 mad.lo.u32 %r36, %r36, %r3, %r33;\n\
                 mad.lo.u32 %r36, %r36, %r4, %r34;\n\
                 mad.lo.u32 %r37, %r25, %r28, %r30;\n\
                 mad.lo.u32 %r37, %r37, %r6, %r31;\n\
                 mad.lo.u32 %r37, %r37, %r7, %r32;\n\
                 mul.wide.u32 %rd5, %r36, 4;\n\
                 mul.wide.u32 %rd6, %r37, 4;\n\
                 add.s64 %rd7, %rd1, %rd5;\n\
                 add.s64 %rd8, %rd2, %rd6;\n\
                 ld.global.f32 %f2, [%rd7];\n\
                 ld.global.f32 %f3, [%rd8];\n\
                 fma.rn.f32 %f1, %f2, %f3, %f1;\n\
                 {next_kernel_w}:\n\
                 add.u32 %r32, %r32, 1;\n\
                 bra {kernel_w_loop};\n\
                 {kernel_w_done}:\n\
                 add.u32 %r31, %r31, 1;\n\
                 bra {kernel_h_loop};\n\
                 {kernel_h_done}:\n\
                 add.u32 %r30, %r30, 1;\n\
                 bra {input_channel_loop};\n\
                 {input_channel_done}:\n\
                 mul.wide.u32 %rd5, %r18, 4;\n\
                 add.s64 %rd6, %rd4, %rd5;\n\
                 st.global.f32 [%rd6], %f1;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                weight = parameters[1],
                bias = parameters[2],
                output = parameters[3],
                batch = parameters[4],
                channels = parameters[5],
                input_h = parameters[6],
                input_w = parameters[7],
                output_channels = parameters[8],
                kernel_h = parameters[9],
                kernel_w = parameters[10],
                output_h = parameters[11],
                output_w = parameters[12],
                stride_h = parameters[13],
                stride_w = parameters[14],
                pad_h = parameters[15],
                pad_w = parameters[16],
                dilation_h = parameters[17],
                dilation_w = parameters[18],
                groups = parameters[19],
                has_bias = parameters[20],
            ),
            Self::ScaledDotProductAttentionF32 {
                parameters,
                done,
                max_loop,
                max_inner_loop,
                max_inner_done,
                max_next,
                max_done,
                sum_loop,
                sum_inner_loop,
                sum_inner_done,
                sum_next,
                sum_done,
                value_loop,
                value_inner_loop,
                value_inner_done,
                value_next,
                value_done,
            } => write!(
                formatter,
                "ld.param.u64 %rd1, [{query}];\n\
                 ld.param.u64 %rd2, [{key}];\n\
                 ld.param.u64 %rd3, [{value}];\n\
                 ld.param.u64 %rd4, [{output}];\n\
                 ld.param.u32 %r1, [{batch}];\n\
                 ld.param.u32 %r2, [{heads}];\n\
                 ld.param.u32 %r3, [{query_tokens}];\n\
                 ld.param.u32 %r4, [{key_tokens}];\n\
                 ld.param.u32 %r5, [{depth}];\n\
                 mov.u32 %r6, %ctaid.x;\n\
                 mov.u32 %r7, %ntid.x;\n\
                 mov.u32 %r8, %tid.x;\n\
                 mad.lo.u32 %r9, %r6, %r7, %r8;\n\
                 mul.lo.u32 %r10, %r1, %r2;\n\
                 mul.lo.u32 %r10, %r10, %r3;\n\
                 mul.lo.u32 %r10, %r10, %r5;\n\
                 setp.ge.u32 %p1, %r9, %r10;\n\
                 @%p1 bra {done};\n\
                 div.u32 %r11, %r9, %r5;\n\
                 rem.u32 %r12, %r9, %r5;\n\
                 div.u32 %r13, %r11, %r3;\n\
                 rem.u32 %r14, %r11, %r3;\n\
                 div.u32 %r15, %r13, %r2;\n\
                 rem.u32 %r16, %r13, %r2;\n\
                 mad.lo.u32 %r17, %r15, %r2, %r16;\n\
                 mad.lo.u32 %r17, %r17, %r3, %r14;\n\
                 mul.lo.u32 %r17, %r17, %r5;\n\
                 cvt.rn.f32.u32 %f7, %r5;\n\
                 sqrt.rn.f32 %f7, %f7;\n\
                 mov.f32 %f1, 0fFF800000;\n\
                 mov.u32 %r20, 0;\n\
                 {max_loop}:\n\
                 setp.ge.u32 %p1, %r20, %r4;\n\
                 @%p1 bra {max_done};\n\
                 mad.lo.u32 %r18, %r15, %r2, %r16;\n\
                 mad.lo.u32 %r18, %r18, %r4, %r20;\n\
                 mul.lo.u32 %r18, %r18, %r5;\n\
                 mov.f32 %f2, 0f00000000;\n\
                 mov.u32 %r19, 0;\n\
                 {max_inner_loop}:\n\
                 setp.ge.u32 %p1, %r19, %r5;\n\
                 @%p1 bra {max_inner_done};\n\
                 add.u32 %r21, %r17, %r19;\n\
                 add.u32 %r22, %r18, %r19;\n\
                 mul.wide.u32 %rd5, %r21, 4;\n\
                 mul.wide.u32 %rd6, %r22, 4;\n\
                 add.s64 %rd7, %rd1, %rd5;\n\
                 add.s64 %rd8, %rd2, %rd6;\n\
                 ld.global.f32 %f5, [%rd7];\n\
                 ld.global.f32 %f6, [%rd8];\n\
                 fma.rn.f32 %f2, %f5, %f6, %f2;\n\
                 add.u32 %r19, %r19, 1;\n\
                 bra {max_inner_loop};\n\
                 {max_inner_done}:\n\
                 div.rn.f32 %f2, %f2, %f7;\n\
                 max.f32 %f1, %f1, %f2;\n\
                 {max_next}:\n\
                 add.u32 %r20, %r20, 1;\n\
                 bra {max_loop};\n\
                 {max_done}:\n\
                 mov.f32 %f3, 0f00000000;\n\
                 mov.u32 %r20, 0;\n\
                 {sum_loop}:\n\
                 setp.ge.u32 %p1, %r20, %r4;\n\
                 @%p1 bra {sum_done};\n\
                 mad.lo.u32 %r18, %r15, %r2, %r16;\n\
                 mad.lo.u32 %r18, %r18, %r4, %r20;\n\
                 mul.lo.u32 %r18, %r18, %r5;\n\
                 mov.f32 %f2, 0f00000000;\n\
                 mov.u32 %r19, 0;\n\
                 {sum_inner_loop}:\n\
                 setp.ge.u32 %p1, %r19, %r5;\n\
                 @%p1 bra {sum_inner_done};\n\
                 add.u32 %r21, %r17, %r19;\n\
                 add.u32 %r22, %r18, %r19;\n\
                 mul.wide.u32 %rd5, %r21, 4;\n\
                 mul.wide.u32 %rd6, %r22, 4;\n\
                 add.s64 %rd7, %rd1, %rd5;\n\
                 add.s64 %rd8, %rd2, %rd6;\n\
                 ld.global.f32 %f5, [%rd7];\n\
                 ld.global.f32 %f6, [%rd8];\n\
                 fma.rn.f32 %f2, %f5, %f6, %f2;\n\
                 add.u32 %r19, %r19, 1;\n\
                 bra {sum_inner_loop};\n\
                 {sum_inner_done}:\n\
                 div.rn.f32 %f2, %f2, %f7;\n\
                 sub.rn.f32 %f2, %f2, %f1;\n\
                 mul.rn.f32 %f2, %f2, 0f3FB8AA3B;\n\
                 ex2.approx.f32 %f2, %f2;\n\
                 add.rn.f32 %f3, %f3, %f2;\n\
                 {sum_next}:\n\
                 add.u32 %r20, %r20, 1;\n\
                 bra {sum_loop};\n\
                 {sum_done}:\n\
                 mov.f32 %f4, 0f00000000;\n\
                 mov.u32 %r20, 0;\n\
                 {value_loop}:\n\
                 setp.ge.u32 %p1, %r20, %r4;\n\
                 @%p1 bra {value_done};\n\
                 mad.lo.u32 %r18, %r15, %r2, %r16;\n\
                 mad.lo.u32 %r18, %r18, %r4, %r20;\n\
                 mul.lo.u32 %r18, %r18, %r5;\n\
                 mov.f32 %f2, 0f00000000;\n\
                 mov.u32 %r19, 0;\n\
                 {value_inner_loop}:\n\
                 setp.ge.u32 %p1, %r19, %r5;\n\
                 @%p1 bra {value_inner_done};\n\
                 add.u32 %r21, %r17, %r19;\n\
                 add.u32 %r22, %r18, %r19;\n\
                 mul.wide.u32 %rd5, %r21, 4;\n\
                 mul.wide.u32 %rd6, %r22, 4;\n\
                 add.s64 %rd7, %rd1, %rd5;\n\
                 add.s64 %rd8, %rd2, %rd6;\n\
                 ld.global.f32 %f5, [%rd7];\n\
                 ld.global.f32 %f6, [%rd8];\n\
                 fma.rn.f32 %f2, %f5, %f6, %f2;\n\
                 add.u32 %r19, %r19, 1;\n\
                 bra {value_inner_loop};\n\
                 {value_inner_done}:\n\
                 div.rn.f32 %f2, %f2, %f7;\n\
                 sub.rn.f32 %f2, %f2, %f1;\n\
                 mul.rn.f32 %f2, %f2, 0f3FB8AA3B;\n\
                 ex2.approx.f32 %f2, %f2;\n\
                 add.u32 %r22, %r18, %r12;\n\
                 mul.wide.u32 %rd5, %r22, 4;\n\
                 add.s64 %rd6, %rd3, %rd5;\n\
                 ld.global.f32 %f6, [%rd6];\n\
                 fma.rn.f32 %f4, %f2, %f6, %f4;\n\
                 {value_next}:\n\
                 add.u32 %r20, %r20, 1;\n\
                 bra {value_loop};\n\
                 {value_done}:\n\
                 div.rn.f32 %f4, %f4, %f3;\n\
                 mul.wide.u32 %rd5, %r9, 4;\n\
                 add.s64 %rd6, %rd4, %rd5;\n\
                 st.global.f32 [%rd6], %f4;\n\
                 {done}:\n\
                 ret;",
                query = parameters[0],
                key = parameters[1],
                value = parameters[2],
                output = parameters[3],
                batch = parameters[4],
                heads = parameters[5],
                query_tokens = parameters[6],
                key_tokens = parameters[7],
                depth = parameters[8],
            ),
            Self::BroadcastAddF32 { parameters, done, lhs_dim_done, rhs_dim_done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{lhs}];\n\
                 ld.param.u64 %rd2, [{rhs}];\n\
                 ld.param.u64 %rd3, [{output}];\n\
                 ld.param.u32 %r1, [{count}];\n\
                 mov.u32 %r2, %ctaid.x;\n\
                 mov.u32 %r3, %ntid.x;\n\
                 mov.u32 %r4, %tid.x;\n\
                 mad.lo.s32 %r5, %r2, %r3, %r4;\n\
                 setp.ge.u32 %p1, %r5, %r1;\n\
                 @%p1 bra {done};\n\
                 mov.u32 %r6, %r5;\n\
                 ld.param.u32 %r12, [{out3}];\n\
                 rem.u32 %r16, %r6, %r12;\n\
                 div.u32 %r6, %r6, %r12;\n\
                 ld.param.u32 %r12, [{out2}];\n\
                 rem.u32 %r15, %r6, %r12;\n\
                 div.u32 %r6, %r6, %r12;\n\
                 ld.param.u32 %r12, [{out1}];\n\
                 rem.u32 %r14, %r6, %r12;\n\
                 div.u32 %r6, %r6, %r12;\n\
                 ld.param.u32 %r12, [{out0}];\n\
                 rem.u32 %r13, %r6, %r12;\n\
                 mov.u32 %r8, 0;\n\
                 mov.u32 %r9, 0;\n\
                 ld.param.u32 %r10, [{lhs0}];\n\
                 mul.lo.u32 %r8, %r8, %r10;\n\
                 setp.eq.u32 %p1, %r10, 1;\n\
                 @%p1 bra {lhs0_done};\n\
                 add.u32 %r8, %r8, %r13;\n\
                 {lhs0_done}:\n\
                 ld.param.u32 %r11, [{rhs0}];\n\
                 mul.lo.u32 %r9, %r9, %r11;\n\
                 setp.eq.u32 %p1, %r11, 1;\n\
                 @%p1 bra {rhs0_done};\n\
                 add.u32 %r9, %r9, %r13;\n\
                 {rhs0_done}:\n\
                 ld.param.u32 %r10, [{lhs1}];\n\
                 mul.lo.u32 %r8, %r8, %r10;\n\
                 setp.eq.u32 %p1, %r10, 1;\n\
                 @%p1 bra {lhs1_done};\n\
                 add.u32 %r8, %r8, %r14;\n\
                 {lhs1_done}:\n\
                 ld.param.u32 %r11, [{rhs1}];\n\
                 mul.lo.u32 %r9, %r9, %r11;\n\
                 setp.eq.u32 %p1, %r11, 1;\n\
                 @%p1 bra {rhs1_done};\n\
                 add.u32 %r9, %r9, %r14;\n\
                 {rhs1_done}:\n\
                 ld.param.u32 %r10, [{lhs2}];\n\
                 mul.lo.u32 %r8, %r8, %r10;\n\
                 setp.eq.u32 %p1, %r10, 1;\n\
                 @%p1 bra {lhs2_done};\n\
                 add.u32 %r8, %r8, %r15;\n\
                 {lhs2_done}:\n\
                 ld.param.u32 %r11, [{rhs2}];\n\
                 mul.lo.u32 %r9, %r9, %r11;\n\
                 setp.eq.u32 %p1, %r11, 1;\n\
                 @%p1 bra {rhs2_done};\n\
                 add.u32 %r9, %r9, %r15;\n\
                 {rhs2_done}:\n\
                 ld.param.u32 %r10, [{lhs3}];\n\
                 mul.lo.u32 %r8, %r8, %r10;\n\
                 setp.eq.u32 %p1, %r10, 1;\n\
                 @%p1 bra {lhs3_done};\n\
                 add.u32 %r8, %r8, %r16;\n\
                 {lhs3_done}:\n\
                 ld.param.u32 %r11, [{rhs3}];\n\
                 mul.lo.u32 %r9, %r9, %r11;\n\
                 setp.eq.u32 %p1, %r11, 1;\n\
                 @%p1 bra {rhs3_done};\n\
                 add.u32 %r9, %r9, %r16;\n\
                 {rhs3_done}:\n\
                 mul.wide.u32 %rd4, %r8, 4;\n\
                 mul.wide.u32 %rd5, %r9, 4;\n\
                 mul.wide.u32 %rd6, %r5, 4;\n\
                 add.s64 %rd4, %rd1, %rd4;\n\
                 add.s64 %rd5, %rd2, %rd5;\n\
                 add.s64 %rd6, %rd3, %rd6;\n\
                 ld.global.f32 %f1, [%rd4];\n\
                 ld.global.f32 %f2, [%rd5];\n\
                 add.rn.f32 %f3, %f1, %f2;\n\
                 st.global.f32 [%rd6], %f3;\n\
                 {done}:\n\
                 ret;",
                lhs = parameters[0],
                rhs = parameters[1],
                output = parameters[2],
                count = parameters[3],
                lhs0 = parameters[4],
                lhs1 = parameters[5],
                lhs2 = parameters[6],
                lhs3 = parameters[7],
                rhs0 = parameters[8],
                rhs1 = parameters[9],
                rhs2 = parameters[10],
                rhs3 = parameters[11],
                out0 = parameters[12],
                out1 = parameters[13],
                out2 = parameters[14],
                out3 = parameters[15],
                lhs0_done = lhs_dim_done[0],
                lhs1_done = lhs_dim_done[1],
                lhs2_done = lhs_dim_done[2],
                lhs3_done = lhs_dim_done[3],
                rhs0_done = rhs_dim_done[0],
                rhs1_done = rhs_dim_done[1],
                rhs2_done = rhs_dim_done[2],
                rhs3_done = rhs_dim_done[3],
            ),
            Self::SiluF32 { parameters, done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{count}];\n\
                 mov.u32 %r2, %ctaid.x;\n\
                 mov.u32 %r3, %ntid.x;\n\
                 mov.u32 %r4, %tid.x;\n\
                 mad.lo.s32 %r5, %r2, %r3, %r4;\n\
                 setp.ge.u32 %p1, %r5, %r1;\n\
                 @%p1 bra {done};\n\
                 mul.wide.u32 %rd3, %r5, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 add.s64 %rd5, %rd2, %rd3;\n\
                 ld.global.f32 %f1, [%rd4];\n\
                 sub.rn.f32 %f2, 0f00000000, %f1;\n\
                 mul.rn.f32 %f2, %f2, 0f3FB8AA3B;\n\
                 ex2.approx.f32 %f2, %f2;\n\
                 add.rn.f32 %f2, %f2, 0f3F800000;\n\
                 div.rn.f32 %f3, %f1, %f2;\n\
                 st.global.f32 [%rd5], %f3;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                count = parameters[2],
            ),
            Self::GeluF32 { parameters, done, negative, signed_done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{count}];\n\
                 mov.u32 %r2, %ctaid.x;\n\
                 mov.u32 %r3, %ntid.x;\n\
                 mov.u32 %r4, %tid.x;\n\
                 mad.lo.s32 %r5, %r2, %r3, %r4;\n\
                 setp.ge.u32 %p1, %r5, %r1;\n\
                 @%p1 bra {done};\n\
                 mul.wide.u32 %rd3, %r5, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 add.s64 %rd5, %rd2, %rd3;\n\
                 ld.global.f32 %f1, [%rd4];\n\
                 mul.rn.f32 %f2, %f1, 0f3F3504F3;\n\
                 mov.f32 %f3, %f2;\n\
                 setp.lt.f32 %p2, %f2, 0f00000000;\n\
                 @%p2 bra {negative};\n\
                 mov.f32 %f4, 0f3F800000;\n\
                 bra {signed_done};\n\
                 {negative}:\n\
                 mov.f32 %f4, 0fBF800000;\n\
                 sub.rn.f32 %f3, 0f00000000, %f3;\n\
                 {signed_done}:\n\
                 mul.rn.f32 %f5, %f3, 0f3EA7BA05;\n\
                 add.rn.f32 %f5, %f5, 0f3F800000;\n\
                 div.rn.f32 %f5, 0f3F800000, %f5;\n\
                 mov.f32 %f6, 0f3F87DC22;\n\
                 fma.rn.f32 %f6, %f6, %f5, 0fBFBA00E3;\n\
                 fma.rn.f32 %f6, %f6, %f5, 0f3FB5F0E3;\n\
                 fma.rn.f32 %f6, %f6, %f5, 0fBE91A98E;\n\
                 fma.rn.f32 %f6, %f6, %f5, 0f3E827906;\n\
                 mul.rn.f32 %f6, %f6, %f5;\n\
                 mul.rn.f32 %f7, %f3, %f3;\n\
                 sub.rn.f32 %f7, 0f00000000, %f7;\n\
                 mul.rn.f32 %f7, %f7, 0f3FB8AA3B;\n\
                 ex2.approx.f32 %f7, %f7;\n\
                 mul.rn.f32 %f6, %f6, %f7;\n\
                 sub.rn.f32 %f6, 0f3F800000, %f6;\n\
                 mul.rn.f32 %f6, %f6, %f4;\n\
                 add.rn.f32 %f6, %f6, 0f3F800000;\n\
                 mul.rn.f32 %f6, %f6, %f1;\n\
                 mul.rn.f32 %f6, %f6, 0f3F000000;\n\
                 st.global.f32 [%rd5], %f6;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                count = parameters[2],
            ),
            Self::QuickGeluF32 { parameters, done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{count}];\n\
                 ld.param.f32 %f2, [{slope}];\n\
                 mov.u32 %r2, %ctaid.x;\n\
                 mov.u32 %r3, %ntid.x;\n\
                 mov.u32 %r4, %tid.x;\n\
                 mad.lo.s32 %r5, %r2, %r3, %r4;\n\
                 setp.ge.u32 %p1, %r5, %r1;\n\
                 @%p1 bra {done};\n\
                 mul.wide.u32 %rd3, %r5, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 add.s64 %rd5, %rd2, %rd3;\n\
                 ld.global.f32 %f1, [%rd4];\n\
                 mul.rn.f32 %f3, %f1, %f2;\n\
                 sub.rn.f32 %f3, 0f00000000, %f3;\n\
                 mul.rn.f32 %f3, %f3, 0f3FB8AA3B;\n\
                 ex2.approx.f32 %f3, %f3;\n\
                 add.rn.f32 %f3, %f3, 0f3F800000;\n\
                 div.rn.f32 %f3, %f1, %f3;\n\
                 st.global.f32 [%rd5], %f3;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                count = parameters[2],
                slope = parameters[3],
            ),
            Self::SoftmaxF32 { parameters, done, max_loop, max_done, sum_loop, sum_done, normalize_loop } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{rows}];\n\
                 ld.param.u32 %r2, [{cols}];\n\
                 mov.u32 %r3, %ctaid.x;\n\
                 mov.u32 %r4, %ntid.x;\n\
                 mov.u32 %r5, %tid.x;\n\
                 mad.lo.s32 %r6, %r3, %r4, %r5;\n\
                 setp.ge.u32 %p1, %r6, %r1;\n\
                 @%p1 bra {done};\n\
                 mul.lo.u32 %r7, %r6, %r2;\n\
                 mov.u32 %r8, 0;\n\
                 mov.f32 %f1, 0fFF7FFFFF;\n\
                 {max_loop}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {max_done};\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd3, %r9, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 ld.global.f32 %f2, [%rd4];\n\
                 max.f32 %f1, %f1, %f2;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {max_loop};\n\
                 {max_done}:\n\
                 mov.u32 %r8, 0;\n\
                 mov.f32 %f3, 0f00000000;\n\
                 {sum_loop}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {sum_done};\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd3, %r9, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 add.s64 %rd5, %rd2, %rd3;\n\
                 ld.global.f32 %f2, [%rd4];\n\
                 sub.rn.f32 %f4, %f2, %f1;\n\
                 mul.rn.f32 %f4, %f4, 0f3FB8AA3B;\n\
                 ex2.approx.f32 %f4, %f4;\n\
                 add.rn.f32 %f3, %f3, %f4;\n\
                 st.global.f32 [%rd5], %f4;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {sum_loop};\n\
                 {sum_done}:\n\
                 mov.u32 %r8, 0;\n\
                 {normalize_loop}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {done};\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd3, %r9, 4;\n\
                 add.s64 %rd5, %rd2, %rd3;\n\
                 ld.global.f32 %f4, [%rd5];\n\
                 div.rn.f32 %f4, %f4, %f3;\n\
                 st.global.f32 [%rd5], %f4;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {normalize_loop};\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                rows = parameters[2],
                cols = parameters[3],
            ),
            Self::ReductionSumF32 { parameters, done, loop_label } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{rows}];\n\
                 ld.param.u32 %r2, [{cols}];\n\
                 mov.u32 %r3, %ctaid.x;\n\
                 mov.u32 %r4, %ntid.x;\n\
                 mov.u32 %r5, %tid.x;\n\
                 mad.lo.s32 %r6, %r3, %r4, %r5;\n\
                 setp.ge.u32 %p1, %r6, %r1;\n\
                 @%p1 bra {done};\n\
                 mul.lo.u32 %r7, %r6, %r2;\n\
                 mov.u32 %r8, 0;\n\
                 mov.f32 %f1, 0f00000000;\n\
                 {loop_label}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {done}_store;\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd3, %r9, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 ld.global.f32 %f2, [%rd4];\n\
                 add.rn.f32 %f1, %f1, %f2;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {loop_label};\n\
                 {done}_store:\n\
                 mul.wide.u32 %rd5, %r6, 4;\n\
                 add.s64 %rd6, %rd2, %rd5;\n\
                 st.global.f32 [%rd6], %f1;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                rows = parameters[2],
                cols = parameters[3],
            ),
            Self::ConcatF32 { parameters, done, right } => write!(
                formatter,
                "ld.param.u64 %rd1, [{lhs}];\n\
                 ld.param.u64 %rd2, [{rhs}];\n\
                 ld.param.u64 %rd3, [{output}];\n\
                 ld.param.u32 %r1, [{lhs_elements}];\n\
                 ld.param.u32 %r2, [{total_elements}];\n\
                 mov.u32 %r3, %ctaid.x;\n\
                 mov.u32 %r4, %ntid.x;\n\
                 mov.u32 %r5, %tid.x;\n\
                 mad.lo.s32 %r6, %r3, %r4, %r5;\n\
                 setp.ge.u32 %p1, %r6, %r2;\n\
                 @%p1 bra {done};\n\
                 mul.wide.u32 %rd4, %r6, 4;\n\
                 add.s64 %rd5, %rd3, %rd4;\n\
                 setp.ge.u32 %p2, %r6, %r1;\n\
                 @%p2 bra {right};\n\
                 add.s64 %rd6, %rd1, %rd4;\n\
                 ld.global.f32 %f1, [%rd6];\n\
                 st.global.f32 [%rd5], %f1;\n\
                 bra {done};\n\
                 {right}:\n\
                 sub.u32 %r7, %r6, %r1;\n\
                 mul.wide.u32 %rd7, %r7, 4;\n\
                 add.s64 %rd8, %rd2, %rd7;\n\
                 ld.global.f32 %f1, [%rd8];\n\
                 st.global.f32 [%rd5], %f1;\n\
                 {done}:\n\
                 ret;",
                lhs = parameters[0],
                rhs = parameters[1],
                output = parameters[2],
                lhs_elements = parameters[3],
                total_elements = parameters[4],
            ),
            Self::TransposeF32 { parameters, done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{rows}];\n\
                 ld.param.u32 %r2, [{cols}];\n\
                 mov.u32 %r3, %ctaid.x;\n\
                 mov.u32 %r4, %ntid.x;\n\
                 mov.u32 %r5, %tid.x;\n\
                 mad.lo.s32 %r6, %r3, %r4, %r5;\n\
                 mul.lo.u32 %r7, %r1, %r2;\n\
                 setp.ge.u32 %p1, %r6, %r7;\n\
                 @%p1 bra {done};\n\
                 div.u32 %r8, %r6, %r1;\n\
                 rem.u32 %r9, %r6, %r1;\n\
                 mul.lo.u32 %r10, %r9, %r2;\n\
                 add.u32 %r11, %r10, %r8;\n\
                 mul.wide.u32 %rd3, %r11, 4;\n\
                 add.s64 %rd4, %rd1, %rd3;\n\
                 ld.global.f32 %f1, [%rd4];\n\
                 mul.wide.u32 %rd5, %r6, 4;\n\
                 add.s64 %rd6, %rd2, %rd5;\n\
                 st.global.f32 [%rd6], %f1;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                rows = parameters[2],
                cols = parameters[3],
            ),
            Self::SliceF32 { parameters, done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\nld.param.u64 %rd2, [{output}];\nld.param.u32 %r1, [{start}];\nld.param.u32 %r2, [{step}];\nld.param.u32 %r3, [{count}];\nmov.u32 %r4, %ctaid.x;\nmov.u32 %r5, %ntid.x;\nmov.u32 %r6, %tid.x;\nmad.lo.s32 %r7, %r4, %r5, %r6;\nsetp.ge.u32 %p1, %r7, %r3;\n@%p1 bra {done};\nmad.lo.u32 %r8, %r7, %r2, %r1;\nmul.wide.u32 %rd3, %r8, 4;\nadd.s64 %rd4, %rd1, %rd3;\nld.global.f32 %f1, [%rd4];\nmul.wide.u32 %rd5, %r7, 4;\nadd.s64 %rd6, %rd2, %rd5;\nst.global.f32 [%rd6], %f1;\n{done}:\nret;",
                input = parameters[0],
                output = parameters[1],
                start = parameters[2],
                step = parameters[3],
                count = parameters[4],
                done = done
            ),
            Self::ResizeNearest2dF32 { parameters, done } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{output}];\n\
                 ld.param.u32 %r1, [{n}];\n\
                 ld.param.u32 %r2, [{c}];\n\
                 ld.param.u32 %r3, [{in_h}];\n\
                 ld.param.u32 %r4, [{in_w}];\n\
                 ld.param.u32 %r5, [{out_h}];\n\
                 ld.param.u32 %r6, [{out_w}];\n\
                 mov.u32 %r7, %ctaid.x;\n\
                 mov.u32 %r8, %ntid.x;\n\
                 mov.u32 %r9, %tid.x;\n\
                 mad.lo.s32 %r10, %r7, %r8, %r9;\n\
                 mul.lo.u32 %r11, %r1, %r2;\n\
                 mul.lo.u32 %r11, %r11, %r5;\n\
                 mul.lo.u32 %r11, %r11, %r6;\n\
                 setp.ge.u32 %p1, %r10, %r11;\n\
                 @%p1 bra {done};\n\
                 rem.u32 %r12, %r10, %r6;\n\
                 div.u32 %r13, %r10, %r6;\n\
                 rem.u32 %r14, %r13, %r5;\n\
                 div.u32 %r15, %r13, %r5;\n\
                 rem.u32 %r16, %r15, %r2;\n\
                 div.u32 %r17, %r15, %r2;\n\
                 mul.lo.u32 %r18, %r14, %r3;\n\
                 div.u32 %r18, %r18, %r5;\n\
                 mul.lo.u32 %r19, %r12, %r4;\n\
                 div.u32 %r19, %r19, %r6;\n\
                 mul.lo.u32 %r20, %r17, %r2;\n\
                 add.u32 %r20, %r20, %r16;\n\
                 mul.lo.u32 %r20, %r20, %r3;\n\
                 add.u32 %r20, %r20, %r18;\n\
                 mul.lo.u32 %r20, %r20, %r4;\n\
                 add.u32 %r20, %r20, %r19;\n\
                 mul.wide.u32 %rd3, %r20, 4;\n\
                 mul.wide.u32 %rd4, %r10, 4;\n\
                 add.s64 %rd5, %rd1, %rd3;\n\
                 add.s64 %rd6, %rd2, %rd4;\n\
                 ld.global.f32 %f1, [%rd5];\n\
                 st.global.f32 [%rd6], %f1;\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                output = parameters[1],
                n = parameters[2],
                c = parameters[3],
                in_h = parameters[4],
                in_w = parameters[5],
                out_h = parameters[6],
                out_w = parameters[7],
            ),
            Self::LayerNormF32 {
                parameters,
                done,
                mean_loop,
                mean_done,
                var_loop,
                var_done,
                store_loop,
                no_gamma,
                no_beta,
            } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{gamma}];\n\
                 ld.param.u64 %rd3, [{beta}];\n\
                 ld.param.u64 %rd4, [{output}];\n\
                 ld.param.u32 %r1, [{rows}];\n\
                 ld.param.u32 %r2, [{cols}];\n\
                 ld.param.f32 %f6, [{epsilon}];\n\
                 ld.param.u32 %r10, [{has_gamma}];\n\
                 ld.param.u32 %r11, [{has_beta}];\n\
                 mov.u32 %r3, %ctaid.x;\n\
                 mov.u32 %r4, %ntid.x;\n\
                 mov.u32 %r5, %tid.x;\n\
                 mad.lo.u32 %r6, %r3, %r4, %r5;\n\
                 setp.ge.u32 %p1, %r6, %r1;\n\
                 @%p1 bra {done};\n\
                 mul.lo.u32 %r7, %r6, %r2;\n\
                 mov.f32 %f1, 0f00000000;\n\
                 mov.u32 %r8, 0;\n\
                 {mean_loop}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {mean_done};\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd5, %r9, 4;\n\
                 add.s64 %rd6, %rd1, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 add.rn.f32 %f1, %f1, %f2;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {mean_loop};\n\
                 {mean_done}:\n\
                 cvt.rn.f32.u32 %f7, %r2;\n\
                 div.rn.f32 %f1, %f1, %f7;\n\
                 mov.f32 %f3, 0f00000000;\n\
                 mov.u32 %r8, 0;\n\
                 {var_loop}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {var_done};\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd5, %r9, 4;\n\
                 add.s64 %rd6, %rd1, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 sub.rn.f32 %f2, %f2, %f1;\n\
                 mul.rn.f32 %f2, %f2, %f2;\n\
                 add.rn.f32 %f3, %f3, %f2;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {var_loop};\n\
                 {var_done}:\n\
                 div.rn.f32 %f3, %f3, %f7;\n\
                 add.rn.f32 %f3, %f3, %f6;\n\
                 rsqrt.approx.f32 %f4, %f3;\n\
                 mov.u32 %r8, 0;\n\
                 {store_loop}:\n\
                 setp.ge.u32 %p2, %r8, %r2;\n\
                 @%p2 bra {done};\n\
                 add.u32 %r9, %r7, %r8;\n\
                 mul.wide.u32 %rd5, %r9, 4;\n\
                 add.s64 %rd6, %rd1, %rd5;\n\
                 add.s64 %rd7, %rd4, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 sub.rn.f32 %f5, %f2, %f1;\n\
                 mul.rn.f32 %f5, %f5, %f4;\n\
                 setp.eq.u32 %p3, %r10, 0;\n\
                 @%p3 bra {no_gamma};\n\
                 mul.wide.u32 %rd8, %r8, 4;\n\
                 add.s64 %rd9, %rd2, %rd8;\n\
                 ld.global.f32 %f2, [%rd9];\n\
                 mul.rn.f32 %f5, %f5, %f2;\n\
                 {no_gamma}:\n\
                 setp.eq.u32 %p3, %r11, 0;\n\
                 @%p3 bra {no_beta};\n\
                 mul.wide.u32 %rd8, %r8, 4;\n\
                 add.s64 %rd9, %rd3, %rd8;\n\
                 ld.global.f32 %f2, [%rd9];\n\
                 add.rn.f32 %f5, %f5, %f2;\n\
                 {no_beta}:\n\
                 st.global.f32 [%rd7], %f5;\n\
                 add.u32 %r8, %r8, 1;\n\
                 bra {store_loop};\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                gamma = parameters[1],
                beta = parameters[2],
                output = parameters[3],
                rows = parameters[4],
                cols = parameters[5],
                epsilon = parameters[6],
                has_gamma = parameters[7],
                has_beta = parameters[8],
            ),
            Self::GroupNormF32 {
                parameters,
                done,
                mean_loop,
                mean_done,
                var_loop,
                var_done,
                store_loop,
                no_gamma,
                no_beta,
            } => write!(
                formatter,
                "ld.param.u64 %rd1, [{input}];\n\
                 ld.param.u64 %rd2, [{gamma}];\n\
                 ld.param.u64 %rd3, [{beta}];\n\
                 ld.param.u64 %rd4, [{output}];\n\
                 ld.param.u32 %r1, [{n}];\n\
                 ld.param.u32 %r2, [{channels}];\n\
                 ld.param.u32 %r3, [{height}];\n\
                 ld.param.u32 %r4, [{width}];\n\
                 ld.param.u32 %r5, [{groups}];\n\
                 ld.param.f32 %f6, [{epsilon}];\n\
                 ld.param.u32 %r14, [{has_gamma}];\n\
                 ld.param.u32 %r15, [{has_beta}];\n\
                 mov.u32 %r6, %ctaid.x;\n\
                 mov.u32 %r7, %ntid.x;\n\
                 mov.u32 %r8, %tid.x;\n\
                 mad.lo.u32 %r9, %r6, %r7, %r8;\n\
                 mul.lo.u32 %r10, %r1, %r5;\n\
                 setp.ge.u32 %p1, %r9, %r10;\n\
                 @%p1 bra {done};\n\
                 div.u32 %r11, %r9, %r5;\n\
                 rem.u32 %r12, %r9, %r5;\n\
                 div.u32 %r13, %r2, %r5;\n\
                 mul.lo.u32 %r16, %r3, %r4;\n\
                 mul.lo.u32 %r17, %r13, %r16;\n\
                 mul.lo.u32 %r18, %r9, %r17;\n\
                 mov.f32 %f1, 0f00000000;\n\
                 mov.u32 %r19, 0;\n\
                 {mean_loop}:\n\
                 setp.ge.u32 %p2, %r19, %r17;\n\
                 @%p2 bra {mean_done};\n\
                 add.u32 %r20, %r18, %r19;\n\
                 mul.wide.u32 %rd5, %r20, 4;\n\
                 add.s64 %rd6, %rd1, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 add.rn.f32 %f1, %f1, %f2;\n\
                 add.u32 %r19, %r19, 1;\n\
                 bra {mean_loop};\n\
                 {mean_done}:\n\
                 cvt.rn.f32.u32 %f7, %r17;\n\
                 div.rn.f32 %f1, %f1, %f7;\n\
                 mov.f32 %f3, 0f00000000;\n\
                 mov.u32 %r19, 0;\n\
                 {var_loop}:\n\
                 setp.ge.u32 %p2, %r19, %r17;\n\
                 @%p2 bra {var_done};\n\
                 add.u32 %r20, %r18, %r19;\n\
                 mul.wide.u32 %rd5, %r20, 4;\n\
                 add.s64 %rd6, %rd1, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 sub.rn.f32 %f2, %f2, %f1;\n\
                 mul.rn.f32 %f2, %f2, %f2;\n\
                 add.rn.f32 %f3, %f3, %f2;\n\
                 add.u32 %r19, %r19, 1;\n\
                 bra {var_loop};\n\
                 {var_done}:\n\
                 div.rn.f32 %f3, %f3, %f7;\n\
                 add.rn.f32 %f3, %f3, %f6;\n\
                 rsqrt.approx.f32 %f4, %f3;\n\
                 mov.u32 %r19, 0;\n\
                 {store_loop}:\n\
                 setp.ge.u32 %p2, %r19, %r17;\n\
                 @%p2 bra {done};\n\
                 add.u32 %r20, %r18, %r19;\n\
                 mul.wide.u32 %rd5, %r20, 4;\n\
                 add.s64 %rd6, %rd1, %rd5;\n\
                 add.s64 %rd7, %rd4, %rd5;\n\
                 ld.global.f32 %f2, [%rd6];\n\
                 sub.rn.f32 %f5, %f2, %f1;\n\
                 mul.rn.f32 %f5, %f5, %f4;\n\
                 div.u32 %r21, %r19, %r16;\n\
                 mad.lo.u32 %r21, %r12, %r13, %r21;\n\
                 setp.eq.u32 %p3, %r14, 0;\n\
                 @%p3 bra {no_gamma};\n\
                 mul.wide.u32 %rd8, %r21, 4;\n\
                 add.s64 %rd9, %rd2, %rd8;\n\
                 ld.global.f32 %f2, [%rd9];\n\
                 mul.rn.f32 %f5, %f5, %f2;\n\
                 {no_gamma}:\n\
                 setp.eq.u32 %p3, %r15, 0;\n\
                 @%p3 bra {no_beta};\n\
                 mul.wide.u32 %rd8, %r21, 4;\n\
                 add.s64 %rd9, %rd3, %rd8;\n\
                 ld.global.f32 %f2, [%rd9];\n\
                 add.rn.f32 %f5, %f5, %f2;\n\
                 {no_beta}:\n\
                 st.global.f32 [%rd7], %f5;\n\
                 add.u32 %r19, %r19, 1;\n\
                 bra {store_loop};\n\
                 {done}:\n\
                 ret;",
                input = parameters[0],
                gamma = parameters[1],
                beta = parameters[2],
                output = parameters[3],
                n = parameters[4],
                channels = parameters[5],
                height = parameters[6],
                width = parameters[7],
                groups = parameters[8],
                epsilon = parameters[9],
                has_gamma = parameters[10],
                has_beta = parameters[11],
            ),
        }
    }
}

struct Entry {
    name: Identifier,
    parameters: Vec<Parameter>,
    registers: Vec<RegisterDeclaration>,
    instructions: Vec<PtxInstruction>,
}

impl Entry {
    fn elementwise_f32(name: Identifier, operation: ElementwiseOperation) -> Self {
        let parameter_names = [
            name.parameter(ParameterIndex(0)),
            name.parameter(ParameterIndex(1)),
            name.parameter(ParameterIndex(2)),
            name.parameter(ParameterIndex(3)),
        ];
        let parameters = vec![
            Parameter { name: parameter_names[0].clone(), kind: ParameterKind::GlobalF32Pointer },
            Parameter { name: parameter_names[1].clone(), kind: ParameterKind::GlobalF32Pointer },
            Parameter { name: parameter_names[2].clone(), kind: ParameterKind::GlobalF32Pointer },
            Parameter { name: parameter_names[3].clone(), kind: ParameterKind::U32 },
        ];
        let predicate = |index| Register::new(RegisterClass::Predicate, index);
        let b32 = |index| Register::new(RegisterClass::B32, index);
        let b64 = |index| Register::new(RegisterClass::B64, index);
        let f32 = |index| Register::new(RegisterClass::F32, index);
        let done = Label(name.suffix("_done"));
        Self {
            name,
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(8).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
            ],
            instructions: vec![
                PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
                PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
                PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
                PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[3].clone() },
                PtxInstruction::MoveCtaIdX { destination: b32(2) },
                PtxInstruction::MoveNtidX { destination: b32(3) },
                PtxInstruction::MoveTidX { destination: b32(4) },
                PtxInstruction::MultiplyAddLoS32 { destination: b32(5), left: b32(2), right: b32(3), addend: b32(4) },
                PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(5), right: b32(1) },
                PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
                PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(5), right: 4 },
                PtxInstruction::AddS64 { destination: b64(5), left: b64(1), right: b64(4) },
                PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(4) },
                PtxInstruction::AddS64 { destination: b64(7), left: b64(3), right: b64(4) },
                PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(5) },
                PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
                PtxInstruction::ArithmeticF32 { destination: f32(3), operation, left: f32(1), right: f32(2) },
                PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(3) },
                PtxInstruction::DefineLabel(done),
                PtxInstruction::Return,
            ],
        }
    }

    fn gemm_f32(name: Identifier) -> Self {
        let parameter_names = [
            name.parameter(ParameterIndex(0)),
            name.parameter(ParameterIndex(1)),
            name.parameter(ParameterIndex(2)),
            name.parameter(ParameterIndex(3)),
            name.parameter(ParameterIndex(4)),
            name.parameter(ParameterIndex(5)),
        ];
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 3 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        let done = Label(name.suffix("_done"));
        let loop_label = Label(name.suffix("_k_loop"));
        Self {
            name,
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(14).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(10).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
            ],
            instructions: vec![PtxInstruction::GemmF32 { parameters: parameter_names, done, loop_label }],
        }
    }

    fn conv2d_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 4 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(38).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
            ],
            instructions: vec![PtxInstruction::Conv2dF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                no_bias: Label(name.suffix("_no_bias")),
                input_channel_loop: Label(name.suffix("_input_channel_loop")),
                kernel_h_loop: Label(name.suffix("_kernel_h_loop")),
                kernel_w_loop: Label(name.suffix("_kernel_w_loop")),
                next_kernel_w: Label(name.suffix("_next_kernel_w")),
                kernel_w_done: Label(name.suffix("_kernel_w_done")),
                kernel_h_done: Label(name.suffix("_kernel_h_done")),
                input_channel_done: Label(name.suffix("_input_channel_done")),
            }],
        }
    }

    fn scaled_dot_product_attention_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 4 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(23).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(8).unwrap() },
            ],
            instructions: vec![PtxInstruction::ScaledDotProductAttentionF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                max_loop: Label(name.suffix("_max_loop")),
                max_inner_loop: Label(name.suffix("_max_inner_loop")),
                max_inner_done: Label(name.suffix("_max_inner_done")),
                max_next: Label(name.suffix("_max_next")),
                max_done: Label(name.suffix("_max_done")),
                sum_loop: Label(name.suffix("_sum_loop")),
                sum_inner_loop: Label(name.suffix("_sum_inner_loop")),
                sum_inner_done: Label(name.suffix("_sum_inner_done")),
                sum_next: Label(name.suffix("_sum_next")),
                sum_done: Label(name.suffix("_sum_done")),
                value_loop: Label(name.suffix("_value_loop")),
                value_inner_loop: Label(name.suffix("_value_inner_loop")),
                value_inner_done: Label(name.suffix("_value_inner_done")),
                value_next: Label(name.suffix("_value_next")),
                value_done: Label(name.suffix("_value_done")),
            }],
        }
    }

    fn broadcast_add_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 3 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(17).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
            ],
            instructions: vec![PtxInstruction::BroadcastAddF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                lhs_dim_done: std::array::from_fn(|index| Label(name.suffix(&format!("_lhs_{index}_done")))),
                rhs_dim_done: std::array::from_fn(|index| Label(name.suffix(&format!("_rhs_{index}_done")))),
            }],
        }
    }

    fn silu_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
            ],
            instructions: vec![PtxInstruction::SiluF32 { parameters: parameter_names, done: Label(name.suffix("_done")) }],
        }
    }

    fn gelu_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(10).unwrap() },
            ],
            instructions: vec![PtxInstruction::GeluF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                negative: Label(name.suffix("_negative")),
                signed_done: Label(name.suffix("_signed_done")),
            }],
        }
    }

    fn quick_gelu_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 {
                    ParameterKind::GlobalF32Pointer
                }
                else if index == 2 {
                    ParameterKind::U32
                }
                else {
                    ParameterKind::F32
                },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
            ],
            instructions: vec![PtxInstruction::QuickGeluF32 { parameters: parameter_names, done: Label(name.suffix("_done")) }],
        }
    }

    fn softmax_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(10).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(5).unwrap() },
            ],
            instructions: vec![PtxInstruction::SoftmaxF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                max_loop: Label(name.suffix("_max_loop")),
                max_done: Label(name.suffix("_max_done")),
                sum_loop: Label(name.suffix("_sum_loop")),
                sum_done: Label(name.suffix("_sum_done")),
                normalize_loop: Label(name.suffix("_normalize_loop")),
            }],
        }
    }

    fn reduction_sum_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(10).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(3).unwrap() },
            ],
            instructions: vec![PtxInstruction::ReductionSumF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                loop_label: Label(name.suffix("_loop")),
            }],
        }
    }

    fn transpose_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(12).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
            ],
            instructions: vec![PtxInstruction::TransposeF32 { parameters: parameter_names, done: Label(name.suffix("_done")) }],
        }
    }

    fn slice_f32(name: Identifier) -> Self {
        let parameters = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        Self {
            name: name.clone(),
            parameters: parameters
                .iter()
                .enumerate()
                .map(|(index, parameter)| Parameter {
                    name: parameter.clone(),
                    kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
                })
                .collect(),
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(10).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
            ],
            instructions: vec![PtxInstruction::SliceF32 { parameters, done: Label(name.suffix("_done")) }],
        }
    }

    fn resize_nearest2d_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(21).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
            ],
            instructions: vec![PtxInstruction::ResizeNearest2dF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
            }],
        }
    }

    fn concat_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 3 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(8).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
            ],
            instructions: vec![PtxInstruction::ConcatF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                right: Label(name.suffix("_right")),
            }],
        }
    }

    fn layer_norm_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 4 {
                    ParameterKind::GlobalF32Pointer
                }
                else if index == 6 {
                    ParameterKind::F32
                }
                else {
                    ParameterKind::U32
                },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(4).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(12).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(10).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(8).unwrap() },
            ],
            instructions: vec![PtxInstruction::LayerNormF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                mean_loop: Label(name.suffix("_mean_loop")),
                mean_done: Label(name.suffix("_mean_done")),
                var_loop: Label(name.suffix("_var_loop")),
                var_done: Label(name.suffix("_var_done")),
                store_loop: Label(name.suffix("_store_loop")),
                no_gamma: Label(name.suffix("_no_gamma")),
                no_beta: Label(name.suffix("_no_beta")),
            }],
        }
    }

    fn group_norm_f32(name: Identifier) -> Self {
        let parameter_names = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 4 {
                    ParameterKind::GlobalF32Pointer
                }
                else if index == 9 {
                    ParameterKind::F32
                }
                else {
                    ParameterKind::U32
                },
            })
            .collect();
        Self {
            name: name.clone(),
            parameters,
            registers: vec![
                RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(4).unwrap() },
                RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(22).unwrap() },
                RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(10).unwrap() },
                RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(8).unwrap() },
            ],
            instructions: vec![PtxInstruction::GroupNormF32 {
                parameters: parameter_names,
                done: Label(name.suffix("_done")),
                mean_loop: Label(name.suffix("_mean_loop")),
                mean_done: Label(name.suffix("_mean_done")),
                var_loop: Label(name.suffix("_var_loop")),
                var_done: Label(name.suffix("_var_done")),
                store_loop: Label(name.suffix("_store_loop")),
                no_gamma: Label(name.suffix("_no_gamma")),
                no_beta: Label(name.suffix("_no_beta")),
            }],
        }
    }
}

impl fmt::Display for Entry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, ".visible .entry {}(", self.name)?;
        for (index, parameter) in self.parameters.iter().enumerate() {
            write!(formatter, "    {parameter}")?;
            if index + 1 != self.parameters.len() {
                writeln!(formatter, ",")?;
            }
            else {
                writeln!(formatter)?;
            }
        }
        writeln!(formatter, ")")?;
        writeln!(formatter, "{{")?;
        for register in &self.registers {
            writeln!(formatter, "    {register}")?;
        }
        writeln!(formatter)?;
        for instruction in &self.instructions {
            writeln!(formatter, "    {instruction}")?;
        }
        writeln!(formatter, "}}")
    }
}
