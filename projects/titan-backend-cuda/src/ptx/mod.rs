//! Private typed PTX AST and lowering for the CUDA backend.

mod abi;
mod ast;
mod entries;
mod validate;

use titan_kernel::{AddressSpace, Instruction as IrInstruction, IrType, KernelAbi, KernelError, KernelModule};
use titan_types::{BackendId, DType, DeviceFingerprint};

use ast::{
    AddressSize, ElementwiseOperation, Entry, FmaAddend, Identifier, PtxModule, PtxVersion, Target,
};
use abi::*;
use validate::validate_entry;

pub(crate) const MINIMUM_SM: u16 = 70;

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
        } else {
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
    } else if ir.kernel_id.0 == "conv2d.f32" {
        validate_conv2d_abi(ir, abi)?;
        validate_conv2d_ir(ir)?;
        Entry::conv2d_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "scaled_dot_product_attention.f32" {
        validate_attention_abi(ir, abi)?;
        validate_attention_ir(ir)?;
        Entry::scaled_dot_product_attention_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "broadcast.add.f32" {
        validate_broadcast_add_abi(ir, abi)?;
        validate_broadcast_add_ir(ir)?;
        Entry::broadcast_add_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "silu.f32" {
        validate_silu_abi(ir, abi)?;
        validate_silu_ir(ir)?;
        Entry::silu_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "gelu.f32" {
        validate_gelu_abi(ir, abi)?;
        validate_gelu_ir(ir)?;
        Entry::gelu_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "quick_gelu.f32" {
        validate_quick_gelu_abi(ir, abi)?;
        validate_quick_gelu_ir(ir)?;
        Entry::quick_gelu_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "softmax.f32" {
        validate_softmax_abi(ir, abi)?;
        validate_softmax_ir(ir)?;
        Entry::softmax_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "reduction.sum.f32" {
        validate_reduction_sum_abi(ir, abi)?;
        validate_reduction_sum_ir(ir)?;
        Entry::reduction_sum_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "concat.f32" {
        validate_concat_abi(ir, abi)?;
        validate_concat_ir(ir)?;
        Entry::concat_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "transpose.f32" {
        validate_transpose_abi(ir, abi)?;
        validate_transpose_ir(ir)?;
        Entry::transpose_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "slice.f32" {
        validate_slice_abi(ir, abi)?;
        validate_slice_ir(ir)?;
        Entry::slice_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "resize.nearest2d.f32" {
        validate_resize_nearest2d_abi(ir, abi)?;
        validate_resize_nearest2d_ir(ir)?;
        Entry::resize_nearest2d_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "layer_norm.f32" {
        validate_layer_norm_abi(ir, abi)?;
        validate_layer_norm_ir(ir)?;
        Entry::layer_norm_f32(entry_name.clone())
    } else if ir.kernel_id.0 == "group_norm.f32" {
        validate_group_norm_abi(ir, abi)?;
        validate_group_norm_ir(ir)?;
        Entry::group_norm_f32(entry_name.clone())
    } else {
        validate_abi(ir, abi)?;
        reject_non_global_pointers(ir)?;
        let operation = validate_elementwise_ir(ir)?;
        Entry::elementwise_f32(entry_name.clone(), operation)
    };
    validate_entry(&entry)?;
    let module = PtxModule { version: PtxVersion::V80, target, address_size: AddressSize::Bits64, entry };
    let source = module.to_string();
    let artifact = PtxArtifact::from_driver_bytes(source.as_bytes())
        .map_err(|detail| KernelError::Compile(format!("typed PTX emitter produced invalid artifact: {detail}")))?;
    Ok(LoweredPtx { artifact, entry: entry_name })
}

