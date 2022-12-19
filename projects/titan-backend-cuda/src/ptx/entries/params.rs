//! Shared parameter naming, ABI parameter lists, register banks, and param loads.

use std::num::NonZeroU8;

use super::{
    super::ast::{Identifier, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass, RegisterDeclaration},
    regs::{b32, b64, f32},
};

/// `kernel_param_0 .. kernel_param_{N-1}`.
pub(super) fn named_params<const N: usize>(kernel: &Identifier) -> [Identifier; N] {
    std::array::from_fn(|index| kernel.parameter(ParameterIndex(index as u8)))
}

/// Pair each name with an explicit parameter kind.
pub(super) fn declare_params(names: &[Identifier], kinds: &[ParameterKind]) -> Vec<Parameter> {
    debug_assert_eq!(names.len(), kinds.len());
    names.iter().zip(kinds.iter().copied()).map(|(name, kind)| Parameter { name: name.clone(), kind }).collect()
}

/// First `buffers` parameters are global f32 pointers; the rest are u32 scalars.
pub(super) fn buffer_u32_params(names: &[Identifier], buffers: usize) -> Vec<Parameter> {
    names
        .iter()
        .enumerate()
        .map(|(index, name)| Parameter {
            name: name.clone(),
            kind: if index < buffers { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
        })
        .collect()
}

/// Standard `.reg` bank: predicate / b32 / b64 / f32 counts.
pub(super) fn regs(pred: u8, r: u8, rd: u8, f: u8) -> Vec<RegisterDeclaration> {
    vec![
        RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(pred).expect("pred count") },
        RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(r).expect("b32 count") },
        RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(rd).expect("b64 count") },
        RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(f).expect("f32 count") },
    ]
}

/// Destination register index for a typed parameter load.
#[derive(Clone, Copy)]
pub(super) enum ParamLoad {
    /// `ld.param.u64 %rd{n}`
    Ptr(u8),
    /// `ld.param.u32 %r{n}`
    U32(u8),
    /// `ld.param.f32 %f{n}`
    F32(u8),
}

/// Emit one `ld.param.*` per name/load pair (same order as ABI parameters).
pub(super) fn load_params(names: &[Identifier], loads: &[ParamLoad]) -> Vec<PtxInstruction> {
    debug_assert_eq!(names.len(), loads.len());
    names
        .iter()
        .zip(loads.iter().copied())
        .map(|(parameter, load)| match load {
            ParamLoad::Ptr(index) => PtxInstruction::LoadParameterU64 { destination: b64(index), parameter: parameter.clone() },
            ParamLoad::U32(index) => PtxInstruction::LoadParameterU32 { destination: b32(index), parameter: parameter.clone() },
            ParamLoad::F32(index) => PtxInstruction::LoadParameterF32 { destination: f32(index), parameter: parameter.clone() },
        })
        .collect()
}
