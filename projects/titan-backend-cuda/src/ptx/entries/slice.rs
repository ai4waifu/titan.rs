use std::num::NonZeroU8;

use super::{
    super::ast::{
        Entry, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass, RegisterDeclaration,
        U32Value,
    },
    regs::{b32, b64, f32, predicate},
};

pub(super) fn slice_f32(name: Identifier) -> Entry {
    let parameter_names: [Identifier; 5] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
    let parameters = parameter_names
        .iter()
        .enumerate()
        .map(|(index, parameter)| Parameter {
            name: parameter.clone(),
            kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
        })
        .collect();

    let done = Label(name.suffix("_done"));
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(10).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
        ],
        instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[3].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(3), parameter: parameter_names[4].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(4) },
            PtxInstruction::MoveNtidX { destination: b32(5) },
            PtxInstruction::MoveTidX { destination: b32(6) },
            PtxInstruction::MultiplyAddLoS32 { destination: b32(7), left: b32(4), right: b32(5), addend: b32(6) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(7), right: U32Value::Reg(b32(3)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::MadLoU32 { destination: b32(8), left: b32(7), right: b32(2), addend: b32(1) },
            PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(8), right: 4 },
            PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(7), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(5) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
    }
}
