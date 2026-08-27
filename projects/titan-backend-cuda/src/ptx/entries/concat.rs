use std::num::NonZeroU8;

use super::{
    super::ast::{
        Entry, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass, RegisterDeclaration,
        U32Value,
    },
    prologue::linear_index_guard,
    regs::{b32, b64, f32, predicate},
};

pub(super) fn concat_f32(name: Identifier) -> Entry {
    let parameter_names: [Identifier; 5] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
    let parameters = parameter_names
        .iter()
        .enumerate()
        .map(|(index, parameter)| Parameter {
            name: parameter.clone(),
            kind: if index < 3 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
        })
        .collect();

    let done = Label(name.suffix("_done"));
    let right = Label(name.suffix("_right"));
    let mut instructions = vec![
        PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
        PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
        PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
        PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[3].clone() },
        PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[4].clone() },
    ];
    instructions.extend(linear_index_guard(3, 4, 5, 6, U32Value::Reg(b32(2)), 1, &done, true));
    instructions.extend([
        PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(6), right: 4 },
        PtxInstruction::AddS64 { destination: b64(5), left: b64(3), right: b64(4) },
        PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::BranchIf { predicate: predicate(2), target: right.clone() },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(4) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(6) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(1) },
        PtxInstruction::Branch { target: done.clone() },
        PtxInstruction::DefineLabel(right),
        PtxInstruction::SubU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(7), left: b32(7), right: 4 },
        PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(7) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(8) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(1) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(8).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
        ],
        instructions,
    }
}
