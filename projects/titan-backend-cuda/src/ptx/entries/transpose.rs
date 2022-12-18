use std::num::NonZeroU8;

use super::{
    super::ast::{
        Entry, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass, RegisterDeclaration,
        U32Value,
    },
    prologue::{bounds_guard, linear_tid},
    regs::{b32, b64, f32},
};

pub(super) fn transpose_f32(name: Identifier) -> Entry {
    let parameter_names: [Identifier; 4] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
    let parameters = parameter_names
        .iter()
        .enumerate()
        .map(|(index, parameter)| Parameter {
            name: parameter.clone(),
            kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
        })
        .collect();

    let done = Label(name.suffix("_done"));
    let mut instructions = vec![
        PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
        PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
        PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
        PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[3].clone() },
    ];
    instructions.extend(linear_tid(3, 4, 5, 6, true));
    instructions.push(PtxInstruction::MulLoU32 { destination: b32(7), left: b32(1), right: U32Value::Reg(b32(2)) });
    instructions.extend(bounds_guard(6, U32Value::Reg(b32(7)), 1, &done));
    instructions.extend([
        PtxInstruction::DivU32 { destination: b32(8), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::RemU32 { destination: b32(9), left: b32(6), right: U32Value::Reg(b32(1)) },
        PtxInstruction::MulLoU32 { destination: b32(10), left: b32(9), right: U32Value::Reg(b32(2)) },
        PtxInstruction::AddU32 { destination: b32(11), left: b32(10), right: U32Value::Reg(b32(8)) },
        PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(11), right: 4 },
        PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
        PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
        PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(6), right: 4 },
        PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(5) },
        PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
        PtxInstruction::DefineLabel(done),
        PtxInstruction::Return,
    ]);
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(12).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
        ],
        instructions,
    }
}
