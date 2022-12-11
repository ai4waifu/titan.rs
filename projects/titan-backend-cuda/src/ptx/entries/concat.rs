use std::num::NonZeroU8;

use super::super::ast::{
    Entry, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
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

    let predicate = |index| Register::new(RegisterClass::Predicate, index);
    let b32 = |index| Register::new(RegisterClass::B32, index);
    let b64 = |index| Register::new(RegisterClass::B64, index);
    let f32 = |index| Register::new(RegisterClass::F32, index);
        let done = Label(name.suffix("_done"));
        let right = Label(name.suffix("_right"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(8).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[3].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[4].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(3) },
            PtxInstruction::MoveNtidX { destination: b32(4) },
            PtxInstruction::MoveTidX { destination: b32(5) },
            PtxInstruction::MultiplyAddLoS32 { destination: b32(6), left: b32(3), right: b32(4), addend: b32(5) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(6), right: U32Value::Reg(b32(2)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(6), right: 4 },
            PtxInstruction::AddS64 { destination: b64(5), left: b64(3), right: b64(4) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(6), right: U32Value::Reg(b32(1)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: right.clone() },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(4) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(6) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(1) },
            PtxInstruction::Branch { target: done.clone() },
            PtxInstruction::DefineLabel(right.clone()),
            PtxInstruction::SubU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(1)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(7), left: b32(7), right: 4 },
            PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(7) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(8) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(1) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
