use std::num::NonZeroU8;

use super::super::ast::{
    Entry, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn resize_nearest2d_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 8] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 2 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();

    let predicate = |index| Register::new(RegisterClass::Predicate, index);
    let b32 = |index| Register::new(RegisterClass::B32, index);
    let b64 = |index| Register::new(RegisterClass::B64, index);
    let f32 = |index| Register::new(RegisterClass::F32, index);
        let done = Label(name.suffix("_done"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(21).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(7).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(2).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[3].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(3), parameter: parameter_names[4].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(4), parameter: parameter_names[5].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(5), parameter: parameter_names[6].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(6), parameter: parameter_names[7].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(7) },
            PtxInstruction::MoveNtidX { destination: b32(8) },
            PtxInstruction::MoveTidX { destination: b32(9) },
            PtxInstruction::MultiplyAddLoS32 { destination: b32(10), left: b32(7), right: b32(8), addend: b32(9) },
            PtxInstruction::MulLoU32 { destination: b32(11), left: b32(1), right: U32Value::Reg(b32(2)) },
            PtxInstruction::MulLoU32 { destination: b32(11), left: b32(11), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MulLoU32 { destination: b32(11), left: b32(11), right: U32Value::Reg(b32(6)) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(10), right: U32Value::Reg(b32(11)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::RemU32 { destination: b32(12), left: b32(10), right: U32Value::Reg(b32(6)) },
            PtxInstruction::DivU32 { destination: b32(13), left: b32(10), right: U32Value::Reg(b32(6)) },
            PtxInstruction::RemU32 { destination: b32(14), left: b32(13), right: U32Value::Reg(b32(5)) },
            PtxInstruction::DivU32 { destination: b32(15), left: b32(13), right: U32Value::Reg(b32(5)) },
            PtxInstruction::RemU32 { destination: b32(16), left: b32(15), right: U32Value::Reg(b32(2)) },
            PtxInstruction::DivU32 { destination: b32(17), left: b32(15), right: U32Value::Reg(b32(2)) },
            PtxInstruction::MulLoU32 { destination: b32(18), left: b32(14), right: U32Value::Reg(b32(3)) },
            PtxInstruction::DivU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MulLoU32 { destination: b32(19), left: b32(12), right: U32Value::Reg(b32(4)) },
            PtxInstruction::DivU32 { destination: b32(19), left: b32(19), right: U32Value::Reg(b32(6)) },
            PtxInstruction::MulLoU32 { destination: b32(20), left: b32(17), right: U32Value::Reg(b32(2)) },
            PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(16)) },
            PtxInstruction::MulLoU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(3)) },
            PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(18)) },
            PtxInstruction::MulLoU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(4)) },
            PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(20), right: 4 },
            PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(10), right: 4 },
            PtxInstruction::AddS64 { destination: b64(5), left: b64(1), right: b64(3) },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(2), right: b64(4) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(5) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(1) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
