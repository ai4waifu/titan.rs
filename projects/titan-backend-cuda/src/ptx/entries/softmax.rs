use std::num::NonZeroU8;

use super::super::ast::{
    Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn softmax_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 4] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
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
        let max_loop = Label(name.suffix("_max_loop"));
        let max_done = Label(name.suffix("_max_done"));
        let sum_loop = Label(name.suffix("_sum_loop"));
        let sum_done = Label(name.suffix("_sum_done"));
        let normalize_loop = Label(name.suffix("_normalize_loop"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(10).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(5).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[3].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(3) },
            PtxInstruction::MoveNtidX { destination: b32(4) },
            PtxInstruction::MoveTidX { destination: b32(5) },
            PtxInstruction::MultiplyAddLoS32 { destination: b32(6), left: b32(3), right: b32(4), addend: b32(5) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(6), right: U32Value::Reg(b32(1)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::MulLoU32 { destination: b32(7), left: b32(6), right: U32Value::Reg(b32(2)) },
            PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
            PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0xFF7FFFFF },
            PtxInstruction::DefineLabel(max_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: max_done.clone() },
            PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
            PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(4) },
            PtxInstruction::MaxF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: max_loop.clone() },
            PtxInstruction::DefineLabel(max_done.clone()),
            PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
            PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
            PtxInstruction::DefineLabel(sum_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: sum_done.clone() },
            PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
            PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
            PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(4) },
            PtxInstruction::SubRnF32 { destination: f32(4), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
            PtxInstruction::MulRnF32 { destination: f32(4), left: F32Value::Reg(f32(4)), right: F32Value::ImmBits(0x3FB8AA3B) },
            PtxInstruction::Ex2ApproxF32 { destination: f32(4), source: F32Value::Reg(f32(4)) },
            PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(4)) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(4) },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: sum_loop.clone() },
            PtxInstruction::DefineLabel(sum_done.clone()),
            PtxInstruction::MoveU32 { destination: b32(8), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(normalize_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(8), right: U32Value::Reg(b32(2)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: done.clone() },
            PtxInstruction::AddU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(8)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(9), right: 4 },
            PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
            PtxInstruction::LoadGlobalF32 { destination: f32(4), pointer: b64(5) },
            PtxInstruction::DivRnF32 { destination: f32(4), left: F32Value::Reg(f32(4)), right: F32Value::Reg(f32(3)) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(4) },
            PtxInstruction::AddU32 { destination: b32(8), left: b32(8), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: normalize_loop.clone() },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
