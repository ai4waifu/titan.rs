use std::num::NonZeroU8;

use super::super::ast::{
    Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn gelu_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 3] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
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
        let negative = Label(name.suffix("_negative"));
        let signed_done = Label(name.suffix("_signed_done"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(6).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(6).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(10).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[2].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(2) },
            PtxInstruction::MoveNtidX { destination: b32(3) },
            PtxInstruction::MoveTidX { destination: b32(4) },
            PtxInstruction::MultiplyAddLoS32 { destination: b32(5), left: b32(2), right: b32(3), addend: b32(4) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(5), right: U32Value::Reg(b32(1)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::MultiplyWideU32 { destination: b64(3), left: b32(5), right: 4 },
            PtxInstruction::AddS64 { destination: b64(4), left: b64(1), right: b64(3) },
            PtxInstruction::AddS64 { destination: b64(5), left: b64(2), right: b64(3) },
            PtxInstruction::LoadGlobalF32 { destination: f32(1), pointer: b64(4) },
            PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(1)), right: F32Value::ImmBits(0x3F3504F3) },
            PtxInstruction::MoveF32 { destination: f32(3), source: f32(2) },
            PtxInstruction::SetPredicateLtF32 { destination: predicate(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x00000000) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: negative.clone() },
            PtxInstruction::MoveF32Imm { destination: f32(4), bits: 0x3F800000 },
            PtxInstruction::Branch { target: signed_done.clone() },
            PtxInstruction::DefineLabel(negative.clone()),
            PtxInstruction::MoveF32Imm { destination: f32(4), bits: 0xBF800000 },
            PtxInstruction::SubRnF32 { destination: f32(3), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(3)) },
            PtxInstruction::DefineLabel(signed_done.clone()),
            PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(3)), right: F32Value::ImmBits(0x3EA7BA05) },
            PtxInstruction::AddRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::ImmBits(0x3F800000) },
            PtxInstruction::DivRnF32 { destination: f32(5), left: F32Value::ImmBits(0x3F800000), right: F32Value::Reg(f32(5)) },
            PtxInstruction::MoveF32Imm { destination: f32(6), bits: 0x3F87DC22 },
            PtxInstruction::FmaRnF32 { destination: f32(6), a: F32Value::Reg(f32(6)), b: F32Value::Reg(f32(5)), c: F32Value::ImmBits(0xBFBA00E3) },
            PtxInstruction::FmaRnF32 { destination: f32(6), a: F32Value::Reg(f32(6)), b: F32Value::Reg(f32(5)), c: F32Value::ImmBits(0x3FB5F0E3) },
            PtxInstruction::FmaRnF32 { destination: f32(6), a: F32Value::Reg(f32(6)), b: F32Value::Reg(f32(5)), c: F32Value::ImmBits(0xBE91A98E) },
            PtxInstruction::FmaRnF32 { destination: f32(6), a: F32Value::Reg(f32(6)), b: F32Value::Reg(f32(5)), c: F32Value::ImmBits(0x3E827906) },
            PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(5)) },
            PtxInstruction::MulRnF32 { destination: f32(7), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(3)) },
            PtxInstruction::SubRnF32 { destination: f32(7), left: F32Value::ImmBits(0x00000000), right: F32Value::Reg(f32(7)) },
            PtxInstruction::MulRnF32 { destination: f32(7), left: F32Value::Reg(f32(7)), right: F32Value::ImmBits(0x3FB8AA3B) },
            PtxInstruction::Ex2ApproxF32 { destination: f32(7), source: F32Value::Reg(f32(7)) },
            PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(7)) },
            PtxInstruction::SubRnF32 { destination: f32(6), left: F32Value::ImmBits(0x3F800000), right: F32Value::Reg(f32(6)) },
            PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(4)) },
            PtxInstruction::AddRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::ImmBits(0x3F800000) },
            PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::Reg(f32(1)) },
            PtxInstruction::MulRnF32 { destination: f32(6), left: F32Value::Reg(f32(6)), right: F32Value::ImmBits(0x3F000000) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(5), value: f32(6) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
