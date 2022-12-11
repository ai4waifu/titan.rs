use std::num::NonZeroU8;

use super::super::ast::{
    Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn scaled_dot_product_attention_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 9] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
        let parameters = parameter_names
            .iter()
            .enumerate()
            .map(|(index, parameter)| Parameter {
                name: parameter.clone(),
                kind: if index < 4 { ParameterKind::GlobalF32Pointer } else { ParameterKind::U32 },
            })
            .collect();

    let predicate = |index| Register::new(RegisterClass::Predicate, index);
    let b32 = |index| Register::new(RegisterClass::B32, index);
    let b64 = |index| Register::new(RegisterClass::B64, index);
    let f32 = |index| Register::new(RegisterClass::F32, index);
        let done = Label(name.suffix("_done"));
        let max_loop = Label(name.suffix("_max_loop"));
        let max_inner_loop = Label(name.suffix("_max_inner_loop"));
        let max_inner_done = Label(name.suffix("_max_inner_done"));
        let max_next = Label(name.suffix("_max_next"));
        let max_done = Label(name.suffix("_max_done"));
        let sum_loop = Label(name.suffix("_sum_loop"));
        let sum_inner_loop = Label(name.suffix("_sum_inner_loop"));
        let sum_inner_done = Label(name.suffix("_sum_inner_done"));
        let sum_next = Label(name.suffix("_sum_next"));
        let sum_done = Label(name.suffix("_sum_done"));
        let value_loop = Label(name.suffix("_value_loop"));
        let value_inner_loop = Label(name.suffix("_value_inner_loop"));
        let value_inner_done = Label(name.suffix("_value_inner_done"));
        let value_next = Label(name.suffix("_value_next"));
        let value_done = Label(name.suffix("_value_done"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(2).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(23).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(9).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(8).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(4), parameter: parameter_names[3].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[4].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[5].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(3), parameter: parameter_names[6].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(4), parameter: parameter_names[7].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(5), parameter: parameter_names[8].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(6) },
            PtxInstruction::MoveNtidX { destination: b32(7) },
            PtxInstruction::MoveTidX { destination: b32(8) },
            PtxInstruction::MadLoU32 { destination: b32(9), left: b32(6), right: b32(7), addend: b32(8) },
            PtxInstruction::MulLoU32 { destination: b32(10), left: b32(1), right: U32Value::Reg(b32(2)) },
            PtxInstruction::MulLoU32 { destination: b32(10), left: b32(10), right: U32Value::Reg(b32(3)) },
            PtxInstruction::MulLoU32 { destination: b32(10), left: b32(10), right: U32Value::Reg(b32(5)) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(9), right: U32Value::Reg(b32(10)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::DivU32 { destination: b32(11), left: b32(9), right: U32Value::Reg(b32(5)) },
            PtxInstruction::RemU32 { destination: b32(12), left: b32(9), right: U32Value::Reg(b32(5)) },
            PtxInstruction::DivU32 { destination: b32(13), left: b32(11), right: U32Value::Reg(b32(3)) },
            PtxInstruction::RemU32 { destination: b32(14), left: b32(11), right: U32Value::Reg(b32(3)) },
            PtxInstruction::DivU32 { destination: b32(15), left: b32(13), right: U32Value::Reg(b32(2)) },
            PtxInstruction::RemU32 { destination: b32(16), left: b32(13), right: U32Value::Reg(b32(2)) },
            PtxInstruction::MadLoU32 { destination: b32(17), left: b32(15), right: b32(2), addend: b32(16) },
            PtxInstruction::MadLoU32 { destination: b32(17), left: b32(17), right: b32(3), addend: b32(14) },
            PtxInstruction::MulLoU32 { destination: b32(17), left: b32(17), right: U32Value::Reg(b32(5)) },
            PtxInstruction::CvtRnF32U32 { destination: f32(7), source: b32(5) },
            PtxInstruction::SqrtRnF32 { destination: f32(7), source: F32Value::Reg(f32(7)) },
            PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0xFF800000 },
            PtxInstruction::MoveU32 { destination: b32(20), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(max_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(20), right: U32Value::Reg(b32(4)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: max_done.clone() },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(15), right: b32(2), addend: b32(16) },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(4), addend: b32(20) },
            PtxInstruction::MulLoU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MoveF32Imm { destination: f32(2), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(max_inner_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(19), right: U32Value::Reg(b32(5)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: max_inner_done.clone() },
            PtxInstruction::AddU32 { destination: b32(21), left: b32(17), right: U32Value::Reg(b32(19)) },
            PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(21), right: 4 },
            PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(22), right: 4 },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
            PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
            PtxInstruction::LoadGlobalF32 { destination: f32(5), pointer: b64(7) },
            PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(8) },
            PtxInstruction::FmaRnF32 { destination: f32(2), a: F32Value::Reg(f32(5)), b: F32Value::Reg(f32(6)), c: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: max_inner_loop.clone() },
            PtxInstruction::DefineLabel(max_inner_done.clone()),
            PtxInstruction::DivRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(7)) },
            PtxInstruction::MaxF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::DefineLabel(max_next.clone()),
            PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: max_loop.clone() },
            PtxInstruction::DefineLabel(max_done.clone()),
            PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(20), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(sum_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(20), right: U32Value::Reg(b32(4)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: sum_done.clone() },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(15), right: b32(2), addend: b32(16) },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(4), addend: b32(20) },
            PtxInstruction::MulLoU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MoveF32Imm { destination: f32(2), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(sum_inner_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(19), right: U32Value::Reg(b32(5)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: sum_inner_done.clone() },
            PtxInstruction::AddU32 { destination: b32(21), left: b32(17), right: U32Value::Reg(b32(19)) },
            PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(21), right: 4 },
            PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(22), right: 4 },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
            PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
            PtxInstruction::LoadGlobalF32 { destination: f32(5), pointer: b64(7) },
            PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(8) },
            PtxInstruction::FmaRnF32 { destination: f32(2), a: F32Value::Reg(f32(5)), b: F32Value::Reg(f32(6)), c: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: sum_inner_loop.clone() },
            PtxInstruction::DefineLabel(sum_inner_done.clone()),
            PtxInstruction::DivRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(7)) },
            PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
            PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x3FB8AA3B) },
            PtxInstruction::Ex2ApproxF32 { destination: f32(2), source: F32Value::Reg(f32(2)) },
            PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::DefineLabel(sum_next.clone()),
            PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: sum_loop.clone() },
            PtxInstruction::DefineLabel(sum_done.clone()),
            PtxInstruction::MoveF32Imm { destination: f32(4), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(20), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(value_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(20), right: U32Value::Reg(b32(4)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: value_done.clone() },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(15), right: b32(2), addend: b32(16) },
            PtxInstruction::MadLoU32 { destination: b32(18), left: b32(18), right: b32(4), addend: b32(20) },
            PtxInstruction::MulLoU32 { destination: b32(18), left: b32(18), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MoveF32Imm { destination: f32(2), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(value_inner_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(19), right: U32Value::Reg(b32(5)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: value_inner_done.clone() },
            PtxInstruction::AddU32 { destination: b32(21), left: b32(17), right: U32Value::Reg(b32(19)) },
            PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(21), right: 4 },
            PtxInstruction::MultiplyWideU32 { destination: b64(6), left: b32(22), right: 4 },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(1), right: b64(5) },
            PtxInstruction::AddS64 { destination: b64(8), left: b64(2), right: b64(6) },
            PtxInstruction::LoadGlobalF32 { destination: f32(5), pointer: b64(7) },
            PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(8) },
            PtxInstruction::FmaRnF32 { destination: f32(2), a: F32Value::Reg(f32(5)), b: F32Value::Reg(f32(6)), c: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: value_inner_loop.clone() },
            PtxInstruction::DefineLabel(value_inner_done.clone()),
            PtxInstruction::DivRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(7)) },
            PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
            PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::ImmBits(0x3FB8AA3B) },
            PtxInstruction::Ex2ApproxF32 { destination: f32(2), source: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(22), left: b32(18), right: U32Value::Reg(b32(12)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(22), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(3), right: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(6), pointer: b64(6) },
            PtxInstruction::FmaRnF32 { destination: f32(4), a: F32Value::Reg(f32(2)), b: F32Value::Reg(f32(6)), c: F32Value::Reg(f32(4)) },
            PtxInstruction::DefineLabel(value_next.clone()),
            PtxInstruction::AddU32 { destination: b32(20), left: b32(20), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: value_loop.clone() },
            PtxInstruction::DefineLabel(value_done.clone()),
            PtxInstruction::DivRnF32 { destination: f32(4), left: F32Value::Reg(f32(4)), right: F32Value::Reg(f32(3)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(9), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(4), right: b64(5) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(6), value: f32(4) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
