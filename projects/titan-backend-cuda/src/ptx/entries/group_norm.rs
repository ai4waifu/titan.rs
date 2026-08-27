use std::num::NonZeroU8;

use super::{
    super::ast::{
        Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction, RegisterClass,
        RegisterDeclaration, U32Value,
    },
    regs::{b32, b64, f32, predicate},
};

pub(super) fn group_norm_f32(name: Identifier) -> Entry {
    let parameter_names: [Identifier; 12] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
    let parameters = parameter_names
        .iter()
        .enumerate()
        .map(|(index, parameter)| Parameter {
            name: parameter.clone(),
            kind: if index < 4 {
                ParameterKind::GlobalF32Pointer
            }
            else if matches!(index, 9) {
                ParameterKind::F32
            }
            else {
                ParameterKind::U32
            },
        })
        .collect();

    let done = Label(name.suffix("_done"));
    let mean_loop = Label(name.suffix("_mean_loop"));
    let mean_done = Label(name.suffix("_mean_done"));
    let var_loop = Label(name.suffix("_var_loop"));
    let var_done = Label(name.suffix("_var_done"));
    let store_loop = Label(name.suffix("_store_loop"));
    let no_gamma = Label(name.suffix("_no_gamma"));
    let no_beta = Label(name.suffix("_no_beta"));
    Entry {
        name,
        parameters,
        registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(4).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(22).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(10).unwrap() },
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
            PtxInstruction::LoadParameterF32 { destination: f32(6), parameter: parameter_names[9].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(14), parameter: parameter_names[10].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(15), parameter: parameter_names[11].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(6) },
            PtxInstruction::MoveNtidX { destination: b32(7) },
            PtxInstruction::MoveTidX { destination: b32(8) },
            PtxInstruction::MadLoU32 { destination: b32(9), left: b32(6), right: b32(7), addend: b32(8) },
            PtxInstruction::MulLoU32 { destination: b32(10), left: b32(1), right: U32Value::Reg(b32(5)) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(9), right: U32Value::Reg(b32(10)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::DivU32 { destination: b32(11), left: b32(9), right: U32Value::Reg(b32(5)) },
            PtxInstruction::RemU32 { destination: b32(12), left: b32(9), right: U32Value::Reg(b32(5)) },
            PtxInstruction::DivU32 { destination: b32(13), left: b32(2), right: U32Value::Reg(b32(5)) },
            PtxInstruction::MulLoU32 { destination: b32(16), left: b32(3), right: U32Value::Reg(b32(4)) },
            PtxInstruction::MulLoU32 { destination: b32(17), left: b32(13), right: U32Value::Reg(b32(16)) },
            PtxInstruction::MulLoU32 { destination: b32(18), left: b32(9), right: U32Value::Reg(b32(17)) },
            PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(mean_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(19), right: U32Value::Reg(b32(17)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: mean_done.clone() },
            PtxInstruction::AddU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(20), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
            PtxInstruction::AddRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: mean_loop.clone() },
            PtxInstruction::DefineLabel(mean_done.clone()),
            PtxInstruction::CvtRnF32U32 { destination: f32(7), source: b32(17) },
            PtxInstruction::DivRnF32 { destination: f32(1), left: F32Value::Reg(f32(1)), right: F32Value::Reg(f32(7)) },
            PtxInstruction::MoveF32Imm { destination: f32(3), bits: 0x00000000 },
            PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(var_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(19), right: U32Value::Reg(b32(17)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: var_done.clone() },
            PtxInstruction::AddU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(20), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
            PtxInstruction::SubRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
            PtxInstruction::MulRnF32 { destination: f32(2), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: var_loop.clone() },
            PtxInstruction::DefineLabel(var_done.clone()),
            PtxInstruction::DivRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(7)) },
            PtxInstruction::AddRnF32 { destination: f32(3), left: F32Value::Reg(f32(3)), right: F32Value::Reg(f32(6)) },
            PtxInstruction::RsqrtApproxF32 { destination: f32(4), source: F32Value::Reg(f32(3)) },
            PtxInstruction::MoveU32 { destination: b32(19), value: U32Value::Imm(0) },
            PtxInstruction::DefineLabel(store_loop.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(19), right: U32Value::Reg(b32(17)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: done.clone() },
            PtxInstruction::AddU32 { destination: b32(20), left: b32(18), right: U32Value::Reg(b32(19)) },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(20), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(5) },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(4), right: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
            PtxInstruction::SubRnF32 { destination: f32(5), left: F32Value::Reg(f32(2)), right: F32Value::Reg(f32(1)) },
            PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(4)) },
            PtxInstruction::DivU32 { destination: b32(21), left: b32(19), right: U32Value::Reg(b32(16)) },
            PtxInstruction::MadLoU32 { destination: b32(21), left: b32(12), right: b32(13), addend: b32(21) },
            PtxInstruction::SetPredicateEqU32 { destination: predicate(3), left: b32(14), right: U32Value::Imm(0) },
            PtxInstruction::BranchIf { predicate: predicate(3), target: no_gamma.clone() },
            PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(21), right: 4 },
            PtxInstruction::AddS64 { destination: b64(9), left: b64(2), right: b64(8) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(9) },
            PtxInstruction::MulRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::DefineLabel(no_gamma.clone()),
            PtxInstruction::SetPredicateEqU32 { destination: predicate(3), left: b32(15), right: U32Value::Imm(0) },
            PtxInstruction::BranchIf { predicate: predicate(3), target: no_beta.clone() },
            PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(21), right: 4 },
            PtxInstruction::AddS64 { destination: b64(9), left: b64(3), right: b64(8) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(9) },
            PtxInstruction::AddRnF32 { destination: f32(5), left: F32Value::Reg(f32(5)), right: F32Value::Reg(f32(2)) },
            PtxInstruction::DefineLabel(no_beta.clone()),
            PtxInstruction::StoreGlobalF32 { pointer: b64(7), value: f32(5) },
            PtxInstruction::AddU32 { destination: b32(19), left: b32(19), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: store_loop.clone() },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
    }
}
