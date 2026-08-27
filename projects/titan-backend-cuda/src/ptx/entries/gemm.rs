use std::num::NonZeroU8;

use super::super::ast::{
    Entry, F32Value, Identifier, Label, Parameter, ParameterIndex, ParameterKind, PtxInstruction,
    Register, RegisterClass, RegisterDeclaration, U32Value,
};

pub(super) fn gemm_f32(name: Identifier) -> Entry {
        let parameter_names: [Identifier; 6] = std::array::from_fn(|index| name.parameter(ParameterIndex(index as u8)));
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
        let done_store = Label(name.suffix("_done_store"));
        let loop_label = Label(name.suffix("_k_loop"));
        Entry {
            name,
            parameters,
            registers: vec![
            RegisterDeclaration { class: RegisterClass::Predicate, count: NonZeroU8::new(3).unwrap() },
            RegisterDeclaration { class: RegisterClass::B32, count: NonZeroU8::new(14).unwrap() },
            RegisterDeclaration { class: RegisterClass::B64, count: NonZeroU8::new(10).unwrap() },
            RegisterDeclaration { class: RegisterClass::F32, count: NonZeroU8::new(4).unwrap() },
        ],
            instructions: vec![
            PtxInstruction::LoadParameterU64 { destination: b64(1), parameter: parameter_names[0].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(2), parameter: parameter_names[1].clone() },
            PtxInstruction::LoadParameterU64 { destination: b64(3), parameter: parameter_names[2].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(1), parameter: parameter_names[3].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(2), parameter: parameter_names[4].clone() },
            PtxInstruction::LoadParameterU32 { destination: b32(3), parameter: parameter_names[5].clone() },
            PtxInstruction::MoveCtaIdX { destination: b32(4) },
            PtxInstruction::MoveNtidX { destination: b32(5) },
            PtxInstruction::MoveTidX { destination: b32(6) },
            PtxInstruction::MadLoU32 { destination: b32(7), left: b32(4), right: b32(5), addend: b32(6) },
            PtxInstruction::MulLoU32 { destination: b32(8), left: b32(1), right: U32Value::Reg(b32(2)) },
            PtxInstruction::SetPredicateGeU32 { destination: predicate(1), left: b32(7), right: U32Value::Reg(b32(8)) },
            PtxInstruction::BranchIf { predicate: predicate(1), target: done.clone() },
            PtxInstruction::DivU32 { destination: b32(9), left: b32(7), right: U32Value::Reg(b32(2)) },
            PtxInstruction::RemU32 { destination: b32(10), left: b32(7), right: U32Value::Reg(b32(2)) },
            PtxInstruction::MoveU32 { destination: b32(11), value: U32Value::Imm(0) },
            PtxInstruction::MoveF32Imm { destination: f32(1), bits: 0x00000000 },
            PtxInstruction::DefineLabel(loop_label.clone()),
            PtxInstruction::SetPredicateGeU32 { destination: predicate(2), left: b32(11), right: U32Value::Reg(b32(3)) },
            PtxInstruction::BranchIf { predicate: predicate(2), target: done_store.clone() },
            PtxInstruction::MadLoU32 { destination: b32(12), left: b32(9), right: b32(3), addend: b32(11) },
            PtxInstruction::MadLoU32 { destination: b32(13), left: b32(11), right: b32(2), addend: b32(10) },
            PtxInstruction::MultiplyWideU32 { destination: b64(4), left: b32(12), right: 4 },
            PtxInstruction::MultiplyWideU32 { destination: b64(5), left: b32(13), right: 4 },
            PtxInstruction::AddS64 { destination: b64(6), left: b64(1), right: b64(4) },
            PtxInstruction::AddS64 { destination: b64(7), left: b64(2), right: b64(5) },
            PtxInstruction::LoadGlobalF32 { destination: f32(2), pointer: b64(6) },
            PtxInstruction::LoadGlobalF32 { destination: f32(3), pointer: b64(7) },
            PtxInstruction::FmaRnF32 { destination: f32(1), a: F32Value::Reg(f32(2)), b: F32Value::Reg(f32(3)), c: F32Value::Reg(f32(1)) },
            PtxInstruction::AddU32 { destination: b32(11), left: b32(11), right: U32Value::Imm(1) },
            PtxInstruction::Branch { target: loop_label.clone() },
            PtxInstruction::DefineLabel(done_store.clone()),
            PtxInstruction::MultiplyWideU32 { destination: b64(8), left: b32(7), right: 4 },
            PtxInstruction::AddS64 { destination: b64(9), left: b64(3), right: b64(8) },
            PtxInstruction::StoreGlobalF32 { pointer: b64(9), value: f32(1) },
            PtxInstruction::DefineLabel(done.clone()),
            PtxInstruction::Return,
        ],
        }
}
